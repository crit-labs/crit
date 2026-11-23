"""row-ops against the operations' FP64 references (crit_kernels.contracts): every function within its tolerance, a
row's bits the same alone and beside other rows and at any row count, strided rows as their copies, a CUDA graph
replaying to the eager bits, and a build with other knobs within the same bounds. The kernel is built for this GPU by
crit_kernels.build (default knobs, or a build holding other knobs, selected)."""
import pytest
import torch

from crit_kernels import contracts
from crit_kernels.build import load

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="row-ops runs on CUDA")
BF16 = torch.bfloat16


@pytest.fixture(scope="module")
def ops():
    return load("row-ops")


def randn(seed, *shape, dtype=None, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(*shape, generator=g) * scale).cuda()
    return x if dtype is None else x.to(dtype)


def ok(operation, got, want, arithmetic=None):
    problem = contracts.check(operation, got.float(), want.float(), arithmetic=arithmetic)
    assert problem is None, problem


def alone(f, *rows):
    """f over all the rows, then over rows 3..5 alone and over the rows four times: those rows' bits alike."""
    whole = f(*rows)
    assert torch.equal(whole[3:5], f(*(r[3:5].contiguous() for r in rows)))
    assert torch.equal(whole, f(*(torch.cat([r] * 4) for r in rows))[:rows[0].shape[0]])
    return whole


def test_head_clip(ops):
    x, scale, tau = randn(1, 6, 24, 128, scale=3.0), randn(2, 24, 128).abs() + 0.5, randn(3, 6).abs() + 1
    reference = contracts.reference("head-clip")
    ok("head-clip", ops.head_clip(x, 22.6, scale), reference(x, 22.6, scale), "fp32-output")
    ok("head-clip", ops.head_clip(x, 22.6, scale, tau, round_bf16=True), reference(x, 22.6, scale, tau, True),
       "fp32-output")
    ok("head-clip", ops.head_clip(x.to(BF16), 1.0, out_dtype=BF16), reference(x.to(BF16), 1.0))
    alone(lambda r: ops.head_clip(r, 22.6, scale), x)
    wide = randn(4, 6, 24, 3 * 128)[:, :, :128]           # tokens at a stride: the same bits as their copy
    assert torch.equal(ops.head_clip(wide, 2.0), ops.head_clip(wide.contiguous(), 2.0))


def test_small_product(ops):
    r, proj = randn(5, 5, 24, 16), randn(6, 16, 1024)
    ok("small-product", ops.small_product(r, proj), contracts.reference("small-product")(r, proj))
    alone(lambda x: ops.small_product(x, proj), r)


def test_sink_gate(ops):
    a, lse = randn(7, 6, 24, 128, dtype=BF16), randn(8, 6, 24)
    rows = randn(9, 6, 24 * 128 * 2 + 24, dtype=BF16)
    sink, gate = rows[:, -24:], rows[:, :24 * 128]           # strided views of one product's rows
    ok("sink-gate", ops.sink_gate(a, lse, sink, gate), contracts.reference("sink-gate")(a, lse, sink, gate))
    alone(ops.sink_gate, a, lse, sink.contiguous(), gate.contiguous())


def test_scaled_add(ops):
    y, x = randn(10, 6, 2304), randn(11, 6, 2304, dtype=BF16)
    want = ((y + x.float()) * 0.5).to(BF16)
    assert torch.equal(ops.scaled_add(y, x, 0.5, BF16), want)       # one FP32 add and product, one rounding
    assert torch.equal(ops.scaled_add(y, x, 1.0, torch.float32), y + x.float())
    ok("scaled-add", ops.scaled_add(y, x, 0.5), contracts.reference("scaled-add")(y, x, 0.5))


def test_summary_mix(ops):
    for n, q in ((0, 1), (3, 1), (5, 3), (2, 4)):
        blocks, h, query = randn(12 + n, 7, n, 2304, dtype=BF16), randn(13, 7, 2304), randn(14 + q, q, 2304)
        found = ops.summary_mix(blocks, h, query)
        ok("summary-mix", found, contracts.reference("summary-mix")(blocks, h, query))
        alone(lambda b, x: ops.summary_mix(b, x, query), blocks, h)


def test_pair_norm_and_gated_join(ops):
    x, m = randn(15, 6, 2304), randn(16, 6, 2304)
    ok("pair-norm", ops.pair_norm(x, m), contracts.reference("pair-norm")(x, m))
    alone(ops.pair_norm, x, m)
    fa = randn(17, 6, 2 * 2304)
    f, a = fa[:, :2304], fa[:, 2304:]                         # two halves of one product's rows
    ok("gated-join", ops.gated_join(x, m, f, a), contracts.reference("gated-join")(x, m, f, a))


def test_low_rank_gates(ops):
    t, g, h, r = 6, 3, 2304, 64
    x, m = randn(18, t, g, h), randn(19, t, g, h)
    down = randn(20, g, r, 2 * h, dtype=BF16, scale=0.02)
    up = randn(21, g, 2 * h, r, dtype=BF16, scale=0.1)
    found = ops.low_rank_gates(x, m, down, up)
    ok("gated-join", found, contracts.reference("gated-join")(x, m, down=down, up=up), "low-rank")
    shared = ops.low_rank_gates(x[:, :1].expand(t, g, h), m, down, up)   # one input for every gate
    assert torch.equal(shared[:, 0], found[:, 0])
    alone(lambda a, b: ops.low_rank_gates(a, b, down, up), x, m)


def test_span_attention(ops):
    rows, kvh, g, d, t = 3, 2, 32, 128, 1500
    q = randn(22, rows, kvh, g, d, dtype=BF16)
    k, v = randn(23, rows, kvh, t, d, dtype=BF16), randn(24, rows, kvh, t, d, dtype=BF16)
    first = torch.tensor([0, 300, 1000], device="cuda")
    last = torch.tensor([7, 1331, 1499], device="cuda")
    found = ops.span_attention(q, k, v, first, last)
    ok("span-attention", found, contracts.reference("span-attention")(q, k, v, first, last))
    assert torch.equal(found[1:2], ops.span_attention(q[1:2], k[1:2], v[1:2], first[1:2], last[1:2]))


def test_a_graph_replays_the_eager_bits(ops):
    x, scale = randn(25, 6, 24, 128), randn(26, 24, 128).abs() + 0.5
    eager = ops.head_clip(x, 2.0, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ops.head_clip(x, 2.0, scale)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_other_knobs_keep_the_bounds():
    other = load("row-ops", knobs={"summary-mix": {"CHUNK": 128, "WARPS": 8}, "span-attention": {"CHUNK": 256,
                                                                                                "BN": 32},
                                   "small-product": {"BLOCK": 128, "WARPS": 2}})
    blocks, h, query = randn(27, 5, 3, 2304, dtype=BF16), randn(28, 5, 2304), randn(29, 3, 2304)
    ok("summary-mix", other.summary_mix(blocks, h, query), contracts.reference("summary-mix")(blocks, h, query))
    r, proj = randn(30, 5, 24, 16), randn(31, 16, 1024)
    ok("small-product", other.small_product(r, proj), contracts.reference("small-product")(r, proj))
    q = randn(32, 2, 2, 16, 64, dtype=BF16)
    k, v = randn(33, 2, 2, 700, 64, dtype=BF16), randn(34, 2, 2, 700, 64, dtype=BF16)
    first, last = torch.tensor([0, 100], device="cuda"), torch.tensor([699, 400], device="cuda")
    ok("span-attention", other.span_attention(q, k, v, first, last),
       contracts.reference("span-attention")(q, k, v, first, last))
