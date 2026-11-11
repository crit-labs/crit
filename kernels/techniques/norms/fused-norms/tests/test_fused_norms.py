"""fused-norms against a float64 reference: every function within FP32 rounding (FP32 out) or one output rounding
(16-bit out), weights per stream or head and the stream mixes too; each row's bits the same alone and beside others,
at any row count; strided views as their copies; out= and in-place rope; a CUDA graph replaying to the eager bits; the
two-pass path for rows wider than one block; a build with other knobs within the same bounds, its rows' bits
independent of the row count and its elementwise kernels' bits those of the defaults. The kernel is built for this GPU
by crit_kernels.build (default knobs, or a build holding other knobs, selected)."""
import math

import pytest
import torch

from crit_kernels.build import load, select

fn = load("fused-norms")
dev = torch.device("cuda")
FP32 = dict(atol=2e-5, rtol=2e-5)
STEP = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11, torch.float32: 0.0}


def gen(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def normal(g, *shape, scale=1.0, dtype=torch.float32):
    return (torch.randn(*shape, generator=g) * scale).to(dev, dtype)


def inv_freq(rotary, theta):
    return (1.0 / theta ** (torch.arange(0, rotary, 2, dtype=torch.float64) / rotary)).float().to(dev)


def close(found, expected, dtype=torch.float32):
    found, expected = found.double(), expected.double()
    tol = FP32["atol"] + (FP32["rtol"] + STEP[dtype]) * expected.abs()
    if STEP[dtype]:
        tol = tol + STEP[dtype] * 0.01 * float(expected.abs().max())
    bad = (found - expected).abs() > tol
    assert not bad.any(), f"{int(bad.sum())} values off; worst {float((found - expected).abs().max())}"


# -- float64 references --------------------------------------------------------------------------------------------
def ref_rms(x, w, eps, offset=0.0):
    x = x.double()
    y = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * (offset + w.double())


def ref_rope(x, positions, freqs, rotary, style, scale=1.0):
    x = x.double()
    shape = x.shape
    x = x if x.dim() == 3 else x[:, None]
    half = rotary // 2
    if freqs.dim() == 1:
        angle = (positions.float()[:, None] * freqs[None, :]).double()   # the FP32 angle, as the contract names it
        cos, sin = angle.cos() * scale, angle.sin() * scale
    else:
        rows = freqs.double()[positions.clamp(0, freqs.shape[0] - 1)]
        cos, sin = rows[:, :half] * scale, rows[:, half:] * scale
    cos, sin = cos[:, None], sin[:, None]
    y = x.clone()
    if style == "half":
        a, b = x[..., :half], x[..., half:rotary]
        y[..., :half], y[..., half:rotary] = a * cos - b * sin, b * cos + a * sin
    else:
        a, b = x[..., 0:rotary:2], x[..., 1:rotary:2]
        y[..., 0:rotary:2], y[..., 1:rotary:2] = a * cos - b * sin, b * cos + a * sin
    return y.reshape(shape)


def ref_act(name, g):
    g = g.double()
    if name == "silu":
        return g * torch.sigmoid(g)
    if name == "gelu":
        return 0.5 * g * (1 + torch.erf(g / math.sqrt(2)))
    if name == "gelu_tanh":
        return 0.5 * g * (1 + torch.tanh(math.sqrt(2 / math.pi) * (g + 0.044715 * g ** 3)))
    return torch.sigmoid(g)


DTYPES = [torch.bfloat16, torch.float16, torch.float32]


# -- against the reference -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [64, 520, 2560, 5120, 7168, 20000, 40000])
def test_rms_norm(dtype, width):
    g = gen(width)
    x, r, w = normal(g, 5, width, scale=2, dtype=dtype), normal(g, 5, width, scale=2), normal(g, width) * 0.3 + 1
    for out_dtype in (torch.float32, dtype):
        close(fn.rms_norm(x, w, 1e-6, out_dtype=out_dtype), ref_rms(x, w, 1e-6), out_dtype)
        close(fn.rms_norm(x, w.to(dtype), 1e-5, offset=1.0, out_dtype=out_dtype), ref_rms(x, w.to(dtype), 1e-5, 1.0),
              out_dtype)
    close(fn.rms_norm(x, None, 1e-6, out_dtype=torch.float32), ref_rms(x, None, 1e-6))
    residual = r.clone()
    y, kept = fn.rms_norm(x, w, 1e-6, residual=residual, out_dtype=torch.float32)
    assert kept is residual
    close(kept, r.double() + x.double())
    close(y, ref_rms(r.double() + x.double(), w, 1e-6))
    low = r.to(dtype, copy=True)
    sums = low.double() + x.double()
    y, kept = fn.rms_norm(x, w, 1e-6, residual=low)
    assert y.dtype == dtype
    close(kept, sums, dtype)
    close(y, ref_rms(kept, w, 1e-6), dtype)   # the sums as the residual keeps them are what is normalized


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads,dim,rotary", [(24, 256, 64), (56, 128, 128), (2, 256, 64), (3, 96, 32), (4, 64, 64)])
def test_heads_rope_gates_l2(dtype, heads, dim, rotary):
    g = gen(heads * dim)
    x, z = normal(g, 7, heads, dim, scale=2, dtype=dtype), normal(g, 7, heads, dim, scale=2, dtype=dtype)
    w = normal(g, dim) * 0.3 + 1
    positions = torch.randint(0, 1 << 18, (7,), generator=g).to(dev)
    close(fn.rms_norm_heads(x, w, 1e-6, offset=0.5, out_dtype=torch.float32), ref_rms(x, w, 1e-6, 0.5))
    for style in ("half", "interleaved"):
        freqs = inv_freq(rotary, 1e7)
        expected = ref_rope(x, positions, freqs, rotary, style, 1.5)
        for out_dtype in (torch.float32, dtype):
            close(fn.rope(x, positions, freqs, rotary, style, 1.5, out_dtype=out_dtype), expected, out_dtype)
        close(fn.rms_norm_rope(x, w, 1e-6, positions, freqs, 0.5, rotary, style, out_dtype=torch.float32),
              ref_rope(ref_rms(x, w, 1e-6, 0.5), positions, freqs, rotary, style))
        p = torch.arange(64, dtype=torch.float64)[:, None] * freqs.double().cpu()[None, :]
        table = torch.cat([p.cos(), p.sin()], -1).float().to(dev)
        small = positions % 64
        close(fn.rope(x, small, table, style=style, out_dtype=torch.float32), ref_rope(x, small, table, rotary, style))
    for activation in ("silu", "sigmoid"):
        close(fn.gated_rms_norm(x, z, w, 1e-6, activation=activation, out_dtype=torch.float32),
              ref_rms(x, w, 1e-6) * ref_act(activation, z))
    xd = x.double()
    close(fn.l2_norm(x, 1e-6, out_dtype=torch.float32), xd / torch.sqrt(xd.pow(2).sum(-1, keepdim=True) + 1e-6))


@pytest.mark.parametrize("dtype", DTYPES)
def test_products(dtype):
    g = gen(1)
    x = normal(g, 6, 2 * 1500, scale=3, dtype=dtype)
    gate, up = x[:, :1500], x[:, 1500:]
    for out_dtype in (torch.float32, dtype):
        close(fn.silu_mul(x, out_dtype=out_dtype), ref_act("silu", gate) * up.double(), out_dtype)
    close(fn.silu_mul(x, layout="interleaved", out_dtype=torch.float32), ref_act("silu", x[:, 0::2]) * x[:, 1::2].double())
    close(fn.silu_mul(gate, up, out_dtype=torch.float32), ref_act("silu", gate) * up.double())
    close(fn.gelu_mul(x, out_dtype=torch.float32), ref_act("gelu", gate) * up.double())
    close(fn.gelu_mul(x, layout="interleaved", approximate="tanh", out_dtype=torch.float32),
          ref_act("gelu_tanh", x[:, 0::2]) * x[:, 1::2].double())
    close(fn.sigmoid_mul(up, gate, out_dtype=torch.float32), up.double() * torch.sigmoid(gate.double()))
    big = torch.tensor([[-100.0, 100.0, -30.0, 0.0]], device=dev)
    assert torch.isfinite(fn.silu_mul(big, big)).all() and torch.isfinite(fn.gelu_mul(big, big, approximate="tanh")).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_weight_per_stream_and_the_stream_mixes(dtype):
    g = gen(7)
    streams = normal(g, 5, 4, 2560, scale=2, dtype=dtype)
    w = normal(g, 4, 2560) * 0.3
    close(fn.rms_norm(streams, w, 1e-6, offset=1.0, out_dtype=torch.float32), ref_rms(streams, w, 1e-6, 1.0))
    heads = normal(g, 3, 4, 64, scale=2, dtype=dtype)
    hw = normal(g, 4, 64) * 0.3 + 1
    positions = torch.randint(0, 4096, (3,), generator=g).to(dev)
    freqs = inv_freq(32, 1e4)
    close(fn.rms_norm_rope(heads, hw, 1e-6, positions, freqs, out_dtype=torch.float32),
          ref_rope(ref_rms(heads, hw, 1e-6), positions, freqs, 32, "half"))
    gates, per_stream = normal(g, 5, 4, 2560, scale=2, dtype=dtype), normal(g, 5, 4)
    sd = streams.double()
    for gate, activation, scale, factor in (
            (None, "none", 0.25, torch.ones_like(sd)), (gates, "sigmoid", 1.0, torch.sigmoid(gates.double())),
            (per_stream, "none", 1.0, per_stream.double()[:, :, None]),
            (per_stream, "silu", 0.5, ref_act("silu", per_stream)[:, :, None]), (gates, "none", 1.0, gates.double())):
        expected = scale * (factor * sd).sum(1)
        close(fn.stream_sum(streams, gate, activation, scale, out_dtype=torch.float32), expected)
    row = normal(g, 5, 2560, scale=2, dtype=dtype)
    for weights in (None, per_stream):
        kept = streams.float().clone()
        factor = 1.0 if weights is None else weights.double()[:, :, None]
        expected = kept.double() + factor * row.double()[:, None, :]
        assert fn.stream_add(kept, row, weights) is kept
        close(kept, expected)


# -- rows alone and beside others ----------------------------------------------------------------------------------
def calls(width, heads, dim, rotary, fn=fn):
    g = gen(width)
    n = 320
    x, r = normal(g, n, width, scale=2, dtype=torch.bfloat16), normal(g, n, width, scale=2)
    w = normal(g, width) * 0.3 + 1
    h, z = normal(g, n, heads, dim, scale=2, dtype=torch.bfloat16), normal(g, n, heads, dim, scale=2, dtype=torch.bfloat16)
    hw = normal(g, dim) * 0.3 + 1
    positions = torch.randint(0, 1 << 18, (n,), generator=g).to(dev)
    freqs = inv_freq(rotary, 1e7)
    gate_up = normal(g, n, 2 * width, scale=3, dtype=torch.bfloat16)
    streams, sw = normal(g, n, 4, width, scale=2), normal(g, 4, width) * 0.3 + 1
    gates, per_stream = normal(g, n, 4, width, scale=2, dtype=torch.bfloat16), normal(g, n, 4)

    def added(k):
        kept = streams[:k].clone()
        return fn.stream_add(kept, x[:k].clone(), per_stream[:k].clone())

    def added_normed(k):
        kept = streams[:k].clone()
        y, _ = fn.stream_add_norm(kept, x[:k].clone(), per_stream[:k].clone(), sw, 1e-6, 1.0, torch.bfloat16)
        return torch.cat([y.float(), kept], -1)

    kv = normal(g, n, max(1, heads // 4), dim, scale=2, dtype=torch.bfloat16)
    ba = normal(g, n, 96, scale=4, dtype=torch.bfloat16)
    rate, bias = normal(g, 48).exp(), normal(g, 48)

    return {
        "rms_norm": lambda k: fn.rms_norm(x[:k].clone(), w, 1e-6),
        "rms_norm residual": lambda k: torch.cat([t.float() for t in fn.rms_norm(
            x[:k].clone(), w, 1e-6, offset=1.0, residual=r[:k].clone(), out_dtype=torch.bfloat16)], -1),
        "rms_norm_heads": lambda k: fn.rms_norm_heads(h[:k].clone(), hw, 1e-6, offset=1.0),
        "rms_norm_rope": lambda k: fn.rms_norm_rope(h[:k].clone(), hw, 1e-6, positions[:k].clone(), freqs),
        "rope": lambda k: fn.rope(h[:k].clone(), positions[:k].clone(), freqs, style="interleaved"),
        "gated_rms_norm": lambda k: fn.gated_rms_norm(h[:k].clone(), z[:k].clone(), hw, 1e-6),
        "l2_norm": lambda k: fn.l2_norm(h[:k].clone()),
        "silu_mul": lambda k: fn.silu_mul(gate_up[:k].clone()),
        "gelu_mul": lambda k: fn.gelu_mul(gate_up[:k].clone(), layout="interleaved"),
        "sigmoid_mul": lambda k: fn.sigmoid_mul(x[:k].clone(), gate_up[:k, :width].clone()),
        "rms_norm per stream": lambda k: fn.rms_norm(streams[:k].clone(), sw, 1e-6, offset=1.0, out_dtype=torch.bfloat16),
        "stream_sum": lambda k: fn.stream_sum(streams[:k].clone(), gates[:k].clone(), "sigmoid", 0.25),
        "stream_add": added,
        "stream_add_norm": added_normed,
        "rms_norm_rope_pair": lambda k: torch.cat([t.flatten(1) for t in fn.rms_norm_rope_pair(
            h[:k].clone(), kv[:k].clone(), hw, hw, 1e-6, positions[:k].clone(), freqs, offset=1.0)], -1),
        "softplus_decay": lambda k: torch.cat(fn.softplus_decay(ba[:k, 48:].clone(), bias, rate,
                                                                ba[:k, :48].clone()), -1),
    }


@pytest.mark.parametrize("width,heads,dim,rotary", [(5120, 24, 256, 64), (7168, 56, 128, 128), (2560, 2, 256, 64),
                                                    (20000, 4, 64, 16)])
def test_a_row_has_its_bits_alone_and_beside_others(width, heads, dim, rotary):
    for name, call in calls(width, heads, dim, rotary).items():
        whole = call(320)
        for k in (1, 2, 3, 4, 5, 7, 8, 64):
            assert torch.equal(call(k), whole[:k]), f"{name}: a {k}-row call differs from the same rows of 320"


def test_unaligned_layouts_keep_a_rows_bits_too():
    # A width no multiple of 16 (BF16 in, FP32 out) and a head view whose token stride is odd: their launches must not
    # depend on the row count either (a 1-row call specializes like a 64-row call).
    g = gen(41)
    f32 = torch.float32
    x, r = normal(g, 64, 4100, scale=2, dtype=torch.bfloat16), normal(g, 64, 4100)
    w, z = normal(g, 4100) * 0.1 + 1, normal(g, 64, 4100, dtype=torch.bfloat16)
    big = normal(g, 64, 8 * 128 + 1, dtype=torch.bfloat16)
    q = big[:, :1024].view(64, 8, 128)
    hw = normal(g, 128) * 0.1 + 1
    positions = torch.randint(0, 4096, (64,), generator=g).to(dev)
    freqs = inv_freq(128, 1e4)
    calls = {"rms_norm": lambda k: fn.rms_norm(x[:k], w, 1e-6, out_dtype=f32),
             "l2_norm": lambda k: fn.l2_norm(x[:k], out_dtype=f32),
             "gated": lambda k: fn.gated_rms_norm(x[:k], z[:k], w, 1e-6, out_dtype=f32),
             "residual": lambda k: torch.cat(fn.rms_norm(x[:k], w, 1e-6, residual=r[:k].clone(), out_dtype=f32), -1),
             "heads view": lambda k: fn.rms_norm_heads(q[:k], hw, 1e-6, out_dtype=f32),
             "rope view": lambda k: fn.rms_norm_rope(q[:k], hw, 1e-6, positions[:k], freqs, out_dtype=f32),
             "l2 view": lambda k: fn.l2_norm(q[:k], out_dtype=f32)}
    for name, call in calls.items():
        whole = call(64)
        for k in (1, 2, 3, 8):
            assert torch.equal(call(k), whole[:k]), f"{name}: a {k}-row call differs from the same rows of 64"
    assert torch.equal(calls["heads view"](64), fn.rms_norm_heads(q.contiguous(), hw, 1e-6, out_dtype=f32))
    close(fn.rms_norm(x, w, 1e-6, out_dtype=f32), ref_rms(x, w, 1e-6))


@pytest.mark.parametrize("dtype,shape", [(torch.float32, (2048, 32, 256)), (torch.bfloat16, (2048, 4, 4096)),
                                         (torch.float32, (4096, 8, 1024))])
def test_rope_in_place_equals_rope_out_of_place(dtype, shape):
    g = gen(17)
    x = normal(g, *shape, dtype=dtype)
    positions = torch.randint(0, 100000, (shape[0],), generator=g).to(dev)
    freqs = inv_freq(shape[-1], 1e4)
    w = normal(g, shape[-1]) * 0.1 + 1
    for call in (lambda t, out=None: fn.rope(t, positions, freqs, out=out),
                 lambda t, out=None: fn.rms_norm_rope(t, w, 1e-6, positions, freqs, out=out)):
        expected = call(x)
        for _ in range(10):
            again = x.clone()
            call(again, out=again)
            assert torch.equal(again, expected)


def test_views_give_the_bits_of_their_copies():
    g = gen(3)
    qg = normal(g, 6, 4, 128, dtype=torch.bfloat16)
    q, gate = qg[..., :64], qg[..., 64:]
    w = normal(g, 64) * 0.2 + 1
    positions = torch.randint(0, 4096, (6,), generator=g).to(dev)
    freqs = inv_freq(16, 1e4)
    for call in (lambda t: fn.rms_norm_heads(t, w, 1e-6, offset=1.0),
                 lambda t: fn.rms_norm_rope(t, w, 1e-6, positions, freqs, offset=1.0),
                 lambda t: fn.rope(t, positions, freqs), lambda t: fn.l2_norm(t),
                 lambda t: fn.sigmoid_mul(t, gate), lambda t: fn.gated_rms_norm(t, gate, w, 1e-6)):
        assert torch.equal(call(q), call(q.contiguous()))
    fused = normal(g, 6, 1536, dtype=torch.bfloat16)
    wide = normal(g, 512) * 0.2 + 1
    assert torch.equal(fn.rms_norm(fused[:, 512:1024], wide, 1e-6), fn.rms_norm(fused[:, 512:1024].contiguous(), wide, 1e-6))
    assert torch.equal(fn.silu_mul(fused[:, :512], fused[:, 1024:]),
                       fn.silu_mul(fused[:, :512].contiguous(), fused[:, 1024:].contiguous()))
    split = normal(g, 6, 2, 192, dtype=torch.bfloat16)
    x4, z4 = split[..., :64].view(6, 2, 2, 32), split[..., 64:128].view(6, 2, 2, 32)
    hw = normal(g, 32) * 0.2 + 1
    assert torch.equal(fn.gated_rms_norm(x4, z4, hw, 1e-6), fn.gated_rms_norm(x4.contiguous(), z4.contiguous(), hw, 1e-6))
    odd = normal(g, 4, 6, 3, 32, dtype=torch.bfloat16).transpose(1, 2)   # leading dimensions no two levels cover
    assert torch.equal(fn.l2_norm(odd), fn.l2_norm(odd.contiguous()))


def test_out_in_place_and_dtypes():
    g = gen(5)
    x = normal(g, 3, 2, 64, dtype=torch.bfloat16)
    positions = torch.randint(0, 4096, (3,), generator=g).to(dev)
    freqs = inv_freq(64, 1e4)
    expected = fn.rope(x, positions, freqs)
    out = torch.empty_like(x)
    assert fn.rope(x, positions, freqs, out=out) is out and torch.equal(out, expected)
    inplace = x.clone()
    assert fn.rope(inplace, positions, freqs, out=inplace) is inplace and torch.equal(inplace, expected)
    w = normal(g, 64) + 1
    for out_dtype in DTYPES:
        assert fn.rms_norm(x, w, 1e-6, out_dtype=out_dtype).dtype == out_dtype
    assert tuple(fn.silu_mul(x.view(3, 128)).shape) == (3, 64)
    assert tuple(fn.silu_mul(x, layout="interleaved").shape) == (3, 2, 32)
    for bad in (lambda: fn.rms_norm(x, normal(g, 32), 1e-6), lambda: fn.silu_mul(x.view(3, 128)[:, :127]),
                lambda: fn.rope(x, positions[:2], freqs), lambda: fn.rope(x, positions, inv_freq(80, 1e4)),
                lambda: fn.gated_rms_norm(x, x[:2], w, 1e-6), lambda: fn.rms_norm(x.cpu(), None, 1e-6),
                lambda: fn.rms_norm(x.view(6, 64), w, 1e-6, out=torch.empty(64, 6, dtype=x.dtype, device=dev).t()),
                lambda: fn.silu_mul(x.view(3, 128), out=torch.empty(64, 3, dtype=x.dtype, device=dev).t()),
                lambda: fn.rms_norm_rope(normal(g, 4, 64), normal(g, 4, 64), 1e-6, positions[:3].repeat(2)[:4],
                                         freqs)):
        with pytest.raises(ValueError):
            bad()


def test_a_captured_graph_replays_the_eager_bits():
    g = gen(9)
    x, residual = normal(g, 3, 2560, dtype=torch.bfloat16), normal(g, 3, 2560)
    w = normal(g, 2560) + 1
    h, z = normal(g, 3, 4, 128, dtype=torch.bfloat16), normal(g, 3, 4, 128, dtype=torch.bfloat16)
    hw = normal(g, 128) + 1
    positions = torch.randint(0, 1 << 16, (3,), generator=g).to(dev)
    freqs = inv_freq(32, 1e6)
    gate_up = normal(g, 3, 5120, dtype=torch.bfloat16)

    def run():
        y, kept = fn.rms_norm(x, w, 1e-6, offset=1.0, residual=residual, out_dtype=torch.bfloat16)
        return [y, kept.clone(), fn.rms_norm_heads(h, hw, 1e-6), fn.rms_norm_rope(h, hw, 1e-6, positions, freqs),
                fn.rope(h, positions, freqs, style="interleaved"), fn.gated_rms_norm(h, z, hw, 1e-6), fn.l2_norm(h),
                fn.silu_mul(gate_up), fn.gelu_mul(gate_up, approximate="tanh"), fn.sigmoid_mul(x, gate_up[:, :2560])]

    start = residual.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    residual.copy_(start)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    g2 = gen(10)
    for t in (x, residual, w, h, z, hw, gate_up):
        t.copy_(normal(g2, *t.shape, dtype=t.dtype))
    positions.copy_(torch.randint(0, 1 << 16, (3,), generator=g2).to(dev))
    start = residual.clone()
    graph.replay()
    replayed = [t.clone() for t in captured]
    residual.copy_(start)
    for a, b in zip(replayed, run()):
        assert torch.equal(a, b)


# -- fused forms ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hq,hk,dim,rotary", [(24, 4, 256, 64), (32, 8, 128, 128), (56, 8, 128, 64), (3, 5, 96, 32),
                                              (16, 16, 64, 64)])
def test_the_query_key_pair_is_the_two_calls_in_one_launch(dtype, hq, hk, dim, rotary):
    # the pair gives each tensor the bits its own rms_norm_rope call gives: views of a gated qkv projection's output,
    # weights per head or shared, inverse frequencies or a table, both styles, FP32 or 16-bit out, in place
    g = gen(hq * 1000 + dim)
    for T in (1, 4, 9, 77):
        qkv = normal(g, T, hq * 2 * dim + 2 * hk * dim, scale=2, dtype=dtype)
        q = qkv[:, :hq * 2 * dim].view(T, hq, 2 * dim)[..., :dim]
        k = qkv[:, hq * 2 * dim:hq * 2 * dim + hk * dim].view(T, hk, dim)
        positions = torch.randint(0, 1 << 18, (T,), generator=g).to(dev)
        table = torch.cat([torch.cos(torch.arange(4096.0)[:, None] * inv_freq(rotary, 1e4).cpu()[None]),
                           torch.sin(torch.arange(4096.0)[:, None] * inv_freq(rotary, 1e4).cpu()[None])], -1).to(dev)
        for wq, wk in ((normal(g, dim) * 0.2, normal(g, dim) * 0.2), (normal(g, hq, dim), normal(g, hk, dim)),
                       (None, None)):
            for freqs, style, out_dtype in ((inv_freq(rotary, 1e7), "half", torch.bfloat16),
                                            (table, "interleaved", torch.float32), (inv_freq(rotary, 1e4), "half", None)):
                one = [fn.rms_norm_rope(t, w, 1e-6, positions, freqs, 1.0, rotary, style, 0.5, out_dtype)
                       for t, w in ((q, wq), (k, wk))]
                pair = fn.rms_norm_rope_pair(q, k, wq, wk, 1e-6, positions, freqs, 1.0, rotary, style, 0.5, out_dtype)
                for a, b in zip(pair, one):
                    assert torch.equal(a, b)
                if wq is None:
                    pair = fn.rope_pair(q, k, positions, freqs, rotary, style, 0.5, out_dtype)
                    one = [fn.rope(t, positions, freqs, rotary, style, 0.5, out_dtype) for t in (q, k)]
                    for a, b in zip(pair, one):
                        assert torch.equal(a, b)
        qi, ki = q.contiguous(), k.contiguous()
        expected = fn.rms_norm_rope_pair(qi, ki, wq, wk, 1e-6, positions, inv_freq(rotary, 1e4))
        found = fn.rms_norm_rope_pair(qi, ki, wq, wk, 1e-6, positions, inv_freq(rotary, 1e4), q_out=qi, k_out=ki)
        assert found[0] is qi and found[1] is ki
        assert torch.equal(qi, expected[0]) and torch.equal(ki, expected[1])
    pair = fn.rms_norm_rope_pair(q, k, None, None, 1e-6, positions, inv_freq(rotary, 1e4), 1.0, rotary, "half", 0.5)
    close(pair[0], ref_rope(ref_rms(q, None, 1e-6), positions, inv_freq(rotary, 1e4), rotary, "half", 0.5), dtype)
    close(pair[1], ref_rope(ref_rms(k, None, 1e-6), positions, inv_freq(rotary, 1e4), rotary, "half", 0.5), dtype)


def test_softplus_decay_is_torchs_softplus_and_sigmoid():
    g = gen(23)
    for T in (1, 4, 9, 512):
        ba = normal(g, T, 96, scale=4, dtype=torch.bfloat16)
        rate, bias = normal(g, 48).exp(), normal(g, 48)
        f32 = ba.float()
        decay, beta = fn.softplus_decay(ba[:, 48:], bias, rate, gate=ba[:, :48])
        assert torch.equal(decay, -rate * torch.nn.functional.softplus(f32[:, 48:] + bias))
        assert torch.equal(beta, torch.sigmoid(f32[:, :48]))
        wide = normal(g, T, 48, scale=30)
        # FP32 inside: against FP64 from the FP32 sum x + bias (values below FP32's normal range flush to zero)
        expected = -rate.double() * torch.nn.functional.softplus((wide + bias).double())
        found = fn.softplus_decay(wide, bias, rate)
        assert ((found.double() - expected).abs() <= 5e-7 * expected.abs() + 1e-37).all()
        assert fn.softplus_decay(wide, bias, rate, out_dtype=torch.bfloat16).dtype == torch.bfloat16
    nan = torch.full((2, 48), float("nan"), device=dev)
    assert torch.isnan(fn.softplus_decay(nan, bias, rate, nan)[0]).all()
    assert torch.isnan(fn.softplus_decay(nan, bias, rate, nan)[1]).all()
    for bad in (lambda: fn.softplus_decay(ba, bias, rate), lambda: fn.softplus_decay(ba[:, 48:], bias[:47], rate),
                lambda: fn.softplus_decay(ba[:, 48:], bias, rate, ba[:2, :48]),
                lambda: fn.softplus_decay(ba[:, 48:].cpu(), bias.cpu(), rate.cpu())):
        with pytest.raises(ValueError):
            bad()


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_row_added_into_the_streams_with_their_norms_is_the_two_calls(dtype):
    g = gen(29)
    for S, D in ((4, 2560), (4, 7168), (2, 5120), (3, 4100)):
        for T in (1, 4, 9, 100):
            streams, row = normal(g, T, S, D, scale=2, dtype=dtype), normal(g, T, D, scale=2, dtype=torch.bfloat16)
            for weights, weight in ((normal(g, T, S), normal(g, S, D) * 0.2), (None, normal(g, D) * 0.2),
                                    (normal(g, T, S), None)):
                kept = streams.clone()
                fn.stream_add(kept, row, weights)
                expected = fn.rms_norm(kept, weight, 1e-6, 1.0, out_dtype=torch.bfloat16)
                found = streams.clone()
                y, same = fn.stream_add_norm(found, row, weights, weight, 1e-6, 1.0, torch.bfloat16)
                assert same is found and torch.equal(found, kept) and torch.equal(y, expected)
    odd = normal(g, 4, 3 * 2560 + 1, dtype=dtype)[:, 1:].view(4, 3, 2560)        # unaligned streams: padded copies
    row, weights = normal(g, 4, 2560, dtype=dtype), normal(g, 4, 3)
    kept = odd.clone()
    fn.stream_add(kept, row, weights)
    y, _ = fn.stream_add_norm(odd, row, weights, None, 1e-6)
    assert torch.equal(odd, kept) and torch.equal(y, fn.rms_norm(kept, None, 1e-6))
    for bad in (lambda: fn.stream_add_norm(odd, row[:3], weights, None, 1e-6),
                lambda: fn.stream_add_norm(odd, row, weights[:, :2], None, 1e-6),
                lambda: fn.stream_add_norm(odd[:, 0], row, None, None, 1e-6)):
        with pytest.raises(ValueError):
            bad()


def test_a_captured_graph_replays_the_fused_forms_bits():
    g = gen(31)
    T = 5
    qkv = normal(g, T, 24 * 512 + 2048, dtype=torch.bfloat16)
    q, k = qkv[:, :12288].view(T, 24, 512)[..., :256], qkv[:, 12288:13312].view(T, 4, 256)
    wq, wk = normal(g, 256) * 0.2, normal(g, 256) * 0.2
    positions = torch.randint(0, 1 << 16, (T,), generator=g).to(dev)
    freqs = inv_freq(64, 1e7)
    ba, rate, bias = normal(g, T, 96, dtype=torch.bfloat16), normal(g, 48).exp(), normal(g, 48)
    streams, row, weights = normal(g, T, 4, 2560), normal(g, T, 2560, dtype=torch.bfloat16), normal(g, T, 4)
    sw = normal(g, 4, 2560) * 0.2

    def run():
        y, kept = fn.stream_add_norm(streams, row, weights, sw, 1e-6, 1.0, torch.bfloat16)
        return [*fn.rms_norm_rope_pair(q, k, wq, wk, 1e-6, positions, freqs, 1.0, 64),
                *fn.softplus_decay(ba[:, 48:], bias, rate, ba[:, :48]), y, kept.clone()]

    start = streams.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    streams.copy_(start)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    g2 = gen(32)
    for t in (qkv, ba, streams, row, weights):
        t.copy_(normal(g2, *t.shape, dtype=t.dtype))
    positions.copy_(torch.randint(0, 1 << 16, (T,), generator=g2).to(dev))
    start = streams.clone()
    graph.replay()
    replayed = [t.clone() for t in captured]
    streams.copy_(start)
    for a, b in zip(replayed, run()):
        assert torch.equal(a, b)


# -- knobs ---------------------------------------------------------------------------------------------------------
# A configuration away from every default: fewer values a program and more warps for the rows, more heads a program
# and fewer warps for rope, short chunks for the two passes, small and large blocks for the elementwise kernels.
OTHER = {"rows": {"ELEMENTS": 1024, "SPAN": 128}, "rope": {"ELEMENTS": 16384, "SPAN": 2048},
         "long-rows": {"CHUNK": 1024, "WARPS": 16, "PDL": 0}, "act-mul": {"BLOCK": 256, "WARPS": 8, "PDL": 2},
         "stream-sum": {"BLOCK": 4096, "WARPS": 1}, "stream-add": {"BLOCK": 512, "WARPS": 16},
         "softplus-decay": {"BLOCK": 32, "WARPS": 1}}
ELEMENTWISE = {"silu_mul", "gelu_mul", "sigmoid_mul", "stream_sum", "stream_add", "softplus_decay"}


def test_the_default_knobs_are_the_launches_from_before_the_knobs():
    # values a program 4,096 and a warp per 512 of them: (block, rows a program, warps) as the constants gave them
    for width, expected in ((64, (64, 64, 8)), (128, (128, 32, 8)), (2560, (4096, 1, 8)), (5120, (8192, 1, 16)),
                            (16384, (16384, 1, 16))):
        block, rows, span = fn._config(width)
        assert (block, rows, fn._warps(block, rows, span)) == expected
        assert fn._config(width, "rope") == (block, rows, span)
    # long rows: the row in one pass up to 32,768 values (16 warps), wider ones in two passes of 4,096 (8 warps)
    assert fn._knobs("long-rows", 1, D=20000) == {"CHUNK": 0, "WARPS": 0, "PDL": 3}
    assert fn._long_config(20000, torch.device(dev))[:3] == (32768, 16, True)
    assert fn._long_config(40000, torch.device(dev))[:3] == (4096, 8, False)
    assert fn._knobs("softplus-decay", 7, N=1500) == {"BLOCK": 1024, "WARPS": 4}
    # activation products: the knobs by row count (default_rows), programmatic dependent launch at every count
    for rows, expected in ((1, (512, 4, 1)), (8, (512, 4, 1)), (9, (1024, 4, 1)), (64, (1024, 4, 3)),
                           (512, (2048, 8, 2))):
        k = fn._knobs("act-mul", rows, N=17408)
        assert (k["BLOCK"], k["WARPS"], k["PDL"]) == expected, rows
    for structure in ("stream-sum", "stream-add"):
        assert fn._knobs(structure, 7, S=4, D=2560) == {"BLOCK": 1024, "WARPS": 4}


@pytest.mark.parametrize("width,heads,dim,rotary", [(5120, 24, 256, 64), (2560, 2, 256, 64), (20000, 4, 64, 16)])
def test_other_knobs_keep_the_bounds_and_a_rows_bits(width, heads, dim, rotary):
    # one build holding the defaults and OTHER, OTHER selected for every structure (as a sweep selects a configuration)
    other = load("fused-norms", configs={structure: [knobs] for structure, knobs in OTHER.items()})
    for structure in OTHER:
        select(other, structure, 1)
    assert other._knobs("rows", 1, D=width) == dict(OTHER["rows"])
    mine, base = calls(width, heads, dim, rotary, other), calls(width, heads, dim, rotary)
    for name, call in mine.items():
        whole = call(64)
        for k in (1, 3, 8):
            assert torch.equal(call(k), whole[:k]), f"{name}: a {k}-row call differs from the same rows of 64"
        if name in ELEMENTWISE:   # speed knobs only: the defaults' bits
            assert torch.equal(whole, base[name](64)), name
        else:                     # another order: within a rounding of the defaults (each within the reference)
            torch.testing.assert_close(whole.float(), base[name](64).float(), atol=1e-2, rtol=1e-2, msg=name)
    g = gen(width)
    x, w = normal(g, 9, width, scale=2, dtype=torch.bfloat16), normal(g, width) * 0.3 + 1
    close(other.rms_norm(x, w, 1e-6, out_dtype=torch.float32), ref_rms(x, w, 1e-6))
    q = normal(g, 9, heads, dim, scale=2, dtype=torch.bfloat16)
    positions = torch.randint(0, 1 << 18, (9,), generator=g).to(dev)
    freqs = inv_freq(rotary, 1e7)
    close(other.rms_norm_rope(q, w[:dim], 1e-6, positions, freqs, out_dtype=torch.float32),
          ref_rope(ref_rms(q, w[:dim], 1e-6), positions, freqs, rotary, "half"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
