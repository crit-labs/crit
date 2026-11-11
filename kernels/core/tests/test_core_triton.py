"""The kernel core's Triton blocks (python/crit_kcore) on the device against torch: every code of each format, the E4M3
encoder on a dense sweep of FP32 values, the fixed-order trees (their bits for every warp count), the partial sums,
the activations and the grouped tile order."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

import triton.language as tl   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from crit_kcore import epilogue, formats, load, reduce   # noqa: E402

CAP = torch.cuda.get_device_capability()
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1_ref(codes):
    c = codes.long().cpu()
    return (E2M1[c & 7] * (1 - 2 * ((c >> 3) & 1))).float()


@triton.jit
def _codes_kernel(b_ptr, e2_ptr, e4_ptr, e4hw_ptr, e8_ptr, lo_ptr, hi_ptr, CVT: tl.constexpr, HW_FP8: tl.constexpr):
    i = tl.arange(0, 256)
    b = tl.load(b_ptr + i)
    tl.store(e2_ptr + i, formats.e2m1(b & 15) * 16384.0)
    tl.store(e4_ptr + i, formats.e4m3_values(b))
    tl.store(e4hw_ptr + i, formats.e4m3_to(b, tl.float32, HW_FP8))
    tl.store(e8_ptr + i, formats.e8m0_values(b))
    if CVT:
        lo, hi = formats.e2m1_pair_cvt(b)
        tl.store(lo_ptr + i, lo.to(tl.float32))
        tl.store(hi_ptr + i, hi.to(tl.float32))


@triton.jit
def _encode_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out_ptr + i, formats.e4m3_codes(tl.load(x_ptr + i, mask=m, other=0.0)), mask=m)


def test_formats_every_code():
    dev = "cuda"
    b = torch.arange(256, dtype=torch.uint8, device=dev)
    outs = [torch.empty(256, dtype=torch.float32, device=dev) for _ in range(6)]
    cvt = CAP[0] >= 10   # Triton compiles for the specific target (sm_100a / sm_120a), which has the conversion
    _codes_kernel[(1,)](b, *outs, CVT=cvt, HW_FP8=CAP >= (8, 9))
    e2, e4, e4hw, e8, lo, hi = [o.cpu() for o in outs]
    codes = torch.arange(256)
    assert torch.equal(e2, e2m1_ref(codes & 15))
    ref = codes.to(torch.uint8).view(torch.float8_e4m3fn).float()
    nan = (codes & 0x7F) == 0x7F
    assert torch.equal(e4[~nan], ref[~nan]) and torch.equal(e4[nan].abs(), torch.full((2,), 480.0))
    assert torch.equal(e4hw[~nan], ref[~nan])
    e8ref = torch.where(codes == 255, torch.tensor(float("nan")), torch.ldexp(torch.ones(256), codes - 127))
    assert torch.equal(e8[:255], e8ref[:255]) and torch.isnan(e8[255])
    if cvt:
        assert torch.equal(lo, e2m1_ref(codes & 15)) and torch.equal(hi, e2m1_ref(codes >> 4))


def test_e4m3_codes_match_torch():
    dev = "cuda"
    # every FP32 bit pattern in [0, 448] at a stride, plus their negatives
    bits = torch.arange(0, 0x43E00001, 97, dtype=torch.int64)
    x = bits.to(torch.int32).view(torch.float32)
    x = torch.cat([x, -x]).to(dev)
    out = torch.empty(x.numel(), dtype=torch.uint8, device=dev)
    _encode_kernel[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
    ref = x.to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(out, ref)


@triton.jit
def _tree_kernel(x_ptr, rows_ptr, one_ptr, ROWS: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, ROWS)
    c = tl.arange(0, N)
    x = tl.load(x_ptr + r[:, None] * N + c[None, :])
    tl.store(rows_ptr + r, reduce.tree_sum_rows(x, ROWS, N))
    tl.store(one_ptr, reduce.tree_sum(tl.load(x_ptr + c), N))


def _host_tree(x):
    """The pairwise tree of adjacent elements over the last axis, in FP32."""
    while x.shape[-1] > 1:
        x = x[..., 0::2] + x[..., 1::2]
    return x[..., 0]


@pytest.mark.parametrize("warps", [1, 4, 8])
def test_tree_sums_have_the_trees_bits_for_every_warp_count(warps):
    rows, n = 16, 256
    x = (torch.randn(rows, n, device="cuda") * torch.logspace(-6, 6, n, device="cuda")).float()
    got_rows = torch.empty(rows, device="cuda")
    got_one = torch.empty(1, device="cuda")
    _tree_kernel[(1,)](x, got_rows, got_one, ROWS=rows, N=n, num_warps=warps)
    ref = _host_tree(x.cpu())
    assert torch.equal(got_rows.cpu(), ref)
    assert torch.equal(got_one.cpu()[0], ref[0])


@triton.jit
def _partials_kernel(p_ptr, out_ptr, n, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out_ptr + i, reduce.sum_partials(p_ptr, n, i, i < n, PARTS), mask=i < n)


def test_sum_partials_in_order():
    parts, n = 5, 100
    p = torch.randn(parts, n, device="cuda")
    out = torch.empty(n, device="cuda")
    _partials_kernel[(1,)](p, out, n, PARTS=parts, BLOCK=128)
    ref = p[0].clone()
    for s in range(1, parts):
        ref = ref + p[s]
    assert torch.equal(out, ref)


@triton.jit
def _act_kernel(x_ptr, out_ptr, n, ACT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m, other=0.0)
    tl.store(out_ptr + i, epilogue.act(x, ACT), mask=m)


@pytest.mark.parametrize("act", [epilogue.SILU, epilogue.GELU, epilogue.GELU_TANH, epilogue.SIGMOID,
                                 epilogue.IDENTITY])
def test_activations(act):
    x = torch.linspace(-30, 30, 1000, device="cuda")
    out = torch.empty_like(x)
    _act_kernel[(1,)](x, out, x.numel(), ACT=act, BLOCK=1024)
    f = torch.nn.functional
    ref = {epilogue.SILU: f.silu, epilogue.GELU: f.gelu, epilogue.GELU_TANH: lambda v: f.gelu(v, approximate="tanh"),
           epilogue.SIGMOID: torch.sigmoid, epilogue.IDENTITY: lambda v: v}[act](x.double()).float()
    torch.testing.assert_close(out, ref, rtol=2e-6, atol=2e-6)


@triton.jit
def _tile_kernel(tm_ptr, tn_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    tm, tn = load.grouped_tile(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
    tl.store(tm_ptr + pid, tm)
    tl.store(tn_ptr + pid, tn)


def test_grouped_tile_covers_every_tile_once():
    M, N, bm, bn = 1000, 700, 64, 128
    tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    tm = torch.empty(tiles, dtype=torch.int32, device="cuda")
    tn = torch.empty_like(tm)
    _tile_kernel[(tiles,)](tm, tn, M, N, BLOCK_M=bm, BLOCK_N=bn, GROUP_M=8)
    pairs = set(zip(tm.tolist(), tn.tolist()))
    assert len(pairs) == tiles and all(0 <= a < triton.cdiv(M, bm) and 0 <= b < triton.cdiv(N, bn) for a, b in pairs)
