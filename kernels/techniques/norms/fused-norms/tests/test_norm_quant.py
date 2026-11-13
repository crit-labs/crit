"""The quantizing norms (rms_norm_fp8_rows, rms_norm_nvfp4) against rms_norm followed by the projections' quantizers
(mma-rows quantize_fp8_rows and quantize_nvfp4): the same codes and scales, bit for bit, with and without a residual,
an offset weight, NaN rows and rows of zeros, at row counts across a program's rows; the residual as rms_norm leaves
it. Built for this GPU by crit_kernels.build; the quantizers' kernel too."""
import pytest
import torch

from crit_kernels.build import load

fn = load("fused-norms")
dev = torch.device("cuda")


@pytest.fixture(scope="module")
def mma():
    try:
        return load("mma-rows")
    except Exception as error:  # noqa: BLE001 - no toolkit or an architecture it does not cover
        pytest.skip(f"mma-rows does not build here: {error}")


def rows(seed, m, k, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(m, k, generator=g) * scale).to(dev, torch.bfloat16)


def weight(seed, k):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(k, generator=g) * 0.1 + 1).to(dev, torch.bfloat16)


def specials(x):
    x = x.clone()
    x[1].zero_()
    if x.shape[0] > 2:
        x[2, 5] = float("nan")
    return x


@pytest.mark.parametrize("m", [1, 3, 17, 130])
@pytest.mark.parametrize("k", [2048, 5120])
@pytest.mark.parametrize("res", [False, True])
def test_fp8_rows_equal_norm_then_quantizer(mma, m, k, res):
    x, w = rows(m * 7 + k, m, k), weight(k, k)
    if m > 1:
        x = specials(x)
    r1 = rows(5, m, k).float() if res else None
    r2 = r1.clone() if res else None
    y = fn.rms_norm(x, w, 1e-6, 0.5, residual=r1, out_dtype=torch.bfloat16)
    y = y[0] if res else y
    want_q = torch.empty(m, k, dtype=torch.uint8, device=dev)
    want_s = torch.empty(m, dtype=torch.float32, device=dev)
    mma.quantize_fp8_rows(y, want_q, want_s)
    got = fn.rms_norm_fp8_rows(x, w, 1e-6, 0.5, residual=r2)
    assert torch.equal(got[0], want_q)
    assert torch.equal(got[1].view(torch.int32), want_s.view(torch.int32))
    if res:
        assert torch.equal(r1.view(torch.int32), r2.view(torch.int32))


@pytest.mark.parametrize("m", [1, 3, 17, 130])
@pytest.mark.parametrize("k", [2048, 5120])
@pytest.mark.parametrize("res", [False, True])
def test_nvfp4_equal_norm_then_quantizer(mma, m, k, res):
    x, w = rows(m * 11 + k, m, k), weight(k + 1, k)
    if m > 1:
        x = specials(x)
    r1 = rows(9, m, k).float() if res else None
    r2 = r1.clone() if res else None
    g = 448.0 * 6.0 / 3.0   # an input scale of the order a checkpoint stores
    y = fn.rms_norm(x, w, 1e-6, residual=r1, out_dtype=torch.bfloat16)
    y = y[0] if res else y
    want_c = torch.empty(m, k // 2, dtype=torch.uint8, device=dev)
    want_s = torch.zeros(fn.nvfp4_scale_bytes(m, k), dtype=torch.uint8, device=dev)
    mma.quantize_nvfp4(y, g, want_c, want_s)
    got = fn.rms_norm_nvfp4(x, w, 1e-6, g, residual=r2)
    assert torch.equal(got[0], want_c)
    assert torch.equal(got[1], want_s)
    if res:
        assert torch.equal(r1.view(torch.int32), r2.view(torch.int32))


def test_shapes_are_refused():
    x = rows(1, 4, 40)
    with pytest.raises(ValueError):
        fn.rms_norm_nvfp4(x, None, 1e-6, 1.0)   # K not a multiple of 16
    with pytest.raises(ValueError):
        fn.rms_norm_fp8_rows(rows(1, 2, 64)[None], None, 1e-6)
