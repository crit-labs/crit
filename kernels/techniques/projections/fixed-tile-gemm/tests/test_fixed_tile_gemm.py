"""The fixed-tile GEMM: x times dense within FP32 accumulation of the FP64 product, and a row's result the same bits
whatever rows share the call, across the switch between the large and the small tile, and across knob settings that
share BLOCK_K (the speed knobs never move a row's bits). The kernel is staged with crit_kernels.build.load (its
default knobs, or the ones a test names)."""
import pytest
import torch

from crit_kernels.build import load, select

ftg = load("fixed-tile-gemm")
dev = torch.device("cuda")


def operands(m, n, k, dtype=torch.bfloat16, seed=4):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(m, k, generator=g).to(dev, dtype), (torch.randn(n, k, generator=g) * 0.05).to(dev, dtype))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_it_multiplies_within_fp32_accumulation_of_the_fp64_product(dtype):
    x, w = operands(300, 96, 256, dtype)
    exact = x.double() @ w.double().t()
    found = ftg.mm(x, w, torch.float32).double()
    assert float((found - exact).abs().max() / exact.abs().max()) < 1e-5
    assert ftg.mm(x, w).dtype == dtype


def test_a_row_keeps_its_bits_whatever_rows_share_the_call():
    # 45,056 rows of a 96-row matrix fill 352 large tiles (the large one on a device of up to 176 multiprocessors);
    # the slices below fill too few and take the small one: the switch keeps every row's order.
    rows = 45056
    x, w = operands(rows, 96, 256)
    whole = ftg.mm(x, w)
    for first, count in ((0, 1), (0, 9), (17, 40), (300, 700), (rows - 9, 9)):
        assert torch.equal(ftg.mm(x[first:first + count].clone(), w), whole[first:first + count]), (first, count)
    assert torch.equal(ftg.mm(x, w), whole)   # and on a repeat


def test_real_widths_keep_rows_too():
    x, w = operands(1024, 5120, 2048)
    whole = ftg.mm(x, w)
    for first, count in ((0, 9), (511, 64), (1000, 24)):
        assert torch.equal(ftg.mm(x[first:first + count].clone(), w), whole[first:first + count])


def test_it_refuses_mixed_or_unsupported_dtypes():
    x, w = operands(16, 32, 64)
    with pytest.raises(ValueError):
        ftg.mm(x, w.half())
    with pytest.raises(ValueError):
        ftg.mm(x.float(), w.float())


def test_speed_knobs_keep_a_row_s_bits_and_defaults_are_the_untuned_tiles():
    dev_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    assert ftg.tile(4096 * dev_sms, 128, 256, dev) == ftg.MM_TILE and ftg.tile(64, 128, 256, dev) == ftg.MM_SMALL
    # 1 to 32 rows (default_rows): a 16 x 64 tile in 4 warps and 6 stages, the same K loop
    assert ftg.tile(16, 128, 256, dev) == dict(ftg.MM_SMALL, BLOCK_M=16, num_stages=6)
    x, w = operands(1024, 1536, 512)
    whole = ftg.mm(x, w)
    settings = [{"BLOCK_M": 64, "BLOCK_N": 256, "WARPS": 8, "STAGES": 2, "SMALL": 0},
                {"BLOCK_M": 256, "BLOCK_N": 64, "WARPS": 8, "STAGES": 3, "SMALL": 0},
                {"BLOCK_M": 64, "BLOCK_N": 64, "WARPS": 4, "STAGES": 3, "SMALL": 1}]
    other = load("fixed-tile-gemm", configs={"mm": settings})   # the defaults are configuration 0
    for index, knobs in enumerate(settings, 1):
        select(other, "mm", index)
        assert torch.equal(other.mm(x, w), whole), knobs
        assert torch.equal(other.mm(x[:9].clone(), w), whole[:9]), knobs


def test_the_small_row_tile_keeps_a_row_s_bits_at_real_widths():
    # Qwen3.8-27B-like widths: 1 to 32 rows take the 16 x 64 tile, more the small or the large one
    x, w = operands(600, 5120, 6144)
    whole = ftg.mm(x, w)
    for first, count in ((0, 1), (0, 16), (5, 32), (100, 33), (200, 128)):
        assert torch.equal(ftg.mm(x[first:first + count].clone(), w), whole[first:first + count]), (first, count)


@pytest.mark.parametrize("m,n,k", [(16, 5120, 6144), (64, 5120, 17408), (128, 1536, 4096), (300, 96, 256)])
def test_split_k_is_within_fp32_accumulation_and_deterministic(m, n, k):
    x, w = operands(m, n, k)
    exact = x.double() @ w.double().t()
    found = ftg.mm_split(x, w, torch.float32)
    assert float((found.double() - exact).abs().max() / exact.abs().max()) < 1e-5
    assert torch.equal(ftg.mm_split(x, w, torch.float32), found)          # the same bits on a repeat
    assert ftg.mm_split(x, w).dtype == torch.bfloat16
    t = ftg.split_tile(m, n, k, dev)
    assert 1 <= t["SPLIT"] <= ftg.SPLITS_MAX


def test_split_k_splits_narrow_matrices_at_few_rows():
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    assert ftg.split_tile(64, 5120, 17408, dev)["SPLIT"] == max(1, min(4, round(sms / 80)))
    assert ftg.split_tile(8192, 5120, 17408, dev)["SPLIT"] == 1
    assert ftg.split_tile(16, 5120, 1024, dev)["SPLIT"] == 1      # K too short to split


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_the_cuda_gemm_is_within_fp32_accumulation_of_the_fp64_product(dtype):
    """mm_cuda and mm_cuda_split (structures cuda-mm, cuda-mm-split) against FP64, within 1e-5 of the products'
    magnitude (FP32 sums over K up to 17,408): rows and columns off the tiles, FP32 and 16-bit out (the FP32 sums
    rounded once), the same bits on a repeat."""
    for m, n, k in [(300, 96, 256), (16, 5120, 6144), (129, 200, 1032), (64, 5120, 17408)]:
        x, w = operands(m, n, k, dtype)
        exact = x.double() @ w.double().t()
        magnitude = x.double().abs() @ w.double().abs().t()   # FP32 sums over K: the bound scales with it
        for f in (ftg.mm_cuda, ftg.mm_cuda_split):
            found = f(x, w, torch.float32)
            assert bool(((found.double() - exact).abs() <= magnitude * 1e-5).all()), (f.__name__, m, n, k)
            half = f(x, w)
            assert half.dtype == dtype and torch.equal(half, found.to(dtype)), (f.__name__, m, n, k)
            assert torch.equal(f(x, w, torch.float32), found)


def test_the_cuda_gemm_keeps_a_row_s_bits_whatever_rows_and_configuration():
    """cuda-mm: a row's bits are the same in every slice of the call and in every configuration (the tile, warps,
    stages, epilogue and BK never move an output's order), the defaults' small-row tiles included."""
    x, w = operands(600, 5120, 2048)
    whole = ftg.mm_cuda(x, w)
    for first, count in ((0, 1), (0, 16), (5, 32), (100, 33), (200, 128), (300, 300)):
        assert torch.equal(ftg.mm_cuda(x[first:first + count].clone(), w), whole[first:first + count]), (first, count)
    settings = [{"TM": 64, "TN": 256, "WM": 1, "WN": 4, "BK": 64, "STAGES": 4, "EPI": 0},
                {"TM": 256, "TN": 64, "WM": 4, "WN": 2, "BK": 128, "STAGES": 2, "EPI": 1},
                {"TM": 16, "TN": 128, "WM": 1, "WN": 4, "BK": 128, "STAGES": 3, "EPI": 1}]
    other = load("fixed-tile-gemm", configs={"cuda-mm": settings})
    for index, knobs in enumerate(settings, 1):
        select(other, "cuda-mm", index)
        assert torch.equal(other.mm_cuda(x, w), whole), knobs
        assert torch.equal(other.mm_cuda(x[:9].clone(), w), whole[:9]), knobs
    select(other, "cuda-mm", -1)
    assert ftg.cuda_ok(x, w) and not ftg.cuda_ok(x[:, 1:], w[:, 1:])
