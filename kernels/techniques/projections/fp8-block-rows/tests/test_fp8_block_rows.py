"""The decode rows and the prompt GEMMs (the register one and the staged one) give the same bits for every output, on
a repeat too, and under every knob setting tried (the knobs never move an output's order). tests/fp8_block_tests.cu checks the raw kernels against the
declared order's reference kernel in C++. The kernel is built for this GPU with crit_kernels.build.load (its default
knobs, or several configurations a test selects among)."""
import pytest
import torch

from crit_kernels.build import load, select

fb = load("fp8-block-rows")
dev = torch.device("cuda")


def operands():
    g = torch.Generator().manual_seed(0)
    rows, cols, tokens = 96, 704, 21
    codes = torch.randint(0, 0x7E, (rows, cols), dtype=torch.uint8, generator=g)
    codes = (codes | (torch.randint(0, 2, (rows, cols), generator=g) * 0x80).to(torch.uint8)).to(dev)
    scales = torch.randint(120, 134, (rows, cols // 32), dtype=torch.uint8, generator=g).to(dev)  # E8M0 bytes
    x = torch.randn(tokens, cols, generator=g).to(dev, torch.bfloat16)
    act_codes, act_scales = fb.quantize(x, 32, fb.MX)
    return rows, tokens, codes, scales, act_codes, act_scales


def run(module, rows, tokens, codes, scales, act_codes, act_scales):
    table = module.matrix_table([(codes, scales, 1, 32, 0.5)])  # the table holds pointers: keep the tensors alive
    decode = torch.zeros(tokens, rows, device=dev)
    jobs = [(0, list(range(t, min(t + 8, tokens))), list(range(t, min(t + 8, tokens)))) for t in range(0, tokens, 8)]
    module.rows(table, module.row_jobs(jobs, dev), rows, act_codes, act_scales, 32, decode)
    prompt = torch.zeros(tokens, rows, device=dev)
    module.gemm(table, module.gemm_jobs([(0, 0, tokens)], dev), rows, tokens, act_codes, act_scales, 32, prompt)
    again = torch.zeros_like(prompt)
    module.gemm(table, module.gemm_jobs([(0, 0, tokens)], dev), rows, tokens, act_codes, act_scales, 32, again)
    assert torch.equal(again, prompt)
    staged = torch.zeros_like(prompt)
    module.gemm_staged(table, module.gemm_jobs([(0, 0, tokens)], dev), rows, tokens, act_codes, act_scales, 32, staged)
    assert torch.equal(staged.view(torch.int32), prompt.view(torch.int32))   # the staged GEMM: the same bits
    return decode, prompt, again


def test_rows_and_gemm_agree_bit_for_bit():
    decode, prompt, _ = run(fb, *operands())   # and the GEMM's repeat its bits (run)
    assert torch.equal(decode.view(torch.int32), prompt.view(torch.int32))


def test_every_knob_setting_gives_the_default_bits():
    data = operands()
    decode, prompt, _ = run(fb, *data)
    rows = [{"WARPS": 4, "AHEAD": 1, "STREAM": 1}, {"WARPS": 8, "AHEAD": 4, "STREAM": 0}]
    gemm = [{"WARPS": 2, "TILES": 1, "UNROLL": 1}, {"WARPS": 16, "TILES": 8, "UNROLL": 2},
            {"WARPS": 4, "TILES": 2, "UNROLL": 4}]
    staged = [{"BM": 32, "BN": 16, "WM": 2, "WN": 1, "GROUP": 1, "RUN": 1},
              {"BM": 64, "BN": 64, "WM": 2, "WN": 2, "GROUP": 2, "RUN": 2},
              {"BM": 128, "BN": 32, "WM": 4, "WN": 2, "GROUP": 1, "RUN": 2}]
    other = load("fp8-block-rows", configs={"rows": rows, "gemm": gemm, "gemm-staged": staged})   # defaults: index 0
    for index in range(1, max(len(rows), len(gemm)) + 1):
        select(other, "rows", min(index, len(rows)))
        select(other, "gemm", min(index, len(gemm)))
        select(other, "gemm-staged", min(index, len(staged)))
        d, p, _ = run(other, *data)
        assert torch.equal(d.view(torch.int32), decode.view(torch.int32)), index
        assert torch.equal(p.view(torch.int32), prompt.view(torch.int32)), index
    # rows forward to decode-gemv's rows-fp8-block, and select() follows: a configuration it does not hold is refused
    select(other, "rows", len(rows) + 1)
    with pytest.raises(RuntimeError):
        run(other, *data)
    select(other, "rows", -1)


def _fp64(codes, scales, scale_rows, scale_k, act_codes, act_scales, group):
    """The exact product of the stored values in FP64, and of their magnitudes (the FP32 accumulation bound's scale)."""
    w = codes.view(torch.float8_e4m3fn).double()
    s = torch.pow(2.0, scales.double() - 127) if scales.dtype == torch.uint8 else scales.double()
    w = w * s.repeat_interleave(scale_rows, 0)[: w.shape[0]].repeat_interleave(scale_k, 1)
    a = act_codes.view(torch.float8_e4m3fn).double() * act_scales.double().repeat_interleave(group, 1)
    return a @ w.t() * 0.5, a.abs() @ w.abs().t() * 0.5


def test_the_native_gemm_is_within_fp32_accumulation_of_the_exact_product():
    """gemm-native is not the declared order: no bits to match, only the FP64 product (MXFP8 on the block-scaled path,
    and an FP8 checkpoint's 128 x 128 FP32 scales with amax activations on the generic one), two jobs, tokens and
    rows off the tiles; and not NaN anywhere."""
    g = torch.Generator().manual_seed(5)
    for rows, cols, tokens, mx in [(200, 768, 37, True), (96, 2560, 300, True), (130, 704, 45, False)]:
        codes = torch.randint(0, 0x7E, (rows, cols), dtype=torch.uint8, generator=g)
        codes = (codes | (torch.randint(0, 2, (rows, cols), generator=g) * 0x80).to(torch.uint8)).to(dev)
        if mx:
            scales, scale_rows, scale_k = torch.randint(120, 134, (rows, cols // 32), dtype=torch.uint8,
                                                        generator=g).to(dev), 1, 32
        else:
            scales, scale_rows, scale_k = (torch.rand((rows + 127) // 128, cols // 64, generator=g) * 0.01 +
                                           0.001).to(dev), 128, 64
        x = torch.randn(tokens, cols, generator=g).to(dev)
        act_codes, act_scales = fb.quantize(x, 32, fb.MX if mx else fb.AMAX)
        table = fb.matrix_table([(codes, scales, scale_rows, scale_k, 0.5)])
        jobs = fb.gemm_jobs([(0, 0, tokens // 3), (0, tokens // 3, tokens - tokens // 3)], dev)
        out = torch.full((tokens, rows), float("nan"), device=dev)
        fb.gemm_native(table, jobs, rows, tokens - tokens // 3, act_codes, act_scales, 32, out, mx=mx)
        value, magnitude = _fp64(codes, scales, scale_rows, scale_k, act_codes, act_scales, 32)
        error = (out.double() - value).abs()
        # the block-scaled MMA's own block sums are not FP32 round to nearest (both paths run it on sm_120a): 1e-5 of
        # the magnitude (3.3e-6 measured)
        assert not out.isnan().any() and bool((error <= magnitude * 1e-5 + 1e-30).all()), (rows, cols, tokens, mx)


def test_the_channel_gemm_is_within_fp32_accumulation_of_the_exact_product():
    """gemm-native-channel (per-channel W8A8: FP32 weight scales constant along K, one activation scale a token) in FP32
    and BF16 outputs, against the FP64 product: a scale a row, a block of 128 rows and the tensor; two jobs, tokens and
    rows off the tiles; gemm_native(mx=False) takes it for such a table and agrees with it bit for bit. A matrix whose
    scales change along K gets NaN outputs (the call's statement broken)."""
    g = torch.Generator().manual_seed(7)
    for rows, cols, tokens, scale_rows in [(200, 768, 37, 1), (96, 2560, 300, 128), (130, 704, 45, 1 << 20)]:
        codes = torch.randint(0, 0x7E, (rows, cols), dtype=torch.uint8, generator=g)
        codes = (codes | (torch.randint(0, 2, (rows, cols), generator=g) * 0x80).to(torch.uint8)).to(dev)
        scales = (torch.rand(-(-rows // scale_rows), 1, generator=g) * 0.01 + 0.001).to(dev)
        x = torch.randn(tokens, cols, generator=g).to(dev)
        act_codes, act_scales = fb.quantize(x, cols, fb.AMAX)   # one scale a token
        table = fb.matrix_table([(codes, scales, scale_rows, cols, 0.5)])
        assert table.crit_channel
        jobs = fb.gemm_jobs([(0, 0, tokens // 3), (0, tokens // 3, tokens - tokens // 3)], dev)
        value, magnitude = _fp64(codes, scales, scale_rows, cols, act_codes, act_scales, cols)
        out = torch.full((tokens, rows), float("nan"), device=dev)
        fb.gemm_native_channel(table, jobs, rows, tokens - tokens // 3, act_codes, act_scales, out)
        error = (out.double() - value).abs()
        assert not out.isnan().any() and bool((error <= magnitude * 1e-5 + 1e-30).all()), (rows, cols, tokens)
        half = torch.full((tokens, rows), float("nan"), device=dev, dtype=torch.bfloat16)
        fb.gemm_native(table, jobs, rows, tokens - tokens // 3, act_codes, act_scales, cols, half, mx=False)
        assert torch.equal(half, out.to(torch.bfloat16)), (rows, cols, tokens)   # one rounding of the same sums
    blocks = (torch.rand(rows, cols // 32, generator=g) + 0.5).to(dev)   # scales along K: not channel-scaled
    table = fb.matrix_table([(codes, blocks, 1, 32, 1.0)])
    assert not table.crit_channel
    out = torch.zeros(tokens, rows, device=dev)
    fb.gemm_native_channel(table, fb.gemm_jobs([(0, 0, tokens)], dev), rows, tokens, act_codes, act_scales, out)
    assert out.isnan().all()


def test_the_native_gemm_writes_bf16_outputs():
    """gemm_native into a BF16 out: the FP32 out's values rounded once (MXFP8 block-scaled and generic paths)."""
    g = torch.Generator().manual_seed(9)
    rows, cols, tokens = 200, 768, 37
    codes = torch.randint(0, 0x7E, (rows, cols), dtype=torch.uint8, generator=g)
    codes = (codes | (torch.randint(0, 2, (rows, cols), generator=g) * 0x80).to(torch.uint8)).to(dev)
    scales = torch.randint(120, 134, (rows, cols // 32), dtype=torch.uint8, generator=g).to(dev)
    x = torch.randn(tokens, cols, generator=g).to(dev)
    table = fb.matrix_table([(codes, scales, 1, 32, 0.5)])
    jobs = fb.gemm_jobs([(0, 0, tokens)], dev)
    for mx in (True, False):
        act_codes, act_scales = fb.quantize(x, 32, fb.MX if mx else fb.AMAX)
        out = torch.zeros(tokens, rows, device=dev)
        fb.gemm_native(table, jobs, rows, tokens, act_codes, act_scales, 32, out, mx=mx)
        half = torch.zeros(tokens, rows, device=dev, dtype=torch.bfloat16)
        fb.gemm_native(table, jobs, rows, tokens, act_codes, act_scales, 32, half, mx=mx)
        assert torch.equal(half, out.to(torch.bfloat16)), mx


def test_the_blocks_gemm_is_within_fp32_accumulation_of_the_exact_product():
    """gemm_native(mx=False) on scales that change along K (gemm-native-blocks: each scale span summed in the tensor
    cores, one FMA a span, scales constant along K in the epilogue), FP32 and BF16 out, two jobs, tokens and rows off the
    tiles, against the FP64 product: 128 x 128 FP32 weight blocks with activations per 128 (an FP8 block checkpoint),
    the same weights with one activation scale a token, row scales with activations per 128, MXFP8 E8M0 weights with
    amax activation scales per 32 (span one block), and 128 x 128 blocks against activations per 32."""
    g = torch.Generator().manual_seed(11)
    cases = [  # rows, cols, tokens, weight scales (fp32 or e8m0), scale_rows, scale_k, activation group
        (200, 768, 37, "fp32", 128, 128, 128),
        (130, 1024, 300, "fp32", 128, 128, 1024),
        (96, 640, 45, "fp32", 1, 640, 128),
        (100, 512, 70, "e8m0", 1, 32, 32),
        (260, 768, 33, "fp32", 128, 128, 32),
    ]
    for rows, cols, tokens, kind, scale_rows, scale_k, group in cases:
        codes = torch.randint(0, 0x7E, (rows, cols), dtype=torch.uint8, generator=g)
        codes = (codes | (torch.randint(0, 2, (rows, cols), generator=g) * 0x80).to(torch.uint8)).to(dev)
        shape = (-(-rows // scale_rows), cols // scale_k)
        if kind == "e8m0":
            scales = torch.randint(120, 134, shape, dtype=torch.uint8, generator=g).to(dev)
        else:
            scales = (torch.rand(shape, generator=g) * 0.01 + 0.001).to(dev)
        x = torch.randn(tokens, cols, generator=g).to(dev)
        act_codes, act_scales = fb.quantize(x, group, fb.AMAX)
        table = fb.matrix_table([(codes, scales, scale_rows, scale_k, 0.5)])
        assert table.crit_channel == (kind == "fp32" and scale_k == cols)   # the channel GEMM needs one a token too
        jobs = fb.gemm_jobs([(0, 0, tokens // 3), (0, tokens // 3, tokens - tokens // 3)], dev)
        value, magnitude = _fp64(codes, scales, scale_rows, scale_k, act_codes, act_scales, group)
        out = torch.full((tokens, rows), float("nan"), device=dev)
        fb.gemm_native(table, jobs, rows, tokens - tokens // 3, act_codes, act_scales, group, out, mx=False)
        error = (out.double() - value).abs()
        case = (rows, cols, tokens, kind, scale_rows, scale_k, group)
        assert not out.isnan().any() and bool((error <= magnitude * 1e-5 + 1e-30).all()), case
        half = torch.full((tokens, rows), float("nan"), device=dev, dtype=torch.bfloat16)
        fb.gemm_native(table, jobs, rows, tokens - tokens // 3, act_codes, act_scales, group, half, mx=False)
        assert torch.equal(half, out.to(torch.bfloat16)), case   # one rounding of the same sums
