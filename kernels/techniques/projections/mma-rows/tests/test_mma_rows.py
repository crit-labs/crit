"""mma-rows against an FP64 reference of the stored values: every weight format within FP32 accumulation, with and
without splits of K, each row's bits the same alone and among up to 63 others (and whatever `groups` and `warps`), the
quantized activations the bits of torch quantizers written as a prompt's are, and the calls captured in a CUDA graph
(the split counters left zero for the next replay); a build with other knobs than the defaults takes its plan from the
tuning and gives the default build's bits for the same plan.

The kernel is built for this GPU by crit_kernels.build (default knobs; the cache is CRIT_KERNEL_CACHE)."""
import math

import pytest
import torch

from crit_kernels import build as kbuild

mr = kbuild.load("mma-rows")
dev = torch.device("cuda")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    dtype=torch.float64)
FORMATS = [("nvfp4", "bf16", False), ("nvfp4", "bf16", True), ("nvfp4", "nvfp4", True), ("fp8", "bf16", False),
           ("fp8", "fp8_rows", False), ("mxfp8", "bf16", False), ("bf16", "bf16", False)]
INPUT_SCALE = 300.0


def to_blocked(scales):
    rows, columns = scales.shape
    rt, ct = math.ceil(rows / 128), math.ceil(columns / 4)
    padded = torch.zeros(rt * 128, ct * 4, dtype=scales.dtype, device=scales.device)
    padded[:rows, :columns] = scales
    tiles = padded.view(rt, 128, ct, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def matrix(weights, n, k, blocked=False, seed=5):
    """(w, scales, row_scale) as a checkpoint stores them, and the values in FP64 [N][K] (what the sums multiply,
    the row scale included)."""
    g = torch.Generator().manual_seed(seed)
    if weights == "bf16":
        w = (torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16)
        return (w.to(dev), None, None), w.double()
    if weights == "nvfp4":
        codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)
        scales = (torch.rand(n, k // 16, generator=g) * 4 + 0.01).to(torch.float8_e4m3fn)
        inverse = (torch.rand(n, generator=g) + 0.5) / 64
        c = codes.long()
        v = torch.stack((E2M1[c & 15], E2M1[c >> 4]), dim=-1).reshape(n, -1).view(n, -1, 16)
        v = (v * scales.double()[..., None]).reshape(n, k) * inverse.double()[:, None]
        stored = to_blocked(scales) if blocked else scales
        return (codes.to(dev), stored.view(torch.uint8).to(dev), inverse.to(dev)), v
    codes = torch.randint(0, 0x7E, (n, k), generator=g, dtype=torch.uint8)
    codes |= torch.randint(0, 2, (n, k), generator=g, dtype=torch.uint8) << 7
    values = codes.view(torch.float8_e4m3fn).double()
    if weights == "fp8":
        scale = torch.rand(n, generator=g) / 50 + 1e-3
        return (codes.to(dev), None, scale.to(dev)), values * scale.double()[:, None]
    exponents = torch.randint(118, 128, (n, k // 32), generator=g, dtype=torch.uint8)
    power = torch.exp2(exponents.double() - 127).repeat_interleave(32, dim=1)
    return (codes.to(dev), exponents.to(dev), None), values * power


def activations(m, k, seed=8):
    return torch.randn(m, k, generator=torch.Generator().manual_seed(seed)).to(dev, torch.bfloat16)


def buffers(m, n, k, splits=1, groups=1, warps=4):
    """scratch, partial and counters for a call (counters zero)."""
    scratch = torch.empty(mr.scratch_bytes(m, k), dtype=torch.uint8, device=dev)
    partial = torch.empty(max(1, mr.partial_elements(m, n, splits)), dtype=torch.float32, device=dev)
    counters = torch.zeros(mr.row_tiles(n, 1, warps), dtype=torch.int32, device=dev)   # tiles of 64 rows take one group
    return scratch, partial, counters


def call(x, stored, weights, acts="bf16", blocked=False, dtype=torch.float32, groups=1, warps=4, splits=1, out=None,
         bufs=None):
    w, scales, row_scale = stored
    m, n, k = x.shape[0], w.shape[0], x.shape[1]
    out = torch.empty(m, n, dtype=dtype, device=dev) if out is None else out
    scratch, partial, counters = bufs or buffers(m, n, k, splits, groups, warps)
    return mr.mm_rows(x, w, out, weights, scales, row_scale, acts, INPUT_SCALE, blocked, groups, warps, splits, scratch,
                      partial, counters)


# The quantizers as torch writes a prompt's (a division by a Python constant is a product with its FP32 reciprocal).
def fp8_rows(x):
    x = x.float()
    scale = (x.abs().amax(dim=-1) / 448.0).clamp(min=1e-12)
    return (x / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
TIES = torch.tensor([0.75, 1.75, 3.5])


def nvfp4_quantize(x, input_scale):
    x = x.float()
    t, k = x.shape
    blocks = x.view(t, k // 16, 16)
    scale = (blocks.abs().amax(dim=-1) / 6.0 * input_scale).clamp(max=448.0).to(torch.float8_e4m3fn)
    decoded = scale.float() / input_scale
    scaled = torch.where(decoded[..., None] > 0, blocks / decoded[..., None].clamp(min=1e-30), torch.zeros_like(blocks))
    magnitude = scaled.abs().clamp(max=6.0)
    code = torch.bucketize(magnitude, BOUNDS.to(x.device), right=False)
    code = code + torch.isin(magnitude, TIES.to(x.device)).to(code.dtype)
    code = (code | ((scaled < 0).to(code.dtype) << 3)).to(torch.uint8).view(t, k)
    return code, scale


def nvfp4_values(x, input_scale):
    code, scale = nvfp4_quantize(x, input_scale)
    t, k = code.shape
    values = E2M1.float().to(x.device)[code.long()].view(t, -1, 16) * scale.float()[..., None]
    return values.view(t, k).to(torch.bfloat16)


@pytest.mark.parametrize("fmt", [f for f in FORMATS if f[1] == "bf16"])
@pytest.mark.parametrize("shape", [(96, 256), (1000, 5120), (130, 2048), (40, 640), (72, 96), (300, 1056)])
def test_every_format_multiplies_by_its_stored_values(fmt, shape):
    weights, acts, blocked = fmt
    n, k = shape
    stored, values = matrix(weights, n, k, blocked)
    steps = math.ceil(k / mr.VALUES[weights])
    for rows in (1, 7, 16, 64):
        x = activations(rows, k)
        exact = x.double() @ values.t().to(dev)
        magnitude = x.double().abs() @ values.abs().t().to(dev)
        for splits in sorted({1, min(3, steps), min(32, steps)}):
            found = call(x, stored, weights, acts, blocked, splits=splits).double()
            assert bool(((found - exact).abs() <= magnitude * 2e-5 + 1e-30).all()), (weights, rows, splits)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("splits", [1, 3])
def test_a_row_keeps_its_bits_alone_and_among_up_to_64_and_whatever_the_groups_and_warps(fmt, splits):
    weights, acts, blocked = fmt
    n, k = 1000, 2048
    stored, _ = matrix(weights, n, k, blocked)
    x = activations(64, k)
    alone = torch.cat([call(x[r:r + 1].clone(), stored, weights, acts, blocked, splits=splits) for r in range(64)])
    for rows in list(range(2, 18)) + [23, 31, 32, 33, 47, 64]:
        for groups, warps in ((1, 4), (2, 4), (1, 1), (2, 2)):
            part = call(x[:rows], stored, weights, acts, blocked, groups=groups, warps=warps, splits=splits)
            assert torch.equal(part, alone[:rows]), (weights, acts, rows, groups, warps, splits)


def test_16_bit_outputs_round_the_fp32_result_once():
    stored, _ = matrix("nvfp4", 300, 1024)
    x = activations(5, 1024)
    full = call(x, stored, "nvfp4")
    assert torch.equal(call(x, stored, "nvfp4", dtype=torch.bfloat16), full.to(torch.bfloat16))
    assert torch.equal(call(x, stored, "nvfp4", dtype=torch.float16), full.to(torch.float16))


def test_the_quantizers_give_the_torch_quantizers_bits():
    x = activations(37, 2048, seed=3) * 4
    x[3, :100] = 0   # a block of zeros
    q = torch.empty(37, 2048, dtype=torch.float8_e4m3fn, device=dev)
    scale = torch.empty(37, dtype=torch.float32, device=dev)
    mr.quantize_fp8_rows(x, q, scale)
    want_q, want_scale = fp8_rows(x)
    assert torch.equal(scale, want_scale)
    assert torch.equal(q.view(torch.uint8), want_q.view(torch.uint8))
    for k in (1001, 20000):   # values a load apart from 8, and past what a thread holds
        xk = (activations(5, 20480, seed=6) * 2)[:, :k]
        qk = torch.empty(5, k, dtype=torch.float8_e4m3fn, device=dev)
        sk = torch.empty(5, dtype=torch.float32, device=dev)
        mr.quantize_fp8_rows(xk, qk, sk)
        want_q, want_scale = fp8_rows(xk)
        assert torch.equal(sk, want_scale) and torch.equal(qk.view(torch.uint8), want_q.view(torch.uint8)), k
    codes = torch.empty(37, 1024, dtype=torch.uint8, device=dev)
    blocked = torch.full((128 * 128,), 0x55, dtype=torch.uint8, device=dev)   # the padding is written too
    mr.quantize_nvfp4(x, INPUT_SCALE, codes, blocked)
    want_codes, want_scales = nvfp4_quantize(x, INPUT_SCALE)
    assert torch.equal(codes, want_codes[:, 0::2] | (want_codes[:, 1::2] << 4))
    assert torch.equal(blocked, to_blocked(want_scales.view(torch.uint8)))


def test_quantized_activations_multiply_as_the_quantizers_values():
    n, k = 500, 2048
    x = activations(16, k, seed=4) * 3
    stored, _ = matrix("fp8", n, k)
    q, s = fp8_rows(x)
    want = call(q.to(torch.bfloat16), stored, "fp8") * s[:, None]
    assert torch.equal(call(x, stored, "fp8", "fp8_rows"), want)
    stored, _ = matrix("nvfp4", n, k, blocked=True)
    want = call(nvfp4_values(x, INPUT_SCALE), stored, "nvfp4", blocked=True) * (1.0 / INPUT_SCALE)
    assert torch.equal(call(x, stored, "nvfp4", "nvfp4", blocked=True), want)


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_captured_call_replays_the_eager_bits(fmt):
    weights, acts, blocked = fmt
    stored, _ = matrix(weights, 640, 1024, blocked)
    for rows, splits in ((1, 1), (16, 2), (40, 2)):
        static = activations(rows, 1024)
        out = torch.empty(rows, 640, dtype=torch.float32, device=dev)
        bufs = buffers(rows, 640, 1024, splits)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call(static, stored, weights, acts, blocked, splits=splits, out=out, bufs=bufs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(static, stored, weights, acts, blocked, splits=splits, out=out, bufs=bufs)
        for seed in (21, 22):   # twice: the counters are back at zero after a replay
            fresh = activations(rows, 1024, seed=seed)
            static.copy_(fresh)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, call(fresh, stored, weights, acts, blocked, splits=splits)), (weights, acts, rows)
        assert int(bufs[2].abs().sum()) == 0


def test_it_refuses_what_it_cannot_read():
    stored, _ = matrix("nvfp4", 96, 256)
    w, scales, inverse = stored
    out = torch.empty(65, 96, device=dev)
    with pytest.raises(RuntimeError, match="1 to 64 rows"):
        mr.mm_rows(activations(65, 256), w, out, "nvfp4", scales, inverse)
    with pytest.raises(RuntimeError, match="multiple of 32"):
        mr.mm_rows(activations(2, 208), w[:, :104].contiguous(), out[:2], "nvfp4", scales[:, :13].contiguous(), inverse)
    with pytest.raises(RuntimeError, match="16-byte boundaries"):
        mr.mm_rows(torch.zeros(2, 264, dtype=torch.bfloat16, device=dev)[:, 4:].contiguous().as_strided((2, 256), (260, 1)),
                   w, out[:2], "nvfp4", scales, inverse)
    with pytest.raises(RuntimeError, match="FP8 rows"):
        mr.mm_rows(activations(2, 256), w, out[:2], "nvfp4", scales, inverse, acts="fp8_rows")
    with pytest.raises(RuntimeError, match="row scale"):
        mr.mm_rows(activations(2, 256), w, out[:2], "nvfp4", scales, None)
    with pytest.raises(RuntimeError, match="x's device"):
        mr.mm_rows(activations(2, 256), w, out[:2], "nvfp4", scales, inverse.cpu())
    with pytest.raises(RuntimeError, match="partial and counters"):
        mr.mm_rows(activations(2, 256), w, out[:2], "nvfp4", scales, inverse, splits=2)
    with pytest.raises(RuntimeError, match="scratch"):
        mr.mm_rows(activations(2, 256), w, out[:2], "nvfp4", scales, inverse, acts="nvfp4", input_scale=3.0)


def test_plan_comes_from_the_shape_and_the_device():
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    for n, k, weights in ((34816, 5120, "nvfp4"), (5120, 17408, "nvfp4"), (16384, 5120, "fp8"), (96, 5120, "bf16"),
                          (248320, 2560, "fp8"), (2560, 640, "nvfp4"), (44032, 7168, "nvfp4"), (7168, 22016, "nvfp4")):
        groups, warps, splits = mr.plan(n, k, weights)
        assert (groups, warps, splits) == mr.plan(n, k, weights)
        assert groups in (1, 2) and 1 <= warps <= 4
        assert 1 <= splits <= min(32, math.ceil(k / mr.VALUES[weights]))
        tiles = mr.row_tiles(n, groups, warps)
        assert tiles * splits >= min(sms, tiles * math.ceil(k / mr.VALUES[weights]), tiles * 32)
    with pytest.raises(ValueError, match="multiple of 32"):
        mr.plan(100, 200, "nvfp4")


FP4 = pytest.mark.skipif(not mr.fp4_mma(dev), reason="no FP4 MMA on this device or build (the sm_120 family)")


def fp4_call(x, stored, splits=1, warps=4, out=None, dtype=torch.float32, bufs=None):
    w, scales, row_scale = stored
    m, n, k = x.shape[0], w.shape[0], x.shape[1]
    out = torch.empty(m, n, dtype=dtype, device=dev) if out is None else out
    if bufs is None:
        scratch = torch.empty(mr.scratch_bytes(m, k, "nvfp4_mma"), dtype=torch.uint8, device=dev)
        partial = torch.empty(max(1, mr.partial_elements(m, n, splits)), dtype=torch.float32, device=dev)
        counters = torch.zeros(mr.row_tiles(n, 1, warps), dtype=torch.int32, device=dev)
        bufs = scratch, partial, counters
    return mr.mm_rows(x, w, out, "nvfp4", scales, row_scale, "nvfp4_mma", INPUT_SCALE, True, 1, warps, splits, *bufs)


@FP4
@pytest.mark.parametrize("shape", [(96, 256), (1000, 5120), (130, 2048), (300, 1152)])
def test_the_fp4_mma_multiplies_the_quantized_activations_by_the_stored_values(shape):
    n, k = shape
    stored, values = matrix("nvfp4", n, k, True)
    steps = k // mr.FP4_STEP
    for rows in (1, 7, 16, 33, 64):
        x = activations(rows, k)
        q = nvfp4_values(x, INPUT_SCALE).double()
        exact = q @ values.t().to(dev) / INPUT_SCALE
        magnitude = q.abs() @ values.abs().t().to(dev) / INPUT_SCALE
        for splits in sorted({1, min(3, steps), min(32, steps)}):
            found = fp4_call(x, stored, splits=splits).double()
            assert bool(((found - exact).abs() <= magnitude * 2e-5 + 1e-30).all()), (rows, splits)
        bf16 = call(x, stored, "nvfp4", "nvfp4", True).double()   # the same activations as BF16 values
        assert float((found - bf16).abs().max() / bf16.abs().max()) < 1e-5


@FP4
@pytest.mark.parametrize("splits", [1, 3])
def test_on_the_fp4_mma_a_row_keeps_its_bits_alone_and_among_up_to_64(splits):
    n, k = 1000, 2048
    stored, _ = matrix("nvfp4", n, k, True)
    x = activations(64, k)
    alone = torch.cat([fp4_call(x[r:r + 1].clone(), stored, splits=splits) for r in range(64)])
    for rows in list(range(2, 18)) + [23, 31, 32, 33, 47, 64]:
        for warps in (4, 2, 1):
            assert torch.equal(fp4_call(x[:rows], stored, splits=splits, warps=warps), alone[:rows]), (rows, warps)
        for first in (1, 64 - rows):
            assert torch.equal(fp4_call(x[first:first + rows].clone(), stored, splits=splits),
                               alone[first:first + rows]), (rows, first)


@FP4
def test_the_fp4_mma_captures_with_its_pre_pass_and_refuses_what_it_cannot_read():
    n, k = 512, 4096
    stored, _ = matrix("nvfp4", n, k, True)
    x = activations(9, k)
    scratch = torch.empty(mr.scratch_bytes(9, k, "nvfp4_mma"), dtype=torch.uint8, device=dev)
    partial = torch.empty(mr.partial_elements(9, n, 4), dtype=torch.float32, device=dev)
    counters = torch.zeros(mr.row_tiles(n, 1, 4), dtype=torch.int32, device=dev)
    out = torch.empty(9, n, dtype=torch.bfloat16, device=dev)
    fp4_call(x, stored, splits=4, out=out, bufs=(scratch, partial, counters))
    expected = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fp4_call(x, stored, splits=4, out=out, bufs=(scratch, partial, counters))
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
    assert not counters.any()
    w, scales, row_scale = stored
    with pytest.raises(RuntimeError):   # row-major scales
        mr.mm_rows(x, w, out, "nvfp4", scales, row_scale, "nvfp4_mma", INPUT_SCALE, False, 1, 4, 1, scratch, partial,
                   counters)
    small = matrix("nvfp4", 64, 96, True)[0]
    with pytest.raises((RuntimeError, ValueError)):   # K no multiple of 128
        fp4_call(activations(4, 96), small)
    with pytest.raises(ValueError):
        mr.plan(64, 96, "nvfp4", dev, acts="nvfp4_mma")
    assert mr.plan(34816, 5120, "nvfp4", dev, acts="nvfp4_mma")[0] == 1


# Knobs other than the defaults (one build for every structure). The decode tile's and the FP4 MMA's plan comes from
# them (groups, warps and splits; the split is part of the sums' order, the rest is speed only); HOIST and the prompt
# tile's warps and rows a CTA only change when loads run and which outputs a CTA computes, never an output's chain.
OTHER_KNOBS = {"decode-tile": {"GROUPS": 2, "WARPS": 2, "SPLITS": 3},
               "fp4-tile": {"WARPS": 2, "SPLITS": 4, "HOIST": 4},
               "prompt-tile": {"PW": 4, "PT": 32}}


def test_other_knobs_plan_from_the_tuning_and_keep_the_default_bits():
    other = kbuild.load("mma-rows", knobs=OTHER_KNOBS)
    assert other is not mr
    assert other.plan(1000, 2048, "nvfp4") == (2, 2, 3)
    assert other.plan(1000, 96, "bf16") == (2, 2, 1)   # splits capped at the matrix's steps
    for weights, acts, blocked in FORMATS:
        stored, _ = matrix(weights, 1000, 2048, blocked)
        x = activations(64, 2048)
        w, scales, row_scale = stored
        for rows in (1, 5, 16, 40, 64):
            want = call(x[:rows], stored, weights, acts, blocked, groups=2, warps=2, splits=3)
            got = torch.empty_like(want)
            scratch = torch.empty(mr.scratch_bytes(rows, 2048), dtype=torch.uint8, device=dev)
            partial = torch.empty(mr.partial_elements(rows, 1000, 3), dtype=torch.float32, device=dev)
            counters = torch.zeros(mr.row_tiles(1000, 1, 2), dtype=torch.int32, device=dev)
            other.mm_rows(x[:rows], w, got, weights, scales, row_scale, acts, INPUT_SCALE, blocked, 2, 2, 3, scratch,
                          partial, counters)
            assert torch.equal(got, want), (weights, acts, rows)
            prompt = torch.empty_like(want)
            other.mm_prompt(x[:rows], w, prompt, weights, scales, row_scale, acts, INPUT_SCALE, blocked, 3, scratch)
            assert torch.equal(prompt, want), ("prompt", weights, acts, rows)
    if mr.fp4_mma(dev):
        assert other.plan(1024, 2048, "nvfp4", dev, acts="nvfp4_mma") == (1, 2, 4)
        stored, _ = matrix("nvfp4", 1024, 2048, True)
        x = activations(16, 2048)
        for rows in (1, 9, 16):
            want = fp4_call(x[:rows], stored, splits=4, warps=2)
            w, scales, row_scale = stored
            got = torch.empty_like(want)
            bufs = (torch.empty(mr.scratch_bytes(rows, 2048, "nvfp4_mma"), dtype=torch.uint8, device=dev),
                    torch.empty(mr.partial_elements(rows, 1024, 4), dtype=torch.float32, device=dev),
                    torch.zeros(mr.row_tiles(1024, 1, 2), dtype=torch.int32, device=dev))
            other.mm_rows(x[:rows], w, got, "nvfp4", scales, row_scale, "nvfp4_mma", INPUT_SCALE, True, 1, 2, 4, *bufs)
            assert torch.equal(got, want), ("fp4", rows)


# The staged decode tile (decode-tile's SW and SPF, picked per matrix and row count): one build holding every staged
# configuration, select() running each. Its outputs keep the register tile's bits (the same MMAs in the same order) at
# every row count it takes (1 to 16), for every format and quantization, any groups, warps and splits, K with a partial
# last step and N past the last full tile.
STAGED = [{"SW": 4}, {"SW": 4, "SPF": 1}, {"SW": 8}, {"SW": 8, "SPF": 1}]


@pytest.mark.parametrize("shape", [(1000, 2048), (130, 1056)])
def test_the_staged_tile_keeps_the_register_tiles_bits(shape):
    n, k = shape
    staged = kbuild.load("mma-rows", configs={"decode-tile": STAGED})
    try:
        for weights, acts, blocked in FORMATS:
            stored, _ = matrix(weights, n, k, blocked)
            x = activations(16, k)
            w, scales, row_scale = stored
            steps = math.ceil(k / mr.VALUES[weights])
            for groups, warps, splits in ((1, 4, 1), (2, 1, min(3, steps)), (1, 2, steps)):
                alone = torch.cat([call(x[r:r + 1].clone(), stored, weights, acts, blocked, groups=groups, warps=warps,
                                        splits=splits) for r in range(16)])
                for index in range(1, len(STAGED) + 1):
                    kbuild.select(staged, "decode-tile", index)
                    for rows in range(1, 17):
                        got = torch.empty(rows, n, dtype=torch.float32, device=dev)
                        scratch, partial, counters = buffers(rows, n, k, splits, groups, warps)
                        staged.mm_rows(x[:rows], w, got, weights, scales, row_scale, acts, INPUT_SCALE, blocked, groups,
                                       warps, splits, scratch, partial, counters)
                        assert torch.equal(got, alone[:rows]), (weights, acts, STAGED[index - 1], groups, warps,
                                                                splits, rows)
                        assert not bool(counters.any())
    finally:
        kbuild.select(staged, "decode-tile", -1)
