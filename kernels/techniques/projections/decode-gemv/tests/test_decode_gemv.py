"""decode-gemv against an FP64 reference of the stored values: the CUDA and the Triton GEMV within FP32 accumulation,
each row's bits the same alone and beside up to 7 others, both E2M1 decodes the same bits, the decoders one FP32
product rounded once, and the CUDA GEMVs captured in a CUDA graph; INT4 with a zero point (AWQ and GPTQ tensors
repacked, GPTQ's act order) within FP32 accumulation, its rows the same bits at every row count; the format-generic
Triton GEMV (gemv_any) on every format and both multiply paths within FP32 accumulation, its rows the same bits at
every row count 1 to 16, an alias's tuning applying to it; the FP8 dot GEMV within FP32 accumulation,
its rows the same bits at every row count 1 to 16 and at offsets, with and without parts of K, and captured; a build
with other knobs than the defaults gives the default build's bits wherever its knobs keep the order.

The kernel is built for this GPU by crit_kernels.build (default knobs; the cache is CRIT_KERNEL_CACHE)."""
import os
import shutil
import sys
import tempfile

import pytest
import torch

from crit_kernels import build as kbuild

nr = kbuild.load("decode-gemv")
dev = torch.device("cuda")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    dtype=torch.float64)


def matrix(n, k, seed=5):
    """Random codes, E4M3 block scales and inverse global scales [N]."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)
    scales = (torch.rand(n, k // 16, generator=g) * 4 + 0.01).to(torch.float8_e4m3fn)
    inverse = (torch.rand(n, generator=g) + 0.5) / 64
    return codes.to(dev), scales.to(dev), inverse.to(dev)


def values(codes, scales, inverse):
    """The stored values in FP64: e2m1 * e4m3 * inverse [N][K]."""
    c = codes.cpu().long()
    v = torch.stack((E2M1[c & 15], E2M1[c >> 4]), dim=-1).reshape(c.shape[0], -1)
    v = v.view(c.shape[0], -1, 16) * scales.cpu().double()[..., None]
    return v.reshape(c.shape[0], -1) * inverse.cpu().double()[:, None]


def activations(m, k, dtype=torch.bfloat16, seed=8):
    return torch.randn(m, k, generator=torch.Generator().manual_seed(seed)).to(dev, dtype)


def nvfp4_gemv(x, c, s, i, dtype=torch.float32, path="cuda"):
    out = torch.empty(x.shape[0], c.shape[0], dtype=dtype, device=dev)
    (nr.gemv if path == "cuda" else nr.gemv_triton)(x, c, s, i, out)
    return out


@pytest.mark.parametrize("path", ["cuda", "triton"])
@pytest.mark.parametrize("shape", [(96, 256), (1000, 5120), (130, 2048)])
def test_the_gemv_multiplies_by_the_stored_values_and_keeps_rows(path, shape):
    n, k = shape
    c, s, i = matrix(n, k)
    most = nr.CUDA_ROWS if path == "cuda" else nr.TRITON_ROWS
    x = activations(most, k)
    exact = x.double() @ values(c, s, i).t().to(dev)
    batch = nvfp4_gemv(x, c, s, i, path=path)
    magnitude = (x.double().abs() @ values(c, s, i).abs().t().to(dev))
    assert bool(((batch.double() - exact).abs() <= magnitude * 2e-5 + 1e-30).all())
    for rows in range(1, most + 1):
        part = nvfp4_gemv(x[:rows], c, s, i, path=path)
        for r in range(rows):
            assert torch.equal(part[r], nvfp4_gemv(x[r:r + 1].clone(), c, s, i, path=path)[0]), (rows, r)
    assert torch.equal(nvfp4_gemv(x, c, s, i, path=path), batch)   # a repeat


def test_16_bit_outputs_and_fp16_activations():
    c, s, i = matrix(96, 256)
    x = activations(3, 256)
    full = nvfp4_gemv(x, c, s, i)
    assert torch.equal(nvfp4_gemv(x, c, s, i, torch.bfloat16), full.to(torch.bfloat16))
    half = nvfp4_gemv(x.half(), c, s, i, torch.float16)
    assert half.dtype == torch.float16
    exact = x.half().double() @ values(c, s, i).t().to(dev)
    assert float((half.double() - exact).abs().max() / exact.abs().max()) < 1e-3


def test_the_cuda_gemv_refuses_what_it_cannot_read():
    c, s, i = matrix(96, 256)
    x = activations(9, 256)
    with pytest.raises(RuntimeError):
        nvfp4_gemv(activations(17, 256), c, s, i)                  # 17 rows
    with pytest.raises(RuntimeError):
        nvfp4_gemv(activations(2, 260)[:, 4:], c, s, i)          # rows not on 16-byte boundaries
    with pytest.raises(RuntimeError):
        nvfp4_gemv(x[:2].float(), c, s, i)                        # FP32 activations


def test_the_bf16_gemv_multiplies_and_keeps_rows():
    g = torch.Generator().manual_seed(3)
    w = (torch.randn(1000, 2048, generator=g) * 0.05).to(dev, torch.bfloat16)
    x = activations(16, 2048)
    exact = x.double() @ w.double().t()
    out = torch.empty(16, 1000, device=dev)
    nr.gemv_bf16(x, w, out)
    assert float((out.double() - exact).abs().max() / exact.abs().max()) < 1e-5
    for rows in range(1, 17):
        part = torch.empty(rows, 1000, device=dev)
        nr.gemv_bf16(x[:rows], w, part)
        assert torch.equal(part, out[:rows]), rows


def test_the_triton_gemv_gives_the_same_bits_with_either_e2m1_decode():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("the hardware E2M1 conversion needs compute capability 10 or later")
    switch = sys.modules[nr.gemv_triton.__module__]._HW
    c, s, i = matrix(96, 256)
    x = activations(8, 256)
    found = {}
    try:
        for hardware in (True, False):
            switch[dev] = hardware
            found[hardware] = [nvfp4_gemv(x[:rows], c, s, i, path="triton") for rows in (1, 3, 8)]
    finally:
        switch.pop(dev, None)
    for a, b in zip(found[True], found[False]):
        assert torch.equal(a, b)


def test_cuda_and_triton_agree_within_fp32_rounding():
    c, s, i = matrix(1000, 5120)
    x = activations(8, 5120)
    a, b = nvfp4_gemv(x, c, s, i), nvfp4_gemv(x, c, s, i, path="triton")
    assert float((a - b).abs().max() / b.abs().max()) < 1e-5


def test_the_decoders_round_one_fp32_product_once():
    c, s, i = matrix(96, 256)
    product = (values(c, s, torch.ones_like(i)).float().to(dev) * i[:, None])   # FP32: e2m1 * scale exact, * inverse
    for dtype in (torch.bfloat16, torch.float16):
        out = torch.empty(96, 256, dtype=dtype, device=dev)
        assert torch.equal(nr.decode(c, s, i, out), product.to(dtype)), dtype
    factor = i * 64
    out = torch.empty(96, 256, dtype=torch.float8_e4m3fn, device=dev)
    want = (values(c, s, torch.ones_like(i)).float().to(dev) * factor[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(nr.decode_fp8(c, s, factor, out).view(torch.uint8), want.view(torch.uint8))


def test_the_gemvs_capture_in_a_cuda_graph():
    c, s, i = matrix(1000, 2048)
    w = (torch.randn(1000, 2048, generator=torch.Generator().manual_seed(1)) * 0.05).to(dev, torch.bfloat16)
    x = activations(4, 2048)
    out, out16 = torch.empty(4, 1000, device=dev), torch.empty(4, 1000, device=dev)
    eager = nvfp4_gemv(x, c, s, i)
    nr.gemv_bf16(x, w, out16)
    eager16 = out16.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        nr.gemv(x, c, s, i, out)
        nr.gemv_bf16(x, w, out16)
    out.zero_()
    out16.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager) and torch.equal(out16, eager16)


def test_the_cuda_gemv_gives_the_same_bits_with_either_e2m1_decode():
    """The build's code decodes by the hardware conversion on sm_100a and later; a build of the same source for the
    plain target (sm_120 for sm_120a) decodes by the bit pattern: both must give the same bits. Builds that second
    variant with torch's extension builder (skipped without nvcc)."""
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip("the hardware E2M1 conversion needs compute capability 10 or later")
    if not nr.hardware_decode():
        pytest.skip("this build decodes by the bit pattern already")
    if shutil.which("nvcc") is None and not os.environ.get("CUDA_HOME"):
        pytest.skip("no CUDA compiler for the bit-pattern variant")
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _, manifest = kbuild.find("decode-gemv")
    include = tempfile.mkdtemp(prefix="decode_gemv_plain_")   # the default knobs' crit_tuning.h, as the build writes it
    with open(os.path.join(include, "crit_tuning.h"), "w") as handle:
        handle.write(kbuild.tuning_header(manifest, kbuild.default_tuning(manifest)))
    before = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        plain = load(name=f"decode_gemv_plain_{major}{minor}",
                     sources=[os.path.join(here, "tests", "plain_variant.cu"),
                              os.path.join(here, "decode_gemv_cuda", "decode_gemv.cu")],
                     extra_include_paths=[os.path.join(here, "decode_gemv_cuda"), include,
                                          str(kbuild.core_folder() / "include")], extra_cuda_cflags=["-O3"])
    finally:
        if before is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = before
    assert plain.hardware_decode() == 0
    c, s, i = matrix(96, 256)
    x = activations(8, 256)
    for rows in (1, 3, 8):
        out = torch.empty(rows, 96, device=dev)
        plain.gemv(x[:rows], c, s, i, out)
        assert torch.equal(out, nvfp4_gemv(x[:rows], c, s, i)), rows


def fp8_matrix(n, k, mx, seed=6):
    """E4M3 codes (no NaN) and FP32 row scales, or E8M0 exponents per 32 inputs (MXFP8)."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 0x7E, (n, k), generator=g, dtype=torch.uint8)
    codes = codes | (torch.randint(0, 2, (n, k), generator=g, dtype=torch.uint8) << 7)
    if mx:
        scales = torch.randint(118, 132, (n, k // 32), generator=g, dtype=torch.uint8)
    else:
        scales = torch.rand(n, generator=g) / 100 + 1e-3
    return codes.to(dev), scales.to(dev)


def fp8_values(codes, scales):
    v = codes.cpu().view(torch.float8_e4m3fn).double()
    if scales.dtype == torch.uint8:
        return v * torch.exp2(scales.cpu().double() - 127).repeat_interleave(32, dim=1)
    return v * scales.cpu().double()[:, None]


@pytest.mark.parametrize("mx", [False, True])
@pytest.mark.parametrize("shape", [(96, 256), (1024, 5120), (130, 2048)])
def test_the_fp8_gemv_multiplies_by_the_stored_values_and_keeps_rows(mx, shape):
    n, k = shape
    c, s = fp8_matrix(n, k, mx)
    x = activations(16, k)
    values = fp8_values(c, s).to(dev)
    exact = x.double() @ values.t()
    magnitude = x.double().abs() @ values.abs().t()
    out = torch.empty(16, n, device=dev)
    nr.gemv_fp8(x, c, s, out)
    assert bool(((out.double() - exact).abs() <= magnitude * 2e-5 + 1e-30).all())
    for rows in range(1, 17):
        part = torch.empty(rows, n, device=dev)
        nr.gemv_fp8(x[:rows], c, s, part)
        assert torch.equal(part, out[:rows]), rows
        alone = torch.empty(1, n, device=dev)
        nr.gemv_fp8(x[rows - 1:rows].clone(), c, s, alone)
        assert torch.equal(alone[0], out[rows - 1]), rows
    small = (x[:3] / 64).half()   # FP16 out: keep the sums inside FP16's range
    half = torch.empty(3, n, dtype=torch.float16, device=dev)
    nr.gemv_fp8(small, c.view(torch.float8_e4m3fn), s, half)
    want = small.double() @ values.t()
    assert float((half.double() - want).abs().max() / want.abs().max()) < 2e-3


def test_the_fp8_gemv_refuses_what_it_cannot_read():
    c, s = fp8_matrix(96, 256, False)
    out = torch.empty(2, 96, device=dev)
    with pytest.raises(RuntimeError):
        nr.gemv_fp8(activations(2, 256), c, s[:10], out)          # a scale per row
    cm, sm = fp8_matrix(96, 256, True)
    with pytest.raises(RuntimeError):
        nr.gemv_fp8(activations(2, 256), cm, sm[:, :4], out)      # an exponent per 32 inputs
    with pytest.raises(RuntimeError):
        nr.gemv_fp8(activations(2, 256).float(), c, s, out)      # 16-bit activations


def test_every_triton_tile_gives_the_same_bits():
    """The autotuner picks a tile per shape by timing it: every tile it may pick gives the same bits, so two processes
    agree whatever they picked."""
    module = sys.modules[nr.gemv_triton.__module__]
    c, s, i = matrix(130, 2048)
    x = activations(8, 2048)
    found = []
    for config in module.GEMV_TILES:
        out = torch.empty(8, 130, device=dev)
        module.fma(0, x, c, s.view(torch.uint8), None, i, out, 0, config.kwargs["BLOCK_N"], config.kwargs["BLOCK_K"],
                   config.num_warps, 3)
        found.append(out)
    for out in found[1:]:
        assert torch.equal(out, found[0])


def triton_cdiv(a, b):
    return (a + b - 1) // b


def to_blocked(scales):
    """[rows][blocks] scale bytes in cuBLASLt's 128 x 4 tiled layout (rows padded to 128, blocks to 4)."""
    rows, cols = scales.shape
    rt, ct = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(rt * 128, ct * 4, dtype=scales.dtype, device=scales.device)
    padded[:rows, :cols] = scales
    tiles = padded.view(rt, 128, ct, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


@pytest.mark.parametrize("shape", [(96, 256), (130, 2080), (1000, 5120)])
def test_blocked_scales_give_the_row_major_bits(shape):
    n, k = shape
    c, s, i = matrix(n, k)
    x = activations(16, k)
    blocked = to_blocked(s.view(torch.uint8))
    for rows in (1, 5, 16):
        want = nvfp4_gemv(x[:rows], c, s, i)
        found = torch.empty_like(want)
        nr.gemv(x[:rows], c, blocked, i, found, blocked=True)
        assert torch.equal(found, want), (shape, rows)
    with pytest.raises(RuntimeError):
        nr.gemv(x[:2], c, blocked[:-4], i, torch.empty(2, n, device=dev), blocked=True)


def test_tensors_on_another_device_or_misaligned_are_refused():
    c, s, i = matrix(96, 256)
    x = activations(2, 256)
    out = torch.empty(2, 96, device=dev)
    with pytest.raises(RuntimeError):
        nr.gemv(x, c, s, i.cpu(), out)
    odd = torch.empty(96 * 16 + 1, dtype=torch.uint8, device=dev)[1:].view(96, 16)
    odd.copy_(s.view(torch.uint8))
    with pytest.raises(RuntimeError):
        nr.gemv(x, c, odd, i, out)                       # scales read two bytes at a time
    fc, fs = fp8_matrix(96, 256, False)
    with pytest.raises(RuntimeError):
        nr.gemv_fp8(x, fc, fs.cpu(), out)


def fp8_dot_matrix(n, k, seed=6):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 0x7E, (n, k), generator=g, dtype=torch.uint8)
    codes |= torch.randint(0, 2, (n, k), generator=g, dtype=torch.uint8) << 7
    row_scale = torch.rand(n, generator=g) / 64 + 1e-3
    return codes.to(dev).view(torch.float8_e4m3fn), row_scale.to(dev)


@pytest.mark.parametrize("shape,plan", [((300, 1024), None), ((512, 4096), (32, 256, 4, 4)), ((96, 640), None),
                                        ((2048, 5120), (64, 512, 8, 2))])
def test_the_fp8_dot_gemv_multiplies_by_the_stored_values_and_keeps_rows(shape, plan):
    n, k = shape
    plan = plan or nr.fp8_dot_tile(n, k, torch.cuda.get_device_properties(dev).multi_processor_count)
    w, s = fp8_dot_matrix(n, k)
    partial = torch.empty(nr.dot_partial_elements(16, n, plan[3]) or 1, dtype=torch.float32, device=dev)
    x = activations(16, k)
    exact = x.double() @ (w.float().double() * s.double()[:, None]).t()
    for out_dtype in (torch.float32, torch.bfloat16):
        whole = nr.gemv_fp8_dot(x, w, s, torch.empty(16, n, dtype=out_dtype, device=dev), plan, partial)
        bound = 2e-5 if out_dtype == torch.float32 else 1e-2
        assert float((whole.double() - exact).abs().max() / exact.abs().max()) < bound
        for m in range(1, 16):
            for first in (0, 16 - m):
                part = nr.gemv_fp8_dot(x[first:first + m].clone(), w, s, torch.empty(m, n, dtype=out_dtype, device=dev),
                                       plan, partial)
                assert torch.equal(part, whole[first:first + m]), (m, first, out_dtype)


def test_the_fp8_dot_gemv_tiles_follow_the_shape_and_refuse_what_they_cannot_read():
    for n, k in ((16384, 5120), (5120, 6144), (14336, 5120), (248320, 5120), (2560, 640), (1280, 2560)):
        bn, bk, warps, split = nr.fp8_dot_tile(n, k, 170)
        assert bn in (32, 64) and k % (bk * split) == 0 and warps in (4, 8)
    with pytest.raises(ValueError):
        nr.fp8_dot_tile(512, 1000, 170)
    w, s = fp8_dot_matrix(256, 1024)
    with pytest.raises(ValueError):
        nr.gemv_fp8_dot(activations(17, 1024), w, s, torch.empty(17, 256, device=dev), (32, 256, 4, 1))
    with pytest.raises(ValueError):
        nr.gemv_fp8_dot(activations(4, 1024), w, s, torch.empty(4, 256, device=dev), (32, 256, 4, 2))


def test_the_fp8_dot_gemv_captures_in_a_cuda_graph():
    w, s = fp8_dot_matrix(512, 4096)
    plan = (32, 256, 4, 4)
    partial = torch.empty(nr.dot_partial_elements(16, 512, 4), dtype=torch.float32, device=dev)
    x = activations(9, 4096)
    out = torch.empty(9, 512, dtype=torch.float32, device=dev)
    nr.gemv_fp8_dot(x, w, s, out, plan, partial)
    expected = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        nr.gemv_fp8_dot(x, w, s, out, plan, partial)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)



# INT4 with a zero point a group (AWQ, GPTQ): checkpoint tensors packed as those formats pack them, repacked into crit's
# rows by repack_awq / repack_gptq, multiplied by gemv_int4.
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]   # nibble p of an AWQ word holds element AWQ_ORDER[p] (along N)


def int4_checkpoint(n, k, group, seed=11, act_order=False):
    """Codes [K][N] 0..15, zero points [K / group][N] 1..15 (a nibble each way: AWQ's and GPTQ's own zero, and GPTQ's
    zero - 1; the kernels' zero 16 is the bench's and the core tests'), FP16 scales, g_idx [K]."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(0, 16, (k, n), generator=g)
    z = torch.randint(1, 16, (k // group, n), generator=g)
    s = ((torch.rand(k // group, n, generator=g) + 0.5) / 64).half()
    g_idx = torch.arange(k) // group
    if act_order:
        g_idx = g_idx[torch.randperm(k, generator=g)]
    return q, z, s, g_idx


def pack_words(v, order):
    """[..., 8 j + e] values 0..15 -> int32 [..., j], nibble p holding element order[p]."""
    v = v.reshape(*v.shape[:-1], -1, 8).long()
    word = torch.zeros(v.shape[:-1], dtype=torch.int64)
    for p in range(8):
        word |= (v[..., order[p]] & 15) << (4 * p)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


def int4_reference(q, z, s, g_idx):
    """[N][K] FP64: (code - zero[g_idx[k]]) * scale[g_idx[k]]."""
    return ((q.double() - z[g_idx].double()) * s[g_idx].double()).t()


@pytest.mark.parametrize("group", [32, 128])
@pytest.mark.parametrize("shape", [(96, 256), (1024, 5120), (136, 2048)])
def test_the_int4_gemv_multiplies_awq_and_gptq_values_and_keeps_rows(shape, group):
    n, k = shape
    if k % group:
        pytest.skip("K a multiple of the group")
    q, z, s, g_idx = int4_checkpoint(n, k, group)
    exact_w = int4_reference(q, z, s, g_idx).to(dev)
    x = activations(16, k)
    exact = x.double() @ exact_w.t()
    magnitude = x.double().abs() @ exact_w.abs().t()
    awq = nr.repack_awq(pack_words(q, AWQ_ORDER).to(dev), pack_words(z, AWQ_ORDER).to(dev), s.to(dev), group)
    seq = list(range(8))
    gptq = nr.repack_gptq(pack_words(q.t(), seq).t().contiguous().to(dev), pack_words(z - 1, seq).to(dev), s.to(dev),
                          group, g_idx=g_idx)
    assert gptq[3] is None
    for packed in (awq, gptq[:3]):
        assert torch.equal(packed[0], awq[0]) and torch.equal(packed[1], awq[1]) and torch.equal(packed[2], awq[2])
    codes, scales, zeros = awq
    out = torch.empty(16, n, device=dev)
    nr.gemv_int4(x, codes, scales, zeros, out, group)
    assert bool(((out.double() - exact).abs() <= magnitude * 2e-5 + 1e-30).all())
    for rows in range(1, 17):
        part = torch.empty(rows, n, device=dev)
        nr.gemv_int4(x[:rows], codes, scales, zeros, part, group)
        assert torch.equal(part, out[:rows]), rows
        alone = torch.empty(1, n, device=dev)
        nr.gemv_int4(x[rows - 1:rows].clone(), codes, scales, zeros, alone, group)
        assert torch.equal(alone[0], out[rows - 1]), rows
    half = torch.empty(3, n, dtype=torch.float16, device=dev)
    nr.gemv_int4((x[:3] / 8).half(), codes, scales, zeros, half, group)
    want = (x[:3] / 8).half().double() @ exact_w.t()
    assert float((half.double() - want).abs().max() / want.abs().max()) < 2e-3


def test_gptq_act_order_sorts_the_inputs_by_group():
    n, k, group = 256, 1024, 128
    q, z, s, g_idx = int4_checkpoint(n, k, group, act_order=True)
    exact_w = int4_reference(q, z, s, g_idx).to(dev)
    seq = list(range(8))
    for offset in (1, 0):
        codes, scales, zeros, perm = nr.repack_gptq(pack_words(q.t(), seq).t().contiguous().to(dev),
                                                    pack_words(z - offset, seq).to(dev), s.to(dev), group,
                                                    g_idx=g_idx, zero_offset=offset)
        assert perm is not None and torch.equal(g_idx.to(dev)[perm], torch.arange(k, device=dev) // group)
        x = activations(5, k)
        exact = x.double() @ exact_w.t()
        magnitude = x.double().abs() @ exact_w.abs().t()
        out = torch.empty(5, n, device=dev)
        nr.gemv_int4(x[:, perm].contiguous(), codes, scales, zeros, out, group)
        assert bool(((out.double() - exact).abs() <= magnitude * 2e-5 + 1e-30).all()), offset
    with pytest.raises(ValueError):
        nr.repack_gptq(pack_words(q.t(), seq).t().contiguous().to(dev), pack_words(z - 1, seq).to(dev), s.to(dev),
                       group, g_idx=torch.zeros(k, dtype=torch.int64))


def test_the_int4_gemv_refuses_what_it_cannot_read():
    codes = torch.zeros(64, 128, dtype=torch.uint8, device=dev)
    scales = torch.ones(64, 2, dtype=torch.float16, device=dev)
    zeros = torch.zeros(64, 2, dtype=torch.uint8, device=dev)
    x = activations(2, 256)
    out = torch.empty(2, 64, device=dev)
    for bad in (lambda: nr.gemv_int4(x, codes, scales, zeros, out, 64),       # scales do not cover groups of 64
                lambda: nr.gemv_int4(x, codes, scales, zeros, out, 96),       # not a multiple of 32 dividing K
                lambda: nr.gemv_int4(x, codes, scales.float(), zeros, out, 128)):
        with pytest.raises(RuntimeError):
            bad()

# Knobs other than the defaults (one build for every structure): the CUDA GEMVs' and the Triton GEMV's WARPS, ROWS,
# PREF, STREAM, BLOCK_N and STAGES only choose which rows a block or program takes and how its loads run, never a
# row's order, so their bits are the default build's; the dot GEMV's plan comes from the tuning.
OTHER_KNOBS = {"gemv-cuda": {"WARPS": 2, "ROWS": 4, "PREF": 1, "STREAM": 1},
               "gemv-nvfp4": {"WARPS": 8, "ROWS": 1, "PREF": 2, "STREAM": 1},
               "gemv-fp8": {"WARPS": 2, "ROWS": 8, "PREF": 1, "STREAM": 1},
               "gemv-bf16": {"WARPS": 8, "ROWS": 1, "PREF": 2, "STREAM": 1},
               "gemv-nvfp4-triton": {"BLOCK_N": 16, "WARPS": 8, "STAGES": 2},
               "gemv-fp8-dot": {"BN": 16, "BK": 128, "WARPS": 2, "SPLIT": 2}}


def test_other_knobs_give_the_default_bits_where_they_keep_the_order():
    other = kbuild.load("decode-gemv", knobs=OTHER_KNOBS)
    assert other is not nr
    c, s, i = matrix(1000, 2080)
    fc, fs = fp8_matrix(1000, 2048, False)
    mc, ms = fp8_matrix(1000, 2048, True)
    w = (torch.randn(1000, 2048, generator=torch.Generator().manual_seed(2)) * 0.05).to(dev, torch.bfloat16)
    q, z, sc, g_idx = int4_checkpoint(1000, 2048, 128)
    i4 = nr.repack_awq(pack_words(q, AWQ_ORDER).to(dev), pack_words(z, AWQ_ORDER).to(dev), sc.to(dev), 128)
    for rows in (1, 3, 7, 16):
        x = activations(rows, 2048)
        xn = activations(rows, 2080)
        for out_dtype in (torch.float32, torch.bfloat16):
            pair = [torch.empty(rows, 1000, dtype=out_dtype, device=dev) for _ in range(2)]
            for module, out in zip((nr, other), pair):
                module.gemv(xn, c, s, i, out)
            assert torch.equal(*pair), ("nvfp4", rows, out_dtype)
            for codes, scales in ((fc, fs), (mc, ms)):
                for module, out in zip((nr, other), pair):
                    module.gemv_fp8(x, codes, scales, out)
                assert torch.equal(*pair), ("fp8", scales.dtype, rows, out_dtype)
            for module, out in zip((nr, other), pair):
                module.gemv_bf16(x, w, out)
            assert torch.equal(*pair), ("bf16", rows, out_dtype)
            for module, out in zip((nr, other), pair):   # INT4: gemv-cuda's own knobs (no alias)
                module.gemv_int4(x, i4[0], i4[1], i4[2], out, 128)
            assert torch.equal(*pair), ("int4", rows, out_dtype)
            if rows <= nr.TRITON_ROWS:
                for module, out in zip((nr, other), pair):
                    module.gemv_triton(xn, c, s, i, out)
                assert torch.equal(*pair), ("triton", rows, out_dtype)
    assert other.fp8_dot_tile(1000, 2048, 170) == (16, 128, 2, 2)
    assert other.fp8_dot_tile(1000, 640, 170) == nr.fp8_dot_tile(1000, 640, 170)   # split * BK does not divide K
    w8, s8 = fp8_dot_matrix(1000, 2048)
    x = activations(16, 2048)
    exact = x.double() @ (w8.float().double() * s8.double()[:, None]).t()
    partial = torch.empty(other.dot_partial_elements(16, 1000, 2), dtype=torch.float32, device=dev)
    whole = other.gemv_fp8_dot(x, w8, s8, torch.empty(16, 1000, device=dev), other.fp8_dot_tile(1000, 2048, 170),
                               partial)
    assert float((whole.double() - exact).abs().max() / exact.abs().max()) < 2e-5


# ---- gemv_any: the format-generic Triton GEMV (structure gemv-triton), every format on both multiply paths ----

any_module = sys.modules[nr.gemv_any.__module__]


def any_matrix(weights, n, k, seed=21):
    """(kwargs for gemv_any, the FP64 values [N][K]) of a random matrix in format `weights` (INT4: group 64, 32 where
    64 does not divide K)."""
    group = 64 if k % 64 == 0 else 32
    if weights == 0:
        c, s, i = matrix(n, k, seed)
        return {"codes": c, "scales": s, "factor": i}, values(c, s, i)
    if weights in (1, 2):
        c, s = fp8_matrix(n, k, weights == 2, seed)
        kw = {"codes": c, "scales": s} if weights == 2 else {"codes": c, "factor": s}
        return kw, fp8_values(c, s)
    g = torch.Generator().manual_seed(seed)
    if weights == 3:
        w = (torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16)
        return {"codes": w.to(dev)}, w.double()
    codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)
    scales = ((torch.rand(n, k // group, generator=g) + 0.5) / 64).half()
    zeros = torch.randint(0, 17, (n, k // group), generator=g, dtype=torch.uint8)
    q = torch.stack((codes & 15, codes >> 4), -1).reshape(n, k).double()
    v = (q - zeros.double().repeat_interleave(group, 1)) * scales.double().repeat_interleave(group, 1)
    return {"codes": codes.to(dev), "scales": scales.to(dev), "zeros": zeros.to(dev), "group": group}, v


def any_path(weights, x, kw, out, path):
    """gemv_any's arithmetic on one path: "rule" (gemv_any as built), "fma" (the autotuner's tile) or a dot plan."""
    if path == "rule":
        return nr.gemv_any(x, weights, out=out, **kw)
    codes = kw["codes"] if weights == 3 else kw["codes"].view(torch.uint8)
    scales = kw.get("scales")
    if scales is not None:
        scales = scales.view(torch.int16) if weights == 4 else scales.view(torch.uint8)
    args = (weights, x, codes, scales, kw.get("zeros"), kw.get("factor"), out, kw.get("group", 0))
    if path == "fma":
        return any_module.fma(*args, 0, 512, 4, 3)
    return any_module.dot(*args, path)


@pytest.mark.parametrize("weights", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("shape,path", [((300, 1024), "rule"), ((300, 1024), "fma"), ((300, 1024), (32, 256, 4, 2)),
                                        ((130, 2080), "rule"), ((1000, 5120), (64, 512, 8, 1)),
                                        ((1000, 5120), (16, 128, 4, 4))])
def test_the_triton_gemv_takes_every_format_on_both_paths_and_keeps_rows(weights, shape, path):
    n, k = shape
    kw, v = any_matrix(weights, n, k)
    for xd in (torch.bfloat16, torch.float16):
        if weights == 3 and xd == torch.float16:
            continue   # BF16 weights take BF16 activations
        x = activations(16, k, xd)
        exact = x.double() @ v.t().to(dev)
        magnitude = x.double().abs() @ v.abs().t().to(dev)
        whole = any_path(weights, x, kw, torch.empty(16, n, device=dev), path)
        assert bool(((whole.double() - exact).abs() <= magnitude * 1e-5).all()), (weights, shape, path, xd)
        for rows in range(1, 17):
            for first in (0, 16 - rows):
                part = any_path(weights, x[first:first + rows].clone(), kw, torch.empty(rows, n, device=dev), path)
                assert torch.equal(part, whole[first:first + rows]), (weights, shape, path, xd, rows, first)


def test_the_triton_gemv_plans_from_the_shape_and_an_alias_tuning_still_applies():
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    assert nr.triton_plan(1, 16384, 5120, 1) == ("dot", nr.fp8_dot_tile(16384, 5120, sms))
    assert nr.triton_plan(0, 1000, 2080, 4)[0] == "fma"          # K not a multiple of 128
    assert nr.triton_plan(2, 1000, 2048, 4)[0] == "fma"          # MXFP8: fma by rule
    assert nr.triton_plan(3, 1000, 2048, 4, torch.float16)[0] == "fma"
    for rows in (1, 4, 16):                                       # the order knobs: one plan a matrix
        assert nr.triton_plan(0, 5000, 4096, rows) == nr.triton_plan(0, 5000, 4096, 1)
    tuning = any_module._crit_tuning.TUNING
    saved = {sid: tuning[sid] for sid in ("gemv-nvfp4-triton", "gemv-fp8-dot")}
    try:
        old = dict(saved["gemv-nvfp4-triton"]["configs"][0], BLOCK_N=16, WARPS=8, STAGES=2)
        tuning["gemv-nvfp4-triton"] = {"configs": [saved["gemv-nvfp4-triton"]["configs"][0], old],
                                       "table": [{"key": {"N": 5000, "K": 4096}, "rows": [1, 8], "config": 1}]}
        assert nr.triton_plan(0, 5000, 4096, 4) == ("fma", 16, 512, 8, 2)
        # The tuning tunes the alias and not gemv-triton: the alias's defaults for its other shapes (the autotuner)
        assert nr.triton_plan(0, 5001, 4096, 4) == ("fma", 0, 512, 4, 3)
        own = tuning["gemv-triton"]
        tuning["gemv-triton"] = {"configs": own["configs"], "table": [{"key": None, "rows": [1, 16], "config": 0}]}
        try:   # once gemv-triton is tuned too, the alias wins only where it names a configuration of its own
            assert nr.triton_plan(0, 5000, 4096, 4) == ("fma", 16, 512, 8, 2)
            assert nr.triton_plan(0, 5001, 4096, 4)[0] == "dot"
        finally:
            tuning["gemv-triton"] = own
        dot = dict(saved["gemv-fp8-dot"]["configs"][0], BN=16, BK=128, WARPS=2, SPLIT=2)
        tuning["gemv-fp8-dot"] = {"configs": [saved["gemv-fp8-dot"]["configs"][0], dot],
                                  "table": [{"key": {"N": 5000, "K": 4096}, "rows": [1, 16], "config": 1}]}
        assert nr.triton_plan(1, 5000, 4096, 9) == ("dot", (16, 128, 2, 2))
    finally:
        tuning.update(saved)


def test_the_triton_gemv_captures_in_a_cuda_graph():
    kw, _ = any_matrix(4, 512, 4096)
    x = activations(9, 4096)
    out = torch.empty(9, 512, device=dev)
    partial = torch.empty(nr.dot_partial_elements(16, 512, 8), dtype=torch.float32, device=dev)
    nr.gemv_any(x, 4, out=out, partial=partial, **kw)
    expected = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        nr.gemv_any(x, 4, out=out, partial=partial, **kw)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
