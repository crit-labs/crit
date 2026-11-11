"""mma-rows' INT4 weights with a zero point a group (int4_zp: AWQ and GPTQ in crit's rows) against an FP64 reference
of their values (code - zero) * scale, with and without splits of K; each row's bits the same alone and among up to
63 others whatever `groups` and `warps`; the staged decode tile (SW, SPF) and the prompt tile (mm_prompt) the register
tile's bits; graph capture. The two-part decode (hi + lo, one MMA each) makes every product exact, so the bound is the
other formats' FP32 accumulation bound."""
import math

import pytest
import torch

from crit_kernels import build as kbuild

mr = kbuild.load("mma-rows")
dev = torch.device("cuda")


def int4_matrix(n, k, group, seed=5):
    """(codes [N][K / 2], FP16 scales, uint8 zeros 0..16 [N][K / group]) and the values in FP64 [N][K]."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)
    scales = ((torch.rand(n, k // group, generator=g) + 0.5) / 64).half()
    zeros = torch.randint(0, 17, (n, k // group), generator=g, dtype=torch.uint8)
    q = torch.stack(((codes & 15).long(), (codes >> 4).long()), -1).reshape(n, k).double()
    values = (q - zeros.double().repeat_interleave(group, 1)) * scales.double().repeat_interleave(group, 1)
    return (codes.to(dev), scales.to(dev), zeros.to(dev)), values


def activations(m, k, seed=8):
    return torch.randn(m, k, generator=torch.Generator().manual_seed(seed)).to(dev, torch.bfloat16)


def call(module, x, stored, group, dtype=torch.float32, groups=1, warps=4, splits=1):
    codes, scales, zeros = stored
    m, n, k = x.shape[0], codes.shape[0], x.shape[1]
    out = torch.empty(m, n, dtype=dtype, device=dev)
    partial = torch.empty(max(1, module.partial_elements(m, n, splits)), dtype=torch.float32, device=dev)
    counters = torch.zeros(module.row_tiles(n, 1, warps), dtype=torch.int32, device=dev)
    return module.mm_rows(x, codes, out, "int4_zp", scales, None, "bf16", 1.0, False, groups, warps, splits, None,
                          partial, counters, zeros, group)


@pytest.mark.parametrize("group", [32, 64, 128])
@pytest.mark.parametrize("shape", [(96, 256), (1000, 5120), (130, 2048), (300, 1152)])
def test_int4_rows_multiply_by_the_exact_values_and_keep_their_bits(shape, group):
    n, k = shape
    if k % group:
        pytest.skip("K a multiple of the group")
    stored, values = int4_matrix(n, k, group)
    steps = math.ceil(k / mr.VALUES["int4_zp"])
    x = activations(64, k)
    exact = x.double().cpu() @ values.t()
    magnitude = x.double().cpu().abs() @ values.abs().t()
    for splits in sorted({1, min(3, steps), mr.plan(n, k, "int4_zp")[2]}):
        full = call(mr, x, stored, group, splits=splits)
        assert bool(((full.double().cpu() - exact).abs() <= magnitude * 2e-5 + 1e-30).all()), splits
        for rows in (1, 2, 5, 8, 9, 16, 33, 63):
            for groups, warps in ((1, 4), (2, 2), (1, 1)):
                if rows > 32 and groups != 1:
                    continue
                part = call(mr, x[:rows].clone(), stored, group, groups=groups, warps=warps, splits=splits)
                assert torch.equal(part, full[:rows]), (splits, rows, groups, warps)
        prompt = torch.empty(64, n, dtype=torch.float32, device=dev)
        mr.mm_prompt(x, stored[0], prompt, "int4_zp", stored[1], None, "bf16", 1.0, False, splits, None, stored[2],
                     group)
        assert torch.equal(prompt, full), splits


def test_the_staged_tile_gives_the_register_tiles_bits():
    staged = kbuild.load("mma-rows", configs={"decode-tile": [{"SW": sw, "SPF": spf} for sw in (4, 8)
                                                              for spf in (0, 1)]})
    n, k, group = 1000, 2048, 128
    stored, _ = int4_matrix(n, k, group, seed=7)
    x = activations(16, k, seed=9)
    for splits in (1, 3):
        want = call(mr, x, stored, group, splits=splits)
        for index in range(1, 5):
            kbuild.select(staged, "decode-tile", index)
            for rows in (1, 4, 9, 16):
                got = call(staged, x[:rows].clone(), stored, group, splits=splits)
                assert torch.equal(got, want[:rows]), (index, rows, splits)
    kbuild.select(staged, "decode-tile", -1)


def test_int4_calls_capture_and_refuse_what_they_cannot_read():
    n, k, group = 512, 1024, 128
    stored, _ = int4_matrix(n, k, group, seed=3)
    x = activations(4, k)
    want = call(mr, x, stored, group, dtype=torch.bfloat16, splits=2)
    out = torch.empty(4, n, dtype=torch.bfloat16, device=dev)
    partial = torch.empty(mr.partial_elements(4, n, 2), dtype=torch.float32, device=dev)
    counters = torch.zeros(mr.row_tiles(n, 1, 4), dtype=torch.int32, device=dev)
    args = (x, stored[0], out, "int4_zp", stored[1], None, "bf16", 1.0, False, 1, 4, 2, None, partial, counters,
            stored[2], group)
    mr.mm_rows(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mr.mm_rows(*args)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, want)
    for bad in (dict(group=96), dict(group=0), dict(zeros=None)):
        kw = {"zeros": stored[2], "group": group, **bad}
        with pytest.raises(RuntimeError):
            mr.mm_rows(x, stored[0], out, "int4_zp", stored[1], None, "bf16", 1.0, False, 1, 4, 1, None, None, None,
                       kw["zeros"], kw["group"])
