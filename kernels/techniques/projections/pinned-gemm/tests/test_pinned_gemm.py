"""pinned-gemm: every format's pinned algorithm gives a row the same bits at every row count from 1 to 4096 and at
other offsets, values within FP32 accumulation of the FP64 product of the dequantized operands, a column block of a
wider output, a captured call, the same choice in every pin, and the same row exactness under a non-default knob
setting. Formats this device's cuBLASLt lists no split-K-free algorithm for are skipped (FP8 row scales on sm_120 with
cuBLASLt 13.1). The kernel is built for this GPU with crit_kernels.build.load (its default knobs, or the ones a test
names)."""
import pytest
import torch

from crit_kernels.build import load, select

pg = load("pinned-gemm")
dev = torch.device("cuda")
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
MAJOR = torch.cuda.get_device_capability()[0]
FORMATS = ["bf16:bf16", "bf16:fp32", "fp16:fp32"] + (["fp8e4m3-row:bf16", "fp8e4m3:fp32", "fp8e4m3-tensor:bf16"]
                                                      if torch.cuda.get_device_capability() >= (8, 9) else []) + \
    (["nvfp4:bf16", "nvfp4:fp32", "mxfp8:bf16"] if MAJOR >= 10 else [])


def dequantized(fmt, x, scale, side):
    f = pg.parse(fmt)
    if f[side] == "e2m1":
        table = torch.tensor(E2M1, dtype=torch.float64, device=x.device)
        v = torch.stack((table[(x & 15).long()], table[(x >> 4).long()]), dim=-1).reshape(x.shape[0], -1)
    else:
        v = x.double()
    if f["scale"] == "tensor":
        v = v * scale.double()
    elif f["scale"] == "row":
        v = v * scale.double()[:, None]
    elif f["scale"] == "vec32":
        v = v * torch.pow(2.0, scale.double() - 127).repeat_interleave(32, dim=1)
    elif f["scale"] == "vec16":
        v = v * scale.view(torch.float8_e4m3fn).double().repeat_interleave(16, dim=1)
    return v


def pinned(fmt, n, k, **options):
    try:
        return pg.pin(fmt, n, k, **options)
    except pg.UnsupportedFormat as exc:
        pytest.skip(str(exc))


def call(p, fmt, a, b, scales, first, rows, out=None):
    f = pg.parse(fmt)
    sa, sb = scales["a"], scales["b"]
    if f["scale"] == "row":
        sa = sa[first:first + rows].contiguous()
    elif f["block"]:
        sa, sb = pg.blocked_scales(sa[first:first + rows]), pg.blocked_scales(sb)
    return p.mm(a[first:first + rows], b, sa, sb, scales["alpha"], out=out)


@pytest.mark.parametrize("fmt", FORMATS)
def test_rows_keep_their_bits_at_every_row_count_and_offset(fmt):
    n, k = 384, 512
    p = pinned(fmt, n, k)
    assert p.algo[3] <= 1 and p.algo[4] == 0   # split-K free, no reduction scheme
    a, b, scales = pg.operands(fmt, n, k, 4096, dev, 11)
    whole = call(p, fmt, a, b, scales, 0, 4096)
    for rows in range(1, 4097):
        assert torch.equal(call(p, fmt, a, b, scales, 0, rows), whole[:rows]), rows
    for first, rows in ((1, 1), (5, 17), (63, 130), (1000, 2000), (4000, 96)):
        assert torch.equal(call(p, fmt, a, b, scales, first, rows), whole[first:first + rows]), (first, rows)


@pytest.mark.parametrize("fmt", FORMATS)
def test_values_are_the_dequantized_product_within_fp32_sums(fmt):
    n, k = 256, 1024
    p = pinned(fmt, n, k)
    a, b, scales = pg.operands(fmt, n, k, 300, dev, 3)
    exact = dequantized(fmt, a, scales["a"], "a") @ dequantized(fmt, b, scales["b"], "b").t()
    if scales["alpha"] is not None:
        exact = exact * scales["alpha"].double()
    found = call(p, fmt, a, b, scales, 0, 300).double()
    bound = 2e-5 if pg.parse(fmt)["out"] == "fp32" else 1e-2
    assert float((found - exact).abs().max() / exact.abs().max()) < bound


def test_a_column_block_of_a_wider_output_and_a_captured_call():
    fmt = FORMATS[-2] if MAJOR >= 10 else "bf16:fp32"
    n, k = 256, 512
    p = pg.pin(fmt, n, k)
    a, b, scales = pg.operands(fmt, n, k, 200, dev, 7)
    alone = call(p, fmt, a, b, scales, 0, 200)
    wide = torch.zeros(200, 3 * n, dtype=p.dtype, device=dev)
    call(p, fmt, a, b, scales, 0, 200, out=wide[:, n:2 * n])
    assert torch.equal(wide[:, n:2 * n], alone) and not wide[:, :n].any() and not wide[:, 2 * n:].any()
    out = torch.empty(200, n, dtype=p.dtype, device=dev)
    f = pg.parse(fmt)
    sa = pg.blocked_scales(scales["a"]) if f["block"] else scales["a"]
    sb = pg.blocked_scales(scales["b"]) if f["block"] else scales["b"]
    p.mm(a, b, sa, sb, scales["alpha"], out=out)   # the plan meets 200 rows once eagerly
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p.mm(a, b, sa, sb, scales["alpha"], out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, alone)


def test_every_pin_makes_the_same_choice_and_checks_with_the_caller_s_weight():
    fmt = "nvfp4:bf16" if MAJOR >= 10 else "bf16:bf16"
    first, again = pg.pin(fmt, 5120, 2048), pg.pin(fmt, 5120, 2048)
    assert first.algo == again.algo
    _, b, scales = pg.operands(fmt, 5120, 2048, 1, dev, 9)
    f = pg.parse(fmt)
    weight = (b, pg.blocked_scales(scales["b"]) if f["block"] else scales["b"])
    assert pg.pin(fmt, 5120, 2048, weight=weight).algo == first.algo


def test_refusals():
    with pytest.raises(ValueError):
        pg.parse("nvfp4-row:bf16")
    with pytest.raises(ValueError):
        pg.parse("int8:bf16")
    p = pg.pin("bf16:bf16", 64, 64)
    with pytest.raises(RuntimeError):
        p.mm(torch.zeros(4, 32, dtype=torch.bfloat16, device=dev), torch.zeros(64, 64, dtype=torch.bfloat16,
                                                                                device=dev))
    assert pg.version() > 0


def test_a_tuned_knob_setting_pins_a_row_exact_algorithm_and_defaults_are_today_s():
    assert pg.tuned("bf16:bf16", 384, 512) == {"REP_ROWS": 2048, "CHOICE": 0}
    fmt = "nvfp4:bf16" if MAJOR >= 10 else "bf16:bf16"
    sid, _ = pg.structure(fmt, 384, 512)
    other = load("pinned-gemm", configs={sid: [{"REP_ROWS": 512, "CHOICE": 1}]})   # the defaults are configuration 0
    select(other, sid, 1)
    assert other.tuned(fmt, 384, 512) == {"REP_ROWS": 512, "CHOICE": 1}
    p = other.pin(fmt, 384, 512)
    assert p.algo[3] <= 1 and p.algo[4] == 0
    a, b, scales = other.operands(fmt, 384, 512, 1024, dev, 5)
    whole = call(p, fmt, a, b, scales, 0, 1024)
    for first, rows in ((0, 1), (0, 77), (300, 129), (1000, 24)):
        assert torch.equal(call(p, fmt, a, b, scales, first, rows), whole[first:first + rows]), (first, rows)
