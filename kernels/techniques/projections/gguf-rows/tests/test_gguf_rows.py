"""gguf-rows on this GPU, every format: the stored values (dequantize in FP32, and bench.py's host decoders) equal the
format's own reference decoder (gguf-py's dequantize; Core's host decoders for Q1_0 and Q2_0 where the installed gguf-py
has none, which Core's tests check against the formats' rules) bit for bit; gemv equals the declared order (bench.product) bit
for bit for 1 to 8 rows and any split; a row's bits are the same alone and among others; a gathered row equals the plain
call on its expert, also read in place from slots that hold other bytes beside each expert; and what the kernel cannot
read is refused."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from crit_kernels.build import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import BLOCKS, FORMATS, activations, blocks, product, stored_values  # noqa: E402

gguf = pytest.importorskip("gguf", reason="gguf-py is the formats' reference decoder")
gr = load("gguf-rows")
dev = torch.device("cuda")
NAMES = {v: k for k, v in FORMATS.items()}
SHAPES = [(64, 256), (96, 1024), (40, 4608)]


def reference_values(raw, fmt, k):
    """gguf-py's dequantize of the stored blocks, FP32 [N][K] (Core's decoders where gguf-py has no decoder for the
    type)."""
    from gguf import GGMLQuantizationType, quants
    try:
        return quants.dequantize(raw, GGMLQuantizationType(fmt)).astype(np.float32).reshape(raw.shape[0], k)
    except (ValueError, NotImplementedError, KeyError):
        if fmt not in (41, 42):
            raise
        return stored_values(raw, fmt, k)


@pytest.mark.parametrize("fmt", sorted(BLOCKS))
def test_stored_values_equal_the_formats_reference_decoder(fmt):
    for n, k in SHAPES:
        raw = blocks(fmt, n, k, seed=fmt * 7 + n)
        want = reference_values(raw, fmt, k)
        assert np.array_equal(stored_values(raw, fmt, k).view(np.uint32), want.view(np.uint32)), NAMES[fmt]
        w = gr.GgufMatrix(torch.from_numpy(raw).to(dev), fmt, k)
        got = gr.dequantize(w, torch.float32).cpu().numpy()
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), NAMES[fmt]
        # BF16: one rounding of the stored value.
        bf16 = gr.dequantize(w).cpu()
        assert torch.equal(bf16, torch.from_numpy(want).to(torch.bfloat16))


@pytest.mark.parametrize("fmt", sorted(BLOCKS))
def test_gemv_is_the_declared_order_bit_for_bit_and_a_rows_bits_its_own(fmt):
    for n, k in SHAPES:
        raw = blocks(fmt, n, k, seed=fmt * 11 + k)
        w = gr.GgufMatrix(torch.from_numpy(raw).to(dev), fmt, k)
        x = activations(8, k, dev)
        alone = gr.gemv(x[:1], w)
        for rows in (1, 2, 3, 5, 8):
            got = gr.gemv(x[:rows], w)
            want = product(raw, fmt, k, x[:rows], gr.splits(w)).to(torch.bfloat16)
            assert torch.equal(got.cpu(), want), (NAMES[fmt], n, k, rows)
            assert torch.equal(got[:1], alone)
        # FP32 and FP16 activations take the same order.
        for dtype in (torch.float32, torch.float16):
            xd = activations(3, k, dev, dtype)
            assert torch.equal(gr.gemv(xd, w).cpu(), product(raw, fmt, k, xd, gr.splits(w)).to(dtype))


@pytest.mark.parametrize("fmt", sorted(BLOCKS))
def test_a_matrix_at_any_byte_offset_has_the_same_bits(fmt):
    # The decoders read a block's fields by wider loads when its address allows (gguf.cuh's low_iq, IQ1_M) and byte by
    # byte otherwise: the same values either way.
    n, k = 40, 4608
    raw = blocks(fmt, n, k, seed=fmt * 13)
    aligned = gr.GgufMatrix(torch.from_numpy(raw).to(dev), fmt, k)
    x = activations(3, k, dev)
    want_y, want_v = gr.gemv(x, aligned).cpu(), gr.dequantize(aligned, torch.float32).cpu()
    buffer = torch.zeros(raw.nbytes + 16, dtype=torch.uint8, device=dev)
    for offset in (1, 2, 4):
        view = buffer[offset:offset + raw.nbytes].view(raw.shape)
        view.copy_(torch.from_numpy(raw).to(dev))
        w = gr.GgufMatrix(view, fmt, k)
        assert torch.equal(gr.gemv(x, w).cpu(), want_y), (NAMES[fmt], offset)
        assert torch.equal(gr.dequantize(w, torch.float32).cpu().view(torch.int32), want_v.view(torch.int32))


@pytest.mark.parametrize("parts", [1, 2, 3, 8])
def test_every_split_is_its_own_declared_order(parts):
    # The part count is part of the order: each one the reference reproduces.
    fmt, n, k = 12, 48, 8192
    raw = blocks(fmt, n, k, seed=5)
    w = gr.GgufMatrix(torch.from_numpy(raw).to(dev), fmt, k)
    x = activations(4, k, dev)
    xq, xd, xs = gr.quantize(x)
    yp = torch.empty(parts, 4, n, dtype=torch.float32, device=dev)
    gr.ops.gemv(w.blocks, fmt, k, xq, xd, xs, None, yp)
    y = torch.empty(4, n, dtype=torch.float32, device=dev)
    gr.ops.sum_parts(yp, y)
    assert torch.equal(y.cpu(), product(raw, fmt, k, x, parts))


@pytest.mark.parametrize("fmt", sorted(BLOCKS))
def test_a_gathered_row_equals_the_plain_call_on_its_expert(fmt):
    n, k, experts = 32, 1024, 4
    raw = blocks(fmt, n, k, seed=fmt, experts=experts)
    stack = gr.GgufMatrix(torch.from_numpy(raw).to(dev), fmt, k)
    ids = torch.tensor([3, 0, 3, 1, 2], dtype=torch.int32, device=dev)
    x = activations(5, k, dev)
    got = gr.gemv(x, stack, ids)
    for r, e in enumerate(ids.tolist()):
        one = gr.GgufMatrix(stack.blocks[e].contiguous(), fmt, k)
        assert torch.equal(got[r:r + 1], gr.gemv(x[r:r + 1], one)), (NAMES[fmt], r)
        assert torch.equal(gr.dequantize(stack, torch.float32, expert=e), gr.dequantize(one, torch.float32))


@pytest.mark.parametrize("fmt", [8, 16, 20, 22, 29, 42])
def test_a_stack_in_slots_with_other_bytes_reads_as_the_contiguous_stack(fmt):
    # An expert cache's slots: each expert's rows at the start of a slot of more bytes (another layer's larger
    # matrices, a down projection after them); the kernel reads the stack at the slots' stride, in place.
    n, k, experts = 24, 512, 5
    raw = blocks(fmt, n, k, seed=fmt + 3, experts=experts)
    size = raw.shape[1] * raw.shape[2] + 192   # a slot: the expert's rows and 192 other bytes
    slots = torch.randint(0, 256, (experts, size), dtype=torch.uint8, device=dev)
    slots[:, :raw.shape[1] * raw.shape[2]] = torch.from_numpy(raw.reshape(experts, -1)).to(dev)
    strided = slots.view(-1).as_strided((experts, n, raw.shape[2]), (size, raw.shape[2], 1))
    plain = gr.GgufMatrix(torch.from_numpy(raw).to(dev), fmt, k)
    ids = torch.tensor([4, 0, 2, 2, 1, 3], dtype=torch.int32, device=dev)
    x = activations(6, k, dev)
    assert torch.equal(gr.gemv(x, gr.GgufMatrix(strided, fmt, k), ids), gr.gemv(x, plain, ids)), NAMES[fmt]
    for e in range(experts):
        assert torch.equal(gr.dequantize(gr.GgufMatrix(strided, fmt, k), torch.float32, expert=e),
                           gr.dequantize(plain, torch.float32, expert=e))
    with pytest.raises(RuntimeError, match="do not overlap"):
        gr.gemv(x, gr.GgufMatrix(slots.view(-1).as_strided((experts, n, raw.shape[2]), (8, raw.shape[2], 1)), fmt,
                                 k), ids)


def test_a_nan_reaches_its_rows_outputs_and_no_others():
    raw = blocks(8, 16, 256, seed=1)
    w = gr.GgufMatrix(torch.from_numpy(raw).to(dev), 8, 256)
    x = activations(2, 256, dev)
    x[0, 5] = float("nan")
    y = gr.gemv(x, w)
    assert torch.isnan(y[0]).all() and not torch.isnan(y[1]).any()
    assert torch.equal(y[1:], gr.gemv(x[1:], w))


def test_the_device_grid_is_cores():
    # crit_kcore/gguf.cuh's IQ3_S table is crit_core's gguf_grids.json, word for word.
    import json
    import re
    from crit_core.model import gguf_blocks
    header = Path(__file__).resolve().parents[4] / "core" / "include" / "crit_kcore" / "gguf.cuh"
    text = header.read_text(encoding="utf-8")
    body = text[text.index("kIq3sGrid[512] = {"):]
    words = re.findall(r"0x([0-9a-f]{8})", body[:body.index("};")])
    grid = json.loads(Path(gguf_blocks.__file__).with_name("gguf_grids.json").read_text(encoding="utf-8"))["iq3_s"]
    assert words == grid and len(words) == 512


def test_the_device_iq_grids_are_cores():
    # crit_kcore/gguf_grids.cuh's tables are crit_core's gguf_grids.json, word for word.
    import json
    import re
    from crit_core.model import gguf_blocks
    header = Path(__file__).resolve().parents[4] / "core" / "include" / "crit_kcore" / "gguf_grids.cuh"
    text = header.read_text(encoding="utf-8")
    grids = json.loads(Path(gguf_blocks.__file__).with_name("gguf_grids.json").read_text(encoding="utf-8"))
    for name, table in (("iq2_xxs", "kIq2xxsGrid"), ("iq2_xs", "kIq2xsGrid"), ("iq2_s", "kIq2sGrid"),
                        ("iq3_xxs", "kIq3xxsGrid"), ("iq1_s", "kIq1sGrid")):
        body = text[text.index(f"{table}["):]
        words = re.findall(r"0x([0-9a-f]+)u", body[:body.index("};")])
        assert words == grids[name], name


def test_what_the_kernel_cannot_read_is_refused():
    raw = torch.from_numpy(blocks(12, 8, 256, seed=2)).to(dev)
    x = activations(1, 256, dev)
    with pytest.raises(RuntimeError, match="GGUF type 1 is not one of"):
        gr.gemv(x, gr.GgufMatrix(raw, 1, 256))
    with pytest.raises(RuntimeError, match="multiple of the format's 256"):
        gr.dequantize(gr.GgufMatrix(raw[:, :72].contiguous(), 12, 128))
    with pytest.raises(RuntimeError, match="holds 288 bytes, not 144"):
        gr.gemv(activations(1, 512, dev), gr.GgufMatrix(raw, 12, 512))
    with pytest.raises(RuntimeError, match="1..8"):
        gr.gemv(activations(9, 256, dev), gr.GgufMatrix(raw, 12, 256))


@pytest.mark.parametrize("fmt", sorted(BLOCKS))
def test_the_prompt_gemm_is_gemvs_order_bit_for_bit(fmt):
    for n, k in ((48, 1024), (40, 4608)):
        raw = blocks(fmt, n, k, seed=fmt * 13 + n)
        w = gr.GgufMatrix(torch.from_numpy(raw).to(dev), fmt, k)
        x = activations(100, k, dev)
        full = gr.gemm(x, w)
        # Each output gemv's bits for its row, at any row count, and the declared order's.
        assert torch.equal(full[:8], gr.gemv(x[:8], w)), (NAMES[fmt], n, k)
        for rows in (1, 9, 33, 100):
            assert torch.equal(gr.gemm(x[:rows], w), full[:rows])
        assert torch.equal(full.cpu(), product(raw, fmt, k, x, gr.splits(w)).to(torch.bfloat16))
        assert torch.equal(gr.matmul(x[:5], w), full[:5]) and torch.equal(gr.matmul(x[:20], w), full[:20])


@pytest.mark.parametrize("fmt", sorted(BLOCKS))
def test_the_prompt_gemm_staged_in_chunks_has_the_same_bits(fmt):
    # The matrix's rows decoded a chunk at a time (here 16, then the 8 left) give the outputs of one staging.
    n, k = 40, 1024
    raw = blocks(fmt, n, k, seed=fmt * 17 + 3)
    w = gr.GgufMatrix(torch.from_numpy(raw).to(dev), fmt, k)
    x = activations(37, k, dev)
    xq, xd, xs = gr.quantize(x, by_subblock=True)
    y = torch.empty(37, n, dtype=x.dtype, device=dev)
    gr.ops.gemm(w.blocks, fmt, k, xq, xd, xs, gr.splits(w), y, 16)
    assert torch.equal(y, gr.gemm(x, w)) and torch.equal(y[:8], gr.gemv(x[:8], w)), NAMES[fmt]


@pytest.mark.parametrize("fmt", sorted(BLOCKS))
def test_the_grouped_prompt_gemm_gives_each_row_its_experts_gemm_bits(fmt):
    # A mixture of experts' prompt rows sorted by expert: groups of rows, each times its own expert of a stack read in
    # place from slots (experts in any order, one appearing once, groups shorter and longer than a tile), staged in
    # one launch and multiplied in one: every output the plain gemm's bits for its row on its expert.
    n, k, experts = 48, 1024, 6
    raw = blocks(fmt, n, k, seed=fmt * 7 + 1, experts=experts)
    size = raw.shape[1] * raw.shape[2] + 128
    slots = torch.randint(0, 256, (experts, size), dtype=torch.uint8, device=dev)
    slots[:, :raw.shape[1] * raw.shape[2]] = torch.from_numpy(raw.reshape(experts, -1)).to(dev)
    stack = gr.GgufMatrix(slots.view(-1).as_strided((experts, n, raw.shape[2]), (size, raw.shape[2], 1)), fmt, k)
    ids, counts = [4, 0, 5, 2], [3, 70, 1, 130]
    x = activations(sum(counts), k, dev)
    got = gr.gemm_experts(x, stack, ids, counts)
    start = 0
    for e, c in zip(ids, counts):
        one = gr.GgufMatrix(torch.from_numpy(raw[e]).to(dev), fmt, k)
        assert torch.equal(got[start:start + c], gr.gemm(x[start:start + c], one)), (NAMES[fmt], e)
        start += c
