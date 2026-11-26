"""Matched FP8 projections in crit's declared per-output order: E4M3 weights (MXFP8's E8M0 block scales, an FP8
checkpoint's 128 x 128 FP32 block scales, or row scales) against E4M3 activations quantized per group. Each 32-element
block is one e4m3 tensor-core MMA with a zero accumulator; the FP32 step (lane b % 16 = fma(block sum, weight scale *
activation scale, lane), the halving tree, post) is declared, so the decode rows and the prompt GEMM give the same bits
for every output: decode, verify rounds, prefill and a prompt cache agree bit for bit. Same bits on one GPU
architecture; check on the target (tests/fp8_block_tests.cu)."""
import struct
from typing import Sequence

import torch

from ._crit_aliases import module as _alias
from ._ops import ops

__all__ = ["NATIVE_BF16_OUT", "matrix_table", "row_jobs", "gemm_jobs", "rows", "gemm", "gemm_staged", "gemm_native",
           "gemm_native_channel", "quantize", "quantized_values", "to_visit_weights", "visit_positions",
           "to_visit_order", "MX", "AMAX", "PLAIN", "VISIT"]

NATIVE_BF16_OUT = True   # gemm_native and gemm_native_channel take a BF16 out (the FP32 sums rounded once)
MX, AMAX = 0, 1                          # activation scale kinds: MXFP8's 2^(floor(log2 amax) - 8), or amax / 448
PLAIN, VISIT = 0, 1                      # block orders: plain, or the visit order (activations: gemm_staged's and
                                         # quantize's; weights: matrix_table's order, to_visit_weights)
_MATRIX = struct.Struct("<QQiiiifii4x")  # Matrix: codes, scales, rows, cols, scale_rows, scale_k, post, scale_e8m0,
                                         # order (the weight codes' blocks), padding
_ROW_JOB = struct.Struct("<ii8i8i")      # RowJob: matrix, tokens, act_row[8], out_row[8]
_GEMM_JOB = struct.Struct("<iii")        # GemmJob: matrix, first, count


def _device_bytes(packed: bytes, device) -> torch.Tensor:
    return torch.frombuffer(bytearray(packed), dtype=torch.uint8).to(device)


def matrix_table(matrices: Sequence, device=None) -> torch.Tensor:
    """One Matrix per (codes [rows, cols] uint8 E4M3, scales, scale_rows, scale_k, post[, order]) of CUDA tensors.
    scales are fp32 [ceil(rows / scale_rows), cols / scale_k], or uint8 E8M0 bytes as an MXFP8 checkpoint stores them.
    order: PLAIN codes, or VISIT codes (to_visit_weights: each row's blocks of 32 in the visit order, stored once at
    load), which rows, gemm and gemm_staged read with the same bits and the native GEMMs refuse. The table holds the
    tensors' pointers: keep them alive while it is used."""
    packed, channel, visit = b"", True, False
    for entry in matrices:
        codes, scales, scale_rows, scale_k, post = entry[:5]
        order = int(entry[5]) if len(entry) > 5 else PLAIN
        if scales.dtype not in (torch.float32, torch.uint8):
            raise ValueError("scales are fp32 or uint8 E8M0")
        if order not in (PLAIN, VISIT) or (order == VISIT and codes.size(1) % 32):
            raise ValueError("a matrix's order is PLAIN or VISIT (whole blocks of 32 a row)")
        packed += _MATRIX.pack(codes.data_ptr(), scales.data_ptr(), codes.size(0), codes.size(1), int(scale_rows),
                               int(scale_k), float(post), int(scales.dtype == torch.uint8), order)
        channel = channel and scales.dtype == torch.float32 and int(scale_k) == codes.size(1)
        visit = visit or order == VISIT
    table = _device_bytes(packed, device or matrices[0][0].device)
    table.crit_channel = channel   # every matrix's scales constant along K: gemm_native may take the channel GEMM
    table.crit_visit = visit       # a matrix in the visit order: the native GEMMs (plain codes only) refuse the table
    return table


def row_jobs(jobs: Sequence, device) -> torch.Tensor:
    """Decode jobs: (matrix, [activation rows], [output rows]) with at most 8 tokens each."""
    packed = b""
    for matrix, act_rows, out_rows in jobs:
        if not 0 < len(act_rows) <= 8 or len(act_rows) != len(out_rows):
            raise ValueError("a decode job has 1 to 8 tokens, one output row each")
        pad = [0] * (8 - len(act_rows))
        packed += _ROW_JOB.pack(matrix, len(act_rows), *act_rows, *pad, *out_rows, *pad)
    return _device_bytes(packed, device)


def gemm_jobs(jobs: Sequence, device) -> torch.Tensor:
    """Prompt jobs: (matrix, first activation row, count): out[first + i] = matrix applied to activation row first + i."""
    return _device_bytes(b"".join(_GEMM_JOB.pack(m, first, count) for m, first, count in jobs), device)


def rows(table, jobs, max_rows, codes, scales, group, out):
    """Decode rows, 1 to 8 tokens a job: decode-gemv's rows-fp8-block (this kernel's structure rows aliases it), the
    kernel core's declared FP8 rows, built with this kernel's tuning for rows."""
    _alias("rows").rows_fp8_block(table, jobs, max_rows, codes, scales, group, out)


def gemm(table, jobs, max_rows, max_count, codes, scales, group, out):
    ops.gemm(table, jobs, max_rows, max_count, codes, scales, group, out)


def gemm_staged(table, jobs, max_rows, max_count, codes, scales, group, out, order=PLAIN):
    """gemm with shared-memory staging (structure gemm-staged): the same bits, faster on prompt chunks. codes' rows
    16-byte aligned (cols a multiple of 16). order: PLAIN activations, or quantize(order=VISIT)'s (faster: a visit's
    blocks of a token one contiguous piece)."""
    ops.gemm_staged(table, jobs, max_rows, max_count, codes, scales, group, order, out)


def gemm_native(table, jobs, max_rows, max_count, codes, scales, group, out, mx=True):
    """The prompt GEMM accumulated in the tensor cores (structure gemm-native): NOT the declared order, so not the bits
    of rows / gemm / gemm_staged, and an output's bits may change with the tuning and the job's token count (not row
    invariant); within FP32 accumulation of the exact product. mx: the activation scales are MXFP8's powers of two
    (quantize(kind=MX)), which lets an sm_120a GPU multiply E8M0 MXFP8 weights at group 32 on block-scaled tensor cores;
    any other scales run the generic path; per-channel scales (a table whose matrices' scales are constant along K and
    one activation scale a token) the channel GEMM (gemm_native_channel). out fp32 or bf16 (the sums times post
    rounded once). codes' rows 16-byte aligned."""
    if getattr(table, "crit_visit", False):
        raise ValueError("gemm_native reads plain weight codes: a matrix of this table is in the visit order")
    if not mx and scales.size(1) == 1 and getattr(table, "crit_channel", False):
        return gemm_native_channel(table, jobs, max_rows, max_count, codes, scales, out)
    ops.gemm_native(table, jobs, max_rows, max_count, codes, scales, group, bool(mx), out)


def gemm_native_channel(table, jobs, max_rows, max_count, codes, scales, out):
    """The prompt GEMM of matrices whose FP32 scales do not change along K (scale_k = cols: a scale a row, a block of
    rows or the tensor) against activations with one scale a token (scales [tokens, 1]), an FP8 checkpoint's
    per-channel W8A8 (structure gemm-native-channel): every block of K summed in the tensor cores' FP32 accumulator,
    token scale x weight scale x post applied in the epilogue, out fp32 or bf16 (rounded once). Not the declared
    order, not row invariant. gemm_native(mx=False) takes it for such a table (matrix_table's crit_channel) and such
    scales. A matrix that is not channel-scaled gets NaN outputs. codes' rows 16-byte aligned."""
    if getattr(table, "crit_visit", False):
        raise ValueError("gemm_native_channel reads plain weight codes: a matrix of this table is in the visit order")
    ops.gemm_native_channel(table, jobs, max_rows, max_count, codes, scales, out)


def quantize(x: torch.Tensor, group: int = 32, kind: int = MX, order: int = PLAIN):
    """x [rows, cols] fp32 or bf16 -> (codes uint8 E4M3, scales fp32 per group): crit's declared quantizer. order
    PLAIN: codes [rows, cols], scales [rows, cols / group]; VISIT, for gemm_staged only: the same codes with each row's
    blocks of 32 in the kernel core's visit order (visit_positions), scales [cols / group, rows]."""
    codes = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    shape = (x.size(1) // group, x.size(0)) if order == VISIT else (x.size(0), x.size(1) // group)
    scales = torch.empty(shape, dtype=torch.float32, device=x.device)
    ops.quantize(x.contiguous(), group, kind, order, codes, scales)
    return codes, scales


def visit_positions(cols: int) -> list:
    """Where the visit order keeps each block of 32 of a row of `cols` codes (the kernel core's VisitOrder): the tree's
    steps in sibling pairs, a pair's blocks interleaved by their index along the lane, the pairs in the tree's order."""
    blocks = cols // 32
    lane_at = [int(f"{at:04b}"[::-1], 2) for at in range(16)]
    count = [len(range(lane_at[at], blocks, 16)) for at in range(16)]
    pos = [0] * blocks
    start = 0
    for pair in range(8):
        for at in (2 * pair, 2 * pair + 1):
            for k in range(count[at]):
                pos[lane_at[at] + 16 * k] = start + 2 * k + (at & 1)
        start += count[2 * pair] + count[2 * pair + 1]
    return pos


def to_visit_order(codes: torch.Tensor, scales: torch.Tensor):
    """Plain activations (codes [rows, cols], scales [rows, groups]) in gemm_staged's visit order: the same bytes as
    quantize(order=VISIT) gives."""
    rows, cols = codes.shape
    pos = torch.tensor(visit_positions(cols), device=codes.device)
    out = torch.empty_like(codes).view(rows, cols // 32, 32)
    out[:, pos] = codes.view(rows, cols // 32, 32)
    return out.view(rows, cols), scales.t().contiguous()


def to_visit_weights(codes: torch.Tensor) -> torch.Tensor:
    """A matrix's codes [rows, cols] (plain) in the visit order, a new tensor: each row's blocks of 32 where the kernel
    core's VisitOrder keeps them (visit_positions), for matrix_table's order VISIT. The scales stay as they are (they
    follow a block's index, not its position)."""
    rows, cols = codes.shape
    if cols % 32:
        raise ValueError("the visit order takes whole blocks of 32 a row")
    pos = torch.tensor(visit_positions(cols), device=codes.device)
    out = torch.empty_like(codes).view(rows, cols // 32, 32)
    out[:, pos] = codes.reshape(rows, cols // 32, 32)
    return out.view(rows, cols)


def quantized_values(x: torch.Tensor, group: int = 32, kind: int = MX):
    """x [rows, cols] fp32 or bf16 -> bf16 [rows, cols]: each element's code of quantize() times its group's scale
    (the FP32 product rounded to BF16: exact for MXFP8's powers of two), in one launch. For a caller that multiplies
    16-bit rows of quantized activations: the bits of (codes.float() * scales).to(bfloat16)."""
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    ops.quantized_values(x.contiguous(), group, kind, out)
    return out
