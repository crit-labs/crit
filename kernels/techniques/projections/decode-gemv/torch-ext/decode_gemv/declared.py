"""decode-gemv's declared decode rows (structures rows-int-block and rows-fp8-block): the kernel core's declared rows
on its declared multiply paths, which int-block-experts' and fp8-block-rows' rows forward to (their kernel.json
aliases), the same bits. The records are those packages' (int-block-experts: Matrix codes, scales, rows, cols, post,
int4; fp8-block-rows: codes, scales, rows, cols, scale_rows, scale_k, post, scale_e8m0; RowJob matrix, tokens,
act_row[8], out_row[8]), so a table either package packs runs here too."""
import struct
from typing import Sequence

import torch

from ._ops import ops

_INT_MATRIX = struct.Struct("<QQiifi")
_FP8_MATRIX = struct.Struct("<QQiiiifi")
_ROW_JOB = struct.Struct("<ii8i8i")


def _device_bytes(packed: bytes, device) -> torch.Tensor:
    return torch.frombuffer(bytearray(packed), dtype=torch.uint8).to(device)


def int_block_table(matrices: Sequence, device=None) -> torch.Tensor:
    """One record per (codes [rows, cols / 2] uint8, scales [rows, cols / 16] uint8 E4M3, post, int4) of CUDA tensors
    (int-block-experts' matrix_table). The table holds the tensors' pointers: keep them alive while it is used."""
    packed = b"".join(_INT_MATRIX.pack(codes.data_ptr(), scales.data_ptr(), codes.size(0), codes.size(1) * 2,
                                       float(post), int(bool(int4))) for codes, scales, post, int4 in matrices)
    return _device_bytes(packed, device or matrices[0][0].device)


def fp8_block_table(matrices: Sequence, device=None) -> torch.Tensor:
    """One record per (codes [rows, cols] uint8 E4M3, scales fp32 or uint8 E8M0 [ceil(rows / scale_rows),
    cols / scale_k], scale_rows, scale_k, post) of CUDA tensors (fp8-block-rows' matrix_table)."""
    packed = b""
    for codes, scales, scale_rows, scale_k, post in matrices:
        if scales.dtype not in (torch.float32, torch.uint8):
            raise ValueError("scales are fp32 or uint8 E8M0")
        packed += _FP8_MATRIX.pack(codes.data_ptr(), scales.data_ptr(), codes.size(0), codes.size(1), int(scale_rows),
                                   int(scale_k), float(post), int(scales.dtype == torch.uint8))
    return _device_bytes(packed, device or matrices[0][0].device)


def row_jobs(jobs: Sequence, device) -> torch.Tensor:
    """Decode jobs: (matrix, [activation rows], [output rows]) with at most 8 tokens each."""
    packed = b""
    for matrix, act_rows, out_rows in jobs:
        if not 0 < len(act_rows) <= 8 or len(act_rows) != len(out_rows):
            raise ValueError("a decode job has 1 to 8 tokens, one output row each")
        pad = [0] * (8 - len(act_rows))
        packed += _ROW_JOB.pack(matrix, len(act_rows), *act_rows, *pad, *out_rows, *pad)
    return _device_bytes(packed, device)


def rows_int_block(table, jobs, max_rows, values, scales, block, out):
    """out[out_row[j]][r] = the job's matrix row r times activation row act_row[j] in the declared integer order:
    values int8 [rows, cols] (E2M1 x 2 or INT8), scales FP32 per `block` (16 or 32); out FP32."""
    ops.rows_int_block(table, jobs, int(max_rows), values, scales, int(block), out)


def rows_fp8_block(table, jobs, max_rows, codes, scales, group, out):
    """The same in the declared FP8 order: codes uint8 E4M3 [rows, cols], scales FP32 per `group` (a multiple of 32);
    out FP32. Compute capability 8.9 on."""
    ops.rows_fp8_block(table, jobs, int(max_rows), codes, scales, int(group), out)
