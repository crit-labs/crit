"""Integer block expert projections in crit's declared integer order: 4-bit weights (NVFP4's E2M1 codes or INT4, E4M3
scale per 16) against activations quantized per block (E2M1 x 2 or INT8). Each 16-block's integer dot is exact; the
FP32 step (lane b % 16 = fma(sum, weight scale * activation scale, lane), the halving tree, post) is declared, so the
decode rows, the prompt GEMM and crit's CPU rows (host_int.h) give the same bits for every output: a CPU expert tier,
a device cache and the prompt path agree bit for bit. The quantizers and SwiGLU are crit's declared ones."""
import struct
from typing import Sequence

import torch

from ._crit_aliases import module as _alias
from ._ops import ops

__all__ = ["matrix_table", "row_jobs", "gemm_jobs", "rows", "gemm", "gemm_staged", "gemm_native", "tiled_supported", "TILED_ACTS", "TILED_WEIGHTS", "NATIVE_BF16_OUT", "NATIVE_COLUMN_SCALES", "NATIVE_TILED_SCALES", "quantize_e2m1x2", "quantize_nvfp4", "quantize_int8", "swiglu"]

_MATRIX = struct.Struct("<QQiifi")       # Matrix: codes, scales, rows, cols, post, int4
_ROW_JOB = struct.Struct("<ii8i8i")      # RowJob: matrix, tokens, act_row[8], out_row[8]
_GEMM_JOB = struct.Struct("<iii")        # GemmJob: matrix, first, count


def _device_bytes(packed: bytes, device) -> torch.Tensor:
    return torch.frombuffer(bytearray(packed), dtype=torch.uint8).to(device)


def matrix_table(matrices: Sequence, device=None) -> torch.Tensor:
    """One Matrix per (codes [rows, cols / 2] uint8, scales [rows, cols / 16] uint8 E4M3, post, int4) of CUDA tensors."""
    packed = b"".join(_MATRIX.pack(codes.data_ptr(), scales.data_ptr(), codes.size(0), codes.size(1) * 2, float(post),
                                   int(bool(int4))) for codes, scales, post, int4 in matrices)
    table = _device_bytes(packed, device or matrices[0][0].device)
    # What the host knows of the matrices (codes pointer, rows, cols), for gemm_native's tensor maps.
    table.crit_host = torch.tensor([[codes.data_ptr(), codes.size(0), codes.size(1) * 2] for codes, _, _, _ in matrices],
                                   dtype=torch.int64)
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


def rows(table, jobs, max_rows, values, scales, block, out):
    """Decode rows, 1 to 8 tokens a job: decode-gemv's rows-int-block (this kernel's structure rows aliases it), the
    kernel core's declared integer rows, built with this kernel's tuning for rows."""
    _alias("rows").rows_int_block(table, jobs, max_rows, values, scales, block, out)


def gemm(table, jobs, max_rows, max_count, values, scales, block, out):
    ops.gemm(table, jobs, max_rows, max_count, values, scales, block, out)


def gemm_staged(table, jobs, max_rows, max_count, values, scales, block, out):
    """gemm with shared-memory staging (structure gemm-staged): the same bits, faster on prompt chunks."""
    ops.gemm_staged(table, jobs, max_rows, max_count, values, scales, block, out)


def quantize_e2m1x2(x: torch.Tensor, global_scale: float):
    """x [rows, cols] fp32 -> (values int8 E2M1 x 2, scales fp32 per 16): crit's declared quantizer."""
    values = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty(x.size(0), x.size(1) // 16, dtype=torch.float32, device=x.device)
    ops.quantize_e2m1x2(x.contiguous(), float(global_scale), values, scales)
    return values, scales


TILED_ACTS, TILED_WEIGHTS = 1, 2   # gemm_native's tiled: which scale bytes are in cuBLASLt's 128 x 4 tiled layout
# What this build's gemm_native takes, for callers that also run older builds: a BF16 `out`, `col_scale`, `tiled`.
NATIVE_BF16_OUT = True
NATIVE_COLUMN_SCALES = True
NATIVE_TILED_SCALES = True


def gemm_native(table, jobs, max_rows, max_count, codes, scales, global_scale, out, int4=False, col_scale=None,
                tiled=0):
    """The prompt GEMM accumulated in the tensor cores (structure gemm-native) on quantize_nvfp4's activations: NOT the
    declared order, so not the bits of rows / gemm / gemm_staged or the CPU rows, and an output's bits may change with
    the tuning and the job's token count (not row invariant); within FP32 accumulation of the exact product. int4
    False states that every matrix holds E2M1 weights, which sm_120a GPUs multiply on block-scaled FP4 tensor cores
    (warp-specialized; an INT4 matrix then gets NaN outputs); int4 True runs the s8 MMA a 16-block, as GPUs without
    block-scaled MMAs do.

    out: FP32 or BF16 [rows, out_ld] (BF16: the FP32 result rounded once, to nearest even, in the epilogue).
    col_scale: optional FP32 [max_rows], output column c times col_scale[c] after the global scale.
    tiled: TILED_ACTS when `scales` is in cuBLASLt's 128 x 4 tiled layout (rows padded to 128; every job's first row a
    multiple of 128), TILED_WEIGHTS when every matrix's scales are (rows padded to 128); sm_120a's E2M1 path only
    (tiled_supported())."""
    ops.gemm_native(table, jobs, max_rows, max_count, codes, scales, float(global_scale), bool(int4), out, col_scale,
                    int(tiled), getattr(table, "crit_host", None))


def tiled_supported(int4=False) -> bool:
    """Whether gemm_native takes tiled scale bytes here (the warp-specialized block-scaled path: E2M1 weights on a
    compute capability 12 GPU)."""
    return not int4 and torch.cuda.get_device_capability()[0] == 12


def quantize_nvfp4(x: torch.Tensor, global_scale: float, tiled: bool = False):
    """x [rows, cols] fp32 or bf16 -> (codes uint8 [rows, cols / 2], two E2M1 codes a byte, scales uint8 E4M3 bytes):
    quantize_e2m1x2's values packed for gemm_native (the same codes; a block's value is 2 x E2M1 x the byte's E4M3 value
    / global_scale, the E2M1 x 2 integer times quantize_e2m1x2's scale; a BF16 x widened exactly first). scales
    [rows, cols / 16], or with tiled [ceil(rows / 128) * 128, cols / 16] bytes in cuBLASLt's 128 x 4 tiled layout
    (gemm_native's tiled=TILED_ACTS; the padding rows' bytes unset)."""
    x = x if x.dtype in (torch.float32, torch.bfloat16) else x.float()
    x = x.contiguous()
    codes = torch.empty(x.size(0), x.size(1) // 2, dtype=torch.uint8, device=x.device)
    rows = -(-x.size(0) // 128) * 128 if tiled else x.size(0)
    scales = torch.empty(rows, x.size(1) // 16, dtype=torch.uint8, device=x.device)
    ops.quantize_nvfp4(x, float(global_scale), codes, scales, TILED_ACTS if tiled else 0)
    return codes, scales


def quantize_int8(x: torch.Tensor, block: int = 32):
    """x [rows, cols] fp32 -> (values int8, scales fp32 per block of 16 or 32): crit's declared quantizer."""
    values = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty(x.size(0), x.size(1) // block, dtype=torch.float32, device=x.device)
    ops.quantize_int8(x.contiguous(), block, values, scales)
    return values, scales


def swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """gate_up [rows, 2 * inter] fp32 -> [rows, inter] fp32, crit's declared SwiGLU."""
    out = torch.empty(gate_up.size(0), gate_up.size(1) // 2, dtype=torch.float32, device=gate_up.device)
    ops.swiglu(gate_up.contiguous(), out)
    return out
