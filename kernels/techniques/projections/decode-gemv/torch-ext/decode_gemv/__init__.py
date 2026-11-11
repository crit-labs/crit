"""Decode GEMVs from crit's quantized-projections technique, weight only: NVFP4 weights (W4A16), FP8 E4M3 weights with a
row scale or MXFP8's E8M0 scale per 32 inputs (W8A16) and BF16 weights times 1 to 16 rows of 16-bit activations on
CUDA cores; the same formats in Triton for 1 to 16 rows (gemv_any, structure gemv-triton: the weight format a decoder
over the kernel core's, the multiply an FP32 sum a 16-value group (fma) or tl.dot on the values exact in 16 bits (dot),
WEIGHTS' codes); its NVFP4 fma path for 1 to 8 rows as gemv_triton and its FP8 dot path as gemv_fp8_dot (a program of
16 rows by a block of the matrix, a fixed split of K added in order; fp8_dot_tile gives a matrix its program shape),
the GEMVs they were; and the decoders that write an NVFP4 matrix's values for a provider's GEMM. FP32
sums; each row's result is the same bits alone and beside the other rows of a call (the order never depends on the row
count). The GEMVs sum in different orders, so a process keeps to one a matrix.

NVFP4 matrices as checkpoints store them: codes [N][K / 2] uint8 (element 2i in the low nibble of byte i), E4M3 block
scales [N][K / 16] (row major), and inverse [N] FP32, one over each row's global scale (a value is e2m1 * scale *
inverse). FP8: codes [N][K] E4M3 and FP32 row scales [N], or E8M0 bytes [N][K / 32] (MXFP8, modelopt's layout).
INT4 with a zero point a group (AWQ, GPTQ): repack_awq / repack_gptq put a checkpoint's tensors in crit's rows
(codes [N][K / 2] unsigned, element 2i in the low nibble of byte i; FP16 scales and uint8 zero points [N][K / group]),
which gemv_int4 multiplies: a value is (code - zero) * scale, exact in FP32.

The declared decode rows (rows_int_block, rows_fp8_block: structures rows-int-block and rows-fp8-block) are crit's
declared integer and FP8 block orders on the kernel core's declared rows, which int-block-experts' and fp8-block-rows'
rows forward to.

The CUDA GEMVs are one kernel (kernel.json's structure gemv-cuda) with the weight format a run option (WEIGHTS: the
codes mma-rows uses too); gemv, gemv_fp8, gemv_bf16 and gemv_int4 call it for their format."""
import torch

from ._ops import ops
from .triton_gemv import MAX_ROWS as TRITON_ROWS, decode, decode_fp8, gemv_triton, hardware
from .dot_gemv import MAX_ROWS as DOT_ROWS, gemv_fp8_dot, partial_elements as dot_partial_elements, \
    tile as fp8_dot_tile
from .triton_any import MAX_ROWS as ANY_ROWS, gemv_any, plan as triton_plan
from .declared import fp8_block_table, int_block_table, row_jobs as declared_row_jobs, rows_fp8_block, rows_int_block

CUDA_ROWS = 16   # rows the CUDA GEMVs take in one call
WEIGHTS = {"nvfp4": 0, "fp8": 1, "mxfp8": 2, "bf16": 3, "int4_zp": 4}   # gemv-cuda's weight formats (its key)

__all__ = ["ANY_ROWS", "CUDA_ROWS", "DOT_ROWS", "TRITON_ROWS", "WEIGHTS", "dot_partial_elements", "fp8_dot_tile", "gemv",
           "gemv_any", "gemv_fp8", "gemv_fp8_dot", "gemv_bf16", "gemv_int4", "gemv_triton", "decode", "decode_fp8",
           "hardware_decode", "hardware", "repack_awq", "repack_gptq", "triton_plan", "declared_row_jobs",
           "fp8_block_table", "int_block_table", "rows_fp8_block", "rows_int_block"]


def gemv(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, inverse: torch.Tensor, out: torch.Tensor,
         blocked: bool = False):
    """out [M][N] (FP32 or x's dtype) = x [M][K] (M 1..16, BF16 or FP16, rows on 16-byte boundaries, K a multiple of
    32) times the NVFP4 matrix, on CUDA cores. blocked: the E4M3 block scales are in cuBLASLt's 128 x 4 tiled layout
    (what its block-scaled GEMM reads; rows padded to 128, blocks to 4) instead of row major [N][K / 16]."""
    ops.gemv(x, codes, scales, inverse, blocked, out)
    return out


def gemv_fp8(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, out: torch.Tensor):
    """out [M][N] (FP32 or x's dtype) = x [M][K] (M 1..16, BF16 or FP16, rows on 16-byte boundaries) times the FP8
    matrix: codes [N][K] E4M3 (float8_e4m3fn or uint8, K a multiple of 16) with scales FP32 [N] (one per row, applied
    after the sum) or uint8 E8M0 [N][K / 32] (MXFP8, each value times its power of two), on CUDA cores."""
    ops.gemv_fp8(x, codes.view(torch.uint8), scales, out)
    return out


def gemv_bf16(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor):
    """out [M][N] (FP32 or BF16) = x [M][K] (M 1..16, BF16, rows on 16-byte boundaries) times w [N][K] BF16 (K a
    multiple of 8), on CUDA cores."""
    ops.gemv_bf16(x, w, out)
    return out


def gemv_int4(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, out: torch.Tensor,
              group: int):
    """out [M][N] (FP32 or x's dtype) = x [M][K] (M 1..16, BF16 or FP16, rows on 16-byte boundaries) times the INT4
    matrix in crit's rows with a zero point a group (repack_awq, repack_gptq): codes [N][K / 2] uint8, scales FP16 and
    zeros uint8 [N][K / group], group a multiple of 32 dividing K; a value is (code - zero) * scale. On CUDA cores, FP32
    sums of exact products in the order of every other format (a row's bits independent of the rows beside it)."""
    ops.gemv_int4(x, codes, scales, zeros, int(group), out)
    return out


def _repack(kind, qweight, qzeros, scales, n, k, group, perm, zero_offset):
    device = qweight.device
    codes = torch.empty(n, k // 2, dtype=torch.uint8, device=device)
    zeros = torch.empty(n, k // group, dtype=torch.uint8, device=device)
    scales_out = torch.empty(n, k // group, dtype=torch.float16, device=device)
    ops.repack_int4(kind, qweight.contiguous().view(torch.int32), qzeros.contiguous().view(torch.int32),
                    scales.contiguous(), None if perm is None else perm.to(device, torch.int32).contiguous(),
                    int(zero_offset), int(group), codes, zeros, scales_out)
    return codes, scales_out, zeros


def repack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group: int):
    """An AWQ checkpoint's tensors in crit's INT4 rows, exactly: qweight int32 [K][N / 8] and qzeros int32
    [K / group][N / 8] (eight values a word in AWQ's order 0, 2, 4, 6, 1, 3, 5, 7), scales FP16 [K / group][N] ->
    (codes [N][K / 2], scales [N][K / group], zeros [N][K / group]) for gemv_int4 (and mma-rows' int4_zp)."""
    return _repack(0, qweight, qzeros, scales, qweight.shape[1] * 8, qweight.shape[0], group, None, 0)


def repack_gptq(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group: int, g_idx=None,
                zero_offset: int = 1):
    """A GPTQ checkpoint's tensors in crit's INT4 rows, exactly: qweight int32 [K / 8][N] (eight inputs a word in
    sequence), qzeros int32 [K / group][N / 8] (in sequence; zero = stored + zero_offset, 1 for checkpoints that store
    zero - 1, GPTQ's original format, 0 for those that store the zero), scales FP16 [K / group][N], g_idx int [K] or
    None. Returns (codes [N][K / 2], scales [N][K / group], zeros [N][K / group], perm): with act order (g_idx other
    than k // group) the inputs are sorted by group (stable), the row's input k' is the checkpoint's input perm[k'],
    and the caller multiplies x[:, perm]; perm is None otherwise. A g_idx whose groups are not group inputs each is
    refused."""
    k, n = qweight.shape[0] * 8, qweight.shape[1]
    perm = None
    if g_idx is not None:
        g_idx = g_idx.to(qweight.device, torch.int64)
        plain = torch.arange(k, device=qweight.device) // group
        if not torch.equal(g_idx, plain):
            perm = torch.sort(g_idx, stable=True).indices
            if not torch.equal(g_idx[perm], plain):
                raise ValueError("repack_gptq: g_idx's groups are not of `group` inputs each")
    codes, scales_out, zeros = _repack(1, qweight, qzeros, scales, n, k, group, perm, zero_offset)
    return codes, scales_out, zeros, perm


def hardware_decode(device=None) -> bool:
    """Whether the CUDA GEMV's code for `device` decodes E2M1 by the hardware conversion (architecture-specific
    targets from sm_100a on) rather than by the bit pattern; both give the same bits."""
    device = torch.device(device or "cuda")
    return bool(ops.hardware_decode(torch.empty(0, device=device)))
