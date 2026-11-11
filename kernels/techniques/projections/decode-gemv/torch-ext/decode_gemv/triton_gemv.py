"""The Triton side of decode-gemv: an NVFP4 W4A16 GEMV (1 to 8 rows, its own order: the format-generic Triton GEMV's
fma path, structure gemv-nvfp4-triton an alias of gemv-triton) and the decoders that write a matrix's values for a
provider's GEMM (BF16/FP16, or FP8 rows in units of a row scale).

An E2M1 code's FP16 bit pattern (sign at bit 15, magnitude bits at 11..9) is its value times 2^-14; the kernels build
values that way and fold the 2^14 into the row factor. On a device with the hardware conversion (compute capability 10
and later: cvt.rn.f16x2.e2m1x2) the GEMV decodes with it instead, four bytes an instruction; both give the exact values,
so the results are the same bits either way.
"""
import torch
import triton
import triton.language as tl

from ._crit_tuning import knobs

from .triton_any import BLOCK_K, GEMV_TILES, _HW, fma, hardware

MAX_ROWS = 8


@triton.jit
def _e2m1(codes):
    bits = ((codes & 7).to(tl.uint16) << 9) | ((codes & 8).to(tl.uint16) << 12)
    return bits.to(tl.float16, bitcast=True).to(tl.float32)


def gemv_triton(x, codes, scales, inverse, out):
    """x [M][K] (M <= 8, unit column stride) times the NVFP4 matrix (codes [N][K / 2] uint8, scales [N][K / 16] E4M3,
    inverse [N] FP32) -> out [M][N] (any float dtype, unit column stride), FP32 accumulation. The fma path of the
    format-generic Triton GEMV (triton_any, structure gemv-triton) on NVFP4 with gemv-nvfp4-triton's knobs: the same
    operations, so the same bits, as this GEMV had."""
    M, K = x.shape
    N = codes.shape[0]
    if not 1 <= M <= MAX_ROWS or K % 16 or codes.shape[1] * 2 != K or x.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError(f"gemv_triton: x [1..{MAX_ROWS}][K] (K a multiple of 16, unit column stride) for codes "
                         f"[N][K / 2], not x {tuple(x.shape)} and codes {tuple(codes.shape)}")
    tuned = knobs("gemv-nvfp4-triton", {"N": N, "K": K}, M)
    # BLOCK_N 0: the autotuner's tile, as before the knobs
    return fma(0, x, codes, scales.view(torch.uint8), None, inverse, out, 0, tuned["BLOCK_N"], tuned["BLOCK_K"],
               tuned["WARPS"], tuned["STAGES"])


@triton.jit
def _decode_kernel(w_ptr, s_ptr, f_ptr, out_ptr, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    k0 = tl.program_id(1) * BLOCK_K
    HALF: tl.constexpr = BLOCK_K // 2
    j = tl.arange(0, HALF)
    kmask = k0 + 2 * j < K
    packed = tl.load(w_ptr + row * (K // 2) + k0 // 2 + j, mask=kmask, other=0)
    scale = tl.load(s_ptr + row * (K // 16) + (k0 + 2 * j) // 16, mask=kmask, other=0.0).to(tl.float32)
    f = tl.load(f_ptr + row) * 16384.0
    lo = _e2m1(packed) * scale * f
    hi = _e2m1(packed >> 4) * scale * f
    offs = k0 + tl.arange(0, BLOCK_K)
    tl.store(out_ptr + row * K + offs, tl.interleave(lo, hi).to(out_ptr.dtype.element_ty), mask=offs < K)


def decode(codes, scales, inverse, out):
    """The matrix's values [N][K] in out's dtype (16-bit, contiguous): e2m1 * scale * inverse, one FP32 product
    rounded once."""
    N, K = out.shape
    _decode_kernel[(N, triton.cdiv(K, 512))](codes, scales, inverse, out, K, BLOCK_K=512, num_warps=4)
    return out


@triton.jit
def _decode_fp8_kernel(w_ptr, s_ptr, f_ptr, out_ptr, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    k0 = tl.program_id(1) * BLOCK_K
    HALF: tl.constexpr = BLOCK_K // 2
    j = tl.arange(0, HALF)
    kmask = k0 + 2 * j < K
    packed = tl.load(w_ptr + row * (K // 2) + k0 // 2 + j, mask=kmask, other=0)
    scale = tl.load(s_ptr + row * (K // 16) + (k0 + 2 * j) // 16, mask=kmask, other=0.0).to(tl.float32)
    f = tl.load(f_ptr + row) * 16384.0
    lo = tl.clamp(_e2m1(packed) * scale * f, -448.0, 448.0)
    hi = tl.clamp(_e2m1(packed >> 4) * scale * f, -448.0, 448.0)
    offs = k0 + tl.arange(0, BLOCK_K)
    tl.store(out_ptr + row * K + offs, tl.interleave(lo, hi).to(tl.float8e4nv), mask=offs < K)


def decode_fp8(codes, scales, factor, out):
    """The matrix's values [N][K] as FP8 E4M3 (out contiguous float8_e4m3fn) in units of each row's scale: e2m1 *
    scale * factor[row], rounded once to nearest even within +-448."""
    N, K = out.shape
    _decode_fp8_kernel[(N, triton.cdiv(K, 512))](codes, scales, factor, out, K, BLOCK_K=512, num_warps=4)
    return out
