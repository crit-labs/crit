// Stored-tile low-precision prompt GEMMs: FP8 W8A8 and NVFP4 W4A4 on tensor cores, read straight from one
// value-preserving weight layout that a 1-8 token GEMV can share.
//
// Weight tiles (written once at load by repack_fp8 / repack_fp4, every value unchanged):
//   FP8   16 rows x 32 columns = 512 bytes. Lane l = 4 g + t (g = l / 4, t = l % 4) owns 16 bytes:
//         for each 16-column half s: (row g, cols 16 s + 2 t, +1), (row g + 8, same), (row g, cols + 8, + 9),
//         (row g + 8, same). Row scales stay a separate BF16 vector.
//   NVFP4 16 rows x 64 columns = 512 bytes of E2M1 codes (low nibble first). For each 16-column scale group s:
//         (row g, cols 16 s + 2 t, +1), (row g + 8, same), (row g, cols + 8, + 9), (row g + 8, same) as bytes.
//         Scale tile of 64 bytes: [g][row g: groups 0..3][row g + 8: groups 0..3], E4M3.
// Each tile byte order is the A fragment of mma.m16n8k16 for FP16 operands (what a GEMV converts and multiplies),
// so the decode GEMV and these prompt GEMMs read the same bytes.
//
// FP8 GEMM: mma.m16n8k32 e4m3 wants k = 4 t .. 4 t + 3 per lane. Within each 16-column half the k order is
// permuted so that a lane's tile bytes are its fragment: slot 4 t + i holds column 2 t + (i < 2 ? i : 6 + i),
// assembled with byte permutes; activations are packed in the same order (per-token E4M3, scale amax / 448).
// NVFP4 GEMM: mma.m16n8k64 kind::mxf4nvf4 scale_vec::4X wants each 16-value scale group of a row inside one lane
// pair (lanes t >> 1 equal), while the tile spreads a group over the four lanes of a quad. Lanes t and t ^ 2 swap
// one word per register pair; slot (lane t, register r, nibble i) then holds column
// 16 s + 2 t' + (i & 3 < 2 ? i & 3 : 6 + (i & 3)), s = (t >> 1) + 2 r, t' = i < 4 ? t : t ^ 2.
// Activations use NVFP4 input quantization with a static global scale gx (scale e4m3(amax gx / 6) per 16 values).
//
// Requirements: FP8 kernels need sm_89+; NVFP4 kernels need the block-scaled MMA of sm_120a / sm_121a (compile
// with an arch-specific target, e.g. -arch=sm_120a). Shapes: N a multiple of 128 (256 with RT = 2), K a multiple
// of 32 (FP8) or 64 (NVFP4). Accumulation is FP32 in a fixed order per (row, token) for a given launch shape.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace crit::stored_tile {

// Launch geometry and the scratch a GEMM needs.
struct Plan {
    int nt = 8, rt = 1, sk = 1, tpad = 0;
    // Scratch the caller provides: packed activations, per-token scales (FP8) and split-K partials.
    std::size_t activation_bytes = 0, token_scale_count = 0, partial_floats = 0;
};

// Token groups per block: 16 (128 tokens) above `wide_tokens`, else 8; two row tiles per warp when N allows.
// Split K until the grid holds at least `min_blocks` blocks (at most 8 parts).
Plan plan(bool fp4, int N, int K, int T, int wide_tokens = 256, int min_blocks = 340, bool two_row_tiles = true);

struct Fp8Weights { const std::uint8_t* tiles; const __nv_bfloat16* row_scale; int N, K; };
// Dequantization: weight = code * scale / weight_global_scale (the checkpoint's divisor); activations are quantized
// with the static input_global_scale gx: scale = e4m3(amax * gx / 6), code = e2m1(x * gx / scale).
struct Fp4Weights { const std::uint8_t* tiles; const std::uint8_t* scale_tiles; float weight_global_scale, input_global_scale; int N, K; };
struct Scratch { void* activations; float* token_scale; float* partials; };

// Repack row-major stored weights into tiles (values unchanged). FP8: codes [N][K] -> tiles [N/16][K/32][512].
// NVFP4: packed codes [N][K/2] and E4M3 scales [N][K/16] -> tiles [N/16][K/64][512] and scale tiles [N/16][K/64][64].
cudaError_t enqueue_repack_fp8(const std::uint8_t* codes, std::uint8_t* tiles, int N, int K, cudaStream_t stream);
cudaError_t enqueue_repack_fp4(const std::uint8_t* codes, const std::uint8_t* scales, std::uint8_t* tiles,
                               std::uint8_t* scale_tiles, int N, int K, cudaStream_t stream);

// Y[t][n] (row stride ldo) = (add ? Y : 0) + sum_k W[n][k] x[t][k] with each format's dequantization; x is BF16
// [T][K] row-major. Returns cudaErrorInvalidValue for unsupported shapes; the NVFP4 path traps on a device without
// the block-scaled MMA (sm_120a / sm_121a builds only).
cudaError_t enqueue_fp8(const Fp8Weights& w, const __nv_bfloat16* x, int T, float* y, int ldo, bool add,
                        const Plan& p, const Scratch& s, cudaStream_t stream);
cudaError_t enqueue_fp4(const Fp4Weights& w, const __nv_bfloat16* x, int T, float* y, int ldo, bool add,
                        const Plan& p, const Scratch& s, cudaStream_t stream);

// True when this build contains the NVFP4 tensor-core kernel (compiled for sm_120a or sm_121a).
bool fp4_mma_compiled();

}  // namespace crit::stored_tile
