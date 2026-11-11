// Projections of 1 to 64 activation rows on tensor cores, one fixed tile: an NVFP4, FP8, MXFP8 or BF16 matrix [N][K]
// times rows [M][K] of BF16 activations, used as they are (weight only: W4A16, W8A16) or quantized as a prompt's
// activations are (FP8 per row: W8A8; NVFP4 per 16-value block: W4A4), FP32 sums, out [M][N].
//
// The weights are the A operand of mma.m16n8k16 (BF16 in, FP32 accumulate), 16 matrix rows at a time; the activation
// rows are its B operand, 8 at a time. Every value enters the MMA exactly: an E2M1 code times its E4M3 block scale,
// an E4M3 code (times its E8M0 power of two for MXFP8; the FP8 row scale applies after the sum), a BF16 weight; a BF16
// activation, or its quantized value (an E4M3 code, or an E2M1 code times the block's E4M3 scale; the row scale or the
// input scale applies after the sum).
//
// A CTA's `warps` warps each take 16 x `groups` matrix rows over the CTA's share of K: K goes in steps of 256 bytes a
// matrix row (step_values: the four lanes of a quad read 4 x kLaneBytes contiguous bytes of it), and
// the steps split into `splits` contiguous parts, one a CTA (grid: row tiles x splits; the warps of a CTA read the same
// activations, which the L1 serves). With splits > 1 each CTA writes its partial sums to `partial` and the last CTA
// of a row tile to finish (a counter in `counters`, reset by it) adds them in split order. A sum's order depends on N,
// K and `splits` alone, never on M, `groups`, `warps` or the rows beside it: a row gets the same bits alone, in a
// decode call or in a prompt's tile. Each weight is read once a call.
//
// Quantized activations come from a pre-pass that writes their values (BF16, exact) and the FP8 row scales into
// `scratch`; the projection is launched as its programmatic dependent (compute capability 9.0 on), so it streams its
// first weights while the pre-pass runs and waits for it before reading the activations. Nothing allocates and
// nothing waits for the host: a CUDA graph captures a call (calls sharing `partial`, `counters` or `scratch` run in
// stream order).
//
// Weights (codes row major, one row after another):
//   NVFP4  codes [N][K / 2] bytes (element 2i in the low nibble of byte i), scales E4M3 [N][K / 16] (or cuBLASLt's
//          128 x 4 tiled layout: ct > 0 tiles of 4 blocks a row of tiles), row_scale [N] FP32: one over the row's
//          global scale.
//   FP8    codes [N][K] E4M3, row_scale [N] FP32.
//   MXFP8  codes [N][K] E4M3, scales E8M0 [N][K / 32] (2^(e - 127); 255 is NaN).
//   BF16   w [N][K].
//   INT4ZP INT4 with a zero point a group (AWQ, GPTQ in crit's rows: decode-gemv's repack_awq / repack_gptq): codes
//          [N][K / 2] unsigned (element 2i in the low nibble of byte i), scales FP16 [N][K / group], zeros [N][K /
//          group] bytes 0..16. A value (code - zero) * scale does not fit BF16: the decoder gives it as two BF16
//          parts, hi + lo, and each MMA runs twice, hi then lo, so every product is exact. group a multiple of 32
//          dividing K; BF16 activations only.
// Activations: x [M][K] BF16, rows ldx elements apart (16-byte aligned), K a multiple of 32.
//   kActBf16     x as it is.
//   kActFp8Rows  each row r in E4M3 by its own scale s_r = max(amax_r * fp32(1 / 448), 1e-12): q = e4m3(clamp(x / s_r,
//                -448, 448)); out = (sum * row_scale[n]) * s_r.
//   kActNvfp4    each 16-value block in E2M1 under input scale g: block scale b = e4m3(min((amax * fp32(1 / 6)) * g,
//                448)), d = b * fp32(1 / g), code = nearest E2M1 magnitude of x / d (ties to the even code; 0 when d is
//                0), value = e2m1(code) * b; out = (sum * row_scale[n]) * fp32(1 / g).
// The quantizers are those of quantize_fp8_rows and quantize_nvfp4 (the same bits as the prompt rows' quantization).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace crit::kernels::mma_rows {

constexpr int kMaxRows = 64;     // activation rows a call takes (8 MMA columns at a time)
constexpr int kMaxWarps = 4;     // warps a CTA (decode tiles keep four CTAs a multiprocessor)
constexpr int kMaxSplits = 32;
constexpr int kStep = 32;        // K is a multiple of it (a lane's sub-step)
#ifndef CRIT_MMA_LANE_BYTES
#define CRIT_MMA_LANE_BYTES 64
#endif
constexpr int kLaneBytes = CRIT_MMA_LANE_BYTES;   // a matrix row's contiguous bytes a lane reads a step

enum Weights : int { kNvfp4 = 0, kFp8 = 1, kMxfp8 = 2, kBf16 = 3, kInt4Zp = 4 };
// kActNvfp4Mma: the activations quantized to NVFP4 as kActNvfp4, multiplied on the block-scaled FP4 MMA (E2M1 x E2M1
// with each operand's UE4M3 scale per 16 along K) instead of as BF16 values: NVFP4 weights with blocked scales, K a
// multiple of 128, devices with the MMA (fp4_mma()).
enum Acts : int { kActBf16 = 0, kActFp8Rows = 1, kActNvfp4 = 2, kActNvfp4Mma = 3 };
enum Out : int { kOutF32 = 0, kOutBf16 = 1, kOutF16 = 2 };

// K values a step covers (4 x kLaneBytes of a matrix row): part of the sums' order.
__host__ __device__ constexpr int step_values(int weights) {
  return 4 * kLaneBytes * (weights == kNvfp4 || weights == kInt4Zp ? 2 : 1) / (weights == kBf16 ? 2 : 1);
}
// The tile's rows a CTA takes.
__host__ __device__ constexpr int tile_rows(int groups, int warps) { return 16 * groups * warps; }
// Bytes of `scratch` quantized activations need (values, then the FP8 row scales), and FP32 `partial` elements.
inline std::size_t scratch_bytes(int M, int K) { return ((std::size_t)M * K * 2 + 15) / 16 * 16 + (std::size_t)M * 4; }
inline std::size_t partial_elements(int M, int N, int splits) { return splits > 1 ? (std::size_t)splits * M * N : 0; }
inline int row_tiles(int N, int groups, int warps) { return (N + tile_rows(groups, warps) - 1) / tile_rows(groups, warps); }

struct Problem {
  const void* x = nullptr;
  int64_t ldx = 0;
  const std::uint8_t* w = nullptr;
  const std::uint8_t* scales = nullptr;
  const float* row_scale = nullptr;
  float input_scale = 1.f, input_rcp = 1.f;   // kActNvfp4: g and fp32(1 / g)
  void* out = nullptr;
  int64_t ldo = 0;
  int out_kind = kOutF32;
  int M = 0, N = 0, K = 0;
  int weights = kNvfp4, acts = kActBf16;
  int ct = 0;            // blocked NVFP4 scales (> 0) or row major (0)
  int groups = 1;        // 16-row groups a warp takes: 1, 2 or 4 (speed only)
  int warps = 4;         // warps a CTA (speed only)
  int splits = 1;        // contiguous parts of K's steps, one a CTA: part of the sums' order, fixed per matrix
  std::uint8_t* scratch = nullptr;   // quantized activations: scratch_bytes(M, K), 16-byte aligned
  float* partial = nullptr;          // splits > 1: partial_elements(M, N, splits)
  int* counters = nullptr;           // splits > 1: row_tiles(N, 1, warps) zeros (left zero; calls of more than 32
                                     // rows take one group a warp)
  const float* act_scale = nullptr;  // set by enqueue: the FP8 row scales in scratch
  const std::uint8_t* zeros = nullptr;   // kInt4Zp: zero points [N][K / group]
  int group = 0;                         // kInt4Zp: inputs a scale and zero point cover
};

cudaError_t enqueue(const Problem& p, cudaStream_t stream);
// CTAs of kMaxWarps warps a multiprocessor of the current device holds of the one-row kernel for `weights` and
// `acts` (its occupancy: registers and the launch bounds; the FP4 MMA's for kActNvfp4Mma, the decode tile's
// otherwise), which plan() fills with splits of K; 0 on an error.
int decode_ctas(int weights, int acts);
// The quantized activations' pre-pass of a call (acts other than kActBf16): their values (BF16, exact) and FP8 row
// scales into p.scratch, then p.x, p.ldx and p.act_scale point there. Nothing for kActBf16.
cudaError_t prepass(Problem& p, cudaStream_t stream);

// The activations' quantizers alone, for a provider's GEMM on the same activations (prompt rows):
// FP8 rows: q [M][K] E4M3 bytes, scale [M] FP32, as kActFp8Rows. NVFP4: codes [M][K / 2] (two per byte, low nibble
// first), block scales E4M3 in cuBLASLt's 128 x 4 tiled layout ([ceil(M / 128) * 128][ceil(K / 64) * 4], zero padded:
// the padding is written too), as kActNvfp4. x [M][K] BF16 (K a multiple of 16; FP8 rows: any K), any M.
cudaError_t enqueue_quantize_fp8_rows(const void* x, int64_t ldx, int M, int K, std::uint8_t* q, int64_t ldq,
                                      float* scale, cudaStream_t stream);
cudaError_t enqueue_quantize_nvfp4(const void* x, int64_t ldx, int M, int K, float input_scale, float input_rcp,
                                   std::uint8_t* codes, std::uint8_t* blocked_scales, cudaStream_t stream);

// The prompt tile (prompt_tile.cu): out [M][N] for any number of activation rows M >= 1, every element with the bits
// enqueue() gives it at any row count (the same Problem: weights, scales, acts, splits; `groups`, `warps`, `partial`
// and `counters` are not read), each decoded weight reused for kPromptTokens rows a CTA. Quantized activations take
// scratch_bytes(M, K) of scratch, as enqueue()'s pre-pass does.
constexpr int kPromptTokens = 64;   // activation rows a prompt-tile CTA takes, untuned (knob PT)
cudaError_t enqueue_prompt(const Problem& p, cudaStream_t stream);

}  // namespace crit::kernels::mma_rows
