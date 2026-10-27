// Stored-tile low-precision projections: FP8 W8A8 and NVFP4 W4A4 prompt GEMMs on tensor cores and a 1-16 row
// decode / verify GEMV with BF16 activations, all read straight from one value-preserving weight layout.
//
// Weight tiles (written once at load by repack_fp8 / repack_fp4, every value unchanged):
//   FP8   16 rows x 32 columns = 512 bytes. Lane l = 4 g + t (g = l / 4, t = l % 4) owns 16 bytes:
//         for each 16-column half s: (row g, cols 16 s + 2 t, +1), (row g + 8, same), (row g, cols + 8, + 9),
//         (row g + 8, same). Row scales stay a separate BF16 vector.
//   NVFP4 16 rows x 64 columns = 512 bytes of E2M1 codes (low nibble first). For each 16-column scale group s:
//         (row g, cols 16 s + 2 t, +1), (row g + 8, same), (row g, cols + 8, + 9), (row g + 8, same) as bytes.
//         Scale tile of 64 bytes: [g][row g: groups 0..3][row g + 8: groups 0..3], E4M3.
// Each tile byte order is the A fragment of mma.m16n8k16 for 16-bit operands (what the GEMV decodes to BF16 and
// multiplies), so the decode GEMV and the prompt GEMMs read the same bytes.
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
// Requirements: the FP8 prompt GEMMs need sm_89+; the NVFP4 prompt GEMMs need the block-scaled MMA of sm_120a /
// sm_121a (compile with an arch-specific target, e.g. -arch=sm_120a); the GEMV needs sm_80+. Prompt GEMM shapes: N a
// multiple of 128 (256 with RT = 2), K a multiple of 32 (FP8) or 64 (NVFP4). Accumulation is FP32 in a fixed order
// per (row, token) for a given plan.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace crit::stored_tile {

// Launch geometry and the scratch a GEMM needs, for one format and shape (fp4, n, k) and at most `tokens` tokens.
struct Plan {
    bool fp4 = false;
    int n = 0, k = 0, tokens = 0;
    int nt = 8, rt = 1, sk = 1, tpad = 0;
    int group = 1;                  // weight bands per raster group (l2_raster_group); 1 is the plain launch order
    // Scratch the caller provides: packed activations, per-token scales (FP8) and split-K partials.
    std::size_t activation_bytes = 0, token_scale_count = 0, partial_floats = 0;
};

// Token groups per block: 16 (128 tokens) above `wide_tokens`, else 8; two row tiles per warp when N allows.
// min_blocks > 0: split K until the grid holds at least `min_blocks` blocks (at most 8 parts). min_blocks <= 0 (the
// default): split K only while the split grid still fits in one wave of resident blocks, the current device's SM
// count times the occupancy of the kernel the plan launches (at most 8 parts), so the split fills SMs the unsplit
// grid leaves idle and never adds a wave or partials beyond it. wide_tokens = 256 was measured on one sm_120 device
// and is a starting point for the engine to measure on its own. The split changes the accumulation order, so the
// plan (and the bits of the result) can differ between devices.
// A plan serves launches of its own format and N, K with 1..T tokens; scratch sized from it covers that range. A
// launch with T tokens packs and computes ceil(T / (8 nt)) token blocks, so a short chunk costs what it uses, and a
// token's result depends only on the plan and its own row, not on T.
Plan plan(bool fp4, int N, int K, int T, int wide_tokens = 256, int min_blocks = 0, bool two_row_tiles = true);
// Raster grouping from the current device's L2 size (cudaDevAttrL2CacheSize): 1 while the plan's packed activations
// (tokens x K, at the plan's activation width) fit in half the L2, else the largest power of two G of weight bands
// (128 rt rows x K each) whose G bands fit in a quarter of it. The launch order then keeps G bands L2-resident while
// the token blocks pass over them, which reads the activations once per G bands. The result's bits are unchanged.
// Plans start at group 1: set it (plan.group = l2_raster_group(plan)) where a measurement on the device shows a gain.
int l2_raster_group(const Plan&);

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
// The same repacks straight into place in a larger matrix with the same K: source tile row r (rows 16 r .. 16 r + 15)
// becomes destination tile row first + step * r, where tiles / scale_tiles point at the larger matrix's tile row 0.
// first = (rows before) / 16 with step 1 concatenates along N (for example q | k | v); first 0 and 1 with step 2
// interleave two matrices' 16-row tiles (gate | up for the gated launches below). enqueue_repack_fp8 / _fp4 are
// first 0, step 1. Values unchanged; first >= 0, step >= 1.
cudaError_t enqueue_repack_fp8_rows(const std::uint8_t* codes, std::uint8_t* tiles, int N, int K, int first, int step,
                                    cudaStream_t stream);
cudaError_t enqueue_repack_fp4_rows(const std::uint8_t* codes, const std::uint8_t* scales, std::uint8_t* tiles,
                                    std::uint8_t* scale_tiles, int N, int K, int first, int step, cudaStream_t stream);
// FP8 row scales in the same placement: dst[(first + step * (n / 16)) * 16 + n % 16] = src[n] for n < N (N % 16 == 0).
cudaError_t enqueue_place_row_scales(const __nv_bfloat16* src, __nv_bfloat16* dst, int N, int first, int step,
                                     cudaStream_t stream);

// Y[t][n] (row stride ldo) = (add ? Y : 0) + sum_k W[n][k] x[t][k] with each format's dequantization; x is BF16
// [T][K] row-major. Returns cudaErrorInvalidValue for unsupported shapes and for a plan made for another format,
// N or K, or for fewer than T tokens (its scratch would overflow); the NVFP4 path traps on a device without the
// block-scaled MMA (sm_120a / sm_121a builds only).
cudaError_t enqueue_fp8(const Fp8Weights& w, const __nv_bfloat16* x, int T, float* y, int ldo, bool add,
                        const Plan& p, const Scratch& s, cudaStream_t stream);
cudaError_t enqueue_fp4(const Fp4Weights& w, const __nv_bfloat16* x, int T, float* y, int ldo, bool add,
                        const Plan& p, const Scratch& s, cudaStream_t stream);

// Gated MLP activation: out = bf16(act(gate) * up), act and the product in FP32 (divisions by __fdividef).
//   silu       v / (1 + exp(-v))
//   gelu_tanh  0.5 v (1 + tanh(u)), u = sqrt(2 / pi) (v + 0.044715 v^3), computed as v / (1 + exp(-2 u))
//   gelu       0.5 v (1 + erf(v / sqrt(2))), computed as 0.5 v erfc(-v / sqrt(2))
// (the computed forms are the same functions without the cancellation of 1 + tanh and 1 + erf for negative v).
enum class Activation : int { silu = 0, gelu_tanh = 1, gelu = 2 };

// Gated MLP input: W holds gate and up interleaved in 16-row tiles [gate rows 0-15, up rows 0-15, gate rows 16-31,
// ...] (N = 2 I, FP8 row scales placed alike; enqueue_repack_*_rows with step 2 builds it), and
// out[t][j] (row stride ldo >= I) = bf16(act(gate[t][j]) * up[t][j]) for j < I, where gate and up are exactly the
// values enqueue_fp8 / enqueue_fp4 would write for those rows under the same plan. No FP32 [T][N] output is written;
// a split plan (sk > 1) writes its partials as the plain launch does and the finish applies the activation. Needs
// a plan with two row tiles per warp (rt == 2, N % 256 == 0). NVFP4: the activations are quantized once, so gate
// and up share input_global_scale; w.weight_global_scale is gate's divisor and up_weight_global_scale up's.
cudaError_t enqueue_fp8_gated(const Fp8Weights& w, const __nv_bfloat16* x, int T, Activation act, __nv_bfloat16* out,
                              int ldo, const Plan& p, const Scratch& s, cudaStream_t stream);
cudaError_t enqueue_fp4_gated(const Fp4Weights& w, float up_weight_global_scale, const __nv_bfloat16* x, int T,
                              Activation act, __nv_bfloat16* out, int ldo, const Plan& p, const Scratch& s,
                              cudaStream_t stream);

// True when this build contains the NVFP4 tensor-core kernel (compiled for sm_120a or sm_121a).
bool fp4_mma_compiled();

// ---- Decode / verify GEMV over the same tiles, BF16 activations, M = 1..16 rows (sm_80+) ----
// y[m][n] (row stride ldy) = (add ? y : 0) + sum_k W[n][k] x[m][k], x BF16 [M][ldx]. Weights are decoded exactly to
// BF16 in registers (E4M3; E2M1 x E4M3 group scale) and multiplied on BF16 tensor cores (mma.m16n8k16) with FP32
// accumulation; the FP8 row scale (a multiply) or the NVFP4 weight_global_scale (a division) is applied after the
// reduction. No activation quantization.
// A block owns row pairs (two 16-row tiles, 32 rows); `groups` warps of a row pair split K: warp q takes K tiles q,
// q + groups, q + 2 groups, ... (32 FP8 / 64 NVFP4 columns each) and chains their MMAs in that order, then the warps'
// FP32 partials are summed in warp order. Rows 8 .. 15 use a second MMA whose inputs do not touch rows 0 .. 7, so
// each row's result is bitwise independent of M and of the other rows (a verify row equals plain decode), and a
// plan's launches are deterministic.
// What a launch is for, in its kernel's name: launches of a plan with role draft run the same arithmetic (the same
// bits) as a kernel instantiated with DraftRole, so a kernel trace (crit eval kernels) tells a drafter's GEMVs from
// the target's verify and decode GEMVs by name ("draft" in it) without any engine bookkeeping.
enum class KernelRole { target, draft };
struct TargetRole {};
struct DraftRole {};

struct GemvPlan {
    bool fp4 = false;
    int n = 0, k = 0;
    int groups = 0;  // 4, 8 or 16 warps per row pair
    KernelRole role = KernelRole::target;  // set draft for a drafter's launches
};
// groups <= 0 derives it from N and the current device (see the README); the reduction order then depends on the
// device's SM count, as a GEMM plan's split does. Pass groups explicitly to fix the bits across devices.
GemvPlan plan_gemv(bool fp4, int N, int K, int groups = 0);
// Requirements (cudaErrorInvalidValue otherwise): a plan for this format, N and K; N % 16 == 0 and K % 32 (FP8) / 64
// (NVFP4); 1 <= M <= 16; ldx >= K and even, x 4-byte aligned; ldy >= N; tiles 16-byte and scale tiles 8-byte
// aligned; y must not overlap x or the weights.
cudaError_t enqueue_gemv_fp8(const Fp8Weights& w, const __nv_bfloat16* x, int ldx, int M, float* y, int ldy, bool add,
                             const GemvPlan& p, cudaStream_t stream);
cudaError_t enqueue_gemv_fp4(const Fp4Weights& w, const __nv_bfloat16* x, int ldx, int M, float* y, int ldy, bool add,
                             const GemvPlan& p, cudaStream_t stream);
// Gated MLP (gate | up interleaved in 16-row tiles, N = 2 I, N % 32 == 0): out[m][j] (row stride ldo >= I) =
// bf16(act(gate[m][j]) * up[m][j]), gate and up bitwise equal to the plain GEMV's rows under the same plan. NVFP4:
// w.weight_global_scale divides gate, up_weight_global_scale up (input_global_scale is not used: no activation
// quantization).
cudaError_t enqueue_gemv_fp8_gated(const Fp8Weights& w, const __nv_bfloat16* x, int ldx, int M, Activation act,
                                   __nv_bfloat16* out, int ldo, const GemvPlan& p, cudaStream_t stream);
cudaError_t enqueue_gemv_fp4_gated(const Fp4Weights& w, float up_weight_global_scale, const __nv_bfloat16* x, int ldx,
                                   int M, Activation act, __nv_bfloat16* out, int ldo, const GemvPlan& p,
                                   cudaStream_t stream);

// ---- Prompt GEMMs with BF16 activations (W8A16 / W4A16) over the same tiles (sm_80+) ----
// The GEMV's arithmetic for any number of tokens: weights decoded exactly to BF16 in registers, BF16 mma.m16n8k16
// with FP32 accumulation, the FP8 row scale (a multiply) or NVFP4 weight_global_scale (a division) after the
// reduction, no activation quantization (input_global_scale is not used). For checkpoints whose quality budget cannot
// afford W8A8 / W4A4 activation quantization. Not bitwise equal to the GEMV (another K order), equally accurate.
// A block covers 256 rows x 64 tokens; per (row, token) the MMAs chain over 512 columns in K order and each chain is
// folded into an FP32 running sum in K order; split parts are summed in part order. The order is fixed per plan and a
// token's bits do not depend on T.
struct A16Plan {
    bool fp4 = false;
    int n = 0, k = 0, tokens = 0;
    int sk = 1;                     // K parts
    int group = 1;                  // weight bands per raster group (l2_raster_group); 1 is the plain launch order
    std::size_t partial_floats = 0;  // scratch for sk > 1: sk x tokens x n floats
};
// min_blocks > 0: split K until the grid holds min_blocks blocks (at most 8 parts); min_blocks <= 0: split only while
// the split grid fits in one wave of resident blocks (SM count x the kernel's occupancy), as plan() does.
A16Plan plan_a16(bool fp4, int N, int K, int T, int min_blocks = 0);
// The same rule for BF16 activations (2 bytes per value) and 256-row bands.
int l2_raster_group(const A16Plan&);
// y[t][n] (row stride ldo) = (add ? y : 0) + sum_k W[n][k] x[t][k], x BF16 [T][K] row-major and 16-byte aligned.
// N % 16 == 0 (rows past a 256-row block are skipped), K % 32 (FP8; a K % 64 == 32 tail is handled) or K % 64
// (NVFP4); T <= plan tokens; partials: plan.partial_floats floats (may be null when sk == 1). cudaErrorInvalidValue
// otherwise, including for another format's or shape's plan.
cudaError_t enqueue_fp8_a16(const Fp8Weights& w, const __nv_bfloat16* x, int T, float* y, int ldo, bool add,
                            const A16Plan& p, float* partials, cudaStream_t stream);
cudaError_t enqueue_fp4_a16(const Fp4Weights& w, const __nv_bfloat16* x, int T, float* y, int ldo, bool add,
                            const A16Plan& p, float* partials, cudaStream_t stream);
// Gated MLP (gate | up interleaved in 16-row tiles, N = 2 I, N % 32 == 0): out[t][j] (row stride ldo >= I) =
// bf16(act(gate[t][j]) * up[t][j]), gate and up exactly the values enqueue_*_a16 writes for those rows; no FP32
// [T][N] output (a split plan's partials are summed and activated by the finish kernel).
cudaError_t enqueue_fp8_a16_gated(const Fp8Weights& w, const __nv_bfloat16* x, int T, Activation act,
                                  __nv_bfloat16* out, int ldo, const A16Plan& p, float* partials, cudaStream_t stream);
cudaError_t enqueue_fp4_a16_gated(const Fp4Weights& w, float up_weight_global_scale, const __nv_bfloat16* x, int T,
                                  Activation act, __nv_bfloat16* out, int ldo, const A16Plan& p, float* partials,
                                  cudaStream_t stream);

}  // namespace crit::stored_tile
