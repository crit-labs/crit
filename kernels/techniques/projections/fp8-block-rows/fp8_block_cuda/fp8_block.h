// Matched FP8 projections: E4M3 weights (MXFP8's power-of-two block scales, an FP8 checkpoint's FP32 128 x 128 block
// scales, or one scale per row) against E4M3 activations quantized per group, in crit's declared per-output order
// (crit inference/quantized-projections, fp8_block_device.cuh): per 32-element block, the block's products summed by
// one e4m3 mma.sync m16n8k32 with a zero accumulator (weights the A operand, tokens B, crit's fragment order); lane
// b % 16 = fma(block sum, weight scale * activation scale, lane) with the product rounded, blocks in order; the 16 lanes
// reduced by the halving tree; times the matrix's post.
//
// Two kernels give the same bits for every output: `rows` for decode (16 rows a CTA, the lanes split over four warps,
// 1 to 8 tokens a job) and `gemm` for prompt chunks (a warp 16 rows x 32 tokens, the lanes folded as they finish). An
// output depends only on its row and its token, not on the job's other tokens or the kernel, so decode, verify rounds,
// prefill and a prompt cache agree bit for bit. The quantizer is crit's declared one (MXFP8's shared exponent or amax /
// 448, E4M3 round to nearest even with saturation). NVIDIA sm_89 and newer (e4m3 mma.sync).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace crit::techniques::fp8_block {

// One matrix: codes [rows][cols] E4M3; scales [ceil(rows / scale_rows)][cols / scale_k], FP32 or (scale_e8m0) MXFP8's
// E8M0 bytes as stored (MXFP8: 1, 32; 128 x 128 blocks: 128, 128; per row: 1, cols). cols and scale_k multiples of 32.
// An E8M0 byte e is the scale 2^(e - 127), the same value as its FP32 decoding, so both give the same bits.
struct Matrix {
    const std::uint8_t* codes = nullptr;
    const void* scales = nullptr;
    int rows = 0, cols = 0, scale_rows = 1, scale_k = 32;
    float post = 1.0f;
    int scale_e8m0 = 0;
};

// Quantized activations: codes [row][ld] E4M3, scales [row][lds] FP32, one per `group` codes (a multiple of 32).
struct Acts {
    const std::uint8_t* codes = nullptr;
    const float* scales = nullptr;
    int ld = 0, lds = 0, group = 32;
};

enum class ScaleKind : int { mx = 0, amax = 1 };

constexpr int kMaxJobTokens = 8;

// Decode: one matrix and up to 8 activation rows; out[out_row[j]][r] for its tokens j and rows r < rows.
struct RowJob {
    int matrix = 0;
    int tokens = 0;
    int act_row[kMaxJobTokens] = {};
    int out_row[kMaxJobTokens] = {};
};

// Prompt: one matrix and `count` consecutive activation rows from `first`; out[first + i][r].
struct GemmJob {
    int matrix = 0;
    int first = 0, count = 0;
};

// matrices: a device array of Matrix; jobs: a device array; out [..][out_ld] FP32. tokens: the call's tokens (a job's at
// most 8), which only picks the tuned configuration (0: the defaults); max_rows and acts.ld are the tuning's shape key.
cudaError_t enqueue_rows(const Matrix* matrices, const RowJob* jobs, int job_count, int max_rows, const Acts& acts,
                         float* out, int out_ld, cudaStream_t stream, int tokens = 0);
cudaError_t enqueue_gemm(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows, int max_count,
                         const Acts& acts, float* out, int out_ld, cudaStream_t stream);
// The same prompt GEMM with shared-memory staging (structure gemm-staged): the same bits as enqueue_gemm. Activation
// rows 16-byte aligned (acts.ld a multiple of 16).
cudaError_t enqueue_gemm_staged(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows,
                                int max_count, const Acts& acts, float* out, int out_ld, cudaStream_t stream);

// The prompt GEMM accumulated in the tensor cores (structure gemm-native): NOT the declared order. An output is within
// FP32 accumulation of the exact product, but its bits may depend on the configuration and the job's token count (not
// row invariant); for prompt chunks where speed matters more than matching the decode rows. mx states that the call is
// MXFP8 throughout: activation scales MXFP8's powers of two at group 32 (enqueue_quantize with ScaleKind::mx) and every
// matrix E8M0 per 32 of one row (scale_rows 1, scale_k 32) with cols a multiple of 128; an sm_120a device then runs
// block-scaled MMAs (a job that is not MXFP8 gets NaN outputs). mx false runs the generic path (an e4m3 MMA a block,
// the scales on the CUDA cores) for any scales, as every architecture without block-scaled MMAs does. Activation rows
// 16-byte aligned.
cudaError_t enqueue_gemm_native(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows,
                                int max_count, const Acts& acts, bool mx, float* out, int out_ld, cudaStream_t stream);
// The same into FP32 (out_bf16 false) or BF16 outputs (out_bf16: each FP32 sum times post rounded once to nearest
// even), out [..][out_ld] of that type.
cudaError_t enqueue_gemm_native(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows,
                                int max_count, const Acts& acts, bool mx, void* out, bool out_bf16, int out_ld,
                                cudaStream_t stream);

// The prompt GEMM of matrices whose scales do not change along K (structure gemm-native-channel; an FP8 checkpoint's
// per-channel W8A8): every matrix FP32 scales with scale_k == cols (one a row, a block of scale_rows rows or the
// tensor) and one activation scale a token (acts.lds 1, acts.group the width). Every block of K sums in the tensor
// cores' FP32 accumulator; the epilogue multiplies by activation scale x (weight scale x post) and rounds once to FP32
// or BF16 (out_bf16). Not the declared order, not row invariant, within FP32 accumulation of the exact product. A job
// whose matrix is not channel-scaled gets NaN outputs. Activation rows 16-byte aligned.
cudaError_t enqueue_gemm_native_channel(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows,
                                        int max_count, const Acts& acts, void* out, bool out_bf16, int out_ld,
                                        cudaStream_t stream);

// The declared quantizer: rows x cols values (x [row][ld_x], FP32 or BF16) into acts, one scale per acts.group.
cudaError_t enqueue_quantize(const void* x, bool bf16, int ld_x, int rows, int cols, ScaleKind kind, const Acts& acts,
                             cudaStream_t stream);

}  // namespace crit::techniques::fp8_block
