// Published 4-bit weight checkpoints with BF16 activations (W4A16) on CUDA: GPTQ, AWQ and compressed-tensors int4
// groups and MXFP4 (gpt-oss) converted once to one canonical layout, a decode GEMV for up to 8 tokens and a prompt GEMM
// on BF16 tensor cores (sm_80+). See w4.py for the exact unpacking of every format.
//
// Canonical layout of an N x K weight: codes [N][K / 2] bytes (the even column in the low nibble), and either
//   int4 groups: value = (q - zero) * scale, FP32 scales and uint8 zeros [N][K / group], group 32, 64 or 128; or
//   MXFP4:       value = e2m1(q) * 2^(e - 127), one E8M0 byte e per row and 32 columns, [N][K / 32].
// An optional perm [K] (GPTQ act-order) says canonical column j is checkpoint column perm[j]; gather activations with
// enqueue_gather_columns before the projection.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace crit::w4a16 {

enum class Codec : int { int4_groups = 0, mxfp4 = 1 };

struct Weights {
    Codec codec = Codec::int4_groups;
    const std::uint8_t* codes = nullptr;   // [N][K / 2]
    const float* scales = nullptr;         // int4_groups: [N][K / group]
    const std::uint8_t* zeros = nullptr;   // int4_groups: [N][K / group]
    const std::uint8_t* exponents = nullptr;  // mxfp4: [N][K / 32] E8M0
    int N = 0, K = 0, group = 0;           // group: 32, 64 or 128 for int4 groups (32 for MXFP4)
};

// Decode and verify: y[t][n] (+)= sum_j w[n][j] x[t][j] for T <= 8 tokens, FP32, one warp per output row walking K in a
// fixed order, so each token's result does not depend on T (a column alone equals the same column in a verify).
// K a multiple of 1024 / 32 = 32 per lane step; x rows 16-byte aligned (x_stride a multiple of 8).
cudaError_t enqueue_gemv(const Weights& w, const __nv_bfloat16* x, int x_stride, int T, float* y, int y_stride, bool add,
                         cudaStream_t stream);

// Prompts: the same product for any T on BF16 mma.m16n8k16, weights dequantized to BF16 per 64 x 64 tile in shared
// memory. N a multiple of 64, K a multiple of 64.
cudaError_t enqueue_gemm(const Weights& w, const __nv_bfloat16* x, int x_stride, int T, float* y, int y_stride, bool add,
                         cudaStream_t stream);

// x'[t][j] = x[t][perm[j]] for act-order checkpoints.
cudaError_t enqueue_gather_columns(const __nv_bfloat16* x, int x_stride, const std::int32_t* perm, int K, int T,
                                   __nv_bfloat16* out, int out_stride, cudaStream_t stream);

// ---- conversion to the canonical layout (once, at load) --------------------------------------------------------
// GPTQ: qweight int32 [K / 8][N], qzeros int32 [G][N / 8], scales FP16 [G][N], optional g_idx [K]. With g_idx the
// columns are sorted by group (stably) and perm [K] receives the order; zero_offset is 1 for checkpoint_format "gptq"
// (it stores zero - 1) and 0 for "gptq_v2". Uses sorted_scratch of K int32 when g_idx is set.
struct Gptq {
    const std::int32_t* qweight = nullptr;
    const std::int32_t* qzeros = nullptr;
    const __half* scales = nullptr;
    const std::int32_t* g_idx = nullptr;
    int zero_offset = 1;
};
cudaError_t enqueue_from_gptq(const Gptq& in, int N, int K, int group, std::uint8_t* codes, float* scales, std::uint8_t* zeros,
                              std::int32_t* perm, cudaStream_t stream);
// AWQ (GEMM packing): qweight int32 [K][N / 8], qzeros int32 [G][N / 8] packed along N in order 0, 2, 4, 6, 1, 3, 5, 7;
// scales FP16 [G][N].
cudaError_t enqueue_from_awq(const std::int32_t* qweight, const std::int32_t* qzeros, const __half* scales, int N, int K, int group,
                             std::uint8_t* codes, float* out_scales, std::uint8_t* zeros, cudaStream_t stream);
// compressed-tensors pack-quantized: weight_packed int32 [N][K / 8] (unsigned codes), weight_scale BF16 [N][G], zero_point
// int32 [N][G] or null (symmetric: zero 8).
cudaError_t enqueue_from_compressed(const std::int32_t* packed, const __nv_bfloat16* scale, const std::int32_t* zero_point, int N,
                                    int K, int group, std::uint8_t* codes, float* out_scales, std::uint8_t* zeros, cudaStream_t stream);
// MXFP4 blocks [N][K / 32][16] and scales [N][K / 32] are already canonical: codes = blocks, exponents = scales.

}  // namespace crit::w4a16
