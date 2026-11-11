// Decode GEMVs on CUDA cores from crit's quantized-projections technique: weights stored as NVFP4, FP8 (E4M3 with a
// row scale, or MXFP8's E8M0 scale per 32 inputs) or BF16 times 1 to 16 rows of 16-bit activations (weight only:
// W4A16, W8A16), FP32 sums, each row's result the same bits alone and beside up to 15 others.
//
// NVFP4: out[m][n] = inverse[n] * sum_k x[m][k] * e2m1(codes[n][k]) * e4m3(scales[n][k / 16])
//   codes [N][K / 2] bytes (element 2i in the low nibble of byte i), scales [N][K / 16] E4M3 bytes, inverse [N] FP32
//   (one over the row's global scale), x [M][K] BF16 or FP16 with rows ldx elements apart (16-byte aligned, K a multiple
//   of 32), out [M][N] FP32 or x's type with rows ldo apart.
// FP8: out[m][n] = row_scale[n] * sum_k x[m][k] * e4m3(codes[n][k]), or (MXFP8) sum_k x[m][k] * e4m3(codes[n][k]) *
//   2^(mx[n][k / 32] - 127); codes [N][K] E4M3 bytes (K a multiple of 16, 32 for MXFP8), row_scale [N] FP32 or mx
//   [N][K / 32] E8M0 bytes (exactly one of the two), x and out as for NVFP4.
// BF16: out[m][n] = sum_k x[m][k] * w[n][k], w [N][K] BF16 (K a multiple of 8), x [M][K] BF16, out FP32 or BF16.
//
// Each warp takes a fixed number of output rows; each lane reads 16 bytes of every one of them per step. A weight's
// value (an E2M1 code times its E4M3 block scale, an E4M3 code, times a power-of-two block scale for MXFP8) is exact in
// FP32 and so is its product with an activation; each lane sums its products in FP32 in a fixed order and one
// xor-shuffle tree adds the lanes. Nothing of this depends on M or on the rows a warp takes, so a row's result does not
// depend on the rows beside it. E2M1 codes decode by the hardware conversion (cvt.rn.f16x2.e2m1x2) on the
// architecture-specific targets that have it (sm_100a and later) and by their FP16 bit pattern elsewhere: exact either
// way, so both give the same bits.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace crit::kernels::decode_gemv {

constexpr int kMaxRows = 16;

// The weight formats gemv-cuda takes (the same codes as mma-rows' weights), a run option of one kernel: a format is a
// decoder (decode_gemv.cu's Fmt), not a kernel.
//   kNvfp4   codes [N][K / 2], E4M3 scales [N][K / 16] (ct > 0: cuBLASLt's 128 x 4 tiles), row_scale: inverse [N]
//   kFp8     codes [N][K] E4M3, row_scale [N]
//   kMxfp8   codes [N][K] E4M3, scales E8M0 [N][K / 32]
//   kBf16    codes: w [N][K] BF16 (BF16 activations)
//   kInt4Zp  INT4 with a zero point (AWQ, GPTQ in crit's rows, enqueue_repack_int4): codes [N][K / 2] unsigned
//            (element 2i in the low nibble of byte i), scales FP16 [N][K / group], zeros [N][K / group] bytes 0..16;
//            a value is (code - zero) * scale. group a multiple of 32, K a multiple of group.
enum Weights : int { kNvfp4 = 0, kFp8 = 1, kMxfp8 = 2, kBf16 = 3, kInt4Zp = 4 };

struct Matrix {
  const std::uint8_t* codes = nullptr;
  const std::uint8_t* scales = nullptr;
  const float* row_scale = nullptr;
  const std::uint8_t* zeros = nullptr;
  int ct = 0;
  int group = 0;
};

// out [M][N] = x [M][K] times the matrix in `weights`' format (x BF16 or FP16 rows ldx apart on 16 bytes, BF16 for
// kBf16; out FP32 or x's type).
cudaError_t enqueue_gemv_weights(int weights, const Matrix& m, const void* x, int ldx, bool x_bf16, void* out, int ldo,
                                 bool out_fp32, int M, int N, int K, cudaStream_t stream);

// AWQ's and GPTQ's INT4 tensors in crit's rows (kInt4Zp), exactly (the same codes, zero points and scales):
//   AWQ   qweight int32 [K][N / 8] and qzeros int32 [K / group][N / 8] in AWQ's order (crit_kcore::awq_code), scales
//         FP16 [K / group][N].
//   GPTQ  qweight int32 [K / 8][N] (along K in sequence), qzeros int32 [K / group][N / 8] (along N in sequence, each
//         plus zero_offset: 1 for checkpoints that store zero - 1), scales FP16 [K / group][N].
// perm (or null): row input k' of the result reads checkpoint input perm[k'] (GPTQ's act order, sorted by group so
// that each group's inputs are contiguous: the caller multiplies x[:, perm]). Writes codes [N][K / 2], zeros
// [N][K / group], scales_out [N][K / group].
enum Repack : int { kRepackAwq = 0, kRepackGptq = 1 };
cudaError_t enqueue_repack_int4(int kind, const std::uint32_t* qweight, const std::uint32_t* qzeros,
                                const std::uint16_t* scales, const int* perm, int zero_offset, int N, int K, int group,
                                std::uint8_t* codes, std::uint8_t* zeros, std::uint16_t* scales_out,
                                cudaStream_t stream);

// blocked: the scales are in cuBLASLt's 128 x 4 tiled layout (rows padded to 128, blocks to 4) instead of row major.
cudaError_t enqueue_gemv(const void* x, int ldx, bool x_bf16, const std::uint8_t* codes, const std::uint8_t* scales,
                         bool blocked, const float* inverse, void* out, int ldo, bool out_fp32, int M, int N, int K,
                         cudaStream_t stream);

cudaError_t enqueue_gemv_fp8(const void* x, int ldx, bool x_bf16, const std::uint8_t* codes, const float* row_scale,
                             const std::uint8_t* mx_scale, void* out, int ldo, bool out_fp32, int M, int N, int K,
                             cudaStream_t stream);

cudaError_t enqueue_gemv_bf16(const void* x, int ldx, const void* w, void* out, int ldo, bool out_fp32, int M, int N,
                              int K, cudaStream_t stream);

// Whether this build decodes E2M1 by the hardware conversion on the current device's target (1) or by the bit
// pattern (0): the two give the same bits; tests compare them.
int hardware_decode();

}  // namespace crit::kernels::decode_gemv
