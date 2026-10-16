#pragma once
// Rotated low-bit KV rows on the device. Reference kernels, not for performance:
// one block per row, `dim` threads, a shared-memory normalized Walsh-Hadamard
// transform. R = H * diag(signs); keys/values are stored as int8 codes of R x with
// one FP32 scale per 32 values; queries are rotated by the same R; outputs are
// rotated back with R^T = diag(signs) * H.
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace crit::techniques::rotated_kv {
inline constexpr std::uint32_t group = 32;

struct Rows {
    std::uint32_t rows = 0;
    std::uint32_t dim = 0;          // power of two, 32..1024
    const float* signs = nullptr;   // [dim], each +1 or -1
};
// codes[r][dim] (int8 in +-qmax, qmax = 2^(bits-1) - 1) and scales[r][dim / 32] of R x[r]; bits is 4 or 8.
cudaError_t enqueue_rotate_quantize(const Rows&, const float* x, std::size_t ldx, int bits, std::int8_t* codes, float* scales,
                                    cudaStream_t);
// y[r] = R x[r] (queries), in place allowed.
cudaError_t enqueue_rotate(const Rows&, const float* x, std::size_t ldx, float* y, std::size_t ldy, cudaStream_t);
// y[r] = R^T x[r] (attention outputs computed in the rotated basis), in place allowed.
cudaError_t enqueue_unrotate(const Rows&, const float* x, std::size_t ldx, float* y, std::size_t ldy, cudaStream_t);
// y[r] = dequantized stored row (still rotated): codes * scale.
cudaError_t enqueue_dequantize(const Rows&, const std::int8_t* codes, const float* scales, float* y, std::size_t ldy, cudaStream_t);
}
