// Reference kernels, not for performance. One block per row with `dim` threads.
#include "rotated_kv.h"
#include <cmath>

namespace crit::techniques::rotated_kv {
namespace {
// In-place normalized Walsh-Hadamard transform of s[0..dim) by `dim` threads.
__device__ void fwht(float* s, std::uint32_t dim) {
    const auto t = threadIdx.x;
    for(std::uint32_t h = 1; h < dim; h *= 2) {
        __syncthreads();
        if(!(t & h)) {
            const float a = s[t], b = s[t + h];
            s[t] = a + b;
            s[t + h] = a - b;
        }
    }
    __syncthreads();
    s[t] *= rsqrtf(float(dim));
    __syncthreads();
}

__global__ void rotate_rows(Rows p, const float* x, std::size_t ldx, float* y, std::size_t ldy, bool inverse) {
    extern __shared__ float s[];
    const auto t = threadIdx.x;
    const float v = x[std::size_t(blockIdx.x) * ldx + t];
    s[t] = inverse ? v : v * p.signs[t];
    fwht(s, p.dim);
    y[std::size_t(blockIdx.x) * ldy + t] = inverse ? s[t] * p.signs[t] : s[t];
}

__global__ void rotate_quantize_rows(Rows p, const float* x, std::size_t ldx, int qmax, std::int8_t* codes, float* scales) {
    extern __shared__ float s[];
    const auto t = threadIdx.x;
    s[t] = x[std::size_t(blockIdx.x) * ldx + t] * p.signs[t];
    fwht(s, p.dim);
    const float v = s[t];
    float amax = fabsf(v);  // one warp is one 32-value group
    for(int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float scale = amax > 0.0f ? amax / float(qmax) : 1.0f;
    const float q = fminf(fmaxf(rintf(v / scale), -float(qmax)), float(qmax));
    codes[std::size_t(blockIdx.x) * p.dim + t] = std::int8_t(q);
    if(!(t % group)) scales[std::size_t(blockIdx.x) * (p.dim / group) + t / group] = scale;
}

__global__ void dequantize_rows(Rows p, const std::int8_t* codes, const float* scales, float* y, std::size_t ldy) {
    const auto t = threadIdx.x;
    const std::size_t r = blockIdx.x;
    y[r * ldy + t] = float(codes[r * p.dim + t]) * scales[r * (p.dim / group) + t / group];
}

bool valid(const Rows& p) {
    return p.signs && p.dim >= group && p.dim <= 1024 && !(p.dim & (p.dim - 1));
}
}

cudaError_t enqueue_rotate_quantize(const Rows& p, const float* x, std::size_t ldx, int bits, std::int8_t* codes, float* scales, cudaStream_t s) {
    if(!valid(p) || !x || !codes || !scales || ldx < p.dim || (bits != 4 && bits != 8)) return cudaErrorInvalidValue;
    if(!p.rows) return cudaSuccess;
    rotate_quantize_rows<<<p.rows, p.dim, p.dim * sizeof(float), s>>>(p, x, ldx, (1 << (bits - 1)) - 1, codes, scales);
    return cudaPeekAtLastError();
}
cudaError_t enqueue_rotate(const Rows& p, const float* x, std::size_t ldx, float* y, std::size_t ldy, cudaStream_t s) {
    if(!valid(p) || !x || !y || ldx < p.dim || ldy < p.dim) return cudaErrorInvalidValue;
    if(!p.rows) return cudaSuccess;
    rotate_rows<<<p.rows, p.dim, p.dim * sizeof(float), s>>>(p, x, ldx, y, ldy, false);
    return cudaPeekAtLastError();
}
cudaError_t enqueue_unrotate(const Rows& p, const float* x, std::size_t ldx, float* y, std::size_t ldy, cudaStream_t s) {
    if(!valid(p) || !x || !y || ldx < p.dim || ldy < p.dim) return cudaErrorInvalidValue;
    if(!p.rows) return cudaSuccess;
    rotate_rows<<<p.rows, p.dim, p.dim * sizeof(float), s>>>(p, x, ldx, y, ldy, true);
    return cudaPeekAtLastError();
}
cudaError_t enqueue_dequantize(const Rows& p, const std::int8_t* codes, const float* scales, float* y, std::size_t ldy, cudaStream_t s) {
    if(!valid(p) || !codes || !scales || !y || ldy < p.dim) return cudaErrorInvalidValue;
    if(!p.rows) return cudaSuccess;
    dequantize_rows<<<p.rows, p.dim, 0, s>>>(p, codes, scales, y, ldy);
    return cudaPeekAtLastError();
}
}
