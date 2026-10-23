// GGUF block projections; see gguf_cuda.h.
//
// A warp owns one weight row. For 256-value formats lane l decodes values [8l, 8l + 8) of each block in turn (one
// block per warp step); for 32-value formats four lanes share a block and a warp step covers eight blocks. Each lane
// multiplies its eight values with the matching activations of every batch row, and the warp reduces at the end.
#include "gguf_cuda.h"

namespace crit::gguf_blocks {
namespace {

constexpr int kWarps = 4;

template <Kind K>
__global__ void dequantize_kernel(const std::uint8_t* __restrict__ blocks, std::size_t count, float* __restrict__ out) {
    constexpr Format f = format(K);
    constexpr int groups = f.values / 8;
    const std::size_t total = count * groups;
    for(std::size_t g = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; g < total; g += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t block = g / groups;
        const int first = static_cast<int>(g % groups) * 8;
        float v[8];
        decode8(K, blocks + block * f.bytes, first, v);
        float* o = out + block * f.values + first;
#pragma unroll
        for(int t = 0; t < 8; ++t) o[t] = v[t];
    }
}

template <Kind K>
__global__ void __launch_bounds__(kWarps * 32) gemv_kernel(const std::uint8_t* __restrict__ w, int rows, int cols,
                                                           const float* __restrict__ x, int batch, float* __restrict__ y) {
    constexpr Format f = format(K);
    const int lane = threadIdx.x & 31, row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if(row >= rows) return;
    const int nblocks = cols / f.values;
    const std::uint8_t* rw = w + static_cast<std::size_t>(row) * nblocks * f.bytes;
    float acc[kMaxBatch];
#pragma unroll
    for(int b = 0; b < kMaxBatch; ++b) acc[b] = 0.f;
    constexpr int lanes_per_block = f.values / 8, blocks_per_step = 32 / lanes_per_block;
    for(int base = 0; base < nblocks; base += blocks_per_step) {
        const int block = base + lane / lanes_per_block;
        if(block >= nblocks) continue;
        const int first = (lane % lanes_per_block) * 8;
        float v[8];
        decode8(K, rw + static_cast<std::size_t>(block) * f.bytes, first, v);
        const int k0 = block * f.values + first;
#pragma unroll
        for(int b = 0; b < kMaxBatch; ++b) {
            if(b >= batch) break;
            const float4 x0 = *reinterpret_cast<const float4*>(x + static_cast<std::size_t>(b) * cols + k0);
            const float4 x1 = *reinterpret_cast<const float4*>(x + static_cast<std::size_t>(b) * cols + k0 + 4);
            float s = acc[b];
            s = fmaf(v[0], x0.x, s); s = fmaf(v[1], x0.y, s); s = fmaf(v[2], x0.z, s); s = fmaf(v[3], x0.w, s);
            s = fmaf(v[4], x1.x, s); s = fmaf(v[5], x1.y, s); s = fmaf(v[6], x1.z, s); s = fmaf(v[7], x1.w, s);
            acc[b] = s;
        }
    }
#pragma unroll
    for(int b = 0; b < kMaxBatch; ++b) {
        if(b >= batch) break;
        float s = acc[b];
#pragma unroll
        for(int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if(lane == 0) y[static_cast<std::size_t>(b) * rows + row] = s;
    }
}

template <template <Kind> class Launch, class... Args>
cudaError_t dispatch(Kind kind, Args... args) {
    switch(kind) {
    case Kind::q4_0: return Launch<Kind::q4_0>::go(args...);
    case Kind::q8_0: return Launch<Kind::q8_0>::go(args...);
    case Kind::q4_k: return Launch<Kind::q4_k>::go(args...);
    case Kind::q5_k: return Launch<Kind::q5_k>::go(args...);
    case Kind::q6_k: return Launch<Kind::q6_k>::go(args...);
    }
    return cudaErrorInvalidValue;
}
template <Kind K> struct Dequantize {
    static cudaError_t go(const void* blocks, std::size_t count, float* out, cudaStream_t s) {
        const std::size_t groups = count * (format(K).values / 8);
        const int grid = static_cast<int>(std::min<std::size_t>((groups + 255) / 256, 65535));
        dequantize_kernel<K><<<grid, 256, 0, s>>>(static_cast<const std::uint8_t*>(blocks), count, out);
        return cudaGetLastError();
    }
};
template <Kind K> struct Gemv {
    static cudaError_t go(const void* blocks, int rows, int cols, const float* x, int batch, float* y, cudaStream_t s) {
        gemv_kernel<K><<<(rows + kWarps - 1) / kWarps, kWarps * 32, 0, s>>>(static_cast<const std::uint8_t*>(blocks), rows, cols, x, batch, y);
        return cudaGetLastError();
    }
};

}  // namespace

cudaError_t dequantize(Kind kind, const void* blocks, std::size_t count, float* out, cudaStream_t stream) {
    if(!blocks || !out || !count || static_cast<int>(kind) < 0 || static_cast<int>(kind) > 4) return cudaErrorInvalidValue;
    return dispatch<Dequantize>(kind, blocks, count, out, stream);
}

cudaError_t gemv(Kind kind, const void* blocks, int rows, int cols, const float* x, int batch, float* y, cudaStream_t stream) {
    if(!blocks || !x || !y || rows <= 0 || cols <= 0 || batch < 1 || batch > kMaxBatch || static_cast<int>(kind) < 0 ||
       static_cast<int>(kind) > 4 || cols % format(kind).values || reinterpret_cast<std::uintptr_t>(x) % 16)
        return cudaErrorInvalidValue;
    return dispatch<Gemv>(kind, blocks, rows, cols, x, batch, y, stream);
}

}  // namespace crit::gguf_blocks
