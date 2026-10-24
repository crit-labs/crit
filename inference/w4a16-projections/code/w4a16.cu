// W4A16 projections; see w4a16.h and w4.py.
#include "w4a16.h"

#include <cuda_fp16.h>

namespace crit::w4a16 {
namespace {

__constant__ float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// The eight values of four code bytes (columns k0 .. k0 + 7 of row n), dequantized in FP32.
__device__ __forceinline__ void dequant8(const Weights& w, int n, int k0, std::uint32_t four, float (&v)[8]) {
    if(w.codec == Codec::int4_groups) {
        const std::size_t g = std::size_t(n) * (w.K / w.group) + k0 / w.group;
        const float s = w.scales[g], z = float(w.zeros[g]);
#pragma unroll
        for(int i = 0; i < 8; ++i) v[i] = (float((four >> (4 * i)) & 0xFu) - z) * s;
    } else {
        const int e = int(w.exponents[std::size_t(n) * (w.K / 32) + k0 / 32]) - 127;
#pragma unroll
        for(int i = 0; i < 8; ++i) v[i] = ldexpf(kE2M1[(four >> (4 * i)) & 0xFu], e);
    }
}

__device__ __forceinline__ float bf(const __nv_bfloat16* p) { return __bfloat162float(*p); }

// ---------------------------------------------------------------------------------------------- decode GEMV
constexpr int kMaxTokens = 8;

// Warp per row; lane l takes 32 columns (16 code bytes) per step, the warp 1,024 columns. Per-token sums run in a fixed
// order (lane-local, then a fixed shuffle tree to lane 0), whatever T is.
__global__ void __launch_bounds__(256) gemv_kernel(Weights w, const __nv_bfloat16* __restrict__ x, int x_stride, int T,
                                                   float* __restrict__ y, int y_stride, bool add) {
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if(n >= w.N) return;
    float acc[kMaxTokens];
#pragma unroll
    for(int t = 0; t < kMaxTokens; ++t) acc[t] = 0.f;
    const std::uint8_t* row = w.codes + std::size_t(n) * (w.K / 2);
    for(int k0 = lane * 32; k0 < w.K; k0 += 1024) {
        const uint4 c = *reinterpret_cast<const uint4*>(row + k0 / 2);
        const std::uint32_t words[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
        for(int q = 0; q < 4; ++q) {
            float v[8];
            dequant8(w, n, k0 + 8 * q, words[q], v);
#pragma unroll
            for(int t = 0; t < kMaxTokens; ++t) {
                if(t >= T) break;
                const __nv_bfloat16* xr = x + std::size_t(t) * x_stride + k0 + 8 * q;
#pragma unroll
                for(int i = 0; i < 8; ++i) acc[t] = fmaf(v[i], bf(xr + i), acc[t]);
            }
        }
    }
#pragma unroll
    for(int t = 0; t < kMaxTokens; ++t) {
        if(t >= T) break;
        float s = acc[t];
#pragma unroll
        for(int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if(lane == 0) {
            float* out = y + std::size_t(t) * y_stride + n;
            *out = add ? *out + s : s;
        }
    }
}

// ---------------------------------------------------------------------------------------------- prompt GEMM
constexpr int kTile = 64;   // rows, tokens and columns per block tile

__device__ __forceinline__ int swz(int row, int chunk) { return row * 8 + (chunk ^ (row & 7)); }
__device__ __forceinline__ void ldsm_x4(std::uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ std::uint32_t pack2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const std::uint32_t*>(&v);
}

// Block (64 rows, 64 tokens), 4 warps: warp w owns rows 16w .. 16w + 15 against all 64 tokens. Per 64-column step the
// weights are dequantized to BF16 into a swizzled tile (as A, row-major) and the activations staged beside them (as B,
// one token per row, the K-major layout mma's col operand wants).
__global__ void __launch_bounds__(128) gemm_kernel(Weights w, const __nv_bfloat16* __restrict__ x, int x_stride, int T,
                                                   float* __restrict__ y, int y_stride, bool add) {
    __shared__ uint4 Ws[kTile * 8], Xs[kTile * 8];
    const int n0 = blockIdx.x * kTile, t0 = blockIdx.y * kTile, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float acc[8][4];
#pragma unroll
    for(int i = 0; i < 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    for(int k0 = 0; k0 < w.K; k0 += kTile) {
        for(int e = tid; e < kTile * 8; e += 128) {
            const int r = e / 8, c = e % 8;
            const std::uint32_t four = *reinterpret_cast<const std::uint32_t*>(w.codes + std::size_t(n0 + r) * (w.K / 2) + (k0 + 8 * c) / 2);
            float v[8];
            dequant8(w, n0 + r, k0 + 8 * c, four, v);
            Ws[swz(r, c)] = make_uint4(pack2(v[0], v[1]), pack2(v[2], v[3]), pack2(v[4], v[5]), pack2(v[6], v[7]));
            const int tok = t0 + r;
            Xs[swz(r, c)] = tok < T ? *reinterpret_cast<const uint4*>(x + std::size_t(tok) * x_stride + k0 + 8 * c) : make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();
#pragma unroll
        for(int ks = 0; ks < kTile / 16; ++ks) {
            std::uint32_t a[4];
            ldsm_x4(a, Ws + swz(warp * 16 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
            for(int np = 0; np < 4; ++np) {
                std::uint32_t b[4];
                ldsm_x4(b, Xs + swz(np * 16 + (lane & 7) + ((lane >> 4) << 3), ks * 2 + ((lane >> 3) & 1)));
                mma_bf16(acc[2 * np], a, b[0], b[1]);
                mma_bf16(acc[2 * np + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
    }
    const int gl = lane >> 2, t4 = lane & 3;
#pragma unroll
    for(int nt = 0; nt < 8; ++nt)
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const int row = n0 + warp * 16 + gl + (i >> 1) * 8, tok = t0 + nt * 8 + 2 * t4 + (i & 1);
            if(tok < T) {
                float* out = y + std::size_t(tok) * y_stride + row;
                *out = add ? *out + acc[nt][i] : acc[nt][i];
            }
        }
}

__global__ void gather_kernel(const __nv_bfloat16* x, int x_stride, const std::int32_t* perm, int K, int T, __nv_bfloat16* out,
                              int out_stride) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if(j < K && t < T) out[std::size_t(t) * out_stride + j] = x[std::size_t(t) * x_stride + perm[j]];
}

// ---------------------------------------------------------------------------------------------- conversion
__device__ __forceinline__ std::uint32_t nibble(std::int32_t word, int i) { return (std::uint32_t(word) >> (4 * i)) & 0xFu; }

// Stable counting order of the columns by group (act-order checkpoints give every group `group` columns): one thread
// with the per-group cursors in shared memory, K steps per pass, once per weight at load.
__global__ void act_order_kernel(const std::int32_t* g_idx, int K, int groups, std::int32_t* perm) {
    extern __shared__ int cursor[];
    if(threadIdx.x) return;
    for(int g = 0; g < groups; ++g) cursor[g] = 0;
    for(int k = 0; k < K; ++k)
        if(g_idx[k] >= 0 && g_idx[k] < groups) ++cursor[g_idx[k]];
    for(int g = 0, run = 0; g < groups; ++g) {
        const int count = cursor[g];
        cursor[g] = run;
        run += count;
    }
    for(int k = 0; k < K; ++k)
        if(g_idx[k] >= 0 && g_idx[k] < groups) perm[cursor[g_idx[k]]++] = k;
}

__global__ void gptq_codes_kernel(const std::int32_t* qweight, const std::int32_t* perm, int N, int K, std::uint8_t* codes) {
    const int n = blockIdx.y, j = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(n >= N || j >= K) return;
    const int k0 = perm ? perm[j] : j, k1 = perm ? perm[j + 1] : j + 1;
    const std::uint32_t lo = nibble(qweight[std::size_t(k0 / 8) * N + n], k0 % 8);
    const std::uint32_t hi = nibble(qweight[std::size_t(k1 / 8) * N + n], k1 % 8);
    codes[std::size_t(n) * (K / 2) + j / 2] = std::uint8_t(lo | (hi << 4));
}

__global__ void gptq_groups_kernel(const std::int32_t* qzeros, const __half* scales, int N, int G, int zero_offset, float* out_scales,
                                   std::uint8_t* zeros) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, g = blockIdx.y;
    if(n >= N || g >= G) return;
    out_scales[std::size_t(n) * G + g] = __half2float(scales[std::size_t(g) * N + n]);
    zeros[std::size_t(n) * G + g] = std::uint8_t(nibble(qzeros[std::size_t(g) * (N / 8) + n / 8], n % 8) + std::uint32_t(zero_offset));
}

__constant__ int kAwqNibble[8] = {0, 4, 1, 5, 2, 6, 3, 7};   // column c % 8 sits at nibble kAwqNibble[c % 8]

__global__ void awq_kernel(const std::int32_t* qweight, const std::int32_t* qzeros, const __half* scales, int N, int K, int group,
                           std::uint8_t* codes, float* out_scales, std::uint8_t* zeros) {
    const int n = blockIdx.y, j = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(n >= N || j >= K) return;
    const int slot = kAwqNibble[n % 8];
    const std::uint32_t lo = nibble(qweight[std::size_t(j) * (N / 8) + n / 8], slot);
    const std::uint32_t hi = nibble(qweight[std::size_t(j + 1) * (N / 8) + n / 8], slot);
    codes[std::size_t(n) * (K / 2) + j / 2] = std::uint8_t(lo | (hi << 4));
    if(j % group == 0) {
        const int g = j / group, G = K / group;
        out_scales[std::size_t(n) * G + g] = __half2float(scales[std::size_t(g) * N + n]);
        zeros[std::size_t(n) * G + g] = std::uint8_t(nibble(qzeros[std::size_t(g) * (N / 8) + n / 8], slot));
    }
}

__global__ void compressed_kernel(const std::int32_t* packed, const __nv_bfloat16* scale, const std::int32_t* zero_point, int N, int K,
                                  int group, std::uint8_t* codes, float* out_scales, std::uint8_t* zeros) {
    const int n = blockIdx.y, j = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(n >= N || j >= K) return;
    const std::int32_t* row = packed + std::size_t(n) * (K / 8);
    const std::uint32_t lo = nibble(row[j / 8], j % 8), hi = nibble(row[(j + 1) / 8], (j + 1) % 8);
    codes[std::size_t(n) * (K / 2) + j / 2] = std::uint8_t(lo | (hi << 4));
    if(j % group == 0) {
        const int g = j / group, G = K / group;
        out_scales[std::size_t(n) * G + g] = __bfloat162float(scale[std::size_t(n) * G + g]);
        zeros[std::size_t(n) * G + g] = std::uint8_t(zero_point ? zero_point[std::size_t(n) * G + g] : 8);
    }
}

bool valid(const Weights& w) {
    if(!w.codes || w.N < 1 || w.K < 64 || w.K % 64) return false;
    if(w.codec == Codec::int4_groups) return w.scales && w.zeros && (w.group == 32 || w.group == 64 || w.group == 128) && w.K % w.group == 0;
    return w.codec == Codec::mxfp4 && w.exponents;
}

}  // namespace

cudaError_t enqueue_gemv(const Weights& w, const __nv_bfloat16* x, int x_stride, int T, float* y, int y_stride, bool add,
                         cudaStream_t stream) {
    if(!valid(w) || !x || !y || T < 1 || T > kMaxTokens || x_stride < w.K || x_stride % 8 || y_stride < w.N) return cudaErrorInvalidValue;
    gemv_kernel<<<(w.N + 7) / 8, 256, 0, stream>>>(w, x, x_stride, T, y, y_stride, add);
    return cudaGetLastError();
}

cudaError_t enqueue_gemm(const Weights& w, const __nv_bfloat16* x, int x_stride, int T, float* y, int y_stride, bool add,
                         cudaStream_t stream) {
    if(!T) return cudaSuccess;
    if(!valid(w) || !x || !y || T < 0 || w.N % kTile || x_stride < w.K || x_stride % 8 || y_stride < w.N) return cudaErrorInvalidValue;
    gemm_kernel<<<dim3(w.N / kTile, (T + kTile - 1) / kTile), 128, 0, stream>>>(w, x, x_stride, T, y, y_stride, add);
    return cudaGetLastError();
}

cudaError_t enqueue_gather_columns(const __nv_bfloat16* x, int x_stride, const std::int32_t* perm, int K, int T, __nv_bfloat16* out,
                                   int out_stride, cudaStream_t stream) {
    if(!T) return cudaSuccess;
    if(!x || !perm || !out || K < 1 || T < 0 || T > 65535 || x_stride < K || out_stride < K) return cudaErrorInvalidValue;
    gather_kernel<<<dim3((K + 255) / 256, T), 256, 0, stream>>>(x, x_stride, perm, K, T, out, out_stride);
    return cudaGetLastError();
}

cudaError_t enqueue_from_gptq(const Gptq& in, int N, int K, int group, std::uint8_t* codes, float* scales, std::uint8_t* zeros,
                              std::int32_t* perm, cudaStream_t stream) {
    if(!in.qweight || !in.qzeros || !in.scales || !codes || !scales || !zeros || (in.g_idx && !perm) || N < 8 || N % 8 || K < 8 ||
       K % 8 || group < 8 || K % group || N > 65535)
        return cudaErrorInvalidValue;
    const int G = K / group;
    if(in.g_idx) {
        const std::size_t bytes = sizeof(int) * std::size_t(G);
        if(bytes > 48 * 1024) return cudaErrorInvalidValue;
        act_order_kernel<<<1, 32, bytes, stream>>>(in.g_idx, K, G, perm);
    }
    gptq_codes_kernel<<<dim3((K / 2 + 255) / 256, N), 256, 0, stream>>>(in.qweight, in.g_idx ? perm : nullptr, N, K, codes);
    gptq_groups_kernel<<<dim3((N + 255) / 256, G), 256, 0, stream>>>(in.qzeros, in.scales, N, G, in.zero_offset, scales, zeros);
    return cudaGetLastError();
}

cudaError_t enqueue_from_awq(const std::int32_t* qweight, const std::int32_t* qzeros, const __half* scales, int N, int K, int group,
                             std::uint8_t* codes, float* out_scales, std::uint8_t* zeros, cudaStream_t stream) {
    if(!qweight || !qzeros || !scales || !codes || !out_scales || !zeros || N < 8 || N % 8 || K < 2 || K % 2 || group < 2 ||
       group % 2 || K % group || N > 65535)
        return cudaErrorInvalidValue;
    awq_kernel<<<dim3((K / 2 + 255) / 256, N), 256, 0, stream>>>(qweight, qzeros, scales, N, K, group, codes, out_scales, zeros);
    return cudaGetLastError();
}

cudaError_t enqueue_from_compressed(const std::int32_t* packed, const __nv_bfloat16* scale, const std::int32_t* zero_point, int N, int K,
                                    int group, std::uint8_t* codes, float* out_scales, std::uint8_t* zeros, cudaStream_t stream) {
    if(!packed || !scale || !codes || !out_scales || !zeros || N < 1 || K < 8 || K % 8 || group < 2 || group % 2 || K % group ||
       N > 65535)
        return cudaErrorInvalidValue;
    compressed_kernel<<<dim3((K / 2 + 255) / 256, N), 256, 0, stream>>>(packed, scale, zero_point, N, K, group, codes, out_scales, zeros);
    return cudaGetLastError();
}

}  // namespace crit::w4a16
