// Stored-tile low-precision prompt GEMMs; see stored_tile_gemm.h for layouts and the public interface.
#include "stored_tile_gemm.h"

#include "stored_tile_device.cuh"

#include <cuda_fp8.h>
#include <cuda_fp4.h>

namespace crit::stored_tile {

using detail::cp_async16;
using detail::cp_async8;
using detail::cp_async_commit;
using detail::cp_async_wait;
using detail::gated_value;
using detail::load_streaming;

#if defined(__CUDA_ARCH__) && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define CRIT_STORED_TILE_FP4_MMA 1
#else
#define CRIT_STORED_TILE_FP4_MMA 0
#endif

// ------------------------------------------------------------------------------------------- weight repacking
// One thread per (tile, lane). src: row-major E4M3 codes [N][K]; dst: tiles [N / 16][K / 32][512 bytes], source tile
// row rt written to destination tile row first + step * rt.
__global__ void repack_fp8_kernel(const std::uint8_t* __restrict__ src, std::uint8_t* __restrict__ dst, int N, int K,
                                  int first, int step) {
    const std::size_t tid = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const int KT = K / 32;
    const std::size_t tiles = static_cast<std::size_t>(N / 16) * KT;
    if(tid >= tiles * 32) return;
    const std::size_t tile = tid / 32;
    const int lane = static_cast<int>(tid % 32), g = lane >> 2, t4 = lane & 3;
    const int rt = static_cast<int>(tile / KT), kt = static_cast<int>(tile % KT);
    const std::size_t rg = static_cast<std::size_t>(rt) * 16 + g, rg8 = rg + 8;
    const std::size_t dtile = (static_cast<std::size_t>(first) + static_cast<std::size_t>(step) * rt) * KT + kt;
    std::uint8_t* d = dst + dtile * 512 + lane * 16;
    for(int s = 0; s < 2; ++s) {
        const std::size_t col = static_cast<std::size_t>(kt) * 32 + s * 16 + 2 * t4;
        const std::uint8_t* a = src + rg * K + col;
        const std::uint8_t* b = src + rg8 * K + col;
        d[8 * s + 0] = a[0]; d[8 * s + 1] = a[1];
        d[8 * s + 2] = b[0]; d[8 * s + 3] = b[1];
        d[8 * s + 4] = a[8]; d[8 * s + 5] = a[9];
        d[8 * s + 6] = b[8]; d[8 * s + 7] = b[9];
    }
}

// codes: row-major packed E2M1 [N][K / 2] (low nibble = even column); scales: E4M3 [N][K / 16].
// dst_codes: tiles [N / 16][K / 64][512]; dst_scales: [N / 16][K / 64][64]; tile rows placed as in repack_fp8_kernel.
__global__ void repack_fp4_kernel(const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
                                  std::uint8_t* __restrict__ dst_codes, std::uint8_t* __restrict__ dst_scales, int N, int K,
                                  int first, int step) {
    const std::size_t tid = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const int KT = K / 64;
    const std::size_t tiles = static_cast<std::size_t>(N / 16) * KT;
    if(tid >= tiles * 32) return;
    const std::size_t tile = tid / 32;
    const int lane = static_cast<int>(tid % 32), g = lane >> 2, t4 = lane & 3;
    const int rt = static_cast<int>(tile / KT), kt = static_cast<int>(tile % KT);
    const std::size_t rg = static_cast<std::size_t>(rt) * 16 + g, rg8 = rg + 8;
    const std::size_t rb = static_cast<std::size_t>(K / 2);
    const std::size_t dtile = (static_cast<std::size_t>(first) + static_cast<std::size_t>(step) * rt) * KT + kt;
    for(int s = 0; s < 4; ++s) {
        const int col = kt * 64 + s * 16 + 2 * t4;
        std::uint8_t* d = dst_codes + dtile * 512 + lane * 16 + s * 4;
        d[0] = codes[rg * rb + col / 2];
        d[1] = codes[rg8 * rb + col / 2];
        d[2] = codes[rg * rb + (col + 8) / 2];
        d[3] = codes[rg8 * rb + (col + 8) / 2];
    }
    if(t4 == 0) {
        const std::size_t sb = static_cast<std::size_t>(K / 16);
        for(int s = 0; s < 4; ++s) {
            dst_scales[dtile * 64 + g * 8 + s] = scales[rg * sb + kt * 4 + s];
            dst_scales[dtile * 64 + g * 8 + 4 + s] = scales[rg8 * sb + kt * 4 + s];
        }
    }
}

__global__ void place_row_scales_kernel(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst, int N,
                                        int first, int step) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= N) return;
    dst[(static_cast<std::size_t>(first) + static_cast<std::size_t>(step) * (n / 16)) * 16 + n % 16] = src[n];
}

// ------------------------------------------------------------------------------------------- activation packing
// Per-token E4M3 scale amax / 448 (1 for an all-zero row). One block per token.
__global__ void token_amax_kernel(const __nv_bfloat16* __restrict__ x, int K, float* __restrict__ scale) {
    __shared__ float red[32];
    const __nv_bfloat16* row = x + static_cast<std::size_t>(blockIdx.x) * K;
    float m = 0.f;
    for(int i = threadIdx.x; i < K; i += blockDim.x) m = fmaxf(m, fabsf(__bfloat162float(row[i])));
#pragma unroll
    for(int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if(threadIdx.x == 0) {
        float v = 0.f;
        for(int w = 0; w < static_cast<int>(blockDim.x >> 5); ++w) v = fmaxf(v, red[w]);
        scale[blockIdx.x] = v > 0.f ? v / 448.f : 1.f;
    }
}

// FP8 activations in fragment order: x8[(group of 8 tokens * KT + kt) * 32 + lane] = {b0, b1}, lane 4 g + t holding
// token g, columns {2t, 2t+1, 2t+8, 2t+9} of each 16-column half (b0: first half, b1: second). Tokens >= T are zero.
__global__ void pack_fp8_activations_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ scale, int T, int K,
                                            int Tpad, uint2* __restrict__ x8) {
    const std::size_t tid = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const int KT = K / 32;
    const std::size_t total = static_cast<std::size_t>(Tpad / 8) * KT * 32;
    if(tid >= total) return;
    const int lane = static_cast<int>(tid % 32), g = lane >> 2, t4 = lane & 3;
    const int kt = static_cast<int>((tid / 32) % KT), tg = static_cast<int>(tid / 32 / KT);
    const int tok = tg * 8 + g;
    uint2 v = make_uint2(0u, 0u);
    if(tok < T) {
        const float inv = 1.f / scale[tok];
        const __nv_bfloat16* xr = x + static_cast<std::size_t>(tok) * K + kt * 32;
        std::uint32_t w[2];
#pragma unroll
        for(int h = 0; h < 2; ++h) {
            const int c = h * 16 + 2 * t4;
            const int cols[4] = {c, c + 1, c + 8, c + 9};
            std::uint32_t r = 0u;
#pragma unroll
            for(int b = 0; b < 4; ++b) {
                const std::uint32_t q8 = __nv_cvt_float_to_fp8(__bfloat162float(xr[cols[b]]) * inv, __NV_SATFINITE, __NV_E4M3);
                r |= q8 << (8 * b);
            }
            w[h] = r;
        }
        v = make_uint2(w[0], w[1]);
    }
    x8[tid] = v;
}

__device__ __forceinline__ int fp4_mma_column(int t, int r, int i) {
    const int s = (t >> 1) + 2 * r, tp = i < 4 ? t : (t ^ 2), ii = i & 3;
    return 16 * s + 2 * tp + (ii < 2 ? ii : 6 + ii);
}
__device__ __forceinline__ float e4m3_value(std::uint8_t code) {
    __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code), __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&h));
}

// NVFP4 activations in fragment order, one thread per (token, 64-column tile):
// x4[(group of 8 tokens * KT + kt) * 32 + 4 g + t] = {b0, b1, scale word of token g, 0}.
__global__ void pack_fp4_activations_kernel(const __nv_bfloat16* __restrict__ x, int T, int K, int Tpad, float gx,
                                            uint4* __restrict__ x4) {
    const std::size_t tid = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const int KT = K / 64;
    if(tid >= static_cast<std::size_t>(Tpad) * KT) return;
    const int tok = static_cast<int>(tid / KT), kt = static_cast<int>(tid % KT);
    const int tg = tok >> 3, g = tok & 7;
    uint4* o = x4 + (static_cast<std::size_t>(tg) * KT + kt) * 32 + g * 4;
    if(tok >= T) {
#pragma unroll
        for(int t = 0; t < 4; ++t) o[t] = make_uint4(0u, 0u, 0u, 0u);
        return;
    }
    const __nv_bfloat16* p = x + static_cast<std::size_t>(tok) * K + kt * 64;
    float v[64];
#pragma unroll
    for(int i = 0; i < 64; ++i) v[i] = __bfloat162float(p[i]);
    float mul[4];
    std::uint32_t sw = 0u;
#pragma unroll
    for(int s = 0; s < 4; ++s) {
        float m = 0.f;
#pragma unroll
        for(int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(v[16 * s + i]));
        const __nv_fp8_storage_t sc = __nv_cvt_float_to_fp8(m * gx / 6.f, __NV_SATFINITE, __NV_E4M3);
        const float sv = e4m3_value(static_cast<std::uint8_t>(sc));
        mul[s] = sv > 0.f ? gx / sv : 0.f;
        sw |= static_cast<std::uint32_t>(sc) << (8 * s);
    }
#pragma unroll
    for(int t = 0; t < 4; ++t) {
        std::uint32_t b[2];
#pragma unroll
        for(int r = 0; r < 2; ++r) {
            std::uint32_t w = 0u;
#pragma unroll
            for(int i = 0; i < 8; i += 2) {
                const int c0 = fp4_mma_column(t, r, i), c1 = fp4_mma_column(t, r, i + 1);
                const float2 f = make_float2(v[c0] * mul[c0 >> 4], v[c1] * mul[c1 >> 4]);
                w |= static_cast<std::uint32_t>(__nv_cvt_float2_to_fp4x2(f, __NV_E2M1, cudaRoundNearest)) << (4 * i);
            }
            b[r] = w;
        }
        o[t] = make_uint4(b[0], b[1], sw, 0u);
    }
}

// ------------------------------------------------------------------------------------------- tensor-core helpers
__device__ __forceinline__ void mma_e4m3(float (&c)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t a2, std::uint32_t a3,
                                         std::uint32_t b0, std::uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    __trap();
#endif
}
__device__ __forceinline__ void mma_nvfp4(float (&c)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t a2, std::uint32_t a3,
                                          std::uint32_t b0, std::uint32_t b1, std::uint32_t sa, std::uint32_t sb) {
#if CRIT_STORED_TILE_FP4_MMA
    // Scale selectors {0, 0}: lane 4 g supplies row g's four group scales, lane 4 g + 1 row g + 8's; lane 4 g
    // supplies token g's.
    asm("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#else
    (void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1; (void)sa; (void)sb;
    __trap();
#endif
}

// ------------------------------------------------------------------------------------------- GEMM kernels
// Grid (Tpad / (8 NT), N / (128 RT), SK). A block of 8 warps; warp w owns RT consecutive 16-row tiles; a B fragment
// from shared memory feeds RT MMAs (the NVFP4 kernel reads it once per row tile instead, so only one row tile's A
// fragment is live). Weights stream through a PF-deep register ring; activations through a 3-stage cp.async ring.
// SK > 1 writes FP32 partials [SK][T][N] for finish_partials_kernel. GATED (RT == 2, SK == 1): the warp's row tiles
// are the gate and up tiles of the same 16 intermediate rows, and the epilogue writes bf16(act(gate) * up) to G
// (row stride ldo) instead of Y, gate and up scaled exactly as the plain epilogue scales them.
template <int NT, int RT, int PF, bool GATED = false>
__global__ void __launch_bounds__(256) gemm_fp8_kernel(const std::uint8_t* __restrict__ W, const __nv_bfloat16* __restrict__ row_scale,
                                                      const uint2* __restrict__ X8, const float* __restrict__ token_scale,
                                                      float* __restrict__ Y, float* __restrict__ part, int N, int K, int T,
                                                      int ldo, int add, __nv_bfloat16* __restrict__ G = nullptr, int act = 0) {
    static_assert(!GATED || RT == 2, "the gated epilogue pairs a warp's two row tiles");
    __shared__ uint2 xsm[3][NT][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int rt0 = (blockIdx.y * 8 + warp) * RT;
    const int tg0 = blockIdx.x * NT;
    const int KT = K / 32;
    const int SK = gridDim.z, per = (KT + SK - 1) / SK, kt0 = blockIdx.z * per, kt1 = min(KT, kt0 + per);
    float acc[RT][NT][4];
#pragma unroll
    for(int r = 0; r < RT; ++r)
#pragma unroll
        for(int n = 0; n < NT; ++n) acc[r][n][0] = acc[r][n][1] = acc[r][n][2] = acc[r][n][3] = 0.f;
    auto stage = [&](int kt) {
        if(kt < kt1) {
            uint2 (*buf)[32] = xsm[kt % 3];
#pragma unroll
            for(int n = threadIdx.x >> 5; n < NT; n += 8)
                cp_async8(&buf[n][lane], X8 + (static_cast<std::size_t>(tg0 + n) * KT + kt) * 32 + lane);
        }
        cp_async_commit();
    };
    uint4 wv[PF][RT];
    auto load_w = [&](int kt, int i) {
        if(kt >= kt1) return;
#pragma unroll
        for(int r = 0; r < RT; ++r) wv[i][r] = load_streaming(W + (static_cast<std::size_t>(rt0 + r) * KT + kt) * 512 + lane * 16);
    };
#pragma unroll
    for(int i = 0; i < PF; ++i) load_w(kt0 + i, i);
    stage(kt0);
    stage(kt0 + 1);
    cp_async_wait<1>();
    __syncthreads();
    for(int kt = kt0; kt < kt1; kt += PF) {
#pragma unroll
        for(int u = 0; u < PF; ++u) {
            const int kk = kt + u;
            if(kk >= kt1) break;
            stage(kk + 2);
            const uint2 (*xb)[32] = xsm[kk % 3];
            std::uint32_t a[RT][4];
#pragma unroll
            for(int r = 0; r < RT; ++r) {
                const uint4 w = wv[u][r];
                a[r][0] = __byte_perm(w.x, w.y, 0x5410); a[r][1] = __byte_perm(w.x, w.y, 0x7632);
                a[r][2] = __byte_perm(w.z, w.w, 0x5410); a[r][3] = __byte_perm(w.z, w.w, 0x7632);
            }
            load_w(kk + PF, u);
#pragma unroll
            for(int n = 0; n < NT; ++n) {
                const uint2 bb = xb[n][lane];
#pragma unroll
                for(int r = 0; r < RT; ++r) mma_e4m3(acc[r][n], a[r][0], a[r][1], a[r][2], a[r][3], bb.x, bb.y);
            }
            cp_async_wait<1>();
            __syncthreads();
        }
    }
    cp_async_wait<0>();
    if constexpr(GATED) {
        // Scales read once ahead of the math (token index clamped; the store is guarded), the same products as below.
        const int r0 = rt0 * 16 + g;
        const float sg[2] = {__bfloat162float(row_scale[r0]), __bfloat162float(row_scale[r0 + 8])};
        const float su[2] = {__bfloat162float(row_scale[r0 + 16]), __bfloat162float(row_scale[r0 + 24])};
#pragma unroll
        for(int n = 0; n < NT; ++n) {
            const int tok0 = (tg0 + n) * 8 + 2 * t4;
            const float ts[2] = {token_scale[min(tok0, T - 1)], token_scale[min(tok0 + 1, T - 1)]};
            __nv_bfloat16 o[4];
#pragma unroll
            for(int i = 0; i < 4; ++i) o[i] = gated_value(act, acc[0][n][i] * sg[i >> 1] * ts[i & 1], acc[1][n][i] * su[i >> 1] * ts[i & 1]);
#pragma unroll
            for(int i = 0; i < 4; ++i)
                if(tok0 + (i & 1) < T) G[static_cast<std::size_t>(tok0 + (i & 1)) * ldo + rt0 / 2 * 16 + g + (i >> 1) * 8] = o[i];
        }
    } else {
#pragma unroll
        for(int r = 0; r < RT; ++r) {
            const int row = (rt0 + r) * 16 + g;
#pragma unroll
            for(int n = 0; n < NT; ++n) {
#pragma unroll
                for(int i = 0; i < 4; ++i) {
                    const int rr = row + (i >> 1) * 8, tok = (tg0 + n) * 8 + 2 * t4 + (i & 1);
                    if(tok >= T || rr >= N) continue;
                    if(SK > 1) { part[(static_cast<std::size_t>(blockIdx.z) * T + tok) * N + rr] = acc[r][n][i]; continue; }
                    const float v = acc[r][n][i] * __bfloat162float(row_scale[rr]) * token_scale[tok];
                    float* p = Y + static_cast<std::size_t>(tok) * ldo + rr;
                    *p = add ? *p + v : v;
                }
            }
        }
    }
}

// GATED as in gemm_fp8_kernel; alpha scales the gate tile, alpha_up the up tile.
template <int NT, int RT, int PF, bool GATED = false>
__global__ void __launch_bounds__(256) gemm_fp4_kernel(const std::uint8_t* __restrict__ W, const std::uint8_t* __restrict__ S,
                                                      const uint4* __restrict__ X4, float alpha,
                                                      float* __restrict__ Y, float* __restrict__ part, int N, int K, int T,
                                                      int ldo, int add, __nv_bfloat16* __restrict__ G = nullptr, int act = 0,
                                                      float alpha_up = 0.f) {
    static_assert(!GATED || RT == 2, "the gated epilogue pairs a warp's two row tiles");
    __shared__ uint4 xsm[3][NT][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int rt0 = (blockIdx.y * 8 + warp) * RT;
    const int tg0 = blockIdx.x * NT;
    const int KT = K / 64;
    const int SK = gridDim.z, per = (KT + SK - 1) / SK, kt0 = blockIdx.z * per, kt1 = min(KT, kt0 + per);
    float acc[RT][NT][4];
#pragma unroll
    for(int r = 0; r < RT; ++r)
#pragma unroll
        for(int n = 0; n < NT; ++n) acc[r][n][0] = acc[r][n][1] = acc[r][n][2] = acc[r][n][3] = 0.f;
    auto stage = [&](int kt) {
        if(kt < kt1) {
            uint4 (*buf)[32] = xsm[kt % 3];
#pragma unroll
            for(int n = threadIdx.x >> 5; n < NT; n += 8)
                cp_async16(&buf[n][lane], X4 + (static_cast<std::size_t>(tg0 + n) * KT + kt) * 32 + lane);
        }
        cp_async_commit();
    };
    uint4 wv[PF][RT];
    std::uint32_t sv[PF][RT];
    auto load_w = [&](int kt, int i) {
        if(kt >= kt1) return;
#pragma unroll
        for(int r = 0; r < RT; ++r) {
            const std::size_t tile = static_cast<std::size_t>(rt0 + r) * KT + kt;
            wv[i][r] = load_streaming(W + tile * 512 + lane * 16);
            sv[i][r] = __ldg(reinterpret_cast<const std::uint32_t*>(S + tile * 64 + g * 8 + (t4 & 1) * 4));
        }
    };
#pragma unroll
    for(int i = 0; i < PF; ++i) load_w(kt0 + i, i);
    stage(kt0);
    stage(kt0 + 1);
    cp_async_wait<1>();
    __syncthreads();
    const bool hi = (t4 >> 1) != 0;
    for(int kt = kt0; kt < kt1; kt += PF) {
#pragma unroll
        for(int u = 0; u < PF; ++u) {
            const int kk = kt + u;
            if(kk >= kt1) break;
            stage(kk + 2);
            const uint4 (*xb)[32] = xsm[kk % 3];
#pragma unroll
            for(int r = 0; r < RT; ++r) {
                const uint4 w = wv[u][r];
                // own groups s0 = t >> 1 (a0/a1) and s0 + 2 (a2/a3); lane t ^ 2 holds the other half of each
                const std::uint32_t own0 = hi ? w.y : w.x, own2 = hi ? w.w : w.z;
                const std::uint32_t give0 = hi ? w.x : w.y, give2 = hi ? w.z : w.w;
                const std::uint32_t got0 = __shfl_xor_sync(0xffffffffu, give0, 2), got2 = __shfl_xor_sync(0xffffffffu, give2, 2);
                const std::uint32_t a0 = __byte_perm(own0, got0, 0x6420), a1 = __byte_perm(own0, got0, 0x7531);
                const std::uint32_t a2 = __byte_perm(own2, got2, 0x6420), a3 = __byte_perm(own2, got2, 0x7531);
                const std::uint32_t sa = sv[u][r];
#pragma unroll
                for(int n = 0; n < NT; ++n) {
                    const uint4 bb = xb[n][lane];
                    mma_nvfp4(acc[r][n], a0, a1, a2, a3, bb.x, bb.y, sa, bb.z);
                }
            }
            load_w(kk + PF, u);
            cp_async_wait<1>();
            __syncthreads();
        }
    }
    cp_async_wait<0>();
    if constexpr(GATED) {
#pragma unroll
        for(int n = 0; n < NT; ++n)
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const int tok = (tg0 + n) * 8 + 2 * t4 + (i & 1);
                if(tok >= T) continue;
                const float gate = acc[0][n][i] * alpha, up = acc[1][n][i] * alpha_up;
                G[static_cast<std::size_t>(tok) * ldo + rt0 / 2 * 16 + g + (i >> 1) * 8] = gated_value(act, gate, up);
            }
    } else {
#pragma unroll
        for(int r = 0; r < RT; ++r) {
            const int row = (rt0 + r) * 16 + g;
#pragma unroll
            for(int n = 0; n < NT; ++n) {
#pragma unroll
                for(int i = 0; i < 4; ++i) {
                    const int rr = row + (i >> 1) * 8, tok = (tg0 + n) * 8 + 2 * t4 + (i & 1);
                    if(tok >= T || rr >= N) continue;
                    if(SK > 1) { part[(static_cast<std::size_t>(blockIdx.z) * T + tok) * N + rr] = acc[r][n][i]; continue; }
                    const float v = acc[r][n][i] * alpha;
                    float* p = Y + static_cast<std::size_t>(tok) * ldo + rr;
                    *p = add ? *p + v : v;
                }
            }
        }
    }
}

// Fixed-order sum of SK partials, then FP8 row/token scales (row_scale non-null) or the NVFP4 alpha.
__global__ void finish_partials_kernel(const float* __restrict__ part, int SK, const __nv_bfloat16* __restrict__ row_scale,
                                       const float* __restrict__ token_scale, float alpha, float* __restrict__ Y, int N, int T,
                                       int ldo, int add) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if(i >= static_cast<std::size_t>(T) * N) return;
    const int tok = static_cast<int>(i / N), rr = static_cast<int>(i % N);
    float v = 0.f;
    for(int z = 0; z < SK; ++z) v += part[(static_cast<std::size_t>(z) * T + tok) * N + rr];
    v = row_scale ? v * __bfloat162float(row_scale[rr]) * token_scale[tok] : v * alpha;
    float* p = Y + static_cast<std::size_t>(tok) * ldo + rr;
    *p = add ? *p + v : v;
}

// The gated finish: gate and up of intermediate row j (interleaved rows 32 (j / 16) + j % 16 and 16 further) summed
// and scaled as finish_partials_kernel does, then G[tok][j] = bf16(act(gate) * up).
__global__ void finish_gated_kernel(const float* __restrict__ part, int SK, const __nv_bfloat16* __restrict__ row_scale,
                                    const float* __restrict__ token_scale, float alpha, float alpha_up, int act,
                                    __nv_bfloat16* __restrict__ G, int N, int T, int ldo) {
    const int I = N / 2;
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if(i >= static_cast<std::size_t>(T) * I) return;
    const int tok = static_cast<int>(i / I), j = static_cast<int>(i % I), rg = j / 16 * 32 + j % 16, ru = rg + 16;
    float gate = 0.f, up = 0.f;
    for(int z = 0; z < SK; ++z) {
        gate += part[(static_cast<std::size_t>(z) * T + tok) * N + rg];
        up += part[(static_cast<std::size_t>(z) * T + tok) * N + ru];
    }
    gate = row_scale ? gate * __bfloat162float(row_scale[rg]) * token_scale[tok] : gate * alpha;
    up = row_scale ? up * __bfloat162float(row_scale[ru]) * token_scale[tok] : up * alpha_up;
    G[static_cast<std::size_t>(tok) * ldo + j] = gated_value(act, gate, up);
}

// ------------------------------------------------------------------------------------------- host launch helpers
namespace {
// Blocks of the plain GEMM kernel for (format, nt, rt) the current device holds at once; 0 when unknown.
long resident_blocks(bool fp4, int nt, int rt) {
    int device = 0, sms = 0, per_sm = 0;
    if(cudaGetDevice(&device) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
        return 0;
    const void* k = nullptr;
    if(fp4) k = nt == 16 ? (rt == 2 ? reinterpret_cast<const void*>(gemm_fp4_kernel<16, 2, 4>) : reinterpret_cast<const void*>(gemm_fp4_kernel<16, 1, 4>))
                         : (rt == 2 ? reinterpret_cast<const void*>(gemm_fp4_kernel<8, 2, 4>) : reinterpret_cast<const void*>(gemm_fp4_kernel<8, 1, 4>));
    else k = nt == 16 ? (rt == 2 ? reinterpret_cast<const void*>(gemm_fp8_kernel<16, 2, 4>) : reinterpret_cast<const void*>(gemm_fp8_kernel<16, 1, 4>))
                      : (rt == 2 ? reinterpret_cast<const void*>(gemm_fp8_kernel<8, 2, 4>) : reinterpret_cast<const void*>(gemm_fp8_kernel<8, 1, 4>));
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, 256, 0) != cudaSuccess) return 0;
    return static_cast<long>(sms) * per_sm;
}
}  // namespace

Plan plan(bool fp4, int N, int K, int T, int wide_tokens, int min_blocks, bool two_row_tiles) {
    Plan p;
    p.fp4 = fp4;
    p.n = N;
    p.k = K;
    p.tokens = T;
    p.nt = T > wide_tokens ? 16 : 8;
    p.rt = two_row_tiles && N % 256 == 0 ? 2 : 1;
    const int tb = 8 * p.nt;
    p.tpad = (T + tb - 1) / tb * tb;
    const long blocks = static_cast<long>(p.tpad / tb) * (N / (128 * p.rt));
    if(min_blocks > 0) {
        while(p.sk < 8 && blocks * p.sk < min_blocks) p.sk *= 2;
    } else {
        // Split K only into the block slots the unsplit grid leaves idle in one wave (SM count x the occupancy of the
        // kernel this plan launches), never past it: each part adds T x N partials to write, read and sum, and a part
        // that starts a second wave runs after the first.
        const long slots = resident_blocks(fp4, p.nt, p.rt);
        while(p.sk < 8 && blocks * p.sk * 2 <= slots) p.sk *= 2;
    }
    p.activation_bytes = static_cast<std::size_t>(p.tpad) * K;  // FP8: 1 byte per value; NVFP4: 16 bytes per 16 values
    p.token_scale_count = fp4 ? 0 : static_cast<std::size_t>(T);
    p.partial_floats = p.sk > 1 ? static_cast<std::size_t>(p.sk) * T * N : 0;
    return p;
}

namespace {
// Tokens a launch with T tokens packs and computes: whole blocks of 8 nt tokens, not the plan's tpad. A token's
// accumulation order depends only on the plan (nt, rt, sk) and its activation scales only on its own row, so its
// result is the same for every T the plan serves.
int launch_tpad(const Plan& p, int T) {
    const int tb = 8 * p.nt;
    return (T + tb - 1) / tb * tb;
}

bool fp8_launch_ok(const Fp8Weights& w, int T, const Plan& p) {
    return !(w.N % (128 * p.rt) || w.K % 32 || T <= 0 || p.fp4 || w.N != p.n || w.K != p.k || T > p.tokens);
}
bool fp4_launch_ok(const Fp4Weights& w, int T, const Plan& p) {
    return !(w.N % (128 * p.rt) || w.K % 64 || T <= 0 || !p.fp4 || w.N != p.n || w.K != p.k || T > p.tokens ||
             !(w.weight_global_scale > 0.f) || !(w.input_global_scale > 0.f));
}
bool activation_ok(Activation act) {
    return act == Activation::silu || act == Activation::gelu_tanh || act == Activation::gelu;
}

const uint2* pack_fp8(const __nv_bfloat16* x, int T, int K, int tpad, const Scratch& s, cudaStream_t stream) {
    token_amax_kernel<<<T, 256, 0, stream>>>(x, K, s.token_scale);
    const std::size_t xe = static_cast<std::size_t>(tpad / 8) * (K / 32) * 32;
    uint2* x8 = static_cast<uint2*>(s.activations);
    pack_fp8_activations_kernel<<<static_cast<unsigned>((xe + 255) / 256), 256, 0, stream>>>(x, s.token_scale, T, K, tpad, x8);
    return x8;
}
const uint4* pack_fp4(const __nv_bfloat16* x, int T, int K, int tpad, float gx, const Scratch& s, cudaStream_t stream) {
    uint4* x4 = static_cast<uint4*>(s.activations);
    const std::size_t xe = static_cast<std::size_t>(tpad) * (K / 64);
    pack_fp4_activations_kernel<<<static_cast<unsigned>((xe + 127) / 128), 128, 0, stream>>>(x, T, K, tpad, gx, x4);
    return x4;
}

// The plain kernels for a plan (they also write the partials of a split gated launch).
void launch_fp8(const Fp8Weights& w, const uint2* x8, const float* ts, int T, float* y, int ldo, bool add, const Plan& p,
                float* partials, dim3 grid, cudaStream_t stream) {
#define CRIT_ST_FP8(NT_, RT_) gemm_fp8_kernel<NT_, RT_, 4><<<grid, 256, 0, stream>>>(w.tiles, w.row_scale, x8, ts, y, partials, w.N, w.K, T, ldo, add ? 1 : 0)
    if(p.nt == 16 && p.rt == 2) CRIT_ST_FP8(16, 2);
    else if(p.nt == 16) CRIT_ST_FP8(16, 1);
    else if(p.rt == 2) CRIT_ST_FP8(8, 2);
    else CRIT_ST_FP8(8, 1);
#undef CRIT_ST_FP8
}
void launch_fp4(const Fp4Weights& w, const uint4* x4, float alpha, int T, float* y, int ldo, bool add, const Plan& p,
                float* partials, dim3 grid, cudaStream_t stream) {
#define CRIT_ST_FP4(NT_, RT_) gemm_fp4_kernel<NT_, RT_, 4><<<grid, 256, 0, stream>>>(w.tiles, w.scale_tiles, x4, alpha, y, partials, w.N, w.K, T, ldo, add ? 1 : 0)
    if(p.nt == 16 && p.rt == 2) CRIT_ST_FP4(16, 2);
    else if(p.nt == 16) CRIT_ST_FP4(16, 1);
    else if(p.rt == 2) CRIT_ST_FP4(8, 2);
    else CRIT_ST_FP4(8, 1);
#undef CRIT_ST_FP4
}
unsigned blocks_for(std::size_t n) { return static_cast<unsigned>((n + 255) / 256); }
}  // namespace

cudaError_t enqueue_fp8(const Fp8Weights& w, const __nv_bfloat16* x, int T, float* y, int ldo, bool add,
                               const Plan& p, const Scratch& s, cudaStream_t stream) {
    if(!fp8_launch_ok(w, T, p)) return cudaErrorInvalidValue;
    const int tpad = launch_tpad(p, T);
    const uint2* x8 = pack_fp8(x, T, w.K, tpad, s, stream);
    launch_fp8(w, x8, s.token_scale, T, y, ldo, add, p, s.partials, dim3(tpad / (8 * p.nt), w.N / (128 * p.rt), p.sk), stream);
    if(p.sk > 1)
        finish_partials_kernel<<<blocks_for(static_cast<std::size_t>(T) * w.N), 256, 0, stream>>>(
            s.partials, p.sk, w.row_scale, s.token_scale, 1.f, y, w.N, T, ldo, add ? 1 : 0);
    return cudaGetLastError();
}

cudaError_t enqueue_fp4(const Fp4Weights& w, const __nv_bfloat16* x, int T, float* y, int ldo, bool add,
                               const Plan& p, const Scratch& s, cudaStream_t stream) {
    if(!fp4_launch_ok(w, T, p)) return cudaErrorInvalidValue;
    const int tpad = launch_tpad(p, T);
    const uint4* x4 = pack_fp4(x, T, w.K, tpad, w.input_global_scale, s, stream);
    const float alpha = 1.f / (w.weight_global_scale * w.input_global_scale);
    launch_fp4(w, x4, alpha, T, y, ldo, add, p, s.partials, dim3(tpad / (8 * p.nt), w.N / (128 * p.rt), p.sk), stream);
    if(p.sk > 1)
        finish_partials_kernel<<<blocks_for(static_cast<std::size_t>(T) * w.N), 256, 0, stream>>>(
            s.partials, p.sk, nullptr, nullptr, alpha, y, w.N, T, ldo, add ? 1 : 0);
    return cudaGetLastError();
}

cudaError_t enqueue_fp8_gated(const Fp8Weights& w, const __nv_bfloat16* x, int T, Activation act, __nv_bfloat16* out,
                              int ldo, const Plan& p, const Scratch& s, cudaStream_t stream) {
    if(p.rt != 2 || !fp8_launch_ok(w, T, p) || !activation_ok(act) || ldo < w.N / 2) return cudaErrorInvalidValue;
    const int tpad = launch_tpad(p, T);
    const uint2* x8 = pack_fp8(x, T, w.K, tpad, s, stream);
    const dim3 grid(tpad / (8 * p.nt), w.N / 256, p.sk);
    if(p.sk > 1) {
        launch_fp8(w, x8, s.token_scale, T, nullptr, 0, false, p, s.partials, grid, stream);
        finish_gated_kernel<<<blocks_for(static_cast<std::size_t>(T) * (w.N / 2)), 256, 0, stream>>>(
            s.partials, p.sk, w.row_scale, s.token_scale, 1.f, 1.f, static_cast<int>(act), out, w.N, T, ldo);
    } else if(p.nt == 16) {
        gemm_fp8_kernel<16, 2, 4, true><<<grid, 256, 0, stream>>>(w.tiles, w.row_scale, x8, s.token_scale, nullptr, nullptr, w.N, w.K, T,
                                                                   ldo, 0, out, static_cast<int>(act));
    } else {
        gemm_fp8_kernel<8, 2, 4, true><<<grid, 256, 0, stream>>>(w.tiles, w.row_scale, x8, s.token_scale, nullptr, nullptr, w.N, w.K, T,
                                                                  ldo, 0, out, static_cast<int>(act));
    }
    return cudaGetLastError();
}

cudaError_t enqueue_fp4_gated(const Fp4Weights& w, float up_weight_global_scale, const __nv_bfloat16* x, int T,
                              Activation act, __nv_bfloat16* out, int ldo, const Plan& p, const Scratch& s,
                              cudaStream_t stream) {
    if(p.rt != 2 || !fp4_launch_ok(w, T, p) || !(up_weight_global_scale > 0.f) || !activation_ok(act) || ldo < w.N / 2)
        return cudaErrorInvalidValue;
    const int tpad = launch_tpad(p, T);
    const uint4* x4 = pack_fp4(x, T, w.K, tpad, w.input_global_scale, s, stream);
    const float alpha = 1.f / (w.weight_global_scale * w.input_global_scale);
    const float alpha_up = 1.f / (up_weight_global_scale * w.input_global_scale);
    const dim3 grid(tpad / (8 * p.nt), w.N / 256, p.sk);
    if(p.sk > 1) {
        launch_fp4(w, x4, alpha, T, nullptr, 0, false, p, s.partials, grid, stream);
        finish_gated_kernel<<<blocks_for(static_cast<std::size_t>(T) * (w.N / 2)), 256, 0, stream>>>(
            s.partials, p.sk, nullptr, nullptr, alpha, alpha_up, static_cast<int>(act), out, w.N, T, ldo);
    } else if(p.nt == 16) {
        gemm_fp4_kernel<16, 2, 4, true><<<grid, 256, 0, stream>>>(w.tiles, w.scale_tiles, x4, alpha, nullptr, nullptr, w.N, w.K, T, ldo, 0,
                                                                   out, static_cast<int>(act), alpha_up);
    } else {
        gemm_fp4_kernel<8, 2, 4, true><<<grid, 256, 0, stream>>>(w.tiles, w.scale_tiles, x4, alpha, nullptr, nullptr, w.N, w.K, T, ldo, 0,
                                                                  out, static_cast<int>(act), alpha_up);
    }
    return cudaGetLastError();
}

cudaError_t enqueue_repack_fp8_rows(const std::uint8_t* codes, std::uint8_t* tiles, int N, int K, int first, int step,
                                    cudaStream_t stream) {
    if(N <= 0 || K <= 0 || N % 16 || K % 32 || first < 0 || step < 1) return cudaErrorInvalidValue;
    const std::size_t threads = static_cast<std::size_t>(N / 16) * (K / 32) * 32;
    repack_fp8_kernel<<<blocks_for(threads), 256, 0, stream>>>(codes, tiles, N, K, first, step);
    return cudaGetLastError();
}

cudaError_t enqueue_repack_fp4_rows(const std::uint8_t* codes, const std::uint8_t* scales, std::uint8_t* tiles,
                                    std::uint8_t* scale_tiles, int N, int K, int first, int step, cudaStream_t stream) {
    if(N <= 0 || K <= 0 || N % 16 || K % 64 || first < 0 || step < 1) return cudaErrorInvalidValue;
    const std::size_t threads = static_cast<std::size_t>(N / 16) * (K / 64) * 32;
    repack_fp4_kernel<<<blocks_for(threads), 256, 0, stream>>>(codes, scales, tiles, scale_tiles, N, K, first, step);
    return cudaGetLastError();
}

cudaError_t enqueue_repack_fp8(const std::uint8_t* codes, std::uint8_t* tiles, int N, int K, cudaStream_t stream) {
    return enqueue_repack_fp8_rows(codes, tiles, N, K, 0, 1, stream);
}

cudaError_t enqueue_repack_fp4(const std::uint8_t* codes, const std::uint8_t* scales, std::uint8_t* tiles,
                               std::uint8_t* scale_tiles, int N, int K, cudaStream_t stream) {
    return enqueue_repack_fp4_rows(codes, scales, tiles, scale_tiles, N, K, 0, 1, stream);
}

cudaError_t enqueue_place_row_scales(const __nv_bfloat16* src, __nv_bfloat16* dst, int N, int first, int step,
                                     cudaStream_t stream) {
    if(N <= 0 || N % 16 || first < 0 || step < 1) return cudaErrorInvalidValue;
    place_row_scales_kernel<<<blocks_for(static_cast<std::size_t>(N)), 256, 0, stream>>>(src, dst, N, first, step);
    return cudaGetLastError();
}

namespace {
__global__ void fp4_compiled_kernel(int* out) { *out = CRIT_STORED_TILE_FP4_MMA; }
}  // namespace

bool fp4_mma_compiled() {
    int* d = nullptr;
    int h = 0;
    if(cudaMalloc(&d, sizeof(int)) != cudaSuccess) return false;
    fp4_compiled_kernel<<<1, 1>>>(d);
    const bool ok = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(d);
    return ok && h == 1;
}

}  // namespace crit::stored_tile
