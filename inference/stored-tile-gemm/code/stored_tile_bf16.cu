// BF16-activation kernels over the stored tiles: the decode / verify GEMV and the W8A16 / W4A16 prompt GEMMs. See
// stored_tile_gemm.h for the interface.
#include "stored_tile_gemm.h"

#include "stored_tile_device.cuh"

namespace crit::stored_tile {
namespace {

using namespace detail;

constexpr int kMaxRows = 16;

struct GemvArgs {
    const std::uint8_t* tiles;
    const std::uint8_t* scale_tiles;  // NVFP4
    const __nv_bfloat16* row_scale;   // FP8
    float wgs, wgs_up;                // NVFP4 divisors (gate / up of a gated launch)
    int N, K;
    const __nv_bfloat16* x;
    int ldx, M;
    float* y;
    int ldy, add;
    __nv_bfloat16* out;  // gated
    int ldo, act;
};

// RB row pairs per block, G = W / RB warps per pair: warp q of pair rp takes K tiles q, q + G, ... and chains one
// m16n8k16 per 16 columns and 8-row group, in K order. Each lane's 16 tile bytes are its A fragments; B fragments are
// the activations of rows 8 n + g, loaded one step ahead. Warps q and G + q of a block read the same activations at
// the same time (shared in L1). A pair's G partials are summed in warp order in shared memory, then scaled.
template <bool FP4, int NT, int W, int RB, int PF, bool GATED>
__global__ void __launch_bounds__(W * 32) gemv_kernel(const GemvArgs a) {
    constexpr int RT = 2, G = W / RB, KS = FP4 ? 4 : 2, KW = 16 * KS;
    static_assert(PF % 2 == 0, "the activation double buffer alternates with the step parity");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int rp = warp / G, q = warp % G;
    const int tiles_n = a.N / 16, KT = a.K / KW;
    const int rt0 = (static_cast<int>(blockIdx.x) * RB + rp) * RT;  // the pair's first row tile
    const int J = KT > q ? (KT - q + G - 1) / G : 0;
    const std::size_t wstride = static_cast<std::size_t>(KT) * 512, sstride = static_cast<std::size_t>(KT) * 64;
    const std::uint8_t* wb = a.tiles + static_cast<std::size_t>(rt0) * wstride + lane * 16;
    const std::uint8_t* sb = FP4 ? a.scale_tiles + static_cast<std::size_t>(rt0) * sstride + g * 8 : nullptr;
    bool tv[RT];
#pragma unroll
    for(int r = 0; r < RT; ++r) tv[r] = rt0 + r < tiles_n;

    const __nv_bfloat16* xp[NT];
    bool xv[NT];
#pragma unroll
    for(int n = 0; n < NT; ++n) {
        const int row = 8 * n + g;
        xv[n] = row < a.M;
        xp[n] = a.x + static_cast<std::size_t>(xv[n] ? row : 0) * a.ldx + 2 * t4;
    }
    float acc[RT][NT][4];
#pragma unroll
    for(int r = 0; r < RT; ++r)
#pragma unroll
        for(int n = 0; n < NT; ++n) acc[r][n][0] = acc[r][n][1] = acc[r][n][2] = acc[r][n][3] = 0.f;

    uint4 wv[PF][RT];
    uint2 sv[PF][RT];
#pragma unroll
    for(int i = 0; i < PF; ++i)
#pragma unroll
        for(int r = 0; r < RT; ++r) {
            wv[i][r] = make_uint4(0u, 0u, 0u, 0u);
            sv[i][r] = make_uint2(0u, 0u);
        }
    auto load = [&](int j, int slot, int r) {
        if(j < J && tv[r]) {
            const int kt = q + j * G;
            wv[slot][r] = load_streaming(wb + r * wstride + static_cast<std::size_t>(kt) * 512);
            if constexpr(FP4) sv[slot][r] = load_streaming8(sb + r * sstride + static_cast<std::size_t>(kt) * 64);
        }
    };
    std::uint32_t bq[2][NT][KS][2];
    auto load_b = [&](int j, int buf) {
        if(j < J) {
            const int kt = q + j * G;
#pragma unroll
            for(int n = 0; n < NT; ++n)
#pragma unroll
                for(int s = 0; s < KS; ++s) {
                    const __nv_bfloat16* p = xp[n] + kt * KW + 16 * s;
                    bq[buf][n][s][0] = xv[n] ? __ldg(reinterpret_cast<const unsigned int*>(p)) : 0u;
                    bq[buf][n][s][1] = xv[n] ? __ldg(reinterpret_cast<const unsigned int*>(p + 8)) : 0u;
                }
        }
    };
#pragma unroll
    for(int i = 0; i < PF; ++i)
#pragma unroll
        for(int r = 0; r < RT; ++r) load(i, i, r);
    load_b(0, 0);
    for(int j0 = 0; j0 < J; j0 += PF) {
#pragma unroll
        for(int u = 0; u < PF; ++u) {
            const int j = j0 + u;
            if(j >= J) break;
            const int cur = u & 1;  // j0 is a multiple of PF (even), so this is j's parity
            load_b(j + 1, cur ^ 1);
            const std::uint32_t (&b)[NT][KS][2] = bq[cur];
#pragma unroll
            for(int r = 0; r < RT; ++r) {
                const std::uint32_t w[4] = {wv[u][r].x, wv[u][r].y, wv[u][r].z, wv[u][r].w};
                const uint2 sc = sv[u][r];
                load(j + PF, u, r);  // the slot's values are held above; refill it now
#pragma unroll
                for(int s = 0; s < KS; ++s) {
                    std::uint32_t af[4];
                    if constexpr(FP4) fp4_a_frag(w, sc, s, af);
                    else fp8_a_frag(w, s, af);
#pragma unroll
                    for(int n = 0; n < NT; ++n) mma_bf16(acc[r][n], af[0], af[1], af[2], af[3], b[n][s][0], b[n][s][1]);
                }
            }
        }
    }

    // Fixed-order reduction over each pair's G warps (K groups 0, 1, ..., G - 1), then the scale.
    __shared__ float4 red[W * RT * NT * 32];
#pragma unroll
    for(int r = 0; r < RT; ++r)
#pragma unroll
        for(int n = 0; n < NT; ++n)
            red[((warp * RT + r) * NT + n) * 32 + lane] = make_float4(acc[r][n][0], acc[r][n][1], acc[r][n][2], acc[r][n][3]);
    __syncthreads();
    const float* rf = reinterpret_cast<const float*>(red);
    constexpr int per_warp = RT * NT * 128, per_tile = NT * 128;
    auto pair_sum = [&](int pr, int f) {
        const float* src = rf + static_cast<std::size_t>(pr) * G * per_warp + f;
        float s = src[0];
#pragma unroll
        for(int w = 1; w < G; ++w) s = __fadd_rn(s, src[w * per_warp]);
        return s;
    };
    // element f of a pair: i = f & 3 (C register), l = lane, n = (f >> 7) % NT, r = (f >> 7) / NT
    auto token_of = [](int f) { return 8 * ((f >> 7) % NT) + 2 * ((f >> 2) & 3) + (f & 1); };
    auto row_in_tile = [](int f) { return ((f >> 4) & 7) + ((f >> 1) & 1) * 8; };
    if constexpr(!GATED) {
        for(int e = threadIdx.x; e < RB * per_warp; e += W * 32) {
            const int pr = e / per_warp, f = e % per_warp, tok = token_of(f);
            const int row = ((static_cast<int>(blockIdx.x) * RB + pr) * RT + (f >> 7) / NT) * 16 + row_in_tile(f);
            if(tok >= a.M || row >= a.N) continue;
            const float s = pair_sum(pr, f);
            const float v = FP4 ? __fdiv_rn(s, a.wgs) : __fmul_rn(s, __bfloat162float(a.row_scale[row]));
            float* p = a.y + static_cast<std::size_t>(tok) * a.ldy + row;
            *p = a.add ? __fadd_rn(*p, v) : v;
        }
    } else {
        for(int e = threadIdx.x; e < RB * per_tile; e += W * 32) {
            const int pr = e / per_tile, f = e % per_tile, tok = token_of(f);
            const int pair = static_cast<int>(blockIdx.x) * RB + pr, rg = pair * 32 + row_in_tile(f);
            if(tok >= a.M || rg >= a.N) continue;
            const float sg = pair_sum(pr, f), su = pair_sum(pr, f + per_tile);
            const float gate = FP4 ? __fdiv_rn(sg, a.wgs) : __fmul_rn(sg, __bfloat162float(a.row_scale[rg]));
            const float up = FP4 ? __fdiv_rn(su, a.wgs_up) : __fmul_rn(su, __bfloat162float(a.row_scale[rg + 16]));
            a.out[static_cast<std::size_t>(tok) * a.ldo + pair * 16 + row_in_tile(f)] = gated_value(a.act, gate, up);
        }
    }
}

// Weight steps in flight per warp (even, for the activation double buffer). It changes no accumulation order; the
// NVFP4 kernel and two 8-row groups keep two steps so the kernel stays within its registers.
template <bool FP4, int NT>
constexpr int ring_depth() { return FP4 || NT == 2 ? 2 : 4; }

// groups -> (warps, row pairs per block): 4 -> (8, 2), 8 -> (16, 2), 16 -> (16, 1); a block of 16 warps with two
// pairs and one with one pair of 8 warps would both give 8 groups, so the choice keeps one shape per group count.
template <bool FP4, bool GATED>
cudaError_t launch_gemv(const GemvArgs& a, int groups, cudaStream_t stream) {
    const int pairs = (a.N / 16 + 1) / 2;
#define CRIT_ST_GEMV(W_, RB_)                                                                                         \
    do {                                                                                                               \
        const dim3 grid(static_cast<unsigned>((pairs + RB_ - 1) / RB_));                                              \
        if(a.M > 8) gemv_kernel<FP4, 2, W_, RB_, ring_depth<FP4, 2>(), GATED><<<grid, W_ * 32, 0, stream>>>(a);        \
        else gemv_kernel<FP4, 1, W_, RB_, ring_depth<FP4, 1>(), GATED><<<grid, W_ * 32, 0, stream>>>(a);               \
    } while(0)
    if(groups == 4) CRIT_ST_GEMV(8, 2);
    else if(groups == 8) CRIT_ST_GEMV(16, 2);
    else CRIT_ST_GEMV(16, 1);
#undef CRIT_ST_GEMV
    return cudaGetLastError();
}

bool aligned(const void* p, std::size_t b) { return (reinterpret_cast<std::uintptr_t>(p) & (b - 1)) == 0; }

bool gemv_ok(const GemvPlan& p, bool fp4, int N, int K, const std::uint8_t* tiles, const __nv_bfloat16* x, int ldx, int M) {
    return p.fp4 == fp4 && p.n == N && p.k == K && (p.groups == 4 || p.groups == 8 || p.groups == 16) && N > 0 && K > 0 &&
           N % 16 == 0 && K % (fp4 ? 64 : 32) == 0 && M >= 1 && M <= kMaxRows && tiles && aligned(tiles, 16) && x &&
           aligned(x, 4) && ldx >= K && ldx % 2 == 0;
}
bool activation_ok(Activation act) {
    return act == Activation::silu || act == Activation::gelu_tanh || act == Activation::gelu;
}

// ------------------------------------------------------------------------------------------- BF16-activation GEMM
// Block: 8 warps x 2 row tiles = 256 rows (a warp's two tiles are the gate and up tiles of the interleaved layout) by
// 64 tokens. Each lane decodes its tile bytes exactly to BF16 as the GEMV does; weights stream through a register
// ring, activations through a cp.async ring in shared memory (64 columns per stage, 16-byte chunks XOR-swizzled by
// token, zero-filled past T and K) and reach the MMA through ldmatrix. Accumulation per (row, token): MMA chains of
// 32 16-column steps (512 columns) in K order, each folded into an FP32 running sum in K order (one chain over a long
// K loses accuracy to the tensor core's accumulation); split parts are summed in part order by a finish kernel.
constexpr int kA16Warps = 8, kA16RT = 2, kA16Rows = kA16Warps * kA16RT * 16;  // 256
constexpr int kA16NT = 8;       // 8-token groups per block (64 tokens)
constexpr int kA16Chunk = 64;   // K columns per pipeline stage
constexpr int kA16Stages = 4;   // activation stages in shared memory
constexpr int kA16Ring = 2;     // weight chunks in flight per warp
constexpr int kA16Fold = 8;     // chunks (512 columns) per MMA chain

struct A16Args {
    const std::uint8_t* tiles;
    const std::uint8_t* scale_tiles;  // NVFP4
    const __nv_bfloat16* row_scale;   // FP8
    float wgs, wgs_up;                // NVFP4 divisors (gate / up of a gated launch)
    int N, K;
    const __nv_bfloat16* x;           // [T][K]
    int T;
    float* y;
    int ldy, add;
    float* part;                      // [sk][T][N] when split
    __nv_bfloat16* out;               // gated
    int ldo, act;
};

__device__ __forceinline__ void cp_async16_zfill(void* smem, const void* gmem, int src_bytes) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void ldsm_x4(std::uint32_t (&r)[4], const void* p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
// The scale after the reduction, as the GEMV applies it: FP8 row scale (a multiply), NVFP4 global scale (a division).
template <bool FP4>
__device__ __forceinline__ float a16_scale(const A16Args& a, float s, int row, bool up) {
    if constexpr(FP4) return __fdiv_rn(s, up ? a.wgs_up : a.wgs);
    else return __fmul_rn(s, __bfloat162float(a.row_scale[row]));
}

template <bool FP4, bool GATED>
__global__ void __launch_bounds__(kA16Warps * 32, 1) a16_kernel(const A16Args a) {
    constexpr int NT = kA16NT, BN = 8 * NT, RT = kA16RT, TPC = FP4 ? 1 : 2;  // tiles per 64-column chunk
    __shared__ __align__(16) uint4 xs[kA16Stages][BN * 8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int rt0 = (static_cast<int>(blockIdx.y) * kA16Warps + warp) * RT;
    const int tok0 = static_cast<int>(blockIdx.x) * BN;
    const int tiles_n = a.N / 16, KT = a.K / (FP4 ? 64 : 32), KC = (a.K + kA16Chunk - 1) / kA16Chunk;
    const int SK = gridDim.z, per = (KC + SK - 1) / SK, c0 = static_cast<int>(blockIdx.z) * per, c1 = min(KC, c0 + per);
    const std::size_t wstride = static_cast<std::size_t>(KT) * 512, sstride = static_cast<std::size_t>(KT) * 64;
    const std::uint8_t* wb = a.tiles + static_cast<std::size_t>(rt0) * wstride + lane * 16;
    const std::uint8_t* sb = FP4 ? a.scale_tiles + static_cast<std::size_t>(rt0) * sstride + g * 8 : nullptr;
    bool tv[RT];
#pragma unroll
    for(int r = 0; r < RT; ++r) tv[r] = rt0 + r < tiles_n;

    float acc[RT][NT][4], tot[RT][NT][4];
#pragma unroll
    for(int r = 0; r < RT; ++r)
#pragma unroll
        for(int n = 0; n < NT; ++n)
#pragma unroll
            for(int i = 0; i < 4; ++i) acc[r][n][i] = tot[r][n][i] = 0.f;
    auto fold = [&]() {
#pragma unroll
        for(int r = 0; r < RT; ++r)
#pragma unroll
            for(int n = 0; n < NT; ++n)
#pragma unroll
                for(int i = 0; i < 4; ++i) {
                    tot[r][n][i] = __fadd_rn(tot[r][n][i], acc[r][n][i]);
                    acc[r][n][i] = 0.f;
                }
    };
    auto stage = [&](int c) {
        if(c < c1) {
            uint4* buf = xs[c % kA16Stages];
#pragma unroll
            for(int i = threadIdx.x; i < BN * 8; i += kA16Warps * 32) {
                const int r = i >> 3, q = i & 7, tok = tok0 + r, col = c * kA16Chunk + q * 8;
                const bool ok = tok < a.T && col < a.K;
                cp_async16_zfill(&buf[r * 8 + (q ^ (r & 7))], a.x + (ok ? static_cast<std::size_t>(tok) * a.K + col : 0), ok ? 16 : 0);
            }
        }
        cp_async_commit();
    };
    uint4 wv[kA16Ring][RT][TPC];
    uint2 sv[kA16Ring][RT];
    auto load_w = [&](int c, int slot) {
        if(c >= c1) return;
#pragma unroll
        for(int r = 0; r < RT; ++r) {
#pragma unroll
            for(int u = 0; u < TPC; ++u) {
                const int kt = c * TPC + u;  // an FP8 tail chunk (K % 64 == 32) has one tile; its other half is zero
                wv[slot][r][u] = tv[r] && kt < KT ? load_streaming(wb + r * wstride + static_cast<std::size_t>(kt) * 512) : make_uint4(0u, 0u, 0u, 0u);
            }
            if constexpr(FP4) sv[slot][r] = tv[r] ? load_streaming8(sb + r * sstride + static_cast<std::size_t>(c) * 64) : make_uint2(0u, 0u);
        }
    };

#pragma unroll
    for(int i = 0; i < kA16Ring; ++i) load_w(c0 + i, i);
#pragma unroll
    for(int i = 0; i < kA16Stages - 1; ++i) stage(c0 + i);
    cp_async_wait<kA16Stages - 2>();
    __syncthreads();
    for(int cb = c0; cb < c1; cb += kA16Ring) {
#pragma unroll
        for(int u = 0; u < kA16Ring; ++u) {
            const int c = cb + u;
            if(c >= c1) break;
            stage(c + kA16Stages - 1);
            const uint4* xb = xs[c % kA16Stages];
            std::uint32_t w[RT][TPC][4];
            uint2 sc[RT];
#pragma unroll
            for(int r = 0; r < RT; ++r) {
#pragma unroll
                for(int v = 0; v < TPC; ++v) {
                    w[r][v][0] = wv[u][r][v].x;
                    w[r][v][1] = wv[u][r][v].y;
                    w[r][v][2] = wv[u][r][v].z;
                    w[r][v][3] = wv[u][r][v].w;
                }
                sc[r] = FP4 ? sv[u][r] : make_uint2(0u, 0u);
            }
            load_w(c + kA16Ring, u);
#pragma unroll
            for(int s = 0; s < 4; ++s) {  // 16-column steps of the chunk
                std::uint32_t af[RT][4];
#pragma unroll
                for(int r = 0; r < RT; ++r) {
                    if constexpr(FP4) fp4_a_frag(w[r][0], sc[r], s, af[r]);
                    else fp8_a_frag(w[r][s >> 1], s & 1, af[r]);
                }
#pragma unroll
                for(int p = 0; p < NT / 2; ++p) {
                    // matrices: (tokens 16p.., cols 16s..+7), (16p.., +8..), (16p+8.., 16s..), (16p+8.., +8..)
                    const int r = 16 * p + ((lane >> 4) << 3) + (lane & 7), q = 2 * s + ((lane >> 3) & 1);
                    std::uint32_t b[4];
                    ldsm_x4(b, &xb[r * 8 + (q ^ (r & 7))]);
#pragma unroll
                    for(int rr = 0; rr < RT; ++rr) {
                        mma_bf16(acc[rr][2 * p], af[rr][0], af[rr][1], af[rr][2], af[rr][3], b[0], b[1]);
                        mma_bf16(acc[rr][2 * p + 1], af[rr][0], af[rr][1], af[rr][2], af[rr][3], b[2], b[3]);
                    }
                }
            }
            if((c - c0 + 1) % kA16Fold == 0) fold();
            cp_async_wait<kA16Stages - 2>();
            __syncthreads();
        }
    }
    cp_async_wait<0>();
    if((c1 - c0) % kA16Fold != 0) fold();  // the last, partial chain

    // C fragment: i = 0 (row g, token 2t), 1 (row g, 2t + 1), 2 (row g + 8, 2t), 3 (row g + 8, 2t + 1)
    if constexpr(GATED) {
#pragma unroll
        for(int n = 0; n < NT; ++n)
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const int rg = rt0 * 16 + g + (i >> 1) * 8, tok = tok0 + 8 * n + 2 * t4 + (i & 1);
                if(tok >= a.T || rg >= a.N) continue;
                const float gate = a16_scale<FP4>(a, tot[0][n][i], rg, false), up = a16_scale<FP4>(a, tot[1][n][i], rg + 16, true);
                a.out[static_cast<std::size_t>(tok) * a.ldo + rt0 / 2 * 16 + g + (i >> 1) * 8] = gated_value(a.act, gate, up);
            }
    } else {
#pragma unroll
        for(int r = 0; r < RT; ++r)
#pragma unroll
            for(int n = 0; n < NT; ++n)
#pragma unroll
                for(int i = 0; i < 4; ++i) {
                    const int row = (rt0 + r) * 16 + g + (i >> 1) * 8, tok = tok0 + 8 * n + 2 * t4 + (i & 1);
                    if(tok >= a.T || row >= a.N) continue;
                    if(SK > 1) {
                        a.part[(static_cast<std::size_t>(blockIdx.z) * a.T + tok) * a.N + row] = tot[r][n][i];
                        continue;
                    }
                    const float v = a16_scale<FP4>(a, tot[r][n][i], row, false);
                    float* p = a.y + static_cast<std::size_t>(tok) * a.ldy + row;
                    *p = a.add ? __fadd_rn(*p, v) : v;
                }
    }
}

// Fixed-order sum of the split parts (part 0, 1, ...), then the scale and the optional add, or the gated output.
template <bool FP4>
__global__ void a16_finish_kernel(const A16Args a, int SK) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if(i >= static_cast<std::size_t>(a.T) * a.N) return;
    const int tok = static_cast<int>(i / a.N), row = static_cast<int>(i % a.N);
    float v = a.part[i];
    for(int z = 1; z < SK; ++z) v = __fadd_rn(v, a.part[(static_cast<std::size_t>(z) * a.T + tok) * a.N + row]);
    v = a16_scale<FP4>(a, v, row, false);
    float* p = a.y + static_cast<std::size_t>(tok) * a.ldy + row;
    *p = a.add ? __fadd_rn(*p, v) : v;
}
template <bool FP4>
__global__ void a16_finish_gated_kernel(const A16Args a, int SK) {
    const int I = a.N / 2;
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if(i >= static_cast<std::size_t>(a.T) * I) return;
    const int tok = static_cast<int>(i / I), j = static_cast<int>(i % I), rg = j / 16 * 32 + j % 16;
    const float* pg = a.part + static_cast<std::size_t>(tok) * a.N + rg;
    float sg = pg[0], su = pg[16];
    for(int z = 1; z < SK; ++z) {
        sg = __fadd_rn(sg, pg[static_cast<std::size_t>(z) * a.T * a.N]);
        su = __fadd_rn(su, pg[static_cast<std::size_t>(z) * a.T * a.N + 16]);
    }
    a.out[static_cast<std::size_t>(tok) * a.ldo + j] = gated_value(a.act, a16_scale<FP4>(a, sg, rg, false), a16_scale<FP4>(a, su, rg + 16, true));
}

template <bool FP4, bool GATED>
cudaError_t launch_a16(const A16Args& a, int sk, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>((a.T + 8 * kA16NT - 1) / (8 * kA16NT)), static_cast<unsigned>((a.N + kA16Rows - 1) / kA16Rows),
                    static_cast<unsigned>(sk));
    if(sk == 1) {
        a16_kernel<FP4, GATED><<<grid, kA16Warps * 32, 0, stream>>>(a);
    } else {
        a16_kernel<FP4, false><<<grid, kA16Warps * 32, 0, stream>>>(a);  // partials
        const std::size_t n = static_cast<std::size_t>(a.T) * (GATED ? a.N / 2 : a.N);
        if(GATED) a16_finish_gated_kernel<FP4><<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(a, sk);
        else a16_finish_kernel<FP4><<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(a, sk);
    }
    return cudaGetLastError();
}

bool a16_ok(const A16Plan& p, bool fp4, int N, int K, int T, const std::uint8_t* tiles, const __nv_bfloat16* x, const float* partials) {
    return p.fp4 == fp4 && p.n == N && p.k == K && T >= 1 && T <= p.tokens && p.sk >= 1 && p.sk <= 8 && N > 0 && N % 16 == 0 &&
           K > 0 && K % (fp4 ? 64 : 32) == 0 && tiles && aligned(tiles, 16) && x && aligned(x, 16) && (p.sk == 1 || partials);
}

}  // namespace

GemvPlan plan_gemv(bool fp4, int N, int K, int groups) {
    GemvPlan p;
    p.fp4 = fp4;
    p.n = N;
    p.k = K;
    if(groups > 0) {
        p.groups = groups;
        return p;
    }
    // The largest group count whose grid (pairs x groups warps) still fits in one wave of the device's resident warps
    // (SM count x the decode kernel's occupancy): splitting K fills SMs a narrow matrix would leave idle, and a grid
    // beyond one wave only adds partial sums.
    int device = 0, sms = 0;
    if(cudaGetDevice(&device) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
        sms = 0;
    const long pairs = (N / 16 + 1) / 2;
    p.groups = 4;
    for(const int g : {8, 16}) {
        const void* k = fp4 ? (g == 8 ? reinterpret_cast<const void*>(gemv_kernel<true, 1, 16, 2, ring_depth<true, 1>(), false>)
                                      : reinterpret_cast<const void*>(gemv_kernel<true, 1, 16, 1, ring_depth<true, 1>(), false>))
                            : (g == 8 ? reinterpret_cast<const void*>(gemv_kernel<false, 1, 16, 2, ring_depth<false, 1>(), false>)
                                      : reinterpret_cast<const void*>(gemv_kernel<false, 1, 16, 1, ring_depth<false, 1>(), false>));
        int per_sm = 0;
        if(sms <= 0 || cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, 16 * 32, 0) != cudaSuccess) break;
        if(pairs * g > static_cast<long>(sms) * per_sm * 16) break;
        p.groups = g;
    }
    return p;
}

cudaError_t enqueue_gemv_fp8(const Fp8Weights& w, const __nv_bfloat16* x, int ldx, int M, float* y, int ldy, bool add,
                             const GemvPlan& p, cudaStream_t stream) {
    if(!gemv_ok(p, false, w.N, w.K, w.tiles, x, ldx, M) || !w.row_scale || !y || ldy < w.N) return cudaErrorInvalidValue;
    const GemvArgs a{w.tiles, nullptr, w.row_scale, 1.f, 1.f, w.N, w.K, x, ldx, M, y, ldy, add ? 1 : 0, nullptr, 0, 0};
    return launch_gemv<false, false>(a, p.groups, stream);
}

cudaError_t enqueue_gemv_fp4(const Fp4Weights& w, const __nv_bfloat16* x, int ldx, int M, float* y, int ldy, bool add,
                             const GemvPlan& p, cudaStream_t stream) {
    if(!gemv_ok(p, true, w.N, w.K, w.tiles, x, ldx, M) || !w.scale_tiles || !aligned(w.scale_tiles, 8) || !y || ldy < w.N ||
       !(w.weight_global_scale > 0.f))
        return cudaErrorInvalidValue;
    const GemvArgs a{w.tiles, w.scale_tiles, nullptr, w.weight_global_scale, w.weight_global_scale, w.N, w.K, x, ldx, M, y, ldy,
                     add ? 1 : 0, nullptr, 0, 0};
    return launch_gemv<true, false>(a, p.groups, stream);
}

cudaError_t enqueue_gemv_fp8_gated(const Fp8Weights& w, const __nv_bfloat16* x, int ldx, int M, Activation act,
                                   __nv_bfloat16* out, int ldo, const GemvPlan& p, cudaStream_t stream) {
    if(!gemv_ok(p, false, w.N, w.K, w.tiles, x, ldx, M) || w.N % 32 || !w.row_scale || !out || ldo < w.N / 2 || !activation_ok(act))
        return cudaErrorInvalidValue;
    const GemvArgs a{w.tiles, nullptr, w.row_scale, 1.f, 1.f, w.N, w.K, x, ldx, M, nullptr, 0, 0, out, ldo, static_cast<int>(act)};
    return launch_gemv<false, true>(a, p.groups, stream);
}

cudaError_t enqueue_gemv_fp4_gated(const Fp4Weights& w, float up_weight_global_scale, const __nv_bfloat16* x, int ldx,
                                   int M, Activation act, __nv_bfloat16* out, int ldo, const GemvPlan& p,
                                   cudaStream_t stream) {
    if(!gemv_ok(p, true, w.N, w.K, w.tiles, x, ldx, M) || w.N % 32 || !w.scale_tiles || !aligned(w.scale_tiles, 8) || !out ||
       ldo < w.N / 2 || !(w.weight_global_scale > 0.f) || !(up_weight_global_scale > 0.f) || !activation_ok(act))
        return cudaErrorInvalidValue;
    const GemvArgs a{w.tiles, w.scale_tiles, nullptr, w.weight_global_scale, up_weight_global_scale, w.N, w.K, x, ldx, M,
                     nullptr, 0, 0, out, ldo, static_cast<int>(act)};
    return launch_gemv<true, true>(a, p.groups, stream);
}

A16Plan plan_a16(bool fp4, int N, int K, int T, int min_blocks) {
    A16Plan p;
    p.fp4 = fp4;
    p.n = N;
    p.k = K;
    p.tokens = T;
    if(N <= 0 || K <= 0 || T <= 0) return p;
    const long blocks = static_cast<long>((T + 8 * kA16NT - 1) / (8 * kA16NT)) * ((N + kA16Rows - 1) / kA16Rows);
    if(min_blocks > 0) {
        while(p.sk < 8 && blocks * p.sk < min_blocks) p.sk *= 2;
    } else {
        int device = 0, sms = 0, per_sm = 0;
        const void* k = fp4 ? reinterpret_cast<const void*>(a16_kernel<true, false>) : reinterpret_cast<const void*>(a16_kernel<false, false>);
        if(cudaGetDevice(&device) == cudaSuccess && cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
           cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, kA16Warps * 32, 0) == cudaSuccess) {
            const long slots = static_cast<long>(sms) * per_sm;
            while(p.sk < 8 && blocks * p.sk * 2 <= slots) p.sk *= 2;
        }
    }
    p.partial_floats = p.sk > 1 ? static_cast<std::size_t>(p.sk) * T * N : 0;
    return p;
}

cudaError_t enqueue_fp8_a16(const Fp8Weights& w, const __nv_bfloat16* x, int T, float* y, int ldo, bool add,
                            const A16Plan& p, float* partials, cudaStream_t stream) {
    if(!a16_ok(p, false, w.N, w.K, T, w.tiles, x, partials) || !w.row_scale || !y || ldo < w.N) return cudaErrorInvalidValue;
    const A16Args a{w.tiles, nullptr, w.row_scale, 1.f, 1.f, w.N, w.K, x, T, y, ldo, add ? 1 : 0, partials, nullptr, 0, 0};
    return launch_a16<false, false>(a, p.sk, stream);
}

cudaError_t enqueue_fp4_a16(const Fp4Weights& w, const __nv_bfloat16* x, int T, float* y, int ldo, bool add,
                            const A16Plan& p, float* partials, cudaStream_t stream) {
    if(!a16_ok(p, true, w.N, w.K, T, w.tiles, x, partials) || !w.scale_tiles || !aligned(w.scale_tiles, 8) || !y || ldo < w.N ||
       !(w.weight_global_scale > 0.f))
        return cudaErrorInvalidValue;
    const A16Args a{w.tiles, w.scale_tiles, nullptr, w.weight_global_scale, w.weight_global_scale, w.N, w.K, x, T, y, ldo, add ? 1 : 0,
                    partials, nullptr, 0, 0};
    return launch_a16<true, false>(a, p.sk, stream);
}

cudaError_t enqueue_fp8_a16_gated(const Fp8Weights& w, const __nv_bfloat16* x, int T, Activation act,
                                  __nv_bfloat16* out, int ldo, const A16Plan& p, float* partials, cudaStream_t stream) {
    if(!a16_ok(p, false, w.N, w.K, T, w.tiles, x, partials) || w.N % 32 || !w.row_scale || !out || ldo < w.N / 2 || !activation_ok(act))
        return cudaErrorInvalidValue;
    const A16Args a{w.tiles, nullptr, w.row_scale, 1.f, 1.f, w.N, w.K, x, T, nullptr, 0, 0, partials, out, ldo, static_cast<int>(act)};
    return launch_a16<false, true>(a, p.sk, stream);
}

cudaError_t enqueue_fp4_a16_gated(const Fp4Weights& w, float up_weight_global_scale, const __nv_bfloat16* x, int T,
                                  Activation act, __nv_bfloat16* out, int ldo, const A16Plan& p, float* partials,
                                  cudaStream_t stream) {
    if(!a16_ok(p, true, w.N, w.K, T, w.tiles, x, partials) || w.N % 32 || !w.scale_tiles || !aligned(w.scale_tiles, 8) || !out ||
       ldo < w.N / 2 || !(w.weight_global_scale > 0.f) || !(up_weight_global_scale > 0.f) || !activation_ok(act))
        return cudaErrorInvalidValue;
    const A16Args a{w.tiles, w.scale_tiles, nullptr, w.weight_global_scale, up_weight_global_scale, w.N, w.K, x, T, nullptr, 0, 0,
                    partials, out, ldo, static_cast<int>(act)};
    return launch_a16<true, true>(a, p.sk, stream);
}

}  // namespace crit::stored_tile
