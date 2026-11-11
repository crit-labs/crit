// fixed-tile-gemm's CUDA GEMM (kernel.json "cuda"; structures cuda-mm and cuda-mm-split): x [M][K] times dense [N][K],
// both BF16 (variant T 0) or both FP16 (T 1), FP32 accumulation, out [M][N] in the 16-bit type or FP32. Compiled by
// crit_kernels.build for the build's GPU as a shared library a variant (CRIT_VARIANT_T), with the build's
// crit_tuning.h, loaded by the package through ctypes (_crit_cuda.library).
//
// Built from the kernel core: tile.cuh's K-major byte tiles (a 16-bit row of BK bytes is BK / 2 values; a k-step of
// 32 bytes is one m16n8k16 MMA, whose A and B fragments are ldsm_a's and ldsm_b2's), the swizzled cp.async ring and the
// persistent grouped CTA order of fp8-block-rows' native GEMMs, mma.cuh's m16n8k16 BF16 / FP16 MMA, and
// gemm_epilogue.cuh's stores (one rounding of each FP32 sum). The mechanisms of CUTLASS's multistage GEMM (BSD-3-Clause;
// written anew, no code copied).
//
// cuda-mm: every output's sum is the MMAs' over K in k16 steps in increasing order into one FP32 accumulator, whatever
// the configuration: the tile, warps, stages, epilogue and the token count choose which outputs a CTA computes and how
// its loads are staged, never an output's order, so a row's bits do not depend on the rows beside it (row invariant;
// tests/test_fixed_tile_gemm.py checks configurations and row counts against each other).
// cuda-mm-split: the same tiles with K split over SPLIT CTAs a tile (FP32 partials, added in split order by a second
// pass and rounded once): more CTAs for narrow matrices at a few rows; the split may follow the row count, so NOT row
// invariant (deterministic on a repeat).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "crit_tuning.h"
#include "crit_kcore/gemm_epilogue.cuh"
#include "crit_kcore/mma.cuh"
#include "crit_kcore/tile.cuh"

#ifndef CRIT_VARIANT_T
#define CRIT_VARIANT_T 0          // 0: BF16 operands, 1: FP16
#endif

namespace {

namespace kc = crit::kcore;
namespace kt = crit::kcore::tile;
namespace go = crit::kcore::gemm_out;

using Half = std::conditional_t<CRIT_VARIANT_T == 0, __nv_bfloat16, __half>;

struct Params {
    const void* x;            // [M][ldx] 16-bit
    const void* w;            // [N][ldw] 16-bit
    void* out;                // [M][ldo]: Half or FP32 (out_f32)
    float* part;              // split partials [splits][M][N] FP32 (cuda-mm-split with splits > 1)
    long long ldx, ldw, ldo;  // elements
    int M, N, K, splits, kchunk, out_f32;
};

template <int TM, int TN, int WM, int WN, int BK, int STAGES>
struct Cfg {
    static constexpr int kThreads = 32 * WM * WN, kWTM = TM / WM, kWTN = TN / WN;
    static constexpr int kMT = kWTM / 16, kNT = kWTN / 8, kKB = BK / 32;
    static_assert(kWTM % 16 == 0 && kWTN % 16 == 0 && kMT >= 1, "warp tiles of whole 16-row, 16-column tiles");
    static_assert(BK == 64 || BK == 128, "64 or 128 bytes (32 or 64 values) of K a stage");
    static_assert(STAGES >= 2, "a ring of two stages or more");
    static constexpr int kA = 0, kB = TM * BK, kStage = (TM + TN) * BK, kSmem = STAGES * kStage;
};

__device__ __forceinline__ void mma16(float* c, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    if constexpr(CRIT_VARIANT_T == 0) kc::mma_bf16_m16n8k16(c, a[0], a[1], a[2], a[3], b0, b1);
    else kc::mma_f16_m16n8k16(c, a[0], a[1], a[2], a[3], b0, b1);
}

// A CTA's work items, persistent: item blockIdx.x, + gridDim.x, ...: tile item / splits (tile_of's grouped order) and
// split item % splits, its K range [split kchunk, min(K, (split + 1) kchunk)). Each item's first STAGES - 1 stages are
// issued before the previous item's epilogue (EPI 0) or right after it (EPI 1, through shared memory).
template <class C, class OUT, int TM, int TN, int WM, int BK, int STAGES, int GROUP, int EPI>
__global__ void __launch_bounds__(C::kThreads, 1) gemm_kernel(Params p) {
    constexpr int T = C::kThreads, MT = C::kMT, NT = C::kNT, KB = C::kKB, BV = BK / 2;
    extern __shared__ __align__(128) unsigned char smem[];
    const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32;
    const int wm0 = (warp % WM) * C::kWTM, wn0 = (warp / WM) * C::kWTN;
    const int tiles_m = (p.M + TM - 1) / TM, tiles_n = (p.N + TN - 1) / TN, total = tiles_m * tiles_n * p.splits;
    const unsigned char* xb = static_cast<const unsigned char*>(p.x);
    const unsigned char* wb = static_cast<const unsigned char*>(p.w);

    struct Item {
        int m0, n0, rows, cols, k0, k1, KT;
        kt::KTileCopy<TM, BK, T> a;
        kt::KTileCopy<TN, BK, T> b;
    };
    auto set = [&](Item& s, int linear) {
        int tm, tn;
        kt::tile_of(linear / p.splits, tiles_m, tiles_n, GROUP, tm, tn);
        const int split = linear % p.splits;
        s.m0 = tm * TM;
        s.n0 = tn * TN;
        s.rows = min(TM, p.M - s.m0);
        s.cols = min(TN, p.N - s.n0);
        s.k0 = split * p.kchunk;
        s.k1 = min(p.K, s.k0 + p.kchunk);
        s.KT = s.k1 > s.k0 ? (s.k1 - s.k0 + BV - 1) / BV : 0;
        s.a.init(xb + (std::size_t(s.m0) * std::size_t(p.ldx) + std::size_t(s.k0)) * 2, std::size_t(p.ldx) * 2,
                 s.rows, tid);
        s.b.init(wb + (std::size_t(s.n0) * std::size_t(p.ldw) + std::size_t(s.k0)) * 2, std::size_t(p.ldw) * 2,
                 s.cols, tid);
    };
    auto issue = [&](const Item& s, int k_tile) {
        unsigned char* stage = smem + (k_tile % STAGES) * C::kStage;
        // bytes of the item's K range: the split's end zero-fills the rest of the last stage
        s.a.issue(stage + C::kA, k_tile * BK, (s.k1 - s.k0) * 2);
        s.b.issue(stage + C::kB, k_tile * BK, (s.k1 - s.k0) * 2);
    };
    auto prologue = [&](const Item& s) {
#pragma unroll
        for(int st = 0; st < STAGES - 1; ++st) {
            if(st < s.KT) issue(s, st);
            kc::cp_async_commit();
        }
    };
    auto fragments = [&](const unsigned char* stage, int ks, uint32_t (&a)[MT][4], uint32_t (&b)[NT][2]) {
#pragma unroll
        for(int i = 0; i < MT; ++i) kt::ldsm_a<BK>(stage + C::kA, wm0 + 16 * i, ks, lane, a[i]);
#pragma unroll
        for(int j = 0; j < NT; j += 2) {
            uint32_t four[4];
            kt::ldsm_b2<BK>(stage + C::kB, wn0 + 8 * j, ks, lane, four);
            b[j][0] = four[0], b[j][1] = four[1], b[j + 1][0] = four[2], b[j + 1][1] = four[3];
        }
    };

    int linear = int(blockIdx.x);
    if(linear >= total) return;
    Item cur;
    set(cur, linear);
    prologue(cur);
    for(;;) {
        float acc[MT][NT][4];
#pragma unroll
        for(int i = 0; i < MT; ++i)
#pragma unroll
            for(int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
        if(cur.KT > 0) {
            // A k-step's fragments read while the previous k-step's MMAs run, the next stage's wait and barrier before
            // a stage's last k-step, a stage's copies issued after its first.
            uint32_t a[2][MT][4], b[2][NT][2];
            kc::cp_async_wait<STAGES - 2>();
            __syncthreads();
            fragments(smem, 0, a[0], b[0]);
            for(int k_tile = 0; k_tile < cur.KT; ++k_tile) {
                const unsigned char* stage = smem + (k_tile % STAGES) * C::kStage;
                const unsigned char* next = smem + ((k_tile + 1) % STAGES) * C::kStage;
#pragma unroll
                for(int ks = 0; ks < KB; ++ks) {
                    if(ks == KB - 1) {
                        kc::cp_async_wait<STAGES - 2>();
                        __syncthreads();
                        if(k_tile + 1 < cur.KT) fragments(next, 0, a[(ks + 1) & 1], b[(ks + 1) & 1]);
                    } else {
                        fragments(stage, ks + 1, a[(ks + 1) & 1], b[(ks + 1) & 1]);
                    }
                    if(ks == 0) {
                        if(k_tile + STAGES - 1 < cur.KT) issue(cur, k_tile + STAGES - 1);
                        kc::cp_async_commit();
                    }
#pragma unroll
                    for(int i = 0; i < MT; ++i)
#pragma unroll
                        for(int j = 0; j < NT; ++j) mma16(acc[i][j], a[ks & 1][i], b[ks & 1][j][0], b[ks & 1][j][1]);
                }
            }
        }
        float rs[MT][2], cs[NT][2];
#pragma unroll
        for(int i = 0; i < MT; ++i) rs[i][0] = rs[i][1] = 1.0f;
#pragma unroll
        for(int j = 0; j < NT; ++j) cs[j][0] = cs[j][1] = 1.0f;
        const int next = linear + int(gridDim.x);
        const bool more = next < total;
        Item nx;
        // where the item's outputs go: the output, or its split's partials
        const bool partial = p.splits > 1;
        OUT* tile_out = nullptr;
        float* part_out = nullptr;
        std::size_t ld;
        if(partial) {
            part_out = p.part + std::size_t(linear % p.splits) * std::size_t(p.M) * std::size_t(p.N) +
                       std::size_t(cur.m0) * std::size_t(p.N) + std::size_t(cur.n0);
            ld = std::size_t(p.N);
        } else {
            tile_out = static_cast<OUT*>(p.out) + std::size_t(cur.m0) * std::size_t(p.ldo) + std::size_t(cur.n0);
            ld = std::size_t(p.ldo);
        }
        if constexpr(EPI == 0) {
            if(more) {
                __syncthreads();   // every warp's last reads of the ring are done
                set(nx, next);
                prologue(nx);
            }
            if(partial) go::store_scaled<float>(acc, rs, cs, wm0, wn0, lane, part_out, ld, cur.rows, cur.cols);
            else go::store_scaled<OUT>(acc, rs, cs, wm0, wn0, lane, tile_out, ld, cur.rows, cur.cols);
        } else {
            kc::cp_async_wait<0>();
            if(partial) {
                constexpr int kChunk = go::staged_chunk<float>(TM, TN, C::kSmem);
                go::store_scaled_staged<float, TM, TN, MT, NT, kChunk, T>(acc, rs, cs, wm0, wn0, lane,
                                                                         reinterpret_cast<float*>(smem), part_out, ld,
                                                                         cur.rows, cur.cols, tid);
            } else {
                constexpr int kChunk = go::staged_chunk<OUT>(TM, TN, C::kSmem);
                go::store_scaled_staged<OUT, TM, TN, MT, NT, kChunk, T>(acc, rs, cs, wm0, wn0, lane,
                                                                       reinterpret_cast<OUT*>(smem), tile_out, ld,
                                                                       cur.rows, cur.cols, tid);
            }
            if(more) {
                __syncthreads();   // every thread's reads of the staged rows are done
                set(nx, next);
                prologue(nx);
            }
        }
        if(!more) break;
        cur = nx;
        linear = next;
    }
    kc::cp_async_wait<0>();
}

// The split partials added in split order, rounded once: out[r][c] = sum over s of part[s][r][c].
template <class OUT>
__global__ void split_sum_kernel(Params p) {
    const std::size_t total = std::size_t(p.M) * std::size_t(p.N), plane = total;
    for(std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
        i += std::size_t(gridDim.x) * blockDim.x) {
        float v = p.part[i];
        for(int s = 1; s < p.splits; ++s) v = __fadd_rn(v, p.part[std::size_t(s) * plane + i]);
        const std::size_t r = i / std::size_t(p.N), c = i % std::size_t(p.N);
        static_cast<OUT*>(p.out)[r * std::size_t(p.ldo) + c] = go::round_to<OUT>(v);
    }
}

int g_sms = 0;
int sms() {
    if(!g_sms) {
        int device = 0;
        (void)cudaGetDevice(&device);
        (void)cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, device);
        if(g_sms <= 0) g_sms = 1;
    }
    return g_sms;
}

template <int TM, int TN, int WM, int WN, int BK, int STAGES, int GROUP, int EPI, class OUT>
int launch(Params p, cudaStream_t stream) {
    using C = Cfg<TM, TN, WM, WN, BK, STAGES>;
    auto kernel = gemm_kernel<C, OUT, TM, TN, WM, BK, STAGES, GROUP, EPI>;
    static const cudaError_t attribute =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem);
    if(attribute != cudaSuccess) {
        (void)cudaGetLastError();
        return 3;
    }
    static const int resident = [kernel] {
        int per = 0;
        (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kernel, C::kThreads, C::kSmem);
        (void)cudaGetLastError();
        return sms() * (per > 0 ? per : 1);
    }();
    const long long items = (long long)((p.M + TM - 1) / TM) * ((p.N + TN - 1) / TN) * p.splits;
    if(items <= 0) return 0;
    if(items > 0x7fffffffLL) return 4;
    kernel<<<unsigned(items < resident ? items : resident), C::kThreads, C::kSmem, stream>>>(p);
    if(cudaGetLastError() != cudaSuccess) return 5;
    if(p.splits > 1) {
        const long long n = (long long)p.M * p.N;
        const long long blocks = (n + 255) / 256;
        split_sum_kernel<OUT><<<unsigned(blocks < 8 * sms() ? blocks : 8 * sms()), 256, 0, stream>>>(p);
        if(cudaGetLastError() != cudaSuccess) return 5;
    }
    return 0;
}

template <int TM, int TN, int WM, int WN, int BK, int STAGES, int GROUP, int EPI>
int launch_any(Params p, cudaStream_t stream) {
    return p.out_f32 ? launch<TM, TN, WM, WN, BK, STAGES, GROUP, EPI, float>(p, stream)
                     : launch<TM, TN, WM, WN, BK, STAGES, GROUP, EPI, Half>(p, stream);
}

constexpr int kSplitsMax = 4, kSplitKMin = 1024;   // the untuned split rule's bounds (SPLIT 0)

}  // namespace

// The configuration of a call (structure 0: cuda-mm, 1: cuda-mm-split): out = {tm, tn, splits, kchunk}. The split
// (cuda-mm-split): SPLIT, or with SPLIT 0 as many (up to 4, each at least 1,024 values of K) as bring the tiles to the
// device's multiprocessors; kchunk a multiple of the stage's values. Returns 0, or 1 where this build holds no such
// configuration.
extern "C" int crit_ftg_plan(int structure, long long N, long long K, int M, int* out) {
    int tm = 0, tn = 0, bk = 0, split = 1;
    if(structure == 0) {
        switch(crit_lookup_cuda_mm(N, K, M)) {
#define FTG_PLAN(index, TM_, TN_, WM_, WN_, BK_, STAGES_, GROUP_, EPI_) \
            case index: tm = TM_, tn = TN_, bk = BK_; break;
            CRIT_CONFIGS_cuda_mm(FTG_PLAN)
#undef FTG_PLAN
            default: return 1;
        }
    } else {
        switch(crit_lookup_cuda_mm_split(N, K, M)) {
#define FTG_PLAN_SPLIT(index, TM_, TN_, WM_, WN_, BK_, STAGES_, GROUP_, EPI_, SPLIT_) \
            case index: tm = TM_, tn = TN_, bk = BK_, split = SPLIT_; break;
            CRIT_CONFIGS_cuda_mm_split(FTG_PLAN_SPLIT)
#undef FTG_PLAN_SPLIT
            default: return 1;
        }
        if(split == 0) {
            const long long tiles = ((M + tm - 1) / tm) * ((N + tn - 1) / tn);
            long long s = tiles > 0 ? (sms() + tiles / 2) / tiles : 1;
            s = s < kSplitsMax ? s : kSplitsMax;
            s = s < K / kSplitKMin ? s : K / kSplitKMin;
            split = int(s < 1 ? 1 : s);
        }
    }
    const int bv = bk / 2;
    long long chunk = (K + split - 1) / split;
    chunk = (chunk + bv - 1) / bv * bv;
    split = int((K + chunk - 1) / chunk);   // no empty split
    out[0] = tm, out[1] = tn, out[2] = split < 1 ? 1 : split, out[3] = int(chunk);
    return 0;
}

// The GEMM on `stream`: x [M][ldx], w [N][ldw] (16-bit, rows 16-byte aligned, K a multiple of 8), out [M][ldo] (16-bit
// or out_f32), part a workspace of splits x M x N FP32 where splits > 1 (crit_ftg_plan's). Returns 0, 1 (no such
// configuration in this build), 2 (operands not aligned), 3 (more shared memory than the GPU gives), 4, 5 (launch).
extern "C" int crit_ftg_mm(int structure, const void* x, long long ldx, const void* w, long long ldw, void* out,
                           long long ldo, int out_f32, int M, int N, int K, float* part, int splits, int kchunk,
                           void* stream) {
    if(K % 8 || ldx % 8 || ldw % 8 || reinterpret_cast<std::uintptr_t>(x) % 16 ||
       reinterpret_cast<std::uintptr_t>(w) % 16)
        return 2;
    Params p{x, w, out, part, ldx, ldw, ldo, M, N, K, splits < 1 ? 1 : splits, kchunk > 0 ? kchunk : K, out_f32};
    if(p.splits > 1 && !part) return 2;
    if(p.splits == 1) p.kchunk = K;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    if(structure == 0) {
        switch(crit_lookup_cuda_mm(N, K, M)) {
#define FTG_MM(index, TM_, TN_, WM_, WN_, BK_, STAGES_, GROUP_, EPI_) \
            case index: return launch_any<TM_, TN_, WM_, WN_, BK_, STAGES_, GROUP_, EPI_>(p, s);
            CRIT_CONFIGS_cuda_mm(FTG_MM)
#undef FTG_MM
            default: return 1;
        }
    }
    switch(crit_lookup_cuda_mm_split(N, K, M)) {
#define FTG_SPLIT(index, TM_, TN_, WM_, WN_, BK_, STAGES_, GROUP_, EPI_, SPLIT_) \
        case index: return launch_any<TM_, TN_, WM_, WN_, BK_, STAGES_, GROUP_, EPI_>(p, s);
        CRIT_CONFIGS_cuda_mm_split(FTG_SPLIT)
#undef FTG_SPLIT
        default: return 1;
    }
}
