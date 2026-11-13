// paged-kv-attention's CUDA flash tile (structures cuda-window, cuda-fused, cuda-prefill; the kernel's README): attention
// over paged KV for decode tokens, verify windows, prompt pieces and prompt chunks. kernel.json's "cuda" section names
// this source: crit_kernels.build compiles it for the build's GPU, one library a variant (the CRIT_VARIANT_* defines:
// query dtype, page kind, head widths, and the caller's warps, stages or value split where it overrides the tuning),
// with the build's crit_tuning.h, on the package's first call of that variant (_crit_cuda.library).
//
// A program (CTA) reads one tile of a row's query rows (the row's columns times a KV head's query heads: tile row j is
// column j / G, head j % G) over one KV head's positions. The queries stay in shared memory (swizzled, loaded once by
// cp.async) in row groups of 16 rows, VW warps a group (Tile): their fragments come from shared memory a k-step at a
// time (ldmatrix), the keys and values of a block of BN positions are staged through a STAGES-deep cp.async ring the
// CTA's warps share (8-bit pages: their codes, converted in registers (IREG) or once a block into a 16-bit buffer),
// the scores and the output are FP32 mma.sync m16n8k16 accumulators, and each row group runs its rows' online softmax
// in registers (the kernel core's attention_softmax: no state crosses row groups). So a row's bits depend on its own
// query, its keys and values, the page kind, BN and the split (CHUNK): never on the rows beside it, the warps or the
// tile.
//
// A row's positions: blocks of BN from 0 up to the tile's last column; splits of CHUNK positions (a multiple of BN; with
// a ramp, smaller ones first: split_of), one program a split. Split launches write a split's state (its output rounded
// to BF16, maximum, sum) for crit_cw_merge (cw_merge2 for window launches), which folds the splits holding a column's
// positions in split order. Fused launches (no partials a split: a workspace that follows the tiles, not the row's
// length) are the same programs: each writes its state into a ring of RING x GROUP slots of its tile, the next
// group's programs fold a group's states in split order into the tile's running state (each its share of the rows),
// the last group's last program folds that group into the outputs: the merge's steps and rounding, the same bits as
// the split launch and its merge. Their programs take their work from a ticket in split order (every tile's first
// split, the longest tiles first, then the next), so a program only ever waits for programs that hold an earlier
// ticket and are running. ONE_SPLIT launches (a prompt chunk in one range) write the output directly. Every block is
// masked the same way (causal: a column sees positions up to first + c; trees: those before the window and its
// ancestors; a block every live row of a row group sees whole needs no mask), positions past the tile's last column
// are not read (zero filled), so a block a row does not see changes none of its bits.
//
// Mechanisms after FlashInfer's batch prefill over paged KV (Apache-2.0): queries in shared memory with 16 rows a warp
// and fragments loaded per k-step, grouped heads packed into the tile's rows, a cp.async multistage ring of paged KV
// rows (each position's slot from the page table), 128-byte swizzled rows read by ldmatrix (.trans for the values),
// the probabilities' accumulators reused as the next product's A operand, the longest tiles first.
#include "crit_tuning.h"

#include "crit_kcore/arch.cuh"
#include "crit_kcore/attention_softmax.cuh"
#include "crit_kcore/formats.cuh"
#include "crit_kcore/load.cuh"
#include "crit_kcore/mma.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace kc = crit::kcore;

// Everything but the extern "C" entry points has internal linkage: every variant's library holds the same template
// instantiations under the same names, and a process loads several (one would otherwise bind another's tables).
namespace {

// -- the variant (kernel.json "cuda" "variant"): fixed for a library
#ifndef CRIT_VARIANT_T
#define CRIT_VARIANT_T 0          // 0: BF16 queries (and 16-bit pages), 1: FP16
#endif
#ifndef CRIT_VARIANT_KIND
#define CRIT_VARIANT_KIND 0       // 0 16-bit pages, 1 fp8 (a multiplier per layer), 2 fp8:row, 3 int8
#endif
#ifndef CRIT_VARIANT_D
#define CRIT_VARIANT_D 128
#endif
#ifndef CRIT_VARIANT_DV
#define CRIT_VARIANT_DV CRIT_VARIANT_D
#endif
#ifndef CRIT_VARIANT_W
#define CRIT_VARIANT_W 0          // a caller's warps for every launch (0: the tuning's)
#endif
#ifndef CRIT_VARIANT_S
#define CRIT_VARIANT_S 0          // a caller's stages (0: the tuning's)
#endif
#ifndef CRIT_VARIANT_VS
#define CRIT_VARIANT_VS 0         // a caller's value split of fused launches (0: the tuning's)
#endif
#ifndef CRIT_VARIANT_SMEM
#define CRIT_VARIANT_SMEM 101376  // the GPU's shared memory a block (opt-in): fewer stages, then warps, past it
#endif

constexpr int TQ = CRIT_VARIANT_T, KIND = CRIT_VARIANT_KIND, D = CRIT_VARIANT_D, DV = CRIT_VARIANT_DV;
constexpr bool BYTE_PAGES = KIND != 0;
static_assert(D % 64 == 0 && DV % 64 == 0 && D <= 256 && DV <= 256, "head widths of 64, 128 or 256");

enum : int {
    F_COUNTS = 1, F_OFFSETS = 2, F_TREE = 4, F_LSE = 8, F_FRESH = 16, F_ONE_ROW = 32, F_POS_T = 64, F_FIRST_T = 128,
    F_DEV_CHUNK = 256, F_ONE_SPLIT = 512, F_PAR64 = 1024, F_SINKS = 2048, F_STATE_OUT = 4096, F_STATE_IN = 8192
};

}  // namespace

// The launch's arguments (mirrored by the package's ctypes structure: pointers and 64-bit integers first): external
// linkage, the entry points take it.
struct Params {
    const void* q;
    const void* kp;
    const void* vp;
    const float* ks;
    const float* vs;
    const int* tables;
    const int* rows;
    const int* starts;
    const int* counts;
    const long long* firsts;
    const void* parents;
    const int* offsets;
    float* po;              // split launches: the partials' outputs (BF16 pairs); fused: the ring's states
    float* pm;
    float* pl;
    void* out;
    float* lse;
    const void* fk;
    const void* fv;
    const long long* fshift;
    const long long* ffirst;
    float* run;             // fused: each tile's running state (FP32 outputs, then maxima and sums)
    int* sync;              // fused: the ticket and done counters, then a tile's flag and two group counters
    const float* sinks;     // F_SINKS: each query head's sink logit (FP32 [hq], natural log), folded in last
    float* state;           // F_STATE_OUT / F_STATE_IN: every thread's running state between a chunk's two passes
    long long stride_qt, stride_qh, stride_ot, stride_oh, stride_fk, stride_fv;
    long long first0, ffirst0;
    int stride_fr, tab_stride, tab_width, hq, hkv, g, page, n_parts, chunk, cols, flags, page_shift;
    int block, group, tiles, merge, ring, ramp, ramp_k;
    int window;             // a sliding window: positions a token sees, its own included (0: all)
    int state_part;         // F_STATE_*: the pass's positions (0: below the cut, 1: the cut's block, 2: past it)
    float qk_scale, k_mult, v_mult, pad2;
};

namespace {

using T = std::conditional_t<TQ == 1, __half, __nv_bfloat16>;

__device__ __forceinline__ void mma16(float* c, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    if constexpr(TQ == 1) kc::mma_f16_m16n8k16(c, a[0], a[1], a[2], a[3], b0, b1);
    else kc::mma_bf16_m16n8k16(c, a[0], a[1], a[2], a[3], b0, b1);
}

__device__ __forceinline__ uint32_t pack_out(float x, float y) {
    if constexpr(TQ == 1) return kc::u32_of(__floats2half2_rn(x, y));
    else return kc::pack_bf16x2(x, y);
}

// Element offset of 16-byte chunk ch of row `row` in a tile of W 16-bit values a row, chunks XOR-swizzled by the row
// (W >= 64: 8 chunks or more, so 8 rows read in one column of chunks hit 8 bank groups).
template <int W>
__device__ __forceinline__ int sw(int row, int ch) { return row * W + ((ch ^ (row & 7)) << 3); }

// Byte offset of 16-byte chunk ch of row `row` in a tile of W bytes a row (8-bit codes, IREG), chunks XOR-swizzled so
// that 8 consecutive rows read in one column of chunks hit 8 bank groups (W = 64: two rows a 128-byte line).
template <int W>
__device__ __forceinline__ int sw8(int row, int ch) {
    return row * W + ((ch ^ (W >= 128 ? (row & 7) : ((row >> 1) & 3))) << 4);
}

__device__ __forceinline__ long long cdivll(long long a, long long b) { return (a + b - 1) / b; }

// A row's splits (an order knob: a column's bits follow them, fixed per cache, the same for every row). ramp 0: CHUNK
// positions each, split s from s x CHUNK. ramp R > 0 (RAMP): the row's first positions in R bands of K splits
// (RAMP_SPLITS) each, band b's splits of CHUNK / 2^(R - b) positions, then splits of CHUNK: a short row in many small
// splits, a long one in about as many as with ramp 0. Every split a whole number of blocks (CHUNK a multiple of BN x
// 2^R). split_of: split s's first position and size; splits_to: the splits holding positions 0 .. n - 1 (n >= 1).
struct Split {
    int lo, size;
};
__device__ __forceinline__ Split split_of(int s, int chunk, int ramp, int rk) {
    if(ramp == 0) return Split{s * chunk, chunk};
    const int s0 = chunk >> ramp;
    if(s < rk * ramp) {
        const int b = s / rk, i = s % rk;
        return Split{rk * s0 * ((1 << b) - 1) + i * (s0 << b), s0 << b};
    }
    return Split{rk * s0 * ((1 << ramp) - 1) + (s - rk * ramp) * chunk, chunk};
}
__device__ __forceinline__ long long splits_to(long long n, long long chunk, int ramp, int rk) {
    if(ramp == 0) return cdivll(n, chunk);
    const long long s0 = chunk >> ramp, band = rk * s0, edge = band * ((1 << ramp) - 1);
    if(n > edge) return (long long)rk * ramp + cdivll(n - edge, chunk);
    const long long p = n - 1;                            // the last position, in band b: [band (2^b - 1), band (2^(b+1) - 1))
    const int b = 63 - __clzll((p + band) / band);        // floor(log2((p + band) / band))
    return (long long)rk * b + (p - band * ((1LL << b) - 1)) / (s0 << b) + 1;
}

// A fused tile's flag: read with acquire and written with release at GPU scope (the ring's states and the running
// state a program publishes are visible to the program that sees its flag).
__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
    asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// A split partial's layout: a row's DV / 2 BF16 pairs lane-major, pair e (values 2 e, 2 e + 1: n8 tile e / 4, lane
// e % 4 of its quad) at word (e % 4) x DV / 8 + e / 4, so each lane stores its pairs 16 bytes at a time.
__device__ __forceinline__ int partial_word(int e) { return (e & 3) * (DV / 8) + (e >> 2); }

// 8 codes (an 8-byte word pair) of an 8-bit page as 8 16-bit values (exact).
__device__ __forceinline__ uint4 codes_to_16(uint2 w) {
    uint32_t o[4];
    if constexpr(KIND == 3) {
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const uint32_t word = i < 2 ? w.x : w.y;
            const int sh = (i & 1) * 16;
            const float lo = float(int(int8_t((word >> sh) & 0xff)));
            const float hi = float(int(int8_t((word >> (sh + 8)) & 0xff)));
            if constexpr(TQ == 1) o[i] = kc::u32_of(__floats2half2_rn(lo, hi));
            else o[i] = kc::pack_bf16x2(lo, hi);
        }
    } else {
        if constexpr(TQ == 1) {
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const uint32_t word = i < 2 ? w.x : w.y;
                const uint32_t two = (word >> ((i & 1) * 16)) & 0xffffu;
#if CRIT_KCORE_HAS_FP8_MMA
                o[i] = kc::u32_of(kc::e4m3x2_to_half2(two));
#else
                o[i] = kc::u32_of(__floats2half2_rn(kc::e4m3_value(two & 0xff), kc::e4m3_value(two >> 8)));
#endif
            }
        } else {
            kc::e4m3x4_to_bf16x2(w.x, o);
            kc::e4m3x4_to_bf16x2(w.y, o + 2);
        }
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}

// Shared memory of a tile configuration: the queries, the stages' codes (and raw scales), 8-bit pages' 16-bit block
// (where their codes are converted in shared memory, not in registers: IREG), and with VW warps a row group the
// block's scores they share (each warp computes some of the block's positions).
__host__ __device__ constexpr int smem_bytes(int dvp, int bn, int warps, int stages, int vw = 1, bool ireg = false) {
    const int cb = BYTE_PAGES ? 1 : 2;
    const int stage = bn * D * cb + bn * dvp * cb + (BYTE_PAGES ? bn * 8 : 0);
    const int conv = BYTE_PAGES && !ireg ? bn * (D + dvp) * 2 + bn * 8 : 0;
    const int scores = vw > 1 ? warps * (bn / 8) * 32 * 16 : 0;
    return 16 * warps * D * 2 + stages * stage + conv + scores;
}

// Tile memory of the fused launch's sync words: ticket, done, then three a tile.
constexpr int SYNC_HEAD = 2;

// WARPS row groups of 16 query rows, VW warps a row group: the group's warps share its rows' scores (each computing
// some of a block's positions, n8 tiles of the scores, which they exchange through shared memory) and split its value
// columns (each the outputs of NV / VW n8 tiles). Every warp of a group runs the group's online softmax on the same
// scores (the same bits), and an output's accumulator is its own chain of mma.sync steps whichever warp holds it: the
// row group's warps change no bit.
//
// IREG (8-bit pages): the codes stay in their stage (16-byte chunks of a row XOR-swizzled, sw8) and every warp
// converts the ones its fragments take in registers (the keys' pairs read from their rows, the values' by ldmatrix
// .trans of 16-bit pairs and a quad shuffle), with codes_to_16 as the shared block's conversion: the same operands, the
// same bits, no 16-bit block and no pass converting it.
template <int DVP, int BN, int WARPS, int VW, int STAGES, bool IREG, bool FUSED>
struct Tile {
    static_assert(!IREG || BYTE_PAGES, "in-register conversion takes 8-bit pages");
    static_assert(DVP % 64 == 0, "value columns a program: a multiple of 64");
    static_assert(BN % 16 == 0 && BN <= 64, "blocks of 16, 32 or 64 positions");
    static_assert(STAGES >= 1 && STAGES <= 6, "one to six stages");
    static constexpr int ROWS = 16 * WARPS, THREADS = 32 * WARPS * VW, NT = BN / 8, NV = DVP / 8, KSTEPS = D / 16;
    static constexpr int NVW = NV / VW;                              // value n8 tiles a warp
    static_assert(NV % VW == 0 && NVW % 4 == 0, "a row group's warps hold 4 value n8 tiles or more each");
    static constexpr int CB = BYTE_PAGES ? 1 : 2;                    // bytes a staged code
    static constexpr int Q_BYTES = ROWS * D * 2;
    static constexpr int STAGE_K = BN * D * CB, STAGE_V = BN * DVP * CB;
    static constexpr int STAGE_S = BYTE_PAGES ? BN * 2 * 4 : 0;      // a block's raw scales (fp8:row, int8)
    static constexpr int STAGE_BYTES = STAGE_K + STAGE_V + STAGE_S;
    static constexpr int CONV_BYTES = BYTE_PAGES && !IREG ? BN * (D + DVP) * 2 + BN * 2 * 4 : 0;
    static constexpr int SC_BYTES = VW > 1 ? WARPS * NT * 32 * 16 : 0;   // the groups' block scores (fragment order)
    static constexpr int SMEM = smem_bytes(DVP, BN, WARPS, STAGES, VW, IREG);
    static_assert(SMEM == Q_BYTES + STAGES * STAGE_BYTES + CONV_BYTES + SC_BYTES, "");
    // The warp of a row group that computes scores n8 tile t: pairs of tiles (one ldmatrix x4) where they spread
    // evenly over the group's warps, else single tiles.
    static constexpr bool PAIRS = (NT / 2) % VW == 0;
    __host__ __device__ static constexpr int owner(int t) { return PAIRS ? (t / 2) % VW : t % VW; }
};

// What a fused program owes: its ticket's work, then (when its group is complete) the group's fold.
struct Work {
    int mt, b, rk, split;
};

template <int DVP, int BN, int WARPS, int VW, int STAGES, bool IREG, bool FUSED>
__device__ __forceinline__ void cw_tile(const Params& p, const Work w, unsigned char* smem) {
    using C = Tile<DVP, BN, WARPS, VW, STAGES, IREG, FUSED>;
    constexpr int ROWS = C::ROWS, THREADS = C::THREADS, NT = C::NT, NV = C::NV, KSTEPS = C::KSTEPS;
    constexpr int NVW = C::NVW;
    constexpr int Q_BYTES = C::Q_BYTES, STAGE_K = C::STAGE_K, STAGE_V = C::STAGE_V, STAGE_BYTES = C::STAGE_BYTES;
    __shared__ int s_word;
    T* sq = reinterpret_cast<T*>(smem);
    unsigned char* stages = smem + Q_BYTES;
    T* ck = reinterpret_cast<T*>(stages + STAGES * STAGE_BYTES);   // 8-bit pages: the block's values in 16 bits
    T* cv = ck + BN * D;
    float* cks = reinterpret_cast<float*>(cv + BN * DVP);          // and each position's key and value factors
    float* cvs = cks + BN;
    float4* ssc = reinterpret_cast<float4*>(stages + STAGES * STAGE_BYTES + C::CONV_BYTES);   // VW > 1: scores

    const int tid = threadIdx.x, lane = tid & 31;
    // warp: the row group (16 rows); vw: the warp's part of the group's score tiles and value columns
    const int warp = (tid >> 5) / VW, vw = (tid >> 5) % VW;
    const int tv0 = vw * NVW;   // the warp's first value n8 tile
    const int gid = lane >> 2, tig = lane & 3;
    const int flags = p.flags;
    const bool one_row = flags & F_ONE_ROW, fresh = flags & F_FRESH, tree = flags & F_TREE;
    const bool one_split = flags & F_ONE_SPLIT;
    const int mt = w.mt, b = w.b, rk = w.rk;
    const int r = rk / p.hkv, kvh = rk % p.hkv, G = p.g;

    // Positions fit 31 bits (a row of up to 2^31 positions): 32-bit arithmetic in the loops.
    int first, count;
    if(one_row) {
        count = p.cols;
        first = int((flags & F_POS_T) ? p.firsts[0] : p.first0);
    } else {
        count = (flags & F_COUNTS) ? min(p.counts[r], p.cols) : p.cols;
        first = int(p.firsts[r]);
    }
    const int j0 = mt * ROWS;
    const int c_lo = j0 / G;
    if(c_lo >= count) return;
    const int c_hi = min((j0 + ROWS - 1) / G, count - 1);
    const int n_keys = first + c_hi + 1;
    int chunk = p.chunk;
    if(flags & F_DEV_CHUNK) chunk = int(cdivll(cdivll(first + count, p.n_parts), BN) * BN);
    int lo, hi;
    const int vofs = FUSED ? b * DVP : 0;
    if(one_split) {
        lo = 0, hi = n_keys;
        if(flags & (F_STATE_OUT | F_STATE_IN)) {
            // A chunk in passes over one range, each continuing the state the one before it left (F_STATE_IN) and
            // leaving its own (F_STATE_OUT): the blocks below its first fresh position's block (part 0), that block
            // (part 1), the blocks past it (part 2). The cut is a whole block: the passes run the one range's blocks
            // in its order.
            const long long ff = (flags & F_FIRST_T) ? p.ffirst[0] : p.ffirst0;
            const int cut = int(min(ff, (long long)INT32_MAX - BN) / BN * BN);
            const int part = p.state_part;
            const int a = part == 0 ? 0 : part == 1 ? cut : cut + BN;
            if(part < 2) hi = min(hi, part == 0 ? cut : cut + BN);
            lo = min(a, hi);
        }
    } else {
        const Split sp = split_of(w.split, chunk, p.ramp, p.ramp_k);
        lo = sp.lo;
        if(lo >= n_keys) return;
        if(!FUSED && p.window > 0) {
            // A split wholly below the tile's window (every column's) writes no state: the merge starts after it.
            const int wlo = first + c_lo - p.window + 1;
            if(wlo > 0 && w.split < splits_to(wlo, chunk, p.ramp, p.ramp_k) - 1) return;
        }
        hi = min(lo + sp.size, n_keys);
    }
    const int win = p.window;
    if(win > 0) {
        // The blocks wholly below the tile's lowest seen position are skipped (a block no row sees changes no bit);
        // a split left without positions keeps an empty state, which its merge folds as nothing.
        const int wlo = first + c_lo - win + 1;
        if(wlo > lo) lo = min(hi, lo + (wlo - lo) / BN * BN);
    }
    const long long start = one_row ? 0 : p.starts[r];
    const long long trow = one_row ? 0 : p.rows[r];
    const int* table = p.tables + trow * p.tab_stride;
    int fshift = first, ffirst = first;
    if(fresh) {
        if(one_row) {
            ffirst = int(min((flags & F_FIRST_T) ? p.ffirst[0] : p.ffirst0, (long long)INT32_MAX));
        } else {
            fshift = int(p.fshift[(long long)r * p.stride_fr]);
            ffirst = int(min(p.ffirst[(long long)r * p.stride_fr], (long long)INT32_MAX));
        }
    } else {
        ffirst = INT32_MAX;
    }

    // -- the queries: ROWS x D into shared memory (swizzled), zeros past the row's columns
    {
        constexpr int CH = D / 8;
        for(int i = tid; i < ROWS * CH; i += THREADS) {
            const int row = i / CH, ch = i % CH;
            const int j = j0 + row, c = j / G, g = j % G;
            const bool live = c < count;
            const T* src = reinterpret_cast<const T*>(p.q) +
                           (live ? (start + c) * p.stride_qt + (long long)(kvh * G + g) * p.stride_qh + ch * 8 : 0);
            kc::cp_async16(sq + sw<D>(row, ch), src, live);
        }
        kc::cp_async_commit();
    }

    // -- this lane's rows: h = 0 row gid, h = 1 row gid + 8 of the warp's 16
    int col[2], hd[2];
    bool live[2];
    int lim[2];
    unsigned long long anc[2];
#pragma unroll
    for(int h = 0; h < 2; ++h) {
        const int j = j0 + 16 * warp + gid + 8 * h;
        col[h] = j / G, hd[h] = j % G;
        live[h] = col[h] < count;
        lim[h] = first + col[h];
        anc[h] = 0;
        if(tree && live[h]) {
            const bool p64 = flags & F_PAR64;
            auto parent = [&](long long at) -> long long {
                return p64 ? reinterpret_cast<const long long*>(p.parents)[start + at]
                           : (long long)reinterpret_cast<const int*>(p.parents)[start + at];
            };
            unsigned long long a = 1ull << col[h];
            long long par = parent(col[h]);
            for(int s = 0; s < p.cols && par >= 0; ++s) {
                a |= 1ull << par;
                par = parent(par);
            }
            anc[h] = a;
        }
    }
    // The warp's last position seen (its rows are in column order): blocks past it are skipped (a block a row does
    // not see changes none of its bits).
    const int wj_last = j0 + 16 * warp + 15;
    const bool warp_live = (j0 + 16 * warp) / G < count;
    const int warp_last = first + min(wj_last / G, count - 1);
    // The row group's first column (its rows are in column order): a block every one of its live rows sees in full
    // (ending at or before first + that column, before the window for trees) needs no mask, which changes no bit (its
    // rows not live are never written).
    const int warp_seen = tree ? first - 1 : first + (j0 + 16 * warp) / G;

    // -- the block loader: positions n0 .. n0 + BN - 1 into a stage (codes of the page kind, or 16-bit fresh rows)
    const T* fk = reinterpret_cast<const T*>(p.fk);
    const T* fv = reinterpret_cast<const T*>(p.fv);
    const int page_shift = p.page_shift;   // log2 of the page where it is a power of two, else -1
    auto slot_of = [&](int n) -> long long {
        if(page_shift >= 0) {
            const int pg = __ldg(table + min(n >> page_shift, p.tab_width - 1));
            return ((long long)pg << page_shift) | (n & ((1 << page_shift) - 1));
        }
        const int pg = __ldg(table + min(n / p.page, p.tab_width - 1));
        return (long long)pg * p.page + n % p.page;
    };
    // Threads take positions (TPP threads a position, its row's slot looked up once a block), each its row's chunks.
    constexpr int TPP = THREADS / BN < 1 ? 1 : THREADS / BN;
    auto issue = [&](int n0, int s, int parts = 3) {   // parts: 1 the keys, 2 the values (16-bit pages, IREG)
        unsigned char* st = stages + s * STAGE_BYTES;
        for(int pos = tid / TPP; pos < BN; pos += THREADS / TPP) {
            const int n = n0 + pos, sub = tid % TPP;
            if constexpr(!BYTE_PAGES) {
                T* sk = reinterpret_cast<T*>(st);
                T* sv = reinterpret_cast<T*>(st + STAGE_K);
                const bool ok = n < hi;
                const T* krow = reinterpret_cast<const T*>(p.kp);
                const T* vrow = reinterpret_cast<const T*>(p.vp);
                if(ok) {
                    if(n >= ffirst) {
                        const long long t = n - fshift;
                        krow = fk + t * p.stride_fk + (long long)kvh * D;
                        vrow = fv + t * p.stride_fv + (long long)kvh * DV + vofs;
                    } else {
                        const long long row = slot_of(n) * p.hkv + kvh;
                        krow += row * D;
                        vrow += row * DV + vofs;
                    }
                }
                if(parts & 1) {
#pragma unroll
                    for(int ch = sub; ch < D / 8; ch += TPP)
                        kc::cp_async16(sk + sw<D>(pos, ch), krow + (ok ? ch * 8 : 0), ok);
                }
                if(parts & 2) {
#pragma unroll
                    for(int ch = sub; ch < DVP / 8; ch += TPP)
                        kc::cp_async16(sv + sw<DVP>(pos, ch), vrow + (ok ? ch * 8 : 0), ok);
                }
            } else {
                unsigned char* sk = st;
                unsigned char* sv = st + STAGE_K;
                float* ss = reinterpret_cast<float*>(st + STAGE_K + STAGE_V);
                const bool ok = n < hi && n < ffirst;
                const unsigned char* krow = reinterpret_cast<const unsigned char*>(p.kp);
                const unsigned char* vrow = reinterpret_cast<const unsigned char*>(p.vp);
                long long row = 0;
                if(ok) {
                    row = slot_of(n) * p.hkv + kvh;
                    krow += row * D;
                    vrow += row * DV + vofs;
                }
                if(parts & 1) {
#pragma unroll
                    for(int ch = sub; ch < D / 16; ch += TPP)
                        kc::cp_async16(sk + (IREG ? sw8<D>(pos, ch) : pos * D + ch * 16), krow + (ok ? ch * 16 : 0),
                                       ok);
                }
                if(parts & 2) {
#pragma unroll
                    for(int ch = sub; ch < DVP / 16; ch += TPP)
                        kc::cp_async16(sv + (IREG ? sw8<DVP>(pos, ch) : pos * DVP + ch * 16),
                                       vrow + (ok ? ch * 16 : 0), ok);
                }
                if constexpr(KIND >= 2) {
                    if(sub == 0) {
                        if(parts & 1) kc::cp_async_ca<4>(ss + pos, ok ? p.ks + row : p.ks, ok);
                        if(parts & 2) kc::cp_async_ca<4>(ss + BN + pos, ok ? p.vs + row : p.vs, ok);
                    }
                }
            }
        }
    };
    // 8-bit pages: a landed stage's codes into the 16-bit buffer, each position's key factor (its scale times the
    // softmax scale) and value factor; fresh positions from their 16-bit rows (factors: the softmax scale and 1).
    auto convert = [&](int n0, int s) {
        if constexpr(BYTE_PAGES && !IREG) {
            const unsigned char* st = stages + s * STAGE_BYTES;
            const float* ss = reinterpret_cast<const float*>(st + STAGE_K + STAGE_V);
            constexpr int KC = D / 8, VC = DVP / 8;
            for(int i = tid; i < BN * (KC + VC); i += THREADS) {
                const bool is_k = i < BN * KC;
                const int at = is_k ? i : i - BN * KC;
                const int pos = is_k ? at / KC : at / VC, ch = is_k ? at % KC : at % VC;
                const int n = n0 + pos;
                uint4 v;
                if(n >= ffirst && n < hi) {
                    const long long t = n - fshift;
                    const T* src = is_k ? fk + t * p.stride_fk + (long long)kvh * D + ch * 8
                                        : fv + t * p.stride_fv + (long long)kvh * DV + vofs + ch * 8;
                    v = *reinterpret_cast<const uint4*>(src);
                } else {
                    const unsigned char* src = is_k ? st + pos * D + ch * 8 : st + STAGE_K + pos * DVP + ch * 8;
                    v = codes_to_16(*reinterpret_cast<const uint2*>(src));
                }
                *reinterpret_cast<uint4*>(is_k ? ck + sw<D>(pos, ch) : cv + sw<DVP>(pos, ch)) = v;
            }
            for(int i = tid; i < 2 * BN; i += THREADS) {
                const int pos = i % BN;
                const int n = n0 + pos;
                const bool own = n >= ffirst;
                float f;
                if(i < BN) {
                    const float scale = KIND >= 2 ? ss[pos] : p.k_mult;
                    f = own ? p.qk_scale : __fmul_rn(scale, p.qk_scale);
                } else {
                    const float scale = KIND >= 2 ? ss[BN + pos] : p.v_mult;
                    f = own ? 1.0f : scale;
                }
                (i < BN ? cks : cvs)[pos] = f;
            }
        }
    };

    // -- the state: the split's output, maximum and lane sums
    float acc[NVW][4];
    float m[2], l[2];
#pragma unroll
    for(int t = 0; t < NVW; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
    kc::softmax_rows_init(m, l);
    // A chunk's second pass: every thread's state as the first pass left it (the same tile, block for block).
    const long long state_at = ((long long)(blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * THREADS + tid;
    const long long state_n = (long long)gridDim.x * gridDim.y * gridDim.z * THREADS;
    if(flags & F_STATE_IN) {
#pragma unroll
        for(int t = 0; t < NVW; ++t)
#pragma unroll
            for(int i = 0; i < 4; ++i) acc[t][i] = __ldcg(p.state + (4 * t + i) * state_n + state_at);
#pragma unroll
        for(int h = 0; h < 2; ++h) {
            m[h] = __ldcg(p.state + (4 * NVW + h) * state_n + state_at);
            l[h] = __ldcg(p.state + (4 * NVW + 2 + h) * state_n + state_at);
        }
    }

    const int nblk = (hi - lo + BN - 1) / BN;
    // The lane's ldmatrix offsets (elements), the same every block: k-step kk's chunk 2 kk + h of a swizzled row
    // (sw) is at offs[kk % 4] + (kk / 4) * 64 (the swizzle touches a chunk index's low three bits only).
    int qoff[4], koff[4], voff[4];
#pragma unroll
    for(int j = 0; j < 4; ++j) {
        const int qrow = 16 * warp + (lane & 15);
        qoff[j] = qrow * D + (((2 * j + (lane >> 4)) ^ (qrow & 7)) << 3);
        koff[j] = ((lane & 7) + ((lane >> 4) << 3)) * D + (((2 * j + ((lane >> 3) & 1)) ^ (lane & 7)) << 3);
        voff[j] = ((lane & 7) + (((lane >> 3) & 1) << 3)) * DVP + (((2 * j + (lane >> 4)) ^ (lane & 7)) << 3);
    }
    // One stage of 16-bit pages: the keys and the values in buffers of their own, each loaded while the other is
    // read (FlashInfer's single-stage form: the next block's keys during this block's softmax and values, its values
    // during the next block's scores), so a CTA takes a third less shared memory and two fit an SM.
    constexpr bool SPLIT_KV = STAGES == 1 && (!BYTE_PAGES || IREG);
    if constexpr(SPLIT_KV) {
        if(nblk > 0) issue(lo, 0, 1);
        kc::cp_async_commit();
        if(nblk > 0) issue(lo, 0, 2);
        kc::cp_async_commit();
    } else {
#pragma unroll
        for(int s = 0; s < STAGES - 1; ++s) {
            if(s < nblk) issue(lo + s * BN, s);
            kc::cp_async_commit();
        }
    }
    for(int it = 0; it < nblk; ++it) {
        const int n0 = lo + it * BN;
        const int s = it % STAGES;
        if constexpr(SPLIT_KV) {
            kc::cp_async_wait<1>();
            __syncthreads();
        } else if constexpr(STAGES == 1) {
            __syncthreads();
            issue(n0, 0);
            kc::cp_async_commit();
            kc::cp_async_wait<0>();
            __syncthreads();
        } else {
            kc::cp_async_wait<STAGES - 2>();
            __syncthreads();
            const int next = it + STAGES - 1;
            if(next < nblk) issue(lo + next * BN, next % STAGES);
            kc::cp_async_commit();
        }
        if constexpr(BYTE_PAGES && !IREG) {
            convert(n0, s);
            __syncthreads();
        }
        // A warp skips the math of blocks past its rows' last position (their syncs it keeps).
        // (warp votes: the same on every lane, which the compiler then knows, so these branches take no divergence)
        const bool active = __all_sync(0xffffffffu, warp_live && n0 <= warp_last);
        const T* kb = BYTE_PAGES ? ck : reinterpret_cast<const T*>(stages + s * STAGE_BYTES);
        const T* vb = BYTE_PAGES ? cv : reinterpret_cast<const T*>(stages + s * STAGE_BYTES + STAGE_K);
        // IREG: the stage's codes and raw scales
        const unsigned char* kb8 = stages + s * STAGE_BYTES;
        const unsigned char* vb8 = kb8 + STAGE_K;
        const float* ss8 = reinterpret_cast<const float*>(kb8 + STAGE_K + STAGE_V);
        float sc[NT][4];
        float kf[NT][2];   // IREG: the lane's positions' key factors (read before the keys' stage is reloaded)
        if(active) {
            // scores: the row group's 16 rows x BN positions (VW > 1: the warp's n8 tiles of them)
#pragma unroll
            for(int t = 0; t < NT; ++t) sc[t][0] = sc[t][1] = sc[t][2] = sc[t][3] = 0.0f;
#pragma unroll
            for(int kk = 0; kk < KSTEPS; ++kk) {
                uint32_t a[4];
                kc::ldmatrix_x4(a[0], a[1], a[2], a[3], sq + qoff[kk & 3] + (kk >> 2) * 64);
                if constexpr(IREG) {
                    // B: position 8 t + gid's codes d = 16 kk + 2 tig, + 1 (b0) and + 8, + 9 (b1)
#pragma unroll
                    for(int t = 0; t < NT; ++t) {
                        if(VW > 1 && C::owner(t) != vw) continue;
                        const unsigned char* at = kb8 + sw8<D>(8 * t + gid, kk) + 2 * tig;
                        const uint32_t lo = *reinterpret_cast<const unsigned short*>(at);
                        const uint32_t hi = *reinterpret_cast<const unsigned short*>(at + 8);
                        const uint4 b = codes_to_16(make_uint2(lo | (hi << 16), 0u));
                        mma16(sc[t], a, b.x, b.y);
                    }
                } else if constexpr(VW == 1 || C::PAIRS) {
#pragma unroll
                    for(int u = 0; u < NT / 2; ++u) {
                        if(VW > 1 && C::owner(2 * u) != vw) continue;
                        uint32_t b0, b1, b2, b3;
                        kc::ldmatrix_x4(b0, b1, b2, b3, kb + 16 * u * D + koff[kk & 3] + (kk >> 2) * 64);
                        mma16(sc[2 * u], a, b0, b1);
                        mma16(sc[2 * u + 1], a, b2, b3);
                    }
                } else {
#pragma unroll
                    for(int t = 0; t < NT; ++t) {
                        if(C::owner(t) != vw) continue;
                        uint32_t b0, b1;
                        kc::ldmatrix_x2(b0, b1, kb + sw<D>(8 * t + (lane & 7), 2 * kk + ((lane >> 3) & 1)));
                        mma16(sc[t], a, b0, b1);
                    }
                }
            }
        }
        if constexpr(IREG) {
            if(active) {
#pragma unroll
                for(int t = 0; t < NT; ++t) {
#pragma unroll
                    for(int j = 0; j < 2; ++j) {
                        const float scale = KIND >= 2 ? ss8[8 * t + 2 * tig + j] : p.k_mult;
                        kf[t][j] = __fmul_rn(scale, p.qk_scale);
                    }
                }
            }
        }
        // scale, mask (VW > 1: the warp's tiles, then shared with the group's other warps)
        const bool full = __all_sync(0xffffffffu, n0 + BN - 1 <= warp_seen && (win == 0 || n0 > warp_last - win));
        auto scale_mask = [&](int t) {
            if(full) {
#pragma unroll
                for(int i = 0; i < 4; ++i)
                    sc[t][i] = __fmul_rn(sc[t][i], IREG ? kf[t][i & 1] : BYTE_PAGES ? cks[8 * t + 2 * tig + (i & 1)]
                                                                                   : p.qk_scale);
                return;
            }
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const int h = i >> 1;
                const int pos = 8 * t + 2 * tig + (i & 1);
                const int n = n0 + pos;
                const float f = IREG ? kf[t][i & 1] : BYTE_PAGES ? cks[pos] : p.qk_scale;
                bool vis = live[h] && n < hi;
                if(tree) {
                    const int wd = n - first;
                    vis = vis && (n < first || (wd >= 0 && wd < count && ((anc[h] >> wd) & 1ull)));
                } else {
                    vis = vis && n <= lim[h];
                }
                if(win > 0) vis = vis && n > lim[h] - win;
                sc[t][i] = vis ? __fmul_rn(sc[t][i], f) : -INFINITY;
            }
        };
        if constexpr(VW > 1) {
            if(active) {
#pragma unroll
                for(int t = 0; t < NT; ++t) {
                    if(C::owner(t) != vw) continue;
                    scale_mask(t);
                    ssc[(warp * NT + t) * 32 + lane] = make_float4(sc[t][0], sc[t][1], sc[t][2], sc[t][3]);
                }
            }
            if constexpr(!SPLIT_KV) __syncthreads();   // (SPLIT_KV: the sync below)
        }
        if constexpr(SPLIT_KV) {
            __syncthreads();
            if(it + 1 < nblk) issue(n0 + BN, 0, 1);
            kc::cp_async_commit();
        }
        if(active) {
            if constexpr(VW > 1) {
#pragma unroll
                for(int t = 0; t < NT; ++t) {
                    if(C::owner(t) == vw) continue;
                    const float4 x = ssc[(warp * NT + t) * 32 + lane];
                    sc[t][0] = x.x, sc[t][1] = x.y, sc[t][2] = x.z, sc[t][3] = x.w;
                }
            } else {
#pragma unroll
                for(int t = 0; t < NT; ++t) scale_mask(t);
            }
            float alpha[2];
            kc::softmax_step_m16<NT>(sc, m, l, alpha);
            // The outputs rescaled where a row's maximum moved (x 1 changes no bit: skipped where no row's did).
            if(__any_sync(0xffffffffu, alpha[0] != 1.0f || alpha[1] != 1.0f)) {
#pragma unroll
                for(int t = 0; t < NVW; ++t) {
                    acc[t][0] = __fmul_rn(acc[t][0], alpha[0]);
                    acc[t][1] = __fmul_rn(acc[t][1], alpha[0]);
                    acc[t][2] = __fmul_rn(acc[t][2], alpha[1]);
                    acc[t][3] = __fmul_rn(acc[t][3], alpha[1]);
                }
            }
            if constexpr(BYTE_PAGES && !IREG) {
#pragma unroll
                for(int t = 0; t < NT; ++t) {
#pragma unroll
                    for(int i = 0; i < 4; ++i) sc[t][i] = __fmul_rn(sc[t][i], cvs[8 * t + 2 * tig + (i & 1)]);
                }
            }
        }
        if constexpr(SPLIT_KV) {
            kc::cp_async_wait<1>();
            __syncthreads();
        }
        if(active) {
            if constexpr(IREG) {   // the probabilities times the positions' value factors (their stage landed)
#pragma unroll
                for(int t = 0; t < NT; ++t) {
#pragma unroll
                    for(int i = 0; i < 4; ++i) {
                        const float f = KIND >= 2 ? ss8[BN + 8 * t + 2 * tig + (i & 1)] : p.v_mult;
                        sc[t][i] = __fmul_rn(sc[t][i], f);
                    }
                }
            }
            // output += P V (the warp's value tiles)
#pragma unroll
            for(int kp = 0; kp < BN / 16; ++kp) {
                uint32_t a[4];
                kc::probabilities_to_a<T>(sc[2 * kp], sc[2 * kp + 1], a);
                if constexpr(IREG) {
                    // 32 value columns (chunks 2 q, 2 q + 1) x 16 positions by one ldmatrix .trans of 16-bit pairs:
                    // a lane gets positions 2 tig, + 1 (and + 8, + 9) of columns 2 gid, + 1 of each chunk; the
                    // quad's lanes 4 h + gid / 2 hold column 8 h + gid's codes.
#pragma unroll
                    for(int g = 0; g < NVW / 4; ++g) {
                        const int q = tv0 / 4 + g;
                        const int row = 16 * kp + (lane & 7) + (((lane >> 3) & 1) << 3);
                        uint32_t r0, r1, r2, r3;
                        kc::ldmatrix_x4_trans(r0, r1, r2, r3, vb8 + sw8<DVP>(row, 2 * q + (lane >> 4)));
                        const uint32_t ev[2] = {__byte_perm(r0, r1, 0x6420), __byte_perm(r2, r3, 0x6420)};
                        const uint32_t od[2] = {__byte_perm(r0, r1, 0x7531), __byte_perm(r2, r3, 0x7531)};
#pragma unroll
                        for(int c = 0; c < 2; ++c) {
#pragma unroll
                            for(int h = 0; h < 2; ++h) {
                                const int src = (lane & ~31) | ((4 * h + (gid >> 1)) << 2) | tig;
                                const uint32_t xe = __shfl_sync(0xffffffffu, ev[c], src);
                                const uint32_t xo = __shfl_sync(0xffffffffu, od[c], src);
                                const uint4 b = codes_to_16(make_uint2((gid & 1) ? xo : xe, 0u));
                                mma16(acc[4 * g + 2 * c + h], a, b.x, b.y);
                            }
                        }
                    }
                    continue;
                }
#pragma unroll
                for(int v = 0; v < NVW / 2; ++v) {
                    uint32_t b0, b1, b2, b3;
                    if constexpr(NVW % 8 == 0) {   // the warp's first value chunk a multiple of 8: voff's form
                        kc::ldmatrix_x4_trans(b0, b1, b2, b3,
                                              vb + 16 * kp * DVP + tv0 * 8 + voff[v & 3] + (v >> 2) * 64);
                    } else {
                        const int pos = 16 * kp + (lane & 7) + (((lane >> 3) & 1) << 3);
                        kc::ldmatrix_x4_trans(b0, b1, b2, b3, vb + sw<DVP>(pos, tv0 + 2 * v + (lane >> 4)));
                    }
                    mma16(acc[2 * v], a, b0, b1);
                    mma16(acc[2 * v + 1], a, b2, b3);
                }
            }
        }
        if constexpr(SPLIT_KV) {
            __syncthreads();
            if(it + 1 < nblk) issue(n0 + BN, 0, 2);
            kc::cp_async_commit();
        }
    }
    kc::cp_async_wait<0>();

    if(flags & F_STATE_OUT) {   // a chunk's first pass: the state for its second
#pragma unroll
        for(int t = 0; t < NVW; ++t)
#pragma unroll
            for(int i = 0; i < 4; ++i) __stcg(p.state + (4 * t + i) * state_n + state_at, acc[t][i]);
#pragma unroll
        for(int h = 0; h < 2; ++h) {
            __stcg(p.state + (4 * NVW + h) * state_n + state_at, m[h]);
            __stcg(p.state + (4 * NVW + 2 + h) * state_n + state_at, l[h]);
        }
        return;
    }

    // -- the result
    float lt[2];
#pragma unroll
    for(int h = 0; h < 2; ++h) lt[h] = kc::softmax_row_total(l[h]);
    if(one_split) {
        if(!warp_live) return;
        T* out = reinterpret_cast<T*>(p.out);
#pragma unroll
        for(int h = 0; h < 2; ++h) {
            if(!live[h]) continue;
            T* row = out + (start + col[h]) * p.stride_ot + (long long)(kvh * G + hd[h]) * p.stride_oh;
            float mh = m[h], lh = lt[h], keep = 1.0f;
            const bool sink = flags & F_SINKS;
            if(sink) {
                const kc::SoftmaxFold2 f = kc::softmax_sink2(mh, lh, p.sinks[kvh * G + hd[h]]);
                mh = f.m, lh = f.l, keep = f.keep;
            }
#pragma unroll
            for(int t = 0; t < NVW; ++t) {
                const float ax = sink ? __fmul_rn(acc[t][2 * h], keep) : acc[t][2 * h];
                const float ay = sink ? __fmul_rn(acc[t][2 * h + 1], keep) : acc[t][2 * h + 1];
                const float x = kc::softmax_finish(ax, lh);
                const float y = kc::softmax_finish(ay, lh);
                *reinterpret_cast<uint32_t*>(row + 8 * (tv0 + t) + 2 * tig) = pack_out(x, y);
            }
            if((flags & F_LSE) && tig == 0 && vw == 0)
                p.lse[(start + col[h]) * p.hq + kvh * G + hd[h]] = kc::softmax_lse2(mh, lh);
        }
        return;
    }
    if constexpr(!FUSED) {
        if(!warp_live) return;
        long long prow;
        if(flags & F_OFFSETS) prow = p.offsets[r] - p.offsets[0];
        else prow = (long long)r * p.cols;
#pragma unroll
        for(int h = 0; h < 2; ++h) {
            if(!live[h]) continue;
            const long long idx = ((prow + col[h]) * p.hq + kvh * G + hd[h]) * p.n_parts + w.split;
            // A split's output as BF16 pairs (half the bytes of FP32 partials; the merge folds the rounded values),
            // lane-major (partial_word: the lane's NV pairs of the row contiguous), 16 bytes a store
            uint4* o = reinterpret_cast<uint4*>(reinterpret_cast<uint32_t*>(p.po) + idx * (DV / 2) + tig * NV) +
                       tv0 / 4;
#pragma unroll
            for(int u = 0; u < NVW / 4; ++u)
                o[u] = make_uint4(kc::pack_bf16x2(acc[4 * u][2 * h], acc[4 * u][2 * h + 1]),
                                  kc::pack_bf16x2(acc[4 * u + 1][2 * h], acc[4 * u + 1][2 * h + 1]),
                                  kc::pack_bf16x2(acc[4 * u + 2][2 * h], acc[4 * u + 2][2 * h + 1]),
                                  kc::pack_bf16x2(acc[4 * u + 3][2 * h], acc[4 * u + 3][2 * h + 1]));
            if(tig == 0 && vw == 0) {
                p.pm[idx] = m[h];
                p.pl[idx] = lt[h];
            }
        }
    } else {
        // The tile's chain: ring slots of its group, its running state and sync words (an arrival count a ring slot of
        // groups, cumulative over the launch). Group h's programs (in split order of ticket) each store their state,
        // then fold their share of the rows of group h - 1's states into the running state (h - 1's fold spread over
        // h's programs: every one waited for group h - 1 to be complete before storing, which also frees its ring
        // slot), then arrive; the last arrival of the last group folds that group over every row and writes the
        // outputs. A program waits only for programs of the group before its own, whose tickets are earlier: they run
        // already and wait only for earlier ones in turn. Each row's fold is the merge's, group by group in split
        // order: the split launch's bits.
        const int GROUP = p.group;
        const int chain = (rk * (DV / DVP) + b) * p.tiles + mt;
        const int RING = p.ring;                              // groups of slots a tile's ring holds
        int* flag = p.sync + SYNC_HEAD + (1 + RING) * chain;  // flag[1 + g % RING]: arrivals of groups g, g + RING ...
        const int n_splits = int(splits_to(n_keys, chunk, p.ramp, p.ramp_k));
        const int grp = w.split / GROUP, n_groups = (n_splits + GROUP - 1) / GROUP;
        const int in_group = min(GROUP, n_splits - grp * GROUP);
        const long long slot0 = (long long)chain * RING * GROUP;
        const int my_slot = (grp % RING) * GROUP + w.split % GROUP;
        const int live_rows = min(ROWS, count * G - j0);
        float* run = p.run + (long long)chain * ROWS * (DVP + 2);
        // Fold rows [r0, r1) of group g's n states into the running state (into the outputs where `final`).
        auto fold_rows = [&](int g, int n, int r0, int r1, bool final) {
            // The rows as items of 8 values (a lane's 4 pairs of a row, as the slots hold them) over every thread of
            // the program, a row's items one warp's (or part of one), each item's states of the group loaded at once
            // and folded in split order with the merge's steps.
            T* out = reinterpret_cast<T*>(p.out);
            const unsigned int* states = reinterpret_cast<const unsigned int*>(p.po);
            constexpr int Q = NV / 4;                 // items a row and lane
            constexpr int MAXG = 16;                  // states a group at most (FUSED_GROUP)
            const int items = (r1 - r0) * 4 * Q;
            for(int it = tid; it < items; it += THREADS) {
                const int rr = r0 + it / (4 * Q), lq = it % (4 * Q), tq = lq / Q, u = lq % Q;
                float m_run = -INFINITY, l_run = 0.0f;
                float o[8];
                float* rv = run + ((rr * 4 + tq) * Q + u) * 8;
                if(g > 0) {
                    const float4 x = __ldcg(reinterpret_cast<const float4*>(rv));
                    const float4 y = __ldcg(reinterpret_cast<const float4*>(rv) + 1);
                    o[0] = x.x, o[1] = x.y, o[2] = x.z, o[3] = x.w, o[4] = y.x, o[5] = y.y, o[6] = y.z, o[7] = y.w;
                    m_run = __ldcg(run + ROWS * DVP + 2 * rr);
                    l_run = __ldcg(run + ROWS * DVP + 2 * rr + 1);
                } else {
#pragma unroll
                    for(int i = 0; i < 8; ++i) o[i] = 0.0f;
                }
                uint4 sv[MAXG];
                float ms[MAXG], ls[MAXG];
#pragma unroll
                for(int s = 0; s < MAXG; ++s) {
                    if(s < n) {
                        const long long idx = (slot0 + (g % RING) * GROUP + s) * ROWS + rr;
                        sv[s] = __ldcg(reinterpret_cast<const uint4*>(states + idx * (DVP / 2) + tq * NV) + u);
                        ms[s] = __ldcg(p.pm + idx), ls[s] = __ldcg(p.pl + idx);
                    }
                }
#pragma unroll
                for(int s = 0; s < MAXG; ++s) {
                    if(s < n) {
                        const kc::SoftmaxFold2 f = kc::softmax_fold2(m_run, l_run, ms[s], ls[s]);
                        const uint32_t w4[4] = {sv[s].x, sv[s].y, sv[s].z, sv[s].w};
#pragma unroll
                        for(int i = 0; i < 4; ++i) {
                            o[2 * i] = kc::fold_output(o[2 * i], kc::bf16_lo(w4[i]), f);
                            o[2 * i + 1] = kc::fold_output(o[2 * i + 1], kc::bf16_hi(w4[i]), f);
                        }
                        m_run = f.m, l_run = f.l;
                    }
                }
                if(final) {
                    const int j = j0 + rr, c = j / G, hh = j % G;
                    if(flags & F_SINKS) {
                        const kc::SoftmaxFold2 f = kc::softmax_sink2(m_run, l_run, p.sinks[kvh * G + hh]);
#pragma unroll
                        for(int i = 0; i < 8; ++i) o[i] = __fmul_rn(o[i], f.keep);
                        m_run = f.m, l_run = f.l;
                    }
                    T* row = out + (start + c) * p.stride_ot + (long long)(kvh * G + hh) * p.stride_oh + vofs;
#pragma unroll
                    for(int i = 0; i < 4; ++i)   // pair t = 4 u + i: columns 8 t + 2 tq
                        *reinterpret_cast<uint32_t*>(row + 8 * (4 * u + i) + 2 * tq) =
                            pack_out(kc::softmax_finish(o[2 * i], l_run), kc::softmax_finish(o[2 * i + 1], l_run));
                    if((flags & F_LSE) && b == 0 && lq == 0)
                        p.lse[(start + c) * p.hq + kvh * G + hh] = kc::softmax_lse2(m_run, l_run);
                } else {
                    __stcg(reinterpret_cast<float4*>(rv), make_float4(o[0], o[1], o[2], o[3]));
                    __stcg(reinterpret_cast<float4*>(rv) + 1, make_float4(o[4], o[5], o[6], o[7]));
                    if(lq == 0) {
                        __stcg(run + ROWS * DVP + 2 * rr, m_run);
                        __stcg(run + ROWS * DVP + 2 * rr + 1, l_run);
                    }
                }
            }
        };
        // Group h - 1 complete (every one of its programs arrived: a full group, GROUP of them).
        if(grp > 0) {
            if(tid == 0) {
                const int want = ((grp - 1) / RING + 1) * GROUP;
                while(ld_acquire(flag + 1 + (grp - 1) % RING) < want) __nanosleep(64);
            }
            __syncthreads();
            __threadfence();
        }
        if(warp_live) {
#pragma unroll
            for(int h = 0; h < 2; ++h) {
                const int rr = 16 * warp + gid + 8 * h;
                const long long idx = (slot0 + my_slot) * ROWS + rr;
                // The lane's NV pairs of the row contiguous, so the folding program reads them 16 bytes at a time.
                uint4* o = reinterpret_cast<uint4*>(reinterpret_cast<uint32_t*>(p.po) + idx * (DVP / 2) + tig * NV) +
                           tv0 / 4;
#pragma unroll
                for(int u = 0; u < NVW / 4; ++u)
                    __stcg(o + u, make_uint4(kc::pack_bf16x2(acc[4 * u][2 * h], acc[4 * u][2 * h + 1]),
                                             kc::pack_bf16x2(acc[4 * u + 1][2 * h], acc[4 * u + 1][2 * h + 1]),
                                             kc::pack_bf16x2(acc[4 * u + 2][2 * h], acc[4 * u + 2][2 * h + 1]),
                                             kc::pack_bf16x2(acc[4 * u + 3][2 * h], acc[4 * u + 3][2 * h + 1])));
                if(tig == 0 && vw == 0) {
                    __stcg(p.pm + idx, m[h]);
                    __stcg(p.pl + idx, lt[h]);
                }
            }
        }
        if(grp > 0) {   // this program's rows of group h - 1's fold
            const int per = (live_rows + in_group - 1) / in_group, i = w.split % GROUP;
            const int r0 = min(live_rows, i * per), r1 = min(live_rows, r0 + per);
            if(r0 < r1) fold_rows(grp - 1, GROUP, r0, r1, false);
        }
        __threadfence();
        __syncthreads();
        if(tid == 0) {
            const int arrived = atomicAdd(flag + 1 + grp % RING, 1);
            s_word = grp == n_groups - 1 && arrived == (grp / RING) * GROUP + in_group - 1;
        }
        __syncthreads();
        if(!s_word) return;
        // The last group's last program: every group before it folded (each of its programs folded its rows of the
        // group before it before arriving); this group over every row, into the outputs.
        __threadfence();
        fold_rows(grp, in_group, 0, live_rows, true);
        __syncthreads();
        if(tid == 0) {
#pragma unroll 1
            for(int i = 0; i <= RING; ++i) flag[i] = 0;   // the chain is done: its words ready for the next launch
        }
    }
}

template <int DVP, int BN, int WARPS, int VW, int STAGES, bool IREG, bool FUSED>
__global__ void __launch_bounds__(32 * WARPS * VW) cw_window(const Params p) {
    extern __shared__ __align__(128) unsigned char smem[];
    if constexpr(!FUSED) {
        const Work w{int(gridDim.x - 1 - blockIdx.x), 0, int(blockIdx.z), int(blockIdx.y)};   // the longest tiles first
        cw_tile<DVP, BN, WARPS, VW, STAGES, IREG, false>(p, w, smem);
    } else {
        // A ticket in split order (every tile's first split, the longest tiles first, then every tile's second...):
        // a program waits only for programs of its tile's earlier splits, which hold earlier tickets and run already,
        // and the programs in flight spread over every tile (so their groups fill and fold side by side, not one
        // tile's splits waiting for its ring while the others wait for the GPU).
        __shared__ int s_ticket;
        const int total = int(gridDim.x * gridDim.y * gridDim.z);
        if(threadIdx.x == 0) s_ticket = atomicAdd(p.sync, 1);
        __syncthreads();
        const int t = s_ticket;
        const int chains = int(gridDim.x) * (DV / DVP) * int(gridDim.z);
        const int split = t / chains, chain = t % chains;
        const int mt = p.tiles - 1 - chain % p.tiles, rest = chain / p.tiles;
        const Work w{mt, rest % (DV / DVP), rest / (DV / DVP), split};
        cw_tile<DVP, BN, WARPS, VW, STAGES, IREG, true>(p, w, smem);
        __syncthreads();
        if(threadIdx.x == 0 && atomicAdd(p.sync + 1, 1) == total - 1) {
            p.sync[0] = 0;   // every program is done: the counters are ready for the next launch
            p.sync[1] = 0;
        }
    }
}

// The split launch's merge: program (column, MERGE_HEADS query heads, row), DV / 4 threads a head of 4 values each,
// the splits holding the column's positions folded in split order (the fused launch's steps), the next splits'
// partials loaded while one folds.
constexpr int MERGE_LANES = DV / 4, MERGE_HEADS = 256 / MERGE_LANES, MERGE_AHEAD = 4;
__global__ void __launch_bounds__(256) cw_merge(const Params p) {
    const int c = blockIdx.x, r = blockIdx.z;
    const int h = blockIdx.y * MERGE_HEADS + int(threadIdx.x) / MERGE_LANES;
    if(h >= p.hq) return;
    const int flags = p.flags;
    const bool one_row = flags & F_ONE_ROW;
    long long first;
    int count;
    if(one_row) {
        count = p.cols;
        first = (flags & F_POS_T) ? p.firsts[0] : p.first0;
    } else {
        count = (flags & F_COUNTS) ? min(p.counts[r], p.cols) : p.cols;
        first = p.firsts[r];
    }
    if(c >= count) return;
    const long long start = one_row ? 0 : p.starts[r];
    long long chunk = p.chunk;
    if(flags & F_DEV_CHUNK) chunk = cdivll(cdivll(first + count, p.n_parts), p.block) * p.block;
    const int nb_all = int(min((long long)p.n_parts, splits_to(first + c + 1, chunk, p.ramp, p.ramp_k)));
    // Under a sliding window the splits wholly below the column's window hold nothing (the window launch skips them):
    // the fold starts after them, which changes no bit.
    int below = 0;
    if(p.window > 0 && first + c - p.window + 1 > 0)
        below = int(min((long long)nb_all, splits_to(first + c - p.window + 1, chunk, p.ramp, p.ramp_k) - 1));
    const int nb = nb_all - below;
    long long prow;
    if(flags & F_OFFSETS) prow = p.offsets[r] - p.offsets[0];
    else prow = (long long)r * p.cols;
    const long long base = ((prow + c) * p.hq + h) * p.n_parts + below;
    const int e = 4 * (int(threadIdx.x) % MERGE_LANES);
    float m_run = -INFINITY, l_run = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t* pw = reinterpret_cast<const uint32_t*>(p.po);   // BF16 partial outputs (pairs, partial_word)
    const int w0 = partial_word(e / 2), w1 = partial_word(e / 2 + 1);
    // A ring of MERGE_AHEAD partials in flight (loads only: the folds stay in split order).
    float mq[MERGE_AHEAD], lq[MERGE_AHEAD];
    uint2 oq[MERGE_AHEAD];
#pragma unroll
    for(int i = 0; i < MERGE_AHEAD; ++i) {
        if(i < nb) {
            mq[i] = p.pm[base + i], lq[i] = p.pl[base + i];
            oq[i] = make_uint2(pw[(base + i) * (DV / 2) + w0], pw[(base + i) * (DV / 2) + w1]);
        }
    }
    for(int s0 = 0; s0 < nb; s0 += MERGE_AHEAD) {
#pragma unroll
        for(int i = 0; i < MERGE_AHEAD; ++i) {
            const int s = s0 + i;
            if(s < nb) {
                const float m = mq[i], l = lq[i];
                const float4 o = make_float4(kc::bf16_lo(oq[i].x), kc::bf16_hi(oq[i].x), kc::bf16_lo(oq[i].y),
                                             kc::bf16_hi(oq[i].y));
                const int nx = s + MERGE_AHEAD;
                if(nx < nb) {
                    mq[i] = p.pm[base + nx], lq[i] = p.pl[base + nx];
                    oq[i] = make_uint2(pw[(base + nx) * (DV / 2) + w0], pw[(base + nx) * (DV / 2) + w1]);
                }
                const kc::SoftmaxFold2 f = kc::softmax_fold2(m_run, l_run, m, l);
                acc[0] = kc::fold_output(acc[0], o.x, f);
                acc[1] = kc::fold_output(acc[1], o.y, f);
                acc[2] = kc::fold_output(acc[2], o.z, f);
                acc[3] = kc::fold_output(acc[3], o.w, f);
                m_run = f.m, l_run = f.l;
            }
        }
    }
    if(flags & F_SINKS) {
        const kc::SoftmaxFold2 f = kc::softmax_sink2(m_run, l_run, p.sinks[h]);
#pragma unroll
        for(int i = 0; i < 4; ++i) acc[i] = __fmul_rn(acc[i], f.keep);
        m_run = f.m, l_run = f.l;
    }
    T* out = reinterpret_cast<T*>(p.out) + (start + c) * p.stride_ot + (long long)h * p.stride_oh + e;
    uint32_t w2[2];
#pragma unroll
    for(int i = 0; i < 2; ++i)
        w2[i] = pack_out(kc::softmax_finish(acc[2 * i], l_run), kc::softmax_finish(acc[2 * i + 1], l_run));
    *reinterpret_cast<uint2*>(out) = make_uint2(w2[0], w2[1]);
    if((flags & F_LSE) && e == 0) p.lse[(start + c) * p.hq + h] = kc::softmax_lse2(m_run, l_run);
}

// The split launch's merge for rows of many splits: program (column, query head, row), a factor warp and DV / 2 fold
// threads (a BF16 pair of the head's outputs each). The column's partial outputs stream through shared memory by
// cp.async in batches of MERGE_BATCH splits, MERGE_STAGES batches in flight; the factor warp folds the batch's maxima
// and sums in split order (softmax_fold2: the running maximum, the sum, each split's keep and take, a batch ahead of
// the fold threads), and every fold thread folds its pair in split order with those factors (fold_output): the steps
// and bits of cw_merge, with the loads of a whole batch in flight at once.
constexpr int MERGE2_THREADS = 32 + DV / 2;
template <int MERGE_BATCH, int MERGE_STAGES>
constexpr int merge2_smem() { return MERGE_STAGES * MERGE_BATCH * DV * 2 + 2 * MERGE_BATCH * 2 * 4 + 2 * 32 * 4; }

template <int MERGE_BATCH, int MERGE_STAGES>
__global__ void __launch_bounds__(MERGE2_THREADS) cw_merge2(const Params p) {
    static_assert(MERGE_BATCH <= 32, "a batch's maxima and sums one a lane of the factor warp");
    constexpr int MERGE2_STAGE = MERGE_BATCH * DV * 2;
    extern __shared__ __align__(128) unsigned char smem[];
    const int c = blockIdx.x, h = blockIdx.y, r = blockIdx.z;
    const int flags = p.flags;
    const bool one_row = flags & F_ONE_ROW;
    long long first;
    int count;
    if(one_row) {
        count = p.cols;
        first = (flags & F_POS_T) ? p.firsts[0] : p.first0;
    } else {
        count = (flags & F_COUNTS) ? min(p.counts[r], p.cols) : p.cols;
        first = p.firsts[r];
    }
    if(c >= count) return;
    const long long start = one_row ? 0 : p.starts[r];
    long long chunk = p.chunk;
    if(flags & F_DEV_CHUNK) chunk = cdivll(cdivll(first + count, p.n_parts), p.block) * p.block;
    const int nb_all = int(min((long long)p.n_parts, splits_to(first + c + 1, chunk, p.ramp, p.ramp_k)));
    // Under a sliding window the splits wholly below the column's window hold nothing (the window launch skips them):
    // the fold starts after them, which changes no bit.
    int below = 0;
    if(p.window > 0 && first + c - p.window + 1 > 0)
        below = int(min((long long)nb_all, splits_to(first + c - p.window + 1, chunk, p.ramp, p.ramp_k) - 1));
    const int nb = nb_all - below;
    long long prow;
    if(flags & F_OFFSETS) prow = p.offsets[r] - p.offsets[0];
    else prow = (long long)r * p.cols;
    const long long base = ((prow + c) * p.hq + h) * p.n_parts + below;
    const int tid = threadIdx.x, lane = tid & 31;
    const bool factor_warp = tid < 32;
    // fold threads: word k of a split's row (consecutive words, no bank conflict), the head's pair e (values 2 e,
    // 2 e + 1) with partial_word(e) = k
    const int k = tid - 32, e = 4 * (k % (DV / 8)) + k / (DV / 8);
    float* keep = reinterpret_cast<float*>(smem + MERGE_STAGES * MERGE2_STAGE);   // [2][MERGE_BATCH]
    float* take = keep + 2 * MERGE_BATCH;                                          // [2][MERGE_BATCH]
    float* ml = take + 2 * MERGE_BATCH;                                            // [2][32]: m, l of a batch
    const unsigned char* po = reinterpret_cast<const unsigned char*>(p.po);
    const int batches = (nb + MERGE_BATCH - 1) / MERGE_BATCH;

    auto issue = [&](int bt) {
        if(bt < batches) {
            unsigned char* st = smem + (bt % MERGE_STAGES) * MERGE2_STAGE;
            constexpr int CH = DV * 2 / 16;   // 16-byte chunks a split's output
            for(int i = tid; i < MERGE_BATCH * CH; i += MERGE2_THREADS) {
                const int s = bt * MERGE_BATCH + i / CH;
                const bool ok = s < nb;
                kc::cp_async16(st + i * 16, po + ((base + (ok ? s : 0)) * DV * 2 + (i % CH) * 16), ok);
            }
        }
        kc::cp_async_commit();
    };
    // The factor warp's state: the running maximum and sum (lane 0), the next batch's maxima and sums in registers.
    float m_run = -INFINITY, l_run = 0.0f;
    float m_next = -INFINITY, l_next = 0.0f;
    auto fetch = [&](int bt) {   // factor warp: lane i the batch's split i
        const int s = bt * MERGE_BATCH + lane;
        if(lane < MERGE_BATCH && bt < batches && s < nb) m_next = p.pm[base + s], l_next = p.pl[base + s];
    };
    auto factors = [&](int bt) {   // factor warp: batch bt's keep and take into buffer bt & 1
        ml[lane] = m_next, ml[32 + lane] = l_next;
        __syncwarp();
        fetch(bt + 1);
        if(lane == 0) {
            const int n = min(MERGE_BATCH, nb - bt * MERGE_BATCH);
            float* kb = keep + (bt & 1) * MERGE_BATCH;
            float* tb = take + (bt & 1) * MERGE_BATCH;
#pragma unroll 8
            for(int i = 0; i < n; ++i) {
                const kc::SoftmaxFold2 f = kc::softmax_fold2(m_run, l_run, ml[i], ml[32 + i]);
                kb[i] = f.keep, tb[i] = f.take;
                m_run = f.m, l_run = f.l;
            }
        }
        __syncwarp();
    };
#pragma unroll
    for(int s = 0; s < MERGE_STAGES - 1; ++s) issue(s);
    if(factor_warp) {
        fetch(0);
        factors(0);
    }
    float acc0 = 0.0f, acc1 = 0.0f;
    for(int bt = 0; bt < batches; ++bt) {
        kc::cp_async_wait<MERGE_STAGES - 2>();
        __syncthreads();   // batch bt landed, its factors ready, batch bt - 1 folded (its stage free)
        issue(bt + MERGE_STAGES - 1);
        if(factor_warp) {
            if(bt + 1 < batches) factors(bt + 1);
        } else {
            const uint32_t* st = reinterpret_cast<const uint32_t*>(smem + (bt % MERGE_STAGES) * MERGE2_STAGE) + k;
            const float* kb = keep + (bt & 1) * MERGE_BATCH;
            const float* tb = take + (bt & 1) * MERGE_BATCH;
            const int n = min(MERGE_BATCH, nb - bt * MERGE_BATCH);
#pragma unroll 8
            for(int i = 0; i < n; ++i) {
                const uint32_t w = st[i * (DV / 2)];
                const kc::SoftmaxFold2 f{0.0f, 0.0f, kb[i], tb[i]};
                acc0 = kc::fold_output(acc0, kc::bf16_lo(w), f);
                acc1 = kc::fold_output(acc1, kc::bf16_hi(w), f);
            }
        }
    }
    kc::cp_async_wait<0>();
    if(tid == 0) {
        float keep = 1.0f;
        if(flags & F_SINKS) {   // the head's sink folded into the final state, its factor shared with the fold threads
            const kc::SoftmaxFold2 f = kc::softmax_sink2(m_run, l_run, p.sinks[h]);
            m_run = f.m, l_run = f.l, keep = f.keep;
        }
        ml[0] = m_run, ml[1] = l_run, ml[2] = keep;
    }
    __syncthreads();
    if(factor_warp) {
        if(tid == 0 && (flags & F_LSE)) p.lse[(start + c) * p.hq + h] = kc::softmax_lse2(m_run, l_run);
        return;
    }
    const float lt = ml[1];
    if(flags & F_SINKS) acc0 = __fmul_rn(acc0, ml[2]), acc1 = __fmul_rn(acc1, ml[2]);
    T* out = reinterpret_cast<T*>(p.out) + (start + c) * p.stride_ot + (long long)h * p.stride_oh + 2 * e;
    *reinterpret_cast<uint32_t*>(out) = pack_out(kc::softmax_finish(acc0, lt), kc::softmax_finish(acc1, lt));
}

// -- configurations: the tuning's (crit_tuning.h), their rules resolved for this variant

struct Resolved {
    int dvp, bn, warps, stages, vsplit, vw;
    bool ireg, ok;
};

// The untuned tile shapes a window or fused launch picks from by the query rows of a tile (a row's columns times a KV
// head's query heads, crit_cw_plan): RULE_SHAPES of them, (row groups, warps a row group).
constexpr int RULE_SHAPES = 3;
constexpr int RULE_WARPS[RULE_SHAPES] = {1, 2, 4}, RULE_VW[RULE_SHAPES] = {4, 2, 1};
__host__ __device__ constexpr int rule_shape(long long query_rows) {
    return query_rows <= 16 ? 0 : query_rows <= 32 ? 1 : 2;
}

// A configuration's rules: window and fused launches over 16-bit pages blocks of 16 in one stage (the keys and the
// values in buffers of their own, each loaded while the other is read: the least shared memory, so the most programs
// an SM), over 8-bit pages blocks of 32 in two stages (their codes are converted a block at a time, which one stage
// would not overlap); prompt chunks blocks of 32 in two stages; warps 0 -> the rule's tile shape for the call's query
// rows (`shape`, rule_shape): as many row groups as the rows fill up to four, the rest of four warps sharing the
// groups' scores and value columns (VW); a tuning's warps without VWARPS one warp a row group; the caller's overrides
// win; fewer stages, then fewer warps, where the GPU's shared memory does not hold the tile.
constexpr Resolved resolve(int bn, int warps, int stages, int vsplit, bool fused, bool prefill = false, int vw = 0,
                           int shape = RULE_SHAPES - 1, int convert = 0, bool fresh = true) {
    const bool narrow = !prefill && !BYTE_PAGES;
    bn = bn ? bn : (narrow ? 16 : 32);
    const bool rule = !CRIT_VARIANT_W && !warps && !prefill;
    warps = CRIT_VARIANT_W ? CRIT_VARIANT_W : (warps ? warps : (prefill ? 4 : RULE_WARPS[shape]));
    vw = prefill ? 1 : (vw ? vw : (rule ? RULE_VW[shape] : 1));
    int split = 1;
    if(fused) {
        split = CRIT_VARIANT_VS ? CRIT_VARIANT_VS : (vsplit ? vsplit : 1);
        while(split > 1 && (DV / split < 64 || DV % split)) split /= 2;
    }
    const int dvp = DV / split;
    // 8-bit pages' codes converted in registers (convert 2, the rule) or in a shared 16-bit block (1, and every launch
    // with fresh rows: those positions come as 16-bit rows)
    const bool ireg = BYTE_PAGES && !fresh && convert != 1;
    while(vw > 1 && (dvp / 8 % vw || dvp / 8 / vw < 4)) vw /= 2;
    while(vw > 1 && warps * vw > 16) vw /= 2;
    if(CRIT_VARIANT_S) stages = CRIT_VARIANT_S;
    else if(!stages) stages = narrow ? 1 : 2;
    while(stages > 1 && smem_bytes(dvp, bn, warps, stages, vw, ireg) > CRIT_VARIANT_SMEM) --stages;
    while(warps > 1 && smem_bytes(dvp, bn, warps, stages, vw, ireg) > CRIT_VARIANT_SMEM) --warps;
    return Resolved{dvp, bn, warps, stages, split, vw, ireg,
                    smem_bytes(dvp, bn, warps, stages, vw, ireg) <= CRIT_VARIANT_SMEM};
}

struct WindowKnobs { int chunk, bn, warps, stages, vwarps, convert, ramp; };
struct FusedKnobs { int warps, stages, vsplit, vwarps, convert; };
struct PrefillKnobs { int bn, warps, stages, splits; };
#define CW_WINDOW(i, chunk, bn, warps, stages, vwarps, convert, ramp) \
    WindowKnobs{chunk, bn, warps, stages, vwarps, convert, ramp},
#define CW_FUSED(i, warps, stages, vsplit, vwarps, convert) FusedKnobs{warps, stages, vsplit, vwarps, convert},
#define CW_PREFILL(i, bn, warps, stages, splits) PrefillKnobs{bn, warps, stages, splits},
constexpr WindowKnobs kWindow[] = {CRIT_CONFIGS_cuda_window(CW_WINDOW)};
constexpr FusedKnobs kFused[] = {CRIT_CONFIGS_cuda_fused(CW_FUSED)};
constexpr PrefillKnobs kPrefill[] = {CRIT_CONFIGS_cuda_prefill(CW_PREFILL)};
constexpr int NW = sizeof(kWindow) / sizeof(kWindow[0]);
constexpr int NF = sizeof(kFused) / sizeof(kFused[0]);
constexpr int NP = sizeof(kPrefill) / sizeof(kPrefill[0]);

// A sweep build instantiates only the structure it sweeps (its configurations, the others' defaults unused).
#ifdef CRIT_SWEEP_BUILD
#ifdef CRIT_SWEEP_STRUCTURE_cuda_window
constexpr bool BUILD_WINDOW = true;
#else
constexpr bool BUILD_WINDOW = false;
#endif
#ifdef CRIT_SWEEP_STRUCTURE_cuda_fused
constexpr bool BUILD_FUSED = true;
#else
constexpr bool BUILD_FUSED = false;
#endif
#ifdef CRIT_SWEEP_STRUCTURE_cuda_prefill
constexpr bool BUILD_PREFILL = true;
#else
constexpr bool BUILD_PREFILL = false;
#endif
constexpr bool SWEEP = true;
#else
constexpr bool BUILD_WINDOW = true, BUILD_FUSED = true, BUILD_PREFILL = true, SWEEP = false;
#endif

using Launch = int (*)(const Params*, cudaStream_t, int, int, int);

template <int DVP, int BN, int WARPS, int VW, int STAGES, bool IREG, bool FUSED>
int launch_tile(const Params* p, cudaStream_t stream, int gx, int gy, int gz) {
    using C = Tile<DVP, BN, WARPS, VW, STAGES, IREG, FUSED>;
    static bool ready = false;
    if(!ready) {
        const cudaError_t e = cudaFuncSetAttribute(cw_window<DVP, BN, WARPS, VW, STAGES, IREG, FUSED>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        if(e != cudaSuccess) return int(e);
        ready = true;
    }
    // fused: gy splits times the value split's programs
    cw_window<DVP, BN, WARPS, VW, STAGES, IREG, FUSED><<<dim3(gx, FUSED ? gy * (DV / DVP) : gy, gz), C::THREADS,
                                                         C::SMEM, stream>>>(*p);
    return int(cudaGetLastError());
}

// Window entries: (configuration for the call's rows, configuration for one row: its block, an order knob, the rule's
// tile shape, fresh rows or not), NW x NW x VARIANTS (a sweep build: the diagonal, select() picks one configuration for
// both). A configuration whose warps are given takes the same tile at every shape (one instantiation), and 16-bit pages
// the same with fresh rows or without.
constexpr int VARIANTS = RULE_SHAPES * 2;
template <int I>
constexpr Resolved window_resolved() {
    constexpr int c = I / VARIANTS / NW, o = I / VARIANTS % NW, shape = I % VARIANTS / 2;
    return resolve(kWindow[o].bn, kWindow[c].warps, kWindow[c].stages, 0, false, false, kWindow[c].vwarps, shape,
                   kWindow[c].convert, I % 2);
}
template <int I>
Launch window_entry() {
    constexpr Resolved r = window_resolved<I>();
    constexpr int c = I / VARIANTS / NW, o = I / VARIANTS % NW;
    if constexpr(BUILD_WINDOW && r.ok && (!SWEEP || c == o))
        return &launch_tile<r.dvp, r.bn, r.warps, r.vw, r.stages, r.ireg, false>;
    else
        return nullptr;
}
// Fused entries: (fused configuration, window configuration for one row: the block, the rule's tile shape, fresh rows
// or not), NF x NW x VARIANTS.
template <int I>
constexpr Resolved fused_resolved() {
    constexpr int c = I / VARIANTS / NW, o = I / VARIANTS % NW, shape = I % VARIANTS / 2;
    return resolve(kWindow[o].bn, kFused[c].warps, kFused[c].stages, kFused[c].vsplit, true, false, kFused[c].vwarps,
                   shape, kFused[c].convert, I % 2);
}
template <int I>
Launch fused_entry() {
    constexpr Resolved r = fused_resolved<I>();
    if constexpr(BUILD_FUSED && r.ok) return &launch_tile<r.dvp, r.bn, r.warps, r.vw, r.stages, r.ireg, true>;
    else return nullptr;
}
template <int I>
constexpr Resolved prefill_resolved() {
    // Prompt entries: (configuration, fresh rows or not), NP x 2: without fresh rows 8-bit pages convert in registers.
    return resolve(kPrefill[I / 2].bn, kPrefill[I / 2].warps, kPrefill[I / 2].stages, 0, false, true, 0,
                   RULE_SHAPES - 1, 0, I % 2);
}
template <int I>
Launch prefill_entry() {
    constexpr Resolved r = prefill_resolved<I>();
    if constexpr(BUILD_PREFILL && D == DV && r.ok)
        return &launch_tile<r.dvp, r.bn, r.warps, 1, r.stages, r.ireg, false>;
    else return nullptr;
}

template <int... I>
const Launch* window_table(std::integer_sequence<int, I...>) {
    static const Launch t[] = {window_entry<I>()...};
    return t;
}
template <int... I>
const Launch* fused_table(std::integer_sequence<int, I...>) {
    static const Launch t[] = {fused_entry<I>()...};
    return t;
}
template <int... I>
const Launch* prefill_table(std::integer_sequence<int, I...>) {
    static const Launch t[] = {prefill_entry<I>()...};
    return t;
}
template <int... I>
const Resolved* window_info(std::integer_sequence<int, I...>) {
    static const Resolved t[] = {window_resolved<I>()...};
    return t;
}
template <int... I>
const Resolved* fused_info(std::integer_sequence<int, I...>) {
    static const Resolved t[] = {fused_resolved<I>()...};
    return t;
}
template <int... I>
const Resolved* prefill_info(std::integer_sequence<int, I...>) {
    static const Resolved t[] = {prefill_resolved<I>()...};
    return t;
}

enum : int { S_WINDOW = 0, S_FUSED = 1, S_PREFILL = 2 };

static const Launch* table_of(int structure) {
    if(structure == S_WINDOW) return window_table(std::make_integer_sequence<int, NW * NW * VARIANTS>{});
    if(structure == S_FUSED) return fused_table(std::make_integer_sequence<int, NF * NW * VARIANTS>{});
    return prefill_table(std::make_integer_sequence<int, NP * 2>{});
}

}  // namespace

// The configuration of a call (fresh: the launch reads fresh rows; length: the positions its page tables hold, the
// tuning's length ranges'; -1 any): out = {entry, warps (row groups), stages, block,
// value split, shared memory, warps a row group, codes converted in registers}. Returns 0, or 1 where the build holds
// no such configuration (a sweep build's other structures), 2 where it does not fit the GPU.
extern "C" int crit_cw_plan(int structure, long long hq, long long hkv, long long page, int rows, int fresh,
                            long long length, int* out) {
    int entry;
    const Resolved* info;
    const int variant = rule_shape((long long)rows * (hq / hkv)) * 2 + (fresh ? 1 : 0);
    if(structure == S_WINDOW) {
        const int i = crit_lookup_cuda_window(D, DV, hq, hkv, page, KIND, rows, length);
        const int j = crit_lookup_cuda_window(D, DV, hq, hkv, page, KIND, 1, length);
        entry = (i * NW + j) * VARIANTS + variant;
        info = window_info(std::make_integer_sequence<int, NW * NW * VARIANTS>{});
    } else if(structure == S_FUSED) {
        const int i = crit_lookup_cuda_fused(D, DV, hq, hkv, page, KIND, rows, length);
        const int j = crit_lookup_cuda_window(D, DV, hq, hkv, page, KIND, 1, length);
        entry = (i * NW + j) * VARIANTS + variant;
        info = fused_info(std::make_integer_sequence<int, NF * NW * VARIANTS>{});
    } else {
        entry = crit_lookup_cuda_prefill(D, hq, hkv, page, KIND, 0, rows) * 2 + (fresh ? 1 : 0);
        info = prefill_info(std::make_integer_sequence<int, NP * 2>{});
    }
    const Resolved r = info[entry];
    out[0] = entry, out[1] = r.warps, out[2] = r.stages, out[3] = r.bn, out[4] = r.vsplit;
    out[5] = smem_bytes(r.dvp, r.bn, r.warps, r.stages, r.vw, r.ireg), out[6] = r.vw, out[7] = r.ireg;
    if(!r.ok) return 2;
    return table_of(structure)[entry] ? 0 : 1;
}

extern "C" int crit_cw_params_size() { return int(sizeof(Params)); }

extern "C" int crit_cw_launch(int structure, int entry, const Params* p, void* stream, int gx, int gy, int gz) {
    const Launch fn = table_of(structure)[entry];
    if(!fn) return int(cudaErrorInvalidDeviceFunction);
    return fn(p, static_cast<cudaStream_t>(stream), gx, gy, gz);
}

extern "C" int crit_cw_merge(const Params* p, void* stream, int gx, int gy, int gz) {
    if(p->merge) {   // the staged merge (rows of many splits)
        static bool ready = false;
        if(!ready) {
            cudaError_t e = cudaFuncSetAttribute(cw_merge2<32, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 merge2_smem<32, 4>());
            if(e == cudaSuccess)
                e = cudaFuncSetAttribute(cw_merge2<16, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         merge2_smem<16, 3>());
            if(e != cudaSuccess) return int(e);
            ready = true;
        }
        if(p->merge == 1)
            cw_merge2<32, 4><<<dim3(gx, gy, gz), MERGE2_THREADS, merge2_smem<32, 4>(),
                               static_cast<cudaStream_t>(stream)>>>(*p);
        else
            cw_merge2<16, 3><<<dim3(gx, gy, gz), MERGE2_THREADS, merge2_smem<16, 3>(),
                               static_cast<cudaStream_t>(stream)>>>(*p);
        return int(cudaGetLastError());
    }
    // gy: the query heads, in programs of MERGE_HEADS
    cw_merge<<<dim3(gx, (gy + MERGE_HEADS - 1) / MERGE_HEADS, gz), 256, 0, static_cast<cudaStream_t>(stream)>>>(*p);
    return int(cudaGetLastError());
}

extern "C" const char* crit_cw_error(int e) { return cudaGetErrorString(static_cast<cudaError_t>(e)); }
