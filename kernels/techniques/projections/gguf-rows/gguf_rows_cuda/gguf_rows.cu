// GGUF projections (gguf_rows.h): the activation quantizer, the dp4a GEMV straight from the stored blocks (plain and
// gathered), the parts' sum and the dequantization, in the order gguf_rows.h declares.
#include "gguf_rows.h"

#include "crit_tuning.h"
#include "crit_kcore/gguf.cuh"
#include "crit_kcore/load.cuh"
#include "crit_kcore/mma.cuh"
#include "crit_kcore/reduce.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Knobs (kernel.json, structure gemv; crit_tuning.h picks them per N, K and row count):
//   WARPS   warps a block, one output column (and part) a warp
//   SPLITS  contiguous parts of a row's sub-blocks, added in part order (0: enough warps for 32 a multiprocessor,
//           at least 32 sub-blocks a part): changes the sums' order, so one value a matrix (an order knob)
// structure gemm (the prompt rows): WN warps across weight rows (16 each), WT warps across tokens, TB 8-token tiles
// a warp, RUN sub-blocks of a lane a stage, STAGES stages copied ahead; none of them changes an output's bits.

namespace crit::techniques::gguf_rows {
namespace {
namespace gg = crit::kcore::gguf;

// Every format the kernels take (gguf.cuh's), for the launches' switches.
#define CRIT_GGUF_FORMATS(X) \
    X(Q4_0) X(Q4_1) X(Q5_0) X(Q5_1) X(Q8_0) X(Q2_K) X(Q3_K) X(Q4_K) X(Q5_K) X(Q6_K) X(IQ2_XXS) X(IQ2_XS) X(IQ3_XXS) \
    X(IQ1_S) X(IQ4_NL) X(IQ3_S) X(IQ2_S) X(IQ4_XS) X(IQ1_M) X(MXFP4)

__device__ __forceinline__ float load(const void* p, int kind, int64_t i) {
    if(kind == 0) return static_cast<const float*>(p)[i];
    if(kind == 1) return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
    return __half2float(static_cast<const __half*>(p)[i]);
}
__device__ __forceinline__ void store(void* p, int kind, int64_t i, float v) {
    if(kind == 0) static_cast<float*>(p)[i] = v;
    else if(kind == 1) static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
    else static_cast<__half*>(p)[i] = __float2half_rn(v);
}

// One warp a 32-value block: max |x| by the butterfly, d = max / 127, q = rint(x / d).
__global__ void quantize_kernel(const void* x, int kind, int64_t ld, int M, int K, bool by_subblock, int8_t* xq,
                                float* xd, int32_t* xs) {
    const int lane = threadIdx.x & 31;
    const int block = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int row = blockIdx.y;
    if(block >= K / 32) return;
    const int64_t k = int64_t(block) * 32 + lane;
    const float v = load(x, kind, int64_t(row) * ld + k);
    float amax = fabsf(v);
#pragma unroll
    for(int off = 16; off; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    // A NaN anywhere in the block makes d NaN, so it reaches every output the block feeds (fmaxf skips NaN).
    const bool nan = __any_sync(0xffffffffu, v != v);
    const float d = nan ? __int_as_float(0x7fffffff) : __fdiv_rn(amax, 127.0f);
    const int q = d == 0.0f || nan ? 0 : max(-127, min(127, __float2int_rn(__fdiv_rn(v, d))));
    int sum = q;
#pragma unroll
    for(int off = 16; off; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    // [M][K] and [M][K / 32], or by sub-block [K / 32][M][32] and [K / 32][M]
    const int64_t at = by_subblock ? int64_t(block) * M + row : int64_t(row) * (K / 32) + block;
    xq[by_subblock ? at * 32 + lane : int64_t(row) * K + k] = static_cast<int8_t>(q);
    if(lane == 0) {
        xd[at] = d;
        xs[at] = sum;
    }
}

// The term of one sub-block for one row: its exact integer dots weighted as gguf_rows.h declares.
template <int F>
__device__ __forceinline__ float term(const int32_t q[8], float s0, float s1, float m, const int8_t* xq, float d,
                                      int32_t sum) {
    const int4 a = reinterpret_cast<const int4*>(xq)[0];
    const int4 b = reinterpret_cast<const int4*>(xq)[1];
    int lo = __dp4a(q[0], a.x, 0);
    lo = __dp4a(q[1], a.y, lo);
    lo = __dp4a(q[2], a.z, lo);
    lo = __dp4a(q[3], a.w, lo);
    int hi = __dp4a(q[4], b.x, 0);
    hi = __dp4a(q[5], b.y, hi);
    hi = __dp4a(q[6], b.z, hi);
    hi = __dp4a(q[7], b.w, hi);
    float c;
    if constexpr(gg::two_scales(F)) {
        c = __fadd_rn(__fmul_rn(__fmul_rn(s0, d), float(lo)), __fmul_rn(__fmul_rn(s1, d), float(hi)));
    } else {
        c = __fmul_rn(__fmul_rn(s0, d), float(lo + hi));
    }
    if constexpr(gg::has_min(F)) c = __fsub_rn(c, __fmul_rn(__fmul_rn(m, d), float(sum)));
    return c;
}

// Q2_K's term (gguf.cuh's q2k_scaled): the scaled codes' exact dot isum and the minimums' imin = mn0 first + mn1
// (sum - first), first the activations' sum of their first 16 codes; fmul(d_x, fsub(fmul(d, isum), fmul(dmin, imin))).
__device__ __forceinline__ float q2k_term(const int32_t q[8], float dw, float dmin, int mn0, int mn1, const int8_t* xq,
                                          float d, int32_t sum) {
    const int4 a = reinterpret_cast<const int4*>(xq)[0];
    const int4 b = reinterpret_cast<const int4*>(xq)[1];
    int dot = __dp4a(q[0], a.x, 0);
    dot = __dp4a(q[1], a.y, dot);
    dot = __dp4a(q[2], a.z, dot);
    dot = __dp4a(q[3], a.w, dot);
    dot = __dp4a(q[4], b.x, dot);
    dot = __dp4a(q[5], b.y, dot);
    dot = __dp4a(q[6], b.z, dot);
    dot = __dp4a(q[7], b.w, dot);
    int first = __dp4a(0x01010101, a.x, 0);
    first = __dp4a(0x01010101, a.y, first);
    first = __dp4a(0x01010101, a.z, first);
    first = __dp4a(0x01010101, a.w, first);
    const int imin = mn0 * first + mn1 * (sum - first);
    return __fmul_rn(d, __fsub_rn(__fmul_rn(dw, float(dot)), __fmul_rn(dmin, float(imin))));
}

// One sub-block's decode and its term for row r, by format (Q2_K's integer form, else gguf.cuh's subblock).
template <int F>
struct Decoded {
    int32_t q[8];
    float s0, s1, m;
    int mn0, mn1;
    __device__ __forceinline__ void decode(const uint8_t* block, int ib) {
        if constexpr(F == gg::Q2_K) {
            gg::q2k_scaled(block, ib, q, s0, s1, mn0, mn1);
        } else {
            gg::subblock<F>(block, ib, q, s0, s1, m);
        }
    }
    __device__ __forceinline__ float term(const int8_t* xq, float d, int32_t sum) const {
        if constexpr(F == gg::Q2_K) return q2k_term(q, s0, s1, mn0, mn1, xq, d, sum);
        else return gguf_rows::term<F>(q, s0, s1, m, xq, d, sum);
    }
};

// One warp an (output column, part); all M rows (or, gathered, the rows whose expert this is not: each row its own
// expert's matrix). Lane l takes the part's sub-blocks start + l, start + l + 32, ...
template <int F, int M>
__global__ void gemv_kernel(Matrix w, const int8_t* xq, const float* xd, const int32_t* xs, const int32_t* ids,
                            int rows, float* yp, int parts) {
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int p = blockIdx.y;
    if(n >= w.N) return;
    constexpr int per_block = gg::block_values(F) / 32;
    const int J = w.K / 32;
    const int start = int(int64_t(J) * p / parts), end = int(int64_t(J) * (p + 1) / parts);
    float acc[M];
#pragma unroll
    for(int r = 0; r < M; ++r) acc[r] = 0.0f;
    if(ids == nullptr) {
        const uint8_t* row = w.blocks + int64_t(n) * w.row_bytes;
        for(int j = start + lane; j < end; j += 32) {
            Decoded<F> s;
            s.decode(row + int64_t(j / per_block) * gg::block_bytes(F), j % per_block);
#pragma unroll
            for(int r = 0; r < M; ++r) {
                if(r < rows) {
                    acc[r] = __fadd_rn(acc[r], s.term(xq + int64_t(r) * w.K + 32 * j,
                                                       xd[int64_t(r) * J + j], xs[int64_t(r) * J + j]));
                }
            }
        }
    } else {
#pragma unroll
        for(int r = 0; r < M; ++r) {
            if(r >= rows) continue;
            const uint8_t* row = w.blocks + int64_t(ids[r]) * w.expert_stride + int64_t(n) * w.row_bytes;
            for(int j = start + lane; j < end; j += 32) {
                Decoded<F> s;
                s.decode(row + int64_t(j / per_block) * gg::block_bytes(F), j % per_block);
                acc[r] = __fadd_rn(acc[r], s.term(xq + int64_t(r) * w.K + 32 * j,
                                                   xd[int64_t(r) * J + j], xs[int64_t(r) * J + j]));
            }
        }
    }
#pragma unroll
    for(int r = 0; r < M; ++r) {
#pragma unroll
        for(int off = 16; off; off >>= 1) acc[r] = __fadd_rn(acc[r], __shfl_xor_sync(0xffffffffu, acc[r], off));
        if(lane == 0 && r < rows) yp[(int64_t(p) * rows + r) * w.N + n] = acc[r];
    }
}

// The prompt GEMM in the same declared order, in two steps. stage_kernel decodes each sub-block of a chunk of W's
// rows once, as gemv decodes it (gguf.cuh's subblock: the int8 codes and FP32 scales, bit for bit), sub-block-major
// (codes [J][rows][32], scales [plane][J][rows]), so a stage of the GEMM reads them as contiguous runs. gemm_kernel
// then takes BN = 16 WN weight rows x BT = 8 TB WT tokens a CTA, WN x WT warps of 16 rows x 8 TB tokens (one A tile,
// TB B tiles of the s8 MMA). For each part, the part's sub-blocks are visited lane by lane in the order the 32-lane
// tree consumes the lanes (reduce.cuh's lane_at32), each lane's sub-blocks in increasing order, RUN of them a stage;
// a stage's codes, scales and the tokens' quantized sub-blocks are copied into shared memory by cp.async, STAGES - 1
// stages ahead of its MMAs. One m16n8k32 MMA a tile and sub-block gives the exact dots (two m16n8k16 for a format
// with 16-value scales), each term is added to the output's running lane as gemv adds it, and a finished lane folds
// into the tree's partial sums (fold32's binary counter). The parts add in order; the sum rounds once. Every output
// has gemv's bits for the same row, whatever the chunk, the tile, RUN and STAGES.
template <int F>
__global__ void stage_kernel(Matrix w, int n0, int rows, int8_t* codes, float* scales) {
    // One thread a sub-block: (row r, sub-block j), j fastest.
    const int J = w.K / 32;
    const int64_t s = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(s >= int64_t(rows) * J) return;
    const int r = int(s / J), j = int(s % J);
    constexpr int per_block = gg::block_values(F) / 32;
    Decoded<F> sb;
    sb.decode(w.blocks + int64_t(n0 + r) * w.row_bytes + int64_t(j / per_block) * gg::block_bytes(F), j % per_block);
    int4* dst = reinterpret_cast<int4*>(codes + (int64_t(j) * rows + r) * 32);
    dst[0] = make_int4(sb.q[0], sb.q[1], sb.q[2], sb.q[3]);
    dst[1] = make_int4(sb.q[4], sb.q[5], sb.q[6], sb.q[7]);
    scales[int64_t(j) * rows + r] = sb.s0;
    if constexpr(F == gg::Q2_K) {   // d, dmin, the minimums (mn0 | mn1 << 8 as the word's bits)
        scales[(int64_t(J) + j) * rows + r] = sb.s1;
        scales[(2 * int64_t(J) + j) * rows + r] = __int_as_float(sb.mn0 | (sb.mn1 << 8));
    } else {
        if constexpr(gg::two_scales(F)) scales[(int64_t(J) + j) * rows + r] = sb.s1;
        if constexpr(gg::has_min(F)) scales[(int64_t(J) + j) * rows + r] = sb.m;
    }
}

// The arithmetic of a format's terms: one scale, two scales (s0 for values 0..15, s1 for 16..31), a scale and a
// minimum, or Q2_K's integer form (scaled codes, d, dmin and the 4-bit minimums). The planes of scales: s0, then s1 or
// m; Q2_K's d, dmin, mn0 | mn1 << 8.
enum ScaleKind : int { kOneScale = 0, kTwoScales = 1, kMinimum = 2, kScaledCodes = 3 };
__host__ __device__ constexpr int scale_kind(int type) {
    return type == gg::Q2_K ? kScaledCodes : gg::two_scales(type) ? kTwoScales : gg::has_min(type) ? kMinimum
                                                                                                  : kOneScale;
}
__host__ __device__ constexpr int scale_planes(int kind) { return kind == kOneScale ? 1 : kind == kScaledCodes ? 3 : 2; }

__host__ __device__ constexpr int align16(int x) { return (x + 15) / 16 * 16; }

// A gemm CTA's shared memory (bytes), STAGES slots of each: the weight rows' codes and scales, the tokens' codes and
// their d and sums.
template <int SC, int WN, int WT, int TB, int RUN, int STAGES>
struct GemmSmem {
    static constexpr int BN = 16 * WN, BT = 8 * TB * WT, planes = scale_planes(SC);
    static constexpr int codes = 0;
    static constexpr int scales = codes + STAGES * RUN * BN * 32;
    static constexpr int tokens = align16(scales + STAGES * planes * RUN * BN * 4);
    static constexpr int token_scales = tokens + STAGES * RUN * BT * 32;
    static constexpr int bytes = token_scales + STAGES * 2 * RUN * BT * 4;
};

// Word w of row r's 32 codes in a tile: its two 16-byte halves swapped on every other group of four rows, so a
// fragment's eight rows read 32 different banks.
__device__ __forceinline__ int swizzled(int r, int w) { return ((((w >> 2) ^ (r >> 2)) & 1) << 2) | (w & 3); }

// Where a CTA is in the visiting order: part p, the tree's step at (lane lane_at32(at)), the stage's first sub-block
// j0 (the lane's sub-blocks j0, j0 + 32, ... up to RUN of them). A lane without sub-blocks in its part is one stage of
// none (its sum, +0, still folds in).
template <int RUN>
struct Visit {
    int p, at, j0, start, end;
    __device__ __forceinline__ void begin(int J, int parts) {
        p = at = start = j0 = 0;
        end = int(int64_t(J) / parts);
    }
    __device__ __forceinline__ bool valid(int parts) const { return p < parts; }
    __device__ __forceinline__ int count() const { return j0 < end ? min(RUN, (end - j0 + 31) / 32) : 0; }
    __device__ __forceinline__ bool last() const { return j0 + 32 * RUN >= end; }
    __device__ __forceinline__ void next(int J, int parts) {
        if(!last()) {
            j0 += 32 * RUN;
            return;
        }
        if(++at == 32) {
            at = 0;
            ++p;
            start = int(int64_t(J) * p / parts);
            end = int(int64_t(J) * (p + 1) / parts);
        }
        j0 = start + crit::kcore::lane_at32(at);
    }
};

template <int SC, int WN, int WT, int TB, int RUN, int STAGES>
__global__ void __launch_bounds__(32 * WN * WT, 1) gemm_kernel(Staged w, const int8_t* xq, const float* xd,
                                                               const int32_t* xs, int M, int parts, void* y, int kind,
                                                               int64_t ld) {
    using S = GemmSmem<SC, WN, WT, TB, RUN, STAGES>;
    constexpr int BN = S::BN, BT = S::BT, kThreads = 32 * WN * WT, planes = S::planes;
    extern __shared__ __align__(16) uint8_t smem[];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, gid = lane >> 2, tig = lane & 3;
    const int n0 = blockIdx.y * BN, t0 = blockIdx.x * BT;   // token tiles along x: CTAs beside each other share a weight tile
    const int wn = (warp / WT) * 16, wt = (warp % WT) * 8 * TB;
    const int J = w.K / 32;

    // A stage's codes, scales and tokens into slot `at` (one commit group; a finished visit's stage copies nothing).
    const auto load = [&](const Visit<RUN>& v, int at) {
        const int count = v.valid(parts) ? v.count() : 0;
        uint8_t* wq = smem + S::codes + at * RUN * BN * 32;
        for(int e = tid; e < count * BN * 2; e += kThreads) {
            const int u = e / (BN * 2), r = e / 2 % BN, half = e % 2;
            const int n = min(n0 + r, w.rows - 1), j = v.j0 + 32 * u;
            crit::kcore::cp_async16(wq + (u * BN + r) * 32 + 4 * swizzled(r, 4 * half),
                                    w.codes + (int64_t(j) * w.rows + n) * 32 + 16 * half);
        }
        float* ws = reinterpret_cast<float*>(smem + S::scales) + at * planes * RUN * BN;
        for(int e = tid; e < count * planes * BN; e += kThreads) {
            const int u = e / (planes * BN), plane = e / BN % planes, r = e % BN;
            const int n = min(n0 + r, w.rows - 1), j = v.j0 + 32 * u;
            crit::kcore::cp_async_ca<4>(ws + (plane * RUN + u) * BN + r, w.scales + (int64_t(plane) * J + j) * w.rows + n);
        }
        uint8_t* tq = smem + S::tokens + at * RUN * BT * 32;
        for(int e = tid; e < count * BT * 2; e += kThreads) {
            const int u = e / (BT * 2), r = e / 2 % BT, half = e % 2;
            const int t = min(t0 + r, M - 1), j = v.j0 + 32 * u;
            crit::kcore::cp_async16(tq + (u * BT + r) * 32 + 4 * swizzled(r, 4 * half),
                                    xq + (int64_t(j) * M + t) * 32 + 16 * half);
        }
        float* ts = reinterpret_cast<float*>(smem + S::token_scales) + at * 2 * RUN * BT;
        for(int e = tid; e < count * 2 * BT; e += kThreads) {
            const int u = e / (2 * BT), which = e / BT % 2, r = e % BT;
            const int t = min(t0 + r, M - 1), j = v.j0 + 32 * u;
            const void* src = which ? static_cast<const void*>(xs + int64_t(j) * M + t)
                                    : static_cast<const void*>(xd + int64_t(j) * M + t);
            crit::kcore::cp_async_ca<4>(ts + (which * RUN + u) * BT + r, src);
        }
        crit::kcore::cp_async_commit();
    };

    float out[TB][4], run[TB][4], tree[TB][4][5];
#pragma unroll
    for(int b = 0; b < TB; ++b)
#pragma unroll
        for(int i = 0; i < 4; ++i) out[b][i] = run[b][i] = 0.0f;

    // A stage's MMAs and terms from slot `at`, then, at a lane's end, its fold into the tree.
    const auto multiply = [&](const Visit<RUN>& v, int at) {
        const int count = v.count();
        const uint32_t* wq = reinterpret_cast<const uint32_t*>(smem + S::codes + at * RUN * BN * 32);
        const float* ws = reinterpret_cast<const float*>(smem + S::scales) + at * planes * RUN * BN;
        const uint32_t* tq = reinterpret_cast<const uint32_t*>(smem + S::tokens + at * RUN * BT * 32);
        const float* ts = reinterpret_cast<const float*>(smem + S::token_scales) + at * 2 * RUN * BT;
#pragma unroll
        for(int u = 0; u < RUN; ++u) {   // in increasing j: the lane's order
            if(u >= count) break;
            const int ra = wn + gid, rb = ra + 8;
            const uint32_t* arow0 = wq + (u * BN + ra) * 8;
            const uint32_t* arow1 = wq + (u * BN + rb) * 8;
            const uint32_t a0 = arow0[swizzled(ra, tig)], a1 = arow1[swizzled(rb, tig)];
            const uint32_t a2 = arow0[swizzled(ra, 4 + tig)], a3 = arow1[swizzled(rb, 4 + tig)];
            const float s0a = ws[u * BN + ra], s0b = ws[u * BN + rb];
            float s1a = 0.0f, s1b = 0.0f;   // s1 or m
            if constexpr(SC != kOneScale) s1a = ws[(RUN + u) * BN + ra], s1b = ws[(RUN + u) * BN + rb];
            // Q2_K: the minimums as an A tile (each row's mn0 over k 0..15, mn1 over 16..31).
            uint32_t n0a = 0, n0b = 0, n1a = 0, n1b = 0;
            if constexpr(SC == kScaledCodes) {
                const uint32_t pa = __float_as_uint(ws[(2 * RUN + u) * BN + ra]);
                const uint32_t pb = __float_as_uint(ws[(2 * RUN + u) * BN + rb]);
                n0a = (pa & 0xffu) * 0x01010101u, n1a = (pa >> 8) * 0x01010101u;
                n0b = (pb & 0xffu) * 0x01010101u, n1b = (pb >> 8) * 0x01010101u;
            }
#pragma unroll
            for(int b = 0; b < TB; ++b) {
                const int rt = wt + 8 * b + gid;
                const uint32_t* brow = tq + (u * BT + rt) * 8;
                const uint32_t b0 = brow[swizzled(rt, tig)], b1 = brow[swizzled(rt, 4 + tig)];
                int lo[4] = {0, 0, 0, 0}, hi[4] = {0, 0, 0, 0};
                if constexpr(SC == kTwoScales) {
                    crit::kcore::mma_s8_m16n8k16(lo, a0, a1, b0);
                    crit::kcore::mma_s8_m16n8k16(hi, a2, a3, b1);
                } else if constexpr(SC == kScaledCodes) {   // lo: the scaled codes' dot; hi: the minimums'
                    crit::kcore::mma_s8_m16n8k32(lo, a0, a1, a2, a3, b0, b1);
                    crit::kcore::mma_s8_m16n8k32(hi, n0a, n0b, n1a, n1b, b0, b1);
                } else {
                    crit::kcore::mma_s8_m16n8k32(lo, a0, a1, a2, a3, b0, b1);
                }
                const int c = wt + 8 * b + 2 * tig;
                const float2 d = *reinterpret_cast<const float2*>(ts + u * BT + c);
#pragma unroll
                for(int i = 0; i < 4; ++i) {
                    const float di = (i & 1) ? d.y : d.x;
                    const float s0 = i < 2 ? s0a : s0b, s1 = i < 2 ? s1a : s1b;
                    float term;
                    if constexpr(SC == kScaledCodes) {   // s0 d, s1 dmin
                        term = __fmul_rn(di, __fsub_rn(__fmul_rn(s0, float(lo[i])), __fmul_rn(s1, float(hi[i]))));
                    } else if constexpr(SC == kTwoScales) {
                        term = __fadd_rn(__fmul_rn(__fmul_rn(s0, di), float(lo[i])),
                                         __fmul_rn(__fmul_rn(s1, di), float(hi[i])));
                    } else {
                        term = __fmul_rn(__fmul_rn(s0, di), float(lo[i]));
                    }
                    if constexpr(SC == kMinimum) {
                        const int sum = reinterpret_cast<const int32_t*>(ts)[(RUN + u) * BT + c + (i & 1)];
                        term = __fsub_rn(term, __fmul_rn(__fmul_rn(s1, di), float(sum)));
                    }
                    run[b][i] = __fadd_rn(run[b][i], term);
                }
            }
        }
        if(!v.last()) return;
        // The lane is done: fold32's binary counter over the tree's step (its trailing ones add the partial sums of
        // those levels, the first zero keeps the sum there; the total after step 31), each level a constant so the
        // partial sums stay in registers.
        const int ones = __ffs(~v.at) - 1;   // 0 .. 5
#pragma unroll
        for(int b = 0; b < TB; ++b)
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                float x = run[b][i];
#pragma unroll
                for(int level = 0; level < 5; ++level)
                    if(level < ones) x = __fadd_rn(tree[b][i][level], x);
#pragma unroll
                for(int level = 0; level < 5; ++level)
                    if(level == ones) tree[b][i][level] = x;
                if(ones == 5) out[b][i] = v.p == 0 ? x : __fadd_rn(out[b][i], x);
                run[b][i] = 0.0f;
            }
    };

    Visit<RUN> loads, multiplies;
    loads.begin(J, parts);
    multiplies.begin(J, parts);
    for(int s = 0; s < STAGES - 1; ++s) {
        load(loads, s);
        loads.next(J, parts);
    }
    for(int s = 0; multiplies.valid(parts); ++s) {
        crit::kcore::cp_async_wait<STAGES - 2>();   // stage s's copies landed
        __syncthreads();
        load(loads, (s + STAGES - 1) % STAGES);      // the slot stage s - 1 was multiplied from
        loads.next(J, parts);
        multiply(multiplies, s % STAGES);
        multiplies.next(J, parts);
    }
    crit::kcore::cp_async_wait<0>();
#pragma unroll
    for(int b = 0; b < TB; ++b)
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const int n = n0 + wn + gid + 8 * (i >> 1), t = t0 + wt + 8 * b + 2 * tig + (i & 1);
            if(n < w.rows && t < M) store(y, kind, int64_t(t) * ld + n, out[b][i]);
        }
}

__global__ void sum_kernel(const float* yp, int parts, int M, int N, void* y, int kind, int64_t ld) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= int64_t(M) * N) return;
    const int64_t r = i / N, n = i % N;
    float v = yp[i];
    for(int p = 1; p < parts; ++p) v = __fadd_rn(v, yp[int64_t(p) * M * N + i]);
    store(y, kind, r * ld + n, v);
}

template <int F>
__global__ void dequantize_kernel(Matrix w, int expert, void* out, int kind) {
    // One thread a sub-block: its 32 values.
    const int64_t s = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int J = w.K / 32;
    if(s >= int64_t(w.N) * J) return;
    const int64_t n = s / J;
    const int j = int(s % J);
    constexpr int per_block = gg::block_values(F) / 32;
    const uint8_t* row = w.blocks + int64_t(expert) * w.expert_stride + n * w.row_bytes;
    int32_t q[8];
    float s0, s1, m, m1;
    gg::subblock<F>(row + int64_t(j / per_block) * gg::block_bytes(F), j % per_block, q, s0, s1, m, m1);
#pragma unroll
    for(int e = 0; e < 32; ++e) store(out, kind, n * w.K + 32 * j + e, gg::value(q, e, s0, s1, m, m1));
}

template <int F, int M>
cudaError_t launch_rows(const Matrix& w, const int8_t* xq, const float* xd, const int32_t* xs, int rows,
                        const int32_t* ids, float* yp, int parts, int warps, cudaStream_t stream) {
    const dim3 grid((w.N + warps - 1) / warps, parts);
    gemv_kernel<F, M><<<grid, 32 * warps, 0, stream>>>(w, xq, xd, xs, ids, rows, yp, parts);
    return cudaGetLastError();
}

template <int F>
cudaError_t launch_format(const Matrix& w, const int8_t* xq, const float* xd, const int32_t* xs, int M,
                          const int32_t* ids, float* yp, int parts, int warps, cudaStream_t stream) {
    if(M <= 1) return launch_rows<F, 1>(w, xq, xd, xs, M, ids, yp, parts, warps, stream);
    if(M <= 2) return launch_rows<F, 2>(w, xq, xd, xs, M, ids, yp, parts, warps, stream);
    if(M <= 4) return launch_rows<F, 4>(w, xq, xd, xs, M, ids, yp, parts, warps, stream);
    return launch_rows<F, 8>(w, xq, xd, xs, M, ids, yp, parts, warps, stream);
}

template <int SC>
cudaError_t launch_gemm(const Staged& w, int key_n, const int8_t* xq, const float* xd, const int32_t* xs, int M,
                        int parts, void* y, int kind, int64_t ld, cudaStream_t stream) {
    switch(crit_lookup_gemm(key_n, w.K, M)) {
#define CRIT_GGUF_GEMM(index, wn, wt, tb, run, stages)                                                               \
    case index: {                                                                                                    \
        constexpr int bytes = GemmSmem<SC, wn, wt, tb, run, stages>::bytes;                                          \
        const auto kernel = gemm_kernel<SC, wn, wt, tb, run, stages>;                                                \
        const cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);      \
        if(e != cudaSuccess) return e;                                                                               \
        const dim3 grid((M + 8 * tb * wt - 1) / (8 * tb * wt), (w.rows + 16 * wn - 1) / (16 * wn));                  \
        kernel<<<grid, 32 * wn * wt, bytes, stream>>>(w, xq, xd, xs, M, parts, y, kind, ld);                         \
        return cudaGetLastError();                                                                                   \
    }
        CRIT_CONFIGS_gemm(CRIT_GGUF_GEMM)
#undef CRIT_GGUF_GEMM
        default:
            return cudaErrorInvalidConfiguration;
    }
}

bool valid(const Matrix& w) {
    return gg::supported(w.type) && w.K > 0 && w.N > 0 && w.K % gg::block_values(w.type) == 0 &&
           w.row_bytes == int64_t(w.K / gg::block_values(w.type)) * gg::block_bytes(w.type);
}
}  // namespace

cudaError_t enqueue_quantize(const void* x, int kind, int64_t ld, int M, int K, int8_t* xq, float* xd, int32_t* xs,
                             bool by_subblock, cudaStream_t stream) {
    if(M < 1 || K % 32 || M > 65535) return cudaErrorInvalidValue;
    constexpr int warps = 8;
    quantize_kernel<<<dim3((K / 32 + warps - 1) / warps, M), 32 * warps, 0, stream>>>(x, kind, ld, M, K, by_subblock,
                                                                                        xq, xd, xs);
    return cudaGetLastError();
}

int splits(const Matrix& w, int multiprocessors) {
    int knob = 0;
    switch(crit_lookup_gemv(w.N, w.K, 1)) {
#define CRIT_GGUF_PLAN(index, warps, split_knob) \
    case index:                                  \
        knob = split_knob;                       \
        break;
        CRIT_CONFIGS_gemv(CRIT_GGUF_PLAN)
#undef CRIT_GGUF_PLAN
        default:
            break;
    }
    const int J = w.K / 32;
    if(knob > 0) return knob < J ? knob : J;
    // 0: enough warps for 32 a multiprocessor, each part at least 32 sub-blocks.
    int parts = 1;
    while(int64_t(w.N) * parts < 32LL * multiprocessors && J / (2 * parts) >= 32) parts *= 2;
    return parts;
}

cudaError_t enqueue_gemv(const Matrix& w, const int8_t* xq, const float* xd, const int32_t* xs, int M,
                         const int32_t* ids, float* yp, int parts, cudaStream_t stream) {
    if(!valid(w) || M < 1 || M > kMaxRows || parts < 1 || parts > 65535) return cudaErrorInvalidValue;
    int warps = 4;
    switch(crit_lookup_gemv(w.N, w.K, M)) {
#define CRIT_GGUF_CASE(index, warp_knob, split_knob) \
    case index:                                      \
        warps = warp_knob;                           \
        break;
        CRIT_CONFIGS_gemv(CRIT_GGUF_CASE)
#undef CRIT_GGUF_CASE
        default:
            break;
    }
    switch(w.type) {
#define CRIT_GGUF_GEMV(F) \
    case gg::F: return launch_format<gg::F>(w, xq, xd, xs, M, ids, yp, parts, warps, stream);
        CRIT_GGUF_FORMATS(CRIT_GGUF_GEMV)
#undef CRIT_GGUF_GEMV
        default: return cudaErrorInvalidValue;
    }
}

cudaError_t enqueue_stage(const Matrix& w, int n0, int rows, int8_t* codes, float* scales, cudaStream_t stream) {
    if(!valid(w) || n0 < 0 || rows < 1 || n0 + rows > w.N) return cudaErrorInvalidValue;
    const int64_t count = int64_t(rows) * (w.K / 32);
    const unsigned grid = static_cast<unsigned>((count + 127) / 128);
    switch(w.type) {
#define CRIT_GGUF_STAGE(F) \
    case gg::F: stage_kernel<gg::F><<<grid, 128, 0, stream>>>(w, n0, rows, codes, scales); break;
        CRIT_GGUF_FORMATS(CRIT_GGUF_STAGE)
#undef CRIT_GGUF_STAGE
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

cudaError_t enqueue_gemm(const Staged& w, int key_n, const int8_t* xq, const float* xd, const int32_t* xs, int M,
                         int parts, void* y, int kind, int64_t ld, cudaStream_t stream) {
    if(!gg::supported(w.type) || w.rows < 1 || w.K < 32 || w.K % 32 || M < 1 || parts < 1 ||
       (w.rows + 15) / 16 > 65535)
        return cudaErrorInvalidValue;
    switch(scale_kind(w.type)) {
        case kOneScale: return launch_gemm<kOneScale>(w, key_n, xq, xd, xs, M, parts, y, kind, ld, stream);
        case kTwoScales: return launch_gemm<kTwoScales>(w, key_n, xq, xd, xs, M, parts, y, kind, ld, stream);
        case kScaledCodes: return launch_gemm<kScaledCodes>(w, key_n, xq, xd, xs, M, parts, y, kind, ld, stream);
        default: return launch_gemm<kMinimum>(w, key_n, xq, xd, xs, M, parts, y, kind, ld, stream);
    }
}

cudaError_t enqueue_sum(const float* yp, int parts, int M, int N, void* y, int kind, int64_t ld, cudaStream_t stream) {
    const int64_t count = int64_t(M) * N;
    sum_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(yp, parts, M, N, y, kind, ld);
    return cudaGetLastError();
}

cudaError_t enqueue_dequantize(const Matrix& w, int expert, void* out, int kind, cudaStream_t stream) {
    if(!valid(w)) return cudaErrorInvalidValue;
    const int64_t count = int64_t(w.N) * (w.K / 32);
    const unsigned grid = static_cast<unsigned>((count + 127) / 128);
    switch(w.type) {
#define CRIT_GGUF_DEQUANTIZE(F) \
    case gg::F: dequantize_kernel<gg::F><<<grid, 128, 0, stream>>>(w, expert, out, kind); break;
        CRIT_GGUF_FORMATS(CRIT_GGUF_DEQUANTIZE)
#undef CRIT_GGUF_DEQUANTIZE
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

}  // namespace crit::techniques::gguf_rows
