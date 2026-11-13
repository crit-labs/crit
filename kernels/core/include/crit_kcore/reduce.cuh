// crit's kernel core: reductions (kernels/core/README.md, "reduce"). Warp and block sums, maxima and scans, the
// selection keys and argmax, crit's fixed-order trees, and the online softmax's merge. Every reduction here has one
// order, fixed by its shape (lane count, tree), never by the data or by how many rows a call holds: a sum computed in
// it has the same bits wherever it runs (crit's fixed-order rule, kernels/TEMPLATES.md).
#pragma once
#include "crit_kcore/arch.cuh"

#include <climits>
#include <cstdint>

namespace crit::kcore {

// ------------------------------------------------------------------------------------------- warp butterflies
// The xor butterfly over WIDTH lanes (a power of two <= 32; lanes in aligned groups of WIDTH): offsets WIDTH / 2 .. 1,
// every lane of a group ends with the same value. The order is the butterfly's, the same for every call.

template <int WIDTH = 32>
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for(int o = WIDTH / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
template <int WIDTH = 32>
__device__ __forceinline__ int warp_sum(int v) {
#pragma unroll
    for(int o = WIDTH / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
template <int WIDTH = 32>
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for(int o = WIDTH / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
template <int WIDTH = 32>
__device__ __forceinline__ float warp_min(float v) {
#pragma unroll
    for(int o = WIDTH / 2; o > 0; o >>= 1) v = fminf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// The butterfly over the offsets [FROM, 32) only (FROM a power of two): lanes that differ in those bits combined, e.g.
// FROM 4 combines the 8 lanes of an MMA fragment's column (same tig) across gid.
template <int FROM>
__device__ __forceinline__ float warp_max_from(float v) {
#pragma unroll
    for(int o = FROM; o < 32; o <<= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// ------------------------------------------------------------------------------------------- halving trees

// The mask of the calling lane's half-warp (lanes 0..15 or 16..31).
__device__ __forceinline__ unsigned half_warp_mask() { return 0xffffu << (threadIdx.x % 32u & 16u); }

// The halving tree to the group's first lane by shuffle-down over WIDTH lanes (16: a half-warp, mask
// half_warp_mask()): v + v[+8], then + [+4], [+2], [+1], each add rounded (__fadd_rn: never contracted). Lane 0 of
// the group holds the sum ((l0 + l8) + (l4 + l12)) + ((l2 + l10) + (l6 + l14)) + (odd lanes alike).
template <int WIDTH = 16>
__device__ __forceinline__ float tree_sum_down(float v, unsigned mask) {
#pragma unroll
    for(int w = WIDTH / 2; w >= 1; w /= 2) v = __fadd_rn(v, __shfl_down_sync(mask, v, w, WIDTH));
    return v;
}

// The same 16-lane halving tree computed by one thread from lanes' finished sums as they come: kLaneOrder16[s] is
// the lane the s-th call of fold16 adds, and x[0..3] the tree's partial sums (x[0] the total after step 15). The bits
// equal tree_sum_down<16>'s (IEEE addition is commutative: only the tree's shape matters).
__device__ constexpr int kLaneOrder16[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
__device__ __forceinline__ void fold16(int step, float a, float* x) {
    switch(step) {
        case 0: x[0] = a; break;
        case 1: x[0] = __fadd_rn(x[0], a); break;
        case 2: x[1] = a; break;
        case 3: x[1] = __fadd_rn(x[1], a); x[0] = __fadd_rn(x[0], x[1]); break;
        case 4: x[1] = a; break;
        case 5: x[1] = __fadd_rn(x[1], a); break;
        case 6: x[2] = a; break;
        case 7: x[2] = __fadd_rn(x[2], a); x[1] = __fadd_rn(x[1], x[2]); x[0] = __fadd_rn(x[0], x[1]); break;
        case 8: x[1] = a; break;
        case 9: x[1] = __fadd_rn(x[1], a); break;
        case 10: x[2] = a; break;
        case 11: x[2] = __fadd_rn(x[2], a); x[1] = __fadd_rn(x[1], x[2]); break;
        case 12: x[2] = a; break;
        case 13: x[2] = __fadd_rn(x[2], a); break;
        case 14: x[3] = a; break;
        case 15: x[3] = __fadd_rn(x[3], a); x[2] = __fadd_rn(x[2], x[3]); x[1] = __fadd_rn(x[1], x[2]);
                 x[0] = __fadd_rn(x[0], x[1]); break;
    }
}

// The 32-lane halving tree (a warp's xor butterfly over 16, 8, 4, 2, 1, or shuffle-down from 16: lane 0 holds
// ((((l0 + l16) + (l8 + l24)) + ((l4 + l20) + (l12 + l28))) + ...) computed by one thread from finished lanes as they
// come: step s adds lane lane_at32(s) (a 5-bit reversal: 0, 16, 8, 24, 4, ...), x[0..5] the tree's partial sums (a
// binary counter: x[5] the total after step 31). Unroll the 32 steps so `step` is a constant: then only the partial
// sums still needed live. The bits equal the butterfly's (IEEE addition is commutative: only the tree's shape matters).
__host__ __device__ constexpr int lane_at32(int at) {
    return ((at & 1) << 4) | ((at & 2) << 2) | (at & 4) | ((at & 8) >> 2) | ((at & 16) >> 4);
}
__device__ __forceinline__ void fold32(int step, float a, float* x) {
    float v = a;
    int level = 0;
#pragma unroll
    for(int s = step; s & 1; s >>= 1) v = __fadd_rn(x[level++], v);
    x[level] = v;
}

// ------------------------------------------------------------------------------------------- block reductions

// The block's sum of v (blockDim.x a multiple of 32, every thread calls): each warp's butterfly, then warp 0 adds the
// warps' sums in warp order. `buffer` holds 32 floats of shared memory; every thread gets the sum.
__device__ __forceinline__ float block_sum(float v, float* buffer) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    v = warp_sum(v);
    if(lane == 0) buffer[warp] = v;
    __syncthreads();
    if(warp == 0) {
        float s = buffer[0];
        for(int w = 1; w < warps; ++w) s += buffer[w];
        if(lane == 0) buffer[32 - 1] = s;
    }
    __syncthreads();
    const float total = buffer[32 - 1];
    __syncthreads();
    return total;
}

// The block's inclusive scan of v over threadIdx.x (blockDim.x a multiple of 32, at most 1024), the sum or (MAX) the
// maximum; `buffer` holds 32 ints of shared memory.
template <bool MAX>
__device__ __forceinline__ int block_scan(int v, int* buffer) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
#pragma unroll
    for(int o = 1; o < 32; o <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, v, o);
        if(lane >= o) v = MAX ? max(v, other) : v + other;
    }
    if(lane == 31) buffer[warp] = v;
    __syncthreads();
    if(warp == 0) {
        int w = lane < warps ? buffer[lane] : 0;
#pragma unroll
        for(int o = 1; o < 32; o <<= 1) {
            const int other = __shfl_up_sync(0xffffffffu, w, o);
            if(lane >= o) w = MAX ? max(w, other) : w + other;
        }
        if(lane < warps) buffer[lane] = w;
    }
    __syncthreads();
    const int before = warp ? buffer[warp - 1] : 0;
    const int out = MAX ? max(v, before) : v + before;
    __syncthreads();
    return out;
}

// ------------------------------------------------------------------------------------------- selection

// The larger of two values, NaN if either is (torch's amax).
__device__ __forceinline__ float nanmax(float a, float b) { return (a > b || a != a) ? a : b; }

// Selection keys: a float as an int of the same order (-0 and +0 equal), NaN below every number (-inf included) and
// kPastKey below NaN (for lanes past the data).
constexpr int kNanKey = INT_MIN + 1, kPastKey = INT_MIN;
__device__ __forceinline__ int order_key(float c) {
    if(isnan(c)) return kNanKey;
    const int i = c == 0.0f ? 0 : __float_as_int(c);
    return i < 0 ? i ^ 0x7fffffff : i;
}

// The warp's largest (key, index): the larger key, then the lower index; every lane ends with the winner.
__device__ __forceinline__ void warp_argmax(int& key, int& index) {
#pragma unroll
    for(int o = 16; o > 0; o >>= 1) {
        const int ok = __shfl_xor_sync(0xffffffffu, key, o);
        const int oi = __shfl_xor_sync(0xffffffffu, index, o);
        if(ok > key || (ok == key && oi < index)) {
            key = ok;
            index = oi;
        }
    }
}

// ------------------------------------------------------------------------------------------- online softmax

// The merge of a partial softmax state (maximum m, sum l) into a running one (m_run, l_run), in a fixed order of
// partials: the new maximum, the new sum fma(l_run, keep, l take), and the factors the running output (keep) and the
// partial's output (take) are multiplied by. A state with no position seen keeps -inf and no NaN appears. expf: the
// math library's (within its ulp bound); merges are few, so the accurate exponential costs little.
struct SoftmaxMerge {
    float m, l, keep, take;
};
__device__ __forceinline__ SoftmaxMerge softmax_merge(float m_run, float l_run, float m, float l) {
    const float m_new = fmaxf(m_run, m);
    const float m_sub = m_new == -INFINITY ? 0.0f : m_new;
    const float keep = expf(m_run - m_sub), take = expf(m - m_sub);
    return SoftmaxMerge{m_new, fmaf(l_run, keep, l * take), keep, take};
}

}  // namespace crit::kcore
