// crit's kernel core: chunked-scan blocks (kernels/core/README.md, "scan"). A chunked scan of a linear recurrence (the
// WY / UT-transform form of the gated delta rule, a state-space model's chunks) splits a prompt into chunks: a parallel
// pass builds each chunk's state-independent matrices, a sequential pass carries the state across the chunks with matrix
// products on tensor cores. These are the pieces such scans share: the unit lower triangular inverse a column a thread,
// the reuse of MMA accumulators as the next product's A operand, and a warp's transposed state tile kept in accumulators.
// The products' rounding is the caller's (FP16 operands, FP32 accumulation here); each block has one order.
#pragma once
#include "crit_kcore/mma.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace crit::kcore::scan {

// Column j of T = (I + A)^-1 for a strictly lower triangular N x N matrix A (row-major at A, rows `stride` floats
// apart, 16-byte aligned, entries on and above the diagonal zero), by forward substitution in FP32:
//   t_i = [i == j] - sum_{m < i} A_im t_m,   summed in increasing m.
// Every thread runs the same instructions (t_m = 0 for m < j), so a warp's threads read A's rows as broadcasts.
template <int N>
__device__ __forceinline__ void unit_lower_inverse_column(const float* A, int stride, int j, float (&t)[N]) {
    static_assert(N % 4 == 0, "rows read 4 at a time");
#pragma unroll
    for(int i = 0; i < N; ++i) t[i] = 0.f;
#pragma unroll
    for(int i = 0; i < N; ++i) {
        float acc = i == j ? 1.f : 0.f;
        const float* row = A + i * stride;
#pragma unroll
        for(int m = 0; m < (i & ~3); m += 4) {
            const float4 a = *reinterpret_cast<const float4*>(row + m);
            acc = fmaf(-a.x, t[m], acc);
            acc = fmaf(-a.y, t[m + 1], acc);
            acc = fmaf(-a.z, t[m + 2], acc);
            acc = fmaf(-a.w, t[m + 3], acc);
        }
#pragma unroll
        for(int m = i & ~3; m < i; ++m) acc = fmaf(-row[m], t[m], acc);
        t[i] = acc;
    }
}

// Two FP32 values as an FP16 pair (lo in the low half), each rounded to nearest.
__device__ __forceinline__ uint32_t f16x2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Two adjacent m16 x n8 FP32 accumulator tiles (columns 0..7 in lo, 8..15 in hi; mma.cuh's C layout) as the FP16 A
// operand of an m16n8k16 product over those 16 columns: the C layout of a lane's pair of columns is the A layout of its
// pair of k, so no data moves between lanes (the reuse flash attention makes of its score tile).
__device__ __forceinline__ void f16_a_of_accumulators(uint32_t (&a)[4], const float (&lo)[4], const float (&hi)[4]) {
    a[0] = f16x2(lo[0], lo[1]);
    a[1] = f16x2(lo[2], lo[3]);
    a[2] = f16x2(hi[0], hi[1]);
    a[3] = f16x2(hi[2], hi[3]);
}

// A warp's slice of a recurrent state S [DK keys][dv values] (row-major in memory), kept transposed, S^T [16 values]
// [DK keys], in FP32 MMA accumulators: s[kt] is the m16 x n8 tile of keys 8 kt .. 8 kt + 7, lane (gid, tig) holding
// values v0 + gid (+ 8) and keys 8 kt + 2 tig (+ 1). The same registers are
//   - the A operand of S^T X (k = keys, a_operand(ks) for keys 16 ks .. 16 ks + 15, rounded to FP16), and
//   - the accumulator of an update S^T += D^T Y (rows values, columns keys).
// Memory access: for each key, lanes of one tig read 8 consecutive values (32-byte sectors used whole).
template <int DK>
struct TransposedState {
    static_assert(DK % 16 == 0, "keys in 16-key steps");
    float s[DK / 8][4];

    __device__ __forceinline__ static int key(int lane, int kt, int i) { return 8 * kt + mma_c_col(lane, i); }
    __device__ __forceinline__ static int value(int lane, int i) { return mma_c_row(lane, i); }

    // From head [DK][dv] at values v0 .. v0 + 15 (zeros when !live).
    __device__ __forceinline__ void load(const float* head, int dv, int v0, int lane, bool live) {
#pragma unroll
        for(int kt = 0; kt < DK / 8; ++kt)
#pragma unroll
            for(int i = 0; i < 4; ++i)
                s[kt][i] = live ? head[std::size_t(key(lane, kt, i)) * dv + v0 + value(lane, i)] : 0.f;
    }
    __device__ __forceinline__ void store(float* head, int dv, int v0, int lane) const {
#pragma unroll
        for(int kt = 0; kt < DK / 8; ++kt)
#pragma unroll
            for(int i = 0; i < 4; ++i) head[std::size_t(key(lane, kt, i)) * dv + v0 + value(lane, i)] = s[kt][i];
    }
    __device__ __forceinline__ void scale(float e) {
#pragma unroll
        for(int kt = 0; kt < DK / 8; ++kt)
#pragma unroll
            for(int i = 0; i < 4; ++i) s[kt][i] *= e;
    }
    // The FP16 A operand over keys 16 ks .. 16 ks + 15.
    __device__ __forceinline__ void a_operand(int ks, uint32_t (&a)[4]) const {
        f16_a_of_accumulators(a, s[2 * ks], s[2 * ks + 1]);
    }
};

}  // namespace crit::kcore::scan
