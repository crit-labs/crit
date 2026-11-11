// crit's kernel core: column-scan blocks (kernels/core/README.md, "scan"). A column scan runs a linear recurrence token
// after token with its state in registers, the same expression for every column, so a token decoded alone equals the
// same token inside a verify window and a row's bits never depend on the rows beside it. These are the pieces such a
// scan shares: the decay of one token rounded once, the one update expression, and the warp's fixed-order reductions of
// a lane's partial dot products. Each has one order, fixed by its shape (crit's fixed-order rule, kernels/TEMPLATES.md).
#pragma once
#include <cuda_runtime.h>

namespace crit::kcore::scan {

// A token's decay e^x: FP64 exp rounded once to FP32, the correctly rounded value unless e^x lies within an FP64 ulp of
// a rounding boundary. expf is not correctly rounded and errs upward more often than downward, so a decay near 1
// applied token after token drifts a state; a scan computes each token's decay once and every column uses that value.
__device__ __forceinline__ float decay_exact(double x) { return __double2float_rn(exp(x)); }

// The gated delta rule's update of one state value: e s + k delta, one fused multiply-add after the decay's product.
// Every path that advances a state (decode, verify, a stashed step's replay, a recompute) applies this expression.
__device__ __forceinline__ float delta_update(float s, float e, float k, float delta) { return fmaf(k, delta, s * e); }

// The 8 partial sums of a lane (a warp's 4 columns' k . S and q . S, say) summed over the warp and broadcast, each with
// the bits of the xor butterfly over 32 lanes (offsets 16, 8, 4, 2, 1; IEEE addition is commutative, so only the pairs
// matter): lanes exchange halves of their values at distances 16, 8 and 4 (a reduce-scatter: lanes 4 i .. 4 i + 3 end
// with value i), sum the partials at distances 2 and 1, and value i is broadcast from lane 4 i. 17 shuffles, not 8 x 5.
__device__ __forceinline__ void column_sum8(float (&x)[8]) {
    const int lane = threadIdx.x & 31;
    float a[4], b[2], c;
    {   // distance 16: a lane keeps values 0..3 (lane bit 4 clear) or 4..7 (set) and sends the others
        const bool hi = lane & 16;
#pragma unroll
        for(int i = 0; i < 4; ++i) a[i] = (hi ? x[i + 4] : x[i]) + __shfl_xor_sync(0xffffffffu, hi ? x[i] : x[i + 4], 16);
    }
    {   // distance 8
        const bool hi = lane & 8;
#pragma unroll
        for(int i = 0; i < 2; ++i) b[i] = (hi ? a[i + 2] : a[i]) + __shfl_xor_sync(0xffffffffu, hi ? a[i] : a[i + 2], 8);
    }
    {   // distance 4
        const bool hi = lane & 4;
        c = (hi ? b[1] : b[0]) + __shfl_xor_sync(0xffffffffu, hi ? b[0] : b[1], 4);
    }
    c += __shfl_xor_sync(0xffffffffu, c, 2);
    c += __shfl_xor_sync(0xffffffffu, c, 1);
#pragma unroll
    for(int i = 0; i < 8; ++i) x[i] = __shfl_sync(0xffffffffu, c, 4 * i);
}

// The same scheme for 4 values (|k|^2, |q|^2, q . k and a spare): distances 16 and 8 scatter, then 4, 2, 1; value i
// broadcast from lane 8 i. The bits of the 32-lane butterfly on each value.
__device__ __forceinline__ void column_sum4(float (&x)[4]) {
    const int lane = threadIdx.x & 31;
    float a[2], c;
    {
        const bool hi = lane & 16;
#pragma unroll
        for(int i = 0; i < 2; ++i) a[i] = (hi ? x[i + 2] : x[i]) + __shfl_xor_sync(0xffffffffu, hi ? x[i] : x[i + 2], 16);
    }
    {
        const bool hi = lane & 8;
        c = (hi ? a[1] : a[0]) + __shfl_xor_sync(0xffffffffu, hi ? a[0] : a[1], 8);
    }
    c += __shfl_xor_sync(0xffffffffu, c, 4);
    c += __shfl_xor_sync(0xffffffffu, c, 2);
    c += __shfl_xor_sync(0xffffffffu, c, 1);
#pragma unroll
    for(int i = 0; i < 4; ++i) x[i] = __shfl_sync(0xffffffffu, c, 8 * i);
}

// The same reduce-scatter for any N of 2 to 32 partial sums (a power of two): distances 16, 8, .. scatter while a lane
// still holds more than one value, the rest sum, and value i is broadcast from lane (32 / N) i. Every value has the bits
// of the 32-lane butterfly, as in column_sum8 and column_sum4 (N 8 and 4 give their bits), so a scan holding 4, 8 or
// 16 columns a lane computes the same sums.
namespace detail {
template <int N, int D>
__device__ __forceinline__ void column_scatter(float* c, int lane) {
    if constexpr(D >= 1) {
        if constexpr(N > 1) {
            const bool hi = lane & D;
#pragma unroll
            for(int i = 0; i < N / 2; ++i)
                c[i] = (hi ? c[i + N / 2] : c[i]) + __shfl_xor_sync(0xffffffffu, hi ? c[i] : c[i + N / 2], D);
            column_scatter<N / 2, D / 2>(c, lane);
        } else {
            c[0] += __shfl_xor_sync(0xffffffffu, c[0], D);
            column_scatter<1, D / 2>(c, lane);
        }
    }
}
}  // namespace detail

template <int N>
__device__ __forceinline__ void column_sum(float (&x)[N]) {
    static_assert(N >= 2 && N <= 32 && (N & (N - 1)) == 0, "2 to 32 values, a power of two");
    const int lane = threadIdx.x & 31;
    float c[N];
#pragma unroll
    for(int i = 0; i < N; ++i) c[i] = x[i];
    detail::column_scatter<N, 16>(c, lane);
#pragma unroll
    for(int i = 0; i < N; ++i) x[i] = __shfl_sync(0xffffffffu, c[0], (32 / N) * i);
}

}  // namespace crit::kcore::scan
