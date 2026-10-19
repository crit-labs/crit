// Device side of host_nvfp4.h's declared order: lanes = L threads of one warp (16 or 32), lane j accumulating
// elements k with k % L == j in block order, explicit round-to-nearest intrinsics so nvcc cannot contract the
// product into the FMA, and a shuffle tree that pairs lanes exactly like the host's halving reduction. A device
// kernel that computes its rows this way gives the host's bits for the same expert.
#pragma once
#include <cstdint>

namespace crit::techniques::host_nvfp4 {

__device__ inline float e2m1_device(unsigned code) {
    const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float m = magnitude[code & 7u];
    return (code & 8u) ? -m : m;
}

// Called by L converged lanes (a whole warp, or one half for L = 16). scales: the row's cols / 16 decoded block scales.
// Returns the row's value on lane 0 (other lanes return partial sums).
template <int L>
__device__ inline float dot_row_halving(const std::uint8_t* codes, const float* scales, const float* x, int cols,
                                        float post) {
    static_assert(L == 16 || L == 32, "16 or 32 lanes");
    const unsigned lane = threadIdx.x % 32u;  // L = 16: two rows per warp, one per half
    float acc = 0.0f;
    for(int k = int(lane % unsigned(L)); k < cols; k += L) {
        const std::uint8_t byte = codes[k / 2];
        const float w = e2m1_device(k % 2 ? byte >> 4 : byte & 15u);
        acc = __fmaf_rn(__fmul_rn(w, scales[k / 16]), x[k], acc);
    }
    const unsigned group = L == 32 ? 0xffffffffu : (0xffffu << (lane & 16u));
    for(int w = L / 2; w >= 1; w /= 2) acc = __fadd_rn(acc, __shfl_down_sync(group, acc, w, L));
    return __fmul_rn(acc, post);
}

}  // namespace crit::techniques::host_nvfp4

namespace crit::techniques::host_nvfp4 {

// The block_dot order: lane j (of L) takes blocks j, j + L, ... in order; each block's dot runs sequentially over
// its 16 elements from 0 (unscaled codes), then lane = fma(dot, scale, lane); same shuffle tree and post. A lane
// xor butterfly (v += shfl_xor(v, L/2), ... 1) leaves lane 0 the same value, since each add is commutative.
template <int L>
__device__ inline float dot_row_block(const std::uint8_t* codes, const float* scales, const float* x, int cols,
                                      float post) {
    static_assert(L == 16 || L == 32, "16 or 32 lanes");
    const unsigned lane = threadIdx.x % 32u;
    float acc = 0.0f;
    for(int b = int(lane % unsigned(L)); b < cols / 16; b += L) {
        float d = 0.0f;
        for(int e = 0; e < 16; ++e) {
            const std::uint8_t byte = codes[(16 * b + e) / 2];
            d = __fmaf_rn(e2m1_device(e % 2 ? byte >> 4 : byte & 15u), x[16 * b + e], d);
        }
        acc = __fmaf_rn(d, scales[b], acc);
    }
    const unsigned group = L == 32 ? 0xffffffffu : (0xffffu << (lane & 16u));
    for(int w = L / 2; w >= 1; w /= 2) acc = __fadd_rn(acc, __shfl_down_sync(group, acc, w, L));
    return __fmul_rn(acc, post);
}

}  // namespace crit::techniques::host_nvfp4
