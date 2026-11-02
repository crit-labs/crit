// Device side of host_nvfp4.h's declared order, for CUDA and HIP from one source: lanes = L threads of one warp or
// wavefront (16 or 32), lane j accumulating elements k with k % L == j in block order, explicit round-to-nearest
// intrinsics so the compiler cannot contract the product into the FMA, and a shuffle tree that pairs lanes exactly
// like the host's halving reduction. A device kernel that computes its rows this way gives the host's bits for the
// same expert. The format functions take any host_nvfp4::Format layout (4- or 8-bit codes, 16 or 32 per block) and
// its element table in device memory; the NVFP4 ones decode E2M1 in registers. Metal has the same functions in
// host_nvfp4_device.metal.
#pragma once
#include <cstdint>

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
// HIP's shuffles take no member mask; `width` keeps a 16- or 32-lane group inside a 32- or 64-lane wavefront.
#define CRIT_HOST_NVFP4_SHFL_DOWN(mask, value, delta, width) __shfl_down((value), (delta), (width))
#else
#define CRIT_HOST_NVFP4_SHFL_DOWN(mask, value, delta, width) __shfl_down_sync((mask), (value), (delta), (width))
#endif

namespace crit::techniques::host_nvfp4 {

__device__ inline float e2m1_device(unsigned code) {
    const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float m = magnitude[code & 7u];
    return (code & 8u) ? -m : m;
}

namespace device_detail {
// Element k of a row: Bits 4 (two codes a byte, low nibble first) or 8, through `values` (null: E2M1).
template <int Bits>
__device__ inline float element(const std::uint8_t* codes, int k, const float* values) {
    if constexpr(Bits == 8) return values[codes[k]];
    const std::uint8_t byte = codes[k / 2];
    const unsigned code = k % 2 ? byte >> 4 : byte & 15u;
    return values ? values[code] : e2m1_device(code);
}
// The members of this lane's group of L in its warp (CUDA's shuffle mask; HIP ignores it).
template <int L>
__device__ inline unsigned group_mask() {
    const unsigned lane = threadIdx.x % 32u;
    return L == 32 ? 0xffffffffu : (0xffffu << (lane & 16u));
}
template <int L>
__device__ inline float halving(float acc) {
    const unsigned group = group_mask<L>();
    for(int w = L / 2; w >= 1; w /= 2) acc = __fadd_rn(acc, CRIT_HOST_NVFP4_SHFL_DOWN(group, acc, w, L));
    return acc;
}
}  // namespace device_detail

// scaled_elements in any format: called by L converged lanes (a whole group of L; L = 16 puts two rows in 32
// threads). scales: the row's cols / Block decoded block scales. Returns the row's value on the group's first lane.
template <int L, int Bits, int Block>
__device__ inline float dot_row_halving_format(const std::uint8_t* codes, const float* scales, const float* values,
                                               const float* x, int cols, float post) {
    static_assert(L == 16 || L == 32, "16 or 32 lanes");
    static_assert(Bits == 4 || Bits == 8, "4- or 8-bit codes");
    static_assert(Block == 16 || Block == 32, "16 or 32 elements per block scale");
    const unsigned lane = threadIdx.x % 32u;
    float acc = 0.0f;
    for(int k = int(lane % unsigned(L)); k < cols; k += L)
        acc = __fmaf_rn(__fmul_rn(device_detail::element<Bits>(codes, k, values), scales[k / Block]), x[k], acc);
    return __fmul_rn(device_detail::halving<L>(acc), post);
}

// block_dot in any format: lane j (of L) takes blocks j, j + L, ... in order; each block's dot runs sequentially
// over its Block elements from 0 (unscaled codes), then lane = fma(dot, scale, lane); same shuffle tree and post.
// A lane xor butterfly (v += shfl_xor(v, L/2), ... 1) leaves lane 0 the same value, since each add is commutative.
template <int L, int Bits, int Block>
__device__ inline float dot_row_block_format(const std::uint8_t* codes, const float* scales, const float* values,
                                             const float* x, int cols, float post) {
    static_assert(L == 16 || L == 32, "16 or 32 lanes");
    static_assert(Bits == 4 || Bits == 8, "4- or 8-bit codes");
    static_assert(Block == 16 || Block == 32, "16 or 32 elements per block scale");
    const unsigned lane = threadIdx.x % 32u;
    float acc = 0.0f;
    for(int b = int(lane % unsigned(L)); b < cols / Block; b += L) {
        float d = 0.0f;
        for(int e = 0; e < Block; ++e)
            d = __fmaf_rn(device_detail::element<Bits>(codes, Block * b + e, values), x[Block * b + e], d);
        acc = __fmaf_rn(d, scales[b], acc);
    }
    return __fmul_rn(device_detail::halving<L>(acc), post);
}

// NVFP4 (E2M1 codes, 16 per scale), as before the format functions.
template <int L>
__device__ inline float dot_row_halving(const std::uint8_t* codes, const float* scales, const float* x, int cols,
                                        float post) {
    return dot_row_halving_format<L, 4, 16>(codes, scales, nullptr, x, cols, post);
}
template <int L>
__device__ inline float dot_row_block(const std::uint8_t* codes, const float* scales, const float* x, int cols,
                                      float post) {
    return dot_row_block_format<L, 4, 16>(codes, scales, nullptr, x, cols, post);
}

}  // namespace crit::techniques::host_nvfp4
