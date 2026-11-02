// Metal side of host_nvfp4.h's declared order: the functions of host_nvfp4_device.cuh in the Metal Shading
// Language, for Apple GPUs. Lanes = L threads of one SIMD group (16 or 32), lane j accumulating elements k with
// k % L == j in block order, and a simd_shuffle_down tree that pairs lanes like the host's halving reduction.
//
// Build the library with fast math off (MTLCompileOptions.mathMode = MTLMathModeSafe, or -fno-fast-math for the
// metal compiler): under fast math the compiler may contract the product into the FMA or reorder the adds, and the
// bits no longer follow the order. metal::fma is a fused multiply-add with one rounding, and a float multiply or add
// rounds to nearest even once, as the order declares.
//
// Not run on Apple hardware in crit's tests: the CUDA and host paths are checked against the oracle, this file
// declares the same operations for an engine to check on its device (host_nvfp4_tests' fixtures give the bits).
#pragma once
#include <metal_stdlib>

namespace crit_host_nvfp4 {

inline float e2m1_device(uint code) {
    const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float m = magnitude[code & 7u];
    return (code & 8u) ? -m : m;
}

// Element k of a row: Bits 4 (two codes a byte, low nibble first) or 8, through `values` (null: E2M1).
template <int Bits>
inline float element(device const uchar* codes, int k, device const float* values) {
    if(Bits == 8) return values[codes[k]];
    const uchar byte = codes[k / 2];
    const uint code = k % 2 ? uint(byte >> 4) : uint(byte & 15u);
    return values ? values[code] : e2m1_device(code);
}

// The halving tree over a group of L lanes of the SIMD group; `lane` is thread_index_in_simdgroup. For L = 16 the
// two halves of a 32-wide SIMD group reduce separately: lanes of the other half are never read.
template <int L>
inline float halving(float acc, uint lane) {
    for(ushort w = L / 2; w >= 1; w /= 2) {
        const float other = metal::simd_shuffle_down(acc, w);
        // a lane whose partner lies outside its group keeps its value (only the group's first lane is the result)
        acc = (lane % L) + w < uint(L) ? acc + other : acc;
    }
    return acc;
}

// scaled_elements in any format: called by L converged lanes of one SIMD group. Returns the row's value on the
// group's first lane.
template <int L, int Bits, int Block>
inline float dot_row_halving_format(device const uchar* codes, device const float* scales,
                                    device const float* values, device const float* x, int cols, float post,
                                    uint lane) {
    float acc = 0.0f;
    for(int k = int(lane % uint(L)); k < cols; k += L) {
        const float w = element<Bits>(codes, k, values) * scales[k / Block];
        acc = metal::fma(w, x[k], acc);
    }
    return halving<L>(acc, lane) * post;
}

// block_dot in any format: lane j takes blocks j, j + L, ...; each block's dot runs sequentially over its Block
// elements from 0, then lane = fma(dot, scale, lane).
template <int L, int Bits, int Block>
inline float dot_row_block_format(device const uchar* codes, device const float* scales, device const float* values,
                                  device const float* x, int cols, float post, uint lane) {
    float acc = 0.0f;
    for(int b = int(lane % uint(L)); b < cols / Block; b += L) {
        float d = 0.0f;
        for(int e = 0; e < Block; ++e) d = metal::fma(element<Bits>(codes, Block * b + e, values), x[Block * b + e], d);
        acc = metal::fma(d, scales[b], acc);
    }
    return halving<L>(acc, lane) * post;
}

}  // namespace crit_host_nvfp4
