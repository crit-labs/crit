// Radix select's pieces: what a selection that finds a boundary key without sorting shares (sparse-indexer's k best
// scores, token-sampling's prefix searches of the full law): the keys, the digit a pass keeps, and a warp's search of
// a 256-bin histogram. Integer and comparison work only, except digit_from_top's running total, which adds the bins
// from the top in order, so a selection keeps its picks and its sums whichever of these it takes.
#pragma once

#include <cstdint>

namespace crit::kcore {

// A float's bits as a uint32 key of the same order (-0 just below +0), a NaN as 0: below every number, -inf
// included (radix_key_bits takes the bits, for keys made in place).
__device__ __forceinline__ std::uint32_t radix_key_bits(std::uint32_t bits) {
    const std::uint32_t key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    return (bits & 0x7fffffffu) > 0x7f800000u ? 0u : key;
}
__device__ __forceinline__ std::uint32_t radix_key(float v) { return radix_key_bits(__float_as_uint(v)); }

// The float of a number's key (radix_key's inverse; key 0 gives a NaN).
__device__ __forceinline__ float radix_float(std::uint32_t key) {
    return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
}

// A pass's digit: the highest bin b (BINS - 1 down to 0, bins of no positive mass skipped) at which the running total
// from the top, `cum` plus the bins above b plus b's mass, reaches `target` (passes it when `reach` is false). `cum`
// enters as the total above every bin and leaves as the total above the digit found. -1 when the bins' whole mass
// stays below the target (rounding at the last bin). mass(b) is the bin's mass (an int count or a sum of weights).
template<int BINS, class T, class Mass>
__device__ __forceinline__ int digit_from_top(Mass mass, T target, bool reach, T& cum) {
    for(int b = BINS - 1; b >= 0; --b) {
        const T m = mass(b);
        if(!(m > T(0))) continue;
        if(reach ? cum + m >= target : cum + m > target) return b;
        cum += m;
    }
    return -1;
}

// The digit of a 256-bin count histogram that holds the need-th largest key, found by one warp from shared memory
// with no word from other warps: a lane reads 8 bins (8 lane .. 8 lane + 7), a suffix sum over the lanes gives each
// group the keys at or above it, the highest group with `need` keys at or above it finds its highest such bin, and
// shuffles tell the warp. Every lane calls it with the same bins and need. clamp: need first becomes at most every
// key the bins hold (a first pass, where a row may have fewer than k). Returns the digit and leaves in `need` what
// is still needed among that digit's keys.
__device__ __forceinline__ int warp_digit_256(const int* bins, int& need, bool clamp) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    int group[8], ge = 0;
#pragma unroll
    for(int i = 0; i < 8; ++i) {
        group[i] = bins[8 * lane + i];
        ge += group[i];
    }
#pragma unroll
    for(int off = 1; off < 32; off <<= 1) {
        const int v = __shfl_down_sync(0xffffffffu, ge, off);
        if(lane + off < 32) ge += v;
    }
    if(clamp) need = min(need, __shfl_sync(0xffffffffu, ge, 0));
    // The highest group with `need` keys at or above it (the last one where need is 0), then its highest such bin.
    const int holder = 31 - __clz(__ballot_sync(0xffffffffu, ge >= need) | 1u);
    int above = ge, digit = 8 * lane, left = need;
#pragma unroll
    for(int i = 0; i < 8; ++i) above -= group[i];   // the keys above the lane's group
#pragma unroll
    for(int i = 7; i >= 0; --i) {
        if(above + group[i] >= left && digit == 8 * lane) {   // (the first bin from the top that reaches it)
            digit = 8 * lane + i + 256;
            left -= above;
        }
        above += group[i];
    }
    digit = __shfl_sync(0xffffffffu, digit, holder) & 0xFF;
    need = __shfl_sync(0xffffffffu, left, holder);
    return digit;
}

}  // namespace crit::kcore
