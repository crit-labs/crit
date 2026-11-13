// EXL3 trellis tiles: the format's procedural codebooks, a tile's circular bitstream and the order its 256 positions
// fill a 16 x 16 tile (the kernel core's decoder for the exl3 checkpoint scheme; crit-eval's CPU decoder is the
// reference, tests/exl3_tests.cu and the exl3-rows kernel's tests the checks).
//
// A matrix [in][out] is stored as trellis [in / 16][out / 16][16 K] 16-bit words, K bits a weight (1 to 8, or 1.5, 2.5,
// 3.5 with the mul1 codebook). A tile's words, read as 32-bit words of two (the first in the low half), form a
// circular stream of 256 K bits read most significant bit first. Position p's state is the 16 bits ending at bit
// (p + 1) K, or, for a half-integer K = KA + 0.5, at (p / 2)(2 KA + 1) + KA for even p and (p / 2 + 1)(2 KA + 1) for
// odd p (odd positions carry the extra bit). A codebook turns the state into an FP16 value with integer arithmetic
// and one FP16 rounding. Position p = 8 L + j (L the lane of a 32-lane group, j 0..7) sits at tile row
// 2 (L % 4) + j % 2 + 8 ((j / 2) % 2), column 2 (L / 8 + 4 (j / 4)) + (L / 4) % 2.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace crit::kcore {

enum Exl3Codebook : int { kExl3Inst = 0, kExl3Mcg = 1, kExl3Mul1 = 2 };

// The FP16 value (as FP32, exactly) of a 16-bit state in codebook CB.
template <int CB>
__device__ __forceinline__ float exl3_value(uint32_t state) {
    if constexpr (CB == kExl3Mul1) {
        const uint32_t x = state * 0x83DCD12Du;
        const uint32_t sum = __dp4a(x, 0x01010101u, 0x6400u);   // the four bytes' sum on 1024.0's FP16 bits
        const __half h = __ushort_as_half(static_cast<unsigned short>(sum & 0xffffu));
        return __half2float(__hfma(h, __ushort_as_half(0x1eee), __ushort_as_half(0xc931)));
    } else {
        uint32_t x = CB == kExl3Inst ? state * 89226354u + 64248484u : state * 0xCBAC1FEDu;
        x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
        return __half2float(__hadd(__ushort_as_half(static_cast<unsigned short>(x & 0xffffu)),
                                   __ushort_as_half(static_cast<unsigned short>(x >> 16))));
    }
}

// The 32-bit words of a tile of `bits` (KA for a half-integer rate) bits a weight.
__host__ __device__ __forceinline__ constexpr int exl3_words(int bits, bool half_bit) {
    return 8 * bits + (half_bit ? 4 : 0);
}

// Position p's 16-bit state from a tile's words.
__device__ __forceinline__ uint32_t exl3_state(const uint32_t* words, int count, int bits, bool half_bit, int p) {
    const int pair = 2 * bits + 1;
    int end = half_bit ? (p >> 1) * pair + ((p & 1) ? pair : bits) : (p + 1) * bits;
    end += 32 * count;   // a window near the start wraps to the stream's end
    const int last = ((end - 1) >> 5) % count, first = ((end - 16) >> 5) % count;
    const int shift = (((end - 1) >> 5) + 1) * 32 - end;
    return __funnelshift_r(words[last], words[first], shift) & 0xffffu;
}

// Tile row and column of position 8 lane + j.
__host__ __device__ __forceinline__ constexpr int exl3_row(int lane, int j) {
    return 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1);
}
__host__ __device__ __forceinline__ constexpr int exl3_col(int lane, int j) {
    return 2 * ((lane >> 3) + 4 * (j >> 2)) + ((lane >> 2) & 1);
}

// Two states' FP16 values as a pair's bits (the first in the low half): exl3_value's arithmetic, two at once.
template <int CB>
__device__ __forceinline__ uint32_t exl3_pair(uint32_t s0, uint32_t s1) {
    __half2 v;
    if constexpr (CB == kExl3Mul1) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t h0 = __dp4a(x0, 0x01010101u, 0x6400u) & 0xffffu, h1 = __dp4a(x1, 0x01010101u, 0x6400u) & 0xffffu;
        uint32_t packed = h0 | (h1 << 16);
        v = __hfma2(*reinterpret_cast<const __half2*>(&packed), __half2half2(__ushort_as_half(0x1eee)),
                    __half2half2(__ushort_as_half(0xc931)));
    } else {
        uint32_t x0 = CB == kExl3Inst ? s0 * 89226354u + 64248484u : s0 * 0xCBAC1FEDu;
        uint32_t x1 = CB == kExl3Inst ? s1 * 89226354u + 64248484u : s1 * 0xCBAC1FEDu;
        x0 = (x0 & 0x8fff8fffu) ^ 0x3b603b60u;
        x1 = (x1 & 0x8fff8fffu) ^ 0x3b603b60u;
        const uint32_t lows = (x0 & 0xffffu) | (x1 << 16), highs = (x0 >> 16) | (x1 & 0xffff0000u);
        v = __hadd2(*reinterpret_cast<const __half2*>(&lows), *reinterpret_cast<const __half2*>(&highs));
    }
    return *reinterpret_cast<const uint32_t*>(&v);
}

// Where lane `lane`'s windows of a tile at BITS bits a weight (HALF: BITS + 0.5) lie: the three 32-bit words that
// hold them (its windows span 7 BITS + 16 bits at most and start in the first word) and the first window's bit in it.
// The same for every tile, so a kernel computes it once.
struct Exl3Lane {
    int w0, w1, w2, r;
};
template <int BITS, bool HALF>
__device__ __forceinline__ Exl3Lane exl3_lane(int lane) {
    constexpr int count = exl3_words(BITS, HALF), pair = 2 * BITS + 1;
    const int end = HALF ? 4 * lane * pair + BITS : (8 * lane + 1) * BITS;
    const int start = end - 16 + 32 * count;   // the first window's first bit, unwrapped
    const int w0 = (start >> 5) % count;
    return {w0, (w0 + 1) % count, (w0 + 2) % count, start & 31};
}

// A lane's three words of a tile (exl3_lane's), loaded ahead of their decode.
__device__ __forceinline__ uint3 exl3_lane_words(const uint32_t* __restrict__ tile, const Exl3Lane& at) {
    return make_uint3(__ldg(tile + at.w0), __ldg(tile + at.w1), __ldg(tile + at.w2));
}

// The states of positions 8 lane .. 8 lane + 7 from the lane's words: shifted once so the first window starts at bit
// 0, then constant shifts (positions 8 lane are even, so every lane's windows lie alike after that).
template <int BITS, bool HALF>
__device__ __forceinline__ void exl3_lane_decode(const uint3& w, int r, uint32_t (&s)[8]) {
    constexpr int pair = 2 * BITS + 1;
    auto end_of = [](int p) { return HALF ? (p >> 1) * pair + ((p & 1) ? pair : BITS) : (p + 1) * BITS; };
    const uint32_t A = __funnelshift_l(w.y, w.x, r), B = __funnelshift_l(w.z, w.y, r), C = w.z << r;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int e = end_of(j) - end_of(0) + 16;   // the window's end from the first window's start: a constant
        if (e <= 32) s[j] = (A >> (32 - e)) & 0xffffu;
        else if (e <= 48) s[j] = __funnelshift_r(B, A, 64 - e) & 0xffffu;
        else if (e <= 64) s[j] = (B >> (64 - e)) & 0xffffu;
        else s[j] = __funnelshift_r(C, B, 96 - e) & 0xffffu;
    }
}

// Both at once.
template <int BITS, bool HALF>
__device__ __forceinline__ void exl3_lane_states(const uint32_t* __restrict__ tile, const Exl3Lane& at,
                                                 uint32_t (&s)[8]) {
    exl3_lane_decode<BITS, HALF>(exl3_lane_words(tile, at), at.r, s);
}

}  // namespace crit::kcore
