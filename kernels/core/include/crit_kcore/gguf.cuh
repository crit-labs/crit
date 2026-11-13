// GGUF block formats (the GGUF file format's quantized tensor types): decoders of a block's 32-value sub-blocks into
// the int8 values dp4a and the s8 MMA multiply, and the FP32 scales that weight them, exact on every architecture.
//
// A row of a GGUF matrix is its blocks in order, block_bytes(F) bytes for block_values(F) values (32, or a 256-value
// superblock of eight 32-value sub-blocks). subblock<F>(block, ib, q, s0, s1, m) gives sub-block ib's 32 values as
// integers q[0..7] (four int8 a word, value 4 w + i in byte i of word w) and its scales: value e is s0 * q_e (e < 16)
// or s1 * q_e (e >= 16), less m where the format has a minimum (Q4_K and Q5_K: s * q - m; Q4_1 and Q5_1 store an
// offset added to s * q, so m is its negation, exact). s0 == s1 for a format with one scale a sub-block (all but Q3_K
// and Q6_K, whose scales cover 16 values). The scales are FP32 products of the stored FP16 scale and the sub-block's
// integer scale, rounded once (__fmul_rn), so value() = fsub(fmul(s, q_e), m) is the stored value bit for bit as the
// format's reference decoder computes it in FP32 (IQ2's scales are fmul(fmul(d, 0.5 + ls), 0.25), 0.5 for IQ3_XXS,
// as the formats compute them; IQ1's offset of an eighth folds into the codes: code 8 g +- 1, scale its d l / 8, which
// is exact, so s * code is d l (g +- 1/8) rounded once). Q2_K has a minimum a 16 values too: the seven-argument subblock()
// gives m0 (values 0..15) and m1 (16..31); the six-argument one is for the formats with one minimum a sub-block.
//
// Formats (their GGUF type ids): Q4_0 (2), Q4_1 (3), Q5_0 (6), Q5_1 (7), Q8_0 (8), Q2_K (10), Q3_K (11), Q4_K (12), Q5_K (13),
// Q6_K (14), IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ1_S (19), IQ4_NL (20), IQ3_S (21), IQ2_S (22), IQ4_XS (23),
// IQ1_M (29), MXFP4 (39: E2M1 values doubled as integers, the E8M0 byte e halved: 2^(e - 128), exact). The IQ grids (here and in gguf_grids.cuh) and the IQ4 values are the formats' own tables.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

#include "gguf_grids.cuh"

namespace crit::kcore::gguf {

enum Format : int {
    Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, IQ1_M = 29,
    MXFP4 = 39
};

// A block's bytes in a GGUF type (0: not one of these formats), whether the type is one, and its block's values
// (host and device).
__host__ __device__ constexpr int block_bytes(int type) {
    return type == Q4_0 ? 18 : type == Q4_1 ? 20 : type == Q5_0 ? 22 : type == Q5_1 ? 24 : type == Q8_0 ? 34
         : type == Q2_K ? 84 : type == Q3_K ? 110 : type == Q4_K ? 144 : type == Q5_K ? 176 : type == Q6_K ? 210 : type == IQ4_NL ? 18
         : type == IQ3_S ? 110 : type == IQ4_XS ? 136 : type == IQ2_XXS ? 66 : type == IQ2_XS ? 74 : type == IQ3_XXS ? 98
         : type == IQ1_S ? 50 : type == IQ2_S ? 82 : type == IQ1_M ? 56 : type == MXFP4 ? 17 : 0;
}
__host__ __device__ constexpr bool supported(int type) { return block_bytes(type) != 0; }
__host__ __device__ constexpr int block_values(int type) {
    return type == Q4_0 || type == Q4_1 || type == Q5_0 || type == Q5_1 || type == Q8_0 || type == IQ4_NL ||
                   type == MXFP4
               ? 32
               : 256;
}
// Whether value e's scale is s0 for e < 16 and s1 for e >= 16 (Q2_K, Q3_K, Q6_K, IQ2_XS, IQ2_S, IQ1_M), whether the
// format has a minimum (Q4_1, Q5_1, Q4_K, Q5_K, Q2_K), and whether its minimum is m0 for e < 16 and m1 for e >= 16
// (Q2_K).
__host__ __device__ constexpr bool two_scales(int type) {
    return type == Q2_K || type == Q3_K || type == Q6_K || type == IQ2_XS || type == IQ2_S || type == IQ1_M;
}
__host__ __device__ constexpr bool has_min(int type) {
    return type == Q4_1 || type == Q5_1 || type == Q4_K || type == Q5_K || type == Q2_K;
}
__host__ __device__ constexpr bool two_mins(int type) { return type == Q2_K; }

// IQ4_NL's and IQ4_XS's 16 values (-127, -104, ..., 113), four a word, low byte first; MXFP4's (E2M1 doubled: 0, 1,
// 2, 3, 4, 6, 8, 12, then their negations).
static __device__ const uint32_t kIq4Values[4] = {0xbfad9881u, 0xf6eaddcfu, 0x26190d01u, 0x71594535u};
static __device__ const uint32_t kMxfp4Values[4] = {0x03020100u, 0x0c080604u, 0xfdfeff00u, 0xf4f8fafcu};
// IQ3_S's 512 grid entries of four values (1, 3, ..., 15), low byte first: crit_core/model/gguf_grids.json's iq3_s,
// which crit_core.model.gguf_blocks reads (tests check the two equal).
static __device__ const uint32_t kIq3sGrid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

__device__ __forceinline__ float half_at(const uint8_t* p) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(p[0] | (p[1] << 8))));
}
__device__ __forceinline__ uint32_t word_at(const uint8_t* p) {   // four bytes, any alignment
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
// Four 4-bit codes (bits 4 i .. 4 i + 3 of `nibbles`) through a 16-value table (IQ4's, MXFP4's): one int8x4 word.
__device__ __forceinline__ uint32_t table_word(const uint32_t* table, uint32_t nibbles) {
    uint32_t out = 0;
#pragma unroll
    for(int i = 0; i < 4; ++i) {
        const uint32_t code = (nibbles >> (4 * i)) & 15u;
        out |= ((table[code >> 2] >> (8 * (code & 3u))) & 0xffu) << (8 * i);
    }
    return out;
}
__device__ __forceinline__ uint32_t iq4_word(uint32_t nibbles) { return table_word(kIq4Values, nibbles); }
// The nibbles of four bytes' low (shift 0) or high (shift 4) halves, gathered into the low 16 bits.
__device__ __forceinline__ uint32_t nibbles_of(uint32_t bytes, int shift) {
    const uint32_t n = (bytes >> shift) & 0x0f0f0f0fu;
    return (n & 0xfu) | ((n >> 4) & 0xf0u) | ((n >> 8) & 0xf00u) | ((n >> 12) & 0xf000u);
}
// Four int8 values (a grid's word) with value i negated where bit i of `bits` is set.
__device__ __forceinline__ uint32_t signed_word(uint32_t grid, uint32_t bits) {
    const uint32_t mask = ((bits & 1u) ? 0xffu : 0u) | ((bits & 2u) ? 0xff00u : 0u) | ((bits & 4u) ? 0xff0000u : 0u) |
                          ((bits & 8u) ? 0xff000000u : 0u);
    return __vsub4(grid ^ mask, mask);   // (g ^ 0xff) - 0xff = -g in a byte
}
// IQ2's and IQ3_XXS's eight signs from a 7-bit index: the eighth makes their count even.
__device__ __forceinline__ uint32_t even_signs(uint32_t index) { return index | ((__popc(index) & 1u) << 7); }
// IQ1's four int8 grid values g (-1, 0, 1) as 8 g + 1 (offset +1/8) or 8 g - 1 (-1/8).
__device__ __forceinline__ uint32_t iq1_word(uint32_t grid, bool negative) {
    const uint32_t eight = (grid << 3) & 0xf8f8f8f8u;
    return negative ? __vsub4(eight, 0x01010101u) : __vadd4(eight, 0x01010101u);
}
__device__ __forceinline__ uint32_t grid_half(uint64_t entry, int half) { return uint32_t(entry >> (32 * half)); }
// Bits 0..3 of `bits` into bit 4 of bytes 0..3: a 5-bit format's high bits over four values.
__device__ __forceinline__ uint32_t bit4_of(uint32_t bits) {
    return ((bits & 1u) << 4) | ((bits & 2u) << 11) | ((bits & 4u) << 18) | ((bits & 8u) << 25);
}
// Q3_K's 6-bit scale i (0..15) of the block's 12 bytes, unbiased (less 32).
__device__ __forceinline__ int q3k_scale(const uint8_t* scales, int i) {
    const int low = i < 8 ? scales[i] & 0x0f : scales[i - 8] >> 4;
    return (low | (((scales[8 + i % 4] >> (2 * (i / 4))) & 3) << 4)) - 32;
}
// Q4_K's and Q5_K's 6-bit scale and minimum of sub-block j (scales: the block's 12 bytes).
__device__ __forceinline__ void q4k_scale_min(const uint8_t* scales, int j, int& sc, int& mn) {
    if(j < 4) {
        sc = scales[j] & 63;
        mn = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4);
        mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

template <int F>
__device__ __forceinline__ void subblock(const uint8_t* b, int ib, int32_t q[8], float& s0, float& s1, float& m,
                                         float& m1) {
    m = m1 = 0.0f;
    if constexpr(F == Q8_0) {
        s0 = s1 = half_at(b);
#pragma unroll
        for(int w = 0; w < 8; ++w) q[w] = static_cast<int32_t>(word_at(b + 2 + 4 * w));
    } else if constexpr(F == MXFP4) {
        // The E8M0 byte, then 16 bytes: values 0..15 their low nibbles, 16..31 their high ones. The scale is half the
        // byte's power of two (the values are doubled), 2^(e - 128), as the format's decoder builds it.
        const uint32_t e = b[0];
        s0 = s1 = __uint_as_float(e < 2 ? 0x00200000u << e : (e - 1) << 23);
#pragma unroll
        for(int w = 0; w < 4; ++w) {
            const uint32_t bytes = word_at(b + 1 + 4 * w);
            q[w] = static_cast<int32_t>(table_word(kMxfp4Values, nibbles_of(bytes, 0)));
            q[w + 4] = static_cast<int32_t>(table_word(kMxfp4Values, nibbles_of(bytes, 4)));
        }
    } else if constexpr(F == IQ4_NL || F == IQ4_XS) {
        // 32 values from 16 bytes: values 0..15 the bytes' low nibbles, 16..31 their high nibbles.
        const uint8_t* qs;
        if constexpr(F == IQ4_NL) {
            s0 = s1 = half_at(b);
            qs = b + 2;
        } else {
            const uint32_t scales_h = b[2] | (b[3] << 8);
            const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0x0f) | (((scales_h >> (2 * ib)) & 3) << 4);
            s0 = s1 = __fmul_rn(half_at(b), float(ls - 32));
            qs = b + 8 + 16 * ib;
        }
#pragma unroll
        for(int w = 0; w < 4; ++w) {
            const uint32_t bytes = word_at(qs + 4 * w);
            q[w] = static_cast<int32_t>(iq4_word(nibbles_of(bytes, 0)));
            q[w + 4] = static_cast<int32_t>(iq4_word(nibbles_of(bytes, 4)));
        }
    } else if constexpr(F == Q4_0 || F == Q4_1 || F == Q5_0 || F == Q5_1) {
        // 32 values from 16 bytes: values 0..15 the bytes' low nibbles, 16..31 their high nibbles; Q5's fifth bit of
        // value e is bit e of the 32-bit qh. Q4_0 and Q5_0 are centred (less 8, less 16); Q4_1 and Q5_1 add an offset.
        constexpr bool five = F == Q5_0 || F == Q5_1, offset = F == Q4_1 || F == Q5_1;
        s0 = s1 = half_at(b);
        if constexpr(offset) m = -half_at(b + 2);
        const uint8_t* qs = b + (offset ? 4 : 2) + (five ? 4 : 0);
        const uint32_t qh = five ? word_at(b + (offset ? 4 : 2)) : 0u;
        constexpr uint32_t centre = offset ? 0u : five ? 0x10101010u : 0x08080808u;
#pragma unroll
        for(int w = 0; w < 4; ++w) {
            const uint32_t bytes = word_at(qs + 4 * w);
            uint32_t lo = bytes & 0x0f0f0f0fu, hi = (bytes >> 4) & 0x0f0f0f0fu;
            if constexpr(five) {
                lo |= bit4_of(qh >> (4 * w));
                hi |= bit4_of(qh >> (16 + 4 * w));
            }
            q[w] = static_cast<int32_t>(__vsub4(lo, centre));
            q[w + 4] = static_cast<int32_t>(__vsub4(hi, centre));
        }
    } else if constexpr(F == Q5_K) {
        // Q4_K's layout with a fifth bit a value: value l of sub-block ib has bit ib of qh[l] (32 bytes after the
        // scales) as bit 4.
        int sc, mn;
        q4k_scale_min(b + 4, ib, sc, mn);
        s0 = s1 = __fmul_rn(half_at(b), float(sc));
        m = __fmul_rn(half_at(b + 2), float(mn));
        const uint8_t* qh = b + 16;
        const uint8_t* qs = b + 48 + 32 * (ib / 2);
        const int shift = 4 * (ib % 2);
#pragma unroll
        for(int w = 0; w < 8; ++w) {
            const uint32_t lo = (word_at(qs + 4 * w) >> shift) & 0x0f0f0f0fu;
            q[w] = static_cast<int32_t>(lo | (((word_at(qh + 4 * w) >> ib) & 0x01010101u) << 4));
        }
    } else if constexpr(F == Q2_K) {
        // scales 16 bytes (a 4-bit scale, low nibble, and a 4-bit minimum a 16 values), qs 64, d, dmin. Sub-block
        // ib = 4 h + g: value l is bits 2 g, 2 g + 1 of qs[32 h + l]; scales 2 ib and 2 ib + 1.
        const float d = half_at(b + 80), dmin = half_at(b + 82);
        const int lo_sc = b[2 * ib], hi_sc = b[2 * ib + 1];
        s0 = __fmul_rn(d, float(lo_sc & 15));
        s1 = __fmul_rn(d, float(hi_sc & 15));
        m = __fmul_rn(dmin, float(lo_sc >> 4));
        m1 = __fmul_rn(dmin, float(hi_sc >> 4));
        const uint8_t* qs = b + 16 + 32 * (ib / 4);
        const int shift = 2 * (ib % 4);
#pragma unroll
        for(int w = 0; w < 8; ++w) q[w] = static_cast<int32_t>((word_at(qs + 4 * w) >> shift) & 0x03030303u);
    } else if constexpr(F == Q3_K) {
        // hmask 32 bytes, qs 64, scales 12, d. Sub-block ib = 4 h + g: value l is bits 2 g, 2 g + 1 of qs[32 h + l],
        // less 4 where bit ib of hmask[l] is clear; scales 2 ib and 2 ib + 1 (6-bit, less 32), one a 16 values.
        const float d = half_at(b + 108);
        const int h = ib / 4, g = ib % 4;
        s0 = __fmul_rn(d, float(q3k_scale(b + 96, 2 * ib)));
        s1 = __fmul_rn(d, float(q3k_scale(b + 96, 2 * ib + 1)));
        const uint8_t* qs = b + 32 + 32 * h;
#pragma unroll
        for(int w = 0; w < 8; ++w) {
            const uint32_t lo = (word_at(qs + 4 * w) >> (2 * g)) & 0x03030303u;
            const uint32_t hi = ((word_at(b + 4 * w) >> ib) & 0x01010101u) << 2;
            q[w] = static_cast<int32_t>(__vsub4(lo | hi, 0x04040404u));   // each byte less 4, as int8
        }
    } else if constexpr(F == Q4_K) {
        int sc, mn;
        q4k_scale_min(b + 4, ib, sc, mn);
        s0 = s1 = __fmul_rn(half_at(b), float(sc));
        m = __fmul_rn(half_at(b + 2), float(mn));
        // 64 values a 32-byte run of qs: sub-block 2 j the low nibbles, 2 j + 1 the high ones.
        const uint8_t* qs = b + 16 + 32 * (ib / 2);
        const int shift = 4 * (ib % 2);
#pragma unroll
        for(int w = 0; w < 8; ++w) q[w] = static_cast<int32_t>((word_at(qs + 4 * w) >> shift) & 0x0f0f0f0fu);
    } else if constexpr(F == Q6_K) {
        // 128-value halves: ql 64 bytes, qh 32; within a half, value 32 g + l (g < 4) is ql[32 (g & 1) + l]'s nibble
        // g / 2 and qh[l]'s bits 2 g and 2 g + 1, less 32. Scales: int8, one a 16 values.
        const float d = half_at(b + 208);
        const int half = ib / 4, g = ib % 4;
        const uint8_t* ql = b + 64 * half + 32 * (g & 1);
        const uint8_t* qh = b + 128 + 32 * half;
        const int8_t* scales = reinterpret_cast<const int8_t*>(b + 192);
        s0 = __fmul_rn(d, float(scales[2 * ib]));
        s1 = __fmul_rn(d, float(scales[2 * ib + 1]));
        const int lshift = 4 * (g >> 1), hshift = 2 * g;
#pragma unroll
        for(int w = 0; w < 8; ++w) {
            const uint32_t lo = (word_at(ql + 4 * w) >> lshift) & 0x0f0f0f0fu;
            const uint32_t hi = ((word_at(qh + 4 * w) >> hshift) & 0x03030303u) << 4;
            q[w] = static_cast<int32_t>(__vsub4(lo | hi, 0x20202020u));   // each byte less 32, as int8
        }
    } else if constexpr(F == IQ2_XXS) {
        // 8 bytes a sub-block: four grid indices (eight values each), then a word of four 7-bit sign indices and the
        // 4-bit scale (bits 28..31).
        const uint8_t* g = b + 2 + 8 * ib;
        const uint32_t aux = word_at(g + 4);
        s0 = s1 = __fmul_rn(__fmul_rn(half_at(b), 0.5f + float(aux >> 28)), 0.25f);
#pragma unroll
        for(int w = 0; w < 8; ++w) {
            const uint32_t signs = even_signs((aux >> (7 * (w / 2))) & 127u) >> (4 * (w & 1));
            q[w] = static_cast<int32_t>(signed_word(grid_half(kIq2xxsGrid[g[w / 2]], w & 1), signs));
        }
    } else if constexpr(F == IQ2_XS) {
        // Four 16-bit entries a sub-block: a 9-bit grid index and a 7-bit sign index; a 4-bit scale a 16 values.
        const uint8_t* g = b + 2 + 8 * ib;
        const float d = half_at(b);
        const int scales = b[66 + ib];
        s0 = __fmul_rn(__fmul_rn(d, 0.5f + float(scales & 15)), 0.25f);
        s1 = __fmul_rn(__fmul_rn(d, 0.5f + float(scales >> 4)), 0.25f);
#pragma unroll
        for(int w = 0; w < 8; ++w) {
            const uint32_t entry = g[2 * (w / 2)] | (uint32_t(g[2 * (w / 2) + 1]) << 8);
            const uint32_t signs = even_signs(entry >> 9) >> (4 * (w & 1));
            q[w] = static_cast<int32_t>(signed_word(grid_half(kIq2xsGrid[entry & 511u], w & 1), signs));
        }
    } else if constexpr(F == IQ2_S) {
        // qs 32 bytes (four grid indices' low 8 bits a sub-block), signs 32 (a byte an eight values), qh 8 (the
        // indices' bits 8 and 9), scales 8 (a 4-bit scale a 16 values).
        const float d = half_at(b);
        const int scales = b[74 + ib], qh = b[66 + ib];
        s0 = __fmul_rn(__fmul_rn(d, 0.5f + float(scales & 15)), 0.25f);
        s1 = __fmul_rn(__fmul_rn(d, 0.5f + float(scales >> 4)), 0.25f);
#pragma unroll
        for(int w = 0; w < 8; ++w) {
            const int l = w / 2;
            const uint32_t index = b[2 + 4 * ib + l] | (((qh >> (2 * l)) & 3u) << 8);
            const uint32_t signs = uint32_t(b[34 + 4 * ib + l]) >> (4 * (w & 1));
            q[w] = static_cast<int32_t>(signed_word(grid_half(kIq2sGrid[index], w & 1), signs));
        }
    } else if constexpr(F == IQ3_XXS) {
        // qs 64 bytes (eight grid indices of four values a sub-block), then a word a sub-block: four 7-bit sign
        // indices and the 4-bit scale.
        const uint32_t aux = word_at(b + 66 + 4 * ib);
        s0 = s1 = __fmul_rn(__fmul_rn(half_at(b), 0.5f + float(aux >> 28)), 0.5f);
#pragma unroll
        for(int w = 0; w < 8; ++w) {
            const uint32_t signs = even_signs((aux >> (7 * (w / 2))) & 127u) >> (4 * (w & 1));
            q[w] = static_cast<int32_t>(signed_word(kIq3xxsGrid[b[2 + 8 * ib + w]], signs));
        }
    } else if constexpr(F == IQ1_S) {
        // qs 32 bytes (four 11-bit grid indices' low bits a sub-block), then a 16-bit qh a sub-block: the indices'
        // bits 8..10 (3 each), the 3-bit scale (bits 12..14) and the offset's sign (bit 15).
        const uint32_t qh = b[34 + 2 * ib] | (uint32_t(b[35 + 2 * ib]) << 8);
        s0 = s1 = __fmul_rn(__fmul_rn(half_at(b), float(2 * ((qh >> 12) & 7u) + 1)), 0.125f);
        const bool negative = (qh & 0x8000u) != 0;
#pragma unroll
        for(int w = 0; w < 8; ++w) {
            const int l = w / 2;
            const uint32_t index = b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7u) << 8);
            q[w] = static_cast<int32_t>(iq1_word(grid_half(kIq1sGrid[index], w & 1), negative));
        }
    } else if constexpr(F == IQ1_M) {
        // qs 32 bytes, qh 16 (a byte two indices: bits 8..10 of each and its group's offset sign, bits 3 and 7), and
        // four 16-bit scale words: four 3-bit scales each (one a 16 values) and, in their top nibbles, the FP16 d.
        const uint8_t* sc = b + 48;
        const uint32_t w0 = sc[0] | (sc[1] << 8), w1 = sc[2] | (sc[3] << 8), w2 = sc[4] | (sc[5] << 8),
                       w3 = sc[6] | (sc[7] << 8);
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(
            (w0 >> 12) | ((w1 >> 8) & 0x00f0u) | ((w2 >> 4) & 0x0f00u) | (w3 & 0xf000u))));
        const uint32_t word = sc[2 * (ib / 2)] | (sc[2 * (ib / 2) + 1] << 8);
        const int shift = 6 * (ib % 2);
        s0 = __fmul_rn(__fmul_rn(d, float(2 * ((word >> shift) & 7u) + 1)), 0.125f);
        s1 = __fmul_rn(__fmul_rn(d, float(2 * ((word >> (shift + 3)) & 7u) + 1)), 0.125f);
#pragma unroll
        for(int w = 0; w < 8; ++w) {
            const int l = w / 2;
            const uint32_t qh = b[32 + 2 * ib + l / 2] >> (4 * (l & 1));
            const uint32_t index = b[4 * ib + l] | ((qh & 7u) << 8);
            q[w] = static_cast<int32_t>(iq1_word(grid_half(kIq1sGrid[index], w & 1), (qh & 8u) != 0));
        }
    } else if constexpr(F == IQ3_S) {
        const uint8_t* qs = b + 2 + 8 * ib;
        const uint32_t qh = b[66 + ib];
        const uint32_t signs = word_at(b + 74 + 4 * ib);
        const int scale = (b[106 + ib / 2] >> (4 * (ib % 2))) & 0x0f;
        s0 = s1 = __fmul_rn(half_at(b), float(1 + 2 * scale));
#pragma unroll
        for(int w = 0; w < 8; ++w) {
            const uint32_t grid = kIq3sGrid[qs[w] | (((qh >> w) & 1u) << 8)];
            // Value 4 w + i's sign: bit 4 w + i of the sub-block's signs; a set bit negates its byte.
            q[w] = static_cast<int32_t>(signed_word(grid, (signs >> (4 * w)) & 15u));
        }
    }
    if constexpr(!two_mins(F)) m1 = m;   // one minimum a sub-block: m1 is m
}

// Q2_K in integer form, what gguf-rows multiplies: sub-block ib's codes times their 4-bit scales (at most 3 x 15 = 45,
// an int8; values 0..15 by the first scale, 16..31 by the second), the block's d and dmin, and the sub-block's two
// 4-bit minimums (values 0..15, 16..31). Value e is d sc q - dmin mn, which the scaled codes' exact integer dot with
// the activations and the minimums' with their sums give for a row at once.
__device__ __forceinline__ void q2k_scaled(const uint8_t* b, int ib, int32_t q[8], float& d, float& dmin, int& mn0,
                                           int& mn1) {
    d = half_at(b + 80);
    dmin = half_at(b + 82);
    const uint32_t lo_sc = b[2 * ib], hi_sc = b[2 * ib + 1];
    mn0 = int(lo_sc >> 4);
    mn1 = int(hi_sc >> 4);
    const uint8_t* qs = b + 16 + 32 * (ib / 4);
    const int shift = 2 * (ib % 4);
#pragma unroll
    for(int w = 0; w < 8; ++w)   // bytes at most 45: the word's multiply carries into no other byte
        q[w] = static_cast<int32_t>(((word_at(qs + 4 * w) >> shift) & 0x03030303u) * ((w < 4 ? lo_sc : hi_sc) & 15u));
}

// The formats with one minimum a sub-block (or none): m for every value.
template <int F>
__device__ __forceinline__ void subblock(const uint8_t* b, int ib, int32_t q[8], float& s0, float& s1, float& m) {
    static_assert(!two_mins(F), "a format with a minimum a 16 values takes subblock's m0 and m1");
    float m1;
    subblock<F>(b, ib, q, s0, s1, m, m1);
}

// Value e (0..31) of a decoded sub-block, as the format's reference decoder computes it in FP32.
__device__ __forceinline__ int8_t code(const int32_t q[8], int e) {
    return static_cast<int8_t>((static_cast<uint32_t>(q[e / 4]) >> (8 * (e % 4))) & 0xffu);
}
__device__ __forceinline__ float value(const int32_t q[8], int e, float s0, float s1, float m) {
    return __fsub_rn(__fmul_rn(e < 16 ? s0 : s1, float(code(q, e))), m);
}
__device__ __forceinline__ float value(const int32_t q[8], int e, float s0, float s1, float m0, float m1) {
    return __fsub_rn(__fmul_rn(e < 16 ? s0 : s1, float(code(q, e))), e < 16 ? m0 : m1);
}

}  // namespace crit::kcore::gguf
