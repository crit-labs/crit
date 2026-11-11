// crit's kernel core: tensor-core block sums for block-scaled dots in a declared order (kernels/core/README.md,
// "blockscaled").
//
// A declared block order (blockscaled_order.cuh) needs each block's sum on its own, before its scales: one mma.sync
// a block with a constant accumulator, the scales applied on the CUDA cores by step(). The wrappers here:
//
//   block_sum_e4m3   E4M3 x E4M3 m16n8k32 with a zero accumulator (sm_89 on): one 32-element block's FP32 sum per
//                    output, in the tensor core's own order (PTX leaves it unspecified: the same on one architecture).
//   block_sum_s8     INT8 x INT8 m16n8k16 (sm_80 on): one 16-element block's exact integer sum, accumulated onto
//                    kSumBias, so biased_sum() turns it into the exact FP32 value with one FP32 subtraction (no
//                    integer-to-float conversion, which runs at a fraction of the FP32 rate).
//
// Fragment layout (PTX m16n8kK): with lane = 4 g + q, d[2h + c] is A row g + 8h against B column 2q + c. A's rows and
// B's columns may move between tile positions without changing an output's bits (each output is its own dot); the k
// positions of an E4M3 block may not, so block_sum_e4m3 takes them as crit's fragment order: thread q's bytes 8q..8q+7
// of the block (a0 = row g bytes 8q..8q+3 at MMA k 4q.., a2 = bytes 8q+4..8q+7 at MMA k 16 + 4q..).
#pragma once
#include <cstdint>

#include "crit_kcore/mma.cuh"

namespace crit::kcore::blockscaled {

// D = A x B with D = 0: a = {row g bytes 8q.. 8q+7, row g + 8 bytes 8q.. 8q+7} (uint2 each), b = column g's bytes
// 8q.. 8q+7 (crit's fragment order on mma.cuh's E4M3 MMA). Traps where the instruction does not exist.
__device__ __forceinline__ void block_sum_e4m3(uint2 a_g, uint2 a_g8, uint2 b, float (&d)[4]) {
    mma_e4m3_m16n8k32_zero(d, a_g.x, a_g8.x, a_g.y, a_g8.y, b.x, b.y);
}

// 0x4B400000 is 1.5 * 2^23 as FP32: an integer s with |s| < 2^22 added to it as an integer gives the FP32 value
// 1.5 * 2^23 + s exactly (the ulp there is 1), and subtracting 1.5 * 2^23 gives float(s) exactly.
constexpr int kSumBias = 0x4B400000;
__device__ __forceinline__ float biased_sum(int biased) { return __fsub_rn(__int_as_float(biased), 12582912.0f); }

// D = A x B + kSumBias on mma.cuh's s8 m16n8k16 fragments (a0 = A row g bytes 4q..4q+3, a1 = row g + 8, b = B column
// g bytes 4q..4q+3). Sixteen int8 x int8 products sum to at most 2^18 in magnitude, inside biased_sum's exact range.
__device__ __forceinline__ void block_sum_s8(uint32_t a0, uint32_t a1, uint32_t b, int (&d)[4]) {
    d[0] = d[1] = d[2] = d[3] = kSumBias;
    mma_s8_m16n8k16(d, a0, a1, b);
}

// The declared step of a block on its lane: lane = fma(sum, weight scale * activation scale, lane), the product
// rounded. Explicit round-to-nearest intrinsics keep the compiler from contracting or reassociating it.
__device__ __forceinline__ float step(float sum, float weight_scale, float act_scale, float lane) {
    return __fmaf_rn(sum, __fmul_rn(weight_scale, act_scale), lane);
}

// Four 4-bit codes (a 16-bit word, the even element in the low nibble) to four int8 values by two tables of eight:
// `pos` the values of codes 0..7 and `neg` those of codes 8..15, each as two words of bytes (lo: codes 0..3 or 8..11).
// Three byte permutes: the low three bits pick a value from each table, the high bit picks the table.
struct NibbleTable {
    unsigned pos_lo, pos_hi, neg_lo, neg_hi;
};
// The tables of E2M1 codes as twice their value (integers: 0, 1, 2, 3, 4, 6, 8, 12 and their negatives) and of INT4
// two's complement codes (0..7, -8..-1).
__host__ __device__ __forceinline__ NibbleTable nibble_table(bool int4) {
    return int4 ? NibbleTable{0x03020100u, 0x07060504u, 0xFBFAF9F8u, 0xFFFEFDFCu}
                : NibbleTable{0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu};
}

__device__ __forceinline__ unsigned nibbles_to_s8x4(unsigned codes16, const NibbleTable& t) {
    const unsigned pick = codes16 & 0x7777u;
    const unsigned pos = __byte_perm(t.pos_lo, t.pos_hi, pick), neg = __byte_perm(t.neg_lo, t.neg_hi, pick);
    return __byte_perm(pos, neg, 0x3210u | ((codes16 >> 1) & 0x4444u));
}

}  // namespace crit::kcore::blockscaled
