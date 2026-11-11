// crit's kernel core: the fastest accumulating E4M3 tensor-core MMA a target has (kernels/core/README.md, "mma_fp8").
// On the sm_120 family's specific targets (CRIT_KCORE_HAS_BLOCK_SCALED_MMA) the block-scaled kind::mxf8f6f4 m16n8k32
// runs at twice the rate of the plain e4m3 mma.sync (RTX 5090, from registers: 120 G against 61 G MMAs a second), so
// a GEMM whose scales do not change along K (one scale a weight row and a token, applied in the epilogue) runs it with
// unit scales (E8M0 127 on both operands); elsewhere the plain e4m3 m16n8k32 (8.9 on). Both accumulate in FP32 in the
// tensor core's own order: a sum's bits depend on the instruction (not the declared order).
#pragma once
#include "crit_kcore/mma.cuh"

#include <cstdint>

namespace crit::kcore {

// E8M0 1.0 in every byte: the unit scales of a block-scaled MMA.
constexpr uint32_t kE8M0Ones = 0x7f7f7f7fu;

// c += A (16 x 32 E4M3) B (32 x 8 E4M3), FP32 accumulation, at the target's fastest E4M3 rate.
__device__ __forceinline__ void mma_e4m3_m16n8k32_fast(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                       uint32_t b0, uint32_t b1) {
#if CRIT_KCORE_HAS_BLOCK_SCALED_MMA
    mma_mxf8_e4m3_m16n8k32(c, a0, a1, a2, a3, b0, b1, kE8M0Ones, kE8M0Ones);
#else
    mma_e4m3_m16n8k32(c, a0, a1, a2, a3, b0, b1);
#endif
}

// d = A B from a zero accumulator (a sum's first block), at the same rate: no registers to clear first.
__device__ __forceinline__ void mma_e4m3_m16n8k32_fast_zero(float* d, uint32_t a0, uint32_t a1, uint32_t a2,
                                                            uint32_t a3, uint32_t b0, uint32_t b1) {
#if CRIT_KCORE_HAS_BLOCK_SCALED_MMA
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10}, {%11}, {%12,%13}, {%14}, {%15,%16};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f), "r"(kE8M0Ones), "h"((uint16_t)0),
          "h"((uint16_t)0), "r"(kE8M0Ones), "h"((uint16_t)0), "h"((uint16_t)0));
#else
    mma_e4m3_m16n8k32_zero(d, a0, a1, a2, a3, b0, b1);
#endif
}

}  // namespace crit::kcore
