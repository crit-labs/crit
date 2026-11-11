// crit's kernel core: tensor-core multiply wrappers per GPU generation (kernels/core/README.md, "mma"). Each wrapper
// is one instruction; the fragment layouts below are the PTX ISA's, and tests/mma_tests.cu checks every wrapper
// against a host product on the device it runs on.
//
// Warp-level mma.sync, lane = 4 gid + tig (gid = lane >> 2, tig = lane & 3), m16 x n8 tiles:
//   C/D (FP32 or INT32, 4 registers): c[i] is row gid + 8 (i >> 1), column 2 tig + (i & 1).
//   16-bit A (m16n8k16, 4 registers of 2 values): a0 row gid, k 2 tig + {0, 1}; a1 row gid + 8, same k; a2 row gid,
//     k 8 + 2 tig + {0, 1}; a3 row gid + 8, k 8 + 2 tig + {0, 1}.
//   16-bit B (k16 x n8, 2 registers): b0 column gid, k 2 tig + {0, 1}; b1 column gid, k 8 + 2 tig + {0, 1}.
//   8-bit A (s8 k16: 2 registers; s8/e4m3 k32: 4): a0 row gid, k 4 tig + {0..3}; a1 row gid + 8, same k; (k32) a2 row
//     gid, k 16 + 4 tig + {0..3}; a3 row gid + 8, same k.
//   8-bit B (k16: 1 register; k32: 2): b0 column gid, k 4 tig + {0..3}; (k32) b1 column gid, k 16 + 4 tig + {0..3}.
//   4-bit A (e2m1 k64, 4 registers of 8 codes): a0 row gid, k 8 tig + {0..7}; a1 row gid + 8; a2 row gid,
//     k 32 + 8 tig + {0..7}; a3 row gid + 8. 4-bit B: b0 column gid, k 8 tig + {0..7}; b1 k 32 + 8 tig + {0..7}.
//   Block scales (block-scaled kinds, byte and thread selectors 0): the lane supplies the scales of A row
//     gid + 8 (tig & 1) and of B column gid, one byte per scale block of the instruction's K, block 0 in the low byte.
// The hardware's order of the products' sum within one instruction is its own: an exact (integer, or exactly
// representable) block is order free; a kernel that promises row invariance keeps the instruction chain per output
// fixed (crit's rule, kernels/TEMPLATES.md).
#pragma once
#include "crit_kcore/arch.cuh"

#include <cstdint>

namespace crit::kcore {

// ------------------------------------------------------------------------------------------- fragment positions

__device__ __forceinline__ int mma_gid(int lane) { return lane >> 2; }
__device__ __forceinline__ int mma_tig(int lane) { return lane & 3; }
// Row and column of accumulator register i of a lane (m16 x n8 tiles).
__device__ __forceinline__ int mma_c_row(int lane, int i) { return (lane >> 2) + 8 * (i >> 1); }
__device__ __forceinline__ int mma_c_col(int lane, int i) { return 2 * (lane & 3) + (i & 1); }

// Word i (0..3) of a 16-byte vector.
__device__ __forceinline__ uint32_t word_of(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// The quad's 4 x 4 word transpose: quad lane t holds v (4 words); afterwards y[s] is word t of quad lane s. A lane that
// read 16 contiguous bytes of a row gets the words the k-major fragment layouts want, by three shuffles.
__device__ __forceinline__ void quad_transpose(const uint4& v, int t, int lane, uint32_t (&y)[4]) {
    const uint32_t own = word_of(v, t);
    y[0] = y[1] = y[2] = y[3] = own;
#pragma unroll
    for(int r = 1; r < 4; ++r) {
        const uint32_t got = __shfl_sync(0xffffffffu, word_of(v, (t + r) & 3), (lane & ~3) | ((t - r) & 3));
        const int s = (t - r) & 3;
        y[0] = s == 0 ? got : y[0];
        y[1] = s == 1 ? got : y[1];
        y[2] = s == 2 ? got : y[2];
        y[3] = s == 3 ? got : y[3];
    }
}

// ------------------------------------------------------------------------------------------- 16-bit (8.0 on)

// c += A (16 x 16 BF16) B (16 x 8 BF16), FP32 accumulation.
__device__ __forceinline__ void mma_bf16_m16n8k16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// c += A (16 x 16 FP16) B (16 x 8 FP16), FP32 accumulation.
__device__ __forceinline__ void mma_f16_m16n8k16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                 uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// c += A (16 x 8 BF16: a0 row gid, a1 row gid + 8, k 2 tig + {0, 1}) B (8 x 8: b0 column gid, k 2 tig + {0, 1}).
__device__ __forceinline__ void mma_bf16_m16n8k8(float* c, uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}

// ------------------------------------------------------------------------------------------- INT8 (8.0 on)

// c += A (16 x 16 s8) B (16 x 8 s8), INT32 accumulation (exact).
__device__ __forceinline__ void mma_s8_m16n8k16(int* c, uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}
// c = A B from a zero accumulator (one exact block sum a call).
__device__ __forceinline__ void mma_s8_m16n8k16_zero(int a0, int a1, int b, int& c0, int& c1, int& c2, int& c3) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, "
                 "{%7, %7, %7, %7};\n"
                 : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                 : "r"(a0), "r"(a1), "r"(b), "r"(0));
}
// c += A (16 x 32 s8) B (32 x 8 s8), INT32 accumulation (exact).
__device__ __forceinline__ void mma_s8_m16n8k32(int* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                                uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ------------------------------------------------------------------------------------------- FP8 E4M3 (8.9 on)

// c += A (16 x 32 E4M3) B (32 x 8 E4M3), FP32 accumulation. Traps where the instruction does not exist.
__device__ __forceinline__ void mma_e4m3_m16n8k32(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1) {
#if CRIT_KCORE_HAS_FP8_MMA
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)c, (void)a0, (void)a1, (void)a2, (void)a3, (void)b0, (void)b1;
    __trap();
#endif
}
// d = A B from a zero accumulator (one block's sum a call). Traps where the instruction does not exist.
__device__ __forceinline__ void mma_e4m3_m16n8k32_zero(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                       uint32_t b0, uint32_t b1) {
#if CRIT_KCORE_HAS_FP8_MMA
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %10, %10, %10};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
#else
    (void)a0, (void)a1, (void)a2, (void)a3, (void)b0, (void)b1;
    d[0] = d[1] = d[2] = d[3] = 0.0f;
    __trap();
#endif
}

// ------------------------------------------------------------------------------------------- block-scaled (sm_120a)
// Scales: sfa the lane's A row's (gid + 8 (tig & 1)) scale bytes, sfb its B column's (gid), block 0 in the low byte.
// Each traps where the instruction does not exist (CRIT_KCORE_HAS_BLOCK_SCALED_MMA).

// c += A (16 x 64 E2M1) B (64 x 8 E2M1) with a UE4M3 scale per 16 values of K on both (NVFP4; scale_vec::4X: four
// scale bytes a word), FP32 accumulation.
__device__ __forceinline__ void mma_nvf4_m16n8k64(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
#if CRIT_KCORE_HAS_BLOCK_SCALED_MMA
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0), "r"(sfb),
          "h"((uint16_t)0), "h"((uint16_t)0));
#else
    (void)c, (void)a0, (void)a1, (void)a2, (void)a3, (void)b0, (void)b1, (void)sfa, (void)sfb;
    __trap();
#endif
}

// c += A (16 x 64 E2M1) B (64 x 8 E2M1) with a UE8M0 scale per 32 values of K on both (MXFP4; scale_vec::2X: two
// scale bytes, the low two of the word), FP32 accumulation.
__device__ __forceinline__ void mma_mxf4_m16n8k64(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
#if CRIT_KCORE_HAS_BLOCK_SCALED_MMA
    asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0), "r"(sfb),
          "h"((uint16_t)0), "h"((uint16_t)0));
#else
    (void)c, (void)a0, (void)a1, (void)a2, (void)a3, (void)b0, (void)b1, (void)sfa, (void)sfb;
    __trap();
#endif
}

// c += A (16 x 32 E4M3) B (32 x 8 E4M3) with a UE8M0 scale per 32 values of K on both (MXFP8; scale_vec::1X: the low
// byte of the word), FP32 accumulation.
__device__ __forceinline__ void mma_mxf8_e4m3_m16n8k32(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                       uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
#if CRIT_KCORE_HAS_BLOCK_SCALED_MMA
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0), "r"(sfb),
          "h"((uint16_t)0), "h"((uint16_t)0));
#else
    (void)c, (void)a0, (void)a1, (void)a2, (void)a3, (void)b0, (void)b1, (void)sfa, (void)sfb;
    __trap();
#endif
}

// ------------------------------------------------------------------------------------------- wgmma (sm_90a)
// Warpgroup MMA: the 128 threads of 4 consecutive warps multiply A (64 x 16, shared memory) by B (16 x N, shared
// memory) into FP32 accumulators in registers, asynchronously. Thread t (warp w = t / 32 of the group, lane = 4 gid +
// tig) holds N / 2 accumulators: d[i] at row 16 w + gid + 8 ((i >> 1) & 1), column 8 (i >> 2) + 2 tig + (i & 1).
// Sequence: wgmma_fence() (after writing the accumulators or shared memory), Wgmma<N>::bf16<TA, TB>(...) per k16 step,
// wgmma_commit(), wgmma_wait<0>() before reading d. TA / TB: 0 K-major, 1 MN-major (transposed). Built for sm_90a
// only (CRIT_KCORE_HAS_WGMMA; elsewhere they trap) and compile-checked here: no sm_90 device ran them.

// A shared-memory matrix descriptor: start address, leading and stride byte offsets (multiples of 16), and the
// swizzle (0 none, 1 128B, 2 64B, 3 32B; the matrix then 1024-, 512- or 256-byte aligned).
__device__ __forceinline__ uint64_t wgmma_desc(const void* smem, uint32_t leading_bytes, uint32_t stride_bytes,
                                               int swizzle) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    return uint64_t((addr & 0x3FFFFu) >> 4) | (uint64_t((leading_bytes & 0x3FFFFu) >> 4) << 16) |
           (uint64_t((stride_bytes & 0x3FFFFu) >> 4) << 32) | (uint64_t(swizzle & 3) << 62);
}
__device__ __forceinline__ void wgmma_fence() {
#if CRIT_KCORE_HAS_WGMMA
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_commit() {
#if CRIT_KCORE_HAS_WGMMA
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
#if CRIT_KCORE_HAS_WGMMA
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#endif
}

#if CRIT_KCORE_HAS_WGMMA
template <int N>
struct Wgmma;
template <>
struct Wgmma<8> {
    template <int TA, int TB>
    __device__ __forceinline__ static void bf16(float* d, uint64_t a, uint64_t b, int accumulate) {
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
                     "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, "
                     "%4, %5, p, 1, 1, %7, %8;\n\t}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "l"(a), "l"(b), "r"(accumulate), "n"(TA), "n"(TB));
    }
};
template <>
struct Wgmma<16> {
    template <int TA, int TB>
    __device__ __forceinline__ static void bf16(float* d, uint64_t a, uint64_t b, int accumulate) {
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %10, 0;\n\t"
                     "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
                     "%8, %9, p, 1, 1, %11, %12;\n\t}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                       "+f"(d[7])
                     : "l"(a), "l"(b), "r"(accumulate), "n"(TA), "n"(TB));
    }
};
template <>
struct Wgmma<32> {
    template <int TA, int TB>
    __device__ __forceinline__ static void bf16(float* d, uint64_t a, uint64_t b, int accumulate) {
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %18, 0;\n\t"
                     "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, "
                     "%10, %11, %12, %13, %14, %15}, "
                     "%16, %17, p, 1, 1, %19, %20;\n\t}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                       "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                       "+f"(d[14]), "+f"(d[15])
                     : "l"(a), "l"(b), "r"(accumulate), "n"(TA), "n"(TB));
    }
};
template <>
struct Wgmma<64> {
    template <int TA, int TB>
    __device__ __forceinline__ static void bf16(float* d, uint64_t a, uint64_t b, int accumulate) {
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %34, 0;\n\t"
                     "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, "
                     "%10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
                     "%28, %29, %30, %31}, "
                     "%32, %33, p, 1, 1, %35, %36;\n\t}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                       "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                       "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
                       "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
                       "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                     : "l"(a), "l"(b), "r"(accumulate), "n"(TA), "n"(TB));
    }
};
template <>
struct Wgmma<128> {
    template <int TA, int TB>
    __device__ __forceinline__ static void bf16(float* d, uint64_t a, uint64_t b, int accumulate) {
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %66, 0;\n\t"
                     "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, "
                     "%9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
                     "%28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, "
                     "%46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
                     "%64, %65, p, 1, 1, %67, %68;\n\t}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                       "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                       "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
                       "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
                       "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
                       "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
                       "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
                       "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
                       "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]),
                       "+f"(d[63])
                     : "l"(a), "l"(b), "r"(accumulate), "n"(TA), "n"(TB));
    }
};
template <>
struct Wgmma<256> {
    template <int TA, int TB>
    __device__ __forceinline__ static void bf16(float* d, uint64_t a, uint64_t b, int accumulate) {
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %130, 0;\n\t"
                     "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, "
                     "%9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
                     "%28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, "
                     "%46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
                     "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
                     "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, "
                     "%100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, "
                     "%115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, "
                     "%128, %129, p, 1, 1, %131, %132;\n\t}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                       "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                       "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
                       "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
                       "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
                       "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
                       "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
                       "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
                       "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]),
                       "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]),
                       "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]),
                       "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]),
                       "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]),
                       "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]),
                       "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
                       "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]),
                       "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]),
                       "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]),
                       "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
                     : "l"(a), "l"(b), "r"(accumulate), "n"(TA), "n"(TB));
    }
};
#else
template <int N>
struct Wgmma {
    template <int TA, int TB>
    __device__ __forceinline__ static void bf16(float*, uint64_t, uint64_t, int) { __trap(); }
};
#endif

// ------------------------------------------------------------------------------------------- tcgen05 (sm_100a)
// 5th-generation MMA: one thread issues D (tensor memory) += A (shared memory) B (shared memory), asynchronously;
// tcgen05_commit arrives on an mbarrier when the issued MMAs are done; warps read D back with tcgen05_ld. Tensor
// memory is allocated in columns (a power of two >= 32) by one warp. Built for the sm_100 family's specific targets
// only (CRIT_KCORE_HAS_TCGEN05; elsewhere they trap) and compile-checked here: no sm_100 device ran them. Descriptor
// layouts after CUTLASS (BSD-3-Clause: cute's UMMA descriptors).

// A shared-memory matrix descriptor (sm_100): start address, leading and stride byte offsets (multiples of 16),
// version 1, and the layout (0 none, 2 128B swizzle, 4 64B, 6 32B, 1 128B with 32B atoms).
__device__ __forceinline__ uint64_t umma_desc(const void* smem, uint32_t leading_bytes, uint32_t stride_bytes,
                                              int layout) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    return uint64_t((addr & 0x3FFFFu) >> 4) | (uint64_t((leading_bytes & 0x3FFFFu) >> 4) << 16) |
           (uint64_t((stride_bytes & 0x3FFFFu) >> 4) << 32) | (uint64_t(1) << 46) | (uint64_t(layout & 7) << 61);
}
// The instruction descriptor of kind::f16 with FP32 accumulation: A and B BF16 (or FP16), M x N (M 64 or 128, N a
// multiple of 8 (M 64) or 16 up to 256), A / B K-major (0) or MN-major (1).
__host__ __device__ __forceinline__ constexpr uint32_t umma_idesc_f16(int m, int n, bool bf16 = true, int a_major = 0,
                                                                       int b_major = 0) {
    return (1u << 4) | (uint32_t(bf16) << 7) | (uint32_t(bf16) << 10) | (uint32_t(a_major) << 15) |
           (uint32_t(b_major) << 16) | (uint32_t(n >> 3) << 17) | (uint32_t(m >> 4) << 24);
}

// Allocate `cols` tensor-memory columns (one warp, all lanes); the address lands in *dst (shared memory).
__device__ __forceinline__ void tcgen05_alloc(uint32_t* dst, uint32_t cols) {
#if CRIT_KCORE_HAS_TCGEN05
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                 "r"(cols)
                 : "memory");
#else
    (void)dst, (void)cols;
    __trap();
#endif
}
__device__ __forceinline__ void tcgen05_dealloc(uint32_t taddr, uint32_t cols) {
#if CRIT_KCORE_HAS_TCGEN05
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(taddr), "r"(cols) : "memory");
#endif
}
__device__ __forceinline__ void tcgen05_relinquish() {
#if CRIT_KCORE_HAS_TCGEN05
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void tcgen05_fence_before() {
#if CRIT_KCORE_HAS_TCGEN05
    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void tcgen05_fence_after() {
#if CRIT_KCORE_HAS_TCGEN05
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
#endif
}
// D[taddr] (+)= A B: kind::f16 from two shared-memory descriptors (one thread issues).
__device__ __forceinline__ void tcgen05_mma_f16(uint32_t d_taddr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc,
                                                int accumulate) {
#if CRIT_KCORE_HAS_TCGEN05
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(d_taddr),
                 "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate)
                 : "memory");
#else
    (void)d_taddr, (void)a_desc, (void)b_desc, (void)idesc, (void)accumulate;
    __trap();
#endif
}
// Arrive on `bar` (shared memory) once the MMAs this thread issued are done.
__device__ __forceinline__ void tcgen05_commit(uint64_t* bar) {
#if CRIT_KCORE_HAS_TCGEN05
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(bar)))
                 : "memory");
#endif
}
// 8 consecutive 32-bit columns of the warp's 32 lanes of tensor memory (lane i of the warp: TMEM lane 32 (warp % 4) +
// i) from taddr; tcgen05_wait_ld() before using them.
__device__ __forceinline__ void tcgen05_ld_32x32b_x8(uint32_t taddr, uint32_t (&r)[8]) {
#if CRIT_KCORE_HAS_TCGEN05
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "r"(taddr));
#else
    (void)taddr;
    for(int i = 0; i < 8; ++i) r[i] = 0u;
    __trap();
#endif
}
__device__ __forceinline__ void tcgen05_wait_ld() {
#if CRIT_KCORE_HAS_TCGEN05
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
#endif
}

}  // namespace crit::kcore
