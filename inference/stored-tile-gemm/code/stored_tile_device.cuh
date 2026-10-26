// Device helpers shared by the stored-tile kernels (internal to this entry): weight-tile loads, cp.async, the gated
// MLP activation, exact E4M3 / E2M1 decoding of tile bytes into BF16 MMA fragments, and the BF16 MMA.
#pragma once
#include "stored_tile_gemm.h"

#include <cuda_fp16.h>

#include <cstdint>

namespace crit::stored_tile::detail {

// Weights are read once per launch: no L1 allocation.
__device__ __forceinline__ uint4 load_streaming(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 load_streaming8(const void* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N_)); }

// The gated MLP activation (see Activation). The GELU forms avoid the cancellation of 1 + tanh(u) and 1 + erf(z) for
// negative v: 0.5 (1 + tanh u) = 1 / (1 + e^-2u) and 1 + erf(z) = erfc(-z), so small results keep their relative
// precision. The divisions use __fdividef (at most 2 FP32 ulp; the IEEE division's slow-path branch cost 6% of a
// long-prompt FP8 gated GEMM on one sm_120 device), far below the BF16 output's resolution.
__device__ __forceinline__ float activate(int act, float v) {
    if(act == static_cast<int>(Activation::silu)) return __fdividef(v, 1.f + expf(-v));
    if(act == static_cast<int>(Activation::gelu_tanh))
        return __fdividef(v, 1.f + expf(-1.5957691216057308f * (v + 0.044715f * v * v * v)));
    return 0.5f * v * erfcf(-0.7071067811865476f * v);
}
__device__ __forceinline__ __nv_bfloat16 gated_value(int act, float gate, float up) {
    return __float2bfloat16_rn(activate(act, gate) * up);
}

// ------------------------------------------------------------------------------------------- exact decoding
// E4M3 and E2M1 values, and E2M1 x E4M3 products (at most 6 significant bits, magnitudes in [2^-10, 2688]), are normal
// FP16 and BF16 numbers, so every conversion below is exact. NaN codes (E4M3 0x7F / 0xFF) are not valid weights.
__device__ __forceinline__ std::uint32_t hmul2_u32(std::uint32_t a, std::uint32_t b) {
    std::uint32_t d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
// Four E4M3 bytes [b0 b1 b2 b3] -> lo = f16x2 (b0, b1), hi = f16x2 (b2, b3).
__device__ __forceinline__ void e4m3x4_to_f16x2(std::uint32_t w, std::uint32_t& lo, std::uint32_t& hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %2;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\tcvt.rn.f16x2.e4m3x2 %1, h;\n\t}"
        : "=r"(lo), "=r"(hi) : "r"(w));
#else
    // Software decode: the 7 magnitude bits shifted into an FP16's exponent and mantissa fields are the value times
    // 2^-8 (E4M3 subnormals become FP16 subnormals), so one exact FP16 multiply by 256 gives it; then the sign.
    const std::uint32_t m = w & 0x7F7F7F7Fu, s = w & 0x80808080u;
    lo = hmul2_u32(((m << 7) & 0x3F80u) | ((m << 15) & 0x3F800000u), 0x5C005C00u) | ((s << 8) & 0x8000u) | ((s << 16) & 0x80000000u);
    hi = hmul2_u32(((m >> 9) & 0x3F80u) | ((m >> 1) & 0x3F800000u), 0x5C005C00u) | ((s >> 8) & 0x8000u) | (s & 0x80000000u);
#endif
}
// Eight E2M1 codes (four bytes, low nibble first) -> h[i] = f16x2 (low nibble, high nibble) of byte i.
__device__ __forceinline__ void e2m1x8_to_f16x2(std::uint32_t w, std::uint32_t (&h)[4]) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\tmov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n\tcvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n\tcvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(w));
#else
    // Software decode: the FP16 high bytes of 0, 0.5, 1, 1.5, 2, 3, 4, 6 form an 8-byte table that one byte permute
    // indexes with four magnitudes at a time; the sign bits move to bits 15 and 31.
    constexpr std::uint32_t t0 = 0x3E3C3800u, t1 = 0x46444240u;
    const std::uint32_t r01 = __byte_perm(t0, t1, w & 0x7777u), r23 = __byte_perm(t0, t1, (w >> 16) & 0x7777u);
    h[0] = __byte_perm(r01, 0u, 0x1404) | ((w << 12) & 0x8000u) | ((w << 24) & 0x80000000u);
    h[1] = __byte_perm(r01, 0u, 0x3424) | ((w << 4) & 0x8000u) | ((w << 16) & 0x80000000u);
    h[2] = __byte_perm(r23, 0u, 0x1404) | ((w >> 4) & 0x8000u) | ((w << 8) & 0x80000000u);
    h[3] = __byte_perm(r23, 0u, 0x3424) | ((w >> 12) & 0x8000u) | (w & 0x80000000u);
#endif
}
__device__ __forceinline__ std::uint32_t f16x2_to_bf16x2(std::uint32_t h) {
    const float lo = __half2float(__ushort_as_half(static_cast<unsigned short>(h & 0xFFFFu)));
    const float hi = __half2float(__ushort_as_half(static_cast<unsigned short>(h >> 16)));
    const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const std::uint32_t*>(&b);
}

// A fragment (m16n8k16, BF16) of 16-column step s (0, 1) of a lane's FP8 tile words: words 2 s, 2 s + 1 hold
// [row g c, c + 1, row g + 8 c, c + 1] and the same at c + 8 (c = 16 s + 2 t).
__device__ __forceinline__ void fp8_a_frag(const std::uint32_t (&w)[4], int s, std::uint32_t (&a)[4]) {
    std::uint32_t lo, hi;
    e4m3x4_to_f16x2(w[2 * s], lo, hi);
    a[0] = f16x2_to_bf16x2(lo);
    a[1] = f16x2_to_bf16x2(hi);
    e4m3x4_to_f16x2(w[2 * s + 1], lo, hi);
    a[2] = f16x2_to_bf16x2(lo);
    a[3] = f16x2_to_bf16x2(hi);
}
// A fragment of scale group s (0..3) of a lane's NVFP4 tile words: word s holds [row g (2t, 2t+1)] [row g+8 (2t, 2t+1)]
// [row g (2t+8, 2t+9)] [row g+8 (2t+8, 2t+9)]; sc.x = row g's group scales 0..3, sc.y = row g + 8's. Each value is
// E2M1 x E4M3 group scale (the weight times weight_global_scale), exact in BF16.
__device__ __forceinline__ void fp4_a_frag(const std::uint32_t (&w)[4], uint2 sc, int s, std::uint32_t (&a)[4]) {
    std::uint32_t h[4];
    e2m1x8_to_f16x2(w[s], h);
    const std::uint32_t sel = static_cast<std::uint32_t>(s) * 0x1111u + 0x4400u;  // [S(g, s), S(g, s), S(g+8, s), S(g+8, s)]
    std::uint32_t sg, sg8;
    e4m3x4_to_f16x2(__byte_perm(sc.x, sc.y, sel), sg, sg8);
    a[0] = f16x2_to_bf16x2(hmul2_u32(h[0], sg));
    a[1] = f16x2_to_bf16x2(hmul2_u32(h[1], sg8));
    a[2] = f16x2_to_bf16x2(hmul2_u32(h[2], sg));
    a[3] = f16x2_to_bf16x2(hmul2_u32(h[3], sg8));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t a2, std::uint32_t a3,
                                         std::uint32_t b0, std::uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    __trap();
#endif
}

}  // namespace crit::stored_tile::detail
