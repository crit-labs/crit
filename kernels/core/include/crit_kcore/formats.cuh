// crit's kernel core: number formats (kernels/core/README.md, "formats"). Exact decoders of the stored formats crit's
// kernels read (BF16/FP16 pairs, E4M3, E8M0, E2M1 (NVFP4 / MXFP4), INT4 groups, 2 x E2M1 as integers), the encoders
// and quantizer steps crit declares, and the block-scale layouts (row major and cuBLASLt's 128 x 4 tiles).
//
// Packing conventions: a 32-bit word of 16-bit values holds element 2i in its low half; a byte of 4-bit codes holds
// element 2i in its low nibble; a word of 8-bit codes holds element i in byte i. Every decoder here is exact (the
// value it gives is the format's value, no rounding) unless its comment says otherwise.
#pragma once
#include "crit_kcore/arch.cuh"

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace crit::kcore {

// ------------------------------------------------------------------------------------------- 16-bit pairs

__device__ __forceinline__ uint32_t u32_of(__nv_bfloat162 v) { return *reinterpret_cast<const uint32_t*>(&v); }
__device__ __forceinline__ uint32_t u32_of(__half2 v) { return *reinterpret_cast<const uint32_t*>(&v); }
__device__ __forceinline__ __nv_bfloat162 bf162_of(uint32_t v) { return *reinterpret_cast<const __nv_bfloat162*>(&v); }
__device__ __forceinline__ __half2 half2_of(uint32_t v) { return *reinterpret_cast<const __half2*>(&v); }

// Two floats as a BF16 pair (round to nearest even), lo in the low half.
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) { return u32_of(__floats2bfloat162_rn(lo, hi)); }
// The BF16 halves of a word as floats (exact).
__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// A 16-bit value as float, and a float as a 16-bit value (round to nearest even); float passes through.
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half_rn(v); }

// ------------------------------------------------------------------------------------------- E4M3 (FP8)

// An E4M3 byte's value from its bits as FP16 (the value times 2^-8, subnormals included: both formats' subnormals have
// exponent field 0), so the conversion is exact on every architecture. The NaN codes 0x7F / 0xFF read as +-480: use it
// for scales, which never hold NaN.
__device__ __forceinline__ float e4m3_value(uint32_t b) {
    const unsigned short h = static_cast<unsigned short>(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8));
    return __half2float(__ushort_as_half(h)) * 256.0f;
}

// An E4M3 byte's value by the CUDA conversion (hardware from 8.9 on): exact, NaN codes as NaN.
__device__ __forceinline__ float e4m3_cvt(uint32_t b) {
    __nv_fp8_e4m3 v;
    v.__x = static_cast<__nv_fp8_storage_t>(b & 0xffu);
    return float(v);
}

// Two E4M3 bytes (the low 16 bits, low byte first) as an FP16 pair (exact; NaN codes as NaN).
__device__ __forceinline__ __half2 e4m3x2_to_half2(uint32_t two) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two & 0xffffu), __NV_E4M3);
    return *reinterpret_cast<const __half2*>(&h);
}

// Four E4M3 bytes of a word as two BF16 pairs, exactly: the conversion from 8.9 on (NaN codes as NaN), e4m3_value
// before (NaN codes as +-480).
__device__ __forceinline__ void e4m3x4_to_bf16x2(uint32_t w, uint32_t* out) {
    float f[4];
#if CRIT_KCORE_HAS_FP8_MMA
    const float2 a = __half22float2(e4m3x2_to_half2(w & 0xFFFFu)), b = __half22float2(e4m3x2_to_half2(w >> 16));
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
#else
#pragma unroll
    for(int i = 0; i < 4; ++i) f[i] = e4m3_value((w >> (8 * i)) & 0xFFu);
#endif
    out[0] = pack_bf16x2(f[0], f[1]);
    out[1] = pack_bf16x2(f[2], f[3]);
}

// The nearest E4M3 code of a non-negative finite x, ties to the even code, saturating at 448 (0x7E): crit's NVFP4 block
// scale encoder, integer arithmetic only (the same codes as e4m3_encode_satfinite for x >= 0, tests/formats_tests.cu).
__device__ __forceinline__ uint32_t e4m3_encode(float x) {
    x = fminf(x, 448.0f);
    if(x < 0.015625f) return static_cast<uint32_t>(__float2int_rn(__fmul_rn(x, 512.0f)));   // the subnormal grid
    const uint32_t bits = __float_as_uint(x);
    const uint32_t keep = (bits >> 20) & 7u, rest = bits & 0xFFFFFu;
    uint32_t code = ((((bits >> 23) & 0xFFu) - 120u) << 3) | keep;
    if(rest > 0x80000u || (rest == 0x80000u && (keep & 1u))) ++code;
    return code > 0x7Eu ? 0x7Eu : code;
}

// The nearest E4M3 code of v (signed), ties to even, saturating to +-448, by the CUDA conversion.
__device__ __forceinline__ uint32_t e4m3_encode_satfinite(float v) {
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// Two floats as E4M3 codes (lo in the low byte), nearest, ties to even, saturating to +-448: one cvt from 8.9 on.
__device__ __forceinline__ uint32_t e4m3x2_encode_satfinite(float lo, float hi) {
    return __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
}

// ------------------------------------------------------------------------------------------- E8M0 (MX scales)

// An E8M0 byte as its FP32 scale 2^(e - 127): byte 0 is 2^-127 (an FP32 subnormal), byte 255 NaN.
__device__ __forceinline__ float e8m0_value(uint32_t e) {
    e &= 0xFFu;
    return __uint_as_float(e == 0u ? 0x00400000u : e == 255u ? 0x7fc00000u : e << 23);
}

// An E8M0 byte as a BF16 pair of 2^(byte - 127) (byte 0 is 2^-127, a BF16 subnormal; byte 255 NaN).
__device__ __forceinline__ __nv_bfloat162 e8m0_bf16x2(uint32_t b) {
    b &= 0xFFu;
    const uint32_t bits = b == 0u ? 0x0040u : b == 255u ? 0x7FC0u : b << 7;
    return bf162_of(bits | (bits << 16));
}

// ------------------------------------------------------------------------------------------- E2M1 (FP4)

// An E2M1 code's value (bit 3 the sign) by its bit pattern in FP32.
__device__ __forceinline__ float e2m1_value(uint32_t c) {
    const uint32_t e = (c >> 1) & 3u, f = c & 1u;
    const uint32_t bits = e ? ((e + 126u) << 23) | (f << 22) : (f ? (126u << 23) : 0u);
    return __uint_as_float(bits | ((c & 8u) << 28));
}

// The magnitude of a 3-bit E2M1 magnitude code: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
__device__ __forceinline__ float e2m1_magnitude(int code) {
    return code < 4 ? 0.5f * float(code) : code < 6 ? float(code - 2) : code == 6 ? 4.0f : 6.0f;
}

// The nearest E2M1 magnitude code of a >= 0, ties to the even code, 7 past 5 (NaN: 0).
__device__ __forceinline__ int e2m1_code(float a) {
    return int(a > 0.25f) + int(a >= 0.75f) + int(a > 1.25f) + int(a >= 1.75f) + int(a > 2.5f) + int(a >= 3.5f) +
           int(a > 5.0f);
}

// The nearest E2M1 code of v, ties to the even code, saturating at 6, sign bit 3 set whenever v < 0 (a small negative
// v gives 8, -0); NaN: 0.
__device__ __forceinline__ uint32_t e2m1_code_signed(float v) {
    return static_cast<uint32_t>(e2m1_code(fabsf(v))) | (v < 0.f ? 8u : 0u);
}

// Two finite floats (already over their block's scale) as a byte of E2M1 codes, lo in the low nibble: nearest, ties to
// the even code, saturating at +-6, sign bit 3 the float's sign bit (-0 and small negatives give 8). One
// cvt.rn.satfinite.e2m1x2 where it exists, the same codes by e2m1_code elsewhere (tests/formats_tests.cu).
__device__ __forceinline__ uint32_t e2m1x2_encode(float lo, float hi) {
#if CRIT_KCORE_HAS_E2M1_CVT
    uint32_t r;
    asm("{\n\t.reg .b8 t;\n\tcvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n\tcvt.u32.u8 %0, t;\n\t}" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
#else
    const uint32_t a = uint32_t(e2m1_code(fabsf(lo))) | (signbit(lo) ? 8u : 0u);
    const uint32_t b = uint32_t(e2m1_code(fabsf(hi))) | (signbit(hi) ? 8u : 0u);
    return a | (b << 4);
#endif
}

// The value of e2m1(code) * 2^-14 is the FP16 number whose bits are the code's sign at bit 15 and magnitude at bits
// 11..9: the unit e2m1x8_to_half2 gives its values in where the conversion instruction does not exist.
__device__ __forceinline__ constexpr float e2m1_half2_unit() { return CRIT_KCORE_HAS_E2M1_CVT ? 1.0f : 1.0f / 16384.0f; }

// A word's eight E2M1 codes as four FP16 pairs h[0..3] (pair b: the two codes of byte b, low nibble first), equal to
// the codes' values times e2m1_half2_unit(): one cvt.rn.f16x2.e2m1x2 per byte where it exists (unit 1), else the bit
// pattern (unit 2^-14). Exact either way; fold the unit's inverse into a scale.
__device__ __forceinline__ void e2m1x8_to_half2(uint32_t word, uint32_t* h) {
#if CRIT_KCORE_HAS_E2M1_CVT
    asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(word));
#else
#pragma unroll
    for(int b = 0; b < 4; ++b) {
        const uint32_t byte = (word >> (8 * b)) & 0xffu;
        const uint32_t lo = ((byte & 7u) << 9) | ((byte & 8u) << 12);
        const uint32_t hi = (((byte >> 4) & 7u) << 9) | (((byte >> 4) & 8u) << 12);
        h[b] = lo | (hi << 16);
    }
#endif
}

// A word's eight E2M1 codes as values w[0..7] (element order) times `scale`: e2m1x8_to_half2, widened, times scale
// (pass the block scale times 1 / e2m1_half2_unit()). One FP32 multiply a value: exact when the product is.
__device__ __forceinline__ void e2m1x8_to_float(uint32_t word, float scale, float* w) {
    uint32_t h[4];
    e2m1x8_to_half2(word, h);
#pragma unroll
    for(int b = 0; b < 4; ++b) {
        const float2 f = __half22float2(half2_of(h[b]));
        w[2 * b] = f.x * scale;
        w[2 * b + 1] = f.y * scale;
    }
}

// Eight E2M1 codes of a word as four BF16 pairs, exactly, on every architecture: two byte permutations per four
// codes (the magnitude's high and low BF16 bytes from 8-entry tables, the sign from bit 3).
__device__ __forceinline__ void e2m1x8_to_bf16x2(uint32_t w, uint32_t* out) {
    constexpr uint32_t kHiA = 0x3F3F3F00u, kHiB = 0x40404040u, kLoA = 0xC0800000u, kLoB = 0xC0804000u;
#pragma unroll
    for(int half = 0; half < 2; ++half) {
        const uint32_t v = w >> (16 * half), sel = v & 0x7777u;
        const uint32_t hi = __byte_perm(kHiA, kHiB, sel), lo = __byte_perm(kLoA, kLoB, sel);
        out[2 * half] = __byte_perm(lo, hi, 0x5140) | ((v << 12) & 0x8000u) | ((v << 24) & 0x80000000u);
        out[2 * half + 1] = __byte_perm(lo, hi, 0x7362) | ((v << 4) & 0x8000u) | ((v << 16) & 0x80000000u);
    }
}

// A byte's two E2M1 codes (low nibble first) times an FP32 block scale, as a BF16 pair (exact when the products fit
// BF16, as NVFP4's do: at most 6 significant bits).
__device__ __forceinline__ uint32_t e2m1x2_to_bf16x2_scaled(uint32_t byte, float scale) {
#if CRIT_KCORE_HAS_E2M1_CVT
    uint32_t h;
    asm("{\n\t.reg .b8 t0, t1, t2, t3;\n\tmov.b32 {t0, t1, t2, t3}, %1;\n\tcvt.rn.f16x2.e2m1x2 %0, t0;\n\t}"
        : "=r"(h) : "r"(byte));
    const float2 f = __half22float2(half2_of(h));
    return pack_bf16x2(f.x * scale, f.y * scale);
#else
    return pack_bf16x2(e2m1_value(byte & 15u) * scale, e2m1_value((byte >> 4) & 15u) * scale);
#endif
}

// A word's eight E2M1 codes times the block scale (`scale` in FP32, `scale2` the same in FP16 on both halves) as four
// BF16 pairs: where the conversion exists, one unpack and four conversions to FP16, the scale applied in FP16 (exact
// for NVFP4: at most 6 significant bits, 2^-10 to 2688), widened to BF16; elsewhere by the bit pattern in FP32.
__device__ __forceinline__ void e2m1x8_to_bf16x2_scaled(uint32_t word, float scale, uint32_t scale2, uint32_t* out) {
#if CRIT_KCORE_HAS_E2M1_CVT
    uint32_t h[4];
    e2m1x8_to_half2(word, h);
#pragma unroll
    for(int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(__hmul2(half2_of(h[i]), half2_of(scale2)));
        out[i] = pack_bf16x2(f.x, f.y);
    }
#else
    (void)scale2;
#pragma unroll
    for(int i = 0; i < 4; ++i) out[i] = e2m1x2_to_bf16x2_scaled((word >> (8 * i)) & 0xffu, scale);
#endif
}

// ------------------------------------------------------------------------------------------- INT4 groups

// A two's complement nibble's value.
__device__ __forceinline__ int int4_value(uint32_t code) { return int(code << 28) >> 28; }

// Eight INT4 codes (two's complement nibbles) times a BF16 scale pair as four BF16 pairs: (128 + (code ^ 8)) is exact
// in BF16 and fma(that, s, -136 s) = value * s exactly (the product needs at most 7 significant bits).
__device__ __forceinline__ void int4x8_to_bf16x2_scaled(uint32_t w, __nv_bfloat162 s, uint32_t* out) {
    const uint32_t x = w ^ 0x88888888u;
    const __nv_bfloat162 bias = __hmul2(s, __floats2bfloat162_rn(-136.0f, -136.0f));
#pragma unroll
    for(int p = 0; p < 4; ++p) {
        const uint32_t byte = (x >> (8 * p)) & 0xFFu;
        const uint32_t word = 0x43004300u | (byte & 0xFu) | ((byte & 0xF0u) << 12);
        out[p] = u32_of(__hfma2(bf162_of(word), s, bias));
    }
}

// ------------------------------------------------------------------------------------------- INT4 with a zero point

// Unsigned 4-bit codes with a zero point and a scale per group of inputs (AWQ, GPTQ and other asymmetric INT4
// groups): a value is (code - zero) * scale, the scale FP16 as the checkpoints store it, the zero an integer 0 to 16
// (GPTQ's +1 convention can give 16). |code - zero| <= 16 needs at most 4 significant bits, the scale 11, so the value
// is exact in FP32, and so is its product with a BF16 activation (8 bits more).

// Nibble i of a word packed in sequence (element i in bits 4i .. 4i + 3): crit's rows of 4-bit codes, GPTQ's qweight
// along K and its qzeros along N.
__host__ __device__ __forceinline__ uint32_t nibble(uint32_t word, int i) { return (word >> (4 * i)) & 15u; }

// AWQ packs eight values a 32-bit word in the order 0, 2, 4, 6, 1, 3, 5, 7 (nibble p holds element order[p], its
// qweight [K][N / 8] and qzeros [K / group][N / 8] along N): the nibble that holds element j, element j, and the word
// in sequence (element j in nibble j).
__host__ __device__ __forceinline__ constexpr int awq_nibble(int j) { return (j >> 1) + 4 * (j & 1); }
__host__ __device__ __forceinline__ uint32_t awq_code(uint32_t word, int j) { return nibble(word, awq_nibble(j)); }
__host__ __device__ __forceinline__ uint32_t awq_to_sequential(uint32_t word) {
    uint32_t out = 0;
#pragma unroll
    for(int j = 0; j < 8; ++j) out |= awq_code(word, j) << (4 * j);
    return out;
}

// GPTQ's zero point from its stored nibble: stored + offset (offset 1 for checkpoints that store zero - 1, GPTQ's
// original format; 0 for those that store the zero itself).
__host__ __device__ __forceinline__ int gptq_zero(uint32_t stored, int offset) { return int(stored) + offset; }
// The group input k reads: g_idx[k] with GPTQ's act order (a permuted group per input index), else k / group.
__host__ __device__ __forceinline__ int gptq_group(const int* g_idx, int k, int group) {
    return g_idx ? g_idx[k] : k / group;
}

// An FP16 scale's bits as FP32 (exact).
__device__ __forceinline__ float f16_value(uint32_t bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits & 0xFFFFu)));
}

// The exact FP32 value (code - zero) * scale.
__device__ __forceinline__ float int4_zp_value(uint32_t code, int zero, float scale) {
    return __fmul_rn(float(int(code & 15u) - zero), scale);
}

// Eight codes of a word packed in sequence as values w[0..7]: (code - zero) * scale each, exact. The code is read from
// its FP32 bit pattern above 2^23 (0x4B000000 | code is 2^23 + code), minus 2^23 + zero (exact: integers below 2^24),
// times the scale: two FP32 operations a value.
__device__ __forceinline__ void int4x8_zp_to_float(uint32_t word, int zero, float scale, float* w) {
    const float bias = 8388608.0f + float(zero);
#pragma unroll
    for(int i = 0; i < 8; ++i) w[i] = __fmul_rn(__fsub_rn(__uint_as_float(0x4B000000u | nibble(word, i)), bias), scale);
}

// Eight codes of a word packed in sequence as BF16 pairs hi[0..3] and lo[0..3] (pair p: elements 2p, 2p + 1) with
// hi + lo = (code - zero) * scale exactly: hi the value rounded to BF16, lo the rest (the value has at most 15
// significant bits, hi keeps 8, the rest fits BF16's 8). Two BF16 MMAs of the same activations, one with hi and one
// with lo, multiply the exact value; a value that fits BF16 has lo 0.
__device__ __forceinline__ void int4x8_zp_to_bf16x2_split(uint32_t word, int zero, float scale, uint32_t* hi,
                                                          uint32_t* lo) {
    float w[8];
    int4x8_zp_to_float(word, zero, scale, w);
#pragma unroll
    for(int p = 0; p < 4; ++p) {
        const uint32_t h = pack_bf16x2(w[2 * p], w[2 * p + 1]);
        hi[p] = h;
        lo[p] = pack_bf16x2(__fsub_rn(w[2 * p], bf16_lo(h)), __fsub_rn(w[2 * p + 1], bf16_hi(h)));
    }
}

// ------------------------------------------------------------------------------------------- 4-bit codes as integers

// A 4-bit weight code as an integer: INT4 by sign extension, or E2M1 times 2 ({0, 1, 2, 3, 4, 6, 8, 12}, sign at bit
// 3) from two packed words by __byte_perm. Integer weights feed dp4a and the s8 MMA exactly.
__device__ __forceinline__ int four_bit_int(uint32_t code, bool int4) {
    if(int4) return int4_value(code);
    const int magnitude = int(__byte_perm(0x03020100u, 0x0c080604u, code & 7u) & 0xffu);
    return (code & 8u) ? -magnitude : magnitude;
}
// Four consecutive codes (16 bits, element i in nibble i) as packed int8 values (element i in byte i).
__device__ __forceinline__ int four_bit_int8x4(uint32_t pair, bool int4) {
    uint32_t packed = 0;
#pragma unroll
    for(int i = 0; i < 4; ++i) packed |= (uint32_t(four_bit_int((pair >> (4 * i)) & 15u, int4)) & 0xffu) << (8 * i);
    return int(packed);
}

// A 256-entry table in shared memory, filled by the block's threads (call __syncthreads after): byte -> the BF16 pair
// of its two codes' values (low nibble first), E2M1 or INT4. A decode then costs a lookup and a multiply a pair.
template <bool INT4>
__device__ __forceinline__ void four_bit_pair_table(uint32_t* table) {
    for(int b = threadIdx.x; b < 256; b += blockDim.x) {
        float v[2];
#pragma unroll
        for(int h = 0; h < 2; ++h) {
            const int code = (b >> (4 * h)) & 15;
            if constexpr(INT4) v[h] = float(code < 8 ? code : code - 16);
            else v[h] = (code & 8) ? -e2m1_magnitude(code & 7) : e2m1_magnitude(code & 7);
        }
        table[b] = pack_bf16x2(v[0], v[1]);
    }
}

// ------------------------------------------------------------------------------------------- NVFP4 quantizer steps

// crit's declared NVFP4 quantizer, step by step: a 16-value block of maximum magnitude amax under global scale g has
// the decoded block scale e4m3(encode(amax g / 6)) (448 where that is not finite)...
__device__ __forceinline__ float nvfp4_block_scale(float amax, float g) {
    float wanted = __fdiv_rn(__fmul_rn(amax, g), 6.0f);
    if(!isfinite(wanted)) wanted = 448.0f;
    return e4m3_value(e4m3_encode(wanted));
}
// ... and a value v of it the operand e2m1(code) * decoded, the code the nearest of (v g) / decoded (0 where the
// decoded scale is not positive).
__device__ __forceinline__ float nvfp4_operand(float v, float g, float decoded) {
    if(!(decoded > 0.0f)) return 0.0f;
    const float q = __fdiv_rn(__fmul_rn(v, g), decoded);
    const int code = e2m1_code(fabsf(q));
    const float m = __fmul_rn(e2m1_magnitude(code), decoded);
    return (q < 0.0f && code) ? -m : m;
}

// ------------------------------------------------------------------------------------------- block-scale layouts

// The byte of block-scale column j of matrix row n in cuBLASLt's blocked layout (VEC16_UE4M3 / VEC32_UE8M0): tiles of
// 128 rows x 4 columns, 512 bytes each, ct tiles a row of tiles (ceil(columns / 4)), rows padded to 128.
__host__ __device__ __forceinline__ size_t blocked_scale_offset(int n, int j, int ct) {
    const int r = n & 127;
    return ((size_t)(n >> 7) * ct + (j >> 2)) * 512 + (size_t)(r & 31) * 16 + (r >> 5) * 4 + (j & 3);
}
// Row major [rows][ks] when ct is 0, else the blocked layout.
__host__ __device__ __forceinline__ size_t scale_offset(int n, int j, int ks, int ct) {
    return ct ? blocked_scale_offset(n, j, ct) : (size_t)n * ks + j;
}
// The bytes `rows` rows of block scales take in the blocked layout (rows padded to 128, k / block columns to 4).
__host__ __device__ __forceinline__ long long blocked_scale_bytes(long long rows, long long k, long long block) {
    return (rows + 127) / 128 * 128 * ((k / block + 3) / 4 * 4);
}

}  // namespace crit::kcore
