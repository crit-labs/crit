// GGUF Q4_0, Q8_0, Q4_K, Q5_K and Q6_K blocks, decoded on the host and on the device with gguf-py's operations in
// gguf-py's order (quants.dequantize), each FP32 multiply and subtract rounded once and never fused, so both equal
// gguf-py bit for bit. A matrix stores each row as consecutive blocks of `values` elements.
#pragma once
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define GGUF_HD __host__ __device__ __forceinline__
#else
#define GGUF_HD inline
#endif

namespace crit::gguf_blocks {

enum class Kind : int { q4_0 = 0, q8_0 = 1, q4_k = 2, q5_k = 3, q6_k = 4 };
struct Format { int values, bytes; };

GGUF_HD constexpr Format format(Kind kind) {
    return kind == Kind::q4_0 ? Format{32, 18} : kind == Kind::q8_0 ? Format{32, 34} : kind == Kind::q4_k ? Format{256, 144}
         : kind == Kind::q5_k ? Format{256, 176} : Format{256, 210};
}

// Exact IEEE half to single conversion (subnormals, infinities and NaNs included).
GGUF_HD float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16, exponent = (h >> 10) & 31u, mantissa = h & 0x3ffu;
    std::uint32_t bits;
    if(exponent == 31) bits = sign | 0x7f800000u | (mantissa << 13);
    else if(exponent) bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    else if(!mantissa) bits = sign;
    else {
        int shift = 0;
        std::uint32_t m = mantissa;
        while(!(m & 0x400u)) { m <<= 1; ++shift; }
        bits = sign | (std::uint32_t(113 - shift) << 23) | ((m & 0x3ffu) << 13);
    }
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
#endif
}

// One FP32 rounding per operation, as numpy's element-wise float32 arithmetic does.
GGUF_HD float mul(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    volatile float r = a * b;
    return r;
#endif
}
GGUF_HD float sub(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fsub_rn(a, b);
#else
    volatile float r = a - b;
    return r;
#endif
}
GGUF_HD std::uint16_t load16(const std::uint8_t* p) { return std::uint16_t(p[0] | (p[1] << 8)); }

// The 6-bit scale and min of sub-block j of a K-quant block from its 12 packed bytes.
GGUF_HD void scale_min(const std::uint8_t* s, int j, int& sc, int& m) {
    if(j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
    else { sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
}

// Values [first, first + 8) of block `b`; first is a multiple of 8 below the format's values.
GGUF_HD void decode8(Kind kind, const std::uint8_t* b, int first, float* out) {
    switch(kind) {
    case Kind::q8_0: {
        const float d = half_to_float(load16(b));
        for(int t = 0; t < 8; ++t) out[t] = mul(d, float(static_cast<std::int8_t>(b[2 + first + t])));
        return;
    }
    case Kind::q4_0: {
        const float d = half_to_float(load16(b));
        for(int t = 0; t < 8; ++t) {
            const int e = first + t, q = e < 16 ? (b[2 + e] & 15) : (b[2 + e - 16] >> 4);
            out[t] = mul(d, float(q - 8));
        }
        return;
    }
    case Kind::q4_k: case Kind::q5_k: {
        const int j = first / 32, l = first % 32, i = j / 2, h = j % 2;
        int sc = 0, m = 0;
        scale_min(b + 4, j, sc, m);
        const float dj = mul(half_to_float(load16(b)), float(sc)), mj = mul(half_to_float(load16(b + 2)), float(m));
        const std::uint8_t* qs = b + (kind == Kind::q4_k ? 16 : 48);
        for(int t = 0; t < 8; ++t) {
            int q = (qs[32 * i + l + t] >> (4 * h)) & 15;
            if(kind == Kind::q5_k) q |= ((b[16 + l + t] >> j) & 1) << 4;
            out[t] = sub(mul(dj, float(q)), mj);
        }
        return;
    }
    case Kind::q6_k: {
        const int row = first / 32, col = first % 32, a = row / 4, t4 = row % 4;
        const float ds = mul(half_to_float(load16(b + 208)), float(static_cast<std::int8_t>(b[192 + first / 16])));
        for(int t = 0; t < 8; ++t) {
            const int low = (b[64 * a + 32 * (t4 % 2) + col + t] >> (4 * (t4 / 2))) & 15;
            const int high = (b[128 + 32 * a + col + t] >> (2 * t4)) & 3;
            out[t] = mul(ds, float((low | (high << 4)) - 32));
        }
        return;
    }
    }
}

}  // namespace crit::gguf_blocks
