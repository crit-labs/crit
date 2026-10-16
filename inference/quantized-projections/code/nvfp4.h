#pragma once
// NVFP4 weight codec and scalar batched projection (header-only C++20).
// value = e2m1(code) * e4m3(block_scale) / global_scale, one E4M3FN scale per 16
// values, codes packed "adjacent" (low nibble = even index). Scalar reference: each
// 16-value block is decoded once and reused across all token rows; FP32
// accumulation in a fixed order. Optimized ISA providers belong in Kernels.
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace crit::nvfp4 {

inline constexpr std::size_t block = 16;
inline constexpr std::array<float, 8> e2m1_magnitude = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

inline float e2m1(std::uint8_t code) {
    const float m = e2m1_magnitude[code & 7];
    return (code & 8) ? -m : m;
}
inline float e4m3(std::uint8_t byte) {
    const int exponent = (byte >> 3) & 15, mantissa = byte & 7;
    const float sign = (byte & 0x80) ? -1.0f : 1.0f;
    if(exponent == 15 && mantissa == 7) return std::numeric_limits<float>::quiet_NaN();
    if(exponent == 0) return sign * std::ldexp(float(mantissa) / 8.0f, -6);
    return sign * std::ldexp(1.0f + float(mantissa) / 8.0f, exponent - 7);
}
namespace detail {
template <class Table>
inline std::uint8_t nearest(float magnitude, const Table& table, std::size_t count) {
    std::size_t best = 0;
    for(std::size_t i = 1; i < count; ++i) {
        const float d = std::fabs(magnitude - table(i)), bd = std::fabs(magnitude - table(best));
        if(d < bd || (d == bd && i % 2 == 0 && best % 2 == 1)) best = i;
    }
    return std::uint8_t(best);
}
}
// Round to nearest even, saturating (E4M3 at 448, E2M1 at 6). NaN throws.
inline std::uint8_t e4m3_encode(float x) {
    if(std::isnan(x)) throw std::invalid_argument("NaN scale");
    const auto code = detail::nearest(std::fmin(std::fabs(x), 448.0f), [](std::size_t i) { return e4m3(std::uint8_t(i)); }, 0x7F);
    return std::uint8_t(code | (x < 0 && code ? 0x80 : 0));
}
inline std::uint8_t e2m1_encode(float x) {
    if(std::isnan(x)) throw std::invalid_argument("NaN value");
    const auto code = detail::nearest(std::fmin(std::fabs(x), 6.0f), [](std::size_t i) { return e2m1_magnitude[i]; }, 8);
    return std::uint8_t(code | (x < 0 && code ? 8 : 0));
}

struct Matrix {
    std::size_t rows = 0, cols = 0;            // cols is a multiple of 16
    std::vector<std::uint8_t> packed;          // [rows][cols / 2]
    std::vector<std::uint8_t> scales;          // [rows][cols / 16] E4M3FN bytes
    float global_scale = 1.0f;                 // divisor

    void validate() const {
        if(cols == 0 || cols % block || packed.size() != rows * cols / 2 || scales.size() != rows * cols / block)
            throw std::invalid_argument("NVFP4 geometry mismatch");
        if(!std::isfinite(global_scale) || global_scale <= 0) throw std::invalid_argument("global scale must be positive and finite");
    }
    // Decodes one 16-value block of row n into out[16].
    void decode_block(std::size_t n, std::size_t b, float* out) const {
        const float s = e4m3(scales[n * (cols / block) + b]) / global_scale;
        const std::uint8_t* p = packed.data() + n * (cols / 2) + b * (block / 2);
        for(std::size_t i = 0; i < block / 2; ++i) {
            out[2 * i] = e2m1(p[i] & 15) * s;
            out[2 * i + 1] = e2m1(p[i] >> 4) * s;
        }
    }
    // Quantizes a dense row-major matrix with a tensor-wide global scale.
    static Matrix quantize(std::span<const float> dense, std::size_t rows, std::size_t cols) {
        if(dense.size() != rows * cols || cols == 0 || cols % block) throw std::invalid_argument("NVFP4 geometry mismatch");
        float amax = 0;
        for(float v : dense) {
            if(!std::isfinite(v)) throw std::invalid_argument("non-finite weight");
            amax = std::fmax(amax, std::fabs(v));
        }
        Matrix m{rows, cols, std::vector<std::uint8_t>(rows * cols / 2), std::vector<std::uint8_t>(rows * cols / block),
                 amax > 0 ? 6.0f * 448.0f / amax : 1.0f};
        for(std::size_t n = 0; n < rows; ++n)
            for(std::size_t b = 0; b < cols / block; ++b) {
                const float* x = dense.data() + n * cols + b * block;
                float bmax = 0;
                for(std::size_t i = 0; i < block; ++i) bmax = std::fmax(bmax, std::fabs(x[i]));
                const auto s = e4m3_encode(bmax * m.global_scale / 6.0f);
                m.scales[n * (cols / block) + b] = s;
                const float decoded = e4m3(s);
                for(std::size_t i = 0; i < block; ++i) {
                    const auto code = decoded > 0 ? e2m1_encode(x[i] * m.global_scale / decoded) : std::uint8_t(0);
                    auto& byte = m.packed[n * (cols / 2) + b * (block / 2) + i / 2];
                    byte = std::uint8_t(i % 2 ? (byte | (code << 4)) : code);
                }
            }
        return m;
    }
};

// y[t * ldy + n] = sum_k W[n][k] * x[t * ldx + k] for t < tokens. Each block is
// decoded once per row and reused for every token.
inline void project(const Matrix& w, std::span<const float> x, std::size_t ldx, std::size_t tokens, std::span<float> y, std::size_t ldy) {
    w.validate();
    if(ldx < w.cols || ldy < w.rows || (tokens && (x.size() < (tokens - 1) * ldx + w.cols || y.size() < (tokens - 1) * ldy + w.rows)))
        throw std::invalid_argument("projection geometry mismatch");
    float decoded[block];
    for(std::size_t n = 0; n < w.rows; ++n) {
        for(std::size_t t = 0; t < tokens; ++t) y[t * ldy + n] = 0.0f;
        for(std::size_t b = 0; b < w.cols / block; ++b) {
            w.decode_block(n, b, decoded);
            for(std::size_t t = 0; t < tokens; ++t) {
                const float* xt = x.data() + t * ldx + b * block;
                float acc = 0.0f;
                for(std::size_t i = 0; i < block; ++i) acc += decoded[i] * xt[i];
                y[t * ldy + n] += acc;
            }
        }
    }
}

} // namespace crit::nvfp4
