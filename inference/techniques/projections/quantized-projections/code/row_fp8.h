#pragma once
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace crit::fp8 {
inline std::size_t extent(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::length_error("FP8 extent overflow");
  return a * b;
}
inline float bf16(std::uint16_t word) { return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16); }
inline std::uint16_t bf16_word(float value) {
  if (!std::isfinite(value)) throw std::invalid_argument("Finite BF16 input required");
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const auto result = static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
  if (!std::isfinite(bf16(result))) throw std::overflow_error("BF16 rounding overflow");
  return result;
}
inline float decode(std::uint8_t code) {
  const unsigned magnitude = code & 127U, exponent = magnitude >> 3, mantissa = magnitude & 7U;
  if (magnitude == 127) throw std::invalid_argument("NaN is not an FP8 weight");
  const float value = exponent ? std::ldexp(static_cast<float>(8U + mantissa), static_cast<int>(exponent) - 10)
                               : std::ldexp(static_cast<float>(mantissa), -9);
  return code & 128U ? -value : value;
}
// Nearest finite E4M3FN, ties to even significand; saturate beyond +/-448.
inline std::uint8_t encode(float value) {
  if (!std::isfinite(value)) throw std::invalid_argument("Finite FP8 input required");
  const auto sign = std::signbit(value) ? 128U : 0U;
  const float absolute = std::abs(value);
  unsigned best = 0;
  double distance = absolute;
  for (unsigned code = 1; code <= 126; ++code) {
    const double next = std::abs(static_cast<double>(absolute) - decode(static_cast<std::uint8_t>(code)));
    if (next < distance || (next == distance && !(code & 1U))) { best = code; distance = next; }
  }
  return static_cast<std::uint8_t>(sign | best);
}
struct RowWeights {
  std::size_t rows, width;
  std::vector<std::uint8_t> codes;
  std::vector<std::uint16_t> scales;
  void validate() const {
    if (!rows || !width || codes.size() != extent(rows, width) || scales.size() != rows)
      throw std::invalid_argument("FP8 row geometry differs");
    for (std::size_t row = 0; row < rows; ++row) {
      if ((scales[row] & 0x8000U) || !std::isfinite(bf16(scales[row])))
        throw std::invalid_argument("Nonnegative finite represented BF16 scales required");
      for (std::size_t col = 0; col < width; ++col) {
        const auto code = codes[row * width + col];
        (void)decode(code);
        if (!scales[row] && (code & 127U)) throw std::invalid_argument("Zero scale requires signed-zero codes");
      }
    }
  }
};
// Conversion policy: finite FP32 input, maxabs/448 rounded to BF16, then
// nearest-even E4M3FN. An underflowed nonzero scale is rescued to BF16 0x0001.
inline RowWeights quantize(std::span<const float> values, std::size_t rows, std::size_t width) {
  if (!rows || !width || values.size() != extent(rows, width)) throw std::invalid_argument("FP8 conversion shape differs");
  RowWeights out{rows, width, std::vector<std::uint8_t>(values.size()), std::vector<std::uint16_t>(rows)};
  for (std::size_t row = 0; row < rows; ++row) {
    float maximum = 0;
    for (float value : values.subspan(row * width, width)) {
      if (!std::isfinite(value)) throw std::invalid_argument("Finite FP8 conversion input required");
      maximum = std::max(maximum, std::abs(value));
    }
    auto scale = maximum ? bf16_word(maximum / 448.f) : std::uint16_t{0};
    if (maximum && !scale) scale = 1;
    out.scales[row] = scale;
    for (std::size_t col = 0; col < width; ++col) {
      const float value = values[row * width + col];
      out.codes[row * width + col] = encode(scale ? value / bf16(scale) : value);
    }
  }
  return out;
}
// High-precision arithmetic oracle for the encoded weights, not a promise of
// bitwise equality with a native FP32 reduction. Input [token,K], output [token,N].
inline std::vector<double> project(const RowWeights& weights, std::span<const std::uint16_t> input,
                                    std::size_t tokens) {
  weights.validate();
  if (!tokens || input.size() != extent(tokens, weights.width)) throw std::invalid_argument("FP8 activation shape differs");
  for (auto word : input) if (!std::isfinite(bf16(word))) throw std::invalid_argument("Finite BF16 activations required");
  std::vector<double> output(extent(tokens, weights.rows));
  for (std::size_t token = 0; token < tokens; ++token) for (std::size_t row = 0; row < weights.rows; ++row) {
    double value = 0;
    for (std::size_t col = 0; col < weights.width; ++col)
      value += static_cast<double>(decode(weights.codes[row * weights.width + col])) * bf16(input[token * weights.width + col]);
    output[token * weights.rows + row] = value * bf16(weights.scales[row]);
  }
  return output;
}
}
