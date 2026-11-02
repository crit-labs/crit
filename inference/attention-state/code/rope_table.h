#pragma once
// Rotary inverse frequencies and the attention factor from a config's rope parameters (Core's RopeConfig, read from
// config.json by crit::core::load_model_config), derived as Hugging Face's ROPE_INIT_FUNCTIONS derive them
// (transformers.modeling_rope_utils), in double, so an engine reads its rotary table instead of re-deriving YaRN,
// Llama 3 or LongRoPE scaling; rotary() in attention.h applies the frequencies. The Python twin is rope_table.py.
#include <crit/json.h>
#include <crit/model_config.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace crit::techniques {
// One inverse frequency per rotated pair (rotary_dim / 2 of them) and the factor the cos and sin tables are
// multiplied by (1 except for YaRN and LongRoPE).
struct RopeTable {
  std::vector<double> inv_freq = {};
  double attention_factor = 1.0;
};

// int(head_dim x partial_rotary_factor), the dimensions that rotate: a positive even number, else
// FormatError("rope", ...).
inline std::uint32_t rotary_dim(const core::RopeConfig& rope, std::uint32_t head_dim) {
  if (!head_dim) throw core::FormatError("rope", "head dimension", "0 is not a positive integer");
  const double factor = rope.partial_rotary_factor.value_or(1.0);
  if (!(factor > 0 && factor <= 1))
    throw core::FormatError("rope", "partial_rotary_factor", std::to_string(factor) + " is outside (0, 1]");
  const auto dim = std::uint32_t(double(head_dim) * factor);
  if (dim < 2 || dim % 2)
    throw core::FormatError("rope", "rotary dimension", std::to_string(dim) + " is not a positive even number");
  return dim;
}

// type "" is "default"; "mrope", "linear", "dynamic", "yarn", "llama3" and "longrope" are supported.
// max_position_embeddings is the model's declared length (ModelConfig::max_positions): dynamic NTK needs it, and
// YaRN, Llama 3 and LongRoPE read it when the rope object leaves factor or original_max_position_embeddings out.
// sequence_length selects dynamic NTK's base and LongRoPE's long factors past the original length; nullopt means the
// declared length. Other types, missing parameters and a rotary dimension that is not a positive even number throw
// FormatError("rope", ...).
inline RopeTable rope_table(const core::RopeConfig& rope, std::uint32_t head_dim,
                            std::optional<std::uint64_t> max_position_embeddings = {},
                            std::optional<std::uint64_t> sequence_length = {}) {
  constexpr double pi = 3.14159265358979323846;
  const std::string kind = rope.type.empty() ? "default" : rope.type;
  const auto fail = [&](const std::string& reason) { return core::FormatError("rope", "type '" + kind + "'", reason); };
  static constexpr std::array<std::string_view, 7> types{"default", "mrope", "linear", "dynamic", "yarn", "llama3",
                                                         "longrope"};
  if (std::find(types.begin(), types.end(), kind) == types.end()) throw fail("has no table");
  const std::uint32_t dim = rotary_dim(rope, head_dim), half = dim / 2;
  const double base = rope.theta.value_or(10000.0);   // Hugging Face's default rope_theta
  if (!(base > 1)) throw core::FormatError("rope", "rope_theta", std::to_string(base) + " must exceed 1");
  const auto need = [&](const char* name, const std::optional<double>& value) {
    if (!value) throw fail(std::string("needs ") + name);
    return *value;
  };
  const auto declared = [&] {
    if (!max_position_embeddings) throw fail("needs max_position_embeddings");
    return double(*max_position_embeddings);
  };
  const auto original = [&] {
    if (rope.original_max_position_embeddings) return double(*rope.original_max_position_embeddings);
    if (!max_position_embeddings) throw fail("needs original_max_position_embeddings (or max_position_embeddings)");
    return double(*max_position_embeddings);
  };
  const auto factor_or_ratio = [&] { return rope.factor ? *rope.factor : declared() / original(); };
  std::vector<double> plain(half);
  for (std::uint32_t i = 0; i < half; ++i) plain[i] = 1.0 / std::pow(base, 2.0 * i / dim);
  RopeTable table{plain, 1.0};
  if (kind == "default" || kind == "mrope") return table;
  if (kind == "linear") {
    const double factor = need("factor", rope.factor);
    for (auto& f : table.inv_freq) f /= factor;
    return table;
  }
  if (kind == "dynamic") {
    const double factor = need("factor", rope.factor), length0 = declared();
    if (dim == 2) throw fail("needs a rotary dimension above 2");
    const double length = std::max(sequence_length ? double(*sequence_length) : length0, length0);
    const double scaled = base * std::pow(factor * length / length0 - (factor - 1), double(dim) / (dim - 2));
    for (std::uint32_t i = 0; i < half; ++i) table.inv_freq[i] = 1.0 / std::pow(scaled, 2.0 * i / dim);
    return table;
  }
  if (kind == "yarn") {
    const double factor = factor_or_ratio(), start = original();
    const auto mscale = [](double scale, double m) { return scale <= 1 ? 1.0 : 0.1 * m * std::log(scale) + 1.0; };
    // Hugging Face's rule: mscale and mscale_all_dim count only when both are nonzero.
    if (rope.attention_factor) table.attention_factor = *rope.attention_factor;
    else if (rope.mscale.value_or(0) != 0 && rope.mscale_all_dim.value_or(0) != 0)
      table.attention_factor = mscale(factor, *rope.mscale) / mscale(factor, *rope.mscale_all_dim);
    else table.attention_factor = mscale(factor, 1.0);
    const double fast = rope.beta_fast.value_or(0) != 0 ? *rope.beta_fast : 32.0;
    const double slow = rope.beta_slow.value_or(0) != 0 ? *rope.beta_slow : 1.0;
    const auto correction = [&](double rotations) {
      return dim * std::log(start / (rotations * 2 * pi)) / (2 * std::log(base));
    };
    double low = correction(fast), high = correction(slow);
    const auto* truncate =
        rope.raw && rope.raw->kind() == core::json::Kind::object ? rope.raw->find("truncate") : nullptr;
    if (!(truncate && truncate->kind() == core::json::Kind::boolean && !truncate->boolean())) {
      low = std::floor(low);
      high = std::ceil(high);
    }
    low = std::max(low, 0.0);
    high = std::min(high, double(dim - 1));
    if (low == high) high += 0.001;   // Hugging Face's guard against a zero-width ramp
    for (std::uint32_t i = 0; i < half; ++i) {
      const double extrapolate = 1.0 - std::clamp((double(i) - low) / (high - low), 0.0, 1.0);
      table.inv_freq[i] = plain[i] / factor * (1.0 - extrapolate) + plain[i] * extrapolate;
    }
    return table;
  }
  if (kind == "llama3") {
    const double factor = need("factor", rope.factor), low = need("low_freq_factor", rope.low_freq_factor);
    const double high = need("high_freq_factor", rope.high_freq_factor), start = original();
    for (auto& f : table.inv_freq) {
      const double wavelength = 2 * pi / f;
      if (wavelength < start / high) continue;
      if (wavelength > start / low) {
        f /= factor;
        continue;
      }
      const double smooth = (start / wavelength - low) / (high - low);
      f = (1 - smooth) * f / factor + smooth * f;
    }
    return table;
  }
  // longrope
  const double start = original(), factor = factor_or_ratio();
  const std::pair<const char*, const std::vector<double>*> lists[]{{"short_factor", &rope.short_factor},
                                                                   {"long_factor", &rope.long_factor}};
  for (const auto& [name, values] : lists)
    if (values->size() != half)
      throw core::FormatError("rope", name, "has " + std::to_string(values->size()) +
                                                " entries, not rotary_dim / 2 = " + std::to_string(half));
  if (rope.attention_factor) table.attention_factor = *rope.attention_factor;
  else table.attention_factor = factor <= 1 ? 1.0 : std::sqrt(1 + std::log(factor) / std::log(start));
  const auto& scales = sequence_length && double(*sequence_length) > start ? rope.long_factor : rope.short_factor;
  for (std::uint32_t i = 0; i < half; ++i) table.inv_freq[i] = plain[i] / scales[i];
  return table;
}
} // namespace crit::techniques
