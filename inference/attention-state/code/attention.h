#pragma once
#include "state_pages.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace crit::techniques {
namespace attention_detail {
inline std::size_t add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b) throw std::length_error("Attention extent overflow");
  return a + b;
}
inline std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::length_error("Attention extent overflow");
  return a * b;
}
inline float fp32(double value) {
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
    throw std::overflow_error("Attention FP32 intermediate overflow");
  return static_cast<float>(value);
}
} // namespace attention_detail

// Either token-major or head-major, with optional padding. Only logical values
// are read/validated; the caller retains the borrowed storage through return.
struct HeadTensorView {
  std::span<const float> values;
  std::size_t rows, heads, width, row_stride, head_stride;
  void validate() const {
    using namespace attention_detail;
    if (!rows || !heads || !width || row_stride < width || head_stride < width)
      throw std::invalid_argument("Positive nonoverlapping attention tensor geometry required");
    const auto row_extent = add(product(heads - 1, head_stride), width);
    const auto head_extent = add(product(rows - 1, row_stride), width);
    if ((rows > 1 && heads > 1 && row_stride < row_extent && head_stride < head_extent) ||
        add(product(rows - 1, row_stride), row_extent) > values.size())
      throw std::invalid_argument("Attention tensor stride/extent mismatch");
    for (std::size_t row = 0; row < rows; ++row) for (std::size_t head = 0; head < heads; ++head)
      for (float value : at(row, head))
        if (!std::isfinite(value)) throw std::invalid_argument("Finite attention tensor values required");
  }
  std::span<const float> at(std::size_t row, std::size_t head) const {
    if (row >= rows || head >= heads) throw std::out_of_range("Attention tensor index outside shape");
    const auto offset = attention_detail::add(attention_detail::product(row, row_stride),
                                              attention_detail::product(head, head_stride));
    if (offset > values.size() || width > values.size() - offset)
      throw std::out_of_range("Attention tensor index outside storage");
    return values.subspan(offset, width);
  }
};

enum class RotaryLayout { interleaved, split_half };
// Returns packed [row, head, width]. Frequencies and position scaling are supplied
// explicitly; no architecture's theta/scaling policy is guessed from dimensions.
inline std::vector<float> rotary(const HeadTensorView& input, std::span<const std::uint64_t> positions,
                                  std::span<const double> frequencies, RotaryLayout layout) {
  using namespace attention_detail;
  input.validate();
  if (positions.size() != input.rows || frequencies.empty() || frequencies.size() > input.width / 2 ||
      (layout != RotaryLayout::interleaved && layout != RotaryLayout::split_half))
    throw std::invalid_argument("Rotary positions, even partial width and layout must match");
  for (auto position : positions)
    if (position > (std::uint64_t{1} << 53) - 1) throw std::invalid_argument("Rotary position exceeds exact double integer range");
  for (double frequency : frequencies)
    if (!std::isfinite(frequency) || frequency < 0) throw std::invalid_argument("Finite nonnegative rotary frequencies required");
  std::vector<float> result(product(product(input.rows, input.heads), input.width));
  for (std::size_t row = 0; row < input.rows; ++row) for (std::size_t head = 0; head < input.heads; ++head) {
    const auto source = input.at(row, head);
    auto out = std::span<float>(result).subspan((row * input.heads + head) * input.width, input.width);
    std::copy(source.begin(), source.end(), out.begin());
    for (std::size_t i = 0; i < frequencies.size(); ++i) {
      const double angle = static_cast<double>(positions[row]) * frequencies[i];
      if (!std::isfinite(angle)) throw std::overflow_error("Rotary angle overflow");
      const auto a = layout == RotaryLayout::interleaved ? i * 2 : i;
      const auto b = layout == RotaryLayout::interleaved ? i * 2 + 1 : i + frequencies.size();
      const double cosine = std::cos(angle), sine = std::sin(angle);
      out[a] = fp32(source[a] * cosine - source[b] * sine);
      out[b] = fp32(source[a] * sine + source[b] * cosine);
    }
  }
  return result;
}

struct AttentionGeometry {
  std::size_t query_heads, kv_heads, key_width, value_width;
  double scale;
  void validate() const {
    if (!query_heads || !kv_heads || !key_width || !value_width || query_heads % kv_heads ||
        !std::isfinite(scale) || scale <= 0)
      throw std::invalid_argument("Attention needs positive widths/scale and query heads divisible by KV heads");
  }
};
struct AttentionMask {
  bool causal = true;
  std::size_t window = 0; // 0 = unlimited; otherwise includes the query's own row.
  std::span<const std::uint8_t> keep; // Optional [query, key], 1 = visible.
  std::span<const float> bias; // Optional [query, query_head, key]; -inf masks a cell.
};
struct AttentionResult {
  std::vector<float> values; // Packed [query, query_head, value_width].
  std::vector<double> logsumexp; // [query, query_head]; all-masked rows use -inf.
};

// Synchronous packed FP32 GEMM contract. B is [n,k] when transpose_b, otherwise
// [k,n]. C is fully replaced, never accumulated; no input span may be retained.
struct ScalarAttentionGemm {
  void operator()(bool transpose_b, std::size_t m, std::size_t n, std::size_t k,
                  std::span<const float> a, std::span<const float> b, std::span<float> c) const {
    using namespace attention_detail;
    if (!m || !n || !k || a.size() != product(m, k) || b.size() != product(n, k) || c.size() != product(m, n))
      throw std::invalid_argument("Attention GEMM extents differ");
    for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < n; ++j) {
      double value = 0;
      for (std::size_t p = 0; p < k; ++p) value += static_cast<double>(a[i * k + p]) * b[transpose_b ? j * k + p : p * n + j];
      c[i * n + j] = fp32(value);
    }
  }
};

template<class KVView> class BasicAttentionPlan {
  AttentionGeometry geometry_;
  KVView kv_;
  std::vector<std::size_t> query_rows_;
  std::size_t query_tile_, key_tile_, group_, workspace_bytes_ = 0, output_bytes_ = 0, metadata_bytes_ = 0;
  std::size_t q_size_, k_size_, v_size_, score_size_, partial_size_, accumulator_size_;
  std::vector<float> floats_;
  std::vector<double> doubles_;
  std::atomic_flag running_ = ATOMIC_FLAG_INIT;
  struct Guard {
    std::atomic_flag& flag;
    explicit Guard(std::atomic_flag& value) : flag(value) {
      if (flag.test_and_set(std::memory_order_acquire)) throw std::logic_error("Attention workspace is already in use");
    }
    ~Guard() { flag.clear(std::memory_order_release); }
  };
  static void stop(const std::atomic_bool* cancelled) {
    if (cancelled && cancelled->load(std::memory_order_relaxed)) throw std::runtime_error("Attention cancelled");
  }
  bool visible(const AttentionMask& mask, std::size_t query, std::size_t key) const {
    const auto position = query_rows_[query];
    return (!mask.causal || key <= position) &&
        (!mask.window || (key <= position && position - key < mask.window)) &&
        (mask.keep.empty() || mask.keep[query * kv_.rows() + key] != 0);
  }
  double score(const AttentionMask& mask, std::size_t query, std::size_t head, std::size_t key, float dot) const {
    if (!visible(mask, query, key)) return -std::numeric_limits<double>::infinity();
    const double bias = mask.bias.empty() ? 0 : mask.bias[(query * geometry_.query_heads + head) * kv_.rows() + key];
    if (bias == -std::numeric_limits<double>::infinity()) return bias;
    const double value = dot * geometry_.scale + bias;
    if (!std::isfinite(value)) throw std::overflow_error("Attention logit overflow or incomplete GEMM output");
    return value;
  }
public:
  BasicAttentionPlan(const KVView& kv, AttentionGeometry geometry, std::span<const std::size_t> query_rows,
                      std::size_t query_tile, std::size_t key_tile, std::size_t workspace_limit,
                      std::size_t output_limit = std::numeric_limits<std::size_t>::max(),
                      std::size_t metadata_limit = std::numeric_limits<std::size_t>::max())
      : geometry_(geometry), query_tile_(query_tile), key_tile_(key_tile) {
    using namespace attention_detail;
    geometry.validate();
    if (!kv.rows() || query_rows.empty() || !query_tile || !key_tile ||
        kv.block(0).key_width != product(geometry.kv_heads, geometry.key_width) ||
        kv.block(0).value_width != product(geometry.kv_heads, geometry.value_width))
      throw std::invalid_argument("Paged attention geometry, visible KV and tile sizes must agree");
    for (auto row : query_rows)
      if (row >= kv.rows()) throw std::out_of_range("Query logical position lies outside materialized KV");
    query_tile_ = std::min(query_tile, query_rows.size());
    key_tile_ = std::min(key_tile, kv.rows());
    group_ = geometry.query_heads / geometry.kv_heads;
    const auto m = product(query_tile_, group_);
    q_size_ = product(m, geometry.key_width); k_size_ = product(key_tile_, geometry.key_width);
    v_size_ = product(key_tile_, geometry.value_width); score_size_ = product(m, key_tile_);
    partial_size_ = accumulator_size_ = product(m, geometry.value_width);
    const auto float_count = add(add(add(q_size_, k_size_), add(v_size_, score_size_)), partial_size_);
    const auto double_count = add(accumulator_size_, product(2, m));
    workspace_bytes_ = add(product(float_count, sizeof(float)), product(double_count, sizeof(double)));
    if constexpr (requires { kv.scratch_bytes(); }) workspace_bytes_ = add(workspace_bytes_, kv.scratch_bytes());
    output_bytes_ = product(product(query_rows.size(), geometry.query_heads), add(product(geometry.value_width, sizeof(float)), sizeof(double)));
    metadata_bytes_ = add(sizeof(*this), add(product(query_rows.size(), sizeof(std::size_t)), kv.metadata_bytes()));
    if (workspace_bytes_ > workspace_limit) throw std::length_error("Attention workspace byte limit exceeded");
    if (output_bytes_ > output_limit) throw std::length_error("Attention output byte limit exceeded");
    if (metadata_bytes_ > metadata_limit) throw std::length_error("Attention metadata byte limit exceeded");
    // No state mutation occurs on allocation failure. Views and query indices are
    // separately owned plan metadata; returned result bytes are reported separately.
    kv_ = kv; query_rows_.assign(query_rows.begin(), query_rows.end());
    floats_.resize(float_count); doubles_.resize(double_count);
  }
  BasicAttentionPlan(const BasicAttentionPlan&) = delete;
  BasicAttentionPlan& operator=(const BasicAttentionPlan&) = delete;
  std::size_t workspace_bytes() const { return workspace_bytes_; }
  std::size_t output_bytes() const { return output_bytes_; }
  std::size_t metadata_bytes() const { return metadata_bytes_; }
  template<class Gemm> requires std::invocable<Gemm&, bool, std::size_t, std::size_t, std::size_t,
                                              std::span<const float>, std::span<const float>, std::span<float>>
  AttentionResult run(const HeadTensorView& queries, const AttentionMask& mask, Gemm& gemm,
                       const std::atomic_bool* cancelled = nullptr) {
    using namespace attention_detail;
    Guard guard(running_);
    stop(cancelled);
    queries.validate();
    const auto& g = geometry_;
    if (queries.rows != query_rows_.size() || queries.heads != g.query_heads || queries.width != g.key_width)
      throw std::invalid_argument("Query tensor differs from attention plan");
    const auto cells = product(query_rows_.size(), kv_.rows());
    if ((!mask.keep.empty() && mask.keep.size() != cells) ||
        (!mask.bias.empty() && mask.bias.size() != product(cells, g.query_heads)))
      throw std::invalid_argument("Attention mask or bias extent differs");
    for (auto value : mask.keep) if (value > 1) throw std::invalid_argument("Attention keep mask must contain zero or one");
    for (float value : mask.bias)
      if (!std::isfinite(value) && value != -std::numeric_limits<float>::infinity())
        throw std::invalid_argument("Attention bias must be finite or negative infinity");
    AttentionResult result{std::vector<float>(product(product(queries.rows, g.query_heads), g.value_width), 0),
        std::vector<double>(product(queries.rows, g.query_heads), -std::numeric_limits<double>::infinity())};
    auto memory = std::span<float>(floats_);
    auto q = memory.first(q_size_), k = memory.subspan(q_size_, k_size_);
    auto v = memory.subspan(q_size_ + k_size_, v_size_);
    auto scores = memory.subspan(q_size_ + k_size_ + v_size_, score_size_);
    auto partial = memory.last(partial_size_);
    auto acc = std::span<double>(doubles_).first(accumulator_size_);
    auto maximum = std::span<double>(doubles_).subspan(accumulator_size_, query_tile_ * group_);
    auto total = std::span<double>(doubles_).last(query_tile_ * group_);
    for (std::size_t start = 0; start < queries.rows; start += query_tile_)
      for (std::size_t kv_head = 0; kv_head < g.kv_heads; ++kv_head) {
        stop(cancelled);
        const auto query_count = std::min(query_tile_, queries.rows - start), m = query_count * group_;
        for (std::size_t i = 0; i < m; ++i) {
          const auto source = queries.at(start + i / group_, kv_head * group_ + i % group_);
          std::copy(source.begin(), source.end(), q.begin() + static_cast<std::ptrdiff_t>(i * g.key_width));
        }
        std::fill(acc.begin(), acc.end(), 0); std::fill(total.begin(), total.end(), 0);
        std::fill(maximum.begin(), maximum.end(), -std::numeric_limits<double>::infinity());
        std::size_t first_key = 0;
        for (std::size_t page = 0; page < kv_.pages(); ++page) {
          const auto block = [&] { if constexpr (requires { kv_.block_metadata(page); }) return kv_.block_metadata(page); else return kv_.block(page); }();
          for (std::size_t offset = 0; offset < block.rows; offset += key_tile_) {
            stop(cancelled);
            const auto n = std::min(key_tile_, block.rows - offset);
            if constexpr (requires { kv_.gather_head(page, offset, n, kv_head, g.key_width, g.value_width, k.first(n*g.key_width), v.first(n*g.value_width)); }) {
              kv_.gather_head(page, offset, n, kv_head, g.key_width, g.value_width, k.first(n*g.key_width), v.first(n*g.value_width));
            } else for (std::size_t j = 0; j < n; ++j) {
              const auto base = (offset + j) * block.row_stride;
              std::copy_n(block.values.data() + base + kv_head * g.key_width, g.key_width, k.data() + j * g.key_width);
              std::copy_n(block.values.data() + base + block.key_width + kv_head * g.value_width,
                          g.value_width, v.data() + j * g.value_width);
            }
            std::fill_n(scores.data(), m * n, std::numeric_limits<float>::quiet_NaN());
            gemm(true, m, n, g.key_width, q.first(m * g.key_width), k.first(n * g.key_width), scores.first(m * n));
            stop(cancelled);
            for (float value : scores.first(m * n))
              if (!std::isfinite(value)) throw std::overflow_error("Attention dot product is nonfinite or incomplete");
            for (std::size_t i = 0; i < m; ++i) {
              const auto query = start + i / group_, head = kv_head * group_ + i % group_;
              double tile_top = -std::numeric_limits<double>::infinity();
              for (std::size_t j = 0; j < n; ++j)
                tile_top = std::max(tile_top, score(mask, query, head, first_key + offset + j, scores[i * n + j]));
              if (tile_top == -std::numeric_limits<double>::infinity()) {
                std::fill_n(scores.data() + i * n, n, 0.f); continue;
              }
              const auto next_top = std::max(maximum[i], tile_top);
              const auto previous = total[i] * std::exp(maximum[i] - next_top);
              double denominator = previous;
              for (std::size_t j = 0; j < n; ++j)
                denominator += std::exp(score(mask, query, head, first_key + offset + j, scores[i * n + j]) - next_top);
              const double old_weight = previous / denominator;
              for (std::size_t c = 0; c < g.value_width; ++c) acc[i * g.value_width + c] *= old_weight;
              for (std::size_t j = 0; j < n; ++j)
                scores[i * n + j] = fp32(std::exp(score(mask, query, head, first_key + offset + j, scores[i * n + j]) - next_top) / denominator);
              maximum[i] = next_top; total[i] = denominator;
            }
            std::fill_n(partial.data(), m * g.value_width, std::numeric_limits<float>::quiet_NaN());
            gemm(false, m, g.value_width, n, scores.first(m * n), v.first(n * g.value_width), partial.first(m * g.value_width));
            stop(cancelled);
            for (std::size_t i = 0; i < m * g.value_width; ++i) {
              if (!std::isfinite(partial[i])) throw std::overflow_error("Attention value reduction is nonfinite or incomplete");
              acc[i] += partial[i];
            }
          }
          first_key += block.rows;
        }
        for (std::size_t i = 0; i < m; ++i) {
          const auto output_row = (start + i / group_) * g.query_heads + kv_head * group_ + i % group_;
          if (total[i]) {
            result.logsumexp[output_row] = maximum[i] + std::log(total[i]);
            if (!std::isfinite(result.logsumexp[output_row])) throw std::overflow_error("Attention logsumexp overflow");
          }
          for (std::size_t c = 0; c < g.value_width; ++c) result.values[output_row * g.value_width + c] = fp32(acc[i * g.value_width + c]);
        }
      }
    stop(cancelled);
    return result;
  }
  AttentionResult run(const HeadTensorView& queries, const AttentionMask& mask = {}, const std::atomic_bool* cancelled = nullptr) {
    ScalarAttentionGemm gemm;
    return run(queries, mask, gemm, cancelled);
  }
};
using PagedAttentionPlan = BasicAttentionPlan<StatePagesView>;

// Scalar native reference: streams directly over paged FP32 rows with online
// softmax and double accumulation. The caller chooses the visible causal prefix.
inline std::vector<float> paged_attention(const StatePagesView& view, std::span<const float> query, double scale) {
  if (!view.rows() || query.size() != view.block(0).key_width || !std::isfinite(scale) || scale <= 0 ||
      std::any_of(query.begin(), query.end(), [](float v) { return !std::isfinite(v); }))
    throw std::invalid_argument("Nonempty KV prefix, matching finite query and positive finite scale required");
  std::vector<double> sum(view.block(0).value_width, 0);
  double top = -std::numeric_limits<double>::infinity(), normalizer = 0;
  for (std::size_t row = 0; row < view.rows(); ++row) {
    double score = 0;
    const auto key = view.key(row), value = view.value(row);
    for (std::size_t i = 0; i < query.size(); ++i) score += static_cast<double>(query[i]) * key[i];
    score *= scale;
    if (!std::isfinite(score)) throw std::overflow_error("KV attention score overflow");
    const double next_top = std::max(top, score), old = std::exp(top - next_top), weight = std::exp(score - next_top);
    normalizer = normalizer * old + weight;
    for (std::size_t i = 0; i < sum.size(); ++i) sum[i] = sum[i] * old + weight * value[i];
    top = next_top;
  }
  std::vector<float> result(sum.size());
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = attention_detail::fp32(sum[i] / normalizer);
  return result;
}
} // namespace crit::techniques
