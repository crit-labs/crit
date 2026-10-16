#include "attention.h"
#include "batch_attention.h"
#include "attention_gemm.h"
#include "state_pages_fixture.h"
#include <array>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>

using namespace crit::techniques;
void require(bool valid, const char* message) { if (!valid) throw std::runtime_error(message); }
template<class F> void rejects(F action) {
  try { action(); } catch (const std::exception&) { return; }
  throw std::runtime_error("Expected rejection");
}
struct Checked { std::size_t cases = 0; double output_error = 0, lse_error = 0; };
AttentionResult reference(const HeadTensorView& q, std::span<const float> keys, std::span<const float> values,
                           std::size_t count, AttentionGeometry g, std::span<const std::size_t> positions,
                           const AttentionMask& mask) {
  AttentionResult out{std::vector<float>(q.rows * g.query_heads * g.value_width, 0),
      std::vector<double>(q.rows * g.query_heads, -std::numeric_limits<double>::infinity())};
  for (std::size_t row = 0; row < q.rows; ++row) for (std::size_t head = 0; head < g.query_heads; ++head) {
    std::vector<double> logits(count, -std::numeric_limits<double>::infinity());
    double top = -std::numeric_limits<double>::infinity();
    for (std::size_t key = 0; key < count; ++key) {
      if ((mask.causal && key > positions[row]) ||
          (mask.window && (key > positions[row] || positions[row] - key >= mask.window)) ||
          (!mask.keep.empty() && mask.keep[row * count + key] == 0)) continue;
      const auto kv_head = head / (g.query_heads / g.kv_heads);
      double dot = 0;
      for (std::size_t d = 0; d < g.key_width; ++d)
        dot += static_cast<double>(q.values[row * q.row_stride + head * q.head_stride + d]) *
            keys[(key * g.kv_heads + kv_head) * g.key_width + d];
      logits[key] = dot * g.scale + (mask.bias.empty() ? 0 : mask.bias[(row * g.query_heads + head) * count + key]);
      top = std::max(top, logits[key]);
    }
    if (!std::isfinite(top)) continue;
    double total = 0;
    for (auto& value : logits) { value = std::exp(value - top); total += value; }
    out.logsumexp[row * g.query_heads + head] = top + std::log(total);
    for (std::size_t d = 0; d < g.value_width; ++d) {
      double value = 0;
      for (std::size_t key = 0; key < count; ++key)
        value += logits[key] / total * values[(key * g.kv_heads + head / (g.query_heads / g.kv_heads)) * g.value_width + d];
      out.values[(row * g.query_heads + head) * g.value_width + d] = static_cast<float>(value);
    }
  }
  return out;
}
void compare(const AttentionResult& actual, const AttentionResult& expected, Checked& checked) {
  require(actual.values.size() == expected.values.size() && actual.logsumexp.size() == expected.logsumexp.size(), "Output geometry differs");
  for (std::size_t i = 0; i < actual.values.size(); ++i) {
    const double error = std::abs(static_cast<double>(actual.values[i]) - expected.values[i]);
    require(std::isfinite(actual.values[i]) && error <= 2e-5 * (1 + std::abs(expected.values[i])), "Attention output differs from independent reference");
    checked.output_error = std::max(error, checked.output_error);
  }
  for (std::size_t i = 0; i < actual.logsumexp.size(); ++i) {
    if (!std::isfinite(expected.logsumexp[i])) require(actual.logsumexp[i] == expected.logsumexp[i], "All-masked logsumexp differs");
    else {
      const double error = std::abs(actual.logsumexp[i] - expected.logsumexp[i]);
      require(std::isfinite(actual.logsumexp[i]) && error <= 2e-5 * (1 + std::abs(expected.logsumexp[i])), "Logsumexp differs from independent reference");
      checked.lse_error = std::max(error, checked.lse_error);
    }
  }
  ++checked.cases;
}
template<class Gemm> Checked arithmetic(Gemm& gemm) {
  Checked checked;
  std::mt19937 rng(75910);
  std::uniform_real_distribution<float> sample(-1.5f, 1.5f);
  for (const auto& heads : {std::pair{1u, 1u}, std::pair{4u, 4u}, std::pair{4u, 2u}, std::pair{4u, 1u}})
    for (const auto& widths : {std::pair{2u, 1u}, std::pair{6u, 5u}})
      for (std::size_t count : {1, 7, 19}) for (std::size_t page_rows : {1, 4, 16}) {
        AttentionGeometry g{heads.first, heads.second, widths.first, widths.second, .47};
        PagedState kv(32, static_cast<std::uint32_t>(page_rows), g.kv_heads * g.key_width, g.kv_heads * g.value_width,
                      static_cast<std::uint32_t>(count));
        const auto state = kv.open();
        std::vector<float> keys(count * g.kv_heads * g.key_width), values(count * g.kv_heads * g.value_width);
        for (auto& value : keys) value = sample(rng);
        for (auto& value : values) value = sample(rng);
        kv.append(state, keys, values);
        for (int mode = 0; mode < 4; ++mode) {
          const auto queries = mode == 0 ? count : mode == 1 ? 1 : std::min(count, std::size_t{4});
          const auto start = count - queries;
          std::vector<std::size_t> positions(queries);
          for (std::size_t i = 0; i < queries; ++i) positions[i] = start + i;
          if (mode == 3) for (std::size_t i = 0; i < queries; ++i) positions[i] = i % 2 ? count - 1 : i / 2;
          const auto rs = mode == 1 || mode == 2 ? g.key_width + 1 : g.query_heads * (g.key_width + 1) + 2;
          const auto hs = mode == 1 || mode == 2 ? queries * rs + 2 : g.key_width + 1;
          std::vector<float> query_data((queries - 1) * rs + (g.query_heads - 1) * hs + g.key_width,
                                        std::numeric_limits<float>::quiet_NaN());
          HeadTensorView q{query_data, queries, g.query_heads, g.key_width, rs, hs};
          for (std::size_t row = 0; row < queries; ++row) for (std::size_t head = 0; head < g.query_heads; ++head)
            for (std::size_t d = 0; d < g.key_width; ++d) query_data[row * rs + head * hs + d] = sample(rng);
          const auto original = query_data;
          for (int mask_case = 0; mask_case < 4; ++mask_case) {
            AttentionMask mask;
            std::vector<std::uint8_t> keep(queries * count, 1);
            std::vector<float> bias(queries * g.query_heads * count, 0);
            if (mask_case == 1) mask.causal = false;
            if (mask_case == 2) mask.window = 3;
            if (mask_case == 3) {
              for (std::size_t row = 0; row < queries; ++row) for (std::size_t key = 0; key < count; ++key) {
                keep[row * count + key] = row == 0 || key % 3 == 0 ? 0 : 1;
                for (std::size_t head = 0; head < g.query_heads; ++head)
                  bias[(row * g.query_heads + head) * count + key] = key % 5 == 0 ?
                    -std::numeric_limits<float>::infinity() : sample(rng);
              }
              mask.keep = keep; mask.bias = bias;
            }
            const auto expected = reference(q, keys, values, count, g, positions, mask);
            for (const auto& tiles : {std::pair{1u, 3u}, std::pair{4u, 8u}}) {
              PagedAttentionPlan plan(kv.view(state), g, positions, tiles.first, tiles.second, 1 << 20);
              compare(plan.run(q, mask, gemm), expected, checked);
            }
          }
          require(std::memcmp(original.data(), query_data.data(), query_data.size() * sizeof(float)) == 0, "Query or padding was modified");
        }
        const auto retained = kv.view(state);
        for (std::size_t row = 0; row < count; ++row) {
          require(std::equal(retained.key(row).begin(), retained.key(row).end(), keys.begin() + static_cast<std::ptrdiff_t>(row * g.kv_heads * g.key_width)), "Keys changed");
          require(std::equal(retained.value(row).begin(), retained.value(row).end(), values.begin() + static_cast<std::ptrdiff_t>(row * g.kv_heads * g.value_width)), "Values changed");
        }
      }
  return checked;
}
void rotary_checks() {
  const double pi = std::acos(-1.0);
  const std::array<float, 6> input{1, 2, 3, 4, 77, 88};
  HeadTensorView view{input, 1, 1, 6, 6, 6};
  const auto interleaved = rotary(view, std::array<std::uint64_t, 1>{1}, std::array{pi / 2, pi}, RotaryLayout::interleaved);
  const auto split = rotary(view, std::array<std::uint64_t, 1>{1}, std::array{pi / 2, pi}, RotaryLayout::split_half);
  const std::array<float, 6> expected_i{-2, 1, -3, -4, 77, 88}, expected_s{-3, -2, 1, -4, 77, 88};
  for (std::size_t i = 0; i < input.size(); ++i) {
    require(std::abs(interleaved[i] - expected_i[i]) < 1e-6, "Interleaved partial RoPE differs");
    require(std::abs(split[i] - expected_s[i]) < 1e-6, "Split-half partial RoPE differs");
  }
  require(rotary(view, std::array<std::uint64_t, 1>{0}, std::array{1., .1}, RotaryLayout::split_half) == std::vector<float>(input.begin(), input.end()), "Position zero changed input");
  const std::array<float, 12> padded{1, 2, 99, 3, 4, 99, 5, 6, 99, 7, 8, 99};
  const auto multiple = rotary({padded, 2, 2, 2, 3, 6}, std::array<std::uint64_t, 2>{0, 1},
                              std::array{pi / 2}, RotaryLayout::interleaved);
  const std::array<float, 8> expected_multiple{1, 2, 5, 6, -4, 3, -8, 7};
  for (std::size_t i = 0; i < multiple.size(); ++i)
    require(std::abs(multiple[i] - expected_multiple[i]) < 1e-6, "Padded head-major RoPE row/head mapping differs");
  rejects([&] { rotary(view, std::array<std::uint64_t, 1>{std::uint64_t{1} << 54}, std::array{1.}, RotaryLayout::interleaved); });
  rejects([&] { rotary(view, std::array<std::uint64_t, 1>{1}, std::array{-1.}, RotaryLayout::interleaved); });
  rejects([&] { rotary(view, std::array<std::uint64_t, 1>{1}, std::array{1.}, static_cast<RotaryLayout>(42)); });
}
template<class Gemm> void chunked_continuation(Gemm& gemm, Checked& checked) {
  const AttentionGeometry geometry{4, 2, 3, 2, .5};
  constexpr std::size_t count = 13, query_stride = 12, key_stride = 6, value_stride = 4;
  std::mt19937 rng(271828);
  std::uniform_real_distribution<float> sample(-1, 1);
  std::vector<float> queries(count * query_stride), keys(count * key_stride), values(count * value_stride);
  for (auto* data : {&queries, &keys, &values}) for (auto& value : *data) value = sample(rng);
  std::vector<std::size_t> all_positions(count);
  for (std::size_t i = 0; i < count; ++i) all_positions[i] = i;
  const auto expected = reference({queries, count, 4, 3, query_stride, 3}, keys, values,
                                  count, geometry, all_positions, {});
  for (std::size_t page : {std::size_t{2}, std::size_t{4}, std::size_t{7}}) {
    // Restore a shared prefix just before, at, and after each page boundary.
    for (std::size_t prefix : {page - 1, page, page + 1}) {
      for (std::size_t chunk : {std::size_t{1}, std::size_t{3}, std::size_t{13}}) {
        PagedState kv(64, static_cast<std::uint32_t>(page), key_stride, value_stride, count);
        const auto source = kv.open();
        kv.append(source, std::span<const float>(keys).first(prefix * key_stride),
                  std::span<const float>(values).first(prefix * value_stride));
        // The fork shares the prefix pages; its first row in a partial tail makes
        // the cache copy that page through StatePages::copy_rows.
        const auto continued = kv.cache.fork(source);
        const auto pinned = kv.view(source);
        for (std::size_t start = prefix; start < count; start += chunk) {
          const auto rows = std::min(chunk, count - start);
          kv.append(continued, std::span<const float>(keys).subspan(start * key_stride, rows * key_stride),
                    std::span<const float>(values).subspan(start * value_stride, rows * value_stride));
          auto positions = std::span<const std::size_t>(all_positions).subspan(start, rows);
          PagedAttentionPlan plan(kv.view(continued), geometry, positions, 2, 3, 1 << 20);
          const auto actual = plan.run({std::span<const float>(queries).subspan(start * query_stride, rows * query_stride),
                                       rows, 4, 3, query_stride, 3}, {}, gemm);
          compare(actual, {{expected.values.begin() + static_cast<std::ptrdiff_t>(start * 8),
                            expected.values.begin() + static_cast<std::ptrdiff_t>((start + rows) * 8)},
                           {expected.logsumexp.begin() + static_cast<std::ptrdiff_t>(start * 4),
                            expected.logsumexp.begin() + static_cast<std::ptrdiff_t>((start + rows) * 4)}}, checked);
        }
        require(kv.cache.committed(source) == prefix && pinned.rows() == prefix, "Chunked continuation mutated its shared prefix");
        for (std::size_t row = 0; row < prefix; ++row)
          require(std::equal(pinned.key(row).begin(), pinned.key(row).end(), keys.begin() + static_cast<std::ptrdiff_t>(row * key_stride)) &&
                  std::equal(pinned.value(row).begin(), pinned.value(row).end(), values.begin() + static_cast<std::ptrdiff_t>(row * value_stride)),
                  "Continuation wrote into the shared prefix pages");
        for (std::size_t row = 0; row < count; ++row) {
          const auto view = kv.view(continued);
          require(std::equal(view.key(row).begin(), view.key(row).end(), keys.begin() + static_cast<std::ptrdiff_t>(row * key_stride)),
                  "Chunk decomposition changed KV contents");
        }
      }
    }
  }
}
template<class Gemm> Checked integration(Gemm& gemm) {
  AttentionGeometry g{4, 2, 4, 3, .5};
  std::mt19937 rng(513);
  std::uniform_real_distribution<float> sample(-1, 1);
  std::vector<float> q(5 * 4 * 4), k(5 * 2 * 4), v(5 * 2 * 3);
  for (auto* data : {&q, &k, &v}) for (auto& x : *data) x = sample(rng);
  const std::array<std::uint64_t, 5> rope_positions{10, 11, 12, 13, 14};
  const auto rotated_q = rotary({q, 5, 4, 4, 16, 4}, rope_positions, std::array{1., .01}, RotaryLayout::split_half);
  const auto rotated_k = rotary({k, 5, 2, 4, 8, 4}, rope_positions, std::array{1., .01}, RotaryLayout::split_half);
  PagedState kv(12, 2, 8, 6, 5);
  const auto sequential = kv.open(), full = kv.open();
  kv.append(full, rotated_k, v);
  PagedAttentionPlan all(kv.view(full), g, std::array<std::size_t, 5>{0, 1, 2, 3, 4}, 4, 3, 1 << 20);
  const auto expected = all.run({rotated_q, 5, 4, 4, 16, 4}, {}, gemm);
  Checked checked;
  compare(expected, reference({rotated_q, 5, 4, 4, 16, 4}, rotated_k, v, 5, g,
      std::array<std::size_t, 5>{0, 1, 2, 3, 4}, {}), checked);
  for (std::size_t row = 0; row < 5; ++row) {
    const auto round = kv.pages.stage(kv.cache, sequential, std::span<const float>(rotated_k).subspan(row * 8, 8),
                                      std::span<const float>(v).subspan(row * 6, 6));
    PagedAttentionPlan step(kv.view(sequential, true), g, std::array{row}, 1, 3, 1 << 20);
    const auto result = step.run({std::span<const float>(rotated_q).subspan(row * 16, 16), 1, 4, 4, 16, 4}, {}, gemm);
    AttentionResult expected_row{{expected.values.begin() + static_cast<std::ptrdiff_t>(row * 12), expected.values.begin() + static_cast<std::ptrdiff_t>((row + 1) * 12)},
      {expected.logsumexp.begin() + static_cast<std::ptrdiff_t>(row * 4), expected.logsumexp.begin() + static_cast<std::ptrdiff_t>((row + 1) * 4)}};
    compare(result, expected_row, checked); kv.cache.commit(sequential, round.id, std::array<crit::core::Token, 1>{0});
  }
  // A plan keeps its exact read image after the request rewinds: released pages
  // stay retiring, never handed out, until retire() covers their epoch.
  kv.cache.rewind(full, 0);
  require(kv.cache.stats().retiring_pages == 3, "Rewound pages left retirement early");
  // Fill every allocatable page with zeros: no retiring page is handed out.
  std::vector<crit::core::SequenceId> fillers;
  for (int i = 0; i < 2; ++i) {
    fillers.push_back(kv.open());
    kv.append(fillers.back(), std::vector<float>(5 * 8, 0), std::vector<float>(5 * 6, 0));
  }
  fillers.push_back(kv.open());
  rejects([&] { kv.append(fillers.back(), std::vector<float>(8, 0), std::vector<float>(6, 0)); });
  compare(all.run({rotated_q, 5, 4, 4, 16, 4}, {}, gemm), expected, checked);
  for (auto filler : fillers) kv.cache.close(filler);
  kv.settle();
  kv.cache.rewind(sequential, 3);
  const auto branch = kv.cache.fork(sequential);
  const auto round = kv.pages.stage(kv.cache, branch, std::span<const float>(rotated_k).subspan(24), std::span<const float>(v).subspan(18));
  PagedAttentionPlan verify(kv.view(branch, true), g, std::array<std::size_t, 2>{3, 4}, 2, 2, 1 << 20);
  auto result = verify.run({std::span<const float>(rotated_q).subspan(48), 2, 4, 4, 16, 4}, {}, gemm);
  kv.cache.commit(branch, round.id, std::array<crit::core::Token, 1>{0});
  require(kv.cache.committed(branch) == 4 && kv.cache.committed(sequential) == 3, "Verification acceptance changed another branch");
  compare(result, {{expected.values.begin() + 36, expected.values.end()},
                    {expected.logsumexp.begin() + 12, expected.logsumexp.end()}}, checked);
  // Chunked continuation shares the error bounds but not the fixed integration count.
  Checked continuation;
  chunked_continuation(gemm, continuation);
  checked.output_error = std::max(checked.output_error, continuation.output_error);
  checked.lse_error = std::max(checked.lse_error, continuation.lse_error);
  return checked;
}
void failures() {
  PagedState kv(3, 2, 2, 1, 3); const auto state = kv.open();
  kv.append(state, std::array{1.f, 0.f, 0.f, 1.f}, std::array{2.f, 10.f});
  AttentionGeometry g{1, 1, 2, 1, 1};
  const std::array<std::size_t, 2> positions{0, 1};
  const std::array<float, 4> data{0, 0, 0, 0};
  HeadTensorView q{data, 2, 1, 2, 2, 2};
  PagedAttentionPlan plan(kv.view(state), g, positions, 1, 1, 4096);
  require(plan.output_bytes() == 2 * (sizeof(float) + sizeof(double)) && plan.metadata_bytes() > 0, "Memory requirements differ");
  rejects([&] { PagedAttentionPlan invalid(kv.view(state), g, positions, 1, 1, plan.workspace_bytes() - 1); });
  rejects([&] { PagedAttentionPlan invalid(kv.view(state), {3, 2, 1, 1, 1}, positions, 1, 1, 4096); });
  rejects([&] { PagedAttentionPlan invalid(kv.view(state), g, std::array<std::size_t, 1>{2}, 1, 1, 4096); });
  auto wrong = q; wrong.row_stride = std::numeric_limits<std::size_t>::max();
  rejects([&] { plan.run(wrong); });
  rejects([&] { plan.run(q, {true, 0, std::array<std::uint8_t, 1>{1}, {}}); });
  rejects([&] { plan.run(q, {true, 0, std::array<std::uint8_t, 4>{1, 1, 1, 2}, {}}); });
  rejects([&] { plan.run(q, {true, 0, {}, std::array{0.f, 0.f, 0.f, std::numeric_limits<float>::quiet_NaN()}}); });
  const auto baseline = plan.run(q);
  require(baseline.values == std::vector<float>({2, 6}), "Causal row convention differs");
  auto no_write = [](bool, std::size_t, std::size_t, std::size_t, auto, auto, auto) {};
  rejects([&] { plan.run(q, {}, no_write); });
  rejects([&] { plan.run(q, {true, 0, std::array<std::uint8_t, 4>{0, 0, 0, 0}, {}}, no_write); });
  auto incomplete = [](bool transpose, std::size_t m, std::size_t n, std::size_t k, auto a, auto b, auto c) {
    ScalarAttentionGemm{}(transpose, m, n, k, a, b, c);
    if (!transpose) c.back() = std::numeric_limits<float>::quiet_NaN();
  };
  rejects([&] { plan.run(q, {}, incomplete); });
  std::size_t calls = 0;
  auto failing = [&](bool transpose, std::size_t m, std::size_t n, std::size_t k, auto a, auto b, auto c) {
    if (++calls == 2) throw std::runtime_error("backend failed after QK");
    ScalarAttentionGemm{}(transpose, m, n, k, a, b, c);
  };
  auto unpublished = baseline;
  rejects([&] { unpublished = plan.run(q, {}, failing); });
  require(unpublished.values == baseline.values, "Partial result escaped failure");
  std::atomic_bool cancelled{false};
  auto cancel = [&](bool transpose, std::size_t m, std::size_t n, std::size_t k, auto a, auto b, auto c) {
    ScalarAttentionGemm{}(transpose, m, n, k, a, b, c); cancelled = true;
  };
  rejects([&] { plan.run(q, {}, cancel, &cancelled); });
  rejects([&] { plan.run(q, {}, &cancelled); });
  auto reentrant = [&](bool transpose, std::size_t m, std::size_t n, std::size_t k, auto a, auto b, auto c) {
    rejects([&] { plan.run(q); }); ScalarAttentionGemm{}(transpose, m, n, k, a, b, c);
  };
  require(plan.run(q, {}, reentrant).values == baseline.values, "Workspace guard did not recover");
  require(plan.run(q).values == baseline.values && kv.cache.committed(state) == 2, "Failure mutated retained KV or reusable workspace");
  // Softmax must remain stable when naive exponentiation would overflow, and
  // when successive tiles move the maximum by thousands of logit units.
  PagedState extreme_kv(5, 1, 1, 1, 4); const auto extreme = extreme_kv.open();
  const std::array<float, 4> extreme_k{-1000, 0, 1000, 999}, extreme_v{7, 5, 3, 1};
  extreme_kv.append(extreme, extreme_k, extreme_v);
  const std::array<float, 2> extreme_q{1, -1};
  PagedAttentionPlan stress(extreme_kv.view(extreme), {1, 1, 1, 1, 1}, positions, 2, 1, 4096);
  Checked checked;
  compare(stress.run({extreme_q, 2, 1, 1, 1, 1}, {false, 0, {}, {}}),
          reference({extreme_q, 2, 1, 1, 1, 1}, extreme_k, extreme_v, 4, {1, 1, 1, 1, 1}, positions, {false, 0, {}, {}}), checked);
}
Checked batches() {
  Checked checked;
  std::mt19937 rng(82371); std::uniform_real_distribution<float> random(-1.f, 1.f);
  struct Data {
    std::vector<float> keys, values, queries;
    std::vector<std::size_t> positions;
    AttentionResult expected;
  };
  std::vector<Data> data(36);
  std::vector<AttentionBatchSpec<StatePagesView>> specs;
  std::vector<std::unique_ptr<PagedState>> stores; // Storage outlives every view and plan.
  std::vector<AttentionBatchInput> inputs;
  for (std::size_t i = 0; i < data.size(); ++i) {
    const auto count = i + 1, queries = i % 5 + 1 > count ? count : i % 5 + 1;
    const auto kv_heads = i % 2 + 1, heads = kv_heads * (i % 3 + 1);
    AttentionGeometry g{heads, kv_heads, i % 4 + 1, i % 3 + 1, .5};
    auto& d = data[i];
    d.keys.resize(count * kv_heads * g.key_width); d.values.resize(count * kv_heads * g.value_width);
    d.queries.resize(queries * heads * g.key_width); d.positions.resize(queries);
    for (auto& x : d.keys) x = random(rng);
    for (auto& x : d.values) x = random(rng);
    for (auto& x : d.queries) x = random(rng);
    for (std::size_t j = 0; j < queries; ++j) d.positions[j] = count - j - 1;
    auto& kv = *stores.emplace_back(std::make_unique<PagedState>(40, static_cast<std::uint32_t>(i % 7 + 1),
        kv_heads * g.key_width, kv_heads * g.value_width, static_cast<std::uint32_t>(count)));
    const auto sequence = kv.open();
    kv.append(sequence, d.keys, d.values);
    HeadTensorView q{d.queries, queries, heads, g.key_width, heads * g.key_width, g.key_width};
    AttentionMask mask; mask.window = i % 2 ? 3 : 0;
    d.expected = reference(q, d.keys, d.values, count, g, d.positions, mask);
    const AttentionRequestKey key{100 - i, i + 1};
    specs.push_back({key, kv.view(sequence), g, d.positions}); inputs.push_back({key, q, mask});
    kv.cache.close(sequence); // Closed but unretired: the view still reads its pages.
  }
  const AttentionBatchLimits generous{40, 256, 1 << 20, 1 << 20, 1 << 20};
  PagedAttentionBatch batch(specs, 3, 5, generous);
  auto exact = generous;
  exact.workspace_bytes = batch.workspace_bytes(); exact.output_bytes = batch.output_bytes(); exact.metadata_bytes = batch.metadata_bytes();
  PagedAttentionBatch tight(specs, 3, 5, exact);
  std::shuffle(inputs.begin(), inputs.end(), rng);
  const auto result = tight.run(inputs);
  for (const auto& row : result) compare(row.attention, data[100 - row.key.request].expected, checked);
  for (int field = 0; field < 5; ++field) {
    auto limit = exact;
    if (field == 0) limit.requests = specs.size() - 1;
    if (field == 1) limit.query_rows = 1;
    if (field == 2) --limit.workspace_bytes;
    if (field == 3) --limit.output_bytes;
    if (field == 4) --limit.metadata_bytes;
    rejects([&] { PagedAttentionBatch invalid(specs, 3, 5, limit); });
  }
  auto duplicate = specs; duplicate.back().key = duplicate.front().key;
  rejects([&] { PagedAttentionBatch invalid(duplicate, 3, 5, generous); });
  auto bad = inputs; bad.back().key = bad.front().key;
  std::size_t calls = 0;
  auto counting = [&](bool transpose, std::size_t m, std::size_t n, std::size_t k, auto a, auto b, auto c) {
    ++calls; ScalarAttentionGemm{}(transpose, m, n, k, a, b, c);
  };
  rejects([&] { batch.run(bad, counting); }); require(calls == 0, "Invalid membership reached arithmetic");
  bad = inputs; ++bad.front().key.generation;
  rejects([&] { batch.run(bad, counting); }); require(calls == 0, "Stale identity reached arithmetic");
  bad.pop_back(); rejects([&] { batch.run(bad); });
  auto published = result;
  auto failing = [&](bool transpose, std::size_t m, std::size_t n, std::size_t k, auto a, auto b, auto c) {
    if (++calls == 20) throw std::runtime_error("Injected later-row failure");
    ScalarAttentionGemm{}(transpose, m, n, k, a, b, c);
  };
  rejects([&] { published = batch.run(inputs, failing); });
  require(published.size() == result.size() && published.front().attention.values == result.front().attention.values, "Partial batch escaped failure");
  std::atomic_bool cancelled{false}; calls = 0;
  auto cancelling = [&](bool transpose, std::size_t m, std::size_t n, std::size_t k, auto a, auto b, auto c) {
    ScalarAttentionGemm{}(transpose, m, n, k, a, b, c); if (++calls == 20) cancelled = true;
  };
  rejects([&] { batch.run(inputs, cancelling, &cancelled); });
  rejects([&] { batch.run(inputs, &cancelled); });
  auto* cancellation_pointer = &cancelled;
  rejects([&] { batch.run(inputs, cancellation_pointer); });
  auto reentrant = [&](bool transpose, std::size_t m, std::size_t n, std::size_t k, auto a, auto b, auto c) {
    rejects([&] { batch.run(inputs); }); ScalarAttentionGemm{}(transpose, m, n, k, a, b, c);
  };
  const auto recovered = batch.run(inputs, reentrant);
  for (const auto& row : recovered) compare(row.attention, data[100 - row.key.request].expected, checked);
  // Retained shared prefix, private partial tails, and mutations after planning.
  PagedState shared_kv(8, 4, 1, 1, 16); const auto prefix = shared_kv.open();
  shared_kv.append(prefix, std::array{1.f, 2.f, 3.f}, std::array{4.f, 5.f, 6.f});
  const auto a = shared_kv.cache.fork(prefix), b = shared_kv.cache.fork(prefix);
  shared_kv.append(a, std::array{4.f}, std::array{7.f});
  shared_kv.append(b, std::array{5.f, 6.f}, std::array{8.f, 9.f});
  const std::array<std::size_t, 1> row_a{3}, row_b{4}; const std::array<float, 1> q{1};
  const std::array<AttentionBatchSpec<StatePagesView>, 2> shared{{{{1, 1}, shared_kv.view(a), {1, 1, 1, 1, 1}, row_a},
                                                               {{2, 1}, shared_kv.view(b), {1, 1, 1, 1, 1}, row_b}}};
  PagedAttentionBatch retained(shared, 1, 2, generous);
  // Rewinds and closes release pages to retirement; unretired, the plans still read them.
  shared_kv.cache.rewind(a, 2); shared_kv.cache.close(b); shared_kv.cache.close(prefix);
  const std::array<AttentionBatchInput, 2> queries{{{{2, 1}, {q, 1, 1, 1, 1, 1}}, {{1, 1}, {q, 1, 1, 1, 1, 1}}}};
  const auto kept = retained.run(queries);
  compare(kept[0].attention, reference(queries[0].queries, std::array{1.f, 2.f, 3.f, 4.f}, std::array{4.f, 5.f, 6.f, 7.f}, 4, {1, 1, 1, 1, 1}, row_a, {}), checked);
  compare(kept[1].attention, reference(queries[0].queries, std::array{1.f, 2.f, 3.f, 5.f, 6.f}, std::array{4.f, 5.f, 6.f, 8.f, 9.f}, 5, {1, 1, 1, 1, 1}, row_b, {}), checked);
  return checked;
}
template<class Char> std::string ascii(const Char* value) {
  std::string result;
  for (; *value; ++value) {
    if (static_cast<std::uint32_t>(*value) > 127 || result.size() == 128) throw std::invalid_argument("Bounded ASCII argument required");
    result += static_cast<char>(*value);
  }
  return result;
}
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  try {
    if (argc < 2) throw std::invalid_argument("Choose arithmetic, rotary, integration, failures or openblas LIBRARY BITS SGEMM CONFIG THREADS");
    Checked checked, integrated;
    int integer_bits = 0;
    std::string configuration = "scalar-double-accumulation-fp32-gemm";
    const auto mode = ascii(argv[1]);
    ScalarAttentionGemm scalar;
    if (mode == "arithmetic" && argc == 2) checked = arithmetic(scalar);
    else if (mode == "rotary" && argc == 2) rotary_checks();
    else if (mode == "integration" && argc == 2) integrated = integration(scalar);
    else if (mode == "failures" && argc == 2) failures();
    else if (mode == "batches" && argc == 2) checked = batches();
    else if (mode == "openblas" && argc == 7) {
      const auto bits = ascii(argv[3]);
      if (bits != "32" && bits != "64") throw std::invalid_argument("Explicit BLAS ABI required");
      crit_kernels::OpenBlas blas(std::filesystem::path(argv[2]), bits == "64" ? 64 : 32, ascii(argv[4]), ascii(argv[5]), ascii(argv[6]));
      crit_kernels::OpenBlasAttentionGemm gemm{blas};
      integer_bits = bits == "64" ? 64 : 32; configuration = blas.configuration();
      for (unsigned char c : configuration)
        if (c < 32 || c > 126) throw std::runtime_error("Printable provider configuration required");
      checked = arithmetic(gemm); integrated = integration(gemm); rotary_checks();
    } else throw std::invalid_argument("Invalid attention fixture arguments");
    std::cout << std::setprecision(10) << "{\"schema_version\":1,\"record_kind\":\"attention_check\",\"cases\":" << checked.cases
      << ",\"integration_cases\":" << integrated.cases << ",\"integer_bits\":" << integer_bits
      << ",\"configuration\":" << std::quoted(configuration)
      << ",\"max_output_error\":" << std::max(checked.output_error, integrated.output_error)
      << ",\"max_logsumexp_error\":" << std::max(checked.lse_error, integrated.lse_error) << "}\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
