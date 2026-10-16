#include "recurrent_recovery.h"
#include <array>
#include <iostream>
#include <string_view>

using namespace crit::techniques::recurrent;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function function) {
  try { function(); } catch (const std::exception&) { return; }
  throw std::runtime_error("Expected recurrent-state rejection");
}
void close(std::span<const double> actual, std::span<const long double> expected) {
  require(actual.size() == expected.size(), "Oracle shape differs");
  for (std::size_t i = 0; i < actual.size(); ++i)
    require(std::abs(static_cast<long double>(actual[i]) - expected[i]) <=
                2e-12L * (1 + std::abs(expected[i])), "Independent recurrent oracle differs");
}

// Independent matrix equation: H' = decay * (I - beta*k*k^T)*H + beta*k*v^T.
// This materializes a long-double transition matrix rather than using the
// implementation's residual update or recovery routine as its oracle.
std::pair<std::vector<long double>, std::vector<long double>> oracle_step(
    std::span<const long double> state, const Step& step, Geometry geometry) {
  auto normalized = [&](const std::vector<double>& input, bool enabled) {
    std::vector<long double> result(input.begin(), input.end());
    if (enabled) {
      long double squared = geometry.epsilon;
      for (auto x : result) squared += x * x;
      for (auto& x : result) x /= std::sqrt(squared);
    }
    return result;
  };
  const auto k = normalized(step.key, geometry.normalize_keys);
  const auto q = normalized(step.query, geometry.normalize_queries);
  std::vector<long double> transition(geometry.keys * geometry.keys);
  for (std::size_t row = 0; row < geometry.keys; ++row)
    for (std::size_t column = 0; column < geometry.keys; ++column)
      transition[row * geometry.keys + column] = std::exp(static_cast<long double>(step.log_decay)) *
          ((row == column ? 1.0L : 0.0L) - static_cast<long double>(step.beta) * k[row] * k[column]);
  std::vector<long double> next(geometry.keys * geometry.values);
  for (std::size_t row = 0; row < geometry.keys; ++row)
    for (std::size_t v = 0; v < geometry.values; ++v) {
      auto& value = next[row * geometry.values + v];
      value = static_cast<long double>(step.beta) * k[row] * step.value[v];
      for (std::size_t column = 0; column < geometry.keys; ++column)
        value += transition[row * geometry.keys + column] * state[column * geometry.values + v];
    }
  std::vector<long double> outputs(geometry.values);
  for (std::size_t v = 0; v < geometry.values; ++v)
    for (std::size_t row = 0; row < geometry.keys; ++row)
      outputs[v] += q[row] * next[row * geometry.values + v] * geometry.query_scale;
  return {next, outputs};
}
Step step(Geometry geometry, std::size_t number) {
  Step result;
  for (std::size_t k = 0; k < geometry.keys; ++k) {
    result.query.push_back(std::sin(static_cast<double>(number + k + 1)) * 0.4);
    result.key.push_back(std::cos(static_cast<double>(number * 2 + k + 1)) * 0.5);
  }
  for (std::size_t v = 0; v < geometry.values; ++v)
    result.value.push_back(static_cast<double>((number + v) % 7) / 5.0 - 0.5);
  result.log_decay = -static_cast<double>(number % 4) / 5.0;
  result.beta = static_cast<double>(number % 5) / 4.0;
  return result;
}

void arithmetic_and_boundaries() {
  for (std::size_t keys : {std::size_t{1}, std::size_t{3}, std::size_t{7}})
    for (std::size_t values : {std::size_t{1}, std::size_t{4}})
      for (bool normalization : {false, true}) {
        const Geometry geometry{keys, values, 1e-6, 0.7, normalization, normalization};
        Snapshot initial{62, std::vector<double>(keys * values, 0.1), {{9}}};
        std::vector<Node> nodes;
        const std::vector<std::size_t> path{0, 2, 5, 7};
        for (std::size_t i = 0; i < 8; ++i) nodes.push_back({no_parent, step(geometry, i + 1)});
        nodes[2].parent = 0; nodes[5].parent = 2; nodes[7].parent = 5;
        std::vector<std::vector<long double>> states{
            std::vector<long double>(initial.matrix.begin(), initial.matrix.end())};
        std::vector<long double> output;
        for (auto physical : path) {
          auto next = oracle_step(states.back(), nodes[physical].step, geometry);
          states.push_back(next.first);
          output.insert(output.end(), next.second.begin(), next.second.end());
        }
        for (std::size_t accepted = 0; accepted <= path.size(); ++accepted) {
          Recovery recovery(geometry, initial);
          auto prepared = recovery.prepare(nodes, path);
          close(prepared.outputs, output);
          require(recovery.state().matrix == initial.matrix, "Verify mutated committed recurrence");
          std::size_t boundary = std::min(std::size_t{2}, accepted);
          const auto result = recovery.commit(prepared.id, accepted, initial.after_tokens + boundary);
          close(result.state.matrix, states[accepted]);
          close(result.checkpoint->matrix, states[boundary]);
          require(result.state.after_tokens == 62 + accepted, "Accepted absolute position differs");
          require(result.checkpoint->after_tokens == 62 + boundary, "Checkpoint absolute position differs");
          if (accepted >= 2)
            require(checkpoint_node(62, std::span(path).first(accepted), 64) == 2,
                    "Boundary used the next token or accepted ordinal as physical node");
          rejects([&] { recovery.commit(prepared.id, 0); });
        }
        // Restore the exact boundary state and independently continue. A one-token
        // ahead checkpoint would fail here even if its endpoint looks plausible.
        Recovery source(geometry, initial);
        auto prepared = source.prepare(nodes, path);
        auto result = source.commit(prepared.id, 4, 64);
        Recovery restored(geometry, *result.checkpoint);
        auto continuation = restored.prepare({{no_parent, nodes[5].step}, {0, nodes[7].step}}, {0, 1});
        auto continued = restored.commit(continuation.id, 2);
        close(continued.state.matrix, states.back());
      }
  require(checkpoint_node(63, std::array<std::size_t, 3>{1, 4, 7}, 64) == 1,
          "Boundary one token after prefix differs");
  require(checkpoint_node(64, std::array<std::size_t, 3>{1, 4, 7}, 64) == std::nullopt,
          "Initial boundary unexpectedly has a verify node");
  rejects([] { checkpoint_node(65, std::array<std::size_t, 1>{2}, 64); });
  rejects([] { checkpoint_node(63, std::array<std::size_t, 1>{2}, 65); });
}

void transactions() {
  const Geometry geometry{2, 2};
  Snapshot initial{63, {0.1, 0.2, 0.3, 0.4}, {{7}, {8}}};
  Recovery recovery(geometry, initial);
  const std::vector<Node> nodes{{no_parent, step(geometry, 1)}, {0, step(geometry, 2)}};
  auto unchanged = [&] {
    require(recovery.state().matrix == initial.matrix && recovery.state().siblings == initial.siblings &&
            recovery.state().after_tokens == initial.after_tokens, "Failed/zero commit mutated original bundle");
  };
  auto prepared = recovery.prepare(nodes, {0, 1});
  int calls = 0;
  recovery.commit(prepared.id, 0, std::nullopt, [&](auto, auto&) { ++calls; });
  require(calls == 0, "Zero commit ran sibling mutation"); unchanged();
  prepared = recovery.prepare(nodes, {0, 1});
  rejects([&] { recovery.commit(prepared.id, 2, 64, [&](auto, Siblings& siblings) {
    siblings.front().push_back(3);
    if (++calls == 2) throw std::runtime_error("Failure after private checkpoint and partial sibling update");
  }); });
  unchanged(); require(!recovery.pending(), "Failed round was retained as retryable");
  rejects([&] { recovery.commit(prepared.id, 1); });
  prepared = recovery.prepare(nodes, {0, 1});
  rejects([&] { recovery.commit(prepared.id, 3); }); unchanged();
  prepared = recovery.prepare(nodes, {0, 1});
  rejects([&] { recovery.commit(prepared.id, 1, 65); }); unchanged();
  std::atomic_bool cancellation{true};
  rejects([&] { recovery.prepare(nodes, {0, 1}, &cancellation); }); unchanged();
  cancellation = false; prepared = recovery.prepare(nodes, {0, 1});
  rejects([&] { recovery.commit(prepared.id, 2, std::nullopt,
      [&](auto, auto&) { cancellation = true; }, &cancellation); });
  unchanged(); cancellation = false;
  prepared = recovery.prepare(nodes, {0, 1});
  rejects([&] { recovery.prepare(nodes, {0}); });
  auto result = recovery.commit(prepared.id, 1, std::nullopt, [&](auto path, Siblings& siblings) {
    rejects([&] { recovery.prepare(nodes, {0}); });
    siblings.front().push_back(static_cast<std::uint8_t>(path.back()));
  });
  require(result.state.after_tokens == 64 && result.state.siblings.front() == std::vector<std::uint8_t>({7, 0}),
          "Private siblings were not published with accepted recurrence");
  std::vector<long double> serial(initial.matrix.begin(), initial.matrix.end());
  serial = oracle_step(serial, nodes.front().step, geometry).first;
  for (std::size_t round = 0; round < 12; ++round) {
    auto next_step = step(geometry, round + 5);
    auto next = recovery.prepare({{no_parent, next_step}}, {0});
    serial = oracle_step(serial, next_step, geometry).first;
    recovery.commit(next.id, 1);
    close(recovery.state().matrix, serial);
  }
  prepared = recovery.prepare(nodes, {0});
  recovery.discard(prepared.id);
  rejects([&] { recovery.discard(prepared.id); });
}

// GCC 13 reports a false -Wmaybe-uninitialized when copying an empty optional into commit().
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
void chunked_continuation() {
  const Geometry geometry{3, 2, 1e-6, 0.7, true, true};
  const Snapshot initial{0, std::vector<double>(6, 0.1), {{0}, {0}}};
  std::vector<std::vector<long double>> expected{
      std::vector<long double>(initial.matrix.begin(), initial.matrix.end())};
  for (std::size_t i = 0; i < 17; ++i)
    expected.push_back(oracle_step(expected.back(), step(geometry, i + 1), geometry).first);
  for (std::size_t chunk : {std::size_t{1}, std::size_t{3}, std::size_t{8}, std::size_t{17}}) {
    for (std::uint64_t boundary : {std::uint64_t{7}, std::uint64_t{8}, std::uint64_t{9}}) {
      Recovery recovery(geometry, initial);
      std::optional<Snapshot> checkpoint;
      for (std::size_t start = 0; start < 17; start += chunk) {
        const auto count = std::min(chunk, std::size_t{17} - start);
        std::vector<Node> nodes;
        std::vector<std::size_t> path;
        for (std::size_t i = 0; i < count; ++i) {
          nodes.push_back({i == 0 ? no_parent : i - 1, step(geometry, start + i + 1)});
          path.push_back(i);
        }
        const auto prepared = recovery.prepare(nodes, path);
        const auto capture = start < boundary && boundary <= start + count ?
            std::optional<std::uint64_t>(boundary) : std::nullopt;
        auto result = recovery.commit(prepared.id, count, capture, [&](auto accepted, Siblings& siblings) {
          // Synthetic codecs retain each sibling's exact after-token frontier.
          // Recovery supplies a prefix of physical nodes, never the next node.
          const auto after = static_cast<std::uint8_t>(start + accepted.size());
          siblings = {{after}, {after}};
        });
        if (result.checkpoint) checkpoint = result.checkpoint;
        close(result.state.matrix, expected[start + count]);
        // A no-draft/no-step round after successful acceptance must preserve all
        // committed siblings and recurrence, not recover an older verify column.
        auto empty = recovery.prepare({}, {});
        const auto unchanged = recovery.commit(empty.id, 0, result.state.after_tokens);
        close(unchanged.state.matrix, expected[start + count]);
        require(unchanged.state.siblings == result.state.siblings &&
                unchanged.checkpoint->siblings == result.state.siblings,
                "Zero-step round changed committed sibling state");
      }
      require(checkpoint && checkpoint->after_tokens == boundary &&
              checkpoint->siblings == Siblings{{static_cast<std::uint8_t>(boundary)}, {static_cast<std::uint8_t>(boundary)}},
              "Checkpoint siblings captured different after-token boundaries");
      close(checkpoint->matrix, expected[static_cast<std::size_t>(boundary)]);
      Recovery restored(geometry, *checkpoint);
      for (std::size_t i = static_cast<std::size_t>(boundary); i < 17; ++i) {
        auto prepared = restored.prepare({{no_parent, step(geometry, i + 1)}}, {0});
        const auto after = static_cast<std::uint8_t>(i + 1);
        const auto result = restored.commit(prepared.id, 1, std::nullopt, [&](auto, Siblings& siblings) {
          siblings = {{after}, {after}};
        });
        close(result.state.matrix, expected[i + 1]);
        require(result.state.siblings == Siblings{{after}, {after}}, "Restored siblings did not continue with recurrence");
      }
      close(restored.state().matrix, expected.back());
    }
  }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

void rejection() {
  const Geometry geometry{2, 1};
  const Snapshot initial{0, {0, 0}, {}};
  Recovery recovery(geometry, initial, Limits{2, 2, 1, 1, 1, 1, 8, 100});
  auto good = step(geometry, 1);
  auto invalid = good; invalid.key[0] = std::numeric_limits<double>::quiet_NaN();
  rejects([&] { recovery.prepare({{no_parent, invalid}}, {0}); });
  invalid = good; invalid.log_decay = 1;
  rejects([&] { recovery.prepare({{no_parent, invalid}}, {0}); });
  invalid = good; invalid.beta = 1.1;
  rejects([&] { recovery.prepare({{no_parent, invalid}}, {0}); });
  invalid = good; invalid.query.clear();
  rejects([&] { recovery.prepare({{no_parent, invalid}}, {0}); });
  rejects([&] { recovery.prepare({{1, good}}, {0}); });
  rejects([&] { recovery.prepare({{no_parent, good}}, {1}); });
  rejects([&] { recovery.prepare({{no_parent, good}, {no_parent, good}}, {0, 1}); });
  auto prepared = recovery.prepare({{no_parent, good}}, {0});
  rejects([&] { recovery.commit(prepared.id, 1, std::nullopt,
      [](auto, Siblings& siblings) { siblings = {{1, 2}}; }); });
  require(recovery.state().matrix == initial.matrix, "Bound failure published state");
  rejects([&] { Recovery bad({0, 1}, initial); });
  rejects([&] { Recovery bad({2, 1, 0}, initial); });
  Recovery no_space(geometry, Snapshot{8, {0, 0}, {}}, Limits{2, 2, 1, 1, 1, 1, 8, 100});
  rejects([&] { no_space.prepare({{no_parent, good}}, {0}); });
}

int main() {
  try {
    arithmetic_and_boundaries(); transactions(); chunked_continuation(); rejection();
    std::cout << "Recurrent recovery arithmetic, checkpoint boundaries and atomic commits passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
