#include "speculative.h"
#include "../../session-scheduling/code/physical_batches.h"
#include "../../recorded-selection/code/selection.h"
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>

using namespace crit::techniques;
using namespace crit::techniques::physical;
namespace {
// Synthetic integer arithmetic makes continuation equivalence exact. This is
// an executable integration example, not trained weights or a speed benchmark.
constexpr std::size_t context = 96, draft_width = 7, lanes = 16;
constexpr std::uint32_t vocabulary = 17;
constexpr LinearWorkspace layout{0, lanes * sizeof(std::uint32_t)};
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class Error, class F> void rejects(F f) {
  try { f(); } catch (const Error&) { return; }
  throw std::runtime_error("Expected failure");
}

struct Execution {
  BatchLimit limit;
  SplitRules rules;
  HostWorkspace scratch;
  std::size_t calls = 0, fail_on = 0, cancel_on = 0;
  std::atomic_bool cancelled = false;
  Execution(BatchLimit cap, std::size_t padding = 1)
      : limit(cap), rules{true, 1, padding},
        scratch(BatchPlan(context, limit, rules, layout).peak_workspace_bytes()) {}
};

class SyntheticState final : public GreedyTargetState {
  std::shared_ptr<Execution> execution_;
  std::vector<SpecToken> tokens_;
  std::vector<std::uint32_t> states_{1}; // State after each token, plus empty prefix.
  SpecToken prediction(std::size_t length) const {
    return static_cast<SpecToken>((states_.at(length) * 7 + length * 3) % vocabulary);
  }
public:
  explicit SyntheticState(std::shared_ptr<Execution> execution) : execution_(std::move(execution)) {}
  std::unique_ptr<GreedyTargetState> fork() const override { return std::make_unique<SyntheticState>(*this); }
  std::span<const SpecToken> tokens() const noexcept override { return tokens_; }
  SpecToken next_token() const override { return prediction(tokens_.size()); }
  std::uint32_t digest() const { return states_.back(); }
  void extend(std::span<const SpecToken> input) {
    const auto start = tokens_.size();
    if (input.size() > context - start) throw std::length_error("Fixed context exhausted");
    for (auto token : input)
      if (token < 0 || static_cast<std::uint32_t>(token) >= vocabulary) throw std::invalid_argument("Token out of range");
    const BatchPlan plan(input.size(), execution_->limit, execution_->rules, layout,
                         execution_->scratch.allocated_bytes());
    execution_->scratch.run(plan, [&](const BatchShape& shape, std::span<std::byte> scratch) {
      // Batched column projection occupies real scratch. memcpy avoids imposing
      // an alignment/object-lifetime convention on a generic byte allocator.
      for (std::size_t i = 0; i < shape.physical_columns; ++i) {
        for (std::size_t j = 0; j < lanes; ++j) {
          const auto feature = i < shape.columns
              ? static_cast<std::uint32_t>((static_cast<std::size_t>(input[shape.offset + i]) + 1) * (j + 3) + start + shape.offset + i)
              : 0u;
          std::memcpy(scratch.data() + (i * lanes + j) * sizeof(feature), &feature, sizeof(feature));
        }
      }
      // Only valid columns update the ordered recurrence. Padding never consumes
      // positions/state/RNG. Output/state retention is separate from scratch.
      for (std::size_t i = 0; i < shape.columns; ++i) {
        std::uint32_t sum = 0;
        for (std::size_t j = 0; j < lanes; ++j) {
          std::uint32_t feature;
          std::memcpy(&feature, scratch.data() + (i * lanes + j) * sizeof(feature), sizeof(feature));
          sum += feature;
        }
        states_.push_back((states_.back() * 13 + sum) % 65521);
        tokens_.push_back(input[shape.offset + i]);
      }
      ++execution_->calls;
      if (execution_->calls == execution_->fail_on) throw std::runtime_error("Injected settled call failure");
      if (execution_->calls == execution_->cancel_on) execution_->cancelled = true;
    }, &execution_->cancelled);
  }
  void append(SpecToken token) override { extend(std::span<const SpecToken>(&token, 1)); }
  void rewind(std::size_t count) override {
    if (count > tokens_.size()) throw std::out_of_range("Invalid rewind");
    tokens_.resize(count); states_.resize(count + 1);
  }
  std::vector<SpecToken> verify(std::span<const SpecToken> drafts) override {
    const auto start = tokens_.size();
    extend(drafts);
    std::vector<SpecToken> predictions;
    for (std::size_t i = 0; i <= drafts.size(); ++i) predictions.push_back(prediction(start + i));
    return predictions; // D+1 rows in logical order, independent of physical tiling.
  }
};

class StatefulDraft {
  SyntheticState accepted_;
public:
  explicit StatefulDraft(std::shared_ptr<Execution> execution) : accepted_(std::move(execution)) {}
  std::vector<SpecToken> operator()(std::span<const SpecToken> history, std::size_t width) {
    const auto old = accepted_.tokens();
    require(old.size() <= history.size() && std::equal(old.begin(), old.end(), history.begin()), "Draft accepted prefix diverged");
    auto caught_up = accepted_; // Ingestion/catch-up failure leaves accepted draft intact.
    caught_up.extend(history.subspan(old.size()));
    accepted_ = std::move(caught_up);
    auto branch = accepted_;
    std::vector<SpecToken> drafts;
    for (std::size_t i = 0; i < width; ++i) {
      auto next = branch.next_token();
      if (branch.tokens().size() % 11 == 0) next = (next + 1) % vocabulary; // Exercise rejection/repair.
      drafts.push_back(next); branch.append(next);
    }
    return drafts;
  }
  void check_prefix(std::span<const SpecToken> history) const {
    require(accepted_.tokens().size() <= history.size() &&
        std::equal(accepted_.tokens().begin(), accepted_.tokens().end(), history.begin()), "Draft contains rejected tokens");
  }
};

std::vector<SpecToken> prompt() {
  std::vector<SpecToken> result;
  for (SpecToken i = 0; i < 23; ++i) result.push_back(i % vocabulary);
  return result;
}
std::vector<SpecToken> ordinary() {
  SyntheticState state(std::make_shared<Execution>(BatchLimit{1}));
  state.extend(prompt());
  for (int i = 0; i < 48; ++i) state.append(state.next_token());
  return {state.tokens().begin(), state.tokens().end()};
}
struct Run {
  std::vector<SpecToken> history;
  std::vector<std::size_t> proposed;
  std::size_t target_bytes, draft_bytes, target_calls, draft_calls;
};
Run run(std::size_t draft_cap, std::size_t target_cap = 16, std::size_t padding = 1, bool partial = false) {
  auto target_exec = std::make_shared<Execution>(BatchLimit{target_cap}, padding);
  auto draft_exec = std::make_shared<Execution>(BatchLimit{target_cap, draft_cap}, padding);
  auto initial = std::make_unique<SyntheticState>(target_exec);
  initial->extend(prompt());
  GreedySpeculator loop(std::move(initial), {context, draft_width, vocabulary});
  StatefulDraft propose(draft_exec);
  Run result{{}, {}, target_exec->scratch.allocated_bytes(), draft_exec->scratch.allocated_bytes(), 0, 0};
  std::size_t round_number = 0;
  while (loop.history().size() < 71) {
    const auto remaining = 71 - loop.history().size();
    // Zero-proposal rounds leave more accepted work to catch up on next time.
    const auto budget = partial && round_number % 4 == 0 ? 0 : draft_width;
    const auto round = loop.prepare(remaining, budget, propose);
    result.proposed.push_back(round.proposed);
    const auto count = partial ? std::min<std::size_t>(2, round.tokens.size()) : round.tokens.size();
    loop.commit(round.id, count);
    propose.check_prefix(loop.history());
    ++round_number;
  }
  result.history.assign(loop.history().begin(), loop.history().end());
  result.target_calls = target_exec->calls; result.draft_calls = draft_exec->calls;
  // Verify continuation state, not merely the visible output, by a further round.
  SyntheticState control(std::make_shared<Execution>(BatchLimit{1}));
  control.extend(result.history);
  const auto next = loop.prepare(1, 0, propose);
  require(next.tokens.front() == control.next_token(), "Continuation state differs");
  loop.discard(next.id);
  return result;
}

void failures() {
  auto execution = std::make_shared<Execution>(BatchLimit{2});
  auto state = std::make_unique<SyntheticState>(execution);
  state->extend(prompt());
  GreedySpeculator loop(std::move(state), {context, draft_width, vocabulary});
  const auto initial = prompt();
  auto exact = [&](std::span<const SpecToken> history, std::size_t count) {
    SyntheticState serial(std::make_shared<Execution>(BatchLimit{1}));
    serial.extend(history);
    std::vector<SpecToken> drafts;
    for (std::size_t i = 0; i < count; ++i) { drafts.push_back(serial.next_token()); serial.append(drafts.back()); }
    return drafts;
  };
  const auto unchanged = [&] {
    require(!loop.pending() && std::equal(initial.begin(), initial.end(), loop.history().begin(), loop.history().end()), "Failed branch was published");
  };
  execution->fail_on = execution->calls + 2;
  rejects<std::runtime_error>([&] { loop.prepare(8, draft_width, exact); });
  unchanged(); execution->fail_on = 0;
  execution->cancel_on = execution->calls + 2;
  rejects<BatchCancelled>([&] { loop.prepare(8, draft_width, exact); });
  unchanged(); execution->cancelled = false; execution->cancel_on = 0;
  const auto pending = loop.prepare(8, draft_width, exact);
  execution->fail_on = execution->calls + 1; // Failure after materializing the bonus.
  rejects<std::runtime_error>([&] { loop.commit(pending.id, pending.tokens.size()); });
  unchanged(); execution->fail_on = 0;
  const auto retry = loop.prepare(8, draft_width, exact);
  loop.commit(retry.id, retry.tokens.size());
  const auto expected = ordinary();
  require(std::equal(loop.history().begin(), loop.history().end(), expected.begin()), "Retry continuation differs");

  auto draft_execution = std::make_shared<Execution>(BatchLimit{16, 4});
  StatefulDraft draft(draft_execution);
  draft_execution->fail_on = 2; // Fail partway through prompt ingestion.
  rejects<std::runtime_error>([&] { draft(initial, draft_width); });
  draft.check_prefix({}); // No partially ingested accepted draft survived.
  draft_execution->fail_on = 0;
  require(draft(initial, draft_width).size() == draft_width, "Draft ingestion did not recover");
}
} // namespace

int main() { try {
  const auto expected = ordinary(), baseline = run(0).history;
  require(baseline == expected, "Inherited physical cap differs from ordinary output");
  SyntheticState serial(std::make_shared<Execution>(BatchLimit{1}));
  SyntheticState tiled(std::make_shared<Execution>(BatchLimit{16, 4}, 4));
  serial.extend(expected); tiled.extend(expected);
  require(serial.digest() == tiled.digest(), "Tiling changed complete recurrent state");
  serial.rewind(29); tiled.rewind(29);
  require(serial.digest() == tiled.digest(), "Tiling changed rewind state");
  for (int i = 0; i < 12; ++i) {
    require(serial.next_token() == tiled.next_token(), "Rewind continuation prediction differs");
    serial.append(serial.next_token()); tiled.append(tiled.next_token());
    require(serial.digest() == tiled.digest(), "Rewind continuation state differs");
  }
  const auto inherited = run(0), smaller = run(4);
  require(inherited.proposed == smaller.proposed && smaller.history == expected, "Physical cap changed logical proposals/output");
  require(inherited.target_bytes == smaller.target_bytes && smaller.draft_bytes < inherited.draft_bytes, "Draft cap did not independently shrink allocation");
  require(inherited.target_calls == smaller.target_calls && smaller.draft_calls > inherited.draft_calls, "Draft cap did not independently change execution");
  for (std::size_t cap = 1; cap <= 17; ++cap) {
    const auto result = run(cap);
    require(result.history == expected && result.proposed == inherited.proposed, "Cap-dependent speculative continuation");
    require(run(cap, 16, 1, true).history == expected, "Partial/zero-budget continuation");
  }
  require(run(4, 16, 4).history == expected && run(1, 1).history == expected, "Padding or target cap changed continuation");
  failures();

  // Stable candidate order and full configuration identity are engine supplied.
  // Record measured complete synthetic runs; this demonstrates selection plumbing,
  // not a useful hardware tuning result. Serving replay must not time candidates.
  constexpr std::array<std::size_t, 4> caps{0, 1, 4, 8};
  const std::string key = "synthetic-integer-v1-host-layout16-u32-context96-target16-pad1-width7-caps0_1_4_8";
  selection::Table tuning({}, true);
  const auto chosen = tuning.choose(key, static_cast<std::uint32_t>(caps.size()), [&](std::uint32_t index) {
    const auto start = std::chrono::steady_clock::now();
    const auto measured = run(caps[index]);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    require(measured.history == expected, "Tuning accepted an incorrect configuration");
    return ms;
  });
  selection::Table replay(selection::Table::parse(tuning.to_json()));
  const auto restored = replay.choose(key, static_cast<std::uint32_t>(caps.size()), [](auto) -> double { throw std::runtime_error("Serving timed a candidate"); });
  require(restored.source == selection::Source::recorded && chosen.index == restored.index, "Recorded cap changed");
  // Rebuild/revalidate the exact plan at load; a table index is not admission.
  require(run(caps[restored.index]).history == expected, "Restored cap invalid");
  std::cout << "Fixed logical draft width " << draft_width << ", target scratch " << smaller.target_bytes
            << " bytes; independent draft cap reduces payload " << inherited.draft_bytes << " -> " << smaller.draft_bytes << " bytes\n"
            << "48 tokens match ordinary decoding across physical caps, padding and partial publication; failure recovery passed\n"
            << "Measured toy-run choice round-tripped without serving-time tuning: " << tuning.to_json() << '\n';
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; } }
