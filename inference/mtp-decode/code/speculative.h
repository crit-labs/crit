#pragma once
#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace crit::techniques {
using SpecToken = std::int32_t;

// A synchronous, fully materialized accepted prefix plus its next greedy token.
// fork() must isolate every mutable state component (COW is allowed). Methods
// settle all physical work before return/throw; destructors release settled state.
// Sampling/tie rules, weights, tokenizer and position conventions stay identical.
class GreedyTargetState {
public:
  virtual ~GreedyTargetState() = default;
  virtual std::unique_ptr<GreedyTargetState> fork() const = 0;
  virtual std::span<const SpecToken> tokens() const noexcept = 0;
  virtual SpecToken next_token() const = 0;
  virtual void append(SpecToken token) = 0;
  // Restore all state, including next-token logits, at this materialized length.
  virtual void rewind(std::size_t length) = 0;
  // Override with one causal target pass for useful acceleration. The default is
  // an executable serial fallback. Result i uses prefix + drafts[:i].
  virtual std::vector<SpecToken> verify(std::span<const SpecToken> drafts) {
    std::vector<SpecToken> predictions;
    if (drafts.size() == std::numeric_limits<std::size_t>::max()) throw std::length_error("Draft extent overflow");
    predictions.reserve(drafts.size() + 1);
    predictions.push_back(next_token());
    for (auto token : drafts) { append(token); predictions.push_back(next_token()); }
    return predictions;
  }
};

// Stateless bounded-window prompt/history copying. No speculative tokens enter
// the next proposal: it reads only the loop's committed history. Unlike the Python
// index, this small native proposer scans at most lookback prior start positions.
class PromptCopyProposer {
  std::size_t window_, lookback_;
public:
  PromptCopyProposer(std::size_t window, std::size_t lookback) : window_(window), lookback_(lookback) {
    if (!window || !lookback) throw std::invalid_argument("Positive copy window and lookback required");
  }
  std::vector<SpecToken> operator()(std::span<const SpecToken> history, std::size_t limit) const {
    if (!limit || history.size() <= window_) return {};
    const auto suffix = history.last(window_);
    std::vector<SpecToken> best;
    const auto last = history.size() - window_ - 1;
    const auto first = last >= lookback_ - 1 ? last - (lookback_ - 1) : 0;
    for (auto start = last;; --start) {
      if (std::equal(suffix.begin(), suffix.end(), history.begin() + static_cast<std::ptrdiff_t>(start))) {
        const auto count = std::min(limit, history.size() - start - window_);
        if (count > best.size()) {
          const auto source = history.subspan(start + window_, count);
          best.assign(source.begin(), source.end());
          if (count == limit) break;
        }
      }
      if (start == first) break;
    }
    return best;
  }
};

struct SpeculationLimits {
  std::size_t context, max_draft;
  std::uint32_t vocabulary;
};
struct SpeculativeRound {
  std::uint64_t id;
  std::span<const SpecToken> tokens; // Borrowed until commit/discard/destruction.
  std::size_t proposed, accepted_drafts;
};
struct SpeculativeCommit {
  std::vector<SpecToken> tokens;
  std::size_t proposed, accepted_drafts;
  bool correction_or_bonus, stopped;
};

class GreedySpeculator {
  struct Pending {
    std::uint64_t id;
    std::unique_ptr<GreedyTargetState> branch;
    std::vector<SpecToken> drafts, predictions, output;
    std::size_t matched;
  };
  SpeculationLimits limits_;
  std::unique_ptr<GreedyTargetState> state_;
  std::vector<SpecToken> history_, stops_;
  std::optional<Pending> pending_;
  std::uint64_t generation_ = 0;
  bool stopped_ = false, poisoned_ = false;
  std::atomic_flag running_ = ATOMIC_FLAG_INIT;
  struct Guard {
    std::atomic_flag& flag;
    explicit Guard(std::atomic_flag& value) : flag(value) {
      if (flag.test_and_set(std::memory_order_acquire)) throw std::logic_error("Speculator already in use");
    }
    ~Guard() { flag.clear(std::memory_order_release); }
  };
  void token(SpecToken value) const {
    if (value < 0 || static_cast<std::uint32_t>(value) >= limits_.vocabulary)
      throw std::invalid_argument("Speculative token outside vocabulary");
  }
  bool is_stop(SpecToken value) const { return std::binary_search(stops_.begin(), stops_.end(), value); }
  static void stop(const std::atomic_bool* cancelled) {
    if (cancelled && cancelled->load(std::memory_order_relaxed)) throw std::runtime_error("Speculation cancelled");
  }
  void parent() {
    const auto actual = state_->tokens();
    if (poisoned_ || !std::equal(history_.begin(), history_.end(), actual.begin(), actual.end())) {
      poisoned_ = true;
      throw std::logic_error("Committed target history changed outside publication; speculator poisoned");
    }
  }
  void image(const GreedyTargetState& state, std::span<const SpecToken> suffix) const {
    const auto actual = state.tokens();
    if (actual.size() != history_.size() + suffix.size() ||
        !std::equal(history_.begin(), history_.end(), actual.begin()) ||
        !std::equal(suffix.begin(), suffix.end(), actual.begin() + static_cast<std::ptrdiff_t>(history_.size())))
      throw std::logic_error("Target branch does not contain the expected materialized prefix");
  }
public:
  GreedySpeculator(std::unique_ptr<GreedyTargetState> state, SpeculationLimits limits,
                  std::span<const SpecToken> stop_tokens = {}) : limits_(limits), state_(std::move(state)) {
    if (!state_ || !limits.context || !limits.vocabulary ||
        limits.vocabulary > std::uint64_t{1} + static_cast<std::uint32_t>(std::numeric_limits<SpecToken>::max()))
      throw std::invalid_argument("Target, context and signed-32-bit vocabulary required");
    const auto initial = state_->tokens();
    if (initial.empty() || initial.size() > limits.context) throw std::invalid_argument("Nonempty materialized prompt within capacity required");
    for (auto value : initial) token(value);
    for (auto value : stop_tokens) token(value);
    token(state_->next_token());
    history_.assign(initial.begin(), initial.end()); stops_.assign(stop_tokens.begin(), stop_tokens.end());
    std::sort(stops_.begin(), stops_.end()); stops_.erase(std::unique(stops_.begin(), stops_.end()), stops_.end());
  }
  GreedySpeculator(const GreedySpeculator&) = delete;
  GreedySpeculator& operator=(const GreedySpeculator&) = delete;
  std::span<const SpecToken> history() const noexcept { return history_; }
  bool finished() const noexcept { return stopped_ || history_.size() == limits_.context; }
  bool poisoned() const noexcept { return poisoned_; }
  bool pending() const noexcept { return pending_.has_value(); }
  // Serialized access includes these borrowed views. Only cancellation may be
  // updated concurrently. The guard rejects overlapping/reentrant mutations.
  template<class Proposer> requires std::same_as<std::invoke_result_t<Proposer&, std::span<const SpecToken>, std::size_t>, std::vector<SpecToken>>
  SpeculativeRound prepare(std::size_t output_budget, std::size_t draft_budget, Proposer& propose,
                           const std::atomic_bool* cancelled = nullptr) {
    Guard guard(running_);
    parent(); stop(cancelled);
    if (pending_) throw std::logic_error("Commit or discard the pending round first");
    if (!output_budget || finished()) throw std::length_error("No remaining speculative output/context budget");
    if (generation_ == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Speculative round identity exhausted");
    const auto remaining = std::min(output_budget, limits_.context - history_.size());
    const auto limit = std::min({remaining, limits_.max_draft, draft_budget});
    try {
      auto drafts = limit ? propose(history_, limit) : std::vector<SpecToken>{};
      parent(); stop(cancelled);
      if (drafts.size() > limit) throw std::length_error("Proposer exceeded its token budget");
      for (auto value : drafts) token(value);
      // A proposal cannot usefully continue past a stop, even during verification.
      const auto end = std::find_if(drafts.begin(), drafts.end(), [&](auto value) { return is_stop(value); });
      if (end != drafts.end()) drafts.resize(static_cast<std::size_t>(end - drafts.begin()) + 1);
      auto branch = state_->fork();
      if (branch.get() == state_.get()) { branch.release(); throw std::logic_error("Target fork returned the committed owner"); }
      if (!branch) throw std::logic_error("Target fork returned no state");
      parent(); image(*branch, {}); stop(cancelled);
      const auto first = branch->next_token(); token(first);
      if (first != state_->next_token()) throw std::logic_error("Fork changed the committed next-token prediction");
      auto predictions = branch->verify(drafts);
      parent(); image(*branch, drafts); stop(cancelled);
      if (predictions.size() != drafts.size() + 1) throw std::logic_error("Verification needs D+1 conditioned predictions");
      for (auto value : predictions) token(value);
      if (predictions.front() != first || predictions.back() != branch->next_token())
        throw std::logic_error("Target verification endpoints differ from retained state");
      std::vector<SpecToken> output;
      output.reserve(std::min(remaining, drafts.size() + 1));
      std::size_t matched = 0;
      while (matched < drafts.size() && output.size() < remaining && drafts[matched] == predictions[matched]) {
        output.push_back(drafts[matched++]);
        if (is_stop(output.back())) break;
      }
      if (output.size() < remaining && (output.empty() || !is_stop(output.back()))) output.push_back(predictions[matched]);
      stop(cancelled);
      pending_.emplace(Pending{generation_ + 1, std::move(branch), std::move(drafts), std::move(predictions), std::move(output), matched});
      ++generation_;
      return {pending_->id, pending_->output, pending_->drafts.size(), pending_->matched};
    } catch (...) { parent(); throw; }
  }
  SpeculativeCommit commit(std::uint64_t id, std::size_t accepted, const std::atomic_bool* cancelled = nullptr) {
    Guard guard(running_);
    parent();
    if (!pending_ || pending_->id != id) throw std::logic_error("Unknown, stale or consumed speculative round");
    if (accepted > pending_->output.size()) throw std::out_of_range("Commit exceeds the verified output prefix");
    auto round = std::move(*pending_); pending_.reset(); // A failed commit consumes its branch.
    try {
      stop(cancelled);
      SpeculativeCommit result{{round.output.begin(), round.output.begin() + static_cast<std::ptrdiff_t>(accepted)},
          round.drafts.size(), std::min(accepted, round.matched), accepted > round.matched, false};
      if (!accepted) return result;
      auto next_history = history_;
      next_history.insert(next_history.end(), result.tokens.begin(), result.tokens.end());
      round.branch->rewind(history_.size() + result.accepted_drafts);
      parent(); image(*round.branch, std::span<const SpecToken>(result.tokens).first(result.accepted_drafts)); stop(cancelled);
      if (round.branch->next_token() != round.predictions[result.accepted_drafts])
        throw std::logic_error("Rewind did not restore the verified next-token prediction");
      if (result.correction_or_bonus) {
        round.branch->append(result.tokens.back());
        parent(); image(*round.branch, result.tokens); stop(cancelled);
      }
      token(round.branch->next_token()); parent(); stop(cancelled);
      result.stopped = is_stop(result.tokens.back());
      // All potentially failing work precedes these nonthrowing ownership swaps.
      state_.swap(round.branch); history_.swap(next_history); stopped_ = result.stopped;
      return result;
    } catch (...) { parent(); throw; }
  }
  void discard(std::uint64_t id) {
    Guard guard(running_);
    if (!pending_ || pending_->id != id) throw std::logic_error("Unknown, stale or consumed speculative round");
    pending_.reset();
  }
};
} // namespace crit::techniques
