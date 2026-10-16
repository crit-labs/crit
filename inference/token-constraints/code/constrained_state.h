#pragma once
// Constrained speculative decoding: Core's request-private TokenMatcher (crit/constraints.h) joined with a raw-logit
// model state into the ProbabilityState that mtp-decode's SampledSpeculator drafts, verifies and repairs.
#include "../../mtp-decode/code/sampled.h"
#include <crit/constraints.h>

namespace crit::techniques {
using crit::core::TokenMatcher;

// The constrained law as a full-vocabulary Distribution: gather the allowed ids in ascending order, apply the
// temperature/top-k/top-p/min-p sampler to that support, scatter back. Masking precedes the filters (filtering the
// full row first could discard every allowed token); top-k above the allowed count keeps the whole set, top-k above
// the vocabulary is an error, and greedy ties choose the smallest original id. Logits must be finite.
inline Distribution constrained_logits(std::span<const float> scores, std::span<const SpecToken> ids,
                                       double temperature = 1, std::size_t top_k = 0, double top_p = 1, double min_p = 0) {
  if(scores.empty() || scores.size() > 1048576 || ids.empty() || ids.size() > scores.size() ||
     !std::all_of(scores.begin(), scores.end(), [](float x) { return std::isfinite(x); }))
    throw std::invalid_argument("Finite logits and nonempty constraint required");
  SpecToken previous = -1;
  for(auto id : ids) {
    if(id <= previous || id < 0 || static_cast<std::size_t>(id) >= scores.size())
      throw std::invalid_argument("Allowed IDs must be sorted, unique and in vocabulary");
    previous = id;
  }
  if(top_k > scores.size()) throw std::invalid_argument("top-k exceeds vocabulary");
  std::vector<float> compact;
  compact.reserve(ids.size());
  for(auto id : ids) compact.push_back(scores[static_cast<std::size_t>(id)]);
  auto law = Distribution::logits(compact, temperature, std::min(top_k, ids.size()), top_p, min_p);
  std::vector<double> full(scores.size());
  for(std::size_t i = 0; i < ids.size(); ++i) full[static_cast<std::size_t>(ids[i])] = law.values()[i];
  return Distribution(std::move(full));
}

// Same isolated-fork/complete-rewind/physical-settlement contract as ProbabilityState.
class LogitState {
public:
  virtual ~LogitState() = default;
  virtual std::unique_ptr<LogitState> fork() const = 0;
  virtual std::span<const SpecToken> tokens() const noexcept = 0;
  virtual std::vector<float> logits() const = 0;
  virtual void append(SpecToken) = 0;
  virtual void rewind(std::size_t) = 0;
};
struct ConstraintSampling { double temperature = 1; std::size_t top_k = 0; double top_p = 1, min_p = 0; };

// The matcher starts after the model's existing prompt. append checks the token on a private matcher copy before the
// model mutates; rewind repairs model and matcher to the same output frontier. A failed model append/rewind poisons
// the state: later sampling, forks and mutations throw and the branch must be retired.
class ConstrainedProbabilityState final : public ProbabilityState {
  std::unique_ptr<LogitState> model_;
  std::unique_ptr<TokenMatcher> matcher_;
  ConstraintSampling sampling_;
  std::size_t prompt_;
  bool poisoned_ = false;
  void alive() const {
    if(poisoned_ || !model_ || !matcher_ || model_->tokens().size() != prompt_ + matcher_->size())
      throw std::logic_error("Constraint/model state is unusable; retire this branch");
  }
  std::unique_ptr<TokenMatcher> fork_matcher() const {
    auto matcher = matcher_->fork();
    if(matcher.get() == matcher_.get()) { (void)matcher.release(); throw std::logic_error("Matcher fork aliases parent"); }
    if(!matcher || matcher->size() != matcher_->size() || matcher->vocabulary() != matcher_->vocabulary() ||
       matcher->ended() != matcher_->ended())
      throw std::logic_error("Matcher fork changed its frontier");
    return matcher;
  }
  ConstrainedProbabilityState(std::unique_ptr<LogitState> model, std::unique_ptr<TokenMatcher> matcher,
                              ConstraintSampling sampling, std::size_t prompt)
      : model_(std::move(model)), matcher_(std::move(matcher)), sampling_(sampling), prompt_(prompt) { alive(); }
public:
  ConstrainedProbabilityState(std::unique_ptr<LogitState> model, std::unique_ptr<TokenMatcher> matcher,
                              ConstraintSampling sampling = {})
      : model_(std::move(model)), matcher_(std::move(matcher)), sampling_(sampling),
        prompt_(model_ ? model_->tokens().size() : 0) {
    if(!model_ || !matcher_ || matcher_->size() || matcher_->ended())
      throw std::invalid_argument("Model and fresh output matcher required");
    (void)next();
  }
  std::unique_ptr<ProbabilityState> fork() const override {
    alive();
    auto matcher = fork_matcher();
    auto model = model_->fork();
    if(model.get() == model_.get()) { (void)model.release(); throw std::logic_error("Model fork aliases parent"); }
    if(!model || !std::equal(model->tokens().begin(), model->tokens().end(), model_->tokens().begin(), model_->tokens().end()))
      throw std::logic_error("Model fork changed its prefix");
    return std::unique_ptr<ProbabilityState>(
        new ConstrainedProbabilityState(std::move(model), std::move(matcher), sampling_, prompt_));
  }
  std::span<const SpecToken> tokens() const noexcept override { return model_->tokens(); }
  Distribution next() const override {
    alive();
    auto scores = model_->logits();
    if(scores.size() != matcher_->vocabulary()) throw std::invalid_argument("Model/constraint vocabulary differs");
    const auto allowed = matcher_->allowed();
    return constrained_logits(scores, allowed, sampling_.temperature, sampling_.top_k, sampling_.top_p, sampling_.min_p);
  }
  void append(SpecToken token) override {
    alive();
    auto matcher = fork_matcher();
    matcher->append(token); // validate before model mutation
    if(matcher->size() != matcher_->size() + 1 || matcher->vocabulary() != matcher_->vocabulary())
      throw std::logic_error("Matcher append changed its extent or vocabulary");
    try {
      model_->append(token);
      if(model_->tokens().size() != prompt_ + matcher->size() || model_->tokens().back() != token)
        throw std::logic_error("Model append did not materialize the selected token");
      matcher_ = std::move(matcher);
    } catch(...) { poisoned_ = true; throw; }
  }
  void rewind(std::size_t count) override {
    alive();
    if(count < prompt_) throw std::out_of_range("Cannot rewind into prompt");
    auto matcher = fork_matcher();
    matcher->rewind(count - prompt_);
    if(matcher->size() != count - prompt_ || matcher->vocabulary() != matcher_->vocabulary())
      throw std::logic_error("Matcher rewind did not restore its frontier");
    try {
      model_->rewind(count);
      if(model_->tokens().size() != count) throw std::logic_error("Model rewind did not restore its frontier");
      matcher_ = std::move(matcher);
    } catch(...) { poisoned_ = true; throw; }
  }
};
} // namespace crit::techniques
