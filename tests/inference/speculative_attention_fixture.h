#pragma once
#include "attention.h"
#include "speculative.h"
#include "state_pages_fixture.h"
#include <array>

// Actual native CPU arithmetic using synthetic weights. This is an integration
// fixture, not a trained language model or a quality/performance qualification.
// KV lives in one Core StateCache over StatePages; every target and fork is a sequence.
struct AttentionWeights {
  PagedState kv{128, 4, 4, 4, 512, 32};
  std::size_t forwards = 0, rows = 0, verifications = 0;
  bool fail_repair = false;
};
class AttentionTarget final : public crit::techniques::GreedyTargetState {
  using Token = crit::techniques::SpecToken;
  std::shared_ptr<AttentionWeights> model_;
  crit::core::SequenceId sequence_;
  std::vector<Token> tokens_, predictions_;
  void extend(std::span<const Token> input) {
    using namespace crit::techniques;
    if (input.empty()) return;
    const auto count = input.size(), start = tokens_.size();
    std::vector<float> q(count * 8), k(count * 4), v(count * 4);
    std::vector<std::uint64_t> rope_positions(count);
    std::vector<std::size_t> rows(count);
    for (std::size_t i = 0; i < count; ++i) {
      rows[i] = start + i; rope_positions[i] = start + i;
      for (std::size_t d = 0; d < 4; ++d) {
        const double base = static_cast<double>(input[i] + 1) * static_cast<double>(d + 1);
        k[i * 4 + d] = static_cast<float>(std::sin(base * .37));
        v[i * 4 + d] = static_cast<float>(std::cos(base * .61));
        q[i * 8 + d] = static_cast<float>(std::sin(base * .71));
        q[i * 8 + 4 + d] = static_cast<float>(std::cos(base * .43));
      }
    }
    const auto rotated_q = rotary({q, count, 2, 4, 8, 4}, rope_positions, std::array{1., .01}, RotaryLayout::split_half);
    const auto rotated_k = rotary({k, count, 1, 4, 4, 4}, rope_positions, std::array{1., .01}, RotaryLayout::split_half);
    auto& kv = model_->kv;
    const auto round = kv.pages.stage(kv.cache, sequence_, rotated_k, v);
    AttentionResult output;
    try {
      PagedAttentionPlan plan(kv.view(sequence_, true), {2, 1, 4, 4, .5}, rows, 8, 4, 8192);
      output = plan.run({rotated_q, count, 2, 4, 8, 4});
    } catch (...) { kv.cache.discard(sequence_, round.id); throw; }
    std::vector<Token> next(count);
    for (std::size_t i = 0; i < count; ++i) {
      double best = -std::numeric_limits<double>::infinity();
      for (Token token = 0; token < 5; ++token) {
        double logit = 0;
        for (std::size_t d = 0; d < 8; ++d)
          logit += output.values[i * 8 + d] * std::sin(static_cast<double>(token + 1) * static_cast<double>(d + 1) * .29);
        if (logit > best) { best = logit; next[i] = token; }
      }
    }
    tokens_.insert(tokens_.end(), input.begin(), input.end());
    predictions_.insert(predictions_.end(), next.begin(), next.end());
    kv.cache.commit(sequence_, round.id, std::vector<crit::core::Token>(input.begin(), input.end()));
    kv.settle(); ++model_->forwards; model_->rows += count;
  }
public:
  AttentionTarget(std::shared_ptr<AttentionWeights> model, std::span<const Token> prompt)
      : model_(std::move(model)), sequence_(model_->kv.open()) {
    try { extend(prompt); } catch (...) { model_->kv.cache.close(sequence_); throw; }
  }
  AttentionTarget(const AttentionTarget& other)
      : model_(other.model_), sequence_(model_->kv.cache.fork(other.sequence_)), tokens_(other.tokens_), predictions_(other.predictions_) {}
  AttentionTarget& operator=(const AttentionTarget&) = delete;
  ~AttentionTarget() override { model_->kv.cache.close(sequence_); model_->kv.settle(); }
  std::unique_ptr<GreedyTargetState> fork() const override { return std::make_unique<AttentionTarget>(*this); }
  std::span<const Token> tokens() const noexcept override { return tokens_; }
  Token next_token() const override { return predictions_.back(); }
  void append(Token token) override {
    extend(std::span<const Token>(&token, 1));
    if (model_->fail_repair) throw std::runtime_error("Attention repair failed after materializing the correction");
  }
  void rewind(std::size_t length) override {
    if (!length || length > tokens_.size()) throw std::out_of_range("Invalid attention target rewind");
    model_->kv.cache.rewind(sequence_, static_cast<std::uint32_t>(length)); model_->kv.settle();
    tokens_.resize(length); predictions_.resize(length);
  }
  std::vector<Token> verify(std::span<const Token> drafts) override {
    const auto first = next_token();
    const auto start = predictions_.size();
    extend(drafts); ++model_->verifications;
    std::vector<Token> result{first};
    result.insert(result.end(), predictions_.begin() + static_cast<std::ptrdiff_t>(start), predictions_.end());
    return result;
  }
  crit::techniques::StatePagesView view() const { return model_->kv.view(sequence_); }
};
