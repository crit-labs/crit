#include "speculative.h"
#include <array>
#include <iostream>

// Synthetic cycle target, not trained weights. A real target retains logits/KV
// and overrides verify() with a single correctly masked block forward pass.
class Cycle final : public crit::techniques::GreedyTargetState {
  std::vector<crit::techniques::SpecToken> history_;
public:
  explicit Cycle(std::span<const crit::techniques::SpecToken> prompt) : history_(prompt.begin(), prompt.end()) {}
  std::unique_ptr<GreedyTargetState> fork() const override { return std::make_unique<Cycle>(*this); }
  std::span<const crit::techniques::SpecToken> tokens() const noexcept override { return history_; }
  crit::techniques::SpecToken next_token() const override { return (history_.back() + 1) % 5; }
  void append(crit::techniques::SpecToken token) override { history_.push_back(token); }
  void rewind(std::size_t length) override {
    if (!length || length > history_.size()) throw std::out_of_range("Invalid cycle rewind");
    history_.resize(length);
  }
};
int main() {
  using namespace crit::techniques;
  try {
    const std::array<SpecToken, 8> prompt{0, 1, 2, 3, 4, 0, 1, 2};
    GreedySpeculator loop(std::make_unique<Cycle>(prompt), {40, 6, 5});
    PromptCopyProposer propose(2, 32);
    std::vector<SpecToken> generated;
    std::size_t rounds = 0, matched = 0;
    while (generated.size() < 20) {
      const auto round = loop.prepare(20 - generated.size(), 6, propose);
      // In Core, return round.tokens from advance and save round.id; only call
      // commit from Session::commit with Core's accepted prefix length.
      const auto accepted = loop.commit(round.id, round.tokens.size());
      generated.insert(generated.end(), accepted.tokens.begin(), accepted.tokens.end());
      matched += accepted.accepted_drafts; ++rounds;
    }
    for (std::size_t i = 0; i < generated.size(); ++i)
      if (generated[i] != static_cast<SpecToken>((i + 3) % 5)) throw std::runtime_error("Greedy continuation differs");
    std::cout << "20 ordinary-greedy-equivalent tokens in " << rounds << " rounds; " << matched << " accepted drafts\n";
    std::cout << "Target state includes every emitted correction/bonus before publication\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
