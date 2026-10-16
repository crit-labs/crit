// Constrained sampled speculation on a synthetic logit model: Core's matchers (crit/constraints.h) inside
// mtp-decode's SampledSpeculator. Forbidden "x" has the largest raw logit but cannot escape the language.
#include "constrained_state.h"
#include <iostream>
using namespace crit::techniques;
using crit::core::ByteGrammar;
using crit::core::ByteVocabulary;
using crit::core::ChoiceGrammar;
using crit::core::ChoiceMatcher;
using crit::core::GrammarMatcher;
using crit::core::JsonObjectMatcher;
class ToyLogits final : public LogitState {
  std::vector<SpecToken> tokens_{5};
  bool draft_, json_;
public:
  explicit ToyLogits(bool draft, bool json = false) : draft_(draft), json_(json) {}
  std::unique_ptr<LogitState> fork() const override { return std::make_unique<ToyLogits>(*this); }
  std::span<const SpecToken> tokens() const noexcept override { return tokens_; }
  std::vector<float> logits() const override {
    auto scores = draft_ ? std::vector<float>{2, 3, 0, 1, 0, 99} : std::vector<float>{1, 0, 2, 3, 2, 99};
    if(json_ && tokens_.size() > 1 && tokens_.back() == 0) { scores[0] = -1000; scores[4] = -1000; }
    return scores;
  }
  void append(SpecToken token) override { tokens_.push_back(token); }
  void rewind(std::size_t n) override {
    if(n > tokens_.size()) throw std::out_of_range("Toy rewind");
    tokens_.resize(n);
  }
};
template<class MakeMatcher>
std::string generate(const std::vector<std::string>& pieces, MakeMatcher make, bool json, std::uint64_t seed) {
  SampledSpeculator loop(std::make_unique<ConstrainedProbabilityState>(std::make_unique<ToyLogits>(false, json), make()),
                         std::make_unique<ConstrainedProbabilityState>(std::make_unique<ToyLogits>(true, json), make()),
                         SpeculationLimits{16, 4, 6}, seed, std::vector<SpecToken>{3});
  std::string answer;
  while(!loop.finished()) {
    auto round = loop.prepare(8, 3);
    auto result = loop.commit(round.id, round.tokens.size());
    for(auto id : result.tokens) answer += pieces[static_cast<std::size_t>(id)];
  }
  return answer;
}
int main() {
  const std::vector<std::string> pieces{"a", "b", "c", "", "ab", "x"};
  auto vocabulary = std::make_shared<const ByteVocabulary>(pieces, 3);
  const std::vector<std::string> choices{"ab", "ac"};
  auto grammar = std::make_shared<const ChoiceGrammar>(choices);
  auto answer = generate(pieces, [&] { return std::make_unique<ChoiceMatcher>(grammar, vocabulary); }, false, 17);
  if(answer != "ab" && answer != "ac") throw std::runtime_error("Constraint violation");
  std::cout << "Verified constrained answer: " << answer << '\n';
  const std::vector<std::string> json_pieces{"{\"reply\":", "true}", "false}", "", "{}", "invalid"};
  auto json_vocabulary = std::make_shared<const ByteVocabulary>(json_pieces, 3);
  auto object = generate(json_pieces, [&] { return std::make_unique<JsonObjectMatcher>(json_vocabulary); }, true, 19);
  if(object != "{}" && object != "{\"reply\":true}" && object != "{\"reply\":false}")
    throw std::runtime_error("JSON constraint violation");
  std::cout << "Verified JSON object: " << object << '\n';
  // crit_core.constraints.compile_language emits this format from a schema, byte regex or EBNF.
  auto compiled = ByteGrammar::load("crit-byte-grammar-v1 1 2 0\n0 2 97 98\n0 2 97 99\n");
  ConstrainedProbabilityState constrained(std::make_unique<ToyLogits>(false),
                                          std::make_unique<GrammarMatcher>(compiled, vocabulary));
  constrained.append(4); // One token spans both grammar terminals: ab.
  constrained.append(3);
  if(constrained.next()[3] != 1) throw std::runtime_error("Grammar EOS violation");
  std::cout << "Verified compiled grammar and terminal law\n";
}
