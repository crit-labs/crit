// Constrained speculative decoding: ConstrainedProbabilityState over Core's matchers inside SampledSpeculator.
// Matcher languages themselves are Core's (tests/core/constraints_tests.cpp).
#include "constrained_state.h"
#include <array>
#include <iostream>
using namespace crit::techniques;
using crit::core::ByteGrammar;
using crit::core::ByteVocabulary;
using crit::core::ChoiceGrammar;
using crit::core::ChoiceMatcher;
using crit::core::GrammarMatcher;
using crit::core::JsonObjectMatcher;
void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
template<class F>void rejects(F f) {try{f();}catch(const std::exception&){return;}throw std::runtime_error("Expected rejection");}
class RawModel final:public LogitState {
  std::vector<SpecToken> history_{5};
  bool draft_;
  std::shared_ptr<bool> fail_;
public:
  RawModel(bool draft,std::shared_ptr<bool> fail):draft_(draft),fail_(std::move(fail)) {}
  std::unique_ptr<LogitState> fork() const override {return std::make_unique<RawModel>(*this);}
  std::span<const SpecToken> tokens() const noexcept override {return history_;}
  std::vector<float> logits() const override {return draft_?std::vector<float>{2,3,0,1,0,99}:std::vector<float>{1,0,2,3,2,99};}
  void append(SpecToken token) override {history_.push_back(token);if(*fail_)throw std::runtime_error("Append failed");}
  void rewind(std::size_t n) override {if(n>history_.size())throw std::out_of_range("Rewind");history_.resize(n);}
};
const std::vector<std::string> toy_pieces{"a","b","c","","ab","x"};
auto toy_vocabulary() {return std::make_shared<const ByteVocabulary>(toy_pieces,3);}
std::unique_ptr<TokenMatcher> grammar() {
  static const std::vector<std::string> choices{"ab","ac"};
  return std::make_unique<ChoiceMatcher>(std::make_shared<const ChoiceGrammar>(choices),toy_vocabulary());
}
class FaultModel final:public LogitState {
  std::vector<SpecToken> history_{5};
  int fault_;
public:
  explicit FaultModel(int fault):fault_(fault) {}
  std::unique_ptr<LogitState> fork() const override {
    if(fault_==3)return {};
    if(fault_==4)return std::unique_ptr<LogitState>(const_cast<FaultModel*>(this));
    auto copy=std::make_unique<FaultModel>(*this);
    if(fault_==5)copy->history_[0]=0;
    return copy;
  }
  std::span<const SpecToken> tokens() const noexcept override {return history_;}
  std::vector<float> logits() const override {return {1,0,2,3,2,99};}
  void append(SpecToken token) override {history_.push_back(fault_==1?5:token);}
  void rewind(std::size_t n) override {
    history_.resize(n);
    if(fault_==2)throw std::runtime_error("Rewind failed after mutation");
  }
};
class BadMatcher final:public TokenMatcher {
  std::shared_ptr<TokenMatcher> state_{grammar()};
  bool bad_append_;
public:
  explicit BadMatcher(bool bad_append):bad_append_(bad_append) {}
  BadMatcher(const BadMatcher& other):state_(other.state_->fork()),bad_append_(other.bad_append_) {}
  std::unique_ptr<TokenMatcher> fork() const override {return std::make_unique<BadMatcher>(*this);}
  std::size_t size() const noexcept override {return state_->size();}
  std::size_t vocabulary() const noexcept override {return state_->vocabulary();}
  bool ended() const noexcept override {return state_->ended();}
  std::vector<SpecToken> allowed() const override {return state_->allowed();}
  void append(SpecToken token) override {if(!bad_append_)state_->append(token);}
  void rewind(std::size_t) override {} // malformed backend: fails to restore
};
void arithmetic() {
  const std::vector<float> scores{100,0,2,2,-100};const std::vector<SpecToken> ids{1,2,3};
  auto law=constrained_logits(scores,ids,0,1,.1,.9);
  check(law[2]==1 && law[0]==0 && law[3]==0,"Mask/greedy tie order");
  for(double temperature: {0.,.5,1.,2.})for(std::size_t k:{0U,1U,2U,5U})for(double p:{.2,1.})for(double m:{0.,.3,1.}) {
    auto actual=constrained_logits(scores,ids,temperature,k,p,m);
    auto expected=Distribution::logits(std::vector<float>{0,2,2},temperature,std::min(k,ids.size()),p,m);
    for(std::size_t i=0;i<ids.size();++i)check(std::abs(actual[ids[i]]-expected.values()[i])<1e-14,"Constrained filters differ");
    check(actual[0]==0 && actual[4]==0,"Forbidden probability escaped");
  }
  rejects([&]{constrained_logits(scores,std::vector<SpecToken>{2,1});});
  rejects([&]{constrained_logits(scores,std::vector<SpecToken>{2,2});});
  rejects([&]{constrained_logits(scores,std::vector<SpecToken>{5});});
  rejects([&]{constrained_logits(scores,std::vector<SpecToken>{});});
  rejects([&]{constrained_logits(scores,ids,1,6);});
  auto bad=scores;bad[0]=std::numeric_limits<float>::infinity();
  rejects([&]{constrained_logits(bad,ids);});
}
void transactions() {
  auto fail=std::make_shared<bool>(false);
  ConstrainedProbabilityState state(std::make_unique<RawModel>(false,fail),grammar());
  auto branch=state.fork();branch->append(0);branch->append(1);branch->rewind(2);
  check(branch->next()[2]>0 && state.tokens().size()==1,"Private branch/rewind");
  auto before=state.next();rejects([&]{state.append(5);});check(state.next()==before,"Invalid append mutated parent");
  rejects([&]{state.rewind(0);});
  const std::vector<SpecToken> stops{3};
  SampledSpeculator loop(std::make_unique<ConstrainedProbabilityState>(std::make_unique<RawModel>(false,fail),grammar()),
    std::make_unique<ConstrainedProbabilityState>(std::make_unique<RawModel>(true,fail),grammar()),SpeculationLimits{16,4,6},17,stops);
  auto discarded=loop.prepare(8,3);std::vector<SpecToken> output(discarded.tokens.begin(),discarded.tokens.end());loop.discard(discarded.id);
  auto retry=loop.prepare(8,3);check(std::equal(output.begin(),output.end(),retry.tokens.begin(),retry.tokens.end()),"Discard consumed RNG");
  loop.commit(retry.id,1);check(loop.history().size()==2,"Partial publication");
  auto parent=std::vector<SpecToken>(loop.history().begin(),loop.history().end());
  auto round=loop.prepare(8,0);*fail=true;rejects([&]{loop.commit(round.id,round.tokens.size());});*fail=false;
  check(std::equal(parent.begin(),parent.end(),loop.history().begin(),loop.history().end()),"Failed repair published state");
  rejects([&]{loop.commit(round.id,1);});
  ConstrainedProbabilityState broken(std::make_unique<RawModel>(false,fail),grammar());
  *fail=true;rejects([&]{broken.append(0);});*fail=false;
  rejects([&]{broken.next();});rejects([&]{broken.fork();});
  rejects([&]{broken.append(0);});rejects([&]{broken.rewind(1);});
  ConstrainedProbabilityState wrong_token(std::make_unique<FaultModel>(1),grammar());
  rejects([&]{wrong_token.append(0);});rejects([&]{wrong_token.next();});
  ConstrainedProbabilityState broken_rewind(std::make_unique<FaultModel>(2),grammar());
  broken_rewind.append(0);rejects([&]{broken_rewind.rewind(1);});
  rejects([&]{broken_rewind.next();});rejects([&]{broken_rewind.fork();});
  for(int fault:{3,4,5}) {
    ConstrainedProbabilityState malformed(std::make_unique<FaultModel>(fault),grammar());
    auto law=malformed.next();rejects([&]{malformed.fork();});check(malformed.next()==law,"Bad fork damaged parent");
  }
  for(bool bad_append:{true,false}) {
    ConstrainedProbabilityState bad_matcher(std::make_unique<RawModel>(false,fail),std::make_unique<BadMatcher>(bad_append));
    if(!bad_append)bad_matcher.append(0);
    auto prefix=std::vector<SpecToken>(bad_matcher.tokens().begin(),bad_matcher.tokens().end());
    auto law=bad_matcher.next();
    if(bad_append)rejects([&]{bad_matcher.append(0);});else rejects([&]{bad_matcher.rewind(1);});
    check(std::equal(prefix.begin(),prefix.end(),bad_matcher.tokens().begin(),bad_matcher.tokens().end()) &&
      bad_matcher.next()==law,"Malformed matcher mutated model state");
  }
}
auto byte_vocabulary() {
  std::vector<std::string> pieces;
  for(unsigned int byte=0;byte<256;++byte)pieces.emplace_back(1,static_cast<char>(byte));
  pieces.emplace_back();return std::make_shared<const ByteVocabulary>(std::move(pieces),256);
}
bool json_accepts(const std::string& bytes) {
  try {
    JsonObjectMatcher matcher(byte_vocabulary());
    for(char raw:bytes)matcher.append(static_cast<unsigned char>(raw));
    matcher.append(256);return matcher.ended();
  } catch(const std::exception&) {return false;}
}
class JsonModel final:public LogitState {
  std::vector<SpecToken> history_{5};
  bool draft_;
public:
  explicit JsonModel(bool draft):draft_(draft) {}
  std::unique_ptr<LogitState> fork() const override {return std::make_unique<JsonModel>(*this);}
  std::span<const SpecToken> tokens() const noexcept override {return history_;}
  std::vector<float> logits() const override {
    auto scores=draft_?std::vector<float>{2,0,3,4,2,99}:std::vector<float>{1,3,0,2,4,99};
    // A toy vocabulary without a standalone closing brace cannot complete a
    // nested object. This synthetic model has zero law on those paths, leaving
    // three reachable answers whose causal probabilities are known exactly.
    if(history_.size()>1 && history_.back()==0) {scores[0]=-1000;scores[4]=-1000;}
    return scores;
  }
  void append(SpecToken token) override {history_.push_back(token);}
  void rewind(std::size_t n) override {
    if(n>history_.size())throw std::out_of_range("JSON model rewind");
    history_.resize(n);
  }
};
void json_sampled() {
  const std::vector<std::string> pieces{"{\"x\":", "true}","false}","", "{}", "invalid"};
  auto vocabulary=std::make_shared<const ByteVocabulary>(pieces,3);
  // Three complete answers with two token segmentations and no unbounded cycles.
  const double first=1/(1+std::exp(3.)),yes=1/(1+std::exp(-3.));
  const std::array<double,3> expected{1-first,first*yes,first*(1-yes)};
  for(std::size_t budget:{0U,3U}) {
    std::array<std::size_t,3> counts{};
    for(std::uint64_t seed=0;seed<3000;++seed) {
      SampledSpeculator loop(
        std::make_unique<ConstrainedProbabilityState>(std::make_unique<JsonModel>(false),std::make_unique<JsonObjectMatcher>(vocabulary)),
        std::make_unique<ConstrainedProbabilityState>(std::make_unique<JsonModel>(true),std::make_unique<JsonObjectMatcher>(vocabulary)),
        SpeculationLimits{16,4,6},seed,std::vector<SpecToken>{3});
      auto discarded=loop.prepare(8,budget);loop.discard(discarded.id);
      std::string answer;
      while(!loop.finished()) {
        auto round=loop.prepare(8,budget);auto result=loop.commit(round.id,1);
        for(auto token:result.tokens)answer+=pieces[static_cast<std::size_t>(token)];
      }
      check(json_accepts(answer),"Speculative JSON was malformed");
      if(answer=="{}")++counts[0];else if(answer=="{\"x\":true}")++counts[1];
      else if(answer=="{\"x\":false}")++counts[2];else throw std::runtime_error("Unexpected JSON answer");
    }
    for(std::size_t i=0;i<counts.size();++i)
      check(std::abs(static_cast<double>(counts[i])/3000-expected[i])<.025,"JSON constraint changed target law");
  }
}
void sampled() {
  // Independent language probabilities: the target may emit 'ab' as one piece
  // or 'a' then 'b'. Both segmentations contribute to the same answer.
  const double a=1/(1+std::exp(1.)),b=1/(1+std::exp(2.));
  const double expected_ab=(1-a)+a*b;
  std::size_t ab=0,ordinary_ab=0;
  for(std::uint64_t seed=0;seed<4000;++seed)for(std::size_t budget:{0U,3U}) {
    auto fail=std::make_shared<bool>(false);const std::vector<SpecToken> stops{3};
    SampledSpeculator loop(std::make_unique<ConstrainedProbabilityState>(std::make_unique<RawModel>(false,fail),grammar()),
      std::make_unique<ConstrainedProbabilityState>(std::make_unique<RawModel>(true,fail),grammar()),SpeculationLimits{16,4,6},seed,stops);
    std::string answer;const std::array<std::string,6> pieces{"a","b","c","","ab","x"};
    while(!loop.finished()) {
      auto round=loop.prepare(8,budget);auto result=loop.commit(round.id,round.tokens.size());
      for(auto token:result.tokens)answer+=pieces[static_cast<std::size_t>(token)];
    }
    check(answer=="ab" || answer=="ac","Sampled output violated language");
    if(answer=="ab") {if(budget)++ab;else ++ordinary_ab;}
  }
  check(std::abs(static_cast<double>(ab)/4000-expected_ab)<.025,"Speculation changed constrained target law");
  check(std::abs(static_cast<double>(ordinary_ab)/4000-expected_ab)<.025,"Ordinary constrained law differs");
}
auto byte_grammar() {
  return std::make_shared<const ByteGrammar>(2, 0,
    std::vector<ByteGrammar::Production>{{0, {97, 257}}, {1, {98}}, {1, {99}}});
}
void grammar_tests() {
  auto vocabulary = toy_vocabulary();
  // The same constrained law and seed must produce the same ordinary/speculative
  // sequence for the compiled matcher and independent finite-choice matcher.
  for (std::uint64_t seed = 0; seed < 200; ++seed) for (std::size_t budget : {0U, 3U}) {
    auto fail = std::make_shared<bool>(false);
    auto state = [&](bool draft) {
      return std::make_unique<ConstrainedProbabilityState>(std::make_unique<RawModel>(draft, fail),
             std::make_unique<GrammarMatcher>(byte_grammar(), vocabulary));
    };
    const std::vector<SpecToken> stops{3};
    SampledSpeculator actual(state(false), state(true), SpeculationLimits{16,4,6}, seed, stops);
    SampledSpeculator expected(std::make_unique<ConstrainedProbabilityState>(std::make_unique<RawModel>(false,fail),grammar()),
      std::make_unique<ConstrainedProbabilityState>(std::make_unique<RawModel>(true,fail),grammar()),SpeculationLimits{16,4,6},seed,stops);
    while (!actual.finished()) {
      auto a = actual.prepare(8,budget), e = expected.prepare(8,budget);
      check(std::equal(a.tokens.begin(), a.tokens.end(), e.tokens.begin(), e.tokens.end()),
            "Compiled grammar changed constrained sampled law");
      actual.commit(a.id,a.tokens.size()); expected.commit(e.id,e.tokens.size());
    }
    check(expected.finished(), "Grammar EOS differed from choice oracle");
  }
}
int main(int argc,char** argv) {
  try {
    const std::string mode=argc>1?argv[1]:"arithmetic";
    if(mode=="arithmetic")arithmetic();else if(mode=="transactions")transactions();else if(mode=="sampled")sampled();
    else if(mode=="json_sampled")json_sampled();
    else if(mode=="grammar")grammar_tests();
    else throw std::invalid_argument("Unknown mode");
    std::cout<<"constraint "<<mode<<" passed\n";return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
