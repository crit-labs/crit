#include "sampled.h"
#include <array>
#include <chrono>
#include <iostream>
using namespace crit::techniques;
class Toy final:public ProbabilityState {
  std::vector<SpecToken> prefix_{0};
  bool draft_;
public:
  explicit Toy(bool draft):draft_(draft){}
  std::unique_ptr<ProbabilityState> fork() const override{return std::make_unique<Toy>(*this);}
  std::span<const SpecToken> tokens() const noexcept override{return prefix_;}
  Distribution next() const override {
    const std::array<float,3> weights=draft_?std::array<float,3>{.7F,.2F,.1F}:std::array<float,3>{.2F,.5F,.3F};
    std::array<float,3> scores{};for (std::size_t i=0;i<scores.size();++i) scores[i]=std::log(weights[i]);
    return Distribution::logits(scores,1,0,1,.25);
  }
  void append(SpecToken t) override{prefix_.push_back(t);}
  void rewind(std::size_t n) override{if(!n || n>prefix_.size())throw std::out_of_range("Invalid rewind");prefix_.resize(n);}
};
class SparseToy final:public SparseProbabilityState {
  Toy inner_;
public:
  explicit SparseToy(bool draft):inner_(draft) {}
  std::unique_ptr<SparseProbabilityState> fork() const override {return std::make_unique<SparseToy>(*this);}
  std::span<const SpecToken> tokens() const noexcept override {return inner_.tokens();}
  SparseDistribution next() const override {return SparseDistribution::from_dense(inner_.next(),2);}
  void append(SpecToken token) override {inner_.append(token);}
  void rewind(std::size_t count) override {inner_.rewind(count);}
};
int main(){
  // Draft min-p leaves two positive tokens; target needs three and therefore
  // keeps its complete law in the exact dense fallback at capacity two.
  if (SparseToy(true).next().is_dense() || !SparseToy(false).next().is_dense())
    throw std::runtime_error("Expected sparse draft and complete target fallback");
  if (SparseDistribution(3,{{1,1},{1,0}},0)[1]!=1)
    throw std::runtime_error("Repeated padded ID lost positive mass");
  SampledSpeculator dense(std::make_unique<Toy>(false),std::make_unique<Toy>(true),{13,4,3},42);
  SparseSampledSpeculator loop(std::make_unique<SparseToy>(false),std::make_unique<SparseToy>(true),{13,4,3},42);
  AdaptiveDraftBudget policy(std::array<std::size_t,3>{0,2,4});
  while(!loop.finished()){
    const auto remaining=13-loop.history().size(),width=policy.choose(remaining);
    const auto measurement=policy.begin_observation(width);
    const auto start=std::chrono::steady_clock::now();
    auto round=loop.prepare(remaining,width);
    auto committed=loop.commit(round.id,round.tokens.size());
    policy.observe(measurement,committed.tokens.size(),std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    auto ordinary=dense.prepare(remaining,width);auto control=dense.commit(ordinary.id,ordinary.tokens.size());
    if (control.tokens!=committed.tokens) throw std::runtime_error("Sparse and dense implementations diverged");
    for(auto token:committed.tokens)std::cout<<token<<' ';
  }
  if (policy.pending() || !std::equal(dense.history().begin(),dense.history().end(),loop.history().begin(),loop.history().end()))
    throw std::runtime_error("Unsettled observation or divergent continuation state");
  std::cout<<"\nSampled sparse/dense example completed with min-p and tagged supported-width observations\n";
}
