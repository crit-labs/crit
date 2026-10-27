#pragma once
#include "speculative.h"
#include <cmath>
#include <numeric>
#include <random>

namespace crit::techniques {
// Sampling and verification must agree on each model's filtered distribution. This is a
// synchronous CPU controller; allocations and device completion belong to the engine.
class Distribution {
  std::vector<double> probabilities_;
  // Only the checked sparse representation may export already-normalized bytes.
  struct Normalized {};
  Distribution(std::vector<double> probabilities,Normalized) : probabilities_(std::move(probabilities)) {}
  friend class SparseDistribution;
public:
  explicit Distribution(std::vector<double> weights) : probabilities_(std::move(weights)) {
    if (probabilities_.empty() || probabilities_.size()>1048576) throw std::invalid_argument("Invalid probability extent");
    long double sum=0;
    for (auto p:probabilities_) {
      if (!std::isfinite(p) || p<0) throw std::invalid_argument("Nonfinite/negative probability");
      sum+=p;
    }
    if (!(sum>0) || !std::isfinite(sum)) throw std::invalid_argument("Empty/nonfinite probability mass");
    for (auto& p:probabilities_) p=static_cast<double>(static_cast<long double>(p)/sum);
  }
  std::span<const double> values() const noexcept { return probabilities_; }
  std::size_t size() const noexcept { return probabilities_.size(); }
  double operator[](SpecToken token) const { return probabilities_.at(static_cast<std::size_t>(token)); }
  bool operator==(const Distribution&) const = default;
  SpecToken sample(double uniform) const {
    if (!(uniform>=0 && uniform<1)) throw std::invalid_argument("Sampling draw must be in [0,1)");
    long double sum=0;std::size_t last=0;
    for (std::size_t i=0;i<size();++i) {
      if (probabilities_[i]>0) last=i;
      sum+=probabilities_[i];
      if (static_cast<long double>(uniform)<sum) return static_cast<SpecToken>(i);
    }
    return static_cast<SpecToken>(last); // Normalization roundoff at the final CDF bin.
  }
  static Distribution logits(std::span<const float> scores,double temperature=1,
                             std::size_t top_k=0,double top_p=1,double min_p=0) {
    if (scores.empty() || scores.size()>1048576 || !std::isfinite(temperature) || temperature<0 ||
        !std::isfinite(top_p) || top_p<=0 || top_p>1 || top_k>scores.size() ||
        !std::isfinite(min_p) || min_p<0 || min_p>1 ||
        !std::all_of(scores.begin(),scores.end(),[](auto v){return std::isfinite(v);}))
      throw std::invalid_argument("Invalid sampling parameters/logits");
    const auto best=std::max_element(scores.begin(),scores.end());
    std::vector<double> weights(scores.size());
    if (!temperature) {weights[static_cast<std::size_t>(best-scores.begin())]=1;return Distribution(std::move(weights));}
    std::vector<std::size_t> order(scores.size());std::iota(order.begin(),order.end(),0);
    if (top_k || top_p<1) std::sort(order.begin(),order.end(),[&](auto a,auto b){return scores[a]==scores[b]?a<b:scores[a]>scores[b];});
    const auto count=top_k?top_k:scores.size();long double sum=0;
    for (std::size_t i=0;i<count;++i) {auto at=order[i];weights[at]=std::exp((static_cast<double>(scores[at])-*best)/temperature);sum+=weights[at];}
    if (top_p<1) {
      long double accumulated=0;bool retained=true;
      for (std::size_t i=0;i<count;++i) {
        auto at=order[i];
        if (retained) {accumulated+=weights[at];if (accumulated>=top_p*sum)retained=false;}
        else weights[at]=0;
      }
    }
    // Relative to the largest surviving probability; normalization cancels.
    // Fixed-count top-k ties and top-p order above remain identical to ordinary
    // sampling. Temperature zero bypasses these filters after validating them.
    if (min_p>0) for (auto& weight:weights) if (weight<min_p) weight=0;
    return Distribution(std::move(weights));
  }
};

struct ProbabilityMass {
  SpecToken token;
  double weight;
  bool operator==(const ProbabilityMass&) const=default;
};

// Owns sorted unique positive (token, probability) pairs, or a complete dense
// fallback. Capacity limits representation only: it never truncates target mass.
class SparseDistribution {
  static constexpr std::size_t maximum_=1048576;
  std::size_t vocabulary_,capacity_;
  std::vector<ProbabilityMass> support_;
  std::optional<Distribution> dense_;
  struct Normalized {};
  SparseDistribution(std::size_t vocabulary,std::size_t capacity,Normalized)
      :vocabulary_(vocabulary),capacity_(capacity) {
    if (!vocabulary || vocabulary>maximum_ || capacity>maximum_)
      throw std::invalid_argument("Invalid sparse vocabulary/capacity");
  }
  void dense_if_needed() {
    if (support_.size()<=capacity_) return;
    std::vector<double> values(vocabulary_);
    for (const auto& entry:support_) values[static_cast<std::size_t>(entry.token)]=entry.weight;
    dense_.emplace(Distribution(std::move(values),Distribution::Normalized{}));
    std::vector<ProbabilityMass>().swap(support_);
  }
public:
  explicit SparseDistribution(std::size_t vocabulary,std::vector<ProbabilityMass> weights,
                              std::size_t sparse_capacity=256)
      :SparseDistribution(vocabulary,sparse_capacity,Normalized{}) {
    if (weights.empty() || weights.size()>maximum_) throw std::invalid_argument("Invalid sparse support extent");
    for (const auto& entry:weights) {
      if (entry.token<0 || static_cast<std::size_t>(entry.token)>=vocabulary_ ||
          !std::isfinite(entry.weight) || entry.weight<0)
        throw std::invalid_argument("Invalid sparse token/weight");
    }
    // A canonical accumulation order also makes duplicate input permutations
    // agree; zero padding for a repeated ID cannot overwrite its positive mass.
    std::sort(weights.begin(),weights.end(),[](const auto& a,const auto& b){
      return a.token==b.token?a.weight<b.weight:a.token<b.token;
    });
    long double total=0;for (const auto& entry:weights) total+=entry.weight;
    if (!(total>0) || !std::isfinite(total)) throw std::invalid_argument("Empty/nonfinite sparse probability mass");
    std::size_t write=0;
    for (std::size_t read=0;read<weights.size();) {
      const auto id=weights[read].token;long double mass=0;
      do {mass+=weights[read++].weight;} while (read<weights.size() && weights[read].token==id);
      const auto probability=static_cast<double>(mass/total);
      if (probability>0) weights[write++]={id,probability};
    }
    // Do not retain a large duplicate/padding allocation behind a small support.
    support_.assign(weights.begin(),weights.begin()+static_cast<std::ptrdiff_t>(write));dense_if_needed();
  }
  static SparseDistribution from_dense(Distribution probabilities,std::size_t sparse_capacity=256) {
    SparseDistribution result(probabilities.size(),sparse_capacity,Normalized{});
    // Scan once, without renormalizing. Fallback preserves the original bytes.
    for (std::size_t id=0;id<probabilities.size();++id) if (probabilities.values()[id]>0) {
      if (result.support_.size()==sparse_capacity) {
        result.dense_=std::move(probabilities);
        std::vector<ProbabilityMass>().swap(result.support_);return result;
      }
      result.support_.push_back({static_cast<SpecToken>(id),probabilities.values()[id]});
    }
    return result;
  }
  static SparseDistribution logits(std::span<const float> scores,double temperature=1,
      std::size_t top_k=0,double top_p=1,double min_p=0,std::size_t sparse_capacity=256) {
    return from_dense(Distribution::logits(scores,temperature,top_k,top_p,min_p),sparse_capacity);
  }
  std::size_t size() const noexcept {return vocabulary_;}
  std::size_t sparse_capacity() const noexcept {return capacity_;}
  bool is_dense() const noexcept {return dense_.has_value();}
  // Borrowed until this owner is mutated/destroyed. Only one view is populated.
  std::span<const ProbabilityMass> support() const noexcept {return support_;}
  std::span<const double> dense_values() const noexcept {return dense_?dense_->values():std::span<const double>{};}
  double operator[](SpecToken id) const {
    if (id<0 || static_cast<std::size_t>(id)>=vocabulary_) throw std::out_of_range("Sparse token outside vocabulary");
    if (dense_) return (*dense_)[id];
    const auto entry=std::lower_bound(support_.begin(),support_.end(),id,
                                    [](const auto& value,SpecToken token){return value.token<token;});
    return entry!=support_.end() && entry->token==id?entry->weight:0;
  }
  Distribution to_dense() const {
    if (dense_) return *dense_;
    std::vector<double> values(vocabulary_);
    for (const auto& entry:support_) values[static_cast<std::size_t>(entry.token)]=entry.weight;
    return Distribution(std::move(values),Distribution::Normalized{});
  }
  bool operator==(const SparseDistribution& other) const {
    if (size()!=other.size()) return false;
    if (!dense_ && !other.dense_) return support_==other.support_;
    if (dense_ && other.dense_) return *dense_==*other.dense_;
    const auto& packed=dense_?other:*this;
    const auto& values=dense_?dense_->values():other.dense_->values();
    std::size_t next=0;
    for (std::size_t id=0;id<size();++id) {
      double expected=0;
      if (next<packed.support_.size() && static_cast<std::size_t>(packed.support_[next].token)==id)
        expected=packed.support_[next++].weight;
      if (values[id]!=expected) return false;
    }
    return true;
  }
  SpecToken sample(double uniform) const {
    if (dense_) return dense_->sample(uniform);
    if (!(uniform>=0 && uniform<1)) throw std::invalid_argument("Sampling draw must be in [0,1)");
    long double mass=0;
    for (const auto& entry:support_) {
      mass+=entry.weight;if (static_cast<long double>(uniform)<mass) return entry.token;
    }
    return support_.back().token;
  }
};

inline double acceptance_probability(const Distribution& target,const Distribution& draft,SpecToken token) {
  if (target.size()!=draft.size() || !(draft[token]>0)) throw std::invalid_argument("Invalid proposal distribution/token");
  return std::min(1.,target[token]/draft[token]);
}
// The residual law max(weight * p - q, 0), normalized: weight 1 is the token rule's; block verification passes the
// prefix weight of the rejected position. residual_mass is its unnormalized mass Z.
inline Distribution rejection_distribution(const Distribution& target,const Distribution& draft,double weight=1) {
  if (target.size()!=draft.size()) throw std::invalid_argument("Probability vocabularies differ");
  std::vector<double> weights(target.size());
  for (std::size_t i=0;i<weights.size();++i) weights[i]=std::max(0.,weight*target.values()[i]-draft.values()[i]);
  return Distribution(std::move(weights));
}
inline double residual_mass(const Distribution& target,const Distribution& draft,double weight) {
  if (target.size()!=draft.size()) throw std::invalid_argument("Probability vocabularies differ");
  long double mass=0;
  for (std::size_t i=0;i<target.size();++i) mass+=std::max(0.,weight*target.values()[i]-draft.values()[i]);
  return static_cast<double>(mass);
}

inline double acceptance_probability(const SparseDistribution& target,const SparseDistribution& draft,SpecToken token) {
  if (target.size()!=draft.size() || !(draft[token]>0)) throw std::invalid_argument("Invalid sparse proposal distribution/token");
  return std::min(1.,target[token]/draft[token]);
}
inline SparseDistribution rejection_distribution(const SparseDistribution& target,const SparseDistribution& draft,
                                                 double weight=1) {
  if (target.size()!=draft.size()) throw std::invalid_argument("Sparse probability vocabularies differ");
  std::vector<ProbabilityMass> residual;
  const auto append=[&](SpecToken id,double probability) {
    const auto mass=std::max(0.,weight*probability-draft[id]);if (mass>0) residual.push_back({id,mass});
  };
  if (target.is_dense()) {
    const auto values=target.dense_values();
    for (std::size_t id=0;id<values.size();++id) if (values[id]>0) append(static_cast<SpecToken>(id),values[id]);
  } else for (const auto& entry:target.support()) append(entry.token,entry.weight);
  return SparseDistribution(target.size(),std::move(residual),std::min(target.sparse_capacity(),draft.sparse_capacity()));
}
inline double residual_mass(const SparseDistribution& target,const SparseDistribution& draft,double weight) {
  if (target.size()!=draft.size()) throw std::invalid_argument("Sparse probability vocabularies differ");
  long double mass=0;
  if (target.is_dense()) {
    const auto values=target.dense_values();
    for (std::size_t id=0;id<values.size();++id)
      if (values[id]>0) mass+=std::max(0.,weight*values[id]-draft[static_cast<SpecToken>(id)]);
  } else for (const auto& entry:target.support()) mass+=std::max(0.,weight*entry.weight-draft[entry.token]);
  return static_cast<double>(mass);
}

// How a round decides which drafts stand. token: each draft in turn with min(1, p/q) (Leviathan et al.). block:
// block verification (Sun et al., arXiv 2403.10444), which accepts the longest prefix by the joint ratios of its
// drafts and accepts at least as many drafts in expectation, with the same output law and one draw per position.
enum class VerifyRule { token, block };

// Block verification of one round: p holds the target's D + 1 laws (after drafts[0, i)), q the draft's laws the D
// drafts were drawn from, `draw` returns uniforms in [0, 1). Returns tau, the number of standing drafts, and the
// token after them: the bonus from p[D] when tau == D, else a draw from max(w_tau * p[tau] - q[tau], 0). With the
// prefix weights w_0 = 1 and w_i = min(1, w_(i-1) * p[i-1](d) / q[i-1](d)), draft count i stands with probability
// h_D = w_D and h_i = Z_i / (Z_i + 1 - w_i) below it (Z_i the residual mass at row i, h_i = 1 when w_i = 1), and
// tau is the largest i whose Bernoulli(h_i) succeeds, scanned from D down.
template<class Law,class Draw>
std::pair<std::size_t,SpecToken> block_verify(std::span<const Law> p,std::span<const Law> q,
                                              std::span<const SpecToken> drafts,Draw&& draw) {
  const std::size_t D=drafts.size();
  if (p.size()!=D+1 || q.size()<D) throw std::invalid_argument("Block verification needs D + 1 target and D draft laws");
  std::vector<double> w(D+1,1.);
  for (std::size_t i=1;i<=D;++i) {
    const auto d=drafts[i-1];const double qd=q[i-1][d];
    if (!(qd>0)) throw std::invalid_argument("Invalid proposal distribution/token");
    w[i]=std::min(1.,w[i-1]*p[i-1][d]/qd);
  }
  std::size_t tau=0;
  for (std::size_t i=D;;--i) {
    double h=w[i];
    if (i<D && w[i]<1) {const double z=residual_mass(p[i],q[i],w[i]);h=z>0?z/(z+1-w[i]):0;}
    if (h>=1 || (h>0 && draw()<h)) {tau=i;break;}
    if (i==0) break;   // unreachable: h_0 = 1
  }
  const SpecToken next=tau==D?p[D].sample(draw()):rejection_distribution(p[tau],q[tau],w[tau]).sample(draw());
  return {tau,next};
}

// Owns a completely materialized prefix and its next distribution. fork isolates
// all mutable state; rewind restores KV, position and filtered next probabilities.
// All calls settle physical work before returning or throwing, including failures.
class ProbabilityState {
public:
  virtual ~ProbabilityState()=default;
  virtual std::unique_ptr<ProbabilityState> fork() const=0;
  virtual std::span<const SpecToken> tokens() const noexcept=0;
  virtual Distribution next() const=0;
  virtual void append(SpecToken)=0;
  virtual void rewind(std::size_t)=0;
  virtual std::vector<Distribution> verify(std::span<const SpecToken> drafts) {
    std::vector<Distribution> result;result.reserve(drafts.size()+1);result.push_back(next());
    for (auto t:drafts) {append(t);result.push_back(next());}return result;
  }
};

// Same settled-prefix/fork/rewind contract, with owned sparse or dense-fallback
// probabilities. Engines can keep all D+1 verification rows in this form.
class SparseProbabilityState {
public:
  virtual ~SparseProbabilityState()=default;
  virtual std::unique_ptr<SparseProbabilityState> fork() const=0;
  virtual std::span<const SpecToken> tokens() const noexcept=0;
  virtual SparseDistribution next() const=0;
  virtual void append(SpecToken)=0;
  virtual void rewind(std::size_t)=0;
  virtual std::vector<SparseDistribution> verify(std::span<const SpecToken> drafts) {
    std::vector<SparseDistribution> result;result.reserve(drafts.size()+1);result.push_back(next());
    for (auto t:drafts) {append(t);result.push_back(next());}return result;
  }
};

namespace sampled_detail {
template<class State,class Law> class Controller {
  struct Pending {
    std::uint64_t id;
    std::unique_ptr<State> target,draft;
    std::vector<Law> p,q;
    std::vector<SpecToken> proposals,output;
    std::size_t matched;
    std::mt19937_64 rng;
  };
  SpeculationLimits limits_;
  std::unique_ptr<State> target_,draft_;
  std::vector<SpecToken> history_,stops_;
  Law p_,q_;
  std::mt19937_64 rng_;
  std::optional<Pending> pending_;
  VerifyRule rule_;
  std::uint64_t generation_=0;
  bool stopped_=false,poisoned_=false;
  std::atomic_flag running_=ATOMIC_FLAG_INIT;
  struct Guard {
    std::atomic_flag& flag;
    explicit Guard(std::atomic_flag& f):flag(f){if(flag.test_and_set(std::memory_order_acquire))throw std::logic_error("Speculator already in use");}
    ~Guard(){flag.clear(std::memory_order_release);}
  };
  static void cancellation(const std::atomic_bool* c) {if(c && c->load(std::memory_order_relaxed))throw std::runtime_error("Speculation cancelled");}
  bool stop_token(SpecToken t) const {return std::binary_search(stops_.begin(),stops_.end(),t);}
  void token(SpecToken t) const {if(t<0 || static_cast<std::uint32_t>(t)>=limits_.vocabulary)throw std::invalid_argument("Token outside vocabulary");}
  void distribution(const Law& d) const {if(d.size()!=limits_.vocabulary)throw std::invalid_argument("Distribution vocabulary differs");}
  void image(const State& state,std::span<const SpecToken> suffix) const {
    const auto actual=state.tokens();
    if(actual.size()!=history_.size()+suffix.size() || !std::equal(history_.begin(),history_.end(),actual.begin()) ||
        !std::equal(suffix.begin(),suffix.end(),actual.begin()+static_cast<std::ptrdiff_t>(history_.size())))
      throw std::logic_error("Branch contains an unexpected prefix");
  }
  void parent() {
    try {image(*target_,{});image(*draft_,{});
      if(poisoned_ || target_->next()!=p_ || draft_->next()!=q_)throw std::logic_error("Committed probabilities changed");
    } catch(...) {poisoned_=true;throw;}
  }
  std::unique_ptr<State> fork(const State& source) {
    auto branch=source.fork();
    if(branch.get()==&source){branch.release();throw std::logic_error("Fork returned the committed owner");}
    if(!branch)throw std::logic_error("Missing fork");
    image(*branch,{});if(branch->next()!=source.next())throw std::logic_error("Fork changed next probabilities");return branch;
  }
  static double draw(std::mt19937_64& rng){return std::generate_canonical<double,53>(rng);}
  static Law required_next(const std::unique_ptr<State>& state) {
    if (!state) throw std::invalid_argument("Target and draft owners required");
    return state->next();
  }
public:
  Controller(std::unique_ptr<State> target,std::unique_ptr<State> draft,
                    SpeculationLimits limits,std::uint64_t seed,std::span<const SpecToken> stops={},
                    VerifyRule rule=VerifyRule::token)
      :limits_(limits),target_(std::move(target)),draft_(std::move(draft)),
       p_(required_next(target_)),q_(required_next(draft_)),rng_(seed),rule_(rule) {
    if(!target_ || !draft_ || !limits.context || !limits.vocabulary || limits.vocabulary>1048576)
      throw std::invalid_argument("Target, draft, context and bounded vocabulary required");
    auto initial=target_->tokens();
    if(initial.empty() || initial.size()>limits.context)throw std::invalid_argument("Nonempty materialized prompt within context required");
    history_.assign(initial.begin(),initial.end());for(auto t:history_)token(t);
    stops_.assign(stops.begin(),stops.end());for(auto t:stops_)token(t);
    std::sort(stops_.begin(),stops_.end());stops_.erase(std::unique(stops_.begin(),stops_.end()),stops_.end());
    distribution(p_);distribution(q_);parent();
  }
  Controller(const Controller&)=delete;
  Controller& operator=(const Controller&)=delete;
  std::span<const SpecToken> history() const noexcept{return history_;}
  bool finished() const noexcept{return stopped_ || history_.size()==limits_.context;}
  bool pending() const noexcept{return pending_.has_value();}
  bool poisoned() const noexcept{return poisoned_;}
  SpeculativeRound prepare(std::size_t output_budget,std::size_t draft_budget,const std::atomic_bool* cancelled=nullptr) {
    Guard guard(running_);parent();cancellation(cancelled);
    if(pending_)throw std::logic_error("Commit or discard pending round first");
    if(!output_budget || finished())throw std::length_error("No remaining output/context");
    if(generation_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("Round identity exhausted");
    auto remaining=std::min(output_budget,limits_.context-history_.size());auto count=std::min({remaining,draft_budget,limits_.max_draft});
    try {
      auto t=fork(*target_),d=fork(*draft_);auto rng=rng_;
      std::vector<SpecToken> proposals;std::vector<Law> q;q.push_back(d->next());
      for(std::size_t i=0;i<count;++i) {
        cancellation(cancelled);distribution(q.back());auto value=q.back().sample(draw(rng));token(value);
        proposals.push_back(value);d->append(value);q.push_back(d->next());
        if(stop_token(value))break;
      }
      parent();image(*d,proposals);cancellation(cancelled);
      auto p=t->verify(proposals);parent();image(*t,proposals);cancellation(cancelled);
      if(p.size()!=proposals.size()+1)throw std::logic_error("Verification requires D+1 distributions");
      for(const auto& value:p)distribution(value);
      for(const auto& value:q)distribution(value);
      if(p.front()!=p_ || p.back()!=t->next() || q.front()!=q_ || q.back()!=d->next())throw std::logic_error("Verification endpoints differ");
      std::vector<SpecToken> output;std::size_t matched=0;
      if(rule_==VerifyRule::block) {
        // A stop draft is always the last proposal, so the round's output is a prefix of the block round's.
        const auto [tau,next]=block_verify<Law>(p,q,proposals,[&]{return draw(rng);});
        output.assign(proposals.begin(),proposals.begin()+static_cast<std::ptrdiff_t>(tau));
        if(tau<proposals.size() || output.empty() || !stop_token(output.back()))output.push_back(next);
        if(output.size()>remaining)output.resize(remaining);
        matched=std::min(tau,output.size());
      } else
      while(matched<proposals.size() && output.size()<remaining) {
        auto value=proposals[matched];
        if(draw(rng)<acceptance_probability(p[matched],q[matched],value)) {
          output.push_back(value);++matched;if(stop_token(value))break;
        }else {output.push_back(rejection_distribution(p[matched],q[matched]).sample(draw(rng)));break;}
      }
      if(rule_==VerifyRule::token && matched==proposals.size() && output.size()<remaining &&
         (output.empty() || !stop_token(output.back())))
        output.push_back(p[matched].sample(draw(rng)));
      if(output.empty())throw std::logic_error("Empty sampled round");
      parent();cancellation(cancelled);
      pending_.emplace(Pending{generation_+1,std::move(t),std::move(d),std::move(p),std::move(q),std::move(proposals),std::move(output),matched,rng});
      ++generation_;return {pending_->id,pending_->output,pending_->proposals.size(),pending_->matched};
    }catch(...){parent();throw;}
  }
  SpeculativeCommit commit(std::uint64_t id,std::size_t accepted,const std::atomic_bool* cancelled=nullptr) {
    Guard guard(running_);parent();
    if(!pending_ || pending_->id!=id)throw std::logic_error("Unknown/stale round");
    if(accepted>pending_->output.size())throw std::out_of_range("Commit exceeds output");
    auto round=std::move(*pending_);pending_.reset();
    try {
      cancellation(cancelled);
      SpeculativeCommit result{{round.output.begin(),round.output.begin()+static_cast<std::ptrdiff_t>(accepted)},
        round.proposals.size(),std::min(accepted,round.matched),accepted>round.matched,false};
      if(!accepted)return result;
      auto history=history_;history.insert(history.end(),result.tokens.begin(),result.tokens.end());
      const auto prefix=std::span<const SpecToken>(result.tokens).first(result.accepted_drafts);
      round.target->rewind(history_.size()+result.accepted_drafts);round.draft->rewind(history_.size()+result.accepted_drafts);
      parent();image(*round.target,prefix);image(*round.draft,prefix);cancellation(cancelled);
      if(round.target->next()!=round.p[result.accepted_drafts] || round.draft->next()!=round.q[result.accepted_drafts])
        throw std::logic_error("Rewind did not restore next probabilities");
      if(result.correction_or_bonus){round.target->append(result.tokens.back());round.draft->append(result.tokens.back());}
      parent();image(*round.target,result.tokens);image(*round.draft,result.tokens);cancellation(cancelled);
      auto p=round.target->next(),q=round.draft->next();distribution(p);distribution(q);
      result.stopped=stop_token(result.tokens.back());cancellation(cancelled);
      // Distribution vector moves, ownership swaps and RNG assignment do not throw.
      target_.swap(round.target);draft_.swap(round.draft);history_.swap(history);p_=std::move(p);q_=std::move(q);
      rng_=round.rng;stopped_=result.stopped;return result;
    }catch(...){parent();throw;}
  }
  void discard(std::uint64_t id) {
    Guard guard(running_);if(!pending_ || pending_->id!=id)throw std::logic_error("Unknown/stale round");pending_.reset();
  }
};
} // namespace sampled_detail

class SampledSpeculator:public sampled_detail::Controller<ProbabilityState,Distribution> {
public:
  using Controller::Controller;
};
class SparseSampledSpeculator final:public sampled_detail::Controller<SparseProbabilityState,SparseDistribution> {
public:
  using Controller::Controller;
};

struct DraftObservation {
  std::uint64_t policy,id;
  std::size_t budget;
  std::uint64_t configuration_generation;
  bool operator==(const DraftObservation&) const=default;
};
struct DraftBudgetStatistics {
  std::size_t budget,samples=0;
  std::uint64_t configuration_generation=1;
  double seconds_per_published_token=0;
  std::size_t preparations=0;
  double preparation_seconds=0;
};

// Serialized host policy. Delayed observations carry an issued producing-width
// ticket; learning never infers graph width from how many tokens were accepted.
// Graph preparation is recorded separately from settled per-round timings.
class AdaptiveDraftBudget {
  inline static std::atomic<std::uint64_t> identities_{1};
  std::vector<DraftBudgetStatistics> rows_;
  std::vector<DraftObservation> pending_;
  std::uint64_t identity_,next_observation_=1;
  std::size_t maximum_pending_;
  std::size_t rounds_=0;
  static std::uint64_t new_identity() {
    auto next=identities_.load(std::memory_order_relaxed);
    for (;;) {
      if (next==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Policy identity exhausted");
      if (identities_.compare_exchange_weak(next,next+1,std::memory_order_relaxed)) return next;
    }
  }
  static std::vector<std::size_t> default_budgets(std::size_t maximum) {
    std::vector<std::size_t> result{0};
    for (std::size_t n=1;n<=maximum;) {result.push_back(n);if(n>maximum/2)break;n*=2;}
    if (result.back()!=maximum) result.push_back(maximum);
    return result;
  }
  auto row(std::size_t budget) {
    auto at=std::find_if(rows_.begin(),rows_.end(),[&](const auto& value){return value.budget==budget;});
    if (at==rows_.end()) throw std::invalid_argument("Unsupported draft budget");
    return at;
  }
  auto ticket(const DraftObservation& issued) {
    if (issued.policy!=identity_) throw std::logic_error("Foreign observation policy");
    auto at=std::find_if(pending_.begin(),pending_.end(),[&](const auto& value){return value.id==issued.id;});
    if (at==pending_.end() || *at!=issued) throw std::logic_error("Unknown, changed or consumed observation");
    return at;
  }
  static void timing(double seconds) {
    if (!std::isfinite(seconds) || seconds<0) throw std::invalid_argument("Invalid settled time");
  }
  void learn(DraftBudgetStatistics& target,std::size_t published,double seconds) {
    if (!published) return; // A discard/failure is not a free successful round.
    if (target.samples==std::numeric_limits<std::size_t>::max() || rounds_==std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Observation count exhausted");
    const auto cost=seconds/static_cast<double>(published);
    target.seconds_per_published_token=target.samples?.75*target.seconds_per_published_token+.25*cost:cost;
    ++target.samples;++rounds_;
  }
public:
  explicit AdaptiveDraftBudget(std::size_t maximum):AdaptiveDraftBudget(default_budgets(maximum)) {}
  explicit AdaptiveDraftBudget(std::span<const std::size_t> supported,std::size_t maximum_pending=64)
      :identity_(new_identity()),maximum_pending_(maximum_pending) {
    if (supported.empty() || supported.size()>1024 || !maximum_pending || maximum_pending>4096)
      throw std::invalid_argument("Invalid supported budgets or pending bound");
    for (auto budget:supported) rows_.push_back({budget});
    std::sort(rows_.begin(),rows_.end(),[](const auto& a,const auto& b){return a.budget<b.budget;});
    if (rows_.front().budget!=0 || std::adjacent_find(rows_.begin(),rows_.end(),
        [](const auto& a,const auto& b){return a.budget==b.budget;})!=rows_.end())
      throw std::invalid_argument("Budgets must include zero and be unique");
  }
  AdaptiveDraftBudget(const AdaptiveDraftBudget&)=delete;
  AdaptiveDraftBudget& operator=(const AdaptiveDraftBudget&)=delete;
  std::size_t choose(std::size_t remaining) const {
    for(const auto& row:rows_)if(row.budget<=remaining && !row.samples)return row.budget;
    if(rounds_ && rounds_%32==0) {
      auto at=(rounds_/32)%rows_.size();if(rows_[at].budget<=remaining)return rows_[at].budget;
    }
    const DraftBudgetStatistics* best=&rows_.front();
    for(const auto& row:rows_)if(row.budget<=remaining && row.samples &&
        (!best->samples || row.seconds_per_published_token<best->seconds_per_published_token))best=&row;
    return best->budget;
  }
  DraftObservation begin_observation(std::size_t budget) {
    const auto found=row(budget);
    if (pending_.size()==maximum_pending_) throw std::length_error("Observation slots exhausted");
    if (next_observation_==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Observation identity exhausted");
    DraftObservation issued{identity_,next_observation_,budget,found->configuration_generation};
    pending_.push_back(issued);++next_observation_;return issued;
  }
  bool observe(const DraftObservation& issued,std::size_t published,double complete_seconds) {
    timing(complete_seconds);const auto at=ticket(issued);const auto found=row(issued.budget);
    const bool current=found->configuration_generation==issued.configuration_generation;
    if (current) learn(*found,published,complete_seconds);
    pending_.erase(at);return current && published>0;
  }
  void abandon(const DraftObservation& issued) {pending_.erase(ticket(issued));}
  std::size_t pending() const noexcept {return pending_.size();}
  // Record a new graph/backend configuration. Old issued observations remain
  // consumable, but cannot train the new generation. No device work is implied.
  std::uint64_t record_preparation(std::size_t budget,double settled_seconds) {
    timing(settled_seconds);const auto found=row(budget);
    const auto total=found->preparation_seconds+settled_seconds;
    if (!std::isfinite(total) || found->configuration_generation==std::numeric_limits<std::uint64_t>::max() ||
        found->preparations==std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("Preparation accounting exhausted");
    ++found->configuration_generation;++found->preparations;found->preparation_seconds=total;
    found->samples=0;found->seconds_per_published_token=0;return found->configuration_generation;
  }
  DraftBudgetStatistics statistics(std::size_t budget) const {
    const auto at=std::find_if(rows_.begin(),rows_.end(),[&](const auto& value){return value.budget==budget;});
    if (at==rows_.end()) throw std::invalid_argument("Unsupported draft budget");
    return *at;
  }
  // Compatibility for immediate synchronous measurements only; never mix this
  // untagged form with outstanding delayed measurements.
  void observe(std::size_t budget,std::size_t published,double complete_seconds) {
    timing(complete_seconds);const auto found=row(budget);
    if (!pending_.empty()) throw std::logic_error("Delayed observations require issued tickets");
    learn(*found,published,complete_seconds);
  }
};

// Online estimate of each draft's acceptance from the drafter's own confidence, for drafters without a confidence
// head (vLLM's adaptive verification estimator): p(k) = sigmoid(slope * logit(q_max) + intercept[k]), where q_max
// is the draft law's largest probability at draft position k, the slope is shared and each position has its own
// intercept. observe() records each verified draft's feature and outcome in a bounded window; refit() runs Newton
// steps over it, where the shared slope makes the Hessian an arrowhead, so each step is a scalar solve. A draft is
// worth verifying while its survival, the product of the predicted acceptances of it and every draft before it,
// stays high: cut() returns how many drafts of a round to keep for a survival floor, and admit() splits one budget
// across requests by survival, a prefix of each. The floor and the budget are the engine's to measure (for
// example with AdaptiveDraftBudget's seconds per published token); before `minimum_samples` outcomes the estimator
// predicts 1 and cuts nothing. Trimming drafts never changes the output law of a verification rule.
class AcceptanceEstimator {
  struct Sample {double feature;std::uint32_t position;bool accepted;};
  std::vector<Sample> window_;
  std::size_t next_=0,samples_=0,minimum_samples_;
  double slope_=0,ridge_;
  std::vector<double> intercept_;
  bool fitted_=false;
public:
  explicit AcceptanceEstimator(std::size_t positions,std::size_t window=4096,std::size_t minimum_samples=256,double ridge=1e-3)
      :minimum_samples_(minimum_samples),ridge_(ridge),intercept_(positions,0.) {
    if (!positions || positions>1024 || !window || window>(std::size_t{1}<<24) || !std::isfinite(ridge) || ridge<=0)
      throw std::invalid_argument("Invalid estimator positions/window/ridge");
    window_.reserve(window);
  }
  // logit(q_max), clamped away from 0 and 1 so a one-hot draft law stays finite.
  static double feature(double q_max) {
    if (!(q_max>=0 && q_max<=1)) throw std::invalid_argument("Draft probability outside [0, 1]");
    const double q=std::clamp(q_max,1e-6,1-1e-6);
    return std::log(q/(1-q));
  }
  void observe(std::size_t position,double q_max,bool accepted) {
    if (position>=intercept_.size()) throw std::out_of_range("Draft position outside the estimator");
    const Sample sample{feature(q_max),static_cast<std::uint32_t>(position),accepted};
    if (window_.size()<window_.capacity()) window_.push_back(sample);
    else {window_[next_]=sample;next_=(next_+1)%window_.size();}
    ++samples_;
  }
  // Newton steps on the ridge-regularized log-likelihood of the window. Returns the steps taken.
  std::size_t refit(std::size_t steps=8) {
    if (window_.size()<minimum_samples_) return 0;
    const std::size_t K=intercept_.size();
    std::vector<double> gb(K),hb(K),hwb(K);
    std::size_t taken=0;
    for (;taken<steps;++taken) {
      double gw=-ridge_*slope_,hw=ridge_;
      for (std::size_t k=0;k<K;++k){gb[k]=-ridge_*intercept_[k];hb[k]=ridge_;hwb[k]=0;}
      for (const auto& x:window_) {
        const double mu=1/(1+std::exp(-(slope_*x.feature+intercept_[x.position]))),r=(x.accepted?1.:0.)-mu,v=mu*(1-mu);
        gw+=r*x.feature;hw+=v*x.feature*x.feature;gb[x.position]+=r;hb[x.position]+=v;hwb[x.position]+=v*x.feature;
      }
      // Arrowhead solve: eliminate the intercepts, solve the slope, back-substitute.
      double schur=hw,rhs=gw;
      for (std::size_t k=0;k<K;++k){schur-=hwb[k]*hwb[k]/hb[k];rhs-=hwb[k]*gb[k]/hb[k];}
      const double dw=rhs/schur;double largest=std::abs(dw);
      slope_+=dw;
      for (std::size_t k=0;k<K;++k){const double db=(gb[k]-hwb[k]*dw)/hb[k];intercept_[k]+=db;largest=std::max(largest,std::abs(db));}
      if (largest<1e-9){++taken;break;}
    }
    fitted_=true;return taken;
  }
  bool fitted() const noexcept {return fitted_;}
  std::size_t samples() const noexcept {return samples_;}
  double slope() const noexcept {return slope_;}
  double intercept(std::size_t position) const {return intercept_.at(position);}
  double predict(std::size_t position,double q_max) const {
    if (position>=intercept_.size()) throw std::out_of_range("Draft position outside the estimator");
    if (!fitted_) return 1;
    return 1/(1+std::exp(-(slope_*feature(q_max)+intercept_[position])));
  }
  // Survival of each draft: the product of the predicted acceptances up to it.
  std::vector<double> survival(std::span<const double> q_max) const {
    std::vector<double> result(q_max.size());double alive=1;
    for (std::size_t k=0;k<q_max.size();++k){alive*=predict(k,q_max[k]);result[k]=alive;}
    return result;
  }
  // Drafts to keep: the longest prefix whose survival stays at or above the floor.
  std::size_t cut(std::span<const double> q_max,double floor) const {
    if (!(floor>=0 && floor<=1)) throw std::invalid_argument("Survival floor outside [0, 1]");
    const auto alive=survival(q_max);std::size_t kept=0;
    while (kept<alive.size() && alive[kept]>=floor) ++kept;
    return kept;
  }
  // One draft budget over several requests: the `budget` (request, draft) slots of highest survival, ties to the
  // earlier request and draft. Survival falls along a request, so each request keeps a prefix. Returns per request.
  static std::vector<std::size_t> admit(std::span<const std::vector<double>> survivals,std::size_t budget) {
    struct Slot {double survival;std::size_t request,draft;};
    std::vector<Slot> slots;
    for (std::size_t r=0;r<survivals.size();++r)
      for (std::size_t k=0;k<survivals[r].size();++k) {
        if (k && survivals[r][k]>survivals[r][k-1]) throw std::invalid_argument("Survival must not rise along a request");
        slots.push_back({survivals[r][k],r,k});
      }
    const auto order=[](const Slot& a,const Slot& b){
      return a.survival!=b.survival?a.survival>b.survival:a.request!=b.request?a.request<b.request:a.draft<b.draft;};
    std::sort(slots.begin(),slots.end(),order);
    std::vector<std::size_t> kept(survivals.size());
    for (std::size_t i=0;i<std::min(budget,slots.size());++i) kept[slots[i].request]=std::max(kept[slots[i].request],slots[i].draft+1);
    return kept;
  }
};
} // namespace crit::techniques
