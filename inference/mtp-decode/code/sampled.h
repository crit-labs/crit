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
inline Distribution rejection_distribution(const Distribution& target,const Distribution& draft) {
  if (target.size()!=draft.size()) throw std::invalid_argument("Probability vocabularies differ");
  std::vector<double> weights(target.size());
  for (std::size_t i=0;i<weights.size();++i) weights[i]=std::max(0.,target.values()[i]-draft.values()[i]);
  return Distribution(std::move(weights));
}

inline double acceptance_probability(const SparseDistribution& target,const SparseDistribution& draft,SpecToken token) {
  if (target.size()!=draft.size() || !(draft[token]>0)) throw std::invalid_argument("Invalid sparse proposal distribution/token");
  return std::min(1.,target[token]/draft[token]);
}
inline SparseDistribution rejection_distribution(const SparseDistribution& target,const SparseDistribution& draft) {
  if (target.size()!=draft.size()) throw std::invalid_argument("Sparse probability vocabularies differ");
  std::vector<ProbabilityMass> residual;
  const auto append=[&](SpecToken id,double probability) {
    const auto mass=std::max(0.,probability-draft[id]);if (mass>0) residual.push_back({id,mass});
  };
  if (target.is_dense()) {
    const auto values=target.dense_values();
    for (std::size_t id=0;id<values.size();++id) if (values[id]>0) append(static_cast<SpecToken>(id),values[id]);
  } else for (const auto& entry:target.support()) append(entry.token,entry.weight);
  return SparseDistribution(target.size(),std::move(residual),std::min(target.sparse_capacity(),draft.sparse_capacity()));
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
                    SpeculationLimits limits,std::uint64_t seed,std::span<const SpecToken> stops={})
      :limits_(limits),target_(std::move(target)),draft_(std::move(draft)),
       p_(required_next(target_)),q_(required_next(draft_)),rng_(seed) {
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
      while(matched<proposals.size() && output.size()<remaining) {
        auto value=proposals[matched];
        if(draw(rng)<acceptance_probability(p[matched],q[matched],value)) {
          output.push_back(value);++matched;if(stop_token(value))break;
        }else {output.push_back(rejection_distribution(p[matched],q[matched]).sample(draw(rng)));break;}
      }
      if(matched==proposals.size() && output.size()<remaining && (output.empty() || !stop_token(output.back())))
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
} // namespace crit::techniques
