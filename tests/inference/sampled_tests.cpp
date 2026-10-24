#include "sampled.h"
#include <array>
#include <iostream>
using namespace crit::techniques;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class F>void rejects(F f){try{f();}catch(const std::exception&){return;}throw std::runtime_error("Expected rejection");}
struct Fault {bool verify=false,repair=false,rewind=false,bad_count=false,bad_endpoint=false,null_fork=false,alias_fork=false;int live=0;std::atomic_bool* cancel=nullptr;};
class Markov final:public ProbabilityState {
  std::vector<SpecToken> history_;
  bool draft_;
  std::shared_ptr<Fault> faults_;
public:
  Markov(std::vector<SpecToken> h,bool draft,std::shared_ptr<Fault> f):history_(std::move(h)),draft_(draft),faults_(std::move(f)){++faults_->live;}
  Markov(const Markov& o):history_(o.history_),draft_(o.draft_),faults_(o.faults_){++faults_->live;}
  ~Markov() override{--faults_->live;}
  std::unique_ptr<ProbabilityState> fork() const override {
    if(faults_->null_fork)return {};
    if(faults_->alias_fork)return std::unique_ptr<ProbabilityState>(const_cast<Markov*>(this));
    return std::make_unique<Markov>(*this);
  }
  std::span<const SpecToken> tokens() const noexcept override{return history_;}
  Distribution next() const override {
    if(draft_)return Distribution(history_.back()==0?std::vector<double>{.7,.2,.1}:std::vector<double>{.1,.7,.2});
    return Distribution(history_.back()==0?std::vector<double>{.1,.3,.6}:history_.back()==1?std::vector<double>{.5,.2,.3}:std::vector<double>{.4,.5,.1});
  }
  void append(SpecToken t) override {history_.push_back(t);if(faults_->repair)throw std::bad_alloc();}
  void rewind(std::size_t n) override {history_.resize(n);if(faults_->rewind)throw std::runtime_error("rewind failed");}
  std::vector<Distribution> verify(std::span<const SpecToken> d) override {
    auto result=ProbabilityState::verify(d);
    if(faults_->cancel)*faults_->cancel=true;
    if(faults_->verify)throw std::runtime_error("verify failed");
    if(faults_->bad_count)result.pop_back();
    if(faults_->bad_endpoint)result.front()=Distribution({1,0,0});
    return result;
  }
};
std::unique_ptr<SampledSpeculator> loop(std::uint64_t seed,std::shared_ptr<Fault> t,std::shared_ptr<Fault> d,std::span<const SpecToken> stops={}) {
  return std::make_unique<SampledSpeculator>(std::make_unique<Markov>(std::vector<SpecToken>{0},false,t),
    std::make_unique<Markov>(std::vector<SpecToken>{0},true,d),SpeculationLimits{16,8,3},seed,stops);
}
// Retain the same model arithmetic and faults while independently exercising
// sparse/fallback rows, including representation changes after rewind.
class SparseMarkov final:public SparseProbabilityState {
  std::unique_ptr<ProbabilityState> inner_;
  std::shared_ptr<Fault> faults_;
  std::size_t capacity_;
public:
  SparseMarkov(std::unique_ptr<ProbabilityState> state,std::shared_ptr<Fault> faults,std::size_t capacity)
      :inner_(std::move(state)),faults_(std::move(faults)),capacity_(capacity) {}
  std::unique_ptr<SparseProbabilityState> fork() const override {
    if (faults_->null_fork) return {};
    if (faults_->alias_fork) return std::unique_ptr<SparseProbabilityState>(const_cast<SparseMarkov*>(this));
    return std::make_unique<SparseMarkov>(inner_->fork(),faults_,capacity_);
  }
  std::span<const SpecToken> tokens() const noexcept override {return inner_->tokens();}
  SparseDistribution next() const override {
    return SparseDistribution::from_dense(inner_->next(),tokens().back()==1?0:capacity_);
  }
  void append(SpecToken token) override {inner_->append(token);}
  void rewind(std::size_t count) override {inner_->rewind(count);}
  std::vector<SparseDistribution> verify(std::span<const SpecToken> proposals) override {
    auto dense=inner_->verify(proposals);std::vector<SparseDistribution> result;
    for (auto& row:dense) result.push_back(SparseDistribution::from_dense(std::move(row),capacity_));
    return result;
  }
};
std::unique_ptr<SparseSampledSpeculator> sparse_loop(std::uint64_t seed,std::shared_ptr<Fault> t,
    std::shared_ptr<Fault> d,std::span<const SpecToken> stops={},std::size_t capacity=3) {
  return std::make_unique<SparseSampledSpeculator>(
    std::make_unique<SparseMarkov>(std::make_unique<Markov>(std::vector<SpecToken>{0},false,t),t,capacity),
    std::make_unique<SparseMarkov>(std::make_unique<Markov>(std::vector<SpecToken>{0},true,d),d,capacity),
    SpeculationLimits{16,8,3},seed,stops);
}
bool same(std::span<const SpecToken> a,std::span<const SpecToken> b) {
  return std::equal(a.begin(),a.end(),b.begin(),b.end());
}

std::array<double,3> filter_oracle(const std::array<float,3>& scores,double temperature,
                                 std::size_t top_k,double top_p,double min_p) {
  std::array<double,3> result{};
  const auto best=std::max_element(scores.begin(),scores.end());
  if (!temperature) {result[static_cast<std::size_t>(best-scores.begin())]=1;return result;}
  // Rank each token by the declared strict score/ID ordering, without a sort.
  std::array<std::size_t,3> order{};
  for (std::size_t x=0;x<3;++x) {
    std::size_t rank=0;
    for (std::size_t y=0;y<3;++y) if (scores[y]>scores[x] || (scores[y]==scores[x] && y<x)) ++rank;
    order[rank]=x;
  }
  const auto retained=top_k?top_k:3;std::array<long double,3> mass{};long double total=0;
  for (std::size_t rank=0;rank<retained;++rank) {
    const auto id=order[rank];mass[id]=std::exp((static_cast<long double>(scores[id])-*best)/temperature);total+=mass[id];
  }
  long double previous=0,filtered=0;
  for (std::size_t rank=0;rank<retained;++rank) {
    const auto id=order[rank];const auto value=mass[id];
    const bool nucleus=previous<static_cast<long double>(top_p)*total;
    previous+=value;
    if (!nucleus || value<min_p) mass[id]=0;
    filtered+=mass[id];
  }
  for (std::size_t id=0;id<3;++id) result[id]=static_cast<double>(mass[id]/filtered);
  return result;
}
void sparse_arithmetic() {
  const std::array<ProbabilityMass,5> entries{{{2,0},{0,.25},{2,.5},{0,.25},{1,0}}};
  auto permuted=entries;std::array<std::size_t,5> order{0,1,2,3,4};
  const SparseDistribution canonical(4,{entries.begin(),entries.end()},2);
  do {
    for (std::size_t i=0;i<order.size();++i) permuted[i]=entries[order[i]];
    check(SparseDistribution(4,{permuted.begin(),permuted.end()},2)==canonical,"Duplicate permutation changed canonical law");
  } while (std::next_permutation(order.begin(),order.end()));
  check(canonical.support().size()==2 && canonical[0]==.5 && canonical[2]==.5 && canonical[1]==0,
        "Duplicate merging or zero compaction failed");
  check(SparseDistribution(4,{{2,1},{2,0},{2,0}},0)[2]==1,"Zero duplicate overwrote greedy probability");
  rejects([]{SparseDistribution(0,{{0,1}});});rejects([]{SparseDistribution(1048577,{{0,1}});});
  rejects([]{SparseDistribution(3,{});});rejects([]{SparseDistribution(3,{{0,0},{1,0}});});
  rejects([]{SparseDistribution(3,{{-1,1}});});rejects([]{SparseDistribution(3,{{3,1}});});
  rejects([]{SparseDistribution(3,{{0,-1}});});rejects([]{SparseDistribution(3,{{0,NAN}});});
  rejects([]{SparseDistribution(3,{{0,INFINITY}});});rejects([]{SparseDistribution(3,{{0,1}},1048577);});
  rejects([&]{canonical[-1];});rejects([&]{canonical[4];});rejects([&]{canonical.sample(NAN);});
  rejects([&]{canonical.sample(1);});rejects([&]{canonical.sample(-.1);});
  rejects([]{acceptance_probability(SparseDistribution(2,{{0,1}}),SparseDistribution(3,{{0,1}}),0);});
  rejects([]{acceptance_probability(SparseDistribution(3,{{0,1}}),SparseDistribution(3,{{1,1}}),0);});
  rejects([]{rejection_distribution(SparseDistribution(3,{{0,1}}),SparseDistribution(3,{{0,1}}));});

  // Every three-token integer-mass law with nonempty support, across all sparse
  // capacities. This exercises disjoint supports and mixed sparse/dense rows.
  std::vector<Distribution> laws;
  for (int a=0;a<3;++a) for (int b=0;b<3;++b) for (int c=0;c<3;++c)
    if (a+b+c) laws.emplace_back(std::vector<double>{static_cast<double>(a),static_cast<double>(b),static_cast<double>(c)});
  for (const auto& p:laws) for (const auto& q:laws) for (std::size_t capacity=0;capacity<=3;++capacity) {
    const auto sparse_p=SparseDistribution::from_dense(p,capacity),sparse_q=SparseDistribution::from_dense(q,3-capacity);
    check(sparse_p.to_dense()==p && sparse_q.to_dense()==q,"Sparse conversion renormalized the dense law");
    check(sparse_p.support().size()<=capacity && (sparse_p.is_dense()==sparse_p.support().empty()),"Sparse capacity/views disagree");
    check(sparse_p==SparseDistribution::from_dense(p,3-capacity),"Equivalent representations compare differently");
    std::array<double,3> actual{};double rejected=0;
    for (SpecToken id=0;id<3;++id) if (q[id]>0) {
      actual[static_cast<std::size_t>(id)]=q[id]*acceptance_probability(sparse_p,sparse_q,id);
      rejected+=q[id]-actual[static_cast<std::size_t>(id)];
    }
    if (rejected>1e-16) {
      const auto correction=rejection_distribution(sparse_p,sparse_q);const auto dense=rejection_distribution(p,q);
      check(correction.to_dense()==dense,"Sparse correction changed normalized residual mass");
      for (SpecToken id=0;id<3;++id) actual[static_cast<std::size_t>(id)]+=rejected*correction[id];
    }
    for (SpecToken id=0;id<3;++id) check(std::abs(actual[static_cast<std::size_t>(id)]-p[id])<3e-15,"Sparse accepted/correction mass differs");
    for (int draw=0;draw<101;++draw) check(sparse_p.sample(static_cast<double>(draw)/101)==p.sample(static_cast<double>(draw)/101),"Sparse CDF changes ID order");
  }

  // Actual two-position conditional mass: enumerate every complete draft path,
  // first/second acceptance and correction; a first rejection continues from
  // its correction token's conditional law, never from the rejected proposal.
  for (std::size_t i=0;i<laws.size();++i) for (std::size_t j=0;j<laws.size();++j) {
    std::array<SparseDistribution,3> p{SparseDistribution::from_dense(laws[i],i%4),
      SparseDistribution::from_dense(laws[(i+1)%laws.size()],1),SparseDistribution::from_dense(laws[(i+3)%laws.size()],3)};
    std::array<SparseDistribution,3> q{SparseDistribution::from_dense(laws[j],j%4),
      SparseDistribution::from_dense(laws[(j+2)%laws.size()],3),SparseDistribution::from_dense(laws[(j+4)%laws.size()],0)};
    std::array<double,9> joint{};
    for (SpecToken a=0;a<3;++a) if (q[0][a]>0) for (SpecToken b=0;b<3;++b) if (q[static_cast<std::size_t>(a)][b]>0) {
      const auto path=q[0][a]*q[static_cast<std::size_t>(a)][b],first=acceptance_probability(p[0],q[0],a);
      const auto second=acceptance_probability(p[static_cast<std::size_t>(a)],q[static_cast<std::size_t>(a)],b);
      joint[static_cast<std::size_t>(a*3+b)]+=path*first*second;
      if (second<1) {
        const auto residual=rejection_distribution(p[static_cast<std::size_t>(a)],q[static_cast<std::size_t>(a)]);
        for (SpecToken c=0;c<3;++c) joint[static_cast<std::size_t>(a*3+c)]+=path*first*(1-second)*residual[c];
      }
      if (first<1) {
        const auto residual=rejection_distribution(p[0],q[0]);
        for (SpecToken c=0;c<3;++c) for (SpecToken d=0;d<3;++d)
          joint[static_cast<std::size_t>(c*3+d)]+=path*(1-first)*residual[c]*p[static_cast<std::size_t>(c)][d];
      }
    }
    for (SpecToken a=0;a<3;++a) for (SpecToken b=0;b<3;++b)
      check(std::abs(joint[static_cast<std::size_t>(a*3+b)]-p[0][a]*p[static_cast<std::size_t>(a)][b])<8e-15,"Conditional sparse joint mass differs");
  }

  for (float a:{-1.F,0.F,1.F}) for (float b:{-1.F,0.F,1.F}) for (float c:{-1.F,0.F,1.F})
    for (double temperature:{0.,.5,1.,2.}) for (std::size_t k=0;k<=3;++k)
      for (double top_p:{.2,.5,1.}) for (double min_p:{0.,.25,.5,1.}) {
        const std::array<float,3> scores{a,b,c};const auto expected=filter_oracle(scores,temperature,k,top_p,min_p);
        const auto dense=Distribution::logits(scores,temperature,k,top_p,min_p);
        for (SpecToken id=0;id<3;++id) check(std::abs(dense[id]-expected[static_cast<std::size_t>(id)])<3e-15,"Filter order/tie/min-p law differs");
        for (std::size_t capacity=0;capacity<=3;++capacity)
          check(SparseDistribution::logits(scores,temperature,k,top_p,min_p,capacity).to_dense()==dense,"Bounded support changed filtering law");
      }
  const auto ties=SparseDistribution::logits(std::array<float,4>{1,1,1,0},1,2,1,1,1);
  check(ties.is_dense() && ties[0]==.5 && ties[1]==.5 && ties[2]==0,"Fixed-count top-k tie support truncated/expanded");
  check(SparseDistribution::logits(std::array<float,4>{1,1,1,0},1,0,1,1,2).is_dense(),"Min-p ties silently truncated at capacity");
  rejects([]{Distribution::logits(std::array<float,1>{0},1,0,1,-.1);});
  rejects([]{Distribution::logits(std::array<float,1>{0},0,0,1,NAN);});
  rejects([]{Distribution::logits(std::array<float,1>{0},1,0,1,1.1);});
}
void arithmetic() {
  rejects([]{Distribution({0,0});});rejects([]{Distribution({1,-1});});rejects([]{Distribution({NAN,1});});
  const std::array<std::array<double,3>,6> distributions{{{{1,0,0}},{{0,0,1}},{{.2,.3,.5}},{{.5,.5,0}},{{1e-14,1,0}},{{.333,.333,.334}}}};
  // Exhaustive token support: accepted proposal mass + residual mass equals p.
  for(auto a:distributions)for(auto b:distributions) {
    Distribution p({a.begin(),a.end()}),q({b.begin(),b.end()});double rejected=0;std::array<double,3> actual{};
    for(SpecToken x=0;x<3;++x)if(q[x]>0){actual[x]=q[x]*acceptance_probability(p,q,x);rejected+=q[x]-actual[x];}
    if(rejected>1e-16){auto residual=rejection_distribution(p,q);for(SpecToken x=0;x<3;++x)actual[x]+=rejected*residual[x];}
    for(SpecToken x=0;x<3;++x)check(std::abs(actual[x]-p[x])<3e-15,"Exact support distribution mismatch");
    // Exhaustive two-position conditional probabilities use independently p(y|x).
    for(SpecToken x=0;x<3;++x)for(SpecToken y=0;y<3;++y)check(std::abs(actual[x]*distributions[2][y]-p[x]*distributions[2][y])<3e-15,"Joint mass mismatch");
  }
  auto logits=Distribution::logits(std::array<float,4>{0,2,1,-2},1,2,.7);
  check(logits[1]==1 && logits[0]==0 && logits[2]==0,"Top-k/top-p boundary wrong");
  check(Distribution::logits(std::array<float,3>{1,1,0},0)[0]==1,"Greedy tie differs");
  rejects([]{Distribution({1}).sample(1);});rejects([]{Distribution::logits(std::array<float,1>{0},-1);});
  sparse_arithmetic();
}
// Three tokens per trial with draft budgets 0 .. 4: a round can accept two drafts and reject (or resample after)
// the third, so the residual law is exercised after accepted drafts, not only at a round's first position.
template<class Factory> void statistical_case(const char* name,Factory make) {
  constexpr std::size_t trials=90000;std::array<std::size_t,27> counts{};
  for(std::size_t seed=0;seed<trials;++seed) {
    auto t=std::make_shared<Fault>(),d=std::make_shared<Fault>();auto s=make(seed,t,d);
    std::vector<SpecToken> emitted;
    while(emitted.size()<3){auto r=s->prepare(3-emitted.size(),seed%5);auto c=s->commit(r.id,r.tokens.size());emitted.insert(emitted.end(),c.tokens.begin(),c.tokens.end());}
    ++counts[static_cast<std::size_t>(emitted[0]*9+emitted[1]*3+emitted[2])];s.reset();check(t->live==0 && d->live==0,"Statistical fixture leaked states");
  }
  const std::array<double,3> p{.1,.3,.6};
  const std::array<std::array<double,3>,3> conditional{{{{.1,.3,.6}},{{.5,.2,.3}},{{.4,.5,.1}}}};
  // Fixed 0.008 absolute-frequency bound, chosen before measurement (>= 4.8
  // standard deviations even at probability .25); no seeded trajectory claim.
  // It holds for the two-token marginal and for each of the 27 three-token cells.
  for(std::size_t x=0;x<3;++x)for(std::size_t y=0;y<3;++y) {
    std::size_t pair=0;
    for(std::size_t z=0;z<3;++z) {
      pair+=counts[x*9+y*3+z];
      check(std::abs(static_cast<double>(counts[x*9+y*3+z])/trials-p[x]*conditional[x][y]*conditional[y][z])<.008,"Sampled three-token distribution differs");
    }
    check(std::abs(static_cast<double>(pair)/trials-p[x]*conditional[x][y])<.008,"Sampled joint distribution differs");
  }
  // Pearson's statistic over the 27 cells, against chi-square(26)'s 1 - 1e-6 quantile (75.55), a fixed quantile
  // rather than a fit to the measured value. It sees what the per-cell bound cannot: a wrong residual only after two accepted drafts (drawing
  // from p instead of the residual) gives 138 here, the rule as written about 32.
  double pearson=0;
  for(std::size_t i=0;i<27;++i) {
    const double expected=trials*p[i/9]*conditional[i/9][(i/3)%3]*conditional[(i/3)%3][i%3],observed=static_cast<double>(counts[i]);
    pearson+=(observed-expected)*(observed-expected)/expected;
  }
  check(pearson<75.55,"Sampled three-token distribution fails the chi-square bound");
  std::cout<<name<<": "<<trials<<" three-token histories passed fixed distribution bounds (chi-square "<<pearson<<" of 75.55)\n";
}
void statistical() {
  statistical_case("dense",[](std::uint64_t seed,std::shared_ptr<Fault> t,std::shared_ptr<Fault> d){return loop(seed,t,d);});
  statistical_case("sparse",[](std::uint64_t seed,std::shared_ptr<Fault> t,std::shared_ptr<Fault> d){return sparse_loop(seed,t,d);});
  statistical_case("sparse fallback",[](std::uint64_t seed,std::shared_ptr<Fault> t,std::shared_ptr<Fault> d){return sparse_loop(seed,t,d,{},2);});
}
template<class Factory> void transaction_cases(Factory make) {
  auto t=std::make_shared<Fault>(),d=std::make_shared<Fault>();auto s=make(42,t,d),ordinary=make(42,t,d);
  auto r=s->prepare(8,4);auto expected=std::vector<SpecToken>(r.tokens.begin(),r.tokens.end());
  rejects([&]{s->commit(r.id+1,1);});rejects([&]{s->commit(r.id,r.tokens.size()+1);});check(s->pending(),"Invalid commit consumed round");
  s->discard(r.id);auto retry=s->prepare(8,4);check(std::equal(expected.begin(),expected.end(),retry.tokens.begin(),retry.tokens.end()),"Discard advanced RNG");
  s->commit(retry.id,0);retry=s->prepare(8,4);check(std::equal(expected.begin(),expected.end(),retry.tokens.begin(),retry.tokens.end()),"Zero commit advanced RNG");
  s->commit(retry.id,1);check(s->history().size()==2,"Partial commit wrong");
  std::atomic_bool cancel=true;rejects([&]{s->prepare(4,4,&cancel);});cancel=false;
  auto c=s->prepare(4,4);cancel=true;rejects([&]{s->commit(c.id,1,&cancel);});check(!s->pending() && s->history().size()==2,"Cancelled commit changed parent");
  cancel=false;
  for(auto member:{&Fault::verify,&Fault::bad_count,&Fault::bad_endpoint,&Fault::null_fork,&Fault::alias_fork}) {
    t.get()->*member=true;rejects([&]{s->prepare(4,4);});t.get()->*member=false;
    check(s->history().size()==2 && !s->pending(),"Failed verification published state");
  }
  c=s->prepare(4,4);t->rewind=true;rejects([&]{s->commit(c.id,1);});t->rewind=false;
  check(!s->pending() && s->history().size()==2,"Failed rewind changed parent");
  c=s->prepare(1,0);t->repair=true;rejects([&]{s->commit(c.id,1);});t->repair=false;
  check(!s->pending() && s->history().size()==2,"Failed correction changed parent");
  auto retry_after_failure=s->prepare(1,0);auto same_seed=make(42,t,d);
  auto original=same_seed->prepare(8,4);same_seed->commit(original.id,1);
  auto control=same_seed->prepare(1,0);
  check(std::equal(control.tokens.begin(),control.tokens.end(),retry_after_failure.tokens.begin(),retry_after_failure.tokens.end()),"Failed commit advanced RNG");
  s->discard(retry_after_failure.id);same_seed.reset();
  c=s->prepare(1,0);d->repair=true;rejects([&]{s->commit(c.id,1);});d->repair=false;
  check(!s->pending() && s->history().size()==2,"Failed draft correction changed target parent");
  t->cancel=&cancel;rejects([&]{s->prepare(4,4,&cancel);});t->cancel=nullptr;cancel=false;
  auto stop=make(99,t,d,std::array<SpecToken,3>{0,1,2});auto e=stop->prepare(8,8);check(e.tokens.size()==1,"Output continued past stop");stop->commit(e.id,1);check(stop->finished(),"Stop not terminal");
  s.reset();ordinary.reset();stop.reset();check(!t->live && !d->live,"Transactions leaked states");
}
void transactions() {
  transaction_cases([](std::uint64_t seed,std::shared_ptr<Fault> t,std::shared_ptr<Fault> d,std::span<const SpecToken> stops={}){return loop(seed,t,d,stops);});
  transaction_cases([](std::uint64_t seed,std::shared_ptr<Fault> t,std::shared_ptr<Fault> d,std::span<const SpecToken> stops={}){return sparse_loop(seed,t,d,stops);});
  rejects([]{SampledSpeculator(nullptr,nullptr,{16,8,3},42);});
  rejects([]{SparseSampledSpeculator(nullptr,nullptr,{16,8,3},42);});
  for (std::uint64_t seed=0;seed<128;++seed) for (std::size_t capacity=0;capacity<=3;++capacity) {
    auto t=std::make_shared<Fault>(),d=std::make_shared<Fault>();auto dense=loop(seed,t,d);auto sparse=sparse_loop(seed,t,d,{},capacity);
    std::size_t attempt=0;
    while (!dense->finished()) {
      const auto remaining=16-dense->history().size(),budget=(attempt+static_cast<std::size_t>(seed))%9;
      const auto a=dense->prepare(remaining,budget),b=sparse->prepare(remaining,budget);
      check(same(a.tokens,b.tokens) && a.proposed==b.proposed && a.accepted_drafts==b.accepted_drafts,"Sparse round differs from dense under same draws");
      if (attempt%7==0) {dense->discard(a.id);sparse->discard(b.id);}
      else {
        const auto publish=attempt%7==1?0:attempt%7==2?std::size_t{1}:a.tokens.size();
        const auto x=dense->commit(a.id,publish),y=sparse->commit(b.id,publish);
        check(same(x.tokens,y.tokens) && x.accepted_drafts==y.accepted_drafts && x.correction_or_bonus==y.correction_or_bonus,
              "Sparse partial/zero commit differs");
      }
      check(same(dense->history(),sparse->history()),"Sparse committed prefix differs");
      ++attempt;check(attempt<100,"Sparse coupling failed to make progress");
    }
    dense.reset();sparse.reset();check(t->live==0 && d->live==0,"Sparse coupling leaked states");
  }
}
void adaptive() {
  AdaptiveDraftBudget policy(8);
  for(auto budget:{0U,1U,2U,4U,8U}){check(policy.choose(8)==budget,"Exploration budget wrong");policy.observe(budget,2,budget==4?.01:.1);}
  check(policy.choose(8)==4,"Policy ignored total cost");check(policy.choose(1)<=1,"Policy exceeded remaining budget");
  policy.observe(0,0,0);check(policy.choose(8)==4,"Discard was treated as a free round");rejects([&]{policy.observe(3,1,.1);});
  rejects([]{AdaptiveDraftBudget(std::array<std::size_t,2>{1,2});});
  rejects([]{AdaptiveDraftBudget(std::array<std::size_t,3>{0,2,2});});
  rejects([]{AdaptiveDraftBudget(std::span<const std::size_t>{});});
  rejects([]{AdaptiveDraftBudget(std::array<std::size_t,1>{0},0);});
  AdaptiveDraftBudget tagged(std::array<std::size_t,4>{8,0,2,5},2),foreign(8);
  check(tagged.choose(0)==0 && tagged.choose(8)==0,"Explicit budgets lack bounded zero fallback");
  const auto wide=tagged.begin_observation(5),narrow=tagged.begin_observation(2);
  rejects([&]{tagged.begin_observation(8);});rejects([&]{tagged.begin_observation(3);});
  rejects([&]{tagged.observe(0,1,.1);});rejects([&]{foreign.observe(wide,1,.1);});
  auto changed=wide;changed.budget=8;rejects([&]{tagged.observe(changed,1,.1);});
  changed=wide;++changed.configuration_generation;rejects([&]{tagged.observe(changed,1,.1);});
  rejects([&]{tagged.observe(wide,1,NAN);});check(tagged.pending()==2,"Bad observation consumed valid ticket");
  check(tagged.observe(narrow,1,.04),"Out-of-order narrow observation ignored");
  check(tagged.observe(wide,1,.02),"Wide observation incorrectly inferred from one accepted token");
  check(tagged.statistics(5).samples==1 && tagged.statistics(2).seconds_per_published_token==.04,"Width cost assigned to wrong row");
  rejects([&]{tagged.observe(wide,1,.001);});rejects([&]{tagged.abandon(wide);});
  const auto old=tagged.begin_observation(5);const auto old_generation=old.configuration_generation;
  check(tagged.record_preparation(5,10)>old_generation,"Graph configuration generation did not advance");
  check(!tagged.observe(old,1,.000001),"Stale graph observation trained replacement");
  check(tagged.statistics(5).samples==0 && tagged.statistics(5).preparation_seconds==10,"Preparation mixed into successful round cost");
  const auto current=tagged.begin_observation(5);
  check(tagged.observe(current,2,.02) && tagged.statistics(5).seconds_per_published_token==.01,"Current graph observation lost");
  const auto discarded=tagged.begin_observation(5);check(!tagged.observe(discarded,0,0),"Discard credited as a free successful round");
  const auto abandoned=tagged.begin_observation(8);tagged.abandon(abandoned);rejects([&]{tagged.observe(abandoned,1,.1);});
  rejects([&]{tagged.record_preparation(5,-1);});rejects([&]{tagged.record_preparation(3,1);});
  check(tagged.pending()==0 && tagged.statistics(5).samples==1,"Observation retirement/statistics inconsistent");
  tagged.observe(0,1,.1);tagged.observe(8,1,.1);
  check(tagged.choose(8)==5 && tagged.choose(4)==2,"Policy selected an unsupported or slower width");
}
int main(int argc,char** argv){try{std::string mode=argc>1?argv[1]:"all";if(mode=="all"||mode=="arithmetic")arithmetic();if(mode=="all"||mode=="statistical")statistical();if(mode=="all"||mode=="transactions")transactions();if(mode=="all"||mode=="adaptive")adaptive();std::cout<<"Sampled "<<mode<<" passed\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
