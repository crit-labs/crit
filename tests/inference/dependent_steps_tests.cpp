#include "recurrent_pipeline.h"
#include <iostream>
#include <condition_variable>
#include <mutex>
#include <map>
#include <set>
using namespace crit::examples::pipeline;
void require(bool ok,const char* reason) { if(!ok) throw std::runtime_error(reason); }
struct Backend final : cc::Backend { void bind_thread() override {} void drain() override {} };
struct Decoder final : cc::TextDecoder {
  std::string push(std::span<const cc::Token> ids) override {
    std::string out; for(auto t:ids)out+=char('a'+t); return out;
  }
  std::string finish() override { return {}; }
};
struct Frontend final : cc::TextFrontend {
  std::vector<cc::Token> tokenize(std::string_view text) const override {
    std::vector<cc::Token> ids; for(auto c:text)ids.push_back(c-'a'); return ids;
  }
  std::string detokenize(std::span<const cc::Token> ids) const override { return Decoder().push(ids); }
  std::unique_ptr<cc::TextDecoder> create_decoder() const override { return std::make_unique<Decoder>(); }
};
struct TextModel : Model {
  using Model::Model;
  std::shared_ptr<const cc::TextFrontend> text_frontend() const override { return std::make_shared<Frontend>(); }
};
struct Observe final : Hooks {
  std::mutex mutex;
  std::condition_variable changed;
  std::set<std::uint64_t> approved_steps;
  std::map<std::uint64_t,int> live;
  int peak=0, discarded=0;
  bool gate=false, released=false, ahead=false;
  cc::BatchExecution::Notify notify;
  void submitted(cc::BatchExecution::Notify n) override { std::lock_guard lock(mutex); notify=std::move(n); }
  void computed(cc::StepKey key) override {
    std::lock_guard lock(mutex);
    peak=std::max(peak,++live[key.request.request]);
    if(key.request.request==1 && key.sequence>=3 && !approved_steps.contains(2)) ahead=true;
    changed.notify_all();
  }
  void approved(cc::StepKey key,bool discard) override {
    std::lock_guard lock(mutex); --live[key.request.request];
    if(key.request.request==1) approved_steps.insert(key.sequence);
    discarded+=int(discard);
  }
  bool publishable(cc::StepKey key) override {
    std::lock_guard lock(mutex); return !gate || released || key.request.request!=1 || key.sequence!=2;
  }
  void wait_ahead() {
    std::unique_lock lock(mutex);
    require(changed.wait_for(lock,std::chrono::seconds(5),[&]{return ahead;}),"dependent compute waited for host publication");
    require(!approved_steps.contains(2),"gated predecessor was published early");
  }
  void release() { std::lock_guard lock(mutex); released=true; if(notify)notify(); }
  void quiesce() override { release(); }
};
cc::Profile profile(std::uint32_t depth) {
  cc::Profile p{64,8,8*1024*1024,1024,2,4};
  p.max_batch_tokens=8; p.max_inflight_batches=6; p.max_request_steps=depth; p.max_step_output_tokens=2;
  return p;
}
cc::Result generate(std::uint32_t depth,const cc::RequestOptions& options={},std::uint32_t output=9,bool varying=false,
                    const std::vector<cc::Token>& prompt={1,3,7,2}) {
  cc::Engine engine(profile(depth),std::make_shared<Backend>(),[&](const auto&,cc::Services services) {
    return std::make_unique<TextModel>(services.resources,32,std::make_shared<Hooks>(),varying);
  });
  auto request=engine.submit(prompt,output,options); return request.wait();
}
struct HostConstraint final : cc::RequestContext { std::size_t host_bytes() const noexcept override { return 0; } };
struct ImmovableModel final : Model {
  using Model::Model;
  std::size_t resident=0;
  std::unique_ptr<cc::ConcurrentSession> open(const cc::RequestView& view) override { ++resident; return Model::open(view); }
  cc::BatchPreparation prepare_incremental(std::span<const cc::WorkItem> items) override {
    // An engine with two distinct non-shareable pools: its initial states fit,
    // but each next step needs the other resident's non-recoverable allocation.
    if(resident==2) for(const auto& item:items) if(item.budget.prefill_complete)
      return {cc::Admission::deferred,{},item.key,std::make_shared<Pressure>()};
    return Model::prepare_incremental(items);
  }
  bool reclaimable(const cc::MemoryPressure&,const cc::ContinuationView&,const cc::ConcurrentSession&) override { return false; }
  std::unique_ptr<cc::Retirement> retire(cc::RequestKey key,cc::ConcurrentSession& session,cc::Finish finish) override {
    --resident; return Model::retire(key,session,finish);
  }
};
void immovable() {
  auto p=profile(1); p.memory_preemption=true;
  ImmovableModel* model=nullptr;
  cc::Engine engine(p,std::make_shared<Backend>(),[&](const auto&,cc::Services services) {
    auto result=std::make_unique<ImmovableModel>(services.resources,16); result->start=false; model=result.get(); return result;
  });
  auto first=engine.submit({1},4), second=engine.submit({2},4); model->start=true;
  int failed=0, completed=0;
  for(auto* request:{&first,&second}) try { require(request->wait().tokens.size()==4,"peer did not finish"); ++completed; }
    catch(const std::runtime_error& e) { require(std::string(e.what()).find("no reclaimable")!=std::string::npos,"unexpected failure"); ++failed; }
  require(failed==1 && completed==1,"unrecoverable pressure did not release a progress route");
}
struct FaultBatch final : cc::BatchExecution {
  std::unique_ptr<cc::BatchExecution> inner;
  std::string fault;
  bool affected=false;
  void submit(Notify notify) override { inner->submit(std::move(notify)); }
  std::shared_ptr<cc::StepContinuation> continuation(cc::StepKey key) override { return inner->continuation(key); }
  bool ready() override { return inner->ready(); }
  std::vector<cc::RowProgress> progress() override {
    auto rows=inner->progress();
    if(affected && fault=="stale") ++rows.front().step;
    return rows;
  }
  void commit(std::span<const cc::CommitDecision> decisions) override {
    if(affected) throw std::runtime_error("injected dependent commit failure");
    inner->commit(decisions);
  }
  void quiesce() override {
    inner->quiesce(); if(affected && fault=="quarantine") throw std::runtime_error("injected unsettled descendant");
  }
};
struct FaultModel final : Model {
  std::string fault;
  using Model::Model;
  std::unique_ptr<cc::BatchExecution> prepare_batch(std::span<const cc::WorkItem> items) override {
    auto operation=std::make_unique<FaultBatch>(); operation->inner=Model::prepare_batch(items); operation->fault=fault;
    operation->affected=std::any_of(items.begin(),items.end(),[](const auto& item){return item.step==2;}); return operation;
  }
};
void fault_case(const std::string& fault) {
  auto observed=std::make_shared<Observe>(); observed->gate=true;
  const auto quarantined=cc::quarantined_engines();
  {
    FaultModel* model=nullptr;
    cc::Engine engine(profile(3),std::make_shared<Backend>(),[&](const auto&,cc::Services services) {
      auto result=std::make_unique<FaultModel>(services.resources,32,observed); result->fault=fault; model=result.get(); return result;
    });
    auto request=engine.submit({1},9);
    struct Release { std::shared_ptr<Observe> observed; ~Release(){observed->release();} } release{observed};
    observed->wait_ahead(); observed->release();
    bool failed=false; std::vector<cc::Token> published;
    try { request.wait([&](cc::Token t){published.push_back(t);}); } catch(const std::runtime_error&) {failed=true;}
    require(failed && published.size()==1,"failed chain published invalid output");
    if(fault!="quarantine") require(model->pool.usage(cc::Memory::host).live==0,"settled descendants retained state");
  }
  require(cc::quarantined_engines()==quarantined+(fault=="quarantine" ? 1 : 0),"descendant quarantine mismatch");
}
void dependency(bool cancel) {
  auto observed=std::make_shared<Observe>(); observed->gate=true;
  auto p=profile(3); p.max_batch_tokens=1;
  cc::Engine engine(p,std::make_shared<Backend>(),[&](const auto&,cc::Services services) {
    return std::make_unique<Model>(services.resources,32,observed,true);
  });
  auto request=engine.submit({1},9);
  struct Release { std::shared_ptr<Observe> observed; ~Release(){observed->release();} } release{observed};
  observed->wait_ahead();
  auto peer=engine.submit({3,1},4);
  require(peer.wait().tokens.size()==4,"independent peer was blocked by gated predecessor");
  if(cancel) request.cancel();
  observed->release();
  const auto result=request.wait();
  if(cancel) {
    require(result.finish==cc::Finish::cancelled && result.tokens.size()<9,"cancellation published suffix");
    require(observed->discarded>0,"terminal descendants were not discarded");
  } else require(result.tokens==generate(1,{},9,true,{1}).tokens,"dependent recurrent continuation changed output");
}
void recovery() {
  auto p=profile(3); p.memory_preemption=true;
  Model* model=nullptr;
  cc::Engine engine(p,std::make_shared<Backend>(),[&](const auto&,cc::Services services) {
    auto result=std::make_unique<Model>(services.resources,3); result->start=false; model=result.get(); return result;
  });
  cc::RequestOptions options; options.sampling.temperature=.7; options.sampling.seed=912;
  options.sampling.frequency_penalty=.4; options.sampling.repetition_penalty=1.2;
  auto a=engine.submit({1,3,7,2},9,options), b=engine.submit({1,3,7,2},9,options);
  model->start=true;
  const auto ar=a.wait(), br=b.wait();
  const auto expected=generate(1,options).tokens;
  require(ar.tokens==expected && br.tokens==expected,"snapshot lost recurrent or sampling history");
  require(model->suspensions && model->restorations && br.timing.preemptions,"combined pressure did not suspend/resume");
  require(model->pool.usage(cc::Memory::host).live==0,"private state leaked");
}
int main(int argc,char** argv) {
  try {
    const std::string mode=argc>1 ? argv[1] : "compare";
    if(mode=="ahead" || mode=="cancel") dependency(mode=="cancel");
    else if(mode=="recovery") recovery();
    else if(mode=="immovable") immovable();
    else if(mode=="stale" || mode=="commit" || mode=="quarantine") fault_case(mode);
    else {
      for(bool varying:{false,true}) for(std::uint32_t output:{1,2,3,9,20}) {
        cc::RequestOptions options; options.sampling.temperature=.8; options.sampling.seed=77;
        options.sampling.frequency_penalty=.5; options.sampling.repetition_penalty=1.3;
        const auto baseline=generate(1,options,output,varying);
        const auto ahead=generate(3,options,output,varying);
        require(ahead.tokens==baseline.tokens && ahead.tokens.size()==output,"step grant terminated request or changed seeded output");
        options.stop_tokens=std::vector<cc::Token>{baseline.tokens[output/2]};
        require(generate(3,options,output,varying).tokens==generate(1,options,output,varying).tokens,"token stop crossed step incorrectly");
        options.stop_tokens=std::vector<cc::Token>{};
        if(output>=3) {
          options.stop_strings={std::string{char('a'+baseline.tokens[1]),char('a'+baseline.tokens[2])}};
          const auto plain=generate(1,options,output,varying), pipelined=generate(3,options,output,varying);
          require(plain.tokens==pipelined.tokens && plain.text==pipelined.text && plain.finish==pipelined.finish,"text stop crossed step incorrectly");
        }
      }
      auto observed=std::make_shared<Observe>();
      cc::Engine engine(profile(3),std::make_shared<Backend>(),[&](const auto&,cc::Services services) {
        return std::make_unique<Model>(services.resources,32,observed);
      });
      cc::RequestOptions options; options.context=std::make_shared<HostConstraint>();
      auto request=engine.submit({1,2},9,options); require(request.wait().tokens.size()==9,"fallback request failed");
      require(observed->peak==1,"request-specific fallback executed ahead");
    }
    std::cout<<mode<<" passed\n";
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
