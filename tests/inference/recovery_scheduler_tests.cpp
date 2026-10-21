#include "recoverable_engine.h"
#include <iostream>
#include <future>
#include <thread>
#include <mutex>
using namespace crit::examples::attention;
struct Control {
  std::atomic_bool suspended{false}, restoring{false}, release{false};
  std::string mode;
  std::mutex mutex;
  cc::BatchExecution::Notify notify;
  void wake() { std::lock_guard lock(mutex); if(notify) notify(); }
};
struct GatedOperation final : cc::BatchExecution {
  std::unique_ptr<cc::BatchExecution> inner;
  std::shared_ptr<Control> control;
  void submit(Notify notify) override {
    { std::lock_guard lock(control->mutex); control->notify=notify; }
    inner->submit(std::move(notify));
  }
  bool ready() override { return (!control->suspended || control->release) && inner->ready(); }
  std::vector<cc::RowProgress> progress() override { return inner->progress(); }
  void commit(std::span<const cc::CommitDecision> d) override { inner->commit(d); }
  void quiesce() override { inner->quiesce(); }
};
struct ObservedSuspend final : cc::Suspension {
  std::unique_ptr<cc::Suspension> inner;
  std::shared_ptr<Control> control;
  void submit(cc::BatchExecution::Notify notify) override {
    inner->submit(std::move(notify));
    if(control->mode=="suspend_failure") throw std::runtime_error("injected suspension failure");
  }
  bool ready() override { return inner->ready(); }
  std::unique_ptr<cc::ResumeImage> take() override { control->suspended=true; return inner->take(); }
  void quiesce() override { inner->quiesce(); }
};
struct GatedRestore final : cc::Restoration {
  std::unique_ptr<cc::Restoration> inner;
  std::shared_ptr<Control> control;
  void submit(cc::BatchExecution::Notify notify) override {
    inner->submit(std::move(notify)); control->restoring=true;
    if(control->mode=="restore_failure" || control->mode=="quarantine") throw std::runtime_error("injected restoration failure");
  }
  bool ready() override { return control->release && inner->ready(); }
  std::unique_ptr<cc::ConcurrentSession> take() override { return inner->take(); }
  std::uint64_t recomputed_tokens() const noexcept override { return inner->recomputed_tokens(); }
  void quiesce() override {
    inner->quiesce();
    if(control->mode=="quarantine") throw std::runtime_error("unsettled restoration");
  }
};
struct ControlledModel final : RecoverableModel {
  std::shared_ptr<Control> control;
  ControlledModel(std::shared_ptr<Stats> s,cc::Resources& r,std::shared_ptr<Control> c)
      : RecoverableModel(std::move(s),r,recovery_config()),control(std::move(c)) {}
  std::unique_ptr<cc::BatchExecution> prepare_batch(std::span<const cc::WorkItem> items) override {
    auto inner=RecoverableModel::prepare_batch(items);
    if(control->mode!="cancel_suspended") return inner;
    auto operation=std::make_unique<GatedOperation>(); operation->inner=std::move(inner); operation->control=control;
    return operation;
  }
  std::unique_ptr<cc::Suspension> suspend(const cc::ContinuationView& view,cc::ConcurrentSession& session) override {
    auto operation=std::make_unique<ObservedSuspend>(); operation->control=control;
    operation->inner=RecoverableModel::suspend(view,session); return operation;
  }
  std::unique_ptr<cc::Restoration> restore(const cc::ContinuationView& view,const cc::ResumeImage& image) override {
    auto operation=std::make_unique<GatedRestore>(); operation->control=control;
    operation->inner=RecoverableModel::restore(view,image); return operation;
  }
};
void await(const std::atomic_bool& flag) {
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while(!flag) {
    require(std::chrono::steady_clock::now()<deadline,"test control did not reach its gate");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
void controlled(const std::string& mode) {
  auto stats=std::make_shared<Stats>(); auto control=std::make_shared<Control>(); control->mode=mode;
  cc::Profile profile{64,2,8*1024*1024,1024,2,2};
  profile.max_batch_tokens=4; profile.max_inflight_batches=2; profile.memory_preemption=true;
  const auto quarantined=cc::quarantined_engines();
  {
    cc::Engine engine(profile,std::make_shared<Backend>(),[&](const auto&,cc::Services services) {
      return std::make_unique<ControlledModel>(stats,services.resources,control);
    });
    auto first=engine.submit({1,2,3,4,5,6},5), second=engine.submit({6,5,4,3,2,1},5);
    stats->start=true;
    if(mode=="cancel_suspended" || mode=="cancel_restore") {
      await(mode=="cancel_suspended" ? control->suspended : control->restoring);
      second.cancel(); control->release=true; control->wake();
      const auto result=second.wait();
      require(result.finish==cc::Finish::cancelled,"recovery cancellation failed");
      require(result.tokens.size()<5,"cancelled recovery published a suffix");
      if(mode=="cancel_suspended") require(!control->restoring,"cancelled suspended request was needlessly restored");
      require(first.wait().tokens==dense({1,2,3,4,5,6},5),"cancellation damaged peer");
    } else {
      control->release=true;
      bool failed=false;
      try { first.wait(); second.wait(); } catch(const std::runtime_error&) { failed=true; }
      require(failed,"recovery failure did not propagate");
    }
  }
  require(cc::quarantined_engines()==quarantined+(mode=="quarantine" ? 1 : 0),"recovery ownership quarantine mismatch");
}
int main(int argc,char** argv) {
  try {
    if(argc>1) { controlled(argv[1]); return 0; }
    auto stats=std::make_shared<Stats>();
    cc::Profile profile{64,2,8*1024*1024,1024,2,4};
    profile.max_batch_tokens=4; profile.max_inflight_batches=2; profile.memory_preemption=true;
    RecoverableModel* model=nullptr;
    cc::Engine engine(profile,std::make_shared<Backend>(),[&](const auto&,cc::Services services) {
      auto result=std::make_unique<RecoverableModel>(stats,services.resources,recovery_config());
      model=result.get(); return result;
    });
    const std::vector<cc::Token> a{1,2,3,4,5,6}, b{6,5,4,3,2,1};
    auto first=engine.submit(a,5), second=engine.submit(b,5);
    stats->start=true;
    const auto ar=first.wait(), br=second.wait();
    require(ar.tokens==dense(a,5) && br.tokens==dense(b,5),"replay changed attention output");
    require(ar.request_id!=br.request_id && br.request_id==second.id(),"resume changed request identity");
    require(br.timing.preemptions>0 && br.timing.recomputed_tokens>0 && br.timing.recovery_ns>0,"recovery was not exercised");
    require(ar.timing.prefill_tokens==a.size() && br.timing.prefill_tokens==b.size(),"replay inflated prompt usage");
    require(model->suspensions>0 && model->released_pages>0 && model->restorations>0,"physical pages were not reused");
    require(model->cache.stats().live_pages==0 && model->cache.stats().free_pages==12,"pages leaked after recovery");
    // Requests exceeding solo capacity fail individually; a later feasible request still works.
    bool rejected=false;
    try { auto impossible=engine.submit(std::vector<cc::Token>(10,1),5); impossible.wait(); }
    catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"solo infeasibility was not rejected");
    auto next=engine.submit(a,3);
    require(next.wait().tokens==dense(a,3),"engine did not survive infeasible request");
    std::cout<<"real attention pages reclaimed and replayed without duplicate output or usage\n";
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
