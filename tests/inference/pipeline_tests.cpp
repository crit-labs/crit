#include "host_batch.h"
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
using namespace crit::core;
using crit::techniques::HostBatch;
using crit::techniques::OwnedBatchRow;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
struct State {
  std::mutex mutex;std::condition_variable cv;
  bool entered=false,released=false;
  std::atomic<int> readers{0},closed{0},unsafe{0},peak{0},destroyed_batches{0};
  std::string fault;
  void release(){std::lock_guard lock(mutex);released=true;cv.notify_all();}
  void observe(){std::unique_lock lock(mutex);require(cv.wait_for(lock,std::chrono::seconds(2),[&]{return entered;}),"No async work");}
};
struct B:Backend {void bind_thread()override{} void drain()override{}};
struct S:ConcurrentSession {
  std::shared_ptr<State> state;Token token;std::uint32_t frontier=0,generated=0,pending=0;
  std::atomic<bool> active{false};Reservation bytes;
  S(std::shared_ptr<State> s,Token t,Resources& r):state(std::move(s)),token(t),bytes(r.reserve(Memory::host,64)){}
  void close(Finish)noexcept override {if(active)++state->unsafe;++state->closed;}
};
struct Done:Retirement {bool ready()override{return true;}};
struct FaultBatch:BatchExecution {
  std::unique_ptr<HostBatch> inner;std::shared_ptr<State> state;
  FaultBatch(std::unique_ptr<HostBatch> b,std::shared_ptr<State>s):inner(std::move(b)),state(std::move(s)){}
  ~FaultBatch()override{++state->destroyed_batches;}
  void submit(Notify notify)override{inner->submit(std::move(notify));if(state->fault=="submit" || state->fault=="quiesce")throw std::runtime_error("submit failure");}
  bool ready()override{return state->fault=="timeout" ? false : inner->ready();}
  std::vector<RowProgress> progress()override{return inner->progress();}
  void commit(std::span<const CommitDecision>d)override{if(state->fault=="commit")throw std::runtime_error("commit failure");inner->commit(d);}
  void quiesce()override{inner->quiesce();if(state->fault=="quiesce")throw std::runtime_error("ambiguous quiesce");}
};
struct M:ConcurrentModel {
  std::shared_ptr<State> state;Resources& resources;
  M(std::shared_ptr<State>s,Resources&r):state(std::move(s)),resources(r){}
  ModelInfo info()const override {ModelInfo m{"portable.async.fixture",128,32,8,{},true,false,4};m.mixed_batches=true;m.async_batches=true;return m;}
  std::unique_ptr<ConcurrentSession> open(const RequestView&v)override{return std::make_unique<S>(state,v.prompt.front(),resources);}
  RowProgress prefill(const WorkItem&)override{throw std::logic_error("legacy callback");}
  std::vector<RowProgress> decode(std::span<const WorkItem>)override{throw std::logic_error("legacy callback");}
  void commit(std::span<const CommitDecision>)override{throw std::logic_error("global commit used for private operation");}
  std::unique_ptr<BatchExecution> prepare_batch(std::span<const WorkItem> items)override {
    if(state->fault=="prepare")throw std::runtime_error("prepare failure");
    auto work=[state=state](std::span<const WorkItem> input) {
      if(state->fault=="work")throw std::runtime_error("producer failure");
      std::vector<OwnedBatchRow> rows;
      for(const auto& item:input) {
        auto& s=static_cast<S&>(*item.session);
        require(!s.active.exchange(true),"Same lane submitted twice");
        const auto readers=++state->readers;auto peak=state->peak.load();
        while(peak<readers && !state->peak.compare_exchange_weak(peak,readers)){}
        if(s.token==1 && !s.generated) {
          std::unique_lock lock(state->mutex);state->entered=true;state->cv.notify_all();
          if(!state->cv.wait_for(lock,std::chrono::seconds(3),[&]{return state->released;})) {
            s.active=false;--state->readers;throw std::runtime_error("test gate timeout");
          }
        }
        s.pending=item.budget.prefill_complete ? 0 : item.budget.prefill_tokens;
        auto key=item.key;if(state->fault=="stale")++key.generation;
        rows.push_back({key,s.pending,true,{s.token+1+Token(s.generated)},0,1});
        s.active=false;--state->readers;
      }
      return rows;
    };
    auto commit=[](std::span<const CommitDecision>d){for(const auto& row:d){auto& s=static_cast<S&>(*row.session);s.frontier+=s.pending;s.generated+=row.accepted;}};
    auto batch=std::make_unique<HostBatch>(items,work,commit,[state=state]{state->release();});
    return std::make_unique<FaultBatch>(std::move(batch),state);
  }
  std::unique_ptr<Retirement> retire(RequestKey,ConcurrentSession& session,Finish)override {
    require(!static_cast<S&>(session).active,"Retired live reader");return std::make_unique<Done>();
  }
  void quiesce()override{require(!state->readers,"Model quiesced before batch readers");}
  std::vector<Token> tokenize(std::string_view)override{return {1};}
  std::string detokenize(std::span<const Token>)override{return {};}
};
Profile profile(){Profile p{32,1,1<<20,1<<20,3,4,3000,3000};p.max_batch_tokens=1;p.max_inflight_batches=2;p.trace={2048,1000};return p;}

void verify_steps(const TraceSnapshot& trace,bool failed=false) {
  struct Step {std::uint64_t operation,started;bool committed=false,published=false;};
  std::map<std::uint64_t,Step> active;
  std::set<std::uint64_t> slow;
  unsigned ended=0,failures=0;
  require(!trace.dropped && !trace.overwritten,"Incomplete test trace");
  for(const auto& event:trace.events) {
    if(event.phase==TracePhase::step && event.mark==TraceMark::begin) {
      require(active.emplace(event.key.request,Step{event.operation,event.timestamp_ns}).second,"Overlapping steps for one request");
    } else if(event.phase==TracePhase::commit && event.mark==TraceMark::end) {
      require(active.contains(event.key.request),"Commit outside step");
      active.at(event.key.request).committed=true;
    } else if(event.phase==TracePhase::publication) {
      require(active.contains(event.key.request),"Publication outside step");
      active.at(event.key.request).published=true;
    } else if(event.phase==TracePhase::step) {
      require(active.contains(event.key.request),"Duplicate step end");
      const auto step=active.at(event.key.request);
      require(step.operation==event.operation && event.duration_ns==event.timestamp_ns-step.started,"Step identity or duration changed");
      if(event.mark==TraceMark::failed)++failures;
      else require(event.mark==TraceMark::end && step.committed && step.published,"Step ended before commit/publication");
      if(event.slow)slow.insert(event.operation);
      active.erase(event.key.request);++ended;
    }
  }
  require(ended && active.empty(),"Unclosed async step");
  require((failures!=0)==failed,"Wrong failed-step classification");
  require(trace.slow_steps==slow.size(),"Slow steps counted other than once per operation");
}

// Exercise notification races without timer-resolution-dependent speed limits.
// Completion comes from the test thread, or inside submit/the first ready poll.
struct WakeState {
  std::mutex mutex;std::condition_variable cv;
  BatchExecution::Notify notify;
  bool completed=false;unsigned polls=0;
  void complete() {
    BatchExecution::Notify wake;
    {std::lock_guard lock(mutex);completed=true;wake=notify;}
    wake();wake(); // Repeated notifications must be harmless.
  }
};
struct WakeBatch:BatchExecution {
  std::shared_ptr<WakeState> state;std::string mode;WorkItem item;Token token=2;
  WakeBatch(std::shared_ptr<WakeState>s,std::string m,WorkItem i):state(std::move(s)),mode(std::move(m)),item(i){}
  void submit(Notify notify)override {
    {std::lock_guard lock(state->mutex);state->notify=std::move(notify);}
    if(mode=="immediate")state->complete();
  }
  bool ready()override {
    bool first,done;
    {std::lock_guard lock(state->mutex);first=++state->polls==1;done=state->completed;state->cv.notify_all();}
    if(mode=="racing" && first) {state->complete();return false;}
    return done;
  }
  std::vector<RowProgress> progress()override {
    return {{item.key,{item.budget.prefill_tokens,true,std::span<const Token>(&token,1),nullptr,false,0,1}}};
  }
  void commit(std::span<const CommitDecision>)override{}
  void quiesce()override{}
};
struct WakeModel:M {
  std::shared_ptr<WakeState> wake;std::string mode;
  WakeModel(std::shared_ptr<State>s,Resources&r,std::shared_ptr<WakeState>w,std::string m)
      :M(std::move(s),r),wake(std::move(w)),mode(std::move(m)){}
  std::unique_ptr<BatchExecution> prepare_batch(std::span<const WorkItem> items)override {
    require(items.size()==1,"Wake fixture membership");return std::make_unique<WakeBatch>(wake,mode,items[0]);
  }
};
void wakeup_tests() {
  for(const std::string mode:{"immediate","racing","delayed"}) {
    auto state=std::make_shared<State>();auto wake=std::make_shared<WakeState>();
    Engine engine(profile(),std::make_shared<B>(),[&](const Profile&,Services svc){
      return std::make_unique<WakeModel>(state,svc.resources,wake,mode);
    });
    auto request=engine.submit({1},1);
    bool polled_while_idle=false;
    if(mode=="delayed") {
      {
        std::unique_lock lock(wake->mutex);
        require(wake->cv.wait_for(lock,std::chrono::seconds(2),[&]{return wake->polls!=0;}),"No initial readiness poll");
        polled_while_idle=wake->cv.wait_for(lock,std::chrono::milliseconds(50),[&]{return wake->polls>1;});
      }
      for(const auto& event:engine.trace().events)
        require(event.phase!=TracePhase::step || event.mark==TraceMark::begin,"Pending step already ended");
      wake->complete();
    }
    require(request.wait().tokens==std::vector<Token>({2}),"Lost completion notification");
    require(!polled_while_idle,"Pending batch used timer polling instead of notification");
    const auto trace=engine.trace();verify_steps(trace);
    if(mode=="delayed") {
      require(trace.slow_steps==1,"Delayed completion omitted from slow steps");
      for(const auto& event:trace.events)if(event.phase==TracePhase::step && event.mark==TraceMark::end)
        require(event.duration_ns>=40000000,"Step omitted pending execution time");
    }
  }
}
int main(){try {
  wakeup_tests();
  for(bool cancel:{false,true}) {
    auto s=std::make_shared<State>();
    Engine e(profile(),std::make_shared<B>(),[s](const Profile&,Services svc){return std::make_unique<M>(s,svc.resources);});
    auto a=e.submit({1},2);s->observe();auto b=e.submit({10},2);
    auto result=b.wait(); // Must finish while A's producer is still blocked.
    require(result.tokens==std::vector<Token>({11,12}),"Independent batch result");
    require(s->readers==1 && s->peak>=2,"No actual execution overlap");
    if(cancel)a.cancel();
    s->release();auto first=a.wait();
    require(cancel ? first.tokens.empty() && first.finish==Finish::cancelled : first.tokens==std::vector<Token>({2,3}),"Ordered/cancelled output");
    auto c=e.submit({20},2);require(c.wait().tokens==std::vector<Token>({21,22}),"Lane refill failed");
    require(!s->unsafe,"Closed an active session");
    verify_steps(e.trace());
  }
  for(const std::string fault:{"prepare","submit","work","commit","stale","timeout"}) {
    auto s=std::make_shared<State>();s->fault=fault;
    auto p=profile();p.batch_timeout_ms=10;
    Engine e(p,std::make_shared<B>(),[s](const Profile&,Services svc){return std::make_unique<M>(s,svc.resources);});
    auto a=e.submit({1},2);
    if(fault=="commit" || fault=="stale")s->release();
    bool failed=false;try{a.wait();}catch(const std::exception&){failed=true;}
    require(failed && !s->readers && !s->unsafe && s->closed==1,"Failure released live batch ownership");
    verify_steps(e.trace(),true);
  }
  {
    const auto before=quarantined_engines();
    auto s=std::make_shared<State>();s->fault="quiesce";
    {
      Engine e(profile(),std::make_shared<B>(),[s](const Profile&,Services svc){return std::make_unique<M>(s,svc.resources);});
      auto request=e.submit({1},2);bool failed=false;try{request.wait();}catch(const std::exception&){failed=true;}
      require(failed && !s->destroyed_batches && !s->closed,"Unknown operation settlement released owners");
      verify_steps(e.trace(),true);
    }
    require(quarantined_engines()==before+1 && !s->destroyed_batches && !s->closed,"Pipeline quarantine lost owners");
  }
  std::cout<<"Portable pipeline overlap, cancellation, refill and failure tests passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
