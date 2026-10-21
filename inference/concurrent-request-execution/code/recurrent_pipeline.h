#pragma once
#include <crit/core.h>
#include <crit/sampling.h>
#include <array>
#include <future>
#include <algorithm>
#include <stdexcept>

// Portable arithmetic reference, not a checkpoint implementation. Each dependent
// task owns a complete private recurrent/sampling state; public commit never edits
// state a successor is reading. The constrained pool owns real state allocations.
namespace crit::examples::pipeline {
namespace cc=crit::core;
struct Hooks {
  virtual ~Hooks()=default;
  virtual void computed(cc::StepKey) {}
  virtual void submitted(cc::BatchExecution::Notify) {}
  virtual void approved(cc::StepKey,bool) {}
  virtual bool publishable(cc::StepKey) { return true; }
  virtual void quiesce() {}
};
struct State {
  std::uint64_t recurrent=0;
  std::uint32_t prompt=0, generated=0;
  cc::Token pending=0;
  cc::TokenHistory history{std::span<const cc::Token>{}};
  cc::Reservation allocation, history_bytes;
  void copy(const State& other,cc::Resources& resources) {
    recurrent=other.recurrent; prompt=other.prompt; generated=other.generated; pending=other.pending;
    history=other.history;
    if(!history_bytes.bytes()) history_bytes=resources.reserve(cc::Memory::host,cc::TokenHistory::host_bytes(64,64));
  }
};
struct Carry final : cc::StepContinuation { std::shared_future<std::shared_ptr<State>> value; };
struct Session final : cc::ConcurrentSession {
  cc::RequestView view;
  cc::SamplingParams sampling;
  std::shared_ptr<State> approved;
  void close(cc::Finish) noexcept override { approved.reset(); }
};
struct Receipt final : cc::Retirement { bool ready() override { return true; } };
struct Pressure final : cc::MemoryPressure {};
struct Image final : cc::ResumeImage {
  State state;
  cc::Reservation bytes;
};
struct Suspend final : cc::Suspension {
  Session& session;
  std::unique_ptr<Image> image;
  bool done=false;
  explicit Suspend(Session& s):session(s) {}
  void submit(cc::BatchExecution::Notify notify) override { session.approved.reset(); done=true; notify(); }
  bool ready() override { return done; }
  std::unique_ptr<cc::ResumeImage> take() override { return std::move(image); }
  void quiesce() override {}
};
struct Restore final : cc::Restoration {
  std::unique_ptr<Session> session;
  bool done=false;
  void submit(cc::BatchExecution::Notify notify) override { done=true; notify(); }
  bool ready() override { return done; }
  std::unique_ptr<cc::ConcurrentSession> take() override { return std::move(session); }
  std::uint64_t recomputed_tokens() const noexcept override { return 0; }
  void quiesce() override {}
};
struct Batch final : cc::BatchExecution {
  struct Row {
    cc::WorkItem item;
    std::shared_ptr<State> initial, state;
    std::shared_ptr<Carry> carry=std::make_shared<Carry>();
    std::promise<std::shared_ptr<State>> result;
    std::vector<cc::Token> tokens;
    std::uint32_t prompt=0;
    bool prefilled=false;
  };
  cc::Resources& resources;
  std::shared_ptr<Hooks> hooks;
  std::vector<Row> rows;
  std::future<void> task;
  std::shared_future<void> settled;
  cc::Reservation scratch;
  bool variable;
  Batch(cc::Resources& r,std::shared_ptr<Hooks> h,bool varying):resources(r),hooks(std::move(h)),
      scratch(r.reserve(cc::Memory::host,cc::Sampler::scratch_bytes(16))),variable(varying) {}
  ~Batch() override { if(task.valid())task.wait(); }
  void submit(Notify notify) override {
    hooks->submitted(notify);
    std::promise<void> completion; settled=completion.get_future().share();
    task=std::async(std::launch::async,[this,completion=std::move(completion),notify=std::move(notify)]() mutable {
      try {
        cc::Sampler sampler(16);
        for(auto& row:rows) {
          const auto& item=row.item;
          const auto& session=static_cast<const Session&>(*item.session);
          if(item.predecessor) {
            const auto parent=std::dynamic_pointer_cast<Carry>(item.predecessor);
            if(!parent) throw std::runtime_error("invalid recurrent predecessor");
            row.state->copy(*parent->value.get(),resources);
          } else if(row.initial) row.state->copy(*row.initial,resources);
          else {
            row.state->history=cc::TokenHistory(session.view.prompt);
          }
          auto& state=*row.state;
          const auto consume=[&](cc::Token token) { state.recurrent=(state.recurrent*33+std::uint32_t(token)+17)%65521; };
          if(!item.budget.prefill_complete) {
            row.prompt=std::min(item.budget.prefill_tokens,std::uint32_t(session.view.prompt.size())-state.prompt);
            for(std::uint32_t i=0;i<row.prompt;++i) consume(session.view.prompt[state.prompt++]);
          }
          row.prefilled=state.prompt==session.view.prompt.size();
          if(row.prefilled) {
            auto count=std::min(item.budget.output_tokens,item.budget.token_work);
            if(!item.budget.prefill_complete) count=1;
            if(variable) count=std::min(count,1+state.generated%2);
            for(std::uint32_t i=0;i<count;++i) {
              if(state.generated) consume(state.pending);
              std::array<float,16> logits{};
              for(std::uint32_t t=0;t<logits.size();++t) logits[t]=float((state.recurrent+(t+1)*(t+13))%97)/16.f;
              const auto token=sampler.sample(logits,session.sampling,state.history).token;
              state.pending=token; ++state.generated; row.tokens.push_back(token);
              state.history.stage(std::span<const cc::Token>(&token,1)); state.history.commit(1);
            }
          }
          row.result.set_value(row.state);
          hooks->computed({item.key,item.step});
        }
        completion.set_value();
      } catch(...) {
        const auto error=std::current_exception();
        for(auto& row:rows) try { row.result.set_exception(error); } catch(const std::future_error&) {}
        completion.set_exception(error);
      }
      notify();
    });
  }
  std::shared_ptr<cc::StepContinuation> continuation(cc::StepKey key) override {
    for(auto& row:rows) if(row.item.key==key.request && row.item.step==key.sequence) return row.carry;
    throw std::logic_error("unknown step continuation");
  }
  bool ready() override {
    if(settled.wait_for(std::chrono::seconds(0))!=std::future_status::ready) return false;
    for(const auto& row:rows) if(!hooks->publishable({row.item.key,row.item.step})) return false;
    return true;
  }
  std::vector<cc::RowProgress> progress() override {
    settled.get();
    std::vector<cc::RowProgress> result;
    for(const auto& row:rows) result.push_back({row.item.key,{row.prompt,row.prefilled,row.tokens,nullptr,false,0,
        row.item.budget.prefill_complete ? std::uint32_t(row.tokens.size()) : row.prompt},row.item.step});
    return result;
  }
  void commit(std::span<const cc::CommitDecision> decisions) override {
    for(const auto& decision:decisions) {
      hooks->approved({decision.key,decision.step},decision.discard);
      if(decision.discard || decision.cancelled || decision.terminal) continue;
      auto it=std::find_if(rows.begin(),rows.end(),[&](const auto& row){return row.item.key==decision.key && row.item.step==decision.step;});
      if(it==rows.end() || decision.accepted!=it->tokens.size()) throw std::runtime_error("private continuation was partially approved");
      static_cast<Session&>(*decision.session).approved=it->state;
    }
  }
  void quiesce() override { hooks->quiesce(); if(task.valid())task.wait(); }
};
struct Model : cc::ConcurrentModel {
  cc::Resources& resources;
  cc::Resources pool;
  cc::Reservation arena;
  std::shared_ptr<Hooks> hooks;
  bool variable;
  std::atomic_bool start{true};
  std::size_t suspensions=0, restorations=0;
  Model(cc::Resources& r,std::size_t states=16,std::shared_ptr<Hooks> h=std::make_shared<Hooks>(),bool varying=false)
      :resources(r),pool(states*sizeof(State),0),arena(r.reserve(cc::Memory::host,states*sizeof(State))),hooks(std::move(h)),variable(varying) {}
  cc::ModelInfo info() const override {
    cc::ModelInfo m{"synthetic.recurrent.pipeline",16,64,8,{},true,false,8};
    m.mixed_batches=m.async_batches=m.memory_preemption=true; m.max_request_steps=8; m.decode_token_work=4;
    m.sampling={true,true,true,true,true,true,true}; m.request_context=true; return m;
  }
  std::uint32_t execution_depth(const cc::RequestView& view) override { return view.options->context ? 1 : 8; }
  cc::Admission admission(const cc::RequestView&) override { return start ? cc::Admission::ready : cc::Admission::deferred; }
  std::unique_ptr<cc::ConcurrentSession> open(const cc::RequestView& view) override {
    auto session=std::make_unique<Session>(); session->view=view;
    cc::SamplingParams defaults; defaults.temperature=0;
    session->sampling=cc::resolve_sampling(view.options->sampling,defaults,16); return session;
  }
  std::shared_ptr<State> allocate() {
    auto reservation=pool.reserve(cc::Memory::host,sizeof(State));
    auto state=std::make_shared<State>(); state->allocation=std::move(reservation);
    state->history_bytes=resources.reserve(cc::Memory::host,cc::TokenHistory::host_bytes(64,64)); return state;
  }
  cc::BatchPreparation prepare_incremental(std::span<const cc::WorkItem> items) override {
    auto batch=std::make_unique<Batch>(resources,hooks,variable);
    for(const auto& item:items) {
      const auto usage=pool.usage(cc::Memory::host);
      if(usage.limit-usage.live<sizeof(State)) {
        batch.reset(); return {cc::Admission::deferred,{},item.key,std::make_shared<Pressure>()};
      }
      Batch::Row row; row.item=item;
      if(!item.predecessor) row.initial=static_cast<Session&>(*item.session).approved;
      row.state=allocate();
      row.carry->value=row.result.get_future().share(); batch->rows.push_back(std::move(row));
    }
    return {cc::Admission::ready,std::move(batch),{}, {}};
  }
  std::unique_ptr<cc::BatchExecution> prepare_batch(std::span<const cc::WorkItem> items) override {
    auto preparation=prepare_incremental(items);
    if(preparation.status!=cc::Admission::ready) throw std::runtime_error("reference state pool exhausted");
    return std::move(preparation.execution);
  }
  bool reclaimable(const cc::MemoryPressure& pressure,const cc::ContinuationView&,const cc::ConcurrentSession& raw) override {
    const auto& state=static_cast<const Session&>(raw).approved;
    return dynamic_cast<const Pressure*>(&pressure) && state && state.use_count()==1;
  }
  std::unique_ptr<cc::Suspension> suspend(const cc::ContinuationView&,cc::ConcurrentSession& raw) override {
    auto& session=static_cast<Session&>(raw); auto operation=std::make_unique<Suspend>(session);
    operation->image=std::make_unique<Image>(); operation->image->bytes=resources.reserve(cc::Memory::host,sizeof(Image));
    operation->image->state.copy(*session.approved,resources); ++suspensions; return operation;
  }
  std::unique_ptr<cc::Restoration> restore(const cc::ContinuationView& view,const cc::ResumeImage& raw) override {
    const auto& image=dynamic_cast<const Image&>(raw);
    const auto usage=pool.usage(cc::Memory::host);
    if(usage.limit-usage.live<2*sizeof(State)) throw std::runtime_error("recovery lacks first-step capacity");
    auto operation=std::make_unique<Restore>();
    auto base=open(view.request); operation->session.reset(static_cast<Session*>(base.release()));
    operation->session->approved=allocate(); operation->session->approved->copy(image.state,resources);
    ++restorations; return operation;
  }
  cc::RowProgress prefill(const cc::WorkItem&) override { throw std::logic_error("use mixed batches"); }
  std::vector<cc::RowProgress> decode(std::span<const cc::WorkItem>) override { throw std::logic_error("use mixed batches"); }
  void commit(std::span<const cc::CommitDecision>) override { throw std::logic_error("use private batch commit"); }
  std::unique_ptr<cc::Retirement> retire(cc::RequestKey,cc::ConcurrentSession& session,cc::Finish) override {
    session.close(cc::Finish::length); return std::make_unique<Receipt>();
  }
  void quiesce() override { hooks->quiesce(); }
};
} // namespace crit::examples::pipeline
