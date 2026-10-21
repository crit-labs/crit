#pragma once
#include "batch_engine.h"
#include <crit/native_engine.h>
#include <functional>

namespace crit::examples::attention {
struct Pressure final : cc::MemoryPressure {};
struct Image final : cc::ResumeImage {
  cc::Token last;
  cc::Reservation storage;
  Image(cc::Token token, cc::Resources& resources) : last(token), storage(resources.reserve(cc::Memory::host,sizeof(Image))) {}
};
struct Suspend final : cc::Suspension {
  std::unique_ptr<Image> image;
  std::function<void()> release;
  bool done=false;
  void submit(cc::BatchExecution::Notify notify) override { release(); done=true; notify(); }
  bool ready() override { return done; }
  std::unique_ptr<cc::ResumeImage> take() override { return std::move(image); }
  void quiesce() override {}
};
struct Restore final : cc::Restoration {
  std::unique_ptr<Session> session;
  std::function<void(Session&)> replay;
  std::uint64_t recomputed=0;
  bool done=false;
  void submit(cc::BatchExecution::Notify notify) override { replay(*session); done=true; notify(); }
  bool ready() override { return done; }
  std::unique_ptr<cc::ConcurrentSession> take() override { return std::move(session); }
  std::uint64_t recomputed_tokens() const noexcept override { return recomputed; }
  void quiesce() override {}
};
// Synchronous CPU arithmetic implements the async ownership protocol too; the
// runtime's suspension/restore machinery does not depend on device events.
struct Operation final : cc::BatchExecution {
  Model& model;
  std::vector<cc::WorkItem> items;
  std::vector<cc::RowProgress> rows;
  bool done=false;
  Operation(Model& m,std::span<const cc::WorkItem> work) : model(m),items(work.begin(),work.end()) {}
  void submit(Notify notify) override { rows=model.execute(items); done=true; notify(); }
  bool ready() override { return done; }
  std::vector<cc::RowProgress> progress() override { return rows; }
  void commit(std::span<const cc::CommitDecision> decisions) override { model.commit(decisions); }
  void quiesce() override {}
};
struct RecoverableModel : Model {
  cc::Resources& resources;
  std::uint32_t physical_tokens;
  std::size_t suspensions=0, released_pages=0, restorations=0;
  RecoverableModel(std::shared_ptr<Stats> s, cc::Resources& r, cc::StateCacheConfig config)
      : Model(std::move(s),r,config),resources(r),physical_tokens(config.pages*config.page_tokens) {}
  cc::ModelInfo info() const override {
    auto result=Model::info(); result.async_batches=result.memory_preemption=true; return result;
  }
  cc::Admission admission(const cc::RequestView& view) override {
    if(view.prompt.size()+view.output_tokens-1>physical_tokens) return cc::Admission::infeasible;
    return Model::admission(view);
  }
  std::unique_ptr<cc::BatchExecution> prepare_batch(std::span<const cc::WorkItem> items) override {
    return std::make_unique<Operation>(*this,items);
  }
  cc::BatchPreparation prepare_incremental(std::span<const cc::WorkItem> items) override {
    for(const auto& item:items) {
      auto& session=static_cast<Session&>(*item.session);
      session.round=cache.try_begin(session.sequence,item.budget.prefill_complete ? 1 : item.budget.prefill_tokens);
      if(!session.round) {
        for(const auto& prior:items) {
          auto& s=static_cast<Session&>(*prior.session);
          if(s.round) { cache.discard(s.sequence,s.round->id); s.round.reset(); }
        }
        cache.retire(cache.epoch());
        return {cc::Admission::deferred,{},item.key,std::make_shared<Pressure>()};
      }
    }
    return {cc::Admission::ready,prepare_batch(items),{}, {}};
  }
  bool reclaimable(const cc::MemoryPressure& pressure,const cc::ContinuationView&,const cc::ConcurrentSession& s) override {
    return dynamic_cast<const Pressure*>(&pressure) && cache.committed(static_cast<const Session&>(s).sequence)>0;
  }
  std::unique_ptr<cc::Suspension> suspend(const cc::ContinuationView&,cc::ConcurrentSession& raw) override {
    auto& session=static_cast<Session&>(raw);
    auto operation=std::make_unique<Suspend>();
    operation->image=std::make_unique<Image>(session.last,resources);
    operation->release=[this,&session] {
      const auto before=cache.stats().free_pages;
      cache.close(session.sequence); cache.retire(cache.epoch());
      released_pages+=cache.stats().free_pages-before; ++suspensions;
    };
    return operation;
  }
  std::unique_ptr<cc::Restoration> restore(const cc::ContinuationView& view,const cc::ResumeImage& raw) override {
    const auto& image=dynamic_cast<const Image&>(raw);
    auto operation=std::make_unique<Restore>();
    operation->session=std::make_unique<Session>(view.request,cache);
    auto& session=*operation->session;
    session.last=image.last;
    const auto computed=view.prompt_tokens+(view.accepted.empty() ? 0 : std::uint32_t(view.accepted.size()-1));
    // Reserve the replay frontier and first useful new row before any writes.
    session.round=cache.begin(session.sequence,computed+1);
    operation->recomputed=computed;
    operation->replay=[this,view,computed](Session& s) {
      s.inputs.assign(view.request.prompt.begin(),view.request.prompt.begin()+view.prompt_tokens);
      if(!view.accepted.empty()) s.inputs.insert(s.inputs.end(),view.accepted.begin(),view.accepted.end()-1);
      for(std::uint32_t i=0;i<computed;++i) pages.write(s.round->kv[i],key(s.inputs[i]),value(s.inputs[i]));
      cache.commit(s.sequence,s.round->id,s.inputs); s.round.reset();
      cache.retire(cache.epoch()); ++restorations;
    };
    return operation;
  }
};
inline cc::StateCacheConfig recovery_config(std::uint32_t pages=12) {
  auto config=cache_config(); config.pages=pages; config.page_tokens=1;
  config.max_tokens=pages; config.max_round_rows=pages;
  config.prefix_cache=false; return config;
}
} // namespace crit::examples::attention
