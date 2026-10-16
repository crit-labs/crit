#pragma once
#include <crit/core.h>
#include <chrono>
#include <functional>
#include <future>
#include <stdexcept>
#include <utility>

namespace crit::techniques {
struct OwnedBatchRow {
  crit::core::RequestKey key;
  std::uint32_t prompt_tokens=0;
  bool prefill_complete=false;
  std::vector<crit::core::Token> tokens;
  std::uint32_t reused_prompt_tokens=0, token_work=0;
};
// A real CPU task adapter for Core's optional batch pipeline. Device adapters
// implement BatchExecution with their own streams/events and physical owners.
class HostBatch final : public crit::core::BatchExecution {
public:
  using Work=std::function<std::vector<OwnedBatchRow>(std::span<const crit::core::WorkItem>)>;
  using Commit=std::function<void(std::span<const crit::core::CommitDecision>)>;
private:
  std::vector<crit::core::WorkItem> items_;
  Work work_;
  Commit commit_;
  std::function<void()> stop_;
  std::vector<std::shared_ptr<void>> owners_;
  std::future<std::vector<OwnedBatchRow>> future_;
  std::future<void> worker_;
  std::vector<OwnedBatchRow> rows_;
  bool submitted_=false, materialized_=false, committed_=false;
public:
  HostBatch(std::span<const crit::core::WorkItem> items, Work work, Commit commit,
            std::function<void()> stop={}, std::vector<std::shared_ptr<void>> owners={})
      :items_(items.begin(),items.end()),work_(std::move(work)),commit_(std::move(commit)),
       stop_(std::move(stop)),owners_(std::move(owners)) {
    if(items.empty() || items.size()>crit::core::max_request_concurrency || !work_ || !commit_)
      throw std::invalid_argument("Bounded batch membership, work and commit required");
  }
  ~HostBatch() override {
    // Join before member destruction, including on standalone use outside Core.
    if(worker_.valid())worker_.wait();
  }
  void submit(Notify notify) override {
    if(submitted_)throw std::logic_error("Batch already submitted");
    if(!notify)throw std::invalid_argument("Completion notification required");
    submitted_=true;
    std::promise<std::vector<OwnedBatchRow>> result;
    future_=result.get_future();
    worker_=std::async(std::launch::async,[this,result=std::move(result),notify=std::move(notify)]() mutable {
      try { result.set_value(work_(items_)); }
      catch(...) { result.set_exception(std::current_exception()); }
      // Publish readiness before waking Core. The worker future itself is not
      // ready until this callable returns, so it cannot serve both purposes.
      notify();
    });
  }
  bool ready() override {
    if(!submitted_)throw std::logic_error("Batch not submitted");
    return materialized_ || (future_.valid() && future_.wait_for(std::chrono::seconds(0))==std::future_status::ready);
  }
  std::vector<crit::core::RowProgress> progress() override {
    if(!ready())throw std::logic_error("Batch progress is pending");
    if(!materialized_) {rows_=future_.get();materialized_=true;}
    std::vector<crit::core::RowProgress> result;result.reserve(rows_.size());
    for(const auto& row:rows_)result.push_back({row.key,{row.prompt_tokens,row.prefill_complete,row.tokens,
        nullptr,false,row.reused_prompt_tokens,row.token_work}});
    return result;
  }
  void commit(std::span<const crit::core::CommitDecision> decisions) override {
    if(!materialized_ || committed_)throw std::logic_error("Batch cannot commit");
    committed_=true;commit_(decisions);
  }
  void quiesce() override {
    if(stop_)stop_();
    if(worker_.valid())worker_.wait();
  }
};
} // namespace crit::techniques
