#include <crit/cuda.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>

// Synthetic recurrent arithmetic: demonstrates device-token dependencies, not an
// optimized model. A transfer gate makes predecessor output physically unreadable
// while its successor computes from the device carry on the compute stream.
namespace cc=crit::core;
void check(cudaError_t status) { if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status)); }
void require(bool ok,const char* why) { if(!ok)throw std::runtime_error(why); }
struct DeviceState { std::uint32_t recurrent; cc::Token token; };
__global__ void next_token(const DeviceState* input,DeviceState* output) {
  const auto state=(input->recurrent*33+std::uint32_t(input->token)+17)%65521;
  *output={state,cc::Token(state%32)};
}
struct Control {
  std::mutex mutex;
  std::condition_variable changed;
  bool gated=false,released=false,child=false;
  void hold() { std::unique_lock lock(mutex); changed.wait(lock,[&]{return released;}); }
  void release() { std::lock_guard lock(mutex); released=true; changed.notify_all(); }
  void computed(std::uint64_t step) { std::lock_guard lock(mutex); if(step==3)child=true; changed.notify_all(); }
  void observe() {
    std::unique_lock lock(mutex);
    require(changed.wait_for(lock,std::chrono::seconds(5),[&]{return child;}),"successor did not execute while readback was blocked");
    require(!released,"readback gate released early");
  }
};
void CUDART_CB hold_readback(void* value) { static_cast<Control*>(value)->hold(); }
struct Slot final : cc::StepContinuation {
  cc::CudaBuffer device,host;
  cc::CudaEvent produced,readback;
  std::mutex event_mutex;
  explicit Slot(cc::CudaBackend& backend):device(backend.allocate(cc::Memory::device,sizeof(DeviceState))),
      host(backend.allocate(cc::Memory::host,sizeof(DeviceState))),produced(backend.event()),readback(backend.event()) {}
};
struct Session final : cc::ConcurrentSession {
  std::shared_ptr<Slot> approved;
  void close(cc::Finish) noexcept override { approved.reset(); }
};
struct Batch final : cc::BatchExecution {
  cc::CudaBackend& backend;
  cc::WorkItem item;
  std::shared_ptr<Slot> input,output;
  std::shared_ptr<Control> control;
  std::future<void> producer;
  std::atomic_bool settled{false};
  std::exception_ptr error;
  Batch(cc::CudaBackend& b,cc::WorkItem work,std::shared_ptr<Slot> source,std::shared_ptr<Slot> target,std::shared_ptr<Control> c)
      :backend(b),item(std::move(work)),input(std::move(source)),output(std::move(target)),control(std::move(c)) {}
  ~Batch() override { if(producer.valid())producer.wait(); }
  void submit(Notify notify) override {
    if(item.predecessor) {
      std::lock_guard lock(input->event_mutex);
      input->produced.order(backend.stream()); // The producer recorded this before exposing its carry.
    }
    next_token<<<1,1,0,backend.stream()>>>(static_cast<const DeviceState*>(input->device.data()),static_cast<DeviceState*>(output->device.data()));
    check(cudaGetLastError());
    output->produced.record(backend.stream());
    output->produced.order(backend.stream(cc::Queue::transfer));
    if(control->gated && item.step==2) check(cudaLaunchHostFunc(backend.stream(cc::Queue::transfer),hold_readback,control.get()));
    backend.copy_async(output->host,0,output->device,0,sizeof(DeviceState),cc::Queue::transfer);
    output->readback.record(backend.stream(cc::Queue::transfer));
    producer=std::async(std::launch::async,[this,notify=std::move(notify)] {
      try {
        backend.bind_thread();
        // Observe compute independently of readback. Event API access is exclusive;
        // polling never holds the mutex while waiting for device completion.
        for(;;) {
          bool ready;
          { std::lock_guard lock(output->event_mutex); ready=output->produced.ready(); }
          if(ready)break;
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        control->computed(item.step);
        output->readback.wait();
      } catch(...) { error=std::current_exception(); }
      settled.store(true,std::memory_order_release); notify();
    });
  }
  std::shared_ptr<cc::StepContinuation> continuation(cc::StepKey key) override {
    require(key.request==item.key && key.sequence==item.step,"unknown CUDA step"); return output;
  }
  bool ready() override { return settled.load(std::memory_order_acquire); }
  std::vector<cc::RowProgress> progress() override {
    if(error)std::rethrow_exception(error);
    const auto* result=static_cast<const DeviceState*>(output->host.data());
    return {{item.key,{item.budget.prefill_complete ? 0u : 1u,true,{&result->token,1},nullptr,false,0,1},item.step}};
  }
  void commit(std::span<const cc::CommitDecision> decisions) override {
    for(const auto& d:decisions) if(!d.terminal && !d.discard && !d.cancelled) static_cast<Session&>(*d.session).approved=output;
  }
  void quiesce() override { control->release(); if(producer.valid())producer.wait(); }
};
struct Receipt final : cc::Retirement { bool ready() override { return true; } };
struct Model final : cc::ConcurrentModel {
  using cc::ConcurrentModel::open;
  cc::CudaBackend& backend;
  std::shared_ptr<Control> control;
  std::vector<std::shared_ptr<Slot>> slots;
  Model(cc::CudaBackend& b,const cc::Profile& p,std::shared_ptr<Control> c):backend(b),control(std::move(c)) {
    // Every slot has its own state, pinned readback and recorded events. No
    // per-step allocation can synchronize behind the deliberately held transfer.
    for(std::uint32_t i=0;i<p.max_request_steps+2;++i)slots.push_back(std::make_shared<Slot>(backend));
  }
  std::shared_ptr<Slot> free_slot() {
    for(const auto& slot:slots)if(slot.use_count()==1)return slot;
    throw std::runtime_error("CUDA continuation slots exhausted");
  }
  cc::ModelInfo info() const override {
    cc::ModelInfo m{"synthetic.cuda.dependent",32,64,8,{},true,false,1};
    m.mixed_batches=m.async_batches=true; m.max_request_steps=8; return m;
  }
  std::uint32_t execution_depth(const cc::RequestView&) override { return 8; }
  std::unique_ptr<cc::ConcurrentSession> open(const cc::RequestView& view) override {
    require(view.prompt.size()==1,"CUDA arithmetic fixture takes one seed token");
    auto session=std::make_unique<Session>(); session->approved=free_slot();
    const DeviceState seed{0,view.prompt.front()};
    session->approved->device.upload(std::as_bytes(std::span(&seed,1)));
    return session;
  }
  std::unique_ptr<cc::BatchExecution> prepare_batch(std::span<const cc::WorkItem> items) override {
    require(items.size()==1,"CUDA arithmetic fixture uses one lane"); const auto& item=items.front();
    auto input=item.predecessor ? std::dynamic_pointer_cast<Slot>(item.predecessor) : static_cast<Session&>(*item.session).approved;
    require(bool(input),"missing CUDA carry");
    return std::make_unique<Batch>(backend,item,std::move(input),free_slot(),control);
  }
  cc::RowProgress prefill(const cc::WorkItem&) override { throw std::logic_error("use async batches"); }
  std::vector<cc::RowProgress> decode(std::span<const cc::WorkItem>) override { throw std::logic_error("use async batches"); }
  void commit(std::span<const cc::CommitDecision>) override { throw std::logic_error("use operation commit"); }
  std::unique_ptr<cc::Retirement> retire(cc::RequestKey,cc::ConcurrentSession& session,cc::Finish finish) override {
    session.close(finish); return std::make_unique<Receipt>();
  }
  void quiesce() override { control->release(); }
};
int main() {
  try {
    for(bool cancel:{false,true}) {
      auto control=std::make_shared<Control>(); control->gated=true;
      auto backend=std::make_shared<cc::CudaBackend>();
      cc::Profile p{64,1,1<<20,1<<20}; p.max_batch_tokens=1; p.max_inflight_batches=3; p.max_request_steps=3;
      cc::Engine engine(p,backend,[&](const auto& settings,cc::Services){return std::make_unique<Model>(*backend,settings,control);});
      auto request=engine.submit({3},9);
      struct Release { std::shared_ptr<Control> c; ~Release(){c->release();} } release{control};
      control->observe(); // Step 3 ran on CUDA while step 2's copy was physically blocked.
      if(cancel)request.cancel();
      control->release(); const auto result=request.wait();
      if(cancel)require(result.finish==cc::Finish::cancelled && result.tokens.size()<9,"CUDA suffix escaped cancellation");
      else {
        DeviceState state{0,3}; std::vector<cc::Token> expected;
        for(int i=0;i<9;++i) {state.recurrent=(state.recurrent*33+std::uint32_t(state.token)+17)%65521; state.token=cc::Token(state.recurrent%32);expected.push_back(state.token);}
        require(result.tokens==expected,"CUDA device continuation changed arithmetic");
      }
    }
    std::cout<<"CUDA successor executed from device carry before predecessor readback; cancellation settled\n";
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
