#include "trace.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace crit::core::detail {
namespace {
std::uint64_t increment(std::atomic<std::uint64_t>& value) noexcept {
    auto old = value.load(std::memory_order_relaxed);
    while(old != std::numeric_limits<std::uint64_t>::max()) {
        if(value.compare_exchange_weak(old,old+1,std::memory_order_relaxed)) return old+1;
    }
    return 0; // Counter exhaustion cannot wrap into a valid identity.
}
}
TraceBuffer::TraceBuffer(TraceConfig config):config_(config),epoch_(std::chrono::steady_clock::now()) {
    if(config.capacity > 65536 || config.slow_step_us > std::numeric_limits<std::uint64_t>::max()/1000)
        throw std::invalid_argument("invalid trace capacity/threshold");
    if(config.capacity) events_ = std::make_unique<TraceEvent[]>(config.capacity);
}
std::uint64_t TraceBuffer::now() const noexcept {
    return std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-epoch_).count());
}
void TraceBuffer::write(TraceEvent event) noexcept {
    if(!enabled()) return;
    try {
        // Snapshot readers never block inference. Contended events are counted.
        std::unique_lock lock(mutex_,std::try_to_lock);
        if(!lock.owns_lock()) { increment(dropped_); return; }
        event.sequence = increment(attempted_);
        if(!event.sequence) { increment(dropped_); return; }
        if(!event.timestamp_ns) event.timestamp_ns=now();
        events_[next_]=event;
        next_=(next_+1)%config_.capacity;
        if(size_ < config_.capacity) ++size_;
        else if(overwritten_ != std::numeric_limits<std::uint64_t>::max()) ++overwritten_;
    } catch(...) { increment(dropped_); }
}
void TraceBuffer::point(TracePhase phase,RequestKey key,Finish finish) noexcept {
    if(enabled()) write({0,0,0,key,phase,finish==Finish::failed ? TraceMark::failed : TraceMark::point,0,0,false,finish});
}
void TraceBuffer::device(std::uint64_t operation,RequestKey key,const DeviceTiming& timing) noexcept {
    if(enabled()) write({0,operation,0,key,TracePhase::device,TraceMark::point,timing.duration_ns,0,false,Finish::length,timing.stage,timing.index});
}
void TraceBuffer::drop() noexcept { increment(dropped_); }
TraceSnapshot TraceBuffer::snapshot() const {
    TraceSnapshot result;
    result.enabled=enabled();
    if(!enabled()) return result;
    result.events.resize(config_.capacity); // No allocation while holding the writer lock.
    std::size_t count;
    {
        std::lock_guard lock(mutex_);
        count=size_;
        const auto first=(next_+config_.capacity-size_)%config_.capacity;
        for(std::size_t i=0;i<count;++i) result.events[i]=events_[(first+i)%config_.capacity];
        result.overwritten=overwritten_;
        result.dropped=dropped_.load(std::memory_order_relaxed);
        result.slow_steps=slow_steps_.load(std::memory_order_relaxed);
    }
    result.events.resize(count);
    return result;
}
std::uint64_t TraceBuffer::operation() noexcept {
    return operation_ == std::numeric_limits<std::uint64_t>::max() ? 0 : ++operation_;
}
std::uint64_t TraceBuffer::step_gap(std::uint64_t started) const noexcept {
    return last_step_end_ ? started-last_step_end_ : 0;
}
bool TraceBuffer::end_step(std::uint64_t started,std::uint64_t ended) noexcept {
    last_step_end_=ended;
    const bool slow=config_.slow_step_us && ended-started >= config_.slow_step_us*1000;
    if(slow) increment(slow_steps_);
    return slow;
}
TraceScope::TraceScope(TraceBuffer& trace,TracePhase phase,RequestKey key) noexcept:
    trace_(trace.enabled() ? &trace : nullptr),phase_(phase) {
    single_key_=key; begin();
}
TraceScope::TraceScope(TraceBuffer& trace,TracePhase phase,std::span<const WorkItem> items) noexcept:
    trace_(trace.enabled() ? &trace : nullptr),phase_(phase),items_(items),count_(items.size()) {
    begin();
}
void TraceScope::begin() noexcept {
    if(!trace_) return;
    operation_=trace_->operation();
    started_=trace_->now();
    exceptions_=std::uncaught_exceptions();
    if(phase_==TracePhase::step) gap_=trace_->step_gap(started_);
    for(std::size_t i=0;i<count_;++i)
        trace_->write({0,operation_,started_,key(i),phase_,TraceMark::begin,0,gap_});
}
TraceScope::~TraceScope() {
    finish(std::uncaught_exceptions()>exceptions_);
}
void TraceScope::finish(bool failed) noexcept {
    if(!trace_) return;
    const auto ended=trace_->now();
    const bool slow=phase_==TracePhase::step && trace_->end_step(started_,ended);
    const auto mark=failed ? TraceMark::failed : TraceMark::end;
    for(std::size_t i=0;i<count_;++i)
        trace_->write({0,operation_,ended,key(i),phase_,mark,ended-started_,gap_,slow});
    trace_=nullptr;
}
}
