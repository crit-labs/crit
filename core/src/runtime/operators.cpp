#include "crit/trace.h"
#include <algorithm>

namespace crit::trace {
namespace {
thread_local Sink* bound = nullptr;
}
Recorder::Recorder(std::size_t capacity,std::chrono::steady_clock::time_point epoch)
    :epoch_(epoch),ranges_(capacity),open_(max_depth) {}
std::uint64_t Recorder::now() const noexcept {
    const auto elapsed=std::chrono::steady_clock::now()-epoch_;
    return elapsed.count()>0 ? std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()) : 0;
}
void Recorder::begin(const Work& work) noexcept {
    if(depth_==open_.size()) {
        // The matching end is ignored, so shallower ranges keep their nesting.
        ++overflow_;
        dropped_.fetch_add(1,std::memory_order_relaxed);
        return;
    }
    auto& range=open_[depth_];
    range.depth=std::uint32_t(depth_++);
    const auto count=std::min(work.stage.size(),range.stage.size()-1);
    std::copy_n(work.stage.data(),count,range.stage.data());
    range.stage[count]='\0';
    range.tokens=work.tokens; range.batch=work.batch; range.context=work.context;
    range.useful_bytes_min=work.useful_bytes_min; range.useful_bytes_max=work.useful_bytes_max;
    range.floating_ops=work.floating_ops;
    range.begin_ns=now();
}
void Recorder::end(bool failed) noexcept {
    const auto ended=now();
    if(overflow_) { --overflow_; return; }
    if(!depth_) { dropped_.fetch_add(1,std::memory_order_relaxed); return; }
    auto& range=open_[--depth_];
    range.end_ns=ended;
    range.failed=failed;
    commit(range);
}
void Recorder::commit(Range& range) noexcept {
    // Readers never block the writer; a contended or capacity-free write is counted.
    std::unique_lock lock(mutex_,std::try_to_lock);
    if(!lock.owns_lock() || ranges_.empty()) { dropped_.fetch_add(1,std::memory_order_relaxed); return; }
    range.sequence=++sequence_;
    ranges_[next_]=range;
    next_=(next_+1)%ranges_.size();
    if(size_<ranges_.size()) ++size_;
    else ++overwritten_;
}
Snapshot Recorder::snapshot() const {
    Snapshot result;
    result.enabled=true;
    result.ranges.resize(ranges_.size()); // No allocation while holding the writer lock.
    std::size_t count=0;
    {
        std::lock_guard lock(mutex_);
        count=size_;
        const auto first=ranges_.empty() ? 0 : (next_+ranges_.size()-size_)%ranges_.size();
        for(std::size_t i=0;i<count;++i) result.ranges[i]=ranges_[(first+i)%ranges_.size()];
        result.overwritten=overwritten_;
        result.dropped=dropped_.load(std::memory_order_relaxed);
    }
    result.ranges.resize(count);
    return result;
}
Sink* current_sink() noexcept { return bound; }
SinkBinding::SinkBinding(Sink* sink) noexcept:previous_(bound) { bound=sink; }
SinkBinding::~SinkBinding() { bound=previous_; }
}
