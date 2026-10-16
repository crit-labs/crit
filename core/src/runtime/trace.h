#pragma once
#include "crit/core.h"
#include <chrono>
#include <exception>
#include <mutex>

namespace crit::core::detail {
class TraceBuffer {
public:
    explicit TraceBuffer(TraceConfig);
    bool enabled() const noexcept { return config_.capacity != 0; }
    std::chrono::steady_clock::time_point epoch() const noexcept { return epoch_; }
    std::uint64_t now() const noexcept;
    void write(TraceEvent) noexcept;
    void point(TracePhase, RequestKey = {}, Finish = Finish::length) noexcept;
    void device(std::uint64_t operation, RequestKey, const DeviceTiming&) noexcept;
    void drop() noexcept;
    TraceSnapshot snapshot() const;
    // Span construction occurs only on the engine worker (or before/after it).
    std::uint64_t operation() noexcept;
    std::uint64_t step_gap(std::uint64_t started) const noexcept;
    bool end_step(std::uint64_t started, std::uint64_t ended) noexcept;
private:
    TraceConfig config_;
    std::chrono::steady_clock::time_point epoch_;
    std::unique_ptr<TraceEvent[]> events_;
    mutable std::mutex mutex_;
    std::size_t next_ = 0, size_ = 0;
    std::uint64_t overwritten_ = 0, operation_ = 0, last_step_end_ = 0;
    std::atomic<std::uint64_t> attempted_{0}, dropped_{0}, slow_steps_{0};
};

class TraceScope {
public:
    TraceScope(TraceBuffer&, TracePhase, RequestKey = {}) noexcept;
    TraceScope(TraceBuffer&, TracePhase, std::span<const WorkItem>) noexcept;
    ~TraceScope();
    // End retained asynchronous spans explicitly on success or failure. Safe to
    // call more than once; destruction still closes ordinary stack scopes.
    void finish(bool failed = false) noexcept;
    std::uint64_t operation() const noexcept { return operation_; }
    TraceScope(const TraceScope&) = delete;
    TraceScope& operator=(const TraceScope&) = delete;
private:
    TraceBuffer* trace_;
    TracePhase phase_;
    RequestKey single_key_{};
    // Group items and their identities must outlive the scope unchanged.
    std::span<const WorkItem> items_;
    std::size_t count_ = 1;
    std::uint64_t operation_ = 0, started_ = 0, gap_ = 0;
    int exceptions_ = 0;
    RequestKey key(std::size_t i) const noexcept { return items_.empty() ? single_key_ : items_[i].key; }
    void begin() noexcept;
};
}
