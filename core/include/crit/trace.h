#pragma once
// Optional operator ranges: work-annotated host scopes around backend operators.
// Scopes compile to nothing unless CRIT_TRACE_ENABLED is defined. The bounded
// Recorder keeps completed ranges; an Engine can own one (TraceConfig::operators)
// whose ranges share the lifecycle trace's clock and appear in TraceSnapshot.
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string_view>
#include <vector>

namespace crit::trace {
// Caller-supplied estimates of useful work. They are not measured DRAM traffic
// or device duration; keep useful_bytes_min <= useful_bytes_max.
struct Work {
    std::string_view stage;
    std::uint64_t tokens = 0, batch = 0, context = 0;
    std::uint64_t useful_bytes_min = 0, useful_bytes_max = 0, floating_ops = 0;
};
// A sink outlives its scopes; begin/end of one scope run on the same thread.
class Sink {
public:
    virtual ~Sink() = default;
    virtual void begin(const Work&) noexcept = 0;
    // failed: the scope ended by exception unwind.
    virtual void end(bool failed) noexcept = 0;
};
// Exception-safe range on a sink; a null sink records nothing.
class Scope {
public:
    Scope(Sink* sink, const Work& work) noexcept : sink_(sink), exceptions_(std::uncaught_exceptions()) {
        if(sink_) sink_->begin(work);
    }
    Scope(Sink& sink, const Work& work) noexcept : Scope(&sink, work) {}
    ~Scope() {
        if(sink_) sink_->end(std::uncaught_exceptions() > exceptions_);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
private:
    Sink* sink_;
    int exceptions_;
};

inline constexpr std::size_t stage_bytes = 96;  // including the terminator; longer names are truncated
inline constexpr std::size_t max_depth = 64;    // deeper ranges are dropped and counted
// One completed range. Timestamps are host steady-clock nanoseconds since the
// recorder's epoch (an Engine's lifecycle trace epoch for its own recorder).
struct Range {
    std::uint64_t sequence = 0;  // completion order, starting at 1
    std::uint64_t begin_ns = 0, end_ns = 0;
    std::uint32_t depth = 0;     // ranges open around this one on the same recorder
    bool failed = false;
    std::array<char, stage_bytes> stage{};
    std::uint64_t tokens = 0, batch = 0, context = 0;
    std::uint64_t useful_bytes_min = 0, useful_bytes_max = 0, floating_ops = 0;
    std::string_view name() const noexcept { return stage.data(); }
};
struct Snapshot {
    bool enabled = false;
    // overwritten: oldest ranges replaced by newer ones; dropped: ranges deeper
    // than max_depth, unmatched ends and writes that met a concurrent reader.
    std::uint64_t overwritten = 0, dropped = 0;
    std::vector<Range> ranges;  // completion order, oldest first
};
// Bounded ring of completed ranges, allocated at construction and never grown.
// One writer thread at a time calls begin/end; snapshot() is safe from any thread
// and never blocks the writer (a write that meets a reader is dropped and counted).
class Recorder final : public Sink {
public:
    explicit Recorder(std::size_t capacity,
                      std::chrono::steady_clock::time_point epoch = std::chrono::steady_clock::now());
    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;
    void begin(const Work&) noexcept override;
    void end(bool failed) noexcept override;
    Snapshot snapshot() const;
    std::size_t capacity() const noexcept { return ranges_.size(); }
private:
    std::chrono::steady_clock::time_point epoch_;
    std::vector<Range> ranges_;
    std::vector<Range> open_;  // begun ranges, innermost last
    std::size_t depth_ = 0, overflow_ = 0, next_ = 0, size_ = 0;
    std::uint64_t sequence_ = 0, overwritten_ = 0;
    std::atomic<std::uint64_t> dropped_{0};
    mutable std::mutex mutex_;
    std::uint64_t now() const noexcept;
    void commit(Range&) noexcept;
};
// The calling thread's operator sink (null unless bound); CRIT_TRACE_OPERATOR uses it.
// An Engine with TraceConfig::operators binds its recorder on the threads that
// run model callbacks: its worker, and the caller during construction and shutdown.
Sink* current_sink() noexcept;
// Binds a sink to the calling thread for this object's lifetime, restoring the previous one.
class SinkBinding {
public:
    explicit SinkBinding(Sink*) noexcept;
    ~SinkBinding();
    SinkBinding(const SinkBinding&) = delete;
    SinkBinding& operator=(const SinkBinding&) = delete;
private:
    Sink* previous_;
};
#ifdef CRIT_TRACE_NVTX
// Host-thread NVTX ranges carrying the work fields; creates no CUDA events and
// never waits for device work. Built into crit_core with CRIT_CORE_NVTX=ON, so
// the NVTX 3 headers (and windows.h, which they include) stay private to Core.
class NvtxSink final : public Sink {
public:
    void begin(const Work& work) noexcept override;
    void end(bool) noexcept override;
};
#endif
}  // namespace crit::trace

#define CRIT_TRACE_JOIN_(a, b) a##b
#define CRIT_TRACE_JOIN(a, b) CRIT_TRACE_JOIN_(a, b)
#ifdef CRIT_TRACE_ENABLED
// The work expression is evaluated when tracing is compiled in, even without a bound sink.
#define CRIT_TRACE_SCOPE(sink, ...) \
    ::crit::trace::Scope CRIT_TRACE_JOIN(crit_trace_scope_, __LINE__)((sink), (__VA_ARGS__))
#define CRIT_TRACE_OPERATOR(...) \
    ::crit::trace::Scope CRIT_TRACE_JOIN(crit_trace_scope_, __LINE__)(::crit::trace::current_sink(), (__VA_ARGS__))
#else
// Ordinary builds evaluate neither the sink nor the work expression.
#define CRIT_TRACE_SCOPE(...) ((void)0)
#define CRIT_TRACE_OPERATOR(...) ((void)0)
#endif
