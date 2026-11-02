#pragma once
#include "request.h"
#include "trace.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <vector>

namespace crit::core {
struct Engine::Impl : std::enable_shared_from_this<Engine::Impl> {
    using Wakeup = detail::RequestWakeup;
    using Clock = detail::Clock;
    using Request = detail::Request;
    Profile profile;
    // Native wrapper resources must outlive backend/model destruction, including quarantine.
    std::shared_ptr<void> native_owner;
    Resources resources;
    Reservation trace_storage;
    detail::TraceBuffer trace_buffer;
    Reservation operator_storage;
    std::unique_ptr<trace::Recorder> operator_ranges;
    std::shared_ptr<Backend> backend;
    std::unique_ptr<Model> model;
    std::shared_ptr<const TextFrontend> frontend;
    ModelInfo declaration;
    ConcurrentModel* concurrent = nullptr;
    struct Lane {
        RequestKey key;
        std::shared_ptr<Request> request;
        std::unique_ptr<ConcurrentSession> session;
        std::unique_ptr<Retirement> receipt;
        std::unique_ptr<Suspension> suspension;
        std::unique_ptr<Restoration> restoration;
        std::unique_ptr<ResumeImage> image;
        Clock::time_point recovery_started;
        std::uint64_t blocked_epoch = std::numeric_limits<std::uint64_t>::max();
        std::uint32_t frontier = 0;
        std::uint32_t prefill_wait_steps = 0;
        std::uint64_t prefix_affinity = 0;
        bool prefilled = false, retiring = false, terminal = false, frozen = false;
        std::uint32_t executing = 0, reserved_output = 0, step_limit = 1;
        std::uint64_t next_step = 1, publish_step = 1;
        std::shared_ptr<StepContinuation> tail;
        Finish finish = Finish::length;
        std::chrono::steady_clock::time_point retirement_started;
        std::uint64_t trace_retirement = 0, trace_started = 0;
    };
    std::vector<Lane> lanes;
    std::list<Lane> suspended;
    // Memory pressure: pressure_owner is the request whose progress holds admissions (the oldest resident, or a
    // restored request); the hold ends when request pressure_release (the blocked or restored one) gets a step
    // prepared, or when either retires.
    std::uint64_t capacity_epoch = 0, pressure_owner = 0, pressure_release = 0;
    std::vector<std::uint64_t> generations;
    std::vector<RowProgress*> row_lookup;
    struct PendingBatch {
        std::vector<WorkItem> items;
        std::unique_ptr<BatchExecution> execution;
        Clock::time_point started;
        // Stable list storage keeps the scope's borrowed item identities alive.
        detail::TraceScope step;
        PendingBatch(std::vector<WorkItem> work, detail::TraceBuffer& trace)
            :items(std::move(work)),started(Clock::now()),step(trace,TracePhase::step,std::span<const WorkItem>(items)) {}
    };
    std::list<PendingBatch> batches;
    std::deque<std::shared_ptr<Request>> waiting;
    std::shared_ptr<std::atomic<std::size_t>> outstanding = std::make_shared<std::atomic<std::size_t>>(0);
    std::uint64_t next_request = 0;
    std::size_t prefill_cursor = 0, decode_cursor = 0;
    bool prefer_decode = true;
    bool reserved_prefill_last = false;
    // Completion callbacks own only this signal, never the engine or sessions.
    std::shared_ptr<Wakeup> wakeup = std::make_shared<Wakeup>();
    std::shared_ptr<Request> pending;
    std::shared_ptr<Request> active_score;
    std::shared_ptr<Request> retained_request;
    std::unique_ptr<Session> retained_session;
    bool busy = false, stopping = false, poisoned = false;
    Impl* retained_next = nullptr;
    std::thread worker;
    // engine.cpp: construction, declaration checks, shutdown and completion.
    Impl(Profile p, std::shared_ptr<Backend> b);
    static std::unique_ptr<trace::Recorder> make_operator_ranges(const TraceConfig& config,const detail::TraceBuffer& lifecycle);
    // The engine's recorder while operator ranges are enabled; otherwise the thread keeps its own sink.
    trace::Sink* operator_sink_here() const noexcept {
        return operator_ranges ? operator_ranges.get() : trace::current_sink();
    }
    void drain_backend(RequestKey key = {});
    void quiesce_model();
    void initialize(Factory factory);
    bool shutdown() noexcept;
    std::span<const Token> stops(const Request& r) const {
        return r.options.stop_tokens ? std::span<const Token>(*r.options.stop_tokens) : std::span<const Token>(declaration.stop_tokens);
    }
    // make_exception_ptr copies its argument, and the copy shares the argument's message. Building the error here
    // destroys the argument before complete() publishes the copy, so afterwards only consumers touch the message.
    static std::exception_ptr runtime_failure(const char* message) {
        return std::make_exception_ptr(std::runtime_error(message));
    }
    void complete(const std::shared_ptr<Request>& r, Finish finish, std::exception_ptr error = {}, bool retained = false) noexcept;
    // submit.cpp
    Handle submit(std::vector<Token> prompt, std::uint32_t output, bool scoring, std::uint32_t first, RequestOptions,
                  std::optional<ScoreQueries> queries = std::nullopt);
    // serial.cpp: one request at a time.
    void execute(const std::shared_ptr<Request>& request);
    void run() noexcept;
    // scoring.cpp
    Finish score(const std::shared_ptr<Request>& request);
    Finish score_queries(const std::shared_ptr<Request>& request);
    bool score_concurrent();
    // admission.cpp: concurrent requests into free lanes. The scheduler lock (wakeup->mutex) is held except around
    // model callbacks.
    void set_step_limit(Lane& lane);
    bool admit_reuse(std::size_t lane_index, Clock::time_point now, std::unique_lock<std::mutex>& lock);
    void admit();
    // batches.cpp: concurrent step groups, their execution and publication.
    WorkItem work(const Lane& lane) const;
    void advise_prefill(WorkItem& item, std::uint32_t decode_work=0);
    void publish_group(std::span<const WorkItem> items, std::vector<RowProgress> rows, std::uint64_t operation, BatchExecution* execution=nullptr);
    bool step_concurrent();
    bool step_mixed();
    bool poll_batches();
    // concurrent.cpp: the concurrent worker, lane retirement, memory recovery and failure.
    void begin_retirement(Lane& lane, Finish finish);
    void poll_retirements();
    ContinuationView continuation(const Lane& lane) const;
    void poll_recovery();
    void memory_blocked(const BatchPreparation& preparation);
    void fail_concurrent(std::exception_ptr error) noexcept;
    void run_concurrent() noexcept;
};
// handle.cpp
struct Handle::Impl {
    std::shared_ptr<Engine::Impl> engine;
    std::shared_ptr<detail::Request> request;
    std::shared_ptr<std::atomic<std::size_t>> outstanding;
    ~Impl();
    void settle(std::unique_lock<std::mutex>& lock) noexcept;
    template<class Copy, class Deliver> Result consume(bool streaming, Copy copy, Deliver deliver);
};
}
