#pragma once
#include "request.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
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
    // Native wrapper resources must outlive the model's destruction.
    std::shared_ptr<void> native_owner;
    Resources resources;
    std::unique_ptr<Model> model;
    ModelInfo declaration;
    struct Lane {
        RequestKey key;
        std::shared_ptr<Request> request;
        std::unique_ptr<Session> session;
        std::uint32_t frontier = 0;
        std::uint32_t prefill_wait_steps = 0;
        detail::Pass pass;   // the prefill pass the model holds for it (ModelInfo's prefill_pass_tokens)
        bool prefilled = false, terminal = false;
        Finish finish = Finish::length;
    };
    std::vector<Lane> lanes;
    std::vector<std::uint64_t> generations;
    std::vector<RowProgress*> row_lookup;
    std::deque<std::shared_ptr<Request>> waiting;
    std::shared_ptr<std::atomic<std::size_t>> outstanding = std::make_shared<std::atomic<std::size_t>>(0);
    std::uint64_t next_request = 0;
    std::size_t prefill_cursor = 0, decode_cursor = 0;
    bool prefer_decode = true;
    bool reserved_prefill_last = false;
    // Completion callbacks own only this signal, never the engine or sessions.
    std::shared_ptr<Wakeup> wakeup = std::make_shared<Wakeup>();
    std::shared_ptr<Request> active_score;
    bool busy = false, stopping = false, poisoned = false;
    std::thread worker;
    // engine.cpp: construction, declaration checks, shutdown and completion.
    explicit Impl(Profile p);
    void initialize(Factory factory);
    void shutdown() noexcept;
    std::span<const Token> stops(const Request& r) const {
        return r.options.stop_tokens ? std::span<const Token>(*r.options.stop_tokens) : std::span<const Token>(declaration.stop_tokens);
    }
    // make_exception_ptr copies its argument, and the copy shares the argument's message. Building the error here
    // destroys the argument before complete() publishes the copy, so afterwards only consumers touch the message.
    static std::exception_ptr runtime_failure(const char* message) {
        return std::make_exception_ptr(std::runtime_error(message));
    }
    void complete(const std::shared_ptr<Request>& r, Finish finish, std::exception_ptr error = {}) noexcept;
    // submit.cpp
    Handle submit(std::vector<Token> prompt, std::uint32_t output, bool scoring, std::uint32_t first, RequestOptions,
                  std::optional<ScoreQueries> queries = std::nullopt);
    // scoring.cpp: a score runs alone, after every generation before it retired.
    Finish score(const std::shared_ptr<Request>& request);
    Finish score_queries(const std::shared_ptr<Request>& request);
    bool score_next();
    // admission.cpp: requests into free lanes, first in first out. The scheduler lock (wakeup->mutex) is
    // held except around model callbacks.
    void admit();
    // batches.cpp: step groups, their execution and publication.
    WorkItem work(const Lane& lane) const;
    void publish_group(std::span<const WorkItem> items, std::vector<RowProgress> rows);
    bool step();
    bool step_mixed();
    // worker.cpp: the worker, lane retirement and failure.
    void retire(Lane& lane, Finish finish);
    void retire_ended();
    void fail(std::exception_ptr error) noexcept;
    void run() noexcept;
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
