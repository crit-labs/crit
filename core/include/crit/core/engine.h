#pragma once
#include "crit/core/scheduling.h"
#include "crit/core/model.h"
#include "crit/core/profile.h"
#include "crit/core/request.h"
#include "crit/core/resources.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace crit::core {
namespace detail { struct RequestWakeup; }
// Builds the model; it may keep the ledger for its own reservations.
using Factory = std::function<std::unique_ptr<Model>(const Profile&, Resources&)>;
// Host steady-clock phase durations Core observed for one request, in
// nanoseconds; zero when a phase did not occur. Admission is when the worker
// opens the request. Prefill ends when Core accepts progress reporting prefill
// complete; token times are publications. These are host observations.
struct RequestTiming {
    std::uint64_t queued_ns = 0;      // submission -> admission
    std::uint64_t prefill_ns = 0;     // admission -> prefill complete
    std::uint64_t first_token_ns = 0; // admission -> first published token
    std::uint64_t decode_ns = 0;      // prefill complete -> last published token
    std::uint64_t total_ns = 0;       // admission -> completion
    std::uint32_t prefill_tokens = 0; // prompt tokens the model reported consuming
    std::uint32_t reused_prompt_tokens = 0; // of prefill_tokens, restored without computation
    std::uint32_t decode_steps = 0;   // generation steps after prefill completed
};
struct Result {
    std::vector<Token> tokens;
    std::vector<float> scores;
    // Query scoring only: targets x per_target log-probabilities and the argmax per target (empty if not provided).
    std::vector<float> query_scores;
    std::vector<Token> argmax;
    Finish finish = Finish::length;
    std::uint64_t request_id = 0;
    RequestTiming timing;
    // With RequestOptions::logprobs: one log-probability per token. top_logprobs holds
    // tokens x n alternatives for generation, or targets x ScoreQueries::top_k for scoring.
    std::vector<float> logprobs = {};
    std::vector<TokenLogprob> top_logprobs = {};
};
using TokenSink = std::function<void(Token)>;
// The tokens published since the last call, in order; the span is valid only during the call.
using TokensSink = std::function<void(std::span<const Token>)>;
// One published token. Spans are valid only during the sink call.
struct OutputEvent {
    std::size_t index = 0;
    Token token = 0;
    std::optional<float> logprob = {};
    std::span<const TokenLogprob> top_logprobs = {};
};
using OutputSink = std::function<void(const OutputEvent&)>;
class Handle {
public:
    Handle() noexcept;
    ~Handle();
    Handle(Handle&&) noexcept;
    Handle& operator=(Handle&&) noexcept;
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    // May run concurrently with wait(); move/destruction require exclusive ownership.
    void cancel() noexcept;
    std::uint64_t id() const noexcept;
    // When wait()/wait_output() returns or throws, or the destructor returns, the request's host charges are
    // released as seen from this thread. Whatever leaves a sink (an
    // exception, a foreign exception or forced unwinding) cancels the request and propagates unchanged after that.
    Result wait(TokenSink sink = {});
    // Like wait(), with every token published since the last call in one call: a step's (or a round's) tokens at once.
    Result wait_tokens(TokensSink sink);
    // Like wait(), with log-probabilities per published token.
    Result wait_output(OutputSink sink);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::shared_ptr<std::atomic<bool>> cancellation_;
    std::shared_ptr<detail::RequestWakeup> wake_cancel_;
    std::uint64_t id_ = 0;
    explicit Handle(std::unique_ptr<Impl>);
    friend class Engine;
};
class Engine {
public:
    Engine(Profile, Factory);
    ~Engine();
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Handle submit(std::vector<Token> prompt, std::uint32_t output_tokens, RequestOptions = {});
    Handle submit_score(std::vector<Token> prompt, std::uint32_t first_target, RequestOptions = {});
    // Requires ModelInfo::query_scoring; queries.ids holds queries.per_target ids for every target.
    Handle submit_score(std::vector<Token> prompt, std::uint32_t first_target, ScoreQueries queries, RequestOptions = {});
    ModelInfo info() const;
    Profile profile() const; // Actual core admission/budget/resource settings.
    Usage usage(Memory) const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Handle;
    friend struct NativeEngine;
};
}
