#pragma once
#include "crit/core/concurrent.h"
#include "crit/core/model.h"
#include "crit/core/profile.h"
#include "crit/core/request.h"
#include "crit/core/resources.h"
#include "crit/trace.h"
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
enum class TracePhase { submitted, admission, open, step, prefill, decode, score,
                        readiness, commit, publication, retirement, quiesce, drain, completion, device, mixed };
enum class TraceMark { point, begin, end, failed };
struct TraceEvent {
    std::uint64_t sequence = 0, operation = 0, timestamp_ns = 0;
    RequestKey key; // Zero request denotes engine-wide work; generation zero has no generation lane (queued or scoring).
    TracePhase phase = TracePhase::submitted;
    TraceMark mark = TraceMark::point;
    std::uint64_t duration_ns = 0, gap_ns = 0;
    bool slow = false;
    Finish finish = Finish::length; // Meaningful on completion events.
    DeviceStage stage = DeviceStage::other; // Device events: adapter stage and index;
    std::uint32_t index = 0;                // operation is the enclosing step's.
};
struct TraceSnapshot {
    bool enabled = false;
    std::uint64_t overwritten = 0, dropped = 0, slow_steps = 0;
    std::vector<TraceEvent> events; // Recording order; loss is counted separately.
    // Operator ranges (TraceConfig::operators), timed on the same clock as events.
    ::crit::trace::Snapshot operators;
};
using Factory = std::function<std::unique_ptr<Model>(const Profile&, Services)>;
// Host steady-clock phase durations Core observed for one request, in
// nanoseconds; zero when a phase did not occur. Admission is when the worker
// starts (or, for concurrent models, opens) the request. Prefill ends when Core
// accepts progress reporting prefill complete; token times are publications.
// These are host observations around the model's readiness, not device timings.
struct RequestTiming {
    std::uint64_t queued_ns = 0;      // submission -> admission
    std::uint64_t prefill_ns = 0;     // admission -> prefill complete
    std::uint64_t first_token_ns = 0; // admission -> first published token
    std::uint64_t decode_ns = 0;      // prefill complete -> last published token
    std::uint64_t total_ns = 0;       // admission -> completion
    std::uint32_t prefill_tokens = 0; // prompt tokens the model reported consuming
    std::uint32_t reused_prompt_tokens = 0; // of prefill_tokens, restored without computation
    std::uint32_t decode_steps = 0;   // generation steps after prefill completed
    std::uint32_t preemptions = 0;
    // Successful restoration prepare -> readiness; excludes time suspended.
    std::uint64_t recovery_ns = 0, recomputed_tokens = 0;
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
    // With RequestOptions::text or stop_strings: the released text, and for each token the
    // end offset in `text` of the text released when that token was published.
    std::string text = {};
    std::vector<std::uint32_t> text_ends = {};
};
using TokenSink = std::function<void(Token)>;
// One published token. Spans and text are valid only during the sink call.
struct OutputEvent {
    std::size_t index = 0;
    Token token = 0;
    std::optional<float> logprob = {};
    std::span<const TokenLogprob> top_logprobs = {};
    // Text released when this token was published; may be empty.
    std::string_view text = {};
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
    // released as seen from this thread; a quarantined runtime keeps them instead. Whatever leaves a sink (an
    // exception, a foreign exception or forced unwinding) cancels the request and propagates unchanged after that.
    Result wait(TokenSink sink = {});
    // Like wait(), with log-probabilities and text per published token.
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
    Engine(Profile, std::shared_ptr<Backend>, Factory);
    ~Engine();
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Handle submit(std::vector<Token> prompt, std::uint32_t output_tokens, RequestOptions = {});
    Handle submit_score(std::vector<Token> prompt, std::uint32_t first_target, RequestOptions = {});
    // Requires ModelInfo::query_scoring; queries.ids holds queries.per_target ids for every target.
    Handle submit_score(std::vector<Token> prompt, std::uint32_t first_target, ScoreQueries queries, RequestOptions = {});
    std::vector<Token> tokenize(std::string_view);
    std::string detokenize(std::span<const Token>);
    // Requires an independent frontend with an incremental decoder. The returned
    // decoder retains its frontend, validates token IDs, and may outlive Engine.
    std::unique_ptr<TextDecoder> create_decoder();
    ModelInfo info() const;
    Profile profile() const; // Actual core admission/budget/resource settings.
    Usage usage(Memory) const;
    // Safe while the worker is running, including a stuck callback. Allocates
    // only in the reading thread. Does not fence device work or invoke the model.
    TraceSnapshot trace() const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Handle;
    friend struct NativeEngine;
};
}
