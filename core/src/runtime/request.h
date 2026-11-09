#pragma once
#include "crit/core.h"
#include "crit/text.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

// The runtime's request record and the request rules its files share (defined in progress.cpp and frontend.cpp).
namespace crit::core::detail {
struct RequestWakeup {
    std::mutex mutex;
    std::condition_variable cv;
    bool changed=false;
    void notify() {
        { std::lock_guard lock(mutex); changed=true; }
        cv.notify_one();
    }
};
using Clock = std::chrono::steady_clock;
// Worker-private phase observations; folded into Result::timing at completion.
struct PhaseClock {
    Clock::time_point admitted, prefilled, first, last;
    bool has_admitted = false, has_prefilled = false, has_first = false;
    std::uint32_t prefill_tokens = 0, reused_tokens = 0, decode_steps = 0;
    void admit() { admitted = Clock::now(); has_admitted = true; }
    // Reuse is valid only for a prompt prefix: before any computed prompt token.
    bool reuse_allowed() const { return prefill_tokens == reused_tokens; }
    // One accepted step: prompt tokens consumed (of which reused), whether prefill
    // had already completed before the step, whether it completes now, tokens published.
    void step(std::uint32_t prompt_tokens, std::uint32_t reused, bool was_prefilled, bool prefilled_now, std::size_t published) {
        const auto now = Clock::now();
        prefill_tokens += prompt_tokens;
        reused_tokens += reused;
        if(was_prefilled) ++decode_steps;
        if(!was_prefilled && prefilled_now) { prefilled = now; has_prefilled = true; }
        if(published) { if(!has_first) { first = now; has_first = true; } last = now; }
    }
};
inline std::uint64_t elapsed_ns(Clock::time_point from, Clock::time_point to) {
    return to > from ? std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count()) : 0;
}
struct Request {
    std::uint64_t id = 0;
    RequestKey trace_key;
    std::chrono::steady_clock::time_point submitted;
    PhaseClock phases;
    Reservation storage, staging_storage;
    std::vector<Token> prompt;
    RequestOptions options;
    std::uint32_t output = 0, first_target = 0;
    std::uint32_t admission_bypasses = 0;
    bool scoring = false;
    std::vector<Token> staging;
    // The model may still have queued writes into its returned score allocation.
    // Keep that allocation in the quarantinable owner until drain succeeds.
    std::vector<float> score_output;
    ScoreQueries queries;
    bool has_queries = false;
    QueryScores query_output;
    std::shared_ptr<std::atomic<bool>> cancelled = std::make_shared<std::atomic<bool>>(false);
    std::mutex mutex;
    std::condition_variable cv;
    Result result;
    std::exception_ptr error;
    bool done = false;
    // Whether the waiter takes each token as it is published (a streaming sink), under the lock: a waiter that takes
    // its tokens at completion is woken only then.
    bool streaming = true;
    // Generation text (RequestOptions::text or stop_strings): the stream is worker-private after submission;
    // result.text grows only inside its charged capacity.
    bool text = false;
    std::unique_ptr<TextStream> text_stream;
    std::vector<Reservation> text_storage;
    std::size_t text_charged = 0;
    // A concurrent row failure confined to this request (text decoding or storage), reported at completion.
    std::exception_ptr failure;
    // Set with done, under the lock, when the quarantined runtime keeps this request: an unsettled reader may still
    // use its prompt, options or returned storage, so its charges stay with that runtime.
    bool retained = false;
    Request() = default;
    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    ~Request() { release(); }
    // Frees every host owner charged to this request, then returns its charges; idempotent. Needs exclusive access:
    // the submitting thread before publication, the consumer once done is set on a request that is not retained
    // (completion is the worker's last access to these owners), or the last owner.
    void release() noexcept {
        std::vector<Token>().swap(prompt);
        options = {};
        std::vector<Token>().swap(staging);
        std::vector<float>().swap(score_output);
        queries = {};
        query_output = {};
        result = {};
        text_stream.reset();
        // The ledger last: memory is freed before its charge becomes available again.
        std::vector<Reservation>().swap(text_storage);
        text_charged = 0;
        staging_storage = {};
        storage = {};
    }
};
inline std::uint32_t valid_vocabulary(const ModelInfo& info) {
    return info.valid_vocabulary ? info.valid_vocabulary : info.vocabulary;
}
// Valid-vocabulary checks pass a message that names the valid vocabulary.
void validate_tokens(std::span<const Token> tokens, std::uint32_t vocabulary,
                     const char* message = "token outside declared vocabulary");
// Shape only, before materialization; the caller owns readiness ordering.
void validate_deferred_progress(const Progress& p, const char* message);
// Shared serial/batched contract after readiness and deferred materialization.
// This only validates: frontier changes, mixed work accounting and publication stay with the caller.
void validate_progress(const Progress& p, const WorkBudget& budget, std::uint32_t frontier,
                       const Request& request, const char* message);
inline bool has_sampling(const SamplingOptions& s) {
    return s.temperature || s.top_p || s.top_k || s.seed || s.min_p || s.repetition_penalty || s.frequency_penalty ||
           s.presence_penalty || !s.logit_bias.empty() || s.min_tokens;
}
inline bool has_generation_options(const RequestOptions& o) {
    return has_sampling(o.sampling) || o.stop_tokens || o.logprobs || !o.stop_strings.empty() || o.text;
}
// Generation log-probabilities: a value never NaN or +inf and at most zero within the tolerance (-inf is a
// zero probability). Ranked rows of `width` alternatives hold distinct valid ids in non-increasing order.
bool logprob_value(float x);
void validate_ranked(std::span<const TokenLogprob> rows, std::size_t width, std::uint32_t valid);
// The requested rows of one step's tokens (Progress::tokens, or the final span of a deferred round). Counts are
// exact; deferred rounds may supply spans covering at least the final count, of which Core reads the first rows.
struct LogprobRows { std::span<const float> chosen; std::span<const TokenLogprob> top; std::size_t width = 0; };
LogprobRows logprob_rows(const RequestOptions& options, const Progress& progress, std::uint32_t valid);
// Released text of one step and, per decoded accepted token, its end offset in `bytes`.
struct StepText { std::string bytes; std::vector<std::uint32_t> ends; };
struct Decision { std::uint32_t accepted = 0; Finish finish = Finish::length; bool terminal = false; };
// The one trim rule for both paths: stop tokens (kept, never decoded), the output budget, then stop strings over
// the accepted tokens' text. A terminal step flushes the held text into the last accepted token's piece; a stop
// that completes in that tail still ends with Finish::stop. Text exceptions propagate to the caller.
Decision decide(std::span<const Token> tokens, std::span<const Token> stops, std::uint32_t remaining,
                TextStream* text, StepText& out);
// Result extents before staging one step, restored when its commit fails.
struct Extents { std::size_t tokens = 0, logprobs = 0, top = 0, text = 0, text_ends = 0; };
Extents extents(const Result& r);
void restore(Result& r, const Extents& e);
// Appends one step's accepted rows and text, copied before commit can invalidate borrowed rows. Lock held;
// nothing is appended when the text reservation fails.
void stage_output(Request& r, Resources& resources, const LogprobRows& rows, std::uint32_t accepted, const StepText& text);
// The frontend's own decoder, or offset_decoder seeded with the prompt tail when the frontend keeps the default
// create_decoder() (which throws exactly std::logic_error). Other failures and a null decoder propagate.
std::unique_ptr<TextStream> make_text_stream(const std::shared_ptr<const TextFrontend>& frontend, std::span<const Token> prompt,
                                             const std::vector<std::string>& stops, std::uint32_t vocabulary);
}
