#pragma once
#include "crit/core/request.h"
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace crit::core {
struct ModelInfo {
    std::string implementation;
    std::uint32_t vocabulary = 0, context = 0, prefill_chunk = 0;
    std::vector<Token> stop_tokens;
    bool generation = true, scoring = false;
    std::uint32_t max_concurrency = 1;
    SamplingSupport sampling = {};
    bool request_context = false;
    // Model::score_queries: per target, log-probabilities of caller-chosen token ids and the argmax.
    bool query_scoring = false;
    bool mixed_batches = false;
    // Preferred decode work per row; execute() must also support smaller grants.
    std::uint32_t decode_token_work = 1;
    bool scheduling_hints = false;
    bool prefill_advice = false;
    bool async_batches = false;
    // Ids below this are real tokens; zero means vocabulary. Padding ids at or above
    // it never appear in log-probability rows, top-k rows or argmax.
    std::uint32_t valid_vocabulary = 0;
    // Progress::logprobs for generated tokens (see TokenLogprob for the definition).
    bool token_logprobs = false;
    // Most alternatives per generated token or scored target; at most top_logprobs_limit.
    std::uint32_t max_top_logprobs = 0;
    // Named scoring paths (for example prefill and decode); empty means one default path.
    std::vector<std::string> score_paths = {};
    bool memory_preemption = false;
    std::uint32_t max_request_steps = 1;
    // Mixed prefill work is granted in multiples of this quantum, including
    // padded columns. Prompt consumption remains bounded by actual input.
    // One preserves single-token grants; decode always supports them.
    std::uint32_t prefill_work_quantum = 1;
};
class Readiness {
public:
    virtual ~Readiness() = default;
    virtual void wait() = 0;
};
struct WorkBudget {
    std::uint32_t prefill_tokens, output_tokens;
    bool prefill_complete;
    // Mixed execution only: prompt computation, draft/verification and padding
    // must fit this grant. Token work is a quota, not a latency estimate.
    std::uint32_t token_work = 0;
};
struct Progress {
    std::uint32_t prompt_tokens = 0;
    bool prefill_complete = false;
    std::span<const Token> tokens;
    Readiness* readiness = nullptr;
    // A device round may learn how many tokens it produced only after its
    // readback. With deferred_tokens set, tokens must be empty, prefill complete
    // and readiness non-null; after readiness Core reads the final span from
    // Session::ready_tokens() (ConcurrentModel::ready_tokens for batched rows).
    bool deferred_tokens = false;
    // Of prompt_tokens, those restored from state the model retained from an
    // earlier request instead of being computed. Reuse must cover a prompt
    // prefix (before any computed prompt token) and does not count against the
    // step's prefill budget; reported in RequestTiming::reused_prompt_tokens.
    std::uint32_t reused_prompt_tokens = 0;
    // Actual token work used by execute(), including hidden/padded columns.
    std::uint32_t token_work = 0;
    // With RequestOptions::logprobs: one log-probability per token and n ranked
    // alternatives per token (row-major), borrowed like tokens. With deferred_tokens
    // they must cover at least the final token count, which Core reads after readiness.
    std::span<const float> logprobs = {};
    std::span<const TokenLogprob> top_logprobs = {};
};
// Device stage durations measured by the adapter, e.g. around graph launches.
enum class DeviceStage : std::uint8_t { forward, draft, verify, accept, state, head, other };
struct DeviceTiming {
    DeviceStage stage;
    std::uint32_t index = 0;
    std::uint64_t duration_ns = 0;
};
enum class Finish { length, stop, cancelled, failed };
struct PrefillWindow {
    std::uint32_t occupied_tokens, remaining_tokens, token_limit, decode_work;
    std::uint64_t target_ns;
};
class Session {
public:
    virtual ~Session() = default;
    virtual Progress advance(WorkBudget) = 0;
    // Pure bounded advice; zero keeps the offered limit. Nonzero must fit it.
    virtual std::uint32_t prefill_limit(const PrefillWindow&) { return 0; }
    // Called only while tracing, after readiness and before commit. Storage is
    // borrowed until commit; a throw drops the timings, not the request.
    virtual std::span<const DeviceTiming> device_timings() { return {}; }
    // Called once after readiness when advance() returned deferred_tokens.
    // Storage follows the same borrowing rule as Progress::tokens.
    virtual std::span<const Token> ready_tokens();
    // Token storage remains live until commit/close. Commit may throw; no output
    // is published on failure. Async state updates must obey backend ordering.
    virtual void commit(std::uint32_t accepted, bool terminal) = 0;
    // Called after backend drain. Must not throw, enqueue, or retain prompt views.
    virtual void close(Finish) noexcept = 0;
};
// Private state for one text stream. Emit only complete text; retain incomplete
// byte/token fragments until a later push or finish. Access each decoder serially.
class TextDecoder {
public:
    virtual ~TextDecoder() = default;
    virtual std::string push(std::span<const Token>) = 0;
    virtual std::string finish() = 0;
};
// Optional host-only service, independent of mutable Model/Backend state. All
// methods must be safe for concurrent calls, including while generation runs.
// Retain everything needed independently of the engine's lifetime.
class TextFrontend {
public:
    virtual ~TextFrontend() = default;
    virtual std::vector<Token> tokenize(std::string_view) const = 0;
    virtual std::string detokenize(std::span<const Token>) const = 0;
    // The default throws std::logic_error; request text (RequestOptions::text or
    // stop_strings) then decodes with offset_decoder from crit/text.h, seeded with
    // the prompt. A frontend's own decoder gets no prompt context: continuation
    // spacing and a prompt that ends inside a character are its responsibility.
    virtual std::unique_ptr<TextDecoder> create_decoder() const;
};
class Model {
public:
    virtual ~Model() = default;
    virtual ModelInfo info() const = 0;
    virtual std::unique_ptr<Session> open(std::span<const Token> prompt) = 0;
    // Options remain live through close and any failed-drain quarantine.
    // The default preserves legacy open only when no adapter options are set.
    virtual std::unique_ptr<Session> open_request(std::span<const Token>, const RequestOptions&);
    // Legacy idle-only text callbacks. The defaults throw logic_error; prefer text_frontend().
    virtual std::vector<Token> tokenize(std::string_view);
    virtual std::string detokenize(std::span<const Token>);
    // Called once at construction. Empty preserves serialized idle-only access
    // to the legacy tokenize/detokenize callbacks above.
    virtual std::shared_ptr<const TextFrontend> text_frontend() const { return {}; }
    // Return one finite score per target. Join host producers before returning;
    // Core drains backend queue work before validating/publishing these values.
    // ConcurrentModel scoring runs only with all generation sessions retired.
    // Reserve returned scratch in model-owned resources until the next ordinary
    // request callback or destruction. Core separately reserves public results
    // and retains returned storage through drain, including failed-drain quarantine.
    virtual std::vector<float> score(std::span<const Token>, std::uint32_t first_target);
    virtual std::vector<float> score_request(std::span<const Token>, std::uint32_t first_target, const RequestOptions&);
    // Requires ModelInfo::query_scoring. Same ownership and draining rules as score(); `target` holds log-probabilities.
    virtual QueryScores score_queries(std::span<const Token>, std::uint32_t first_target, const ScoreQueries&,
                                      const RequestOptions&);
};
}
