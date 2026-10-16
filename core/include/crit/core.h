#pragma once
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
using Token = std::int32_t;
enum class Memory { host, device };
struct Usage { std::size_t live = 0, peak = 0, limit = 0; };

// A reservation is shared by aliases and releases exactly once at the last owner.
class Reservation {
public:
    Reservation() = default;
    std::size_t bytes() const noexcept;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    explicit Reservation(std::shared_ptr<Impl>);
    friend class Resources;
};
class Resources {
public:
    Resources(std::size_t host_limit, std::size_t device_limit);
    Reservation reserve(Memory, std::size_t bytes);
    Usage usage(Memory) const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

// Borrowed storage. Marks/rewind are arithmetic operations, never device fences.
class Arena {
public:
    Arena(void* storage, std::size_t capacity);
    std::span<std::byte> allocate(std::size_t bytes, std::size_t alignment = 256);
    std::size_t mark() const noexcept;
    void rewind(std::size_t mark);
    std::size_t peak() const noexcept;
private:
    std::byte* storage_;
    std::size_t capacity_, offset_ = 0, peak_ = 0;
};

struct TraceConfig {
    // Zero disables tracing. Capacity is bounded to 65536 fixed-size events.
    std::size_t capacity = 0;
    // Zero disables slow-step classification; timestamps remain host timings.
    std::uint64_t slow_step_us = 0;
    // Zero disables operator ranges. Otherwise the engine owns a crit::trace::Recorder
    // of this many completed ranges (at most 65536), independent of capacity, bound as
    // the operator sink of the threads that run model callbacks (see crit/trace.h).
    std::size_t operators = 0;
};
// Bounds host scheduler metadata; actual capacity also needs model admission.
inline constexpr std::uint32_t max_request_concurrency = 4096;
enum class BatchPolicy { fair, decode_first };
enum class AdmissionPolicy { fifo, reuse_aware };
struct Profile {
    std::uint32_t context = 32768, prefill_chunk = 1024;
    std::size_t host_limit = 0, device_limit = 0;
    // Generation concurrency is opt-in. Scores join its FIFO as exclusive
    // barriers after generation retirement. Completed, unconsumed handles also
    // occupy the outstanding allowance (max_concurrency + max_pending).
    std::uint32_t max_concurrency = 1, max_pending = 0;
    std::uint32_t pending_timeout_ms = 30000, retirement_timeout_ms = 30000;
    TraceConfig trace = {};
    // Zero preserves separate prefill/decode callbacks. Otherwise require a
    // mixed_batches model and bound total token work per execute() group.
    std::uint32_t max_batch_tokens = 0;
    BatchPolicy batch_policy = BatchPolicy::fair;
    // With decode_first, a prefill waiting this many execute groups is eligible
    // for the oldest-waiter's reserved unit. Other aged prefills take turns.
    std::uint32_t prefill_starvation_steps = 8;
    // Zero preserves FIFO. Otherwise inspect at most this many followers of a
    // deferred head, never crossing a score barrier or a capped skipped request.
    std::uint32_t admission_lookahead = 0, max_admission_bypasses = 8;
    AdmissionPolicy admission_policy = AdmissionPolicy::fifo;
    std::uint32_t admission_starvation_ms = 1000;
    // Zero keeps fixed chunks. Advice may shrink grants, never enlarge them.
    std::uint64_t prefill_target_ns = 0;
    // Opt-in mixed execution pipeline. Each lane belongs to at most one batch.
    std::uint32_t max_inflight_batches = 0, batch_timeout_ms = 30000;
};
// Most alternatives any log-probability row may carry.
inline constexpr std::uint32_t top_logprobs_limit = 64;
// Additive logit bias: finite, or negative infinity to ban the token.
struct LogitBias { Token token = 0; float bias = 0; };
struct SamplingOptions {
    // Unset uses the adapter default. Temperature zero requests greedy output;
    // top_k zero disables that filter. crit/sampling.h implements these rules.
    std::optional<double> temperature, top_p;
    std::optional<std::uint32_t> top_k;
    std::optional<std::uint64_t> seed;
    // In [0, 1]: drop tokens whose probability is below min_p times the largest.
    std::optional<double> min_p = {};
    // Repetition (finite, > 0, 1 is off) covers prompt and output tokens; frequency
    // and presence (finite, in [-2, 2]) cover output tokens only.
    std::optional<double> repetition_penalty = {}, frequency_penalty = {}, presence_penalty = {};
    // Unique in-vocabulary tokens, at most 1024 entries.
    std::vector<LogitBias> logit_bias = {};
    // Stop tokens are suppressed until this many tokens were generated.
    std::optional<std::uint32_t> min_tokens = {};
};
struct SamplingSupport {
    bool temperature = false, top_p = false, top_k = false, seed = false;
    bool min_p = false, penalties = false, logit_bias = false, min_tokens = false;
};
class RequestContext {
public:
    virtual ~RequestContext() = default;
    // Immutable host-only extension parameters. Report their retained payload
    // extent; Core charges it conservatively for each request, even when shared.
    // This metadata call must not block, enqueue work, or change the extent.
    virtual std::size_t host_bytes() const noexcept = 0;
};
struct RequestOptions {
    SamplingOptions sampling;
    // Unset inherits ModelInfo.stop_tokens; an empty vector disables token stops.
    std::optional<std::vector<Token>> stop_tokens;
    std::shared_ptr<const RequestContext> context;
    // Generation log-probabilities. Unset requests none; n returns each token's own
    // log-probability plus its n most likely alternatives. Requires
    // ModelInfo::token_logprobs and n <= ModelInfo::max_top_logprobs.
    std::optional<std::uint32_t> logprobs = {};
    // Text stops matched on the decoded output (at most 16 non-empty strings of at
    // most 256 bytes). Requires an independent text frontend. A match ends the
    // request with Finish::stop after the token that completed it; Result::text
    // excludes the stop text.
    std::vector<std::string> stop_strings = {};
    // Decode accepted tokens into Result::text and OutputEvent::text. Implied by
    // non-empty stop_strings; requires an independent text frontend.
    bool text = false;
};
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
};
// A token and its natural-log probability under the raw logits of the pass that
// produced or scored the position: before temperature, penalties, bias and filters,
// normalized over the valid vocabulary. Rows of alternatives are ordered by
// log-probability descending, then token id ascending.
struct TokenLogprob { Token token = 0; float logprob = 0; };
// Query scoring (for example against a reference's top-k tokens): `ids` holds `per_target` token ids for each
// scored target, row-major in target order; per_target zero with no ids requests target log-probabilities only.
struct ScoreQueries {
    std::vector<Token> ids;
    std::uint32_t per_target = 0;
    // The model's own top_k alternatives per target; requires ModelInfo::max_top_logprobs >= top_k.
    std::uint32_t top_k = 0;
    // One of ModelInfo::score_paths; empty selects the default path.
    std::string path = {};
};
// One natural-log probability per target, targets x per_target query log-probabilities (finite or negative
// infinity) and, optionally, the model's argmax per target (empty when not provided).
struct QueryScores {
    std::vector<float> target;
    std::vector<float> queries;
    std::vector<Token> argmax;
    // targets x top_k ranked alternatives when ScoreQueries::top_k is nonzero.
    std::vector<TokenLogprob> top = {};
};
class Backend {
public:
    virtual ~Backend() = default;
    // Called once before model construction. Owners may retain a copy of the ledger.
    virtual void initialize(Resources&) {}
    virtual void bind_thread() = 0;
    // Settles all backend work submitted by model/session calls, including failures.
    virtual void drain() = 0;
};
struct Services { Resources& resources; Backend& backend; };
// Failed backend drains retain the complete owner rather than freeing live storage.
// A permanently failed backend requires process recovery; this counter is diagnostic.
std::size_t quarantined_engines() noexcept;
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
// These identities are logical, never physical allocation or graph-cache keys.
// All callback methods run on the one engine worker. They must return; deadlines
// cannot preempt a stuck extension callback. No callback may call this Engine.
struct RequestKey {
    std::uint64_t request = 0, generation = 0;
    std::uint32_t lane = 0;
    bool operator==(const RequestKey&) const = default;
};
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
class ConcurrentSession {
public:
    virtual ~ConcurrentSession() = default;
    // Only after a successful retirement receipt or quiesce + backend drain.
    // Must not enqueue, throw, or retain borrowed prompt views.
    virtual void close(Finish) noexcept = 0;
};
struct RequestView {
    RequestKey key;
    std::span<const Token> prompt;
    std::uint32_t output_tokens;
    // Always non-null when supplied by Core; same lifetime as the prompt view.
    const RequestOptions* options = nullptr;
};
enum class Admission { ready, deferred, infeasible };
struct SchedulingHint {
    // Zero cost is unknown. Include restoration and remaining computation.
    std::uint64_t remaining_ns = 0;
    std::uint32_t reusable_tokens = 0;
    // Nonzero equal keys suggest a shared pending prefix. Advisory only: they
    // never authorize state reuse. The engine defines compatible namespaces.
    std::uint64_t prefix_affinity = 0;
    bool prefix_pending = false;
};
struct WorkItem {
    RequestKey key;
    ConcurrentSession* session;
    WorkBudget budget;
};
struct RowProgress { RequestKey key; Progress progress; };
struct CommitDecision {
    RequestKey key;
    ConcurrentSession* session;
    std::uint32_t accepted;
    bool terminal, cancelled;
};
class Retirement {
public:
    virtual ~Retirement() = default;
    // Nonblocking certificate: every session reader AND host producer has settled.
    // False preserves the occupied lane. An exception poisons the whole engine.
    virtual bool ready() = 0;
};
class BatchExecution {
public:
    using Notify = std::function<void()>;
    virtual ~BatchExecution() = default;
    // Core retains the operation and its immutable WorkItems before submit.
    // A throwing/ambiguous submission stays retained through quiesce/drain.
    // Call notify after ready() can observe settlement, including failed work.
    // It is safe on any thread, including before submit returns. It only wakes
    // Core; it neither publishes output nor certifies readiness. Quiesce and
    // destruction must join producers, including any outstanding notification.
    virtual void submit(Notify notify) = 0;
    // Nonblocking; true proves all submitted operation readers/producers settled.
    virtual bool ready() = 0;
    virtual std::vector<RowProgress> progress() = 0;
    virtual std::span<const Token> ready_tokens(const RequestKey&) { return {}; }
    virtual std::span<const DeviceTiming> device_timings() { return {}; }
    // No output is published if commit throws. Transfer any owners needed by
    // newly enqueued state work into the sessions; operation storage may retire.
    virtual void commit(std::span<const CommitDecision>) = 0;
    virtual void quiesce() = 0; // Idempotently stop/join its host producers.
};
class ConcurrentModel : public Model {
public:
    std::unique_ptr<Session> open(std::span<const Token>) final;
    // Inspection only; no submissions or provisional allocations. Full request
    // memory entitlement is extension policy, reserved by open before execution.
    virtual Admission admission(const RequestView&) { return Admission::ready; }
    // Bounded, host-only inspection; no work submission, waits or retained views.
    virtual SchedulingHint scheduling_hint(const RequestView&) { return {}; }
    virtual std::uint32_t prefill_limit(const WorkItem&, const PrefillWindow&) { return 0; }
    virtual std::unique_ptr<ConcurrentSession> open(const RequestView&) = 0;
    virtual RowProgress prefill(const WorkItem&) = 0;
    // Exactly one row per compact WorkItem, in any order; variable candidates per
    // row allow ordinary, batched speculative, or solo-specific strategies.
    virtual std::vector<RowProgress> decode(std::span<const WorkItem>) = 0;
    // Optional mixed prefill/decode group. Exactly one progress row per item;
    // all grants are ceilings. One-token grants must make progress. Returned
    // storage/readiness/commit obey the same group contract as decode().
    virtual std::vector<RowProgress> execute(std::span<const WorkItem>);
    // Prepare ownership only, without launching work. Different in-flight
    // operations use disjoint lanes and private metadata/output/scratch.
    virtual std::unique_ptr<BatchExecution> prepare_batch(std::span<const WorkItem>);
    // Timings for the last decode()/prefill()/execute() group, for every row key.
    // Same calling and lifetime rules as Session::device_timings().
    virtual std::span<const DeviceTiming> device_timings() { return {}; }
    // Final tokens of a row whose RowProgress set deferred_tokens; called once per
    // such row after every row's readiness, before commit.
    virtual std::span<const Token> ready_tokens(const RequestKey&);
    // All borrowed rows are copied before this call. No output is published until
    // it returns successfully. A throw need not roll back model state: it causes
    // whole-engine quiescence/failure, with no publication from this group.
    virtual void commit(std::span<const CommitDecision>) = 0;
    virtual std::unique_ptr<Retirement> retire(RequestKey, ConcurrentSession&, Finish) = 0;
    // Idempotently stop/join producers, preventing all subsequent submissions.
    // Called before final/failure backend drain; throwing retains complete owners.
    virtual void quiesce() = 0;
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
};
}
