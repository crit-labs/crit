#include "crit/native_engine.h"
#include "crit/text.h"
#include "trace.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <limits>
#include <list>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>
#include <utility>

namespace crit::core {
namespace {
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
std::uint64_t elapsed_ns(Clock::time_point from, Clock::time_point to) {
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
constexpr std::size_t logit_bias_limit = 1024;
// Log-probabilities may exceed zero only by accumulated rounding.
constexpr float logprob_tolerance = 1e-3F;
std::uint32_t valid_vocabulary(const ModelInfo& info) {
    return info.valid_vocabulary ? info.valid_vocabulary : info.vocabulary;
}
// Valid-vocabulary checks pass a message that names the valid vocabulary.
void validate_tokens(std::span<const Token> tokens, std::uint32_t vocabulary,
                     const char* message = "token outside declared vocabulary") {
    for(auto token:tokens) if(token < 0 || std::uint32_t(token) >= vocabulary)
        throw std::invalid_argument(message);
}
// Shape only, before materialization; the caller owns readiness ordering.
void validate_deferred_progress(const Progress& p, const char* message) {
    if(p.deferred_tokens && (!p.readiness || !p.tokens.empty() || !p.prefill_complete))
        throw std::runtime_error(message);
}
// Shared serial/batched contract after readiness and deferred materialization.
// This only validates: frontier changes, mixed work accounting and publication stay with the caller.
void validate_progress(const Progress& p, const WorkBudget& budget, std::uint32_t frontier,
                       const Request& request, const char* message) {
    const auto next_frontier = frontier + std::uint64_t(p.prompt_tokens);
    if(p.reused_prompt_tokens > p.prompt_tokens || p.prompt_tokens - p.reused_prompt_tokens > budget.prefill_tokens ||
       (p.reused_prompt_tokens && !request.phases.reuse_allowed()) ||
       next_frontier > request.prompt.size() || (budget.prefill_complete && p.prompt_tokens) ||
       (budget.prefill_complete && !p.prefill_complete) ||
       p.prefill_complete != (next_frontier == request.prompt.size()) ||
       (!p.prefill_complete && !p.tokens.empty()) || p.tokens.size() > budget.output_tokens ||
       (!p.prompt_tokens && p.tokens.empty()))
        throw std::runtime_error(message);
}
bool has_sampling(const SamplingOptions& s) {
    return s.temperature || s.top_p || s.top_k || s.seed || s.min_p || s.repetition_penalty || s.frequency_penalty ||
           s.presence_penalty || !s.logit_bias.empty() || s.min_tokens;
}
bool has_generation_options(const RequestOptions& o) {
    return has_sampling(o.sampling) || o.stop_tokens || o.logprobs || !o.stop_strings.empty() || o.text;
}
void validate_penalty(const std::optional<double>& value, const char* message) {
    if(value && (!std::isfinite(*value) || *value < -2 || *value > 2)) throw std::invalid_argument(message);
}
void validate_options(const RequestOptions& options, const ModelInfo& info, bool scoring, bool frontend) {
    const auto& s = options.sampling;
    const auto& support = info.sampling;
    if(scoring && has_generation_options(options))
        throw std::invalid_argument("scoring does not accept generation options");
    const std::pair<bool,const char*> unsupported[]{
        {s.temperature && !support.temperature,"temperature"},{s.top_p && !support.top_p,"top_p"},
        {s.top_k && !support.top_k,"top_k"},{s.seed && !support.seed,"seed"},{s.min_p && !support.min_p,"min_p"},
        {s.repetition_penalty && !support.penalties,"repetition_penalty"},
        {s.frequency_penalty && !support.penalties,"frequency_penalty"},
        {s.presence_penalty && !support.penalties,"presence_penalty"},
        {!s.logit_bias.empty() && !support.logit_bias,"logit_bias"},{s.min_tokens && !support.min_tokens,"min_tokens"}};
    for(const auto& [missing,field]:unsupported)
        if(missing) throw std::invalid_argument(std::string("model does not support a requested sampling option: ")+field);
    if(s.temperature && (!std::isfinite(*s.temperature) || *s.temperature < 0))
        throw std::invalid_argument("temperature must be finite and nonnegative");
    if(s.top_p && (!std::isfinite(*s.top_p) || *s.top_p <= 0 || *s.top_p > 1))
        throw std::invalid_argument("top_p must be finite and in (0, 1]");
    if(s.top_k && *s.top_k > info.vocabulary)
        throw std::invalid_argument("top_k exceeds vocabulary");
    if(s.min_p && (!std::isfinite(*s.min_p) || *s.min_p < 0 || *s.min_p > 1))
        throw std::invalid_argument("min_p must be finite and in [0, 1]");
    if(s.repetition_penalty && (!std::isfinite(*s.repetition_penalty) || *s.repetition_penalty <= 0))
        throw std::invalid_argument("repetition_penalty must be finite and positive");
    validate_penalty(s.frequency_penalty,"frequency_penalty must be finite and in [-2, 2]");
    validate_penalty(s.presence_penalty,"presence_penalty must be finite and in [-2, 2]");
    if(s.logit_bias.size() > logit_bias_limit) throw std::invalid_argument("logit_bias exceeds 1024 entries");
    if(!s.logit_bias.empty()) {
        // Padding ids are never tokens, so a bias could only promote a non-token.
        std::vector<Token> ids;
        ids.reserve(s.logit_bias.size());
        for(const auto& entry:s.logit_bias) {
            if(entry.token < 0 || std::uint32_t(entry.token) >= valid_vocabulary(info))
                throw std::invalid_argument("logit_bias token outside the valid vocabulary");
            if(std::isnan(entry.bias) || entry.bias == std::numeric_limits<float>::infinity())
                throw std::invalid_argument("logit_bias must be finite or negative infinity");
            ids.push_back(entry.token);
        }
        std::sort(ids.begin(),ids.end());
        if(std::adjacent_find(ids.begin(),ids.end()) != ids.end()) throw std::invalid_argument("logit_bias tokens must be unique");
    }
    if(options.stop_tokens) validate_tokens(*options.stop_tokens,info.vocabulary);
    if(options.context && !info.request_context)
        throw std::invalid_argument("model does not support request context");
    if(options.logprobs && !info.token_logprobs) throw std::invalid_argument("model does not support generation logprobs");
    if(options.logprobs && *options.logprobs > info.max_top_logprobs) throw std::invalid_argument("logprobs exceeds max_top_logprobs");
    if(options.stop_strings.size() > stop_strings_limit) throw std::invalid_argument("stop_strings exceeds 16 entries");
    for(const auto& stop:options.stop_strings) {
        if(stop.empty()) throw std::invalid_argument("stop_strings entries must be nonempty");
        if(stop.size() > stop_string_bytes_limit) throw std::invalid_argument("stop_strings entry exceeds 256 bytes");
        if(!valid_utf8(stop)) throw std::invalid_argument("stop_strings entry is not valid utf-8");
    }
    if(!options.stop_strings.empty() && !frontend) throw std::invalid_argument("stop strings require an independent text frontend");
    if(options.text && !frontend) throw std::invalid_argument("text output requires an independent text frontend");
}
void charge_bytes(std::size_t& total, std::size_t count, std::size_t width = 1) {
    if(width && count > (std::numeric_limits<std::size_t>::max()-total)/width)
        throw std::overflow_error("request storage accounting overflow");
    total += count*width;
}
// Generation log-probabilities: a value never NaN or +inf and at most zero within the tolerance (-inf is a
// zero probability). Ranked rows of `width` alternatives hold distinct valid ids in non-increasing order.
bool logprob_value(float x) { return !std::isnan(x) && x <= logprob_tolerance; }
void validate_ranked(std::span<const TokenLogprob> rows, std::size_t width, std::uint32_t valid) {
    std::array<Token,top_logprobs_limit> ids{};
    for(std::size_t start = 0; width && start < rows.size(); start += width) {
        const auto row = rows.subspan(start,width);
        for(std::size_t k = 0; k < width; ++k) {
            if(!logprob_value(row[k].logprob)) throw std::runtime_error("logprob value is nan, infinite or positive");
            if(row[k].token < 0 || std::uint32_t(row[k].token) >= valid)
                throw std::runtime_error("logprob row token outside the valid vocabulary");
            if(k && row[k].logprob > row[k-1].logprob) throw std::runtime_error("top logprob row is not ranked");
            ids[k] = row[k].token;
        }
        std::sort(ids.begin(),ids.begin()+std::ptrdiff_t(width));
        if(std::adjacent_find(ids.begin(),ids.begin()+std::ptrdiff_t(width)) != ids.begin()+std::ptrdiff_t(width))
            throw std::runtime_error("top logprob row ids are not distinct");
    }
}
// The requested rows of one step's tokens (Progress::tokens, or the final span of a deferred round). Counts are
// exact; deferred rounds may supply spans covering at least the final count, of which Core reads the first rows.
struct LogprobRows { std::span<const float> chosen; std::span<const TokenLogprob> top; std::size_t width = 0; };
LogprobRows logprob_rows(const RequestOptions& options, const Progress& progress, std::uint32_t valid) {
    if(!options.logprobs) return {};
    const std::size_t width = *options.logprobs, count = progress.tokens.size();
    const bool counts = progress.deferred_tokens
        ? progress.logprobs.size() >= count && progress.top_logprobs.size() >= count*width
        : progress.logprobs.size() == count && progress.top_logprobs.size() == count*width;
    if(!counts) throw std::runtime_error("logprob rows count mismatch");
    LogprobRows rows{progress.logprobs.first(count),progress.top_logprobs.first(count*width),width};
    if(!std::all_of(rows.chosen.begin(),rows.chosen.end(),logprob_value))
        throw std::runtime_error("logprob value is nan, infinite or positive");
    validate_ranked(rows.top,width,valid);
    return rows;
}
// Released text of one step and, per decoded accepted token, its end offset in `bytes`.
struct StepText { std::string bytes; std::vector<std::uint32_t> ends; };
struct Decision { std::uint32_t accepted = 0; Finish finish = Finish::length; bool terminal = false; };
// The one trim rule for both paths: stop tokens (kept, never decoded), the output budget, then stop strings over
// the accepted tokens' text. A terminal step flushes the held text into the last accepted token's piece; a stop
// that completes in that tail still ends with Finish::stop. Text exceptions propagate to the caller.
Decision decide(std::span<const Token> tokens, std::span<const Token> stops, std::uint32_t remaining,
                TextStream* text, StepText& out) {
    Decision d{std::uint32_t(tokens.size())};
    for(std::uint32_t i = 0; i < d.accepted; ++i) if(std::find(stops.begin(),stops.end(),tokens[i]) != stops.end()) {
        d.accepted = i+1; d.finish = Finish::stop; break;
    }
    d.terminal = d.finish == Finish::stop || d.accepted == remaining;
    if(!text || !d.accepted) return d;
    const auto decoded = d.finish == Finish::stop ? d.accepted-1 : d.accepted;
    const auto offset = [&]{
        if(out.bytes.size() > std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error("text offsets exceed 32 bits");
        return std::uint32_t(out.bytes.size());
    };
    for(std::uint32_t i = 0; i < decoded; ++i) {
        const auto piece = text->push(tokens.subspan(i,1));
        out.bytes += piece.text;
        out.ends.push_back(offset());
        if(piece.stopped) return {i+1,Finish::stop,true};
    }
    if(d.terminal) {
        const auto tail = text->finish();
        out.bytes += tail.text;
        if(tail.stopped) d.finish = Finish::stop;
        if(decoded < d.accepted) out.ends.push_back(offset()); else out.ends.back() = offset();
    }
    return d;
}
// Result extents before staging one step, restored when its commit fails.
struct Extents { std::size_t tokens = 0, logprobs = 0, top = 0, text = 0, text_ends = 0; };
Extents extents(const Result& r) {
    return {r.tokens.size(),r.logprobs.size(),r.top_logprobs.size(),r.text.size(),r.text_ends.size()};
}
void restore(Result& r, const Extents& e) {
    r.tokens.resize(e.tokens); r.logprobs.resize(e.logprobs); r.top_logprobs.resize(e.top);
    r.text.resize(e.text); r.text_ends.resize(e.text_ends);
}
// Grows result.text's charged capacity to `needed` bytes in doubling chunks of at least 256 bytes, or by the exact
// need near the ceiling. Call with the request lock held.
void reserve_text(Request& r, Resources& resources, std::size_t needed) {
    if(needed > std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error("text offsets exceed 32 bits");
    if(needed <= r.text_charged) return;
    const auto need = needed-r.text_charged, chunk = std::max({need,r.text_charged,std::size_t(256)});
    Reservation grant;
    try { grant = resources.reserve(Memory::host,chunk); }
    catch(const std::runtime_error&) { if(chunk == need) throw; grant = resources.reserve(Memory::host,need); }
    r.text_storage.push_back(std::move(grant));
    r.text_charged += r.text_storage.back().bytes();
    r.result.text.reserve(r.text_charged);
}
// Appends one step's accepted rows and text, copied before commit can invalidate borrowed rows. Lock held;
// nothing is appended when the text reservation fails.
void stage_output(Request& r, Resources& resources, const LogprobRows& rows, std::uint32_t accepted, const StepText& text) {
    auto& result = r.result;
    if(r.text) {
        reserve_text(r,resources,result.text.size()+text.bytes.size());
        const auto base = std::uint32_t(result.text.size());
        result.text += text.bytes;
        for(const auto end:text.ends) result.text_ends.push_back(base+end);
    }
    if(r.options.logprobs) {
        result.logprobs.insert(result.logprobs.end(),rows.chosen.begin(),rows.chosen.begin()+accepted);
        result.top_logprobs.insert(result.top_logprobs.end(),rows.top.begin(),rows.top.begin()+std::ptrdiff_t(accepted*rows.width));
    }
}
std::atomic<std::size_t> quarantined_count{0};
class OwnedTextDecoder final : public TextDecoder {
    // Destroy the decoder before the frontend whose tables it may borrow.
    std::shared_ptr<const TextFrontend> owner_;
    std::unique_ptr<TextDecoder> decoder_;
    std::uint32_t vocabulary_;
    bool closed_ = false;
public:
    OwnedTextDecoder(std::shared_ptr<const TextFrontend> owner,
                     std::unique_ptr<TextDecoder> decoder, std::uint32_t vocabulary)
        : owner_(std::move(owner)), decoder_(std::move(decoder)), vocabulary_(vocabulary) {}
    std::string push(std::span<const Token> tokens) override {
        if(closed_) throw std::logic_error("text decoder is closed");
        validate_tokens(tokens,vocabulary_);
        try { return decoder_->push(tokens); }
        catch(...) { closed_ = true; throw; }
    }
    std::string finish() override {
        if(closed_) throw std::logic_error("text decoder is closed");
        closed_ = true;
        return decoder_->finish();
    }
};
// The frontend's own decoder, or offset_decoder seeded with the prompt tail when the frontend keeps the default
// create_decoder() (which throws exactly std::logic_error). Other failures and a null decoder propagate.
std::unique_ptr<TextStream> make_text_stream(const std::shared_ptr<const TextFrontend>& frontend, std::span<const Token> prompt,
                                             const std::vector<std::string>& stops, std::uint32_t vocabulary) {
    std::unique_ptr<TextDecoder> decoder;
    try { decoder = frontend->create_decoder(); }
    catch(const std::logic_error& e) {
        if(typeid(e) != typeid(std::logic_error)) throw;
        decoder = offset_decoder(frontend,prompt);
    }
    if(!decoder) throw std::runtime_error("text frontend returned null decoder");
    return std::make_unique<TextStream>(std::make_unique<OwnedTextDecoder>(frontend,std::move(decoder),vocabulary),stops);
}
}
std::size_t quarantined_engines() noexcept { return quarantined_count.load(); }
std::unique_ptr<TextDecoder> TextFrontend::create_decoder() const {
    throw std::logic_error("text frontend lacks incremental decoding");
}
std::unique_ptr<Session> Model::open_request(std::span<const Token> prompt, const RequestOptions& options) {
    if(has_sampling(options.sampling) || options.context || options.logprobs || !options.stop_strings.empty() || options.text)
        throw std::logic_error("model must implement open_request for adapter options");
    return open(prompt);
}
std::vector<float> Model::score(std::span<const Token>, std::uint32_t) {
    throw std::logic_error("model does not implement scoring");
}
std::vector<float> Model::score_request(std::span<const Token> prompt, std::uint32_t first, const RequestOptions& options) {
    if(has_generation_options(options) || options.context)
        throw std::logic_error("model must implement score_request for adapter options");
    return score(prompt,first);
}
QueryScores Model::score_queries(std::span<const Token>, std::uint32_t, const ScoreQueries&, const RequestOptions&) {
    throw std::logic_error("model declares query_scoring but does not implement score_queries");
}
std::vector<Token> Model::tokenize(std::string_view) {
    throw std::logic_error("model lacks legacy tokenizer callbacks");
}
std::string Model::detokenize(std::span<const Token>) {
    throw std::logic_error("model lacks legacy tokenizer callbacks");
}
std::unique_ptr<Session> ConcurrentModel::open(std::span<const Token>) {
    throw std::logic_error("concurrent model requires identified request admission");
}
std::span<const Token> Session::ready_tokens() {
    throw std::logic_error("session returned deferred_tokens without ready_tokens");
}
std::span<const Token> ConcurrentModel::ready_tokens(const RequestKey&) {
    throw std::logic_error("concurrent model returned deferred_tokens without ready_tokens");
}
std::vector<RowProgress> ConcurrentModel::execute(std::span<const WorkItem>) {
    throw std::logic_error("mixed_batches model must implement execute");
}
std::unique_ptr<BatchExecution> ConcurrentModel::prepare_batch(std::span<const WorkItem>) {
    throw std::logic_error("async_batches model must implement prepare_batch");
}
struct Engine::Impl : std::enable_shared_from_this<Engine::Impl> {
    struct Wakeup {
        std::mutex mutex;
        std::condition_variable cv;
        bool changed = false;
        void notify() {
            { std::lock_guard lock(mutex); changed = true; }
            cv.notify_one();
        }
    };
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
        std::uint32_t frontier = 0;
        std::uint32_t prefill_wait_steps = 0;
        std::uint64_t prefix_affinity = 0;
        bool prefilled = false, retiring = false, executing = false;
        Finish finish = Finish::length;
        std::chrono::steady_clock::time_point retirement_started;
        std::uint64_t trace_retirement = 0, trace_started = 0;
    };
    std::vector<Lane> lanes;
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
    Impl(Profile p, std::shared_ptr<Backend> b):profile(p),resources(p.host_limit,p.device_limit),
        trace_storage(resources.reserve(Memory::host,p.trace.capacity <= 65536 ? p.trace.capacity*sizeof(TraceEvent) : 0)),
        trace_buffer(p.trace),
        operator_storage(resources.reserve(Memory::host,p.trace.operators <= 65536 ? p.trace.operators*sizeof(trace::Range) : 0)),
        operator_ranges(make_operator_ranges(p.trace,trace_buffer)),backend(std::move(b)) {
        if(!p.context || !p.prefill_chunk || p.prefill_chunk > p.context || !p.host_limit || !p.device_limit || !backend ||
           !p.max_concurrency || p.max_concurrency > max_request_concurrency || p.max_pending > 1024 ||
           !p.pending_timeout_ms || !p.retirement_timeout_ms || !p.prefill_starvation_steps ||
           !p.max_admission_bypasses || !p.admission_starvation_ms || !p.batch_timeout_ms || p.max_inflight_batches>64 ||
           (p.max_inflight_batches && !p.max_batch_tokens) || p.admission_lookahead > max_request_concurrency + 1024 ||
           (p.admission_policy != AdmissionPolicy::fifo && p.admission_policy != AdmissionPolicy::reuse_aware) ||
           (p.batch_policy != BatchPolicy::fair && p.batch_policy != BatchPolicy::decode_first) ||
           (p.batch_policy == BatchPolicy::decode_first && !p.max_batch_tokens))
            throw std::invalid_argument("invalid core profile/backend/factory");
    }
    static std::unique_ptr<trace::Recorder> make_operator_ranges(const TraceConfig& config,const detail::TraceBuffer& lifecycle) {
        if(config.operators > 65536) throw std::invalid_argument("invalid trace operator capacity");
        return config.operators ? std::make_unique<trace::Recorder>(config.operators,lifecycle.epoch()) : nullptr;
    }
    // The engine's recorder while operator ranges are enabled; otherwise the thread keeps its own sink.
    trace::Sink* operator_sink_here() const noexcept {
        return operator_ranges ? operator_ranges.get() : trace::current_sink();
    }
    void drain_backend(RequestKey key = {}) {
        detail::TraceScope span(trace_buffer,TracePhase::drain,key);
        backend->drain();
    }
    void quiesce_model() {
        if(concurrent) {
            detail::TraceScope span(trace_buffer,TracePhase::quiesce);
            std::exception_ptr failure;
            for(auto& batch:batches) if(batch.execution)try { batch.execution->quiesce(); } catch(...) { if(!failure)failure=std::current_exception(); }
            try { concurrent->quiesce(); } catch(...) { if(!failure)failure=std::current_exception(); }
            if(failure)std::rethrow_exception(failure);
        }
    }
    void initialize(Factory factory) {
        if(!factory) throw std::invalid_argument("null model factory");
        const trace::SinkBinding operator_sink(operator_sink_here());
        backend->initialize(resources);
        backend->bind_thread();
        model = factory(profile,{resources,*backend});
        if(!model) throw std::invalid_argument("model factory returned null");
        concurrent = dynamic_cast<ConcurrentModel*>(model.get());
        drain_backend();
        declaration = model->info();
        if(declaration.implementation.empty() || !declaration.vocabulary || declaration.context < profile.context || declaration.prefill_chunk < profile.prefill_chunk)
            throw std::invalid_argument("model declaration does not cover core profile");
        if(declaration.query_scoring && !declaration.scoring)
            throw std::invalid_argument("query scoring requires the scoring capability");
        if(declaration.valid_vocabulary > declaration.vocabulary)
            throw std::invalid_argument("valid_vocabulary exceeds vocabulary");
        if(declaration.max_top_logprobs > top_logprobs_limit || declaration.max_top_logprobs > valid_vocabulary(declaration))
            throw std::invalid_argument("max_top_logprobs exceeds top_logprobs_limit or the valid vocabulary");
        if(declaration.token_logprobs && !declaration.generation)
            throw std::invalid_argument("token_logprobs requires generation");
        if(declaration.max_top_logprobs && !declaration.token_logprobs && !declaration.query_scoring)
            throw std::invalid_argument("max_top_logprobs requires token_logprobs or query_scoring");
        auto paths = declaration.score_paths;
        std::sort(paths.begin(),paths.end());
        if(std::any_of(paths.begin(),paths.end(),[](const std::string& path){return path.empty();}) ||
           std::adjacent_find(paths.begin(),paths.end()) != paths.end())
            throw std::invalid_argument("score paths must be nonempty and unique");
        if((declaration.mixed_batches && (!concurrent || !declaration.decode_token_work)) ||
           (profile.max_batch_tokens && !declaration.mixed_batches))
            throw std::invalid_argument("mixed token work requires a capable concurrent model");
        validate_tokens(declaration.stop_tokens,declaration.vocabulary);
        if(profile.admission_policy == AdmissionPolicy::reuse_aware && (!concurrent || !declaration.scheduling_hints))
            throw std::invalid_argument("reuse-aware admission requires scheduling hints");
        if(profile.prefill_target_ns && !declaration.prefill_advice)
            throw std::invalid_argument("latency chunk sizing requires prefill advice");
        if(profile.max_inflight_batches && (!concurrent || !declaration.async_batches || !declaration.mixed_batches))
            throw std::invalid_argument("execution pipeline requires async mixed batches");
        frontend = model->text_frontend();
        if(!declaration.max_concurrency || declaration.max_concurrency > max_request_concurrency || declaration.max_concurrency < profile.max_concurrency ||
           (!concurrent && (profile.max_concurrency != 1 || profile.max_pending)))
            throw std::invalid_argument("model concurrency declaration does not cover core profile");
        if(concurrent) {
            lanes.resize(profile.max_concurrency); generations.resize(profile.max_concurrency);
            row_lookup.resize(profile.max_concurrency);
        }
        worker = std::thread([this]{run();});
    }
    bool shutdown() noexcept {
        {
            std::lock_guard lock(wakeup->mutex);
            stopping = true;
            wakeup->changed = true;
            if(pending) *pending->cancelled = true;
        }
        wakeup->cv.notify_all();
        if(worker.joinable()) worker.join();
        const trace::SinkBinding operator_sink(operator_sink_here());
        try { backend->bind_thread(); quiesce_model(); drain_backend(); } catch(...) {return false;}
        batches.clear(); // Release operation owners before their borrowed sessions.
        for(auto& lane:lanes) {
            lane.receipt.reset();
            if(lane.session) lane.session->close(Finish::failed);
            lane.session.reset();
            lane = Lane{};
        }
        waiting.clear();
        active_score.reset();
        if(retained_session) {retained_session->close(Finish::failed); retained_session.reset();}
        retained_request.reset();
        model.reset();
        return true;
    }
    Handle submit(std::vector<Token> prompt, std::uint32_t output, bool scoring, std::uint32_t first, RequestOptions,
                  std::optional<ScoreQueries> queries = std::nullopt);
    std::span<const Token> stops(const Request& r) const {
        return r.options.stop_tokens ? std::span<const Token>(*r.options.stop_tokens) : std::span<const Token>(declaration.stop_tokens);
    }
    // make_exception_ptr copies its argument, and the copy shares the argument's message. Building the error here
    // destroys the argument before complete() publishes the copy, so afterwards only consumers touch the message.
    static std::exception_ptr runtime_failure(const char* message) {
        return std::make_exception_ptr(std::runtime_error(message));
    }
    // Publishes completion, exactly once per request; this is the worker's last access to the request's owners. From
    // here they belong to the consumer, who releases them before wait() returns or the handle is destroyed
    // (Handle::Impl::settle), unless `retained` marks a request the quarantined runtime keeps for unsettled readers.
    // Callers move their reference to `error` in, so the consumer, not the worker, normally destroys the exception.
    void complete(const std::shared_ptr<Request>& r, Finish finish, std::exception_ptr error = {}, bool retained = false) noexcept {
        trace_buffer.point(TracePhase::completion,r->trace_key,finish);
        r->text_stream.reset(); // Worker-private; the published text stays in the result.
        {
            std::lock_guard lock(r->mutex);
            // Staging is worker-private and no longer needed after retirement.
            std::vector<Token>().swap(r->staging);
            r->staging_storage = {};
            r->result.finish = finish; r->error = std::move(error); r->retained = retained; r->done = true;
            const auto now = Clock::now();
            auto& p = r->phases;
            auto& t = r->result.timing;
            t.queued_ns = elapsed_ns(r->submitted, p.has_admitted ? p.admitted : now);
            if(p.has_admitted) {
                t.total_ns = elapsed_ns(p.admitted, now);
                if(p.has_prefilled) t.prefill_ns = elapsed_ns(p.admitted, p.prefilled);
                if(p.has_first) t.first_token_ns = elapsed_ns(p.admitted, p.first);
                if(p.has_prefilled && p.has_first) t.decode_ns = elapsed_ns(p.prefilled, p.last);
            }
            t.prefill_tokens = p.prefill_tokens; t.reused_prompt_tokens = p.reused_tokens; t.decode_steps = p.decode_steps;
        }
        r->cv.notify_all();
    }
    void begin_retirement(Lane& lane, Finish finish) {
        lane.finish = finish;
        lane.retirement_started = std::chrono::steady_clock::now();
        if(trace_buffer.enabled()) {
            lane.trace_retirement=trace_buffer.operation(); lane.trace_started=trace_buffer.now();
            trace_buffer.write({0,lane.trace_retirement,lane.trace_started,lane.key,TracePhase::retirement,TraceMark::begin});
        }
        lane.receipt = concurrent->retire(lane.key,*lane.session,finish);
        if(!lane.receipt) throw std::runtime_error("null retirement receipt");
        lane.retiring = true;
    }
    void poll_retirements() {
        for(auto& lane:lanes) if(lane.request) {
            if(!lane.retiring && !lane.executing && *lane.request->cancelled) begin_retirement(lane,Finish::cancelled);
            if(!lane.retiring) continue;
            if(lane.receipt->ready()) {
                // Receipt destruction can reference the session; release it first.
                lane.receipt.reset(); lane.session->close(lane.finish); lane.session.reset();
                if(trace_buffer.enabled()) {
                    const auto now=trace_buffer.now();
                    trace_buffer.write({0,lane.trace_retirement,now,lane.key,TracePhase::retirement,TraceMark::end,now-lane.trace_started});
                }
                const auto request=lane.request; const auto finish=lane.finish;
                lane = Lane{};
                {
                    std::lock_guard lock(wakeup->mutex);
                    busy=!waiting.empty() || std::any_of(lanes.begin(),lanes.end(),[](const auto& value){return bool(value.request);});
                }
                complete(request,finish,std::move(request->failure));
            } else if(std::chrono::steady_clock::now()-lane.retirement_started >= std::chrono::milliseconds(profile.retirement_timeout_ms))
                throw std::runtime_error("retirement deadline expired without settlement");
        }
    }
    bool admit_reuse(std::size_t lane_index, Clock::time_point now) {
        auto& lane=lanes[lane_index];
        if(generations[lane_index]==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("lane generation exhausted");
        std::size_t selected=waiting.size();
        SchedulingHint selected_hint;
        long double best=-1;
        for(std::size_t j=0; j<waiting.size() && j<=profile.admission_lookahead; ++j) {
            const auto& r=waiting[j];
            if(r->scoring) break;
            if(!r->output) { complete(r,Finish::length); waiting.erase(waiting.begin()+std::ptrdiff_t(j)); return true; }
            RequestView view{{r->id,generations[lane_index]+1,std::uint32_t(lane_index)},r->prompt,r->output,&r->options};
            const auto hint=concurrent->scheduling_hint(view);
            if(hint.reusable_tokens>r->prompt.size()) throw std::runtime_error("invalid reusable token estimate");
            const bool forced=r->admission_bypasses>=profile.max_admission_bypasses ||
                now-r->submitted>=std::chrono::milliseconds(profile.admission_starvation_ms);
            const auto admission=concurrent->admission(view);
            if(admission==Admission::infeasible) {
                complete(r,Finish::failed,runtime_failure("request memory entitlement infeasible"));
                waiting.erase(waiting.begin()+std::ptrdiff_t(j)); return true;
            }
            if(admission!=Admission::ready && admission!=Admission::deferred)
                throw std::runtime_error("invalid admission decision");
            const bool shared=hint.prefix_pending || (hint.prefix_affinity && std::any_of(lanes.begin(),lanes.end(),[&](const auto& active) {
                return active.request && !active.retiring && !active.prefilled && active.prefix_affinity==hint.prefix_affinity;
            }));
            if(admission==Admission::ready && (!shared || forced)) {
                const auto cost=hint.remaining_ns ? hint.remaining_ns : std::numeric_limits<std::uint64_t>::max();
                const auto priority=(static_cast<long double>(elapsed_ns(r->submitted,now))+1)/static_cast<long double>(cost);
                if(forced || priority>best) { selected=j; selected_hint=hint; best=priority; }
            }
            if(forced) break; // Never bypass an aged/capped request to reach a follower.
        }
        if(selected==waiting.size()) return false;
        auto r=waiting[selected];
        for(std::size_t j=0;j<selected;++j) ++waiting[j]->admission_bypasses;
        RequestKey key{r->id,++generations[lane_index],std::uint32_t(lane_index)};
        lane.key=key; lane.request=r; lane.prefix_affinity=selected_hint.prefix_affinity;
        r->trace_key=key; waiting.erase(waiting.begin()+std::ptrdiff_t(selected));
        // A hint was only a prediction: open must revalidate and own actual state.
        r->phases.admit(); detail::TraceScope span(trace_buffer,TracePhase::open,key);
        lane.session=concurrent->open({key,r->prompt,r->output,&r->options});
        if(!lane.session) throw std::runtime_error("model opened null concurrent session");
        return true;
    }
    void admit() {
        std::lock_guard lock(wakeup->mutex);
        const auto now = std::chrono::steady_clock::now();
        for(auto it=waiting.begin();it!=waiting.end();) {
            auto r = *it;
            if(*r->cancelled || stopping) {
                complete(r,Finish::cancelled); it = waiting.erase(it);
            } else if(now-r->submitted >= std::chrono::milliseconds(profile.pending_timeout_ms)) {
                complete(r,Finish::failed,runtime_failure("pending request deadline expired"));
                it = waiting.erase(it);
            } else ++it;
        }
        for(std::size_t i=0;i<lanes.size() && !waiting.empty();++i) {
            auto& lane = lanes[i];
            if(lane.request) continue;
            if(profile.admission_policy==AdmissionPolicy::reuse_aware) {
                while(!lane.request && !waiting.empty()) if(!admit_reuse(i,now)) return;
                continue;
            }
            std::vector<std::shared_ptr<Request>> skipped;
            std::size_t index=0;
            for(std::uint32_t inspected=0; index<waiting.size() && inspected<=profile.admission_lookahead; ++inspected) {
                auto r=waiting[index];
                if(r->scoring) break; // Backfill never crosses an exclusive barrier.
                const auto erase=[&]{waiting.erase(waiting.begin()+std::ptrdiff_t(index));};
                if(!r->output) { erase(); complete(r,Finish::length); continue; }
                if(generations[i] == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("lane generation exhausted");
                RequestKey key{r->id,generations[i]+1,std::uint32_t(i)};
                RequestView view{key,r->prompt,r->output,&r->options};
                const auto admission = [&]{
                    detail::TraceScope span(trace_buffer,TracePhase::admission,key);
                    return concurrent->admission(view);
                }();
                if(admission == Admission::deferred) {
                    if(r->admission_bypasses >= profile.max_admission_bypasses || inspected==profile.admission_lookahead) break;
                    skipped.push_back(r); ++index; continue;
                }
                if(admission == Admission::infeasible) {
                    erase();
                    complete(r,Finish::failed,runtime_failure("request memory entitlement infeasible"));
                    continue;
                }
                if(admission != Admission::ready) throw std::runtime_error("invalid admission decision");
                for(auto& earlier:skipped) ++earlier->admission_bypasses;
                // Retain before open: a throwing open may leave live readers.
                lane.key=key; lane.request=r; generations[i]=key.generation;
                r->trace_key=key; erase();
                detail::TraceScope span(trace_buffer,TracePhase::open,key);
                r->phases.admit(); lane.session=concurrent->open(view);
                if(!lane.session) throw std::runtime_error("model opened null concurrent session");
                break;
            }
            // A deferred head/barrier applies equally to all free lanes. Retry
            // on the next worker turn, after active work/retirement can progress.
            if(!lane.request) break;
        }
    }
    WorkItem work(const Lane& lane) const {
        return {lane.key,lane.session.get(),{
            std::min(profile.prefill_chunk,std::uint32_t(lane.request->prompt.size())-lane.frontier),
            lane.request->output-std::uint32_t(lane.request->result.tokens.size()),lane.prefilled}};
    }
    void advise_prefill(WorkItem& item, std::uint32_t decode_work=0) {
        if(!profile.prefill_target_ns || item.budget.prefill_complete || !item.budget.prefill_tokens)return;
        const auto& lane=lanes[item.key.lane];
        const auto limit=concurrent->prefill_limit(item,{lane.frontier,
            std::uint32_t(lane.request->prompt.size())-lane.frontier,item.budget.prefill_tokens,decode_work,profile.prefill_target_ns});
        if(limit>item.budget.prefill_tokens)throw std::runtime_error("prefill advice exceeds granted work");
        if(limit) {
            item.budget.prefill_tokens=limit;
            if(item.budget.token_work)item.budget.token_work=limit;
        }
    }
    void publish_group(std::span<const WorkItem> items, std::vector<RowProgress> rows, std::uint64_t operation, BatchExecution* execution=nullptr) {
        if(items.size()!=rows.size()) throw std::runtime_error("decode row membership count mismatch");
        struct Staged {
            Lane* lane; std::span<const Token> tokens;
            std::uint32_t frontier, reused; bool prefilled; Finish finish;
            Extents before; std::exception_ptr failure;
        };
        std::vector<Staged> staged;
        std::vector<CommitDecision> decisions;
        staged.reserve(items.size()); decisions.reserve(items.size());
        // Reuse the lane-indexed table, clearing it even on validation/fence
        // failures. Validate the whole group before waiting on borrowed rows.
        struct ClearRows {
            std::vector<RowProgress*>& lookup; std::vector<RowProgress>& rows;
            ~ClearRows() { for(const auto& row:rows) if(row.key.lane<lookup.size()) lookup[row.key.lane]=nullptr; }
        } clear{row_lookup,rows};
        for(auto& row:rows) {
            if(row.key.lane>=lanes.size() || lanes[row.key.lane].key!=row.key)
                throw std::runtime_error("stale row identity");
            auto& slot=row_lookup[row.key.lane];
            if(slot) throw std::runtime_error("duplicate row identity");
            slot=&row;
        }
        for(const auto& item:items) if(!row_lookup[item.key.lane] || row_lookup[item.key.lane]->key!=item.key)
            throw std::runtime_error("missing row identity");
        for(const auto& item:items) {
            const auto* row=row_lookup[item.key.lane];
            if(row->progress.readiness) {
                detail::TraceScope span(trace_buffer,TracePhase::readiness,item.key);
                row->progress.readiness->wait();
            }
        }
        for(auto& row:rows) if(row.progress.deferred_tokens) {
            validate_deferred_progress(row.progress,"invalid deferred concurrent progress");
            row.progress.tokens = execution ? execution->ready_tokens(row.key) : concurrent->ready_tokens(row.key);
        }
        if(trace_buffer.enabled()) {
            try {
                for(const auto& timing:execution ? execution->device_timings() : concurrent->device_timings())
                    for(const auto& item:items) trace_buffer.device(operation,item.key,timing);
            } catch(...) { trace_buffer.drop(); }
        }
        // Validate every row before any text decoding, so a contract violation stages nothing.
        std::vector<LogprobRows> logprobs;
        logprobs.reserve(items.size());
        for(const auto& item:items) {
            const auto& p = row_lookup[item.key.lane]->progress;
            auto& lane = lanes[item.key.lane];
            if(item.budget.token_work &&
               (p.token_work > item.budget.token_work ||
                p.reused_prompt_tokens > p.prompt_tokens ||
                p.prompt_tokens - p.reused_prompt_tokens > p.token_work ||
                (lane.prefilled && !p.token_work)))
                throw std::runtime_error("mixed token work exceeds grant or omits computation");
            validate_progress(p,item.budget,lane.frontier,*lane.request,"invalid or stalled concurrent progress");
            validate_tokens(p.tokens,valid_vocabulary(declaration),"generated token outside the valid vocabulary");
            logprobs.push_back(logprob_rows(lane.request->options,p,valid_vocabulary(declaration)));
        }
        // Every borrowed row is materialized before commit invalidates any row. Logprob rows and text go straight
        // into the pre-reserved result, invisible until publication extends result.tokens.
        for(std::size_t i=0;i<items.size();++i) {
            const auto& item = items[i];
            const auto& p = row_lookup[item.key.lane]->progress;
            auto& lane = lanes[item.key.lane];
            auto& request = *lane.request;
            const bool cancelled = *request.cancelled;
            Decision d{0,Finish::cancelled,true};
            Extents before;
            std::exception_ptr failure;
            {
                std::lock_guard lock(request.mutex);
                before = extents(request.result);
            }
            // A text decoding or text storage failure ends only this row: it commits like a cancellation and
            // retires with Finish::failed, reporting the error from wait().
            if(!cancelled) try {
                StepText text;
                d = decide(p.tokens,stops(request),item.budget.output_tokens,request.text_stream.get(),text);
                std::lock_guard lock(request.mutex);
                try { stage_output(request,resources,logprobs[i],d.accepted,text); }
                catch(...) { restore(request.result,before); throw; }
            } catch(...) { failure = std::current_exception(); d = {0,Finish::failed,true}; }
            // Admission has reserved the largest possible accepted row. Reuse
            // it so a later submission cannot consume this group's headroom.
            auto& copied=request.staging;
            copied.clear();
            if(d.accepted) copied.assign(p.tokens.begin(),p.tokens.begin()+d.accepted);
            staged.push_back({&lane,copied,lane.frontier+p.prompt_tokens,p.reused_prompt_tokens,p.prefill_complete,d.finish,before,failure});
            decisions.push_back({item.key,item.session,d.accepted,d.terminal,cancelled || failure});
        }
        try {
            detail::TraceScope span(trace_buffer,TracePhase::commit,items);
            if(execution)execution->commit(decisions); else concurrent->commit(decisions);
        } catch(...) {
            for(const auto& value:staged) {
                std::lock_guard lock(value.lane->request->mutex);
                restore(value.lane->request->result,value.before);
            }
            throw;
        }
        for(std::size_t i=0;i<staged.size();++i) {
            auto& value = staged[i]; auto& lane = *value.lane;
            lane.request->phases.step(value.frontier-lane.frontier,value.reused,lane.prefilled,value.prefilled,value.tokens.size());
            lane.frontier=value.frontier; lane.prefilled=value.prefilled;
            if(value.failure) lane.request->failure = value.failure;
            {
                std::lock_guard lock(lane.request->mutex);
                lane.request->result.tokens.insert(lane.request->result.tokens.end(),value.tokens.begin(),value.tokens.end());
            }
            trace_buffer.point(TracePhase::publication,lane.key);
            lane.request->cv.notify_all();
        }
        for(std::size_t i=0;i<staged.size();++i) if(decisions[i].terminal) begin_retirement(*staged[i].lane,staged[i].finish);
    }
    bool step_concurrent() {
        if(profile.max_inflight_batches && batches.size()>=profile.max_inflight_batches)return false;
        if(profile.max_batch_tokens) return step_mixed();
        std::vector<WorkItem> decode;
        std::size_t prefill = lanes.size();
        for(std::size_t n=0;n<lanes.size();++n) {
            const auto i=(prefill_cursor+n)%lanes.size(); const auto& lane=lanes[i];
            if(!lane.request || lane.retiring) continue;
            if(lane.prefilled) decode.push_back(work(lane));
            else if(prefill==lanes.size()) prefill=i;
        }
        if(!decode.empty() && (prefer_decode || prefill==lanes.size())) {
            detail::TraceScope step(trace_buffer,TracePhase::step,std::span<const WorkItem>(decode));
            auto rows=[&]{
                detail::TraceScope span(trace_buffer,TracePhase::decode,std::span<const WorkItem>(decode));
                return concurrent->decode(decode);
            }();
            publish_group(decode,std::move(rows),step.operation()); prefer_decode=false; return true;
        }
        if(prefill!=lanes.size()) {
            auto item=work(lanes[prefill]); advise_prefill(item);
            detail::TraceScope step(trace_buffer,TracePhase::step,item.key);
            auto row=[&]{
                detail::TraceScope span(trace_buffer,TracePhase::prefill,item.key);
                return concurrent->prefill(item);
            }();
            publish_group(std::span<const WorkItem>(&item,1),{std::move(row)},step.operation());
            prefill_cursor=(prefill+1)%lanes.size(); prefer_decode=true; return true;
        }
        return false;
    }
    bool step_mixed() {
        std::vector<WorkItem> items;
        auto remaining=profile.max_batch_tokens;
        const auto eligible=[](const Lane& lane){return lane.request && !lane.retiring && !lane.executing;};
        const auto add=[&](std::size_t i) {
            auto item=work(lanes[i]); item.budget.token_work=1;
            items.push_back(item); --remaining;
        };
        const auto expand=[&](WorkItem& item) {
            const auto desired=item.budget.prefill_complete ? declaration.decode_token_work : item.budget.prefill_tokens;
            const auto extra=std::min(remaining,desired-item.budget.token_work);
            item.budget.token_work+=extra; remaining-=extra;
        };
        if(profile.batch_policy==BatchPolicy::fair) {
            // Minimum grants before widening; rotate spare capacity fairly.
            for(std::size_t n=0;n<lanes.size() && remaining;++n) {
                const auto i=(prefill_cursor+n)%lanes.size();
                if(eligible(lanes[i])) add(i);
            }
            if(!items.empty()) prefill_cursor=(items.front().key.lane+1)%lanes.size();
            for(auto& item:items) expand(item);
        } else {
            // One oldest prefill receives a minimum grant before decoders once
            // aged. This still makes progress with a one-token group budget.
            auto urgent=lanes.size();
            for(std::size_t n=0;n<lanes.size();++n) {
                const auto i=(prefill_cursor+n)%lanes.size(); const auto& lane=lanes[i];
                if(eligible(lane) && !lane.prefilled && lane.prefill_wait_steps>=profile.prefill_starvation_steps &&
                   (urgent==lanes.size() || lane.prefill_wait_steps>lanes[urgent].prefill_wait_steps)) urgent=i;
            }
            // With only one unit, a sustained queue of aged prefills must not
            // starve decoders in turn. Alternate forced prefill and decode.
            if(remaining==1 && reserved_prefill_last && std::any_of(lanes.begin(),lanes.end(),[&](const auto& lane) {
                return eligible(lane) && lane.prefilled;
            })) urgent=lanes.size();
            reserved_prefill_last=urgent!=lanes.size();
            if(urgent!=lanes.size()) add(urgent);
            const auto decode_start=items.size();
            for(std::size_t n=0;n<lanes.size() && remaining;++n) {
                const auto i=(decode_cursor+n)%lanes.size();
                if(eligible(lanes[i]) && lanes[i].prefilled) add(i);
            }
            if(items.size()>decode_start) decode_cursor=(items[decode_start].key.lane+1)%lanes.size();
            for(std::size_t j=decode_start;j<items.size();++j) expand(items[j]);
            for(std::size_t n=0;n<lanes.size() && remaining;++n) {
                const auto i=(prefill_cursor+n)%lanes.size();
                if(i!=urgent && eligible(lanes[i]) && !lanes[i].prefilled) add(i);
            }
            bool moved_cursor=false;
            for(auto& item:items) if(!item.budget.prefill_complete) {
                if(!moved_cursor) { prefill_cursor=(item.key.lane+1)%lanes.size(); moved_cursor=true; }
                expand(item);
            }
            for(auto& lane:lanes) if(eligible(lane) && !lane.prefilled &&
                lane.prefill_wait_steps<std::numeric_limits<std::uint32_t>::max()) ++lane.prefill_wait_steps;
        }
        if(items.empty()) return false;
        std::uint32_t decode_work=0;
        for(const auto& item:items) if(item.budget.prefill_complete)decode_work+=item.budget.token_work;
        for(auto& item:items) if(!item.budget.prefill_complete) {
            item.budget.prefill_tokens=item.budget.token_work;
            advise_prefill(item,decode_work);
            lanes[item.key.lane].prefill_wait_steps=0;
        }
        if(profile.max_inflight_batches) {
            // Retain the batch before prepare/submit; neither a throwing submit
            // nor a lost completion can release its sessions or storage early.
            batches.emplace_back(std::move(items),trace_buffer);
            auto& batch=batches.back();
            batch.execution=concurrent->prepare_batch(batch.items);
            if(!batch.execution)throw std::runtime_error("null batch execution");
            for(const auto& item:batch.items)lanes[item.key.lane].executing=true;
            batch.execution->submit([signal=wakeup]{signal->notify();});
            return true;
        }
        detail::TraceScope step(trace_buffer,TracePhase::step,std::span<const WorkItem>(items));
        auto rows=[&]{
            detail::TraceScope span(trace_buffer,TracePhase::mixed,std::span<const WorkItem>(items));
            return concurrent->execute(items);
        }();
        publish_group(items,std::move(rows),step.operation());
        return true;
    }
    bool poll_batches() {
        bool progress=false;
        for(auto it=batches.begin();it!=batches.end();) {
            if(it->execution->ready()) {
                auto rows=it->execution->progress();
                publish_group(it->items,std::move(rows),it->step.operation(),it->execution.get());
                for(const auto& item:it->items)lanes[item.key.lane].executing=false;
                it=batches.erase(it); progress=true;
            } else {
                if(Clock::now()-it->started>=std::chrono::milliseconds(profile.batch_timeout_ms))
                    throw std::runtime_error("batch execution deadline expired without settlement");
                ++it;
            }
        }
        return progress;
    }
    Finish score(const std::shared_ptr<Request>& request) {
        if(*request->cancelled) return Finish::cancelled;
        if(request->has_queries) return score_queries(request);
        request->score_output = [&]{
            detail::TraceScope span(trace_buffer,TracePhase::score,request->trace_key);
            return model->score_request(request->prompt,request->first_target,request->options);
        }();
        // Retain returned storage through the fence, including cancellation.
        drain_backend(request->trace_key);
        const auto& scores = request->score_output;
        if(scores.size() != request->prompt.size()-request->first_target ||
           !std::all_of(scores.begin(),scores.end(),[](float x){return std::isfinite(x);}))
            throw std::runtime_error("scoring output count/finiteness mismatch");
        if(*request->cancelled) {
            std::vector<float>().swap(request->score_output);
            return Finish::cancelled;
        }
        // Keep Core's pre-reserved extent, not an arbitrary backend capacity.
        request->result.scores.assign(scores.begin(),scores.end());
        std::vector<float>().swap(request->score_output);
        trace_buffer.point(TracePhase::publication,request->trace_key);
        return Finish::length;
    }
    Finish score_queries(const std::shared_ptr<Request>& request) {
        request->query_output = [&]{
            detail::TraceScope span(trace_buffer,TracePhase::score,request->trace_key);
            return model->score_queries(request->prompt,request->first_target,request->queries,request->options);
        }();
        // Retain returned storage through the fence, including cancellation.
        drain_backend(request->trace_key);
        const auto& out = request->query_output;
        const std::size_t targets = request->prompt.size()-request->first_target;
        const auto& queries = request->queries;
        if(out.target.size() != targets || out.queries.size() != targets*queries.per_target ||
           out.top.size() != targets*queries.top_k || (!out.argmax.empty() && out.argmax.size() != targets) ||
           !std::all_of(out.target.begin(),out.target.end(),[](float x){return std::isfinite(x);}) ||
           !std::all_of(out.queries.begin(),out.queries.end(),[](float x){return !std::isnan(x) && x != std::numeric_limits<float>::infinity();}))
            throw std::runtime_error("query scoring output count/finiteness mismatch");
        if(!std::all_of(out.target.begin(),out.target.end(),logprob_value) || !std::all_of(out.queries.begin(),out.queries.end(),logprob_value))
            throw std::runtime_error("query scoring log-probability above zero");
        validate_ranked(out.top,queries.top_k,valid_vocabulary(declaration));
        validate_tokens(out.argmax,valid_vocabulary(declaration),"argmax token outside the valid vocabulary");
        if(*request->cancelled) {
            request->query_output = {};
            return Finish::cancelled;
        }
        // Keep Core's pre-reserved extents, not arbitrary backend capacities.
        request->result.scores.assign(out.target.begin(),out.target.end());
        request->result.query_scores.assign(out.queries.begin(),out.queries.end());
        request->result.argmax.assign(out.argmax.begin(),out.argmax.end());
        request->result.top_logprobs.assign(out.top.begin(),out.top.end());
        request->query_output = {};
        trace_buffer.point(TracePhase::publication,request->trace_key);
        return Finish::length;
    }
    bool score_concurrent() {
        {
            std::lock_guard lock(wakeup->mutex);
            if(waiting.empty() || !waiting.front()->scoring ||
               std::any_of(lanes.begin(),lanes.end(),[](const auto& lane){return bool(lane.request);}))
                return false;
            // Retain before invoking the model: failure may leave live readers.
            active_score = waiting.front(); waiting.pop_front();
        }
        trace_buffer.point(TracePhase::admission,active_score->trace_key);
        active_score->phases.admit();
        const auto finish = score(active_score);
        {
            std::lock_guard lock(wakeup->mutex);
            busy = !waiting.empty();
        }
        complete(active_score,finish);
        active_score.reset();
        return true;
    }
    void fail_concurrent(std::exception_ptr error) noexcept {
        // Every failed request shares `error`; the last completion takes the worker's own reference. Submission
        // stops with the poison, so the queue cannot grow after it is counted.
        std::size_t failing=0;
        { std::lock_guard lock(wakeup->mutex); poisoned=true; failing=waiting.size(); }
        failing+=std::size_t(bool(active_score))+std::size_t(std::count_if(lanes.begin(),lanes.end(),[](const auto& lane){return bool(lane.request);}));
        const auto fail=[&](const std::shared_ptr<Request>& request, bool retained) {
            if(--failing) complete(request,Finish::failed,error,retained);
            else complete(request,Finish::failed,std::move(error),retained);
        };
        // Failure closes observation spans even if cleanup cannot prove that
        // physical owners are safe to release and must quarantine them instead.
        for(auto& batch:batches)batch.step.finish(true);
        bool settled=false;
        try { quiesce_model(); drain_backend(); settled=true; } catch(...) {}
        if(settled)batches.clear();
        if(active_score) {
            const auto request=active_score;
            if(settled) {
                std::vector<float>().swap(request->score_output);
                request->query_output = {};
                active_score.reset();
            }
            fail(request,!settled);
        }
        for(auto& lane:lanes) if(lane.request) {
            const auto request=lane.request;
            if(settled) {
                lane.receipt.reset(); if(lane.session) lane.session->close(Finish::failed);
                lane = Lane{};
            }
            // Match normal retirement: a completed handle must not race
            // session close or the release of settled session reservations.
            // Unknown work still retains the complete lane for quarantine.
            fail(request,!settled);
        }
        std::lock_guard lock(wakeup->mutex);
        for(auto& r:waiting) fail(r,!settled);
        if(settled) waiting.clear();
        busy=false;
    }
    void run_concurrent() noexcept {
        std::exception_ptr failure;
        try {
            backend->bind_thread();
            for(;;) {
                // Clear before inspection, not before sleeping: a completion
                // arriving during inspection must prevent the subsequent wait.
                { std::lock_guard lock(wakeup->mutex); wakeup->changed=false; }
                const bool advanced=poll_batches();
                poll_retirements(); admit();
                if(score_concurrent()) continue;
                if(step_concurrent()) continue;
                if(advanced)continue;
                std::unique_lock lock(wakeup->mutex);
                const bool occupied=std::any_of(lanes.begin(),lanes.end(),[](const auto& lane){return bool(lane.request);});
                busy=occupied || !waiting.empty();
                if(stopping && !busy) return;
                if(!busy) wakeup->cv.wait(lock,[&]{return stopping || !waiting.empty();});
                else {
                    auto deadline=Clock::time_point::max();
                    for(const auto& batch:batches)
                        deadline=std::min(deadline,batch.started+std::chrono::milliseconds(profile.batch_timeout_ms));
                    // Admission and retirement still expose polling interfaces.
                    // Pending batches instead wake us on completion or deadline.
                    if(batches.empty() || !waiting.empty() || std::any_of(lanes.begin(),lanes.end(),
                       [](const auto& lane){return lane.request && !lane.executing;}))
                        deadline=std::min(deadline,Clock::now()+std::chrono::milliseconds(1));
                    wakeup->cv.wait_until(lock,deadline,[&]{return wakeup->changed;});
                }
            }
        } catch(...) { failure=std::current_exception(); }
        // Outside the handler, so the worker's only reference to the error is the one fail_concurrent hands on.
        fail_concurrent(std::move(failure));
    }
    void execute(const std::shared_ptr<Request>& request) {
        request->trace_key={request->id,request->scoring ? 0u : 1u,0};
        const auto key=request->trace_key;
        trace_buffer.point(TracePhase::admission,key);
        request->phases.admit();
        std::unique_ptr<Session> session;
        Finish finish = Finish::failed;
        std::exception_ptr error;
        bool retained = false;
        try {
            backend->bind_thread();
            if(*request->cancelled) finish = Finish::cancelled;
            else if(request->scoring) finish = score(request);
            else if(!request->output) finish = Finish::length;
            else {
                {
                    detail::TraceScope span(trace_buffer,TracePhase::open,key);
                    session = model->open_request(request->prompt,request->options);
                }
                if(!session) throw std::runtime_error("model opened null session");
                std::uint32_t frontier = 0;
                bool prefill_complete = false;
                finish = Finish::length;
                while(request->result.tokens.size() < request->output) {
                    detail::TraceScope step(trace_buffer,TracePhase::step,key);
                    if(*request->cancelled) {finish = Finish::cancelled; break;}
                    const auto remaining = request->output-std::uint32_t(request->result.tokens.size());
                    auto chunk = std::min(profile.prefill_chunk,std::uint32_t(request->prompt.size())-frontier);
                    if(profile.prefill_target_ns && !prefill_complete && chunk) {
                        const auto advised=session->prefill_limit({frontier,std::uint32_t(request->prompt.size())-frontier,
                            chunk,0,profile.prefill_target_ns});
                        if(advised>chunk)throw std::runtime_error("prefill advice exceeds granted work");
                        if(advised)chunk=advised;
                    }
                    const WorkBudget budget{chunk,remaining,prefill_complete};
                    auto progress = [&]{
                        detail::TraceScope span(trace_buffer,prefill_complete ? TracePhase::decode : TracePhase::prefill,key);
                        return session->advance(budget);
                    }();
                    validate_deferred_progress(progress,"invalid deferred model progress");
                    if(progress.readiness) {
                        detail::TraceScope span(trace_buffer,TracePhase::readiness,key);
                        progress.readiness->wait();
                    }
                    if(progress.deferred_tokens) progress.tokens = session->ready_tokens();
                    if(trace_buffer.enabled()) {
                        try { for(const auto& timing:session->device_timings()) trace_buffer.device(step.operation(),key,timing); }
                        catch(...) { trace_buffer.drop(); }
                    }
                    if(*request->cancelled) {finish = Finish::cancelled; break;}
                    validate_progress(progress,budget,frontier,*request,"invalid or stalled model progress");
                    frontier += progress.prompt_tokens;
                    const bool was_prefilled = prefill_complete;
                    prefill_complete = progress.prefill_complete;
                    validate_tokens(progress.tokens,valid_vocabulary(declaration),"generated token outside the valid vocabulary");
                    const auto rows = logprob_rows(request->options,progress,valid_vocabulary(declaration));
                    // A text decoding failure fails this request before its commit.
                    StepText text;
                    const auto d = decide(progress.tokens,stops(*request),remaining,request->text_stream.get(),text);
                    if(d.finish == Finish::stop) finish = Finish::stop;
                    {
                        std::lock_guard lock(request->mutex);
                        const auto before = extents(request->result);
                        // Commit may invalidate the borrowed batch. Stage its accepted
                        // values first, withholding publication under the request lock.
                        try {
                            stage_output(*request,resources,rows,d.accepted,text);
                            if(d.accepted) request->result.tokens.insert(request->result.tokens.end(),progress.tokens.begin(),progress.tokens.begin()+d.accepted);
                            detail::TraceScope span(trace_buffer,TracePhase::commit,key);
                            session->commit(d.accepted,d.terminal);
                        }
                        catch(...) {restore(request->result,before); throw;}
                    }
                    request->phases.step(progress.prompt_tokens,progress.reused_prompt_tokens,was_prefilled,prefill_complete,d.accepted);
                    trace_buffer.point(TracePhase::publication,key);
                    request->cv.notify_all();
                    if(d.terminal) break;
                }
            }
        } catch(...) { error = std::current_exception(); finish = Finish::failed; }
        try {
            drain_backend(key);
            std::vector<float>().swap(request->score_output);
            request->query_output = {};
        }
        catch(...) {
            if(!error) error = std::current_exception();
            finish = Finish::failed;
            std::lock_guard lock(wakeup->mutex); poisoned = true;
            retained_session = std::move(session);
            retained_request = request;
            retained = true;
        }
        if(session) {
            detail::TraceScope span(trace_buffer,TracePhase::retirement,key);
            session->close(finish); session.reset();
        }
        {
            std::lock_guard lock(wakeup->mutex);
            busy = false;
        }
        complete(request,finish,std::move(error),retained);
    }
    void run() noexcept {
        const trace::SinkBinding operator_sink(operator_sink_here());
        if(concurrent) { run_concurrent(); return; }
        for(;;) {
            std::shared_ptr<Request> request;
            {
                std::unique_lock lock(wakeup->mutex);
                wakeup->cv.wait(lock,[&]{return stopping || pending;});
                if(stopping && !pending) return;
                request = std::move(pending);
            }
            execute(request);
        }
    }
};
struct Handle::Impl {
    std::shared_ptr<Engine::Impl> engine;
    std::shared_ptr<Request> request;
    std::shared_ptr<std::atomic<std::size_t>> outstanding;
    ~Impl() {
        if(request) {
            const auto held = request; // Outlives the lock on its mutex.
            *held->cancelled = true;
            std::unique_lock lock(held->mutex);
            held->cv.wait(lock,[&]{return held->done;});
            settle(lock);
        }
        if(outstanding) --*outstanding;
    }
    // Ends this handle's use of its done request, with the request lock held; returns with it released. The caller
    // keeps its own reference. Completion was the worker's last access to the request's owners, so they are freed
    // and their charges returned here, on the consuming thread: usage() read on this thread after wait() returns or
    // the handle is destroyed no longer includes them. A retained request keeps its charges with the quarantined
    // runtime. The worker may still drop its own reference afterwards; that releases nothing.
    void settle(std::unique_lock<std::mutex>& lock) noexcept {
        const auto held = std::move(request);
        const bool retained = held->retained;
        // An unconsumed error ends on this thread too, after the lock: its destructor may be user code.
        const auto unconsumed = std::move(held->error);
        lock.unlock();
        if(!retained) held->release();
    }
    // Waits for completion and moves the result out. With `streaming`, each published token is copied under the
    // request lock by `copy(result, index)` and delivered outside it by `deliver()`, in order on this thread. A
    // throwing delivery cancels the request, waits for it to finish, settles and rethrows; the handle stays consumed.
    template<class Copy, class Deliver> Result consume(bool streaming, Copy copy, Deliver deliver) {
        const auto held = request; // Outlives the lock on its mutex.
        std::size_t consumed = 0;
        std::unique_lock lock(held->mutex);
        for(;;) {
            held->cv.wait(lock,[&]{return held->done || (streaming && consumed < held->result.tokens.size());});
            if(streaming && consumed < held->result.tokens.size()) {
                copy(std::as_const(held->result),consumed++);
                lock.unlock();
                try { deliver(); } catch(...) {
                    // Rethrown from this handler: std::current_exception() is empty for a foreign exception, and
                    // forced unwinding (pthread_exit, thread cancellation) must not end inside a handler.
                    *held->cancelled = true;
                    lock.lock();
                    held->cv.wait(lock,[&]{return held->done;});
                    settle(lock);
                    throw;
                }
                lock.lock();
                continue;
            }
            break;
        }
        const auto error = std::move(held->error);
        auto result = std::move(held->result);
        settle(lock);
        if(error) std::rethrow_exception(error);
        return result;
    }
};
Handle::Handle() noexcept = default;
Handle::Handle(std::unique_ptr<Impl> p):impl_(std::move(p)),cancellation_(impl_ ? impl_->request->cancelled : nullptr),id_(impl_ ? impl_->request->id : 0) {}
Handle::~Handle() = default;
Handle::Handle(Handle&&) noexcept = default;
Handle& Handle::operator=(Handle&&) noexcept = default;
void Handle::cancel() noexcept { if(cancellation_) *cancellation_ = true; }
std::uint64_t Handle::id() const noexcept { return cancellation_ ? id_ : 0; }
Result Handle::wait(TokenSink sink) {
    if(!impl_ || !impl_->request) throw std::logic_error("request handle is empty/consumed");
    auto state = std::move(impl_);
    Token token = 0;
    return state->consume(bool(sink),[&](const Result& r,std::size_t i){token = r.tokens[i];},[&]{sink(token);});
}
Result Handle::wait_output(OutputSink sink) {
    if(!impl_ || !impl_->request) throw std::logic_error("request handle is empty/consumed");
    auto state = std::move(impl_);
    // Options and the text flag are immutable after submission. The row and the text piece are copied under the
    // lock: publication may grow the result while the sink runs.
    const auto& request = *state->request;
    const bool logprobs = request.options.logprobs.has_value(), text = request.text;
    const std::size_t width = request.options.logprobs.value_or(0);
    OutputEvent event;
    std::vector<TokenLogprob> top;
    std::string piece;
    top.reserve(width);
    return state->consume(bool(sink),[&](const Result& r,std::size_t i) {
        event.index = i; event.token = r.tokens[i];
        if(logprobs) {
            event.logprob = r.logprobs[i];
            top.assign(r.top_logprobs.begin()+std::ptrdiff_t(i*width),r.top_logprobs.begin()+std::ptrdiff_t((i+1)*width));
        }
        if(text) {
            const std::size_t begin = i ? r.text_ends[i-1] : 0;
            piece.assign(r.text,begin,r.text_ends[i]-begin);
        }
    },[&]{
        event.top_logprobs = top; event.text = piece;
        sink(std::as_const(event));
    });
}
Engine::Engine(Profile p,std::shared_ptr<Backend> b,Factory f) {
    impl_ = std::shared_ptr<Impl>(new Impl(p,std::move(b)),[](Impl* owner) noexcept {
        if(owner->shutdown()) {delete owner; return;}
        // Keep the services facades, model, prompt and storage owners at their
        // original addresses after an unprovable final drain. Never call their
        // destructors on this path. The process must recover its failed backend.
        static Impl* retained = nullptr;
        static std::mutex retention_mutex;
        try {std::lock_guard lock(retention_mutex); owner->retained_next = retained; retained = owner;}
        catch(...) { /* Never free unknown live work even if diagnostics fail. */ }
        ++quarantined_count;
        std::fprintf(stderr,"crit core: failed backend drain; complete runtime retained for process recovery\n");
    });
    impl_->initialize(std::move(f));
}
Engine::~Engine() = default;
void NativeEngine::retain_owner() const noexcept {
    if(engine && engine->impl_) engine->impl_->native_owner = owner;
}
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;
Handle Engine::submit(std::vector<Token> prompt,std::uint32_t output,RequestOptions options) {
    if(!impl_) throw std::logic_error("engine is moved from");
    if(!impl_->declaration.generation) throw std::logic_error("model lacks generation capability");
    return impl_->submit(std::move(prompt),output,false,0,std::move(options));
}
Handle Engine::submit_score(std::vector<Token> prompt,std::uint32_t first,RequestOptions options) {
    if(!impl_) throw std::logic_error("engine is moved from");
    if(!impl_->declaration.scoring) throw std::logic_error("model lacks scoring capability");
    if(prompt.size() < 2 || !first || first >= prompt.size()) throw std::invalid_argument("invalid scoring target domain");
    return impl_->submit(std::move(prompt),0,true,first,std::move(options));
}
Handle Engine::submit_score(std::vector<Token> prompt,std::uint32_t first,ScoreQueries queries,RequestOptions options) {
    if(!impl_) throw std::logic_error("engine is moved from");
    const auto& info = impl_->declaration;
    if(!info.scoring || !info.query_scoring) throw std::logic_error("model lacks query scoring capability");
    if(prompt.size() < 2 || !first || first >= prompt.size()) throw std::invalid_argument("invalid scoring target domain");
    const std::size_t targets = prompt.size()-first;
    // per_target zero with no ids scores the targets alone.
    if(queries.per_target ? queries.ids.size() / queries.per_target != targets || queries.ids.size() % queries.per_target
                          : !queries.ids.empty())
        throw std::invalid_argument("query ids must hold per_target ids for every scoring target");
    if(queries.top_k > info.max_top_logprobs) throw std::invalid_argument("top_k exceeds max_top_logprobs");
    if(!queries.path.empty() && std::find(info.score_paths.begin(),info.score_paths.end(),queries.path) == info.score_paths.end())
        throw std::invalid_argument("unknown score path");
    validate_tokens(queries.ids,valid_vocabulary(info),"query id outside the valid vocabulary");
    // A target at or past the valid vocabulary has no log-probability; reject it before any reservation.
    validate_tokens(std::span<const Token>(prompt).subspan(first),valid_vocabulary(info),
                    "scoring target outside the valid vocabulary");
    return impl_->submit(std::move(prompt),0,true,first,std::move(options),std::move(queries));
}
Handle Engine::Impl::submit(std::vector<Token> prompt,std::uint32_t output,bool scoring,std::uint32_t first,RequestOptions options,
                            std::optional<ScoreQueries> queries) {
    if(prompt.empty() || prompt.size() > profile.context || output > profile.context-prompt.size())
        throw std::invalid_argument("prompt plus output exceeds context");
    validate_tokens(prompt,declaration.vocabulary);
    validate_options(options,declaration,scoring,bool(frontend));
    // The text stream exists before admission so decoder creation failures reject the submission. The frontend
    // is thread-safe; its decoder is used serially, by the worker after publication below.
    const bool text = options.text || !options.stop_strings.empty();
    std::unique_ptr<TextStream> text_stream;
    if(text) text_stream = make_text_stream(frontend,prompt,options.stop_strings,declaration.vocabulary);
    // Reject overload before charging request memory. Keep the allowance check
    // and publication atomic with other submitters; failed construction leaves
    // neither an outstanding slot nor a retained reservation.
    std::lock_guard lock(wakeup->mutex);
    if(stopping || poisoned) throw std::runtime_error("engine unavailable");
    if(!concurrent && busy) throw std::runtime_error("single-request core is busy");
    if(concurrent && outstanding->load() >= std::size_t(profile.max_concurrency)+profile.max_pending)
        throw std::runtime_error("outstanding request allowance exhausted");
    if(next_request == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("request identity exhausted");
    auto request = std::make_shared<Request>();
    // Account retained prompt and maximum result bytes, including handle lifetime.
    const auto result_items = scoring ? prompt.size()-first : output;
    std::size_t bytes = 0;
    charge_bytes(bytes,prompt.capacity(),sizeof(Token));
    charge_bytes(bytes,result_items,scoring ? sizeof(float) : sizeof(Token));
    if(queries) {
        charge_bytes(bytes,queries->ids.capacity(),sizeof(Token));           // retained query ids
        charge_bytes(bytes,queries->ids.size(),sizeof(float));               // query log-probabilities
        charge_bytes(bytes,result_items,sizeof(Token));                      // argmax per target
        charge_bytes(bytes,result_items,queries->top_k*sizeof(TokenLogprob)); // ranked alternatives per target
        charge_bytes(bytes,queries->path.capacity());                        // retained path name
    }
    if(options.stop_tokens) charge_bytes(bytes,options.stop_tokens->capacity(),sizeof(Token));
    if(options.context) charge_bytes(bytes,options.context->host_bytes());
    charge_bytes(bytes,options.sampling.logit_bias.capacity(),sizeof(LogitBias));
    charge_bytes(bytes,options.stop_strings.capacity(),sizeof(std::string));
    for(const auto& stop:options.stop_strings) charge_bytes(bytes,stop.capacity());
    if(options.logprobs) {
        charge_bytes(bytes,output,sizeof(float));                              // chosen log-probabilities
        charge_bytes(bytes,output,*options.logprobs*sizeof(TokenLogprob));     // ranked alternatives
    }
    if(text) {
        charge_bytes(bytes,output,sizeof(std::uint32_t));                      // text_ends
        // The stream's matcher copy: each stop byte with its border byte, offsets, states and the holdback.
        // Released text is charged as it grows (reserve_text).
        for(const auto& stop:options.stop_strings) charge_bytes(bytes,stop.size(),2);
        charge_bytes(bytes,options.stop_strings.size()+1,sizeof(std::uint32_t)+sizeof(std::uint16_t));
        charge_bytes(bytes,stop_string_bytes_limit);
    }
    request->storage = resources.reserve(Memory::host,bytes);
    if(concurrent) {
        request->staging_storage = resources.reserve(Memory::host,std::size_t(output)*sizeof(Token));
        request->staging.reserve(output);
    }
    request->prompt = std::move(prompt);
    request->options = std::move(options);
    request->output = output; request->scoring = scoring; request->first_target = first;
    if(scoring) request->result.scores.reserve(result_items);
    else request->result.tokens.reserve(output);
    if(request->options.logprobs) {
        request->result.logprobs.reserve(output);
        request->result.top_logprobs.reserve(std::size_t(output)**request->options.logprobs);
    }
    if(text) {
        request->result.text_ends.reserve(output);
        request->text = true;
        request->text_stream = std::move(text_stream);
    }
    if(queries) {
        request->result.query_scores.reserve(queries->ids.size());
        request->result.argmax.reserve(result_items);
        request->result.top_logprobs.reserve(result_items*queries->top_k);
        request->queries = std::move(*queries);
        request->has_queries = true;
    }
    // Retain the runtime before the worker can observe the request.
    auto handle = std::make_unique<Handle::Impl>();
    handle->engine = shared_from_this();
    request->id = ++next_request; request->result.request_id=request->id;
    request->trace_key={request->id,0,0};
    request->submitted=std::chrono::steady_clock::now();
    if(concurrent) {
        waiting.push_back(request);
        ++*outstanding; handle->outstanding=outstanding;
    } else pending=request;
    // No throwing operation may follow attaching an uncompleted request to a
    // handle: its destructor waits. In particular queue growth precedes this.
    handle->request = request;
    busy = true;
    trace_buffer.point(TracePhase::submitted,request->trace_key);
    wakeup->changed=true;
    wakeup->cv.notify_one();
    return Handle(std::move(handle));
}
std::vector<Token> Engine::tokenize(std::string_view text) {
    if(!impl_) throw std::logic_error("engine is moved from");
    if(impl_->frontend) {
        auto tokens = impl_->frontend->tokenize(text);
        validate_tokens(tokens,impl_->declaration.vocabulary);
        return tokens;
    }
    std::lock_guard lock(impl_->wakeup->mutex);
    if(impl_->busy || impl_->poisoned) throw std::runtime_error("engine unavailable for frontend access");
    auto tokens = impl_->model->tokenize(text);
    validate_tokens(tokens,impl_->declaration.vocabulary);
    return tokens;
}
std::string Engine::detokenize(std::span<const Token> tokens) {
    if(!impl_) throw std::logic_error("engine is moved from");
    validate_tokens(tokens,impl_->declaration.vocabulary);
    if(impl_->frontend) return impl_->frontend->detokenize(tokens);
    std::lock_guard lock(impl_->wakeup->mutex);
    if(impl_->busy || impl_->poisoned) throw std::runtime_error("engine unavailable for frontend access");
    return impl_->model->detokenize(tokens);
}
std::unique_ptr<TextDecoder> Engine::create_decoder() {
    if(!impl_) throw std::logic_error("engine is moved from");
    auto frontend = impl_->frontend;
    if(!frontend) throw std::logic_error("model lacks an independent text frontend");
    auto decoder = frontend->create_decoder();
    if(!decoder) throw std::runtime_error("text frontend returned null decoder");
    return std::make_unique<OwnedTextDecoder>(std::move(frontend),std::move(decoder),impl_->declaration.vocabulary);
}
ModelInfo Engine::info() const { if(!impl_) throw std::logic_error("engine is moved from"); return impl_->declaration; }
Profile Engine::profile() const { if(!impl_) throw std::logic_error("engine is moved from"); return impl_->profile; }
Usage Engine::usage(Memory memory) const { if(!impl_) throw std::logic_error("engine is moved from"); return impl_->resources.usage(memory); }
TraceSnapshot Engine::trace() const {
    if(!impl_) throw std::logic_error("engine is moved from");
    auto result = impl_->trace_buffer.snapshot();
    if(impl_->operator_ranges) result.operators = impl_->operator_ranges->snapshot();
    return result;
}
}
