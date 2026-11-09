#include "crit/step_engine.h"
#include "crit/json.h"
#include "json_text.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace crit::core {
namespace {
using detail::quoted;
using detail::real;
// The settings an "engine" object may hold, from the table crit_core.native's StepEngine also reads
// (python/crit_core/step_settings.json).
#include "crit_step_settings.inc"

struct Settings {
    ModelInfo info;
    Profile profile;
    std::uint32_t row_tokens = 1;
    std::string facts = "{}";
    // The engine's own per-row counters of a native round, named (round_counters): the result block's words after its
    // accepted drafts, summed into each request's facts under these names.
    std::vector<std::string> counters;
};
std::uint32_t count(const json::Value& root, const char* key, std::uint32_t fallback) {
    const auto* v = root.find(key);
    if(!v || v->is_null()) return fallback;
    const auto n = v->u64();
    if(n > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument(std::string("step engine ") + key + " is too large");
    return std::uint32_t(n);
}
bool flag(const json::Value& root, const char* key) {
    const auto* v = root.find(key);
    return v && !v->is_null() && v->boolean();
}
Settings settings(std::string_view configuration) {
    const auto document = json::Document::parse(configuration, "step engine configuration");
    const auto& root = document.root();
    if(root.kind() != json::Kind::object) throw std::invalid_argument("step engine configuration must be a JSON object");
    const auto* engine = root.find("engine");
    if(!engine || engine->kind() != json::Kind::object) throw std::invalid_argument("step engine configuration needs an \"engine\" object");
    const auto& e = *engine;
    for(const auto& member:e.object())
        if(std::find(step_settings.begin(), step_settings.end(), member.key) == step_settings.end())
            throw std::invalid_argument("step engine setting \"" + std::string(member.key) +
                                        "\" is not one of Core's (an engine's own facts go in \"facts\")");
    Settings s;
    auto& i = s.info;
    const auto* name = e.find("implementation");
    i.implementation = name && !name->is_null() ? std::string(name->string()) : "python step engine";
    i.vocabulary = count(e, "vocabulary", 0);
    i.valid_vocabulary = count(e, "valid_vocabulary", 0);
    i.context = count(e, "context", 0);
    if(!i.vocabulary || !i.context) throw std::invalid_argument("step engine needs a positive vocabulary and context");
    i.prefill_chunk = count(e, "prefill_chunk", std::min<std::uint32_t>(512, i.context));
    if(const auto* stops = e.find("stop_tokens"); stops && !stops->is_null())
        for(const auto& t:stops->array()) {
            const auto id = t.i64();   // checked before the cast, which would wrap an id past 2^31 - 1 into another
            if(id < 0 || id > std::numeric_limits<Token>::max())
                throw std::invalid_argument("step engine stop_tokens hold a token id outside [0, 2^31)");
            i.stop_tokens.push_back(Token(id));
        }
    i.max_concurrency = count(e, "max_concurrency", 8);
    s.row_tokens = count(e, "max_row_tokens", 1);
    if(!s.row_tokens) throw std::invalid_argument("step engine max_row_tokens must be positive");
    i.decode_token_work = count(e, "decode_token_work", 1);
    // Prompt work in whole multiples of this (Core's grants; the engine checks it fits max_batch_tokens): with the
    // prefill chunk as the quantum, a prompt's chunks start at the same positions whatever shares its steps.
    i.prefill_work_quantum = count(e, "prefill_work_quantum", 1);
    if(!i.prefill_work_quantum) throw std::invalid_argument("step engine prefill_work_quantum must be positive");
    if(const auto* sampling = e.find("sampling"); sampling && !sampling->is_null()) {
        auto& support = i.sampling;
        support = {flag(*sampling, "temperature"), flag(*sampling, "top_p"), flag(*sampling, "top_k"), flag(*sampling, "seed"),
                   flag(*sampling, "min_p"), flag(*sampling, "penalties"), flag(*sampling, "logit_bias"), flag(*sampling, "min_tokens")};
    }
    i.token_logprobs = flag(e, "token_logprobs");
    i.max_top_logprobs = count(e, "max_top_logprobs", 0);
    i.scoring = i.query_scoring = flag(e, "scoring");
    if(const auto* paths = e.find("score_paths"); paths && !paths->is_null())
        for(const auto& p:paths->array()) i.score_paths.emplace_back(p.string());
    // Replay continuations and plain decoding arrive as request contexts (crit_engine_generate's options).
    i.request_context = true;
    auto& p = s.profile;
    p.context = i.context;
    p.prefill_chunk = i.prefill_chunk;
    p.max_concurrency = i.max_concurrency;
    p.max_pending = count(e, "max_pending", i.max_concurrency);
    p.max_batch_tokens = count(e, "max_batch_tokens", i.prefill_chunk + i.max_concurrency * s.row_tokens);
    i.mixed_batches = p.max_batch_tokens > 0;
    if(const auto* policy = e.find("batch_policy"); policy && !policy->is_null()) {
        if(policy->string() == "decode_first") p.batch_policy = BatchPolicy::decode_first;
        else if(policy->string() != "fair") throw std::invalid_argument("step engine batch_policy must be fair or decode_first");
    }
    if(const auto* names = e.find("round_counters"); names && !names->is_null())
        for(const auto& counter:names->array()) {
            const std::string text(counter.string());
            if(text.empty() || std::find(s.counters.begin(), s.counters.end(), text) != s.counters.end())
                throw std::invalid_argument("step engine round_counters are distinct names");
            s.counters.push_back(text);
        }
    const auto* host = e.find("host_limit");
    p.host_limit = host && !host->is_null() ? std::size_t(host->u64()) : std::size_t(1) << 30;
    // Device memory is the engine's own; Core reserves none of it.
    p.device_limit = std::numeric_limits<std::size_t>::max();
    if(const auto* facts = root.find("facts"); facts && !facts->is_null()) {
        if(facts->kind() != json::Kind::object) throw std::invalid_argument("step engine facts must be a JSON object");
        s.facts = detail::facts_object(*facts);
    }
    return s;
}

// The request's options as the engine reads them (StepRequest::options_json).
std::string options_json(const RequestView& view, const ModelInfo& info) {
    const auto& o = *view.options;
    const auto& sampling = o.sampling;
    std::string out = "{";
    const auto key = [&](const char* name) {
        if(out.size() > 1) out += ',';
        quoted(out, name);
        out += ':';
    };
    const auto number = [&](const char* name, const std::optional<double>& v) { if(v) { key(name); real(out, *v); } };
    number("temperature", sampling.temperature);
    number("top_p", sampling.top_p);
    if(sampling.top_k) { key("top_k"); out += std::to_string(*sampling.top_k); }
    number("min_p", sampling.min_p);
    if(sampling.seed) { key("seed"); out += std::to_string(*sampling.seed); }
    number("repetition_penalty", sampling.repetition_penalty);
    number("frequency_penalty", sampling.frequency_penalty);
    number("presence_penalty", sampling.presence_penalty);
    if(!sampling.logit_bias.empty()) {
        key("logit_bias");
        out += '{';
        for(std::size_t b = 0; b < sampling.logit_bias.size(); ++b) {
            if(b) out += ',';
            quoted(out, std::to_string(sampling.logit_bias[b].token));
            out += ':';
            real(out, double(sampling.logit_bias[b].bias));
        }
        out += '}';
    }
    if(sampling.min_tokens) { key("min_tokens"); out += std::to_string(*sampling.min_tokens); }
    const auto& stops = o.stop_tokens ? *o.stop_tokens : info.stop_tokens;
    key("stop_tokens");
    out += '[';
    for(std::size_t t = 0; t < stops.size(); ++t) out += (t ? "," : "") + std::to_string(stops[t]);
    out += ']';
    if(o.logprobs) { key("logprobs"); out += std::to_string(*o.logprobs); }
    if(const auto* replay = dynamic_cast<const ReplayContinuation*>(o.context.get())) {
        key("replay");
        out += '[';
        for(std::size_t t = 0; t < replay->tokens.size(); ++t) out += (t ? "," : "") + std::to_string(replay->tokens[t]);
        out += ']';
    }
    const auto* attachment = dynamic_cast<const HostAttachment*>(o.context.get());
    if(attachment) {
        key("attachment");
        out += std::to_string(attachment->attachment);
        if(attachment->constrained) { key("constrained"); out += "true"; }
    }
    if(dynamic_cast<const PlainDecoding*>(o.context.get()) || (attachment && (attachment->plain || attachment->constrained))) {
        key("plain");
        out += "true";
    }
    return out + "}";
}

struct Host final : Backend {
    void bind_thread() override {}
    void drain() override {}
};

// Each retired request's facts until its waiter merges them into its result (NativeEngine::result), by request id: only
// requests that end with a result (length, stop) are stored, so every entry is taken; `limit` bounds it all the same.
class RequestFacts {
    std::mutex mutex_;
    std::unordered_map<std::uint64_t, std::string> facts_;
    std::deque<std::uint64_t> order_;
    static constexpr std::size_t limit = 4096;
public:
    void put(std::uint64_t request, std::string text) {
        std::lock_guard lock(mutex_);
        if(facts_.insert_or_assign(request, std::move(text)).second) order_.push_back(request);
        while(order_.size() > limit) {
            facts_.erase(order_.front());
            order_.pop_front();
        }
    }
    std::string take(std::uint64_t request) {
        std::lock_guard lock(mutex_);
        const auto found = facts_.find(request);
        if(found == facts_.end()) return "{}";
        auto text = std::move(found->second);
        facts_.erase(found);
        order_.erase(std::find(order_.begin(), order_.end(), request));
        return text;
    }
};

// A session's decode rounds, as its result's facts report them.
struct RoundFacts {
    std::uint32_t rounds = 0;
    std::vector<std::uint32_t> accepted;
    std::vector<std::int64_t> drafted;   // -1: not reported
    std::vector<char> native;            // per round: 1 when Core ran it natively
    std::vector<std::uint64_t> counters; // the engine's round counters, summed over its native rounds
    bool drafting = false;
    void add(std::uint32_t accepted_drafts, std::int64_t drafts, bool natively) {
        ++rounds;
        accepted.push_back(accepted_drafts);
        drafted.push_back(drafts);
        native.push_back(natively ? 1 : 0);
        drafting = drafting || accepted_drafts > 0 || drafts > 0;
    }
    // with_rounds: each round's kind (`_native`, 1 native, 0 step()) for the engine's merge (crit_core.native.merge_facts).
    std::string json(const std::vector<std::string>& names, bool with_rounds) const {
        if(!rounds) return "{}";
        std::string out = "{\"rounds\":" + std::to_string(rounds);
        const auto list = [&](const char* key, const auto& values) {
            out += ",\"";
            out += key;
            out += "\":[";
            for(std::size_t i = 0; i < values.size(); ++i) out += (i ? "," : "") + std::to_string(values[i]);
            out += ']';
        };
        if(drafting) {
            list("accepted", accepted);
            if(std::all_of(drafted.begin(), drafted.end(), [](std::int64_t d) { return d >= 0; })) list("drafted", drafted);
        }
        if(with_rounds) {
            std::vector<int> kinds(native.begin(), native.end());
            list("_native", kinds);
        }
        if(std::find(native.begin(), native.end(), 1) != native.end())
            for(std::size_t c = 0; c < names.size() && c < counters.size(); ++c) {
                out += ',';
                quoted(out, names[c]);
                out += ':' + std::to_string(counters[c]);
            }
        return out + "}";
    }
};

class StepModel final : public ConcurrentModel {
    struct Receipt final : Retirement { bool ready() override { return true; } };
    struct Session final : ConcurrentSession {
        StepModel* model;
        std::uint64_t handle;
        std::optional<std::uint32_t> logprobs;
        bool constrained = false;   // Core's host side masks its tokens: never in a native round
        bool replay = false;        // a replay continuation: its native rounds run the log-probability form
        RoundFacts rounds;
        std::vector<Token> tokens;
        std::vector<float> token_logprobs;
        std::vector<TokenLogprob> top;
        Session(StepModel* m, std::uint64_t h, std::optional<std::uint32_t> n, bool c, bool r)
            : model(m), handle(h), logprobs(n), constrained(c), replay(r) {}
        ~Session() override { model->release(handle); }
        void close(Finish) noexcept override {}
    };

    StepCallbacks cb_;
    const StepRounds* rounds_ = nullptr;
    Settings settings_;
    Resources& resources_;
    std::shared_ptr<RequestFacts> facts_;
    std::vector<StepRow> rows_;
    std::vector<StepProgress> progress_;
    std::vector<std::int32_t> tokens_, top_ids_;
    std::vector<float> logprobs_, top_values_;
    std::vector<StepDecision> decisions_;
    Reservation scores_;
    std::vector<std::uint32_t> slot_of_;
    std::vector<std::uint64_t> words_;   // each native row's words: accepted drafts, then the engine's counters
    std::vector<char> opened_;           // per row count: 1 when open gave it a width (it may run natively)
    bool native_last_ = false;
    bool logprobs_round_ = false;        // the native round native() chose runs its log-probability form

    // A native round serves these items when every one is a decode row of a session in slots 0 .. n - 1 (each slot
    // once), the engine gives n rows a width, every row's grant holds the round's columns, no request is constrained,
    // and a request asking for log-probabilities asks for no more alternatives than the rounds rank (StepRounds'
    // logprobs and top). slot_of_[i] is item i's slot. The round runs its log-probability form when a row's request
    // asks for log-probabilities or replays a continuation (logprobs_round_), else its plain form.
    bool native(std::span<const WorkItem> items, bool mixed) {
        const auto* r = rounds_;
        if(!r || !r->launch || !r->wait || !r->block || !r->slot_session || !r->widths) return false;
        const auto n = items.size();
        if(!n || n > r->max_rows || r->widths[n - 1] < 0 || !opened_[n - 1]) return false;
        if(r->widths[n - 1] > std::int32_t(r->widest))
            throw std::runtime_error("native round width " + std::to_string(r->widths[n - 1]) + " is wider than the widest (" +
                                     std::to_string(r->widest) + ")");
        const auto columns = std::uint32_t(r->widths[n - 1]) + 1;
        slot_of_.assign(n, 0);
        logprobs_round_ = false;
        std::vector<char> taken(n, 0);
        for(std::size_t i = 0; i < n; ++i) {
            const auto& session = *static_cast<Session*>(items[i].session);
            if(!items[i].budget.prefill_complete || session.constrained) return false;
            if(session.logprobs && !(r->logprobs && settings_.info.token_logprobs && *session.logprobs <= r->top))
                return false;
            logprobs_round_ = logprobs_round_ || session.logprobs || session.replay;
            if(mixed && items[i].budget.token_work < columns) return false;   // step() decodes within a short grant
            std::size_t s = 0;
            while(s < n && r->slot_session[s] != session.handle) ++s;
            if(s == n || taken[s]) return false;
            taken[s] = 1;
            slot_of_[i] = std::uint32_t(s);
        }
        return true;
    }
    std::uint64_t entry(std::uint32_t slot, std::uint32_t column) const {
        const auto* r = rounds_;
        const auto* at = static_cast<const unsigned char*>(r->block) + (std::size_t(slot) * r->row_stride + column) * r->entry_bytes;
        if(r->entry_bytes == 8) {
            std::int64_t v;
            std::memcpy(&v, at, 8);
            if(v < 0 || v > std::numeric_limits<std::int32_t>::max()) throw std::runtime_error("native round block entry out of range");
            return std::uint64_t(v);
        }
        std::int32_t v;
        std::memcpy(&v, at, 4);
        if(v < 0) throw std::runtime_error("native round block entry out of range");
        return std::uint64_t(v);
    }
    void run_native(std::span<const WorkItem> items) {
        const auto n = items.size();
        const auto* r = rounds_;
        const auto width = std::uint32_t(r->widths[n - 1]);   // native() checked it against widest
        if(r->entry_bytes != 4 && r->entry_bytes != 8) throw std::runtime_error("native round entries are 4 or 8 bytes");
        if(r->row_stride < 3 + width + 1) throw std::runtime_error("native round block rows are narrower than the round");
        call(r->launch(r->user, std::uint32_t(n), width, logprobs_round_ ? 1 : 0), "native round launch");
        call(r->wait(r->user), "native round wait");
        const std::size_t row = settings_.row_tokens;
        const auto words = r->telemetry, per = words ? 1 + r->counters : 0;
        words_.assign(n * per, 0);
        for(std::size_t i = 0; i < n; ++i) {
            const auto slot = slot_of_[i];
            const auto count = entry(slot, 0);
            if(count > width + 1 || count > row) throw std::runtime_error("native round emitted more tokens than its width or max_row_tokens");
            auto& p = progress_[i];
            p.prefill_complete = 1;
            p.tokens = std::uint32_t(count);
            p.token_work = width + 1;   // every column the round verified (the model's decode_token_work covers it)
            p.drafted = std::int32_t(width) + 1;
            for(std::uint32_t t = 0; t < count; ++t) tokens_[i * row + t] = std::int32_t(entry(slot, 3 + t));
            for(std::uint32_t w = 0; w < per; ++w) words_[i * per + w] = entry(slot, words + w);
            // The round's log-probabilities of its emitted tokens, for a request that asked (native() checked them).
            if(static_cast<const Session*>(items[i].session)->logprobs) {
                const std::size_t columns = std::size_t(r->widest) + 1, top = settings_.info.max_top_logprobs;
                const auto alternatives = std::min<std::size_t>(top, r->top);
                for(std::uint32_t t = 0; t < count; ++t) {
                    const auto at = std::size_t(slot) * columns + t;
                    logprobs_[i * row + t] = r->logprobs[at];
                    for(std::size_t a = 0; a < alternatives; ++a) {
                        top_ids_[(i * row + t) * top + a] = std::int32_t(r->top_ids[at * r->top + a]);
                        top_values_[(i * row + t) * top + a] = r->top_logprobs[at * r->top + a];
                    }
                }
            }
        }
        if(r->last) {
            r->last[0] = std::uint32_t(n);
            r->last[1] = width;
            r->last[2] = 1;
        }
        if(r->runs) {
            r->runs[0] += 1;
            r->runs[1] += n;
            r->runs[2] += logprobs_round_ ? 1 : 0;
        }
    }

    [[noreturn]] void failed(const char* what) const {
        std::string message;
        if(cb_.error) {
            std::array<char, 4096> buffer{};
            const int n = cb_.error(cb_.user, buffer.data(), int(buffer.size()));
            if(n > 0) message.assign(buffer.data(), std::min<std::size_t>(std::size_t(n), buffer.size() - 1));
        }
        throw std::runtime_error(std::string("step engine ") + what + " failed" + (message.empty() ? "" : ": " + message));
    }
    void call(int code, const char* what) const { if(code) failed(what); }
    // Score scratch is kept until the next ordinary request callback.
    void ordinary() { scores_ = {}; }
    void release(std::uint64_t handle) noexcept { if(cb_.release) cb_.release(cb_.user, handle); }
    static StepRequest request(const RequestView& view, const std::string& options) {
        StepRequest r;
        r.request = view.key.request;
        r.generation = view.key.generation;
        r.lane = view.key.lane;
        r.prompt = view.prompt.data();
        r.prompt_tokens = std::uint32_t(view.prompt.size());
        r.output_tokens = view.output_tokens;
        r.options_json = options.c_str();
        return r;
    }
    std::vector<RowProgress> run(std::span<const WorkItem> items, bool mixed) {
        ordinary();
        const auto n = items.size(), row = std::size_t(settings_.row_tokens), top = std::size_t(settings_.info.max_top_logprobs);
        const bool logprobs = settings_.info.token_logprobs;
        rows_.resize(n);
        progress_.assign(n, {});
        tokens_.assign(n * row, -1);
        if(logprobs) {
            logprobs_.assign(n * row, 0.f);
            top_ids_.assign(n * row * top, -1);
            top_values_.assign(n * row * top, 0.f);
        }
        for(std::size_t i = 0; i < n; ++i) {
            const auto& budget = items[i].budget;
            rows_[i] = {static_cast<Session*>(items[i].session)->handle, budget.prefill_tokens, budget.output_tokens,
                        mixed ? budget.token_work : 0, budget.prefill_complete ? 1 : 0};
        }
        native_last_ = native(items, mixed);
        if(native_last_) {
            run_native(items);
        } else {
            if(rounds_ && rounds_->last) rounds_->last[2] = 0;
            call(cb_.step(cb_.user, rows_.data(), std::uint32_t(n), progress_.data(), tokens_.data(), logprobs ? logprobs_.data() : nullptr,
                          logprobs ? top_ids_.data() : nullptr, logprobs ? top_values_.data() : nullptr), "step");
        }
        std::vector<RowProgress> out;
        out.reserve(n);
        for(std::size_t i = 0; i < n; ++i) {
            const auto& p = progress_[i];
            auto& session = *static_cast<Session*>(items[i].session);
            if(p.tokens > row) throw std::runtime_error("step engine reported more tokens than max_row_tokens");
            if(p.reused_prompt_tokens > p.prompt_tokens) throw std::runtime_error("step engine reused more prompt tokens than it reported");
            session.tokens.assign(tokens_.begin() + std::ptrdiff_t(i * row), tokens_.begin() + std::ptrdiff_t(i * row + p.tokens));
            if(items[i].budget.prefill_complete && p.tokens) {
                // A decode row is one round: a native round's words (accepted drafts, the engine's counters), else
                // its tokens less one accepted.
                const bool words = native_last_ && rounds_->telemetry;
                const auto per = words ? std::size_t(1 + rounds_->counters) : 0;
                session.rounds.add(words ? std::uint32_t(words_[i * per]) : p.tokens - 1, p.drafted > 0 ? p.drafted - 1 : -1,
                                   native_last_);
                if(words) {
                    auto& sums = session.rounds.counters;
                    sums.resize(rounds_->counters, 0);
                    for(std::size_t c = 0; c < rounds_->counters; ++c) sums[c] += words_[i * per + 1 + c];
                }
            }
            Progress progress;
            progress.prompt_tokens = p.prompt_tokens;
            progress.reused_prompt_tokens = p.reused_prompt_tokens;
            progress.prefill_complete = p.prefill_complete != 0;
            progress.tokens = session.tokens;
            if(mixed) progress.token_work = p.token_work ? p.token_work : std::max(p.prompt_tokens - p.reused_prompt_tokens, p.tokens);
            if(logprobs && session.logprobs) {
                session.token_logprobs.assign(logprobs_.begin() + std::ptrdiff_t(i * row), logprobs_.begin() + std::ptrdiff_t(i * row + p.tokens));
                session.top.clear();
                const auto alternatives = std::min<std::size_t>(*session.logprobs, top);
                for(std::size_t t = 0; t < p.tokens; ++t)
                    for(std::size_t a = 0; a < alternatives; ++a) {
                        const auto at = (i * row + t) * top + a;
                        session.top.push_back({top_ids_[at], top_values_[at]});
                    }
                progress.logprobs = session.token_logprobs;
                progress.top_logprobs = session.top;
            }
            out.push_back({items[i].key, progress, items[i].step});
        }
        return out;
    }
    QueryScores scored(std::span<const Token> prompt, std::uint32_t first_target, const ScoreQueries* queries) {
        ordinary();
        if(!cb_.score) throw std::logic_error("step engine declares scoring without a score callback");
        const auto targets = prompt.size() - first_target;
        const auto per = queries ? std::size_t(queries->per_target) : 0, top_k = queries ? std::size_t(queries->top_k) : 0;
        QueryScores out;
        out.target.assign(targets, 0.f);
        out.queries.assign(targets * per, 0.f);
        std::vector<std::int32_t> argmax(targets, -1), top_ids(targets * top_k, -1);
        std::vector<float> top_values(targets * top_k, 0.f);
        std::int32_t has_argmax = 0;
        const std::string path = queries ? queries->path : std::string();
        call(cb_.score(cb_.user, prompt.data(), std::uint32_t(prompt.size()), first_target, path.c_str(),
                       per ? queries->ids.data() : nullptr, std::uint32_t(per), std::uint32_t(top_k), out.target.data(),
                       per ? out.queries.data() : nullptr, argmax.data(), &has_argmax, top_k ? top_ids.data() : nullptr,
                       top_k ? top_values.data() : nullptr), "score");
        if(has_argmax) out.argmax.assign(argmax.begin(), argmax.end());
        for(std::size_t i = 0; i < top_ids.size(); ++i) out.top.push_back({top_ids[i], top_values[i]});
        scores_ = resources_.reserve(Memory::host, (out.target.size() + out.queries.size()) * sizeof(float) +
                                                   out.argmax.size() * sizeof(Token) + out.top.size() * sizeof(TokenLogprob));
        return out;
    }
public:
    StepModel(const StepCallbacks& callbacks, const StepRounds* rounds, Settings configured, Services services,
              std::shared_ptr<RequestFacts> facts)
        : cb_(callbacks), rounds_(rounds), settings_(std::move(configured)), resources_(services.resources),
          facts_(std::move(facts)) {
        if(rounds_ && rounds_->widths)
            for(std::uint32_t r = 0; r < rounds_->max_rows; ++r) opened_.push_back(rounds_->widths[r] >= 0 ? 1 : 0);
    }
    ModelInfo info() const override { return settings_.info; }
    Admission admission(const RequestView& view) override {
        ordinary();
        if(!cb_.admission) return Admission::ready;
        const auto options = options_json(view, settings_.info);
        const auto r = request(view, options);
        switch(cb_.admission(cb_.user, &r)) {
            case 0: return Admission::ready;
            case 1: return Admission::deferred;
            case 2: return Admission::infeasible;
            default: failed("admission");
        }
    }
    std::unique_ptr<ConcurrentSession> open(const RequestView& view) override {
        ordinary();
        const auto options = options_json(view, settings_.info);
        const auto r = request(view, options);
        std::uint64_t handle = 0;
        call(cb_.open(cb_.user, &r, &handle), "open");
        try {
            const auto* context = view.options->context.get();
            const auto* attachment = dynamic_cast<const HostAttachment*>(context);
            return std::make_unique<Session>(this, handle, view.options->logprobs, attachment && attachment->constrained,
                                             dynamic_cast<const ReplayContinuation*>(context) != nullptr);
        } catch(...) {
            release(handle);
            throw;
        }
    }
    RowProgress prefill(const WorkItem& item) override { return run(std::span<const WorkItem>(&item, 1), false).front(); }
    std::vector<RowProgress> decode(std::span<const WorkItem> items) override { return run(items, false); }
    std::vector<RowProgress> execute(std::span<const WorkItem> items) override { return run(items, true); }
    void commit(std::span<const CommitDecision> decisions) override {
        ordinary();
        // Discards mark descendants of steps still in flight, and a step engine runs one step at a time: every step it
        // ran is kept or ends its request (crit/step_engine.h, StepDecision). A model need not undo a step.
        if(std::any_of(decisions.begin(), decisions.end(), [](const CommitDecision& d) { return d.discard; }))
            throw std::logic_error("Core discarded a step a step engine ran, which runs one step at a time");
        if(!cb_.commit) return;
        // After a native round the engine hears only of rows Core trimmed, ended, cancelled or discarded.
        if(native_last_ && std::all_of(decisions.begin(), decisions.end(), [](const CommitDecision& d) {
               return !d.terminal && !d.cancelled && !d.discard &&
                      d.accepted == static_cast<const Session*>(d.session)->tokens.size();
           }))
            return;
        decisions_.clear();
        for(const auto& d:decisions)
            decisions_.push_back({static_cast<Session*>(d.session)->handle, d.accepted, d.terminal ? 1 : 0, d.cancelled ? 1 : 0, d.discard ? 1 : 0});
        call(cb_.commit(cb_.user, decisions_.data(), std::uint32_t(decisions_.size())), "commit");
    }
    std::unique_ptr<Retirement> retire(RequestKey key, ConcurrentSession& session, Finish finish) override {
        ordinary();
        auto& s = static_cast<Session&>(session);
        // Only a request that ends with a result has its facts read: a cancelled or failed one's waiter never merges them.
        if(finish == Finish::length || finish == Finish::stop) facts_->put(key.request, request_facts(s));
        if(cb_.retire) call(cb_.retire(cb_.user, s.handle, int(finish)), "retire");
        return std::make_unique<Receipt>();
    }
    // The retiring session's facts: Core's of its rounds, with the engine's (StepCallbacks::facts) where it has them. The
    // callback writes into a buffer of the length it asks for; asked again, it writes what it wrote the first time.
    std::string request_facts(const Session& s) const {
        if(!cb_.facts) return s.rounds.json(settings_.counters, false);
        const auto core = s.rounds.json(settings_.counters, true);
        std::vector<char> buffer(std::max<std::size_t>(std::size_t(1) << 16, 2 * core.size() + 1));
        for(int attempt = 0; attempt < 2; ++attempt) {
            const int n = cb_.facts(cb_.user, s.handle, core.c_str(), buffer.data(), int(buffer.size()));
            if(n < 0) return s.rounds.json(settings_.counters, false);
            if(std::size_t(n) < buffer.size()) {
                std::string text(buffer.data(), std::size_t(n));
                const auto document = json::Document::parse(text, "request facts");
                if(document.root().kind() != json::Kind::object) throw std::runtime_error("request facts must be a JSON object");
                return text;
            }
            buffer.resize(std::size_t(n) + 1);
        }
        throw std::runtime_error("request facts grew while they were written");
    }
    void quiesce() override {}  // every callback returns with its work done
    std::vector<float> score_request(std::span<const Token> prompt, std::uint32_t first_target, const RequestOptions&) override {
        return scored(prompt, first_target, nullptr).target;
    }
    QueryScores score_queries(std::span<const Token> prompt, std::uint32_t first_target, const ScoreQueries& queries,
                              const RequestOptions&) override {
        return scored(prompt, first_target, &queries);
    }
};
}  // namespace

namespace {
// Why native rounds over `r` cannot run on an engine of these settings, refused at open rather than when a round would
// exceed its row's grant mid-serve: every row count given a width may run the widest round, whose columns must fit
// max_row_tokens, each row's decode grant (decode_token_work) and, with mixed batches, max_batch_tokens for every row;
// the block's words lie past the widest round's tokens and inside its rows, one per declared counter.
void check_rounds(const StepRounds& r, const Settings& s) {
    if(r.entry_bytes != 4 && r.entry_bytes != 8) throw std::invalid_argument("native round entries are 4 or 8 bytes");
    if(!r.max_rows || !r.widths) throw std::invalid_argument("native rounds need a width per row count");
    if(r.widest > std::uint32_t(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument("native rounds' widest round is out of range");
    const auto widest = std::int32_t(r.widest);
    const auto columns = std::uint32_t(widest) + 1;
    for(std::uint32_t rows = 1; rows <= r.max_rows; ++rows) {
        const auto width = r.widths[rows - 1];
        if(width < 0) continue;
        if(width > widest)
            throw std::invalid_argument("native rounds of width " + std::to_string(width) + " are wider than their widest (" +
                                        std::to_string(widest) + ")");
        if(columns > s.row_tokens || columns > s.info.decode_token_work)
            throw std::invalid_argument("native rounds of width " + std::to_string(widest) + " emit and compute " +
                                        std::to_string(columns) + " columns a row: max_row_tokens and decode_token_work "
                                        "must hold them");
        if(s.profile.max_batch_tokens && std::uint64_t(rows) * columns > s.profile.max_batch_tokens)
            throw std::invalid_argument("native rounds of " + std::to_string(rows) + " rows at width " +
                                        std::to_string(widest) + " need max_batch_tokens of at least " +
                                        std::to_string(std::uint64_t(rows) * columns));
    }
    if(r.row_stride < 3 + std::uint32_t(widest) + 1) throw std::invalid_argument("native round block rows are narrower than the widest round");
    if(r.telemetry) {
        if(r.telemetry < 3 + std::uint32_t(widest) + 1 || r.telemetry + 1 + r.counters > r.row_stride)
            throw std::invalid_argument("native round words must lie past the widest round's tokens and inside its rows");
        if(r.counters != s.counters.size())
            throw std::invalid_argument("native rounds carry " + std::to_string(r.counters) + " counters and the engine names " +
                                        std::to_string(s.counters.size()) + " (round_counters)");
    } else if(!s.counters.empty()) {
        throw std::invalid_argument("the engine names round_counters, and its native rounds carry no words");
    }
    if(r.top && (!r.logprobs || !r.top_ids || !r.top_logprobs))
        throw std::invalid_argument("native rounds that rank alternatives give their log-probabilities, ids and values");
}
}  // namespace

NativeEngine step_engine(std::string_view configuration_json, const StepCallbacks& callbacks, const StepRounds* rounds) {
    if(!callbacks.open || !callbacks.step) throw std::invalid_argument("a step engine needs open and step callbacks");
    auto s = settings(configuration_json);
    if(s.info.scoring && !callbacks.score) throw std::invalid_argument("a step engine that scores needs a score callback");
    if(rounds) check_rounds(*rounds, s);
    const auto profile = s.profile;
    const auto facts = s.facts;
    const auto cb = callbacks;
    auto requests = std::make_shared<RequestFacts>();
    NativeEngine native;
    native.engine = std::make_unique<Engine>(profile, std::make_shared<Host>(), [cb, rounds, s, requests](const Profile&, Services services) {
        return std::unique_ptr<Model>(std::make_unique<StepModel>(cb, rounds, s, services, requests));
    });
    native.concurrent_requests = true;
    native.info = [facts] { return facts; };
    native.result = [requests](const Result& result) { return requests->take(result.request_id); };
    return native;
}
}
