#include "crit/step_engine.h"
#include "crit/json.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace crit::core {
namespace {
// Keys crit_engine_info writes itself; facts never repeat them.
constexpr std::array<std::string_view, 17> info_keys{"implementation", "vocabulary", "valid_vocabulary", "context",
    "prefill_chunk", "stop_tokens", "generation", "scoring", "query_scoring", "request_context", "token_logprobs",
    "max_top_logprobs", "max_concurrency", "score_paths", "sampling", "request_handles", "max_concurrent_requests"};

void quoted(std::string& out, std::string_view text) {
    out += '"';
    for(const char c:text) {
        switch(c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if(static_cast<unsigned char>(c) < 0x20) {
                    char escape[8];
                    std::snprintf(escape, sizeof escape, "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += escape;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}
// A number as JSON text. A banned token's -inf logit bias is spelled as Python's json writes and reads it.
void real(std::string& out, double value) {
    if(!std::isfinite(value)) { out += std::isnan(value) ? "NaN" : value < 0 ? "-Infinity" : "Infinity"; return; }
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.17g", value);
    out += buffer;
}
// A parsed value written back as JSON text (the facts object crit_engine_info merges).
void write(std::string& out, const json::Value& value) {
    switch(value.kind()) {
        case json::Kind::null: out += "null"; break;
        case json::Kind::boolean: out += value.boolean() ? "true" : "false"; break;
        case json::Kind::integer:
            if(value.exceeds_64_bits()) throw std::invalid_argument("step engine facts hold an integer beyond 64 bits");
            out += value.i64() < 0 ? std::to_string(value.i64()) : std::to_string(value.u64());
            break;
        case json::Kind::real: real(out, value.real()); break;
        case json::Kind::string: quoted(out, value.string()); break;
        case json::Kind::array: {
            out += '[';
            bool first = true;
            for(const auto& item:value.array()) { if(!first) out += ','; first = false; write(out, item); }
            out += ']';
            break;
        }
        case json::Kind::object: {
            out += '{';
            bool first = true;
            for(const auto& member:value.object()) {
                if(!first) out += ',';
                first = false;
                quoted(out, member.key);
                out += ':';
                write(out, member.value);
            }
            out += '}';
            break;
        }
    }
}

struct Settings {
    ModelInfo info;
    Profile profile;
    std::uint32_t row_tokens = 1;
    std::string facts = "{}";
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
    const auto* host = e.find("host_limit");
    p.host_limit = host && !host->is_null() ? std::size_t(host->u64()) : std::size_t(1) << 30;
    // Device memory is the engine's own; Core reserves none of it.
    p.device_limit = std::numeric_limits<std::size_t>::max();
    if(const auto* facts = root.find("facts"); facts && !facts->is_null()) {
        if(facts->kind() != json::Kind::object) throw std::invalid_argument("step engine facts must be a JSON object");
        s.facts = "{";
        bool first = true;
        for(const auto& member:facts->object()) {
            if(std::find(info_keys.begin(), info_keys.end(), member.key) != info_keys.end()) continue;
            if(!first) s.facts += ',';
            first = false;
            quoted(s.facts, member.key);
            s.facts += ':';
            write(s.facts, member.value);
        }
        s.facts += '}';
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
    if(dynamic_cast<const PlainDecoding*>(o.context.get())) { key("plain"); out += "true"; }
    return out + "}";
}

struct Host final : Backend {
    void bind_thread() override {}
    void drain() override {}
};

class StepModel final : public ConcurrentModel {
    struct Receipt final : Retirement { bool ready() override { return true; } };
    struct Session final : ConcurrentSession {
        StepModel* model;
        std::uint64_t handle;
        std::optional<std::uint32_t> logprobs;
        std::vector<Token> tokens;
        std::vector<float> token_logprobs;
        std::vector<TokenLogprob> top;
        Session(StepModel* m, std::uint64_t h, std::optional<std::uint32_t> n) : model(m), handle(h), logprobs(n) {}
        ~Session() override { model->release(handle); }
        void close(Finish) noexcept override {}
    };

    StepCallbacks cb_;
    Settings settings_;
    Resources& resources_;
    std::vector<StepRow> rows_;
    std::vector<StepProgress> progress_;
    std::vector<std::int32_t> tokens_, top_ids_;
    std::vector<float> logprobs_, top_values_;
    std::vector<StepDecision> decisions_;
    Reservation scores_;

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
        call(cb_.step(cb_.user, rows_.data(), std::uint32_t(n), progress_.data(), tokens_.data(), logprobs ? logprobs_.data() : nullptr,
                      logprobs ? top_ids_.data() : nullptr, logprobs ? top_values_.data() : nullptr), "step");
        std::vector<RowProgress> out;
        out.reserve(n);
        for(std::size_t i = 0; i < n; ++i) {
            const auto& p = progress_[i];
            auto& session = *static_cast<Session*>(items[i].session);
            if(p.tokens > row) throw std::runtime_error("step engine reported more tokens than max_row_tokens");
            if(p.reused_prompt_tokens > p.prompt_tokens) throw std::runtime_error("step engine reused more prompt tokens than it reported");
            session.tokens.assign(tokens_.begin() + std::ptrdiff_t(i * row), tokens_.begin() + std::ptrdiff_t(i * row + p.tokens));
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
    StepModel(const StepCallbacks& callbacks, Settings configured, Services services)
        : cb_(callbacks), settings_(std::move(configured)), resources_(services.resources) {}
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
            return std::make_unique<Session>(this, handle, view.options->logprobs);
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
        if(!cb_.commit) return;
        decisions_.clear();
        for(const auto& d:decisions)
            decisions_.push_back({static_cast<Session*>(d.session)->handle, d.accepted, d.terminal ? 1 : 0, d.cancelled ? 1 : 0, d.discard ? 1 : 0});
        call(cb_.commit(cb_.user, decisions_.data(), std::uint32_t(decisions_.size())), "commit");
    }
    std::unique_ptr<Retirement> retire(RequestKey, ConcurrentSession& session, Finish finish) override {
        ordinary();
        if(cb_.retire) call(cb_.retire(cb_.user, static_cast<Session&>(session).handle, int(finish)), "retire");
        return std::make_unique<Receipt>();
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

NativeEngine step_engine(std::string_view configuration_json, const StepCallbacks& callbacks) {
    if(!callbacks.open || !callbacks.step) throw std::invalid_argument("a step engine needs open and step callbacks");
    auto s = settings(configuration_json);
    if(s.info.scoring && !callbacks.score) throw std::invalid_argument("a step engine that scores needs a score callback");
    const auto profile = s.profile;
    const auto facts = s.facts;
    NativeEngine native;
    native.engine = std::make_unique<Engine>(profile, std::make_shared<Host>(), [callbacks, s](const Profile&, Services services) {
        return std::unique_ptr<Model>(std::make_unique<StepModel>(callbacks, s, services));
    });
    native.concurrent_requests = true;
    native.info = [facts] { return facts; };
    return native;
}
}
