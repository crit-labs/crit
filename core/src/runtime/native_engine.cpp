#include "crit/native_engine.h"
#include "crit/json.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

namespace crit::core::native_api {
namespace {
thread_local std::string last;

struct Cancelled {};

void fail(const char* what) noexcept {
    try {
        last = what;
    } catch(...) {
    }
}

std::string json_string(std::string_view text) {
    std::string out = "\"";
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
    return out + "\"";
}

std::string number(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.6f", value);
    return buffer;
}

std::string tokens(std::span<const Token> values) {
    std::string out = "[";
    for(std::size_t i = 0; i < values.size(); ++i) out += (i ? "," : "") + std::to_string(values[i]);
    return out + "]";
}

// `base` is a JSON object; the members of the factory's object (validated) are appended to it.
std::string merged(std::string base, const std::function<std::string()>& extra) {
    if(!extra) return base;
    const std::string text = extra();
    const auto document = json::Document::parse(text, "engine facts");
    if(document.root().kind() != json::Kind::object) throw std::runtime_error("engine facts must be a JSON object");
    const auto open = text.find('{'), close = text.rfind('}');
    const std::string members = text.substr(open + 1, close - open - 1);
    if(members.find_first_not_of(" \t\r\n") == std::string::npos) return base;
    base.pop_back();
    return base + "," + members + "}";
}

int copy(const std::string& text, char* out, int capacity) {
    if(out && capacity > 0) {
        const auto n = std::min<std::size_t>(text.size(), std::size_t(capacity - 1));
        std::memcpy(out, text.data(), n);
        out[n] = '\0';
    }
    return int(text.size());
}

const char* finish_name(Finish finish) {
    switch(finish) {
        case Finish::length: return "length";
        case Finish::stop: return "stop";
        case Finish::cancelled: return "cancelled";
        case Finish::failed: return "failed";
    }
    return "unknown";
}

NativeEngine& engine_of(void* handle) {
    if(!handle) throw std::invalid_argument("null engine handle");
    auto& native = *static_cast<NativeEngine*>(handle);
    if(!native.engine) throw std::logic_error("engine is closed");
    return native;
}

std::vector<Token> token_array(const json::Value& value, const char* name) {
    if(value.kind() != json::Kind::array) throw std::invalid_argument(std::string(name) + " must be an array of token ids");
    std::vector<Token> out;
    for(const auto& item:value.array()) out.push_back(Token(item.i64()));
    return out;
}

RequestOptions request_options(const char* options_json) {
    RequestOptions options;
    if(!options_json || !*options_json) return options;
    const auto document = json::Document::parse(options_json, "generate options");
    const auto& root = document.root();
    if(root.kind() != json::Kind::object) throw std::invalid_argument("generate options must be a JSON object");
    if(const auto* stops = root.find("stop_tokens"); stops && !stops->is_null()) options.stop_tokens = token_array(*stops, "stop_tokens");
    auto& sampling = options.sampling;
    if(const auto* v = root.find("temperature"); v && !v->is_null()) sampling.temperature = v->real();
    if(const auto* v = root.find("top_p"); v && !v->is_null()) sampling.top_p = v->real();
    if(const auto* v = root.find("top_k"); v && !v->is_null()) sampling.top_k = std::uint32_t(v->u64());
    if(const auto* v = root.find("min_p"); v && !v->is_null()) sampling.min_p = v->real();
    if(const auto* v = root.find("seed"); v && !v->is_null()) sampling.seed = v->u64();
    if(const auto* v = root.find("replay"); v && !v->is_null()) {
        auto replay = std::make_shared<ReplayContinuation>();
        replay->tokens = token_array(*v, "replay");
        options.context = std::move(replay);
    }
    if(const auto* v = root.find("plain"); v && !v->is_null() && v->boolean()) {
        if(options.context) throw std::invalid_argument("plain and replay exclude each other");
        options.context = std::make_shared<PlainDecoding>();
    }
    return options;
}
}  // namespace

void* open(NativeFactory factory, const char* configuration_json) noexcept {
    try {
        auto native = std::make_unique<NativeEngine>(factory(configuration_json ? configuration_json : "{}"));
        if(!native->engine) throw std::runtime_error("the engine factory returned no engine");
        return native.release();
    } catch(const std::exception& error) {
        fail(error.what());
    } catch(...) {
        fail("unknown engine factory failure");
    }
    return nullptr;
}

void close(void* handle) noexcept {
    if(!handle) return;
    auto* native = static_cast<NativeEngine*>(handle);
    // The engine retires first; its owner (backend, borrowed resources) after it.
    native->engine.reset();
    native->info = nullptr;
    native->request = nullptr;
    delete native;
}

int info(void* handle, char* out, int capacity) noexcept {
    try {
        auto& native = engine_of(handle);
        const ModelInfo i = native.engine->info();
        std::string text = "{\"implementation\":" + json_string(i.implementation) +
                           ",\"vocabulary\":" + std::to_string(i.vocabulary) +
                           ",\"valid_vocabulary\":" + std::to_string(i.valid_vocabulary ? i.valid_vocabulary : i.vocabulary) +
                           ",\"context\":" + std::to_string(i.context) + ",\"prefill_chunk\":" + std::to_string(i.prefill_chunk) +
                           ",\"stop_tokens\":" + tokens(i.stop_tokens) + ",\"generation\":" + (i.generation ? "true" : "false") +
                           ",\"scoring\":" + (i.scoring ? "true" : "false") + ",\"query_scoring\":" + (i.query_scoring ? "true" : "false") +
                           ",\"request_context\":" + (i.request_context ? "true" : "false") +
                           ",\"max_concurrency\":" + std::to_string(i.max_concurrency) + ",\"score_paths\":[";
        for(std::size_t p = 0; p < i.score_paths.size(); ++p) text += (p ? "," : "") + json_string(i.score_paths[p]);
        const auto& s = i.sampling;
        text += std::string("],\"sampling\":{\"temperature\":") + (s.temperature ? "true" : "false") + ",\"top_p\":" + (s.top_p ? "true" : "false") +
                ",\"top_k\":" + (s.top_k ? "true" : "false") + ",\"seed\":" + (s.seed ? "true" : "false") +
                ",\"min_p\":" + (s.min_p ? "true" : "false") + "}}";
        return copy(merged(std::move(text), native.info), out, capacity);
    } catch(const std::exception& error) {
        fail(error.what());
    } catch(...) {
        fail("unknown engine info failure");
    }
    return -1;
}

int generate(void* handle, const std::int32_t* prompt, int count, int output_tokens, const char* options_json,
             TokenCallback callback, void* user, std::int32_t* out, char* result, int capacity) noexcept {
    try {
        auto& native = engine_of(handle);
        if(!prompt || count <= 0 || output_tokens <= 0 || !out) throw std::invalid_argument("generate needs a prompt, an output count and an output buffer");
        auto request = native.engine->submit(std::vector<Token>(prompt, prompt + count), std::uint32_t(output_tokens),
                                             request_options(options_json));
        int written = 0;
        Result r;
        try {
            r = request.wait([&](Token token) {
                if(written < output_tokens) out[written++] = token;
                if(callback && callback(user, &token, 1)) throw Cancelled{};
            });
        } catch(const Cancelled&) {
            return -2;
        }
        const auto ms = [](std::uint64_t ns) { return number(double(ns) / 1e6); };
        const auto& t = r.timing;
        std::string text = std::string("{\"finish\":\"") + finish_name(r.finish) + "\",\"timing\":{\"queued_ms\":" + ms(t.queued_ns) +
                           ",\"prefill_ms\":" + ms(t.prefill_ns) + ",\"first_token_ms\":" + ms(t.first_token_ns) +
                           ",\"decode_ms\":" + ms(t.decode_ns) + ",\"total_ms\":" + ms(t.total_ns) +
                           ",\"prefill_tokens\":" + std::to_string(t.prefill_tokens) +
                           ",\"reused_prompt_tokens\":" + std::to_string(t.reused_prompt_tokens) +
                           ",\"decode_steps\":" + std::to_string(t.decode_steps) + "}}";
        copy(merged(std::move(text), native.request), result, capacity);
        return written;
    } catch(const std::exception& error) {
        fail(error.what());
    } catch(...) {
        fail("unknown generation failure");
    }
    return -1;
}

int score(void* handle, const std::int32_t* ids, int count, int first_target, const char* path,
          const std::int32_t* queries, int per_target, float* target, float* query, std::int32_t* argmax,
          int* has_argmax) noexcept {
    try {
        auto& native = engine_of(handle);
        if(!ids || count <= 0 || first_target < 0 || first_target >= count || !target || per_target < 0)
            throw std::invalid_argument("score needs ids, a first target inside them and a target buffer");
        const auto targets = std::size_t(count - first_target);
        std::vector<Token> sequence(ids, ids + count);
        Handle request;
        // Query scoring also returns the argmax, so engines that support it take that path even without queries.
        if(per_target > 0 || (path && *path) || native.engine->info().query_scoring) {
            ScoreQueries q;
            q.per_target = std::uint32_t(per_target);
            if(per_target) q.ids.assign(queries, queries + targets * std::size_t(per_target));
            q.path = path ? path : "";
            request = native.engine->submit_score(std::move(sequence), std::uint32_t(first_target), std::move(q));
        } else {
            request = native.engine->submit_score(std::move(sequence), std::uint32_t(first_target));
        }
        const Result r = request.wait();
        if(r.scores.size() != targets) throw std::runtime_error("engine returned the wrong number of target scores");
        std::copy(r.scores.begin(), r.scores.end(), target);
        if(per_target && query) std::copy(r.query_scores.begin(), r.query_scores.end(), query);
        const bool argmaxed = r.argmax.size() == targets;
        if(argmaxed && argmax) std::copy(r.argmax.begin(), r.argmax.end(), argmax);
        if(has_argmax) *has_argmax = argmaxed ? 1 : 0;
        return 0;
    } catch(const std::exception& error) {
        fail(error.what());
    } catch(...) {
        fail("unknown scoring failure");
    }
    return -1;
}

const char* last_error() noexcept { return last.c_str(); }
}  // namespace crit::core::native_api
