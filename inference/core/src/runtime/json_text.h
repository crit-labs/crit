#pragma once
// JSON text Core writes for engines: crit_engine_info and a generation's result, a step engine's request options, and
// the facts an engine adds to crit_engine_info, written back from their parsed form.
#include "crit/json.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

namespace crit::core::detail {
// Keys crit_engine_info writes itself; an engine's facts never repeat them.
inline constexpr std::array<std::string_view, 17> info_keys{"implementation", "vocabulary", "valid_vocabulary",
    "context", "prefill_chunk", "stop_tokens", "generation", "scoring", "query_scoring", "request_context",
    "token_logprobs", "max_top_logprobs", "max_concurrency", "score_paths", "sampling", "request_handles",
    "max_concurrent_requests"};

inline void quoted(std::string& out, std::string_view text) {
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
inline std::string quoted(std::string_view text) {
    std::string out;
    quoted(out, text);
    return out;
}
// A number as JSON text, the shortest that reads back as the same double. A banned token's -inf logit bias is spelled
// as Python's json writes and reads it.
inline void real(std::string& out, double value) {
    if(!std::isfinite(value)) { out += std::isnan(value) ? "NaN" : value < 0 ? "-Infinity" : "Infinity"; return; }
    char buffer[32];
    const auto written = std::to_chars(buffer, buffer + sizeof buffer, value);
    out.append(buffer, written.ptr);
}
inline std::string real(double value) {
    std::string out;
    real(out, value);
    return out;
}
// A parsed value written back as JSON text.
inline void write(std::string& out, const json::Value& value) {
    switch(value.kind()) {
        case json::Kind::null: out += "null"; break;
        case json::Kind::boolean: out += value.boolean() ? "true" : "false"; break;
        case json::Kind::integer:
            if(value.exceeds_64_bits()) throw std::invalid_argument("engine facts hold an integer beyond 64 bits");
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
// An engine's facts object written back without the keys crit_engine_info writes itself: Core's values stand.
inline std::string facts_object(const json::Value& facts) {
    if(facts.kind() != json::Kind::object) throw std::invalid_argument("engine facts must be a JSON object");
    std::string out = "{";
    for(const auto& member:facts.object()) {
        if(std::find(info_keys.begin(), info_keys.end(), member.key) != info_keys.end()) continue;
        if(out.size() > 1) out += ',';
        quoted(out, member.key);
        out += ':';
        write(out, member.value);
    }
    return out + '}';
}
}
