#pragma once
// Recorded selection among interchangeable implementations (kernel variants,
// library algorithms, tile shapes). A choice is keyed by an operation shape and
// a power-of-two width bucket. Recorded choices are replayed exactly; timing
// happens only when the engine explicitly tunes, and the result is exported as
// a table for the frozen configuration. Header-only, C++20 standard library.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace crit::techniques::selection {

// Smallest power of two >= width, at least `minimum`, clipped to `cap` (0 = none).
inline std::uint32_t bucket(std::uint32_t width, std::uint32_t minimum = 1, std::uint32_t cap = 0) {
    if(width == 0 || minimum == 0 || (minimum & (minimum - 1)) != 0) throw std::invalid_argument("bucket needs width > 0 and a power-of-two minimum");
    if(cap && width > cap) throw std::invalid_argument("width exceeds the bucket cap");
    std::uint32_t b = minimum;
    while(b < width) {
        if(b > std::numeric_limits<std::uint32_t>::max() / 2) throw std::overflow_error("bucket overflow");
        b *= 2;
    }
    return cap && b > cap ? cap : b;
}

enum class Source { recorded, tuned, fallback };

struct Choice {
    std::uint32_t index = 0;
    Source source = Source::fallback;
};

class Table {
public:
    Table() = default;
    explicit Table(std::map<std::string, std::uint32_t> recorded, bool tuning = false)
        : recorded_(std::move(recorded)), tuning_(tuning) {}

    // `count` candidates in a stable order (for example a library's heuristic list
    // for the bucket width). `time_ms(i)` runs candidate i and returns its time, or
    // a negative value when the candidate cannot run; it is called only when tuning
    // and the key has no recorded choice. Ties keep the lower index.
    Choice choose(const std::string& key, std::uint32_t count, const std::function<double(std::uint32_t)>& time_ms = {}) {
        if(count == 0) throw std::invalid_argument("selection needs at least one candidate");
        if(const auto it = chosen_.find(key); it != chosen_.end()) return {std::min(it->second.index, count - 1), it->second.source};
        Choice c;
        if(const auto r = recorded_.find(key); r != recorded_.end()) {
            if(r->second >= count) throw std::out_of_range("recorded choice for " + key + " exceeds the candidate count");
            c = {r->second, Source::recorded};
        } else if(tuning_ && time_ms) {
            double best = std::numeric_limits<double>::infinity();
            for(std::uint32_t i = 0; i < count; ++i) {
                const double t = time_ms(i);
                if(t >= 0 && t < best) { best = t; c = {i, Source::tuned}; }
            }
        }
        chosen_.emplace(key, c);
        return c;
    }
    // Every choice made so far as a flat JSON object {"key": index, ...}, sorted by key.
    std::string to_json() const {
        std::string out = "{";
        for(const auto& [k, c] : chosen_) {
            if(out.size() > 1) out += ",";
            out += "\"" + k + "\":" + std::to_string(c.index);
        }
        return out + "}";
    }
    // Parses the flat object written by to_json (string keys without escapes,
    // non-negative integers). Anything else is rejected.
    static std::map<std::string, std::uint32_t> parse(const std::string& text) {
        std::map<std::string, std::uint32_t> table;
        std::size_t i = 0;
        auto ws = [&] { while(i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i; };
        auto expect = [&](char ch) { ws(); if(i >= text.size() || text[i] != ch) throw std::invalid_argument("malformed selection table"); ++i; };
        expect('{');
        ws();
        if(i < text.size() && text[i] == '}') { ++i; ws(); if(i != text.size()) throw std::invalid_argument("trailing data"); return table; }
        for(;;) {
            expect('"');
            const auto end = text.find('"', i);
            if(end == std::string::npos) throw std::invalid_argument("unterminated key");
            std::string key = text.substr(i, end - i);
            if(key.empty() || key.find('\\') != std::string::npos) throw std::invalid_argument("invalid key");
            i = end + 1;
            expect(':');
            ws();
            if(i >= text.size() || !std::isdigit(static_cast<unsigned char>(text[i]))) throw std::invalid_argument("index must be a non-negative integer");
            std::uint64_t v = 0;
            while(i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
                v = v * 10 + std::uint64_t(text[i++] - '0');
                if(v > std::numeric_limits<std::uint32_t>::max()) throw std::out_of_range("index too large");
            }
            if(!table.emplace(std::move(key), std::uint32_t(v)).second) throw std::invalid_argument("duplicate key");
            ws();
            if(i < text.size() && text[i] == ',') { ++i; continue; }
            expect('}');
            ws();
            if(i != text.size()) throw std::invalid_argument("trailing data");
            return table;
        }
    }
    bool tuning() const { return tuning_; }

private:
    std::map<std::string, std::uint32_t> recorded_;
    std::map<std::string, Choice> chosen_;
    bool tuning_ = false;
};

} // namespace crit::techniques::selection
