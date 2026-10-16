#include "crit/constraints.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace crit::core {
namespace {
constexpr std::size_t no_node = std::numeric_limits<std::size_t>::max();
void check_token(Token token, std::size_t size) {
    if(token < 0 || std::size_t(token) >= size) throw std::out_of_range("constraint token outside the vocabulary");
}
}

ByteVocabulary::ByteVocabulary(std::vector<std::string> pieces, Token eos) : pieces_(std::move(pieces)), eos_(eos) {
    if(pieces_.empty() || pieces_.size() > constraint_vocabulary_limit)
        throw std::invalid_argument("constraint vocabulary must hold 1 to 1048576 pieces");
    if(eos < 0 || std::size_t(eos) >= pieces_.size() || !pieces_[std::size_t(eos)].empty())
        throw std::invalid_argument("constraint EOS must be in the vocabulary with an empty piece");
    std::size_t bytes = 0;
    for(const auto& piece:pieces_) {
        if(piece.size() > constraint_vocabulary_bytes - bytes) throw std::length_error("constraint vocabulary exceeds 64 MiB");
        bytes += piece.size();
    }
}
std::string_view ByteVocabulary::piece(Token token) const {
    check_token(token, pieces_.size());
    return pieces_[std::size_t(token)];
}

// ---- Finite choices --------------------------------------------------------------------------------------------
ChoiceGrammar::ChoiceGrammar(std::span<const std::string> choices) {
    if(choices.empty() || choices.size() > 65536) throw std::invalid_argument("choice grammar needs 1 to 65536 choices");
    struct Building { std::vector<std::pair<unsigned char, std::uint32_t>> edges; bool terminal = false; };
    std::vector<Building> nodes(1);
    std::size_t bytes = 0;
    for(const auto& choice:choices) {
        if(choice.size() > 1048576 - bytes) throw std::length_error("choice grammar exceeds 1 MiB of choices");
        bytes += choice.size();
        std::size_t node = 0;
        for(const char raw:choice) {
            const auto byte = static_cast<unsigned char>(raw);
            auto& edges = nodes[node].edges;
            const auto found = std::find_if(edges.begin(), edges.end(), [&](const auto& e) { return e.first == byte; });
            if(found != edges.end()) { node = found->second; continue; }
            const auto next = static_cast<std::uint32_t>(nodes.size());
            edges.emplace_back(byte, next);
            nodes.emplace_back();
            node = next;
        }
        nodes[node].terminal = true;
    }
    first_.reserve(nodes.size() + 1);
    terminal_.reserve(nodes.size());
    for(auto& node:nodes) {
        first_.push_back(static_cast<std::uint32_t>(edges_.size()));
        std::sort(node.edges.begin(), node.edges.end());
        for(const auto& [byte, next]:node.edges) edges_.push_back({byte, next});
        terminal_.push_back(node.terminal);
    }
    first_.push_back(static_cast<std::uint32_t>(edges_.size()));
}
bool ChoiceGrammar::terminal(std::size_t node) const { return terminal_.at(node); }
std::size_t ChoiceGrammar::walk(std::size_t node, std::string_view bytes) const {
    if(node >= terminal_.size()) throw std::out_of_range("choice grammar node out of range");
    for(const char raw:bytes) {
        const auto byte = static_cast<unsigned char>(raw);
        const auto begin = edges_.begin() + first_[node], end = edges_.begin() + first_[node + 1];
        const auto found = std::lower_bound(begin, end, byte, [](const Edge& e, unsigned char b) { return e.byte < b; });
        if(found == end || found->byte != byte) return no_node;
        node = found->next;
    }
    return node;
}

ChoiceMatcher::ChoiceMatcher(std::shared_ptr<const ChoiceGrammar> grammar, std::shared_ptr<const ByteVocabulary> vocabulary)
    : grammar_(std::move(grammar)), vocabulary_(std::move(vocabulary)) {
    if(!grammar_ || !vocabulary_) throw std::invalid_argument("choice matcher needs a grammar and a vocabulary");
}
std::unique_ptr<TokenMatcher> ChoiceMatcher::fork() const { return std::make_unique<ChoiceMatcher>(*this); }
std::size_t ChoiceMatcher::vocabulary() const noexcept { return vocabulary_->size(); }
std::size_t ChoiceMatcher::next(Token token) const {
    const auto piece = vocabulary_->piece(token);
    const auto& here = history_.back();
    if(here.ended) return no_node;
    if(token == vocabulary_->eos()) return grammar_->terminal(here.node) ? here.node : no_node;
    return piece.empty() ? no_node : grammar_->walk(here.node, piece);
}
std::vector<Token> ChoiceMatcher::allowed() const {
    if(ended()) return {vocabulary_->eos()};
    std::vector<Token> ids;
    for(std::size_t i = 0; i < vocabulary_->size(); ++i) if(next(Token(i)) != no_node) ids.push_back(Token(i));
    if(ids.empty()) throw std::domain_error("no token can continue the choice language");
    return ids;
}
void ChoiceMatcher::append(Token token) {
    const auto node = next(token);
    if(node == no_node) throw std::domain_error("token violates the choice language");
    history_.push_back({node, token == vocabulary_->eos()});
}
void ChoiceMatcher::rewind(std::size_t output_tokens) {
    if(output_tokens > size()) throw std::out_of_range("constraint rewind cannot extend output");
    history_.resize(output_tokens + 1);
}

// ---- JSON objects ----------------------------------------------------------------------------------------------
namespace constraints_detail {
enum class Phase : unsigned char { key_or_end, key, colon, object_value, object_separator, array_value_or_end,
                                   array_value, array_separator };
enum class Lex : unsigned char { normal, string, escape, hex, literal, number };
enum class Number : unsigned char { minus, zero, integer, dot, fraction, exponent, exponent_sign, exponent_digits };
// One byte-level parser frontier: the container stack plus the lexical state inside a token.
struct JsonCursor {
    std::vector<Phase> frames;
    Lex lex = Lex::normal;
    Number number = Number::zero;
    std::string_view literal;
    std::size_t literal_at = 0, hex_left = 0, utf8_left = 0, bytes = 0;
    unsigned char utf8_low = 0x80, utf8_high = 0xbf;
    bool string_key = false, started = false, complete = false, ended = false;

    static bool space(unsigned char c) noexcept { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
    static bool digit(unsigned char c) noexcept { return c >= '0' && c <= '9'; }
    void value_done() {
        if(frames.empty()) { complete = true; return; }
        switch(frames.back()) {
            case Phase::object_value: frames.back() = Phase::object_separator; break;
            case Phase::array_value_or_end: case Phase::array_value: frames.back() = Phase::array_separator; break;
            default: throw std::logic_error("invalid JSON value frontier");
        }
    }
    bool open(Phase phase, std::size_t depth) {
        if(frames.size() == depth) return false;
        frames.push_back(phase);
        started = true;
        return true;
    }
    bool start_value(unsigned char c, std::size_t depth) {
        if(c == '{') return open(Phase::key_or_end, depth);
        if(c == '[') return open(Phase::array_value_or_end, depth);
        if(c == '"') { lex = Lex::string; string_key = false; return true; }
        if(c == 't' || c == 'f' || c == 'n') {
            lex = Lex::literal;
            literal = c == 't' ? "true" : c == 'f' ? "false" : "null";
            literal_at = 1;
            return true;
        }
        if(c == '-' || digit(c)) {
            lex = Lex::number;
            number = c == '-' ? Number::minus : c == '0' ? Number::zero : Number::integer;
            return true;
        }
        return false;
    }
    bool normal(unsigned char c, std::size_t depth) {
        if(space(c)) return true;
        if(complete) return false;
        if(frames.empty()) return !started && c == '{' && open(Phase::key_or_end, depth);
        const Phase phase = frames.back();
        if(c == '}' && (phase == Phase::key_or_end || phase == Phase::object_separator)) {
            frames.pop_back(); value_done(); return true;
        }
        if(c == ']' && (phase == Phase::array_value_or_end || phase == Phase::array_separator)) {
            frames.pop_back(); value_done(); return true;
        }
        switch(phase) {
            case Phase::key_or_end: case Phase::key:
                if(c != '"') return false;
                lex = Lex::string; string_key = true; return true;
            case Phase::colon:
                if(c != ':') return false;
                frames.back() = Phase::object_value; return true;
            case Phase::object_separator:
                if(c != ',') return false;
                frames.back() = Phase::key; return true;
            case Phase::array_separator:
                if(c != ',') return false;
                frames.back() = Phase::array_value; return true;
            case Phase::object_value: case Phase::array_value_or_end: case Phase::array_value:
                return start_value(c, depth);
        }
        return false;
    }
    bool string_byte(unsigned char c) {
        if(utf8_left) {
            if(c < utf8_low || c > utf8_high) return false;
            --utf8_left; utf8_low = 0x80; utf8_high = 0xbf;
            return true;
        }
        if(c == '"') {
            lex = Lex::normal;
            if(string_key) frames.back() = Phase::colon; else value_done();
            return true;
        }
        if(c == '\\') { lex = Lex::escape; return true; }
        if(c < 0x20) return false;
        if(c < 0x80) return true;
        utf8_low = 0x80; utf8_high = 0xbf;
        if(c >= 0xc2 && c <= 0xdf) utf8_left = 1;
        else if(c >= 0xe0 && c <= 0xef) {
            utf8_left = 2;
            if(c == 0xe0) utf8_low = 0xa0;
            if(c == 0xed) utf8_high = 0x9f;
        } else if(c >= 0xf0 && c <= 0xf4) {
            utf8_left = 3;
            if(c == 0xf0) utf8_low = 0x90;
            if(c == 0xf4) utf8_high = 0x8f;
        } else return false;
        return true;
    }
    bool number_byte(unsigned char c, std::size_t depth) {
        switch(number) {
            case Number::minus:
                if(!digit(c)) return false;
                number = c == '0' ? Number::zero : Number::integer; return true;
            case Number::zero: if(digit(c)) return false; break;
            case Number::integer: if(digit(c)) return true; break;
            case Number::dot:
                if(!digit(c)) return false;
                number = Number::fraction; return true;
            case Number::fraction: if(digit(c)) return true; break;
            case Number::exponent:
                if(c == '+' || c == '-') { number = Number::exponent_sign; return true; }
                if(!digit(c)) return false;
                number = Number::exponent_digits; return true;
            case Number::exponent_sign:
                if(!digit(c)) return false;
                number = Number::exponent_digits; return true;
            case Number::exponent_digits: if(digit(c)) return true; break;
        }
        if(c == '.' && (number == Number::zero || number == Number::integer)) { number = Number::dot; return true; }
        if((c == 'e' || c == 'E') && (number == Number::zero || number == Number::integer || number == Number::fraction)) {
            number = Number::exponent; return true;
        }
        if(!space(c) && c != ',' && c != ']' && c != '}') return false;
        lex = Lex::normal;
        value_done();
        return normal(c, depth);
    }
    bool consume(unsigned char c, std::size_t depth) {
        switch(lex) {
            case Lex::normal: return normal(c, depth);
            case Lex::string: return string_byte(c);
            case Lex::escape:
                if(c == 'u') { lex = Lex::hex; hex_left = 4; return true; }
                if(std::string_view("\"\\/bfnrt").find(static_cast<char>(c)) == std::string_view::npos) return false;
                lex = Lex::string; return true;
            case Lex::hex:
                if(!digit(c) && !(c >= 'a' && c <= 'f') && !(c >= 'A' && c <= 'F')) return false;
                if(--hex_left == 0) lex = Lex::string;
                return true;
            case Lex::literal:
                if(c != static_cast<unsigned char>(literal[literal_at])) return false;
                if(++literal_at == literal.size()) { lex = Lex::normal; value_done(); }
                return true;
            case Lex::number: return number_byte(c, depth);
        }
        return false;
    }
};
}
using constraints_detail::JsonCursor;

JsonObjectMatcher::JsonObjectMatcher(std::shared_ptr<const ByteVocabulary> vocabulary, JsonLimits limits)
    : vocabulary_(std::move(vocabulary)), limits_(limits), history_(1) {
    if(!vocabulary_ || !limits.depth || limits.depth > 256 || !limits.bytes || limits.bytes > (std::size_t{1} << 30) ||
       !limits.tokens || limits.tokens > 1048576)
        throw std::invalid_argument("invalid JSON matcher vocabulary or limits");
}
JsonObjectMatcher::JsonObjectMatcher(const JsonObjectMatcher&) = default;
JsonObjectMatcher::~JsonObjectMatcher() = default;
std::unique_ptr<TokenMatcher> JsonObjectMatcher::fork() const { return std::make_unique<JsonObjectMatcher>(*this); }
std::size_t JsonObjectMatcher::size() const noexcept { return history_.size() - 1; }
std::size_t JsonObjectMatcher::vocabulary() const noexcept { return vocabulary_->size(); }
bool JsonObjectMatcher::ended() const noexcept { return history_.back().ended; }
bool JsonObjectMatcher::complete() const noexcept {
    const auto& here = history_.back();
    return here.complete && here.lex == constraints_detail::Lex::normal;
}
bool JsonObjectMatcher::trial(Token token, JsonCursor& out) const {
    const auto piece = vocabulary_->piece(token);
    const JsonCursor& here = history_.back();
    if(here.ended || size() == limits_.tokens) return false;
    if(token == vocabulary_->eos()) {
        if(!complete()) return false;
        out = here;
        out.ended = true;
        return true;
    }
    if(piece.empty() || piece.size() > limits_.bytes - here.bytes) return false;
    out = here;
    for(const char raw:piece) if(!out.consume(static_cast<unsigned char>(raw), limits_.depth)) return false;
    out.bytes += piece.size();
    return true;
}
std::vector<Token> JsonObjectMatcher::allowed() const {
    if(ended()) return {vocabulary_->eos()};
    std::vector<Token> ids;
    JsonCursor scratch;
    for(std::size_t i = 0; i < vocabulary_->size(); ++i) if(trial(Token(i), scratch)) ids.push_back(Token(i));
    if(ids.empty()) throw std::domain_error("no token can continue the JSON object within its limits");
    return ids;
}
void JsonObjectMatcher::append(Token token) {
    JsonCursor next;
    if(!trial(token, next)) throw std::domain_error("token violates JSON object syntax or limits");
    history_.push_back(std::move(next));
}
void JsonObjectMatcher::rewind(std::size_t output_tokens) {
    if(output_tokens > size()) throw std::out_of_range("constraint rewind cannot extend output");
    history_.resize(output_tokens + 1);
}

// ---- Readout ---------------------------------------------------------------------------------------------------
ConstraintReadout constraint_readout(std::span<const float> logits, std::span<const Token> allowed) {
    constexpr float infinity = std::numeric_limits<float>::infinity();
    if(allowed.empty() || allowed.size() > logits.size())
        throw std::invalid_argument("allowed tokens must be nonempty and fit the logits row");
    double all_max = -infinity, set_max = -infinity;
    for(const float x:logits) {
        if(std::isnan(x) || x == infinity) throw std::invalid_argument("logits must be finite or -inf");
        all_max = std::max(all_max, double(x));
    }
    Token previous = -1;
    for(const Token id:allowed) {
        if(id <= previous || std::size_t(id) >= logits.size())
            throw std::invalid_argument("allowed tokens must be sorted, unique and in the logits row");
        previous = id;
        set_max = std::max(set_max, double(logits[std::size_t(id)]));
    }
    if(set_max == -double(infinity)) throw std::invalid_argument("no selectable token");
    // Separate shifted maxima keep the restricted law exact even when the original mass underflows to zero.
    double total = 0, restricted = 0;
    for(const float x:logits) if(x != -infinity) total += std::exp(double(x) - all_max);
    ConstraintReadout result;
    result.ids.assign(allowed.begin(), allowed.end());
    result.probabilities.reserve(allowed.size());
    for(const Token id:allowed) {
        const float x = logits[std::size_t(id)];
        const double weight = x == -infinity ? 0.0 : std::exp(double(x) - set_max);
        result.probabilities.push_back(weight);
        restricted += weight;
    }
    for(double& p:result.probabilities) p /= restricted;
    result.original_mass = restricted*std::exp(set_max - all_max)/total;
    return result;
}
}
