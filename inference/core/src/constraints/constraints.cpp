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
// Allowed ids collected as a bitmask during a trie walk (any order), returned sorted.
struct Collected {
    std::vector<std::uint64_t> bits;
    explicit Collected(std::size_t vocabulary) : bits((vocabulary + 63) / 64) {}
    void add(Token t) { bits[std::size_t(t) / 64] |= std::uint64_t{1} << (std::size_t(t) % 64); }
    std::vector<Token> ids() const {
        std::size_t count = 0;
        for(const auto word:bits) for(auto w = word; w; w &= w - 1) ++count;
        std::vector<Token> out;
        out.reserve(count);
        for(std::size_t i = 0; i < bits.size(); ++i)
            for(auto w = bits[i]; w; w &= w - 1) {
                int bit = 0;
                while(!((w >> bit) & 1u)) ++bit;
                out.push_back(Token(i * 64 + std::size_t(bit)));
            }
        return out;
    }
};
void append_raw(std::string& key, std::uint64_t value) {
    char bytes[sizeof(value)];
    std::memcpy(bytes, &value, sizeof(value));
    key.append(bytes, sizeof(value));
}
std::string limits_key(std::initializer_list<std::size_t> values) {
    std::string key;
    for(const auto v:values) append_raw(key, v);
    return key;
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
        longest_ = std::max(longest_, piece.size());
    }
    // The trie: nonempty pieces in byte order, each sharing its longest common prefix with the one before.
    std::vector<Token> order;
    order.reserve(pieces_.size());
    for(std::size_t t = 0; t < pieces_.size(); ++t) if(!pieces_[t].empty()) order.push_back(Token(t));
    std::stable_sort(order.begin(), order.end(), [&](Token a, Token b) { return pieces_[std::size_t(a)] < pieces_[std::size_t(b)]; });
    std::vector<std::uint32_t> parent{0};
    std::vector<unsigned char> byte_of{0};
    std::vector<std::pair<std::uint32_t, Token>> endings;
    endings.reserve(order.size());
    std::vector<std::uint32_t> path{0};
    std::string_view previous;
    for(const Token t:order) {
        const std::string_view piece = pieces_[std::size_t(t)];
        std::size_t common = 0;
        while(common < piece.size() && common < previous.size() && piece[common] == previous[common]) ++common;
        path.resize(common + 1);
        for(std::size_t i = common; i < piece.size(); ++i) {
            parent.push_back(path.back());
            byte_of.push_back(static_cast<unsigned char>(piece[i]));
            path.push_back(std::uint32_t(parent.size() - 1));
        }
        endings.emplace_back(path.back(), t);
        previous = piece;
    }
    const std::size_t nodes = parent.size();
    child_begin_.assign(nodes + 1, 0);
    for(std::size_t n = 1; n < nodes; ++n) ++child_begin_[parent[n] + 1];
    for(std::size_t n = 0; n < nodes; ++n) child_begin_[n + 1] += child_begin_[n];
    edge_child_.resize(nodes - 1);
    edge_byte_.resize(nodes - 1);
    std::vector<std::uint32_t> fill(child_begin_.begin(), child_begin_.end() - 1);
    for(std::size_t n = 1; n < nodes; ++n) {   // creation order is byte order among siblings
        const auto at = fill[parent[n]]++;
        edge_child_[at] = std::uint32_t(n);
        edge_byte_[at] = byte_of[n];
    }
    token_begin_.assign(nodes + 1, 0);
    for(const auto& [node, token]:endings) ++token_begin_[node + 1];
    for(std::size_t n = 0; n < nodes; ++n) token_begin_[n + 1] += token_begin_[n];
    tokens_.resize(endings.size());
    std::vector<std::uint32_t> placed(token_begin_.begin(), token_begin_.end() - 1);
    for(const auto& [node, token]:endings) tokens_[placed[node]++] = token;
}
std::string_view ByteVocabulary::piece(Token token) const {
    check_token(token, pieces_.size());
    return pieces_[std::size_t(token)];
}

// ---- Allowed-set cache and shared helpers ------------------------------------------------------------------------
AllowedCache::AllowedCache(std::size_t max_bytes) : max_bytes_(max_bytes) {
    if(!max_bytes) throw std::invalid_argument("allowed cache needs a positive byte bound");
}
AllowedCache::Stats AllowedCache::stats() const {
    std::lock_guard lock(mutex_);
    return {hits_, misses_, entries_.size(), bytes_};
}
void AllowedCache::bind(const void* vocabulary, const void* language, std::string_view limits) {
    std::lock_guard lock(mutex_);
    if(!bound_) { vocabulary_ = vocabulary; language_ = language; limits_ = std::string(limits); bound_ = true; return; }
    if(vocabulary_ != vocabulary || language_ != language || limits_ != limits)
        throw std::invalid_argument("allowed cache is bound to another language, vocabulary or limits");
}
std::shared_ptr<const std::vector<Token>> AllowedCache::find(const std::string& key, std::size_t budget) {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(key);
    if(it == entries_.end() || it->second.needed > budget) { ++misses_; return {}; }
    ++hits_;
    order_.splice(order_.begin(), order_, it->second.order);
    return it->second.ids;
}
void AllowedCache::insert(const std::string& key, std::shared_ptr<const std::vector<Token>> ids, std::size_t needed) {
    const std::size_t bytes = key.size() + ids->size() * sizeof(Token) + 64;
    std::lock_guard lock(mutex_);
    if(bytes > max_bytes_ || entries_.count(key)) return;
    while(bytes_ + bytes > max_bytes_ && !order_.empty()) {
        const auto oldest = entries_.find(*order_.back());
        bytes_ -= oldest->second.bytes;
        order_.pop_back();
        entries_.erase(oldest);
    }
    const auto it = entries_.emplace(key, Entry{std::move(ids), needed, bytes, {}}).first;
    order_.push_front(&it->first);
    it->second.order = order_.begin();
    bytes_ += bytes;
}

void TokenMatcher::allowed_bits(std::span<std::uint32_t> words) const {
    if(words.size() != (vocabulary() + 31) / 32) throw std::invalid_argument("allowed bitmask must hold (vocabulary + 31) / 32 words");
    const auto ids = allowed();
    std::fill(words.begin(), words.end(), 0u);
    for(const Token t:ids) words[std::size_t(t) / 32] |= std::uint32_t{1} << (std::size_t(t) % 32);
}

std::vector<Token> allowed_by_scan(const TokenMatcher& matcher) {
    const auto vocabulary = matcher.vocabulary();
    std::vector<Token> ids;
    if(matcher.ended()) {
        for(std::size_t t = 0; t < vocabulary && ids.empty(); ++t) {   // after EOS only EOS stays allowed
            const auto copy = matcher.fork();
            copy->rewind(copy->size() - 1);
            try { copy->append(Token(t)); if(copy->ended()) ids.push_back(Token(t)); } catch(const std::domain_error&) {}
        }
        return ids;
    }
    for(std::size_t t = 0; t < vocabulary; ++t) {
        auto copy = matcher.fork();
        try { copy->append(Token(t)); ids.push_back(Token(t)); } catch(const std::domain_error&) {}
    }
    if(ids.empty()) throw std::domain_error("no token can continue the constraint");
    return ids;
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

ChoiceMatcher::ChoiceMatcher(std::shared_ptr<const ChoiceGrammar> grammar, std::shared_ptr<const ByteVocabulary> vocabulary,
                             std::shared_ptr<AllowedCache> cache)
    : grammar_(std::move(grammar)), vocabulary_(std::move(vocabulary)), cache_(std::move(cache)) {
    if(!grammar_ || !vocabulary_) throw std::invalid_argument("choice matcher needs a grammar and a vocabulary");
    if(!cache_) cache_ = std::make_shared<AllowedCache>(allowed_cache_private_bytes);
    cache_->bind(vocabulary_.get(), grammar_.get(), "choice");
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
    const std::size_t node = history_.back().node;
    std::string key;
    append_raw(key, node);
    if(const auto hit = cache_->find(key, 0)) return *hit;
    // The choice trie walked together with the piece trie: each shared piece prefix is followed once.
    Collected collected(vocabulary_->size());
    std::vector<std::size_t> states(vocabulary_->longest_piece() + 1);
    states[0] = node;
    vocabulary_->walk(vocabulary_->longest_piece(), [&](std::size_t depth, unsigned char byte) {
        states[depth + 1] = grammar_->walk(states[depth], std::string_view(reinterpret_cast<const char*>(&byte), 1));
        return states[depth + 1] != no_node;
    }, [&](Token t) { collected.add(t); });
    if(grammar_->terminal(node)) collected.add(vocabulary_->eos());
    auto ids = collected.ids();
    if(ids.empty()) throw std::domain_error("no token can continue the choice language");
    cache_->insert(key, std::make_shared<const std::vector<Token>>(ids), 0);
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

JsonObjectMatcher::JsonObjectMatcher(std::shared_ptr<const ByteVocabulary> vocabulary, JsonLimits limits, std::shared_ptr<AllowedCache> cache)
    : vocabulary_(std::move(vocabulary)), limits_(limits), cache_(std::move(cache)), history_(1) {
    if(!vocabulary_ || !limits.depth || limits.depth > 256 || !limits.bytes || limits.bytes > (std::size_t{1} << 30) ||
       !limits.tokens || limits.tokens > 1048576)
        throw std::invalid_argument("invalid JSON matcher vocabulary or limits");
    if(!cache_) cache_ = std::make_shared<AllowedCache>(allowed_cache_private_bytes);
    static const char json_language = 0;
    cache_->bind(vocabulary_.get(), &json_language, limits_key({limits.depth, limits.bytes, limits.tokens}));
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
    const JsonCursor& here = history_.back();
    if(size() == limits_.tokens) throw std::domain_error("no token can continue the JSON object within its limits");
    // The allowed set is a function of the cursor and of the byte room up to the longest piece.
    const std::size_t room = std::min(limits_.bytes - here.bytes, vocabulary_->longest_piece());
    std::string key;
    key.reserve(48 + here.frames.size());
    for(const std::size_t v:{std::size_t(here.lex), std::size_t(here.number), here.literal.empty() ? 0 : std::size_t(here.literal[0]),
                             here.literal_at, here.hex_left, here.utf8_left, std::size_t(here.utf8_low), std::size_t(here.utf8_high),
                             std::size_t(here.string_key) | std::size_t(here.started) << 1 | std::size_t(here.complete) << 2, room})
        append_raw(key, v);
    for(const auto frame:here.frames) key.push_back(static_cast<char>(frame));
    if(const auto hit = cache_->find(key, 0)) return *hit;
    Collected collected(vocabulary_->size());
    std::vector<JsonCursor> states(room + 1);
    states[0] = here;
    vocabulary_->walk(room, [&](std::size_t depth, unsigned char byte) {
        states[depth + 1] = states[depth];
        return states[depth + 1].consume(byte, limits_.depth);
    }, [&](Token t) { collected.add(t); });
    if(complete()) collected.add(vocabulary_->eos());
    auto ids = collected.ids();
    if(ids.empty()) throw std::domain_error("no token can continue the JSON object within its limits");
    cache_->insert(key, std::make_shared<const std::vector<Token>>(ids), 0);
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
