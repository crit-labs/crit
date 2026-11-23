#include "crit/constraints.h"
#include <algorithm>
#include <charconv>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace crit::core {
namespace {
constexpr std::size_t rule_limit = 4096, production_limit = 16384, symbol_limit = 262144;
constexpr std::size_t artifact_limit = std::size_t{4} << 20;

// Whitespace-separated decimal fields of an artifact.
class Fields {
public:
    explicit Fields(std::string_view text) : text_(text) {}
    bool word(std::string_view& out) {
        skip();
        const std::size_t begin = at_;
        while(at_ < text_.size() && !space(text_[at_])) ++at_;
        out = text_.substr(begin, at_ - begin);
        return !out.empty();
    }
    bool number(std::size_t& out) {
        std::string_view digits;
        if(!word(digits) || digits.size() > 9) return false;
        const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), out);
        return error == std::errc() && end == digits.data() + digits.size();
    }
    bool done() { skip(); return at_ == text_.size(); }
private:
    std::string_view text_;
    std::size_t at_ = 0;
    static bool space(char c) noexcept { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }
    void skip() { while(at_ < text_.size() && space(text_[at_])) ++at_; }
};
// The items of the chart being closed, as packed keys in an open-addressing table that is stamped instead of
// cleared and kept per thread, so closing a chart allocates only when a chart outgrows every earlier one.
// production < 2^14, dot < 2^19 and origin < 2^21 by the grammar and matcher limits.
class ItemSet {
public:
    void reset() {
        if(keys_.empty()) grow(256);
        count_ = 0;
        if(++stamp_ == 0) { std::fill(stamps_.begin(), stamps_.end(), 0u); stamp_ = 1; }
    }
    bool insert(const GrammarMatcher::Item& item) {   // true when the item is new
        const std::uint64_t key = std::uint64_t(item.production) << 40 | std::uint64_t(item.dot) << 21 | item.origin;
        if(2 * (count_ + 1) > keys_.size()) rehash();
        for(std::size_t i = hash(key) & (keys_.size() - 1);; i = (i + 1) & (keys_.size() - 1)) {
            if(stamps_[i] != stamp_) { stamps_[i] = stamp_; keys_[i] = key; ++count_; return true; }
            if(keys_[i] == key) return false;
        }
    }
private:
    std::vector<std::uint64_t> keys_;
    std::vector<std::uint32_t> stamps_;
    std::uint32_t stamp_ = 0;
    std::size_t count_ = 0;
    static std::size_t hash(std::uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdull; return std::size_t(k ^ (k >> 33)); }
    void grow(std::size_t size) { keys_.assign(size, 0); stamps_.assign(size, 0); }
    void rehash() {
        std::vector<std::uint64_t> live;
        live.reserve(count_);
        for(std::size_t i = 0; i < keys_.size(); ++i) if(stamps_[i] == stamp_) live.push_back(keys_[i]);
        grow(keys_.size() * 2);
        stamp_ = 1;
        count_ = 0;
        for(const auto key:live)
            for(std::size_t i = hash(key) & (keys_.size() - 1);; i = (i + 1) & (keys_.size() - 1))
                if(stamps_[i] != stamp_) { stamps_[i] = stamp_; keys_[i] = key; ++count_; break; }
    }
};
thread_local ItemSet closing;
}

ByteGrammar::ByteGrammar(std::size_t rules, std::size_t start, std::vector<Production> productions)
    : start_(start), productions_(std::move(productions)) {
    if(!rules || rules > rule_limit || start >= rules || productions_.empty() || productions_.size() > production_limit)
        throw std::invalid_argument("invalid bounded byte grammar");
    rules_.resize(rules);
    nullable_.resize(rules);
    std::size_t symbols = 0;
    for(std::size_t i = 0; i < productions_.size(); ++i) {
        const auto& p = productions_[i];
        if(p.lhs >= rules || p.rhs.size() > symbol_limit - symbols) throw std::invalid_argument("invalid grammar production extent");
        symbols += p.rhs.size();
        for(const auto symbol:p.rhs) if(symbol >= 256 + rules) throw std::invalid_argument("undefined grammar symbol");
        rules_[p.lhs].push_back(i);
    }
    std::vector<bool> productive(rules);
    for(bool changed = true; changed;) {
        changed = false;
        for(const auto& p:productions_) {
            bool empty = true, ready = true;
            for(const auto s:p.rhs) {
                empty = empty && s >= 256 && nullable_[s - 256];
                ready = ready && (s < 256 || productive[s - 256]);
            }
            if(empty && !nullable_[p.lhs]) { nullable_[p.lhs] = true; changed = true; }
            if(ready && !productive[p.lhs]) { productive[p.lhs] = true; changed = true; }
        }
    }
    if(std::find(productive.begin(), productive.end(), false) != productive.end())
        throw std::invalid_argument("grammar has undefined or unproductive rules");
    classes_.resize(rules);
    for(std::size_t r = 0; r < rules; ++r) {
        bool single = true;
        for(const auto i:rules_[r]) single = single && productions_[i].rhs.size() == 1 && productions_[i].rhs[0] < 256;
        if(!single) continue;
        classes_[r].any = true;
        for(const auto i:rules_[r]) classes_[r].bits[productions_[i].rhs[0] / 64] |= std::uint64_t{1} << (productions_[i].rhs[0] % 64);
    }
}
std::shared_ptr<const ByteGrammar> ByteGrammar::load(std::string_view artifact) {
    if(artifact.size() > artifact_limit) throw std::length_error("grammar artifact exceeds 4 MiB");
    Fields fields(artifact);
    std::string_view version;
    std::size_t rules = 0, count = 0, start = 0;
    if(!fields.word(version) || version != "crit-byte-grammar-v1" || !fields.number(rules) || !fields.number(count) ||
       !fields.number(start) || !rules || rules > rule_limit || !count || count > production_limit || start >= rules)
        throw std::invalid_argument("invalid grammar artifact header");
    std::vector<Production> productions;
    productions.reserve(count);
    std::size_t symbols = 0;
    for(std::size_t i = 0; i < count; ++i) {
        std::size_t lhs = 0, size = 0;
        if(!fields.number(lhs) || !fields.number(size) || lhs >= rules || size > symbol_limit - symbols)
            throw std::invalid_argument("invalid grammar artifact production");
        Production p{lhs, std::vector<std::size_t>(size)};
        symbols += size;
        for(auto& symbol:p.rhs)
            if(!fields.number(symbol) || symbol >= 256 + rules) throw std::invalid_argument("invalid grammar artifact symbol");
        productions.push_back(std::move(p));
    }
    if(!fields.done()) throw std::invalid_argument("trailing grammar artifact data");
    return std::make_shared<const ByteGrammar>(rules, start, std::move(productions));
}

GrammarMatcher::GrammarMatcher(std::shared_ptr<const ByteGrammar> grammar, std::shared_ptr<const ByteVocabulary> vocabulary,
                               GrammarLimits limits, std::shared_ptr<AllowedCache> cache)
    : grammar_(std::move(grammar)), vocabulary_(std::move(vocabulary)), limits_(limits), cache_(std::move(cache)) {
    if(!grammar_ || !vocabulary_ || !limits.bytes || limits.bytes > 1048576 || !limits.tokens || limits.tokens > 1048576 ||
       !limits.states_per_byte || limits.states_per_byte > 65536 || !limits.total_states || limits.total_states > 4194304)
        throw std::invalid_argument("invalid grammar matcher grammar, vocabulary or limits");
    if(!cache_) cache_ = std::make_shared<AllowedCache>(allowed_cache_private_bytes);
    std::string key;
    for(const std::size_t v:{limits.bytes, limits.tokens, limits.states_per_byte, limits.total_states}) {
        char bytes[sizeof(v)];
        std::memcpy(bytes, &v, sizeof(v));
        key.append(bytes, sizeof(v));
    }
    cache_->bind(vocabulary_.get(), grammar_.get(), key);
    std::vector<Chart> start(1);
    for(const auto p:grammar_->rule(grammar_->start())) start[0].push_back({p, 0, 0});
    std::size_t states = 0;
    close(start, states);
    charts_ = std::move(start);
    states_ = states;
}
std::unique_ptr<TokenMatcher> GrammarMatcher::fork() const { return std::make_unique<GrammarMatcher>(*this); }
std::size_t GrammarMatcher::vocabulary() const noexcept { return vocabulary_->size(); }

// Completes and predicts the last chart of charts_ + extension. Positions below charts_.size() read the matcher's
// own charts, so a trial never copies earlier charts. `states` counts the extension's items.
void GrammarMatcher::close(std::vector<Chart>& extension, std::size_t& states) const {
    const std::size_t base = charts_.size(), position = base + extension.size() - 1;
    const auto chart_at = [&](std::size_t i) -> const Chart& { return i < base ? charts_[i] : extension[i - base]; };
    Chart& chart = extension.back();
    ItemSet& seen = closing;
    seen.reset();
    // Seeds are distinct scans of distinct items: they are the chart's first items, deduplicated in place.
    std::size_t kept = 0;
    for(std::size_t i = 0; i < chart.size(); ++i) {
        if(!seen.insert(chart[i])) continue;
        if(kept == limits_.states_per_byte || states_ + states == limits_.total_states)
            throw std::length_error("grammar chart budget exhausted");
        chart[kept++] = chart[i];
        ++states;
    }
    chart.resize(kept);
    const auto add = [&](Item item) {
        if(!seen.insert(item)) return;
        if(chart.size() == limits_.states_per_byte || states_ + states == limits_.total_states)
            throw std::length_error("grammar chart budget exhausted");
        chart.push_back(item);
        ++states;
    };
    for(std::size_t at = 0; at < chart.size(); ++at) {
        const Item item = chart[at];
        const auto& p = grammar_->production(item.production);
        if(item.dot == p.rhs.size()) {
            // Index the origin chart each time: when it is this chart, add() may reallocate it. Parents predicted
            // after an empty completion are advanced by the nullable rule below.
            for(std::size_t j = 0; j < chart_at(item.origin).size(); ++j) {
                Item parent = chart_at(item.origin)[j];
                const auto& rule = grammar_->production(parent.production);
                if(parent.dot < rule.rhs.size() && rule.rhs[parent.dot] == 256 + p.lhs) { ++parent.dot; add(parent); }
            }
        } else if(p.rhs[item.dot] >= 256) {
            const std::size_t symbol = p.rhs[item.dot] - 256;
            if(grammar_->byte_class(symbol)) continue;   // scanned as one terminal by step()
            for(const auto index:grammar_->rule(symbol)) add({index, 0, position});
            if(grammar_->nullable(symbol)) add({item.production, item.dot + 1, item.origin});
        }
    }
}
bool GrammarMatcher::step(unsigned char byte, std::vector<Chart>& extension, std::size_t& states) const {
    const Chart& last = extension.empty() ? charts_.back() : extension.back();
    Chart scanned;
    for(const auto& item:last) {
        const auto& p = grammar_->production(item.production);
        if(item.dot == p.rhs.size()) continue;
        const auto symbol = p.rhs[item.dot];
        if(symbol == byte || (symbol >= 256 && grammar_->byte_class(symbol - 256) && grammar_->class_contains(symbol - 256, byte)))
            scanned.push_back({item.production, item.dot + 1, item.origin});
    }
    if(scanned.empty()) return false;
    extension.push_back(std::move(scanned));
    close(extension, states);
    return true;
}
bool GrammarMatcher::scan(std::string_view piece, std::vector<Chart>& extension) const {
    std::size_t states = 0;
    for(const char raw:piece) if(!step(static_cast<unsigned char>(raw), extension, states)) return false;
    return true;
}
// What a continuation can read: whether a whole sentence ended here (EOS), the last chart's items still waiting for
// a symbol, and, through their origins, the waiting items of every chart they reach (a later completion advances
// only waiting parents). Completed items have acted already and are left out, so a long repetition does not chain
// earlier charts in. Charts are renumbered by discovery and items sorted, so equal structures at other offsets give
// equal keys; equal keys have equal futures.
std::string GrammarMatcher::state_key() const {
    constexpr std::size_t item_limit = 4096;
    std::vector<std::size_t> order{charts_.size() - 1};
    std::unordered_map<std::size_t, std::uint32_t> rank{{charts_.size() - 1, 0}};
    std::vector<Item> waiting;
    std::string key(1, complete() ? char(1) : char(0));
    std::size_t items = 0;
    const auto put = [&](std::uint32_t v) { char b[4]; std::memcpy(b, &v, 4); key.append(b, 4); };
    for(std::size_t k = 0; k < order.size(); ++k) {
        waiting.clear();
        for(const auto& item:charts_[order[k]])
            if(item.dot < grammar_->production(item.production).rhs.size()) waiting.push_back(item);
        items += waiting.size();
        if(items > item_limit) return {};
        std::sort(waiting.begin(), waiting.end(), [](const Item& a, const Item& b) {
            return a.production != b.production ? a.production < b.production : a.dot != b.dot ? a.dot < b.dot : a.origin > b.origin;
        });
        put(std::uint32_t(waiting.size()));
        for(const auto& item:waiting) {
            auto found = rank.find(item.origin);
            if(found == rank.end()) { found = rank.emplace(item.origin, std::uint32_t(order.size())).first; order.push_back(item.origin); }
            put(std::uint32_t(item.production)); put(std::uint32_t(item.dot)); put(found->second);
        }
    }
    return key;
}
bool GrammarMatcher::complete() const noexcept {
    for(const auto& item:charts_.back()) {
        const auto& p = grammar_->production(item.production);
        if(!item.origin && p.lhs == grammar_->start() && item.dot == p.rhs.size()) return true;
    }
    return false;
}
std::vector<Token> GrammarMatcher::allowed() const {
    if(ended()) return {vocabulary_->eos()};
    if(size() >= limits_.tokens) throw std::domain_error("no token can continue the grammar within its limits");
    const std::size_t room = std::min(limits_.bytes - marks_.back().bytes, vocabulary_->longest_piece());
    std::string key = state_key();
    if(!key.empty()) {
        char bytes[sizeof(room)];
        std::memcpy(bytes, &room, sizeof(room));
        key.append(bytes, sizeof(room));
        if(const auto hit = cache_->find(key, limits_.total_states - states_)) return *hit;
    }
    // One Earley step per trie edge: a piece prefix shared by many tokens is scanned once. The extension holds the
    // charts of the current path; `added[d]` counts its items to depth d, the budget close() checks.
    std::vector<std::uint64_t> bits((vocabulary_->size() + 63) / 64);
    std::vector<Chart> extension;
    std::vector<std::size_t> added(room + 1);
    std::size_t needed = 0;
    vocabulary_->walk(room, [&](std::size_t depth, unsigned char byte) {
        extension.resize(depth);
        std::size_t states = added[depth];
        if(!step(byte, extension, states)) return false;
        added[depth + 1] = states;
        needed = std::max(needed, states);
        return true;
    }, [&](Token t) { bits[std::size_t(t) / 64] |= std::uint64_t{1} << (std::size_t(t) % 64); });
    if(complete()) {
        const auto eos = std::size_t(vocabulary_->eos());
        bits[eos / 64] |= std::uint64_t{1} << (eos % 64);
    }
    std::vector<Token> ids;
    for(std::size_t i = 0; i < bits.size(); ++i)
        for(auto w = bits[i]; w; w &= w - 1) {
            int bit = 0;
            while(!((w >> bit) & 1u)) ++bit;
            ids.push_back(Token(i * 64 + std::size_t(bit)));
        }
    if(ids.empty()) throw std::domain_error("no token can continue the grammar within its limits");
    if(!key.empty()) cache_->insert(key, std::make_shared<const std::vector<Token>>(ids), needed);
    return ids;
}
void GrammarMatcher::append(Token token) {
    const auto piece = vocabulary_->piece(token);
    if(ended() || size() == limits_.tokens) throw std::domain_error("token violates the grammar or its limits");
    if(token == vocabulary_->eos()) {
        if(!complete()) throw std::domain_error("token violates the grammar or its limits");
        marks_.push_back({marks_.back().bytes, true});
        return;
    }
    std::vector<Chart> extension;
    if(piece.empty() || piece.size() > limits_.bytes - marks_.back().bytes || !scan(piece, extension))
        throw std::domain_error("token violates the grammar or its limits");
    std::size_t added = 0;
    for(const auto& chart:extension) added += chart.size();
    // Reserve first so publishing cannot fail halfway.
    marks_.reserve(marks_.size() + 1);
    charts_.reserve(charts_.size() + extension.size());
    marks_.push_back({marks_.back().bytes + piece.size(), false});
    for(auto& chart:extension) charts_.push_back(std::move(chart));
    states_ += added;
}
void GrammarMatcher::rewind(std::size_t output_tokens) {
    if(output_tokens > size()) throw std::out_of_range("constraint rewind cannot extend output");
    marks_.resize(output_tokens + 1);
    charts_.resize(marks_.back().bytes + 1);
    states_ = 0;
    for(const auto& chart:charts_) states_ += chart.size();
}
}
