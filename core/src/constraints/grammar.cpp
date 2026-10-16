#include "crit/constraints.h"
#include <algorithm>
#include <charconv>
#include <stdexcept>
#include <unordered_set>
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
struct ItemHash {
    std::size_t operator()(const GrammarMatcher::Item& i) const noexcept {
        return (i.production*1315423911U + i.dot)*2654435761U + i.origin;
    }
};
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
                               GrammarLimits limits)
    : grammar_(std::move(grammar)), vocabulary_(std::move(vocabulary)), limits_(limits) {
    if(!grammar_ || !vocabulary_ || !limits.bytes || limits.bytes > 1048576 || !limits.tokens || limits.tokens > 1048576 ||
       !limits.states_per_byte || limits.states_per_byte > 65536 || !limits.total_states || limits.total_states > 4194304)
        throw std::invalid_argument("invalid grammar matcher grammar, vocabulary or limits");
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
    std::unordered_set<Item, ItemHash> seen;
    Chart seeds;
    seeds.swap(chart);
    const auto add = [&](Item item) {
        if(seen.contains(item)) return;
        if(chart.size() == limits_.states_per_byte || states_ + states == limits_.total_states)
            throw std::length_error("grammar chart budget exhausted");
        seen.insert(item);
        chart.push_back(item);
        ++states;
    };
    for(const auto& seed:seeds) add(seed);
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
            for(const auto index:grammar_->rule(symbol)) add({index, 0, position});
            if(grammar_->nullable(symbol)) add({item.production, item.dot + 1, item.origin});
        }
    }
}
bool GrammarMatcher::scan(std::string_view piece, std::vector<Chart>& extension) const {
    std::size_t states = 0;
    for(const char raw:piece) {
        const auto byte = static_cast<unsigned char>(raw);
        const Chart& last = extension.empty() ? charts_.back() : extension.back();
        Chart scanned;
        for(const auto& item:last) {
            const auto& p = grammar_->production(item.production);
            if(item.dot < p.rhs.size() && p.rhs[item.dot] == byte) scanned.push_back({item.production, item.dot + 1, item.origin});
        }
        if(scanned.empty()) return false;
        extension.push_back(std::move(scanned));
        close(extension, states);
    }
    return true;
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
    std::vector<Token> ids;
    std::vector<Chart> extension;
    if(size() < limits_.tokens) {
        for(std::size_t i = 0; i < vocabulary_->size(); ++i) {
            const Token token = Token(i);
            const auto piece = vocabulary_->piece(token);
            bool ok = false;
            if(token == vocabulary_->eos()) ok = complete();
            else if(!piece.empty() && piece.size() <= limits_.bytes - marks_.back().bytes) {
                extension.clear();
                ok = scan(piece, extension);
            }
            if(ok) ids.push_back(token);
        }
    }
    if(ids.empty()) throw std::domain_error("no token can continue the grammar within its limits");
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
