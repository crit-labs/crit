#include "crit/sampling.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace crit::core {
namespace {
constexpr double negative_infinity = -std::numeric_limits<double>::infinity();
constexpr float float_infinity = std::numeric_limits<float>::infinity();
// Token ids are int32: a row may hold at most 2^31 entries.
constexpr std::size_t token_id_limit = std::size_t(std::numeric_limits<Token>::max()) + 1;
constexpr std::uint64_t history_limit = std::uint64_t(std::numeric_limits<Token>::max());

struct RowScan { double max = negative_infinity; std::size_t argmax = 0; };
// Validates a raw row and returns its maximum and first argmax.
RowScan scan_row(std::span<const float> row) {
    RowScan scan;
    for(std::size_t i = 0; i < row.size(); ++i) {
        const float x = row[i];
        if(std::isnan(x) || x == float_infinity) throw std::invalid_argument("logits must be finite or -inf");
        if(x > scan.max) { scan.max = x; scan.argmax = i; }
    }
    if(scan.max == negative_infinity) throw std::invalid_argument("no selectable token");
    return scan;
}
// log(sum of exp(x - max)) over a scanned row's unmasked entries, accumulated in double in id order.
double row_log_sum(std::span<const float> row, double max) {
    double sum = 0;
    for(const float x:row) if(x != -float_infinity) sum += std::exp(double(x) - max);
    return std::log(sum);
}
// A raw log-probability: (x - max) - log_sum. Folding max + log_sum first would absorb log_sum into a
// large-magnitude max (|max| around 1e10 and beyond) and break the normalization.
float raw_logprob(float x, double max, double log_sum) { return static_cast<float>((double(x) - max) - log_sum); }
struct RowStats { double max = negative_infinity, log_sum = 0; std::size_t argmax = 0; };
RowStats row_stats(std::span<const float> row) {
    const RowScan scan = scan_row(row);
    return {scan.max, row_log_sum(row, scan.max), scan.argmax};
}
bool ranks_before(float a, Token i, float b, Token j) { return a > b || (a == b && i < j); }
// The out.size() best entries of `row` by (float log-probability desc, id asc): ranking the rounded values keeps
// the TokenLogprob order exact when nearby logits round to one log-probability. Uses `out` as a heap whose front
// is the worst kept entry, so it allocates nothing: O(V log n) worst case, O(V) typical.
void select_top(std::span<const float> row, std::span<TokenLogprob> out, double max, double log_sum) {
    if(out.empty()) return;
    const auto worse_last = [](const TokenLogprob& a, const TokenLogprob& b) {
        return ranks_before(a.logprob, a.token, b.logprob, b.token);
    };
    const auto logprob = [&](std::size_t i) { return raw_logprob(row[i], max, log_sum); };
    for(std::size_t i = 0; i < out.size(); ++i) out[i] = {Token(i), logprob(i)};
    std::make_heap(out.begin(), out.end(), worse_last);
    for(std::size_t i = out.size(); i < row.size(); ++i) {
        const float value = logprob(i);
        if(!ranks_before(value, Token(i), out.front().logprob, out.front().token)) continue;
        std::pop_heap(out.begin(), out.end(), worse_last);
        out.back() = {Token(i), value};
        std::push_heap(out.begin(), out.end(), worse_last);
    }
    std::sort_heap(out.begin(), out.end(), worse_last);
}
void check_row_ids(std::size_t size) {
    if(size > token_id_limit) throw std::invalid_argument("logits row exceeds the token id range");
}
// Shared by validate() and Sampler, which passes preallocated scratch for the uniqueness check.
void validate_into(const SamplingParams& p, std::uint32_t vocabulary, std::vector<Token>& ids) {
    if(vocabulary == 0) throw std::invalid_argument("sampling vocabulary must be positive");
    if(!std::isfinite(p.temperature) || p.temperature < 0)
        throw std::invalid_argument("temperature must be finite and nonnegative");
    if(!std::isfinite(p.top_p) || p.top_p <= 0 || p.top_p > 1)
        throw std::invalid_argument("top_p must be finite and in (0, 1]");
    if(!std::isfinite(p.min_p) || p.min_p < 0 || p.min_p > 1)
        throw std::invalid_argument("min_p must be finite and in [0, 1]");
    if(!std::isfinite(p.repetition_penalty) || p.repetition_penalty <= 0)
        throw std::invalid_argument("repetition_penalty must be finite and positive");
    if(!std::isfinite(p.frequency_penalty) || p.frequency_penalty < -2 || p.frequency_penalty > 2)
        throw std::invalid_argument("frequency_penalty must be finite and in [-2, 2]");
    if(!std::isfinite(p.presence_penalty) || p.presence_penalty < -2 || p.presence_penalty > 2)
        throw std::invalid_argument("presence_penalty must be finite and in [-2, 2]");
    if(p.logit_bias.size() > logit_bias_limit) throw std::invalid_argument("logit_bias exceeds 1024 entries");
    for(const auto& entry:p.logit_bias) {
        if(entry.token < 0 || std::uint32_t(entry.token) >= vocabulary)
            throw std::invalid_argument("logit_bias token outside vocabulary");
        if(std::isnan(entry.bias) || entry.bias == float_infinity)
            throw std::invalid_argument("logit_bias must be finite or negative infinity");
    }
    if(p.logit_bias.size() > 1) {
        ids.clear();
        ids.reserve(p.logit_bias.size());
        for(const auto& entry:p.logit_bias) ids.push_back(entry.token);
        std::sort(ids.begin(), ids.end());
        if(std::adjacent_find(ids.begin(), ids.end()) != ids.end())
            throw std::invalid_argument("logit_bias tokens must be unique");
    }
    // The list applies only while min_tokens > 0. Ids at or past the row are ignored: like padding stop tokens,
    // they can never be drawn.
    if(p.min_tokens) for(const Token token:p.suppress_until_min)
        if(token < 0) throw std::invalid_argument("suppress_until_min tokens must be nonnegative");
}
std::uint32_t mix_token(Token token) noexcept {
    auto x = std::uint32_t(token);
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
// The slot holding `token`, or the empty slot where it belongs. Triangular probing over a power-of-two table at
// most half full visits every slot, so it terminates.
template<class Slot, class Entries> Slot* probe(std::span<Slot> table, const Entries& entries, Token token) noexcept {
    const std::size_t mask = table.size() - 1;
    std::size_t at = mix_token(token) & mask;
    for(std::size_t step = 1;; ++step) {
        Slot& value = table[at];
        if(!value || entries[value - 1].token == token) return &value;
        at = (at + step) & mask;
    }
}
void check_history_tokens(std::span<const Token> tokens) {
    for(const Token token:tokens) if(token < 0) throw std::invalid_argument("token history tokens must be nonnegative");
}
// Whether two float ranges share an element; empty ranges never do.
bool overlaps(std::span<const float> a, std::span<const float> b) noexcept {
    const std::less<const float*> less;
    return !a.empty() && !b.empty() && less(a.data(), b.data() + b.size()) && less(b.data(), a.data() + a.size());
}
}

bool suppression_yields(AllowedTokens allowed, const SamplingParams& p) noexcept {
    if(allowed.ids.empty()) return false;
    for(const Token id:allowed.ids)
        if(std::find(p.suppress_until_min.begin(), p.suppress_until_min.end(), id) == p.suppress_until_min.end()) return false;
    return true;
}
SamplingParams resolve_sampling(const SamplingOptions& options, const SamplingParams& defaults, std::uint32_t vocabulary) {
    const auto& bias = options.logit_bias.empty() ? defaults.logit_bias : options.logit_bias;
    if(bias.size() > logit_bias_limit) throw std::invalid_argument("logit_bias exceeds 1024 entries");
    SamplingParams p;
    p.temperature = options.temperature.value_or(defaults.temperature);
    p.top_p = options.top_p.value_or(defaults.top_p);
    p.min_p = options.min_p.value_or(defaults.min_p);
    p.top_k = options.top_k.value_or(defaults.top_k);
    p.repetition_penalty = options.repetition_penalty.value_or(defaults.repetition_penalty);
    p.frequency_penalty = options.frequency_penalty.value_or(defaults.frequency_penalty);
    p.presence_penalty = options.presence_penalty.value_or(defaults.presence_penalty);
    p.min_tokens = options.min_tokens.value_or(defaults.min_tokens);
    p.seed = options.seed.value_or(defaults.seed);
    p.stream = defaults.stream;
    p.logit_bias = bias;
    p.suppress_until_min = defaults.suppress_until_min;
    validate(p, vocabulary);
    return p;
}
void validate(const SamplingParams& params, std::uint32_t vocabulary) {
    std::vector<Token> ids;
    validate_into(params, vocabulary, ids);
}

TokenHistory::TokenHistory(std::span<const Token> prompt, bool count_prompt_for_repetition) {
    if(prompt.size() > history_limit) throw std::runtime_error("token history prompt exceeds 2^31 - 1 tokens");
    check_history_tokens(prompt);
    if(!count_prompt_for_repetition) return;
    // Room grows with the distinct tokens (doubling), so a long repetitive prompt stays small.
    for(const Token token:prompt) {
        if(!table_.empty() && *probe(std::span(table_), entries_, token)) continue;
        reserve(entries_.size() + 1);
        auto* at = probe(std::span(table_), entries_, token);
        entries_.push_back({token, 0, true});
        *at = std::uint32_t(entries_.size());
    }
}
// A vector copy keeps only size(): restore the source's capacities so the copy keeps the room stage() reserved.
TokenHistory::TokenHistory(const TokenHistory& other) : table_(other.table_), outputs_(other.outputs_) {
    entries_.reserve(other.entries_.capacity());
    entries_.assign(other.entries_.begin(), other.entries_.end());
    staged_.reserve(other.staged_.capacity());
    staged_.assign(other.staged_.begin(), other.staged_.end());
}
TokenHistory& TokenHistory::operator=(const TokenHistory& other) {
    if(this != &other) *this = TokenHistory(other);
    return *this;
}
TokenHistory::TokenHistory(TokenHistory&& other) noexcept
    : entries_(std::move(other.entries_)), table_(std::move(other.table_)), staged_(std::move(other.staged_)),
      outputs_(std::exchange(other.outputs_, 0)) {}
TokenHistory& TokenHistory::operator=(TokenHistory&& other) noexcept {
    TokenHistory taken(std::move(other));                  // leaves `other` empty, also on self-move
    entries_.swap(taken.entries_);
    table_.swap(taken.table_);
    staged_.swap(taken.staged_);
    std::swap(outputs_, taken.outputs_);
    return *this;
}
// Grows entries_ to a power-of-two capacity and the table to twice that, so the table stays at most half full.
// vector::reserve and the table swap each change nothing when they throw.
void TokenHistory::reserve(std::size_t entries) {
    if(!entries) return;
    if(entries > entries_.capacity()) entries_.reserve(std::bit_ceil(std::max<std::size_t>(entries, 16)));
    const std::size_t slots = 2*std::bit_ceil(std::max<std::size_t>(entries, 8));
    if(slots <= table_.size()) return;
    std::vector<std::uint32_t> table(slots, 0);
    for(std::size_t i = 0; i < entries_.size(); ++i) *probe(std::span(table), entries_, entries_[i].token) = std::uint32_t(i + 1);
    table_.swap(table);
}
void TokenHistory::stage(std::span<const Token> provisional) {
    if(!staged_.empty()) throw std::logic_error("token history already has staged tokens");
    check_history_tokens(provisional);
    if(provisional.size() > history_limit - outputs_) throw std::overflow_error("token history output count overflow");
    if(provisional.size() > staged_.capacity()) staged_.reserve(std::bit_ceil(std::max<std::size_t>(provisional.size(), 16)));
    reserve(entries_.size() + provisional.size());
    staged_.assign(provisional.begin(), provisional.end());
}
void TokenHistory::commit(std::size_t accepted) {
    if(accepted > staged_.size()) throw std::out_of_range("commit exceeds staged tokens");
    // stage() reserved room for every staged token and copies keep it, so this reserve is a no-op; were the room
    // ever missing, it would throw here before any count changes. Nothing below allocates or throws.
    reserve(entries_.size() + accepted);
    for(std::size_t i = 0; i < accepted; ++i) {
        auto* at = probe(std::span(table_), entries_, staged_[i]);
        if(*at) { ++entries_[*at - 1].outputs; continue; }
        entries_.push_back({staged_[i], 1, false});
        *at = std::uint32_t(entries_.size());
    }
    outputs_ += accepted;
    staged_.clear();
}
void TokenHistory::rollback() noexcept { staged_.clear(); }
std::uint64_t TokenHistory::outputs() const noexcept { return outputs_; }
std::size_t TokenHistory::staged() const noexcept { return staged_.size(); }
std::span<const Token> TokenHistory::staged_tokens() const noexcept { return staged_; }
std::uint32_t TokenHistory::output_count(Token token) const noexcept {
    if(token < 0 || table_.empty()) return 0;
    const auto* at = probe(std::span(table_), entries_, token);
    return *at ? entries_[*at - 1].outputs : 0;
}
std::size_t TokenHistory::distinct() const noexcept { return entries_.size(); }
std::size_t TokenHistory::host_bytes(std::size_t prompt, std::size_t outputs) noexcept {
    constexpr std::size_t cap = std::numeric_limits<std::size_t>::max()/256;
    if(prompt > cap || outputs > cap - prompt) return std::numeric_limits<std::size_t>::max();
    // Capacities grow to powers of two; a reallocation briefly holds the old half as well (factor 3/2).
    const std::size_t entries = std::bit_ceil(std::max<std::size_t>(prompt + outputs, 16));
    const std::size_t slots = 2*std::bit_ceil(std::max<std::size_t>(prompt + outputs, 8));
    const std::size_t staged = std::bit_ceil(std::max<std::size_t>(outputs, 16));
    return 3*(entries*sizeof(Entry) + slots*sizeof(std::uint32_t) + staged*sizeof(Token))/2;
}

// Kept set after the filters: candidates_[0, kept), in rank order up to `sorted`, their total weight, and the
// maximum processed logit and temperature behind the weights. Greedy laws carry only the token. Every law carries
// the raw row's maximum and log-sum, the normalization of reported log-probabilities.
struct Sampler::Law {
    bool greedy = false;
    Token token = 0;
    std::size_t kept = 0, sorted = 0;
    double total = 0, max = 0, temperature = 1;
    double raw_max = 0, raw_log_sum = 0;
};
namespace {
// Log-weight histogram resolution: bin = floor((max - value) / temperature * 16), capped at the last bin. Bins are
// monotone in rank, so the tokens in the first b bins are exactly a rank prefix.
constexpr std::size_t histogram_bins = 2048;
constexpr double bins_per_nat = 16;
bool candidate_before(double a, std::uint32_t i, double b, std::uint32_t j) { return a > b || (a == b && i < j); }
}
Sampler::Sampler(std::uint32_t vocabulary) : vocabulary_(vocabulary) {
    if(vocabulary == 0) throw std::invalid_argument("sampler vocabulary must be positive");
    if(vocabulary > token_id_limit) throw std::invalid_argument("sampler vocabulary exceeds the token id range");
    candidates_.resize(vocabulary);
    counts_.assign(vocabulary, 0);
    bins_.resize(histogram_bins);
    bias_ids_.reserve(logit_bias_limit);
}
Sampler::Sampler(Sampler&& other) noexcept
    : vocabulary_(std::exchange(other.vocabulary_, 0)), candidates_(std::move(other.candidates_)),
      counts_(std::move(other.counts_)), bins_(std::move(other.bins_)), bias_ids_(std::move(other.bias_ids_)) {}
Sampler& Sampler::operator=(Sampler&& other) noexcept {
    if(this == &other) return *this;
    vocabulary_ = std::exchange(other.vocabulary_, 0);
    candidates_ = std::move(other.candidates_);
    counts_ = std::move(other.counts_);
    bins_ = std::move(other.bins_);
    bias_ids_ = std::move(other.bias_ids_);
    other.candidates_.clear(); other.counts_.clear(); other.bins_.clear(); other.bias_ids_.clear();
    return *this;
}
std::uint32_t Sampler::vocabulary() const noexcept { return vocabulary_; }
std::size_t Sampler::scratch_bytes() const noexcept {
    return candidates_.capacity()*sizeof(Candidate) + counts_.capacity()*sizeof(std::uint32_t) +
           bins_.capacity()*sizeof(Bin) + bias_ids_.capacity()*sizeof(Token);
}
std::size_t Sampler::scratch_bytes(std::uint32_t vocabulary) noexcept {
    return std::size_t(vocabulary)*(sizeof(Candidate) + sizeof(std::uint32_t)) + histogram_bins*sizeof(Bin) +
           logit_bias_limit*sizeof(Token);
}
// Applies repetition then frequency/presence penalties to the tokens the history tracks plus staged[0, column).
// The repetition result is clamped to the finite double range, so only -inf (raw, banned, suppressed) masks a
// token; the frequency/presence term (at most about 4.3e9) cannot leave that range. counts_ is all zero on entry
// and on exit; nothing here throws.
void Sampler::penalize(const SamplingParams& p, const TokenHistory& history, std::size_t column) noexcept {
    constexpr double highest = std::numeric_limits<double>::max();
    const auto staged = history.staged_tokens().first(column);
    const auto apply = [&](std::uint32_t id, bool seen, std::uint64_t outputs) {
        double& value = candidates_[id].value;
        if(value == negative_infinity) return;
        if(seen && p.repetition_penalty != 1.0)
            value = value > 0 ? std::min(value/p.repetition_penalty, highest) : std::max(value*p.repetition_penalty, -highest);
        if(outputs) value -= p.frequency_penalty*double(outputs) + p.presence_penalty;
    };
    for(const Token token:staged) if(std::uint32_t(token) < vocabulary_) ++counts_[std::uint32_t(token)];
    for(const auto& entry:history.entries_) {
        const auto id = std::uint32_t(entry.token);
        if(id >= vocabulary_) continue;
        const std::uint64_t outputs = std::uint64_t(entry.outputs) + counts_[id];
        counts_[id] = 0;
        apply(id, entry.prompt || outputs, outputs);
    }
    for(const Token token:staged) {
        const auto id = std::uint32_t(token);
        if(id >= vocabulary_ || !counts_[id]) continue;
        const std::uint64_t outputs = counts_[id];
        counts_[id] = 0;
        apply(id, true, outputs);
    }
}
// Ensures candidates_[0, need) are the `need` best of [0, count) in rank order, given that [0, sorted) already
// are: one partial selection of the next block, then a sort of that block. Blocks grow geometrically.
void Sampler::sort_prefix(std::size_t need, std::size_t count, std::size_t& sorted) {
    need = std::min(need, count);
    if(need <= sorted) return;
    const auto before = [](const Candidate& a, const Candidate& b) { return candidate_before(a.value, a.id, b.value, b.id); };
    const auto first = candidates_.begin();
    std::size_t next = std::max({need, sorted*4, std::size_t(64)});
    if(next >= count/2) next = count;
    if(next < count)
        std::nth_element(first + std::ptrdiff_t(sorted), first + std::ptrdiff_t(next), first + std::ptrdiff_t(count), before);
    std::sort(first + std::ptrdiff_t(sorted), first + std::ptrdiff_t(next), before);
    sorted = next;
}
// Before a rank-order walk to cumulative weight `target` over the kept set [0, count): sorts the shortest
// histogram rank prefix whose weight reaches `target` (with a margin for summation order) using one partition by
// bin, so the walk stays inside it. The walk's own sort_prefix calls still cover any shortfall.
void Sampler::sort_to_target(double target, std::size_t count, std::size_t& sorted) {
    const double goal = target*(1 + 1e-9);
    double accumulated = 0;
    std::size_t tokens = 0, last = histogram_bins - 1;
    for(std::size_t b = 0; b < histogram_bins; ++b) {
        accumulated += bins_[b].weight;
        tokens += bins_[b].count;
        if(tokens && accumulated >= goal) { last = b; break; }
    }
    if(tokens <= sorted) return;
    const auto first = candidates_.begin() + std::ptrdiff_t(sorted);
    const auto end = std::partition(first, candidates_.begin() + std::ptrdiff_t(count),
                                    [&](const Candidate& c) { return c.bin <= last; });
    std::sort(first, end, [](const Candidate& a, const Candidate& b) { return candidate_before(a.value, a.id, b.value, b.id); });
    sorted = std::size_t(end - candidates_.begin());
}
Sampler::Law Sampler::prepare(std::span<const float> logits, const SamplingParams& p, const TokenHistory& history,
                              std::size_t column, const AllowedTokens* allowed) {
    if(logits.size() != vocabulary_) throw std::invalid_argument("logits size must equal the sampler vocabulary");
    validate_into(p, vocabulary_, bias_ids_);
    if(column > history.staged()) throw std::out_of_range("column exceeds staged tokens");
    if(allowed) {
        Token previous = -1;
        for(const Token id:allowed->ids) {
            if(id <= previous || std::uint32_t(id) >= vocabulary_)
                throw std::invalid_argument("allowed tokens must be nonempty, sorted, unique and in the vocabulary");
            previous = id;
        }
        if(allowed->ids.empty())
            throw std::invalid_argument("allowed tokens must be nonempty, sorted, unique and in the vocabulary");
    }
    const RowScan raw = scan_row(logits);
    const bool suppress = !p.suppress_until_min.empty() && history.outputs() + column < p.min_tokens &&
                          !(allowed && suppression_yields(*allowed, p));
    const bool penalties = (p.repetition_penalty != 1.0 || p.frequency_penalty != 0.0 || p.presence_penalty != 0.0) &&
                           (!history.entries_.empty() || column);
    const bool modified = !p.logit_bias.empty() || suppress || penalties || allowed;
    Law law;
    law.raw_max = raw.max;
    if(p.temperature == 0 && !modified) {
        law.raw_log_sum = row_log_sum(logits, raw.max);
        law.greedy = true;
        law.token = Token(raw.argmax);
        return law;
    }
    // Selectable tokens compacted in id order; `best` is the first maximum.
    std::size_t count = 0, best = 0;
    const auto push = [&](const Candidate& c) {
        if(!count || c.value > candidates_[best].value) best = count;
        candidates_[count++] = c;
    };
    if(modified) {
        for(std::size_t i = 0; i < logits.size(); ++i) candidates_[i] = {double(logits[i]), 0, std::uint32_t(i)};
        if(allowed) {
            // Mask first: ids between consecutive allowed ids (and after the last) leave the selectable set.
            std::size_t next = 0;
            for(const Token id:allowed->ids) {
                for(; next < std::size_t(id); ++next) candidates_[next].value = negative_infinity;
                next = std::size_t(id) + 1;
            }
            for(; next < logits.size(); ++next) candidates_[next].value = negative_infinity;
        }
        for(const auto& entry:p.logit_bias) candidates_[std::size_t(entry.token)].value += double(entry.bias);
        if(suppress) for(const Token token:p.suppress_until_min)
            if(std::uint32_t(token) < vocabulary_) candidates_[std::size_t(token)].value = negative_infinity;
        if(penalties) penalize(p, history, column);
        for(std::size_t i = 0; i < logits.size(); ++i) if(candidates_[i].value != negative_infinity) push(candidates_[i]);
    } else {
        for(std::size_t i = 0; i < logits.size(); ++i)
            if(logits[i] != -float_infinity) push({double(logits[i]), 0, std::uint32_t(i)});
    }
    if(!count) throw std::invalid_argument("no selectable token");
    if(p.temperature == 0) {
        law.raw_log_sum = row_log_sum(logits, raw.max);
        law.greedy = true;
        law.token = Token(candidates_[best].id);
        return law;
    }
    law.max = candidates_[best].value;
    law.temperature = p.temperature;
    const auto before = [](const Candidate& a, const Candidate& b) { return candidate_before(a.value, a.id, b.value, b.id); };
    const auto first = candidates_.begin();
    std::size_t survivors = count, sorted = 0;
    if(p.top_k && p.top_k < count) {
        // Exact top_k: a bounded heap (front = worst kept) for small k, otherwise partial selection; then a sort.
        const auto k = std::ptrdiff_t(p.top_k);
        if(p.top_k <= 1024) {
            std::make_heap(first, first + k, before);
            for(std::size_t i = p.top_k; i < count; ++i) {
                if(!before(candidates_[i], candidates_.front())) continue;
                std::pop_heap(first, first + k, before);
                std::swap(candidates_[std::size_t(k) - 1], candidates_[i]);
                std::push_heap(first, first + k, before);
            }
            std::sort_heap(first, first + k, before);
        } else {
            std::nth_element(first, first + k, first + std::ptrdiff_t(count), before);
            std::sort(first, first + k, before);
        }
        survivors = sorted = p.top_k;
    }
    // Weights exp((value - max) / T); totals run in rank order after top_k, otherwise in id order. min_p keeps
    // p >= min_p * p_max, i.e. weight >= min_p as the top weight is exp(0) = 1; weights that underflow to zero
    // carry no probability. Both drop a rank suffix, so the kept tokens stay a rank prefix.
    double total = 0, kept_total = 0;
    std::size_t kept = 0;
    const auto keep = [&](double weight) { return weight > 0 && weight >= p.min_p; };
    if(sorted) {
        for(std::size_t i = 0; i < survivors; ++i) {
            auto& c = candidates_[i];
            c.weight = std::exp((c.value - law.max)/p.temperature);
            total += c.weight;
            if(kept == i && keep(c.weight)) { ++kept; kept_total += c.weight; }
        }
    } else {
        std::fill(bins_.begin(), bins_.end(), Bin{});
        for(std::size_t i = 0; i < survivors; ++i) {
            Candidate c = candidates_[i];
            c.weight = std::exp((c.value - law.max)/p.temperature);
            total += c.weight;
            if(!keep(c.weight)) continue;
            kept_total += c.weight;
            const double scaled = (law.max - c.value)/p.temperature*bins_per_nat;
            c.bin = scaled < double(histogram_bins - 1) ? std::uint32_t(scaled) : std::uint32_t(histogram_bins - 1);
            bins_[c.bin].weight += c.weight;
            ++bins_[c.bin].count;
            candidates_[kept++] = c;
        }
    }
    // Unmodified rows at T = 1 without top_k summed exactly the raw log-sum terms, in the same order.
    law.raw_log_sum = !modified && !sorted && p.temperature == 1.0 ? std::log(total) : row_log_sum(logits, raw.max);
    if(p.top_p < 1) {
        // The smallest rank prefix whose cumulative weight reaches top_p of the top_k total (at least one token).
        const double target = p.top_p*total;
        if(sorted < kept) sort_to_target(target, kept, sorted);
        double accumulated = 0;
        std::size_t nucleus = kept;
        for(std::size_t r = 0; r < kept; ++r) {
            sort_prefix(r + 1, kept, sorted);
            accumulated += candidates_[r].weight;
            if(accumulated >= target) { nucleus = r + 1; break; }
        }
        kept = nucleus;
        kept_total = accumulated;
    }
    law.kept = kept;
    law.sorted = std::min(sorted, kept);
    law.total = kept_total;
    return law;
}
Choice Sampler::sample(std::span<const float> logits, const SamplingParams& p, const TokenHistory& history,
                       std::size_t column, std::span<TokenLogprob> top) {
    return sample_law(logits, p, history, column, top, nullptr);
}
Choice Sampler::sample_law(std::span<const float> logits, const SamplingParams& p, const TokenHistory& history,
                           std::size_t column, std::span<TokenLogprob> top, const AllowedTokens* allowed) {
    if(top.size() > logits.size()) throw std::invalid_argument("top logprobs exceed the logits row");
    Law law = prepare(logits, p, history, column, allowed);
    Token token = law.token;
    if(!law.greedy) {
        // Walk the CDF in rank order to the first cumulative weight above u * total (the last kept token if
        // rounding leaves the target unreached).
        const double target = uniform(p.seed, history.outputs() + column, p.stream, DrawPurpose::sample)*law.total;
        if(law.sorted < law.kept) sort_to_target(target, law.kept, law.sorted);
        double accumulated = 0;
        std::size_t chosen = law.kept - 1;
        for(std::size_t r = 0; r < law.kept; ++r) {
            sort_prefix(r + 1, law.kept, law.sorted);
            accumulated += candidates_[r].weight;
            if(target < accumulated) { chosen = r; break; }
        }
        token = Token(candidates_[chosen].id);
    }
    select_top(logits, top, law.raw_max, law.raw_log_sum);
    return {token, raw_logprob(logits[std::size_t(token)], law.raw_max, law.raw_log_sum)};
}
Choice Sampler::sample(std::span<const float> logits, AllowedTokens allowed, const SamplingParams& p,
                       const TokenHistory& history, std::size_t column, std::span<TokenLogprob> top) {
    return sample_law(logits, p, history, column, top, &allowed);
}
void Sampler::distribution(std::span<const float> logits, AllowedTokens allowed, const SamplingParams& p,
                           const TokenHistory& history, std::size_t column, std::vector<TokenLogprob>& out) {
    distribution_law(logits, p, history, column, out, &allowed);
}
void Sampler::distribution(std::span<const float> logits, const SamplingParams& p, const TokenHistory& history,
                           std::size_t column, std::vector<TokenLogprob>& out) {
    distribution_law(logits, p, history, column, out, nullptr);
}
void Sampler::distribution_law(std::span<const float> logits, const SamplingParams& p, const TokenHistory& history,
                               std::size_t column, std::vector<TokenLogprob>& out, const AllowedTokens* allowed) {
    Law law = prepare(logits, p, history, column, allowed);
    if(law.greedy) { out.assign(1, {law.token, 0.0f}); return; }
    sort_prefix(law.kept, law.kept, law.sorted);
    out.resize(law.kept);
    const double log_total = std::log(law.total);
    for(std::size_t r = 0; r < law.kept; ++r) {
        const auto& c = candidates_[r];
        out[r] = {Token(c.id), static_cast<float>((c.value - law.max)/law.temperature - log_total)};
    }
}

void log_softmax(std::span<const float> logits, std::span<float> out) {
    if(out.size() != logits.size()) throw std::invalid_argument("log_softmax output size must equal logits");
    if(logits.data() != out.data() && overlaps(logits, out))
        throw std::invalid_argument("log_softmax output overlaps logits");
    const RowStats stats = row_stats(logits);
    for(std::size_t i = 0; i < logits.size(); ++i) out[i] = raw_logprob(logits[i], stats.max, stats.log_sum);
}
void top_logprobs(std::span<const float> logits, std::span<TokenLogprob> out) {
    check_row_ids(logits.size());
    if(out.size() > logits.size()) throw std::invalid_argument("top logprobs exceed the logits row");
    const RowStats stats = row_stats(logits);
    select_top(logits, out, stats.max, stats.log_sum);
}
void query_logprobs(std::span<const float> logits, std::span<const Token> ids, std::span<float> out) {
    if(out.size() != ids.size()) throw std::invalid_argument("query output size must equal ids");
    // Writing out[i] while later ids still read the row would corrupt them, so any overlap is rejected.
    if(overlaps(logits, out)) throw std::invalid_argument("query output overlaps logits");
    for(const Token id:ids) if(id < 0 || std::size_t(id) >= logits.size())
        throw std::invalid_argument("query token outside the logits row");
    const RowStats stats = row_stats(logits);
    for(std::size_t i = 0; i < ids.size(); ++i)
        out[i] = raw_logprob(logits[std::size_t(ids[i])], stats.max, stats.log_sum);
}
QueryScores score_logit_rows(const LogitRows& rows, std::span<const Token> targets, std::span<const Token> queries,
                             std::uint32_t per_target, std::uint32_t top_k, bool argmax) {
    constexpr std::size_t size_max = std::numeric_limits<std::size_t>::max();
    if(rows.valid == 0) throw std::invalid_argument("valid vocabulary must be positive");
    if(rows.valid > token_id_limit) throw std::invalid_argument("valid vocabulary exceeds the token id range");
    if(rows.stride < rows.valid) throw std::invalid_argument("logit row stride is smaller than the valid vocabulary");
    if(targets.size() != rows.rows) throw std::invalid_argument("targets must hold one token per row");
    if(rows.rows) {
        const std::size_t last = rows.rows - 1;
        if(last > (size_max - rows.valid)/rows.stride || rows.logits.size() < last*rows.stride + rows.valid)
            throw std::invalid_argument("logit rows exceed the logits span");
    }
    if((per_target && rows.rows > size_max/per_target) || queries.size() != std::size_t(rows.rows)*per_target)
        throw std::invalid_argument("queries must hold per_target ids per row");
    if(top_k > rows.valid) throw std::invalid_argument("top_k exceeds the valid vocabulary");
    if(top_k && rows.rows > size_max/top_k) throw std::overflow_error("score output size overflow");
    for(const Token id:targets) if(id < 0 || std::uint32_t(id) >= rows.valid)
        throw std::invalid_argument("target token outside the valid vocabulary");
    for(const Token id:queries) if(id < 0 || std::uint32_t(id) >= rows.valid)
        throw std::invalid_argument("query token outside the valid vocabulary");
    for(std::size_t r = 0; r < rows.rows; ++r) scan_row(rows.logits.subspan(r*rows.stride, rows.valid));
    QueryScores out;
    out.target.resize(rows.rows);
    out.queries.resize(queries.size());
    if(argmax) out.argmax.resize(rows.rows);
    out.top.resize(std::size_t(rows.rows)*top_k);
    for(std::size_t r = 0; r < rows.rows; ++r) {
        const auto row = rows.logits.subspan(r*rows.stride, rows.valid);
        const RowStats stats = row_stats(row);
        out.target[r] = raw_logprob(row[std::size_t(targets[r])], stats.max, stats.log_sum);
        for(std::size_t q = 0; q < per_target; ++q) {
            const std::size_t at = r*per_target + q;
            out.queries[at] = raw_logprob(row[std::size_t(queries[at])], stats.max, stats.log_sum);
        }
        if(argmax) out.argmax[r] = Token(stats.argmax);
        select_top(row, std::span(out.top).subspan(r*top_k, top_k), stats.max, stats.log_sum);
    }
    return out;
}
Acceptance accept_greedy(std::span<const Token> drafts, std::span<const Token> predictions,
                         std::uint32_t remaining, std::span<const Token> stops) {
    if(drafts.size() >= std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("too many drafts");
    if(predictions.size() != drafts.size() + 1)
        throw std::invalid_argument("predictions must hold one more token than drafts");
    const auto negative = [](Token token) { return token < 0; };
    if(std::any_of(drafts.begin(), drafts.end(), negative) || std::any_of(predictions.begin(), predictions.end(), negative) ||
       std::any_of(stops.begin(), stops.end(), negative))
        throw std::invalid_argument("token ids must be nonnegative");
    const auto is_stop = [&](Token token) { return std::find(stops.begin(), stops.end(), token) != stops.end(); };
    Acceptance result;
    while(result.matched < drafts.size() && result.emitted < remaining &&
          drafts[result.matched] == predictions[result.matched]) {
        ++result.emitted;
        if(is_stop(drafts[result.matched++])) { result.stopped = true; return result; }
    }
    if(result.emitted < remaining) {
        ++result.emitted;
        result.stopped = is_stop(predictions[result.matched]);
    }
    return result;
}
}
