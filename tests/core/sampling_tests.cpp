// Host sampling: Philox known answers, the filter law and draw frequencies against independent long-double
// oracles, penalty and bias rules, position-keyed redo, logprob math, row scoring, greedy acceptance, every error
// case, allocation-free hot paths, history copies and a 250k-vocabulary timing check.
#include <crit/sampling.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
using namespace crit::core;

// Counting global allocator: hot paths must not allocate, and TokenHistory::host_bytes must bound what a history
// holds. A 16-byte header keeps each block's size and whether it was tracked (malloc alignment is preserved).
namespace {
std::size_t allocation_count = 0, tracked_live = 0, tracked_peak = 0;
bool tracking = false;
}
// Header offsets go through integers so optimizers cannot pair these blocks with the objects built in them.
void* operator new(std::size_t size) {
    void* block = std::malloc(size + 16);
    if(!block) throw std::bad_alloc();
    auto* header = static_cast<std::size_t*>(block);
    header[0] = size;
    header[1] = tracking;
    ++allocation_count;
    if(tracking) { tracked_live += size; tracked_peak = std::max(tracked_peak, tracked_live); }
    return reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(block) + 16);
}
void operator delete(void* pointer) noexcept {
    if(!pointer) return;
    void* block = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(pointer) - 16);
    const auto* header = static_cast<const std::size_t*>(block);
    if(header[1]) tracked_live -= header[0];
    std::free(block);
}
void operator delete(void* pointer, std::size_t) noexcept { operator delete(pointer); }

namespace {
using Real = long double;
constexpr float minus_inf = -std::numeric_limits<float>::infinity();
constexpr Real real_inf = std::numeric_limits<Real>::infinity();

void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }
template<class E, class F> void rejects(F action, std::string_view message) {
    try { action(); } catch(const E& e) {
        if(std::string_view(e.what()).find(message) == std::string_view::npos)
            throw std::runtime_error("unexpected rejection: " + std::string(e.what()));
        return;
    } catch(const std::exception& e) { throw std::runtime_error("wrong exception type: " + std::string(e.what())); }
    throw std::runtime_error("expected rejection missing: " + std::string(message));
}
bool near(Real a, Real b, Real tolerance) { return a == b || std::fabs(a - b) <= tolerance; }
// A float log-probability against a long-double reference: 1e-6 plus the float rounding of the value.
bool close(float got, Real want) { return near(got, want, 1e-6L + std::fabs(want)*2e-7L); }
// Runs `action` with the allocator tracking its blocks.
template<class F> void tracked(F action) { tracking = true; action(); tracking = false; }
std::span<const Token> one(const Token& token) { return {&token, 1}; }
std::vector<Token> concat(std::vector<Token> a, std::span<const Token> b) { a.insert(a.end(), b.begin(), b.end()); return a; }

// Independent oracle of the documented rules. Processing (bias, suppression, penalties) uses the documented double
// arithmetic, so near-ties rank exactly as specified; the softmax, filters and CDF walk run in long double.
std::vector<Real> oracle_processed(const std::vector<float>& logits, const SamplingParams& p, const std::vector<Token>& prompt,
                                   bool count_prompt, const std::vector<Token>& outputs) {
    std::vector<double> v(logits.begin(), logits.end());
    for(const auto& entry:p.logit_bias) v[std::size_t(entry.token)] += double(entry.bias);
    if(outputs.size() < p.min_tokens) for(const Token token:p.suppress_until_min)
        v[std::size_t(token)] = -std::numeric_limits<double>::infinity();
    std::set<Token> seen(outputs.begin(), outputs.end());
    if(count_prompt) seen.insert(prompt.begin(), prompt.end());
    std::map<Token, std::size_t> counts;
    for(const Token token:outputs) ++counts[token];
    constexpr double highest = std::numeric_limits<double>::max();
    for(const Token token:seen) if(std::size_t(token) < v.size() && p.repetition_penalty != 1) {
        auto& x = v[std::size_t(token)];
        if(x != -std::numeric_limits<double>::infinity())
            x = x > 0 ? std::min(x/p.repetition_penalty, highest) : std::max(x*p.repetition_penalty, -highest);
    }
    for(const auto& [token, n]:counts) if(std::size_t(token) < v.size())
        v[std::size_t(token)] -= p.frequency_penalty*double(n) + p.presence_penalty;
    return {v.begin(), v.end()};
}
std::vector<std::size_t> rank_order(const std::vector<Real>& v) {
    std::vector<std::size_t> order;
    for(std::size_t i = 0; i < v.size(); ++i) if(v[i] != -real_inf) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return v[a] > v[b] || (v[a] == v[b] && a < b); });
    return order;
}
// Probabilities by id; empty when nothing is selectable.
std::vector<Real> oracle_law(const std::vector<Real>& v, const SamplingParams& p) {
    auto order = rank_order(v);
    if(order.empty()) return {};
    std::vector<Real> prob(v.size(), 0);
    if(p.temperature == 0) { prob[order[0]] = 1; return prob; }
    if(p.top_k && p.top_k < order.size()) order.resize(p.top_k);
    std::vector<Real> w(order.size());
    Real total = 0;
    for(std::size_t r = 0; r < order.size(); ++r) {
        w[r] = std::exp((v[order[r]] - v[order[0]])/Real(p.temperature));
        total += w[r];
    }
    Real previous = 0, kept = 0;
    for(std::size_t r = 0; r < order.size(); ++r) {
        const bool nucleus = p.top_p >= 1 || previous < Real(p.top_p)*total;
        previous += w[r];
        if(!nucleus || w[r] < Real(p.min_p)) w[r] = 0;
        kept += w[r];
    }
    for(std::size_t r = 0; r < order.size(); ++r) prob[order[r]] = w[r]/kept;
    return prob;
}
// The token a draw u selects walking the law in rank order; margin = distance from u to the nearest CDF edge.
Token oracle_pick(const std::vector<Real>& v, const std::vector<Real>& prob, double u, Real& margin) {
    Real cumulative = 0;
    Token chosen = -1, last = -1;
    margin = 1;
    for(const auto id:rank_order(v)) {
        if(!(prob[id] > 0)) continue;
        cumulative += prob[id];
        margin = std::min(margin, std::fabs(Real(u) - cumulative));
        last = Token(id);
        if(chosen < 0 && Real(u) < cumulative) chosen = Token(id);
    }
    return chosen < 0 ? last : chosen;
}
Real oracle_lse(const std::vector<float>& row) {
    Real best = -real_inf, sum = 0;
    for(const float x:row) best = std::max(best, Real(x));
    for(const float x:row) if(x != minus_inf) sum += std::exp(Real(x) - best);
    return best + std::log(sum);
}
std::vector<float> random_row(std::mt19937_64& rng, std::size_t size, float spread, bool quarter_grid) {
    std::normal_distribution<float> normal(0.f, spread);
    std::vector<float> row(size);
    for(auto& x:row) x = quarter_grid ? std::round(normal(rng)*4.f)/4.f : normal(rng);
    return row;
}
TokenHistory make_history(const std::vector<Token>& prompt, bool count_prompt, const std::vector<Token>& committed,
                          const std::vector<Token>& staged) {
    TokenHistory history(prompt, count_prompt);
    history.stage(committed);
    history.commit(committed.size());
    history.stage(staged);
    return history;
}
// Compares Sampler::distribution and one draw against the oracle; returns false when nothing is selectable.
bool check_law(Sampler& sampler, const std::vector<float>& logits, const SamplingParams& p, const std::vector<Token>& prompt,
               bool count_prompt, const std::vector<Token>& committed, const std::vector<Token>& staged, std::size_t column) {
    const auto history = make_history(prompt, count_prompt, committed, staged);
    const auto outputs = concat(committed, std::span(staged).first(column));
    const auto v = oracle_processed(logits, p, prompt, count_prompt, outputs);
    const auto prob = oracle_law(v, p);
    std::vector<TokenLogprob> law;
    if(prob.empty()) {
        rejects<std::invalid_argument>([&] { sampler.sample(logits, p, history, column); }, "no selectable token");
        rejects<std::invalid_argument>([&] { sampler.distribution(logits, p, history, column, law); }, "no selectable token");
        return false;
    }
    sampler.distribution(logits, p, history, column, law);
    std::vector<double> got(logits.size(), 0);
    float previous = std::numeric_limits<float>::infinity();
    for(const auto& entry:law) {
        require(entry.token >= 0 && std::size_t(entry.token) < logits.size() && got[std::size_t(entry.token)] == 0,
                "distribution ids are distinct and in range");
        require(entry.logprob <= previous && entry.logprob <= 0, "distribution is in rank order");
        previous = entry.logprob;
        got[std::size_t(entry.token)] = std::exp(double(entry.logprob));
    }
    for(std::size_t i = 0; i < logits.size(); ++i)
        require(near(got[i], prob[i], 2e-6L + 2e-5L*prob[i]), "distribution matches the long-double oracle");
    const double u = uniform(p.seed, committed.size() + column, p.stream);
    Real margin = 0;
    const Token want = oracle_pick(v, prob, u, margin);
    const auto choice = sampler.sample(logits, p, history, column);
    require(choice.token == want || margin < 1e-9L, "draw matches the oracle away from a CDF edge");
    require(prob[std::size_t(choice.token)] > 0, "draw lies in the kept set");
    require(close(choice.logprob, Real(logits[std::size_t(choice.token)]) - oracle_lse(logits)),
            "choice carries the raw log-probability");
    return true;
}

void philox() {
    static_assert(philox4x32(0, 0, 0, 0).word[0] == 0x6627e8d5u && philox4x32(0, 0, 0, 0).word[3] == 0x9b00dbd8u,
                  "philox is usable in constant expressions");
    const auto check = [](PhiloxBlock block, std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t d) {
        require(block.word[0] == a && block.word[1] == b && block.word[2] == c && block.word[3] == d,
                "philox4x32-10 matches the Random123 known answer");
    };
    check(philox4x32(0, 0, 0, 0), 0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u);
    check(philox4x32(~0ull, ~0ull, ~0u, ~0u), 0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu);
    // counter {243f6a88, 85a308d3, 13198a2e, 03707344}, key {a4093822, 299f31d0}
    check(philox4x32(0x299f31d0a4093822ull, 0x85a308d3243f6a88ull, 0x13198a2eu, 0x03707344u),
          0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u);
    // Each counter and key word changes the block.
    const auto base = philox4x32(1, 2, 3, 4);
    for(const auto& other:{philox4x32(2, 2, 3, 4), philox4x32(1ull << 32 | 1, 2, 3, 4), philox4x32(1, 3, 3, 4),
                           philox4x32(1, 1ull << 32 | 2, 3, 4), philox4x32(1, 2, 4, 4), philox4x32(1, 2, 3, 5)})
        require(other.word[0] != base.word[0] || other.word[1] != base.word[1], "every input word reaches the output");
}
void uniform_draws() {
    static_assert(uniform(0, 0) == 0xcc4fd1abc2d38p-53 && uniform(~0ull, ~0ull, ~0u, DrawPurpose(~0u)) == 0x811e4eda83907p-53,
                  "uniform known answers from the philox vectors");
    constexpr std::size_t draws = 200000;
    double sum = 0, low = 1, high = 0;
    for(std::uint64_t position = 0; position < draws; ++position) {
        const double u = uniform(42, position, 7, DrawPurpose::draft);
        const auto block = philox4x32(42, position, 7, std::uint32_t(DrawPurpose::draft));
        const std::uint64_t bits = (std::uint64_t(block.word[0]) << 32 | block.word[1]) >> 11;
        require(u == double(bits)/9007199254740992.0, "uniform takes the top 53 bits of words 0 and 1");
        require(u >= 0 && u < 1, "uniform lies in [0, 1)");
        require(u == uniform(42, position, 7, DrawPurpose::draft), "uniform is deterministic");
        sum += u; low = std::min(low, u); high = std::max(high, u);
    }
    const double mean = sum/draws;
    require(std::fabs(mean - 0.5) < 4/std::sqrt(12.0*draws), "uniform mean is one half");
    require(low < 1e-4 && high > 1 - 1e-4, "uniform covers the interval");
    require(uniform(1, 5) == uniform(1, 5, 0, DrawPurpose::sample), "defaults are stream 0 and the sample purpose");
    std::size_t equal = 0;
    for(std::uint64_t position = 0; position < 1000; ++position) {
        const double u = uniform(9, position);
        equal += u == uniform(9, position, 1) || u == uniform(9, position, 0, DrawPurpose::accept) ||
                 u == uniform(10, position) || u == uniform(9, position + 1);
    }
    require(equal == 0, "seed, position, stream and purpose select independent draws");
}
void validation() {
    const auto bad = [](auto edit, std::string_view message, std::uint32_t vocabulary = 10) {
        SamplingParams p;
        edit(p);
        rejects<std::invalid_argument>([&] { validate(p, vocabulary); }, message);
    };
    const double nan = std::nan(""), inf = std::numeric_limits<double>::infinity();
    validate(SamplingParams{}, 1);
    bad([](SamplingParams&) {}, "sampling vocabulary must be positive", 0);
    for(const double t:{-0.5, nan, inf}) bad([&](SamplingParams& p) { p.temperature = t; }, "temperature must be finite and nonnegative");
    for(const double t:{0.0, -0.1, 1.5, nan, inf}) bad([&](SamplingParams& p) { p.top_p = t; }, "top_p must be finite and in (0, 1]");
    // top_k is unbounded: the runtime bounds it by the declared vocabulary, which can exceed the valid one.
    for(const std::uint32_t k:{11u, std::numeric_limits<std::uint32_t>::max()}) {
        SamplingParams wide_k;
        wide_k.top_k = k;
        validate(wide_k, 10);
    }
    for(const double t:{-0.1, 1.1, nan}) bad([&](SamplingParams& p) { p.min_p = t; }, "min_p must be finite and in [0, 1]");
    for(const double t:{0.0, -1.0, nan, inf}) bad([&](SamplingParams& p) { p.repetition_penalty = t; }, "repetition_penalty must be finite and positive");
    for(const double t:{-2.5, 2.5, nan, inf})
        bad([&](SamplingParams& p) { p.frequency_penalty = t; }, "frequency_penalty must be finite and in [-2, 2]");
    for(const double t:{-2.5, 2.5, nan, -inf})
        bad([&](SamplingParams& p) { p.presence_penalty = t; }, "presence_penalty must be finite and in [-2, 2]");
    bad([](SamplingParams& p) { p.logit_bias = {{10, 1.f}}; }, "logit_bias token outside vocabulary");
    bad([](SamplingParams& p) { p.logit_bias = {{-1, 1.f}}; }, "logit_bias token outside vocabulary");
    bad([](SamplingParams& p) { p.logit_bias = {{2, std::nanf("")}}; }, "logit_bias must be finite or negative infinity");
    bad([](SamplingParams& p) { p.logit_bias = {{2, -minus_inf}}; }, "logit_bias must be finite or negative infinity");
    bad([](SamplingParams& p) { p.logit_bias = {{2, 1.f}, {5, 0.f}, {2, -1.f}}; }, "logit_bias tokens must be unique");
    bad([](SamplingParams& p) { p.min_tokens = 1; p.suppress_until_min = {3, -1}; }, "suppress_until_min tokens must be nonnegative");
    // The list is unused without min_tokens, and ids at or past the row are ignored like padding stop tokens.
    SamplingParams unused_list, padded_list;
    unused_list.suppress_until_min = {-1, 3, 10};
    validate(unused_list, 10);
    padded_list.min_tokens = 2; padded_list.suppress_until_min = {3, 10, 4096};
    validate(padded_list, 10);
    bad([](SamplingParams& p) { p.logit_bias.resize(logit_bias_limit + 1); }, "logit_bias exceeds 1024 entries", 4096);
    SamplingParams edges;
    edges.temperature = 0; edges.top_p = 1; edges.top_k = 10; edges.min_p = 1; edges.repetition_penalty = 1e-9;
    edges.frequency_penalty = -2; edges.presence_penalty = 2; edges.logit_bias = {{9, minus_inf}, {0, -1e30f}};
    edges.suppress_until_min = {9, 9};
    validate(edges, 10);
    SamplingParams full;
    for(Token t = 0; t < Token(logit_bias_limit); ++t) full.logit_bias.push_back({t, 0.5f});
    validate(full, 4096);

    // resolve_sampling: set request fields override the adapter defaults.
    SamplingParams defaults;
    defaults.temperature = 0.7; defaults.top_p = 0.9; defaults.top_k = 40; defaults.min_p = 0.05;
    defaults.repetition_penalty = 1.1; defaults.frequency_penalty = 0.1; defaults.presence_penalty = 0.2;
    defaults.logit_bias = {{1, -1.f}}; defaults.min_tokens = 2; defaults.suppress_until_min = {3}; defaults.seed = 99;
    defaults.stream = 4;
    auto r = resolve_sampling(SamplingOptions{}, defaults, 50);
    require(r.temperature == 0.7 && r.top_p == 0.9 && r.top_k == 40 && r.min_p == 0.05 && r.repetition_penalty == 1.1 &&
            r.frequency_penalty == 0.1 && r.presence_penalty == 0.2 && r.logit_bias.size() == 1 && r.min_tokens == 2 &&
            r.suppress_until_min == std::vector<Token>{3} && r.seed == 99 && r.stream == 4, "unset options keep the defaults");
    SamplingOptions o;
    o.temperature = 0; o.top_p = 0.5; o.top_k = 0; o.seed = 7; o.min_p = 0; o.repetition_penalty = 1.3;
    o.frequency_penalty = -1; o.presence_penalty = 0; o.logit_bias = {{2, 3.f}, {4, minus_inf}}; o.min_tokens = 9;
    r = resolve_sampling(o, defaults, 50);
    require(r.temperature == 0 && r.top_p == 0.5 && r.top_k == 0 && r.min_p == 0 && r.repetition_penalty == 1.3 &&
            r.frequency_penalty == -1 && r.presence_penalty == 0 && r.logit_bias.size() == 2 && r.logit_bias[1].token == 4 &&
            r.min_tokens == 9 && r.seed == 7 && r.stream == 4 && r.suppress_until_min == std::vector<Token>{3},
            "set options override, and suppress_until_min and stream stay adapter-owned");
    const auto bad_option = [&](auto edit, std::string_view message) {
        SamplingOptions options;
        edit(options);
        rejects<std::invalid_argument>([&] { resolve_sampling(options, defaults, 50); }, message);
    };
    bad_option([](SamplingOptions& x) { x.temperature = -1; }, "temperature must be finite and nonnegative");
    bad_option([](SamplingOptions& x) { x.top_p = 0; }, "top_p must be finite and in (0, 1]");
    bad_option([](SamplingOptions& x) { x.min_p = 2; }, "min_p must be finite and in [0, 1]");
    bad_option([](SamplingOptions& x) { x.repetition_penalty = 0; }, "repetition_penalty must be finite and positive");
    bad_option([](SamplingOptions& x) { x.frequency_penalty = 3; }, "frequency_penalty must be finite and in [-2, 2]");
    bad_option([](SamplingOptions& x) { x.presence_penalty = -3; }, "presence_penalty must be finite and in [-2, 2]");
    bad_option([](SamplingOptions& x) { x.logit_bias = {{50, 1.f}}; }, "logit_bias token outside vocabulary");
    bad_option([](SamplingOptions& x) { x.logit_bias.resize(logit_bias_limit + 1); }, "logit_bias exceeds 1024 entries");
    SamplingParams broken = defaults;
    broken.suppress_until_min = {-2};
    rejects<std::invalid_argument>([&] { resolve_sampling(SamplingOptions{}, broken, 50); }, "suppress_until_min tokens must be nonnegative");
    broken.suppress_until_min = {3, 50};
    require(resolve_sampling(SamplingOptions{}, broken, 50).suppress_until_min == std::vector<Token>({3, 50}),
            "a padding suppress id resolves and is kept for the sampler to ignore");
    rejects<std::invalid_argument>([&] { resolve_sampling(SamplingOptions{}, defaults, 0); }, "sampling vocabulary must be positive");
    // A top_k the runtime accepted against a padded declared vocabulary resolves against the valid vocabulary.
    SamplingOptions padded;
    padded.top_k = 151700;
    require(resolve_sampling(padded, SamplingParams{}, 151665).top_k == 151700, "top_k past the valid vocabulary resolves");

    // An unset request seed takes defaults.seed: requests sharing one defaults object share their draws, and a
    // fresh per-request seed in the defaults gives independent draws. A set seed, even 0, overrides.
    SamplingParams shared, fresh;
    shared.seed = 11; fresh.seed = 12;
    SamplingOptions unseeded;
    unseeded.temperature = 1;
    SamplingOptions seeded = unseeded;
    seeded.seed = 0;
    require(resolve_sampling(unseeded, shared, 1000).seed == 11 && resolve_sampling(unseeded, fresh, 1000).seed == 12 &&
            resolve_sampling(seeded, fresh, 1000).seed == 0, "an unset request seed takes defaults.seed");
    std::mt19937_64 rng(13);
    const auto row = random_row(rng, 1000, 1.f, false);
    Sampler sampler(1000);
    const auto draws = [&](const SamplingParams& p) {
        TokenHistory h({});
        std::vector<Token> tokens;
        for(int i = 0; i < 16; ++i) {
            tokens.push_back(sampler.sample(row, p, h).token);
            h.stage(one(tokens.back()));
            h.commit(1);
        }
        return tokens;
    };
    const auto first = draws(resolve_sampling(unseeded, shared, 1000));
    require(first == draws(resolve_sampling(unseeded, shared, 1000)), "unseeded requests with shared defaults draw alike");
    require(first != draws(resolve_sampling(unseeded, fresh, 1000)), "a fresh default seed per request draws independently");
}
void history() {
    const std::vector<Token> prompt{4, 4, 9, 1};
    TokenHistory h(prompt);
    require(h.outputs() == 0 && h.staged() == 0 && h.distinct() == 3, "the prompt counts distinct tokens");
    require(h.output_count(4) == 0 && h.output_count(-3) == 0, "prompt tokens are not outputs");
    const std::vector<Token> first{7, 4, 7, 2};
    h.stage(first);
    require(h.staged() == 4 && h.outputs() == 0 && std::equal(first.begin(), first.end(), h.staged_tokens().begin()),
            "staging keeps provisional tokens apart");
    rejects<std::logic_error>([&] { h.stage(first); }, "token history already has staged tokens");
    rejects<std::out_of_range>([&] { h.commit(5); }, "commit exceeds staged tokens");
    require(h.staged() == 4, "a rejected commit changes nothing");
    h.commit(3);
    require(h.outputs() == 3 && h.staged() == 0 && h.output_count(7) == 2 && h.output_count(4) == 1 &&
            h.output_count(2) == 0 && h.distinct() == 4, "commit keeps the accepted prefix and drops the rest");
    h.stage(first);
    h.rollback();
    require(h.staged() == 0 && h.outputs() == 3, "rollback drops staged tokens");
    h.stage({});
    h.commit(0);
    require(h.outputs() == 3, "empty rounds are no-ops");
    rejects<std::invalid_argument>([&] { TokenHistory bad(std::vector<Token>{1, -2}); }, "token history tokens must be nonnegative");
    const std::vector<Token> negative{3, -1};
    rejects<std::invalid_argument>([&] { h.stage(negative); }, "token history tokens must be nonnegative");
    require(h.staged() == 0, "a rejected stage changes nothing");
    std::size_t before = allocation_count;
    TokenHistory outputs_only(prompt, false);
    require(outputs_only.distinct() == 0 && allocation_count == before, "uncounted prompts are not tracked");
    // Memory follows the distinct tokens, not the prompt length: 200k tokens over 100 ids.
    std::vector<Token> repetitive(200000);
    for(std::size_t i = 0; i < repetitive.size(); ++i) repetitive[i] = Token(i % 100 * 7919);
    tracked_live = tracked_peak = 0;
    {
        std::optional<TokenHistory> long_prompt;
        tracked([&] { long_prompt.emplace(repetitive); });
        require(long_prompt->distinct() == 100, "a repetitive prompt tracks its distinct tokens");
        require(tracked_peak <= TokenHistory::host_bytes(100, 0), "a repetitive prompt holds room for its distinct tokens only");
        for(Token t = 0; t < 100; ++t) require(long_prompt->output_count(t*7919) == 0, "prompt tokens are not outputs");
    }
    require(tracked_live == 0, "a history releases its memory");
    require(TokenHistory::host_bytes(std::numeric_limits<std::size_t>::max(), 1) == std::numeric_limits<std::size_t>::max() &&
            TokenHistory::host_bytes(1, std::numeric_limits<std::size_t>::max()) == std::numeric_limits<std::size_t>::max(),
            "host_bytes saturates");
    require(TokenHistory::host_bytes(100, 100) <= TokenHistory::host_bytes(100, 1000), "host_bytes grows with outputs");

    // Randomized oracle: counts, frontier and memory bound over many rounds, with tokens drawn from a large id
    // space so a dense table would be visible.
    std::mt19937_64 rng(5);
    for(int trial = 0; trial < 20; ++trial) {
        const bool count_prompt = trial % 2 == 0;
        std::vector<Token> start(rng() % 300);
        std::uniform_int_distribution<Token> id(0, trial < 10 ? 50 : 2000000000);
        for(auto& token:start) token = id(rng);
        tracked_live = tracked_peak = 0;
        std::size_t most = 0;
        {
            std::optional<TokenHistory> model;
            tracked([&] { model.emplace(start, count_prompt); });
            std::map<Token, std::uint32_t> counts;
            std::set<Token> tracked_tokens;
            if(count_prompt) tracked_tokens.insert(start.begin(), start.end());
            std::uint64_t committed = 0;
            for(int step = 0; step < 300; ++step) {
                std::vector<Token> round(rng() % 9);
                for(auto& token:round) token = id(rng);
                tracked([&] { model->stage(round); });
                most = std::max<std::size_t>(most, committed + round.size());
                if(rng() % 4 == 0) { model->rollback(); continue; }
                const std::size_t accepted = rng() % (round.size() + 1);
                tracked([&] { model->commit(accepted); });
                for(std::size_t i = 0; i < accepted; ++i) { ++counts[round[i]]; tracked_tokens.insert(round[i]); }
                committed += accepted;
                require(model->outputs() == committed && model->staged() == 0, "frontier matches the oracle");
            }
            require(model->distinct() == tracked_tokens.size(), "distinct counts tracked tokens only");
            for(const auto& [token, n]:counts) require(model->output_count(token) == n, "output counts match the oracle");
            for(const Token token:start) if(!counts.count(token)) require(model->output_count(token) == 0, "prompt is not output");
        }
        require(tracked_peak <= TokenHistory::host_bytes(start.size(), most), "host_bytes bounds the peak heap use");
        require(tracked_live == 0, "a history releases its memory");
    }
}
void copies() {
    // A 16-token prompt fills the entries room exactly, so staging new ids makes stage() reserve more; a copy taken
    // between stage and commit must keep that room.
    std::vector<Token> prompt(16);
    for(std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = Token(i);
    TokenHistory parent(prompt);
    const std::vector<Token> round{3, 100, 101};
    parent.stage(round);
    TokenHistory child(parent);
    require(child.staged() == 3 && std::equal(round.begin(), round.end(), child.staged_tokens().begin()) &&
            child.distinct() == 16 && child.outputs() == 0, "a copy keeps the staged round");
    std::size_t before = allocation_count;
    child.commit(3);
    require(allocation_count == before, "commit on a copy taken between stage and commit never allocates");
    require(child.outputs() == 3 && child.output_count(3) == 1 && child.output_count(100) == 1 &&
            child.output_count(101) == 1 && child.distinct() == 18, "the copy commits its round once");
    require(parent.staged() == 3 && parent.outputs() == 0 && parent.output_count(3) == 0 && parent.distinct() == 16,
            "the source is unaffected by its copy");
    TokenHistory assigned({});
    assigned = parent;
    before = allocation_count;
    assigned.commit(2);
    require(allocation_count == before && assigned.outputs() == 2 && assigned.output_count(100) == 1 &&
            assigned.output_count(101) == 0, "commit on a copy-assigned history never allocates");
    parent.commit(1);
    require(parent.outputs() == 1 && parent.output_count(3) == 1 && parent.output_count(100) == 0, "the source keeps its round");
    const TokenHistory& same = assigned;
    assigned = same;
    require(assigned.outputs() == 2 && assigned.output_count(100) == 1, "self-assignment keeps the history");

    // Moves transfer the history and leave the source empty and usable.
    TokenHistory moved(std::move(child));
    require(moved.outputs() == 3 && moved.output_count(3) == 1 && moved.distinct() == 18, "a move transfers the history");
    require(child.outputs() == 0 && child.staged() == 0 && child.distinct() == 0 && child.output_count(3) == 0,
            "a moved-from history is empty");
    child.stage(round);
    child.commit(3);
    require(child.outputs() == 3 && child.output_count(101) == 1 && child.distinct() == 3, "a moved-from history is usable");
    TokenHistory target(prompt);
    target = std::move(moved);
    require(target.outputs() == 3 && target.distinct() == 18 && moved.outputs() == 0 && moved.distinct() == 0,
            "move assignment transfers the history");
    TokenHistory& alias = target;
    target = std::move(alias);
    require(target.outputs() == 3 && target.output_count(100) == 1, "self-move keeps the history");

    // A copy is its own history for host_bytes: copying, staging new ids and committing them stays within its bound.
    std::vector<Token> wide(1000);
    for(std::size_t i = 0; i < wide.size(); ++i) wide[i] = Token(i);
    TokenHistory big(wide);
    tracked_live = tracked_peak = 0;
    {
        std::optional<TokenHistory> fork;
        const std::vector<Token> fresh{5000, 5001, 5002, 5003};
        tracked([&] { fork.emplace(big); fork->stage(fresh); fork->commit(4); });
        require(fork->outputs() == 4 && fork->distinct() == 1004, "the fork commits new ids");
        require(tracked_peak <= TokenHistory::host_bytes(1000, 4), "host_bytes bounds a copied history");
    }
    require(tracked_live == 0, "a copied history releases its memory");

    // Randomized oracle: forks by copy construction, copy assignment and move assignment at random points, mostly
    // between stage and commit; every commit stays allocation-free and the counts match.
    std::mt19937_64 rng(31);
    for(int trial = 0; trial < 20; ++trial) {
        std::uniform_int_distribution<Token> id(0, trial % 2 ? 40 : 1000000000);
        std::vector<Token> start(rng() % 200);
        for(auto& token:start) token = id(rng);
        TokenHistory model(start);
        std::map<Token, std::uint32_t> counts;
        std::set<Token> tracked_tokens(start.begin(), start.end());
        std::uint64_t committed = 0;
        for(int step = 0; step < 200; ++step) {
            std::vector<Token> next(rng() % 12);
            for(auto& token:next) token = id(rng);
            model.stage(next);
            switch(rng() % 4) {
            case 0: { TokenHistory fork(model); model = std::move(fork); break; }
            case 1: { const TokenHistory fork(model); model = fork; break; }
            case 2: { TokenHistory fork(model); model.rollback(); model = std::move(fork); break; }
            default: break;
            }
            const std::size_t accepted = rng() % (next.size() + 1);
            before = allocation_count;
            model.commit(accepted);
            require(allocation_count == before, "commit never allocates, also on copies");
            for(std::size_t i = 0; i < accepted; ++i) { ++counts[next[i]]; tracked_tokens.insert(next[i]); }
            committed += accepted;
            require(model.outputs() == committed && model.staged() == 0, "forked frontier matches the oracle");
        }
        require(model.distinct() == tracked_tokens.size(), "forked histories track distinct tokens");
        for(const auto& [token, n]:counts) require(model.output_count(token) == n, "forked output counts match the oracle");
    }
}
void greedy() {
    constexpr std::size_t V = 50000;
    std::mt19937_64 rng(11);
    auto row = random_row(rng, V, 2.f, true);
    row[777] = row[31337] = 30.f;
    Sampler s(V);
    TokenHistory h({});
    SamplingParams p;
    p.temperature = 0;
    const Real lse = oracle_lse(row);
    for(std::uint64_t seed = 0; seed < 4; ++seed) {
        p.seed = seed;
        const auto c = s.sample(row, p, h);
        require(c.token == 777, "greedy ties go to the lowest id");
        require(close(c.logprob, Real(30) - lse), "greedy reports the raw log-probability");
    }
    std::vector<TokenLogprob> law;
    s.distribution(row, p, h, 0, law);
    require(law.size() == 1 && law[0].token == 777 && law[0].logprob == 0, "greedy law is a point mass");
    SamplingParams k;
    k.temperature = 1.5; k.top_k = 1;
    SamplingParams tiny;
    tiny.temperature = 1.5; tiny.top_p = 1e-6;
    for(std::uint64_t seed = 0; seed < 50; ++seed) {
        k.seed = tiny.seed = seed;
        require(s.sample(row, k, h).token == 777, "top_k 1 reduces to greedy");
        require(s.sample(row, tiny, h).token == 777, "a tiny top_p reduces to greedy");
    }
    p.logit_bias = {{777, minus_inf}};
    require(s.sample(row, p, h).token == 31337, "greedy honours bans");
    p.logit_bias = {{5, 100.f}};
    require(s.sample(row, p, h).token == 5, "greedy honours bias");
    p.logit_bias.clear();
    p.repetition_penalty = 2;
    const std::vector<Token> prompt{777};
    require(s.sample(row, p, TokenHistory(prompt)).token == 31337, "greedy honours penalties");
    require(s.sample(row, p, TokenHistory(prompt, false)).token == 777, "uncounted prompts are not penalized");
    // Ties after processing still go to the lowest id.
    const std::vector<float> flat{1.f, 3.f, 2.f, 3.f};
    SamplingParams lifted;
    lifted.temperature = 0;
    lifted.logit_bias = {{2, 1.f}};
    Sampler small(4);
    require(small.sample(flat, lifted, h).token == 1, "processed ties go to the lowest id");
}
void law() {
    // Randomized scenarios: quarter-grid logits with ties and masks, every filter, penalties, bias, suppression
    // and verify columns, against the long-double oracle.
    std::mt19937_64 rng(2026);
    std::size_t checked = 0, empty = 0;
    const std::vector<double> temperatures{0, 0.25, 0.7, 1, 1.6};
    for(int trial = 0; trial < 4000; ++trial) {
        const std::size_t V = 1 + rng() % 40;
        auto logits = random_row(rng, V, 1.5f, true);
        for(auto& x:logits) if(rng() % 9 == 0) x = minus_inf;
        SamplingParams p;
        p.temperature = temperatures[rng() % temperatures.size()];
        p.top_k = rng() % 3 ? std::uint32_t(rng() % (V + 3)) : 0;   // some past the vocabulary
        std::uniform_real_distribution<double> unit(0.01, 1);
        p.top_p = rng() % 2 ? 1.0 : unit(rng);
        p.min_p = rng() % 3 ? 0.0 : unit(rng)*0.5;
        p.repetition_penalty = rng() % 2 ? 1.0 : (rng() % 2 ? 1.3 : 0.8);
        p.frequency_penalty = rng() % 2 ? 0.0 : double(int(rng() % 17) - 8)/4;
        p.presence_penalty = rng() % 2 ? 0.0 : double(int(rng() % 17) - 8)/4;
        for(Token t = 0; t < Token(V); ++t) if(rng() % 6 == 0)
            p.logit_bias.push_back({t, rng() % 4 ? float(int(rng() % 33) - 16)/8 : minus_inf});
        p.min_tokens = std::uint32_t(rng() % 6);
        for(Token t = 0; t < Token(V); ++t) if(rng() % 8 == 0) p.suppress_until_min.push_back(t);
        p.seed = rng();
        p.stream = std::uint32_t(rng() % 3);
        std::uniform_int_distribution<Token> id(0, Token(V) + 2);   // some history ids lie past the row
        std::vector<Token> prompt(rng() % 6), committed(rng() % 5), staged(rng() % 5);
        for(auto& t:prompt) t = id(rng);
        for(auto& t:committed) t = id(rng);
        for(auto& t:staged) t = id(rng);
        const std::size_t column = rng() % (staged.size() + 1);
        Sampler sampler(static_cast<std::uint32_t>(V));
        if(check_law(sampler, logits, p, prompt, rng() % 2 == 0, committed, staged, column)) ++checked; else ++empty;
    }
    require(checked > 3000 && empty > 10, "scenarios cover selectable and fully masked rows");
    // Larger rows reach the partial-selection paths: top_k above the heap limit, and top_p/min_p walks.
    for(int trial = 0; trial < 12; ++trial) {
        const std::size_t V = 3000 + rng() % 3000;
        auto logits = random_row(rng, V, 2.f, trial % 2 == 0);
        SamplingParams p;
        p.temperature = trial % 3 == 0 ? 1.0 : 0.6;
        p.top_k = trial % 4 == 0 ? std::uint32_t(1500 + rng() % 1000) : 0;
        p.top_p = trial % 2 ? 0.9 : 1.0;
        p.min_p = trial % 5 == 1 ? 1e-4 : 0.0;
        p.seed = rng();
        Sampler sampler(static_cast<std::uint32_t>(V));
        require(check_law(sampler, logits, p, {}, true, {}, {}, 0), "large rows are selectable");
        for(std::uint64_t position = 1; position < 40; ++position)
            require(check_law(sampler, logits, p, {}, true, std::vector<Token>(position, Token(V + 1)), {}, 0),
                    "large-row draws match the oracle");
    }
    // Exact boundaries: reaching top_p counts, and equality with min_p is kept.
    const std::vector<float> equal{0.f, 0.f, 0.f, 0.f};
    Sampler four(4);
    TokenHistory h({});
    std::vector<TokenLogprob> out;
    const auto kept = [&](SamplingParams p) { four.distribution(equal, p, h, 0, out); return out.size(); };
    SamplingParams p;
    p.top_p = 0.5;
    require(kept(p) == 2, "a prefix reaching top_p exactly stops there");
    p.top_p = 0.5000001;
    require(kept(p) == 3, "a prefix short of top_p continues");
    p.top_p = 0.25;
    require(kept(p) == 1, "top_p keeps at least the first token");
    const std::vector<float> steps{0.f, 0.f, -1.f};
    Sampler three(3);
    SamplingParams m;
    m.min_p = std::exp(-1.0);
    three.distribution(steps, m, h, 0, out);
    require(out.size() == 3, "min_p keeps probabilities equal to the threshold");
    m.min_p = std::nextafter(std::exp(-1.0), 1.0);
    three.distribution(steps, m, h, 0, out);
    require(out.size() == 2 && out[0].token == 0 && out[1].token == 1, "min_p drops probabilities below the threshold");
    m.min_p = 1;
    three.distribution(steps, m, h, 0, out);
    require(out.size() == 2, "min_p 1 keeps the tokens tied at the maximum");
    // Weights that underflow to zero carry no probability and are never drawn.
    const std::vector<float> far{0.f, -2000.f, -3000.f};
    SamplingParams warm;
    three.distribution(far, warm, h, 0, out);
    require(out.size() == 1 && out[0].token == 0 && out[0].logprob == 0, "underflowed weights are dropped");
}
void frequency() {
    const std::vector<float> logits{1.0f, 0.5f, 0.0f, -0.5f, 2.0f, minus_inf, 1.25f, -3.0f};
    SamplingParams p;
    p.temperature = 0.9; p.top_p = 0.97; p.min_p = 0.02; p.seed = 20261015; p.stream = 2;
    const auto run = [&](Sampler& s, const std::vector<float>& row, int draws) {
        TokenHistory h({});
        std::vector<TokenLogprob> law;
        s.distribution(row, p, h, 0, law);
        std::vector<double> expected(row.size(), 0);
        double mass = 0;
        for(const auto& e:law) { expected[std::size_t(e.token)] = std::exp(double(e.logprob)); mass += expected[std::size_t(e.token)]; }
        require(std::fabs(mass - 1) < 1e-5, "the law is normalized");
        std::vector<int> seen(row.size(), 0);
        for(int i = 0; i < draws; ++i) {
            const Token token = s.sample(row, p, h).token;
            ++seen[std::size_t(token)];
            h.stage(one(token));
            h.commit(1);
        }
        return std::pair(expected, seen);
    };
    Sampler small(std::uint32_t(logits.size()));
    constexpr int draws = 200000;
    const auto [expected, seen] = run(small, logits, draws);
    for(std::size_t i = 0; i < logits.size(); ++i) {
        const double pr = expected[i], got = double(seen[i])/draws;
        require(pr > 0 || seen[i] == 0, "tokens outside the law are never drawn");
        require(std::fabs(got - pr) < 4*std::sqrt(pr*(1 - pr)/draws) + 1e-3, "frequencies follow distribution()");
    }
    // A wide row exercises the histogram-guided rank walk: the top tokens individually and the rest in bulk.
    std::mt19937_64 rng(8);
    const auto wide = random_row(rng, 1000, 1.5f, false);
    Sampler large(1000);
    p.top_p = 0.9; p.min_p = 0; p.temperature = 1;
    constexpr int wide_draws = 10000;
    const auto [wide_law, counts] = run(large, wide, wide_draws);
    std::vector<std::size_t> order(wide.size());
    for(std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return wide_law[a] > wide_law[b]; });
    double rest_expected = 0, rest_seen = 0;
    for(std::size_t r = 0; r < order.size(); ++r) {
        const double pr = wide_law[order[r]], got = double(counts[order[r]])/wide_draws;
        require(pr > 0 || counts[order[r]] == 0, "tokens outside the nucleus are never drawn");
        if(r < 8) require(std::fabs(got - pr) < 4*std::sqrt(pr*(1 - pr)/wide_draws) + 1e-3, "top frequencies follow the law");
        else { rest_expected += pr; rest_seen += got; }
    }
    require(std::fabs(rest_seen - rest_expected) < 4*std::sqrt(rest_expected*(1 - rest_expected)/wide_draws) + 1e-3,
            "tail mass follows the law");
}
void redo() {
    constexpr std::size_t V = 64;
    std::mt19937_64 rng(3);
    std::vector<std::vector<float>> rows;
    for(int c = 0; c < 5; ++c) rows.push_back(random_row(rng, V, 1.f, false));
    SamplingParams p;
    p.temperature = 0.8; p.top_p = 0.95; p.repetition_penalty = 1.3; p.frequency_penalty = 0.4; p.presence_penalty = 0.2;
    p.seed = 77; p.stream = 3;
    const std::vector<Token> prompt{1, 2, 3, 2}, committed{5, 6}, drafts{7, 5, 9, 11};
    Sampler s(V);
    auto h = make_history(prompt, true, committed, {});
    h.stage(drafts);
    std::vector<Choice> first;
    for(std::size_t c = 0; c <= drafts.size(); ++c) first.push_back(s.sample(rows[c], p, h, c));
    const auto same = [&](const char* message) {
        for(std::size_t c = 0; c <= drafts.size(); ++c) {
            const auto again = s.sample(rows[c], p, h, c);
            require(again.token == first[c].token && again.logprob == first[c].logprob, message);
        }
    };
    same("repeated draws are identical");
    h.rollback();
    h.stage(drafts);
    same("a draw after rollback is identical");
    h.commit(0);
    h.stage(drafts);
    same("a draw after a discarded round is identical");
    for(std::size_t c = 0; c <= drafts.size(); ++c) {
        const auto direct = make_history(prompt, true, concat(committed, std::span(drafts).first(c)), {});
        const auto choice = s.sample(rows[c], p, direct, 0);
        require(choice.token == first[c].token && choice.logprob == first[c].logprob,
                "column c sees committed outputs plus staged[0, c) at position outputs + c");
    }
    h.commit(2);
    require(h.outputs() == 4, "commit advances the draw position");
    const auto after = s.sample(rows[2], p, h, 0);
    require(after.token == first[2].token, "the first draw after a partial commit redoes the matching column");
    // Different streams give different draw sequences.
    SamplingParams other = p;
    other.stream = 4;
    auto plain = make_history({}, true, {}, {});
    int differ = 0;
    for(int i = 0; i < 50; ++i) {
        differ += s.sample(rows[0], p, plain).token != s.sample(rows[0], other, plain).token;
        const Token token = s.sample(rows[0], p, plain).token;
        plain.stage(one(token));
        plain.commit(1);
    }
    require(differ > 5, "streams are independent");
}
void penalties() {
    const std::vector<float> logits{2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 1.0f};
    Sampler s(6);
    SamplingParams p;
    p.repetition_penalty = 2; p.frequency_penalty = 0.5; p.presence_penalty = 0.25;
    std::vector<TokenLogprob> law;
    const auto processed = [&](const TokenHistory& h, std::size_t column, const SamplingParams& q) {
        s.distribution(logits, q, h, column, law);
        std::vector<double> lp(6, 0);
        for(const auto& e:law) lp[std::size_t(e.token)] = e.logprob;
        return lp;
    };
    const auto expect = [&](const std::vector<double>& lp, const std::vector<double>& v, const char* message) {
        for(std::size_t i = 1; i < v.size(); ++i) require(near(lp[i] - lp[0], v[i] - v[0], 2e-6L), message);
    };
    // Repetition counts distinct prompt and output tokens once (HF: l > 0 ? l / r : l * r); frequency and presence
    // count outputs only (l -= f * count + p).
    const std::vector<Token> prompt{0, 1, 1, 100}, outputs{3, 3, 5};
    const auto h = make_history(prompt, true, outputs, {});
    expect(processed(h, 0, p), {1, -2, 0.5, 0.25, -2, -0.25}, "penalties match hand-computed logits");
    expect(processed(make_history(prompt, false, outputs, {}), 0, p), {2, -1, 0.5, 0.25, -2, -0.25},
           "uncounted prompts skip the repetition penalty");
    const std::vector<Token> committed{3, 3}, staged{5, 2};
    const auto spec = make_history(prompt, true, committed, staged);
    expect(processed(spec, 1, p), {1, -2, 0.5, 0.25, -2, -0.25}, "column 1 sees staged[0, 1)");
    expect(processed(spec, 2, p), {1, -2, -0.5, 0.25, -2, -0.25}, "column 2 sees staged[0, 2)");
    expect(processed(spec, 0, p), {1, -2, 0.5, 1.5 - 1.25, -2, 1}, "column 0 sees committed outputs only");
    SamplingParams reward;
    reward.frequency_penalty = -1;
    expect(processed(h, 0, reward), {2, -1, 0.5, 5, -2, 2}, "negative frequency penalties reward outputs");
    SamplingParams presence;
    presence.presence_penalty = 2;
    expect(processed(h, 0, presence), {2, -1, 0.5, 1, -2, -1}, "presence penalties apply once per output token");
    SamplingParams g = p;
    g.temperature = 0;
    require(s.sample(logits, SamplingParams{.temperature = 0}, h).token == 3, "unpenalized greedy picks the maximum");
    require(s.sample(logits, g, h).token == 0, "penalized greedy picks the processed maximum");
    // Scale: thousands of distinct outputs stay sparse and penalize correctly.
    constexpr std::uint32_t V = 250000;
    std::vector<float> wide(V, 0.f);
    std::vector<Token> many;
    for(Token t = 0; t < 30000; ++t) many.push_back(t*7 % Token(V));
    const auto big = make_history({}, true, many, {});
    require(big.distinct() == 30000, "history tracks distinct outputs only");
    Sampler ws(V);
    SamplingParams wg;
    wg.temperature = 0; wg.presence_penalty = 1;
    require(ws.sample(wide, wg, big).token == 1, "penalties cover every tracked output");
    // Extreme repetition penalties clamp to the finite double range instead of producing +inf or a new mask.
    Sampler es(4);
    const auto seen = make_history({}, true, {0, 1}, {});
    const std::vector<float> huge{3e38f, 2e38f, 1.f, 0.f}, tiny{-3e38f, -2e38f, minus_inf, -1.f};
    SamplingParams boost;
    boost.repetition_penalty = 1e-300;
    es.distribution(huge, boost, seen, 0, law);
    require(law.size() == 2 && law[0].token == 0 && law[1].token == 1 && close(law[0].logprob, std::log(0.5L)) &&
            close(law[1].logprob, std::log(0.5L)), "overflowing boosts saturate and tie by id");
    const Choice boosted = es.sample(huge, boost, seen);
    require((boosted.token == 0 || boosted.token == 1) && boosted.logprob <= 0, "saturated rows still draw");
    boost.temperature = 0;
    require(es.sample(huge, boost, seen).token == 0, "saturated greedy picks the lowest id");
    SamplingParams crush;
    crush.repetition_penalty = 1e300;
    es.distribution(tiny, crush, seen, 0, law);
    require(law.size() == 1 && law[0].token == 3, "crushed tokens stay finite but carry no weight");
    const std::vector<float> all_seen{-3e38f, -2e38f, minus_inf, minus_inf};
    es.distribution(all_seen, crush, seen, 0, law);
    require(law.size() == 2 && law[0].token == 0 && law[1].token == 1, "crushed tokens are not masked");
}
void bias() {
    const std::vector<float> logits{0.f, 1.f, 2.f, 3.f, 4.f};
    Sampler s(5);
    TokenHistory h({});
    SamplingParams p;
    p.temperature = 0;
    p.logit_bias = {{4, minus_inf}};
    require(s.sample(logits, p, h).token == 3, "a -inf bias bans the token");
    p.logit_bias = {{0, 10.f}};
    require(s.sample(logits, p, h).token == 0, "a positive bias promotes the token");
    p.logit_bias = {{4, minus_inf}, {3, minus_inf}};
    p.temperature = 1;
    std::vector<TokenLogprob> law;
    s.distribution(logits, p, h, 0, law);
    require(law.size() == 3 && law[0].token == 2, "banned tokens leave the law");
    for(int i = 0; i < 200; ++i) {
        p.seed = std::uint64_t(i);
        const Token token = s.sample(logits, p, h).token;
        require(token != 3 && token != 4, "banned tokens are never drawn");
    }
    p.logit_bias = {{0, -1.5f}, {2, 0.75f}};
    s.distribution(logits, p, h, 0, law);
    std::vector<double> lp(5);
    for(const auto& e:law) lp[std::size_t(e.token)] = e.logprob;
    require(near(lp[2] - lp[0], (2 + 0.75) - (0 - 1.5), 2e-6L) && near(lp[4] - lp[1], 3, 2e-6L), "bias adds to logits");
    p.logit_bias.clear();
    for(Token t = 0; t < 5; ++t) p.logit_bias.push_back({t, minus_inf});
    rejects<std::invalid_argument>([&] { s.sample(logits, p, h); }, "no selectable token");
    p.temperature = 0;
    rejects<std::invalid_argument>([&] { s.sample(logits, p, h); }, "no selectable token");
    // A bias cannot unmask a -inf logit.
    const std::vector<float> masked{minus_inf, 0.f};
    Sampler two(2);
    SamplingParams lift;
    lift.temperature = 0;
    lift.logit_bias = {{0, 100.f}};
    require(two.sample(masked, lift, h).token == 1, "bias keeps masks");
}
void min_tokens() {
    const std::vector<float> logits{0.f, 1.f, 2.f, 3.f, 9.f};   // 4 is the stop token
    Sampler s(5);
    SamplingParams p;
    p.temperature = 0; p.min_tokens = 3; p.suppress_until_min = {4};
    const std::vector<Token> drafts{1, 1, 1};
    auto h = make_history({}, true, {}, drafts);
    require(s.sample(logits, p, h, 0).token == 3, "stops are suppressed before min_tokens");
    require(s.sample(logits, p, h, 2).token == 3, "verify columns count staged outputs");
    require(s.sample(logits, p, h, 3).token == 4, "suppression ends at min_tokens");
    h.commit(3);
    require(s.sample(logits, p, h, 0).token == 4, "committed outputs end suppression");
    SamplingParams empty_list = p;
    empty_list.suppress_until_min.clear();
    require(s.sample(logits, empty_list, make_history({}, true, {}, {}), 0).token == 4, "an empty list suppresses nothing");
    const std::vector<float> only_stop{minus_inf, minus_inf, minus_inf, minus_inf, 1.f};
    rejects<std::invalid_argument>([&] { s.sample(only_stop, p, make_history({}, true, {}, {})); }, "no selectable token");
    SamplingParams sampled = p;
    sampled.temperature = 1;
    for(int i = 0; i < 100; ++i) {
        sampled.seed = std::uint64_t(i);
        require(s.sample(logits, sampled, make_history({}, true, {}, {})).token != 4, "sampling suppresses stops");
    }
}
void logprobs() {
    std::mt19937_64 rng(17);
    for(int trial = 0; trial < 60; ++trial) {
        const std::size_t V = 1 + rng() % 300;
        auto row = random_row(rng, V, trial % 3 == 0 ? 20.f : 3.f, trial % 2 == 0);
        for(auto& x:row) if(rng() % 7 == 0) x = minus_inf;
        if(std::all_of(row.begin(), row.end(), [](float x) { return x == minus_inf; })) row[0] = 0.f;
        const Real lse = oracle_lse(row);
        std::vector<float> lp(V);
        log_softmax(row, lp);
        for(std::size_t i = 0; i < V; ++i) {
            if(row[i] == minus_inf) require(lp[i] == minus_inf, "masked entries get -inf");
            else require(close(lp[i], Real(row[i]) - lse), "log_softmax matches the oracle");
        }
        auto in_place = row;
        log_softmax(in_place, in_place);
        require(in_place == lp, "log_softmax runs in place");
        std::vector<std::size_t> order(V);
        for(std::size_t i = 0; i < V; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return lp[a] > lp[b] || (lp[a] == lp[b] && a < b); });
        for(const std::size_t n:{std::size_t(0), std::size_t(1), std::min<std::size_t>(V, 10), V}) {
            std::vector<TokenLogprob> top(n);
            top_logprobs(row, top);
            for(std::size_t r = 0; r < n; ++r) {
                require(top[r].token == Token(order[r]), "top rows rank by logprob desc, then id asc");
                require(top[r].logprob == lp[order[r]], "top rows carry log_softmax values");
            }
        }
        std::vector<Token> ids(rng() % 20);
        for(auto& t:ids) t = Token(rng() % V);
        std::vector<float> q(ids.size());
        query_logprobs(row, ids, q);
        for(std::size_t i = 0; i < ids.size(); ++i) require(q[i] == lp[std::size_t(ids[i])], "queries gather log_softmax values");
        // The chosen logprob and alternatives are raw: independent of temperature, bias and penalties.
        Sampler s(static_cast<std::uint32_t>(V));
        SamplingParams p;
        p.temperature = 0.3 + double(trial % 4); p.repetition_penalty = 1.4; p.logit_bias = {{0, 0.5f}};
        p.seed = std::uint64_t(trial);
        const std::vector<Token> prompt{0, Token(V - 1)};
        std::vector<TokenLogprob> alternatives(std::min<std::size_t>(V, 5)), reference(alternatives.size());
        const auto choice = s.sample(row, p, TokenHistory(prompt), 0, alternatives);
        top_logprobs(row, reference);
        require(choice.logprob == lp[std::size_t(choice.token)], "choice logprob is the raw log_softmax value");
        const auto unit = s.sample(row, SamplingParams{.seed = std::uint64_t(trial)}, TokenHistory({}));
        require(unit.logprob == lp[std::size_t(unit.token)], "the T = 1 shortcut equals log_softmax exactly");
        for(std::size_t r = 0; r < alternatives.size(); ++r)
            require(alternatives[r].token == reference[r].token && alternatives[r].logprob == reference[r].logprob,
                    "sample alternatives equal top_logprobs");
    }
    // Ties rank by id; extreme magnitudes stay finite.
    const std::vector<float> ties{1.f, 2.f, 2.f, 1.f, 2.f};
    std::vector<TokenLogprob> top(5);
    top_logprobs(ties, top);
    const std::vector<Token> ranked{1, 2, 4, 0, 3};
    for(std::size_t r = 0; r < 5; ++r) require(top[r].token == ranked[r], "tied logprobs rank by id");
    // Distinct logits that round to one float log-probability still rank by id, and selection follows that order.
    const std::vector<float> rounded{30.f, std::nextafter(1e-3f, 0.f), 1e-3f};
    std::vector<float> rounded_lp(3);
    log_softmax(rounded, rounded_lp);
    require(rounded_lp[1] == rounded_lp[2], "nearby logits round to one log-probability");
    std::vector<TokenLogprob> pair(2);
    top_logprobs(rounded, pair);
    require(pair[0].token == 0 && pair[1].token == 1, "rounded ties select and rank by id");
    const std::vector<float> extreme{1e30f, 0.f, -1e30f};
    std::vector<float> lp(3);
    log_softmax(extreme, lp);
    require(lp[0] == 0 && lp[1] == -1e30f && lp[2] == -2e30f, "log_softmax is stable at extreme magnitudes");
}
void infinity() {
    const std::vector<float> row{minus_inf, 0.5f, minus_inf, 0.5f, -1.f};
    std::vector<TokenLogprob> top(5);
    top_logprobs(row, top);
    require(top[3].token == 0 && top[4].token == 2 && top[3].logprob == minus_inf && top[4].logprob == minus_inf,
            "masked alternatives rank last by id with -inf");
    Sampler s(5);
    TokenHistory h({});
    SamplingParams hot;
    hot.temperature = 50;
    for(int i = 0; i < 300; ++i) {
        hot.seed = std::uint64_t(i);
        const Token token = s.sample(row, hot, h).token;
        require(token != 0 && token != 2, "masked tokens are never drawn");
    }
    const std::vector<float> none(5, minus_inf), nan_row{0.f, std::nanf(""), 1.f, 0.f, 0.f}, plus{0.f, 1.f, -minus_inf, 0.f, 0.f};
    std::vector<float> out(5);
    std::vector<TokenLogprob> law;
    const std::vector<Token> ids{0};
    std::vector<float> q(1);
    for(const auto* bad:{&nan_row, &plus}) {
        const auto& r = *bad;
        rejects<std::invalid_argument>([&] { s.sample(r, SamplingParams{}, h); }, "logits must be finite or -inf");
        rejects<std::invalid_argument>([&] { s.sample(r, SamplingParams{.temperature = 0}, h); }, "logits must be finite or -inf");
        rejects<std::invalid_argument>([&] { s.distribution(r, SamplingParams{}, h, 0, law); }, "logits must be finite or -inf");
        rejects<std::invalid_argument>([&] { log_softmax(r, out); }, "logits must be finite or -inf");
        rejects<std::invalid_argument>([&] { top_logprobs(r, top); }, "logits must be finite or -inf");
        rejects<std::invalid_argument>([&] { query_logprobs(r, ids, q); }, "logits must be finite or -inf");
        const std::vector<Token> target{1};
        rejects<std::invalid_argument>([&] { score_logit_rows({r, 5, 1, 5}, target, {}, 0, 0, false); }, "logits must be finite or -inf");
    }
    rejects<std::invalid_argument>([&] { s.sample(none, SamplingParams{}, h); }, "no selectable token");
    rejects<std::invalid_argument>([&] { s.sample(none, SamplingParams{.temperature = 0}, h); }, "no selectable token");
    rejects<std::invalid_argument>([&] { log_softmax(none, out); }, "no selectable token");
    rejects<std::invalid_argument>([&] { top_logprobs(none, top); }, "no selectable token");
    rejects<std::invalid_argument>([&] { query_logprobs(none, ids, q); }, "no selectable token");
    rejects<std::invalid_argument>([&] { log_softmax({}, {}); }, "no selectable token");
}
void errors() {
    rejects<std::invalid_argument>([] { Sampler(0); }, "sampler vocabulary must be positive");
    rejects<std::invalid_argument>([] { Sampler(2147483649u); }, "sampler vocabulary exceeds the token id range");
    Sampler s(4);
    require(s.vocabulary() == 4 && s.scratch_bytes() == Sampler::scratch_bytes(4), "scratch is reported exactly");
    const std::vector<float> row{0.f, 1.f, 2.f, 3.f}, short_row{0.f, 1.f};
    const std::vector<Token> staged{1, 2};
    const auto h = make_history({}, true, {}, staged);
    rejects<std::invalid_argument>([&] { s.sample(short_row, SamplingParams{}, h); }, "logits size must equal the sampler vocabulary");
    rejects<std::out_of_range>([&] { s.sample(row, SamplingParams{}, h, 3); }, "column exceeds staged tokens");
    std::vector<TokenLogprob> big(5), law;
    rejects<std::invalid_argument>([&] { s.sample(row, SamplingParams{}, h, 0, big); }, "top logprobs exceed the logits row");
    rejects<std::out_of_range>([&] { s.distribution(row, SamplingParams{}, h, 3, law); }, "column exceeds staged tokens");
    // A sampler over the valid vocabulary (4 here, as if the declared vocabulary were padded to 8) treats a top_k at
    // or above its row size as no filter, the same law and draws as top_k 0.
    SamplingParams p, unfiltered;
    std::vector<TokenLogprob> want;
    s.distribution(row, unfiltered, h, 1, want);
    for(const std::uint32_t k:{4u, 5u, 8u, std::numeric_limits<std::uint32_t>::max()}) {
        p.top_k = k;
        s.distribution(row, p, h, 1, law);
        require(law.size() == want.size() && law.size() == 4, "top_k past the valid vocabulary keeps every token");
        for(std::size_t i = 0; i < law.size(); ++i)
            require(law[i].token == want[i].token && law[i].logprob == want[i].logprob, "top_k past the row is no filter");
        for(std::uint64_t seed = 0; seed < 20; ++seed) {
            p.seed = unfiltered.seed = seed;
            require(s.sample(row, p, h, 1).token == s.sample(row, unfiltered, h, 1).token, "top_k past the row draws alike");
        }
    }
    p = {};
    p.logit_bias = {{1, 1.f}, {1, 2.f}};
    rejects<std::invalid_argument>([&] { s.sample(row, p, h); }, "logit_bias tokens must be unique");
    p = {};
    p.min_tokens = 1; p.suppress_until_min = {-4};
    rejects<std::invalid_argument>([&] { s.distribution(row, p, h, 0, law); }, "suppress_until_min tokens must be nonnegative");
    // A moved-from sampler keeps vocabulary 0 and rejects rows; the destination works.
    Sampler moved(std::move(s));
    require(s.vocabulary() == 0 && moved.vocabulary() == 4, "moves transfer the scratch");
    rejects<std::invalid_argument>([&] { s.sample(row, SamplingParams{}, h); }, "logits size must equal the sampler vocabulary");
    require(moved.sample(row, SamplingParams{.temperature = 0}, h).token == 3, "a moved sampler samples");
    Sampler assigned(2);
    assigned = std::move(moved);
    require(assigned.vocabulary() == 4 && moved.vocabulary() == 0, "move assignment transfers the scratch");
    std::vector<float> out(3), one_out(1);
    rejects<std::invalid_argument>([&] { log_softmax(row, out); }, "log_softmax output size must equal logits");
    std::vector<float> buffer{0.f, 1.f, 2.f, 3.f, 4.f};
    rejects<std::invalid_argument>([&] { log_softmax(std::span(buffer).first(4), std::span(buffer).last(4)); },
                                   "log_softmax output overlaps logits");
    rejects<std::invalid_argument>([&] { top_logprobs(row, big); }, "top logprobs exceed the logits row");
    const std::vector<Token> ids{0, 1}, outside{4}, negative{-1};
    rejects<std::invalid_argument>([&] { query_logprobs(row, ids, one_out); }, "query output size must equal ids");
    rejects<std::invalid_argument>([&] { query_logprobs(row, outside, one_out); }, "query token outside the logits row");
    rejects<std::invalid_argument>([&] { query_logprobs(row, negative, one_out); }, "query token outside the logits row");
    // Query outputs that overlap the row would corrupt later reads; adjacent ranges are fine.
    std::vector<float> joined{0.f, 1.f, 2.f, 3.f, 7.f, 7.f};
    const std::vector<Token> reversed{1, 0};
    const auto logits_part = std::span(joined).first(4);
    rejects<std::invalid_argument>([&] { query_logprobs(logits_part, reversed, std::span(joined).first(2)); },
                                   "query output overlaps logits");
    rejects<std::invalid_argument>([&] { query_logprobs(logits_part, reversed, std::span(joined).subspan(3, 2)); },
                                   "query output overlaps logits");
    std::vector<float> separate(2);
    query_logprobs(logits_part, reversed, separate);
    query_logprobs(logits_part, reversed, std::span(joined).last(2));
    require(joined[4] == separate[0] && joined[5] == separate[1] && joined[0] == 0.f && joined[1] == 1.f,
            "an adjacent query output is accepted");
    query_logprobs(logits_part, {}, std::span(joined).subspan(1, 0));
}
void allocation() {
    constexpr std::uint32_t V = 4096;
    std::mt19937_64 rng(4);
    const auto row = random_row(rng, V, 2.f, false);
    Sampler s(V);
    const std::vector<Token> prompt{1, 2, 3, 4, 5, 6}, committed{7, 8, 9}, staged{10, 11, 12, 7};
    auto h = make_history(prompt, true, committed, staged);
    std::vector<TokenLogprob> top(8), law;
    law.reserve(V);
    SamplingParams plain, greedy, filtered, modified;
    greedy.temperature = 0;
    filtered.temperature = 0.7; filtered.top_k = 50; filtered.top_p = 0.9; filtered.min_p = 0.01;
    modified.temperature = 1.2; modified.top_p = 0.8; modified.repetition_penalty = 1.2; modified.frequency_penalty = 0.3;
    modified.logit_bias = {{3, minus_inf}, {9, 1.f}, {100, -2.f}}; modified.min_tokens = 6; modified.suppress_until_min = {0, 1};
    for(const auto* p:{&plain, &greedy, &filtered, &modified}) {
        s.sample(row, *p, h, 2, top);
        const std::size_t before = allocation_count;
        for(std::size_t column = 0; column <= staged.size(); ++column) {
            s.sample(row, *p, h, column, top);
            s.distribution(row, *p, h, column, law);
        }
        require(allocation_count == before, "sample and distribution never allocate");
    }
    h.rollback();
    h.stage(staged);
    const std::size_t before = allocation_count;
    h.commit(3);
    h.rollback();
    require(allocation_count == before, "commit and rollback never allocate");
}
void scale() {
    constexpr std::uint32_t V = 250000;
    std::mt19937_64 rng(250);
    const auto row = random_row(rng, V, 3.f, false);
    Sampler s(V);
    require(s.scratch_bytes() == Sampler::scratch_bytes(V) && s.scratch_bytes() < std::size_t(V)*40,
            "scratch is O(vocabulary) and reported up front");
    std::vector<Token> outputs;
    for(int i = 0; i < 2000; ++i) outputs.push_back(Token(rng() % V));
    auto h = make_history(outputs, true, outputs, {});
    SamplingParams greedy, nucleus, top_k, penalized;
    greedy.temperature = 0;
    nucleus.temperature = 0.8; nucleus.top_p = 0.95;
    top_k.top_k = 50;
    penalized.temperature = 0.7; penalized.repetition_penalty = 1.2; penalized.frequency_penalty = 0.5; penalized.min_p = 0.01;
    std::vector<TokenLogprob> top(20);
#ifdef NDEBUG
    constexpr int samples = 40;
#else
    constexpr int samples = 3;
#endif
    for(const auto& [name, p]:{std::pair{"greedy", greedy}, std::pair{"temperature", SamplingParams{}},
                               std::pair{"top_p", nucleus}, std::pair{"top_k", top_k}, std::pair{"penalties", penalized}}) {
        const auto start = std::chrono::steady_clock::now();
        for(int i = 0; i < samples; ++i) {
            const Token token = s.sample(row, p, h, 0, top).token;
            h.stage(one(token));
            h.commit(1);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()/samples;
        std::cout << "crit core sampling 250k vocabulary " << name << ": " << ms << " ms per sample\n";
#ifdef NDEBUG
        require(ms < 60, "a 250k-vocabulary sample stays within the host budget");
#endif
    }
}
void score_rows() {
    constexpr std::uint32_t rows = 4, valid = 6;
    constexpr std::size_t stride = 9;
    std::mt19937_64 rng(6);
    std::vector<float> logits(stride*(rows - 1) + valid + 2);
    for(auto& x:logits) x = std::round(std::normal_distribution<float>(0, 2)(rng)*2)/2;
    for(std::uint32_t r = 0; r < rows; ++r) {
        for(std::size_t i = valid; i < stride && r*stride + i < logits.size(); ++i)
            logits[r*stride + i] = i == valid ? 1000.f : std::nanf("");   // padding is never read
        logits[r*stride + 1] = minus_inf;
    }
    const std::vector<Token> targets{0, 5, 1, 3}, queries{1, 2, 5, 5, 0, 4, 3, 3};
    const auto scores = score_logit_rows({logits, stride, rows, valid}, targets, queries, 2, 4, true);
    require(scores.target.size() == rows && scores.queries.size() == 8 && scores.argmax.size() == rows &&
            scores.top.size() == rows*4, "score shapes follow rows, per_target and top_k");
    for(std::uint32_t r = 0; r < rows; ++r) {
        const std::vector<float> row(logits.begin() + std::ptrdiff_t(r*stride), logits.begin() + std::ptrdiff_t(r*stride + valid));
        const Real lse = oracle_lse(row);
        const auto lp = [&](Token id) { return row[std::size_t(id)] == minus_inf ? -real_inf : Real(row[std::size_t(id)]) - lse; };
        require(close(scores.target[r], lp(targets[r])), "targets are raw log-probabilities over the valid row");
        for(std::size_t q = 0; q < 2; ++q) require(close(scores.queries[r*2 + q], lp(queries[r*2 + q])), "queries match");
        std::vector<TokenLogprob> top(4);
        top_logprobs(row, top);
        require(scores.argmax[r] == top[0].token, "argmax is the first ranked token of the valid row");
        for(std::size_t k = 0; k < 4; ++k)
            require(scores.top[r*4 + k].token == top[k].token && scores.top[r*4 + k].logprob == top[k].logprob, "top rows match");
    }
    require(scores.target[2] == minus_inf, "a masked target yields -inf");
    const auto lean = score_logit_rows({logits, stride, rows, valid}, targets, {}, 0, 0, false);
    require(lean.target == scores.target && lean.queries.empty() && lean.argmax.empty() && lean.top.empty(), "optional rows stay empty");
    const auto none = score_logit_rows({{}, 1, 0, 1}, {}, {}, 3, 1, true);
    require(none.target.empty() && none.queries.empty() && none.argmax.empty() && none.top.empty(), "zero rows give empty scores");
    const LogitRows good{logits, stride, rows, valid};
    const std::vector<Token> wide{0, 6, 1, 3}, few{0, 1};
    rejects<std::invalid_argument>([&] { score_logit_rows({logits, stride, rows, 0}, targets, {}, 0, 0, false); }, "valid vocabulary must be positive");
    rejects<std::invalid_argument>([] { score_logit_rows({{}, 2147483649u, 0, 2147483649u}, {}, {}, 0, 0, false); },
                                   "valid vocabulary exceeds the token id range");
    rejects<std::invalid_argument>([&] { score_logit_rows({logits, 5, rows, valid}, targets, {}, 0, 0, false); }, "logit row stride is smaller than the valid vocabulary");
    rejects<std::invalid_argument>([&] { score_logit_rows(good, few, {}, 0, 0, false); }, "targets must hold one token per row");
    rejects<std::invalid_argument>([&] { score_logit_rows({std::span(logits).first(stride*3 + valid - 1), stride, rows, valid}, targets, {}, 0, 0, false); },
                                   "logit rows exceed the logits span");
    rejects<std::invalid_argument>([&] { score_logit_rows({logits, std::numeric_limits<std::size_t>::max(), rows, valid}, targets, {}, 0, 0, false); },
                                   "logit rows exceed the logits span");
    rejects<std::invalid_argument>([&] { score_logit_rows(good, targets, few, 2, 0, false); }, "queries must hold per_target ids per row");
    rejects<std::invalid_argument>([&] { score_logit_rows(good, targets, few, 0, 0, false); }, "queries must hold per_target ids per row");
    rejects<std::invalid_argument>([&] { score_logit_rows(good, targets, {}, 0, valid + 1, false); }, "top_k exceeds the valid vocabulary");
    rejects<std::invalid_argument>([&] { score_logit_rows(good, wide, {}, 0, 0, false); }, "target token outside the valid vocabulary");
    const std::vector<Token> bad_queries{0, 1, 2, 3, 4, 5, 6, 0};
    rejects<std::invalid_argument>([&] { score_logit_rows(good, targets, bad_queries, 2, 0, false); }, "query token outside the valid vocabulary");
    std::vector<float> masked_row(logits);
    for(std::size_t i = 0; i < valid; ++i) masked_row[stride + i] = minus_inf;
    rejects<std::invalid_argument>([&] { score_logit_rows({masked_row, stride, rows, valid}, targets, {}, 0, 0, false); }, "no selectable token");
}
// Padding stop tokens: an adapter passes the stop tokens the runtime accepted (declared vocabulary 64) as
// suppress_until_min to a sampler over the valid vocabulary (60). Ids at or past the row are ignored, so {3, 61}
// resolves and samples exactly like {3}, with or without min_tokens.
void suppress_padding() {
    constexpr std::uint32_t valid = 60;
    SamplingParams defaults, reference_defaults;
    defaults.suppress_until_min = {3, 61};
    reference_defaults.suppress_until_min = {3};
    const auto unset = resolve_sampling(SamplingOptions{}, defaults, valid);
    require(unset.min_tokens == 0 && unset.suppress_until_min == defaults.suppress_until_min, "unused padding list resolves");
    std::mt19937_64 rng(61);
    auto row = random_row(rng, valid, 1.f, false);
    row[3] = 6.f;   // the stop token dominates unless suppressed
    Sampler s(valid);
    std::vector<TokenLogprob> law, reference_law, top(5), reference_top(5);
    for(const double temperature:{0.0, 0.8, 1.0}) {
        SamplingOptions o;
        o.min_tokens = 2; o.temperature = temperature;
        const auto padded = resolve_sampling(o, defaults, valid), reference = resolve_sampling(o, reference_defaults, valid);
        require(padded.min_tokens == 2, "min_tokens resolves with a padding suppress id");
        for(std::size_t outputs = 0; outputs < 4; ++outputs) {
            const auto h = make_history({1, 2}, true, std::vector<Token>(outputs, 7), {});
            s.distribution(row, padded, h, 0, law);
            s.distribution(row, reference, h, 0, reference_law);
            require(law.size() == reference_law.size(), "padding suppress id changes the law");
            for(std::size_t i = 0; i < law.size(); ++i)
                require(law[i].token == reference_law[i].token && law[i].logprob == reference_law[i].logprob,
                        "padding suppress id changes the law");
            const bool suppressed = std::none_of(law.begin(), law.end(), [](const TokenLogprob& e) { return e.token == 3; });
            require(suppressed == (outputs < 2), "the in-row stop is suppressed until min_tokens");
            for(std::uint64_t seed = 0; seed < 16; ++seed) {
                SamplingParams a = padded, b = reference;
                a.seed = b.seed = seed;
                const auto x = s.sample(row, a, h, 0, top), y = s.sample(row, b, h, 0, reference_top);
                require(x.token == y.token && x.logprob == y.logprob, "padding suppress id changes the draw");
                for(std::size_t r = 0; r < top.size(); ++r)
                    require(top[r].token == reference_top[r].token && top[r].logprob == reference_top[r].logprob,
                            "padding suppress id changes the alternatives");
            }
        }
    }
    // Only padding ids in the list: nothing in the row is suppressed.
    SamplingParams only_padding;
    only_padding.temperature = 0; only_padding.min_tokens = 5; only_padding.suppress_until_min = {60, 63, 2147483647};
    require(s.sample(row, only_padding, TokenHistory({})).token == 3, "padding-only list suppresses nothing");
}
// Large-magnitude rows: the raw maximum and log(sum) stay separate, so log(sum) is not absorbed into the maximum.
// Each row has three tied entries at b and one far below; the tied log-probabilities are exactly -log(3) and the
// probabilities sum to 1 (folding max + log(sum) first gave -2 at 1e16 and 0 at 1e20).
void magnitude() {
    const float third = static_cast<float>(-std::log(3.0));
    for(const float b:{1e10f, 1e12f, 1e16f, 1e20f, -1e20f, 1e38f}) {
        const float low = b > 0 ? -5.f : -1e38f;
        const std::vector<float> row{b, low, b, b};
        const Real expected_low = Real(low) - Real(b) - std::log(Real(3));
        std::vector<float> lp(4);
        log_softmax(row, lp);
        require(lp[0] == third && lp[2] == third && lp[3] == third, "tied large logits give -log(3)");
        require(close(lp[1], expected_low), "the low entry keeps its distance from the maximum");
        Real total = 0;
        for(const float x:lp) total += std::exp(Real(x));
        require(near(total, 1, 1e-6L), "large-magnitude probabilities sum to 1");
        std::vector<TokenLogprob> top(4);
        top_logprobs(row, top);
        const std::vector<Token> ranked{0, 2, 3, 1};
        for(std::size_t r = 0; r < 4; ++r)
            require(top[r].token == ranked[r] && top[r].logprob == lp[std::size_t(ranked[r])], "large-magnitude alternatives");
        const std::vector<Token> ids{3, 1, 0};
        std::vector<float> q(3);
        query_logprobs(row, ids, q);
        require(q[0] == third && q[1] == lp[1] && q[2] == third, "large-magnitude queries");
        const std::vector<Token> target{2}, queries{1, 3};
        const auto scores = score_logit_rows({row, 4, 1, 4}, target, queries, 2, 2, true);
        require(scores.target[0] == third && scores.queries[0] == lp[1] && scores.queries[1] == third && scores.argmax[0] == 0 &&
                scores.top[0].token == 0 && scores.top[1].token == 2 && scores.top[1].logprob == third, "large-magnitude scoring");
        // Every Sampler path: the T = 1 shortcut, the general path (T != 1, top_k, penalties), plain greedy and
        // greedy after bias, suppression or penalties modified the row.
        Sampler s(4);
        SamplingParams unit, warm, truncated, penalized, greedy, greedy_biased, greedy_suppressed, greedy_penalized;
        warm.temperature = 0.6;
        truncated.top_k = 2;
        penalized.repetition_penalty = 1.5;
        greedy.temperature = 0;
        greedy_biased.temperature = 0;
        greedy_biased.logit_bias = {{1, -1.f}};
        greedy_suppressed.temperature = 0;
        greedy_suppressed.min_tokens = 1;
        greedy_suppressed.suppress_until_min = {1};
        greedy_penalized.temperature = 0;
        greedy_penalized.repetition_penalty = 1.5;
        const std::vector<Token> prompt{1};
        for(const auto* p:{&unit, &warm, &truncated, &penalized, &greedy, &greedy_biased, &greedy_suppressed, &greedy_penalized})
            for(std::uint64_t seed = 0; seed < 8; ++seed) {
                SamplingParams params = *p;
                params.seed = seed;
                std::vector<TokenLogprob> alternatives(2);
                const auto choice = s.sample(row, params, TokenHistory(prompt), 0, alternatives);
                require(choice.token != 1 && choice.logprob == third, "large-magnitude choice logprob");
                require(alternatives[0].token == 0 && alternatives[0].logprob == third && alternatives[1].logprob == third,
                        "large-magnitude sample alternatives");
            }
        std::vector<TokenLogprob> law;
        s.distribution(row, SamplingParams{}, TokenHistory({}), 0, law);
        Real law_total = 0;
        for(const auto& entry:law) law_total += std::exp(Real(entry.logprob));
        require(law.size() == 3 && near(law_total, 1, 1e-6L), "large-magnitude law sums to 1");
    }
}
void accept() {
    const auto check = [](std::vector<Token> drafts, std::vector<Token> predictions, std::uint32_t remaining,
                          std::vector<Token> stops, Acceptance want, const char* message) {
        const auto got = accept_greedy(drafts, predictions, remaining, stops);
        require(got.emitted == want.emitted && got.matched == want.matched && got.stopped == want.stopped, message);
    };
    check({5, 6, 7}, {5, 6, 7, 8}, 10, {}, {4, 3, false}, "all drafts match and the bonus follows");
    check({5, 6, 7}, {5, 9, 7, 8}, 10, {}, {2, 1, false}, "a mismatch emits the correction");
    check({5}, {6, 7}, 10, {}, {1, 0, false}, "a first mismatch emits only the correction");
    check({}, {4}, 10, {}, {1, 0, false}, "no drafts is plain decoding");
    check({5, 2, 7}, {5, 2, 7, 8}, 10, {2}, {2, 2, true}, "a stop inside the drafts ends the round");
    check({5, 6}, {5, 2, 9}, 10, {2}, {2, 1, true}, "a stop as the correction ends the round");
    check({5}, {5, 2}, 10, {2, 3}, {2, 1, true}, "a stop as the bonus ends the round");
    check({5, 6, 7}, {5, 6, 7, 8}, 2, {}, {2, 2, false}, "the remaining budget caps matched drafts");
    check({5, 6, 7}, {5, 6, 7, 8}, 3, {}, {3, 3, false}, "the remaining budget can exclude the bonus");
    check({5, 6, 7}, {5, 9, 7, 8}, 1, {}, {1, 1, false}, "a budget of one emits the first match");
    check({5, 6}, {5, 6, 7}, 0, {}, {0, 0, false}, "a zero budget emits nothing");
    rejects<std::invalid_argument>([] { accept_greedy(std::vector<Token>{1, 2}, std::vector<Token>{1, 2}, 4, {}); },
                                   "predictions must hold one more token than drafts");
    rejects<std::invalid_argument>([] { accept_greedy(std::vector<Token>{-1}, std::vector<Token>{1, 2}, 4, {}); }, "token ids must be nonnegative");
    rejects<std::invalid_argument>([] { accept_greedy(std::vector<Token>{1}, std::vector<Token>{1, 2}, 4, std::vector<Token>{-3}); },
                                   "token ids must be nonnegative");
    // Randomized oracle of the accept loop.
    std::mt19937_64 rng(12);
    for(int trial = 0; trial < 5000; ++trial) {
        std::vector<Token> drafts(rng() % 6), predictions, stops(rng() % 3);
        for(auto& t:drafts) t = Token(rng() % 4);
        for(std::size_t i = 0; i <= drafts.size(); ++i) predictions.push_back(i < drafts.size() && rng() % 3 ? drafts[i] : Token(rng() % 4));
        for(auto& t:stops) t = Token(rng() % 4);
        const std::uint32_t remaining = std::uint32_t(rng() % 8);
        const auto is_stop = [&](Token t) { return std::find(stops.begin(), stops.end(), t) != stops.end(); };
        std::vector<Token> emitted;
        std::size_t matched = 0;
        while(matched < drafts.size() && emitted.size() < remaining && drafts[matched] == predictions[matched]) {
            emitted.push_back(drafts[matched++]);
            if(is_stop(emitted.back())) break;
        }
        if(emitted.size() < remaining && (emitted.empty() || !is_stop(emitted.back()))) emitted.push_back(predictions[matched]);
        const auto got = accept_greedy(drafts, predictions, remaining, stops);
        require(got.emitted == emitted.size() && got.matched == matched &&
                got.stopped == (!emitted.empty() && is_stop(emitted.back())), "acceptance matches the oracle loop");
        require(std::equal(emitted.begin(), emitted.end(), predictions.begin()), "emitted tokens are a prediction prefix");
    }
}
}

void constraint() {
    // An allowed set equals banning every other id with a -inf bias: same law under every filter, same draws, and
    // the reported log-probabilities stay raw over the whole row.
    constexpr std::uint32_t V = 64;
    std::mt19937_64 rng(23);
    auto row = random_row(rng, V, 2.f, false);
    row[40] = row[41] = 9.f;                                      // a tie inside the allowed set
    const std::vector<Token> allowed{3, 7, 8, 20, 40, 41, 63};
    Sampler s(V);
    TokenHistory h({});
    SamplingParams plain, greedy, filtered, wide, penalized;
    greedy.temperature = 0;
    filtered.temperature = 0.7; filtered.top_k = 3; filtered.top_p = 0.8; filtered.min_p = 0.05;
    wide.top_k = 50;                                              // above the allowed count keeps the whole set
    penalized.repetition_penalty = 1.3; penalized.logit_bias = {{8, 2.f}, {63, minus_inf}};
    const std::vector<Token> prompt{7, 8};
    std::vector<float> raw(V);
    log_softmax(row, raw);
    for(const auto* p:{&plain, &greedy, &filtered, &wide, &penalized}) {
        SamplingParams banned = *p;
        for(Token t = 0; t < Token(V); ++t)
            if(!std::binary_search(allowed.begin(), allowed.end(), t)) banned.logit_bias.push_back({t, minus_inf});
        std::sort(banned.logit_bias.begin(), banned.logit_bias.end(), [](auto a, auto b) { return a.token < b.token; });
        banned.logit_bias.erase(std::unique(banned.logit_bias.begin(), banned.logit_bias.end(),
                                            [](auto a, auto b) { return a.token == b.token; }), banned.logit_bias.end());
        const TokenHistory history(prompt);
        std::vector<TokenLogprob> got, want;
        s.distribution(row, AllowedTokens{allowed}, *p, history, 0, got);
        s.distribution(row, banned, history, 0, want);
        require(got.size() == want.size(), "constrained law keeps the allowed support");
        for(std::size_t i = 0; i < got.size(); ++i)
            require(got[i].token == want[i].token && got[i].logprob == want[i].logprob, "constraint equals -inf bans");
        for(std::uint64_t seed = 0; seed < 64; ++seed) {
            SamplingParams a = *p, b = banned;
            a.seed = b.seed = seed;
            const auto c = s.sample(row, AllowedTokens{allowed}, a, history);
            require(std::binary_search(allowed.begin(), allowed.end(), c.token), "draws stay inside the allowed set");
            require(c.token == s.sample(row, b, history).token, "constrained draws equal banned draws");
            require(c.logprob == raw[std::size_t(c.token)], "logprob stays raw over the row");
        }
    }
    require(s.sample(row, AllowedTokens{allowed}, greedy, h).token == 40, "greedy ties go to the lowest allowed id");
    const std::vector<Token> only{5};
    std::vector<TokenLogprob> top(3);
    const auto forced = s.sample(row, AllowedTokens{only}, plain, h, 0, top);
    std::vector<TokenLogprob> raw_top(3);
    top_logprobs(row, raw_top);
    require(forced.token == 5 && top[0].token == raw_top[0].token && top[0].logprob == raw_top[0].logprob,
            "alternatives stay raw over the row");
    require(forced.logprob == raw[5], "a forced token reports its raw log-probability");
    const std::vector<Token> empty, unsorted{4, 3}, duplicate{3, 3}, outside{3, Token(V)}, negative{-1, 2};
    for(const auto* ids:{&empty, &unsorted, &duplicate, &outside, &negative}) {
        rejects<std::invalid_argument>([&] { s.sample(row, AllowedTokens{*ids}, plain, h); }, "allowed tokens must be");
        std::vector<TokenLogprob> out;
        rejects<std::invalid_argument>([&] { s.distribution(row, AllowedTokens{*ids}, greedy, h, 0, out); },
                                       "allowed tokens must be");
    }
    SamplingParams suppressed = plain;
    suppressed.min_tokens = 2; suppressed.suppress_until_min = {5};
    rejects<std::invalid_argument>([&] { s.sample(row, AllowedTokens{only}, suppressed, h); }, "no selectable token");
    auto masked = row;
    masked[5] = minus_inf;
    rejects<std::invalid_argument>([&] { s.sample(masked, AllowedTokens{only}, plain, h); }, "no selectable token");
    std::vector<TokenLogprob> law;
    law.reserve(V);
    s.sample(row, AllowedTokens{allowed}, filtered, h, 0, top);
    const std::size_t before = allocation_count;
    for(const auto* p:{&plain, &greedy, &filtered, &penalized}) {
        s.sample(row, AllowedTokens{allowed}, *p, h, 0, top);
        s.distribution(row, AllowedTokens{allowed}, *p, h, 0, law);
    }
    require(allocation_count == before, "constrained sample and distribution never allocate");
}
int main(int argc, char** argv) {
    try {
        const std::string_view name = argc > 1 ? argv[1] : "all";
        const std::vector<std::pair<std::string_view, std::function<void()>>> cases{
            {"sampling_philox", philox}, {"sampling_uniform", uniform_draws}, {"sampling_validate", validation},
            {"sampling_history", history}, {"sampling_copy", copies}, {"sampling_greedy", greedy}, {"sampling_law", law},
            {"sampling_frequency", frequency}, {"sampling_redo", redo}, {"sampling_penalties", penalties},
            {"sampling_bias", bias}, {"sampling_min_tokens", min_tokens}, {"sampling_logprobs", logprobs},
            {"sampling_infinity", infinity}, {"sampling_errors", errors}, {"sampling_allocation", allocation},
            {"sampling_scale", scale}, {"sampling_score_rows", score_rows}, {"sampling_accept", accept},
            {"sampling_suppress_padding", suppress_padding}, {"sampling_magnitude", magnitude},
            {"sampling_constraint", constraint}};
        bool found = false;
        for(const auto& [case_name, run]:cases) if(name == "all" || name == case_name) {
            run();
            found = true;
            std::cout << "crit core sampling passed: " << case_name << '\n';
        }
        require(found, "unknown test case");
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
