// Core's whole sampling law on the device against crit::core::Sampler: full-vocabulary rows without a
// candidate cap, bias, min_tokens suppression and penalties over committed plus staged outputs, min_p, the
// dense law, speculative verification equal to Core's own draws (greedy drafts) or the rejection rule (sampled
// drafts), first-token frequencies equal to the target law for both, and device commits of counts and positions.
#include "sampling.h"

#include <crit/sampling.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace cc = crit::core;
namespace ts = crit::techniques::token_sampling;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }

template<class T> struct Device {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Device(std::size_t count) : n(count) { ok(cudaMalloc(&p, sizeof(T) * std::max<std::size_t>(count, 1)), "malloc"); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) { put(v); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void put(const std::vector<T>& v) { ok(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice), "upload"); }
    std::vector<T> get() const {
        std::vector<T> v(n);
        ok(cudaMemcpy(v.data(), p, sizeof(T) * n, cudaMemcpyDeviceToHost), "download");
        return v;
    }
};

// One request's processing on both sides: the device Request and Core's SamplingParams plus history.
struct Case {
    int vocab;
    cc::SamplingParams rules;
    std::vector<cc::Token> prompt, outputs;
    std::vector<std::int32_t> bias_ids;
    std::vector<float> bias_values;
    std::vector<std::int32_t> suppress;
    std::vector<std::int32_t> counts;
    std::vector<std::uint8_t> seen;

    explicit Case(int v) : vocab(v), counts(std::size_t(v), 0), seen(std::size_t(v), 0) {}
    void finish() {
        std::map<std::int32_t, float> merged;
        for(std::size_t i = 0; i < bias_ids.size(); ++i) merged[bias_ids[i]] += bias_values[i];
        bias_ids.clear();
        bias_values.clear();
        rules.logit_bias.clear();
        for(auto [t, b] : merged) { bias_ids.push_back(t); bias_values.push_back(b); rules.logit_bias.push_back({t, b}); }
        rules.suppress_until_min = suppress;
        std::fill(counts.begin(), counts.end(), 0);
        for(auto t : outputs) ++counts[std::size_t(t)];
        std::fill(seen.begin(), seen.end(), 0);
        for(auto t : prompt) seen[std::size_t(t)] = 1;
    }
    cc::TokenHistory history() const {
        cc::TokenHistory h(prompt);
        if(!outputs.empty()) { h.stage(outputs); h.commit(outputs.size()); }
        return h;
    }
};

// Device buffers holding a Case's Request.
struct DeviceCase {
    Device<std::int32_t> bias_ids, suppress, counts;
    Device<float> bias_values;
    Device<std::uint8_t> seen;
    ts::Request request;
    explicit DeviceCase(const Case& c)
        : bias_ids(c.bias_ids), suppress(c.suppress), counts(c.counts), bias_values(c.bias_values), seen(c.seen) {
        request.temperature = c.rules.temperature;
        request.top_p = c.rules.top_p;
        request.min_p = c.rules.min_p;
        request.top_k = std::int32_t(c.rules.top_k);
        request.repetition = c.rules.repetition_penalty;
        request.frequency = c.rules.frequency_penalty;
        request.presence = c.rules.presence_penalty;
        request.min_tokens = c.rules.min_tokens;
        request.stream = c.rules.stream;
        request.seed = c.rules.seed;
        request.position = c.outputs.size();
        request.bias_ids = bias_ids.p;
        request.bias_values = bias_values.p;
        request.bias_count = std::uint32_t(c.bias_ids.size());
        request.suppress_ids = suppress.p;
        request.suppress_count = std::uint32_t(c.suppress.size());
        request.counts = counts.p;
        request.prompt_seen = seen.p;
    }
};

// Core's law at one column as a dense vector, plus the rank order of the kept tokens.
std::vector<double> core_law(cc::Sampler& host, std::span<const float> row, const Case& c, std::span<const cc::Token> staged,
                             std::vector<cc::Token>* order = nullptr) {
    auto h = c.history();
    if(!staged.empty()) h.stage(staged);
    std::vector<cc::TokenLogprob> law;
    host.distribution(row, c.rules, h, staged.size(), law);
    std::vector<double> p(std::size_t(c.vocab), 0.);
    if(order) order->clear();
    for(const auto& e : law) {
        p[std::size_t(e.token)] = std::exp(double(e.logprob));
        if(order) order->push_back(e.token);
    }
    return p;
}

// Rank-order CDF walk over `weights` (normalized or not); returns the token and the distance of the
// draw to the nearest cumulative boundary in normalized units.
std::pair<cc::Token, double> walk(const std::vector<cc::Token>& order, const std::vector<double>& weights, double u) {
    double total = 0.;
    for(auto t : order) total += weights[std::size_t(t)];
    const double target = u * total;
    double c = 0., margin = 1.;
    cc::Token chosen = order.empty() ? -1 : order.back();
    bool found = false;
    for(auto t : order) {
        const double w = weights[std::size_t(t)];
        if(!(w > 0.)) continue;
        c += w;
        margin = std::min(margin, std::abs(target - c) / total);
        if(!found && target < c) { chosen = t; found = true; }
    }
    return {chosen, margin};
}
} // namespace

int main() {
    try {
        std::mt19937 rng(23);
        std::normal_distribution<float> normal(0.f, 1.f);
        cc::Sampler host50(50000);

        // 1. Full vocabulary: a nearly flat row, temperature 1, no filters. The two-stage path used to keep
        //    64 candidates; every draw must now come from Core's whole law, most of them beyond rank 64.
        {
            constexpr int V = 50000;
            std::vector<float> row(V);
            for(auto& v : row) v = 0.05f * normal(rng);
            Device<float> logits(row);
            Device<unsigned char> scratch(ts::select_scratch_bytes(V, 1));
            Device<std::int32_t> out(1);
            Device<ts::Params> params(1);
            ts::Params p;
            p.temperature = 1.f;
            p.seed = 4242;
            params.put({p});
            std::vector<cc::Token> order(V);
            for(int i = 0; i < V; ++i) order[std::size_t(i)] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return row[std::size_t(a)] == row[std::size_t(b)] ? a < b : row[std::size_t(a)] > row[std::size_t(b)]; });
            std::vector<int> rank(V);
            for(int r = 0; r < V; ++r) rank[std::size_t(order[std::size_t(r)])] = r;
            cc::SamplingParams rules;
            rules.temperature = 1.0;
            rules.seed = p.seed;
            const cc::TokenHistory h(std::span<const cc::Token>{});
            std::vector<cc::TokenLogprob> law;
            host50.distribution(row, rules, h, 0, law);
            require(law.size() == std::size_t(V), "Core keeps the whole vocabulary");
            // Cumulative interval of each token in Core's rank order: a device draw must land in its token's
            // interval up to rounding, since 50,000 narrow bins put every draw near some boundary.
            std::vector<double> lo(V), hi(V);
            double cum = 0.;
            for(const auto& e : law) {
                lo[std::size_t(e.token)] = cum;
                cum += std::exp(double(e.logprob));
                hi[std::size_t(e.token)] = cum;
            }
            int beyond = 0, exact = 0;
            for(int i = 0; i < 400; ++i) {
                ok(ts::enqueue_select(logits.p, V, 1, V, params.p, scratch.p, out.p, nullptr), "select");
                const int got = out.get()[0];
                require(got >= 0 && got < V, "draw outside the vocabulary");
                const double u = cc::uniform(p.seed, std::uint64_t(i), 0) * cum;
                require(u >= lo[std::size_t(got)] - 1e-5 && u < hi[std::size_t(got)] + 1e-5, "draw outside its token's interval of Core's law");
                exact += u >= lo[std::size_t(got)] && u < hi[std::size_t(got)];
                beyond += rank[std::size_t(got)] >= ts::kMaxTopK;
            }
            require(params.get()[0].position == 400, "the full law advances the position");
            require(beyond > 300, "full-vocabulary draws stay inside the top 64");
            std::cout << "full vocabulary: " << beyond << " of 400 draws beyond rank 64, " << exact << " inside Core's exact interval\n";
        }

        // 1b. Ties at the top-k boundary on a 151,936-token row: logits on a 0.25 grid, so every boundary k falls inside
        //     a run of equal values and ties must resolve by ascending id as Core's do. For k in {40, 64, 65, 1024, 4096,
        //     the vocabulary, 0}, with and without penalties, the device's kept set equals Core's id for id, its law
        //     matches, and ten launches give the same law and draw bytes (no atomics, fixed-order sums).
        {
            constexpr int V = 151936;
            cc::Sampler host(V);
            std::vector<float> row(V);
            for(auto& v : row) v = std::round(4.f * 1.5f * normal(rng)) / 4.f;
            {
                // The premise, checked: at every finite k the k-th and (k+1)-th ranked logits are equal.
                std::vector<int> ranked(V);
                for(int i = 0; i < V; ++i) ranked[std::size_t(i)] = i;
                std::sort(ranked.begin(), ranked.end(), [&](int a, int b) { return row[std::size_t(a)] == row[std::size_t(b)] ? a < b : row[std::size_t(a)] > row[std::size_t(b)]; });
                for(const int k : {40, 64, 65, 1024, 4096})
                    require(row[std::size_t(ranked[std::size_t(k - 1)])] == row[std::size_t(ranked[std::size_t(k)])], "a top-k boundary is not inside a tie run");
            }
            int checked = 0, draws = 0;
            for(const std::uint32_t k : {40u, 64u, 65u, 1024u, 4096u, std::uint32_t(V), 0u})
                for(const bool penalties : {false, true}) {
                    Case c(V);
                    c.rules.temperature = 0.8;
                    c.rules.top_k = k;
                    c.rules.seed = 4242;
                    if(penalties) {
                        c.rules.repetition_penalty = 1.1;
                        c.rules.frequency_penalty = 0.5;
                        c.rules.presence_penalty = 0.25;
                        // History on tokens of the top runs, so penalties move tokens across the boundary.
                        std::vector<cc::Token> top(V);
                        for(int i = 0; i < V; ++i) top[std::size_t(i)] = i;
                        std::partial_sort(top.begin(), top.begin() + 64, top.end(), [&](int a, int b) {
                            return row[std::size_t(a)] == row[std::size_t(b)] ? a < b : row[std::size_t(a)] > row[std::size_t(b)];
                        });
                        for(int i = 0; i < 64; i += 3) c.prompt.push_back(top[std::size_t(i)]);
                        for(int i = 0; i < 64; i += 5) c.outputs.push_back(top[std::size_t(i)]);
                    }
                    c.finish();
                    DeviceCase dc(c);
                    Device<ts::Request> request(1);
                    Device<float> logits(row), scratch(ts::scratch_floats(V, 1)), probs(V);
                    Device<std::int32_t> out(1);
                    ts::Rows rows;
                    rows.logits = logits.p; rows.stride = V; rows.vocab = V; rows.request = request.p; rows.scratch = scratch.p;
                    std::vector<float> first;
                    std::int32_t first_token = -2;
                    for(int run = 0; run < 10; ++run) {
                        request.put({dc.request});
                        ok(ts::enqueue_sample(rows, cc::DrawPurpose::sample, out.p, probs.p, nullptr), "sample");
                        const auto dense = probs.get();
                        const auto token = out.get()[0];
                        if(run == 0) { first = dense; first_token = token; continue; }
                        require(std::memcmp(dense.data(), first.data(), sizeof(float) * V) == 0 && token == first_token,
                                "a repeated launch changed the law or the draw");
                    }
                    std::vector<cc::Token> order;
                    const auto want = core_law(host, row, c, {}, &order);
                    // The draw equals Core's sampler away from FP32 rounding of a cumulative boundary.
                    const auto [walked, margin] = walk(order, want, cc::uniform(c.rules.seed, c.outputs.size(), c.rules.stream));
                    if(margin >= 1e-4) {
                        require(first_token == walked, "the device draw differs from Core's sampler at a tied boundary");
                        ++draws;
                    }
                    std::size_t kept = 0;
                    for(int t = 0; t < V; ++t) {
                        const double a = first[std::size_t(t)], b = want[std::size_t(t)];
                        require(std::isfinite(a), "a law entry is not finite");
                        require((a > 0.) == (b > 0.), "the device keeps a different token set than Core at a tied boundary");
                        require(std::abs(a - b) <= 2e-4 * std::max(b, 1e-3), "device law differs from Core's distribution");
                        kept += b > 0.;
                    }
                    require(k == 0 || k >= std::uint32_t(V) ? kept == std::size_t(V) : kept == k, "the kept set has the wrong size");
                    ++checked;
                }
            std::cout << "tied boundaries: " << checked << " (top_k, penalties) cases on " << V
                      << " tokens keep Core's set id for id, ten launches byte-identical, " << draws
                      << " draws equal Core's sampler\n";
        }

        // 1c. Core's rules as Core holds them (double): equal logits make every weight exactly 1, so the top-p boundary is
        //     an exact comparison of an integer count with top_p * n. 0.8, 0.1 and 0.3 are not floats; their float
        //     roundings lie above them and would keep one token more than Core. Values Core holds in double beyond the
        //     float range (a bias pushing a logit past FLT_MAX, a negative one past -FLT_MAX) stay finite and selectable.
        //     The device's kept set and every draw equal Core's.
        {
            constexpr int V = 1000;
            cc::Sampler host(V);
            int cases = 0;
            for(const auto& [top_p, n] : {std::pair{0.8, 5}, std::pair{0.1, 10}, std::pair{0.3, 10}})
                for(const bool select : {false, true}) {
                    std::vector<float> row(V, -INFINITY);
                    for(int i = 0; i < n; ++i) row[std::size_t(37 + 11 * i)] = 0.5f;
                    cc::SamplingParams rules;
                    rules.temperature = 1.0;
                    rules.top_p = top_p;
                    const cc::TokenHistory h(std::span<const cc::Token>{});
                    std::vector<cc::TokenLogprob> law;
                    host.distribution(row, rules, h, 0, law);
                    std::vector<double> want(V, 0.);
                    for(const auto& e : law) want[std::size_t(e.token)] = std::exp(double(e.logprob));
                    Device<float> logits(row), scratch(ts::scratch_floats(V, 8));
                    Device<std::int32_t> out(8);
                    std::vector<std::int32_t> tokens(8);
                    if(select) {
                        // enqueue_select with Params (top_k 0: the full law), eight positions in a row.
                        ts::Params p;
                        p.temperature = 1.0;
                        p.top_p = top_p;
                        p.seed = 31;
                        Device<ts::Params> params(std::vector<ts::Params>{p});
                        Device<unsigned char> select_scratch(ts::select_scratch_bytes(V, 1));
                        for(int i = 0; i < 8; ++i)
                            ok(ts::enqueue_select(logits.p, V, 1, V, params.p, select_scratch.p, out.p + i, nullptr), "select");
                        tokens = out.get();
                    } else {
                        ts::Request q;
                        q.temperature = rules.temperature;
                        q.top_p = rules.top_p;
                        q.seed = 31;
                        std::vector<ts::Request> reqs(8, q);
                        for(int i = 0; i < 8; ++i) reqs[std::size_t(i)].position = std::uint64_t(i);
                        Device<ts::Request> request(reqs);
                        std::vector<float> rep(std::size_t(8) * V);
                        for(int i = 0; i < 8; ++i) std::copy(row.begin(), row.end(), rep.begin() + std::ptrdiff_t(i) * V);
                        Device<float> repeated(rep), dense(std::size_t(8) * V);
                        ts::Rows rows;
                        rows.logits = repeated.p; rows.stride = V; rows.vocab = V; rows.requests = 8; rows.request = request.p;
                        rows.scratch = scratch.p;
                        ok(ts::enqueue_sample(rows, cc::DrawPurpose::sample, out.p, dense.p, nullptr), "sample");
                        tokens = out.get();
                        const auto got = dense.get();
                        for(int t = 0; t < V; ++t)
                            require((got[std::size_t(t)] > 0.f) == (want[std::size_t(t)] > 0.),
                                    "the device keeps a different top-p set than Core's double rule");
                    }
                    cc::TokenHistory history(std::span<const cc::Token>{});
                    for(int i = 0; i < 8; ++i) {
                        rules.seed = 31;
                        const auto expect = host.sample(row, rules, history).token;
                        require(tokens[std::size_t(i)] == expect, "a draw differs from Core's at an exact top-p boundary");
                        const cc::Token emitted = expect;
                        history.stage(std::span<const cc::Token>(&emitted, 1));
                        history.commit(1);
                    }
                    ++cases;
                }
            // Bias beyond the float range: Core holds 3e38 + 1e38 and -3e38 - 1e38 in double; on floats the first would
            // be +inf (a NaN law) and the second -inf (not selectable).
            {
                std::vector<float> row(V, -INFINITY);
                row[5] = 3e38f;
                row[9] = 3e38f;
                row[700] = -3e38f;
                for(const bool low : {false, true}) {
                    Case c(V);
                    c.rules.temperature = 0.9;
                    c.rules.seed = 8;
                    if(low) {
                        row[5] = row[9] = -INFINITY;
                        c.bias_ids = {700};
                        c.bias_values = {-1e38f};
                    } else {
                        c.bias_ids = {9};
                        c.bias_values = {1e38f};
                    }
                    c.finish();
                    DeviceCase dc(c);
                    Device<ts::Request> request(std::vector<ts::Request>{dc.request});
                    Device<float> logits(row), probs(V), scratch(ts::scratch_floats(V, 1));
                    Device<std::int32_t> out(1);
                    ts::Rows rows;
                    rows.logits = logits.p; rows.stride = V; rows.vocab = V; rows.request = request.p; rows.scratch = scratch.p;
                    ok(ts::enqueue_sample(rows, cc::DrawPurpose::sample, out.p, probs.p, nullptr), "sample");
                    const auto want = core_law(host, row, c, {});
                    const auto got = probs.get();
                    for(int t = 0; t < V; ++t)
                        require(std::abs(double(got[std::size_t(t)]) - want[std::size_t(t)]) <= 1e-6,
                                "a value beyond the float range changes the law");
                    require(out.get()[0] == host.sample(row, c.rules, c.history()).token, "a value beyond the float range changes the draw");
                    ++cases;
                }
            }
            std::cout << "double rules: " << cases << " cases keep Core's top-p sets, values beyond the float range and draws\n";
        }

        // 1d. The fast path (greedy rows and 1 <= top_k <= 64) runs Core's law in FP64 in Core's order on the merged
        //     candidates: draws equal crit::core::Sampler's exactly and laws to float rounding, on vocabularies of one
        //     partial stage block, several blocks and a partial last block, with and without bias, suppression and
        //     penalties over staged columns, and with the row's selectable count below top_k (Core's id-order sums).
        {
            struct Rule { double T, top_p, min_p; std::uint32_t top_k; bool processing; };
            const Rule rules[] = {{0., 1., 0., 0, false}, {0., 1., 0., 0, true}, {0.7, 1., 0., 1, true}, {1.0, 0.95, 0., 20, false},
                                  {0.7, 0.8, 0., 20, true}, {1.3, 0.9, 0.05, 64, true}, {1.0, 1., 0., 2, false},
                                  {0.9, 0.5, 0.02, 64, false}, {1.0, 0.95, 0., 40, true}};
            int rows_checked = 0, draws = 0, skipped = 0;
            for(const int V : {7, 100, 3 * 4096, 200003}) {
                cc::Sampler host(static_cast<std::uint32_t>(V));
                constexpr int columns = 3;
                int trial = 0;
                for(const auto& rule : rules) {
                    Case c(V);
                    c.rules.temperature = rule.T;
                    c.rules.top_p = rule.top_p;
                    c.rules.min_p = rule.min_p;
                    c.rules.top_k = rule.top_k;
                    c.rules.seed = 7000 + std::uint64_t(trial);
                    c.rules.stream = std::uint32_t(trial % 2);
                    std::vector<float> row(std::size_t(columns) * V);
                    for(auto& v : row) v = 2.5f * normal(rng);
                    std::vector<cc::Token> staged(columns - 1);
                    for(auto& t : staged) t = cc::Token(std::uniform_int_distribution<int>(0, V - 1)(rng));
                    if(rule.processing) {
                        c.rules.repetition_penalty = 1.2;
                        c.rules.frequency_penalty = 0.3;
                        c.rules.presence_penalty = 0.4;
                        c.rules.min_tokens = 3;
                        std::vector<cc::Token> hot(6);
                        for(auto& t : hot) t = cc::Token(std::uniform_int_distribution<int>(0, V - 1)(rng));
                        for(int col = 0; col < columns; ++col) row[std::size_t(col) * V + std::size_t(hot[0])] = 8.f;
                        c.prompt = {hot[0], hot[1], hot[2]};
                        c.outputs = {hot[0], hot[3]};
                        c.bias_ids = {hot[4], hot[5]};
                        c.bias_values = {3.f, -INFINITY};
                        c.suppress = {hot[1], V + 5};
                        staged[0] = hot[0];
                    }
                    c.finish();
                    DeviceCase dc(c);
                    Device<ts::Request> request(std::vector<ts::Request>{dc.request});
                    Device<float> logits(row), scratch(ts::scratch_floats(std::uint32_t(V), columns)), probs(std::size_t(columns) * V);
                    Device<std::int32_t> staged_d(std::vector<std::int32_t>(staged.begin(), staged.end())), out(columns);
                    ts::Rows rows;
                    rows.logits = logits.p;
                    rows.stride = std::uint32_t(V);
                    rows.vocab = std::uint32_t(V);
                    rows.columns = columns;
                    rows.request = request.p;
                    rows.staged = staged_d.p;
                    rows.staged_stride = std::uint32_t(staged.size());
                    rows.scratch = scratch.p;
                    ok(ts::enqueue_sample(rows, cc::DrawPurpose::sample, out.p, probs.p, nullptr), "sample");
                    const auto tokens = out.get();
                    const auto dense = probs.get();
                    for(int col = 0; col < columns; ++col) {
                        const std::span<const float> r(row.data() + std::size_t(col) * V, std::size_t(V));
                        const std::span<const cc::Token> st(staged.data(), std::size_t(col));
                        std::vector<cc::Token> order;
                        const auto want = core_law(host, r, c, st, &order);
                        for(int t = 0; t < V; ++t) {
                            const double a = dense[std::size_t(col) * V + std::size_t(t)], b = want[std::size_t(t)];
                            require((a > 0.) == (b > 0.), "the fast path keeps a different token set than Core");
                            require(std::abs(a - b) <= 1e-6 * b, "the fast path's law differs from Core's distribution");
                        }
                        auto h = c.history();
                        h.stage(st);
                        const auto expect = host.sample(r, c.rules, h, std::size_t(col)).token;
                        const auto [walked, margin] = walk(order, want, cc::uniform(c.rules.seed, c.outputs.size() + std::size_t(col), c.rules.stream));
                        (void)walked;
                        if(tokens[std::size_t(col)] != expect) {
                            require(margin < 1e-12, "a fast-path draw differs from Core's sampler");
                            ++skipped;
                        }
                        ++draws;
                    }
                    ++rows_checked;
                    ++trial;
                }
            }
            std::cout << "fast path: " << draws << " draws over " << rows_checked << " requests equal Core's sampler and laws ("
                      << skipped << " within 1e-12 of a boundary)\n";
        }

        // 1e. Open rows (temperature > 0, top_k 0): settled by the fast kernels when top_p or min_p closes the kept set
        //     within the top 64 or the draw lands there, and by the full law otherwise (a nearly flat row). Draws equal
        //     Core's away from the whole-row weight's FP32 rounding (1e-6) or the full law's (1e-4).
        {
            constexpr int V = 200003;
            cc::Sampler host(V);
            struct Rule { double scale, T, top_p, min_p; double margin; };
            const Rule rules[] = {{4., 1.0, 0.9, 0., 1e-6}, {4., 0.8, 1., 0.1, 1e-6}, {6., 1.0, 1., 0., 1e-6}, {3., 0.7, 0.95, 0.05, 1e-6},
                                  {0.05, 1.0, 0.9, 0., 1e-4}, {0.05, 1.0, 1., 0., 1e-4}};
            int draws = 0, skipped = 0;
            std::uint64_t seed = 900;
            for(const auto& rule : rules) {
                std::vector<float> row(V);
                for(auto& v : row) v = float(rule.scale) * normal(rng);
                constexpr int R = 16;   // one request per seed, all on the same row
                std::vector<ts::Request> reqs(R);
                std::vector<float> rows(std::size_t(R) * V);
                for(int r = 0; r < R; ++r) {
                    auto& q = reqs[std::size_t(r)];
                    q.temperature = rule.T;
                    q.top_p = rule.top_p;
                    q.min_p = rule.min_p;
                    q.seed = seed + std::uint64_t(r);
                    std::copy(row.begin(), row.end(), rows.begin() + std::ptrdiff_t(r) * V);
                }
                Device<ts::Request> request(reqs);
                Device<float> logits(rows), scratch(ts::scratch_floats(V, R));
                Device<std::int32_t> out(R);
                ts::Rows rw;
                rw.logits = logits.p;
                rw.stride = V;
                rw.vocab = V;
                rw.requests = R;
                rw.request = request.p;
                rw.scratch = scratch.p;
                ok(ts::enqueue_sample(rw, cc::DrawPurpose::sample, out.p, nullptr, nullptr), "sample");
                const auto tokens = out.get();
                for(int r = 0; r < R; ++r) {
                    Case c(V);
                    c.rules.temperature = rule.T;
                    c.rules.top_p = rule.top_p;
                    c.rules.min_p = rule.min_p;
                    c.rules.seed = seed + std::uint64_t(r);
                    std::vector<cc::Token> order;
                    const auto want = core_law(host, row, c, {}, &order);
                    const auto [expect, margin] = walk(order, want, cc::uniform(c.rules.seed, 0, 0));
                    require(expect == host.sample(row, c.rules, c.history()).token || margin < 1e-9, "test walk differs from Core");
                    ++draws;
                    if(tokens[std::size_t(r)] != expect) {
                        require(margin < rule.margin, "an open row's draw differs from Core's sampler");
                        ++skipped;
                    }
                }
                seed += R;
            }
            std::cout << "open rows: " << draws << " draws equal Core's sampler (" << skipped << " near a boundary)\n";
        }

        // 1f. Distinct processed doubles that round down to one float share a key up to the id, so the key order
        //     between them is by id, not by value. With such a group at the top_k (or greedy) boundary the merged
        //     candidates are not the exact best ones, and the row goes to the full law: here the true maximum is the
        //     group's last id, which a re-ranking of the first two candidates would miss.
        {
            constexpr int V = 64;
            cc::Sampler host(V);
            const double f = double(-1.2f), u = std::ldexp(1.0, -23);   // a float in [-1.25, -1.125) and its ulp
            const int ids[3] = {10, 20, 30};
            const double offsets[3] = {1. / 8, 2. / 8, 7. / 8};
            std::vector<float> row(V, -INFINITY);
            for(int i = 0; i < 3; ++i) {
                row[std::size_t(ids[i])] = float(f + offsets[i] * u + 1.0);
                require(double(row[std::size_t(ids[i])]) - 1.0 == f + offsets[i] * u, "the premise: x - 1 is exact in double");
            }
            int checked = 0;
            for(const double T : {0.0, 1.0}) {
                Case c(V);
                c.rules.temperature = T;
                c.rules.top_k = 1;
                c.rules.presence_penalty = 1.0;   // values x - 1: three doubles inside one float's ulp
                c.outputs = {10, 20, 30};
                c.finish();
                DeviceCase dc(c);
                Device<ts::Request> request(std::vector<ts::Request>{dc.request});
                Device<float> logits(row), scratch(ts::scratch_floats(V, 1)), probs(V);
                Device<std::int32_t> out(1);
                ts::Rows rw;
                rw.logits = logits.p; rw.stride = V; rw.vocab = V; rw.request = request.p; rw.scratch = scratch.p;
                ok(ts::enqueue_sample(rw, cc::DrawPurpose::sample, out.p, probs.p, nullptr), "sample");
                const auto expect = host.sample(row, c.rules, c.history()).token;
                require(expect == 30, "the premise: Core's maximum is the group's last id");
                require(out.get()[0] == expect, "a tie group of inexact keys at the boundary picked the wrong token");
                require(probs.get()[30] == 1.f, "a tie group of inexact keys at the boundary kept the wrong token");
                ++checked;
            }
            std::cout << "inexact key groups: " << checked << " boundary cases equal Core's choice\n";
        }

        // 1g. One launch, every path: requests with greedy, fast, open and full-law rules (top_k 100) side by side,
        //     each row equal to Core's law and draw independently of its neighbours.
        {
            constexpr int V = 30011, R = 8, columns = 2;
            cc::Sampler host(V);
            struct Rule { double T, top_p; std::uint32_t top_k; double margin; };
            const Rule rules[R] = {{0., 1., 0, 1e-12}, {1.0, 0.95, 20, 1e-12}, {0.8, 0.9, 0, 1e-4}, {1.0, 0.95, 100, 1e-4},
                                   {0.6, 1., 64, 1e-12}, {1.2, 1., 0, 1e-4}, {0., 1., 7, 1e-12}, {1.0, 0.8, 1, 1e-12}};
            std::vector<float> row(std::size_t(R) * columns * V);
            for(std::size_t i = 0; i < row.size(); ++i) row[i] = (i / std::size_t(V)) % 3 == 2 ? 0.05f * normal(rng) : 3.f * normal(rng);
            std::vector<ts::Request> reqs(R);
            for(int r = 0; r < R; ++r) {
                reqs[std::size_t(r)].temperature = rules[r].T;
                reqs[std::size_t(r)].top_p = rules[r].top_p;
                reqs[std::size_t(r)].top_k = std::int32_t(rules[r].top_k);
                reqs[std::size_t(r)].seed = 31 + std::uint64_t(r);
                reqs[std::size_t(r)].position = std::uint64_t(r);
            }
            Device<ts::Request> request(reqs);
            Device<float> logits(row), scratch(ts::scratch_floats(V, R * columns));
            Device<std::int32_t> out(std::size_t(R) * columns);
            ts::Rows rw;
            rw.logits = logits.p; rw.stride = V; rw.vocab = V; rw.requests = R; rw.columns = columns; rw.request = request.p;
            rw.scratch = scratch.p;
            ok(ts::enqueue_sample(rw, cc::DrawPurpose::sample, out.p, nullptr, nullptr), "sample");
            const auto tokens = out.get();
            int skipped = 0;
            for(int r = 0; r < R; ++r)
                for(int col = 0; col < columns; ++col) {
                    const std::span<const float> x(row.data() + (std::size_t(r) * columns + std::size_t(col)) * V, V);
                    Case c(V);
                    c.rules.temperature = rules[r].T;
                    c.rules.top_p = rules[r].top_p;
                    c.rules.top_k = rules[r].top_k;
                    c.rules.seed = reqs[std::size_t(r)].seed;
                    std::vector<cc::Token> order;
                    const auto want = core_law(host, x, c, {}, &order);
                    const auto [expect, margin] = walk(order, want, cc::uniform(c.rules.seed, std::uint64_t(r + col), 0));
                    if(tokens[std::size_t(r) * columns + std::size_t(col)] != expect) {
                        require(margin < rules[r].margin, "a row of a mixed launch differs from Core's sampler");
                        ++skipped;
                    }
                }
            std::cout << "mixed launch: " << R * columns << " rows on every path equal Core's draws (" << skipped << " near a boundary)\n";
        }

        // 2. Processing and filters against Core over staged columns: bias (including a ban), suppression before
        //    min_tokens, repetition over prompt and outputs, frequency and presence over committed plus staged
        //    outputs, top_k, top_p and min_p. Dense law and draws per column.
        {
            constexpr int V = 5000, columns = 4;
            cc::Sampler host(V);
            int mismatches = 0, draws = 0;
            for(int trial = 0; trial < 12; ++trial) {
                Case c(V);
                c.rules.temperature = 0.6 + 0.1 * (trial % 5);
                c.rules.top_k = trial % 3 == 0 ? 0u : 40u + std::uint32_t(trial);
                c.rules.top_p = trial % 2 ? 0.9 : 1.0;
                c.rules.min_p = trial % 4 == 1 ? 0.03 : 0.0;
                c.rules.repetition_penalty = 1.0 + 0.1 * (trial % 4);
                c.rules.frequency_penalty = trial % 2 ? 0.4 : 0.0;
                c.rules.presence_penalty = trial % 3 ? 0.5 : 0.0;
                c.rules.min_tokens = trial % 2 ? 6u : 0u;
                c.rules.seed = 900 + std::uint64_t(trial);
                c.rules.stream = std::uint32_t(trial % 3);
                std::vector<float> row(std::size_t(columns) * V);
                for(auto& v : row) v = 3.f * normal(rng);
                std::vector<cc::Token> hot(8);
                for(int i = 0; i < 8; ++i) hot[std::size_t(i)] = cc::Token(std::uniform_int_distribution<int>(0, V - 1)(rng));
                for(int i = 0; i < 20; ++i) c.prompt.push_back(hot[std::size_t(i % 8)]);
                for(int i = 0; i < 5; ++i) c.outputs.push_back(hot[std::size_t(i % 3)]);
                for(int col = 0; col < columns; ++col) row[std::size_t(col) * V + std::size_t(hot[0])] = 9.f;
                c.bias_ids = {hot[1], hot[4], 17};
                c.bias_values = {2.5f, -INFINITY, -1.f};
                c.suppress = {hot[2], V + 3};
                c.finish();
                const std::vector<cc::Token> staged = {hot[0], hot[3], hot[0]};
                DeviceCase dc(c);
                Device<ts::Request> request(1);
                request.put({dc.request});
                Device<float> logits(row), scratch(ts::scratch_floats(V, columns)), probs(std::size_t(columns) * V);
                Device<std::int32_t> staged_d(std::vector<std::int32_t>(staged.begin(), staged.end())), out(columns);
                ts::Rows rows;
                rows.logits = logits.p;
                rows.stride = V;
                rows.vocab = V;
                rows.columns = columns;
                rows.request = request.p;
                rows.staged = staged_d.p;
                rows.staged_stride = std::uint32_t(staged.size());
                rows.scratch = scratch.p;
                ok(ts::enqueue_sample(rows, cc::DrawPurpose::sample, out.p, probs.p, nullptr), "sample");
                const auto tokens = out.get();
                const auto dense = probs.get();
                for(int col = 0; col < columns; ++col) {
                    const std::span<const float> r(row.data() + std::size_t(col) * V, V);
                    std::vector<cc::Token> order;
                    const auto want = core_law(host, r, c, std::span<const cc::Token>(staged.data(), std::size_t(col)), &order);
                    for(int t = 0; t < V; ++t) {
                        const double a = dense[std::size_t(col) * V + std::size_t(t)], b = want[std::size_t(t)];
                        require(std::abs(a - b) <= 2e-4 * std::max(b, 1e-3), "device law differs from Core's distribution");
                    }
                    auto h = c.history();
                    h.stage(std::span<const cc::Token>(staged.data(), std::size_t(col)));
                    const auto expect = host.sample(r, c.rules, h, std::size_t(col)).token;
                    const auto [walked, margin] = walk(order, want, cc::uniform(c.rules.seed, c.outputs.size() + std::size_t(col), c.rules.stream));
                    require(margin < 1e-4 || walked == expect, "test walk differs from Core's sampler");
                    ++draws;
                    if(tokens[std::size_t(col)] != expect) {
                        require(margin < 1e-4, "device draw differs from Core's sampler away from a boundary");
                        ++mismatches;
                    }
                }
            }
            std::cout << "processing: " << draws << " columns equal Core's law, " << mismatches << " boundary mismatches\n";
        }

        // 3. Verification. Greedy drafts: every column is Core's own draw with the drafts before it staged, so the
        //    emitted text equals plain sampling. Sampled drafts (dense q): the rejection rule with Core's accept,
        //    residual and sample draws.
        {
            constexpr int V = 4000, K = 4;
            cc::Sampler host(V);
            int checked = 0, skipped = 0, accepted = 0;
            for(int trial = 0; trial < 40; ++trial) {
                const bool sampled = trial % 2 == 1;
                Case c(V);
                c.rules.temperature = 0.7 + 0.02 * trial;
                c.rules.top_k = trial % 4 < 2 ? 50u : 0u;
                c.rules.top_p = 0.95;
                c.rules.frequency_penalty = 0.2;
                c.rules.seed = 77 + std::uint64_t(trial);
                for(int i = 0; i < 3; ++i) c.outputs.push_back(cc::Token(i * 13));
                c.finish();
                std::vector<float> row(std::size_t(K + 1) * V), draft_row(std::size_t(K) * V);
                for(auto& v : row) v = 2.5f * normal(rng);
                for(std::size_t i = 0; i < draft_row.size(); ++i) draft_row[i] = row[i] + 0.8f * normal(rng);
                DeviceCase dc(c);
                Device<ts::Request> request(1);
                request.put({dc.request});
                Device<float> logits(row), dlogits(draft_row), scratch(ts::scratch_floats(V, K + 1)), q(std::size_t(K) * V);
                Device<std::int32_t> drafts_d(static_cast<std::size_t>(K)), predictions(K + 1);
                std::vector<cc::Token> drafts(K);
                if(sampled) {
                    // The drafter draws all K drafts at once here (a fixed draft law per column), staging nothing.
                    ts::Rows d;
                    d.logits = dlogits.p;
                    d.stride = V;
                    d.vocab = V;
                    d.columns = K;
                    d.request = request.p;
                    d.scratch = scratch.p;
                    ok(ts::enqueue_sample(d, cc::DrawPurpose::draft, drafts_d.p, q.p, nullptr), "draft sample");
                    const auto back = drafts_d.get();
                    for(int i = 0; i < K; ++i) drafts[std::size_t(i)] = back[std::size_t(i)];
                } else {
                    for(int i = 0; i < K; ++i) {
                        const auto* r = draft_row.data() + std::size_t(i) * V;
                        drafts[std::size_t(i)] = cc::Token(std::max_element(r, r + V) - r);
                    }
                    drafts_d.put(std::vector<std::int32_t>(drafts.begin(), drafts.end()));
                }
                const auto qlaw = q.get();
                ts::Verify v;
                v.target.logits = logits.p;
                v.target.stride = V;
                v.target.vocab = V;
                v.target.columns = K + 1;
                v.target.request = request.p;
                v.target.scratch = scratch.p;
                v.drafts = drafts_d.p;
                v.draft_probabilities = sampled ? q.p : nullptr;
                v.predictions = predictions.p;
                ok(ts::enqueue_verify(v, nullptr), "verify");
                const auto got = predictions.get();
                for(int col = 0; col <= K; ++col) {
                    const std::span<const float> r(row.data() + std::size_t(col) * V, V);
                    std::vector<cc::Token> order;
                    auto p = core_law(host, r, c, std::span<const cc::Token>(drafts.data(), std::size_t(col)), &order);
                    const std::uint64_t at = c.outputs.size() + std::size_t(col);
                    cc::Token want = -1;
                    double margin = 1.;
                    if(col == K || !sampled) {
                        std::tie(want, margin) = walk(order, p, cc::uniform(c.rules.seed, at, c.rules.stream));
                        if(col < K && want == drafts[std::size_t(col)]) ++accepted;
                    } else {
                        const cc::Token d = drafts[std::size_t(col)];
                        const double qd = double(qlaw[std::size_t(col) * V + std::size_t(d)]);
                        const double ua = cc::uniform(c.rules.seed, at, c.rules.stream, cc::DrawPurpose::accept);
                        margin = std::abs(ua * qd - p[std::size_t(d)]);
                        if(ua * qd < p[std::size_t(d)]) {
                            want = d;
                            ++accepted;
                        } else {
                            std::vector<double> residual(std::size_t(V), 0.);
                            for(int t = 0; t < V; ++t) residual[std::size_t(t)] = std::max(p[std::size_t(t)] - double(qlaw[std::size_t(col) * V + std::size_t(t)]), 0.);
                            double m2 = 1.;
                            std::tie(want, m2) = walk(order, residual, cc::uniform(c.rules.seed, at, c.rules.stream, cc::DrawPurpose::residual));
                            margin = std::min(margin, m2);
                        }
                    }
                    if(margin < 1e-4) { ++skipped; continue; }
                    require(got[std::size_t(col)] == want, sampled ? "device verify differs from the rejection rule" : "a greedy-draft column differs from Core's own draw");
                    ++checked;
                }
            }
            std::cout << "verify: " << checked << " columns equal Core's draws or the rejection rule (" << accepted << " accepts), "
                      << skipped << " near boundaries skipped\n";
        }

        // 3b. Cut drafts: a drafter that stops early (hidden-state-drafters' confidence cut) writes -1 for the rest of
        //     the round, while its q rows still hold the laws it computed. A cut column is verified against p alone,
        //     the target's own draw with u_sample like the bonus column, so a round cut after j drafts emits what a
        //     width-j round would. (The rejection rule there would draw from max(p - q, 0), a different law.)
        {
            constexpr int V = 64, K = 2, R = 256;
            cc::Sampler host(V);
            std::vector<float> row(V, 0.f);
            for(auto& v : row) v = 1.5f * normal(rng);
            const int top = int(std::max_element(row.begin(), row.end()) - row.begin());
            cc::SamplingParams rules;
            rules.temperature = 1.0;
            std::vector<ts::Request> reqs(R);
            std::vector<std::int32_t> drafts(std::size_t(R) * K, -1);
            for(int r = 0; r < R; ++r) {
                reqs[std::size_t(r)].temperature = rules.temperature;
                reqs[std::size_t(r)].seed = 500 + std::uint64_t(r);
                if(r % 2) drafts[std::size_t(r) * K] = top;   // odd requests cut after one draft, even ones at once
            }
            std::vector<float> rows(std::size_t(R) * (K + 1) * V), qrows(std::size_t(R) * K * V, 0.f);
            for(int i = 0; i < R * (K + 1); ++i) std::copy(row.begin(), row.end(), rows.begin() + std::ptrdiff_t(i) * V);
            for(int i = 0; i < R * K; ++i) qrows[std::size_t(i) * V + std::size_t(top)] = 1.f;   // the drafter's laws
            Device<ts::Request> request(reqs);
            Device<float> logits(rows), q(qrows), scratch(ts::scratch_floats(V, R * (K + 1)));
            Device<std::int32_t> drafts_d(drafts), predictions(std::size_t(R) * (K + 1));
            ts::Verify v;
            v.target.logits = logits.p;
            v.target.stride = V;
            v.target.vocab = V;
            v.target.requests = R;
            v.target.columns = K + 1;
            v.target.request = request.p;
            v.target.scratch = scratch.p;
            v.drafts = drafts_d.p;
            v.draft_probabilities = q.p;
            v.predictions = predictions.p;
            ok(ts::enqueue_verify(v, nullptr), "verify");
            const auto got = predictions.get();
            int checked = 0, near = 0, tops = 0;
            for(int r = 0; r < R; ++r) {
                const int col = r % 2;   // the first cut column
                rules.seed = reqs[std::size_t(r)].seed;
                Case c(V);
                c.rules = rules;
                const cc::Token staged = cc::Token(top);
                std::vector<cc::Token> order;
                const auto p = core_law(host, row, c, std::span<const cc::Token>(&staged, std::size_t(col)), &order);
                const auto [want, margin] = walk(order, p, cc::uniform(rules.seed, std::uint64_t(col), rules.stream));
                if(margin < 1e-5) { ++near; continue; }
                require(got[std::size_t(r) * (K + 1) + std::size_t(col)] == want, "a cut draft column is not the target's own draw");
                tops += want == top;
                ++checked;
            }
            require(tops > 0, "no cut column drew the token the drafter's q covers");
            std::cout << "cut drafts: " << checked << " cut columns equal the target's own draw (" << tops
                      << " of them the token q covered), " << near << " near boundaries skipped\n";
        }

        // 3c. Sparse draft laws: enqueue_sample's kept pairs equal its dense rows on fast rows (count -1 on full-law rows),
        //     and enqueue_verify against them writes the predictions it writes against the dense rows, on fast, open and
        //     full-law target rows alike; a full-law draft without a dense row takes the target's own draw (the
        //     greedy-draft rule, which keeps the law), and an empty sparse law is a cut column.
        {
            constexpr int V = 20011, K = 4, R = 24;
            std::vector<float> target(std::size_t(R) * (K + 1) * V), drafter(std::size_t(R) * K * V);
            for(auto& x : target) x = 2.5f * normal(rng);
            for(int r = 0; r < R; ++r)
                for(int c = 0; c < K; ++c)
                    for(int t = 0; t < V; ++t)
                        drafter[(std::size_t(r) * K + std::size_t(c)) * V + std::size_t(t)] =
                            target[(std::size_t(r) * (K + 1) + std::size_t(c)) * V + std::size_t(t)] + 0.6f * normal(rng);
            std::vector<ts::Request> draft_reqs(R), target_reqs(R);
            for(int r = 0; r < R; ++r) {
                auto& d = draft_reqs[std::size_t(r)];
                d.temperature = 0.9;
                d.top_k = r % 5 == 4 ? 100 : 20;   // every fifth drafter takes the full law: no sparse form
                d.top_p = 0.95;
                d.seed = 60 + std::uint64_t(r);
                auto& q = target_reqs[std::size_t(r)];
                q.temperature = 0.9;
                q.top_k = r % 3 == 0 ? 20 : (r % 3 == 1 ? 100 : 0);   // fast, full-law and open target rows
                q.top_p = 0.95;
                q.seed = d.seed;
            }
            Device<ts::Request> dreq(draft_reqs), treq(target_reqs);
            Device<float> tlogits(target), dlogits(drafter), scratch(ts::scratch_floats(V, R * (K + 1))), dense(std::size_t(R) * K * V);
            Device<std::int32_t> drafts(std::size_t(R) * K), predictions(std::size_t(R) * (K + 1));
            Device<ts::SparseLaw> sparse(std::size_t(R) * K);
            ts::Rows d;
            d.logits = dlogits.p; d.stride = V; d.vocab = V; d.requests = R; d.columns = K; d.request = dreq.p; d.scratch = scratch.p;
            ok(ts::enqueue_sample(d, cc::DrawPurpose::draft, drafts.p, dense.p, sparse.p, ts::Logprobs{}, nullptr), "draft sample");
            const auto q = dense.get();
            auto laws = sparse.get();
            int sparse_rows = 0;
            for(int i = 0; i < R * K; ++i) {
                const auto& law = laws[std::size_t(i)];
                if(draft_reqs[std::size_t(i / K)].top_k > ts::kMaxTopK) {
                    require(law.count == -1, "a full-law draft row has a sparse form");
                    continue;
                }
                require(law.count > 0 && law.count <= ts::kMaxTopK, "a fast draft row has no sparse form");
                int nonzero = 0;
                for(int t = 0; t < V; ++t) nonzero += q[std::size_t(i) * V + std::size_t(t)] > 0.f;
                require(nonzero == law.count, "the sparse law keeps a different set than the dense row");
                for(int j = 0; j < law.count; ++j)
                    require(q[std::size_t(i) * V + std::size_t(law.ids[j])] == law.probabilities[j], "a sparse probability differs from the dense row");
                ++sparse_rows;
            }
            ts::Verify v;
            v.target.logits = tlogits.p; v.target.stride = V; v.target.vocab = V; v.target.requests = R; v.target.columns = K + 1;
            v.target.request = treq.p; v.target.scratch = scratch.p;
            v.drafts = drafts.p;
            v.predictions = predictions.p;
            auto run = [&](const float* dq, const ts::SparseLaw* sq) {
                v.draft_probabilities = dq;
                v.draft_laws = sq;
                ok(ts::enqueue_verify(v, nullptr), "verify");
                return predictions.get();
            };
            const auto with_dense = run(dense.p, nullptr), with_both = run(dense.p, sparse.p), with_sparse = run(nullptr, sparse.p),
                       greedy_rule = run(nullptr, nullptr);
            require(with_both == with_dense, "sparse laws (dense rows for full-law drafts) verify differently than dense rows");
            for(int r = 0; r < R; ++r)
                for(int c = 0; c <= K; ++c) {
                    const std::size_t i = std::size_t(r) * (K + 1) + std::size_t(c);
                    const bool full = draft_reqs[std::size_t(r)].top_k > ts::kMaxTopK;
                    require(with_sparse[i] == (full && c < K ? greedy_rule[i] : with_dense[i]),
                            "a column verified against its sparse law differs");
                }
            // An empty sparse law at request 0's column 1: a cut column, the target's own draw.
            laws[1].count = 0;
            sparse.put(laws);
            const auto cut = run(nullptr, sparse.p);
            require(cut[1] == greedy_rule[1], "an empty sparse law is not a cut column");
            const auto first = drafts.get();
            int accepted = 0;
            for(int r = 0; r < R; ++r) accepted += with_sparse[std::size_t(r) * (K + 1)] == first[std::size_t(r) * K];
            std::cout << "sparse laws: " << sparse_rows << " draft rows equal their dense rows; verify predictions equal over "
                      << R << " requests on fast, open and full-law targets (" << accepted << " first drafts accepted)\n";
        }

        // 4. Distribution: over many seeds the first emitted token (prediction 0) follows the target law, with a
        //    greedy draft that is not the target's argmax and with drafts sampled from a different draft law.
        {
            constexpr int V = 12, R = 16384;
            const std::vector<float> target = {1.2f, 0.8f, 0.5f, 0.1f, 0.0f, -0.3f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -9.0f};
            const std::vector<float> drafter = {-1.0f, 1.5f, 0.2f, 0.9f, -0.5f, 0.0f, 0.3f, -2.0f, -0.4f, -3.0f, 0.6f, -1.0f};
            std::vector<ts::Request> reqs(R);
            for(int r = 0; r < R; ++r) { reqs[std::size_t(r)].temperature = 1.f; reqs[std::size_t(r)].seed = 1 + std::uint64_t(r); }
            double z = 0.;
            for(float x : target) z += std::exp(double(x));
            for(int mode = 0; mode < 3; ++mode) {   // greedy drafts, sampled drafts with dense q, with sparse q
                Device<ts::Request> request(reqs);
                std::vector<float> rows(std::size_t(R) * 2 * V), draft_rows(std::size_t(R) * V);
                for(int r = 0; r < R; ++r) {
                    std::copy(target.begin(), target.end(), rows.begin() + std::ptrdiff_t(std::size_t(r) * 2 * V));
                    std::copy(target.begin(), target.end(), rows.begin() + std::ptrdiff_t(std::size_t(r) * 2 * V + V));
                    std::copy(drafter.begin(), drafter.end(), draft_rows.begin() + std::ptrdiff_t(std::size_t(r) * V));
                }
                Device<float> logits(rows), dlogits(draft_rows), scratch(ts::scratch_floats(V, 2 * R)), q(std::size_t(R) * V);
                Device<std::int32_t> drafts(static_cast<std::size_t>(R)), predictions(std::size_t(R) * 2);
                Device<ts::SparseLaw> sparse(static_cast<std::size_t>(R));
                if(mode == 0) {
                    drafts.put(std::vector<std::int32_t>(std::size_t(R), 1));   // greedy draft token 1, not the argmax
                } else {
                    ts::Rows d;
                    d.logits = dlogits.p;
                    d.stride = V;
                    d.vocab = V;
                    d.requests = R;
                    d.request = request.p;
                    d.scratch = scratch.p;
                    if(mode == 1) ok(ts::enqueue_sample(d, cc::DrawPurpose::draft, drafts.p, q.p, nullptr), "draft sample");
                    else ok(ts::enqueue_sample(d, cc::DrawPurpose::draft, drafts.p, nullptr, sparse.p, ts::Logprobs{}, nullptr), "draft sample");
                }
                ts::Verify v;
                v.target.logits = logits.p;
                v.target.stride = V;
                v.target.vocab = V;
                v.target.requests = R;
                v.target.columns = 2;
                v.target.request = request.p;
                v.target.scratch = scratch.p;
                v.drafts = drafts.p;
                v.draft_probabilities = mode == 1 ? q.p : nullptr;
                v.draft_laws = mode == 2 ? sparse.p : nullptr;
                v.predictions = predictions.p;
                ok(ts::enqueue_verify(v, nullptr), "verify");
                const auto got = predictions.get();
                std::vector<int> hist(V, 0);
                for(int r = 0; r < R; ++r) {
                    const int t = got[std::size_t(r) * 2];
                    require(t >= 0 && t < V, "prediction outside the vocabulary");
                    ++hist[std::size_t(t)];
                }
                for(int t = 0; t < V; ++t) {
                    const double pr = std::exp(double(target[std::size_t(t)])) / z, f = double(hist[std::size_t(t)]) / R;
                    require(std::abs(f - pr) < 4.5 * std::sqrt(pr * (1 - pr) / R) + 1e-3,
                            mode == 2 ? "sparse draft laws change the output law"
                                      : mode ? "sampled drafts change the output law" : "greedy drafts change the output law");
                }
                if(mode == 2) require(sparse.get()[0].count == V, "a 12-token draft law has no sparse form");
            }
            std::cout << "distribution: first tokens follow the target law for greedy drafts and sampled drafts with dense or sparse laws\n";
        }

        // 5. Commits: counts and positions follow the device emitted counts; null counts advance only positions.
        {
            constexpr int V = 32;
            Device<std::int32_t> counts(std::vector<std::int32_t>(V, 0));
            std::vector<ts::Request> reqs(2);
            reqs[0].counts = counts.p;
            reqs[0].position = 5;
            reqs[1].position = 9;
            Device<ts::Request> request(reqs);
            Device<std::int32_t> tokens(std::vector<std::int32_t>{3, 3, 7, -1, 4, 4, 4, 4}), emitted(std::vector<std::int32_t>{3, 0, 2, 0});
            ok(ts::enqueue_commit(request.p, 2, V, tokens.p, 4, emitted.p, 2, nullptr), "commit");
            const auto back = request.get();
            const auto n = counts.get();
            require(back[0].position == 8 && back[1].position == 11, "commit positions");
            require(n[3] == 2 && n[7] == 1 && n[4] == 0, "commit counts");
            ok(ts::enqueue_commit(request.p, 2, V, tokens.p, 4, nullptr, 0, nullptr), "commit one");
            require(request.get()[0].position == 9 && counts.get()[3] == 3, "single-token commit");
            ts::Rows bad;
            require(ts::enqueue_sample(bad, cc::DrawPurpose::sample, tokens.p, nullptr, nullptr) == cudaErrorInvalidValue, "empty rows");
            ts::Verify v;
            require(ts::enqueue_verify(v, nullptr) == cudaErrorInvalidValue, "empty verify");
            std::cout << "commit: counts and positions follow the emitted counts\n";
        }
        ok(cudaDeviceSynchronize(), "sync");
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
