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
        request.temperature = float(c.rules.temperature);
        request.top_p = float(c.rules.top_p);
        request.min_p = float(c.rules.min_p);
        request.top_k = std::int32_t(c.rules.top_k);
        request.repetition = float(c.rules.repetition_penalty);
        request.frequency = float(c.rules.frequency_penalty);
        request.presence = float(c.rules.presence_penalty);
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
            Device<float> logits(row), cand_v(std::size_t(ts::kStageBlocks) * ts::kMaxTopK);
            Device<std::int32_t> cand_i(std::size_t(ts::kStageBlocks) * ts::kMaxTopK), out(1);
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
                ok(ts::enqueue_select(logits.p, V, params.p, cand_v.p, cand_i.p, out.p, nullptr), "select");
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
            for(int mode = 0; mode < 2; ++mode) {
                Device<ts::Request> request(reqs);
                std::vector<float> rows(std::size_t(R) * 2 * V), draft_rows(std::size_t(R) * V);
                for(int r = 0; r < R; ++r) {
                    std::copy(target.begin(), target.end(), rows.begin() + std::ptrdiff_t(std::size_t(r) * 2 * V));
                    std::copy(target.begin(), target.end(), rows.begin() + std::ptrdiff_t(std::size_t(r) * 2 * V + V));
                    std::copy(drafter.begin(), drafter.end(), draft_rows.begin() + std::ptrdiff_t(std::size_t(r) * V));
                }
                Device<float> logits(rows), dlogits(draft_rows), scratch(ts::scratch_floats(V, 2 * R)), q(std::size_t(R) * V);
                Device<std::int32_t> drafts(static_cast<std::size_t>(R)), predictions(std::size_t(R) * 2);
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
                    ok(ts::enqueue_sample(d, cc::DrawPurpose::draft, drafts.p, q.p, nullptr), "draft sample");
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
                v.draft_probabilities = mode ? q.p : nullptr;
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
                            mode ? "sampled drafts change the output law" : "greedy drafts change the output law");
                }
            }
            std::cout << "distribution: first tokens follow the target law for greedy and sampled drafts\n";
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
