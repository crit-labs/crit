// Sampled speculative rounds on the device: token-sampling's verify predictions for greedy drafts through the
// unchanged greedy accept kernel and the history commit equal a host round over Core's laws and draws (the
// target's own draw per column, drafts kept while they match, stops and output budgets), so the round emits what
// plain sampling would; a zero-width round equals one ordinary draw, and positions and counts follow the emitted
// and truncated counts.
#include "greedy_accept.h"
#include "sampling.h"

#include <crit/sampling.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cc = crit::core;
namespace ts = crit::techniques::token_sampling;
namespace dr = crit::techniques::device_round;

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

// Core's law at one column (outputs committed, drafts staged), dense and in rank order.
std::vector<double> law(cc::Sampler& host, std::span<const float> row, const cc::SamplingParams& rules,
                        const std::vector<cc::Token>& outputs, std::span<const cc::Token> staged, std::vector<cc::Token>& order) {
    cc::TokenHistory h(std::span<const cc::Token>{});
    if(!outputs.empty()) { h.stage(outputs); h.commit(outputs.size()); }
    if(!staged.empty()) h.stage(staged);
    std::vector<cc::TokenLogprob> kept;
    host.distribution(row, rules, h, staged.size(), kept);
    std::vector<double> p(row.size(), 0.);
    order.clear();
    for(const auto& e : kept) { p[std::size_t(e.token)] = std::exp(double(e.logprob)); order.push_back(e.token); }
    return p;
}

// Rank-order CDF walk; margin is the draw's distance to the nearest boundary in normalized units.
cc::Token walk(const std::vector<cc::Token>& order, const std::vector<double>& w, double u, double& margin) {
    double total = 0.;
    for(auto t : order) total += w[std::size_t(t)];
    double c = 0.;
    cc::Token chosen = order.back();
    bool found = false;
    for(auto t : order) {
        if(!(w[std::size_t(t)] > 0.)) continue;
        c += w[std::size_t(t)];
        margin = std::min(margin, std::abs(u * total - c) / total);
        if(!found && u * total < c) { chosen = t; found = true; }
    }
    return chosen;
}
} // namespace

int main() {
    try {
        constexpr int V = 3000, K = 3, R = 48;
        std::mt19937 rng(5);
        std::normal_distribution<float> normal(0.f, 2.f);
        cc::Sampler host(V);
        const cc::Token stop = 11;
        int compared = 0, skipped = 0, emitted_total = 0;
        for(int trial = 0; trial < 6; ++trial) {
            cc::SamplingParams rules;
            rules.temperature = 0.6 + 0.15 * trial;
            rules.top_p = trial % 2 ? 0.9 : 1.0;
            rules.top_k = trial % 3 ? 0u : 64u;
            rules.presence_penalty = 0.3;
            std::vector<std::vector<cc::Token>> outputs(std::size_t(R));
            std::vector<ts::Request> reqs(std::size_t(R));
            std::vector<std::int32_t> counts(std::size_t(R) * V, 0);
            Device<std::int32_t> counts_d(counts);
            for(int r = 0; r < R; ++r) {
                for(int i = 0; i < 2 + r % 3; ++i) outputs[std::size_t(r)].push_back(cc::Token((r * 7 + i * 31) % V));
                auto& q = reqs[std::size_t(r)];
                q.temperature = float(rules.temperature);
                q.top_p = float(rules.top_p);
                q.top_k = std::int32_t(rules.top_k);
                q.presence = float(rules.presence_penalty);
                q.seed = 1000 * std::uint64_t(trial) + std::uint64_t(r);
                q.position = outputs[std::size_t(r)].size();
                q.counts = counts_d.p + std::size_t(r) * V;
                for(auto t : outputs[std::size_t(r)]) ++counts[std::size_t(r) * V + std::size_t(t)];
            }
            counts_d.put(counts);
            Device<ts::Request> request(reqs);
            // Target logits for K + 1 columns per request; greedy drafts from a perturbed copy (the drafter).
            std::vector<float> rows(std::size_t(R) * (K + 1) * V);
            for(auto& x : rows) x = normal(rng);
            std::vector<std::int32_t> drafts(std::size_t(R) * K);
            for(int r = 0; r < R; ++r)
                for(int c = 0; c < K; ++c) {
                    const float* row = rows.data() + (std::size_t(r) * (K + 1) + c) * V;
                    std::vector<float> noisy(row, row + V);
                    for(auto& x : noisy) x += 0.7f * normal(rng);
                    drafts[std::size_t(r) * K + c] = std::int32_t(std::max_element(noisy.begin(), noisy.end()) - noisy.begin());
                }
            if(trial == 1) drafts[0] = stop;   // an accepted stop ends the round
            std::vector<std::int32_t> control(std::size_t(R) * dr::control::fields, 0);
            std::vector<std::uint32_t> budget(std::size_t(R));
            for(int r = 0; r < R; ++r) {
                budget[std::size_t(r)] = 1 + std::uint32_t(r % (K + 1));
                control[std::size_t(r) * dr::control::fields + dr::control::frontier] = 100 + r;
                control[std::size_t(r) * dr::control::fields + dr::control::last_token] = outputs[std::size_t(r)].back();
                control[std::size_t(r) * dr::control::fields + dr::control::output_budget] = std::int32_t(budget[std::size_t(r)]);
            }
            Device<float> logits(rows), scratch(ts::scratch_floats(V, R * (K + 1)));
            Device<std::int32_t> drafts_d(drafts), predictions(std::size_t(R) * (K + 1)), control_d(control),
                result(std::size_t(R) * dr::result_width(K)), stops(std::vector<std::int32_t>{stop});
            ts::Verify v;
            v.target.logits = logits.p;
            v.target.stride = V;
            v.target.vocab = V;
            v.target.requests = R;
            v.target.columns = K + 1;
            v.target.request = request.p;
            v.target.scratch = scratch.p;
            v.drafts = drafts_d.p;
            v.predictions = predictions.p;
            ok(ts::enqueue_verify(v, nullptr), "verify");
            dr::AcceptRows a;
            a.drafts = drafts_d.p;
            a.predictions = predictions.p;
            a.stops = stops.p;
            a.stop_count = 1;
            a.control = control_d.p;
            a.result = result.p;
            a.rows = R;
            a.draft_tokens = K;
            ok(dr::enqueue_greedy_accept(a, nullptr), "accept");
            // Core trims one request at a stop it alone knew: truncate before the history commit.
            ok(dr::enqueue_truncate(control_d.p, result.p, K, 0, 1, nullptr), "truncate");
            ok(ts::enqueue_commit(request.p, R, V, result.p + 1, dr::result_width(K), control_d.p + dr::control::n_emitted,
                                  dr::control::fields, nullptr), "commit");
            const auto res = result.get();
            const auto ctl = control_d.get();
            const auto back = request.get();
            const auto after = counts_d.get();
            for(int r = 0; r < R; ++r) {
                const auto* got = res.data() + std::size_t(r) * dr::result_width(K);
                // Host round with Core's laws and draws: the target's draw per column, drafts kept while they match.
                std::vector<cc::Token> out;
                double margin = 1.;
                bool stopped = false;
                std::uint32_t matched = 0;
                const auto& q = reqs[std::size_t(r)];
                while(matched < std::uint32_t(K) && out.size() < budget[std::size_t(r)]) {
                    const std::span<const float> row(rows.data() + (std::size_t(r) * (K + 1) + matched) * V, V);
                    const std::span<const cc::Token> staged(drafts.data() + std::size_t(r) * K, matched);
                    std::vector<cc::Token> order;
                    auto p = law(host, row, rules, outputs[std::size_t(r)], staged, order);
                    const cc::Token d = drafts[std::size_t(r) * K + matched];
                    const std::uint64_t at = q.position + matched;
                    // A greedy draft stands exactly when it equals the target's own draw at its position.
                    const cc::Token t = walk(order, p, cc::uniform(q.seed, at, 0, cc::DrawPurpose::sample), margin);
                    out.push_back(t);
                    if(t != d) break;
                    ++matched;
                    if(d == stop) { stopped = true; break; }
                }
                if(matched == std::uint32_t(K) && out.size() < budget[std::size_t(r)] && !stopped) {
                    const std::span<const float> row(rows.data() + (std::size_t(r) * (K + 1) + K) * V, V);
                    const std::span<const cc::Token> staged(drafts.data() + std::size_t(r) * K, K);
                    std::vector<cc::Token> order;
                    const auto p = law(host, row, rules, outputs[std::size_t(r)], staged, order);
                    out.push_back(walk(order, p, cc::uniform(q.seed, q.position + K, 0, cc::DrawPurpose::sample), margin));
                }
                if(r == 0 && out.size() > 1) out.resize(1);   // the host-side truncation above
                const auto n = ctl[std::size_t(r) * dr::control::fields + dr::control::n_emitted];
                require(n >= 1 && std::uint32_t(n) <= budget[std::size_t(r)], "emitted count outside the allowance");
                require(ctl[std::size_t(r) * dr::control::fields + dr::control::frontier] == 100 + r + n, "frontier follows the emitted count");
                require(back[std::size_t(r)].position == q.position + std::uint64_t(n), "commit advances the position by the emitted count");
                for(int i = 0; i < n; ++i)
                    require(after[std::size_t(r) * V + std::size_t(got[1 + i])] >= counts[std::size_t(r) * V + std::size_t(got[1 + i])] + 1,
                            "commit counts the emitted tokens");
                emitted_total += n;
                if(margin < 1e-4) { ++skipped; continue; }
                require(std::size_t(n) == out.size(), "device round emits a different count than the host round");
                for(int i = 0; i < n; ++i) require(got[1 + i] == out[std::size_t(i)], "device round differs from the host round");
                ++compared;
            }
        }

        // K = 0: a zero-width round is one ordinary draw at the request's position.
        {
            constexpr int V0 = 500;
            std::vector<float> row(V0);
            for(auto& x : row) x = normal(rng);
            ts::Request q;
            q.temperature = 0.9f;
            q.top_p = 0.95f;
            q.seed = 31;
            q.position = 7;
            Device<ts::Request> request(std::vector<ts::Request>{q});
            Device<float> logits(row), scratch(ts::scratch_floats(V0, 1));
            Device<std::int32_t> prediction(1), drawn(1);
            ts::Verify v;
            v.target.logits = logits.p;
            v.target.stride = V0;
            v.target.vocab = V0;
            v.target.request = request.p;
            v.target.scratch = scratch.p;
            v.predictions = prediction.p;
            ok(ts::enqueue_verify(v, nullptr), "zero-width verify");
            ok(ts::enqueue_sample(v.target, cc::DrawPurpose::sample, drawn.p, nullptr, nullptr), "ordinary draw");
            require(prediction.get()[0] == drawn.get()[0], "a zero-width round differs from an ordinary draw");
        }
        ok(cudaDeviceSynchronize(), "sync");
        std::cout << "sampled rounds: " << compared << " requests equal the host round, " << skipped
                  << " near boundaries skipped, " << emitted_total << " tokens emitted\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
