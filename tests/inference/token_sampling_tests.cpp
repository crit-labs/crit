// Device token selection against Core's host sampler: greedy ties, top-k 1 and tiny
// top-p reduce to greedy, seeded draws repeat and advance the position, chosen tokens
// equal crit::core::Sampler's away from FP32 rounding of a boundary, frequencies
// follow the tempered distribution, and constraint bitmasks mask logits as the host does.
#include "sampling.h"

#include <crit/sampling.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace cc = crit::core;
namespace ts = crit::techniques::token_sampling;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }

struct Sampler {
    float* logits = nullptr;
    float* cand_v = nullptr;
    std::int32_t* cand_i = nullptr;
    std::int32_t* out = nullptr;
    ts::Params* params = nullptr;
    int vocab;
    explicit Sampler(int v) : vocab(v) {
        ok(cudaMalloc(&logits, sizeof(float) * v), "malloc");
        ok(cudaMalloc(&cand_v, sizeof(float) * ts::kStageBlocks * ts::kMaxTopK), "malloc");
        ok(cudaMalloc(&cand_i, sizeof(std::int32_t) * ts::kStageBlocks * ts::kMaxTopK), "malloc");
        ok(cudaMalloc(&out, sizeof(std::int32_t)), "malloc");
        ok(cudaMalloc(&params, sizeof(ts::Params)), "malloc");
    }
    ~Sampler() { cudaFree(logits); cudaFree(cand_v); cudaFree(cand_i); cudaFree(out); cudaFree(params); }
    void load(const std::vector<float>& row) { ok(cudaMemcpy(logits, row.data(), sizeof(float) * vocab, cudaMemcpyHostToDevice), "upload"); }
    void set(const ts::Params& p) { ok(cudaMemcpy(params, &p, sizeof p, cudaMemcpyHostToDevice), "params"); }
    int draw() {
        ok(ts::enqueue_select(logits, vocab, params, cand_v, cand_i, out, nullptr), "select");
        int v = -1;
        ok(cudaMemcpy(&v, out, sizeof v, cudaMemcpyDeviceToHost), "download");
        return v;
    }
    std::uint64_t position() {
        ts::Params p;
        ok(cudaMemcpy(&p, params, sizeof p, cudaMemcpyDeviceToHost), "params");
        return p.position;
    }
};

// Core's rules for the device parameters; the device caps top_k (0 included) at kMaxTopK.
cc::SamplingParams core_params(const ts::Params& p) {
    cc::SamplingParams q;
    q.temperature = p.temperature > 0.f ? double(p.temperature) : 0.0;
    q.top_p = p.top_p > 0.f && p.top_p < 1.f ? double(p.top_p) : 1.0;
    q.top_k = static_cast<std::uint32_t>(p.top_k > 0 ? std::min(p.top_k, ts::kMaxTopK) : ts::kMaxTopK);
    q.seed = p.seed;
    q.stream = p.stream;
    return q;
}
} // namespace

int main() {
    try {
        constexpr int V = 50000;
        Sampler s(V);
        std::mt19937 rng(11);
        std::normal_distribution<float> n(0.f, 2.f);
        std::vector<float> row(V);

        // greedy: lowest index among tied maxima, also for top-k 1 and a tiny top-p
        for(auto& v : row) v = std::round(n(rng) * 4.f) / 4.f;
        row[777] = row[31337] = 30.f;
        s.load(row);
        s.set(ts::Params{});
        require(s.draw() == 777, "greedy tie breaks to the lowest index");
        require(s.position() == 1, "greedy advances the position");
        cc::Sampler host(V);
        const cc::TokenHistory empty(std::span<const cc::Token>{});
        require(host.sample(row, core_params(ts::Params{}), empty).token == 777, "Core's greedy choice differs");
        ts::Params p;
        p.temperature = 1.5f;
        p.top_k = 1;
        s.set(p);
        require(s.draw() == 777, "top-k 1 is greedy");
        p.top_k = 0;
        p.top_p = 1e-6f;
        s.set(p);
        require(s.draw() == 777, "tiny top-p is greedy");

        // seeded draws: repeatable, position advances, equal to Core's draw
        int mismatches = 0, draws = 0;
        for(int t = 0; t < 40; ++t) {
            for(auto& v : row) v = n(rng);
            s.load(row);
            ts::Params q;
            q.temperature = 0.5f + 0.05f * t;
            q.top_p = t % 2 ? 0.8f : 1.f;
            q.top_k = t % 3 == 0 ? 0 : 20;
            q.seed = 1000 + t;
            q.stream = static_cast<std::uint32_t>(t % 4);
            s.set(q);
            std::vector<int> first;
            for(int c = 0; c < 6; ++c) first.push_back(s.draw());
            require(s.position() == 6, "position advances per draw");
            s.set(q);
            for(int c = 0; c < 6; ++c) require(s.draw() == first[c], "same seed repeats");
            // Core's law (kept tokens in rank order) and its draw at each position.
            const auto rules = core_params(q);
            std::vector<cc::TokenLogprob> law;
            host.distribution(row, rules, empty, 0, law);
            cc::TokenHistory history(std::span<const cc::Token>{});
            for(int c = 0; c < 6; ++c) {
                const int want = host.sample(row, rules, history).token;
                const cc::Token emitted = first[c];
                history.stage(std::span<const cc::Token>(&emitted, 1));
                history.commit(1);
                const double u = cc::uniform(q.seed, static_cast<std::uint64_t>(c), q.stream);
                double margin = 1, upper = 0;
                bool in_pool = false;
                for(const auto& e : law) {
                    upper += std::exp(double(e.logprob));
                    margin = std::min(margin, std::abs(u - upper));
                    in_pool |= e.token == first[c];
                }
                require(in_pool, "draw outside Core's kept set");
                ++draws;
                if(first[c] != want) {
                    require(margin < 1e-4, "draw differs from Core's sampler away from a boundary");
                    ++mismatches;
                }
            }
        }

        // frequencies follow softmax(logits / T) over the candidates
        std::fill(row.begin(), row.end(), -100.f);
        const float lg[4] = {1.0f, 0.5f, 0.0f, -0.5f};
        for(int i = 0; i < 4; ++i) row[100 + 7 * i] = lg[i];
        s.load(row);
        ts::Params f;
        f.temperature = 1.f;
        f.top_k = 4;
        f.seed = 99;
        s.set(f);
        std::vector<int> hist(4, 0);
        constexpr int N = 8000;
        for(int c = 0; c < N; ++c) {
            const int t = s.draw();
            require(t >= 100 && (t - 100) % 7 == 0 && (t - 100) / 7 < 4, "draw outside top-k");
            ++hist[(t - 100) / 7];
        }
        double z = 0;
        for(float v : lg) z += std::exp(double(v));
        for(int i = 0; i < 4; ++i) {
            const double pr = std::exp(double(lg[i])) / z, got = double(hist[i]) / N;
            require(std::abs(got - pr) < 4 * std::sqrt(pr * (1 - pr) / N) + 1e-3, "frequency differs from the distribution");
        }

        // Device masks from a constraint's bitmask: exactly the host's -inf pattern on strided rows, and selection on
        // the masked row equal to Core's constrained Sampler (AllowedTokens), ties to the lowest allowed id.
        {
            constexpr int rows = 3, stride = V + 7, words = (V + 31) / 32, mask_stride = words + 3;
            std::vector<float> block(static_cast<std::size_t>(rows) * stride);
            for(auto& x : block) x = std::round(n(rng) * 4.f) / 4.f;
            std::vector<std::uint32_t> mask(static_cast<std::size_t>(rows) * mask_stride);
            std::uniform_int_distribution<std::uint32_t> bits;
            for(auto& w : mask) w = bits(rng) & bits(rng);   // about a quarter of the tokens allowed
            float* d_block = nullptr;
            std::uint32_t* d_mask = nullptr;
            ok(cudaMalloc(&d_block, sizeof(float) * block.size()), "malloc");
            ok(cudaMalloc(&d_mask, sizeof(std::uint32_t) * mask.size()), "malloc");
            ok(cudaMemcpy(d_block, block.data(), sizeof(float) * block.size(), cudaMemcpyHostToDevice), "upload");
            ok(cudaMemcpy(d_mask, mask.data(), sizeof(std::uint32_t) * mask.size(), cudaMemcpyHostToDevice), "upload");
            ok(ts::enqueue_mask(d_block, V, stride, d_mask, mask_stride, rows, nullptr), "mask");
            std::vector<float> masked(block.size());
            ok(cudaMemcpy(masked.data(), d_block, sizeof(float) * masked.size(), cudaMemcpyDeviceToHost), "download");
            for(int r = 0; r < rows; ++r)
                for(int t = 0; t < stride; ++t) {
                    const std::size_t i = static_cast<std::size_t>(r) * stride + t;
                    const bool allowed = t >= V || ((mask[static_cast<std::size_t>(r) * mask_stride + t / 32] >> (t % 32)) & 1u);
                    require(allowed ? masked[i] == block[i] : std::isinf(masked[i]) && masked[i] < 0, "device mask differs from the host's");
                }
            for(int r = 0; r < rows; ++r) {
                std::vector<float> raw(block.begin() + static_cast<std::ptrdiff_t>(r) * stride, block.begin() + static_cast<std::ptrdiff_t>(r) * stride + V);
                std::vector<cc::Token> ids;
                for(int t = 0; t < V; ++t) if((mask[static_cast<std::size_t>(r) * mask_stride + t / 32] >> (t % 32)) & 1u) ids.push_back(t);
                s.load(std::vector<float>(masked.begin() + static_cast<std::ptrdiff_t>(r) * stride, masked.begin() + static_cast<std::ptrdiff_t>(r) * stride + V));
                s.set(ts::Params{});
                cc::SamplingParams greedy;
                greedy.temperature = 0.0;
                require(s.draw() == host.sample(raw, cc::AllowedTokens{ids}, greedy, empty).token, "masked greedy differs from Core's constrained choice");
                ts::Params q;
                q.temperature = 0.8f;
                q.top_k = 30;
                q.seed = 77 + static_cast<std::uint64_t>(r);
                s.set(q);
                const int got = s.draw();
                require(std::binary_search(ids.begin(), ids.end(), got), "a masked draw left the allowed set");
            }
            require(ts::enqueue_mask(d_block, V, V - 1, d_mask, words, 1, nullptr) == cudaErrorInvalidValue, "short logit stride");
            require(ts::enqueue_mask(d_block, V, V, d_mask, words - 1, 1, nullptr) == cudaErrorInvalidValue, "short mask stride");
            cudaFree(d_block);
            cudaFree(d_mask);
        }

        require(ts::enqueue_select(nullptr, V, s.params, s.cand_v, s.cand_i, s.out, nullptr) == cudaErrorInvalidValue, "null logits");
        require(ts::enqueue_select(s.logits, 0, s.params, s.cand_v, s.cand_i, s.out, nullptr) == cudaErrorInvalidValue, "empty vocab");
        std::cout << "token sampling: greedy ties, nucleus draws (" << mismatches << " boundary mismatches of " << draws
                  << "), repeatable seeds and frequencies within bounds\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
