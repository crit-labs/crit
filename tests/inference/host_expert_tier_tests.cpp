// The CPU expert tier: a layer's routes give the bits of a serial computation of the same declared arithmetic
// (host_int's scalar rows, swiglu_declared, the declared quantizer) for every pool size, ISA and store layout; a route
// order with an expert's tokens split over chunks of 8 gives the same bits; per-row posts and FP32 inputs give the bits
// of a serial computation, and so does the mailbox payload; the pool runs many small batches, started and stopped, and
// rethrows a task's exception. Prints ms per layer and GB/s of expert
// bytes for Flash-sized experts (not asserted).
#include "expert_pool.h"
#include "host_tier.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ht = crit::techniques::host_tier;
namespace hi = crit::techniques::host_int;
namespace hn = crit::techniques::host_nvfp4;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
std::uint32_t bits(float f) { std::uint32_t b; std::memcpy(&b, &f, 4); return b; }

struct Store {
    std::size_t hidden, inter;
    std::vector<std::vector<std::uint8_t>> gu_codes, gu_scales, dn_codes, dn_scales, gu_inter, dn_inter;
    std::vector<hn::MatrixView> gu_views, dn_views;
    std::vector<ht::TierExpert> experts(bool interleaved, ht::Intermediate kind) {
        std::vector<ht::TierExpert> out;
        for(std::size_t e = 0; e < gu_codes.size(); ++e) {
            ht::TierExpert x;
            x.gate_up = {interleaved ? gu_inter[e].data() : gu_codes[e].data(), gu_scales[e].data(), 2 * inter, hidden, 0, 0,
                         hi::Weights::e2m1x2, interleaved ? hi::Layout::interleaved : hi::Layout::rows};
            x.down = {interleaved ? dn_inter[e].data() : dn_codes[e].data(), dn_scales[e].data(), hidden, inter, 0, 0,
                      hi::Weights::e2m1x2, interleaved ? hi::Layout::interleaved : hi::Layout::rows};
            x.gate_up_scales = hn::e4m3_scales(gu_views[e]);
            x.down_scales = hn::e4m3_scales(dn_views[e]);
            x.gate_up_post = 0.25f / 64.0f;
            x.down_post = 0.25f / 32.0f;
            x.intermediate = kind;
            x.intermediate_global = 16.0f;
            x.intermediate_block = 32;
            out.push_back(x);
        }
        return out;
    }
};

Store make_store(std::size_t experts, std::size_t hidden, std::size_t inter, std::mt19937& rng) {
    Store s{hidden, inter, {}, {}, {}, {}, {}, {}, {}, {}};
    std::uniform_int_distribution<int> byte(0, 255), scale(0x28, 0x48);
    for(std::size_t e = 0; e < experts; ++e) {
        auto fill = [&](std::size_t n, bool sc) {
            std::vector<std::uint8_t> v(n);
            for(auto& x : v) x = std::uint8_t(sc ? scale(rng) : byte(rng));
            return v;
        };
        s.gu_codes.push_back(fill(2 * inter * hidden / 2, false));
        s.gu_scales.push_back(fill(2 * inter * hidden / 16, true));
        s.dn_codes.push_back(fill(hidden * inter / 2, false));
        s.dn_scales.push_back(fill(hidden * inter / 16, true));
    }
    for(std::size_t e = 0; e < experts; ++e) {
        s.gu_views.push_back({s.gu_codes[e].data(), s.gu_scales[e].data(), 2 * inter, hidden});
        s.dn_views.push_back({s.dn_codes[e].data(), s.dn_scales[e].data(), hidden, inter});
        s.gu_inter.emplace_back(2 * inter * hi::row_bytes(hi::Layout::interleaved, hidden));
        s.dn_inter.emplace_back(hidden * hi::row_bytes(hi::Layout::interleaved, inter));
        ht::interleave_matrix({s.gu_codes[e].data(), nullptr, 2 * inter, hidden}, s.gu_inter[e].data());
        ht::interleave_matrix({s.dn_codes[e].data(), nullptr, hidden, inter}, s.dn_inter[e].data());
    }
    return s;
}

struct Inputs {
    std::vector<std::int32_t> expert;
    std::vector<std::int8_t> values;
    std::vector<float> scales;
};
Inputs make_inputs(int routes, std::size_t experts, std::size_t hidden, std::mt19937& rng) {
    Inputs in{std::vector<std::int32_t>(std::size_t(routes)), std::vector<std::int8_t>(std::size_t(routes) * hidden),
              std::vector<float>(std::size_t(routes) * hidden / 32)};
    std::uniform_int_distribution<int> pick(0, int(experts) - 1), act(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.02f);
    for(auto& e : in.expert) e = pick(rng);
    for(auto& v : in.values) v = std::int8_t(act(rng));
    for(auto& v : in.scales) v = sc(rng);
    return in;
}

// The serial reference: every route on its own, scalar rows, the declared SwiGLU and quantizer.
std::vector<float> reference(Store& s, const Inputs& in, ht::Intermediate kind) {
    auto experts = s.experts(false, kind);
    const std::size_t routes = in.expert.size();
    std::vector<float> out(routes * s.hidden);
    for(std::size_t r = 0; r < routes; ++r) {
        const auto& e = experts[std::size_t(in.expert[r])];
        std::vector<float> gu(2 * s.inter), h(s.inter);
        hi::dot_rows_int(hi::Isa::scalar, {}, e.gate_up, e.gate_up_scales,
                         {&in.values[r * s.hidden], s.hidden, &in.scales[r * s.hidden / 32], s.hidden / 32, 32}, 1, 1u, 0,
                         2 * s.inter, e.gate_up_post, gu.data(), 2 * s.inter);
        for(std::size_t k = 0; k < s.inter; ++k) h[k] = hi::swiglu_declared(gu[k], gu[s.inter + k]);
        std::vector<std::int8_t> hv(s.inter);
        std::vector<float> hs(s.inter / 16);
        int block = 16;
        if(kind == ht::Intermediate::e2m1x2) hi::quantize_e2m1x2(h.data(), s.inter, e.intermediate_global, hv.data(), hs.data());
        else {
            block = e.intermediate_block;
            hi::quantize_int8(h.data(), s.inter, block, hv.data(), hs.data());
        }
        hi::dot_rows_int(hi::Isa::scalar, {}, e.down, e.down_scales,
                         {hv.data(), s.inter, hs.data(), s.inter / std::size_t(block), block}, 1, 1u, 0, s.hidden,
                         e.down_post, &out[r * s.hidden], s.hidden);
    }
    return out;
}

void check_tier() {
    std::mt19937 rng(31);
    const std::size_t hidden = 512, inter = 256;
    Store store = make_store(6, hidden, inter, rng);
    // 21 routes over 6 experts: some expert has more than 8 tokens, so its routes split over chunks.
    Inputs in = make_inputs(21, 6, hidden, rng);
    for(int r = 0; r < 10; ++r) in.expert[std::size_t(r)] = 2;
    int compared = 0;
    for(auto kind : {ht::Intermediate::e2m1x2, ht::Intermediate::int8}) {
        const auto expect = reference(store, in, kind);
        for(int workers : {0, 2, 5})
            for(auto isa : {hi::Isa::scalar, hi::best_int_isa()})
                for(bool interleaved : {false, true}) {
                    ht::ExpertPool pool({workers, false});
                    ht::HostExpertTier tier(store.experts(interleaved, kind), pool, isa, 64);
                    std::vector<float> out(in.expert.size() * hidden, -1.0f);
                    tier.compute({int(in.expert.size()), in.expert.data(), in.values.data(), hidden, in.scales.data(),
                                  hidden / 32, 32, out.data(), hidden});
                    for(std::size_t i = 0; i < out.size(); ++i)
                        require(bits(out[i]) == bits(expect[i]),
                                "tier differs at " + std::to_string(i) + " (workers " + std::to_string(workers) + ", " +
                                    hn::isa_name(isa) + (interleaved ? ", interleaved" : ", rows") + ")");
                    require(tier.stats().routes == in.expert.size(), "tier stats");
                    ++compared;
                }
    }
    std::cout << "tier: " << compared << " configurations bitwise equal to the serial reference\n";
}

void check_pool() {
    ht::ExpertPool pool({3, false});
    std::vector<int> hits(1000);
    for(int round = 0; round < 2000; ++round) {
        const std::size_t n = std::size_t(round % 37 + 1);
        std::fill(hits.begin(), hits.begin() + std::ptrdiff_t(n), 0);
        pool.run(n, [&](std::size_t i) { hits[i] += 1; });
        for(std::size_t i = 0; i < n; ++i) require(hits[i] == 1, "a task ran other than once");
    }
    bool thrown = false;
    try {
        pool.run(10, [](std::size_t i) { if(i == 7) throw std::runtime_error("task 7"); });
    } catch(const std::runtime_error& e) {
        thrown = std::string(e.what()) == "task 7";
    }
    require(thrown, "a task's exception is rethrown");
    pool.run(4, [](std::size_t) {});   // the pool works after a failed batch
    std::cout << "pool: 2,000 batches, each task once; a task's exception rethrown\n";
}

// Per-row posts (a bundle's 1 / global per row times a factor), FP32 inputs quantized by each expert's own quantizer, the
// fused gate/up + intermediate phase and the input payload: the bits of a serial computation that quantizes, multiplies
// with post 1 and then by each row's post.
void check_inputs_and_row_posts() {
    std::mt19937 rng(41);
    const std::size_t hidden = 512, inter = 192, experts = 5;
    Store store = make_store(experts, hidden, inter, rng);
    std::uniform_real_distribution<float> inv(0.005f, 0.05f), xs(-3.0f, 3.0f);
    std::vector<std::vector<float>> gu_inv(experts, std::vector<float>(2 * inter)), dn_inv(experts, std::vector<float>(hidden));
    for(auto& v : gu_inv) for(auto& f : v) f = inv(rng);
    for(auto& v : dn_inv) for(auto& f : v) f = inv(rng);
    auto table = [&](ht::Intermediate kind) {
        auto list = store.experts(false, kind);
        for(std::size_t e = 0; e < experts; ++e) {
            list[e].gate_up_row_post = gu_inv[e].data();
            list[e].down_row_post = dn_inv[e].data();
            list[e].row_post_factor = kind == ht::Intermediate::e2m1x2 ? 0.25f : 1.0f;
            list[e].input = kind;
            list[e].input_global = 3.0f + float(e);
            list[e].input_block = 32;
        }
        return list;
    };
    const int tokens = 6, routes = 17;
    std::vector<float> x(std::size_t(tokens) * hidden);
    for(auto& v : x) v = xs(rng);
    std::vector<std::int32_t> expert(routes), token(routes);
    for(int r = 0; r < routes; ++r) {
        expert[std::size_t(r)] = r < 9 ? 1 : int(std::uint32_t(rng()) % experts);   // expert 1 has more than 8 tokens
        token[std::size_t(r)] = int(std::uint32_t(rng()) % tokens);
    }
    int compared = 0;
    for(auto kind : {ht::Intermediate::e2m1x2, ht::Intermediate::int8}) {
        const auto list = table(kind);
        const int ib = kind == ht::Intermediate::e2m1x2 ? 16 : 32;
        std::vector<float> expect(std::size_t(routes) * hidden);
        for(int r = 0; r < routes; ++r) {
            const auto& e = list[std::size_t(expert[std::size_t(r)])];
            const float* xr = &x[std::size_t(token[std::size_t(r)]) * hidden];
            std::vector<std::int8_t> xv(hidden), hv(inter);
            std::vector<float> xsc(hidden / 16), hs(inter / 16), gu(2 * inter), h(inter);
            if(kind == ht::Intermediate::e2m1x2) hi::quantize_e2m1x2(xr, hidden, e.input_global, xv.data(), xsc.data());
            else hi::quantize_int8(xr, hidden, 32, xv.data(), xsc.data());
            hi::dot_rows_int(hi::Isa::scalar, {}, e.gate_up, e.gate_up_scales,
                             {xv.data(), hidden, xsc.data(), hidden / std::size_t(ib), ib}, 1, 1u, 0, 2 * inter, 1.0f, gu.data(),
                             2 * inter);
            for(std::size_t k = 0; k < 2 * inter; ++k) gu[k] = gu[k] * (e.row_post_factor * e.gate_up_row_post[k]);
            for(std::size_t k = 0; k < inter; ++k) h[k] = hi::swiglu_declared(gu[k], gu[inter + k]);
            if(kind == ht::Intermediate::e2m1x2) hi::quantize_e2m1x2(h.data(), inter, e.intermediate_global, hv.data(), hs.data());
            else hi::quantize_int8(h.data(), inter, 32, hv.data(), hs.data());
            float* o = &expect[std::size_t(r) * hidden];
            hi::dot_rows_int(hi::Isa::scalar, {}, e.down, e.down_scales, {hv.data(), inter, hs.data(), inter / std::size_t(ib), ib},
                             1, 1u, 0, hidden, 1.0f, o, hidden);
            for(std::size_t k = 0; k < hidden; ++k) o[k] = o[k] * (e.row_post_factor * e.down_row_post[k]);
        }
        for(int workers : {0, 3, 5})
            for(auto isa : {hi::Isa::scalar, hi::best_int_isa()})
                for(std::size_t rows_per_task : {std::size_t(64), std::size_t(128), std::size_t(384)}) {
                    ht::ExpertPool pool({workers, false});
                    ht::HostExpertTier tier(hidden, inter, pool, isa, rows_per_task);
                    for(std::size_t e = 0; e < experts; ++e) tier.set_expert(e, list[e]);
                    std::vector<float> out(std::size_t(routes) * hidden, -1.0f);
                    tier.compute_inputs({routes, expert.data(), token.data(), x.data(), hidden, out.data(), hidden});
                    for(std::size_t i = 0; i < out.size(); ++i)
                        require(bits(out[i]) == bits(expect[i]), "compute_inputs differs at " + std::to_string(i) + " (workers " +
                                                                     std::to_string(workers) + ", " + hn::isa_name(isa) + ")");
                    ++compared;
                }
        ht::ExpertPool pool({2, false});
        ht::HostExpertTier tier(hidden, inter, pool);
        for(std::size_t e = 0; e < experts; ++e) tier.set_expert(e, list[e]);
        const auto layout = ht::HostExpertTier::input_layout(hidden, 32, 8);
        std::vector<std::byte> payload(layout.bytes);
        const std::uint32_t head[4] = {std::uint32_t(routes), std::uint32_t(tokens), 32u, 8u};
        std::memcpy(payload.data(), head, sizeof(head));
        std::memcpy(payload.data() + layout.experts, expert.data(), std::size_t(routes) * 4);
        std::memcpy(payload.data() + layout.token, token.data(), std::size_t(routes) * 4);
        std::memcpy(payload.data() + layout.x, x.data(), x.size() * 4);
        tier.serve_inputs(payload, 32, 8);
        for(std::size_t i = 0; i < std::size_t(routes) * hidden; ++i) {
            float v;
            std::memcpy(&v, payload.data() + layout.out + i * 4, 4);
            require(bits(v) == bits(expect[i]), "input payload output differs at " + std::to_string(i));
        }
        bool refused = false;
        try {
            const std::uint32_t bad[4] = {std::uint32_t(routes), 1u, 32u, 8u};   // routes reading past the payload's tokens
            std::memcpy(payload.data(), bad, sizeof(bad));
            tier.serve_inputs(payload, 32, 8);
        } catch(const std::invalid_argument&) {
            refused = true;
        }
        require(refused, "a payload whose routes read past its tokens is refused");
    }
    std::cout << "tier inputs: " << compared << " configurations with per-row posts and FP32 inputs bitwise equal to the "
              << "serial reference, input payload served\n";
}

// Pools started and stopped many times, batches of random sizes and task lengths published while workers spin, yield or
// sleep: every task of every batch runs once, under its own batch's function, and no late worker touches a finished
// batch (a worker claiming under an old epoch would run the old function on a new index).
void check_pool_races(int pools) {
    std::mt19937 rng(77);
    std::uint64_t batches = 0, wakes = 0;
    for(int p = 0; p < pools; ++p) {
        ht::PoolOptions options;
        options.workers = 1 + p % 5;
        options.pin = false;
        options.spin = std::chrono::microseconds(p % 3 == 0 ? 0 : 5 * (p % 7));
        options.sleep_after = std::chrono::microseconds(p % 2 ? 0 : 50);
        if(p % 4 == 1) options.spin_max = std::chrono::microseconds(200 + 100 * (p % 5));   // the adaptive spin
        const int rounds = 60 + p % 50;
        std::vector<std::vector<std::atomic<int>>> runs(static_cast<std::size_t>(rounds));
        std::vector<std::size_t> sizes(static_cast<std::size_t>(rounds));
        {
            ht::ExpertPool pool(options);
            for(int b = 0; b < rounds; ++b) {
                const std::size_t n = 1 + std::size_t(rng()) % 64;
                sizes[std::size_t(b)] = n;
                runs[std::size_t(b)] = std::vector<std::atomic<int>>(n);
                const int spin = int(std::uint32_t(rng()) % 4);
                auto& mine = runs[std::size_t(b)];
                pool.run(n, [&mine, spin, n](std::size_t i) {
                    if(i >= n) throw std::runtime_error("a task index past its batch");
                    mine[i].fetch_add(1, std::memory_order_relaxed);
                    for(int k = 0; k < spin * 200; ++k) std::atomic_signal_fence(std::memory_order_seq_cst);
                });
                for(std::size_t i = 0; i < n; ++i) require(mine[i].load() == 1, "a task ran other than once in its batch");
                if(rng() % 8 == 0) std::this_thread::sleep_for(std::chrono::microseconds(rng() % 300));
            }
            batches += pool.batches();
            wakes += pool.wakes();
        }   // the pool stops here, its workers joined
        for(int b = 0; b < rounds; ++b)
            for(std::size_t i = 0; i < sizes[std::size_t(b)]; ++i)
                require(runs[std::size_t(b)][i].load() == 1, "a late worker ran a finished batch's task");
    }
    std::cout << "pool races: " << pools << " pools started and stopped, " << batches << " batches (" << wakes
              << " published to sleeping workers), each task once in its own batch\n";
}

void throughput() {
    std::mt19937 rng(5);
    const std::size_t hidden = 2560, inter = 640, experts = 64;   // 2.76 MB each, past the caches
    Store store = make_store(experts, hidden, inter, rng);
    ht::ExpertPool pool;
    ht::HostExpertTier tier(store.experts(true, ht::Intermediate::e2m1x2), pool);
    for(int per_expert : {1, 2, 3}) {
        // A layer's CPU share: 6 experts with `per_expert` tokens each, a different set every layer.
        const int layers = 48;
        Inputs in = make_inputs(6 * per_expert * layers, experts, hidden, rng);
        for(int l = 0; l < layers; ++l)
            for(int e = 0; e < 6; ++e)
                for(int t = 0; t < per_expert; ++t) in.expert[std::size_t((l * 6 + e) * per_expert + t)] = (l * 6 + e) % int(experts);
        std::vector<float> out(in.expert.size() * hidden);
        tier.reset_stats();
        const auto t0 = std::chrono::steady_clock::now();
        for(int l = 0; l < layers; ++l) {
            const std::size_t first = std::size_t(l * 6 * per_expert);
            tier.compute({6 * per_expert, &in.expert[first], &in.values[first * hidden], hidden, &in.scales[first * hidden / 32],
                          hidden / 32, 32, &out[first * hidden], hidden});
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "  " << pool.workers() + 1 << " threads, 6 experts x " << per_expert << " tokens a layer: "
                  << s * 1000 / layers << " ms a layer, " << tier.stats().weight_bytes / s / 1e9 << " GB/s of experts\n";
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        check_pool();
        check_pool_races(argc > 1 ? std::atoi(argv[1]) : 300);
        check_tier();
        check_inputs_and_row_posts();
        throughput();
        std::cout << "host expert tier tests passed\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
