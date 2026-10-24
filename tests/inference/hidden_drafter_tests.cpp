// Hidden-state drafter glue on the device against drafting.py's rules: EAGLE-3 catch-up columns from the round's
// packed result and (truncated) emitted count, block-drafter input, draft-vocabulary mapping and law scatter, and
// confidence truncation that leaves the captured width unchanged.
#include "greedy_accept.h"
#include "hidden_drafters.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hd = crit::techniques::hidden_drafters;
namespace dr = crit::techniques::device_round;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }

template<class T> struct Device {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Device(std::size_t count) : n(count) { ok(cudaMalloc(&p, sizeof(T) * (count ? count : 1)), "malloc"); }
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
}  // namespace

int main() {
    try {
        constexpr std::uint32_t K = 4, R = 3, W = dr::result_width(K);
        // Three requests that emitted 1, 3 and 5 tokens (all of K + 1) from base frontiers 20, 40, 60.
        std::vector<std::int32_t> control(R * dr::control::fields, 0), result(R * W, -1);
        const std::int32_t emitted[R] = {1, 3, 5};
        for(std::uint32_t r = 0; r < R; ++r) {
            control[r * dr::control::fields + dr::control::n_emitted] = emitted[r];
            control[r * dr::control::fields + dr::control::base_frontier] = std::int32_t(20 * (r + 1));
            control[r * dr::control::fields + dr::control::frontier] = std::int32_t(20 * (r + 1)) + emitted[r];
            control[r * dr::control::fields + dr::control::last_token] = 100 + std::int32_t(r);
            result[r * W] = emitted[r];
            for(std::int32_t i = 0; i < emitted[r]; ++i) result[r * W + 1 + std::uint32_t(i)] = 10 * std::int32_t(r) + i;
        }
        Device<std::int32_t> dctl(control), dres(result), tokens(R * (K + 1)), positions(R * (K + 1));
        ok(hd::enqueue_eagle_ingest(dctl.p, dres.p, K, tokens.p, positions.p, R, nullptr), "ingest");
        const auto t = tokens.get(), p = positions.get();
        for(std::uint32_t r = 0; r < R; ++r)
            for(std::uint32_t c = 0; c <= K; ++c) {
                const bool live = std::int32_t(c) < emitted[r];
                require(t[r * (K + 1) + c] == (live ? 10 * std::int32_t(r) + std::int32_t(c) : -1), "catch-up token: the c-th emitted token");
                require(p[r * (K + 1) + c] == (live ? std::int32_t(20 * (r + 1) + c) : -1), "catch-up position: base_frontier + c");
            }
        // Core trims request 2 to 2 tokens: the next ingest follows the truncated count.
        ok(dr::enqueue_truncate(dctl.p, dres.p, K, 2, 2, nullptr), "truncate");
        ok(hd::enqueue_eagle_ingest(dctl.p, dres.p, K, tokens.p, positions.p, R, nullptr), "ingest after trim");
        const auto t2 = tokens.get();
        require(t2[2 * (K + 1) + 1] == 21 && t2[2 * (K + 1) + 2] == -1, "ingest follows Core's truncated count");

        // Block input: anchor = pending token, then mask placeholders at frontier + i.
        constexpr std::uint32_t B = 6;
        Device<std::int32_t> bt(R * B), bp(R * B);
        ok(hd::enqueue_block_input(dctl.p, 7777, B, bt.p, bp.p, R, nullptr), "block input");
        const auto btok = bt.get(), bpos = bp.get(), ctl = dctl.get();
        for(std::uint32_t r = 0; r < R; ++r)
            for(std::uint32_t i = 0; i < B; ++i) {
                require(btok[r * B + i] == (i ? 7777 : ctl[r * dr::control::fields + dr::control::last_token]), "block tokens");
                require(bpos[r * B + i] == ctl[r * dr::control::fields + dr::control::frontier] + std::int32_t(i), "block positions");
            }

        // d2t: draft id d is target d + d2t[d]; laws scatter to those ids.
        const std::vector<std::int64_t> d2t = {0, 4, 7, 7, 20};   // targets 0, 5, 9, 10, 24
        Device<std::int64_t> dd2t(d2t);
        Device<std::int32_t> ids(std::vector<std::int32_t>{0, 1, 2, 3, 4, -1});
        ok(hd::enqueue_map_tokens(ids.p, 6, dd2t.p, 5, nullptr), "map");
        require(ids.get() == std::vector<std::int32_t>{0, 5, 9, 10, 24, -1}, "draft ids map through d2t");
        const std::vector<float> q = {0.1f, 0.2f, 0.3f, 0.15f, 0.25f, 0.5f, 0.5f, 0.f, 0.f, 0.f};
        Device<float> dq(q), out(2 * 30);
        ok(hd::enqueue_scatter_law(dq.p, 5, dd2t.p, out.p, 30, 2, nullptr), "scatter");
        const auto law = out.get();
        const std::uint32_t target[5] = {0, 5, 9, 10, 24};
        for(std::uint32_t r = 0; r < 2; ++r) {
            float sum = 0.f;
            for(std::uint32_t v = 0; v < 30; ++v) sum += law[r * 30 + v];
            require(sum > 0.999f && sum < 1.001f, "a scattered law keeps its mass");
            for(std::uint32_t d = 0; d < 5; ++d) require(law[r * 30 + target[d]] == q[r * 5 + d], "scattered probability");
        }

        // Confidence truncation: drafts from the first low-confidence one are -1.
        Device<std::int32_t> drafts(std::vector<std::int32_t>{1, 2, 3, 4, 5, 6, 7, 8});
        Device<float> conf(std::vector<float>{0.9f, 0.6f, 0.4f, 0.8f, 0.3f, 0.9f, 0.9f, 0.9f});
        ok(hd::enqueue_truncate_drafts(drafts.p, conf.p, 4, 0.5f, 2, nullptr), "truncate drafts");
        require(drafts.get() == std::vector<std::int32_t>{1, 2, -1, -1, -1, -1, -1, -1}, "drafts cut from the first low confidence");
        require(hd::enqueue_block_input(dctl.p, 1, 1, bt.p, bp.p, R, nullptr) == cudaErrorInvalidValue, "a block needs a mask slot");
        ok(cudaDeviceSynchronize(), "sync");
        std::cout << "hidden-state drafters: catch-up, block input, d2t mapping, law scatter and truncation\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
