// CUDA gather-argmax over a draft vocabulary subset against the full head.
#include "subset_argmax.h"
#include <crit/cuda.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace crit::core;
namespace ds = crit::techniques::draft_subset;

namespace {
void check(cudaError_t s, const char* w) { if(s != cudaSuccess) throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(s)); }
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
struct Fixture {
    Resources resources{1 << 26, 1 << 28};
    std::shared_ptr<CudaBackend> backend = std::make_shared<CudaBackend>();
    Fixture() { backend->initialize(resources); }
    template<class T> CudaBuffer put(const std::vector<T>& v) {
        auto b = backend->allocate(Memory::device, std::max<std::size_t>(v.size(), 1) * sizeof(T));
        if(!v.empty()) b.upload(std::as_bytes(std::span(v)), 0, Queue::compute);
        return b;
    }
    template<class T> std::vector<T> get(const CudaBuffer& b, std::size_t n) {
        std::vector<T> out(n);
        b.download(std::as_writable_bytes(std::span(out)), 0, Queue::compute);
        return out;
    }
};
// Frequency-ranked subset of `size` ascending IDs (Zipf-like synthetic counts).
std::vector<std::int32_t> choose(std::uint32_t vocabulary, std::uint32_t size, std::mt19937& random) {
    std::vector<std::int32_t> ids(vocabulary);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), random); // rank r has count ~ 1/(r+1)
    ids.resize(size);
    std::sort(ids.begin(), ids.end());
    return ids;
}

void subset() {
    Fixture f;
    std::mt19937 random(9);
    std::normal_distribution<float> normal(0, 1);
    constexpr std::uint32_t vocabulary = 1000, hidden = 64, batch = 512, stride = 72;
    std::vector<float> weights(std::size_t(vocabulary) * stride, 0), states(std::size_t(batch) * hidden);
    for(std::uint32_t r = 0; r < vocabulary; ++r) for(std::uint32_t h = 0; h < hidden; ++h) weights[r * stride + h] = normal(random);
    for(auto& x : states) x = normal(random);
    // Exact ties: duplicate rows so first-maximum must pick the lowest ID.
    for(std::uint32_t h = 0; h < hidden; ++h) weights[700 * stride + h] = weights[300 * stride + h] = weights[10 * stride + h] *= 4.0f;
    for(std::uint32_t h = 0; h < hidden; ++h) states[h] = weights[10 * stride + h]; // row 0 strongly prefers 10/300/700
    auto d_w = f.put(weights), d_s = f.put(states);
    auto d_full_tokens = f.backend->allocate(Memory::device, batch * 4), d_full_logits = f.backend->allocate(Memory::device, std::size_t(batch) * vocabulary * 4);
    ds::GatherArgmax full{static_cast<const float*>(d_w.data()), stride, hidden, nullptr, vocabulary,
                          static_cast<const float*>(d_s.data()), batch, static_cast<std::int32_t*>(d_full_tokens.data()),
                          static_cast<float*>(d_full_logits.data())};
    check(ds::enqueue_gather_argmax(full, f.backend->stream()), "full head");
    const auto full_tokens = f.get<std::int32_t>(d_full_tokens, batch);
    const auto full_logits = f.get<float>(d_full_logits, std::size_t(batch) * vocabulary);
    require(full_tokens[0] == 10, "tie must select the lowest token ID");
    for(std::uint32_t b = 0; b < batch; ++b) {
        const auto* row = full_logits.data() + std::size_t(b) * vocabulary;
        require(full_tokens[b] == std::max_element(row, row + vocabulary) - row, "full argmax is not the first maximum");
        for(std::uint32_t t = 0; t < vocabulary; t += 97) {
            double ref = 0;
            for(std::uint32_t h = 0; h < hidden; ++h) ref += double(weights[t * stride + h]) * states[b * hidden + h];
            require(std::abs(row[t] - ref) <= 1e-4 * (1 + std::abs(ref)), "full-head logit differs from FP64 dot");
        }
    }
    std::size_t inside = 0;
    for(std::uint32_t size : {1u, 37u, 128u, 400u, vocabulary}) {
        auto rows = size == vocabulary ? [&]{ std::vector<std::int32_t> all(vocabulary); std::iota(all.begin(), all.end(), 0); return all; }()
                                       : choose(vocabulary, size, random);
        if(size >= 37) { // keep the tie rows inside so the tie rule is exercised through the subset
            for(auto t : {10, 300, 700}) if(!std::binary_search(rows.begin(), rows.end(), t)) { rows.back() = t; std::sort(rows.begin(), rows.end()); }
            rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        }
        const auto count = std::uint32_t(rows.size());
        auto d_rows = f.put(rows);
        auto d_tokens = f.backend->allocate(Memory::device, batch * 4), d_logits = f.backend->allocate(Memory::device, std::size_t(batch) * count * 4);
        ds::GatherArgmax gathered{static_cast<const float*>(d_w.data()), stride, hidden, static_cast<const std::int32_t*>(d_rows.data()), count,
                                  static_cast<const float*>(d_s.data()), batch, static_cast<std::int32_t*>(d_tokens.data()),
                                  static_cast<float*>(d_logits.data())};
        check(ds::enqueue_gather_argmax(gathered, f.backend->stream()), "subset head");
        const auto tokens = f.get<std::int32_t>(d_tokens, batch);
        const auto logits = f.get<float>(d_logits, std::size_t(batch) * count);
        for(std::uint32_t b = 0; b < batch; ++b) {
            // Row gather, no requantization: every gathered logit is the full-head logit bit for bit.
            for(std::uint32_t i = 0; i < count; ++i)
                require(std::memcmp(&logits[std::size_t(b) * count + i], &full_logits[std::size_t(b) * vocabulary + std::size_t(rows[i])], 4) == 0,
                        "gathered logit differs from the full head");
            require(std::binary_search(rows.begin(), rows.end(), tokens[b]), "subset argmax mapped outside the subset");
            const auto* row = logits.data() + std::size_t(b) * count;
            require(tokens[b] == rows[std::size_t(std::max_element(row, row + count) - row)], "subset argmax is not the first maximum");
            if(std::binary_search(rows.begin(), rows.end(), full_tokens[b])) {
                require(tokens[b] == full_tokens[b], "subset draft differs from target although the target token is in the subset");
                ++inside;
            }
        }
        if(size == vocabulary) require(tokens == full_tokens, "identity subset differs from full head");
    }
    require(inside > 600, "too few target tokens inside subsets");
    std::cout << "batch " << batch << " vocabulary " << vocabulary << " subset hits " << inside << '\n';
    ds::GatherArgmax bad = full;
    bad.row_stride = hidden - 1;
    require(ds::enqueue_gather_argmax(bad, f.backend->stream()) == cudaErrorInvalidValue, "short row stride accepted");
}
}

int main() {
    try { subset(); std::cout << "PASS subset\n"; return 0; }
    catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
