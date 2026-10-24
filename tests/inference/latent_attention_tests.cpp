// Absorbed latent attention against an FP64 reference over the BF16 cache: DeepSeek-V3 widths (rank 512, rope 64,
// 16 heads) and a smaller geometry, contiguous and paged caches, sinks, three verify columns over several key splits,
// each column alone equal bit for bit to the same column in the verify, and rejected shapes.
#include "latent_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace la = crit::latent_attention;

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

std::vector<__nv_bfloat16> random_bf16(std::size_t n, std::mt19937& rng, std::vector<float>& host, float s) {
    std::normal_distribution<float> d(0.f, s);
    host.resize(n);
    std::vector<__nv_bfloat16> out(n);
    for(std::size_t i = 0; i < n; ++i) { out[i] = __float2bfloat16(d(rng)); host[i] = __bfloat162float(out[i]); }
    return out;
}

void run(int rank, int rope, int heads, bool paged, bool sinks, std::mt19937& rng) {
    const int W = rank + rope, L = 700, T = 3, pos0 = L - T, splits = 5, page = 16;
    std::vector<float> qh, ch;
    const auto q = random_bf16(std::size_t(T) * heads * W, rng, qh, 1.f), cache = random_bf16(std::size_t(L) * W, rng, ch, 1.f);
    // Paged: logical page i at physical page (i * 7) % pages of a larger pool.
    const int pages = (L + page - 1) / page, pool = pages * 2;
    std::vector<std::int32_t> table(std::size_t(pages));
    for(int i = 0; i < pages; ++i) table[std::size_t(i)] = (i * 7) % pool;
    std::vector<__nv_bfloat16> stored = cache;
    if(paged) {
        stored.assign(std::size_t(pool) * page * W, __float2bfloat16(0.f));
        for(int j = 0; j < L; ++j)
            std::copy_n(cache.begin() + std::ptrdiff_t(std::size_t(j) * W), W,
                        stored.begin() + std::ptrdiff_t((std::size_t(table[std::size_t(j / page)]) * page + std::size_t(j % page)) * W));
    }
    std::vector<float> sink(std::size_t(heads));
    for(int h = 0; h < heads; ++h) sink[std::size_t(h)] = 0.1f * float(h);
    Device<__nv_bfloat16> dq(q), dc(stored);
    Device<std::int32_t> dt(table);
    Device<float> ds(sink), out(std::size_t(T) * heads * rank), partial(la::partial_floats(T, heads, rank, splits));
    la::Problem p;
    p.q = dq.p; p.cache = dc.p; p.table = paged ? dt.p : nullptr; p.page_tokens = page; p.out = out.p; p.T = T; p.pos0 = pos0;
    p.heads = heads; p.rank = rank; p.rope = rope; p.scale = 1.f / std::sqrt(float(W)); p.sinks = sinks ? ds.p : nullptr;
    p.splits = splits; p.partial = partial.p;
    ok(la::enqueue(p, nullptr), "latent attention");
    const auto got = out.get();
    double worst = 0., top = 0.;
    for(int t = 0; t < T; ++t)
        for(int h = 0; h < heads; ++h) {
            std::vector<double> s(std::size_t(pos0 + t + 1));
            double mx = sinks ? double(sink[std::size_t(h)]) : -1e300;
            for(int j = 0; j <= pos0 + t; ++j) {
                double acc = 0.;
                for(int x = 0; x < W; ++x) acc += double(qh[(std::size_t(t) * heads + std::size_t(h)) * W + std::size_t(x)]) * ch[std::size_t(j) * W + std::size_t(x)];
                s[std::size_t(j)] = acc * p.scale;
                mx = std::max(mx, s[std::size_t(j)]);
            }
            double den = sinks ? std::exp(double(sink[std::size_t(h)]) - mx) : 0.;
            for(auto& v : s) { v = std::exp(v - mx); den += v; }
            for(int d = 0; d < rank; ++d) {
                double num = 0.;
                for(int j = 0; j <= pos0 + t; ++j) num += s[std::size_t(j)] * ch[std::size_t(j) * W + std::size_t(d)];
                const double want = num / den;
                worst = std::max(worst, std::abs(want - got[(std::size_t(t) * heads + std::size_t(h)) * rank + std::size_t(d)]));
                top = std::max(top, std::abs(want));
            }
        }
    require(worst / top < 1e-4, "latent attention differs from the reference");
    // Each verify column alone equals the same column of the verify.
    for(int t = 0; t < T; ++t) {
        Device<__nv_bfloat16> one(std::vector<__nv_bfloat16>(q.begin() + std::ptrdiff_t(std::size_t(t) * heads * W),
                                                             q.begin() + std::ptrdiff_t(std::size_t(t + 1) * heads * W)));
        Device<float> o1(std::size_t(heads) * rank), part1(la::partial_floats(1, heads, rank, splits));
        la::Problem single = p;
        single.q = one.p; single.out = o1.p; single.T = 1; single.pos0 = pos0 + t; single.partial = part1.p;
        ok(la::enqueue(single, nullptr), "single column");
        const auto alone = o1.get();
        for(std::size_t i = 0; i < alone.size(); ++i)
            require(alone[i] == got[std::size_t(t) * heads * rank + i], "a latent-attention column alone differs from the verify column");
    }
    std::cout << "latent attention rank " << rank << " rope " << rope << " heads " << heads << (paged ? " paged" : " contiguous")
              << (sinks ? " with sinks" : "") << ": " << worst / top << " of max, columns invariant\n";
}
}  // namespace

int main() {
    try {
        std::mt19937 rng(43);
        run(512, 64, 16, false, false, rng);
        run(512, 64, 16, true, true, rng);
        run(128, 32, 8, true, false, rng);
        la::Problem bad;
        require(la::enqueue(bad, nullptr) == cudaErrorInvalidValue, "an empty problem is rejected");
        ok(cudaDeviceSynchronize(), "sync");
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
