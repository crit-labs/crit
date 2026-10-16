// CUDA row log-softmax / NLL / KL divergence against an FP64 host reference.
// Built with nvcc by tests/test_eval_distributions.py (crit_eval.logprob.build); CUDA runtime only.
#include "logprob_compare.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace dc = crit::eval::distribution;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
void check(cudaError_t s, const char* what) { require(s == cudaSuccess, std::string(what) + ": " + cudaGetErrorString(s)); }
struct Buffer {
    void* p = nullptr;
    explicit Buffer(std::size_t bytes) { check(cudaMalloc(&p, std::max<std::size_t>(bytes, 1)), "cudaMalloc"); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void* data() const { return p; }
};
struct Fixture {
    cudaStream_t stream{};
    std::vector<std::unique_ptr<Buffer>> owned;
    Fixture() { check(cudaStreamCreate(&stream), "cudaStreamCreate"); }
    ~Fixture() { owned.clear(); cudaStreamDestroy(stream); }
    template<class T> Buffer& put(const std::vector<T>& v) {
        owned.push_back(std::make_unique<Buffer>(v.size() * sizeof(T)));
        if(!v.empty()) check(cudaMemcpyAsync(owned.back()->data(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
        return *owned.back();
    }
    Buffer& zeros(std::size_t floats) { return put(std::vector<float>(floats, 0.0f)); }
    template<class T> std::vector<T> get(const Buffer& b, std::size_t n) {
        std::vector<T> out(n);
        check(cudaMemcpyAsync(out.data(), b.data(), n * sizeof(T), cudaMemcpyDeviceToHost, stream), "download");
        check(cudaStreamSynchronize(stream), "synchronize");
        return out;
    }
    void run(const dc::RowCompare& p) { check(dc::enqueue_row_compare(p, stream), "launch"); }  // downloads are ordered on the same stream
};
std::vector<double> host_logprobs(const float* row, std::uint32_t valid) {
    double top = -INFINITY;
    for(std::uint32_t i = 0; i < valid; ++i) top = std::max(top, double(row[i]));
    double sum = 0;
    for(std::uint32_t i = 0; i < valid; ++i) sum += std::exp(double(row[i]) - top);
    std::vector<double> lp(valid);
    for(std::uint32_t i = 0; i < valid; ++i) lp[i] = double(row[i]) - top - std::log(sum);
    return lp;
}
}

int main() {
    try {
        Fixture f;
        constexpr std::uint32_t rows = 37, valid = 1000, stride = 1003;
        std::mt19937 random(7);
        std::normal_distribution<float> normal(0.0f, 3.0f);
        std::vector<float> reference(std::size_t(rows) * stride), candidate(reference.size());
        std::vector<std::int32_t> targets(rows);
        for(std::uint32_t r = 0; r < rows; ++r) {
            for(std::uint32_t i = 0; i < stride; ++i) {
                reference[r * stride + i] = i < valid ? normal(random) : 1e30f;  // padding is never read
                candidate[r * stride + i] = i < valid ? reference[r * stride + i] + 0.05f * normal(random) : -1e30f;
            }
            targets[r] = std::int32_t(random() % valid);
        }
        auto &ref_d = f.put(reference), &cand_d = f.put(candidate), &targets_d = f.put(targets);
        auto &stored = f.zeros(std::size_t(rows) * valid), &nll_ref = f.zeros(rows), &nll_cand = f.zeros(rows), &kld = f.zeros(rows), &kld_self = f.zeros(rows);
        auto* lr = static_cast<float*>(ref_d.data());
        auto* lc = static_cast<float*>(cand_d.data());
        auto* tg = static_cast<const std::int32_t*>(targets_d.data());
        // Reference pass stores log-probabilities; candidate and self passes compare against them.
        f.run({lr, stride, rows, valid, tg, static_cast<float*>(nll_ref.data()), static_cast<float*>(stored.data()), nullptr, nullptr});
        f.run({lc, stride, rows, valid, tg, static_cast<float*>(nll_cand.data()), nullptr, static_cast<const float*>(stored.data()), static_cast<float*>(kld.data())});
        f.run({lr, stride, rows, valid, nullptr, nullptr, nullptr, static_cast<const float*>(stored.data()), static_cast<float*>(kld_self.data())});
        const auto nr = f.get<float>(nll_ref, rows), nc = f.get<float>(nll_cand, rows), kl = f.get<float>(kld, rows), ks = f.get<float>(kld_self, rows);
        const auto st = f.get<float>(stored, std::size_t(rows) * valid);
        for(std::uint32_t r = 0; r < rows; ++r) {
            const auto pr = host_logprobs(&reference[r * stride], valid), pc = host_logprobs(&candidate[r * stride], valid);
            double expect_kl = 0;
            for(std::uint32_t i = 0; i < valid; ++i) {
                expect_kl += std::exp(pr[i]) * (pr[i] - pc[i]);
                require(std::abs(st[r * valid + i] - pr[i]) < 2e-5, "stored log-probability");
            }
            require(std::abs(nr[r] + pr[targets[r]]) < 2e-5 && std::abs(nc[r] + pc[targets[r]]) < 2e-5, "target NLL");
            require(std::abs(kl[r] - expect_kl) < 2e-5 + 1e-3 * expect_kl, "KL divergence");
            require(ks[r] >= 0.0f && ks[r] < 2e-5, "self KL divergence");
        }
        // Deterministic: a second launch is bitwise identical.
        auto& kld2 = f.zeros(rows);
        f.run({lc, stride, rows, valid, nullptr, nullptr, nullptr, static_cast<const float*>(stored.data()), static_cast<float*>(kld2.data())});
        const auto kl2 = f.get<float>(kld2, rows);
        require(std::memcmp(kl.data(), kl2.data(), rows * sizeof(float)) == 0, "repeat launch differs");
        // Out-of-range targets give NaN; invalid geometry and pairings are rejected.
        auto& bad_targets = f.put(std::vector<std::int32_t>{-1, std::int32_t(valid)});
        auto& nll_bad = f.zeros(2);
        f.run({lr, stride, 2, valid, static_cast<const std::int32_t*>(bad_targets.data()), static_cast<float*>(nll_bad.data()), nullptr, nullptr, nullptr});
        const auto nb = f.get<float>(nll_bad, 2);
        require(std::isnan(nb[0]) && std::isnan(nb[1]), "out-of-range target");
        const auto s = f.stream;
        require(dc::enqueue_row_compare({lr, valid - 1, rows, valid, tg, static_cast<float*>(nll_ref.data())}, s) == cudaErrorInvalidValue, "short stride");
        require(dc::enqueue_row_compare({lr, stride, rows, valid, tg, nullptr}, s) == cudaErrorInvalidValue, "targets without output");
        require(dc::enqueue_row_compare({lr, stride, rows, valid}, s) == cudaErrorInvalidValue, "nothing to produce");
        require(dc::enqueue_row_compare({lr, stride, 0, valid, tg, static_cast<float*>(nll_ref.data())}, s) == cudaSuccess, "empty launch");
        std::cout << "distribution comparison ok\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
