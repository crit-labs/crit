// Flash-style causal prompt attention against an FP64 softmax reference (grouped-query, chunk offsets).
#include "flash_prefill.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace fp = crit::flash_prefill;

namespace {

bool run_case(int head_dim, int heads, int kv_heads, int T, int pos0, std::mt19937& rng) {
    const int L = pos0 + T, qs = heads * head_dim, kvs = kv_heads * head_dim;
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<__nv_bfloat16> q(static_cast<std::size_t>(T) * qs), k(static_cast<std::size_t>(L) * kvs), v(k.size());
    for(auto& x : q) x = __float2bfloat16(normal(rng));
    for(auto& x : k) x = __float2bfloat16(normal(rng));
    for(auto& x : v) x = __float2bfloat16(normal(rng));
    const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    std::vector<double> want(static_cast<std::size_t>(T) * qs, 0.0);
    for(int t = 0; t < T; ++t)
        for(int h = 0; h < heads; ++h) {
            const int g = h / (heads / kv_heads), visible = pos0 + t + 1;
            std::vector<double> s(visible);
            double mx = -1e300;
            for(int j = 0; j < visible; ++j) {
                double acc = 0.0;
                for(int d = 0; d < head_dim; ++d)
                    acc += static_cast<double>(__bfloat162float(q[static_cast<std::size_t>(t) * qs + h * head_dim + d])) *
                           __bfloat162float(k[static_cast<std::size_t>(j) * kvs + g * head_dim + d]);
                s[j] = acc * scale;
                mx = std::fmax(mx, s[j]);
            }
            double sum = 0.0;
            for(int j = 0; j < visible; ++j) sum += s[j] = std::exp(s[j] - mx);
            for(int d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for(int j = 0; j < visible; ++j) acc += s[j] * __bfloat162float(v[static_cast<std::size_t>(j) * kvs + g * head_dim + d]);
                want[static_cast<std::size_t>(t) * qs + h * head_dim + d] = acc / sum;
            }
        }
    __nv_bfloat16 *dq = nullptr, *dk = nullptr, *dv = nullptr;
    float* dout = nullptr;
    cudaMalloc(&dq, q.size() * 2); cudaMalloc(&dk, k.size() * 2); cudaMalloc(&dv, v.size() * 2);
    cudaMalloc(&dout, want.size() * 4);
    cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice);
    const fp::Problem problem{dq, qs, dk, dv, kvs, dout, qs, T, pos0, heads, kv_heads, head_dim, scale};
    const cudaError_t launched = fp::enqueue(problem, nullptr);
    std::vector<float> got(want.size());
    const bool ok = launched == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
                    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
    if(!ok) { std::printf("launch failed: %s\n", cudaGetErrorString(launched)); return false; }
    double worst = 0.0, scale_out = 0.0;
    for(std::size_t i = 0; i < want.size(); ++i) scale_out = std::fmax(scale_out, std::fabs(want[i]));
    for(std::size_t i = 0; i < want.size(); ++i) worst = std::fmax(worst, std::fabs(got[i] - want[i]));
    const double rel = worst / scale_out;
    std::printf("head_dim %d heads %d/%d T %d pos0 %d: max abs err %.2e (%.2e of max |out|)\n", head_dim, heads, kv_heads, T, pos0, worst, rel);
    return rel < 1e-2;
}

}  // namespace

int main() {
    int count = 0;
    if(cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { std::printf("no CUDA device; skipped\n"); return 0; }
    std::mt19937 rng(3);
    bool ok = true;
    for(const int hd : {64, 128, 256}) {
        ok = run_case(hd, 8, 2, 1, 0, rng) && ok;
        ok = run_case(hd, 8, 2, 37, 0, rng) && ok;
        ok = run_case(hd, 6, 1, 130, 100, rng) && ok;  // a later chunk: keys from earlier chunks plus its own
    }
    ok = run_case(256, 4, 4, 65, 31, rng) && ok;       // multi-head (no grouping)
    const fp::Problem bad{nullptr, 0, nullptr, nullptr, 0, nullptr, 0, 1, 0, 3, 2, 96, 1.f};
    ok = fp::enqueue(bad, nullptr) == cudaErrorInvalidValue && ok;
    std::printf(ok ? "flash prefill passed\n" : "flash prefill FAILED\n");
    return ok ? 0 : 1;
}
