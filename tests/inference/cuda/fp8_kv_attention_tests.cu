// FP8 KV cache attention against FP64 references over the dequantized cache, and decode column invariance.
#include "fp8_kv_attention.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace kv = crit::fp8_kv;

namespace {

float e4m3(std::uint8_t b) {
    const int sign = b >> 7, exponent = (b >> 3) & 15, mantissa = b & 7;
    const float magnitude = exponent ? std::ldexp(1.f + mantissa / 8.f, exponent - 7) : std::ldexp(mantissa / 8.f, -6);
    return sign ? -magnitude : magnitude;
}
float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }

struct Device {
    std::vector<void*> owned;
    template <class T> T* alloc(std::size_t n) { void* p = nullptr; cudaMalloc(&p, n * sizeof(T)); cudaMemset(p, 0, n * sizeof(T)); owned.push_back(p); return static_cast<T*>(p); }
    ~Device() { for(void* p : owned) cudaFree(p); }
};

struct Setup {
    int hd, heads, kv_heads, length;
    Device dev;
    kv::Cache cache;
    std::vector<__nv_bfloat16> k, v;
    std::vector<float> kd, vd;  // the cache's values as the kernels see them: bf16(code * scale)
    float worst_write = 0.f;

    Setup(int hd_, int heads_, int kv_heads_, int length_, std::mt19937& rng) : hd(hd_), heads(heads_), kv_heads(kv_heads_), length(length_) {
        const std::size_t n = static_cast<std::size_t>(length) * kv_heads * hd;
        std::normal_distribution<float> normal(0.f, 1.f);
        k.resize(n); v.resize(n);
        for(auto& x : k) x = __float2bfloat16(normal(rng));
        for(auto& x : v) x = __float2bfloat16(2.f * normal(rng));
        cache = {dev.alloc<std::uint8_t>(n), dev.alloc<std::uint8_t>(n), dev.alloc<float>(length * kv_heads), dev.alloc<float>(length * kv_heads), kv_heads, hd};
        auto* dk = dev.alloc<__nv_bfloat16>(n);
        auto* dv = dev.alloc<__nv_bfloat16>(n);
        cudaMemcpy(dk, k.data(), n * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, v.data(), n * 2, cudaMemcpyHostToDevice);
        kv::write(cache, dk, dv, kv_heads * hd, length, 0, nullptr);  // positions 0 .. length - 1 in one call
        std::vector<std::uint8_t> kc(n), vc(n);
        std::vector<float> ks(length * kv_heads), vs(length * kv_heads);
        cudaMemcpy(kc.data(), cache.k, n, cudaMemcpyDeviceToHost);
        cudaMemcpy(vc.data(), cache.v, n, cudaMemcpyDeviceToHost);
        cudaMemcpy(ks.data(), cache.k_scale, ks.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(vs.data(), cache.v_scale, vs.size() * 4, cudaMemcpyDeviceToHost);
        kd.resize(n); vd.resize(n);
        for(std::size_t i = 0; i < n; ++i) {
            const std::size_t row = i / hd;
            kd[i] = bf16_round(e4m3(kc[i]) * ks[row]);
            vd[i] = bf16_round(e4m3(vc[i]) * vs[row]);
            // E4M3 keeps 3 mantissa bits: relative error at most 1/16 of the value, plus the subnormal step
            const float bound_k = std::fabs(__bfloat162float(k[i])) / 15.f + ks[row] * 0.002f;
            const float bound_v = std::fabs(__bfloat162float(v[i])) / 15.f + vs[row] * 0.002f;
            worst_write = std::fmax(worst_write, std::fmax(std::fabs(kd[i] - __bfloat162float(k[i])) / bound_k,
                                                           std::fabs(vd[i] - __bfloat162float(v[i])) / bound_v));
        }
    }

    // FP64 attention of query row (heads x hd) at position pos over the dequantized cache
    std::vector<double> attend(const std::vector<__nv_bfloat16>& q, std::size_t row, int pos, float scale) const {
        std::vector<double> out(static_cast<std::size_t>(heads) * hd, 0.0);
        for(int h = 0; h < heads; ++h) {
            const int g = h / (heads / kv_heads);
            std::vector<double> s(pos + 1);
            double mx = -1e300;
            for(int j = 0; j <= pos; ++j) {
                double acc = 0.0;
                for(int d = 0; d < hd; ++d)
                    acc += static_cast<double>(__bfloat162float(q[row * heads * hd + h * hd + d])) * kd[(static_cast<std::size_t>(j) * kv_heads + g) * hd + d];
                s[j] = acc * scale;
                mx = std::fmax(mx, s[j]);
            }
            double sum = 0.0;
            for(auto& x : s) sum += x = std::exp(x - mx);
            for(int d = 0; d < hd; ++d) {
                double acc = 0.0;
                for(int j = 0; j <= pos; ++j) acc += s[j] * vd[(static_cast<std::size_t>(j) * kv_heads + g) * hd + d];
                out[static_cast<std::size_t>(h) * hd + d] = acc / sum;
            }
        }
        return out;
    }
};

double rel_error(const std::vector<float>& got, const std::vector<double>& want) {
    double worst = 0.0, top = 0.0;
    for(std::size_t i = 0; i < want.size(); ++i) { top = std::fmax(top, std::fabs(want[i])); worst = std::fmax(worst, std::fabs(got[i] - want[i])); }
    return worst / top;
}

// fp8: prefill_fp8 (E4M3 queries and probabilities), checked against the same FP64 reference with a wider bound.
bool prefill_case(int hd, int heads, int kv_heads, int T, int pos0, std::mt19937& rng, bool fp8 = false) {
    Setup s(hd, heads, kv_heads, pos0 + T, rng);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<__nv_bfloat16> q(static_cast<std::size_t>(T) * heads * hd);
    for(auto& x : q) x = __float2bfloat16(normal(rng));
    auto* dq = s.dev.alloc<__nv_bfloat16>(q.size());
    auto* dout = s.dev.alloc<float>(q.size());
    cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    const kv::Prefill p{dq, heads * hd, s.cache, dout, heads * hd, T, pos0, heads, scale};
    const cudaError_t e = fp8 ? kv::prefill_fp8(p, nullptr) : kv::prefill(p, nullptr);
    std::vector<float> got(q.size());
    const bool ran = e == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
                     cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    if(!ran) { std::printf("prefill launch failed: %s\n", cudaGetErrorString(e)); return false; }
    std::vector<double> want;
    for(int t = 0; t < T; ++t) {
        const auto row = s.attend(q, t, pos0 + t, scale);
        want.insert(want.end(), row.begin(), row.end());
    }
    const double rel = rel_error(got, want);
    std::printf("prefill%s hd %d heads %d/%d T %d pos0 %d: %.2e of max |out|; write error %.2f of its bound\n", fp8 ? "_fp8" : "", hd, heads,
                kv_heads, T, pos0, rel, s.worst_write);
    // prefill_fp8 rounds queries and probabilities to E4M3 (3 mantissa bits): a few % of max |out| on random data,
    // where a layout error would be of order 1.
    return rel < (fp8 ? 1e-1 : 1e-2) && s.worst_write <= 1.f;
}

bool decode_case(int hd, int heads, int kv_heads, int T, int pos0, int blocks, std::mt19937& rng) {
    Setup s(hd, heads, kv_heads, pos0 + T, rng);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<__nv_bfloat16> q(static_cast<std::size_t>(T) * heads * hd);
    for(auto& x : q) x = __float2bfloat16(normal(rng));
    auto* dq = s.dev.alloc<__nv_bfloat16>(q.size());
    auto* dout = s.dev.alloc<float>(q.size());
    auto* dpos = s.dev.alloc<int>(1);
    auto* partial = s.dev.alloc<float>(kv::decode_partial_floats(kv_heads, hd, blocks));
    cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, &pos0, sizeof(int), cudaMemcpyHostToDevice);
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    // all T columns at once, position from device memory
    kv::Decode p{dq, heads * hd, s.cache, dout, heads * hd, T, 0, heads, dpos, scale, blocks, partial};
    cudaError_t e = kv::decode(p, nullptr);
    std::vector<float> got(q.size());
    if(e != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) { std::printf("decode launch failed: %s\n", cudaGetErrorString(e)); return false; }
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<double> want;
    for(int t = 0; t < T; ++t) {
        const auto row = s.attend(q, t, pos0 + t, scale);
        want.insert(want.end(), row.begin(), row.end());
    }
    const double rel = rel_error(got, want);
    // each column alone, at its own position: bit for bit the same as inside the multi-column call
    bool invariant = true;
    for(int t = 0; t < T; ++t) {
        kv::Decode one{dq + static_cast<std::size_t>(t) * heads * hd, heads * hd, s.cache, dout, heads * hd, 1, pos0 + t, heads, nullptr, scale, blocks, partial};
        e = kv::decode(one, nullptr);
        std::vector<float> alone(static_cast<std::size_t>(heads) * hd);
        if(e != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return false;
        cudaMemcpy(alone.data(), dout, alone.size() * 4, cudaMemcpyDeviceToHost);
        invariant = invariant && std::memcmp(alone.data(), got.data() + static_cast<std::size_t>(t) * heads * hd, alone.size() * 4) == 0;
    }
    std::printf("decode hd %d heads %d/%d T %d pos0 %d blocks %d: %.2e of max |out|, columns invariant %s\n", hd, heads, kv_heads, T, pos0, blocks, rel,
                invariant ? "yes" : "NO");
    return rel < 1e-2 && invariant;
}

}  // namespace

int main() {
    int count = 0;
    if(cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { std::printf("no CUDA device; skipped\n"); return 0; }
    std::mt19937 rng(7);
    bool ok = true;
    for(const int hd : {64, 128, 256}) {
        ok = prefill_case(hd, 8, 2, 1, 0, rng) && ok;
        ok = prefill_case(hd, 8, 2, 150, 0, rng) && ok;
        ok = prefill_case(hd, 6, 1, 70, 300, rng) && ok;   // a later chunk over earlier positions
        ok = decode_case(hd, 8, 2, 4, 0, 3, rng) && ok;
        ok = decode_case(hd, 6, 1, 8, 900, 3, rng) && ok;  // several 128-key pieces per block
    }
    for(const int hd : {64, 128, 256}) {
        ok = prefill_case(hd, 8, 2, 1, 0, rng, true) && ok;
        ok = prefill_case(hd, 8, 2, 150, 0, rng, true) && ok;     // several 64-key tiles, diagonal masking
        ok = prefill_case(hd, 6, 1, 70, 300, rng, true) && ok;    // a later chunk over earlier positions
    }
    ok = prefill_case(128, 4, 4, 33, 7, rng) && ok;       // multi-head (no grouping)
    ok = prefill_case(128, 4, 4, 33, 7, rng, true) && ok;
    ok = decode_case(128, 4, 4, 1, 4000, 7, rng) && ok;
    const kv::Cache none{};
    ok = kv::prefill(kv::Prefill{}, nullptr) == cudaErrorInvalidValue && ok;
    ok = kv::prefill_fp8(kv::Prefill{}, nullptr) == cudaErrorInvalidValue && ok;
    ok = kv::decode(kv::Decode{}, nullptr) == cudaErrorInvalidValue && ok;
    ok = kv::write(none, nullptr, nullptr, 0, 1, 0, nullptr) == cudaErrorInvalidValue && ok;
    std::printf(ok ? "fp8 kv attention passed\n" : "fp8 kv attention FAILED\n");
    return ok ? 0 : 1;
}
