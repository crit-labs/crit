// W4A16 projections against host references: GPTQ (act-order, both zero conventions), AWQ (its column interleave) and
// compressed-tensors checkpoints convert to the same canonical bytes a host packer writes; the decode GEMV for 1 to 8
// tokens and the prompt GEMM equal an FP64 product over the represented weights (GEMV to FP32 rounding, GEMM to BF16
// weights), for int4 groups of 32, 64 and 128 and for MXFP4; a token alone equals the same token in a batch; invalid
// shapes are rejected.
#include "w4a16.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace w4 = crit::w4a16;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }

template<class T> struct Device {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Device(std::size_t count) : n(count) { ok(cudaMalloc(&p, sizeof(T) * (count ? count : 1)), "malloc"); ok(cudaMemset(p, 0, sizeof(T) * (count ? count : 1)), "memset"); }
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

const float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
const int kAwqNibble[8] = {0, 4, 1, 5, 2, 6, 3, 7};

float bf16(float x) { return __bfloat162float(__float2bfloat16(x)); }

// An int4 checkpoint in canonical form plus its represented weights.
struct Int4 {
    int N, K, group;
    std::vector<std::uint8_t> q;          // [N][K] codes, canonical column order
    std::vector<float> scales;            // [N][G]
    std::vector<std::uint8_t> zeros;      // [N][G]
    std::vector<std::uint8_t> packed() const {
        std::vector<std::uint8_t> out(std::size_t(N) * K / 2);
        for(int n = 0; n < N; ++n)
            for(int k = 0; k < K; k += 2)
                out[(std::size_t(n) * K + std::size_t(k)) / 2] = std::uint8_t(q[std::size_t(n) * K + std::size_t(k)] | (q[std::size_t(n) * K + std::size_t(k) + 1] << 4));
        return out;
    }
    double weight(int n, int k) const {
        const std::size_t g = std::size_t(n) * (K / group) + std::size_t(k / group);
        return (double(q[std::size_t(n) * K + std::size_t(k)]) - double(zeros[g])) * double(scales[g]);
    }
};

Int4 random_int4(int N, int K, int group, std::mt19937& rng) {
    Int4 m{N, K, group, std::vector<std::uint8_t>(std::size_t(N) * K), std::vector<float>(std::size_t(N) * (K / group)),
           std::vector<std::uint8_t>(std::size_t(N) * (K / group))};
    std::uniform_int_distribution<int> code(0, 15), zero(1, 15);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for(auto& v : m.q) v = std::uint8_t(code(rng));
    for(auto& v : m.scales) v = __half2float(__float2half(scale(rng)));   // checkpoints store FP16 scales
    for(auto& v : m.zeros) v = std::uint8_t(zero(rng));
    return m;
}

std::vector<__nv_bfloat16> activations(int T, int K, std::mt19937& rng, std::vector<float>& host) {
    std::normal_distribution<float> d(0.f, 1.f);
    host.resize(std::size_t(T) * K);
    std::vector<__nv_bfloat16> x(host.size());
    for(std::size_t i = 0; i < host.size(); ++i) { host[i] = bf16(d(rng)); x[i] = __float2bfloat16(host[i]); }
    return x;
}

template<class Weight>
double check_product(const w4::Weights& w, Weight weight, int T, std::mt19937& rng, bool gemm, double tolerance, const char* what) {
    std::vector<float> xh;
    Device<__nv_bfloat16> x(activations(T, w.K, rng, xh));
    Device<float> y(std::size_t(T) * w.N);
    ok(gemm ? w4::enqueue_gemm(w, x.p, w.K, T, y.p, w.N, false, nullptr) : w4::enqueue_gemv(w, x.p, w.K, T, y.p, w.N, false, nullptr), what);
    const auto got = y.get();
    double worst = 0., top = 0.;
    for(int t = 0; t < T; ++t)
        for(int n = 0; n < w.N; ++n) {
            double acc = 0.;
            for(int k = 0; k < w.K; ++k) acc += (gemm ? double(bf16(float(weight(n, k)))) : weight(n, k)) * xh[std::size_t(t) * w.K + std::size_t(k)];
            worst = std::max(worst, std::abs(acc - got[std::size_t(t) * w.N + std::size_t(n)]));
            top = std::max(top, std::abs(acc));
        }
    const double rel = worst / std::max(top, 1e-30);
    require(rel < tolerance, std::string(what) + " differs from the FP64 product");
    return rel;
}

void case_formats(std::mt19937& rng) {
    constexpr int N = 128, K = 256, group = 64, G = K / group;
    const Int4 m = random_int4(N, K, group, rng);
    const auto want = m.packed();
    // GPTQ with act-order: checkpoint column k sits in group g_idx[k]; canonical column j = perm[j].
    std::vector<int> order(K);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<std::int32_t> g_idx(K), perm_want(K);
    for(int j = 0; j < K; ++j) { g_idx[std::size_t(order[std::size_t(j)])] = j / group; }
    for(int g = 0, j = 0; g < G; ++g)
        for(int k = 0; k < K; ++k) if(g_idx[std::size_t(k)] == g) perm_want[std::size_t(j++)] = k;
    for(int zero_offset = 0; zero_offset <= 1; ++zero_offset) {
        std::vector<std::int32_t> qweight(std::size_t(K / 8) * N, 0), qzeros(std::size_t(G) * (N / 8), 0);
        std::vector<__half> scales(std::size_t(G) * N);
        for(int n = 0; n < N; ++n)
            for(int j = 0; j < K; ++j) {   // canonical column j is checkpoint column perm_want[j]
                const int k = perm_want[std::size_t(j)];
                qweight[std::size_t(k / 8) * N + std::size_t(n)] |= std::int32_t(std::uint32_t(m.q[std::size_t(n) * K + std::size_t(j)]) << (4 * (k % 8)));
            }
        for(int g = 0; g < G; ++g)
            for(int n = 0; n < N; ++n) {
                qzeros[std::size_t(g) * (N / 8) + std::size_t(n / 8)] |=
                    std::int32_t(std::uint32_t(m.zeros[std::size_t(n) * G + std::size_t(g)] - zero_offset) << (4 * (n % 8)));
                scales[std::size_t(g) * N + std::size_t(n)] = __float2half(m.scales[std::size_t(n) * G + std::size_t(g)]);
            }
        Device<std::int32_t> dq(qweight), dz(qzeros), dg(g_idx), perm(K);
        Device<__half> ds(scales);
        Device<std::uint8_t> codes(want.size()), zeros(std::size_t(N) * G);
        Device<float> sc(std::size_t(N) * G);
        w4::Gptq in{dq.p, dz.p, ds.p, dg.p, zero_offset};
        ok(w4::enqueue_from_gptq(in, N, K, group, codes.p, sc.p, zeros.p, perm.p, nullptr), "from gptq");
        require(codes.get() == want && zeros.get() == m.zeros && sc.get() == m.scales, "GPTQ converts to the canonical layout");
        require(perm.get() == perm_want, "GPTQ act-order permutation");
    }
    // AWQ: packed along N with the interleave; no act-order.
    {
        std::vector<std::int32_t> qweight(std::size_t(K) * (N / 8), 0), qzeros(std::size_t(G) * (N / 8), 0);
        std::vector<__half> scales(std::size_t(G) * N);
        for(int n = 0; n < N; ++n) {
            const int slot = kAwqNibble[n % 8];
            for(int k = 0; k < K; ++k)
                qweight[std::size_t(k) * (N / 8) + std::size_t(n / 8)] |= std::int32_t(std::uint32_t(m.q[std::size_t(n) * K + std::size_t(k)]) << (4 * slot));
            for(int g = 0; g < G; ++g) {
                qzeros[std::size_t(g) * (N / 8) + std::size_t(n / 8)] |= std::int32_t(std::uint32_t(m.zeros[std::size_t(n) * G + std::size_t(g)]) << (4 * slot));
                scales[std::size_t(g) * N + std::size_t(n)] = __float2half(m.scales[std::size_t(n) * G + std::size_t(g)]);
            }
        }
        Device<std::int32_t> dq(qweight), dz(qzeros);
        Device<__half> ds(scales);
        Device<std::uint8_t> codes(want.size()), zeros(std::size_t(N) * G);
        Device<float> sc(std::size_t(N) * G);
        ok(w4::enqueue_from_awq(dq.p, dz.p, ds.p, N, K, group, codes.p, sc.p, zeros.p, nullptr), "from awq");
        require(codes.get() == want && zeros.get() == m.zeros && sc.get() == m.scales, "AWQ converts to the canonical layout");
    }
    // compressed-tensors pack-quantized with explicit zero points.
    {
        std::vector<std::int32_t> packed(std::size_t(N) * (K / 8), 0), zp(std::size_t(N) * G);
        std::vector<__nv_bfloat16> scale(std::size_t(N) * G);
        for(int n = 0; n < N; ++n) {
            for(int k = 0; k < K; ++k)
                packed[std::size_t(n) * (K / 8) + std::size_t(k / 8)] |= std::int32_t(std::uint32_t(m.q[std::size_t(n) * K + std::size_t(k)]) << (4 * (k % 8)));
            for(int g = 0; g < G; ++g) {
                zp[std::size_t(n) * G + std::size_t(g)] = m.zeros[std::size_t(n) * G + std::size_t(g)];
                scale[std::size_t(n) * G + std::size_t(g)] = __float2bfloat16(m.scales[std::size_t(n) * G + std::size_t(g)]);
            }
        }
        Device<std::int32_t> dp(packed), dzp(zp);
        Device<__nv_bfloat16> dsc(scale);
        Device<std::uint8_t> codes(want.size()), zeros(std::size_t(N) * G);
        Device<float> sc(std::size_t(N) * G);
        ok(w4::enqueue_from_compressed(dp.p, dsc.p, dzp.p, N, K, group, codes.p, sc.p, zeros.p, nullptr), "from compressed");
        const auto got_scales = sc.get();
        bool same = true;
        for(std::size_t i = 0; i < got_scales.size(); ++i) same = same && got_scales[i] == bf16(m.scales[i]);
        require(codes.get() == want && zeros.get() == m.zeros && same, "compressed-tensors converts to the canonical layout");
    }
    std::cout << "formats: GPTQ (act-order, both zero conventions), AWQ and compressed-tensors convert exactly\n";
}

void case_products(std::mt19937& rng) {
    for(const int group : {32, 64, 128}) {
        constexpr int N = 192, K = 1024 + 512;
        const Int4 m = random_int4(N, K, group, rng);
        Device<std::uint8_t> codes(m.packed()), zeros(m.zeros);
        Device<float> scales(m.scales);
        w4::Weights w;
        w.codec = w4::Codec::int4_groups; w.codes = codes.p; w.scales = scales.p; w.zeros = zeros.p; w.N = N; w.K = K; w.group = group;
        auto weight = [&](int n, int k) { return m.weight(n, k); };
        for(int T = 1; T <= 8; ++T) check_product(w, weight, T, rng, false, 1e-5, "int4 GEMV");
        const double e = check_product(w, weight, 100, rng, true, 1e-2, "int4 GEMM");
        // A token alone equals the same token inside a batch of five.
        std::vector<float> xh;
        Device<__nv_bfloat16> x(activations(5, K, rng, xh));
        Device<float> y5(std::size_t(5) * N), y1(N);
        ok(w4::enqueue_gemv(w, x.p, K, 5, y5.p, N, false, nullptr), "gemv batch");
        const auto batch = y5.get();
        for(int t = 0; t < 5; ++t) {
            ok(w4::enqueue_gemv(w, x.p + std::size_t(t) * K, K, 1, y1.p, N, false, nullptr), "gemv alone");
            const auto alone = y1.get();
            for(int n = 0; n < N; ++n) require(alone[std::size_t(n)] == batch[std::size_t(t) * N + std::size_t(n)], "a token alone differs from the batch");
        }
        std::cout << "int4 group " << group << ": GEMV for 1..8 tokens, GEMM " << e << " of max, token invariance\n";
    }
    {   // MXFP4: E2M1 codes with one E8M0 exponent per 32 columns.
        constexpr int N = 128, K = 2048;
        std::vector<std::uint8_t> codes(std::size_t(N) * K / 2), exps(std::size_t(N) * (K / 32));
        std::uniform_int_distribution<int> byte(0, 255), expo(120, 130);
        for(auto& b : codes) b = std::uint8_t(byte(rng));
        for(auto& e : exps) e = std::uint8_t(expo(rng));
        auto weight = [&](int n, int k) {
            const std::uint8_t b = codes[(std::size_t(n) * K + std::size_t(k)) / 2];
            const int code = k % 2 ? b >> 4 : b & 0xF;
            return double(std::ldexp(kE2M1[code], int(exps[std::size_t(n) * (K / 32) + std::size_t(k / 32)]) - 127));
        };
        Device<std::uint8_t> dc(codes), de(exps);
        w4::Weights w;
        w.codec = w4::Codec::mxfp4; w.codes = dc.p; w.exponents = de.p; w.N = N; w.K = K; w.group = 32;
        for(int T = 1; T <= 8; ++T) check_product(w, weight, T, rng, false, 1e-5, "MXFP4 GEMV");
        check_product(w, weight, 70, rng, true, 1e-2, "MXFP4 GEMM");
        std::cout << "MXFP4: GEMV for 1..8 tokens and GEMM equal the FP64 product\n";
    }
}

void case_rejects() {
    w4::Weights w;
    require(w4::enqueue_gemv(w, nullptr, 0, 1, nullptr, 0, false, nullptr) == cudaErrorInvalidValue, "empty weights");
    Device<std::uint8_t> codes(64 * 32);
    Device<float> scales(64);
    Device<std::uint8_t> zeros(64);
    w.codes = codes.p; w.scales = scales.p; w.zeros = zeros.p; w.N = 64; w.K = 64; w.group = 48;
    require(w4::enqueue_gemv(w, reinterpret_cast<const __nv_bfloat16*>(codes.p), 64, 1, scales.p, 64, false, nullptr) == cudaErrorInvalidValue,
            "a group of 48 is rejected");
    w.group = 64;
    require(w4::enqueue_gemv(w, reinterpret_cast<const __nv_bfloat16*>(codes.p), 64, 9, scales.p, 64, false, nullptr) == cudaErrorInvalidValue,
            "more than 8 tokens in the GEMV is rejected");
    w.N = 96;
    require(w4::enqueue_gemm(w, reinterpret_cast<const __nv_bfloat16*>(codes.p), 64, 4, scales.p, 96, false, nullptr) == cudaErrorInvalidValue,
            "a GEMM N that is not a multiple of 64 is rejected");
    std::cout << "rejects: empty weights, group sizes, GEMV width and GEMM shape\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::mt19937 rng(41);
        const std::string which = argc > 1 ? argv[1] : "all";
        if(which == "formats" || which == "all") case_formats(rng);
        if(which == "products" || which == "all") case_products(rng);
        if(which == "rejects" || which == "all") case_rejects();
        ok(cudaDeviceSynchronize(), "sync");
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
