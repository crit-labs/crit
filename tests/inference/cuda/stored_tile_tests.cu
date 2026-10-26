// Stored-tile FP8 / NVFP4 prompt GEMMs against an FP64 host reference with the same activation quantization.
#include "stored_tile_gemm.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace st = crit::stored_tile;

namespace {

#define CHECK(call)                                                                                        \
    do {                                                                                                   \
        const cudaError_t error_ = (call);                                                                 \
        if(error_ != cudaSuccess) {                                                                        \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(error_));            \
            return false;                                                                                  \
        }                                                                                                  \
    } while(0)

float e4m3(std::uint8_t code) { __nv_fp8_e4m3 v; v.__x = code; return static_cast<float>(v); }
std::uint8_t to_e4m3(float x) { return static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)); }
float e2m1(int code) {
    static const float magnitude[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (code & 8 ? -1.f : 1.f) * magnitude[code & 7];
}
int to_e2m1(float x) {
    const auto pair = static_cast<std::uint8_t>(__nv_cvt_float2_to_fp4x2(make_float2(x, 0.f), __NV_E2M1, cudaRoundNearest));
    return pair & 0x0F;
}

template <class T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    if(cudaMalloc(&device, host.size() * sizeof(T) + 16) != cudaSuccess) return nullptr;
    cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    return device;
}

double compare(const std::vector<float>& got, const std::vector<double>& want) {
    double worst = 0.0, scale = 0.0;
    for(std::size_t i = 0; i < want.size(); ++i) scale = std::fmax(scale, std::fabs(want[i]));
    for(std::size_t i = 0; i < want.size(); ++i) worst = std::fmax(worst, std::fabs(got[i] - want[i]) / (scale + 1e-30));
    return worst;
}

bool fp8_case(int N, int K, int T, bool add, std::mt19937& rng, int min_blocks = 340, int plan_tokens = 0) {
    std::uniform_int_distribution<int> code(0, 255);
    std::uniform_real_distribution<float> value(-2.f, 2.f);
    std::vector<std::uint8_t> w(static_cast<std::size_t>(N) * K);
    for(auto& c : w) {
        do c = static_cast<std::uint8_t>(code(rng)); while((c & 0x7F) == 0x7F);  // no NaN codes
    }
    std::vector<__nv_bfloat16> rs(N), x(static_cast<std::size_t>(T) * K);
    for(auto& v : rs) v = __float2bfloat16(0.01f + std::fabs(value(rng)) * 0.01f);
    for(auto& v : x) v = __float2bfloat16(value(rng));
    std::vector<float> y0(static_cast<std::size_t>(T) * N);
    for(auto& v : y0) v = add ? value(rng) : 0.f;
    // Host reference: per-token amax / 448 scales and E4M3 activations, FP64 dot products.
    std::vector<double> want(y0.begin(), y0.end());
    for(int t = 0; t < T; ++t) {
        float amax = 0.f;
        for(int k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(__bfloat162float(x[static_cast<std::size_t>(t) * K + k])));
        const float s = amax > 0.f ? amax / 448.f : 1.f;
        std::vector<double> q(K);
        for(int k = 0; k < K; ++k) q[k] = e4m3(to_e4m3(__bfloat162float(x[static_cast<std::size_t>(t) * K + k]) * (1.f / s)));
        for(int n = 0; n < N; ++n) {
            double acc = 0.0;
            for(int k = 0; k < K; ++k) acc += static_cast<double>(e4m3(w[static_cast<std::size_t>(n) * K + k])) * q[k];
            const double v = acc * __bfloat162float(rs[n]) * s;
            want[static_cast<std::size_t>(t) * N + n] = add ? want[static_cast<std::size_t>(t) * N + n] + v : v;
        }
    }
    const st::Plan p = st::plan(false, N, K, std::max(T, plan_tokens), 256, min_blocks);
    std::uint8_t* d_codes = upload(w);
    std::uint8_t* d_tiles = nullptr;
    CHECK(cudaMalloc(&d_tiles, w.size()));
    __nv_bfloat16* d_rs = upload(rs);
    __nv_bfloat16* d_x = upload(x);
    float* d_y = upload(y0);
    void* d_act = nullptr;
    float *d_ts = nullptr, *d_part = nullptr;
    CHECK(cudaMalloc(&d_act, p.activation_bytes));
    CHECK(cudaMalloc(&d_ts, (p.token_scale_count + 1) * sizeof(float)));
    CHECK(cudaMalloc(&d_part, (p.partial_floats + 1) * sizeof(float)));
    CHECK(st::enqueue_repack_fp8(d_codes, d_tiles, N, K, nullptr));
    CHECK(st::enqueue_fp8({d_tiles, d_rs, N, K}, d_x, T, d_y, N, add, p, {d_act, d_ts, d_part}, nullptr));
    CHECK(cudaDeviceSynchronize());
    std::vector<float> got(y0.size());
    CHECK(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for(void* ptr : {static_cast<void*>(d_codes), static_cast<void*>(d_tiles), static_cast<void*>(d_rs), static_cast<void*>(d_x),
                     static_cast<void*>(d_y), d_act, static_cast<void*>(d_ts), static_cast<void*>(d_part)}) cudaFree(ptr);
    const double err = compare(got, want);
    std::printf("fp8 N=%d K=%d T=%d add=%d nt=%d rt=%d sk=%d: max rel err %.2e\n", N, K, T, add, p.nt, p.rt, p.sk, err);
    return err < 1e-5;
}

bool fp4_case(int N, int K, int T, bool add, std::mt19937& rng, int min_blocks = 340, int plan_tokens = 0) {
    std::uniform_int_distribution<int> nibble(0, 15), scale_code(0x28, 0x48);
    std::uniform_real_distribution<float> value(-3.f, 3.f);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(N) * K / 2), scales(static_cast<std::size_t>(N) * K / 16);
    for(auto& c : codes) c = static_cast<std::uint8_t>(nibble(rng) | (nibble(rng) << 4));
    for(auto& s : scales) s = static_cast<std::uint8_t>(scale_code(rng));
    const float wgs = 448.f * 6.f / 3.f, gx = 448.f * 6.f / 4.f;
    std::vector<__nv_bfloat16> x(static_cast<std::size_t>(T) * K);
    for(auto& v : x) v = __float2bfloat16(value(rng));
    std::vector<float> y0(static_cast<std::size_t>(T) * N);
    for(auto& v : y0) v = add ? value(rng) : 0.f;
    std::vector<double> want(y0.begin(), y0.end());
    for(int t = 0; t < T; ++t) {
        std::vector<double> q(K);
        for(int b = 0; b < K / 16; ++b) {
            float amax = 0.f;
            for(int i = 0; i < 16; ++i) amax = std::fmax(amax, std::fabs(__bfloat162float(x[static_cast<std::size_t>(t) * K + 16 * b + i])));
            const float sv = e4m3(to_e4m3(amax * gx / 6.f));
            const float mul = sv > 0.f ? gx / sv : 0.f;
            for(int i = 0; i < 16; ++i) {
                const float xv = __bfloat162float(x[static_cast<std::size_t>(t) * K + 16 * b + i]);
                q[16 * b + i] = static_cast<double>(e2m1(to_e2m1(xv * mul))) * sv / gx;
            }
        }
        for(int n = 0; n < N; ++n) {
            double acc = 0.0;
            for(int k = 0; k < K; ++k) {
                const std::uint8_t byte = codes[(static_cast<std::size_t>(n) * K + k) / 2];
                const int c = k % 2 ? byte >> 4 : byte & 0x0F;
                acc += static_cast<double>(e2m1(c)) * e4m3(scales[static_cast<std::size_t>(n) * (K / 16) + k / 16]) / wgs * q[k];
            }
            want[static_cast<std::size_t>(t) * N + n] = add ? want[static_cast<std::size_t>(t) * N + n] + acc : acc;
        }
    }
    const st::Plan p = st::plan(true, N, K, std::max(T, plan_tokens), 256, min_blocks);
    std::uint8_t* d_codes = upload(codes);
    std::uint8_t* d_scales = upload(scales);
    std::uint8_t *d_tiles = nullptr, *d_stiles = nullptr;
    CHECK(cudaMalloc(&d_tiles, codes.size()));
    CHECK(cudaMalloc(&d_stiles, scales.size()));
    __nv_bfloat16* d_x = upload(x);
    float* d_y = upload(y0);
    void* d_act = nullptr;
    float* d_part = nullptr;
    CHECK(cudaMalloc(&d_act, p.activation_bytes));
    CHECK(cudaMalloc(&d_part, (p.partial_floats + 1) * sizeof(float)));
    CHECK(st::enqueue_repack_fp4(d_codes, d_scales, d_tiles, d_stiles, N, K, nullptr));
    CHECK(st::enqueue_fp4({d_tiles, d_stiles, wgs, gx, N, K}, d_x, T, d_y, N, add, p, {d_act, nullptr, d_part}, nullptr));
    CHECK(cudaDeviceSynchronize());
    std::vector<float> got(y0.size());
    CHECK(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for(void* ptr : {static_cast<void*>(d_codes), static_cast<void*>(d_scales), static_cast<void*>(d_tiles), static_cast<void*>(d_stiles),
                     static_cast<void*>(d_x), static_cast<void*>(d_y), d_act, static_cast<void*>(d_part)}) cudaFree(ptr);
    const double err = compare(got, want);
    std::printf("fp4 N=%d K=%d T=%d add=%d nt=%d rt=%d sk=%d: max rel err %.2e\n", N, K, T, add, p.nt, p.rt, p.sk, err);
    return err < 1e-5;
}

// One plan serves 1..T_max tokens and a launch computes only the token blocks its T needs: every token's bits must
// equal the T_max launch's, with and without the K split and for both token-block widths.
bool bits_independent_of_tokens(bool fp4, int Tmax, int min_blocks, std::mt19937& rng) {
    const int N = 256, K = 512;
    std::uniform_int_distribution<int> byte(0, 255), scale_code(0x28, 0x48);
    std::uniform_real_distribution<float> value(-2.f, 2.f);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(N) * K / (fp4 ? 2 : 1)), scales(fp4 ? static_cast<std::size_t>(N) * K / 16 : 0);
    for(auto& c : codes) {
        do c = static_cast<std::uint8_t>(byte(rng)); while(!fp4 && (c & 0x7F) == 0x7F);
    }
    for(auto& s : scales) s = static_cast<std::uint8_t>(scale_code(rng));
    std::vector<__nv_bfloat16> rs(N), x(static_cast<std::size_t>(Tmax) * K);
    for(auto& v : rs) v = __float2bfloat16(0.01f + std::fabs(value(rng)) * 0.01f);
    for(auto& v : x) v = __float2bfloat16(value(rng));
    const st::Plan p = st::plan(fp4, N, K, Tmax, 256, min_blocks);
    std::uint8_t* d_codes = upload(codes);
    std::uint8_t* d_scales = upload(scales);
    std::uint8_t *d_tiles = nullptr, *d_stiles = nullptr;
    CHECK(cudaMalloc(&d_tiles, codes.size()));
    CHECK(cudaMalloc(&d_stiles, scales.size() + 16));
    __nv_bfloat16* d_rs = upload(rs);
    __nv_bfloat16* d_x = upload(x);
    float *d_y = nullptr, *d_ts = nullptr, *d_part = nullptr;
    void* d_act = nullptr;
    CHECK(cudaMalloc(&d_y, static_cast<std::size_t>(Tmax) * N * sizeof(float)));
    CHECK(cudaMalloc(&d_act, p.activation_bytes));
    CHECK(cudaMalloc(&d_ts, (p.token_scale_count + 1) * sizeof(float)));
    CHECK(cudaMalloc(&d_part, (p.partial_floats + 1) * sizeof(float)));
    if(fp4) CHECK(st::enqueue_repack_fp4(d_codes, d_scales, d_tiles, d_stiles, N, K, nullptr));
    else CHECK(st::enqueue_repack_fp8(d_codes, d_tiles, N, K, nullptr));
    const auto run = [&](int T, std::vector<float>& out) {
        if(cudaMemset(d_y, 0xFF, static_cast<std::size_t>(Tmax) * N * sizeof(float)) != cudaSuccess) return false;
        const cudaError_t e = fp4 ? st::enqueue_fp4({d_tiles, d_stiles, 448.f * 6.f / 3.f, 448.f * 6.f / 4.f, N, K}, d_x, T, d_y, N, false, p,
                                                    {d_act, nullptr, d_part}, nullptr)
                                  : st::enqueue_fp8({d_tiles, d_rs, N, K}, d_x, T, d_y, N, false, p, {d_act, d_ts, d_part}, nullptr);
        out.resize(static_cast<std::size_t>(T) * N);
        return e == cudaSuccess && cudaMemcpy(out.data(), d_y, out.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    };
    std::vector<float> full, part;
    bool ok = run(Tmax, full);
    for(const int T : {1, 7, 8, 63, 64, 65, 129, 255, 299}) {
        if(T >= Tmax) continue;
        ok = run(T, part) && ok;
        ok = ok && std::memcmp(part.data(), full.data(), part.size() * sizeof(float)) == 0;
    }
    for(void* ptr : {static_cast<void*>(d_codes), static_cast<void*>(d_scales), static_cast<void*>(d_tiles), static_cast<void*>(d_stiles),
                     static_cast<void*>(d_rs), static_cast<void*>(d_x), static_cast<void*>(d_y), d_act, static_cast<void*>(d_ts),
                     static_cast<void*>(d_part)}) cudaFree(ptr);
    std::printf("%s plan T_max=%d nt=%d sk=%d: tokens of T = 1..%d launches %s the T_max launch's bits\n", fp4 ? "fp4" : "fp8", Tmax,
                p.nt, p.sk, Tmax - 1, ok ? "equal" : "DIFFER from");
    return ok;
}

// ------------------------------------------------------------------------------------------- shared fixtures
template <class T>
struct Dev {
    T* p = nullptr;
    explicit Dev(std::size_t count) {
        if(cudaMalloc(&p, count * sizeof(T) + 256) != cudaSuccess) p = nullptr;
    }
    explicit Dev(const std::vector<T>& h) : Dev(h.size()) {
        if(p && !h.empty()) cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    std::vector<T> get(std::size_t count) const {
        std::vector<T> h(count);
        cudaMemcpy(h.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost);
        return h;
    }
};

float bf(__nv_bfloat16 v) { return __bfloat162float(v); }

// Row-major matrices as a checkpoint stores them, with their exact weight values.
struct Fp8Host {
    int N = 0, K = 0;
    std::vector<std::uint8_t> codes;  // [N][K]
    std::vector<__nv_bfloat16> rs;    // [N]
    double w(int n, int k) const { return static_cast<double>(e4m3(codes[static_cast<std::size_t>(n) * K + k])) * bf(rs[n]); }
};
struct Fp4Host {
    int N = 0, K = 0;
    std::vector<std::uint8_t> codes, scales;  // [N][K / 2], [N][K / 16]
    float wgs = 1.f;
    double w(int n, int k) const {
        const std::uint8_t b = codes[(static_cast<std::size_t>(n) * K + k) / 2];
        return static_cast<double>(e2m1(k % 2 ? b >> 4 : b & 0x0F)) * e4m3(scales[static_cast<std::size_t>(n) * (K / 16) + k / 16]) / wgs;
    }
};
// rs_scale sets the size of FP8 results (random codes span E4M3's whole range).
Fp8Host random_fp8(int N, int K, float rs_scale, std::mt19937& rng) {
    std::uniform_int_distribution<int> code(0, 255);
    std::uniform_real_distribution<float> u(1.f, 3.f);
    Fp8Host h{N, K, std::vector<std::uint8_t>(static_cast<std::size_t>(N) * K), std::vector<__nv_bfloat16>(N)};
    for(auto& c : h.codes) {
        do c = static_cast<std::uint8_t>(code(rng)); while((c & 0x7F) == 0x7F);  // no NaN codes
    }
    for(auto& v : h.rs) v = __float2bfloat16(rs_scale * u(rng));
    return h;
}
Fp4Host random_fp4(int N, int K, float wgs, std::mt19937& rng) {
    std::uniform_int_distribution<int> nibble(0, 15), scale_code(0x28, 0x48);
    Fp4Host h{N, K, std::vector<std::uint8_t>(static_cast<std::size_t>(N) * K / 2), std::vector<std::uint8_t>(static_cast<std::size_t>(N) * K / 16), wgs};
    for(auto& c : h.codes) c = static_cast<std::uint8_t>(nibble(rng) | (nibble(rng) << 4));
    for(auto& s : h.scales) s = static_cast<std::uint8_t>(scale_code(rng));
    return h;
}
// Rows of `parts` stacked (interleave = false) or in alternating 16-row tiles of two parts (interleave = true).
template <class H>
H fuse(const std::vector<const H*>& parts, bool interleave) {
    H f = *parts[0];
    const int K = f.K;
    std::vector<std::pair<const H*, int>> rows;  // (part, row)
    if(interleave)
        for(int t = 0; t < parts[0]->N / 16; ++t)
            for(const H* q : parts)
                for(int r = 0; r < 16; ++r) rows.push_back({q, t * 16 + r});
    else
        for(const H* q : parts)
            for(int r = 0; r < q->N; ++r) rows.push_back({q, r});
    f.N = static_cast<int>(rows.size());
    f.codes.clear();
    if constexpr(std::is_same_v<H, Fp8Host>) f.rs.clear();
    else f.scales.clear();
    for(const auto& [q, r] : rows) {
        const std::size_t cb = std::is_same_v<H, Fp8Host> ? K : K / 2;
        f.codes.insert(f.codes.end(), q->codes.begin() + r * cb, q->codes.begin() + (r + 1) * cb);
        if constexpr(std::is_same_v<H, Fp8Host>) f.rs.push_back(q->rs[r]);
        else f.scales.insert(f.scales.end(), q->scales.begin() + static_cast<std::size_t>(r) * (K / 16),
                             q->scales.begin() + static_cast<std::size_t>(r + 1) * (K / 16));
    }
    return f;
}

// Activations as the GEMMs quantize them, dequantized (FP8: per-token amax / 448; NVFP4: static gx, 16-value groups).
std::vector<double> quantized_fp8(const __nv_bfloat16* x, int K) {
    float amax = 0.f;
    for(int k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(bf(x[k])));
    const float s = amax > 0.f ? amax / 448.f : 1.f;
    std::vector<double> q(K);
    for(int k = 0; k < K; ++k) q[k] = static_cast<double>(e4m3(to_e4m3(bf(x[k]) * (1.f / s)))) * s;
    return q;
}
std::vector<double> quantized_fp4(const __nv_bfloat16* x, int K, float gx) {
    std::vector<double> q(K);
    for(int b = 0; b < K / 16; ++b) {
        float amax = 0.f;
        for(int i = 0; i < 16; ++i) amax = std::fmax(amax, std::fabs(bf(x[16 * b + i])));
        const float sv = e4m3(to_e4m3(amax * gx / 6.f));
        const float mul = sv > 0.f ? gx / sv : 0.f;
        for(int i = 0; i < 16; ++i) q[16 * b + i] = static_cast<double>(e2m1(to_e2m1(bf(x[16 * b + i]) * mul))) * sv / gx;
    }
    return q;
}
template <class H>
double dot(const H& h, int n, const std::vector<double>& x) {
    double s = 0.0;
    for(int k = 0; k < h.K; ++k) s += h.w(n, k) * x[k];
    return s;
}

// In double, in forms without cancellation (0.5 (1 + tanh u) = 1 / (1 + e^-2u), 1 + erf z = erfc(-z)): the textbook
// forms lose every digit for v below about -6 even in double.
double activation(st::Activation a, double v) {
    if(a == st::Activation::silu) return v / (1.0 + std::exp(-v));
    if(a == st::Activation::gelu_tanh) return v / (1.0 + std::exp(-1.5957691216057308 * (v + 0.044715 * v * v * v)));
    return 0.5 * v * std::erfc(-v * 0.7071067811865476);
}
const char* name_of(st::Activation a) {
    return a == st::Activation::silu ? "silu" : a == st::Activation::gelu_tanh ? "gelu_tanh" : "gelu";
}
std::uint16_t bits(__nv_bfloat16 v) {
    std::uint16_t b;
    std::memcpy(&b, &v, 2);
    return b;
}
// Distance in BF16 units in the last place (same sign, or both zero, expected for these checks).
int ulps(__nv_bfloat16 a, __nv_bfloat16 b) {
    const int ia = bits(a) & 0x8000 ? -(bits(a) & 0x7FFF) : bits(a), ib = bits(b) & 0x8000 ? -(bits(b) & 0x7FFF) : bits(b);
    return std::abs(ia - ib);
}
// Gated outputs against bf16(act(gate) * up) of FP32 gate / up in double: exact, one ulp, or worse. A result whose
// activation falls below FP32's normal range (|act(v)| < 2^-126 v, from v below about -10) keeps only a subnormal's
// bits, so outputs under 1e-6 of the largest count as agreeing.
struct UlpCount {
    std::size_t exact = 0, one = 0, worse = 0;
};
UlpCount count_ulps(const std::vector<__nv_bfloat16>& o, const std::vector<float>& gate, const std::vector<float>& up, st::Activation a) {
    UlpCount c;
    double scale = 0.0;
    for(std::size_t i = 0; i < o.size(); ++i) scale = std::fmax(scale, std::fabs(activation(a, gate[i]) * up[i]));
    for(std::size_t i = 0; i < o.size(); ++i) {
        const double want = activation(a, gate[i]) * up[i];
        const int d = ulps(o[i], __float2bfloat16(static_cast<float>(want)));
        if(d == 0) ++c.exact;
        else if(d == 1 || std::fabs(bf(o[i]) - want) <= 1e-6 * scale) ++c.one;
        else ++c.worse;
    }
    return c;
}

// ------------------------------------------------------------------------------------------- placement
// Repacking parts straight into their place (concatenated along N, or gate | up interleaved in 16-row tiles) must
// give the bytes of repacking the fused row-major matrix.
bool placement_case(bool fp4, bool interleave, std::mt19937& rng) {
    const int K = 256;
    const std::vector<int> sizes = interleave ? std::vector<int>{96, 96} : std::vector<int>{64, 32, 160};
    std::vector<Fp8Host> p8;
    std::vector<Fp4Host> p4;
    for(int n : sizes) {
        if(fp4) p4.push_back(random_fp4(n, K, 1.f, rng));
        else p8.push_back(random_fp8(n, K, 0.01f, rng));
    }
    int N = 0;
    for(int n : sizes) N += n;
    const std::size_t cb = static_cast<std::size_t>(N) * K / (fp4 ? 2 : 1), sb = static_cast<std::size_t>(N) * K / 16;
    Dev<std::uint8_t> placed(cb), placed_s(sb), want(cb), want_s(sb);
    Dev<__nv_bfloat16> placed_rs(N);
    bool ok = placed.p && placed_s.p && want.p && want_s.p && placed_rs.p;
    for(std::size_t i = 0, first = 0; ok && i < sizes.size(); ++i) {
        const int f = interleave ? static_cast<int>(i) : static_cast<int>(first), step = interleave ? 2 : 1;
        if(fp4) {
            Dev<std::uint8_t> c(p4[i].codes), s(p4[i].scales);
            ok = st::enqueue_repack_fp4_rows(c.p, s.p, placed.p, placed_s.p, sizes[i], K, f, step, nullptr) == cudaSuccess && ok;
        } else {
            Dev<std::uint8_t> c(p8[i].codes);
            Dev<__nv_bfloat16> r(p8[i].rs);
            ok = st::enqueue_repack_fp8_rows(c.p, placed.p, sizes[i], K, f, step, nullptr) == cudaSuccess && ok;
            ok = st::enqueue_place_row_scales(r.p, placed_rs.p, sizes[i], f, step, nullptr) == cudaSuccess && ok;
        }
        ok = cudaDeviceSynchronize() == cudaSuccess && ok;
        first += sizes[i] / 16;
    }
    bool same = false;
    if(fp4) {
        std::vector<const Fp4Host*> parts;
        for(const auto& h : p4) parts.push_back(&h);
        const Fp4Host f = fuse(parts, interleave);
        Dev<std::uint8_t> c(f.codes), s(f.scales);
        ok = st::enqueue_repack_fp4(c.p, s.p, want.p, want_s.p, N, K, nullptr) == cudaSuccess && ok;
        same = placed.get(cb) == want.get(cb) && placed_s.get(sb) == want_s.get(sb);
    } else {
        std::vector<const Fp8Host*> parts;
        for(const auto& h : p8) parts.push_back(&h);
        const Fp8Host f = fuse(parts, interleave);
        Dev<std::uint8_t> c(f.codes);
        ok = st::enqueue_repack_fp8(c.p, want.p, N, K, nullptr) == cudaSuccess && ok;
        const auto rs = placed_rs.get(N);
        same = placed.get(cb) == want.get(cb) &&
               std::memcmp(rs.data(), f.rs.data(), static_cast<std::size_t>(N) * sizeof(__nv_bfloat16)) == 0;
    }
    ok = ok && same;
    std::printf("%s placement (%s): %s\n", fp4 ? "fp4" : "fp8", interleave ? "gate | up interleaved" : "three parts concatenated",
                ok ? "bytes equal the fused matrix's repack" : "MISMATCH");
    return ok;
}

// ------------------------------------------------------------------------------------------- gated prompt GEMM
// Gate and up placed interleaved; the gated launch's out must be bf16(act(gate) * up) of the plain launch's own FP32
// gate and up rows under the same plan (at most one BF16 ulp from the host's double activation), and close to the
// FP64 reference with the same activation quantization.
bool gated_case(bool fp4, int I, int K, int T, int min_blocks, std::mt19937& rng) {
    const float wgs_gate = 448.f * 6.f / 3.f, wgs_up = 448.f * 6.f / 2.f, gx = 448.f * 6.f / 4.f;
    Fp8Host g8, u8;
    Fp4Host g4, u4;
    if(fp4) {
        g4 = random_fp4(I, K, wgs_gate, rng);
        u4 = random_fp4(I, K, wgs_up, rng);
    } else {
        g8 = random_fp8(I, K, 2e-4f, rng);
        u8 = random_fp8(I, K, 2e-4f, rng);
    }
    const int N = 2 * I;
    std::uniform_real_distribution<float> value(-2.f, 2.f);
    std::vector<__nv_bfloat16> x(static_cast<std::size_t>(T) * K);
    for(auto& v : x) v = __float2bfloat16(value(rng));
    const st::Plan p = st::plan(fp4, N, K, T, 256, min_blocks);
    Dev<std::uint8_t> tiles(static_cast<std::size_t>(N) * K / (fp4 ? 2 : 1)), stiles(static_cast<std::size_t>(N) * K / 16);
    Dev<__nv_bfloat16> rs(N), dx(x), out(static_cast<std::size_t>(T) * I);
    Dev<float> y(static_cast<std::size_t>(T) * N), ts(p.token_scale_count + 1), part(p.partial_floats + 1);
    Dev<std::uint8_t> act(p.activation_bytes);
    bool ok = true;
    for(int s = 0; s < 2; ++s) {  // gate tiles at even tile rows, up at odd
        if(fp4) {
            const Fp4Host& h = s ? u4 : g4;
            Dev<std::uint8_t> c(h.codes), sc(h.scales);
            ok = st::enqueue_repack_fp4_rows(c.p, sc.p, tiles.p, stiles.p, I, K, s, 2, nullptr) == cudaSuccess && ok;
        } else {
            const Fp8Host& h = s ? u8 : g8;
            Dev<std::uint8_t> c(h.codes);
            Dev<__nv_bfloat16> r(h.rs);
            ok = st::enqueue_repack_fp8_rows(c.p, tiles.p, I, K, s, 2, nullptr) == cudaSuccess && ok;
            ok = st::enqueue_place_row_scales(r.p, rs.p, I, s, 2, nullptr) == cudaSuccess && ok;
        }
        ok = cudaDeviceSynchronize() == cudaSuccess && ok;
    }
    const st::Scratch sc{act.p, ts.p, part.p};
    // The plain launch's gate and up (NVFP4: each with its own weight_global_scale).
    std::vector<float> gate(static_cast<std::size_t>(T) * I), up(gate.size());
    for(int s = 0; s < (fp4 ? 2 : 1); ++s) {
        const cudaError_t e = fp4 ? st::enqueue_fp4({tiles.p, stiles.p, s ? wgs_up : wgs_gate, gx, N, K}, dx.p, T, y.p, N, false, p, sc, nullptr)
                                  : st::enqueue_fp8({tiles.p, rs.p, N, K}, dx.p, T, y.p, N, false, p, sc, nullptr);
        ok = e == cudaSuccess && ok;
        const auto yh = y.get(static_cast<std::size_t>(T) * N);
        for(int t = 0; t < T; ++t)
            for(int j = 0; j < I; ++j) {
                const std::size_t row = static_cast<std::size_t>(t) * N + j / 16 * 32 + j % 16;
                if(!fp4 || s == 0) gate[static_cast<std::size_t>(t) * I + j] = yh[row];
                if(!fp4 || s == 1) up[static_cast<std::size_t>(t) * I + j] = yh[row + 16];
            }
    }
    // FP64 reference of gate and up for the first tokens.
    const int Tr = std::min(T, 8);
    std::vector<double> gref(static_cast<std::size_t>(Tr) * I), uref(gref.size());
    for(int t = 0; t < Tr; ++t) {
        const auto q = fp4 ? quantized_fp4(x.data() + static_cast<std::size_t>(t) * K, K, gx) : quantized_fp8(x.data() + static_cast<std::size_t>(t) * K, K);
        for(int j = 0; j < I; ++j) {
            gref[static_cast<std::size_t>(t) * I + j] = fp4 ? dot(g4, j, q) : dot(g8, j, q);
            uref[static_cast<std::size_t>(t) * I + j] = fp4 ? dot(u4, j, q) : dot(u8, j, q);
        }
    }
    for(const st::Activation a : {st::Activation::silu, st::Activation::gelu_tanh, st::Activation::gelu}) {
        const cudaError_t e = fp4 ? st::enqueue_fp4_gated({tiles.p, stiles.p, wgs_gate, gx, N, K}, wgs_up, dx.p, T, a, out.p, I, p, sc, nullptr)
                                  : st::enqueue_fp8_gated({tiles.p, rs.p, N, K}, dx.p, T, a, out.p, I, p, sc, nullptr);
        ok = e == cudaSuccess && ok;
        const auto o = out.get(static_cast<std::size_t>(T) * I);
        const auto [exact, one, worse] = count_ulps(o, gate, up, a);
        double scale = 0.0, err = 0.0;
        for(std::size_t i = 0; i < gref.size(); ++i) scale = std::fmax(scale, std::fabs(activation(a, gref[i]) * uref[i]));
        for(std::size_t i = 0; i < gref.size(); ++i) err = std::fmax(err, std::fabs(bf(o[i]) - activation(a, gref[i]) * uref[i]) / scale);
        const bool pass = !worse && err < 8e-3;
        std::printf("%s gated %-9s I=%d K=%d T=%d nt=%d sk=%d: vs the plain launch %zu exact, %zu one ulp, %zu worse; vs FP64 %.1e%s\n",
                    fp4 ? "fp4" : "fp8", name_of(a), I, K, T, p.nt, p.sk, exact, one, worse, err, pass ? "" : "  FAIL");
        ok = ok && pass;
    }
    return ok;
}

// ------------------------------------------------------------------------------------------- decode / verify GEMV
// A host matrix repacked on the device (one matrix, or gate | up interleaved through the placement repacks).
struct DeviceTiles {
    Dev<std::uint8_t> tiles, scale_tiles;
    Dev<__nv_bfloat16> rs;
    int N, K;
    bool ok = true;
    template <class H>
    DeviceTiles(const std::vector<const H*>& parts, bool interleave)
        : tiles(fused_rows(parts) * static_cast<std::size_t>(parts[0]->K)), scale_tiles(fused_rows(parts) * static_cast<std::size_t>(parts[0]->K) / 16),
          rs(fused_rows(parts)), N(static_cast<int>(fused_rows(parts))), K(parts[0]->K) {
        int first = 0;
        for(std::size_t i = 0; i < parts.size(); ++i) {
            const H& h = *parts[i];
            const int f = interleave ? static_cast<int>(i) : first, step = interleave ? 2 : 1;
            Dev<std::uint8_t> c(h.codes);
            if constexpr(std::is_same_v<H, Fp4Host>) {
                Dev<std::uint8_t> s(h.scales);
                ok = st::enqueue_repack_fp4_rows(c.p, s.p, tiles.p, scale_tiles.p, h.N, K, f, step, nullptr) == cudaSuccess && ok;
            } else {
                Dev<__nv_bfloat16> r(h.rs);
                ok = st::enqueue_repack_fp8_rows(c.p, tiles.p, h.N, K, f, step, nullptr) == cudaSuccess && ok;
                ok = st::enqueue_place_row_scales(r.p, rs.p, h.N, f, step, nullptr) == cudaSuccess && ok;
            }
            ok = cudaDeviceSynchronize() == cudaSuccess && ok;
            first += h.N / 16;
        }
    }
    template <class H>
    static std::size_t fused_rows(const std::vector<const H*>& parts) {
        std::size_t n = 0;
        for(const H* p : parts) n += p->N;
        return n;
    }
    st::Fp8Weights fp8() const { return {tiles.p, rs.p, N, K}; }
    st::Fp4Weights fp4(float wgs) const { return {tiles.p, scale_tiles.p, wgs, 1.f, N, K}; }
};

std::vector<__nv_bfloat16> random_rows(int M, int ld, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__nv_bfloat16> x(static_cast<std::size_t>(M) * ld);
    for(auto& v : x) v = __float2bfloat16(nd(rng));
    return x;
}

// Row invariance (row m of every M = 1..16 launch, and of a 16-row launch with the rows reversed, equals the M = 1
// launch on that row, bitwise), determinism, add, and accuracy against FP64: |err| <= 5e-5 sum_k |w x| per element.
template <class H>
bool gemv_case(const H& h, int groups, std::mt19937& rng) {
    constexpr bool fp4 = std::is_same_v<H, Fp4Host>;
    const int N = h.N, K = h.K, ldx = K + 8, R = 16;
    DeviceTiles d(std::vector<const H*>{&h}, false);
    const st::GemvPlan p = st::plan_gemv(fp4, N, K, groups);
    const auto x = random_rows(R, ldx, rng);
    std::vector<__nv_bfloat16> xr(x.size());
    for(int m = 0; m < R; ++m) std::copy_n(x.begin() + static_cast<std::size_t>(R - 1 - m) * ldx, ldx, xr.begin() + static_cast<std::size_t>(m) * ldx);
    Dev<__nv_bfloat16> dx(x), dxr(xr);
    Dev<float> y(static_cast<std::size_t>(R) * N);
    bool ok = d.ok;
    float wgs = 1.f;
    if constexpr(fp4) wgs = h.wgs;
    const auto run = [&](const __nv_bfloat16* xp, int M, bool add) {
        return (fp4 ? st::enqueue_gemv_fp4(d.fp4(wgs), xp, ldx, M, y.p, N, add, p, nullptr)
                    : st::enqueue_gemv_fp8(d.fp8(), xp, ldx, M, y.p, N, add, p, nullptr)) == cudaSuccess;
    };
    std::vector<float> one(static_cast<std::size_t>(R) * N);
    for(int m = 0; m < R; ++m) {
        ok = run(dx.p + static_cast<std::size_t>(m) * ldx, 1, false) && ok;
        const auto r = y.get(N);
        std::copy(r.begin(), r.end(), one.begin() + static_cast<std::size_t>(m) * N);
    }
    bool invariant = true;
    for(int M = 1; M <= R; ++M) {
        ok = cudaMemset(y.p, 0xFF, static_cast<std::size_t>(R) * N * sizeof(float)) == cudaSuccess && run(dx.p, M, false) && ok;
        const auto r = y.get(static_cast<std::size_t>(M) * N);
        invariant = invariant && std::memcmp(r.data(), one.data(), r.size() * sizeof(float)) == 0;
    }
    ok = run(dxr.p, R, false) && ok;
    const auto rev = y.get(static_cast<std::size_t>(R) * N);
    for(int m = 0; m < R; ++m)
        invariant = invariant && std::memcmp(rev.data() + static_cast<std::size_t>(m) * N, one.data() + static_cast<std::size_t>(R - 1 - m) * N,
                                             static_cast<std::size_t>(N) * sizeof(float)) == 0;
    ok = run(dx.p, R, false) && ok;
    const auto y16 = y.get(static_cast<std::size_t>(R) * N);
    ok = run(dx.p, R, false) && ok;
    const bool repeat = y.get(y16.size()) == y16;
    // A drafter's launches (role draft) run the same arithmetic under a kernel name that says so.
    st::GemvPlan draft = p;
    draft.role = st::KernelRole::draft;
    ok = (fp4 ? st::enqueue_gemv_fp4(d.fp4(wgs), dx.p, ldx, R, y.p, N, false, draft, nullptr)
              : st::enqueue_gemv_fp8(d.fp8(), dx.p, ldx, R, y.p, N, false, draft, nullptr)) == cudaSuccess && ok;
    const bool drafted = y.get(y16.size()) == y16;
    // add: y0 + v rounded once (IEEE FP32 addition of the plain result)
    std::vector<float> y0(y16.size());
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for(auto& v : y0) v = u(rng);
    ok = cudaMemcpy(y.p, y0.data(), y0.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess && run(dx.p, R, true) && ok;
    const auto ya = y.get(y16.size());
    bool add_ok = true;
    for(std::size_t i = 0; i < ya.size(); ++i) {
        const float want = y0[i] + y16[i];
        add_ok = add_ok && std::memcmp(&ya[i], &want, 4) == 0;
    }
    double worst = 0.0;
    std::vector<double> xd(K);
    for(int m = 0; m < R; ++m) {
        for(int k = 0; k < K; ++k) xd[k] = bf(x[static_cast<std::size_t>(m) * ldx + k]);
        for(int n = 0; n < N; ++n) {
            double s = 0.0, mag = 0.0;
            for(int k = 0; k < K; ++k) {
                const double t = h.w(n, k) * xd[k];
                s += t;
                mag += std::fabs(t);
            }
            worst = std::fmax(worst, std::fabs(y16[static_cast<std::size_t>(m) * N + n] - s) / (mag + 1e-300));
        }
    }
    ok = ok && invariant && repeat && drafted && add_ok && worst <= 5e-5;
    std::printf("gemv %s N=%d K=%d groups=%d: rows of M = 1..16 and reversed %s; repeat %s; draft role %s; add %s; "
                "max |err| / sum|wx| %.1e%s\n",
                fp4 ? "fp4" : "fp8", N, K, p.groups, invariant ? "bitwise equal" : "DIFFER", repeat ? "identical" : "DIFFERS",
                drafted ? "identical" : "DIFFERS", add_ok ? "exact" : "WRONG", worst, ok ? "" : "  FAIL");
    return ok;
}

// Every code decodes exactly: FP8 row n holds code n (NaN codes zeroed), NVFP4 row n code n % 16 with group scale code
// n / 16; one-hot activation rows pick one column each, so y must equal the stored value exactly (FP4: e2m1 * e4m3 /
// weight_global_scale, the division in FP32).
bool gemv_decodes_every_code(bool fp4, int groups) {
    const int N = fp4 ? 4096 : 256, K = fp4 ? 256 : 128, R = 16;
    Fp8Host h8{N, K, std::vector<std::uint8_t>(static_cast<std::size_t>(N) * K), std::vector<__nv_bfloat16>(N, __float2bfloat16(1.f))};
    Fp4Host h4{N, K, std::vector<std::uint8_t>(static_cast<std::size_t>(N) * K / 2), std::vector<std::uint8_t>(static_cast<std::size_t>(N) * K / 16), 3.f};
    for(int n = 0; n < N; ++n) {
        if(fp4) {
            const int c = n % 16, s = n / 16;
            std::fill_n(h4.codes.begin() + static_cast<std::size_t>(n) * K / 2, K / 2, static_cast<std::uint8_t>(c | (c << 4)));
            std::fill_n(h4.scales.begin() + static_cast<std::size_t>(n) * K / 16, K / 16, static_cast<std::uint8_t>((s & 0x7F) == 0x7F ? 0 : s));
        } else {
            std::fill_n(h8.codes.begin() + static_cast<std::size_t>(n) * K, K, static_cast<std::uint8_t>((n & 0x7F) == 0x7F ? 0 : n));
        }
    }
    std::vector<__nv_bfloat16> x(static_cast<std::size_t>(R) * K, __float2bfloat16(0.f));
    std::vector<int> col(R);
    for(int m = 0; m < R; ++m) {
        col[m] = (m * 37 + 5) % K;
        x[static_cast<std::size_t>(m) * K + col[m]] = __float2bfloat16(1.f);
    }
    std::unique_ptr<DeviceTiles> d = fp4 ? std::make_unique<DeviceTiles>(std::vector<const Fp4Host*>{&h4}, false)
                                         : std::make_unique<DeviceTiles>(std::vector<const Fp8Host*>{&h8}, false);
    Dev<__nv_bfloat16> dx(x);
    Dev<float> y(static_cast<std::size_t>(R) * N);
    const st::GemvPlan p = st::plan_gemv(fp4, N, K, groups);
    bool ok = d->ok && (fp4 ? st::enqueue_gemv_fp4(d->fp4(h4.wgs), dx.p, K, R, y.p, N, false, p, nullptr)
                            : st::enqueue_gemv_fp8(d->fp8(), dx.p, K, R, y.p, N, false, p, nullptr)) == cudaSuccess;
    const auto got = y.get(static_cast<std::size_t>(R) * N);
    std::size_t bad = 0;
    for(int m = 0; m < R; ++m)
        for(int n = 0; n < N; ++n) {
            const int c = n % 16, s = (n / 16 & 0x7F) == 0x7F ? 0 : n / 16;
            const float want = fp4 ? e2m1(c) * e4m3(static_cast<std::uint8_t>(s)) / h4.wgs : static_cast<float>(h8.w(n, col[m]));
            if(got[static_cast<std::size_t>(m) * N + n] != want) ++bad;  // -0 codes sum to +0
        }
    ok = ok && !bad;
    std::printf("gemv %s decode of every %s at 16 columns: %zu of %d values wrong%s\n", fp4 ? "fp4" : "fp8",
                fp4 ? "E2M1 code x E4M3 scale" : "E4M3 code", bad, R * N, ok ? "" : "  FAIL");
    return ok;
}

// The gated GEMV: bf16(act(gate) * up) of the plain GEMV's own rows (within one BF16 ulp of the host's double
// activation), row invariant like the plain GEMV.
template <class H>
bool gemv_gated_case(const H& gate, const H& up, int groups, std::mt19937& rng) {
    constexpr bool fp4 = std::is_same_v<H, Fp4Host>;
    const int I = gate.N, N = 2 * I, K = gate.K, R = 16;
    DeviceTiles d(std::vector<const H*>{&gate, &up}, true);
    const st::GemvPlan p = st::plan_gemv(fp4, N, K, groups);
    const auto x = random_rows(R, K, rng);
    Dev<__nv_bfloat16> dx(x), out(static_cast<std::size_t>(R) * I);
    Dev<float> y(static_cast<std::size_t>(R) * N);
    bool ok = d.ok;
    std::vector<float> g(static_cast<std::size_t>(R) * I), u(g.size());
    for(int s = 0; s < (fp4 ? 2 : 1); ++s) {
        float wgs = 1.f;
        if constexpr(fp4) wgs = s ? up.wgs : gate.wgs;
        ok = (fp4 ? st::enqueue_gemv_fp4(d.fp4(wgs), dx.p, K, R, y.p, N, false, p, nullptr)
                  : st::enqueue_gemv_fp8(d.fp8(), dx.p, K, R, y.p, N, false, p, nullptr)) == cudaSuccess && ok;
        const auto yh = y.get(static_cast<std::size_t>(R) * N);
        for(int m = 0; m < R; ++m)
            for(int j = 0; j < I; ++j) {
                const std::size_t row = static_cast<std::size_t>(m) * N + j / 16 * 32 + j % 16;
                if(!fp4 || s == 0) g[static_cast<std::size_t>(m) * I + j] = yh[row];
                if(!fp4 || s == 1) u[static_cast<std::size_t>(m) * I + j] = yh[row + 16];
            }
    }
    float gate_wgs = 1.f, up_wgs = 1.f;
    if constexpr(fp4) {
        gate_wgs = gate.wgs;
        up_wgs = up.wgs;
    }
    for(const st::Activation a : {st::Activation::silu, st::Activation::gelu_tanh, st::Activation::gelu}) {
        const auto launch = [&](const __nv_bfloat16* xp, int M) {
            return (fp4 ? st::enqueue_gemv_fp4_gated(d.fp4(gate_wgs), up_wgs, xp, K, M, a, out.p, I, p, nullptr)
                        : st::enqueue_gemv_fp8_gated(d.fp8(), xp, K, M, a, out.p, I, p, nullptr)) == cudaSuccess;
        };
        ok = launch(dx.p, R) && ok;
        const auto o = out.get(static_cast<std::size_t>(R) * I);
        const auto [exact, one, worse] = count_ulps(o, g, u, a);
        bool invariant = true;
        for(int m = 0; m < R; ++m) {
            ok = launch(dx.p + static_cast<std::size_t>(m) * K, 1) && ok;
            const auto r = out.get(I);
            invariant = invariant && std::memcmp(r.data(), o.data() + static_cast<std::size_t>(m) * I, static_cast<std::size_t>(I) * 2) == 0;
        }
        const bool pass = !worse && invariant;
        std::printf("gemv %s gated %-9s I=%d K=%d groups=%d: vs the plain GEMV %zu exact, %zu one ulp, %zu worse; rows vs M = 1 %s%s\n",
                    fp4 ? "fp4" : "fp8", name_of(a), I, K, p.groups, exact, one, worse, invariant ? "bitwise equal" : "DIFFER",
                    pass ? "" : "  FAIL");
        ok = ok && pass;
    }
    return ok;
}

bool gemv_rejects() {
    const auto* t = reinterpret_cast<const std::uint8_t*>(0x1000);
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(0x1000);
    auto* y = reinterpret_cast<float*>(0x100000);
    auto* o = reinterpret_cast<__nv_bfloat16*>(0x100000);
    const st::Fp8Weights w8{t, x, 256, 512};
    const st::Fp4Weights w4{t, t, 1.f, 1.f, 256, 512};
    const st::GemvPlan p8 = st::plan_gemv(false, 256, 512, 4), p4 = st::plan_gemv(true, 256, 512, 4);
    const auto bad8 = [&](const st::Fp8Weights& w, int ldx, int M, int ldy, const st::GemvPlan& p) {
        return st::enqueue_gemv_fp8(w, x, ldx, M, y, ldy, false, p, nullptr) == cudaErrorInvalidValue;
    };
    const bool ok = bad8(w8, 512, 0, 256, p8) && bad8(w8, 512, 17, 256, p8) && bad8(w8, 513, 1, 256, p8) && bad8(w8, 511, 1, 256, p8) &&
                    bad8(w8, 512, 1, 255, p8) && bad8(w8, 512, 1, 256, p4) && bad8(w8, 512, 1, 256, st::plan_gemv(false, 512, 512, 4)) &&
                    bad8(w8, 512, 1, 256, st::plan_gemv(false, 256, 512, 5)) && bad8({t, x, 264, 512}, 512, 1, 264, st::plan_gemv(false, 264, 512, 4)) &&
                    st::enqueue_gemv_fp4({t, t, 0.f, 1.f, 256, 512}, x, 512, 1, y, 256, false, p4, nullptr) == cudaErrorInvalidValue &&
                    st::enqueue_gemv_fp4(w4, x, 512, 1, y, 256, false, p8, nullptr) == cudaErrorInvalidValue &&
                    st::enqueue_gemv_fp8_gated({t, x, 272, 512}, x, 512, 1, st::Activation::silu, o, 136, st::plan_gemv(false, 272, 512, 4), nullptr) ==
                        cudaErrorInvalidValue &&
                    st::enqueue_gemv_fp8_gated(w8, x, 512, 1, st::Activation::silu, o, 127, p8, nullptr) == cudaErrorInvalidValue &&
                    st::enqueue_gemv_fp4_gated(w4, 0.f, x, 512, 1, st::Activation::silu, o, 128, p4, nullptr) == cudaErrorInvalidValue;
    std::printf("gemv rejections (M 0 / 17, odd or short ldx, short ldy, other plan, N %% 16, gated N %% 32, ldo < I, scale <= 0): %s\n",
                ok ? "rejected" : "NOT rejected");
    return ok;
}

// ------------------------------------------------------------------------------------------- BF16-activation GEMM
// Against FP64 of the stored weights and the BF16 activations as they are (|err| <= 5e-5 sum_k |w x| per element),
// add exact, and every token's bits equal to the plan's longest launch.
template <class H>
bool a16_case(const H& h, int T, int min_blocks, std::mt19937& rng) {
    constexpr bool fp4 = std::is_same_v<H, Fp4Host>;
    const int N = h.N, K = h.K;
    DeviceTiles d(std::vector<const H*>{&h}, false);
    const st::A16Plan p = st::plan_a16(fp4, N, K, T, min_blocks);
    const auto x = random_rows(T, K, rng);
    Dev<__nv_bfloat16> dx(x);
    Dev<float> y(static_cast<std::size_t>(T) * N), part(p.partial_floats + 1);
    float wgs = 1.f;
    if constexpr(fp4) wgs = h.wgs;
    const auto run = [&](int t, bool add) {
        return (fp4 ? st::enqueue_fp4_a16(d.fp4(wgs), dx.p, t, y.p, N, add, p, part.p, nullptr)
                    : st::enqueue_fp8_a16(d.fp8(), dx.p, t, y.p, N, add, p, part.p, nullptr)) == cudaSuccess;
    };
    bool ok = d.ok && run(T, false);
    const auto got = y.get(static_cast<std::size_t>(T) * N);
    bool shorter = true;
    for(const int t : {1, 7, 63, 65, T - 1}) {
        if(t < 1 || t >= T) continue;
        ok = cudaMemset(y.p, 0xFF, static_cast<std::size_t>(T) * N * sizeof(float)) == cudaSuccess && run(t, false) && ok;
        const auto r = y.get(static_cast<std::size_t>(t) * N);
        shorter = shorter && std::memcmp(r.data(), got.data(), r.size() * sizeof(float)) == 0;
    }
    std::vector<float> y0(got.size());
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for(auto& v : y0) v = u(rng);
    ok = cudaMemcpy(y.p, y0.data(), y0.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess && run(T, true) && ok;
    const auto ya = y.get(got.size());
    bool add_ok = true;
    for(std::size_t i = 0; i < ya.size(); ++i) {
        const float want = y0[i] + got[i];
        add_ok = add_ok && std::memcmp(&ya[i], &want, 4) == 0;
    }
    double worst = 0.0;
    std::vector<double> xd(K), wr(K);
    for(int n = 0; n < N; ++n) {
        for(int k = 0; k < K; ++k) wr[k] = h.w(n, k);
        for(int t = 0; t < T; ++t) {
            double s = 0.0, mag = 0.0;
            for(int k = 0; k < K; ++k) {
                const double v = wr[k] * bf(x[static_cast<std::size_t>(t) * K + k]);
                s += v;
                mag += std::fabs(v);
            }
            worst = std::fmax(worst, std::fabs(got[static_cast<std::size_t>(t) * N + n] - s) / (mag + 1e-300));
        }
    }
    ok = ok && shorter && add_ok && worst <= 5e-5;
    std::printf("a16 %s N=%d K=%d T=%d sk=%d: max |err| / sum|wx| %.1e; shorter launches %s; add %s%s\n", fp4 ? "fp4" : "fp8", N, K, T, p.sk,
                worst, shorter ? "bitwise equal" : "DIFFER", add_ok ? "exact" : "WRONG", ok ? "" : "  FAIL");
    return ok;
}

// The gated launch: bf16(act(gate) * up) of the plain launch's own gate and up rows under the same plan.
template <class H>
bool a16_gated_case(const H& gate, const H& up, int T, int min_blocks, std::mt19937& rng) {
    constexpr bool fp4 = std::is_same_v<H, Fp4Host>;
    const int I = gate.N, N = 2 * I, K = gate.K;
    DeviceTiles d(std::vector<const H*>{&gate, &up}, true);
    const st::A16Plan p = st::plan_a16(fp4, N, K, T, min_blocks);
    const auto x = random_rows(T, K, rng);
    Dev<__nv_bfloat16> dx(x), out(static_cast<std::size_t>(T) * I);
    Dev<float> y(static_cast<std::size_t>(T) * N), part(p.partial_floats + 1);
    float gate_wgs = 1.f, up_wgs = 1.f;
    if constexpr(fp4) {
        gate_wgs = gate.wgs;
        up_wgs = up.wgs;
    }
    bool ok = d.ok;
    std::vector<float> g(static_cast<std::size_t>(T) * I), u(g.size());
    for(int s = 0; s < (fp4 ? 2 : 1); ++s) {
        ok = (fp4 ? st::enqueue_fp4_a16(d.fp4(s ? up_wgs : gate_wgs), dx.p, T, y.p, N, false, p, part.p, nullptr)
                  : st::enqueue_fp8_a16(d.fp8(), dx.p, T, y.p, N, false, p, part.p, nullptr)) == cudaSuccess && ok;
        const auto yh = y.get(static_cast<std::size_t>(T) * N);
        for(int t = 0; t < T; ++t)
            for(int j = 0; j < I; ++j) {
                const std::size_t row = static_cast<std::size_t>(t) * N + j / 16 * 32 + j % 16;
                if(!fp4 || s == 0) g[static_cast<std::size_t>(t) * I + j] = yh[row];
                if(!fp4 || s == 1) u[static_cast<std::size_t>(t) * I + j] = yh[row + 16];
            }
    }
    for(const st::Activation a : {st::Activation::silu, st::Activation::gelu_tanh, st::Activation::gelu}) {
        ok = (fp4 ? st::enqueue_fp4_a16_gated(d.fp4(gate_wgs), up_wgs, dx.p, T, a, out.p, I, p, part.p, nullptr)
                  : st::enqueue_fp8_a16_gated(d.fp8(), dx.p, T, a, out.p, I, p, part.p, nullptr)) == cudaSuccess && ok;
        const auto [exact, one, worse] = count_ulps(out.get(static_cast<std::size_t>(T) * I), g, u, a);
        std::printf("a16 %s gated %-9s I=%d K=%d T=%d sk=%d: vs the plain launch %zu exact, %zu one ulp, %zu worse%s\n", fp4 ? "fp4" : "fp8",
                    name_of(a), I, K, T, p.sk, exact, one, worse, worse ? "  FAIL" : "");
        ok = ok && !worse;
    }
    return ok;
}

bool a16_rejects() {
    const auto* t = reinterpret_cast<const std::uint8_t*>(0x1000);
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(0x1000);
    auto* y = reinterpret_cast<float*>(0x100000);
    auto* o = reinterpret_cast<__nv_bfloat16*>(0x100000);
    const st::Fp8Weights w8{t, x, 256, 512};
    const st::Fp4Weights w4{t, t, 1.f, 1.f, 256, 512};
    const st::A16Plan p8 = st::plan_a16(false, 256, 512, 64, 1), p4 = st::plan_a16(true, 256, 512, 64, 1), split = st::plan_a16(false, 256, 512, 64, 1000);
    const bool ok = st::enqueue_fp8_a16(w8, x, 65, y, 256, false, p8, nullptr, nullptr) == cudaErrorInvalidValue &&      // T > plan
                    st::enqueue_fp8_a16(w8, x, 0, y, 256, false, p8, nullptr, nullptr) == cudaErrorInvalidValue &&
                    st::enqueue_fp8_a16(w8, x, 8, y, 256, false, p4, nullptr, nullptr) == cudaErrorInvalidValue &&       // format
                    st::enqueue_fp4_a16(w4, x, 8, y, 256, false, p8, nullptr, nullptr) == cudaErrorInvalidValue &&
                    st::enqueue_fp8_a16({t, x, 512, 512}, x, 8, y, 512, false, p8, nullptr, nullptr) == cudaErrorInvalidValue &&  // N
                    st::enqueue_fp8_a16({t, x, 264, 512}, x, 8, y, 264, false, st::plan_a16(false, 264, 512, 64, 1), nullptr, nullptr) ==
                        cudaErrorInvalidValue &&                                                                         // N % 16
                    st::enqueue_fp8_a16(w8, x + 1, 8, y, 256, false, p8, nullptr, nullptr) == cudaErrorInvalidValue &&   // alignment
                    st::enqueue_fp8_a16(w8, x, 8, y, 255, false, p8, nullptr, nullptr) == cudaErrorInvalidValue &&       // ldo
                    split.sk > 1 && st::enqueue_fp8_a16(w8, x, 8, y, 256, false, split, nullptr, nullptr) == cudaErrorInvalidValue &&
                    st::enqueue_fp4_a16({t, t, 0.f, 1.f, 256, 512}, x, 8, y, 256, false, p4, nullptr, nullptr) == cudaErrorInvalidValue &&
                    st::enqueue_fp8_a16_gated({t, x, 272, 512}, x, 8, st::Activation::silu, o, 136, st::plan_a16(false, 272, 512, 64, 1), nullptr,
                                              nullptr) == cudaErrorInvalidValue &&                                       // gated N % 32
                    st::enqueue_fp8_a16_gated(w8, x, 8, st::Activation::silu, o, 127, p8, nullptr, nullptr) == cudaErrorInvalidValue &&
                    st::enqueue_fp4_a16_gated(w4, 0.f, x, 8, st::Activation::silu, o, 128, p4, nullptr, nullptr) == cudaErrorInvalidValue;
    std::printf("a16 rejections (T, format, N, N %% 16, alignment, ldo, missing partials, scales, gated N %% 32 / ldo): %s\n",
                ok ? "rejected" : "NOT rejected");
    return ok;
}

// The default split stays within one wave of resident blocks (at most the device's thread capacity in 256-thread
// blocks), splits a grid of one block, and leaves a grid that fills the device unsplit.
bool default_split_stays_in_one_wave() {
    int device = 0, sms = 0, threads = 0;
    if(cudaGetDevice(&device) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
       cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerMultiProcessor, device) != cudaSuccess)
        return false;
    const long wave_bound = static_cast<long>(sms) * (threads / 256);
    bool ok = true;
    for(const bool fp4 : {false, true})
        for(const int N : {128, 256, 1024, 4096, 8192})
            for(const int T : {1, 64, 200, 300, 1024, 4096}) {
                const st::Plan p = st::plan(fp4, N, 512, T);
                const long blocks = static_cast<long>(p.tpad / (8 * p.nt)) * (N / (128 * p.rt));
                ok = ok && p.sk >= 1 && p.sk <= 8 && (p.sk == 1 || blocks * p.sk <= wave_bound);
            }
    ok = ok && st::plan(false, 128, 512, 8).sk > 1 && st::plan(true, 128, 512, 8).sk > 1 && st::plan(false, 8192, 512, 4096).sk == 1 &&
         st::plan(true, 8192, 512, 4096).sk == 1;
    std::printf("default K split within one wave of resident blocks, a one-block grid split, a full grid unsplit: %s\n", ok ? "yes" : "NO");
    return ok;
}

// A plan sizes scratch for its format, N, K and at most its token count; any other launch must be refused before
// it writes past the activations, token scales or partials. Checks run before any launch, so no buffers are needed.
bool fp8_rejects_plan_mismatch() {
    const st::Plan p = st::plan(false, 256, 512, 64, 256, 340);
    const auto run = [](const st::Fp8Weights& w, int T, const st::Plan& plan) {
        return st::enqueue_fp8(w, nullptr, T, nullptr, w.N, false, plan, {}, nullptr) == cudaErrorInvalidValue;
    };
    const bool ok = run({nullptr, nullptr, 256, 512}, 65, p) && run({nullptr, nullptr, 256, 512}, p.tpad + 1, p) &&
                    run({nullptr, nullptr, 512, 512}, 64, p) && run({nullptr, nullptr, 256, 256}, 64, p) &&
                    run({nullptr, nullptr, 256, 512}, 64, st::plan(true, 256, 512, 64, 256, 340)) &&
                    run({nullptr, nullptr, 256, 512}, 1, st::Plan{});
    std::printf("fp8 plan mismatch (T > plan tokens, other N / K / format): %s\n", ok ? "rejected" : "NOT rejected");
    // gated: one row tile per warp, an unknown activation, ldo < I, a mismatched plan
    const st::Fp8Weights w{nullptr, nullptr, 256, 512};
    const auto gated = [&](const st::Plan& plan, int act, int ldo, int T) {
        return st::enqueue_fp8_gated(w, nullptr, T, static_cast<st::Activation>(act), nullptr, ldo, plan, {}, nullptr) == cudaErrorInvalidValue;
    };
    const bool gok = gated(st::plan(false, 256, 512, 64, 256, 340, false), 0, 128, 8) && gated(p, 3, 128, 8) && gated(p, 0, 127, 8) &&
                     gated(p, 0, 128, 65) && gated(st::plan(true, 256, 512, 64), 0, 128, 8);
    std::printf("fp8 gated (one row tile per warp, unknown activation, ldo < I, plan mismatch): %s\n", gok ? "rejected" : "NOT rejected");
    return ok && gok;
}

bool fp4_rejects_plan_mismatch() {
    const st::Plan p = st::plan(true, 256, 512, 64, 256, 340);
    const auto run = [](const st::Fp4Weights& w, int T, const st::Plan& plan) {
        return st::enqueue_fp4(w, nullptr, T, nullptr, w.N, false, plan, {}, nullptr) == cudaErrorInvalidValue;
    };
    const bool ok = run({nullptr, nullptr, 1.f, 1.f, 256, 512}, 65, p) && run({nullptr, nullptr, 1.f, 1.f, 256, 512}, p.tpad + 1, p) &&
                    run({nullptr, nullptr, 1.f, 1.f, 512, 512}, 64, p) && run({nullptr, nullptr, 1.f, 1.f, 256, 256}, 64, p) &&
                    run({nullptr, nullptr, 1.f, 1.f, 256, 512}, 64, st::plan(false, 256, 512, 64, 256, 340)) &&
                    run({nullptr, nullptr, 1.f, 1.f, 256, 512}, 1, st::Plan{});
    std::printf("fp4 plan mismatch (T > plan tokens, other N / K / format): %s\n", ok ? "rejected" : "NOT rejected");
    const st::Fp4Weights w{nullptr, nullptr, 1.f, 1.f, 256, 512};
    const auto gated = [&](const st::Plan& plan, float up_wgs, int act, int ldo) {
        return st::enqueue_fp4_gated(w, up_wgs, nullptr, 8, static_cast<st::Activation>(act), nullptr, ldo, plan, {}, nullptr) == cudaErrorInvalidValue;
    };
    const bool gok = gated(st::plan(true, 256, 512, 64, 256, 340, false), 1.f, 0, 128) && gated(p, 0.f, 0, 128) && gated(p, 1.f, -1, 128) &&
                     gated(p, 1.f, 0, 127);
    std::printf("fp4 gated (one row tile per warp, up global scale <= 0, unknown activation, ldo < I): %s\n", gok ? "rejected" : "NOT rejected");
    return ok && gok;
}

// The architecture this file's device code was compiled for (a PTX build for an older target runs JIT-compiled on a
// newer device, which is how the pre-sm_89 software decodes are exercised on one GPU).
__global__ void compiled_arch_kernel(int* out) {
#ifdef __CUDA_ARCH__
    *out = __CUDA_ARCH__;
#endif
}
int compiled_arch() {
    Dev<int> d(1);
    compiled_arch_kernel<<<1, 1>>>(d.p);
    return cudaDeviceSynchronize() == cudaSuccess ? d.get(1)[0] : 0;
}

}  // namespace

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    if(cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::printf("no CUDA device; skipped\n");
        return 0;
    }
    const int cc = std::min(prop.major * 10 + prop.minor, compiled_arch() / 10);
    std::printf("device sm_%d%d, device code compiled for sm_%d\n", prop.major, prop.minor, compiled_arch() / 10);
    std::mt19937 rng(7);
    bool ok = true;
    for(const bool fp4 : {false, true})
        for(const bool interleave : {false, true}) ok = placement_case(fp4, interleave, rng) && ok;
    ok = default_split_stays_in_one_wave() && ok;
    if(cc >= 80) {
        for(const int groups : {4, 8, 16}) {
            ok = gemv_case(random_fp8(512, 1184, 0.01f, rng), groups, rng) && ok;  // 37 K tiles: uneven per warp
            ok = gemv_case(random_fp8(272, 512, 0.01f, rng), groups, rng) && ok;   // 17 row tiles: a half pair
            ok = gemv_case(random_fp4(512, 1216, 448.f * 6.f / 3.f, rng), groups, rng) && ok;
            ok = gemv_case(random_fp4(272, 512, 448.f * 6.f / 3.f, rng), groups, rng) && ok;
            ok = gemv_decodes_every_code(false, groups) && ok;
            ok = gemv_decodes_every_code(true, groups) && ok;
            ok = gemv_gated_case(random_fp8(128, 512, 2e-3f, rng), random_fp8(128, 512, 2e-3f, rng), groups, rng) && ok;
            ok = gemv_gated_case(random_fp4(144, 512, 448.f * 6.f / 3.f, rng), random_fp4(144, 512, 448.f * 3.f, rng), groups, rng) && ok;
        }
        ok = gemv_rejects() && ok;
        for(const int mb : {1, 100000}) {  // unsplit, and split into 8 parts
            ok = a16_case(random_fp8(256, 1024, 0.01f, rng), 200, mb, rng) && ok;
            ok = a16_case(random_fp8(272, 544, 0.01f, rng), 37, mb, rng) && ok;   // a 16-row block tail, a 32-column K tail
            ok = a16_case(random_fp4(272, 1216, 448.f * 6.f / 3.f, rng), 65, mb, rng) && ok;  // 19 chunks: a partial 512-column chain
            ok = a16_case(random_fp4(256, 512, 448.f * 6.f / 3.f, rng), 1, mb, rng) && ok;
            ok = a16_gated_case(random_fp8(128, 512, 2e-3f, rng), random_fp8(128, 512, 2e-3f, rng), 70, mb, rng) && ok;
            ok = a16_gated_case(random_fp4(144, 576, 448.f * 6.f / 3.f, rng), random_fp4(144, 576, 448.f * 3.f, rng), 33, mb, rng) && ok;
        }
        for(const int mb : {1, 0}) {  // long K: 32 folded 512-column chains unsplit, and the default split
            ok = a16_case(random_fp8(256, 16384, 0.01f, rng), 64, mb, rng) && ok;
            ok = a16_case(random_fp4(256, 16384, 448.f * 6.f / 3.f, rng), 64, mb, rng) && ok;
        }
        ok = a16_rejects() && ok;
    } else {
        std::printf("BF16 MMA needs sm_80+; GEMV skipped\n");
    }
    if(cc >= 89) {
        for(const int T : {1, 7, 64, 200, 300})
            ok = fp8_case(256, 512, T, T % 2 == 1, rng) && ok;
        ok = fp8_case(128, 256, 33, false, rng) && ok;  // one row tile per warp
        ok = fp8_case(256, 512, 130, true, rng, 1) && ok;   // no split: scales applied in the GEMM epilogue
        ok = fp8_case(256, 512, 300, false, rng, 1) && ok;
        ok = st::enqueue_fp8({nullptr, nullptr, 100, 64}, nullptr, 1, nullptr, 0, false, st::plan(false, 100, 64, 1), {}, nullptr)
                 == cudaErrorInvalidValue && ok;
        ok = fp8_case(256, 512, 40, false, rng, 340, 300) && ok;  // fewer tokens than the plan's
        ok = fp8_case(256, 512, 200, false, rng, 0) && ok;  // split threshold from the device's SM count
        for(const int Tmax : {200, 300})  // 8 and 16 token groups per block
            for(const int min_blocks : {340, 1}) ok = bits_independent_of_tokens(false, Tmax, min_blocks, rng) && ok;
        ok = gated_case(false, 128, 512, 37, 340, rng) && ok;  // split: the finish applies the activation
        ok = gated_case(false, 128, 512, 300, 1, rng) && ok;   // 16 token groups per block, epilogue
        ok = gated_case(false, 256, 256, 64, 1, rng) && ok;
        ok = fp8_rejects_plan_mismatch() && ok;
    } else {
        std::printf("FP8 MMA needs sm_89+; skipped\n");
    }
    if(st::fp4_mma_compiled()) {
        for(const int T : {1, 9, 64, 200, 300})
            ok = fp4_case(256, 512, T, T % 2 == 0, rng) && ok;
        ok = fp4_case(128, 256, 40, false, rng) && ok;
        ok = fp4_case(256, 512, 130, true, rng, 1) && ok;
        ok = fp4_case(256, 512, 300, false, rng, 1) && ok;
        ok = st::enqueue_fp4({nullptr, nullptr, 0.f, 1.f, 256, 64}, nullptr, 1, nullptr, 0, false, st::plan(true, 256, 64, 1), {}, nullptr)
                 == cudaErrorInvalidValue && ok;
        ok = fp4_case(256, 512, 40, true, rng, 340, 300) && ok;
        ok = fp4_case(256, 512, 200, false, rng, 0) && ok;
        for(const int Tmax : {200, 300})  // 8 and 16 token groups per block
            for(const int min_blocks : {340, 1}) ok = bits_independent_of_tokens(true, Tmax, min_blocks, rng) && ok;
        ok = gated_case(true, 128, 512, 37, 340, rng) && ok;
        ok = gated_case(true, 128, 512, 300, 1, rng) && ok;
        ok = gated_case(true, 256, 256, 64, 1, rng) && ok;
        ok = fp4_rejects_plan_mismatch() && ok;
    } else {
        std::printf("NVFP4 block-scaled MMA not in this build (needs sm_120a / sm_121a); skipped\n");
    }
    std::printf(ok ? "stored-tile GEMMs passed\n" : "stored-tile GEMMs FAILED\n");
    return ok ? 0 : 1;
}
