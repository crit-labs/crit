// Integer block expert projections: the decode rows and the prompt GEMM give the declared order's bits (a scalar
// oracle here, the same as crit's host_int.h rows_scalar) for E2M1 x 2 and INT4 weights, 16- and 32-element activation
// blocks, decode jobs of 1 to 8 tokens and prompt jobs of 1 to 70 rows (the register and the staged GEMM); the
// quantizers and SwiGLU give the declared functions' bits. Prints the decode and GEMM rates on an expert-sized matrix
// (not asserted).
#include "int_block.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ib = crit::techniques::int_block;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
template<class T> T* dev(std::size_t n) { void* p = nullptr; ok(cudaMalloc(&p, (n ? n : 1) * sizeof(T)), "malloc"); return static_cast<T*>(p); }
template<class T> std::vector<T> get(const T* p, std::size_t n) {
    std::vector<T> v(n);
    ok(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}
template<class T> void put(T* p, const std::vector<T>& v) { ok(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
std::uint32_t bits(float f) { std::uint32_t b; std::memcpy(&b, &f, 4); return b; }

float e4m3(std::uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? std::ldexp(float(m) / 8.0f, -6) : std::ldexp(1.0f + float(m) / 8.0f, e - 7);
    return b & 0x80 ? -v : v;
}
int value(unsigned code, bool int4) {
    if(int4) return code < 8 ? int(code) : int(code) - 16;
    const int magnitude[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    return code & 8u ? -magnitude[code & 7u] : magnitude[code & 7u];
}

// The declared order: block sums exact, lane b % 16 = fma(sum, ws * as, lane), halving tree, post.
float oracle(const std::uint8_t* codes, const std::uint8_t* scales, int cols, bool int4, const std::int8_t* x,
             const float* xs, int ablock, float post) {
    float acc[16] = {};
    for(int b = 0; b < cols / 16; ++b) {
        int sum = 0;
        for(int e = 0; e < 16; ++e) {
            const int k = 16 * b + e;
            sum += value(k % 2 ? codes[k / 2] >> 4 : codes[k / 2] & 15, int4) * x[k];
        }
        acc[b % 16] = std::fma(float(sum), e4m3(scales[b]) * xs[b / (ablock / 16)], acc[b % 16]);
    }
    for(int w = 8; w >= 1; w /= 2)
        for(int i = 0; i < w; ++i) acc[i] = acc[i] + acc[i + w];
    return acc[0] * post;
}

struct Problem {
    int rows, cols, tokens, ablock;
    bool int4;
    std::vector<std::uint8_t> codes, scales;
    std::vector<std::int8_t> x;
    std::vector<float> xs;
};

Problem make(int rows, int cols, int tokens, int ablock, bool int4, std::mt19937& rng) {
    Problem p{rows, cols, tokens, ablock, int4, std::vector<std::uint8_t>(std::size_t(rows) * cols / 2),
              std::vector<std::uint8_t>(std::size_t(rows) * cols / 16), std::vector<std::int8_t>(std::size_t(tokens) * cols),
              std::vector<float>(std::size_t(tokens) * cols / ablock)};
    std::uniform_int_distribution<int> byte(0, 255), scale(0x20, 0x5f), act(-127, 127);
    std::uniform_real_distribution<float> s(0.001f, 0.1f);
    for(auto& v : p.codes) v = std::uint8_t(byte(rng));
    for(auto& v : p.scales) v = std::uint8_t(scale(rng) | (byte(rng) & 0x80));
    for(auto& v : p.x) v = std::int8_t(act(rng));
    for(auto& v : p.xs) v = s(rng);
    return p;
}

void check_kernels() {
    std::mt19937 rng(17);
    int checked = 0;
    for(bool int4 : {false, true})
        for(int ablock : {16, 32})
            for(int cols : {64, 704, 2560}) {
                const int rows = 200, tokens = 70;
                Problem p = make(rows, cols, tokens, ablock, int4, rng);
                auto* codes = dev<std::uint8_t>(p.codes.size());
                auto* scales = dev<std::uint8_t>(p.scales.size());
                auto* x = dev<std::int8_t>(p.x.size());
                auto* xs = dev<float>(p.xs.size());
                put(codes, p.codes); put(scales, p.scales); put(x, p.x); put(xs, p.xs);
                ib::Matrix m{codes, scales, rows, cols, 0.375f, int4 ? 1 : 0};
                auto* mdev = dev<ib::Matrix>(1);
                put(mdev, std::vector<ib::Matrix>{m});
                const ib::Acts acts{x, xs, cols, cols / ablock, ablock};
                std::vector<float> expect(std::size_t(tokens) * rows);
                for(int t = 0; t < tokens; ++t)
                    for(int r = 0; r < rows; ++r)
                        expect[std::size_t(t) * rows + r] =
                            oracle(p.codes.data() + std::size_t(r) * cols / 2, p.scales.data() + std::size_t(r) * cols / 16,
                                   cols, int4, p.x.data() + std::size_t(t) * cols, p.xs.data() + std::size_t(t) * cols / ablock,
                                   ablock, 0.375f);
                // Decode: jobs of 1..8 tokens over the 70 rows of activations, out rows = activation rows.
                std::vector<ib::RowJob> jobs;
                for(int t0 = 0, n = 1; t0 < tokens; t0 += n, n = n % 8 + 1) {
                    ib::RowJob j;
                    j.tokens = std::min(n, tokens - t0);
                    for(int i = 0; i < j.tokens; ++i) j.act_row[i] = j.out_row[i] = t0 + i;
                    jobs.push_back(j);
                }
                auto* jdev = dev<ib::RowJob>(jobs.size());
                put(jdev, jobs);
                auto* out = dev<float>(expect.size());
                ok(ib::enqueue_rows(mdev, jdev, int(jobs.size()), rows, acts, out, rows, nullptr), "rows");
                ok(cudaDeviceSynchronize(), "rows sync");
                auto got = get(out, expect.size());
                for(std::size_t i = 0; i < got.size(); ++i)
                    require(bits(got[i]) == bits(expect[i]), "rows differ at " + std::to_string(i) + " cols " + std::to_string(cols));
                // Prompt: jobs of 1, 15, 16, 17 and the rest of the rows.
                std::vector<ib::GemmJob> gjobs;
                for(int first = 0, i = 0; first < tokens; ++i) {
                    const int sizes[4] = {1, 15, 16, 17};
                    const int count = std::min(i < 4 ? sizes[i] : tokens, tokens - first);
                    gjobs.push_back({0, first, count});
                    first += count;
                }
                auto* gdev = dev<ib::GemmJob>(gjobs.size());
                put(gdev, gjobs);
                ok(cudaMemset(out, 0xff, expect.size() * sizeof(float)), "clear");
                ok(ib::enqueue_gemm(mdev, gdev, int(gjobs.size()), rows, tokens, acts, out, rows, nullptr), "gemm");
                ok(cudaDeviceSynchronize(), "gemm sync");
                got = get(out, expect.size());
                for(std::size_t i = 0; i < got.size(); ++i)
                    require(bits(got[i]) == bits(expect[i]), "gemm differs at " + std::to_string(i) + " cols " + std::to_string(cols));
                ok(cudaMemset(out, 0xff, expect.size() * sizeof(float)), "clear");
                ok(ib::enqueue_gemm_staged(mdev, gdev, int(gjobs.size()), rows, tokens, acts, out, rows, nullptr), "staged");
                ok(cudaDeviceSynchronize(), "staged sync");
                got = get(out, expect.size());
                for(std::size_t i = 0; i < got.size(); ++i)
                    require(bits(got[i]) == bits(expect[i]),
                            "staged gemm differs at " + std::to_string(i) + " cols " + std::to_string(cols));
                checked += 3;
                for(void* q : {(void*)codes, (void*)scales, (void*)x, (void*)xs, (void*)mdev, (void*)jdev, (void*)gdev, (void*)out}) cudaFree(q);
            }
    std::cout << "rows, gemm and staged gemm: " << checked << " problems bitwise equal to the declared order\n";
}

// The declared quantizers and SwiGLU, as crit's host_int.cpp writes them (compiled without contraction by nvcc's
// host compiler for these plain expressions; std::fma where the order has an FMA).
std::uint8_t e2m1_encode(float x) {
    const float m = std::fmin(std::fabs(x), 6.0f);
    const float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    unsigned code = 0;
    while(code < 7 && (m > mids[code] || (m == mids[code] && code % 2 == 1))) ++code;
    return std::uint8_t(code | (x < 0.0f && code ? 8u : 0u));
}
std::uint8_t e4m3_encode(float x) {
    const float m = std::fmin(std::fabs(x), 448.0f);
    unsigned best = 0;
    for(unsigned i = 1; i < 0x7F; ++i) {
        const float d = std::fabs(m - e4m3(std::uint8_t(i))), bd = std::fabs(m - e4m3(std::uint8_t(best)));
        if(d < bd || (d == bd && i % 2 == 0 && best % 2 == 1)) best = i;
    }
    return std::uint8_t(best | (x < 0.0f && best ? 0x80u : 0u));
}
float e2m1(unsigned code) {
    const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return code & 8u ? -magnitude[code & 7u] : magnitude[code & 7u];
}
volatile float sink;
float mul(float a, float b) { sink = a * b; return sink; }   // one rounded product, never contracted
float div(float a, float b) { sink = a / b; return sink; }
float exp_declared(float x) {
    x = x > 88.0f ? 88.0f : x < -87.0f ? -87.0f : x;
    const float n = std::nearbyint(mul(x, 1.44269504f));
    float r = std::fma(n, -0.693145752f, x);
    r = std::fma(n, -1.42860677e-06f, r);
    float p = 1.98412698e-04f;
    for(float c : {1.38888889e-03f, 8.33333333e-03f, 4.16666667e-02f, 1.66666667e-01f, 0.5f, 1.0f, 1.0f}) p = std::fma(p, r, c);
    const std::uint32_t b = std::uint32_t(int(n) + 127) << 23;
    float scale;
    std::memcpy(&scale, &b, 4);
    return mul(p, scale);
}

void check_quantizers() {
    std::mt19937 rng(9);
    std::normal_distribution<float> n(0.0f, 3.0f);
    const int rows = 13, cols = 2560;
    std::vector<float> host(std::size_t(rows) * cols);
    for(auto& v : host) v = n(rng);
    auto* x = dev<float>(host.size());
    put(x, host);
    auto* values = dev<std::int8_t>(host.size());
    auto* scales = dev<float>(host.size() / 16);
    const float global = 64.0f;
    ok(ib::enqueue_quantize_e2m1x2(x, cols, rows, cols, global, {values, scales, cols, cols / 16, 16}, nullptr), "e2m1");
    ok(cudaDeviceSynchronize(), "e2m1 sync");
    auto q = get(values, host.size());
    auto s = get(scales, host.size() / 16);
    for(std::size_t b = 0; b < host.size() / 16; ++b) {
        float amax = 0.0f;
        for(int i = 0; i < 16; ++i) amax = std::fmax(amax, std::fabs(host[16 * b + i]));
        const float decoded = e4m3(e4m3_encode(div(mul(amax, global), 6.0f)));
        require(bits(s[b]) == bits(div(decoded, global)), "e2m1 scale");
        for(int i = 0; i < 16; ++i)
            require(float(q[16 * b + i]) == 2.0f * e2m1(e2m1_encode(div(mul(host[16 * b + i], global), decoded))), "e2m1 value");
    }
    ok(ib::enqueue_quantize_int8(x, cols, rows, cols, {values, scales, cols, cols / 32, 32}, nullptr), "int8");
    ok(cudaDeviceSynchronize(), "int8 sync");
    q = get(values, host.size());
    s = get(scales, host.size() / 32);
    for(std::size_t b = 0; b < host.size() / 32; ++b) {
        float amax = 0.0f;
        for(int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(host[32 * b + i]));
        const float d = div(amax, 127.0f);
        require(bits(s[b]) == bits(d), "int8 scale");
        for(int i = 0; i < 32; ++i) {
            float v = std::nearbyint(div(host[32 * b + i], d));
            v = v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v;
            require(float(q[32 * b + i]) == v, "int8 value");
        }
    }
    const int inter = 640;
    auto* out = dev<float>(std::size_t(4) * inter);
    ok(ib::enqueue_swiglu(x, inter, 4, out, nullptr), "swiglu");
    ok(cudaDeviceSynchronize(), "swiglu sync");
    const auto y = get(out, std::size_t(4) * inter);
    for(int r = 0; r < 4; ++r)
        for(int k = 0; k < inter; ++k) {
            const float gate = host[std::size_t(r) * 2 * inter + k], up = host[std::size_t(r) * 2 * inter + inter + k];
            sink = 1.0f + exp_declared(-gate);
            const float silu = div(gate, sink);
            require(bits(y[std::size_t(r) * inter + k]) == bits(mul(silu, up)), "swiglu");
        }
    for(void* p : {(void*)x, (void*)values, (void*)scales, (void*)out}) cudaFree(p);
    std::cout << "quantizers and swiglu: bitwise equal to the declared functions\n";
}

void rates() {
    std::mt19937 rng(1);
    const int rows = 1920, cols = 2560;
    Problem p = make(rows, cols, 4096, 32, false, rng);
    auto* codes = dev<std::uint8_t>(p.codes.size());
    auto* scales = dev<std::uint8_t>(p.scales.size());
    auto* x = dev<std::int8_t>(p.x.size());
    auto* xs = dev<float>(p.xs.size());
    put(codes, p.codes); put(scales, p.scales); put(x, p.x); put(xs, p.xs);
    auto* mdev = dev<ib::Matrix>(1);
    put(mdev, std::vector<ib::Matrix>{{codes, scales, rows, cols, 1.0f, 0}});
    const ib::Acts acts{x, xs, cols, cols / 32, 32};
    auto* out = dev<float>(std::size_t(4096) * rows);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for(int tokens : {1, 3, 8}) {
        ib::RowJob j;
        j.tokens = tokens;
        for(int i = 0; i < tokens; ++i) j.act_row[i] = j.out_row[i] = i;
        std::vector<ib::RowJob> jobs(64, j);   // 64 jobs on the same matrix: bytes from L2 after the first
        auto* jdev = dev<ib::RowJob>(jobs.size());
        put(jdev, jobs);
        cudaEventRecord(a);
        ok(ib::enqueue_rows(mdev, jdev, int(jobs.size()), rows, acts, out, rows, nullptr), "rows rate");
        cudaEventRecord(b);
        ok(cudaEventSynchronize(b), "rows rate sync");
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        std::cout << "  rows, " << tokens << " tokens: " << ms * 1000 / jobs.size() << " us per 2.76 MB matrix\n";
        cudaFree(jdev);
    }
    std::vector<ib::GemmJob> g{{0, 0, 4096}};
    auto* gdev = dev<ib::GemmJob>(1);
    put(gdev, g);
    ok(ib::enqueue_gemm(mdev, gdev, 1, rows, 4096, acts, out, rows, nullptr), "gemm warm");
    cudaEventRecord(a);
    ok(ib::enqueue_gemm(mdev, gdev, 1, rows, 4096, acts, out, rows, nullptr), "gemm rate");
    cudaEventRecord(b);
    ok(cudaEventSynchronize(b), "gemm rate sync");
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    std::cout << "  gemm, 4096 tokens x 1920 x 2560: " << ms << " ms, " << 2.0 * 4096 * rows * cols / ms / 1e9 << " TOPS\n";
    ok(ib::enqueue_gemm_staged(mdev, gdev, 1, rows, 4096, acts, out, rows, nullptr), "staged warm");
    cudaEventRecord(a);
    ok(ib::enqueue_gemm_staged(mdev, gdev, 1, rows, 4096, acts, out, rows, nullptr), "staged rate");
    cudaEventRecord(b);
    ok(cudaEventSynchronize(b), "staged rate sync");
    cudaEventElapsedTime(&ms, a, b);
    std::cout << "  staged gemm, 4096 tokens x 1920 x 2560: " << ms << " ms, " << 2.0 * 4096 * rows * cols / ms / 1e9
              << " TOPS\n";
}
}  // namespace

int main() {
    try {
        check_kernels();
        check_quantizers();
        rates();
        std::cout << "int_block tests passed\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
