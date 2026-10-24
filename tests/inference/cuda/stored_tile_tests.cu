// Stored-tile FP8 / NVFP4 prompt GEMMs against an FP64 host reference with the same activation quantization.
#include "stored_tile_gemm.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
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
    return ok;
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
    return ok;
}

}  // namespace

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    if(cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::printf("no CUDA device; skipped\n");
        return 0;
    }
    const int cc = prop.major * 10 + prop.minor;
    std::mt19937 rng(7);
    bool ok = true;
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
        ok = fp4_rejects_plan_mismatch() && ok;
    } else {
        std::printf("NVFP4 block-scaled MMA not in this build (needs sm_120a / sm_121a); skipped\n");
    }
    std::printf(ok ? "stored-tile GEMMs passed\n" : "stored-tile GEMMs FAILED\n");
    return ok ? 0 : 1;
}
