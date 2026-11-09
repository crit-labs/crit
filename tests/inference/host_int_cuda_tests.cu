// Integer block rows on the device: host_int_device.cuh's dot_row_int gives host_int's oracle bits (and so every host
// path's) for E2M1 x 2 and INT4 weights, 16- and 32-element activation blocks, 16 and 32 lanes and many shapes; the
// device quantizers, exp and SwiGLU give their declared host functions' bits.
#include "host_int.h"
#include "host_int_device.cuh"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace hi = crit::techniques::host_int;
namespace hn = crit::techniques::host_nvfp4;

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

template <int L>
__global__ void rows_kernel(const std::uint8_t* codes, const float* wscales, const std::int8_t* x, const float* xs,
                            int ablock, int rows, int cols, int tokens, bool int4, float post, float* out) {
    const int row = int(blockIdx.x) * (int(blockDim.x) / L) + int(threadIdx.x) / L;
    if(row >= rows) return;
    for(int t = 0; t < tokens; ++t) {
        const float v = hi::dot_row_int<L>(codes + std::size_t(row) * cols / 2, wscales + std::size_t(row) * cols / 16,
                                           x + std::size_t(t) * cols, xs + std::size_t(t) * cols / ablock, ablock, cols,
                                           int4, post);
        if(threadIdx.x % L == 0) out[std::size_t(t) * rows + row] = v;
    }
}

__global__ void functions_kernel(const float* x, int n, float global, std::int8_t* e2m1, float* e2m1_scales,
                                 std::int8_t* int8, float* int8_scales, float* exps, float* swiglu) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if(i < n / 16) e2m1_scales[i] = hi::quantize_e2m1x2_block(x + 16 * i, global, e2m1 + 16 * i);
    if(i < n / 32) int8_scales[i] = hi::quantize_int8_block(x + 32 * i, 32, int8 + 32 * i);
    if(i < n) exps[i] = hi::exp_declared_device(x[i]);
    if(i < n / 2) swiglu[i] = hi::swiglu_declared_device(x[2 * i], x[2 * i + 1]);
}

void check_rows() {
    std::mt19937 rng(23);
    int checked = 0;
    for(auto weights : {hi::Weights::e2m1x2, hi::Weights::int4})
        for(int ablock : {16, 32})
            for(int cols : {64, 192, 704, 2560}) {
                const int rows = 77, tokens = 3;
                std::vector<std::uint8_t> codes(std::size_t(rows) * cols / 2), scales(std::size_t(rows) * cols / 16);
                std::vector<std::int8_t> x(std::size_t(tokens) * cols);
                std::vector<float> xs(std::size_t(tokens) * cols / ablock);
                std::uniform_int_distribution<int> byte(0, 255), scale(0x20, 0x5f), act(-127, 127);
                std::uniform_real_distribution<float> s(0.001f, 0.1f);
                for(auto& v : codes) v = std::uint8_t(byte(rng));
                for(auto& v : scales) v = std::uint8_t(scale(rng) | (byte(rng) & 0x80));
                for(auto& v : x) v = std::int8_t(act(rng));
                for(auto& v : xs) v = s(rng);
                const hi::IntMatrix m{codes.data(), scales.data(), std::size_t(rows), std::size_t(cols), 0, 0, weights};
                const hn::MatrixView view{codes.data(), scales.data(), std::size_t(rows), std::size_t(cols)};
                const hn::ScaleDecoder decoder = hn::e4m3_scales(view);
                std::vector<float> decoded(scales.size());
                for(int r = 0; r < rows; ++r) decoder.decode(decoder.context, std::size_t(r), decoded.data() + std::size_t(r) * cols / 16);
                auto* dcodes = dev<std::uint8_t>(codes.size());
                auto* dscales = dev<float>(decoded.size());
                auto* dx = dev<std::int8_t>(x.size());
                auto* dxs = dev<float>(xs.size());
                auto* dout = dev<float>(std::size_t(tokens) * rows);
                put(dcodes, codes); put(dscales, decoded); put(dx, x); put(dxs, xs);
                for(int lanes : {16, 32}) {
                    std::vector<float> expect(std::size_t(tokens) * rows);
                    hi::dot_rows_int(hi::Isa::scalar, {lanes, hn::Reduce::halving}, m, decoder,
                                     {x.data(), std::size_t(cols), xs.data(), std::size_t(cols / ablock), ablock}, tokens,
                                     0xff, 0, std::size_t(rows), 0.375f, expect.data(), std::size_t(rows));
                    if(lanes == 16)
                        rows_kernel<16><<<(rows + 7) / 8, 128>>>(dcodes, dscales, dx, dxs, ablock, rows, cols, tokens,
                                                                 weights == hi::Weights::int4, 0.375f, dout);
                    else
                        rows_kernel<32><<<(rows + 3) / 4, 128>>>(dcodes, dscales, dx, dxs, ablock, rows, cols, tokens,
                                                                 weights == hi::Weights::int4, 0.375f, dout);
                    ok(cudaDeviceSynchronize(), "rows");
                    const auto got = get(dout, expect.size());
                    for(std::size_t i = 0; i < got.size(); ++i)
                        require(bits(got[i]) == bits(expect[i]), "device row differs at " + std::to_string(i) +
                                                                     " cols " + std::to_string(cols));
                    ++checked;
                }
                for(void* p : {(void*)dcodes, (void*)dscales, (void*)dx, (void*)dxs, (void*)dout}) cudaFree(p);
            }
    std::cout << "device integer rows: " << checked << " problems bitwise equal to the host oracle\n";
}

void check_functions() {
    std::mt19937 rng(4);
    std::normal_distribution<float> normal(0.0f, 4.0f);
    const int n = 1 << 16;
    std::vector<float> x(n);
    for(auto& v : x) v = normal(rng);
    for(int i = 0; i < 64; ++i) x[std::size_t(i)] = -90.0f + 2.9f * float(i);   // exp's clamps and range
    auto* dx = dev<float>(n);
    put(dx, x);
    auto* e2m1 = dev<std::int8_t>(n);
    auto* e2m1_scales = dev<float>(n / 16);
    auto* int8 = dev<std::int8_t>(n);
    auto* int8_scales = dev<float>(n / 32);
    auto* exps = dev<float>(n);
    auto* swiglu = dev<float>(n / 2);
    functions_kernel<<<n / 256, 256>>>(dx, n, 32.0f, e2m1, e2m1_scales, int8, int8_scales, exps, swiglu);
    ok(cudaDeviceSynchronize(), "functions");
    std::vector<std::int8_t> hq(n);
    std::vector<float> hs(n / 16);
    hi::quantize_e2m1x2(x.data(), std::size_t(n), 32.0f, hq.data(), hs.data());
    require(get(e2m1, n) == hq, "e2m1 values");
    const auto ds = get(e2m1_scales, n / 16);
    for(int i = 0; i < n / 16; ++i) require(bits(ds[std::size_t(i)]) == bits(hs[std::size_t(i)]), "e2m1 scales");
    std::vector<float> h8(n / 32);
    hi::quantize_int8(x.data(), std::size_t(n), 32, hq.data(), h8.data());
    require(get(int8, n) == hq, "int8 values");
    const auto d8 = get(int8_scales, n / 32);
    for(int i = 0; i < n / 32; ++i) require(bits(d8[std::size_t(i)]) == bits(h8[std::size_t(i)]), "int8 scales");
    const auto de = get(exps, n), dsw = get(swiglu, n / 2);
    for(int i = 0; i < n; ++i) require(bits(de[std::size_t(i)]) == bits(hi::exp_declared(x[std::size_t(i)])), "exp at " + std::to_string(i));
    for(int i = 0; i < n / 2; ++i)
        require(bits(dsw[std::size_t(i)]) == bits(hi::swiglu_declared(x[2 * std::size_t(i)], x[2 * std::size_t(i) + 1])), "swiglu");
    for(void* p : {(void*)dx, (void*)e2m1, (void*)e2m1_scales, (void*)int8, (void*)int8_scales, (void*)exps, (void*)swiglu}) cudaFree(p);
    std::cout << "device quantizers, exp and SwiGLU: bitwise equal to the host functions\n";
}
}  // namespace

int main() {
    try {
        check_rows();
        check_functions();
        std::cout << "host_int cuda tests passed\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
