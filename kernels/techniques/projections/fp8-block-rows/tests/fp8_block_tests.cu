// Matched FP8 projections: the decode rows and the prompt GEMM give the declared order's bits for every output (a
// reference kernel here, one warp per 16 x 8 tile as crit's fp8_block_device.cuh writes it) for MXFP8, 128 x 128
// block-scaled and row-scaled weights, decode jobs of 1 to 8 tokens in any order and prompt jobs of any split (the
// register GEMM and the staged one); every output within a bound of its FP64 value; the quantizer gives the host
// twin's codes and scales from FP32 and BF16.
// Prints the decode rate from DRAM and the GEMM rate (not asserted). Needs sm_89 or newer.
#include "fp8_block.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fb = crit::techniques::fp8_block;

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

float e4m3(std::uint8_t c) {
    const int e = (c >> 3) & 15, m = c & 7;
    const float v = e == 0 ? std::ldexp(float(m), -9) : std::ldexp(float(8 + m), e - 10);
    return c & 0x80 ? -v : v;
}
// Nearest E4M3 (ties to the even code), saturating at 448: the hardware conversion's result.
std::uint8_t e4m3_encode(float v) {
    const float a = std::fabs(v);
    unsigned best = 0;
    for(unsigned c = 1; c < 0x7f; ++c) {
        const float d = std::fabs(a - e4m3(std::uint8_t(c))), bd = std::fabs(a - e4m3(std::uint8_t(best)));
        if(d < bd || (d == bd && c % 2 == 0)) best = c;
    }
    return std::uint8_t(best | (std::signbit(v) ? 0x80u : 0u));
}
float quantize_host(const float* v, int n, fb::ScaleKind kind, std::uint8_t* codes) {
    float amax = 0.0f;
    for(int k = 0; k < n; ++k) amax = std::fmax(amax, std::fabs(v[k]));
    float scale;
    if(kind == fb::ScaleKind::mx) {
        int e = 0;
        (void)std::frexp(amax, &e);
        int shared = amax > 0.0f ? e - 1 - 8 : 0;
        shared = std::clamp(shared, -127, 127);
        scale = std::ldexp(1.0f, shared);
    } else {
        scale = amax / 448.0f;
    }
    for(int k = 0; k < n; ++k) codes[k] = scale > 0.0f ? e4m3_encode(v[k] / scale) : 0;
    return scale;
}

// The declared order as crit writes it: one warp per tile of 16 rows and 8 tokens, all 16 lanes held, the tree at the
// end.
__global__ void reference_kernel(fb::Matrix m, fb::Acts x, int tokens, float* out) {
    const int lane = int(threadIdx.x), g = lane / 4, q = lane % 4;
    const int row0 = int(blockIdx.x) * 16, tok0 = int(blockIdx.y) * 8;
    const int ra = min(row0 + g, m.rows - 1), rb = min(row0 + g + 8, m.rows - 1), tg = min(tok0 + g, tokens - 1);
    const int r[2] = {row0 + g, row0 + g + 8}, t[2] = {tok0 + 2 * q, tok0 + 2 * q + 1};
    float lanes[4][16] = {};
    for(int b = 0; b < m.cols / 32; ++b) {
        const uint2 a_g = *reinterpret_cast<const uint2*>(m.codes + std::size_t(ra) * m.cols + 32 * b + 8 * q);
        const uint2 a_g8 = *reinterpret_cast<const uint2*>(m.codes + std::size_t(rb) * m.cols + 32 * b + 8 * q);
        const uint2 t_g = *reinterpret_cast<const uint2*>(x.codes + std::size_t(tg) * x.ld + 32 * b + 8 * q);
        float d[4];
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                     "{%10, %10, %10, %10};\n"
                     : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                     : "r"(a_g.x), "r"(a_g8.x), "r"(a_g.y), "r"(a_g8.y), "r"(t_g.x), "r"(t_g.y), "f"(0.0f));
        for(int o = 0; o < 4; ++o) {
            const int row = min(r[o / 2], m.rows - 1), token = min(t[o % 2], tokens - 1);
            const float ws = static_cast<const float*>(m.scales)[std::size_t(row / m.scale_rows) * (m.cols / m.scale_k) +
                                                                  32 * b / m.scale_k];
            const float as = x.scales[std::size_t(token) * x.lds + 32 * b / x.group];
            lanes[o][b % 16] = __fmaf_rn(d[o], __fmul_rn(ws, as), lanes[o][b % 16]);
        }
    }
    for(int o = 0; o < 4; ++o) {
        for(int w = 8; w >= 1; w /= 2)
            for(int i = 0; i < w; ++i) lanes[o][i] = __fadd_rn(lanes[o][i], lanes[o][i + w]);
        if(r[o / 2] < m.rows && t[o % 2] < tokens) out[std::size_t(t[o % 2]) * m.rows + r[o / 2]] = __fmul_rn(lanes[o][0], m.post);
    }
}

__global__ void stream_read(const std::uint8_t* p, std::size_t n16, float* out) {
    unsigned acc = 0;
    for(std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += std::size_t(gridDim.x) * blockDim.x) {
        const uint4 v = reinterpret_cast<const uint4*>(p)[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if(acc == 0x12345678u) out[0] = 1.0f;
}

struct Format {
    const char* name;
    int scale_rows, scale_k, group;
    fb::ScaleKind kind;
};

void check() {
    std::mt19937 rng(17);
    const Format formats[] = {{"MXFP8", 1, 32, 32, fb::ScaleKind::mx},
                              {"128 x 128 blocks", 128, 128, 128, fb::ScaleKind::amax},
                              {"row scales", 1, 0, 0, fb::ScaleKind::amax}};
    int problems = 0;
    double worst = 0.0;
    for(const auto& f : formats)
        for(int cols : {32, 128, 704, 2560}) {
            const int scale_k = f.scale_k ? f.scale_k : cols, group = f.group ? f.group : cols;
            if(cols % scale_k || cols % group) continue;
            for(int rows : {50, 300}) {
                const int tokens = 45;
                std::vector<std::uint8_t> codes(std::size_t(rows) * cols);
                for(auto& c : codes) do c = std::uint8_t(rng()); while((c & 0x7f) == 0x7f);
                const int srows = (rows + f.scale_rows - 1) / f.scale_rows, sld = cols / scale_k;
                std::vector<float> ws(std::size_t(srows) * sld);
                for(auto& s : ws) s = f.kind == fb::ScaleKind::mx ? std::ldexp(1.0f, int(rng() % 13) - 6)
                                                                  : 0.001f + float(rng() % 1000) * 1e-5f;
                std::normal_distribution<float> normal(0.0f, 1.0f);
                std::vector<float> xf(std::size_t(tokens) * cols);
                for(auto& v : xf) v = normal(rng);
                for(int k = 0; k < 6; ++k) xf[rng() % xf.size()] = 30.0f * normal(rng);
                std::uint8_t* d_codes = dev<std::uint8_t>(codes.size());
                float* d_ws = dev<float>(ws.size());
                float* d_xf = dev<float>(xf.size());
                std::uint8_t* d_x = dev<std::uint8_t>(xf.size());
                float* d_xs = dev<float>(std::size_t(tokens) * (cols / group));
                put(d_codes, codes);
                put(d_ws, ws);
                put(d_xf, xf);
                const fb::Acts a{d_x, d_xs, cols, cols / group, group};
                ok(fb::enqueue_quantize(d_xf, false, cols, tokens, cols, f.kind, a, nullptr), "quantize");
                const auto x = get(d_x, xf.size());
                const auto xs = get(d_xs, std::size_t(tokens) * (cols / group));
                for(int t = 0; t < tokens; ++t)
                    for(int k = 0; k < cols / group; ++k) {
                        std::vector<std::uint8_t> hc(static_cast<std::size_t>(group));
                        const float hs = quantize_host(&xf[std::size_t(t) * cols + k * group], group, f.kind, hc.data());
                        require(bits(hs) == bits(xs[std::size_t(t) * (cols / group) + k]) &&
                                    std::equal(hc.begin(), hc.end(), x.begin() + std::ptrdiff_t(t * cols + k * group)),
                                std::string(f.name) + ": quantizer differs from the host twin");
                    }
                const fb::Matrix m{d_codes, d_ws, rows, cols, f.scale_rows, scale_k, 0.5f, 0};
                // MXFP8 runs the hub kernels from the E8M0 bytes, the reference from their FP32 values.
                std::uint8_t* d_e8 = dev<std::uint8_t>(ws.size());
                fb::Matrix stored = m;
                if(f.kind == fb::ScaleKind::mx) {
                    std::vector<std::uint8_t> e8(ws.size());
                    for(std::size_t i = 0; i < ws.size(); ++i) {
                        int e = 0;
                        (void)std::frexp(ws[i], &e);
                        e8[i] = std::uint8_t(e - 1 + 127);
                        require(std::ldexp(1.0f, int(e8[i]) - 127) == ws[i], "E8M0 scale");
                    }
                    put(d_e8, e8);
                    stored.scales = d_e8;
                    stored.scale_e8m0 = 1;
                }
                fb::Matrix* d_m = dev<fb::Matrix>(1);
                put(d_m, std::vector<fb::Matrix>{stored});
                const std::size_t n = std::size_t(tokens) * rows;
                float* d_ref = dev<float>(n);
                float* d_out = dev<float>(n);
                reference_kernel<<<dim3(unsigned((rows + 15) / 16), unsigned((tokens + 7) / 8)), 32>>>(m, a, tokens, d_ref);
                ok(cudaDeviceSynchronize(), "reference");
                const auto ref = get(d_ref, n);
                auto compare = [&](const char* what) {
                    const auto got = get(d_out, n);
                    for(std::size_t i = 0; i < n; ++i)
                        require(bits(got[i]) == bits(ref[i]), std::string(f.name) + ": " + what + " differs at token " +
                                                                  std::to_string(i / std::size_t(rows)) + ", row " +
                                                                  std::to_string(i % std::size_t(rows)) + " (cols " +
                                                                  std::to_string(cols) + ")");
                    ++problems;
                };
                // Decode: tokens in shuffled jobs of 1 to 8.
                std::vector<int> order(static_cast<std::size_t>(tokens));
                std::iota(order.begin(), order.end(), 0);
                std::shuffle(order.begin(), order.end(), rng);
                std::vector<fb::RowJob> jobs;
                for(std::size_t at = 0; at < order.size();) {
                    fb::RowJob j;
                    j.tokens = std::min<int>(1 + int(rng() % 8u), int(order.size() - at));
                    for(int k = 0; k < j.tokens; ++k, ++at) j.act_row[k] = j.out_row[k] = order[at];
                    jobs.push_back(j);
                }
                fb::RowJob* d_jobs = dev<fb::RowJob>(jobs.size());
                put(d_jobs, jobs);
                ok(cudaMemset(d_out, 0xff, n * 4), "clear");
                ok(fb::enqueue_rows(d_m, d_jobs, int(jobs.size()), rows, a, d_out, rows, nullptr), "rows");
                ok(cudaDeviceSynchronize(), "rows run");
                compare("decode rows");
                // Prompt: the tokens split into jobs at random points.
                std::vector<fb::GemmJob> gjobs;
                for(int first = 0; first < tokens;) {
                    const int count = std::min(tokens - first, 1 + int(rng() % 40u));
                    gjobs.push_back({0, first, count});
                    first += count;
                }
                fb::GemmJob* d_gjobs = dev<fb::GemmJob>(gjobs.size());
                put(d_gjobs, gjobs);
                ok(cudaMemset(d_out, 0xff, n * 4), "clear");
                ok(fb::enqueue_gemm(d_m, d_gjobs, int(gjobs.size()), rows, 40, a, d_out, rows, nullptr), "gemm");
                ok(cudaDeviceSynchronize(), "gemm run");
                compare("prompt gemm");
                ok(cudaMemset(d_out, 0xff, n * 4), "clear");
                ok(fb::enqueue_gemm_staged(d_m, d_gjobs, int(gjobs.size()), rows, 40, a, d_out, rows, nullptr), "staged");
                ok(cudaDeviceSynchronize(), "staged gemm run");
                compare("staged prompt gemm");
                for(int t = 0; t < tokens; ++t)
                    for(int r = 0; r < rows; ++r) {
                        double sum = 0.0, magnitude = 0.0;
                        for(int k = 0; k < cols; ++k) {
                            const double p = double(e4m3(codes[std::size_t(r) * cols + k])) * double(e4m3(x[std::size_t(t) * cols + k])) *
                                             double(ws[std::size_t(r / f.scale_rows) * sld + k / scale_k]) *
                                             double(xs[std::size_t(t) * (cols / group) + k / group]);
                            sum += p;
                            magnitude += std::fabs(p);
                        }
                        const double err = std::fabs(double(ref[std::size_t(t) * rows + r]) - 0.5 * sum);
                        if(magnitude > 0.0) worst = std::max(worst, err / (0.5 * magnitude));
                        require(err <= std::ldexp(0.5 * magnitude, -9) + 1e-30, std::string(f.name) + ": far from FP64");
                    }
                for(void* p : {(void*)d_codes, (void*)d_ws, (void*)d_xf, (void*)d_x, (void*)d_xs, (void*)d_m, (void*)d_ref,
                               (void*)d_out, (void*)d_jobs, (void*)d_gjobs, (void*)d_e8})
                    cudaFree(p);
            }
        }
    // BF16 input quantizes like its FP32 value.
    std::vector<__nv_bfloat16> hb(256);
    std::vector<float> hf(256);
    for(std::size_t i = 0; i < hb.size(); ++i) {
        hb[i] = __float2bfloat16(std::sin(float(i)) * 7.0f);
        hf[i] = __bfloat162float(hb[i]);
    }
    __nv_bfloat16* d_b = dev<__nv_bfloat16>(hb.size());
    std::uint8_t* d_c = dev<std::uint8_t>(hb.size());
    float* d_s = dev<float>(8);
    put(d_b, hb);
    ok(fb::enqueue_quantize(d_b, true, 256, 1, 256, fb::ScaleKind::mx, {d_c, d_s, 256, 8, 32}, nullptr), "bf16 quantize");
    const auto c = get(d_c, hb.size());
    const auto s = get(d_s, 8);
    for(int k = 0; k < 8; ++k) {
        std::uint8_t hc[32];
        require(bits(quantize_host(&hf[std::size_t(k) * 32], 32, fb::ScaleKind::mx, hc)) == bits(s[std::size_t(k)]) &&
                    std::equal(hc, hc + 32, c.begin() + 32 * k),
                "BF16 quantizer differs");
    }
    std::cout << "fp8 block: " << problems << " decode and prompt runs bitwise equal to the declared order; worst "
              << "error against FP64 " << worst << " of the products' magnitude; quantizer matches (FP32, BF16)\n";
}

void timing() {
    // Decode from DRAM: 48 MXFP8 matrices of 5120 x 2560 (630 MB, past the L2), one job each; rates count the codes.
    const int rows = 5120, cols = 2560, count = 48;
    std::vector<std::uint8_t*> codes(count);
    for(auto& p : codes) {
        p = dev<std::uint8_t>(std::size_t(rows) * cols);
        ok(cudaMemset(p, 0x3a, std::size_t(rows) * cols), "fill");
    }
    // MXFP8 as stored: one E8M0 byte (127, the scale 1) per 32 codes.
    std::uint8_t* scales = dev<std::uint8_t>(std::size_t(rows) * (cols / 32));
    ok(cudaMemset(scales, 127, std::size_t(rows) * (cols / 32)), "fill");
    std::vector<fb::Matrix> ms;
    for(auto* p : codes) ms.push_back({p, scales, rows, cols, 1, 32, 1.0f, 1});
    fb::Matrix* d_m = dev<fb::Matrix>(ms.size());
    put(d_m, ms);
    const int max_tokens = 4096;
    std::uint8_t* x = dev<std::uint8_t>(std::size_t(max_tokens) * cols);
    ok(cudaMemset(x, 0x38, std::size_t(max_tokens) * cols), "fill");
    float* xs = dev<float>(std::size_t(max_tokens) * (cols / 32));
    std::vector<float> xones(std::size_t(max_tokens) * (cols / 32), 1.0f);
    put(xs, xones);
    const fb::Acts a{x, xs, cols, cols / 32, 32};
    float* out = dev<float>(std::size_t(max_tokens) * rows);
    cudaEvent_t e0, e1;
    ok(cudaEventCreate(&e0), "event");
    ok(cudaEventCreate(&e1), "event");
    {
        // The attainable read rate at this size: a plain streaming read of the same matrices, back to back.
        for(int warm = 0; warm < 2; ++warm)
            for(int i = 0; i < count; ++i) stream_read<<<1360, 256>>>(codes[std::size_t(i)], std::size_t(rows) * cols / 16, out);
        ok(cudaEventRecord(e0), "record");
        for(int r = 0; r < 5; ++r)
            for(int i = 0; i < count; ++i) stream_read<<<1360, 256>>>(codes[std::size_t(i)], std::size_t(rows) * cols / 16, out);
        ok(cudaEventRecord(e1), "record");
        ok(cudaEventSynchronize(e1), "sync");
        float ms_total = 0.0f;
        ok(cudaEventElapsedTime(&ms_total, e0, e1), "elapsed");
        const double per = double(ms_total) / (5 * count);
        std::cout << "  plain read of the same matrices: " << per * 1000.0 << " us each, "
                  << double(rows) * cols / (per * 1e-3) / 1e9 << " GB/s\n";
    }
    for(int tokens : {1, 2, 4, 8}) {
        fb::RowJob j;
        j.tokens = tokens;
        for(int k = 0; k < tokens; ++k) j.act_row[k] = j.out_row[k] = k;
        fb::RowJob* d_j = dev<fb::RowJob>(1);
        put(d_j, std::vector<fb::RowJob>{j});
        for(int warm = 0; warm < 2; ++warm)
            for(int i = 0; i < count; ++i) ok(fb::enqueue_rows(d_m + i, d_j, 1, rows, a, out, rows, nullptr), "rows");
        ok(cudaEventRecord(e0), "record");
        const int reps = 5;
        for(int r = 0; r < reps; ++r)
            for(int i = 0; i < count; ++i) ok(fb::enqueue_rows(d_m + i, d_j, 1, rows, a, out, rows, nullptr), "rows");
        ok(cudaEventRecord(e1), "record");
        ok(cudaEventSynchronize(e1), "sync");
        float ms_total = 0.0f;
        ok(cudaEventElapsedTime(&ms_total, e0, e1), "elapsed");
        const double per = double(ms_total) / (reps * count);
        std::cout << "  decode rows, " << tokens << " token(s): " << per * 1000.0 << " us per 13.1 MB matrix, "
                  << double(rows) * cols / (per * 1e-3) / 1e9 << " GB/s\n";
        cudaFree(d_j);
    }
    for(int tokens : {512, 4096}) {
        fb::GemmJob j{0, 0, tokens};
        fb::GemmJob* d_j = dev<fb::GemmJob>(1);
        put(d_j, std::vector<fb::GemmJob>{j});
        ok(fb::enqueue_gemm(d_m, d_j, 1, rows, tokens, a, out, rows, nullptr), "gemm");
        ok(cudaEventRecord(e0), "record");
        const int reps = 10;
        for(int r = 0; r < reps; ++r) ok(fb::enqueue_gemm(d_m + r % count, d_j, 1, rows, tokens, a, out, rows, nullptr), "gemm");
        ok(cudaEventRecord(e1), "record");
        ok(cudaEventSynchronize(e1), "sync");
        float ms_total = 0.0f;
        ok(cudaEventElapsedTime(&ms_total, e0, e1), "elapsed");
        const double per = double(ms_total) / reps;
        std::cout << "  prompt gemm, " << tokens << " tokens x " << rows << " x " << cols << ": " << per << " ms, "
                  << 2.0 * tokens * rows * cols / (per * 1e-3) / 1e12 << " TFLOPS\n";
        ok(fb::enqueue_gemm_staged(d_m, d_j, 1, rows, tokens, a, out, rows, nullptr), "staged");
        ok(cudaEventRecord(e0), "record");
        for(int r = 0; r < reps; ++r)
            ok(fb::enqueue_gemm_staged(d_m + r % count, d_j, 1, rows, tokens, a, out, rows, nullptr), "staged");
        ok(cudaEventRecord(e1), "record");
        ok(cudaEventSynchronize(e1), "sync");
        ok(cudaEventElapsedTime(&ms_total, e0, e1), "elapsed");
        const double staged = double(ms_total) / reps;
        std::cout << "  staged prompt gemm, " << tokens << " tokens x " << rows << " x " << cols << ": " << staged
                  << " ms, " << 2.0 * tokens * rows * cols / (staged * 1e-3) / 1e12 << " TFLOPS\n";
        cudaFree(d_j);
    }
}
}  // namespace

int main() {
    try {
        int major = 0, minor = 0;
        ok(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), "major");
        ok(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), "minor");
        if(major * 10 + minor < 89) { std::cout << "e4m3 mma.sync needs sm_89 or newer; skipped\n"; return 0; }
        check();
        timing();
        std::cout << "fp8 block tests passed\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
