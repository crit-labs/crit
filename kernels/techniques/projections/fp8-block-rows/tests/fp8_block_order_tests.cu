// The matched FP8 order itself (order/fp8_block_device.cuh): three kernels that implement the declared order differently (one warp
// per tile with all 16 lanes; lanes in kLaneOrder folded as they finish; lanes split over four warps with the tree in
// shared memory) give the same bits for every output, with rows and tokens moved to other tile positions and other
// companions, for MXFP8, 128 x 128 block-scaled and row-scaled weights; a repeat gives the same bits; every output is
// within a bound of its FP64 value; the device quantizer gives its host twin's codes and scales. Prints the largest
// error against FP64 relative to the sum of the products' magnitudes. Needs sm_89 or newer; skips otherwise.
// Build and run on the target: nvcc -std=c++20 -O2 -arch=sm_<cc> -I tests/order tests/fp8_block_order_tests.cu
#include "fp8_block.h"
#include "fp8_block_device.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fb = crit::fp8_block;

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

// Kernel 1: one warp per 16 x 8 tile, fb::tile as written.
__global__ void reference_kernel(fb::Matrix m, fb::Acts x, int tokens, float* out) {
    int row_of[16], token_of[8];
    for(int i = 0; i < 16; ++i) row_of[i] = int(blockIdx.x) * 16 + i < m.rows ? int(blockIdx.x) * 16 + i : -1;
    for(int c = 0; c < 8; ++c) token_of[c] = int(blockIdx.y) * 8 + c < tokens ? int(blockIdx.y) * 8 + c : -1;
    float v[4];
    fb::tile(m, x, row_of, token_of, v);
    const int g = int(threadIdx.x) / 4, q = int(threadIdx.x) % 4;
    for(int o = 0; o < 4; ++o) {
        const int row = row_of[g + 8 * (o / 2)], token = token_of[2 * q + o % 2];
        if(row >= 0 && token >= 0) out[std::size_t(token) * m.rows + row] = v[o];
    }
}

// Kernel 2: arbitrary tiles (rows_of [tile][16], tokens_of [tile][8]), lanes in kLaneOrder folded as each finishes.
__global__ void folded_kernel(fb::Matrix m, fb::Acts x, const int* rows_of, const int* tokens_of, float* out) {
    const int* row_of = rows_of + 16 * blockIdx.x;
    const int* token_of = tokens_of + 8 * blockIdx.x;
    const int g = int(threadIdx.x) / 4, q = int(threadIdx.x) % 4;
    const int r[2] = {row_of[g], row_of[g + 8]}, t[2] = {token_of[2 * q], token_of[2 * q + 1]};
    const int blocks = m.cols / fb::kBlock;
    float partial[4][4];
    for(int at = 0; at < fb::kLanes; ++at) {
        const int j = fb::kLaneOrder[at];
        float lane[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for(int b = j; b < blocks; b += fb::kLanes) {
            float d[4];
            fb::mma_block(fb::fragment(m.codes, std::size_t(m.cols), r[0], b, q),
                          fb::fragment(m.codes, std::size_t(m.cols), r[1], b, q),
                          fb::fragment(x.codes, std::size_t(x.ld), token_of[g], b, q), d);
            for(int o = 0; o < 4; ++o)
                if(r[o / 2] >= 0 && t[o % 2] >= 0)
                    lane[o] = fb::step(d[o], fb::weight_scale(m, r[o / 2], b), fb::act_scale(x, t[o % 2], b), lane[o]);
        }
        for(int o = 0; o < 4; ++o) fb::fold(at, lane[o], partial[o]);
    }
    for(int o = 0; o < 4; ++o)
        if(r[o / 2] >= 0 && t[o % 2] >= 0) out[std::size_t(t[o % 2]) * m.rows + r[o / 2]] = __fmul_rn(partial[o][0], m.post);
}

// Kernel 3: arbitrary tiles, four warps each computing lanes w, w + 4, w + 8, w + 12; the tree over shared memory, one
// thread an output.
__global__ void split_kernel(fb::Matrix m, fb::Acts x, const int* rows_of, const int* tokens_of, float* out) {
    __shared__ float lanes[fb::kLanes][16][8];
    const int* row_of = rows_of + 16 * blockIdx.x;
    const int* token_of = tokens_of + 8 * blockIdx.x;
    const int warp = int(threadIdx.x) / 32, lane_id = int(threadIdx.x) % 32, g = lane_id / 4, q = lane_id % 4;
    const int r[2] = {row_of[g], row_of[g + 8]}, t[2] = {token_of[2 * q], token_of[2 * q + 1]};
    const int blocks = m.cols / fb::kBlock;
    for(int j = warp; j < fb::kLanes; j += 4) {
        float lane[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for(int b = j; b < blocks; b += fb::kLanes) {
            float d[4];
            fb::mma_block(fb::fragment(m.codes, std::size_t(m.cols), r[0], b, q),
                          fb::fragment(m.codes, std::size_t(m.cols), r[1], b, q),
                          fb::fragment(x.codes, std::size_t(x.ld), token_of[g], b, q), d);
            for(int o = 0; o < 4; ++o)
                if(r[o / 2] >= 0 && t[o % 2] >= 0)
                    lane[o] = fb::step(d[o], fb::weight_scale(m, r[o / 2], b), fb::act_scale(x, t[o % 2], b), lane[o]);
        }
        for(int o = 0; o < 4; ++o) lanes[j][g + 8 * (o / 2)][2 * q + o % 2] = lane[o];
    }
    __syncthreads();
    const int pos_r = int(threadIdx.x) / 8, pos_t = int(threadIdx.x) % 8;
    const int row = row_of[pos_r], token = token_of[pos_t];
    if(row < 0 || token < 0) return;
    float v[fb::kLanes];
    for(int j = 0; j < fb::kLanes; ++j) v[j] = lanes[j][pos_r][pos_t];
    out[std::size_t(token) * m.rows + row] = __fmul_rn(fb::halving_tree(v), m.post);
}

__global__ void quantize_kernel(const float* v, int groups, int group, fb::ScaleKind kind, std::uint8_t* codes,
                                float* scales) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if(i < groups) scales[i] = fb::quantize_group_device(v + std::size_t(i) * group, group, kind, codes + std::size_t(i) * group);
}

struct Format {
    const char* name;
    int scale_rows, scale_k, group;
    fb::ScaleKind kind;
};

std::uint8_t random_code(std::mt19937& rng) {
    for(;;) {
        const auto c = std::uint8_t(rng() & 0xffu);
        if((c & 0x7fu) != 0x7fu) return c;
    }
}

// Every (row, token) once, in tiles with shuffled positions and holes.
void scrambled_tiles(int rows, int tokens, std::mt19937& rng, std::vector<int>& rows_of, std::vector<int>& tokens_of) {
    std::vector<int> r(static_cast<std::size_t>(rows)), t(static_cast<std::size_t>(tokens));
    std::iota(r.begin(), r.end(), 0);
    std::iota(t.begin(), t.end(), 0);
    std::shuffle(r.begin(), r.end(), rng);
    std::shuffle(t.begin(), t.end(), rng);
    auto groups = [&](std::vector<int>& items, int width) {
        std::vector<std::vector<int>> out;
        std::size_t at = 0;
        while(at < items.size()) {
            std::vector<int> tile(std::size_t(width), -1);
            const int take = std::min<int>(int(items.size() - at), 1 + int(rng() % unsigned(width)));
            std::vector<int> slots(static_cast<std::size_t>(width));
            std::iota(slots.begin(), slots.end(), 0);
            std::shuffle(slots.begin(), slots.end(), rng);
            for(int k = 0; k < take; ++k) tile[std::size_t(slots[std::size_t(k)])] = items[at++];
            out.push_back(tile);
        }
        return out;
    };
    const auto rg = groups(r, 16), tg = groups(t, 8);
    rows_of.clear();
    tokens_of.clear();
    for(const auto& a : rg)
        for(const auto& b : tg) {
            rows_of.insert(rows_of.end(), a.begin(), a.end());
            tokens_of.insert(tokens_of.end(), b.begin(), b.end());
        }
}

void check_projections() {
    std::mt19937 rng(41);
    const Format formats[] = {{"MXFP8", 1, 32, 32, fb::ScaleKind::mx},
                              {"128 x 128 blocks", 128, 128, 128, fb::ScaleKind::amax},
                              {"row scales", 1, 0, 0, fb::ScaleKind::amax}};
    double worst = 0.0;
    int compared = 0;
    for(const auto& f : formats)
        for(int cols : {32, 96, 512, 2560}) {
            const int scale_k = f.scale_k ? f.scale_k : cols, group = f.group ? f.group : cols;
            if(cols % scale_k || cols % group) continue;
            const int rows = 200, tokens = 13;
            std::vector<std::uint8_t> codes(std::size_t(rows) * cols);
            for(auto& c : codes) c = random_code(rng);
            for(int k = 0; k < 64; ++k) codes[rng() % codes.size()] = std::uint8_t(0x7e | (rng() & 0x80u));   // +-448
            for(int k = 0; k < 64; ++k) codes[rng() % codes.size()] = std::uint8_t(rng() % 8u);                // subnormals, 0
            const int srows = (rows + f.scale_rows - 1) / f.scale_rows, sk = cols / scale_k;
            std::vector<float> wscales(std::size_t(srows) * sk);
            std::uniform_int_distribution<int> e8(120, 132);
            std::uniform_real_distribution<float> fs(0.001f, 0.01f);
            for(auto& s : wscales) s = f.kind == fb::ScaleKind::mx ? fb::e8m0(std::uint8_t(e8(rng))) : fs(rng);
            std::normal_distribution<float> normal(0.0f, 1.0f);
            std::vector<float> xf(std::size_t(tokens) * cols);
            for(auto& v : xf) v = normal(rng);
            for(int k = 0; k < 8; ++k) xf[rng() % xf.size()] = 40.0f * normal(rng);   // outliers
            // Quantize on the device; the host twin must agree.
            const int groups = tokens * (cols / group);
            float* d_xf = dev<float>(xf.size());
            std::uint8_t* d_x = dev<std::uint8_t>(xf.size());
            float* d_xs = dev<float>(std::size_t(groups));
            put(d_xf, xf);
            quantize_kernel<<<(groups + 63) / 64, 64>>>(d_xf, groups, group, f.kind, d_x, d_xs);
            ok(cudaGetLastError(), "quantize");
            const auto x = get(d_x, xf.size());
            const auto xs = get(d_xs, std::size_t(groups));
            for(int i = 0; i < groups; ++i) {
                std::vector<std::uint8_t> hc(static_cast<std::size_t>(group));
                const float hs = fb::quantize_group(&xf[std::size_t(i) * group], std::size_t(group), f.kind, hc.data());
                require(bits(hs) == bits(xs[std::size_t(i)]), std::string(f.name) + ": quantizer scale differs");
                require(std::equal(hc.begin(), hc.end(), x.begin() + std::ptrdiff_t(i) * group),
                        std::string(f.name) + ": quantizer codes differ");
            }
            std::uint8_t* d_codes = dev<std::uint8_t>(codes.size());
            float* d_ws = dev<float>(wscales.size());
            put(d_codes, codes);
            put(d_ws, wscales);
            const fb::Matrix m{d_codes, d_ws, rows, cols, f.scale_rows, scale_k, 0.75f};
            const fb::Acts a{d_x, d_xs, cols, cols / group, group};
            const std::size_t n = std::size_t(tokens) * rows;
            float* d_ref = dev<float>(n);
            float* d_other = dev<float>(n);
            const dim3 grid(unsigned((rows + 15) / 16), unsigned((tokens + 7) / 8));
            reference_kernel<<<grid, 32>>>(m, a, tokens, d_ref);
            ok(cudaDeviceSynchronize(), "reference");
            const auto ref = get(d_ref, n);
            reference_kernel<<<grid, 32>>>(m, a, tokens, d_other);
            require(get(d_other, n) == ref, std::string(f.name) + ": a repeat differs");
            for(int round = 0; round < 3; ++round) {
                std::vector<int> rows_of, tokens_of;
                scrambled_tiles(rows, tokens, rng, rows_of, tokens_of);
                int* d_rows_of = dev<int>(rows_of.size());
                int* d_tokens_of = dev<int>(tokens_of.size());
                put(d_rows_of, rows_of);
                put(d_tokens_of, tokens_of);
                const unsigned tiles = unsigned(rows_of.size() / 16);
                for(int kernel = 0; kernel < 2; ++kernel) {
                    ok(cudaMemset(d_other, 0xff, n * sizeof(float)), "clear");
                    if(kernel == 0) folded_kernel<<<tiles, 32>>>(m, a, d_rows_of, d_tokens_of, d_other);
                    else split_kernel<<<tiles, 128>>>(m, a, d_rows_of, d_tokens_of, d_other);
                    ok(cudaDeviceSynchronize(), kernel ? "split" : "folded");
                    const auto got = get(d_other, n);
                    for(std::size_t i = 0; i < n; ++i)
                        require(bits(got[i]) == bits(ref[i]),
                                std::string(f.name) + (kernel ? ": split" : ": folded") + " kernel differs at token " +
                                    std::to_string(i / std::size_t(rows)) + ", row " + std::to_string(i % std::size_t(rows)) +
                                    " (cols " + std::to_string(cols) + ")");
                    ++compared;
                }
                cudaFree(d_rows_of);
                cudaFree(d_tokens_of);
            }
            for(int t = 0; t < tokens; ++t)
                for(int r = 0; r < rows; ++r) {
                    double magnitude = 0.0;
                    const double exact = 0.75 * fb::project_fp64(&codes[std::size_t(r) * cols],
                                                                 &wscales[std::size_t(r / f.scale_rows) * sk], std::size_t(scale_k),
                                                                 &x[std::size_t(t) * cols], &xs[std::size_t(t) * (cols / group)],
                                                                 std::size_t(group), std::size_t(cols), &magnitude);
                    const double err = std::fabs(double(ref[std::size_t(t) * rows + r]) - exact);
                    if(magnitude > 0.0) worst = std::max(worst, err / (0.75 * magnitude));
                    require(err <= std::ldexp(0.75 * magnitude, -9) + 1e-30, std::string(f.name) + ": far from FP64");
                }
            for(void* p : {(void*)d_xf, (void*)d_x, (void*)d_xs, (void*)d_codes, (void*)d_ws, (void*)d_ref, (void*)d_other}) cudaFree(p);
        }
    std::cout << "projections: " << compared << " scrambled runs bitwise equal to the reference, repeats equal; worst "
              << "error against FP64 " << worst << " of the products' magnitude\n";
}

void check_quantizer_edges() {
    // Ties and saturation at MX scale 1 (amax 256): 1.0625 -> 1, 1.1875 -> 1.25, 464 -> 448, 2^-10 -> 0, -0 -> -0.
    std::vector<float> v(32, 0.0f);
    const float edges[] = {256.0f, 1.0625f, 1.1875f, -1.0625f, 0.0009765625f, -0.0f, 3.0f, -255.9f};
    for(std::size_t k = 0; k < std::size(edges); ++k) v[k] = edges[k];
    std::vector<float> zeros(32, 0.0f);
    std::vector<float> all = v;
    all.insert(all.end(), zeros.begin(), zeros.end());
    for(auto kind : {fb::ScaleKind::mx, fb::ScaleKind::amax}) {
        float* d_v = dev<float>(all.size());
        std::uint8_t* d_c = dev<std::uint8_t>(all.size());
        float* d_s = dev<float>(2);
        put(d_v, all);
        quantize_kernel<<<1, 32>>>(d_v, 2, 32, kind, d_c, d_s);
        ok(cudaDeviceSynchronize(), "quantize edges");
        const auto c = get(d_c, all.size());
        const auto s = get(d_s, 2);
        for(int i = 0; i < 2; ++i) {
            std::vector<std::uint8_t> hc(32);
            const float hs = fb::quantize_group(&all[std::size_t(i) * 32], 32, kind, hc.data());
            require(bits(hs) == bits(s[std::size_t(i)]) &&
                        std::equal(hc.begin(), hc.end(), c.begin() + std::ptrdiff_t(i) * 32),
                    "quantizer edges differ from the host twin");
        }
        if(kind == fb::ScaleKind::mx) {
            require(s[0] == 1.0f && c[1] == 0x38 && c[2] == 0x3a && c[3] == 0xb8 && c[4] == 0x00 && c[5] == 0x80,
                    "MX ties: round to nearest even");
            require(s[1] == 1.0f && c[32] == 0, "an all-zero group gets scale 1 and zero codes");
        }
        cudaFree(d_v);
        cudaFree(d_c);
        cudaFree(d_s);
    }
    std::cout << "quantizer: ties, saturation, signed zero and zero groups match the host twin\n";
}
}  // namespace

int main() {
    try {
        int device = 0, major = 0, minor = 0, count = 0;
        if(cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { std::cout << "no CUDA device; skipped\n"; return 0; }
        ok(cudaGetDevice(&device), "device");
        ok(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "major");
        ok(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), "minor");
        if(major * 10 + minor < 89) { std::cout << "e4m3 mma.sync needs sm_89 or newer; skipped\n"; return 0; }
        check_quantizer_edges();
        check_projections();
        std::cout << "fp8 block tests passed\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
