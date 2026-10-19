// The declared order on the device (host_nvfp4_device.cuh) gives the host's bits for the same rows: every
// supported host ISA and the CUDA kernel agree bitwise for 16 and 32 lanes and both products on expert-sized shapes. A device
// kernel that lets nvcc contract the product into the FMA is counted for contrast (not required to differ).
#include "host_nvfp4.h"
#include "host_nvfp4_device.cuh"
#include "nvfp4.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace hn = crit::techniques::host_nvfp4;

namespace {
void require(bool c, const std::string& m) {
    if(!c) throw std::runtime_error(m);
}
void ok(cudaError_t e, const char* what) {
    if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
std::uint32_t bits(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

// One row per group of L lanes.
template <int L>
__global__ void rows_kernel(const std::uint8_t* codes, const float* scales, const float* x, int rows, int cols,
                            float post, float* out) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / L;
    if(row >= rows) return;  // rows is a multiple of 32 / L so whole groups exit together
    const float v = hn::dot_row_halving<L>(codes + std::size_t(row) * cols / 2, scales + std::size_t(row) * cols / 16,
                                           x, cols, post);
    if(threadIdx.x % L == 0) out[row] = v;
}
template <int L>
__global__ void block_rows_kernel(const std::uint8_t* codes, const float* scales, const float* x, int rows, int cols,
                                  float post, float* out) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / L;
    if(row >= rows) return;
    const float v = hn::dot_row_block<L>(codes + std::size_t(row) * cols / 2, scales + std::size_t(row) * cols / 16,
                                         x, cols, post);
    if(threadIdx.x % L == 0) out[row] = v;
}

// The same loop written plainly: nvcc may contract w * s + acc and reduce in its own order.
__global__ void naive_kernel(const std::uint8_t* codes, const float* scales, const float* x, int rows, int cols,
                             float post, float* out) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= rows) return;
    float acc = 0.f;
    for(int k = 0; k < cols; ++k) {
        const std::uint8_t b = codes[std::size_t(row) * cols / 2 + k / 2];
        acc += hn::e2m1_device(k % 2 ? b >> 4 : b & 15u) * scales[std::size_t(row) * cols / 16 + k / 16] * x[k];
    }
    out[row] = acc * post;
}

template <class T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    ok(cudaMalloc(&d, v.size() * sizeof(T)), "malloc");
    ok(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

void check(std::size_t rows, std::size_t cols, int lanes, hn::Product product, std::mt19937& rng, int& naive_differs) {
    std::vector<std::uint8_t> codes(rows * cols / 2), sbytes(rows * cols / 16);
    std::vector<float> x(cols), scales(rows * cols / 16);
    std::uniform_int_distribution<int> byte(0, 255), scale(0x20, 0x5f);
    std::normal_distribution<float> act(0.f, 1.f);
    for(auto& b : codes) b = std::uint8_t(byte(rng));
    for(auto& s : sbytes) s = std::uint8_t(scale(rng) | (byte(rng) & 0x80));
    for(auto& v : x) v = act(rng);
    for(std::size_t i = 0; i < sbytes.size(); ++i) scales[i] = crit::nvfp4::e4m3(sbytes[i]);
    const float post = 0.03125f;
    const hn::MatrixView view{codes.data(), sbytes.data(), rows, cols};
    const hn::Order order{lanes, hn::Reduce::halving, product};
    std::vector<float> host(rows);
    hn::dot_rows(hn::Isa::scalar, order, view, hn::e4m3_scales(view), x.data(), cols, 1, 1u, 0, rows, post, host.data(),
                 rows);
    for(hn::Isa isa : {hn::Isa::avx2, hn::Isa::avx512}) {
        if(!hn::supported(isa)) continue;
        std::vector<float> simd(rows);
        hn::dot_rows(isa, order, view, hn::e4m3_scales(view), x.data(), cols, 1, 1u, 0, rows, post, simd.data(), rows);
        for(std::size_t r = 0; r < rows; ++r) require(bits(simd[r]) == bits(host[r]), "host ISAs disagree");
    }
    std::uint8_t* dc = upload(codes);
    float* ds = upload(scales);
    float* dx = upload(x);
    float* dout = nullptr;
    ok(cudaMalloc(&dout, rows * sizeof(float)), "malloc out");
    const int threads = 128, blocks = int((rows * std::size_t(lanes) + threads - 1) / threads);
    const bool block = product == hn::Product::block_dot;
    if(lanes == 32 && block) block_rows_kernel<32><<<blocks, threads>>>(dc, ds, dx, int(rows), int(cols), post, dout);
    else if(block) block_rows_kernel<16><<<blocks, threads>>>(dc, ds, dx, int(rows), int(cols), post, dout);
    else if(lanes == 32) rows_kernel<32><<<blocks, threads>>>(dc, ds, dx, int(rows), int(cols), post, dout);
    else rows_kernel<16><<<blocks, threads>>>(dc, ds, dx, int(rows), int(cols), post, dout);
    ok(cudaGetLastError(), "launch");
    std::vector<float> device(rows);
    ok(cudaMemcpy(device.data(), dout, rows * sizeof(float), cudaMemcpyDeviceToHost), "download");
    for(std::size_t r = 0; r < rows; ++r)
        require(bits(device[r]) == bits(host[r]), "device differs from the host at row " + std::to_string(r) + " cols " +
                                                      std::to_string(cols) + " lanes " + std::to_string(lanes));
    naive_kernel<<<int((rows + 127) / 128), 128>>>(dc, ds, dx, int(rows), int(cols), post, dout);
    ok(cudaMemcpy(device.data(), dout, rows * sizeof(float), cudaMemcpyDeviceToHost), "download naive");
    for(std::size_t r = 0; r < rows; ++r) naive_differs += bits(device[r]) != bits(host[r]);
    cudaFree(dc);
    cudaFree(ds);
    cudaFree(dx);
    cudaFree(dout);
}
}  // namespace

int main() {
    try {
        std::mt19937 rng(20261019);
        int naive = 0, rows = 0;
        for(auto [r, c] : {std::pair<std::size_t, std::size_t>{256, 1024}, {256, 2688}, {128, 4096}, {64, 512}})
            for(int lanes : {16, 32}) {
                check(r, c, lanes, hn::Product::scaled_elements, rng, naive);
                rows += int(r);
            }
        int block_rows = 0;
        for(auto [r, c] : {std::pair<std::size_t, std::size_t>{256, 2560}, {256, 640}, {64, 1040}})
            for(int lanes : {16, 32}) {
                check(r, c, lanes, hn::Product::block_dot, rng, naive);
                block_rows += int(r);
            }
        std::cout << "device: " << rows << " scaled_elements and " << block_rows
                  << " block_dot rows bitwise equal to the host oracle and every host ISA; a naive loop differs on "
                  << naive << " of them\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
