// A typed device buffer for Inference's CUDA tests: allocated for its count (at least one element), filled from and
// read back into host vectors, freed with it. ok() turns a CUDA error into the test's failure.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace crit::inference_tests {

inline void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }

template<class T> struct Device {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Device(std::size_t count) : n(count) { ok(cudaMalloc(&p, sizeof(T) * std::max<std::size_t>(count, 1)), "malloc"); }
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

// Untyped device bytes (at least one), for fixtures that hand kernels raw pointers: upload and download are
// synchronous copies, so they wait for the work enqueued before them on the default stream.
struct DeviceBytes {
    void* p = nullptr;
    std::size_t n = 0;
    explicit DeviceBytes(std::size_t bytes) : n(bytes) { ok(cudaMalloc(&p, std::max<std::size_t>(bytes, 1)), "malloc"); }
    ~DeviceBytes() { cudaFree(p); }
    DeviceBytes(DeviceBytes&& other) noexcept : p(std::exchange(other.p, nullptr)), n(other.n) {}
    DeviceBytes(const DeviceBytes&) = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
    void* data() const noexcept { return p; }
    void upload(std::span<const std::byte> bytes) { ok(cudaMemcpy(p, bytes.data(), bytes.size(), cudaMemcpyHostToDevice), "upload"); }
    void download(std::span<std::byte> bytes) const { ok(cudaMemcpy(bytes.data(), p, bytes.size(), cudaMemcpyDeviceToHost), "download"); }
};

}  // namespace crit::inference_tests
