// The core tests' harness: device buffers, checks, and the architecture flags the device code was built with (a test
// binary built as PTX for compute_80 runs on a newer GPU with __CUDA_ARCH__ 800, so one GPU checks the older paths).
#pragma once
#include "crit_kcore/arch.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace check {

inline int failures = 0;

inline void ok(cudaError_t e, const char* what) {
    if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
template <class T>
T* dev(std::size_t n) {
    void* p = nullptr;
    ok(cudaMalloc(&p, (n ? n : 1) * sizeof(T)), "cudaMalloc");
    ok(cudaMemset(p, 0, (n ? n : 1) * sizeof(T)), "cudaMemset");
    return static_cast<T*>(p);
}
template <class T>
T* dev(const std::vector<T>& v) {
    T* p = dev<T>(v.size());
    ok(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
template <class T>
std::vector<T> get(const T* p, std::size_t n) {
    std::vector<T> v(n);
    ok(cudaDeviceSynchronize(), "kernel");
    ok(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

inline uint32_t bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
inline float from_bits(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
// Equal bits, or both NaN.
inline bool same(float a, float b) { return bits(a) == bits(b) || (a != a && b != b); }

inline void report(const char* name, long long bad, const std::string& detail = "") {
    if(bad) {
        ++failures;
        std::printf("FAIL %s: %lld mismatches %s\n", name, bad, detail.c_str());
    } else {
        std::printf("PASS %s\n", name);
    }
}

// The device code's architecture flags (CRIT_KCORE_HAS_*), as the binary was built.
struct Flags {
    int arch, cp_async, fp8_mma, tma, pdl, e2m1_cvt, block_scaled, wgmma, tcgen05;
};
__global__ void flags_kernel(Flags* f) {
    *f = Flags{CRIT_KCORE_ARCH, CRIT_KCORE_HAS_CP_ASYNC, CRIT_KCORE_HAS_FP8_MMA, CRIT_KCORE_HAS_TMA, CRIT_KCORE_HAS_PDL,
               CRIT_KCORE_HAS_E2M1_CVT, CRIT_KCORE_HAS_BLOCK_SCALED_MMA, CRIT_KCORE_HAS_WGMMA, CRIT_KCORE_HAS_TCGEN05};
}
inline Flags flags() {
    Flags* d = dev<Flags>(1);
    flags_kernel<<<1, 1>>>(d);
    Flags f = get(d, 1)[0];
    cudaFree(d);
    std::printf("device code: arch %d cp.async %d fp8 mma %d tma %d pdl %d e2m1 cvt %d block-scaled mma %d\n", f.arch,
                f.cp_async, f.fp8_mma, f.tma, f.pdl, f.e2m1_cvt, f.block_scaled);
    return f;
}

inline int finish() {
    if(failures) std::printf("%d checks failed\n", failures);
    else std::printf("all checks passed\n");
    return failures ? 1 : 0;
}

}  // namespace check
