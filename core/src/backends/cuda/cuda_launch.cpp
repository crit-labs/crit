#include "crit/cuda_launch.h"

#include <array>
#include <atomic>

namespace crit::core {
namespace {
// Per device: 0 unknown, 1 unsupported, 2 supported. Devices past the table are queried every time.
std::array<std::atomic<int>, 64> known{};
}

bool programmatic_launch_supported(int device) noexcept {
    if(device >= 0 && device < int(known.size())) {
        const int cached = known[std::size_t(device)].load(std::memory_order_relaxed);
        if(cached) return cached == 2;
    }
    int major = 0;
    const bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess;
    if(!ok) {
        static_cast<void>(cudaGetLastError());   // a failed query is not the caller's launch error
        return false;
    }
    const bool supported = major >= 9;
    if(device >= 0 && device < int(known.size())) known[std::size_t(device)].store(supported ? 2 : 1, std::memory_order_relaxed);
    return supported;
}

bool programmatic_launch_supported(cudaStream_t stream) noexcept {
    int device = 0;
    if(cudaStreamGetDevice(stream, &device) != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        if(cudaGetDevice(&device) != cudaSuccess) {
            static_cast<void>(cudaGetLastError());
            return false;
        }
    }
    return programmatic_launch_supported(device);
}

}  // namespace crit::core
