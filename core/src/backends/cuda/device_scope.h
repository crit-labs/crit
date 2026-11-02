#pragma once
// The GPU backend's calls bind the device they act on and give the calling thread its current device back when
// they end, so an engine's binding survives Core's calls: a kernel launch needs its stream's device current, and a
// device set's calls bind each member in turn. Private to the backend's sources.
#include "crit/gpu_runtime.h"
#include <new>
#include <stdexcept>
#include <string>

namespace crit::core::detail {
class DeviceScope {
public:
    // Binds `device`; std::runtime_error when it cannot.
    explicit DeviceScope(int device) : previous_(current()) { check(set(device)); }
    // Binds `device` for cleanup paths that must not throw; status() says whether it worked.
    DeviceScope(int device, std::nothrow_t) noexcept : previous_(current()) { status_ = set(device); }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    ~DeviceScope() {
        if(previous_ >= 0 && current_ >= 0 && current_ != previous_) static_cast<void>(cudaSetDevice(previous_));
    }
    // Moves the binding to another device inside the scope (a device set's members in turn).
    void bind(int device) { check(set(device)); }
    cudaError_t status() const noexcept { return status_; }
private:
    int previous_, current_ = -1;
    cudaError_t status_ = cudaSuccess;
    static int current() noexcept {
        int device = -1;
        if(cudaGetDevice(&device) != cudaSuccess) { static_cast<void>(cudaGetLastError()); device = -1; }
        return device;
    }
    cudaError_t set(int device) noexcept {
        if(device == (current_ >= 0 ? current_ : previous_)) { current_ = device; return cudaSuccess; }
        const auto status = cudaSetDevice(device);
        if(status == cudaSuccess) current_ = device;
        return status;
    }
    static void check(cudaError_t status) {
        if(status != cudaSuccess)
            throw std::runtime_error(std::string("cudaSetDevice: ") + cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
    }
};
}
