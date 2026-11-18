// crit's kernel core: what a launch keeps per device (kernels/core/README.md, "device"). Host code: no kernel.
//
// A launch runs on the thread's current device, and three things it reads are that device's own: a kernel's raised
// shared-memory limit (cudaFuncSetAttribute sets it on the function as it is loaded on the current device), the CTAs
// of a kernel the device holds at once, and its multiprocessors. Kept once a process (a function-local static, a
// global), they are the first device's on every other one: a second GPU then refuses every launch above 48 KB of
// shared memory and sizes its grids by the first GPU's count. PerDevice keeps one value a device; the helpers below
// are its three uses, each over a slot the caller holds as a static of the function that launches one kernel.
#pragma once
#include <cuda_runtime.h>

#include <mutex>

namespace crit::kcore {

constexpr int kMaxDevices = 64;

// The thread's current device; -1 where it cannot be read or is past kMaxDevices.
inline int current_device() {
    int device = -1;
    if(cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }
    return device >= 0 && device < kMaxDevices ? device : -1;
}

// One value a device. get(make) gives the current device's: make() on that device's first call (once, also among
// threads), the kept value after. Where the device cannot be named, make() runs on every call.
template <class T>
class PerDevice {
public:
    template <class Make>
    T get(Make&& make) {
        const int device = current_device();
        if(device < 0) return make();
        std::call_once(once_[device], [&] { value_[device] = make(); });
        return value_[device];
    }

private:
    std::once_flag once_[kMaxDevices];
    T value_[kMaxDevices] = {};
};

// The current device's multiprocessors (1 where they cannot be read).
inline int multiprocessors() {
    static PerDevice<int> counts;
    return counts.get([] {
        int device = 0, sms = 0;
        if(cudaGetDevice(&device) != cudaSuccess ||
           cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
            (void)cudaGetLastError();
        return sms > 0 ? sms : 1;
    });
}

// `kernel`'s dynamic shared memory a block raised to `bytes` on the current device (the opt-in above 48 KB), once a
// device. cudaSuccess, or the device's refusal (more than it gives a block), kept and returned on every call there.
template <class Kernel>
cudaError_t shared_memory_optin(PerDevice<cudaError_t>& slot, Kernel kernel, int bytes) {
    const cudaError_t set = slot.get(
        [&] { return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); });
    if(set != cudaSuccess) (void)cudaGetLastError();
    return set;
}

// The CTAs of `kernel` (at `threads` threads and `smem` bytes of dynamic shared memory) the current device holds at
// once: its multiprocessors times the blocks one of them holds, one each at least. Call it after shared_memory_optin
// where smem is above 48 KB (the occupancy of a launch the device would refuse is 0).
template <class Kernel>
int resident_ctas(PerDevice<int>& slot, Kernel kernel, int threads, int smem) {
    return slot.get([&] {
        int per = 0;
        if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kernel, threads, smem) != cudaSuccess)
            (void)cudaGetLastError();
        return multiprocessors() * (per > 0 ? per : 1);
    });
}

}  // namespace crit::kcore
