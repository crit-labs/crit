// Device side of a handoff (handoff.h), for kernels that post host work and read its results: a plugin's own kernel
// includes this header (crit_plugins.host.INCLUDE is its folder) and is given the slot's device addresses
// (Handoff.words(slot) in Python). Post from ONE thread after every payload write of this job is system-fenced
// (__threadfence_system() in each writing thread, then __syncthreads(), or writes from an earlier kernel in stream
// order).
#pragma once
#include <cstddef>
#include <cstdint>

namespace crit::host {

// Status words. The device writes skipped; the host writes the others before done.
enum Status : std::uint32_t { pending = 0, served = 1, skipped = 2, failed = 3, released = 4 };

// One slot's words and payload as the device addresses them (mapped host memory).
struct DeviceSlot {
    std::uint32_t* ready = nullptr;
    std::uint32_t* done = nullptr;
    std::uint32_t* status = nullptr;
    std::byte* payload = nullptr;
};

#if defined(__CUDACC__)

// Reset done before raising ready: the host writes done only after it has seen this ready.
__device__ inline void post_job(const DeviceSlot& slot) {
    *reinterpret_cast<volatile std::uint32_t*>(slot.status) = 0u;
    *reinterpret_cast<volatile std::uint32_t*>(slot.done) = 0u;
    __threadfence_system();
    *reinterpret_cast<volatile std::uint32_t*>(slot.ready) = 1u;
    __threadfence_system();
}

// No host work for this slot this time: answer the join directly.
__device__ inline void skip_job(const DeviceSlot& slot) {
    *reinterpret_cast<volatile std::uint32_t*>(slot.status) = std::uint32_t(skipped);
    __threadfence_system();
    *reinterpret_cast<volatile std::uint32_t*>(slot.done) = 1u;
    __threadfence_system();
}

// After the join: read host-written results with volatile (uncached) loads, as mapped memory the host changed
// between replays must not be served from a stale line.
template <class T> __device__ inline T read_result(const T* p) { return *reinterpret_cast<const volatile T*>(p); }
__device__ inline std::uint32_t read_status(const DeviceSlot& slot) { return read_result(slot.status); }

#endif  // __CUDACC__

}  // namespace crit::host
