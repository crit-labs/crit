// Device side of the mailbox protocol (mailbox.h), for engines that post from their own kernels. Call from ONE
// thread after every payload write of this job is system-fenced (__threadfence_system() in each writing thread,
// then __syncthreads(), or writes from an earlier kernel in stream order).
#pragma once
#include <cstdint>

namespace crit::techniques::host_mailbox {

// Reset done before raising ready: the host writes done only after it has seen this ready.
__device__ inline void post_job(std::uint32_t* device_ready, std::uint32_t* device_done, std::uint32_t* device_status) {
    *reinterpret_cast<volatile std::uint32_t*>(device_status) = 0u;
    *reinterpret_cast<volatile std::uint32_t*>(device_done) = 0u;
    __threadfence_system();
    *reinterpret_cast<volatile std::uint32_t*>(device_ready) = 1u;
    __threadfence_system();
}

// No host work for this slot this time: answer the join directly.
__device__ inline void skip_job(std::uint32_t* device_done, std::uint32_t* device_status) {
    *reinterpret_cast<volatile std::uint32_t*>(device_status) = 2u;  // Status::skipped
    __threadfence_system();
    *reinterpret_cast<volatile std::uint32_t*>(device_done) = 1u;
    __threadfence_system();
}

// After the join: read host-written results with volatile (uncached) loads, as mapped memory the host changed
// between replays must not be served from a stale line.
template <class T> __device__ inline T read_result(const T* p) { return *reinterpret_cast<const volatile T*>(p); }
__device__ inline std::uint32_t read_status(const std::uint32_t* device_status) { return read_result(device_status); }

}  // namespace crit::techniques::host_mailbox
