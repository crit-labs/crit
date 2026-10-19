// Device side of the host-work mailbox fixture: a kernel that writes a job and posts it the way an engine's own
// kernel would, and a consumer that reads the host's answer after the join.
#include "mailbox.h"
#include "mailbox_device.cuh"

#include <cstdint>

namespace hm = crit::techniques::host_mailbox;

namespace {
// payload: [0] input the device writes (a per-replay counter), [1] the host's result.
__global__ void produce_kernel(hm::SlotView v, std::uint32_t* counter, int skip) {
    if(threadIdx.x != 0) return;
    if(skip) {
        hm::skip_job(v.device_done, v.device_status);
        return;
    }
    const std::uint32_t value = ++*counter;
    reinterpret_cast<volatile std::uint32_t*>(v.device_payload)[0] = value;
    __threadfence_system();
    hm::post_job(v.device_ready, v.device_done, v.device_status);
}
// sums[0] += result of served jobs; sums[1 + status] counts every status seen.
__global__ void consume_kernel(hm::SlotView v, unsigned long long* sums) {
    if(threadIdx.x != 0) return;
    const std::uint32_t status = hm::read_status(v.device_status);
    if(status == hm::served) sums[0] += hm::read_result(reinterpret_cast<const std::uint32_t*>(v.device_payload) + 1);
    sums[1 + (status < 5 ? status : 0)] += 1;
}
}  // namespace

cudaError_t fixture_produce(const hm::SlotView& v, std::uint32_t* counter, bool skip, cudaStream_t s) {
    produce_kernel<<<1, 32, 0, s>>>(v, counter, skip ? 1 : 0);
    return cudaGetLastError();
}
cudaError_t fixture_consume(const hm::SlotView& v, unsigned long long* sums, cudaStream_t s) {
    consume_kernel<<<1, 32, 0, s>>>(v, sums);
    return cudaGetLastError();
}
