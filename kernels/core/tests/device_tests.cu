// device.cuh on the device: a value kept per device is made once on it, the multiprocessor count is the device's own,
// a launch above 48 KB of shared memory is refused before the kernel's opt-in and runs after it, a request for more
// than the device gives is refused and kept refused, and the resident CTAs are the device's occupancy.
#include "check.cuh"
#include "crit_kcore/device.cuh"

using namespace crit::kcore;

// Every thread writes its index into dynamic shared memory; thread 0 reports the last one's plus one (the block size).
__global__ void touch_kernel(int* out) {
    extern __shared__ int smem[];
    smem[threadIdx.x] = threadIdx.x;
    __syncthreads();
    if(threadIdx.x == 0) *out = smem[blockDim.x - 1] + 1;
}

int main() {
    check::flags();
    int device = 0, sms = 0;
    check::ok(cudaGetDevice(&device), "cudaGetDevice");
    check::ok(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "multiprocessors");

    PerDevice<int> value;
    int made = 0;
    const int first = value.get([&] { return ++made; });
    const int again = value.get([&] { return ++made; });
    check::report("PerDevice makes a device's value once", (first != 1) + (again != 1) + (made != 1));
    check::report("current_device, multiprocessors", (current_device() != device) + (multiprocessors() != sms));

    constexpr int kBytes = 64 * 1024, kThreads = 64;
    int* out = check::dev<int>(1);
    touch_kernel<<<1, kThreads, kBytes>>>(out);
    const bool refused = cudaGetLastError() != cudaSuccess;   // the default limit a block: 48 KB
    PerDevice<cudaError_t> optin;
    const cudaError_t set = shared_memory_optin(optin, touch_kernel, kBytes);
    const cudaError_t kept = shared_memory_optin(optin, touch_kernel, kBytes);
    touch_kernel<<<1, kThreads, kBytes>>>(out);
    const bool launched = cudaGetLastError() == cudaSuccess;
    const int got = launched ? check::get(out, 1)[0] : -1;
    check::report("shared_memory_optin: refused before, runs after",
                  !refused + (set != cudaSuccess) + (kept != cudaSuccess) + !launched + (got != kThreads));

    PerDevice<cudaError_t> too_much;
    const cudaError_t no = shared_memory_optin(too_much, touch_kernel, 1 << 30);
    const cudaError_t still_no = shared_memory_optin(too_much, touch_kernel, 1 << 30);
    check::report("shared_memory_optin refuses more than the device gives", (no == cudaSuccess) + (still_no != no));

    int per = 0;
    check::ok(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, touch_kernel, kThreads, kBytes), "occupancy");
    PerDevice<int> held;
    const int resident = resident_ctas(held, touch_kernel, kThreads, kBytes);
    check::report("resident_ctas", (resident != sms * (per > 0 ? per : 1)) + (resident < sms));
    cudaFree(out);
    return check::finish();
}
