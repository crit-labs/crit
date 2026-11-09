// The device commit (recurrent_cuda.h): a copy selected by a device-held count, and the count recorded for a deferred
// commit. No recurrence arithmetic: the scans that write checkpoints and stashes are a kernel provider's.
#include "recurrent_cuda.h"

namespace crit::techniques::recurrent::cuda {
namespace {
__global__ void select_accepted(DeviceGeometry g, const float* start, const float* checkpoints,
                                const std::int32_t* accepted, std::uint32_t columns, float* state) {
    const auto total = std::size_t(g.heads) * g.keys * g.values;
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= total) return;
    auto n = *accepted;
    if(n < 0) n = 0;
    if(std::uint32_t(n) > columns) n = std::int32_t(columns);
    state[i] = n ? checkpoints[std::size_t(n - 1) * total + i] : start[i];
}

__global__ void record_accepted(const std::int32_t* accepted, std::uint32_t columns, std::int32_t* pending, std::uint32_t count,
                                std::uint32_t groups, std::size_t stride) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= std::size_t(count) * groups) return;
    auto n = *accepted;
    if(n < 0) n = 0;
    if(std::uint32_t(n) > columns) n = std::int32_t(columns);
    pending[i / count * stride + i % count] = n;
}

bool valid(const DeviceGeometry& g) {
    return g.heads && g.keys && g.values && g.values <= 1024 && g.epsilon > 0 && g.query_scale > 0;
}
}

cudaError_t enqueue_select_accepted(const DeviceGeometry& g, const float* start, const float* checkpoints,
                                    const std::int32_t* accepted, std::uint32_t columns, float* state, cudaStream_t stream) {
    if(!valid(g) || !start || !accepted || !state || (columns && !checkpoints)) return cudaErrorInvalidValue;
    const auto total = state_elements(g);
    constexpr unsigned threads = 256;
    select_accepted<<<unsigned((total + threads - 1) / threads), threads, 0, stream>>>(g, start, checkpoints, accepted, columns, state);
    return cudaPeekAtLastError();
}

cudaError_t enqueue_record_accepted(const std::int32_t* accepted, std::uint32_t columns, std::int32_t* pending, cudaStream_t stream) {
    return enqueue_record_accepted(accepted, columns, pending, 1, 1, 1, stream);
}

cudaError_t enqueue_record_accepted(const std::int32_t* accepted, std::uint32_t columns, std::int32_t* pending,
                                    std::uint32_t count, std::uint32_t groups, std::size_t stride, cudaStream_t stream) {
    if(!accepted || !pending || !count || !groups || (groups > 1 && stride < count)) return cudaErrorInvalidValue;
    const auto total = std::size_t(count) * groups;
    constexpr unsigned threads = 256;
    record_accepted<<<unsigned((total + threads - 1) / threads), threads, 0, stream>>>(accepted, columns, pending, count, groups, stride);
    return cudaPeekAtLastError();
}
}
