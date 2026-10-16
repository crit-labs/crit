#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

namespace crit::tests::device_round {
#ifdef __CUDACC__
__host__ __device__
#endif
inline std::uint32_t mix(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
inline std::uint32_t step_state_host(std::uint32_t s, std::int32_t token) { return s * 1000003u + std::uint32_t(token) + 7u; }
// Device buffers of the synthetic hash target. Plain pointers; owners live in the test.
struct HashTarget {
    std::uint32_t* state = nullptr;       // committed state
    std::uint32_t* start = nullptr;       // round-start copy
    std::uint32_t* checkpoints = nullptr; // [K+1] state after each verify column
    std::int32_t* history = nullptr;      // [context] materialized inputs
    std::uint32_t vocabulary = 0, corrupt_period = 3, corrupt_seed = 0;
};
cudaError_t enqueue_draft(const HashTarget&, const std::int32_t* control, std::int32_t* drafts, std::uint32_t k, cudaStream_t);
cudaError_t enqueue_verify(const HashTarget&, const std::int32_t* control, const std::int32_t* drafts,
                           std::int32_t* predictions, std::uint32_t k, cudaStream_t);
cudaError_t enqueue_commit_state(const HashTarget&, const std::int32_t* control, cudaStream_t);
cudaError_t enqueue_prefill(const HashTarget&, const std::int32_t* tokens, std::uint32_t count, cudaStream_t);
}
