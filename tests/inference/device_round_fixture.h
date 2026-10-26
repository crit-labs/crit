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
// Forced replay: predictions[c] = continuation[control[frontier] - base + c] for c = 0..k, -1 outside [0, length).
cudaError_t enqueue_forced(const std::int32_t* control, const std::int32_t* continuation, std::int32_t base, std::int32_t length,
                           std::int32_t* predictions, std::uint32_t k, cudaStream_t);
// Synthetic log-probability rows for columns 0..k from each column's prediction p (logprob_of and alternative_of).
cudaError_t enqueue_logprobs(const std::int32_t* predictions, std::uint32_t k, float* logprob, std::int32_t* top_ids,
                             float* top_logprobs, std::uint32_t top, std::uint32_t vocabulary, cudaStream_t);
#ifdef __CUDACC__
__host__ __device__
#endif
inline float logprob_of(std::int32_t p) { return -0.125f * float(p % 7) - 0.0625f; }
// A sampling-history stand-in with token_sampling::enqueue_commit's contract: append tokens[0, *count) to
// log[*log_count..] and advance *log_count by *count.
cudaError_t enqueue_take(const std::int32_t* count, const std::int32_t* tokens, std::int32_t* log, std::int32_t* log_count,
                         cudaStream_t);
}
