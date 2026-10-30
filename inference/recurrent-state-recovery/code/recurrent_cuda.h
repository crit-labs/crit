#pragma once
// Generic CUDA reference for the gated-delta recurrence and the device commit of
// an accepted speculative prefix. FP32 storage and arithmetic; reference kernels,
// not for performance. The accepted count is read from device memory.
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace crit::techniques::recurrent::cuda {
struct DeviceGeometry {
    std::uint32_t heads = 1, keys = 0, values = 0; // values <= 1024 (one thread per value column)
    float epsilon = 1e-6f, query_scale = 1.0f;
    bool normalize_keys = true, normalize_queries = true;
};
// Projected, gated step inputs for `columns` positions; entry (t, h) is at
// query/key[(t * heads + h) * keys], value[(t * heads + h) * values],
// log_decay/beta[t * heads + h]. log_decay <= 0, beta in [0, 1].
struct DeviceSteps {
    const float* query = nullptr;
    const float* key = nullptr;
    const float* value = nullptr;
    const float* log_decay = nullptr;
    const float* beta = nullptr;
};
// State layout per head: matrix[h * keys * values + k * values + v].
inline std::size_t state_elements(const DeviceGeometry& g) { return std::size_t(g.heads) * g.keys * g.values; }

// Verify `columns` steps from `start` and write outputs[(t * heads + h) * values + v].
// Strategy A (checkpoints): non-null `checkpoints` receives the state after every
// column, [columns][state_elements]. Strategy B (recompute): null `checkpoints`
// and a `work` buffer of state_elements; only outputs survive.
cudaError_t enqueue_verify(const DeviceGeometry&, const float* start, DeviceSteps, std::uint32_t columns,
                           float* outputs, float* checkpoints, float* work, cudaStream_t);
// Strategy A commit: state = accepted ? checkpoints[accepted - 1] : start, with
// accepted read from device memory (for example a round's control[n_emitted]).
cudaError_t enqueue_select_accepted(const DeviceGeometry&, const float* start, const float* checkpoints,
                                    const std::int32_t* accepted, std::uint32_t columns, float* state, cudaStream_t);
// Strategy B commit: state = start advanced through the first `accepted` saved
// steps (accepted read from device memory, clamped to `columns`). state != start.
cudaError_t enqueue_recompute_accepted(const DeviceGeometry&, const float* start, DeviceSteps,
                                       const std::int32_t* accepted, std::uint32_t columns, float* state, cudaStream_t);

// Strategy C (deferred): one live state, no round-start copy. Verify first
// replays the previous round's pending accepted prefix from a compact stash into
// the live state, then verifies the new columns and overwrites the stash. Commit
// only records the accepted count, so it is a scalar write that can rerun with a
// smaller count after a host truncation. Stash layout for `capacity` columns:
// key[(t * heads + h) * keys] (normalized key), decay[t * heads + h] (exp(log_decay)),
// delta[(t * heads + h) * values] (beta * residual).
struct DeferredStash {
    float* key = nullptr;
    float* decay = nullptr;
    float* delta = nullptr;
    std::uint32_t capacity = 0; // stash columns; columns <= capacity
};
// Applies min(*pending, capacity) stashed steps to `state`, then verifies
// `columns` steps (columns may be zero: apply only) and zeroes *pending, so each
// recorded prefix is applied once. `work` holds one running copy.
cudaError_t enqueue_verify_deferred(const DeviceGeometry&, float* state, DeviceSteps, std::uint32_t columns,
                                    std::int32_t* pending, DeferredStash, float* outputs, float* work, cudaStream_t);
// Strategy C commit: *pending = clamp(*accepted, 0, columns). Idempotent; rerun it
// only before the next enqueue_verify_deferred, which consumes the count.
cudaError_t enqueue_record_accepted(const std::int32_t* accepted, std::uint32_t columns, std::int32_t* pending, cudaStream_t);
// The same for `count` counts in each of `groups` groups `stride` ints apart, in one
// launch: pending[g * stride + i] = clamp(*accepted, 0, columns). A deferred commit that
// keeps a count per block (an engine's deferred convolution and column scan)
// records all of a round's layers with one call.
cudaError_t enqueue_record_accepted(const std::int32_t* accepted, std::uint32_t columns, std::int32_t* pending,
                                    std::uint32_t count, std::uint32_t groups, std::size_t stride, cudaStream_t);
}
