#pragma once
// The device commit of an accepted speculative prefix of a gated-delta layer: selecting the state after the accepted
// columns from per-column checkpoints, and recording the accepted count for a deferred commit, both from a count in
// device memory (a round's control[n_emitted]), so a captured round commits without a host read. Mechanism, not
// arithmetic: the scans that write the checkpoints or the deferred stash come from a kernel provider (crit-labs'
// recurrent-scans).
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace crit::techniques::recurrent::cuda {
struct DeviceGeometry {
    std::uint32_t heads = 1, keys = 0, values = 0; // values <= 1024
    float epsilon = 1e-6f, query_scale = 1.0f;
    bool normalize_keys = true, normalize_queries = true;
};
// State layout per head: matrix[h * keys * values + k * values + v].
inline std::size_t state_elements(const DeviceGeometry& g) { return std::size_t(g.heads) * g.keys * g.values; }

// Per-column checkpoints commit: state = accepted ? checkpoints[accepted - 1] : start, with accepted read from device
// memory and clamped to 0 .. columns; checkpoints [columns][state_elements], the state after each verified column.
cudaError_t enqueue_select_accepted(const DeviceGeometry&, const float* start, const float* checkpoints,
                                    const std::int32_t* accepted, std::uint32_t columns, float* state, cudaStream_t);

// Deferred commit: *pending = clamp(*accepted, 0, columns). Idempotent; rerun it only before the next deferred verify,
// which applies the recorded prefix and consumes the count.
cudaError_t enqueue_record_accepted(const std::int32_t* accepted, std::uint32_t columns, std::int32_t* pending, cudaStream_t);
// The same for `count` counts in each of `groups` groups `stride` ints apart, in one launch:
// pending[g * stride + i] = clamp(*accepted, 0, columns). A deferred commit that keeps a count per block (a column scan
// and a convolution) records all of a round's layers with one call.
cudaError_t enqueue_record_accepted(const std::int32_t* accepted, std::uint32_t columns, std::int32_t* pending,
                                    std::uint32_t count, std::uint32_t groups, std::size_t stride, cudaStream_t);
}
