#pragma once
// Per-row log-softmax, target NLL and KL divergence from stored reference
// log-probabilities, on the device. Reference kernel, not for performance: FP32
// rows, one block per row, fixed-order tree reductions (deterministic).
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace crit::eval::distribution {
struct RowCompare {
    const float* logits = nullptr;      // row r at logits + r * logits_stride
    std::size_t logits_stride = 0;      // elements, >= valid
    std::uint32_t rows = 0;
    std::uint32_t valid = 0;            // outputs scored: the first `valid` entries of each row
    const std::int32_t* targets = nullptr; // optional [rows]; a target outside [0, valid) yields NaN
    float* nll = nullptr;               // [rows] when targets is set
    float* store = nullptr;             // optional [rows][valid]: this row's log-probabilities (to become a reference)
    const float* reference = nullptr;   // optional [rows][valid]: reference log-probabilities
    float* kld = nullptr;               // [rows] when reference is set: KL(reference || this row), nats
};
// Validates pointers and geometry, then launches one block per row on `stream`.
cudaError_t enqueue_row_compare(const RowCompare&, cudaStream_t);
}
