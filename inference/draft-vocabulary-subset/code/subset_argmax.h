#pragma once
// Gather-argmax over an indexed row subset of an output head (or the full head).
// Reference kernel, not for performance: FP32 rows, fixed sequential dot order.
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace crit::techniques::draft_subset {
struct GatherArgmax {
    const float* weights = nullptr;     // head rows, row r at weights + r * row_stride
    std::size_t row_stride = 0;         // elements, >= hidden
    std::uint32_t hidden = 0;
    const std::int32_t* rows = nullptr; // ascending subset token IDs; null selects the full head
    std::uint32_t count = 0;            // subset size S, or vocabulary V for the full head
    const float* states = nullptr;      // [batch][hidden]
    std::uint32_t batch = 0;
    std::int32_t* tokens = nullptr;     // [batch] winning token ID (mapped through rows)
    float* logits = nullptr;            // optional [batch][count]
};
// One block per batch row. Ties keep the lowest subset index, which is the
// lowest token ID because rows ascend: the full head's first-maximum rule.
cudaError_t enqueue_gather_argmax(const GatherArgmax&, cudaStream_t);
}
