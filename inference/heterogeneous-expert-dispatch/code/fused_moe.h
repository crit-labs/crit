#pragma once
#include "fp8_cuda.h"

namespace crit::moe {
// Entry = router projection plus BF16-rounded SiLU(gate)*up, in one grid.
// All matrices have the same K; gate/up have the same intermediate rows.
void entry_cuda(fp8::DeviceRows router, fp8::DeviceRows gate, fp8::DeviceRows up,
                std::span<const std::uint16_t> input, std::size_t tokens,
                std::span<float> scores, std::span<std::uint16_t> activation, cudaStream_t);
// How router logits become selected experts and weights; routing.py's Routing and its
// host reference `route` define the same fields. The defaults are top-k on raw logits
// (descending, lower expert ID on ties) and softmax over the selected logits.
enum class Scoring : std::uint8_t { softmax, sigmoid };
struct Routing {
  Scoring scoring = Scoring::softmax;
  bool score_before_selection = false;          // score all experts, then select on scores
  std::span<const float> correction_bias = {};  // device, one per expert; changes selection only
  std::uint32_t groups = 1, selected_groups = 1, group_score_top = 1;  // group = sum of its best N
  bool renormalize = false;                     // divide by max(sum + epsilon, floor)
  float renormalize_epsilon = 0, renormalize_floor = 0, scaling = 1;
};
// Routing and shared-down are independent after entry. One grid performs both:
// deterministic routing under `routing` and BF16 down[N,I] x activation[T,I] -> FP32 shared[T,N].
// This returns unweighted shared output; model-specific shared gates remain with
// the engine. It neither evaluates routed experts nor changes their identities.
void route_down_cuda(std::span<const float> scores, std::size_t experts, std::size_t top, const Routing& routing,
                     std::span<const std::uint16_t> activation, std::size_t intermediate,
                     std::span<const std::uint16_t> down, std::size_t outputs, std::size_t tokens,
                     std::span<std::int32_t> ids, std::span<float> weights, std::span<float> shared,
                     cudaStream_t);
// Borrowed device views; caller validates finite payloads and retains all owners
// through stream completion on success or error. Outputs cannot overlap inputs
// or each other. 1..8 tokens, 1..1024 experts, 1..32 selected experts and 1..32 selected
// groups; groups divide the experts. Ties keep the lower expert or group index.
}
