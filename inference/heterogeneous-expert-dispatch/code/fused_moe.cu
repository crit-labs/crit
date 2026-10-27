#include "fused_moe.h"
#include "fp8_tile.cuh"
#include <array>
#include <cmath>
#include <string>
#include <math_constants.h>

namespace crit::moe {
namespace {
template<class T> fp8::DeviceRange range(std::span<T> values) { return {values.data(), values.size_bytes()}; }
template<class T> void aligned(std::span<T> values) {
  if (!values.data() || reinterpret_cast<std::uintptr_t>(values.data()) % 4)
    throw std::invalid_argument("Four-byte-aligned MoE device view required");
}
void launched() {
  const auto error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(std::string("MoE launch: ") + cudaGetErrorString(error));
}
template<int Tokens> __global__ void entry(fp8::DeviceRows router, fp8::DeviceRows gate, fp8::DeviceRows up,
    const std::uint16_t* input, float* scores, std::uint16_t* activation, unsigned router_blocks) {
  // Both branches take a uniform path within their entire CTA.
  __shared__ fp8::detail::Tile<Tokens, true> pair;
  __shared__ fp8::detail::Tile<Tokens, false> single;
  float a[Tokens] = {}, b[Tokens] = {};
  const auto lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (blockIdx.x < router_blocks) {
    const auto row0 = static_cast<std::size_t>(blockIdx.x) * 8;
    fp8::detail::calculate<Tokens, false>(router, {}, input, row0, single, a, b);
    if (!lane && row0 + warp < router.rows)
      for (int t = 0; t < Tokens; ++t) scores[t * router.rows + row0 + warp] = a[t];
  } else {
    const auto row0 = static_cast<std::size_t>(blockIdx.x - router_blocks) * 8;
    fp8::detail::calculate<Tokens, true>(gate, up, input, row0, pair, a, b);
    if (!lane && row0 + warp < gate.rows)
      for (int t = 0; t < Tokens; ++t)
        activation[t * gate.rows + row0 + warp] = __bfloat16_as_ushort(__float2bfloat16_rn(fp8::detail::silu(a[t]) * b[t]));
  }
}
// Routing as kernel arguments; bias is null without a correction bias.
struct Rule {
  const float* bias; int groups, selected_groups, group_top; float epsilon, floor, scaling;
  bool sigmoid, before, renormalize;
};
// The block's 256 threads reduce through `scratch` with `op`; every thread receives the result.
template<class Op> __device__ float reduce(float value, float* scratch, Op op) {
  const auto tid = threadIdx.x;
  scratch[tid] = value; __syncthreads();
  for (unsigned stride = 128; stride; stride /= 2) {
    if (tid < stride) scratch[tid] = op(scratch[tid], scratch[tid + stride]);
    __syncthreads();
  }
  const auto result = scratch[0]; __syncthreads();
  return result;
}
// Every thread calls this: the `top` largest values[0..count) in descending order,
// lower index on ties, written to chosen_id/chosen_value.
__device__ void select_top(const float* values, int count, int top, int* chosen_id, float* chosen_value,
                           float* best_value, int* best_id) {
  const auto tid = threadIdx.x;
  for (int rank = 0; rank < top; ++rank) {
    float best = -CUDART_INF_F; int index = 0x7fffffff;
    for (int i = static_cast<int>(tid); i < count; i += 256) {
      bool used = false;
      for (int previous = 0; previous < rank; ++previous) used |= chosen_id[previous] == i;
      const auto value = values[i];
      if (!used && (value > best || (value == best && i < index))) { best = value; index = i; }
    }
    best_value[tid] = best; best_id[tid] = index; __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
      if (tid < stride && (best_value[tid + stride] > best_value[tid] ||
          (best_value[tid + stride] == best_value[tid] && best_id[tid + stride] < best_id[tid]))) {
        best_value[tid] = best_value[tid + stride]; best_id[tid] = best_id[tid + stride];
      }
      __syncthreads();
    }
    if (!tid) { chosen_value[rank] = best_value[0]; chosen_id[rank] = best_id[0]; }
    __syncthreads();
  }
}
__global__ void route_down(const float* scores, int experts, int top, Rule rule, const std::uint16_t* activation,
    std::size_t intermediate, const std::uint16_t* down, std::size_t outputs, int tokens,
    std::int32_t* ids, float* weights, float* shared) {
  const auto tid = threadIdx.x;
  if (blockIdx.x < static_cast<unsigned>(tokens)) {
    const auto token = blockIdx.x;
    const auto* logits = scores + static_cast<std::size_t>(token) * experts;
    __shared__ float choice[1024], group_value[1024], best_value[256], selected_value[32], group_chosen[32];
    __shared__ int best_id[256], selected_id[32], group_id[32];
    // Softmax before selection normalizes over every expert.
    float peak = 0, total = 1;
    if (rule.before && !rule.sigmoid) {
      float local = -CUDART_INF_F, sum = 0;
      for (int expert = static_cast<int>(tid); expert < experts; expert += 256) local = fmaxf(local, logits[expert]);
      peak = reduce(local, best_value, [](float a, float b) { return fmaxf(a, b); });
      for (int expert = static_cast<int>(tid); expert < experts; expert += 256) sum += expf(logits[expert] - peak);
      total = reduce(sum, best_value, [](float a, float b) { return a + b; });
    }
    const auto score = [&](float logit) { return rule.sigmoid ? 1.f / (1.f + expf(-logit)) : expf(logit - peak) / total; };
    // Selection value: the raw logit or its score, plus the correction bias when present. NaN ranks as -inf, so a
    // row of NaN (a padding row, or a NaN that reached the router) still selects in-range experts.
    for (int expert = static_cast<int>(tid); expert < experts; expert += 256) {
      const auto value = rule.before ? score(logits[expert]) : logits[expert];
      const auto biased = rule.bias ? value + rule.bias[expert] : value;
      choice[expert] = biased == biased ? biased : -CUDART_INF_F;
    }
    __syncthreads();
    if (rule.selected_groups < rule.groups) {
      // Each group ranks by the sum of its group_top best selection values.
      const int size = experts / rule.groups;
      for (int group = static_cast<int>(tid); group < rule.groups; group += 256) {
        float sum = 0, last = CUDART_INF_F; int last_id = -1;
        for (int n = 0; n < rule.group_top; ++n) {
          float best = -CUDART_INF_F; int index = 0x7fffffff;
          for (int i = group * size; i < (group + 1) * size; ++i) {
            const auto value = choice[i];
            if ((value < last || (value == last && i > last_id)) && (value > best || (value == best && i < index))) {
              best = value; index = i;
            }
          }
          sum += best; last = best; last_id = index;
        }
        group_value[group] = sum;
      }
      __syncthreads();
      select_top(group_value, rule.groups, rule.selected_groups, group_id, group_chosen, best_value, best_id);
      for (int expert = static_cast<int>(tid); expert < experts; expert += 256) {
        bool kept = false;
        for (int g = 0; g < rule.selected_groups; ++g) kept |= group_id[g] == expert / size;
        if (!kept) choice[expert] = -CUDART_INF_F;
      }
      __syncthreads();
    }
    select_top(choice, experts, top, selected_id, selected_value, best_value, best_id);
    if (!tid) {
      float weight[32], denominator = 1;
      if (rule.before || rule.sigmoid) {
        for (int r = 0; r < top; ++r) weight[r] = score(logits[selected_id[r]]);
      } else {  // softmax over the selected logits
        float selected_peak = logits[selected_id[0]], sum = 0;
        for (int r = 0; r < top; ++r) selected_peak = fmaxf(selected_peak, logits[selected_id[r]]);
        for (int r = 0; r < top; ++r) sum += expf(logits[selected_id[r]] - selected_peak);
        for (int r = 0; r < top; ++r) weight[r] = expf(logits[selected_id[r]] - selected_peak) / sum;
      }
      if (rule.renormalize) {
        float sum = 0;
        for (int r = 0; r < top; ++r) sum += weight[r];
        denominator = fmaxf(sum + rule.epsilon, rule.floor);
      }
      for (int r = 0; r < top; ++r) {
        ids[token * top + r] = selected_id[r];
        weights[token * top + r] = (rule.renormalize ? weight[r] / denominator : weight[r]) * rule.scaling;
      }
    }
  } else {
    const auto row = static_cast<std::size_t>(blockIdx.x - tokens) * 8 + tid / 32;
    const auto lane = tid % 32;
    for (int t = 0; t < tokens; ++t) {
      float sum = 0;
      if (row < outputs) for (std::size_t k = lane; k < intermediate; k += 32)
        sum = fmaf(fp8::detail::bfloat(down[row * intermediate + k]),
                   fp8::detail::bfloat(activation[t * intermediate + k]), sum);
      for (int delta = 16; delta; delta /= 2) sum += __shfl_down_sync(0xffffffffU, sum, delta);
      if (!lane && row < outputs) shared[t * outputs + row] = sum;
    }
  }
}
}
void entry_cuda(fp8::DeviceRows router, fp8::DeviceRows gate, fp8::DeviceRows up,
                std::span<const std::uint16_t> input, std::size_t tokens,
                std::span<float> scores, std::span<std::uint16_t> activation, cudaStream_t stream) {
  for (auto matrix : {router, gate, up}) fp8::validate(matrix);
  if (!tokens || tokens > 8 || router.width != gate.width || gate.width != up.width || gate.rows != up.rows ||
      input.size() != fp8::extent(tokens, router.width) || scores.size() != fp8::extent(tokens, router.rows) ||
      activation.size() != fp8::extent(tokens, gate.rows)) throw std::invalid_argument("MoE entry geometry differs");
  aligned(input); aligned(scores); aligned(activation);
  const std::array inputs{range(router.codes), range(router.scales), range(gate.codes), range(gate.scales), range(up.codes), range(up.scales), range(input)};
  const std::array outputs{range(scores), range(activation)};
  fp8::disjoint(outputs, inputs);
  const auto router_blocks = static_cast<unsigned>((router.rows + 7) / 8);
  const auto blocks = router_blocks + static_cast<unsigned>((gate.rows + 7) / 8);
  #define CRIT_MOE_CASE(T) case T: entry<T><<<blocks, 256, 0, stream>>>(router, gate, up, input.data(), scores.data(), activation.data(), router_blocks); break
  switch (tokens) { CRIT_MOE_CASE(1); CRIT_MOE_CASE(2); CRIT_MOE_CASE(3); CRIT_MOE_CASE(4);
                    CRIT_MOE_CASE(5); CRIT_MOE_CASE(6); CRIT_MOE_CASE(7); CRIT_MOE_CASE(8); }
  #undef CRIT_MOE_CASE
  launched();
}
void route_down_cuda(std::span<const float> scores, std::size_t experts, std::size_t top, const Routing& routing,
    std::span<const std::uint16_t> activation, std::size_t intermediate,
    std::span<const std::uint16_t> down, std::size_t outputs, std::size_t tokens,
    std::span<std::int32_t> ids, std::span<float> weights, std::span<float> shared, cudaStream_t stream) {
  using fp8::extent;
  if (!tokens || tokens > 8 || !experts || experts > 1024 || !top || top > std::min<std::size_t>(32, experts) ||
      !intermediate || intermediate > 0x7fffffffU || !outputs || outputs > 0x7fffffffU ||
      scores.size() != extent(tokens, experts) || activation.size() != extent(tokens, intermediate) ||
      down.size() != extent(outputs, intermediate) || ids.size() != extent(tokens, top) ||
      weights.size() != extent(tokens, top) || shared.size() != extent(tokens, outputs))
    throw std::invalid_argument("MoE route/down geometry differs");
  const std::size_t groups = routing.groups, size = groups ? experts / groups : 0;
  const auto term = [](float value) { return std::isfinite(value) && value >= 0; };
  if ((routing.scoring != Scoring::softmax && routing.scoring != Scoring::sigmoid) || !groups || experts % groups ||
      !routing.selected_groups || routing.selected_groups > std::min<std::size_t>(32, groups) ||
      !routing.group_score_top || routing.group_score_top > size || top > routing.selected_groups * size ||
      (!routing.correction_bias.empty() && routing.correction_bias.size() != experts) ||
      (routing.renormalize && routing.scoring == Scoring::softmax && !routing.score_before_selection) ||
      !term(routing.renormalize_epsilon) || !term(routing.renormalize_floor) || !term(routing.scaling) || !routing.scaling)
    throw std::invalid_argument("MoE routing differs from its geometry");
  aligned(scores); aligned(activation); aligned(down); aligned(ids); aligned(weights); aligned(shared);
  const bool bias = !routing.correction_bias.empty();
  if (bias) aligned(routing.correction_bias);
  const std::array inputs{range(scores), range(activation), range(down), range(routing.correction_bias)};
  const std::array destinations{range(ids), range(weights), range(shared)};
  fp8::disjoint(destinations, std::span(inputs).first(bias ? 4 : 3));
  const Rule rule{routing.correction_bias.data(), static_cast<int>(routing.groups), static_cast<int>(routing.selected_groups),
                  static_cast<int>(routing.group_score_top), routing.renormalize_epsilon, routing.renormalize_floor,
                  routing.scaling, routing.scoring == Scoring::sigmoid, routing.score_before_selection, routing.renormalize};
  const auto blocks = static_cast<unsigned>(tokens + (outputs + 7) / 8);
  route_down<<<blocks, 256, 0, stream>>>(scores.data(), static_cast<int>(experts), static_cast<int>(top), rule,
      activation.data(), intermediate, down.data(), outputs, static_cast<int>(tokens), ids.data(), weights.data(), shared.data());
  launched();
}
}
