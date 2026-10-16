#include "fused_moe.h"
#include "fp8_tile.cuh"
#include <array>
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
__global__ void route_down(const float* scores, int experts, int top, const std::uint16_t* activation,
    std::size_t intermediate, const std::uint16_t* down, std::size_t outputs, int tokens,
    std::int32_t* ids, float* weights, float* shared) {
  const auto tid = threadIdx.x;
  if (blockIdx.x < static_cast<unsigned>(tokens)) {
    const auto token = blockIdx.x;
    __shared__ float best_value[256], selected_value[32];
    __shared__ int best_id[256], selected_id[32];
    for (int rank = 0; rank < top; ++rank) {
      float best = -CUDART_INF_F; int index = 0x7fffffff;
      for (int expert = static_cast<int>(tid); expert < experts; expert += 256) {
        bool used = false;
        for (int previous = 0; previous < rank; ++previous) used |= selected_id[previous] == expert;
        const auto value = scores[static_cast<std::size_t>(token) * experts + expert];
        if (!used && (value > best || (value == best && expert < index))) { best = value; index = expert; }
      }
      best_value[tid] = best; best_id[tid] = index; __syncthreads();
      for (unsigned stride = 128; stride; stride /= 2) {
        if (tid < stride && (best_value[tid + stride] > best_value[tid] ||
            (best_value[tid + stride] == best_value[tid] && best_id[tid + stride] < best_id[tid]))) {
          best_value[tid] = best_value[tid + stride]; best_id[tid] = best_id[tid + stride];
        }
        __syncthreads();
      }
      if (!tid) { selected_value[rank] = best_value[0]; selected_id[rank] = best_id[0]; }
      __syncthreads();
    }
    if (!tid) {
      float denominator = 0;
      for (int r = 0; r < top; ++r) denominator += expf(selected_value[r] - selected_value[0]);
      for (int r = 0; r < top; ++r) {
        ids[token * top + r] = selected_id[r];
        weights[token * top + r] = expf(selected_value[r] - selected_value[0]) / denominator;
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
void route_down_cuda(std::span<const float> scores, std::size_t experts, std::size_t top,
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
  aligned(scores); aligned(activation); aligned(down); aligned(ids); aligned(weights); aligned(shared);
  const std::array inputs{range(scores), range(activation), range(down)};
  const std::array destinations{range(ids), range(weights), range(shared)};
  fp8::disjoint(destinations, inputs);
  const auto blocks = static_cast<unsigned>(tokens + (outputs + 7) / 8);
  route_down<<<blocks, 256, 0, stream>>>(scores.data(), static_cast<int>(experts), static_cast<int>(top),
      activation.data(), intermediate, down.data(), outputs, static_cast<int>(tokens), ids.data(), weights.data(), shared.data());
  launched();
}
}
