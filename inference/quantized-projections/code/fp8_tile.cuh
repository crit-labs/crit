#pragma once
#include "fp8_cuda.h"
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

namespace crit::fp8::detail {
constexpr int rows_per_block = 8, chunk = 128, threads = 256;
__device__ inline float bfloat(std::uint16_t word) { return __uint_as_float(static_cast<unsigned>(word) << 16); }
__device__ inline float code_value(std::uint8_t code) {
  const unsigned mag = code & 127U, e = mag >> 3, m = mag & 7U;
  const float v = e ? __uint_as_float(((e + 120U) << 23) | (m << 20)) : static_cast<float>(m) / 512.f;
  return code & 128U ? -v : v;
}
template<int Tokens, bool Pair> struct alignas(16) Tile {
  std::uint8_t codes[2][Pair ? 2 : 1][rows_per_block][chunk];
  std::uint16_t input[2][Tokens][chunk];
};
// One warp per output row. Code/activation chunks are double-buffered;
// aligned words use cp.async, tails and odd row strides use guarded scalar loads.
template<int Tokens, bool Pair>
__device__ void calculate(DeviceRows a, DeviceRows b, const std::uint16_t* input, std::size_t row0,
                           Tile<Tokens, Pair>& tile, float (&sum)[Tokens], float (&other)[Tokens]) {
  const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32;
  const auto groups = (a.width + chunk - 1) / chunk;
  const auto load = [&](std::size_t group, int stage) {
    for (int matrix = 0; matrix < (Pair ? 2 : 1); ++matrix) {
      const auto weights = matrix ? b : a;
      for (int word = tid; word < rows_per_block * chunk / 4; word += threads) {
        const int row = word / (chunk / 4), col = (word % (chunk / 4)) * 4;
        const auto k = group * chunk + static_cast<std::size_t>(col);
        auto* out = &tile.codes[stage][matrix][row][col];
        if (row0 + row < a.rows && k + 4 <= a.width && a.width % 4 == 0)
          __pipeline_memcpy_async(out, weights.codes.data() + (row0 + row) * a.width + k, 4);
        else for (int j = 0; j < 4; ++j)
          out[j] = row0 + row < a.rows && k + j < a.width ? weights.codes[(row0 + row) * a.width + k + j] : 0;
      }
    }
    for (int word = tid; word < Tokens * chunk / 2; word += threads) {
      const int token = word / (chunk / 2), col = (word % (chunk / 2)) * 2;
      const auto k = group * chunk + static_cast<std::size_t>(col);
      auto* out = &tile.input[stage][token][col];
      if (k + 2 <= a.width && a.width % 2 == 0)
        __pipeline_memcpy_async(out, input + token * a.width + k, 4);
      else for (int j = 0; j < 2; ++j) out[j] = k + j < a.width ? input[token * a.width + k + j] : 0;
    }
    __pipeline_commit();
  };
  load(0, 0);
  for (std::size_t group = 0; group < groups; ++group) {
    if (group + 1 < groups) { load(group + 1, static_cast<int>((group + 1) % 2)); __pipeline_wait_prior(1); }
    else __pipeline_wait_prior(0);
    __syncthreads();
    const int stage = static_cast<int>(group % 2);
    for (int k = lane; k < chunk; k += 32) {
      const float weight = code_value(tile.codes[stage][0][warp][k]);
      const float paired = Pair ? code_value(tile.codes[stage][Pair ? 1 : 0][warp][k]) : 0.f;
      #pragma unroll
      for (int t = 0; t < Tokens; ++t) {
        const float x = bfloat(tile.input[stage][t][k]);
        sum[t] = fmaf(weight, x, sum[t]);
        if constexpr (Pair) other[t] = fmaf(paired, x, other[t]);
      }
    }
    __syncthreads(); // The next issue may reuse the stage just consumed.
  }
  #pragma unroll
  for (int t = 0; t < Tokens; ++t) {
    for (int delta = 16; delta; delta /= 2) {
      sum[t] += __shfl_down_sync(0xffffffffU, sum[t], delta);
      if constexpr (Pair) other[t] += __shfl_down_sync(0xffffffffU, other[t], delta);
    }
    if (row0 + warp < a.rows) {
      sum[t] *= bfloat(a.scales[row0 + warp]);
      if constexpr (Pair) other[t] *= bfloat(b.scales[row0 + warp]);
    }
  }
}
__device__ inline float silu(float x) { return x >= 0 ? x / (1.f + expf(-x)) : x * expf(x) / (1.f + expf(x)); }
}
