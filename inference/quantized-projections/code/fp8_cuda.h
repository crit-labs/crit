#pragma once
#include "row_fp8.h"
#include <cuda_runtime_api.h>

namespace crit::fp8 {
// Borrowed contiguous device views. Validate words with RowWeights before
// upload; the caller retains immutable inputs and exclusive outputs until the
// supplied stream settles, including after any possibly partial launch failure.
struct DeviceRows {
  std::span<const std::uint8_t> codes;
  std::span<const std::uint16_t> scales;
  std::size_t rows, width;
};
struct DeviceRange { const void* pointer; std::size_t bytes; };
void validate(DeviceRows);
void disjoint(std::span<const DeviceRange> outputs, std::span<const DeviceRange> inputs);
void project_cuda(DeviceRows, std::span<const std::uint16_t> input, std::size_t tokens,
                  std::span<float> output, cudaStream_t stream);
// SM80+; 1..8 tokens, arbitrary positive N/K and tails. FP32 output and
// accumulation; E4M3FN weights, BF16 activations and per-row multipliers.
}
