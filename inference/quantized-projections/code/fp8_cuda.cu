#include "fp8_tile.cuh"
#include <array>
#include <string>

namespace crit::fp8 {
void validate(DeviceRows rows) {
  if (!rows.rows || !rows.width || rows.rows > 0x7fffffffU || rows.width > 0x7fffffffU ||
      rows.codes.size() != extent(rows.rows, rows.width) || rows.scales.size() != rows.rows ||
      !rows.codes.data() || !rows.scales.data() || reinterpret_cast<std::uintptr_t>(rows.scales.data()) % 2 ||
      reinterpret_cast<std::uintptr_t>(rows.codes.data()) % 4)
    throw std::invalid_argument("Invalid aligned contiguous FP8 device matrix");
}
void disjoint(std::span<const DeviceRange> outputs, std::span<const DeviceRange> inputs) {
  const auto end = [](DeviceRange r) {
    const auto address = reinterpret_cast<std::uintptr_t>(r.pointer);
    if (!address || !r.bytes || r.bytes > std::numeric_limits<std::uintptr_t>::max() - address)
      throw std::invalid_argument("Invalid device byte range");
    return address + r.bytes;
  };
  const auto overlaps = [&](DeviceRange a, DeviceRange b) {
    return reinterpret_cast<std::uintptr_t>(a.pointer) < end(b) && reinterpret_cast<std::uintptr_t>(b.pointer) < end(a);
  };
  for (auto r : inputs) (void)end(r);
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    (void)end(outputs[i]);
    for (auto r : inputs) if (overlaps(outputs[i], r)) throw std::invalid_argument("Projection output aliases input");
    for (std::size_t j = 0; j < i; ++j) if (overlaps(outputs[i], outputs[j])) throw std::invalid_argument("Projection outputs overlap");
  }
}
template<int Tokens> __global__ void projection(DeviceRows weights, const std::uint16_t* input, float* output) {
  __shared__ detail::Tile<Tokens, false> tile;
  float sum[Tokens] = {}, unused[Tokens] = {};
  const auto row0 = static_cast<std::size_t>(blockIdx.x) * detail::rows_per_block;
  detail::calculate<Tokens, false>(weights, {}, input, row0, tile, sum, unused);
  const auto row = row0 + threadIdx.x / 32;
  if (threadIdx.x % 32 == 0 && row < weights.rows)
    for (int t = 0; t < Tokens; ++t) output[t * weights.rows + row] = sum[t];
}
void project_cuda(DeviceRows weights, std::span<const std::uint16_t> input, std::size_t tokens,
                  std::span<float> output, cudaStream_t stream) {
  validate(weights);
  if (!tokens || tokens > 8 || input.size() != extent(tokens, weights.width) || output.size() != extent(tokens, weights.rows) ||
      reinterpret_cast<std::uintptr_t>(input.data()) % 4 || reinterpret_cast<std::uintptr_t>(output.data()) % 4)
    throw std::invalid_argument("FP8 device projection shape/alignment differs");
  const std::array inputs{DeviceRange{weights.codes.data(), weights.codes.size_bytes()},
      DeviceRange{weights.scales.data(), weights.scales.size_bytes()}, DeviceRange{input.data(), input.size_bytes()}};
  const std::array outputs{DeviceRange{output.data(), output.size_bytes()}};
  disjoint(outputs, inputs);
  const auto blocks = static_cast<unsigned>((weights.rows + 7) / 8);
  #define CRIT_FP8_CASE(T) case T: projection<T><<<blocks, detail::threads, 0, stream>>>(weights, input.data(), output.data()); break
  switch (tokens) { CRIT_FP8_CASE(1); CRIT_FP8_CASE(2); CRIT_FP8_CASE(3); CRIT_FP8_CASE(4);
                    CRIT_FP8_CASE(5); CRIT_FP8_CASE(6); CRIT_FP8_CASE(7); CRIT_FP8_CASE(8); }
  #undef CRIT_FP8_CASE
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(std::string("FP8 launch: ") + cudaGetErrorString(status));
}
}
