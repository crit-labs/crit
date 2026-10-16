#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstddef>

// Independent unfused baseline: projections run separately, then this ordinary
// pointwise kernel forms the shared activation. No fused implementation helpers.
__global__ void activation_reference(const float* gate, const float* up, std::uint16_t* out, std::size_t count) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float g = gate[i];
  const float activated = g >= 0 ? g / (1.f + expf(-g)) : g * expf(g) / (1.f + expf(g));
  out[i] = __bfloat16_as_ushort(__float2bfloat16_rn(activated * up[i]));
}
cudaError_t unfused_activation(const float* gate, const float* up, std::uint16_t* out, std::size_t count, cudaStream_t stream) {
  activation_reference<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(gate, up, out, count);
  return cudaGetLastError();
}
