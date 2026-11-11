// A test-only binding of decode_gemv.cu for another target of the device (tests/test_decode_gemv.py builds it for the
// plain target, sm_120 on an sm_120a device, where E2M1 decodes by the bit pattern), to compare its bits with the build's.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "decode_gemv.h"

namespace nr = crit::kernels::decode_gemv;

void gemv(torch::Tensor x, torch::Tensor codes, torch::Tensor scales, torch::Tensor inverse, torch::Tensor out) {
  TORCH_CHECK(nr::enqueue_gemv(x.data_ptr(), static_cast<int>(x.stride(0)), x.scalar_type() == at::kBFloat16,
                               codes.data_ptr<std::uint8_t>(), reinterpret_cast<const std::uint8_t*>(scales.data_ptr()),
                               false, inverse.data_ptr<float>(), out.data_ptr(), static_cast<int>(out.stride(0)),
                               out.scalar_type() == at::kFloat, static_cast<int>(x.size(0)),
                               static_cast<int>(codes.size(0)), static_cast<int>(x.size(1)),
                               at::cuda::getCurrentCUDAStream()) == cudaSuccess,
              "gemv");
}

int64_t hardware_decode() { return nr::hardware_decode(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv", &gemv);
  m.def("hardware_decode", &hardware_decode);
}
