#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

// The ops the workbench's kernel.py calls by format: run_bf16, run_fp8.
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("run_bf16(Tensor x, Tensor w) -> Tensor");
  ops.def("run_fp8(Tensor x, Tensor x_scale, Tensor w, Tensor w_scale) -> Tensor");
#if defined(CUDA_KERNEL)
  ops.impl("run_bf16", torch::kCUDA, &run_bf16);
  ops.impl("run_fp8", torch::kCUDA, &run_fp8);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
