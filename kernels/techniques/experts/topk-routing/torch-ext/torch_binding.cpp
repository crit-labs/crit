#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("route(Tensor logits, int top, Tensor? bias, bool softmax, bool before, bool renormalize, float epsilon, "
          "float floor, float scaling, Tensor! ids, Tensor! weights) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("route", torch::kCUDA, &route);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
