#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("rotate(Tensor rows, Tensor? pre, Tensor? post, Tensor! out) -> ()");
  ops.def("splits(Tensor trellis, int codebook, int rows) -> int");
  ops.def("tiles(Tensor trellis, int codebook, Tensor xr, Tensor! yr) -> ()");
  ops.def("decode(Tensor trellis, int codebook, Tensor! q) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("rotate", torch::kCUDA, &rotate);
  ops.impl("splits", torch::kCUDA, &splits);
  ops.impl("tiles", torch::kCUDA, &tiles);
  ops.impl("decode", torch::kCUDA, &decode);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
