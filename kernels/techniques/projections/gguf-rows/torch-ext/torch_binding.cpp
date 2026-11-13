#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("quantize(Tensor x, Tensor! xq, Tensor! xd, Tensor! xs, bool by_subblock=False) -> ()");
  ops.def("splits(Tensor blocks, int type, int k) -> int");
  ops.def("gemv(Tensor blocks, int type, int k, Tensor xq, Tensor xd, Tensor xs, Tensor? ids, Tensor! yp) -> ()");
  ops.def("gemm(Tensor blocks, int type, int k, Tensor xq, Tensor xd, Tensor xs, int parts, Tensor! y, int chunk_rows=0) -> ()");
  ops.def("sum_parts(Tensor yp, Tensor! y) -> ()");
  ops.def("dequantize(Tensor blocks, int type, int k, int expert, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("quantize", torch::kCUDA, &quantize);
  ops.impl("splits", torch::kCUDA, &splits);
  ops.impl("gemv", torch::kCUDA, &gemv);
  ops.impl("gemm", torch::kCUDA, &gemm);
  ops.impl("sum_parts", torch::kCUDA, &sum_parts);
  ops.impl("dequantize", torch::kCUDA, &dequantize);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
