#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("rows(Tensor table, Tensor jobs, int max_rows, Tensor codes, Tensor scales, int group, Tensor! out) -> ()");
  ops.def("gemm(Tensor table, Tensor jobs, int max_rows, int max_count, Tensor codes, Tensor scales, int group, "
          "Tensor! out) -> ()");
  ops.def("gemm_staged(Tensor table, Tensor jobs, int max_rows, int max_count, Tensor codes, Tensor scales, "
          "int group, int order, Tensor! out) -> ()");
  ops.def("gemm_native(Tensor table, Tensor jobs, int max_rows, int max_count, Tensor codes, Tensor scales, "
          "int group, bool mx, Tensor! out) -> ()");
  ops.def("gemm_native_channel(Tensor table, Tensor jobs, int max_rows, int max_count, Tensor codes, Tensor scales, "
          "Tensor! out) -> ()");
  ops.def("quantize(Tensor x, int group, int kind, int order, Tensor! codes, Tensor! scales) -> ()");
  ops.def("quantized_values(Tensor x, int group, int kind, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("rows", torch::kCUDA, &rows);
  ops.impl("gemm", torch::kCUDA, &gemm);
  ops.impl("gemm_staged", torch::kCUDA, &gemm_staged);
  ops.impl("gemm_native", torch::kCUDA, &gemm_native);
  ops.impl("gemm_native_channel", torch::kCUDA, &gemm_native_channel);
  ops.impl("quantize", torch::kCUDA, &quantize);
  ops.impl("quantized_values", torch::kCUDA, &quantized_values);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
