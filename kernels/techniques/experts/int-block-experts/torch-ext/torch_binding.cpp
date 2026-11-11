#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("rows(Tensor table, Tensor jobs, int max_rows, Tensor values, Tensor scales, int block, Tensor! out) -> ()");
  ops.def("gemm(Tensor table, Tensor jobs, int max_rows, int max_count, Tensor values, Tensor scales, int block, "
          "Tensor! out) -> ()");
  ops.def("gemm_staged(Tensor table, Tensor jobs, int max_rows, int max_count, Tensor values, Tensor scales, "
          "int block, Tensor! out) -> ()");
  ops.def("quantize_e2m1x2(Tensor x, float global_scale, Tensor! values, Tensor! scales) -> ()");
  ops.def("gemm_native(Tensor table, Tensor jobs, int max_rows, int max_count, Tensor codes, Tensor scales, "
          "float global_scale, bool int4, Tensor! out, Tensor? col_scale, int tiled, Tensor? host_matrices) -> ()");
  ops.def("quantize_nvfp4(Tensor x, float global_scale, Tensor! codes, Tensor! scales, int tiled) -> ()");
  ops.def("quantize_int8(Tensor x, int block, Tensor! values, Tensor! scales) -> ()");
  ops.def("swiglu(Tensor gate_up, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("rows", torch::kCUDA, &rows);
  ops.impl("gemm", torch::kCUDA, &gemm);
  ops.impl("gemm_staged", torch::kCUDA, &gemm_staged);
  ops.impl("quantize_e2m1x2", torch::kCUDA, &quantize_e2m1x2);
  ops.impl("gemm_native", torch::kCUDA, &gemm_native);
  ops.impl("quantize_nvfp4", torch::kCUDA, &quantize_nvfp4);
  ops.impl("quantize_int8", torch::kCUDA, &quantize_int8);
  ops.impl("swiglu", torch::kCUDA, &swiglu);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
