#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gemv(Tensor x, Tensor codes, Tensor scales, Tensor inverse, bool blocked, Tensor! out) -> ()");
  ops.def("gemv_fp8(Tensor x, Tensor codes, Tensor scales, Tensor! out) -> ()");
  ops.def("gemv_bf16(Tensor x, Tensor w, Tensor! out) -> ()");
  ops.def("gemv_int4(Tensor x, Tensor codes, Tensor scales, Tensor zeros, int group, Tensor! out) -> ()");
  ops.def("repack_int4(int kind, Tensor qweight, Tensor qzeros, Tensor scales, Tensor? perm, int zero_offset, int group, "
          "Tensor! codes, Tensor! zeros, Tensor! scales_out) -> ()");
  ops.def("hardware_decode(Tensor device_of) -> int");
  ops.def("rows_int_block(Tensor table, Tensor jobs, int max_rows, Tensor values, Tensor scales, int block, "
          "Tensor! out) -> ()");
  ops.def("rows_fp8_block(Tensor table, Tensor jobs, int max_rows, Tensor codes, Tensor scales, int group, "
          "Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("gemv", torch::kCUDA, &gemv);
  ops.impl("gemv_fp8", torch::kCUDA, &gemv_fp8);
  ops.impl("gemv_bf16", torch::kCUDA, &gemv_bf16);
  ops.impl("gemv_int4", torch::kCUDA, &gemv_int4);
  ops.impl("repack_int4", torch::kCUDA, &repack_int4);
  ops.impl("hardware_decode", torch::kCUDA, &hardware_decode);
  ops.impl("rows_int_block", torch::kCUDA, &rows_int_block);
  ops.impl("rows_fp8_block", torch::kCUDA, &rows_fp8_block);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
