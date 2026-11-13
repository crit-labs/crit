#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "mm_rows(Tensor x, Tensor w, Tensor? scales, Tensor? row_scale, int weights, int acts, float input_scale, "
      "bool blocked, int groups, int warps, int splits, Tensor(c!)? scratch, Tensor(a!)? partial, Tensor(b!)? counters, "
      "Tensor! out, Tensor? zeros=None, int group=0, int glu=0) -> ()");
  ops.def("quantize_fp8_rows(Tensor x, Tensor! q, Tensor! scale) -> ()");
  ops.def("quantize_nvfp4(Tensor x, float input_scale, Tensor! codes, Tensor! scales) -> ()");
  ops.def("step_values(int weights) -> int");
  ops.def("fp4_mma(Tensor like) -> bool");
  ops.def("fp4_scratch_bytes(int rows, int k) -> int");
  ops.def(
      "mm_prompt(Tensor x, Tensor w, Tensor? scales, Tensor? row_scale, int weights, int acts, float input_scale, "
      "bool blocked, int splits, Tensor(c!)? scratch, Tensor! out, Tensor? zeros=None, int group=0, int glu=0) -> ()");
  ops.def("decode_ctas(int weights, int acts) -> int");
#if defined(CUDA_KERNEL)
  ops.impl("mm_rows", torch::kCUDA, &mm_rows);
  ops.impl("quantize_fp8_rows", torch::kCUDA, &quantize_fp8_rows);
  ops.impl("quantize_nvfp4", torch::kCUDA, &quantize_nvfp4);
  ops.impl("step_values", &step_values);
  ops.impl("fp4_mma", torch::kCUDA, &fp4_mma);
  ops.impl("fp4_scratch_bytes", &fp4_scratch_bytes);
  ops.impl("mm_prompt", torch::kCUDA, &mm_prompt);
  ops.impl("decode_ctas", &decode_ctas);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
