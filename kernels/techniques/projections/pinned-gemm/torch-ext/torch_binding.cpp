#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("candidates(str[] fmt, int n, int k, int rep_rows, int workspace) -> int");
  ops.def("plan(str[] fmt, int n, int k, int rep_rows, int workspace, int index) -> int");
  ops.def("describe(int plan) -> int[]");
  ops.def(
      "run(int plan, Tensor a, Tensor b, Tensor! out, Tensor? scale_a, Tensor? scale_b, Tensor? alpha, Tensor? beta, "
      "Tensor! workspace) -> ()");
  ops.def("version() -> int");
#if defined(CUDA_KERNEL)
  ops.impl("candidates", &pinned_candidates);
  ops.impl("plan", &pinned_plan);
  ops.impl("describe", &pinned_describe);
  ops.impl("run", torch::kCUDA, &pinned_run);
  ops.impl("version", &pinned_version);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
