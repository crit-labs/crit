#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("plan(Tensor! key_slot, Tensor! slot_key, Tensor! slot_ref, Tensor! slot_pin, Tensor! state, "
          "Tensor! counters, Tensor! seen, Tensor! route_counts, Tensor quota, int admit_quota, int layer, "
          "int experts, Tensor ids, int top, Tensor rows, Tensor! row_counters, Tensor ring, Tensor! out, "
          "Tensor! jobs, Tensor! job_count, Tensor! slot_last, Tensor victims, Tensor! recent, Tensor! pending, "
          "Tensor! range, bool link) -> ()");
  ops.def("fill(Tensor source, Tensor! store, Tensor jobs, Tensor job_count, int blocks, int vector, "
          "Tensor job_first) -> ()");
  ops.def("fill_program(Tensor source, Tensor! store, Tensor jobs, Tensor job_count, Tensor program, int shared, "
          "int blocks) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("plan", torch::kCUDA, &plan);
  ops.impl("fill", torch::kCUDA, &fill);
  ops.impl("fill_program", torch::kCUDA, &fill_program);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
