#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("layout(int hidden, int routes, int tokens) -> int[]", &layout);
  ops.def("post(int ready, int done, int status, int payload, int hidden, int capacity, int token_capacity, "
          "Tensor ids, Tensor slots, Tensor x, int key_base, Tensor! cpu_routes, Tensor! cpu_count, "
          "Tensor! counters) -> ()");
  ops.def("scatter(int ready, int done, int status_word, int payload, int hidden, int capacity, int token_capacity, "
          "Tensor! y, Tensor cpu_routes, Tensor cpu_count, Tensor! status, int routes) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("post", torch::kCUDA, &post);
  ops.impl("scatter", torch::kCUDA, &scatter);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
