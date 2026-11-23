#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("plan(Tensor ids, Tensor? slots, int experts, Tensor! item_slot, Tensor! item_routes, Tensor! item_count) -> ()");
  ops.def("items(Tensor order, Tensor counts, Tensor starts, Tensor item_end, Tensor? slots, int experts, "
          "Tensor! item_slot, Tensor! item_routes, Tensor! item_count) -> ()");
  ops.def("gate_up(Tensor x, int top, Tensor store, int[] layout, bool a4, Tensor item_slot, Tensor item_routes, "
          "Tensor item_count, Tensor! h, int tokens=0) -> ()");
  ops.def("down(Tensor h, Tensor store, int[] layout, bool a4, Tensor item_slot, Tensor item_routes, Tensor item_count, "
          "Tensor! y, int tokens=0) -> ()");
  ops.def("combine(Tensor y, Tensor ids, Tensor weights, Tensor? slots, int experts, bool skip_negative, Tensor! out) -> ()");
  ops.def("quantize_e2m1x2(Tensor x, float g, Tensor! values, Tensor! scales) -> ()");
  ops.def("a4_inputs(Tensor x, int top, Tensor store, int[] layout, Tensor item_slot, Tensor item_routes, "
          "Tensor item_count, Tensor! out) -> ()");
  ops.def("declared_inputs(Tensor x, int top, Tensor store, int[] layout, Tensor item_slot, Tensor item_routes, "
          "Tensor item_count, Tensor! values, Tensor! scales) -> ()");
  ops.def("declared_rows(Tensor values, Tensor scales, Tensor store, int[] layout, bool gate_up, Tensor item_slot, "
          "Tensor item_routes, Tensor item_count, Tensor! out) -> ()");
  ops.def("declared_intermediate(Tensor gate_up, Tensor store, int[] layout, Tensor item_slot, Tensor item_routes, "
          "Tensor item_count, Tensor! values, Tensor! scales) -> ()");
  ops.def("fp4_available() -> bool");
  ops.def("fp4_inputs(Tensor x, int top, Tensor store, int[] layout, Tensor item_slot, Tensor item_routes, "
          "Tensor item_count, Tensor! codes, Tensor! scales) -> ()");
  ops.def("fp4_gate_up(Tensor codes, Tensor scales, Tensor store, int[] layout, Tensor item_slot, Tensor item_routes, "
          "Tensor item_count, Tensor! h_codes, Tensor! h_scales, int tokens=0, int tile=0) -> ()");
  ops.def("fp4_down(Tensor h_codes, Tensor h_scales, Tensor store, int[] layout, Tensor item_slot, Tensor item_routes, "
          "Tensor item_count, Tensor! y, int tokens=0, int tile=0) -> ()");
  ops.def("fp4_rows(Tensor rows, int div, Tensor item_slot, Tensor item_routes, Tensor item_count, int terms, "
          "Tensor! codes, Tensor! scales, Tensor! rcp) -> ()");
  ops.def("fp4_gate_up_rows(Tensor codes, Tensor scales, Tensor rcp, int terms, Tensor store, int[] layout, "
          "Tensor item_slot, Tensor item_routes, Tensor item_count, Tensor! h, int tokens=0, int tile=0) -> ()");
  ops.def("fp4_down_rows(Tensor h_codes, Tensor h_scales, Tensor rcp, int terms, Tensor store, int[] layout, "
          "Tensor item_slot, Tensor item_routes, Tensor item_count, Tensor! y, int tokens=0, int tile=0) -> ()");
  ops.def("fp8_available() -> bool");
  ops.def("fp8_inputs(Tensor rows, int div, Tensor item_slot, Tensor item_routes, Tensor item_count, Tensor! codes, "
          "Tensor! scales) -> ()");
  ops.def("fp8_gate_up(Tensor codes, Tensor scales, Tensor store, int[] layout, Tensor item_slot, Tensor item_routes, "
          "Tensor item_count, Tensor! h, int tokens=0) -> ()");
  ops.def("fp8_down(Tensor h_codes, Tensor h_scales, Tensor store, int[] layout, Tensor item_slot, Tensor item_routes, "
          "Tensor item_count, Tensor! y, int tokens=0) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("fp8_available", &fp8_available);
  ops.impl("fp8_inputs", torch::kCUDA, &fp8_inputs);
  ops.impl("fp8_gate_up", torch::kCUDA, &fp8_gate_up);
  ops.impl("fp8_down", torch::kCUDA, &fp8_down);
  ops.impl("fp4_available", &fp4_available);
  ops.impl("fp4_inputs", torch::kCUDA, &fp4_inputs);
  ops.impl("fp4_gate_up", torch::kCUDA, &fp4_gate_up);
  ops.impl("fp4_down", torch::kCUDA, &fp4_down);
  ops.impl("fp4_rows", torch::kCUDA, &fp4_rows);
  ops.impl("fp4_gate_up_rows", torch::kCUDA, &fp4_gate_up_rows);
  ops.impl("fp4_down_rows", torch::kCUDA, &fp4_down_rows);
  ops.impl("a4_inputs", torch::kCUDA, &a4_inputs);
  ops.impl("declared_inputs", torch::kCUDA, &declared_inputs);
  ops.impl("declared_rows", torch::kCUDA, &declared_rows);
  ops.impl("declared_intermediate", torch::kCUDA, &declared_intermediate);
  ops.impl("plan", torch::kCUDA, &plan);
  ops.impl("items", torch::kCUDA, &items);
  ops.impl("gate_up", torch::kCUDA, &gate_up);
  ops.impl("down", torch::kCUDA, &down);
  ops.impl("combine", torch::kCUDA, &combine);
  ops.impl("quantize_e2m1x2", torch::kCUDA, &quantize_e2m1x2);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
