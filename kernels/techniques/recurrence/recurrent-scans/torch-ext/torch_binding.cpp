#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("causal_conv1d(Tensor! out, Tensor! state, Tensor x, Tensor weight, Tensor? bias, bool silu) -> ()");
  ops.def("gated_delta_prefill(Tensor! out, Tensor! state, Tensor q, Tensor k, Tensor v, Tensor log_decay, "
          "Tensor beta, float scale) -> ()");
  ops.def("gated_delta_prefill_tc(Tensor! out, Tensor! state, Tensor q, Tensor k, Tensor v, Tensor log_decay, "
          "Tensor beta, float scale) -> ()");
  ops.def("gated_delta_scan(Tensor! out, Tensor! state, Tensor q, Tensor k, Tensor v, Tensor log_decay, "
          "Tensor beta, float scale, Tensor? checkpoints) -> ()");
  ops.def("ssd_prefill(Tensor! out, Tensor! state, Tensor x, Tensor dt, Tensor B, Tensor C, Tensor A, Tensor? D) -> ()");
  ops.def("ssd_scan(Tensor! out, Tensor! state, Tensor x, Tensor dt, Tensor B, Tensor C, Tensor A, Tensor? D, "
          "Tensor? checkpoints) -> ()");
  ops.def("pool_delta(Tensor! out, Tensor! states, Tensor! stash, Tensor! control, Tensor(a!)? scratch, Tensor q, Tensor k, "
          "Tensor v, Tensor log_decay, Tensor beta, Tensor? meta, Tensor? starts, Tensor? slots, Tensor? table, int rows, "
          "int kind, int layer, int capacity, float epsilon, float scale, bool normalize_keys, bool normalize_queries, bool pipelined) -> ()");
  ops.def("pool_delta_replay(Tensor! states, Tensor! stash, Tensor! control, Tensor? scratch, Tensor? accepted, "
          "Tensor? meta, Tensor? slots, Tensor? table, int rows, int columns, int capacity, int key_heads, int mode) -> ()");
  ops.def("pool_conv(Tensor! out, Tensor! window, Tensor! stash, Tensor! pending, Tensor(a!)? scratch, Tensor x, "
          "Tensor weight, Tensor? bias, Tensor? meta, Tensor? starts, Tensor? slots, Tensor? table, int rows, int kind, "
          "int layer, int layers, int capacity, bool silu) -> ()");
  ops.def("conv_segments(Tensor! out, Tensor window_in, Tensor! window_out, Tensor x, Tensor weight, Tensor? bias, "
          "bool silu) -> ()");
  ops.def("pool_conv_prompt(Tensor! out, Tensor! window, Tensor stash, Tensor! pending, Tensor x, Tensor weight, "
          "Tensor? bias, Tensor meta, int layer, int capacity, bool silu) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("causal_conv1d", torch::kCUDA, &causal_conv1d);
  ops.impl("gated_delta_prefill", torch::kCUDA, &gated_delta_prefill);
  ops.impl("gated_delta_prefill_tc", torch::kCUDA, &gated_delta_prefill_tc);
  ops.impl("gated_delta_scan", torch::kCUDA, &gated_delta_scan);
  ops.impl("ssd_prefill", torch::kCUDA, &ssd_prefill);
  ops.impl("ssd_scan", torch::kCUDA, &ssd_scan);
  ops.impl("pool_delta", torch::kCUDA, &pool_delta);
  ops.impl("pool_delta_replay", torch::kCUDA, &pool_delta_replay);
  ops.impl("pool_conv", torch::kCUDA, &pool_conv);
  ops.impl("conv_segments", torch::kCUDA, &conv_segments);
  ops.impl("pool_conv_prompt", torch::kCUDA, &pool_conv_prompt);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
