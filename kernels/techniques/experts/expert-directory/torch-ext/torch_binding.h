#pragma once

#include <torch/torch.h>

// The directory's launches (directory_cuda/directory_ops.cu): an empty tensor stands for an absent buffer.
//
// plan: one routed layer's plan (or a prompt chunk's layer into ring slots) on the device directory: the slot table
// `out` [experts] and the copies `jobs` [experts + 1][2] with their count `job_count` [1].
void plan(const torch::Tensor& key_slot, const torch::Tensor& slot_key, const torch::Tensor& slot_ref,
          const torch::Tensor& slot_pin, const torch::Tensor& state, const torch::Tensor& counters,
          const torch::Tensor& seen, const torch::Tensor& route_counts, const torch::Tensor& quota, int64_t admit_quota,
          int64_t layer, int64_t experts, const torch::Tensor& ids, int64_t top, const torch::Tensor& rows,
          const torch::Tensor& row_counters, const torch::Tensor& ring, const torch::Tensor& out,
          const torch::Tensor& jobs, const torch::Tensor& job_count, const torch::Tensor& slot_last,
          const torch::Tensor& victims, const torch::Tensor& recent, const torch::Tensor& pending,
          const torch::Tensor& range, bool link);
// fill: a plan's copies (key -> slot) from the device-readable store `source` [keys][bundle] into `store`
// [slots][bundle], `vector` bytes a load, over `blocks` blocks (0: half the multiprocessors); job_first [1] or empty.
void fill(const torch::Tensor& source, const torch::Tensor& store, const torch::Tensor& jobs,
          const torch::Tensor& job_count, int64_t blocks, int64_t vector, const torch::Tensor& job_first);
// fill_program: the same from a store in another layout, rebuilt by a program [ops][8] int64 (expert_store.csf_layout).
void fill_program(const torch::Tensor& source, const torch::Tensor& store, const torch::Tensor& jobs,
                  const torch::Tensor& job_count, const torch::Tensor& program, int64_t shared, int64_t blocks);
