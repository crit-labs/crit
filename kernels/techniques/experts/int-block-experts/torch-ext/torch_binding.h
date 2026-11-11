#pragma once

#include <torch/torch.h>

#include <optional>

// table: one Matrix per matrix (uint8); jobs: RowJob or GemmJob records (uint8); values [rows, cols] int8 and scales
// [rows, cols / block] fp32: the quantized activations; out [rows, out_ld] fp32.
void rows(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, const torch::Tensor& values,
          const torch::Tensor& scales, int64_t block, torch::Tensor& out);
void gemm(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, int64_t max_count,
          const torch::Tensor& values, const torch::Tensor& scales, int64_t block, torch::Tensor& out);
void gemm_staged(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, int64_t max_count,
                 const torch::Tensor& values, const torch::Tensor& scales, int64_t block, torch::Tensor& out);
// The prompt GEMM accumulated in the tensor cores (not the declared order) on quantize_nvfp4's packed activations.
void gemm_native(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, int64_t max_count,
                 const torch::Tensor& codes, const torch::Tensor& scales, double global, bool int4,
                 torch::Tensor& out, const std::optional<torch::Tensor>& col_scale, int64_t tiled,
                 const std::optional<torch::Tensor>& host_matrices);
// x [rows, cols] fp32 or bf16 -> E2M1 codes uint8 [rows, cols / 2] and E4M3 scale bytes (undivided), [rows, cols / 16]
// or (tiled bit 0) in cuBLASLt's 128 x 4 tiled layout.
void quantize_nvfp4(const torch::Tensor& x, double global, torch::Tensor& codes, torch::Tensor& scales, int64_t tiled);
// x [rows, cols] fp32 -> values int8 and scales fp32 (E2M1 x 2 per 16 with a global divisor, or INT8 per block).
void quantize_e2m1x2(const torch::Tensor& x, double global, torch::Tensor& values, torch::Tensor& scales);
void quantize_int8(const torch::Tensor& x, int64_t block, torch::Tensor& values, torch::Tensor& scales);
// gate_up [rows, 2 * inter] fp32 -> out [rows, inter] fp32, crit's declared SwiGLU.
void swiglu(const torch::Tensor& gate_up, torch::Tensor& out);
