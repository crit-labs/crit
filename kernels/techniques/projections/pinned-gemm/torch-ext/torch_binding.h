#pragma once

#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

// One cuBLASLt algorithm per GEMM shape and format, run for every row count (see pinned_gemm.cu).
//
// fmt is [a type, b type, output type, scale mode]: types fp16, bf16, fp32, e4m3, e5m2, e2m1 (two a byte, the first
// element in the low nibble); outputs bf16, fp16, fp32; scale modes none, tensor (one FP32 element an operand, absent
// means 1), row (FP32 vectors of a's M rows and b's N rows), vec32 (one UE8M0 scale per 32 elements along K), vec16
// (one UE4M3 scale per 16, with the FP32 global scale alpha read from device memory). Block scales are in cuBLASLt's
// blocked layout (128-row by 4-column tiles, rows padded to 128 and columns to 4).

// How many split-K-free algorithms cuBLASLt's heuristic lists for an [N, K] GEMM of the format at rep_rows rows
// within `workspace` bytes.
int64_t pinned_candidates(const std::vector<std::string>& fmt, int64_t n, int64_t k, int64_t rep_rows,
                          int64_t workspace);
// A plan id for the index-th of those algorithms; the plan runs any row count with it.
int64_t pinned_plan(const std::vector<std::string>& fmt, int64_t n, int64_t k, int64_t rep_rows, int64_t workspace,
                    int64_t index);
// The plan's algorithm configuration: algorithm id, tile, stages, split-K count, reduction scheme, swizzle, custom
// option, inner shape, cluster shape.
std::vector<int64_t> pinned_describe(int64_t plan);
// out [M, N] = alpha (scale_a a) (scale_b b)^T: a [M, K] row major (activation rows), b [N, K] row major (a weight),
// FP32 accumulation, with the plan's algorithm whatever M is; out may be N columns of a wider row-major matrix. A
// vec16 plan takes alpha and beta (zero) as one FP32 element each on the device. workspace: at least the plan's
// workspace bytes, on the device.
void pinned_run(int64_t plan, const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& out,
                const std::optional<torch::Tensor>& scale_a, const std::optional<torch::Tensor>& scale_b,
                const std::optional<torch::Tensor>& alpha, const std::optional<torch::Tensor>& beta,
                torch::Tensor& workspace);
// cuBLASLt's version (cublasLtGetVersion) of the library the process loaded.
int64_t pinned_version();
