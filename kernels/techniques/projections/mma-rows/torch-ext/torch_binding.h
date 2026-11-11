#pragma once

#include <optional>

#include <torch/torch.h>

// out [M][N] (FP32, BF16 or FP16) = x [M][K] (M 1..64, BF16) times the matrix w (weights: 0 NVFP4 codes [N][K / 2]
// with E4M3 block scales and row_scale = one over each row's global scale; 1 FP8 E4M3 [N][K] with row_scale; 2 MXFP8
// E4M3 [N][K] with E8M0 scales [N][K / 32]; 3 BF16 [N][K]); acts: 0 x as it is, 1 x quantized per row to E4M3 (FP8
// weights), 2 x quantized per 16-value block to NVFP4 under input_scale (NVFP4 weights). blocked: NVFP4 scales in
// cuBLASLt's 128 x 4 tiled layout. groups (1, 2): 16-row groups a warp takes; warps (1..4): warps a CTA (speed only).
// splits (1..32): contiguous parts of K, one a CTA, part of the sums' order; splits > 1 take partial (FP32,
// splits * M * N) and counters (int32 zeros, one a row tile). Quantized activations take scratch (M * K * 2 + M * 4
// bytes, rounded to 16). INT4 weights (4) take FP16 scales and uint8 zeros [N][K / group].
void mm_rows(const torch::Tensor& x, const torch::Tensor& w, const std::optional<torch::Tensor>& scales,
             const std::optional<torch::Tensor>& row_scale, int64_t weights, int64_t acts, double input_scale,
             bool blocked, int64_t groups, int64_t warps, int64_t splits, const std::optional<torch::Tensor>& scratch,
             const std::optional<torch::Tensor>& partial, const std::optional<torch::Tensor>& counters,
             torch::Tensor& out, const std::optional<torch::Tensor>& zeros, int64_t group);
// q [M][K] E4M3 bytes and scale [M] FP32: x [M][K] BF16 quantized per row as mm_rows' acts 1.
void quantize_fp8_rows(const torch::Tensor& x, torch::Tensor& q, torch::Tensor& scale);
// codes [M][K / 2] bytes and scales E4M3 bytes in cuBLASLt's 128 x 4 tiled layout: x [M][K] BF16 quantized per
// 16-value block under input_scale as mm_rows' acts 2.
void quantize_nvfp4(const torch::Tensor& x, double input_scale, torch::Tensor& codes, torch::Tensor& scales);
// K values one step of mm_rows covers for `weights` (the plan's splits are of steps).
int64_t step_values(int64_t weights);
// Whether this build multiplies activations 3 (NVFP4 on the FP4 MMA) on like's device; the first call per device runs
// a one-thread probe kernel and waits for it (call it at load, not inside a capture).
bool fp4_mma(const torch::Tensor& like);
// Scratch bytes activations 3 take for `rows` rows of K.
int64_t fp4_scratch_bytes(int64_t rows, int64_t k);
// out [M][N] for any number of rows M >= 1, each element with mm_rows' bits for the same matrix, activations and
// splits (the prompt tile: each decoded weight reused for 64 rows a CTA). Quantized activations take scratch (M * K * 2
// + M * 4 bytes, rounded to 16); no partial sums or counters.
void mm_prompt(const torch::Tensor& x, const torch::Tensor& w, const std::optional<torch::Tensor>& scales,
               const std::optional<torch::Tensor>& row_scale, int64_t weights, int64_t acts, double input_scale,
               bool blocked, int64_t splits, const std::optional<torch::Tensor>& scratch, torch::Tensor& out, const std::optional<torch::Tensor>& zeros, int64_t group);
// CTAs a multiprocessor of the current device holds of mm_rows' one-row kernel for `weights` and `acts` (the FP4
// MMA's for acts 3, the decode tile's otherwise): plan()'s splits fill them.
int64_t decode_ctas(int64_t weights, int64_t acts);
