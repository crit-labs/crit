#pragma once

#include <optional>

#include <torch/torch.h>

// out [M][L] = rows [M][L] (FP32, BF16, FP16) in 128-value blocks: times pre, the normalized Hadamard, times post (FP16
// sign vectors [L], or none).
void rotate(const torch::Tensor& rows, const std::optional<torch::Tensor>& pre,
            const std::optional<torch::Tensor>& post, torch::Tensor& out);
// The splits of the input's tiles gemv takes for this matrix and row count on this device.
int64_t splits(const torch::Tensor& trellis, int64_t codebook, int64_t rows);
// yr [splits][M][out] FP32: each split's sum of xr [M][in] FP32 times the trellis tiles (codebook 0 3INST, 1 MCG,
// 2 MUL1) over its part of the input, M 1..16.
void tiles(const torch::Tensor& trellis, int64_t codebook, const torch::Tensor& xr, torch::Tensor& yr);
// q [in][out] FP16: the trellis tiles' stored values.
void decode(const torch::Tensor& trellis, int64_t codebook, torch::Tensor& q);
