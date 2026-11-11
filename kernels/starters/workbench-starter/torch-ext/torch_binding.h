#pragma once

#include <torch/torch.h>

// y [columns][N] FP32 = x [columns][K] . w [N][K]^T: BF16, or E4M3 with x_scale [columns] and w_scale [N] FP32.
torch::Tensor run_bf16(const torch::Tensor& x, const torch::Tensor& w);
torch::Tensor run_fp8(const torch::Tensor& x, const torch::Tensor& x_scale, const torch::Tensor& w,
                      const torch::Tensor& w_scale);
