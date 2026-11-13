#pragma once

#include <optional>

#include <torch/torch.h>

// xq int8 [M][K], xd FP32 [M][K / 32], xs int32 [M][K / 32]: x [M][K] (FP32, BF16, FP16) in 32-value blocks; by
// sub-block (gemm's layout) xq [K / 32][M][32], xd and xs [K / 32][M].
void quantize(const torch::Tensor& x, torch::Tensor& xq, torch::Tensor& xd, torch::Tensor& xs, bool by_subblock);
// The parts gemv splits a row of this matrix into on this device.
int64_t splits(const torch::Tensor& blocks, int64_t type, int64_t k);
// yp FP32 [splits][M][N]: each part's sums of the quantized rows times the GGUF matrix (blocks uint8 [N][row bytes]),
// or, with ids int32 [M], row m times expert ids[m] of blocks [experts][N][row bytes].
void gemv(const torch::Tensor& blocks, int64_t type, int64_t k, const torch::Tensor& xq, const torch::Tensor& xd,
          const torch::Tensor& xs, const std::optional<torch::Tensor>& ids, torch::Tensor& yp);
// y [M][N]: any number of quantized rows (by sub-block) times the matrix, each output gemv's bits (`parts` its
// splits); the matrix's rows decoded chunk_rows at a time (0: as many as a 128 MiB staging buffer holds).
void gemm(const torch::Tensor& blocks, int64_t type, int64_t k, const torch::Tensor& xq, const torch::Tensor& xd,
          const torch::Tensor& xs, int64_t parts, torch::Tensor& y, int64_t chunk_rows);
// y [M][N] = yp's parts added in part order, rounded once to y's dtype.
void sum_parts(const torch::Tensor& yp, torch::Tensor& y);
// out [N][K] (FP32, BF16, FP16): the stored values of expert `expert` (0 for a plain matrix).
void dequantize(const torch::Tensor& blocks, int64_t type, int64_t k, int64_t expert, torch::Tensor& out);
