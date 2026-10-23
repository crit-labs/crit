// CUDA projections over GGUF blocks as stored in the file: a dequantizing copy and a GEMV for up to 8 activation rows
// that decodes each block in registers. Decoded values equal gguf-py's quants.dequantize bit for bit (gguf_blocks.h).
#pragma once
#include "gguf_blocks.h"

#include <cstddef>
#include <cuda_runtime.h>

namespace crit::gguf_blocks {

// out[i] for every element of `blocks` consecutive blocks (FP32).
cudaError_t dequantize(Kind kind, const void* blocks, std::size_t count, float* out, cudaStream_t stream);

// y[b][r] = sum_c w[r][c] * x[b][c] for b < batch (1..8): w is rows x cols stored as blocks, row after row (cols a
// multiple of the format's values; the row stride is cols / values blocks), x is [batch][cols] and y [batch][rows]
// in FP32. One warp per row accumulates in FP32 in a fixed order, so results are deterministic.
constexpr int kMaxBatch = 8;
cudaError_t gemv(Kind kind, const void* blocks, int rows, int cols, const float* x, int batch, float* y, cudaStream_t stream);

}  // namespace crit::gguf_blocks
