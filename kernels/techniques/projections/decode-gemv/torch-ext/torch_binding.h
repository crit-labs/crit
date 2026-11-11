#pragma once

#include <torch/torch.h>

// x [M][K] (M 1..16) BF16 or FP16, codes [N][K / 2] uint8, scales [N][K / 16] E4M3 bytes, inverse [N] FP32 ->
// out [M][N] FP32 or x's dtype.
// blocked: scales in cuBLASLt's 128 x 4 tiled layout instead of row major [N][K / 16].
void gemv(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& scales,
          const torch::Tensor& inverse, bool blocked, torch::Tensor& out);
// x [M][K] (M 1..16) BF16 or FP16, codes [N][K] E4M3 bytes, scales FP32 [N] (one per row) or E8M0 bytes [N][K / 32]
// (MXFP8) -> out [M][N] FP32 or x's dtype.
void gemv_fp8(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& scales, torch::Tensor& out);
// x [M][K] (M 1..16) BF16, w [N][K] BF16 -> out [M][N] FP32 or BF16.
void gemv_bf16(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out);
// x [M][K] (M 1..16) BF16 or FP16, INT4 with a zero point in crit's rows (codes [N][K / 2] uint8, scales FP16 and
// zeros uint8 [N][K / group]) -> out [M][N] FP32 or x's dtype.
void gemv_int4(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& scales,
               const torch::Tensor& zeros, int64_t group, torch::Tensor& out);
// AWQ (kind 0) or GPTQ (kind 1) tensors into crit's INT4 rows (decode_gemv.h, enqueue_repack_int4).
void repack_int4(int64_t kind, const torch::Tensor& qweight, const torch::Tensor& qzeros, const torch::Tensor& scales,
                 const std::optional<torch::Tensor>& perm, int64_t zero_offset, int64_t group, torch::Tensor& codes,
                 torch::Tensor& zeros, torch::Tensor& scales_out);
// 1 when the build's code for the device of `device_of` decodes E2M1 by the hardware conversion, 0 by the bit pattern.
int64_t hardware_decode(const torch::Tensor& device_of);
// The declared integer rows (structure rows-int-block): int-block-experts' rows (table of its Matrix records, jobs of
// RowJob records, int8 values and FP32 scales per block) -> out FP32.
void rows_int_block(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows,
                    const torch::Tensor& values, const torch::Tensor& scales, int64_t block, torch::Tensor& out);
// The declared FP8 rows (structure rows-fp8-block): fp8-block-rows' rows (E4M3 codes and FP32 scales per group).
void rows_fp8_block(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, const torch::Tensor& codes,
                    const torch::Tensor& scales, int64_t group, torch::Tensor& out);
