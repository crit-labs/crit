// GGUF projections: y = x W^T for W [N][K] stored in a GGUF block format (crit_kcore/gguf.cuh: Q4_0, Q4_1, Q5_0,
// Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS,
// MXFP4), read at its stored bits.
//
// The declared order, the same bits on every GPU and for any rows beside a row:
//  1. quantize: each row's 32-value blocks of x become int8 q = rint(x / d) (d = max |x| / 127 in FP32, q = 0 where
//     d is 0) with d and the block's sum of q;
//  2. a matrix row's 32-value sub-blocks j, in order, each give c_j in FP32 from the sub-block's exact integer dot
//     (dp4a): fmul(fmul(s, d), dot), or with two scales (Q2_K, Q3_K, Q6_K, IQ2_XS, IQ2_S, IQ1_M) fadd of one such term a 16-value half,
//     less fmul(fmul(m, d), sum) where the format has a minimum (Q4_1, Q5_1, Q4_K, Q5_K); Q2_K's (integer scales and
//     minimums a 16 values) fmul(d_x, fsub(fmul(dw, isum), fmul(dmin, imin))), isum the exact dot of the codes times
//     their scales with q, imin = mn0 first + mn1 (sum - first), first the integer sum of the block's first 16 q;
//  3. the sub-blocks split into `splits` contiguous parts (j from J p / P up to J (p + 1) / P, J sub-blocks a row);
//     in a part, sub-block j goes to lane (j - start) % 32, a lane adds its terms in order, the 32 lanes reduce by the
//     xor butterfly (16, 8, 4, 2, 1), and the parts add in part order; the sum rounds once to the output's dtype.
// splits is a function of the matrix's shape and the device (or the tuning's SPLITS), never of the rows.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace crit::techniques::gguf_rows {

constexpr int kMaxRows = 8;   // activation rows a gemv call takes

struct Matrix {
    const std::uint8_t* blocks = nullptr;   // [experts][N][row_bytes], experts 1 for a plain matrix
    int type = 0;                            // GGUF type id
    int N = 0, K = 0;
    int64_t row_bytes = 0;                   // K / block_values * block_bytes
    int64_t expert_stride = 0;               // bytes between experts (N row_bytes)
};

// x [M][K] (FP32, BF16 or FP16 by kind 0, 1, 2) -> xq [M][K] int8, xd [M][K / 32] FP32, xs [M][K / 32] int32; or,
// by_subblock (gemm's layout, a sub-block's rows together), xq [K / 32][M][32], xd and xs [K / 32][M].
cudaError_t enqueue_quantize(const void* x, int kind, int64_t ld, int M, int K, int8_t* xq, float* xd, int32_t* xs,
                             bool by_subblock, cudaStream_t stream);
// The parts gemv splits a row's sub-blocks into for this matrix on this device (the tuning's SPLITS, or 0's rule).
int splits(const Matrix& w, int multiprocessors);
// yp [splits][M][N] FP32: part p's sums of the quantized rows times W (rows 1..kMaxRows); with `ids` ([M] int32),
// row m multiplies expert ids[m] of w (the gathered form).
cudaError_t enqueue_gemv(const Matrix& w, const int8_t* xq, const float* xd, const int32_t* xs, int M, const int32_t* ids,
                         float* yp, int splits, cudaStream_t stream);
// Rows n0 .. n0 + rows of W decoded for the prompt GEMM, each sub-block as gemv decodes it: codes int8 [J][rows][32]
// and scales FP32 [planes][J][rows] (s0, then s1 for a format with two scales or m for one with a minimum; Q2_K's
// codes times their scales and three planes d, dmin, mn0 | mn1 << 8; J = K / 32).
struct Staged {
    const int8_t* codes = nullptr;
    const float* scales = nullptr;
    int rows = 0, K = 0;
    int type = 0;   // GGUF type id
};
cudaError_t enqueue_stage(const Matrix& w, int n0, int rows, int8_t* codes, float* scales, cudaStream_t stream);
// y [M][rows] (kind, row stride ld): any number of quantized rows (enqueue_quantize's by_subblock layout) times the
// staged rows in gemv's order, each output gemv's bits for its row (the prompt GEMM on the s8 tensor cores). key_n: the
// whole matrix's N, which the tuning looks up.
cudaError_t enqueue_gemm(const Staged& w, int key_n, const int8_t* xq, const float* xd, const int32_t* xs, int M,
                         int parts, void* y, int kind, int64_t ld, cudaStream_t stream);
// y [M][N] (kind) = the parts of yp added in part order, rounded once.
cudaError_t enqueue_sum(const float* yp, int splits, int M, int N, void* y, int kind, int64_t ld, cudaStream_t stream);
// out [N][K] (kind) = W's stored values (one rounding for BF16 and FP16; FP32 exact). expert: which of w's experts.
cudaError_t enqueue_dequantize(const Matrix& w, int expert, void* out, int kind, cudaStream_t stream);

}  // namespace crit::techniques::gguf_rows
