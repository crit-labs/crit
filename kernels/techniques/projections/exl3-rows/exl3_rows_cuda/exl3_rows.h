// EXL3 projections (the exl3 checkpoint scheme): y = x W for W [in][out] = diag(su) H Q H diag(sv), Q the trellis
// tiles (crit_kcore/exl3.cuh), H the block-diagonal normalized 128-point Hadamard, su and sv the sign vectors.
//
// A call rotates the activations (x su H in FP32, stored in FP16), multiplies them by the tiles decoded on the fly on
// FP16 tensor cores (mma.m16n8k16, FP32 sums), and rotates the result (y' H sv): the weights are read at their stored
// K bits a value and each tile is decoded once for all 1 to 16 rows. Each warp owns one 16-column tile strip and
// walks its split of the input tiles in order; the splits add in order. A row's sums never depend on the rows beside
// it. A prompt's rows go through the same tiles kernel in slabs of up to kPromptRows rows (slab_rows), the tiles
// decoded once a slab and multiplied by every row group of it: each row's instructions are those of a decode call, so
// a prompt row has the bits of the same token decoded. dequantize writes W^T [out][in] in BF16 or FP16 for a GEMM on
// another kernel or a library, which gives other bits.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace crit::techniques::exl3_rows {

constexpr int kMaxRows = 16;      // activation rows a gemv call takes
constexpr int kPromptRows = 64;   // activation rows a tiles call takes at most (a prompt's slab)
constexpr int kBlock = 128;    // the Hadamard's block

struct Matrix {
    const std::uint16_t* trellis = nullptr;   // [in / 16][out / 16][16 K]
    int in = 0, out = 0;
    int bits = 0;              // K's integer part
    bool half_bit = false;     // K = bits + 0.5 (mul1 codebook only)
    int codebook = 0;          // 0 3INST, 1 MCG, 2 MUL1
};

// rows [M][L] (FP32, BF16 or FP16 by `in_kind`: 0, 1, 2) -> out [M][L] (`out_kind`): each 128-value block times pre,
// rotated by H, times post (pre, post: FP16 [L] signs or null). FP32 rows may come as `parts` arrays part_stride
// apart, added in order first (a split GEMV's partial sums).
cudaError_t enqueue_rotate(const void* rows, int in_kind, int64_t ld_in, int M, int L, const void* pre,
                           const void* post, void* out, int out_kind, int64_t ld_out, cudaStream_t stream,
                           int parts = 1, int64_t part_stride = 0);
// The splits of the input's tiles a gemv of this matrix runs on this device (its tuning's SPLITS, or 0's rule: enough
// warps for every multiprocessor's 32, at most one split a tile): part of the sums' order, by the shape and device alone.
int splits(const Matrix& w, int rows, int multiprocessors);
// xh [M][in] FP16 (rotated activations) -> yr [splits][M][out] FP32: split s's sum of xh Q over its input tiles. M up
// to slab_rows(w): past kMaxRows the matrix's configuration at kMaxRows rows runs with eight row groups a tile.
cudaError_t enqueue_tiles(const Matrix& w, const void* xh, int M, float* yr, int splits, cudaStream_t stream);
// The rows a tiles call takes for a prompt of this matrix: kPromptRows, or kMaxRows where the configuration's block
// with eight row groups would take more shared memory than a block may.
int slab_rows(const Matrix& w);
// Q [in][out] FP16, the tiles as stored values (for checks and dequantize).
cudaError_t enqueue_decode(const Matrix& w, void* q, cudaStream_t stream);

}  // namespace crit::techniques::exl3_rows
