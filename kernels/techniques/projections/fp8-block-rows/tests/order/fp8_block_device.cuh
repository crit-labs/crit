// The declared order of the matched FP8 pair as device functions, the reference this project's kernels follow (moved
// here from crit's quantized-projections technique, whose code keeps the host twin fp8_block.h and the order's
// description).
//
// Matched FP8 decode and prompt projections: one declared per-output order for E4M3 weights against E4M3 activations,
// with MXFP8's power-of-two block scales, an FP8 checkpoint's FP32 group scales or one scale per row, so a decode
// kernel shaped for 1-8 tokens and a prompt GEMM give the same bits for every output (crit's honesty, request-history
// and prompt-cache gates compare them).
//
// The order. Weights are the A operand of an e4m3 mma.sync m16n8k32 (16 rows), tokens the B operand (8 columns).
// Per 32-element block b of a row and a token:
//   1. the block's 32 products summed by one MMA with a zero accumulator, its bytes in the fragment order below;
//   2. lane b % 16 = fma(block sum, weight scale * activation scale (the product rounded), lane), lanes from zero,
//      blocks in increasing order;
//   3. the 16 lanes reduced by the halving tree: (i, i + 8), then (i, i + 4), (i, i + 2), (i, i + 1);
//   4. times the matrix's post.
// A kernel may compute the lanes in any order (kLaneOrder and fold() keep one running lane and four partial sums),
// split them over warps, and tile any number of rows and tokens: each output's operations stay the same.
//
// What it rests on. The tensor core computes D[i][j] from row i of A and column j of B alone, the same way at every
// position of the tile, so a block sum does not depend on the tile's other rows and tokens or on where in the tile they
// sit. PTX does not specify the tensor core's internal accumulation, so this is checked, not assumed:
// tests/fp8_block_order_tests.cu moves rows and tokens across tile positions and companions and requires the same
// bits. The internal sum may differ between GPU architectures: check on the target, and keep the CPU out of FP8
// projections (it cannot reproduce the tensor core's sum; host-expert-arithmetic's integer rows are the exact
// cross-processor path for 4-bit weights).
//
// The fragment order. Within a block, thread q of a quad (lane % 4) holds bytes 8q .. 8q + 7 of its row (A) or token
// (B): bytes 8q .. 8q + 3 in the fragment's first register of the pair (k = 4q .. 4q + 3), bytes 8q + 4 .. 8q + 7 in
// the second (k = 16 + 4q .. 16 + 4q + 3). One 8-byte load per row or token and block.
//
// The activation quantizer is declared too (quantize_group_device): MXFP8's shared exponent 2^(floor(log2 amax) - 8)
// or an FP8 checkpoint's amax / 448, E4M3 with round to nearest even and saturation (the hardware conversion), so a
// token's codes do not depend on the kernel that quantized it. quantize_group (fp8_block.h) is its host twin.
// e4m3 mma.sync needs sm_89 or newer; on older GPUs mma_block traps.
#pragma once
#include "fp8_block.h"

#include <cstdint>
#include <cuda_fp8.h>

namespace crit::fp8_block {

constexpr int kBlock = 32;
constexpr int kLanes = 16;

// codes [rows][cols] E4M3; scales FP32 [rows / scale_rows][cols / scale_k] (MXFP8: E8M0 bytes decoded once, scale_rows
// 1, scale_k 32; an FP8 checkpoint's 128 x 128 blocks: 128, 128; one scale per row: 1, cols). cols a multiple of 32,
// scale_k a multiple of 32 dividing cols.
struct Matrix {
    const std::uint8_t* codes = nullptr;
    const float* scales = nullptr;
    int rows = 0, cols = 0, scale_rows = 1, scale_k = kBlock;
    float post = 1.0f;
};
// Quantized activations: codes [token][ld] E4M3, scales FP32 [token][lds], one per `group` (a multiple of 32) codes.
struct Acts {
    const std::uint8_t* codes = nullptr;
    const float* scales = nullptr;
    int ld = 0, lds = 0, group = kBlock;
};
__device__ inline float weight_scale(const Matrix& m, int row, int b) {
    return m.scales[std::size_t(row / m.scale_rows) * std::size_t(m.cols / m.scale_k) +
                    std::size_t(b * kBlock / m.scale_k)];
}
__device__ inline float act_scale(const Acts& x, int token, int b) {
    return x.scales[std::size_t(token) * std::size_t(x.lds) + std::size_t(b * kBlock / x.group)];
}
// One row's or token's 8 bytes of block b for quad thread q (zero for a missing row or token, -1).
__device__ inline uint2 fragment(const std::uint8_t* base, std::size_t ld, int index, int b, int q) {
    if(index < 0) return make_uint2(0u, 0u);
    return *reinterpret_cast<const uint2*>(base + std::size_t(index) * ld + std::size_t(b * kBlock + 8 * q));
}

// Block sums of rows g and g + 8 (a_g, a_g8) against token g's bytes (t_g): d[2h + c] is row g + 8h against the token
// in column 2q + c, as mma.sync lays D out.
__device__ inline void mma_block(uint2 a_g, uint2 a_g8, uint2 t_g, float (&d)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %10, %10, %10};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a_g.x), "r"(a_g8.x), "r"(a_g.y), "r"(a_g8.y), "r"(t_g.x), "r"(t_g.y), "f"(0.0f));
#else
    (void)a_g, (void)a_g8, (void)t_g;
    d[0] = d[1] = d[2] = d[3] = 0.0f;
    __trap();
#endif
}

// Step 2 for one output: lane = fma(block sum, ws * as, lane), the product rounded first.
__device__ inline float step(float sum, float ws, float as, float lane) {
    return __fmaf_rn(sum, __fmul_rn(ws, as), lane);
}

// Step 3 over all 16 lanes.
__device__ inline float halving_tree(float (&lanes)[kLanes]) {
#pragma unroll
    for(int w = kLanes / 2; w >= 1; w /= 2)
#pragma unroll
        for(int i = 0; i < w; ++i) lanes[i] = __fadd_rn(lanes[i], lanes[i + w]);
    return lanes[0];
}

// The 16 lanes in the order the halving tree consumes them; fold(step, a, x) adds lane kLaneOrder[step]'s finished sum
// into the partial sums x[0..3], so an output holds one running lane and four partial sums instead of 16 lanes:
// ((a0 + a8) + (a4 + a12)) + ((a2 + a10) + (a6 + a14)), then the same over the odd lanes, then their sum.
__device__ constexpr int kLaneOrder[kLanes] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
__device__ inline void fold(int at, float a, float (&x)[4]) {
    switch(at) {
        case 0: x[0] = a; break;
        case 1: x[0] = __fadd_rn(x[0], a); break;
        case 2: x[1] = a; break;
        case 3: x[1] = __fadd_rn(x[1], a); x[0] = __fadd_rn(x[0], x[1]); break;
        case 4: x[1] = a; break;
        case 5: x[1] = __fadd_rn(x[1], a); break;
        case 6: x[2] = a; break;
        case 7: x[2] = __fadd_rn(x[2], a); x[1] = __fadd_rn(x[1], x[2]); x[0] = __fadd_rn(x[0], x[1]); break;
        case 8: x[1] = a; break;
        case 9: x[1] = __fadd_rn(x[1], a); break;
        case 10: x[2] = a; break;
        case 11: x[2] = __fadd_rn(x[2], a); x[1] = __fadd_rn(x[1], x[2]); break;
        case 12: x[2] = a; break;
        case 13: x[2] = __fadd_rn(x[2], a); break;
        case 14: x[3] = a; break;
        default: x[3] = __fadd_rn(x[3], a); x[2] = __fadd_rn(x[2], x[3]); x[1] = __fadd_rn(x[1], x[2]);
                 x[0] = __fadd_rn(x[0], x[1]); break;
    }
}

// The declared order for one tile, by one warp: rows row_of[0..15] and tokens token_of[0..7] (-1 for none), every lane
// from zero over its blocks in order, the halving tree, post. out[2h + c] is row row_of[g + 8h] against token
// token_of[2q + c] (g = lane / 4, q = lane % 4); missing rows or tokens give unused outputs.
__device__ inline void tile(const Matrix& m, const Acts& x, const int* row_of, const int* token_of, float (&out)[4]) {
    const int lane = int(threadIdx.x % 32u), g = lane / 4, q = lane % 4;
    const int ra = row_of[g], rb = row_of[g + 8], tg = token_of[g];
    const int r[2] = {ra, rb}, t[2] = {token_of[2 * q], token_of[2 * q + 1]};
    const int blocks = m.cols / kBlock;
    float lanes[4][kLanes];
#pragma unroll
    for(int o = 0; o < 4; ++o)
#pragma unroll
        for(int j = 0; j < kLanes; ++j) lanes[o][j] = 0.0f;
    for(int b = 0; b < blocks; ++b) {
        float d[4];
        mma_block(fragment(m.codes, std::size_t(m.cols), ra, b, q), fragment(m.codes, std::size_t(m.cols), rb, b, q),
                  fragment(x.codes, std::size_t(x.ld), tg, b, q), d);
#pragma unroll
        for(int o = 0; o < 4; ++o) {
            const int row = r[o / 2], token = t[o % 2];
            if(row < 0 || token < 0) continue;
            lanes[o][b % kLanes] = step(d[o], weight_scale(m, row, b), act_scale(x, token, b), lanes[o][b % kLanes]);
        }
    }
#pragma unroll
    for(int o = 0; o < 4; ++o) out[o] = __fmul_rn(halving_tree(lanes[o]), m.post);
}

// The declared activation quantizer for one group of n values: codes and the group's scale (MXFP8: 2^(floor(log2 amax)
// - 8), clamped to E8M0's 2^-127 .. 2^127; amax: amax / 448). An all-zero group gets scale 1 (mx) or 0 (amax) and zero
// codes. Finite inputs.
__device__ inline float quantize_group_device(const float* v, int n, ScaleKind kind, std::uint8_t* codes) {
    float amax = 0.0f;
    for(int k = 0; k < n; ++k) amax = fmaxf(amax, fabsf(v[k]));
    float scale;
    if(kind == ScaleKind::mx) {
        int e = 0;
        (void)frexpf(amax, &e);   // amax = f * 2^e, f in [0.5, 1): floor(log2 amax) = e - 1
        int shared = amax > 0.0f ? e - 1 - 8 : 0;
        shared = shared < -127 ? -127 : shared > 127 ? 127 : shared;
        scale = ldexpf(1.0f, shared);
    } else {
        scale = __fdiv_rn(amax, 448.0f);
    }
    for(int k = 0; k < n; ++k)
        codes[k] = scale > 0.0f
                       ? std::uint8_t(__nv_cvt_float_to_fp8(__fdiv_rn(v[k], scale), __NV_SATFINITE, __NV_E4M3))
                       : std::uint8_t(0);
    return scale;
}

}  // namespace crit::fp8_block
