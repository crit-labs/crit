// crit's kernel core: multiply paths (kernels/core/README.md, "multiply"). A projection kernel is a decoder (the weight
// format: formats.cuh, a trait each in the kernel) and a multiply path, chosen by (weight format, activation format,
// architecture):
//
//   exact_fp32    16-bit activations (W*A16): each weight decoded to its exact FP32 value, its product with an
//                 activation exact, FP32 sums in the kernel's fixed order (decode-gemv's gemv-cuda, the Triton GEMV's
//                 fma path), or the values exact in a 16-bit MMA operand (mma-rows, grouped-experts, the dot path).
//   declared_int  crit's declared integer block order (host-expert-arithmetic): 4-bit weights (E2M1 x 2 or INT4, an
//                 E4M3 scale per 16) against activations quantized per block to int8 (E2M1 x 2 or INT8, an FP32 scale
//                 per 16 or 32): each 16-block's integer dot exact (dp4a, the s8 MMA), then the declared step on lane
//                 b % 16 and the halving tree (blockscaled_mma.cuh's step, reduce.cuh's tree).
//   declared_fp8  crit's declared FP8 block order (quantized-projections): E4M3 weights with a scale grid (MXFP8's
//                 E8M0 per 32, 128 x 128 FP32 blocks, a scale a row) against E4M3 activations per group: each 32-block
//                 summed by one e4m3 m16n8k32 MMA with a zero accumulator in crit's fragment order, then the same
//                 step and tree.
//
// The declared paths are a different multiply from the exact one (a block sum, then one rounded product and one FMA a
// block), so a format's bits differ between paths: a kernel's structure names its path, never mixes them for one
// output. Their activations come from a quantizer pre-pass (the declared quantizers, int-block-experts and
// fp8-block-rows), and their weight scales are side data of the decoder (the E4M3 byte a block; ScaleGrid for FP8).
// The declared decode rows themselves are declared_rows.cuh, on these blocks.
#pragma once
#include "crit_kcore/blockscaled_mma.cuh"
#include "crit_kcore/formats.cuh"

#include <cstddef>
#include <cstdint>

namespace crit::kcore::multiply {

// Weight formats: decode-gemv's and mma-rows' codes (0-4, multiplied by 16-bit activations), then the declared
// paths' block formats.
enum Weights : int {
    kNvfp4 = 0, kFp8 = 1, kMxfp8 = 2, kBf16 = 3, kInt4Zp = 4,
    kFourBitBlocks = 5,   // E2M1 x 2 or INT4 codes two a byte, an E4M3 scale per 16, a post factor a matrix
    kE4m3Grid = 6,        // E4M3 codes, a ScaleGrid, a post factor a matrix
};
// Activation formats.
enum Acts : int {
    kAct16 = 0,        // BF16 or FP16 values
    kInt8Blocks = 1,   // int8 values (E2M1 x 2 or INT8) and an FP32 scale per 16 or 32 (the declared quantizers)
    kE4m3Groups = 2,   // E4M3 codes and an FP32 scale per group, a multiple of 32
};
enum Path : int { kNoPath = -1, kExactFp32 = 0, kDeclaredInt = 1, kDeclaredFp8 = 2 };

// The multiply path for a weight format, an activation format and an architecture (compute capability as 10 major +
// minor, 80 for sm_80): the declared integer path needs the s8 MMA and dp4a (sm_80 on), the FP8 one the e4m3 MMA
// (sm_89 on); kNoPath where no kernel multiplies the pair exactly or in a declared order.
__host__ __device__ constexpr Path path_of(int weights, int acts, int arch) {
    if(acts == kAct16) return weights >= kNvfp4 && weights <= kInt4Zp && arch >= 80 ? kExactFp32 : kNoPath;
    if(acts == kInt8Blocks) return weights == kFourBitBlocks && arch >= 80 ? kDeclaredInt : kNoPath;
    if(acts == kE4m3Groups) return weights == kE4m3Grid && arch >= 89 ? kDeclaredFp8 : kNoPath;
    return kNoPath;
}

// The declared integer block: 16 weight codes (8 bytes) to int8 values, their exact dot with 16 int8 activations by
// four dp4a, the step on its lane.
struct DeclaredInt {
    static constexpr int kBlock = 16, kLanes = 16;
    __device__ __forceinline__ static void weights(uint2 q, const blockscaled::NibbleTable& table, int (&w)[4]) {
        w[0] = int(blockscaled::nibbles_to_s8x4(q.x & 0xffffu, table));
        w[1] = int(blockscaled::nibbles_to_s8x4(q.x >> 16, table));
        w[2] = int(blockscaled::nibbles_to_s8x4(q.y & 0xffffu, table));
        w[3] = int(blockscaled::nibbles_to_s8x4(q.y >> 16, table));
    }
    __device__ __forceinline__ static int dot(const int (&w)[4], int4 x) {
        int sum = __dp4a(w[0], x.x, 0);
        sum = __dp4a(w[1], x.y, sum);
        sum = __dp4a(w[2], x.z, sum);
        return __dp4a(w[3], x.w, sum);
    }
    __device__ __forceinline__ static float step(int sum, float weight_scale, float act_scale, float lane) {
        return blockscaled::step(float(sum), weight_scale, act_scale, lane);
    }
    // The halving tree of a half-warp's 16 lanes (lane b % 16 in lane b of the half `group` masks), by shuffles:
    // lane 0 gets ((l0 + l8) + (l4 + l12)) + ... .
    __device__ __forceinline__ static float tree(float v, unsigned group) {
        for(int w = 8; w >= 1; w /= 2) v = __fadd_rn(v, __shfl_down_sync(group, v, w, 16));
        return v;
    }
};

// The declared FP8 block: a 32-element block's sums for two weight rows (g, g + 8) and two tokens (2q, 2q + 1) by one
// e4m3 MMA with a zero accumulator, the step on its lane, the tree over 16 lane values.
struct DeclaredFp8 {
    static constexpr int kBlock = 32, kLanes = 16;
    __device__ __forceinline__ static void block(uint2 a_g, uint2 a_g8, uint2 b, float (&d)[4]) {
        blockscaled::block_sum_e4m3(a_g, a_g8, b, d);
    }
    __device__ __forceinline__ static float step(float sum, float weight_scale, float act_scale, float lane) {
        return blockscaled::step(sum, weight_scale, act_scale, lane);
    }
    __device__ __forceinline__ static float tree(float (&v)[kLanes]) {
#pragma unroll
        for(int w = kLanes / 2; w >= 1; w /= 2)
#pragma unroll
            for(int i = 0; i < w; ++i) v[i] = __fadd_rn(v[i], v[i + w]);
        return v[0];
    }
};

// The FP8 path's weight scale grid, side data of its decoder: [ceil(rows / scale_rows)][cols / scale_k] FP32 or E8M0
// bytes (MXFP8: 1, 32; 128 x 128 blocks: 128, 128; a scale a row: 1, cols).
struct ScaleGrid {
    const std::uint8_t* scales;
    int cols, scale_rows, scale_k, e8m0;
    // The scale row of weight row r.
    __device__ __forceinline__ const std::uint8_t* row(int r) const {
        const std::size_t ld = std::size_t(cols / scale_k) * (e8m0 ? 1u : 4u);
        return scales + std::size_t(r / scale_rows) * ld;
    }
    // Scale k of a scale row: FP32, or an E8M0 byte (2^(e - 127)).
    __device__ __forceinline__ static float at(const std::uint8_t* row, int k, int e8m0) {
        return e8m0 ? e8m0_value(row[k]) : reinterpret_cast<const float*>(row)[k];
    }
};

}  // namespace crit::kcore::multiply
