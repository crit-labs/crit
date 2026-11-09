// Integer block rows: block-scaled expert rows against activations quantized per block, exact on any processor and
// fast on every one, so tiers that compute the same expert give the same bits without paying for FP32 rows.
//
// The idea. A 4-bit weight's value times 2 is a small integer (E2M1: 0, 1, 2, 3, 4, 6, 8, 12 and their negatives; INT4:
// -8..7), and so is a quantized activation (E2M1 x 2 for 4-bit activations, -127..127 for 8-bit ones). A block's dot of
// those integers is an exact int32 whatever order the processor adds it in (at most 16 x 12 x 127 = 24,384), so VNNI,
// AVX2's maddubs, Arm's SDOT, CUDA's dp4a and int8 tensor cores all give the same block sums. Only the FP32 part
// is declared: block b goes to lane b % lanes; lane = fma(float(sum_b), wscale_b * ascale_b, lane) with the scale
// product rounded first; blocks in order; the lanes reduced by host_nvfp4's halving tree (or sequentially); the
// result times post. A device row or a device GEMM that declares the same (host_nvfp4_device.cuh's dot_row_int, the
// crit-labs/int-block-experts Hub kernel) gives the same bits, so the CPU can take an expert's rows at its
// integer speed (DRAM-bound with a few threads) and the output does not depend on where they ran.
//
// For a checkpoint whose experts declare 4-bit input activations (NVFP4 W4A4), this is the arithmetic the checkpoint
// declares, the same bits as the device's; for weight-only checkpoints, 8-bit activations are a quantization of the
// engine's own. Against Eval's reference (exact activations) both are drift whatever the checkpoint declares:
// speed's budget admits it, preserve measures it, and host_nvfp4.h's FP32 rows are the drift-free order.
//
// The activation quantizers, the SiLU and the SwiGLU product are declared too (IEEE operations only, an exp of
// FMAs and an exact power-of-two scaling), so the expert's intermediate is quantized to the same codes on either
// processor. Compile without FP contraction, like host_nvfp4.
#pragma once
#include "host_isa.h"
#include "host_nvfp4.h"

#include <cstddef>
#include <cstdint>

namespace crit::techniques::host_int {

using host_isa::Isa;
constexpr int kMaxTokens = 8;
constexpr int kWeightBlock = 16;

// Integer values of 4-bit weight codes (low nibble the even element): E2M1 times 2 or INT4 two's complement.
enum class Weights { e2m1x2, int4 };

// Where a row's codes sit. rows: the checkpoint's [cols / 2] bytes, low nibble the even element. interleaved: a host
// store's layout for the CPU, per group of 16 blocks (256 elements) 4 chunks of 32 bytes, chunk c holding elements
// 4c..4c+3 of each block, block i < 8 in the low nibbles of bytes 4i..4i+3 and block i >= 8 in the high nibbles of
// bytes 4(i - 8)..; a short last group is padded with zero codes. A vector lane then holds one block's elements, so
// vpdpbusd (or vpmaddubsw) accumulates exact block sums across chunks without reducing within a vector. Values and
// bits are the same in both layouts: only the addresses differ (interleave converts).
enum class Layout { rows, interleaved };

// One weight matrix: codes in `layout`, cols a multiple of 64; block scales per 16 weights through the decoder
// (host_nvfp4's E4M3 decoder for NVFP4, or an engine's hook), or byte_table over `scales` bytes.
struct IntMatrix {
    const std::uint8_t* codes = nullptr;
    const std::uint8_t* scales = nullptr;
    std::size_t rows = 0, cols = 0;
    std::size_t code_stride = 0, scale_stride = 0;   // bytes; defaults row_bytes(layout, cols) and cols / 16
    Weights weights = Weights::e2m1x2;
    Layout layout = Layout::rows;
};
// The bytes of one row's codes in a layout: cols / 2, or 128 per group of 16 blocks.
std::size_t row_bytes(Layout, std::size_t cols) noexcept;
// One row of codes from the rows layout into the interleaved one (row_bytes(interleaved, cols) bytes).
void interleave_row(const std::uint8_t* row, std::size_t cols, std::uint8_t* out);

// Quantized activations of `tokens` rows: integer values [t * ld + k] and FP32 scales [t * lds + k / block], block 16
// or 32 (a multiple of the weights' 16). A value times its scale is the activation the arithmetic means; weights'
// and activations' factors of 2 and global scales go in post.
struct IntActs {
    const std::int8_t* values = nullptr;
    std::size_t ld = 0;
    const float* scales = nullptr;
    std::size_t lds = 0;
    int block = 16;
};

struct IntOrder {
    int lanes = 16;   // 16 or 32
    host_nvfp4::Reduce reduce = host_nvfp4::Reduce::halving;
};

// The fastest integer path this build compiled and this CPU runs: avx512 with VNNI, else avx2, else scalar.
Isa best_int_isa() noexcept;
// Whether `isa` runs the integer rows here: avx512 needs AVX512-BW and VNNI besides AVX-512F.
bool int_supported(Isa) noexcept;

// out[t * ldo + r] for each token t < tokens whose bit is set in `mask` and rows r in [row0, row1). Rows are
// independent, so any split over threads gives the same bits. Throws std::invalid_argument for a geometry the order
// cannot run or an ISA this CPU does not support.
void dot_rows_int(Isa, const IntOrder&, const IntMatrix&, const host_nvfp4::ScaleDecoder&, const IntActs&, int tokens,
                  std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out, std::size_t ldo);

// Declared activation quantizers, one token's n values (n a multiple of the block): per block, amax = max |x|.
// E2M1 x 2 (W4A4): scale byte = e4m3_encode(amax * global / 6) (product, then quotient, each rounded), decoded = its
// value; value_i = 2 * e2m1(e2m1_encode(x_i * global / decoded)) (0 when decoded is 0); scales[b] = decoded / global.
// `global` is crit::nvfp4's divisor. The 2 of the codes is the arithmetic's: put 0.5 per E2M1 side in post.
void quantize_e2m1x2(const float* x, std::size_t n, float global, std::int8_t* values, float* scales,
                     std::uint8_t* scale_bytes = nullptr);
// INT8 (W4A8): d = amax / 127 (rounded); value_i = round-half-even(x_i / d), clamped to +-127 (0 when d is 0);
// scales[b] = d.
void quantize_int8(const float* x, std::size_t n, int block, std::int8_t* values, float* scales);

// IEEE-only exp, SiLU and SwiGLU: the same bits wherever an IEEE FP32 FMA runs (std::fma here, __fmaf_rn on the
// device, host_nvfp4_device.cuh's exp_declared). exp: n = rint(x * log2(e)), r = x - n * ln2 (two FMAs, hi and lo
// parts), a degree-7 polynomial in r by FMAs, scaled by 2^n through the exponent bits; x clamped to [-87, 88].
float exp_declared(float x) noexcept;
float silu_declared(float x) noexcept;                 // x / (1 + exp(-x)), each step rounded
float swiglu_declared(float gate, float up) noexcept;  // silu(gate) * up

namespace detail {
// One implementation per ISA. `scales` holds row r's decoded weight block scales at (r - row0) * (cols / 16), or
// `table` is set and they are table[matrix.scales byte].
using IntRowsFn = void (*)(const IntOrder&, const IntMatrix&, const float* scales, const float* table, const IntActs&,
                           int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out,
                           std::size_t ldo);
void int_rows_scalar(const IntOrder&, const IntMatrix&, const float*, const float*, const IntActs&, int, std::uint32_t,
                     std::size_t, std::size_t, float, float*, std::size_t);
void int_rows_avx2(const IntOrder&, const IntMatrix&, const float*, const float*, const IntActs&, int, std::uint32_t,
                   std::size_t, std::size_t, float, float*, std::size_t);
void int_rows_avx512(const IntOrder&, const IntMatrix&, const float*, const float*, const IntActs&, int, std::uint32_t,
                     std::size_t, std::size_t, float, float*, std::size_t);
// The integer weight values by code (16 entries, int8).
const std::int8_t* weight_values(Weights) noexcept;
}  // namespace detail

}  // namespace crit::techniques::host_int
