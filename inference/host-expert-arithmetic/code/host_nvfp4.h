// Host arithmetic for NVFP4 block-scaled rows that reproduces a device kernel bit for bit.
//
// A row dot is defined by a declared operation order, not by "the same math": element k of the row goes to
// accumulator lane k % lanes; each term is fma(e2m1(code) * scale[k / 16], x[k], lane) with the product rounded
// to FP32 first; blocks run in order; the lanes are reduced by a halving tree ((i, i + lanes/2), then /4, ...,
// leaving lane 0) or sequentially (lane 0 + lane 1 + ...); the sum is multiplied by `post`. A device kernel that
// declares the same order (host_nvfp4_device.cuh is one) gives the same bits, so an expert's contribution does
// not depend on which processor computed it. The scalar path is the oracle; the AVX2 and AVX-512 paths are
// chosen at run time from the CPU's features and match it exactly.
//
// A second product form, block_dot, is the other common GEMV order: per block b, d = sequential
// fma(e2m1(code), x, d) over its 16 elements from 0, then lane = fma(d, scale[b], lane) with block b going to
// lane b % lanes; the same reductions and post multiply follow.
//
// Weights use crit's NVFP4 layout (inference/quantized-projections, crit::nvfp4): codes [rows][cols / 2], low
// nibble = even element, one 8-bit block scale per 16 values. Block scales go through a decode hook, so
// checkpoint-specific scale encodings stay in the engine; the standard hook reads E4M3 bytes.
#pragma once
#include "host_isa.h"

#include <cstddef>
#include <cstdint>

namespace crit::techniques::host_nvfp4 {

constexpr std::size_t kBlock = 16;
constexpr int kMaxTokens = 8;

enum class Reduce { halving, sequential };
enum class Product { scaled_elements, block_dot };
struct Order {
    int lanes = 16;  // 16 or 32; with scaled_elements cols must be a multiple of it
    Reduce reduce = Reduce::halving;
    Product product = Product::scaled_elements;
};

// A view of one matrix. Strides in bytes; defaults (0) are cols / 2 and cols / 16.
struct MatrixView {
    const std::uint8_t* codes = nullptr;
    const std::uint8_t* scales = nullptr;
    std::size_t rows = 0, cols = 0;
    std::size_t code_stride = 0, scale_stride = 0;
};

// Writes the cols / 16 block scales of `row` as the exact floats the device multiplies by.
// A decoder that is a pure byte -> float table also sets byte_table; the kernels then look scales up inline
// instead of decoding each row first (same floats, so the same bits).
struct ScaleDecoder {
    void (*decode)(const void* context, std::size_t row, float* out) = nullptr;
    const void* context = nullptr;
    const float* byte_table = nullptr;  // optional [256], indexed by matrix.scales bytes
};
// The standard decoder: E4M3FN bytes of `matrix.scales` through crit::nvfp4::e4m3. `matrix` must outlive it.
ScaleDecoder e4m3_scales(const MatrixView& matrix);

// The ISA files this build compiled in and the CPU and the OS report (host_isa.h).
using Isa = host_isa::Isa;
using host_isa::best_isa;
using host_isa::isa_name;
using host_isa::supported;

// out[t * ldo + r] for every token t < tokens whose bit is set in `mask` and rows r in [row0, row1): the declared
// dot of row r with x[t * ldx ...], times post. Other tokens' outputs are left unwritten. Rows are independent,
// so any split over threads gives the same bits. Throws std::invalid_argument for a geometry or order the
// selected ISA cannot run, and for an ISA this CPU does not support.
void dot_rows(Isa, const Order&, const MatrixView&, const ScaleDecoder&, const float* x, std::size_t ldx, int tokens,
              std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out, std::size_t ldo);

namespace detail {
// One implementation per ISA; `scales` holds the decoded block scales of row r at r - row0, or `table` is set and
// block scales are table[matrix.scales byte]; for block_dot, `xt` holds the activations transposed as
// xt[(t * 16 + e) * blocks_padded + b] = x[t][16 b + e] with blocks_padded = round_up(cols / 16, 16); `e2m1` holds the 16
// code values (passed in, so ISA files compiled with other flags share no inline code with the rest).
using RowsFn = void (*)(const float* e2m1, const Order&, const MatrixView&, const float* scales, const float* table,
                        const float* xt, const float* x,
                        std::size_t ldx,
                        int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out,
                        std::size_t ldo);
void rows_scalar(const float*, const Order&, const MatrixView&, const float*, const float*, const float*, const float*, std::size_t, int, std::uint32_t,
                 std::size_t, std::size_t, float, float*, std::size_t);
void rows_avx2(const float*, const Order&, const MatrixView&, const float*, const float*, const float*, const float*, std::size_t, int, std::uint32_t,
               std::size_t, std::size_t, float, float*, std::size_t);
void rows_avx512(const float*, const Order&, const MatrixView&, const float*, const float*, const float*, const float*, std::size_t, int, std::uint32_t,
                 std::size_t, std::size_t, float, float*, std::size_t);
}  // namespace detail

}  // namespace crit::techniques::host_nvfp4
