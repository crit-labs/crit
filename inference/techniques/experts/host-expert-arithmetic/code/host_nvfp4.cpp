// Scalar oracle, ISA dispatch and the scale decode step. Compile without FP contraction
// (-ffp-contract=off): every multiply and add here is one declared FP32 operation.
#include "host_nvfp4.h"

#include "nvfp4.h"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace crit::techniques::host_nvfp4 {
namespace {

std::size_t block_of(const MatrixView& m) { return std::size_t(m.format.block); }
std::size_t code_stride(const MatrixView& m) {
    return m.code_stride ? m.code_stride : m.cols * std::size_t(m.format.bits) / 8;
}
std::size_t scale_stride(const MatrixView& m) { return m.scale_stride ? m.scale_stride : m.cols / block_of(m); }

struct E4m3Table {
    std::array<float, 256> value{};
    E4m3Table() {
        for(int b = 0; b < 256; ++b) value[std::size_t(b)] = crit::nvfp4::e4m3(std::uint8_t(b));
    }
};
const E4m3Table& e4m3_table() {
    static const E4m3Table table;
    return table;
}
struct E2m1Table {
    float code[16];
    E2m1Table() {
        for(int c = 0; c < 16; ++c) code[c] = crit::nvfp4::e2m1(std::uint8_t(c));
    }
};
const E2m1Table& e2m1_table() {
    static const E2m1Table table;
    return table;
}

// OCP FP8 E5M2: 1 sign, 5 exponent (bias 15), 2 mantissa bits; exponent 31 is infinity (mantissa 0) or NaN.
float e5m2(std::uint8_t b) {
    const int e = (b >> 2) & 31, f = b & 3;
    const float sign = (b & 0x80) ? -1.0f : 1.0f;
    if(e == 31) return f ? std::numeric_limits<float>::quiet_NaN() : sign * std::numeric_limits<float>::infinity();
    return sign * (e ? std::ldexp(1.0f + float(f) / 4.0f, e - 15) : std::ldexp(float(f) / 4.0f, -14));
}
struct Tables {
    float int4[16], e4m3[256], e5m2[256], int8[256], e8m0[256];
    Tables() {
        for(int c = 0; c < 16; ++c) int4[c] = float(c < 8 ? c : c - 16);
        for(int b = 0; b < 256; ++b) {
            e4m3[b] = crit::nvfp4::e4m3(std::uint8_t(b));
            e5m2[b] = host_nvfp4::e5m2(std::uint8_t(b));
            int8[b] = float(b < 128 ? b : b - 256);
            e8m0[b] = b == 255 ? std::numeric_limits<float>::quiet_NaN() : std::ldexp(1.0f, b - 127);
        }
    }
};
const Tables& tables() {
    static const Tables t;
    return t;
}

void decode_e8m0(const void* context, std::size_t row, float* out) {
    const auto& m = *static_cast<const MatrixView*>(context);
    const std::uint8_t* s = m.scales + row * scale_stride(m);
    for(std::size_t b = 0; b < m.cols / block_of(m); ++b) out[b] = tables().e8m0[s[b]];
}

const float* values_of(const MatrixView& m) { return m.format.values ? m.format.values : e2m1_table().code; }

// Element k of a row: its code through the format's value table.
inline float element(const std::uint8_t* codes, std::size_t k, int bits, const float* values) {
    if(bits == 8) return values[codes[k]];
    const std::uint8_t byte = codes[k / 2];
    return values[k % 2 ? byte >> 4 : byte & 15];
}

void decode_e4m3(const void* context, std::size_t row, float* out) {
    const auto& m = *static_cast<const MatrixView*>(context);
    const std::uint8_t* s = m.scales + row * scale_stride(m);
    const auto& table = e4m3_table().value;
    for(std::size_t b = 0; b < m.cols / block_of(m); ++b) out[b] = table[s[b]];
}

}  // namespace

ScaleDecoder e4m3_scales(const MatrixView& matrix) { return {decode_e4m3, &matrix, e4m3_table().value.data()}; }
ScaleDecoder e8m0_scales(const MatrixView& matrix) { return {decode_e8m0, &matrix, tables().e8m0}; }

const float* e2m1_values() { return e2m1_table().code; }
const float* int4_values() { return tables().int4; }
const float* e4m3_values() { return tables().e4m3; }
const float* e5m2_values() { return tables().e5m2; }
const float* int8_values() { return tables().int8; }
Format nvfp4_format() { return {4, 16, e2m1_values()}; }
Format mxfp4_format() { return {4, 32, e2m1_values()}; }
Format mxfp8_e4m3_format() { return {8, 32, e4m3_values()}; }
Format mxfp8_e5m2_format() { return {8, 32, e5m2_values()}; }
Format mxint8_format() { return {8, 32, int8_values()}; }

namespace detail {
// The oracle: one lane at a time, in the declared order.
void rows_scalar(const float*, const Order& order, const MatrixView& m, const float* scales, const float* table,
                 const float*, const float* x, std::size_t ldx,
                 int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out,
                 std::size_t ldo) {
    const std::size_t L = std::size_t(order.lanes), B = block_of(m), groups = m.cols / B;
    const float* values = values_of(m);
    const int bits = m.format.bits;
    float acc[32];
    for(std::size_t r = row0; r < row1; ++r) {
        const std::uint8_t* codes = m.codes + r * code_stride(m);
        const float* s = table ? nullptr : scales + (r - row0) * groups;
        const std::uint8_t* sb = m.scales + r * scale_stride(m);
        for(int t = 0; t < tokens; ++t) {
            if(!(mask & (1u << t))) continue;
            const float* xt = x + std::size_t(t) * ldx;
            for(std::size_t j = 0; j < L; ++j) acc[j] = 0.0f;
            auto code = [&](std::size_t k) { return element(codes, k, bits, values); };
            if(order.product == Product::block_dot) {
                for(std::size_t b = 0; b < groups; ++b) {
                    float d = 0.0f;
                    for(std::size_t e = 0; e < B; ++e) d = std::fma(code(b * B + e), xt[b * B + e], d);
                    acc[b % L] = std::fma(d, table ? table[sb[b]] : s[b], acc[b % L]);
                }
            } else {
                for(std::size_t k = 0; k < m.cols; ++k) {
                    const float ws = code(k) * (table ? table[sb[k / B]] : s[k / B]);
                    acc[k % L] = std::fma(ws, xt[k], acc[k % L]);
                }
            }
            float sum;
            if(order.reduce == Reduce::halving) {
                for(std::size_t w = L / 2; w >= 1; w /= 2)
                    for(std::size_t i = 0; i < w; ++i) acc[i] = acc[i] + acc[i + w];
                sum = acc[0];
            } else {
                sum = acc[0];
                for(std::size_t i = 1; i < L; ++i) sum = sum + acc[i];
            }
            out[std::size_t(t) * ldo + r] = sum * post;
        }
    }
}

#define CRIT_PORTABLE_ROWS rows_portable
#include "host_nvfp4_portable.inc"
#undef CRIT_PORTABLE_ROWS
}  // namespace detail

void dot_rows(Isa isa, const Order& order, const MatrixView& m, const ScaleDecoder& decoder, const float* x,
              std::size_t ldx, int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post,
              float* out, std::size_t ldo) {
    const std::size_t B = block_of(m);
    const std::size_t multiple = order.product == Product::block_dot ? B : std::size_t(order.lanes);
    if((order.lanes != 16 && order.lanes != 32) || (m.format.bits != 4 && m.format.bits != 8) ||
       (B != 16 && B != 32) || !m.codes || !decoder.decode || !x || !out || !m.cols || m.cols % multiple ||
       m.cols % B || ldx < m.cols || tokens < 1 || tokens > kMaxTokens || row0 > row1 || row1 > m.rows ||
       ldo < m.rows)
        throw std::invalid_argument("host_nvfp4: geometry, format, order or token count");
    if(!supported(isa)) throw std::invalid_argument(std::string("host_nvfp4: ") + isa_name(isa) + " not supported here");
    // The hand-written x86 paths cover the NVFP4 layout; every other format runs the portable path there.
    const bool nvfp4_layout = m.format.bits == 4 && B == kBlock && values_of(m) == e2m1_table().code;
    detail::RowsFn fn = isa == Isa::scalar ? detail::rows_scalar : detail::rows_portable;
#if defined(CRIT_HOST_ISA_FILES)
    if(isa == Isa::avx512) fn = nvfp4_layout ? detail::rows_avx512 : detail::rows_portable_avx512;
    if(isa == Isa::avx2) fn = nvfp4_layout ? detail::rows_avx2 : detail::rows_portable_avx2;
#else
    (void)nvfp4_layout;
#endif
    // block_dot paths vectorize across blocks: activations transposed once per call.
    thread_local std::vector<float> transposed;
    const float* xt = nullptr;
    if(order.product == Product::block_dot && nvfp4_layout && (isa == Isa::avx2 || isa == Isa::avx512)) {
        const std::size_t blocks = m.cols / kBlock, padded = (blocks + 15) / 16 * 16;
        transposed.assign(std::size_t(tokens) * kBlock * padded, 0.0f);
        for(int t = 0; t < tokens; ++t)
            if(mask & (1u << t))
                for(std::size_t b = 0; b < blocks; ++b)
                    for(std::size_t e = 0; e < kBlock; ++e)
                        transposed[(std::size_t(t) * kBlock + e) * padded + b] = x[std::size_t(t) * ldx + b * kBlock + e];
        xt = transposed.data();
    }
    if(decoder.byte_table) {
        if(!m.scales) throw std::invalid_argument("host_nvfp4: a byte-table decoder needs matrix.scales");
        fn(values_of(m), order, m, nullptr, decoder.byte_table, xt, x, ldx, tokens, mask, row0, row1, post, out, ldo);
        return;
    }
    // Scales are decoded for a chunk of rows at a time, then the chunk runs on the selected ISA.
    constexpr std::size_t kChunk = 16;
    thread_local std::vector<float> scales;
    const std::size_t groups = m.cols / B;
    scales.resize(kChunk * groups);
    for(std::size_t r0 = row0; r0 < row1; r0 += kChunk) {
        const std::size_t r1 = r0 + kChunk < row1 ? r0 + kChunk : row1;
        for(std::size_t r = r0; r < r1; ++r) decoder.decode(decoder.context, r, scales.data() + (r - r0) * groups);
        fn(values_of(m), order, m, scales.data(), nullptr, xt, x, ldx, tokens, mask, r0, r1, post, out, ldo);
    }
}

}  // namespace crit::techniques::host_nvfp4
