// Integer block rows: the scalar oracle, dispatch, the declared activation quantizers and the declared exp, SiLU and
// SwiGLU. Compile without FP contraction (-ffp-contract=off): every multiply and add here is one declared FP32
// operation, and every FMA is written as std::fma.
#include "host_int.h"

#include "nvfp4.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace crit::techniques::host_int {
namespace {

constexpr std::int8_t kE2m1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
constexpr std::int8_t kInt4[16] = {0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1};

std::size_t code_stride(const IntMatrix& m) { return m.code_stride ? m.code_stride : row_bytes(m.layout, m.cols); }

// The code of element k of a row in either layout.
inline unsigned code_at(const std::uint8_t* codes, std::size_t k, Layout layout) {
    if(layout == Layout::rows) {
        const std::uint8_t byte = codes[k / 2];
        return k % 2 ? byte >> 4 : byte & 15u;
    }
    const std::size_t group = k / 256, block = (k % 256) / 16, e = k % 16, chunk = e / 4;
    const std::uint8_t byte = codes[group * 128 + chunk * 32 + 4 * (block % 8) + e % 4];
    return block < 8 ? byte & 15u : byte >> 4;
}
std::size_t scale_stride(const IntMatrix& m) { return m.scale_stride ? m.scale_stride : m.cols / kWeightBlock; }

float reduce(float* acc, const IntOrder& order) {
    const int L = order.lanes;
    if(order.reduce == host_nvfp4::Reduce::halving) {
        for(int w = L / 2; w >= 1; w /= 2)
            for(int i = 0; i < w; ++i) acc[i] = acc[i] + acc[i + w];
        return acc[0];
    }
    float sum = acc[0];
    for(int i = 1; i < L; ++i) sum = sum + acc[i];
    return sum;
}

// e2m1_encode's nearest-even rounding of a magnitude at most 6, as a code (crit::nvfp4's, without the throw): the
// midpoints between the E2M1 magnitudes, a tie going to the even code (the device's e2m1_encode_device). The neighbors'
// distances that crit::nvfp4's search compares are exact, so both choose the same code.
std::uint8_t e2m1_code(float x) {
    if(!(x == x)) return 0;
    constexpr float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    const float m = std::fmin(std::fabs(x), 6.0f);
    unsigned code = 0;
    while(code < 7 && (m > mids[code] || (m == mids[code] && code % 2 == 1))) ++code;
    return std::uint8_t(code | (x < 0.0f && code ? 8u : 0u));
}

// crit::nvfp4::e4m3_encode of a finite value: its linear search over the 127 codes keeps the code at the least FP32
// distance (ties to the even code); the nearest is one of the two ascending values around the magnitude, and a binary
// search finds them, so the code is the same at a fraction of the work (one search a block, not 127 decodes).
std::uint8_t e4m3_code(float x) {
    struct Values {
        float v[127];
        Values() {
            for(int i = 0; i < 127; ++i) v[i] = crit::nvfp4::e4m3(std::uint8_t(i));
        }
    };
    static const Values table;
    const float m = std::fmin(std::fabs(x), 448.0f);
    int lo = 0, hi = 127;   // the first index whose value is >= m
    while(lo < hi) {
        const int mid = (lo + hi) / 2;
        if(table.v[mid] < m) lo = mid + 1;
        else hi = mid;
    }
    const int h = lo < 1 ? 1 : lo > 126 ? 126 : lo, l = h - 1;
    const float dl = std::fabs(m - table.v[l]), dh = std::fabs(m - table.v[h]);
    const int best = (dh < dl || (dh == dl && h % 2 == 0)) ? h : l;
    return std::uint8_t(best | (x < 0.0f && best ? 0x80 : 0));
}

}  // namespace

namespace detail {

const std::int8_t* weight_values(Weights w) noexcept { return w == Weights::int4 ? kInt4 : kE2m1x2; }

void int_rows_scalar(const IntOrder& order, const IntMatrix& m, const float* scales, const float* table,
                     const IntActs& a, int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post,
                     float* out, std::size_t ldo) {
    const std::int8_t* values = weight_values(m.weights);
    const std::size_t blocks = m.cols / kWeightBlock, ratio = std::size_t(a.block) / kWeightBlock;
    float acc[32];
    for(std::size_t r = row0; r < row1; ++r) {
        const std::uint8_t* codes = m.codes + r * code_stride(m);
        const std::uint8_t* sb = m.scales ? m.scales + r * scale_stride(m) : nullptr;
        const float* s = table ? nullptr : scales + (r - row0) * blocks;
        for(int t = 0; t < tokens; ++t) {
            if(!(mask & (1u << t))) continue;
            const std::int8_t* x = a.values + std::size_t(t) * a.ld;
            const float* xs = a.scales + std::size_t(t) * a.lds;
            for(int j = 0; j < order.lanes; ++j) acc[j] = 0.0f;
            for(std::size_t b = 0; b < blocks; ++b) {
                std::int32_t sum = 0;
                for(std::size_t e = 0; e < kWeightBlock; ++e)
                    sum += std::int32_t(values[code_at(codes, b * kWeightBlock + e, m.layout)]) * x[b * kWeightBlock + e];
                const float p = (table ? table[sb[b]] : s[b]) * xs[b / ratio];
                acc[b % std::size_t(order.lanes)] = std::fma(float(sum), p, acc[b % std::size_t(order.lanes)]);
            }
            out[std::size_t(t) * ldo + r] = reduce(acc, order) * post;
        }
    }
}

}  // namespace detail

std::size_t row_bytes(Layout layout, std::size_t cols) noexcept {
    return layout == Layout::rows ? cols / 2 : (cols + 255) / 256 * 128;
}

void interleave_row(const std::uint8_t* row, std::size_t cols, std::uint8_t* out) {
    std::memset(out, 0, row_bytes(Layout::interleaved, cols));
    for(std::size_t k = 0; k < cols; ++k) {
        const std::uint8_t byte = row[k / 2];
        const unsigned code = k % 2 ? byte >> 4 : byte & 15u;
        const std::size_t group = k / 256, block = (k % 256) / 16, e = k % 16, chunk = e / 4;
        std::uint8_t& target = out[group * 128 + chunk * 32 + 4 * (block % 8) + e % 4];
        target = std::uint8_t(block < 8 ? (target & 0xF0u) | code : (target & 0x0Fu) | (code << 4));
    }
}

bool int_supported(Isa isa) noexcept {
    if(isa == Isa::scalar) return true;
#if defined(CRIT_HOST_ISA_FILES)
    if(isa == Isa::avx2) return host_isa::supported(Isa::avx2);
    if(isa == Isa::avx512) return host_isa::supported(Isa::avx512) && host_isa::avx512_vnni();
#endif
    return false;
}

Isa best_int_isa() noexcept {
    if(int_supported(Isa::avx512)) return Isa::avx512;
    if(int_supported(Isa::avx2)) return Isa::avx2;
    return Isa::scalar;
}

void dot_rows_int(Isa isa, const IntOrder& order, const IntMatrix& m, const host_nvfp4::ScaleDecoder& decoder,
                  const IntActs& a, int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post,
                  float* out, std::size_t ldo) {
    if((order.lanes != 16 && order.lanes != 32) || !m.codes || !a.values || !a.scales || !out || !m.cols ||
       m.cols % 64 || (a.block != 16 && a.block != 32) || a.ld < m.cols || a.lds < m.cols / std::size_t(a.block) ||
       tokens < 1 || tokens > kMaxTokens || row0 > row1 || row1 > m.rows || ldo < m.rows ||
       (!decoder.decode && !decoder.byte_table))
        throw std::invalid_argument("host_int: geometry, order, activations or token count");
    if(isa == Isa::portable) isa = Isa::scalar;   // no portable integer path: the scalar one is every CPU's
    if(!int_supported(isa))
        throw std::invalid_argument(std::string("host_int: ") + host_isa::isa_name(isa) + " not supported here");
    detail::IntRowsFn fn = detail::int_rows_scalar;
#if defined(CRIT_HOST_ISA_FILES)
    if(isa == Isa::avx512) fn = detail::int_rows_avx512;
    if(isa == Isa::avx2) fn = detail::int_rows_avx2;
#endif
    if(decoder.byte_table) {
        if(!m.scales) throw std::invalid_argument("host_int: a byte-table decoder needs matrix.scales");
        fn(order, m, nullptr, decoder.byte_table, a, tokens, mask, row0, row1, post, out, ldo);
        return;
    }
    constexpr std::size_t kChunk = 16;
    thread_local std::vector<float> scales;
    const std::size_t blocks = m.cols / kWeightBlock;
    scales.resize(kChunk * blocks);
    for(std::size_t r0 = row0; r0 < row1; r0 += kChunk) {
        const std::size_t r1 = r0 + kChunk < row1 ? r0 + kChunk : row1;
        for(std::size_t r = r0; r < r1; ++r) decoder.decode(decoder.context, r, scales.data() + (r - r0) * blocks);
        fn(order, m, scales.data(), nullptr, a, tokens, mask, r0, r1, post, out, ldo);
    }
}

void quantize_e2m1x2(const float* x, std::size_t n, float global, std::int8_t* values, float* scales,
                     std::uint8_t* scale_bytes) {
    if(n % 16 || !(global > 0.0f) || !std::isfinite(global)) throw std::invalid_argument("host_int: e2m1 quantizer");
    for(std::size_t b = 0; b < n / 16; ++b) {
        const float* v = x + 16 * b;
        float amax = 0.0f;
        for(int i = 0; i < 16; ++i) amax = std::fmax(amax, std::fabs(v[i]));
        const float wanted = (amax * global) / 6.0f;
        const std::uint8_t byte = e4m3_code(std::isfinite(wanted) ? wanted : 448.0f);
        const float decoded = crit::nvfp4::e4m3(byte);
        for(int i = 0; i < 16; ++i)
            values[16 * b + std::size_t(i)] =
                decoded > 0.0f ? kE2m1x2[e2m1_code((v[i] * global) / decoded)] : std::int8_t(0);
        scales[b] = decoded / global;
        if(scale_bytes) scale_bytes[b] = byte;
    }
}

void quantize_int8(const float* x, std::size_t n, int block, std::int8_t* values, float* scales) {
    if((block != 16 && block != 32) || n % std::size_t(block)) throw std::invalid_argument("host_int: int8 quantizer");
    for(std::size_t b = 0; b < n / std::size_t(block); ++b) {
        const float* v = x + std::size_t(block) * b;
        float amax = 0.0f;
        for(int i = 0; i < block; ++i) amax = std::fmax(amax, std::fabs(v[i]));
        const float d = amax / 127.0f;
        for(int i = 0; i < block; ++i) {
            float q = d > 0.0f ? std::nearbyint(v[i] / d) : 0.0f;
            q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q;
            values[std::size_t(block) * b + std::size_t(i)] = std::int8_t(q);
        }
        scales[b] = d;
    }
}

float exp_declared(float x) noexcept {
    // Cody-Waite reduction by ln 2 (hi part exact in FP32 times any n in range), then a degree-7 Taylor polynomial in
    // Horner form on |r| <= ln2 / 2, then 2^n by the exponent bits (n in [-126, 127] after the clamp).
    x = x > 88.0f ? 88.0f : x < -87.0f ? -87.0f : x;
    const float n = std::nearbyint(x * 1.44269504f);
    float r = std::fma(n, -0.693145752f, x);
    r = std::fma(n, -1.42860677e-06f, r);
    float p = 1.98412698e-04f;
    p = std::fma(p, r, 1.38888889e-03f);
    p = std::fma(p, r, 8.33333333e-03f);
    p = std::fma(p, r, 4.16666667e-02f);
    p = std::fma(p, r, 1.66666667e-01f);
    p = std::fma(p, r, 0.5f);
    p = std::fma(p, r, 1.0f);
    p = std::fma(p, r, 1.0f);
    const std::uint32_t bits = std::uint32_t(int(n) + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, 4);
    return p * scale;
}

float silu_declared(float x) noexcept { return x / (1.0f + exp_declared(-x)); }

float swiglu_declared(float gate, float up) noexcept { return silu_declared(gate) * up; }

}  // namespace crit::techniques::host_int
