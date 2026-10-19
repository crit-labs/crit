// Scalar oracle, run-time ISA selection and the scale decode step. Compile without FP contraction
// (-ffp-contract=off): every multiply and add here is one declared FP32 operation.
#include "host_nvfp4.h"

#include "nvfp4.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace crit::techniques::host_nvfp4 {
namespace {

std::size_t code_stride(const MatrixView& m) { return m.code_stride ? m.code_stride : m.cols / 2; }
std::size_t scale_stride(const MatrixView& m) { return m.scale_stride ? m.scale_stride : m.cols / kBlock; }

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

void decode_e4m3(const void* context, std::size_t row, float* out) {
    const auto& m = *static_cast<const MatrixView*>(context);
    const std::uint8_t* s = m.scales + row * scale_stride(m);
    const auto& table = e4m3_table().value;
    for(std::size_t b = 0; b < m.cols / kBlock; ++b) out[b] = table[s[b]];
}

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
void cpuid(int leaf, int sub, unsigned r[4]) {
    int v[4];
    __cpuidex(v, leaf, sub);
    for(int i = 0; i < 4; ++i) r[i] = unsigned(v[i]);
}
unsigned long long xcr0() { return _xgetbv(0); }
#define CRIT_HOST_NVFP4_X86 1
#elif defined(__x86_64__) || defined(__i386__)
void cpuid(int leaf, int sub, unsigned r[4]) { __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]); }
unsigned long long xcr0() {
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
}
#define CRIT_HOST_NVFP4_X86 1
#endif

struct Features {
    bool avx2 = false, avx512 = false;
    Features() {
#ifdef CRIT_HOST_NVFP4_X86
        unsigned r[4];
        cpuid(0, 0, r);
        const unsigned max_leaf = r[0];
        cpuid(1, 0, r);
        const bool osxsave = r[2] & (1u << 27), avx = r[2] & (1u << 28), fma = r[2] & (1u << 12);
        if(!osxsave || !avx || max_leaf < 7) return;
        const unsigned long long x = xcr0();
        const bool ymm = (x & 0x6) == 0x6, zmm = (x & 0xe6) == 0xe6;  // SSE+AVX state; plus opmask and ZMM
        cpuid(7, 0, r);
        avx2 = ymm && fma && (r[1] & (1u << 5));
        avx512 = zmm && fma && (r[1] & (1u << 16));
#endif
    }
};
const Features& features() {
    static const Features f;
    return f;
}

}  // namespace

ScaleDecoder e4m3_scales(const MatrixView& matrix) { return {decode_e4m3, &matrix, e4m3_table().value.data()}; }

const char* isa_name(Isa isa) noexcept {
    switch(isa) {
        case Isa::avx512: return "avx512";
        case Isa::avx2: return "avx2";
        default: return "scalar";
    }
}

bool supported(Isa isa) noexcept {
    switch(isa) {
#if defined(CRIT_HOST_NVFP4_AVX512)
        case Isa::avx512: return features().avx512;
#endif
#if defined(CRIT_HOST_NVFP4_AVX2)
        case Isa::avx2: return features().avx2;
#endif
        case Isa::scalar: return true;
        default: return false;
    }
}

Isa best_isa() noexcept {
    if(supported(Isa::avx512)) return Isa::avx512;
    if(supported(Isa::avx2)) return Isa::avx2;
    return Isa::scalar;
}

namespace detail {
// The oracle: one lane at a time, in the declared order.
void rows_scalar(const float*, const Order& order, const MatrixView& m, const float* scales, const float* table,
                 const float*, const float* x, std::size_t ldx,
                 int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out,
                 std::size_t ldo) {
    const std::size_t L = std::size_t(order.lanes), groups = m.cols / kBlock;
    float acc[32];
    for(std::size_t r = row0; r < row1; ++r) {
        const std::uint8_t* codes = m.codes + r * code_stride(m);
        const float* s = table ? nullptr : scales + (r - row0) * groups;
        const std::uint8_t* sb = m.scales + r * scale_stride(m);
        for(int t = 0; t < tokens; ++t) {
            if(!(mask & (1u << t))) continue;
            const float* xt = x + std::size_t(t) * ldx;
            for(std::size_t j = 0; j < L; ++j) acc[j] = 0.0f;
            auto code = [&](std::size_t k) {
                const std::uint8_t byte = codes[k / 2];
                return crit::nvfp4::e2m1(std::uint8_t(k % 2 ? byte >> 4 : byte & 15));
            };
            if(order.product == Product::block_dot) {
                for(std::size_t b = 0; b < groups; ++b) {
                    float d = 0.0f;
                    for(std::size_t e = 0; e < kBlock; ++e) d = std::fma(code(b * kBlock + e), xt[b * kBlock + e], d);
                    acc[b % L] = std::fma(d, table ? table[sb[b]] : s[b], acc[b % L]);
                }
            } else {
                for(std::size_t k = 0; k < m.cols; ++k) {
                    const float ws = code(k) * (table ? table[sb[k / kBlock]] : s[k / kBlock]);
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
}  // namespace detail

void dot_rows(Isa isa, const Order& order, const MatrixView& m, const ScaleDecoder& decoder, const float* x,
              std::size_t ldx, int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post,
              float* out, std::size_t ldo) {
    const std::size_t multiple = order.product == Product::block_dot ? kBlock : std::size_t(order.lanes);
    if((order.lanes != 16 && order.lanes != 32) || !m.codes || !decoder.decode || !x || !out || !m.cols ||
       m.cols % multiple || ldx < m.cols || tokens < 1 || tokens > kMaxTokens || row0 > row1 ||
       row1 > m.rows || ldo < m.rows)
        throw std::invalid_argument("host_nvfp4: geometry, order or token count");
    if(!supported(isa)) throw std::invalid_argument(std::string("host_nvfp4: ") + isa_name(isa) + " not supported here");
    detail::RowsFn fn = detail::rows_scalar;
#if defined(CRIT_HOST_NVFP4_AVX512)
    if(isa == Isa::avx512) fn = detail::rows_avx512;
#endif
#if defined(CRIT_HOST_NVFP4_AVX2)
    if(isa == Isa::avx2) fn = detail::rows_avx2;
#endif
    // block_dot paths vectorize across blocks: activations transposed once per call.
    thread_local std::vector<float> transposed;
    const float* xt = nullptr;
    if(order.product == Product::block_dot) {
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
        fn(e2m1_table().code, order, m, nullptr, decoder.byte_table, xt, x, ldx, tokens, mask, row0, row1, post, out, ldo);
        return;
    }
    // Scales are decoded for a chunk of rows at a time, then the chunk runs on the selected ISA.
    constexpr std::size_t kChunk = 16;
    thread_local std::vector<float> scales;
    const std::size_t groups = m.cols / kBlock;
    scales.resize(kChunk * groups);
    for(std::size_t r0 = row0; r0 < row1; r0 += kChunk) {
        const std::size_t r1 = r0 + kChunk < row1 ? r0 + kChunk : row1;
        for(std::size_t r = r0; r < r1; ++r) decoder.decode(decoder.context, r, scales.data() + (r - r0) * groups);
        fn(e2m1_table().code, order, m, scales.data(), nullptr, xt, x, ldx, tokens, mask, r0, r1, post, out, ldo);
    }
}

}  // namespace crit::techniques::host_nvfp4
