// Integer block rows on AVX-512 with VNNI (vpdpbusd): compiled with -mavx512f -mavx512bw -mavx512vnni and chosen at
// run time (host_int::int_supported). 64 weight codes decode once to signed integers by a byte shuffle; per token one
// masked negation of the activations and one vpdpbusd give 4-element partial sums, two in-lane adds give each 16-block's
// exact sum, and 16 block sums at a time go through the declared FP32 step: lane b % 16 = fma(sum, wscale * ascale,
// lane). Integer sums are exact in any order, and the FP32 operations are the oracle's, so the bits are the oracle's.
// A group's 16 weight scales come by one gather from the byte table (or one load of decoded scales), and a row's
// lanes reduce by the halving tree in registers: the same pairs added in the same order as the oracle's loop.
//
// The declared quantizer and the SwiGLU run here too, 16 values at a time: every FP32 operation is the scalar
// function's (IEEE multiply, divide, add, FMA, round to nearest even), so each lane holds the scalar result.
#include "host_int.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace crit::techniques::host_int::detail {
namespace {

constexpr int kChunk = 64;           // elements per vector
constexpr int kGroup = 16;           // blocks per FP32 step: one lane each

// 64 codes (32 bytes) in element order as byte indices 0..15: each byte widened to 16 bits holds its low nibble in
// the low byte once the byte is also shifted up by 4 (the high nibble lands in the high byte).
inline __m512i nibbles(const std::uint8_t* p) {
    const __m512i w = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
    return _mm512_and_si512(_mm512_or_si512(w, _mm512_slli_epi16(w, 4)), _mm512_set1_epi8(15));
}

// The weight scales of the `live` blocks of a group: the byte table's values of the group's scale bytes (a masked
// load: nothing past the row's bytes is read) or the decoded scales themselves; the other lanes are zero.
inline __m512 block_scales(const float* table, const std::uint8_t* bytes, const float* decoded, __mmask16 live) {
    if(!table) return _mm512_maskz_loadu_ps(live, decoded);
    const __m512i index = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_maskz_loadu_epi8(__mmask64(live), bytes)));
    return _mm512_mask_i32gather_ps(_mm512_setzero_ps(), live, index, table, 4);
}

// A row's lanes to its value: host_nvfp4's halving tree (lanes[i] + lanes[i + w] for w = L/2 .. 1, the same pairs in
// the same order in registers) or the sequential sum.
inline float reduce_lanes(__m512 a0, __m512 a1, const IntOrder& order) {
    const int L = order.lanes;
    if(order.reduce == host_nvfp4::Reduce::halving) {
        const __m512 v = L == 32 ? _mm512_add_ps(a0, a1) : a0;
        const __m256 h = _mm256_add_ps(_mm512_castps512_ps256(v),
                                       _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1)));
        __m128 q = _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps(h, 1));
        q = _mm_add_ps(q, _mm_movehl_ps(q, q));
        return _mm_cvtss_f32(_mm_add_ss(q, _mm_shuffle_ps(q, q, 1)));
    }
    alignas(64) float lanes[32];
    _mm512_store_ps(lanes, a0);
    _mm512_store_ps(lanes + 16, a1);
    float sum = lanes[0];
    for(int i = 1; i < L; ++i) sum = sum + lanes[i];
    return sum;
}

// exp_declared on 16 values: the scalar function's operations in its order.
inline __m512 exp_declared_16(__m512 x) {
    x = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(x, _mm512_set1_ps(88.0f), _CMP_GT_OQ), x, _mm512_set1_ps(88.0f));
    x = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(x, _mm512_set1_ps(-87.0f), _CMP_LT_OQ), x, _mm512_set1_ps(-87.0f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504f)),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fmadd_ps(n, _mm512_set1_ps(-0.693145752f), x);
    r = _mm512_fmadd_ps(n, _mm512_set1_ps(-1.42860677e-06f), r);
    __m512 p = _mm512_set1_ps(1.98412698e-04f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.38888889e-03f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.33333333e-03f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.16666667e-02f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.66666667e-01f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    const __m512i bits = _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(p, _mm512_castsi512_ps(bits));
}

// Four chunks' vpdpbusd partials (chunk c holds blocks 4c..4c+3, block j in 128-bit lane j as four 4-element sums)
// reduced to the 16 blocks' sums in block order: a 4x4 transpose-add within each 128-bit lane leaves block 4c + j at
// position 4j + c, and one permutation puts it at 4c + j. Integer adds, exact in any order.
inline __m512i block_sums(__m512i v0, __m512i v1, __m512i v2, __m512i v3) {
    const __m512i t01 = _mm512_add_epi32(_mm512_unpacklo_epi32(v0, v1), _mm512_unpackhi_epi32(v0, v1));
    const __m512i t23 = _mm512_add_epi32(_mm512_unpacklo_epi32(v2, v3), _mm512_unpackhi_epi32(v2, v3));
    const __m512i s = _mm512_add_epi32(_mm512_unpacklo_epi64(t01, t23), _mm512_unpackhi_epi64(t01, t23));
    return _mm512_permutexvar_epi32(_mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), s);
}

// The activations of `tokens` tokens in the interleaved layout, offset by 128 to unsigned bytes for vpdpbusd: per group
// of 16 blocks, chunk c's lane i holds elements 4c..4c+3 of block i.
const std::uint8_t* offset_activations(const IntActs& a, std::size_t cols, int tokens, std::uint32_t mask) {
    thread_local std::vector<std::uint8_t> buffer;
    const std::size_t padded = (cols + 255) / 256 * 256;
    buffer.assign(std::size_t(tokens) * padded, 128u);
    for(int t = 0; t < tokens; ++t) {
        if(!(mask & (1u << t))) continue;
        const std::int8_t* x = a.values + std::size_t(t) * a.ld;
        std::uint8_t* o = buffer.data() + std::size_t(t) * padded;
        // Four bytes at a time: block i's elements 4c..4c+3 to chunk c's lane i; x + 128 is x ^ 0x80 in two's complement.
        for(std::size_t k = 0; k < cols; k += 4) {
            const std::size_t group = k / 256, block = (k % 256) / 16, chunk = (k % 16) / 4;
            std::uint32_t word;
            std::memcpy(&word, x + k, 4);
            word ^= 0x80808080u;
            std::memcpy(o + group * 256 + chunk * 64 + 4 * block, &word, 4);
        }
    }
    return buffer.data();
}

// Interleaved weights (host_int.h Layout::interleaved): one vector lane per block, so four vpdpbusd per token give the
// group's 16 exact block sums (offset activations, minus 128 x the block's weight sum computed once per row).
void interleaved_avx512(const IntOrder& order, const IntMatrix& m, const float* scales, const float* table,
                        const IntActs& a, int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post,
                        float* out, std::size_t ldo) {
    const std::int8_t* values = weight_values(m.weights);
    alignas(16) std::int8_t lut16[16];
    std::memcpy(lut16, values, 16);
    const __m512i lut = _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<const __m128i*>(lut16)));
    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i pairs = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const std::size_t blocks = m.cols / kWeightBlock, ratio = std::size_t(a.block) / kWeightBlock;
    const std::size_t cstride = m.code_stride ? m.code_stride : row_bytes(Layout::interleaved, m.cols);
    const std::size_t sstride = m.scale_stride ? m.scale_stride : blocks;
    const std::size_t padded = (m.cols + 255) / 256 * 256;
    const std::uint8_t* acts = offset_activations(a, m.cols, tokens, mask);
    const int L = order.lanes;
    for(std::size_t r = row0; r < row1; ++r) {
        const std::uint8_t* codes = m.codes + r * cstride;
        const std::uint8_t* sb = m.scales ? m.scales + r * sstride : nullptr;
        const float* ws = table ? nullptr : scales + (r - row0) * blocks;
        __m512 acc[kMaxTokens][2];
        for(int t = 0; t < tokens; ++t) acc[t][0] = acc[t][1] = _mm512_setzero_ps();
        for(std::size_t b0 = 0, g = 0; b0 < blocks; b0 += kGroup, ++g) {
            const int count = int(blocks - b0 < kGroup ? blocks - b0 : kGroup);
            __m512i w[4];
            __m512i weight_sum = _mm512_setzero_si512();
            for(int c = 0; c < 4; ++c) {
                const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + g * 128 + std::size_t(c) * 32));
                const __m256i lo = _mm256_and_si256(q, _mm256_set1_epi8(15));
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), _mm256_set1_epi8(15));
                w[c] = _mm512_shuffle_epi8(lut, _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
                weight_sum = _mm512_dpbusd_epi32(weight_sum, ones, w[c]);
            }
            const __m512i correction = _mm512_slli_epi32(weight_sum, 7);
            const __mmask16 live = count == kGroup ? __mmask16(0xffff) : __mmask16((1u << count) - 1u);
            const __m512 wv = block_scales(table, sb ? sb + b0 : nullptr, ws ? ws + b0 : nullptr, live);
            for(int t = 0; t < tokens; ++t) {
                if(!(mask & (1u << t))) continue;
                const std::uint8_t* x = acts + std::size_t(t) * padded + g * 256;
                // Four independent vpdpbusd (no chain through the accumulator), then integer adds: exact in any order.
                const __m512i zero = _mm512_setzero_si512();
                const __m512i s0 = _mm512_dpbusd_epi32(zero, _mm512_loadu_si512(x), w[0]);
                const __m512i s1 = _mm512_dpbusd_epi32(zero, _mm512_loadu_si512(x + 64), w[1]);
                const __m512i s2 = _mm512_dpbusd_epi32(zero, _mm512_loadu_si512(x + 128), w[2]);
                const __m512i s3 = _mm512_dpbusd_epi32(zero, _mm512_loadu_si512(x + 192), w[3]);
                const __m512i sum = _mm512_add_epi32(_mm512_add_epi32(s0, s1), _mm512_add_epi32(s2, s3));
                const __m512 s = _mm512_cvtepi32_ps(_mm512_sub_epi32(sum, correction));
                const float* xs = a.scales + std::size_t(t) * a.lds;
                const __m512 as = ratio == 1 ? _mm512_maskz_loadu_ps(live, xs + b0)
                                             : _mm512_permutexvar_ps(pairs, _mm512_maskz_loadu_ps(
                                                   __mmask16((1u << (count / 2)) - 1u), xs + b0 / 2));
                __m512& lane = acc[t][L == 32 ? g % 2 : 0];
                // A full group stores the whole vector: a masked store would not forward to the next group's load.
                lane = count == kGroup ? _mm512_fmadd_ps(s, _mm512_mul_ps(wv, as), lane)
                                       : _mm512_mask3_fmadd_ps(s, _mm512_mul_ps(wv, as), lane, live);
            }
        }
        for(int t = 0; t < tokens; ++t)
            if(mask & (1u << t)) out[std::size_t(t) * ldo + r] = reduce_lanes(acc[t][0], acc[t][1], order) * post;
    }
}

}  // namespace

void quantize_e2m1x2_avx512(const float* x, std::size_t n, float global, std::int8_t* values, float* scales,
                            std::uint8_t* scale_bytes) {
    const __m512 g = _mm512_set1_ps(global), six = _mm512_set1_ps(6.0f), zero = _mm512_setzero_ps();
    const __m512i one = _mm512_set1_epi32(1);
    const __m512i value_of = _mm512_setr_epi32(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    for(std::size_t b = 0; b < n / 16; ++b) {
        const __m512 v = _mm512_loadu_ps(x + 16 * b);
        // amax = fmax over |v|, from 0: a NaN is passed over.
        const __mmask16 numbers = _mm512_cmp_ps_mask(v, v, _CMP_ORD_Q);
        const float amax = _mm512_reduce_max_ps(_mm512_maskz_mov_ps(numbers, _mm512_abs_ps(v)));
        const float wanted = (amax * global) / 6.0f;
        float decoded;
        const std::uint8_t byte = e4m3_code(std::isfinite(wanted) ? wanted : 448.0f, &decoded);
        if(decoded > 0.0f) {
            const __m512 q = _mm512_div_ps(_mm512_mul_ps(v, g), _mm512_set1_ps(decoded));
            const __m512 m = _mm512_min_ps(_mm512_abs_ps(q), six);
            // The code is how many of the E2M1 magnitudes' midpoints the value is past: a tie goes to the even code,
            // so the odd midpoints (0.75, 1.75, 3.5: the next code is even) count at equality.
            __m512i code = _mm512_setzero_si512();
            const auto past = [&](__mmask16 k) { code = _mm512_mask_add_epi32(code, k, code, one); };
            past(_mm512_cmp_ps_mask(m, _mm512_set1_ps(0.25f), _CMP_GT_OQ));
            past(_mm512_cmp_ps_mask(m, _mm512_set1_ps(0.75f), _CMP_GE_OQ));
            past(_mm512_cmp_ps_mask(m, _mm512_set1_ps(1.25f), _CMP_GT_OQ));
            past(_mm512_cmp_ps_mask(m, _mm512_set1_ps(1.75f), _CMP_GE_OQ));
            past(_mm512_cmp_ps_mask(m, _mm512_set1_ps(2.5f), _CMP_GT_OQ));
            past(_mm512_cmp_ps_mask(m, _mm512_set1_ps(3.5f), _CMP_GE_OQ));
            past(_mm512_cmp_ps_mask(m, _mm512_set1_ps(5.0f), _CMP_GT_OQ));
            const __mmask16 negative = _mm512_cmp_ps_mask(q, zero, _CMP_LT_OQ) & _mm512_test_epi32_mask(code, code);
            code = _mm512_mask_or_epi32(code, negative, code, _mm512_set1_epi32(8));
            code = _mm512_maskz_mov_epi32(_mm512_cmp_ps_mask(q, q, _CMP_ORD_Q), code);   // a NaN quantizes to 0
            _mm_storeu_si128(reinterpret_cast<__m128i*>(values + 16 * b),
                             _mm512_cvtepi32_epi8(_mm512_permutexvar_epi32(code, value_of)));
        } else {
            std::memset(values + 16 * b, 0, 16);
        }
        scales[b] = decoded / global;
        if(scale_bytes) scale_bytes[b] = byte;
    }
}

void swiglu_rows_avx512(const float* gate, const float* up, std::size_t n, float* out) {
    const __m512i sign = _mm512_set1_epi32(int(0x80000000u));
    const __m512 one = _mm512_set1_ps(1.0f);
    for(std::size_t k = 0; k + 16 <= n; k += 16) {
        const __m512 x = _mm512_loadu_ps(gate + k);
        const __m512 minus = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(x), sign));
        const __m512 silu = _mm512_div_ps(x, _mm512_add_ps(one, exp_declared_16(minus)));
        _mm512_storeu_ps(out + k, _mm512_mul_ps(silu, _mm512_loadu_ps(up + k)));
    }
}

void int_rows_avx512(const IntOrder& order, const IntMatrix& m, const float* scales, const float* table, const IntActs& a,
                     int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out,
                     std::size_t ldo) {
    if(m.layout == Layout::interleaved) {
        interleaved_avx512(order, m, scales, table, a, tokens, mask, row0, row1, post, out, ldo);
        return;
    }
    const std::int8_t* values = weight_values(m.weights);
    alignas(16) std::int8_t lut16[16];
    std::memcpy(lut16, values, 16);
    const __m512i lut = _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<const __m128i*>(lut16)));
    const __m512i zero = _mm512_setzero_si512();
    const __m512i pairs = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);   // a 32-block scale twice
    const std::size_t blocks = m.cols / kWeightBlock, ratio = std::size_t(a.block) / kWeightBlock;
    const std::size_t cstride = m.code_stride ? m.code_stride : m.cols / 2;
    const std::size_t sstride = m.scale_stride ? m.scale_stride : blocks;
    const int L = order.lanes;
    for(std::size_t r = row0; r < row1; ++r) {
        const std::uint8_t* codes = m.codes + r * cstride;
        const std::uint8_t* sb = m.scales ? m.scales + r * sstride : nullptr;
        const float* ws = table ? nullptr : scales + (r - row0) * blocks;
        __m512 acc[kMaxTokens][2];
        for(int t = 0; t < tokens; ++t) acc[t][0] = acc[t][1] = _mm512_setzero_ps();
        for(std::size_t b0 = 0; b0 < blocks; b0 += kGroup) {
            const int count = int(blocks - b0 < kGroup ? blocks - b0 : kGroup);   // a multiple of 4
            const int chunks = count / 4;
            __m512i aw[4];
            __mmask64 neg[4];
            for(int c = 0; c < chunks; ++c) {
                const __m512i w = _mm512_shuffle_epi8(lut, nibbles(codes + (b0 * kWeightBlock + std::size_t(c) * kChunk) / 2));
                aw[c] = _mm512_abs_epi8(w);
                neg[c] = _mm512_movepi8_mask(w);
            }
            const __mmask16 live = count == kGroup ? __mmask16(0xffff) : __mmask16((1u << count) - 1u);
            const __m512 wv = block_scales(table, sb ? sb + b0 : nullptr, ws ? ws + b0 : nullptr, live);
            for(int t = 0; t < tokens; ++t) {
                if(!(mask & (1u << t))) continue;
                const std::int8_t* x = a.values + std::size_t(t) * a.ld + b0 * kWeightBlock;
                const float* xs = a.scales + std::size_t(t) * a.lds;
                __m512i partial[4];
                for(int c = 0; c < 4; ++c) {
                    if(c >= chunks) { partial[c] = zero; continue; }
                    const __m512i v = _mm512_loadu_si512(x + c * kChunk);
                    const __m512i signed_v = _mm512_mask_sub_epi8(v, neg[c], zero, v);
                    partial[c] = _mm512_dpbusd_epi32(zero, aw[c], signed_v);
                }
                const __m512 s = _mm512_cvtepi32_ps(block_sums(partial[0], partial[1], partial[2], partial[3]));
                // The activation scale of each block, then the product rounded as the oracle rounds it.
                const __m512 as = ratio == 1 ? _mm512_maskz_loadu_ps(live, xs + b0)
                                             : _mm512_permutexvar_ps(pairs, _mm512_maskz_loadu_ps(
                                                   __mmask16((1u << (count / 2)) - 1u), xs + b0 / 2));
                __m512& lane = acc[t][L == 32 ? (b0 / kGroup) % 2 : 0];
                lane = count == kGroup ? _mm512_fmadd_ps(s, _mm512_mul_ps(wv, as), lane)
                                       : _mm512_mask3_fmadd_ps(s, _mm512_mul_ps(wv, as), lane, live);   // off lanes keep theirs
            }
        }
        for(int t = 0; t < tokens; ++t)
            if(mask & (1u << t)) out[std::size_t(t) * ldo + r] = reduce_lanes(acc[t][0], acc[t][1], order) * post;
    }
}

}  // namespace crit::techniques::host_int::detail
