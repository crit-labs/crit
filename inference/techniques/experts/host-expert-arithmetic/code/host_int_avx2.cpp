// Integer block rows on AVX2: compiled with -mavx2 -mfma and chosen at run time (host_int::int_supported). 32 weight
// codes decode to signed integers by a byte shuffle; per token, vpmaddubsw of the magnitudes with the sign-applied
// activations (pairs at most 2 x 12 x 127, no saturation), vpmaddwd by ones and two in-lane adds give each 16-block's
// exact sum, and 8 block sums at a time go through the declared FP32 step, as host_int_avx512.cpp does.
#include "host_int.h"

#include <immintrin.h>

#include <cstring>
#include <vector>

namespace crit::techniques::host_int::detail {
namespace {

constexpr int kGroup = 8;   // blocks per FP32 step

// 32 codes (16 bytes) in element order as byte indices 0..15.
inline __m256i nibbles(const std::uint8_t* p) {
    const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i lo = _mm_and_si128(q, _mm_set1_epi8(15));
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(15));
    return _mm256_setr_m128i(_mm_unpacklo_epi8(lo, hi), _mm_unpackhi_epi8(lo, hi));
}

// Four chunks' 4-element partials (chunk c holds blocks 2c and 2c + 1, block j in 128-bit lane j) reduced to the 8
// blocks' sums in block order: a 4x4 transpose-add within each 128-bit lane leaves block 2c + j at position 4j + c,
// and one permutation puts it at 2c + j.
inline __m256i block_sums(__m256i v0, __m256i v1, __m256i v2, __m256i v3) {
    const __m256i t01 = _mm256_add_epi32(_mm256_unpacklo_epi32(v0, v1), _mm256_unpackhi_epi32(v0, v1));
    const __m256i t23 = _mm256_add_epi32(_mm256_unpacklo_epi32(v2, v3), _mm256_unpackhi_epi32(v2, v3));
    const __m256i s = _mm256_add_epi32(_mm256_unpacklo_epi64(t01, t23), _mm256_unpackhi_epi64(t01, t23));
    return _mm256_permutevar8x32_epi32(s, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

// The activations in the interleaved layout, offset by 128 to unsigned bytes (host_int_avx512.cpp's arrangement).
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

// Interleaved weights: blocks 0..7 of a group in one vector's lanes (the codes' low nibbles), 8..15 in another (the
// high nibbles); vpmaddubsw and vpmaddwd per chunk accumulate exact block sums across the group's four chunks.
void interleaved_avx2(const IntOrder& order, const IntMatrix& m, const float* scales, const float* table,
                      const IntActs& a, int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post,
                      float* out, std::size_t ldo) {
    const std::int8_t* values = weight_values(m.weights);
    alignas(16) std::int8_t lut16[16];
    std::memcpy(lut16, values, 16);
    const __m128i lut128 = _mm_load_si128(reinterpret_cast<const __m128i*>(lut16));
    const __m256i lut = _mm256_setr_m128i(lut128, lut128);
    const __m256i ones16 = _mm256_set1_epi16(1), ones8 = _mm256_set1_epi8(1);
    const __m256i pairs = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    const std::size_t blocks = m.cols / kWeightBlock, ratio = std::size_t(a.block) / kWeightBlock;
    const std::size_t cstride = m.code_stride ? m.code_stride : row_bytes(Layout::interleaved, m.cols);
    const std::size_t sstride = m.scale_stride ? m.scale_stride : blocks;
    const std::size_t padded = (m.cols + 255) / 256 * 256;
    const std::uint8_t* acts = offset_activations(a, m.cols, tokens, mask);
    const int L = order.lanes;
    const __m256i all = _mm256_set1_epi32(-1), four = _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
    for(std::size_t r = row0; r < row1; ++r) {
        const std::uint8_t* codes = m.codes + r * cstride;
        const std::uint8_t* sb = m.scales ? m.scales + r * sstride : nullptr;
        const float* ws = table ? nullptr : scales + (r - row0) * blocks;
        __m256 acc[kMaxTokens][4];
        for(int t = 0; t < tokens; ++t)
            for(int j = 0; j < 4; ++j) acc[t][j] = _mm256_setzero_ps();
        for(std::size_t b0 = 0, g = 0; b0 < blocks; b0 += 16, ++g) {
            const int count = int(blocks - b0 < 16 ? blocks - b0 : 16);   // 4, 8, 12 or 16
            __m256i w[4][2], weight_sum[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
            for(int c = 0; c < 4; ++c) {
                const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + g * 128 + std::size_t(c) * 32));
                w[c][0] = _mm256_shuffle_epi8(lut, _mm256_and_si256(q, _mm256_set1_epi8(15)));
                w[c][1] = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(q, 4), _mm256_set1_epi8(15)));
                for(int h = 0; h < 2; ++h)
                    weight_sum[h] = _mm256_add_epi32(weight_sum[h],
                                                     _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, w[c][h]), ones16));
            }
            alignas(32) float wscale[16] = {};
            for(int i = 0; i < count; ++i) wscale[i] = table ? table[sb[b0 + std::size_t(i)]] : ws[b0 + std::size_t(i)];
            for(int t = 0; t < tokens; ++t) {
                if(!(mask & (1u << t))) continue;
                const std::uint8_t* x = acts + std::size_t(t) * padded + g * 256;
                const float* xs = a.scales + std::size_t(t) * a.lds;
                for(int h = 0; h < 2; ++h) {
                    const int lanes_here = count - 8 * h < 8 ? count - 8 * h : 8;
                    if(lanes_here <= 0) break;
                    __m256i sum = _mm256_setzero_si256();
                    for(int c = 0; c < 4; ++c) {
                        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + c * 64 + h * 32));
                        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_maddubs_epi16(v, w[c][h]), ones16));
                    }
                    const __m256 s = _mm256_cvtepi32_ps(_mm256_sub_epi32(sum, _mm256_slli_epi32(weight_sum[h], 7)));
                    const __m256i live = lanes_here == 8 ? all : four;
                    const std::size_t first = b0 + std::size_t(8 * h);
                    const __m256 as = ratio == 1 ? _mm256_maskload_ps(xs + first, live)
                                                 : _mm256_permutevar8x32_ps(_mm256_maskload_ps(xs + first / 2,
                                                       lanes_here == 8 ? four : _mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0)),
                                                       pairs);
                    __m256& lane = acc[t][(2 * g + std::size_t(h)) % std::size_t(L / 8)];
                    const __m256 next = _mm256_fmadd_ps(s, _mm256_mul_ps(_mm256_load_ps(wscale + 8 * h), as), lane);
                    lane = lanes_here == 8 ? next : _mm256_blend_ps(lane, next, 0x0f);
                }
            }
        }
        for(int t = 0; t < tokens; ++t) {
            if(!(mask & (1u << t))) continue;
            alignas(32) float lanes[32];
            for(int j = 0; j < 4; ++j) _mm256_store_ps(lanes + 8 * j, acc[t][j]);
            if(order.reduce == host_nvfp4::Reduce::halving) {
                for(int width = L / 2; width >= 1; width /= 2)
                    for(int i = 0; i < width; ++i) lanes[i] = lanes[i] + lanes[i + width];
                out[std::size_t(t) * ldo + r] = lanes[0] * post;
            } else {
                float total = lanes[0];
                for(int i = 1; i < L; ++i) total = total + lanes[i];
                out[std::size_t(t) * ldo + r] = total * post;
            }
        }
    }
}

}  // namespace

void int_rows_avx2(const IntOrder& order, const IntMatrix& m, const float* scales, const float* table, const IntActs& a,
                   int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out,
                   std::size_t ldo) {
    if(m.layout == Layout::interleaved) {
        interleaved_avx2(order, m, scales, table, a, tokens, mask, row0, row1, post, out, ldo);
        return;
    }
    const std::int8_t* values = weight_values(m.weights);
    alignas(16) std::int8_t lut16[16];
    std::memcpy(lut16, values, 16);
    const __m128i lut128 = _mm_load_si128(reinterpret_cast<const __m128i*>(lut16));
    const __m256i lut = _mm256_setr_m128i(lut128, lut128);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i pairs = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);   // a 32-block activation scale twice
    const std::size_t blocks = m.cols / kWeightBlock, ratio = std::size_t(a.block) / kWeightBlock;
    const std::size_t cstride = m.code_stride ? m.code_stride : m.cols / 2;
    const std::size_t sstride = m.scale_stride ? m.scale_stride : blocks;
    const int L = order.lanes;
    for(std::size_t r = row0; r < row1; ++r) {
        const std::uint8_t* codes = m.codes + r * cstride;
        const std::uint8_t* sb = m.scales ? m.scales + r * sstride : nullptr;
        const float* ws = table ? nullptr : scales + (r - row0) * blocks;
        __m256 acc[kMaxTokens][4];
        for(int t = 0; t < tokens; ++t)
            for(int j = 0; j < 4; ++j) acc[t][j] = _mm256_setzero_ps();
        for(std::size_t b0 = 0; b0 < blocks; b0 += kGroup) {
            const int count = int(blocks - b0 < kGroup ? blocks - b0 : kGroup);   // 4 or 8
            const int chunks = count / 2;
            __m256i aw[4], neg[4];
            for(int c = 0; c < chunks; ++c) {
                const __m256i w = _mm256_shuffle_epi8(lut, nibbles(codes + (b0 * kWeightBlock + std::size_t(c) * 32) / 2));
                aw[c] = _mm256_abs_epi8(w);
                neg[c] = w;
            }
            alignas(32) float wscale[kGroup] = {};
            for(int i = 0; i < count; ++i) wscale[i] = table ? table[sb[b0 + std::size_t(i)]] : ws[b0 + std::size_t(i)];
            const __m256 wv = _mm256_load_ps(wscale);
            // Lanes this group fills: all 8, or the first 4 of a short group.
            const __m256i live = count == kGroup ? _mm256_set1_epi32(-1) : _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
            const __m256i halflive = count == kGroup ? _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0)
                                                     : _mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0);
            for(int t = 0; t < tokens; ++t) {
                if(!(mask & (1u << t))) continue;
                const std::int8_t* x = a.values + std::size_t(t) * a.ld + b0 * kWeightBlock;
                const float* xs = a.scales + std::size_t(t) * a.lds;
                __m256i partial[4];
                for(int c = 0; c < 4; ++c) {
                    if(c >= chunks) { partial[c] = _mm256_setzero_si256(); continue; }
                    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + c * 32));
                    const __m256i signed_v = _mm256_sign_epi8(v, neg[c]);   // v negated where w < 0, zeroed where w == 0
                    partial[c] = _mm256_madd_epi16(_mm256_maddubs_epi16(aw[c], signed_v), ones);
                }
                const __m256 s = _mm256_cvtepi32_ps(block_sums(partial[0], partial[1], partial[2], partial[3]));
                const __m256 as = ratio == 1 ? _mm256_maskload_ps(xs + b0, live)
                                             : _mm256_permutevar8x32_ps(_mm256_maskload_ps(xs + b0 / 2, halflive), pairs);
                __m256& lane = acc[t][(b0 / kGroup) % std::size_t(L / kGroup)];
                const __m256 next = _mm256_fmadd_ps(s, _mm256_mul_ps(wv, as), lane);
                // A short group's unused lanes keep their value (an FMA with 0 could turn -0 into +0).
                lane = count == kGroup ? next : _mm256_blend_ps(lane, next, 0x0f);
            }
        }
        for(int t = 0; t < tokens; ++t) {
            if(!(mask & (1u << t))) continue;
            alignas(32) float lanes[32];
            for(int j = 0; j < 4; ++j) _mm256_store_ps(lanes + 8 * j, acc[t][j]);
            if(order.reduce == host_nvfp4::Reduce::halving) {
                for(int w = L / 2; w >= 1; w /= 2)
                    for(int i = 0; i < w; ++i) lanes[i] = lanes[i] + lanes[i + w];
                out[std::size_t(t) * ldo + r] = lanes[0] * post;
            } else {
                float sum = lanes[0];
                for(int i = 1; i < L; ++i) sum = sum + lanes[i];
                out[std::size_t(t) * ldo + r] = sum * post;
            }
        }
    }
}

}  // namespace crit::techniques::host_int::detail
