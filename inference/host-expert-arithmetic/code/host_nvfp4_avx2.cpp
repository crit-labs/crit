// AVX2 + FMA path (compiled with AVX2 and FMA enabled for this file only, without FP contraction). Two YMM
// registers hold 16 lanes; each lane performs the oracle's operations in the oracle's order.
#include "host_nvfp4.h"

#include <immintrin.h>

namespace crit::techniques::host_nvfp4::detail {
namespace {

struct Lut {
    __m256 magnitude;  // codes 0..7
};
Lut lut(const float* e2m1) { return {_mm256_loadu_ps(e2m1)}; }

// Eight E2M1 codes (one per int lane) as floats: magnitude by table, sign bit from bit 3 (code 8 is -0.0).
inline __m256 decode8(__m256i codes, const Lut& l) {
    const __m256 mag = _mm256_permutevar8x32_ps(l.magnitude, _mm256_and_si256(codes, _mm256_set1_epi32(7)));
    const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(codes, _mm256_set1_epi32(8)), 28);
    return _mm256_xor_ps(mag, _mm256_castsi256_ps(sign));
}

// One block's 16 values as two halves of 8 (element order).
inline void block16(const std::uint8_t* p, const Lut& l, __m256& lo8, __m256& hi8) {
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i lo = _mm_and_si128(b, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), mask);
    const __m128i e = _mm_unpacklo_epi8(lo, hi);  // 16 codes in element order
    lo8 = decode8(_mm256_cvtepu8_epi32(e), l);
    hi8 = decode8(_mm256_cvtepu8_epi32(_mm_srli_si128(e, 8)), l);
}

// Halving tree over 8 lanes: (i, i + 4), (i, i + 2), (i, i + 1).
inline float halving8(__m256 q8) {
    const __m128 q4 = _mm_add_ps(_mm256_castps256_ps128(q8), _mm256_extractf128_ps(q8, 1));
    const __m128 q2 = _mm_add_ps(q4, _mm_movehl_ps(q4, q4));
    return _mm_cvtss_f32(_mm_add_ss(q2, _mm_shuffle_ps(q2, q2, 1)));
}

template <int L, int T>
void rows_t(const float* e2m1, const Order& order, const MatrixView& m, const float* scales, const float* table, const float* xt, const float* x, std::size_t ldx,
            std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out, std::size_t ldo) {
    constexpr int R = L / 8;  // YMM registers per accumulator
    (void)xt;
    const Lut l = lut(e2m1);
    const std::size_t groups = m.cols / kBlock, stride = m.code_stride ? m.code_stride : m.cols / 2,
                      sstride = m.scale_stride ? m.scale_stride : m.cols / kBlock;
    // N rows at a time so their FMA chains overlap; every row keeps its own lanes and order.
    auto rows = [&]<int N>(std::size_t r) {
        __m256 acc[N][T][R];
        for(int i = 0; i < N; ++i)
            for(int t = 0; t < T; ++t)
                for(int q = 0; q < R; ++q) acc[i][t][q] = _mm256_setzero_ps();
        for(std::size_t g = 0; g < groups; ++g) {
            const int q = int(g % (L / 16)) * 2;  // block g holds lanes 16 (g % (L / 16)) .. + 15
            __m256 lo8[N], hi8[N];
            for(int i = 0; i < N; ++i) {
                block16(m.codes + (r + i) * stride + g * 8, l, lo8[i], hi8[i]);
                const __m256 sv = _mm256_set1_ps(table ? table[m.scales[(r + i) * sstride + g]] : scales[(r + i - row0) * groups + g]);
                lo8[i] = _mm256_mul_ps(lo8[i], sv);
                hi8[i] = _mm256_mul_ps(hi8[i], sv);
            }
            for(int t = 0; t < T; ++t) {
                if(!(mask & (1u << t))) continue;
                const float* xt = x + std::size_t(t) * ldx + g * kBlock;
                const __m256 xa = _mm256_loadu_ps(xt), xb = _mm256_loadu_ps(xt + 8);
                for(int i = 0; i < N; ++i) {
                    acc[i][t][q] = _mm256_fmadd_ps(lo8[i], xa, acc[i][t][q]);
                    acc[i][t][q + 1] = _mm256_fmadd_ps(hi8[i], xb, acc[i][t][q + 1]);
                }
            }
        }
        for(int i = 0; i < N; ++i)
            for(int t = 0; t < T; ++t) {
                if(!(mask & (1u << t))) continue;
                float sum;
                if(order.reduce == Reduce::halving) {
                    // 32 lanes: (i, i + 16) first; then (i, i + 8) pairs the two halves of 16.
                    __m256 a = acc[i][t][0], b = acc[i][t][1];
                    if constexpr(R == 4) {
                        a = _mm256_add_ps(acc[i][t][0], acc[i][t][2]);
                        b = _mm256_add_ps(acc[i][t][1], acc[i][t][3]);
                    }
                    sum = halving8(_mm256_add_ps(a, b));
                } else {
                    alignas(32) float lanes[L];
                    for(int q = 0; q < R; ++q) _mm256_store_ps(lanes + 8 * q, acc[i][t][q]);
                    sum = lanes[0];
                    for(int j = 1; j < L; ++j) sum = sum + lanes[j];
                }
                out[std::size_t(t) * ldo + r + i] = sum * post;
            }
    };
    std::size_t r = row0;
    constexpr int kRows = T <= 4 ? 4 : 2;  // keep N * T accumulators in registers
    for(; r + kRows <= row1; r += kRows) rows.template operator()<kRows>(r);
    for(; r < row1; ++r) rows.template operator()<1>(r);
}

// block_dot: one vector lane per block (8 blocks per chunk), each lane running its block's sequential dot;
// lanes without a block (the tail) keep their accumulator bits.
template <int L, int T>
void block_rows(const float* e2m1, const Order& order, const MatrixView& m, const float* scales, const float* table,
                const float* xt, std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out,
                std::size_t ldo) {
    constexpr int R = L / 8;
    const Lut l = lut(e2m1);
    const std::size_t groups = m.cols / kBlock, stride = m.code_stride ? m.code_stride : m.cols / 2,
                      sstride = m.scale_stride ? m.scale_stride : m.cols / kBlock, padded = (groups + 15) / 16 * 16;
    const __m256i offsets = _mm256_setr_epi32(0, 8, 16, 24, 32, 40, 48, 56);
    const __m256i nib = _mm256_set1_epi32(15);
    for(std::size_t r = row0; r < row1; ++r) {
        const std::uint8_t* codes = m.codes + r * stride;
        __m256 acc[T][R];
        for(int t = 0; t < T; ++t)
            for(int q = 0; q < R; ++q) acc[t][q] = _mm256_setzero_ps();
        for(std::size_t b0 = 0; b0 < groups; b0 += 8) {
            const std::size_t nb = groups - b0 < 8 ? groups - b0 : 8;
            alignas(32) int live_bits[8];
            for(int j = 0; j < 8; ++j) live_bits[j] = std::size_t(j) < nb ? -1 : 0;
            const __m256i live = _mm256_load_si256(reinterpret_cast<const __m256i*>(live_bits));
            const int* base = reinterpret_cast<const int*>(codes + b0 * 8);
            const __m256i lo = _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), base, offsets, live, 1);
            const __m256i hi = _mm256_mask_i32gather_epi32(_mm256_setzero_si256(),
                                                           reinterpret_cast<const int*>(codes + b0 * 8 + 4), offsets, live, 1);
            alignas(32) float sv[8] = {};
            for(std::size_t j = 0; j < nb; ++j)
                sv[j] = table ? table[m.scales[r * sstride + b0 + j]] : scales[(r - row0) * groups + b0 + j];
            const __m256 s = _mm256_load_ps(sv);
            __m256 d[T];
            for(int t = 0; t < T; ++t) d[t] = _mm256_setzero_ps();
            for(int k = 0; k < 8; ++k) {
                const __m256i word = k < 4 ? lo : hi;
                const __m256i byte = _mm256_srli_epi32(word, 8 * (k % 4));
                const __m256 w0 = decode8(_mm256_and_si256(byte, nib), l);
                const __m256 w1 = decode8(_mm256_and_si256(_mm256_srli_epi32(byte, 4), nib), l);
                for(int t = 0; t < T; ++t) {
                    if(!(mask & (1u << t))) continue;
                    const float* x0 = xt + (std::size_t(t) * kBlock + 2 * k) * padded + b0;
                    d[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x0), d[t]);
                    d[t] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x0 + padded), d[t]);
                }
            }
            const int q = int((b0 / 8) % R);
            const __m256 keep = _mm256_castsi256_ps(live);
            for(int t = 0; t < T; ++t)
                if(mask & (1u << t))
                    acc[t][q] = _mm256_blendv_ps(acc[t][q], _mm256_fmadd_ps(d[t], s, acc[t][q]), keep);
        }
        for(int t = 0; t < T; ++t) {
            if(!(mask & (1u << t))) continue;
            float sum;
            if(order.reduce == Reduce::halving) {
                __m256 a = acc[t][0], b = acc[t][1];
                if constexpr(R == 4) {
                    a = _mm256_add_ps(acc[t][0], acc[t][2]);
                    b = _mm256_add_ps(acc[t][1], acc[t][3]);
                }
                sum = halving8(_mm256_add_ps(a, b));
            } else {
                alignas(32) float lanes[L];
                for(int q = 0; q < R; ++q) _mm256_store_ps(lanes + 8 * q, acc[t][q]);
                sum = lanes[0];
                for(int j = 1; j < L; ++j) sum = sum + lanes[j];
            }
            out[std::size_t(t) * ldo + r] = sum * post;
        }
    }
}

template <int L>
void by_tokens(const float* e2m1, const Order& order, const MatrixView& m, const float* scales, const float* table, const float* xt, const float* x,
               std::size_t ldx, int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post,
               float* out, std::size_t ldo) {
    switch(tokens) {
        case 1: return order.product == Product::block_dot ? block_rows<L, 1>(e2m1, order, m, scales, table, xt, mask, row0, row1, post, out, ldo) : rows_t<L, 1>(e2m1, order, m, scales, table, xt, x, ldx, mask, row0, row1, post, out, ldo);
        case 2: return order.product == Product::block_dot ? block_rows<L, 2>(e2m1, order, m, scales, table, xt, mask, row0, row1, post, out, ldo) : rows_t<L, 2>(e2m1, order, m, scales, table, xt, x, ldx, mask, row0, row1, post, out, ldo);
        case 3: return order.product == Product::block_dot ? block_rows<L, 3>(e2m1, order, m, scales, table, xt, mask, row0, row1, post, out, ldo) : rows_t<L, 3>(e2m1, order, m, scales, table, xt, x, ldx, mask, row0, row1, post, out, ldo);
        case 4: return order.product == Product::block_dot ? block_rows<L, 4>(e2m1, order, m, scales, table, xt, mask, row0, row1, post, out, ldo) : rows_t<L, 4>(e2m1, order, m, scales, table, xt, x, ldx, mask, row0, row1, post, out, ldo);
        case 5: return order.product == Product::block_dot ? block_rows<L, 5>(e2m1, order, m, scales, table, xt, mask, row0, row1, post, out, ldo) : rows_t<L, 5>(e2m1, order, m, scales, table, xt, x, ldx, mask, row0, row1, post, out, ldo);
        case 6: return order.product == Product::block_dot ? block_rows<L, 6>(e2m1, order, m, scales, table, xt, mask, row0, row1, post, out, ldo) : rows_t<L, 6>(e2m1, order, m, scales, table, xt, x, ldx, mask, row0, row1, post, out, ldo);
        case 7: return order.product == Product::block_dot ? block_rows<L, 7>(e2m1, order, m, scales, table, xt, mask, row0, row1, post, out, ldo) : rows_t<L, 7>(e2m1, order, m, scales, table, xt, x, ldx, mask, row0, row1, post, out, ldo);
        default: return order.product == Product::block_dot ? block_rows<L, 8>(e2m1, order, m, scales, table, xt, mask, row0, row1, post, out, ldo) : rows_t<L, 8>(e2m1, order, m, scales, table, xt, x, ldx, mask, row0, row1, post, out, ldo);
    }
}

}  // namespace

void rows_avx2(const float* e2m1, const Order& order, const MatrixView& m, const float* scales, const float* table, const float* xt, const float* x, std::size_t ldx,
               int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out,
               std::size_t ldo) {
    if(order.lanes == 32) by_tokens<32>(e2m1, order, m, scales, table, xt, x, ldx, tokens, mask, row0, row1, post, out, ldo);
    else by_tokens<16>(e2m1, order, m, scales, table, xt, x, ldx, tokens, mask, row0, row1, post, out, ldo);
}

}  // namespace crit::techniques::host_nvfp4::detail
