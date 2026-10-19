// AVX-512F path (compiled with AVX-512F and FMA enabled for this file only, without FP contraction). One ZMM
// register holds 16 lanes; 32 lanes use two. Each lane performs the oracle's operations in the oracle's order.
#include "host_nvfp4.h"

#include <immintrin.h>

namespace crit::techniques::host_nvfp4::detail {
namespace {

// E2M1 values by code (index 8 is -0.0, as crit::nvfp4::e2m1 decodes it).
__m512 e2m1_lut(const float* e2m1) { return _mm512_loadu_ps(e2m1); }

// The 16 E2M1 values of one block (8 bytes, low nibble first) as floats in element order.
inline __m512 block16(const std::uint8_t* p, __m512 lut) {
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i lo = _mm_and_si128(b, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), mask);
    return _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi)), lut);
}

// Halving tree over 16 lanes: (i, i + 8), (i, i + 4), (i, i + 2), (i, i + 1).
inline float halving16(__m512 p) {
    const __m256 lo = _mm512_castps512_ps256(p);
    const __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(p), 1));
    const __m256 q8 = _mm256_add_ps(lo, hi);
    const __m128 q4 = _mm_add_ps(_mm256_castps256_ps128(q8), _mm256_extractf128_ps(q8, 1));
    const __m128 q2 = _mm_add_ps(q4, _mm_movehl_ps(q4, q4));
    return _mm_cvtss_f32(_mm_add_ss(q2, _mm_shuffle_ps(q2, q2, 1)));
}

inline float sequential(const float* lanes, int n) {
    float s = lanes[0];
    for(int i = 1; i < n; ++i) s = s + lanes[i];
    return s;
}

template <int L, int T>
void rows_t(const float* e2m1, const Order& order, const MatrixView& m, const float* scales, const float* table, const float* xt, const float* x, std::size_t ldx,
            std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out, std::size_t ldo) {
    constexpr int R = L / 16;  // registers per accumulator
    (void)xt;
    const __m512 lut = e2m1_lut(e2m1);
    const std::size_t groups = m.cols / kBlock, stride = m.code_stride ? m.code_stride : m.cols / 2,
                      sstride = m.scale_stride ? m.scale_stride : m.cols / kBlock;
    // N rows at a time so their FMA chains overlap; every row keeps its own lanes and order.
    auto rows = [&]<int N>(std::size_t r) {
        __m512 acc[N][T][R];
        for(int i = 0; i < N; ++i)
            for(int t = 0; t < T; ++t)
                for(int q = 0; q < R; ++q) acc[i][t][q] = _mm512_setzero_ps();
        for(std::size_t g = 0; g < groups; ++g) {
            const int q = int(g % R);  // block g holds lanes 16q .. 16q + 15
            __m512 ws[N];
            for(int i = 0; i < N; ++i)
                ws[i] = _mm512_mul_ps(block16(m.codes + (r + i) * stride + g * 8, lut),
                                      _mm512_set1_ps(table ? table[m.scales[(r + i) * sstride + g]] : scales[(r + i - row0) * groups + g]));
            for(int t = 0; t < T; ++t) {
                if(!(mask & (1u << t))) continue;
                const __m512 xv = _mm512_loadu_ps(x + std::size_t(t) * ldx + g * kBlock);
                for(int i = 0; i < N; ++i) acc[i][t][q] = _mm512_fmadd_ps(ws[i], xv, acc[i][t][q]);
            }
        }
        for(int i = 0; i < N; ++i)
            for(int t = 0; t < T; ++t) {
                if(!(mask & (1u << t))) continue;
                float sum;
                if(order.reduce == Reduce::halving) {
                    __m512 p = acc[i][t][0];
                    if constexpr(R == 2) p = _mm512_add_ps(acc[i][t][0], acc[i][t][1]);  // (i, i + 16) first
                    sum = halving16(p);
                } else {
                    alignas(64) float lanes[L];
                    for(int q = 0; q < R; ++q) _mm512_store_ps(lanes + 16 * q, acc[i][t][q]);
                    sum = sequential(lanes, L);
                }
                out[std::size_t(t) * ldo + r + i] = sum * post;
            }
    };
    std::size_t r = row0;
    constexpr int kRows = T <= 4 ? 4 : 2;  // keep N * T accumulators in registers
    for(; r + kRows <= row1; r += kRows) rows.template operator()<kRows>(r);
    for(; r < row1; ++r) rows.template operator()<1>(r);
}

// block_dot: one vector lane per block (16 blocks per chunk), each lane running its block's sequential dot;
// lanes without a block (the tail) keep their accumulator bits.
template <int L, int T>
void block_rows(const float* e2m1, const Order& order, const MatrixView& m, const float* scales, const float* table,
                const float* xt, std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out,
                std::size_t ldo) {
    constexpr int R = L / 16;
    const __m512 lut = e2m1_lut(e2m1);
    const std::size_t groups = m.cols / kBlock, stride = m.code_stride ? m.code_stride : m.cols / 2,
                      sstride = m.scale_stride ? m.scale_stride : m.cols / kBlock, padded = (groups + 15) / 16 * 16;
    const __m512i offsets = _mm512_setr_epi32(0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120);
    const __m512i nib = _mm512_set1_epi32(15);
    for(std::size_t r = row0; r < row1; ++r) {
        const std::uint8_t* codes = m.codes + r * stride;
        __m512 acc[T][R];
        for(int t = 0; t < T; ++t)
            for(int q = 0; q < R; ++q) acc[t][q] = _mm512_setzero_ps();
        for(std::size_t b0 = 0; b0 < groups; b0 += 16) {
            const std::size_t nb = groups - b0 < 16 ? groups - b0 : 16;
            const __mmask16 live = __mmask16((1u << nb) - 1);
            const __m512i lo = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), live, offsets, codes + b0 * 8, 1);
            const __m512i hi = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), live, offsets, codes + b0 * 8 + 4, 1);
            alignas(64) float sv[16] = {};
            for(std::size_t j = 0; j < nb; ++j)
                sv[j] = table ? table[m.scales[r * sstride + b0 + j]] : scales[(r - row0) * groups + b0 + j];
            const __m512 s = _mm512_load_ps(sv);
            __m512 d[T];
            for(int t = 0; t < T; ++t) d[t] = _mm512_setzero_ps();
            for(int k = 0; k < 8; ++k) {
                const __m512i word = k < 4 ? lo : hi;
                const __m512i byte = _mm512_srli_epi32(word, 8 * (k % 4));
                const __m512 w0 = _mm512_permutexvar_ps(_mm512_and_si512(byte, nib), lut);
                const __m512 w1 = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(byte, 4), nib), lut);
                for(int t = 0; t < T; ++t) {
                    if(!(mask & (1u << t))) continue;
                    const float* x0 = xt + (std::size_t(t) * kBlock + 2 * k) * padded + b0;
                    d[t] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x0), d[t]);
                    d[t] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(x0 + padded), d[t]);
                }
            }
            const int q = int((b0 / 16) % R);
            for(int t = 0; t < T; ++t)
                if(mask & (1u << t)) acc[t][q] = _mm512_mask3_fmadd_ps(d[t], s, acc[t][q], live);
        }
        for(int t = 0; t < T; ++t) {
            if(!(mask & (1u << t))) continue;
            float sum;
            if(order.reduce == Reduce::halving) {
                __m512 p = acc[t][0];
                if constexpr(R == 2) p = _mm512_add_ps(acc[t][0], acc[t][1]);
                sum = halving16(p);
            } else {
                alignas(64) float lanes[L];
                for(int q = 0; q < R; ++q) _mm512_store_ps(lanes + 16 * q, acc[t][q]);
                sum = sequential(lanes, L);
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

void rows_avx512(const float* e2m1, const Order& order, const MatrixView& m, const float* scales, const float* table, const float* xt, const float* x, std::size_t ldx,
                 int tokens, std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out,
                 std::size_t ldo) {
    if(order.lanes == 32) by_tokens<32>(e2m1, order, m, scales, table, xt, x, ldx, tokens, mask, row0, row1, post, out, ldo);
    else by_tokens<16>(e2m1, order, m, scales, table, xt, x, ldx, tokens, mask, row0, row1, post, out, ldo);
}

}  // namespace crit::techniques::host_nvfp4::detail
