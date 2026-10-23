// AVX2 + FMA host GEMV over BF16 rows (host_dense.h). Compiled with -mavx2 -mfma (/arch:AVX2); called only when the
// CPU and OS report AVX2. Each row keeps 8 FP32 lanes per batch row over 16 columns per step (two 8-lane halves), then
// sums the lanes.
#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace crit::dense_offload::detail {

namespace {
inline __m256 bf16x8(const std::uint16_t* p) {
    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}
inline float sum8(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
}  // namespace

void rows_avx2(const std::uint16_t* w, std::size_t cols, const float* x, std::size_t ldx, int batch, std::size_t row0, std::size_t row1,
               float* y, std::size_t ldy) {
    for(std::size_t r = row0; r < row1; ++r) {
        const std::uint16_t* row = w + r * cols;
        __m256 acc[8];
        for(int b = 0; b < 8; ++b) acc[b] = _mm256_setzero_ps();
        for(std::size_t c = 0; c < cols; c += 16) {
            const __m256 w0 = bf16x8(row + c), w1 = bf16x8(row + c + 8);
            for(int b = 0; b < batch; ++b) {
                const float* xr = x + static_cast<std::size_t>(b) * ldx + c;
                acc[b] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xr), acc[b]);
                acc[b] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xr + 8), acc[b]);
            }
        }
        for(int b = 0; b < batch; ++b) y[static_cast<std::size_t>(b) * ldy + r] = sum8(acc[b]);
    }
}

}  // namespace crit::dense_offload::detail
