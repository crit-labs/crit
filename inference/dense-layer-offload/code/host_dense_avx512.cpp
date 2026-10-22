// AVX-512F host GEMV over BF16 rows (host_dense.h). Compiled with -mavx512f -mfma (/arch:AVX512); called only when the
// CPU and OS report AVX-512F. Each row keeps 16 FP32 lanes per batch row over 16 columns per step, then sums the lanes.
#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace crit::dense_offload::detail {

void rows_avx512(const std::uint16_t* w, std::size_t cols, const float* x, std::size_t ldx, int batch, std::size_t row0, std::size_t row1,
                 float* y, std::size_t ldy) {
    for(std::size_t r = row0; r < row1; ++r) {
        const std::uint16_t* row = w + r * cols;
        __m512 acc[8];
        for(int b = 0; b < 8; ++b) acc[b] = _mm512_setzero_ps();
        for(std::size_t c = 0; c < cols; c += 16) {
            const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + c));
            const __m512 wv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
            for(int b = 0; b < batch; ++b) acc[b] = _mm512_fmadd_ps(wv, _mm512_loadu_ps(x + static_cast<std::size_t>(b) * ldx + c), acc[b]);
        }
        for(int b = 0; b < batch; ++b) y[static_cast<std::size_t>(b) * ldy + r] = _mm512_reduce_add_ps(acc[b]);
    }
}

}  // namespace crit::dense_offload::detail
