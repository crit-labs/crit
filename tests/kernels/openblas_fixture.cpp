// Synthetic C ABI fixture. This is not OpenBLAS and supplies no performance evidence.
#include <cstdint>
#include <cstdlib>
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
#if BLAS_BITS == 64
using Integer = std::int64_t;
#else
using Integer = std::int32_t;
#endif
EXPORT const char* fixture_config() {
#if BLAS_BITS == 64
    return "OpenBLAS synthetic-fixture USE64BITINT";
#else
    return "OpenBLAS synthetic-fixture";
#endif
}
EXPORT void fixture_threads(int threads) { if (threads != 1) std::abort(); }
EXPORT void fixture_sgemm(int order, int ta, int tb, Integer m, Integer n, Integer k, float alpha,
    const float* a, Integer lda, const float* b, Integer ldb, float beta, float* c, Integer ldc) {
    if (order != 101 || (ta != 111 && ta != 112) || (tb != 111 && tb != 112)) std::abort();
    for (Integer i = 0; i < m; ++i) for (Integer j = 0; j < n; ++j) {
        double value = beta == 0 ? 0 : beta * static_cast<double>(c[i * ldc + j]);
        for (Integer p = 0; p < k; ++p)
            value += alpha * static_cast<double>(a[ta == 111 ? i * lda + p : p * lda + i]) * b[tb == 111 ? p * ldb + j : j * ldb + p];
        c[i * ldc + j] = static_cast<float>(value);
    }
}
