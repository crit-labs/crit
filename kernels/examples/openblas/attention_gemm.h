#pragma once
#include "openblas.h"

namespace crit_kernels {
// Adapter for Inference's synchronous packed attention GEMM contract. The caller
// keeps the OpenBlas library alive through every run; no async work escapes.
struct OpenBlasAttentionGemm {
  const OpenBlas& blas;
  void operator()(bool transpose_b, std::size_t m, std::size_t n, std::size_t k,
                  std::span<const float> a, std::span<const float> b, std::span<float> c) const {
    blas.gemm(false, transpose_b, 1, {a, m, k, k},
              {b, transpose_b ? n : k, transpose_b ? k : n, transpose_b ? k : n},
              0, {c, m, n, n});
  }
};
} // namespace crit_kernels
