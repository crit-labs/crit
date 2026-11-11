// Decode GEMVs on CUDA cores (decode_gemv.h): one kernel template for every weight format (kernel.json's structure
// gemv-cuda, the format a run option: NVFP4, FP8 with row scales, MXFP8, BF16, INT4 with a zero point a group), the
// format's decoder a trait (Fmt below) over the kernel core's decoders. Its earlier per-format structures gemv-nvfp4,
// gemv-fp8 and gemv-bf16 stay as aliases of it for one format (a tuning that names them still applies): the same
// kernel, the same lane order, and so the same bits as the per-format kernels this file held before.
// The decoders, the 16-bit conversions, the block-scale layout, the weight loads and the warp sums are the kernel
// core's (kernels/core: crit_kcore/formats.cuh, load.cuh, reduce.cuh).
#include "decode_gemv.h"
#include "crit_tuning.h"

#include "crit_kcore/formats.cuh"
#include "crit_kcore/load.cuh"
#include "crit_kcore/reduce.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace crit::kernels::decode_gemv {
namespace {

using crit::kcore::from_float;
using crit::kcore::scale_offset;
using crit::kcore::to_float;

template <typename X>
__device__ __forceinline__ void load8(const X* p, float* out) {
  const uint4 a = __ldg(reinterpret_cast<const uint4*>(p));
  const X* va = reinterpret_cast<const X*>(&a);
#pragma unroll
  for (int e = 0; e < 8; ++e) out[e] = to_float(va[e]);
}

template <typename X>
__device__ __forceinline__ void load16(const X* p, float* out) {
  const uint4 a = __ldg(reinterpret_cast<const uint4*>(p));
  const uint4 b = __ldg(reinterpret_cast<const uint4*>(p) + 1);
  const X* va = reinterpret_cast<const X*>(&a);
  const X* vb = reinterpret_cast<const X*>(&b);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    out[e] = to_float(va[e]);
    out[8 + e] = to_float(vb[e]);
  }
}

template <int STREAM>
__device__ __forceinline__ uint4 load_codes(const uint8_t* p) { return crit::kcore::ld16<STREAM>(p); }

// ---- the weight formats: a decoder trait each ----
//
// A lane reads 16 bytes of a matrix row a step; the trait says how many values they hold (VALUES), how many it decodes
// at once (CHUNK, values in element order), the side data a row's 16 bytes need (Side: their scale bits, loaded with
// them), the exact FP32 values of chunk c, and what a finished sum is multiplied by. Every value is exact in FP32
// and so is its product with a 16-bit activation, so a row's bits depend on the lane order alone.
template <int WF>
struct Fmt;

// NVFP4: element 2i in the low nibble of byte i, E4M3 block scales per 16 (row major or cuBLASLt's 128 x 4 tiles),
// the inverse global scale after the sum. A lane's 32 values cover two scale blocks: the two scale bytes in one load.
template <>
struct Fmt<kNvfp4> {
  static constexpr int VALUES = 32, CHUNK = 8;
  using Side = uint16_t;
  static __device__ __forceinline__ Side side(const Matrix& m, int n, int k, int K) {
    return __ldg(reinterpret_cast<const uint16_t*>(m.scales + scale_offset(n, k >> 4, K >> 4, m.ct)));
  }
  // The block scale over the core's E2M1 unit (times 2^14 on the bit pattern path).
  static __device__ __forceinline__ float block_scale(uint32_t bits) {
    return crit::kcore::e4m3_cvt(bits) * (1.0f / crit::kcore::e2m1_half2_unit());
  }
  static __device__ __forceinline__ void decode(const uint4& cw, Side s, int c, float* w) {
    const uint32_t* words = reinterpret_cast<const uint32_t*>(&cw);
    crit::kcore::e2m1x8_to_float(words[c], block_scale(uint8_t(c >= 2 ? (s >> 8) : (s & 0xff))), w);
  }
  static __device__ __forceinline__ float finish(const Matrix& m, int n, float v) { return v * m.row_scale[n]; }
};

// FP8 E4M3 codes, one FP32 scale a row after the sum.
template <>
struct Fmt<kFp8> {
  static constexpr int VALUES = 16, CHUNK = 16;
  using Side = uint8_t;
  static __device__ __forceinline__ Side side(const Matrix&, int, int, int) { return 0; }
  static __device__ __forceinline__ void decode(const uint4& cw, Side, int, float* w) {
    const uint32_t* words = reinterpret_cast<const uint32_t*>(&cw);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float2 f = __half22float2(crit::kcore::e4m3x2_to_half2(words[e >> 1] >> (16 * (e & 1))));
      w[2 * e] = f.x;
      w[2 * e + 1] = f.y;
    }
  }
  static __device__ __forceinline__ float finish(const Matrix& m, int n, float v) { return v * m.row_scale[n]; }
};

// MXFP8: E4M3 codes, each times its E8M0 power of two per 32 inputs (one per lane step).
template <>
struct Fmt<kMxfp8> {
  static constexpr int VALUES = 16, CHUNK = 16;
  using Side = uint8_t;
  static __device__ __forceinline__ Side side(const Matrix& m, int n, int k, int K) {
    return __ldg(m.scales + (size_t)n * (K >> 5) + (k >> 5));
  }
  static __device__ __forceinline__ void decode(const uint4& cw, Side s, int, float* w) {
    const uint32_t* words = reinterpret_cast<const uint32_t*>(&cw);
    const float scale = crit::kcore::e8m0_value(s);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float2 f = __half22float2(crit::kcore::e4m3x2_to_half2(words[e >> 1] >> (16 * (e & 1))));
      w[2 * e] = f.x * scale;
      w[2 * e + 1] = f.y * scale;
    }
  }
  static __device__ __forceinline__ float finish(const Matrix&, int, float v) { return v; }
};

// BF16 weights.
template <>
struct Fmt<kBf16> {
  static constexpr int VALUES = 8, CHUNK = 8;
  using Side = uint8_t;
  static __device__ __forceinline__ Side side(const Matrix&, int, int, int) { return 0; }
  static __device__ __forceinline__ void decode(const uint4& cw, Side, int, float* w) {
    const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&cw);
#pragma unroll
    for (int e = 0; e < 8; ++e) w[e] = to_float(b[e]);
  }
  static __device__ __forceinline__ float finish(const Matrix&, int, float v) { return v; }
};

// INT4 with a zero point (AWQ, GPTQ, other asymmetric INT4 groups in crit's rows, repack_int4): unsigned codes,
// element 2i in the low nibble of byte i, FP16 scales [N][K / group] and zero points [N][K / group] (bytes, 0 to 16); a
// value is (code - zero) * scale, exact in FP32. A lane's 32 values lie in one group (group a multiple of 32).
template <>
struct Fmt<kInt4Zp> {
  static constexpr int VALUES = 32, CHUNK = 8;
  using Side = uint32_t;   // the scale's FP16 bits, the zero point at bits 16 and up
  static __device__ __forceinline__ Side side(const Matrix& m, int n, int k, int K) {
    const size_t at = (size_t)n * (K / m.group) + k / m.group;
    return uint32_t(__ldg(reinterpret_cast<const uint16_t*>(m.scales) + at)) | (uint32_t(__ldg(m.zeros + at)) << 16);
  }
  static __device__ __forceinline__ void decode(const uint4& cw, Side s, int c, float* w) {
    const uint32_t* words = reinterpret_cast<const uint32_t*>(&cw);
    crit::kcore::int4x8_zp_to_float(words[c], int(s >> 16), crit::kcore::f16_value(s), w);
  }
  static __device__ __forceinline__ float finish(const Matrix&, int, float v) { return v; }
};

// ---- the kernel (structure gemv-cuda) ----
//
// Knobs (kernel.json, structure gemv-cuda; crit_tuning.h picks them per format, shape and row count):
//   WARPS_N  warps a block
//   ROWS     output rows a warp owns (0: the format's rule, rule_rows)
//   WPR      warps that share the same rows, each taking every WPR-th step of K; their partial sums are added in warp
//            order through shared memory. It changes the sum order (an order knob: one value a matrix for every row
//            count), never the dependence of a row on the rows beside it.
//   PREF     steps of weight loads in flight ahead of the arithmetic (0: load, then compute)
//   STREAM   read codes with L1::no_allocate (they are read once)
// A lane reads 16 bytes of each of its rows a step (VALUES values from k = VALUES (part * 32 + lane)), multiplies
// them in element order and one xor-shuffle tree adds the lanes; whatever the knobs, a lane's products and sums for a
// row run in the same order for every M, so a row's result is the same bits alone and beside up to 15 others.
template <int WF, int WARPS_N, int ROWS, int WPR, int PREF, int STREAM, int M, typename X, typename O>
__global__ void __launch_bounds__(WARPS_N * 32) gemv_kernel(const X* __restrict__ x, int ldx, const Matrix m,
                                                            O* __restrict__ out, int ldo, int N, int K) {
  using F = Fmt<WF>;
  using Side = typename F::Side;
  constexpr int GROUPS = WARPS_N / WPR;
  constexpr int SLOTS = PREF > 0 ? PREF : 1;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int group = warp / WPR, part = warp % WPR;
  const int row0 = (blockIdx.x * GROUPS + group) * ROWS;
  const bool live = row0 < N;
  const int kb = (K / F::VALUES) << 4;   // a row's bytes
  const int stride = 512 * WPR;
  float acc[ROWS][M];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int i = 0; i < M; ++i) acc[r][i] = 0.f;
  uint4 wv[SLOTS][ROWS];
  Side sv[SLOTS][ROWS];
  auto fetch = [&](int slot, int o) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int n = min(row0 + r, N - 1);   // a row past N reads row N - 1 and writes nothing
      wv[slot][r] = load_codes<STREAM>(m.codes + (size_t)n * kb + o);
      sv[slot][r] = F::side(m, n, (o >> 4) * F::VALUES, K);   // o >= 0: a shift, not a signed division
    }
  };
  int off = (part * 32 + lane) * 16;
  if (live) {
#pragma unroll
    for (int p = 0; p < PREF; ++p)
      if (off + p * stride < kb) fetch(p, off + p * stride);
    for (; off < kb; off += stride) {
      if constexpr (PREF == 0) fetch(0, off);
      uint4 cw[ROWS];
      Side cs[ROWS];
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        cw[r] = wv[0][r];
        cs[r] = sv[0][r];
      }
      // The loads move down a slot (static indices: registers) and the newest slot is refilled PREF steps ahead.
#pragma unroll
      for (int p = 0; p + 1 < PREF; ++p)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          wv[p][r] = wv[p + 1][r];
          sv[p][r] = sv[p + 1][r];
        }
      if constexpr (PREF > 0) {
        if (off + PREF * stride < kb) fetch(SLOTS - 1, off + PREF * stride);
      }
      const int k = (off >> 4) * F::VALUES;
      // The lane's chunks in element order: each row's sums take the same order whatever ROWS and M are.
#pragma unroll
      for (int c = 0; c < F::VALUES / F::CHUNK; ++c) {
        float w[ROWS][F::CHUNK];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) F::decode(cw[r], cs[r], c, w[r]);
#pragma unroll
        for (int i = 0; i < M; ++i) {
          float xf[F::CHUNK];
          if constexpr (F::CHUNK == 16) load16(x + (size_t)i * ldx + k + 16 * c, xf);
          else load8(x + (size_t)i * ldx + k + 8 * c, xf);
#pragma unroll
          for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int e = 0; e < F::CHUNK; ++e) acc[r][i] = fmaf(w[r][e], xf[e], acc[r][i]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int i = 0; i < M; ++i) acc[r][i] = crit::kcore::warp_sum(acc[r][i]);
  if constexpr (WPR > 1) {
    __shared__ float partial[WARPS_N][ROWS][M];
    if (lane == 0)
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) partial[warp][r][i] = acc[r][i];
    __syncthreads();
    if (part != 0) return;
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int i = 0; i < M; ++i) {
        float v = partial[group * WPR][r][i];
#pragma unroll
        for (int p = 1; p < WPR; ++p) v += partial[group * WPR + p][r][i];   // in warp order
        acc[r][i] = v;
      }
  }
  if (!live || lane != 0) return;
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int n = row0 + r;
    if (n < N)
#pragma unroll
      for (int i = 0; i < M; ++i) out[(size_t)i * ldo + n] = from_float<O>(F::finish(m, n, acc[r][i]));
  }
}

// The current device's multiprocessor count, per device (queried once each).
int multiprocessors() {
  static int counts[64] = {};
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) return 1;
  if (!counts[device]) cudaDeviceGetAttribute(&counts[device], cudaDevAttrMultiProcessorCount, device);
  return counts[device] > 0 ? counts[device] : 1;
}

template <int WF, int WARPS_N, int ROWS, int WPR, int PREF, int STREAM, int M, typename X, typename O>
void launch_rows(const void* x, int ldx, const Matrix& m, void* out, int ldo, int N, int K, cudaStream_t stream) {
  constexpr int BLOCK_ROWS = WARPS_N / WPR * ROWS;
  const int blocks = (N + BLOCK_ROWS - 1) / BLOCK_ROWS;
  gemv_kernel<WF, WARPS_N, ROWS, WPR, PREF, STREAM, M, X, O><<<blocks, WARPS_N * 32, 0, stream>>>(
      reinterpret_cast<const X*>(x), ldx, m, reinterpret_cast<O*>(out), ldo, N, K);
}

// ROWS 0: each format's rows a warp from before the knobs (speed only). NVFP4 and INT4: 2, 4 at 3-4 activation rows, 8
// at 5-8, 4 at 9-16 (gemv-nvfp4's default_rows); BF16: 2, 4 at 3-8 (gemv-bf16's); FP8 and MXFP8: fewer at more
// activation rows (each weight is decoded once for them all) and where the matrix would not give the device two
// blocks a multiprocessor (gemv-fp8's ROWS 0).
template <int WF, int M>
constexpr int rule_rows() {
  if constexpr (WF == kBf16) return M >= 3 && M <= 8 ? 4 : 2;
  else return M >= 3 && M <= 4 ? 4 : M >= 5 && M <= 8 ? 8 : M >= 9 ? 4 : 2;
}

template <int WF, int WARPS_N, int ROWS, int WPR, int PREF, int STREAM, int M, typename X, typename O>
void launch(const void* x, int ldx, const Matrix& m, void* out, int ldo, int N, int K, cudaStream_t stream) {
  if constexpr (ROWS != 0) {
    launch_rows<WF, WARPS_N, ROWS, WPR, PREF, STREAM, M, X, O>(x, ldx, m, out, ldo, N, K, stream);
  } else if constexpr (WF == kFp8 || WF == kMxfp8) {
    constexpr int MORE = M <= 4 ? 4 : M <= 8 ? 2 : 1;
    constexpr int FEWER = MORE == 1 ? 1 : MORE / 2;
    if ((N + MORE * WARPS_N - 1) / (MORE * WARPS_N) >= 2 * multiprocessors())
      launch_rows<WF, WARPS_N, MORE, WPR, PREF, STREAM, M, X, O>(x, ldx, m, out, ldo, N, K, stream);
    else
      launch_rows<WF, WARPS_N, FEWER, WPR, PREF, STREAM, M, X, O>(x, ldx, m, out, ldo, N, K, stream);
  } else {
    launch_rows<WF, WARPS_N, rule_rows<WF, M>(), WPR, PREF, STREAM, M, X, O>(x, ldx, m, out, ldo, N, K, stream);
  }
}

template <int WF, int WARPS_N, int ROWS, int WPR, int PREF, int STREAM, typename X, typename O>
bool dispatch_rows(int M, const void* x, int ldx, const Matrix& m, void* out, int ldo, int N, int K,
                   cudaStream_t stream) {
  if constexpr (WPR > WARPS_N || WARPS_N % WPR != 0) {
    return false;
  } else {
#define CALL(r) launch<WF, WARPS_N, ROWS, WPR, PREF, STREAM, r, X, O>(x, ldx, m, out, ldo, N, K, stream)
#define CRIT_ROWS_CASE(r) case r: CALL(r); return true;
    switch (M) {
#ifdef CRIT_SWEEP_BUILD
      // A sweep build calls the row counts kernel.json's tune_rows lists, and only those.
      CRIT_ROWS_CASE(1) CRIT_ROWS_CASE(2) CRIT_ROWS_CASE(4) CRIT_ROWS_CASE(8) CRIT_ROWS_CASE(16)
#else
      CRIT_ROWS_CASE(1) CRIT_ROWS_CASE(2) CRIT_ROWS_CASE(3) CRIT_ROWS_CASE(4) CRIT_ROWS_CASE(5) CRIT_ROWS_CASE(6)
      CRIT_ROWS_CASE(7) CRIT_ROWS_CASE(8) CRIT_ROWS_CASE(9) CRIT_ROWS_CASE(10) CRIT_ROWS_CASE(11) CRIT_ROWS_CASE(12)
      CRIT_ROWS_CASE(13) CRIT_ROWS_CASE(14) CRIT_ROWS_CASE(15) CRIT_ROWS_CASE(16)
#endif
      default: return false;
    }
#undef CRIT_ROWS_CASE
#undef CALL
  }
}

// gemv-cuda's configuration for this format, matrix and row count (crit_tuning.h; 0, the defaults, where it names
// none).
template <int WF, typename X, typename O>
bool dispatch_generic(int M, const void* x, int ldx, const Matrix& m, void* out, int ldo, int N, int K,
                      cudaStream_t stream) {
  switch (crit_lookup_gemv_cuda(WF, N, K, M)) {
#define CRIT_GEMV_CONFIG(index, warps, rows, wpr, pref, stream_) \
  case index: return dispatch_rows<WF, warps, rows, wpr, pref, stream_, X, O>(M, x, ldx, m, out, ldo, N, K, stream);
    CRIT_CONFIGS_gemv_cuda(CRIT_GEMV_CONFIG)
#undef CRIT_GEMV_CONFIG
    default: return false;
  }
}

// The aliases (gemv-nvfp4, gemv-fp8, gemv-bf16: gemv-cuda for one format, with the knobs they had). A build holds an
// alias's dispatch only where its tuning gives the alias more than its defaults or a table (a tuning made before
// gemv-cuda); a call takes the alias's configuration where a sweep selected one, where the alias's table names a
// configuration other than its defaults for the call, or where the tuning tunes the alias and not gemv-cuda.
#define CRIT_ALIAS_BUILT(S) (CRIT_CONFIG_COUNT_##S > 1 || crit_table_size_##S > 0)
#define CRIT_ALIAS_TAKES(S, index) \
  (crit_selected_##S >= 0 || (index) != 0 || (crit_table_size_##S > 0 && crit_table_size_gemv_cuda == 0))

template <int WF, typename X, typename O>
bool dispatch(int M, const void* x, int ldx, const Matrix& m, void* out, int ldo, int N, int K, cudaStream_t stream) {
  if constexpr (WF == kNvfp4 && CRIT_ALIAS_BUILT(gemv_nvfp4)) {
    const int index = crit_lookup_gemv_nvfp4(N, K, M);
    if (CRIT_ALIAS_TAKES(gemv_nvfp4, index)) {
      switch (index) {
#define CRIT_NVFP4_CONFIG(i, warps, rows, wpr, pref, stream_) \
  case i: return dispatch_rows<WF, warps, rows, wpr, pref, stream_, X, O>(M, x, ldx, m, out, ldo, N, K, stream);
        CRIT_CONFIGS_gemv_nvfp4(CRIT_NVFP4_CONFIG)
#undef CRIT_NVFP4_CONFIG
        default: return false;
      }
    }
  }
  if constexpr ((WF == kFp8 || WF == kMxfp8) && CRIT_ALIAS_BUILT(gemv_fp8)) {
    const int index = crit_lookup_gemv_fp8(N, K, M);
    if (CRIT_ALIAS_TAKES(gemv_fp8, index)) {
      switch (index) {
#define CRIT_FP8_CONFIG(i, warps, rows, pref, stream_) \
  case i: return dispatch_rows<WF, warps, rows, 1, pref, stream_, X, O>(M, x, ldx, m, out, ldo, N, K, stream);
        CRIT_CONFIGS_gemv_fp8(CRIT_FP8_CONFIG)
#undef CRIT_FP8_CONFIG
        default: return false;
      }
    }
  }
  if constexpr (WF == kBf16 && CRIT_ALIAS_BUILT(gemv_bf16)) {
    const int index = crit_lookup_gemv_bf16(N, K, M);
    if (CRIT_ALIAS_TAKES(gemv_bf16, index)) {
      switch (index) {
#define CRIT_BF16_CONFIG(i, warps, rows, pref, stream_) \
  case i: return dispatch_rows<WF, warps, rows, 1, pref, stream_, X, O>(M, x, ldx, m, out, ldo, N, K, stream);
        CRIT_CONFIGS_gemv_bf16(CRIT_BF16_CONFIG)
#undef CRIT_BF16_CONFIG
        default: return false;
      }
    }
  }
  return dispatch_generic<WF, X, O>(M, x, ldx, m, out, ldo, N, K, stream);
}
#undef CRIT_ALIAS_TAKES
#undef CRIT_ALIAS_BUILT

template <int WF>
bool by_types(bool x_bf16, bool out_fp32, int M, const void* x, int ldx, const Matrix& m, void* out, int ldo, int N,
              int K, cudaStream_t stream) {
#ifdef CRIT_SWEEP_BUILD
  // A sweep build times BF16 activations into BF16 outputs, the bench's call, and compiles nothing else.
  return x_bf16 && !out_fp32 && dispatch<WF, __nv_bfloat16, __nv_bfloat16>(M, x, ldx, m, out, ldo, N, K, stream);
#else
  if constexpr (WF == kBf16) {   // BF16 weights take BF16 activations
    if (!x_bf16) return false;
    return out_fp32 ? dispatch<WF, __nv_bfloat16, float>(M, x, ldx, m, out, ldo, N, K, stream)
                    : dispatch<WF, __nv_bfloat16, __nv_bfloat16>(M, x, ldx, m, out, ldo, N, K, stream);
  } else {
    if (out_fp32)
      return x_bf16 ? dispatch<WF, __nv_bfloat16, float>(M, x, ldx, m, out, ldo, N, K, stream)
                    : dispatch<WF, __half, float>(M, x, ldx, m, out, ldo, N, K, stream);
    return x_bf16 ? dispatch<WF, __nv_bfloat16, __nv_bfloat16>(M, x, ldx, m, out, ldo, N, K, stream)
                  : dispatch<WF, __half, __half>(M, x, ldx, m, out, ldo, N, K, stream);
  }
#endif
}

__global__ void hardware_kernel(int* out) { *out = CRIT_KCORE_HAS_E2M1_CVT; }

// AWQ or GPTQ tensors into crit's INT4 rows (repack_int4): one thread a byte of codes (two inputs of a row), then one a
// group of a row for its zero point and scale.
__global__ void repack_codes_kernel(int kind, const uint32_t* __restrict__ qweight, const int* __restrict__ perm,
                                    uint8_t* __restrict__ codes, int N, int K) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)N * (K / 2)) return;
  const int n = int(i / (K / 2)), j = int(i % (K / 2));
  uint32_t pair = 0;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int k = perm ? perm[2 * j + h] : 2 * j + h;
    const uint32_t code = kind == kRepackAwq
                              ? crit::kcore::awq_code(qweight[(size_t)k * (N / 8) + n / 8], n & 7)    // [K][N / 8]
                              : crit::kcore::nibble(qweight[(size_t)(k / 8) * N + n], k & 7);     // [K / 8][N]
    pair |= code << (4 * h);
  }
  codes[i] = uint8_t(pair);
}

__global__ void repack_groups_kernel(int kind, const uint32_t* __restrict__ qzeros, const uint16_t* __restrict__ scales,
                                     int zero_offset, uint8_t* __restrict__ zeros, uint16_t* __restrict__ scales_out,
                                     int N, int G) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)N * G) return;
  const int n = int(i / G), g = int(i % G);
  const uint32_t word = qzeros[(size_t)g * (N / 8) + n / 8];   // [G][N / 8], AWQ's order or in sequence
  zeros[i] = uint8_t(kind == kRepackAwq ? int(crit::kcore::awq_code(word, n & 7))
                                        : crit::kcore::gptq_zero(crit::kcore::nibble(word, n & 7), zero_offset));
  scales_out[i] = scales[(size_t)g * N + n];   // [G][N] -> [N][G]
}

}  // namespace

cudaError_t enqueue_gemv_weights(int weights, const Matrix& m, const void* x, int ldx, bool x_bf16, void* out, int ldo,
                                 bool out_fp32, int M, int N, int K, cudaStream_t stream) {
  if (M < 1 || M > kMaxRows || N < 1 || K < 8) return cudaErrorInvalidValue;
  bool ok = false;
  switch (weights) {
    case kNvfp4:
      if (K % 32 || !m.scales || !m.row_scale) return cudaErrorInvalidValue;
      ok = by_types<kNvfp4>(x_bf16, out_fp32, M, x, ldx, m, out, ldo, N, K, stream);
      break;
    case kFp8:
      if (K % 16 || !m.row_scale) return cudaErrorInvalidValue;
      ok = by_types<kFp8>(x_bf16, out_fp32, M, x, ldx, m, out, ldo, N, K, stream);
      break;
    case kMxfp8:
      if (K % 32 || !m.scales) return cudaErrorInvalidValue;
      ok = by_types<kMxfp8>(x_bf16, out_fp32, M, x, ldx, m, out, ldo, N, K, stream);
      break;
    case kBf16:
      if (K % 8) return cudaErrorInvalidValue;
      ok = by_types<kBf16>(x_bf16, out_fp32, M, x, ldx, m, out, ldo, N, K, stream);
      break;
    case kInt4Zp:
      if (m.group < 32 || m.group % 32 || K % m.group || !m.scales || !m.zeros) return cudaErrorInvalidValue;
      ok = by_types<kInt4Zp>(x_bf16, out_fp32, M, x, ldx, m, out, ldo, N, K, stream);
      break;
    default: return cudaErrorInvalidValue;
  }
  return ok ? cudaGetLastError() : cudaErrorInvalidValue;
}

cudaError_t enqueue_gemv(const void* x, int ldx, bool x_bf16, const std::uint8_t* codes, const std::uint8_t* scales,
                         bool blocked, const float* inverse, void* out, int ldo, bool out_fp32, int M, int N, int K,
                         cudaStream_t stream) {
  if (K < 32 || K % 32) return cudaErrorInvalidValue;
  Matrix m;
  m.codes = codes;
  m.scales = scales;
  m.row_scale = inverse;
  m.ct = blocked ? ((K >> 4) + 3) / 4 : 0;
  return enqueue_gemv_weights(kNvfp4, m, x, ldx, x_bf16, out, ldo, out_fp32, M, N, K, stream);
}

cudaError_t enqueue_gemv_bf16(const void* x, int ldx, const void* w, void* out, int ldo, bool out_fp32, int M, int N,
                              int K, cudaStream_t stream) {
  Matrix m;
  m.codes = static_cast<const std::uint8_t*>(w);
  return enqueue_gemv_weights(kBf16, m, x, ldx, true, out, ldo, out_fp32, M, N, K, stream);
}

cudaError_t enqueue_gemv_fp8(const void* x, int ldx, bool x_bf16, const std::uint8_t* codes, const float* row_scale,
                             const std::uint8_t* mx_scale, void* out, int ldo, bool out_fp32, int M, int N, int K,
                             cudaStream_t stream) {
  if (K < 16 || K % 16 || (mx_scale && K % 32) || (!row_scale == !mx_scale)) return cudaErrorInvalidValue;
  Matrix m;
  m.codes = codes;
  m.row_scale = row_scale;
  m.scales = mx_scale;
  return enqueue_gemv_weights(mx_scale ? kMxfp8 : kFp8, m, x, ldx, x_bf16, out, ldo, out_fp32, M, N, K, stream);
}

cudaError_t enqueue_repack_int4(int kind, const std::uint32_t* qweight, const std::uint32_t* qzeros,
                                const std::uint16_t* scales, const int* perm, int zero_offset, int N, int K, int group,
                                std::uint8_t* codes, std::uint8_t* zeros, std::uint16_t* scales_out,
                                cudaStream_t stream) {
  if ((kind != kRepackAwq && kind != kRepackGptq) || N < 8 || N % 8 || K < group || group < 8 || K % group || K % 8)
    return cudaErrorInvalidValue;
  const long long bytes = (long long)N * (K / 2), groups = (long long)N * (K / group);
  repack_codes_kernel<<<unsigned((bytes + 255) / 256), 256, 0, stream>>>(kind, qweight, perm, codes, N, K);
  repack_groups_kernel<<<unsigned((groups + 255) / 256), 256, 0, stream>>>(kind, qzeros, scales, zero_offset, zeros,
                                                                           scales_out, N, K / group);
  return cudaGetLastError();
}

int hardware_decode() {
  int* found = nullptr;
  int value = -1;
  if (cudaMalloc(&found, sizeof(int)) != cudaSuccess) return -1;
  hardware_kernel<<<1, 1>>>(found);
  if (cudaMemcpy(&value, found, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) value = -1;
  cudaFree(found);
  return value;
}

}  // namespace crit::kernels::decode_gemv
