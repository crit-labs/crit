// The device parts mma-rows' two tiles share (mma_rows.cu's decode tile for 1 to 64 rows, prompt_tile.cu's prompt
// tile for any number of rows): the value decoders, a step's weights in registers (Row), the activations' B
// fragments, the MMA, programmatic dependent launch, the quantizers' value functions and a finished sum's scales.
// One definition, so both tiles feed every MMA the same operands and finish every sum with the same instructions.
//
// The decoders, the block-scale layout, the MMA, cp.async, programmatic dependent launch and the NaN-propagating
// maximum are the kernel core's (kernels/core: crit_kcore/formats.cuh, mma.cuh, load.cuh, reduce.cuh), the same
// instructions as this file held before the core existed, so every output keeps its bits.
#pragma once
#include "mma_rows.h"

#include "crit_kcore/formats.cuh"
#include "crit_kcore/load.cuh"
#include "crit_kcore/mma.cuh"
#include "crit_kcore/reduce.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace crit::kernels::mma_rows {
namespace {

// fp32(1 / 448) and fp32(1 / 6): the quantizers multiply by them, as a division by a constant is done in torch.
constexpr float kRcp448 = 1.0f / 448.0f;
constexpr float kRcp6 = 1.0f / 6.0f;

using crit::kcore::bf16_hi;
using crit::kcore::bf16_lo;
using crit::kcore::scale_offset;
using crit::kcore::word_of;
__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) { return crit::kcore::pack_bf16x2(lo, hi); }

using crit::kcore::nanmax;   // the larger of two values, NaN if either is (as torch's amax)

__device__ __forceinline__ float e4m3(uint32_t bits) { return crit::kcore::e4m3_cvt(bits); }
__device__ __forceinline__ uint32_t e4m3_bits(float v) { return crit::kcore::e4m3_encode_satfinite(v); }

// Two E4M3 codes (the low 16 bits, low byte first) as a BF16 pair, times `scale` (a power of two for MXFP8).
template <bool SCALED>
__device__ __forceinline__ uint32_t fp8_pair(uint32_t two, float scale) {
  const float2 f = __half22float2(crit::kcore::e4m3x2_to_half2(two));
  return SCALED ? bf16x2(f.x * scale, f.y * scale) : bf16x2(f.x, f.y);
}

// One matrix row's kLaneBytes contiguous bytes a lane reads a step (the four lanes of a quad read 4 x kLaneBytes of the
// row in a run), in SUBS sub-steps of 32 values: the step from `base` covers [base, base + 128 SUBS), the lane of quad
// position tig its 32 SUBS values from base + 32 SUBS tig, sub-step q the 32 from there + 32 q. pairs(q, out)
// gives sub-step q's 32 values as 16 BF16 pairs: MMA s takes pairs 2 s (values 4 s, + 1) and 2 s + 1 (4 s + 2, + 3). A sub-step past K
// loads nothing and multiplies as zeros (K is a multiple of 32).
template <int WF>
struct Row;

__device__ __forceinline__ uint4 load16(const uint8_t* p, bool valid) {
  return valid ? __ldg(reinterpret_cast<const uint4*>(p)) : make_uint4(0u, 0u, 0u, 0u);
}

template <>
struct Row<kNvfp4> {
  static constexpr int SUBS = kLaneBytes / 16;
  uint4 c[SUBS];
  uint32_t sc[SUBS];   // each sub-step's two E4M3 block scales
  __device__ __forceinline__ void load(const Problem& p, int n, int base, int tig) {
    const uint8_t* row = p.w + (size_t)n * (p.K >> 1);
#pragma unroll
    for (int q = 0; q < SUBS; ++q) {
      const int k = base + 32 * (SUBS * tig + q);
      const bool inside = k < p.K;
      c[q] = load16(row + (k >> 1), inside);
      sc[q] = inside ? __ldg(reinterpret_cast<const uint16_t*>(p.scales + scale_offset(n, k >> 4, p.K >> 4, p.ct)))
                     : 0u;
    }
  }
  // Word w (0..3) of sub-step q: its 8 values as BF16 pairs 4w .. 4w + 3 of the sub-step (MMAs 2w and 2w + 1).
  __device__ __forceinline__ void quad(int q, int w, uint32_t* out) const {
    const float s = e4m3(sc[q] >> (8 * (w >> 1)));
    const __half2 h = __float2half2_rn(s);
    crit::kcore::e2m1x8_to_bf16x2_scaled(word_of(c[q], w), s, *reinterpret_cast<const uint32_t*>(&h), out);
  }
  // Sub-step q's 32 values as 16 BF16 pairs (pair i: values 2i, 2i + 1).
  __device__ __forceinline__ void pairs(int q, uint32_t (&out)[16]) const {
    const float s0 = e4m3(sc[q]), s1 = e4m3(sc[q] >> 8);
    const __half2 h0 = __float2half2_rn(s0), h1 = __float2half2_rn(s1);
    crit::kcore::e2m1x8_to_bf16x2_scaled(c[q].x, s0, *reinterpret_cast<const uint32_t*>(&h0), out);
    crit::kcore::e2m1x8_to_bf16x2_scaled(c[q].y, s0, *reinterpret_cast<const uint32_t*>(&h0), out + 4);
    crit::kcore::e2m1x8_to_bf16x2_scaled(c[q].z, s1, *reinterpret_cast<const uint32_t*>(&h1), out + 8);
    crit::kcore::e2m1x8_to_bf16x2_scaled(c[q].w, s1, *reinterpret_cast<const uint32_t*>(&h1), out + 12);
  }
};

template <>
struct Row<kFp8> {
  static constexpr int SUBS = kLaneBytes / 32;
  uint4 c[2 * SUBS];
  __device__ __forceinline__ void load(const Problem& p, int n, int base, int tig) {
    const uint8_t* row = p.w + (size_t)n * p.K;
#pragma unroll
    for (int q = 0; q < SUBS; ++q) {
      const int k = base + 32 * (SUBS * tig + q);
      c[2 * q] = load16(row + k, k < p.K);
      c[2 * q + 1] = load16(row + k + 16, k < p.K);
    }
  }
  __device__ __forceinline__ void quad(int q, int w, uint32_t* out) const {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int i = 2 * w + j;
      const uint32_t v = word_of(c[2 * q + (i >> 2)], i & 3);
      out[2 * j] = fp8_pair<false>(v, 1.f);
      out[2 * j + 1] = fp8_pair<false>(v >> 16, 1.f);
    }
  }
  __device__ __forceinline__ void pairs(int q, uint32_t (&out)[16]) const {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t w = word_of(c[2 * q + (i >> 2)], i & 3);
      out[2 * i] = fp8_pair<false>(w, 1.f);
      out[2 * i + 1] = fp8_pair<false>(w >> 16, 1.f);
    }
  }
};

template <>
struct Row<kMxfp8> {
  static constexpr int SUBS = kLaneBytes / 32;
  uint4 c[2 * SUBS];
  uint32_t mx[SUBS];   // each sub-step's E8M0 exponent
  __device__ __forceinline__ void load(const Problem& p, int n, int base, int tig) {
    const uint8_t* row = p.w + (size_t)n * p.K;
#pragma unroll
    for (int q = 0; q < SUBS; ++q) {
      const int k = base + 32 * (SUBS * tig + q);
      c[2 * q] = load16(row + k, k < p.K);
      c[2 * q + 1] = load16(row + k + 16, k < p.K);
      mx[q] = k < p.K ? __ldg(p.scales + (size_t)n * (p.K >> 5) + (k >> 5)) : 127u;
    }
  }
  __device__ __forceinline__ void quad(int q, int w, uint32_t* out) const {
    const float scale = crit::kcore::e8m0_value(mx[q]);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int i = 2 * w + j;
      const uint32_t v = word_of(c[2 * q + (i >> 2)], i & 3);
      out[2 * j] = fp8_pair<true>(v, scale);
      out[2 * j + 1] = fp8_pair<true>(v >> 16, scale);
    }
  }
  __device__ __forceinline__ void pairs(int q, uint32_t (&out)[16]) const {
    const float scale = crit::kcore::e8m0_value(mx[q]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t w = word_of(c[2 * q + (i >> 2)], i & 3);
      out[2 * i] = fp8_pair<true>(w, scale);
      out[2 * i + 1] = fp8_pair<true>(w >> 16, scale);
    }
  }
};

template <>
struct Row<kBf16> {
  static constexpr int SUBS = kLaneBytes / 64;
  uint4 c[4 * SUBS];
  __device__ __forceinline__ void load(const Problem& p, int n, int base, int tig) {
#pragma unroll
    for (int q = 0; q < SUBS; ++q) {
      const int k = base + 32 * (SUBS * tig + q);
      const uint8_t* at = p.w + ((size_t)n * p.K + k) * 2;
#pragma unroll
      for (int i = 0; i < 4; ++i) c[4 * q + i] = load16(at + 16 * i, k < p.K);
    }
  }
  __device__ __forceinline__ void quad(int q, int w, uint32_t* out) const {
    out[0] = c[4 * q + w].x;
    out[1] = c[4 * q + w].y;
    out[2] = c[4 * q + w].z;
    out[3] = c[4 * q + w].w;
  }
  __device__ __forceinline__ void pairs(int q, uint32_t (&out)[16]) const {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      out[4 * i] = c[4 * q + i].x;
      out[4 * i + 1] = c[4 * q + i].y;
      out[4 * i + 2] = c[4 * q + i].z;
      out[4 * i + 3] = c[4 * q + i].w;
    }
  }
};

// INT4 with a zero point a group (kInt4Zp): codes as NVFP4's (32 values a 16-byte sub-step, which lies in one group:
// group a multiple of 32), each sub-step's FP16 scale and zero point. A value (code - zero) * scale needs up to 15
// significant bits: the kernel core's split decoder gives it as hi + lo, two BF16 values, and every MMA of the weights
// runs twice, hi then lo, with the same activations (kSplit), so each product enters the FP32 sum exactly.
template <int WF>
constexpr bool kSplit = WF == kInt4Zp;

template <>
struct Row<kInt4Zp> {
  static constexpr int SUBS = kLaneBytes / 16;
  uint4 c[SUBS];
  uint32_t sz[SUBS];   // each sub-step's FP16 scale bits, its zero point at bits 16 and up
  __device__ __forceinline__ void load(const Problem& p, int n, int base, int tig) {
    const uint8_t* row = p.w + (size_t)n * (p.K >> 1);
    const int groups = p.K / p.group;
#pragma unroll
    for (int q = 0; q < SUBS; ++q) {
      const int k = base + 32 * (SUBS * tig + q);
      const bool inside = k < p.K;
      c[q] = load16(row + (k >> 1), inside);
      const size_t at = (size_t)n * groups + (inside ? k / p.group : 0);
      sz[q] = inside ? uint32_t(__ldg(reinterpret_cast<const uint16_t*>(p.scales) + at)) |
                           (uint32_t(__ldg(p.zeros + at)) << 16)
                     : 0u;
    }
  }
  // Word w (0..3) of sub-step q as BF16 pairs 4w .. 4w + 3 of the sub-step, hi and lo parts.
  __device__ __forceinline__ void split_quad(int q, int w, uint32_t* hi, uint32_t* lo) const {
    crit::kcore::int4x8_zp_to_bf16x2_split(word_of(c[q], w), int(sz[q] >> 16), crit::kcore::f16_value(sz[q]), hi, lo);
  }
  // Sub-step q's 32 values as 16 BF16 pairs (pair i: values 2i, 2i + 1), hi and lo parts.
  __device__ __forceinline__ void split_pairs(int q, uint32_t (&hi)[16], uint32_t (&lo)[16]) const {
#pragma unroll
    for (int w = 0; w < 4; ++w) split_quad(q, w, hi + 4 * w, lo + 4 * w);
  }
};

// A token's B fragments for a sub-step: its 32 values [kb, kb + 32) as 16 BF16 pairs (word i: values 2i, 2i + 1);
// zeros for a token past M or a sub-step past K.
__device__ __forceinline__ void act_frags(const Problem& p, const __nv_bfloat16* x, int tok, int kb, bool inside,
                                          uint32_t (&b)[16]) {
  if (tok >= p.M || !inside) {
#pragma unroll
    for (int i = 0; i < 16; ++i) b[i] = 0u;
    return;
  }
  const uint4* q = reinterpret_cast<const uint4*>(x + (size_t)tok * p.ldx + kb);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 v = __ldg(q + i);
    b[4 * i] = v.x;
    b[4 * i + 1] = v.y;
    b[4 * i + 2] = v.z;
    b[4 * i + 3] = v.w;
  }
}

__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  crit::kcore::mma_bf16_m16n8k16(c, a0, a1, a2, a3, b0, b1);
}

// cp.async (8.0 on): 16 bytes global -> shared through L2 (zeros when !valid), and its groups.
using crit::kcore::cp_async16;
__device__ __forceinline__ void cp_commit() { crit::kcore::cp_async_commit(); }
template <int N>
__device__ __forceinline__ void cp_wait() { crit::kcore::cp_async_wait<N>(); }

// Programmatic dependent launch (compute capability 9.0 on; nothing below): wait for the previous kernel in the stream
// before reading what it wrote, and let the next one start.
__device__ __forceinline__ void wait_prior() { crit::kcore::griddep_wait(); }
__device__ __forceinline__ void launch_next() { crit::kcore::griddep_launch_dependents(); }

// ---- the activations' quantizers (one definition: the pre-passes and the quantize ops) ----

__device__ __forceinline__ float fp8_value(float v, float scale) {
  v = __fdiv_rn(v, scale);
  v = v > 448.f ? 448.f : v < -448.f ? -448.f : v;
  return e4m3(e4m3_bits(v));
}

// A 16-value block (8 BF16 pairs) in NVFP4 under input scale g: its E4M3 scale bits and E2M1 codes (low nibble
// first: element 2i in byte i's low nibble, the 8 bytes in two words).
__device__ __forceinline__ uint32_t nvfp4_block(const uint32_t (&w)[8], float g, float g_rcp, uint32_t& lo, uint32_t& hi) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) amax = nanmax(nanmax(amax, fabsf(bf16_lo(w[i]))), fabsf(bf16_hi(w[i])));
  float bs = (amax * kRcp6) * g;
  bs = bs > 448.f ? 448.f : bs;
  const uint32_t sbits = e4m3_bits(bs);
  float d = e4m3(sbits) * g_rcp;
  const bool zero = !(d > 0.f);
  d = d < 1e-30f ? 1e-30f : d;
  lo = hi = 0u;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t c0 = crit::kcore::e2m1_code_signed(zero ? 0.f : __fdiv_rn(bf16_lo(w[i]), d));
    const uint32_t c1 = crit::kcore::e2m1_code_signed(zero ? 0.f : __fdiv_rn(bf16_hi(w[i]), d));
    const uint32_t byte = c0 | (c1 << 4);
    if (i < 4) lo |= byte << (8 * i);
    else hi |= byte << (8 * (i - 4));
  }
  return sbits;
}

// A sum with its scales applied in a fixed order.
template <int WF, int ACT>
__device__ __forceinline__ float scaled(const Problem& p, int tok, int n, float v) {
  if constexpr (WF == kNvfp4 || WF == kFp8) v *= p.row_scale[n];
  if constexpr (ACT == kActNvfp4) v *= p.input_rcp;
  if constexpr (ACT == kActFp8Rows) v *= p.act_scale[tok];
  return v;
}
// v rounded to out's type (as a store and a load give it back).
__device__ __forceinline__ float out_rounded(const Problem& p, float v) {
  if (p.out_kind == kOutBf16) return __bfloat162float(__float2bfloat16_rn(v));
  if (p.out_kind == kOutF16) return __half2float(__float2half_rn(v));
  return v;
}
__device__ __forceinline__ void store_out(const Problem& p, size_t at, float v) {
  if (p.out_kind == kOutF32) static_cast<float*>(p.out)[at] = v;
  else if (p.out_kind == kOutBf16) static_cast<__nv_bfloat16*>(p.out)[at] = __float2bfloat16_rn(v);
  else static_cast<__half*>(p.out)[at] = __float2half_rn(v);
}
// A finished sum: its scales applied in a fixed order, then stored in out's type.
template <int WF, int ACT>
__device__ __forceinline__ void finish(const Problem& p, int tok, int n, float v) {
  store_out(p, (size_t)tok * p.ldo + n, scaled<WF, ACT>(p, tok, n, v));
}
// The glu output of gate row n (n % 16 < 8 of the interleaved matrix) and its up row n + 8: each finished and rounded
// to out's type, then silu(gate) * up, stored at column (n / 16) * 8 + n % 8.
__device__ __forceinline__ void store_glu(const Problem& p, int tok, int n, float gate, float up) {
  const float g = out_rounded(p, gate), u = out_rounded(p, up);
  store_out(p, (size_t)tok * p.ldo + (n >> 4) * 8 + (n & 7), g * (1.0f / (1.0f + expf(-g))) * u);
}
template <int WF, int ACT>
__device__ __forceinline__ void finish_glu(const Problem& p, int tok, int n, float gate, float up) {
  store_glu(p, tok, n, scaled<WF, ACT>(p, tok, n, gate), scaled<WF, ACT>(p, tok, n + 8, up));
}

// A launch on `stream`, as the programmatic dependent of the kernel before it where the device has it (9.0 on).
template <class... Params, class... Args>
cudaError_t launch_kernel(void (*kernel)(Params...), dim3 grid, dim3 block, cudaStream_t stream, Args... args) {
  return crit::kcore::launch_programmatic(kernel, grid, block, 0, stream, args...);
}

}  // namespace
}  // namespace crit::kernels::mma_rows
