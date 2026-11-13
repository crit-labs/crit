// The block-scaled FP4 MMA path of mma-rows: see fp4_mma.h and the kernel's comment below.
#include "fp4_mma.h"
#include "crit_tuning.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// The block-scaled FP4 mma.sync (kind::mxf4nvf4, m16n8k64) exists on the sm_120 family's specific targets.
#if defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200 || __CUDA_ARCH_SPECIFIC__ == 1210)
#define CRIT_FP4_MMA 1
#else
#define CRIT_FP4_MMA 0
#endif

namespace crit::kernels::mma_rows {
namespace {

__device__ __forceinline__ size_t blocked_offset(int n, int j, int ct) {
  const int r = n & 127;
  return ((size_t)(n >> 7) * ct + (j >> 2)) * 512 + (size_t)(r & 31) * 16 + (r >> 5) * 4 + (j & 3);
}

__device__ __forceinline__ void wait_prior() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// A sum with the row's inverse global scale, then one over the input scale.
__device__ __forceinline__ float fp4_scaled(const Problem& p, int n, float v) {
  v *= p.row_scale[n];
  v *= p.input_rcp;
  return v;
}
// A finished sum, stored in out's type.
__device__ __forceinline__ void store(const Problem& p, int tok, int n, float v) {
  const size_t at = (size_t)tok * p.ldo + n;
  v = fp4_scaled(p, n, v);
  if (p.out_kind == kOutF32) static_cast<float*>(p.out)[at] = v;
  else if (p.out_kind == kOutBf16) static_cast<__nv_bfloat16*>(p.out)[at] = __float2bfloat16_rn(v);
  else static_cast<__half*>(p.out)[at] = __float2half_rn(v);
}
// glu (mma_rows.h): gate row n and up row n + 8 finished, rounded to out's type, silu(gate) * up.
__device__ __forceinline__ void store_glu(const Problem& p, int tok, int n, float gate, float up) {
  auto rounded = [&](float v) {
    if (p.out_kind == kOutBf16) return __bfloat162float(__float2bfloat16_rn(v));
    if (p.out_kind == kOutF16) return __half2float(__float2half_rn(v));
    return v;
  };
  const float g = rounded(fp4_scaled(p, n, gate)), u = rounded(fp4_scaled(p, n + 8, up)), v = g * (1.0f / (1.0f + expf(-g))) * u;
  const size_t at = (size_t)tok * p.ldo + (n >> 4) * 8 + (n & 7);
  if (p.out_kind == kOutF32) static_cast<float*>(p.out)[at] = v;
  else if (p.out_kind == kOutBf16) static_cast<__nv_bfloat16*>(p.out)[at] = __float2bfloat16_rn(v);
  else static_cast<__half*>(p.out)[at] = __float2half_rn(v);
}

// ---- NVFP4 x NVFP4 on the block-scaled FP4 MMA ----
//
// mma.sync kind::mxf4nvf4 m16n8k64: the weights the A operand (16 matrix rows a warp), the tokens the B operand (8 a
// column, T columns), E2M1 codes with a UE4M3 scale per 16 values of K on both, FP32 sums. A lane holds, for an MMA,
// bytes 4 tig.. and 16 + 4 tig.. of its rows' 32-byte k64 block and of its token's; the scales of row gid + 8 (tig & 1)
// and of token gid as a word of four. A lane reads 16 contiguous bytes of each of its two rows a step (128 values, two
// MMAs) and a 4 x 4 word transpose in the quad gives every lane its fragments. Each output's sum runs over the MMAs in
// K order (the step's two, steps in order, `splits` contiguous parts added in part order), whatever M is.
__device__ __forceinline__ void mma_fp4(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1, uint32_t sfa, uint32_t sfb) {
#if CRIT_FP4_MMA
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0), "r"(sfb),
        "h"((uint16_t)0), "h"((uint16_t)0));
#endif
}

__device__ __forceinline__ uint32_t word_of(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// The quad's 4 x 4 word transpose: lane t gets word t of quad lane s as y[s].
__device__ __forceinline__ void quad_transpose(const uint4& v, int t, int lane, uint32_t (&y)[4]) {
  const uint32_t own = word_of(v, t);
  y[0] = y[1] = y[2] = y[3] = own;
#pragma unroll
  for (int r = 1; r < 4; ++r) {
    const uint32_t got = __shfl_sync(0xffffffffu, word_of(v, (t + r) & 3), (lane & ~3) | ((t - r) & 3));
    const int s = (t - r) & 3;
    y[0] = s == 0 ? got : y[0];
    y[1] = s == 1 ? got : y[1];
    y[2] = s == 2 ? got : y[2];
    y[3] = s == 3 ? got : y[3];
  }
}

// U: steps of weights a warp loads together (kernel.json, structure fp4-tile, knob HOIST; kFp4Hoist untuned). It
// changes when the loads run, never the order of the MMAs.
template <int T, int U>
__global__ void __launch_bounds__(kMaxWarps * 32, 4) fp4_kernel(const Problem p) {
  __shared__ int last;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int tile = blockIdx.x, split = blockIdx.y, rows = tile_rows(1, p.warps);
  const int r0 = tile * rows + warp * 16;
  const int steps = p.K / kFp4Step;
  const int s_lo = split * steps / p.splits, s_hi = (split + 1) * steps / p.splits;
  const size_t row_bytes = (size_t)p.K / 2;
  const uint8_t* wa = p.w + (size_t)min(r0 + gid, p.N - 1) * row_bytes + 16 * tig;   // a row past N reads row N - 1
  const uint8_t* wb = p.w + (size_t)min(r0 + gid + 8, p.N - 1) * row_bytes + 16 * tig;
  const int n_scale = min(r0 + gid + 8 * (tig & 1), p.N - 1);   // the row whose scales this lane gives the MMA
  const uint8_t* xq = p.scratch;
  const uint8_t* xs = p.scratch + ((size_t)p.M * row_bytes + 15) / 16 * 16;
  const int ct = (p.K / 16 + 3) / 4;
  float c[T][4];
#pragma unroll
  for (int t = 0; t < T; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[t][i] = 0.f;
  uint4 va[U], vb[U];
  auto load = [&](int st0) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int st = min(st0 + u, s_hi - 1);
      va[u] = __ldg(reinterpret_cast<const uint4*>(wa + (size_t)st * 64));
      vb[u] = __ldg(reinterpret_cast<const uint4*>(wb + (size_t)st * 64));
    }
  };
  if (s_lo < s_hi) load(s_lo);   // weights do not depend on the previous kernel (the pre-pass): read them first
  wait_prior();
  for (int st0 = s_lo; st0 < s_hi; st0 += U) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int st = st0 + u;
      if (st >= s_hi) break;
      uint32_t ya[4], yb[4];
      quad_transpose(va[u], tig, lane, ya);
      quad_transpose(vb[u], tig, lane, yb);
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int j = st * 8 + 4 * m;   // the k64 block's first scale column
        const uint32_t sfa = __ldg(reinterpret_cast<const uint32_t*>(p.scales + blocked_offset(n_scale, j, p.ct)));
#pragma unroll
        for (int t = 0; t < T; ++t) {
          const int tok = 8 * t + gid;
          uint32_t b0 = 0u, b1 = 0u, sfb = 0u;
          if (tok < p.M) {
            const uint8_t* xr = xq + (size_t)tok * row_bytes + (size_t)st * 64 + 32 * m;
            b0 = *reinterpret_cast<const uint32_t*>(xr + 4 * tig);
            b1 = *reinterpret_cast<const uint32_t*>(xr + 16 + 4 * tig);
            sfb = *reinterpret_cast<const uint32_t*>(xs + blocked_offset(tok, j, ct));
          }
          mma_fp4(c[t], ya[2 * m], yb[2 * m], ya[2 * m + 1], yb[2 * m + 1], b0, b1, sfa, sfb);
        }
      }
    }
    if (st0 + U < s_hi) load(st0 + U);
  }

  // c[t][i]: matrix row r0 + gid + 8 (i >> 1), token 8 t + 2 tig + (i & 1).
  if (p.splits == 1) {
    if (p.glu) {
#pragma unroll
      for (int t = 0; t < T; ++t)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int n = r0 + gid, tok = 8 * t + 2 * tig + i;
          if (n < p.N && tok < p.M) store_glu(p, tok, n, c[t][i], c[t][i + 2]);
        }
      return;
    }
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = r0 + gid + (i >> 1) * 8, tok = 8 * t + 2 * tig + (i & 1);
        if (n < p.N && tok < p.M) store(p, tok, n, c[t][i]);
      }
    return;
  }
  float* part = p.partial + (size_t)split * p.M * p.N;
#pragma unroll
  for (int t = 0; t < T; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int n = r0 + gid + (i >> 1) * 8, tok = 8 * t + 2 * tig + (i & 1);
      if (n < p.N && tok < p.M) __stcg(part + (size_t)tok * p.N + n, c[t][i]);
    }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(p.counters + tile, 1) == p.splits - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (threadIdx.x == 0) p.counters[tile] = 0;
  const int first = tile * rows;
  if (p.glu) {   // one thread a (gate, up) pair, every lane busy: gate rows n % 16 < 8, each up row 8 past it
    const int pairs = rows / 2;
    for (int i = threadIdx.x; i < p.M * pairs; i += blockDim.x) {
      const int tok = i / pairs, h = i - tok * pairs, n = first + (h >> 3) * 16 + (h & 7);
      if (n >= p.N) continue;
      float v = __ldcg(p.partial + (size_t)tok * p.N + n), u = __ldcg(p.partial + (size_t)tok * p.N + n + 8);
      for (int s = 1; s < p.splits; ++s) {
        v += __ldcg(p.partial + ((size_t)s * p.M + tok) * p.N + n);
        u += __ldcg(p.partial + ((size_t)s * p.M + tok) * p.N + n + 8);
      }
      store_glu(p, tok, n, v, u);
    }
    return;
  }
  for (int i = threadIdx.x; i < p.M * rows; i += blockDim.x) {
    const int tok = i / rows, n = first + (i - tok * rows);
    if (n >= p.N) continue;
    float v = __ldcg(p.partial + (size_t)tok * p.N + n);
    for (int s = 1; s < p.splits; ++s) v += __ldcg(p.partial + ((size_t)s * p.M + tok) * p.N + n);
    store(p, tok, n, v);
  }
}

__global__ void fp4_mma_kernel(int* found) { *found = CRIT_FP4_MMA; }

template <int T, int U>
cudaError_t launch_fp4(const Problem& p, cudaStream_t stream) {
  int major = 0, device = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaLaunchAttribute attribute[1];
  cudaLaunchConfig_t config{};
  config.gridDim = dim3((p.N + tile_rows(1, p.warps) - 1) / tile_rows(1, p.warps), p.splits);
  config.blockDim = dim3(32 * p.warps);
  config.stream = stream;
  if (major >= 9) {   // the quantizer before it lets it start: the weights load while it runs
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attribute;
    config.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&config, fp4_kernel<T, U>, p);
}

template <int U>
cudaError_t by_rows(const Problem& p, cudaStream_t stream) {
  if (p.M <= 8) return launch_fp4<1, U>(p, stream);
  if (p.M <= 16) return launch_fp4<2, U>(p, stream);
#ifdef CRIT_SWEEP_BUILD
  return cudaErrorInvalidValue;   // a sweep build calls up to 16 rows (the bench's tune_rows)
#else
  if (p.M <= 32) return launch_fp4<4, U>(p, stream);
  return launch_fp4<8, U>(p, stream);
#endif
}

}  // namespace

cudaError_t enqueue_fp4(const Problem& given, cudaStream_t stream) {
  Problem p = given;
  if (p.M < 1 || p.M > kMaxRows || p.N < 1 || p.warps < 1 || p.warps > kMaxWarps || p.splits < 1 ||
      p.splits > kMaxSplits || !p.scratch || p.weights != kNvfp4 || !p.ct || p.K % kFp4Step ||
      p.splits > p.K / kFp4Step || (p.splits > 1 && (!p.partial || !p.counters)))
    return cudaErrorInvalidValue;
  uint8_t* codes = p.scratch;
  uint8_t* scales = p.scratch + ((size_t)p.M * p.K / 2 + 15) / 16 * 16;
  const cudaError_t e = enqueue_quantize_nvfp4(p.x, p.ldx, p.M, p.K, p.input_scale, p.input_rcp, codes, scales, stream);
  if (e != cudaSuccess) return e;
  // The configuration crit_tuning.h's table names for this matrix and row count (0, the defaults, where it names
  // none); warps and splits come with p (plan()'s, from the same table).
  switch (crit_lookup_fp4_tile(p.N, p.K, p.M)) {
#define CRIT_FP4_CONFIG(index, warps, splits, hoist) \
  case index:                                        \
    return by_rows<hoist>(p, stream);
    CRIT_CONFIGS_fp4_tile(CRIT_FP4_CONFIG)
#undef CRIT_FP4_CONFIG
    default: return cudaErrorInvalidValue;
  }
}

bool fp4_mma(cudaStream_t stream) {
  static int found[64] = {};   // per device: 0 unknown, 1 yes, 2 no
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) return false;
  if (!found[device]) {
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    int value = 0;
    if (major == 12) {
      int* d = nullptr;
      if (cudaMalloc(&d, sizeof(int)) == cudaSuccess) {
        fp4_mma_kernel<<<1, 1, 0, stream>>>(d);
        if (cudaMemcpyAsync(&value, d, sizeof(int), cudaMemcpyDeviceToHost, stream) == cudaSuccess)
          cudaStreamSynchronize(stream);
        cudaFree(d);
      }
    }
    found[device] = value == 1 ? 1 : 2;
  }
  return found[device] == 1;
}

int fp4_ctas() {
  int n = 0;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fp4_kernel<1, kFp4Hoist>, kMaxWarps * 32, 0) == cudaSuccess
             ? n
             : 0;
}

}  // namespace crit::kernels::mma_rows
