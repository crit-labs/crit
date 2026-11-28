// The block-scaled FP4 MMA path of mma-rows: see fp4_mma.h and the kernel's comment below.
#include "fp4_mma.h"
#include "crit_tuning.h"
#include "prompt_parts.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

#include "crit_kcore/arch.cuh"
#include "crit_kcore/device.cuh"
#include "crit_kcore/formats.cuh"
#include "crit_kcore/load.cuh"
#include "crit_kcore/mma.cuh"
#include "crit_kcore/tile.cuh"

// The block-scaled FP4 mma.sync (kind::mxf4nvf4, m16n8k64) exists on the sm_120 family's specific targets
// (arch.cuh's CRIT_KCORE_HAS_BLOCK_SCALED_MMA); its wrappers, the quad's word transpose and cuBLASLt's blocked scale
// offsets are the kernel core's (mma.cuh, formats.cuh).
#define CRIT_FP4_MMA CRIT_KCORE_HAS_BLOCK_SCALED_MMA

namespace crit::kernels::mma_rows {
namespace {

namespace kc = crit::kcore;
using kc::quad_transpose;

__device__ __forceinline__ size_t blocked_offset(int n, int j, int ct) { return kc::blocked_scale_offset(n, j, ct); }

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
__device__ __forceinline__ float glu_value(const Problem& p, int n, float gate, float up) {
  auto rounded = [&](float v) {
    if (p.out_kind == kOutBf16) return __bfloat162float(__float2bfloat16_rn(v));
    if (p.out_kind == kOutF16) return __half2float(__float2half_rn(v));
    return v;
  };
  const float g = rounded(fp4_scaled(p, n, gate)), u = rounded(fp4_scaled(p, n + 8, up));
  return g * (1.0f / (1.0f + expf(-g))) * u;
}
__device__ __forceinline__ void store_glu(const Problem& p, int tok, int n, float gate, float up) {
  const float v = glu_value(p, n, gate, up);
  const size_t at = (size_t)tok * p.ldo + (n >> 4) * 8 + (n & 7);
  if (p.out_kind == kOutF32) static_cast<float*>(p.out)[at] = v;
  else if (p.out_kind == kOutBf16) static_cast<__nv_bfloat16*>(p.out)[at] = __float2bfloat16_rn(v);
  else static_cast<__half*>(p.out)[at] = __float2half_rn(v);
}

// A value's 16 bits in a BF16 or F16 out.
__device__ __forceinline__ uint16_t out_bits16(const Problem& p, float v) {
  return p.out_kind == kOutBf16 ? __bfloat16_as_ushort(__float2bfloat16_rn(v)) : __half_as_ushort(__float2half_rn(v));
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
  kc::mma_nvf4_m16n8k64(c, a0, a1, a2, a3, b0, b1, sfa, sfb);
}
// The same MMA into c from zero: a chain's first (c need not hold zeros before it).
__device__ __forceinline__ void mma_fp4_first(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  kc::mma_nvf4_m16n8k64_zero(c, a0, a1, a2, a3, b0, b1, sfa, sfb);
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


// ---- the FP4 prompt tile: any number of rows, every output with fp4_kernel's bits ----
//
// An output's sum in fp4_kernel is one chain of block-scaled m16n8k64 MMAs per split of K's steps, from zero: each
// step's two MMAs in order (m = 0, 1), the weights the A operand (matrix row n at A row n mod 16: a0 and a2 the row's
// bytes 4 tig.. of the k64 block's two 16-byte chunks, a1 and a3 those of row n + 8), the token the B operand (its
// bytes 4 tig.. of the two chunks), each operand's four scale bytes of the block; the splits' chains added in split
// order; then store(). This tile runs that sequence for every output, so a prompt row has the bits the same row gets
// as a decode row, while a CTA reuses each weight for PT tokens as a GEMM does.
//
// The data movement is the warp-specialized one of CUTLASS's TMA GEMMs (BSD-3-Clause; the mechanisms, written from
// the PTX ISA on the kernel core's blocks): a ring of shared-memory stages, each one step (128 values) of the CTA's
// 16 PW PR matrix rows and of its PT tokens (two tensor-map tile copies, swizzled as tile.cuh's offset<64>, rows past
// the operands zero) and of both operands' scale bytes (bulk copies of their 128-row tiles), completing on the
// stage's full mbarrier. A producer warp issues a stage once the PW consumer warps have released it; each consumer
// (PR tiles of 16 matrix rows over all the CTA's tokens) reads its fragments by ldmatrix at fixed offsets, a pair of
// token columns ahead of their MMAs and the next step's weights during a step's last MMAs. A CTA runs its splits of
// K one after another, with a running total an output: a chain's first MMA of a split starts from zero right after
// the chain before it joined its total, so a split's boundary costs the MMAs no pause. With partial sums and
// counters, a tile's splits run in several CTAs (grid z, knob PK; prompt_parts.h): part 0 keeps its running total
// and stores it, every later part stores each of its splits' chains on its own, and the tile's last CTA adds them in
// split order, the same additions. Whole tiles of 16-bit outputs leave through the ring in 16-byte stores. CTAs of one row
// tile are neighbours in the grid (the tokens along x), so they read its weights from L2.
template <bool SPLIT, int PW, int PR, int PT>
struct PromptTile {
  static constexpr int kRows = 16 * PW * PR, kGroups = PT / 8, kThreads = 32 * (PW + 1);
  static_assert(PT == 32 || PT == 64 || PT == 128, "a CTA's tokens within one scale tile, in pairs of columns");
  // A stage: weight codes [kRows][64], token codes [PT][64] (swizzled tiles), then the weights' 128-row scale tiles
  // under the CTA's rows (two 512-byte column tiles a step each) and the tokens' one.
  static constexpr int kScaleTiles = 128 % kRows == 0 ? 1 : kRows % 128 == 0 ? kRows / 128 : (kRows + 126) / 128 + 1;
  static constexpr bool kAhead = true;   // the next step's weight fragments read during a step's last MMAs
  static constexpr int kW = 0, kX = kRows * 64, kSw = kX + PT * 64, kSx = kSw + kScaleTiles * 1024;
  static constexpr int kStage = kSx + 1024;                      // a multiple of 1024 (the TMA swizzle's period)
  // With splits of K a thread keeps a running total beside each chain: those of its first kGroups / 2 token columns
  // in registers, the others' in shared memory after the barriers, [total of the thread][thread] (a warp's lanes side
  // by side). All in registers they leave too few for the chains and their fragments (168 a thread with 8 consumer
  // warps beside the producer), all in shared memory too little of its 99 KB for the ring.
  static constexpr int kHeld = kGroups / 2;
  static constexpr int kTotalBytes = SPLIT ? PR * (kGroups - kHeld) * 4 * (32 * PW) * 4 : 0;
  static constexpr int kFit = (101376 - 1024 - 256 - kTotalBytes) / kStage;   // 99 KB less the alignment and the barriers
  static constexpr int kStages = kFit < 4 ? kFit : 4;
  static_assert(kStages >= 2, "two stages within a CTA's shared memory");
  static constexpr int kBar = kStages * kStage, kTotals = kBar + 2 * kStages * 8;
  static constexpr int kSmem = 1024 + kTotals + kTotalBytes;
  // A whole tile's 16-bit outputs on their way out: a row of kRows (+ 8: the lanes' stores apart) a token, in the
  // ring once every stage is read.
  static constexpr int kOutLd = kRows + 8;
  static_assert(PT * kOutLd * 2 <= kBar, "a tile of 16-bit outputs within the ring");
};

// The tensor maps of one launch (made on the host): the matrix's codes and the quantized tokens', boxes of a CTA's
// rows x 64 bytes.
struct alignas(64) PromptMaps {
  CUtensorMap w, x;
};

// A barrier of the consumer warps alone (the producer has left).
__device__ __forceinline__ void prompt_consumers_sync(int threads) {
  asm volatile("bar.sync 1, %0;\n" ::"r"(threads) : "memory");
}

// A row's scale word of a step's k64 block 0 in a staged 128-row scale tile (block 1: 512 bytes on).
__device__ __forceinline__ uint32_t staged_scale(int r) {
  return uint32_t((r >> 7) * 1024 + (r & 31) * 16 + ((r >> 5) & 3) * 4);
}

// One CTA: PW consumer warps x PR tiles of 16 matrix rows (blockIdx.y) times PT tokens (blockIdx.x) over part
// blockIdx.z of K's splits, and a producer warp. SPLIT: the plan has more than one split of K (a running total beside
// each split's chain); PARTS: the splits run in gridDim.z parts (an instance of its own, so the one-part kernel's
// registers stay as they were). PW, PR, PT and PK are speed knobs (kernel.json, structure fp4-prompt-tile): no
// output's chain depends on them.
template <bool SPLIT, bool PARTS, int PW, int PR, int PT>
__global__ void __launch_bounds__(32 * (PW + 1), 1)
    fp4_prompt_kernel(const Problem p, const __grid_constant__ PromptMaps maps) {
  using C = PromptTile<SPLIT, PW, PR, PT>;
  constexpr int G = C::kGroups, ST = C::kStages, H = C::kHeld;
  extern __shared__ __align__(128) unsigned char prompt_smem[];
#if CRIT_FP4_MMA && CRIT_KCORE_HAS_TMA
  unsigned char* base = prompt_smem + ((1024u - (kc::smem_u32(prompt_smem) & 1023u)) & 1023u);
  uint64_t* full = reinterpret_cast<uint64_t*>(base + C::kBar);
  uint64_t* empty = full + ST;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3, quarter = lane >> 3;
  const int t0 = blockIdx.x * PT, n0 = blockIdx.y * C::kRows;
  const int steps = p.K / kFp4Step;
  const int parts = PARTS ? (int)gridDim.z : 1, part = PARTS ? (int)blockIdx.z : 0;
  const int sp_lo = part * p.splits / parts, sp_hi = (part + 1) * p.splits / parts;   // this CTA's splits
  const int st_lo = sp_lo * steps / p.splits, st_hi = sp_hi * steps / p.splits;     // and steps
  __shared__ int last;
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) {
      kc::mbarrier_init(full + s, 1);
      kc::mbarrier_init(empty + s, PW);
    }
    kc::fence_barrier_init();
  }
  __syncthreads();
  wait_prior();   // the quantizer before this kernel writes the tokens' codes and scales

  if (warp == PW) {
    // The producer: every step in order, each into its ring slot once the consumers have released the slot.
    if (lane != 0) return;
    const size_t row_bytes = (size_t)p.K / 2;
    const uint8_t* xs = p.scratch + ((size_t)p.M * row_bytes + 15) / 16 * 16;
    const int ct = (p.K / 16 + 3) / 4;
    // The matrix's scale tiles under the CTA's rows.
    const int tiles = min((p.N + 127) / 128, (n0 + C::kRows + 127) / 128) - (n0 >> 7);
    const uint32_t bytes = uint32_t((C::kRows + PT) * 64 + (tiles + 1) * 1024);
    for (int st = st_lo; st < st_hi; ++st) {   // the ring counts the CTA's steps from its first
      const int r = st - st_lo, s = r % ST;
      if (r >= ST) kc::mbarrier_wait(empty + s, uint32_t((r / ST - 1) & 1));
      unsigned char* stage = base + s * C::kStage;
      kc::mbarrier_arrive_expect_tx(full + s, bytes);
      kc::tma_load_2d(stage + C::kW, &maps.w, st * 64, n0, full + s);
      kc::tma_load_2d(stage + C::kX, &maps.x, st * 64, t0, full + s);
      for (int tile = 0; tile < tiles; ++tile)
        kc::tma_load_1d(stage + C::kSw + tile * 1024,
                        p.scales + ((size_t)((n0 >> 7) + tile) * p.ct + 2 * st) * 512, 1024u, full + s);
      kc::tma_load_1d(stage + C::kSx, xs + ((size_t)(t0 >> 7) * ct + 2 * st) * 512, 1024u, full + s);
    }
    return;
  }

  // MMA columns of 8 tokens holding a token below M: a partly filled last tile multiplies only those.
  const int live = min(G, (p.M - t0 + 7) / 8);
  const int r0 = warp * 16 * PR;   // the warp's first row in the CTA's tile
  // Each lane's addresses in a stage. A fragment of tile g, block m: rows gid and gid + 8, the block's two chunks
  // (tile.cuh's ldsm_a); B fragments of token columns 2 j and 2 j + 1 (ldsm_b2); the scale words of row
  // gid + 8 (tig & 1) and of token gid.
  uint32_t a_at[PR][2], sa_at[PR], b_at[2], sb_at;
#pragma unroll
  for (int g = 0; g < PR; ++g) {
#pragma unroll
    for (int m = 0; m < 2; ++m)
      a_at[g][m] = uint32_t(C::kW) +
                   kc::tile::offset<64>(r0 + 16 * g + (lane & 7) + 8 * (quarter & 1), 2 * m + (quarter >> 1));
    sa_at[g] = uint32_t(C::kSw) + staged_scale((n0 & 127) + r0 + 16 * g + gid + 8 * (tig & 1));
  }
#pragma unroll
  for (int m = 0; m < 2; ++m)
    b_at[m] = uint32_t(C::kX) + kc::tile::offset<64>((lane & 7) + 8 * (quarter >> 1), 2 * m + (quarter & 1));
  sb_at = uint32_t(C::kSx) + staged_scale((t0 & 127) + gid);
  constexpr auto apart = [](int rows) { return uint32_t((rows % 32) * 16 + (rows / 32) * 4); };   // staged_scale's step

  float c[PR][G][4];
  // SPLIT: the running totals, of token columns below H in registers, of the others in shared memory (an MMA's four
  // one 16-byte load and store).
  float held[SPLIT ? PR : 1][SPLIT ? H : 1][4];
  float4* totals = reinterpret_cast<float4*>(base + C::kTotals) + threadIdx.x;
  // c[g][t] joins its total (fp4_kernel's reduce: the splits' sums added in order); `keep`: the sum goes back to c.
  auto join = [&](int g, int t, bool keep) {
    float (&sum)[4] = c[g][t];
    if (t < H) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        held[g][t][i] += sum[i];
        if (keep) sum[i] = held[g][t][i];
      }
    } else {
      float4& at = totals[(g * (G - H) + t - H) * (32 * PW)];
      const float4 v = make_float4(at.x + sum[0], at.y + sum[1], at.z + sum[2], at.w + sum[3]);
      if (keep) sum[0] = v.x, sum[1] = v.y, sum[2] = v.z, sum[3] = v.w;
      else at = v;
    }
  };
#pragma unroll
  for (int g = 0; g < PR; ++g)
#pragma unroll
    for (int t = 0; t < G; ++t) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        c[g][t][i] = 0.f;
        if (SPLIT && t < H) held[g][t][i] = 0.f;
      }
      if (SPLIT && t >= H) totals[(g * (G - H) + t - H) * (32 * PW)] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  // The fragments in registers: the weights' of a step (two sets where kAhead), a pair of token columns' and the
  // pair's after it.
  uint32_t fa[C::kAhead ? 2 : 1][PR][2][4], sfa[C::kAhead ? 2 : 1][PR][2], fb[2][4], sfb[2][2];
  auto load_a = [&](const unsigned char* stage, int set) {
#pragma unroll
    for (int g = 0; g < PR; ++g)
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        kc::ldmatrix_x4(fa[set][g][m][0], fa[set][g][m][1], fa[set][g][m][2], fa[set][g][m][3], stage + a_at[g][m]);
        sfa[set][g][m] = *reinterpret_cast<const uint32_t*>(stage + sa_at[g] + m * 512);
      }
  };
  auto fetch = [&](const unsigned char* stage, int q) {
    const int m = q / (G / 2), j = q % (G / 2), slot = q & 1;
    kc::ldmatrix_x4(fb[slot][0], fb[slot][1], fb[slot][2], fb[slot][3], stage + b_at[m] + 16 * j * 64);
    sfb[slot][0] = *reinterpret_cast<const uint32_t*>(stage + sb_at + m * 512 + apart(16 * j));
    sfb[slot][1] = *reinterpret_cast<const uint32_t*>(stage + sb_at + m * 512 + apart(16 * j + 8));
  };
  // One step's MMAs: its first k64 block for every output, then its second, so an output's chain keeps its order
  // (m = 0, then 1) and the tile's other outputs' MMAs run between two MMAs of one accumulator. A pair of token
  // columns' fragments is read while the pair before it multiplies, the next step's first (and, where kAhead, its
  // weights') during this step's last MMAs. `first` (a split's first step): each chain starts from zero in its first
  // MMA, right after the chain before it joined the total (fp4_kernel's reduce: the splits' sums added in order), so
  // a split's boundary costs the MMAs no pause.
  // The tile's partial sums (parts > 1): slot `slot`, a consumer thread's (g, t, i) at ((g G + t) 4 + i) 32 PW + its
  // index; chain (g, t) of the split before `split` stored on its own (a later part's split boundary).
  constexpr int kTileSums = 32 * PW * PR * G * 4;
  const int tile = blockIdx.y * gridDim.x + blockIdx.x;
  auto slot_at = [&](int slot) { return p.partial + ((size_t)slot * gridDim.x * gridDim.y + tile) * kTileSums; };
  auto keep = [&](int slot, int g, int t) {
    float* at = slot_at(slot) + (g * G + t) * 4 * (32 * PW) + threadIdx.x;
#pragma unroll
    for (int i = 0; i < 4; ++i) __stcg(at + i * (32 * PW), c[g][t][i]);
  };
  int split = sp_lo, bound = st_lo;   // the next split and its first step
  // A split's boundary for chain (g, t): part 0 joins the chain before to its total (fp4_kernel's reduce), a later
  // part stores it (nothing before its first split).
  auto boundary = [&](int g, int t) {
    if (part == 0) join(g, t, false);
    else if (split > sp_lo) keep(1 + split - 1 - first_part_splits(p.splits, parts), g, t);
  };
  auto step = [&](int st, auto first) {
    const int r = st - st_lo;   // the step's ring slot counts from the CTA's first step
    const unsigned char* stage = base + (r % ST) * C::kStage;
    const unsigned char* next = base + ((r + 1) % ST) * C::kStage;
    const bool more = st + 1 < st_hi;
    constexpr int set = 0, ahead = C::kAhead ? 1 : 0;   // fixed register sets: the step's own, the next step's
#pragma unroll
    for (int q = 0; q < G; ++q) {
      const int m = q / (G / 2), j = q % (G / 2), slot = q & 1;
      if (q == G - 2 && more) {
        kc::mbarrier_wait(full + (r + 1) % ST, uint32_t(((r + 1) / ST) & 1));
        if constexpr (C::kAhead) load_a(next, ahead);
      }
      if (q + 1 < G) fetch(stage, q + 1);
      else if (more) fetch(next, 0);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        if (2 * j + h >= live) continue;
#pragma unroll
        for (int g = 0; g < PR; ++g) {
          float (&sum)[4] = c[g][2 * j + h];
          if (decltype(first)::value && m == 0) {
            if constexpr (PARTS) boundary(g, 2 * j + h);
            else if constexpr (SPLIT) join(g, 2 * j + h, false);
            mma_fp4_first(sum, fa[set][g][m][0], fa[set][g][m][1], fa[set][g][m][2], fa[set][g][m][3], fb[slot][2 * h],
                          fb[slot][2 * h + 1], sfa[set][g][m], sfb[slot][h]);
          } else {
            mma_fp4(sum, fa[set][g][m][0], fa[set][g][m][1], fa[set][g][m][2], fa[set][g][m][3], fb[slot][2 * h],
                    fb[slot][2 * h + 1], sfa[set][g][m], sfb[slot][h]);
          }
        }
      }
    }
    __syncwarp();
    if (lane == 0) kc::mbarrier_arrive(empty + r % ST);   // the warp is done with the stage
    if constexpr (C::kAhead) {   // the next step's fragments become the step's own
#pragma unroll
      for (int g = 0; g < PR; ++g)
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
          for (int i = 0; i < 4; ++i) fa[set][g][m][i] = fa[ahead][g][m][i];
          sfa[set][g][m] = sfa[ahead][g][m];
        }
    } else if (more) {
      load_a(next, set);
    }
  };
  kc::mbarrier_wait(full, 0u);
  load_a(base, 0);
  fetch(base, 0);
  for (int st = st_lo; st < st_hi; ++st) {
    if (st == bound) {
      step(st, std::true_type{});
      bound = ++split * steps / p.splits;
    } else {
      step(st, std::false_type{});
    }
  }
  if constexpr (SPLIT) {
    if (part == 0) {   // the last split's chains join their totals: c holds the finished sums (part 0's)
#pragma unroll
      for (int g = 0; g < PR; ++g)
#pragma unroll
        for (int t = 0; t < G; ++t) join(g, t, true);
    }
    if (PARTS && parts > 1) {   // part 0's total, or a later part's last chain; the tile's last CTA adds them in order
#pragma unroll
      for (int g = 0; g < PR; ++g)
#pragma unroll
        for (int t = 0; t < G; ++t)
          if (t < live) keep(part == 0 ? 0 : 1 + sp_hi - 1 - first_part_splits(p.splits, parts), g, t);
      __threadfence();
      prompt_consumers_sync(32 * PW);
      if (threadIdx.x == 0) last = atomicAdd(p.counters + tile, 1) == parts - 1;
      prompt_consumers_sync(32 * PW);
      if (!last) return;
      __threadfence();
      if (threadIdx.x == 0) p.counters[tile] = 0;   // every part has counted: ready for the next call
      const int slots = part_slots(p.splits, parts);
#pragma unroll
      for (int g = 0; g < PR; ++g)
#pragma unroll
        for (int t = 0; t < G; ++t)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            if (t < live) c[g][t][i] = __ldcg(slot_at(0) + ((g * G + t) * 4 + i) * (32 * PW) + threadIdx.x);
#pragma unroll 2
      for (int s = 1; s < slots; ++s) {
        const float* at = slot_at(s) + threadIdx.x;
#pragma unroll
        for (int g = 0; g < PR; ++g)
#pragma unroll
          for (int t = 0; t < G; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i)
              if (t < live) c[g][t][i] += __ldcg(at + ((g * G + t) * 4 + i) * (32 * PW));
      }
    }
  }
  // c[g][t][i]: matrix row n0 + r0 + 16 g + gid + 8 (i >> 1), token t0 + 8 t + 2 tig + (i & 1).
  // A whole tile of 16-bit outputs (store()'s and store_glu()'s values) leaves through the ring: each lane's values
  // into a [PT][kOutLd] tile, then every token's row in 16-byte stores by consecutive threads.
  if (n0 + C::kRows <= p.N && t0 + PT <= p.M && p.out_kind != kOutF32 && p.ldo % 8 == 0 &&
      reinterpret_cast<std::uintptr_t>(p.out) % 16 == 0) {
    uint16_t* tile = reinterpret_cast<uint16_t*>(base);
    prompt_consumers_sync(32 * PW);   // every warp has read its last stage
#pragma unroll
    for (int g = 0; g < PR; ++g) {
      const int r = r0 + 16 * g + gid;
      if (p.glu) {
#pragma unroll
        for (int t = 0; t < G; ++t)
#pragma unroll
          for (int i = 0; i < 2; ++i)
            tile[(8 * t + 2 * tig + i) * C::kOutLd + (r0 + 16 * g) / 2 + gid] = out_bits16(
                p, glu_value(p, n0 + r, c[g][t][i], c[g][t][i + 2]));
      } else {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float row_scale = p.row_scale[n0 + r + 8 * h];
#pragma unroll
          for (int t = 0; t < G; ++t)
#pragma unroll
            for (int i = 0; i < 2; ++i) {
              float v = c[g][t][2 * h + i];
              v *= row_scale;
              v *= p.input_rcp;
              tile[(8 * t + 2 * tig + i) * C::kOutLd + r + 8 * h] = out_bits16(p, v);
            }
        }
      }
    }
    prompt_consumers_sync(32 * PW);
    const int chunks = (p.glu ? C::kRows / 2 : C::kRows) / 8;   // 16-byte pieces of a token's row
    uint8_t* out = static_cast<uint8_t*>(p.out) + ((size_t)t0 * p.ldo + (p.glu ? n0 / 2 : n0)) * 2;
    for (int e = threadIdx.x; e < PT * chunks; e += 32 * PW) {
      const int tok = e / chunks, piece = e - tok * chunks;
      *reinterpret_cast<uint4*>(out + (size_t)tok * p.ldo * 2 + 16 * piece) =
          *reinterpret_cast<const uint4*>(base + tok * (C::kOutLd * 2) + 16 * piece);
    }
    return;
  }
#pragma unroll
  for (int g = 0; g < PR; ++g)
#pragma unroll
    for (int t = 0; t < G; ++t) {
      if (t >= live) break;
      if (p.glu) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int n = n0 + r0 + 16 * g + gid, tok = t0 + 8 * t + 2 * tig + i;
          if (n < p.N && tok < p.M)
            store_glu(p, tok, n, c[g][t][i], c[g][t][i + 2]);
        }
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int n = n0 + r0 + 16 * g + gid + (i >> 1) * 8, tok = t0 + 8 * t + 2 * tig + (i & 1);
          if (n < p.N && tok < p.M) store(p, tok, n, c[g][t][i]);
        }
      }
    }
#else
  (void)p, (void)maps, (void)prompt_smem;
  __trap();
#endif
}

// The launch's grid: tokens along x, row tiles along y, the parts of a tile's splits along z: the tuning's PK, or
// untuned (PK 0) prompt_parts' rule where the tiles fill a sixteenth of the resident CTAs at most. A part's ring fill,
// producer and stored chains cost the FP4 tile more than the decode-rate tiles: on an RTX 5090 a 320 x 10240 matrix
// (3 to 6 tiles) took 1.1-1.7x less time from 16 to 256 rows in parts, while 12 to 40 tiles (320 x 10240 at 512
// rows, 2560 x 2560 and 5120 x 17408 at 16 to 512) took 0.35-1.13x the time.
template <bool SPLIT, int PW, int PR, int PT>
dim3 prompt_fp4_grid(const Problem& p, int knob) {
  using C = PromptTile<SPLIT, PW, PR, PT>;
  dim3 grid((p.M + PT - 1) / PT, (p.N + C::kRows - 1) / C::kRows);
  if (SPLIT && grid.y <= 65535u) {
    // The one-part kernel's residency (its shared memory opted in first: the occupancy of more than the default is
    // asked of a kernel that may take it).
    static const int slots = [] {
      static kc::PerDevice<cudaError_t> optin;
      auto one = fp4_prompt_kernel<SPLIT, false, PW, PR, PT>;
      return kc::shared_memory_optin(optin, one, C::kSmem) == cudaSuccess
                 ? multiprocessors() * resident(one, C::kThreads, C::kSmem)
                 : 0;
    }();
    if (knob > 0 || 16L * grid.x * grid.y <= slots)
      grid.z = prompt_parts(p, grid.x * grid.y, grid.x, 32 * PW * PR * C::kGroups * 4, slots, knob);
  }
  return grid;
}

template <bool SPLIT, int PW, int PR, int PT>
cudaError_t launch_prompt_fp4(const Problem& p, const PromptMaps& maps, int knob, cudaStream_t stream) {
  using C = PromptTile<SPLIT, PW, PR, PT>;
  const dim3 grid = prompt_fp4_grid<SPLIT, PW, PR, PT>(p, knob);
  if (grid.y > 65535u) return cudaErrorInvalidValue;
  auto kernel = grid.z > 1 ? fp4_prompt_kernel<true, true, PW, PR, PT> : fp4_prompt_kernel<SPLIT, false, PW, PR, PT>;
  static kc::PerDevice<cudaError_t> optin, optin_parts;
  const cudaError_t set = kc::shared_memory_optin(grid.z > 1 ? optin_parts : optin, kernel, C::kSmem);
  if (set != cudaSuccess) return set;
  int major = 0, device = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaLaunchAttribute attribute[1];
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = dim3(C::kThreads);
  config.dynamicSmemBytes = C::kSmem;
  config.stream = stream;
  if (major >= 9) {   // queued behind the quantizer without a gap: the kernel waits for it itself
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attribute;
    config.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&config, kernel, p, maps);
}

// The launch's tensor maps (the driver's encoding: host arithmetic, no device call), then the kernel. The copies want
// both operands' codes and scale tiles at 16-byte addresses, as the decode tile's 16-byte loads do.
template <int PW, int PR, int PT>
cudaError_t prompt_fp4_config(const Problem& p, int knob, cudaStream_t stream, int* ctas) {
  using C = PromptTile<false, PW, PR, PT>;
  if (ctas) {   // its CTAs counted, nothing launched
    const dim3 grid = p.splits > 1 ? prompt_fp4_grid<true, PW, PR, PT>(p, knob) : prompt_fp4_grid<false, PW, PR, PT>(p, knob);
    *ctas = grid.x * grid.y * grid.z;
    return cudaSuccess;
  }
  const uint64_t row_bytes = uint64_t(p.K) / 2;
  PromptMaps maps{};
  auto map = [&](CUtensorMap* m, const void* codes, int rows) {
    return kc::make_tensor_map_2d(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, codes, uint64_t(rows), row_bytes, row_bytes,
                                  uint32_t(m == &maps.w ? C::kRows : PT), 64u, CU_TENSOR_MAP_SWIZZLE_64B) == CUDA_SUCCESS;
  };
  if (!map(&maps.w, p.w, p.N) || !map(&maps.x, p.scratch, p.M)) return cudaErrorInvalidValue;
  return p.splits > 1 ? launch_prompt_fp4<true, PW, PR, PT>(p, maps, knob, stream)
                      : launch_prompt_fp4<false, PW, PR, PT>(p, maps, knob, stream);
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
#define CRIT_FP4_CONFIG(index, warps, hoist) \
  case index:                                        \
    return by_rows<hoist>(p, stream);
    CRIT_CONFIGS_fp4_tile(CRIT_FP4_CONFIG)
#undef CRIT_FP4_CONFIG
    default: return cudaErrorInvalidValue;
  }
}

// The configuration crit_tuning.h's table names for this matrix and row count (0, the defaults, where it names none);
// the splits come with p (plan()'s for acts nvfp4_mma, the decode tile's). ctas: its CTAs counted instead.
cudaError_t dispatch_prompt_fp4(const Problem& p, cudaStream_t stream, int* ctas) {
  switch (crit_lookup_fp4_prompt_tile(p.N, p.K, p.M)) {
#define CRIT_FP4_PROMPT_CONFIG(index, pw, pr, pt, pk) \
  case index:                                         \
    return prompt_fp4_config<pw, pr, pt>(p, pk, stream, ctas);
    CRIT_CONFIGS_fp4_prompt_tile(CRIT_FP4_PROMPT_CONFIG)
#undef CRIT_FP4_PROMPT_CONFIG
    default: return cudaErrorInvalidValue;
  }
}

cudaError_t enqueue_prompt_fp4(const Problem& given, cudaStream_t stream) {
  Problem p = given;
  auto unaligned = [](const void* at) { return reinterpret_cast<std::uintptr_t>(at) % 16 != 0; };
  if (p.M < 1 || p.N < 1 || p.splits < 1 || p.splits > kMaxSplits || !p.scratch || p.weights != kNvfp4 || !p.ct ||
      p.K % kFp4Step || p.splits > p.K / kFp4Step || p.ct != p.K / 64 || unaligned(p.w) || unaligned(p.scales) ||
      unaligned(p.scratch))
    return cudaErrorInvalidValue;
  uint8_t* codes = p.scratch;
  uint8_t* scales = p.scratch + ((size_t)p.M * p.K / 2 + 15) / 16 * 16;
  const cudaError_t e = enqueue_quantize_nvfp4(p.x, p.ldx, p.M, p.K, p.input_scale, p.input_rcp, codes, scales, stream);
  if (e != cudaSuccess) return e;
  return dispatch_prompt_fp4(p, stream, nullptr);
}

int prompt_fp4_ctas(const Problem& p) {
  int ctas = 0;
  return dispatch_prompt_fp4(p, nullptr, &ctas) == cudaSuccess ? ctas : 0;
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
