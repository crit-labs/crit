// Projections of 1 to 64 rows on tensor cores, one fixed tile, and the activations' quantizers (mma_rows.h).
#include "mma_rows.h"
#include "fp4_mma.h"
#include "tile_parts.cuh"
#include "crit_tuning.h"

namespace crit::kernels::mma_rows {
namespace {

// One CTA a row: its scale (amax * fp32(1 / 448), at least 1e-12) and its values, as E4M3 bytes (VALUES false) or as
// BF16 values (true: the projection's pre-pass). Rows with K a multiple of 8 on 16-byte boundaries (aligned) read 8
// values a load and keep up to kHeld of them a thread between the maximum and the quantization; others go by value.
constexpr int kRowThreads = 512, kHeld = 4;

template <bool VALUES>
__global__ void __launch_bounds__(kRowThreads) fp8_rows_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int K,
                                                               void* __restrict__ q, int64_t ldq,
                                                               float* __restrict__ scale, bool aligned) {
  __shared__ float part[kRowThreads / 32];
  __shared__ float row_scale;
  wait_prior();
  launch_next();
  const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const __nv_bfloat16* xr = x + (size_t)row * ldx;
  float m = 0.f;
  uint4 held[kHeld];
  if (aligned) {
#pragma unroll
    for (int i = 0; i < kHeld; ++i) {
      const int k = 8 * (threadIdx.x + i * kRowThreads);
      held[i] = k < K ? __ldg(reinterpret_cast<const uint4*>(xr + k)) : make_uint4(0u, 0u, 0u, 0u);
      const uint32_t w[4] = {held[i].x, held[i].y, held[i].z, held[i].w};
#pragma unroll
      for (int j = 0; j < 4; ++j) m = nanmax(nanmax(m, fabsf(bf16_lo(w[j]))), fabsf(bf16_hi(w[j])));
    }
    for (int k = 8 * (threadIdx.x + kHeld * kRowThreads); k < K; k += 8 * kRowThreads) {
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(xr + k));
      const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) m = nanmax(nanmax(m, fabsf(bf16_lo(w[j]))), fabsf(bf16_hi(w[j])));
    }
  } else {
    for (int k = threadIdx.x; k < K; k += blockDim.x) m = nanmax(m, fabsf(__bfloat162float(xr[k])));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) m = nanmax(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (lane == 0) part[warp] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) a = nanmax(a, part[w]);
    const float sc = a * kRcp448;
    row_scale = sc < 1e-12f ? 1e-12f : sc;
    scale[row] = row_scale;
  }
  __syncthreads();
  const float sc = row_scale;
  auto put8 = [&](int k, const uint4& v) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    if constexpr (VALUES) {
      uint32_t o[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) o[j] = bf16x2(fp8_value(bf16_lo(w[j]), sc), fp8_value(bf16_hi(w[j]), sc));
      *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(q) + (size_t)row * ldq + k) = make_uint4(o[0], o[1], o[2], o[3]);
    } else {
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t two = e4m3_bits(fp8_value(bf16_lo(w[j]), sc)) | (e4m3_bits(fp8_value(bf16_hi(w[j]), sc)) << 8);
        if (j < 2) lo |= two << (16 * j);
        else hi |= two << (16 * (j - 2));
      }
      *reinterpret_cast<uint2*>(static_cast<uint8_t*>(q) + (size_t)row * ldq + k) = make_uint2(lo, hi);
    }
  };
  if (aligned) {
#pragma unroll
    for (int i = 0; i < kHeld; ++i) {
      const int k = 8 * (threadIdx.x + i * kRowThreads);
      if (k < K) put8(k, held[i]);
    }
    for (int k = 8 * (threadIdx.x + kHeld * kRowThreads); k < K; k += 8 * kRowThreads)
      put8(k, __ldg(reinterpret_cast<const uint4*>(xr + k)));
  } else {
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
      const float v = fp8_value(__bfloat162float(xr[k]), sc);
      if constexpr (VALUES) static_cast<__nv_bfloat16*>(q)[(size_t)row * ldq + k] = __float2bfloat16_rn(v);
      else static_cast<uint8_t*>(q)[(size_t)row * ldq + k] = static_cast<uint8_t>(e4m3_bits(v));
    }
  }
}

// One thread a 16-value block: its values in BF16 (the projection's pre-pass; e2m1(code) * the block's scale).
__global__ void __launch_bounds__(256) nvfp4_values_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int M,
                                                           int K, float g, float g_rcp,
                                                           __nv_bfloat16* __restrict__ values) {
  wait_prior();
  launch_next();
  const int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int blocks = K >> 4;
  if (id >= (int64_t)M * blocks) return;
  const int row = (int)(id / blocks), j = (int)(id - (int64_t)row * blocks);
  const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)row * ldx + 16 * j);
  const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
  const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
  uint32_t lo, hi;
  const float scale = e4m3(nvfp4_block(w, g, g_rcp, lo, hi));
  uint32_t out[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t byte = ((i < 4 ? lo : hi) >> (8 * (i & 3))) & 0xffu;
    out[i] = bf16x2(crit::kcore::e2m1_value(byte & 15u) * scale, crit::kcore::e2m1_value(byte >> 4) * scale);
  }
  uint4* dst = reinterpret_cast<uint4*>(values + (size_t)row * K + 16 * j);
  dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
  dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
}

// One thread a 16-value block of the padded [ceil(M / 128) * 128][ceil(K / 64) * 4] block grid: its codes and its
// scale, or a zero scale in the padding.
__global__ void __launch_bounds__(256) nvfp4_codes_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int M,
                                                          int K, float g, float g_rcp, uint8_t* __restrict__ codes,
                                                          uint8_t* __restrict__ scales, int rows_padded,
                                                          int blocks_padded) {
  wait_prior();
  launch_next();
  const int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= (int64_t)rows_padded * blocks_padded) return;
  const int row = (int)(id / blocks_padded), j = (int)(id - (int64_t)row * blocks_padded);
  const size_t at = scale_offset(row, j, 0, blocks_padded >> 2);
  if (row >= M || j >= (K >> 4)) {
    scales[at] = 0;
    return;
  }
  const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)row * ldx + 16 * j);
  const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
  const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
  uint32_t lo, hi;
  scales[at] = static_cast<uint8_t>(nvfp4_block(w, g, g_rcp, lo, hi));
  *reinterpret_cast<uint2*>(codes + (size_t)row * (K >> 1) + 8 * j) = make_uint2(lo, hi);
}

// ---- the projection ----


// A row tile's sums written (splits 1) or added in split order by the tile's last CTA (both tiles of this file: the
// same instructions, so the same bits). c0, c1: matrix row gid, tokens 2 tig and 2 tig + 1; c2, c3: row gid + 8.
template <int WF, int ACT, int R, int T>
__device__ __forceinline__ void sums_out(const Problem& p, const float (&c)[R][T][4], int tile, int split, int rows,
                                         int r0, int gid, int tig, int& last) {
  if (p.splits == 1) {
#pragma unroll
    for (int g = 0; g < R; ++g)
#pragma unroll
      for (int t = 0; t < T; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int n = r0 + 16 * g + gid + (i >> 1) * 8, tok = 8 * t + 2 * tig + (i & 1);
          if (n < p.N && tok < p.M) finish<WF, ACT>(p, tok, n, c[g][t][i]);
        }
    return;
  }
  float* part = p.partial + (size_t)split * p.M * p.N;
#pragma unroll
  for (int g = 0; g < R; ++g)
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = r0 + 16 * g + gid + (i >> 1) * 8, tok = 8 * t + 2 * tig + (i & 1);
        if (n < p.N && tok < p.M) __stcg(part + (size_t)tok * p.N + n, c[g][t][i]);
      }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(p.counters + tile, 1) == p.splits - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (threadIdx.x == 0) p.counters[tile] = 0;   // every split has counted: ready for the next call
  const int first = tile * rows;
  for (int i = threadIdx.x; i < p.M * rows; i += blockDim.x) {
    const int tok = i / rows, n = first + (i - tok * rows);
    if (n >= p.N) continue;
    float v = __ldcg(p.partial + (size_t)tok * p.N + n);
    for (int s = 1; s < p.splits; ++s) v += __ldcg(p.partial + ((size_t)s * p.M + tok) * p.N + n);
    finish<WF, ACT>(p, tok, n, v);
  }
}

// One CTA: a row tile (warps x 16 R matrix rows, each warp its own) times every token (up to 8 T) over split
// blockIdx.y of K's steps.
template <int WF, int ACT, int R, int T>
// Decode tiles (up to 16 rows) at most 128 registers a thread, so four CTAs of four warps fit a multiprocessor (a few
// percent faster than the compiler's own choice at 1-4 rows); prompt tiles as the compiler likes.
__global__ void __launch_bounds__(kMaxWarps * 32, T <= 2 ? 4 : 1) mma_kernel(const Problem p) {
  constexpr int SUBS = Row<WF>::SUBS, STEP = 128 * SUBS;
  __shared__ int last;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int tile = blockIdx.x, split = blockIdx.y, rows = tile_rows(R, p.warps);
  const int r0 = tile * rows + warp * 16 * R;
  const int steps = (p.K + STEP - 1) / STEP;
  const int s_lo = split * steps / p.splits, s_hi = (split + 1) * steps / p.splits;
  const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(p.x);

  // One accumulator chain a tile (two chains, even and odd MMAs, measured slower: more registers, fewer warps).
  float c[R][T][4];
#pragma unroll
  for (int g = 0; g < R; ++g)
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) c[g][t][i] = 0.f;

  // One step's weights in registers; the warps a multiprocessor holds keep the loads in flight (a second set of
  // registers for the next step measured slower: fewer warps fit).
  Row<WF> wa[R], wb[R];
  auto load = [&](int st) {
#pragma unroll
    for (int g = 0; g < R; ++g) {
      wa[g].load(p, min(r0 + 16 * g + gid, p.N - 1), st * STEP, tig);   // a row past N reads row N - 1, writes nothing
      wb[g].load(p, min(r0 + 16 * g + gid + 8, p.N - 1), st * STEP, tig);
    }
  };
  if (s_lo < s_hi) load(s_lo);   // weights do not depend on the previous kernel: read them before waiting for it
  wait_prior();
  for (int st = s_lo; st < s_hi;) {
#pragma unroll
    for (int q = 0; q < SUBS; ++q) {
      const int kb = st * STEP + 32 * (SUBS * tig + q);
      uint32_t b[T][16];
#pragma unroll
      for (int t = 0; t < T; ++t) act_frags(p, x, 8 * t + gid, kb, kb < p.K, b[t]);
#pragma unroll
      for (int g = 0; g < R; ++g) {
        uint32_t pa[16], pb[16];   // MMA s takes pairs 2 s (values 4 s, + 1) and 2 s + 1 of rows gid and gid + 8
        if constexpr (kSplit<WF>) {   // each MMA twice: the weights' hi parts, then their lo parts
          uint32_t la[16], lb[16];
          wa[g].split_pairs(q, pa, la);
          wb[g].split_pairs(q, pb, lb);
#pragma unroll
          for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int t = 0; t < T; ++t) {
              mma(c[g][t], pa[2 * s], pb[2 * s], pa[2 * s + 1], pb[2 * s + 1], b[t][2 * s], b[t][2 * s + 1]);
              mma(c[g][t], la[2 * s], lb[2 * s], la[2 * s + 1], lb[2 * s + 1], b[t][2 * s], b[t][2 * s + 1]);
            }
        } else {
          wa[g].pairs(q, pa);
          wb[g].pairs(q, pb);
#pragma unroll
          for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int t = 0; t < T; ++t)
              mma(c[g][t], pa[2 * s], pb[2 * s], pa[2 * s + 1], pb[2 * s + 1], b[t][2 * s], b[t][2 * s + 1]);
        }
      }
    }
    if (++st < s_hi) load(st);
  }

  sums_out<WF, ACT, R, T>(p, c, tile, split, rows, r0, gid, tig, last);
}

// ---- the staged decode tile (structure decode-tile, knob SW > 0) ----
//
// The decode tile's chains exactly (every MMA with the operands mma_kernel gives it, in its order, the same splits and
// reduce), with the activations of each step staged once a CTA in shared memory by cp.async, three steps in flight
// and one barrier a step, instead of each warp reading its B fragments from global memory: SW warps share a step's
// activations, so the activation reads that grow with the rows leave L2 for shared memory. A step of a token is its
// STEP values in memory order, 16-byte chunk c = 4 (SUBS tig + q) + w (lane quad position tig's sub-step q, values
// 8w .. 8w + 7); the chunk's slot in the token's row XORs its low bits with tig and the token's parity, so the 8 lanes
// of a 16-byte read phase (two tokens, four quad positions) hit 8 bank groups.
constexpr int kStagedBuffers = 3;

template <int SUBS>
__device__ __forceinline__ int staged_slot(int tok, int c) {
  if constexpr (SUBS >= 2) {
    const int tig = c / (4 * SUBS);   // at bit 3 or above
    return c ^ (((tig << 1) | (tok & 1)) & 7);
  } else {
    const int tig = c >> 2;           // bits 2-3: XOR bits 0-1 only
    return c ^ (((tig >> 1) << 1) | (tok & 1));
  }
}

// Three steps of activations a CTA take 24 KB (one MMA column of tokens) or 48 KB (two): four CTAs of four warps a
// multiprocessor at 1 to 8 rows, two at 9 to 16, and so at most 128 or 255 registers a thread.
template <int WF, int ACT, int R, int T, int SW, bool PF>
__global__ void __launch_bounds__(SW * 32, (T <= 1 ? 16 : 8) / SW) staged_kernel(const Problem p) {
  constexpr int SUBS = Row<WF>::SUBS, STEP = 128 * SUBS, CH = STEP / 8, TOK = 8 * T, BUF = TOK * CH;
  __shared__ uint4 xs[kStagedBuffers * BUF];
  // The reduce's flag shares the buffers (48 KB at 9 to 16 rows, the static limit): sums_out sets it after a barrier
  // every warp reaches with its steps done and every copy landed.
  int& last = *reinterpret_cast<int*>(xs);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int tile = blockIdx.x, split = blockIdx.y, rows = tile_rows(R, SW);
  const int r0 = tile * rows + warp * 16 * R;
  const int steps = (p.K + STEP - 1) / STEP;
  const int s_lo = split * steps / p.splits, s_hi = (split + 1) * steps / p.splits;
  const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(p.x);

  // Step st's activations of every token of the call (zeros past M and past K, as act_frags gives them).
  auto stage = [&](int st) {
    uint4* buffer = xs + (st % kStagedBuffers) * BUF;
#pragma unroll
    for (int j = 0; j < (BUF + SW * 32 - 1) / (SW * 32); ++j) {
      const int i = threadIdx.x + j * SW * 32;
      if (BUF % (SW * 32) && i >= BUF) break;
      const int tok = i / CH, c = i - tok * CH;
      const int k = st * STEP + 8 * c;
      const bool valid = tok < p.M && k < p.K;   // K is a multiple of 32: a 32-value run is inside or past K whole
      cp_async16(buffer + tok * CH + staged_slot<SUBS>(tok, c), valid ? x + (size_t)tok * p.ldx + k : x, valid);
    }
    cp_commit();
  };

  float c[R][T][4];
#pragma unroll
  for (int g = 0; g < R; ++g)
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) c[g][t][i] = 0.f;

  auto load = [&](Row<WF>(&a)[R], Row<WF>(&b)[R], int st) {
#pragma unroll
    for (int g = 0; g < R; ++g) {
      a[g].load(p, min(r0 + 16 * g + gid, p.N - 1), st * STEP, tig);
      b[g].load(p, min(r0 + 16 * g + gid + 8, p.N - 1), st * STEP, tig);
    }
  };
  // One step on the weights in (a, b), the next step's weights loaded into (na, nb): before the step's MMAs with PF
  // (two steps of weights in registers, the loads in flight across the MMAs), after them without.
  auto step = [&](int st, Row<WF>(&a)[R], Row<WF>(&b)[R], Row<WF>(&na)[R], Row<WF>(&nb)[R]) {
    if (st + 1 < s_hi) cp_wait<1>();
    else cp_wait<0>();
    __syncthreads();   // step st is in; every warp is done with step st - 1, whose buffer step st + 2 takes
    if (st + 2 < s_hi) stage(st + 2);
    if (PF && st + 1 < s_hi) load(na, nb, st + 1);
    const uint4* buffer = xs + (st % kStagedBuffers) * BUF;
#pragma unroll
    for (int q = 0; q < SUBS; ++q) {
      uint32_t bf[T][16];
#pragma unroll
      for (int t = 0; t < T; ++t) {
        const int tok = 8 * t + gid;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const uint4 v = buffer[tok * CH + staged_slot<SUBS>(tok, 4 * (SUBS * tig + q) + w)];
          bf[t][4 * w] = v.x;
          bf[t][4 * w + 1] = v.y;
          bf[t][4 * w + 2] = v.z;
          bf[t][4 * w + 3] = v.w;
        }
      }
#pragma unroll
      for (int g = 0; g < R; ++g) {
        uint32_t pa[16], pb[16];
        if constexpr (kSplit<WF>) {   // the register tile's MMAs: hi, then lo
          uint32_t la[16], lb[16];
          a[g].split_pairs(q, pa, la);
          b[g].split_pairs(q, pb, lb);
#pragma unroll
          for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int t = 0; t < T; ++t) {
              mma(c[g][t], pa[2 * s], pb[2 * s], pa[2 * s + 1], pb[2 * s + 1], bf[t][2 * s], bf[t][2 * s + 1]);
              mma(c[g][t], la[2 * s], lb[2 * s], la[2 * s + 1], lb[2 * s + 1], bf[t][2 * s], bf[t][2 * s + 1]);
            }
        } else {
          a[g].pairs(q, pa);
          b[g].pairs(q, pb);
#pragma unroll
          for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int t = 0; t < T; ++t)
              mma(c[g][t], pa[2 * s], pb[2 * s], pa[2 * s + 1], pb[2 * s + 1], bf[t][2 * s], bf[t][2 * s + 1]);
        }
      }
    }
    if (!PF && st + 1 < s_hi) load(na, nb, st + 1);
  };

  Row<WF> wa[R], wb[R], va[R], vb[R];
  if (s_lo < s_hi) load(wa, wb, s_lo);   // weights do not depend on the previous kernel: read them before waiting
  wait_prior();
  if (s_lo < s_hi) stage(s_lo);
  if (s_lo + 1 < s_hi) stage(s_lo + 1);
  if constexpr (PF) {
    for (int st = s_lo; st < s_hi; st += 2) {   // the two register sets alternate
      step(st, wa, wb, va, vb);
      if (st + 1 < s_hi) step(st + 1, va, vb, wa, wb);
    }
  } else {
    for (int st = s_lo; st < s_hi; ++st) step(st, wa, wb, wa, wb);
  }
  sums_out<WF, ACT, R, T>(p, c, tile, split, rows, r0, gid, tig, last);
}

template <int WF, int ACT, int R, int T>
cudaError_t launch(const Problem& p, cudaStream_t stream) {
  const dim3 grid(row_tiles(p.N, R, p.warps), p.splits);
  return launch_kernel(mma_kernel<WF, ACT, R, T>, grid, dim3(32 * p.warps), stream, p);
}

// The staged tile: SW warps a CTA whatever the plan's warps (SW >= kMaxWarps >= warps, so its row tiles are no more
// than the counters the plan's warps give).
template <int WF, int ACT, int R, int T, int SW, bool PF>
cudaError_t launch_staged(const Problem& p, cudaStream_t stream) {
  static_assert(SW >= kMaxWarps, "a staged CTA has at least the plan's warps");
  const dim3 grid(row_tiles(p.N, R, SW), p.splits);
  return launch_kernel(staged_kernel<WF, ACT, R, T, SW, PF>, grid, dim3(32 * SW), stream, p);
}

template <int WF, int ACT, int SW, bool PF>
cudaError_t by_staged(const Problem& p, int T, int R, cudaStream_t stream) {
  if (T == 1)
    return R == 2 ? launch_staged<WF, ACT, 2, 1, SW, PF>(p, stream) : launch_staged<WF, ACT, 1, 1, SW, PF>(p, stream);
  return R == 2 ? launch_staged<WF, ACT, 2, 2, SW, PF>(p, stream) : launch_staged<WF, ACT, 1, 2, SW, PF>(p, stream);
}

// The staged tile's warps for this call and whether it loads weights a step ahead (kernel.json decode-tile's SW and
// SPF, picked per matrix and row count by the build's tuning; SW 0: the register tile): SW + 1 with SPF.
inline int staged_warps(const Problem& p) {
  switch (crit_lookup_decode_tile(p.weights, p.N, p.K, p.M)) {
#define CRIT_DECODE_CONFIG(index, groups, warps, splits, sw, spf) \
  case index:                                                    \
    return sw ? sw + (spf ? 1 : 0) : 0;
    CRIT_CONFIGS_decode_tile(CRIT_DECODE_CONFIG)
#undef CRIT_DECODE_CONFIG
    default: return 0;
  }
}

// Whether the build's tuning holds a configuration of the staged tile (staged_warps' value v): only those are
// compiled, so a build without them (the defaults') holds the register tile alone.
template <int V>
constexpr bool tuned_staged() {
  bool found = false;
#define CRIT_DECODE_CONFIG(index, groups, warps, splits, sw, spf) found = found || ((sw ? sw + (spf ? 1 : 0) : 0) == V);
  CRIT_CONFIGS_decode_tile(CRIT_DECODE_CONFIG)
#undef CRIT_DECODE_CONFIG
  return found;
}

// R * T <= 8 keeps a lane's accumulators at 32 registers. The decode tile's order knobs (kernel.json, structure
// decode-tile: groups, warps, splits) are runtime arguments plan() reads from the tuning; SW and SPF (the staged tile,
// up to 16 rows) are the build's per matrix and row count.
template <int WF, int ACT>
cudaError_t by_tile(const Problem& p, cudaStream_t stream) {
  const int T = p.M <= 8 ? 1 : p.M <= 16 ? 2 : p.M <= 32 ? 4 : 8;
  const int R = T <= 4 ? p.groups : 1;
  if (T <= 2) {
    const int sw = staged_warps(p);
    if constexpr (tuned_staged<4>()) if (sw == 4) return by_staged<WF, ACT, 4, false>(p, T, R, stream);
    if constexpr (tuned_staged<5>()) if (sw == 5) return by_staged<WF, ACT, 4, true>(p, T, R, stream);
    if constexpr (tuned_staged<8>()) if (sw == 8) return by_staged<WF, ACT, 8, false>(p, T, R, stream);
    if constexpr (tuned_staged<9>()) if (sw == 9) return by_staged<WF, ACT, 8, true>(p, T, R, stream);
  }
#ifdef CRIT_SWEEP_BUILD
  // A sweep build calls up to 16 rows (the benches' tune_rows), and compiles nothing more.
  if (T > 2) return cudaErrorInvalidValue;
  return T == 1 ? (R == 2 ? launch<WF, ACT, 2, 1>(p, stream) : launch<WF, ACT, 1, 1>(p, stream))
                : (R == 2 ? launch<WF, ACT, 2, 2>(p, stream) : launch<WF, ACT, 1, 2>(p, stream));
#else
  if (T == 1) return R == 2 ? launch<WF, ACT, 2, 1>(p, stream) : launch<WF, ACT, 1, 1>(p, stream);
  if (T == 2) return R == 2 ? launch<WF, ACT, 2, 2>(p, stream) : launch<WF, ACT, 1, 2>(p, stream);
  if (T == 4) return R == 2 ? launch<WF, ACT, 2, 4>(p, stream) : launch<WF, ACT, 1, 4>(p, stream);
  return launch<WF, ACT, 1, 8>(p, stream);
#endif
}

template <int WF>
int resident_ctas() {
  int n = 0;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, mma_kernel<WF, kActBf16, 1, 1>, kMaxWarps * 32, 0) ==
                 cudaSuccess ? n : 0;
}

}  // namespace

int decode_ctas(int weights, int acts) {
  if (acts == kActNvfp4Mma) return weights == kNvfp4 ? fp4_ctas() : 0;
  switch (weights) {
    case kNvfp4: return resident_ctas<kNvfp4>();
    case kFp8: return resident_ctas<kFp8>();
    case kMxfp8: return resident_ctas<kMxfp8>();
    case kBf16: return resident_ctas<kBf16>();
    case kInt4Zp: return resident_ctas<kInt4Zp>();
    default: return 0;
  }
}

cudaError_t prepass(Problem& p, cudaStream_t stream) {
  if (p.acts == kActBf16) return cudaSuccess;
  if (!p.scratch) return cudaErrorInvalidValue;
  __nv_bfloat16* values = reinterpret_cast<__nv_bfloat16*>(p.scratch);
  float* scales = reinterpret_cast<float*>(p.scratch + ((size_t)p.M * p.K * 2 + 15) / 16 * 16);
  cudaError_t e;
  if (p.acts == kActFp8Rows) {
    e = launch_kernel(fp8_rows_kernel<true>, dim3(p.M), dim3(kRowThreads), stream,
                      static_cast<const __nv_bfloat16*>(p.x), p.ldx, p.K, static_cast<void*>(values), (int64_t)p.K,
                      scales, p.K % 8 == 0 && p.ldx % 8 == 0);
  } else {
    const int64_t threads = (int64_t)p.M * (p.K >> 4);
    e = launch_kernel(nvfp4_values_kernel, dim3((unsigned)((threads + 255) / 256)), dim3(256), stream,
                      static_cast<const __nv_bfloat16*>(p.x), p.ldx, p.M, p.K, p.input_scale, p.input_rcp, values);
  }
  if (e != cudaSuccess) return e;
  p.x = values;
  p.ldx = p.K;
  p.act_scale = scales;
  return cudaSuccess;
}

cudaError_t enqueue(const Problem& given, cudaStream_t stream) {
  Problem p = given;
  if (p.M < 1 || p.M > kMaxRows || p.K < kStep || p.K % kStep || p.N < 1 || p.warps < 1 || p.warps > kMaxWarps ||
      (p.groups != 1 && p.groups != 2) || p.splits < 1 || p.splits > kMaxSplits ||
      (p.splits > 1 && (!p.partial || !p.counters)))
    return cudaErrorInvalidValue;
  // A row tile's rows (a quad of a tile's warps reads its 16) are rows of one tile: the counters are a tile's.
  if (p.acts == kActNvfp4Mma) return enqueue_fp4(p, stream);   // fp4_mma.cu
  if (const cudaError_t e = prepass(p, stream); e != cudaSuccess) return e;
  if (p.weights == kNvfp4 && p.acts == kActBf16) return by_tile<kNvfp4, kActBf16>(p, stream);
#ifndef CRIT_SWEEP_BUILD   // a sweep build times the activations as they are (the benches' call)
  if (p.weights == kNvfp4 && p.acts == kActNvfp4) return by_tile<kNvfp4, kActNvfp4>(p, stream);
#endif
  if (p.weights == kFp8 && p.acts == kActBf16) return by_tile<kFp8, kActBf16>(p, stream);
#ifndef CRIT_SWEEP_BUILD   // a sweep build times the activations as they are (the benches' call)
  if (p.weights == kFp8 && p.acts == kActFp8Rows) return by_tile<kFp8, kActFp8Rows>(p, stream);
#endif
  if (p.weights == kMxfp8 && p.acts == kActBf16) return by_tile<kMxfp8, kActBf16>(p, stream);
  if (p.weights == kBf16 && p.acts == kActBf16) return by_tile<kBf16, kActBf16>(p, stream);
  if (p.weights == kInt4Zp && p.acts == kActBf16) return by_tile<kInt4Zp, kActBf16>(p, stream);
  return cudaErrorInvalidValue;
}

cudaError_t enqueue_quantize_fp8_rows(const void* x, int64_t ldx, int M, int K, std::uint8_t* q, int64_t ldq,
                                      float* scale, cudaStream_t stream) {
  if (M < 1 || K < 1) return cudaErrorInvalidValue;
  const bool aligned = K % 8 == 0 && ldx % 8 == 0 && ldq % 8 == 0 && reinterpret_cast<std::uintptr_t>(x) % 16 == 0 &&
                       reinterpret_cast<std::uintptr_t>(q) % 8 == 0;
  return launch_kernel(fp8_rows_kernel<false>, dim3(M), dim3(kRowThreads), stream, static_cast<const __nv_bfloat16*>(x),
                       ldx, K, static_cast<void*>(q), ldq, scale, aligned);
}

cudaError_t enqueue_quantize_nvfp4(const void* x, int64_t ldx, int M, int K, float input_scale, float input_rcp,
                                   std::uint8_t* codes, std::uint8_t* blocked_scales, cudaStream_t stream) {
  if (M < 1 || K < 16 || K % 16) return cudaErrorInvalidValue;
  const int rows_padded = (M + 127) / 128 * 128, blocks_padded = (K / 16 + 3) / 4 * 4;
  const int64_t threads = (int64_t)rows_padded * blocks_padded;
  return launch_kernel(nvfp4_codes_kernel, dim3((unsigned)((threads + 255) / 256)), dim3(256), stream,
                       static_cast<const __nv_bfloat16*>(x), ldx, M, K, input_scale, input_rcp, codes, blocked_scales,
                       rows_padded, blocks_padded);
}

}  // namespace crit::kernels::mma_rows
