// The prompt tile: projections of any number of activation rows whose every output has the decode tile's bits
// (mma_rows.h, enqueue_prompt).
//
// An output's sum in the decode tile (mma_rows.cu) is one chain of mma.m16n8k16 per split of K's steps, starting at
// zero: for each step of the split, each sub-step q, each of its 8 MMAs, in that order, with every operand placed as
// that tile places it (lane quad position tig holds the K values [step + 32 (SUBS tig + q), + 32), MMA s its values
// 4s .. 4s + 3, matrix row n at A row n mod 16, the weights decoded by tile_parts.cuh's Row); the splits' chains are
// added in split order (the first, plus the second, ...), then finish() applies the scales. This tile computes each
// output by exactly that sequence, so it has the same bits, while a CTA reuses each decoded weight for 64 activation
// rows: 8 warps, each its own 16 matrix rows, over the CTA's 64 rows, whose sub-steps are staged in shared memory
// (cp.async, two buffers, a swizzle that keeps the B fragments' 16-byte reads free of bank conflicts). A CTA runs every
// split of K itself, one after another, adding each split's chain into a running total as the decode tile's last CTA
// adds the partial sums, so it needs no partial or counter buffers.
#include "mma_rows.h"
#include "tile_parts.cuh"
#include "crit_tuning.h"

namespace crit::kernels::mma_rows {
namespace {

// The prompt tile's knobs (kernel.json, structure prompt-tile; crit_tuning.h picks them per matrix and row count):
//   PW  warps a CTA, each its own 16 matrix rows (8 untuned)
//   PT  activation rows a CTA (kPromptTokens, 64, untuned)
// Neither changes an output's chain: every output keeps the decode tile's bits.
constexpr int kChunks = 16;   // a staged sub-step of one activation row: 4 lanes' 32 values in 16-byte chunks (chunk
                              // 4 tig + w: values 8w .. 8w + 7 of lane tig's)

// The shared-memory chunk of activation row `tok`'s chunk c (4 tig + w): within the row's 16, its quad position's
// bits 1 and 0 XORed with tig's high bit and the row's low bit, so that each 8-lane phase of a B-fragment read (two
// rows, the four quad positions) covers all 32 banks.
__device__ __forceinline__ int chunk_at(int tok, int c) { return tok * kChunks + (c ^ (2 * (c >> 3)) ^ (tok & 1)); }

// One CTA: 16 PW matrix rows (blockIdx.y) times PT activation rows (blockIdx.x) over all of K.
// SPLIT: the plan has more than one split of K (a running total beside each split's chain).
template <int WF, int ACT, bool SPLIT, int PW, int PT>
__global__ void __launch_bounds__(PW * 32, 1) prompt_kernel(const Problem p) {
  constexpr int SUBS = Row<WF>::SUBS, STEP = 128 * SUBS;
  constexpr int kPromptRows = 16 * PW;      // matrix rows a CTA
  constexpr int kPromptGroups = PT / 8;     // MMA columns of 8 activation rows a CTA
  constexpr int kBuffer = PT * kChunks;     // 16-byte chunks a staged sub-step takes
  __shared__ uint4 staged[2 * kBuffer];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int t0 = blockIdx.x * PT, r0 = blockIdx.y * kPromptRows + warp * 16;
  const int steps = (p.K + STEP - 1) / STEP, units = steps * SUBS;
  const int na = min(r0 + gid, p.N - 1), nb = min(r0 + gid + 8, p.N - 1);   // a row past N reads row N - 1
  const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(p.x);
  // MMA columns of 8 activation rows holding a row below M: a partly filled last tile stages and multiplies only
  // those (the others' outputs are not stored), so a 65-row prompt's second CTA does one column's work, not eight.
  const int live = min(kPromptGroups, (p.M - t0 + 7) / 8);

  // Sub-step u (step u / SUBS, sub-step u % SUBS) of the CTA's activation rows into buffer u & 1; zeros for a row past
  // M or a sub-step past K, as act_frags gives them.
  auto stage = [&](int u) {
    const int st = u / SUBS, q = u - st * SUBS;
    uint4* buffer = staged + (u & 1) * kBuffer;
    for (int i = threadIdx.x; i < live * 8 * kChunks; i += PW * 32) {
      const int tok = i / kChunks, c = i - tok * kChunks;
      const int kb = st * STEP + 32 * (SUBS * (c >> 2) + q);
      const bool valid = t0 + tok < p.M && kb < p.K;
      const __nv_bfloat16* src = valid ? x + (size_t)(t0 + tok) * p.ldx + kb + 8 * (c & 3) : x;
      cp_async16(buffer + chunk_at(tok, c), src, valid);
    }
    cp_commit();
  };

  float c[kPromptGroups][4];
  float total[SPLIT ? kPromptGroups : 1][4];
  Row<WF> wa, wb, next_a, next_b;
  if (steps) {   // the weights do not depend on the previous kernel: read them before waiting for it
    wa.load(p, na, 0, tig);
    wb.load(p, nb, 0, tig);
  }
  wait_prior();
  if (units) stage(0);
  for (int split = 0; split < p.splits; ++split) {
    const int s_lo = split * steps / p.splits, s_hi = (split + 1) * steps / p.splits;
#pragma unroll
    for (int t = 0; t < kPromptGroups; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) c[t][i] = 0.f;
    for (int st = s_lo; st < s_hi; ++st) {
      if (st + 1 < steps) {
        next_a.load(p, na, (st + 1) * STEP, tig);
        next_b.load(p, nb, (st + 1) * STEP, tig);
      }
#pragma unroll
      for (int q = 0; q < SUBS; ++q) {
        const int u = st * SUBS + q;
        if (u + 1 < units) {
          stage(u + 1);
          cp_wait<1>();
        } else {
          cp_wait<0>();
        }
        __syncthreads();
        const uint4* buffer = staged + (u & 1) * kBuffer;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          uint32_t pa[4], pb[4];   // MMAs 2w and 2w + 1 of the sub-step: pairs 4w .. 4w + 3 of rows gid and gid + 8
          uint4 b[kPromptGroups];   // each column's values 8w .. 8w + 7 of lane tig's 32
          if constexpr (kSplit<WF>) {   // the decode tile's MMAs: each twice, hi then lo
            uint32_t la[4], lb[4];
            wa.split_quad(q, w, pa, la);
            wb.split_quad(q, w, pb, lb);
#pragma unroll
            for (int t = 0; t < kPromptGroups; ++t) if (t < live) b[t] = buffer[chunk_at(8 * t + gid, 4 * tig + w)];
#pragma unroll
            for (int t = 0; t < kPromptGroups; ++t) {
              if (t >= live) continue;
              mma(c[t], pa[0], pb[0], pa[1], pb[1], b[t].x, b[t].y);
              mma(c[t], la[0], lb[0], la[1], lb[1], b[t].x, b[t].y);
            }
#pragma unroll
            for (int t = 0; t < kPromptGroups; ++t) {
              if (t >= live) continue;
              mma(c[t], pa[2], pb[2], pa[3], pb[3], b[t].z, b[t].w);
              mma(c[t], la[2], lb[2], la[3], lb[3], b[t].z, b[t].w);
            }
          } else {
            wa.quad(q, w, pa);
            wb.quad(q, w, pb);
#pragma unroll
            for (int t = 0; t < kPromptGroups; ++t) if (t < live) b[t] = buffer[chunk_at(8 * t + gid, 4 * tig + w)];
#pragma unroll
            for (int t = 0; t < kPromptGroups; ++t) if (t < live) mma(c[t], pa[0], pb[0], pa[1], pb[1], b[t].x, b[t].y);
#pragma unroll
            for (int t = 0; t < kPromptGroups; ++t) if (t < live) mma(c[t], pa[2], pb[2], pa[3], pb[3], b[t].z, b[t].w);
          }
        }
        __syncthreads();   // every warp is done with the buffer before stage() fills it again
      }
      if (st + 1 < steps) {
        wa = next_a;
        wb = next_b;
      }
    }
    if constexpr (SPLIT) {   // the decode tile's reduce: the first split's sum, then each later one added in order
#pragma unroll
      for (int t = 0; t < kPromptGroups; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) total[t][i] = split ? total[t][i] + c[t][i] : c[t][i];
    }
  }
  // c0, c1: matrix row gid, columns 2 tig and 2 tig + 1; c2, c3: matrix row gid + 8.
  if (p.glu) {
#pragma unroll
    for (int t = 0; t < kPromptGroups; ++t)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int n = r0 + gid, tok = t0 + 8 * t + 2 * tig + i;
        if (n < p.N && tok < p.M)
          finish_glu<WF, ACT>(p, tok, n, SPLIT ? total[t][i] : c[t][i], SPLIT ? total[t][i + 2] : c[t][i + 2]);
      }
    return;
  }
#pragma unroll
  for (int t = 0; t < kPromptGroups; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int n = r0 + gid + (i >> 1) * 8, tok = t0 + 8 * t + 2 * tig + (i & 1);
      if (n < p.N && tok < p.M) finish<WF, ACT>(p, tok, n, SPLIT ? total[t][i] : c[t][i]);
    }
}

template <int WF, int ACT, int PW, int PT>
cudaError_t launch_tile(const Problem& p, cudaStream_t stream) {
  const dim3 grid((p.M + PT - 1) / PT, (p.N + 16 * PW - 1) / (16 * PW));
  if (grid.y > 65535u) return cudaErrorInvalidValue;
  if (p.splits > 1) return launch_kernel(prompt_kernel<WF, ACT, true, PW, PT>, grid, dim3(PW * 32), stream, p);
  return launch_kernel(prompt_kernel<WF, ACT, false, PW, PT>, grid, dim3(PW * 32), stream, p);
}

template <int WF, int ACT>
cudaError_t launch_prompt(const Problem& p, cudaStream_t stream) {
  switch (crit_lookup_prompt_tile(p.weights, p.N, p.K, p.M)) {
#define CRIT_PROMPT_CONFIG(index, warps, tokens) \
  case index:                                    \
    return launch_tile<WF, ACT, warps, tokens>(p, stream);
    CRIT_CONFIGS_prompt_tile(CRIT_PROMPT_CONFIG)
#undef CRIT_PROMPT_CONFIG
    default: return cudaErrorInvalidValue;
  }
}

}  // namespace

cudaError_t enqueue_prompt(const Problem& given, cudaStream_t stream) {
  Problem p = given;
  if (p.M < 1 || p.K < kStep || p.K % kStep || p.N < 1 || p.splits < 1 || p.splits > kMaxSplits)
    return cudaErrorInvalidValue;
  if (p.acts == kActNvfp4Mma) return cudaErrorInvalidValue;   // the FP4 MMA sums in an order of its own: no prompt tile
  if (const cudaError_t e = prepass(p, stream); e != cudaSuccess) return e;
  if (p.weights == kNvfp4 && p.acts == kActBf16) return launch_prompt<kNvfp4, kActBf16>(p, stream);
#ifndef CRIT_SWEEP_BUILD   // a sweep build times the activations as they are (the bench's call)
  if (p.weights == kNvfp4 && p.acts == kActNvfp4) return launch_prompt<kNvfp4, kActNvfp4>(p, stream);
#endif
  if (p.weights == kFp8 && p.acts == kActBf16) return launch_prompt<kFp8, kActBf16>(p, stream);
#ifndef CRIT_SWEEP_BUILD   // a sweep build times the activations as they are (the bench's call)
  if (p.weights == kFp8 && p.acts == kActFp8Rows) return launch_prompt<kFp8, kActFp8Rows>(p, stream);
#endif
  if (p.weights == kMxfp8 && p.acts == kActBf16) return launch_prompt<kMxfp8, kActBf16>(p, stream);
  if (p.weights == kBf16 && p.acts == kActBf16) return launch_prompt<kBf16, kActBf16>(p, stream);
  if (p.weights == kInt4Zp && p.acts == kActBf16) return launch_prompt<kInt4Zp, kActBf16>(p, stream);
  return cudaErrorInvalidValue;
}

}  // namespace crit::kernels::mma_rows
