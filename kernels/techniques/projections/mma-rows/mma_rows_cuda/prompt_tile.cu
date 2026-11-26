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
// (cp.async, three buffers and one barrier a sub-step, a swizzle that keeps the B fragments' 16-byte reads free of
// bank conflicts).
//
// The splits of K run in `parts` CTAs a tile (grid z, knob PK): one part, a CTA runs every split itself, one after
// another, adding each split's chain into a running total as the decode tile's last CTA adds the partial sums (no
// partial or counter buffers). More parts, part z takes the splits [z S / parts, (z + 1) S / parts): part 0 adds its
// splits' chains in order (the prefix that starts at split 0, so the same additions) and stores that total, every
// other part stores each of its splits' chains on its own (FP32 addition is not associative: a later run of splits
// may not be added before the prefix), and the tile's last CTA to finish (a counter, left zero) adds them to the
// total in split order and finishes the outputs. A short prompt or a narrow matrix then fills the multiprocessors
// with up to S times the tiles' CTAs, every output's additions unchanged.
#include "mma_rows.h"
#include "fp4_mma.h"
#include "tile_parts.cuh"
#include "prompt_parts.h"
#include "crit_tuning.h"

namespace crit::kernels::mma_rows {
namespace {

// The prompt tile's knobs (kernel.json, structure prompt-tile; crit_tuning.h picks them per matrix and row count):
//   PW  warps a CTA, each its own 16 matrix rows (8 untuned)
//   PT  activation rows a CTA (kPromptTokens, 64, untuned)
//   PK  CTAs a tile over K's splits (0 untuned: prompt_parts()' rule)
//   PS  staged sub-steps in shared memory: 2 (a barrier before and after each sub-step's MMAs) or 3 (one barrier a
//       sub-step: the buffer a copy refills was read a sub-step before)
//   PF  1: a step's weights loaded a step ahead (two steps of weights in registers); 0: at the step, the other
//       warps of the multiprocessor hiding the load
//   PR  16-row groups a warp (each B fragment read once for them; 2 loads the weights at the step, PF 0); 0
//       untuned: launch_groups()' rule from the grid's waves
// None changes an output's chain: every output keeps the decode tile's bits.
constexpr int kChunks = 16;   // a staged sub-step of one activation row: 4 lanes' 32 values in 16-byte chunks (chunk
                              // 4 tig + w: values 8w .. 8w + 7 of lane tig's)

// The shared-memory chunk of activation row `tok`'s chunk c (4 tig + w): within the row's 16, its quad position's
// bits 1 and 0 XORed with tig's high bit and the row's low bit, so that each 8-lane phase of a B-fragment read (two
// rows, the four quad positions) covers all 32 banks.
__device__ __forceinline__ int chunk_at(int tok, int c) { return tok * kChunks + (c ^ (2 * (c >> 3)) ^ (tok & 1)); }

// One CTA: 16 PW PR matrix rows (blockIdx.y) times PT activation rows (blockIdx.x) over part blockIdx.z of K's
// splits. SPLIT: the plan has more than one split of K (a running total beside each split's chain).
template <int WF, int ACT, bool SPLIT, int PW, int PR, int PT, int PS, int PF>
__global__ void __launch_bounds__(PW * 32, 1) prompt_kernel(const Problem p) {
  static_assert(PS == 2 || PS == 3, "two or three staged sub-steps");
  constexpr int SUBS = Row<WF>::SUBS, STEP = 128 * SUBS;
  constexpr int kPromptRows = 16 * PW * PR;   // matrix rows a CTA
  constexpr int kPromptGroups = PT / 8;       // MMA columns of 8 activation rows a CTA
  constexpr int kBuffer = PT * kChunks;       // 16-byte chunks a staged sub-step takes
  constexpr int kFragments = PR * kPromptGroups * 4;   // a thread's sums
  constexpr int kTileSums = PW * 32 * kFragments;      // FP32 sums a tile's slot holds
  __shared__ uint4 staged[PS * kBuffer];
  // The reduce's flag shares the buffers (three of 64 rows are the 48 KB static limit): it is set after a barrier
  // every warp reaches with its sub-steps done and every copy landed.
  int& last = *reinterpret_cast<int*>(staged);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int t0 = blockIdx.x * PT, r0 = blockIdx.y * kPromptRows + warp * 16 * PR;   // the warp's first matrix row
  const int steps = (p.K + STEP - 1) / STEP;
  const int parts = gridDim.z, part = blockIdx.z;
  const int sp_lo = part * p.splits / parts, sp_hi = (part + 1) * p.splits / parts;   // this CTA's splits
  const int u_lo = sp_lo * steps / p.splits * SUBS, u_hi = sp_hi * steps / p.splits * SUBS;   // and sub-steps
  int na[PR], nb[PR];   // group g's rows gid and gid + 8 (a row past N reads row N - 1)
#pragma unroll
  for (int g = 0; g < PR; ++g) na[g] = min(r0 + 16 * g + gid, p.N - 1), nb[g] = min(r0 + 16 * g + gid + 8, p.N - 1);
  const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(p.x);
  // MMA columns of 8 activation rows holding a row below M: a partly filled last tile stages and multiplies only
  // those (the others' outputs are not stored), so a 65-row prompt's second CTA does one column's work, not eight.
  const int live = min(kPromptGroups, (p.M - t0 + 7) / 8);

  // Sub-step u (step u / SUBS, sub-step u % SUBS) of the CTA's activation rows into buffer u % PS; zeros for a row
  // past M or a sub-step past K, as act_frags gives them.
  auto stage = [&](int u) {
    const int st = u / SUBS, q = u - st * SUBS;
    uint4* buffer = staged + (u % PS) * kBuffer;
    for (int i = threadIdx.x; i < live * 8 * kChunks; i += PW * 32) {
      const int tok = i / kChunks, c = i - tok * kChunks;
      const int kb = st * STEP + 32 * (SUBS * (c >> 2) + q);
      const bool valid = t0 + tok < p.M && kb < p.K;
      const __nv_bfloat16* src = valid ? x + (size_t)(t0 + tok) * p.ldx + kb + 8 * (c & 3) : x;
      cp_async16(buffer + chunk_at(tok, c), src, valid);
    }
    cp_commit();
  };
  // Slot `slot` of this tile's partial sums: a thread's fragment (g, t, i) at ((g G + t) 4 + i) PW 32 + threadIdx.x.
  const int tile = blockIdx.y * gridDim.x + blockIdx.x;
  auto slot_at = [&](int slot) { return p.partial + ((size_t)slot * gridDim.x * gridDim.y + tile) * kTileSums; };
  auto at_of = [&](int g, int t, int i) { return ((g * kPromptGroups + t) * 4 + i) * PW * 32 + threadIdx.x; };
  auto store = [&](int slot, const float (&v)[PR][kPromptGroups][4]) {
    float* at = slot_at(slot);
#pragma unroll
    for (int g = 0; g < PR; ++g)
#pragma unroll
      for (int t = 0; t < kPromptGroups; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          if (t < live) __stcg(at + at_of(g, t, i), v[g][t][i]);
  };

  float c[PR][kPromptGroups][4];
  float total[SPLIT ? PR : 1][SPLIT ? kPromptGroups : 1][4];
  Row<WF> wa[PR], wb[PR], next_a[PR], next_b[PR];
  auto load = [&](Row<WF>(&a)[PR], Row<WF>(&b)[PR], int st) {
#pragma unroll
    for (int g = 0; g < PR; ++g) {
      a[g].load(p, na[g], st * STEP, tig);
      b[g].load(p, nb[g], st * STEP, tig);
    }
  };
  if (u_lo < u_hi) load(wa, wb, u_lo / SUBS);   // the weights do not depend on the previous kernel: read them first
  wait_prior();
  if (u_lo < u_hi) stage(u_lo);
  if (PS == 3 && u_lo + 1 < u_hi) stage(u_lo + 1);
  for (int split = sp_lo; split < sp_hi; ++split) {
    const int s_lo = split * steps / p.splits, s_hi = (split + 1) * steps / p.splits;
#pragma unroll
    for (int g = 0; g < PR; ++g)
#pragma unroll
      for (int t = 0; t < kPromptGroups; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[g][t][i] = 0.f;
    for (int st = s_lo; st < s_hi; ++st) {
      if constexpr (PF) {
        if ((st + 1) * SUBS < u_hi) load(next_a, next_b, st + 1);
      } else if (st * SUBS != u_lo) {   // the CTA's first step was read before waiting for the previous kernel
        load(wa, wb, st);
      }
#pragma unroll
      for (int q = 0; q < SUBS; ++q) {
        const int u = st * SUBS + q;
        if constexpr (PS == 2) {
          if (u + 1 < u_hi) {
            stage(u + 1);
            cp_wait<1>();
          } else {
            cp_wait<0>();
          }
          __syncthreads();
        } else {   // sub-steps u and u + 1 in flight: u landed, and every warp is done with u - 1's buffer
          if (u + 1 < u_hi) cp_wait<1>();
          else cp_wait<0>();
          __syncthreads();
          if (u + 2 < u_hi) stage(u + 2);
        }
        const uint4* buffer = staged + (u % PS) * kBuffer;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          // MMAs 2w and 2w + 1 of the sub-step: pairs 4w .. 4w + 3 of each group's rows gid and gid + 8; each
          // column's B fragment (values 8w .. 8w + 7 of lane tig's 32): one group, every column's read first and
          // each accumulator's MMAs 8 columns apart; two, a column's read just before its MMAs for both groups (fewer
          // registers). Each accumulator's MMAs run in the decode tile's order (2w, then 2w + 1; INT4: hi, then lo).
          uint32_t pa[PR][4], pb[PR][4];
          if constexpr (kSplit<WF>) {
            uint32_t la[PR][4], lb[PR][4];
#pragma unroll
            for (int g = 0; g < PR; ++g) {
              wa[g].split_quad(q, w, pa[g], la[g]);
              wb[g].split_quad(q, w, pb[g], lb[g]);
            }
            if constexpr (PR == 1) {   // every column's fragment first, each accumulator's MMAs 8 columns apart
              uint4 b[kPromptGroups];
#pragma unroll
              for (int t = 0; t < kPromptGroups; ++t) if (t < live) b[t] = buffer[chunk_at(8 * t + gid, 4 * tig + w)];
#pragma unroll
              for (int t = 0; t < kPromptGroups; ++t) {
                if (t >= live) continue;
                mma(c[0][t], pa[0][0], pb[0][0], pa[0][1], pb[0][1], b[t].x, b[t].y);
                mma(c[0][t], la[0][0], lb[0][0], la[0][1], lb[0][1], b[t].x, b[t].y);
              }
#pragma unroll
              for (int t = 0; t < kPromptGroups; ++t) {
                if (t >= live) continue;
                mma(c[0][t], pa[0][2], pb[0][2], pa[0][3], pb[0][3], b[t].z, b[t].w);
                mma(c[0][t], la[0][2], lb[0][2], la[0][3], lb[0][3], b[t].z, b[t].w);
              }
              continue;
            }
#pragma unroll
            for (int t = 0; t < kPromptGroups; ++t) {   // a column's fragment read once for the groups
              if (t >= live) continue;
              const uint4 v = buffer[chunk_at(8 * t + gid, 4 * tig + w)];
#pragma unroll
              for (int g = 0; g < PR; ++g) {
                mma(c[g][t], pa[g][0], pb[g][0], pa[g][1], pb[g][1], v.x, v.y);
                mma(c[g][t], la[g][0], lb[g][0], la[g][1], lb[g][1], v.x, v.y);
                mma(c[g][t], pa[g][2], pb[g][2], pa[g][3], pb[g][3], v.z, v.w);
                mma(c[g][t], la[g][2], lb[g][2], la[g][3], lb[g][3], v.z, v.w);
              }
            }
          } else {
#pragma unroll
            for (int g = 0; g < PR; ++g) {
              wa[g].quad(q, w, pa[g]);
              wb[g].quad(q, w, pb[g]);
            }
            if constexpr (PR == 1) {   // every column's fragment first, each accumulator's MMAs 8 columns apart
              uint4 b[kPromptGroups];
#pragma unroll
              for (int t = 0; t < kPromptGroups; ++t) if (t < live) b[t] = buffer[chunk_at(8 * t + gid, 4 * tig + w)];
#pragma unroll
              for (int t = 0; t < kPromptGroups; ++t)
                if (t < live) mma(c[0][t], pa[0][0], pb[0][0], pa[0][1], pb[0][1], b[t].x, b[t].y);
#pragma unroll
              for (int t = 0; t < kPromptGroups; ++t)
                if (t < live) mma(c[0][t], pa[0][2], pb[0][2], pa[0][3], pb[0][3], b[t].z, b[t].w);
              continue;
            }
#pragma unroll
            for (int t = 0; t < kPromptGroups; ++t) {   // a column's fragment read once for the groups
              if (t >= live) continue;
              const uint4 v = buffer[chunk_at(8 * t + gid, 4 * tig + w)];
#pragma unroll
              for (int g = 0; g < PR; ++g) {
                mma(c[g][t], pa[g][0], pb[g][0], pa[g][1], pb[g][1], v.x, v.y);
                mma(c[g][t], pa[g][2], pb[g][2], pa[g][3], pb[g][3], v.z, v.w);
              }
            }
          }
        }
        if constexpr (PS == 2) __syncthreads();   // every warp is done with the buffer before stage() fills it
      }
      if (PF && (st + 1) * SUBS < u_hi) {
#pragma unroll
        for (int g = 0; g < PR; ++g) wa[g] = next_a[g], wb[g] = next_b[g];
      }
    }
    if constexpr (SPLIT) {
      if (part == 0) {   // the decode tile's reduce: the first split's sum, then each later one added in order
#pragma unroll
        for (int g = 0; g < PR; ++g)
#pragma unroll
          for (int t = 0; t < kPromptGroups; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) total[g][t][i] = split ? total[g][t][i] + c[g][t][i] : c[g][t][i];
      } else {   // a later part: each split's chain on its own
        store(1 + split - first_part_splits(p.splits, parts), c);
      }
    }
  }
  if constexpr (SPLIT) {
    if (parts > 1) {   // part 0's total stored too; the tile's last CTA adds the chains to it in split order
      if (part == 0) store(0, total);
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) last = atomicAdd(p.counters + tile, 1) == parts - 1;
      __syncthreads();
      if (!last) return;
      __threadfence();
      if (threadIdx.x == 0) p.counters[tile] = 0;   // every part has counted: ready for the next call
      // Each output's slots in order; a slot's loads of every output a thread holds at once (independent loads in
      // flight, not one chain of loads an output).
      const int slots = part_slots(p.splits, parts);
#pragma unroll
      for (int g = 0; g < PR; ++g)
#pragma unroll
        for (int t = 0; t < kPromptGroups; ++t)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            if (t < live) total[g][t][i] = __ldcg(slot_at(0) + at_of(g, t, i));
#pragma unroll 2
      for (int s = 1; s < slots; ++s) {
        const float* at = slot_at(s);
#pragma unroll
        for (int g = 0; g < PR; ++g)
#pragma unroll
          for (int t = 0; t < kPromptGroups; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i)
              if (t < live) total[g][t][i] += __ldcg(at + at_of(g, t, i));
      }
    }
  }
  // c0, c1: a group's matrix row gid, columns 2 tig and 2 tig + 1; c2, c3: its row gid + 8.
#pragma unroll
  for (int g = 0; g < PR; ++g) {
    if (p.glu) {
#pragma unroll
      for (int t = 0; t < kPromptGroups; ++t)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int n = r0 + 16 * g + gid, tok = t0 + 8 * t + 2 * tig + i;
          if (n < p.N && tok < p.M)
            finish_glu<WF, ACT>(p, tok, n, SPLIT ? total[g][t][i] : c[g][t][i],
                                SPLIT ? total[g][t][i + 2] : c[g][t][i + 2]);
        }
      continue;
    }
#pragma unroll
    for (int t = 0; t < kPromptGroups; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = r0 + 16 * g + gid + (i >> 1) * 8, tok = t0 + 8 * t + 2 * tig + (i & 1);
        if (n < p.N && tok < p.M) finish<WF, ACT>(p, tok, n, SPLIT ? total[g][t][i] : c[g][t][i]);
      }
  }
}

// One configuration's launch (ctas: its CTAs counted, nothing launched).
template <int WF, int ACT, int PW, int PT, int PS, int PF, int PR>
cudaError_t launch_tile(const Problem& p, int knob, cudaStream_t stream, int* ctas) {
  void (*const split_kernel)(const Problem) = prompt_kernel<WF, ACT, true, PW, PR, PT, PS, PF>;
  dim3 grid((p.M + PT - 1) / PT, (p.N + 16 * PW * PR - 1) / (16 * PW * PR));
  if (grid.y > 65535u) return cudaErrorInvalidValue;
  if (p.splits > 1) {
    static const int slots = multiprocessors() * resident(split_kernel, PW * 32);
    grid.z = prompt_parts(p, grid.x * grid.y, grid.x, PW * 32 * PR * PT / 2, slots, knob);
  }
  if (ctas) {
    *ctas = grid.x * grid.y * grid.z;
    return cudaSuccess;
  }
  if (p.splits > 1) return launch_kernel(split_kernel, grid, dim3(PW * 32), stream, p);
  return launch_kernel(prompt_kernel<WF, ACT, false, PW, PR, PT, PS, PF>, grid, dim3(PW * 32), stream, p);
}

// PR 0: two 16-row groups a warp (loading each step's weights at the step: their registers are the second group's)
// where the grid of the larger tiles takes no more waves of resident CTAs, times their twice the work, than the
// grid of one group a warp, a sixteenth more allowed (what the larger tiles gain where both grids' last waves are
// full: on many waves their tails cost both about the same), K has 8 steps or more, and the call is not a BF16
// matrix's one-tile weight stream; else one group. Each wave's last CTAs leave multiprocessors idle, and two groups a warp read each staged B fragment once
// for both. On an RTX 5090, 1.03-1.15x on FP8 weights from 16 rows on and on the other formats at 2,048 and 8,192
// rows; 0.84-0.96x where K had 1 to 3 steps (the weights' load at the step is not hidden) and on a 34816 x 5120 BF16
// matrix's weight stream at 16 and 64 rows (0.94x); 0.91-0.96x where the larger tiles' grid took more waves.
template <int WF, int ACT, int PW, int PT, int PS, int PF>
cudaError_t launch_groups(const Problem& p, int knob, cudaStream_t stream, int* ctas) {
  static const int one = multiprocessors() * resident(prompt_kernel<WF, ACT, false, PW, 1, PT, PS, PF>, PW * 32);
  static const int two = multiprocessors() * resident(prompt_kernel<WF, ACT, false, PW, 2, PT, PS, 0>, PW * 32);
  constexpr int STEP = 128 * Row<WF>::SUBS;
  const long m_tiles = (p.M + PT - 1) / PT;
  const long t1 = m_tiles * ((p.N + 16 * PW - 1) / (16 * PW)), t2 = m_tiles * ((p.N + 32 * PW - 1) / (32 * PW));
  const bool stream_only = WF == kBf16 && m_tiles == 1;
  const long waves1 = (t1 + one - 1) / one, waves2 = (t2 + two - 1) / two;
  if (one > 0 && two > 0 && p.K >= 8 * STEP && !stream_only && 32 * waves2 <= 17 * waves1)
    return launch_tile<WF, ACT, PW, PT, PS, 0, 2>(p, knob, stream, ctas);
  return launch_tile<WF, ACT, PW, PT, PS, PF, 1>(p, knob, stream, ctas);
}

// The configuration's launch: PR 1 or 2 as the tuning gives it, 0 by launch_groups' rule.
template <int WF, int ACT, int PW, int PT, int PS, int PF, int PR>
cudaError_t launch_config(const Problem& p, int knob, cudaStream_t stream, int* ctas) {
  if constexpr (PR == 0) return launch_groups<WF, ACT, PW, PT, PS, PF>(p, knob, stream, ctas);
  else return launch_tile<WF, ACT, PW, PT, PS, PR == 2 ? 0 : PF, PR>(p, knob, stream, ctas);
}

template <int WF, int ACT>
cudaError_t launch_prompt(const Problem& p, cudaStream_t stream, int* ctas) {
  switch (crit_lookup_prompt_tile(p.weights, p.N, p.K, p.M)) {
#define CRIT_PROMPT_CONFIG(index, warps, tokens, parts, stages, ahead, groups) \
  case index:                                                                   \
    return launch_config<WF, ACT, warps, tokens, stages, ahead, groups>(p, parts, stream, ctas);
    CRIT_CONFIGS_prompt_tile(CRIT_PROMPT_CONFIG)
#undef CRIT_PROMPT_CONFIG
    default: return cudaErrorInvalidValue;
  }
}

// The call's launch (ctas: its CTAs counted instead).
cudaError_t dispatch_prompt(const Problem& p, cudaStream_t stream, int* ctas) {
  if (p.weights == kNvfp4 && p.acts == kActBf16) return launch_prompt<kNvfp4, kActBf16>(p, stream, ctas);
#ifndef CRIT_SWEEP_BUILD   // a sweep build times the activations as they are (the bench's call)
  if (p.weights == kNvfp4 && p.acts == kActNvfp4) return launch_prompt<kNvfp4, kActNvfp4>(p, stream, ctas);
#endif
  if (p.weights == kFp8 && p.acts == kActBf16) return launch_prompt<kFp8, kActBf16>(p, stream, ctas);
#ifndef CRIT_SWEEP_BUILD   // a sweep build times the activations as they are (the bench's call)
  if (p.weights == kFp8 && p.acts == kActFp8Rows) return launch_prompt<kFp8, kActFp8Rows>(p, stream, ctas);
#endif
  if (p.weights == kMxfp8 && p.acts == kActBf16) return launch_prompt<kMxfp8, kActBf16>(p, stream, ctas);
  if (p.weights == kBf16 && p.acts == kActBf16) return launch_prompt<kBf16, kActBf16>(p, stream, ctas);
  if (p.weights == kInt4Zp && p.acts == kActBf16) return launch_prompt<kInt4Zp, kActBf16>(p, stream, ctas);
  return cudaErrorInvalidValue;
}

bool valid_prompt(const Problem& p) {
  return p.M >= 1 && p.K >= kStep && p.K % kStep == 0 && p.N >= 1 && p.splits >= 1 && p.splits <= kMaxSplits;
}

}  // namespace

cudaError_t enqueue_prompt(const Problem& given, cudaStream_t stream) {
  Problem p = given;
  if (!valid_prompt(p)) return cudaErrorInvalidValue;
  if (!p.partial || !p.counters) p.partial_size = 0, p.counter_count = 0;   // its splits in one CTA a tile
  if (p.acts == kActNvfp4Mma) return enqueue_prompt_fp4(p, stream);   // the FP4 MMA's own prompt tile (fp4_mma.cu)
  if (const cudaError_t e = prepass(p, stream); e != cudaSuccess) return e;
  return dispatch_prompt(p, stream, nullptr);
}

int prompt_ctas(const Problem& p) {
  int ctas = 0;
  if (!valid_prompt(p)) return 0;
  if (p.acts == kActNvfp4Mma) return p.K % kFp4Step || p.splits > p.K / kFp4Step ? 0 : prompt_fp4_ctas(p);
  return dispatch_prompt(p, nullptr, &ctas) == cudaSuccess ? ctas : 0;
}

}  // namespace crit::kernels::mma_rows
