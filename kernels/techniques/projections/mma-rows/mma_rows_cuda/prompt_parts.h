// The prompt tiles' parts (prompt_tile.cu's prompt-tile, fp4_mma.cu's fp4-prompt-tile): a tile's splits of K in
// several CTAs (grid z), part 0 adding its splits' chains in order, every later split's chain stored on its own, the
// tile's last CTA adding them in split order, so every output keeps the one-CTA order's bits.
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

#include "mma_rows.h"

namespace crit::kernels::mma_rows {
namespace {

// The partial sums of a tile's parts (parts > 1): slot 0 holds part 0's total, slot 1 + s - h0 split s's chain for
// every split s >= h0 (h0: part 0's splits), each slot a tile's fragments in thread order (coalesced stores and loads).
__host__ __device__ constexpr int first_part_splits(int splits, int parts) { return splits / parts; }
__host__ __device__ constexpr int part_slots(int splits, int parts) {
  return parts > 1 ? splits - first_part_splits(splits, parts) + 1 : 0;
}

// CTAs a multiprocessor of the current device holds of `kernel` launched with `threads` threads and `smem` bytes of
// dynamic shared memory (0 on an error), and the device's multiprocessors.
template <class Kernel>
int resident(Kernel kernel, int threads, std::size_t smem = 0) {
  int n = 0;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, threads, smem) == cudaSuccess ? n : 0;
}
inline int multiprocessors() {
  int device = 0, n = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
    return 0;
  return n;
}

// The parts a tile's splits of K run in (grid z): the tuning's `knob` (PK) where it gives one (fewer where the
// partial sums do not fit), else the fewest splits a CTA whose CTAs fit one wave of `slots` resident CTAs and whose
// partial sums fit (each as few parts as keep that many splits a CTA): a grid that already fills the GPU keeps one
// part. Past one wave, the stored chains' traffic and the short CTAs' fixed costs took more than the parts gave on
// BF16 weights (on an RTX 5090 a 12288 x 5120 BF16 matrix at 64 rows went from 89 to 130 us in 7 parts), while
// narrow NVFP4 and FP8 ones gained (a 324 x 10240 NVFP4 matrix at 2,048 rows 1.4x in 5): PK tunes those. Where the
// tiles already fill a third of the resident CTAs, parts are taken only while a CTA keeps kPartValues of K or more
// (a split's chain stored and added again costs a short CTA more than it saves: a 4608 x 1152 BF16 matrix at 128
// rows went from 23.1 to 27.2 us in 2 parts of 640 values), and not for a BF16 matrix's weight stream (one tile of
// rows, the weights used as they are: 76 such CTAs of a 9728 x 4608 matrix read 1.25 TB/s, and 2 parts took 5-8%
// longer). One part needs no partial sums or counters.
constexpr int kPartValues = 1024;
inline int prompt_parts(const Problem& p, int tiles, int m_tiles, int tile_sums, int slots, int knob) {
  if (p.splits < 2 || !p.partial_size || tiles > p.counter_count) return 1;
  // `per` splits a CTA: as few parts as give it, and whether their partial sums fit.
  auto parts_of = [&](int per) { return (p.splits + per - 1) / per; };
  auto fits = [&](int parts) {
    return parts == 1 || (std::size_t)part_slots(p.splits, parts) * tiles * tile_sums <= p.partial_size;
  };
  if (knob > 0) {   // the tuning's parts, or the most that fit with as many splits a CTA or more
    const int asked = knob < p.splits ? knob : p.splits;
    for (int per = (p.splits + asked - 1) / asked;; ++per)
      if (fits(parts_of(per))) return parts_of(per);
  }
  if (slots <= 0) return 1;
  const bool filling = 3L * tiles >= slots;
  if (filling && p.weights == kBf16 && m_tiles == 1) return 1;
  int parts = 1;
  for (int per = p.splits; per >= 1; --per) {   // fewer splits a CTA while the CTAs fit one wave and their sums fit
    const int k = parts_of(per);
    const int most = (p.splits + k - 1) / k;   // the splits of the largest part
    if ((long)tiles * k > slots || !fits(k) || (filling && (long)most * p.K < (long)kPartValues * p.splits)) break;
    parts = k;
  }
  return parts;
}

}  // namespace
}  // namespace crit::kernels::mma_rows
