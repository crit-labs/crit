// crit's kernel core: shared-memory staging for block-scaled GEMMs in a declared block order (kernels/core/README.md,
// "blockscaled").
//
// A stage holds one block column (the block's bytes of each of a CTA's weight rows and token rows, and their scales);
// stages are filled in the visiting order of blockscaled_order.cuh's LaneVisits through load.cuh's cp.async ring
// (STAGES - 1 in flight, one barrier a stage), the multistage mainloop of CUTLASS (BSD-3-Clause; the mechanism). Rows
// are stored at a stride of one block, so a warp's fragment reads (eight rows of a block) are contiguous and free of
// bank conflicts without a swizzle. Scales are stored as FP32 with rows g and g + 8 of each 16-row tile side by side
// (pair_slot), so a thread reads both of its rows' scales with one 8-byte load.
#pragma once
#include "crit_kcore/load.cuh"

namespace crit::kcore::blockscaled {

// The slot of row r's scale in a stage's scale array: rows g and g + 8 of each 16-row tile side by side.
__host__ __device__ constexpr int pair_slot(int r) { return (r & ~15) | ((r & 7) << 1) | ((r >> 3) & 1); }

}  // namespace crit::kcore::blockscaled
