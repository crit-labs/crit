// The block-scaled FP4 MMA path of mma-rows (activations kActNvfp4Mma): NVFP4 weights with blocked scales times the
// activations quantized to NVFP4, both as E2M1 codes with their UE4M3 block scales on mma.sync kind::mxf4nvf4.
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

#include "mma_rows.h"

namespace crit::kernels::mma_rows {

constexpr int kFp4Step = 128;    // the FP4 MMA's step: 64 bytes of a row (two k64 MMAs), a quad's 16 bytes a lane
constexpr int kFp4Hoist = 2;     // the FP4 MMA's steps of weights a warp loads together, untuned (knob HOIST)

// Whether this build multiplies kActNvfp4Mma on the current device (the code for its architecture has the MMA).
bool fp4_mma(cudaStream_t stream);
// CTAs of kMaxWarps warps a multiprocessor of the current device holds of the one-row FP4 MMA (plan()'s slots for
// acts nvfp4_mma, through decode_ctas); 0 on an error.
int fp4_ctas();
// Scratch bytes kActNvfp4Mma's activations take: codes [M][K / 2], then blocked scales of ceil(M / 128) * 128 rows.
inline std::size_t fp4_scratch_bytes(int M, int K) {
  return ((std::size_t)M * K / 2 + 15) / 16 * 16 + (std::size_t)(M + 127) / 128 * 128 * ((K / 16 + 3) / 4 * 4);
}

// kActNvfp4Mma's call (enqueue sends it here): the NVFP4 quantizer into scratch, then the FP4 MMA, both on `stream`
// (the second waits on the first through programmatic dependent launch).
cudaError_t enqueue_fp4(const Problem& p, cudaStream_t stream);

}  // namespace crit::kernels::mma_rows
