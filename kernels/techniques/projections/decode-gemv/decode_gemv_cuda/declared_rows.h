// decode-gemv's declared decode rows (structures rows-int-block and rows-fp8-block): the kernel core's declared rows
// (crit_kcore/declared_rows.cuh) on multiply.cuh's declared paths, 1 to 8 tokens a job, a job a matrix of a table.
// int-block-experts' and fp8-block-rows' rows forward here (their kernel.json aliases), with the same bits.
#pragma once
#include <cuda_runtime.h>

#include "crit_kcore/declared_rows.cuh"

namespace crit::kernels::decode_gemv {

// The declared integer rows: 4-bit weights (E2M1 x 2 or INT4, E4M3 per 16) against int8 activations per 16 or 32.
// matrices, jobs: device arrays; out [..][out_ld] FP32. The tuning's key: (max_rows, acts.ld), acts.rows rows (0:
// the job count).
cudaError_t enqueue_rows_int_block(const crit::kcore::declared::IntBlockMatrix* matrices,
                                   const crit::kcore::declared::RowJob* jobs, int job_count, int max_rows,
                                   const crit::kcore::declared::IntBlockActs& acts, float* out, int out_ld,
                                   cudaStream_t stream);
// The declared FP8 rows: E4M3 weights with a scale grid against E4M3 activations per group (sm_89 on). tokens: the
// call's tokens (at most 8), which only picks the tuned configuration.
cudaError_t enqueue_rows_fp8_block(const crit::kcore::declared::Fp8BlockMatrix* matrices,
                                   const crit::kcore::declared::RowJob* jobs, int job_count, int max_rows,
                                   const crit::kcore::declared::Fp8BlockActs& acts, float* out, int out_ld,
                                   cudaStream_t stream, int tokens);

}  // namespace crit::kernels::decode_gemv
