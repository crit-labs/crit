// decode-gemv's declared decode rows (declared_rows.h): the kernel core's templates, dispatched through the build's
// crit_tuning.h (structures rows-int-block and rows-fp8-block, the knobs int-block-experts' and fp8-block-rows' rows
// had, the same defaults). No knob changes an output's operations or their order.
#include "declared_rows.h"
#include "crit_tuning.h"

namespace crit::kernels::decode_gemv {
namespace {

namespace kd = crit::kcore::declared;

template <int THREADS, int ROWS, int STREAM, int UNROLL>
void launch_int(const kd::IntBlockMatrix* matrices, const kd::RowJob* jobs, int job_count, int max_rows,
                const kd::IntBlockActs& acts, float* out, int out_ld, cudaStream_t stream) {
  constexpr int per_cta = THREADS / 16 * ROWS;
  const dim3 grid(unsigned((max_rows + per_cta - 1) / per_cta), unsigned(job_count));
  kd::int_rows_kernel<THREADS, ROWS, STREAM, UNROLL, kd::kJobTokens, kd::IntBlockMatrix, kd::IntBlockActs, kd::RowJob>
      <<<grid, THREADS, 0, stream>>>(matrices, jobs, acts, out, out_ld);
}

template <int WARPS, int AHEAD, int STREAM>
void launch_fp8(const kd::Fp8BlockMatrix* matrices, const kd::RowJob* jobs, int job_count, int max_rows,
                const kd::Fp8BlockActs& acts, float* out, int out_ld, cudaStream_t stream) {
  const dim3 grid(unsigned((max_rows + 15) / 16), unsigned(job_count));
  kd::fp8_rows_kernel<WARPS, AHEAD, STREAM, kd::kJobTokens, kd::Fp8BlockMatrix, kd::Fp8BlockActs, kd::RowJob>
      <<<grid, 32 * WARPS, 0, stream>>>(matrices, jobs, acts, out, out_ld);
}

bool bad_int(const kd::IntBlockActs& a) {
  return !a.values || !a.scales || (a.block != 16 && a.block != 32) || a.ld % 4;
}
bool bad_fp8(const kd::Fp8BlockActs& a) { return !a.codes || !a.scales || a.group <= 0 || a.group % 32 || a.ld % 8; }

}  // namespace

cudaError_t enqueue_rows_int_block(const kd::IntBlockMatrix* matrices, const kd::RowJob* jobs, int job_count,
                                   int max_rows, const kd::IntBlockActs& acts, float* out, int out_ld,
                                   cudaStream_t stream) {
  if (!matrices || !jobs || job_count < 0 || max_rows < 0 || !out || bad_int(acts)) return cudaErrorInvalidValue;
  if (job_count == 0 || max_rows == 0) return cudaSuccess;
  switch (crit_lookup_rows_int_block(max_rows, acts.ld, acts.rows > 0 ? acts.rows : job_count)) {
#define CRIT_DG_INT_ROWS(index, threads, rows, stream_, unroll)                                                    \
  case index:                                                                                                     \
    launch_int<threads, rows, stream_, unroll>(matrices, jobs, job_count, max_rows, acts, out, out_ld, stream); \
    break;
    CRIT_CONFIGS_rows_int_block(CRIT_DG_INT_ROWS)
#undef CRIT_DG_INT_ROWS
    default: return cudaErrorInvalidConfiguration;
  }
  return cudaGetLastError();
}

cudaError_t enqueue_rows_fp8_block(const kd::Fp8BlockMatrix* matrices, const kd::RowJob* jobs, int job_count,
                                   int max_rows, const kd::Fp8BlockActs& acts, float* out, int out_ld,
                                   cudaStream_t stream, int tokens) {
  if (!matrices || !jobs || job_count < 0 || max_rows < 0 || !out || bad_fp8(acts)) return cudaErrorInvalidValue;
  if (job_count == 0 || max_rows == 0) return cudaSuccess;
  switch (crit_lookup_rows_fp8_block(max_rows, acts.ld, tokens)) {
#define CRIT_DG_FP8_ROWS(index, warps, ahead, stream_)                                                    \
  case index:                                                                                            \
    launch_fp8<warps, ahead, stream_>(matrices, jobs, job_count, max_rows, acts, out, out_ld, stream); \
    break;
    CRIT_CONFIGS_rows_fp8_block(CRIT_DG_FP8_ROWS)
#undef CRIT_DG_FP8_ROWS
    default: return cudaErrorInvalidConfiguration;
  }
  return cudaGetLastError();
}

}  // namespace crit::kernels::decode_gemv
