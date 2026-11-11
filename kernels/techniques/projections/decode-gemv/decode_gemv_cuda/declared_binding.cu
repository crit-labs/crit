// Torch entry points over declared_rows.h, the checks int-block-experts' and fp8-block-rows' rows made: the matrix
// table is a uint8 CUDA tensor of IntBlockMatrix or Fp8BlockMatrix records and the jobs one of RowJob records, as
// those packages' Python layers pack them (and decode_gemv.declared).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <algorithm>

#include "declared_rows.h"

namespace nr = crit::kernels::decode_gemv;
namespace kd = crit::kcore::declared;

namespace {
void check_ok(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }
void check_cuda(const torch::Tensor& t, const char* name, torch::ScalarType type) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
    TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}
int job_count(const torch::Tensor& jobs) {
    TORCH_CHECK(jobs.numel() % static_cast<int64_t>(sizeof(kd::RowJob)) == 0, "jobs must hold RowJob records");
    return static_cast<int>(jobs.numel() / static_cast<int64_t>(sizeof(kd::RowJob)));
}
}  // namespace

void rows_int_block(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows,
                    const torch::Tensor& values, const torch::Tensor& scales, int64_t block, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(values.device());
    check_cuda(table, "table", torch::kUInt8);
    check_cuda(jobs, "jobs", torch::kUInt8);
    check_cuda(out, "out", torch::kFloat32);
    check_cuda(values, "values", torch::kInt8);
    check_cuda(scales, "scales", torch::kFloat32);
    TORCH_CHECK(values.dim() == 2 && scales.dim() == 2 && values.size(0) == scales.size(0), "values [rows, cols] and "
                "scales [rows, cols / block]");
    const kd::IntBlockActs acts{values.data_ptr<std::int8_t>(), scales.data_ptr<float>(),
                                static_cast<int>(values.size(1)), static_cast<int>(scales.size(1)),
                                static_cast<int>(block), static_cast<int>(values.size(0))};
    check_ok(nr::enqueue_rows_int_block(reinterpret_cast<const kd::IntBlockMatrix*>(table.data_ptr()),
                                        reinterpret_cast<const kd::RowJob*>(jobs.data_ptr()), job_count(jobs),
                                        static_cast<int>(max_rows), acts, out.data_ptr<float>(),
                                        static_cast<int>(out.size(1)), at::cuda::getCurrentCUDAStream()),
             "rows_int_block");
}

void rows_fp8_block(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, const torch::Tensor& codes,
                    const torch::Tensor& scales, int64_t group, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(codes.device());
    const auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major * 10 + props->minor >= 89, "rows_fp8_block: the e4m3 MMA needs compute capability 8.9 on");
    check_cuda(table, "table", torch::kUInt8);
    check_cuda(jobs, "jobs", torch::kUInt8);
    check_cuda(out, "out", torch::kFloat32);
    check_cuda(codes, "codes", torch::kUInt8);
    check_cuda(scales, "scales", torch::kFloat32);
    TORCH_CHECK(codes.dim() == 2 && scales.dim() == 2 && codes.size(0) == scales.size(0), "codes [rows, cols] and "
                "scales [rows, cols / group]");
    const kd::Fp8BlockActs acts{codes.data_ptr<std::uint8_t>(), scales.data_ptr<float>(),
                                static_cast<int>(codes.size(1)), static_cast<int>(scales.size(1)),
                                static_cast<int>(group)};
    check_ok(nr::enqueue_rows_fp8_block(reinterpret_cast<const kd::Fp8BlockMatrix*>(table.data_ptr()),
                                        reinterpret_cast<const kd::RowJob*>(jobs.data_ptr()), job_count(jobs),
                                        static_cast<int>(max_rows), acts, out.data_ptr<float>(),
                                        static_cast<int>(out.size(1)), at::cuda::getCurrentCUDAStream(),
                                        static_cast<int>(std::min<int64_t>(codes.size(0), kd::kJobTokens))),
             "rows_fp8_block");
}
