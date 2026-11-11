// Torch entry points over fp8_block.h: shape checks and the current stream. The matrix table is a uint8 CUDA tensor
// holding one Matrix per matrix and the job lists uint8 CUDA tensors of RowJob or GemmJob, as the Python layer packs
// them. E4M3 codes travel as uint8 tensors.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <algorithm>

#include "fp8_block.h"

namespace fb = crit::techniques::fp8_block;

namespace {
void check_ok(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }
void check_cuda(const torch::Tensor& t, const char* name, torch::ScalarType type) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
    TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}
fb::Acts acts_of(const torch::Tensor& codes, const torch::Tensor& scales, int64_t group) {
    check_cuda(codes, "codes", torch::kUInt8);
    check_cuda(scales, "scales", torch::kFloat32);
    TORCH_CHECK(codes.dim() == 2 && scales.dim() == 2 && codes.size(0) == scales.size(0), "codes [rows, cols] and "
                "scales [rows, cols / group]");
    return {codes.data_ptr<std::uint8_t>(), scales.data_ptr<float>(), static_cast<int>(codes.size(1)),
            static_cast<int>(scales.size(1)), static_cast<int>(group)};
}
}  // namespace

void rows(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, const torch::Tensor& codes,
          const torch::Tensor& scales, int64_t group, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(codes.device());
    check_cuda(table, "table", torch::kUInt8);
    check_cuda(jobs, "jobs", torch::kUInt8);
    check_cuda(out, "out", torch::kFloat32);
    TORCH_CHECK(jobs.numel() % static_cast<int64_t>(sizeof(fb::RowJob)) == 0, "jobs must hold RowJob records");
    check_ok(fb::enqueue_rows(reinterpret_cast<const fb::Matrix*>(table.data_ptr()),
                              reinterpret_cast<const fb::RowJob*>(jobs.data_ptr()),
                              static_cast<int>(jobs.numel() / static_cast<int64_t>(sizeof(fb::RowJob))),
                              static_cast<int>(max_rows), acts_of(codes, scales, group), out.data_ptr<float>(),
                              static_cast<int>(out.size(1)), at::cuda::getCurrentCUDAStream(),
                              static_cast<int>(std::min<int64_t>(codes.size(0), fb::kMaxJobTokens))), "rows");
}

void gemm(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, int64_t max_count,
          const torch::Tensor& codes, const torch::Tensor& scales, int64_t group, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(codes.device());
    check_cuda(table, "table", torch::kUInt8);
    check_cuda(jobs, "jobs", torch::kUInt8);
    check_cuda(out, "out", torch::kFloat32);
    TORCH_CHECK(jobs.numel() % static_cast<int64_t>(sizeof(fb::GemmJob)) == 0, "jobs must hold GemmJob records");
    check_ok(fb::enqueue_gemm(reinterpret_cast<const fb::Matrix*>(table.data_ptr()),
                              reinterpret_cast<const fb::GemmJob*>(jobs.data_ptr()),
                              static_cast<int>(jobs.numel() / static_cast<int64_t>(sizeof(fb::GemmJob))),
                              static_cast<int>(max_rows), static_cast<int>(max_count), acts_of(codes, scales, group),
                              out.data_ptr<float>(), static_cast<int>(out.size(1)), at::cuda::getCurrentCUDAStream()),
             "gemm");
}

void gemm_staged(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, int64_t max_count,
                 const torch::Tensor& codes, const torch::Tensor& scales, int64_t group, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(codes.device());
    check_cuda(table, "table", torch::kUInt8);
    check_cuda(jobs, "jobs", torch::kUInt8);
    check_cuda(out, "out", torch::kFloat32);
    TORCH_CHECK(jobs.numel() % static_cast<int64_t>(sizeof(fb::GemmJob)) == 0, "jobs must hold GemmJob records");
    check_ok(fb::enqueue_gemm_staged(reinterpret_cast<const fb::Matrix*>(table.data_ptr()),
                                     reinterpret_cast<const fb::GemmJob*>(jobs.data_ptr()),
                                     static_cast<int>(jobs.numel() / static_cast<int64_t>(sizeof(fb::GemmJob))),
                                     static_cast<int>(max_rows), static_cast<int>(max_count),
                                     acts_of(codes, scales, group), out.data_ptr<float>(),
                                     static_cast<int>(out.size(1)), at::cuda::getCurrentCUDAStream()),
             "gemm_staged");
}

namespace {
void* out_of(const torch::Tensor& out, bool& bf16) {
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dim() == 2, "out must be a contiguous 2-D CUDA tensor");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32 || out.scalar_type() == torch::kBFloat16, "out is fp32 or bf16");
    bf16 = out.scalar_type() == torch::kBFloat16;
    return out.data_ptr();
}
}  // namespace

void gemm_native(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, int64_t max_count,
                 const torch::Tensor& codes, const torch::Tensor& scales, int64_t group, bool mx, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(codes.device());
    check_cuda(table, "table", torch::kUInt8);
    check_cuda(jobs, "jobs", torch::kUInt8);
    bool bf16 = false;
    void* o = out_of(out, bf16);
    TORCH_CHECK(jobs.numel() % static_cast<int64_t>(sizeof(fb::GemmJob)) == 0, "jobs must hold GemmJob records");
    check_ok(fb::enqueue_gemm_native(reinterpret_cast<const fb::Matrix*>(table.data_ptr()),
                                     reinterpret_cast<const fb::GemmJob*>(jobs.data_ptr()),
                                     static_cast<int>(jobs.numel() / static_cast<int64_t>(sizeof(fb::GemmJob))),
                                     static_cast<int>(max_rows), static_cast<int>(max_count),
                                     acts_of(codes, scales, group), mx, o, bf16, static_cast<int>(out.size(1)),
                                     at::cuda::getCurrentCUDAStream()),
             "gemm_native");
}

void gemm_native_channel(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, int64_t max_count,
                         const torch::Tensor& codes, const torch::Tensor& scales, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(codes.device());
    check_cuda(table, "table", torch::kUInt8);
    check_cuda(jobs, "jobs", torch::kUInt8);
    bool bf16 = false;
    void* o = out_of(out, bf16);
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 1, "one activation scale a token: scales [tokens, 1]");
    TORCH_CHECK(jobs.numel() % static_cast<int64_t>(sizeof(fb::GemmJob)) == 0, "jobs must hold GemmJob records");
    check_ok(fb::enqueue_gemm_native_channel(reinterpret_cast<const fb::Matrix*>(table.data_ptr()),
                                             reinterpret_cast<const fb::GemmJob*>(jobs.data_ptr()),
                                             static_cast<int>(jobs.numel() / static_cast<int64_t>(sizeof(fb::GemmJob))),
                                             static_cast<int>(max_rows), static_cast<int>(max_count),
                                             acts_of(codes, scales, codes.size(1)), o, bf16,
                                             static_cast<int>(out.size(1)), at::cuda::getCurrentCUDAStream()),
             "gemm_native_channel");
}

void quantize(const torch::Tensor& x, int64_t group, int64_t kind, torch::Tensor& codes, torch::Tensor& scales) {
    const c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2, "x must be a contiguous 2-D CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kBFloat16, "x must be fp32 or bf16");
    TORCH_CHECK(kind == 0 || kind == 1, "kind: 0 for MXFP8 scales, 1 for amax / 448");
    check_ok(fb::enqueue_quantize(x.data_ptr(), x.scalar_type() == torch::kBFloat16, static_cast<int>(x.size(1)),
                                  static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
                                  static_cast<fb::ScaleKind>(kind), acts_of(codes, scales, group),
                                  at::cuda::getCurrentCUDAStream()), "quantize");
}
