// Torch entry points over int_block.h: shape checks and the current stream. The matrix table is a uint8 CUDA tensor
// holding one Matrix per matrix and the job lists uint8 CUDA tensors of RowJob or GemmJob, as the Python layer packs
// them.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <optional>
#include <vector>

#include "int_block.h"

namespace ib = crit::techniques::int_block;

namespace {
void check_ok(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }
void check_cuda(const torch::Tensor& t, const char* name, torch::ScalarType type) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
    TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}
ib::Acts acts_of(const torch::Tensor& values, const torch::Tensor& scales, int64_t block) {
    check_cuda(values, "values", torch::kInt8);
    check_cuda(scales, "scales", torch::kFloat32);
    TORCH_CHECK(values.dim() == 2 && scales.dim() == 2 && values.size(0) == scales.size(0), "values [rows, cols] and "
                "scales [rows, cols / block]");
    return {values.data_ptr<std::int8_t>(), scales.data_ptr<float>(), static_cast<int>(values.size(1)),
            static_cast<int>(scales.size(1)), static_cast<int>(block), static_cast<int>(values.size(0))};
}
}  // namespace

void rows(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, const torch::Tensor& values,
          const torch::Tensor& scales, int64_t block, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(values.device());
    check_cuda(table, "table", torch::kUInt8);
    check_cuda(jobs, "jobs", torch::kUInt8);
    check_cuda(out, "out", torch::kFloat32);
    TORCH_CHECK(jobs.numel() % static_cast<int64_t>(sizeof(ib::RowJob)) == 0, "jobs must hold RowJob records");
    check_ok(ib::enqueue_rows(reinterpret_cast<const ib::Matrix*>(table.data_ptr()),
                              reinterpret_cast<const ib::RowJob*>(jobs.data_ptr()),
                              static_cast<int>(jobs.numel() / static_cast<int64_t>(sizeof(ib::RowJob))),
                              static_cast<int>(max_rows), acts_of(values, scales, block), out.data_ptr<float>(),
                              static_cast<int>(out.size(1)), at::cuda::getCurrentCUDAStream()), "rows");
}

void gemm(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, int64_t max_count,
          const torch::Tensor& values, const torch::Tensor& scales, int64_t block, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(values.device());
    check_cuda(table, "table", torch::kUInt8);
    check_cuda(jobs, "jobs", torch::kUInt8);
    check_cuda(out, "out", torch::kFloat32);
    TORCH_CHECK(jobs.numel() % static_cast<int64_t>(sizeof(ib::GemmJob)) == 0, "jobs must hold GemmJob records");
    check_ok(ib::enqueue_gemm(reinterpret_cast<const ib::Matrix*>(table.data_ptr()),
                              reinterpret_cast<const ib::GemmJob*>(jobs.data_ptr()),
                              static_cast<int>(jobs.numel() / static_cast<int64_t>(sizeof(ib::GemmJob))),
                              static_cast<int>(max_rows), static_cast<int>(max_count), acts_of(values, scales, block),
                              out.data_ptr<float>(), static_cast<int>(out.size(1)), at::cuda::getCurrentCUDAStream()),
             "gemm");
}

void gemm_staged(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, int64_t max_count,
                 const torch::Tensor& values, const torch::Tensor& scales, int64_t block, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(values.device());
    check_cuda(table, "table", torch::kUInt8);
    check_cuda(jobs, "jobs", torch::kUInt8);
    check_cuda(out, "out", torch::kFloat32);
    TORCH_CHECK(jobs.numel() % static_cast<int64_t>(sizeof(ib::GemmJob)) == 0, "jobs must hold GemmJob records");
    check_ok(ib::enqueue_gemm_staged(reinterpret_cast<const ib::Matrix*>(table.data_ptr()),
                                     reinterpret_cast<const ib::GemmJob*>(jobs.data_ptr()),
                                     static_cast<int>(jobs.numel() / static_cast<int64_t>(sizeof(ib::GemmJob))),
                                     static_cast<int>(max_rows), static_cast<int>(max_count),
                                     acts_of(values, scales, block), out.data_ptr<float>(),
                                     static_cast<int>(out.size(1)), at::cuda::getCurrentCUDAStream()),
             "gemm_staged");
}

void quantize_e2m1x2(const torch::Tensor& x, double global, torch::Tensor& values, torch::Tensor& scales) {
    const c10::cuda::CUDAGuard guard(x.device());
    check_cuda(x, "x", torch::kFloat32);
    check_ok(ib::enqueue_quantize_e2m1x2(x.data_ptr<float>(), static_cast<int>(x.size(1)), static_cast<int>(x.size(0)),
                                         static_cast<int>(x.size(1)), static_cast<float>(global),
                                         acts_of(values, scales, 16), at::cuda::getCurrentCUDAStream()), "quantize_e2m1x2");
}

void gemm_native(const torch::Tensor& table, const torch::Tensor& jobs, int64_t max_rows, int64_t max_count,
                 const torch::Tensor& codes, const torch::Tensor& scales, double global, bool int4,
                 torch::Tensor& out, const std::optional<torch::Tensor>& col_scale, int64_t tiled,
                 const std::optional<torch::Tensor>& host_matrices) {
    const c10::cuda::CUDAGuard guard(codes.device());
    check_cuda(table, "table", torch::kUInt8);
    check_cuda(jobs, "jobs", torch::kUInt8);
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dim() == 2 &&
                (out.scalar_type() == torch::kFloat32 || out.scalar_type() == torch::kBFloat16),
                "out must be a contiguous [rows, out_ld] FP32 or BF16 CUDA tensor");
    check_cuda(codes, "codes", torch::kUInt8);
    check_cuda(scales, "scales", torch::kUInt8);
    TORCH_CHECK(codes.dim() == 2 && codes.size(1) % 8 == 0, "codes [rows, cols / 2]");
    const int64_t blocks = codes.size(1) / 8;
    if(tiled & ib::kTiledActs) {
        TORCH_CHECK(blocks % 4 == 0 && scales.numel() >= (codes.size(0) + 127) / 128 * 128 * blocks,
                    "tiled scales: [ceil(rows / 128) * 128] x [cols / 16] bytes in cuBLASLt's 128 x 4 layout");
    } else {
        TORCH_CHECK(scales.dim() == 2 && codes.size(0) == scales.size(0) && scales.size(1) == blocks,
                    "codes [rows, cols / 2] and scales [rows, cols / 16]");
    }
    TORCH_CHECK(jobs.numel() % static_cast<int64_t>(sizeof(ib::GemmJob)) == 0, "jobs must hold GemmJob records");
    ib::ActsFp4 acts;
    acts.codes = codes.data_ptr<std::uint8_t>();
    acts.scales = scales.data_ptr<std::uint8_t>();
    acts.ld = static_cast<int>(codes.size(1));
    acts.lds = static_cast<int>(blocks);
    acts.global = static_cast<float>(global);
    acts.tiled = static_cast<int>(tiled);
    acts.rows = static_cast<int>(codes.size(0));
    // The table's matrices as the host knows them (matrix_table's record: codes pointer, rows, cols), for tensor maps.
    std::vector<ib::HostMatrix> host;
    if(host_matrices.has_value()) {
        const torch::Tensor h = host_matrices->to(torch::kCPU, torch::kLong).contiguous();
        TORCH_CHECK(h.dim() == 2 && h.size(1) == 3, "host_matrices [matrices, 3]: codes pointer, rows, cols");
        const int64_t* v = h.data_ptr<int64_t>();
        for(int64_t i = 0; i < h.size(0); ++i)
            host.push_back({reinterpret_cast<const void*>(v[3 * i]), static_cast<int>(v[3 * i + 1]),
                            static_cast<int>(v[3 * i + 2])});
    }
    ib::NativeOut o;
    o.data = out.data_ptr();
    o.ld = static_cast<int>(out.size(1));
    o.bf16 = out.scalar_type() == torch::kBFloat16;
    o.rows = static_cast<int>(out.size(0));
    if(col_scale.has_value()) {
        check_cuda(*col_scale, "col_scale", torch::kFloat32);
        TORCH_CHECK(col_scale->numel() >= max_rows, "col_scale holds a factor for every output column");
        o.col = col_scale->data_ptr<float>();
    }
    const int job_count = static_cast<int>(jobs.numel() / static_cast<int64_t>(sizeof(ib::GemmJob)));
    const std::size_t bytes = ib::gemm_native_workspace(job_count, static_cast<int>(max_rows),
                                                        static_cast<int>(max_count), acts, int4);
    torch::Tensor workspace;
    if(bytes) workspace = torch::empty({static_cast<int64_t>(bytes)}, codes.options());
    check_ok(ib::enqueue_gemm_native(reinterpret_cast<const ib::Matrix*>(table.data_ptr()),
                                     reinterpret_cast<const ib::GemmJob*>(jobs.data_ptr()), job_count,
                                     static_cast<int>(max_rows), static_cast<int>(max_count), acts, int4, o,
                                     bytes ? workspace.data_ptr() : nullptr, bytes, at::cuda::getCurrentCUDAStream(),
                                     host.empty() ? nullptr : host.data(), static_cast<int>(host.size())),
             "gemm_native");
}

void quantize_nvfp4(const torch::Tensor& x, double global, torch::Tensor& codes, torch::Tensor& scales,
                    int64_t tiled) {
    const c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 &&
                (x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kBFloat16),
                "x must be a contiguous [rows, cols] FP32 or BF16 CUDA tensor");
    check_cuda(codes, "codes", torch::kUInt8);
    check_cuda(scales, "scales", torch::kUInt8);
    TORCH_CHECK(x.size(1) % 16 == 0 && codes.dim() == 2 && codes.size(0) == x.size(0) && codes.size(1) * 2 == x.size(1),
                "codes [rows, cols / 2]");
    const int64_t blocks = x.size(1) / 16;
    if(tiled & ib::kTiledActs) {
        TORCH_CHECK(blocks % 4 == 0 && scales.numel() >= (x.size(0) + 127) / 128 * 128 * blocks,
                    "tiled scales: [ceil(rows / 128) * 128] x [cols / 16] bytes");
    } else {
        TORCH_CHECK(scales.dim() == 2 && scales.size(0) == x.size(0) && scales.size(1) == blocks,
                    "scales [rows, cols / 16]");
    }
    ib::ActsFp4 acts;
    acts.codes = codes.data_ptr<std::uint8_t>();
    acts.scales = scales.data_ptr<std::uint8_t>();
    acts.ld = static_cast<int>(codes.size(1));
    acts.lds = static_cast<int>(blocks);
    acts.global = static_cast<float>(global);
    acts.tiled = static_cast<int>(tiled & ib::kTiledActs);
    check_ok(ib::enqueue_quantize_nvfp4(x.data_ptr(), x.scalar_type() == torch::kBFloat16, static_cast<int>(x.size(1)),
                                        static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
                                        static_cast<float>(global), acts, at::cuda::getCurrentCUDAStream()),
             "quantize_nvfp4");
}

void quantize_int8(const torch::Tensor& x, int64_t block, torch::Tensor& values, torch::Tensor& scales) {
    const c10::cuda::CUDAGuard guard(x.device());
    check_cuda(x, "x", torch::kFloat32);
    check_ok(ib::enqueue_quantize_int8(x.data_ptr<float>(), static_cast<int>(x.size(1)), static_cast<int>(x.size(0)),
                                       static_cast<int>(x.size(1)), acts_of(values, scales, block),
                                       at::cuda::getCurrentCUDAStream()), "quantize_int8");
}

void swiglu(const torch::Tensor& gate_up, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(gate_up.device());
    check_cuda(gate_up, "gate_up", torch::kFloat32);
    check_cuda(out, "out", torch::kFloat32);
    TORCH_CHECK(gate_up.size(1) == 2 * out.size(1) && gate_up.size(0) == out.size(0), "gate_up must be [rows, 2 * inter]");
    check_ok(ib::enqueue_swiglu(gate_up.data_ptr<float>(), static_cast<int>(out.size(1)), static_cast<int>(out.size(0)),
                                out.data_ptr<float>(), at::cuda::getCurrentCUDAStream()), "swiglu");
}
