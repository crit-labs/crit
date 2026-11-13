// Torch entry points over gguf_rows.h: checks and the current stream.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <algorithm>
#include <optional>

#include "gguf_rows.h"
#include "crit_kcore/gguf.cuh"

namespace gr = crit::techniques::gguf_rows;
namespace gg = crit::kcore::gguf;

namespace {
int kind_of(const torch::Tensor& t, const char* what) {
    if (t.scalar_type() == torch::kFloat32) return 0;
    if (t.scalar_type() == torch::kBFloat16) return 1;
    TORCH_CHECK(t.scalar_type() == torch::kFloat16, what, " is FP32, BF16 or FP16");
    return 2;
}

// blocks: uint8 [N][row_bytes] or [experts][N][row_bytes], contiguous, on a CUDA device.
gr::Matrix matrix(const torch::Tensor& blocks, int64_t type, int64_t k) {
    TORCH_CHECK(gg::supported(static_cast<int>(type)), "GGUF type ", type, " is not one of Q4_0 (2), Q4_1 (3), "
                "Q5_0 (6), Q5_1 (7), Q8_0 (8), Q2_K (10), Q3_K (11), Q4_K (12), Q5_K (13), Q6_K (14), IQ2_XXS (16), IQ2_XS (17), "
                "IQ3_XXS (18), IQ1_S (19), IQ4_NL (20), IQ3_S (21), IQ2_S (22), IQ4_XS (23), IQ1_M (29), "
                "MXFP4 (39)");
    TORCH_CHECK(blocks.is_cuda() && blocks.is_contiguous() && blocks.scalar_type() == torch::kUInt8 &&
                    (blocks.dim() == 2 || blocks.dim() == 3),
                "blocks are contiguous uint8 [N][row bytes] or [experts][N][row bytes] on a CUDA device");
    const int values = gg::block_values(static_cast<int>(type));
    TORCH_CHECK(k > 0 && k % values == 0, "K is a multiple of the format's ", values, " values a block");
    gr::Matrix w;
    w.blocks = blocks.data_ptr<uint8_t>();
    w.type = static_cast<int>(type);
    w.K = static_cast<int>(k);
    w.N = static_cast<int>(blocks.size(-2));
    w.row_bytes = k / values * gg::block_bytes(w.type);
    TORCH_CHECK(blocks.size(-1) == w.row_bytes, "a row of K = ", k, " values holds ", w.row_bytes, " bytes, not ",
                blocks.size(-1));
    w.expert_stride = int64_t(w.N) * w.row_bytes;
    return w;
}
}  // namespace

void quantize(const torch::Tensor& x, torch::Tensor& xq, torch::Tensor& xd, torch::Tensor& xs, bool by_subblock) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.stride(1) == 1, "quantize: x is [M][K] with contiguous rows");
    const int64_t m = x.size(0), k = x.size(1);
    TORCH_CHECK(k % 32 == 0, "quantize: K is a multiple of 32");
    const bool shaped = by_subblock ? xq.dim() == 3 && xq.size(0) == k / 32 && xq.size(1) == m && xq.size(2) == 32
                                    : xq.dim() == 2 && xq.size(0) == m && xq.size(1) == k;
    TORCH_CHECK(xq.is_cuda() && xq.is_contiguous() && xq.scalar_type() == torch::kInt8 && shaped &&
                    xd.is_contiguous() && xd.scalar_type() == torch::kFloat32 && xd.numel() == m * (k / 32) &&
                    xs.is_contiguous() && xs.scalar_type() == torch::kInt32 && xs.numel() == m * (k / 32),
                "quantize: xq int8 [M][K] (by sub-block [K / 32][M][32]), xd FP32 and xs int32 [M][K / 32] (by "
                "sub-block [K / 32][M]), contiguous");
    const c10::cuda::CUDAGuard guard(x.device());
    const cudaError_t e = gr::enqueue_quantize(x.data_ptr(), kind_of(x, "x"), x.stride(0), static_cast<int>(m),
                                               static_cast<int>(k), xq.data_ptr<int8_t>(), xd.data_ptr<float>(),
                                               xs.data_ptr<int32_t>(), by_subblock, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(e == cudaSuccess, "quantize: ", cudaGetErrorString(e));
}

int64_t splits(const torch::Tensor& blocks, int64_t type, int64_t k) {
    const gr::Matrix w = matrix(blocks, type, k);
    const c10::cuda::CUDAGuard guard(blocks.device());
    int count = 0;
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, blocks.get_device());
    return gr::splits(w, count);
}

void gemv(const torch::Tensor& blocks, int64_t type, int64_t k, const torch::Tensor& xq, const torch::Tensor& xd,
          const torch::Tensor& xs, const std::optional<torch::Tensor>& ids, torch::Tensor& yp) {
    const gr::Matrix w = matrix(blocks, type, k);
    const int64_t m = xq.size(0);
    TORCH_CHECK(xq.is_cuda() && xq.is_contiguous() && xq.dim() == 2 && xq.size(1) == k && m >= 1 &&
                    m <= gr::kMaxRows && xq.device() == blocks.device(),
                "gemv: xq is the quantized rows [1..", gr::kMaxRows, "][K] on the matrix's device");
    TORCH_CHECK(yp.is_cuda() && yp.is_contiguous() && yp.scalar_type() == torch::kFloat32 && yp.dim() == 3 &&
                    yp.size(1) == m && yp.size(2) == w.N,
                "gemv: yp is contiguous FP32 [splits][M][N]");
    const int32_t* id = nullptr;
    if (ids && ids->defined()) {
        TORCH_CHECK(blocks.dim() == 3 && ids->is_cuda() && ids->is_contiguous() &&
                        ids->scalar_type() == torch::kInt32 && ids->numel() == m,
                    "gemv: ids are int32 [M] expert indices into blocks [experts][N][row bytes]");
        id = ids->data_ptr<int32_t>();
    } else {
        TORCH_CHECK(blocks.dim() == 2, "gemv: a stack of experts takes ids");
    }
    const c10::cuda::CUDAGuard guard(blocks.device());
    const cudaError_t e = gr::enqueue_gemv(w, xq.data_ptr<int8_t>(), xd.data_ptr<float>(), xs.data_ptr<int32_t>(),
                                           static_cast<int>(m), id, yp.data_ptr<float>(), static_cast<int>(yp.size(0)),
                                           at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(e == cudaSuccess, "gemv: ", cudaGetErrorString(e));
}

// The staging buffer's budget: rows of W decoded at a time so its codes and scales stay within it (a multiple of 64
// rows, every row of a smaller matrix).
constexpr int64_t kStageBytes = int64_t(128) << 20;

void gemm(const torch::Tensor& blocks, int64_t type, int64_t k, const torch::Tensor& xq, const torch::Tensor& xd,
          const torch::Tensor& xs, int64_t parts, torch::Tensor& y, int64_t chunk_rows) {
    const gr::Matrix w = matrix(blocks, type, k);
    TORCH_CHECK(blocks.dim() == 2, "gemm: one matrix [N][row bytes]");
    TORCH_CHECK(xq.dim() == 3 && xq.size(0) == k / 32 && xq.size(2) == 32, "gemm: xq is the quantized rows by "
                "sub-block [K / 32][M][32] (quantize's by_subblock)");
    const int64_t m = xq.size(1);
    TORCH_CHECK(xq.is_cuda() && xq.is_contiguous() && m >= 1 && xq.device() == blocks.device() && xd.is_contiguous() &&
                    xd.numel() == m * (k / 32) && xs.is_contiguous() && xs.numel() == m * (k / 32),
                "gemm: xq [K / 32][M][32], xd and xs [K / 32][M], contiguous on the matrix's device");
    TORCH_CHECK(y.is_cuda() && y.dim() == 2 && y.stride(1) == 1 && y.size(0) == m && y.size(1) == w.N,
                "gemm: y is [M][N] with contiguous rows");
    const c10::cuda::CUDAGuard guard(blocks.device());
    const int64_t J = k / 32, planes = w.type == gg::Q2_K ? 3 : 2;
    int64_t rows = chunk_rows > 0 ? chunk_rows : std::max<int64_t>(64, kStageBytes / (k + J * 4 * planes) / 64 * 64);
    rows = std::min<int64_t>(rows, w.N);
    const auto opts = blocks.options();
    torch::Tensor codes = torch::empty({J * rows * 32}, opts.dtype(torch::kInt8));
    torch::Tensor scales = torch::empty({planes * J * rows}, opts.dtype(torch::kFloat32));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int kind = kind_of(y, "y");
    for (int64_t n0 = 0; n0 < w.N; n0 += rows) {
        const int r = static_cast<int>(std::min<int64_t>(rows, w.N - n0));
        cudaError_t e = gr::enqueue_stage(w, static_cast<int>(n0), r, codes.data_ptr<int8_t>(), scales.data_ptr<float>(),
                                          stream);
        TORCH_CHECK(e == cudaSuccess, "gemm: staging: ", cudaGetErrorString(e));
        gr::Staged staged;
        staged.codes = codes.data_ptr<int8_t>();
        staged.scales = scales.data_ptr<float>();
        staged.rows = r;
        staged.K = w.K;
        staged.type = w.type;
        e = gr::enqueue_gemm(staged, w.N, xq.data_ptr<int8_t>(), xd.data_ptr<float>(), xs.data_ptr<int32_t>(),
                             static_cast<int>(m), static_cast<int>(parts),
                             static_cast<char*>(y.data_ptr()) + n0 * y.element_size(), kind, y.stride(0), stream);
        TORCH_CHECK(e == cudaSuccess, "gemm: ", cudaGetErrorString(e));
    }
}

void sum_parts(const torch::Tensor& yp, torch::Tensor& y) {
    TORCH_CHECK(yp.is_cuda() && yp.is_contiguous() && yp.scalar_type() == torch::kFloat32 && yp.dim() == 3 &&
                    y.is_cuda() && y.dim() == 2 && y.stride(1) == 1 && y.size(0) == yp.size(1) &&
                    y.size(1) == yp.size(2),
                "sum_parts: yp FP32 [parts][M][N], y [M][N] with contiguous rows");
    const c10::cuda::CUDAGuard guard(y.device());
    const cudaError_t e = gr::enqueue_sum(yp.data_ptr<float>(), static_cast<int>(yp.size(0)), static_cast<int>(yp.size(1)),
                                          static_cast<int>(yp.size(2)), y.data_ptr(), kind_of(y, "y"), y.stride(0),
                                          at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(e == cudaSuccess, "sum_parts: ", cudaGetErrorString(e));
}

void dequantize(const torch::Tensor& blocks, int64_t type, int64_t k, int64_t expert, torch::Tensor& out) {
    const gr::Matrix w = matrix(blocks, type, k);
    const int64_t experts = blocks.dim() == 3 ? blocks.size(0) : 1;
    TORCH_CHECK(expert >= 0 && expert < experts, "dequantize: expert ", expert, " of ", experts);
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dim() == 2 && out.size(0) == w.N && out.size(1) == k,
                "dequantize: out is contiguous [N][K]");
    const c10::cuda::CUDAGuard guard(blocks.device());
    const cudaError_t e = gr::enqueue_dequantize(w, static_cast<int>(expert), out.data_ptr(), kind_of(out, "out"),
                                                 at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(e == cudaSuccess, "dequantize: ", cudaGetErrorString(e));
}
