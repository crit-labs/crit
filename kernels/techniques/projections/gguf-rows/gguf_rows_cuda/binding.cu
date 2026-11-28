// Torch entry points over gguf_rows.h: checks and the current stream.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <algorithm>
#include <optional>
#include <vector>

#include "gguf_rows.h"
#include "crit_kcore/gguf.cuh"

namespace gr = crit::techniques::gguf_rows;
namespace gg = crit::kcore::gguf;

// The multiprocessors the untuned split rule counts: a definition (an RTX 5090's 170 SMs, where the rule was measured),
// never this device's, so a row's sums take the same order on every GPU.
constexpr int kOrderMultiprocessors = 170;

namespace {
int kind_of(const torch::Tensor& t, const char* what) {
    if (t.scalar_type() == torch::kFloat32) return 0;
    if (t.scalar_type() == torch::kBFloat16) return 1;
    TORCH_CHECK(t.scalar_type() == torch::kFloat16, what, " is FP32, BF16 or FP16");
    return 2;
}

// blocks: uint8 [N][row_bytes], contiguous, or [experts][N][row_bytes] with contiguous rows and any stride between
// experts (a stack read in place where each expert lies in a slot of its own, beside other bytes), on a CUDA device.
gr::Matrix matrix(const torch::Tensor& blocks, int64_t type, int64_t k) {
    TORCH_CHECK(gg::supported(static_cast<int>(type)), "GGUF type ", type, " is not one of Q4_0 (2), Q4_1 (3), "
                "Q5_0 (6), Q5_1 (7), Q8_0 (8), Q2_K (10), Q3_K (11), Q4_K (12), Q5_K (13), Q6_K (14), IQ2_XXS (16), IQ2_XS (17), "
                "IQ3_XXS (18), IQ1_S (19), IQ4_NL (20), IQ3_S (21), IQ2_S (22), IQ4_XS (23), IQ1_M (29), "
                "TQ1_0 (34), TQ2_0 (35), MXFP4 (39), Q1_0 (41), Q2_0 (42)");
    TORCH_CHECK(blocks.is_cuda() && blocks.scalar_type() == torch::kUInt8 && (blocks.dim() == 2 || blocks.dim() == 3),
                "blocks are uint8 [N][row bytes] or [experts][N][row bytes] on a CUDA device");
    const bool rows = blocks.stride(-1) == 1 && blocks.stride(-2) == blocks.size(-1);
    TORCH_CHECK(blocks.dim() == 2 ? blocks.is_contiguous() : rows && blocks.stride(0) >= blocks.size(1) * blocks.size(2),
                "blocks are contiguous [N][row bytes], or [experts][N][row bytes] with contiguous rows and experts "
                "that do not overlap");
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
    w.expert_stride = blocks.dim() == 3 ? blocks.stride(0) : int64_t(w.N) * w.row_bytes;
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
    // the order's multiprocessors, never this device's: a row's bits follow the parts
    return gr::splits(w, kOrderMultiprocessors);
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

// The grouped prompt GEMM over a stack of experts (blocks [experts][N][row bytes], experts at any stride): the rows
// of xq (by sub-block, [K / 32][M][32]) in groups of counts[g] rows in order, group g's rows times expert experts[g].
// Groups are staged a batch at a time within the staging budget, one launch each for the staging and the GEMM.
void gemm_experts(const torch::Tensor& blocks, int64_t type, int64_t k, const std::vector<int64_t>& experts,
                  const std::vector<int64_t>& counts, const torch::Tensor& xq, const torch::Tensor& xd,
                  const torch::Tensor& xs, int64_t parts, torch::Tensor& y) {
    const gr::Matrix w = matrix(blocks, type, k);
    TORCH_CHECK(blocks.dim() == 3, "gemm_experts: a stack of experts [experts][N][row bytes]");
    TORCH_CHECK(experts.size() == counts.size() && !experts.empty(), "gemm_experts: one count an expert");
    TORCH_CHECK(xq.dim() == 3 && xq.size(0) == k / 32 && xq.size(2) == 32, "gemm_experts: xq is the quantized rows by "
                "sub-block [K / 32][M][32] (quantize's by_subblock)");
    const int64_t m = xq.size(1);
    int64_t total = 0;
    for (size_t g = 0; g < counts.size(); ++g) {
        TORCH_CHECK(counts[g] >= 1, "gemm_experts: every group has rows");
        TORCH_CHECK(experts[g] >= 0 && experts[g] < blocks.size(0), "gemm_experts: expert ", experts[g], " of ",
                    blocks.size(0));
        total += counts[g];
    }
    TORCH_CHECK(total == m, "gemm_experts: the groups' ", total, " rows are not xq's ", m);
    TORCH_CHECK(xq.is_cuda() && xq.is_contiguous() && xq.device() == blocks.device() && xd.is_contiguous() &&
                    xd.numel() == m * (k / 32) && xs.is_contiguous() && xs.numel() == m * (k / 32),
                "gemm_experts: xq [K / 32][M][32], xd and xs [K / 32][M], contiguous on the stack's device");
    TORCH_CHECK(y.is_cuda() && y.dim() == 2 && y.stride(1) == 1 && y.size(0) == m && y.size(1) == w.N,
                "gemm_experts: y is [M][N] with contiguous rows");
    const c10::cuda::CUDAGuard guard(blocks.device());
    const int64_t J = k / 32, planes = w.type == gg::Q2_K ? 3 : 2;
    const int64_t group_codes = J * w.N * 32, group_scales = planes * J * w.N;
    const int64_t per_group = group_codes + 4 * group_scales;
    const int64_t batch = std::max<int64_t>(1, std::min<int64_t>(65535, kStageBytes / per_group));
    const int64_t key_m = std::max<int64_t>(1, m / int64_t(counts.size()));   // a group's rows, on average
    const int tile = gr::gemm_tile_rows(w.N, w.K, static_cast<int>(key_m));
    TORCH_CHECK(tile > 0, "gemm_experts: no gemm configuration for this matrix");
    const auto opts = blocks.options();
    const int64_t groups_here = std::min<int64_t>(batch, int64_t(counts.size()));
    torch::Tensor codes = torch::empty({groups_here * group_codes}, opts.dtype(torch::kInt8));
    torch::Tensor scales = torch::empty({groups_here * group_scales}, opts.dtype(torch::kFloat32));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int kind = kind_of(y, "y");
    int64_t start = 0;
    for (size_t g0 = 0; g0 < counts.size(); g0 += size_t(batch)) {
        const size_t g1 = std::min(counts.size(), g0 + size_t(batch));
        // the batch's experts, tiles (group, first row) and ends, in one host tensor copied once
        std::vector<int32_t> table;
        for (size_t g = g0; g < g1; ++g) table.push_back(static_cast<int32_t>(experts[g]));
        const size_t ends_at = table.size();
        int64_t row = start;
        for (size_t g = g0; g < g1; ++g) {
            row += counts[g];
            table.push_back(static_cast<int32_t>(row));
        }
        const size_t tiles_at = table.size();
        row = start;
        for (size_t g = g0; g < g1; ++g) {
            for (int64_t t = 0; t < counts[g]; t += tile) {
                table.push_back(static_cast<int32_t>(g - g0));
                table.push_back(static_cast<int32_t>(row + t));
            }
            row += counts[g];
        }
        const int tile_count = static_cast<int>((table.size() - tiles_at) / 2);
        torch::Tensor device = torch::from_blob(table.data(), {int64_t(table.size())}, torch::kInt32)
                                   .to(blocks.device(), /*non_blocking=*/false);
        const int32_t* base = device.data_ptr<int32_t>();
        cudaError_t e = gr::enqueue_stage_experts(w, base, static_cast<int>(g1 - g0), codes.data_ptr<int8_t>(),
                                                  scales.data_ptr<float>(), group_codes, group_scales, stream);
        TORCH_CHECK(e == cudaSuccess, "gemm_experts: staging: ", cudaGetErrorString(e));
        gr::Staged staged;
        staged.codes = codes.data_ptr<int8_t>();
        staged.scales = scales.data_ptr<float>();
        staged.rows = w.N;
        staged.K = w.K;
        staged.type = w.type;
        e = gr::enqueue_gemm_grouped(staged, group_codes, group_scales, base + tiles_at, tile_count, base + ends_at,
                                     w.N, static_cast<int>(key_m), xq.data_ptr<int8_t>(), xd.data_ptr<float>(),
                                     xs.data_ptr<int32_t>(), static_cast<int>(m), static_cast<int>(parts), y.data_ptr(),
                                     kind, y.stride(0), stream);
        TORCH_CHECK(e == cudaSuccess, "gemm_experts: ", cudaGetErrorString(e));
        start = row;
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
