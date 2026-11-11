// The kernel workbench's starter matmul: y[columns][N] = x[columns][K] . w[N][K] in BF16 or FP8 (E4M3 with a scale per
// row of w and per column of x), FP32 output. A correct, simple starting point (one warp per output row, every column
// in that warp, FP32 accumulation in a fixed order) that `crit kernels new --starter --from` copies into a workbench
// folder: the engine replaces it with the kernel it is writing. Each column's sum runs the same instruction sequence
// whatever the column count, so a column computed alone equals the same column in a multi-column call.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/all.h>

namespace {

constexpr int kMaxColumns = 16;

__device__ __forceinline__ float load(const __nv_bfloat16* p, std::int64_t i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ float load(const __nv_fp8_e4m3* p, std::int64_t i) { return float(p[i]); }

template <class T>
__global__ void matmul_kernel(const T* x, const float* x_scale, const T* w, const float* w_scale, float* y, int columns,
                              int n, int k) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if(row >= n) return;
    float acc[kMaxColumns] = {};
    for(int i = lane; i < k; i += 32) {
        const float wi = load(w, std::int64_t(row) * k + i);
        for(int c = 0; c < columns; ++c) acc[c] += wi * load(x, std::int64_t(c) * k + i);
    }
    for(int c = 0; c < columns; ++c) {
        float v = acc[c];
        for(int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if(lane == 0) y[std::int64_t(c) * n + row] = w_scale ? v * w_scale[row] * x_scale[c] : v;
    }
}

template <class T>
torch::Tensor launch(const torch::Tensor& x, const float* x_scale, const torch::Tensor& w, const float* w_scale) {
    TORCH_CHECK(x.is_cuda() && w.device() == x.device(), "x and w are CUDA tensors on one device");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "x and w must be contiguous");
    const int columns = x.size(0), k = x.size(1), n = w.size(0);
    TORCH_CHECK(w.size(1) == k && columns >= 1 && columns <= kMaxColumns, "x [columns <= 16, K], w [N, K]");
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty({columns, n}, x.options().dtype(torch::kFloat32));
    const int warps = 8;
    matmul_kernel<T><<<(n + warps - 1) / warps, warps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const T*>(x.data_ptr()), x_scale, reinterpret_cast<const T*>(w.data_ptr()), w_scale,
        y.data_ptr<float>(), columns, n, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

}  // namespace

torch::Tensor run_bf16(const torch::Tensor& x, const torch::Tensor& w) {
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16, "run_bf16: BF16 x and w");
    return launch<__nv_bfloat16>(x, nullptr, w, nullptr);
}

torch::Tensor run_fp8(const torch::Tensor& x, const torch::Tensor& x_scale, const torch::Tensor& w,
                      const torch::Tensor& w_scale) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat8_e4m3fn && w.scalar_type() == torch::kFloat8_e4m3fn,
                "run_fp8: E4M3 x and w");
    TORCH_CHECK(x_scale.scalar_type() == torch::kFloat32 && w_scale.scalar_type() == torch::kFloat32 &&
                    x_scale.is_contiguous() && w_scale.is_contiguous() && x_scale.numel() == x.size(0) &&
                    w_scale.numel() == w.size(0),
                "run_fp8: FP32 x_scale [columns] and w_scale [N]");
    return launch<__nv_fp8_e4m3>(x, x_scale.data_ptr<float>(), w, w_scale.data_ptr<float>());
}
