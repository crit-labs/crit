// {name}: y[columns][N] = x[columns][K] . w[N][K] in {format}, FP32 output. A correct, simple starting point (one warp
// per output row, every column in that warp, FP32 accumulation in a fixed order): replace it with the kernel you are
// writing. Each column's sum runs the same instruction sequence whatever the column count, so a column computed alone
// equals the same column in a multi-column call.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "x and w must be contiguous");
    const int columns = x.size(0), k = x.size(1), n = w.size(0);
    TORCH_CHECK(w.size(1) == k && columns <= kMaxColumns, "x [columns <= 16, K], w [N, K]");
    auto y = torch::empty({columns, n}, x.options().dtype(torch::kFloat32));
    const int warps = 8;
    matmul_kernel<T><<<(n + warps - 1) / warps, warps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const T*>(x.data_ptr()), x_scale, reinterpret_cast<const T*>(w.data_ptr()), w_scale,
        y.data_ptr<float>(), columns, n, k);
    return y;
}

}  // namespace

torch::Tensor run_bf16(torch::Tensor x, torch::Tensor w) { return launch<__nv_bfloat16>(x, nullptr, w, nullptr); }

torch::Tensor run_fp8(torch::Tensor x, torch::Tensor x_scale, torch::Tensor w, torch::Tensor w_scale) {
    return launch<__nv_fp8_e4m3>(x, x_scale.contiguous().data_ptr<float>(), w, w_scale.contiguous().data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run_{format}, "{name}");
}
