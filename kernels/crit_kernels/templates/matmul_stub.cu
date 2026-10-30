// {name}: y[columns][N] = x[columns][K] . w[N][K] in {format}, FP32 output. Write the kernel here, or start from an
// existing one (kernels/SOURCES.md); kernel.py's candidate() calls run(). `crit eval workbench check` compares it with
// the FP64 reference, bit for bit across repeats and column counts; `crit eval workbench bench` times it against the
// floor and the yardstick.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor run_bf16(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(false, "{name}: write the kernel (x [columns, K] bf16, w [N, K] bf16 -> y [columns, N] float32)");
    return {};
}

torch::Tensor run_fp8(torch::Tensor x, torch::Tensor x_scale, torch::Tensor w, torch::Tensor w_scale) {
    TORCH_CHECK(false, "{name}: write the kernel (x [columns, K] e4m3 with x_scale [columns, 1], w [N, K] e4m3 with "
                       "w_scale [N, 1] -> y [columns, N] float32)");
    return {};
}

torch::Tensor run_nvfp4(torch::Tensor x, torch::Tensor w, torch::Tensor w_scale, torch::Tensor w_global) {
    TORCH_CHECK(false, "{name}: write the kernel (x [columns, K] bf16; w [N, K / 2] uint8 E2M1 pairs, low nibble first, "
                       "with w_scale [N, K / 16] e4m3 and the FP32 divisor w_global [1]: w = e2m1 * w_scale / w_global "
                       "-> y [columns, N] float32)");
    return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run_{format}, "{name}");
}
