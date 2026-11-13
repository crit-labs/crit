// Torch entry points over exl3_rows.h: checks and the current stream.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <optional>

#include "exl3_rows.h"

namespace er = crit::techniques::exl3_rows;

namespace {
int kind_of(const torch::Tensor& t, const char* what) {
    if (t.scalar_type() == torch::kFloat32) return 0;
    if (t.scalar_type() == torch::kBFloat16) return 1;
    TORCH_CHECK(t.scalar_type() == torch::kFloat16, what, " is FP32, BF16 or FP16");
    return 2;
}

er::Matrix matrix(const torch::Tensor& trellis, int64_t codebook) {
    TORCH_CHECK(trellis.is_cuda() && trellis.is_contiguous() && trellis.dim() == 3 &&
                    (trellis.scalar_type() == torch::kInt16 || trellis.scalar_type() == torch::kUInt16),
                "trellis is contiguous [in / 16][out / 16][16 K] 16-bit words on a CUDA device");
    const int64_t width = trellis.size(2);
    TORCH_CHECK(width % 16 == 0 || width % 16 == 8, "a tile holds 16 K words, K whole or a half");
    er::Matrix w;
    w.trellis = static_cast<const std::uint16_t*>(trellis.data_ptr());
    w.in = static_cast<int>(16 * trellis.size(0));
    w.out = static_cast<int>(16 * trellis.size(1));
    w.bits = static_cast<int>(width / 16);
    w.half_bit = width % 16 == 8;
    w.codebook = static_cast<int>(codebook);
    return w;
}

const void* signs(const std::optional<torch::Tensor>& s, int64_t size, const torch::Tensor& like, const char* what) {
    if (!s || !s->defined()) return nullptr;
    TORCH_CHECK(s->is_cuda() && s->is_contiguous() && s->scalar_type() == torch::kFloat16 && s->numel() == size &&
                    s->device() == like.device(),
                what, " is FP16 [", size, "] on the call's device");
    return s->data_ptr();
}
}  // namespace

void rotate(const torch::Tensor& rows, const std::optional<torch::Tensor>& pre,
            const std::optional<torch::Tensor>& post, torch::Tensor& out) {
    // rows [M][L], or FP32 [parts][M][L] contiguous: the parts added in order first.
    const bool parted = rows.dim() == 3;
    TORCH_CHECK(rows.is_cuda() && (rows.dim() == 2 || (parted && rows.is_contiguous() &&
                                                       rows.scalar_type() == torch::kFloat32)) &&
                    rows.stride(-1) == 1 && out.is_cuda() && out.dim() == 2 && out.stride(1) == 1 &&
                    out.size(0) == rows.size(-2) && out.size(1) == rows.size(-1) && out.device() == rows.device(),
                "rotate: rows ([M][L], or FP32 [parts][M][L]) and out [M][L] with contiguous rows on one CUDA device");
    TORCH_CHECK(rows.size(-1) % er::kBlock == 0, "rotate: L is a multiple of ", er::kBlock);
    const c10::cuda::CUDAGuard guard(rows.device());
    const int64_t length = rows.size(-1);
    const cudaError_t e = er::enqueue_rotate(rows.data_ptr(), kind_of(rows, "rows"), rows.stride(-2),
                                             static_cast<int>(rows.size(-2)), static_cast<int>(length),
                                             signs(pre, length, rows, "pre"), signs(post, length, rows, "post"),
                                             out.data_ptr(), kind_of(out, "out"), out.stride(0),
                                             at::cuda::getCurrentCUDAStream(), parted ? static_cast<int>(rows.size(0)) : 1,
                                             parted ? rows.stride(0) : 0);
    TORCH_CHECK(e == cudaSuccess, "rotate: ", cudaGetErrorString(e));
}

int64_t splits(const torch::Tensor& trellis, int64_t codebook, int64_t rows) {
    const er::Matrix w = matrix(trellis, codebook);
    const c10::cuda::CUDAGuard guard(trellis.device());
    int count = 0;
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, trellis.get_device());
    return er::splits(w, static_cast<int>(rows), count);
}

void tiles(const torch::Tensor& trellis, int64_t codebook, const torch::Tensor& xr, torch::Tensor& yr) {
    const er::Matrix w = matrix(trellis, codebook);
    TORCH_CHECK(xr.is_cuda() && xr.is_contiguous() && xr.scalar_type() == torch::kFloat16 && xr.dim() == 2 &&
                    xr.size(1) == w.in && xr.size(0) >= 1 && xr.size(0) <= er::kMaxRows && xr.device() == trellis.device(),
                "tiles: xr is contiguous FP16 [1..", er::kMaxRows, "][in] (the rotated activations)");
    TORCH_CHECK(yr.is_cuda() && yr.is_contiguous() && yr.scalar_type() == torch::kFloat32 && yr.dim() == 3 &&
                    yr.size(1) == xr.size(0) && yr.size(2) == w.out && yr.device() == trellis.device(),
                "tiles: yr is contiguous FP32 [splits][rows][out]");
    const c10::cuda::CUDAGuard guard(xr.device());
    const cudaError_t e = er::enqueue_tiles(w, xr.data_ptr(), static_cast<int>(xr.size(0)), yr.data_ptr<float>(),
                                            static_cast<int>(yr.size(0)), at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(e == cudaSuccess, "tiles: ", cudaGetErrorString(e), " (in and out multiples of 128, K 1 to 8, a half "
                "K only with codebook 2 up to 3.5)");
}

void decode(const torch::Tensor& trellis, int64_t codebook, torch::Tensor& q) {
    const er::Matrix w = matrix(trellis, codebook);
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kFloat16 && q.dim() == 2 &&
                    q.size(0) == w.in && q.size(1) == w.out && q.device() == trellis.device(),
                "decode: q is contiguous FP16 [in][out]");
    const c10::cuda::CUDAGuard guard(q.device());
    const cudaError_t e = er::enqueue_decode(w, q.data_ptr(), at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(e == cudaSuccess, "decode: ", cudaGetErrorString(e));
}
