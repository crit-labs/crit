// Torch entry points over decode_gemv.h: shape, dtype and alignment checks, the current stream. Nothing allocates and
// nothing waits for the device, so a CUDA graph captures the calls.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "decode_gemv.h"

namespace nr = crit::kernels::decode_gemv;

namespace {
void check_ok(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }

bool aligned(const torch::Tensor& t, std::uintptr_t bytes) {
    return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % bytes == 0;
}
bool aligned16(const torch::Tensor& t) { return aligned(t, 16); }

void same_device(const torch::Tensor& x, std::initializer_list<const torch::Tensor*> others, const char* what) {
    for (const torch::Tensor* t : others)
        TORCH_CHECK(t->device() == x.device(), what, ": every tensor is on x's device (", x.device(), "), not ",
                    t->device());
}

void check_rows(const torch::Tensor& x, const char* what) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2, what, ": x is a 2-D CUDA tensor [rows][K]");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= nr::kMaxRows, what, ": 1 to ", nr::kMaxRows, " rows, not ",
                x.size(0));
    TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && aligned16(x), what,
                ": x's rows are contiguous and start on 16-byte boundaries");
}

void check_out(const torch::Tensor& out, const torch::Tensor& x, int64_t n, const char* what) {
    TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.size(0) == x.size(0) && out.size(1) == n && out.stride(1) == 1,
                what, ": out is [rows][N] with contiguous rows");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32 || out.scalar_type() == x.scalar_type(), what,
                ": out is FP32 or x's dtype");
}
}  // namespace

void gemv(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& scales,
          const torch::Tensor& inverse, bool blocked, torch::Tensor& out) {
    check_rows(x, "gemv");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat16, "gemv: x is BF16 or FP16");
    const int64_t n = codes.size(0), k = x.size(1);
    TORCH_CHECK(k % 32 == 0, "gemv: K is a multiple of 32, not ", k);
    TORCH_CHECK(codes.is_cuda() && codes.is_contiguous() && codes.scalar_type() == torch::kUInt8 && codes.dim() == 2 &&
                    codes.size(1) * 2 == k && aligned16(codes),
                "gemv: codes are contiguous uint8 [N][K / 2] on 16 bytes");
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.element_size() == 1 && aligned(scales, 2),
                "gemv: scales are contiguous E4M3 bytes on 2 bytes");
    if (blocked) {
        TORCH_CHECK(scales.numel() == ((n + 127) / 128) * 128 * ((k / 16 + 3) / 4) * 4,
                    "gemv: blocked scales are cuBLASLt's 128 x 4 tiles of [N][K / 16] (rows padded to 128, blocks to 4)");
    } else {
        TORCH_CHECK(scales.dim() == 2 && scales.size(0) == n && scales.size(1) * 16 == k,
                    "gemv: scales are E4M3 bytes [N][K / 16]");
    }
    TORCH_CHECK(inverse.is_cuda() && inverse.is_contiguous() && inverse.scalar_type() == torch::kFloat32 &&
                    inverse.numel() == n && aligned(inverse, 4),
                "gemv: inverse is contiguous FP32 [N]");
    check_out(out, x, n, "gemv");
    same_device(x, {&codes, &scales, &inverse, &out}, "gemv");
    const c10::cuda::CUDAGuard guard(x.device());
    check_ok(nr::enqueue_gemv(x.data_ptr(), static_cast<int>(x.stride(0)), x.scalar_type() == torch::kBFloat16,
                              codes.data_ptr<std::uint8_t>(), reinterpret_cast<const std::uint8_t*>(scales.data_ptr()),
                              blocked, inverse.data_ptr<float>(), out.data_ptr(), static_cast<int>(out.stride(0)),
                              out.scalar_type() == torch::kFloat32, static_cast<int>(x.size(0)), static_cast<int>(n),
                              static_cast<int>(k), at::cuda::getCurrentCUDAStream()),
             "gemv");
}

void gemv_fp8(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& scales, torch::Tensor& out) {
    check_rows(x, "gemv_fp8");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat16,
                "gemv_fp8: x is BF16 or FP16");
    const int64_t n = codes.size(0), k = x.size(1);
    TORCH_CHECK(codes.is_cuda() && codes.is_contiguous() && codes.element_size() == 1 && codes.dim() == 2 &&
                    codes.size(1) == k && k % 16 == 0 && aligned16(codes),
                "gemv_fp8: codes are contiguous E4M3 bytes [N][K] on 16 bytes, K a multiple of 16");
    const bool mx = scales.scalar_type() == torch::kUInt8;
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous(), "gemv_fp8: scales are a contiguous CUDA tensor");
    if (mx) {
        TORCH_CHECK(scales.dim() == 2 && scales.size(0) == n && k % 32 == 0 && scales.size(1) * 32 == k,
                    "gemv_fp8: MXFP8 scales are E8M0 bytes [N][K / 32], K a multiple of 32");
    } else {
        TORCH_CHECK(scales.scalar_type() == torch::kFloat32 && scales.numel() == n && aligned(scales, 4),
                    "gemv_fp8: row scales are FP32 [N] (or E8M0 bytes [N][K / 32] for MXFP8)");
    }
    check_out(out, x, n, "gemv_fp8");
    same_device(x, {&codes, &scales, &out}, "gemv_fp8");
    const c10::cuda::CUDAGuard guard(x.device());
    check_ok(nr::enqueue_gemv_fp8(x.data_ptr(), static_cast<int>(x.stride(0)), x.scalar_type() == torch::kBFloat16,
                                  reinterpret_cast<const std::uint8_t*>(codes.data_ptr()),
                                  mx ? nullptr : scales.data_ptr<float>(),
                                  mx ? scales.data_ptr<std::uint8_t>() : nullptr, out.data_ptr(),
                                  static_cast<int>(out.stride(0)), out.scalar_type() == torch::kFloat32,
                                  static_cast<int>(x.size(0)), static_cast<int>(n), static_cast<int>(k),
                                  at::cuda::getCurrentCUDAStream()),
             "gemv_fp8");
}

void gemv_bf16(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out) {
    check_rows(x, "gemv_bf16");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "gemv_bf16: x is BF16");
    const int64_t n = w.size(0), k = x.size(1);
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kBFloat16 && w.dim() == 2 &&
                    w.size(1) == k && k % 8 == 0 && aligned16(w),
                "gemv_bf16: w is contiguous BF16 [N][K] on 16 bytes, K a multiple of 8");
    check_out(out, x, n, "gemv_bf16");
    same_device(x, {&w, &out}, "gemv_bf16");
    const c10::cuda::CUDAGuard guard(x.device());
    check_ok(nr::enqueue_gemv_bf16(x.data_ptr(), static_cast<int>(x.stride(0)), w.data_ptr(), out.data_ptr(),
                                   static_cast<int>(out.stride(0)), out.scalar_type() == torch::kFloat32,
                                   static_cast<int>(x.size(0)), static_cast<int>(n), static_cast<int>(k),
                                   at::cuda::getCurrentCUDAStream()),
             "gemv_bf16");
}

void gemv_int4(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& scales,
               const torch::Tensor& zeros, int64_t group, torch::Tensor& out) {
    check_rows(x, "gemv_int4");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat16,
                "gemv_int4: x is BF16 or FP16");
    const int64_t n = codes.size(0), k = x.size(1);
    TORCH_CHECK(group >= 32 && group % 32 == 0 && k % group == 0,
                "gemv_int4: the group is a multiple of 32 dividing K (",
                group, ", K ", k, ")");
    TORCH_CHECK(codes.is_cuda() && codes.is_contiguous() && codes.scalar_type() == torch::kUInt8 && codes.dim() == 2 &&
                    codes.size(1) * 2 == k && aligned16(codes),
                "gemv_int4: codes are contiguous uint8 [N][K / 2] on 16 bytes");
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.scalar_type() == torch::kFloat16 &&
                    scales.dim() == 2 && scales.size(0) == n && scales.size(1) * group == k,
                "gemv_int4: scales are contiguous FP16 [N][K / group]");
    TORCH_CHECK(zeros.is_cuda() && zeros.is_contiguous() && zeros.scalar_type() == torch::kUInt8 && zeros.dim() == 2 &&
                    zeros.size(0) == n && zeros.size(1) * group == k,
                "gemv_int4: zeros are contiguous uint8 [N][K / group]");
    check_out(out, x, n, "gemv_int4");
    same_device(x, {&codes, &scales, &zeros, &out}, "gemv_int4");
    const c10::cuda::CUDAGuard guard(x.device());
    nr::Matrix m;
    m.codes = codes.data_ptr<std::uint8_t>();
    m.scales = reinterpret_cast<const std::uint8_t*>(scales.data_ptr());
    m.zeros = zeros.data_ptr<std::uint8_t>();
    m.group = static_cast<int>(group);
    check_ok(nr::enqueue_gemv_weights(nr::kInt4Zp, m, x.data_ptr(), static_cast<int>(x.stride(0)),
                                      x.scalar_type() == torch::kBFloat16, out.data_ptr(),
                                      static_cast<int>(out.stride(0)), out.scalar_type() == torch::kFloat32,
                                      static_cast<int>(x.size(0)), static_cast<int>(n), static_cast<int>(k),
                                      at::cuda::getCurrentCUDAStream()),
             "gemv_int4");
}

void repack_int4(int64_t kind, const torch::Tensor& qweight, const torch::Tensor& qzeros, const torch::Tensor& scales,
                 const std::optional<torch::Tensor>& perm, int64_t zero_offset, int64_t group, torch::Tensor& codes,
                 torch::Tensor& zeros, torch::Tensor& scales_out) {
    TORCH_CHECK(kind == nr::kRepackAwq || kind == nr::kRepackGptq, "repack_int4: kind 0 AWQ, 1 GPTQ");
    TORCH_CHECK(qweight.is_cuda() && qweight.is_contiguous() && qweight.scalar_type() == torch::kInt32 &&
                    qweight.dim() == 2, "repack_int4: qweight is contiguous int32");
    const int64_t n = kind == nr::kRepackAwq ? qweight.size(1) * 8 : qweight.size(1);
    const int64_t k = kind == nr::kRepackAwq ? qweight.size(0) : qweight.size(0) * 8;
    TORCH_CHECK(group >= 8 && k % group == 0 && n % 8 == 0, "repack_int4: K a multiple of the group, N of 8");
    const int64_t g = k / group;
    TORCH_CHECK(qzeros.is_cuda() && qzeros.is_contiguous() && qzeros.scalar_type() == torch::kInt32 &&
                    qzeros.dim() == 2 && qzeros.size(0) == g && qzeros.size(1) * 8 == n,
                "repack_int4: qzeros are contiguous int32 [K / group][N / 8]");
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.scalar_type() == torch::kFloat16 &&
                    scales.dim() == 2 && scales.size(0) == g && scales.size(1) == n,
                "repack_int4: scales are contiguous FP16 [K / group][N]");
    if (perm) {
        TORCH_CHECK(perm->is_cuda() && perm->is_contiguous() && perm->scalar_type() == torch::kInt32 &&
                        perm->numel() == k, "repack_int4: perm is contiguous int32 [K]");
    }
    TORCH_CHECK(codes.is_contiguous() && codes.scalar_type() == torch::kUInt8 && codes.numel() == n * k / 2 &&
                    zeros.is_contiguous() && zeros.scalar_type() == torch::kUInt8 && zeros.numel() == n * g &&
                    scales_out.is_contiguous() && scales_out.scalar_type() == torch::kFloat16 &&
                    scales_out.numel() == n * g,
                "repack_int4: codes uint8 [N][K / 2], zeros uint8 and scales FP16 [N][K / group]");
    same_device(qweight, {&qzeros, &scales, &codes, &zeros, &scales_out}, "repack_int4");
    const c10::cuda::CUDAGuard guard(qweight.device());
    check_ok(nr::enqueue_repack_int4(static_cast<int>(kind), reinterpret_cast<const std::uint32_t*>(qweight.data_ptr()),
                                     reinterpret_cast<const std::uint32_t*>(qzeros.data_ptr()),
                                     reinterpret_cast<const std::uint16_t*>(scales.data_ptr()),
                                     perm ? perm->data_ptr<int>() : nullptr, static_cast<int>(zero_offset),
                                     static_cast<int>(n), static_cast<int>(k), static_cast<int>(group),
                                     codes.data_ptr<std::uint8_t>(), zeros.data_ptr<std::uint8_t>(),
                                     reinterpret_cast<std::uint16_t*>(scales_out.data_ptr()),
                                     at::cuda::getCurrentCUDAStream()),
             "repack_int4");
}

int64_t hardware_decode(const torch::Tensor& device_of) {
    const c10::cuda::CUDAGuard guard(device_of.device());
    return nr::hardware_decode();
}
