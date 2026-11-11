// Torch entry points over mma_rows.h: shape, dtype, device and alignment checks, the current stream. Nothing
// allocates and nothing waits for the device, so a CUDA graph captures the calls.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "mma_rows.h"
#include "fp4_mma.h"

namespace mr = crit::kernels::mma_rows;

namespace {
void check_ok(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }

bool aligned(const torch::Tensor& t, std::uintptr_t bytes) {
  return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % bytes == 0;
}

void on_device(const torch::Tensor& x, const torch::Tensor& t, const char* what) {
  TORCH_CHECK(t.device() == x.device(), what, ": every tensor is on x's device (", x.device(), "), not ", t.device());
}

void check_x(const torch::Tensor& x, int64_t most, const char* what) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == torch::kBFloat16, what,
              ": x is a 2-D BF16 CUDA tensor [rows][K]");
  TORCH_CHECK(x.size(0) >= 1 && (most == 0 || x.size(0) <= most), what, ": 1 to ", most, " rows, not ", x.size(0));
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && aligned(x, 16), what,
              ": x's rows are contiguous and start on 16-byte boundaries");
}

int out_kind(const torch::Tensor& out) {
  if (out.scalar_type() == torch::kFloat32) return mr::kOutF32;
  if (out.scalar_type() == torch::kBFloat16) return mr::kOutBf16;
  TORCH_CHECK(out.scalar_type() == torch::kFloat16, "mm_rows: out is FP32, BF16 or FP16");
  return mr::kOutF16;
}
}  // namespace

void mm_rows(const torch::Tensor& x, const torch::Tensor& w, const std::optional<torch::Tensor>& scales,
             const std::optional<torch::Tensor>& row_scale, int64_t weights, int64_t acts, double input_scale,
             bool blocked, int64_t groups, int64_t warps, int64_t splits, const std::optional<torch::Tensor>& scratch,
             const std::optional<torch::Tensor>& partial, const std::optional<torch::Tensor>& counters,
             torch::Tensor& out, const std::optional<torch::Tensor>& zeros, int64_t group) {
  check_x(x, mr::kMaxRows, "mm_rows");
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  TORCH_CHECK(k % mr::kStep == 0, "mm_rows: K is a multiple of ", mr::kStep, ", not ", k);
  TORCH_CHECK(weights >= mr::kNvfp4 && weights <= mr::kInt4Zp,
              "mm_rows: weights 0 NVFP4, 1 FP8, 2 MXFP8, 3 BF16, 4 INT4 with a zero point");
  const bool valid = acts == mr::kActBf16 || (acts == mr::kActNvfp4 && weights == mr::kNvfp4) ||
                     (acts == mr::kActFp8Rows && weights == mr::kFp8) ||
                     (acts == mr::kActNvfp4Mma && weights == mr::kNvfp4 && blocked);
  TORCH_CHECK(valid, "mm_rows: activations 0 BF16 (any weights), 1 FP8 rows (FP8 weights), 2 NVFP4 (NVFP4 weights), "
                     "3 NVFP4 on the FP4 MMA (NVFP4 weights with blocked scales)");
  if (acts == mr::kActNvfp4Mma) {
    TORCH_CHECK(k % mr::kFp4Step == 0 && splits <= k / mr::kFp4Step,
                "mm_rows: the FP4 MMA takes K a multiple of ", mr::kFp4Step, " and at most K / ", mr::kFp4Step,
                " splits");
    TORCH_CHECK(mr::fp4_mma(at::cuda::getCurrentCUDAStream()), "mm_rows: this device or build has no FP4 MMA");
  }
  TORCH_CHECK(groups == 1 || groups == 2, "mm_rows: groups 1 or 2");
  TORCH_CHECK(warps >= 1 && warps <= mr::kMaxWarps, "mm_rows: 1 to ", mr::kMaxWarps, " warps");
  TORCH_CHECK(splits >= 1 && splits <= mr::kMaxSplits, "mm_rows: 1 to ", mr::kMaxSplits, " splits");
  const int64_t row_bytes =
      weights == mr::kNvfp4 || weights == mr::kInt4Zp ? k / 2 : weights == mr::kBf16 ? 2 * k : k;
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.dim() == 2 && w.size(1) * w.element_size() == row_bytes &&
                  aligned(w, 16) && n >= 1,
              "mm_rows: w is contiguous [N][", row_bytes, " bytes] on 16 bytes");
  on_device(x, w, "mm_rows");
  TORCH_CHECK(weights != mr::kBf16 || w.scalar_type() == torch::kBFloat16, "mm_rows: BF16 weights are BF16");
  mr::Problem p;
  p.x = x.data_ptr();
  p.ldx = x.stride(0);
  p.w = static_cast<const std::uint8_t*>(w.data_ptr());
  p.M = static_cast<int>(m);
  p.N = static_cast<int>(n);
  p.K = static_cast<int>(k);
  p.weights = static_cast<int>(weights);
  p.acts = static_cast<int>(acts);
  p.groups = static_cast<int>(groups);
  p.warps = static_cast<int>(warps);
  p.splits = static_cast<int>(splits);
  if (splits > 1) {
    TORCH_CHECK(partial.has_value() && counters.has_value(), "mm_rows: splits > 1 take partial and counters");
    const torch::Tensor& f = *partial;
    const torch::Tensor& c = *counters;
    on_device(x, f, "mm_rows");
    on_device(x, c, "mm_rows");
    TORCH_CHECK(f.is_contiguous() && f.scalar_type() == torch::kFloat32 &&
                    f.numel() >= (int64_t)mr::partial_elements((int)m, (int)n, (int)splits),
                "mm_rows: partial is FP32 with splits * rows * N elements or more");
    TORCH_CHECK(c.is_contiguous() && c.scalar_type() == torch::kInt32 &&
                    c.numel() >= mr::row_tiles((int)n, 1, (int)warps),
                "mm_rows: counters are int32 zeros, one a row tile of one group (ceil(N / (16 * warps))) or more");
    p.partial = f.data_ptr<float>();
    p.counters = c.data_ptr<int>();
  }
  if (acts != mr::kActBf16) {
    TORCH_CHECK(scratch.has_value(), "mm_rows: quantized activations take a scratch buffer");
    const torch::Tensor& s = *scratch;
    on_device(x, s, "mm_rows");
    const int64_t need = acts == mr::kActNvfp4Mma ? (int64_t)mr::fp4_scratch_bytes((int)m, (int)k)
                                                  : (int64_t)mr::scratch_bytes((int)m, (int)k);
    TORCH_CHECK(s.is_contiguous() && aligned(s, 16) && s.numel() * s.element_size() >= need,
                "mm_rows: scratch is contiguous on 16 bytes with ", need, " bytes or more (scratch_bytes)");
    p.scratch = static_cast<std::uint8_t*>(s.data_ptr());
  }
  if (weights == mr::kNvfp4 || weights == mr::kMxfp8) {
    TORCH_CHECK(scales.has_value(), "mm_rows: NVFP4 and MXFP8 weights take their block scales");
    const torch::Tensor& s = *scales;
    on_device(x, s, "mm_rows");
    TORCH_CHECK(s.is_contiguous() && s.element_size() == 1 && aligned(s, 2), "mm_rows: scales are contiguous bytes");
    if (weights == mr::kMxfp8) {
      TORCH_CHECK(!blocked && s.numel() == n * (k / 32), "mm_rows: MXFP8 scales are E8M0 bytes [N][K / 32]");
    } else if (blocked) {
      TORCH_CHECK(s.numel() == ((n + 127) / 128) * 128 * ((k / 16 + 3) / 4) * 4,
                  "mm_rows: blocked scales are cuBLASLt's 128 x 4 tiles of [N][K / 16]");
      p.ct = static_cast<int>((k / 16 + 3) / 4);
    } else {
      TORCH_CHECK(s.numel() == n * (k / 16), "mm_rows: NVFP4 scales are E4M3 bytes [N][K / 16]");
    }
    p.scales = static_cast<const std::uint8_t*>(s.data_ptr());
  }
  if (weights == mr::kInt4Zp) {
    TORCH_CHECK(scales.has_value() && zeros.has_value() && group >= 32 && group % 32 == 0 && k % group == 0,
                "mm_rows: INT4 weights take FP16 scales and uint8 zeros [N][K / group], the group a multiple of 32 ",
                "dividing K");
    const torch::Tensor& s = *scales;
    const torch::Tensor& z = *zeros;
    on_device(x, s, "mm_rows");
    on_device(x, z, "mm_rows");
    TORCH_CHECK(s.is_contiguous() && s.scalar_type() == torch::kFloat16 && s.numel() == n * (k / group) &&
                    z.is_contiguous() && z.scalar_type() == torch::kUInt8 && z.numel() == n * (k / group),
                "mm_rows: INT4 scales are FP16 and zeros uint8, [N][K / group] each");
    p.scales = static_cast<const std::uint8_t*>(s.data_ptr());
    p.zeros = z.data_ptr<std::uint8_t>();
    p.group = static_cast<int>(group);
  }
  if (weights == mr::kNvfp4 || weights == mr::kFp8) {
    TORCH_CHECK(row_scale.has_value(), "mm_rows: NVFP4 and FP8 weights take a row scale [N]");
    const torch::Tensor& r = *row_scale;
    on_device(x, r, "mm_rows");
    TORCH_CHECK(r.is_contiguous() && r.scalar_type() == torch::kFloat32 && r.numel() == n && aligned(r, 4),
                "mm_rows: row_scale is FP32 [N]");
    p.row_scale = r.data_ptr<float>();
  }
  if (acts == mr::kActNvfp4 || acts == mr::kActNvfp4Mma) {
    TORCH_CHECK(input_scale > 0, "mm_rows: NVFP4 activations take their input scale (> 0)");
    p.input_scale = static_cast<float>(input_scale);
    p.input_rcp = 1.0f / p.input_scale;
  }
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.size(0) == m && out.size(1) == n && out.stride(1) == 1,
              "mm_rows: out is [rows][N] with contiguous rows");
  on_device(x, out, "mm_rows");
  p.out = out.data_ptr();
  p.ldo = out.stride(0);
  p.out_kind = out_kind(out);
  const c10::cuda::CUDAGuard guard(x.device());
  check_ok(mr::enqueue(p, at::cuda::getCurrentCUDAStream()), "mm_rows");
}

void quantize_fp8_rows(const torch::Tensor& x, torch::Tensor& q, torch::Tensor& scale) {
  check_x(x, 0, "quantize_fp8_rows");
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(q.is_cuda() && q.dim() == 2 && q.size(0) == m && q.size(1) == k && q.stride(1) == 1 &&
                  q.element_size() == 1,
              "quantize_fp8_rows: q is E4M3 bytes [rows][K] with contiguous rows");
  TORCH_CHECK(scale.is_cuda() && scale.is_contiguous() && scale.scalar_type() == torch::kFloat32 && scale.numel() == m,
              "quantize_fp8_rows: scale is FP32 [rows]");
  on_device(x, q, "quantize_fp8_rows");
  on_device(x, scale, "quantize_fp8_rows");
  const c10::cuda::CUDAGuard guard(x.device());
  check_ok(mr::enqueue_quantize_fp8_rows(x.data_ptr(), x.stride(0), static_cast<int>(m), static_cast<int>(k),
                                         static_cast<std::uint8_t*>(q.data_ptr()), q.stride(0), scale.data_ptr<float>(),
                                         at::cuda::getCurrentCUDAStream()),
           "quantize_fp8_rows");
}

void quantize_nvfp4(const torch::Tensor& x, double input_scale, torch::Tensor& codes, torch::Tensor& scales) {
  check_x(x, 0, "quantize_nvfp4");
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(k % 16 == 0, "quantize_nvfp4: K is a multiple of 16, not ", k);
  TORCH_CHECK(input_scale > 0, "quantize_nvfp4: the input scale is > 0");
  TORCH_CHECK(codes.is_cuda() && codes.is_contiguous() && codes.element_size() == 1 && codes.numel() == m * (k / 2) &&
                  aligned(codes, 8),
              "quantize_nvfp4: codes are contiguous bytes [rows][K / 2] on 8 bytes");
  TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.element_size() == 1 &&
                  scales.numel() == ((m + 127) / 128) * 128 * ((k / 16 + 3) / 4) * 4,
              "quantize_nvfp4: scales are cuBLASLt's 128 x 4 tiles of [rows][K / 16] (rows padded to 128, blocks to 4)");
  on_device(x, codes, "quantize_nvfp4");
  on_device(x, scales, "quantize_nvfp4");
  const float g = static_cast<float>(input_scale);
  const c10::cuda::CUDAGuard guard(x.device());
  check_ok(mr::enqueue_quantize_nvfp4(x.data_ptr(), x.stride(0), static_cast<int>(m), static_cast<int>(k), g, 1.0f / g,
                                      static_cast<std::uint8_t*>(codes.data_ptr()),
                                      static_cast<std::uint8_t*>(scales.data_ptr()), at::cuda::getCurrentCUDAStream()),
           "quantize_nvfp4");
}

bool fp4_mma(const torch::Tensor& like) {
  TORCH_CHECK(like.is_cuda(), "fp4_mma: a CUDA tensor names the device");
  const c10::cuda::CUDAGuard guard(like.device());
  return mr::fp4_mma(at::cuda::getCurrentCUDAStream());
}

int64_t fp4_scratch_bytes(int64_t rows, int64_t k) { return (int64_t)mr::fp4_scratch_bytes((int)rows, (int)k); }

int64_t step_values(int64_t weights) {
  TORCH_CHECK(weights >= mr::kNvfp4 && weights <= mr::kInt4Zp, "step_values: weights 0 NVFP4 .. 4 INT4");
  return mr::step_values(static_cast<int>(weights));
}
