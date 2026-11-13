// Torch entry point of the prompt tile (mma_rows.h, enqueue_prompt): mm_rows' checks for any number of rows, the
// current stream. Nothing allocates and nothing waits for the device, so a CUDA graph captures a call.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "mma_rows.h"

namespace mr = crit::kernels::mma_rows;

namespace {
bool aligned_to(const torch::Tensor& t, std::uintptr_t bytes) {
  return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % bytes == 0;
}
void same_device(const torch::Tensor& x, const torch::Tensor& t) {
  TORCH_CHECK(t.device() == x.device(), "mm_prompt: every tensor is on x's device (", x.device(), "), not ",
              t.device());
}
}  // namespace

void mm_prompt(const torch::Tensor& x, const torch::Tensor& w, const std::optional<torch::Tensor>& scales,
               const std::optional<torch::Tensor>& row_scale, int64_t weights, int64_t acts, double input_scale,
               bool blocked, int64_t splits, const std::optional<torch::Tensor>& scratch, torch::Tensor& out,
               const std::optional<torch::Tensor>& zeros, int64_t group, int64_t glu) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == torch::kBFloat16 && x.size(0) >= 1,
              "mm_prompt: x is a 2-D BF16 CUDA tensor [rows][K], one row or more");
  TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max(), "mm_prompt: at most 2^31 - 1 rows");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && aligned_to(x, 16),
              "mm_prompt: x's rows are contiguous and start on 16-byte boundaries");
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  TORCH_CHECK(k % mr::kStep == 0 && k >= mr::kStep, "mm_prompt: K is a multiple of ", mr::kStep, ", not ", k);
  TORCH_CHECK(weights >= mr::kNvfp4 && weights <= mr::kInt4Zp,
              "mm_prompt: weights 0 NVFP4, 1 FP8, 2 MXFP8, 3 BF16, 4 INT4 with a zero point");
  const bool valid = acts == mr::kActBf16 || (acts == mr::kActNvfp4 && weights == mr::kNvfp4) ||
                     (acts == mr::kActFp8Rows && weights == mr::kFp8);
  TORCH_CHECK(valid, "mm_prompt: activations 0 BF16 (any weights), 1 FP8 rows (FP8 weights), 2 NVFP4 (NVFP4 weights)");
  TORCH_CHECK(splits >= 1 && splits <= mr::kMaxSplits, "mm_prompt: 1 to ", mr::kMaxSplits, " splits");
  const int64_t row_bytes =
      weights == mr::kNvfp4 || weights == mr::kInt4Zp ? k / 2 : weights == mr::kBf16 ? 2 * k : k;
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.dim() == 2 && w.size(1) * w.element_size() == row_bytes &&
                  aligned_to(w, 16) && n >= 1,
              "mm_prompt: w is contiguous [N][", row_bytes, " bytes] on 16 bytes");
  same_device(x, w);
  TORCH_CHECK(weights != mr::kBf16 || w.scalar_type() == torch::kBFloat16, "mm_prompt: BF16 weights are BF16");
  mr::Problem p;
  p.x = x.data_ptr();
  p.ldx = x.stride(0);
  p.w = static_cast<const std::uint8_t*>(w.data_ptr());
  p.M = static_cast<int>(m);
  p.N = static_cast<int>(n);
  p.K = static_cast<int>(k);
  p.weights = static_cast<int>(weights);
  p.acts = static_cast<int>(acts);
  p.splits = static_cast<int>(splits);
  if (acts != mr::kActBf16) {
    TORCH_CHECK(scratch.has_value(), "mm_prompt: quantized activations take a scratch buffer");
    const torch::Tensor& s = *scratch;
    same_device(x, s);
    TORCH_CHECK(s.is_contiguous() && aligned_to(s, 16) &&
                    s.numel() * s.element_size() >= (int64_t)mr::scratch_bytes((int)m, (int)k),
                "mm_prompt: scratch is contiguous on 16 bytes with rows * K * 2 + rows * 4 bytes or more (rounded)");
    p.scratch = static_cast<std::uint8_t*>(s.data_ptr());
  }
  if (weights == mr::kNvfp4 || weights == mr::kMxfp8) {
    TORCH_CHECK(scales.has_value(), "mm_prompt: NVFP4 and MXFP8 weights take their block scales");
    const torch::Tensor& s = *scales;
    same_device(x, s);
    TORCH_CHECK(s.is_contiguous() && s.element_size() == 1 && aligned_to(s, 2),
                "mm_prompt: scales are contiguous bytes");
    if (weights == mr::kMxfp8) {
      TORCH_CHECK(!blocked && s.numel() == n * (k / 32), "mm_prompt: MXFP8 scales are E8M0 bytes [N][K / 32]");
    } else if (blocked) {
      TORCH_CHECK(s.numel() == ((n + 127) / 128) * 128 * ((k / 16 + 3) / 4) * 4,
                  "mm_prompt: blocked scales are cuBLASLt's 128 x 4 tiles of [N][K / 16]");
      p.ct = static_cast<int>((k / 16 + 3) / 4);
    } else {
      TORCH_CHECK(s.numel() == n * (k / 16), "mm_prompt: NVFP4 scales are E4M3 bytes [N][K / 16]");
    }
    p.scales = static_cast<const std::uint8_t*>(s.data_ptr());
  }
  if (weights == mr::kInt4Zp) {
    TORCH_CHECK(scales.has_value() && zeros.has_value() && group >= 32 && group % 32 == 0 && k % group == 0,
                "mm_prompt: INT4 weights take FP16 scales and uint8 zeros [N][K / group], the group a multiple of 32 ",
                "dividing K");
    const torch::Tensor& s = *scales;
    const torch::Tensor& z = *zeros;
    same_device(x, s);
    same_device(x, z);
    TORCH_CHECK(s.is_contiguous() && s.scalar_type() == torch::kFloat16 && s.numel() == n * (k / group) &&
                    z.is_contiguous() && z.scalar_type() == torch::kUInt8 && z.numel() == n * (k / group),
                "mm_prompt: INT4 scales are FP16 and zeros uint8, [N][K / group] each");
    p.scales = static_cast<const std::uint8_t*>(s.data_ptr());
    p.zeros = z.data_ptr<std::uint8_t>();
    p.group = static_cast<int>(group);
  }
  if (weights == mr::kNvfp4 || weights == mr::kFp8) {
    TORCH_CHECK(row_scale.has_value(), "mm_prompt: NVFP4 and FP8 weights take a row scale [N]");
    const torch::Tensor& r = *row_scale;
    same_device(x, r);
    TORCH_CHECK(r.is_contiguous() && r.scalar_type() == torch::kFloat32 && r.numel() == n && aligned_to(r, 4),
                "mm_prompt: row_scale is FP32 [N]");
    p.row_scale = r.data_ptr<float>();
  }
  if (acts == mr::kActNvfp4) {
    TORCH_CHECK(input_scale > 0, "mm_prompt: NVFP4 activations take their input scale (> 0)");
    p.input_scale = static_cast<float>(input_scale);
    p.input_rcp = 1.0f / p.input_scale;
  }
  TORCH_CHECK(glu == 0 || (glu == 1 && n % 16 == 0),
              "mm_prompt: glu 1 takes interleaved gate and up rows, N a multiple of 16");
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.size(0) == m && out.size(1) == (glu ? n / 2 : n) &&
                  out.stride(1) == 1,
              "mm_prompt: out is [rows][N] with contiguous rows ([rows][N / 2] with glu)");
  p.glu = static_cast<int>(glu);
  same_device(x, out);
  p.out = out.data_ptr();
  p.ldo = out.stride(0);
  p.out_kind = out.scalar_type() == torch::kFloat32 ? mr::kOutF32
               : out.scalar_type() == torch::kBFloat16 ? mr::kOutBf16
                                                       : mr::kOutF16;
  TORCH_CHECK(p.out_kind != mr::kOutF16 || out.scalar_type() == torch::kFloat16,
              "mm_prompt: out is FP32, BF16 or FP16");
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaError_t e = mr::enqueue_prompt(p, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == cudaSuccess, "mm_prompt: ", cudaGetErrorString(e));
}

int64_t decode_ctas(int64_t weights, int64_t acts) {
  TORCH_CHECK(weights >= mr::kNvfp4 && weights <= mr::kInt4Zp, "decode_ctas: weights 0 NVFP4 .. 4 INT4");
  TORCH_CHECK(acts >= mr::kActBf16 && acts <= mr::kActNvfp4Mma, "decode_ctas: activations 0 to 3");
  const int n = mr::decode_ctas(static_cast<int>(weights), static_cast<int>(acts));
  TORCH_CHECK(n > 0, "decode_ctas: the kernel's occupancy is not known on this device (or acts 3 with weights "
                     "other than NVFP4)");
  return n;
}
