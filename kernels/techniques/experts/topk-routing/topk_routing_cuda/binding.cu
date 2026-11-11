// Torch entry point over topk_routing.h: checks and the current stream.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <optional>

#include "topk_routing.h"

namespace tr = crit::techniques::topk_routing;

void route(const torch::Tensor& logits, int64_t top, const std::optional<torch::Tensor>& bias, bool softmax,
           bool before, bool renormalize, double epsilon, double floor, double scaling, torch::Tensor& ids,
           torch::Tensor& weights) {
    const c10::cuda::CUDAGuard guard(logits.device());
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2 &&
                    logits.stride(1) == 1,
                "logits are [tokens][experts] FP32 CUDA rows");
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type() == torch::kInt64 &&
                    ids.size(0) == logits.size(0) && ids.size(1) == top,
                "ids are [tokens][top] int64");
    TORCH_CHECK(weights.is_cuda() && weights.is_contiguous() && weights.scalar_type() == torch::kFloat32 &&
                    weights.sizes() == ids.sizes(),
                "weights are [tokens][top] FP32");
    const float* b = nullptr;
    if(bias && bias->defined() && bias->numel()) {
        TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() && bias->scalar_type() == torch::kFloat32 &&
                        bias->numel() == logits.size(1),
                    "bias is [experts] FP32");
        b = bias->data_ptr<float>();
    }
    tr::Rule rule;
    rule.softmax = softmax, rule.before = before, rule.renormalize = renormalize;
    rule.epsilon = static_cast<float>(epsilon), rule.floor = static_cast<float>(floor);
    rule.scaling = static_cast<float>(scaling);
    const cudaError_t e = tr::enqueue_route(logits.data_ptr<float>(), static_cast<int>(logits.stride(0)),
                                            static_cast<int>(logits.size(0)), static_cast<int>(logits.size(1)),
                                            static_cast<int>(top), b, rule,
                                            reinterpret_cast<long long*>(ids.data_ptr<int64_t>()),
                                            weights.data_ptr<float>(), at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(e == cudaSuccess, "route: ", cudaGetErrorString(e));
}
