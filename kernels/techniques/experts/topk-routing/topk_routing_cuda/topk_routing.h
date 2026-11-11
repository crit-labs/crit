// Top-k expert routing of mixture-of-experts router logits in one kernel, a warp per token row, for crit's topk-router
// plugin (inference/base/topk-router) and its Routing description (heterogeneous-expert-dispatch's routing.py): softmax
// or sigmoid scores before or after selection, a per-expert correction bias that only changes the selection,
// renormalization and scaling. Ties go to the lower expert id; a NaN logit ranks below every number (-inf included), so
// a row of NaN logits (a captured round's junk positions) still picks valid distinct experts (the lowest ids) and gets
// weights 0. One launch, no host synchronization: a call at a fixed token count is captured in a CUDA graph. Group
// limits are the plugin's torch path.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace crit::techniques::topk_routing {

struct Rule {
    int softmax = 1;            // 1 softmax, 0 sigmoid
    int before = 0;             // score every expert before selection (else score the selected logits)
    int renormalize = 0;        // divide by max(sum + epsilon, floor)
    float epsilon = 0.0f, floor = 0.0f, scaling = 1.0f;
};

constexpr int kMaxExperts = 1024, kMaxTop = 32;

// logits [tokens][experts] FP32 (rows ld apart), bias [experts] FP32 or null -> ids [tokens][top] int64 (selection
// order), weights [tokens][top] FP32.
cudaError_t enqueue_route(const float* logits, int ld, int tokens, int experts, int top, const float* bias,
                          const Rule& rule, long long* ids, float* weights, cudaStream_t stream);

}  // namespace crit::techniques::topk_routing
