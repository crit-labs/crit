#pragma once
// Tiny synthetic target for the composed device-round example: embedding, one
// causal attention layer, one gated-delta layer and an output head, plus a bigram
// drafter. Plain reference kernels, not for performance. Every column's arithmetic
// is independent of how many columns a launch processes, so a K+1-column verify
// and a one-column ordinary step produce bitwise-identical predictions.
#include <cstdint>
#include <cuda_runtime_api.h>

namespace crit::techniques::device_round::toy {
inline constexpr std::uint32_t vocabulary = 48, width = 32, keys = 16, values = 16;
inline constexpr std::uint32_t context = 256, max_columns = 8;

struct Weights { // FP32, row-major [out][in]
    const float* embed = nullptr; // [vocabulary][width]
    const float* wq = nullptr; const float* wk = nullptr; const float* wv = nullptr; const float* wo = nullptr; // [width][width]
    const float* pq = nullptr; const float* pk = nullptr; // [keys][width]
    const float* pv = nullptr;                            // [values][width]
    const float* pa = nullptr; const float* pb = nullptr; // [width] decay and beta gates
    const float* po = nullptr;                            // [width][values]
    const float* head = nullptr;                          // [vocabulary][width]
};
struct State { // per request
    float* k_cache = nullptr; float* v_cache = nullptr;   // [context][width]
    float* x1 = nullptr;                                  // [max_columns][width]
    float* q = nullptr; float* k = nullptr;               // [max_columns][keys]
    float* v = nullptr;                                   // [max_columns][values]
    float* log_decay = nullptr; float* beta = nullptr;    // [max_columns]
    float* gdn_out = nullptr;                             // [max_columns][values]
    float* gdn_live = nullptr; float* gdn_start = nullptr; // [keys][values]
    float* gdn_checkpoints = nullptr;                     // [max_columns][keys][values]
};
// Feed inputs[0..columns) at positions *frontier.., writing KV rows, gated-delta
// checkpoints for every column and predictions[t] = greedy argmax after column t.
// The live gated-delta state is copied to gdn_start first and left unchanged.
cudaError_t enqueue_forward(const Weights&, const State&, const std::int32_t* inputs, const std::int32_t* frontier,
                            std::uint32_t columns, std::int32_t* predictions, cudaStream_t);
// Live gated-delta state = checkpoint[*accepted - 1] (start when zero); no host read.
cudaError_t enqueue_commit(const State&, const std::int32_t* accepted, std::uint32_t columns, cudaStream_t);
// inputs = {control[last_token], drafts[0..k)}.
cudaError_t enqueue_round_inputs(const std::int32_t* control, const std::int32_t* drafts, std::uint32_t k,
                                 std::int32_t* inputs, cudaStream_t);
// drafts[0] = table[control[last_token]], drafts[i] = table[drafts[i-1]].
cudaError_t enqueue_bigram_draft(const std::int32_t* table, const std::int32_t* control, std::uint32_t k,
                                 std::int32_t* drafts, cudaStream_t);
cudaError_t enqueue_set(std::int32_t* target, std::int32_t value, cudaStream_t);
}
