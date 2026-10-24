// Causal prompt attention on BF16 tensor cores without a score matrix in memory (flash style), for grouped-query
// attention over a contiguous KV cache.
//
// For query t of a chunk (absolute position pos0 + t), keys and values at positions 0 .. pos0 + t are visible;
// the cache holds positions 0 .. pos0 + T - 1, the chunk's own keys included. Head h reads key/value head
// h / (heads / kv_heads). out[t][h] = sum_j softmax_j(scale * q_t . k_j) v_j, accumulated in FP32 with the
// probabilities rounded to BF16 for the P V product (the row sums use the same rounded values).
//
// Variants several model families use: a sliding window (query position p sees keys p - window + 1 .. p, and keys
// before the chunk's first window are never read, so a ring of window + T - 1 rows holds a window layer: position j
// at row j % ring), per-head sink logits joining each row's softmax denominator, and a logit soft-cap
// (score = softcap * tanh(score / softcap) after the scale).
#pragma once
#include <cstddef>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace crit::flash_prefill {

struct Problem {
    const __nv_bfloat16* q;    // [T][q_stride]; head h at column h * head_dim
    int q_stride;
    const __nv_bfloat16* k;    // [pos0 + T][kv_stride]; head g at column g * head_dim
    const __nv_bfloat16* v;    // same layout as k
    int kv_stride;
    float* out;                // [T][out_stride]; head h at column h * head_dim
    int out_stride;
    int T, pos0, heads, kv_heads, head_dim;
    float scale;               // multiplies q . k before the softmax (1 / sqrt(head_dim), or 1 for pre-scaled q)
    int window = 0;            // > 0: sliding window of `window` keys, the query's own included
    int ring = 0;              // > 0: position j is row j % ring of k and v (needs window; ring >= window + T - 1)
    const float* sinks = nullptr;  // [heads] sink logits in scaled-score units; null: none
    float softcap = 0.f;       // > 0: score = softcap * tanh(score / softcap)
    bool causal = true;        // false: every query also sees the chunk's later keys (a block drafter's non-causal block)
};

// Supported head_dim: 64, 128, 256. Rows of q, k and v must be 16-byte aligned (strides multiples of 8).
// Returns cudaErrorInvalidValue for unsupported geometry; otherwise the launch error.
cudaError_t enqueue(const Problem& problem, cudaStream_t stream);

// Dynamic shared memory one block uses for a head dimension (64 queries, two 32-key K/V stages).
std::size_t shared_bytes(int head_dim);

}  // namespace crit::flash_prefill
