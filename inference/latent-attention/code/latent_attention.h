// Absorbed multi-head latent attention (MLA) over a latent cache: every head attends over the same cache rows
// [latent ; rope key] with its absorbed query [W_UK^T q_nope ; q_rope] and gets its latent output sum_j p_j latent_j,
// to which the engine applies W_UV. See reference.py for the identity with per-head attention.
//
// Cache: BF16 rows of `rank + rope` values per position ([latent][rope], the rope key with RoPE applied), contiguous
// (position j at row j) or paged through a block table (Core's StateCache: row table[j / page_tokens] * page_tokens +
// j % page_tokens). Decode, speculative verify and (reference-speed) prompt chunks share one kernel: T query columns
// at positions pos0 + t, each seeing keys 0 .. pos0 + t. Work splits over keys in 32-key tiles assigned by position
// (tile i to split i mod splits) and a combine adds the splits in order, so a column's result depends only on its own
// position and `splits`: a column decoded alone equals the same column in a verify.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace crit::latent_attention {

struct Problem {
    const __nv_bfloat16* q = nullptr;     // [T][heads][rank + rope] absorbed queries (W_UK^T q_nope ; q_rope)
    const __nv_bfloat16* cache = nullptr; // rows of rank + rope values
    const std::int32_t* table = nullptr;  // optional block table; null: contiguous rows
    int page_tokens = 0;                  // with a table
    float* out = nullptr;                 // [T][heads][rank] latent outputs, FP32
    int T = 0, pos0 = 0, heads = 0;       // heads a multiple of 8
    int rank = 512, rope = 64;            // rank a multiple of 32, rank + rope a multiple of 8 and at most 640
    float scale = 0.f;                    // multiplies the 576-wide dot product (1 / sqrt(qk_nope + qk_rope) in DeepSeek)
    const float* sinks = nullptr;         // optional [heads] sink logits joining each row's denominator
    int splits = 1;                       // key splits per column, 1 .. 256; about SMs / (T * heads / 8)
    float* partial = nullptr;             // partial_floats(...) floats of scratch
};
std::size_t partial_floats(int T, int heads, int rank, int splits);
cudaError_t enqueue(const Problem&, cudaStream_t stream);

}  // namespace crit::latent_attention
