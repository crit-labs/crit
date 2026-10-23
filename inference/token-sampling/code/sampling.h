// Token selection on the device after the head: greedy, or temperature / top-k /
// top-p sampling. Draws are Core's counter-based uniform (crit/sampling.h), so the
// device follows crit::core::Sampler's rules and draw for the same row and
// parameters, with top_k capped at kMaxTopK and weights in FP32.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace crit::techniques::token_sampling {

constexpr int kMaxTopK = 64;     // top_k 0 (no filter) is capped at this many candidates
constexpr int kStageBlocks = 256;

// Device-resident; one per sequence. position advances by one per selected token.
struct Params {
    float temperature = 0.f;    // <= 0 selects greedily
    float top_p = 1.f;          // in (0, 1]
    int top_k = 0;              // 0 = kMaxTopK
    std::uint32_t stream = 0;   // Core's draw stream (SamplingParams::stream), e.g. the choice index
    std::uint64_t seed = 0;
    std::uint64_t position = 0; // Core's draw position: committed outputs before this selection
};

// logits[vocab] -> *out. Candidates rank by (value descending, index ascending), so
// greedy selection returns the lowest index among tied maxima. A sampled token uses
// u = crit::core::uniform(seed, position, stream, DrawPurpose::sample). Scratch:
// cand_values and cand_ids hold kStageBlocks * kMaxTopK entries each.
cudaError_t enqueue_select(const float* logits, int vocab, Params* params, float* cand_values, std::int32_t* cand_ids,
                           std::int32_t* out, cudaStream_t stream);

// Structured output on the device: sets logits[r * stride + t] to -inf wherever bit t % 32 of word
// mask[r * mask_stride + t / 32] is clear (crit::core::TokenMatcher::allowed_bits), for `rows` rows of `vocab` logits.
// Selection on the masked row then follows Core's constrained Sampler (AllowedTokens), ties to the lowest allowed id.
// mask_stride >= (vocab + 31) / 32 words; stride >= vocab.
cudaError_t enqueue_mask(float* logits, int vocab, int stride, const std::uint32_t* mask, int mask_stride, int rows,
                         cudaStream_t stream);

} // namespace crit::techniques::token_sampling
