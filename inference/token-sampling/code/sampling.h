// Token selection on the device after the head: greedy, or Core's sampling law over the
// full vocabulary. Draws are Core's counter-based uniform (crit/sampling.h), so the device
// follows crit::core::Sampler's rules and draw for the same row, parameters and history,
// with FP32 values and weights. Speculative verification uses the same law per column.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace crit::core { enum class DrawPurpose : std::uint32_t; }

namespace crit::techniques::token_sampling {

constexpr int kMaxTopK = 64;     // enqueue_select's fast path for 1 <= top_k <= kMaxTopK
constexpr int kStageBlocks = 256;

// Device-resident; one per sequence. position advances by one per selected token.
struct Params {
    float temperature = 0.f;    // <= 0 selects greedily
    float top_p = 1.f;          // in (0, 1]
    int top_k = 0;              // 0 = no top-k filter (the full vocabulary)
    std::uint32_t stream = 0;   // Core's draw stream (SamplingParams::stream), e.g. the choice index
    std::uint64_t seed = 0;
    std::uint64_t position = 0; // Core's draw position: committed outputs before this selection
};

// logits[vocab] -> *out. Candidates rank by (value descending, index ascending), so
// greedy selection returns the lowest index among tied maxima. A sampled token uses
// u = crit::core::uniform(seed, position, stream, DrawPurpose::sample). With
// 1 <= top_k <= kMaxTopK selection runs in two stages over the cand_values and cand_ids
// scratch (kStageBlocks * kMaxTopK entries each); top_k 0 or above kMaxTopK runs the
// full-vocabulary law. The parameters choose the path on the device, so one captured
// launch serves both. *out is -1 when no logit is selectable (all -inf), on every path.
cudaError_t enqueue_select(const float* logits, int vocab, Params* params, float* cand_values, std::int32_t* cand_ids,
                           std::int32_t* out, cudaStream_t stream);

// Structured output on the device: sets logits[r * stride + t] to -inf wherever bit t % 32 of word
// mask[r * mask_stride + t / 32] is clear (crit::core::TokenMatcher::allowed_bits), for `rows` rows of `vocab` logits.
// Selection on the masked row then follows Core's constrained Sampler (AllowedTokens), ties to the lowest allowed id.
// mask_stride >= (vocab + 31) / 32 words; stride >= vocab.
cudaError_t enqueue_mask(float* logits, int vocab, int stride, const std::uint32_t* mask, int mask_stride, int rows,
                         cudaStream_t stream);

// ---- Core's whole law on the device -------------------------------------------------------------
// Device-resident, one per sequence, set from crit::core::resolve_sampling. Steps 2 to 5 and 7 to 12
// of Core's Sampler (core/src/sampling/README.md) run on the device; constraint masks go through
// enqueue_mask first. Pointers are device memory owned by the engine and borrowed by every launch.
struct Request {
    float temperature = 0.f;            // <= 0 greedy (lowest id on ties)
    float top_p = 1.f;                  // (0, 1]
    float min_p = 0.f;                  // [0, 1]; keeps weights >= min_p relative to the top weight
    std::int32_t top_k = 0;             // 0 or >= the selectable count: no top-k filter
    float repetition = 1.f;             // > 0; 1 disables
    float frequency = 0.f, presence = 0.f;
    std::uint32_t min_tokens = 0;       // suppress_ids are banned while position + column < min_tokens
    std::uint32_t stream = 0;
    std::uint64_t seed = 0;
    std::uint64_t position = 0;         // committed outputs: Core's outputs(), the draw position of column 0
    const std::int32_t* bias_ids = nullptr;   // ascending unique ids (merge duplicates on the host)
    const float* bias_values = nullptr;       // added to the logit; -inf bans the token
    std::uint32_t bias_count = 0;
    const std::int32_t* suppress_ids = nullptr;  // Core's suppress_until_min; ids outside the vocabulary are ignored
    std::uint32_t suppress_count = 0;
    std::int32_t* counts = nullptr;           // [vocab] committed output counts n(t); null counts nothing
    const std::uint8_t* prompt_seen = nullptr; // [vocab] nonzero for prompt tokens the repetition penalty counts
};

// Logit rows of one or more requests, `columns` per request: row (r, c) starts at
// logits + (r * columns + c) * stride and holds `vocab` valid entries. Column c counts
// staged[r * staged_stride + i] for i < staged_base + c as provisional outputs (verify drafts,
// or the drafts before a draft step) and draws at request[r].position + position_base + c.
// With `tree` set, columns are the nodes of a token tree instead: column c's prefix is its
// root-to-node path (bit j of tree[r * columns + c] set for each node j on it, c included), it
// stages tree_tokens[r * columns + j] for every set bit j >= 1 (node 0, the root, is committed)
// and draws at position + position_base + depth, depth = popcount - 1.
struct Rows {
    const float* logits = nullptr;
    std::uint32_t stride = 0, vocab = 0;
    std::uint32_t requests = 1, columns = 1;
    const Request* request = nullptr;         // [requests]
    const std::int32_t* staged = nullptr;     // [requests][staged_stride]; may be null with staged_base 0 and no verify
    std::uint32_t staged_stride = 0, staged_base = 0;
    std::uint32_t position_base = 0;
    float* scratch = nullptr;                 // scratch_floats(vocab, requests * columns) processed values
    const std::uint64_t* tree = nullptr;      // [requests][columns] path masks, columns <= 64; overrides staged
    const std::int32_t* tree_tokens = nullptr; // [requests][columns] node tokens
};
std::size_t scratch_floats(std::uint32_t vocab, std::uint32_t rows) noexcept;

// One draw per row from its law with Core's uniform at the row's position and `purpose`
// (sample for output tokens, draft for a drafter's own draws). out[r * columns + c] gets the token,
// or -1 when processing leaves no selectable token. With `probabilities` non-null, row (r, c) of
// [requests][columns][vocab] receives the law as dense FP32 probabilities (zero outside the kept set):
// the q a sampled draft passes to enqueue_verify.
cudaError_t enqueue_sample(const Rows&, crit::core::DrawPurpose purpose, std::int32_t* out, float* probabilities,
                           cudaStream_t stream);

// Speculative verification, per request and column, writing predictions the device greedy accept
// kernel consumes unchanged (it emits drafts while drafts[c] == predictions[c], then the first
// differing prediction):
//   greedy drafts (draft_probabilities null): every column writes the target's own draw from p_c with
//                  u_sample. A draft is accepted exactly when it equals that draw: probability p_c(d),
//                  the rejection rule for a one-hot q, and the emitted text equals plain sampled
//                  decoding token for token for the same seed when the verify logits equal one-token
//                  steps (column-invariant kernels);
//   sampled drafts: column c < K accepts draft d with u_accept * q_c(d) < p_c(d) (Leviathan et al.) and
//                  writes d, otherwise a draw from max(p_c - q_c, 0) with u_residual; column K writes
//                  the bonus from p_K with u_sample.
// rows.columns is K + 1 target columns; column c's law stages drafts[0, c) unless rows.staged is set.
// Draws use Core's uniform at request.position + position_base + c, so an emitted token at output
// position n depends only on draws at n and the round output is a sample of the target law whatever
// the drafts were. Temperature 0 writes each column's argmax, the greedy rule. Token trees with greedy
// drafts need no verify call: enqueue_sample over the nodes (Rows::tree) writes each node's draw.
struct Verify {
    Rows target;
    const std::int32_t* drafts = nullptr;              // [requests][K]
    const float* draft_probabilities = nullptr;        // [requests][K][vocab] laws the drafts were drawn from; null: greedy drafts
    std::int32_t* predictions = nullptr;               // [requests][K + 1]
};
cudaError_t enqueue_verify(const Verify&, cudaStream_t stream);

// After publication: add tokens[r * token_stride + i] for i < n to request r's counts and advance its
// position by n, where n = emitted ? emitted[r * emitted_stride] : 1. In a device speculative round
// pass the packed result's tokens and control[n_emitted] as the first Forward kernel, so the history
// follows Core's (possibly truncated) count before any law of the next round reads it.
cudaError_t enqueue_commit(Request* requests, std::uint32_t count, std::uint32_t vocab, const std::int32_t* tokens,
                           std::uint32_t token_stride, const std::int32_t* emitted, std::uint32_t emitted_stride,
                           cudaStream_t stream);

} // namespace crit::techniques::token_sampling
