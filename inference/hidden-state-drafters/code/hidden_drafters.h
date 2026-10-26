// Device glue for drafters that read the target's hidden states (EAGLE-3) or draft a block per pass (DFlash, DSpark),
// over device-speculative-round's control block; see drafting.py for the host reference of each rule. Reference
// kernels: the drafter's own arithmetic (fc projection, decoder layer, head) is the engine's.
#pragma once
#include "greedy_accept.h"

#include <cstdint>
#include <cuda_runtime_api.h>

namespace crit::techniques::hidden_drafters {

// EAGLE-3 catch-up over a fixed width of K + 1 columns: column c of request r gets tokens[r * (K + 1) + c] = the c-th
// token the round emitted (result[r][1 + c]) and positions[...] = control[base_frontier] + c for c < n_emitted, -1
// beyond. The drafter reads verify column c's taps beside them (its input at position p is the target's taps at p
// with the token at p + 1). Run first in the next round's draft step: Core's truncation has settled n_emitted by then.
cudaError_t enqueue_eagle_ingest(const std::int32_t* control, const std::int32_t* result, std::uint32_t draft_tokens,
                                 std::int32_t* tokens, std::int32_t* positions, std::uint32_t rows, cudaStream_t stream);
// The same catch-up over `columns` W columns (tokens and positions [rows][W]) from result rows `result_stride` int32
// apart (>= W + 1). With draft widths chosen per round over buffers sized for the widest K, the previous round may
// have emitted up to that width + 1 tokens, so W is the widest K + 1 whatever this round's K is (columns c >= n_emitted
// stay -1); the K + 1 form would drop the tokens a wider round emitted beyond the narrower one's K + 1.
cudaError_t enqueue_eagle_ingest(const std::int32_t* control, const std::int32_t* result, std::uint32_t result_stride,
                                 std::uint32_t columns, std::int32_t* tokens, std::int32_t* positions, std::uint32_t rows,
                                 cudaStream_t stream);

// A block drafter's input: [control[last_token], mask_token x (block - 1)] at positions control[frontier] + i.
cudaError_t enqueue_block_input(const std::int32_t* control, std::int32_t mask_token, std::uint32_t block,
                                std::int32_t* tokens, std::int32_t* positions, std::uint32_t rows, cudaStream_t stream);

// Draft-vocabulary ids to target ids in place: id -> id + d2t[id] (EAGLE-3's offset table); negative ids stay.
cudaError_t enqueue_map_tokens(std::int32_t* ids, std::uint32_t count, const std::int64_t* d2t, std::uint32_t draft_vocab,
                               cudaStream_t stream);

// A sampled drafter's draft-vocabulary law rows q[rows][draft_vocab] as target-vocabulary rows out[rows][target_vocab]
// (zeros elsewhere), the q token-sampling's enqueue_verify takes.
cudaError_t enqueue_scatter_law(const float* q, std::uint32_t draft_vocab, const std::int64_t* d2t, float* out,
                                std::uint32_t target_vocab, std::uint32_t rows, cudaStream_t stream);

// From the first draft whose confidence[r * K + i] is below threshold on, drafts[r * K + i] = -1, which no prediction
// matches: the round stops there with the captured width unchanged.
cudaError_t enqueue_truncate_drafts(std::int32_t* drafts, const float* confidence, std::uint32_t draft_tokens, float threshold,
                                    std::uint32_t rows, cudaStream_t stream);

}  // namespace crit::techniques::hidden_drafters
