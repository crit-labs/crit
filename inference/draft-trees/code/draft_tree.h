// Token trees for speculative decoding on the device: EAGLE-2-style candidate expansion and selection, the tree of
// the kept candidates with its attention masks, the tree walk that accepts a path, and compaction of the accepted
// rows. Reference kernels, not tuned; see tree.py for the host reference of every rule.
//
// A tree has N <= 64 nodes in topological order: node 0 is the round's pending token, node i > 0 a draft token whose
// parent parents[i - 1] < i. masks[i] has bit j set for each node j on the root-to-i path (i included); depth[i] is
// popcount - 1. The target verifies all nodes in one pass with these masks (an attention kernel's tree
// masks, RoPE positions frontier + depth) and writes one prediction per node: its argmax, or for sampled
// requests its own draw at the node's depth (token-sampling Rows::tree).
#pragma once
#include "greedy_accept.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace crit::techniques::draft_trees {

constexpr int kMaxNodes = 64, kMaxBranch = 16;

// Candidates of `rows` requests, appended by expansion in a fixed order (frontier entry, then rank), so indices and
// tie-breaks never depend on timing. Index -1 is the root; a hole (an unused frontier entry) has depth -1.
struct Candidates {
    std::int32_t* token = nullptr;   // [rows][capacity]
    std::int32_t* parent = nullptr;  // [rows][capacity]
    float* score = nullptr;          // [rows][capacity]: draft log-probability of the root-to-candidate path
    std::int32_t* depth = nullptr;   // [rows][capacity]
    std::int32_t* count = nullptr;   // [rows]: candidates so far; reset to 0 before a round's first expansion
    int capacity = 0;
};

// Frontier entry f of request r (frontier[r * width + f]: a candidate index, -1 for the root, -2 unused) has the draft
// logits row logits + (r * width + f) * stride. Appends `branch` children per entry, the best by (logit desc, id asc),
// scored parent score + log_softmax, at count[r] + f * branch + i; then count[r] += width * branch.
cudaError_t enqueue_expand(const float* logits, int stride, int vocab, const std::int32_t* frontier, int width, int branch,
                           const Candidates& candidates, int rows, cudaStream_t stream);

// The m best candidates of each request by (score desc, index asc), among depth `depth` (or every depth when 0), into
// out[r * m + i]; -1 pads. With depth = the newest depth it is the next frontier (the beam); with 0 the kept nodes.
cudaError_t enqueue_select(const Candidates& candidates, int depth, int m, std::int32_t* out, int rows, cudaStream_t stream);

// The tree of the kept candidates kept[r * (nodes - 1) + i] (rank order, from enqueue_select with depth 0), nodes by
// depth then rank. Writes drafts[r * (nodes - 1) + i - 1] = token of node i (the device round's drafts), parents in the
// same layout, masks[r * nodes + i] and depths[r * nodes + i]; with tokens set, tokens[r * nodes + i] including the
// root token roots[r * root_stride] (for token-sampling's Rows::tree_tokens). With fewer kept candidates than nodes the
// rest are padding nodes: token 0 under the root with parent -1, which the walk never follows.
struct Build {
    const std::int32_t* kept = nullptr;
    const std::int32_t* roots = nullptr;
    int root_stride = 1;
    std::int32_t* drafts = nullptr;
    std::int32_t* parents = nullptr;
    std::uint64_t* masks = nullptr;
    std::int32_t* depths = nullptr;
    std::int32_t* tokens = nullptr;
    int nodes = 0, rows = 0;
};
cudaError_t enqueue_build(const Candidates& candidates, const Build& build, cudaStream_t stream);

// The walk, with device_round's control and packed-result contract (so enqueue_truncate and the round's commit work
// unchanged): rows.drafts holds node tokens 1..N-1 (draft_tokens = N - 1), rows.predictions one per node. From the
// root, move to the child whose token equals the node's prediction, emitting it, until no child matches, then emit
// that prediction; stop after an accepted stop token or at control[output_budget]. path[r * N + i] gets 0 and then
// the accepted nodes (-1 pads): the rows to keep are path[0 .. n_emitted - 1].
cudaError_t enqueue_tree_accept(const device_round::AcceptRows& rows, const std::int32_t* parents, std::int32_t* path,
                                cudaStream_t stream);

// Draft ranks, to choose tree widths from data: after a chain round's accept (device_round's control block), for each
// draft position d the round compared (d < control[n_matched], or d == n_matched when the walk emitted a correction
// there, n_emitted > n_matched; a round cut off by its output budget or ending on an accepted stop never compared
// the next position; d < draft_tokens), the rank of the target's prediction d among the drafter's logits at
// d (logit desc, id asc: the order enqueue_expand takes children in), counted in
// histogram[d * (ranks + 1) + min(rank, ranks)]; the last column counts ranks >= `ranks`. Row d's rank-0 count is the
// drafts accepted at d, and the share of its counts below w is the share of reached rounds whose target token a tree of
// width w at depth d would have held (from the chain's context). Counts accumulate over rounds (zero the histogram
// once), the launch has no host dependency, so it can run in a captured round, and the host reads it when it decides.
struct DraftRanks {
    const float* logits = nullptr;              // [rows][draft_tokens][stride]: the drafter's row at each position
    int stride = 0, vocab = 0;
    const std::int32_t* predictions = nullptr;  // [rows][draft_tokens + 1]
    const std::int32_t* control = nullptr;      // [rows][control::fields], after the round's accept
    unsigned long long* histogram = nullptr;    // [draft_tokens][ranks + 1]
    int rows = 0, draft_tokens = 0, ranks = 0;
};
cudaError_t enqueue_draft_ranks(const DraftRanks& ranks, cudaStream_t stream);

// Rows of one state array: node i of request r lives at base + slot * row_stride with slot = slots[r * nodes + i], or
// control[r][base_frontier] + i when slots is null (a contiguous cache). row_bytes per row, 16-byte aligned for the
// vector path.
struct RowArray {
    void* base = nullptr;
    std::size_t row_bytes = 0, row_stride = 0;
    const std::int32_t* slots = nullptr;
};
// Moves the kept rows path[i] -> i for 1 <= i < n_emitted, in increasing i (a later move never reads a row an earlier
// one wrote, since path nodes increase), for `count` arrays (a device table) of every request. enqueue_settle then sets
// path[i] = i for i < n_emitted, so rerunning both after a Core truncation (smaller n_emitted) moves nothing. Run both
// in the round's commit_state.
cudaError_t enqueue_compact(const RowArray* arrays, int count, const std::int32_t* control, const std::int32_t* path, int nodes,
                            int rows, cudaStream_t stream);
cudaError_t enqueue_settle(const std::int32_t* control, std::int32_t* path, int nodes, int rows, cudaStream_t stream);

}  // namespace crit::techniques::draft_trees
