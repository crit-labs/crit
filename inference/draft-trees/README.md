# Verify a tree of drafts in one target pass

Draft several continuations at once as a token tree, verify every node in one target pass
with an ancestor-only attention mask, and accept the longest path the target agrees with.
At batch size 1 a pass over a few dozen tokens costs about as much as one token, so a tree
that covers the drafter's second and third choices accepts more tokens per round than a
chain of its first choices. Every emitted token is still the target's own choice (argmax,
or its own draw for sampled requests), so trees change how many tokens a round emits, never
which.

## When to use it

Use it once a chain round is correct on the device
([device-speculative-round](../device-speculative-round/README.md)) and the drafter's
first choice is often wrong while its second or third is right: confidence varies along
the draft, as with MTP heads and EAGLE-style drafters. Size the tree by measured verify
cost: published batch-1 numbers on an 8B model show acceptance length rising from 2.25
(chain) to 2.92 (wide tree) while the per-node acceptance rate falls, so a tree pays only
while its extra verify columns stay cheap.

Not for layers with recurrent state (gated delta, Mamba): their state after a node depends
on its path, so a tree needs one state per node; keep chains for hybrid models. Not for
batched serving where verify is compute bound, and not with drafts sampled from the
drafter's law (multi-draft rejection sampling is not implemented): tree children are
deterministic (top-k).

## How it works

**Tree.** `N <= 64` nodes in topological order. Node 0 is the pending token, node `i > 0`
a draft token with parent `parents[i] < i`; `mask[i]` has bit `j` set for every node `j`
on its root-to-node path, `depth[i] = popcount(mask[i]) - 1`.

**Drafting (EAGLE-2).** Expand the root's `branch` best tokens; for `depth - 1` more
steps run the drafter on the `beam` best candidates of the newest depth (with the tree
mask in the drafter's own attention) and expand each. A candidate's score is its path's
draft log-probability; keep the `N - 1` best of all depths. Children never outscore their
parents and ties go to the earlier candidate, so the kept set contains every kept node's
parent. Nodes are ordered by depth, then score, so parents come first. A NaN logit ranks
as -inf, and a candidate scoring -inf is never kept, so a row of NaN from the drafter
drafts nothing instead of an out-of-range token.

**Verify.** One target pass over the `N` nodes at KV slots `frontier + i` and RoPE
positions `frontier + depth[i]`, with the masks as attention trees (a decode attention kernel that takes a per-row
ancestor mask). Each node's prediction is the target's
argmax, or for a sampled request its own draw at output position `outputs + depth[i]`:
[token-sampling](../token-sampling/README.md)'s `enqueue_sample` with `Rows::tree`, which
also stages the path's tokens for penalties.

**Walk.** From the root, move to the child whose token equals the node's prediction,
emitting it; when no child matches, emit the prediction (a correction, or the bonus at a
leaf); stop after an accepted stop token or at the output allowance. With greedy
predictions this is greedy speculation over the tree; with the target's own draws a node's
emitted token is exactly what plain sampling draws there, so the law is the model's.

**Compaction.** Emitting `m` tokens materializes the root and the first `m - 1` accepted
nodes, whose rows sit at `frontier + path[i]`. Moving `path[i] -> i` in increasing `i`
places them at `frontier + 0 .. m - 1` without overwriting a row a later move reads (path
nodes increase). Afterwards the path is set to the identity, so rerunning commit after a
Core truncation moves nothing.

## What the code supplies

| File | Supplies |
| --- | --- |
| [tree.py](code/tree.py) | Host reference: `expand`, `draft` (EAGLE-2 candidates and the kept tree), `build`, `chain`, `accept`, `compaction`, `apply`, `draft_rank` and `DraftRanks` |
| [draft_tree.h](code/draft_tree.h) / [draft_tree.cu](code/draft_tree.cu) | CUDA `enqueue_expand` (top-`branch` children with log-softmax scores, fixed candidate order), `enqueue_select` (beam or kept nodes by score, index tie-break), `enqueue_build` (tokens, parents, masks, depths, padding nodes), `enqueue_tree_accept` (device-round control and packed-result contract), `enqueue_compact`, `enqueue_settle` and `enqueue_draft_ranks` (per-position rank histogram) |

`device_round::RoundSteps::accept` lets [device-speculative-round](../device-speculative-round/README.md)
run the tree walk in its Finish graph instead of the greedy kernel, with the same control
block, truncation and single readback.

## What the engine supplies

The drafter's forward over the frontier (its own KV and tree mask), the target's verify
over all nodes with the masks and depth positions, predictions (argmax kernel or
token-sampling), and the state arrays to compact (every attention layer's K, V and scale
rows, through `RowArray`, with Core's `Round::kv` slots for paged caches). A tree round
with the device controller:

```text
draft:        enqueue_expand (root), then per depth: drafter on the beam, enqueue_select(depth), enqueue_expand
              enqueue_select(0) -> kept, enqueue_build -> drafts (the round's), parents, masks, depths, tokens
verify:       target over N nodes with Decode::tree = masks, RoPE positions frontier + depths
              predictions: argmax per node, or enqueue_sample(Rows::tree = masks, tree_tokens = tokens)
accept:       enqueue_tree_accept(rows, parents, path)          (RoundSteps::accept)
commit_state: enqueue_compact(arrays, ...), enqueue_settle(...)
```

Reserve `N` KV slots per round (Core's `max_round_rows`), `branch * (1 + (depth - 1) *
beam)` candidates per request and the path and mask buffers. With fewer candidates than
nodes the build pads with token 0 under no parent, which the walk never follows.

## Choosing widths from data

Measure before building a tree. Run chain rounds and, after each accept, `enqueue_draft_ranks`
with the drafter's logit row at each draft position: for every position `d` the round
compared (`d < n_matched`, and `d = n_matched` when a correction was emitted there; a round
cut off by its output budget or ending on an accepted stop compared no further) it counts the rank of the
target's token in the drafter's row, ranked as `enqueue_expand` ranks children, in a device
histogram `[positions][ranks + 1]`, with no readback. Row `d`'s rank-0 count is the drafts
accepted at `d`; the share of its counts below `w` (`DraftRanks.covered(d, w)` in tree.py)
is how often a tree of width `w` at depth `d` would have held the target's token. Widen a
depth only where that share grows by more than the extra verify columns cost, measured on
the device that runs the engine. The counts come from the chain's context: past the first
widened depth they estimate a tree's acceptance, they do not measure it.

## Worked example

From `code/`: a toy drafter that prefers `x + 1`, then `x + 2`, after token `x`, and a
target that takes the drafter's second choice after the first token.

```python
from tree import DraftRanks, accept, apply, chain, compaction, draft

def drafter(paths):
    """After token x the drafter prefers x + 1, then x + 2, ... (mod 10)."""
    return [[-1.5 * ((t - (p[-1] if p else 0) - 1) % 10) for t in range(10)] for p in paths]

_, tree = draft(drafter, root=0, depth=3, branch=2, beam=2, nodes=7)
assert tree.tokens == (0, 1, 2, 2, 3, 3, 3) and tree.parents == (-1, 0, 0, 1, 1, 2, 3)
assert tree.masks[6] == 0b1001011 and tree.depths[6] == 3       # node 6: path 0 -> 1 -> 3 -> 6

target = {(): 1, (1,): 3, (1, 3): 7}                          # the target's next token after each path
def predictions(t):
    paths = {0: ()}
    for i in range(1, t.nodes):
        paths[i] = paths[t.parents[i]] + (t.tokens[i],)
    return [target.get(paths[i], -1) for i in range(t.nodes)]

emitted, path = accept(tree, predictions(tree), remaining=8)
assert emitted == [1, 3, 7] and path == [0, 1, 4]
line = chain(0, [1, 2, 3])                                     # the drafter's first choices only
assert accept(line, predictions(line), remaining=8)[0] == [1, 3]
assert compaction(path, len(emitted)) == [(4, 2)]
assert apply(["r%d" % i for i in range(7)], compaction(path, 3))[:3] == ["r0", "r1", "r4"]
ranks = DraftRanks(positions=3, ranks=2)                       # the chain round above: its first draft accepted
ranks.record(drafter([[], [1], [1, 2]]), [1, 3, 7], matched=1, emitted=2)  # the target's 3: drafter's second choice
assert ranks.counts == [[1, 0, 0], [0, 1, 0], [0, 0, 0]] and ranks.covered(1, 2) == 1.0
```

## Adapting it

Preserve: parents before children, masks equal to the root-to-node paths, predictions
conditioned on exactly those paths, the walk's rule and control-block writes, compaction in
increasing position, and the identity path after commit. Adaptable: `branch`, `beam`,
`depth` and `N` (measure them together with verify cost), the drafter's tree attention, a
static tree shape instead of dynamic selection, fusing selection into the drafter's head,
and a per-round width from recent acceptance (one graph per width).

## Costs and validation

A tree round costs the drafter's `depth` passes over at most `beam` rows, `N` verify
columns instead of `K + 1`, small selection kernels and the row moves of compaction (at most
`depth` rows per array). The kernels are reference speed; `enqueue_expand` makes `branch`
passes over each frontier row.

[draft_tree_tests.cpp](../../tests/inference/draft_tree_tests.cpp) compares beams, kept
nodes and built trees (tokens, parents, masks, depths, padding) with a host mirror on a
synthetic drafter; the walk's tokens, path and control fields on 200 random trees with
budgets and stops, and on chains against the greedy accept kernel; and compaction of
contiguous and slot-mapped rows, including a rerun after a smaller count; and draft rank
histograms over four rounds of chain walks, with tied logits, against a host mirror.
[test_draft_trees.py](../../tests/test_draft_trees.py) checks the reference: kept sets are
ancestor closed, chains walk as greedy speculation does, emitted tokens are always the
target's, compaction places exactly the kept rows, and a draft rank below `w` at the first
position is exactly when a one-level tree of width `w` accepts the target's token. No
end-to-end speedup is claimed.
