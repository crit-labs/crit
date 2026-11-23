# Copy proposals and complete greedy speculative execution

Propose draft tokens by copying continuations of repeated token windows, verify
them against the target with greedy acceptance, and repair state so only accepted
tokens are published. The entry also supplies a native greedy loop, a
distribution-preserving sampled loop and adaptive draft budgets. The legacy
`mtp-decode` ID is kept for compatibility; there are no learned multi-token
predictor heads or trained draft models here.

## When to use it

Use prompt/history copying for repeated or edited text when it proposes useful
continuations cheaply. Any draft source can use the greedy verifier. Compare
proposal, verification and repair cost with ordinary decoding; acceptance rate
alone does not establish a speedup. Sampling needs probability-based
acceptance, not the greedy verifier: [token-sampling](../../sampling/token-sampling/README.md)'s
verify predictions, on the device.

Use the checkpoint's drafter (MTP head or companion draft model) as published. It
may be re-encoded or given a vocabulary subset, but never trained or fine-tuned:
a drafter trained on text resembling the evaluation raises acceptance there and
nowhere else.

A drafter with a single decoder layer (an MTP head, a one-layer EAGLE drafter) computes
a position's keys and values from that position's own input alone: its fused input,
the layer's input norm and the key/value projections, never the attention. Prompt
ingestion for such a drafter therefore needs only those steps for every prompt
position, not the attention or the MLP, and its cost does not grow with the context.
Keep the full layer for the rows whose output a draft reads.

## How it works

`PromptCopyIndex` indexes exact token windows over accepted context, keeping at
most `max_matches` recent occurrences per key (tuple equality resolves hash
collisions). Lookup matches the current suffix and copies the following tokens;
generated-history matches require the longer `history_window`. The newest
full-length candidate wins, otherwise the longest. Stops, vocabulary bounds and
output budgets truncate it; no match returns an empty draft. `append` adds
accepted tokens; `rewind` truncates and rebuilds, restoring evicted occurrences.
`propose_matched(..., longest=64)` returns the
same draft and how many accepted tokens before the copied continuation equal the
context's last ones: at least `window`, at most `longest`, 0 without a draft. A long
match is the evidence for letting a copy override another drafter's tokens
([copied drafts](../device-speculative-round/README.md#copied-drafts)).

For `D` draft tokens, `verify` consumes `D + 1` greedy target predictions.
Prediction `i` must use the committed prefix plus the first `i` draft tokens.
Accept the matching prefix, then the first target correction; if all match, emit
the bonus when budget permits. Stop after an accepted stop token. Predictions
conditioned on rejected tokens are discarded.

## On the device inside captured rounds

The plugin's device form (`draft:device`, [device_drafts.py](code/device_drafts.py)) keeps each round slot's tokens
(prompt and outputs, the pending token last) in a ring on the device and computes `PromptCopyIndex.propose_matched`'s
copy for every row at once in torch: the longest continuation of the context's last `window` tokens among their 64 most
recent occurrences (outputs needing `history_window` matching tokens), the most recent of equally long ones, and how
far back the match goes, over the ring's last `capacity - (widest + 1)` positions (a round's commit writes up to
`widest + 1` positions past its frontier, and fewer again after a truncation, never into the searched span). A slot's history starts at `seed(slot, tokens)` (the prompt and the first output token),
which the model calls after the round driver binds the slot. `commit_round` writes the round's emitted tokens into the ring from the device's counts, so
copies need no host between rounds (Core's native rounds). The copies equal the host index's wherever the sequence
fits that span (`capacity`, default 16,384 tokens). Without `override` (0, the default) the plugin proposes copies alone and never asks the model's draft head: a
checkpoint's draft layer is not used at all, so a model plugin that loads one refuses this selection
(`uses_head` is False). For the draft layer's chain alone select `greedy-head-drafter`. With `override` the model's draft head drafts the chain
(`head_chain`) and a copy matching at least `override` tokens replaces its first drafts. The override is a bet that
the copy is right where the head is not: a short match occurs by chance in a long context (a few common tokens recur
throughout code, markup and prose), and the copy then replaces drafts the head had right, so a round that would have
accepted its whole chain accepts nothing. Measure the threshold, never guess it: serve the engine with the override
off and at a few lengths on prompts long enough to repeat themselves (`crit run ... ablate --data EVAL_DATA --set
mtp-decode.override=N`, whose Eval cases include long tool-call and agent prompts) and keep the one that pays there
and on the held-out prompts. A copied draft is drawn from no
law: in sampled rounds its column's law is cleared and the verify takes the target's own draw there.
[device_verify.py](code/device_verify.py) is this entry's verify for every row of a round (chain-verifier's
`verify:device`): it writes each row's emitted tokens and counts, and the round driver advances its own state.

## What the code supplies

| File | Supplies |
| --- | --- |
| [prompt_copy.py](code/prompt_copy.py) | `PromptCopyIndex`, `CopyBatch`, `propose_batch` (charges `requests * (shared_width + 1)` target positions, drafts stay ragged) |
| [implementation.py](code/implementation.py) | `verify`, `CommitPlan`, `commit(plan, apply)` |

## What the engine supplies

Target arithmetic, causal masking and model-state repair: `D + 1` predictions, each conditioned on the committed
prefix plus the drafts before it, and the state at the accepted length restored after a round, next-token logits
and non-KV state included.

Keep one private `PromptCopyIndex` per request and `append` only published accepted tokens (including
corrections), never unverified drafts. `rewind(length)` follows an already-restored model state; it does not rewind
KV. Window choices are content-dependent: tune on general data, never Eval.

`verify` is the **correctness oracle** for greedy speculation. The GPU fast path is
[device-speculative-round](../device-speculative-round/README.md).

## Worked example

Run from `code/`. A copied continuation still loses to the target at a mismatch.

```python
from prompt_copy import PromptCopyIndex, propose_batch
from implementation import verify, commit

index = PromptCopyIndex((1, 2, 3, 4, 1, 2), window=2, history_window=2)
batch = propose_batch((index,), row_budget=4)
assert batch.proposals == ((3, 4, 1),)
assert batch.verification_rows == 4
plan = verify(batch.proposals[0], (3, 9, 0, 0), remaining=3)
assert plan.tokens == (3, 9) and plan.accepted_drafts == 1
assert commit(plan, lambda tokens, _: index.append(tokens)) == (3, 9)
index.rewind(6)
assert index.propose(2) == (3, 4)
assert index.propose(2, stop_tokens=(3,)) == (3,)
assert verify((3, 4), (3, 4, 5), remaining=3).tokens == (3, 4, 5)
```

## Adapting it

Only the target accepts drafts. Preserve prediction conditioning, causal indexing,
stop rules, accepted-state agreement and output bounds. Window lengths, occurrence
retention, proposer, batch shape and state layout are adaptable. A shorter
proposal saves nothing if the verifier still executes the same rectangle.
[Attention state](../../attention/attention-state/README.md) supplies a provisional KV
transaction and Core's [runtime](../../../core/src/runtime/README.md) (concurrent requests)
supplies per-row commit counts.

Greedy speculation reproduces greedy output only if each verifier column gets
exactly the logits of a one-token step; kernels whose summation order depends on
the column count break near ties. See
[column-invariant arithmetic](../device-speculative-round/README.md#column-invariant-arithmetic).

## Costs and validation

Measure `(proposal + verification + repair + publication) / emitted tokens`. For
MoE targets streaming experts from host memory, verification loads the union of
the drafts' experts, so rejected drafts add transfer time. Zero draft budget is
ordinary decoding. Target forks add their real KV/logit/state cost (a shared
partial KV page may need COW space).

Python fixtures (`tests/test_reusable_techniques.py`, `tests/test_techniques.py`)
cover proposals and verification on synthetic targets: trained-model quality, device
completion and performance are unqualified. The approach follows
[Leviathan, Kalman and Matias](https://arxiv.org/abs/2211.17192).

## Sampled requests

A sampled request's drafts are verified by rejection sampling: accept draft `x` at position `i` with probability
`min(1, p_i[x] / q_i[x])`, on rejection draw the correction from the normalized `max(p_i - q_i, 0)`, and after
all accepts draw a bonus. This preserves the target distribution, not the seeded token sequence.
[token-sampling](../../sampling/token-sampling/README.md) supplies the predictions under which the greedy accept rule is this
rule, on the device and in its Python mirror; [device-speculative-round](../device-speculative-round/README.md)'s
`WidthPolicy` chooses the draft width per round from measured acceptance, and
[hidden-state-drafters](../hidden-state-drafters/README.md)' confidence cuts shorten a draft by its drafter's own
confidence.
