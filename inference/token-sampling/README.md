# Select tokens on the device: Core's whole sampling law and speculative verification

Device-side token selection for logits produced on the GPU, without copying a
vocabulary-sized row to the host per token. The device runs Core's sampling law over the
full vocabulary (bias, min_tokens suppression, repetition, frequency and presence
penalties, temperature, top-k, top-p, min-p and Core's draw), and verifies speculative
drafts by rejection sampling, so sampled requests can draft inside a captured decode
graph and still emit samples of the model's own distribution. This is the device path of
Core's sampler ([Core sampling](../../core/src/sampling/README.md)).

## When to use it

When generation samples inside the decode step (after or inside a captured graph), when
sampled requests should speculate, or when a structured-output mask applies to device
logits. Grammar masks ([Core's structured output](../../core/src/constraints/README.md))
go through `enqueue_mask` before selection. Keep the plain argmax path for greedy requests
where bit identity with an existing greedy kernel matters.

## How it works

**Rank order and the law.** Candidates rank by (value descending, id ascending). Values
are the logits after Core's steps 2 to 5: `+bias` (a -inf bias bans), suppression of the
`suppress_ids` while `position + column < min_tokens`, repetition (`v / r` for positive,
`v * r` for negative values) for tokens in the prompt or the outputs, then
`v -= frequency * n + presence` for tokens with `n > 0` outputs. Column `c` of a verify or
draft step counts its staged tokens (the drafts before it) as outputs, so each column has
the law Core's `Sampler` would use with those tokens staged. Weights are
`exp((v - max) / temperature)`; top-k keeps a rank prefix, top-p the shortest rank prefix
whose weight reaches `top_p` of the top-k weight, min-p the weights `>= min_p`. The draw
`u = uniform(seed, position + column, stream, purpose)` picks the first kept token whose
cumulative weight exceeds `u` times the kept weight, Core's step 12.

**Prefix searches without sorting.** One block of 1,024 threads owns a row. Values map to
unsigned keys that order like the floats, so every rank prefix is "key above a boundary,
or equal key and id at most the boundary id". The top-k boundary (a count), the top-p
boundary and the draw (weights) are all the same search: walk the key's eight 4-bit
digits from the top, summing the candidates' weights per digit, keep the digit where the
cumulative crosses the target, then resolve the final key's ties in id order with a
block-wide scan. Each sum is per-thread FP32 over a fixed stride, then FP64 over threads
in a fixed order, so a row's result never depends on timing or on the other rows of the
launch. `enqueue_select` keeps its two-stage top-64 path for `1 <= top_k <= 64` and runs
the full law otherwise; `top_k = 0` is the whole vocabulary, as in Core.

**Speculative verification.** `enqueue_verify` computes the law of all `K + 1` verify
columns in parallel (one block per column and request) and writes predictions for the
[device greedy accept kernel](../device-speculative-round/README.md), which emits the
drafts up to the first column whose prediction differs, then that prediction:

- *Greedy drafts* (deterministic, `q` one-hot): every column writes the target's own draw
  `u_sample`. A draft stands exactly when it equals that draw, with probability `p(d)`,
  the acceptance of the rejection rule for a one-hot `q`, and a rejected column's draw is
  a sample of `p` without `d`, its residual. The emitted text is therefore what plain
  sampled decoding emits with the same seed, token for token, whenever verify computes
  each column's logits as a one-token step does
  ([column-invariant arithmetic](../device-speculative-round/README.md#column-invariant-arithmetic)).
- *Sampled drafts* (drawn from the drafter's processed law `q`, dense rows): column `c < K`
  accepts with `u_accept * q(d) < p(d)` and writes `d`, otherwise a draw from
  `max(p - q, 0)` with `u_residual`; column `K` writes the bonus with `u_sample`, the round
  of [Leviathan, Kalman and Matias](https://arxiv.org/abs/2211.17192). It accepts more often
  at high temperature (`1 - TV(p, q)` instead of `p(argmax q)`).

**Token trees.** With `Rows::tree`, the columns are a tree's nodes: node `c` stages the
tokens on its root-to-node path and draws at its depth's position. With greedy (top-k)
children, verifying a tree is `enqueue_sample` over the nodes followed by a walk that
moves to the child equal to each node's draw ([draft trees](../draft-trees/README.md)),
so a tree keeps the same law, and the same text as plain sampling, as a chain.

**Why counter-based draws keep the law.** The token finally emitted at output position
`n` uses only the draws keyed by `n` (`sample`, or `accept` and `residual` for sampled
drafts, and the drafter's `draft`), and whether a later column's tentative prediction is
kept never depends on those draws. A rejected round's leftover columns are recomputed at the same positions in
the next round with fresh conditioning, so the emitted text is a sample of the target law
for any drafter, any draft width and any stopping rule, and the same seed repeats.
Temperature 0 makes every prediction the column's argmax: the greedy rule.

**History on the device.** `Request.counts` holds the committed output counts and
`prompt_seen` the prompt tokens the repetition penalty counts. `enqueue_commit` adds a
round's emitted tokens and advances `position`; in a device round its first Forward
kernel ingests the previous round's packed result and `control[n_emitted]`, after any Core
truncation, before a law of the new round is computed.

## What the code supplies

[sampling.h](code/sampling.h) / [sampling.cu](code/sampling.cu), linking `crit::core`:

| Interface | Supplies |
| --- | --- |
| `Params`, `enqueue_select` | One selection per call at a device-advanced position: two-stage top-64 path or the full law |
| `enqueue_mask` | A structured-output bitmask (`TokenMatcher::allowed_bits`) applied to logit rows as -inf |
| `Request` | Device-resident resolved sampling, position, bias, suppression and penalty history |
| `Rows`, `enqueue_sample` | One draw per (request, column or tree node) with a chosen `DrawPurpose`, optionally the dense law (a drafter's `q`) |
| `Verify`, `enqueue_verify` | Verify predictions for the device greedy accept kernel: the target's draws for greedy drafts, accept/residual/bonus for sampled drafts |
| `enqueue_commit` | Counts and positions after publication, from a device-held emitted count |

[sampling.py](code/sampling.py) mirrors the processing, law, draw and verification in FP64
with Core's bit-identical draw. The host oracle is `crit::core::Sampler`.

## What the engine supplies

One `Request` per sequence, filled from `crit::core::resolve_sampling` (merge duplicate
bias ids, sort them), with `position` set to the committed output count and `counts` /
`prompt_seen` built from `TokenHistory` at request start. Scratch of
`scratch_floats(vocab, rows)` floats, target logits as FP32 rows (the verify step writes
logits instead of argmax for sampled requests), and for sampled drafts the dense `q` rows
from `enqueue_sample(..., DrawPurpose::draft, ...)`. Report supported options through
`ModelInfo::sampling`. A device round for sampled requests:

```text
Forward: enqueue_commit(previous round's result tokens, control[n_emitted])
         draft (greedy argmax, or enqueue_sample with DrawPurpose::draft and q rows)
         verify writing K + 1 target logit rows, then enqueue_verify -> predictions
Finish:  enqueue_greedy_accept, commit_state       (device-speculative-round, unchanged)
```

## Worked example

From `code/`:

```python
import sampling as s

row = [2.0, 1.0, 2.0, -1.0, 0.5]
assert s.select(row) == 0                  # tie between 0 and 2, lowest index wins
assert [t for t, _ in s.nucleus(row, 0.7, top_p=0.9)] == [0, 2, 1]   # 0.446, 0.446, 0.107
draws = [s.select(row, 0.7, 0.9, 0, seed=1, position=c) for c in range(8)]
assert draws == [2, 2, 0, 2, 0, 0, 0, 2]   # crit::core::Sampler draws the same tokens

flat = [0.0] * 100                         # top_k 0 keeps the whole vocabulary, not 64 candidates
assert len(s.nucleus(flat, 1.0)) == 100
penalized = s.process(row, frequency=0.5, presence=0.5, counts={0: 1}, staged=(2,))
assert penalized[:3] == [1.0, 1.0, 1.0]    # 2 - 0.5 - 0.5 for both repeated tokens

# Greedy drafts: each column is the target's own draw, so a draft stands exactly when it equals it and the
# round emits what plain sampling would. Sampled drafts: rejection sampling against the draft's law q.
assert s.verify([row, row], [1], 0.7, seed=4) == [s.select(row, 0.7, seed=4, position=c) for c in (0, 1)]
q = {1: 0.6, 3: 0.4}
out = s.verify([row, row], [1], 0.7, seed=4, draft_laws=[q])
assert (out[0] == 1) == (s.uniform(4, 0, 0, s.ACCEPT) * q[1] < dict(s.nucleus(row, 0.7))[1])
assert s.verify([row, row], [1], 0.0) == [0, 0]   # temperature 0: the greedy rule
```

## Adapting it

Preserve the rank order and tie-break, Core's processing order, one draw per emitted
token keyed by its output position and purpose, the column's staged drafts in its
penalties, and the predictions contract the accept kernel reads. Block size, the digit
width of the prefix search, splitting a row over several blocks, a fused head and
selection, and the fast-path cap are adaptable.

## Costs and validation

The full law reads its row about 10 times per prefix search (8 digit passes, a sum and a
tie scan); a sampled column with top-k, top-p and a draw runs about three searches, a
rejected verify column one more. On one block per row this is reference speed, not tuned;
measure it against the round time before relying on it, and split rows over blocks when it
shows. FP32 values and weights can pick a neighbouring token when a draw lands within
rounding of a cumulative boundary, and NaN or +inf logits are not filtered (Core rejects
them).

[token_sampling_tests.cpp](../../tests/inference/token_sampling_tests.cpp) checks the
selection path (ties, repeatability, Core's draws away from boundaries, frequencies,
masks). [token_law_tests.cpp](../../tests/inference/token_law_tests.cpp) checks the full
law against `Sampler::distribution` with bias, suppression, penalties over staged
columns, top-k, top-p and min-p; draws beyond rank 64 at `top_k = 0`; every verify column
against the rejection rule with Core's draws; first-token frequencies equal to the target
law for greedy and sampled drafts over 16,384 seeds; and device commits.
[test_token_law.py](../../tests/test_token_law.py) checks the mirror. No throughput or
output-quality claim.
