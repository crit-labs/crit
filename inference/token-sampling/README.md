# Select tokens on the device: Core's whole sampling law and speculative verification

Device-side token selection for logits produced on the GPU, without copying a
vocabulary-sized row to the host per token. The device runs Core's sampling law over the
full vocabulary (bias, min_tokens suppression, repetition, frequency and presence
penalties, temperature, top-k, top-p, min-p and Core's draw), and verifies speculative
drafts by rejection sampling, so sampled requests can draft inside a captured decode
graph and still emit samples of the model's own distribution, with Core's
log-probabilities when a request asks for them. This is the device path of Core's sampler
([Core sampling](../../core/src/sampling/README.md)).

## When to use it

When generation samples inside the decode step (after or inside a captured graph), when
sampled requests should speculate, or when a structured-output mask applies to device
logits. Grammar masks ([Core's structured output](../../core/src/constraints/README.md))
go through `enqueue_mask` before selection. Keep the plain argmax path for greedy requests
where bit identity with an existing greedy kernel matters.

## How it works

**Rank order and the law.** Candidates rank by (value descending, id ascending). Values
are the logits after Core's steps 2 to 5: `+bias` (a -inf bias bans), suppression of the
`suppress_ids` while `position + column < min_tokens` (skipped on rows whose `Rows::yield` is set:
Core's `suppression_yields`, a constraint mask that allows only suppressed tokens), repetition (`v / r` for positive,
`v * r` for negative values) for tokens in the prompt or the outputs, then
`v -= frequency * n + presence` for tokens with `n > 0` outputs. The rules are Core's doubles
(0.8 or 0.3 is not a float, and its float rounding would move a top-p boundary, or a draw near
one, by ~1e-8) and the steps run in IEEE double as Core runs them, without contraction;
the full law then rounds each value to the nearest float, clamped to the finite range, so a value
Core holds beyond the float range stays selectable. Column `c` of a verify or
draft step counts its staged tokens (the drafts before it) as outputs, so each column has
the law Core's `Sampler` would use with those tokens staged. Weights are
`exp((v - max) / temperature)`; top-k keeps a rank prefix, top-p the shortest rank prefix
whose weight reaches `top_p` of the top-k weight, min-p the weights `>= min_p`. The draw
`u = uniform(seed, position + column, stream, purpose)` picks the first kept token whose
cumulative weight exceeds `u` times the kept weight, Core's step 12.

**The fast path: Core's law on merged candidates.** Greedy rows and `1 <= top_k <= 64`
need only the row's best `top_k + 1` tokens. Stage blocks of 4,096 logits each
(`ceil(vocab / 4096)` per row, so the split follows the vocabulary) process their logits
(Core's steps 2 to 5 in double) and keep their best `top_k + 1` as packed keys (the value
rounded down to a float, an inexact bit, the id), with an exact block top-K and no
atomics. One law block per row merges the lists, recomputes the candidates' values in
double and runs Core's steps 7 to 12 on them in FP64 and in Core's summation order: rank
order after top-k, id order when the row has at most `top_k` selectable tokens (the
`(top_k + 1)`-th candidate tells which), the draw and the verify rule included. Tokens and
laws are Core's up to the last bit of `exp`. Two distinct doubles that round down to one
float share their key up to the id, so their key order is by id; when such a pair straddles
the last needed candidate the row goes to the full law instead. Open rows (temperature
above 0 with top-k 0, or at least the vocabulary) take the same kernels with the best 65
candidates and the whole row's weight (FP32 exps, FP64 sums: Core's total to about 1e-7):
they are settled there when top-p or min-p closes the kept set within the top 64, or, with
no filter at all, when the draw lands among them. Every other row (`64 < top_k <` the
vocabulary at once, open rows whose kept set or draw reaches further) is marked unsettled,
and the full-law kernels, launched for every row, take exactly those; the rest exit at
once. So one captured launch serves every request, and each row's path is decided on the
device.

**The full law: prefix searches without sorting.** One block of 1,024 threads owns a row.
Values map to unsigned keys that order like the floats, so every rank prefix is "key above
a boundary, or equal key and id at most the boundary id". The top-k boundary (a count),
the top-p boundary and the draw (weights) are all the same search: walk the key's eight
4-bit digits from the top, summing the candidates' weights per digit, keep the digit where
the cumulative crosses the target, then resolve the final key's ties in id order with a
block-wide scan. Each sum is per-thread FP32 over a fixed stride, then FP64 over threads
in a fixed order, so a row's result never depends on timing or on the other rows of the
launch; `top_k = 0` is the whole vocabulary, as in Core.

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
- *Sampled drafts* (drawn from the drafter's processed law `q`): column `c < K`
  accepts with `u_accept * q(d) < p(d)` and writes `d`, otherwise a draw from
  `max(p - q, 0)` with `u_residual`; column `K` writes the bonus with `u_sample`, the round
  of [Leviathan, Kalman and Matias](https://arxiv.org/abs/2211.17192). It accepts more often
  at high temperature (`1 - TV(p, q)` instead of `p(argmax q)`). `q` is a dense row or a
  `SparseLaw`, the kept (token, probability) pairs a fast-path draft row writes in place of
  a vocabulary-sized row; a draft whose law has no sparse form (count -1: a full-law row)
  falls back to its dense row, or, without one, to the target's own draw, the greedy-draft
  rule, which keeps the law at a lower acceptance.
- *Cut drafts* (a draft id outside the vocabulary, such as the -1 a drafter's
  [confidence cut](../hidden-state-drafters/README.md) writes): the column is the target's own
  draw with `u_sample`, like the bonus column, whatever its `q` row still holds. No draft was
  drawn from that `q`, so the rejection rule's residual would be the wrong law; a round cut after
  `j` drafts emits what a width-`j` round would.

**Block verification.** With `Verify::rule = VerifyRule::block`, a kernel after the
token rule's predictions decides the round as a block, the rule of mtp-decode's
`block_verify` ([Sun et al.](https://arxiv.org/abs/2403.10444)) over each column's kept
law: from the drafts' cumulative ratios `w_i` and each column's residual mass `Z_i`,
draft count `i` stands with probability `h_D = w_D` or `Z_i / (Z_i + 1 - w_i)`, scanned
from the round's width `D` down; the predictions become the standing drafts, then the
bonus (the token rule's column `D` prediction) or a draw from `max(w_tau p - q, 0)` without
the rejected draft, so the accept kernel needs no change. A later draft can stand where an
earlier one alone would have been rejected, which raises the standing count at the same
law. Block decisions use every column's draws, so a position a later round revisits needs
fresh draws: the acceptance draws use purpose `block` at position `(round position << 7) | i`,
and the drafter samples with `block_draft_purpose(round position)`, never `draft`. A request
whose columns hold a full-vocabulary law (no sparse kept law), or temperature 0, keeps the
token rule's predictions. One thread per request computes the rule (`O(D x kept)` with the
kept laws of at most 64 tokens), and rounds are at most 64 columns wide.

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

**Log-probabilities.** Core reports each generated token's log-probability and its ranked
alternatives over the raw row (`TokenLogprob`: `(x - max) - log(sum exp(x - max))` in double,
alternatives by float log-probability descending, then id). `enqueue_logprobs` computes them
for any raw rows and chosen tokens with the same stage blocks: each keeps its raw maximum, its
`exp` sum (FP32 exps, compensated, FP64 across threads and blocks in a fixed order) and its best
`n + 1` raw keys; one merge block per row normalizes and ranks them. Values agree with Core's
double accumulation to about 1e-6. When the `(n + 1)`-th candidate's log-probability equals the
`n`-th answer's (distinct logits collapsing to one float), a scan of the row in id order
resolves the tie group exactly as Core ranks it. `enqueue_sample` and `enqueue_verify` fuse
the same work (`Logprobs`): the stage blocks also write the raw lists, extra blocks of the law
kernel rank them beside the law blocks, and the full-law kernel writes every row's chosen
token's log-probability once the token is known. Rows masked by `enqueue_mask` report over the
masked row; Core's constrained `Sampler` reports over the unmasked one.

**History on the device.** `Request.counts` holds the committed output counts and
`prompt_seen` the prompt tokens the repetition penalty counts. `enqueue_commit` adds a
round's emitted tokens and advances `position`; in a device round its first Forward
kernel ingests the previous round's packed result and `control[commit_tokens]`, the result
tokens the history has still to take in (the emitted count after any Core truncation, 0
after a seed round whose token the engine already committed), before a law of the new round
is computed.

## What the code supplies

[sampling.h](code/sampling.h) / [sampling.cu](code/sampling.cu), linking `crit::core`:

| Interface | Supplies |
| --- | --- |
| `Params`, `enqueue_select`, `select_scratch_bytes` | One selection per strided row with its own rules at a device-advanced position (no processing): the fast path or the full law |
| `enqueue_mask` | A structured-output bitmask (`TokenMatcher::allowed_bits`) applied to logit rows as -inf |
| `Request` | Device-resident resolved sampling, position, bias, suppression and penalty history |
| `Rows`, `enqueue_sample`, `scratch_floats` | One draw per (request, column or tree node) with a chosen `DrawPurpose`, optionally the law as a dense row or a `SparseLaw` (a drafter's `q`) |
| `Verify`, `enqueue_verify` | Verify predictions for the device greedy accept kernel: the target's draws for greedy drafts, accept/residual/bonus for sampled drafts against dense or sparse `q`; `VerifyRule::block` for block verification |
| `Logprobs`, `enqueue_logprobs`, `logprob_scratch_bytes` | Core's `TokenLogprob` values on the device: chosen tokens' log-probabilities and ranked alternatives of raw rows, standalone or fused into `enqueue_sample` / `enqueue_verify` (a device round's `RoundView::logprob` rows) |
| `enqueue_commit` | Counts and positions after publication, from a device-held emitted count |

[sampling.py](code/sampling.py) mirrors the processing, law, draw and verification in FP64
with Core's bit-identical draw. The host oracle is `crit::core::Sampler`.

## What the engine supplies

One `Request` per sequence, filled from `crit::core::resolve_sampling` (merge duplicate
bias ids, sort them), with `position` set to the committed output count and `counts` /
`prompt_seen` built from `TokenHistory` at request start. Scratch of
`scratch_floats(vocab, rows)` floats (or `select_scratch_bytes(vocab, rows)` bytes for
`enqueue_select`), 8-byte aligned, target logits as FP32 rows (the verify step writes
logits instead of argmax for sampled requests), and for sampled drafts the dense `q` rows
or sparse laws from `enqueue_sample(..., DrawPurpose::draft, ...)`. Report supported options through
`ModelInfo::sampling`. A device round for sampled requests:

```text
Forward: enqueue_commit(previous round's result tokens, control[commit_tokens])
         draft (greedy argmax, or enqueue_sample with DrawPurpose::draft and q rows or sparse laws)
         verify writing K + 1 target logit rows, then enqueue_verify -> predictions
                (with Verify::logprobs -> the round's logprob rows, when RoundConfig::logprobs)
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
assert s.verify([row, row], [-1], 0.7, seed=4, draft_laws=[q])[0] == s.select(row, 0.7, seed=4)  # cut: p alone
assert s.verify([row, row], [1], 0.0) == [0, 0]   # temperature 0: the greedy rule
```

## Adapting it

Preserve the rank order and tie-break, Core's processing order, one draw per emitted
token keyed by its output position and purpose, the column's staged drafts in its
penalties, and the predictions contract the accept kernel reads; on the fast path also
Core's summation orders and the hand-off of every row whose candidates cannot be shown
exact. Block sizes, the stage chunk, the digit width of the prefix search, a fused head and
selection, and the fast-path cap are adaptable.

## Costs and validation

The fast path reads each logit once, spread over the whole device, plus one small law
block per row. The full law reads its row about 10 times per prefix search (8 digit
passes, a sum and a tie scan); a sampled column with top-k, top-p and a draw runs about
three searches, a rejected verify column one more, all on one block per row. As an
illustration, on an RTX 5090 with one request's 248,077-token FP32 rows (CUDA graph
replays): a top-k 20 / top-p 0.95 draw took ~15 us on the fast path against ~3.5 ms on the
full law, a greedy draw ~7 us against ~0.28 ms, and a K = 8 verify ~20 us against ~4.6 ms;
open rows the fast kernels cannot settle (a flat row at top-p 0.95) still take the full
law's milliseconds. Measure the law against the round time for the rules your requests use.
The full law's FP32 values and weights can pick a neighbouring token when a draw lands
within rounding of a cumulative boundary (the open rows' whole-row weight within ~1e-7),
and NaN or +inf logits are not filtered (Core rejects them). Log-probabilities fused into a
draw added ~4 us on the same device and rows (one row, 20 alternatives: ~20 us against ~16
us without; `enqueue_logprobs` after the draw ~11 us).

[token_sampling_tests.cpp](../../tests/inference/token_sampling_tests.cpp) checks the
selection path (ties, repeatability, Core's exact draws on the fast path and away from
boundaries elsewhere, strided rows with their own rules in one launch, frequencies,
masks) and the log-probabilities against `log_softmax` and `top_logprobs` on seven row
shapes (moderate, peaked, flat, ties, masked, large and negative logits) with `n` up to 64,
rows whose boundary candidates collapse to one float, rows independent of their launch, and
the fused values equal to `enqueue_logprobs` bit for bit.
[token_law_tests.cpp](../../tests/inference/token_law_tests.cpp) checks the law against
`Sampler::distribution` with bias, suppression, penalties over staged columns, top-k,
top-p and min-p; Core's double rules at exact top-p boundaries and values beyond the float
range; the fast path's draws equal to `Sampler::sample` on vocabularies of 7, 100, 12,288
and 200,003 tokens; open rows settled by the fast kernels or the full law; a group of
inexact keys at the boundary handed to the full law; every path side by side in one
launch; draws beyond rank 64 at `top_k = 0`; every verify column against the rejection rule
with Core's draws; cut columns (-1 drafts with non-zero `q` rows, or an empty sparse law)
equal to the target's own draw; sparse laws equal to the dense rows, and verify predictions
against them equal to those against the dense rows on fast, open and full-law targets;
first-token frequencies equal to the target law for greedy drafts and for sampled drafts
with dense or sparse laws over 16,384 seeds; and device commits.
[test_token_law.py](../../tests/test_token_law.py) checks the mirror. The timings above
illustrate one device; there is no output-quality claim.
