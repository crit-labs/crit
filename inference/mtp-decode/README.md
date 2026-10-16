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
alone does not establish a speedup. Sampling needs the probability-based
acceptance in [sampled.h](code/sampled.h), not the greedy verifier.

Use the checkpoint's drafter (MTP head or companion draft model) as published. It
may be re-encoded or given a vocabulary subset, but never trained or fine-tuned:
a drafter trained on text resembling the evaluation raises acceptance there and
nowhere else.

## How it works

`PromptCopyIndex` indexes exact token windows over accepted context, keeping at
most `max_matches` recent occurrences per key (tuple equality resolves hash
collisions). Lookup matches the current suffix and copies the following tokens;
generated-history matches require the longer `history_window`. The newest
full-length candidate wins, otherwise the longest. Stops, vocabulary bounds and
output budgets truncate it; no match returns an empty draft. `append` adds
accepted tokens; `rewind` truncates and rebuilds, restoring evicted occurrences.

For `D` draft tokens, `verify` consumes `D + 1` greedy target predictions.
Prediction `i` must use the committed prefix plus the first `i` draft tokens.
Accept the matching prefix, then the first target correction; if all match, emit
the bonus when budget permits. Stop after an accepted stop token. Predictions
conditioned on rejected tokens are discarded.

A native `GreedySpeculator` round has two phases. `prepare(output_budget,
draft_budget, proposer)` forks the target, verifies the draft on the fork and
returns provisional output and a round ID. `commit(id, accepted)` rewinds the
branch to the accepted prefix, materializes any correction/bonus with `append`,
and only then replaces the committed target. Any prefix may be committed,
including zero, so Core can truncate at a stop. A failed commit consumes the round;
`discard(id)` rejects it. Stale IDs and duplicate commits are errors.

## What the code supplies

| File | Supplies |
| --- | --- |
| [prompt_copy.py](code/prompt_copy.py) | `PromptCopyIndex`, `CopyBatch`, `propose_batch` (charges `requests * (shared_width + 1)` target positions, drafts stay ragged) |
| [implementation.py](code/implementation.py) | `verify`, `CommitPlan`, `commit(plan, apply)` |
| [speculative.h](code/speculative.h) | C++20 `GreedySpeculator`, the `GreedyTargetState` contract and a stateless `PromptCopyProposer(window, lookback)` |
| [prompt_copy.h](code/prompt_copy.h) | C++20 `IndexedPromptCopy`: keeps all occurrences for exact rewind, bounded by `max_tokens`, `max_matches`, `max_draft` |
| [sampled.h](code/sampled.h) | `SampledSpeculator`, `Distribution`, `SparseDistribution`, `AdaptiveDraftBudget` |

## What the engine supplies

Target arithmetic, causal masking and model-state repair. For the native loop,
implement `GreedyTargetState` for the exact target:

| Method | Required behavior |
| --- | --- |
| `fork()` | Distinct owner with identical weights, representation, prefix and next prediction; isolate every mutable component (COW allowed) |
| `tokens()` | Borrow the materialized prefix; no throw or mutation |
| `next_token()` | The ordinary greedy argmax, same tie rule |
| `verify(drafts)` | Materialize the block privately and return `D+1` predictions conditioned on prefix plus `drafts[:i]` |
| `rewind(length)` | Restore all state at that length, including next-token logits and non-KV state |
| `append(token)` | Materialize one correction/bonus token and its next prediction |

The default `verify` is a serial fallback; override it with a causal block pass
and qualify its masks and KV numerically against ordinary execution. The
[native attention fixture](../../tests/inference/speculative_attention_fixture.h)
shows a block implementation over paged CPU attention.

Calls are synchronous and settle physical work before returning. A branch may be
left partially mutated on an exception; the committed parent never. Serialize all
access. A device adaptation needs its own readiness and drain. In a Core `Session`,
return `round.tokens` from `advance`, keep the ID and call `loop.commit(id,
accepted)` from `Session::commit`; Core withholds publication until it succeeds.

For `IndexedPromptCopy`, keep one private index per request and `append` only
published accepted tokens (including corrections), never unverified drafts.
`rewind(length)` follows an already-restored model state; it does not rewind KV.
Window choices are content-dependent: tune on general data, never Eval.

This synchronous loop is the **correctness oracle** for greedy speculation. The
GPU fast path is [device-speculative-round](../device-speculative-round/README.md),
whose accept kernel is tested against it.

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

The [native example](code/speculative_example.cpp) generates 20 tokens from a
synthetic cyclic target with copied drafts and checks each against ordinary
greedy decoding:

```sh
cmake -S core -B build/core -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/core --config Release
ctest --test-dir build/core -C Release -R crit_speculative --output-on-failure
```

## Adapting it

Only the target accepts drafts. Preserve prediction conditioning, causal indexing,
stop rules, accepted-state agreement and output bounds. Window lengths, occurrence
retention, proposer, batch shape and state layout are adaptable. A shorter
proposal saves nothing if the verifier still executes the same rectangle.
[Attention state](../attention-state/README.md) supplies a provisional KV
transaction and [concurrent requests](../concurrent-request-execution/README.md)
supply per-row commit counts.

Greedy speculation reproduces greedy output only if each verifier column gets
exactly the logits of a one-token step; kernels whose summation order depends on
the column count break near ties. See
[column-invariant arithmetic](../device-speculative-round/README.md#column-invariant-arithmetic).

## Costs and validation

Measure `(proposal + verification + repair + publication) / emitted tokens`. For
MoE targets streaming experts from host memory, verification loads the union of
the drafts' experts, so rejected drafts add transfer time. Zero draft budget is
ordinary decoding. Target forks add their real KV/logit/state cost (a shared
partial KV page may need COW space); reserve it in Core, since `SpeculationLimits`
is not a memory cap.

Python fixtures (`tests/test_reusable_techniques.py`, `tests/test_techniques.py`)
cover proposals and verification. [Native fixtures](../../tests/inference/speculative_tests.cpp)
cover multi-round repair, bonus, partial/zero commits, stops, stale rounds,
cancellation, failures and poisoned mutation; the paged-attention fixture compares
2,400 generated tokens with ordinary decoding, and `ctest -R crit_prompt_copy`
checks the native index against a reverse-scan oracle. All use synthetic weights
on the CPU: trained-model quality, device completion and performance are
unqualified. The approach follows
[Leviathan, Kalman and Matias](https://arxiv.org/abs/2211.17192).

## Physical batch size is independent of proposal width

Use the [physical batch planner](../session-scheduling/README.md#independent-physical-batches-and-workspace)
for independent target and draft execution shapes, keeping
`SpeculationLimits.max_draft` and the per-round draft budget unchanged. With a
target cap of 16 and a draft override of 4, a 23-column draft prompt runs as five
calls of 4 plus 3; the target still uses 16 plus 7. An override of zero inherits
the target cap. Sequential autoregressive draft steps keep their dependency.

Split only operations the adapter declares safe, deciding separately for
ingestion, catch-up, verification and repair. Causal verification must keep D+1
predictions in order across calls and carry KV, recurrent state and positions;
padding must not affect state or random draws.

[physical_batches_example.cpp](code/physical_batches_example.cpp) runs separate
target/draft scratch with synthetic integer projections and recurrent state. At
context 96, lowering the draft cap from 16 to 4 cuts its scratch from **1,024 to
256 bytes** with identical outputs (toy counts, not an LLM saving), and records
the choice with [recorded selection](../recorded-selection/README.md). Rebuild
workspace and graphs at a settled boundary when caps change, and discard a
partially advanced branch on failure.

```sh
cmake -S core -B build/physical-batch-demo -DBUILD_TESTING=ON
cmake --build build/physical-batch-demo --config Release --target crit_physical_speculative_example
ctest --test-dir build/physical-batch-demo -C Release --output-on-failure -R crit_physical_speculative_example
```

It checks 48-token greedy equivalence across draft caps 1–17. Qualify peak memory,
round latency and quality on the real engine; smaller calls can cost more launches
and weight reads.

## Sampled target and draft models

`SampledSpeculator` owns two `ProbabilityState` models (same tokenizer and
positions), private RNG and one provisional round. At position i, draw x from the
draft q_i and accept with probability `min(1, p_i[x]/q_i[x])`; on rejection sample
the correction from normalized `max(p_i - q_i, 0)`; after all accepts, sample a
bonus. This preserves the target distribution, not the seeded token sequence.
Discard or failure consumes no committed random draws.

`Distribution::logits(scores, temperature, top_k, top_p, min_p)` applies
temperature, top-k, top-p, then min-p, ranking ties by ascending token ID;
temperature zero selects the first maximum. The target p must match ordinary
sampling; the draft may use other filters if proposal and acceptance both use its
actual q.

`SparseDistribution` stores sorted positive `{token, weight}` pairs. If support
exceeds `sparse_capacity` it keeps the **complete dense law**: capacity limits
representation, never probability support. `SparseSampledSpeculator` shares the
same contract. Sparse rows cost O(S) storage; fallback costs O(V).

`AdaptiveDraftBudget(supported, maximum_pending=64)` explores the given widths
(include zero) and minimizes an EWMA of round seconds per published token. For
delayed timings, take a ticket from `begin_observation(actual_width)` and call
`observe(ticket, published, seconds)` after completion. `record_preparation(width,
seconds)` resets a width after graph rebuilds.

The [sampled example](code/sampled_example.cpp) and
[native checks](../../tests/inference/sampled_tests.cpp) (`ctest -R crit_sampled`)
cover exhaustive small-support path mass, filters, sparse/dense coupling,
transaction rollback and budget tickets, without a model or GPU.
