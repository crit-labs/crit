# Run a whole speculative round on the device

One Core `Session::advance()` runs a complete speculative round on the GPU (draft
K tokens, verify K+1 columns, accept, select the accepted state, advance the
frontier); the host waits once, on one `readback_async`. Greedy requests compare
drafts with the target's argmax; sampled requests verify by rejection sampling
through [token-sampling](../token-sampling/README.md), which writes predictions the
same accept kernel consumes, so both decode modes share one round.

## When to use it

Use it once a synchronous speculative loop is correct and the remaining cost is
host round trips: per-token readbacks, host acceptance and bookkeeping, or many
small launches. Keep the synchronous loops in [mtp-decode](../mtp-decode/README.md)
(`GreedySpeculator`, `SampledSpeculator`) as the correctness oracles.

It does not help when verifying K+1 columns costs about K+1 ordinary steps.

## How it works

The device owns an int32 control block per row (`frontier`, pending
`last_token`, round-start `base_frontier`/`base_last`, `n_emitted`, `n_matched`,
`output_budget`, and `commit_tokens`, the result tokens the sampling history has
still to take in); the host only mirrors counts it reads back. A round is two
captured graphs and one copy:

1. **Forward**: copy the round allowance from pinned memory into
   `control[output_budget]` (with `RoundConfig::zero_copy`, `enqueue_set_budget`
   reads it in place), run the engine's `draft` step (writes `drafts[K]`
   from `last_token`), then `verify`. Verify feeds `last_token, d1..dK` at
   positions `frontier..frontier+K`, writes `predictions[i]` (the target's greedy
   argmax after `last, d1..di`) and retains whatever provisional state selecting
   any accepted prefix needs: KV columns, recurrent checkpoints or saved inputs.
2. **Finish**: the device greedy-accept kernel, `enqueue_pack` (the readback
   block), then the engine's `commit_state`.
3. **Readback**: one `readback_async` of the block into pinned memory: the packed
   result `{n_emitted, tokens[K+1]}`, `n_matched` and the frontier after the round,
   then optionally each column's log-probability and ranked alternatives (columns
   past `n_emitted` hold -1 and -inf). Its completion event is `Progress::readiness`.
   The host checks the frontier against its mirror plus the count, reads the
   accepted drafts for acceptance statistics, and hands Core the log-probabilities
   without a second copy. With `RoundConfig::zero_copy` the pack kernel writes the
   block straight into mapped pinned memory and the completion event follows
   Finish: a round has no copy at all.

Acceptance is the host `GreedySpeculator` rule. With remaining allowance `R`:

```text
out = 0, matched = 0
while matched < K and out < R and drafts[matched] == predictions[matched]:
    emit drafts[matched]; matched += 1
    if that token is a stop: stop the round (no correction)
if out < R and not stopped: emit predictions[matched]   # correction or bonus
```

Emitting `m` tokens materializes the old pending token and the first `m - 1`
emitted tokens, so verify column `m - 1` holds the accepted state. The kernel sets
`frontier = base_frontier + m` and `last_token = tokens[m - 1]`; `commit_state`
selects column `m - 1` (or the round start when `m == 0`), reading `n_emitted`
from device memory.

`commit(accepted)` publishes only what the device chose. When Core trims the row
(for example at a request stop the device did not know), the controller enqueues
a device truncation that recomputes frontier, pending token and count from
`base_*` and reruns `commit_state` eagerly. Because `commit_state` is a pure
function of round-start state and verify outputs, this is exact. Draft width zero
is ordinary decoding through the same path, the natural timing baseline.

### Sampled requests

The accept kernel only compares `drafts[i]` with `predictions[i]`. For a sampled
request, verify writes the target's K+1 FP32 logit rows instead of argmaxes, and
[`token_sampling::enqueue_verify`](../token-sampling/README.md#how-it-works) turns
them into predictions. With greedy drafts every column holds the target's own draw,
so a draft stands exactly when it equals it and the round emits, token for token,
what plain sampling emits with the same seed (given column-invariant verify
arithmetic). With sampled drafts column `i` holds the draft when the rejection test
`u_accept * q(d) < p(d)` accepts it, otherwise a draw from `max(p - q, 0)`, and
column K a bonus draw from the target. The unchanged greedy rule then emits the
accepted drafts and the first differing prediction. Every draw is Core's
`uniform(seed, output position, stream, purpose)`, so a truncation or a rejected
tail needs no generator state, and the emitted text is a sample of the target's
processed law (temperature, top-k, top-p, min-p, penalties) for any drafter and
width.

```text
Forward: copy the allowance
         token_sampling::enqueue_commit(result + 1, control + commit_tokens)   # the previous round's tokens
         draft: greedy argmax (q one-hot), or enqueue_sample(DrawPurpose::draft) writing q rows
         verify: K + 1 target logit rows, then enqueue_verify -> predictions
Finish:  enqueue_greedy_accept, commit_state
```

The commit kernel runs first in Forward because Core can still trim the previous
round after Finish; `control[commit_tokens]` holds the kept count by the next
launch (the accept and the truncation set it to `n_emitted`), and `start()` and
`seed()` zero it, so the engine initializes `Request.position`, `counts` and
`prompt_seen` (including the pending first output) at request start. Never pass
`n_emitted` instead: after a seed it is 1 for the drafter's catch-up while the seed's
token is already in the history, which would count it twice.
Greedy drafts are simplest; drafts drawn from the drafter's own processed law accept
more often at high temperature (`1 - TV(p, q)` instead of `p(argmax q)`), at the
cost of one law per draft step and K dense `q` rows. A drafter's stopping rule (for
example a confidence threshold) changes only the width, never the output law.

### Forced replay

crit Eval's `replay` ([adapter](https://github.com/crit-labs/crit-eval/blob/main/docs/ADAPTER.md)) and crit serve's score
call run the speculative loop on a fixed continuation: drafts are accepted while they
equal its next tokens and every correction or bonus token comes from it. The accept
needs no mode for that. A verify step that writes `predictions[c] = continuation[outputs + c]`,
where `outputs` is the committed output count (`control[frontier]` minus the position
of the first output, both on the device), makes the unchanged rule emit exactly the
continuation, and Core's allowance, capped at the tokens left, ends it. Verify still
computes and keeps every column's state, so the accepted prefix is materialized as
usual, and log-probability rows for the forced tokens come from the same logits with
`predictions` as the chosen ids. A forced verify is a different step, so give it its
own controller (or a device flag its step reads); the control block and the accept
stay as they are.

### Column-invariant arithmetic

Greedy speculative output equals plain greedy output only when verify computes
each of its K+1 columns exactly as the one-token step computes its single column.
A kernel that picks its K split, tile or reduction tree from the column count
changes low-order bits with the width, and an argmax near a tie can flip. Make
every kernel on the verify and step paths column invariant: a fixed K split per
matrix, partial sums reduced in a fixed order, epilogues without width-dependent
fused multiply-add contraction, and attention that reduces each query's keys in
the same order for one or K+1 queries. One measured engine with column-invariant
GEMVs (8 warps splitting K, reduced in a fixed order) produced plain greedy output
on every speed case in every candidate.
[column_invariance.py](code/column_invariance.py) checks an operator by running
the same columns alone and inside batches of several widths.

### Finite padding

Fixed-width graphs compute columns that no result reads: catch-up columns past the
emitted count, verify columns past the allowance or a cut draft, padding rows of a
batch. Masking keeps them out of the real rows only while everything they hold is
finite. A masked key enters attention, like any masked term of a weighted sum, with
weight 0, and `0 * NaN` and `0 * inf` are NaN, so one non-finite key or value in a
junk column silently poisons every real row that reads it. Zero such buffers when
allocating them (the controller fills its own) instead of leaving them uninitialized,
keep stale values finite (a row that turned non-finite can be read as zeros where it
enters), and give junk columns a valid token id and position when a fixed-width step
gathers an embedding or writes a cache row for them (hidden-state-drafters' catch-up
marks them -1: use the pending token and the plan's position
`frontier - n_emitted + c` there).

## What the code supplies

Reference code, not tuned for performance:

| File | Supplies |
| --- | --- |
| [greedy_accept.h](code/greedy_accept.h) / [.cu](code/greedy_accept.cu) | `control` layout, `result_width(K)`, batched `AcceptRows`, `enqueue_greedy_accept` (one thread per row), `enqueue_truncate`, `enqueue_seed`, `PackedLayout`/`PackRows`/`enqueue_pack` (the readback block), `enqueue_set_budget` (zero copy); unused result slots are -1 |
| [device_round.h](code/device_round.h) / [.cpp](code/device_round.cpp) | `DeviceSpeculativeRound`, a single-row controller: `start()` or `seed()`, `prepare([K])` captures Forward and Finish (`RoundConfig::graphs = false` runs eagerly), `launch(output_tokens[, K])` with K up to the configured widest over one set of buffers, `tokens()`, `matched()`, `logprobs()`/`top_logprobs()` (`RoundConfig::logprobs`, `set_top_logprobs`), `RoundConfig::zero_copy`, `commit(accepted)`, `timings()` (`DeviceStage::forward`/`accept`) |
| [toy_model.h](code/toy_model.h), [toy_model.cu](code/toy_model.cu), [toy_example.cpp](code/toy_example.cpp) | Synthetic target (attention over a device KV cache, a gated-delta layer from [recurrent-state-recovery](../recurrent-state-recovery/README.md), output head) and a bigram drafter through a real Core `Engine` |
| [catch_up.py](code/catch_up.py) | `catch_up_plan` and a `PositionalDrafter` toy for stateful drafters |
| [round_budget.h](code/round_budget.h), [round_budget.py](code/round_budget.py) | `RoundBudget`: tokens per round, round/step ratio, speedup `tokens_per_round / ratio`, pass/fail at 1.3 |

## What the engine supplies

Three capture-safe `DeviceStep` callables over the stable `RoundView` pointers,
plus all model arithmetic, KV layout, sampling and batching across requests:

| Step | Contract |
| --- | --- |
| `draft` | Write `drafts[0..K)` from `control[last_token]` at `control[frontier]`, plus any private draft state |
| `verify` | Feed `last_token` then the drafts at `frontier..frontier+K`; write `predictions[0..K]`; keep per-column state or inputs; with `RoundConfig::logprobs`, write `RoundView::logprob`/`top_ids`/`top_logprobs` for columns `0..K` (Core's `TokenLogprob` definition: raw logits, alternatives by log-probability then id) |
| `commit_state` | Select model state for `control[n_emitted]` relative to the round start; idempotent, may run twice |

An optional fourth step, `accept`, replaces the greedy accept kernel in Finish
with another walk that keeps its control and packed-result contract, such as the
tree walk of [draft trees](../draft-trees/README.md); truncation, commit and the
single readback stay unchanged.

Steps submit only to the supplied stream: no host synchronization, readback,
allocation or pointer that changes between rounds. A recurrent layer can use the
device commit in [recurrent-state-recovery](../recurrent-state-recovery/README.md);
a KV cache advances its provisional frontier as in
[attention-state](../attention-state/README.md). In a Core `Session`:

```text
open_request:     round.set_stops(effective request stops)   # device list, no recapture
                  round.set_top_logprobs(options.logprobs)   # with RequestOptions::logprobs
advance(budget):  prefill as usual; on the last prefill chunk call start(frontier, last)
                  (or seed(frontier, first, previous) for a stateful drafter, below)
                  decode: ready = round.launch(budget.output_tokens)    # or (.., K) per round
                          return {0, true, {}, ready, deferred_tokens = true,
                                  logprobs = round.logprobs(), top_logprobs = round.top_logprobs()}
ready_tokens():   return round.tokens()                       # after Core's one wait
device_timings(): return round.timings()
commit(accepted): round.commit(accepted)
close(finish):    if round.pending(): round.abandon()        # cancelled after launch
```

Core reads the span from `ready_tokens()` after its single wait, so the round's
GPU time appears in the `readiness` span. The pinned result stays valid until
`commit`. Core may close a cancelled session without a commit; `abandon()` waits
for the readback and requires a fresh `start()`, and an engine reusing one
controller must call it or the next `set_stops()`/`start()` rejects the pending
round. Stops live in device memory, up to `max_stops`.

One capture serves every frontier only if launch shapes do not depend on context
length. When a kernel takes a host-side length bound (a split count or grid sized
from the maximum visible keys), capture one Forward/Finish pair per frontier range
and select it from the host frontier mirror. Reserve the controller's buffers and
per-column state in Core resources; the controller allocates through the backend.

**Stateful drafters.** A drafter with its own per-position state must first
ingest what the previous round accepted, a count only the device knows. Process a
fixed `K + 1` columns (with widths chosen per round, `RoundView::max_draft_tokens + 1`:
the previous round may have emitted more than a narrower round's `K + 1`; see
[hidden-state-drafters](../hidden-state-drafters/README.md)' `enqueue_eagle_ingest`
with `columns`) and mask by position: column `c` holds the previous verify's
feature `c` and the `c`-th emitted token at position `frontier - n_emitted + c`;
columns `c >= n_emitted` are junk at positions `>= frontier`, overwritten before
any read. Seed the first draft from column `n_emitted - 1`. When the prefill
materializes the whole prompt and produces the first output itself, call
`seed(frontier, first, previous)` (kernel `enqueue_seed`) instead of `start()`: the
state a one-column round at `frontier - 1` leaves after emitting `first`, so the first
catch-up ingests (last prefilled feature, `first`), with `commit_tokens` 0 because the
engine has put `first` in its sampling history.

Because `commit_state` can run twice (Finish, then after a Core truncation), an
in-place state update would apply the prefix twice. Keep round-start state
separate from live state, or record only the count and apply the prefix at the
next verify (the deferred strategy in
[recurrent-state-recovery](../recurrent-state-recovery/README.md)).

## Worked example

Run from `code/`. A round costing 1.2 ordinary steps that publishes three tokens
passes; a round costing 2.5 steps fails even with four tokens.

```python
from round_budget import DeviceTiming, RoundBudget

budget = RoundBudget()
budget.observe_step([DeviceTiming("forward", 0, 1000)])
budget.observe_round([DeviceTiming("forward", 0, 1100), DeviceTiming("accept", 0, 100)], emitted=3)
report = budget.report()
assert report.passed and abs(report.ratio - 1.2) < 1e-12
assert report.tokens_per_round == 3 and abs(report.speedup - 2.5) < 1e-12

slow = RoundBudget()
slow.observe_step([1000])
slow.observe_round([2000, 500], emitted=4)
assert not slow.report().passed and slow.report().threshold == 1.3

# Column invariance: a fixed summation order gives every column the same bits at any batch width; a split
# chosen from the width does not.
import math
import random
from column_invariance import check, fixed_order_dot, width_split_dot

rng = random.Random(1)
weights = [[rng.uniform(-1, 1) * 10 ** rng.randint(-3, 3) for _ in range(16)] for _ in range(4)]
columns = [[rng.uniform(-1, 1) * 10 ** rng.randint(-3, 3) for _ in range(16)] for _ in range(6)]
filler = [[rng.uniform(-1, 1) for _ in range(16)] for _ in range(4)]
assert check(fixed_order_dot(weights), columns, widths=(2, 4, 8), filler=filler) == []
assert check(width_split_dot(weights), columns, widths=(2, 4, 8), filler=filler)  # some columns change bits

# Finite padding: a masked junk column enters a weighted sum with weight 0, and 0 * NaN is NaN.
probabilities, values = [0.25, 0.75, 0.0], [1.0, 2.0, float("nan")]   # the last column is masked junk
assert math.isnan(sum(p * v for p, v in zip(probabilities, values)))  # the real row is poisoned
values[2] = 0.0                                                         # finite junk: the mask holds
assert sum(p * v for p, v in zip(probabilities, values)) == 1.75
```

The native [toy example](code/toy_example.cpp) checks through a Core `Engine`:
output equal to one-token greedy decoding for two prompts in graph and eager mode
and for width 0; exactly one readback per round (counted at the CUDA API with
`--wrap`); one Forward and one Finish `device` event per round; and a Core-trimmed
stop applied through `commit(accepted)`. With its serial reference kernels it
usually fails the 1.3 threshold even at about 3.4 tokens per round, which is the
signal to fix round structure first. Build and run:

```sh
cmake -S inference -B build/inference-cuda -DCRIT_INFERENCE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/inference-cuda --config Release
ctest --test-dir build/inference-cuda -C Release -R "crit_device_round|crit_toy_round" --output-on-failure
```

## Adapting it

Preserve: device acceptance following the greedy rule exactly; exactly one
readback per round, one device-to-host copy or the block written in place (no
host fence inside `launch`); `frontier`
advancing by `n_emitted` with the accepted state at column `n_emitted - 1`; a
smaller Core count applied before the next round; the sampling history taking
`control[commit_tokens]`, never `n_emitted`; idempotent `commit_state`;
shared token IDs and tie rule between draft and target; column-invariant kernels.

Adaptable: the drafter, draft width, how verify stores provisional state, batching
rows (the accept kernel takes many; a batched launch lays its rows out at its own
K), fusing accept into the verify epilogue, and graph slot reuse across widths.

Keep the `draft` step's kernels recognisable in a kernel trace. Both steps are recorded into the
round's graphs, so host-side ranges (NVTX) cannot separate them; kernel names can. Launch the
drafter's matrices through [stored-tile-gemm](../stored-tile-gemm/README.md) plans with
`role = KernelRole::draft`, give the drafter's own kernels names that contain `draft`, or declare
`kernel_work` rules with `"term": "draft"` in the engine's info. Then `crit eval status RESULT
--kernels kernels.json` splits each round into its verify pass and its draft steps for an engine that
does not time them itself (`verify_ms`, `draft_ms`), and gives their gaps apart.

## Costs and validation

A round costs one draft pass, one K+1-column verify, the accept kernel, state
selection, the pack kernel, two graph launches and a `4 * (K + 4)` byte copy
(`4 * (K + 1) * (1 + 2 * top)` more with log-probabilities), plus `O(K)` control
storage and any per-column state verify keeps. With zero copy the allowance copy
becomes a one-thread kernel reading host memory and the readback copy disappears;
the pack kernel's writes then cross the bus. Copies next to kernels cost more on
some platforms than others, so time rounds both ways on the target
(`crit_device_round_tests timing` for the round mechanics alone, `RoundBudget` for
the engine). For illustration, on one RTX 5090 under WSL2 (driver 596.49, CUDA
13.1) the hash fixture's graph-captured rounds took 170 to 215 us copying and 65
to 100 us zero copy (medians of 7 x 500 rounds, K 0 and 4, three runs): about
50 us per copy there. Tokens and device state are the same either way.

**Profile a whole round first.** Feed `RoundBudget` the ordinary step time (K = 0
or the engine's normal decode) and the round time. A failing ratio usually means
overhead (host syncs, extra readbacks, uncaptured work, small launches), not slow
kernels; a passing ratio with few tokens per round points at the drafter. See
[the profiling recipe](#profile-a-whole-round).

**Report acceptance with every speed** (`accepted_tokens_per_round` beside
`decode_tps`; `matched()` is each round's accepted drafts), and compare round time
when generated text differs.

**Choose draft width on mixed text.** Measure several widths (K = 2..8) on prose,
code, reasoning and another language, such as the
[crit eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md) speed suite. Each extra position adds a
draft step and a verify column, so widen K and cheapen the drafter together (a
cheaper drafter encoding or a [draft vocabulary subset](../draft-vocabulary-subset/README.md)
changes acceptance only). Per-round K from recent acceptance: size the controller
for the widest K and pass each round's K to `launch(output_tokens, draft_tokens)`;
the buffers are shared and each width captures its own Forward/Finish pair on its
first launch (steps see the width in `RoundView::draft_tokens`). With routed experts offloaded to host memory, each column routes
independently, so measure expert misses per round before adding draft tokens.

### Profile a whole round

Enable tracing (`Profile::trace.capacity`), time graph launches with
`CudaStageTimer` and return them from `Session::device_timings()`; Core records
each as a `device` event under the step's `operation`, and the gap to the host
`step` span is host work. `CudaCategoryTimer` splits device time by category
([CUDA services](../../core/src/backends/cuda/README.md#device-time-by-category)).
Confirm the shape with Nsight Systems:

```sh
nsys profile --cuda-graph-trace=graph --trace=cuda,nvtx,osrt --sample=none -o round ./engine-benchmark
nsys stats --report cuda_api_sum,cuda_gpu_trace round.nsys-rep
```

Compare two runs differing only in generated tokens: each extra round should add
2 `cudaGraphLaunch`, 1 device-to-host copy (none with zero copy) and 2
`cudaEventSynchronize` (1 blocking). Only after the ratio passes, use `--cuda-graph-trace=node` or Nsight
Compute for individual kernels.

Failure modes: steps that are not capture-safe fail at `prepare()`; a device count
outside `1..allowance`, an out-of-vocabulary token, a matched count above it or a
device frontier other than the host mirror plus the count rejects `tokens()`; `launch`
rejects a round that could write past `context`; a failed truncation leaves device
state undefined, so the request must fail.

[Python](../../tests/test_device_rounds.py) and
[native fixtures](../../tests/inference/device_round_tests.cpp) check catch-up
against exact ingestion, the accept kernel against `GreedySpeculator` on 4,000
random rows, graph replay against eager and greedy for K = 0..4 and for widths
chosen per round (0..6) over one controller, truncation and one copy per round,
and a sampling-history stand-in that takes in exactly the
tokens Core kept after `start()` and after `seed()`, zero-copy rounds equal to
copying ones with no device-to-host copy, forced replay of a continuation that departs
from greedy decoding emitting exactly that continuation, the readback block against the
result and control block with masked log-probability rows, and log-probability rows
through the controller. The toy example's log-probabilities and alternatives reach
Core's result equal to ordinary decoding's.
[sampled_round_tests.cpp](../../tests/inference/sampled_round_tests.cpp)
runs token-sampling's verify predictions through the accept, truncation and commit
kernels from seeded rows (whose first history commit takes in nothing) and compares each request with a host round over Core's laws and draws (the
target's draw per column, drafts kept while they match, stops, allowances), and
checks that a zero-width round is one ordinary draw. They qualify round mechanics, not
model arithmetic or speed.
