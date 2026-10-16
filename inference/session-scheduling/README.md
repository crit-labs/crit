# Session work and progress staging

Expose a long prompt as bounded, retryable prefill work and preserve generated
tokens across a model-state commit. The entry also supplies a latency-based chunk
selector, a physical batch planner with workspace sizing, and a reasoning-loop
observer. Admission, fairness, request selection and retirement stay with Core;
this is not a batching scheduler.

## When to use it

Use it in an engine adapter to split prefill, choose chunk sizes against a
latency target, or run one logical operation as smaller physical calls. Core's
concurrent runtime already alternates decode groups with bounded prefill.

## How it works

A prefill cursor has a committed offset and at most one pending end offset. The
next chunk is bounded by remaining input, the chunk limit and the current work
budget. Commit advances the offset; rollback drops the pending advance so the same
input can be retried.

A prompt slightly longer than a multiple of the chunk limit leaves a near-empty
last chunk that re-reads every weight for a handful of tokens (in one measured
engine, a 2,052-token prompt ran as 2,048 + 4 and the 4-token chunk cost about
40 ms). With a fold limit, a remainder of at most that many tokens joins the
previous chunk when the budget allows; `chunk_plan` gives the resulting lengths.

Generated progress follows a separate order: wait for readability, copy borrowed
token storage, validate the output bound, reserve its owned bytes, and only then
allow a model commit that could invalidate the borrowed storage. Successful model
commit precedes Core publication.

## What the code supplies

[implementation.py](code/implementation.py) supplies `PrefillCursor` (with
optional tail folding), `chunk_plan`, `StagedProgress` and `stage_progress`. The
readiness hook uses `wait()` and the reservation hook is `reserve(nbytes)`; tokens
are charged four bytes each, which is neither vocabulary validation nor the
tuple's heap size. [reasoning_guard.py](code/reasoning_guard.py) supplies
`ReasoningLoopGuard`. [latency_chunks.h](code/latency_chunks.h) and
[physical_batches.h](code/physical_batches.h) are described below.

## What the engine supplies

Submit the selected chunk and advance only when the model-state transition is
accepted. `rollback()` affects only the cursor; the engine undoes its own
attention, recurrent or other provisional state. A zero-length prompt is already
complete; check `complete` before requesting work, since `next_chunk()` at the end
still creates an empty pending chunk.

For output, prefer Core's staged-progress contract; this reference explains the
producer's obligations. Keep the reservation owner for the published storage's
lifetime. A failing readiness or reservation hook must leave model state
uncommitted.

## Worked example

Run this from `code/`. A rejected prefill attempt retries the same tokens, and a
staged output survives mutation of the model's borrowed list.

```python
from implementation import PrefillCursor, chunk_plan, stage_progress

assert chunk_plan(2052, 2048) == [2048, 4]
assert chunk_plan(2052, 2048, fold_limit=256) == [2052]  # the short tail joins the previous chunk
folded = PrefillCursor(list(range(7)), chunk_limit=5, fold_limit=2)
assert len(folded.next_chunk(budget=8)) == 7

cursor = PrefillCursor([10, 11, 12, 13, 14], chunk_limit=3)
assert cursor.next_chunk(budget=2) == (10, 11)
cursor.rollback()
assert cursor.next_chunk(budget=2) == (10, 11)
cursor.commit()
assert cursor.next_chunk(budget=8) == (12, 13, 14)
cursor.commit()
assert cursor.complete

class Ready:
    def __init__(self):
        self.waited = False
    def wait(self):
        self.waited = True

ready, charges, borrowed = Ready(), [], [7, 8]
def reserve(nbytes):
    assert ready.waited
    charges.append(nbytes)
    return object()

progress, reservation = stage_progress(borrowed, 2, reserve, ready)
borrowed.clear()  # a model commit can invalidate its previous view
assert progress.tokens == (7, 8)
assert progress.charged_bytes == 8 and charges == [8]
assert reservation is not None

from reasoning_guard import ReasoningLoopGuard
guard = ReasoningLoopGuard(passage_words=3, repeats=3, window_words=12,
                           history_words=30, threshold=.5)
for fragment in ("check these cases. ", "CHECK these cases. ", "check these cases. "):
    guard.feed(fragment)
assert guard.observe().triggered
```

### Observe reasoning loops

`ReasoningLoopGuard` casefolds committed reasoning text into words, counts exact
repeated passages and reports the fraction of the recent window they cover. Cost
is O(history * passage length) per observation. It is a heuristic: repeated
analysis can trigger it and paraphrased loops evade it. The engine chooses
warning, stop or recovery.
Feed only committed reasoning, never drafts, tool arguments or answer text, with
one guard per request. An injected thinking closure must pass through the normal
accepted-state path, and with [Core's structured output](../../core/src/constraints/README.md)
must be allowed by the language or occur before constrained output starts.

### Supplying cache-aware admission hints

Core's optional [reuse-aware admission](../../core/src/runtime/README.md#reuse-aware-admission)
can consume remaining-work estimates from checkpoint-storage's `choose_prefix`:
convert predicted seconds to bounded nanoseconds and report the compatible
reusable-token count, including restore time. Give the same nonzero affinity only
to requests whose pending prefix computation can really be shared under the
engine's model/adapter/conditioning identity. This is a scheduling hint, not a
state lookup key; on selection, reacquire and validate the state and fall back to
computation on failure.

### Latency-based chunk selection

`choose_prefill_chunk` ([latency_chunks.h](code/latency_chunks.h)) estimates a
chunk of `n` tokens after `L` occupied tokens as
`call + n*token + (L*n+n*(n+1)/2)*pair + decode*rate` nanoseconds (`pair=0` for
linear/recurrent models) and returns the largest aligned chunk within the target,
capped by remaining prompt, token quota and workspace. If even the minimum chunk
misses the target it returns it with `target_met=false`; invalid calibration
selects a bounded fixed fallback. For example, `choose_prefill_chunk({0,100,100,100,1,16,0,100},
ChunkLatency{10,1,1,0})` selects 12 tokens; at history 100 it selects one and
misses the target (synthetic costs).

To integrate, opt into `ModelInfo.prefill_advice` and `Profile.prefill_target_ns`
and implement `Session::prefill_limit` (or the concurrent model's equivalent),
converting `PrefillWindow` into `ChunkWindow`. Return the token count; zero keeps
the fixed fallback. Advice only shrinks a mixed group. When the quota is smaller
than the alignment the selector returns that quota, so engines must support such
grants. Calibrate from operation shapes, not Eval text, and
measure decode inter-token latency as well as prefill throughput.

### Independent physical batches and workspace

[physical_batches.h](code/physical_batches.h) supplies `physical::BatchLimit`,
`SplitRules`, `BatchPlan`, `LinearWorkspace` and `HostWorkspace`, for running one
logical operation (for example a draft's prompt ingestion or catch-up) as smaller
physical calls while the target keeps larger ones. A zero override inherits the
existing cap; a positive one replaces it. This is independent of Core's work
quota, the cursor's chunk and the number of speculative proposals.

Splitting is off by default. Set `splittable=true` only if the operation carries
causal/recurrent state between calls. `atomic_columns` declares indivisible equal
groups; every cut must be a multiple. Use an unsplit plan for whole-block
operations. `padding_multiple>1` declares that padded columns can be masked
without changing outputs, positions, state, RNG or grammar; the cap includes
padding. An incomplete group or a cap too small for an atomic operation fails
before execution. For example, 15 columns with groups of 3, padding multiple 4
and cap 8 run as valid widths `[6,6,3]` and physical widths `[8,8,4]`.

The engine supplies a workspace size for **every actual call shape**; no
monotonicity is assumed, since a narrow tail can need more memory.
`BatchPlan(17, {12,4}, {true}, LinearWorkspace{32,16})` runs `[4,4,4,4,1]` with
peak scratch 96 bytes; inheriting with `{12,0}` runs `[12,5]` with 224 bytes.
`HostWorkspace(plan.peak_workspace_bytes())` allocates once and
`run(plan, callback, cancelled)` reuses it, with ordered synchronous calls and
cancellation checked around each. Reentrant use of one owner is rejected.

The callback must settle all work before returning or throwing. On failure later
calls are skipped and the **engine discards or repairs its private state**; never
publish a partial operation. Reserve bytes through Core, keep target and draft
scratch separate unless reuse is proven, and rebuild buffers at a settled boundary
when a cap changes.

Smaller calls trade workspace for launches and weight reads. Use
[recorded selection](../recorded-selection/README.md) for measured choices. The
[speculative example](../mtp-decode/README.md#physical-batch-size-is-independent-of-proposal-width)
shows a complete synthetic loop.

## Adapting it

Preserve committed versus pending work, copying before borrowed storage becomes
invalid, admitted output bounds and reservation lifetime. Chunk size, fold limit
(a chunk may exceed the chunk limit by at most the fold limit, so size buffers for
both), storage format and readiness can change. Coordinate rollback with
[attention state](../attention-state/README.md) and multiple candidate rows with
[concurrent execution](../concurrent-request-execution/README.md); separate cursor
objects do not make a combined commit atomic.

## Costs and validation

Smaller chunks cost launches; larger ones delay other requests. A work budget
bounds tokens, not elapsed time. Common bugs: advancing on submission instead of acceptance, rolling back an offset but
not state, and keeping a borrowed output pointer after commit.

Host fixtures cover the cursor, chunk limits, token snapshots and output bounds,
not native readiness, model rollback or fairness. `tests/inference/scheduling_tests.cpp`
compares 6,000 chunk selections against exhaustive search, and
[physical batch tests](../../tests/inference/physical_batches_tests.cpp)
(`ctest -R crit_physical_batches`) cover 5,000 shape oracle cases, atomic and
padded tails, nonlinear workspace, failure and cancellation. An engine should also
test cancellation during a chunk and commit failure after output staging.
