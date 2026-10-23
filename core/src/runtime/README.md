# Request runtime

[runtime.cpp](runtime.cpp) implements the Engine/Session lifecycle: serial and
opt-in bounded concurrent requests, rotating chunked prefill, grouped
commit/publication, readiness, cancellation and independent retirement polling.
The public interface is [core.h](../../include/crit/core.h); model computation
comes through the public Model/Session interfaces. See
[current contracts](../../contracts/README.md).

`Profile.max_concurrency` and `ModelInfo.max_concurrency` admit up to
`max_request_concurrency` (4096) active generation lanes. This bounds scheduler
metadata, not device memory: the model must still admit each request's complete
memory entitlement.

Admission defaults to FIFO (`admission_lookahead=0`). A positive lookahead
examines at most that many followers plus the head per free lane, so ready
followers of a deferred request can backfill. A deferred request may be bypassed
by at most `max_admission_bypasses` successful admissions (default 8); once capped
it blocks further backfill until it is ready, cancelled, failed or expired. A
score is always a barrier. Admission decisions come from the model; Core does not
guess KV costs, evict live state or assume recomputation is safe.

Submission checks the outstanding allowance before retaining payload memory.
Concurrent requests reserve their maximum accepted-token staging buffer at
submission, alongside prompt and result storage. Insufficient Core buffer
capacity rejects the new submission without failing accepted work.

## Model progress

Serial sessions and concurrent rows use the same internal checks for prompt
progress, prefix reuse, prefill completion and output grants. Reuse must precede
any computed prompt token; newly computed tokens must fit the prefill grant.
Output requires complete prefill, and every step must consume prompt tokens or
produce output. Deferred reports require a readiness fence, complete prefill and
an initially empty token span; the final span is read after readiness.

The checks do not advance request state. Each execution path retains its own
readiness, cancellation, commit and cleanup ordering. Concurrent execution also
checks row membership and mixed token-work accounting, and validates every row
before decoding text or staging output for the group.

## Request release

When `Handle::wait()` or `wait_output()` returns (with a result, a cancellation
or a rethrown error, including anything leaving a sink), or a handle's destructor
returns without a wait, every host charge of that request (storage, options,
text and staging) has been released by the consuming thread, so
`Engine::usage(Memory::host)` read there afterwards excludes them. The request's
options and context are destroyed on that thread too.

A concurrent engine failure gives every request the same exception object; the
last consumer destroys it. ThreadSanitizer against an uninstrumented libstdc++
can report false-positive races on that shared exception.

Quarantine is the exception. When settlement is unproved (a failed serial drain,
or failed quiescence or drain after a concurrent failure), `wait()` still returns
the error, but the request's charges stay with the quarantined runtime.

## Mixed token-work batches

Set `Profile.max_batch_tokens` above zero and declare `ModelInfo.mixed_batches`
on a `ConcurrentModel` to use `execute(span<WorkItem>)`. Zero keeps the separate
`prefill` and `decode` callbacks. An unsupported declaration fails at
construction.

| `BatchPolicy` | Grant order |
| --- | --- |
| `fair` (default) | From a rotating cursor, the minimum work grant per eligible row (one for decode, the prefill quantum for prefill), then remaining units up to each prefill chunk or `ModelInfo.decode_token_work`. Prefill and decode can mix in one call. |
| `decode_first` | One unit per decoder, then decoders up to their preference, then prefill chunks; separate rotating cursors. A prefill that waited `prefill_starvation_steps` groups (default 8) reserves its minimum work grant before decoders. Requires positive `max_batch_tokens`. |

`WorkBudget.token_work` bounds computation for a row; `prefill_tokens` bounds
newly computed prompt tokens; `output_tokens` is the remaining output allowance.
A verify pass can compute several columns and publish fewer tokens. The adapter
reports actual `Progress.token_work`, counting every target, draft and padded
column it executes, and must accept grants down to one (falling back to an
ordinary step). Engines whose prefill kernels pad columns can declare
`ModelInfo.prefill_work_quantum`; Core grants prefill work in multiples of that
quantum while keeping prompt consumption within the real input and chunk limit.
The quantum must fit `max_batch_tokens`. It defaults to one, and decode grants
remain as small as one. Latency advice preserves the rounded work charge.
Reused prompt state costs no units. Core checks every row before
commit.

Token work is an admission quota, not equal time; budgets need measurement per
workload. The adapter supplies grouping, kernels, reservations and work
accounting. Core waits for the whole group before publication.

## Request options

`submit(prompt, output_tokens, options)` takes a value-owned `RequestOptions`.
Unset sampling fields use the adapter's defaults. `crit serve` forwards explicit
per-request overrides from packages declaring the corresponding
[sampling capabilities](../../../docs/SERVING.md#per-request-sampling-for-adapters),
including penalties and logit bias through the native engine ABI.
`ModelInfo.sampling` declares
support per field (`penalties` covers all three). Unsupported or invalid options
throw `std::invalid_argument` naming the field before any memory or model call,
for example `model does not support a requested sampling option: min_tokens`.

| Field | Valid values |
| --- | --- |
| `temperature` | Finite and nonnegative; zero is greedy |
| `top_p` | Finite, in `(0, 1]` |
| `top_k` | At most the declared vocabulary; zero disables |
| `seed` | Any unsigned 64-bit value |
| `min_p` | Finite, in `[0, 1]` |
| `repetition_penalty` | Finite and positive (1 is off); prompt and output tokens |
| `frequency_penalty`, `presence_penalty` | Finite, in `[-2, 2]`; output tokens |
| `logit_bias` | At most 1024 unique ids below the valid vocabulary; each finite or negative infinity (a ban) |
| `min_tokens` | Stop tokens suppressed until that many tokens were generated |

The runtime passes values through; the session applies them, normally with Core's
[sampling library](../sampling/README.md) (`resolve_sampling` and `Sampler`) or a
device path following the same rules.

`options.stop_tokens` replaces the model's stop list for this request (unset
inherits, empty disables). Core trims at the first stop, keeps that token and
reports `Finish::stop`. Text stops are under
[request text and stop strings](#request-text-and-stop-strings).

For extra parameters, derive an immutable host-only `RequestContext`, set
`options.context` and declare `ModelInfo.request_context`. `host_bytes()` must be
fast and constant; Core charges it per request even when shared.

Serial adapters implement `open_request(prompt, options)`; concurrent adapters
read `RequestView::options`, valid through session close. Scoring rejects every
generation option. The default `open_request` throws `std::logic_error` when any
option is set, so options are never dropped silently.

## Log-probabilities

`options.logprobs = n` returns, per generated token, its log-probability in
`Result::logprobs` and its `n` most likely alternatives in `Result::top_logprobs`
(tokens x n, row-major). It requires `ModelInfo.token_logprobs` and
`n <= ModelInfo.max_top_logprobs`. Values are natural logs under the raw logits
of the producing pass (before temperature, penalties, bias and filters),
normalized over the valid vocabulary. `ModelInfo.valid_vocabulary` (zero means
`vocabulary`) marks padding ids, which never appear as tokens, alternatives,
argmax or `logit_bias` entries.

The adapter returns `Progress::logprobs` and `Progress::top_logprobs`, borrowed
like `Progress::tokens`. Before commit Core checks exact counts (a deferred
round's spans must cover at least the final count), values that are not NaN or
+inf and at most `1e-3` above zero (-inf is zero probability), and alternatives
with distinct valid ids in non-increasing order (ties by ascending id). A
violation fails the serial request, or the concurrent engine with nothing from
that group published. `max_top_logprobs` is at most `top_logprobs_limit` (64).

## Output events

`Handle::wait_output(sink)` waits like `wait(TokenSink)` and calls the sink once
per published token, in order, on the waiting thread. Each `OutputEvent` holds
the token's index and id, its log-probability and alternatives when requested,
and the text released with it (possibly empty). Spans are valid only during the
call. A throwing sink cancels the request, waits for it, releases it and rethrows
the sink's exception unchanged (forced unwinding and foreign exceptions
included). The handle is consumed either way. A null sink just waits.

## Scoring

`submit_score(prompt, first_target, options)` requires `ModelInfo.scoring` and
returns one finite score per token from `first_target` (which needs a preceding
token) to the end. The adapter defines the semantics; Core checks count and
finiteness only.

**Query scoring.** `submit_score(prompt, first_target, ScoreQueries{ids, per_target, top_k, path}, options)`
requires `ModelInfo.query_scoring` and calls `Model::score_queries()`. Per target
the model returns its log-probability, those of `per_target` caller-chosen ids
(`ids`, row-major, below the valid vocabulary; for example a reference model's
top-k), optionally its argmax and, with `top_k`, its own ranked alternatives
(`QueryScores::top`, needs `ModelInfo.max_top_logprobs >= top_k`). Targets must
be below the valid vocabulary, else submission fails with "scoring target outside
the valid vocabulary". `path` names one of `ModelInfo.score_paths` and is
forwarded unchanged; empty selects the default.

`crit_engine_score` takes `top_k` with the query rows and returns the engine's own
alternatives with the target scores, for crit serve's fixed-text scores (`chat-score-v1`).

Core checks values under the [generation rules](#log-probabilities) and returns
them in `Result::scores`, `query_scores`, `argmax` and `top_logprobs`
(targets x top_k).

On a `ConcurrentModel`, scoring joins the same FIFO as generation but is an
exclusive barrier: earlier generation retires before the hook runs and later
generation cannot pass it. Scores run sequentially through
`Model::score_request()` without a generation session, so long scores delay later
work. Hooks must return; a concurrent scoring failure poisons the engine. The
adapter reserves its returned score allocation through `Resources`, keeping it
until the next ordinary request callback or model destruction; Core copies the
validated scores into its own result.

## Text services

Return a `shared_ptr<const TextFrontend>` from `Model::text_frontend()` to opt in.
Core obtains it once at construction; `Engine::tokenize()` and `detokenize()` then
run without the model lock, including during generation and inside token sinks.
Every method must support concurrent callers and own its tables independently of
model/backend state. Core validates token ids on both boundaries; tokenization
failures do not poison generation.

`Engine::create_decoder()` creates private state for one reply from the
frontend's `create_decoder()`: feed accepted tokens to `push(span)`, append the
returned text and call `finish()` at the end. The adapter owns codec buffering
(partial UTF-8, special tokens, context-sensitive spacing). Without incremental
support, build the generic [`offset_decoder`](../text/README.md#offset_decoder)
over `detokenize()`. Each decoder is used serially; Core keeps the frontend alive
until its decoders are destroyed. A decoder exception or `finish()` closes it.

With no frontend, Core serializes the legacy `Model::tokenize()`/`detokenize()`
and rejects them while generation is busy.

### Request text and stop strings

`options.text = true` publishes decoded output in `Result::text` and, per token,
`Result::text_ends` (end offset of the text released with that token). Non-empty
`stop_strings` (at most 16 non-empty, well-formed UTF-8 strings of at most 256
bytes) imply text. Both need an independent frontend. At submission Core creates
the request's decoder: the frontend's `create_decoder()` or, when that keeps the
default `std::logic_error`, an [`offset_decoder`](../text/README.md#offset_decoder)
seeded with the prompt's last tokens for continuation spacing and split
characters. A frontend's own decoder gets no prompt context.

Each step's accepted tokens go through one [`TextStream`](../text/README.md)
before commit with one trim order: stop tokens (kept, not decoded), the output
budget, then stop strings. A stop string ends the request with `Finish::stop`
after the token that completed it; later tokens of the step are dropped and
`Result::text` ends where the stop begins. Matching is byte-exact over decoded
text, independent of token boundaries; the first stop to complete wins (ties:
earliest start, then lowest index). Text that may begin a stop is held and
flushed at a length or stop-token finish, not on cancellation or failure.

Offsets, the stop matcher and a holdback are charged at submission; released text
is charged as it grows. A decoder exception or text reservation over the host ceiling fails only that
request with `Finish::failed`; on the concurrent path the row commits zero tokens
and other rows publish normally. `wait()` rethrows the error.

## Request timing

Every `Result` carries `timing` (`RequestTiming`), host steady-clock phases
recorded with or without tracing:

| Field | Interval |
| --- | --- |
| `queued_ns` | Submission to admission |
| `prefill_ns` | Admission to the step that completed prefill |
| `first_token_ns` | Admission to the first published token |
| `decode_ns` | Prefill completion to the last published token |
| `total_ns` | Admission to completion |
| `prefill_tokens`, `decode_steps` | Prompt tokens the model reported; steps after prefill |
| `reused_prompt_tokens` | Of `prefill_tokens`, those restored from retained state |

Phases that did not occur stay zero. Times are taken after readiness, so they
include device work. Divide published tokens by `decode_ns`, or prefill tokens
minus `reused_prompt_tokens` by `prefill_ns`, for rates without instrumenting the
adapter.

## Retained prompt prefixes

A model that keeps sequence state between requests reports the prefix it did not
recompute in `Progress::reused_prompt_tokens`, as part of `prompt_tokens`. Reuse
must precede any computed prompt token and does not count against the prefill
budget. The model alone decides whether retained state matches; Core validates
counts.

## Lifecycle tracing

Set tracing before constructing the engine; resource limits must cover its ring:

```cpp
crit::core::Profile profile;
profile.host_limit = 64 * 1024 * 1024;
profile.device_limit = 1024 * 1024 * 1024;
profile.trace = {.capacity = 4096, .slow_step_us = 50000};
// Construct Engine with this profile, backend and factory.
// auto snapshot = engine.trace();
```

`capacity` is the number of fixed-size events (at most 65536; zero disables) and
charges `capacity * sizeof(TraceEvent)` host bytes. Collection does no per-event
allocation or device fence.

| Field | Meaning |
| --- | --- |
| `sequence`, `timestamp_ns` | Recording order; monotonic host time since construction |
| `key` | Request, lane and generation; request zero is engine-wide |
| `phase`, `mark` | Lifecycle phase with point, begin, end or exception exit |
| `operation` | Correlates begin/end; grouped rows share it |
| `duration_ns`, `gap_ns` | Host time of a completed phase; on steps, time since the previous step ended |
| `slow`, `slow_steps` | Threshold classification; whole steps counted once per group |
| `finish` | Request result on completion events |
| `stage`, `index` | On device events, the adapter's `DeviceStage` and index |
| `overwritten`, `dropped` | Ring replacement; events lost to contention |

Phases span submission through retirement, quiesce and drain. A `step`'s host
duration can include waits; it is not GPU time. Async batch steps can overlap, so
their durations do not sum to wall time.

`device` events carry adapter-measured GPU time: with tracing on, Core calls
`Session::device_timings()` (or `ConcurrentModel::device_timings()`) after
readiness and before commit. A throwing hook counts as `dropped`. Snapshots may
run concurrently with requests; writers never wait, so check loss counters and
capture snapshots before later traffic overwrites them.

### Operator ranges

Operator ranges describe work inside a step. [trace.h](../../include/crit/trace.h)
supplies `crit::trace::Work` (stage name, tokens, batch, context, a useful-byte
interval and a FLOP estimate), exception-safe `Scope`s on a `Sink`, and a bounded
`Recorder` ([operators.cpp](operators.cpp)). Counts are caller estimates, not
measured traffic or time; count a fused operator's reads and arithmetic once.

```cpp
#define CRIT_TRACE_ENABLED // before including crit/trace.h, or as a compile definition
profile.trace.operators = 4096; // independent of trace.capacity
// In a session callback:
CRIT_TRACE_OPERATOR(crit::trace::Work{"projection", tokens, 1, 0, bytes_min, bytes_max, flops});
// Later: engine.trace().operators.ranges
```

`CRIT_TRACE_SCOPE(sink, work)` uses an explicit sink, `CRIT_TRACE_OPERATOR(work)`
the thread's `current_sink()` (bound with `SinkBinding`). Without
`CRIT_TRACE_ENABLED` both expand to nothing. A scope begins and ends on one
thread and adds no device work. Device time comes from `device_timings()`, a
profiler or the [CUDA category timer](../backends/cuda/README.md#device-time-by-category).

With `TraceConfig::operators` set (at most 65536), the engine owns a `Recorder`
ring, charges `operators * sizeof(crit::trace::Range)` host bytes and binds it on
the threads running model callbacks. Ranges share the lifecycle clock and appear
in `TraceSnapshot::operators`; its `overwritten` and `dropped` counters report
loss. Other threads record only into sinks they bind.

`crit::trace::NvtxSink` emits host ranges for a profiler. Build Core with
`-DCRIT_CORE_NVTX=ON` (set `CRIT_CORE_NVTX_INCLUDE_DIR` if the CUDA Toolkit's copy
is not found) ([nvtx.cpp](nvtx.cpp)).

## Speculative rounds

One `advance()` (or `decode()` row) covers a whole speculative round on the
device; its `readiness` is one readback of the accepted tokens. When the device
decides the count, return `deferred_tokens = true` with empty `tokens`: after
readiness Core calls `Session::ready_tokens()` (or
`ConcurrentModel::ready_tokens(key)`) and validates that span, so `advance()`
never waits. Core may trim the count for budget or stops but never adds tokens.
With `logprobs`, return spans covering the round's largest possible count.

### Reuse-aware admission

`Profile.admission_policy = AdmissionPolicy::reuse_aware` requires a concurrent
model declaring `ModelInfo.scheduling_hints`. Its host-only
`scheduling_hint(RequestView)` reports estimated remaining nanoseconds (zero is
unknown), reusable tokens, an optional prefix-affinity key and whether that
prefix is pending. Hints never authorize state reuse.

Core examines at most `admission_lookahead + 1` requests without crossing a score
barrier and ranks ready ones by elapsed wait divided by known cost (FIFO ties,
unknown last). `admission_starvation_ms` or `max_admission_bypasses` forces queue
order. A nonzero affinity matching an active unfinished prefill defers duplicate
prefix work. `admission` and `open` remain responsible for capacity and cache
leases.

### Prefill latency advice

`Profile.prefill_target_ns` enables `prefill_limit` on a model declaring
`ModelInfo.prefill_advice`. It receives a `PrefillWindow` (occupied context,
remaining prompt, token ceiling, co-batched decode work, target nanoseconds).
Zero keeps the fixed grant; a positive answer may only shrink it. Core enforces
quotas, not predicted time.

## Asynchronous batch pipeline

Set `Profile.max_inflight_batches` (1 through 64) with mixed batches and declare
`ModelInfo.async_batches` to use `ConcurrentModel.prepare_batch`. Zero keeps the
synchronous path. `prepare_batch` retains inputs and reserves workspace without
launching; Core then calls `submit(notify)`. `ready` is nonblocking and means
production and readers have settled. `progress`, deferred tokens, timings and
`commit` belong to the operation, not global model scratch.

The adapter calls `notify` once `ready()` can observe settlement, including
failure; it is safe on any thread, repeatable, and only wakes Core. Publish
readiness before notifying and join notifications in `quiesce`/destruction.

While operations run, Core admits requests and submits other eligible lanes up to
the bound, by default at most one operation per lane; shared model inspection must be safe
alongside execution. `batch_timeout_ms` (default 30000) bounds a pending
operation: expiry fails the engine. Failure or shutdown calls every operation's
idempotent `quiesce`, then model quiescence and backend drain. Commit must hand
newly launched state work to the sessions before the operation is destroyed. The
bound limits operation count, not bytes; adapters reserve all workspace.

## Dependent execution

`Profile.max_request_steps` defaults to one. Larger values require async mixed
execution and a matching `ModelInfo.max_request_steps` declaration. The model's
`execution_depth(RequestView)` further limits each request (default one); use one
for a host sampler or grammar whose decision cannot be made by a private executor.
`max_step_output_tokens` caps each unresolved output grant. Core reserves their
sum against the true remaining request length; filling a step grant does not end
the request. Prompt work stays depth one until prefix reuse and prefill resolve.

Each opted-in WorkItem carries a monotonic `step` and an owned `predecessor`.
After successful submission, `BatchExecution.continuation(StepKey)` must return
an already-established dependency and private accepted state. Successors execute
after it, using real accepted tokens, counts and positions. RowProgress and
CommitDecision echo the step identity. Each operation owns its scratch, output,
readback and any graph slots. Public commit must not overwrite state a successor
uses or index unapproved output in shared caches. Existing single-pending-round
StateCache/TokenHistory contracts remain unchanged.

Publication waits for earlier steps of every row in a group; unrelated groups
may complete independently. Stops/cancellation mark descendants discard-only.
Core still reads their operation status and detects errors, but does not decode
or publish their tokens. Retirement waits for every descendant to settle. Failure
quiesces all operation owners, or retains them together in quarantine. Memory
preemption freezes younger chains until settlement, then captures the approved
boundary through the same recovery contract below.

The [portable recurrent reference](../../../inference/concurrent-request-execution/code/recurrent_pipeline.h)
uses asynchronous producer tasks and owned futures. A successor can compute from
the predecessor's actual recurrent state while host publication is gated. Private
sampling histories and complete recurrent snapshots preserve seeded output under
preemption. This is functional integration evidence, not a throughput measurement
or automatic acceleration of existing engine packages.

The [CUDA recurrence example](../../../inference/concurrent-request-execution/code/cuda_ahead_example.cu)
uses device-carried state and tokens, producer events and separate retained
readback slots. Its test gates predecessor readback while a successor computes,
then checks ordered output against a host oracle and cancellation cleanup. It
qualifies that synthetic device path, not full-model arithmetic or performance.

## Incremental memory and recovery

`Profile.memory_preemption` requires async mixed batches and the matching model
declaration. `prepare_incremental` returns an owned prepared operation, a deferred
row with engine-defined `MemoryPressure`, or a solo-infeasible row. Deferred
preparation must launch nothing and undo partial reservations; exceptions still
fail the engine. Pressure describes actual pools, not interchangeable byte totals.
`reclaimable` inspects settled victims and must count only capacity really freed,
including shared pages and copy-on-write, all recurrent/draft state and scratch.

Core favors the oldest resident request and considers younger settled victims.
It holds new admissions during pressure, retries blocked rows after capacity or
work changes, and restores suspended requests oldest first with no other resident
requests. This conservative recovery policy reserves a solo route through replay
and useful new work; it does not optimize the cost of preemption. A request that
cannot progress alone fails individually. If all residents are blocked with no
reclaimable victim or pending work, one blocked request fails to release capacity
for its peers. Suspended requests still count toward
the outstanding limit and scoring barriers, and keep their original queue age,
options, accepted output and text decoder. Restored bindings get fresh generations.

`suspend` and `restore` prepare operations without submitting. Core retains each
before `submit`, through readiness, or through quiescence/drain on failure. A
suspension's readiness certifies a complete `ResumeImage` and source release.
After `take`, the old session is destroyed without retirement or completion.
Failure cleanup must tolerate partially completed suspension. Restoration reserves
the old frontier plus at least one new useful step before submitting. Its `take`
returns a settled session. Recovery operations must join all producers in
`quiesce`; uncertain cleanup retains the whole runtime in quarantine.

Images are engine-owned and charged to Core Resources. Replay is opt-in: preserve
conditioning, pending token, positions, sampling/penalty/grammar and recurrent or
draft state, not just KV. Feed accepted tokens without sampling or publishing them.
An engine whose arithmetic cannot reconstruct the same continuation needs a complete
snapshot. `RequestTiming.preemptions`, `recovery_ns` and `recomputed_tokens` report
recovery separately from public prompt/output usage. `recovery_ns` measures each
successful restoration from preparation to observed readiness, excluding time
suspended. Cancellation of a suspended
request drops its image; cancellation during restore settles the operation first.

The [attention recovery reference](../../../inference/attention-state/code/recoverable_engine.h)
uses actual StateCache pages and the existing attention arithmetic. Its portable
CPU operations settle synchronously; it establishes reclamation and replay
correctness, not device overlap or a throughput gain.

## Engine libraries

For concurrent serving, set `NativeEngine.concurrent_requests = true` and declare
`chat-concurrent-v1` in the package. The stock resident adapter exposes the configured
`Profile.max_concurrency + max_pending`, creates a private parser wrapper per request,
and shares the native engine and immutable tokenizer. All engine state must obey the
ConcurrentModel ownership contract. The legacy zero-argument `request` facts callback
is rejected with this flag; use `result(const Result&)` for request-owned facts.

Libraries also export `crit_engine_request_open`, `request_wait`,
`request_cancel` and `request_close` (each with the `crit_engine_` prefix). A request
has one waiter; cancel may run concurrently. Close it after waiting, before closing
the engine. The Python bridge polls cancellation independently of token delivery,
so cancellation during prefill reaches Core. Physical work still settles before
release.

Token log-probabilities are waited on with `crit_engine_request_wait_output`.
A request opened with `"logprobs": n` (the model declaring `token_logprobs` and
`max_top_logprobs` of at least `n`, both in `crit_engine_info`) is waited on with it: its
callback receives each published token with its log-probability and its `n` most likely
alternatives as two ranked arrays of ids and log-probabilities, the
[TokenLogprob](#log-probabilities) rows of `Handle::wait_output`.
`crit_core.native.ResidentEngine` serves them as `chat-logprobs-v1` when `n` can be 20.

Cancellation predicates passed to the Python bridge must be thread-safe and
nonblocking: an independent monitor may call them while token delivery runs.
The native serving integration fixtures exercise real HTTP clients through the
stock resident adapter, including simultaneous streaming and buffered replies,
memory recovery and a connection closed during prefill.

`crit/native_engine.h` exports an engine as a shared library with a small C ABI,
so crit's Python tools reach it without engine-specific glue. Write a factory
that builds the engine from a JSON configuration and export it once:

```cpp
#include <crit/native_engine.h>

crit::core::NativeEngine open_engine(std::string_view configuration_json) {
    crit::core::NativeEngine native;
    native.owner = /* backend, weights and anything else the engine borrows */;
    native.engine = std::make_unique<crit::core::Engine>(profile, backend, factory);
    native.info = [] { return std::string(R"({"speculative": true, "replay": true})"); };
    return native;
}
CRIT_EXPORT_ENGINE(open_engine)
```

After the factory returns successfully, the native bridge pins `owner` to the
runtime. It survives outstanding request handles and a failed final drain that
quarantines the runtime; ordinary cleanup releases it after the model and backend.
Work launched during construction must already have its owners in the backend or
model, as described in the [construction contract](../../contracts/README.md#construction-and-borrowed-backend-handles).

The configuration is the adapter's configuration plus `model` (the model
directory) and `context_tokens` (the context to serve); Eval also passes
`speculative`. Link the static `crit::core` into the library (it builds
position-independent). The ABI (version 2) opens and closes engines, reports
`ModelInfo` merged with `NativeEngine::info`, generates with stop tokens,
sampling options and a replay continuation, streams tokens to a callback that
can cancel, and scores with optional query rows, the engine's own top-k and a named
path. `crit_core.native` refuses a library of another ABI version: rebuild it against
this crit's headers. Each
generation's result carries Core's [request timing](#request-timing) in
milliseconds, merged with `NativeEngine::request` (per-request facts such as
`accepted`, `verify_ms` and `draft_ms`). Eval counts one round per `accepted`
entry; report `rounds` when target passes differ from that, because Core's
`decode_steps` also counts steps without a target pass (seeding a drafter, for
example). Errors return -1 with the message from
`crit_engine_last_error()`; nothing throws across the boundary.

A generation with `"replay"` passes the continuation to the session as a
`ReplayContinuation` in `RequestOptions::context`: accept a draft while it equals
the continuation's next token and take every correction or bonus token from it,
so every engine decodes the same text. Declare `"replay": true` in the info facts
only when the session implements this. A generation with `"plain": true` passes a
`PlainDecoding` instead: decode one token per target pass without drafting. Eval
then checks exactness on the loaded engine rather than opening a second one;
declare `"plain": true` when the session implements it.

`crit_core.native` ([Python](../../python/README.md)) loads such a library and
supplies the stock adapters: `eval_engine(context, library, defaults)` for
[eval-v1](../../../docs/eval/ADAPTER.md) and `resident_engine(context, library,
defaults)` for crit serve's [chat-resident-v1](../../../docs/SERVING.md#adapter-contract).
The [toy engine fixture](../../../tests/core/native_engine_fixture.cpp) and its
[check](../../../tests/core/native_engine_check.py) exercise both.
