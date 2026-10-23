# Native core contracts

The authoritative native interface is [crit/core.h](../include/crit/core.h);
the current source version is 0.5.0. [CONTRACTS.json](CONTRACTS.json) records
its ownership responsibilities and those of the
[component libraries](#component-libraries); it is documentation, not a wire ABI.

Model/Session supply computation and logical state. The core owns the worker,
bounded admission, resource accounting, readiness, commit/publication,
cancellation and retirement. Concurrency is opt-in through both Profile and
ModelInfo. An optional [mixed batch quota](../src/runtime/README.md#mixed-token-work-batches)
lets one `execute()` group hold prefill and decode rows; reported token work must
fit every row's grant. Neither mixed batches nor admission backfill change
publication order or lifetime obligations. Scoring on concurrent models uses an
[exclusive FIFO barrier](../src/runtime/README.md#scoring).

Borrowed progress storage stays valid until commit. The core waits readiness,
copies and charges accepted tokens and publishes only after successful model
commit. A grouped failure may have changed state: the core poisons the request
infrastructure and publishes no rows from that group; earlier output stays valid.
When drain succeeds, a failed request completes after session closure and release
of its reservations; unproved settlement retains the owners and poisons the
runtime ([request release](../src/runtime/README.md#request-release)).

Every asynchronous producer belongs in cleanup. Timeouts do not prove physical
completion. Public CUDA services use explicit compute/transfer ordering and
last-owner physical release.

[Lifecycle tracing](../src/runtime/README.md#lifecycle-tracing) is bounded and
never makes writers wait; loss is explicit and tracing changes no obligations.

An independent [text frontend](../src/runtime/README.md#text-services) allows
concurrent tokenization while the worker generates and must retain its tables
independently of the engine. [Request text and stop strings](../src/runtime/README.md#request-text-and-stop-strings)
require such a frontend: Core decodes accepted tokens before commit and trims stop
tokens, the output budget and then the first completed stop string. A decoder
failure fails only that request.

[Log-probabilities](../src/runtime/README.md#log-probabilities) come from the raw
logits of the pass that produced or scored the position, normalized over the
valid vocabulary, and are trimmed and published with their tokens. Ids at or above
`ModelInfo::valid_vocabulary` never appear in output, alternatives, `logit_bias`
or queries. [`wait_output`](../src/runtime/README.md#output-events) delivers one
event per published token.

## Speculative rounds

A speculative adapter runs a whole round (draft, verify, accept on the device) in
one `advance()`, `decode()` or mixed `execute()` row and returns all accepted
tokens. Mixed execution charges draft, target and padded columns against the
row's grant and falls back to ordinary decoding for small grants. Readiness is
exactly one device-to-host readback; `commit(accepted)` publishes only what the
device chose, without host re-verification. When only the readback reveals the
round's length, `advance()` returns `deferred_tokens` and Core reads
`ready_tokens()` after the wait. `close()` must settle a launched, uncommitted
round. Stage timings go through `device_timings()` as trace `device` events.

## Component libraries

Component headers add libraries, not runtime obligations. The runtime uses
[crit/text.h](../include/crit/text.h) and [crit/trace.h](../include/crit/trace.h);
sessions call the others:

| Library | Contract |
| --- | --- |
| [Sampling](../src/sampling/README.md) | Sessions apply request sampling with `crit/sampling.h` or a device path that follows its rules; the runtime never samples. Draws depend only on seed, position, stream and purpose, so rollback needs no generator state. |
| [State cache](../src/state/README.md) | Page and slot bookkeeping; the engine owns storage and copies; ids are reused only after their epoch retires. |
| [Files](../src/io/README.md), [model checkpoints](../src/model/README.md) | Readers that never decode tensor values; weight encodings and loading stay the engine's. |
| [Operator ranges](../src/runtime/README.md#operator-ranges) | Compile out without `CRIT_TRACE_ENABLED`; bounded, add no device fences; work counts are estimates. |
| [Structured output](../src/constraints/README.md) | A request-private `TokenMatcher` follows accepted output; a dead end fails the request, never falls back to unconstrained output. |
| [Host-memory admission](../src/memory/README.md#host-memory-admission) | Evidence before allocation, not a guarantee. |
| [Device graphs and shared allocations](../src/backends/cuda/README.md#graph-capture) | Any failure after graph submission poisons replay; unproven settlement retains owners and charges. |

The Python package [`crit_core`](../python/README.md) applies the same stop,
decoding and model-file rules.

## Construction and borrowed backend handles

A failing model factory still causes backend drain. Construct and validate owned
state before publishing a borrowed handle to the backend (a derived destructor
does not run when its constructor throws), and make drain tolerate a handle that
was never published. Keep the borrowed target alive through drain and unregister
it after settlement. The host regression `borrowed_startup` covers this pattern.
See [building engines](../../docs/BUILDING_ENGINES.md) and
[Inference contracts](../../inference/README.md).

## Optional cached execution and source ownership

These rules apply only when a consumer uses the corresponding Inference
techniques; an engine can integrate without expert caching, mapped host sources
or CUDA graphs. A consumer that caches experts owns descriptors, admission,
replacement, source-aware projection and exact miss fallback, and publishes a
bundle only after all its component copies settle under the same upload identity.

Retain staging, source mappings and destination owners through the consumer
fence; a failed enqueue requires retained owners and backend quiescence. Moved
addresses or incompatible bindings cannot reuse captured topology. Drain readers
before unregistering a mapping. [Expert residency](../../inference/expert-residency/README.md#with-other-inference-entries)
maps these rules to Inference host code.

Reuse-aware admission hints are bounded, advisory metadata; they cannot grant
memory or authorize cache reuse ([runtime policy](../src/runtime/README.md#reuse-aware-admission)).
