# Native core contracts

The authoritative native interface is [crit/core.h](../include/crit/core.h)
with the parts it includes from [crit/core/](../include/crit/core);
the current source version is 0.20.0. [CONTRACTS.json](CONTRACTS.json) records
its ownership responsibilities and those of the
[component libraries](#component-libraries); it is documentation, not a wire ABI.

Model/Session supply computation and logical state. The core owns the worker,
bounded admission, resource accounting, commit/publication, cancellation and
retirement. Every model callback returns with its work done. An optional
[mixed batch quota](../src/runtime/README.md#mixed-token-work-batches)
lets one `execute()` group hold prefill and decode rows; reported token work must
fit every row's grant. Scoring uses an
[exclusive FIFO barrier](../src/runtime/README.md#scoring).

Borrowed progress storage stays valid until commit. The core copies and charges
accepted tokens and publishes only after successful model commit. A grouped
failure may have changed state: the core fails the engine and publishes no rows
from that group; earlier output stays valid, every session is destroyed and every
request completes with the error
([retirement and failure](../src/runtime/README.md#retirement-and-failure),
[request release](../src/runtime/README.md#request-release)).

[Log-probabilities](../src/runtime/README.md#log-probabilities) come from the raw
logits of the pass that produced or scored the position, normalized over the
valid vocabulary, and are trimmed and published with their tokens. Ids at or above
`ModelInfo::valid_vocabulary` never appear in output, alternatives, `logit_bias`
or queries. [`wait_output`](../src/runtime/README.md#output-events) delivers one
event per published token.

## Speculative rounds

A speculative model runs a whole round (draft, verify, accept on the device) in
one `decode()` or mixed `execute()` row and returns all accepted tokens. Mixed
execution charges draft, target and padded columns against the row's grant and
falls back to ordinary decoding for small grants. `commit(accepted)` publishes
only what the device chose, without host re-verification.

## Component libraries

Component headers add libraries, not runtime obligations; sessions call them:

| Library | Contract |
| --- | --- |
| [Sampling](../src/sampling/README.md) | Sessions apply request sampling by Core's rules, with the draw of `crit/sampling.h`; the runtime never samples. Draws depend only on seed, position, stream and purpose, so rollback needs no generator state. |
| [Batched steps](../python/crit_core/batch.md) | Decode rows first, then prompt chunks; admission reserves every row a request can stage. |
| [Step engines](../src/runtime/README.md#step-engines) | Callbacks run on the engine worker and finish their work before returning; a failing callback fails the engine. |
| [Files](../src/io/README.md) | Strict JSON and read-only files that verify a file did not change while it was read. |
| [Structured output](../src/constraints/README.md) | A request-private `TokenMatcher` follows accepted output; a dead end fails the request, never falls back to unconstrained output. |

The Python package [`crit_core`](../python/README.md) applies the same stop,
decoding and model-file rules.

## Optional cached execution and source ownership

These rules apply only when a consumer uses the corresponding Inference
techniques; an engine can integrate without expert caching, mapped host sources
or CUDA graphs. A consumer that caches experts owns descriptors, admission,
replacement, source-aware projection and exact miss fallback, and publishes a
bundle only after all its component copies settle under the same upload identity.

Retain staging, source mappings and destination owners through the consumer
fence; a failed enqueue requires retained owners. Moved
addresses or incompatible bindings cannot reuse captured topology. Drain readers
before unregistering a mapping. [Expert residency](../../inference/expert-residency/README.md#with-other-inference-entries)
maps these rules to Inference host code.

