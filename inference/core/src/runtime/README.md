# Request runtime

The runtime implements the Engine lifecycle: bounded concurrent requests,
rotating chunked prefill, grouped commit and publication, cancellation and
retirement. The public interface is [core.h](../../include/crit/core.h), which
includes its parts by concern from [crit/core/](../../include/crit/core); model
computation comes through the one `Model`/`Session` interface
([scheduling.h](../../include/crit/core/scheduling.h)), which the
[step engine](#step-engines) implements over a model plugin's callbacks. Every
callback returns with its work done, so Core has no device queue to drain and no
readiness to wait for. See [current contracts](../../contracts/README.md).

| File | Part of the runtime |
| --- | --- |
| [engine.cpp](engine.cpp) | Construction and model declaration checks, shutdown, request completion |
| [submit.cpp](submit.cpp) | Request option checks, storage accounting, `Engine::submit` and `submit_score` |
| [admission.cpp](admission.cpp) | Admission, first in, first out |
| [batches.cpp](batches.cpp) | Step groups (decode, prefill, mixed token work), group publication |
| [worker.cpp](worker.cpp) | The worker loop, retirement, engine failure |
| [scoring.cpp](scoring.cpp) | Scoring, query scoring and the scoring barrier |
| [progress.cpp](progress.cpp) | Progress and log-probability checks, the trim rule and output staging |
| [handle.cpp](handle.cpp) | Request handles: waiting, streaming sinks, release of a consumed request |
| [interfaces.cpp](interfaces.cpp) | Defaults of the `Model` interface |

[engine.h](engine.h) declares the engine's private state (`Engine::Impl`) and
[request.h](request.h) its request record and the request rules these files share.

`Profile.max_concurrency` and `ModelInfo.max_concurrency` admit up to
`max_request_concurrency` (4096) active generation lanes. This bounds scheduler
metadata, not device memory: the model must still admit each request's complete
memory entitlement.

Admission goes by [priority](#request-priority), the oldest first among equal
values: the next request, once deferred, holds the others until it is ready,
cancelled, failed or expired, and a score is always a barrier. Admission
decisions come from the model; Core does not guess KV costs, evict live state or
assume recomputation is safe.

## Request priority

Core decides which waiting request runs next, for every way an engine is
served (crit serve and crit Eval's in-process adapter alike). `RequestOptions::priority`
(a 32-bit signed integer, default 0; the native ABI's `"priority"`) orders the
generations waiting for a lane: of those queued before the first score, the
lowest value is admitted first, the oldest among equal values
([admission.cpp](admission.cpp), `next_waiting`). A running request is never
preempted. A score stays an exclusive barrier: no request overtakes it, it
overtakes none, and a score takes no priority (a nonzero one is a generation
option, which scoring rejects). `crit_core.native.next_request` states the same
rule for a server that holds requests before the engine takes them, so crit
serve's queue hands them over in the order the engine would run them.

How many requests the runtime holds and how long one may wait are Core's
settings too, which `crit_core.native.StepEngine` takes from the engine's caller,
not from the model plugin: `max_concurrent_requests` caps the requests held at
once, lanes first (`max_concurrency`), then waiting places (`max_pending`), so
crit serve's `--max-concurrency` is applied here; `pending_timeout` bounds the
wait for a lane (the step engine's `pending_timeout_ms`; none unless a caller
gives one, where `Profile.pending_timeout_ms` defaults to 30 s).

Submission checks the outstanding allowance before retaining payload memory.
Requests reserve their maximum accepted-token staging buffer at
submission, alongside prompt and result storage. Insufficient Core buffer
capacity rejects the new submission without failing accepted work.

## Model progress

Every row is checked for prompt progress, prefix reuse, prefill completion and
output grants. Reuse must precede any computed prompt token; newly computed
tokens must fit the prefill grant. Output requires complete prefill, and every
step must consume prompt tokens or produce output. Core also checks row
membership and mixed token-work accounting, and validates every row of a group
before staging any output.

**Prefill passes.** A cancel settles between steps, so a step must stay short,
and some work on a prompt pays off only over a long span of it: a model whose
layers' weights do not all fit on the device brings each layer's to the device
once for the whole span when it runs the span one layer after another, where
chunk after chunk it would bring every layer's again for each chunk. Such a
model declares `ModelInfo::prefill_pass_tokens`, the most prompt tokens a
request holds in a pass, and `prefill_pass_steps`, the most steps in a row its
prefill takes without consuming a prompt token. A step of a pass reports the
span in `Progress::pass_tokens` and consumes nothing; the step that ends the
pass reports the span's tokens as consumed, past that step's own grant by the
pass the row held. Core schedules, cancels and retires such a request as any
other between its steps, and a row that passes more steps than declared, or
consumes nothing without reporting a pass, breaks the progress contract. What
a pass is made of (layers, weights, where they live) is the model's alone.

## Request release

When `Handle::wait()` or `wait_output()` returns (with a result, a cancellation
or a rethrown error, including anything leaving a sink), or a handle's destructor
returns without a wait, every host charge of that request (storage, options
and staging) has been released by the consuming thread, so
`Engine::usage(Memory::host)` read there afterwards excludes them. The request's
options and context are destroyed on that thread too.

An engine failure gives every request the same exception object; the last
consumer destroys it. ThreadSanitizer against an uninstrumented libstdc++ can
report false-positive races on that shared exception.

## Retirement and failure

A request that ends (length, stop or cancellation) is retired in the step that
ended it: `Model::retire(key, session, finish)`, then its session is destroyed,
then its waiter sees the result. A throwing callback, or a row that breaks the
progress contract, fails the whole engine: nothing of that group is published,
every session is destroyed without `retire`, every request gets the error and
later submissions are refused.

## Mixed token-work batches

Set `Profile.max_batch_tokens` above zero and declare `ModelInfo.mixed_batches`
to use `execute(span<WorkItem>)`. Zero keeps the separate
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
remain as small as one.
Reused prompt state costs no units. Core checks every row before
commit.

Token work is an admission quota, not equal time; budgets need measurement per
workload. The adapter supplies grouping, kernels, reservations and work
accounting. Core publishes a group only after its commit.

## Request options

`submit(prompt, output_tokens, options)` takes a value-owned `RequestOptions`.
Unset sampling fields use the adapter's defaults. `crit serve` forwards explicit
per-request overrides from packages declaring the corresponding
[sampling capabilities](../../../../docs/SERVING.md#per-request-sampling-in-cores-adapter),
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

The runtime passes values through; the session applies them by Core's
[sampling rules](../sampling/README.md), normally on the device.

`options.stop_tokens` replaces the model's stop list for this request (unset
inherits, empty disables). Core trims at the first stop, keeps that token and
reports `Finish::stop`. Stop strings and reply text are the Python side's
([crit_core.text](../../python/crit_core/text/README.md)).

For extra parameters, derive an immutable host-only `RequestContext`, set
`options.context` and declare `ModelInfo.request_context`. `host_bytes()` must be
fast and constant; Core charges it per request even when shared.

The model reads `RequestView::options`, valid until its session is destroyed.
Scoring rejects every generation option.

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
like `Progress::tokens`. Before commit Core checks exact counts, values that are not NaN or
+inf and at most `1e-3` above zero (-inf is zero probability), and alternatives
with distinct valid ids in non-increasing order (ties by ascending id). A
violation fails the engine with nothing from that group published. `max_top_logprobs` is at most `top_logprobs_limit` (64).

## Output events

`Handle::wait_output(sink)` waits like `wait(TokenSink)` and calls the sink once
per published token, in order, on the waiting thread. Each `OutputEvent` holds
the token's index and id, and its log-probability and alternatives when requested. Spans are valid only during the
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

Scoring joins the same FIFO as generation but is an
exclusive barrier: earlier generation retires before the hook runs and later
generation cannot pass it. Scores run sequentially through
`Model::score_request()` without a generation session, so long scores delay later
work. A scoring failure fails the engine. The
adapter reserves its returned score allocation through `Resources`, keeping it
until the next ordinary request callback or model destruction; Core copies the
validated scores into its own result.

## Request timing

Every `Result` carries `timing` (`RequestTiming`), host steady-clock phases:

| Field | Interval |
| --- | --- |
| `queued_ns` | Submission to admission |
| `prefill_ns` | Admission to the step that completed prefill |
| `first_token_ns` | Admission to the first published token |
| `decode_ns` | Prefill completion to the last published token |
| `total_ns` | Admission to completion |
| `prefill_tokens`, `decode_steps` | Prompt tokens the model reported; steps after prefill |
| `reused_prompt_tokens` | Of `prefill_tokens`, those restored from retained state |

Phases that did not occur stay zero. Times are taken when a callback returns, so they
include its device work. Divide published tokens by `decode_ns`, or prefill tokens
minus `reused_prompt_tokens` by `prefill_ns`, for rates without instrumenting the
adapter.

## Retained prompt prefixes

A model that keeps sequence state between requests reports the prefix it did not
recompute in `Progress::reused_prompt_tokens`, as part of `prompt_tokens`. Reuse
must precede any computed prompt token and does not count against the prefill
budget. The model alone decides whether retained state matches; Core validates
counts. A model plugin keeps such sequences with Core's prompt cache
([crit_plugins.prompt_cache](../../python/crit_plugins/prompt_cache.py),
[the prompt cache](../../../../docs/PLUGINS.md#the-prompt-cache)), and a request says
whether it may be served from it (`StepRequest.prompt_cache`).

## Engine ABI

An engine is never a library of its own: every engine is a plugin package that Core's loader
(`crit_core.engine.load`, [crit plugins](../../../../docs/PLUGINS.md#the-engine)) assembles and runs on Core's
[step engine](#step-engines). `crit/native_engine.h` is the C ABI of that engine, which the `crit_core_steps` shared
library exports ([step_library.cpp](step_library.cpp)) and `crit_core.native` reaches through ctypes: it opens an
engine with the model plugin's step callbacks (`crit_engine_open_steps`, `crit_engine_open_steps_rounds`), reports
`ModelInfo` merged with the model's facts (`crit_engine_info`), runs requests and scores, and closes it. Errors return
-1 with the message from `crit_engine_last_error()`; nothing throws across the boundary. `crit_core.native` refuses a
library of another ABI version (10): rebuild it against this crit's headers.

A request is opened with `crit_engine_request_open(engine, prompt, count, output_tokens, options)` and waited on
with `crit_engine_request_wait` (tokens to a callback that can cancel) or `crit_engine_request_wait_output`; it has
one waiter, `crit_engine_request_cancel` may run concurrently, and it is closed with `crit_engine_request_close` after
waiting, before the engine closes. The Python bridge polls cancellation independently of token delivery, so
cancellation during prefill reaches Core; physical work still settles before release. Cancellation predicates passed
to the bridge must be thread-safe and nonblocking: an independent monitor may call them while token delivery runs.
Its options are the request's sampling fields, `stop_tokens` (`[]` disables them), `replay` (a continuation the
session decodes by, so every engine decodes the same text), `plain` (one token per target pass, no drafting),
`decode_via_prompt` (a debug request whose decode rows the model computes as one-token prompt chunks, never in a
native round: crit smoke's [rounding or fault](../../../../smoke/README.md#rounding-or-fault)) and
`logprobs` (alternatives per token): `crit_engine_request_wait_output`'s callback receives each published token with
its log-probability and its most likely alternatives as two ranked arrays, the [TokenLogprob](#log-probabilities) rows
of `Handle::wait_output`. `crit_engine_request_wait`'s callback receives the tokens published since its last call
together (a step's or a native round's, `Handle::wait_tokens`), and a null callback lets the waiter sleep until the
request ends, never woken per token (`Request::streaming`). Each generation's result carries Core's [request
timing](#request-timing) in milliseconds, merged with its request facts ([step engines](#step-engines): `rounds`,
`accepted`, `drafted`, `expert_reads`, `expert_hits`, and the model's own, such as `verify_ms` and `draft_ms`).
`crit_engine_score` scores fixed text with optional query rows, the engine's own top-k and a named path.

Core's adapters serve that engine: `crit_core.engine.resident` for crit serve's
[chat-resident-v1](../../../../docs/SERVING.md#cores-resident-adapter), which exposes `max_concurrency + max_pending`
requests at once with a private parser wrapper per request, and `crit_core.engine.eval_v1` for
[eval-v1](https://github.com/crit-labs/crit-eval/blob/main/docs/ADAPTER.md) (`crit eval run --package DIR`), which
passes each generation's `prefill_ms` and `first_token_ms` and scores the fixed quality texts as crit serve scores
them (the prompt by query scoring on the `prefill` score path, the continuation by a greedy replay with
log-probabilities), so `--only quality` checks quality in process during development; the gate is `crit run ...
eval`'s, through crit serve.

### Step engines

[crit/step_engine.h](../../include/crit/step_engine.h) makes a `Model` out of C callbacks, for the model
plugin, whose code is in Python. Core keeps the scheduler with [mixed batches](#mixed-token-work-batches), admission,
request options, stop tokens, output limits, cancellation, retirement and the [scoring](#scoring) barrier; the model
plugin implements a step over one row per scheduled request ([step_engine.cpp](step_engine.cpp)).
`crit_core.native.StepEngine(model, library)` opens it on `crit_core_steps` (`library`, else
`CRIT_CORE_STEP_LIBRARY`) and serves `max_concurrent_requests` (`max_concurrency + max_pending`) requests at once;
Core's loader is what constructs one, over an assembly whose model plugin it checked can batch:

| Model method | Called with | Returns |
| --- | --- | --- |
| `info()` | | Facts plus Core's settings: `vocabulary` and `context` (required), `max_concurrency` (2 or more), `max_pending`, `max_batch_tokens` (mixed batches; 0 separates prefill and decode steps), `prefill_chunk`, `prefill_work_quantum` (prompt work granted in its multiples: the prefill chunk keeps a prompt's chunk boundaries whatever shares its steps, which a model whose chunked kernels give other bits at other boundaries declares), `prefill_pass_tokens` and `prefill_pass_steps` ([prefill passes](#model-progress): the most prompt tokens a request holds in a pass, and the most steps in a row its prefill takes without consuming a prompt token), `batch_policy`, `max_row_tokens` (most tokens a row reports), `sampling`, `token_logprobs`, `max_top_logprobs`, `scoring`, `score_paths` |
| `open(request)` | `StepRequest`: key, prompt, output limit, set sampling fields, effective stop tokens, `logprobs`, `replay`, `plain`, `decode_via_prompt` | A session object |
| `step(rows)` | One `StepRow` per scheduled request: session, prefill grant, output allowance, token work, prefill complete, and a constrained request's `allowed` | One `StepResult` per row: prompt tokens consumed, tokens produced, prefill complete, reused prefix, token work, log-probability rows, for a decode row the drafts its round verified (`drafted`), and the prompt tokens the row holds in a prefill pass after the step (`pass_tokens`) |
| `commit(decisions)` | Core's `StepDecision` per row of the last step (tokens kept, terminal, cancelled) | |
| `retire(session, finish)` | `length`, `stop`, `cancelled` or `failed` | |
| `request_facts(session)` | Optional, as the session retires | A dict of the request's facts (`verify_ms`, `draft_ms`, the expert reads and hits of rounds it ran in `step()`, ...) |
| `telemetry()` | Optional (Core's loader passes the assembly's) | The engine's work so far, by section (`StepEngine.telemetry()`, with Core's `core` step counters) |
| `admission(request)` | Optional | `ready`, `deferred` or `infeasible` |
| `score(ids, first_target, path, queries, top_k)` | A fixed-text score, as an exclusive barrier | `Scores` |

A constrained request (`StepEngine.generate(..., constraint=...)`: json mode, a compiled grammar, or a fixed token
set) reaches the runtime as a host attachment (`HostAttachment` in [native_engine.h](../../include/crit/native_engine.h):
the options' `attachment` id with `constrained`): Core decodes it plainly, one token a row, and on the Python side
keeps its matcher ([structured output](../constraints/README.md), through the exports of
[constraint_library.cpp](constraint_library.cpp); `StepEngine.constrain(vocabulary)` gives Core the model's byte
vocabulary). Before each step its row carries the tokens the matcher allows next (`StepRow.allowed`, a
`crit_plugins.Allowed`); after it Core checks the token the row produced and appends it once `commit` keeps it. A
token outside the mask, a second token, or a constraint that cannot continue fails that request alone with
`NativeError`, as a cancel whose tokens are never published. In [native rounds](#native-decode-rounds) that mask, a
constrained request decodes as the others do, its mask staged between rounds.

Every generation's result carries its request facts. Core counts each decode row after prefill as one round
(`rounds`): its accepted drafts (a native round's telemetry word, else the row's tokens less one) and drafts (a native
round's width, else the row's `StepResult.drafted`) go into `accepted` and `drafted` when the request drafted
(`drafted` only when every round's is known), and a native round's expert reads and hits into `expert_reads` and
`expert_hits`. With the model's `request_facts(session)` (the configuration's `request_facts` setting and the
callbacks' `facts`, which Core reads only then) the model's facts go over Core's (`crit_core.native.merge_facts`): its
expert reads and hits add to the native rounds', its other keys replace Core's. Core's own Python work per step is a
few microseconds: the rows, progress and decisions cross as packed bytes, and Core's worker thread keeps its Python
thread state, which a callback from a thread Python did not start would make and free every time.

Every method runs on Core's one engine worker, never two at once, so a model needs no locks of its own. Core trims
stop tokens and the output limit and publishes only what it kept; the model keeps its state in step with `commit`.
[crit_core.batch](../../python/crit_core/batch.py) lays a step's rows out ([batched steps](../../python/crit_core/batch.md)),
so a model whose arithmetic is row local decodes the same tokens batched as alone. A failing callback fails the
engine, with the Python exception in every request's error. Under the Windows display driver (native Windows and
WSL) the engine warns once per device when torch reserved more than the device holds, since the driver then runs the
excess from system memory. The [step engine check](../../../../tests/core/step_engine_check.py) drives a toy step
model: batched and serial runs equal, mixed steps, cancellation and retirement, page-reserving admission,
log-probabilities, replay, stops, scoring and a failing step. [test_core_engine.py](../../../../tests/test_core_engine.py)
runs the toy engine's plugin package through Core's loader and adapters, and the
[concurrency end-to-end check](../../../../tests/core/concurrency_end_to_end_check.py) serves it through `crit serve` four
requests at once and measures it with `crit run measure` at concurrency 4 and with crit Eval's concurrency part: the
aggregate grows with the requests in flight, the peak equals the clients and batched replies equal serial ones.

#### Native decode rounds

A model plugin's decode step is mostly host work around a few graph launches: reading the round's
block, building results, Core's commit callback, the next launch. While the interpreter does that
work the GPU idles. With `StepRounds` (`crit_engine_open_steps_rounds(configuration, callbacks, rounds)`;
`crit_core.native.NativeRounds` from a model's `native_rounds(library)`), Core runs a decode step
itself whenever all its rows are decode rows of sessions the engine bound to slots 0 .. rows - 1 and the
engine gives that row count a width. Core then launches the engine's round (`launch(rows, width, logprobs)`,
a captured graph that drafts, verifies, accepts on the device and copies its block to host memory), waits
(`wait`), and reads each row's tokens from the block (count, matched drafts, frontier, tokens, and from
entry `telemetry`, when the engine names one, the row's telemetry words: accepted drafts, routed expert
reads and expert cache hits, which go into the request's facts). A round has two forms: Core launches its
log-probability form (`logprobs` 1: it writes the rounds' log-probability buffers, which Core gives the
requests that ask, and runs rows that follow a continuation of their own) when a row's request asks for
log-probabilities or replays a continuation, else its plain form, which costs less; `runs` counts the rounds
of each. It also skips the engine's `commit` unless a decision trims, ends or cancels a row (`last`
names the round then). Every other step (prompt chunks, a row outside the slots, a constrained request where the
rounds do not mask, one asking for more alternatives than the rounds rank) goes to `step()` as before, and the engine binds and unbinds slots in its own
`step`, `commit` and `retire`. Core reads a row count's width (`widths`, the engine's own array) before
each round, so the engine may change it between steps: a draft-width policy whose captured round copies
the next round's widths into that array chooses each round's width with no host code in between. Every
width stays within `widest`, the widest round, which Core checks the engine's grants against when it
opens; a row count without a width at open stays on `step()`.

Rounds that mask (`StepRounds::masks`; `NativeRounds(stage=...)`, which the round driver passes when its
sampler has `stage_mask`, token-constraints' `rounds:sampler`) run constrained requests too. A round masks
every logits row of a constrained slot by the allowed set staged for it and cuts its drafts, so the slot
emits one token a round (width 0) from that set while the other rows of the round draft as they do. The
set of round n + 1 depends on round n's token, and nothing runs on the host inside a round, so Core stages
it between the two: after every native round that ran a constrained row Core calls `commit` (it skips it
otherwise), and there `crit_core.native` reads the slot's token from the block, checks it against the set
staged, appends it to the request's matcher, lets the model commit, and stages the matcher's next set for
the slot (`stage(slot, allowed)`: one copy of the bitmask words, vocabulary / 32 words, on the rounds'
stream) before Core launches the next round. A set is also staged after the `step()` that binds a
constrained request and after a `retire` that moves one to another slot. A constrained request so costs
every round it shares one host hop: the commit callback, a matcher step (a walk of the vocabulary's piece
trie, or its allowed-set cache) and the copy, on the critical path of every row of the round, where an
unconstrained round goes from one launch to the next with no Python. No pipelining hides it: the set is
not known before the token that decides it. A request whose constraint failed is staged nothing more and
keeps its last set until Core ends it, its tokens never published. Whether constrained decoding comes
within 10% of an unconstrained width-0 request's speed is a GPU measurement still to make.

`crit_rounds_cuda_*` (exported by `crit_core_steps`, `NativeRounds.cuda`) is the launcher over captured
CUDA graphs. It loads the CUDA driver API at run time, so Core still builds without a CUDA toolkit, and
it replays a framework's graph executable (`torch.cuda.CUDAGraph.raw_cuda_graph_exec()`) on its
stream. Core's round driver ([crit_plugins.rounds](../../python/crit_plugins/rounds.py)) builds it from
the rounds a model plugin captured: `Rounds` composes the library's device forms (the drafter's
`draft:device`, the model's verify over a `RoundBatch`, the sampler's `rounds:sampler` predictions, each
pool's `commit_round`) into one captured round per (rows, width, mode) and form, and `SpeculativeRounds` runs the same
round over an engine's own hooks. The [step engine check](../../../../tests/core/step_engine_check.py) runs a toy
model whose round is a host callback: the oracle's tokens, no `step()` for decode rows, a commit only for rows
that end, a row count without a width decoded through `step()`, settings that cannot hold the rounds refused at
open, constrained requests in rounds that mask with `step()`'s tokens (a set staged before every round, after a
move too, and a round's token outside it failing its request alone), each round's form from its requests (plain for plain requests, the log-probability form's values for a
request asking and a replay's continuation, a plain request beside them decoding its own tokens), and each
request's facts from the block's words (accepted drafts, the engine's `round_counters`) with the model's merged in
round by round.
[test_python_rounds.py](../../../../tests/test_python_rounds.py) (`NativeRoundTests`, CUDA) serves
`SpeculativeRounds` over captured graphs through Core, and [test_core_rounds.py](../../../../tests/test_core_rounds.py)
the composed `Rounds`: no `step()` call for decode, the same tokens as the engine's own Python rounds and
plain greedy decoding, and every round in the request's facts.
