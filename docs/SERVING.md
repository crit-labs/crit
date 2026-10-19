# Serving an engine

`crit serve` loads one sealed candidate, keeps it resident and answers
OpenAI-compatible chat requests until stopped, so any OpenAI client (the `openai`
SDKs, LangChain, Open WebUI, curl) can use the engine. Packages implement the
[adapter contract](#adapter-contract).

## Start the server

After `python -m pip install .` (or `-e .` in a checkout):

```sh
crit serve --workspace /path/to/runs
```

With one run and no `--candidate`, the selected candidate loads and the server prints
`Serving RUN_ID/CANDIDATE_ID at http://127.0.0.1:8000/v1`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
reply = client.chat.completions.create(model="default", messages=[{"role": "user", "content": "Hello"}])
```

| Option | Purpose |
| --- | --- |
| `--workspace PATH` | Folder containing the `run-*` directories (required) |
| `--run ID`, `--candidate ID` | Run (default: the only run) and starting candidate (default: the selected one) |
| `--host`, `--port` | Listen address; default `127.0.0.1:8000` |
| `--api-key KEY` | Require `Authorization: Bearer KEY`; default `CRIT_API_KEY`. A non-local `--host` without a key prints a warning |
| `--cors-origin ORIGIN` | Allow browser requests from this origin, or `*` |
| `--model-name NAME` | Model ID reported for the default candidate |
| `--timeout SECONDS` | Limit per request; default 600 |
| `--load-timeout SECONDS` | Limit on loading the starting candidate; default 600 plus the model's bytes at 200 MB/s (50 MB/s with `--verify-model`) |
| `--verify-model` | Before loading, read and SHA-256 every model file against the run's frozen manifest, and give engines without `chat-model-file-v1` a private read-only copy. By default the host only checks each file's size against the manifest and maps it in place, so the engine starts loading at once; a checkpoint with the same sizes but other bytes (another revision, corruption) is then served unnoticed |
| `--resident-ram-gib N`, `--resident-vram-gib N` | Optional caps on pool-owned model bytes, not total process/device memory |
| `--project PATH` | crit checkout used to verify packages; default this checkout |
| `--write-launchers` | Write `serve.cmd` (Windows, double-click) and `serve.sh` with the same options into the run folder, then exit. Under WSL, `serve.cmd` starts `serve.sh` in the same distribution. They never contain the API key; set `CRIT_API_KEY` |
| `--recover-runtime` | Verify retirement of a crashed runtime, release ownership, exit |

Ctrl+C stops accepting requests, unloads the engine and releases model memory.

The server holds the machine's engine lock while it runs (`crit_core.machine`): `crit eval run`, `pin`, `ab`,
`kernels`, `reference` and `crit run measure` wait for it, and it waits for them, printing who holds the lock. Two
engines side by side can exhaust host RAM when weights live there. `CRIT_ENGINE_LOCK` names the lock file (default
`crit-engine.lock` in the system temp directory), `CRIT_ENGINE_LOCK=off` disables it. Since file locks do not
cross OS instances, the holder also keeps a lease that Windows and its WSL distributions both read
(`CRIT_ENGINE_LEASE`, default `C:\ProgramData\crit\engine.lease`, `/mnt/c/ProgramData/crit/engine.lease` under WSL).

## Endpoints

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | `{"status": "ok", "engine": ...}`; 503 when the runtime is blocked. No key needed |
| `GET /v1/models`, `GET /v1/models/{id}` | The run's sealed resident candidates, with `label`, `context_length` (the candidate's `kv_tokens`), `default`, package `capabilities` and `supported_sampling_parameters` |
| `POST /v1/chat/completions` | Chat completion, JSON or server-sent events |

`model` selects a candidate by `RUN_ID/CANDIDATE_ID` or `CANDIDATE_ID`; any other name
(`"default"`, `"gpt-4o"`) uses the default. An unknown candidate returns 404, one that
fails verification 409. Switching closes the old engine and loads the new one, reusing
compatible model allocations. Requests run one at a time.

| Field | Behavior |
| --- | --- |
| `messages` | Optional leading `system`/`developer` messages (merged into one `system`), then user/assistant turns ending with user. Consecutive same-role turns are joined with a blank line. Content is a string or a list of `text` parts |
| `max_tokens`, `max_completion_tokens` | Output cap, 1 to `context_length - 1`; default 1024 or less to fit |
| `stream`, `stream_options.include_usage` | SSE: role chunk, text chunks, finish chunk, optional usage chunk, `[DONE]` |
| `stop` | Up to 4 strings of at most 256 UTF-8 bytes. Text before the first completed stop returns with `finish_reason: "stop"` (`crit_core.text.StopMatcher`); the turn is cancelled, the engine stays loaded |
| `n` | Must be 1 |
| `response_format` | `text`, `json_object` or `json_schema` ([structured output](#structured-output)) |
| `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias` | Per-request overrides when the package declares support for that field; otherwise 400. Omitted or null fields keep engine defaults |
| `tools`, `tool_choice`, `parallel_tool_calls` | Function calls for packages declaring `chat-tools-v1`; `auto`, `none`, `required` or a named function (adapters that parse free generation, such as [ResidentChat](#tool-calls-from-the-chat-template): `auto` or `none`). Tool-result history is preserved |

Responses carry `usage` when measured and `crit_telemetry` (phase tokens, ms, rates).
Sampling overrides are validated before generation, including for streaming requests.
Unknown parameters and unsupported generation controls return 400 rather than being
silently ignored. Only client metadata (`user`, `metadata`, `store`, `service_tier`)
is ignored and listed in `X-Crit-Ignored-Parameters`; no storage or service tier is applied.
`logprobs`, audio, legacy `functions`/`function_call`, and an unsupported `response_format` return 400. Errors use the
OpenAI shape; timeout is 504, engine failure 500 (the next request reloads). A
disconnecting client cancels its turn.

This changes clients that previously sent ignored sampling values to old packages:
omit those fields to retain the frozen defaults, or build a new package with an
adapter that applies and declares them. Never edit a sealed package's manifest.

### Tool calls

`chat-tools-v1` supports up to 128 function definitions with unique names and up to
128 calls per reply. Assistant history carries `tool_calls` with JSON-string
`function.arguments`; subsequent `role: "tool"` messages carry the matching
`tool_call_id`. All calls need exactly one result before the next turn. A request
may end with the last tool result. The API never executes tools itself.

Responses contain `message.tool_calls` and `finish_reason: "tool_calls"`; a tool-only
message has `content: null`. With `chat-tool-stream-v1`, SSE returns each call's
`index`, id, type and complete name once, followed by incremental argument fragments.
Multiple calls may interleave by index. Without that capability, tool requests are
buffered until the adapter finishes, then emitted as a single tool-call delta.
`parallel_tool_calls: false` permits at most one call. Unknown functions, invalid JSON arguments and replies
violating `tool_choice` fail rather than being exposed as usable calls.

`strict: true` requires `chat-tools-strict-v1`. The function's `parameters` must be
an object schema in Core's [supported subset](#structured-output), including
`additionalProperties: false`. The server compiles it before generation and checks
completed arguments against that grammar. Strict arguments use compact JSON in the
schema's declaration order, as with `response_format`. Unsupported schemas return
400 with `param: "tools"`, `code: "unsupported_schema"`; there is no unconstrained
fallback. Non-strict tools require JSON-object arguments but do not promise schema
conformance. A request may mix strict and non-strict functions.

Streaming deltas are provisional until `finish_reason: "tool_calls"`. The host
validates strict argument prefixes before publishing them and requires the final
calls to match the stream exactly. An incomplete or invalid final call produces an
SSE error and `[DONE]`, without a successful finish chunk. Clients should assemble
calls by index and only act on a successfully completed turn.

Tool requests cannot combine `stop` or a structured `response_format`.

### Tool calls from the chat template

Adapters built on `crit_core.serving.ResidentChat` (and Core's `resident_engine`) offer
tool calls without model-specific code in crit. ResidentChat renders the request's
`tools` (unless `tool_choice` is `none`) and the tool history with the model's own chat
template, splits the reply into content and calls with `crit_core.text.ToolCallParser`,
gives each call an id and returns them as `tool_calls`; with streaming each call arrives
as one delta when its block closes. `capabilities(text, sampling)` then declares
`chat-tools-v1`, plus `chat-tool-stream-v1` when the package streams. A block that does
not parse, names an unknown function or exceeds `parallel_tool_calls: false` stays in
the content as written. `tool_choice` must be `auto` or `none`, since the model is not
constrained to call; strict tools need an adapter with `chat-tools-strict-v1`.

The call syntax is the one the template teaches when crit registers it
(`crit_core.text.templates_tool_format`; crit registers only syntaxes several model
families share, such as Qwen3-Coder style `<function=...>` blocks and Hermes-style JSON
in `<tool_call>`). When a model writes another one, the engine package carries the
parser, never crit: set `ResidentChat.tool_format` to a `crit_core.text.ToolFormat`
(the parse function and markers) and declare `capabilities(text, sampling,
tool_format=...)`. Packages sealed with the earlier `chat-tool-template-v1` keep working:
crit serve reads it as `chat-tools-v1`.

### Reasoning

Reasoning models write their thinking between markers before the reply. crit serve
returns it apart from the reply, as OpenAI-compatible clients expect: the message's
`reasoning_content` in JSON replies, and `delta.reasoning_content` chunks, before the
content, when streaming. Adapters built on `crit_core.serving.ResidentChat` split it in
the markers the model's chat template uses (`crit_core.text.reasoning`; crit registers
the `<think>...</think>` syntax several model families share). Reasoning starts where the
reply starts, either because the generation prompt ends with the opening marker or
because the reply opens with it; a marker later in the reply is reply text, and a reply
cut before the closing marker is all reasoning. Tool calls are parsed from the reply
only. A package whose model writes other markers sets `ResidentChat.reasoning_format`
to a `crit_core.text.ReasoningFormat`, and `False` leaves the reasoning in the reply.

## Structured output

`response_format` restricts the reply's bytes with Core's
[structured-output service](../core/src/constraints/README.md). It needs a package
declaring `chat-constraints-v1`; otherwise it returns 400.

- `{"type": "json_object"}`: one JSON object (surrounding whitespace allowed).
- `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}, "strict": ...}}`:
  compact JSON in the schema's declaration order, compiled with `crit_core.constraints`.
  The [supported subset](../core/src/constraints/README.md#compiling-languages-python):
  closed objects (`additionalProperties: false`), typed arrays, strings with lengths,
  booleans, null, numbers, integers with both bounds, `enum`, `const`. Anything else
  (`$ref`, `anyOf`, `pattern`, `format`, open objects) returns 400 with
  `param: "response_format"`, `code: "unsupported_schema"` and the reason. Always strict.

The engine applies the constraint while sampling (the adapter interface carries no
logits). crit serve re-checks every reply the engine ended itself (`finish_reason:
"stop"`, no stop string) and answers 500 if it does not conform; the engine stays
loaded. Replies cut by the cap or a stop string are returned as cut. When streaming, a
failed final check becomes an error event.

## Adapter contract

A package is servable when it declares **`chat-json-v1` and `chat-resident-v1`** and
`launch[0]` is a retained Python module exporting `create_engine(context)`. Build the
adapter on **`crit_core.serving.ResidentChat`**: it renders the conversation, tools and
tool history with the model's chat template, forwards the request's sampling fields that
the engine lists in `sampling`, streams text without stop tokens, honors cancellation and
reports Core's phase telemetry, so the engine supplies only `generate_tokens(ids,
max_new_tokens, sampling, on_tokens, cancelled)`. Declare
`crit_core.serving.capabilities(text, sampling)`. A Core engine library exported with
`CRIT_EXPORT_ENGINE` can return `crit_core.native.resident_engine(context, library,
defaults)`, a ResidentChat over the library (see
[engine libraries](../core/src/runtime/README.md#engine-libraries)).

`crit run package` refuses a resident package that does not declare
`chat-sampling-temperature-v1`, or that does not declare `chat-tools-v1` while the
model's chat template teaches a tool-call format crit parses: agent clients send both on
every turn.
Serving works with failed or missing checks and
never changes benchmark evidence or passes a gate. [Eval](eval/ADAPTER.md) uses a
separate `eval-v1` adapter (`create_eval_engine(context)`); both can wrap one engine.

### Factory and lifetimes

`context.configuration` is the frozen configuration, `context.arguments` the remaining
literal launch arguments. Relative helper imports use a content-specific package
namespace. Load weights in the factory. The returned engine implements:

| Method | Contract |
| --- | --- |
| `generate(request, cancellation)` | Return the response dictionary. Poll `cancellation.is_set()` or call `cancellation.check()`. Enforce token/context/headroom/deadline limits. Write no response file |
| `reset()` | After every generation, including failure/cancellation: drain queues, graphs, borrowed pointers, writes and producers; reset turn state; keep weights |
| `close()` | Drain and release engine-private resources before switch/unload; crit releases model leases/views afterwards |

Calls are serialized; no concurrent generation. The client owns the conversation (each
request carries all of it); the engine owns tokenizer, private KV/recurrent state,
scratch and graphs until switch or failure; the model pool owns the mapped model files and
device weights until the server stops, one entry per identity/representation, no
eviction. A → B → A closes each engine before the next.

Packages share one process: release views, mutable globals and workers on close. Link
the CUDA runtime dynamically (shared `libcudart`); a static one failed with "invalid
argument" after another engine's had loaded. Construction, reset or close failure can
retire the runtime; a stuck adapter is killed after a one-second cancellation grace.

### Model access and allocation

| API | Behavior |
| --- | --- |
| `context.host("relative/model/file")` | Read-only memoryview of the model file, mapped in place after a size check (pageable). With `--verify-model`, of a hashed private copy (needs temporary disk) |
| `context.model_file(name)` | Native-loader path of the model file when `chat-model-file-v1` is declared; no copy |
| `context.limits` | Context and RAM/VRAM headroom; `kv_tokens` is the built context, `context` (when present) Eval's [basis](eval/CONTEXT.md) |
| `context.device(compatibility, nbytes, allocate, inputs=())` | Generation-bound lease; `lease.buffer` valid for the engine lifetime. `inputs` lists non-model files whose bytes enter the allocation |

Model identity is the full frozen path/size/SHA-256 manifest; treat model files as
immutable. Device keys describe the complete representation:

```python
compatibility = {
    "device": "backend/device-uuid/context-identity",
    "representation": "weight-bundle-and-transform-version",
    "dtype": "encoded-scalar-type",
    "quantization": "exact-format-and-parameters",
    "layout": "packing-and-strides-version",
    "shape": [1024, 1024],
    "abi": "backend-runtime-and-allocation-ABI-version",
}
```

crit adds the model manifest digest and each `inputs` file's content hash.
`allocate()` returns `crit_api.models.DeviceAllocation(buffer, nbytes, wait, free)` once
upload settles; charge actual bytes with padding, or `nbytes=None` if only the allocator
knows. `wait()` proves asynchronous uses settled; `free()` releases. A wait, free or
accounting failure quarantines ownership and retires the host; an ambiguous free is
never retried. Resident caps cover **pool-owned model bytes** only; the adapter budgets
everything else.

### Request

Fields: integer `schema_version: 1`, `record_kind: "chat_request"`,
run/chat/request/candidate IDs, canonical `package_sha256`, UTC creation time, model
locator and frozen manifest, package path, configuration, `limits`, and
`messages: [{role, content}, ...]` (optional first `system`, then alternating
user/assistant ending with user).

With `chat-tools-v1`, messages also preserve assistant `tool_calls`, tool messages
with `tool_call_id`, and optional `name`/`reasoning_content`; tool message boundaries
are not merged. The request also carries `tools`, resolved `tool_choice` and
`parallel_tool_calls` when tools were supplied. Use the model's tool-aware chat
template, apply the requested choice, and parse generated calls in the adapter
(ResidentChat does all three).
Return them as an optional top-level `tool_calls` list in the response record,
using OpenAI's `{id, type: "function", function: {name, arguments}}` shape with
JSON-string arguments. `assistant` stays a string and may be empty for tool-only
replies. Without `chat-tool-stream-v1`, a streaming adapter emits only assistant
text through `emit`, never raw tool syntax; return complete calls in the final record.
Tool arguments and text
share the 65,536-character reply budget, and telemetry counts all generated tokens.

- Apply the system message through the chat template. Use the exact model, tokenizer,
  template and sampling; verify the manifest, never substitute.
- `limits`: `max_output_tokens` (from `max_tokens`), `kv_tokens`, `vram_headroom_bytes`,
  `ram_headroom_bytes`, `timeout_seconds` (from `--timeout`). Tokenize the full templated
  input first; reject overflow, never truncate. Enforce the output cap.
- Paths are read-only; work in turn scratch. Changed artifacts are revalidated and
  invalidate affected replies.

### Response and telemetry

Return real identities and measured values (numbers illustrative); the host writes the
record. Replies are at most 65,536 characters, responses UTF-8 JSON under 1 MiB.

```json
{
  "schema_version": 1,
  "record_kind": "chat_response",
  "request_id": "COPY_FROM_REQUEST",
  "candidate_id": "COPY_FROM_REQUEST",
  "package_sha256": "COPY_FROM_REQUEST",
  "assistant": "Reply text",
  "reasoning": "Optional thinking apart from the reply",
  "telemetry": {
    "prompt_tokens": 512, "output_tokens": 128,
    "prefill_tokens": 512, "prefill_ms": 256,
    "decode_tokens": 127, "decode_ms": 3175,
    "ttft_ms": 290, "missing": {}
  }
}
```

`reasoning` (optional, nonempty, at most 65,536 characters) is the model's thinking
apart from `assistant`, which then holds only the reply; crit serve returns it as
`reasoning_content`. `prompt_tokens` is the full templated input, `output_tokens` all
generated tokens;
`prefill_*`/`decode_*` count tokens actually timed per phase (excluding a reused prefix
or the first token from prefill); `ttft_ms` runs from admission to first emitted token.
Synchronize device work at phase boundaries, exclude startup and loading, and never
estimate tokens from characters.

A missing phase sets **both** count and time to `null` with a reason in
`missing.prefill`/`missing.decode`; missing TTFT is `ttft_ms: null` plus `missing.ttft`.
Counts are integers, decode ≤ output, prompt + output within KV capacity. Identity
mismatch, malformed output, artifact mutation, failed cleanup or violated limits fail
the turn without speeds.

### Streaming resident replies

Add **`chat-stream-v1`** and implement `generate_stream(request, cancellation, emit)`,
which the host calls instead of `generate` (keep `generate`). Return the same response.

```python
def generate_stream(self, request, cancellation, emit):
    pieces = []
    for chunk in self.decode_text(request, cancellation):
        cancellation.check()
        emit(chunk)
        pieces.append(chunk)
    # self.measured_response supplies real token counts and phase timings.
    return self.measured_response(request, assistant="".join(pieces))
```

Call `emit` synchronously on the generation thread with nonempty UTF-8-encodable text
without NUL, only for accepted text (never speculative proposals). Chunks are
concatenated exactly, with no retraction; the final `assistant` must equal them byte
for byte. Callback errors fail the turn even if caught. `reset` and `close` must stop
every producer that could call `emit`. Token timing still comes from the final
response. Packages without `chat-stream-v1` stream their reply as one chunk.
`emit({"reasoning": text})` streams reasoning apart from the reply in the same way; the
final `reasoning` must equal those chunks.

### Structured output for adapters

Declare **`chat-constraints-v1`** when the engine applies `response_format`:

| Value | Apply with |
| --- | --- |
| `{"type": "json_object"}` | `crit::core::JsonObjectMatcher` |
| `{"type": "json_schema", "name", "schema", "grammar"}` | `crit::core::GrammarMatcher` over `ByteGrammar::load(grammar)`; `grammar` is the compiled `crit-byte-grammar-v1` artifact (no compiler needed), `schema` the client's, for reference |

Build one `crit::core::ByteVocabulary` per model from every token's exact decoded bytes
(`crit_core.constraints.byte_vocabulary(tokenizer, eos)` from a
`crit_core.text.Tokenizer`), with the reply-ending token as EOS. Per request, use a
fresh matcher: pass `AllowedTokens{matcher.allowed()}` to `Sampler::sample` at each
position (or the same mask on device), append accepted tokens, rewind rejected
speculative ones, stop at EOS. The whole reply is constrained, including any reasoning
or template text. When `allowed()` throws, fail the request; never generate
unconstrained text. Budget completion room within `max_output_tokens` and the matcher's
limits. Full-vocabulary per-token masks need the native matchers or an equivalent device mask.

### Strict and streaming tools for adapters

Declare **`chat-tools-strict-v1`** alongside `chat-tools-v1` only when the adapter
constrains each strict function's arguments during decoding. The request's
`tool_grammars` maps function names to compiled `crit-byte-grammar-v1` artifacts.
Use a fresh `GrammarMatcher` (or equivalent device constraint) for each call's
arguments, exactly as for structured output. Non-strict functions have no entry.
The server checks the complete arguments too; checking after unconstrained
generation alone does not satisfy this adapter contract.

For live tool deltas, additionally declare **`chat-tool-stream-v1`** and
`chat-stream-v1`. `generate_stream(request, cancellation, emit)` may emit text as
before or an object containing only `tool_calls`, a list of OpenAI-style deltas:

```python
emit({"tool_calls": [{"index": 0, "id": "call_unique_id", "type": "function",
                      "function": {"name": "read_file", "arguments": ""}}]})
emit({"tool_calls": [{"index": 0, "function": {"arguments": '{"path":'}}]})
emit({"tool_calls": [{"index": 0, "function": {"arguments": '"a.py"}'}}]})
```

Introduce indices consecutively from zero, with a new id and complete function
name. Subsequent deltas for that index contain only a nonempty `arguments` suffix;
names and ids cannot be rewritten or emitted again. Use the model-specific parser
to separate text, tool syntax and arguments before emitting. Return the assembled
calls as the final response's `tool_calls` and concatenate only text emits into
`assistant`. Every emitted call must appear unchanged in the final response.
If no tool is selected, text-only output remains valid under `tool_choice: "auto"`.

The same synchronous-thread, cancellation and callback-error rules as text
streaming apply. One bounded cumulative snapshot holds text and calls, so a slow
reader may receive coalesced fragments. Calls are checked for choice policy,
unique ids, append-only arguments, strict-schema prefix membership and the shared
65,536-character limit. A callback failure invalidates the turn even if swallowed
by the adapter. Cancelled or failed partial calls are never committed to a later
request. Sampling, constraints and parser state must remain private to the turn.

### Per-request sampling for adapters

Declare individual capabilities only for fields the engine actually applies:
`chat-sampling-temperature-v1`, `chat-sampling-top-p-v1`, `chat-sampling-top-k-v1`,
`chat-sampling-min-p-v1`, `chat-sampling-seed-v1`, `chat-sampling-presence-penalty-v1`,
`chat-sampling-frequency-penalty-v1`, `chat-sampling-repetition-penalty-v1`, and
`chat-sampling-logit-bias-v1`. The request's optional `sampling` object contains
only explicit, non-null overrides, using the original API field names.

Apply overrides to fresh request options; never mutate the candidate configuration
or resident defaults. Temperature is in `[0, 2]`, top-p in `(0, 1]`, min-p in `[0, 1]`, top-k is
an unsigned 32-bit integer, seed an unsigned 64-bit integer, presence/frequency
penalties in `[-2, 2]`, and repetition penalty strictly positive. All floats must
be finite. `logit_bias` maps canonical nonnegative decimal token-id strings to
biases in `[-100, 100]`, at most 1024 entries; the engine must also validate ids
against its vocabulary.

`crit_core.native.ResidentEngine` forwards these overrides through the native ABI
to Core's `RequestOptions.sampling` and checks `ModelInfo.sampling` support. Its
native library must be rebuilt to forward penalties and logit bias. Only declare
capabilities implemented by that model, including speculative decoding paths.
No request override changes Eval's greedy protocol or the sealed configuration.

### Effective configuration

Declare **`chat-config-v1`** and implement `effective_configuration()`, returning what
the engine actually parsed. Every key of `context.configuration` must be present and
equal (extra keys allowed); otherwise the engine is closed and the load fails naming the keys.

```python
def effective_configuration(self):
    return dict(self.native.applied_configuration())  # what the native engine parsed
```

### Ownership and recovery

A per-user guard (`crit-runtime-USER` in the OS temp folder, or
`CRIT_RUNTIME_DIRECTORY`, never to bypass an owner) lets one runtime own the machine.
The engine runs in a Windows Job Object or POSIX process group retired on stop,
timeout, shutdown or client death. After a crash, `crit serve --recover-runtime`
verifies retirement before releasing admission; never delete the guard. Request
records live in the runtime directory while running (kept after failure for
diagnosis); nothing is written to the run folder.

The guardian and supervisor both retry a transient permission denial when reading
the atomically replaced ownership record, for at most 250 ms. They always read
fresh data; missing/malformed records, changed identities and persistent access
failures still stop the engine. Unexpected resident exits include the exit code
and the retained diagnostic log's location.

The [residency fixtures](../tests/test_chat_residency.py) contain executable example adapters.
