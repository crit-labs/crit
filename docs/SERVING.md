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
| `--load-timeout SECONDS` | Limit on loading the starting candidate; default 600 plus the model's bytes at 50 MB/s, since the host copies and hashes every model file before the engine loads |
| `--resident-ram-gib N`, `--resident-vram-gib N` | Optional caps on pool-owned model bytes, not total process/device memory |
| `--project PATH` | crit checkout used to verify packages; default this checkout |
| `--write-launchers` | Write `serve.cmd` (Windows, double-click) and `serve.sh` with the same options into the run folder, then exit. Under WSL, `serve.cmd` starts `serve.sh` in the same distribution. They never contain the API key; set `CRIT_API_KEY` |
| `--recover-runtime` | Verify retirement of a crashed runtime, release ownership, exit |

Ctrl+C stops accepting requests, unloads the engine and releases model memory.

## Endpoints

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | `{"status": "ok", "engine": ...}`; 503 when the runtime is blocked. No key needed |
| `GET /v1/models`, `GET /v1/models/{id}` | The run's sealed resident candidates, with `label`, `context_length` (the candidate's `kv_tokens`) and `default` |
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

Responses carry `usage` when measured and `crit_telemetry` (phase tokens, ms, rates).
Sampling follows the frozen configuration: `temperature`, `top_p`, `seed`, penalties and
similar are ignored and listed in the `X-Crit-Ignored-Parameters` header. Tools,
`logprobs`, audio and an unsupported `response_format` return 400. Errors use the
OpenAI shape; timeout is 504, engine failure 500 (the next request reloads). A
disconnecting client cancels its turn.

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
`launch[0]` is a retained Python module exporting `create_engine(context)` (native
engines can use a Python wrapper; a Core engine library exported with
`CRIT_EXPORT_ENGINE` can return `crit_core.native.resident_engine(context, library,
defaults)`, which applies the model's chat template and stop tokens, streams text and
reports Core's timing, see [engine libraries](../core/src/runtime/README.md#engine-libraries)).
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
scratch and graphs until switch or failure; the model pool owns verified host bytes and
device weights until the server stops, one entry per identity/representation, no
eviction. A → B → A closes each engine before the next.

Packages share one process: release views, mutable globals and workers on close. Link
the CUDA runtime dynamically (shared `libcudart`); a static one failed with "invalid
argument" after another engine's had loaded. Construction, reset or close failure can
retire the runtime; a stuck adapter is killed after a one-second cancellation grace.

### Model access and allocation

| API | Behavior |
| --- | --- |
| `context.host("relative/model/file")` | Read-only memoryview of a verified private snapshot (pageable; first use copies and hashes, needs temporary disk) |
| `context.model_file(name)` | Verified native-loader path when `chat-model-file-v1` is declared; no second copy |
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
  "telemetry": {
    "prompt_tokens": 512, "output_tokens": 128,
    "prefill_tokens": 512, "prefill_ms": 256,
    "decode_tokens": 127, "decode_ms": 3175,
    "ttft_ms": 290, "missing": {}
  }
}
```

`prompt_tokens` is the full templated input, `output_tokens` all generated tokens;
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

The [residency fixtures](../tests/test_chat_residency.py) contain executable example adapters.
