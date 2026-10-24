# Serving an engine

`crit serve` loads one sealed candidate, keeps it resident and answers
OpenAI-compatible chat requests until stopped, so any OpenAI client (the `openai`
SDKs, LangChain, Open WebUI, curl) can use the engine. Packages implement the
[adapter contract](#adapter-contract). `crit run ... eval` serves a packaged candidate
the same way before it is sealed, for Eval over the chat API
([Building engines](BUILDING_ENGINES.md#run-the-common-eval-suite)).

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
| `--models PATH` | JSON catalog of named models from several runs; replaces `--workspace`, `--run`, `--candidate` and `--model-name` |
| `--timeout SECONDS` | Limit per request; default 600 |
| `--load-timeout SECONDS` | Limit on loading the starting candidate; default 600 plus the model's bytes at 200 MB/s (50 MB/s with `--verify-model`) |
| `--verify-model` | Before loading, read and SHA-256 every model file against the run's frozen manifest, and give engines without `chat-model-file-v1` a private read-only copy. By default the host only checks each file's size against the manifest and maps it in place, so the engine starts loading at once; a checkpoint with the same sizes but other bytes (another revision, corruption) is then served unnoticed |
| `--resident-ram-gib N`, `--resident-vram-gib N` | Optional caps on pool-owned model bytes, not total process/device memory |
| `--project PATH` | crit checkout used to verify packages; default this checkout |
| `--write-launchers` | Write `serve.cmd` (Windows, double-click) and `serve.sh` with the same options into the run folder, then exit. Under WSL, `serve.cmd` starts `serve.sh` in the same distribution. They never contain the API key; set `CRIT_API_KEY` |
| `--recover-runtime` | Verify retirement of a crashed runtime, release ownership, exit |

Ctrl+C (or SIGTERM on Linux) stops accepting requests, cancels active generation
and rejects queued requests before unloading the engine and releasing model
memory. Shutdown uses the same cancellation grace and process retirement as
client cancellation; ownership is released only after cleanup settles. If
retirement fails, ownership stays blocked and cleanup can be retried.

The server holds the machine's engine lock while it runs (`crit_core.machine`): `crit eval run --adapter`, `pin`,
`ab`, `kernels`, `reference` and `crit run measure` wait for it, and it waits for them, printing who holds the lock
(`crit eval run --url` measures the server itself and takes no lock). Two
engines side by side can exhaust host RAM when weights live there. `CRIT_ENGINE_LOCK` names the lock file (default
`crit-engine.lock` in the system temp directory), `CRIT_ENGINE_LOCK=off` disables it. Since file locks do not
cross OS instances, the holder also keeps a lease that Windows and its WSL distributions both read
(`CRIT_ENGINE_LEASE`, default `C:\ProgramData\crit\engine.lease`, `/mnt/c/ProgramData/crit/engine.lease` under WSL).

### Several models on one server

Create a catalog, for example `models.json`:

```json
{
  "default": "text",
  "models": [
    {"name": "text", "workspace": "runs", "run": "run-0001", "candidate": "candidate-0002"},
    {"name": "vision", "workspace": "runs", "run": "run-0002"}
  ]
}
```

Use actual retained run/candidate IDs. Workspace paths are relative to the catalog
file (absolute paths also work). Omitting `candidate` selects that run's selected
candidate at server startup. Names must be unique; `default` defaults to the first
entry. Start with `crit serve --models models.json`. `--write-launchers` writes
the scripts beside the catalog.

`GET /v1/models` lists the configured names. The request's `model` selects that
name; omitting it chooses the default. An unknown name returns 404, rather than
silently generating with another model. Each binding uses its own capabilities,
context limit, sealed configuration and model identity.

Only one model's weights are resident. Switching drains active requests, closes
the old engine, releases its leases and frees its model pool before loading the
next model. Failed cleanup retires the host before another model can enter.
Switching back reloads those weights. Candidates using the same model identity
still reuse compatible allocations. This supports one-model memory budgets;
alternating model requests pay loading latency. It does not run different models
simultaneously.

## Endpoints

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | `{"status": "ok", "engine": ...}`; 503 when the runtime is blocked. No key needed |
| `GET /v1/models`, `GET /v1/models/{id}` | The run's sealed resident candidates, or catalog names, with `label`, `context_length` (the candidate's `kv_tokens`), `default`, package `capabilities`, `supported_sampling_parameters` and `engine_facts` ([facts for Eval](#engine-facts-and-plain-decoding)) |
| `POST /v1/chat/completions` | Chat completion, JSON or server-sent events |
| `POST /v1/crit/score` | [Fixed-text scores](#fixed-text-scores) for packages declaring `chat-score-v1` |
| `POST /v1/messages` | The same engine through the [Anthropic Messages API](#other-api-dialects) |
| `GET /`, `/api/version`, `/api/tags`, `/api/ps`; `POST /api/chat`, `/api/generate`, `/api/show` | The same engine through the [Ollama API](#other-api-dialects) |
| `GET /metrics` | [Prometheus metrics](#metrics) of the server |

`model` selects a candidate by `RUN_ID/CANDIDATE_ID` or `CANDIDATE_ID`; any other name
(`"default"`, `"gpt-4o"`) uses the default. An unknown candidate returns 404, one that
fails verification 409. Switching closes the old engine and loads the new one, reusing
compatible model allocations. Requests run one at a time unless the adapter
declares the concurrent request contract below.

| Field | Behavior |
| --- | --- |
| `messages` | Optional leading `system`/`developer` messages (merged into one `system`), then user/assistant turns ending with user. Consecutive same-role turns are joined with a blank line. Content is a string or ordered `text`/`image_url` parts; images require `chat-images-v1` |
| `max_tokens`, `max_completion_tokens` | Output cap, 1 to `context_length - 1`; default 1024 or less to fit |
| `stream`, `stream_options.include_usage` | SSE: role chunk, text chunks, finish chunk, optional usage chunk, `[DONE]` |
| `stop` | Up to 4 strings of at most 256 UTF-8 bytes. Text before the first completed stop returns with `finish_reason: "stop"` (`crit_core.text.StopMatcher`); the turn is cancelled, the engine stays loaded |
| `n` | Must be 1 |
| `response_format` | `text`, `json_object` or `json_schema` ([structured output](#structured-output)) |
| `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias` | Per-request overrides when the package declares support for that field; otherwise 400. Omitted or null fields keep engine defaults |
| `tools`, `tool_choice`, `parallel_tool_calls` | Function calls for packages declaring `chat-tools-v1`; `auto`, `none`, `required` or a named function (adapters that parse free generation, such as [ResidentChat](#tool-calls-from-the-chat-template): `auto` or `none`). Tool-result history is preserved |
| `logprobs`, `top_logprobs` | [Token log-probabilities](#token-log-probabilities) for packages declaring `chat-logprobs-v1`: every token's own and up to 20 alternatives. Not with `stop` |
| `crit` | crit's own extension: `{"decode": "plain"}` decodes the reply one token per target pass, without drafting, for packages declaring `chat-plain-v1` ([plain decoding](#engine-facts-and-plain-decoding)) |

Responses carry `usage` when measured and `crit_telemetry`: prompt and output tokens, the
prompt tokens the engine did not compute (`usage.prompt_tokens_details.cached_tokens`: its
telemetry's prompt tokens minus `prefill_tokens`, 0 when it computed them all), the
engine's prefill and decode rates, TTFT, and `tokens_per_round` (output tokens per target
pass) when the engine reports its decode rounds, with `accepted` (drafts accepted in each of
those rounds), `verify_ms` and `draft_ms` (device time in verify passes and draft steps) when
it reports them. A stream's usage chunk carries both.
Sampling overrides are validated before generation, including for streaming requests.
Unknown parameters and unsupported generation controls return 400 rather than being
silently ignored. Only client metadata (`user`, `metadata`, `store`, `service_tier`)
is ignored and listed in `X-Crit-Ignored-Parameters`; no storage or service tier is applied.
Audio, legacy `functions`/`function_call`, and an unsupported `response_format` return 400. Errors use the
OpenAI shape; timeout is 504, engine failure 500 (the next request reloads).
Closing the request connection cancels the turn, including ordinary graceful TCP
closes, connection resets and other detected socket failures. This also applies to
non-streaming requests and streaming requests that have not emitted text yet.
Cooperative cancellation resets request state and keeps weights loaded; an adapter
that does not settle within the cancellation grace period is retired before another
turn can run.
Streaming writes also honor the command's deadline and shutdown while a connected
client is not reading. If delivery cannot continue, the connection is closed and
the existing cancellation path settles the request. A partially sent SSE event
is not followed by an error event or `[DONE]`. A responsive client can still
receive the ordinary terminal error when generation times out.
Cancellation also removes a request waiting for the engine without interrupting
the active turn or dispatching the cancelled request later.

If a streaming consumer raises or a supervised command is interrupted, the server
retires the resident host before releasing the request lock. Failed retirement
blocks reuse until cleanup is proved. Unexpected consumer exceptions and interrupts
propagate unchanged after cleanup; existing host/protocol errors keep their
`EngineError` translation. The next request after successful retirement loads a
fresh engine.
The streaming, tool and reasoning regressions exercise these failures through
real resident processes with synthetic adapters, including a queued request
waiting for retirement and cleanup failure blocking further requests.

Clients must keep both sides of the connection open until the response completes.
`crit serve` treats a clean TCP EOF as request abandonment, so explicitly
half-closing the client's sending side also cancels the turn. This is a serving
policy: a graceful full close and a write-half-close look the same to the server.
The connection watcher never consumes bytes of a pipelined request and stops before
the next request uses the connection.

Cancellation must reach the server: the client must abort the HTTP request, and
any proxy in between must propagate the disconnect. A UI that only stops displaying
text while leaving its HTTP request open has not cancelled the server request.
The cancellation regressions in `tests/test_api_cancellation.py` exercise ordinary
Python HTTP client closes and curl request timeouts in both response modes. They
also check that cancelled queue entries never run, cooperative adapters retain
their weights, and uncooperative adapters retire before the next turn.

This changes clients that previously sent ignored sampling values to old packages:
omit those fields to retain the frozen defaults, or build a new package with an
adapter that applies and declares them. Never edit a sealed package's manifest.

### Images

Packages declaring `chat-images-v1` accept ordered text and image parts:

```json
{"role":"user","content":[{"type":"text","text":"Describe this image."},
 {"type":"image_url","image_url":{"url":"data:image/png;base64,...","detail":"auto"}}]}
```

`url` accepts HTTP(S) URLs or base64 `data:image/...` URLs; `detail` is `auto`,
`low` or `high`. Crit validates and carries the source through the private
request record; it does not fetch URLs or open client-supplied file paths.
The HTTP body remains bounded at 8 MiB. Request records containing images allow
16 MiB including metadata; ordinary run records retain their 1 MiB bound.
Unsupported engines return 400. Tool history and image order are preserved.

A `ResidentChat` engine sets `images = True` and declares
`capabilities(text, sampling, images=True)`. Its `prepare_prompt(request,
cancellation)` context manager uses `self.prompt_ids(request)` to render the
checkpoint template, supplies vision processing, and yields
`PreparedPrompt(ids, inputs, token_count)`. Templates receive content lists
with `{"type":"text","text":...}` and `{"type":"image","image":URL,"detail":...}`.
`token_count` is the actual context positions after image expansion; it controls
context checks and prompt telemetry. It must include the text and image positions.
`generate_prepared(prompt, max_new_tokens, sampling, on_tokens, cancelled)`
consumes the engine's prepared inputs and returns the usual tokens/timing pair.
Release request-private vision inputs in the context manager's `finally`, including
on cancellation or context rejection. Image decoding, URL access policy, download
bounds, vision arithmetic and embedding layout belong to the engine; preprocessing
must observe the supplied cancellation object. Text-only adapters keep their
existing `generate_tokens` path.

### Tool calls

`chat-tools-v1` supports up to 128 function definitions with unique names and up to
128 calls per reply. Assistant history carries `tool_calls` with JSON-string
`function.arguments`; subsequent `role: "tool"` messages carry the matching
`tool_call_id`. All calls need exactly one result before the next turn. A request
may end with the last tool result. The API never executes tools itself.

Responses contain `message.tool_calls` and `finish_reason: "tool_calls"`; a tool-only
message has `content: null`. A reply `max_tokens` cuts inside a call returns that call with the
arguments written so far and `finish_reason: "length"`, as OpenAI's API does: act only on the
calls of a `"tool_calls"` finish. With `chat-tool-stream-v1`, SSE returns each call's
`index`, id, type and complete name once, followed by incremental argument fragments.
Multiple calls may interleave by index. Without that capability, tool requests are
buffered until the adapter finishes, then emitted as a single tool-call delta.
`parallel_tool_calls: false` permits at most one call. Unknown functions, invalid JSON arguments and replies
violating `tool_choice` fail rather than being exposed as usable calls.

`strict: true` requires `chat-tools-strict-v1`. The function's `parameters` must be
an object schema in Core's [supported subset](#structured-output), including
`additionalProperties: false`. The server compiles it before generation and checks
completed arguments against that grammar, and a cut call's arguments as a prefix it accepts. Strict arguments use compact JSON in the
schema's declaration order, as with `response_format`. Unsupported schemas return
400 with `param: "tools"`, `code: "unsupported_schema"`; there is no unconstrained
fallback. Non-strict tools require JSON-object arguments but do not promise schema
conformance. A request may mix strict and non-strict functions.

Streaming deltas are provisional until `finish_reason: "tool_calls"`. The host
validates strict argument prefixes before publishing them and requires the final
calls to match the stream exactly. An invalid final call, or an incomplete one in a reply
`max_tokens` did not cut, produces an SSE error and `[DONE]`, without a successful finish chunk. Clients should assemble
calls by index and only act on a successfully completed turn.

Tool requests cannot combine `stop` or a structured `response_format`.

### Tool calls from the chat template

Adapters built on `crit_core.serving.ResidentChat` (and Core's `resident_engine`) offer
tool calls without model-specific code in crit. ResidentChat renders the request's
`tools` (unless `tool_choice` is `none`) and the tool history with the model's own chat
template, splits the reply into content and calls with `crit_core.text.ToolCallParser`,
gives each call an id and returns them as `tool_calls`. With streaming a call goes out as
the model writes it: its id and name once the name is written, then its arguments as they
become certain, so a client sees the call from its first tokens. `capabilities(text,
sampling)` then declares `chat-tools-v1`, plus `chat-tool-stream-v1` when the package
streams. A block that names no offered function or exceeds `parallel_tool_calls: false`
stays in the content as written. One that has begun streaming as a call is that call: if it
closes without parsing, or the model ends its reply inside it, the reply fails; if
`max_tokens` cuts the reply inside it, the call keeps the arguments written so far. `tool_choice` must be `auto` or `none`, since the model is not
constrained to call; strict tools need an adapter with `chat-tools-strict-v1`.

The call syntax is the one the template teaches when crit registers it
(`crit_core.text.templates_tool_format`; crit registers only syntaxes several model
families share, such as Qwen3-Coder style `<function=...>` blocks and Hermes-style JSON
in `<tool_call>`). When a model writes another one, the engine package carries the
parser, never crit: set `ResidentChat.tool_format` to a `crit_core.text.ToolFormat`
(the parse function and markers) and declare `capabilities(text, sampling,
tool_format=...)`. Give it a `stream` function too, which reads the function name and the
start of the arguments from an unfinished block, and its calls stream as they are written;
without one, each call arrives as one delta when its block closes, and a block that does
not parse stays in the content. Packages sealed with the earlier `chat-tool-template-v1` keep working:
crit serve reads it as `chat-tools-v1`.

Reasoning and tool boundaries use token identity when a marker is a single special
token in the loaded tokenizer. Ordinary tokens spelling that marker remain literal
content, including a closing marker inside tool arguments. Markers made from ordinary
or multiple tokens keep text matching. A package can explicitly choose spelling-based
matching with `ToolFormat(..., delimiters="text")` or
`ReasoningFormat(..., delimiters="text")`; the default is `"auto"`. Tool body parsers
still receive text and define their own syntax and escaping rules.

### Reasoning

Reasoning models write their thinking between markers before the reply. crit serve
returns it apart from the reply, as OpenAI-compatible clients expect: the message's
`reasoning_content` in JSON replies, and `delta.reasoning_content` chunks, before the
content, when streaming. Adapters built on `crit_core.serving.ResidentChat` split it in
the markers the model's chat template uses (`crit_core.text.reasoning`; crit registers
the `<think>...</think>` syntax several model families share). Reasoning starts where the
reply starts, either because the generation prompt ends with the opening marker or
because the reply opens with it; a marker later in the reply is reply text, and a reply
cut before the closing marker is all reasoning. That is a valid completion with empty
assistant content, including when the output cap is reached; it keeps the resident
engine available for the next request. Tool calls are parsed from the reply
only. A package whose model writes other markers sets `ResidentChat.reasoning_format`
to a `crit_core.text.ReasoningFormat`, and `False` leaves the reasoning in the reply.

### Token log-probabilities

With `logprobs: true` a reply's `choices[0].logprobs.content` lists every token the
model generated, in order: reasoning and tool-call tokens included, the stop token that
ended the reply left out. Each entry is OpenAI's `{token, logprob, bytes,
top_logprobs}`, where `top_logprobs` holds the `top_logprobs` (0 to 20) most likely
tokens at that position, ranked, in the same shape. `bytes` are the token's UTF-8 bytes;
`token` is their text, with U+FFFD where a token ends inside a character. A
log-probability is natural-log, of the raw logits before temperature, penalties, bias
and constraints; a zero probability reads as -9999. When streaming, the entries arrive
with the finish chunk, since the engine reports them in its final response. A request
with `stop` strings is refused, since its turn is cancelled where a stop completes,
before the engine reports its tokens.

### Fixed-text scores

`POST /v1/crit/score` with `{"model", "prompt_ids", "continuation_ids", "top_logprobs"}` and
optionally `"prompt_rows": false` (token ids as the model's tokenizer numbers them;
`top_logprobs` 0 to 20) scores fixed text
through the engine's own serving paths, for packages declaring `chat-score-v1`: `prompt`
holds one row per prompt position after the first, from one prefill of the prompt as the
engine prefills a request, and `continuation` one row per continuation token, decoded after
that prefill with every step forced to the continuation (a speculative engine accepts a
draft only while it equals the continuation and takes every correction and bonus token from
it). The first continuation row comes from the prefill's last position. With `prompt_rows`
false, `prompt` is empty and the engine only prefills the prompt, as for a reply. Rows are `[token,
logprob, [[token, logprob], ...]]` with exactly `top_logprobs` ranked alternatives,
natural-log, of the raw logits; `crit_telemetry` holds `prefill_ms` and `decode_ms`. With
`"format": "packed"` each part comes as packed arrays instead, `{"count", "alternatives",
"tokens", "logprobs", "top_tokens", "top_logprobs"}`: base64 of little-endian arrays, the
tokens and alternative tokens int32, the log-probabilities float32, the alternatives row by
row (`count` × `alternatives`), and the response names `"format": "packed"`. Eval's quality
gate uses it ([docs/eval](eval/README.md)). A score runs alone on the engine, as Core runs
scores, and is not streamed.

### Engine facts and plain decoding

While a candidate's engine is loaded, its `/v1/models` card carries `engine_facts`, what
the engine reports for Eval's headroom and exactness: `speculative` and `plain` (booleans),
`draft_width`, `bytes_per_pass`, `draft_bytes_per_step`, `draft_ms` and
`kv_bytes_per_token` (non-negative numbers), each only when the engine reports it; `null`
when the candidate is not loaded or reports none. A speculative engine that declares
`chat-plain-v1` decodes a request carrying `"crit": {"decode": "plain"}` one token per
target pass without drafting, which is how Eval compares speculative output with plain
greedy on the served engine.

## Other API dialects

The same engine answers two more APIs, so the tools people already run need only a base URL. Every
request is translated into the chat request above and validated the same way, so the capabilities a
package declares (tools, images, sampling fields, structured output) gate it in every dialect, and an
API key set with `--api-key` guards every endpoint except `/health`, `/` and `/api/version`.

**Anthropic Messages** (`POST /v1/messages`, [crit_api/anthropic.py](../api/crit_api/anthropic.py)): the
API Claude Code and the Anthropic SDKs speak. Point them at the server (`ANTHROPIC_BASE_URL`, or the
SDK's `base_url`) with the key in `x-api-key` or as a Bearer token; `anthropic-version` is accepted.
`system` (a string or text blocks) becomes the leading system message, content blocks become text and
image parts, `tool_use` blocks become tool calls and `tool_result` blocks tool messages (tool results
first, then the turn's text, as Anthropic orders them), `tools` with an `input_schema` become
functions, `tool_choice` (`auto`, `any`, `tool`, `none`, `disable_parallel_tool_use`) and
`stop_sequences` map one to one, `temperature`, `top_p` and `top_k` pass through, and
`output_config.format` of type `json_schema` becomes `response_format`. `max_tokens` is required.
`metadata`, `thinking`, `service_tier`, `betas` and the other service settings are accepted and ignored
(listed in `X-Crit-Ignored-Parameters`); `mcp_servers`, `container`, document blocks and Anthropic-defined
tools (web search, code execution) return 400, and `/v1/messages/count_tokens` 404. The reply is a
Messages response: `thinking` (the engine's reasoning), `text` and `tool_use` blocks, `stop_reason`
`end_turn`, `max_tokens`, `stop_sequence` (with the sequence) or `tool_use`, and `usage` with the prompt
tokens the engine computed as `input_tokens` and the ones its prompt cache served as
`cache_read_input_tokens`. With `stream: true` the reply is the Messages event stream: `message_start`,
`content_block_start`, `content_block_delta` (`text_delta`, `thinking_delta`, `input_json_delta`) and
`content_block_stop` per block in the order the engine produces them, `message_delta` with the stop
reason and usage, `message_stop`; an engine failure is an `error` event. Errors use Anthropic's
shape, `{"type": "error", "error": {"type", "message"}}`.

**Ollama** (`GET /` answers "Ollama is running", `/api/version`, `/api/tags`, `/api/ps`; `POST
/api/chat`, `/api/generate`, `/api/show`; [crit_api/ollama.py](../api/crit_api/ollama.py)): the API the
local chat UIs look for. Models are listed under the ids `/v1/models` shows (and the default name).
`/api/chat` takes Ollama's messages (`images` as bare base64, their type read from the bytes; assistant
`tool_calls` with arguments as objects, given ids when they have none; `tool` messages matched to the
latest assistant's calls in order), `tools`, `format` (`"json"`, or a JSON schema) and `options`:
`temperature`, `top_p`, `top_k`, `min_p`, `seed`, `num_predict` (the output cap), `stop`,
`repeat_penalty`, `presence_penalty` and `frequency_penalty` apply; Ollama's runtime settings
(`num_ctx`, `num_gpu`, `num_thread`, `mirostat`, ...) and `keep_alive` and `think` are ignored and
listed; an unknown option is 400. `/api/generate` renders `prompt` (and `system`) as one user turn
through the model's chat template; `raw`, `suffix`, `template` and `context` are refused. Replies
stream by default as newline-delimited JSON objects (`message.content`, `message.thinking` and
`message.tool_calls`, or `response` for generate), the last with `done`, `done_reason` (`stop` or
`length`), `total_duration`, `prompt_eval_duration` and `eval_duration` in nanoseconds from the
engine's own phase times, and `prompt_eval_count` and `eval_count`; `stream: false` returns one object.
`/api/show` reports the context length as `model_info["crit.context_length"]` and the
`capabilities` (`completion`, `tools`, `vision`) the package declares. Errors are `{"error": message}`;
`/api/pull`, `/api/embed` and the other model-management endpoints return 404.

## Metrics

`GET /metrics` serves the Prometheus text format (behind the API key when one is set): requests
finished by dialect (`openai`, `anthropic`, `ollama`) and outcome (`completed`, `error`, `cancelled`,
`timed_out`), requests in flight, prompt, cached-prompt and completion tokens, wall seconds in
requests, uptime, the last completed request's prefill and decode rates, time to first token and
tokens per round, and `crit_serve_engine_loaded` with the loaded run, candidate and state
([crit_api/metrics.py](../api/crit_api/metrics.py)). Counters start at zero when the server starts.

## Structured output

`response_format` restricts the reply's bytes with Core's
[structured-output service](../core/src/constraints/README.md). It needs a package
declaring `chat-constraints-v1`; otherwise it returns 400.

- `{"type": "json_object"}`: one JSON object (surrounding whitespace allowed).
- `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}, "strict": ...}}`:
  compact JSON in the schema's declaration order, compiled with `crit_core.constraints`.
  The [supported subset](../core/src/constraints/README.md#compiling-languages-python):
  closed objects (`additionalProperties: false`), typed arrays, strings with lengths or a
  `pattern`, booleans, null, numbers, integers with both bounds, `enum`, `const`, `anyOf`,
  type lists, provably disjoint `oneOf` and local `$ref`/`$defs`. Anything else
  (`format`, overlapping `oneOf`, `allOf`, open objects) returns 400 with
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

Calls are serialized by default. The client owns the conversation (each
request carries all of it); the engine owns tokenizer, private KV/recurrent state,
scratch and graphs until switch or failure; the model pool owns the mapped model files and
device weights until that model unloads, one entry per representation. A → B → A
closes each engine before the next; different model identities also free the old
pool before loading.

Packages share one process: release views, mutable globals and workers on close. Link
the CUDA runtime dynamically (shared `libcudart`); a static one failed with "invalid
argument" after another engine's had loaded. Construction, reset or close failure can
retire the runtime; a stuck adapter is killed after a one-second cancellation grace.

### Concurrent requests

A package opts in with `chat-concurrent-v1` (or
`capabilities(text, sampling, concurrent=True)`). Its loaded parent engine exposes
`max_concurrent_requests`, an integer from 1 to 1024, and
`create_request(request)`. Capacity is the adapter's safe bound for the sealed
configuration and memory limits; the host does not infer it from GPU or model names.

The factory returns a distinct object implementing the same generation,
streaming, `reset()` and `close()` methods. It may be a `ResidentChat` sharing
immutable tokenizer/model resources with the parent. KV/recurrent state, scratch,
random sampling state, prepared images and other mutable turn data must be private.
The factory and shared decoder resources must support concurrent calls. Returning
the parent or another live request object is rejected. The parent remains loaded;
each request object is reset and closed after its generation settles.

The host admits at most the declared number for the same resident candidate.
Every command has its own cancellation, deadline, stream and response. Queued candidate/model switches drain active requests first and prevent
later requests from continually overtaking them. Cancelled waiters leave without
dispatch. Shutdown cancels all active requests and waits for cleanup.

Cooperative cancellation preserves peers and shared weights. A failed cleanup,
adapter failure or request that ignores cancellation requires retiring the shared
host process; other active requests fail too, and no new work enters until they
have unwound. This is the existing ownership guarantee applied to concurrent work.
The contract permits an engine's scheduler to batch requests; it does not itself
promise batched device arithmetic or a throughput gain. The
[batched reference engine](../inference/batched-reference-engine/README.md) is a native
library that does both: its steps run every request's rows through one GEMM per weight.

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
record. Replies are at most 65,536 characters, responses UTF-8 JSON under 1 MiB, plus
1 KiB per output token and alternative with [log-probabilities](#token-log-probabilities-for-adapters).

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
    "decode_tokens": 127, "decode_ms": 3175, "decode_rounds": 40,
    "ttft_ms": 290, "missing": {}
  }
}
```

`reasoning` (optional, nonempty, at most 65,536 characters) is the model's thinking
apart from `assistant`, which then holds only the reply; crit serve returns it as
`reasoning_content`. `assistant` may be empty when nonempty reasoning or valid tool
calls are present. Otherwise it must contain nonblank text. Every text channel must
be UTF-8 encodable without NUL; reasoning has its own character budget, separate
from the combined assistant/tool-call budget.
`prompt_tokens` is the full templated input, `output_tokens` all
generated tokens;
`prefill_*`/`decode_*` count tokens actually timed per phase (excluding a reused prefix
or the first token from prefill); `ttft_ms` runs from admission to first emitted token.
Optional `decode_rounds` counts target passes after prefill (a speculative engine's verify
rounds, at most `output_tokens`); ResidentChat reports it from `generate_tokens`' timing
`rounds`, and Core engine libraries report theirs. Optional `accepted` (one count of
accepted drafts per decode round, as many as `decode_rounds`), `verify_ms` and `draft_ms`
(non-negative milliseconds) pass through to `crit_telemetry`; ResidentChat takes them
from `generate_tokens`' timing.
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
final `reasoning` must equal those chunks. A reasoning-only completion emits reasoning
chunks and returns an empty `assistant`; it does not emit an empty text chunk.
Whitespace is valid incremental text, but completed output must satisfy the response
content rules above. Complete tool calls are validated at completion even when the
adapter returns them only in the final record.

Each `emit` reaches the client as one chunk the moment it is made. The resident host
and `crit serve` talk over one loopback connection that the host opens with the
runtime's ownership token and its process id; it carries commands, cancellations, status, stream pieces
and results as messages, so serving adds no polling delay to the first token or
between tokens (about a millisecond of dispatch and delivery in all).

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
`assistant`. Every emitted call must appear unchanged in the final response. A reply that
reaches `max_output_tokens` inside a call returns it with the arguments emitted so far, the
start of a JSON object; crit serve finishes it as `length`.
If no tool is selected, text-only output remains valid under `tool_choice: "auto"`.

The same synchronous-thread, cancellation and callback-error rules as text
streaming apply. Each emitted delta reaches the client as its own chunk, and crit serve
checks it again on arrival. Calls are checked for choice policy,
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

### Token log-probabilities for adapters

Declare **`chat-logprobs-v1`** (`capabilities(text, sampling, logprobs=True)`) when the
engine reports, for every generated token, its log-probability and its 20 most likely
alternatives, and set `logprobs = True` on the engine object; the host refuses to load
the package otherwise. A request that asks for them carries `"logprobs": n`, the
alternatives per token (0 to 20). The response then carries one row per generated
token but stop tokens, and every row's token and alternatives named by their bytes
(lowercase hex), so crit serve needs no tokenizer:

```json
"logprobs": {
  "tokens": [[9906, -0.0123, [[9906, -0.0123], [13347, -4.875]]]],
  "bytes": {"9906": "48656c6c6f", "13347": "4869"}
}
```

Rows are `[token, logprob, [[token, logprob], ...]]` with exactly `n` distinct
alternatives, ranked; log-probabilities are in [-9999, 0]. ResidentChat builds this:
with `logprobs = True` it passes `logprobs=n` to `generate_tokens` (and
`generate_prepared`), and the decoder calls `on_tokens(tokens, rows)` with one
`(logprob, [(token, logprob), ...])` row per token. It drops stop tokens, clamps -inf
and Core's rounding tolerance into the range and reads the bytes from the tokenizer.
`crit_core.native.ResidentEngine` sets `logprobs` when the library exports
`crit_engine_request_wait_output` and its `ModelInfo` declares `token_logprobs` with
`max_top_logprobs` of at least 20 (see
[engine libraries](../core/src/runtime/README.md#engine-libraries)).

### Fixed-text scores for adapters

Declare **`chat-score-v1`** (`capabilities(text, sampling, score=True)`) when the engine
scores fixed text through its serving paths, set `scores = True` and implement
`score_tokens(ids, continuation, top_k, cancelled, prompt_rows)`, returning `(prompt_rows,
continuation_rows, timing)` with one `(logprob, [(token, logprob), ...top_k])` row per
prompt position after the first (none when `prompt_rows` is false: the prompt is then only
prefilled) and per continuation token (None for the rows when
cancelled) and `timing` with `prefill_ms` and `decode_ms`; the host refuses to load the
package otherwise. ResidentChat's `score(request, cancellation)` builds the response from
it. `crit_core.native.ResidentEngine` sets `scores` when the engine's `ModelInfo` declares
`query_scoring`, `token_logprobs` with `max_top_logprobs` of at least 20, and the info facts
`replay`: the prompt is scored by
Core's query scoring with the engine's own top-k on its `"prefill"` score path (the default
path when it names none), which must run the engine's prefill kernels at every position, and
the continuation by a replay generation with log-probabilities.

### Engine facts and plain decoding for adapters

An engine keeps its facts for Eval in `facts`, a dictionary (`crit_core.native`'s engines
hold their `ModelInfo` there); the host publishes the known ones
(`crit_core.serving.ENGINE_FACTS`) as the card's `engine_facts`. A speculative engine
that also decodes plainly declares **`chat-plain-v1`** (`capabilities(text, sampling,
plain=True)`) and sets `plain = True`: a request with `"decode": "plain"` then reaches
`generate_tokens` as `plain=True`. `crit_core.native.ResidentEngine` sets `plain` when its
facts declare `speculative` and `plain`.

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
verifies retirement before releasing admission; never delete the guard. A failed
request's records (`requests/<request_id>/` in the runtime directory) stay for
diagnosis; nothing is written to the run folder.

The guardian and supervisor both retry a transient permission denial when reading
the atomically replaced ownership record, for at most 250 ms. They always read
fresh data; missing/malformed records, changed identities and persistent access
failures still stop the engine. Unexpected resident exits include the exit code
and the retained diagnostic log's location.

The [residency fixtures](../tests/test_chat_residency.py) contain executable example adapters.
