# Serving an engine

`crit serve` loads one sealed candidate, keeps it resident and answers
OpenAI-compatible chat requests until stopped, so any OpenAI client (the `openai`
SDKs, LangChain, Open WebUI, curl) can use the engine. Packages implement the
[adapter contract](#adapter-contract). `crit run ... eval` serves a packaged candidate
the same way before it is sealed, for Eval over the chat API
([Building engines](BUILDING_ENGINES.md#run-the-common-eval-suite)), and `crit run ... serve`
serves it for the checks a candidate passes before sealing, until Ctrl+C.

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
| `--api-key KEY` | Require `Authorization: Bearer KEY`; repeat for several keys, any of which a request may carry; default the comma-separated keys in `CRIT_API_KEY`. A non-local `--host` without a key prints a warning |
| `--api-key-file PATH` | Also accept the keys in this file, one per line; blank lines and lines starting with `#` are skipped, and a file without a key is an error |
| `--ssl-certfile PATH`, `--ssl-keyfile PATH` | Serve HTTPS with this PEM certificate chain and its private key (default: the key in the certificate file); the printed URL starts with `https://` |
| `--cors-origin ORIGIN` | Allow browser requests from this origin, or `*` |
| `--model-name NAME` | Model ID reported for the default candidate |
| `--reasoning-alias` | Return [reasoning](#reasoning) as `reasoning` as well as `reasoning_content`, for clients that read vLLM's and OpenRouter's name |
| `--models PATH` | JSON catalog of named models from several runs; replaces `--workspace`, `--run`, `--candidate` and `--model-name` |
| `--lenient` | Ignore, list and log the request parameters crit does not apply instead of answering 400 ([lenient requests](#lenient-requests)); never used by Eval |
| `--timeout SECONDS` | Limit per request, from when the engine takes it; default 600. A request's `crit.timeout` can only lower it |
| `--max-queue N` | Requests that may [wait](#a-busy-engine) while the engine is busy; default 64. One more is refused at once |
| `--queue-timeout SECONDS` | Limit on a request's wait for the engine, after which it is refused; default none |
| `--sse-ping-interval SECONDS` | [Keep-alive ping](#keep-alive-pings) on an OpenAI or Anthropic stream after this many seconds without output; default 15, 0 sends none |
| `--load-timeout SECONDS` | Limit on loading the starting candidate; default 600 plus the model's bytes at 200 MB/s (50 MB/s with `--verify-model`) |
| `--idle-unload SECONDS` | [Unload the engine](#unloading-the-engine) once no request has used it for this many seconds; the next request loads it again. Default: it stays loaded |
| `--verify-model` | Before loading, read and SHA-256 every model file against the run's frozen manifest, and give engines without `chat-model-file-v1` a private read-only copy. By default the host only checks each file's size against the manifest and maps it in place, so the engine starts loading at once; a checkpoint with the same sizes but other bytes (another revision, corruption) is then served unnoticed |
| `--resident-ram-gib N`, `--resident-vram-gib N` | Optional caps on pool-owned model bytes, not total process/device memory |
| `--project PATH` | crit checkout used to verify packages; default this checkout |
| `--write-launchers` | Write `serve.cmd` (Windows, double-click) and `serve.sh` with the same options into the run folder, then exit. Under WSL, `serve.cmd` starts `serve.sh` in the same distribution. They never contain an API key; they keep `--api-key-file`, or set `CRIT_API_KEY` |
| `--recover-runtime` | Verify retirement of a crashed runtime, release ownership, exit |

A request's key is compared in constant time with every configured key, and the key file is read once, at
startup. A missing key file, or a certificate that does not load, stops `crit serve` before it loads the engine.
With `--ssl-certfile` the server speaks HTTPS only: each connection's TLS handshake runs on the connection's own
thread and gets 10 seconds, so a client stalling in it holds up no other, and a failed handshake (a plain
`http://` request, for one) is logged and the connection closed. Clients must trust the certificate; a
self-signed one goes into each client's CA bundle.

Ctrl+C (or SIGTERM on Linux) stops accepting requests, cancels active generation
and rejects queued requests before unloading the engine and releasing model
memory. Shutdown uses the same cancellation grace and process retirement as
client cancellation; ownership is released only after cleanup settles. If
retirement fails, ownership stays blocked and cleanup can be retried.

The server holds the machine's engine lock while it runs (`crit_core.machine`), also while its engine is
[unloaded](#unloading-the-engine): `crit eval run --adapter`, `pin`,
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
    {"name": "text", "workspace": "runs", "run": "run-0001", "candidate": "candidate-0002",
     "aliases": ["claude-sonnet-*", "gpt-4o"], "display_name": "Text engine"},
    {"name": "vision", "workspace": "runs", "run": "run-0002", "aliases": ["claude-haiku-*"]}
  ]
}
```

Use actual retained run/candidate IDs. Workspace paths are relative to the catalog
file (absolute paths also work). Omitting `candidate` selects that run's selected
candidate at server startup. Names must be unique; `default` defaults to the first
entry. Start with `crit serve --models models.json`. `--write-launchers` writes
the scripts beside the catalog.

An entry's optional `aliases` are other names clients send for it: exact names
(`gpt-4o`, `llama3:8b`, `openai/gpt-4o`) or patterns in which `*` matches any
characters (`claude-sonnet-*` and `claude-haiku-*` catch the main and small models
Claude Code asks for), each 1 to 128 letters, digits, `.`, `_`, `-`, `:`, `/`, `@`
or `*`. Names and aliases must all be distinct. Its optional `display_name` is the
name its model card shows (default: the candidate's label). The catalog's optional
`fallback` names the model that serves any name nothing else selects.

`GET /v1/models`, `/v1/models/{id}` and `/api/tags` know the configured names, not
their aliases. The request's `model` selects the name it equals, else the entry of
the alias it equals, else of the matching pattern with the most characters besides
`*` (the earlier one in the catalog on a tie), else the `fallback`; omitting it
chooses the default. Names match case-sensitively. Without a `fallback` an unknown
name returns 404, rather than silently generating with another model. OpenAI replies
name the configured model, Anthropic and Ollama replies the requested one. Each
binding uses its own capabilities, context limit, sealed configuration and model
identity.

Only one model's weights are resident. Switching drains active requests, closes
the old engine, releases its leases and frees its model pool before loading the
next model. Failed cleanup retires the host before another model can enter.
Switching back reloads those weights. Candidates using the same model identity
still reuse compatible allocations. This supports one-model memory budgets;
alternating model requests pay loading latency. It does not run different models
simultaneously.

### Unloading the engine

`POST /v1/crit/unload` frees the GPU for other work (a game, another model) without stopping the server. It waits
for the requests the engine is running, while requests that arrive meanwhile wait for it; then the resident host
closes the engine and frees the model's weights in host and device memory, with the unload command shutdown uses.
It answers `{"object": "crit.unload", "unloaded": true, "engine": ...}` with the engine's status; `unloaded` is
false when no engine was loaded. With a body `{"model": NAME}` it unloads only that model's engine. The next request
loads the engine again, as a switch to another model does, and waits for the load. `POST /v1/crit/load` with
`{"model": NAME}` (an empty body loads the default model) loads it ahead of time and answers once it is loaded,
with `"object": "crit.load"`; it waits in the [queue](#a-busy-engine) as a request does, so `--max-queue` and
`--queue-timeout` refuse it too. An unload waits in that queue as well, ahead of the requests that arrive after it,
but counts toward neither limit. Both need the API key when one is set. `crit serve --idle-unload SECONDS` unloads the
engine once no request has used it for that many seconds. `/health` and each `/v1/models` card report `loaded`,
`/health` reports `ready` false (`/health?ready=1` is a 503) and `/api/ps` lists no model while the engine is
unloaded. Ollama clients unload and preload through
[`keep_alive` and requests without a prompt](#other-api-dialects).

The resident host process stays, with whatever the engine's `close()` does not release (its CUDA context, for
example), and so do the server's runtime ownership and the machine's engine lock: the other crit commands that take
the lock still wait until the server stops.

## Endpoints

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | `{"status": "ok", "ready": ..., "loaded": ..., "engine": ...}`; `loaded` is false while the engine is [unloaded](#unloading-the-engine), and so is `ready`. 503 when the runtime is blocked, and with `?ready=1` also while `ready` is false. No key needed |
| `GET /v1/models`, `GET /v1/models/{id}` | The run's sealed resident candidates, or catalog names, with `label`, `display_name` (a catalog entry's, else the label), `context_length` (the candidate's `kv_tokens`) and `max_model_len` (the same, vLLM's name for it, which LiteLLM reads), `default`, `loaded` (its engine is the one loaded), package `capabilities`, `supported_sampling_parameters`, `limits` and `engine_facts` ([facts for Eval](#engine-facts-and-plain-decoding)) |
| `POST /v1/chat/completions` | Chat completion, JSON or server-sent events |
| `POST /v1/completions` | [Raw-prompt completions](#raw-prompt-completions) for packages declaring `completion-v1`, JSON or server-sent events |
| `POST /v1/crit/score` | [Fixed-text scores](#fixed-text-scores) for packages declaring `chat-score-v1` |
| `POST /v1/crit/abort` | Cancels chat, score, load and tokenize requests in flight by [request id](#request-ids-and-aborting-requests) |
| `POST /v1/crit/unload`, `POST /v1/crit/load` | [Unload the engine's weights, or load a model](#unloading-the-engine), without stopping the server |
| `POST /tokenize`, `/detokenize`, `/apply-template`, `/v1/messages/count_tokens` | The engine's [tokenizer and chat template](#tokenizing) for packages declaring `chat-tokenize-v1` |
| `POST /v1/messages` | The same engine through the [Anthropic Messages API](#other-api-dialects) |
| `POST /v1/responses` | The same engine through the [OpenAI Responses API](#other-api-dialects), statelessly |
| `GET /`, `/api/version`, `/api/tags`, `/api/ps`; `POST /api/chat`, `/api/generate`, `/api/show` | The same engine through the [Ollama API](#other-api-dialects) |
| `GET /metrics` | [Prometheus metrics](#metrics) of the server |

`HEAD` on any `GET` path returns the `GET` response's status and headers without the body, for
clients that probe a path. Any other method (`DELETE`, `PUT`, `PATCH`) is a 405 with an `Allow`
header on a known path and a 404 elsewhere, in the path's dialect. Request bodies come with
`Content-Length` or `Transfer-Encoding: chunked`, from 1 byte to 8 MiB (`/v1/crit/load` and `/v1/crit/unload`
also take none); a body outside that, or a malformed `Content-Length`, is a 400 (413 over 8 MiB) in the path's
dialect too, and closes the connection.

Endpoints clients look for beside chat are a 404 with code `not_supported` and a message naming the
endpoint: `/v1/completions` for a package without `completion-v1`; `/v1/embeddings`, `/v1/rerank`,
`/v2/rerank`, `/rerank` and `/v1/score`, whose message says to run a separate embedding server for
embeddings and reranking (`/v1/score`'s also points at crit's [fixed-text scores](#fixed-text-scores));
and `/v1/audio/*`, `/v1/images/*`, `/v1/moderations`, `/v1/files` and `/v1/batches`. Other unknown paths
are a 404 `not_found`.

`/health`'s `ready` is true while the host has an engine loaded, ready or generating, and no model
switch, reload or retire is under way; `?ready=1` (or `true`) answers 503 until it is. After a failed
request retires the host, `ready` stays false until the next request loads the engine again, which
the server does not do on its own: use `?ready=1` to wait out a switch or reload, not to hold back all
traffic. A switch its client cancels leaves `ready` true once the host settles with an engine loaded,
the old one or the new. A model card's `limits` states what a request to it is held to, which clients
otherwise learn from a 400: `context_length`, `max_output_tokens` (the largest output cap,
`context_length - 1`; a larger `max_tokens` is a 400, or [fitted](#fitted-output) to the context for
packages declaring `chat-fit-output-v1`), `max_body_bytes` (8 MiB, images included), `max_stops` and
`max_stop_bytes` (16 strings of 256 bytes), `max_tools` (128 functions), `max_top_logprobs` (20) and
`max_reply_characters` (65536: a reply's text and tool calls together, its reasoning apart; a longer
reply fails). Tools and log-probabilities still need their capabilities.

`model` selects a candidate by `RUN_ID/CANDIDATE_ID` or `CANDIDATE_ID`; any other name
(`"default"`, `"gpt-4o"`) uses the default. An unknown candidate returns 404, one that
fails verification 409. Switching closes the old engine and loads the new one, reusing
compatible model allocations. Requests run one at a time unless the adapter
declares the concurrent request contract below. The others wait in arrival order, or
by `priority`, until they run, their client disconnects, the server stops or the queue
refuses them ([a busy engine](#a-busy-engine): `--max-queue`, `--queue-timeout`);
`--timeout` counts from when a request runs.

| Field | Behavior |
| --- | --- |
| `messages` | Optional leading `system`/`developer` messages (merged into one `system`), then user/assistant turns ending with user, or with the assistant message a request [continues](#continuing-the-final-message). Consecutive same-role turns are joined with a blank line. A `system` or `developer` message after the first turn becomes user text marked `System: ` in its place, joined to the user message after it (else the one before it), or a user turn of its own after an assistant reply or tool result; the adapter still receives at most one leading `system`. Content is a string or ordered `text`/`image_url` parts; images require `chat-images-v1` |
| `continue_final_message`, `add_generation_prompt` | vLLM's fields: `true` with `false` [continues](#continuing-the-final-message) the final assistant message, for packages declaring `chat-continue-v1` (otherwise 400 `unsupported_parameter`). Their defaults, `false` and `true`, change nothing; `continue_final_message: true` without `add_generation_prompt: false`, or the reverse, is 400 |
| `max_tokens`, `max_completion_tokens` | Output cap, 1 to `context_length - 1`; default 1024 or less to fit. At or above `context_length` it is [fitted](#fitted-output) for packages declaring `chat-fit-output-v1` |
| `stream`, `stream_options.include_usage` | SSE: role chunk, text chunks, finish chunk, optional usage chunk, `[DONE]`; `: ping` comments while the engine is quiet ([keep-alive pings](#keep-alive-pings)). Any other `stream_options` key is a 400 (ignored with `--lenient`) |
| `stop` | Up to 16 strings of at most 256 UTF-8 bytes (Core's limits, in every dialect). Text before the first completed stop returns with `finish_reason: "stop"` (`crit_core.text.StopMatcher`); the turn is cancelled, the engine stays loaded. With tools, stops are matched in the text only ([tool calls](#tool-calls)) |
| `include_stop_str_in_output` | vLLM's: `true` keeps the stop string that ended the reply at the end of its text, streamed as its last piece; default `false` |
| `n` | Must be 1; omitted or null keeps 1 |
| `response_format` | `text`, `json_object` or `json_schema` ([structured output](#structured-output)); with tools only when `tool_choice` is `none` |
| `structured_outputs` | vLLM's constraint: one of `regex`, `choice`, `grammar` or `json` ([structured output](#structured-output)); vLLM's earlier `guided_regex`, `guided_choice`, `guided_grammar` and `guided_json` too; with tools only when `tool_choice` is `none` |
| `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias` | Per-request overrides when the package declares support for that field; otherwise 400 (ignored with `--lenient`). Omitted or null fields keep engine defaults |
| `typical_p`, `tfs_z`, `mirostat`, `dry_multiplier`, `xtc_probability` | llama.cpp's and Ollama's samplers, which crit's sampling law does not have: the value that turns one off (1, 1, 0, 0 and 0) is a no-op, any other valid value 400 (ignored with `--lenient`) |
| `tools`, `tool_choice`, `parallel_tool_calls` | Function calls for packages declaring `chat-tools-v1`; `tool_choice` `auto` or `none`, and `required` or a named function for packages that also declare `chat-tool-choice-v1` (otherwise 400 `unsupported_parameter`; [ResidentChat](#forced-calls) declares it when its engine decodes to a grammar). Tool-result history is preserved |
| `logprobs`, `top_logprobs` | [Token log-probabilities](#token-log-probabilities) for packages declaring `chat-logprobs-v1`: every token's own and up to 20 alternatives. Not with `stop` |
| `chat_template_kwargs`, `reasoning_effort` | [Thinking controls](#thinking-controls) for packages declaring `chat-template-kwargs-v1`: the chat-template variables the package declares (`enable_thinking`, `thinking`, `reasoning_effort`), and `reasoning_effort` `none`, `minimal`, `low`, `medium`, `high` or `xhigh` mapped onto them; otherwise 400 `unsupported_parameter` (ignored with `--lenient`) |
| `crit` | crit's own extension: `{"decode": "plain"}` decodes the reply one token per target pass, without drafting, for packages declaring `chat-plain-v1` ([plain decoding](#engine-facts-and-plain-decoding)); `{"timeout": SECONDS}`, a positive number, limits this request's turn as `--timeout` does, at most `--timeout` (a larger value keeps it): a turn that runs out is a 504 and a cooperative engine stays loaded |
| `priority` | vLLM's extension: an integer, default 0 (also when null). Requests waiting for the engine go lower values first, then in arrival order, and are reordered only among requests for the same model: a request never moves ahead of a queued model switch, nor a switch ahead of requests queued before it, so priority adds no model loads. A running request is never preempted, and a higher value waits while lower ones keep arriving. Anthropic and Ollama requests, whose APIs have no such field, and fixed-text scores wait at 0 |

Responses carry `usage` when measured and `crit_telemetry`: prompt and output tokens, the
prompt tokens the engine did not compute (`usage.prompt_tokens_details.cached_tokens`: its
telemetry's prompt tokens minus `prefill_tokens`, 0 when it computed them all), each
phase's timed tokens and milliseconds as the engine reports them (`prefill_tokens`,
`prefill_ms`, `decode_tokens`, `decode_ms`; null for a phase it did not measure, see
[Response and telemetry](#response-and-telemetry)), the engine's prefill and decode
rates, TTFT, and `tokens_per_round` (output tokens per target
pass) when the engine reports its decode rounds, with `accepted` (drafts accepted in each of
those rounds), `verify_ms` and `draft_ms` (device time in verify passes and draft steps) when
it reports them. A stream's usage chunk carries both.
Sampling overrides are validated before generation, including for streaming requests.
Unknown parameters and unsupported generation controls return 400 rather than being
silently ignored, unless the server runs with [`--lenient`](#lenient-requests). Only client metadata
(`user`, `metadata`, `store`, `service_tier`, `prompt_cache_key`, `prompt_cache_retention`,
`safety_identifier`) is ignored and listed in `X-Crit-Ignored-Parameters`; no storage, service tier or
cache routing is applied.
`audio`, `modalities`, `prediction` and legacy `functions`/`function_call` return 400
`unsupported_parameter` unless null (`--lenient` ignores `prediction`), as do `reasoning_effort` and
`chat_template_kwargs` for a package without `chat-template-kwargs-v1` ([thinking
controls](#thinking-controls), which `--lenient` ignores there), and an unsupported `response_format`
returns 400. Errors use the OpenAI shape, `type` `server_error` from 500 up
and `invalid_request_error` below; timeout is 504, a stopping or blocked engine 503, engine failure 500
(the next request reloads). A prompt that, with
`max_tokens`, does not fit `context_length` is 400 with code `context_length_exceeded` and OpenAI's
message ("This model's maximum context length is N tokens. However, you requested M tokens (P in the
messages, O in the completion)..."), which clients match to compact the conversation; the engine stays
loaded. A stream has sent its headers before the engine sees the prompt, so there an engine error, this
one included, is an error event before `[DONE]`: the `error` object a response would carry, with its
type and code.
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
the next request uses the connection. Over HTTPS the encrypted stream cannot be peeked,
so any bytes the client sends while its request runs, a TLS close_notify or a pipelined
request, cancel the turn. Only the connection's end or a socket error closes it, so a
pipelined or keep-alive request that follows is still answered.

Cancellation must reach the server: the client must abort the HTTP request, and
any proxy in between must propagate the disconnect, or the client
[aborts the request by its id](#request-ids-and-aborting-requests). A UI that only stops displaying
text while leaving its HTTP request open has not cancelled the server request.
The cancellation regressions in `tests/test_api_cancellation.py` exercise ordinary
Python HTTP client closes and curl request timeouts in both response modes. They
also check that cancelled queue entries never run, cooperative adapters retain
their weights, and uncooperative adapters retire before the next turn.

This changes clients that previously sent ignored sampling values to old packages:
omit those fields to retain the frozen defaults, or build a new package with an
adapter that applies and declares them. Never edit a sealed package's manifest.

### Lenient requests

Chat UIs often send fields an engine does not apply: sampling presets, llama.cpp's samplers, newer
OpenAI options. `crit serve --lenient` ignores them instead of answering 400: unknown top-level keys (on
`/v1/responses`, `/v1/messages`, `/api/chat` and `/api/generate` too), unknown `stream_options` keys and
unknown Ollama `options`, `prediction`, sampling fields the package does not declare, the extra samplers when not off,
and the [thinking controls](#thinking-controls) the package cannot apply: `reasoning_effort` and
`chat_template_kwargs` without `chat-template-kwargs-v1`, and with it a template variable crit does not
set or the package does not declare and a control its template cannot take.
Each is listed in `X-Crit-Ignored-Parameters`, and the server logs each ignored parameter once
(`warning: ignoring request parameter top_p; ...`). The header and the log carry names as sent, so only a
name of 1 to 64 letters, digits, `_`, `.` or `-` is ignored; any other unknown name is still a 400, as is
such a name among the `output_config` settings and `thinking` fields `/v1/messages` ignores. Values crit
knows are still checked: `top_p: 1.5` or `mirostat: 3` is still a 400. Features that change what the
reply is stay refused: `functions`, `function_call`, `audio`, `modalities`, `n` above 1, tools, images,
`logprobs`, `response_format` and a forced `tool_choice` without the package's capability, Anthropic's
`mcp_servers` and `container`, and Ollama's `raw` without `completion-v1`, `suffix`, `template` and
`context`. `POST /v1/crit/score` stays strict. Strict is the default, and `crit run ... eval` and `crit run
measure` always serve strictly; Eval sends only the sampling fields `/v1/models` lists as applied, so a
lenient server measures the same.

### Fitted output

Agent clients (Claude Code, Cline, Aider) send the largest output budget a model allows. For a
package declaring `chat-fit-output-v1` (ResidentChat packages declare it), a `max_tokens` at or
above `context_length` is fitted instead of refused, in every dialect (Ollama's `num_predict` -1
and -2 ask for exactly this): the engine generates at most the room its prompt leaves in the
context and finishes with `length` when it fills it. A reply crit serve fitted carries
`X-Crit-Max-Tokens-Applied`, the cap the engine applied (`context_length` minus the prompt's
tokens); other replies, whose cap is the one asked for, carry none. A stream has sent its headers
before the engine counts the prompt, so it has no such header; its `length` finish and usage show
where the context ended it. A prompt that fills the context by itself is still refused, as above. A
cap below `context_length` is never fitted, so Eval's requests, which always leave their output
room, reach the engine unchanged. Other packages, such as those sealed before the capability, keep
refusing a cap at or above `context_length` with 400.

### A busy engine

Requests the engine cannot take yet wait in a queue, in arrival order or by `priority` (lower values
first, among requests for the same model). A request that finds `--max-queue` requests already waiting (64 by default), or that waits longer than `--queue-timeout` (no limit by
default), is refused before the engine sees it, in the words each API uses for a busy server: OpenAI
clients and `/v1/crit/score` get a 429 (`type` `requests`, `code` `rate_limit_exceeded`), Anthropic
clients a 529 `overloaded_error` and Ollama clients a 503 `{"error"}`, each with `Retry-After: 1`,
which the OpenAI and Anthropic SDKs wait before they retry (Claude Code retries a 529 too). The engine
stays loaded and the queue moves on. A stream sends its status and headers only once the engine takes
the request, so a refused stream gets the same status. `--timeout` starts when the engine takes the
request, and `crit_serve_requests_waiting` in the [metrics](#metrics) counts the queue. `crit run ...
eval` serves without either limit, and Eval sends one request at a time.

### Keep-alive pings

A stream that has sent nothing for `--sse-ping-interval` seconds (default 15) gets a keep-alive
ping, so that proxies with idle timeouts and clients that abort a silent stream do not close it, and
so cancel the turn, while a model loads, a long prompt prefills or a package without `chat-stream-v1`
generates its whole reply. OpenAI streams, Responses streams included, get the SSE comment `: ping`,
which SSE clients skip; Anthropic streams get Anthropic's `event: ping` with `data: {"type": "ping"}`.
Pings start after the role chunk, `response.in_progress` or `message_start`, which a stream sends once
the engine takes the request: a request still [waiting for the engine](#a-busy-engine) has no status
yet, so it gets none, and `--queue-timeout` bounds that wait instead. Pings go out only between whole
events, and stop before the finish chunk and `[DONE]`, the closing Responses events, or `message_delta`
and `message_stop`; `--sse-ping-interval 0` turns them off. Ollama's newline-delimited JSON has no such
line, so Ollama streams get none. Pings are `crit serve`'s alone:
`crit run ... eval` and `measure` serve their candidate without them, and Eval's [request
bound](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md#engines-behind-a-server) counts a ping as no answer, so a request whose engine
stalls still stops Eval as hung, `--url` included.

### Request ids and aborting requests

Every response carries `X-Request-Id`, crit's own id of the request (32 hex digits, a new one for
each request on a connection). Error bodies carry the same id as a top-level `request_id` in each
dialect's shape (Anthropic's API has the field; the OpenAI and Ollama shapes gain it), and so does a
stream's error event. A chat request the engine failed keeps its records under
`requests/<request_id>/` in the runtime directory ([ownership and recovery](#ownership-and-recovery)),
so a client's failure can be matched to them. A client's own `X-Request-Id` (1 to 128 visible ASCII
characters) is written to the server log next to crit's id and can name its request for an abort; it
never names records. With `--cors-origin`, browsers may send `X-Request-Id` and read the response's.

`POST /v1/crit/abort` (behind the API key when one is set) cancels chat and score requests in flight,
[loads](#unloading-the-engine) (`POST /v1/crit/load`, an Ollama request without a prompt) and
[tokenize requests](#tokenizing) waiting to load their candidate, one by `{"request_id": ID}` or every
one by `{"all": true}`. `ID` is crit's id, which a stream's
headers carry before its first event, or the `X-Request-Id` the client sent, which names a request
before any of its response has arrived. An abort sets the cancellation a closed connection sets: a
running turn ends, and its client gets a 499 `Request cancelled` (a stream gets that error as its
last event before `[DONE]`), and a request [waiting for the engine](#a-busy-engine) leaves the queue
without running, its client, a stream's too, getting the 499 as the response's status.
As after a closed connection, a cooperative adapter keeps its weights loaded and one that does not
settle within the grace period is retired. The answer is
`{"object": "crit.abort", "found": true, "request_ids": [...]}`, crit's ids of the requests it
cancelled; an id no request in flight has is `"found": false`, not an error.

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
arguments written so far and `finish_reason: "length"`, as OpenAI's API does, and one it cuts
before a forced call (`required` or a named function) has begun returns no call: act only on the
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

Stop strings (`stop`, Anthropic `stop_sequences`, Ollama `options.stop`) are matched in the reply's
text, never in a call's arguments. One that completes ends the turn there: the reply keeps the text
before it and the calls the model wrote before it (`finish_reason: "tool_calls"`, else `"stop"`),
and a call written after it is dropped. A stop never cuts a call, not even in a stream: a
`chat-tool-stream-v1` package streams text and calls in the order the model wrote them and each call
whole before the text after it ([adapter contract](#strict-and-streaming-tools-for-adapters)), and a
turn with text inside an unfinished call fails. While calls may be made (`tool_choice` other than
`none`) a stop therefore needs `chat-tool-stream-v1` (ResidentChat declares it when the package
streams); for any other package a stop with tools is a 400 `unsupported_parameter`. A
`response_format` (Anthropic `output_config.format`) or `structured_outputs` applies to a tool
request only with `tool_choice: "none"`, when the reply holds no calls; with calls possible it is
a 400, and so is Ollama's `format` with `tools`, since Ollama has no `tool_choice`.

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
`max_tokens` cuts the reply inside it, the call keeps the arguments written so far. The model
decodes freely, so it may or may not call: `tool_choice` is `auto` or `none` unless the engine
[forces calls](#forced-calls). Strict tools need an adapter with `chat-tools-strict-v1`, which
ResidentChat does not declare: under `auto` the reply is free text, which no grammar of calls covers.

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

#### Forced calls

An engine whose decoder keeps a reply to a `crit-byte-grammar-v1` grammar, as an adapter applies
`response_format` ([structured output for adapters](#structured-output-for-adapters)), sets
`ResidentChat.grammars = True` and declares `capabilities(text, sampling, grammars=True)`. That adds
`chat-tool-choice-v1` when crit registers the call syntax and the tokenizer spells its markers as
text, since a grammar cannot write a special token (`crit_core.serving.forces_calls`). ResidentChat
then answers `tool_choice` `required`, a named function, and Anthropic's `any` and `tool` by
passing `generate_tokens` a `grammar`, a `crit-byte-grammar-v1` artifact from
`crit_core.text.tools.call_grammar` of a reply that is nothing but calls, written as the template
writes them:

- calls to an offered function, or only to the named one; one call, or with `parallel_tool_calls`
  one or more;
- arguments shaped by the function's `parameters` as far as their structure goes: declared
  properties in their order (the required ones always, no others), their types, `enum` and
  `const`, array items, and any JSON value where the schema says no more (`$ref`, `anyOf`). For
  `<function=...>` blocks a parameter's value is text unless its schema declares another type;
  JSON takes an optional space after `,` and `:`;
- no text or string holding the marker that would end the block early (`</tool_call>`, and
  `</parameter>` in `<function=...>` blocks);
- reasoning first, in the template's markers, when the tokenizer spells them as text. A generation
  prompt that opens the reasoning makes the reply close it before calling; when a grammar cannot
  write that closing marker, ResidentChat refuses the request with `RequestRejected`.

Without `grammars`, and for a package's own `ToolFormat`, whose syntax crit cannot write, the
package does not declare `chat-tool-choice-v1`: crit serve answers those choices with a 400 before
the request reaches the engine, and ResidentChat refuses them with `RequestRejected`; Core's
`resident_engine` does not apply grammars. The grammar covers the whole reply, reasoning included,
so budget the matcher's limits for it.

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

With `--reasoning-alias` the message and each reasoning delta carry the same text as
`reasoning` too, the name vLLM and OpenRouter use. It is off by default: a client that
reads both names, or joins every text field of a delta as crit Eval does, would read the
reasoning twice. In a tool conversation an assistant message may give its reasoning as
`reasoning_content` or `reasoning`; the adapter receives it as `reasoning_content`. A
`"reasoning": null`, which those servers' replies carry when there is none, is ignored,
and a message giving both with different text is a 400.

### Thinking controls

Hybrid reasoning models turn thinking on or off, or set its effort, through a variable of their chat
template: `enable_thinking` (Qwen3, GLM) or `thinking` (DeepSeek-V3.1, Granite), booleans, and
`reasoning_effort` (gpt-oss), `low`, `medium` or `high`. A package declares `chat-template-kwargs-v1` and,
for each variable its templates read, `chat-template-enable-thinking-v1`, `chat-template-thinking-v1` or
`chat-template-reasoning-effort-v1` ([ResidentChat](#chat-template-variables-for-adapters) declares them
from the templates). Each dialect's control then sets them for one request, over the variables the
candidate's configuration freezes:

| Dialect | Control | Sets |
| --- | --- | --- |
| OpenAI | `chat_template_kwargs` | The variables it names, each a declared one with a value it takes |
| OpenAI | `reasoning_effort` | `none` turns thinking off; `minimal` (as `low`), `low`, `medium`, `high` and `xhigh` (as `high`) turn it on at that effort |
| OpenAI Responses | `reasoning.effort` | As `reasoning_effort`, which it becomes |
| Anthropic | `thinking.type` | `enabled` and `adaptive` turn thinking on, `disabled` and `between_tools` off; `budget_tokens` and the other fields are not applied and are listed in `X-Crit-Ignored-Parameters` |
| Anthropic | `output_config.effort` | `low`, `medium`, `high`, `xhigh` and `max` (both as `high`) turn thinking on at that effort |
| Ollama | `think` | `true` and `false` turn thinking on and off; `low`, `medium` and `high` turn it on at that effort |

Turning thinking on or off sets the switch (`enable_thinking` or `thinking`); an effort also sets
`reasoning_effort` when the template reads it. A template with `reasoning_effort` and no switch always
thinks, so turning thinking off there is a 400. So are a variable the package does not declare, a value
it or the table above does not take, and `reasoning_effort` and `chat_template_kwargs` setting one
variable differently. Without `chat-template-kwargs-v1`, OpenAI's two fields (and the Responses API's
`reasoning.effort`) are a 400 `unsupported_parameter`, and Anthropic's and Ollama's controls are ignored
and listed whatever they hold, as before. [`--lenient`](#lenient-requests) ignores and lists, instead of refusing, OpenAI's two fields
without `chat-template-kwargs-v1` (whatever they hold) and, with it, a variable crit does not set or the
package does not declare (once its value is checked) and a control the template cannot take, such as
turning off one that always thinks; a value no table above takes, and the two fields disagreeing, stay a
400. A request without controls sets no variable, so Eval's requests render as they did.
`/api/show` lists `thinking` among the capabilities of a package that declares a variable.

### Token log-probabilities

With `logprobs: true` a reply's `choices[0].logprobs.content` lists every token the
model generated, in order: reasoning and tool-call tokens included, the stop token that
ended the reply left out. Each entry is OpenAI's `{token, logprob, bytes,
top_logprobs}` with `crit_token_id`, where `top_logprobs` holds the `top_logprobs` (0 to
20) most likely tokens at that position, ranked, in the same shape. `bytes` are the
token's UTF-8 bytes; `token` is their text, with U+FFFD where a token ends inside a
character; `crit_token_id` is the token's id as the model's tokenizer numbers it, as in
[scores](#fixed-text-scores). A
log-probability is natural-log, of the raw logits before temperature, penalties, bias
and constraints; a zero probability reads as -9999. When streaming, the entries arrive
with the finish chunk, since the engine reports them in its final response. A request
with `stop` strings is refused, since its turn is cancelled where a stop completes,
before the engine reports its tokens. The Ollama API returns the same entries as its
`logprobs` ([other API dialects](#other-api-dialects)).

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
gate uses it ([crit Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md)). A score runs alone on the engine, as Core runs
scores, and is not streamed.

### Raw-prompt completions

`POST /v1/completions` is OpenAI's legacy completions endpoint, for packages declaring
`completion-v1` (ResidentChat declares it for any text-only decoder): the engine encodes the
prompt with the model's tokenizer and continues it without the chat template, a system message
or tools. The reply is all the text the model writes, with no reasoning split from it, and may
be empty. Without `completion-v1` the endpoint is a 404 `not_supported`.

| Field | Behavior |
| --- | --- |
| `prompt` | One prompt: a string, encoded with the tokenizer's special tokens (the BOS its post-processor adds, once when the text already starts with it; control-token text reads as those tokens), or a list of token ids, sent as given; either may come as a list of one. Several prompts are a 400 |
| `max_tokens`, `stop`, `include_stop_str_in_output`, sampling fields, `crit`, `priority` | As in [chat](#endpoints), a `max_tokens` at or above `context_length` [fitted](#fitted-output) for packages declaring `chat-fit-output-v1` |
| `stream`, `stream_options.include_usage` | SSE of `text_completion` chunks: text chunks, finish chunk, optional usage chunk, `[DONE]`; `: ping` comments while the engine is quiet ([keep-alive pings](#keep-alive-pings)) |
| `logprobs` | An integer, 0 to 20, for packages declaring `chat-logprobs-v1`: the choice's `logprobs` is `{tokens, token_logprobs, top_logprobs, text_offset}`, each token's text, its log-probability as in [token log-probabilities](#token-log-probabilities), its alternatives as `{text: logprob}` and its character offset in the choice's text. Unlike chat, the stop token that ended the reply is listed too, as in OpenAI's legacy endpoint, though its text is not in the choice's. When streaming they arrive with the finish chunk. Not with `stop` |
| `echo` | The choice's text starts with the prompt's (the given text, or the ids decoded). With `logprobs` the prompt's tokens come first, the first with `null` log-probability and alternatives, the others from the engine's score path (one prefill reading every position, [fixed-text scores](#fixed-text-scores)), so the package must also declare `chat-score-v1`; otherwise 400. This is how lm-eval's `local-completions` scores a continuation (`echo`, `logprobs: 1`, `max_tokens: 1`, reading the rows between the context and the one generated token, which may be the stop token). Not with `stream` or `stop` |
| `suffix` | 400 `unsupported_parameter`: fill-in-the-middle is not supported yet |
| `n`, `best_of` | Must be 1; omitted or null keeps 1 |

Other fields, `messages` and `response_format` among them, are a 400 unknown parameter (ignored with
[`--lenient`](#lenient-requests)), and client metadata is ignored and listed, as in chat. A reply is `{"id": "cmpl-...", "object": "text_completion", "created", "model",
"system_fingerprint", "choices": [{"text", "index": 0, "logprobs", "finish_reason"}]}` with
`usage` and `crit_telemetry` as in chat. A prompt that, with its output cap, does not fit the
context is a 400 `context_length_exceeded` with `param: "prompt"`, in the legacy endpoint's words.
Request ids, aborts, a busy queue's 429 and the metrics are those of the OpenAI dialect.

### Engine facts and plain decoding

While a candidate's engine is loaded, its `/v1/models` card carries `engine_facts`, what
the engine reports for Eval's headroom and exactness: `speculative` and `plain` (booleans),
`draft_width`, `bytes_per_pass`, `draft_bytes_per_step`, `draft_ms` and
`kv_bytes_per_token` (non-negative numbers) and `prompt_activations` (the activation format of
the quantized matrices' prompt GEMMs, `bf16`, `fp16_acc`, `fp8` or `stored`, for every quantized
format, per stored format, or as shares of a format's FLOPs: crit-eval's prefill ceilings by
activation format and their reachable limit), each only when the engine reports it in a valid
form; `null` when the candidate is not loaded or reports none. A speculative engine that declares
`chat-plain-v1` decodes a request carrying `"crit": {"decode": "plain"}` one token per
target pass without drafting, which is how Eval compares speculative output with plain
greedy on the served engine.

### Continuing the final message

A conversation may end with an assistant message that the model continues instead of starting
a new reply: a prefilled start, or a reply a client wants to resume. The request carries vLLM's
`continue_final_message: true` with `add_generation_prompt: false`; a final assistant message
without them is a 400. The [Anthropic](#other-api-dialects) prefill (a final assistant
message) and an Ollama chat that ends with an assistant message are continued without a flag, as
those APIs do. It needs a package declaring `chat-continue-v1`, which
[ResidentChat](#continuing-the-final-message-for-adapters) packages do; otherwise it is a 400
`unsupported_parameter`. The model's chat template renders the message left open (Core's
`continue_final_message`) where the generation prompt would start a new reply. The response holds
only the continuation, which may be empty when the message was already complete; `max_tokens`,
`stop` and `logprobs` apply to the continuation, and the continued message counts as prompt
tokens. A continuation cannot carry `tools`, `response_format` or `structured_outputs` (400),
since calls and constraints apply from the start of a reply.

### Tokenizing

Packages declaring `chat-tokenize-v1` (`crit_core.serving.capabilities()` declares it for ResidentChat
adapters) expose the engine's own tokenizer and chat template, so a client can check a prompt against
`context_length` before sending it:

| Endpoint | Body | Reply |
| --- | --- | --- |
| `POST /tokenize` | `prompt` with `add_special_tokens` (default true: the tokenizer's own special tokens, such as a BOS), or `messages` with the chat request's `tools`, `tool_choice`, `parallel_tool_calls`, `chat_template_kwargs`, `reasoning_effort`, `continue_final_message` and `add_generation_prompt` | `{"tokens", "count", "max_model_len"}`, as vLLM answers |
| `POST /detokenize` | `tokens` | `{"prompt"}`, special tokens included |
| `POST /apply-template` | `messages` and the same chat request fields | `{"prompt"}`, the chat template's text, as llama-server answers |
| `POST /v1/messages/count_tokens` | An Anthropic Messages request's `model`, `system`, `messages`, `tools`, `tool_choice` and `thinking` | `{"input_tokens"}`, which Claude Code reads |

`model` selects the candidate as for a chat request. Messages, tools, [thinking
controls](#thinking-controls) and a [continued final message](#continuing-the-final-message) (an
Anthropic prefill included) are validated as a chat request's and rendered as the engine renders a
reply's prompt, generation prompt included unless the final message is continued, so a count is the
prompt tokens that request would report; image parts are refused, since the engine's vision
processing places them. What a chat request would list in `X-Crit-Ignored-Parameters`, such as
thinking controls the package does not apply, is listed there too, and an unknown parameter is a 400
(ignored with `--lenient`). These requests run on the loaded engine at once, beside any generation and never
queued behind it; one for a candidate that is not loaded waits in the queue and loads it as a reply
would, a [full queue](#a-busy-engine) refusing it as one, and like a reply it leaves without loading
when its client disconnects first or [an abort](#request-ids-and-aborting-requests) names it. A
refused input is a 400 and an adapter failure a 500, and the engine stays loaded either way. A
response is at most 16 MiB; a text with more tokens is a 400. Without `chat-tokenize-v1` the paths
answer 404 with code `not_supported` (`not_found_error` on `/v1/messages/count_tokens`).

## Other API dialects

The same engine answers three more APIs, so the tools people already run need only a base URL. Every
request is translated into the chat request above and validated the same way, so the capabilities a
package declares (tools, images, sampling fields, structured output) gate it in every dialect, and the
API keys set with `--api-key` or `--api-key-file` guard every endpoint except `/health`, `/` and `/api/version`.

**Anthropic Messages** (`POST /v1/messages`, [crit_api/anthropic.py](../api/crit_api/anthropic.py)): the
API Claude Code and the Anthropic SDKs speak. Point them at the server (`ANTHROPIC_BASE_URL`, or the
SDK's `base_url`) with the key in `x-api-key` or as a Bearer token; `anthropic-version` is accepted.
`system` (a string or text blocks) becomes the leading system message, content blocks become text and
image parts, `thinking` blocks the assistant turn's `reasoning_content`, as an OpenAI client sends a
turn's reasoning back (signatures are ignored and `redacted_thinking` blocks dropped), `tool_use` blocks
become tool calls and `tool_result` blocks tool messages (tool results first, then the turn's text, as
Anthropic orders them). A tool result's content may hold text and image blocks, the images on the tool
message as the OpenAI path takes them (they need `chat-images-v1`), and `is_error: true` puts `Error: `
before its text. `tools` with an `input_schema` become functions, `strict` included, which needs
`chat-tools-strict-v1` as on the OpenAI path. `tool_choice` `auto`, `any`, `tool` and `none` become
`auto`, `required`, the named function and `none` (`disable_parallel_tool_use` sets
`parallel_tool_calls`), `stop_sequences` becomes `stop`, `temperature`, `top_p` and `top_k` pass
through, `output_config.format` of type `json_schema` becomes `response_format`, and `thinking` and
`output_config.effort` are [thinking controls](#thinking-controls). A final assistant message is
prefill: the reply [continues](#continuing-the-final-message) it and holds only the continuation.
`max_tokens` is required, and
[fitted](#fitted-output) at or above the context for packages declaring `chat-fit-output-v1`. Since the
result is validated as a chat request, a few things differ from Anthropic's service: `tool_choice`
`any` or `tool` is a 400 unless the package declares `chat-tool-choice-v1` (ResidentChat packages do
when their engine [decodes to a grammar](#forced-calls)), `stop_sequences` takes up to 16 strings of at most 256 bytes and,
with `tools`, needs `chat-tool-stream-v1` and is matched in the text only, and `output_config.format` with `tools`
needs `tool_choice` `none` ([tool calls](#tool-calls)), and a prefill needs `chat-continue-v1` and
cannot carry `tools` or `output_config.format`. `cache_control` on blocks and tools is dropped
without being listed. `metadata`, `service_tier`, `betas` and the other service settings
are accepted and ignored (listed in `X-Crit-Ignored-Parameters`); `mcp_servers`, `container`, document
blocks and Anthropic-defined tools (web search, code execution) return 400.
`/v1/messages/count_tokens` counts a request's prompt as `input_tokens` for packages declaring
`chat-tokenize-v1` ([tokenizing](#tokenizing)). The reply is a Messages response: `thinking` (the
engine's reasoning, with an empty `signature`), `text` and `tool_use` blocks, `stop_reason` `end_turn`, `max_tokens`,
`stop_sequence` (with the sequence) or `tool_use`, and `usage` with the prompt tokens the engine
computed as `input_tokens` and the ones its prompt cache served as `cache_read_input_tokens`. With
`stream: true` the reply is the Messages event stream: `message_start` (its usage zero, since the
engine reports its token counts when it finishes), `content_block_start`,
`content_block_delta` (`text_delta`, `thinking_delta`, `input_json_delta`) and `content_block_stop` per
block in the order the engine produces them, one block open at a time, so a tool_use block closes when
the next call starts (an engine that interleaves its calls' deltas has the later call wait until the
open call's arguments are whole JSON), `message_delta` with the stop reason and usage,
`message_stop`, with [`ping` events](#keep-alive-pings) while the engine is quiet; an engine error is an
`error` event. Errors use Anthropic's shape, `{"type": "error",
"error": {"type", "message"}}`, with the type Anthropic gives the status, in a response and an `error`
event alike: `invalid_request_error` (400), `not_found_error` (404), `api_error` (500),
`overloaded_error` (503, a stopping or blocked engine, and 529, a [full queue](#a-busy-engine), which
clients retry) and `timeout_error` (504).
A prompt over the context is a 400 (or `error` event) `invalid_request_error` "prompt is too long: P
tokens > N maximum", the words Claude Code compacts on ("input length and `max_tokens` exceed context
limit: P + O > N, ..." when only the output does not fit), and the engine stays loaded.

**OpenAI Responses** (`POST /v1/responses`, [crit_api/responses.py](../api/crit_api/responses.py)): the
API Codex CLI, the OpenAI Agents SDK and the Responses modes of LangChain and LibreChat speak, served
statelessly, as Ollama serves it. `instructions` becomes the leading system message and `input`
(a string, or a list of items) the conversation: `message` items (`user`, `assistant`, `system` and
`developer`, with `input_text`, `input_image` by `image_url`, and `output_text` parts; a `system` or
`developer` message after the first turn becomes marked user text, as in the OpenAI API), `function_call`
items as the assistant's tool calls, `function_call_output` items as tool results, and `reasoning` items
as the next assistant turn's `reasoning_content` (their `reasoning_text` content, else their summary).
Function `tools` (`strict` when the tool sets it; OpenAI's default of `true` is not assumed) and
`tool_choice` (`auto`, `none`, `required`, `{"type": "function", "name"}`) map across, `parallel_tool_calls`,
`temperature` and `top_p` pass through, `text.format` becomes `response_format`, `max_output_tokens`
becomes `max_tokens` ([fitted](#fitted-output) at or above the context for packages declaring
`chat-fit-output-v1`), and `reasoning.effort` becomes `reasoning_effort`, a
[thinking control](#thinking-controls) as on `/v1/chat/completions`. `store`, `metadata`, `include`,
`truncation`, `user`, `prompt_cache_key`, `prompt_cache_retention`, `safety_identifier`, `service_tier`,
`stream_options`, `max_tool_calls` and `reasoning.summary` are accepted and ignored (listed in
`X-Crit-Ignored-Parameters`); an unknown parameter is a 400 (ignored with `--lenient`). crit serve keeps no
responses: `previous_response_id`, `conversation`, `prompt`, `background: true` and `item_reference` items
return 400, as do OpenAI's built-in tools (web search, file search, ...), `input_file` parts,
`text.verbosity` and `top_logprobs`, and `GET`, `DELETE` and `POST .../cancel` on `/v1/responses/{id}`
return 404 `not_supported`. The reply is a Response
(`resp_` id) with `status` `completed`, or `incomplete` with `incomplete_details.reason` `max_output_tokens`
(and its last item `incomplete`) when the output cap cut it. Its `output` holds a `reasoning` item whose
`summary` is the engine's reasoning text, a `message` item with `output_text`, and a `function_call` item
per call (`call_id`, `name`, `arguments`), in that order; `usage` has `input_tokens` with
`input_tokens_details.cached_tokens` (the prompt tokens the engine's cache served) and `output_tokens`, while
`output_tokens_details.reasoning_tokens` and `input_tokens_details.cache_write_tokens` are 0, since engines
report neither the reasoning share of their output nor cache writes; `crit_telemetry` comes too. With `stream: true` the reply is the Responses event stream, every event
numbered by `sequence_number`: `response.created` and `response.in_progress` once the engine takes the
request (a [full queue](#a-busy-engine) refuses it first with OpenAI's 429), then for each item in the
order the engine produces them `response.output_item.added`, its part and delta events
(`response.reasoning_summary_part.added`, `response.reasoning_summary_text.delta` and `.done`,
`response.reasoning_summary_part.done`; `response.content_part.added`, `response.output_text.delta` and
`.done`, `response.content_part.done`; `response.function_call_arguments.delta` and `.done`) and
`response.output_item.done` (function calls close when the reply ends), and last `response.completed` or
`response.incomplete` with the whole Response, with [`: ping` comments](#keep-alive-pings) while the engine
is quiet. An engine failure ends the stream with `response.failed`, which carries the `request_id` and
whose `error.code` is `context_length_exceeded` for a prompt over the context (the code Codex compacts
on); other errors use
OpenAI's shape, a context overflow the 400 `context_length_exceeded` of `/v1/chat/completions`, and the
engine stays loaded.

**Ollama** (`GET /` answers "Ollama is running", `/api/version`, `/api/tags`, `/api/ps`; `POST
/api/chat`, `/api/generate`, `/api/show`; [crit_api/ollama.py](../api/crit_api/ollama.py)): the API the
local chat UIs look for. `/api/version` reports Ollama release `0.12.0` (clients compare it with the
Ollama version a feature needs) and crit's own as `crit_version`. Models are listed under the ids `/v1/models` shows (and the default name; a catalog's aliases are not listed but select their model),
each with its package's sha256 as `digest`; `/api/ps` lists the one whose engine is loaded, matched by run, candidate
and package, with its `context_length`, `size_vram` (device bytes) and an `expires_at` of `2318-01-01T00:00:00Z`, since
it stays loaded until it is unloaded or the server stops (with `--idle-unload`, the time it unloads unless a request
uses it). A request to `/api/generate` without a `prompt`, or to `/api/chat` without `messages`, loads its model and
answers one object with `done_reason` `load`; with `keep_alive` 0 (`0`, `"0"`, `"0s"`) it [unloads](#unloading-the-engine)
the engine instead and answers `done_reason` `unload`. `keep_alive` 0 on a request with a prompt unloads the engine
once the reply is sent.
`/api/chat` takes Ollama's messages (`images` as bare base64, their type read from the bytes; assistant
`tool_calls` with arguments as objects, given ids when they have none; `tool` messages matched to the
latest assistant's calls in order; a `system` message after the first turn becomes marked user text, as
in the OpenAI API), `tools`, `format` (`"json"`, or a JSON schema) and `options`:
`temperature`, `top_p`, `top_k`, `min_p`, `seed`, `num_predict` (the output cap; -1 and -2, output
until the context is full, are [fitted](#fitted-output) like a cap at or above the context for packages
declaring `chat-fit-output-v1`, and otherwise keep the default cap of 1024 or less to fit, as 0 and other
negative values do), `stop`,
`repeat_penalty`, `presence_penalty` and `frequency_penalty` apply; Ollama's runtime settings
(`num_ctx`, `num_gpu`, `num_thread`, `mirostat`, ...) and a `keep_alive` other than 0 are ignored and
listed; an unknown option is 400 (ignored with `--lenient`). `think` is a [thinking control](#thinking-controls).
A chat that ends with an assistant message is [continued](#continuing-the-final-message), as Ollama does.
`/api/generate` renders `prompt` (and `system`) as one user turn through the model's chat template.
With `raw: true` it continues the prompt as written instead, as a
[raw-prompt completion](#raw-prompt-completions) without the template or a system message: `system`,
`images` and `format` are a 400, and so is `raw` for a package without `completion-v1`; `think`, with
no template to set, is ignored and listed, and `logprobs` true gives each token `top_logprobs`
alternatives, the reply's stop token included. `suffix`, `template` and `context` are refused. Replies
stream by default as newline-delimited JSON objects (`message.content`, `message.thinking` and
`message.tool_calls`, or `response` for generate), the last with `done`, `done_reason` (`stop` or
`length`), `total_duration`, `prompt_eval_duration` and `eval_duration` in nanoseconds from the
engine's own phase times, `prompt_eval_count`, `prompt_eval_cached_count` (the prompt tokens the engine did not
compute: its telemetry's prompt tokens minus `prefill_tokens`, when it reports them) and `eval_count`; `stream: false`
returns one object. With `logprobs: true` (and `top_logprobs`, 0 to 20) the last object carries the reply's
[token log-probabilities](#token-log-probabilities) as `logprobs`, one Ollama `{token, logprob, bytes, top_logprobs}`
with `crit_token_id` per token (`top_logprobs` left out when none were asked for), in a stream too, since the
engine reports them in its final response; they need a package declaring `chat-logprobs-v1` and are refused with
`options.stop`. As in Ollama, `top_logprobs` without `logprobs` is ignored (and listed).
`/api/show` reports the context length as `model_info["crit.context_length"]`, the
`capabilities` (`completion`, `tools`, `thinking`, `vision`) the package declares and the model card's
`limits`. Errors are `{"error": message}` (a context overflow with status 400, or as a stream's last line);
`/api/pull` and the other model-management endpoints return 404, and `/api/embed` and `/api/embeddings` a 404
that says to run a separate embedding server.

## Metrics

`GET /metrics` serves the Prometheus text format (behind the API key when one is set):
`crit_serve_requests_total` counts every chat request (`/v1/chat/completions`, `/v1/responses`,
`/v1/messages`, `/api/chat`, `/api/generate`) and every `/v1/crit/score` request whose body was read and
whose API key was accepted, by dialect (`openai`, `responses`, `anthropic`, `ollama`, and `crit` for
scores) and outcome:
`completed`, `rejected` (a 4xx: the request failed validation, or the engine refused it, such as a
prompt over the context or a [full queue](#a-busy-engine), which Anthropic and Ollama clients get as a
529 or 503), `cancelled` (the client disconnected), `timed_out` (504) or `error` (any other failure).
A request refused for its API key or its body size, and the other endpoints, are not counted. Beside
it come the chat and score requests in flight, the prompt, cached-prompt and completion tokens of
completed chat requests, wall seconds in requests after validation, uptime, the last completed chat
request's prefill and decode rates, time to first token and tokens per round, and
`crit_serve_engine_loaded` with the loaded run, candidate and state
([crit_api/metrics.py](../api/crit_api/metrics.py)). Counters start at zero when the server starts.

Chat and score requests in flight are `crit_serve_requests_running`, whose command went to the engine,
plus `crit_serve_requests_waiting`, queued for admission or for the engine load they need.
`crit_serve_engine_loads_total` counts completed engine loads (the first, one after a retired host,
one per switch) and `crit_serve_model_switches_total` the loads that replaced another candidate's
engine. Fixed-bucket histograms give percentiles:

| Histogram | Each observation | Buckets |
| --- | --- | --- |
| `crit_serve_request_queue_time_seconds` | A chat or score request the engine started: from validation until its command went to the engine | 5 ms to 2,500 s |
| `crit_serve_time_to_first_token_seconds` | A completed chat request's `ttft_ms`, which the engine times from its own admission; a client also waits the queue time | 5 ms to 2,500 s |
| `crit_serve_time_per_output_token_seconds` | A completed chat request's `decode_ms` over `decode_tokens` | 0.5 ms to 2.5 s |
| `crit_serve_e2e_request_latency_seconds` | A chat or score request's wall time after validation, every outcome: the histogram of `crit_serve_request_seconds_total` | 5 ms to 2,500 s |
| `crit_serve_request_prompt_tokens`, `crit_serve_request_generation_tokens` | A completed chat request's prompt and completion tokens | 1 to 1,000,000 |

## Structured output

`response_format` restricts the reply's bytes with Core's
[structured-output service](../core/src/constraints/README.md). It needs a package
declaring `chat-constraints-v1`; otherwise it returns 400. With `tools` it needs
`tool_choice: "none"` ([tool calls](#tool-calls)).

- `{"type": "json_object"}`: one JSON object (surrounding whitespace allowed).
- `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}, "strict": ...}}`:
  compact JSON in the schema's declaration order, compiled with `crit_core.constraints`.
  The [supported subset](../core/src/constraints/README.md#compiling-languages-python):
  closed objects (`additionalProperties: false`), typed arrays, strings with lengths or a
  `pattern`, booleans, null, numbers, integers with both bounds, `enum`, `const`, `anyOf`,
  type lists, provably disjoint `oneOf` and local `$ref`/`$defs`. Anything else
  (`format`, overlapping `oneOf`, `allOf`, open objects) returns 400 with
  `param: "response_format"`, `code: "unsupported_schema"` and the reason. Always strict.

`structured_outputs`, vLLM's field (OpenAI dialect only), takes exactly one of:

- `regex`: Core's byte regex over the whole reply (optional `^` and `$`; groups, `|`, `?`,
  `*`, `+`, `{m,n}`, `\d`, `\w`, `\s`, `\xHH` and classes of ASCII characters; no
  lookaround, backreferences or flags). `.` and classes match one byte, not one character.
- `choice`: a list of strings; the reply is exactly one of them.
- `grammar`: Core's line EBNF, one `name ::= expression` per line with a required `root`:
  JSON-quoted literals, rule names, byte classes, groups, `|`, `?`, `*`, `+`, `{m,n}` and
  `#` comments. It reads like GBNF but is not GBNF (one line per rule, classes match bytes),
  so llama.cpp's top-level `grammar` is not taken (an unknown parameter).
- `json`: a JSON schema, or its JSON text, compiled as `json_schema` above.

`regex`, `choice` and `grammar` need a package declaring `chat-constraints-grammar-v1` and
`chat-constraints-v1`, `json` only `chat-constraints-v1`; otherwise 400
`unsupported_parameter`. A source Core cannot compile returns 400 with the field as
`param` (`"structured_outputs.regex"`), `code: "unsupported_constraint"` and the
compiler's reason. vLLM's earlier `guided_regex`, `guided_choice`, `guided_grammar` and
`guided_json` name the same fields. A request takes one constraint: two fields, or one
with a `response_format` other than `text`, are a 400. Like `response_format` it needs
`tool_choice: "none"` with `tools` and cannot carry a [continued](#continuing-the-final-message)
message. Null fields count as absent; other fields (`json_object`, `whitespace_pattern`,
...) are a 400.

The engine applies the constraint while sampling (the adapter interface carries no
logits). crit serve re-checks every reply the engine ended itself (`finish_reason:
"stop"`, no stop string) and answers 500 if it does not conform; the engine stays
loaded. Replies cut by the cap or a stop string are returned as cut. When streaming, a
failed final check becomes an error event.

## Adapter contract

A package is servable when it declares **`chat-json-v1` and `chat-resident-v1`** and
`launch[0]` is a retained Python module exporting `create_engine(context)`. Build the
adapter on **`crit_core.serving.ResidentChat`**: it renders the conversation, tools and
tool history with the model's chat template (for a family whose checkpoints carry none, the
one crit ships, such as DeepSeek V4's: [Core's text layer](../core/python/crit_core/text/README.md#chat-templates)),
forwards the request's sampling fields that
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
never changes benchmark evidence or passes a gate. [Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/ADAPTER.md) uses a
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
| `context.limits` | Context and RAM/VRAM headroom; `kv_tokens` is the built context, `context` (when present) Eval's [basis](https://github.com/crit-labs/crit-eval/blob/main/docs/CONTEXT.md) |
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
user/assistant ending with user, or with assistant when the request carries
`"continue_final_message": true`, [chat-continue-v1](#continuing-the-final-message-for-adapters)). With
`chat-template-kwargs-v1` it may carry `template_kwargs`
([chat-template variables](#chat-template-variables-for-adapters)). A
[raw prompt](#raw-prompts-for-adapters) carries `prompt` in place of `messages`.

With `chat-tools-v1`, messages also preserve assistant `tool_calls`, tool messages
with `tool_call_id`, and optional `name`/`reasoning_content` (a client's `reasoning`
arrives as `reasoning_content`); tool message boundaries
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
  `ram_headroom_bytes`, `timeout_seconds` (from `--timeout`, or a request's lower
  `crit.timeout`). Tokenize the full templated input first; reject overflow, never
  truncate: raise `crit_core.serving.ContextExceeded` (ResidentChat does) before
  generating. Enforce the output cap.
- `limits.fit_output: true` reaches only a package declaring **`chat-fit-output-v1`**, when
  the client's cap was at or above the context (`max_output_tokens` is then `kv_tokens - 1`).
  Cap the output at `min(max_output_tokens, kv_tokens - prompt tokens)`, stop there and
  return that cap as the response's `max_output_tokens`; raise `ContextExceeded` only when
  the prompt alone fills the context. ResidentChat does this, and `capabilities(text,
  sampling)` declares the capability.
- A `crit_core.serving.RequestRejected` (ContextExceeded is one) raised before generation
  starts refuses the request: the host reports it as rejected, crit serve answers 400 and the
  engine stays loaded. Any other exception fails the turn and retires the host.
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
calls are present, or when it continues the final message. Otherwise it must contain
nonblank text. Every text channel must be UTF-8 encodable without NUL; reasoning has its
own character budget, separate from the combined assistant/tool-call budget.
`max_output_tokens` is present exactly when the request's `limits` carry `fit_output`: the
cap the engine applied, from `output_tokens` to the request's `max_output_tokens`, with
`prompt_tokens` plus the cap within `kv_tokens`. A reply that reaches it finishes as `length`.
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
| `{"type": "grammar", "language", "source", "grammar"}` | The same `GrammarMatcher` over `ByteGrammar::load(grammar)`; `language` (`regex`, `choice` or `ebnf`) and `source` (the client's text or list of choices) are for reference. Only with `chat-constraints-grammar-v1` |

Declare **`chat-constraints-grammar-v1`** as well when the engine applies any compiled
grammar, whatever language it came from: crit serve then sends `structured_outputs`
regex, choice and grammar constraints as type `grammar`. ResidentChat applies no
constraint and declares neither.

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

Declare **`chat-tool-choice-v1`** alongside `chat-tools-v1` only when the adapter
forces a call for `tool_choice` `required` (at least one call) or a named function (calls to that function only),
for example by constraining decoding to the call syntax. Without it crit serve refuses those choices with a 400
`unsupported_parameter` (`param: "tool_choice"`), and the adapter sees only `auto` and `none`. ResidentChat
does it for an engine that applies its `grammar` as structured output's json_schema grammar above
([forced calls](#forced-calls)).

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
to separate text, tool syntax and arguments before emitting. Emit text and calls in
the order the model wrote them, and a call's arguments to their end, a whole JSON
object, before any text after it (calls may still interleave with each other): crit
serve ends a reply at a stop string by this order, keeping the calls emitted before
the stop, and text emitted inside an unfinished call fails the turn. Return the assembled
calls as the final response's `tool_calls` and concatenate only text emits into
`assistant`. Every emitted call must appear unchanged in the final response. A reply that
reaches `max_output_tokens` inside a call returns it with the arguments emitted so far, the
start of a JSON object; crit serve finishes it as `length`.
If no tool is selected, text-only output remains valid under `tool_choice: "auto"`.

The same synchronous-thread, cancellation and callback-error rules as text
streaming apply. Each emitted delta reaches the client as its own chunk, and crit serve
checks it again on arrival. Calls are checked for choice policy, unique ids,
append-only arguments, whole arguments before later text, strict-schema prefix
membership and the shared 65,536-character limit. A callback failure invalidates the
turn even if swallowed by the adapter. Cancelled or failed partial calls are never
committed to a later request. Sampling, constraints and parser state must remain
private to the turn.

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

### Chat-template variables for adapters

Declare **`chat-template-kwargs-v1`** when the adapter renders a request's chat-template variables, and with
it the capability of each variable of `crit_core.serving.TEMPLATE_KWARGS` the model's templates read
(`chat-template-enable-thinking-v1`, `chat-template-thinking-v1`, `chat-template-reasoning-effort-v1`). A
request that sets any carries `template_kwargs`, `{name: value}` with declared variables and values
TEMPLATE_KWARGS gives them; render the template with them over the configured variables, for that request
only. ResidentChat does: `capabilities(text, sampling)` declares the variables its templates read
(`crit_core.serving.template_kwargs`), and `prompt_ids` sets the request's over configuration
`chat_template_kwargs` and refuses any other with `RequestRejected`. An adapter that overrides
`prepare_prompt` keeps this by rendering through `self.prompt_ids(request)`.

### Token log-probabilities for adapters

Declare **`chat-logprobs-v1`** (`capabilities(text, sampling, logprobs=True)`) when the
engine reports, for every generated token, its log-probability and its 20 most likely
alternatives, and set `logprobs = True` on the engine object; the host refuses to load
the package otherwise. A request that asks for them carries `"logprobs": n`, the
alternatives per token (0 to 20). The response then carries one row per generated
token but stop tokens (which a [raw prompt](#raw-prompts-for-adapters)'s rows keep), and every
row's token and alternatives named by their bytes (lowercase hex), so crit serve needs no
tokenizer:

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
`(logprob, [(token, logprob), ...])` row per token. It drops a chat reply's stop tokens,
clamps -inf and Core's rounding tolerance into the range and reads the bytes from the
tokenizer.
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
it. Rows may also come as a `crit_core.serving.ScoreRows`, flat arrays of the log-probabilities
and the alternatives row by row, which reads as the list it stands for. crit serve's score
request offers `"format": "packed"`, and ResidentChat then answers with each part packed
(`packed_scores`: base64 of int32 tokens and float64 log-probabilities, the same values as the
JSON rows), so a long text's rows cross the host's channel without a JSON row per position; an
adapter that implements `score()` itself may keep answering JSON rows.
`crit_core.native.ResidentEngine` sets `scores` when the engine's `ModelInfo` declares
`query_scoring`, `token_logprobs` with `max_top_logprobs` of at least 20, and the info facts
`replay`: the prompt is scored by
Core's query scoring with the engine's own top-k on its `"prefill"` score path (the default
path when it names none), which must run the engine's prefill kernels at every position, and
the continuation by a replay generation with log-probabilities. The replay prefills the prompt
again, so its `prefill_ms` counts both prefills.

### Raw prompts for adapters

Declare **`completion-v1`** when the adapter continues a raw prompt; `capabilities(text,
sampling)` declares it unless `completion=False`, since ResidentChat continues one with any
decoder, and with `images=True` only with `completion=True`: a raw prompt skips
`prepare_prompt` and reaches `generate_prepared` as `PreparedPrompt(ids)` with `inputs` None,
which an images engine's override must then take. The request then carries `"prompt": {"text":
"..."}` or `{"ids": [...]}` in place of `messages`, never tools or `response_format`, and
`"echo": true` when the client asked for the prompt back. Encode text with the model's
tokenizer and its special tokens, generate without the chat template, and count the prompt's
tokens as `prompt_tokens`. `assistant` is all the text the model writes and may be empty or
blank; there is no `reasoning`. With `logprobs` the reply's rows keep the stop token that ended
it, as OpenAI's legacy endpoint lists it. With `echo` the response's `prompt` holds the
prompt's text (the given text exactly, or the ids decoded), and with `logprobs` the `logprobs`
object also holds `prompt`: one row per prompt token, the first `[token, null, []]`, the others
the score path's prompt rows, every token named in `bytes`. A response may then add 6 bytes per
UTF-8 byte of the prompt (6 KiB per token of an ids prompt), and with log-probabilities 1 KiB
per `kv_tokens` position and alternative. ResidentChat does all of it:
`completion_ids(request)` encodes the prompt (ids outside the vocabulary, text the tokenizer
cannot encode and a prompt of no tokens are a RequestRejected), `generate_tokens` gets the ids
without `prepare_prompt` or the reasoning split, and echo's rows come from `score_tokens(ids,
[], n, cancelled, True)`, which needs `chat-score-v1`.

### Engine facts and plain decoding for adapters

An engine keeps its facts for Eval in `facts`, a dictionary (`crit_core.native`'s engines
hold their `ModelInfo` there); the host publishes the known ones
(`crit_core.serving.ENGINE_FACTS`) as the card's `engine_facts`. A speculative engine
that also decodes plainly declares **`chat-plain-v1`** (`capabilities(text, sampling,
plain=True)`) and sets `plain = True`: a request with `"decode": "plain"` then reaches
`generate_tokens` as `plain=True`. `crit_core.native.ResidentEngine` sets `plain` when its
facts declare `speculative` and `plain`.

### Continuing the final message for adapters

Declare **`chat-continue-v1`** when the adapter continues a final assistant message. A request
carrying `"continue_final_message": true` ends with one and has no `tools` or `response_format`;
render it with the model's chat template left open, Core's `continue_final_message` in place of
`add_generation_prompt` (any `template_kwargs` apply as to any request), and return only the tokens
generated after it as the reply, which may be empty. ResidentChat does this, and `capabilities(text, sampling)` declares it
(`continuation=False` leaves it out for an adapter that renders its own prompt). A template that
changes or drops the final message cannot leave it open, and ResidentChat refuses the request with
`RequestRejected`, as it does one with tools or without a final assistant message, so the engine
stays loaded.

### Tokenizing for adapters

Declare **`chat-tokenize-v1`** when the engine implements `tokenize(text, add_special_tokens)` (the
text's ids, with the tokenizer's own special tokens when asked), `render(request)` (`(text, ids)` of
the prompt a chat request with the request's `messages`, `tools`, `tool_choice`,
`parallel_tool_calls`, `template_kwargs` and `continue_final_message` prefills, the ids exactly those
generation would prefill) and `detokenize(ids)` (their text, special tokens included); the host
refuses to load the package otherwise. The host calls them on their own thread, beside generation
and outside the request queue (on the parent engine of a concurrent package), so they may read only
what generation does not change, such as the tokenizer and the chat template; closing the engine
waits for running ones. Raise `crit_core.serving.RequestRejected`
for input they refuse (a 400); any other exception answers 500 and leaves the engine loaded.
ResidentChat implements all three from its tokenizer, chat template and `prompt_ids()`, refusing invalid
text, ids and image parts with `RequestRejected`, and `capabilities(text, sampling)` declares the
capability; pass `tokenize=False` when a `prompt_ids()` override is not safe beside generation.

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
request's records (`requests/<request_id>/` in the runtime directory, the id its response's
`X-Request-Id` carried) stay for diagnosis; nothing is written to the run folder.

The guardian and supervisor both retry a transient permission denial when reading
the atomically replaced ownership record, for at most 250 ms. They always read
fresh data; missing/malformed records, changed identities and persistent access
failures still stop the engine. Unexpected resident exits include the exit code
and the retained diagnostic log's location.

The [residency fixtures](../tests/test_chat_residency.py) contain executable example adapters.
