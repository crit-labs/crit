# Changelog

All notable changes to crit are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
with versions following [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.9.0] - 2026-10-25

### Overview

crit 0.9.0 is a serving release: `crit serve` behaves the way OpenAI, Anthropic, Ollama and vLLM
clients expect. It answers the OpenAI Responses API statelessly, continues raw prompts on
`/v1/completions` and Ollama's raw `/api/generate`, constrains replies with vLLM's
`structured_outputs`, continues a final assistant message, and exposes the engine's tokenizer and
chat template (`/tokenize`, `/detokenize`, `/apply-template`, `/v1/messages/count_tokens`). A server
shared on a LAN takes several API keys and serves HTTPS, `/health` reports readiness, and the engine
can be unloaded through the API so a shared GPU is freed until the next request. Requests are queued
by priority with a bounded queue that answers 429, 529 or 503 with Retry-After, quiet streams get
keep-alive pings, every request carries an id that `POST /v1/crit/abort` cancels, and `/metrics` gains
latency and token histograms. Thinking controls, stop strings, logprobs, model aliases and many
dialect details now match each API, and fields crit does not apply are a clear 400 unless
`--lenient` is given. The Python distribution and the Core SDK share version 0.9.0, and Core's
Python serving and tool-call layers carry the new request fields; Inference is unchanged.

### Added

- `crit serve --lenient` ignores the request parameters crit does not apply instead of answering 400:
  unknown keys in every dialect (and unknown `stream_options` keys and Ollama options),
  `reasoning_effort` and `prediction`, sampling fields the package does not declare and extra samplers
  that are not off. Each is listed in `X-Crit-Ignored-Parameters` and logged once, so a name that is not
  1 to 64 letters, digits, `_`, `.` or `-`, which the header and the log would carry as sent, is still a
  400, in Anthropic's `output_config` too; values crit knows are still range-checked, and features that
  change the reply (`functions`, `audio`, tools or `logprobs` without the capability) stay refused.
  Strict stays the default, and the servers `crit run ... eval` and `crit run measure` start never use
  it (docs/SERVING.md).
- crit serve takes per-request thinking controls in every dialect for packages declaring the new
  `chat-template-kwargs-v1`: OpenAI `chat_template_kwargs` and `reasoning_effort`, Anthropic `thinking`
  and `output_config.effort`, and Ollama `think` set the chat-template variables the package declares
  (`enable_thinking`, `thinking`, `reasoning_effort`, each with its own capability) for that request,
  over the configured ones, and `/api/show` lists `thinking`. Anthropic's `between_tools` turns thinking
  off, and `xhigh` and `max` efforts, like OpenAI's `xhigh`, take `high`. ResidentChat declares the
  variables its model's templates read and renders a request's; a variable or value the package does not
  take, or turning off a template that always thinks, is a 400. Other packages still refuse the OpenAI
  fields and ignore and list Anthropic's and Ollama's controls, whatever they hold. `--lenient` ignores
  and lists the OpenAI fields of those packages whatever they hold, and on a declaring package a variable
  crit does not set or the package does not declare and a control its template cannot take; values are
  still checked.
- crit serve fits an output cap at or above the context (OpenAI `max_tokens` or
  `max_completion_tokens`, Anthropic `max_tokens`, Ollama `num_predict`) for packages declaring the new
  `chat-fit-output-v1`, which ResidentChat declares: the engine stops where its prompt leaves no room
  and finishes with `length`, and a reply that is not streamed names the cap it applied in
  `X-Crit-Max-Tokens-Applied`. Agent clients that send a model's largest budget got a 400; only a
  prompt that fills the context by itself still does. Ollama's `num_predict` -1 and -2 (until the
  context is full) are fitted the same way; they silently meant 1024, which other packages keep as
  documented. Caps below the context, as Eval sends, reach the engine unchanged.
- `crit serve --max-queue N` (default 64) and `--queue-timeout SECONDS` (default none) bound the queue of
  requests waiting for a busy engine: a request that finds N already waiting, or that waits past the
  timeout, is refused before the engine sees it with a 429 `rate_limit_exceeded` (Anthropic 529
  `overloaded_error`, Ollama 503) and `Retry-After`, and the engine stays loaded. The queue was unbounded
  and outside `--timeout`, so the background calls of chat UIs and agents piled up with no answer. A
  stream sends its headers once the engine takes the request, so a refused stream gets the same status,
  and `GET /metrics` adds `crit_serve_requests_waiting`.
- crit serve takes vLLM's `priority` on `/v1/chat/completions`, an integer that defaults to 0. Requests
  waiting for the engine went strictly in arrival order; now lower values go first, then arrival order, so
  a background job that sends a higher value no longer holds up an interactive chat queued behind it.
  Priority only reorders requests for the same model: a request never moves ahead of a queued model
  switch, nor a switch ahead of requests queued before it, and a running request is never preempted.
  Anthropic and Ollama requests, whose APIs have no such field, wait at 0.
- crit serve sends keep-alive pings on a quiet stream, after `--sse-ping-interval` seconds without output
  (default 15, 0 for none): the SSE comment `: ping` on OpenAI streams and Anthropic's `event: ping` on
  Messages streams, always between whole events. Nothing was written between the role chunk or
  `message_start` and the first token, so while a model loaded or a long prompt prefilled, proxy idle
  timeouts and clients such as Claude Code closed the stream and the disconnect cancelled the turn.
  Pings start when the stream opens, once the engine takes the request, so a request the queue refuses
  still gets its own status. Ollama streams get no pings, nor do the servers `crit run ... eval` and
  `measure` start, and Eval's request bound counts a ping as no answer, so a stalled engine still stops
  it as hung.
- crit serve names every request: each response carries `X-Request-Id`, crit's own id of the request, which
  error bodies and stream error events repeat as `request_id` and which names a failed request's records
  (`requests/<request_id>/` in the runtime directory). A client's own `X-Request-Id` is logged next to it and
  never names records. `POST /v1/crit/abort` with `{"request_id": ID}` (crit's id or the client's) or
  `{"all": true}` cancels requests in flight as a closed connection does: a running turn ends with a 499, a
  queued request leaves the queue without running, and a cooperative adapter keeps its weights loaded.
- crit serve's `GET /metrics` has fixed-bucket histograms for percentiles: queue time, the engine's time
  to first token, time per output token, end-to-end latency and prompt and completion tokens per request.
  `crit_serve_requests_running` and `crit_serve_requests_waiting`, which counts requests waiting for an
  engine load as well as the queue, tell running chat and score requests from queued ones, which
  `crit_serve_requests_in_flight` still counts together, and `crit_serve_engine_loads_total` and
  `crit_serve_model_switches_total` count engine loads and model switches (docs/SERVING.md).
- crit serve model catalogs take `aliases` per model, exact client names (`gpt-4o`, `llama3:8b`) or
  patterns with `*` (`claude-haiku-*`), so Claude Code's main and small models and clients with fixed
  names reach a configured model; the pattern with the most characters besides `*` wins. Names and
  aliases must all be distinct. An opt-in catalog `fallback` names the model that serves any other
  name; without one an unknown name stays a 404. Model cards carry `display_name` (a catalog entry's
  optional `display_name`, else the candidate's label) and `max_model_len` (the context, vLLM's name
  for it, which LiteLLM reads); `/v1/models` and `/api/tags` list the configured names only.
- crit serve's Ollama `/api/chat` and `/api/generate` take `logprobs` and `top_logprobs` (0 to 20) for
  packages declaring `chat-logprobs-v1` and return the reply's token log-probabilities as Ollama's
  `logprobs` on the last object, streamed or not; both were refused as unknown parameters. The last
  object also carries `prompt_eval_cached_count`, the prompt tokens the engine did not compute, and
  `crit_telemetry` passes on the engine's `prefill_tokens`, `prefill_ms`, `decode_tokens` and `decode_ms`.
- `crit serve` takes vLLM's `include_stop_str_in_output`: `true` ends the reply's text with the stop
  string that ended it. A request's `"crit": {"timeout": SECONDS}` gives its turn less time than
  `--timeout` (a larger value keeps `--timeout`), with the same 504 when it runs out and the engine
  kept loaded when the turn settles. Each `logprobs` entry and alternative, OpenAI's and Ollama's,
  carries its token's id as `crit_token_id` (docs/SERVING.md).
- ResidentChat forces tool calls for engines that decode to a grammar: an engine that sets
  `grammars = True` and declares `capabilities(..., grammars=True)` gets `chat-tool-choice-v1` for a
  call syntax crit registers whose markers the tokenizer spells as text, and `tool_choice` `required`
  or a named function (Anthropic `any` or `tool`) then reaches `generate_tokens` as `grammar`, a
  `crit-byte-grammar-v1` artifact from `crit_core.text.tools.call_grammar`: a reply of calls only, in
  the template's syntax, to an offered or the named function, arguments shaped by the function's
  schema, with room to reason first. A reply `max_tokens` cuts before a forced call has begun
  finishes as `length` with no call, where crit serve answered 500 and retired the host.
- crit serve continues a conversation's final assistant message for packages declaring the new
  `chat-continue-v1`, which ResidentChat declares: an OpenAI request with vLLM's
  `continue_final_message: true` and `add_generation_prompt: false`, Anthropic prefill (a final
  assistant message) and an Ollama chat that ends with an assistant message. The chat template
  leaves the message open (Core's `continue_final_message`) and the reply holds only the
  continuation, which may be empty. A final assistant message without the flags is still a 400, and
  so is a continuation with `tools` or `response_format`.
- crit serve takes vLLM's `structured_outputs` on `/v1/chat/completions`: one of `regex`, `choice`,
  `grammar` (Core's line EBNF) or `json` (a JSON schema), or vLLM's earlier `guided_regex`,
  `guided_choice`, `guided_grammar` and `guided_json`, compiled with Core's constraint compiler before
  generation; a source it cannot compile is a 400 with the compiler's reason. Regex, choice and grammar
  reach the engine as a compiled grammar for packages that declare the new `chat-constraints-grammar-v1`
  (`json` as `json_schema`, with `chat-constraints-v1`); only JSON schemas were exposed, though Core
  compiles regex and EBNF too. A second constraint or a `response_format` with one is a 400; like
  `response_format`, one applies with `tools` only when `tool_choice` is `none`, and never to a
  continued final message.
- `crit serve` continues raw prompts for packages declaring `completion-v1` (which ResidentChat
  declares and implements for any decoder, an images engine only with `completion=True`):
  `POST /v1/completions` takes one prompt as text or token ids, encodes text with the model's
  tokenizer and generates without the chat template, with `max_tokens` (fitted as a chat's is),
  sampling, `stop`, `include_stop_str_in_output`, `priority`, `crit`, `stream` (`text_completion`
  chunks, with keep-alive pings), `logprobs` as an integer in the legacy shape (`tokens`,
  `token_logprobs`, `top_logprobs`, `text_offset`, the stop token that ended the reply included) and
  `echo`, whose prompt rows come from the engine's score path (`chat-score-v1`), as lm-eval's
  `local-completions` asks for them. Several prompts and `suffix` are a 400, and `--lenient` ignores
  unknown keys as in chat. Ollama's `/api/generate` with `raw: true` takes the same path, its
  `logprobs` and `top_logprobs` giving the alternatives and `think` ignored and listed. Without the
  capability both answer as before (docs/SERVING.md).
- `crit serve` answers the OpenAI Responses API statelessly (`POST /v1/responses`), which Codex CLI, the
  OpenAI Agents SDK and the Responses modes of LangChain and LibreChat speak: `instructions` and `input`
  (messages, function calls, their outputs and reasoning items) become the conversation, function tools,
  `tool_choice`, `text.format`, `max_output_tokens` and `reasoning.effort` map onto the chat request and
  its validation, and the reply is a Response with reasoning, message and function_call items and usage, or the Responses
  event stream numbered by `sequence_number` and ending in `response.completed`, `response.incomplete` or
  `response.failed`. crit serve keeps no responses, so `previous_response_id`, `conversation`,
  `background` and OpenAI's built-in tools are a 400 and `GET`, `DELETE` and cancel by id a 404
  (docs/SERVING.md).
- crit serve's `/health` reports `ready`, whether an engine is loaded with no model switch, reload or
  retire under way, and `/health?ready=1` answers 503 until it is; `/health` alone said ok while a
  catalog switched models and after a retire left no engine loaded. Model cards (`/v1/models`) and Ollama
  `/api/show` carry `limits`: the context, the largest output cap (a larger `max_tokens` fitted for
  packages declaring `chat-fit-output-v1`), body bytes, stop strings, tools, `top_logprobs` and reply
  characters a request is held to, which clients learned only from a 400.
- `crit serve` takes several API keys, for a server shared on a LAN: `--api-key` may be repeated,
  `CRIT_API_KEY` may hold a comma-separated list, and `--api-key-file PATH` adds one key per line (blank
  lines and `#` comments skipped); a request's key is compared in constant time with each.
  `--ssl-certfile` and `--ssl-keyfile` serve HTTPS, each TLS handshake on its connection's own thread so
  a stalled client holds up no other. Launchers keep `--api-key-file` and the certificate paths, never a
  key.
- `crit serve` unloads and loads its engine through the API, so a GPU shared with games or other tools is freed
  without stopping the server: `POST /v1/crit/unload` waits for the running requests, holds new ones and has the
  resident host close the engine and free the model's weights, and the next request loads it again as a model
  switch does; `POST /v1/crit/load` loads a model ahead of time, waiting in the request queue as a request does,
  and `POST /v1/crit/abort` cancels it. Ollama clients do the same with a request without a prompt or messages
  (`done_reason` `load`) and `keep_alive` 0 (`unload`, or after the reply), `--idle-unload SECONDS` unloads an
  idle engine, and `/health` and the `/v1/models` cards report `loaded`. The server keeps its runtime ownership
  and the machine's engine lock while the engine is unloaded.
- crit serve exposes the engine's tokenizer and chat template for packages declaring the new
  `chat-tokenize-v1`, so clients can check a prompt against the context before sending it: `POST /tokenize`
  (a `prompt`, or `messages` with `tools`, template variables and `continue_final_message` as a chat request
  takes them) returns `tokens`, `count` and `max_model_len` as vLLM does, `/detokenize` the text of `tokens`,
  `/apply-template` the chat template's text as llama-server does, and Anthropic's `/v1/messages/count_tokens`,
  which Claude Code calls, `input_tokens`, with `thinking` and a prefill rendered as `/v1/messages` renders
  them; the last was a 404. They run on the loaded engine beside generation, never queued behind it; one for a
  candidate that is not loaded queues and loads it as a reply would, unless its client disconnects first or an
  abort cancels it, and a full queue refuses it as a reply. A refusal or failure leaves the engine loaded.
  ResidentChat implements the adapter's `tokenize`, `render` and `detokenize`, and `capabilities()` declares the
  capability; packages without it keep a 404 `not_supported`.

### Fixed

- crit serve accepts up to 16 stop strings (`stop`, Anthropic `stop_sequences`, Ollama `options.stop`),
  Core's StopMatcher limit; it refused more than 4, OpenAI's limit, in every dialect, which broke
  template and Modelfile stop lists.
- crit serve answers a prompt that, with its output cap, does not fit the context with a 400 in each
  dialect's words (OpenAI `context_length_exceeded`, Anthropic "prompt is too long", Ollama
  `{"error"}`; an error event in a stream) and keeps the engine loaded; it was a 500 that retired the
  host and reloaded the weights on the next request. Adapters refuse a request this way by raising
  `crit_core.serving.RequestRejected` (ResidentChat raises its `ContextExceeded`) before generating.
- crit serve answers `tool_choice` `required` or a named function (Anthropic `any` or `tool`) with a 400
  `unsupported_parameter` unless the package declares the new `chat-tool-choice-v1`, which adapters
  that force a call declare; ResidentChat packages, which parse calls from free generation, accepted
  the request, failed it with a 500 and retired the host. ResidentChat now refuses such a choice with
  `RequestRejected` too.
- crit serve's Ollama `/api/version` reports an Ollama release, `0.12.0`, with crit's own version as
  `crit_version`; it reported crit's 0.6.0, which clients such as VS Code Copilot and Continue compare
  with a minimum Ollama version and then disabled features or refused the server.
- crit serve's Ollama `/api/ps` lists the loaded model under `--model-name` and in a model catalog, by the
  run, candidate and package the host loaded, with `context_length` and a far-future `expires_at`; it
  matched ids by suffix, so it listed nothing there and could list the wrong candidate, and `expires_at` was
  empty. `/api/tags` digests are the package's sha256, not empty, and `/v1/models` `engine_facts` reach
  catalog models, which had them null.
- crit serve answers `HEAD` on its `GET` paths with the headers alone and other methods (`DELETE`,
  `PUT`, `PATCH`) with a 405 (or a 404 on an unknown path) in the path's dialect; they were http.server's
  501 HTML page. Request bodies may be `Transfer-Encoding: chunked`, up to the same 8 MiB, where
  `Content-Length` was required, and a missing, empty or oversized body is an error in the path's dialect,
  not OpenAI's shape on the Anthropic and Ollama paths.
- crit serve takes `n: null` as the default single choice, as it takes a null sampling field; it was a 400.
  An unknown `stream_options` key is a 400 `unsupported_parameter`, as an unknown top-level key is; it was
  dropped silently.
- crit serve's Anthropic errors carry the type Anthropic gives their status, in a response and as a
  stream's `error` event: a 503 (the engine is stopping or its cleanup is unresolved) is
  `overloaded_error`, which Anthropic clients retry, and a 504 `timeout_error`; both were `api_error`. A
  stream's error event in every dialect is the error the request gets as a response, its status's type
  and code included.
- crit serve's `/metrics` counts every chat request and every `/v1/crit/score` request (as dialect `crit`)
  by outcome, with a new outcome `rejected` for a 4xx: a request that fails validation or a prompt over the
  context. Requests that failed validation and all scores were not counted, and a context overflow counted
  as an `error`.
- docs/SERVING.md lists every OpenAI field refused with a 400 (`modalities`, `reasoning_effort` and
  `prediction` were missing), says where the Anthropic Messages translation differs from Anthropic's
  service, where it said `tool_choice` and `stop_sequences` mapped one to one, and that waiting requests
  queue in arrival order without a limit. README.md and AGENTS.md name the Anthropic and Ollama APIs and
  `/metrics`.
- crit serve keeps an assistant message's `reasoning`, vLLM's and OpenRouter's name for `reasoning_content`, in a
  tool conversation; it was dropped, so clients that replay it lost the model's reasoning across tool turns. A null
  `reasoning`, as those servers return when there is none, is ignored, and a message giving both with different
  text is a 400. `--reasoning-alias` returns reasoning under both names, in replies and stream deltas; it is off by
  default, since a client that joins every field of a delta, as crit Eval does, would read it twice.
- crit serve accepts a `system` (or OpenAI `developer`) message after the first turn, on
  `/v1/chat/completions` and Ollama's `/api/chat`, as user text marked `System: `, joined to the user message
  after it or before it, or as a user turn of its own after an assistant reply or tool result; it answered 400,
  which broke role-play UIs, RAG injectors and agent handoffs that send one. Adapters still receive at most one
  system message, before the conversation; Anthropic's messages have no system role and are unchanged.
- crit serve accepts OpenAI's client metadata `prompt_cache_key`, `prompt_cache_retention` and
  `safety_identifier` as it accepts `user`: ignored and listed in `X-Crit-Ignored-Parameters`; they were a
  400 for the OpenAI SDKs and agent CLIs that send them. `typical_p`, `tfs_z`, `mirostat`,
  `dry_multiplier` and `xtc_probability`, llama.cpp's samplers that crit's sampling law does not have,
  are no-ops at the value that turns them off; any other value is still a 400.
- crit serve's Anthropic Messages API builds the prompt the OpenAI path builds for the same conversation:
  an assistant turn's `thinking` blocks become its `reasoning_content` (they were dropped, so interleaved
  thinking agents lost their reasoning across tool turns), `tool_result` content may hold image blocks,
  which go on the tool message, an `is_error` result's text starts with "Error: ", and a tool's `strict`
  reaches the `chat-tools-strict-v1` gate, where it was silently left out. The event stream keeps one
  content block open at a time: a tool_use block closes when the next call starts, not all at the end.
- crit serve takes stop strings with tools (`stop`, Anthropic `stop_sequences`, Ollama `options.stop`) on
  packages declaring `chat-tool-stream-v1`, and a `response_format` (Anthropic `output_config.format`) with
  tools when `tool_choice` is `none`; it refused both with a 400 in every dialect, though clients such as
  Continue, LibreChat and LiteLLM send stops with tools. Stops are matched in the reply's text, never in a
  call: one ends the reply with the calls the model wrote before it and never cuts a call, since a
  `chat-tool-stream-v1` adapter now emits text and calls in the model's order and each call whole before
  the text after it (the host and crit serve fail a turn with text inside an unfinished call). A format
  with tools that may be called stays a 400.
- crit serve answers `/v1/embeddings`, `/v1/rerank`, `/v2/rerank`, `/rerank`, `/v1/score`, `/v1/audio/*`,
  `/v1/images/*`, `/v1/moderations`, `/v1/files` and `/v1/batches` with a 404 `not_supported` that names the
  endpoint, as it answers `/v1/completions` without `completion-v1`; they were a generic `Unknown endpoint`.
  For embeddings and reranking, and Ollama's `/api/embed`, the message tells RAG clients pointed at crit's
  base URL to run a separate embedding server.

## [0.8.0] - 2026-10-25

### Overview

crit 0.8.0 is a run release: the engine agent's decisions become records crit checks. Before any
code, a candidate starts from a recorded plan that froze the parent's newest Eval: the needed flags
and their evidence, the current strategy's ceiling and the best alternative's, the levers with each
KPI's projected change and a forecast per KPI. crit computes the priorities (decode and prefill are
needed while a lever can still raise them beyond their noise, the one further from its limit first,
time to first token never alone), refuses an incomplete or inconsistent plan, and after Eval sets the
plan's forecast next to the measured result and gives every lever its fate. An early stop is a stop
plan, and the run stops when neither decode nor prefill is needed. Runs created from this version
freeze plan rules 2: crit computes the decode limits Eval leaves out with Eval's own cost model, and
alternatives name their drafter and stay within its measured acceptance. Core keeps a graph capture
across programmatic launches, and four Inference fixes land with five CUDA fixtures that now compile.
The Python distribution and the Core SDK share version 0.8.0.

### Added

- Candidate plans as records: `crit run ... plan --record plan.json`, after the parent's Eval and
  before any code, stores `plans/PLAN_ID/plan.json` with the needed flags and their evidence, the
  current strategy's ceiling and the best alternative's, the levers with each KPI's projected change
  and joint groups, a forecast per KPI and the gates. crit freezes the parent's newest Eval (scores,
  fractions of the limits, noise, gaps), computes the priorities and refuses an incomplete or
  inconsistent plan (a stop decision with a KPI needed, an unanswered gap above the noise on a KPI
  marked not needed, an ungrouped lever below the noise, a p1 KPI without a lever, a TTFT-only plan, a
  forecast without an improvement beyond the noise). A plan proved wrong before its candidate exists
  is superseded by a new one.
- After `crit run ... eval`, a candidate built from a plan gets the plan's outcome,
  `plans/PLAN_ID/outcomes/ATTEMPT.json`: the forecast next to each measured KPI and flags for greedy
  decode, held-out decode and follow-up time to first token when they fell beyond their noise.
  `crit run ... eval` also prints the priorities the result's fractions give.
- An early stop is a plan: `stop --reason search_exhausted` with budgeted candidates left requires a
  stop plan (`"decision": "stop"`) that froze the run's newest Eval, and the stop record's headroom
  accounts for `context_decode` and `context_ttft` whenever that Eval measured the long-context curve.
- Estimated limits in plans: when Eval gives a KPI no limit (a plain engine's decode, or a served engine
  without draft-step bytes), a plan comparing decode with prefill states `estimated_limits` for it, a
  limit in Eval's units and how it was derived. crit refuses a missing, physically inconsistent or
  underived estimate and one that would replace Eval's limit, ranks with the estimated fractions and
  marks them in `priorities.estimated`; the plan's outcome sets each estimate next to the measured score
  and, once Eval gives one, its real limit.
- Lever fates: a candidate built from a plan gives each lever its fate in the proposal's `lever_fates`
  (`built`, `pruned` or `deferred`, with a detail unless built); runs created from this version require
  one fate per lever (`"lever_fates": "required"` in the run spec), older runs check fates when given.
  The plan's outcome lists each lever's fate next to its forecast changes, and `forecast_on_p2` flags a
  forecast whose expected gain falls mostly on a p2 KPI instead of p1.
- Requirement levers in plans: a lever may be `"kind": "requirement"` with `serves` naming the rule or
  gate it is for, such as sampled speculation the rules require before Eval scores it. It needs no
  projected change, is exempt from the noise check and never counts as the plan's lever for a KPI; the
  plan's outcome lists requirement levers separately. Levers default to `speed`, so existing plans stay
  valid.
- Computed limits in plans: in runs created from this version (`"plan_limits": "computed"` in the run
  spec), when the parent's Eval states no decode or long-context decode limit, crit computes one from the
  same attempt, the measured tokens per round over the pass floor (bytes per pass, else the run's model
  files, over the memory bandwidth), and freezes it as `inputs.computed_limits` with its formula and input
  paths. Priorities use Eval's limit, then crit's, then the agent's estimate, and `priorities.sources`
  names each fraction's source; an estimate for a limit crit computed is kept with its difference and not
  used. Decode and long-context decode rank by the best strategy's ceiling: the higher of the current
  strategy's limit and the alternative strategy's ceiling (`strategy.alternative.limits`, its tokens per
  round and draft steps over a round floor crit computes from the attempt or the checkpoint's draft
  layers), whichever strategy the plan chooses, so a plain baseline with a drafter available no longer
  looks nearly done on decode. The outcome compares the alternative's ceiling with the measurement when
  the plan builds it.
- Plan rules 2: runs created from this version freeze `"plan_rules": 2` in the run spec (with
  `plan_limits`), and their plans are checked with the second version of the plan rules; older runs
  keep the first. The alternative strategy's tokens per round is at most its draft steps + 1, so a
  plan can no longer claim a ceiling no round of its drafter reaches.
- Plan rules 2, noise: a noise Eval leaves unknown takes the largest noise the Eval knows (else 5%)
  instead of counting as zero, a family's noise is the largest of its members' instead of the noise of
  the member that sets its fraction, the near tie combines both families' noise in quadrature, and
  checks 4, 6 and 9 weigh each gap, lever change and forecast against its own KPI's noise. A one-percent
  move no longer flips p1 between decode and prefill, and a long-context member that drops below its
  main KPI no longer drops its family from p1.
- Plan rules 2, one cost model, Eval's: crit's computed limit and the alternative strategy's ceiling both
  charge the draft steps at Eval's draft floor (Eval's `draft_floor_ms`, else the engine's draft bytes per
  step, else the checkpoint's draft layers, recorded), as Eval's round floor does, labelled `"bound"` in
  the plan's inputs. Where the draft width is unknown the computed limit stays the pass-floor upper bound,
  labelled as such. Before, the alternative charged its draft steps at the checkpoint's draft-layer bytes
  (several times an engine's own) and the computed limit charged none.
- Plan rules 2, drafters: each alternative limit names the drafter it assumes, the one the parent's Eval
  measured or a new one. Its draft steps stay within the widest draft Eval projects (8, or the parent's
  draft width) and its tokens per round within the measured drafter's acceptance bound at those steps,
  per case `(1 - a^(k+1)) / (1 - a)` at the acceptance rate Eval's own acceptance bound uses (for
  long-context decode, the rate the long-context points measured). A new drafter is held to the same bound
  unless the plan cites another candidate's Eval attempt in the run that measured it, which bounds it by
  its own acceptance; no alternative gives a bare limit, and the plan records every claim's implied
  acceptance rate. A plan can no longer move p1 by claiming acceptance its drafter never showed.
- Plan rules 2: the run's first plan is held to check 3 too, so it builds only with decode or prefill
  needed and stops only with neither.
- Plan rules 2, near the limit (0.85): when both families are needed and both are at 0.85 of their
  limits or above, a near tie puts only the family with the lower fraction in p1 (decode on an exact
  tie); and a family marked not needed below 0.85 needs a `gaps` entry for its limit (`"gap": "limit"`,
  `"answer": "cause"`) with the physical cause that keeps the rest of it out of reach.

### Changed

- The engine agent orders Eval's goals by a priority rule instead of pursuing all at once (AGENTS.md,
  the agent request): decode and prefill are each needed while some lever can still raise it by more
  than its noise, the one further from its limit is p1 and the other p2 (both p1 within the noise of
  each other; long-context decode counts toward decode, long-context time to first token toward
  prefill), and time to first token is p2 or p3, never p1 and never a candidate's only reason.
  Priorities decide conflicts and order, never what is left out. Quality and the required context
  stay gates. When neither decode nor prefill is needed the run stops, even with candidates left.
- Runs created from this version freeze `"plans": "required"` in the run spec: every proposal names a
  recorded plan's `plan_id`, which `candidate` checks (it exists, is the newest of its chain, builds,
  is unlinked and starts from the proposal's parent). Older runs keep `plan_id` optional.

### Fixed

- Core's `launch(..., programmatic = true)` asked `cudaStreamGetDevice` on every launch, which invalidates a
  stream capture, so a programmatic launch could never be captured into a CUDA graph. A capturing stream now
  answers for the current device.
- device-speculative-round's `RoundSteps::accept` had no default member initializer, so the documented
  three-step brace initialization (draft, verify, commit_state) failed under `-Wextra -Werror`; it defaults
  to the greedy accept kernel again.
- Five CUDA fixtures had never compiled (`latent_attention_tests`, `draft_tree_tests`, `sampled_round_tests`,
  `token_law_tests`, `hidden_drafter_tests`): most-vexing-parse declarations, a raw newline inside a string
  literal, typed `cudaMalloc` without `<cuda_runtime.h>`. They build with GCC 13 and pass on an RTX 5090.
- token-sampling's `enqueue_select` answered a token id for a row with no selectable logit on its two-stage
  path (2147483647 when greedy, or an out-of-bounds read that gave token 0 when sampling); it returns -1 there,
  as the full law does, and still advances the device position.

## [0.7.0] - 2026-10-24

### Overview

crit 0.7.0 is an Inference release. Sampled requests speculate on the device and keep the model's
distribution: token-sampling runs Core's whole sampling law on the GPU and verifies drafts by
rejection sampling, greedy drafts stand exactly when they equal the target's own draw, and
EAGLE-2-style draft trees are built, verified in one target pass and walked on the device.
Hidden-state drafters (EAGLE-3, DFlash, DSpark) are read through Core, and `crit models drafter`
distills a draft head from the target. New entries cover hybrid models' recurrent layers, 4-bit
GPTQ/AWQ/compressed-tensors/MXFP4 checkpoints with BF16 activations and absorbed multi-head latent
attention; the attention kernels take sliding windows, sinks, soft-caps and tree masks, and the
fused MoE router takes every common routing form. `crit serve` adds the Anthropic Messages and
Ollama APIs and Prometheus metrics, and `crit run` gains `cost`, a checkpoint fingerprint in recon
and `create --from-engine`. The Python distribution and the Core SDK share version 0.7.0; the
product surface counts 37 Inference entries.

### Added

- Speculative sampling on the device: token-sampling runs Core's whole sampling law on the GPU
  (bias, min_tokens suppression, repetition, frequency and presence penalties over committed plus
  staged outputs, temperature, top-k, top-p, min-p and Core's draw) over the full vocabulary with
  sort-free prefix searches, and `enqueue_verify` turns a round's target logits into
  verify predictions that device-speculative-round's unchanged accept kernel consumes, so sampled
  requests draft inside the decode graph and keep the model's distribution: greedy drafts stand
  exactly when they equal the target's own draw, so the output equals plain sampled decoding token
  for token for the same seed, and drafts sampled from the drafter's law are verified by rejection
  sampling (Core's accept, residual and sample draws). `enqueue_sample` draws drafts with
  their dense law and `enqueue_commit` keeps counts and positions on the device.
- Attention variants in the CUDA attention entries: flash-prefill-attention, fp8-kv-attention and
  paged-attention take a sliding window (keys before the oldest query's window are never read, so
  Core's `window_table` may hold `no_page` there and the contiguous caches can be a ring of
  `window + rows - 1` rows), per-head sink logits, a logit soft-cap, and in decode/verify a tree
  mask per column for speculative token trees. Windowed decode walks only the pieces inside the
  window and keeps column invariance. Gemma-, gpt-oss-, Mistral- and Cohere-style local layers no
  longer need their own kernels.
- Inference entry `draft-trees`: EAGLE-2-style dynamic draft trees on the device (top-k expansion
  with log-softmax path scores, beam and kept-node selection, tree build with ancestor masks and
  depths), a tree walk that accepts the path the target agrees with under device-speculative-round's
  control contract (greedy argmax or the target's own draws, so sampled trees keep the model's
  law), and in-order compaction of the accepted KV rows. `RoundSteps::accept` lets the device round
  run the tree walk instead of the greedy kernel.
- Hidden-state drafters: Core's `crit_core.model.drafters.read_drafter` reads published EAGLE-3,
  DFlash and DSpark drafters (speculators and native formats) into one description with taps as
  residual-stream indices (native DFlash/DSpark layer ids are off by one), and the Inference entry
  `hidden-state-drafters` supplies the device glue: EAGLE-3 catch-up from the round's settled count,
  block-drafter input, `d2t` mapping, draft-law scatter and confidence truncation.
  `crit run ... recon --drafter PATH` (repeatable) records each companion drafter through that reader,
  `crit models drafter` heads included, checked against the target's depth and width.
  flash-prefill-attention gains `causal = false` for a block drafter's non-causal block.
- Inference entry `recurrent-layer-kernels`: CUDA kernels for hybrid models' recurrent layers, a
  causal depthwise convolution with its carried window, gated delta (dk 64 or 128) and Mamba-2
  (d_state 64 or 128), each as a chunked two-pass prefill and a column-invariant column scan that
  writes the state after every column for speculative verify.
- Inference entry `w4a16-projections`: published 4-bit checkpoints run as stored with BF16
  activations. GPTQ (act-order permutation, both zero conventions), AWQ (its column interleave),
  compressed-tensors int4 and MXFP4 (gpt-oss) convert exactly to one canonical layout, read by a
  4-bit decode GEMV for up to 8 tokens and a BF16 tensor-core prompt GEMM on sm_80 and later.
- Core: programmatic dependent launch (`<crit/cuda_launch.h>`). `launch(kernel, shape, stream,
  programmatic, args...)` lets a kernel start while the previous one in its stream finishes, on
  devices the query reports (compute capability 9.0 and later), and launches ordinarily elsewhere;
  kernels call `wait_for_prior_grid()` before reading their predecessor's output and may call
  `launch_dependents()` early. Captured graphs keep the programmatic edges.
- Inference entry `latent-attention`: absorbed multi-head latent attention (DeepSeek-V2/V3, Kimi K2)
  over the shared `[latent ; rope key]` cache rows, for decode, verify and prompt chunks, contiguous
  or paged, with optional sinks and an ordered split combine that keeps columns invariant.

- `crit run ... cost --transcript PATH` accounts a run's build cost from the coding agent's transcripts
  (Claude Code sessions, Codex rollouts, or plain usage JSONL): each model response is counted once and
  attributed by time to the candidate being built (previous seal to own seal), priced at Anthropic's list
  prices per million tokens (input, cache reads, 5-minute and 1-hour cache writes, output; `--prices`
  overrides), with running totals from the run's first event, and written to `cost.json` at the run root.
  AGENTS.md asks the agent to record it before the handoff.
- Recon fingerprints the checkpoint (`gathered.fingerprint` in `work/recon.json`): four hashed sections,
  architecture (config.json's model shape), weights (format, quantization and the stored dtype and shape of
  every tensor pattern, draft layers included), tokenizer (files, special and stop tokens) and template.
  Every candidate records the fingerprint's hash in `extensions.recon`, and `crit run ... recon --compare
  OTHER_MODEL` says which sections another checkpoint changes and what that means for the engine: an
  equal fingerprint is a finetune or re-upload the engine loads unchanged.
- `crit run create --from-engine ENGINE_DIR` starts a run from an engine listed in crit-hub (its
  `engines/<owner>/<name>/` folder): the request records the engine, its source pin, the checkpoint
  files it was measured on and its provenance as `origin_engine`, its published crit Eval results are
  kept under `captured/origin/`, the agent request asks for the imported source as the first candidate
  (proposal `"lineage": "origin-engine"`) measured before any change, and `resolve` records in
  `inputs.json` whether the run's model is that exact checkpoint (`origin_checkpoint`).
- `crit serve` speaks two more APIs over the same engine and the same validation: the Anthropic
  Messages API (`POST /v1/messages`, with `system`, content blocks, tools, tool results, stop
  sequences, `output_config.format` and the Messages event stream, so Claude Code and the Anthropic
  SDKs run on a crit engine with a base URL and `x-api-key`), and the Ollama API (`/`, `/api/version`,
  `/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, `/api/generate`, with Ollama's messages, images,
  tools, `format` and `options`, streaming newline-delimited JSON with Ollama's counts and durations),
  so local chat UIs find it. `GET /metrics` serves Prometheus metrics: requests by dialect and outcome,
  tokens, wall time and the last request's rates (docs/SERVING.md).
- `crit models drafter` distills a draft head for speculative decoding from the target itself, as a
  recorded transform (`method: drafter`, kind `mtp-head`): one decoder layer of the target's architecture
  initialized from a target layer, fed the target's final hidden state and the next token's embedding
  through two norms and a fusion layer, predicting through the target's own head, trained to the target's
  next-token distribution on calibration windows unrelated to Eval, with the held-out greedy agreement and
  KL recorded before and after. The output is `config.json` with a `crit_drafter` block and
  `model.safetensors` under `drafter.`; `crit models plan` and `verify` take drafter recipes and records.
  AGENTS.md lets an agent use such a drafter when the user hands it over with its record, with the
  stored target verifying every draft.
- heterogeneous-expert-dispatch's fused MoE router takes one `Routing` description for every common
  router: softmax or sigmoid scoring before or after selection, a correction bias on the selection
  value only, grouped selection, renormalization and scaling (Mixtral, Qwen-MoE, GPT-OSS, Llama 4,
  DeepSeek-V3, GLM-4.5, Kimi K2). `routing_from_config` maps Core's parsed config fields and fails
  where neither the config nor the model's defaults give one. The defaults are the previous kernel
  bit for bit.
- `settings.example.toml` at the checkout root lists every machine setting `crit run` and `crit eval`
  read, with its default and a comment; copy it to `settings.toml`.

### Changed

- rmsnorm-bf16 takes any hidden size and, besides its own rounding order, a Llama-style one (normalize
  in FP32, round to BF16, then multiply by the weight).
- Core's cgroup host-memory headroom counts no shared memory as reclaimable and honours `memory.high`
  as well as `memory.max`; host-memory-admission's Python reference matches.
- stored-tile-gemm's plans refuse a launch past the tokens, N, K or format they were made for, and split
  work by the device's SM count.
- fp8-kv-attention builds for sm_80 and sm_86; there `prefill_fp8` returns `cudaErrorNotSupported`,
  since those GPUs have no E4M3 tensor-core MMA.

### Fixed

- token-sampling's device selection treated `top_k = 0` (no filter) as the 64 best tokens, which
  truncated the distribution at Eval's default sampling; it now draws from the whole vocabulary,
  as Core does, and keeps the two-stage path for `1 <= top_k <= 64`.
- recurrent-state-recovery's deferred recurrent commit applied a recorded prefix again after a flush
  or an abandoned round; it now applies each recorded prefix exactly once.
- recorded-selection accepted a kept choice beyond the candidate list and keys its table cannot write;
  both are now refused.
- device-expert-cache gave prefetch pool slots to predictions of experts already resident; only
  nonresident predictions take a slot now.
- paged-attention left output rows unwritten for a sequence beyond `max_columns`, `max_queries` or its
  block table; such rows are NaN and the launch raises a fault flag.
- Inference's CMake built the `--wrap` fixtures under MSVC, whose linker accepts the unknown option with a
  warning, so `crit_toy_round_example` failed to link; MSVC builds now omit those fixtures.

## [0.6.1] - 2026-10-24

### Overview

crit 0.6.1 is an Eval release. Results record the machine they ran on (OS, CPU, RAM and each GPU's
VRAM, driver and PCIe link) and no longer store absolute paths that name the user. A plain engine gets
a decode ceiling, a drafting engine that reports no draft step gets one from the checkpoint's own draft
layers, a server whose requests take fewer target passes than tokens counts as speculative, and greedy
requests name neutral penalties so an engine's served defaults cannot reach them. The protocol stays
`crit-eval-v7` and prepared data stays valid. The Python distribution and the Core SDK share version
0.6.1; Core and Inference are unchanged.

### Added

- Eval results record the machine they ran on: `host` keeps `platform` and `python` and adds the OS
  (with distribution and whether it is WSL), the CPU model with physical and logical cores, physical
  RAM and per GPU its VRAM, driver and PCIe link (current generation and width, sampled idle, the
  link maximum and the GPU's and host's own maximum generation). Engines that stream weights or
  experts over PCIe or compute on the CPU depend on these. Every probe is best-effort (null when it
  cannot be read); `run` logs a one-line host summary and `compare` shows it as `Host hardware`.
- A plain engine's decode ceiling: an engine that does not speculate decodes one token per model pass,
  so with `bytes_per_pass` and the bandwidth its ceiling is every pass at its floor, the geometric mean
  of 1 / pass floor over the cases (`headroom.projection.basis` is `one token per pass`). An engine that
  drafts but reports no width still keeps no ceiling.

### Changed

- Eval results no longer store absolute paths that name the user: `adapter` is relative to the
  checkout (or `~/...`, or the file name in another user directory) with a new `adapter_sha256`,
  and the same applies to `native_libraries` (also inside `engine`), the probe's `nvcc` and the
  kernel profile's `toolkit`. A native library's read error no longer repeats its path. Readers
  take old absolute values unchanged.
- A drafting engine that reports no `draft_bytes_per_step` gets the checkpoint's own draft (MTP)
  layers, their routed experts' share and the output head their logits go through as its draft step
  (`headroom.draft_bytes_per_step_source` is `model files`), so a served speculative engine gets a
  decode ceiling; a drafter that is a model of its own has none to derive.
- A served engine counts as speculative when crit serve's `engine_facts` say so or when a measured
  request took fewer target passes than tokens after the first (the first token comes from the
  prefill's logits), so a drafting server without engine facts gets unmeasured exactness instead of
  a skipped one, and a plain engine served through crit serve stays plain.

### Fixed

- Greedy requests sent temperature 0 alone, so an engine's served default penalties still applied (a
  model card's presence penalty of 1.5 sent one run's greedy requests through the host sampler). Every
  greedy request now names presence and frequency 0 and repetition 1 where the engine applies them:
  the speed, held-out, history, curve and honesty requests over a server, an adapter's greedy, plain
  and replayed decoding, and the replay behind crit serve's fixed-text scores.

## [0.6.0] - 2026-10-24

### Overview

crit 0.6.0 measures decode as requests get it. Eval protocol `crit-eval-v7` runs the six speed
cases again at the model's own sampling and makes that **sampled decode** the headline decode,
its KPI, headroom, goal and stop record, with the greedy decode beside it; a `sampling` gate
checks that sampled tokens keep the processed distribution, so speculation under sampling must
verify drafts by rejection sampling. The sampling and the required context are the user's:
recon records the model's sampling and only the user overrides it, and `context_tokens` in
`settings.toml` (default 262,144) caps Eval's max context. Served candidates get exactness and a
decode limit from crit serve's new engine facts, plain decoding and draft telemetry; quality scores
come packed, the quality gate judges prompt positions too, the long-context curve asks the retrieval
question inside the retrieval archive so one prefill serves both, and decode scores carry their
standard error. Breaking: prepare data again for v7 (v6 results keep their greedy decode KPI);
`status.kpis.sampled_decode` is gone, the decode KPI is the sampled one and the greedy one is
`status.kpis.greedy_decode`; adapters add `generate_sampled`. The Python distribution and the Core
SDK share version 0.6.0; Core and Inference are unchanged.

### Changed

- Eval protocol `crit-eval-v7` measures decode as requests get it: the six speed cases run again
  at the model's own sampling (frozen by `crit eval prepare` from `generation_config.json`, fixed
  seeds), and the **sampled decode** is the headline decode next to the greedy one. A new
  `sampling` gate checks that sampled tokens follow the processed distribution (a probability
  integral transform over each token's reported log-probabilities), so speculation under sampling
  must keep the distribution; greedy exactness is unchanged. Adapters add `generate_sampled`;
  Core engines and servers sample through it with no adapter change. Breaking: prepare data again
  for v7.
- From `crit-eval-v7` the decode KPI (`status.kpis.decode`), its headroom and ceiling, the decode
  goal, the stop record's decode and the baseline comparison follow the sampled decode; the greedy
  decode is `status.kpis.greedy_decode`. The ceiling keeps the greedy suite's draft acceptance,
  reachable by rejection sampling, and a new gap projects sampled requests as fast as greedy ones.
  `status.kpis.sampled_decode` is gone. v6 results keep their greedy decode KPI.
- The required context starts from the context the user wants: `context_tokens` in `settings.toml`
  (default 262,144; `--requested-context` on `crit eval run` and `context`, `context_tokens` in a run
  request). Eval's max context is the smallest of it, the model's positions and the KV cache that
  fits; results and the run record the requested context, the bound and a note when the model or
  device allow less. Recon's `recon.md` gains a Required context section: the tokens every candidate
  must serve, why, the KV cache they take, that only the user changes it, and a KV cache plan to
  write.
- The sampled suite's sampling is the model's: prepared data and results record `sampling_source`
  (`generation_config`, `protocol_default` or `override`). Only the user overrides it, with
  `crit eval prepare --sampling` and, in a run, `eval_sampling` at create, which `crit run ... eval`
  requires of the data and recon applies to the served default. Recon's `recon.md` gains an
  "Eval sampling" section that says the agent cannot change it and a drafting plan for both decode
  modes, and a candidate refuses a frozen sampling other than the model's.
- The quality gate judges each set's prompt positions too: every set is judged as `SET/prompt` and
  `SET/served` by the drift budget, so the prefill's arithmetic, which every reply's first token
  comes from, reaches the gate in full; `quality.prompt` is gone.
- The long-context curve asks inside the retrieval archive: every curve point holds the retrieval
  facts and ends with the retrieval check's own framing (`long.json` `curve_user`: list the facts,
  then summarize the archive), the point times the engine's own reply and judges retrieval on it, so
  one prefill serves both. Retrieval points the curve does not reach (4,096 always) keep a request of
  their own. The curve no longer extends the longest speed case or replays a pinned continuation, and
  prepare no longer writes its frame. Prepare data again.
- The honesty check replies at 32K tokens once, to the conversations and to explain-tides (protocol
  `quality.honesty` `long_conversations`), instead of three 32K texts that added six 32K prefills to
  every round.
- The sampled suite measures one repetition per protocol seed with no extra one, and judges a
  disturbed device by the spread of `round_ms` over the repetitions that decoded every output token
  and finished by length (a case with fewer than two is skipped), instead of by decode speed, which
  differs with the text each seed draws.
- Sampled requests name every sampling field: the filters the settings leave unset as off (top_p 1,
  top_k 0, min_p 0), the penalties neutral and the seed, so an engine applies none of its own
  defaults. An engine that lists the fields it applies (crit serve's `supported_sampling_parameters`,
  an adapter's `sampling_fields`) gets only those; one that cannot apply a set field has no sampled
  decode instead of failing the round.
- Decode scores carry their relative standard error from the repetitions' spread; status shows it,
  prefill gets a run-to-run spread too, and a baseline change smaller than the noise is reported as
  within the noise. The goals block prints each goal's share of its limit and its noise, and Eval
  leaves the goals' order to the run.
- Eval's quality score calls ask crit serve for packed scores and fall back to JSON rows once a server
  refuses the field.
- `crit run measure` sends a case's captured sampling (temperature, top_p, top_k, min_p, seed,
  penalties) instead of forcing temperature 0, stays greedy when the capture names none, and records
  the sampling sent in each benchmark's extensions.
- `crit run eval` builds the run's context again in the resident host that served the lower-context
  gate, keeping the model's pool and the adapter's process, instead of loading the model in a second
  crit serve (`ServedEngine.reopen`); the server stops once at the end of the attempt.
- The reference binds a batched form of qwen4_exp's sparse-attention indexer (transformers 5.20.2's
  `Qwen4ExpTextQSAIndexer.forward`, by source hash) that pools and scores every visible key block at
  once when every query sees a prefix of the keys, instead of transformers' per-token loop (214 to
  1,307 s per full-attention layer on Qwen3.8 Flash-Next's 32K-token texts). Scores are unchanged, and
  the reference's identity records the forms it used.

### Added

- `crit serve` reports the loaded candidate's `engine_facts` in `/v1/models` (speculative, plain,
  draft width, bytes per pass and draft step, KV bytes per token), decodes plainly on request (crit
  extension `{"decode": "plain"}`, for `chat-plain-v1` packages), and its telemetry carries the drafts
  accepted per round, `verify_ms` and `draft_ms` when the engine reports them.
- `crit serve` returns fixed-text scores packed on request (`"format": "packed"`): each part's rows as
  base64 little-endian arrays instead of nested JSON lists.
- Served candidates get exactness and a decode limit: Eval reads crit serve's engine facts and each
  reply's accepted drafts, and runs exactness on the served engine through plain decoding when the
  package declares `chat-plain-v1` (otherwise exactness is unmeasured with that reason). An engine
  that reports no bytes per pass gets the stored model's.

### Fixed

- The sampling check judges a sampled token outside the 20 reported alternatives by the rest of the
  vocabulary as one last alternative, instead of counting it outside the distribution, which failed a
  correct sampler whenever top_k was unset or above 20. A position is judged when the alternatives
  determine its processed distribution and reported as unresolved otherwise; the gate is unmeasured
  below `min_tokens` judged tokens, and a server's tokens are matched by their bytes.
- With quality among the parts, a frozen prompt a server renders to another token count (another chat
  template) fails the round at its first request instead of after the speed, held-out and quality
  parts ran on the wrong prompts.

## [0.5.0] - 2026-10-23

### Overview

crit 0.5.0 judges quality on what an engine serves. Eval protocol `crit-eval-v6` scores fixed
text once per model on the reference and stores it; each engine scores the same ids through
crit serve's new `chat-score-v1` (its prefill and its decode forced to the text), is judged on
the positions it serves, and must score its own served replies alike (honesty). No reference
runs per candidate, and the reference itself streams layers ahead, gathers rows in parallel
reads and runs long prompts layer-major. Eval also runs in one attempt with no fixed time cap,
measures follow-up turns and prompt caches, and repeats a speed case only when its runs
disagree. Core adds batched paged serving, hybrid and sliding-window prefix reuse and faster
constraints; Inference gains paged attention, a batched reference engine, GGUF projections and
dense-layer offload. Breaking: v6 quality results do not compare with v5 (prepare data again;
`crit eval status` refuses other majors); the native engine ABI is 2, so engines built on
`crit/native_engine.h` must be rebuilt; `crit run eval` needs packages declaring
`chat-score-v1`, whose `score_tokens` takes `prompt_rows`. The Python distribution and the Core
SDK share version 0.5.0.

### Changed

- Eval protocol `crit-eval-v6` measures quality on fixed text through the engine's serving paths
  instead of scoring every candidate's replies on the reference. Each quality text is a
  conversation's generation prompt followed by its written answer (restored in
  `suite/quality.json`, cut to 256 tokens); `crit eval reference --model --data` scores every
  position once per model, suite and reference version and stores each position's log-probability
  and top 16 under the data (about 16 MB). The engine scores the same ids through crit serve's
  `chat-score-v1`, prompt positions from its prefill and answer tokens from its decode forced to the
  answer, and each set is judged on the positions it serves (`SET/served`: the answer's first token
  from the prefill's last position, the rest from decode); prompt positions, whose prefill logits no
  request returns, are reported in `quality.prompt` and not judged. Both come with figures by
  position bucket so thresholds show as steps. An honesty check
  scores the engine's served greedy replies through the same call; they must agree within 1e-3
  nats. `crit run ... eval` scores the data's texts before serving the first candidate and needs
  `chat-score-v1`; no reference runs per candidate. v5 results do not compare with v6, and a
  baseline compares only within its prepared data, suite and major version. Prepare data again
  for v6.
- `crit eval status` judges results of the current protocol major only and refuses others (their
  status is stored in them), instead of rendering pre-v4 results without the request-history and
  retrieval gates, results without an exactness mode as `required`, and v5 quality as unmeasured.
- v5's per-reply quality scoring is gone: `StreamingReference.score` (each sequence run whole) and
  `metrics.reply_quality`. The reference scores through `score_texts` only.
- Eval measures follow-up turns (protocol `followup`, suite `followup.json`): a conversation's
  turns, each resending the history, may be served from a prompt cache; each reports time to
  first token, its cached prompt tokens and the prefill rate of the rest (`followup` KPIs), and
  the turns run again after every other part must answer as they did first (`prompt_cache`
  gate). An engine without a cache reports its fresh prefill and passes. `crit serve` reports
  the prompt tokens an engine did not compute as `usage.prompt_tokens_details.cached_tokens`.
- At the required context, the long-context curve's prompt holds the retrieval facts, and the
  retrieval request right after that point shares its prompt up to the end of the document,
  then asks the retrieval question (suite `long.json` `retrieval.question`), so an engine with a
  prompt cache prefills that text once. Retrieval requests may be served from a prompt cache
  (they are judged on their text; their cached tokens are recorded); every timed request still
  prefills. Prepare data again.
- Eval measures each speed case twice after its warmup and a third time only when the two
  disagree: decode speed or time to first token spread by more than 5%, or the outputs differ
  (protocol `speed.extra_repetition`); before, every case ran three times.
- Eval's quality phase records where its time goes (`quality.timing`: per score request, honesty
  reply and rescore, the request's seconds, the engine's own prefill and decode time, the rest,
  response bytes and parse time), and the honesty rescore asks the score call for the reply's
  rows alone: `chat-score-v1` takes `"prompt_rows": false`, and `score_tokens` takes
  `prompt_rows`, so the engine prefills the prompt as for a reply instead of scoring every
  position the fixed texts already scored. Results are unchanged.
- `crit run eval` runs every part in one `eval` attempt on one loaded engine instead of within the
  1800 s benchmark cap, which a long required context outgrew and made Eval run in parts, each
  loading the engine again. The attempt runs to the segment's deadline; each request to the
  engine may take the protocol's floor plus a slack times what the engine's earlier requests in
  the same Eval say it should (`requests.timeout`), so a hung engine ends the attempt
  (`eval_failed`). crit serve's own bound on each request is then the attempt's deadline.
- Native engine ABI 2 (`crit/native_engine.h`): `crit_engine_score` takes `top_k` and returns the
  engine's own ranked alternatives with the target scores, query rows and argmax. Every export is
  required: `crit_core.native` no longer probes for the request and log-probability exports and
  refuses ABI-1 libraries; rebuild native engines against this crit's headers.
  `NativeEngine.score(ids, first_target, path, queries, top_k)` returns
  `Scores(target, query, argmax, top)`.

- `crit eval reference` drives the text model's own transformers forward instead of rebuilding
  its embedding, masks, positions and final norm: decoder layers still stream onto the GPU one at
  a time, but what the forward hands each layer is recorded first and its own tail runs on the
  last layer's output. Models that do more around their layers (repeated residual streams,
  multi-axis rotary positions, per-layer inputs, a final mixer, a sparse-attention indexer that
  needs a mask) score as transformers scores them. Experts run transformers' eager per-expert loop
  rather than its grouped GEMM. Scores of models the reference already read are unchanged (the
  existing reference and format tests pass with exact equality).
- The reference reads safetensors tensors through a mapping of the file made per read, so the
  pages it reads stay in the page cache rather than in its own resident memory, and no open
  mapping holds the files.

- `crit eval reference` scores long texts in chunks of 8,192 tokens that resume the cache the
  model's own forward builds (transformers' Cache: keys and values, sliding windows,
  linear-attention and Mamba states), kept per decoder layer in FP32 scratch files while the text
  is scored, so no mask of a whole long prompt is built. A model whose cache cannot be resumed is
  scored by whole passes.
- `crit run ... eval` stops the reference at the reservation's deadline, checked between its
  layers and sequences, and settles the attempt as timed out instead of scoring past it.
- The reference gathers embedding rows (such as a hashed n-gram table far larger than memory) by
  parallel reads of exactly their bytes, every part of a split table in one batch, instead of
  faulting pages in through a mapping one at a time: a Flash-Next pass gathers its 25,791 rows in
  seconds rather than minutes. The rows are byte-identical.
- The reference reads each decoder layer's stored tensors ahead, while the previous layer runs: in
  large reads in file order into one page-locked host buffer (at most one layer, up to 8 GiB),
  copied to the GPU in one transfer and dequantized there as before, instead of mapping, copying
  and transferring thousands of tensors one by one (a Flash-Next layer holds 6,166). Loaded layers
  are bit-identical.
- The reference runs the texts' chunks layer-major: each decoder layer, once loaded, runs every
  chunk it can, rather than every layer being loaded again for each round of 8,192-token chunks.
  Only the layers up to those whose cache state the forward reads between chunks (the first layer
  of each kind) run chunk by chunk: Flash-Next's 32K-token texts load 64 layers instead of 240.
  Reads of another layer's cache state, or of the cache's own attributes, that would differ from
  the chunk-major order make the texts run chunk-major; either order gives the same scores.

- The schema/regex compiler is `crit-language-compiler-2`: it builds unbounded repetition left
  recursive, which an Earley recognizer extends in constant work per item, so compiled artifacts
  change while the languages they accept do not; cached artifacts compile again under the new
  version. Eval compiles no constraints, so its results are unaffected.
- Host ISA detection (cpuid, xgetbv) lives in host-expert-arithmetic's `host_isa.h`, shared by
  every host path with AVX2/AVX-512 files, and `crit_inference_isa_sources` builds those files
  under one x86-64 condition.

### Added

- `crit serve` scores fixed text through an engine's serving paths: `POST /v1/crit/score` with
  prompt and continuation ids returns every prompt position's log-probability and top
  alternatives from the engine's prefill and every continuation token's from its decode,
  forced to the continuation (packages declaring `chat-score-v1`; `ResidentChat.score_tokens`).
  `crit_core.native.ResidentEngine` implements it for engines with query scoring, replay and
  token log-probabilities, through `crit_engine_score` with `top_k`, which returns the engine's own
  top-k with Core's scores.
- The reference maps checkpoint names through transformers' conversion mapping for the model
  type: per-expert matrices (in any stored scheme, NVFP4 included) become the fused expert tensors
  the modeling code holds, sharded tables are concatenated, and a text model stored under a
  multimodal prefix is found by matching its layer names; vision and draft (MTP) tensors are never
  read.
- Embedding lookups in the reference gather only the rows they need from the memory-mapped
  checkpoint and decode those (plain, FP8 and NVFP4, also a sharded table with one scale for the
  whole table), so a hashed n-gram table larger than host memory is never built; building the
  model skeleton allocates no parameter memory either.
- `chunked_attention` honours a supplied mask per block of query rows, and the long prompts get
  sdpa's mask rules: no mask where causality suffices, a boolean one where the modeling code needs
  it.

- Inference entry `paged-attention`: CUDA attention for a ragged batch of sequences whose KV
  pages are addressed through Core StateCache block tables. One decode launch covers every
  sequence of a step (and multi-column speculative verify), a varlen causal prefill covers
  ragged prompt chunks over a paged prefix, and pages hold BF16 values or E4M3 codes with
  per-head-vector scales. Launch shapes depend only on host counts, so one captured graph
  serves every length and table, and a deterministic split-key combine keeps a column decoded
  alone equal to the same column in a verify. `DevicePages` implements the cache's
  copy-on-write for every layer; `BatchTables` packs a step's tables, lengths and slots.
  Validated against the attention-state CPU reference over the same pages. The kernels need sm_80
  or later and build for sm_80, sm_86, sm_89 and sm_120; their tests ran on sm_120.
- Inference entry `batched-reference-engine`: a complete batched serving engine for a
  synthetic transformer. A Core `ConcurrentModel` with mixed batches runs every request's rows
  of a step through one GEMM per weight, keeps KV in StateCache pages with prefix reuse
  (reported as `reused_prompt_tokens`), attends with `paged-attention` and is exported as a
  native engine library with concurrent requests for `crit serve`. Its arithmetic is batch
  invariant, so batched requests equal their serial runs token for token. On an RTX 5090, a 0.98 GB
  synthetic model with 128-token prompts and replies went from 535 completion tokens/s for one
  request to 28,958 for 64 concurrent requests.
- Core's StateCache reuses prefixes of hybrid models: each recurrent component may keep a quota of
  snapshots (its state after an indexed page's last row, copied through `copy_slot` when a commit
  fills the page), `attach` restores the latest one with the pages, and snapshots leave with their
  pages. Snapshot slots are a private pool that never takes the slots sequences are guaranteed.
  Sliding-window layer groups (`StateCacheConfig::windows`) keep their own pages and window
  tables: pages older than the next row's window are dropped, full ones stay linked to the prefix
  index, and reuse ends where a window's pages are gone. New `Round::window_kv`, `window_table`,
  `window_stats`, `StateStorage::copy_window_rows` and `StateCacheStats::snapshots`; the prefix
  cache no longer rejects recurrent components.
- Core's constraint matchers compute allowed sets by walking a byte trie of the vocabulary once per
  state (shared piece prefixes advance the language state once) and cache them by exact state in a
  bounded, thread-safe `AllowedCache` shared by the matchers of one language; results equal the full
  scan (`allowed_by_scan`), grammar chart exhaustion included. `TokenMatcher::allowed_bits` fills a
  bitmask for device masks. Grammar recognizers scan single-byte rules (character classes) as one
  terminal. Measured on one thread of an AMD Ryzen 5 9600 with Nemotron 3.5 Super's 131,072-token
  byte vocabulary, per generated position: a JSON-schema grammar inside a string went from 9.4 s to
  0.2 s for a new state and 0.13 ms for a repeated one; JSON mode from 2-6 ms to 0.01-0.06 ms (small
  allowed sets) and 0.13 ms (repeated string states).
- The JSON schema compiler takes `anyOf`, type lists, root `$defs` with local `$ref` (recursive
  schemas through grammar rules), and string `pattern` (ECMA-262 search semantics on the decoded
  string, through the regex frontend). `oneOf` compiles only when its branches are provably disjoint
  (kinds, literal values or a required discriminator property) and is rejected otherwise.
- Inference `token-sampling` adds `enqueue_mask`, which applies a constraint's bitmask
  (`TokenMatcher::allowed_bits`) to device logit rows before selection.
- Inference entry `gguf-projections`: GGUF Q4_0, Q8_0, Q4_K, Q5_K and Q6_K blocks decoded on the
  host bit for bit as gguf-py's `quants.dequantize` does (checked against gguf 0.19.0), and CUDA
  kernels that decode the same bits in registers: a GEMV over the stored blocks for 1 to 8 activation
  rows and a dequantizing copy for prompt GEMMs.
- Inference entry `dense-layer-offload`: layer-granular placement of dense models larger than device
  memory. A planner predicts each placement's decode step from measured device, upload and host
  rates and the engine times the leading candidates; host layers run AVX2/AVX-512 BF16 GEMVs over a
  thread pool with host KV, and streamed layers upload two ahead into device staging slots. Streamed
  layers give the resident logits bit for bit; on one machine (RTX 5090, AMD Ryzen 5 9600, a 7 GB
  synthetic model) predictions were within 0-16% of measured step times.

### Fixed

- The reference loads stored integer buffers (such as hashing multipliers) in their own dtype
  instead of converting them to FP32.

## [0.4.0] - 2026-10-21

### Overview

crit 0.4.0 reads models in the formats they are published in. Eval, `crit run recon`, `crit serve`
and calibration open a Hugging Face folder or a single weight file through one model source:
safetensors, GGUF (a lone `.gguf` included, with its config and tokenizer derived from its own
metadata), PyTorch state dicts and ONNX graphs. The reference and the decode ceiling read the
common quantization schemes as stored (compressed-tensors, FP8 blocks, AWQ, GPTQ, MLX, bitsandbytes
and ONNX MatMulNBits), each decoded to exactly the FP32 weights its own library gives. EXL2 and
EXL3 are refused by name. crit ships no GGUF runtime; gguf-py is needed only to score a GGUF with
`crit eval reference`. Eval data prepared with 0.3.0 for a safetensors model stays valid. The
Python distribution and the Core SDK share version 0.4.0.

### Added

- GGUF models, pointed at as they come: a lone `.gguf`, the first part of a `gguf-split` set, or a
  folder holding one. `crit_core.model` gets a strict, standard-library GGUF reader (`inspect_gguf`,
  `inspect_gguf_model`) and derives the description files from the GGUF's own metadata
  (`crit_core.model.gguf_files`): `config.json` for the `llama`, `qwen2`, `qwen3`, `qwen3moe`,
  `qwen35` and `qwen35moe` architectures, `tokenizer.json` for byte-level BPE vocabularies with
  the `qwen2`, `qwen35`, `llama-bpe` and `gpt-2` pre-tokenizers, `tokenizer_config.json`, the
  chat templates and `generation_config.json`. The base model's files beside the GGUF are used
  instead where present, and `check_gguf_folder` refuses them when their shapes, norm epsilon,
  RoPE base, tokens or merges disagree with the GGUF. The context, ceiling and model identity count
  the GGUF's stored bytes per block type. The reference decodes blocks with gguf-py and undoes
  the reference converter's layout for the same six architectures (`crit_eval.gguf_layout`); a lone
  GGUF from that converter gives back every tensor, the original config.json and
  tokenizer, and the same reference scores. The derived tokenizers match Hugging Face's on
  the reference converter's vocabulary test sets. crit runs no GGUF runtime.
- Model sources. `crit_core.model.open_model(path)` reads any model folder or weight file as one
  `ModelSource` (weight headers, weight files, description files real or derived), and Eval,
  `crit run recon`, `crit serve` and calibration read models only through it. A weight format is
  one `ModelFormat` (safetensors, GGUF, PyTorch and ONNX today).
- PyTorch state dicts: `pytorch_model.bin` (sharded with its index too), `.pt` and `.pth`.
  `crit_core.model.inspect_torch` reads torch.save's zip format with an unpickler that resolves
  only what a state dict uses and refuses any other global, so the pickle never runs; tied
  weights, float8 and storage offsets are kept, transposed views and the pre-1.6 format refused.
  A safetensors checkpoint beside it wins, as in transformers.
- ONNX graphs exported from a Hugging Face checkpoint (Optimum, torch.onnx), with external data.
  `crit_core.model.inspect_onnx` walks the protobuf with the standard library and names every
  weight as the checkpoint it came from, through the consuming node's module scope (a linear
  layer's transposed MatMul operand becomes `<module>.weight`); MatMulNBits weights keep their
  node's attributes. onnxruntime-genai builder graphs and other quantized ops are refused.
- Quantization schemes, decided in one place for the ceiling and the reference
  (`crit_eval.schemes`): compressed-tensors int (pack-quantized W4A16/W8A16 with or without zero
  points, int-quantized W8A8), FP8 per block (compressed-tensors and `weight_scale_inv`), AWQ GEMM,
  GPTQ (2/3/4/8 bits, `gptq` and `gptq_v2`), MLX (affine, mixed, mxfp4, nvfp4, mxfp8),
  bitsandbytes NF4/FP4/LLM.int8 and ONNX MatMulNBits, beside the FP8 and NVFP4 already read. The
  reference decodes each to exactly the FP32 weights the format's own library gives (tested against
  GPTQModel, AutoAWQ, mlx, bitsandbytes, compressed-tensors and onnxruntime on models those tools
  wrote), and the ceiling counts each scheme's matrices at its GEMM rate.
- EXL2 and EXL3 checkpoints are refused by name, by their config or their tensors, rather than
  counted and read without their matrices: exllama decodes them only in CUDA kernels, so there is
  no CPU dequantization to check a reference decoder against.

## [0.3.0] - 2026-10-21

### Overview

crit 0.3.0 measures engines the way a chat client uses them. Eval protocol `crit-eval-v5` sends
every prompt as chat messages to an OpenAI-compatible `/chat/completions`, so `crit eval run --url`
measures any served engine, built with crit or not, and `crit eval reference` scores quality on the
engine's own greedy replies against the checkpoint the data was prepared from. `crit run measure`
and `crit run eval` measure a packaged candidate through `crit serve`. `crit serve` talks to its
resident host over one loopback connection instead of polled files, streams tool calls as they are
written, returns token log-probabilities and reports tokens per round. Core serves concurrent native
engines, preempts and restores requests under memory pressure and schedules dependent request
steps. Nothing measured before carries over: prepare Eval data again for v5, capture run cases kept
as token ids again as chat messages, and expect no quality figures from an Eval adapter. The Python
distribution and the Core SDK share version 0.3.0.

### Added

- `crit eval run --url` measures any engine behind an OpenAI-compatible chat API, no adapter
  needed: the speed cases, held-out prompts, request history, long-context curve and retrieval
  go to `/chat/completions` as the chat messages `prepare` freezes next to their token ids, with
  lengths from the usage counts the server must stream, and quality from the engine's own replies.
  It takes no engine lock, so it measures an engine `crit serve` is serving.
- `crit serve` returns token log-probabilities (`logprobs`, `top_logprobs` up to 20) for packages
  declaring `chat-logprobs-v1`: every generated token with its bytes and ranked alternatives, in
  the finish chunk when streaming. ResidentChat adapters set `logprobs = True` and receive
  `logprobs=n`; native libraries rebuilt against this `crit/native_engine.h` export
  `crit_engine_request_wait_output` and pass `"logprobs": n` through to Core, so
  `crit_core.native.ResidentEngine` serves them when the model declares 20 alternatives.
- `crit serve` reports `tokens_per_round` (output tokens per target pass) in `crit_telemetry`, and
  a stream's usage chunk now carries `crit_telemetry` too, when the engine reports its decode
  rounds: response telemetry takes an optional `decode_rounds`, ResidentChat passes
  `generate_tokens`' timing `rounds`, and Core engine libraries report theirs as Eval counts them.
- `crit run ... eval --candidate ID --data DIR` measures a packaged candidate the way a chat client
  gets it: under one benchmark reservation it serves the candidate through `crit serve` before
  sealing, runs Eval over the chat API with the run's quality terms and context, stops the server,
  scores the quality replies on the run's model and retains the result as
  `candidates/ID/eval/ATTEMPT.json`. Every engine Eval opens is the candidate served at the context
  it asks for, so the lower-context gate is measured too. Packages need `chat-stream-v1` and
  `chat-logprobs-v1` from their first Eval.
- Core accepts an optional prefill work quantum for engines that pad prompt
  computation. Mixed scheduling reserves those padded columns without consuming
  nonexistent input, exceeding the batch budget or rounding decode grants up.
  This closes a concrete integration gap exposed by a retained engine's kernels.

- Native HTTP integration fixtures exercise concurrent streaming and buffered clients,
  both supplied recovery adapters, and client cancellation during prefill. An optional
  CUDA recurrence example verifies device-carried dependent steps while predecessor
  readback is blocked, with ordered publication and cancellation cleanup.
- Core can schedule bounded dependent steps for one request before host publication,
  using engine-owned continuations and per-request eligibility. It reserves unresolved
  output grants, publishes in order, and settles discarded descendants before retirement.
  A portable asynchronous recurrent reference covers real dependent computation, seeded
  sampling, stops, failure ownership and full-state recovery under memory pressure.
- Core's optional memory preemption path accepts typed pre-submission pressure,
  suspends reclaimable settled requests, and restores their original streams and
  identities after other resident work finishes. Recovery has separate telemetry,
  cancellation and failure ownership. A paged-attention reference reuses physical
  pages and replays accepted history without resampling or duplicate output.
- Native serving supports explicitly concurrent engine libraries through private resident
  request wrappers and the configured Core admission capacity. Additive ABI request handles
  allow cancellation during prefill, before any output; request-owned result facts avoid
  shared mutable statistics. Older libraries retain serial serving.
- Core's reasoning and tool parsers accept tokenizer-bound delimiters and retain token identity
  through incremental decoding and reasoning-to-tool routing. Single special-token boundaries
  distinguish control ids from ordinary tokens with the same spelling; multi-token text
  delimiters and explicit `delimiters="text"` formats retain text matching.

### Changed

- `crit run measure --candidate ID [--seconds N]` measures the run's cases through `crit serve`, as a
  chat client gets them, instead of through the package's Eval adapter in a child process. Each case
  keeps its chat request as a captured `{"messages": [...]}` JSON file, and `crit serve`'s usage must
  count the workload's prompt tokens for it. Every candidate decodes its own greedy reply, so no text
  is pinned any more, and a reply that ends early is measured on the tokens it has. Memory is sampled
  on the resident host process, which now names itself to `crit serve`, and tokens per round comes
  from `crit serve`'s telemetry. `--adapter`, `--model` and `--config` are gone; capture cases kept
  as `{"ids": [...]}` again as messages. The resident host, like the worker before it, writes no
  core file the size of the model when it crashes.
- ResidentChat streams a tool call as the model writes it instead of as one delta
  when its block closes: the call's id and name as soon as the name is written, then its arguments
  as they become certain, for both registered call syntaxes and for a package `ToolFormat` that
  gives the new `stream` function. Eval over `crit serve` therefore times a reply that ends in a
  call by when its tokens were written. A block that has begun streaming is that call: one that
  closes without parsing, or that the model ends its reply inside, fails the reply instead of
  staying content, and one `max_tokens` cuts comes back with the arguments written so far and
  `finish_reason: "length"`, as OpenAI's API returns it. A Hermes-style JSON call that repeats a
  key stays content.
- `crit serve` talks to its resident host over one loopback connection the host opens with the
  runtime's ownership token, instead of command, result and stream-snapshot files polled every
  20 ms. A streamed reply's first token arrives about 30 ms sooner, and each delta the engine
  emits reaches the client as its own chunk instead of one coalesced update per 50 ms; the HTTP
  server sends chunks without Nagle's delay. Only failed requests leave records in the runtime
  directory, and `ServedEngine.generate`'s `on_text` receives each piece instead of the
  cumulative text.
- Eval protocol `crit-eval-v5`: the long-context curve's prompts are the longest speed case
  with its document followed by suite filler text, cut to fit, instead of copies of its whole
  prompt, so every prompt Eval sends is chat messages and served engines run the curve and
  retrieval too. Curve scores move: prepare data again for v5.
- Eval protocol `crit-eval-v5` measures quality on the engine's own replies, as a chat API user
  gets them. Each quality prompt goes as chat messages with `logprobs` and 16 `top_logprobs`;
  `crit eval reference --model --data --result` then scores the greedy reply teacher-forced on the
  checkpoint the data was prepared from: KL over both top-16 lists plus the rest, the perplexity
  ratio and top-1 agreement over every reply token. The conversations and the replies after 1K-,
  4K- and 32K-token prompts are each a set judged by the drift budget. The server must render each
  prompt to the token count the model's chat template does and name reply tokens by their bytes.
  Adapter scoring (`score()`, `score_paths`, `prefill_decode`, and the stock
  `crit_core.native.EvalEngine`'s) and `reference.json` are gone, so an adapter's engine gets no
  quality figures; Core's scoring and `crit_engine_score` stay for engines.
- `crit models eval --record --data --result --evaluation` binds a `crit eval run --url` result of
  the engine serving a transform's output to its record and scores its quality replies on the
  source checkpoint, instead of running an adapter; `models_evaluation` records the server's URL
  and model name in place of the adapter and its configuration.
- Eval protocol ids have a major and an optional minor version: `crit-eval-v4` is 4.0 and
  `crit-eval-v4.1` is 4.1. A change that can move a score takes the next major version,
  any other the next minor; scores compare within a major version and `compare` warns across
  majors. Prepared data runs under every minor version of its major when it froze the current
  suite, and a test pins each version to its protocol and suite, so an edit to either needs a
  new version.

### Fixed

- A resident host run in-process no longer writes a `__pycache__` into the sealed package it loads,
  which failed the package's verification on the next request; the test suite run without `-B` hit it.
- Core fails an individually blocked request when all resident requests hold
  nonreclaimable memory and no work can free capacity, allowing its peers to progress
  instead of waiting indefinitely.
- Resident chat parsing now keeps delimiter token identity through reasoning and tool-call
  handling, including reasoning opened by the prompt. Ordinary tokens spelling a special
  delimiter stay literal content instead of changing channels or creating a tool call.
  Buffered replies and streaming use the same path; text-delimited package formats remain supported.
- `crit eval run --url` no longer stops on a server that streams a reply in one piece: that
  reply's decode speed is unmeasured.
- `crit serve` bounds a conversation by the request body's 8 MiB, in bytes, instead of
  1,000,000 characters, which refused prompts that fit a long-context model (a full
  262,144-token prompt is over a million characters) and failed Eval's capacity and retrieval
  gates at the required context.
- `crit eval run --url` reports an error a server streams as that error, instead of
  "streamed no usage".
- `crit eval reference` scores models the way their transformers modeling code runs them, which
  it could not for Gemma 4 and failed on Qwen3 under current transformers: rotary embeddings per
  layer type, sliding-window masks on sliding-attention layers, the embedding's own scale, stored
  layer buffers (a layer scalar), per-sequence shared keys and values, final logit soft-capping
  and a tied output head. It reads ModelOpt NVFP4 (`weight`, `weight_scale_2`) as well as
  compressed-tensors NVFP4. A tiny Gemma 4 scores within 1e-4 of the model's own FP32 forward.
- `crit eval reference` scores long prompts without copying keys and values per query head, and
  a sliding-window layer reads only the columns its window covers: a 31B model's 32,768-token
  quality prompts no longer exhaust a 32 GB GPU, which failed the scoring there.
- The Eval suite has one identity on every platform. Windows sorted the suite's files without case
  (`agent/README.md.txt` after `agent/cli.py.txt`), so `crit eval prepare` recorded another
  `suite_sha256` there and the test pinning `crit-eval-v5` to its suite failed on Windows; data
  prepared on Linux now runs on Windows and back. Data prepared on Windows before this fix needs
  preparing again.

## [0.2.3] - 2026-10-20

### Overview

crit 0.2.3 makes `crit serve` hold up under real clients: a closed connection, a stalled reader, a
failing consumer or Ctrl+C now cancels the turn and settles the engine instead of leaving it running
or wedged. One server can serve a named catalog of models from several runs, admit bounded concurrent
requests for adapters that declare a capacity, and take image input for vision packages. `crit run`
publishes candidates, seals and packages atomically and recovers cleanly after a crash, `crit run
measure` retires its whole worker tree on every exit, and Core keeps native resources owned until the
work borrowing them has settled. Serving clients must keep their connection open until the reply
completes, since closing or half-closing it now cancels the request. The Python distribution and the
Core SDK share version 0.2.3.

### Added

- `crit serve --models catalog.json` serves named sealed candidates from several runs, each with its
  own capabilities and context limits ([Serving](docs/SERVING.md)). Switching models releases the old
  weights before the new ones load, so one server fits a one-model memory budget; an unknown model
  name returns 404.
- Concurrent serving: an adapter that declares `chat-concurrent-v1` and a request capacity gets that
  many requests at once, each with its own stream, cancellation and cleanup. A model switch queues
  fairly behind active work, and an unrecoverable failure retires the shared host before the next
  request is admitted. Other adapters stay serial.
- Image input: `crit serve` passes ordered image and text parts to `chat-images-v1` packages through
  the chat template, tool history included. `ResidentChat` gives engines request-scoped prepared
  prompts for their vision inputs; text-only adapters are unchanged.

### Changed

- `crit serve` cancels a turn when its client goes away: a closed or reset connection, also during
  non-streaming replies and quiet stretches of a stream. A client that half-closes its sending side
  while waiting cancels too. Cancelled requests leave the wait for the engine at once without
  touching the active turn.
- Final replies, stream snapshots and tool deltas go through one content check. Streams still accept
  whitespace and incomplete tool arguments as prefixes; the completed reply meets the same rules as a
  JSON reply.
- Core validates model progress the same way on serial and batched execution, and run records share
  reservation and attempt-result validation between publication and recovery. Malformed retained
  records are now rejected before anything is published; valid records are read as before.

### Fixed

- `crit serve` accepts a reply that is only reasoning, including one cut off while reasoning, instead
  of rejecting it and retiring the resident host. Empty replies, malformed text, tool-policy
  violations and exceeded budgets still fail.
- A stalled streaming client no longer holds the engine: writes observe the request's deadline,
  cancellation and shutdown, the connection is dropped, and a partial frame is never followed by an
  error or `[DONE]`.
- `crit serve` shutdown (Ctrl+C, SIGINT, SIGTERM) cancels active requests before waiting and answers
  queued ones with 503. An exception escaping a streaming consumer retires the resident host before the
  next request, so it gets a fresh engine. In both cases a failed cleanup stays blocked until a retry
  succeeds instead of passing for a completed close.
- `crit run measure` retires its whole worker tree, pinning process and adapter helpers included, on
  success, failure, timeout and interruption, and when the coordinator dies, on Linux as on Windows.
  Ctrl+C on Windows takes effect promptly. Reservations count the time spent waiting for the engine
  lock, so an expired attempt can neither launch nor publish a benchmark, and every setup error or
  interruption records an outcome.
- `crit run` publishes atomically: a candidate's source and identity together, a package only after its
  bytes and launch recipe check out, a seal with its temporary files outside the candidate. An
  interrupted create, seal or package no longer leaves a half-written directory that blocks recovery,
  and a retry keeps the original outcome. Staging stays outside `work/`, so the whole work directory can
  be an input.
- Run recovery and ordinary writers take one lock on the coordinator lease, so two writers can no
  longer own the same run. Windows recovery recognizes a dead coordinator whose process handle another
  process still holds.
- Core keeps native resources owned until the work using them has settled: graph caches keep
  unsettled executables through cleanup failures, final buffer release no longer loses an allocation's
  ownership when host memory runs out, and native engine libraries keep their resource owner alive
  while request handles are outstanding.

## [0.2.2] - 2026-10-19

### Overview

crit 0.2.2 makes serving one shared path for every engine. `crit_core.serving.ResidentChat`
renders the chat template, parses tool calls in the syntax the template teaches (or one the
engine package supplies), returns reasoning apart from the reply as `reasoning_content`, forwards
per-request sampling and streams, so an engine supplies only its token generator and gets every
serving fix without changes. Engines whose weights live in host memory get Core's `HostBlock`
and a machine-wide engine lock that also covers Windows and WSL together, and Inference gains
entries for host work inside captured graphs and bit-exact NVFP4 experts on the CPU. Config
readers take their key spellings from one shared table, Eval classifies tensors in one place, and
[What belongs in crit](docs/CONTRIBUTION_CRITERIA.md) states the criteria every change must meet.
Eval data prepared for 0.2.1 stays valid; references for Mamba-2 models must be built again (they now
follow the model's own time-step limit). The Python distribution and the Core SDK share version
0.2.2.

### Added

- `crit_core.serving.ResidentChat`: the shared body of a resident chat adapter (chat template,
  tools, per-request sampling, streaming, cancellation, telemetry); an engine supplies only
  its token generator. `crit_core.native.resident_engine` is built on it.
- Tool calls without model-specific code in crit: ResidentChat renders tools and tool history
  with the model's chat template and parses the calls (`crit_core.text.ToolCallParser`) in the
  syntax the template teaches when crit registers it (Qwen3-Coder `<function=...>` and Hermes
  JSON blocks), or in the package's own `ResidentChat.tool_format`. It declares
  `chat-tools-v1` and streams each call as one delta. Packages that declared the interim
  `chat-tool-template-v1` are served as `chat-tools-v1`.
- Core `HostBlock` and `HostBlockPool` (`crit/host_block.h`) with `cuda_pin_host_block`: large host weight
  blocks filled by the loader's threads, then pinned once and touched, and a process-wide pool that reuses a
  block only for the same key, size and identity and frees a mismatched one before the next engine
  allocates, so engines opened one after another never hold two copies.
- A machine-wide engine lock (`crit_core.machine`, `CRIT_ENGINE_LOCK`): `crit eval run/pin/ab/kernels/reference`,
  `crit run measure` and `crit serve` load one model at a time and say who holds the lock while waiting. A
  lease in a location every OS instance reads (`CRIT_ENGINE_LEASE`) makes Windows and WSL wait for each other.
- Eval records the CPU cores other processes kept busy during each speed case and names them in disturbed
  and `lower_context` failures.
- [What belongs in crit](docs/CONTRIBUTION_CRITERIA.md): the criteria every change to crit must meet.
- Reasoning apart from the reply: ResidentChat splits a reasoning model's thinking in the markers its chat
  template uses (`crit_core.text.reasoning`, `<think>...</think>`; a package can set its own
  `ResidentChat.reasoning_format`), and crit serve returns it as `reasoning_content`, streamed before the
  content. Adapters return it as the response record's optional `reasoning` and stream it as
  `emit({"reasoning": text})`.
- Inference entry [host-work-mailbox](inference/host-work-mailbox/README.md): host work the device posts and a
  stream joins inside a captured graph (device-reset doorbells, release, terminal failure answering), with the
  stream-front-end and poll-kernel waits chosen by a measured probe and an eager fallback.
- Inference entry [host-expert-arithmetic](inference/host-expert-arithmetic/README.md): NVFP4 expert rows on the
  CPU in a declared operation order (element-scaled or block-dot products; scalar oracle, AVX2 and AVX-512
  paths chosen at run time) that a matching CUDA function reproduces bit for bit, with a scale-decode hook for
  checkpoint-specific scale encodings.
- Reverse loans in expert-residency: `SettledSlots::begin_slot_loan` lets an owner that lent its memory to the
  expert cache (KV positions a request cannot reach, an idle prompt workspace) reclaim any set of slots with the
  existing loan lifecycle (protected-reader refusal, survivor relocation, binding invalidation, empty return).

### Changed

- Both config readers take the keys that spell each field from one table,
  `crit_core/model/config_keys.json` (CMake generates the C++ arrays), which also reads `llm_config`
  wrappers and `layers_block_type`. Eval's ceiling and context limit classify tensors in one place
  (`crit_eval.tensors`); results are unchanged on the checkpoints checked.
- `crit run package` refuses a resident chat package that does not accept `temperature`, or
  that offers no tool calls although the model's chat template teaches them.
- `crit run measure` pins cases in a separate process that exits before the measured engine loads.
- Eval prepare scores the turn of a template whose generation prompt holds more than a history turn (Gemma 4's
  empty thought channel) from where that turn's content begins, so a reply that starts like the prompt's extra
  text keeps those characters. Prepared data is unchanged for the checkpoints checked.
- Eval's reference gives Mamba-2 mixers config.json's `time_step_limit` instead of transformers'
  `time_step_min` clamp, which broke retrieval beyond about 64K tokens and read as drift.
- Inference entries: exact host/device expert splits and the worker-generation race
  (heterogeneous-expert-dispatch), lent cache capacity (device-expert-cache), W4A4 prompt GEMMs from
  calibrated input scales (quantized-projections), each stated as a rule to measure on the target machine.

## [0.2.1] - 2026-10-18

### Overview

crit 0.2.1 makes Eval catch engines that go wrong only after a long prompt: decode after
1K to 32K prompt tokens is now scored against the reference, long-context retrieval and
repeated requests are gates, and AGENTS.md treats unexplained output differences as bugs.
Serving accepts per-request sampling and tool calls, and `crit serve` loads the model
right away instead of hashing the whole checkpoint first. Eval data prepared for 0.2.0
(`crit-eval-v3`) must be prepared again, and Eval adapters need the new `prefill_decode`
score path. The Python distribution and the Core SDK share version 0.2.1.

### Added

- Serving passes validated per-request sampling overrides to opt-in resident adapters,
  including native Core penalties and logit bias. Model cards list supported sampling
  parameters and package capabilities.
- `chat-tools-v1` carries function definitions, choice policies, correlated tool-result
  history and validated calls in JSON and SSE responses. `chat-tools-strict-v1` compiles
  argument schemas with Core and verifies replies; `chat-tool-stream-v1` carries live,
  indexed argument fragments with append-only and strict-prefix validation.
  Model-specific adapters supply prompting, parsing and constrained decoding.
- Eval protocol `crit-eval-v4` adds three gates. The required `prefill_decode` quality
  path prefills 1,024-, 4,096- and 32,768-token prompts and scores the reference
  continuation through ordinary decode steps against the drift budget, per prompt length.
  The retrieval gate asks for facts placed at the start, middle and end of prompts up to
  the required context and checks the answers verbatim. The request-history gate requires
  the same greedy tokens on a fresh engine and after every other request it served, in
  every exactness mode. With `--baseline`, status notes a drop in plain-greedy equality.
- `crit serve --verify-model` reads and SHA-256s every model file against the run's frozen
  manifest before loading, and gives engines without `chat-model-file-v1` a private copy.

### Changed

- `crit serve` checks each model file's size against the frozen manifest and maps it in
  place by default, so the engine starts loading at once instead of after a full read and
  hash of the checkpoint. A checkpoint with the same sizes but other bytes is served
  unnoticed unless `--verify-model` is given. The default load timeout assumes 200 MB/s
  (50 MB/s with `--verify-model`), and Windows no longer re-hashes copied model files on
  every read.
- AGENTS.md lays out the build loop (recon, plan, build, evaluate, revisit recon, repeat)
  with its commands and stop conditions, and treats a drop in plain-greedy equality or a
  difference between repeated identical requests as a bug until a root cause is shown.
- README case study 02 shows the corrected Flash-Next candidate 6; candidate 5 is marked
  superseded (it used prefetched experts before their scales were set).
- Data prepared for `crit-eval-v3` must be prepared again, with a new reference and pin:
  `prepared.json` now holds the long prompts and the token strings retrieval matches with.
  An Eval adapter must declare and implement the `prefill_decode` score path.
- Unsupported sampling controls and unknown request parameters now return HTTP 400
  instead of being ignored. Existing sealed packages continue to serve plain chat;
  clients must omit unsupported overrides or use a newly built capable adapter.

### Fixed

- Windows resident startup no longer treats a transient permission denial during
  ownership-record replacement as owner death: the guardian uses the supervisor's
  bounded retry. Changed ownership and persistent failures still stop execution.
  Unexpected resident exits report their exit code and diagnostic log location.

## [0.2.0] - 2026-10-17

### Overview

crit 0.2.0 adds the Models module for recorded weight transforms, a recon step that freezes
each model's sampling, chat template and stop tokens into its candidates, measured run
benchmarks, and Core engine libraries that need no adapter code. Eval checks exactness on
the loaded engine, replays pinned text at long context and stops disturbed rounds. The
Python distribution `crit` (now including `crit_models`) and the native Core SDK
(`critCore`) share version 0.2.0. Runs recorded with 0.1.0 still verify.

### Added

- **Models** ([guide](models/README.md)): `crit models` produces transformed weights and
  records how they were made. The first workflow is post-training quantization:
  `plan` resolves a recipe against checkpoint headers, `ptq` quantizes projections to
  FP8 per-channel or NVFP4 by round-to-nearest or GPTQ (layer-streamed calibration
  through the official `transformers` code) and exports a compressed-tensors checkpoint,
  `eval` runs the output in an engine through Eval against the source model's data and
  reference, and `verify` re-hashes it. The `models_transform` record keeps the
  resolved recipe, calibration data and token-window hashes, seed, source and output
  file hashes, error measures and transform cost; `models_evaluation` binds an Eval
  result to it.
- `crit run --id RUN recon` gathers the model's sampling defaults, stop tokens, chat
  templates and key `config.json` fields with file hashes into `work/recon.json` and
  scaffolds `work/recon.md`; later candidates freeze its `frozen` block as
  `configuration.recon`. A run whose model is a folder needs recon before its first
  candidate; single-file models are unchanged.
- `crit serve --load-timeout`. The default bound on loading the starting candidate is now
  600 s plus the model's size at 50 MB/s, instead of a fixed 600 s.
- Run records explain ineligibility. `crit run benchmark` and `seal` warn when a benchmark
  or candidate cannot qualify, and `select` names every blocker. The reasons are derived
  when asked; telemetry keeps the 0.1.0 projection, so earlier runs still verify.
- `crit run measure` benchmarks a candidate's resident cases itself, through the package's
  Eval adapter in a reserved child process. The first measurement of a case pins its greedy
  text, when two runs repeat it, and every later candidate replays it, so decode speeds compare across candidates;
  load time and host and device memory are sampled during the run.
- Core engine libraries: `crit/native_engine.h` exports a Core engine from a shared
  library with a C ABI (`CRIT_EXPORT_ENGINE(factory)`), and `crit_core.native` loads it
  and supplies the stock eval-v1 and chat-resident-v1 adapters, so an engine needs no
  adapter code of its own. `ReplayContinuation` carries Eval's replay text to the session.
- Eval checks exactness on the loaded engine when the adapter declares `plain` and offers
  `generate_plain()`, instead of opening a second engine with `speculative` false. For an
  engine that streams 64 GB of experts from host memory, that removes one of three loads per
  round (about 3 minutes of 19). `PlainDecoding` (`"plain": true` in the generate options)
  carries the request to native engines, and the stock adapter offers it when the engine
  declares `plain`.
- Eval stops a round when a speed case's decode speeds spread by more than
  `disturbed_spread` (25%) between repetitions: the device was disturbed. The partial result is
  marked `aborted`, `run` exits with code 3, and `--keep-disturbed` finishes the round anyway.
  A disturbed Flash-Next round previously ran its full 25 minutes, failed the lower-context gate
  on noise and had to be repeated.
- `fp8-kv-attention` adds `prefill_fp8`: prompt attention with Q·Kᵀ and P·V on E4M3 tensor
  cores for head dimensions 64, 128 and 256, about twice `prefill`'s throughput at long
  contexts with E4M3 rounding of queries and probabilities. Opt-in; `prefill` is unchanged.

### Changed

- A benchmark record that cites no checks binds the case's latest check per gate for the
  same package, instead of staying exploratory.
- Safetensors indexes may list shards in subfolders (`tensors/model-00001-of-00004.safetensors`)
  as portable relative paths, in Core's C++ and Python readers and in the import audit;
  Eval's model identity counts them.
- Core's static libraries build position-independent, so engines can link them into a
  shared library without rebuilding Core.
- Under WSL, `--write-launchers` writes a `serve.cmd` that starts `serve.sh` in the same
  distribution.
- A killed process tree has 60 s instead of 5 s to exit before cleanup counts as failed.
  Engines that pin tens of GB of host memory take longer than 5 s.
- Eval's long-context points replay the longest speed case's pinned continuation when the
  data has one and the adapter replays, so `context_decode` no longer moves with each
  engine's own text; it compares only between results decoded on the same text.
- Eval's prefill and TTFT floors count attention at the faster of the BF16 and FP8 rates.
  An engine with FP8 attention beat the BF16 floor.
- `crit eval kernels` charges each kernel only the time after every earlier kernel ended,
  so a kernel launched early under programmatic dependent launch is not charged its wait;
  `span_ms` keeps raw durations.

### Documentation

- Eval adapter: an adapter may keep host-side weights in the process across the engines one
  `crit eval run` opens one after another; device state is built fresh for each engine.
- RUN_RECORDS and AGENTS: publish a case's checks before the benchmark that cites them.
- BUILDING_ENGINES: the example request no longer overrides `settings.toml` headroom.
- CUDA backend: free memory under WDDM. device-expert-cache: measured next-layer
  prefetch results.
- README: case studies for the Qwen3.8-27B NVFP4 and Qwen3.8-Flash-Next FP4-Local engines
  on one RTX 5090.

## [0.1.0] - 2026-10-16

### Overview

First public release of crit: reusable infrastructure for building a bespoke inference
engine around a chosen model, workload and target hardware, with a common evaluation
suite and an OpenAI-compatible serving API. The Python distribution `crit` and the
native Core SDK (`critCore`) share version 0.1.0.

### Added

- **Core** ([guide](core/README.md)): a C++20 SDK with Model, Session and Backend
  interfaces for generation and query scoring, request scheduling and admission,
  resource accounting, readiness, cancellation, deadlines, quarantine and safe
  retirement. Mixed batching of up to 4096 request lanes with shared token-work budgets,
  decode priority with prefill aging and an optional async batch pipeline. Requests
  carry sampling rules (temperature, top-k, top-p, min-p, penalties, logit bias,
  `min_tokens`), log-probabilities with top alternatives, stop strings, decoded text and
  per-token output events, and report phase timing and reused prompt tokens.
- Core component libraries, each optional and documented beside its sources:
  text (`crit/text.h`: stop matching, streaming detokenization), sampling
  (`crit/sampling.h`: Philox counter-based draws usable in device code, a `Sampler`
  with a documented rule order), state (`crit/state.h`: a storage-agnostic ledger of KV
  pages and recurrent state slots with prefix reuse), files and checkpoints
  (`crit/json.h`, `crit/file.h`, `crit/sha256.h`, `crit/safetensors.h`,
  `crit/model_config.h`), host-memory admission (`crit/host_memory.h`), structured output
  (`crit/constraints.h`: token, choice, JSON-object and byte-grammar matchers), operator
  ranges (`crit/trace.h`, with an optional NVTX sink), and CUDA services (graph capture,
  shared device allocations, category timing, spare device memory).
- The `crit_core` Python package: model files, strict JSON and safetensors readers,
  `config.json` rules, tokenizer, Hugging Face chat templates, generation config,
  streaming detokenization and the JSON schema/regex/EBNF compiler for structured output.
- **Inference** ([guide](inference/README.md)): 26 techniques with implementation code, runnable
  examples, ownership contracts and adaptation guides, grouped by category: attention
  (attention-state, flash-prefill-attention, fp8-kv-attention, partitioned-attention,
  rotated-kv-quantization), projections (quantized-projections, rmsnorm-bf16,
  stored-tile-gemm), experts (async-expert-contributions, device-expert-cache,
  expert-residency, heterogeneous-expert-dispatch, ssd-expert-materialization),
  speculation (device-speculative-round, draft-vocabulary-subset, mtp-decode,
  recurrent-state-recovery), sampling and constraints (token-constraints,
  token-sampling), execution (concurrent-request-execution, recorded-selection,
  session-scheduling) and memory and loading (artifact-loading, checkpoint-storage,
  host-memory-admission, model-import-audit).
- **Kernels** ([guide](kernels/README.md)): checks for outside kernels (native integration,
  the actual device, exactness across column counts), pinned dependency retention,
  artifact verification and preservation of third-party notices, with native
  integration examples.
- **Eval** ([guide](docs/eval/README.md)): one speed, quality and exactness suite that
  every engine runs the same way through a small `eval-v1` Python adapter, with prompts
  frozen as token ids per model and a layer-streamed FP32 reference. It reports decode,
  prefill, time to first token and quality against physical limits, computes an engine's
  max context from device memory, checks capacity and long-context behavior, scores
  held-out prompts in eight languages plus code and SQL, and offers pinned decode text
  (`crit eval pin`), paired A/B runs (`crit eval ab`), a CUPTI kernel report
  (`crit eval kernels`) and device probes (`crit eval probe`).
- **Serving** ([guide](docs/SERVING.md)): `crit serve` serves a sealed candidate as an
  OpenAI-compatible API with `/v1/models`, `/v1/chat/completions` (JSON or streamed,
  stop strings, system messages, `response_format` JSON object and JSON schema), optional
  API keys and CORS, and `/health`; `--write-launchers` writes `serve.cmd` and `serve.sh`.
- **Run records** ([guide](docs/RUN_RECORDS.md)): `crit run` keeps retained candidate
  source and packages, lineage, checks, measurements, candidate budgets, quality and
  exactness terms, a weights policy, process reservations with owner-aware leases,
  cleanup and immutable sealed outcomes. An optional untracked `settings.toml` holds
  machine settings.
- **Validation and packaging:** host and native fixtures, one installable `crit`
  distribution (`crit`, `crit_api`, `crit_eval`, `crit_kernels` and `crit_core`), an
  installable C++ SDK and Windows/Linux CI workflows.

Inference examples include host proofs of concept and native sources with explicit
validation scopes. Model arithmetic, formats, tokenization and sampling choices belong
to each engine; model quality and device performance require target validation.

---

[Unreleased]: https://github.com/crit-labs/crit/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/crit-labs/crit/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/crit-labs/crit/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/crit-labs/crit/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/crit-labs/crit/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/crit-labs/crit/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/crit-labs/crit/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/crit-labs/crit/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/crit-labs/crit/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/crit-labs/crit/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/crit-labs/crit/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/crit-labs/crit/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/crit-labs/crit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/crit-labs/crit/releases/tag/v0.1.0
