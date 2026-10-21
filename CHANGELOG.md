# Changelog

All notable changes to crit are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
with versions following [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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

[Unreleased]: https://github.com/crit-labs/crit/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/crit-labs/crit/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/crit-labs/crit/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/crit-labs/crit/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/crit-labs/crit/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/crit-labs/crit/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/crit-labs/crit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/crit-labs/crit/releases/tag/v0.1.0
