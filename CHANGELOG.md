# Changelog

All notable changes to crit are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
with versions following [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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

[Unreleased]: https://github.com/crit-labs/crit/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/crit-labs/crit/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/crit-labs/crit/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/crit-labs/crit/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/crit-labs/crit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/crit-labs/crit/releases/tag/v0.1.0
