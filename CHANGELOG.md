# Changelog

All notable changes to crit are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
with versions following [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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

[Unreleased]: https://github.com/crit-labs/crit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/crit-labs/crit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/crit-labs/crit/releases/tag/v0.1.0
