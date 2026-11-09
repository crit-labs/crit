# Changelog

All notable changes to crit are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
with versions following [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.17.1] - 2026-11-09

### Overview

crit 0.17.1 runs `crit run` from a pip-installed crit, without a checkout: a run follows the installed
release's guides, and the commands that run Eval name their project to it, so its settings.toml applies. Two
engines in one process no longer break each other's captured prompt chunks (a capture takes its cuBLAS workspace
from a pool kept for it), and a captured round's log-probabilities equal the step and score paths' bit for bit.
The run store verifies each candidate once per operation, and on Windows a process tree keeps a Microsoft Store
Python's children in its job, so crit's shutdown ends them and their files can be deleted at once. crit-labs'
CUDA kernels are still not on the Hugging Face Hub: the plugins that load them run their torch fallbacks. crit
0.17.1 needs crit-eval 0.6.1.

### Changed

- crit needs crit-eval 0.6.1 (`crit-eval==0.6.1`, and `crit-eval[reference]==0.6.1` in the `reference` extra),
  whose Eval reads the settings of the project crit names to it.
- `crit mcp --project P` names P to every job it starts (`CRIT_PROJECT`), so a `crit eval` job reads P's settings.toml as
  `crit run`'s commands do. Before, its jobs used crit's default project.
- crit run's store walks and hashes each retained inventory of a candidate (its source, package and seal) once per
  operation. An operation reads a candidate's records several times over (an append checks the new record, refreshes
  the candidate's telemetry, then the run's state reads every candidate), and each read verified the inventories
  again: a served-engine test's setUp, two candidates through eight appends, walked 137 of them, now 76. Each store
  operation is one verification scope, in which a folder that matched its rows is not walked again for the same rows
  and, except on Windows, a file whose stat is unchanged is not hashed again; the next operation verifies everything
  again.
- `crit run` keeps runs from an installed crit without a checkout (`pip install crit`, or `python -m pip install .`
  run outside the checkout): its `--project` is then any folder (default: the current one), the folder of
  `settings.toml` and the default `runs/`, where a crit running from a checkout still wants a crit repository. Such a
  run's brief (`agent_request.md`) names, as the repository its agent follows, crit's repository at the installed
  release's tag (crit X.Y.Z is published from vX.Y.Z), with the version and the folder crit is installed in, and its
  replays point at AGENTS.md there; runs stay outside the folder crit is installed in, as a checkout's stay outside
  its published folders. A checkout's runs keep the checkout as their repository. `crit serve --workspace` serves
  such runs the same way. tests/test_wheel_install.py keeps a run of the CPU toy engine with the installed wheel,
  outside any checkout: create, recon, capture, resolve, a first plan, a candidate and its build.
- Eval reads the settings of the project the command uses: crit names it to crit-eval in `CRIT_PROJECT` while the
  command runs, `crit run` and `crit bench` their `--project`, `crit eval` crit's checkout or, for a crit installed
  without one, the folder it runs in (unless the environment names a project). crit-eval found the settings only in
  a checkout, so an installed crit's `crit run eval` and `crit eval` ignored the project's headroom,
  `context_reserve_gib`, `context_tokens` and `eval_packages`; a crit-eval that reads `CRIT_PROJECT` (crit-eval's
  Unreleased) takes them. The wheel test's run reads its project's `settings.toml`.

### Fixed

- An installed crit without crit Eval says to install the crit-eval release it pins (`python -m pip install
  crit-eval==0.6.1`), not an editable crit-eval checkout beside it, which only a checkout has.
- `crit run measure` closes the nvidia-smi sampler's pipe when it stops it (Python reported it unclosed).
- crit_plugins: an engine closed in a process that keeps another leaves the other's captured rounds and prompt chunks
  their memory. Every capture runs on Core's one capture stream, a captured graph keeps the address of the cuBLAS
  workspace its products ran with, and torch frees all of a stream's workspaces when any graph captured on it is
  destroyed, so the closed engine's graphs freed the workspace the open engine's still wrote: its memory went to
  other tensors, or back to the driver and the next replay failed with an illegal memory access. Each capture now
  runs with workspaces made in a memory pool Core keeps for them (`crit_plugins.rounds.capture_workspaces`), where a
  freed workspace goes back and nothing else is made. Serving was not affected (crit serve closes an engine before it
  loads the next); in-process tests that open engines side by side were.
- token-sampling: a captured round's log-probabilities are torch's FP32 `log_softmax` of its logits rows, what a torch
  model's step and score paths compute, so a token's log-probability from a round equals the step's and the score
  path's bit for bit for the same logits (torch's kernel reduces a row by its length and its address modulo 16 bytes,
  the same for every row of a vocabulary that is a multiple of 4). Before, the CUDA sampler normalized the rows itself
  (FP32 exps with FP64 sums, after Core's double TokenLogprob) and the torch law in FP64, up to 2.4e-7 from the step
  and score paths. The CUDA sampler now ranks the alternatives over those values as given
  (`enqueue_ranked_logprobs`), in Core's order.

- crit's process trees on Windows keep their processes' children in their job with a Microsoft Store Python too, so
  closing a tree ends and waits for them: crit serve's resident host (behind its launch guardian), an MCP job's
  command's children. A Store Python is a packaged app, and Windows starts a packaged process's children, and a Store
  interpreter that a venv's launcher starts, outside the job of the process that starts them unless that process was
  created with the desktop app policy that keeps them in. A tree now creates its first process with that policy
  (subprocess.Popen cannot pass it), and starts this venv's python.exe as the base interpreter, as guarded launches
  and MCP jobs did, so what a `python` build command of `crit pull` starts stays in its tree too (the Store
  interpreter a venv's launcher starts takes Windows' default policy). Before, the job held that process only:
  retiring the resident host left it running until it noticed its closed channel (a hung host held the retiring
  request, crit serve's shutdown and an interrupted `crit run measure` for as long as it hung), a stopped MCP job's
  children kept running, and the files they held open (the host's `runtime.log`, a job's `.out` and `.err`) could
  not be deleted when crit returned (WinError 32).

## [0.17.0] - 2026-11-08

### Overview

crit 0.17.0 builds every engine from plugins: an engine is Core plus a package of plugin folders that fill its slots,
and engines of the earlier adapter form are refused. crit's library of Inference techniques is plugins, every slot
but `model` served on the device and tested against its host form, and the crit wheel carries it. Core's round driver
captures decode rounds with drafting, sampling and adaptive widths inside them, in a plain and a log-probability form,
and Core runs them natively, so log-probability requests and replays decode in captured rounds too. Compute kernels
left crit: plugins load crit-labs' kernels through the registry. The quality modes are `preserve` and `speed`
(`as_quantized` is refused), and new runs climb the candidate ladder under plan rules 7. Core's engine C ABI is
version 5. For crit serve's clients, the `/api/chat` dialect counts under the metrics label `api_chat` and `GET /`
answers `crit is running`. crit-labs' CUDA kernels are not on the Hugging Face Hub yet: a later release pins them,
and until then each plugin that loads one runs its torch fallback, as the engine's facts say. crit 0.17.0 needs
crit-eval 0.6.0.

### Added

- Engines built from plugins ([docs/PLUGINS.md](docs/PLUGINS.md)): each technique is a plugin folder of the engine that
  fills slots (attention, KV cache, drafter, ...), says what it needs, provides and conflicts with, takes checked
  options, declares the registry kernels it loads and carries its own tests; a fused plugin fills several slots and is
  tested against the separate pair it replaces. `crit_plugins`, in Core's Python package (crit-core), holds the
  contract an engine loads with: `crit_plugins.assemble` puts the plugins together when the engine loads, checking one
  plugin per slot, every need, conflicts and one memory budget, and crit serve's engine facts report the plugins
  loaded. `crit plugins list|check|test|variants` checks the manifests and a selection and runs each plugin's tests in
  a process of its own. `crit run ... ablate` serves a candidate with each plugin switched off and alone, built again
  in the same resident host, with a short bench of the run's cases and crit smoke per variant, and says which plugin
  pays, slows, trades or breaks the engine before Eval measures it.
- Inference techniques as plugins, crit's library of them ([inference/](docs/PLUGINS.md#the-library)): every
  technique that fills a slot is one plugin of the same id, its `plugin.json`, `plugin.py` and tests beside its README
  and code, 21 techniques in 14 slots. A technique's variants are options (`quantized-projections`' `format`,
  `device-expert-cache`'s `policy`) and parts that only work together are one fused plugin (`draft-trees`). The 12
  base plugins in `inference/base/` give each slot its plain plugin, which techniques are tested against (`reference`,
  now for a plugin of one slot as well as for a fused one) and a selection's `base` falls back to; a folder of plugins
  keeps its base plugins in `base/`. Every slot crit names has interfaces and shared cases (`crit_plugins.slots`),
  which `assertImplements` and `assertMatchesReference` hold a plugin to. `crit plugins add ENGINE/plugins ID...`
  copies plugins into an engine with their references, the plugins their tests use (`test_with`) and the code they
  import, and `crit plugins list|check|test` act on the library without a folder. New slots: `expert_residency`,
  `expert_source`, `embeddings`, and `norms` has its interfaces. Three Inference entries only lend their code to
  plugins, the eight an engine is built on rather than plugs in are none (`ENGINE_SERVICES`), and `crit inference
  show` lists a technique's plugin.
- Every slot but `model` has library plugins that serve on the device, each tested against its host form: the slot's
  device form (`crit_plugins.device.DEVICE_INTERFACES`) over torch, providers first (cuBLASLt through torch's scaled
  matmul, FlashInfer, fla), crit-labs' registry kernels where no provider has the operation, and a torch fallback with
  the same contract. One library technique, expert-residency, stays a host reference.
- Device extensions named by the facts plugins provide (`crit_plugins.device.DEVICE_EXTENSIONS`): `rounds:<slot>`
  (the slot's device form inside a captured round over a `RoundBatch` at fixed device addresses), `prompts:<slot>`
  (inside a captured prompt chunk over a `PromptBatch`), `draft:device`, `verify:device`, `experts:bundles`,
  `experts:routes`, `experts:cache`, `experts:store`, `embeddings:host`, `kv:scales`, `kv:keys` and the attention
  facts `attention:blocks`, `attention:tree`, `attention:lse`. Assembling an engine refuses a plugin that provides an
  extension without its methods, and `assertImplements` checks them, on the pool for a pool's facts
  (`POOL_EXTENSIONS`). A plugin whose configuration keeps it out of captured graphs says why (`round_problems()`), and a
  pool with a fixed number of round rows says how many (`round_rows`). `crit_plugins.device.topk_blocks` is the
  canonical block-sparse selection (ties to the lower block id, the incomplete tail block always read),
  `invariant_rows(projections)` the rows a projections plugin keeps invariant, and `capturing()` asks whether a stream
  captures where CUDA is.
- Core's round driver, `crit_plugins.rounds` ([docs/PLUGINS.md](docs/PLUGINS.md#captured-rounds)): `Rounds` composes
  the library's device forms into one captured decode round per (rows, width, mode) (the drafter's `draft:device` from
  the model's draft head, the model's `verify(view, batch)` over a `RoundBatch`, the sampler's `rounds:sampler`
  predictions, Core's chain accept or the verifier's `accept_device`, each pool's `commit_round`), binds and seeds
  sequences in round slots in every pool, holds rows a step does not schedule, runs every part's `start_round(view)`
  and `count_round(view)` hooks inside the round (the extra parts a model registers too, such as a device expert
  cache), counts each round into the engine's named `round_counters`, plans its captures' measured memory through the
  placement plugin (`"rounds:captures"`) and hands its graphs to Core's native rounds. It refuses to capture, naming the
  plugin, when one lacks its round fact or gives a reason of its own, and the model then decodes through `step()`; a
  pool of fewer `round_rows` than the round's rows is refused when the driver is built. `TokenRing` keeps each round
  slot's last tokens on the device, for inputs a model computes from the tokens before each column (hashed n-gram
  ids). device-speculative-round's python_round keeps its rejection and copy-draft helpers over it.
- Adaptive draft widths in Core's round driver (`Rounds(widths=..., width_policy=...)`): rounds are captured at several
  widths and a policy chooses each round's width among them inside the captured round, `FixedWidth` (the default),
  `StepWidth` (one captured width up after a round that kept every draft, one down after a round that kept fewer) or
  `RateWidth` (the width with the most tokens per unit of round time from each draft depth's acceptance rate and the
  captured rounds' measured times, `calibrate()`). The choice reaches Core's native rounds through the engine's widths
  array, which Core reads before each round (`StepRounds`' field `widest` is the widest round, which open checks the
  engine's grants against), so no host code runs between rounds; greedy output is plain greedy at any width and sampled
  rounds keep the target's law. A model's own launches ask `rounds.width(rows)`, its info takes `width_facts()`, and
  the telemetry gives the policy and the rounds run at each width (`drafts.rounds_by_width`).
- Captured prompt chunks through Core (`crit_plugins.prompt_graphs`): a sequence's short prompt chunk runs as one CUDA
  graph in a padded bucket (16, 64, 128 and 512 tokens by default), its padding changing no state, for selections whose
  plugins provide `prompts:<slot>`; pools stage the sequence before each replay and settle it after, a run costs one
  copy of its tokens from pinned staging (positions and validity derived inside the graph), the graphs slot's plugin
  chooses between the eager path and the buckets by their offer times (measured after warm replays), and the captures'
  memory is planned (`"prompts:captures"`).
- Registry kernels (`crit_plugins.kernels`): a plugin declares each crit-labs kernel it loads in plugin.json
  (`"kernels": [{"name", "version", "revision", "fallback"}]`, optionally `backend`), and
  `registry_kernel("crit-labs/NAME", folder=...)` loads it at that pin through the `kernels` package, refusing one the
  manifest does not declare; `LOCAL_KERNELS` points a registry name at a local build, and a kernel that is not there
  leaves the plugin on its torch fallback. The engine's facts carry what each kernel request found under `kernels`
  (source `local`, `hub` or `fallback`, revision, why). A plugin's pin is read once per folder and answered with one
  stat after that.
- Every generation's result carries its rounds: `rounds` (one per decode row), `accepted` and `drafted` per round
  when it drafted, and the engine's `round_counters` (such as `expert_reads`/`expert_hits`) summed from its native
  rounds' words, with the model plugin's optional `request_facts(session)` merged in round by round (its step()
  rounds' entries take their places; its counts of a round counter add to the native rounds'). The eval-v1 adapter's
  in-process cohort rows carry each request's `expert_reads` and `expert_hits`.
- `StepEngine.telemetry()` (the plugins' `telemetry()` dicts through `Assembly.telemetry()`, Core's step and native
  round counters, and the load's device memory, `"load"`), crit serve's /health `engine_telemetry` (asked of the
  engine beside generation; None when its report fails) and the eval-v1 adapter's `telemetry()`.
- Core's loader measures what each plugin's factory takes on the device: a plugin's declared memory counts as planned
  only as far as it was allocated, and a plugin that takes more than it declares is refused by name. It hands torch's
  cached, unused device memory back once every plugin is made, before the engine opens and its captures are measured.
- A step model may declare `prefill_work_quantum` (Core's grants of prompt work in its multiples): with its prefill
  chunk as the quantum, a prompt's chunks start where they would alone, whatever shares its steps.
- `crit_core.engine.stored_facts` reads a mixed-precision checkpoint's per-layer formats (ModelOpt's
  `quantized_layers`): stored:nvfp4, stored:fp8, and the new stored:mxfp8 / stored:mxfp4.
- `crit smoke` ([smoke/](smoke/README.md)): small tests of one quality component each, run against any engine crit
  serve serves in seconds and without a reference: prefill rows well formed, plain prose at a low perplexity, prefixes
  against the whole prompt (masks, tile and chunk edges), decode against prefill, a greedy reply against its rows and
  its score, repeated, isolated and concurrent requests, speculative against plain decoding, and a repeated text
  recalled at a long position. A build catches its own bugs before an Eval round is spent on them.
  `crit run ... serve --check` runs it at the run's context and at a quarter of it (`--no-smoke` leaves it out),
  and a suite past `--budget` seconds (default 300) stops as failed.
- Native decode rounds for step engines (`StepRounds`, `crit_engine_open_steps_rounds`;
  `crit_core.native.NativeRounds`): Core runs a decode step itself when its rows are bound to the engine's
  round slots. It launches the captured round, reads the block, reports the tokens and calls `commit` only for rows
  it trims or ends, so a Python engine's interpreter no longer sits between rounds. `crit_rounds_cuda_*` launches
  captured CUDA graphs through the driver API loaded at run time, and `SpeculativeRounds.native_rounds` builds it
  from a Python engine's captured rounds. Log-probability requests and replays decode in native rounds too: a driver
  built with log-probabilities captures each (rows, width, mode) round twice, sharing one graph pool, and a round
  runs its log-probability form only when one of its rows asks for log-probabilities or follows a continuation, else
  its plain form, which costs less (the telemetry's `rounds.by_form` and `rounds.captured_bytes`).
- New technique and plugin `grouped-experts` (experts slot, reference experts-reference): routed SwiGLU experts over
  expert bundles in a slot store, in bf16, fp8 (channel and 128 x 128 block scales, gate's and up's grids apart), nvfp4
  (W4A16, or with `activations: nvfp4` each layer as its stored matrices declare: W4A4 where they carry input scales,
  W4A16 where not, the facts naming each layer's), int4 (E4M3 per 16) and mxfp4; the routes grouped by expert on the
  device through the registry kernel crit-labs/grouped-experts, its torch fallback elsewhere; a route's bits independent
  of the rows beside it, of chunking and of decode against prompt; capture-safe (`rounds:experts`, `prompts:experts`),
  every workspace a capture used kept for the plugin's life; `experts:bundles` (`layout`, the one definition of a
  bundle; `pack`, which casts nothing; `forward_bundles` over a store, a slot table and the store's format, -2 slots
  skipped for another tier) and `experts:routes` (`route_outputs`, the declared route-order combine). Option
  `arithmetic: declared`: crit's integer block order for NVFP4 W4A4 and int4 W4A8 experts, the CPU expert tier's bits,
  on the kernel only (refused without it). tests/test_hybrid_engine.py runs the hybrid engine a second time with it in
  the experts slot.
- New base plugin torch-norms and technique fused-norms for the `norms` slot: a host form over rows of numbers in FP64
  (`rms_norm_rows`, `gated_rms_norm_rows`, `rope_rows`, `act_mul_rows`, `l2_norm_rows`, `stream_sum_rows`,
  `stream_add_rows`) with crit's cases, and a device form (`rms_norm` with an offset weight, a weight per head or
  stream, a residual advanced in place and a chosen output dtype, `rms_norm_heads`, `rms_norm_rope`, `gated_rms_norm`,
  `rope` in half or interleaved pairs with partial rotation from inverse frequencies or a table, `silu_mul`,
  `gelu_mul`, `sigmoid_mul`, `l2_norm`, `stream_sum`, `stream_add`), FP32 inside, one rounding, each row's bits
  independent of the rows beside it, capture-safe (`rounds:norms`, `prompts:norms`). torch-norms runs torch ops with
  every reduction through torch's fused rms_norm (torch's mean over a row gives a row of a few-row call other bits
  than the same row in a long call at hidden widths) and, on the CPU, sigmoid and SiLU composed from exp. fused-norms
  uses FlashInfer for the calls whose function it computes exactly, one launch per call of the registry kernel
  crit-labs/fused-norms for the rest and torch-norms' arithmetic where the kernel is not there, with conformance checks
  any norms plugin's tests can run. At a 5,120-wide row on one RTX 5090 in a CUDA graph, the residual norm takes
  2.4 us in 1 launch against 11.2 us in 9 for the torch chain. The device and hybrid engine tests' model plugins call
  the norms slot, and the device engine test serves torch-norms in fused-norms' place to the same greedy tokens.
- quantized-projections serves on CUDA with no kernel of its own: stored BF16, FP8 (per channel or tensor), MXFP8 (E4M3
  with an E8M0 exponent per 32 inputs) and NVFP4 matrices; prompt rows through cuBLASLt (torch's scaled matmul) with FP8
  rows or NVFP4 activations (W4A4) quantized by one kernel each, or with 16-bit activations on the matrix decoded a
  chunk of rows at a time into a declared workspace (`prompt`: BF16, FP16 accumulation or FP8 rows; `prompt_roles` per
  matrix role; BF16 and weight-only ones through crit-labs/fixed-tile-gemm with `prompt_gemm: fixed_tiles`, every format
  through crit-labs/pinned-gemm with `prompt_gemm: pinned`: one split-K-free cuBLASLt algorithm per GEMM shape and
  format, chosen at load without timing, so a prompt row's bits are the same at every prompt length, chunk and padded
  bucket (`row_invariance` `prompt_rows`; prompt rows then skip the tile; on an RTX 5090 as fast as torch's NVFP4 GEMM
  past 64 rows, FP8 rows about a third slower at 8,192 rows with their scales applied after the GEMM, small BF16 and
  weight-only GEMMs at 17-64 rows slower still); FP8 matrices' W8A16, the E4M3 weights exact in BF16, with
  `fp8_activations: bf16`); decode rows (calls of up to `decode_rows`, 8 to 16) through crit-labs/mma-rows' tensor-core
  tile (`gemv: mma`) or crit-labs/decode-gemv's GEMVs, at the prompt rows' activations by default (`decode_activations:
  prompt`; `bf16` makes them weight only), each row's bits its own alone and batched; short prompts through the tile as
  far as it measured faster (`tile_rows: auto`). Every call path of a matrix runs once at load (Triton compiles,
  cuBLAS's first use of a shape, CUDA modules), so no request compiles, and the facts state every format's prompt
  arithmetic (`prompt_arithmetic`). Without the registry kernels it falls back to cuBLASLt one row per call or torch
  with the same row invariance. Draft-only matrices are re-encoded at load (`load(tensors, format, role,
  encode="nvfp4"|"fp8")` from stored BF16/FP8, code for code with the FP64 codec; the facts record each role under
  `draft_encoding`), and it provides `rounds:projections` and `prompts:projections`. projections-dense provides
  `rounds:projections` and gives FP32 logits on the CPU too.
- A declared FP8 order for E4M3 weights against E4M3 activations (MXFP8, 128 x 128 block scales, row scales): one
  e4m3 tensor-core MMA per 32-block with a zero accumulator, a declared FP32 step and lane tree, and a declared
  activation quantizer whose host twin is quantized-projections' `fp8_block.h`. Its decode kernel for 1 to 8 tokens
  and prompt GEMM, which keep it so decode needs no prompt GEMM on a handful of rows to agree with prefill, are the
  registry kernel crit-labs/fp8-block-rows (decode rows near a plain read's rate on an RTX 5090, prompt GEMM 79-97
  TFLOPS), with its device reference and the fixture that checks three implementations give the same bits.
- attention-state's KV pool and the attention slot's one device form (attention-state's code/device_attention.py, shared
  by softmax-attention and partitioned-attention): BF16, FP16, FP32 and layer-scaled FP8 pages beside FP8 E4M3
  (`fp8:row`) and INT8 pages with a scale per position and KV head (IEEE quotients, `amax / 448` and `amax / 127`, so a
  kernel computes the same scales); static FP8 scales per layer (`kv:scales`), pools of keys only and rows at chosen
  slots (`kv:keys`); step buffers at fixed addresses (`step_rows`, `step_tokens`) so a model replays its decode steps
  from CUDA graphs; round slots for captured rounds (`rounds:kv_cache`: `bind(slot, sequence, capacity)` reserves the
  pages, slots computed on the device with tokens no bound sequence holds at a dump page, lengths committed where a row
  emitted, a tree's accepted nodes moved into place) and prompt buffers for captured prompt chunks (`prompts:kv_cache`);
  its unused pages lent to a device expert cache (`spare`, `lendable()`, `on_reserve()`); `kv_bytes_per_token` in its
  facts. Attention reads the pages in place: FlashInfer's paged decode and prefill over 16-bit and FP8 pages when it is
  installed (the KV split pinned, its plans made once a step and declared with their workspaces in memory()), then the
  registry kernel crit-labs/paged-kv-attention's split-KV reader for decode and verify windows of every page type
  (partitions walking each row's length on the device, one CUDA graph for every length; one partition count and piece
  size per pool and tiles every GPU multiplies with mma.sync, one summation order whatever the tile, so a column's bits
  are the same alone, in a verify window, in a step and in a round) and its prompt reader of fp8:row and INT8 pages,
  then torch (gathered SDPA, or a capture-safe masked softmax). Rows of up to the pool's `window` tokens (16) read as
  window rows, and `round_rows` (8) bounds the round slots. It takes a RoundBatch capture-safely (`rounds:attention`),
  tree masks (`parents`, `attention:tree`), block-sparse lists (`blocks`, `block_size`, `attention:blocks`) and returns
  log-sum-exps (`lse=True`, `attention:lse`); captured prompt chunks read the staged prompt table on the device
  (`prompts:attention`). softmax-attention's options `decode` (auto or split), `partitions`, `piece`, `flashinfer` (auto
  or off) and `int8_qk` (prompt queries quantized to INT8 on integer tensor cores); partitioned-attention serves on CUDA
  and the CPU with `merge(outs, lses)` for partials from other devices. On an RTX 5090 one decode row's attention (56
  query and 8 KV heads of 128) at 32K context went from 0.28 to 0.10 ms per layer on BF16 pages.
- Prompt reads for honesty: softmax-attention and partitioned-attention take `prompt_reader`: `prefill` (the default)
  reads prompt rows through a prompt reader (FlashInfer's prefill or the registry's), the fastest, another reduction
  than decode rows', so a scored token's log-probability may differ from the served one's by more than 0.001 nats;
  `decode` reads a step's prompt rows and captured prompt chunks in the decode's reduction (window rows of up to 64
  columns through the registry's split-KV reader with the pool's partitions and piece, each program folding its
  partitions as the merge does, or its torch fallback over the page tables), so a scored token equals the same token
  served, bit for bit, over every page type: what a `preserve` engine's exact honesty check needs. Facts
  `prompt_reads` and `served_vs_scored` say which. Over FP8 and INT8 pages a prompt row of more than the pool's
  `window` tokens attends its own chunk's keys and values as the projections gave them (16-bit) and the earlier
  positions from the pages: attention-state keeps the chunk in a fresh buffer it makes when it is made (`fresh_mib`,
  16 MiB, declared in memory(); 0 reads the chunk from the pages, as `preserve`'s exact honesty needs), the attention
  plugins read it (the registry's `prefill(..., fresh=)`, else torch), and facts `prompt_kv` and
  `prompt_kv_fresh_tokens` say which keys and values a prompt row reads. The chunk is still written to the pages at
  their precision, and window rows, rounds, block lists and draft trees read the pages. No model change.
- Sampling inside captured rounds: token-sampling's `rounds:sampler` (`reserve`, `stage`, `laws`, `draft_device`: a
  draft per row from its processed law with Core's `draft` draw, the law kept sparse; `predictions_device`: greedy
  argmax, or rejection sampling with Core's accept, residual and sample draws, exact against `predictions()`;
  `logprobs_device`: the emitted tokens' Core TokenLogprob), all capture-safe, through its sampling law (sampling.cu,
  built at load with a host binding) or the torch law with `rounds: torch`; its facts name which (`round_sampling`).
  token-sampling's `staged(rows, device)` keeps a step's rules in device buffers and draws Core's Philox on the
  device. token-constraints serves on CUDA and the CPU: it masks each constrained row before token-sampling's device
  law (`staged(rows, device, vocabulary)` with the masks in a fixed buffer), and provides `rounds:sampler` with each
  slot's mask staged beside its rules (a constrained slot's drafts cut: one token a round).
- Drafting inside captured rounds: chain-verifier gives `verify:device` (each row's emitted tokens and counts, stops,
  allowance and held rows on the device); greedy-head-drafter, mtp-decode, hidden-state-drafters,
  draft-vocabulary-subset and draft-trees give `draft:device`: the chain over the model's device draft head
  (`draft:head-device`: argmax, or the sampler's draft draws in sampled rounds), copy drafts from each round slot's
  token history ring on the device (mtp-decode; copies verify by the target's own draw, `override` puts long copies
  over the head's chain), drafts mapped by d2t with the confidence cut learned and applied on the device, subset drafts
  over `head.rows` with the subset chosen again on the device, and draft-trees' expansion, kept tree and accept walk on
  the device (`verify:device` too; its rounds need the attention slot's `attention:tree`). Every commit is idempotent
  from the round's start, so a truncation commits again. tests/test_device_drafting.py checks the device forms against
  their host forms and whole rounds through Core's round driver against a host-form loop.
- Recurrence on the device: the recurrence device form gains `convolve(layer, x, weight, bias, pool, batch, activation)`
  (the causal depthwise convolution before a gated-delta layer, its window kept in the state pool), and its pool takes
  `sequences`, `widest`, `conv=(channels, kernel)` or `(channels, kernel, dilation)` (the window the last (kernel - 1) *
  dilation inputs), `key_heads` (grouped queries and keys, no repeat) and `rows` (round slots); `accept` keeps any
  prefix of a row of up to widest + 1 tokens. sequential-recurrence and recurrent-state-recovery serve captured rounds
  (`rounds:recurrence`), the accepted prefix of a verify window kept on the device by the `commit` option (`deferred`,
  the default; `checkpoints`; `recompute`), and captured prompt chunks (`prompts:recurrence`). Decode and verify rows of
  every sequence run in one launch of crit-labs/recurrent-scans' pool kernels (FP32 state, keys and queries normalized
  in the scan, each row's bits its own), a capture-safe torch formula without it; prompt rows of up to `exact_rows` tokens
  (stated: 512 by default, 0 for none) run the column scan, longer ones fla's chunked kernel (sequential-recurrence) or
  recurrent-scans' two-pass prefill (recurrent-state-recovery); the facts report each pool's value and where it came
  from (`recurrence_prompts`). A crossover measured at load chose between the two by timing: under a loaded CPU it
  fell between a prefix's last chunk and the whole prompt's, which then ran different arithmetic (a 27B serve check's
  causal comparison 0.17 apart); a timing now picks only between paths with the same bits.
- Experts larger than the device: device-expert-cache serves on CUDA (`experts:cache`, `cache(...)` -> `ExpertCache`):
  the directory on the device, one capture-safe plan launch per routed layer over a step's routes (duplicates and -1
  allowed), clock admission or ranked swaps (on the device inside captured rounds), an admission quota that sends misses
  to the CPU tier as slot -2 (quota tables by token bucket), fills by SM loads from a registered host store; prompt
  chunks stream through a ring of slots staged by the copy engine a layer ahead and never admit; a KV pool's unused
  pages are slots too, at any bundle size; per-row reads and hits through the round driver's `count_round`, and a
  `telemetry()` that never waits for the device. With the expert_source plugin's `csf_scales` option the store keeps
  NVFP4 scale components CSF-compressed (encoded from the slot-layout bundle or taken from the checkpoint) and the cache
  decodes them into the slot as it fills it; it refuses a configuration that could lose a route. expert-tensors and
  ssd-expert-materialization give bundle stores (`experts:store`, `open_store`): bundles in registered host memory with
  a device view, or a file tier read by the native row reader behind a host tier, a store file bound to its checkpoint's
  identity. device-placement's `off_device` places a model larger than the device (routed experts' bundles and lookup
  tables in host memory in preference order, then in files) and `slots()` sizes the cache.
- The CPU expert tier: `host-expert-tier` serves the experts slot on the CPU (`devices: ["cpu"]`): stored NVFP4 /
  INT4 experts packed as bundles in the experts slot's layout, every route in host-expert-arithmetic's declared integer
  block order (W4A4 / W4A8), summed in route order by FMAs (`experts:bundles`). `ExpertPool` (one pinned worker per
  physical core but the caller's, epoch-tagged lock-free claims) spins as long as the recent gaps between batches
  (`spin_us` to `spin_max_us`, default 3 ms) and then sleeps, and `HostExpertTier` computes a layer's CPU routes: each
  expert's gate/up rows once for all of its tokens, the declared SwiGLU and intermediate quantizer, the down rows,
  unweighted per-route rows; its mailbox payload is FP32 rows (`input_layout`, `serve_inputs`, `crit_tier_serve`). Any
  pool size, ISA or store layout gives the same bits; without a compiler the same order runs in torch.
  heterogeneous-expert-dispatch serves on CUDA: routes the cache's slot table marks -2 go to the CPU tier, posted and
  joined inside the stream through host-work-mailbox (capturable), the rest to the device part, every token summed in
  route order, so any split gives the all-device bits; `quota_table(tokens)` measures the link/CPU split through
  captured layers of posts, link copies and joins for an expert cache's admission. Plan rules 7 check 19: where recon's
  placement prices CPU-computed misses below link copies, candidate 1 starts with the tier or names the measurement
  that rejects it; recon's starting design says so, with the residency profile.
- Integer block rows in host-expert-arithmetic (`host_int.h`): 4-bit expert weights against activations quantized
  per block (E2M1 x 2 or INT8), exact block sums on any processor and one declared FP32 step, so CPU rows
  (AVX-512 VNNI, AVX2, scalar), device rows (`host_int_device.cuh`) and the `int-block-experts` registry kernel's
  decode rows and prompt GEMM give the same bits at integer speed. Declared quantizers, exp, SiLU and SwiGLU keep an
  expert's intermediate identical on either processor, and an interleaved host layout lets a vector lane hold one
  block. The FP32 orders stayed exact but read 14-16 GB/s on a 6-core desktop CPU for a verify window, below its
  PCIe link; the integer rows read 26-42 GB/s there. Fixtures check every path, the device function and the
  quantizers bit for bit (`crit_host_int`, `crit_host_int_cuda`). host-expert-arithmetic serves the projections slot on
  the CPU: stored NVFP4 / INT4 matrices weight only (the 16-lane FP32 order), W4A4 or W4A8 (the integer block order),
  BF16 a row at a time, every row's bits those of the row alone at any count; `declared.py` is the same arithmetic in
  torch on any device, the fallback and an oracle.
- Experts and routing on the device: experts-reference runs batched in fixed chunks, its products in FP64, capture-safe
  (`rounds:experts`, `prompts:experts`), and refuses stored formats; topk-router's device form is capture-safe and
  NaN-safe (NaN ranks below every number, -inf included; a row of NaN logits picks valid distinct experts with weights
  0), selects through the registry kernel crit-labs/topk-routing where it loads and the rule has no group limit
  (option `kernel`), and provides `rounds:router` and `prompts:router`. expert-streamed-prefill's code/streamed.py
  measures the grouped kernel's rate curve by rows per active expert and runs a prompt's routed layer in chunks at its
  knee over a staging source's store and slot table; no plugin runs it yet.
- Embeddings from host memory or disk: lookup-table-rows serves on CUDA (`embeddings:host`, `rounds:embeddings`,
  `prompts:embeddings`): tables in registered host memory gathered over the link, or in files read by a native pool of
  positioned reads with background prefetch into a hot-row cache, the gather a host function on the stream so a CUDA
  graph captures it, stored NVFP4 rows decoded on the device; memory-table-rows gets `rounds:embeddings`,
  `prompts:embeddings` and a `host` option (its table in registered host memory), over lookup-table-rows' code. Both
  plugins' `store()` (and lookup-table-rows' `open()`) return table handles, so one plugin keeps several tables, and
  take stored NVFP4 rows. A host table's captured gather runs at the link (26 GB/s at 512 ids on
  one RTX 5090).
- Background ranked residency in device-expert-cache: plans count routes on the device, `RankedPolicy` (C++ and
  Python) ranks decayed counts of every layer together, and `ExpertCache::begin_swaps` / `commit_swaps` swap experts
  on a side stream between fixed window boundaries. With misses on the CPU tier the cache adapts off the step's
  critical path, and one request history always gives the same placement.
- Core fixed buffer plans (`plan_buffers`, `BufferArena` in `crit/device_memory.h`; `crit_core.buffers` for Python):
  every device buffer named and allocated once at load as one charged allocation, a per-step scratch arena, and the
  rest of the budget after a reserve as equal slots. An engine needs no framework allocator's headroom, and bytes a
  lower context frees become slots at the next plan.
- `RowPrefetch` in lookup-table-rows (Python and C++): a draft token's table rows start the moment the token exists,
  queued tokens are gathered in one batch, and a wait blocks only for rows still in flight, with wait telemetry for
  the round account. device-speculative-round says where to start them in a round.
- `crit run ... serve --check --data DATA` adds crit-eval's gate pre-check to the serving checks: honesty of short
  replies and of one long reply, the request history around them, device memory and the speed cases at the lower
  context (the served engine reloaded at it in the same resident host). Its failures fail the functional check.
  AGENTS.md asks for it before every Eval round.
- Plugins report engine facts of their own work (`facts()`: the projections plugin's `prompt_activations` and
  `row_invariance`, which crit-eval prices prefill by and model plugins split their rows by; attention-state's
  `kv_bytes_per_token`, the device bytes one more token of context takes across the pools it made, which crit serve's
  engine facts and the eval adapter's info carry, so crit-eval sizes the KV cache an engine actually keeps, FP8 pages
  beside a BF16 draft layer for one, instead of the model's 16-bit K and V), device-placement keeps `reserve_mib` free
  beside its plan, and Core warns once when a WDDM device has (almost) no free memory left, where the driver pages
  instead of failing. eager-graphs and recorded-selection are host policy plugins an engine can serve, `crit plugins add
  --update` replaces an engine's copy of a library plugin with the current one, and Core's eval-v1 adapter runs
  crit-eval's concurrency cohort in process, so `crit eval run --package` judges the concurrency gate.
- Plan rules 7, the default of new runs (docs/RUN_RECORDS.md, Plan rules 7). A `limit` claim rests on the
  parent's round account (`round`): every term of 10% of the round or more that is not named in `cause_terms`
  runs at half its floor's rate or better, the device's idle time counts as a term, and the parts add up to the
  measured round. A rejection names the term that bound it (`bound_by`) and is reopened once a later round
  account shows that term 25% smaller. One run stopped on "decode at its bandwidth floor but for the expert
  copies" while its dense matrices ran at 27% of their floor and the device idled 8 of 60 ms, and kept a CPU expert
  rejection measured while link copies bound the round.
- The norms slot gains `softplus_decay` (a gated delta layer's log-decay -rate * softplus(x + bias) and its sigmoid
  beta gate from the projection's rows: one launch of fused-norms' kernel where a model's torch code launched six,
  torch's values on CUDA) and the fact `norms:fused`: `rms_norm_rope_pair`, `rope_pair` and `stream_add_norm`, each
  output with the separate calls' bits. Both norms plugins provide it; the test model plugins use the pairs and
  softplus_decay. One Qwen3.8-27B target pass's norms-slot work and decay glue in a CUDA graph take 768 / 793 / 819 us
  at 4 / 9 / 16 token rows before and 496 / 504 / 515 us after (577 launches become 321).
- attention-state compiles its step window launches at load: a step's packed window rows launch at their query
  tile's width over page tables a multiple of 16 pages wide, and the first eager call over a pool and query
  geometry launches each such variant once, so a request at a new prompt length or round width compiles nothing.
- The round driver runs one eager round of every (rows, width) it has no graph of at load, every row held
  (`SpeculativeRounds.warm`), and Core serves one request of each kind its first request is not (sampled,
  log-probabilities, plain, a replay, a score, two together; `crit_core.engine.warm_requests`, recorded as the load
  telemetry's `warmed`): a model's own round launches and every request path compile at load, none on a request
  (docs/PLUGINS.md, Compiling at load).
- quantized-projections: gemv auto (the new default) picks each matrix's decode kernel at load: FP8 16-bit decode
  rows on decode-gemv's dot GEMV, W4A4 decode rows on mma-rows' block-scaled FP4 MMA (sm_120 family), the tile
  elsewhere. FP8 W8A16 and W4A4 decode-row bits differ from gemv mma's (each still row-invariant at 1-16 rows), and
  facts decode_kernels name each role's kernel. New option decode_formats sets the decode activations per weight
  format (e.g. nvfp4=prompt,fp8=bf16). On a Qwen3.8-27B decode pass at 9 rows: 14.87 ms with every matrix on the
  tile, 14.37 with gemv auto, about 13.4 computed for nvfp4=prompt,fp8=bf16.
- Prompt graphs' offers name the arithmetic each computes (`bits`: the eager path, or padded to a bucket), and
  recorded-selection chooses by measured time only among offers of the same bits, the arithmetic fixed by its
  options (`allow_padding`: the smallest padded bucket): which bucket a chunk ran no longer depended on load-time
  timings, so two loads of an engine give the same prompt rows.
- quantized-projections: `prompt_gemm: tile` (with `gemv: mma`) multiplies every prompt row on crit-labs/mma-rows'
  prompt tile with the decode rows' activations and plan: each prompt row's output has the bits it has as a decode
  row, so a scored token's log-probability is the served one's (preserve mode's honesty) at GEMM-like speed (155-176
  TFLOPS on NVFP4 from 1,024 rows; a Nanbeige4.3-33B layer at 4,096 rows 28.2 ms against 59.5 through 64-row decode
  calls). Facts: `row_invariance.prompt_exact`, prompt arithmetic and activations from the decode rows'.
  PluginTestCase.assertKeepsRows checks a prompt_exact claim. mma-rows' plan() takes the splits of K that fill the
  decode tile's resident CTAs (its occupancy), so no plan overruns a wave (a Nanbeige gate_up at one row 125.4 ->
  114.8 us). `gemv mma` keeps every matrix on the tile; the FP4 MMA and the FP8 dot GEMV serve under `gemv auto`.
- Check repairs, in plan rules 4 and later beside gate repairs (docs/RUN_RECORDS.md, Plan rules 4): a requirement
  lever that serves `check:NAME` repairs a check crit run requires (arithmetic, functional, resources) whose newest
  record of the parent failed, and the plan is exempt from the speed checks a gate repair skips (checks 5 to 9, the
  +10% floor on p1, a critique round's weakest terms, the reach). crit cites the failed records in
  `inputs.objective.check_repairs`, and refuses the repair when the parent has no such check or its newest one passed.
  One run's candidate passed every gate of its Eval but failed the resources check (free device memory at the
  required context below the run's headroom), so select refused it, and the plan of a child that only repaired it
  was refused both as a gate repair (resources is no gate of Eval's status) and as a speed plan. `gate:resources` is
  now refused with a pointer to `check:resources`.
- `pip install crit` ships the plugin library: the crit wheel carries Inference (inference/) as package data,
  `crit/library/inference`, every technique's and base plugin's folder whole (without Inference's native
  `CMakeLists.txt` and `examples/`, which build against a checkout's core/), so `crit plugins list|check|test|add` and
  `crit inference` act on it from an installed crit as from a checkout, and engines' plugins are built from either.
  crit finds the library through one place, crit.inference.LIBRARY: the checkout's inference/ beside the cli/ crit
  runs from, else the wheel's copy; crit mcp serves inference/README.md from it. A crit without its library says its
  install is broken. The wheel grows from 0.40 to 1.65 MB. `crit inference` no longer has to run from the repository,
  and the wheel's packages are listed with their folders in pyproject.toml. tests/test_wheel_install.py builds the
  wheel from a snapshot of HEAD, installs it into a fresh environment and lists, checks, adds and tests library
  plugins from it; CI's packaging step lists and checks the installed library.

### Changed

- crit needs crit-eval 0.6.0 (`crit-eval==0.6.0`, and `crit-eval[reference]==0.6.0` in the `reference` extra),
  whose protocol is crit-eval-v0.6.0.
- `crit run measure` refuses, before it reserves anything, a candidate whose recorded checks would leave its
  benchmarks exploratory (docs/RUN_RECORDS.md, Measured benchmarks): a case with no check of a required gate, or
  whose latest passed one is only reported where the run requires artifact_verified evidence (record the checks
  first: `crit run ... check --candidate ID --from-eval`), and a required check that did not pass or an earlier
  benchmark below the run's memory headroom (no later record undoes them: the fix is a new candidate). `--exploratory`
  measures anyway, for telemetry or memory, and the command warns as before. Before, `measure` only warned after
  measuring: one run measured a candidate before its checks were recorded, sealed it, and the candidate could never
  be selected, since every benchmark of a case must qualify and a sealed candidate takes no new one. The docs' run
  order puts `measure` after `eval` and `check --from-eval`.

- grouped-experts: a long prompt chunk combines each part's routes into its tokens' rows as it goes (no
  [routes][hidden] buffer of every route: 840 MB at 8,192 tokens, top 10, hidden 2,560), and a prompt-sized call of
  fewer tokens than a kept prompt workspace runs in that workspace's buffers, so a chunk and the draft layer's pass
  over it keep one workspace (about 470 MB less at 4,096 tokens); the same bits.

- softmax-attention: `decode` defaults to `split`. A step's window rows are read through the split-KV reader as
  captured rounds read them, so plain decoding and log-probability and replay requests give the bits a drafted
  round gives. Before, with FlashInfer installed, a step read them through FlashInfer's decode, and greedy plain
  decoding could differ from drafted decoding (a serve check's plain_equals_drafted failed; an Eval's exactness
  found one of two replies apart). `flashinfer` (formerly `auto`) keeps FlashInfer's decode for engines
  that decode in steps only.

- New runs climb the candidate ladder ([docs/RUN_RECORDS.md](docs/RUN_RECORDS.md#the-candidate-ladder),
  `"budget_counts": "ladder"`, default `candidate_budget` 5): a candidate's number in the run says how far its
  round goes. Candidate 1 builds, passes its plugins' tests and `crit run ... serve --check` with no Eval
  (`crit run ... eval` refuses it), candidate 2 takes every plugin that fits the checkpoint and the GPU with at
  most a small Eval (Eval's stage 1), candidates 3 and 4 at most a medium one (up to stage 2), and candidate 5
  on, with the segment's last budgeted candidate, the full certified Eval, which `crit run ... eval` runs
  without `--certify`. A candidate that skips its small or medium Eval is sealed `built` after a passed serve
  check, and the next plan starts from it with no Eval to freeze. Plans record the rung in
  `inputs.objective.eval`; candidate 2 is the `plugin_round`, crit rounds run to candidate 5 and critique rounds
  start at 6. Runs created before keep their budget counts.
- Every engine is Core plus plugins, and plugins are the only way crit runs an engine
  ([docs/PLUGINS.md](docs/PLUGINS.md#every-engine-is-plugins)). Core knows no model and no device; a package is
  `plugins/` and the `configuration.json` whose `plugins` selection fills the slots every engine fills: two new
  slots, `model` (the architecture, whose object is Core's batched step model) and `placement` (where tensors live,
  base `device-placement`), beside `projections`, `attention`, `kv_cache` and `sampler`, and every slot a plugin
  `requires`. A manifest names the `devices` a plugin runs on; a plugin without them is a host reference, an oracle no
  engine serves. Core's loader (`crit_core.engine.load`) reads the checkpoint's architecture as facts
  (`model_type:...`), assembles the selection as a whole engine on its device, refuses a model plugin that cannot
  batch (`max_concurrency` of 2 or more) or falls short of what every engine serves (log-probabilities with 20
  alternatives, fixed-text scores, Core's sampling fields, plain decoding when it drafts), and runs it on Core's
  concurrent runtime. crit serve runs packages only through Core's resident adapter (`crit_core.engine.resident`) and
  derives what each serves (`crit_core.engine.capabilities`), serving a run's candidates and the engines crit pull
  installed; crit-eval measures them through Core's eval-v1 adapter (`crit eval run --package DIR`); `crit run
  candidate` and `package` check every source and package without running its code and refuse anything else; a
  package recipe is its `build` only (no `launch`, no `capabilities`), and crit pull installs plugin packages only. A
  package manifest of an older run keeps its launch entry as history and does not serve. A selection's `base` for a
  required slot defaults to the plugin in use.
- Core's engine C ABI is version 5 (was 2): the facts callback is always read, so a model's request facts reach every
  result; native rounds carry log-probabilities and replays; and Core chooses each native round's form. A
  `crit_core_steps` of another version is refused: rebuild it against this crit's headers.
- Core's loader takes its counting proxies off the hot path once its checks pass: every proxy a plugin keeps is
  replaced by its target (`Assembly.unobserve`: attributes, containers, closures, partials, torch modules), and one
  left where Core cannot reach passes calls through. Its check that every computing slot took part in the first
  request counts a replay of a captured decode round, native round or prompt chunk as the slot calls its capture
  made, so an engine whose first requests run on replays is checked as one that runs eagerly.
- Loading a package's plugins never writes bytecode into it and never runs cached bytecode beside it, so a sealed
  package keeps its bytes, and runs them, even when a script without `-B` loads it.
- Core's Python step callbacks cost a few microseconds: Core's worker keeps its Python thread state, a request's
  waiter wakes only when it listens to tokens (a null callback sleeps to the end), a listener gets a step's or a
  round's tokens in one call, and rows cross as packed bytes. A no-op model steps one request in about 9 us, from
  about 130 us.
- The quality modes are `preserve` and `speed` ([docs/RUN_RECORDS.md](docs/RUN_RECORDS.md#quality-modes)), defined
  once everywhere (AGENTS.md, docs, the brief, plan rules 6, recon's prefill plan, technique READMEs,
  settings.example.toml): drift from the selected checkpoint traded for speed, never its storage format. `speed` is the
  default of new runs (the request, or settings.toml's `quality_mode`); a run in it defaults to `weights_policy:
  transformable`.
- Draft-only weights take any encoding whatever the run's `weights_policy`; the target weights alone decide a
  candidate's `weights`, and the drafter's encoding goes in the configuration.
- The commands a `preserve` run gives the agent to measure quality itself pass `--quality-mode preserve`; transformed
  weights are measured with `crit eval run --url` under the run's quality terms, and `crit models eval` records them
  (`summary.requirements`) and refuses a result that states no mode.
- `crit bench` measures in the card's result's mode (every crit-hub result states one), `speed` without a result;
  with `--quality-mode` naming another mode it leaves the card's figures out of the comparison, and the comparison
  names the card's mode.
- A run started from a crit-hub engine records the quality mode of each of its results
  (`origin_engine.evidence[].quality_mode`, required), the brief states it, and flags results judged in another mode
  than the run's: their numbers do not compare with the run's Evals.
- Structured output and forced tool calls are Core's, for every engine: Core's resident adapter declares
  `chat-constraints-v1`, `chat-constraints-grammar-v1` and `chat-tool-choice-v1` (where the template's call syntax is
  one crit writes) whenever Core can match the tokenizer's bytes. A constrained request decodes plainly; before each
  step Core's matcher puts the tokens its row may take on `StepRow.allowed` (`crit_plugins.Allowed`), the model
  plugin hands them to its sampler, which masks every other token (token-sampling on the device too), and Core
  checks the token the row produced, failing only that request on a token outside its mask. The loader refuses a
  model plugin whose rows do not keep to Core's masks. `crit_core_steps` exports Core's matchers
  (`crit_vocabulary_open`, `crit_language_open`, `crit_matcher_*`), and `ResidentChat.generate_tokens` takes one
  `constraint` (a response_format, or the grammar of forced calls) in place of `grammar`.
- Images through Core's adapter: a model plugin that provides `inputs:images` and implements `prepare(ids, images,
  cancelled)` serves `chat-images-v1`. Core decodes the request's base64 `data:` URLs (it fetches no URL), the
  model plugin lays the images out in the rendered prompt on the request's thread, and its inputs reach `open()`
  as `StepRequest.inputs` through the request's host attachment.
- Every plugin that runs on a device speaks its slot's device form, the CPU included, and assembling an engine
  refuses one without it as it is created, so host-form Python can no longer be served by declaring `cpu`. The toy
  engine of crit's tests runs on torch's CPU tensors, its KV in its kv_cache plugin's pool; crit's CI installs
  torch 2.14.1 for the CPU. `PluginTestCase.device()` runs a plugin's tests on a device its manifest names and skips
  them on a machine without one.
- Plugins fit the hardware and the checkpoint, not only the device kind: Core's loader gives their needs the
  checkpoint's stored formats (`stored:plain`, `stored:nvfp4`, `stored:fp8`, ...) and the device's compute capability
  and tensor-core formats (`cuda:sm_120`, `cuda:bf16`, `cuda:fp8`, `cuda:fp4`), from a crit eval probe device file
  (a run's, for `crit run candidate` and `package`) or `nvidia-smi`. A manifest's `needs_when` adds needs where a fact
  holds: quantized-projections needs `cuda:fp4` for an NVFP4 checkpoint and `cuda:fp8` for an FP8 one, and
  projections-dense conflicts with both. `crit plugins list --model DIR --device D [--device-file F] [--all]` lists,
  per slot, the plugins that fit and why the others do not, and the required slots nothing fits.
- The slots' device forms grew with the library: projections' `load(tensors, format, role=None, encode=None)` and
  `matmul(x, matrix, out_dtype=None)`, each row's bits its own in a call of up to `invariant_rows(plugin)` rows (the
  facts' `row_invariance` `rows`, at least `DECODE_ROWS`, 8; `assertImplements` checks as many rows as the facts
  promise, up to 32); the KV pool's `pool(..., dtype, *, value_dim=None)`; attention's `blocks`, `block_size`,
  `parents` and `lse`; the experts device form `forward(layer, x, ids, weights)` over the plugin's loaded experts
  (`load` takes bfloat16 matrices or stored tensors in a named format), a store going through `experts:bundles`'
  `forward_bundles(layer, x, ids, weights, store, slots, format)` with its format required; the router's `select` and
  the embeddings' `store`/`lookup`. greedy-head-drafter and chain-verifier serve as host policy in the drafter and
  verifier slots and on the device in captured rounds. dense-layer-offload is a placement plugin: its planner keeps
  whole layers on the device while they fit and the rest in host memory, choosing among resident, host and streamed
  plans with measured rates given as options.
- A placement plan is the placement, not advice: on CUDA Core's loader refuses an engine that holds more device
  memory after its first requests than its placement plans and its plugins' measured declarations put on the device
  (5% and 64 MiB over them for small tensors), so a model plugin plans every tensor it keeps there, its KV included.
- Scheduling is Core's: the scheduler slot is gone, with its plugins whole-prompt-scheduler and session-scheduling,
  since Core's concurrent runtime schedules every engine's steps and never asked them. session-scheduling and
  concurrent-request-execution stay Inference entries of what that runtime does. `graphs` is documented as a slot the
  model plugin calls. host-work-mailbox lends its code to heterogeneous-expert-dispatch instead of being an engine
  service.
- A hybrid mixture-of-experts engine runs through the whole path on a GPU (tests/test_hybrid_engine.py): a
  Qwen3-Next-layout model plugin over the library's sequential-recurrence, topk-router, experts-reference (then
  grouped-experts), expert-tensors (host policy in the expert_source slot) and fused-norms beside the attention, KV,
  projection and sampling plugins, its scores and greedy choices against transformers and its batched requests against
  each alone; its convolution window lives in the recurrence pool and it passes grouped keys.
- `crit run check --from-eval` judges the functional gate by Eval's concurrency gate too: every engine batches, so
  its batched replies must equal its serial ones and a level's requests must run together.
- New runs spend the candidate budget on candidates that go through Eval (`"budget_counts": "evaluated"` in the run
  spec): a candidate that fails before its first Eval is a draft, sealed with what stopped it and fixed as a new
  candidate from the same plan at no cost and with the same objective. Once the budget is used, an Eval of a
  candidate not yet counted is refused. Older runs keep counting every candidate.
- Plan rules 7: an unknown noise takes its own family's known noise, not the largest of any KPI (a 151% first-token
  spread once became decode's bar), and a plan freezes the parent's newest Eval attempt that measured decode and
  prefill, so a quality-only rerun no longer moves the attempt its gaps and forecast refer to.
- Techniques keep no kernel lists: each technique's kernels.json became its operations.json (per device operation what
  it computes, its CPU reference, how a kernel is verified against it, its invariance); `crit inference check` refuses
  kernel lists, provider names, kernel source paths and a leftover kernels.json, and tests/test_plugin_library.py holds
  a plugin's declared registry kernels to the ones its code loads and finds no kernel-builder project in crit.
- The kernel workbench keeps no kernel source in crit: `crit kernels new` starts `src/` as a kernel-builder project
  (`kernel-builder init`, or `--from DIR`), `kernel.py` calls its op `run_<format>`, and `--starter` takes crit-kernels'
  workbench-starter with `--from`.
- crit run's projection allows two processes' record times `CLOCK_SLACK` (2 s) of disagreement on its lower bounds
  (a candidate after its segment's start, a check or benchmark after its candidate, an attempt's end after its
  reservation); deadlines stay exact.
- crit's `reference` extra (crit Eval's reference, and with it `crit models` GPTQ) states the transformers range crit
  supports, `>=5.20,<6`. The host suite passes on transformers 5.20.0 and 5.20.2, and its modules that import
  transformers pass on 5.22.2 too. They also pass on 5.19.0 (5.18.0's LongCat-Flash router returns no logits), but
  crit Eval's reference needs 5.20. CI installs no transformers, so those fixtures skip there
  ([docs/CONTRIBUTING.md](docs/CONTRIBUTING.md#check-a-change)).
- crit serve's `/api/chat` dialect is the module `crit_api.api_chat`, named after the API it serves instead of after
  the server that defined that API, and no module answers to the old name: its `/api/version` constant is
  `COMPAT_VERSION` (still `0.12.0`), the server's `/api/tags` and `/api/ps` card is `Api.api_chat_card`, and the
  request handler's methods for the dialect are `api_chat_post` (`/api/chat` and `/api/generate`), `api_chat_show`,
  `api_chat_complete` and `api_chat_stream`.
- `crit_serve_requests_total` counts `/api/chat` and `/api/generate` requests under `dialect="api_chat"`, a label
  like the other dialects' instead of the name of the server that defined the API: queries, dashboards and alerts on
  the old label match nothing.
- `GET /` answers the plain text `crit is running` instead of the line the server that defined the `/api/chat` API
  answers there. Clients that probe `/` for its status still find crit serve; one that matches that line's text no
  longer does.
- The GGUF tokenizer conformance tests (tests/test_core_source.py) read the GGUF converter's checkout from
  `CRIT_GGUF_TOOLS`, as crit-eval's GGUF tests do. The variable named after the project that hosts the converter is
  not read any more: a test environment that sets only that one skips them.

### Removed

- Compute kernels in crit ([docs/PLUGINS.md](docs/PLUGINS.md#kernels-and-plugins)): recurrent-state-recovery's CUDA
  reference scans (verify with checkpoints, recompute, deferred verify) and their oracle fixture are crit-labs'
  recurrent-scans in crit-kernels, its commit kernels (`enqueue_select_accepted`, `enqueue_record_accepted`) staying
  in code/recurrent_cuda with their CTest; the device round's toy model computes its gated-delta layer with a fixture
  loop of its own. The kernel workbench's starter matmul and binding stub (kernels/crit_kernels/templates) left crit:
  the starter is crit-kernels' workbench-starter. Every kernel the library's plugins run is a provider's or a
  crit-labs registry kernel.
- The techniques' kernels.json and their kernel lists (`lib:` and `hub:` kernels with calls, formats, archs, shapes,
  `published` and `local`), crit-labs' kernels named `hub:crit-labs/NAME` among them.
- The quality mode name `as_quantized`: it, and a run record without a mode, are refused like any unknown mode (no
  alias, no reading as preserve).
- Inference entry batched-reference-engine: crit hosts no compute kernels, and its value was CUDA kernels only a GPU
  can check (paged attention's KV writes, prefill, decode and verify; the synthetic model's tensor-core GEMM and row
  kernels), while the library's attention-state plugin and crit-labs' paged-kv-attention cover paged attention through
  the registry. With them went the engine, its device pages and throughput tool, its paged attention's host half
  (`crit_paged_reference`, an FP64 oracle only those kernels were checked against) and reference.py's step rules,
  which Core's batched steps hold. dense-layer-offload's synthetic model runner and bench ran on those kernels and went
  too: its planner, streamer, executor, host GEMV and host block stay. `crit_dense_offload_tests` gained a
  `host_block` case (thread splits, batched against single tokens, the cache's written positions) and, on a CUDA
  device, a `stream` case (`run_layers` over device, streamed and host layers with a recording runner: step order,
  each layer reading its own bytes through the two-slot weight stream, positive measured rates).

### Fixed

- The plugin library on Windows: positioned file reads and writes go through crit_plugins.files (os.pread and
  os.pwrite where they exist, else a seek under a lock per descriptor), files are opened in binary mode, the CPU
  expert rows' build lock imports fcntl only where it builds, and host memory regions (libc's mmap and madvise) say
  they need Linux rather than failing inside ctypes. lookup-table-rows' facts say why the Python reader serves where
  the native one did not build.
- CI installs numpy beside torch. torch's CPU wheel does not require it, and without it the prompt batches' staging,
  lookup-table-rows, the expert stores and Core's torch-file reader fail ("Numpy is not available"):
  test_prompt_graphs, test_plugin_library and test_core_torch on Python 3.11 and 3.14.
- crit records: two reads of crit's clock never share a time. While a stepped-back wall clock is behind, each read
  is a microsecond after the last; before, they all took the last time read, so an install and its reinstall could
  carry the same installed_at.
- crit run replay: a file counts as written before the cutoff (or a plan draft as written before the parent's Eval
  ended) only when dated crit's clock slack (two seconds) earlier. File times follow the wall clock, which a
  virtual machine's steps back by up to a second, while crit's records are dated by a clock that never steps
  back, so a next plan's draft written after the Eval could be dated before its end and land in a replay.
- attention-state: block-sparse lists read in one KV partition count in every split-KV launch (PIECE_PARTITIONS: a
  prompt row's pieces, a step's window rows, a round's columns), so a decoded token with a block list has the bits
  of the same token in a prompt. Before, steps and rounds read lists in the device's partition count and parted
  from the prompt by a few thousandths of a nat past a few thousand positions (a serve check's
  decode_long comparison).
- crit_core: Core hands back what its warm requests leave in torch's cache at load (released_after_warm in the load
  telemetry): a long warm prompt's workspaces stayed reserved and Core read the device as full.
- expert-streamed-prefill: `Streamed` takes its chunk from a timed rate curve only for an experts plugin whose facts
  say a route keeps its bits whatever rows share its call (`experts` `row_invariance` `prompt_rows`, as
  grouped-experts' do), and refuses the curve otherwise: state the chunk (no curve, `max_tokens`) or use a
  row-invariant experts kernel. Before, it trusted any kernel, so a timing could choose between chunkings of other
  bits.
- sequential-recurrence: fla's chunked prompt path runs each of its six autotuned Triton kernels on one stated tile
  (`device_states.FLA_TILES`, held as the autotuner's only config), so fla times nothing and a prompt row past
  `exact_rows` gets the same bits on every run. Before, fla's autotune picked each tile by timing at a shape's first
  call and kept the pick in Triton's cache, and two kernels' tiles give other bits: on an RTX 5090 at the 27B's
  gated-delta shape, the L2 norm's 8-row blocks on 8 or 16 warps and 16-row blocks on 16 (outputs up to 2.6e-5 apart
  with FP32 rows and 2.4e-4 with BF16 ones, the final state up to 6.6e-5) and the output kernel's 64-key block (outputs
  7.5e-9 apart with FP32 rows, 1.2e-4 with BF16 ones).

- crit run's record times never step back within a process: a virtual machine's wall clock (WSL's) moving back by up
  to a second dated a check before its candidate ("Check outside segment", "Benchmark outside segment").
- The rejection rule takes the target's own draw for a draft its law gives no mass (`q(d) = 0`) or that has no law
  (a copied draft, a law of more than 64 kept tokens), on the device and in the host form (a column's law `None`).
- `check --from-eval` refuses a result judged in another quality mode than the run's, as it refuses another budget.
- Plan rules 3 and 6 take a quality rejection only from a result judged under the run's quality mode and budget
  (every resume's amendment applied): a share of preserve's budget let BF16 past check 16 in a `speed` run.
- `crit run ... eval` no longer fetches the reference package again whenever the data holds no `allowance.json`.
- Plan rules 6's refusal and recon's prefill plan name `crit eval run --package P`, not `--adapter A`.
- The brief's speed paragraph limits re-encoded dense weights to what the run's `weights_policy` allows.
- Expert routing's transformers fixtures (tests/test_expert_routing.py) build transformers 5.20's router classes from
  minimal configs with an identity gate. DeepSeek-V2's, DeepSeek-V3's and GLM-4.5's MoE blocks no longer route
  (`route_tokens_to_experts` moved into DeepseekV2TopkRouter, DeepseekV3TopkRouter and Glm4MoeTopkRouter), and
  Ernie 4.5's and LongCat-Flash's routers read `num_experts_per_tok` and the common expert-count names and return
  logits, weights and IDs like the others, so the fixtures failed on transformers 5.20.2. The routing they check is unchanged: the moved code computes what the
  old did, and the host reference still matches it.

## [0.16.1] - 2026-11-02

### Fixed

- A release reaches PyPI only once main's CI passed on the tagged commit: the release workflow waits for that run
  (job `main-ci`) and stops on a failed one, before the `pypi` environment asks for approval.
- crit needs crit-eval 0.5.3 (`crit-eval==0.5.3`): a resumed Eval round replays a served engine's request deadline
  as the engine observed each request, so it no longer leaves the journal early when the engine's own timings and
  the token arrivals differ.

## [0.16.0] - 2026-11-02

### Overview

crit 0.16.0 makes quality mode `as_quantized` the mode for speed end to end: its default budget is crit-eval's wider
one, output differences that part at near-ties pass, and kernels may be declared non-repeatable. The device expert cache
becomes a ready component with its host store, residency corpus and provider grouped GEMM operands, and
`expert-streamed-prefill` reads prompts through experts kept in host memory. Core's prefix cache gains recurrent
snapshot placement for hybrid models, eviction notices with a host page tier, retention levels, reuse scopes,
partial-page reuse and a guard page for masked reads. Techniques name the provider kernels they run (`kernels.json`),
and expert arithmetic covers any block-scaled format on any CPU. crit 0.16.0 needs crit-eval 0.5.2.

### Added

- `kernels.json` beside a technique lists the provider kernels for each operation it runs on the device: the exact
  call, formats, GPU architectures, shapes and scale layout, the technique's CPU reference to check a kernel against
  on the target, and how. `crit inference check` validates it and `crit inference show` prints it. attention-state,
  quantized-projections, recurrent-state-recovery, partitioned-attention, token-sampling and expert-streamed-prefill
  have one. crit-labs' own kernels are named `hub:crit-labs/NAME` and kept in `hub-kernels/` until they are published.
- `device-expert-cache`'s `enqueue_operands` writes a provider grouped GEMM's pointer arrays from a plan's `Use`s on
  the device: each component inside the route's bundle, its activation and output rows and its routing weight, with
  bypassed misses optionally left null for the CPU. The expert GEMMs stay the provider's.
- Techniques name the provider for their arithmetic instead of leaving it unsaid: `recurrent-state-recovery` takes its
  scans from flash-linear-attention and says which commit strategy a final-state-only scan serves, `attention-state`
  points device attention at FlashInfer and FlashAttention over BF16 or FP8 pages (its INT8 G32 rows are a host
  codec), and `quantized-projections` names FlashInfer's FP4 quantization and cuBLASLt's block-scaled GEMM, with the
  tiled scale layout they share.
- `concurrent-request-execution` gets the device side of its metadata replica (`device_metadata.h`): a native
  `MetadataTable` with snapshots and deltas, and `DeviceReplica`, which stages only the changed rows in pinned
  double-buffered memory, has the engine copy them into the device table and uploads the gather order only when it
  changed.
- `ssd-expert-materialization` gets its native host tier and read queue (`host_tier.h`): `HostTier` keeps experts in
  pinned host slots the engine owns (least recently used out, the current route's experts protected, refusal instead
  of a half-held expert, fills published only once landed), and `ReadQueue` fills them from the file with several
  positioned readers, demand reads before read-ahead, one read per pending key.
- `lookup-table-rows` has a native form, `table_rows.h`: `RowGather` reads a step's rows with random-access advice and
  positioned reads by a persistent worker pool, reads repeated rows once, keeps an LRU of hot rows and writes them in
  request order into the caller's buffer, and `StagingRing` hands out pinned slots for the device copy, each reused only
  after its copy finished.
- `device-expert-cache` builds and pins its host store: `expert_store.py` packs a checkpoint's expert tensors (per expert,
  or stacked along the first axis) into one contiguous bundle per key with an index and the checkpoint's fingerprint,
  reused while the source is unchanged; `expert_store.h`'s `pin_store` registers Core's read-only file mapping as mapped
  pinned memory where the device allows it, else reads the file with several threads into a pinned `HostBlock`.
- `device-expert-cache` names a general corpus for residency profiles (`corpus.json`): public web text in seven
  languages and code in three to tune on, and held-out test splits, unseen languages and far rows to check on.
  `corpus.py` fetches it through the Hugging Face dataset viewer with the standard library and records each dataset's
  revision. A profile built from it, with its held-out check, meets AGENTS.md's rule for content-dependent choices.
- Quality mode `as_quantized` is the mode for speed, end to end. Its default budget is crit-eval's new one (PPL ratio
  1.4, mean KLD 0.4, p99 KLD 4.0), and output differences that part at near-ties pass Eval's repeatability gates. AGENTS.md
  and the agent brief now treat the stored format's activations, lower-precision KV, heads and dense weights as the
  starting arithmetic instead of levers to justify group by group, and no longer ask for a root cause of near-tie
  differences. `heterogeneous-expert-dispatch`, `host-expert-arithmetic`, Core's batch invariance rules and
  `recorded-selection` say which of their exactness rules are optional there: inexact CPU expert rows, residency-based
  prompt splits, row-count-dependent kernels and per-process tactic choices.
- `crit kernels new --repeatable no` declares a kernel that may round differently on a repeat (atomics, a split the
  workload chooses); crit-eval's workbench check then judges its repeats against the reference within the tolerance
  instead of bit for bit. For engines in quality mode `as_quantized`.
- `expert-streamed-prefill`, a new Inference entry for prefill through routed experts kept in host memory. Its
  schedule (`schedule.py`) picks the prompt chunk at the knee of the grouped projection's measured rate curve, so each
  non-resident expert crosses the link once per chunk instead of once per small chunk, and double-buffers each layer's
  expert copies behind the layer before it. `reference.py` is the FP64 routed SwiGLU block (stable route order,
  BF16 / FP8 row / NVFP4 / MXFP4 decoders) a grouped projection is checked against; the projection itself is a
  provider's, listed in its `kernels.json`.
- `device-expert-cache` is a ready component: `ExpertCache` (`runtime.h`) owns the directory and per-layer scratch
  and gives one capturable `plan_and_fetch` per routed layer, `preload`, `set_admission`, `prefetch` and telemetry
  with `expert_reads` / `expert_hits`. The planner gains admission limits (a quota per plan, a quota by miss count,
  second-touch); a miss over them takes no victim and its `Use` names the store bundle for the CPU or a mapped read
  (CUDA/HIP, Metal and the Python reference). `enqueue_preload` and `residency_profile.py` warm the cache from a
  profile built on a general corpus with a held-out check; `split.py` turns link, CPU and combined layer timings
  measured on the target into the quota table. The README gains a recipe in the order a build should follow. The
  CUDA fixture passes on an RTX 5090, runtime and operands included, and `pin_store` reads a registered store file back
  through its device address.
- `UploadShadow` (`crit/device_memory.h`) skips a host-to-device upload whose bytes the destination already holds,
  for per-step metadata between graph launches; backend-agnostic (the engine passes its copy). Host-checked only;
  the saving per step is not measured.
- `StateCacheConfig::partial_reuse`: after the last whole reused page, `attach` also reuses the matching leading rows
  of the page that last continued it, copied into the sequence's own page, so a prompt that leaves a cached one inside
  a page recomputes only from where it differs (6 of 8 rows instead of 4 on the 4-row fixture). For caches without
  recurrent components or window groups.
- Reuse scopes in Core's prefix cache: `StateCache::open(scope)` indexes a sequence's pages under its scope, and
  `attach` / `reusable` match only the same scope (checked on the root page, so it holds under hash collisions),
  so a server keeps one tenant from reusing, or timing, another's prompts. The host page tier keeps scopes too.
- Debugging masked KV reads: `StateCacheConfig::guard_page` reserves a page no sequence is given, for masked
  block-table entries, and `StateStorage::fresh_page` / `fresh_window_page` announce pages handed out for new rows;
  `StatePages::poison()` fills them with NaN, so a read of a row the request never wrote shows up instead of
  returning another request's values.
- `hidden-state-drafters` gets `ConfidenceCut`: the confidence threshold for truncating drafts, derived online from
  what verification tested (bonus predictions label the first cut draft), for a target acceptance, instead of a
  constant tuned on general text.
- `peak_pages(demands, page_tokens, verify_rows)` in Core's batched steps bounds the pages requests hold at once
  when early finishers free theirs before longer ones peak: 835 pages instead of the 1,504 their maxima reserve for
  16 staggered replies on the fixture. A projection for admission with memory preemption as the fallback, since a
  request that advances faster or starts late can exceed it.
- `crit_core.text.LoopDetector` finds a reply that repeats one span exactly (at least `min_repeats` periods and
  `min_tokens` tokens), and `ResidentChat.loop_stop` ends such a reply through the generator's `cancelled()` with
  finish "stop" instead of letting it run to its cap; off by default, never for grammar-constrained replies.
- Retention levels in Core's prefix cache (`StateCacheConfig::retention_levels`, `StateCache::retain`): a cached page
  at a lower level leaves before any at a higher one, so an engine keeps a shared system or tool prompt (and the
  window pages its end needs) through bursts of one-off prompts. On the state fixture an 8-row prefix that plain
  least-recently-released eviction loses stays reusable at level 1.
- Copied drafts override a neural drafter's where they are likely right: mtp-decode's prompt copy reports how long
  its match is (`propose_matched`, in Python and C++), and `device-speculative-round`'s `CopyDrafts` stages each
  row's copy and match length between rounds and replaces the drafter's tokens on the device where the match covers
  a threshold (`override_drafts`), with one-hot draft laws for sampled rows so rejection sampling stays exact, inside
  captured rounds. Before, prompt copy and a neural drafter could only run one at a time. Checked on the toy target:
  exact greedy output, every copied draft accepted on a row whose copies are right, the sampled law over 60,000 draws.
- `inference/attention-state` gets a host page tier, `HostPageTier`: pages the prefix cache evicts are copied to
  host slots from `StateStorage::evicted`, and a later prompt restores them in one `final_row` round, verified by
  chain hash and exact tokens, with window rows and recurrent state, instead of recomputing them. On a hybrid fixture
  (one recurrent component, one sliding window, ten device pages) a conversation evicted by two other prompts gets
  all 24 earlier rows back for its next turn and computes only the 4 new ones, every value equal to recomputation.
  Device copies are the engine's and are not covered by the tests.
- `StateStorage::evicted(EvictedPage)`: before the prefix cache hands out or frees an evicted page, Core names its
  chain hash, tokens, component snapshot slots and window pages, so an engine can copy it to a lower tier instead of
  losing it. Before, an evicted page was simply dropped.
- `crit_core.text.PromptTokenCache`: a prompt that extends a recent one, as a conversation's next turn does, is
  encoded from near the seam instead of from its start, and the splice is used only when the tokenizer agrees with
  the kept ids after the back-off point; otherwise the prompt is encoded in full. `load_model_text(prompt_cache=True)`
  routes `encode_chat` through it, and `ResidentChat` turns it on (`prompt_token_cache = False` turns it off).
  Over 4,320 simulated turns on nine tokenizers it matched a full encode every time, in a tenth to a sixth of the
  time. `Tokenizer.encode_offsets` returns ids with their character ranges.
- `StateCache::snapshot_rows(seq, prompt, rows, states)` says where a prompt round of a hybrid model should end so a
  recurrent snapshot lands where a later prompt needs one: at the deepest indexed page the prompt matches without a
  snapshot (where it leaves a cached prefix), else at the last page boundary inside a round that ends inside a page.
  Before, a chunked prefill whose last chunk ended inside a page left the prompt's last full pages without a snapshot,
  so a next turn diverging after the earlier prompt reused only to the previous chunk end (8 of 14 rows on the 4-row
  page fixture, 12 with the cut).
- Hardware-agnostic expert arithmetic and cache: `host-expert-arithmetic` computes any block-scaled format (4- or
  8-bit codes, 16 or 32 per scale, any element table; NVFP4, MXFP4, MXFP8 E4M3 and E5M2, MXINT8 and INT4 tables,
  E4M3 and E8M0 scale decoders) in its declared order, with a portable path every compiler vectorizes for its CPU
  (`Isa::portable`, what `best_isa()` returns without AVX2), one device header for CUDA and HIP, and the same
  functions in Metal. `device-expert-cache` builds for HIP through `crit/gpu_runtime.h` and has its kernels in
  Metal. Checked bitwise on x86, on aarch64 under qemu and on CUDA; the HIP and Metal sources are not run in the
  tests.

## [0.15.0] - 2026-11-02

### Overview

crit 0.15.0 keeps Core model- and hardware-agnostic: Core holds no GGUF config rule, chat template or rotary table of
its own. A lone GGUF's config rule and a model's fallback chat template come from Core's registry of model-family
knowledge (`crit_core.registry`), which crit-eval fills for the architectures its reference reads, and the rotary table
is attention-state's. In `as_quantized` runs, plan rules 6 start prompt GEMMs at the checkpoint's own activation format.
crit serve reports device expert cache counts for engines on the stock adapter, `min_tokens` reaches engines, and Core's
CUDA backend stages large copies in pinned pieces and leaves no error behind after a failed allocation. crit 0.15.0
needs crit-eval 0.5.2.

### Added

- Plan rules 6, the default of new runs: in an `as_quantized` run every build plan declares `prompt_activations`, no
  quantized format's prompt GEMMs start at BF16 unless a gaps entry `activation:stored` holds a crit-eval result
  that measured the stored format outside the quality noise, and a plan is refused when its parent served other
  prompt activations than its plan declared. Candidate 1 reads prompts at the checkpoint's own activation format.

### Changed

- crit pull, crit hub report and crit run create --from-engine read crit-hub cards with one reader (`crit/card.py`):
  the card as YAML, the lock as strict JSON (it was plain `json.loads` for `--from-engine`), and one source check,
  so crit pull now also requires the https repository and full commit that `--from-engine` required. Before, each
  read the card its own way.
- crit's generic record helpers (strict JSON, SHA-256 identities, file inventories, canonical paths, field checks)
  are `crit/records.py`, the coordinator lease is `crit/lease.py` and a process's birth identity is
  `crit/process/identity.py`: crit serve, process ownership, crit pull, Kernels and Models import them instead of
  `crit.run.records` and `crit.run.store`, so the process layer no longer depends on the run coordinator.
  `crit.run.records` and `crit.run.store` still name them. `RunStore.attempts` and `crit_core.machine.try_lock` and
  `unlock` are public, where crit serve and the lease used their private names; the lease's dead-owner check reads
  the process identity runtime ownership already used, so it now tells a PID the system cannot see from one that
  exited.
- Core's Python and C++ read two more tables from one source each, as they already did `config_keys.json`: the
  safetensors dtypes (`crit_core/model/safetensors_dtypes.json`, generated into the C++ reader's table in DType
  order and checked against the enum at compile time) and a step engine's settings (`crit_core/step_settings.json`,
  which `crit_core.native.StepEngine` splits a model's info by and Core's step engine now checks its `"engine"`
  object against, refusing a key that is not one of them). Before, each side kept its own copy.
- Core's CUDA backend drops `CudaTimer` and `CudaBackend::timer()`: no engine, Inference entry or crit tool used it,
  beside `CudaStageTimer` (device timings for the trace) and `CudaCategoryTimer` (device time by category), which
  stay. Every event the backend creates now orders work without timing.
- Core writes the JSON it hands an engine's caller with one writer (`core/src/runtime/json_text.h`):
  `crit_engine_info`, a generation's result and a step engine's request options. A number is the shortest text
  that reads back as the same double. Before, the engine library bridge had its own writer, which printed a
  generation's timing with six decimals while step engines' options took 17 significant digits.
- Strict JSON has one rule set: `crit_core.jsonfile` exposes its hooks (`unique_pairs`, `reject_constant`), and the
  schema compiler, the JSON-object matcher, the tool-call parser and crit serve's tool-call checks use them, while
  `crit run`'s records are read with `jsonfile.loads` itself (so a number that overflows a double is refused there
  too). Before, five modules each rolled their own duplicate-key and NaN rules. A script that imports `crit.run`
  or `crit.process` from a checkout now needs Core's Python package beside it, as an installed crit has; the
  process guardian and the OpenBLAS example put it on their path.
- Every package capability crit serve reads (`chat-json-v1`, `chat-resident-v1`, `chat-stream-v1`, the tool,
  image, config, concurrency and constraint capabilities) is named once, in `crit_core.serving`, beside the ones it
  already held; `crit_api`, `crit run` and `crit pull` import them, `crit_api.options` takes its sampling fields and
  `sampling_capability` from there too, and the step-engine path reads the same sampling list. Before, the names
  were defined in five modules and written out as strings in eight more.
- `crit run`'s build, measure, eval and serve settle their attempts through one helper (`crit/run/attempt.py`):
  the same candidate checks, outcome rules and evidence, where each command kept its own copy. A build's log is now
  kept only once its process tree is proven stopped, as every other attempt's is, since a process left running may
  still write to it.
- Core holds no GGUF config rule of its own: a lone GGUF's `config.json` comes from the rule a package registers for
  its architecture in Core's new registry of model-family knowledge, `crit_core.registry` (kind `gguf_config`, at run
  time or in an entry point of group `crit_core.gguf_config`; [config rules](core/python/crit_core/model/README.md#gguf)).
  crit-eval gives the rules for `llama`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `deepseek4` and
  `qwen4exp`, beside its reference's layouts for them (`crit_eval.architectures.gguf_config`, `gguf_deepseek4` and
  `gguf_qwen4exp`), registered when `crit_eval` is imported; crit imports it, and crit serve's host finds a crit-eval
  checkout beside crit's, so every crit command derives what it did. `crit_core.model.gguf_files` keeps the generic
  keys, public for rules to build on (`MetadataReader`, `base_config`, `expert_config`, `layer_flag`);
  `CONFIG_RULES`, `crit_core.model.gguf_deepseek4` and `crit_core.model.gguf_qwen4exp` are gone. Before, Core carried a code path per model family, against its own rule.
- The rotary table leaves Core for attention-state: `rope_table`, `rotary_dim` and `RopeTable` (`crit/model_config.h`,
  `crit_core.model`) are `crit::techniques` names in [rope_table.h](inference/attention-state/code/rope_table.h)
  and its Python twin `rope_table.py`, the same tables from Core's `RopeConfig`, checked against Transformers and
  against each other as before (`tests/test_rope_tables.py`, CTests `crit_rope_table_*`). Core keeps reading the
  rope parameters; an engine that called its `rope_table` includes attention-state's. Before, Core computed
  positional-encoding arithmetic, against its own rule.
- Core ships no chat template: the one a model without its own renders with comes from the registry
  (`crit_core.registry` kind `chat_templates`, by `config.json` `model_type`;
  [chat templates](core/python/crit_core/text/README.md#chat-templates)), and crit-eval gives DeepSeek V4's, byte for
  byte the one Core shipped, so prepared data and model identities are unchanged. `shipped_template` and
  `chat_template_sources` read the registry; `SHIPPED_TEMPLATES` and `crit_core/text/templates/` are gone, and the
  encoder's golden tests went to crit-eval with the template. Before, Core carried
  one model family's template.

### Fixed

- crit serve reports `expert_reads` and `expert_hits` for engines on the stock adapter too:
  `crit_core.native.resident_engine` passed on only an engine's drafting facts, so the device expert cache counts
  never reached `crit_telemetry` for a native or Python engine served through it, while the eval-v1 adapter passed
  them. One list in `crit_core.serving` (`REQUEST_FACTS`: `DRAFT_FACTS` and `EXPERT_FACTS`) now names a
  generation's facts for both adapters and crit serve, and a plain decode keeps the expert cache counts while it
  drops the drafting ones. Before, five hand-kept lists named them.
- The eval-v1 adapter (`crit_core.native.eval_engine`) passes on `host_bytes_per_pass`, so crit-eval prices an
  offloading engine's host reads in process as it does through crit serve. Its fact list is now crit serve's
  `ENGINE_FACTS` plus what only an in-process adapter reports (`kernel_work`, `exact_variants`, `wider_draft`).
  Before, the two lists were kept apart and the adapter's lacked it.
- `crit models eval` takes no `--baseline`: the command loaded the result and never used it once quality scoring
  left it (crit 0.5.0), while models/README.md still promised the speed change. `crit eval compare SOURCE_RESULT
  RESULT` shows it.
- A `crit run ... eval` or `crit run ... serve` that is terminated (crit run turns SIGTERM into SystemExit) settles
  its attempt as interrupted and exits with the signal's status, as `measure` and `build` already did. Before, it
  settled as failed with "SystemExit: 143" and exited 1.
- `HostBlock::touch` reads one word per page of the system's page size instead of every 4 KiB, so a 16 KiB or
  64 KiB page is read once.
- A failed allocation in Core's CUDA backend no longer leaves the runtime's last error set: an engine's next
  `cudaGetLastError()` after a launch reported it as the launch's.
- A native engine's facts no longer shadow Core's keys in `crit_engine_info`: a fact named like one (`context`,
  `sampling`, ...) is dropped, as a step engine's already was. Before, the info held the key twice and Python read
  the engine's.
- `min_tokens` reaches engines: the engine ABI parses it into Core's sampling options and `crit_engine_info`
  says whether the model applies it, and crit serve forwards a request's `min_tokens` (vLLM's field: no stop token
  before that many) to a package that declares `chat-sampling-min-tokens-v1`, as it does the other sampling fields.
  Before, the ABI dropped it without a word and crit serve refused it.
- `CudaBuffer::upload` and `download` stage in pinned pieces of at most 64 MiB and of what the host ledger has
  left, so loading a large tensor no longer pins its whole size or needs a host limit above it.

## [0.14.0] - 2026-11-02

### Overview

crit 0.14.0 makes concurrent requests the normal path: Core's request runtime serves Python engines as step engines,
crit serve reports and caps its concurrency, and `crit run measure` and Eval's concurrency part measure requests at
once. Core's GPU backend builds on ROCm's HIP runtime and gets device sets for several GPUs, and a run on several GPUs
sizes its context from their memory together. `crit pull` fetches a card's drafters and prebuilt artifacts, `crit bench`
and `crit hub report` measure an installed engine against its card, and `crit mcp` serves crit's commands to an agent as
MCP tools. Engines on Windows opt out of power throttling, and crit serve gains a thinking budget. crit 0.14.0 needs
crit-eval 0.5.2.

### Added

- A run on several GPUs: a `device` file from `crit eval probe --devices` sizes the run's context from their memory
  together, `hardware_constraints.context.devices` names them, and `crit run measure` samples device memory on
  every one of them, summed (nvidia-smi `--id` of each). Checked with host tests on one GPU.
- Core's CUDA backend gets device sets (`crit/cuda_devices.h`, [device sets](core/src/backends/cuda/README.md#device-sets)):
  `CudaDeviceSet(devices)` holds one `CudaBackend` per GPU with its own streams, allocations and ledger, enables peer
  access where the hardware allows it, and orders work between members without a host fence: `copy_async` between
  members' device buffers (peer copies, the source kept alive until done), `order` (one member's stream after
  another's work), `barrier` (every member after every other) and a category timer per member;
  `CudaBackend::owns(buffer)` names a buffer's member. Checked on one RTX 5090 with a set that lists it twice; the
  two-device case is in the test and has not run on two GPUs. Before, nothing in Core knew a second device.
- `inference/device-speculative-round` gets a round driver for Python engines, `python_round.py`: in torch, one
  speculative round for every row a Python engine decodes at once (a Core step engine's `step()`, or one request),
  over three engine hooks (draft after catching up on the previous round, verify writing each column's prediction,
  commit from the round's start). Acceptance, stops, output limits, truncation to Core's count and the packed
  readback (one pinned copy, one event per round) stay on the device, and a round of (rows, width, mode) is one CUDA
  graph captured at load; modes greedy, sampled (`rejection_predictions` turns the verify into rejection sampling
  under the same accept rule), replay and forced (scoring a fixed text in columns). `width_policy.py` picks the
  width per round from each row's decayed per-depth acceptance over the fitted round cost. Checked over a toy target
  with two different drafters against plain greedy decoding, and its CUDA graphs against eager rounds on an RTX
  5090; no engine runs it on a real model yet. Before, every Python MTP or EAGLE engine wrote this driver itself.
- `crit bench OWNER/NAME` measures an engine `crit pull` installed with crit Eval on this machine and prints its
  card's stats beside the measured ones ([benchmark your machine](docs/SERVING.md#benchmark-your-machine)): it readies
  Eval's data for the installed checkpoint (a published reference package, else `crit eval prepare` at the card's
  sampling), serves the engine through crit serve and runs Eval's speed suite at the installed context, and quality
  where the data holds the reference's scores (`--reference` scores them). `crit hub report OWNER/NAME RESULT --hub
  CHECKOUT --by ACCOUNT` then adds the result to the card in a crit-hub checkout with crit-hub's `tools/add_result.py`,
  as a reproduction of a GPU and model the card measured or as a new result, after checking that it measured this
  engine at the commit the card pins. Before, reproducing a card's numbers meant replaying the whole run flow, and a
  card had no place for them.
- An engine's decode telemetry can count its device expert cache: `expert_reads` and `expert_hits` (the routed expert
  reads and those the cache served, together, hits at most reads) pass through crit serve's `crit_telemetry`, from
  ResidentChat's `generate_tokens` timing and an engine library's request facts, and crit Eval's memory tiers price
  an offloading engine's floor at that hit rate ([telemetry](docs/SERVING.md#response-and-telemetry)). Before, nothing reported
  how often a device expert cache hit, so Eval could only assume even routing.
- `crit mcp`, a Model Context Protocol server over stdio ([MCP](mcp/README.md)): every subcommand of `crit run`,
  `crit eval`, `crit kernels`, `crit models` and `crit inference`, and `crit pull` and `crit serve`, is a tool whose
  input schema is read from the command's own parser when the server starts, so it never drifts from the command
  line. A `record` option takes the JSON object itself. Each call runs its command as a job, an owned process tree
  with its output and errors in files: the call returns the result (the JSON output parsed) when the command ends
  within `wait_seconds`, else the job, which `crit_job_wait`, `crit_job_output` and `crit_job_stop` (Ctrl+C, so
  crit settles the attempt as interrupted) follow; the server stops its jobs when the client closes it.
  `crit_serve_status` reads a running server's `/health` and `/v1/models`, and AGENTS.md and the guides it links are
  `crit://` resources. `ProcessTree` can send a tree's errors to their own file. The `crit run` actions and
  `crit inference` subcommands that had no help line have one.
- `crit pull` fetches a card's `downloads` ([installed engines](docs/SERVING.md#installed-engines)). A drafter goes
  into the store's `models/` beside the checkpoint, checked file by file against card.lock.json's `downloads` (or by
  the SHA-256 of its one file), and the engine receives its folder as `context.drafters`, in card order, so an
  installed engine never needs a path from the builder's machine. A prebuilt artifact, a tar archive of what the
  build adds to `engine/`, names the toolchain it was built with and for (crit-hub's new `toolchain`: platforms,
  GPU architecture, CUDA or ROCm version, Python): `crit pull` unpacks the first artifact this machine matches (a
  driver that runs that CUDA version or a newer one; no nvcc) instead of running the build, after checking its
  SHA-256, and never lets it replace a file of the pinned source. `--build-from-source` builds anyway and
  `--download REPOSITORY=PATH` takes a local copy of a download. Before, `crit pull` only named the downloads.
- A `crit serve --models` catalog row can name an installed engine, `{"name", "engine": "OWNER/NAME", "store"}`
  (the store relative to the catalog, default `PROJECT/engines`), served beside run candidates under its name and
  aliases ([several models on one server](docs/SERVING.md#several-models-on-one-server)). Before, only
  `--engine` served an installed engine, and only alone.
- `inference/ssd-expert-materialization` gets a host tier for file-backed experts: `HostExpertTier(capacity)`, a
  bounded host copy between the file and the device cache, least recently used first out. `FileExpertSource` serves
  a demand read from it and holds what it reads there, and `retain(key, bytes, protected)` holds an expert the
  device cache evicts, from its device copy, so its next miss crosses the host link instead of rereading the file.
  Retained bytes are checked like a file read, and an expert the current route selects is never evicted to make
  room. Before, the reference deduplicated only pending reads, and every later miss reread the file. Two other
  engines measured the gain (one serving Qwen3.8-Flash-Next from NVMe on an RTX 3060 read 12.4 instead of 39.7 MiB
  per token on a repeated request); crit has not measured it. Recon's disk scenario names the eviction path.
- `crit run measure` measures resident cases of concurrency above 1: each warmup and repetition sends N copies of the
  case's request at once, one per client thread, every request's timings are samples of the per-request metrics (N
  per repetition), and the new metric `aggregate_tps` (tokens/second) is the repetition's completion tokens over its
  wall time; `extensions.concurrency` records the clients, each repetition's peak in flight and the engine's
  `max_concurrent_requests` ([measured benchmarks](docs/RUN_RECORDS.md#measured-benchmarks)). A case's KV must cover
  concurrency x (prompt + output), and `create` raises a wanted context below that for the cases. Before, measure
  refused concurrency above 1 and such a case needed a self-recorded `benchmark --record`.
- `crit serve` reports how many requests the engine runs at once: `max_concurrent_requests` on each `/v1/models`
  card and in `GET /health` (the default model's), a crit-specific field that is 1 for a package without
  `chat-concurrent-v1`, the loaded engine's declared capacity while it is loaded and `null` until a concurrent
  package's engine loads. `crit serve --max-concurrency N` caps the requests run at once at the smaller of N and that
  capacity ([concurrent requests](docs/SERVING.md#concurrent-requests)). Clients such as crit Eval's concurrency part
  read it to know whether a server batches.
- A thinking budget in `crit serve` (`chat-thinking-budget-v1`): OpenAI's `thinking_budget_tokens` and
  Anthropic's `thinking.budget_tokens` with thinking on cap a reply's reasoning. At the budget the reply
  gets the reasoning format's closing marker and the engine continues from the prompt plus the reply so far, within
  `max_tokens`. ResidentChat implements it (`limits.thinking_tokens`) and `capabilities()` declares it where the
  chat templates teach a reasoning format, or for a package's own `reasoning_format`, so the markers come from the
  format, never from a model table. Before, Anthropic's `budget_tokens` was ignored and listed, and a model could
  reason until `max_tokens` ran out. It applies to text prompts without logprobs or a forced tool call; Eval sends
  none. docs/SERVING.md, Thinking budget.
- An engine may report `prompt_cache: false` in its info when it keeps no prompt cache: crit serve publishes it among
  the `engine_facts` and the eval-v1 adapter passes it on, and crit-eval then sends it no
  follow-up turns, which would be fresh prefills, and passes its prompt_cache gate. docs/SERVING.md, Engine facts.
- Plan rules 5, the default of new runs (docs/RUN_RECORDS.md, Plan rules 5): a strategy's ceiling is the limit the
  parent's Eval measures (or crit computes), not the engine's own speed. crit records per needed family the limit,
  the fraction Eval measured and how far the forecast's high reaches toward it (`inputs.reach`), and refuses a build
  plan whose forecast high stays below half of a needed family's limit unless a `gaps` entry `limit` (answer cause)
  names the measured bound that keeps the rest out of reach, with its evidence in the run. Before, a plan could give
  its own in-process speed as the ceiling and forecast around it, and its outcome then read as on target with
  decode at 11% of Eval's limit. Refused rather than warned, as the other plan checks are. Runs created before keep
  their rules.
- device-expert-cache describes adapting during decode and starting warm: keep the cache adapting (admit
  link-fetched experts, no freezing after a warmup), set the CPU/link split from both paths measured alone and
  together, and optionally ship a residency or routing profile in the package that the engine loads at start, built
  from a general corpus unrelated to Eval like any content-dependent choice. Before, an engine whose cache adapted
  between Eval's repetitions failed the disturbed check, so an agent froze its cache to pass it and decoded from a
  cold one. Eval now measures at steady state (crit-eval's `speed.steady_state`, its next release);
  docs/BUILDING_ENGINES.md and docs/RUN_RECORDS.md describe it.
- Core's GPU backend builds on AMD GPUs: `CRIT_CORE_HIP=ON` compiles the CUDA backend's sources on ROCm's HIP
  runtime as `crit::hip`, through `crit/gpu_runtime.h`, which maps the CUDA runtime names the backend uses to HIP's
  under `CRIT_CORE_HIP`. Engines keep the same classes and stream types; `launch()` is an ordinary launch on HIP.
  The backend, timer and memory fixtures build against it. Written against ROCm 6.4 and not yet built on an AMD
  toolchain; the CUDA build is unchanged. core/src/backends/cuda/README.md, HIP.
- `crit pull` checks crit-hub cards with `backend: hip` (the hub's new field): the probed device's `gfx`
  architecture against `gpu_arch`, its memory, and ROCm's version from the probe's HIP runtime against `rocm`.
  Cards for metal or vulkan read as unmet.
- `crit run` samples an AMD GPU's memory from the amdgpu driver's VRAM counters
  (`/sys/class/drm/card*/device/mem_info_vram_*`) when `nvidia-smi` is missing.
- Building engines and Core's guide treat serving concurrent requests as the normal path for an engine: a new
  [serve concurrent requests](docs/BUILDING_ENGINES.md#serve-concurrent-requests) section maps what a batched engine
  needs to Core's pieces (batched steps, page reservations, the invariance rules, the state cache, speculative
  rows, concurrent engine libraries and step engines). Core's default stays one request.
- Batched speculative decoding has a CPU reference in Core: `SpeculativeRow` (`crit/batch.h`,
  [speculative rows](core/src/batch/README.md#speculative-rows)) stages one request's pending token and drafts as a
  StateCache round, accepts by Core's greedy rule and commits exactly the rows Core kept, rolling the rest back in the
  pages, so every request of a step verifies its own drafts at its own width. A toy speculative engine through a Core
  `Engine` decodes exactly the plain greedy tokens alone and six requests at once (`crit_core_batch_speculative_*`).
- Python engines get Core's concurrent runtime by implementing a step instead of `generate()`: `crit/step_engine.h`
  makes a `ConcurrentModel` of C callbacks (admission, open, step, commit, retire, score), the new `crit_core_steps`
  library exports the engine ABI plus `crit_engine_open_steps`, and `crit_core.native.StepEngine` serves a Python step
  model on it ([step engines](core/src/runtime/README.md#step-engines)). Core then schedules the model's requests in
  mixed prefill/decode steps and handles admission, request options, stop tokens, cancellation, retirement and the
  scoring barrier; `open_engine` picks it for any opened engine with `step()`, so `eval_engine` and `resident_engine`
  serve it with `max_concurrent_requests` above 1. `crit_core.batch` is the Python twin of `crit/batch.h`. A toy step
  model decodes the same tokens batched as one at a time (`crit_core_step_engine`). Before, a Python engine served
  concurrent requests only by merging concurrent `generate()` threads itself.
- The batched reference engine's own paged attention gains a host half and a device page store
  ([paged attention](inference/batched-reference-engine/README.md#paged-attention)): `paged_reference.h` (target
  `crit_paged_reference`, no CUDA) holds the KV page format (BF16, or E4M3 codes with one scale per row and key/value
  head), `Tables` packing a step's block tables, lengths, query starts and write slots into the words the kernels
  read, BF16 and E4M3 codecs matching the device conversions, `HostPages` (a StateStorage in the kernels' layout) and
  `attend`, an FP64 reference with sliding windows, sinks, a soft-cap and tree masks; `paged_attention.h` (target
  `crit_paged_attention`) adds `DevicePages` and `BatchTables` beside the kernels, which dense-layer-offload uses too.
  `crit_paged_reference_*` checks the host half against a dense oracle over StateCache pages with reused prefixes
  and copy-on-write forks, and `crit_cuda_paged_*` compares the kernels with it (needs a GPU). Paged attention is
  model and device arithmetic, so it stays in the engine: Core ships none.
- Core ships the bookkeeping of a batched step (`crit/batch.h`, [batched steps](core/src/batch/README.md)), which
  every engine serving concurrent requests needs: `StepLayout` lays out one mixed step's rows, decode items first and
  then prompt chunks, with the step row of every logits row for one gathered head pass; `step_item` derives an item's
  rows from a `WorkItem` within its grants (a verify's columns cut to its token work); and `PageReservations` is page-reserving admission over a StateCache pool (prompt, output and verify columns, no credit
  for shared prefix pages). They generalize the batched reference engine's step layout and admission. A toy paged
  engine through a Core `Engine` decodes the same tokens batched as alone (`crit_core_batch_engine`).
- Eval attempts journal their round and resume where they stopped: `crit run ... eval` keeps crit-eval's journal
  beside the result (`candidates/ID/eval/ATTEMPT.journal.jsonl`), and `--resume [ATTEMPT]` measures in a new attempt
  from the newest stopped attempt's journal (or ATTEMPT's), answering the requests it holds at their recorded times
  and serving the candidate only for the rest. An attempt that stops for any reason settles with its log, journal
  and partial result (marked `aborted`) as evidence, and the run's events record `eval_started` and `eval_resumed`.
  `--resume` settles an attempt whose process died without settling as `interrupted` once the machine's engine lock
  is free and its journal untouched for a minute. Before, an interrupted Eval kept nothing: a Qwen3.8-Flash-Next
  candidate's, stopped an hour in and minutes before its end, had to start over. Needs crit-eval's next release
  (its journal). docs/RUN_RECORDS.md, Eval evidence.
- A Python engine that reserves more device memory than the device holds is warned once per device under WDDM
  (native Windows and WSL): `crit_core.native.PythonEngine` compares torch's peak reservation with the device's
  memory after opening and after each generation and score. There the driver does not fail such allocations; it
  moves the excess to system memory, and every kernel reading it slows down. A Qwen3.8-Flash-Next engine reserving
  39 GiB on a 32 GiB RTX 5090 under WSL ran its weight GEMMs about 7x slower without any error.
  `crit_core.machine.under_wsl()` is now public for the check.

### Changed

- Core's README states the rule Core is held to: it stays model- and hardware-agnostic, with no layer arithmetic,
  no kernels or device code beyond the backends' runtime services, and no number or rule true for only one model
  family, dtype or GPU architecture. What batched or speculative engines share enters Core as bookkeeping; their
  arithmetic stays in the engine, an Inference entry or a kernel provider, which is why paged attention and the GEMM
  split went back to the batched reference engine.
- An offloading engine no longer has to declare `host_bytes_per_pass` for a reachable floor:
  Eval prices the weights recon's placement keeps off the device at host memory's tier, routed experts at the
  expert cache hit rate the engine reports (`expert_reads`, `expert_hits`), so `crit run eval` notes how it priced
  them instead of warning, and AGENTS.md, building engines, run records, recon's scenario note and the reach
  check ask for the hit rate first. It still warns when Eval could not place the model.
- Core's interface and request runtime are split by concern: `crit/core.h` includes its parts from `crit/core/`
  (resources, profile, request, model, concurrent and engine), which the SDK installs beside it, and
  `core/src/runtime/runtime.cpp` became one file per part of the runtime ([request runtime](core/src/runtime/README.md)).
  The interface and `#include <crit/core.h>` are unchanged.

- AGENTS.md says shape tuning must hold for every prompt length: Eval measures the held-out
  prompts and the long-context curve below the required context at lengths each run draws (80-100% of nominal),
  so a CUDA graph, tile or prefill chunk fitted to one length shows as a slower held-out score.
- An engine of an off-device placement is asked for its host bytes: `crit run eval` warns when recon's placement keeps
  weights off the device (`sparse_device_host`, `sparse_device_host_disk`, `dense_device_host`) and the candidate
  declares no `host_bytes_per_pass`, whose absence prices Eval's pass floor as if every weight were on the device
  (`crit.run.evaluation.host_bytes_warning`; crit-eval's next release also lists it under `headroom.missing`).
  Recon's `sparse_device_host` starting design, AGENTS.md and docs/BUILDING_ENGINES.md say to declare it from the
  first candidate.

- `crit run ... eval` prints, under the priorities, each headline KPI's fraction of its limit and its two largest
  gaps (`decode 46.6 tok/s is 11% of its limit 419.5 tok/s; pass at 16% of its floor; largest gaps: ...`;
  `crit.run.plans.standing`). Before, the closing lines gave only the order of the priorities, and a decode at
  11% of its ceiling could be presented as good.

- Recon's placement plan reserves no fixed prompt workspace beside a device expert cache: the starting design for
  `sparse_device_host` says prompt (prefill) buffers are lent from cache slots while a prompt runs and returned for
  decode, and the cache size it prints, crit-eval's placement, counts the reserve as cache (needs crit-eval's next
  release). Before, the reserve sat idle during decode. The plain-pass bounds add misses split between the CPU and
  the host link, bounded by host memory's read rate, which both paths read, and say even routing is the pessimistic
  case and the cache's hit rate under real routing the main lever. device-expert-cache, heterogeneous-expert-dispatch
  and host-expert-arithmetic say the same: lend prompt buffers from slots, time the CPU and link lanes together.

- Kernels' exactness check covers the whole decode step: Eval's honesty check compares a served reply's
  log-probabilities with the score call's replay, which drafts and verifies at other widths, and library GEMMs pick
  their kernel by shape, so a token's arithmetic must not depend on the step's width even where exactness is only
  reported. Running every decode step at one width (padding columns) is the general fix (kernels/README.md). Found
  in a Qwen3.8-Flash-Next run whose candidate failed honesty at 0.44 and passed with a fixed width (0.0).
- host-memory-admission names WSL's pinned-memory limit: past a large pinned share, a further pin fails with a
  sticky CUDA error although headroom admits it, so engines allocate their pinned buffers up front.

- An `as_quantized` run allows transformed weights unless the request says `weights_policy: as_stored`: the default
  `weights_policy` is now `transformable` in that mode (still unspecified in `preserve`). A candidate may re-encode
  stored 16-bit matrices as it loads them (for example BF16 to FP8 per channel), declaring `weights: "transformed"`
  and `configuration.re_encoding`; the quality budget judges it. That is the route for an already-quantized
  checkpoint, which `crit models ptq` refuses. The agent brief says so and asks to measure it first where memory or
  bandwidth bounds decode, and AGENTS.md leaves `weights_policy` to the user. Before, every run request written so
  far said `as_stored`, so a Qwen3.8-Flash-Next NVFP4 engine kept about 8.5 GB of BF16 attention and shared-expert
  weights on a 32 GB GPU, leaving its expert cache 14% of the experts per layer.

### Fixed

- Engines keep their speed on Windows while `crit serve`'s or `crit run eval`'s window is in the background. Windows
  11 lowers a background process's quality of service unless it opts out: its threads can move to efficiency cores
  at lower clocks and its timer-resolution requests are ignored, so a short sleep lasts a timer tick. Another engine
  measured its CPU expert pool's decode falling from 66 to 4 tok/s that way; crit's engines run host expert threads
  and a host-work-mailbox thread that sleeps 50 µs when idle. The resident host now opts out at start
  (`crit_core.machine.keep_full_speed`, `SetProcessInformation(ProcessPowerThrottling)`), from execution speed alone
  where Windows knows no timer-resolution throttling (before Windows 11, Server 2022), and prints a line if Windows
  refuses; elsewhere nothing changes. docs/SERVING.md, Ownership and recovery.
- `crit serve` refuses a port another socket holds on Windows too. Python's HTTP server sets `SO_REUSEADDR`, which
  on Windows lets a second server bind the same port without an error, so a stale `crit serve` and a new one could
  share a port and a request reach either engine. The bind is exclusive there now (`SO_EXCLUSIVEADDRUSE`), and a
  port in use ends `crit serve` with one line naming it instead of a traceback.
- `crit run ... eval` leaves the served engine the allocator environment it was started with:
  `PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF` are restored after the reference scores the quality texts, so
  crit serve's resident host no longer inherits the expandable segments crit-eval's reference exported for its own
  process (crit-eval 0.4.1 and before). An engine that captures CUDA graphs at load broke under them.
- `crit_core.model` reads two ModelOpt MIXED_PRECISION layouts (nvidia/Qwen3.8-Flash-Next-FP4-Local): MXFP8 block
  scales, one E8M0 byte per 32 inputs of each row with the group size per layer, take their block from the scales'
  shape (`fp8_block`: the configured block, else 128 x 128 where the scales have that grid, else the block that
  divides the matrix exactly) instead of assuming 128 x 128; and NVFP4 scales stored CSF-compressed
  (`weight_scale.nvfp4_csf_fixed` and `.nvfp4_csf_exceptions`) are recognised as NVFP4 (`csf_scale_shape`,
  `StoredMatrix.scale_encoding` "csf").
- `crit run ... check --from-eval` reads only Eval attempts that completed. Before, the newest result file was taken
  whatever its attempt's outcome, so a disturbed round's partial result (no status) gave every gate unmeasured.
- Core's `SpeculativeRow::commit` keeps a round the state cache keeps pending. When a commit's prefix snapshot copy
  throws, the cache leaves the round pending (crit/state.h), but the row forgot it: `discard()` no longer ended it
  and the sequence could never stage again ("state round already pending"). The row now stays staged until
  `discard()` or a retried commit ends the round.
- `Checkpoint::digest_files` names a shard in a subfolder by its path relative to the index
  (`tensors/model-00001-of-00004.safetensors`), as crit's run-record rows and `crit_core.files` do. It named the
  file alone, so shards of one name in two subfolders gave two equal rows, which `verify_digests` refused as its own
  input.
- `crit_core.model.TensorReader` reads and maps on several threads at once, as the native reader's const members may.
  On Windows a read seeks the reader's one descriptor, so concurrent reads of one shard returned another tensor's
  bytes or a spurious "truncated" error (4 of 8 threads in a check); a descriptor or mapping opened on first use
  could also be opened twice. Linux reads were unaffected (`preadv`).
- A step engine hands a banned token's logit bias (-inf) to its callbacks as `-Infinity`, which a Python step model's
  `json.loads` reads. It wrote `-inf`, which is not JSON, so the request's open failed the whole engine. Only C++
  callers of the step engine can send a ban: the engine ABI keeps biases within ±100.
- The engine ABI refuses a token id outside [0, 2^31) in `stop_tokens` and `replay` (and in a step engine's
  configured stop tokens) and a `top_k` past 2^32 - 1, where it wrapped the value into another one: `stop_tokens`
  [2^32 + 5] stopped on token 5.
- Core's concurrent admission calls the model's `admission()`, `scheduling_hint()` and `open()` without the scheduler
  lock. With it held, an engine whose batch notification comes from its device queue (a host function) deadlocked
  when `open()` waited for that queue, as `cudaFree` or a synchronous copy does: the notification waited for the
  lock and `open()` for the notification, and no timeout fired. `submit()` and `cancel()` also waited for every
  `open()`.
- One memory-pressure event no longer closes concurrent admission until the oldest request finishes. A blocked row
  held new admissions until the oldest resident retired, so requests sent meanwhile failed with "pending request
  deadline expired" after `pending_timeout_ms`, with lanes free and the pressure long over. Admissions now resume
  once the blocked request gets a step prepared (a restored request, its first new step) or retires.
- Core's CUDA backend and device sets leave the calling thread's current device as they found it. Every call bound
  its own device and kept it, so after a `CudaDeviceSet` copy, order or barrier an engine's next kernel launch into
  another member's stream failed: a launch needs its stream's device current. `bind_thread()` still makes a
  backend's device current. Checked on one RTX 5090; the two-device case is in the test and has not run on two GPUs.

## [0.13.0] - 2026-10-30

### Overview

crit 0.13.0 makes a build's Eval bounded: `crit run ... eval` measures in crit-eval 0.4's stages within a
45-minute budget, and only the candidate the agent proposes is certified at the run's full context. The wanted
context defaults to 200,000 tokens. Inference ships techniques, not compute kernels: Kernels lists where kernels
come from and `crit kernels new` scaffolds a kernel workbench. `crit pull` installs a crit-hub engine and
`crit serve --engine` serves it without a run. Runs record the placement scenario and companion drafters, and
engines report the bytes a pass reads from host memory. The Python distributions and the Core SDK share version
0.13.0; crit depends on crit-eval 0.4.1.

### Added

- `crit pull OWNER/NAME` installs a crit-hub engine and `crit serve --engine OWNER/NAME` serves it, without a run:
  it reads the card (crit-hub's index.json, or a checkout with `--hub`), checks its requirements against the
  machine (a crit eval probe result and nvcc), fetches the source at its pinned commit and checks its configuration
  against the sealed run's, fetches the checkpoint at its pinned revision (or takes `--model`) and checks every
  file against card.lock.json, runs the card's `build` command and installs the package in an engine store
  (`PROJECT/engines`) whose records hold no absolute paths. Before, using a crit-hub engine meant replaying the
  run-record flow by hand to reach `crit serve`. docs/SERVING.md, Installed engines.
- Recon records the run's placement scenario (`gathered.placement`, from crit-eval's `placement`) when the run names
  its device: `fits_device`, `sparse_device_host`, `sparse_device_host_disk`, `dense_device_host` or `beyond_host`,
  where the lookup tables go, and one plain pass's floor per expert source (device, host link, CPU) from the probe's
  rates. recon.md shows it with the scenario's starting design and asks for a `Placement plan`, which runs with plan
  rules 3 need written before the first candidate; candidate 1, the proof of concept, is built in that design.
  AGENTS.md names the scenarios. Before, an overnight Qwen3.8-Flash-Next build streamed every expert over the host
  link without a cache or drafter (11 tok/s).
- AGENTS.md has a rule for long jobs: harness-tracked background commands, never a turn without a wake pending, and
  the next round's CPU work while the GPU is busy. Before, an overnight build sat idle 2.5 hours after its Eval
  reference finished.
- Inference entry `lookup-table-rows`: rows of a file-backed embedding or n-gram table read with random-access
  advice, exact parallel positioned reads and a hot-row cache. Before, a default mapping read 8 MB ahead per row of a
  51 GB n-gram table (500-700 ms per token of cold rows instead of 2-5 ms).
- `crit_core.serving.engine_facts` passes `host_bytes_per_pass` (`{"link", "cpu"}`: the bytes of a pass read from host
  memory) to crit-eval, which prices them at the probe's host rates; AGENTS.md's ceilings price host bytes the same
  way.
- host-expert-arithmetic and heterogeneous-expert-dispatch say to plan the CPU's share of misses from the first
  candidate where host memory reads faster than the host link copies.

- `recon --drafter` records each companion drafter's `weight_bytes` and `pass_bytes` (crit-eval's
  `companion_drafter_bytes`), and plan rules 2 price a separate drafter's round floor from them where the parent's
  Eval measured no draft floor: ceil(draft_steps / drafts_per_pass) drafter passes at the bandwidth, the row naming
  one by `companion` when recon recorded several. Before, a drafter alternative for a checkpoint without draft layers
  could not rank decode at all (Nanbeige4.3-33B with DFlash2).
- A run request's `drafters` names companion drafter folders a candidate may load beside the model; `create` keeps
  their stored weights out of the KV cache when it derives `kv_tokens` from the device (crit-eval's max context), and
  `hardware_constraints.context` records `companion_drafter_bytes`. Before, the required context filled the device
  and left a drafter no room (Nanbeige4.3-33B at 86K: 0.09 GiB free with its DFlash2 drafter loaded).
- `crit kernels new --format nvfp4` scaffolds a gemv or gemm on NVFP4 weights against bf16 activations: the
  checkpoint's NVFP4 shapes (`--model`), an FP64 reference that decodes E2M1 with its E4M3 group scales and global
  divisor, and cuBLAS on the weights widened to BF16 once as the yardstick, so `crit eval workbench check|bench`
  measures an NVFP4 kernel the engine writes. `--starter` stays bf16 and fp8.
- `drafted` request telemetry: an engine whose draft width varies reports the drafts it made in each decode round
  beside `accepted`, and `crit serve` passes it on in `crit_telemetry`, so Eval reads each round against its own width.

### Changed

- crit needs crit-eval 0.4.1. `crit run ... eval` measures in crit-eval's stages within a 45-minute budget
  (`--budget-minutes`, `--stage`), and `--certify` serves the run's `kv_tokens` itself, which the resources check
  needs: AGENTS.md certifies the candidate the agent proposes. The wanted context defaults to 200,000 tokens (was
  262,144; `context_tokens` in `settings.toml`).
- `crit run ... eval` follows crit-eval's protocol crit-eval-v0.3.0: one `crit eval reference` run serves both
  quality modes (the scores, their spread and, for a checkpoint that declares quantized activations, the
  as_quantized allowance), and a run in quality mode `as_quantized` fetches the published package when the data lacks
  the allowance. Before, as_quantized needed a second reference run with its own scores (`reference-as-quantized/`).
- Kernels' sources name mixed-input GEMMs (CUTLASS, Marlin-style kernels in vLLM and SGLang) for prompts that keep
  16-bit activations on 4-bit weights, as `preserve` runs do on NVFP4 checkpoints, instead of widening each chunk's
  weights to BF16 first.
- New runs check plans with plan rules 4: each candidate has an objective by its number in the run, recorded in the
  plan's `inputs.objective` and named in the agent request. Candidate 1 is the proof of concept, 2 and 3 are crit
  rounds (crit's toolkit, each technique measured against its floor and fixed or replaced where it trails), 4 on are
  critique rounds whose plan names the parent's two weakest terms in `weakest`, with evidence and the levers for
  each, beside other levers. Plans for candidates 2 to 5 must forecast at least +10% on a p1 KPI. Each plan reviews
  its parent's plan against the measured outcome (`review`: the cause and evidence of every missed forecast, what
  each lever delivered, the lesson), and the parent's largest p1 lever, pruned or deferred, is planned again or
  rejected on a measurement of another kernel structure or a vendor library. Every speed lever names its `provider`
  (`inference:`, `hub:`, `lib:` or `own`), and the engine's own code carries a measured `yardstick`.
- A run in quality mode `as_quantized` defaults to a drift budget twice `preserve`'s (perplexity ratio 1.1, mean KLD
  0.1, p99 KLD 1.0), crit-eval's own for that mode from protocol crit-eval-v0.2.0; `preserve` is unchanged and a
  request's `quality_budget` still lays over either.
- Kernels lists where compute kernels come from in [SOURCES.md](kernels/SOURCES.md): the Hugging Face Kernel Hub
  first, then cuBLASLt, CUTLASS, cuDNN, FlashAttention, FlashInfer, flash-linear-attention, Triton, vLLM/SGLang,
  ggml, CPU libraries and the ROCm and Apple stacks. AGENTS.md asks for each adopted kernel to be measured against
  its floor and the best other provider at the same arithmetic.
- `crit run replay --before` accepts a run's first plan: the run as it stood before any candidate, for testing an
  agent's first plan.

- `crit kernels new` scaffolds a kernel workbench folder (work/kernels/NAME): the shapes from the checkpoint's own
  matrices (`--model`), an FP64 reference and a library yardstick for gemv and gemm in bf16 or fp8, a binding with an
  empty kernel to write, crit-eval's harness, and for other operations the functions to fill in. `--from` starts from
  an existing kernel's source with its license (a Hub kernel's binding gets its `registration.h`); `--starter` writes a
  simple portable CUDA kernel instead, allowed from a run's fourth candidate on. crit-eval measures the folder.

### Removed

- Inference no longer ships compute kernels; entries are techniques. Removed entries: stored-tile-gemm,
  w4a16-projections, gguf-projections, rmsnorm-bf16, flash-prefill-attention, fp8-kv-attention, paged-attention,
  latent-attention, rotated-kv-quantization, recurrent-layer-kernels and python-bindings (its C ABI wrapped those
  kernels), with their CUDA tests and the CUDA operator suite. quantized-projections keeps its host codecs (G32
  integer, row-scaled FP8, NVFP4) without the FP8 CUDA projection; heterogeneous-expert-dispatch keeps its placement
  planning and Routing reference without the fused router/shared CUDA kernel; batched-reference-engine carries its
  own paged attention kernels as part of the example engine.

### Fixed

- Native DFlash drafters read `block_size` from `dflash_config` as well as the top level, so `crit run recon
  --drafter` takes a DFlash2 drafter (which keeps it there beside its taps and mask) instead of refusing it.
- `crit run plan` under plan rules 2 no longer tells a plan to give a bare limit where it cannot compute an
  alternative's round floor (rules 2 refuse a bare limit): it says to leave that KPI out of the alternative's limits.
- A Python engine's device memory returns when it closes: `crit_core.native.PythonEngine.close()` collects the
  engine's objects and hands torch's cached blocks back. Parts that referred to each other (a drafter and its model)
  kept the first engine allocated until a later collection, so `crit serve` could not reopen the package at full
  context in the same host, as Eval's lower-context comparison does.
- AGENTS.md says the exactness check runs only for a package that declares plain decoding (`chat-plain-v1`).
- `crit run replay` leaves out a later candidate whose retained source kept file times older than the cutoff
  while its record did not; before, the replay copied the folder without the record and failed.
- `crit run build` refuses a recipe with `capabilities: []` before it reserves the build: an empty list kept the
  adapter's `declared_capabilities` from being asked and packaged a candidate that could not be served.

## [0.12.0] - 2026-10-27

### Overview

crit 0.12.0 gives engine builders the plumbing every engine rewrote. Python engines read a checkpoint's tensors
through crit_core (`TensorReader`, `schemes`, stored-tile-gemm's `tile_source`) and get `crit_core.native`'s stock
Eval and serve adapters; the new python-bindings Inference entry calls the CUDA entries through a C ABI and a
ctypes loader; and `crit run` takes over the workflow steps every run rebuilt (`capture`, `build`,
`check --from-eval`, `serve --check`). Inference adds block verification for sampled speculation, an acceptance
estimator for drafters without a confidence head, NVFP4 four-over-six block scales, L2 raster groups for prompt
GEMMs, Core's `rope_table` and stochastic BF16 state rounding, and fixes min_tokens under a finished grammar, NaN
logits in top-k selection and the finish reason of a reply that stops exactly at the output cap. The Python
distributions and the Core SDK share version 0.12.0; crit depends on crit-eval 0.2.0.

### Added

- `crit_core.model.TensorReader` (and `open_tensors`): the tensor bytes of an inspected safetensors checkpoint, as
  zero-copy read-only views or copies into a caller's buffer, refusing a shard rewritten since its header was read.
  The Python side of `Checkpoint::map` and `Checkpoint::read`, so Python engines no longer parse headers or open
  shards themselves.
- `crit_core.model.schemes`: how a checkpoint stores each weight matrix, from headers alone. The recognition moves here
  from crit-eval (which reads it from crit_core from its next release), `layer_prefix` moves here from crit Models,
  and `matrix_parts` names the tensors and scale conventions of an NVFP4 (compressed-tensors or ModelOpt), FP8 or
  plain matrix, so engines that load matrices as stored no longer spell each format's tensor names.
- stored-tile-gemm's `tile_source`: one NVFP4 (compressed-tensors or ModelOpt) or FP8 checkpoint matrix as the repack
  kernels take it, its global and input scales as the kernels' divisors whatever the layout, read through
  `matrix_parts` and `TensorReader`. The load path every engine using stored tiles wrote for itself.
- Python engines get `crit_core.native`'s stock adapters: `eval_engine` and `resident_engine` take a function that
  opens an engine with the engine library's interface (`info`, `generate`, `score`, `close`) in place of the library
  path, so a Python engine writes neither an eval-v1 nor a chat-resident-v1 adapter, and Eval and crit serve reach it
  through the same code as a native engine (`PythonEngine` adds the library's request accounting).
- `crit run ... capture`, `build`, `check --from-eval` and `serve --check`: the workflow steps every run rebuilt for
  itself. `capture` writes a case's chat request that renders to the workload's `prompt_tokens` under the template
  variables candidates serve, and sets its manifest row in the inputs payload; `build` runs a recipe's
  `build.command` on a copy of the retained source under a build attempt (log kept, attempt settled) and packages the
  result, binding the files the build made as dependencies and taking capabilities from the adapter's
  `declared_capabilities(model_path)`; `check --from-eval` records each case's arithmetic, functional and resources
  check from a retained Eval result on the current package; `serve --check` runs the standard serving checks
  (consecutive requests, a system message, streaming, a stop string, cancelling, a tool call, log-probabilities, a
  template switch, stopping), keeps their report and records a functional check per case.
- Inference entry python-bindings (38 entries): one shared library with a plain C ABI over the CUDA entries
  (stored-tile GEMMs, recurrent layers and recovery, FP8 KV attention, token sampling, greedy accept; 74 functions)
  and a standard-library ctypes loader that types every function from the header, so a Python engine calls them on
  its own buffers and streams without writing an extension. Checked on an RTX 5090.
- mtp-decode: block verification for sampled speculation (`VerifyRule::block`, `block_verify`; Sun et al.), which
  keeps the target's law and accepts at least as many drafts as the per-token rule. Checked exactly on a Markov
  fixture (0.865 against 0.723 expected standing drafts at width 3) and by the 90,000-history chi-square check.
  token-sampling runs the same rule on the device (`Verify::rule = VerifyRule::block`, `block_draft_purpose`), and
  Core reserves draw purpose `block` (4).
- mtp-decode: `AcceptanceEstimator`, an online per-position logistic estimate of draft acceptance from the drafter's
  `logit(q_max)`, with survival cuts and a batch budget split, so drafters without a confidence head can trim rounds.
- NVFP4 "four over six" block scales: the Inference codec's `four_over_six` (Python and C++) and the Models format
  `nvfp4-4over6` keep, per 16-value block, whichever of the amax / 6 and amax / 4 scales gives the smaller squared
  error, in the unchanged NVFP4 layout. On Qwen3-1.7B (weights only, RTN) the mean KL divergence from BF16 fell on
  each of crit-eval's six quality conversations, by 14% to 40%, and held on its long document.
- stored-tile-gemm: raster groups (`Plan::group`, `A16Plan::group`, `l2_raster_group`) keep a group of weight bands
  L2-resident while the token blocks pass over them; bit-identical to the plain order. Opt-in (plans keep group 1):
  on an RTX 5090 the derived group took 12% off a W4A4 prompt GEMM with K = 17408, while W8A8 and smaller K stayed
  within a few percent.
- Core: `rope_table` (C++ `crit/model_config.h`, Python `crit_core.model.rope_table`) derives the rotary inverse
  frequencies and attention factor from a `RopeConfig` for default, mRoPE, linear, dynamic NTK, YaRN, Llama 3 and
  LongRoPE scaling, matching Transformers' `ROPE_INIT_FUNCTIONS`, so engines stop re-deriving them.
- recurrent-state-recovery: `stochastic_round.h`, unbiased stochastic rounding of FP32 state to BF16 storage with
  replayable draws keyed by (seed, position, layer, element); opt-in until a crit-eval run shows the gain.

### Changed

- `crit run ... resolve` refuses inputs whose captured chat requests render to another prompt length than the
  workload's `prompt_tokens` under the template variables candidates serve (the request's own
  `chat_template_kwargs` over recon's), so a case captured with thinking on while candidates serve it off is refused
  before any candidate is built instead of failing `measure` afterwards.
- `crit run ... measure` sends the `chat_template_kwargs` a captured request names and records them in
  `extensions.chat_template_kwargs`.

### Fixed

- A missing JSON record file is named as missing (with the directory a relative path resolved against) instead of
  "JSON must be a regular file under 1 MiB".
- Core's sampler: min_tokens suppression yields when a constraint allows only suppressed tokens (`suppression_yields`),
  so a structured request whose grammar has finished, such as `{"type": "boolean"}` after `true`, stops before
  min_tokens instead of failing with "no selectable token". token-sampling takes the same answer per row
  (`Rows::yield`) and its Python reference a `yields` flag.
- heterogeneous-expert-dispatch and draft-trees: a NaN logit ranks as -inf in top-k selection. An all-NaN router row
  returned expert id `0x7fffffff` and read out of bounds, and an all-NaN draft row drafted token `INT_MAX`; now the
  router selects in-range experts and the tree never keeps a candidate scoring -inf (also in tree.py).
- crit serve: a reply that ends on a stop token exactly at the output cap finished as "length" (OpenAI), `max_tokens`
  (Anthropic) or `incomplete` (Responses), because the server inferred the reason from the token count. Engine
  replies may now carry `finish` (`stop` or `length`), which ResidentChat sends and the server uses.

## [0.11.0] - 2026-10-26

### Overview

crit 0.11.0 lets the user choose what quality means, and replays old runs against today's rules. A run's
`quality_mode` is `preserve` (the default: quality against the checkpoint with exact activations, prefill measured
against 16-bit prompt arithmetic) or `as_quantized` (against the checkpoint run as quantized, as serving stacks run
NVFP4 and FP8 checkpoints, with prefill measured against the stored format); crit-eval 0.1.2 scores both references
and prices each mode's limits. `crit run ... replay` copies a run as it stood when a candidate's Eval finished, or just
before a plan, renders its Eval statuses again with today's crit-eval and asks any agent harness for the plan it would
record next, which `plan --check` validates without recording: a change to crit's rules or Eval's reports is tested on
an old run in minutes instead of a new run. Recon serves thinking off by default, as Eval renders prompts, a kernel
term counts as exhausted only near its floor or against another kernel structure, and the README is reorganized. The
Python distributions and the Core SDK share version 0.11.0; crit depends on crit-eval 0.1.2.

### Added

- A run's `quality_mode` (request or `settings.toml`): `preserve`, the default, judges quality against the checkpoint
  with exact activations; `as_quantized` against it run as quantized (crit-eval's quality modes), so the checkpoint's
  own activation quantization is no drift and prefill is measured against its stored format. `crit run ... eval`
  scores, fetches and judges in the run's mode, and the brief names it.
- `crit run ... replay --at CANDIDATE --out DIR --count N`: N copies of a run as it stood when the candidate's Eval
  finished, with every copied Eval status rendered again by today's crit-eval and a brief for an agent to say what it
  would do next (`replay/plan.json`, `replay/next.md`). A replay records nothing: crit refuses candidates, plans,
  attempts, Eval, seals and stops in it. `replay --collect` checks and lists the replays' plans side by side. It shows
  in minutes whether a change to crit or crit-eval changes what an agent decides at that point of an old run.
  `--agent CMD` starts any harness on every replay at once (a command template with `{brief}`, `{folder}`,
  `{root}` and `{number}`), waits for all and collects; crit itself starts no agent otherwise. `--before PLAN` replays
  the run as it stood just before that plan was recorded, with the work done toward it since its parent's Eval;
  drafts of the next plan in work/ are always left out.
- `crit run ... plan --record plan.json --check` runs a plan's checks and prints the record crit would store, recording
  nothing.

### Changed

- The README presents crit's parts in the order Core, Inference, Eval, Kernels, Models, API and CLI: Core says how
  it was distilled and lists its parts, Inference is the register of speed-up strategies by bottleneck with its entry
  layout and contribution criteria, and Eval states crit-eval's four claims as crit-eval's own README does.
- AGENTS.md, the run brief and Building engines: a kernel term is exhausted only when it runs near its floor at
  the arithmetic the engine already uses (crit-eval's achieved TFLOPS and its gap at the rate of the engine's own
  prompt activations) or when another kernel structure or a vendor yardstick was measured on it; tiling and launch
  variants of one kernel reject those variants, not the term. Building engines asks for a trace of every
  prefill-scored case, and stored-tile-gemm gives the `kernel_work` rules of its BF16-activation GEMMs.
- Recon freezes a chat template's `enable_thinking` off (`template_variables`), as crit Eval renders every prompt:
  a candidate served with thinking on by default stopped its Eval at the first request and was sealed
  benchmark_failed. Requests still turn thinking on.
- `crit_core.serving.ResidentChat` renders with the `template_variables` recon froze into configuration `recon` when
  the configuration has no `chat_template_kwargs` of its own; before, the frozen variables reached no prompt unless
  the engine copied them.

## [0.10.0] - 2026-10-26

### Overview

crit 0.10.0 gives crit Eval its own repository and distribution
([crit-labs/crit-eval](https://github.com/crit-labs/crit-eval), crit-eval 0.1.0) and makes crit Core's Python package
a distribution of its own, crit-core, so crit, crit-core and crit-eval install from PyPI; a pushed version tag
publishes crit and crit-core. The Inference register grows the parts of speculative decoding on one GPU:
device-speculative-round chooses each round's draft width, reads back one block per round, runs zero-copy rounds and
replays a fixed continuation; token-sampling gains sparse draft laws, device log-probabilities and an exact fast path
for greedy rows and small top-k; stored-tile-gemm gains a row-invariant decode and verify GEMV, BF16-activation prompt
GEMMs and placement into combined matrices; fp8-kv-attention and recurrent-layer-kernels write from device positions,
decode faster and commit speculative rounds bit for bit. Before the split, Eval moved through protocols v7.1 to v8:
profiles, reference packages, the reference's own spread, and references for GGUF, DeepSeek V4 and mixed-precision
checkpoints. Runs get plan rules 3 and pre-seal serving (`crit run ... serve`), and crit's adapters pass the facts
crit-eval 0.1.0's reports read. The Python distributions and the Core SDK share version 0.10.0;
crit depends on crit-eval 0.1.1, which imports on Python 3.11 again.

### Added

- Plan rules 3, the default of new runs: a plan whose forecast gains more on p2 than on p1, or lets a p1 KPI fall, is
  refused at record time unless it gives a recorded `forecast_justification` (check 12).
- Plan rules 3: below half its limit a family is marked not needed only on measured rejections (gaps `evidence`, a
  file in the run; Eval gaps answered `rejected`), and prefill also rejects every prompt activation format Eval prices
  faster than the engine's own with a quality measurement outside crit-eval's quality spread (`activation:FORMAT`).
- Plan rules 3: recon.md's prefill arithmetic plan (each matrix group's prompt activation format and its quality
  measurement) and drafting plan are written before the first candidate.
- AGENTS.md, Building engines and the run brief point the agent at crit-eval's new reports: `status --kernels`
  (prefill terms against their floors, verify against draft), the prefill ceilings by prompt activation format and
  their reachable limit, in-process quality with `--attribute-drift`, the quality spread, the time to first token by
  part, the sampled cases' errors and the free device memory.
- stored-tile-gemm's `GemvPlan::role`: a plan with `KernelRole::draft` runs the same GEMV (the same bits) under kernel
  names that carry `DraftRole`, so a kernel trace tells a drafter's GEMVs from the target's verify and decode GEMVs
  and `crit eval status --kernels` splits a round into verify pass and draft steps. device-speculative-round and
  Building engines say how to keep draft kernels recognisable.
- crit serve's `engine_facts` carry the engine's declared `prompt_activations` (validated; crit-eval prices the prefill
  ceilings by activation format with them and names the reachable one), and `crit_core.native.eval_engine` passes
  `prompt_activations`, `exact_variants` (`crit eval run --attribute-drift`) and `wider_draft` from the engine's info.
- `crit_core.native.eval_engine` scores fixed text (`score()`) when the engine declares `token_logprobs`,
  `query_scoring` and `replay`, as crit serve's score call does, so `crit eval run --adapter A --only quality` checks
  quality in process during development (crit-eval's development evidence, never a gate result).
- `crit_core.native.eval_engine` passes each generation's `prefill_ms` and `first_token_ms` (Core's timing), so
  crit-eval splits each case's time to first token into the engine's own time and the rest.
- `crit run ... serve --candidate ID` serves a packaged candidate before it is sealed through crit serve, under a
  check attempt, until Ctrl+C or the reservation's deadline, so the checks AGENTS.md asks for before sealing
  (streaming, stop strings, tool calls, Ctrl+C) can run on the candidate itself; `crit serve` still serves only
  sealed candidates.
- device-speculative-round's `DeviceSpeculativeRound::launch(output_tokens, K)` chooses each round's draft width up to
  the configured widest over one set of buffers, with one Forward/Finish graph pair per width; hidden-state-drafters'
  `enqueue_eagle_ingest` takes a column count and row stride, so a narrower round still catches up a wider one's tokens.
- device-speculative-round reads back one block per round (`enqueue_pack`): the packed result, the accepted draft
  count and the frontier, which the controller checks against its mirror, and optionally each emitted token's
  log-probability and ranked alternatives, which `logprobs()` / `top_logprobs()` hand to Core's deferred Progress spans.
- device-speculative-round's `RoundConfig::zero_copy` reads the round's allowance in place (`enqueue_set_budget`) and
  writes the readback block to mapped pinned memory, so a round has no copy; the fixture's rounds went from 170-215 to
  65-100 us per round on one RTX 5090 under WSL2, with the same tokens and state.
- device-speculative-round describes and tests forced replay of a fixed continuation (crit Eval's `replay`, crit
  serve's score call) through the unchanged accept: a verify step writes its predictions from the continuation.
- fp8-kv-attention's `write` takes its position from device memory (`const int* pos0_device`) and skips rows outside
  the cache, so a captured decode graph writes each step's keys and values; FP32 rows are rounded once, straight to
  E4M3, instead of through BF16.
- stored-tile-gemm repacks FP8 / NVFP4 tensors straight into their place in a combined matrix
  (`enqueue_repack_fp8_rows`, `enqueue_repack_fp4_rows`, `enqueue_place_row_scales`: concatenated along N or
  gate | up interleaved in 16-row tiles), and its gated MLP launches (`enqueue_fp8_gated`, `enqueue_fp4_gated`)
  write BF16 `act(gate) * up` for SiLU, tanh-GELU or erf-GELU without an FP32 `[T][2I]` output.
- token-sampling's `enqueue_sample` can write each row's law as kept (token, probability) pairs (`SparseLaw`, count -1
  where the row took the full law) and `enqueue_verify` checks sampled drafts against them (`Verify::draft_laws`), so
  a sampled drafter needs no `[K][vocab]` dense `q` rows; a draft without a sparse form falls back to its dense row,
  or to the target's own draw, which keeps the law.
- recurrent-layer-kernels' deferred commit (recurrent-state-recovery's Strategy C) on its column scan and convolution:
  `enqueue_delta_columns_deferred` and `enqueue_conv_deferred` replay the previous verify's accepted prefix from a
  stash with the scan's own update expression, then verify without writing the state, so a gated-delta layer needs
  no round-start copy and no per-column checkpoints; each block consumes its own count, which recurrent-state-recovery's
  new many-count `enqueue_record_accepted` writes for every layer in one launch (one RTX 5090, 9 columns of 32 heads
  of 128 x 128: 9.6 us with the record against 15.7 us with checkpoints).
- stored-tile-gemm has a decode / verify GEMV over its FP8 and NVFP4 tiles for 1 to 16 BF16 rows (`plan_gemv`,
  `enqueue_gemv_fp8`, `enqueue_gemv_fp4` and gated variants) on BF16 tensor cores from sm_80, with each row's bits
  independent of the row count and its K split derived from N, the SM count and occupancy.
- token-sampling computes Core's `TokenLogprob` values on the device: `enqueue_logprobs` gives raw rows' chosen
  log-probabilities and up to 64 ranked alternatives (FP32 exps with FP64 sums, within ~1e-6 of Core's, ties at
  the boundary resolved by a scan of the row as Core ranks them), and `enqueue_sample` / `enqueue_verify` fuse the
  same values (`Logprobs`), so a device round can fill its `RoundView::logprob` rows.
- stored-tile-gemm has BF16-activation prompt GEMMs over its FP8 and NVFP4 tiles (`plan_a16`, `enqueue_fp8_a16`,
  `enqueue_fp4_a16` and gated variants): exact decode in registers, BF16 MMA with an FP32 fold every 512 columns, a
  one-wave K split, sm_80 and later, for checkpoints whose quality budget cannot afford W8A8 / W4A4 activation
  quantization.
- Eval results record every part's wall seconds in the order the parts ran (`timing.parts`) and the run's total
  (`timing.total_seconds`), and the log prints them.
- Reference packages: `crit eval pack` adds a checkpoint's prepared data and reference scores to a package folder,
  `crit eval fetch` writes them into a data folder after checking every file's sha256, found by the checkpoint's
  identity. With `eval_packages` in `settings.toml` (a folder or an https base URL), `crit eval run` and
  `crit run eval` fetch a missing reference instead of scoring it, so a run on a packaged checkpoint needs only the
  engine.
- crit-eval-v7.1: Eval profiles (`crit eval run --profile NAME`). `hub`, for crit-hub submissions, measures every
  engine at 200,000 tokens of context, the long-context curve at its two ends with the points between interpolated
  (`context.interpolated`), and loads the engine once (the lower-context gate is unmeasured). Results record their
  profile; `compare` and `--baseline` do not compare results of different profiles.
- crit-eval-v7.2: the sampling check judges each speed case's sampled-suite warmup, which now asks for the
  log-probabilities (`sampling_check.replies`), instead of sending one more sampled reply per case; a run without
  warmups, or an engine that cannot serve them so, still gets its own replies.
- Eval prints a time estimate of every part and the run after the first measured speed case (`estimate` in the
  result, crit_eval.estimate), pricing the run's fixed requests with the prefill and decode costs the engine's first
  requests fit.
- Core ships a chat template for a model family whose checkpoints carry none, by `config.json` `model_type`
  (`crit_core.text.shipped_template`, `chat_template_sources`): DeepSeek V4's (`text/templates/deepseek-v4.jinja`), a
  port of DeepSeek's official encoder (`encoding_dsv4.py`, MIT) that renders exactly what it does on its four golden
  tests (tests/fixtures/deepseek_v4) and on every prompt of Eval's suite. `load_model_text` falls back to it, or takes
  `chat_template=SOURCE`, so `crit eval prepare` and crit serve's ResidentChat render DeepSeek V4 checkpoints.
  `crit eval prepare`, `run` and `fetch` take `--chat-template FILE` for any other model without one. prepared.json
  records the template (`chat_template`: origin, sha256, file and source), and one that is none of the model's files
  is part of its identity (`model_identity`), so references, packages and `crit run ... eval` agree on the data and
  `crit eval run` refuses data prepared with another. Data of models with their own template keeps its identity.

### Changed

- crit Eval 0.1.0 (protocol crit-eval-v0.1.0) is its first release and carries the new reports; its sampled speed
  suite repeats a noisy case at further seeds by default (`--no-adaptive-sampled` opts out). Data and results of the
  earlier crit-eval-v0.1.0 checkout measured sampled decode on three seeds alone: prepare the data again (`crit eval
  fetch` from the repacked reference packages, or `crit eval prepare` with the reference scores carried over) and do
  not compare sampled decodes across the two.

- crit Eval moved to its own repository, [crit-labs/crit-eval](https://github.com/crit-labs/crit-eval), with its
  suite, protocol (crit-eval-v8, continued as crit-eval-v0.1.0), probes, docs and tests: crit depends on it and `crit eval`
  runs it as before. crit Core's Python package installs on its own as crit-core (core/python/pyproject.toml), which
  both depend on; crit's wheel no longer carries crit_core or crit_eval. For a checkout, install
  `python -m pip install -e core/python -e ../crit-eval -e .` with crit-eval cloned beside crit.
- crit models' GPTQ feeds each decoder layer what the model's own forward hands it, captured once per calibration
  window through the reference (`StreamingReference._capture`): input states, masks, positions and token inputs. It
  fed every layer the embeddings and one causal mask, so models whose layers take more (Qwen4Exp's four
  hyper-connection streams, PLE n-gram ids and QSA indexer masks) failed; Qwen3-style models calibrate as before. The
  plan finds the decoder layers of models without a layer input norm by the prefix holding the most layer tensors
  (`layers.0.input_layernorm.weight` stays the first rule).
- crit eval reference runs the leading 8,192-token chunks that texts share once: the other texts resume from a copy
  of the first's cache after them, with the same score bytes. The long quality sets' texts share their document, so
  the Qwen3.8-27B reference took 6.4 instead of 9 minutes on an RTX 5090 (Qwen3.8-Flash-Next 16 instead of 17).
- crit-eval-v8: one quality text per long length (`german-letter` at 1K, `python-merge-intervals` at 4K,
  `explain-tides` at 32K; v7 scored three per length that shared their document), the honesty check's long reply is
  the long-context curve's first reply (its request asks for log-probabilities, scored once; a server that cannot
  stream them keeps its curve point), quality fails fast (a set drifting a mean KLD over ten times the budget stops
  the scoring and the parts after quality, `stopped_early`), and the hub profile decodes no held-out prompts and
  runs two sampled seeds. Data prepared under v7 must be prepared and scored again. The reference also scores
  texts of 8,192 tokens or more a second time with TF32 matmuls (an engine-sized perturbation of the model's own
  arithmetic) and stores how far that run lands from its scores (`reference/spread.json`); a set where that spread exceeds the drift budget (a model whose
  routing or recurrent state amplifies rounding, such as Qwen3.8-Flash-Next at 32K) is judged against twice the
  spread instead, and the result says so. Dense models' two runs agree to rounding.
- crit-eval-v7.3: the hub profile judges retrieval at 4,096, 32,768 and the requirement only (protocol
  profiles.hub.retrieval_points): the 131,072-token request the curve no longer makes cost about 50 s per run.
- crit eval reference's qwen4_exp QSA indexer form (`crit_eval.indexers`) selects exactly the blocks transformers'
  loop selects, ties included: it runs the loop's arithmetic per query on tensors of that query's shape, pools each
  block count once and skips queries within their budget. The earlier batched form broke ties among blocks scoring
  exactly 0 its own way and could round near-ties differently (a tiny model's target 0.43 nats off). The form's
  version is part of the reference identity (`FORM_VERSIONS`), so Qwen3.8-Flash-Next references are scored again.
- crit eval reference scores long texts of models whose attention has sinks (DeepSeek V4, gpt-oss and others) with
  the modeling code's own eager attention run in blocks of query rows, with eager's additive masks
  (`chunked_sink_attention`). DeepSeek V4's long texts scored NaN: its compressed entries were judged by position and
  their mask cast to boolean. Other models' long-text attention, and their stored references, are unchanged.
- crit eval reference reads lone DeepSeek V4 (`deepseek4`) and Qwen3.8-Flash-Next (`qwen4exp`) GGUFs. DeepSeek V4:
  every tensor in transformers' DeepseekV4 names (MXFP4 experts and Q8_0 matrices decoded as stored, hash-routing
  tables restored), config.json in DeepSeek's released keys, and the `deepseek-v3` / `joyai-llm` pre-tokenizers.
  Qwen3.8-Flash-Next: the converter's shared-input merges, indexer split, zero-centred norms and moved constants
  undone, and a config whose seed, n-gram base and padding make transformers recompute the PLE hash constants and
  table rows the GGUF stores; GGUFs whose attention layers have no compress ratio are refused. Each is proven by a
  round trip, written to GGUF and read back (tests/test_eval_gguf_deepseek4.py, tests/test_eval_gguf_qwen4exp.py),
  and needs a gguf-py that knows the architecture (newer than the 0.19.0 release).
- crit eval reference reads GGUF embedding tables by rows (`GgufCheckpoint.row_source`): a row of a GGUF tensor is
  whole blocks of its format, so a lookup reads exactly the bytes of the rows it needs and decodes them alone, the
  same values as decoding the whole table, which it no longer does.
- crit eval reference streams experts: a fused experts parameter that would take more than half the device's free
  memory in FP32 keeps its stored bytes on the device and decodes each expert when the eager experts loop reads it,
  with the same score bytes. Qwen3.8-Flash-Next forced to stream peaked at 12 instead of 26 GB, so layers such as
  DeepSeek V4 Flash's (about 26 GB in FP32) fit consumer devices.
- crit serve's fixed-text scores (`chat-score-v1`) cross the resident host's channel as packed arrays
  (`ResidentChat.score` answers a request offering `"format": "packed"`; `score_tokens` may return
  `crit_core.serving.ScoreRows`, and `crit_core.native` returns the engine's arrays without a Python object per
  value), and Eval builds the rows without GC rescans: the same scores bit for bit, about 9x less time outside the
  engine. `crit_core.native.ResidentEngine` now counts the replay's second prefill of the prompt in a score's
  `prefill_ms`.
- crit eval reference keeps the texts' prompt states in host memory, up to half the memory available less the
  staging buffer, instead of writing every layer's state to scratch files and reading it back (about 23 GB each way
  for Qwen3.8-Flash-Next); states that do not fit still go to disk, and the scores are the same bytes.
- AGENTS.md says what a clean test of crit excludes: the Kernel Hub is part of crit's toolkit and stays in
  unless the run's criteria name it; earlier runs' results and engines are what such a test leaves out.
- AGENTS.md's build loop defers a lever still in development instead of waiting for it, bounds a kernel lever by
  the kernel's share of its phase in the profile, treats isolated kernel benchmarks as leads until the integrated
  engine is measured, and has a precision change measured against the quality budget before other levers build
  on it.
- mtp-decode states that a single-layer drafter's prompt ingestion needs only each position's key/value path
  (fused input, input norm, key/value projections), not its attention or MLP, so its cost does not grow with
  the context.
- stored-tile-gemm's `enqueue_fp8` / `enqueue_fp4` pack and compute only the `ceil(T / (8 nt))` token blocks a launch
  needs instead of the plan's padded size, with every token's bits unchanged, so a short last prompt chunk no longer
  costs a full-size GEMM.
- recurrent-layer-kernels' gated-delta column scan reduces a column's eight state dot products together, a
  reduce-scatter whose pair sums are those of the per-value warp tree (17 shuffles instead of 48), so results are
  unchanged bit for bit and verify columns cost less (one tested GPU: 11 to 28% faster at 8 to 64 columns).
- fp8-kv-attention's tile step skips the per-key mask where both of a thread's rows see the whole tile and a warp's
  output rescale where none of its row maxima moved, and unused decode rows see every key so they never force the
  mask. Results are unchanged bit for bit; on one tested GPU decode over 262,144 keys runs 9-21% faster and windowed
  prefill 7-11%.
- token-sampling runs greedy rows and `1 <= top_k <= 64` on an exact multi-block fast path: `ceil(vocab / 4096)` stage
  blocks per row keep each block's best `top_k + 1` processed candidates and one law block per row runs Core's steps
  7 to 12 on them in FP64 in Core's summation order, verify rule included, so draws and laws equal Core's Sampler; open
  rows (no top-k filter) are settled there when top-p or min-p closes within the top 64 or the draw lands there, and
  every other row still takes the full law, chosen per row on the device. On one RTX 5090 with 248,077-token rows a
  top-k 20 / top-p 0.95 draw went from ~3.5 ms to ~15 us and a K = 8 verify from ~4.6 ms to ~20 us. `enqueue_select`
  now takes strided rows with their own `Params` and a `select_scratch_bytes` scratch (`kStageBlocks` and the
  `cand_values` / `cand_ids` buffers are gone), and `scratch_floats` covers the fast path's lists.
- recurrent-layer-kernels' convolution reads its window from `Conv::state_in` and writes it to `Conv::state_out`
  (`Conv::state` is gone; pass the same buffer for both to update it in place), so a verify reads a round-start window
  without copying it. Out of place, a prompt of more than 32 rows runs a thread per (channel, 32 rows) with the
  row-by-row form's bits, and the kernels are instantiated per width, so the window stays in registers (one RTX 5090,
  8192 rows of 10,240 channels: 3.9 ms to 0.45 ms out of place, 4.0 ms to 1.2 ms in place).
- fp8-kv-attention's decode combine computes the block weights in parallel and moves the denominator to its own warp,
  loading partial outputs 16 at a time while the sums keep block order, so results are unchanged bit for bit and one
  tested GPU runs it 8-18% faster (4.4 to 3.8 us at head dimension 256 and 42 blocks).
- device-speculative-round states that junk and padding columns of fixed-width verify and catch-up graphs must hold
  finite values (and valid ids and positions where a step gathers or writes for them): a masked term has weight 0,
  and 0 x NaN poisons the real rows that read it.
- device-speculative-round's sampled-round fixture starts its rows from `enqueue_seed` and takes the history commit's
  count from `control[commit_tokens]`, so the seed fix is also checked with token-sampling's own `enqueue_commit`.
- stored-tile-gemm's default `plan` splits K only while the split grid fits in one wave of resident blocks (SM count
  times the kernel's occupancy) instead of until two blocks per SM, which over-split these one-block-per-SM kernels;
  `min_blocks` still sets the old rule explicitly.
- fp8-kv-attention's decode runs two key streams per block, as paged-attention's decode does, where their shared
  memory (`decode_shared_bytes`, 96 KB at head dimension 256) fits the device's per-block limit, and merges them
  inside the block, so `decode_partial_floats` is unchanged; one tested GPU decodes one column over 262,144 keys in
  356 instead of 494 us. Bits differ from the single-stream decode, which `Decode::streams = 1` keeps (and which runs
  where two streams do not fit); column invariance holds for both.
- token-sampling's README groups its validation by test file, says the timings illustrate one device instead of
  claiming no throughput, and its contract lists the log-probability rows.
- stored-tile-gemm's README and contract state what a GEMV or BF16-activation GEMM adaptation must keep: row chains
  independent of the row count with one plan for decode and verify, exact decodes and a bounded chain before each
  FP32 fold.

### Fixed

- `crit eval reference` reads safetensors stored in a model's own original tensor names (DeepSeek V4's
  `hc_head_fn`, `layers.N.ffn.experts.E.w1`, `attn.attn_sink`, as released and as compressed-tensors quantizations
  keep them) where transformers' renamings for the model type find them; it stopped on the first such tensor.
- compressed-tensors checkpoints with several config groups (mixed-precision, such as W4A16-FP8) decode each matrix
  with the group whose targets name it: every matrix took the first group's arguments, so packed int4 experts after
  an FP8 group were unpacked as 8-bit codes.
- `crit eval reference` scores DeepSeek V4's long texts on a 32 GB device: its indexer scorer runs over blocks of query
  rows (crit_eval.indexers), the same scores bit for bit, where transformers' scorer held three 17 GiB FP32 tensors
  for an 8192-token chunk of a 32K-token text.
- The reference spread scores the long texts again on a model whose decoder layers hold computed buffers of their own
  (DeepSeek V4's compressors' rotary frequencies): a layer released after the first pass gets them back when it loads
  again, where they stayed meta and the second pass stopped.
- `crit eval kernels` keeps kernels in unnamed namespaces readable: their demangled `(anonymous namespace)::`
  prefix is no longer taken for the parameter list, which cut such names to their outer namespace.
- token-sampling holds the sampling rules in double and runs Core's steps 2 to 5 in IEEE double, as Core's
  `SamplingParams` and `Sampler` do: float rules kept one token more than Core at an exact top-p boundary (0.8,
  0.1 or 0.3 over equal weights), and a bias past the float range turned a selectable logit into +inf (a NaN law)
  or -inf (no token). `Params` and `Request` rule fields are now `double`.
- token-sampling's `enqueue_verify` verifies a cut column (a draft id outside the vocabulary, such as the -1 of
  hidden-state-drafters' confidence cut) against the target law alone, the target's own draw like the bonus
  column: it ran the rejection rule's residual against whatever the column's `q` row still held, so the emitted
  token followed `max(p - q, 0)` instead of `p`.
- device-speculative-round: a sampled round's history commit takes the new `control[commit_tokens]` instead of
  `n_emitted`. A one-column seed for a stateful drafter (now `seed()` / `enqueue_seed`) sets `n_emitted` to 1 for the
  catch-up while its token is already in the sampling history, so the documented `n_emitted` counted that token twice.
- fp8-kv-attention's decode read past the cache's buffers when the position it read from device memory put a column
  after the last row. `Cache::capacity` (rows the arrays hold) now makes such a decode, or one at a negative position,
  read nothing and write NaN rows; host positions past the capacity are rejected.
- recurrent-layer-kernels' gated-delta and Mamba-2 column scans round each token's decay once from FP64 `exp`
  instead of `expf`, which errs upward more often than downward: with slow decays the state drifted 5 to 10 times
  further from an FP64 recurrence over 8192 decode tokens than an FP32 recurrence with correctly rounded decays.
- token-sampling's README and `enqueue_commit` comment name `control[commit_tokens]`, the count
  device-speculative-round now hands the sampling history, instead of `control[n_emitted]`, which counts a seed
  round's token twice.

## [0.9.1] - 2026-10-25

### Overview

crit 0.9.1 is a speed and correctness release for the Inference kernels and Core. Core's state cache
never snapshots an unwritten slot in a chunked prefill (`final_row` rounds), gated-delta column
checkpoints cost about 1.5x the plain scan instead of 3.8x, and the stored-tile NVFP4 GEMM is 4-15%
faster at two row tiles with the same bits. Core's `ReadOnlyFile` gains reader slots so concurrent
positional reads scale on Windows, draft trees count the target token's rank per draft position to
choose tree widths from data, and `crit eval probe` records host-to-device rates per transfer size.
New tests pin the NVFP4 scale encoder, sampled speculation's distribution, greedy speculation's
equality with plain decoding, verify widths at long contexts and top-k ties. The Python distribution
and the Core SDK share version 0.9.1.

### Added

- draft-trees' `enqueue_draft_ranks` (and tree.py's `DraftRanks`): per draft position of chain rounds, the rank the
  target's token had in the drafter's row, counted on the device after accept, to choose tree widths from data.
- `crit eval probe` records `host_to_device_gbs`: pinned copy, pageable copy and mapped-read rates per transfer size,
  for engines that size host-resident weights (offloaded layers, host experts) from a measurement.
- Core's `ReadOnlyFile` has reader slots: `set_readers(n)` and `read_exact(offset, dst, worker)`. On Windows each
  slot past the first reads through its own handle reopened from the held one, since synchronous reads through one
  handle are serialised by its file object; on POSIX every slot shares the held descriptor.

### Changed

- stored-tile-gemm's NVFP4 GEMM loops over row tiles outside its token groups, reading each activation fragment
  from shared memory once per row tile, so one row tile's weight fragment is live at a time; two-row-tile launches
  (wide N) run faster and results are unchanged bit for bit.
- recurrent-layer-kernels' gated-delta column scan reads and writes the state and its per-column checkpoints
  as one 16-byte access per lane and key row (scalar when a pointer is not 16-byte aligned), so a speculative
  verify's checkpoints cost far less (one tested GPU: about 1.5x the plain scan at 8 columns instead of 3.8x).
  Results are unchanged bit for bit.

### Fixed

- crit serve closed the connection on a `DELETE`, `PUT` or `PATCH` with a body it answers 405 or 404 without
  reading that body, which on Windows can reset the connection before the client reads the answer; it now
  reads and drops the body and keeps the connection, closing it only for a body it cannot read.
- Core's StateCache took prefix-cache snapshots at page boundaries inside a `column_checkpoints` round from
  the round's per-row scratch, but a chunked recurrent prefill writes only the state after its last row, so a
  later prompt sharing an inner page attached a state the engine never wrote. `begin(seq, rows,
  RoundStates::final_row)` runs such a round's `column_checkpoints` components as `recompute` (one scratch
  slot, snapshots only at the last accepted row); `every_row` stays the default for column scans.

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
- **Eval** ([guide](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md)): one speed, quality and exactness suite that
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

[Unreleased]: https://github.com/crit-labs/crit/compare/v0.17.1...HEAD
[0.17.1]: https://github.com/crit-labs/crit/compare/v0.17.0...v0.17.1
[0.17.0]: https://github.com/crit-labs/crit/compare/v0.16.1...v0.17.0
[0.16.1]: https://github.com/crit-labs/crit/compare/v0.16.0...v0.16.1
[0.16.0]: https://github.com/crit-labs/crit/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/crit-labs/crit/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/crit-labs/crit/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/crit-labs/crit/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/crit-labs/crit/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/crit-labs/crit/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/crit-labs/crit/compare/v0.9.1...v0.10.0
[0.9.1]: https://github.com/crit-labs/crit/compare/v0.9.0...v0.9.1
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
