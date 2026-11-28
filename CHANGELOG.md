# Changelog

All notable changes to crit are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
with versions following [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.25.0] - 2026-11-27

### Overview

crit 0.25.0 is a root-architecture release: what decides how an engine runs now lives in Core, once, and is checked.
Every row of a step carries its phase (prompt or decode) as data, and every plugin reads it from the batch, never
from a row's length: the projections slot and expert-offload require it, model-template, attention-state, the
recurrence pools and the drafters route by it, and a one-token prompt chunk keeps the prompt arithmetic. Core
describes an engine before any weight is read (`crit_core.spec`'s EngineSpec, every value with its source and
plugins' prices), refuses to load an engine that is not its spec, and sizes the engine a build starts from by the
library-default spec (`crit.spec.library_spec`), which crit Eval reads. A closed engine proves it returned its
memory (`crit_plugins.lifetime`), every library plugin's tests are held to that close, and crit serve's
`--one-process` (one engine, one process) is crit run's default wherever it reloads engines. crit run gains a
preflight (`crit run ... preflight`: the library's plugins at their defaults on a cut of the run's checkpoint, in
minutes), one transition table of a candidate's states enforced at reserve, `crit diff` (where an engine leaves the
reference, layer by layer) and random checkpoint geometries (`crit models random`). Finished model plugins may hold
their family's own pieces: gemma4-model ships Gemma 4's tool calls and thought channel, shared-KV attention and its
companion MTP drafter. mma-rows fixes each matrix's summation order by a shipped shape-keyed table, so a row gets the
same bits on every GPU and tuning; the step settings are Core's (model plugins declare what they can take), and
tests/test_one_definition.py holds each calculation Core owns to one definition. crit 0.25.0 ships with crit-eval
0.9.0 (protocol crit-eval-v0.9.0, a new series: the concurrency part asks each prompt at its case's own length).

### Added

- `crit run ... preflight [--budget-minutes 15] [--gpu-lock FILE] [--plugin-tests]`: the check before an engine
  build, on the run's own checkpoint. It cuts the checkpoint into the run's `.plugin-cache/preflight/` (the first
  layer of every kind, the draft layer, the lookup tables; whole shards hard-linked), packages a candidate of the
  library's plugins at their defaults for it, and serves it in one load per quality mode as `serve --check` does,
  with the scenarios the checkpoint's facts make meaningful (`crit.preflight.plan`): a short request, a 32k prompt,
  a cancel inside a prefill, a reopen at smoke's lower context, the engine's lanes at once, a prompt-cache
  follow-up turn past the sliding window's ring, drafting from the draft layer and through the drafter seam, the
  companion drafter the run names, and a second load at the whole model's memory pressure where routed experts live
  partly off the device (a ballast sized from crit-eval's placement on the run's device). Every size comes from the
  run; prompts are sized by the tokens the engine counts. Each load takes the GPU lock (`--gpu-lock`, else
  `CRIT_GPU_LOCK`, else the new `settings.toml` key `gpu_lock`) and prints its ETA; a scenario the time budget has
  no room for is left out and says so. The report (`preflight/KEY/preflight.json` and `.md`, keyed by crit's engine
  code and the checkpoint's revision) is shown by `crit run ... show` (`preflight`) and `next`, which names the
  preflight first on a run without one for this crit and checkpoint, before its first candidate. A failed scenario
  is toolkit feedback at the library's defaults, never a gate. crit mcp offers it as `crit_run_preflight`.
- AGENTS.md, docs/BUILDING_ENGINES.md and docs/RUN_RECORDS.md describe the preflight step and its record; the rule
  that builds start only from a commit where the stress test passes is now: the release-time stress test before a
  release, the run's own preflight before a build.
- Routing conformance in `crit_plugins.testing`: `routing_grid(window, widest)` is a phase x length grid of steps
  (one-token prompt chunks, chunks of the window, past it and past 64, decode rows, verify windows of several widths,
  mixed steps), and `PluginTestCase.assertRoutesByPhase(route)` drives a device plugin over it with recording kernels
  (`route(batch)` returns each token's path) and fails where a row's path follows its length or the rows beside it
  (a size choice the facts declare within one phase may vary, `vary`). `assertProjectionsRouteByPhase` multiplies
  each step as a model does (`row_groups`) and holds a projections plugin's kernel choice to the same rule. It runs
  over attention-state's softmax and latent readers (softmax-attention's `tests/test_routing.py`), projections-dense,
  gguf-projections and, on CUDA, quantized-projections.
- docs/PLUGINS.md states the rule ("A row's phase": a plugin reads a row's phase from the batch, never from its
  length; a size choice within a phase that changes bits is declared in the facts) and the routing-conformance
  helpers.
- A closed engine proves it returned its memory (`crit_plugins.lifetime`). Core's close drops every reference the
  engine holds (its rounds, its parts' step frames, sessions and requests), collects, empties torch's cache, retires
  the step pool and measures the device against its level before the load: every elastic range, giver and captured
  graph the load made must be gone (the device now follows each captured graph by a weak reference) and torch's
  allocated and reserved bytes must be back, with what the process keeps for every engine named apart (the capture
  stream's cuBLAS workspace pool, tensors a plugin lists with `memory.process(owner, name, tensor)`, ballast ranges;
  torch's cuBLAS workspaces are freed) and no tolerance. A close that did not return it reports what holds it (the
  ranges, givers and graphs by name, torch's bytes, the tensors the plugins' modules hold and each plugin object
  still alive with the chain of references that keeps it), carries the report in the closed engine's telemetry
  (`close`) and on the device (`memory.leaking`), warns on stderr, and raises `LeakingClose` under
  `CRIT_CLOSE_CHECK=raise` (tests, checks); the next load on the device measures it again and refuses to start
  beside it with the report as its message, instead of running out of memory later. A load that fails measures
  what it made the same way. A close beside another open engine is not judged (the device's readings do not tell
  engines apart).
- crit serve's resident host reports how its last engine close went (status `last_close`: returned, judged and the
  report's message; `ServedEngine.last_close()`), so a reload, a model switch or a check reads whether the closed
  engine gave its memory back; Core's eval-v1 and chat-resident-v1 adapters' `close()` return the close's report.
- `crit run serve --check` has a `close_returns_memory` item: at its end it closes the engine in the resident host
  (crit serve's unload, as a reload or a model switch closes it) and judges that close with every reload's close
  the check made before it (smoke's and the pre-check's lower context). It fails where a close kept memory, naming
  what holds it. Every reload crit run makes (`crit run ... ablate`'s variants, the checks' lower contexts, the
  preflight's reopens) reads how the closed engine's close went (`Served.closes`) and logs one that kept memory;
  Core refuses the reload beside it.
- `crit run ... ablate`: each plugin variant's row carries how the engine served before it closed
  (`previous_close`: returned, judged, message), so a variant refused beside a leaking close points at the one
  before it.
- The preflight and the stress test reopen the engine at the context it serves (`reopen_same_context`, extra
  `same`, next to `reopen_at_64k`; every preflight's first load, the stress test's first speed load), and both
  reopen scenarios fail where a close the reopen made did not return its memory (or none was judged), naming
  what holds it: a reopen at a lower context fits beside a leak that a same-context reload does not.
- `crit_plugins.testing`'s `PluginTestCase` holds every plugin's own tests to Core's close: `assemble()` and
  `reference()` take the device's level before the plugins are made, close each assembly at the test's cleanup and
  fail the test where the close did not return its memory, naming what holds it. docs/PLUGINS.md states the rule
  ("A close returns its memory") and how a plugin names what the process keeps for every engine.
- `crit_plugins.encodings`: its E4M3 and E2M1 tables are named as the process's on the device (`memory.process`), so
  a closed engine is measured without them.
- quantized-projections: its quantizer constants, NVFP4 decode table and the buffers every matrix on a device shares
  (`_Buffers`) are named as the process's on the device (`device_matrices.process_kept`).
- grouped-experts: its decode tables (E2M1, INT4, E8M0) and the fallback's E2M1 table are named as the process's on
  the device.
- host-expert-arithmetic: its declared-order constant tables are named as the process's on the device.
- A model plugin's own techniques mount beside the models core's: code in the plugin's own folder that its manifest
  names as a source (`techniques/ID/code/KEY.py` giving `BLOCKS`, `RESIDUALS` or `DRAFTERS`) is found by the core's
  `blocks.library()` (`own_sources()`) as the library's techniques are, so the plugin's layout names its blocks,
  residuals and drafters, and a name a library technique gives already is refused. `crit.models.mounted(KEY,
  plugin=FOLDER)` imports such a technique for a test. Before, a plugin's own block went through
  `model_plugin(blocks=...)` and a drafter of its own had no way in.
- A family's own pieces have a home: a finished model plugin (`models/cache/<family>`) may hold code only that family
  uses, a model technique of its own (`techniques/ID`, mounted by its manifest) or a tool-call and reasoning syntax
  (`tools.py`), built from crit's interfaces, reading tensors and config keys through the layout and engine-tested on
  tiny random checkpoints in its two checkpoint layouts against the official implementation. Shared code still names
  no model; a piece a second family needs moves to `models/techniques` (or `inference/techniques`) under its general
  name in the same change. docs/CONTRIBUTION_CRITERIA.md ("A family's own pieces"), models/cache/README.md and
  models/techniques/README.md state it. tests/test_model_neutral.py holds it (a family's sources and code stay in its
  package or point into shared code, its techniques are its manifest's own sources with names no library technique or
  other family gives and are composed by its own layout only; no shared manifest's source lies in models/cache and no
  shared file names a family package), and tests/test_technique_reads.py reads a family's techniques' entries as it
  reads the library's. Before, the criteria kept one-family code out of crit until a second family needed it, and the
  only other place was a sealed engine the next build could not learn from: two Gemma 4 builds in two days each wrote
  the same tool format, shared-KV attention and companion drafter.
- The models core reads a companion drafter: a model plugin's layout may name a draft block whose weights are a draft
  model published in a folder of its own (`"draft": {"block": DRAFTER, "companion": MODEL_TYPE, "names": [...]}`,
  model-template's code/companion.py). The folder is the new option `draft_folder` (a run's `--drafter`), else the one
  folder beside the checkpoint (a sibling or a sibling's snapshot folder, then one level up) whose config.json is that
  model_type and that the drafter technique's `fits(config, shape)` accepts; two that fit are refused by name. The
  drafter technique reads it through `model.draft_companion` (its config and its tensors under the names rules) and is
  loaded as the model's device draft head through the existing draft-head facts (drafter_kind, draft_bytes_per_step,
  draft_width, and `draft_folder`). Before, no library model plugin could read a companion drafter folder at all.
- `crit_core.text.ReasoningFormat` takes a `label`: text right after the opening marker that names the channel the
  reasoning is written in (a chat template that opens a named channel), dropped with the whitespace around it, so a
  package's reasoning format can split such a channel without the channel's name in the reasoning. Reasoning that
  does not begin with it is kept whole.
- gemma4-model holds Gemma 4's tool-call syntax and thought channel (`tools.py`, the manifest's `tool_format` and
  `reasoning_format`): calls `<|tool_call>call:NAME{key:VALUE,...}<tool_call|>` (strings between `<|"|>`, bare
  numbers, true, false, null, nested objects and arrays) parse whole and stream their name once written, and the
  `<|channel>thought ... <channel|>` channel splits from the reply. Its tests render reasoning and calls with the
  official Gemma 4 chat template (kept beside them) and parse them back. Promoted from the 2026-11-25 build's engine
  (`gemma4_tools.py`; the 2026-11-24 build wrote the same), so a Gemma 4 engine's tool calls parse with the library's
  plugin alone.
- A layout's mixer may name the block its layers run where its own block does not serve them (`"otherwise": BLOCK`):
  a block class's `serves(shape, spec)` says why not (a config value, a fact the selection's slot plugin lacks, read
  through the Shape's new `slot_facts`: the loaded parts' at a load, the manifests' in a dry resolve, so both resolve
  the same pools), and the engine's facts name each layer kind that ran its `otherwise` and why (`mixers_instead`).
  A pool of keys a block declares keeps its declared page type (`dtype`), BF16 where it declares none.
- `crit models list` names each finished model plugin's own pieces (its techniques and its tool-call and reasoning
  syntax: `crit.models.family_pieces`, the `pieces` of each plugin row, `--json` too).
- gemma4-model holds shared-KV attention (`techniques/shared-kv-attention`, its own model technique): the full layers
  whose values are their keys' own rows (config `attention_k_eq_v`) keep one row a KV head, `[values | rotated key
  dimensions]`, read through the attention slot's latent attention, so a position takes `kv_heads x (D + rotated)`
  elements instead of `kv_heads x 2D` (Gemma 4 31B: 640 against 1,024 a head, 37.5% less KV that every decode step
  reads). The layout uses it where the config calls for it and the attention plugin provides `attention:latent`,
  else the library's attention (`otherwise`). Its engine tests serve the tiny Gemma 4 checkpoint in both layouts
  against transformers and against the plain attention path. Promoted from the 2026-11-25 build's engine and
  reworked: its reads of the attention plugin's internals (a fused launch of every KV head, a warm-up of the
  plugin's latent kernels, a free-memory threshold measured on one GPU) are gone.
- gemma4-model drafts from Gemma 4's companion MTP drafter (`techniques/assistant-drafter`, its own drafter
  technique; the layout's draft `{"block": "gemma4-assistant", "companion": "gemma4_assistant"}`): a
  `gemma4_assistant` checkpoint beside the target (or option `draft_folder`) is the model's device draft head, its
  query-only layers over the target's last sliding and full layer KV, fed the target's final-normed row; its
  `fits()` checks the backbone width, the vocabulary, the layer types and refuses ordered embeddings. Its arithmetic
  is one function (`step()`) the device head and the tests' host reference run; on tiny random checkpoints it gives
  transformers' official `Gemma4AssistantForCausalLM` logits, next row and greedy chain, with the drafter as
  published and respelled under the layout's names, and the engine finds the folder beside the checkpoint and plans
  it. Promoted from the 2026-11-25 build's engine and reworked (the folder from the models core instead of a path in
  the layout, byte counts from the formats' sizes instead of ratios).
- `crit run ... recon --drafter PATH` records a companion drafter the model plugin's layout reads as its own draft
  block (its config's model_type the layout's draft companion: kind `companion draft block`, `read_by`, `block`, one
  draft a pass, its weight and pass bytes), where before it refused every folder that is not a hidden-state drafter;
  recon.json's `family` and recon.md's "The model family's own pieces" list the pieces the run's model plugin holds
  of its own (`crit.models.companion_draft`). The preflight's `drafting_companion` drafts from such a companion
  through the model's draft head (the device default drafter, option `mtp`, option `draft_folder` at the run's
  folder) instead of hidden-state-drafters; the candidate's facts carry `draft_companion`.
- AGENTS.md and docs/BUILDING_ENGINES.md ("Where an engine's own piece goes once it works") say where a piece an
  engine wrote goes once it works: shared code where any model can use it, else the family's finished model plugin,
  proposed as toolkit feedback with the tests that hold it, so the family's next build starts from it; and that a
  build takes the family's pieces (recon lists them) before writing its own.
- One description of an engine before any weight is read, every value with its source: `crit_core.spec`'s
  EngineSpec (`engine_spec(package, model, configuration)`, schema `crit-engine-spec-v1`): the context it is built
  for, its device and host headrooms, quality mode, exactness, prompt activations, Core's cancel grace and prompt-step
  bound, every selected plugin's options and the facts the plugins resolve from the checkpoint's headers
  (kv_bytes_per_token and the rest of their resolve()), each `{value, source}` with source user, package, mode,
  default, resolved (`by` the plugin), core, measured or protocol. A configuration's `provenance` ({value: source})
  says which of its values the user set, so a default never passes as a user's choice. Core's loader records the spec
  it held the load to (telemetry `load.spec`), and `crit plugins resolve --spec` prints it.
- A value's provenance crosses from crit run to the engine and crit Eval: `crit run create` records where the run's
  KV capacity and headrooms came from (`hardware_constraints.sources`: user where the request or the machine's
  settings gave them, default for Core's, protocol for Eval's max context), a candidate's configuration carries the
  VRAM headroom's (`provenance`, which crit completes and a package's own copy may leave out), the engine's spec
  reads it, and crit Eval's result names each value's source (eval-v1 `info.spec`), so a default is not passed on as
  a user's choice (the 2026-11-25 mode-default limits).
- One transition table of a candidate's states and the operations each lets start (`crit.run.state.OPERATIONS`,
  with each state's reason in `WHY`): `next` names only steps it allows, `next` and `show` list each candidate's
  `allowed` operations, and `RunStore.reserve` and `seal` refuse every operation beside a running attempt or on a
  sealed candidate with the table's reason (an interrupted attempt is settled first, where a cancelled ablate's
  unsettled attempt once failed the next ablate's every variant as "breaks"). tests/test_run_state.py walks every
  state: each has a legal action and the table's refusals carry its reasons.
- `crit run ... show` gives each candidate's state and the operations the run's transition table allows in it
  (`candidate_states`), as `next` does.
- `crit diff --package P --model M`: where an engine leaves the reference, layer by layer. The package's engine and
  the official forward (transformers, FP32) read one token sequence as one prompt; the engine gives the residual's
  row on entry to every layer and its final-normed rows (the models core's `layer_rows()` and `layer_parts()`, a
  general tap of any layout whose residual keeps one row), the reference its hidden states at the same places. Each
  place's gap (largest row difference relative to the reference row's RMS) is judged against the 16-bit baseline (the
  reference in BF16 against itself in FP32): within 8 times it, or 1e-3, it is format-sized; the first place past it
  names the layer that made it, its kind, mixer and FFN blocks and the selection's plugins they call. crit smoke's
  decode_matches_prefill verdicts point at it. Tested on a tiny checkpoint whose one layer's FFN weight the engine reads
  tripled: crit diff names that layer.
- AGENTS.md points a build at `crit diff` where a quality check or gate fails and the place is unknown.
- `crit serve --one-process`: one engine, one process. Each engine runs in a resident host process of its own: an
  engine other than the one the host holds (another candidate, other limits, another configuration) loads in a fresh
  host, and an unload ends the host, so everything the closed engine held (its CUDA context, the allocator's caches,
  captured graphs) returns to the system with its process (`crit_api.residency.ResidentRuntime(one_process=True)`,
  `ServedEngine(one_process=...)`). The default keeps one host across reloads, whose model pool keeps the model's
  mapped files and host-side stores for the next engine.
- Under `crit serve --one-process` a kept store outlasts its host as a file: the server names a folder to every host
  (`CRIT_KEPT_FILES`, the runtime's `kept-files`, deleted when the server stops), Core's
  `LoadContext.kept_file(name, key)` names a store's file in it, and expert-tensors' kept bundle store writes itself
  there at its seal, so the next engine's host reopens it complete (direct reads at the drive's rate) instead of
  packing every expert from the checkpoint again. Elsewhere `kept_file` is None and the store stays in the process.
- A plugin's resolve() may state prices for sizing an engine before it exists (`crit_plugins.assemble.PRICES`:
  `kv_elements_per_token`, `held_matrix_bytes`, `stored_matrix_bytes`), which the EngineSpec carries as resolved
  values; they add up over plugins, and a loaded engine is not held to a price (the plan that sized its load is). The
  models core's resolve() prices `kv_elements_per_token` over the pools a sequence grows, and quantized-projections
  gains a resolve() that prices every 16-bit target matrix of the checkpoint (not a routed expert, a lookup table or a
  multimodal tower's) at its stored bytes and at the bytes load() holds it in under the head and trunk options the
  quality mode resolves (FP8 in speed): what crit Eval sizes an engine's KV and resident weights by, so a re-encoded
  matrix is no longer priced at its stored bytes.
- Random geometry (`crit.preflight.geometry`, `crit models random --seed N --out DIR`): small random checkpoints of
  sampled shapes (head widths 32 to 512, vocabularies to 300k, sliding windows, mixed layer kinds, grouped-query
  ratios, tied heads) for the dense, routed and windowed families. tests/test_random_geometry.py resolves each one's
  library-default spec and scores it through the models core against its reference on the CPU
  (`CRIT_GEOMETRY_SEEDS=A:B` draws more).
- gemma4-model: its thought channel and tool calls are tested through crit serve's reply path (ResidentChat over a
  scripted engine) with Gemma 4's own tokenizer, where the channel and call markers are special tokens, every
  chunking, streamed and whole (skipped where no Gemma 4 checkpoint is on the machine, GEMMA4_TOKENIZER).
- `crit models tiny` also shrinks the companion drafter beside the checkpoint that its model plugin reads (into
  OUT-companion, its backbone width the tiny target's; `--companion FOLDER`, `--no-companion`), and `crit models diff`
  plans a companion beside a tiny checkpoint as the engine's draft head and reports the folder it drafts from
  (`companion`).
- inkling2-loop-model: a GPU test that a request decoding one token a step beside another prompt's chunks scores every
  token with the bits it has alone, with a spy that the looped layers' decode rows took row-ops' fused gates in those
  mixed steps (block-attention, 12cae1d3). It runs where transformers has the Inkling 2 loop architecture.
- The attention slot fact `attention:latent-layers` (softmax-attention): `forward_latent_layers(layers, queries, pool,
  batch, rank)` reads several of one pool's latent layers in one call, each layer's heads with the bits a forward a
  layer gives them; where the layers lie consecutively and prompt rows read through the decode reader's fused launch,
  every layer's pieces go in one launch. gemma4-model's shared-KV attention calls it for its KV heads where the slot
  provides it (the Gemma 4 build had reached into softmax-attention's private form for this). On Gemma 4 31B's
  geometry a 512-token chunk takes 1.20x less time after 8K positions, 1.08x after 32K, 1.02x after 98K.
- tests/test_one_definition.py: a static check that each calculation Core owns keeps one definition: exactness levels
  and modes, the cancel grace, the headrooms and what fits (kv_fit, kv_need), the small engine, the step settings, the
  KV page, the decode rows and a kernel's row limit, the draft-width rule and the summation-order rules'
  multiprocessors.

### Changed

- Core's eval-v1 adapter runs crit-eval's concurrency cohort at per-request output tokens
  (`cohort(clients, max_new_tokens[c][r], sampling)`; one int still means every request): crit-eval-v0.9.0 asks
  each pool prompt its case's own request, a speed case at its 256 tokens and a held-out prompt at its 64, where
  every request was 128. The concurrency end-to-end check (`tests/core/concurrency_end_to_end_check.py`, GPU)
  gives its cases their output tokens and the serial replies their request, as crit-eval's speed part records
  them, and counts each level's tokens per request.
- The stress test's reusable code is a product package, `crit.preflight`, so a crit command can run it without
  importing from `tests/`: layer cuts (`tests/support/layer_cut.py` is now `crit.preflight.cut`, run as
  `python -m crit.preflight.cut SOURCE TARGET`, with `shards()`, `tensor_names()` and the draft prefixes a model
  plugin's layout names), the library-defaults candidate for any checkpoint (`crit.preflight.candidate`:
  `library_selection()` reads the checkpoint's facts and the finished model plugin that serves it, `build_run()`
  packages it in a run), the chat client, the report and the scenarios of a load with every size the caller's
  (`crit.preflight.scenarios.Sizes`), the GPU lock, and the report's files and key (`crit.preflight.records`).
  `tests/stress.py` is the release-time wrapper over them with its fixtures, pressures and tiers, and plans exactly
  the loads it planned before (`tests/test_stress_plan.py` pins them). Its cache key now covers `models/` too (a
  model plugin's change kept a stale run before).
- A step's rows carry their phase (`crit_plugins.StepBatch`): `StepBatch.build` takes rows of (sequence, first
  position, token count, phase), the phase `PREFILL` (a prompt chunk, however short) or `DECODE` (a decode token or
  a verify window, of any width) from `crit_core.batch`, and refuses a row without one. `prompt_row(r)` reads the
  phase and nothing else (it took a window and, where the model marked no rows, called every row past it a prompt
  row), and `prompts` is the set of the batch's prefill rows, never None. A `RoundBatch`'s rows are decode rows and a
  `PromptBatch`'s row a prompt chunk. attention-state's `pages.prompt_row` and its length fallback are gone: the
  pools and the attention read the batch's phase. model-template, the plugin-engine starter's toy-model and the
  tests' model plugins build every row in its placement's phase (a request's `decode_via_prompt` rows prefill rows).
- `crit_plugins.device.row_groups(batch, decode_rows)`: the calls a step's matrices multiply its rows in, by phase,
  kept once in Core (a prompt chunk alone however short, decode tokens in runs of the decode rows). model-template's
  `RowGroups.of` and the tests' llama-model and next-model plugins use it; theirs took every row of more than one
  token for a prompt chunk and a one-token prompt chunk for a decode row.
- The RAM headroom has one default, Core's (`crit_plugins.memory.HOST_HEADROOM`, 2 GiB): crit.settings'
  `RAM_HEADROOM_GIB` and crit Eval's take it, and tests/test_core_ownership.py holds it as it holds the VRAM
  headroom's (no copy of it under another name).
- mma-rows: the sums' order is the matrix's alone. plan()'s splits of K (each split a chain from zero, added in split
  order, in the decode tiles and the prompt tiles alike) come from `order_splits(N, K, weights, acts)`: the fewest
  whose canonical tiles of 64 matrix rows reach `ORDER_CTAS` (512) CTAs, at least two steps of K a split, at most 32.
  They no longer depend on the device's multiprocessors or the build's tuning (a tuning's SPLITS is not read), so two
  GPUs and two tunings give an engine the same bits; a swept tuning changed the splits and so the sums before
  (2026-11-25: recall 1.009 against 1.0). mma-rows' GPU tests pass (185). On an RTX 5090, decode calls of the 31B and
  27B NVFP4 checkpoints' MLP matrices and the 27B's FP8 projections took 0.88 to 1.08 times as long as with the tuned
  splits, except the 27B's W4A4 gate/up projection at one row (1.29x: its tuning had picked three splits), and prompt
  calls of 512 and 2,048 rows the same within 3% (wsl-evidence SUM-ORDER.md). Engines whose tuning picked other
  splits get other bits than before, the same on every GPU.
- Recurrent states are elastic beside an expert cache, as AGENTS.md says: the library-default selection (preflight,
  stress test) and `crit plugins offload` give sequential-recurrence `elastic: true` beside attention-state's where
  the checkpoint routes experts and has linear-attention layers, so the states of sequences that are not open are the
  expert cache's. offload's note that told the builder to add it by hand now states that it did.
- `crit run ablate --quality`: each variant's quality row carries the 16-bit baseline beside its drift
  (`quality.baseline`: the budget a plain 16-bit engine of the model uses of the same limits on the variant's worst
  set, from crit-eval's pre-check item), and the log line names it. Reported, never judged.
- `crit diff` diffs a text longer than one prompt chunk: the models core's `layer_rows()` runs the text's chunks one
  after another (each reading the earlier chunks' KV and recurrent states, as a prompt's chunks do) and joins their
  tapped rows, where it refused more than `prefill_chunk` tokens.
- mma-rows: the sums' order (the splits of K) is a shipped, shape-keyed table chosen by sweep
  (`torch-ext/mma_rows/orders.py`, `ORDERS`, keyed by weights, the MMA, N and K; written by the new
  `benchmarks/order_sweep.py`, which keeps the count whose worst ratio to the fastest over decode calls of 1 to 16
  rows and prompt calls of 512 and 2,048 rows is least, the least of three interleaved passes), the same on every GPU
  and in every tuning; `order_splits`' ORDER_CTAS rule takes the shapes the table does not name. The rule alone made
  the 27B W4A4 gate/up decode about 1.3x slower (one split, where the old tuning had picked three); with the table it
  takes three again (RTX 5090, 1 row 51.7 -> 37.6-43.4 us; the 27B W4A4 MLP layer 85.3 -> 61.9-74.2 us at 1 row),
  prompt rows unchanged within 3% and still bit-equal to decode rows. The dead SPLITS knob leaves kernel.json
  (decode-tile, fp4-tile), `tunings/12.0.json`'s configurations and the CUDA config macros.
- crit kernels: a structure may retire a knob (kernel.json `retired_knobs`: {name: why}), and a tuning written before
  it went loads with the knob dropped from each configuration (`check_tuning`), so an engine's tuning swept with
  mma-rows' SPLITS still loads; any other unknown knob is still refused.
- `crit diff`'s bound also takes the checkpoint's declared-format allowance: where the checkpoint declares the formats
  its own inference quantizes matrices' input activations to (crit-eval's `limits.schemes.activations`, the allowance
  pass's rule), a third reference pass in FP32 with those inputs quantized gives each place's `declared` gap, and a
  place is format-sized within FACTOR (8) times the larger of the 16-bit baseline's and the declared gap, or FLOOR;
  the rule is one factor over the largest gap a format the engine may use moves the reference by. `--reference-device`
  runs the reference's passes on a GPU (TF32 off) for a large checkpoint, where they ran on the CPU only.
- The recurrence pools commit a row by its phase, not its size: a prompt chunk (PREFILL) of any length, one token
  included, keeps its final state and window (accept takes all of it or none), and a decode row (a verify window) is
  committed by prefix; a decode row longer than the pool's verify window (widest + 1) is refused.
  recurrent-state-recovery's `_plan`, `delta_rows`, the chunked path's window rows and the torch convolution read
  `batch.prompt_row(r)` where they read `count > capacity`. The models core's `_pass_counts` no longer avoids chunks
  of up to `width + 1` tokens, the workaround for the size rule. sequential-recurrence's device test helper marks a
  step's rows as a model does.
- crit run's transition table enforces the phase order at reserve, for every caller: an Eval starts only after a
  passed `serve --check` (`serve_checked`, `eval_stopped` for its resume, `evaluated` for another), a measurement
  (`benchmark`) only on a completed Eval; `RunStore.reserve` asks the whole table (`refusal`, strict) where it refused
  only a running or sealed candidate. docs/RUN_RECORDS.md "States" says so. Test fixtures:
  `RunFixture.serve_checked()` and `eval_completed()` walk a candidate through the phases (`on_ladder` uses the
  first), `MeasureFixture.evaluated` leaves a served candidate ready to measure, and the tests whose candidates
  skipped a phase take it (state, attempt validation, eval, measure, actions, plans, external, serve, ablate).
- The engine a build starts from is sized by its EngineSpec before any package exists: `crit.spec.library_spec(model)`
  packages the library's defaults for the checkpoint (the preflight's candidate) in a temporary folder and resolves
  them from the headers alone, and `crit run create`'s KV capacity (`eval_max_context`, in the run's quality mode),
  recon's placement and `crit plugins offload` hand it to crit Eval (`spec`), which prices the selection's KV pools at
  its protocol's precision and the projections' held matrix bytes; the run records `context.sized_by`. Eval's reading
  of the config stays where no finished model plugin serves the checkpoint.
- Core's loader refuses an engine that is not its spec (`crit_core.spec.differences`: every resolved value as
  resolved, prices aside, and at least the context its spec asks), and every loaded engine states the spec's prices it
  was loaded under as an engine fact (`spec_prices`, published by crit serve's engine_facts), which crit Eval's
  placement reads for a served engine.
- The phase is required data on the projections slot and in expert-offload, so no long call without a phase is taken
  as a prompt chunk: `matmul(x, matrix, out_dtype=None, *, prompt)` in every projections plugin (projections-dense,
  gguf-projections, quantized-projections, the starter's toy-projections), a decode call (prompt False) past
  invariant_rows refused (`crit_plugins.device.decode_call`) where it was multiplied as prompt rows by its length;
  expert-offload's `forward(..., *, prompt)` drops its size fallback (a call past max_tokens taken as a prompt chunk).
  The slot contract (projections-dense's plugin.json), docs/PLUGINS.md's row invariance, `assertKeepsRows`, the
  starter's toy-model, the tests' fixture model plugins (which dropped the phase row_groups gives) and every plugin
  test pass the phase; the GPU tests written against the size rule say so explicitly
  (`crit_plugins.testing.long_call`, `long_route_call`).
- One engine, one process is crit run's default wherever it reloads engines (serve --check's reopens, ablate's
  variants, the preflight's reopens, Eval's lower context, through `crit.run.served.Served`): each engine loads in a
  fresh resident host, and a reopen's close is the old host's exit, which counts as a returned close
  (`ResidentRuntime.last_exit`, read by `ServedEngine.last_close` once the new host holds no close of its own; an
  unload's own report, judged by Core in the host before it ended, is kept as it was), so the preflight's
  `judge_closes` and serve --check's `close_returns_memory` judge it. `crit run --same-host` reloads in one host for
  the command. There is no measured reason against the default: the reload's cost (the model mapped again from the
  page cache, a kept expert store reopened from its file) is the end-of-cycle measurement ONE-PROCESS.md names.
- The layer localiser's tap is a Core fact any model plugin may offer: Core's engine reads the model's
  `layer_rows(ids)` and `layer_parts()` (`StepEngine.layer_rows`, `layer_parts`; docs/PLUGINS.md "The layer
  localiser's tap"), which `crit diff` calls, so a model plugin of its own, not over the models core, is localised the
  same way; one that offers none is told the contract by name.
- Kernels: every untuned rule that fixes a row's summation order counts the order's multiprocessors
  (`ORDER_MULTIPROCESSORS`, the RTX 5090's 170 the rules were measured on) instead of the device's, so a row sums in
  the same order on every GPU: paged-kv-attention's partitions, CUDA chunk, flash and CUDA prompt splits and merge
  depth, latent-attention's partitions, fixed-tile-gemm's split of K, and gguf-rows' and exl3-rows' splits. On a
  170-SM GPU every launch is the one it was, so bits and speed there are unchanged; their GPU suites pass, with new
  tests that a smaller device's count changes no rule.
- Step contract 3: a model plugin declares in its manifest what it can take of Core's step settings (`"steps":
  {"lanes": n, "pending": n}`, inherited by the plugins that extend it; model-template declares 64 and 64), and Core's
  step plan chooses lanes and waiting places within it (`Steps.capable`, which the loader calls). model-template's
  `concurrency`, `pending`, `span_seconds` and `activations_mib` default to `CORE` (-1), leaving each to Core; a
  configuration that names a value still bounds Core's choice. Under contract 2 those defaults repeated Core's values
  (4, 2, 0.75 s, 768 MiB), so Core's own never applied. Engines built with the defaults run with the same values as
  before.
- Core's load opts the process out of Windows' power throttling once, before its first engine loads
  (`crit_core.engine.full_speed`), for every caller: crit eval in process, `crit plugins memory` and the other tools
  that load an engine now keep their speed behind another window as crit serve's host did (it calls the same
  function).
- Core's `memory.kv_need` is kv_fit read the other way (the bytes an engine of n KV positions needs free); crit run's
  preflight sizes a cut's ballast by it instead of its own sum.
- Row limits have one definition each: gguf-projections and projections-dense take the slot's decode rows from Core
  (`crit_plugins.device.DECODE_ROWS`), expert-offload's GGUF experts the gguf-rows kernel's `MAX_ROWS`,
  quantized-projections' NVFP4 GEMVs `device_matrices.GEMV_ROWS`; host-expert-arithmetic's own call size is
  `CALL_ROWS`, no longer a second `DECODE_ROWS`.
- The default quality mode is Core's (`crit_plugins.exactness.DEFAULT_QUALITY_MODE`, speed): crit run's records, `crit
  plugins offload`, `crit.spec.library_spec` and crit Eval read it instead of a literal of their own.

### Fixed

- The EngineSpec prices what a sliding-window engine keeps beside its weights: the models core's resolve() states
  `kv_ring_bytes` (a new price, `crit_plugins.assemble.PRICES`: the device bytes of the sliding windows' rings, one a
  sequence the engine runs at once, whole pages, as the placement plan counts them), and quantized-projections'
  resolve() prices a tied head's copy (a checkpoint with no head tensor: the embeddings loaded through the slot as the
  head, held as the head is, beside the table). crit Eval sized the library-default Gemma 4 31B NVFP4 engine in
  speed for 200,000 tokens on an RTX 5090, where its plan needs 31.6 GiB of 28.4 free (the rings 3.2 GiB against
  the config's one window of 0.4, the FP8 head copy 1.3 GiB uncounted); it now sizes 49,152, which loads.
  `crit_models.config.fixed_state_bytes(..., sliding=False)` leaves the config's window out for a caller that prices
  the rings itself.
- crit serve: a reopen closes the engine before it loads it again, also where the spec is the one the host holds (the
  same context and configuration), which the host had kept as it was: the preflight's and the stress test's
  `reopen_same_context` reloaded nothing and failed with "the host reported no engine close". Core now judges that
  close, and under one engine, one process it ends the host as an unload does.
- gemma4-model: shared-KV attention serves a full layer only where its latent rows are a shape the attention slot's
  latent readers take on the device (rank and rotary width powers of two, rank 16 to 1024, rotary 16 to 256); other
  layers run the library's attention (`otherwise`). A checkpoint of 96- or 320-wide full-layer heads failed to load
  on the GPU (the latent kernel refused the shape in its captured rounds); found by `crit models random`'s geometries
  on the GPU.
- The stress test's and the preflight's `cancel_in_prefill` cancels a prompt no earlier request began with: it sent the
  uncancelled prompt again, which the prompt cache kept, so the cancelled request had almost nothing to prefill and
  the cancel could land after it.
- model-template multiplies a step's rows by their phase (`RowGroups.of` reads `StepBatch.prompt_row`): every prefill
  row is a prompt chunk with the prompt rows' arithmetic however short, so a one-token chunk (the remainder after a
  prompt-cache hit) has the bits of the same row in a longer chunk; it was multiplied as a decode row by its length,
  and a repeated request diverged from its first run after about 30 tokens. Decode rows of any width are cut into
  calls of the projections' invariant rows as a captured round cuts its columns (model-template builds no eager
  multi-token decode row: its verify windows run in captured rounds, so no other row's bits change).
  `decode_via_prompt` rows are prefill rows of the batch, so `RowGroups.of` and `_hidden` lose their `via_prompt`
  argument, and the MTP layer's prompt rows take their groups from `RowGroups.of` too.
- attention-state keeps a prompt chunk's keys and values fresh, and reads its own positions from them, by the row's
  phase: over FP8 and INT8 pages with the fresh buffer every prompt chunk, a step's or a captured one, however short,
  attends its own chunk as the projections gave it, where a chunk of up to the pool's `window` (16) tokens read them
  from the pages as a decode row does, so a short chunk's rows had other bits than the same rows in a longer one (the
  bits of such short chunks change; decode rows and verify windows still read the pages). The pool's `window` now
  bounds only the window rows its step buffers hold: a decode row past it is refused there. The facts' `prompt_kv`
  says so.
- attention-state's latent reader routes a step's rows by their phase: every prompt row (a one-token chunk too) through
  the latent-attention prompt reader, or with `prompt_reader` decode as pieces of the fused decode launch, and decode
  rows (decode tokens, verify windows) through the decode reader; a prompt row of up to 64 tokens went through the
  decode reader by its length, so latent prompt rows were not row-exact across prompt lengths and chunkings. A decode
  row past the decode reader's 64 columns is refused.
- block-drafter's head multiplies a prompt chunk's cache rows as one prompt call however few they are, where a chunk
  of up to the decode rows (8) took decode calls by its count; a round's draft rows keep their calls (decode calls up
  to the decode rows, one call past them: the drafter's own rows, which change no verified token).
- host-expert-arithmetic's CUDA form reads a call's `prompt` flag: with `prompt_gemm` native, W4A4 prompt rows take
  `gemm_native` however few they are, where a prompt chunk of up to 16 rows took the declared rows by its count (and
  so had other bits than the same rows in a longer chunk); `matmul` dropped the flag. Decode calls keep the declared
  rows.
- heterogeneous-expert-dispatch takes its caller's phase (`forward(..., prompt=None)`): a decode call past
  `max_tokens` (an eager step's many decode rows) keeps the declared order, where with `prompt_arithmetic` tensor it
  took the tensor arithmetic by its count, so a decode row's bits followed how many rows shared its step. expert-offload
  states the phase of every call it hands the dispatch; a caller that states none keeps the size rule. The facts'
  `prompt_expert_rows` says so.
- block-attention's looped layers gate every decode row through row-ops' fused `low_rank_gates`, in a step beside
  prompt rows too (the decode rows before the step's first prompt chunk), as a round gates them; a step that held a
  prompt row took the GEMM gates for all its rows, so a decode row's bits followed what shared its step. Prompt rows
  keep the GEMMs.
- `crit run ... plan --record` run twice with the same plan no longer records it twice: a plan whose payload equals
  an open plan of the segment (no candidate built from it yet) is refused, naming the plan already recorded
  (2026-11-25: plan-0007 and plan-0008).
- A stop plan from the newest candidate on the ladder stands where that candidate has no completed Eval (sealed
  `built` below its rung's Eval, so its plans freeze none): the stop no longer asks it to freeze the run's newest Eval,
  an older candidate's, which made the 2026-11-25 build plan from an older candidate to stop. docs/RUN_RECORDS.md
  ("States") gives the table and both rules.
- The models core keeps a copy of the first tapped row (a drafter technique's taps, the rows on entry to some layers):
  the residual may add into its stream in place once the layer has run (a fused add and norm), so a tap of layer 0
  read the stream after layer 0 rather than the embedding rows. crit diff found it (every later tap was already a
  copy, made by the cat).
- The library-default selection (`crit.preflight.candidate.library_selection`, so the preflight and the stress test)
  gives every slot a plugin that serves every layer kind the checkpoint declares: `needed_facts` reads what the
  checkpoint's layers need of the attention slot (attention:sliding for sliding-window layers, attention:latent for
  MLA, attention:sinks for stored sinks), and `serving` takes the first of `SLOT_ORDER` (partitioned-attention, then
  softmax-attention) whose manifest provides them all. A sliding-window checkpoint (Gemma 4 among them) took
  partitioned-attention, which provides neither attention:sliding nor attention:latent, so its candidate did not load;
  it now takes softmax-attention.
- Core's compile recording lasts an engine's life, not the process's: `crit_plugins.compiles.record()` returns a
  `Recording` that the engine's close (and every failed load's path) stops, and recording runs while any recorder runs
  (`compiles.recording()`). A test module that loaded an engine in-process left the hooks on for every later module,
  which failed test_compile_guard's watches test; a load refused beside a leaking close no longer starts a recorder at
  all. test_close_memory's cleanup workaround is gone.
- attention-state: a step's cached metadata (windows, pieces, prompt tables, FlashInfer plans) and the warmed launches
  are keyed by the pool's own identity (`device_attention.pool_identity`, a number no other pool of the process ever
  has), not `id(pool)`: a new pool allocated at a freed pool's address with the same step count read the old pool's
  metadata. softmax-attention's routing test no longer keeps every pool alive to dodge it.
- `crit run eval`: Eval's close of a served engine closes it in crit serve's resident host (a new host command
  `close`, `ServedEngine.close_engine`, `Served.close_engine`: the engine alone, as a reload closes it first, the
  model's pool kept) and returns Core's close report, so crit-eval's lower-context round records `context.lower.close`
  and `required_again.close` through crit serve too and fails the gate where the close kept its memory, as it does in
  process. It returned nothing before, and a served lower-context close was judged only by Core's refusal of the next
  load. Under `--one-process` the closed engine's host ends with it. An external candidate's close still reports
  nothing.
- Core's close check sets crit serve's device leases aside: the model pool lists each device allocation it keeps past
  the engines that use it on Core's device memory (`crit_plugins.memory.lease` / `unlease`, owner `model-pool`, the
  bytes the pool accounts for) until the model unloads, and `crit_plugins.lifetime` measures a closed engine without
  them (named in `torch.excluded.by`, the device report's `leased`). A plugin that leases device weights through
  `ModelContext.device` would otherwise have every close fail.
- An engine dropped without its close leaves the device: Core's device-memory watch thread holds the engine's
  `ask_step` weakly (it kept every engine alive for the process before, so a dropped engine was never collected), the
  loader finalizes each engine with `Opened.dropped`, and the next load measures a dropped engine as its close would
  have (`crit_plugins.lifetime.refuse`, `DeviceMemory.abandoned`). Its Opened stayed in `memory.opened`, so every
  later close on that device was beside it and went unjudged.
- tests/test_run_recon.py, broken since the run-plan rule that refuses the same open plan twice (84466f00) and the
  configuration's `provenance` (1a6a2a8b), which its runs did not cover: its fixture builds every candidate from the
  open plan already recorded, and the single-file model's candidate configuration expects the `provenance` crit
  completes.
- An engine dropped without its close also ends its compile recording and its compile watch at its collection (weakref
  finalizers beside `Opened.dropped`), so it leaves no hooks on for the process; the recorders are a list the module
  never rebinds (tests/test_core_ownership.py allows no new module state), and test_compile_guard's watch test sets
  aside recorders other modules' engines hold, and skips the case-studies folder (git-ignored build records) the
  ownership test scanned.
- crit run ablate blames no option or plugin for an infrastructure failure: after a variant fails it serves the engine
  as built again (`_recheck`), and where that fails too the variant's row says so (`infrastructure`) and the verdicts
  (`crit_plugins.ablation.settings`, `blame`) mark it unmeasured with that reason instead of "breaks".
- `crit models inspect` also checks the matched model plugin's own techniques against the checkpoint, except those
  whose reads are a companion drafter's (entry `"reads_of": "companion"`, Gemma 4's assistant-drafter), which would
  otherwise match the target's config by name.
- gemma4-model's shared-KV attention finds the dimensions the rotation turns with a row of ones at one radian a pair,
  so a partial rotary table over 96-, 320- or 512-wide heads (low frequencies that turn a BF16 row by less than its
  rounding at position 1) no longer refuses to load. Found by the random-geometry test.
- token-sampling and token-constraints keep no device memory past an engine's close: the native launches (and their
  empty tensors for absent buffers) are each engine's own instead of the process's, which made the next load on the
  device refuse (seen loading two Gemma 4 engines one after the other), and the round state closes a scratch it made
  for itself where no engine gave one (`close()` on both samplers). token-sampling's own GPU suite, failing 11 of its
  tests on that close check since before this cycle, passes again.
- paged-kv-attention's sliding-window timing test reads a cache four times longer (240k positions), so the full read
  costs several times the launches' fixed cost: the split family's two launches made the window's time a fixed floor
  and the test failed one run in three.
- `crit_plugins.testing`: before a test's assembly is held to its close, the test's attributes set since the assembly
  was made (its plugins, their pools and tensors, which the test object keeps past its run) are dropped, so a GPU test
  that keeps `self.kv = parts["kv_cache"]` no longer fails the close check on its own references (19 of
  softmax-attention's GPU tests and token-sampling's staged tests did).
- paged-kv-attention and latent-attention keep no device tensor of an engine past its close: the dummy for unread
  pointers and the ragged prompt plans are the current owner's (`owner()`, both modules) and go with its `release()`;
  attention-state's device form and KV cache call every module function as their own (`OwnedKernels`, also for
  latent-attention and the pools' writes) and release at close. The dummy, first made inside a capture, held its graph
  pool's 2 MiB after the engine closed, and the next engine of the process refused to load (seen on Qwen3.8 27B and in
  the attention plugins' GPU tests).
- A tensor the process keeps for every engine (`DeviceMemory.process`) is one tensor a name: asked again under the
  same name (another package's copy of a plugin's code, whose module cache asks for the process's buffers again) it
  returns the one kept, and takes a callable that makes it. quantized-projections' shared buffers, quantizer tables
  and pinned GEMMs' alpha and beta, the encodings' tables and host-expert-arithmetic's constants use what it returns.
  Each engine of a new package kept another 64 MiB workspace the process never freed, and the second engine's close
  counted the first's copy (seen loading two Qwen3.8 27B packages one after the other).
- FlashInfer's own process caches (the ALiBi slopes its prefill run makes whether a call reads them or not, its cached
  workspaces) are named as the process's after each FlashInfer call (attention-state), so the close check of an engine
  that read through FlashInfer no longer counts them; softmax-attention's FlashInfer test drops its patched plan at
  its end.
- model-template's KV pools take the kv_cache plugin's page (its `page` option) for the pools it makes, their page
  reservations and the plan's pricing; it passed its own `PAGE` (16) to every pool, so the plugin's option never
  applied. `PAGE` stays the size where a kv_cache plugin states none.
- `crit plugins offload`'s host headroom defaults to settings.toml's `ram_headroom_gib`, else Core's `HOST_HEADROOM`
  (`crit.settings.ram_headroom_bytes`), as every other command's does; it had its own 2 GiB literal.
- `crit_plugins.testing`: a test's close retires the device's step pool after the plugins close, as Core's engine
  close does, so a GPU test that ran steps through the device's step frames no longer fails the close check on its
  last decode pool (expert-offload's elastic-cache test failed so since before this cycle).

### Internal

- `crit run eval`'s comment and `crit pull --bench`'s log line say that the reference's first scoring of new data also
  measures the 16-bit baseline (crit-eval's `score_data` does by default), not only the scores and the allowance. No
  behavior changes.

### Removed

- device-speculative-round: the technique's host width policy (`width_policy.py`, `WidthPolicy`) is retired: Core's
  round driver chooses the draft width per round (width policy `rate`, on the device inside the captured rounds), the
  one rule every engine runs. No engine on crit-hub or in crit-engines selects or imports the copy (checked); its
  README, entry and mtp-decode's README point to Core's.

## [0.24.0] - 2026-11-26

### Overview

crit 0.24.0 makes crit's own kernels faster and ships their tunings. A kernel folder may now carry the tunings
sweeps measured on one compute capability (`tunings/<capability>.json`), which a build takes after the engine's own
tuning; `crit kernels ship` merges a sweep into it, and mma-rows ships the RTX 5090's, merged from 19 sweeps. On an
RTX 5090, every output's bits unchanged: mma-rows' prompt tile splits K across CTAs, gives a warp two row groups and
stages three sub-steps (1.94x at 16 rows, 1.04-1.16x at 2,048 and 8,192), fp8-block-rows' staged prompt GEMM is
1.16-1.39x faster from 512 tokens on the kernel core's reworked declared mainloop, grouped-experts' FP4 prompt layers
run `fp4-tile` by default on a warp-specialized gate/up mainloop (1.29-1.55x over 0.23.0, beside FlashInfer's fused
MoE), and paged-kv-attention's fused prompt reader takes its faster form (short steps 1.3-3.6x). `crit kernels rank`
orders the kernels by their share of a trace, so a sweep starts where the engine spends its time. The Gemma 4 31B
speed build's fixes land: closing an engine frees its captured rounds (a same-host reload at the same context ran out
of memory), kept prompts survive sliding-window rings and resume only where every pool still holds them, a score's
logit slices and the device plan's matrix bytes are sized right, short prompt chunks read through the prompt reader,
and `crit run` passes crit Eval only the quality limits the user set, so a checkpoint's stored activation allowance
applies. `crit run` gains `ablate --quality`, `record-routes` and `switch-check`, and the README carries the
Qwen3.8-Flash-Next case study. Core's engine ABI stays 10. crit 0.24.0 needs crit-eval 0.8.2.

### Added

- Shipped tunings (`crit_kernels.shipped`): a kernel folder may carry `tunings/<capability>.json`, the tuning tables
  sweeps measured on GPUs of one compute capability, keyed by each structure's shape key (never a model), each entry
  naming the sweep it came from (the kernel source digest, the GPU, the capability, the day). On a GPU of that
  capability a build takes, per structure and shape, the engine's own tuning first, then the table, then the default
  knobs, so an engine whose tuning lacks a shape still gets the table's entry; on any other GPU nothing changes. A
  plugin's record says `tuned` and names the table under `shipped`; `load_kernel(..., tuned=False)`, the sweep's
  untuned build, gates' and `crit kernels report`'s default builds take no table. `crit kernels sweep` leaves out the
  shapes a table covers (it logs how many, and writes their entries and report rows into its tuning and report,
  marked shipped), builds nothing where every shape is covered, and sweeps them all with `--no-shipped`; its tuning
  now records the day it was swept. `crit kernels ship ID --tuning SWEEP/ID.json` merges a finished sweep into the
  table for its capability (an entry swept on the current kernel source wins over one from an earlier source, else
  the later sweep), dropping entries whose knobs kernel.json no longer has; a tuning without its provenance is
  refused. A table that fails `check_tuning` is ignored with a warning, and a CPU test checks every shipped table
  against its kernel.json.
- mma-rows ships a tuning for compute capability 12.0 (`tunings/12.0.json`, the RTX 5090's), merged by `crit kernels
  ship` from 19 earlier RTX 5090 sweeps: 84 decode-tile and prompt-tile shapes, 15 fp4-tile and 12 fp4-prompt-tile.
  An engine on such a GPU whose own tuning lacks a shape gets the table's entry instead of the default knobs.

- `crit kernels rank`: crit's kernels ranked by their share of a kernel trace's time (a kernels report from `crit
  eval kernels` or `crit run ... profile`), per phase and over all, from the launches each `kernel.json` now declares
  (`launches`: a crit namespace with `::`, or a name as written, read without template arguments and anonymous
  namespaces; `crit_kernels.launches`). It names a launch two kernels declare, one none declares and the kernels that
  launched nothing, and tests/test_launches.py holds every kernel's launches to its sources. AGENTS.md sweeps from the
  top of the ranking: a build had swept the 11 kernels its plugins load, decode-gemv among them, which launched
  nothing, and paged attention for 45 minutes at 3.6% of the time, where its trace ranks expert-directory 40.6%,
  mma-rows 26.7% and grouped-experts 15.0%.
- `crit run ablate --quality` (with `--data`): the first round scores every variant's Eval quality texts against the
  reference as the gate's pre-check does, at the run's quality mode and budget with the segment's amendments, and each
  variant's row keeps its state, the budget it used and its worst set, so an option that approximates the model is
  judged by its drift beside its speed.
- `crit run ... record-routes`: a candidate's routing trace over a general corpus, as a command. It serves an unsealed
  candidate under a check attempt with the plugin of its selection that records routes (`record_routes`,
  `trace_path`), answers each prompt of the corpus once, closes the engine, which writes the trace, and keeps the trace
  as the attempt's evidence, what `crit plugins offload --trace` replays and a residency profile is built from.
- `crit run ... switch-check`: the switching check AGENTS.md asks for after sealing, as a command. It serves the
  candidate (the run's selected one by default) through crit serve's engine and chat API in the process, switches to
  another sealed candidate (the newest by default) and back with a greedy reply each, writes `work/switch-check.json`
  and fails where the candidate's replies before and after differ.
- The `crit smoke` verdict that a decode gap is rounding names the plugins whose prompt rows run another arithmetic than
  their decode rows (the engine's `exactness.plugins` below exact), the only places such a gap can grow in, and says
  how to find the one (`crit run ablate --smoke-test decode_matches_prefill --set`, each in turn on one arithmetic).
- `crit run ablate` says a single pass that judged a change is not confirmed and asks for `--rounds 2`, and AGENTS.md
  asks the same of a change the next plan acts on: an engine that warms slowly (an expert cache filling, a drafter's
  acceptance settling) moves by more than the 3% noise between passes (one build read +12% decode, then 76.2 against
  76.3 tok/s).
- `crit run serve --check`'s `step_memory`, where a step outgrew Core's step arena, names the most each kind of step
  took (prompt, score, decode), largest first, so the kind that outgrew it is told without a rerun.

### Fixed

- `crit run create` with an explicit `kv_tokens` and a `device` file records the device file
  (`hardware_constraints.context`, source `given`), so the run's Evals price the prefill ceiling and the TTFT floor
  from its rates; it was dropped, and such Evals reported no ceiling and no prefill fraction.
- `crit_plugins.ranking.top_ranked` finds a row's ties at its k-th value from its k + 1 highest values and gathers its
  candidates from the rows in place, so no temporary is wider than a bool mask of the rows: the int64 count of the
  masks (256 MiB for 128 rows of a 262,144-token vocabulary) and a copy of the rows were a long score's allocations
  past the engine's headroom. The ranking is unchanged, bit for bit.
- attention-state reads a prompt chunk of up to its `window` (16) tokens through the prompt reader, as it reads a
  longer one, where the model marks it a prompt chunk (`StepBatch.prompts`, which model-template's row groups set):
  it read such a chunk as window rows, in the decode's reduction, so with `prompt_reader` prefill (speed's
  `row-exact` prompt rows) a short prompt's rows did not have the bits of the same rows in a longer prompt, which
  that level promises. Each reader matches float64 attention at head widths 64 to 256; a model that amplifies
  rounding (a 31B model with sliding layers of 256-wide heads) showed the two reductions 4.1 nats apart on its
  short prefixes, and an engine took the decode reader for every prompt to avoid it.
- The prompt cache resumes no kept sequence before where every pool still holds it (`resumes_from`): a sliding
  window's ring that wrapped (a reply past the ring's spare positions) keeps only its last pages, so a follow-up that
  shared less than its window back from there read positions later ones had written over.
- model-template prices kept prompts against a full KV pool, and where every KV pool is a sliding window's ring
  leaves eviction to the pools: it priced them against the first pool, a ring, so a kept 8,192-token prompt (512
  pages against the ring's 270) was evicted as soon as it was kept and every follow-up turn was prefilled whole.
- model-template's device plan prices each target matrix stored in 16 bits at the bytes the projections slot holds
  it in (quantized-projections' new `held_bytes`: quality mode speed's FP8 trunk and head, its `trunk_keep` roles, its
  row padding and the shapes it keeps as stored), and counts a tied head, a device matrix of its own beside the
  embedding table: it priced them at their stored bytes and the tied head at none, so a speed engine with BF16
  attention or head was planned at up to twice their bytes, and one whose table the plan put off the device was
  short of its head. A projections slot without `held_bytes` is priced as before.
- latent-attention launches with knobs this device's shared memory holds: where a structure's knobs (its defaults, or
  a tuning made on a GPU of more shared memory) do not fit, it steps them down, a pipeline stage and then half the
  query rows a program, to the first that fits, once a geometry (`FITTED`); neither is an order knob, so a token's
  bits are unchanged. At rank 512 with a 128-wide rotary key the window tiles at the defaults took 110,784 bytes
  against sm_120's 101,376, and every launch failed. A sweep's selected configuration is launched as it is, so one
  that does not fit is still dropped.
- model-template's score slices its logits by bytes (`score_slice`: at most 128 rows, fewer where their two FP32
  copies would pass 128 MiB: 64 rows at a 262,144-token vocabulary) and caps logits without a copy of their own. A
  slice's peak, measured on the GPU, was 8 bytes a logit, 12 with a logit cap: 384 MiB for 128 rows of Gemma 4's
  capped 262,144-token vocabulary, now 128 MiB; models up to a 131,072-token vocabulary keep 128 rows. Scores are
  unchanged, bit for bit (a row's logits are its own whatever the slice).
- `crit_kernels.build.select` refuses a configuration index past a build's configurations, naming them (the
  structure's defaults first; a `configs=` entry equal to them is not built twice), where the kernel's first launch
  raised a bare IndexError.
- `StepEngine.close()` drops its native rounds after closing them, so the round driver's captured graphs and their
  memory pools free with the close: they stayed reserved (4.2 GiB for Gemma 4 31B at 200,000 tokens) until the engine
  object went, and crit serve's same-host reload at the same context ran out of memory.
- `crit run ... eval`, `crit serve --check`'s pre-check and `crit run ablate --quality` pass crit Eval only the quality
  limits the user set (the request's `quality_budget` with every resume's amendment, `records.user_quality_budget`),
  none when the user set none. They passed the quality mode's defaults as the run's own budget, and a limit passed as
  the run's wins over the checkpoint's stored allowance (crit Eval's protocol quality.allowance), so a speed run of a
  checkpoint with quantized activations was judged without it. Eval records the same resolved thresholds as before,
  so `checks --from-eval` and plans compare as they did. The agent brief passes `--quality-budget` only when the user
  set one.

- `crit run ... eval --baseline` takes a candidate's id (its newest completed Eval result) or a result's path in the
  run folder, as crit prints them (`candidates/ID/eval/ATTEMPT.json`), where it read a path from the working
  directory only and such a baseline lost the Eval to "JSON file not found".
- `crit run ... plan` asks candidate 1 to start with the CPU expert tier only where recon priced the CPU on a probe
  that measured its expert lane and the turn to the host: without them the CPU's floors are lower bounds, and the plan
  check took one (18.3 ms against the link's 30.7 on a 6-core host behind WSL2) for a win the ablation then measured
  21% slower. RUN_RECORDS.md's plan rule 16 says so.
- `crit run ablate`: an interrupted ablation settles its attempt and keeps the variants it served. A variant
  interrupted while served left an empty entry the report could not combine (IndexError), the attempt never settled,
  and every later serve of the run was refused until its cleanup was settled by hand.
- `crit_plugins.ablation` reads an option variant that failed where the engine as built failed too as unmeasured, with
  the reason, not as the option breaking the engine.
- device-expert-cache refuses a residency profile of another routing geometry naming both geometries and what a group
  counts (a profile over 48 routed layers against a cache of 49 groups, the draft layer's experts among them), where
  the message named neither.
- expert-offload writes the recorded routes to `trace_path` when the engine closes too, where only a `telemetry()`
  call wrote them, so a serve stopped after its corpus left no trace (a 10-minute recording over 32 prompts wrote
  nothing).
- recurrent-state-recovery keeps the units that hold another sequence's prompt-row final states backed until their
  accept when a sequence closes: an elastic pool trimmed them, and under concurrent requests the accept copied from
  unbacked memory.
- token-constraints' captured rounds compute in the engine's scratch, which Core closes with the engine; they made a
  scratch of their own that outlived it, and a reopened engine found token-sampling's scratch still held.
- `crit kernels sweep decode-gemv` runs again: gemv-cuda's bench calls the weight formats and INT4 helpers that the
  alias structures' retirement took out of decode-gemv's bench (a NameError). They live in one module now,
  `crit_kernels.weight_formats`, which mma-rows' bench reads too (its inputs unchanged), and tests/test_bench_shapes.py
  finds every structure's shapes on the CPU.

### Changed

- mma-rows' sweep no longer builds prompt-tile configurations that compile to the same kernel: PR 2 always loads the
  weights at the step, so PR 2 with PF 1 is the kernel of PR 2 with PF 0 and only that one is swept (504 to 420
  configurations). PK, the parts of K a launch runs in, is a launch argument of both prompt tiles, now declared a
  run-time knob of their benches, so a build's consecutive configurations share their compiled kernels: 60 distinct
  prompt-tile kernels and 16 fp4-prompt-tile kernels (of 112 configurations) instead of one a configuration.
- The kernel core's `block_sum_e4m3_fast` takes the block-scaled MMA only where `CRIT_KCORE_FAST_E4M3_SUM` lists the
  target (sm_120a), and the plain e4m3 MMA elsewhere: its bits are measured, not specified, and a new device test
  (`kernels/core/tests/fast_e4m3_tests.cu`) compares the two sums bit for bit over 9.4e8 blocks (every code byte,
  random codes with NaNs and infinities, large, subnormal and cancelling blocks); it passed on an RTX 5090, where
  nothing changes. sm_121a takes the plain MMA until the test passes there.
- fp8-block-rows reads a matrix stored once in the visit order (`to_visit_weights`, `matrix_table`'s order `VISIT`, a
  record field `order` of fp8-block-rows' and decode-gemv's tables): `rows`, `gemm` and `gemm_staged` give the plain
  layout's bits, the native GEMMs refuse such a table. On an RTX 5090 the staged GEMM is 1.0-1.53x faster at 32 and
  128 tokens (24576 x 4096 at 32 tokens 230 -> 150 us) and 0.97-1.0x from 512, the decode rows 0.66-0.90x, so the
  plain layout stays the default; quantized-projections' new `fp8_block_layout` option (default `plain`) takes it for
  fp8:block matrices on the staged prompt GEMM.
- grouped-experts' fp4-tile (FP4 prompt items of 64 routes) runs gate/up on a warp-specialized mainloop (new knob
  `PRODUCER`, 0: warp-specialized for gate/up, the block's cp.async ring for down): a producer warp issues a block's
  copies up to STAGES steps ahead, the weights' codes as TMA boxes of a 3-D tensor map over the store, the scales and
  routes by cp.async on the slot's mbarrier, and the consumer warps release slots on mbarriers instead of a block
  barrier a step; every route's bits unchanged. On an RTX 5090 gate/up passes are 1.02-1.36x faster and FP4 prompt
  layers 1.02-1.20x (512 experts top 10 at 512 tokens 1,350 -> 1,121 us beside FlashInfer's fused MoE 1,098; at
  2,048 1,484 -> 1,317 against its 1,330). The kernel core gains `make_tensor_map_3d`, `tma_load_3d` and
  `cp_async_mbarrier_arrive`.
- paged-kv-attention's fused CUDA launch (the exact prompt reader) runs its serial form from 125 percent of the
  programs the device holds at once (`FUSED_SERIAL`, cuda-fused's `SERIAL` 0), where it took it from 50: below that
  the ring form's programs a split fill the GPU better. On an RTX 5090 over 56 prompt steps of 64 to 1,024 tokens the
  faster form is now taken within 2.6% on average (9.8% before), short steps 1.3-3.6x faster (one 512-token prompt
  over 2 KV heads of 256: 121 -> 91 us; 128 tokens at 16,384 cached over 8 heads of 128: 1,013 -> 279 us). The same
  bits either way.
- expert-directory's bench (`bench.py:Fill`) times the fill from mapped host memory (pinned, at its
  cudaHostGetDevicePointer address), as an engine's store is read, and its yardstick over the same mapping; it timed
  the fill from device memory, 15 to 22 times faster than the link allows, so a sweep ranked grids by a speed no
  engine sees. The README records the fill against the link on an RTX 5090: 25.6 to 26.7 GB/s at every grid from 32
  to 256 blocks and every load width, the copy engine's rate from the same memory (faster for one to four bundles),
  four blocks enough to hold it, and what a fill on a side branch costs the work beside it (+1.6% at 4 blocks, +45%
  at 85).
- fp8-block-rows' declared staged GEMM (`gemm-staged`, the exact prompt path of FP8 block checkpoints) is 1.16-1.39x
  faster at its defaults on an RTX 5090 from 512 tokens and 0.94-1.66x below (12288 x 5120 at 2,048 tokens 1,428 ->
  1,082 us), every output's bits unchanged: the kernel core's declared mainloop plans its visits once a CTA and copies a visit at once,
  the weight rows land in the MMA's operand order, a CTA copies its distinct weight scale rows once and shares the scale
  products of rows on one, each token's blocks of a visit lie side by side, and quantized-projections quantizes the
  prompt rows into the GEMM's new visit order (`quantize(..., order=VISIT)`, `gemm_staged(..., order=VISIT)`: a token's
  blocks of a visit one contiguous piece, which L2 serves at twice the rate); its default tiles are 64 x 128 (128 x 32
  to 48 tokens, 64 x 64 to 96), and on sm_120a its block sums run on the block-scaled MMA with unit scales (the plain
  MMA's bits, measured). int-block-experts' staged GEMM runs on the same mainloop, 0.98-1.09x.
- grouped-experts' FP4 prompt layers run fp4-tile by default (the plugin's `prompt_items` auto without a sweep report),
  with three staged steps, its fragments read by ldmatrix and 4 row tiles a warp for the down pass: on an RTX 5090
  a 512-expert 2560 x 640 layer at 2,048 / 8,192 tokens 1.29x / 1.55x faster than before, a 128-expert 2048 x 768 one
  1.39x / 1.51x, every route's bits unchanged.
- mma-rows' prompt tile runs a tile's splits of K in several CTAs (knob `PK`; partial sums and counters given to
  `mm_prompt`, which quantized-projections passes): part 0 adds its splits' chains, every later split's chain is
  stored on its own and the tile's last CTA adds them in split order, so every output keeps the decode tile's bits.
  Untuned, it takes the fewest splits a CTA that still fits one wave of resident CTAs; `prompt_ctas()` reports a
  call's CTAs. Over both test checkpoints' matrices on an RTX 5090 the tile is 1.94x faster at 16 rows (geomean), 1.30x
  at 512 and level from 2,048; a 336 x 10240 FP8 matrix's 16-row prompt went from 3 CTAs and 164 us to 96 and 26 us.
- mma-rows' prompt tile gives a warp two 16-row groups (knob `PR`; its default a rule from the grid's waves, K's steps
  and the weight format), each staged B fragment read once for both and each step's weights loaded at the step
  (knob `PF`): on an RTX 5090, against 0.23.0's prompt tile, 1.11-1.16x on FP8 and MXFP8 weights and 1.04-1.05x on
  BF16 and NVFP4 at 2,048 and 8,192 rows (the 248320-row FP8 head 1.13x), the bits unchanged. Its parts' default no longer splits short-K matrices or
  a BF16 matrix's weight stream, where it had been up to 15% slower than one CTA a tile.
- mma-rows' FP4 prompt tile runs a tile's splits of K in several CTAs too (knob `PK`, its own kernel instance):
  1.05-1.7x on a 320 x 10240 NVFP4 matrix's 16 to 256 prompt rows; quantized-projections passes it the partial sums
  and quantizes its prompts past the scratch into the workspace.
- mma-rows' prompt tile stages three sub-steps of activations with one barrier a sub-step (knob `PS`, default 3; 2
  is the former two buffers): 1-6% less time on BF16, FP8 and MXFP8 weights from 512 rows, NVFP4 level.
- quantized-projections' W8A8 and W4A4 prompt rows on the prompt tile quantize into the shared 64 MB workspace once
  they outgrow the 4 MB scratch, instead of a call per scratchful (192 rows at K 10240) that read the matrix again
  each time: a 2,048-row prompt of a 336 x 10240 FP8 matrix took 1,756 us, now 213 us.
- The README lists crit's own kernels among what crit solves (templates from a shared core, tuned on your machine for
  your GPU and model) and carries the Qwen3.8-Flash-Next case study: a 117 GB mixture of experts on the 32 GB RTX 5090
  from crit's library as released (crit-labs/ranked-expert-cache-mtp3).
- AGENTS.md: FP8 prompt activations are the rung between a 4-bit format's stored ones and BF16, and quality variants
  are compared against Eval's budget spread in either quality mode.
- crit Eval's reference measures the checkpoint's activation allowance by default (crit-eval 0.8.2), which `crit run
  eval` passes on: a speed run of a checkpoint that declares quantized activations is judged against it.

## [0.23.0] - 2026-11-23

### Overview

crit 0.23.0 makes model support a pillar of its own, beside Inference and Kernels. `models/core` holds the models
core (`model-template` and the new `crit-models` package, which now owns a decoder's mechanisms that Core's config
reader held), `models/techniques` the blocks, residuals and drafters an architecture is composed of, and
`models/cache` the finished model plugins, one an architecture. The new `crit models` lists them, inspects a
checkpoint against them, builds tiny random checkpoints and diffs a plugin's engine against its reference, position
by position. Core moves to `inference/core/` and the Inference entries to `inference/techniques/CATEGORY/ID`. A
guard keeps crit's shared code model-neutral: no model name in it, every technique declares what it reads, and each
one is engine-tested under two checkpoint spellings. The GGUF reader now reads each architecture from its layout's
data. Quality mode `speed` starts from 8- and 4-bit arithmetic wherever it pays (16-bit matrices in FP8, FP8 KV,
NVFP4 experts on their own tensor cores, row-exact prompt rows), and decode speed is now always sampled decode. The
Inkling-2-Mini work lands in the library: a `block-drafter` model technique, the `row-ops` kernel template, the
fused row operations, grouped-experts' `fp4-tile`, and the model's reasoning and tool-call formats. Its rebuilt
engine's certified Eval at crit-eval-v0.8.1 decodes at 196.7 tok/s sampled on one RTX 5090 (0.22.0 quoted about 46
greedy), and the README carries it as a case study, with a caution that the case studies were built on a budget.
The host test suite and CI got much faster, and `tests/run_parallel.py --changed` runs only the tests a change
touches. **Breaking:** an engine sealed with the 0.22.0 library is rebuilt from the current library. Core's engine
ABI stays 10. crit 0.23.0 needs crit-eval 0.8.1, and the release publishes `crit-models` beside `crit-core`.

### Added

- Model plugins can name a reasoning format that repeats (`ReasoningFormat` repeated: reasoning in messages of its
  own that a reply opens again after text), and the splitter reads each opening as more reasoning. A thinking budget
  holds past the first close of such a format (the opening marker biased out, or short steps that close reasoning
  opened again), and a reply continuation reads its own prompt and reply from the prompt cache.
- A `ToolFormat` may name its call before its opening marker (`lead`). The lead is held back while it may start a
  call, dropped with a parsed call, and kept as written otherwise.
- inkling2-loop-model names its reasoning markers (`<|content_thinking|>` ... `<|end_message|>`, repeated) and its
  chat template's tool-call syntax (`NAME<|content_invoke_tool_json|>{name, args}<|end_message|>`) as the plugin's
  `reasoning_format` and `tool_format`.
- Each model technique's entry declares what its code reads (`reads.tensors`, config keys), held both ways by
  tests/test_technique_reads.py (an AST pass over the code; `--write` regenerates), which CI's lint job runs beside
  the model-neutral guard.
- native: a plugin callback that raises prints its traceback to the host log, where the failure message alone did not
  say where it came from.
- paged-kv-attention: a test that a CUDA prompt chunk in one range gives a prefix the same bits at every chunk length,
  with and without a distance bias.
- `python -B tests/run_parallel.py --changed [BASE]` runs only the test modules the working tree's change touches
  (HEAD by default; a branch name adds this branch's commits since it left that branch), and `--fast SECONDS` leaves
  out the units recorded as longer and lists them: the loop while a change is being written, where the choice was
  the whole suite or a guess. [tests/select_tests.py](tests/select_tests.py), which the runner's docstring named and
  no checkout held, chooses them: the modules that import a changed file, followed through the tests' own modules
  and crit's packages from the source (imports inside functions included), and the modules that name a changed
  library item (a plugin's, technique's or kernel's folder, loaded by path) or file; a changed file no test module
  imports or names is listed. A change to one technique's code selects 5 modules, to `crit_plugins/disk_cache.py`
  1, to the run store 49 of 141 (most serving tests build a run). The whole suite stays the check of a finished
  change, and CI runs all of it.
- Model technique `block-drafter` (`models/techniques/drafters/block-drafter`, a layout's `draft: {block:
  block-drafter, prefix, config, keys}`): a drafter stored beside the checkpoint, a few bidirectional layers that fill a
  whole block of drafts in one pass over their own cache of the target's tapped rows, and a recurrent head that orders
  the block's drafts, folded through the model's head; the model's device draft head in Core's captured rounds and
  captured prompt chunks. The models core asks a drafter technique through one interface (`DRAFTERS`: `stored`,
  `Class(model, prefix, spec)`, `taps`, `spare`, `weight_bytes`, `device_bytes`, `head`). Its local layers keep only
  the positions a block pass reads, a doubled ring of window + block positions, so its cache does not grow with the
  context.
- grouped-experts: structure `fp4-tile`, FP4 prompt items of 64 routes through a block tile beside `fp4-prompt`'s
  32-route items (the same bits), and option `prompt_items` (`auto`, `fp4-prompt`, `fp4-tile`): auto takes the
  faster of the two in the engine's sweep report on this device for the call's shape and tokens, fp4-prompt where
  there is none; the facts record what each FP4 prompt call ran and why (`prompt_items`).
- Kernel template `row-ops` (`kernels/techniques/norms/row-ops`): the row operations between the model techniques'
  products (head clips, small products, sink-gated outputs, scaled adds, summary mixes, pair norms, gated joins with
  low-rank gates, span attention), every launch constant a knob `crit kernels sweep` tunes (order knobs looked up by
  the shape alone), benches that read shapes from standard config keys and tensor shapes, and tests against FP64
  references the techniques own (their `operations.json`, `code/reference.py`; `crit_kernels.contracts` reads model
  techniques' operations too). relative-attention, block-attention and block-drafter call it through
  `model.kernel("row-ops")` (declared by the models core; `kernel()` now falls back from the model plugin's
  declarations to the core's) with their torch forms as fallback; the Triton row programs with fixed block sizes that
  sat beside their code (`code/fused_rows.py`, `template/fused.py`) are gone. The defaults are those programs'
  constants.
- A guard that crit's shared code is model-neutral ([test_model_neutral.py](tests/test_model_neutral.py), in the host
  tests and crit's lint job): no model family name or model_type in shared code (Core, Inference, Kernels, the models
  core and techniques, the CLI, API, Eval, MCP, smoke, tools; comments included, JSON prose excepted), only listed
  identifiers a published format or library defines; none of the tensor spellings a models/cache layout renames; and
  every model technique composed by a model plugin whose engine tests serve two checkpoint layouts on tiny random
  checkpoints (each plugin's `tests/checkpoints.json`: transformers' safetensors, the model under another spelling
  that names rules map back through `crit_models.testing.Respelled`, which every plugin has, its `names` rules added
  to the served copy of the layout for the test, and GGUF). It allows no
  names but the identifiers a format or library defines, listed with their fields in
  `tests/support/model_neutral.json`. Core's layering test keeps its plugin-name rule and leaves the family names to
  it.
- Model plugins as their own pillar, built like Inference and Kernels: `models/core` (the models core,
  `model-template`: the step model, the serving contracts, compose() and the checkpoint readers; and `crit-models`),
  `models/techniques` (the 12 blocks and residuals an architecture is composed of, each an entry with what it
  provides, the sources it imports and the tensors and config keys it reads) and `models/cache` (finished model
  plugins, one an architecture: `llama-qwen3-model`, `qwen3-5-model`, `qwen4-exp-model`, `deepseek-v3-model`,
  `gemma4-model`, `inkling2-loop-model`). A model plugin is a folder of its own: a `plugin.json` that fills `model`
  for the config model_types it names and extends the core, a `layout.json`, a `plugin.py` of the core's three
  functions over that layout (`compose.model_plugin`), a class only for a layer part no technique computes, and
  tests that serve a small random checkpoint against its reference. `model-template` fills no slot and names no
  model type: an engine selects the model plugin that serves its checkpoint (`crit plugins list --model`, `crit
  plugins add ID`, which brings the core), and the core mounts the techniques and finds them by their registry
  (`blocks.library()`), so its code holds no block list and no family's layout. A selection that named
  `model-template` in the model slot names the model plugin instead, and sets the core's options under that id; the
  option `layout` is gone (a layout lies in its model plugin).
- `crit models`: `list` the techniques and the finished model plugins; `inspect MODEL` (the checkpoint's fingerprint,
  the model plugin that serves it, the techniques that read its config keys and tensors through the plugin's rename
  rules, and what no technique reads); `tiny MODEL --out DIR` (a random checkpoint of the architecture from its
  reference's own modeling code: sizes shrunk, layer kinds, loops, windows and routing kept); `diff --model TINY` (a
  model plugin's engine through Core's loader against the reference, position by position; `--plugin
  ENGINE/plugins/ID` serves the one being built from its engine's plugins); `cache add ENGINE/plugins/ID` (keeps a
  model plugin a run built in the user's cache, so `crit plugins list` shows it and `crit plugins add ID` puts it
  into any engine with the core and library plugins it uses: an architecture is built once; `--library` writes it
  into a checkout's `models/cache`) and `cache check`. Also MCP tools (`crit_models_*`).
- `crit_models.testing`, the harness a model plugin's tests run on: `Manifest` (its manifest and layout on the
  host) and `Engines` / `Layout` (whole engines of the plugin against transformers, CUDA or CPU), for a finished
  plugin and for one an engine is building alike. `crit_models.plugin.layout(folder)` reads and checks a model
  plugin's layout without a checkpoint.
- A plugin that fills no slot is a core other plugins extend (`"fills": []`, no `entry`): no engine selects it, and
  `crit plugins list` shows it as `core`. `crit_plugins.assemble.plugin_package(manifest)` gives a plugin's package
  without its factory.
- A folder of plugins names the folders it is used with (`plugins.json`: `{"with": [...]}`), so one library spans
  `inference/techniques`, `models/core` and `models/cache`; `crit plugins add` places a models plugin at its id and
  code from `models/` under the engine's `_sources/models/`.

### Changed

- README: the Inkling-2-Mini case study ("Looped MoE, fits VRAM") is filled from the rebuilt engine's certified Eval
  (`block-drafter-fp4-experts`, crit-eval-v0.8.1): 196.7 tok/s sampled decode, 8,036 tok/s prefill, 141 ms to first
  token, 129 tok/s decode at 200,000 tokens. Case study titles name the model kind, whether it fits VRAM and the
  architecture. A caution says the case studies were built with a limited candidate and time budget, that they do not
  compete with established bespoke engines, and that more candidates and time make a better engine.
- serve --check: on a package that ends reasoning at a budget, the check's chat requests carry a 16-token reasoning
  budget, so a model that reasons first still writes the reply they compare. crit-eval's gate pre-check items are
  hints, as crit-eval documents them: reported and marked, never failing the check (the Eval gates judge them).
- block-drafter reads no checkpoint's spelling and taps any residual: its tensors and config under its own names,
  which the layout's rename rules map a checkpoint's onto. The models core keeps a drafter's tapped rows for every
  residual that gives its running row (pre-norm, sandwich and block-attention), so block-attention keeps no taps of
  its own. inkling2-loop-model's layout leaves transformers' own names as they are, so a checkpoint loads under either
  spelling.
- The model-neutral guard closes its holes: a family name counts at a word start whatever follows, inside a word
  where no English word holds it, and in a family's short names; it keeps no recorded debt, so any model name in
  shared code fails.
- tests/durations.json: Linux times for the four classes that run the library plugins' tests and for LibraryTests
  without them (from a run of the whole suite, 6 jobs), Windows' time of the old LibraryTests removed (it would
  count that class four times too long; the classes fall back to Linux's times until a run on main records
  Windows'), and the two lint guards, no longer units, gone.
- `crit_models.testing`: the engine test of prompts run layer by layer uses prompts of 40, 44 and 20 tokens (3, 3
  and 2 chunks of 16, the last partial) and a scored text of 70 (5 chunks), where it used 75, 79 and 35 (5, 5 and 3
  chunks) and 150 (10); the cancelled prompt stays 200 tokens (13 chunks), and each step of its open pass waits until
  the cancel has been asked (looked for every 10 ms), so the cancel lands inside the pass on any host. A pass's steps are its chunks times the model's
  layers, so the longer prompts ran the same calls through every layer more often and no other ones: several chunks
  with a partial last one, passes of several requests beside decoding rows, the score and the cancel inside a pass
  are all still there. The case in the eight classes of tests/test_model_template.py that run it on the CPU: 98.5 s
  -> 84.7 s together (6 jobs, same load); less than the 30 s expected, since two engine loads are most of it. Every
  model plugin's own tests run the case too.
- tests/run_parallel.py runs the short classes of a module the recorded times split several a process (`grouped`:
  the classes recorded as 15 s or less, joined in their order up to 60 recorded seconds a process), where each
  started a process of its own: a class that runs for a fraction of a second, or skips where there is no GPU, cost
  the process's start and its imports (4.4 s for a tests/test_model_template.py class that ran nothing). Every shard
  joins them alike from the same recorded times, a joined unit's time is recorded in equal parts for its classes,
  and its line names them. tests/test_model_template.py, test_api_dialects.py, test_run_plans.py and
  test_plugin_library.py with their importers, 6 jobs, same load: 75 units -> 27; test_model_template's units' time
  333 s -> 242 s.
- tests/test_plugin_library.py runs the library plugins' own tests in four classes, a quarter of the registry each
  (`LibraryPluginTestsA` to `D`), where one test of LibraryTests ran them all: the longest unit of the suite, which
  no shard could end before, went from 121 s to 63 s (6 jobs; the four parts took 25 s, 50 s, 61 s and 63 s side by
  side, more time in sum than the one run because each part runs its plugins' tests four at a time). Every library
  plugin's tests run as before, each in exactly one part.
- The API tests keep one served engine a test class where a test leaves it as it found it (`ServeFixture`'s
  `shared_engine`, set by the fixtures of tests/test_api.py, test_api_options.py, test_tool_stream.py,
  test_api_dialects.py, test_api_completions.py, test_api_eval_extensions.py, test_api_logprobs.py, test_api_score.py
  and test_api_tokenize.py): starting a resident host and copying the run were half of an API test's 0.8 s. A test
  gets the engine of the test before only when that test left the run's folder untouched (every file's size and
  time, and no new file in the runtime folder, where a failed request leaves its records) and the host is the same
  process with its first load still loaded and idle; otherwise the engine is closed, the run copied again and a new
  one opened, so a test that crashes, retires, reloads or switches the engine, or writes to the run, costs the next
  test nothing but the time. Each of the 72 serving test classes ran with it in four orders of its tests (as
  written, reversed, two shuffles): the classes of the fixtures above pass in all four, 31 of them reusing engines;
  the host-lifecycle classes (tests/test_chat_residency.py, test_chat_stream.py, test_serve_shutdown.py, test_api's
  CommandTests) fail or gain nothing with it and stay as they were. The 15 modules that import those fixtures, 6
  jobs, same load, with the entry below: 57 s -> 39 s (their units' time 341 s -> 233 s). One class-level error
  showed once in a forward run under load and in none of 46 repeats; the class's temporary folder is now removed
  as far as it can be when a host is still ending.
- The kernels workflow builds a kernel for every architecture once for what its build reads: each build job has a
  key over the kernel's folder, the kernel core and crit_kernels, Core's Python packages, crit's record reader, the
  techniques' operations and CUDA sources, this workflow (the toolchain's versions) and the runner image, and a job
  whose key passed before does nothing (a note in the Actions cache, kept only when the job succeeds). On the
  0.22.0 pull request every push built all 17 kernels again, 15 to 17 minutes a run with decode-gemv alone 11 to 12;
  a push that changes none of what a kernel reads now skips its build. A pull request reads the notes of its own
  earlier runs and of main. Not run on GitHub yet: the workflow parses and the list step's script gives each
  kernel its key on a checkout; the first run builds everything and the second shows the skips.
- CI keeps uv's package cache on the drive Python is installed on (the runner's tool cache folder), so uv links the
  pinned packages' files from the restored cache instead of copying them: on Windows the temporary folder it was in
  is on another drive, and installing the 39 packages took 53 s of each job's 71 s of setup from a cache that had
  every one of them. Expected: about 40 s less on each of the 11 Windows jobs that use the setup action. Not run on
  GitHub yet: if the drives are not as read from the runner's paths, the install copies as before.
- CI runs the host tests with Python 3.11 on Windows on pushes to main only: a pull request runs Python 3.14 on
  both platforms and 3.11 on Linux. The four Windows shards of 3.11 were 2,191 s of the 6,893 s of runner time of
  the 0.22.0 pull request (32%). A break that only Python 3.11 on Windows shows is found on main, after the merge.
  Not run on GitHub yet: the workflow parses, and the first pull request shows its 8 host jobs (4 Windows, 4 Linux).
- tests/test_api.py: eight of ApiTests' 18 cases no longer run again in StructuredOutputTests, RequestOptionsTests,
  LegacyOptionsTests (tests/test_api_options.py) and AdvancedToolTests (tests/test_tool_stream.py): 32 runs, 34 s of
  the suite's recorded time on Linux. They are the cases no declared capability reaches (`ServerCases`): a port
  another server holds is refused, the models list, a candidate that does not verify, priority reaching admission,
  the 404 endpoints, an engine failure and its reload, the request id on every response, and abort. Those four
  classes still run the ten cases that build a request crit serve checks against the package's capabilities
  (`ApiCases`: completion, length finish, streaming, stop sequences, include_stop_str_in_output, request timeout,
  rejected requests, a null n, a context overflow, a later developer message) beside their own. ApiTests, ApiKeyTests
  and FitOutputTests run all 18 as before.
- The two guards CI's lint job runs as scripts, tests/test_model_neutral.py and tests/test_technique_reads.py, are no
  longer units of tests/run_parallel.py, so CI's host jobs do not run them a second time on every platform and Python
  (21 s of unit time on Linux a host run, four host runs a pull request); `--only test_model_neutral
  test_technique_reads` runs them through the runner. The lint job runs them on Linux with Python 3.14 only, where
  the host jobs also ran them on Windows and Python 3.11: both read the checkout's text and no platform's code.
- tests/test_serve_end_to_end.py: 45 s -> 11 s. Its 8 tests send requests to one engine the class builds
  (`setUpClass`), where each built a run, measured and sealed a candidate and loaded the engine for 7 s first; the
  follow-up turn's test, whose first request must find nothing cached, still builds its own.
- The host tests' own process does not sync the records it writes to the disk (tests/support/runs.py makes
  `os.fsync` return at once there): crit syncs every record, thousands of times in a test module, which was 40% of
  the run store tests' time on Linux, and no test can tell a synced file from a written one. The crit processes the
  tests start sync as they do. tests/test_run_plans.py 32.3 s -> 17.2 s, tests/test_run_records.py 17.4 s -> 10.1 s
  alone; the 45 modules that build a run, 6 jobs, same load, with the entry below: 167 s -> 139 s (their units' time
  999 s -> 834 s).
- CI fills the host shards by the unit times main's last run recorded, not only by the tests/durations.json someone
  last committed: the durations job keeps its merged times in a cache, a `times` job restores them once a run and
  hands them to every shard as one artifact (so all shards of a run split the units by the same times), and
  `tests/run_parallel.py --times FILE` lays them over durations.json's. A unit added or grown since the file was
  committed counts at its own time from the next run on; without a cache the shards fill by durations.json alone.
  The host jobs start after that job (about ten seconds). Not run on GitHub yet: the workflow parses, and the first
  pull request shows whether the cache and artifact steps do what they say.
- tests/durations.json holds the unit times main's CI run of crit 0.22.0 recorded (261 units; Linux 2,599 s, Windows
  5,703 s), where it held those of 0.21.0 (222 units; 1,623 s and 3,100 s). The 39 units it did not know counted as
  20 s each when the shards were filled, tests/test_checkpoint_manifests.py (325 s on Linux, 646 s on Windows) among
  them: on the 0.22.0 pull request one Windows shard ran 913 s and the other three 281 to 308 s, one Linux shard 562 s
  and the other 231 s.
- tests/test_checkpoint_manifests.py: 342 s -> 47 s on Linux. The dry plan (tests/support/dry_plan.py) takes the
  shapes only of `crit_plugins.device.csf_decode` on tensors without bytes, as it does of the encodings: the decode's
  dozen tensor operations cost 3 ms a call on meta tensors, and a checkpoint with compressed scales decodes one per
  expert matrix (74,000 calls, 110 s of that checkpoint's 112 s plan; its arithmetic is tested on real values by
  device-expert-cache's tests). The two tests that read the same checkpoint's plan share it. Every plan reads the
  same tensors as before (374,988 of that checkpoint). In CI this one module took 7 to 12 minutes and set the pace
  of the whole run.
- The serving tests' fixture (tests/support/serve.py `ServeFixture`) builds its run once a test class: the run, its
  two measured and sealed candidates and their packages took a second and a half of records, hashes and fsyncs before
  every test (three quarters of an API test's time), and are now built by the class's first test and copied for each
  test, the first included, to the same path (the run's records name their model by its path). Every test still
  starts from the same untouched run with its own store, runtime folder and resident host; what the fixture's
  construction exercised is `crit run`'s store, which tests/test_run_*.py test. test_api.ApiTests alone: about 27 s
  -> 14 s. The 21 serving test modules together, 6 jobs, same load before and after and with the two entries above:
  203 s -> 109 s (their units' time 1,213 s -> 646 s). `RunFixture.setUp` is split into `patch_ladder()` and
  `create_run(root)` for it.
- crit serve's HTTP server looks for its shutdown every 50 ms (`Server.serve_forever`), where socketserver's half
  second had every stop of a server wait a quarter second on average for nothing: crit run's serve steps, crit pull's
  bench, and each of the several hundred host tests that start one (test_api.ApiTests alone, 18 tests: 33.9 s ->
  about 27 s).
- tests/run_parallel.py runs every unit, and the processes it starts, on one math thread (`OMP_NUM_THREADS=1` unless
  the caller set it). torch started a thread per core in each of them, so units side by side fought over the cores:
  a model-template class that takes 19 s alone on one thread (36 s on six) took 245 s among five other units. Measured
  on a 12-core machine with 6 jobs, same load before and after: test_model_template, test_plugin_library,
  test_hybrid_engine, test_core_engine and test_device_engine together 974 s -> 114 s (their units' time 5,275 s ->
  493 s), and the library plugins' tests no longer run past their 600 s limit there.
- paged-kv-attention: the CUDA flash prompt chunk takes a distance bias (`prefill(..., flash="cuda", bias=)`, FP32
  [tokens][heads][extent], in a library variant of its own: `B=1`), and attention-state reads biased prompt rows and
  captured prompt chunks through it with prompt_reader prefill, where it took the decode's split-KV reduction in
  pieces of 64 tokens. A biased prompt's scores move by the reduction order (on the engine it was measured on,
  prompt NLL 0.6819 / 0.2384 -> 0.6798 / 0.2356 nats over two 4k prompts); a bias of zeros keeps every bit of the
  chunk without one, and prompt_reader decode keeps the old path.
- quantized-projections: under trunk fp8, a 16-bit target matrix whose rows alone the FP8 kernels do not take (not a
  multiple of 16, as a product stacked with a few rows of another projection) is padded with zero rows and encoded,
  where it stayed BF16; its products are its own rows (a view), each output row the bits it has among the padded
  rows. A matrix whose columns FP8 does not take stays as stored under `trunk_kept`.
- crit's shared code names no model: comments and messages that named one as evidence or example describe the shape
  or kind instead (measured numbers keep their source in docs); the GGUF pre-tokenizer table names its regexes by what
  they split and holds whether a file that does not say adds BOS; fused-norms' benchmark shapes are named by what they
  hold. The only names left are identifiers a format or library defines (GGUF tokenizer.ggml.pre and
  tokenizer.ggml.model values, the llama3 rope type, FlashInfer function names), listed with their fields in
  `tests/support/model_neutral.json`.
- A model plugin's layout says how a GGUF of each general.architecture it reads maps onto its names (`gguf`: `parts`,
  `outside`, `draft` and `constants`, the GGUF names to the techniques' names as data; `experts`, the routed experts'
  prefix; `transforms`, the converter's steps with their tensors and config keys: zero-centred, negated-exp,
  squeezed, permuted-rotary, tiled-heads, joined-rows, joined-columns, split-rows, split-heads), where the models
  core's GGUF reader kept a table of architectures and each family's part tables and code paths. The reader keeps
  only GGUF's common convention and the generic steps; `open_checkpoint` and `GgufCheckpoint` take the layout, and
  crit_models.testing's GGUF writer builds a file from the same entry.
- Quality mode `speed` starts from 8- and 4-bit arithmetic wherever it pays, for every model; `preserve` is unchanged.
  quantized-projections gains `trunk` (`auto`, `stored`, `fp8`; default `auto`): in `speed` every target matrix stored
  in 16 bits, not only the head, is re-encoded to FP8 rows at load (facts `trunk_encoding`, `trunk_kept`), but the
  roles `trunk_keep` names (default `router,shared_gate,index_qk`). model-template's `kv` `auto` takes `fp8:row` in
  `speed` where the checkpoint stores no static KV scales, and `mtp_encode` and `mtp_kv` default to `auto` (`speed`:
  NVFP4 matrices, `fp8:row` pages). grouped-experts' `activations` and `arithmetic` default to `auto` (`speed`: NVFP4
  experts W4A4 at their input scales, each layer on its format's own tensor cores, `fp4` or `fp8`, where the GPU and
  the layer take them, else `tensor`; facts `experts.layers.L.arithmetic`), and expert-offload's device rows take the
  same arithmetic. A `speed` run whose `weights_policy` is `as_stored` sets `head` and `trunk` to `stored`
  ([precision](docs/PLUGINS.md#precision)).
- Quality mode `speed` asks `row-exact` rows where a configuration states no level (`preserve` still asks `exact`),
  so every plugin takes its faster prompt path: quantized-projections' pinned prompt GEMMs, the split reader's prompt
  launch, and for NVFP4 experts without input scales `fp4-row` prompt rows in grouped-experts and expert-offload.
  crit smoke judges such an engine's decoding against its prefilling by the drift tolerance; an engine that fails
  `decode_matches_prefill` there asks `exact` prompt rows ([exactness](docs/PLUGINS.md#exactness)).
- Decode speed is sampled decode only. `crit run measure` (and `ablate`, `profile`, `kernels`, which send the same
  requests) decodes a captured request that names no temperature at crit Eval's sampled-suite sampling for the run
  (recon.json's `gathered.eval_sampling`, else temperature 1.0), each measured repetition with its own seed where the
  engine takes one, instead of at temperature 0; a captured temperature-0 request's benchmark carries
  `extensions.comparison_only`. `crit run ablate` decodes data prepared without a sampling at temperature 1.0, not 0.
  A plan's frozen decode and `crit pull bench`'s decode are the sampled suite's only, never the greedy one's, and
  AGENTS.md, the build-engine skill and the docs state the rule; the README's case study no longer quotes a greedy
  decode as a speed.
- Core's eval adapter's `cohort` takes per-request sampling (`sampling[c][r]`, the fields the engine applies kept),
  which crit-eval-v0.8.0's concurrency levels time their sampled run with.
- A decoder's mechanisms leave Core's config reader for `crit-models`, a package of its own built from
  `models/core/python` as `crit-core` is from `inference/core/python`: `crit_models.config.architecture(config)` reads
  the layer kinds, looped spans, routed experts, latent attention, linear-attention and Mamba-2 state and the rotary
  sections over Core's `ModelConfig`, with the rules over them (`attention_shape`, `attention_geometry`,
  `layer_kinds`, `kv_elements_per_token`, `fixed_state_bytes`, `routed_layers`, `layer_applications`), and a family's
  key spellings are its `config_keys.json`. Core keeps the generic fields, the position limit and the head size.
  crit Eval's limits and the model plugins read the mechanisms through it (checked identical on every checkpoint
  config and model on hand: 318 values). A GGUF folder's check of what a mechanism adds (attention by layer kind,
  experts, rotary base) is crit-models', registered with Core (`crit_core.registry` gguf_checks). crit, crit Eval
  and a checkout's tests install or put `models/core/python` on the path beside `inference/core/python`; the release
  publishes `crit-models` beside `crit-core`.
- A layer part's slot and its facts are defined by the slot's base plugin, in its manifest's `defines` (host
  interface, device form, comparison cases, facts and their pools; `crit_plugins.contracts`), not by Core:
  attention and its `attention:*` facts (softmax-attention), projections (projections-dense), experts and
  `experts:*` (experts-reference), expert residency and source with their policy (expert-tensors), router
  (topk-router), recurrence (sequential-recurrence), norms (torch-norms), embeddings (memory-table-rows). Core keeps
  the engine's own slots (model, placement, KV pool, sampler, drafter, verifier) and its round and prompt forms. A
  folder of fixture or engine plugins knows the layer parts' slots through the plugins that define them; a host
  plugin's policy slots are checked against the registry's definitions; tests take a slot's cases with
  `crit_plugins.testing.cases_for(__file__)`.
- Core moves to `inference/core/` and the Inference entries to `inference/techniques/CATEGORY/ID` (attention,
  experts, memory, norms, projections, recurrence, sampling, speculation; an entry's `category` is its folder), each
  slot's plain plugin in its category beside them (no `base/` folder), and the walkthrough to `inference/starters/`,
  so inference is core, techniques and starters as kernels is. A folder of plugins holds plugins and groups of
  them, marked by `plugins.json`; `crit plugins add` copies a plugin to `ENGINE/plugins/CATEGORY/ID`. **Breaking:**
  an engine sealed with the 0.22.0 library or earlier is not supported on this crit (its plugin copies carry no slot
  definitions, and its model plugin asks Core's config for the mechanisms that moved to crit-models); it is rebuilt
  from the current library. Python package names stay. The
  wheel ships the library as `crit/library/inference/techniques` and `crit/library/models`.

### Removed

- tests/test_reusable_techniques.py: the class GgufBlockReferenceTests, which held five reference digests and no test
  since the function they were computed from left the tests (crit 0.5.0), is gone.
- `models/` and `crit models` (PTQ, weight codecs, calibration, drafter distillation and their transform records), with
  their MCP tools (`crit_models_*`) and the `crit_drafter` drafter kind. No engine build used them: a `transformable`
  run re-encodes stored matrices as the engine loads them. Test checkpoints that need NVFP4 experts encode them with
  `tests/nvfp4_checkpoint.py`, bit for bit the codec it replaces. `models/` now holds model support, and the new
  `crit models` (Added) is its command.

### Fixed

- crit run plan: a parent sealed `eval_failed` (its Eval stopped before completing) is planned from as its build.
  Before, `next` asked for a plan that refused the parent, and the run could not go on.
- serving: a reasoning marker the splitter did not consume (a closing marker that also ends the template's other
  messages) stays out of the content where no tool parser reads it.
- model-template: a score's eager prompt chunks fit the device memory free when the score runs (the spans' bytes a
  row, the headroom kept); prompt rows keep their bits at any chunk boundary.
- quantized-projections: a matrix stored without a 16-bit weight (int4's `qweight`) has no FP8 target, so its shape
  is not read; trunk padding read `tensors["weight"]` first and failed.
- block-attention: the dry shape (`resolve`) asks for no kernel, which only the loaded model has.
- Lines that a lost backslash continuation had joined are split again (crit run's plans, crit_models' layer passes,
  paged-kv-attention, model-template, relative-attention, softmax-attention's kernel test and six host test modules),
  and imports the lint job (ruff F401) refuses are gone.
- Tests: token-sampling's and token-constraints' staged-form tests skip where Triton is not installed (a CUDA host
  without it, such as Windows); model-template's GGUF tests skip in an engine package, which carries only its own
  model plugin.

## [0.22.0] - 2026-11-21

### Overview

crit 0.22.0 makes Core the owner of every decision that shapes an engine, and moves crit's kernels onto the formats'
own tensor cores. core/README.md's "What Core decides" names the values that are Core's (the exactness levels and
quality modes, the engine's policy, scheduling, prompt-step budgets, the step's memory, the cancel grace, what fits
the device and the VRAM headroom), and a test fails wherever one is set outside Core. Core orders waiting requests
by priority and caps the requests an engine holds, allocates every byte an engine's steps take, asks the plugins what
the engine will be before it loads, watches what compiles once the load is over, and decodes constrained requests in
its captured rounds instead of one Python step a token. grouped-experts multiplies NVFP4 and FP8 experts on their
tensor cores as exact decode/prompt pairs, quantized-projections loads FP8 block checkpoints as stored, every kernel
operation has an executable FP64 contract checked in CI, and plugins choose among structures by what the sweep
measured. The exact prompt path is fast: on an RTX 5090 an exact 8,192-token prompt chunk through attention takes
5.7 ms instead of 11.1, and recurrent-state-recovery's opt-in `decode: chunk` gives decode, verify and prompt rows
the same bits (Qwen3.8-27B's exact prompt 44.8 ms instead of 312.7 at 8K tokens). model-template composes a looped
decoder, and the released Inkling-2-Mini runs on an RTX 5090 at about 46 tokens a second greedy. A model plugin can
name its tool-call and reasoning syntax. `crit run ... profile` shows where an engine's time goes per token, `ablate
--rounds` measures a 1% change, and the dev tools are faster: serve --check reopens the engine once, the kernel sweep
builds less, native code is cached by content and plugin tests run in parallel. Core's engine ABI is 10: rebuild
`crit_core_steps`. crit 0.22.0 needs crit-eval 0.7.3.

### Added

- model-template composes a looped decoder: the `looped-blocks` layout for Inkling 2 loop checkpoints
  (inkling2_loop, inkling2_loop_text). Each pass of a looped layer is a layer of its own (its own KV and convolution
  windows) over the checkpoint layer's weights and experts, loaded once (Core's `layer_applications()`); the
  `block-attention` residual (attention residuals over blocks of layers, later passes restarted from the row before
  the loop and gated by messages from the pass before) and the `relative-attention` block (a relative-position bias a
  token computes for itself, learned sinks a token, L2-clipped queries and keys, short and dynamic convolutions, a
  gated output). Scored against transformers 5.24 on a small random checkpoint, its greedy choices, batching, seeded
  sampling and layer-by-layer prompts as every layout's (tests/test_model_template.py `LoopedTests`). Parts it needed:
  Core reads a looped decoder's config (sliding layers listed by id, `sliding_window_size`, `route_scale`, a family's
  own layer type names, layer passes counted in the KV and state estimates); `crit_core.model.schemes` reads NVFP4
  whose factor is an amax over a denominator and experts stored as stacks (MXFP8 too, gate and up rows interleaved);
  a layout's `names` rename a checkpoint's own tensor names, `head_rows` trims a padded head and `expert_caps` asks
  the experts for a capped SwiGLU (`experts:activations` softcapped_silu, in the FP64 reference and experts-reference);
  softmax-attention's `attention:distance-bias` (`forward(..., bias=)`), on paged-kv-attention's split and fused
  window launches and attention-state's torch readers; grouped-experts computes the capped SwiGLU (its kernel's
  activation 2); MXFP8 experts reach an experts plugin without MXFP8 bundles as their exact BF16 values. The released
  Inkling-2-Mini runs on an RTX 5090 (42 s load, 22.6 GB, about 46 tokens a second greedy). Not yet: MXFP8 bundles in
  grouped-experts, the capped SwiGLU on expert-offload, the checkpoint's draft layer and vision tower.

- Core's scheduler orders waiting requests by priority, and the cap on the requests an engine holds and how long one
  waits for a lane are Core's runtime settings. crit serve ordered its queue by `priority` in front of Core, whose
  runtime admitted first in, first out, so the requests an engine held ran in arrival order whatever their priority,
  and crit Eval's in-process engines had no such ordering at all. `RequestOptions::priority` (the native ABI's
  `"priority"`; `StepEngine.generate(priority=)`) admits a lower value first, the oldest first among equals, of the
  generations queued before the first score; a score stays a barrier and takes no priority
  ([request priority](inference/core/src/runtime/README.md#request-priority)). `StepEngine(max_concurrent_requests=)` caps
  what the runtime holds at once, lanes first, then waiting places (`crit_core.native.cap_concurrency`), and
  `pending_timeout` bounds the wait for a lane (the step engine's new `pending_timeout_ms` setting). Without one a
  request waits as long as it takes: Core's profile default of 30 s failed a request that waited behind long
  generations ("pending request deadline expired"), though nothing had asked for a deadline. Core's engine ABI is 10:
  rebuild `crit_core_steps`. Tests: tests/core/concurrency_tests.cpp's `admission_priority`,
  tests/test_core_scheduling.py.

- What Core decides is written down and tested. Core's README said what Core holds no knowledge of (models,
  devices) but not what it owns, and the values that decide how an engine computes and schedules had copies in the
  CLI, crit serve, crit_kernels and crit Eval. core/README.md's "What Core decides" gives the rule (a decision is
  Core's when crit serve and Eval would get another engine or another schedule if it were made elsewhere), what Core
  decides (the exactness levels and quality modes, the engine's policy, scheduling, prompt-step budgets, the step's
  memory, the cancel grace, what fits the device and the default VRAM headroom) and what stays out (settings files,
  crit serve's HTTP queue, the GPU lock, the Windows throttling opt-out, placement advice, model-family rules).
  tests/test_core_ownership.py reads crit's source and fails where a value Core owns is assigned or spelled out
  outside its Core module, where Core imports crit_kernels.exactness, or where Core's Python keeps an engine's policy
  in module state (`global`); its allowlist (tests/support/core_ownership_allowlist.json) counts today's exceptions
  and may only shrink. The charter's first rule and the contribution checklist point to it.

- `crit run ... profile` says where a served engine's time goes per token. Engine authors profiled by hand; it serves
  the candidate, a plugin package or an external engine, with the kernel profiler the run's device has (crit-eval's
  tracer: CUPTI, or rocprofiler-sdk on HIP), traces one short request of the run's case and prints per prompt and per
  decode token the GPU time, share and launches by kernel family (matmul, attention, recurrence, norms, elementwise,
  routing, copies, other; crit's own kernels among the names), with the decode step against recon's plain-pass floor.
  Families come from kernel names and the rules `--families` adds; the names none classes count as other and are
  listed (docs/RUN_RECORDS.md). An engine served through `crit serve` now gets crit's own environment variables for
  its host process, as an external engine did.

- `crit kernels report` states each speed row's share of the device's floor (`of_floor`, crit_kernels/floors.py): the
  call's bytes at the memory bandwidth or its multiply-adds at its format's tensor rate, whichever is longer. The
  ceilings come from a `crit eval probe` result (`--probe`) or are measured at the report's start (the faster of a
  1 GiB copy and a read, a cuBLAS GEMM a format). Benches state `flops()` with their arithmetic (mma-rows,
  fixed-tile-gemm, grouped-experts); a row without one is reported unpriced.

- `crit_kernels.select`: a plugin picks among structures by what the sweep measured instead of keeping times in
  code. `fitting(operation, asked, rows)` gives the structures that compute an operation at a level and row count,
  `times()` a sweep report's tuned and library times at a shape, `fastest()` the fastest of them with the plugin's
  own rule where no report covers the shape, `crossover()` the last row count one option is no slower at.
  quantized-projections' `prompt_gemm auto` (crit's native GEMM or the provider's, per matrix) and its GEMV/tile
  crossover take it.

- quantized-projections loads FP8 block checkpoints as stored (format `fp8:block`: E4M3 in b0 x b1 blocks of FP32
  scales, DeepSeek-V3's 128 x 128) on crit's fp8-block-rows: W8A8 against activations per 128 inputs, decode rows on
  its declared rows, prompt rows on its declared staged GEMM (each row the bits of the row decoded) or, with
  prompt_gemm native or auto, its tensor-core GEMM. The plugin states the formats it loads, and model-template hands
  `fp8:block` to a slot that lists it instead of refusing it.

- `crit_plugins.kernels.plugin_folder(module, default)`: the folder of the plugin whose package imported a module.
  Code a plugin borrows from another technique (copied to an engine's `_sources` without a plugin.json) loads its
  kernels as the importing plugin, which declares them.

- kernel.json `retired`: a structure that leaves a kernel names the one that replaced it and how its knobs map, and a
  tuning written before is migrated into the new structure's table when the kernel is built
  (`crit_kernels.build.retire`, applied by `check_tuning`). `host_include` builds a kernel with Core's host-memory
  headers (`crit_plugins.host.INCLUDE`), as `core_include` does with Core's public headers.

- recurrent-scans' `conv_segments_rows` (structure conv-segments-rows, exact: pool-conv's bits): several long rows'
  causal convolutions in one launch, each from its own window; recurrent-state-recovery runs a step's prompt rows'
  convolutions through it (one launch a layer instead of one a row).

- latent-attention's `prefill_rows`: several sequences' prompt chunks in one launch of the prompt kernel (a tile map
  and a row of metadata a sequence), each token with the bits `prefill` gives it in its own launch; attention-state's
  latent path reads a step's prompt rows through it (it was one launch a row and layer).

- The CUDA crit builds outside kernels/ (Core's host handoff) compiles for every architecture crit's kernels claim in
  CI, within the register and spill limits (`python -m crit_kernels.outside`, the kernels workflow's job "CUDA outside
  kernels/"), with no GPU: it was built on one GPU only. The plugins' CUDA that was checked there too (token-sampling's
  law, device-expert-cache's directory, the expert routes' CPU share) moved into kernels/ (see Changed).

- recurrent-state-recovery's `decode` option (`scan`, the default, or `chunk`): with `chunk` every row of the
  recurrence (decode, verify, prompt, captured rounds and captured prompt chunks) runs recurrent-scans' chunk-aligned
  exact form from each sequence's last 64-aligned state (code/chunk_states.py: the boundary state and the chunk's
  tokens a sequence and layer, commits by device tensor work, so a captured round commits in its graph), and prompt
  rows have the bits of decoding their tokens on tensor cores: exact prompt rows. Each decode step recomputes its
  chunk so far (Qwen3.8-27B: +0.84 ms a token at one sequence), so it stays an option an engine build can ablate:
  it wins only for exact prompt rows when answers are short and prompts long (at one sequence an 8K prompt's saving
  pays for about 320 decoded tokens, a 2K prompt's for about 80; the plugin's README has the rule).

- The kernel core's radix select pieces (`crit_kcore/radix.cuh`, `crit_kcore/radix.py`): `radix_key` (a float as a
  uint32 key of the same order, a NaN lowest), `radix_float`, `digit_from_top` (a pass's digit, the running total from
  the top in bin order) and `warp_digit_256` (one warp's search of a 256-bin count histogram). sparse-indexer's CUDA
  and Triton selects and token-sampling's full law take them instead of their own copies, with the same picks and
  draws (token-sampling's key now ranks a NaN lowest like the others; Core rejects NaN logits). topk-routing keeps its
  repeated warp argmax: a router row is one warp's registers and its picks are wanted in selection order.

- token-sampling's full law (the rows the fast path cannot settle, such as a flat row at top-p 0.95) spreads a row
  over 32 one-warp blocks of a cooperative launch, with the one-block law's bits: the same per-thread sums and warp
  trees, the cross-warp steps through slots in scratch and a per-row barrier. It runs where the occupancy API says
  the grid fits at once; otherwise, when the launch fails, or with `CRIT_SAMPLING_ONE_BLOCK=1`, one block a row as
  before. The tie scan also skips the 1,024-id tiles that hold no token of the boundary key. RTX 5090, 248,077-token
  rows: a flat top-p 0.95 draw 2,895 -> 1,316 us, a four-row verify round 6,746 -> 2,014; tokens and verify
  predictions equal to the previous build's. `scratch_floats` grows by the spread rows' counters and slots.

- recurrent-scans' chunk-aligned exact form, `gated_delta_chunk_rows` (structure `delta-chunk-rows`, row-exact):
  `delta-prefill-tc`'s code over many rows in one launch pair, each its sequence's tokens from a 64-aligned position
  from its own state slot, the state after its complete chunks to another slot. A token's tensor-core bits depend only
  on its chunk's tokens up to it and the chunk's start state (the tiles' padding and masks are exact zeros), so a decode
  step that recomputes its chunk so far gives the bits the prompt gives the token, and decode, verify and prompt rows
  agree bit for bit (tests/test_chunk_exact.py). An opt-in form (recurrent-state-recovery's `decode: chunk`). On
  Qwen3.8-27B's 48 gated-delta layers (RTX 5090) a decode step costs +0.84 / +1.34 / +5.1 ms at 1 / 8 / 32 sequences
  over today's scan, and the exact prompt takes 12.3 ms instead of 78.8 at 2K tokens, 44.8 instead of 312.7 at 8K.

- The exact prompt path is fast: paged-kv-attention's fused CUDA launch has a serial form, one program a tile that
  walks its row's splits in turn and folds each into the tile's running state as it ends (the merge's steps on the
  split's state as a split program writes it), the queries of 64 query rows in shared memory, the running outputs in
  registers up to 128 value columns a program. The same states folded in the same order: the split launch's bits,
  with no partials, ring or sync words. A fused launch takes it where its tiles fill the GPU (`FUSED_SERIAL`,
  cuda-fused's speed knob `SERIAL`), so a step's prompt pieces under the `exact` level (the decode's reduction) read
  each tile's keys and values once as a prompt chunk does. RTX 5090, prompt rows as attention-state reads them under
  `exact`, 24 / 2 heads x 256: 2,048 tokens after 1,024 cached 2,828 us through the split-KV reader against 828
  (reader cuda), 8,192 tokens 11,135 against 5,690, four rows of 512 tokens 1,731 against 1,002; the repeatable
  prompt chunk takes 699, 4,964 and 1,180.

- paged-kv-attention's `prefill_rows`: several sequences' prompt chunks in one launch of the Triton prompt reader
  (a block map over the rows' query blocks and one metadata row a sequence, made once a set of rows), each token
  with the bits `prefill` gives it in its row's own launch. Concurrent prompts cost one launch a layer instead of one
  a row.

- grouped-experts multiplies on the formats' own tensor cores, as exact decode/prompt pairs: `fp4-decode` /
  `fp4-prompt` (NVFP4 W4A4 on mma.sync kind::mxf4nvf4, the sm_120 family) and `fp8-decode` / `fp8-prompt`
  (fp8:channel and fp8:block W8A8 on the E4M3 MMA, 8.9 on), one mainloop (`mma_experts.cu`) for both atoms. Every
  expert format ran on BF16 MMAs after decoding, so W4A4 prompts ran at 48 to 78 TFLOPS on a GPU whose FP4 tensor cores
  do 800 and more. The operands are crit's declared quantizers': NVFP4 per 16 values at the input scale (the values
  the BF16 path's W4A4 multiplies, as E2M1 codes and UE4M3 scales), FP8 per 128 values at amax / 448. K is cut into
  parts at fixed 128-k block boundaries, as many as K alone gives (`k_parts`: 4 from 2,048 k, 2 from 512, else 1),
  each output is one MMA chain a part over its K in order (FP8 folding each 128-k block's sum by its two scales) and
  the parts' sums are added in part order, so a prompt item gives a route its decode item's bits. Decode items run a
  part a warp and add the parts through shared memory, prompt items run them in turn in a warp. RTX 5090, a layer at
  one token on 2048 x 768: FP4 33.9 us (the BF16 path's W4A4 33.6), FP8 48.4; FP4 prompts 1.8-2.0x the BF16 path
  (2,046 against 3,980 us a gate/up pass at 8,192 tokens), 12-16% under their speed without the split, which costs a
  second accumulator. `route_outputs(..., arithmetic="fp4" | "fp8")`, `fp4_available()` / `fp8_available()`;
  refused for other formats and where the GPU lacks the MMA. The kernel core gains `nvfp4_scale_code`, `nvfp4_code`,
  `nvfp4_quantize16` and `nvfp4_quantize_pair` (the declared NVFP4 quantizer as codes).
  The grouped-experts plugin takes them as `arithmetic: fp4` (NVFP4 W4A4 layers) and `arithmetic: fp8` (FP8 experts
  W8A8), refused where the GPU lacks the MMA, for other formats and for NVFP4 layers without input scales.

- The operation contract is executable: `crit_kernels/contracts.py` holds every operation's FP64 reference and the
  tolerance a kernel keeps against it (`contracts.reference(op)`, `contracts.check(op, got, want)`: close within
  atol + rtol |want|, a projection's FP32-accumulation bound from the products' magnitudes, or the same bits), one
  definition for the kernels' benches, the gates and the tests. operations.json described them in prose only, and
  most benches and tests carried private references. CI checks that every operation of every technique has a contract
  and that every structure of every kernel maps to operations a technique defines (`contracts.problems()`, in
  crit's tests and the kernels workflow, which now also runs when an operations.json changes); the references are
  tied to the techniques' own FP64 host forms by tests. Operations that kernels computed but no technique defined
  are now defined: page writes (`kv-page-write`), INT4 group projections, EXL3 projections, norms quantized in the
  same launch, expert routing (heterogeneous-expert-dispatch's `routing.py`), Mamba-2's state-space scan and the
  causal convolution (FP64 host forms `ssd_sequential` and `causal_conv_rows` in recurrent-state-recovery); topk-routing,
  exl3-rows and gguf-rows name their operations, and paged-kv-attention's and recurrent-scans' structures name the ones
  each computes (a page write is not attention, a convolution not a gated delta).

- Library interfaces that do not assume the library's own layouts (#57). A model plugin that decodes without drafts
  or verify, its state in a layout of its own, could not capture its steps: the round driver captures only rounds
  whose every slot has `rounds:<slot>` and whose pools have `bind`, `unbind` and `commit_round`. Core's
  `crit_plugins.captures.Captures` captures any capture-safe `forward(key)` over fixed buffers, one graph per key,
  with the drivers' rules (the slots' facts and reasons, warm-ups and `reset_capture()`, the engine's graph pool, the
  memory planned as `"<name>:captures"`, replays counted per slot), behind a capture backend whose `EagerCapture`
  tests a plugin's capture logic on the CPU; prompt graphs share its capture and facts check. device-expert-cache
  caches opaque byte rows of any size in a model's own layout (`experts:rows`: `row_store`, `row_cache`,
  `cache.rows()`), not only an experts plugin's bundles. A plugin builds on a library plugin instead of copying it:
  a source that is a plugin's folder imports as in its own plugin, `extends` takes its options, and
  `crit_plugins.compose` creates the base or another plugin of the package (the test stand-ins' three copies of that
  helper are gone). Tests: tests/test_captures.py, tests/test_plugin_compose.py (a fused router over topk-router,
  bit for bit), device-expert-cache's tests/test_rows.py ([captured passes of a plugin's
  own](docs/PLUGINS.md#captured-passes-of-a-plugins-own), [plugins built on
  plugins](docs/PLUGINS.md#plugins-built-on-plugins)).

- `crit run ... ablate --rounds N` measures a 1% change (#38). One pass serves each variant once and judges against
  3% noise, since the variants are served minutes apart and the device drifts between them by more than a small
  change. With rounds every variant is served N times in alternating order, as `crit eval ab` alternates two engines,
  and each comparison is made round by round: the median of the rounds' paired changes, a KPI counting as better or
  worse only where every round's change is on that side and their median beyond 0.5%
  (`crit_plugins.ablation.PAIRED_NOISE`). The report keeps each variant's rounds and each comparison's paired changes
  (`paired`). In four rounds on the toy engine, decode rounds 1% longer read as slows, and the same options served
  again as neutral.

- Plugins keep the host-side stores they build from the checkpoint for the next engine of the model (#38).
  `context.keep(name, key, build, close)` (`crit_core.kept`) returns the store kept under that name with that key, or
  builds it; crit serve's model pool keeps it while the model is loaded, as it keeps device weights, and Core's
  eval-v1 adapter keeps it in crit eval's process until its `release()`. expert-offload's ram tier keeps its expert
  store this way (expert-tensors' `open_store(kept=)`): an ablation's variants, a measurement's lower context and
  `crit eval ab --package`'s rounds no longer pack every routed expert again at each build. One store a name: asked
  under another key, the kept one is closed before the new one is built, and an engine whose experts go to another
  tier closes it first.

- A plan recorded after its change was measured in the work tree says so: `retroactive: true`. The run's summary had
  to say it in prose. crit checks such a plan like any other and keeps the flag in `state.plans`, the
  `plan_recorded` event and the plan's outcome, and `crit run ... next` names the run's retroactive plans at hand-off
  for summary.md. It is refused where a plan must come first: on a stop plan, in a replay's plan, and as
  `replay --before PLAN`, whose run already holds the measurement (docs/RUN_RECORDS.md#retroactive-plans).

- A run records a rung Eval cannot reach. A run whose model Eval refuses had to name an older candidate as every
  later plan's parent, since a full-rung candidate without its Eval never stood on the ladder. `crit run ... eval` now
  records Eval's refusal as the run's state, "rung unreachable: REASON" (`rung_unreachable.json`), before it reserves
  anything; from then on a candidate sealed `built` after `serve --check` stands on the ladder on any rung, each plan
  starts from the newest one, and `crit run ... next` and `show` say that no candidate is certified and why
  ([rung unreachable](docs/RUN_RECORDS.md#rung-unreachable)).

- crit smoke's `fluent` is judged against the checkpoint's own reference. Its limit was an absolute perplexity of 40
  on smoke's prose: a quantized checkpoint whose own reference reads it at 74 failed every engine, and `--no-smoke`
  then hid real regressions too. `crit run ... recon --data EVAL_DATA` now records smoke's reference in recon.json
  (`gathered.smoke_reference`): four conversations of the prepared data and crit Eval's reference log-probability of
  each answer token. `fluent` scores the same ids and passes on the perplexity over the reference's (1.5, strict 1.1)
  and the drift per target, reported as median, p90 and maximum (median 0.5 and p90 2 nats, strict 0.1 and 0.5).
  `crit run serve --check` and `crit run ablate` pass the run's reference, `crit smoke --reference RECON_JSON` reads
  one; without one `fluent` keeps the absolute limit and says so (#60).

- crit smoke tells a decode gap's rounding from a fault. A failed `decode_matches_prefill` or `decode_long` said how
  far decoding moved from prefilling, not whether the decode arithmetic rounded otherwise or KV, positions or masking
  were wrong. A score may now ask `decode_via_prompt` (`POST /v1/crit/score`), a debug option Core passes the model
  plugin as `StepRequest.decode_via_prompt` and never runs in a native round: each decode row is computed as a
  one-token prompt chunk over the same KV and positions. model-template honours it (info `decode_via_prompt`), and on
  an engine that says so smoke decodes the failed comparison again that way: `rounding` when the rows then match
  prefill, `fault` when they still move.

- `crit run serve --check --data` judges the quality texts' drift before a round: crit-eval's gate pre-check now
  scores every quality text against the reference by the run's drift budget (its quality mode's, every resume's
  amendment applied) and reports `precheck_quality`. A staged Eval scores the long texts only from stage 2 on, so a
  candidate whose approximations drift at depth (fewer routed experts kept) passed serve --check and failed quality
  in its stage 2 Eval. It needs crit-eval with `precheck_quality`.

- Core asks the plugins what the engine will be before it loads. A Flash-Next build's recon priced FP8 KV at 12,288
  bytes a token and planned its device from that; the engine took BF16 pages and indexer keys, 29,952, and only a
  load said so. A plugin's module may now answer `resolve(context, options, dry)`: the engine facts it can tell from
  the checkpoint's headers, its options and the selection alone, no weights read, no device touched. model-template
  resolves the KV its pools take a token (`kv_bytes_per_token`: its layout's pools, a sliding window's apart, the
  draft layer's and a block's keys, each priced by attention-state's `position_bytes`); its load is the same `Shape`
  (code/shape.py, split from the constructor) and makes exactly the pools it listed. Core runs every `resolve()`
  before a load (`crit_plugins.assemble.resolve`, a fact's shares summed over plugins) and refuses an engine whose
  facts are not what was resolved; `crit plugins resolve PACKAGE --model M --context N` prints the answer in
  seconds; and creating a candidate refuses a plan whose `kv_bytes_per_token` its engine does not match at the run's
  context, before anything is built on it. recon.md and AGENTS.md point to it.

- Core watches what an engine compiles once its load is over. Every kernel compiles at load, none on a request, but
  only a plugin's own test that wrapped a request in `crit_plugins.testing.compiles()` checked it, and crit 0.20.0
  fixed six kernels that compiled on a request one by one. The loader now keeps a compile watch for the engine's life
  (`crit_plugins.compiles.Watch`: Triton JIT compiles and torch extension builds; one hook of each for the process,
  which nested watches share), the engine's telemetry counts what it saw (`core.compiles_after_load`, the first few
  named), and `crit run serve --check` fails `compiles_after_load` where any of its requests, over every context it
  opened the engine at, compiled. `crit_plugins.testing.compiles()` is the same watch.
- Core counts the drafting of the requests it serves (`core.drafts` in the telemetry: requests, rounds, drafts
  accepted and, where every round's count is known, proposed), and `crit run serve --check` fails `drafts_accepted`
  for a drafting engine that accepted none of its drafts over 32 or more decode rounds of the check's replies. A
  drafter that never asked the draft head its model loaded served 0 of 285 drafts at 22 tok/s, and only
  model-template's own refusal of that drafter caught it; any model plugin's dead drafter now fails its first check.
- `crit_plugins.ranking.top_ranked(values, k)`: the first k entries of each row in Core's order (a stable descending
  sort: equal values by the lower index first), bit for bit, from a top-k and a sort of the entries at or above each
  row's k-th value. model-template ranked a score's alternatives and a request's top log-probabilities by sorting the
  whole vocabulary: its values, an int64 order and the sort's workspace, 0.76 GB for a slice of 128 rows of a
  248,320-token vocabulary, the largest temporary of a long score and so of Core's step arena. It ranks with
  `top_ranked` now.
- Core's compile watch says why a kernel compiled again. Core's loader records every compile from the start of the
  load (`crit_plugins.compiles.record()`, whose hooks go in the moment Triton or torch's extension builder is
  imported: Core imports neither), and a compile after load with the constexprs of one the process made
  before names the runtime arguments whose specialization changed ("B is now 1, was divisible by 16"), with the
  remedy; a compile with constexprs the process never used says it is a class no warm-up reached. The description
  lists the arguments Triton specialized apart from the declared constexprs, which it used to mix in with them.

- `crit run serve --check` fails `step_memory` where a step's segment was refused, passed the step arena or left
  less free than the run's VRAM headroom over the check's requests. Core's allocator counted them, and nothing read
  the counts. The device's counts now start anew when the engine serves (the load's under `steps.load`), and the
  check sums them over every load it served.

- A build plan states the KV its engine serves (`kv_bytes_per_token`: the device bytes one more token takes in the
  engine's KV pools, as its kv_cache plugin resolves them), and `crit run serve --check` fails `plan_kv` when the
  engine's facts give another figure. A Qwen3.8-Flash-Next build budgeted its expert cache on recon's 12,288 bytes a
  token (the protocol's FP8 price) while its engine served 29,952 (`kv: auto` resolves to BF16 where the checkpoint
  stores no k_scale, and the sparse indexer's keys come on top): 3.7 GB the plan counted on and never had. recon.md
  now says its KV figure is Eval's protocol price for the context, not an engine's.

- A plugin option that drops or approximates the model's work says so in its manifest (`approximates: {exact,
  what}`: expert-offload's `keep_routes`, model-template's `kv` at `fp8:row`), the engine's facts list the
  approximations its configuration sets (`approximations`, an engine fact crit serve reports), and an ablation variant
  that sets one says its speed does not judge it. An ablation judged `keep_routes: 6` on speed and the build took it
  into a stage 2 Eval, where it failed quality at 32k (KLD 0.59, 148% of the drift budget).
- A model plugin can name its model's own tool-call syntax: `tool_format` in its manifest, `FILE.py:NAME`, a
  `crit_core.text.ToolFormat` (docs/PLUGINS.md). crit serve then declares `chat-tools-v1` (and
  `chat-tool-stream-v1`) from the manifest without running plugin code, and `crit_core.engine.ResidentEngine` sets
  `ResidentChat.tool_format` from it (`crit_core.engine.tool_format`), so calls parse for a model such as Gemma 4,
  whose `<|tool_call>call:NAME{...}<tool_call|>` crit does not register. Before, `ResidentChat.tool_format` had no way
  in from a package, and a request with `tools` to such an engine got HTTP 400. Forced calls stay undeclared: a
  grammar cannot write a syntax crit does not register (Dualshok3, #84).
- A model plugin can name its model's reasoning markers the same way: `reasoning_format`, a
  `crit_core.text.ReasoningFormat` (`crit_core.engine.reasoning_format`). crit serve then splits the reasoning from
  the reply and declares `chat-thinking-budget-v1`. Before, Gemma 4's `<|channel>thought ...<channel|>` reached clients
  as reply text without its special-token markers (a stray "thought" before every reply) (Dualshok3, #84).
- Constrained requests decode in Core's captured rounds. A json, grammar or forced-call reply went through `step()`,
  one Python step a token, beside requests that decoded with no Python at all. Where the sampler's rounds mask
  (token-constraints), a constrained request now takes a round slot at width 0 and its mask is applied on the device:
  Core stages the matcher's next allowed set for the slot between two rounds (`StepRounds::masks`,
  `NativeRounds(stage=...)`, `stage_mask`), calls `commit` after every round that ran one, checks the round's token
  against the set and advances the matcher there. The tokens are those `step()` gives; each such round costs one host
  hop, whose speed is not yet measured on a GPU. model-template binds greedy constrained requests to its rounds. The
  engine ABI is 9: rebuild `crit_core_steps` (#36).

### Changed

- The library's starting selection fills the sampler with token-constraints instead of token-sampling: the stress
  tier's selection (tests/stress.py), model-template's example and AGENTS.md's build advice. token-constraints samples
  by token-sampling's law and masks a constrained row first, so JSON, grammar and forced-call requests decode in
  Core's captured rounds (about 96% of plain decode speed on an RTX 5090) instead of one Python step a token (about
  60%). Requests without constraints sample as before.

- `crit kernels sweep` builds less. A configuration the bench says runs none of the model's shapes (`valid_for`) is
  not built (the defaults always are); a kernel with CUDA sources beside its Triton modules groups a structure's
  configurations into builds by the CUDA variants they run, each build compiling only those; and configurations that
  differ only in the knobs a bench names `runtime` (read at run time, not template arguments: paged-kv-attention's
  CHUNK and RAMP of cuda-window, SPLITS of cuda-prefill) are built together, so a build compiles fewer tiles (a
  cuda-window build 17.0 s against 13.8 s). paged-kv-attention for Flash-Next: 9,103 configurations in 384 builds
  and 1,101 CUDA variant compiles before, 5,866 in 251 builds and 516 compiles now. The builds run in spawned
  workers: valid_for() reads the device, and a forked worker cannot use the CUDA its parent initialized. Tests:
  crit_kernels/tests/test_sweep.py.

- `crit run serve --check` reopens the engine once instead of twice: crit smoke's pass at the run's context runs right
  after the serving checks' short requests, on an engine as a fresh one is (its pools are elastic, and after the full
  context's prompt they hold more), then the full context and the gate pre-check as before, then smoke's pass at the
  lower context, where the pre-check left the engine. The engine is no longer opened at the run's context again only
  to run smoke there (`_smoke(full=, lower=)` runs either pass). A reopen of Flash-Next took 189 s, 60 s of it
  reading its 65 GB expert store into host memory again; its whole check took 2,772 s. Tests:
  tests/test_run_actions.py.

- expert-offload keys the expert store in the run's plugin cache by the code that makes its bytes: the plugins in the
  model, expert_source and experts slots and every folder of code they take as sources, their tests apart
  (`packing_folders`). It hashed every plugin of the package, so a candidate that changed any other plugin (adding
  a graphs plugin, another sampler) packed Flash-Next's 25,088 experts again (250 s) and wrote its 66.5 GB store
  (235 s), and the plugin cache ledger evicted another run's store to make room. A selection that does not name the
  three keys the store by every plugin, as before. Tests: inference/expert-offload/tests/test_plugin.py.

- Native code built at load is cached by content, not by where its files lie (`crit_plugins.native`): `library()`
  and `extension()` hash each source, every header of the sources' own folders and of the include folders by its
  name within its folder, and the flags. Every candidate's package holds its own copy of its plugins, so the path in
  the key made each candidate compile its plugins' extensions again (the expert directory, the row reader, the
  sampling law, the CPU routes: nvcc, a minute or two each), and each checkout Core's own host handoff, in its plugin
  tests and again at its load (a Flash-Next load spent minutes of nvcc that way): an unchanged plugin in another
  package or checkout now compiles nothing. torch's extension builder keys a build by its paths too, so an extension
  builds from a copy of its sources and headers under the native cache, in a folder named by their hash. A library is
  still opened once a copy (the build copied beside itself per source folder), so two engines in one process never
  share its state. Tests: tests/test_native.py (two copies build once and count apart; a changed header beside a
  source builds again; two copies stage into one folder).

- `crit plugins test` and serve --check's plugin tests run several plugins at once, a process each, and report them in
  the registry's order: `run_tests(jobs=)`, `crit plugins test --jobs`, `CRIT_PLUGIN_TEST_JOBS`, by default half the
  cores up to 4 (a plugin's tests may use the GPU, which the processes share). They ran one plugin after another:
  Flash-Next's 25 packaged plugins took 715 s of serve --check whenever their cache key changed (any change to
  crit_plugins or crit_core, so the first check of a build run), and the host suite's
  test_plugin_library.LibraryTests, which tests every library plugin, was its longest unit (169 s). Tests:
  tests/test_plugins.py (two plugins that sleep 1.5 s each run at once and are logged in order).

- sequential-recurrence and recurrent-state-recovery share one device form, recurrent-state-recovery's
  `code/slot.py` (`RecurrenceSlot`: the kernel for a pool, its `StatePool`s, prompt-row crossovers, facts,
  exactness, `close()`), instead of two copies of it; each plugin keeps its host prefill and options. Tests: both
  plugins' tests on CUDA.

- Core decides the engine's step settings and the model plugin states its limits (step contract 2,
  `crit_plugins.steps.CONTRACT`; [step settings](docs/PLUGINS.md#step-settings)). model-template chose how many
  requests it steps at once (`concurrency`), how many wait (`pending`), timed and fitted its own prompt spans, planned
  its own activation reserve and released torch's cached memory below its own 640 MiB, and Core's scheduler followed.
  The engine's step plan (`LoadContext.steps`, `crit_plugins.steps.Steps`) now answers each: `lanes(bound)` and
  `pending(bound, lanes)` (Core's 4 and 2, at most the model's bound and within crit serve's `--max-concurrency`, so a
  capped engine also sizes its states and captured rounds for the lanes it gets), `calibrated(plugin, chunk, key=,
  measure=, ...)` (Core settles the model's three timed steps as its `spans` decision and fits them with
  `crit_plugins.steps.fit`, moved from model-template), `seconds(bound)` (a step's bound, at most three quarters of the
  cancel grace: a `span_seconds` above it is taken at that), `activations(need)` (the reserve planned before Core
  measures its step arena) and `release()` below `threshold()` (the configuration's `release_below_mib`, else the
  device's headroom and the step arena: what the next step needs free; it replaces `crit_core.engine.release_threshold`,
  and Core's adapter releases through it at a request's end). For engine authors: model-template's `concurrency`,
  `pending`, `span_seconds` and `activations_mib` are bounds or needs Core decides within; `release_below_mib` is no
  longer read (set the configuration's `release_below_mib` to override Core's threshold); a model plugin states
  `step_contract: 2` in its info. An engine's own model plugin that states none still loads, chooses as before, and
  Core's runtime applies its cap over what it declares.

- crit serve hands scheduling to Core. `--max-concurrency` is no longer applied in crit serve's queue: the host loads
  the engine with it (`limits.max_concurrent_requests`, `crit_core.engine.load(max_concurrent_requests=)`) and Core's
  runtime holds no more requests than it, so the engine itself reports the capped `max_concurrent_requests`. A
  request's `priority` reaches Core's scheduler in its request record, and Core's adapter passes it on
  (`ResidentChat.priorities`). crit serve's queue keeps the loaded model's exclusivity, back-pressure at the engine's
  capacity and the bound and timeout behind its 429s; where requests wait in it, they leave in Core's order
  (`crit_core.native.next_request`). docs/SERVING.md says so.

- The cancel grace has one definition, Core's (`crit_plugins.steps.GRACE`, one second): crit serve's
  `crit_api.residency.CANCEL_GRACE_SECONDS`, by which it retires a host whose cancel does not settle, and the stress
  test's import it instead of keeping their own copies of the second. The ownership test's allowlist is empty.

- The default VRAM headroom and the arithmetic of the KV cache that fits the device are Core's. Core's 0.25 GiB
  (`crit_plugins.memory.HEADROOM`, which its loader leaves free where a package names none) had copies in crit Eval
  and in six readers of the CLI. crit run, crit pull, crit bench and `crit plugins offload` take the default from
  Core now (`crit.settings.vram_headroom_bytes`), and crit Eval too. `crit_plugins.memory.kv_fit(free, kv_bytes_per_token,
  headroom=, reserve=, held=, granularity=)` is the KV that fits (free memory less the headroom, the reserve and what
  stays on the device, over the KV bytes per token, at a granularity), which crit Eval's max context, and so a run's
  KV capacity, calls; Eval's results are unchanged (its tests pass on both paths, and the old formula and `kv_fit`
  agree on 200,000 random inputs, floats included). `crit run ... eval` judges a candidate's memory by the run's
  recorded VRAM headroom, the one its configuration states and Core's loader leaves free, instead of re-reading the
  machine's setting, as `serve --check`'s gate pre-check already did. The `kv_capacity` docstring said the default
  context was 262,144 tokens; it is crit Eval's 200,000.

- settings.toml has one reader for crit's commands, `crit.settings` (`load`, `SETTINGS`, and the host headroom's
  default `RAM_HEADROOM_GIB` with `ram_headroom_bytes`): crit run, crit pull and crit bench read it there instead of
  through crit run's own reader, and the 2 GiB host headroom default written out in two places is one. crit Eval's own
  commands keep reading the file leniently through `crit_eval.machine_settings()`, which finds the project crit names.
  Nothing a setting does changes.

- An engine's policy is its own (`crit_plugins.policy.Policy`, `LoadContext.policy`, `engine.policy`). Core's loader
  told the plugins what a configuration asks (its kernel tuning, decisions record, exactness, quality mode and prompt
  activations) by setting module globals in crit_plugins.kernels and crit_plugins.decisions, so a second engine
  loaded in the same process (crit serve's host building the next engine, ablation variants, `crit eval ab`) took
  the first one's policy, and it was right only because loads never overlapped. Now every load makes its engine a
  policy (`crit_core.engine.policy_of`), activates it while it loads and around every call Core makes into its plugins
  after (a `contextvars` variable set in each step callback on the engine's worker, which is a thread Python did not
  start, and in `prepare`, `telemetry` and `close`: `StepEngine(scope=)`), and the host's worker pool carries the
  caller's policy into its tasks. The functions plugins call (`asks`, `allows`, `quality_mode`, `activation`,
  `tuning_file`, `load_kernel`, `loaded`, `decisions.settle`, `report`) answer from the active engine's policy; the
  record of the kernels an engine asked for and its settled decisions are the engine's, the built kernel modules stay
  the process's. The `use_*` functions set the active engine's policy, or outside any engine the process's own (tests,
  a quick try), and `CRIT_KERNEL_TUNING`, `CRIT_DECISIONS` and `CRIT_PLUGIN_CACHE` stay the fallbacks (`load(...,
  cache=)` names the cache folder outright). `crit plugins resolve` resolves under the package's policy too, and
  `crit plugins decisions` writes the loaded engine's own record. tests/test_engine_policy.py loads two engines with
  different exactness, quality mode, activations, tuning and decisions record in one process and checks what each
  one's steps see; the ownership test's `global` rule now allows only process state.

- The exactness levels and the quality modes are Core's (`crit_plugins.exactness`: `LEVELS`, `PHASES`,
  `QUALITY_MODES`, the mode -> level table `MODE_LEVELS`, `rank()`, `keeps()`, `setting()`, `for_quality_mode()`).
  crit_kernels defined them, so Core imported them from above itself in five places and refused a configuration
  with `"exactness"` where crit's kernels package was not installed, and four modules kept copies (crit_core.serving's
  `EXACTNESS_LEVELS`, crit_plugins.kernels' and the run records' `QUALITY_MODES`, the offload command's choices,
  smoke's level lists). crit_kernels.exactness keeps what reads a kernel.json (`level()`, `keeps()`, `reference()`,
  `same_bits()`, `choices()`, `problems()`) and imports the rest from Core, under the names it had. Core loads a
  configuration with `"exactness"` without crit_kernels. The modes are listed in one order, preserve then speed, so
  Core's messages name them so too (`configuration "quality_mode" is one of preserve, speed`).

- token-sampling's law, device-expert-cache's directory and heterogeneous-expert-dispatch's in-stream CPU routes are
  crit's kernels `token-sampling`, `expert-directory` and `cpu-routes` (kernels/techniques/sampling and experts), each
  with a kernel.json, tests and a README, its bindings torch ops with the plugin's call shapes. The plugins (and
  token-constraints and expert-offload, which borrow their code) load them through
  `crit_plugins.kernels.load_kernel` and run their fallbacks where they do not build. token-sampling's spread law has
  a knob PARTS (8, 16 or 32 blocks a row, the same bits); expert-directory's fill a bench against torch's index ops;
  operations `expert-residency` and `cpu-route-handoff` name what the two compute.

- paged-kv-attention's CUDA window launch takes its decode chunk from the cache geometry (90,112 x sqrt(kv heads) /
  (6 x SMs) positions, RAMP 1 by default: a row's splits follow its own positions only, row-exact). Its decode then
  takes 0.885x split-KV's time over 4 geometries, 1-16 rows and 1K-32K positions. attention-state's reader `auto`,
  now softmax-attention's and partitioned-attention's default, takes the CUDA tile for each pool it covers (out dtype
  the query's), and prompt chunks keep split's row-exact prompt reader. The exact CUDA-serial prompt reader is slower
  at the new chunk (828 -> 1,193 us at 2K, 5,690 -> 7,241 us at 8K), still faster than split's (2,828 / 11,270).

- grouped-experts' fp4-decode and fp8-decode gain a knob FETCH: a lane loads its four MMA fragment words of a weight
  step directly instead of 16 bytes and a quad transpose (the same registers and bits, fewer instructions). The
  default takes it at one token: RTX 5090, a layer on 2048 x 768, FP4 49.9 -> 37.2 us, FP8 80.7 -> 68.8 us; level on
  4096 x 1536, and level or up to 13% slower from 2 tokens (measured before K was split into parts, above; it works
  with the parts, and the split took the same layer to 33.9 and 48.4 us).

- quantized-projections' W4A4 and W8A8 decode rows end in the GEMV's epilogue: the W4A4 input scale is folded into
  the GEMV's row factor at load (no division or cast after it), and W8A8 rows quantize in one kernel where mma-rows
  is built.

- The declared order's staged prompt GEMM is one kernel-core mainloop (`crit_kcore/declared_staged.cuh`: the cp.async
  ring of GroupVisits, the stage decoded a visit ahead, the declared steps, the fold16 tree, the epilogue), and
  fp8-block-rows' and int-block-experts' `gemm-staged` are its two users: each technique keeps only its stage layout,
  its copies, its decode and its block sums (e4m3 from zero, or s8 onto the exact-sum bias). The two kernels carried
  the same 166-line mainloop, two thirds identical. The same bits (both kernels' staged tests).

- The kernels' benches (paged-kv-attention, latent-attention, recurrent-scans, decode-gemv, fused-norms) take their FP64
  references and tolerances from `crit_kernels.contracts` instead of private copies, so the sweep's checks and the
  gates hold every structure to its operation's one tolerance.
- mma-rows' FP4 tiles use the kernel core's NVFP4 MMA wrappers (`mma_nvf4_m16n8k64` and the new
  `mma_nvf4_m16n8k64_zero`), quad transpose and blocked scale offsets instead of their own copies (the same SASS).

- attention-state reads a step's prompt rows through crit's Triton prompt reader in one launch (`prefill_rows`)
  instead of one launch a row and layer, their page tables made once a step; the exact prompt path (prompt pieces of
  up to 64 columns in the decode's reduction) builds its pieces' page tables and metadata once a step too, where
  every layer rebuilt them on the host.

- `crit plugins test` fails a plugin whose tests skip in a way that leaves its correctness unchecked, even where
  every test that ran passed. A model plugin whose comparison with its reference skipped, such as model-template's
  engines without Core's runtime, passed as if it had run. A test now says which kind a skip is: one because no device
  the plugin runs on is here (`skipTest`, `device()`) stays a listed pass; one that leaves the plugin unchecked
  carries `unchecked: ` in its reason (`PluginTestCase.skipUnchecked`, `crit_plugins.testing.unchecked()` for
  `setUpClass`) and fails the plugin, named in the detail, never cached, and failing `crit run serve --check`'s
  `plugin_test_ID`. `core_runtime()` skips so without Core's runtime, and model-template's engines without it or
  transformers. The library suite still passes where Core is not built: CI builds it, so there such a skip fails
  ([docs/PLUGINS.md](docs/PLUGINS.md#each-plugins-own-tests)).
- expert-offload caches a weight-only (W4A16) layer beside W4A4 ones instead of holding it whole on the device,
  where the CPU tier serves the other layers' misses under the recent policy. The CPU tier computes W4A4 experts
  only, so such a layer (FP4-Local Qwen3.8-Flash-Next's draft layer, 512 experts) was kept on the device whole, and
  under extreme memory pressure the cache left beside it held too few slots for a prompt ring (507 left, 513
  needed): the load was refused. The layer is now a link layer of device-expert-cache: its misses are admitted in
  their own plan from a reserve of the round's victims and copied over the link before the layer reads them, its
  rows the kernel's tensor arithmetic, the bits it had held whole. The reserve is what one decode round routes in it:
  model-template tells expert-offload's `begin()` its decode rows (every request's catch-up of width + 1 columns and
  its chain's width - 1 steps, which a captured round plans in the draft layer under one victim list) and top k, 240
  slots on Flash-Next at concurrency 4 and width 3, so a prompt takes 753 slots, not a layer's 512 held whole and a
  ring's 513 beside them. Where the round's other plans took reserve slots, a link plan goes on below the reserve, so
  no route is lost while the round's victim list has open slots. `reasons.link_layers` says which layers and why. The
  ranked policy keeps such a layer whole, as before.
- An ablation switches drafting as one unit: `without:` a drafter or a verifier serves the plain engine (both slots
  at their base), and `only:` one keeps its partner. `without:greedy-head-drafter` kept the verifier, so the variant
  drafted with nothing to draft and reached Core's load before anything refused it (info() declares speculative
  False, but the selection drafts).
- `crit run ... next` names a step its command takes. An evaluated candidate whose checks keep its benchmarks
  exploratory was told to measure, which `measure` refused, while a completed seal refused without a measurement:
  a Flash-Next build's candidate 2 sealed `built` after three tries. The state machine now reads measure's own rule
  (`measure.unqualified`) and names `seal --lifecycle built` there, and the refusals of `measure` and of a completed
  seal name that seal too.
- A draft-only matrix whose shape the asked encoding's kernels do not take stays as stored, and the facts name it
  (quantized-projections' `draft_encoding`, under `ROLE:kept` with `asked` and why; `head_kept` for an FP8 head).
  `mtp_encode: fp8` failed a Flash-Next engine's whole load on one 324 x 10240 draft-layer matrix (cuBLASLt takes
  FP8 rows and columns in multiples of 16); the rest of its role re-encodes now. model-template's FP8-block draft
  matrices go through the same rule (`encoding_problem`, the projections slot's).
- Every launched Triton kernel says which of its runtime arguments follow the request, and the kernel build refuses
  one that does not (`crit_kernels.specialization`). Triton specializes a runtime argument it is not told to leave
  alone (an integer equal to 1 becomes a constant, one divisible by 16 says so, a pointer says it is 16-byte
  aligned), so a token count, a key count, a split count or the stride of a buffer made per call compiled a new
  kernel the first time a request brought a new mix: 14 of crit's 24 launched kernels left such arguments
  specialized, and a Qwen3.8-Flash-Next engine compiled the sparse indexer's scores and the fused norms' rows kernel
  on requests after load. Each kernel now lists them in `do_not_specialize` (`[]` where none follows the request;
  what the model fixes stays specialized, so its loads stay vectorized), and a pointer into a slice at an offset the
  request sets in `do_not_specialize_on_alignment` (the sparse indexer's token ends, `end_ptr`, still compiled its
  scores and top-k again in the whole-checkpoint stress test); the build refuses a name in either list that is not
  the kernel's runtime argument. kernels/TEMPLATES.md says how.

- A measured decision says how it was measured, and a cold one is not replayed. `crit plugins decisions` loaded the
  engine once, so its record held the first load's measurements: stores built, the checkpoint's pages read cold,
  kernels compiled (a Qwen3.8-Flash-Next record fixed its prompt spans at 2.45 s per 512 tokens where a warm load
  times 0.9). It now loads the engine twice and records the second load; every decision carries `conditions` (the
  process's load it was measured in, `first_load`), and a record whose decision was measured on a first load or does
  not say how is measured again (`stale` says why), so records written before this are measured again until
  `crit plugins decisions` records them anew. The round times a width policy weighs (`Rounds.calibrate`) are a
  decision too (`rounds` `width-costs:MODE`): a reopen replays the recorded table instead of timing rounds again.
- A prompt chunk's rows come from Core's prompt-step budget, which learns what the engine's prompt steps take as it
  serves. They were a fit of three timed steps at load (zeroed KV, the cache as it stood at load), used for good, so
  a Qwen3.8-Flash-Next engine's prompt steps at 200,000 tokens ran past the one-second bound with 1,024- and
  2,048-token chunks whatever `span_seconds` said, and a cold first load's fit froze its spans at 2.45 s per 512
  tokens. `crit_plugins.steps.StepBudget` (the load context's `steps`) gives a chunk's rows by its position: the
  calibration's form `seconds = c + a * rows + b * rows * (position + rows)` times, per 8,192 positions, the ratio of
  what the prompt steps Core's step engine timed there took to what the form predicted (a model says which chunks a
  step ran, `ran()`; a slower step sets its depth's ratio at once, faster ones are averaged in), so deep chunks
  shrink where deep steps ran past the bound and shallow ones keep their size. Where the engine asks prompt rows
  below row-exact (a chunk's boundaries change its rows' bits) the spans stay the calibration's and the budget counts the steps past the bound (`over_bound`). model-template's `Spans`
  is gone: it asks the budget and reports its chunks.
- The prompt cache's kept sequences are givers of the device's memory, in one eviction order the device owns. They
  were freed only by evict(), which a model calls when its admission runs short of pages, so a long prompt kept past
  its use held its KV pages (about 6 GB at 200,000 tokens on Qwen3.8-Flash-Next) while the expert cache stayed
  shrunk: decode at 200k measured 60.6 tok/s with the prompt cache and 68.1 without. `crit_plugins.memory`'s givers
  now have a rank (`IDLE`, `COPIES`, `WORK`) and `need()` asks the lowest first; Core's prompt cache over elastic
  pools is two givers, its sequences no prompt reused for a minute (`IDLE`, before an expert cache) and the others
  (`WORK`, after it); a giver that grows back says whether it would take more (`hungry=`, expert-offload's cache while
  it holds fewer slots than experts), and a settle that finds no room for it has the idle sequences give it a slice.
  The device's report counts each giver's releases and regrowth (`givers`). A paged range measures what is left to
  back again after its need (a giver may close sequences of the same pool), and nothing settles inside a need.
- Core allocates every byte an engine's steps take, and the device keeps the run's VRAM headroom, not a plugin. A
  step's temporaries came from torch's allocator, which Core could not see or refuse: it learned what a step took
  afterwards and kept a forecast free (a step's room by chunk size, looks at the step in flight before each routed
  layer, a reserve sized by the largest stride between two looks), and that forecast lived in expert-offload, with
  the headroom its `margin_mib` option, which `crit plugins offload` copied from the run and a build could set to
  anything. Every memory failure of the Flash-Next builds came from that gap: margin misses at a step's peak, a
  score's logits taken between two looks and kept as a prompt step's stride (each later prompt then gave back about
  2.4 GB and grew it again, and decode measured 17 to 27% slower after a 32k-token score), and plugins that had to
  remember to ask the ledger before a large temporary. Now Core's step engine runs every `step()` and `score()` in
  the device's step pool (`crit_plugins.memory`'s `step_pool()`: a torch memory pool over Core's allocator,
  `allocator.cpp`, built at load and handed to torch's pluggable allocator), which makes a segment where the device
  has the bytes free (the lesser of the driver's and the card's reading) and logs each segment made, refused and
  freed; the ledger counts the log at every step frame. Nothing runs Python inside torch's allocation: a hook there
  waited for the GIL under torch's allocator lock while a thread holding the GIL read torch's memory statistics (a
  telemetry read) and waited for that lock, and a GPU suite hung on it for six hours. The givers keep free the headroom (`memory.headroom`, set by Core's loader from the
  configuration's `vram_headroom_bytes`, which `crit run ... candidate` writes from the run; another value is
  refused) and the step arena (`memory.arena`: the most a step took while the engine loaded, or the model's own word
  for its deepest step, info's `step_bytes`, whichever is more), less what the steps hold of it now. The load's warm
  requests run the largest step of each kind: a prompt and a text scored with 20 alternatives as long as the longest
  chunk a prompt-step budget gives (an 8-token score had left a 32,768-token score's logits, their log-softmax and
  ranking at a full slice, 1.1 GiB on a Qwen3.8-Flash-Next engine, to model-template's estimate of 0.6 GiB, and the
  score failed for want of memory). A segment past
  the arena is lent from free memory and counted (`overflows`), one that leaves less than the headroom free is
  counted (`breaches`), one the device does not have is refused; the device's report carries them (`steps`). Once no
  prompt is in flight, or a long prompt's next chunk could not take a chunk's most from the pool within the arena,
  the pool retires and its segments go back to the driver: each deeper chunk needs larger blocks than the pool
  caches, and a pool kept through a whole 199,965-token prompt grew to 1.17 GB against a 0.79 GB arena (15
  overflows, 11 breaches, 96 MiB left free at worst). The forecast, the looks, the strides,
  the reserve and `chunk()` are gone, and so are expert-offload's `margin_mib`, model-template's `headroom_mib` (which
  defaulted to 0, so a dense engine's spans ignored the headroom) and its ledger call before a score's logits;
  model-template states its deepest step (`step_bytes`: a full chunk at the deepest position at the bytes a row its
  calibration took, which runs as Core's measured step, `memory.measured()`: the step pool, counted as a prompt
  step; it had read torch's reserved bytes, which a second run of the same chunk never grows, and stated 0), and
  `crit plugins offload` writes no margin. A measured step that runs out of device memory has the givers make room
  and runs again (`memory.measure(run)`): while the engine loads they keep only the headroom free, an expert cache
  opened before a calibration takes the rest, and Core's allocator cannot ask a giver for a segment, so a
  calibration chunk of 1,024 rows over elastic KV pools ran out on every load of a GGUF Flash-Next engine.
- Core's loader asks the levels of an engine's quality mode wherever the package loads. The levels resolved in two
  places: `crit run ... candidate` wrote the mode's levels into every candidate, while the loader asked only what a
  configuration's `exactness` named, so a package that stated `quality_mode` and no `exactness` (made outside a run,
  or edited) asked nothing, every plugin took its least exact `auto` (quantized-projections' native prompt GEMM,
  gguf-projections' dequantized prompts, the recurrences' chunked prompts) and crit smoke judged it loosely. There is
  now one resolution, `crit_core.engine.exactness_levels`: the configuration's level for a phase it names, the mode's
  for one it leaves out.
- The prompt activations a build plan declares are asked of the engine as its exactness is. `crit run ... candidate`
  writes the plan's `prompt_activations` into the candidate's configuration (a configuration that states others is
  refused), Core tells the plugins before any is made (`crit_plugins.kernels.use_activations`, `activation(format)`),
  quantized-projections' `mxfp8_activations` `auto` takes the activation asked for MXFP8 matrices before the quality
  mode's, and Core's loader refuses an engine whose facts serve other activations than asked, naming what it serves
  and the option that chose it. A candidate of the library's defaults served 16-bit MXFP8 activations where its plan
  declared the stored ones, which was caught after a whole Eval and later at `serve --check`; it can no longer load
  in any process (crit serve, Eval). Ablation variants ask none: they measure what other options serve.

### Removed

- Five library entries whose service Core runs for every engine, the `graphs` policy slot and its base plugin
  eager-graphs. No engine of crit-hub or crit-engines selected any of them, and each kept a copy or a model of what
  Core does:
  - host-memory-admission's observations (physical memory, the commit headroom where the OS enforces commit, the
    cgroup's headroom with only clean page cache reclaimable) are Core's now, `crit_plugins.host.memory` (`observe`,
    `admit`, `admit_working_set`), and `crit_plugins.host.memory_bytes()` answers `available` with the tightest of
    them (and `binding`), so every host allocation sized from it, Core's paged host backing, expert-offload's host
    tier and model-template's, is one the OS can back; it read physical memory only. Its tests are
    tests/test_host_memory.py.
  - host-work-mailbox's protocol description moved beside Core's handoffs (core/python/crit_plugins/host/README.md);
    its Python model of the protocol went with it.
  - recorded-selection filled the `graphs` slot, which chose between a prompt chunk's eager path and its padded
    buckets, and kept a record nothing read; its choice was fixed by its options, never by a time. The slot is gone:
    Core's prompt graphs run the smallest captured bucket that holds a chunk, decided by the chunk's length alone, and
    no longer time their buckets at load; measured choices are `crit_plugins.decisions`', and the pinned prompt GEMMs
    are crit's pinned-gemm kernel.
  - session-scheduling (a chunk cursor and a reasoning-loop guard) and concurrent-request-execution (ready-row
    compaction, metadata replicas) described what Core's runtime and prompt-step budgets do; their code had no user.
  The index (inference/README.md) and docs/PLUGINS.md say where each service lives now, and the entries that linked
  to them link to Core.

- decode-gemv's per-format structures gemv-nvfp4, gemv-fp8, gemv-bf16, gemv-nvfp4-triton and gemv-fp8-dot, kept as
  aliases of gemv-cuda and gemv-triton so old tunings applied: their calls (gemv, gemv_fp8, gemv_bf16, gemv_triton,
  gemv_fp8_dot) run gemv-cuda's and gemv-triton's knobs for their format, with the same bits, and a tuning that names
  them is migrated into those structures' tables when the kernel is built (kernel.json `retired`). The kernel builds
  and sweeps two GEMV structures instead of seven. quantized-projections no longer reads the old structures' reports.

### Performance

- paged-kv-attention's sweep builds hold the CUDA window launch's dispatch table's diagonal only (select() gives a
  sweep's two lookups one configuration), so a build's cost grows with its configurations, not their square: a
  cuda-window sweep build of 96 configurations takes 15 s instead of 895 s (RTX 5090 host, one variant, cold cache;
  the same 8 kernels). Tuned builds keep every pair, and a pair a sweep build does not hold is answered as before.

### Fixed

- A reply that ends at its output cap may be blank. crit serve answered a whitespace-only reply with a 500 ("Reply
  requires assistant text, reasoning or tool calls") even when the engine had spent every token it was given:
  measured on a cut Qwen3.8-Flash-Next engine, whose every `json_object` reply was whitespace (JSON allows any amount
  before its value) until `max_tokens`. Such a reply is now the reply, finish `length`, streamed or not; a blank
  reply that stops before its cap is still refused (tests/test_response_content.py).

- serve --check's `full_context_device_memory` judges the engine, not what other programs on the card hold at the
  moment. Core's probe now samples, per step, the memory outside the engine (`crit_plugins.memory`'s `other()`: the
  CUDA driver, other programs; `[seconds, free, outside]` in the telemetry's `recent`). The check takes each step's
  free memory as if that outside memory had stayed at the least it held during the request, and its detail says
  how much it subtracted and what the device read. On WSL, where free memory is the card's, outside memory moved
  by up to 435 MiB within one check and pushed it below the 0.25 GiB headroom at any KV format (0.11 GiB once,
  then 0.25 and 0.44 GiB for the same engine). The headroom keeps its meaning: the room the engine leaves free on
  top of what others hold. A Core that samples no outside memory is judged by the device's reading.
- serve --check's `step_memory` judges the headroom by the engine's own use too. Core's ledger keeps each segment
  the step pool makes while serving with the memory outside the engine read at the same step frame
  (`crit_plugins.memory`'s `other()`), and the outside memory as serving began (`steps()`' `segment_memory`,
  `outside_at_serve`, the first 4,096 segments). The check takes each segment's free bytes as if that outside
  memory had stayed at its least, by the same rule as `full_context_device_memory` (`crit.run.checks.engine_free`,
  which both items now use), and its detail gives the outside memory's range and the device's own count where they
  differ. On the shared WSL GPU the same Flash-Next candidate's least free segment read 342, 334, 244, 237 and
  142 MiB in five checks, the base code's included, so the check failed or passed on what other programs held. A
  Core that keeps no segment readings, or more segments than it keeps, is judged by the device's count. Tests:
  tests/test_run_serve.py, tests/test_memory.py.
- A crit checkout that Windows and WSL share keeps a Core runtime of each: `crit core build` builds `crit_core_steps`
  into `build/core-steps/<sys.platform>` (`win32`, `linux`, `darwin`), and Core's loader (`step_library()`) takes
  only the file its OS opens (`STEP_LIBRARY_FILE`). One `build/core-steps` held one OS's CMake tree, so building
  the other OS's there failed, and on Linux the loader returned the Windows `crit_core_steps.dll` it found in
  `build/core-steps/Release`, which it cannot load. `CRIT_CORE_STEP_LIBRARY` still names any library.
- `crit audit` and the device's readings name Core's step pools: each step's memory pool (Core's allocator under a
  torch.cuda.MemPool, `crit_plugins.memory`'s step_pool()) is listed as `core/steps`, as a captured graph's pool is
  listed by its driver. They were listed as "unnamed" (eleven pools of a test engine, the largest the 32 MiB cuBLAS
  workspace a matrix product keeps in the first step's pool), and tests/test_audit.py's whole-engine audit failed
  on every GPU. A serving engine makes a step pool at a prompt's end, so the names of pools that hold no segment any
  more go every 64 pools.
- model-template's `test_a_request_asking_decode_via_prompt_decodes_as_one_token_prompt_chunks` passes again (it
  failed in DecoderTests, GgufDecoderTests and GgufDenseDecoderTests, and so `crit plugins test model-template` in
  every serve check). Since constrained requests decode in Core's captured rounds, a replayed request's plain decode
  rows run there, where the test's Python spy on the row products sees none. The test now asks that plain decode rows
  are never prompt chunks and that they ran eagerly or in captured rounds (the round counter), and that the
  decode_via_prompt hook's rows are one-token prompt chunks that ran in no round.
- tests/stress.py's `reopen_at_64k` compares the served engine's elastic ranges, read from its telemetry
  (`memory.elastic`), not the stress process's own. The engine runs in crit serve's resident host, so the process's
  registry was empty and the scenario compared 0 MiB with 0 MiB; it now fails where the engine reports no range.
  Measured on the fnext-nvfp4 cut: 17,580 MiB before, 17,580 MiB after reopening at 65,536 tokens and back.
- `crit kernels sweep` times each build as it finishes instead of after every build. paged-kv-attention's 9,103
  configurations took an hour to build before the first was timed, and a sweep stopped in that hour kept nothing
  measured. The builds still run in the pool of CPU workers; the configurations of each finished build are checked and
  timed at once (the earliest planned first, every structure's defaults first), under the GPU lock, which the sweep
  gives back while it waits on the next build. A stop keeps what was measured, records the rest as not measured with
  how many builds were done, and ends the builds still running. Tests: crit_kernels/tests/test_sweep.py (fake builds that
  finish only once the one before was timed; a stop after the first build keeps its timings).
- model-template states its draft width (`draft_width`, the widest width its rounds draft) among its engine facts.
  crit-eval prices a drafting engine's batched rounds (the concurrency levels' drafted ceilings) at it, and without it
  left them unpriced ("draft_width from the adapter"), as for a Flash-Next engine whose 4 clients ran at 1.21x one
  client's aggregate against 2.16x from the one-token cost model. Test: tests/test_model_template.py's
  HybridDraftTests.
- memory-table-rows serves a lookup table that a placement puts off the device (`embeddings:host`). model-template
  refused to load with it ("the table ... lies off the device, and the embeddings plugin does not provide
  embeddings:host"), as a Flash-Next build found for its 30 GB n-gram table: only lookup-table-rows read tables where
  the checkpoint stores them. memory-table-rows' `open(source, width, dtype, device, format=None, scales=None,
  scale=1.0)` now reads the checkpoint's rows once into registered host memory (lookup-table-rows' HostRows with the
  new `resident` source: no file reader, no file kept open), gathered over the link as its `host` option's tables
  are, BF16, F16, F32, NVFP4, FP8 and GGUF rows alike; `prefetch(ids)` has nothing to read ahead. With it, model-template
  also plans the token embeddings as a lookup table (`tables.embed_tokens`), which a placement may put off the device
  as it may with lookup-table-rows. Tests: memory-table-rows' tests/test_device.py (two files at an offset, NVFP4 rows
  equal to lookup-table-rows', a captured lookup), tests/test_model_template.py's OffDeviceMemoryTablesTests.
- A selection pairing draft-trees with a draft head that drafts chains only is refused at resolve, not in the first
  sampled round. model-template's MTP head keeps one draft feature a row and ignores a frontier's `parents`, so a
  tree's frontier of several entries broke its captured round ("output with shape [1, 1] doesn't match the broadcast
  shape [1, 1, 3]", a Flash-Next build's candidate). draft-trees now needs `draft:head-tree`, the fact of a device
  draft head that drafts a frontier along its parents; model-template does not provide it (drafting a tree through the
  MTP layer needs its features and KV per path), so `crit plugins check`, variants and the load name the missing fact.
  Tests: draft-trees' tests/test_plugin.py (refused without the fact) and device tests, tests/test_draft_trees.py
  (model-template provides draft:head-device and not draft:head-tree), tests/test_device_drafting.py with a tree head.
- `crit run ... ablate --data` warms Eval's speed cases and held-out prompts before measuring their seeds, as it
  warms the run's cases (until two requests in a row take the same time per decode round, within 3%). They were
  measured cold: an engine's first requests of a prompt it has not served run slower (expert residency follows the
  content), so in a Flash-Next build's ablations an Eval case's first seed decoded at 0.87 of the case's median and
  its third at 1.11 (65 cases, the first seed the slowest in 85% of them), a drift within each variant that the
  alternating rounds of `--rounds` do not remove. Test: tests/test_run_ablate.py (a toy engine whose first three
  requests of each prompt run cold: unwarmed, its Eval case decoded at 67 tok/s, warmed above 125).
- tests/stress.py's `reopen_at_64k` runs. The scenario (reopen the engine at 64k tokens, prefill a 32k prompt, reopen
  at the run's context and compare the device's elastic ranges before and after) ran only for a load whose extras
  held `"lower"`, and `plan_loads` gave no load that extra in either tier, so a closed engine's pools outliving it
  (the Flash-Next build's leak) went unseen. The first checkpoint's speed load at the natural pressure now carries it
  in both tiers. Test: tests/test_stress_plan.py (every extra in_load reads is planned in the full tier).
- `crit plugins variants` and `crit run ... ablate` no longer refuse every `only:` variant of a selection whose
  `base` leaves out a slot a plugin in use requires. Such a slot's base was `null`, so every variant that switched
  the toggles off emptied it (model-template's embeddings, recurrence, experts) and the plugins refused it; a Flash-Next
  engine build had to name every slot in its base. With the registry, a slot that a plugin in use requires
  (`requires`, `requires_when`) now falls back to the plugin in use, as Core's required slots do, and is no toggle.
  Test: tests/test_plugins.py's ablation tests.
- An engine closed in a process gives its KV and recurrent state pools back to the device. attention-state's
  paged pools and sequential-recurrence's and recurrent-state-recovery's state pools never closed their elastic
  ranges, and the device's registry of ranges (`crit_plugins.memory.DeviceMemory.ranges`) kept them backed, so an
  engine reopened in the same process (crit serve's reload, crit-eval's lower-context engine) started with the old
  pools still holding device memory: a Flash-Next engine build saw 7,424 instead of 8,130 expert cache slots, a
  slower prefill and a failed lower-context pre-check. `KVPool.discard()` and `StatePool.discard()` close a pool's
  range, and the three plugins' `close()` discards every pool they made. Tests: each plugin's device tests close
  the plugin and find none of its ranges left; tests/stress.py's `reopen_at_64k` fails when the engine reopened at
  the first context holds more elastic ranges than it did before.

- A paired ablation (`crit run ... ablate --rounds`) counts a KPI as changed only where its rounds' median also stands
  out from their own spread (their highest change less their lowest). Four rounds of a KPI that scatters by more
  than 0.5% fall on one side one time in eight, so the same engine served twice read as faster: on a host running
  other suites the "same options served again" variant of tests/test_run_ablate.py read `pays` on a prefill whose
  rounds moved +0.2% to +1.2%. A single pass has no spread and is judged as before.

- A stop no longer kills the engine host of a large engine. crit serve's stop gave the engine's close (the unload it
  sends once every request is answered) 5 seconds; a whole-checkpoint mixture of experts takes longer (its host
  tier's threads, tens of gigabytes of pinned memory), so the stop sent a cancel no unload can settle and retired
  the host a second later ("the engine settled no cancel within 1 s", seen on a Flash-Next GGUF engine). The close
  now has the same bound as an unload, at most 120 s, and an unload past its bound is named as one ("did not close
  within N s").
- An engine with a sparse attention no longer compiles its block selection on the first long request. Rows select
  their blocks only once they pass the attention's budget (2,048 tokens on Qwen3.8-Flash-Next), which no request
  Core served at load reached, so sparse-indexer's score and top-k kernels compiled on the first prompt past it, and
  attention-state's block-list reader and merge at query tiles no short prompt had (its warm-up left block lists to
  their first step); the new `compiles_after_load` check found 13 such compiles in a whole-checkpoint serve check.
  A model's info now names the prompt lengths its paths switch at (`warm_prompts`), which Core's warm requests serve
  at load beside their own; model-template names its sparse attention's budget two chunks past; and attention-state
  launches its block-list reader and merge at every query tile on its first call with block lists.
- A GGUF engine whose file stores some matrices dense is no longer refused in preserve mode. GGUF files keep routers,
  small gates, indexers and n-gram maps as F32, F16 or BF16 (the Flash-Next UD-IQ1_M GGUF's 48 routers among them),
  and gguf-projections ran those matrices' prompt rows through cuBLAS's GEMM, whose bits follow the row count:
  `repeatable`, where preserve mode asks `exact`, so Core's loader refused the engine. With `prompt` exact (what
  `auto` becomes where exact rows are asked) a dense matrix's rows now have their decode rows' bits: on crit's kernel
  mma-rows, decode rows on its BF16 decode tile and prompt rows on its prompt tile (the pair quantized-projections'
  `prompt_gemm` tile runs for a safetensors checkpoint's BF16 matrices, both at the K splits mma-rows' plan gives the
  matrix, fewer where a decode call's partial sums would pass 16 MiB), else, and on a CPU device, one row a call for
  prompt rows too. Its decode rows stay `row-exact` and its prompt rows are `exact`; the facts name each dense
  matrix's path (`dense_paths`), and `row_invariance.prompt_exact` is true on the fallback as well. On the CPU the
  fallback's prompt rows were not even repeatable across chunkings before. On the RTX 5090 the Flash-Next GGUF (about
  240 dense matrix calls a decode token: routers, the 4 x 10240 inject rows, the linear attention's gates) decodes
  at 58.9 sampled / 63.3 greedy tok/s in speed mode at 131,072 tokens; on one split of K, as first built, it decoded
  at 44.2 / 46.2 (a 4 x 10240 row 25.5 us on one split, 3.4 us on the plan's 32, torch's GEMV 3.7 us).
  model-template's test of such an engine checks each greedy reply against its own prefill's top tokens; on CUDA
  the engine without a mode multiplies other bits (cuBLAS over the decoded matrices, int8 activations on the
  exact path), so the two replies may part only at a near-tie of the reference.
- A test for crit-labs/crit#3: on a Microsoft Store Python, a guarded engine command that runs a venv's python.exe
  stays in its tree's job, its interpreter and the worker it starts, and closing the tree ends them
  (`tests/test_guardian_launch.py`, on Windows with a Store Python, the one running the tests or the one on PATH;
  skipped elsewhere). Without the desktop app policy the engine's interpreter left the job and outlived the tree.
- crit serve's `/v1/models`, `/api/tags` and `/api/ps` leave out a sealed candidate that does not verify (a
  package an older crit sealed, a model file gone) instead of failing for every client: a run that kept candidates
  from before a crit release dropped the connection on `GET /v1/models`, so crit Eval's `--url` client and other
  clients that list models first could not reach the candidates that do verify. A request naming such a candidate
  is still refused with the reason (409). The answer is kept until the candidate list is refreshed, so a
  failing candidate is not verified again on every listing (Dualshok3, #84).

## [0.21.0] - 2026-11-18

### Overview

crit 0.21.0 adds the library's first model plugin and runs GGUF checkpoints end to end. model-template composes a
model plugin from tested blocks by a layout, so an agent picks a layout instead of writing a model: Llama and Qwen
decoders, Qwen3-Next and Qwen3.5 hybrids, Qwen4Exp (Qwen3.8-Flash-Next's architecture), DeepSeek V3's latent attention
and Gemma 4, with their draft layers. It reads GGUF files under the names its blocks read (llama, qwen2/3, their MoE
forms, qwen35, qwen4exp, gemma4 and deepseek2, and every file's draft block), and expert-offload caches, streams and
computes GGUF experts as stored, in one grouped GEMM a batch of experts and on the CPU tier in the kernel's declared
order. Unsloth's 68 GB UD-IQ1_M Qwen3.8-Flash-Next serves on an RTX 5090 at 131,072 tokens of context at 63.7 tok/s
sampled and 1,377 tok/s prefill at 16,384 prompt tokens. gguf-rows reads the ternary TQ1_0 and TQ2_0 and the Q1_0 and
Q2_0 formats bit for bit and decodes IQ1 and IQ2 faster, and crit serve opens a lone GGUF. Core's step contract has
prefill passes, so a model runs a long prompt layer by layer where its experts live off the device while a cancel
still settles between short steps (the engine ABI is 8: rebuild `crit_core_steps`). Quality mode `speed` holds the
output head in FP8 and both modes now ask exact prompt rows unless a candidate states otherwise. Checkpoints of tens
of gigabytes load reliably: the expert store owns locked host memory written and read past the page cache, a load that
reports progress is not timed out, and the expert cache keeps its margin at a prompt step's real peak. A pre-build
stress test serves layer-cut and whole real checkpoints under VRAM pressure, and builds and releases start only from a
commit where it and the manifest test pass. Plugin caches share one machine-wide ledger with a size cap and a
free-space floor, every cache lives under `CRIT_CACHE_HOME`, and CI runs in shards with uv and a cached Core runtime.
crit 0.21.0 needs crit-eval 0.7.2.

### Added

- A machine-wide bound on the files plugins keep in their caches (`crit_plugins.disk_cache`): every kept file of any
  run or clone is entered in one ledger (`~/.cache/crit/plugin-cache.json`), and before a new one is written the
  least recently used are removed until those kept stay under `CRIT_PLUGIN_CACHE_GIB` (default 100) and the disk
  keeps `CRIT_DISK_FLOOR_GIB` (default 50) free after the write; where it cannot, the plugin builds in memory.
  expert-offload's stores go through it. A run that was never stopped kept its 65 GB store before, and a few of them
  filled the disk; the old rule only asked for twice the store's size free at the time of the write.
- `CRIT_CACHE_HOME` names the one folder every cache crit keeps lives under (default `~/.cache/crit`: kernels, Core
  builds, native libraries, plugin test results and the plugin cache ledger), through `crit_core.cache_home()`; each
  cache's own variable still overrides it. Every path built its own `~/.cache/crit` default before.
- Importing crit_core turns off core files for the process (a soft limit of 1 byte; `CRIT_CORE_FILES=1` keeps the
  system's), as crit serve's host already did: a crashed test or script holding a model's mapped weights wrote a
  69 GB WSL crash dump to the Windows disk.
- model-template keeps a GGUF's quantized token embeddings as the file stores their rows, through an embeddings plugin
  that decodes the rows a lookup reads (`embeddings:gguf`: lookup-table-rows), on the device or read in place from
  the file where the plan puts the table off it; they were decoded whole to BF16 on the device before (1.2 GiB on
  Qwen3.8-Flash-Next). The n-gram table's GGUF path and the embeddings' share one helper (`gguf_table`).
- expert-offload sizes GGUF bundle slots for the layers most experts are in, not the widest: the slot is the smallest
  layer bundle past which the wider layers hold at most a sixteenth of the experts' bytes, and those layers (a file's
  few layers in a wider format, Qwen3.8-Flash-Next UD-IQ1_M's two IQ4_NL layers) are stored in pieces of a slot in
  extra store layers and kept whole on the device at their own size, outside the cache (`reasons.outsized`). Every
  slot was padded to the widest layer's bundle before (about 30% of a slot on that checkpoint). The device tier holds
  each layer at its own size.
- expert-offload runs a GGUF prompt chunk's routes through gguf-rows' grouped GEMM: the routes sorted by slot and
  multiplied in one staging and one GEMM launch for each batch of experts (two calls a matrix), where each slot's rows
  took their own kernel calls and Python steps before (up to a layer's 512 experts, two matrices each). The bits are
  the same: each row the plain call's on its expert.
- expert-offload's CPU tier computes GGUF experts' misses: `GgufCpuTier` runs gguf-rows' declared order on the host
  (Core's new `crit_core.model.gguf_blocks.declared_rows`, the kernel's host reference moved out of gguf-rows'
  bench.py, with the splits the kernel takes on the device) over the bundles in place in the store, so a route it
  serves has the bits a slot gives it; it is in torch, so its rows join after a synchronization and rounds with it are
  not captured (`round_problems` says so). With misses on auto it serves where the cache is too small for link misses
  (a CPU tier the decision tree had no way to give GGUF experts before) and where it is timed faster than the link.
- model-template reads GGUF files of the gemma4 and deepseek2 architectures and every GGUF's draft (nextn) block:
  Gemma 4's norms in pairs around each block, its layer scalar (layer_output_scale), the router's scale and each
  expert's scale (.scale tensors) and its experts as one gate-up stack at the layer's prefix; DeepSeek V2 and V3's
  latent attention matrices, kv_b_proj joined back from the converter's attn_k_b (transposed) and attn_v_b and
  decoded, the router's score correction (exp_probs_b.bias) and the shared experts; the draft block after the text
  layers as the layout's MTP layer (eh_proj as its fc or, for Qwen4Exp, as fc_embedding and fc_hidden by whole blocks
  of their input columns; enorm, hnorm and shared_head_norm as its norms), with a draft layer's GGUF experts handed to
  expert-offload in their own formats. `tests/gguf_checkpoint.py` writes those architectures and draft blocks as
  llama.cpp's converter does; new whole-engine tests against transformers run the Gemma 4 MoE, DeepSeek V3, Qwen3.5
  drafting from its MTP layer and Qwen4Exp with its draft layer from GGUF copies. A GGUF draft layer's experts loaded
  through any experts plugin no longer fail on the re-encoding argument.
- gguf-rows multiplies a mixture of experts' prompt rows grouped by expert (`gemm_experts(x, experts, ids, counts)`):
  the rows sorted into groups, each group's rows times its own expert of a stack read in place (any stride, as slots
  hold them), the experts staged a batch at a time into the 128 MiB staging buffer by one launch and multiplied by one
  launch of the prompt GEMM over every group's token tiles. Each output has the plain gemm's bits for its row and
  expert (so gemv's), tested in every format for groups shorter and longer than a tile. The plain gemm's code path is
  the same kernel with no groups.
- Measured: Unsloth's UD-IQ1_M Qwen3.8-Flash-Next GGUF (three files, 68 GB: 34.8 GB of experts, its n-gram table
  IQ4_NL in the file) loads and serves through the library's plugins at their defaults (model-template,
  gguf-projections, expert-offload with the experts in host memory, lookup-table-rows reading the tables in place) on
  an RTX 5090 at 131,072 tokens of context: a coherent chat reply, 63.7 tok/s sampled and 65.8 greedy decode, 210,
  835 and 1,377 tok/s prefill at 1,024, 4,096 and 16,384 prompt tokens (254, 406 and 411 before the grouped expert
  GEMM), a cancel settled in 0.49 s, the expert cache 22.3 GiB in 13,589 to 15,863 slots of 1.51 MB (12,792 of 1.97
  MB before the wide layers were kept whole) beside 5.8 GiB of weights and pools, the link at 25.9 GiB/s
  (expert-offload's README has the numbers).
- model-template reads GGUF checkpoints (a `.gguf` file, a split set, or a folder holding one): `code/gguf.py` gives
  the blocks the file's tensors under the Hugging Face names they read, from its general.architecture and the
  converters' naming convention (`blk.N.attn_q`, `blk.N.ffn_up_exps`, `blk.N.ssm_out`, `blk.N.hc_attn_down`,
  `token_embd`, `output`, ...), and undoes what a converter changed: norms stored as 1 + w and A_log as -exp(A_log)
  inverted in FP32, Llama's permuted query and key rows and a gated delta net's tiled value heads moved back as whole
  rows or, along a matrix's inputs, as whole blocks of its stored format (no quantized matrix is decoded for it),
  merged tensors read as their halves and the split indexer query and key rows as one, the n-gram table and its hash
  constants (moved to metadata) as the checkpoint's, routed expert stacks as each expert's matrices. Its rules are
  data a GGUF architecture: llama, qwen2, qwen3, qwen2moe, qwen3moe, qwen35, qwen35moe and qwen4exp
  (Qwen3.8-Flash-Next's GGUFs); a tensor that maps to no name is refused at load. Quantized matrices load as stored
  (`gguf:<TYPE>`, gguf-projections), experts as stored into an experts plugin that takes them (`experts:gguf`:
  expert-offload) and as their BF16 values into any other, a table stored in a block format through an embeddings
  plugin that decodes it (`embeddings:gguf`), and the end tokens come from Core's derived generation_config.json. The
  gated-delta block loads its in_proj_qkv and in_proj_z (and in_proj_b and in_proj_a) as the fewest matrices of one
  format, since a GGUF may store the halves in two (Q5_K beside Q6_K). Tested by `tests/test_gguf.py` (every tensor
  of each architecture's rules read back from a file written as the converter writes it, `tests/gguf_checkpoint.py`)
  and as whole engines against transformers on GGUF copies of the harness's checkpoints, their matrices Q8_0: Llama,
  Qwen3-MoE through expert-offload in host memory, Qwen3.5-MoE and Qwen4Exp, on CUDA and on the CPU.
- expert-offload caches and streams experts as a GGUF checkpoint stores them (format `gguf`, the fact
  `experts:gguf`): `begin(..., "gguf", ..., formats={layer: (gate_up, down)})` takes each layer's block formats (a
  GGUF quantizes every layer's stacks in its own: IQ2_XXS, IQ2_XS or IQ2_S gate-up rows beside Q2_0 or IQ4_NL down
  rows in Unsloth's UD-IQ1_M Qwen3.8-Flash-Next), a bundle is an expert's stored rows (`code/gguf_experts.py`), and
  every tier, store file, cache, ring and the link carry them as any bundle. Their rows run on gguf-rows in the slots
  they lie in: decode rows its gathered gemv, prompt chunks grouped by slot on its matmul (the same bits, so a prompt
  row has its decode row's bits). The CPU tier computes no GGUF format, so every miss goes over the link.
- lookup-table-rows reads tables a GGUF stores in a block format (`format="gguf:<TYPE>"`, the fact `embeddings:gguf`):
  its rows are read and gathered as stored, from the device, host memory or the file in place, and a lookup decodes
  the rows it read on the device with gguf-rows' `dequantize` (Core's host decoders where it does not build, which
  `round_problems()` names). Qwen3.8-Flash-Next's GGUFs store its 320-million-row n-gram table IQ4_NL (26.8 GiB): it
  stays in the file and its rows are decoded as they are read.
- gguf-projections runs on a CPU device through its fallback (the stored values decoded on the host by Core's
  decoders, a BF16 matrix through torch with FP32 sums), so an engine on a GGUF checkpoint, and the tests of one, run
  where there is no GPU; on CUDA nothing changed.
- An engine reads a lone GGUF (a `.gguf` file, or a folder holding one and no config.json) as the model it is: its
  load context's `config` and the package check's facts (`crit_core.engine.architecture_facts`) come from the
  config.json Core derives from the GGUF's metadata (`crit_core.engine.model_config`), so a plugin that requires a
  `model_type` (model-template's layouts) finds it, and its matrices are `stored:gguf` (not `stored:plain`).
- gguf-rows reads a stack of experts in place from slots that hold other bytes beside each expert: `[experts][N][row
  bytes]` with contiguous rows at any stride between the experts (refused where experts would overlap), so an expert
  cache's slot array is multiplied where its bundles lie, a layer's gate-up or down rows a view of it.
- GGUF's Q1_0 (type 41: a sign bit a value, +-d, one FP16 scale d a 128-value block) and Q2_0 (42: 2-bit codes 0
  to 3 less 1 times one FP16 scale d a 64-value block) are read wherever the other GGUF formats are: Core's host
  decoders (`crit_core.model.gguf_blocks`), the kernel core's (`crit_kcore/gguf.cuh`), gguf-rows (`gemv` plain and
  gathered, `gemm`, `dequantize`) and gguf-projections (`gguf:Q1_0`, `gguf:Q2_0`). Unsloth's UD-IQ1_M quantization of
  Qwen3.8-Flash-Next stores 46 of its 48 layers' routed down projections Q2_0. The installed gguf-py has no decoder for
  either yet, so gguf-rows' tests check them against Core's decoders, which Core's tests check against the formats'
  own rules (llama.cpp's dequantize_row_q1_0 and dequantize_row_q2_0).
- A pre-build stress test, `python tests/stress.py`, that finds toolkit bugs before an engine build or a release
  does. It serves layer-cut copies of both Qwen3.8-Flash-Next checkpoints (NVFP4 and FP4-Local, four decoder layers
  and the draft layer, every tensor as stored) as `crit run serve --check` serves a candidate of the library's
  plugins at their defaults: speed and preserve, four device memory pressures made by the VRAM ballast (natural; tight,
  the experts in host memory behind a cache of about three quarters of them; extreme; floor), short, 32k and
  262,144-token prompts, a cancel inside a prefill, reopening at a lower context, drafting from the draft layer and
  through the drafter seam (mtp-decode over it), crit smoke, `--data` with crit-eval's gate pre-check, and that
  pre-check repeated to show its spread. Gates that judge the text against a working model's (smoke's `fluent` and
  `recall`, the tool-call check) are left out and say why. Core's build, the runs and their packages, crit-eval's
  data, expert-offload's store and the plugin tests are built once per commit and cached; each load takes the
  machine's shared GPU lock and crit's engine lock and gives them back after; the report is `stress.json` and
  `stress.md`, a scenario a row, and the exit status fails on any failed scenario.
- A VRAM ballast for tests: `CRIT_VRAM_BALLAST=leave:SIZE` (or `take:SIZE`) makes the device's memory object, when a
  process first asks for it, hold all but SIZE of the device's free memory (or SIZE) in an elastic range owned by
  `ballast`, the driver's memory outside torch's allocator, never given back. Every free-memory reading (the
  ledger's, the driver's, the card's) counts it as taken, as it counts another program's, so an engine loads and
  serves as on a smaller device: expert-offload places its experts off the device at a chosen pressure without a
  filler process. `memory.ballast(take=, leave=)` makes one in code.
- Layer-cut checkpoints for tests on real weights (`tests/support/layer_cut.py SOURCE TARGET [--layers ...]`): a
  checkpoint with only the decoder layers that cover every kind of layer it has (by default its first layers up to
  the first of each kind), every kept tensor byte for byte, the draft layer, embeddings, head, norms and lookup
  tables kept, vision and audio towers left out, and config.json, its text config, the quantization configs and the
  index rewritten to the kept layers. A shard whose tensors are all kept is hard-linked, so a cut of Qwen3.8-Flash-Next
  (its 47.7 GB n-gram table in shards of its own) writes 5.4 GB for NVFP4 and 10.5 GB for FP4-Local.
- Checkpoint manifests and a test that plans model-template over every real checkpoint crit targets, without the
  weights. `tests/support/manifests.py` writes a manifest from a checkpoint's safetensors headers and configs (names,
  dtypes and shapes compacted by their numbers, the bytes of small tensors a model reads values of; 1.1 MB for eight
  checkpoints) into `tests/manifests`; `tests/test_checkpoint_manifests.py` runs model-template's own load over each
  one on the CPU (`tests/support/dry_plan.py`: its ComposedModel over the headers, meta tensors in place of weights,
  the slots filled by stand-ins that take the formats the library's plugins take), with its lookup tables on the
  device and off it, and fails on a tensor the load cannot read and on one no part of the model reads (outside
  vision and audio towers). A model's context may hand its checkpoint (`context.checkpoint`). Today's manifests:
  both Qwen3.8-Flash-Next checkpoints (NVFP4 and FP4-Local), Qwen3.8-27B NVFP4, Gemma 4 26B and 31B NVFP4,
  Nanbeige4.3-33B NVFP4, and DeepSeek V4 Flash and Nemotron 3.5 Super, which no layout covers yet (listed, not
  failed). It found the three load gaps fixed above.
- model-template runs a prompt layer by layer where the experts live off the device (passes). Chunk by chunk, every
  prompt chunk of a hundred tokens or more brings nearly every expert of every layer over the link again: gigabytes
  a chunk whatever its size, so prefill ran at the link's rate divided by the experts' size and a single chunk's
  step passed the cancel grace. Where the experts plugin stages prompt chunks (`stages_prompts()`: expert-offload
  with experts in host memory or a file) the template now runs every chunk of a span through one layer before the
  next layer, over as many of Core's steps as that takes (Core's prefill passes; a step ends within `span_seconds`
  and half the cancel grace), keeping each chunk's residual state between layers in device memory the device's
  givers make room for (`pass_mib`). Nothing is tuned by the engine's author: option `passes` is `auto`. Replies
  and scores are the same bits in both orders (tested on every layout, and on both real checkpoints below). On an
  RTX 5090 at 131,072 tokens of context with 1,024-token chunks, prefill tokens a second chunk by chunk against
  layer by layer: Qwen3.8-Flash-Next (68 GB of NVFP4 experts) 557 against 1,266 at 4,096 prompt tokens, 555 against
  2,131 at 16,384, 2,285 at 65,536, the slowest prompt step 2.18 s against 0.62 s; Gemma 4 26B (46 GB of BF16
  experts) 968 against 2,445, 1,050 against 4,727, 4,323 at 65,536, the slowest step 1.39 s against 0.56 s and a
  cancel inside a long prompt settling after 2.16 s against 0.16 to 0.54 s.
- model-template composes Gemma 4 (layout `sandwich` for gemma4 and gemma4_text), from parts any layout may use: a
  `sandwich` residual (each block between two norms, a scalar a layer), a `parallel-moe` block (a dense MLP and a
  mixture of experts side by side, the router scoring the residual stream with a scale an expert), attention whose
  layer kinds differ in head width, key/value heads and rotary table, with value norms, a fixed scale and values
  taken from the key projection's rows where a layer stores no value projection, GELU-gated MLPs and experts,
  embeddings scaled by the square root of the hidden width (`embed_scale`), logits capped by a tanh
  (`logit_softcap`), the family's switch for its experts (`routed_if`) and config values a layout refuses
  (`requires`: the smaller Gemma 4 variants' shared keys and values, per-layer input and double-wide MLP are not
  computed). Tested as a whole engine against transformers on a small Gemma 4 with experts (experts-reference, and
  expert-offload with the experts in host memory), and on Gemma 4 26B itself with its 46 GB of experts in host
  memory: the answer tokens of a chat turn score within 0.002 nats of transformers' own BF16 on average. Its decode
  runs at 20 tokens a second there: the device's time is the attention plugin's (softmax-attention is the one that
  serves sliding windows, and its kernel takes heads up to 256 wide where the full layers' are 512), not the
  experts'.
- expert-offload brings a prompt's experts to the device once a span, and says so. `stages_prompts()` tells the
  model plugin that prompt chunks stage their layers' experts, so a model plugin that runs a prompt's spans layer by
  layer has each layer's experts cross the link once for the span instead of once a chunk; the ring stays staged
  while a prompt goes on (`end_step(prompts_open=True)`) and goes back when it ends or is cancelled. Its facts and
  telemetry carry `prompts`: the link's rate as measured at `open()` and a sweep, the bytes and seconds of bringing
  every expert the cache does not hold to the device once. Where the CPU tier serves decode misses, a short prompt
  chunk's missed routes go to it when they cost less there than their experts' copies.
- device-expert-cache serves a staged layer to every prompt chunk that reads it, across steps. `prompt(layer, ...)`
  staged the layer's non-resident experts for one chunk and the next chunk's call for the same layer staged them
  again. The calls after the first now read the layer's experts where the first found them, `end_prompt(hold=True)`
  ends a step and keeps the ring with what is staged in it (decode plans of other requests run in the slots beside
  it), and the next step's first call stages only what left the cache since. `prompt_cpu(layer, ids, cpu_us,
  link_us)` gives a short chunk a table that sends its missed routes to a CPU tier where they cost less there than
  their experts' copies by the caller's two measured times, copying nothing and borrowing no ring.
- Routed experts can be gated by GELU. The experts slot's blocks computed `silu(gate) * up` alone, and Gemma 4's
  experts and MLPs are gated by GELU in its tanh form. A model plugin now calls `activate(name)` on an experts
  plugin that provides the fact `experts:activations` (grouped-experts, expert-offload, experts-reference): the
  grouped-experts kernel takes the gate in its gate/up epilogue (`Layout.activation`, `crit_kcore`'s
  `gelu_tanh_mul`), its fallback and the FP64 reference compute the same, and the declared W4A4 order and
  expert-offload's CPU tier stay SiLU's (a GELU-gated layer's misses go over the link).
- attention-state derives `proportional` rotary tables, and a config that keys its rope by layer type gives each type
  its own table (Gemma 4: its full-attention layers turn the first quarter of a 512-wide head at base 1,000,000,
  its sliding layers a whole 256-wide head at 10,000). A proportional table is as wide as the head: the first
  `partial_rotary_factor` share of its pairs turns at `base ** (-2 i / head_dim)`, the rest keeps its values.
- Core's step contract has prefill passes. A cancel settles between steps, so a step must stay short, and until now
  every prefill step had to consume prompt tokens: a model could not work on a long span of a prompt over several
  steps. A model whose layers' weights do not all fit on the device needs exactly that, to bring each layer's weights
  to the device once for a span run layer after layer instead of once for every chunk. A model now declares
  `prefill_pass_tokens` (the most prompt tokens a request holds in a pass) and `prefill_pass_steps` (the most steps in
  a row its prefill takes without consuming a prompt token); a step of a pass reports the span (`pass_tokens` of
  `Progress`, `StepProgress` and a step model's `StepResult`) and consumes nothing, and the step that ends the pass
  reports its tokens consumed, past that step's grant by the pass. Core schedules, cancels and retires such a request
  between its steps as any other, and fails an engine whose row passes more steps than declared or stalls without a
  pass. Core knows nothing of what a pass is made of. `StepProgress` grew, so the engine ABI is 8: rebuild
  `crit_core_steps` (`crit core build`); an older library is refused at load with both versions named.
- model-template's test harness sharpens attention (`sharp_attention` on an `Engines` class: the query and key norms'
  weights set to 3, the value and output projections scaled by 3). A small random model attends every position almost
  alike, so a test of a model that limits what a token attends passed whatever the limit picked: with the indexer of
  the gated-streams layout made to take its lowest-scoring blocks, every check of the sparse-attention class still
  passed. That class now sets it, and the same mutation fails three of its checks, on the CPU and through the
  sparse-indexer kernel alike.
- A run's engines keep expert-offload's bundle store between loads. Building the store reads and packs every routed
  expert, minutes for tens of gigabytes, and without a `store_dir` every load did it again (9 to 16 minutes a load
  measured on Qwen3.8-Flash-Next, at every check, Eval and ablation of a candidate). Core's load context now names a
  folder for files a plugin can build again (`context.cache`, the `CRIT_PLUGIN_CACHE` environment variable's), crit
  run sets it to `.plugin-cache` in the run's folder for every engine it serves (a value set already stays; empty:
  none), and expert-offload keeps its store there when no `store_dir` or `store_path` is given: the first load builds
  the file, later loads reopen it and the model skips its expert loads. In the cache the file is keyed by the
  checkpoint, the layout and the code of the package's plugins, the cache holds one store a checkpoint, and a new file
  is written only on a local disk that keeps as much free again as the store takes (`reasons.store` says which
  applied). `crit run stop` removes the run's cache. An engine loaded outside a run has no cache unless the variable
  names one, and behaves as before.
- The library has a model plugin: `model-template` (Inference entry model-template). An agent composes a model plugin
  from tested blocks instead of writing one: `attention` (grouped-query softmax attention, biases and query/key norms
  where the checkpoint stores them, rotary positions from the config's rope table, partial rotation, sliding windows,
  sinks, a sigmoid output gate), `gated-delta` (the delta rule through the recurrence slot, either projection layout),
  `swiglu`, `moe` (the router slot with the router's rule checked against the config, any experts plugin, expert-offload
  included, a gated shared expert) and the checkpoint's MTP layer as the device draft head, arranged by a layout (JSON:
  `decoder` for llama, qwen2, qwen3 and their mixtures of experts, `gated-hybrid` for qwen3_next and qwen3_5), on the
  serving contracts in one place: exact rows (each prompt chunk alone with `prompt=True`, one-token rows in calls of at
  most `invariant_rows`, a step's decode rows and prompt chunks apart in expert-offload), prompt spans timed at load (a
  measured decision: a short and a full chunk at position 0, a full one at the deepest position served) so every prompt
  step finishes within `span_seconds` and the device's room at every context, every device tensor planned, Core's
  captured rounds (greedy, sampled, drafted), captured prompt chunks, the prompt cache, fixed-text scores in the
  prompt's spans. An engine whose architecture no layout covers passes its own layout and blocks to `compose()`.
  Tested as whole engines through Core's loader against transformers on Llama, Qwen3 with a sliding-window layer,
  Qwen3-MoE (experts-reference and expert-offload), Qwen3-Next and Qwen3.5 with an MTP layer drafting in captured rounds.
- model-template composes Qwen4Exp (Qwen3.8-Flash-Next's architecture, layout `gated-streams` for qwen4_exp and
  qwen4_exp_text) and DeepSeek V3 (layout `latent`), from layer parts any layout or engine may use as blocks:
  - **Residuals.** How blocks join the residual is a part of the layout (`residual`): `pre-norm`, one stream as
    before, or `streams`, several residual streams (hyper-connections): each block reads a row mixed from the
    streams' norms by weights the streams compute themselves and its output is added to every stream by weights of
    its own, through the norms slot's stream operations (`rms_norm` with a weight a stream, `stream_sum`,
    `stream_add`, the fused `stream_add_norm`), the low-rank maps through the step's row groups.
  - **`indexed-attention`.** Block-sparse attention: an indexer scores the complete blocks of positions before a
    token (the mean of their raw index keys, normed and rotated at the block's first position) and the token reads
    its best ones within a budget and its own block, by Core's canonical choice (`topk_blocks`) or crit's kernel
    sparse-indexer with the keys read in place. The layers keep one pool of keys only, a row a position, and a
    complete block's key takes the place of its last position's raw key, so the pool follows a sequence through
    steps, rounds, forks and truncations exactly as its KV does. Where the engine's context cannot pass the budget
    the block is plain attention; otherwise every token reads a block list, below the budget too, so a row's
    arithmetic is one whatever shares its step.
  - **`ngram-rows`.** A block ahead of a layer (`before`): rows of a hashed n-gram table (the hash the checkpoint
    stores, or derived as its reference derives it; n-grams stop at an end token) gated into every stream, with a
    dilated causal convolution through the recurrence slot. The table is planned as a lookup table: on the device
    where it fits, else read where the checkpoint stores it (its shards' files, NVFP4 rows decoded on lookup)
    through an embeddings plugin with `embeddings:host`. A step's token windows come from the request on the host
    and from a `TokenRing` inside captured rounds.
  - **The draft layer over streams**: every stream of the target's feature through `fc_hidden`, the next token's
    normed embedding through `fc_embedding` added to each, one decoder layer with pools of its own (its indexer's
    keys too), its own last mix. transformers runs no draft layer for this architecture, so it is tested for what
    can be: drafted replies are the plain ones.
  - **`latent-attention`** and the layout `latent` (deepseek_v3): multi-head latent attention in its absorbed form.
    A position caches one row, its latent and a rotary key every head shares, in a pool of keys only; the queries'
    part without positions is multiplied into the latent space, the attention slot attends with `latent=`, and the
    values' up-projection follows; interleaved rotary pairs and the YaRN magnitude scale by the config. The router's
    rule check knows sigmoid scores within the best groups (a group's score its two best experts').
  - Blocks declare what they keep beside their layer's pool (`pools`, `tables`), which the model plans, makes and
    follows through steps, rounds and the prompt cache (a mixer with `keeps_kv = False` keeps no KV in the model's
    pools); a layout may key its mixers by the config's own layer type names and gives the config values its
    family's reference assumes where config.json is silent (`defaults`).
  - The indexer scores a prompt step's tokens in groups whose scores fit a bound (`score_bytes`, 64 MiB; the
    kernel's workspace the same), since a step's scores are a row of every earlier block a token: a 4,096-token
    step over 50,000 blocks would hold gigabytes at once. A token's choice is its own, so the grouping changes none.
  - It loads MXFP8 matrices (FP8 with one E8M0 exponent per 32 inputs of a row) and NVFP4 matrices whose scales the
    checkpoint keeps CSF-compressed, as the projections and experts slots take them.
  - It serves on the CPU too (its blocks over the slots' CPU forms, decoding through `step()`), which is how its
    engine tests run where there is no GPU: `tests/test_model_template.py` and the plugin's own tests run a whole
    engine of every layout against transformers on CUDA or on the CPU.
  Tested as whole engines against transformers' Qwen4Exp and DeepSeek V3 on the CPU and on CUDA: scores, greedy
  choices, requests together as alone, prompts in steps of other sizes and with end tokens inside, below and past
  the indexer's budget.
- A captured round has the counters its parts name. A part that adds to counters in `count_round` names them in
  `round_counters` (device-expert-cache's cache and expert-offload name `expert_reads` and `expert_hits`), and
  `crit_plugins.rounds.Rounds` has them beside the model's own `counters`, so a model plugin puts an experts plugin
  with a cache into its rounds without knowing what it counts; the model's info names `rounds.counter_names` as its
  `round_counters`.
- `crit_plugins.device.csf_decode`: the decode of ModelOpt's CSF-compressed NVFP4 scales, one definition for a model
  plugin that loads such a checkpoint's matrices and for device-expert-cache's stores and fills (it lived in
  device-expert-cache's `expert_store`, out of a model plugin's reach).
- A plugin's manifest may require slots only where a fact holds (`requires_when`, as `needs_when` does for needs):
  `crit plugins check` refuses a selection that leaves such a slot empty, naming the fact.
- `crit_core.model.routed_layers` (and `ModelConfig.routed_layers()`): the layers whose MLP routes experts, from
  `first_k_dense_replace`, `decoder_sparse_step` and `mlp_only_layers`, one rule for every reader; `crit plugins offload`
  counts routed layers by it (it ignored `mlp_only_layers`).
- paged-kv-attention derives keys from cached values on read (`derive=`, a `Derive(weight, eps, freqs, rotary_dim)`;
  `derive()` packs its tables once): a layer that caches one row per position and KV head keeps value pages only
  (`write(None, values, ...)`), and `attend` (split and fused) and `prefill` read each row as its value and as the key
  `rope(rms_norm(r) * w, position)` in FP32, multiplied as two 16-bit parts, with sliding windows, sinks, trees,
  `lse`, fresh rows and every page kind; a column keeps its bits alone, in any window, fused or split. The flash
  structures, block lists and `int8_qk` refuse it (Dualshok3, #75).
- gguf-rows reads the ternary TQ1_0 (type 34) and TQ2_0 (35) in every call (`gemv` plain and gathered, `gemm`,
  `dequantize`), bit for bit with gguf-py's decoder, with one scale `d` for every 32-value sub-block and codes -1, 0
  and 1; Core's host decoders (`crit_core.model.gguf_blocks`) and the kernel core's (`crit_kcore/gguf.cuh`) read them,
  and gguf-projections takes `gguf:TQ1_0` and `gguf:TQ2_0` matrices (Dualshok3, #74).
- In quality mode `speed` the output head is held in FP8 without the model plugin doing anything for it. An
  engine's configuration states its `quality_mode` (`speed` or `preserve`): `crit run ... candidate` writes the run's
  mode into every candidate beside its `exactness`, Core's loader tells the plugins before any is made
  (`crit_plugins.kernels.use_quality_mode`, `quality_mode()`), and a plugin's precision option left on `auto` follows
  it. quantized-projections gains the first such option, `head` (`auto`, `stored`, `fp8`; default `auto`): a head
  the model plugin loads with role `head` and stores in 16 bits is re-encoded to FP8 rows at load in `speed` (E4M3
  codes and one FP32 scale a row, the FP64 encoder draft-only matrices use), and stays as stored in `preserve` and
  in an engine that states no mode, so engines built before this run as they did. Its facts record the head under
  `head_encoding`. Before, the mode's FP8 head was guidance only and each model plugin had to encode its own. An FP8
  head re-encodes target weights: set `head` to `stored` in a run whose `weights_policy` is `as_stored`.
  The encoder's slices, the head load's warm-up and, at the plugin's next call after the model plugin dropped it,
  the 16-bit copy go back to the device and not into torch's cache (a head of 248,320 by 2,560 left about 7 GiB
  cached otherwise, and a re-encoded draft-only matrix up to 768 MiB of slices), so what sizes itself by the device's
  free memory can use what the head gave up. docs/PLUGINS.md#precision has the rule.

### Changed

- The stress test's full tier ends with one load of a whole checkpoint (Qwen3.8-Flash-Next FP4-Local at 200,000
  tokens, every gate kept): the first load's expert store, the device memory left at the full context and
  decode-against-prefill drift at the model's own depth showed in no layer cut. A ladder of prompt lengths
  (`prompt_ladder`, in that load and at the tight pressure) fails where a longer prompt is prefilled at a lower rate
  than a shorter one, which is how a cost every prompt pays shows. The loads give attention-state's KV pools
  `elastic`, as `crit plugins offload` tells builds to, and report how long after `serve --check` began the engine
  answered (`ready`).
- A candidate that states no exactness asks exact prompt rows in `speed` as in `preserve` (decode rows `row-exact`,
  prompt rows `exact`; `crit_kernels.exactness.QUALITY_MODES`). `speed` gave `repeatable`, every plugin's fastest
  way, and no engine whose rows multiply quantized activations could pass crit's own serving check so: on
  Qwen3.8-Flash-Next FP4-Local the library's plugins at their defaults failed `causal` (a 17-token prefix 0.38 nats
  from its prompt, limit 0.15: the native prompt GEMM's sums follow the row count, and a 4-bit rounding turns their
  last bit into tenths of a nat), with pinned GEMMs `decode_matches_prefill` (0.57, limit 0.5), and passed only
  asking exact (every row 0.0000 nats from its prompt's, prefill and decode 2 to 3% slower: 1,465 against 1,495
  and 63.0 against 65.2 tok/s). The modes now differ in the precision of what is summed, and a
  lower level is a lever an engine pulls itself, with its speed measured and `serve --check` passed.
- `python tests/stress.py --tier quick` runs one speed load a checkpoint at the natural pressure and the first
  checkpoint at the tight pressure, for every commit; the full tier (the default) runs every scenario before an
  engine build or a release. The first load no longer repeats crit-eval's gate pre-check for its spread (it samples
  each case three times itself since crit-eval's statistical lower-context item); `--spread` still does.
- The stress test measures each prompt arithmetic of expert-offload where the CPU tier computes decode misses, on
  the twelve-layer cut with its experts in host memory: tensor (the speed mode's default) prefills a 32k prompt at
  5,962 tok/s with smoke's decode-against-prefill mean delta 0.110 / 0.157 (after 16 and 200 tokens); declared
  (the decode rows' order) at 2,916 tok/s with 0.149 / 0.110. The drift does not come from that choice, and the
  default stays.
- `crit run serve --check --data` opens the engine again twice for crit smoke's lower-context pass and the gate
  pre-check, where it took three reopens: the pre-check now runs first and ends at the lower context, where smoke's
  second pass starts, and smoke leaves the engine at the run's context. Each reopen of a host builds the engine
  again (about a minute on a layer-cut Qwen3.8-Flash-Next).
- The stress test also serves a twelve-layer cut of the NVFP4 Qwen3.8-Flash-Next (`fnext-nvfp4-deep`: nine
  gated delta layers and three attention layers with their indexer, every layer's experts, the draft layer), where
  an error that grows with depth shows: smoke's decode-against-prefill mean delta is 0.14 there against 0.05 on four
  layers, and `causal` fails at 0.152 against its 0.15. `--no-store-cache` builds expert stores in host memory
  instead of the plugin cache, for a disk without the room, and the ballast leaves the lookup tables out of what the
  device must hold, as the selection places them.
- The stress test's cache follows the code it builds from (core, inference, cli, api, smoke, models, kernels), not
  the commit, so a commit to the harness, the docs or the tests keeps it warm; and its pre-check spread compares
  every run of the load, `crit run serve --check`'s own among them.
- The docs say how crit is checked on real checkpoints (Contributing: the manifest test and the stress test), and
  Building engines that a build, like a release, starts only from a commit where both pass.
- One re-encoder for matrices only a drafter reads, `crit_plugins.encodings`: a stored matrix's exact values (BF16,
  FP8 per row, per tensor and in blocks) and their NVFP4 or FP8-per-row encoding, code for code what
  quantized-projections' reference codec gives. quantized-projections' draft re-encoding (`load(..., encode=)`) is
  now this module, and model-template uses it for a draft layer stored in a format no slot loads.
- The kernels workflow builds, for a pull request, only the kernels whose folders it changes, and every kernel when it
  changes anything else under `kernels/` (Core's kernel headers, crit_kernels) or the workflow; Markdown outside a
  kernel's folder builds none. Each kernel's job installs torch and the CUDA toolkit before building, for 2.5 to 11
  minutes.
- CI's native builds run side by side: Core's SDK with GCC, Clang and MSVC each on its own runner (Inference against
  the installed SDK and the SDK's consumers in the GCC and MSVC jobs, which install it), and Inference's in-tree build
  in a job of its own, every build on the runners' four cores. One job an OS ran them one after another, 6.5 to 10
  minutes on Windows.
- The host tests run in shards, each on its own runner: four on Windows and two on Linux, for each Python.
  `tests/run_parallel.py --shard I/N` fills the shards by the time each unit took
  ([tests/durations.json](tests/durations.json)), longest first, so a platform's shards end together, and every unit
  runs in exactly one of them. A unit is a test module, or one test class of a module the recorded times split (twenty
  of them: `test_api_dialects` alone took 430 to 515 s on Windows, more than a shard's share). Windows' 2,600 to
  3,200 s of tests, four at a time on one runner, set CI's pace at 17 to 18 minutes; a Windows shard now holds about
  840 s of them. main's runs record each unit's time and fold them into the artifact `test-durations`
  (`run_parallel.py --merge`); committing it keeps the shards even as tests grow. The wheel is built and checked once
  per platform and Python, in shard 1.
- CI keeps Core's step runtime (`crit_core_steps`) between runs, keyed by `core/`'s files, and builds it on the
  runners' four cores when `core/` changed: every host job built it again, for 35 to 75 s.
- CI installs its pinned Python dependencies with uv (0.13.8) rather than pip, and keeps uv's cache between runs: the
  setup every host and native job starts with took 100 to 120 s on Windows, most of it pip installing torch, and the
  kernels workflow's torch, CUDA toolkit and ninja about 100 s a job. The documented-install jobs still install crit
  with pip, as Contributing gives it.
- The device ledger's `settle()` is bounded, and a score's step frame says it is a score's. Backing device memory
  costs the driver time by the granule (half a millisecond each under WSL), and an elastic expert cache grew back
  everything a prompt had borrowed in one call, inside the frames of the prompt's last step and of a cancel: two
  gigabytes took 0.56 s there, on Qwen3.8-Flash-Next the difference between a prompt step of 0.6 s and one of
  1.17 s, past crit serve's one-second cancel grace. `settle()` now grows the givers back 64 MiB at a time for 50 ms
  at most and says whether more is left (`owed`); the following steps' settles go on, and while memory is owed
  Core's watch asks the engine for a step of its own between native rounds, as it does when another program takes
  memory. `serve()` still takes everything at once (no step waits on it). Core also tells a begin frame that takes
  the argument that its step is a score (`begin_step(prompt_tokens, score=True)`): a score holds its rows' logits
  beside its chunks' temporaries, and expert-offload had learned that peak as the room of a prompt chunk of as many
  tokens, so every prompt gave back and took again 1.7 GB it never used. The telemetry's `core.prompt_tokens` counts
  the tokens a step computed (a prefill pass's in the step that ends it), `core.pass_steps` the steps in which a row
  worked on a pass, and a probed engine's slowest step of each kind names its parts (`slowest_parts`: begin frames,
  the model's step, end frames). The device contract gains what the experts slot's plugins say of themselves for
  the changes that follow: `stages_prompts()` of `experts:offload` and the fact `experts:activations`.
- The recon scaffold asks for the reference's configuration class beside its modeling file: the defaults it gives the
  keys config.json leaves out. Core reads such a key as unset (`MoeConfig.norm_topk_prob` None) and fills in no
  family's default, which a model plugin written from the config alone took for false where the reference renormalizes
  (model-template's layouts state these defaults, `defaults`).
- gguf-rows: faster decode GEMVs for IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS and IQ2_S, bit for bit: the kernel core's IQ
  decoders read a block's fields with 16-bit loads (IQ1_M's scales with one 8-byte load) where the address allows and
  apply the signs and IQ1's offset with one 32-bit add a word; gemv's new knobs `COLS` (output columns a warp) and
  `GRID` (the format's grid in shared memory, persistent blocks) and `WARPS` 16, swept per format, shape and GPU.
  bench.py checks each configuration against the declared order of its own split count (Dualshok3, #76).

### Fixed

- `crit run plan` no longer leaves a speed run with no way past a parent plan whose `prompt_activations` misstated
  what its engine served when no format ran at BF16, for example the shorthand `"stored"` beside a Q6_K format that
  has no kernel for its stored activations and serves FP8. The check asked for a quality rejection of the stored
  format that was never measured. The next plan now states what was served and why with a gaps entry
  `activation:served` (answer `cause`), recorded as `inputs.served_activations`. A parent that served BF16 still needs
  a measured rejection (`activation:stored`) (Dualshok3, #79).
- `crit run plan` accepts a `check:resources` repair of a parent whose benchmark failed the resources gate by its own
  memory samples (free host or device memory below the run's headroom) while every check record of it passed. A
  candidate that passes its Eval and every check but runs the host short of RAM during `crit run measure` is never
  eligible, and its repair was refused as "did not fail: resources". The plan now cites the benchmark in
  `inputs.objective.check_repairs` (Dualshok3, #80).
- `crit plugins offload` leaves two choices to expert-offload that it used to pin. `prompt_arithmetic`: the plugin's
  `auto` picks the faster arithmetic that keeps the level the engine asks of its prompt rows; the plan set `tensor`
  in every speed run, which with the CPU tier computing decode misses is the pair that failed
  `decode_matches_prefill`. `misses`: where the placement prices the CPU below the link the plan gives `auto`, so
  the plugin times a missed expert both ways in the loaded engine and takes the faster; the plan pinned `cpu` on
  floors measured on an idle device, where the loaded Qwen3.8-Flash-Next engine measured the CPU tier at 502 us a
  missed expert (415 of it the rows' turn to the host and back) against 103 over the link.
- model-template sizes the spans of a prompt run layer by layer (a pass) by what a chunk takes through one layer,
  the unit a pass's step ends after (`Spans(units=)`), not by the fit's seconds through the whole model with its
  fixed part left out. The fit times whole chunks with the experts plugin's copies in them and reads a short
  chunk's fewer copies as the rows' cost: with routed staging it cut Qwen3.8-Flash-Next's chunks from 512 rows to
  160 and an 8,192-token prompt's prefill from 1,400 to 660 tok/s.
- device-expert-cache stages for a prompt chunk of up to a few hundred tokens only the experts the chunk routes,
  where those are at most 85% of the layer's experts that are not resident; a later chunk of the span has what it
  routes beyond them staged beside, and once a span's chunks together route more than that share the later layers
  are staged whole, ahead, as before. Every chunk of 52 tokens or more had every non-resident expert of every layer
  copied: about 2.4 s on Qwen3.8-Flash-Next (54 GB over the link) whatever the prompt's length. A 96-token prompt
  there now takes 0.8 s and a 192-token one 1.1 s.
- expert-offload's elastic cache keeps `margin_mib` free by looking at the step in flight, not by predicting it. A
  prompt step was given the room such steps had taken before, and one that took more (the first of its kind, a
  recurrence scan's workspace, attention at a deeper position) ate into the margin: Qwen3.8-Flash-Next engines left
  0.23 to 0.43 GiB free at the full context against 0.25 asked, a different figure every load, and builds answered
  by raising `margin_mib`. The cache now looks before each routed layer of each prompt chunk (torch's own count of
  its blocks: no call into the driver where nothing grew), gives slots back where the device has less free than the
  margin and a reserve, and sizes that reserve by the largest growth between two looks of the last 64 prompt steps
  (64 MiB at least). The facts count the looks and the times one gave memory back (`elastic.looks`, `extends`,
  `reserve_bytes`). `margin_mib` is the run's VRAM headroom and nothing a build tunes.
- model-template gives its recurrence pools the rounds' rows, so sequential-recurrence's `commit: checkpoints` and
  `recompute` load in captured rounds. They keep a state per round slot and were refused ("give the round's rows"):
  an option the plugin offers could not be selected with the template.
- expert-offload packs its bundles on the host (`host_loads`: model-template hands the plugin CPU tensors), from a
  checkpoint whose layer is read past the page cache (`Checkpoint.hold`: direct reads of the layer's tensors, after
  the checkpoint's hashed pages are dropped). Every expert went to the device and back before, 8 to 36 ms an expert
  against 1.5 ms on the host, and its pages came through the checkpoint's mapping, which under memory pressure
  reads at a fraction of the disk's rate. On Qwen3.8-Flash-Next a first load serves after 11 minutes with the
  65 GiB store built and written.
- expert-offload's store of off-device experts lives in anonymous host memory that the store locks as it fills, is
  written to its file once, past the page cache (`crit_plugins.host.write_memory`), and is read back from a complete
  file the same way (direct reads into the memory, `Reads`' whole-block path), saying how far it is
  (`LoadContext.working`). It was a mapping of the file: on WSL2 the kernel dropped the mapping's clean pages while
  the load went on, so registering 65 GiB for the device read them back from the disk at tens of MB a second and
  the first `crit run serve --check` of a Qwen3.8-Flash-Next build timed out. A later load now opens the store in
  about 70 s.
- quantized-projections' `mxfp8_activations` has `auto`, its default: MXFP8's own activations (`mxfp8`, W8A8) in
  quality mode `speed`, 16-bit ones (`bf16`) in `preserve` and where no mode is stated, as it was; the facts say
  what auto became (`chosen`). A speed run's plan must declare a quantized format's stored activations, and the
  library's default was the 16-bit ones: a candidate of the library's defaults served other activations than its
  plan declared.
- A load that reports its work is not timed out while the work advances, and one that runs out says why. crit serve
  bounds a load by 600 s plus the model's bytes at 200 MB/s and cancelled it at that bound whatever it was doing; the
  failure read "Reply timed out". A plugin now says what a long step of its load is doing
  (`context.working(what, done, total)`, Core's `LoadContext`), the host passes it on as the loading status's `work`,
  crit serve prints it every 30 seconds ("loading: expert store: experts packed 12,032 of 25,088") and lets the load
  go on past its bound while the work last advanced under 5 minutes ago. A load that does run out says "The engine
  did not load within N s", where the bound comes from and what the load last reported.
- `crit run ... stop` refuses while an attempt is reserved and has no result, naming it and the `settle` that records
  its outcome. It was accepted with a check attempt still open, and it removes the run's plugin cache, which that
  attempt's engine may still have mapped.
- `crit run ... next` says before a candidate's seal that a sealed candidate takes no attempt, so the options the
  next plan needs measured on that engine are swept first (`ablate --set`), and the refusals of `ablate` and `serve`
  on a sealed candidate say what to do instead (`serve` prints the `crit serve` command that serves it). A build
  that sealed a candidate and then wanted an option sweep had to package a copy and seal it as a draft.
- A candidate's proposal that names no `parent_candidate_id` takes its plan's. It had to repeat the parent the plan
  already names, and one that left it out was refused as starting "from None".
- `crit run serve --check` compares the prompt activations the candidate's plan declared (`prompt_activations`) with
  the ones the served engine's facts state (`plan_activations`). A candidate whose configuration left an option at
  another activation than its plan named (a plan of stored activations over an MXFP8 checkpoint whose projections
  stayed at 16-bit ones) was only refused at the next plan, after a whole Eval of the wrong engine.
- A served request that names no sampling takes the model's own (`crit_core.serving.served_sampling`): the run's
  frozen recon's `sampling`, else `generation_config.json`'s temperature, filters and repetition penalty where the
  model samples by default, for each field the request leaves out and the engine applies. Core's chat adapter passed
  such a request on with no sampling at all, so every engine answered it greedily, whatever AGENTS.md said of the
  served default. A request with temperature 0 stays exactly what it sent (Eval, crit smoke, the serving checks).
- expert-offload's elastic cache keeps `margin_mib` free at a prompt step's peak: it gives a step the device bytes
  such steps took from the device at their peak (what the device holds outside the memory ledger's elastic ranges
  over the step's start, and torch's blocks at the peak over its end), the reading the run's device-memory checks
  make; a step of the load is measured from an emptied allocator, as a served prompt starts. It counted torch's
  reserved bytes instead, with the blocks the allocator kept unused from the prompt's earlier steps as room in use;
  a block a larger tensor cannot reuse (attention's grow with the context) is not room, and the step took new device
  memory beside it. Every long-context engine missed `full_context_device_memory` by a few tens of MB
  (Qwen3.8-Flash-Next at 200,000 tokens: 0.22 GiB free at the peak against 0.25), and no other setting moved it: the
  cache takes back whatever a smaller KV cache or shorter prompt chunks free. Only raising `margin_mib` hid it.
- gguf-projections made without a device in its context loads on the CPU where the host has no CUDA (the fallback),
  where it loaded onto `cuda` and failed: its own interface test failed on a CPU-only host.
- expert-offload makes the `store_dir` folder it is given when it is not there yet: a ram-tier store in a folder that
  did not exist failed to open with FileNotFoundError (the disk tier's store made its folder already).
- crit serve serves a lone GGUF checkpoint: the resident engine opened the served model at the folder of its
  config.json, which a GGUF without its base model's files does not have (Core derives it), and the load failed with
  `KeyError: 'config.json'`; it now opens the folder the verified files are held in (`ModelContext.model_folder`).
- A paged range asks the device's ledger for the granules its growth maps, not its units' bytes. A unit smaller
  than a granule backs a whole one, a plane at a time, so a KV pool of many layers took up to a granule a plane more
  than its givers had made room for: on a layer-cut Qwen3.8-Flash-Next with its experts in host memory the free
  memory at a full-context prompt step fell 10 MiB under the 0.25 GiB headroom the expert cache keeps. Found by the
  stress test.
- A score's logits come out of the device's free memory only after its experts plugin made room for them. A score
  holds a slice of 128 rows' FP32 logits beside their log-softmax and, with top_k, the sorted rows and their int64
  order: on Qwen3.8-Flash-Next's 248,320-token vocabulary about 1.5 GB a slice, which no prompt step of its size
  takes, so an elastic expert cache that keeps its margin and a step's measured room left the first score of a load
  1.2 GB short (the driver then moves allocations to system memory, under WDDM silently), and `crit run serve
  --check` measured 0.22 GiB free at a prompt step's peak against a 0.25 GiB headroom. model-template now asks the
  device's ledger for each slice's bytes first (`crit_plugins.memory`'s need), as KV pages are asked for: the cache
  gives them back before the slice is made. Found by the stress test.
- expert-offload takes the CPU tier for the misses where the device cache is too small for them over the link. Over
  the link every miss takes a slot, so a decode plan keeps a layer's experts beside the prompt ring's; with fewer
  slots, misses `auto` chose the link by its timing and the cache could not be made. Where the CPU tier computes the
  experts (W4A4 NVFP4, INT4) misses `auto` now takes it, which needs no slots beside the ring, and the facts say why
  (`reasons.misses_small`); with misses pinned to the link such a cache is refused at `open()`. Found by the stress
  test.
- An expert cache that could stage no prompt is refused when it is made. A cache too small for a prompt ring of a
  layer's experts beside the slots a decode plan keeps (a layer's again, without a CPU tier) took a ring of 0 slots,
  opened, and failed at its first prompt with "the cache has no prompt ring (ring_slots 0): it serves decode plans
  only": inside model-template's span calibration, at load, on a layer-cut Flash-Next with 11.0 GiB free and its
  experts in host memory. It now names the slots prompts take, their bytes, and the remedies; a decode-only cache is
  asked for with `ring_slots` 0 (expert-offload passes it from its settings file). Found by the stress test.
- model-template reads the token embeddings where the plan puts them. A placement that puts the lookup tables off
  the device (device-placement's `off_device: "tables."`, as an engine for a checkpoint whose n-gram table is larger
  than the device sets it) put `tables.embed_tokens` there too, and the template stored the rows on the device
  anyway: 1.2 GB the plan did not count. On a layer-cut Qwen3.8-Flash-Next the speed mode's FP8 head hid it (half the
  head's planned bytes), and every preserve load failed Core's plan check ("its factory took 402.1 MiB of device
  memory beyond the placement plans"). Off the device the rows are now read in place through an embeddings plugin
  with `embeddings:host`; token embeddings that cannot be read so (not BF16 rows in a file) are planned as a device
  weight. Found by the stress test.
- model-template loads a draft layer whose routed experts are stored in another format than the decoder's, as
  NVIDIA's NVFP4 Qwen3.8-Flash-Next stores its MTP layer's experts FP8 in 128 x 128 blocks beside NVFP4 decoder
  experts. They are re-encoded at load to the format the experts slot holds the others in (NVFP4 without input
  scales there, W4A16; BF16 where the decoder's are BF16) through `crit_plugins.encodings`; the target verifies every
  draft, and the model's facts say so (`mtp.experts`). FP8 in blocks is now read as stored (`fp8:block`, with its
  multipliers); a projection still refuses it by name unless it is a draft matrix re-encoded by `mtp_encode`. The load
  failed with "FP8 in blocks of (128, 128), which the projections slot does not load".
- model-template reads a hashed n-gram table stored as FP8 (E4M3 codes with one multiplier for the table,
  `weight_scale`), as NVIDIA's NVFP4 Qwen3.8-Flash-Next stores its 47.7 GB table, and lookup-table-rows serves FP8
  rows (format `fp8`: a byte a value, decoded on lookup times the table's scale) on the device, in host memory and
  from files. The table off the device failed the load (`KeyError: 'F8_E4M3'`); on the device it was read as its
  codes cast to BF16 with the scale left behind.
- model-template drafts from a checkpoint whose decoder stores static KV scales and whose draft layer does not.
  With `kv auto` the target's pages take FP8 with the stored `k_scale` and `v_scale`, and the draft layer's attention
  asked for its own as well, though its pool keeps `mtp_kv` pages that take none: loading Qwen3.8-27B NVFP4 with its
  MTP layer failed on `mtp.layers.0.self_attn.k_scale`. Found by the checkpoint manifest test.
- expert-offload timed a missed expert wrong at load, and its `misses: auto` then chose the CPU tier on a machine
  whose link serves a miss three times sooner. Three things skewed the link's time, each by several times: the
  CPU tier's threads, awake from their own timing, took the processor the copies' host side runs on; a device left
  idle while the engine loaded had its link at its lowest speed (an RTX 5090: 3.5 GiB/s for the first 0.2 to 0.4 s
  under load, then 25), and its first batches agreed with each other at that speed; and the CPU tier's time left out
  the way of a layer's rows to the host and back, which a decode layer with a miss there pays (180 to 240 us under
  WSL) and a copy never does. The link is now timed once, before the tier exists, until its rate has stopped
  doubling, and a CPU miss counts its computing and its handoff (`miss_times_us`: `compute`, `handoff`; every timed
  batch in `miss_trials_us`). Qwen3.8-Flash-Next on an RTX 5090: 100 us over the link against 275 to 365 on the CPU
  tier, where the old timing read 740 against 90; with the misses on the link, decode without drafts went from 16
  to 31 tokens a second to 50, and prefill, whose rows no longer run the CPU tier's declared order, from 945 to
  2,285 at a 65,536-token prompt.
- expert-offload's elastic cache grew back everything a prompt had borrowed inside the prompt's last step and
  inside a cancel (0.56 s for two gigabytes under WSL, a 1.17 s step), and learned a score's memory, its rows'
  logits, as the room of a prompt chunk of as many tokens, so every prompt gave back and took again 1.7 GB it did
  not use. It now grows back a bounded settle at a time (the device ledger's) and keeps a score's peak apart from
  the prompt chunks'.
- Three of Core's step engine checks were decided by the threads' timing and failed about one run in four: a request
  cancelled inside a prefill pass was asked to cancel only while a pass was half done, decode rows shared a pass's
  steps only if the requests happened to arrive together, and a step asked for between native rounds was taken by a
  late request's prompt step. Each now sets up the order it checks (a cancel from the first pass step on, a gate that
  holds a step until the second request is queued, one request alone); thirty runs in a row pass.
- model-template refuses a drafter that never asks its draft layer. mtp-decode at its default (`override` 0) proposes
  prompt copies alone: selected beside a checkpoint's MTP layer it left that layer loaded, written at every prompt
  position and caught up every round while every draft was a copy or cut, so no draft was ever the layer's
  (Qwen3.8-Flash-Next: 0 of 285 drafts accepted, 22 tokens a second). A drafter now says whether its configuration
  asks the model's draft head (`uses_head`), and the template refuses one that does not, naming what to select:
  greedy-head-drafter for the layer's chain, or mtp-decode with an `override`. With the chain the same engine accepts
  81% of a greedy reply's drafts and 58% of a sampled one's, 62 tokens a second sampled. A new test gives the
  gated-streams draft layer weights whose drafts are known, since transformers runs no draft layer for that
  architecture.
- model-template plans a large checkpoint's tensors in seconds. Its load built the set of the offloaded experts'
  tensor names once for every tensor it planned: on Qwen3.8-Flash-Next (375,487 tensors, most of them routed experts')
  the load spent over half an hour there before reading its first weight, and had not finished when it was stopped.
  The set is built once.
- A plan whose lever leaves a field out is refused with the lever's place and id and what each field is (`group`: the
  group it pays in, or null where it pays alone) instead of `Missing required fields: group`.
- `crit run build` packages a source that carries its own `configuration.json`. A candidate's source often holds the
  configuration the candidate was made from, as it was written; crit completes the candidate's (the recon's frozen
  block, the exactness of its rows, the quality mode), so the package step refused the source's copy as another
  configuration until the build was run with `--drop configuration.json`. The build folder now gets the candidate's
  completed configuration wherever the source's copy differs in nothing but what crit completes. A copy that names
  other values is still refused, and the refusal now says what to do.
- `crit run serve --check` no longer waits out a prefill that cannot finish. Its full-context request waits while the
  engine keeps finishing steps, bounded only by the attempt's deadline, so an engine that prefilled a 200,000-token
  prompt at 37 tokens a second held the check for 43 minutes until that deadline stopped it. After 30 seconds of
  prompt steps the wait now projects the rest of the prompt at the rate measured so far (a floor: prefill slows as the
  context grows), logs that rate every minute, and fails both full-context items with `prefill too slow, about N
  tok/s` as soon as the projection passes what the attempt has left. The rate comes from a new Core counter, the
  prompt tokens its steps prefilled (`StepEngine.telemetry()`'s `core.prompt_tokens`).
- model-template reads a checkpoint's shards where its index names them. Its reader took the safetensors files at the
  top of the model folder only, so a checkpoint that keeps its shards in a subfolder (`tensors/`, named by
  `model.safetensors.index.json`) had no tensors for it. It now reads the index through Core's inspector
  (`crit_core.model.safetensors.inspect_model`), adds the files beside the index that the index does not list (a draft
  layer stored apart), and refuses a tensor that two files hold.
- A captured round gives a cut draft a token id a table has a row for. A drafter cuts a row's drafts with -1
  (mtp-decode's copy drafts before a slot has tokens, for one), and the round's batch carried that -1 to the model's
  verify: an embedding lookup raised `a row outside the table` in the capture's warm-up rounds, and inside the
  capture, where a lookup cannot check its ids, read outside its table. The batch's columns (and `view.columns()`) now
  hold token 0 for a cut draft; `view.drafts` keeps the -1, which no prediction matches, so nothing past a cut draft
  is kept.
- expert-offload loads a checkpoint whose draft layer's experts are weight-only NVFP4 (W4A16) beside decoder experts
  that carry input scales (W4A4), in every tier. With the experts in host memory and no store file to leave that
  layer out of, `open()` added the layer's experts to the CPU tier, which computes W4A4 experts only and refused
  them: the engine did not load (with a `store_dir` it did, since the layer was left out of host memory and so out
  of the tier). In a file tier the same checkpoint was refused outright (`W4A16 layers beside W4A4 ones need a mapped
  store`), although every miss there goes over the link, whose tensor arithmetic reads each layer as it is stored.
  The layer stays whole on the device only where the CPU tier may serve misses, is never the tier's (nor timed when
  `misses: auto` times a missed expert), and is cached as the others where the link serves. A tier that refuses an
  expert sends the misses over the link with the reason unless `misses: cpu` asked for it.
- An elastic expert cache asked for more memory than it can spare keeps the slots a plan needs. Without a CPU tier
  (misses over the link) a miss must find a slot, so the cache keeps a layer's every expert beside its ring; a shrink
  went down to the ring and one slot, and after a large release (a KV pool backing a long context's pages through the
  ledger) the next decode plan found no victim and failed the request (`the cache lost a route`). A cache that keeps
  its size (`elastic: false`, no elastic ranges on the device, or one that holds every expert) now leaves the model's
  elastic pools the bytes they have reserved and not backed yet (it took them at `open()` and could not give them
  back, so a KV pool sized for the whole context ran out of device memory as its sequences grew); its facts say how
  much.
- expert-offload over a store that keeps scale components CSF-compressed (the expert source's `csf_scales`) no longer
  builds its CPU tier over bundles of another layout: on `misses: auto` the misses go over the link, with the
  reason, and `misses: cpu` is refused by name before the tier reads a byte.
- model-template's `gated-hybrid` layout routes Qwen3.5-MoE as its reference does: that router always renormalizes
  the kept probabilities and its config has no key for it, so the template's router check demanded
  `renormalize=False` and an engine either was refused with the right router options or served unnormalized expert
  weights with the wrong ones. The layout states it (`defaults`), and a Qwen3.5-MoE engine is tested against
  transformers.
- A model plugin whose rounds ran an experts plugin with a device cache (expert-offload with its experts in host
  memory) failed to capture them unless it happened to name the cache's counters: `count_round` asked the round for
  `expert_reads` and got a ValueError from a tuple lookup. model-template was such a plugin, so its engines captured
  no round over a host-memory tier. The round driver now takes the counters from the parts (above), and a counter no
  one named is refused by name, saying where to name it.
- model-template's prompt spans account for a step's fixed part, such as an experts plugin's copies of a chunk's
  routed experts (expert-offload over host memory: gigabytes a step, whatever the chunk's size past a few dozen
  tokens). The budget was half a second and the rule took the shortest span whenever no span met it: with about
  half a second of copies a step that is every position, so an engine prefilled in 16-token steps, each paying most
  of the copies again (37 tokens a second measured on Qwen3.8-Flash-Next with its experts in host memory; a
  200,000-token prompt takes over an hour so). `span_seconds` now defaults to 0.75, three quarters of crit serve's
  cancel grace: the fixed part is paid once a step, so the budget sets the rate (with 0.45 s of copies and 0.48 ms a
  token, 0.5 s gives 96-token steps and 190 tokens a second, 0.75 s gives 624-token steps and 830). Where the fixed
  part alone passes the budget, the span is the one that prefills fastest by the fit, among those predicted to end
  inside the grace where any does, never the shortest. The spans' facts give each step's predicted seconds and rate
  and how much of it no span changes (`steps`, `over_budget`).
- Core's free device memory is never more than the card has left. On Windows and under WSL the driver's free
  reading is the process's own budget: it does not fall when another program takes memory (measured: with other
  programs holding 9.7 GiB of an RTX 5090, a fresh process was told 30.09 GiB were free where the card had 21.32
  GiB left). A cache sized from that reading, or a giver growing back into it, took memory another program held
  and the driver moved somebody's bytes to system memory. `crit_plugins.memory`'s `free()` is now the lesser of
  the driver's reading and the card's own count (`card_free`: NVML through ctypes, the device found by its UUID;
  0.12 ms a reading, no new dependency), so `relieve()`, `watch()`, the givers and expert-offload's sizing at
  `open()` all see what other programs hold. Where the card cannot be asked the driver's reading stands alone.
- `crit run serve --check`'s full-context request no longer fails at a fixed 600 s: it waits while the engine keeps
  finishing steps and fails only when none finishes for 600 s, and crit serve bounds it by the check attempt's
  deadline, as for Eval. An engine whose prefill of the run's whole context takes longer than ten minutes (a large
  context on a small device or a slow link) failed `full_context_prompt_step` and `full_context_device_memory` with
  `TimeoutError` before any step was judged, and the cancel then retired its host (Dualshok3, #77).

## [0.20.0] - 2026-11-18

### Overview

crit 0.20.0 serves a mixture of experts larger than the device and rebuilds Core around the device's memory and the
host. expert-offload is one plugin in the experts slot for such a model: recent residency in an elastic device cache,
misses computed on an AVX-512 CPU tier or copied over the link (`misses: auto` measures both at load), a host tier
between the store file and the cache, a pinned core held on the device alone, and `crit plugins offload` derives its
options from config.json, a probe file and the context. On Qwen3.8-Flash-Next on an RTX 5090 fresh prompts decode at
73 tok/s sampled and 78 greedy, where the run engine it replaced gave 24.8 and 32.5. Core's device memory is one
object (`crit_plugins.memory`) with elastic and paged ranges, step frames, a givers' ledger that gives way to other
programs, one scratch buffer and one graph pool an engine, so KV and recurrent pools are backed only where sequences
hold them and KV pages may lie in host memory; the host is another (`crit_plugins.host`: one worker pool, watched
handoffs, reads at a queue depth past the page cache), and native builds are a service. Core has a prompt cache: a
ledger that keeps ended conversations in a model plugin's pools, hybrid models included, with marks that thin out and
places conversations share. Exactness is stated end to end: kernels state how exact their rows are (replacing
`row_invariant`), an engine states the level it asks of its decode and prompt rows, a run's quality mode sets it, and
Core, crit smoke and Eval hold the engine to it; `crit run ... ablate` measures what a level costs. Measured load
decisions are recorded in the package and replayed, and `crit plugins status` and `crit plugins memory` say how an
engine's plugin copies and device memory stand. New kernel paths: W4A4 prompt rows on the FP4 tensor cores with the
decode tile's bits, sparse-indexer for a sparse attention's key selection, a fused latent-attention launch and EXL3
prompt GEMMs. Core's C++ that no engine linked is gone, with the scheduler features and the second model interface no
step engine reaches (Core's C++ falls from about 15,000 lines to 6,000); `StepDecision.discard` went with them and
rounds give way to other programs, so the engine ABI is 7 and `crit_core_steps` must be rebuilt. The default VRAM
headroom is 0.25 GiB. crit 0.20.0 needs crit-eval 0.7.1.

### Added

- Experts in one place: expert-offload's `pinned` core. In the ram tier host memory held every expert and the
  device held the resident ones a second time. With `pinned` (a share of the cache's slots at `open()`, off by
  default) the experts the `profile` file ranks hottest are read once from the store file straight into slots
  that no plan lists as a victim, no shrink gives back and no preload or prompt ring takes
  (device-expert-cache's `ExpertCache.pin`), and the store leaves them out of host memory: `BundleStore.leave`
  maps a small shared filler over their pages before the one registration (`host_memory.leave_out`), so the
  store stays one range with one device address and each of those experts is held once. Layers kept whole on
  the device (W4A16 beside W4A4) leave host memory the same way. An exchange instead of a fixed core was
  measured and set aside: taking a bundle's pages out of the registered store and back costs 470 to 540 us and
  240 to 280 us an expert beside the 166 us copy, tens of milliseconds a round. The core comes from `profile`
  alone (routing recorded on general text), never from `profile_save`, and needs `misses: cpu`, the recent
  policy and a store file. `crit plugins offload --trace T --profile P` replays a trace with cores of 25, 50 and
  75% of the slots against none (`trace_replay.recent(pinned=)`). On the Qwen3.8-Flash-Next test engine
  `pinned: 0.5` holds 3,991 experts (10.3 GiB) on the device alone: 89.3% hits where the plain rule has 90.7%,
  greedy decode 68.3 tok/s against 68.8, every reply's tokens unchanged. On that PC (an 84 GB Linux VM that
  leaves Windows 10 GB) Windows kept 10.6 GiB available and paged nothing out where it had fallen to 0.1 GiB and
  313,000 pages a second during the load, and the first reply after a cold load ran at 64 tok/s against 22.
  Getting there took two things beside the core: the left-out bundles are read past the OS's page cache
  (Core's host reads), and the kept ones are read in before the registration with the mapping's read-ahead off
  (`host_memory.read_in`), because a fault's read-ahead pulls the left-out pages into the cache in blocks the OS
  does not drop again.
- One pool for an engine's captured graphs, and steps that count their allocations. An engine's load has a
  memory pool of its own for graphs (`context.graphs`, `crit_plugins.memory`'s `pool("core", "graphs")`), and
  Core's round driver and prompt graphs capture into it by default wherever their factory's `parts` carry the
  load (`share_pool=False` gives a driver's graphs pools of their own, as before; the rule a shared pool asks of
  the plugins is unchanged). An engine's graphs never run at once, so a capture reuses the bytes of the temporaries captured
  before it and the pool holds far less than their sum: on the Qwen3.8-Flash-Next test engine the four rounds'
  pools of 494 MiB are one of 270 MiB with the same greedy replies (the largest round's graph alone takes 174;
  the rounds are captured smallest first, and a larger round's temporaries do not fit the blocks of a smaller
  one), and the expert cache takes what that frees. `made()` counts the
  allocations torch's allocator has made on a device; a probed engine (`CRIT_CORE_STEP_PROBE=1`) reports them
  for each kind of step (`core.probe`: `allocations`, `most_allocations`), and with
  `CRIT_CORE_STEP_ALLOCATIONS=0` Core ends an engine at the first step that allocates on the device once its
  load is over, which holds an engine whose every step replays graphs to "nothing allocates while serving".
  The test engine is not such an engine yet: expert-offload stages a prompt chunk's experts from the host, which
  no graph captures, so its prompt steps run eagerly (4,100 to 6,400 allocations a step, 287 MiB at their peak)
  and the audit reports them as such; its rounds allocate nothing.
- State is read where it lies: sparse-indexer scores keys in a pool's store. `scores(..., slots=)` and
  `select(..., slots=)` take the store's rows and each key's slot (`[n]`, or `[groups][n]` for a round whose rows
  are sequences) and load every block of keys through the slots, so a caller no longer gathers a sequence's keys
  for the kernel to read: the same products in the same order, the gathered keys' bits, for BF16, FP16 and E4M3
  keys. A gathered copy is a buffer the size of the state, written every call, and inside a captured round it
  stays in the round's pool for good; docs/PLUGINS.md states the rule for `kv:keys` and for pool state in
  general. On a copy of the Qwen3.8-Flash-Next test engine whose model plugin scores its rounds' block keys this
  way (it gathered every sequence's keys: 146 of the 174 MiB of its largest round's graph), the captured graphs'
  pool went from 0.28 to 0.06 GiB, the expert cache from 22.43 to 22.64 GiB and greedy decode from 68.8 to
  72.6 tok/s (a round 2 ms shorter), with all ten probe replies unchanged.
- KV pages the device has no memory for can lie in host memory (attention-state `host_pages_mib`, with `elastic`).
  A pool that finds the device full and nothing left to close backs its next pages from host memory instead of
  refusing the request: `crit_plugins.memory`'s elastic ranges back granules from the host (`back(first, end,
  host=True)`, a paged range's `host_bytes`), which the driver pins and maps into the range, so every reader, writer
  and captured round uses the addresses it always used and each row has the bits it has on the device. A context
  the device does not hold is served at the link's rate where it was refused. On an RTX 5090 (PCIe 5.0) kernels
  read such pages at 28 GB/s where pages on the device go at 690; the mapping works under the Linux (WSL) and
  the Windows driver. The device's report lists a range's host bytes (`host`). Off by default.
- Measured decisions (`crit_plugins.decisions`, `crit plugins decisions PACKAGE --model MODEL`). A plugin that
  chooses between two ways by timing both at load can choose the other one in the next process, and where the two
  do not give the same bits the same request then gets another text, which no repeat check inside one process
  sees. Such a choice is now settled through Core: `decisions.settle(plugin, name, key=..., measure=...,
  decide=...)` replays the package's record where it holds the decision with the same key and measures otherwise.
  The command loads the engine once and writes the record into the package; the configuration names it
  (`"decisions": "decisions.json"`), as it names the kernel tuning, and Core's loader applies it before any plugin
  is made. The engine's telemetry carries every decision (`decisions`: key, choice, `recorded` or `measured`, the
  measurement) with one digest over the choices. expert-offload's `misses: auto` (the CPU tier or the link for a
  missed expert, which changes the device rows' arithmetic) is the first decision settled this way
  (`miss_times_from`).
- Prompt cache: `spread` and `shared_least`. A model marks every chunk's end and the cache decides which marks
  stay. With `spread` (a token count) they thin out with their distance from the newest, about twice as far apart
  with each step back, so a long prompt changed in its middle resumes near the change where the newest marks
  alone resumed nowhere. With `shared_least` the cache learns where conversations part: a prompt that shares that
  many tokens more with a kept sequence than the place it can resume from (a system prompt and tool list every
  conversation of a client repeats) gets that place as its first stop (`stop(sequence, frontier)`), and the mark
  made there is a shared one, kept outside the count of `marks` and held by every sequence that resumes from it.
  Both are off by default.
- Core's host reads a file at a queue depth (`crit_plugins.host` `reads(owner, path)`): positioned reads from 8
  threads at once, each with its own handle, opened past the OS's page cache where the file system allows
  (`O_DIRECT`, `FILE_FLAG_NO_BUFFERING`), for a tier that keeps bytes of a file larger than host memory itself.
  The host's report lists each open file's reads.
- expert-offload's disk tier holds each expert once (`host_tier_mib`, by default the host memory free beside
  `ram_headroom_mib`). A host tier between the store file and the device cache takes the experts the cache evicts
  (the cache copies them back down the link and hands them over: a store that sets `retains`), and an expert that
  goes up into a cache slot leaves it, so the device's slots and the host's add up instead of holding the same
  experts twice. The file is read through Core's reads, a layer's misses all at once. `telemetry()`'s `file_store`
  counts the host tier's hits, the file's reads and the experts handed back. The disk tier also loads on Windows
  now (its store was written with a call Windows does not have). Not measured on a model larger than host memory.
- Core's device memory gives way to other programs (`crit_plugins.memory` `relieve()`, `watch()`). When a desktop
  or another process takes device memory while an engine serves, the givers (an elastic expert cache) release
  what they keep free and no longer have before the engine's next step, and wait five seconds before they grow
  back when what others hold has grown. Under WDDM the driver would otherwise move allocations to system memory
  without an error. Rounds Core launches itself run no step of the engine, so a watch thread reads the free memory
  five times a second and asks for one: `NativeRounds.ask_step()` sets a word Core's runtime reads before each
  round (`StepRounds::boundary`), and the next decode step runs through `step()` once. The engine ABI is 7 for
  it: an engine's `crit_core_steps` is rebuilt (`crit core build`); model plugins do not change. The device's
  report counts the releases (`outside`).
- A request whose rounds saw logits that were not finite fails instead of returning tokens that are not the
  model's: a model that names `nonfinite` among its round counters has the round driver count, row by row, the
  verified columns whose logits held a NaN or an infinity, and the engine's telemetry counts such requests
  (`core.nonfinite_requests`).
- A host job that does not return no longer hangs the device. A stream's wait on a handoff has no clock, so the
  handoff service has a second thread that only watches: a job that runs past the job budget (20 s;
  `crit_plugins.host` `job_budget(ms)`) fails its handoff, its waits are answered at once, `failures()` names it,
  and Core ends the engine at its next step (its watch asks for one between rounds) instead of serving on what
  such a window computed.
- exl3-rows: `gemm`, a prompt's rows with their decode bits (structure `prompt`, exact, the bits of `gemv`). The
  rows go through the tiles kernel in slabs of up to 64, the tiles decoded once a slab and multiplied by every
  group of eight rows, so the codes are read at their stored bits once a slab and no dense copy is made.
- expert-offload: `keep_routes`, an approximation that is off by default. N keeps each token's first N routes of
  the router's order and leaves the rest out, in every tier and for decode rows and prompt chunks alike, so a
  row's result never depends on what the cache holds and replies stay repeatable. On Qwen3.8-Flash-Next's recorded
  routing (10 routes a token, 8,408 cache slots, replayed) 9 routes miss 119 experts a round where 10 miss 143,
  and 8 miss 98. On the model itself (RTX 5090, six greedy replies of 256 tokens) greedy decode goes from 59.6
  tok/s to 68.9 with 9 routes and 72.8 with 8, the most likely token stays the same at 95.2% and 95.1% of 1,536
  positions (mean total variation 0.039 and 0.045 over the top 20 tokens), and every greedy reply leaves the
  all-routes reply within its first 80 tokens: a changed model, which the run's drift budget judges in quality
  mode `speed`.
- Core's layering is a test ([test_core_layering.py](tests/test_core_layering.py)): Core's Python imports nothing
  above it, its C++ includes no header outside it, and it names no library plugin and no model family beyond the
  names counted in an allowlist that may only shrink.
- sequential-recurrence: a test that a prompt chunk right after a round that rejected drafts starts from the
  state of the kept tokens, by every commit strategy.
- A run started from a crit-hub engine keeps the exactness the engine was measured at. A card's result may state
  `exactness`, the level the engine asked of its rows (one level, or one each for `decode` and `prompt`), beside its
  `quality_mode`; `crit run create --from-engine` checks it, records it with the result in
  `origin_engine.evidence` as `{decode, prompt}`, and the agent request names it with each published result and
  says where the engine was measured below the level the run's quality mode gives, so its numbers are read as what
  they are. A card that states none is read as before.
- The library's other plugins state the exactness they keep, and those with more than one way to compute a
  phase's rows follow the level the engine asks. Twelve plugins' objects gain `exactness()`: softmax-attention,
  partitioned-attention, attention-state, gguf-projections, projections-dense, fused-norms, host-expert-arithmetic,
  grouped-experts, expert-offload, heterogeneous-expert-dispatch, sequential-recurrence and
  recurrent-state-recovery report what their loaded rows keep, so the engine's `exactness` fact covers the whole
  engine and Core's refusal names the plugin that keeps less than is asked. Eight plugins' options gain `auto` and
  default to it: the attention plugins' `prompt_reader` (`decode` where the prompt rows are asked `exact`), gguf-projections'
  `prompt` (`exact` from `row-exact` up), host-expert-arithmetic's `prompt_gemm` (`declared` from `row-exact` up),
  expert-offload's `prompt_arithmetic` (`tensor` at `repeatable` and `row-exact`), heterogeneous-expert-dispatch's
  `prompt_arithmetic` (`tensor` at `repeatable`), and a new `prompt_rows` of both recurrences (`scan` from
  `row-exact` up). With nothing asked each resolves to its former default, so an engine that states no `exactness`
  runs as before. A candidate of a `speed` run now gets expert-offload's and the dispatch's tensor-core prompt
  chunks without setting the option; a candidate of a `preserve` run gets the exact prompt paths, and where a
  plugin has none (projections-dense, BF16 matrices, quantized KV pools with fresh chunk keys) Core refuses it
  until the configuration asks the level it keeps. A plugin whose prompt rows run its decode rows' arithmetic in
  an order that never follows the row count (fused-norms on its kernel, grouped-experts' tensor arithmetic,
  host-expert-arithmetic's 16-bit matrices) states its prompt rows `exact`, so a `preserve` candidate loads over
  it. docs/PLUGINS.md has the table.
- crit smoke judges an engine by the level it asks. An engine whose facts state the exactness it asks of its rows
  is held to it where the level promises bits, whatever the profile: prompt rows asked `row-exact` or higher must not
  change with the prompt's length, so `causal` and `causal_long` are judged with the strict tolerances, and prompt
  rows asked `exact` or `declared` (beside decode rows asked `row-exact` or higher) are the bits of the same token
  decoded, so `decode_matches_prefill` and `decode_long` are too. A lower level keeps the profile's tolerances, the
  drift a path may round by, and a failed item names the level that made it strict. With a run's quality mode giving
  the level, `crit run serve --check` and `ablate` now check a `preserve` candidate's rows strictly and a `speed`
  candidate's as before; a candidate that cannot keep `exact` says so by asking a lower level.
- `crit run ... ablate` measures what a level costs or gains. Among an option variant's settings
  (`--set`), `exactness=LEVEL`, `exactness.prompt=LEVEL` and `exactness.decode=LEVEL` are the level the engine asks
  of its rows, laid over the candidate's own: `ablate --set exactness.prompt=repeatable` serves the candidate as
  built and again asking repeatable prompt rows, in the same host on the same cases, and the report's verdict says
  whether the level pays, slows or breaks. Each variant's row carries the engine's `exactness` fact (what was asked
  and what each plugin kept), so a variant whose plugins could not follow shows it, and one that keeps less than it
  asks fails with Core's refusal as its detail.
- A run's quality mode sets the exactness of the engines it builds. `crit run ... candidate` completes a plugin
  package's `configuration.exactness` to a level for the decode rows and one for the prompt rows: the levels the
  configuration states, and for a phase it leaves out the run's quality mode's, `exact` in `preserve` (decode rows
  `row-exact`, each prompt row the bits of the same token decoded) and `repeatable` in `speed` (every plugin's
  fastest way, as before). The package carries the completed configuration, so Core holds the engine to the level
  it was measured at wherever it is served, and a supplied configuration.json that lacks it is refused with the keys
  that differ named. The run's brief states the levels its mode gives and how to ask another, and AGENTS.md treats
  a phase's level as a lever: changed with its speed and its drift measured. An external engine's configuration is
  left as it is. In a `preserve` run this changes what a new candidate runs: a plugin that follows the level takes
  its exact path unless the configuration asks a lower one (quantized-projections' `prompt_gemm: auto` becomes the
  prompt tile), and an engine whose options keep less than `exact` must say so in `exactness`. Candidates recorded
  before this ask nothing and are served as they were.
- quantized-projections follows the level the engine asks. Its `prompt_gemm: auto` (the default) now takes the
  fastest prompt GEMM that keeps the level the configuration's `exactness` asks of the prompt rows: `tile` where it
  asks `exact` (each prompt row the bits of the row decoded, with `gemv: auto` as `fp4` so every decode row runs
  the tile), `pinned` where it asks `row-exact`, and crit's native GEMMs, as before, where it asks `repeatable` or
  nothing. A `prompt_gemm` set to one way is that way. The plugin's `exactness()` states what the loaded matrices
  keep: decode rows row-exact; prompt rows exact on the prompt tile, row-exact through `fixed_tiles`, `pinned` and
  `pinned_cublaslt`, repeatable through `native` and `provider`. So an engine that asks `exact` and sets
  `prompt_gemm: native` is refused at load, and one that asks nothing runs as it did. Where auto's choice cannot run
  with another option (a `gemv` the prompt tile does not take, a prompt arithmetic the pinned GEMMs do not run), the
  error names the level that chose it.
- An engine states the exactness it wants, and Core holds it to it. A configuration.json may now carry
  `"exactness"`: one level of the kernels' scale for both phases, or one each
  (`{"decode": "row-exact", "prompt": "repeatable"}`; `decode` is the decode and verify rows, `prompt` the prompt
  rows). Core's loader reads it before any plugin is made (`crit_plugins.kernels.use_exactness`), so a plugin that
  can compute a phase's rows in more than one way asks what the phase needs (`asks(phase)`, `allows(phase, level)`)
  where its own option leaves the choice open, and each plugin object may state what its rows keep as loaded
  (`exactness()`). The engine's facts carry both under `exactness`: what the configuration asked, the lowest level
  any plugin keeps in each phase, and each plugin's own; crit serve's facts and Eval's engine facts pass it on. An
  engine whose plugin keeps less than its configuration asks is refused at load, naming the plugin and the phase,
  and a package that asks for a level that does not exist is refused when it is checked. A configuration that asks
  nothing is served exactly as before, by its plugins' options. quantized-projections is the first
  plugin to follow it, and a run's quality mode completes it (the entries above; docs/PLUGINS.md#exactness).
- Kernel structures state how exact they are. The fastest way to compute a row is often not the most exact one,
  and until now a kernel said so with one flag (`row_invariant`), a free-text `order` and the names of its
  structures, which every plugin read its own way. Each structure in a kernel.json now declares its `exactness`,
  one of four levels that each keep what the ones below keep: `declared` (the kernel's declared order: the bits
  the CPU tier's rows, the decode rows and a prompt GEMM of that order all give), `exact` (the bits of another
  structure, named by `bits_of`: a prompt row with the bits of the same token decoded), `row-exact` (an order of
  its own fixed by the shape and the tuning, so a row's bits never follow the rows beside it, the row count or a
  chunk's length) and `repeatable` (the same call gives the same bits, which may follow the row count, the
  chunk's length or the tile). Of the 80 structures 12 are declared, 11 exact, 44 row-exact and 13 repeatable.
  `crit_kernels.exactness` reads them: `level()` and `keeps()` for one structure, `same_bits()` and `choices()`
  for the structures of a kernel a phase may run at a level given the structure its decode rows run, and
  `setting()` for an engine's choice per phase (`decode`, `prompt`), which takes what the engine leaves out from
  the run's quality mode (`preserve` asks exact, `speed` repeatable). `crit kernels list` shows each structure's
  level and, for an exact one, whose bits it has; a kernel.json with a level that does not exist, or an exact
  structure that names no other, is refused when it is read. This is the scale only: no kernel computes anything
  differently and no plugin chooses by it yet (kernels/README.md#exactness). A kernel.json is part of its
  kernel's build, so every kernel builds once more on its next use; tunings apply as before.

- Scratch is a Core service: `context.scratch` (`crit_plugins.memory.Scratch`). What a call only computes in and
  nothing reads after it returns (a kernel's partial sums, a launch's sorted rows) was a fixed workspace of each
  plugin, allocated at load at its largest size, so an engine paid their sum. An engine's calls run one after
  another on its stream, so they can all compute in the same bytes: the engine has one growing buffer at a fixed
  address (a captured graph keeps it), each call takes what it needs from its start every time it runs
  (`take(nbytes)`, or `take(a, b, ...)` for a call's buffers together) and holds nothing there across calls. It
  costs the largest call's bytes, grows outside captures only, belongs to one engine (two engines of a process
  never share bytes) and shows in the census as `core/scratch`. That nobody relies on what an earlier call left
  is now something a test proves: with `CRIT_SCRATCH_POISON=1`, which `crit plugins test` sets, every take is
  filled with 0xFF bytes first (NaN as a float, -1 as an index), inside captured graphs at every replay too. The
  split-KV reader's partials (attention-state's device form, behind softmax-attention and partitioned-attention:
  `workspace_mib` is now the most a launch may take, 128 MiB held from the load until now), token-sampling's
  launch scratch (16 MiB on a 248,320-token vocabulary), the route rows of the experts' device part
  (heterogeneous-expert-dispatch's `KernelRoutes(scratch=)`, which expert-offload uses: eight growing buffers of
  its own until now, 174 MiB after a 1,024-token chunk) and quantized-projections' tile activations and partial
  sums (68 MiB, `device_matrices.use_scratch()`) are taken from it; the attention plugins and the sampler declare
  and allocate no workspace of their own any more. On the Qwen3.8-Flash-Next test engine one scratch of 164 MiB
  (the route rows of a 1,024-token chunk) serves what took 322 MiB in those workspaces, the expert cache gained
  0.17 GiB, and the greedy replies are the same, also in a run with every take poisoned.
- `crit plugins memory PACKAGE --model MODEL`: the memory audit. The census says who holds each byte of a
  device; the audit says what the engine does with it, with no word from the engine's plugins. It loads the
  package in process, serves two short prompt sets of its own (a long prompt and three chats each, none of
  Eval's text) and fingerprints every page of the device memory the process holds before and after
  (`crit_plugins.audit`: torch's blocks in 512-byte pages, every granule the elastic ranges back on the device
  in 4 KiB pages, a range's granules in host memory counted as `host` and never read; a reading writes nothing and allocates nothing but its own 32 MiB, and the engine's own thread takes it
  as a request ends, before the request's sequence closes). Each buffer is reported by the plugin
  and attribute that hold it with the bytes the workload wrote, how far into it the writes reach and what stayed
  zero. Two verdicts are certain, dead (all zero at every reading) and oversized (a part never written), and the
  report keeps them apart from what it cannot know (unchanged bytes that hold something: weights, a table, or a
  workspace only the load wrote). The bytes Core knows to be scratch (the temporaries of captured graphs, growing
  buffers) are filled with a mark first, so every unwritten byte there is certain, and a greedy reply repeated
  after the workload must be the same (`--no-mark` leaves them alone). Each captured graph's pool is listed with
  the allocations its captures made by the code that made them, each paged range with the units backed and
  written, each kind of step with its graph replays, allocations and transient peak; `--out` writes it all as
  JSON. For this, Core's device memory names things it only summed before: `holdings(plugins)` gives every
  reachable tensor with its plugin and where it is held (the census walks it, and now starts at the plugins'
  module globals too, so a module's shared buffers count for their plugin instead of "held by no plugin
  object"), `pool(owner, name)` and `capturing(pool)` name the captured graphs' pools (Core's round driver and
  prompt graphs use them; the census lists `pools` by name), and `record()` turns on torch's allocation history
  for a tool, and Core's step engine has an `audit` a tool sets to see every step and every request's end on the
  engine's thread. On the Qwen3.8-Flash-Next test engine the audit finds by itself what a day of hand-written
  probes found: the split-KV workspace, of which 1,024-token chunks write 43 of 128 MiB; a recurrence pool not
  yet paged, 215 of 864 MiB of states and an eighth of each stash written (the units one request uses); the
  projections' 64 MiB buffer with 3 MiB written; the sampler's scratch (0.2 of 16 MiB); and the rounds' pools,
  492 MiB, filled by the model plugin's gathers of whole sequences' indexer keys (one request at a time replays
  the one-row round's 48 MiB and never the 444 MiB of the two-, three- and four-row rounds). It also shows every
  prompt step running eagerly with 4,100 to 6,400 allocations and up to 287 MiB of transient peak, the native
  rounds allocating nothing, and a lookup table's output buffer made anew and kept for every prompt length met.
  The greedy replies of an audited run are those of a run without it.
- What holds a device beside an expert cache is named tensor by tensor. Core's census of the device
  (`crit_plugins.memory`, the engine's `"memory"` telemetry) lists the largest tensors one by one (`tensors`:
  plugin, bytes, shape, dtype), expert-offload's `device` telemetry and its diagnosis carry the three largest, and
  the load's telemetry (`"load"`) gives at each stage (`made`, `opened`, the new `warm`, `checked`) `other`: the
  device memory outside torch and the elastic ranges that is not free, so what the CUDA context, libraries and
  captured graphs take shows per stage. `crit plugins offload` says in a note what its `tables.` placement means
  for the model plugin: every table a pass gathers a few rows of, the token embeddings among them, is planned as a
  `tables.` group and opened through the embeddings slot. On Qwen3.8-Flash-Next the test engine's model plugin
  held its token table (1.2 GiB, `[248320][2560]` BF16) on the device while the n-gram tables were read from the
  host; opened from the host it gave the expert cache 1.1 GiB (19.8 to 20.9 GiB), the replies bit for bit the
  same and a round's lookup at a device table's cost.
- The host is one Core object, `crit_plugins.host`, as a device's memory is: the plan of who runs where (the first
  physical core answers the device's doorbells, each other core has one worker), the process's one worker pool,
  and handoffs, the work a device's stream posts to the host and joins inside a captured graph. Every plugin gets
  it as `context.host` and the engine's telemetry carries it as `"host"` (cores, host memory, the pool's workers
  and batches, each handoff's join and jobs). Until now host-expert-tier brought its own pool and
  heterogeneous-expert-dispatch built the mailbox into its own extension with a service thread that ran unpinned
  beside the pinned workers; a second plugin with CPU work would have started a second pool on the same cores.
  host-expert-tier's rows now run on Core's pool (given as two addresses, `PoolRef` in Core's `pool.h`),
  heterogeneous-expert-dispatch and expert-offload post through Core's handoff (their `cpu_routes.cu` keeps only
  its post and scatter kernels), and one service thread a process, on the plan's first core, answers every handoff.
  The host-work-mailbox entry describes the technique and Core runs it; the pool and service options `spin_us`,
  `spin_max_us`, `idle_after_us` and `idle_sleep_us` of the two plugins are gone (Core's pool and service keep the
  values they had), and `workers` and `pin` size Core's pool when a plugin is the first to ask for it.
- Native code built at load is a Core service, `crit_plugins.native`: `library()` (C++ by the system's compiler,
  opened with ctypes) and `extension()` (a torch extension for the device's architecture), cached by content, a
  failed build remembered for the process, and every build recorded (`built()`, in the host's telemetry as
  `"native"`). Six plugins carried a build of their own (host-expert-arithmetic's library; the torch extensions
  of heterogeneous-expert-dispatch, device-expert-cache, lookup-table-rows, recorded-selection and token-sampling,
  each with its own cache name and architecture handling): all of them build through Core now.
- A plan gives the goals Eval states outside the priorities a fate: a gap on `followup_ttft` (the follow-up turns'
  first token against its floor with a prompt cache) or `concurrency` (the most clients' aggregate against its
  batched ceiling) is met by a lever whose `changes` improve that KPI or answered in `gaps`, in every plan after
  the Eval that states it. Neither is ranked or forecast, and no candidate is built for one alone. Until now
  plans read five KPIs and dropped every other gap, so no build worked on a prompt cache or on batching.
- `crit plugins status ENGINE/plugins` says how an engine's copies of library plugins stand to the library:
  current, behind (a copy of an earlier library version the engine did not change, with that version's commit
  and the library's changes since), changed by the engine, or the engine's own. An engine holds a copy of every
  library plugin it uses, and until now nothing said which copies had fallen behind the library and which the
  engine had changed: on the two Qwen3.8-27B hub engines 8 and 9 of 17 copies are behind, and none is changed.
  `crit plugins update ENGINE/plugins` replaces the copies that are behind with the library's current ones and
  leaves the ones the engine changed.
- Core has a prompt cache, `crit_plugins.prompt_cache.PromptCache`: a ledger that keeps ended requests' sequences
  in a model plugin's pools, so a prompt that begins as one of them did (a conversation's next turn, an agent's next
  tool step) prefills only the rest. It works through the pools' own `open`, `fork`, `keep` and `close`: KV pools are
  forked and kept to the shared prefix; pools of one state per sequence (a recurrence's, a sliding ring) resume where
  they stood, at a kept sequence's end or a position the model marked. The model plugin calls it in `open()` and
  `retire()` and reports the prefix as `StepResult.reused_prompt_tokens`. Until now no plugin engine kept KV between
  requests: a follow-up turn of 48 new tokens on 8,192 prefilled all of them (1.1 s and 3.5 s on two Qwen3.8-27B
  engines on an RTX 5090, against a floor of 14 ms). `toy-model`'s `prompt_cache` option keeps one, and the device
  tests' Llama-layout model plugin keeps one over attention-state's paged pool, through `step()`, captured rounds
  and drafting rounds. attention-state's pool asks `on_exhausted(callback)` for room before it refuses a page, so
  kept sequences give their pages back to a request that needs them.
- The prompt cache of a model with recurrent state also serves prompts that do not repeat a kept prompt to its
  last token: the same prompt sent again, and a turn whose template ends a prompt otherwise than it renders it
  later (Qwen's thinking tag: the next turn repeats all but the last token). With `tail` the cache marks a place
  that many tokens before a prompt's end beside the end itself, where the model's prompt chunk ends
  (`step_item(..., stop=cache.stop(handle))`; Core takes the shorter chunk from the prompt tokens the step
  reports). The shorter chunk costs a step, so a prompt that resumed at a kept prompt's end has none, nor one
  shorter than `tail_least`. A sequence that resumed at a mark holds it too, without a second state copy, so a
  turn sent again, with the same or another last message, resumes where the turn before did. Marks take no
  state of their own: they may stand in the states no open request uses, and `evict()` and `unmark()` take them
  back. Until now such prompts prefilled everything.
- A request says whether it may be served from the prompt cache: `StepRequest.prompt_cache`, which crit serve sets
  unless the request carries `"crit": {"prompt_cache": false}` (`chat-prompt-cache-v1`, declared for every
  plugin package). crit smoke, `crit run measure` and the ablation's benches send it, as crit-eval's timed and
  scored requests do, so a measurement prefills every prompt token and a repetition is no cache hit; a request
  that does not allow the cache runs as on an engine without one. Core's eval-v1 adapter, scores and replays never
  read it.
- expert-offload: one plugin in the experts slot for a mixture of experts larger than the device. The model plugin
  calls it as for experts that fit (begin, load, open after its pools, forward per routed layer); its decision tree
  places the routed experts on the device, in host memory behind a device expert cache, or in a store file read on
  the host, computes misses on the CPU tier in the device's bits while the link copies them into the cache beside
  the round (else copies every miss, with the device's rows on tensor cores), stages prompt chunks through the
  cache's ring, and reports what it decided and why, with a one-line diagnosis in its telemetry. recon's starting
  designs and AGENTS.md start candidate 1 on it.
- Recent residency for expert caches (device-expert-cache's `policy: recent`, expert-offload's default beside the CPU
  tier): what is resident is what was routed last. A missed expert is computed by the CPU tier, copied in beside the
  round and served from the next round, in the slot of the expert routed longest ago. The ranked policy ported
  before it ranked decayed route counts every few rounds and lagged the routing: on Qwen3.8-Flash-Next's experts
  alone on an RTX 5090 it hit 85% at 40.5 ms a round against 94% at 16.4 ms, and an engine on the plugin decodes
  45.8 tok/s sampled and 59.5 greedy against 36.6 and 41.5 (24.8 and 32.5 for the run engine it replaced).
- The device's memory is one object, `crit_plugins.memory.of(device)` (a plugin's `context.memory`), for Core and
  every plugin: its readings (free, torch's allocator, the blocks it holds unused), elastic ranges on the driver's
  virtual memory API, a givers' ledger, and a census. Core tells it when an engine's load starts and when it is
  over (the givers then take what the load left) and counts, once, which plugin's objects hold which of torch's
  bytes; the engine's telemetry carries the whole device (`"memory"`: free, torch, each elastic range, each
  plugin, the rest). Until now those readings lived in three modules, and nothing said who held a device's bytes:
  on Qwen3.8-Flash-Next it is 6.0 GiB of projection weights, 1.2 of an embedding table and 0.9 of recurrent states
  beside the experts. The census also names what captured graphs' own pools hold (`"graphs"`): torch keeps a pool
  for each graph's replays, in use or not, and on that engine the rounds' and prompt chunks' pools are 0.50 GiB of
  which 0.03 are allocated (the round driver's and the prompt graphs' `share_pool` is the lever).
- Pools take device memory as their sequences come: `crit_plugins.memory`'s paged ranges (`paged()`, `back(units)`,
  `trim(units)`) back the units of a pool's layer tensors together, after asking the device's givers.
  attention-state's elastic KV pools run on them, and sequential-recurrence and recurrent-state-recovery have an
  `elastic` option: a sequence's recurrent state is backed when it opens and goes back when the sequences past it
  closed. On Qwen3.8-Flash-Next a state pool of 8 sequences takes 0.14 GiB instead of 1.0 with one request in
  flight, which the expert cache holds instead (about 300 more slots, 6% fewer misses).
- Device memory that follows the requests: that memory's elastic ranges and ledger, and Core's step frames
  (`begin_step`, `end_step` on any slot's plugin). An
  expert cache holds every byte the requests in flight do not need: expert-offload's cache takes the free memory
  once the load is over, gives a prompt step the bytes a chunk of its size was measured to take and grows back.
  Until now a cache was sized once at load against a guessed margin, and under WDDM a guess too small spilled into
  system memory (rounds of 130 ms instead of 17). A giver says what it keeps free and `need()` leaves it free, so
  KV pages a step reserves never come out of that step's own room; expert-offload keeps `margin_mib` free at
  every moment (the run's VRAM headroom: `crit plugins offload` writes it), counts the blocks torch's allocator
  kept from earlier prompts as the next prompt's room (a prompt of many steps, or one chat turn after another,
  takes nothing from the cache), and a step without prompt tokens costs nothing. Core reads a probed step's free
  device memory when the model's step returns, before the end frames take the step's memory back.
- attention-state's `elastic` option: the KV pools' pages in elastic ranges, backed only where sequences hold them
  (4,096 tokens of every layer at a time), the device's givers asked for the bytes before a pool grows. A pool
  sized for a 200K context held its whole allocation whatever the requests; now what it does not hold is the expert
  cache's, with no lending between the two and no change to the model plugin: Qwen3.8-Flash-Next's engine keeps
  7,250 cache slots beside elastic pools against 6,050 beside whole ones, and decodes 53.5 tok/s sampled and 60.0
  greedy against 39.5 and 49.7 (45.8 and 59.5 with the pool lending its idle pages to a cache of fixed size).
- expert-offload's telemetry and diagnosis say who holds the device's memory (the cache, the model's weights,
  pools and workspaces, what is free) and what a GiB more for the cache is in slots: on Qwen3.8-Flash-Next each
  GiB is about 7% fewer misses, more than any cache setting gives.
- device-expert-cache's `trace_replay` replays a recorded routing trace through recent, ranked and static residency
  in about two seconds a setting, and expert-offload's `bench.py` times the plugin alone on an engine's store file
  and trace (a captured graph a round) in two minutes a setting: a cache setting is judged without an engine build.
- A plugin in a slot a captured round uses takes part in the round (`start_round`, `count_round`) without the model
  plugin passing it in `extra_parts`.
- The CPU expert tier's scalar parts run on AVX-512 (host-expert-arithmetic's E2M1 quantizer and SwiGLU 16 values at a
  time, a group's weight scales by one gather, a row's lanes reduced in registers), every bit the scalar order's:
  a missed expert of 2.78 MB takes 328 us on one thread against 440, and 115 us against 137 inside expert-offload's
  bench, where six threads now read host memory as fast as it gives random experts.
- `crit plugins offload` derives expert-offload's selection and options for a mixture of experts larger than the
  device from config.json, a `crit eval probe` device file and the context (the miss path, the cache's policy, the
  prompt arithmetic, the CPU tier's workers), and `--trace` replays a routing trace through the cache's residency
  at several sizes and copy budgets.
- device-expert-cache's `side_swaps`: a ranked boundary's swapped bundles copied on a side stream beside the round
  (joined at its end, mapped by the next round) instead of before its first layer.
- device-expert-cache's `record_routes` keeps the last decode routes on the device, inside captured rounds too, and
  the cache's `routes()` returns them as the routing trace `crit plugins offload --trace` replays.
- mma-rows: `mm_prompt` takes acts `nvfp4_mma` (structure `fp4-prompt-tile`, the sm_120 family): any number of W4A4
  rows on the block-scaled FP4 tensor cores, every output with the bits the FP4 MMA decode tile gives the same row
  (the same m16n8k64 MMAs in the same order, the plan's splits of K added in split order). Until now a W4A4 engine
  whose scored and served outputs must be the same bits ran its prompt rows on the BF16 MMA at 150 to 170 TFLOPS, and
  the FP4 MMA served decode rows only. The tile is a warp-specialized GEMM (a producer warp's TMA tile copies into a
  ring of stages, consumer warps reading fragments by ldmatrix, a split's chains joining running totals inside the
  MMA stream, whole tiles stored through shared memory). RTX 5090, the quantizer included, at 2,048 rows: a 27B's
  34816 x 5120 gate/up 886 us (824 TFLOPS) against 4,134 us on the BF16 path, its 5120 x 17408 down 458 us (797)
  against 2,243; 4.3 to 5.2 times the BF16 path from 512 rows and 0.78 to 0.86 of cuBLASLt's NVFP4 GEMM, whose rows
  have other bits. Default knobs, not swept.
- quantized-projections: `gemv: fp4` puts W4A4 decode rows on the FP4 MMA where the device has it and every other
  matrix on the tile, and with `prompt_gemm: tile` sends those matrices' prompt rows through the FP4 prompt tile:
  `prompt_exact` stays true with the prompt rows on FP4 tensor cores.

### Changed

- kernel.json's `row_invariant` is replaced by `exactness` (above): true became `row-exact` or a higher level,
  false `repeatable`, and a kernel.json written before still reads by its flag. `crit kernels sweep` runs its
  row check (row 0 beside other rows against row 0 alone) on the structures that are row-exact or higher, the
  ones it checked before. Three of recurrent-scans' structures whose README and tests already promised more
  than their flag said now state it: `delta-columns` and `ssd-columns` (the column scans) are row-exact and
  `conv-segments` has pool_conv's bits. They have no knobs, so no sweep builds them.

- expert-offload's `misses: auto` is measured, no longer the CPU tier wherever it builds. At load it times a missed
  expert on the CPU tier and over the link and takes the CPU tier only where it needs under four fifths of the
  link's time (`miss_times_us` in its telemetry, and the reason it gives). Beside the CPU tier every device row
  runs the declared integer order and the tier's first replies after a load or a pause are slow; over the link
  every expert runs on the tensor cores. On Qwen3.8-Flash-Next (RTX 5090, Ryzen 5 9600) both serve a warm round
  in the same time and the link wins the measurement: five fresh prompts decode at 73 sampled and 78 greedy tok/s
  where the CPU tier gave 45 and 73 (its first replies 24, 40 and 51). An engine that chose the CPU tier by `auto`
  gives other tokens after this change (the link's rows are the tensor cores'); `misses: cpu` pins it.
- MXFP8 activations for a decode call come from one kernel. quantized-projections multiplies 16-bit rows of the
  quantized values (`mxfp8_activations: mxfp8`, `decode_activations: prompt`), and made them with the quantizer
  and three torch kernels a call (the codes as FP32, times their scales, as BF16); fp8-block-rows gains
  `quantized_values`, which writes those bits in one launch. On Qwen3.8-Flash-Next a decode round has 264 such
  products: the captured round goes from 21.6 to 19.4 ms and a reply from 66.5 to 69.1 tok/s, the same tokens.
  For groups of 32 a warp takes a group (a load and a division a thread, the group's amax a warp reduction): 1.1 us
  a call where a thread looping over its group took 6.5, another 1.4 ms of the round's kernel time.
- A prompt step allocates no copy of its sequence's recurrent state. A prompt row that runs the scan keeps its
  final state of every layer until its accept (a cancelled step must leave the sequence as it was, for a prompt
  cache's next turn); the recurrence pools now keep it in the pool's lowest free unit, where a sequence's states
  live, backed for the step. On Qwen3.8-Flash-Next that copy was 108 of the 122 MiB a short prompt step took: a
  step now leaves 5 MiB allocated when it returns instead of 113.
- Two workspaces follow their use instead of standing for good, each through a service of Core's device memory.
  `crit_plugins.memory` gains growing buffers (`growing(owner, name, max_bytes)`, `take(nbytes)`): one buffer at a
  fixed address that holds what its largest user asked for. expert-offload's device part keeps its route rows
  there, so the workspaces of every prompt chunk size share them (grouped-experts' `Workspace(rows=...)`,
  heterogeneous-expert-dispatch's `KernelRoutes(memory=...)`); until now each chunk size a server met left a
  set of rows behind, 95 MiB for a 601-token chunk and 160 MiB for a 1,024-token one on Qwen3.8-Flash-Next. And
  the recurrence pools' `elastic` option backs everything a sequence keeps with it, the stash, the convolution
  window and the convolution's stash beside the states: they are stored with a sequence's layers together and
  handed to the kernels as `[layers][sequences]` views, which recurrent-scans' pool entry points take by their
  strides (an entry of one layer and sequence contiguous). On that engine torch's allocation falls by 0.48 GiB and
  the expert cache holds 22.0 GiB after a long prompt, the replies the same bit for bit.
- quantized-projections makes each of its shared buffers at its first use (a matrix's load runs every path it
  takes) instead of all five in the factory: a device whose matrices never decode a chunk of rows or pin a GEMM
  holds neither the 64 MiB decode workspace nor the 32 MiB GEMM workspace. `memory()` declares the same bytes, now
  the most the buffers take.
- The VRAM headroom's default is 0.25 GiB, no longer 1 GiB: the device memory an engine leaves free at its peak
  where neither the request nor `settings.toml`'s `vram_headroom_gib` says otherwise (crit Eval's
  `VRAM_HEADROOM_GIB`, read by `crit run create`, `crit run eval`, the served pre-check, `crit pull`, `crit bench`
  and `crit plugins offload`, whose `--vram-headroom-gib` now defaults to the machine's setting), and
  expert-offload's `margin_mib` defaults to 256. It is the user's knob: every GiB of it is a GiB the engine's
  weights, KV cache or expert cache do not get. On Qwen3.8-Flash-Next the expert cache holds 21.8 GiB at 0.25 GiB
  against 21.1 at 1 GiB (8,408 against 8,052 slots). A machine whose desktop takes device memory in bursts sets
  it higher: an elastic expert cache gives memory back at the next prompt step, not during a reply's decode
  rounds.
- heterogeneous-expert-dispatch's device part is crit's grouped-experts kernel in its declared arithmetic, reading the
  CPU tier's bundles in place (`device_part: auto`); the torch DeclaredRoutes, one route at a time, is only the fallback
  where the kernel does not build, and the facts say which serves. `prompt_arithmetic: tensor` runs a prompt chunk that
  sends no route to the CPU in the kernel's tensor arithmetic instead of the CPU's integer order, for speed engines
  whose prefill the declared order bounds.
- grouped-experts' decode items ask for their weights 128 bytes of a row at a time (4 blocks of a 4-bit format, 2 of
  FP8, loaded together) instead of a 64-k block at a time (32 or 64 bytes). A decode pass is a few warps a
  multiprocessor, each waiting on its own chain of weight requests, so what a request brings was the pass's rate:
  NVFP4 experts moved at 0.6 to 0.9 TB/s where BF16 experts and the dense decode kernels move 1.5. RTX 5090, a decode
  layer call (plan, gate/up, down, combine) of 512 NVFP4 experts (2560 x 640, top 10) at 1 / 2 / 4 / 8 tokens: 42.9 /
  74.3 / 131.7 / 243.2 us before, 30.4 / 48.9 / 84.9 / 152.3 now; W4A4 46.8 to 248.4 before, 33.9 to 159.6 now; MXFP4
  and INT4 the same; FP8 experts (4096 x 1536, top 8) 122.2 to 681.3 us per channel and 141.7 to 787.6 per block
  before, 109.1 to 589.7 and 106.7 to 581.0 now. Every route keeps its bits (the same operands in the same MMAs). The
  new knob READ of the `decode` structure picks the reads (1 the blocks, 2 the 128 bytes, 0 by the pass's K, the
  default); BF16 experts and prompt items are unchanged.
- sparse-indexer selects a decode call's top k in a CUDA kernel (`cuda_select.cu`, knob `CUDA` of the `topk`
  structure, on by default for calls of fewer tokens than half the multiprocessors): a slice's scores are copied to
  shared memory once and the radix select's five passes run there, where the Triton program read them from global
  memory in every pass. RTX 5090, k 512, one token: 26.5 us before and 14.5 now at 50,000 keys, 65.1 and 19.5 at
  200,000, 13.3 and 10.7 at 8,192. The picks are the same (no arithmetic in a select; the tests compare the two
  paths). Prompt chunks keep the Triton programs. A mixed kernel without CUDA variants names its library `all`.
- sparse-indexer sums a token's heads in an explicit tree (heads 2 i and 2 i + 1, then the pairs' sums) instead of
  `tl.sum` over the tile's head axis, whose tree followed the tile's register layout. TM, BN and WARPS of the `scores`
  structure therefore change no bit any more: they are speed knobs a sweep picks per token count, not order knobs
  held at one value a shape, and a call takes no more tokens a program than it has, so a one-token decode call no
  longer multiplies a tile padded with a second token. A token's scores differ from the earlier kernel's in their
  last bits (another order of the same FP32 terms); decode and prompt rows agree as before. RTX 5090, 64 heads of 128
  over 50,000 keys: one token 13.4 us before, 10.2 now; prompt chunks unchanged (9.7 ms at 2,048 tokens). With the
  CUDA select a decode token's whole selection step takes 24 us where it took 40.
- latent-attention's `attend` takes `fused`: one program a tile reads its partitions in turn and folds them as the
  merge does, so every token has the split launch's bits and no partials are stored (a 64-column window at 128 heads
  and rank 512 held 675 MB of them). By default a launch whose partials would pass 512 MB is fused. A prompt chunk
  read through it as windows of 64 columns has the bits of its tokens decoded one by one, which the kernel's prompt
  reader (`prefill`, another reduction) never gave: attention-state's `prompt_reader: decode` now covers latent
  attention too (a step's prompt rows and captured prompt chunks), at 1.1 to 2.1 times `prefill`'s time.

### Removed

- Core's C++ that no engine links. An engine runs `crit_core_steps` (the runtime, text, structured output, JSON and
  files) and its plugins; the libraries below date from engines written in C++ against Core's SDK, and only their
  own fixtures still built them.
  - `crit/batch.h` (`StepLayout`, `PageReservations`, `SpeculativeRow`, `peak_pages`). A model plugin's step model
    lays its rows out with `crit_core.batch`, and the batched-steps guide with the batch invariance rules moved
    beside it (`core/python/crit_core/batch.md`).
  - `crit/host_memory.h` (`observe_host_memory`, `admit_host`). The host-memory-admission entry's Python
    observations and admission rule are the ones engines use.
  - `crit/safetensors.h`, `crit/model_config.h` and `crit/sha256.h` (`Checkpoint`, `load_model_config`,
    `load_generation_config`, `stop_token_ids`, `Sha256`), with attention-state's `rope_table.h`, which read
    its config through them. `crit_core.model` reads model files and `rope_table.py` derives the rotary table.
  - The host sampler of `crit/sampling.h` (`Sampler`, `SamplingParams`, `resolve_sampling`, `TokenHistory`,
    `log_softmax`, `top_logprobs`, `score_logit_rows`, `accept_greedy`). Engines sample through a sampler
    plugin. The header keeps the draw (`philox4x32`, `uniform`, `DrawPurpose`), which token-sampling's kernel
    includes; the host implementation of the rules lives on as the oracle of token-sampling's CUDA fixtures
    (`tests/inference/host_sampler.h`), and Core's sampling guide keeps the draw and the rules.
  - `crit/state.h` (`StateCache`: KV page, round, prefix and recurrent slot bookkeeping) and the C++ of
    attention-state that stored pages for it (`state_pages.h`, `encoded_kv.h`, `host_page_tier.h`, the CPU
    attention of `attention.h` and `batch_attention.h`, the batch and recovery engines and their examples).
    An engine's KV is the attention-state plugin's paged pool on Core's memory. The operation
    `paged-attention` names the torch arithmetic the kernels are tested against as its reference
    (`device_attention.py`'s `windows_torch`), and recurrent-state-recovery's header carries its own
    `checkpoint_node`.
  - The device services no engine built: `crit/device_graph.h`, `crit/device_memory.h`, `crit/host_block.h` and
    the CUDA backend `crit::cuda` (`crit/cuda.h`, `cuda_graph.h`, `cuda_memory.h`, `cuda_launch.h`,
    `cuda_devices.h`), with the build options `CRIT_CORE_CUDA` and `CRIT_CORE_HIP`, the installed-SDK CUDA consumer,
    `crit_core.buffers` and the references built on them: device-speculative-round's C++ round controller and its
    toy, concurrent-request-execution's `device_metadata.h` and CUDA recurrence example, device-expert-cache's
    `expert_store.h` and the kernels' foreign-stream example. A device's memory, captured rounds and prompt
    chunks, the host's handoffs and native builds are Core's Python services over torch (`crit_plugins.memory`,
    `rounds`, `prompt_graphs`, `host`, `native`). What stays of the CUDA backend is the launcher of captured decode
    rounds, which loads the driver at run time, and `crit/gpu_runtime.h`, the CUDA-to-HIP name map native sources
    include. Inference's CUDA fixtures no longer need a Core CUDA build (`CRIT_INFERENCE_CUDA` needs only the
    toolkit).
- The plugin library's C++ and CUDA that no plugin builds at load, about 7,300 lines, with the fixtures that tested
  only them: the reference loops, controllers and kernels written for engines in C++ (mtp-decode's `speculative.h`,
  `sampled.h` and `prompt_copy.h`, session-scheduling's chunk selector and physical batches, dense-layer-offload's
  executor, expert-residency's `SettledSlots`, device-expert-cache's `device_cache.cu` and `runtime.cu` with their
  Metal version, device-speculative-round's `greedy_accept.cu`, draft-trees', hidden-state-drafters',
  draft-vocabulary-subset's and recurrent-state-recovery's reference kernels, host-expert-arithmetic's device twins,
  the C++ twins of lookup-table-rows, recorded-selection, ssd-expert-materialization and token-constraints, and
  concurrent-request-execution's `host_batch.h` and recurrent reference). What stays is what a plugin builds
  (host-expert-arithmetic's rows, host-expert-tier, device-expert-cache's `directory_ops.cu`, lookup-table-rows'
  reader, recorded-selection's pinned GEMM, token-sampling's law) and the host codecs the operations name as their
  references (`nvfp4.h`, `row_fp8.h`, `fp8_block.h`). Entries without native code left are `reference` entries.
- The scheduler features no engine can reach. Core's loader runs every engine on a step engine, which runs one step
  at a time and declares none of them: the asynchronous batch pipeline (`BatchExecution`, `prepare_batch`,
  `Profile.max_inflight_batches`), dependent execution (`StepContinuation`, `execution_depth`,
  `max_request_steps`), memory preemption (`Suspension`, `Restoration`, `prepare_incremental`), reuse-aware
  admission and admission lookahead (`scheduling_hint`, `AdmissionPolicy`, `admission_lookahead`) and prefill
  latency advice (`prefill_limit`, `prefill_target_ns`). Admission is first in, first out. `RequestTiming` and a
  result's `timing` lose `preemptions`, `recovery_ms` and `recomputed_tokens`, which were always zero.
- The runtime's second way to run a model and the layers only C++ models needed. Core's loader runs every engine
  on a step engine, whose callbacks return with their work done, so the runtime keeps one interface (`Model` and
  `Session` in `crit/core/scheduling.h`, formerly `ConcurrentModel`, `ConcurrentSession` and `concurrent.h`) and
  one worker. Gone are the one-request-at-a-time path (the old `Model`/`Session` with `advance`, `open_request`
  and the legacy text callbacks), `Backend`, `Services` and the backend drain (the factory is given the resource
  ledger), `Readiness` and deferred tokens (`Progress::readiness`, `deferred_tokens`, `ready_tokens`), retirement
  receipts with `Session::close`, `quiesce` and `Profile.retirement_timeout_ms` (`Model::retire` returns nothing
  and the session is destroyed after it), quarantined runtimes (`quarantined_engines`), `Resources`' `Arena`,
  lifecycle tracing and operator ranges (`Profile.trace`, `Engine::trace()`, `device_timings`, `crit/trace.h`,
  `CRIT_CORE_NVTX`), and the C++ text path (`TextFrontend`, `TextDecoder`, `crit/text.h`, `RequestOptions`'
  `stop_strings` and `text`, `Result::text`, `Engine::tokenize`, `detokenize` and `create_decoder`): the engine
  ABI never passed text, which `crit_core.text` handles in Python. The engine ABI and the step ABI are unchanged.
- `StepDecision.discard`, which was always 0 since Core runs one step at a time: the step ABI's decision is
  session, accepted, terminal and cancelled, and the engine ABI is version 6 (rebuild `crit_core_steps`). A model
  plugin that read `d.discard` in `commit` drops it; nothing else changes for it. The runtime's per-row failure
  path, which only a text decoder could take, went with it.

### Fixed

- A request after load no longer compiles a kernel for a prompt row or a row count load had not run. An eager
  step's prompt row past the window launched paged-kv-attention's prompt chunk kernel, which no load had compiled,
  and fused-norms' activation product compiled again at more rows than load had run. attention-state's warm-up
  now launches the prompt reader as such a row does (with and without fresh rows); the prompt chunk kernel no
  longer takes a chunk of one token or at position 1 as a constant; and the build's tuning module lists a shape
  key's knob sets (`variants`) and launches a Triton kernel through `launch`, which compiles the other knob sets
  at a compiled kernel's first launch, so a row count that reads another entry of a tuning table compiles
  nothing (paged-kv-attention's `write` and `prefill`, fused-norms' activation, stream and decay kernels).
- lookup-table-rows no longer keeps a buffer set for every prompt length. A table read from files made `ring`
  sets of pinned ids, pinned rows and device rows for each lookup size it met and kept them for good, and a
  server's prompt chunks come in every length: the memory audit showed four new 1.2 MiB sets after two prompts.
  Eager lookups now share sets sized by powers of two of rows (a lookup reads exactly its rows of one), and a
  captured lookup has a set of its own, made as it is captured, where it took one of its size's ring (so a
  capture no longer fails for want of a larger `ring`).
- expert-offload makes a prompt chunk's row buffers before it stages the chunk's experts. The first chunk of a
  new size grew the device part's rows after the cache had planned the chunk, the growth asked the device's
  givers for memory, and the cache released slots the plan had just named: the device rows then read released
  slots (an illegal memory access on a device filled to its margin, found by a run with the scratch poisoned).
  The device part has `prepare()` for it (heterogeneous-expert-dispatch's `KernelRoutes`).
- Core's memory readings no longer count the empty segments of captured graphs' own pools among the blocks torch
  holds unused (`torch()["unused"]`): `release()` cannot hand them back and no step can take them. On
  Qwen3.8-Flash-Next 234 MiB of them were read as a prompt's room already in hand, and every prompt step's peak
  as that much more than it took (236 MiB for a step that takes 6 to 92), so expert-offload's telemetry and the
  room it keeps for a chat turn were wrong by the same amount in opposite directions.
- An elastic expert cache no longer stays small after a long prompt. Torch's allocator kept the prompt's blocks,
  expert-offload measured a prompt step's need as everything the allocator held during it, and so the blocks a
  1,024-token chunk left were counted as the need of every later short prompt: on Qwen3.8-Flash-Next a 35-token
  prompt after a 6,316-token one "took" 1,328 MiB, the cache kept that much free for the next chat turn and held
  20.4 to 20.75 GiB instead of 21.7. A prompt's blocks now go back to the device when its last step ends (3 ms,
  no change in time to first token), so each prompt is measured on its own, and a need measured while the engine
  loaded (a first step, whose graph captures took memory too) is forgotten once a served prompt of that size or
  larger took less. The rule that kept a workspace under a 32nd of the cache in the allocator is gone.
- Six kernels kept once a process what belongs to a device: paged-kv-attention's CUDA tile and its staged merge,
  int-block-experts, fp8-block-rows, fixed-tile-gemm's CUDA GEMM, recurrent-scans' pool-delta-pipe and exl3-rows
  raised a kernel's shared-memory limit (the opt-in above 48 KB) on the first device that launched it and remembered
  that for the process, and int-block-experts, fp8-block-rows and fixed-tile-gemm also the CTAs that device holds and
  its multiprocessors. The limit is set on the function as loaded on a device, so a second GPU in the process refused
  their launches above 48 KB and sized its grids by the first GPU's count. The kernel core's new `device.cuh`
  (`PerDevice`, `shared_memory_optin`, `resident_ctas`, `multiprocessors`) keeps each once a device, with a device
  test (a launch above 48 KB is refused before the opt-in and runs after it). The two kernels called through ctypes
  (paged-kv-attention's CUDA tile, fixed-tile-gemm's CUDA GEMM) now launch on their operands' device instead of the
  thread's current one. fp8-block-rows' channel and blocks GEMMs also shared one CTA count among every configuration
  of both (the first one launched): each configuration has its own. Outputs are unchanged. Checked on one RTX 5090;
  no second GPU ran it.
- paged-kv-attention's module workspaces (a split launch's partials where the caller gives none, a fused launch's
  ring of states and sync words, a prompt chunk's state between its passes) were one a device for the whole process:
  two engines launching, or replaying their captured graphs, side by side would have read each other's states and
  tickets (wrong outputs, or programs waiting for ever). They are one a device and owner now: launches inside
  `with owner(key):` take workspaces of their own and `release(key)` drops them, and attention-state gives every
  DeviceAttention its own owner (released at close). A prompt chunk's state buffer that a larger chunk replaced was
  freed while an earlier captured graph still wrote into it: it is kept, as the other workspaces are.
- The CUDA attention tile's tree walk ends at a parent past the window's columns (it shifted a 64-bit mask by it and
  read the parents past the row), and int-block-experts' quantizers and SwiGLU refuse a call of 2^31 elements or more
  instead of counting them in a wrapped int.

## [0.19.0] - 2026-11-13

### Overview

crit 0.19.0 runs GGUF and EXL3 checkpoints at their stored bits and the attention of more model families. gguf-rows
reads 20 GGUF block formats (all but the ternary TQ1_0 and TQ2_0) bit for bit with gguf-py, Core reads a GGUF
checkpoint's quantized matrices as stored, and gguf-projections serves them in the projections slot, its prompt GEMM
about twice as fast as before; exl3-rows runs EXL3 projections. Attention sinks and sliding windows work on every
paged-kv-attention structure and through softmax-attention, and attention-state keeps sliding-window layers as rings,
so local layers hold only their window's KV. latent-attention is a new kernel for multi-head latent attention (decode,
prompt chunks, an E4M3 cache, sinks), read as `attention:latent`, at 0.8 to 1.8 times FlashInfer's MLA on an RTX 5090.
CUDA prompt chunks over 8-bit pages are 38 to 55% faster with the same bits. A captured round's rows each keep their
own draft width, so requests that do not draft ride the width-0 capture. Kernel work is guarded against fitting one
GPU: CI builds every kernel for every architecture it claims, `crit kernels gates` checks NaN, repeat, tile-edge,
FP32-sum and limit gates, `crit kernels report` times every kernel against its library on several models' shapes, a
tuning swept on another GPU falls back to the default knobs, and a package carries its own tuning. `serve --check`
sends one request at the full context, a runtime whose owner is gone is recovered by the next one to start, and a
client that disconnects no longer costs the engine its weights. crit 0.19.0 needs crit-eval 0.7.0.

### Changed

- The charter, the kernels README and the contribution criteria say how kernel work avoids fitting one GPU: speed-only
  constants are knobs and device limits are read from the device, an instruction one generation has sits behind a
  feature check beside the other GPUs' path, a structure that wins is added beside the ones it beat, a tuning belongs
  to the GPU it was swept on, every number names its device, and a structure leaves only on evidence from more than one
  GPU (kernels/README.md, "Guarding against one GPU"). The charter's "Not built yet" says that every kernel has so far
  run on one RTX 5090 and other architectures are compile-checked.

- The sweep's benches check a configuration's shared memory against the GPU's own opt-in limit
  (`crit_kernels.gates.shared_memory_optin()`, from the compute capability without initializing CUDA) instead of 99 KB,
  an RTX 5090's: fixed-tile-gemm's cuda-mm, fp8-block-rows' staged and native GEMMs, int-block-experts' staged and
  generic native GEMMs, exl3-rows and recurrent-scans' pool-delta-pipe. A GPU with more (an A100's 163 KB, an H100's or
  B200's 227 KB) now also times the deeper rings that fit there.

- A kernel tuning swept on a GPU of another compute capability is no longer applied where it is found implicitly
  (an engine package's `kernel_tuning`, `CRIT_KERNEL_TUNING`): the kernel builds with its default knobs, its record in
  the engine's facts says `built` with the reason, and stderr says so once. The sweep already wrote the GPU's name and
  capability into each tuning, but nothing read them, so one GPU's knob choices applied on every GPU. Given explicitly
  (`crit kernels build --tuning`, `gates --tuning`), a tuning is still used, with a warning.

- A run record with an unknown field is refused with the fields that object takes, so it is fixed in one retry:
  "Unknown fields; use extensions: reason. Takes detail, fate." (before, the unknown fields alone).

- gguf-rows' gemm decodes the matrix's sub-blocks once a call into a staging buffer (gemv's codes and scales,
  sub-block-major, in row chunks within 128 MiB), then multiplies them in the tree's lane order with the codes, scales
  and tokens copied a stage ahead by cp.async (knobs WN, WT, TB, RUN and the new STAGES, defaults 4, 2, 4, 2, 2 from a
  sweep); outputs keep gemv's bits. On an RTX 5090 at 14336 x 4096 it reaches 92 to 112 TOPS at 512 tokens (1.8 to 2.2
  times dequantizing to BF16 for cuBLAS) and 107 to 129 at 2,048 (1.06 to 1.08 times it for Q8_0, IQ4_NL, IQ3_S and
  IQ4_XS, 0.90 to 0.92 for Q4_K and Q6_K); before, 39 to 54. The quantizer writes gemm's activations sub-block-major
  (`quantize(x, by_subblock=True)`).

- A runtime a stopped process left behind is recovered by the next one to start: when the guard's owner is gone and
  its processes have settled (the proof `crit serve --recover-runtime` runs), the runtime is released with one line on
  stderr instead of refusing until `--recover-runtime` is run. A live owner, or cleanup that cannot be proven, still
  refuses, now with the reason. A client that goes away mid-request, such as a readiness probe that gave up, prints
  one line instead of a traceback.

- Plugin and engine tests find Core's runtime as Core's loader does: `PluginTestCase.core_runtime()` gives the
  `crit_core_steps` library (`CRIT_CORE_STEP_LIBRARY`, else the checkout's build, else what `crit core build` built)
  and skips, saying to run `crit core build`, when there is none; crit's own engine tests gate on the same lookup.
  Tests that checked the environment variable alone skipped even with Core built.

- SERVING.md and AGENTS.md give crit serve's cancellation grace: one second, fixed, from the cancel the server sends.
  An engine that has not returned by then is retired with its resident weights, so every prompt chunk at the full
  context and every decode step must finish well inside it.

- `crit plugins test` lists every skipped test with its reason, in its summary after the counts and under each
  plugin's line (also in `serve --check`'s log); each item carries them as `skipped`, kept with a cached pass. A skip
  still counts as no failure (a CUDA-only plugin's tests skip without CUDA), but a pass no longer hides what it
  skipped, such as model tests that need the step library.

- mma-rows: the prompt tile's last, partly filled CTA stages and multiplies only the columns of eight rows that hold a
  row below M (outputs unchanged): 2 to 6% less time for prompts of 65 to 200 rows on an RTX 5090.

- paged-kv-attention's CUDA prompt chunks over 8-bit pages (fp8, fp8:row, int8) convert the codes in registers, as
  its decode windows do, instead of through a shared 16-bit buffer and a sync that left one CTA an SM and the tensor
  cores idle. A chunk with fresh rows runs in passes chained by every thread's state: the blocks below the first fresh
  position's block in registers, that block through the 16-bit block, then the all-fresh blocks through the 16-bit
  pages' tile, which streams fresh rows asynchronously. Outputs and log-sum-exps keep the single pass's bits. On an RTX
  5090 with Qwen3.8-27B's heads, 2,048 tokens at 32,768 positions over FP8 pages: 92 to 144 TFLOPS without fresh rows,
  84 to 116 with them, and a first chunk at 1.18 times the Triton prompt chunk instead of 0.45. `prefill(splits=1)`
  reads in one range (row-invariant), `cuda_prompt()` says where the tile serves, `workspace=` holds the passes' state
  in the caller's memory, and an `out` of another dtype than the queries is refused instead of written as garbage.

- attention-state's split reader reads prompt chunks on the CUDA flash tile in one range where the tile serves,
  row-invariant like the Triton prompt chunk it replaces there (a token's bits follow its own positions): 1.2 to 1.65
  times as fast on Qwen3.8-27B's heads over FP8 pages (1.55 at 131,072 positions), the passes' state in the split-KV
  reader's declared workspace. The Triton chunk stays for `int8_qk`, outputs in another dtype than the queries, and
  heads or GPUs the tile does not take.

- Core's serving engine hands back the blocks torch's allocator caches and nothing uses after each request when the
  device's free memory is below the run's or crit serve's `vram_headroom_bytes`: a long prompt's or score's
  temporaries stayed reserved, and the next long request found no room. An engine's configuration may set
  `release_below_mib` instead (0 never releases).

- Kernel sweeps and gates on a model also take the stacked matrices engines load, siblings of one module that read the
  same input (q, k and v; gate and up), as their rows together. crit Kernels' model reader takes a multimodal
  checkpoint's `llm_config` and refuses a folder that holds no checkpoint instead of giving no shapes.

- The kernel gates' NaN check follows a structure's causality: `nan_rule` `causal` puts the NaN in the last token and
  requires every earlier row to keep its bytes, and `nan_block` lets a chunked form's dense chunk matmuls carry it
  through their own chunk only, each proven by a planted control. recurrent-scans benches every structure it has (SSD
  prefill and columns, delta columns, causal conv, conv segments).

- Eval refuses a candidate whose newest `serve --check` failed: its build is a draft, to be sealed `check_failed` and
  fixed as a new candidate, and an Eval would count it on the ladder.

- `crit run ... serve --check` fails a drafting engine whose facts declare no `draft_bytes_per_step` (the bytes one
  draft step reads): crit-eval cannot price the round's floor without it.

- AGENTS.md: prompt chunks go to the projections with `prompt=True`; sweep with `--matrices` from the candidate's
  serve --check report; a model plugin's test checkpoint has every part the real one ships. The README's case studies
  are the Qwen3.8-27B plugin engines, speed and preserve, with their active development time.

### Added

- CI builds every CUDA kernel for every compute capability its kernel.json claims and checks each architecture's
  register and shared-memory limits, with the CUDA 13.1 toolkit's wheels and no GPU (`.github/workflows/kernels.yml`,
  one job a kernel, on changes to `kernels/`), and runs crit_kernels' host tests. `crit kernels gates ID
  --limits-only` runs that check alone. Before, the every-architecture build ran only where someone ran it by hand.

- latent-attention reads prompt chunks: `prefill(q, cache, table, position, page, rank=R)` (structure `prompt`), a
  chunk of one sequence causal over the latent cache, its tokens times heads in tensor-core tiles that read each piece
  of cached rows once, the output written directly (no partitions, no partials; a token's bits independent of the
  chunk's length), at a device position for CUDA graphs, with sinks, lse and the E4M3 cache. 0.78 to 0.92 times
  FlashInfer's MLA prefill on an RTX 5090 (16 and 128 heads, chunks of 512 and 2,048). An E4M3 cache takes one scale
  a row too (a KV pool's fp8:row pages).
- softmax-attention reads latent attention (`attention:latent`, `forward(..., latent=R)`): queries of the latent then
  the rotary part over an attention-state pool of keys only with one KV head, whose rows are each position's latent
  then its rotary key (BF16, FP16 or fp8:row pages; FP32 in torch). A step's decode tokens and verify windows and
  captured rounds read through latent-attention's decode reader (a round's column equal to the step's decode of the
  token), prompt rows and captured prompt chunks through its prompt reader. A model with multi-head latent attention
  keeps one row of rank + rotary width a position instead of every head's keys and values.

- attention-state keeps sliding-window layers as rings (`kv:sliding`): a pool made with `pool(..., sliding=W, span=S,
  positions=N)` holds each sequence in ceil((W + S) / page) + 1 pages whatever its length, the attention's page tables
  the ring expanded (a position older than the window aliases a newer one's page, which the window never shows), so
  the kernels read it unchanged. Steps, prompt chunks, forks, rollbacks and captured rounds work over rings; `bind`
  takes no page past the ring; softmax-attention reads such a pool with its window by itself. A model with local
  layers (a window of 128 or 1,024 against a context of 128K) keeps their KV at the window's size: the fact
  `kv_sliding_bytes_per_sequence` counts it a sequence, not a token.

- paged-kv-attention takes sliding windows: `attend(..., sliding=W)` and `prefill(..., sliding=W)` (a token sees its
  last W positions, its own included) on every structure but block lists, with the pieces, blocks and splits below a
  tile's window skipped, so a decode row over a long cache reads only its window (on an RTX 5090 at 60,000
  positions, a window of 128: 12 us against 49 on the split launch, 10 against 38 on flash, 10 against 31 on the
  CUDA tile). Combined with sinks this is the attention of models that alternate local and full layers with learned
  sinks. A fused launch keeps the split launch's bits. softmax-attention provides it as `attention:sliding`
  (`forward(..., sliding=W)`), its torch fallbacks masking the same positions.

- Kernels: latent-attention, multi-head latent attention in its absorbed form (operation `latent-attention`,
  attention-state's float64 reference `latent_attention.py`): decode tokens and verify windows of up to 64 columns of
  every query head over a paged cache of compressed rows (the latent, then the rotary key; the latent the value too, so
  a piece of rows is read once for the scores and the values), split over KV partitions and merged in partition
  order, with log-sum-exps and sinks; the cache in BF16, FP16 or E4M3 (a scale a part and position: twice the
  positions in the same memory, as fast as BF16 with FP16 queries). A token's bits are the same alone, in a verify window and beside other rows; one
  CUDA graph replays at every length. On an RTX 5090 at rank 512 it runs at 0.8 to 1.8 times the speed of
  FlashInfer's MLA wrapper (16 and 128 heads, 4,096 to 32,768 positions), its yardstick.

- paged-kv-attention takes attention sinks: `attend(..., sinks=)` and `prefill(..., sinks=)` (each query head's sink
  logit, FP32 `[heads]`) on every structure (the split launch and merge, fused, flash, flash fused, the CUDA tile split
  and fused, and the three prompt-chunk paths), one more position with no value in every row's softmax, folded into
  the row's final state after its partials (the kernel core's `attention_softmax.sink` and `softmax_sink2`): a fused
  launch keeps the split launch's bits, the log-sum-exp counts the sink, and sinks of -inf change no bit.
  softmax-attention provides them as `attention:sinks` (`forward(..., sinks=)`, a new attention fact): every reader
  of attention-state's device form takes them, its torch fallbacks fold them in after their passes, and FlashInfer
  and torch's SDPA, which have no sinks, give way to crit's readers or the masked softmax.

- gguf-rows reads every GGUF block format for weights but the ternary TQ1_0 and TQ2_0: Q4_0, Q4_1,
  Q5_0, Q5_1, Q2_K, Q3_K, Q5_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and MXFP4 join Q8_0, Q4_K, Q6_K, IQ4_NL,
  IQ3_S and IQ4_XS, in gemv, the gathered gemv, gemm and dequantize, each a decoder in the kernel core
  (`crit_kcore/gguf.cuh`, the IQ grids in `gguf_grids.cuh` from Core's `gguf_grids.json`). Every format's stored values
  equal gguf-py's decoder bit for bit and its products the declared order. Q2_K, whose minimums cover 16 values,
  multiplies its codes times their 4-bit scales and its minimums as two exact integer dots (91 and 105 TOPS at 512 and
  2,048 tokens on an RTX 5090, 14336 x 4096); the formats with one scale a sub-block reach 117 to 132 TOPS, the rest 87
  to 112. Core's `gguf_blocks` decodes the same formats on the host (`subblocks(..., halves=True)` gives Q2_K's two
  minimums), and gguf-projections loads them all as `gguf:<TYPE>`.

- A row of a captured round may have a width of its own (`crit_plugins.rounds` bind's and start_row's `width`): the
  chain accept keeps at most that many of its drafts, so a request that does not draft (width 0) rides the same
  captured round as drafting ones and emits one token a round, a greedy row plain greedy decoding's. The acceptance
  counts offer only the drafts a row's width keeps. The next round of a row count runs at the narrowest captured width
  that holds its decoding rows' own widths, so rounds of requests that do not draft run the width-0 capture where one
  is captured.
- `crit run create` says "Eval cannot score this model" with crit-eval's reason (`scoring_refusal`: no layout for its
  GGUF architecture, no modeling code in the installed transformers, ...) before any candidate is built. The run is
  created; a run found out only after its candidates were built that no reference could judge them.
- `crit kernels sweep --model` (and `gates`) read a GGUF checkpoint: its matrices in a GGUF block format are its
  shapes, by their GGUF type (an expert stack with its count), so gguf-rows tunes on the model's own shapes;
  gguf-projections' facts list every matrix it loaded, for `--matrices` from a serve --check report.

- Inference: gguf-projections, the projections slot over a GGUF checkpoint's matrices at their stored bits (every
  format gguf-rows reads as `gguf:<TYPE>`, the file's dense tensors as `bf16`): decode rows on gguf-rows'
  gemv (each row's bits its own), an expert stack's rows each against their expert (`matmul_experts`), prompt rows
  dequantized to BF16 for cuBLAS (option `prompt` dequantize, the default) or on gguf-rows' gemm with the decode rows'
  bits (`exact`); the host-decoded BF16 values through torch where the kernel cannot be built. A model plugin reads
  the stored rows with Core's `GgufTensors`.

- Core reads a GGUF checkpoint's quantized matrices as stored: `crit_core.model.gguf_blocks.GgufTensors` maps the
  files and gives a tensor's bytes shaped as its rows (`[N][row bytes]`, an expert stack `[E][N][row bytes]`) without
  copying, and `subblocks` / `stored_values` are the exact host decoders of every format gguf-rows reads
  (each 32-value sub-block's codes and FP32 scales; the stored values bit for bit as the formats' own decoder gives
  them). The IQ grids live in Core's `gguf_grids.json`, which the kernel core's tables are checked against;
  gguf-rows' bench takes its decoders from Core.

- gguf-rows: `gemm` multiplies any number of rows (a prompt) on the s8 tensor cores in gemv's declared order, so a
  prompt's rows have the decode rows' bits: a part's sub-blocks are visited lane by lane in the order the 32-lane tree
  takes the lanes, each visit decoding one sub-block of the CTA's weight rows into shared memory for one s8 m16n8k32
  MMA a 16 x 8 tile, and finished lanes fold into the tree's partial sums (the kernel core's new `lane_at32` and
  `fold32`). `matmul` takes gemv up to 8 rows and gemm beyond. On an RTX 5090 it reaches 20 to 34 TOPS on
  14336 x 4096 at 512 and 2,048 tokens, below dequantizing to BF16 for cuBLAS (50 to 121 TFLOPS), which an engine can
  still take where the prompt's bits need not equal the decode rows'.

- Kernels: gguf-rows runs GGUF projections at their stored bits, in Q8_0, Q4_K, Q6_K, IQ4_NL, IQ3_S and IQ4_XS: `gemv`
  for 1 to 8 rows quantizes each row to int8 per 32 values and takes each 32-value sub-block's exact integer dot with
  the stored codes on `dp4a`, weighted in FP32 in a declared order (a row's bits independent of the rows beside it,
  the split of a row's sub-blocks a function of the shape and device), plain or gathered over a stack of experts (each
  row its expert's matrix, bit for bit the plain call); `dequantize` gives the stored values (FP32 exactly as the
  format's own decoder gives them, BF16 one rounding) for prompts. The kernel core's `gguf.cuh` holds the formats'
  decoders. On an RTX 5090 a 1-row call reads the weights at 1,056 to 1,429 GB/s cold on 14336 x 4096 and
  4096 x 14336, against the probe's 1,487 GB/s.

- `crit run serve --check` sends one request at the run's full context and judges it from Core's step probe:
  `full_context_prompt_step` fails when the request's slowest prompt step takes crit serve's cancellation grace or
  longer (a cancel settles only between steps, so the host would be retired with its weights), and
  `full_context_device_memory` fails when the least free device memory at any of its prompt or decode steps' peaks is
  below the run's `vram_headroom_bytes`. The prompt is sized from the tokens the engine counts, through the chat API,
  so any package can take it; an engine that reports no probe leaves both unverified. `CRIT_CORE_STEP_PROBE=1` makes
  Core's StepEngine time every step with its device work finished and sample the free memory at its peak, reported in
  its telemetry's `core.probe` (serve --check sets it for its engine only; native rounds are not probed).

- Kernels: exl3-rows runs EXL3 projections at their stored bits (1 to 8, or 1.5 to 3.5): `gemv` for 1 to 16 rows
  stages each block's trellis slices and activations in shared memory by cp.async (several stages in flight), decodes
  each tile once into the A operand of an FP16 tensor-core MMA whose B operand is the rows, between the matrix's
  Hadamard rotations (FP32 sums, a row's bits independent of the rows beside it, the input split by the shape and
  device; the three launches chained as programmatic dependents), `dequantize` gives the dense BF16 matrix for
  prompts, `decode` the tiles' FP16 values, the format's own decoder's bits for every codebook and bitrate. The kernel
  core's `exl3.cuh` holds the codebooks, the bitstream (three words and constant shifts a lane) and the tile order. On
  an RTX 5090, reading from memory, a 4-bit 5120 x 17408 matrix takes 37 us at 1 row against 111 us for cuBLAS on the
  dense BF16 matrix, level with or faster than the format's own GEMV at 1 to 4 rows on the shapes measured.

- Core's checkpoint schemes recognise EXL3 matrices (`base.trellis` with their sign vectors and codebook marker) as
  stored matrices instead of refusing the checkpoint; crit Eval decodes them for its reference. EXL2 stays refused.

- mma-rows: `glu=True` on `mm_rows` and `mm_prompt` (and the FP4 MMA) takes an MLP's gate and up rows as one matrix
  interleaved by eights (`interleave_glu`) and writes `silu(gate) * up` from the epilogue, out [M][N / 2], so the two
  projections' outputs are not written and read back by a separate activation product. Within one BF16 step of the
  projections followed by `silu_mul`; each row keeps its bits alone, beside others and in the prompt tile.

- fused-norms: `rms_norm_fp8_rows` and `rms_norm_nvfp4` quantize a norm's rows for a W8A8 or W4A4 projection in the
  norm's own launch, so the rows are not written and read back by a separate quantizer. The codes and scales are those
  of `rms_norm` followed by mma-rows' `quantize_fp8_rows` or `quantize_nvfp4`, bit for bit, residual included.

- `crit kernels gates ID`: five gates beyond a kernel's reference check, each proven by a planted control that must
  fail it (`--controls`): a NaN in one row reaches that row's output and no other row's bytes; five calls give the
  same bytes and leave the inputs untouched; the served rows pass at every tile edge (powers of two less one, itself,
  plus one); integer operands whose partial sums pass FP16's range give the exact FP32 sum; every function built for
  every claimed architecture fits its registers and static shared memory (read from the binaries, no GPU). Benches
  may declare scratch buffers, a contained NaN rule and integer inputs (decode-gemv's BF16 weights have them).
  kernels/TEMPLATES.md lists the gates and the memory-ordering points a review checks where no gate does.

- Core: `StepLayout::read_rows()` (Python `layout.read_rows`) and each placement's `read` name the rows whose
  last-layer output anything reads: the logits rows, or every row of an item that sets `readers` (taps, hidden-state
  readouts). The last layer runs its query, attention, output projection and MLP on those rows only; the other rows
  stop at the layer's state writes, and a prompt chunk inside its prompt stops there altogether.

- `crit kernels report --model A --model B`: every kernel's gates and its speed against its yardstick library on
  several models' shapes (each structure's distinct shapes over them, at most `--most`, by default the smallest, middle
  and largest), the default build timed beside the library at its first and last `tune_rows` as the sweep times them,
  as JSON (`--out`) and a markdown summary (`--markdown`), a line a kernel and then each point.

- `crit kernels sweep --model CHECKPOINT --matrices REPORT`: exactly the shapes a candidate's engine loaded, from its
  serve --check report (which now records the served engine's facts), an engine's facts or a list of `{N, K, format}`,
  instead of the checkpoint's tensors and their likely stacks.

- An engine package carries its kernel tuning: configuration `kernel_tuning` names a folder of `crit kernels sweep`
  tables inside the package (`--out tuning` in the candidate's source), applied before any plugin is made, so the
  package served, sealed and published is the one Eval measured. `CRIT_KERNEL_TUNING` stays for a quick try, which no
  package carries.

- Projections `matmul(x, matrix, out_dtype=None, prompt=False)`: `prompt=True` marks a prompt chunk, which takes the
  prompt rows' arithmetic at every length, so a prompt of a few tokens is not multiplied as decode rows and its rows
  equal the same rows in a longer prompt (padded past the decode rows where a plugin splits by count).

- Embeddings `store(..., device=)`: the device a table serves, by default the one its rows are on, so a host table's
  rows read on the CPU go to host memory without passing through the device.

- `crit run ... cost --idle FROM..UNTIL` (repeatable) names stretches in which the build's session was offline with
  nothing running; the record gives the active hours beside the wall hours.

- CONTRIBUTING.md says how pull requests land (applied to the next release's squash commit with credit, then closed),
  with a pull request template that says so. ROADMAP.md groups the roadmap's GitHub issues by theme, and a bounty issue
  template's done-check is the exactness gates plus a re-run on the reference RTX 5090.

### Fixed

- paged-kv-attention turned a NaN query into a zero output on its decode and verify paths (split, fused, flash,
  flash fused, the CUDA tile split and fused) and on the flash and CUDA prompt chunks: the final normalization took a
  NaN sum for a row with no position. The kernel core's `finish` / `lse` and `softmax_finish` / `softmax_lse2` now
  keep 0 only for a sum of 0, so a NaN reaches its token's output (and log-sum-exp) and no other token's; finite
  outputs keep every bit.

- paged-kv-attention keeps a module workspace it replaces when a larger launch outgrows it. A CUDA graph captured
  over the old one replayed into its freed blocks, which other tensors then held: garbage attention and writes over
  those tensors. softmax-attention's `reader: cuda` failed that way with "no selectable token" on a 27B engine above
  about 66k tokens of context, where a prompt bucket's launch is fused and its workspace grew after the smaller
  buckets were captured.
- `crit kernels sweep` stopped during its builds (a `timeout`'s SIGTERM, Ctrl+C) ends the builds and still writes
  the table of every structure's defaults, its report saying how many builds were done; before, it wrote nothing
  (a stop while it measures already wrote what it measured).
- The native serving options tests build their engine with the request-end release threshold `ResidentEngine` now
  sets (none, as on a host without device memory); every one of them failed on the attribute it lacked.
- crit serve keeps the engine when a stream write fails because the client is gone: the OSError cancels the turn as a
  closed connection does, the host settles it and keeps its weights, and the caller gets the error as raised. It used
  to unwind past the cancel and retire the host, so a fast engine whose next piece beat the disconnect watcher lost
  its resident weights. crit serve now prints one stderr line whenever it retires or closes an engine host, with why
  (a cancel not settled within the one-second grace, an engine error, shutdown).

- Core's config rules read attention geometry per layer kind. Full-attention layers may declare their own KV heads and
  head dim and that their keys equal their values (one cached tensor), through the new `full_attention_kv_heads`,
  `full_attention_head_dim` and `full_attention_keys_equal_values` entries of `config_keys.json`; sliding layers keep
  `num_key_value_heads` and `head_dim`. `attention_geometry()` gives both, and `attention_shape` and
  `kv_elements_per_token` count the full layers' own. `experts_per_token` is also spelled `top_k_experts`, in the
  Python and C++ readers alike. The GGUF folder check compares a per-layer `head_count_kv` list layer by layer and
  `key_length_swa` with the sliding layers' head size.

- mtp-decode: `AcceptanceEstimator` learns from confident drafts. `feature(q_max)` takes a probability, which rounds to 1
  in FP32 once the other tokens' mass is under about 6e-8 and is clamped at 1e-6, so every confident draft had the same
  feature. `feature_from_logits(scores, temperature)` computes logit(q_max) from the scores as the top score less the
  log-sum-exp of the rest, in FP64, and `observe_feature`, `predict_feature` and `survival_features` take it.

- Activation quantizers hid a NaN or an infinity: the block maximum dropped a NaN and an infinity saturated to the
  largest scale, so a non-finite activation became finite products. The kernel core's `abs_max` folds both to NaN,
  `nvfp4_block_scale` returns NaN for such a block and the NVFP4 byte path writes the E4M3 NaN byte (the block-scaled
  MMA carries it), and INT8 and INT4 scales come out NaN; int-block-experts and grouped-experts use it in every
  quantizer, across warp lanes too. Only that token's outputs turn NaN.

- fused-norms' `rope` and `rms_norm_rope` pick their head block by the head width alone, so a q/k pair keeps its
  separate calls' bits: with one key head one BF16 bit differed on a 512-wide head.

- fixed-tile-gemm's `mm_split` failed to compile on a one-row call.

- The kernel gates put in-place buffers (a residual, residual streams; `bench.restore`) back before every call, and the
  NaN gate no longer poisons an output buffer the call then overwrites. Benches get the untuned build
  (`bench.default`), as in the sweep.

- The sweep's timer reads a second buffer after its L2 flush, so the flush's dirty lines are not written back during
  the timed call: that took 10 to 20% of a weight-streaming call's DRAM time on an RTX 5090. Ratios to the libraries
  held, but absolute rates and small-call picks were skewed.

- quantized-projections records a draft-only role whose matrices are stored in two formats as a group a format
  (`role:stored` for the second) instead of refusing it, and re-encoding hands its FP64 slices back to the device: a
  whole-vocabulary head's re-encode kept several GiB reserved for the engine's lifetime.

- `crit run ... ablate --set`: a string option, or a choice among strings, takes a number-like value as written
  (`tile_rows=0` is the choice "0").

- The plan check compares a plan's one prompt activation for every format with the engine's same activation per format
  instead of failing it.

- `crit run ... cost` counts no Claude Code `<synthetic>` messages as model requests.

## [0.18.0] - 2026-11-11

### Overview

crit 0.18.0 runs its own kernels. The twelve kernels are templates in crit (`kernels/techniques/<family>/<id>/`), built from a
kernel core on the user's machine for its GPU and tuned by `crit kernels sweep` on the model's own shapes, timed
against the library doing the same arithmetic; nothing is downloaded or shipped prebuilt, and the library plugins
run these kernels by default, the libraries kept only as yardstick options. New this release: crit's own prompt
GEMMs for FP8, NVFP4, MXFP8 and BF16, INT4 with a zero point (AWQ, GPTQ) on every projection and expert kernel, a
format-generic decode GEMV, a CUDA flash tile for paged attention, and sweeps tuned by sequence length. On an RTX
5090, Qwen3.8-27B at 200k context decoded at 188 tok/s and prefilled at about 7,850 tok/s with crit's kernels,
against 176 tok/s and about 7,000 with cuBLASLt's prompt GEMMs. Runs follow one set of plan rules, `crit run ...
next` names each next step, and `crit run ... ablate` measures plugins and options on Eval's data. `crit run`
also takes external candidates, engines crit did not assemble from plugins, started from their own launch command
and served, checked, evaluated, ablated and traced like a plugin package. `crit core build` builds Core's runtime,
and crit serve, crit pull and crit mcp refuse what a web page or a crafted card could send them. crit 0.18.0 needs
crit-eval 0.6.1.

### Added

- crit's kernels are templates each engine builds and tunes for its device. The twelve kernels that were to be published to the Hugging Face
  kernel registry are source in crit, `kernels/techniques/<family>/<id>/` (projections: decode-gemv, mma-rows, pinned-gemm,
  fixed-tile-gemm, fp8-block-rows; attention: paged-kv-attention; recurrence: recurrent-scans; experts:
  grouped-experts, int-block-experts, topk-routing; norms: fused-norms; starters: workbench-starter), each described
  by a `kernel.json`: its operations, sources, compute capabilities and structures, every structure's speed-only
  constants (warps, rows a warp owns, tiles, K splits, stages, loads in flight, Triton block sizes) as knobs with the
  values a sweep may try. Nothing is downloaded or shipped prebuilt. `crit kernels build ID` compiles a kernel for the
  current GPU on first use (cached in `CRIT_KERNEL_CACHE`; `--arch` checks the build for other architectures), with
  the default knobs, which reproduce the kernel as it was hand-tuned. `crit kernels sweep ID --model CHECKPOINT --out
  DIR` builds every knob configuration (many a build, picked at run time), checks each against the operation's
  reference and the row-invariance contract, times it on the checkpoint's own shapes with a cold L2, re-times the
  picks, and writes the tuning table the build reads; a knob that changes the summation order keeps one value a shape
  for every row count. On an RTX 5090 with Qwen3.8-27B's NVFP4 matrices, the decode GEMV's sweep (192 configurations,
  about 6 minutes) was 1.0-1.5x faster than its defaults and within 4% of a hand-tuned specialist GEMV up to 4 rows.
  `crit kernels list` shows them; kernels/TEMPLATES.md is the contract for a new kernel or structure.
- A kernel core (`kernels/core/`): the building blocks kernels are made of, with no algorithm in them. Number formats
  (E4M3, E8M0, E2M1, INT4 groups, encoders, block-scale layouts), tensor-core multiplies per GPU generation (mma.sync,
  block-scaled FP4/FP8 on sm_120a, wgmma and tcgen05 wrappers), cp.async pipelines, ldmatrix, TMA and programmatic
  launch, crit's fixed-order reductions, epilogues, column and chunked scans, the online softmax and the block-scaled
  declared order, CUDA headers (`#include "crit_kcore/<area>.cuh"`) and Triton modules (`from crit_kcore import
  <area>`), each checked on the device. The kernels take these blocks instead of private copies, with the same bits;
  a kernel's build key covers only the core blocks it uses. New tensor-core structures, timed against the library at
  the same arithmetic on an RTX 5090 (vs library, above 1 faster): paged-kv-attention's `flash-window`,
  `flash-fused` and `flash-prefill` (prefill 0.77-1.5, verify windows 0.6-2.5, from 0.36-0.68 and 0.15-0.42);
  recurrent-scans' `delta-prefill-tc`, the chunked gated-delta prefill (1.01-2.34 against FlashInfer, from
  0.17-0.50); mma-rows' staged decode tile (knobs `SW`, `SPF`: 0.89-1.01 against flashinfer's NVFP4 GEMM at 1 to 16
  rows, the same bits as the register tile); fp8-block-rows' and int-block-experts' `gemm-staged` (1.5-3.6x their
  previous prompt GEMMs, 0.09-0.91 of cuBLASLt's block-scaled GEMMs). Prefill structures that give up row invariance
  for speed say so (`row_invariant: false` in kernel.json, `prompt_rows` in the plugin's facts), and the plugins keep
  the previous structure selectable (`reader: split|flash`, `prefill: tensor_cores|fp32`, `gemv: crossover`).
  `crit kernels sweep` times the library as a graph replay where it captures (`library_timed`).
- Kernels take any weight format and more of the libraries' designs, timed on an RTX 5090 (vs library, above 1
  faster). decode-gemv has one CUDA GEMV (`gemv-cuda`) and one Triton GEMV (`gemv-triton`) for every weight format
  (NVFP4, FP8, MXFP8, BF16, INT4 with a zero point a group), the format a decoder trait over the kernel core; the
  previous per-format structures stay as aliases with their bits and their tunings. INT4 with a zero point (AWQ and
  GPTQ, act order included) runs on decode-gemv, mma-rows' tiles and grouped-experts (format `int4_zp`), exactly
  (two BF16 MMAs where BF16 cannot hold the value), with `repack_awq` / `repack_gptq` writing crit's rows from the
  checkpoints; quantized-projections loads `awq` and `gptq` matrices and grouped-experts packs them into bundles.
  recurrent-scans' `pool-delta-pipe` stages the decode rows' state tiles by cp.async (the same bits as `pool-delta`,
  0.75-1.73 against FlashInfer at 1 to 32 rows, the plugins' default `scan`). fp8-block-rows' and int-block-experts'
  `gemm-native` accumulate prompt GEMMs in the block-scaled tensor cores (MXFP8, NVFP4; 0.82-1.76 of cuBLASLt;
  `row_invariant: false`; quantized-projections' `prompt_gemm` picks them, `auto` where they are faster).
  paged-kv-attention has a CUDA flash tile (`cuda-window`, `cuda-fused`, `cuda-prefill`; `reader: cuda`).
- crit's own prompt GEMMs for every format, timed on an RTX 5090 with Qwen3.8-27B's matrices (vs library, above 1
  faster). fp8-block-rows' `gemm-native-channel` (per-channel FP8, BF16 out from the epilogue: 1.04-1.50 of cuBLASLt)
  and `gemm-native-blocks` (128x128 FP32 weight blocks and MXFP8 with scales that are not powers of two, a span
  accumulator folded once a scale span: 1.0-1.33 on small outputs, 0.65-0.78 where compute dominates, against
  cuBLASLt's MXFP8 GEMM since cuBLASLt has no FP32 block-scaled FP8 GEMM on sm_120). fixed-tile-gemm's CUDA BF16 and
  FP16 GEMMs, `cuda-mm` (row invariant, one summation order for every row count) and `cuda-mm-split` (K split over
  CTAs, partials added in split order): at or above cuBLAS from 16 to 8192 rows except 0.97-0.99 on 5120x17408 at 32
  and 64 rows. int-block-experts' NVFP4 `gemm-native` on sm_120a: a producer warp feeding TMA tiles through an
  mbarrier ring to 4, 8 or 12 consumer warps (192x128 tiles), cuBLASLt's tiled scale layout, BF16 out with a column
  scale, TMA stores overlapping the next tile, stream-K for the last wave (planned from the job's capacity, so bits
  stay fixed a configuration), and a faster `quantize_nvfp4` with the declared bits: 0.89-0.96 of cuBLASLt from 512
  to 8192 tokens, 1.14 on the down projection at 128.
- paged-kv-attention's CUDA tile: row groups of several warps (`VWARPS`), 8-bit pages converted in registers
  (`CONVERT`; FP8 decode at 131K positions 208 to 125 us), a staged split merge, fused launches with few partials run
  as a window launch and that merge, the fused ring's folds spread over the next group's programs, lane-major split
  partials, and opt-in order knobs that size splits by position (`RAMP`) or fill one launch shape (`CHUNK` 192, 288,
  320). Default bits are unchanged. Against FlashInfer at 1 to 64 columns after a sweep: 0.76-1.06 on Qwen3.8-27B
  BF16 pages, 0.81-1.06 on FP8 pages, 0.72-1.69 on Nanbeige4.3-33B INT8 pages; 16 columns are the slowest case.
- `crit kernels sweep`: a structure may declare `lengths` (kernel.json `{"field", "values"}`), swept as shapes with
  one order-knob group a shape and speed-knob table entries a length range (`crit_lookup_S(..., length)`,
  `knobs(..., length=)`); older tuning files hold for every length. paged-kv-attention declares 4K, 32K and 131K
  positions. A sweep that kept only the bench's 4K tuned long-context attention at 4K.
- `crit_plugins.kernels.load_kernel(name, folder=..., tuned=False)` loads a kernel with its default knobs whatever the
  engine's tuning (recorded `<id>:untuned`), for a plugin that needs one configuration a shape at every row count.
- A Triton kernel may hold CUDA sources (kernel.json `cuda`), compiled by crit's build like a CUDA kernel's (build key,
  architectures, generated tuning header, one library a variant); and a structure may alias another kernel's
  structure (kernel.json `alias`), its tuning table, sweeps and `select()` following it, a forwarded kernel recorded in
  the engine's facts.
- `crit run ... ablate --data EVAL_DATA` adds crit-eval's speed cases and held-out prompts to every variant, each at
  the model's sampling with `--seeds` requests (default 3, seeded where the package declares seeded sampling), and
  judges each plugin over them with the run's cases. The run's cases are often one short prompt: a drafter whose
  copied drafts replaced correct ones on long prompts was judged neutral there (-1%) while it cost a long-prompt
  Eval case a quarter of its greedy decode.
- `crit run ... ablate --set PLUGIN.OPTION=VALUE[;...]` serves option variants, the full selection with options
  changed, beside `full` on the same cases, with a verdict per variant (`crit_plugins.ablation.option_variants` and
  `settings`): the in-process A/B a build otherwise writes its own probe scripts for.
- A plan may forecast a candidate to improve `context_decode` or `context_ttft` alone: a run's fix for long-context requests that decoded outside its captured rounds was refused
  for improving neither decode nor prefill, and the plan passed only once it added a draft-width lever nobody had
  measured, which then failed the generality gate. A lever's change may cite its measurement, `measured:PATH` (a file
  of the run, recorded with its sha256 in `inputs.lever_evidence`), and the refusal of a lever below the noise says
  to measure it rather than raise its estimate.
- `crit run ... next` reads the run's records and names the one next step, its command and why (`crit.run.state`):
  each open candidate is in one explicit state (`created`, `build_failed`, `packaged`, `check_failed`,
  `serve_checked`, `eval_stopped`, `evaluated`, `running`), and the run's step follows from it, its rung and its plans.
  Agents learned the order of about 13 commands per candidate from prose and from refusals.
- `crit run ... plan --draft --record plan.json` writes the next plan's skeleton (the parent, TODO where the planner
  writes, the parent's scores as a flat forecast, the review and weakest entries the plan owes) and prints what crit
  computes for it: the objective and rung, the priorities, each KPI's score, limit and noise, Eval's gaps and the
  parent plan's outcome.
- `crit run ... serve --check` runs the package's plugin tests first (`crit plugins test`, cached; `--no-plugin-tests`
  leaves them out), so one command is the check before a candidate's Eval or seal.
- External candidates in `crit run`: a configuration's `engine` declares an engine crit did not assemble from plugins
  and the command that starts its own OpenAI-compatible server (`{port}`, `{host}`, `{model}`, `{context_tokens}`,
  `{package}`); its package holds no `plugins/`. `serve` (with `--check`), `eval` and `measure` start it under the
  attempt's reservation, wait for its health path, run the serving checks, crit smoke, Eval and the benchmarks
  against its URL, and stop its whole process tree when the work ends, fails or passes the deadline, keeping its
  output as `logs/engine-ATTEMPT.log`. A plan for one says `"engine": "external"`, its candidate 2 is a technique
  round, and levers name a technique or a kernel template (`kernel:FAMILY/ID`) instead of a plugin.
- `crit run ... ablate` on an external candidate serves only the options its configuration declares (`--set
  OPTION=VALUE`, or each declared switch flipped) beside it as built; an engine that declares none has no ablation, and
  the report says so.
- `crit run ... kernels` traces an external candidate's own process tree with crit-eval's CUPTI tracer while it
  serves the run's cases, or records why it cannot; MCP offers it as `crit_run_kernels`.
- `crit_plugins.device.release_cached_under(free_below)` hands back what torch's caching allocator holds and nothing
  uses only when the device has fewer than `free_below` bytes free: a model plugin calls it where a request's
  temporaries end (a prompt's last chunk, a long score), so one request's cached blocks leave room for the next
  one's allocations without a release, and its wait on the device, after every step.
- `crit core build` builds Core's runtime, `crit_core_steps`, which every engine runs on: in a checkout into
  `build/core-steps`, from an installed crit (whose wheel carries no C++) from Core's source at the release's tag into
  Core's cache. Core's loader finds that library without `CRIT_CORE_STEP_LIBRARY`, and `crit core where` names the one
  it loads.

### Removed

- The kernel register: `kernels/SOURCES.md`, the catalog commands `crit kernels target / inventory / search / inspect /
  retain / verify` with their record format (`kernels/FORMAT.md`), and the OpenBLAS retained-provider example. crit's
  kernels are its templates; the few libraries an engine calls directly are listed in kernels/README.md as each
  template's yardstick. `crit kernels sweep` times that library beside the tuned kernel on the same shapes
  (`yardstick` in kernel.json, `library_us` and `vs_library` in the report).
- Time-budget segments: a request's `budget_minutes` without `candidate_budget`, and `crit run ... resume --minutes`
  without `--candidates`. A segment's budget is candidates on the ladder; `budget_minutes` is its safety window.

### Changed

- crit's kernels live in `kernels/techniques/<family>/<id>/`, the kernel core beside them in `kernels/core/`, the way
  crit keeps `core/` and `inference/` apart; the workbench's starter stays in `kernels/starters/`. An engine picks
  each operation's kernel in three levels (kernels/README.md#levels, AGENTS.md): crit's kernel wherever one covers it,
  a library only where crit has none, the workbench where neither fits.
- The library plugins run crit's kernels by default and keep the libraries only as yardstick options.
  quantized-projections' `prompt_gemm: auto` is `native` for every format (MXFP8, FP8 rows, NVFP4, BF16, FP8-decoded
  weight-only NVFP4 with `prompt: fp8`); `pinned` runs crit's row-invariant GEMMs with their default knobs, and the
  previous cuBLASLt pinning is `pinned_cublaslt`. The attention plugins' `flashinfer` option is off by default,
  fused-norms runs without FlashInfer (one-pass long rows, programmatic launch), and sequential-recurrence's prompt
  rows past `exact_rows` run recurrent-scans' chunked prefill, fla's kernel only with its new `fla` option. On an RTX
  5090, Qwen3.8-27B at 200k context decoded at 188 tok/s and prefilled at about 7,850 tok/s (first token 153 ms) with
  crit's kernels, against 176 tok/s and about 7,000 (172 ms) with cuBLASLt's prompt GEMMs.
- `crit kernels sweep` times each library yardstick beside the configuration picked for that shape and row count,
  both as graph replays where they capture, alternating (`paired_us` in the report): timed once after a long sweep,
  the library read up to 60% slower than it is.
- One set of plan rules for every run. Runs no longer record which version of the rules they follow (`plan_rules`
  2 to 8, `plans`, `plan_limits`, `lever_fates`, `budget_counts`): every new plan is checked by the current rules,
  every candidate names a plan and every lever's fate, and every segment's budget counts the candidates on the
  candidate ladder. Runs of earlier releases that recorded those keys still load, to show, serve and replay.
  docs/RUN_RECORDS.md#plan lists the checks once, flat, instead of eight versions of deltas.
- A plan judged only on its forecast gets a warning (`inputs.warnings`) instead of a refusal: a forecast high below
  half of a needed family's limit, and a candidate from 2 to 5 below +10% on p1. A run inflated its prefill forecast
  to the minimum the reach rule accepted, which AGENTS.md forbids in another paragraph.
- A run started from a crit-hub engine (`create --from-engine`) gives its first candidate, the engine as imported, a
  small Eval: on the ladder candidate 1 had none, so the origin's own baseline went unmeasured.
- The run's brief (`agent_request.md`) holds the run's facts and the user's terms only (about 400 words, from about
  3,700) and points to AGENTS.md and `crit run ... next`. AGENTS.md is the build agent's rules only (about 3,500
  words, from 7,300), with its contradictions resolved: exactness is required only when the run says so, the
  baseline is the run's first Eval result, a qualified candidate is defined once (docs/RUN_RECORDS.md#states), and a
  resume goes on with a stopped Eval rather than measuring again. docs/BUILDING_ENGINES.md keeps the engineering
  details (about 2,350 words, from 5,200) and repeats no procedure.
- The run's `phase` (`state.json`) is the segment's state only; a candidate's lifecycle and the stop's reason live in
  their own fields. Older runs' phases read as `building` or `stopped`.
- AGENTS.md: a plan's levers rest on measurements where one can be taken first (`ablate --data --set`, cited as
  `measured:PATH`), and an estimate is never raised to pass a check; the plugin round keeps only plugins its ablation
  on Eval's data shows paying, neutral ones dropped, and a plugin option (a threshold, a width) is measured, not
  guessed. One run kept a copy-draft override at a threshold it picked without measuring, judged neutral on its own
  short case, at a 9% sampled decode cost on Eval's suite.
- BUILDING_ENGINES.md: every request kind (log-probabilities, replays, constraints, plain) decodes in the captured
  rounds, since Eval's long-context point asks for log-probabilities; and when draft steps dominate a round, the draft
  head is the lever (`draft-vocabulary-subset`, `draft-trees`), with `draft_bytes_per_step` declared for Eval's
  headroom.
- mtp-decode's README says why a short copy override replaces correct chain drafts in long contexts, and how to
  measure the threshold.
- crit smoke's `causal` and `causal_long` say what in the engine's facts makes a prompt's rows follow its length when
  they fail: prompt GEMMs whose `row_invariance` has `prompt_rows` false (quantized-projections' `prompt_gemm:
  pinned` keeps them invariant) and a recurrence whose prompts take its exact scan up to `exact_rows` tokens and its
  chunked kernel beyond. One run spent two draft candidates finding the two options by probe scripts.
- `crit_plugins.device.free_device_bytes()`: a placement plan's free device bytes after torch's allocator hands back
  what it caches for nothing. The example model plugins plan with it, and PLUGINS.md says why: a model plugin that
  re-encoded its head at load planned its KV pool against memory its temporaries still held and refused to load.
- `Rounds.calibrate()` times each captured round as the median of several spans and fits each row count's times as a
  line in the round's columns (`crit_plugins.rounds.fitted_costs`) before a width policy reads them. RateWidth chose
  from the raw times, so where two widths' rates nearly tied a load's timing noise swapped them: an engine opened at a
  lower context decoded the same text at other widths, up to 10% slower, and failed Eval's lower-context gate.
- Plugins load crit's own kernels (`crit_plugins.kernels.load_kernel`, a plugin.json kernel entry `{name, fallback}`),
  built for the GPU with the engine's tuning (`CRIT_KERNEL_TUNING`, a folder of the sweep's tables) or the default
  knobs, instead of registry kernels; the engine's facts say `tuned`, `built` or `fallback` per kernel. A kernel that
  cannot be built here (no CUDA toolkit, an architecture it does not cover) is reported once on stderr with the plugin
  that runs its fallback and what the build needs: an engine on a fallback looked merely slow.
- The workbench's `--starter` starts from crit's `kernels/starters/workbench-starter` without `--from`.
- `crit plugins test` reports a plugin whose tests passed before under the same key as `cached`, without running them
  again: the key covers the plugin's files, the plugins its tests use, crit's `crit_plugins` and `crit_core` and the
  environment (Python, torch, the device, `CRIT_KERNEL_TUNING`, transformers, fla, FlashInfer, Triton) and the
  source of the crit kernels the plugin declares;
  `--no-cache` runs them all. A first candidate's library plugins, copied unchanged, took 13 minutes of tests in one
  run and are now tested once per machine and crit version. Results are kept in `~/.cache/crit/plugin-tests` or
  `CRIT_PLUGIN_TEST_CACHE`.
- `crit mcp` serves every guide AGENTS.md links: `crit://docs/PLUGINS.md`, `crit://docs/CONTRIBUTING.md` and
  `crit://smoke/README.md` join the resources, and an installed crit carries them in `crit/library/guides`, so an
  agent without the checkout reads the plugin contract it builds an engine to.
- `crit run ... build` writes the candidate's `configuration.json` into the build folder beside its plugins (unless
  the source has one) before the recipe's command runs, so a build command that checks or reads the selection finds
  it; the package carries the same file.
- The ablation judges a plugin outside the drafter and verifier slots by the time per decode round in place of decode
  where both variants draft: a short bench's tokens per round move with what it samples, so its decode does too, by
  more than the plugin's own effect. Each plugin's row names the KPIs its verdict read (`judged`).
- The example model plugins of the GPU tests (`tests/support/device_plugins`): llama-model decodes a request asking
  for log-probabilities in rounds, its round driver capturing each round in the log-probability form too, and
  llama-model and next-model score a fixed text a prefill chunk at a time, each chunk's KV and recurrent state kept
  for the next, so a long score takes one chunk's temporaries.
- crit run from a checkout refuses a `crit_core` imported from elsewhere, such as the release on PyPI that its
  version pin also accepts, and names the install that fixes it: `python -m pip install -e core/python -e .`, as
  AGENTS.md and Contributing now say.
- CI lints with ruff's pyflakes rules (`[tool.ruff]` in pyproject.toml, kernels/ joining once its rework settles)
  and installs crit as AGENTS.md says in a fresh environment on Linux and Windows, then builds Core's runtime and
  checks the plugin library on it. The release publishes crit-core before crit, so a failed upload never leaves a
  crit whose pinned Core is missing. Twelve unused imports and variables in Inference and the tests are gone.

### Fixed

- Kernel core TMA loads target `.shared::cta` (CUDA 12.8 and newer). With `.shared::cluster` sm_120 compiled each
  load to a call, and the driver reserved 3.3 GB of device memory for the kernels' dynamic stack on their first
  launch: a Qwen3.8-27B engine no longer fit at 200k context.
- crit serve answers a JSON body nested too deeply with a 400 instead of dropping the connection, and `--host ::1`
  listens on IPv6 loopback instead of failing to bind.
- `crit run` and `crit pull` report a defect in crit (a missing key, a wrong type) with its type and the place it
  happened instead of a bare `ERROR: 'bytes'`, and print the traceback with `CRIT_DEBUG=1`. `crit pull --checkpoint`
  no longer fails on a drafter's rows that carry a hash and no size, or on a checkpoint pinned to a file without a
  revision.
- Core reads the compute capability of the GPU an engine runs on, the first one `CUDA_VISIBLE_DEVICES` names, instead
  of always the machine's GPU 0, when no probe file says it.
- `POST /v1/crit/abort` no longer finds a request it or a closed connection already cancelled. Such a request stays
  in flight until its handler has answered, and its client can have its 499 first, so `{"all": true}` sent right
  after could report it again.

### Security

- crit serve refuses what a web page in the user's browser could send it: on loopback a `Host` that is not loopback
  (DNS rebinding) and, without `--cors-origin` allowing it, any request carrying an `Origin`. A request's API key is
  checked before its body is read, `/health` tells a caller without a key only `status` and `ready`, and a
  connection idle for 120 seconds in a read or write is closed.
- A crit-hub card's `source.path` must be a relative folder in the engine's repository, and `crit pull` passes it to
  git after `--`, so a card cannot hand git an option such as `--output` or `--remote`.
- crit mcp's `crit_serve_status` asks only http(s) servers and sends an `api_key` only to this machine or a private
  network address, `crit_job_output` reads at most 60,000 characters a call, and a server without `--jobs` removes
  its temporary jobs folder when it closes. `crit_job_stop` says that on Windows it ends the tree at once.

## [0.17.2] - 2026-11-09

### Overview

crit 0.17.2 readies `crit mcp` for Claude's and Codex's plugin directories and the MCP Registry. Every tool carries
MCP annotations, so a client runs the read-only ones without asking and confirms the destructive ones;
`crit mcp --without TOOL` leaves out a tool, such as `crit_run_cost`, which reads an agent's transcripts; and an
installed crit serves every guide `crit mcp` names, AGENTS.md first, where it served only one. mcp/README.md says
what `crit mcp` runs, every host it reaches and its privacy terms, crit's PyPI page carries keywords, classifiers,
links and README images that show, and mcp/server.json is `crit mcp`'s MCP Registry entry. crit 0.17.2 needs
crit-eval 0.6.1.

### Added

- `crit mcp --without TOOL` leaves out one tool (repeatable), such as `crit_run_cost`, which reads an agent's
  transcript files.
- `crit mcp`'s tools carry MCP annotations (`title`, `readOnlyHint`, `destructiveHint`, `openWorldHint`), so a
  client runs the read-only ones without asking and confirms the destructive ones (stop, seal, select, recover and
  `crit_job_stop`).
- [mcp/README.md](mcp/README.md) says what `crit mcp` runs, every host it reaches, its privacy terms and example
  prompts.
- crit's PyPI page lists keywords, classifiers and its homepage, docs, changelog and issues links, and
  [mcp/server.json](mcp/server.json) is `crit mcp`'s entry for the MCP Registry (`io.github.crit-labs/crit`),
  which README.md's `mcp-name` line verifies.

### Fixed

- `crit kernels sweep` flushes the L2 cache with a write between samples: a read flush left the lines a call wrote
  back in L2, so a library that writes its state was timed warm (flashinfer's NVFP4 GEMM read 125 us against 69 us on
  its own). Asking it for a structure without a bench names the ones it can sweep.
- Plugin device tests read their plugins folder once a process: discovery on every assemble was most of their time on
  a slow file system (grouped-experts' tests 20 min to 25 s, sequential-recurrence's 11 min to 30 s).
- A crit installed from its wheel serves every guide `crit mcp` names (`crit://AGENTS.md` and the docs it links),
  which the wheel now carries in `crit/library/guides`. Before, it served only `crit://inference/README.md`, though
  its instructions ask the agent to read AGENTS.md first.
- crit's PyPI page shows README.md's images and follows its links: the README links them absolutely, to the
  repository's main branch. Before, PyPI showed broken images and dead links for every relative path.

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
  time or in an entry point of group `crit_core.gguf_config`; [config rules](inference/core/python/crit_core/model/README.md#gguf)).
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
  [chat templates](inference/core/python/crit_core/text/README.md#chat-templates)), and crit-eval gives DeepSeek V4's, byte for
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
- Core's CUDA backend gets device sets (`crit/cuda_devices.h`, [device sets](inference/core/src/backends/cuda/README.md#device-sets)):
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
  model on it ([step engines](inference/core/src/runtime/README.md#step-engines)). Core then schedules the model's requests in
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
  `core/src/runtime/runtime.cpp` became one file per part of the runtime ([request runtime](inference/core/src/runtime/README.md)).
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
- Inference entry [host-expert-arithmetic](inference/techniques/experts/host-expert-arithmetic/README.md): NVFP4 expert rows on the
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

- **Core** ([guide](inference/core/README.md)): a C++20 SDK with Model, Session and Backend
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

[Unreleased]: https://github.com/crit-labs/crit/compare/v0.22.0...HEAD
[0.22.0]: https://github.com/crit-labs/crit/compare/v0.21.0...v0.22.0
[0.21.0]: https://github.com/crit-labs/crit/compare/v0.20.0...v0.21.0
[0.20.0]: https://github.com/crit-labs/crit/compare/v0.19.0...v0.20.0
[0.19.0]: https://github.com/crit-labs/crit/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/crit-labs/crit/compare/v0.17.2...v0.18.0
[0.17.2]: https://github.com/crit-labs/crit/compare/v0.17.1...v0.17.2
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
