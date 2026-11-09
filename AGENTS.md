# Build a bespoke engine with crit

Build a complete engine for the user's LLM and hardware, with recorded candidate
comparisons and a working OpenAI-compatible API (`crit serve`). You own the work from
investigation to delivery. Changes to crit itself must meet [what belongs in crit](docs/CONTRIBUTION_CRITERIA.md)
(general interfaces for any model and any CUDA hardware, never one model's or machine's code) and follow
[Contributing](docs/CONTRIBUTING.md).
Install from the repo root with `python -m pip install -e .`, then run `crit ...`. A plain `python -m pip install .`
builds engines too: the wheel carries the plugin library, so `crit plugins` and `crit inference` work from it as from
the checkout, and `crit run` keeps its records against the checkout (`--project`). An installed crit without a
checkout (`pip install crit`) keeps them in any folder (`--project`, default the current one, holding `settings.toml`
and `runs/`), and the run's brief names its guides: these files at the installed release's tag of crit's repository.
A harness that speaks MCP can run the same commands as tools through `crit mcp` ([MCP](mcp/README.md)), the long
ones as jobs it waits on.

## 1. Set up the project

- Establish model/encoding, hardware/OS, workload, quality criteria, performance
  objective, resource limits, attempt bounds and paid allowances. Reuse user
  decisions, ask about essential gaps, state routine assumptions. Without a user
  target, record one in `targets` from a speed-of-light estimate (section 3) before
  the first candidate.
- The quality mode is the user's too: `speed` (the default) or `preserve`
  ([quality modes](docs/RUN_RECORDS.md#quality-modes); section 3, Compare and select). Set
  `quality_mode` only when the user asks; every run record and Eval result states its mode, and
  one that does not is refused.
- Probe the device (`crit eval probe`) and pass it as the request's `device` (or set
  `device` in `<crit-repo>/settings.toml`); `kv_tokens` then becomes Eval's max
  context for the model on it: the user's wanted context (`context_tokens` in
  `settings.toml`, default 200,000), capped by the model's positions and the KV cache
  that fits. The context is the user's, like the sampling: set `kv_tokens` or
  `context_tokens` only when the user asks. Preflight model, tools and dependencies
  before starting the clock.
- **Study the model before the first candidate.** After `create`, run
  `crit run --id RUN_ID recon` (`candidate` refuses a model folder without it): it reads `generation_config.json` (sampling defaults),
  the tokenizer files and chat templates (BOS/EOS and stop tokens, template switches
  such as `enable_thinking`) and `config.json` (rope scaling, MTP or draft heads,
  attention layout; for a lone GGUF, the files its metadata derives), writes the facts with file hashes to `work/recon.json` and
  scaffolds `work/recon.md`. Finish `recon.md` from the model card (recommended
  sampling per mode, e.g. thinking vs non-thinking, known issues, context limits) and
  the reference implementation in transformers (norm and attention quirks),
  citing sources. Where they override the files, edit the `frozen` block of
  `recon.json` and say why under Decisions. Every later candidate freezes that block
  (sampling, chat template and its variables, stop tokens) as `configuration["recon"]`
  instead of defaulting to greedy. A template's `enable_thinking` is frozen off, as crit
  Eval renders prompts; keep it so. `recon.json` also fingerprints the checkpoint (architecture,
  weight layout, tokenizer, template), and every candidate records it; when the engine should
  run a related checkpoint (a finetune, a re-quantization, a sibling model), `crit run --id RUN_ID
  recon --compare OTHER_MODEL` says what differs for it, and an equal fingerprint means the weight
  values differ only.
- **Name the placement scenario before the first candidate.** With the request's `device`, recon
  records in `recon.json` (`placement`) and `recon.md` where the weights fit at the required context:
  `fits_device` (1), `sparse_device_host` (2: routed experts beyond device memory fit host memory),
  `sparse_device_host_disk` (3: the rest of them on disk), `dense_device_host` (5: weights read on every
  pass exceed device memory) or `beyond_host`; lookup tables (gathered embedding and n-gram rows) are a
  modifier on any of them and go wherever room is left, down to the disk. It also gives one plain pass's
  floor per expert source (on the device, misses over the host link, misses computed on the CPU from host
  memory) from the probe's rates. Write recon.md's `Placement plan` from the scenario's starting design:
  candidate 1 is a proof of concept of the scenario, not a placeholder below it. In `sparse_device_host`
  that means a device expert cache and the checkpoint's drafter from candidate 1 on, with misses on the
  faster of the link and the CPU; an engine that copies every expert over the link on every pass measures
  neither lever. Start from the library: grouped-experts in the experts slot, device-expert-cache's
  `cache()` (its `experts:cache` device form) over a bundle store from the expert_source's
  `open_store()`, and a KV pool made with `spare` bytes for the cache's own slots so it borrows the
  pool's unused pages ([plugins](docs/PLUGINS.md#the-model-plugin); device-expert-cache's README has
  the recipe). Prompt chunks stream their experts through the cache's prompt ring rather than
  re-sending experts for every small chunk (`expert-streamed-prefill` measures the grouped
  projection's rate knee for the chunk size). In `dense_device_host` and `beyond_host`, tell the
  user what speed to expect before building.
- **Start from a crit-hub engine when one exists for the model.** `create --from-engine ENGINE_DIR`
  (the engine's folder in a crit-hub checkout) records it as the run's origin: its source pin, the
  checkpoint it was measured on and its published Eval results under `captured/origin/`, each with
  the quality mode it was judged in (one judged in another mode than the run's does not compare
  with the run's Evals: the first candidate's Eval is the baseline). Check the
  source out at the pinned commit and make it the first candidate, with `"lineage": "origin-engine"`
  in the proposal, built and measured before any change: it is the baseline. `resolve` records
  whether your model is that exact checkpoint; if not, the first candidate absorbs what
  `recon --compare` says differs.
- **The sampling is the model's, not yours.** Recon records the sampling Eval measures under
  "Eval sampling" in `recon.md`: the model's `generation_config.json` (temperature, top_p,
  top_k, min_p when it samples by default), else the protocol default (temperature 1.0, no
  filters). The served default (`frozen.sampling`) follows it, and a candidate refuses a frozen
  sampling that differs. Never pick a temperature, not even one the model card recommends for a
  mode; write card recommendations down, but only the user overrides the sampling, with
  `eval_sampling` at run create. Pass `crit eval prepare --sampling` only when the run declares
  `eval_sampling`, and then exactly its settings. Write the drafting plan in `recon.md` for
  these settings before the first candidate.
- **Build for both decode modes.** Eval scores both, on the same six speed cases.
  Greedy (temperature 0): speculative output must stay token-exact against plain greedy
  (the exactness check: an engine that drafts also decodes one token per pass on request, `chat-plain-v1`, which
  Core requires of its model plugin). Sampled, at the Eval sampling: the headline decode, its goal and
  the stop record's decode; verify drafts by rejection sampling (accept a draft with
  probability min(1, p/q) under the processed target distribution p, sample from
  max(p - q, 0) on the first rejection; a greedy draft is q one-hot) so the output keeps the
  model's distribution (the `sampling` gate), and sample on the device inside the captured
  round rather than copying logits to the host for every token: Core's round driver
  ([captured rounds](docs/PLUGINS.md#captured-rounds), `crit_plugins.rounds`) runs a model
  plugin's rows at once over its draft, verify, accept and commit, with
  [token-sampling](inference/token-sampling/README.md)'s `rounds:sampler` drafting from the
  request's own law and predicting every verify column by rejection sampling; its host oracle is
  [mtp-decode](inference/mtp-decode/README.md). A drafter that only helps
  greedy requests leaves the headline at one token per pass. Requests at other settings
  must stay correct too.
- Create the run per [Building engines](docs/BUILDING_ENGINES.md) in the user's
  workspace (default `<crit-repo>/runs/`). Keep source, models, dependencies, builds
  and results out of product folders; develop in the run's `work/`. Never alter
  existing projects or sealed candidates. Linux-only work under WSL goes in
  `wsl_run_root` from `settings.toml` (ask if unset), never in shared registries or
  home directories.
- For a supplied run, read the request and the current segment's `agent_request.md`
  and honor its scope and bounds; an expired run needs an authorized resume. Building
  an engine does not include publishing repository changes.

## 2. Use the toolkit

An engine is Core plus a plugin selection, and nothing else ([crit plugins](docs/PLUGINS.md)). Core knows no
model and no device: it runs requests, sessions, admission, the concurrent step scheduler, stop rules, sampling
control, chat templates and the serving and Eval contracts. Everything that depends on the model or the device is a
plugin in a slot. A candidate is a plugin package: `plugins/` and the configuration whose `plugins.use` fills every
slot an engine fills (`model`, `placement`, `projections`, `attention`, `kv_cache`, `sampler`, and every slot the
model plugin `requires`) with plugins that run on the device. Core's loader assembles it and runs the model plugin's
batched step on Core's concurrent runtime; `crit run candidate` and `package` refuse anything else, and crit serve
and Eval run nothing of a package but its plugins. There is no adapter, launch entry or engine library to write.

- **Start from the selection.** Choose the `model` plugin for the checkpoint's architecture (its `needs` name the
  `model_type` it implements), then a plugin for each slot it calls. Take them from crit's library and crit-hub
  (`crit plugins list --model MODEL --device cuda` lists, per slot, the plugins that fit the checkpoint and the GPU
  and the slots nothing fits; `crit plugins add`), and put your own device code inside a plugin where it pays: a kernel,
  a fused pair of slots, a model family's forward. Keep each plugin's tests against its reference passing
  (`crit plugins test`) and declare the devices it runs on; the library's host references are oracles and never
  serve.
- **The slots do the work.** Core's loader runs one request on every new engine and refuses it when the model
  plugin never called a computing slot's plugin, or never asked the placement plugin where its tensors live: write
  a slot's arithmetic inside that slot's plugin, never inline in the model plugin.
- **Every engine batches.** The model plugin is Core's step model: `step(rows)` computes the rows of many requests
  at once, prompt chunks and decode rows together, and its info declares `max_concurrency` of 2 or more. A row's
  tokens never depend on the rows beside it: split decode rows at `crit_plugins.device.invariant_rows(projections)`
  ([row invariance](docs/PLUGINS.md#row-invariance)). It also declares what every engine serves: token
  log-probabilities with 20 alternatives, fixed-text scores (`scoring`, `query_scoring`, `replay`), Core's sampling
  fields and, when the selection drafts, plain decoding on request.
- **Rounds and prompt chunks in graphs.** Decode rounds go through Core's round driver (`crit_plugins.rounds.Rounds`,
  [captured rounds](docs/PLUGINS.md#captured-rounds)) over the library's device forms, and short prompt chunks
  through Core's prompt graphs ([captured prompt chunks](docs/PLUGINS.md#captured-prompt-chunks)). A plugin of your
  selection that lacks its `rounds:<slot>` or `prompts:<slot>` fact, or gives a reason of its own
  (`round_problems()`), makes the driver refuse to capture, and the engine then runs those rows through `step()`
  (correct, slower): check the driver's `problems` after loading. A model plugin with draft layers provides
  `draft:head-device` and selects a drafter with `draft:device` (greedy-head-drafter for an MTP head,
  hidden-state-drafters for an EAGLE head with d2t, mtp-decode with `override` to put long copies over the head's
  chain), chain-verifier and token-sampling. In a mixture of experts select grouped-experts in the experts slot from
  candidate 1 (`activations: nvfp4` where the checkpoint's experts declare NVFP4 activations) and give it a residency
  cache's store and slot table through `forward_bundles` when the experts do not fit.
- **[Core](core/README.md):** Core's runtime, batched steps, state cache, sampling and model files, which the
  plugins build on.
- **[Inference](inference/README.md):** read relevant techniques and their code; adapt,
  combine or fuse them as plugins, preserving correctness and ownership invariants. Nothing is mandatory.
  Entries are techniques, not compute kernels: crit hosts none
  ([kernels and plugins](docs/PLUGINS.md#kernels-and-plugins)). A technique's `operations.json` says what each device
  operation computes and how a kernel is checked against its reference; kernels come from the providers in Kernels
  and the kernel registry.
- **[Kernels](kernels/README.md):** search the Hugging Face Kernel Hub and the other providers
  in [SOURCES.md](kernels/SOURCES.md) by operation before writing optimized arithmetic. A kernel a library plugin of
  crit needs and no provider has is a crit-labs registry kernel: the plugin declares it in plugin.json `kernels`,
  loads it through `crit_plugins.kernels.registry_kernel` and keeps a torch fallback; an engine's own kernel stays in
  the engine's plugin. Measure
  each kernel the engine adopts on its own shapes against its floor (`crit eval status --kernels`)
  and against the best other provider at the same arithmetic; where it trails, replace it.
  A kernel the engine writes or adapts gets a workbench folder (`crit kernels new`) and is checked
  and timed with `crit eval workbench`; its `bench.json` is the `yardstick` evidence of the lever.
  These providers are part of crit's toolkit: only the run's criteria can exclude one, and
  only by naming it. A request to build from crit alone (a clean test of crit) keeps them
  and excludes earlier results: past runs, earlier
  engines for the model and records outside this run. Check operations, formats, shapes,
  OS/device, licenses, dependencies and the three checks Kernels lists.
- **[Models](models/README.md):** only when the run's `weights_policy` is `transformable`:
  quantize the source with `crit models ptq`, measure the served output with
  `crit eval run --url` against the source's Eval data under the run's quality terms
  (`--quality-mode`, `--quality-budget`, `--exactness` as the brief gives them), bind the
  result with `crit models eval`, and keep both records with the candidate. In a `transformable` run an
  engine may instead re-encode stored 16-bit matrices into a lower format as it loads them
  (for example BF16 to FP8 per channel; the only route for a checkpoint that is already
  quantized, which `ptq` refuses): declare `weights: "transformed"` and
  `configuration.re_encoding` (matrix groups and their formats). Where memory or bandwidth
  bounds decode, measure re-encoding the 16-bit matrices before other levers: it frees device
  memory for caches and cuts the bytes of every pass. `weights_policy` says what the engine may
  do to the stored target weights, the weights that compute every published token; the quality
  mode, a separate choice, says how much drift it may trade for speed. It is the user's, like
  the context: set it only when the user asks; a `speed` run (the default) defaults to
  `transformable`. Draft-only weights are not covered in any run: a drafter's weights (the
  checkpoint's draft layers, a companion drafter, a copy of the head it drafts through) may be
  re-encoded whatever the policy, since the stored target verifies every token; declare
  `weights` by the target weights alone and record the drafter's encoding in the configuration.
- **[Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md):** measure each candidate whose rung has an Eval
  (section 3) with `crit run ... eval`: it
  scores the data's fixed quality texts on the model's checkpoint once (`crit eval
  reference`), serves the packaged candidate through `crit serve` and runs Eval over the
  chat API, quality through the engine's own score call. Use the same prepared data for
  every engine of one model. Hardware telemetry (tokens per round, acceptance, memory,
  kernels) comes from `crit run ... measure` and Eval's in-process commands with `--package DIR`, Core's
  eval-v1 adapter over the package.

Retain Core revision/build inputs, dependency identities/notices and adaptations with
candidate source. Keep model-specific changes in the engine; record toolkit gaps in the summary. Salvage
from a run reaches crit only as a general mechanism that meets [the criteria](docs/CONTRIBUTION_CRITERIA.md).

## 3. Build and compare candidates

Follow [Building engines](docs/BUILDING_ENGINES.md) and [Run records](docs/RUN_RECORDS.md).
Work in rounds, one candidate per round. Each candidate's number in the run is its rung on the
[candidate ladder](docs/RUN_RECORDS.md#the-candidate-ladder), which says how far its round goes and
how far Eval measures it; the plan's `inputs.objective` names it (`eval`):

| Candidate | Round | Eval |
| --- | --- | --- |
| 1 | Proof of concept: build it, pass its plugins' tests (`crit plugins test`) and `crit run ... serve --check`, seal it `built` | None: `crit run ... eval` refuses it |
| 2 | Plugin round: every plugin crit's library and crit-hub offer that fits the checkpoint and the GPU (`crit plugins list --model MODEL --device cuda`), in the slots it fills, each plugin's tests passing; `ablate` says which pays | At most small: Eval's stage 1 |
| 3, 4 | Crit rounds: as candidate 2, improved with crit's toolkit | At most medium: up to Eval's stage 2 |
| 5 | Crit round: as candidate 4 | Full: `crit run ... eval` certifies it |

The segment's last budgeted candidate gets the full Eval whatever its number, and candidates after the
fifth (a resume) are critique rounds with the full Eval. A candidate whose rung allows a small or medium
Eval may skip it: seal it `built` once its serve check passed. That puts it on the ladder, and the next
plan starts from it without an Eval to freeze. Only the full Eval qualifies a candidate, so the one you
propose is the last one, or one from candidate 5 on.

1. **Recon** (`crit run ... recon`, section 1) before the first candidate. Cite the
   findings that bear on the design (attention layout, draft heads, rope, length
   thresholds, known issues) in each proposal's `strategy` and `rationale`.
2. **Plan** each candidate as an ambitious one-shot from the physical limits and the
   last round: `crit eval status` goals, gaps and failed gates, the profile's heaviest
   kernels, the parent's result. Include every lever that helps a goal without costing
   a higher priority; effort is never a reason to leave something out. Record the plan
   with `crit run ... plan --record plan.json` after the parent's rung (its Eval, or its seal
   when its rung skipped Eval) and before writing code ([Run records](docs/RUN_RECORDS.md#plan)),
   and name its `plan_id` in the proposal; `plan --record plan.json --check` runs the same checks without recording.
   In a [replay](docs/RUN_RECORDS.md#replays) of a run, follow `replay/agent_request.md`:
   it ends at the checked plan and records nothing. A plan proved wrong before its candidate exists is replaced by a new plan
   with `supersedes`.
3. **Build** it (`crit run ... candidate`, then `crit run ... build`); fix or prune what is slow or fails a gate
   until the plan is implemented. Prune a part only after measuring it against its
   floor (Eval's floors per term, the profile's kernels). The proposal gives every lever
   of its plan a fate in `lever_fates`: built, pruned or deferred, with the reason for a
   pruned or deferred one. A lever still in development when the rest of the plan is ready
   is deferred, not waited for: build and measure the candidate without it, and let the
   next plan take it up. Build the engine from [plugins](docs/PLUGINS.md): a lever that fills a
   slot (attention, KV cache, drafter, ...) is a plugin of the same id in that slot, with its own
   tests against the slot's base or the technique's reference (`crit plugins test`), and the engine
   assembles them when it loads. A lever inside one plugin (a faster kernel, another format behind
   the same slot) is that plugin's new version or option, and one the engine is built on rather
   than plugs in (device-speculative-round and the other entries
   [crit's library](docs/PLUGINS.md#the-library) names as no plugin) is the engine's own code. Give
   each slot a base the engine serves, its own plain device path (often the parent's) or none: the
   library's base plugins are host references. Start a slot's lever from its Inference technique,
   which is a plugin of the same id
   ([crit's library](docs/PLUGINS.md#the-library)), when there is one: `crit plugins add
   ENGINE/plugins ID` copies it with its reference, tests and code, then replace its host Python with the device kernel and keep its
   tests passing against the reference it came with. Before the candidate's Eval, `crit run ... ablate --candidate ID` serves it with
   each plugin switched off and alone and says which pays, slows or breaks the engine: fix or drop
   those first, and give each plugin's lever its fate from it. Two techniques that only pay fused become one
   plugin that fills both slots, tested against the separate pair.
4. **Evaluate** on the rungs that have an Eval: `crit run ... eval --baseline FIRST_RESULT` (the
   run's first Eval result), `crit eval kernels`, then
   `crit eval status RESULT --baseline PARENT_RESULT --kernels KERNELS`. A failed gate, an exactness drop
   or a repeat difference is a bug and the next candidate's first item. Read every report
   status gives before planning: each prefill term's measured time against its floor and a
   round's verify pass against its draft steps (from the trace), the prefill ceilings by
   prompt activation format with the reachable limit quality evidence names, each case's time
   to first token split into the engine's own time and the rest, the quality share with its
   spread over positions, the sampled cases' standard errors, and the device memory left free
   at the required context. `crit run ... check --candidate ID --from-eval` records the run's
   arithmetic, functional and resources checks from the candidate's Eval result. Eval measures
   in stages within a 45-minute budget, as far as the candidate's own speed and its rung allow
   (stage 1 on candidate 2, up to stage 2 on candidates 3 and 4), and serves the required context
   itself only certified, which `crit run ... eval` does on the full rung: record the checks of
   the candidate you will propose from that result. Every rung runs `crit run ... serve --candidate
   ID --check`, candidate 1's test that the engine works. Before every Eval round, run it with
   `--data DATA`: beside the serving checks it runs
   [crit smoke](smoke/README.md) at the run's context and at a quarter of it (each engine path
   against another: prefixes, decode against prefill, replies against their scores, requests after
   other requests, long prompts) and asks, in minutes, the gates whole rounds most often fail on
   (honesty of short replies and of one long reply, the request history around them, device memory,
   the speed cases at a lower context). Fix what it fails before spending a round on it; `crit smoke
   --url` runs the suite alone against any served engine while you build. A prompt's rows never
   depend on its length: pin the prompt GEMMs' kernels
   ([recorded selection](inference/recorded-selection/README.md), whose pinned_gemm records and replays one
   cuBLASLt algorithm per shape). A kernel cuBLAS picks by row
   count moves a short prompt's rows, and smoke's `causal` fails it.
5. **Revisit recon** when a result contradicts it, or when the engine switches behaviour
   at a threshold (prefetch, sparse attention, chunk sizes) Eval does not cross: update
   `recon.md` (and the frozen block if needed) and add a run case that crosses it
   before the next candidate.
6. **Repeat** from step 2 until the targets are met, the candidate budget (a number of
   candidates on the ladder, not time; 5 by default) is used, neither decode nor prefill is needed or the
   run is blocked. A candidate spends the budget once it stands on its rung: when its first Eval starts,
   or, below the full Eval, when it is sealed `built` or `completed` with a passed `serve --check`. One that
   fails to build, package or pass `serve --check` before that is a draft. Seal it with what stopped it
   (`build_failed` or `check_failed`) and build the fix as a new candidate from the same `plan_id`, at no
   cost and on the same rung ([budget counts](docs/RUN_RECORDS.md#budget-counts)). So check a candidate
   fully before its Eval: an Eval round on a broken engine spends a candidate. When neither is needed the run stops even with candidates left: record a
   stop plan (`"decision": "stop"`) from the newest Eval, then stop with
   `search_exhausted`. On resume, re-measure first.

- **Long jobs.** Downloads, `crit eval reference`, `crit run ... eval` and builds take minutes to
  hours. Run each as a background command your harness tracks and wakes you on when it exits (never a
  detached `nohup`, `setsid` or `wsl ... &` it cannot see, and never a sleep loop), and never end a
  turn without such a wake pending. While one holds the GPU (Eval and the reference take the machine
  lock), do the CPU work of the next round: read the result so far, write the next plan's code, build
  it. A finished job nobody noticed is time the run loses. An Eval that stopped before its end (its
  session ended, an interrupt, a failure) resumes with `crit run ... eval --resume`; never measure the
  same package again from the start.
- **Strategy first.** Working baseline, measured bottlenecks, then compare strategies,
  not only settings. Reach the highest-ceiling strategy in working form before tuning
  kernels; sessions can end early and tuning does not transfer between strategies.
- **Eval sets the goals.** `crit eval status` gives one goal per KPI: decode, prefill,
  time to first token, and long-context decode and time to first token, each from
  current score to physical limit with its fraction of the limit and its noise. From
  crit-eval-v7 the decode goal is the sampled decode. Never weigh by the end-to-end score.
- **The priority rule orders them.** Decode and prefill are each *needed* while some
  lever can still raise it by more than its run-to-run noise; one stops being needed
  when the distance left is within the noise, or when every Eval gap on it above the
  noise has a physical cause or a measured rejection. Long-context decode counts toward
  decode and long-context time to first token toward prefill; a family's fraction of
  its limit is the lower of its members'. The needed one further from its limit is p1,
  the other p2; both are p1 when their fractions are within the noise of each other.
  Time to first token is p2 or p3: never p1 and never the reason for a candidate on its
  own. Priorities decide conflicts and order, never what is left out: every lever that
  costs no higher priority goes in. crit computes the priorities in each plan. When Eval
  gives decode or long-context decode no limit, crit computes one from the parent's Eval:
  the measured tokens per round over the pass floor (bytes per pass over bandwidth).
  Where it cannot, the plan states an estimated limit and how it was derived (for example a
  drafter's expected tokens per round over the pass floor). The current strategy's limit is
  Eval's, then crit's, then the estimate; crit ranks decode by the best strategy's ceiling,
  the higher of that limit and the alternative strategy's ceiling the plan gives in
  `strategy.alternative.limits` (its tokens per round and draft steps: crit computes the
  round floor), whichever strategy the plan chooses, and names the source of each fraction.
  In runs with `"plan_rules": 2` the alternative's tokens per round is at most its draft
  steps + 1; an unknown noise takes the largest noise the Eval knows (else 5%), a family's
  noise is the largest of its members', the two families' noise bands combine in quadrature
  for the near tie, and each gap and lever change is weighed against its own KPI's noise.
  There crit's computed limit and the alternative's ceiling follow Eval's cost model: tokens
  per round over the pass floor plus the draft steps at Eval's draft floor (where the draft
  width is unknown, the pass floor alone, a labelled upper bound). An alternative names the
  `drafter` it assumes (`measured` or `new`); its tokens per round stays within the measured
  drafter's acceptance bound at its draft steps (at most 8 or the parent's draft width) unless
  the plan cites another candidate's Eval attempt that measured the new drafter. The first
  plan's decision
  follows its needed flags like every other plan's. Near the limit (both families at 0.85 of
  it or above) a near tie puts only the lower one in p1, and a family marked not needed below
  0.85 answers why the rest of its limit is out of reach (`gaps` entry `limit`, answer `cause`).
  In runs with plan rules 3, a plan whose forecast gains more on p2 than on p1, or
  lets a p1 KPI fall, says why (`forecast_justification`). In runs with plan rules 4 (and
  5) each candidate has an objective by its number (`inputs.objective`), its rung on the ladder:
  1 the proof of concept, 2 the plugin round, 3 to 5 crit rounds (crit's toolkit, each technique
  measured against its floor and fixed or replaced where it trails), 6 on critique rounds that name
  the parent's two weakest terms in `weakest` and improve them beside other levers; candidates 2 to 5
  forecast at least +10% on a p1 KPI over the parent's Eval when the parent has one, except a gate
  repair (a requirement lever serving `gate:NAME` for a gate the parent's Eval failed, or `check:NAME` for a check
  crit run requires that the parent failed, such as `check:resources`). Every speed lever names its `provider` (`inference:`, `hub:`, `lib:` or
  `own`), and the engine's own code carries a measured `yardstick` against the best existing one. Each plan after the parent's Eval reviews the parent's plan in
  `review`: the cause and evidence of every KPI that missed its forecast, what each lever
  delivered, the lesson; the parent's largest p1 lever, pruned or deferred, is planned again or
  rejected on a measurement of another kernel structure or a vendor library. In runs with plan
  rules 5 (new runs) the strategy's ceiling is Eval's limit, not the engine's own speed: a
  forecast whose high stays below half of a needed family's limit is refused unless a `gaps`
  entry `limit` names, with its measured evidence, the bound that keeps the rest out of reach.
  A kernel term is exhausted only when it runs near its floor at the arithmetic the engine
  already uses (`crit eval status --kernels`: each matmul's achieved TFLOPS and its gap at the
  rate of its own prompt activations) or when another kernel structure, or a vendor library
  at the same arithmetic as a yardstick, was measured on it. Tiling, block and launch variants
  of one kernel reject those variants, not the term, and no justification leaves a p1 term on
  them alone.
  Below half its limit a family is marked not needed only on measured rejections: every gap
  entry names its evidence in the run, and for prefill every prompt activation format Eval
  prices faster than the engine's own is rejected by a quality measurement outside the quality
  noise (`gap` `activation:FORMAT`).
  Recon writes a prefill arithmetic plan beside the drafting plan: each matrix group's prompt
  activation format and, in `preserve`, its quality measurement (`--quality-mode preserve
  --attribute-drift`), before the first candidate. In `speed` runs (plan rules 6) every build plan declares
  `prompt_activations`, and no quantized format starts at BF16: candidate 1 reads prompts at its
  stored format's activations, and BF16 needs a measured rejection of them (`activation:stored`).
  In runs with plan rules 7 (new runs) a `limit` claim rests on the parent's round account
  (`round`: the measured round, the device's idle time and each term's time beside its floor) and
  names the terms that bind it (`cause_terms`); any other term of 10% of the round or more must
  run at half its floor's rate or better, and idle time counts as a term. A rejection names the
  term that bound it (`bound_by`) and is reopened once a later round shows that term 25% smaller.
  Where recon's placement prices CPU-computed misses below link copies, candidate 1 starts with the
  CPU expert tier (`inference:host-expert-tier`: it serves the experts slot on the CPU and is the CPU half
  heterogeneous-expert-dispatch joins inside a CUDA stream, for the slot -2 routes of the expert cache's
  admission; select heterogeneous-expert-dispatch as the experts plugin of a CUDA engine whose experts do not
  fit, and feed its `quota_table(tokens)` to the cache's admission) or names the measurement that rejects it.
  In `speed` the arithmetic its [quality mode](docs/RUN_RECORDS.md#quality-modes) starts from (the stored
  format's activations on quantized matrices, FP8 or 4-bit KV, an FP8 head and, where `weights_policy`
  allows, re-encoded dense weights) is no lever to justify group by group; the candidate's Eval judges it
  together.
- **Plans are checkable.** Each plan forecasts every KPI (low, expected, high), names the
  current strategy's ceiling and the best alternative's, groups levers that only pay
  together, and puts no lever below the noise into a candidate on its own. Work the rules
  require that Eval does not score (for example sampled speculation before Eval measures
  it) is a requirement lever (`"kind": "requirement"` with the rule it `serves`): it needs
  no projected change and never counts as the plan's lever for a KPI. After the
  candidate's Eval crit writes the plan's outcome next to the forecast; a drop in
  greedy decode, held-out decode or follow-up time to first token beyond its noise is
  flagged there, outside the priorities.
- **The context is a requirement.** Build every candidate for `kv_tokens` and pass it to
  Eval as `--context`. Beyond weights and KV cache the engine gets `context_reserve_gib`
  (default 3 GiB) for activations, workspaces, graphs and speed techniques; a device expert
  cache lends the prompt's buffers from its slots instead of holding that memory idle during
  decode (recon's placement plan sizes the cache so). Speed must
  depend on the tokens held, not the capacity: Eval fails a candidate that is slower
  when opened at a lower context.
- **Never fit the engine to Eval's suite.** Forbidden: sizing or choosing anything
  content-dependent (vocabulary or draft-head subsets, draft tables, n-gram or prompt
  caches, acceptance thresholds, per-content routing) by its Eval score; building such
  a choice from Eval's prompts, pinned continuation, held-out prompts or look-alike
  text; special-casing Eval's token ids, lengths or cases. Hardware/shape tuning (GEMM
  tables, tiles, launch geometry) is fine, for every length: Eval measures the held-out
  prompts and the long-context curve at lengths each run draws (80-100% of nominal), so a
  graph, tile or chunk fitted to one prompt length shows as a slower held-out score. A content-dependent choice needs a general
  corpus unrelated to Eval and a held-out check in several languages and domains, both
  stated in the proposal; when in doubt keep the general option (the full draft head). A residency
  profile built from `device-expert-cache`'s `corpus.json` and checked on its held-out sources meets
  this; name its `describe(...)` statement in the proposal.
  Pass `--baseline` (the run's first candidate) so Eval's generality gate applies, and
  check each candidate against its parent (`crit eval status RESULT --baseline PARENT_RESULT`).
- **Never train or fine-tune a drafter.** Use the checkpoint's drafter, a published
  companion draft model as released, or a drafter the user produced with
  [`crit models drafter`](models/README.md#drafters) and handed you with its transform
  record: retain that record with the candidate's source, name the drafter's
  `output.manifest_sha256` in the configuration, and verify every draft with the stored
  target. The drafter does not change `weights`, which the target weights alone decide.
  Re-encoding a drafter's weights or subsetting a companion's vocabulary is fine whatever the
  run's `weights_policy` (Models, section 2): record its encoding in the configuration.
- **Ceilings, in milliseconds.** Decode = tokens per round ÷ round time. A model pass
  cannot beat bytes read ÷ bandwidth (weights read plus KV and recurrent state). A draft
  step has its own bytes floor; acceptance at a generous width projects every smaller
  width, and draft width, head size, vocabulary and encoding must be tested together.
  Prefill is bound by FLOPs at the GEMM rate per weight format, plus attention. Time
  to first token is at least the larger of the compute floor and one weight read.
  `crit eval probe` measures bandwidth and GEMM rates; status reports each term's floor.
  Bytes that do not come from device memory have their own rates: host memory read (the CPU),
  host-to-device copies (the link), and the disk. Recon's placement prices one plain pass per expert
  source with them, and Eval prices the same tiers in its decode, prefill, first-token and
  long-context floors: the bytes the placement keeps off the device at host memory's tier, routed
  experts by the share a token activates at the device cache's hit rate. Where the placement keeps
  weights off the device, report that hit rate from the first candidate (`expert_reads` and
  `expert_hits` in the decode telemetry, or the expert cache's reads and hits in the engine's
  `telemetry()`; without them the misses beyond what the device's capacity forces are unmeasured and
  the floors optimistic, never an assumed rate), report `kv_bytes_per_token` (the kv_cache plugin's
  facts do; without it Eval leaves the KV term out), and declare `host_bytes_per_pass` (`{"link",
  "cpu"}` bytes per pass, in the engine's facts) where the engine's own path differs from the faster
  of the link and the CPU.
- **Profile** each candidate with `crit eval kernels`; for the heaviest kernels per
  phase far from their floor, check Kernels before replacing them (first the technique's
  `operations.json`: what the operation computes and how a kernel is verified against the
  technique's CPU reference on this GPU and checkpoint before it is trusted; the candidates come
  from kernels/SOURCES.md and the kernel registry), and fix kernels that leave SMs idle. A
  kernel's share of its phase bounds what replacing it can gain
  (share x (1 - 1/speedup)): size and order kernel work by that bound, and a lever whose
  bound is below its KPI's noise follows the plan rule for levers below the noise. A
  kernel's isolated or mimicked benchmark is a lead: measure the integrated engine
  (`crit eval kernels`, Eval) before building on it.
- **Measure** with [`crit eval`](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md) plus the run's cases
  (`stop_record_headroom` with `--baseline` seeds the stop record). Sweep settings on
  the mixed suite, not one prompt.
  Retain the applied configuration, data and results; report skipped
  cases. Eval results neither publish records nor qualify a candidate.
- **Output differences are bugs until explained.** A drop in plain-greedy equality versus
  an earlier candidate, or any output difference between repeated identical requests
  (Eval's `repeatable` and `request_history` gates), is a bug until a root cause is
  demonstrated: the differing kernel or state located and the difference reproduced.
  Never attribute it to rounding, FP8 KV or accumulation order without that evidence,
  even when exactness is only reported. In `speed` runs, Eval judges each difference
  itself ([forks](https://github.com/crit-labs/crit-eval/blob/main/docs/MEASUREMENTS.md#forks-in-quality-mode-speed)):
  one that parts at a near-tie is rounding by construction and needs no root cause, and
  neither does a drop in plain-greedy equality. A fork Eval calls decisive is a bug as above.
  So an engine in `speed` may let its arithmetic depend on what it served before: experts
  on the CPU or the device by residency, CPU rows in a faster order than the device's,
  kernels with atomics or splits chosen by the workload.
- **Record through `crit run`:** freeze inputs, create candidates, retain source and
  packages, publish each case's checks (`check --from-eval`), then benchmark with `crit run measure`, which
  streams each case's chat request to the candidate served through `crit serve` and refuses a candidate whose
  checks are not recorded or did not pass (a benchmark without them stays exploratory for good); seal
  (check telemetry's eligibility first: sealed candidates are immutable). Reserve
  before launch; settle results and cleanup.
  Record parent, changes, rationale, build, results, selection reasoning
  and technique decisions; keep failures visible. Changed source, configuration or
  dependencies need a new candidate.
- **Compare and select** on the same model and workload; explain missing metrics.
  Apply agreed criteria without relaxing them or demanding byte-identical output.
  Spend the `quality_budget` (defaults: perplexity ratio 1.4, mean KL 0.4, p99 KL 4.0 in `speed`; 1.05,
  0.05 and 0.5 in `preserve`; [Run records](docs/RUN_RECORDS.md#request)) on speed; drift below it is no
  gain. The run's `quality_mode` is the user's ([quality modes](docs/RUN_RECORDS.md#quality-modes)): never
  change it; the brief names it. In `speed`, the default, 4-bit or 8-bit activations on quantized matrices
  fit the budget with room to spare, and prefill is measured against the fastest prompt activations the
  mode admits on them. In `preserve`, measure a precision change (activation quantization,
  whatever format the checkpoint declares, a lower-precision KV cache or head) against the budget on the
  model, per group of layers, before building other levers on it: its cost depends on the model and
  decides how much of it fits. Do it in process during development:
  `crit eval run --package P --only quality --quality-mode preserve` (crit eval's own default is
  `speed`; development evidence, never the gate) with the model plugin declaring `prompt_activations` and `exact_variants`
  (one configuration override per matrix group or arithmetic choice switched back to exact),
  and `--attribute-drift` for each group's share of the budget. A difference inside the
  quality spread (`budget_spread`) is no measurement. Speculative output must equal plain greedy only when the run requires
  `exactness`. Failed required checks stay exploratory. Select the best qualified
  candidate and say whether it beats the baseline.

## 4. Deliver a served engine

Include serving unless the user asked for less. crit serve runs the plugin package through Core's loader and
Core's resident adapter ([serving](docs/SERVING.md#cores-resident-adapter)), which brings the chat template, tools,
per-request sampling, streaming, log-probabilities, scores and concurrent requests for every engine. The plugins
load weights once; keep model allocations across requests and switches through the serving context's device leases
(`context.serving`); keep request state private to a session; honor cancellation and cleanup through Core's
decisions (`commit`, `retire`). Before sealing, serve the package with
`crit run ... serve --candidate ID --check` (crit serve under a check attempt): it verifies consecutive requests,
a system message, streaming, a stop string, cancelling, a tool call and its result, log-probabilities, a template
switch and stopping, and records a functional check per case; after sealing, verify candidate switching with `crit serve`.
Report anything unverified.

## 5. Critique and hand off

Review the engine against the physical limits; a working engine or a speedup alone is
not enough. A segment ends when the loop in section 3 stops.

Publish reason, critique, opportunities and cleanup with `crit run ... stop`. Its
`headroom` covers decode, prefill, ttft (and context_decode, context_ttft when
measured): start, reached, limit, what you built and the physical cause of the
remaining distance. Effort, time and candidate count are not causes. When a resume
amends the quality terms, follow the agent request's terms.

Write `summary.md` at the run root and tell the user: target and workload; baseline
versus recommended candidate; gains and tradeoffs; checks and limitations; techniques
and kernels; Eval results and missing coverage; remaining work; source and package
paths. Say when none qualified. Put toolkit feedback here, not into Core or Inference.
Record what the run cost with `crit run ... cost --transcript <your session transcript> --agent "<agent and
model>"`: it attributes your model requests to the candidates built in their spans and writes `cost.json`
(wall time, tokens, API-equivalent price per candidate), which crit-hub shows on the engine's card.

Always give the actual candidate and its serve command:

```text
crit serve --workspace "RUN_ROOT" --run "RUN_ID" --candidate "CANDIDATE_ID" --project "CRIT_ROOT"
```

Clients connect to `http://127.0.0.1:8000/v1`; Anthropic and `/api/chat` clients (Claude Code,
local chat UIs) use `http://127.0.0.1:8000` ([other API dialects](docs/SERVING.md#other-api-dialects)),
and `GET /metrics` serves Prometheus metrics. Also run it with `--write-launchers`
(writes `serve.cmd` and `serve.sh` into the run folder) and name them. Headless agents
deliver transferable artifacts and target-machine instructions. Without the target
hardware, label results unmeasured; never invent measurements or claim to have started
a server on the user's machine.
