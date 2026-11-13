# Build a bespoke engine with crit

Build a complete engine for the user's LLM and hardware, with recorded candidate comparisons and a working
OpenAI-compatible API (`crit serve`). You own the work from investigation to delivery.

Two tools carry the procedure so this file does not have to:

- **`crit run ... next`** reads the run's records and names the one next step, its command and why. Run it after
  every step. crit refuses a step out of order and says what to do instead.
- **The run's brief** (`segments/SEGMENT/agent_request.md`) holds this run's facts and the user's terms.

This file is what crit cannot check for you: how to set up, what to build, and how to judge it. Reference:
[Run records](docs/RUN_RECORDS.md) (every command and payload, the [plan](docs/RUN_RECORDS.md#plan) rules, the
[candidate ladder](docs/RUN_RECORDS.md#the-candidate-ladder)), [Building engines](docs/BUILDING_ENGINES.md) (the
engineering details), [Plugins](docs/PLUGINS.md) and [Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md).
Changes to crit itself must meet [what belongs in crit](docs/CONTRIBUTION_CRITERIA.md) and follow
[Contributing](docs/CONTRIBUTING.md).

Install from the repo root with `python -m pip install -e core/python -e .` (the checkout's Core and its CLI; crit
refuses to run from a checkout whose Core comes from elsewhere), build Core's runtime once with `crit core build`
(CMake and a C++20 compiler), then run `crit ...`. A wheel install (`python -m pip install .`, or `pip install crit`
without a checkout) builds engines too, after the same `crit core build`, which fetches Core's source at the installed
release's tag: records go in the folder `--project` names (default the current one, holding `settings.toml` and
`runs/`), and the brief names the guides at the installed release's tag. A harness that speaks MCP runs the same commands as tools through `crit mcp`
([MCP](mcp/README.md)).

## 1. Set up the project

- **The user's terms are theirs.** Model and encoding, hardware, workload, quality criteria, targets, resource
  limits and the candidate budget come from the user; reuse their decisions, ask about essential gaps, state routine
  assumptions. Set these only when the user asks: `quality_mode` (`speed`, the default, or `preserve`;
  [quality modes](docs/RUN_RECORDS.md#quality-modes)), the context (`kv_tokens` or `context_tokens`), the sampling
  (`eval_sampling`), `weights_policy` and `exactness`. Without a user target, record one in `targets` from a
  speed-of-light estimate (section 3).
- **Create the run** with the user's request ([request](docs/RUN_RECORDS.md#request)) in the user's workspace
  (default `<crit-repo>/runs/`). Probe the device first (`crit eval probe`, or `device` in `settings.toml`): crit
  then sizes the context as Eval's max context, the user's wanted context capped by the model's positions and the KV
  cache that fits. Preflight the model, tools and dependencies before starting the clock. For a supplied run, read
  its brief and honor its scope; an expired run needs an authorized resume. Building an engine never includes
  publishing repository changes.
- **Keep work in the run.** Source, models, dependencies, builds and results go under the run's `work/`, never in
  product folders. Never alter existing projects or sealed candidates. Linux-only work under WSL goes in
  `wsl_run_root` from `settings.toml` (ask if unset), never in shared registries or home directories.
- **Start from a crit-hub engine when one exists for the model.** `create --from-engine ENGINE_DIR` records it as
  the run's origin ([origin engine](docs/RUN_RECORDS.md#origin-engine)). Its source at the pinned commit is
  candidate 1 (`"lineage": "origin-engine"`), unchanged; its small Eval is the run's baseline. When `resolve` says
  the model is another checkpoint, candidate 1 absorbs what `recon --compare` says differs.
- **Study the model before the first candidate.** `recon` reads `generation_config.json`, the tokenizer and chat
  templates and `config.json`, writes the facts to `work/recon.json` and scaffolds `work/recon.md`. Finish
  `recon.md` from the model card and the transformers reference implementation (norm and attention quirks, rope
  scaling, draft heads, known issues), citing sources. Where they override the files, edit `recon.json`'s `frozen`
  block and say why under Decisions; every candidate freezes it (sampling, chat template and its variables, stop
  tokens). A template's `enable_thinking` stays frozen off, as Eval renders prompts.
- **The sampling is the model's, not yours.** Eval's sampled suite and the served default use the model's
  `generation_config.json` (else the protocol default). Never pick a temperature, not even one the model card
  recommends; write card recommendations down. Only the user overrides it, with `eval_sampling`; then pass
  `crit eval prepare --sampling` exactly those settings, and never otherwise.
- **Name the placement scenario.** recon records where the weights fit at the required context: `fits_device`,
  `sparse_device_host` (routed experts beyond device memory fit host memory), `sparse_device_host_disk`,
  `dense_device_host` or `beyond_host`, with one plain pass's floor per expert source (device, link, CPU). Write
  recon.md's placement plan from the scenario's starting design: candidate 1 is a proof of concept of the scenario,
  not a placeholder below it. In `sparse_device_host` that means a device expert cache and the checkpoint's drafter
  from candidate 1 on, misses on the faster of the link and the CPU (the CPU expert tier,
  `inference:host-expert-tier`, where the probe prices it below the link), and a KV pool made with `spare` bytes for
  the cache's slots ([device-expert-cache](inference/device-expert-cache/README.md) has the recipe). In
  `dense_device_host` and `beyond_host`, tell the user what speed to expect before building.
- **Write recon.md's plans** before the first candidate: the drafting plan for both decode modes, the prefill
  arithmetic plan (each matrix group's prompt activation format and, in `preserve`, its quality measurement), and the
  placement plan. crit refuses a candidate while one is missing.

## 2. Use the toolkit

An engine is Core plus a plugin selection, and nothing else ([crit plugins](docs/PLUGINS.md)). Core knows no model
and no device: it runs requests, sessions, admission, the concurrent step scheduler, stop rules, sampling control,
chat templates and the serving and Eval contracts. Everything that depends on the model or the device is a plugin in
a slot. A candidate is a plugin package: `plugins/` and the configuration whose `plugins.use` fills every slot an
engine fills (`model`, `placement`, `projections`, `attention`, `kv_cache`, `sampler`, and every slot the model
plugin `requires`) with plugins that run on the device. There is no adapter, launch entry or engine library to write.
An external engine the user improves (vision A) is the one exception: it stays its own code and server, declared with
its launch command in `configuration.engine`, and `crit run` records, checks, measures and ablates it the same way
([external engines](docs/RUN_RECORDS.md#external-engines)).

- **Start from the selection.** Choose the `model` plugin for the checkpoint's architecture, then a plugin for each
  slot it calls, from crit's library and crit-hub (`crit plugins list --model MODEL --device cuda` lists what fits
  and the slots nothing fits; `crit plugins add`). Put your own device code inside a plugin where it pays, keep each
  plugin's tests against its reference passing (`crit plugins test`) and declare the devices it runs on. A model
  plugin you write tests on a small random checkpoint with every part the real one ships (its draft layers too).
  The library's host references are oracles and never serve.
- **The slots do the work.** Core's loader refuses an engine whose model plugin never called a computing slot or
  never asked the placement plugin where its tensors live: a slot's arithmetic lives in that slot's plugin.
- **Every engine batches.** The model plugin's `step(rows)` computes many requests' prompt chunks and decode rows at
  once and declares `max_concurrency` of 2 or more. A row's tokens never depend on the rows beside it: split decode
  rows at `crit_plugins.device.invariant_rows(projections)` and pass each prompt chunk to `matmul` with `prompt=True`,
  so a short prompt is not multiplied as decode rows ([row invariance](docs/PLUGINS.md#row-invariance)). It
  serves token log-probabilities with 20 alternatives, fixed-text scores and, when the selection drafts, plain
  decoding on request.
- **Rounds and prompt chunks in graphs.** Decode rounds go through Core's round driver (`crit_plugins.rounds`,
  [captured rounds](docs/PLUGINS.md#captured-rounds)) and short prompt chunks through Core's prompt graphs. A plugin
  that lacks its `rounds:` or `prompts:` fact makes the driver fall back to `step()` (correct, slower): check the
  driver's `problems` after loading. Requests that ask for log-probabilities, replays and constrained requests go
  through the rounds too; a drafting engine that sends them through `step()` measures one token a pass on Eval's
  long-context point.
- **[Inference](inference/README.md)** holds techniques (each a plugin of the same id) and their references; adapt,
  combine or fuse them, preserving correctness and ownership invariants. Nothing is mandatory. A technique's
  `operations.json` says what each device operation computes and how a kernel is checked against it.
- **[Kernels](kernels/README.md):** crit's kernels are templates in `kernels/techniques/<family>/<id>/`, built on this machine
  for its GPU; `crit kernels sweep ID --model MODEL` tunes each one's knobs (warps, tiles, splits, stages) on the
  model's shapes against the operation's reference. Sweep the kernels the selection's plugins load into the candidate's source
  (`--out tuning`; `--matrices` with the candidate's serve --check report tunes exactly the shapes it loaded) and name that folder in the configuration (`"kernel_tuning": "tuning"`), so the package carries it;
  `CRIT_KERNEL_TUNING` is for a quick try, and no package or launcher carries it. Pick each operation's kernel in [three levels](kernels/README.md#levels):
  (1) crit's kernel wherever one covers the operation, swept for the model; (2) only where crit has none, a
  [library's](kernels/README.md#libraries); (3) where neither fits, a kernel the engine writes. A crit kernel that trails its
  library yardstick is a lead for a better structure or knob, never a reason to swap in the library. Measure each
  kernel the engine adopts on its own shapes against its floor (`crit eval status --kernels`); a level 2 or 3 kernel
  that trails another provider at the same arithmetic is replaced. A kernel the engine writes stays in the engine; a
  structure that proves out beyond this run can reach its template later, under the contribution criteria. A kernel
  the engine writes gets a workbench folder (`crit kernels new`),
  checked and timed with `crit eval workbench`; its `bench.json` is the lever's `yardstick`. Only the run's criteria
  can exclude a provider, by naming it. A request to build from crit alone keeps the providers and excludes earlier
  results: past runs, earlier engines for the model and records outside this run.
- **[Models](models/README.md):** when `weights_policy` is `transformable` (the default in `speed`), quantize the
  source with `crit models ptq`, or re-encode stored 16-bit matrices as the engine loads them (declare `weights:
  "transformed"` and `configuration.re_encoding`; the only route for an already quantized checkpoint). Where memory
  or bandwidth bounds decode, measure re-encoding the 16-bit matrices before other levers. Draft-only weights take
  any encoding in any run, since the stored target verifies every token.
- **[Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md)** measures a candidate as a chat client
  gets it, through `crit run ... eval`; in-process telemetry (tokens per round, acceptance, kernels) comes from
  `crit eval ... --package DIR`. Use the same prepared data for every engine of one model.

Retain Core revision and build inputs, dependency identities and notices with the candidate's source. Keep
model-specific changes in the engine; record toolkit gaps in the summary.

## 3. Build and compare candidates

Work in rounds, one candidate per round. A candidate's number is its rung on the
[candidate ladder](docs/RUN_RECORDS.md#the-candidate-ladder), which sets how far its round goes and how far Eval
measures it:

| Candidate | Round | Eval |
| --- | --- | --- |
| 1 | Proof of concept of the placement scenario: builds and passes `serve --check` | None (an origin engine: small) |
| 2 | Plugin round: every library and crit-hub plugin that fits, kept only where `ablate --data EVAL_DATA` shows it pays | At most small (stage 1) |
| 3, 4 | Crit rounds: improve with crit's toolkit, each technique measured against its floor | At most medium (stage 2) |
| 5 on | Crit round, then critique rounds: the parent's two weakest terms improved beside other levers | Full, certified |

The segment's last budgeted candidate always takes the full Eval, and only the full Eval qualifies a candidate. A
candidate that fails to build or pass `serve --check` is a draft: seal it with what stopped it and build the fix
from the same plan, at no cost to the budget. So check a candidate fully before its Eval: an Eval of a broken engine
spends a candidate. `serve --check --data EVAL_DATA` is that check: the plugin tests, the serving checks, one
request at the full context (its slowest prompt step against crit serve's one-second cancel grace, the least free
device memory against the run's headroom), [crit smoke](smoke/README.md) at the run's context and at a quarter of it, and Eval's pre-check of the gates rounds
most often fail on. Fix what it fails before spending a round. `crit smoke --url` runs the smoke suite against any
served engine while you build.

Each round:

1. **Plan** the candidate before any code, as an ambitious one-shot from the physical limits and the last round:
   `crit run ... plan --draft` writes the skeleton with the facts crit computes (the parent's scores, limits,
   noise, gaps and the parent plan's outcome); fill it, `plan --check` it, then `plan --record` it. The
   [plan rules](docs/RUN_RECORDS.md#plan) say what a plan holds and what crit refuses. Include every lever that
   helps a goal without costing a higher priority; effort is never a reason to leave one out. A lever's change rests
   on a measurement wherever one can be taken first (`crit run ... ablate --data EVAL_DATA --set
   PLUGIN.OPTION=VALUE`, cited as `measured:PATH`). A forecast is an estimate: crit warns where it looks too low, and
   you never raise one to clear a warning. A plan proved wrong before its candidate exists is replaced by a plan that
   `supersedes` it.
2. **Build** the plan's levers as plugins: a lever that fills a slot is a plugin of the same id with its tests
   against the slot's base or the technique's reference; a lever inside one plugin is its new version or option;
   two techniques that only pay fused become one plugin filling both slots, tested against the separate pair. Start
   a slot's lever from its Inference technique (`crit plugins add`), then replace its host Python with device code
   and keep its tests passing. A lever still in development when the rest is ready is deferred, not waited for.
3. **Check and ablate.** `serve --check --data EVAL_DATA`, then `ablate --candidate ID --data EVAL_DATA`, which
   serves the candidate with each plugin switched off and alone on Eval's speed cases and held-out prompts: fix or
   drop what slows or breaks the engine, drop what is neutral, and measure every option a plugin is given (a
   threshold, a width, a buffer size) the same way, never from a guess. Without `--data`, the run's own case (often
   one short prompt) judges a drafting plugin, and one that loses on long prompts or other languages looks neutral.
4. **Evaluate** on the rungs that have an Eval, with `--baseline` set to the run's first Eval result, then
   `crit eval kernels` and `crit eval status RESULT --baseline PARENT_RESULT --kernels KERNELS`. Read every report
   status gives before the next plan ([Eval's reports](docs/BUILDING_ENGINES.md#run-the-common-eval-suite)). A failed
   gate, an exactness drop or a repeat difference is a bug and the next candidate's first item. Every lever gets a
   fate in the next proposal: built, pruned (only after measuring it against its floor) or deferred.
5. **Revisit recon** when a result contradicts it, or when the engine switches behaviour at a threshold Eval does
   not cross: update `recon.md` and add a run case that crosses it.
6. **Repeat** until the targets are met, the candidate budget is used, neither decode nor prefill is needed, or the
   run is blocked. When neither is needed with candidates left, record a stop plan from the newest Eval and stop with
   `search_exhausted`.

How to judge a round:

- **Eval sets the goals; the priority rule orders them.** `crit eval status` gives one goal per KPI (decode,
  prefill, time to first token, and long-context decode and time to first token), each from its score to its
  physical limit with its noise. The headline decode is the sampled one. Decode and prefill are each *needed* while
  a lever can still raise it by more than its noise; the needed one further from its limit is p1, the other p2, both
  p1 when within noise of each other. Time to first token is never p1 and never the reason for a candidate on its
  own. crit computes the priorities in each plan ([plan](docs/RUN_RECORDS.md#plan)). Priorities decide conflicts and
  order, never what is left out. Never weigh by the end-to-end score.
- **The limit is the ceiling, not the engine's own speed.** Decode = tokens per round / round time. A pass cannot
  beat bytes read / bandwidth (weights plus KV and recurrent state); a draft step has its own bytes floor. Prefill is
  bound by FLOPs at the GEMM rate per weight format, plus attention. Time to first token is at least the larger of
  the compute floor and one weight read. Bytes from host memory, the link or the disk have their own rates, priced
  by recon's placement and by Eval. Report the expert cache's `expert_reads` and `expert_hits` from candidate 1 where
  weights live off the device, and `kv_bytes_per_token`; without them Eval's floors are optimistic and named so.
  `crit eval probe` measures bandwidth and GEMM rates.
- **Strategy first.** Working baseline, measured bottlenecks, then compare strategies, not only settings. Reach the
  highest-ceiling strategy in working form before tuning kernels: tuning does not transfer between strategies.
- **A term is exhausted only when measured.** A kernel term is done when it runs near its floor at the arithmetic
  the engine already uses (each matmul's achieved TFLOPS at its own prompt activations), or when another kernel
  structure or a vendor library at the same arithmetic was measured on it. Tiling, block and launch variants of one
  kernel reject those variants, not the term. A kernel's share of its phase bounds what replacing it can gain
  (share x (1 - 1/speedup)): size kernel work by that bound. An isolated benchmark is a lead; measure the integrated
  engine before building on it.
- **Build for both decode modes.** Eval scores both on the same speed cases. Greedy: speculative output equals plain
  greedy when the run requires `exactness`; by default (`reported`) the drift budget judges it, and in `speed` Eval
  judges each output fork. Sampled, at the model's sampling, the headline: verify drafts by rejection sampling
  (accept with probability min(1, p/q), resample from max(p - q, 0) on the first rejection; a greedy draft is q
  one-hot) and sample on the device inside the captured round ([token-sampling](inference/token-sampling/README.md)'s
  `rounds:sampler`; its host oracle is [mtp-decode](inference/mtp-decode/README.md)). A drafter that only helps
  greedy requests leaves the headline at one token per pass. A model plugin that drafts also decodes plainly on
  request (`chat-plain-v1`).
- **The context is a requirement.** Build every candidate for `kv_tokens`. Beyond weights and KV the engine gets
  `context_reserve_gib` (default 3 GiB) for activations, workspaces, graphs and speed techniques. Speed depends on
  the tokens held, never the capacity: Eval fails a candidate that is slower opened at a lower context. A prompt's
  rows never depend on its length: pin the prompt GEMMs' kernels
  ([recorded selection](inference/recorded-selection/README.md)), or smoke's `causal` fails it.
- **Quality is a budget to spend.** Spend `quality_budget` on speed; drift below it is no gain. In `speed`, the
  stored format's activations on quantized matrices, FP8 or 4-bit KV, an FP8 head and (where `weights_policy`
  allows) re-encoded dense weights are the starting arithmetic, judged together by the Eval, not levers to justify
  one by one. In `preserve`, measure each precision change per group of layers before building on it: `crit eval run
  --package P --only quality --quality-mode preserve --attribute-drift` with the model plugin's `exact_variants`.
  A difference inside the quality spread (`budget_spread`) is no measurement.
- **Output differences are bugs until explained.** A drop in plain-greedy equality against an earlier candidate,
  or a difference between repeated identical requests (Eval's `repeatable` and `request_history` gates), is a bug
  until its root cause is demonstrated: the differing kernel or state located and the difference reproduced. Never
  blame rounding, FP8 KV or accumulation order without that. In `speed`, Eval judges each difference itself
  ([forks](https://github.com/crit-labs/crit-eval/blob/main/docs/MEASUREMENTS.md#forks-in-quality-mode-speed)): one
  that parts at a near-tie needs no root cause; one Eval calls decisive is a bug as above.
- **Never fit the engine to Eval's suite.** Never size or choose anything content-dependent (vocabulary or draft-head
  subsets, draft tables, n-gram or prompt caches, acceptance thresholds, routing) by its Eval score or from Eval's
  prompts, pinned continuation or held-out prompts, and never special-case Eval's token ids, lengths or cases.
  Hardware and shape tuning is fine for every length: Eval draws held-out and long-context lengths per run. A
  content-dependent choice needs a general corpus unrelated to Eval and a held-out check in several languages and
  domains, both stated in the proposal; a residency profile from `device-expert-cache`'s `corpus.json` meets this.
- **Never train or fine-tune a drafter.** Use the checkpoint's drafter, a published companion drafter as released,
  or one the user produced with [`crit models drafter`](models/README.md#drafters) and handed you with its transform
  record (retain it, name its `output.manifest_sha256` in the configuration).
- **Long jobs.** Downloads, `crit eval reference`, Evals and builds take minutes to hours. Run each as a background
  command your harness tracks and wakes you on (never a detached `nohup`, `setsid` or `wsl ... &`, never a sleep
  loop), and never end a turn without such a wake pending. While one holds the GPU, do the next round's CPU work. An
  Eval that stopped goes on with `crit run ... eval --resume`, never from the start. A session restart ends the
  jobs it started: when you resume, read what each left (its log, its outputs) before anything else, and run again
  what died.

## 4. Deliver a served engine

Include serving unless the user asked for less. crit serve runs the plugin package through Core's loader and Core's
resident adapter ([serving](docs/SERVING.md#cores-resident-adapter)), which brings the chat template, tools,
per-request sampling, streaming, log-probabilities, scores and concurrent requests for every engine. The plugins load
weights once; keep model allocations across requests and switches through the serving context's device leases
(`context.serving`); keep request state private to a session; honor cancellation and cleanup through Core's
decisions (`commit`, `retire`). A cancel must settle within crit serve's fixed one-second grace, or the host is
retired with its weights: size prompt chunks so the slowest, at the full context, finishes well inside it. After sealing, verify candidate switching with `crit serve`: serve the selected
candidate, switch to another sealed one and back, and compare a greedy reply before and after. Report anything
unverified.

## 5. Critique and hand off

Review the engine against the physical limits; a working engine or a speedup alone is not enough. Select the best
qualified candidate ([states](docs/RUN_RECORDS.md#states)) and say whether it beats the baseline; apply the agreed
criteria without relaxing them. Publish reason, critique, opportunities and cleanup with `crit run ... stop`: its
`headroom` gives each goal's start, reached and limit, what you built and the physical cause of the remaining
distance. Effort, time and candidate count are not causes.

Write `summary.md` at the run root and tell the user: target and workload; baseline versus recommended candidate;
gains and tradeoffs; checks and limitations; techniques and kernels; Eval results and missing coverage; remaining
work; source and package paths. Say when none qualified. Put toolkit feedback here, not into Core or Inference.
Record the cost with `crit run ... cost --transcript <your session transcript> --agent "<agent and model>"`, and name
with `--idle FROM..UNTIL` each stretch your session was offline with nothing running (a restart, a lost
connection), so the record gives the active hours beside the wall hours.

Always give the actual candidate and its serve command, and run it once with `--write-launchers` (it writes
`serve.cmd` and `serve.sh` into the run folder):

```text
crit serve --workspace "RUN_ROOT" --run "RUN_ID" --candidate "CANDIDATE_ID" --project "CRIT_ROOT"
```

Clients connect to `http://127.0.0.1:8000/v1`; Anthropic and `/api/chat` clients use `http://127.0.0.1:8000`
([other API dialects](docs/SERVING.md#other-api-dialects)), and `GET /metrics` serves Prometheus metrics. Without
the target hardware, label results unmeasured; never invent measurements or claim to have started a server on the
user's machine.
