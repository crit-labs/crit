# Run records

A **run** fixes model, hardware, workload, correctness policy and objective; a
**segment** is a bounded execution window; a **candidate** freezes one engine's source
and configuration, and benchmarks evaluate that exact package. Failed gates leave
measurements exploratory. Write records only through the coordinator (sequence in
[Building engines](BUILDING_ENGINES.md); code in [records.py](../cli/crit/run/records.py)
and [store.py](../cli/crit/run/store.py)).

## CLI actions

```text
crit run --project CRIT_ROOT --root RUN_ROOT --id RUN_ID ACTION ...
```

Default root: `CRIT_ROOT/runs/`, outside product folders and protected retention.
Global options precede the action; `create` omits `--id`.

`CRIT_ROOT` is a crit checkout, and a crit running from one refuses any other `--project`. A crit installed without a
checkout (`pip install crit`) takes any folder instead, by default the one it runs in, for `settings.toml` and the
default root, and keeps runs outside the folder it is installed in. The run's brief (`agent_request.md`) names the
repository its agent follows: the checkout, or crit's repository at the installed release's tag (crit X.Y.Z is
published from `vX.Y.Z`) with the version and the folder crit is installed in.

Optional, untracked `CRIT_ROOT/settings.toml` holds machine settings that `create`
uses when the request omits them; unknown keys are an error. Copy
[settings.example.toml](../settings.example.toml) to `settings.toml`: it lists every key with its
default and a comment. In short:

```toml
wsl_run_root = "/srv/crit/runs"  # folder for work that must run under WSL; empty: none
device = "device.json"              # crit eval probe result (relative: from CRIT_ROOT)
ram_headroom_gib = 2                # host memory free at the engine's peak (default 2)
vram_headroom_gib = 1               # device memory free at the engine's peak (default 1)
context_reserve_gib = 3             # device memory kept out of the KV cache (default 3)
context_tokens = 200000             # the context wanted, prompt plus output (default 200,000)
eval_packages = "/srv/crit/packages" # reference package source, a folder or https URL; unset: none
```

`crit eval run` and `crit eval context` read `vram_headroom_gib`, `context_reserve_gib` and
`context_tokens` from it when their flags are omitted; `crit eval run`, `crit eval fetch` and
`crit run eval` fetch reference packages from `eval_packages`. The reserve holds activations, workspaces, graphs
and speed techniques; a larger one trades context for that room. crit names its project to crit-eval in
`CRIT_PROJECT` while a command runs Eval: `crit run` and `crit bench` their `--project`, `crit eval` crit's checkout
or, for a crit installed without one, the folder it runs in (unless the environment already names a project), so
Eval reads the same `settings.toml` as the run.

| Action | Arguments / effect |
| --- | --- |
| `create` | `--record request.json [--from-engine ENGINE_DIR]`; new run and first segment; with an engine folder of a crit-hub checkout, the run starts from that engine ([origin engine](#origin-engine)) |
| `resolve` | `--record inputs.json`; freezes verified model and input identities |
| `next` | Prints the run's one next step: where the run and its open candidate stand ([states](#states)), the command to run and why |
| `plan` | `--record plan.json [--check \| --draft]`; records the next candidate's plan after the parent stands on the [candidate ladder](#the-candidate-ladder) and before any code, freezes the parent's Eval and computes the priorities ([plan](#plan)); `--check` runs the same checks and prints the record, recording nothing; `--draft` writes the plan's skeleton to the `--record` file and prints the facts crit computes for it |
| `candidate` | `--source DIR --record proposal.json`; allocates an ID, retains source |
| `capture` | `[--case ID] [--text FILE] [--instruction TEXT] [--system TEXT] [--template-kwargs JSON] [--inputs INPUTS_JSON]`; before `resolve`, after `recon`: writes `captured/CASE/request.json`, a chat request whose messages render to the workload's `prompt_tokens` with recon's frozen `template_variables` under its own `chat_template_kwargs` (the given text or a neutral passage, cut to whole words and padded to the exact count), prints its manifest row and sets it in the inputs payload |
| `build` | `--candidate ID --record recipe.json [--seconds N] [--drop GLOB ...] [--no-package]`; reserves a build attempt, copies the retained source to `work/builds/ID-ATTEMPT`, runs the recipe's literal `build.command` there to the deadline (whole process tree; log `logs/build-ATTEMPT.log`), settles the attempt and, when it completed, packages that folder: each file the build added or changed bound as a dependency (`version` `built by ATTEMPT`) after the recipe's own; `--drop` removes intermediates first |
| `reserve` | `--operation discovery/build/check/benchmark [--candidate ID] [--seconds N]`; consumes an attempt before launch; only discovery omits the candidate |
| `settle` | `--reservation RUN_RELATIVE_PATH --record result.json`; exit, logs, cleanup |
| `package` | `--candidate ID --source DIR --record recipe.json`; runnable files and manifest |
| `check`, `benchmark` | `--candidate ID --record payload.json`; appends evidence. `check --candidate ID --from-eval [ATTEMPT]` instead records every case's arithmetic, functional and resources check from the candidate's retained Eval result (default its newest; [Check](#check)) |
| `measure` | `--candidate ID [--seconds N] [--exploratory]`; after `check --from-eval`: reserves a benchmark attempt, serves the packaged candidate through `crit serve`, streams every resident case's chat request to it, settles it and records one benchmark per case ([measured benchmarks](#measured-benchmarks)). Refuses, before reserving, a candidate whose checks would leave its benchmarks exploratory; `--exploratory` measures it anyway |
| `serve` | `--candidate ID [--host HOST] [--port 8000] [--seconds N] [--check [--data DATA]] [--no-smoke] [--no-plugin-tests]`; reserves a check attempt, serves the packaged, not yet sealed candidate through `crit serve` at `http://HOST:PORT/v1` until Ctrl+C or the reservation's deadline, for the checks a candidate passes before sealing, and settles the attempt (log `logs/serve-ATTEMPT.log`); with `--check` crit first runs the package's plugin tests (`crit plugins test`, cached; `--no-plugin-tests` leaves them out), then the standard serving checks itself (consecutive greedy requests, a system message, streaming, a stop string, a client cancelling, a tool call and its result, log-probabilities, a template switch, each where declared, and stopping the engine) and [crit smoke](../smoke/README.md) at the run's context and again with the engine reopened at a quarter of it (`--no-smoke` leaves it out), with `--data` crit-eval's gate pre-check, keeps their report as `logs/serve-check-ATTEMPT.json` and records a functional check per case: the one check before a candidate's Eval or seal; `crit serve` serves the candidate once it is sealed. An [external engine](#external-engines) is started from its launch command instead and has no plugin tests |
| `ablate` | `--candidate ID [--variant NAME ...] [--set ...] [--repetitions 3] [--no-smoke] [--smoke-test TEST ...] [--seconds N]`; reserves a check attempt, serves the packaged, not yet sealed candidate built from plugins through `crit serve`, builds it again in the same resident host with each variant of its plugins, benches the run's resident cases and runs crit smoke on each, and keeps the report as `logs/ablation-ATTEMPT.json` ([ablation](#ablation)); an [external engine](#external-engines) is served A/B on the options it declares only |
| `kernels` | `--candidate ID [--nvcc PATH] [--seconds N]`; an [external engine](#external-engines) only: reserves a check attempt, starts it with crit-eval's CUPTI tracer in its environment, sends each resident case once to warm and once traced, and keeps the kernels per phase as `logs/kernels-ATTEMPT.json`; where tracing is not possible (not Linux, no CUDA toolkit with CUPTI, no kernel recorded) the report says why. A plugin package is traced in process with `crit eval kernels --package` |
| `eval` | `--candidate ID --data EVAL_DATA_DIR [--only PART] [--baseline RESULT] [--label TEXT] [--seconds N] [--stage 1|2|3|full] [--certify] [--budget-minutes N] [--resume [ATTEMPT]]`; on the [candidate ladder](#the-candidate-ladder) measures as far as the candidate's rung allows and refuses candidate 1; reserves a benchmark attempt, serves the packaged candidate through `crit serve`, scores the data's quality texts on the run's model unless stored, runs Eval over the chat API with the run's quality terms and `kv_tokens`, settles it and retains the result as `candidates/ID/eval/ATTEMPT.json` beside its journal `ATTEMPT.journal.jsonl` ([Eval evidence](#eval-evidence)); for a candidate created from a plan it writes the plan's [outcome](#outcome), and it prints the priorities the result's fractions give. `--resume` goes on from an attempt that stopped (default the newest with a journal) in a new attempt |
| `seal` | `--candidate ID --lifecycle STATE [--failure TEXT]`; closes the candidate |
| `select` | `--candidate ID`; selects a sealed eligible candidate |
| `stop` | `--record stop.json` |
| `resume` | `--candidates N [--minutes N] [--attempt-limit N] [--external-agent-stopped]`; new segment for N more candidates on the ladder (`--minutes`: its safety window). `--quality-budget JSON`, `--exactness MODE`, `--approved-by NAME` amend quality terms from this segment on, recorded as the segment's `amendment` |
| `cost` | `--transcript PATH [--transcript PATH ...] [--agent NAME] [--prices JSON] [--until TIMESTAMP]`; accounts the run's build cost from the agent's transcripts and writes `cost.json` ([cost](#cost)) |
| `replay` | `--at CANDIDATE` or `--before PLAN`, `--out DIR [--count N] [--agent CMD]`; copies the run as it stood when the candidate's Eval finished (or just before the plan was recorded) into `DIR/replay-1` ... `DIR/replay-N`, for an agent to say what it would do next with today's crit and crit-eval; `--agent CMD` starts any harness on every replay at once and collects; `--collect` checks and lists the plans the replays wrote ([replays](#replays)) |
| `show`, `recover` | Inspect; or prove abandoned ownership and rebuild active projections |

## Payloads

CLI input objects. The coordinator adds record headers, owner IDs and generated
hashes; add them only where a field below requires one.

### Request

| Field | Meaning / default |
| --- | --- |
| `model_path` | Required existing local model file/directory |
| `correctness_criteria` | Required fixed criteria; gates are `arithmetic`, `functional`, `resources` |
| `device` | A `crit eval probe` result. Without `kv_tokens`, `create` freezes Eval's [max context](https://github.com/crit-labs/crit-eval/blob/main/docs/CONTEXT.md) on it as `kv_tokens`, with the basis (limits, KV precision, memory, device file sha256) in `hardware_constraints.context` |
| `kv_tokens` | Capacity for prompt + output; overrides the max context. One of `kv_tokens`, `device` is required |
| `context_tokens` | The context the user wants (default 200,000, from `settings.toml` when the request omits it). With `device`, `kv_tokens` is the smaller of it, the model's positions and the KV cache that fits; `hardware_constraints.context` records `requested`, the `bound` (`requested`, `model` or `device`) and a `note` when the model or device allow less |
| `ram_headroom_gib`, `vram_headroom_gib` | Physical free-memory requirements; defaults 2 and 1 |
| `context_reserve_gib` | Device memory kept out of the KV cache when deriving `kv_tokens` from `device`; default the protocol's 3 |
| `drafters` | Up to 8 companion drafter folders (EAGLE-3, DFlash, DSpark) a candidate may load beside the model. With `device`, their stored weights are kept out of the KV cache too, so a speculating candidate still fits `kv_tokens`; `hardware_constraints.context` records `companion_drafter_bytes` and each drafter. Name the same folders to `recon --drafter` |
| (devices) | A `device` file of several GPUs (`crit eval probe --devices`) sizes `kv_tokens` from their memory together, and `hardware_constraints.context.devices` names them (`index`, `device`); `crit run measure` samples device memory on every one of them |
| `candidate_budget`, `attempt_limit` | Segment bounds: candidates on the [candidate ladder](#the-candidate-ladder) (default 5, 1–64; [budget counts](#budget-counts)) and attempts (default 64, 1–1000). `budget_minutes` is a safety window, default and maximum 43200 (30 days) |
| `workload` | Defaults: prompt 512, output 128, concurrency 1, repetitions 3, warmups 1, mode `resident` |
| `cases` | Optional 1–32 distinct `{case_id, workload}` objects with complete workloads |
| `benchmark_protocol` | Fixed protocol identity; default `crit-matched-v1` |
| `verification_level` | `reported` or `artifact_verified` (default) |
| `targets` | Optional positive `decode_tps`, `prefill_tps`, `request_ms` thresholds |
| `weights_policy` | Optional: what the engine may do to the stored target weights, the weights that compute every published token, a choice separate from the quality mode: `as_stored` (used as stored) or `transformable` (re-encoded or transformed, [below](#inputs-proposal-and-package)); omitted means `transformable` when `quality_mode` is `speed` (the default), else unspecified (no declaration required, no transform announced). Draft-only weights are outside it in every run |
| `quality_budget` | Optional drift budget against Eval's reference: any of `ppl_ratio_max` (≥ 1), `kld_mean_max`, `kld_p99_max` (≥ 0); defaults Eval's for the run's quality mode: 1.4, 0.4, 4.0 in `speed`, 1.05, 0.05, 0.5 in `preserve`. Agents spend it on speed |
| `exactness` | `reported` (default: speculative output may differ from plain greedy; the drift budget judges it) or `required` (any difference fails Eval's gate) |
| `quality_mode` | `speed` (the default) or `preserve`: how much output drift from the checkpoint the user selected the engine may trade for speed ([quality modes](#quality-modes)). The user's choice, from the request or `settings.toml`'s `quality_mode`; `create` records it in every run, and a run record that states none, or another mode, is refused. `crit run ... eval` judges in it |
| `eval_sampling` | Optional, the user's only: `{settings, approved_by}`, settings `temperature` (> 0, required) and any of `top_p`, `top_k`, `min_p`, as an object or `"temperature=0.6,top_p=0.95"`. Eval's sampled suite then measures at these settings instead of the model's `generation_config.json`: `crit run ... eval` requires data prepared with `crit eval prepare --sampling` and the same settings (and refuses overridden data without it), recon applies them to the served default, and results label them as an override |
| `origin_engine` | Written by `create --from-engine`, never by hand: the crit-hub engine the run starts from ([origin engine](#origin-engine)) |

Every candidate names a recorded [plan](#plan) and gives each of its levers a fate, and a segment's budget counts
the candidates on the [candidate ladder](#the-candidate-ladder). Runs created by earlier crit releases recorded which
version of these rules they followed (`plans`, `plan_limits`, `lever_fates`, `plan_rules`, `budget_counts` in the
spec, and segments with a time budget only); crit still reads them to show, serve and replay such runs, and applies
the current rules to anything new.

#### Quality modes

A quality mode says how much output drift from the checkpoint the user selected the engine may trade for speed. It
says nothing about how that checkpoint is quantized or stored: a mode means the same for a 16-bit, an FP8 and an NVFP4
checkpoint. Both modes judge against the same reference, the checkpoint's stored weights decoded exactly with exact
activations ([Eval's quality](https://github.com/crit-labs/crit-eval/blob/main/docs/MEASUREMENTS.md#quality)).

- `speed`, the default everywhere (new runs, `settings.toml`, crit eval): a budget that only keeps out a broken engine
  (perplexity ratio 1.4, mean KL 0.4, p99 KL 4.0), output differences that part at near-ties allowed, and prefill's
  limits priced at the fastest prompt activations the mode admits on the quantized matrices (the stored format's own,
  FP8 or 16-bit) or a faster arithmetic quality shows passing. The stored format's activations on quantized matrices,
  FP8 or 4-bit KV, an FP8 head and, where `weights_policy` allows, re-encoded dense weights are its starting
  arithmetic, judged together by the candidate's Eval.
- `preserve`: a budget that keeps the output close to the checkpoint's (1.05, 0.05, 0.5), every precision change
  measured against it, and prefill's limits priced at 16-bit prompt activations.

Every run record, Eval result and crit-hub result states its mode, and one that states none, or another (the speed
mode's former name `as_quantized` among them), is refused. Results compare only within one mode. The mode is a choice
apart from `weights_policy`, which says what the engine may do to the stored target weights; a `speed` run defaults
to `transformable`.

#### Budget counts

A segment's `candidate_budget` counts the candidates on the [candidate ladder](#the-candidate-ladder), not
the candidates it creates. A candidate is on it once
the segment reserved its first Eval attempt (`crit run ... eval`), or, on a rung below the full Eval,
once it is sealed `built` or `completed` with a passed `crit run ... serve --check` (a functional check
that `serve --check` recorded with status `passed`). Any other candidate is a draft: its build, its
package or `serve --check` failed, or a fix was found before measuring it; on the full rung, a candidate
sealed without its Eval is one too. Seal it with what stopped it (`build_failed` or `check_failed` and
the failure) and build the fix as a new candidate that names the same `plan_id`: it costs no budget and
keeps the draft's objective and rung, and the segment's `attempt_limit` bounds its builds. Later Evals of a
counted candidate (a resume, a partial rerun) cost nothing more. Once the budget is used, an Eval of a candidate
not yet counted is refused, and so is `candidate`, until a resume.

#### The candidate ladder

A candidate's number in the run (the candidates on the ladder so far plus one; a draft's fix keeps its number)
is its rung, which says how far its round goes and how far Eval measures it. crit records the rung in the plan's
`inputs.objective`: the objective's name and rule, and `eval`, the rung's Eval (`scope`, `rule`, and `stage` for a
small or medium one). `crit run ... next` names the step each rung takes next.

| Candidate | Objective | Eval (`scope`) | `crit run ... eval` |
| --- | --- | --- | --- |
| 1 | `proof_of_concept`: builds and passes `serve --check` (its plugins' tests included), in its placement scenario's starting design | `none`; `small` for an [origin engine](#origin-engine) | refused: seal it `built`; an origin engine's stage 1 is the run's baseline |
| 2 | `plugin_round`: every plugin crit's library and crit-hub offer that fits the checkpoint and the GPU (`crit plugins list --model MODEL --device cuda`), in the slots it fills, each plugin's tests passing; `ablate` says which pays. For an [external engine](#external-engines) `technique_round`: every Inference technique and kernel template that fits, taken into the engine's own code as levers named by technique or kernel and declared as options, which `ablate --set` measures | `small`, optional | Eval's stage 1 (`--stage 1`) |
| 3, 4 | `crit_round` | `medium`, optional | up to Eval's stage 2 (`--stage 1` or 2) |
| 5 | `crit_round` | `full` | certified (`--certify`) |
| 6 on | `critique_round` | `full` | certified |

The segment's last budgeted candidate (the segment's candidates on the ladder plus one reach its
`candidate_budget` when the plan is recorded) gets the `full` Eval whatever its number, so a run with a
smaller budget still ends on a certified candidate: with `candidate_budget` 1 candidate 1 is certified, with 2
candidate 1 builds and passes `serve --check` without an Eval and candidate 2 is certified, and with 3 or 4 the
middle rungs' Evals are optional. A configuration that needs a certified number now takes a run of budget 1. `crit run ... eval` refuses `--certify`, `--stage
full` and a stage above the rung's on a small or medium rung, and `--stage` on the full one; it certifies
the full rung without `--certify`. A candidate that skips its small or medium Eval is sealed `built`
once `serve --check` passed, which puts it on the ladder. A plan starts from a parent on the ladder: after
its Eval, or, when it skipped its Eval, after its seal, and then freezes no Eval (`inputs.eval` null),
like the run's first plan; a parent that went through Eval, or stands on the full rung, needs a
completed Eval attempt. Only the full Eval measures the required context and the lower-context gate, so
the candidate a run proposes stands on the full rung.

#### States

`crit run ... next` reads the run's records (`crit.run.state`) and names the one next step and its command. Each
open candidate is in one state:

| State | Means | Next step |
| --- | --- | --- |
| `created` | source retained, not built | `build` |
| `build_failed` | its newest build failed | seal it `build_failed`; the fix is a new candidate from the same plan |
| `packaged` | built and packaged | `serve --check --data EVAL_DATA` |
| `check_failed` | its newest `serve --check` failed | seal it `check_failed`; the fix is a new candidate from the same plan |
| `serve_checked` | `serve --check` passed | rung 1: seal `built`; rungs 2 to 4: `eval`, or seal `built` to skip it; full rung: `eval` |
| `eval_stopped` | its newest Eval attempt stopped | `eval --resume` |
| `evaluated` | a completed Eval | `check --from-eval`, then `measure`, then seal `completed` |
| `running` | an attempt runs | wait, or settle it when its process is gone |

With every candidate sealed the next step is a candidate from a recorded plan, a new plan (`plan --draft`), or the
stop. A candidate **qualifies** when it is sealed `completed` on the full rung, its certified Eval passed the gates,
and its checks (`check --from-eval`) and every case's benchmark were recorded and passed: `select` takes only such a
candidate, and the run proposes one.

The run's own `phase` (`state.json`) is the segment's state only: `awaiting_agent`, `running`, `discovering`,
`building`, `finished`, `failed`, `launch_failed`, `expired` or `stopped`; the stop's reason is `stopping_reason`.

Workload keys: `prompt_tokens`, `output_tokens` (1–1,000,000), `concurrency` (1–256),
`repetitions` (1–100), `warmup_runs` (0–100), `mode` (`resident`/`cold`). KV capacity
is 1–2,000,000 tokens and must cover every case: `concurrency` x (`prompt_tokens` +
`output_tokens`), the KV its concurrent requests hold at once. With `device`, `create`
raises a wanted context below that to it (`hardware_constraints.context` records
`concurrency_tokens` and the `context_tokens` asked for), never past the model's positions or the
device's KV. Per-process ceilings: discovery 180 s,
build 1800 s, check 3600 s, benchmark 1800 s, configured agent 10800 s, all clipped to
the remaining segment window and requested cap. An `eval` attempt (`crit run eval`) has
no fixed cap: Eval's work scales with the model and the device, so it may run to the
segment's deadline, and Eval bounds each request by the rates the engine measured in the
same attempt (protocol `requests.timeout`), so a hung engine ends it (`eval_failed`).
Provider spend limits are the caller's responsibility.

### Inputs, proposal and package

| Payload | Fields |
| --- | --- |
| Inputs | Nonempty `hardware_manifest`, `environment_manifest`; `case_manifests` mapping **every** fixed case ID to an ordered nonempty list of `{path, bytes, sha256}` under the run's `captured/`. In a run with an origin engine, crit adds `origin_checkpoint`: whether the model is the exact checkpoint the engine was measured on |
| Proposal | `label`, `rationale`, `strategy`, `configuration` object, `store_decisions` object; `plan_id` (the [plan](#plan) it is built from); `lever_fates` (every lever of the plan `built`, `pruned` or `deferred`, with the reason unless built); optional `parent_candidate_id`, `origin_candidate_id` (earlier retained candidates); `weights` (`as_stored` or `transformed`, required when `weights_policy` is `as_stored`); `lineage` (`origin-engine`: the candidate is the origin engine's source as imported, in a run created with `--from-engine`). A run whose model is a folder needs `work/recon.json` (`recon`) before its first candidate; crit adds its `frozen` block as `configuration.recon` (a supplied `recon` must equal it) and `extensions.recon` with its path and SHA-256 |
| Package recipe | `build: {toolchain: {...}, dependencies: [...], command: [...]}`, nothing else: Core's loader runs the package and says what it serves |

In an `as_stored` run, a candidate declaring `transformed` weights (requantized,
pruned or changed) is measured but never eligible. In a `transformable` run, [Models](../models/README.md)
produces transformed weights with a record of recipe, calibration data, seed, hashes and
cost, or the engine re-encodes stored 16-bit matrices as it loads them (the route for an
already-quantized checkpoint) and names the matrix groups and formats in
`configuration.re_encoding`; the quality budget judges both. The coordinator cannot inspect
weights; declare honestly. `weights_policy` and `weights` cover the target weights that compute
every published token, in every run, whatever its quality mode. They do not cover draft-only
weights: whatever the run's `weights_policy`, a drafter (the checkpoint's draft layers,
a companion drafter, a copy of the head it drafts through) may use any encoding, since the
stored target weights verify every token; declare `weights` by the target weights alone and
record the drafter encoding in the configuration. Drafter values come from the checkpoint, a
published companion or `crit models drafter`, and are never trained or fine-tuned by the
agent. Nor do they cover runtime state (KV-cache and activation precision, judged by the
quality gates).

`store_decisions` maps Inference technique names to `{status, reason}`, status one of
`implemented`, `inherited`, `rejected`, `untested`. IDs are coordinator-assigned and
never recycled; labels don't determine paths or ordering.

For `plan_id` crit checks that the plan exists, is not superseded, is not linked to another
candidate (other than a sealed draft that never reached the ladder: [budget counts](#budget-counts)), has `"decision": "build"` and starts from the proposal's `parent_candidate_id`, and keeps
`plan_id` in `candidate.json`. `rationale` and `strategy` stay; they may summarize the plan.

`lever_fates` maps lever ids of that plan to `{fate, detail}`: `fate` is `built`, `pruned` or
`deferred`, and `detail` (why) is required unless built. Unknown ids are refused, and every lever of the
plan has exactly one fate. `candidate.json` keeps them.
Candidate creation stages its source, source manifest and identity together in a
scratch directory at the run root before publishing the candidate directory. An interruption before publication
leaves no candidate ID; after publication, `recover` rebuilds its telemetry from the
retained identity. Inspect the recovered candidates before issuing another `candidate`
command. Abandoned `.candidate-*` scratch directories are unpublished copies.

A candidate's source and its package are plugin packages: `plugins/` and the configuration whose `plugins`
selection fills every slot an engine fills, which `candidate` and `package` check with Core's loader rules
(`crit_core.engine.check_package`) without running any of its code ([crit plugins](PLUGINS.md#the-engine)).
Package dependencies give `name`, `version`, package-relative `path` and exact-byte
`sha256`. Omit `manifest.json` (crit
generates it); crit writes `configuration.json` when absent, and a supplied one must
equal the configuration. On POSIX, files identical to retained source are hard-linked,
so changing either fails verification. Keep dependencies, notices and build inputs in
the package. What the engine serves is Core's ([Serving](SERVING.md#cores-resident-adapter)); a package manifest of a run
from before plugin packages names its `launch` entry and `capabilities`, stays readable as history and does not
serve.

#### External engines

An external engine (the charter's vision A: an engine from elsewhere that takes crit's techniques and kernel
templates into its own code, never rebuilt as plugins) is a candidate of its own kind. Its configuration declares it
in `engine`, and its source and package hold its files and no `plugins/`:

```json
"engine": {
  "kind": "external",
  "launch": {"command": ["python", "server.py", "--port", "{port}", "--model", "{model}", "--context", "{context_tokens}"],
             "cwd": ".", "env": {}, "health": "/v1/models", "ready_seconds": 900},
  "capabilities": ["chat-tools-v1"],
  "max_concurrency": 8,
  "options": {"draft_width": {"flag": "--draft-width", "default": 4},
              "eager": {"flag": "--enforce-eager", "switch": true, "default": false}}
}
```

- `launch.command` starts its OpenAI-compatible server. crit fills `{port}` (required: crit picks a free port, so no
  two attempts meet on one), `{host}`, `{model}` (the run's model folder), `{context_tokens}` and `{package}` in the
  command and in `env`. A command without `{context_tokens}` serves at the engine's own context, and a part that
  asks for another context is served at it; the attempt's log says so. `cwd` is the package folder it starts in
  (default the package), `health` the path that answers 200 once it serves (default `/v1/models`) and
  `ready_seconds` how long it may take (default 900).
- `capabilities` are what it serves beside plain chat, in crit serve's names, for the serving checks (default: its
  model card's `capabilities`); `max_concurrency` how many requests it runs at once when its `/health` and card do
  not say.
- `options` are the switches it exposes, each a flag of the launch command: `default` is the value as built, and a
  `switch` is a bare flag given when its value is true. They are what `ablate` varies.

`serve` (with `--check`), `eval`, `measure`, `ablate` and `kernels` start it from its launch command under the
attempt's reservation, in its own process tree, wait for its health path, run the same serving checks, crit smoke,
Eval and benchmarks against its URL as for a plugin package, and stop the whole tree when the work ends, fails or
passes the reservation's deadline (a cleanup that cannot prove the tree gone leaves the attempt's cleanup unsettled).
Its output is kept as `logs/engine-ATTEMPT.log` beside the attempt's log, and a benchmark's `served_by` names the
launch command. A plan for it says `"engine": "external"`; its candidate must declare `configuration.engine`, a
child keeps its parent's kind, and its rungs name levers by technique or kernel (provider `inference:ID` or
`kernel:FAMILY/ID`) instead of plugin. A configuration is either a plugin selection or an external engine, never
both.

Packaging copies and validates the input inventory and recipe in a scratch directory
at the run root, then renames the complete package into the candidate. A copy or validation
failure can be retried. If publication succeeded but updating projections failed,
`recover` rebuilds them; repeating `package` with identical contents and recipe also
finishes publication without replacing retained bytes. Different contents require a new
candidate. An abrupt exit can leave unpublished `.package-*` scratch directories at
the run root; these are outside candidate inventories and are not runnable packages.
Staging stays outside `work/`, so that whole directory can be used as an input without
copying staged output back into itself.

### Plan

`crit run ... plan --record plan.json` records the next candidate's plan after its parent stands on the
[candidate ladder](#the-candidate-ladder) (after its Eval, or after its seal when its rung skipped Eval) and before
any code. crit assigns `plan-0001`, `plan-0002`, ... (never recycled), stores `plans/PLAN_ID/plan.json`, never
changes it afterwards and records a `plan_recorded` event. `plan --draft --record plan.json` writes the next plan's
skeleton instead (TODO where the planner writes, the parent's scores as a flat forecast, the `review` and `weakest`
entries this plan owes) and prints what crit computes for it: the objective and rung, the priorities while both
families are needed, each KPI's score, limit and noise, Eval's gaps and the parent plan's outcome. `plan --check`
runs every check and prints the record without storing it.

There is one set of plan rules. Every run follows it, whatever crit release created the run.

| Field | Content |
| --- | --- |
| `parent_candidate_id` | The candidate the plan starts from: the newest on the ladder; `null` only while the run has none |
| `supersedes` | Optional: the plan this one replaces ([revisions](#revisions)) |
| `decision` | `build` or `stop` |
| `needed` | `{"decode": {"needed": bool, "evidence": text}, "prefill": {...}}` |
| `gaps` | `[{kpi, gap, answer, detail}]` with optional `evidence` (a file in the run) and `cause_terms`. `gap` is an Eval gap's `name`, or one of crit's own: `limit` (why the rest of a family's limit is out of reach, answer `cause`), `activation:FORMAT` (a prompt activation format rejected on a quality measurement) or `placement:cpu` (the CPU expert tier rejected). `answer` is `cause` (its physical cause) or `rejected` (measured and rejected) |
| `strategy` | `{current: {name, ceiling, evidence}, alternative: {name, ceiling, evidence, limits?}, choice, reason}`; `ceiling` is text: a number, its unit and where it comes from; `choice` is `current` or `alternative`; `limits` below |
| `levers` | `[{id, technique, description, changes, group, confidence, provider, yardstick?, kind?, serves?}]`. `changes` maps KPIs (`decode`, `prefill`, `ttft`, `context_decode`, `context_ttft`) to `{change, source}`: the projected relative change in the KPI's improving direction (0.12 = 12% better), its source `measured:PATH` (a file of the run that measured it, such as an ablation report of the lever as an option variant), `eval-gap:NAME` or `estimate`. `group` names levers that only pay together; `confidence` is `low`, `medium` or `high`. `provider` says what supplies a speed lever's code: `inference:ENTRY`, `hub:ORG/NAME`, `lib:NAME` ([Kernels' sources](../kernels/SOURCES.md)) or `own: ...`, which carries `yardstick` `{evidence, detail}`: the measured best existing provider for the operation, or the providers checked and why none fits. `kind` `requirement` marks work a rule requires that Eval does not score, with `serves` naming it (`gate:NAME` or `check:NAME` for the repair of a gate or check the parent failed); its `changes` may be empty |
| `rejected` | `[{name, reason, bound_by?, evidence?}]`: alternatives considered and not planned; `bound_by` `{family, term}` names the round term that bound the measurement |
| `forecast` | `{KPI: {low, expected, high}}` in Eval's units (tok/s; ms for `ttft` and `context_ttft`) for `decode`, `prefill`, `ttft`, and `context_decode`, `context_ttft` when the parent's Eval measured them |
| `forecast_justification` | Optional: why a forecast that expects more gain on p2 than on p1, or lets a p1 KPI fall, is worth building |
| `gates` | How quality stays inside the budget and the required context is met |
| `estimated_limits` | Optional: `{KPI: {limit, derivation}}`, an estimate of a limit neither the parent's Eval gives nor crit computes |
| `prompt_activations` | `speed` runs: the formats the candidate's prompt GEMMs take per stored format, as the engine declares them to Eval |
| `round` | Optional: the parent's measured round per family, `{total_ms, idle_ms, terms: [{term, ms, floor_ms}], evidence}`, which a `limit` gap rests on |
| `review` | The review of the parent's plan against its outcome, `{plan_id, kpis, levers, lesson}` (check 12) |
| `weakest` | Critique rounds: the parent's two weakest terms, `[{kpi, term, evidence, levers}]` (check 11) |

crit adds the record header (`plan_id`, `segment_id`, `created_at`, `request_sha256`) and:

- `inputs`: the parent's newest completed Eval attempt that measured decode and prefill (`eval`: path and SHA-256),
  `recon_sha256` and a frozen extract of that result's `status`: per KPI the `score`, `limit` and `fraction` of the
  limit, the relative `noise` per KPI, the `untiered` scores (greedy decode, held-out decode, follow-up time to first
  token) and the `gaps`. A missing field is unknown (null), never an error. An unknown noise takes the largest known
  noise of its own family (decode with `context_decode`, prefill with `context_ttft`, time to first token alone),
  else 5% (`inputs.noise_default`). Beside them: `objective` (the candidate's rung, below), `computed_limits`,
  `fraction_sources`, `alternative_limits`, the evidence each check recorded and `warnings`.
- `priorities`: `{p1, p2, p3, estimated, sources}` computed from `needed` and the frozen fractions, or null with
  `priorities_reason`. `context_decode` counts toward the decode family, `context_ttft` toward prefill; a family's
  fraction is the lower of its members', its noise the largest of theirs.

  | Needed | p1 | p2 | p3 |
  | --- | --- | --- | --- |
  | Both, fractions within noise of each other | decode, prefill | ttft | |
  | Both, prefill further from its limit | prefill | decode | ttft |
  | Both, decode further from its limit | decode | prefill | ttft |
  | Only decode | decode | ttft | |
  | Only prefill | prefill | ttft | |
  | Neither | the plan is a stop plan | | |

  Within noise: the fractions differ by at most the two families' noise bands combined in quadrature,
  `√((noise_decode × f_decode)² + (noise_prefill × f_prefill)²)`. Near the limit (both at 0.85 or above) the lower
  one is p1 alone. Priorities order the work and settle conflicts; they never leave a lever out.

#### Computed limits

Where the parent's Eval states no limit for `decode` or `context_decode`, crit computes one from the same attempt,
with Eval's cost model: per measured decode case (long-context point) the measured `tokens_per_round` over the round
floor, the pass floor `(bytes_per_pass + kv_bytes_per_token × (prompt_tokens + output_tokens / 2)) / bandwidth`
plus the engine's draft width at Eval's draft floor, geometric mean. Where the draft width or floor is unknown, the
pass floor alone stands, labelled an upper bound. A limit below the measured score is kept with a note and not used.
Where neither Eval nor crit gives a limit and both families are needed, the plan states `estimated_limits` (at or
above the measured rate, at or below the measured time). A fraction comes from Eval's limit, then crit's, then the
estimate, and `priorities.sources` names which.

The priorities measure the distance to what is reachable, not to the current strategy's ceiling.
`strategy.alternative.limits` gives, for `decode` and `context_decode`, the alternative's `{tokens_per_round,
draft_steps, drafter, derivation}`; crit computes its round floor the same way and ranks the family by the higher of
the current limit and that ceiling. A claim is bounded: `draft_steps` at most the widest draft Eval projects (the
parent's draft width or 8), `tokens_per_round` at most `draft_steps + 1` and within the measured drafter's acceptance
at those steps, `(1 - a^(k+1)) / (1 - a)` per case. `drafter` is `measured` or `new`; a new drafter is assumed to
accept like the measured one unless `acceptance_evidence` cites another candidate's Eval attempt that measured it.
A separate drafter no Eval measured is priced from `recon --drafter`'s pass bytes (`companion` names one when recon
recorded several). Where crit cannot compute the floor, leave the KPI out of `limits`.

#### Checks

crit checks completeness and consistency with the frozen facts, never whether the reasoning is good. A check that
fails refuses the plan with what to fix:

1. Fields, types and text limits; unknown keys are refused. `eval-gap:` sources and Eval-gap entries in `gaps` name
   gaps of the parent's Eval.
2. The parent is on the ladder; a parent that went through Eval, or stands on the full rung, has a completed Eval.
3. `stop` requires decode and prefill both not needed; `build` requires at least one needed.
4. A family marked not needed answers every Eval gap on it above its KPI's noise in `gaps`; below 0.85 of its limit
   it adds a `limit` entry with the physical cause; below half its limit every entry on it carries `evidence` and an
   Eval gap answers `rejected`, and for prefill every faster prompt activation format Eval prices is rejected by a
   crit-eval result that measured it outside the quality noise (`activation:FORMAT`), under the run's quality terms.
5. A build plan names at least one speed lever, each with a KPI change; one improves more than time to first token;
   every needed p1 family has one.
6. A lever whose changes are all within their KPIs' noise shares a `group` with a lever above it. Measure an option
   (`crit run ... ablate --data EVAL_DATA --set PLUGIN.OPTION=VALUE`, cited as `measured:PATH`) rather than raise an
   estimate.
7. The forecast covers every KPI the parent's Eval measured, `0 < low ≤ expected ≤ high`, and improves decode,
   prefill, `context_decode` or `context_ttft` beyond its noise.
8. A forecast that gains more on p2 than on p1, or lets a p1 KPI fall, carries `forecast_justification`.
9. `strategy.current` and `strategy.alternative` are present; `limits` claims stay within their bounds (above).
10. Every speed lever names its `provider`, and an `own` one its `yardstick`.
11. Critique rounds (candidate 6 on) name the parent's two weakest terms in `weakest`, each with its evidence and
    the levers that improve it, and add other levers beside them.
12. Once the parent's plan has an outcome, `review` gives the cause and evidence of every KPI that missed its
    forecast, what each lever delivered and the lesson. The parent's largest p1 lever, pruned or deferred, is planned
    again or rejected on a measurement of another kernel structure or a vendor library at the same arithmetic.
13. `speed` runs declare `prompt_activations`; no quantized format starts at `bf16` without a rejection of the stored
    format (`activation:stored`); the parent served what its plan declared.
14. A `limit` gap rests on the parent's `round` account: its terms and idle time add up to the measured round within
    10%, `cause_terms` names the terms that bind it, and every other term of 10% of the round or more runs at least
    at half its floor's rate (idle time too).
15. A rejection with `bound_by` is reopened once a later round shows that term 25% smaller: the lever is planned
    again or rejected again on new evidence.
16. Where recon's placement prices CPU-computed expert misses below link copies, candidate 1 plans the CPU expert
    tier (`inference:host-expert-tier`) or rejects it with a measurement (`placement:cpu`).
17. `measured:PATH` sources and every `evidence` name files of the run; crit records their SHA-256.
18. `supersedes` names a plan of the same segment and parent, neither linked to a candidate nor superseded.

A gate repair (a requirement lever that `serves` `gate:NAME` for a gate the parent's Eval failed, or `check:NAME`
for a [check](#check) crit run requires that the parent failed) is built for the repair whatever it forecasts:
checks 5, 6, 7's improvement, 11 and the warnings below do not apply to it. The run's first plan skips the checks
that need a parent's Eval and may import a baseline as is.

Two judgements rest on the forecast alone, so crit records them as `inputs.warnings` instead of refusing the plan:
a forecast whose high stays below half of a needed family's limit without a `limit` gap that measured the bound
(`inputs.reach` keeps the share), and a candidate from 2 to 5 that forecasts less than +10% on a p1 KPI. A warning
says a lever may be missing. It is never a reason to raise a forecast.

The candidate's objective is its rung (`inputs.objective`, [candidate ladder](#the-candidate-ladder)): 1 the proof
of concept, 2 the plugin round, 3 to 5 crit rounds (crit's toolkit, each technique measured against its floor and
fixed or replaced where it trails), 6 on critique rounds.

#### Revisions

When building shows the plan was wrong before the candidate exists, record a new plan with `supersedes`.
The old plan stays on disk and in `state.plans`; only the newest plan of a chain can be linked. After a
candidate exists, a change of course is a new candidate with a new plan.

#### Outcome

After `crit run ... eval --candidate ID` retains a completed attempt of a candidate built from a plan,
crit writes `plans/PLAN_ID/outcomes/ATTEMPT.json` (event `plan_outcome_recorded`): for each forecast KPI
the `forecast`, the `measured` score, whether it landed `inside` the range and the relative `error` of
the expected value; `estimates` for each estimated KPI: the `estimate`, the `measured` score, whether it
`exceeded` the estimate (the estimate was too low), and the new result's `eval_limit` with the estimate's
relative `error` when Eval gives one; `requirements`, the plan's requirement levers with what each
`serves`; `levers`, each lever with its forecast `changes` and its `fate` from the proposal (null when none
was given); `forecast_on_p2`, the expected gain summed per tier when most of the forecast's gain falls
on p2 instead of p1 (else null); and `flags` for greedy decode, held-out decode and follow-up time
to first token when they fell against the parent's Eval by more than their noise (at all when it is
unknown). Facts only, no verdict.

### Attempt result

Required: `outcome`, `ended_at`, `exit_code`, `cleanup_settled` (boolean), `logs`
(`{path, sha256}` list), `failure` (message or null). Outcomes: `completed`, `failed`,
`cancelled`, `timed_out`, `interrupted`, `launch_failed`. Completion needs exit zero,
settled cleanup and an end within the reservation. Every attempt counts against the
limit; the candidate's lifecycle follows its latest attempt, and earlier failures stay
recorded. Reservations only record deadlines; the caller supervises processes and
descendant cleanup. A published reservation must be settled before `seal` or `resume`.

Settlement, candidate projections and recovery apply the same result validation,
including retained log hashes. `logs` must be a list, even when empty. A supplied
failure message must be nonempty text (at most 8000 characters); unsuccessful
candidate attempts require it, while agent/discovery results may use null.
Invalid results are rejected before publication; recovery never repairs retained
result bytes by silently normalizing them.

Reservations likewise share validation of their owner/operation, segment window,
process cap and one-attempt cost before publication, settlement or recovery.
Candidate projections additionally bind reservations to the candidate's creation
time, resolved inputs and package; build attempts may precede a package.

### Check

Required: `case_id`, `package_sha256`, `gate`, `criteria_sha256`, `status`,
`rationale`, `evidence` (retained `{path, sha256}` list), `evidence_level`,
`recorded_at`. `criteria_sha256` hashes the run's `correctness` object. Status is
`passed`, `failed`, `unverified` or `not_applicable`; passed checks need retained
evidence, and required failures disqualify. `artifact_verified` means evidence bytes
are bound, not that crit repeated the computation.

`check --from-eval` judges each gate by the Eval gates its criteria name, from the retained result
of a `crit run eval` attempt on the current package under the run's current quality terms (its quality
mode, drift budget and exactness, every resume's amendment applied):
`arithmetic` by `quality` and `sampling`; `functional` by `coverage`, `repeatable`, `request_history`,
`prompt_cache`, `retrieval`, `lower_context` and `concurrency` (every engine batches: batched replies equal serial
ones and more clients serve more tokens a second) (and `exactness` when the run requires it); `resources`
by `capacity` and the device memory left free at the required context against `vram_headroom_bytes`
(host memory comes from `measure`). A failed Eval gate fails the check, an unmeasured one leaves it
`unverified`; the check cites the result at `artifact_verified`.

### Ablation

`crit run ... ablate --candidate ID` serves a packaged, unsealed candidate built from
[plugins](PLUGINS.md#ablation) under a check attempt and builds it again in the same resident
host with each variant of its `configuration.plugins` (as built, at its base, without each plugin, with
each alone). Each variant must report its plugins in crit serve's `engine_facts`, sends the run's
resident cases (warmed until steady, then `--repetitions` 3) and runs crit smoke. The report,
`candidates/ID/logs/ablation-ATTEMPT.json`, is the attempt's evidence: each variant's per-case medians
and smoke items, and per plugin its change in the engine and alone, the smoke tests it breaks and a
verdict. Where both variants of a pair draft, a plugin outside the drafter and verifier slots is judged by the time
per decode round in place of decode, since a short bench's tokens per round move with what it samples; the row's
`judged` names the KPIs its verdict read. It records no check and no benchmark.

An [external engine](#external-engines) has no plugins to switch: `ablate` serves it as built (`full`) and with
`--set OPTION=VALUE[;...]` on the options its configuration declares (without `--set`, each declared switch
flipped), each started again from its launch command, benched and smoke-tested the same way and judged against
`full`. An option it does not declare is refused. An engine that declares no option has no ablation: the report
says so (`"ablation": "unavailable"` and the reason) and nothing is started.

### Benchmark

Required: `case_id`, `package_sha256`, `case_sha256`, `protocol_sha256`, `identities`,
exact `workload`, `metrics`, `missing_metrics`, `checks`, `outcome`, `started_at`,
`ended_at`. `identities` holds `model_sha256`, `input_sha256`, `hardware_sha256`,
`environment_sha256`: canonical hashes of the frozen model manifest, the case's file
list, and the hardware and environment manifests. Use canonical package-manifest, case
and protocol hashes, not file hashes of differently formatted JSON.

`checks` lists the `{path, sha256}` records of the case's checks
(`candidates/ID/checks/check-NNNN.json`). When it is missing or empty, `benchmark`
binds the case's latest recorded check per gate for the same package, so publish a
case's checks first. A benchmark qualifies only when its cited checks pass every
required gate at the run's evidence level. Otherwise it stays `exploratory` and the
candidate cannot be selected: eligibility needs every benchmark of every case to qualify, so one exploratory
benchmark keeps the candidate from ever being eligible. A required check that did not pass keeps every benchmark
of the candidate exploratory, whatever checks follow it.
A sealed candidate cannot be fixed, so `benchmark` and `seal` warn, `measure` refuses to record benchmarks that
could not qualify ([measured benchmarks](#measured-benchmarks)), and `select` names what blocks eligibility.

Each metric is `{unit, samples}` with exactly the declared repetitions; a per-request metric
(`prefill_tps`, `decode_tps`, `request_ms`, `ttft_ms`, `accepted_tokens_per_round`) of a case of
concurrency N may instead hold one sample per request, N per repetition, as `crit run measure` records:

| Metric | Unit / boundary |
| --- | --- |
| `prefill_tps`, `decode_tps` | `tokens/second` |
| `request_ms`, `load_ms` | `milliseconds`; complete request and model load |
| `peak_ram_bytes`, `peak_vram_bytes` | `bytes` |
| `minimum_free_ram_bytes`, `minimum_free_vram_bytes` | `bytes`; worst free sample determines headroom |
| `ttft_ms` (optional) | `milliseconds`, admission to first emitted token |
| `startup_ms` (optional) | `milliseconds`, launch to ready |
| `perplexity` (optional) | `dimensionless`, ≥ 1; fixed corpus/tokenization/scoring |
| `kl_divergence` (optional) | `nats/token`, ≥ 0; mean KL from a fixed reference's next-token distributions over a fixed corpus |
| `accepted_tokens_per_round` (optional) | `tokens/round`, ≥ 1; emitted tokens per verify round |
| `aggregate_tps` (optional) | `tokens/second`; a repetition's concurrent requests' completion tokens over its wall time (concurrency above 1) |

Absent required metrics need reasons in `missing_metrics`; missing is never zero.
Keep metrics paired and every sample. The reader computes medians grouped by exact
model/input/hardware/environment, case, protocol and correctness policy (contexts,
concurrency and cold/resident modes group separately). Eligibility requires the fixed
gates and evidence level; target satisfaction checks all retained benchmarks across
required cases, not the fastest series. Served request timing never enters projections.

### Measured benchmarks

`crit run measure` measures the run's cases on a candidate as a client of `crit serve`
gets it, once its checks are recorded. In the run order it follows `crit run ... eval` and `crit run ... check
--candidate ID --from-eval`, and comes before `seal`: its benchmarks bind the checks recorded before them, and a
sealed candidate takes no new benchmark. Before reserving anything, `measure` refuses a candidate whose recorded
checks would leave its benchmarks exploratory, and says why and what to do:

- a case with no check of a required gate, or whose latest passed one is `reported` where the run requires
  `artifact_verified` evidence: record the checks first (`check --from-eval` records the arithmetic, functional
  and resources checks of every case from the candidate's completed Eval);
- a required check that did not pass (`failed`, `unverified`, `not_applicable`), or an earlier benchmark whose free
  memory fell below the run's headroom (or went unmeasured): no later record undoes it, so the candidate cannot
  become eligible, and the fix is a new candidate (its plan repairs a failed check with a requirement lever serving
  `check:GATE`, [plan checks](#checks)).

`--exploratory` measures such a candidate anyway, for telemetry or memory during development: its measurement log
says why the benchmarks stay exploratory, the command warns as `benchmark` does, and the candidate is never
eligible. The free memory of the measurement itself is judged once its benchmark is recorded.

Each case keeps its chat request as a retained JSON file
`{"messages": [...], "tools": [...]}` (`tools` optional) in its manifest. Under one
benchmark reservation `crit serve` serves the packaged candidate at the run's
`kv_tokens`, and every warmup and repetition streams the request to
`/v1/chat/completions` with the sampling fields the captured request names
(`temperature`, `top_p`, `top_k`, `min_p`, `seed` and the penalties), else `temperature`
0, the chat-template variables the captured request names in `chat_template_kwargs` (for
example `{"enable_thinking": true}` for a workload captured with thinking on, which a candidate
serves off by default), `max_tokens` the workload's `output_tokens` and usage. `extensions.sampling`
records the sampling sent, `extensions.chat_template_kwargs` the variables. `crit serve`'s usage must count the workload's
`prompt_tokens`: a case whose messages render to another length fails the measurement,
naming both counts. Every candidate decodes its own reply, and one that ends
before `output_tokens` is measured on the tokens it has; `extensions.output_tokens`
records each repetition's length. Times are client-observed from the streamed chunks, as
in Eval: time to first token, prefill (the prompt over the time to first token), decode
(the tokens after the first over the time to the last) and the request to its last
token. `load_ms` is the time `crit serve` took to load the engine, and
`accepted_tokens_per_round` is `crit_telemetry.tokens_per_round` when the engine
reports its decode rounds ([SERVING.md](SERVING.md)). Memory is sampled during each
request: host `MemAvailable` (Linux) or `GlobalMemoryStatusEx` (Windows), the resident
memory of `crit serve`'s resident host process, where the engine runs, and free device
memory from `nvidia-smi` for GPU 0. Device memory is the driver's whole-GPU view, and
metrics a host cannot measure go to `missing_metrics`.

A case of `concurrency` N sends each warmup and repetition as N copies of its request started
together, one per client thread, to an engine `crit serve` runs several requests of at once
(a package declaring `chat-concurrent-v1`; [concurrent requests](SERVING.md#concurrent-requests)).
Every request's timings are samples of its metrics, N per repetition, and `aggregate_tps` is the
repetition's completion tokens over its wall time, from the first request sent to the last reply
piece. `extensions.concurrency` records the clients, the requests in flight at once at each
repetition's peak (`peak_concurrency`) and the engine's `max_concurrent_requests`; a peak below N
means the requests did not overlap, and an engine running fewer at once queues the rest.
Memory is one window per repetition. crit Eval measures concurrency on its own suite as well
([concurrency](https://github.com/crit-labs/crit-eval/blob/main/docs/MEASUREMENTS.md#concurrency)).

The resident host runs under `crit serve`'s launch guardian and runtime ownership,
which retire its process tree on completion, failure, the reservation's deadline and
interruption. The attempt settles once the server has stopped; a failed engine cleanup
settles it with `cleanup_settled` false and blocks further measurement or serving.
Once the coordinator has exited, use `crit serve --recover-runtime` from the same
OS/user environment to verify retirement and release that guard. This does not settle
the run's attempt: an abrupt kill can leave its reservation pending, requiring an
explicit `crit run ... settle` record with the verified cleanup evidence. Existing
unsettled attempt results remain retained; runtime recovery does not rewrite them.

The reservation includes time waiting for the machine's engine lock. Expiry
before admission serves nothing; completion after the deadline records `timed_out`
and publishes no benchmarks. Failures and interruptions also settle the attempt,
with the measurement's log (`logs/measure-ATTEMPT.log`). An engine cleanup that
cannot be proven records `cleanup_settled: false`; it does not qualify the candidate
for sealing. If the coordinator cannot write the attempt result itself, the
reservation stays pending. Logs are hashed as retained evidence only after cleanup
is proven; otherwise the log stays in the candidate's `logs/` directory unhashed.

### Eval evidence

[Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md) result JSON is not a `check` or `benchmark` payload and is
never imported automatically. `eval` retains it under the candidate before sealing;
reference its bytes from a check's `evidence`. Keep any Eval adapter and inputs with the
candidate. Publish benchmark samples only under cases and a protocol frozen for them;
an Eval case cannot be relabelled as a run workload, and Eval's model identity differs
from the run's manifest hash. Report skipped cases and overrides. Eval does not measure
physical memory, and an Eval pass passes no run gate.

Every Eval attempt journals its round beside its result (`candidates/ID/eval/ATTEMPT.journal.jsonl`, crit-eval's
[journal](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md#journal-and-resume)): each engine it opens,
part it starts and request it serves, synced as it completes. The run's events record each attempt's start
(`eval_started`, or `eval_resumed` with the journal it resumes) beside its settlement. An attempt that stops for any
reason (interrupted, failed, timed out, a disturbed device or a hung engine) is settled with its log, its journal and
what it measured (the result, marked `aborted`) as its evidence. A disturbed device is judged on the repetitions
after a case reached steady state (crit-eval's `speed.steady_state`, recorded per case as `steady_state` with its
warmups and drift), so an engine that adapts between requests is not stopped for it; only a completed attempt's result judges gates
(`check --from-eval`, plans). `eval --resume [ATTEMPT]` measures in a new attempt from the journal of the
candidate's newest stopped attempt (or ATTEMPT's): the journal answers the requests it holds at their recorded
times and the candidate is served only for the rest. It must be the same package, data and options. An attempt
whose process ended without settling (killed, its session or the machine stopped) is settled `interrupted` by
`--resume` once the machine's engine lock, which Eval holds while it measures, is free and its journal has been
untouched for a minute; with the lock off, settle it first.

### Stop

Supply `reason`, `critique` (the engine and remaining opportunities), `opportunities`
(ranked list), `cleanup` (`external`, `settled` or `unresolved`), optional `message` and
`headroom`. Reasons: `satisfied` (needs an eligible selected candidate meeting the
targets), `search_exhausted`, `blocked`, `budget_exhausted` (refused while fewer of the
segment's candidates than `candidate_budget` stand on the ladder; once it is used, `candidate`
is refused until a resume), `cancelled`, `interrupted`.

`headroom` is `{"kpis": [{kpi, start, reached, limit, built, cause}]}`, `kpi` one of
`decode`, `prefill`, `ttft`, `context_decode`, `context_ttft` (the last two when Eval
measured the required context), each at most once. `start` and `reached` are Eval
scores, `limit` the physical limit (decode ceiling, prefill compute ceiling, ttft floor
in ms, long-context ceiling and floor), each null when Eval lacks it; `built` is what
was built for the KPI, `cause` the physical cause of any remaining distance. Eval's
`status.stop_record_headroom` fills scores and limits (`start` from `--baseline`).
`search_exhausted` and `budget_exhausted` require decode, prefill and ttft, and
`context_decode` and `context_ttft` when the run's newest Eval measured the long-context curve. The
account is recorded in the `run_stopped` event.

Stopping early is a plan: `search_exhausted` while the segment still has budgeted candidates requires that the segment's newest unsuperseded plan is a stop plan whose `inputs`
froze the newest completed Eval attempt of the newest candidate that has one. With the budget used up,
no stop plan is needed.

### Origin engine

`crit run create --record request.json --from-engine ENGINE_DIR` starts a run from an engine listed in
crit-hub: `ENGINE_DIR` is its `engines/<owner>/<name>/` folder in a crit-hub checkout (`card.yaml`,
`card.lock.json`, `evidence/`; reading the card needs PyYAML, as crit-hub's tools do). The request's
`origin_engine` then records the engine (`<owner>/<name>`, title, origin, contributor), the repository,
commit and folder of its source, the card's and lock's hashes, every checkpoint it was measured on with
each file's size and SHA-256, its sealed provenance and its published crit Eval results, each with the
quality mode it was judged in (a card's result that states none is refused), which are copied into the run
under `captured/origin/` as the numbers before the run's work; the agent request names each one's mode, and
says that a result judged in another mode than the run's does not compare with its Evals. The agent request tells the
agent to check the source out at that commit and make it the first candidate, declared with
`"lineage": "origin-engine"` in its proposal: its rung takes a small Eval (stage 1), so the run has a baseline
measured in its own mode before any change. At
`resolve`, `inputs.json` gains `origin_checkpoint`: per checkpoint the files the run's model lacks or
stores with other bytes, and `matched`, the checkpoint the model is, or null; a different checkpoint is
recorded, never refused, and `recon --compare` shows what it changes. crit-hub's exporter writes the
origin into the new engine's card as `derived_from`.

### Cost

`crit run ... cost` writes `cost.json`, the run's build cost: wall time, the agent's model requests and
tokens, and their API-equivalent price, per candidate and for the run. It reads the coding agent's
transcripts (`--transcript`, repeatable: a Claude Code session JSONL, a Codex rollout JSONL, or a plain
JSONL of `{"timestamp", "model", "input_tokens", "output_tokens", "cache_read_input_tokens",
"cache_creation_input_tokens"}`), counts each response once and attributes it by time to the candidate
being built: a candidate's span runs from the previous candidate's seal (the run's creation for the
first) to its own seal, so its row is what it cost to reach it, with running totals from the run's first
event; requests after the last seal are `selection`, requests before the run `before_run`, and the run's
`until` is its stop, `--until`, or now. Tokens are priced at Anthropic's list prices per million tokens
(input, cache reads, cache writes with a 5-minute or 1-hour lifetime, output; the record keeps the table);
`--prices` adds a model or changes a rate. A request whose model has no price keeps its tokens and makes
the bucket's `usd` null, with the priced part in `usd_priced`. `--agent` names who built the run. The
record is derived data, written again on each call; crit-hub's exporter copies its summary into the
engine's card as `built`.

## Files and identities

```text
run-ID/
  run.json, inputs.json          fixed request and resolved inputs
  state.json                    replaceable run projection
  segments/segment-NNNN.json     bounds, counters and deadline
  segments/segment-NNNN/         agent_request.md and discovery/agent attempts
  events/event-NNNNNNNN.json     ordered events
  captured/                     retained workload files
  captured/origin/              the origin engine's published Eval results (create --from-engine)
  work/                         mutable build scratch
  work/recon.json, recon.md     model recon: gathered facts with sources, frozen block, notes
  plans/plan-NNNN/plan.json     fixed candidate plan and its frozen inputs
  plans/plan-NNNN/outcomes/     ATTEMPT.json: the forecast next to each Eval of its candidate
  candidates/candidate-NNNN/
    candidate.json              fixed proposal/configuration
    source/, source.manifest.json
    package/manifest.json       the plugin package's files and build identities
    attempts/attempt-NNNN/      reservation.json and result.json
    checks/, benchmarks/, logs/
    eval/ATTEMPT.json           Eval results of crit run eval
    telemetry.json             active projection
    outcome.json, seal.json     final outcome and complete inventory
  cost.json                    build cost from the agent's transcripts (crit run cost)
  chats/                       separate requests, replies and conversations
```

Records are UTF-8 JSON with unique keys, finite numbers, integer `schema_version: 1`,
the correct `record_kind`, owner IDs and timezone-aware UTC timestamps. Projections
have increasing integer revisions. Optional extensions go in `extensions`; invalid
versions/fields are rejected. Limits: **1 MiB per JSON record, 512 candidates per run,
100,000 files per inventory**; link larger evidence.

Paths are canonical forward-slash paths relative to the owning run (or source/package
for its inventory); traversal and symlink/reparse escapes are rejected. File hashes use
exact bytes. Canonical JSON hashes use sorted keys, compact separators, UTF-8 with
`ensure_ascii=False` and `allow_nan=False`.

Benchmarks may repeat before sealing. Sealed `built` means runnable without a
benchmark claim; `completed` requires measurements; other lifecycles are
`build_failed`, `check_failed`, `benchmark_failed`, `cancelled`, `timed_out`,
`interrupted`. A seal binds the full inventory and final telemetry revision; any
source, flag, configuration, dependency or representation change, or later
evaluation, needs a new candidate.

If sealing is interrupted after writing the outcome, repeat `seal` with the same
lifecycle and failure. It preserves the original outcome and completes the seal.
If the seal was already published, the retry verifies its inventory and updates only
run state and history. A different outcome or damaged sealed artifact is rejected.

## Replays

A replay asks what an agent decides at one point of a finished run, without building anything: after a change to
crit's rules or to crit-eval's reports, replays of an old run at a candidate show whether an agent now plans
differently from that candidate's Eval, in minutes instead of a run.

```text
crit run --root RUN_ROOT --id RUN_ID replay --at CANDIDATE --out DIR --count 3
```

makes `DIR/replay-1` ... `DIR/replay-3`, each a run root holding a copy of `RUN_ID` as it stood when `CANDIDATE`'s
newest completed Eval attempt ended (its attempt's `ended_at`, the plan outcome crit wrote for it and five seconds of
margin): every file written by then, nothing later. File times are the wall clock's, which a virtual machine's steps
back by up to a second, so a file counts as written by a moment only when it is dated two seconds before it (crit's
clock slack). `--before PLAN` ends the copy where that plan's record was made
instead, so it holds the work its agent did after the parent's Eval toward the plan (measurements, rejected tries),
for a decision taken with that evidence. `--before` the run's first plan copies the run as it stood then: the recon
and any work toward the first candidate, no candidate and no Eval, and no plan draft at all. Records and candidates are
copied whole, except a later candidate whose record postdates the copy (its retained source can keep older file
times); `work/` leaves out build products
(shared libraries, objects, weights and files over 16 MiB) and drafts of the next plan (JSON with a plan's
`parent_candidate_id`, `decision`, `needed` and `levers`, written after the parent's Eval). Work the agent kept
outside the run folder, or only in its session, is not in any replay. `state.json` and the projections of unsealed
candidates are rebuilt from the copied records. The run itself is only read.

Each replay renders the Eval status of every copied candidate again with today's crit-eval
(`replay/status/CANDIDATE.json` and `.txt`, against the baseline its result named), and `plan --check` there judges the
parent's Eval by that status. `replay.json` names the run, the candidate, the cutoff and the statuses; while it exists,
crit refuses every command that records or changes a record in the run (candidates, plans, packages, attempts, Eval,
seals, stops, resumes). `replay/agent_request.md` is the replay's brief: the agent studies the run as its agent would
after that Eval, writes the plan it would record next to `replay/plan.json`, checks it with `plan --record
replay/plan.json --check` and says why in `replay/next.md`. Start one agent per replay, each with nothing but its
replay folder (no memory or notes of the original run), so they decide independently. crit starts no agent of its
own and works with any harness: `--agent CMD` runs a command once per replay, all at once, in the replay's run folder,
with `{brief}`, `{folder}`, `{root}` and `{number}` replaced by the replay's brief, run folder, run root and number
(and `CRIT_REPLAY_BRIEF` and `CRIT_REPLAY_FOLDER` set), writes its output to `replay/agent.log` and its exit to
`replay/agent.json`, waits for all and collects:

```text
crit run --root RUN_ROOT --id RUN_ID replay --at CANDIDATE --out DIR --count 3 \
  --agent 'codex exec "Read {brief} and do what it asks."'
```

Any command that starts an agent on a prompt works the same way (another CLI harness, or a script driving a model
API). Agents started some other way are collected afterwards with

```text
crit run --root RUN_ROOT --id RUN_ID replay --collect --out DIR
```

checks each replay's plan and lists its decision, priorities, levers with their KPI changes, rejections and forecast.

## Recovery and Resume

A per-operation coordinator lease serializes writers. Immutable records publish
atomically and exclusively, projections by atomic replacement, with no cross-file
transaction; readers wait for consistent revisions. The lease names the owner's PID,
host and OS. A persistent `.coordinator.guard` file carries a kernel lock around
lease acquisition, dead-owner recovery and writes. The OS releases that lock when
its process exits; the guard file must not be removed or replaced. It prevents two
recoverers from removing each other's replacement lease. The owner lease still
excludes writers across OS instances.

A terminated `crit run` (SIGTERM) releases its lease; a killed one leaves it for
`recover`, which verifies a dead owner, keeps valid orphaned evidence and rebuilds
**active** projections, never rewriting seals, inventing records, resetting counters or
starting processes. Live or unproven owners and corrupt identities block it, and it
must run on the owner's host and OS (WSL and Windows process IDs differ).
Seal publication uses temporary files under `work/`, so an abrupt exit cannot leave
an extra file inside a newly sealed candidate and invalidate its inventory.

Before sealing or resuming, settle every reservation and external process tree; a
vanished coordinator does not prove an agent stopped. `resume` keeps the request and
results, appends fresh bounds and requires sealed candidates plus cleanup confirmation.
After a resume, an Eval that stopped goes on with `eval --resume`; measure a sealed candidate from the start again
only when crit, crit-eval, the driver or the device changed since its Eval (the same package can measure several
percent differently in another session). Changing criteria, model or workload needs a new run.
