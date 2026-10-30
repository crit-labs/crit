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
and speed techniques; a larger one trades context for that room.

| Action | Arguments / effect |
| --- | --- |
| `create` | `--record request.json [--from-engine ENGINE_DIR]`; new run and first segment; with an engine folder of a crit-hub checkout, the run starts from that engine ([origin engine](#origin-engine)) |
| `resolve` | `--record inputs.json`; freezes verified model and input identities |
| `plan` | `--record plan.json [--check]`; records the next candidate's plan after the parent's Eval and before any code, freezes that Eval and computes the priorities ([plan](#plan)); with `--check`, runs the same checks and prints the record it would store, recording nothing |
| `candidate` | `--source DIR --record proposal.json`; allocates an ID, retains source |
| `capture` | `[--case ID] [--text FILE] [--instruction TEXT] [--system TEXT] [--template-kwargs JSON] [--inputs INPUTS_JSON]`; before `resolve`, after `recon`: writes `captured/CASE/request.json`, a chat request whose messages render to the workload's `prompt_tokens` with recon's frozen `template_variables` under its own `chat_template_kwargs` (the given text or a neutral passage, cut to whole words and padded to the exact count), prints its manifest row and sets it in the inputs payload |
| `build` | `--candidate ID --record recipe.json [--seconds N] [--drop GLOB ...] [--no-package]`; reserves a build attempt, copies the retained source to `work/builds/ID-ATTEMPT`, runs the recipe's literal `build.command` there to the deadline (whole process tree; log `logs/build-ATTEMPT.log`), settles the attempt and, when it completed, packages that folder: each file the build added or changed bound as a dependency (`version` `built by ATTEMPT`) after the recipe's own, capabilities the recipe omits from the adapter's `declared_capabilities(model_path)`; `--drop` removes intermediates first |
| `reserve` | `--operation discovery/build/check/benchmark [--candidate ID] [--seconds N]`; consumes an attempt before launch; only discovery omits the candidate |
| `settle` | `--reservation RUN_RELATIVE_PATH --record result.json`; exit, logs, cleanup |
| `package` | `--candidate ID --source DIR --record recipe.json`; runnable files and manifest |
| `check`, `benchmark` | `--candidate ID --record payload.json`; appends evidence. `check --candidate ID --from-eval [ATTEMPT]` instead records every case's arithmetic, functional and resources check from the candidate's retained Eval result (default its newest; [Check](#check)) |
| `measure` | `--candidate ID [--seconds N]`; reserves a benchmark attempt, serves the packaged candidate through `crit serve`, streams every resident case's chat request to it, settles it and records one benchmark per case ([measured benchmarks](#measured-benchmarks)) |
| `serve` | `--candidate ID [--host HOST] [--port 8000] [--seconds N]`; reserves a check attempt, serves the packaged, not yet sealed candidate through `crit serve` at `http://HOST:PORT/v1` until Ctrl+C or the reservation's deadline, for the checks a candidate passes before sealing, and settles the attempt (log `logs/serve-ATTEMPT.log`); with `--check` crit runs the standard serving checks itself (consecutive greedy requests, a system message, streaming, a stop string, a client cancelling, a tool call and its result, log-probabilities, a template switch, each where declared, and stopping the engine), keeps their report as `logs/serve-check-ATTEMPT.json` and records a functional check per case; `crit serve` serves the candidate once it is sealed |
| `eval` | `--candidate ID --data EVAL_DATA_DIR [--only PART] [--baseline RESULT] [--label TEXT] [--seconds N]`; reserves a benchmark attempt, serves the packaged candidate through `crit serve`, scores the data's quality texts on the run's model unless stored, runs Eval over the chat API with the run's quality terms and `kv_tokens`, settles it and retains the result as `candidates/ID/eval/ATTEMPT.json` ([Eval evidence](#eval-evidence)); for a candidate created from a plan it writes the plan's [outcome](#outcome), and it prints the priorities the result's fractions give |
| `seal` | `--candidate ID --lifecycle STATE [--failure TEXT]`; closes the candidate |
| `select` | `--candidate ID`; selects a sealed eligible candidate |
| `stop` | `--record stop.json` |
| `resume` | `--candidates N [--attempt-limit N] [--external-agent-stopped]`; new segment for N more candidates (`--minutes N` alone: legacy time-budget segment). `--quality-budget JSON`, `--exactness MODE`, `--approved-by NAME` amend quality terms from this segment on, recorded as the segment's `amendment` |
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
| `candidate_budget`, `attempt_limit` | Segment bounds: candidates to build (default 3, 1–64) and attempts (default 64, 1–1000). `budget_minutes` is a safety window, default and maximum 43200 (30 days). `budget_minutes` without `candidate_budget` creates a legacy time-budget segment (1–180) |
| `workload` | Defaults: prompt 512, output 128, concurrency 1, repetitions 3, warmups 1, mode `resident` |
| `cases` | Optional 1–32 distinct `{case_id, workload}` objects with complete workloads |
| `benchmark_protocol` | Fixed protocol identity; default `crit-matched-v1` |
| `verification_level` | `reported` or `artifact_verified` (default) |
| `targets` | Optional positive `decode_tps`, `prefill_tps`, `request_ms` thresholds |
| `weights_policy` | Optional `as_stored` (supplied weights used as stored) or `transformable`; omitted means unspecified |
| `quality_budget` | Optional drift budget against Eval's reference: any of `ppl_ratio_max` (≥ 1), `kld_mean_max`, `kld_p99_max` (≥ 0); defaults 1.05, 0.05, 0.5 (`as_quantized`: 1.1, 0.1, 1.0, Eval's own for that mode). Agents spend it on speed |
| `exactness` | `reported` (default: speculative output may differ from plain greedy; the drift budget judges it) or `required` (any difference fails Eval's gate) |
| `quality_mode` | `preserve` (default: Eval's quality reference keeps the activations exact, and prefill's limits are priced at 16-bit prompt activations) or `as_quantized` (the same reference, each quality set's limits widened by the drift of the checkpoint's own activation quantization, crit-eval's allowance, and prefill's limits priced at its stored format); the user's choice, from the request or `settings.toml`'s `quality_mode`. `crit run ... eval` judges in it |
| `eval_sampling` | Optional, the user's only: `{settings, approved_by}`, settings `temperature` (> 0, required) and any of `top_p`, `top_k`, `min_p`, as an object or `"temperature=0.6,top_p=0.95"`. Eval's sampled suite then measures at these settings instead of the model's `generation_config.json`: `crit run ... eval` requires data prepared with `crit eval prepare --sampling` and the same settings (and refuses overridden data without it), recon applies them to the served default, and results label them as an override |
| `origin_engine` | Written by `create --from-engine`, never by hand: the crit-hub engine the run starts from ([origin engine](#origin-engine)) |

`create` freezes `"plans": "required"` into the run spec: every candidate of the run names a recorded
[plan](#plan). Runs created before it have no `plans` field; their proposals may name a plan, which is
then checked. It also freezes `"plan_limits": "computed"`: the run's plans rank with the decode limits
crit computes when Eval states none ([computed limits](#computed-limits)); runs without it keep the
agent's estimates. And it freezes `"lever_fates": "required"`: each proposal gives every lever of its
plan a fate; runs without it keep fates optional, checked when given. With `"plan_limits"` it freezes
`"plan_rules": 2`: the run's plans follow the second version of the plan rules ([plan rules
2](#plan-rules-2)); runs without it keep the first.

Workload keys: `prompt_tokens`, `output_tokens` (1–1,000,000), `concurrency` (1–256),
`repetitions` (1–100), `warmup_runs` (0–100), `mode` (`resident`/`cold`). KV capacity
is 1–2,000,000 tokens and must cover every case. Per-process ceilings: discovery 180 s,
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
| Proposal | `label`, `rationale`, `strategy`, `configuration` object, `store_decisions` object; `plan_id` (the [plan](#plan) it is built from; required when the run spec has `"plans": "required"`); optional `parent_candidate_id`, `origin_candidate_id` (earlier retained candidates); `weights` (`as_stored` or `transformed`, required when `weights_policy` is `as_stored`); `lineage` (`origin-engine`: the candidate is the origin engine's source as imported, in a run created with `--from-engine`). A run whose model is a folder needs `work/recon.json` (`recon`) before its first candidate; crit adds its `frozen` block as `configuration.recon` (a supplied `recon` must equal it) and `extensions.recon` with its path and SHA-256 |
| Package recipe | `build: {toolchain: {...}, dependencies: [...], command: [...]}`, literal `launch` argument list, `capabilities` list |

In an `as_stored` run, a candidate declaring `transformed` weights (requantized,
pruned or changed) is measured but never eligible. In a `transformable` run, [Models](../models/README.md)
produces transformed weights with a record of recipe, calibration data, seed, hashes and
cost. The coordinator cannot inspect
weights; declare honestly. `as_stored` covers the target weights that compute every
published token. It does not cover draft-only weights (a drafter may use any encoding
when stored target weights verify every token; declare `as_stored` and record the
drafter encoding in the configuration; drafter values come from the checkpoint or a
published companion and are never trained or fine-tuned) or runtime state (KV-cache
and activation precision, judged by the quality gates).

`store_decisions` maps Inference technique names to `{status, reason}`, status one of
`implemented`, `inherited`, `rejected`, `untested`. IDs are coordinator-assigned and
never recycled; labels don't determine paths or ordering.

For `plan_id` crit checks that the plan exists, is not superseded, is not linked to another
candidate, has `"decision": "build"` and starts from the proposal's `parent_candidate_id`, and keeps
`plan_id` in `candidate.json`. `rationale` and `strategy` stay; they may summarize the plan.

`lever_fates` maps lever ids of that plan to `{fate, detail}`: `fate` is `built`, `pruned` or
`deferred`, and `detail` (why) is required unless built. Unknown ids are refused. In a run with
`"lever_fates": "required"` every lever of the plan has exactly one fate; elsewhere fates are optional
and checked when given. `candidate.json` keeps them.
Candidate creation stages its source, source manifest and identity together in a
scratch directory at the run root before publishing the candidate directory. An interruption before publication
leaves no candidate ID; after publication, `recover` rebuilds its telemetry from the
retained identity. Inspect the recovered candidates before issuing another `candidate`
command. Abandoned `.candidate-*` scratch directories are unpublished copies.

Package dependencies give `name`, `version`, package-relative `path` and exact-byte
`sha256`; `launch[0]` names a retained relative file. Omit `manifest.json` (crit
generates it); crit writes `configuration.json` when absent, and a supplied one must
equal the configuration. On POSIX, files identical to retained source are hard-linked,
so changing either fails verification. Keep dependencies, notices and build inputs in
the package. Serving capabilities (`chat-json-v1`, `chat-resident-v1`, optional
`chat-stream-v1`, `chat-tools-v1`, `chat-config-v1`, `chat-constraints-v1`) are in
[Serving](SERVING.md#adapter-contract).

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

`crit run ... plan --record plan.json` records the next candidate's plan after the parent's Eval and
before any code. crit assigns `plan-0001`, `plan-0002`, ... (never recycled), stores
`plans/PLAN_ID/plan.json`, never changes it afterwards and records a `plan_recorded` event.

| Field | Content |
| --- | --- |
| `parent_candidate_id` | The candidate whose Eval the plan starts from; `null` only while the run has no candidate |
| `supersedes` | Optional: the plan this one replaces ([revisions](#revisions)) |
| `decision` | `build` or `stop` |
| `needed` | `{"decode": {"needed": bool, "evidence": text}, "prefill": {...}}` |
| `gaps` | For KPIs marked not needed: `[{kpi, gap, answer, detail}]`, `gap` the Eval gap's `name`, `answer` `cause` (its physical cause) or `rejected` (measured and rejected); optional `evidence`, a file in the run, which [plan rules 3](#plan-rules-3) require below half the limit |
| `strategy` | `{current: {name, ceiling, evidence}, alternative: {name, ceiling, evidence}, choice, reason}`; `ceiling` is text: a number with its unit and where it comes from; `choice` is `current` or `alternative` |
| `levers` | `[{id, technique, description, changes, group, confidence}]`. `technique` is an Inference entry name or free text; `changes` maps KPIs (`decode`, `prefill`, `ttft`, `context_decode`, `context_ttft`) to `{change, source}`: `change` is the projected relative change of that KPI's score in its improving direction (0.12 = 12% better, negative = worse), `source` is `eval-gap:<gap name>` or `estimate`. `group` is null or a name shared by levers that only pay together; `confidence` is `low`, `medium` or `high`. Optional `kind`: `speed` (the default) or `requirement`, work a rule or gate requires that Eval does not score, with `serves` naming that rule or gate (e.g. `AGENTS.md: sampled requests speculate`); its `changes` may be empty or 0 |
| `rejected` | `[{name, reason}]`: alternatives considered and not planned |
| `forecast_justification` | Optional (plan rules 3): why a forecast that expects more gain on p2 than on p1, or lets a p1 KPI fall, is worth building |
| `weakest` | Plan rules 4, critique rounds: the parent's two weakest terms, `[{kpi, term, evidence, levers}]` ([plan rules 4](#plan-rules-4)) |
| `review` | Plan rules 4: the review of the parent's plan against its outcome, `{plan_id, kpis, levers, lesson}` ([plan rules 4](#plan-rules-4)) |
| `forecast` | `{KPI: {low, expected, high}}` in Eval's units (tok/s; ms for `ttft` and `context_ttft`) for `decode`, `prefill`, `ttft`, and `context_decode`, `context_ttft` when the parent's Eval measured them |
| `gates` | How quality stays inside the budget and the required context is met |
| `estimated_limits` | Optional: `{KPI: {limit, derivation}}`, the agent's estimate of a limit the parent's Eval does not give, in Eval's units (tok/s; ms for `ttft` and `context_ttft`), and how it was derived (for example a drafter's expected tokens per round over the pass floor) |

crit adds the record header (`plan_id`, `segment_id`, `created_at`, `request_sha256`) and:

- `inputs`: the parent's newest completed Eval attempt (`eval`: path and SHA-256), `recon_sha256`
  (`work/recon.json`) and a frozen extract of that result's `status`: per KPI the `score`, `limit` and
  `fraction` of the limit (each goal's `fraction_of_limit`, else the KPI's `fraction_of_ceiling` or
  `fraction_of_floor`), the relative `noise` per KPI (each goal's `noise`; for results without it,
  decode is two standard errors or the end-to-end decode noise, ttft the end-to-end ttft noise, the
  others unknown), the `untiered` scores (greedy decode, held-out decode, follow-up time to first token)
  and the `gaps`. A missing field is unknown (null), never an error. `estimated` lists the KPIs whose
  `fraction` crit computed from `estimated_limits` (their `limit` stays Eval's null).
- `priorities`: `{p1, p2, p3, estimated}` lists computed from `needed` and the frozen fractions, or null with
  `priorities_reason` (no Eval yet, a missing limit, or a stop plan). `context_decode` counts toward
  the decode family, `context_ttft` toward prefill; a family's fraction is the lower of its members'.

  | Needed | p1 | p2 | p3 |
  | --- | --- | --- | --- |
  | Both, fractions within noise of each other | decode, prefill | ttft | |
  | Both, prefill further from its limit | prefill | decode | ttft |
  | Both, decode further from its limit | decode | prefill | ttft |
  | Only decode | decode | ttft | |
  | Only prefill | prefill | ttft | |
  | Neither | the plan is a stop plan | | |

  Within noise: the fractions differ by at most the larger of (relative noise × fraction) of the two
  families, each with the noise of the member that sets its fraction (unknown counts as 0).

  When both families are needed, every measured member of them whose limit neither Eval gives nor crit
  computes needs an estimated limit; crit refuses the plan naming the KPI otherwise. An estimate is
  physically consistent with the frozen score (a ceiling at or above a rate, a floor at or below a time),
  has a nonempty derivation, and never replaces a limit Eval gives. crit computes the missing fractions
  from Eval's limit, then its own computed limit, then the estimate; every check uses them like Eval's,
  `priorities.estimated` lists the KPIs whose estimate the ranking rests on and, in runs that compute
  limits, `priorities.sources` names the source (`eval`, `computed`, `estimate`) of each compared fraction.

#### Computed limits

In a run with `"plan_limits": "computed"`, when the parent's Eval states no limit for `decode` or
`context_decode`, crit computes one from the same attempt: per measured decode case (long-context
point) the measured `tokens_per_round` (1 where none is measured) over its pass floor, `(bytes_per_pass +
kv_bytes_per_token × (prompt_tokens + output_tokens / 2)) / bandwidth_gbs`, and their geometric mean.
`bytes_per_pass` is the attempt's (`headroom.bytes_per_pass`, the engine's or Eval's model-files figure),
else the run's model files read as Eval reads them (`crit_eval.limits.context.pass_bytes`). It uses the
acceptance measured so far, never full acceptance, and leaves the draft steps' own cost out, so it is
deterministic and an upper bound at that acceptance. `inputs.computed_limits[KPI]` keeps the `limit`,
`unit`, `formula`, the JSON paths of its `inputs` and whether it was `used`; a limit below the measured
score is kept with a `note` and not used. `inputs.fraction_sources` names each fraction's source. An
agent's estimate for a KPI crit computed is kept in `inputs.estimates_not_used` with its relative
`difference` from crit's limit and is not used.

The priorities measure the distance to what is reachable, not to the current strategy's own ceiling. In
these runs `strategy.alternative` may add `limits` for `decode` and `context_decode`: `{tokens_per_round,
draft_steps, derivation}`, from which crit computes the alternative's ceiling on the parent's Eval cases
(points), `tokens_per_round` over the pass floor plus `draft_steps` draft steps at `draft_bytes_per_step`
(the attempt's, else the checkpoint's draft layers, `crit_eval.limits.context.draft_step_bytes`) over the
bandwidth; or `{limit, derivation}` only where crit cannot compute that round floor. crit ranks decode and
`context_decode` by the higher of the current strategy's limit and the alternative's ceiling, whatever
`strategy.choice` says. `inputs.alternative_limits[KPI]` keeps the `limit`, how it was computed
(`computed_by` `crit` or `plan`, `tokens_per_round`, `draft_steps`, `round_floor_ms`, `inputs`, `formula`),
the `current_limit` and its `current_source`, and whether it `won`; a winning alternative is the source
`alternative` in `priorities.sources`. When the plan chooses the alternative, its outcome compares that
ceiling with the measured score (`alternative`). Runs without `plan_limits` refuse `limits`.

#### Plan rules 2

Runs with `"plan_rules": 2` check their plans with these changes to the rules above:

- `strategy.alternative.limits.KPI.tokens_per_round` is at most `draft_steps + 1`: a round of `draft_steps`
  draft steps emits at most its drafts and the target's own token.
- No noise is unknown. A KPI whose noise the Eval leaves unknown takes the largest relative noise that
  Eval knows among its KPIs, else 5%; `inputs.noise_default` keeps that `noise` and its `source` (the
  KPI, or `default`). `inputs.noise` stays as frozen.
- A family's noise is the largest of its members' with a fraction, whichever member sets the fraction.
- Within noise: the fractions differ by at most the two families' noise bands combined in quadrature,
  `√((noise_decode × f_decode)² + (noise_prefill × f_prefill)²)`.
- Check 4 compares each gap with its own KPI's noise, check 6 each lever change with its KPI's noise and
  check 9 the forecast with the KPI's noise, all with the default where Eval left it unknown.
- crit's computed limits and the alternative's ceiling follow Eval's cost model, the round floor: tokens
  per round over the pass floor with the tokens held plus the draft steps at Eval's draft floor, taken from
  `headroom.projection.draft_floor_ms`, else `headroom.draft_bytes_per_step` or
  `engine.draft_bytes_per_step` over the bandwidth, else the checkpoint's draft layers; `inputs` names
  the source (`draft_floor_ms`). The computed limit charges the engine's draft width (`draft_steps`,
  `draft_floor_ms`), the alternative its own `draft_steps`; both carry `"bound": "Eval's round floor,
  draft steps at Eval's draft floor"`, so the same strategy at the same tokens per round gets the same
  limit from either, and Eval's own limit where it gives one. Where the draft width is unknown (a plain
  engine, or a drafter that reports none) or no draft floor is known, the computed limit is the pass
  floor alone, `"bound": "upper bound, draft cost excluded"`.
- Each `strategy.alternative.limits` row names the `drafter` its ceiling assumes: `measured`, the drafter
  the parent's Eval measured, or `new`. A claim (`tokens_per_round` at `draft_steps` = k) is bounded:
  - k is at most the widest draft Eval projects, the parent's draft width or the protocol's
    `acceptance_ceiling_width` (8), whichever is larger, and `tokens_per_round` at most k + 1;
  - both drafters stay within the measured drafter's acceptance bound at k: per case
    `(1 - a^(k+1)) / (1 - a)` at its acceptance rate a, geometric mean over the cases, the projection
    behind Eval's acceptance bound. For `decode`, a is the rate Eval's acceptance bound uses (the first
    draft's conditional acceptance over the decode suite's measured rounds, the greedy suite's where it is
    higher). For `context_decode`, a is each long-context point's measured tokens per round at the draft
    width inverted to a rate; where the width is unknown, the points' measured tokens per round caps the
    claim. A new drafter is assumed to accept like the measured one;
  - a new drafter is bounded by its own measured acceptance instead when the row cites another
    candidate's completed Eval attempt retained in the run that measured it, `acceptance_evidence`
    (`candidates/<candidate>/eval/<attempt>.json`, not the parent's; crit records its path, SHA-256 and
    source, and refuses an attempt without a draft width and per-round accepted counts);
  - without a measured drafter (a plain baseline, or no per-round accepted counts) the physical bound
    applies, with a `note`.
  No row gives a bare `limit` in these runs: where crit cannot compute the round floor (no bandwidth or bytes
  per pass), the alternative does not rank the families. Where the parent's Eval measured no draft floor and the
  checkpoint has no draft layers, a separate drafter's draft cost comes from the companion drafter
  `recon --drafter` recorded with its `pass_bytes` (what one drafter pass reads, its output head included):
  ceil(k / drafts_per_pass) passes at the bandwidth, so a block drafter's one pass prices its whole block. With
  several recorded, the row names one by its recorded path (`companion`). Each entry keeps the `drafter`, the claim's
  `implied_acceptance` (the constant rate a at which k draft steps emit its tokens per round), and the
  `acceptance_bound` (`tokens_per_round`, `draft_steps`, `widest`, `source`, and the `rates` or the `cap`)
  or the `acceptance_evidence`.

Known limits of plan rules 2, checked only after the build by the plan's outcome (the forecast next to
the measurement, an alternative's ceiling next to its measured score):

- A measured drafter whose Eval has no per-round accepted counts keeps the physical bound.
- `estimated_limits` have no upper bound beyond being at or above the measured score.
- A family marked not needed with a `limit` cause is not checked further: its cause is free text.
- Check 3 applies to the run's first plan too: it builds only with decode or prefill needed and stops
  only with neither.
- Near the limit (`NEAR_LIMIT` = 0.85 in `crit.run.plans`): when both families are needed and both
  fractions are 0.85 or above, a near tie no longer puts both in p1; the family with the lower fraction is
  p1 alone (decode on an exact tie), the other p2 and time to first token p3.
- A family marked not needed with a parent Eval and a fraction below 0.85 answers why the rest of its
  limit is out of reach: a `gaps` entry `{"kpi": "decode" or "prefill", "gap": "limit", "answer": "cause",
  "detail": the physical cause}`. Such an entry names no Eval gap and must answer `cause`.

#### Plan rules 3

Runs created from this version on have `"plan_rules": 3`: the rules 2 above and these.

- Check 12, at record time: a build plan whose forecast expects more gain on p2 than on p1 (the gain
  summed per tier, as the outcome's `forecast_on_p2` sums it), or forecasts a p1 KPI below the parent's
  score (above it for a time), says why in `forecast_justification`; crit records it and lists what it
  answers in `inputs.forecast_checks`. Without it the plan is refused.
- Below half the limit (`MEASURED_ONLY` = 0.5 in `crit.run.plans`) a family is marked not needed only on
  measured rejections. Every `gaps` entry on it names its `evidence`, a file retained in the run (crit
  records its path and SHA-256 in `inputs.gap_evidence`), and an Eval gap answers `rejected`: a cause
  alone, such as "the quality budget", no longer stands. For prefill, every prompt activation format the
  parent's Eval prices above the engine's own declared `prompt_activations` (crit-eval's
  `kpis.prefill.activations`) is rejected too, unless Eval's own quality evidence rules it out: a `gaps`
  entry `{"kpi": "prefill", "gap": "activation:FORMAT", "answer": "rejected", "evidence": RESULT, "detail"}`
  whose evidence is a crit-eval result that measured that format (its engine's `prompt_activations`) and
  kept some quality set over its budget at the 5th percentile of its spread over positions
  (`budget_spread`). A rejection inside the quality noise is not a rejection, and a result without the
  spread cannot give one. An engine that declares no prompt activations cannot mark prefill not needed
  there: Eval names neither the format in use nor the faster ones.
- recon.md's `Drafting plan for both decode modes` and `Prefill arithmetic plan` are written, neither
  missing nor starting with TODO, before the first candidate, and its `Placement plan` too where
  `recon.json` records a placement scenario (`gathered.placement`, from the request's device). The prefill plan gives each matrix group's
  prompt activation format and the quality measurement per group (`crit eval run --attribute-drift`).

#### Plan rules 4

Runs created from this version on have `"plan_rules": 4`: the rules 3 above and these. crit records each
plan's `inputs.objective` (`crit.run.plans.objective`): the candidate's number in the run (the run's
candidates so far plus one), its objective's name and rule, and `min_gain` where it applies.

- Candidate 1, `proof_of_concept`: the first engine that works end to end (it serves, passes the
  functional gates and the quality gate at the required context).
- Candidates 2 and 3, `crit_round`: improve the parent as much as possible with crit's toolkit, Inference
  techniques and Kernels Hub kernels, each measured against its floor and fixed or replaced where it trails.
- Candidates 4 on, `critique_round` (check 13): the plan names the parent's two weakest terms in
  `weakest`, `[{kpi, term, evidence, levers}]`: what is slow with its measured time and floor, a file in
  the run that measured it, and the ids of the plan's speed levers that improve its KPI's family. The plan
  has at least one other speed lever beside those.
- Candidates 2 to 5 (check 13): the forecast's expected gain on some p1 KPI over the parent's Eval is at
  least `MIN_GAIN` (10%); a plan below it is refused, with no justification that lets it pass.
- Check 13, providers: every speed lever names what supplies its code in `provider`: `inference:ENTRY`,
  `hub:ORG/NAME` (a Kernel Hub repository), `lib:NAME` (a library, [Kernels' sources](../kernels/SOURCES.md))
  or `own: ...` (the engine's own code). An `own` lever carries `yardstick` `{evidence, detail}`: a file in the
  run with the measurement of the best existing provider for the operation, or the providers it checked and why
  none fits.
- Check 14, the review: once the parent's plan has an outcome at the Eval attempt this plan freezes
  (`plans/PLAN_ID/outcomes/ATTEMPT.json`, written by `crit run ... eval`), the plan reviews it in `review`:
  `{plan_id, kpis, levers, lesson}`. `kpis` gives `{cause, evidence}` for every KPI the outcome measured
  short of its forecast's low (above its high for a time), the evidence a file in the run; `levers` says
  what each lever the parent's proposal built, pruned or deferred delivered (`{result}`); `lesson` what
  this plan changes because of it. crit records what it checked in `inputs.review` (the outcome's path
  and SHA-256, the missed KPIs, the levers by fate). The parent plan's largest p1 lever (its largest
  forecast change on a p1 family), when pruned or deferred, is planned again under the same id or its
  `levers` entry adds `evidence` and `structure`: the measurement of another kernel structure or a vendor
  library at the same arithmetic that rejected it, never variants of one kernel.

crit checks completeness and consistency with the frozen facts, never whether the reasoning is good,
and refuses a plan with a message that says what to fix:

1. Fields, types and text limits; unknown keys are refused. `eval-gap:` sources and `gaps` entries name
   gaps of the parent's Eval.
2. `parent_candidate_id` exists and has a completed Eval attempt, unless the run has no Eval yet; the
   plan freezes the newest.
3. `stop` requires decode and prefill both not needed; `build` requires at least one needed.
4. For each KPI marked not needed, every Eval gap in its family whose `projected_gain` exceeds the
   family's noise (every positive gap when the noise is unknown) has an entry in `gaps`.
5. A build plan names at least one speed lever, each with at least one KPI change.
6. A lever whose positive changes are all within their KPIs' known noise shares a `group` with a lever
   that has a change above the noise.
7. Every needed KPI in p1 has a lever with a positive change on its family.
8. Some lever improves decode, prefill, `context_decode` or `context_ttft`: no plan targets time to
   first token alone.

Requirement levers are exempt from checks 5 and 6 and never count toward checks 7 and 8.
9. A build plan's `forecast` covers every required KPI with `0 < low ≤ expected ≤ high` and, with a
   parent Eval, forecasts decode or prefill better than the parent's score by more than its noise
   (unknown noise: any improvement). A stop plan's forecast may be null.
10. `strategy.current` and `strategy.alternative` are both present and nonempty.
11. `supersedes` names a plan of the same segment and parent that is neither linked to a candidate nor
    already superseded.

The run's first plan (no parent) skips 3, 4, 6, 7 and 9's improvement test, may have no levers (it
imports a baseline as is) and may forecast null.

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
of a `crit run eval` attempt on the current package under the run's current quality terms:
`arithmetic` by `quality` and `sampling`; `functional` by `coverage`, `repeatable`, `request_history`,
`prompt_cache`, `retrieval` and `lower_context` (and `exactness` when the run requires it); `resources`
by `capacity` and the device memory left free at the required context against `vram_headroom_bytes`
(host memory comes from `measure`). A failed Eval gate fails the check, an unmeasured one leaves it
`unverified`; the check cites the result at `artifact_verified`.

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
candidate cannot be selected.
A sealed candidate cannot be fixed, so `benchmark` and `seal` warn, and `select` names
what blocks eligibility.

Each metric is `{unit, samples}` with exactly the declared repetitions:

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

Absent required metrics need reasons in `missing_metrics`; missing is never zero.
Keep metrics paired and every sample. The reader computes medians grouped by exact
model/input/hardware/environment, case, protocol and correctness policy (contexts,
concurrency and cold/resident modes group separately). Eligibility requires the fixed
gates and evidence level; target satisfaction checks all retained benchmarks across
required cases, not the fastest series. Served request timing never enters projections.

### Measured benchmarks

`crit run measure` measures the run's cases on a candidate as a client of `crit serve`
gets it. Each case keeps its chat request as a retained JSON file
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

### Stop

Supply `reason`, `critique` (the engine and remaining opportunities), `opportunities`
(ranked list), `cleanup` (`external`, `settled` or `unresolved`), optional `message` and
`headroom`. Reasons: `satisfied` (needs an eligible selected candidate meeting the
targets), `search_exhausted`, `blocked`, `budget_exhausted` (refused while the segment
has fewer candidates than `candidate_budget`; once it is used, `candidate` is refused
until a resume), `cancelled`, `interrupted`.

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

Stopping early is a plan: in a run that requires plans, `search_exhausted` while the segment still has
budgeted candidates requires that the segment's newest unsuperseded plan is a stop plan whose `inputs`
froze the newest completed Eval attempt of the newest candidate that has one. With the budget used up,
no stop plan is needed.

### Origin engine

`crit run create --record request.json --from-engine ENGINE_DIR` starts a run from an engine listed in
crit-hub: `ENGINE_DIR` is its `engines/<owner>/<name>/` folder in a crit-hub checkout (`card.yaml`,
`card.lock.json`, `evidence/`; reading the card needs PyYAML, as crit-hub's tools do). The request's
`origin_engine` then records the engine (`<owner>/<name>`, title, origin, contributor), the repository,
commit and folder of its source, the card's and lock's hashes, every checkpoint it was measured on with
each file's size and SHA-256, its sealed provenance and its published crit Eval results, which are copied
into the run under `captured/origin/` as the numbers before the run's work. The agent request tells the
agent to check the source out at that commit and make it the first candidate, declared with
`"lineage": "origin-engine"` in its proposal and measured before any change, so it is the baseline. At
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
    package/manifest.json       runnable files, build and launch identities
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
margin): every file written by then, nothing later. `--before PLAN` ends the copy where that plan's record was made
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
Changing criteria, model or workload needs a new run.
