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
context_tokens = 262144             # the context wanted, prompt plus output (default 262,144)
```

`crit eval run` and `crit eval context` read `vram_headroom_gib`, `context_reserve_gib` and
`context_tokens` from it when their flags are omitted. The reserve holds activations, workspaces, graphs
and speed techniques; a larger one trades context for that room.

| Action | Arguments / effect |
| --- | --- |
| `create` | `--record request.json [--from-engine ENGINE_DIR]`; new run and first segment; with an engine folder of a crit-hub checkout, the run starts from that engine ([origin engine](#origin-engine)) |
| `resolve` | `--record inputs.json`; freezes verified model and input identities |
| `candidate` | `--source DIR --record proposal.json`; allocates an ID, retains source |
| `reserve` | `--operation discovery/build/check/benchmark [--candidate ID] [--seconds N]`; consumes an attempt before launch; only discovery omits the candidate |
| `settle` | `--reservation RUN_RELATIVE_PATH --record result.json`; exit, logs, cleanup |
| `package` | `--candidate ID --source DIR --record recipe.json`; runnable files and manifest |
| `check`, `benchmark` | `--candidate ID --record payload.json`; appends evidence |
| `measure` | `--candidate ID [--seconds N]`; reserves a benchmark attempt, serves the packaged candidate through `crit serve`, streams every resident case's chat request to it, settles it and records one benchmark per case ([measured benchmarks](#measured-benchmarks)) |
| `eval` | `--candidate ID --data EVAL_DATA_DIR [--only PART] [--baseline RESULT] [--label TEXT] [--seconds N]`; reserves a benchmark attempt, serves the packaged candidate through `crit serve`, scores the data's quality texts on the run's model unless stored, runs Eval over the chat API with the run's quality terms and `kv_tokens`, settles it and retains the result as `candidates/ID/eval/ATTEMPT.json` ([Eval evidence](#eval-evidence)) |
| `seal` | `--candidate ID --lifecycle STATE [--failure TEXT]`; closes the candidate |
| `select` | `--candidate ID`; selects a sealed eligible candidate |
| `stop` | `--record stop.json` |
| `resume` | `--candidates N [--attempt-limit N] [--external-agent-stopped]`; new segment for N more candidates (`--minutes N` alone: legacy time-budget segment). `--quality-budget JSON`, `--exactness MODE`, `--approved-by NAME` amend quality terms from this segment on, recorded as the segment's `amendment` |
| `cost` | `--transcript PATH [--transcript PATH ...] [--agent NAME] [--prices JSON] [--until TIMESTAMP]`; accounts the run's build cost from the agent's transcripts and writes `cost.json` ([cost](#cost)) |
| `show`, `recover` | Inspect; or prove abandoned ownership and rebuild active projections |

## Payloads

CLI input objects. The coordinator adds record headers, owner IDs and generated
hashes; add them only where a field below requires one.

### Request

| Field | Meaning / default |
| --- | --- |
| `model_path` | Required existing local model file/directory |
| `correctness_criteria` | Required fixed criteria; gates are `arithmetic`, `functional`, `resources` |
| `device` | A `crit eval probe` result. Without `kv_tokens`, `create` freezes Eval's [max context](eval/CONTEXT.md) on it as `kv_tokens`, with the basis (limits, KV precision, memory, device file sha256) in `hardware_constraints.context` |
| `kv_tokens` | Capacity for prompt + output; overrides the max context. One of `kv_tokens`, `device` is required |
| `context_tokens` | The context the user wants (default 262,144, from `settings.toml` when the request omits it). With `device`, `kv_tokens` is the smaller of it, the model's positions and the KV cache that fits; `hardware_constraints.context` records `requested`, the `bound` (`requested`, `model` or `device`) and a `note` when the model or device allow less |
| `ram_headroom_gib`, `vram_headroom_gib` | Physical free-memory requirements; defaults 2 and 1 |
| `context_reserve_gib` | Device memory kept out of the KV cache when deriving `kv_tokens` from `device`; default the protocol's 3 |
| `candidate_budget`, `attempt_limit` | Segment bounds: candidates to build (default 3, 1–64) and attempts (default 64, 1–1000). `budget_minutes` is a safety window, default and maximum 43200 (30 days). `budget_minutes` without `candidate_budget` creates a legacy time-budget segment (1–180) |
| `workload` | Defaults: prompt 512, output 128, concurrency 1, repetitions 3, warmups 1, mode `resident` |
| `cases` | Optional 1–32 distinct `{case_id, workload}` objects with complete workloads |
| `benchmark_protocol` | Fixed protocol identity; default `crit-matched-v1` |
| `verification_level` | `reported` or `artifact_verified` (default) |
| `targets` | Optional positive `decode_tps`, `prefill_tps`, `request_ms` thresholds |
| `weights_policy` | Optional `as_stored` (supplied weights used as stored) or `transformable`; omitted means unspecified |
| `quality_budget` | Optional drift budget against Eval's reference: any of `ppl_ratio_max` (≥ 1), `kld_mean_max`, `kld_p99_max` (≥ 0); defaults 1.05, 0.05, 0.5. Agents spend it on speed |
| `exactness` | `reported` (default: speculative output may differ from plain greedy; the drift budget judges it) or `required` (any difference fails Eval's gate) |
| `eval_sampling` | Optional, the user's only: `{settings, approved_by}`, settings `temperature` (> 0, required) and any of `top_p`, `top_k`, `min_p`, as an object or `"temperature=0.6,top_p=0.95"`. Eval's sampled suite then measures at these settings instead of the model's `generation_config.json`: `crit run ... eval` requires data prepared with `crit eval prepare --sampling` and the same settings (and refuses overridden data without it), recon applies them to the served default, and results label them as an override |
| `origin_engine` | Written by `create --from-engine`, never by hand: the crit-hub engine the run starts from ([origin engine](#origin-engine)) |

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
| Proposal | `label`, `rationale`, `strategy`, `configuration` object, `store_decisions` object; optional `parent_candidate_id`, `origin_candidate_id` (earlier retained candidates); `weights` (`as_stored` or `transformed`, required when `weights_policy` is `as_stored`); `lineage` (`origin-engine`: the candidate is the origin engine's source as imported, in a run created with `--from-engine`). A run whose model is a folder needs `work/recon.json` (`recon`) before its first candidate; crit adds its `frozen` block as `configuration.recon` (a supplied `recon` must equal it) and `extensions.recon` with its path and SHA-256 |
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
0, `max_tokens` the workload's `output_tokens` and usage. `extensions.sampling` records the
sampling sent. `crit serve`'s usage must count the workload's
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

[Eval](eval/README.md) result JSON is not a `check` or `benchmark` payload and is
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
`search_exhausted` and `budget_exhausted` require decode, prefill and ttft. The account
is recorded in the `run_stopped` event.

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
