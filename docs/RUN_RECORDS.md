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
uses when the request omits them; unknown keys are an error:

```toml
wsl_run_root = "/llm/engines/crit"  # folder for work that must run under WSL; empty: none
device = "device.json"              # crit eval probe result (relative: from CRIT_ROOT)
ram_headroom_gib = 2                # host memory free at the engine's peak (default 2)
vram_headroom_gib = 1               # device memory free at the engine's peak (default 1)
context_reserve_gib = 3             # device memory kept out of the KV cache (default 3)
```

`crit eval run` and `crit eval context` read `vram_headroom_gib` and `context_reserve_gib`
from it when their flags are omitted. The reserve holds activations, workspaces, graphs
and speed techniques; a larger one trades context for that room.

| Action | Arguments / effect |
| --- | --- |
| `create` | `--record request.json`; new run and first segment |
| `resolve` | `--record inputs.json`; freezes verified model and input identities |
| `candidate` | `--source DIR --record proposal.json`; allocates an ID, retains source |
| `reserve` | `--operation discovery/build/check/benchmark [--candidate ID] [--seconds N]`; consumes an attempt before launch; only discovery omits the candidate |
| `settle` | `--reservation RUN_RELATIVE_PATH --record result.json`; exit, logs, cleanup |
| `package` | `--candidate ID --source DIR --record recipe.json`; runnable files and manifest |
| `check`, `benchmark` | `--candidate ID --record payload.json`; appends evidence |
| `seal` | `--candidate ID --lifecycle STATE [--failure TEXT]`; closes the candidate |
| `select` | `--candidate ID`; selects a sealed eligible candidate |
| `stop` | `--record stop.json` |
| `resume` | `--candidates N [--attempt-limit N] [--external-agent-stopped]`; new segment for N more candidates (`--minutes N` alone: legacy time-budget segment). `--quality-budget JSON`, `--exactness MODE`, `--approved-by NAME` amend quality terms from this segment on, recorded as the segment's `amendment` |
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

Workload keys: `prompt_tokens`, `output_tokens` (1–1,000,000), `concurrency` (1–256),
`repetitions` (1–100), `warmup_runs` (0–100), `mode` (`resident`/`cold`). KV capacity
is 1–2,000,000 tokens and must cover every case. Per-process ceilings: discovery 180 s,
build 1800 s, check 3600 s, benchmark 1800 s, configured agent 10800 s, all clipped to
the remaining segment window and requested cap. Provider spend limits are the caller's
responsibility.

### Inputs, proposal and package

| Payload | Fields |
| --- | --- |
| Inputs | Nonempty `hardware_manifest`, `environment_manifest`; `case_manifests` mapping **every** fixed case ID to an ordered nonempty list of `{path, bytes, sha256}` under the run's `captured/` |
| Proposal | `label`, `rationale`, `strategy`, `configuration` object, `store_decisions` object; optional `parent_candidate_id`, `origin_candidate_id` (earlier retained candidates); `weights` (`as_stored` or `transformed`, required when `weights_policy` is `as_stored`) |
| Package recipe | `build: {toolchain: {...}, dependencies: [...], command: [...]}`, literal `launch` argument list, `capabilities` list |

In an `as_stored` run, a candidate declaring `transformed` weights (requantized,
pruned or changed) is measured but never eligible. The coordinator cannot inspect
weights; declare honestly. `as_stored` covers the target weights that compute every
published token. It does not cover draft-only weights (a drafter may use any encoding
when stored target weights verify every token; declare `as_stored` and record the
drafter encoding in the configuration; drafter values come from the checkpoint or a
published companion and are never trained or fine-tuned) or runtime state (KV-cache
and activation precision, judged by the quality gates).

`store_decisions` maps Inference technique names to `{status, reason}`, status one of
`implemented`, `inherited`, `rejected`, `untested`. IDs are coordinator-assigned and
never recycled; labels don't determine paths or ordering.

Package dependencies give `name`, `version`, package-relative `path` and exact-byte
`sha256`; `launch[0]` names a retained relative file. Omit `manifest.json` (crit
generates it); crit writes `configuration.json` when absent, and a supplied one must
equal the configuration. On POSIX, files identical to retained source are hard-linked,
so changing either fails verification. Keep dependencies, notices and build inputs in
the package. Serving capabilities (`chat-json-v1`, `chat-resident-v1`, optional
`chat-stream-v1`, `chat-config-v1`, `chat-constraints-v1`) are in
[Serving](SERVING.md#adapter-contract).

### Attempt result

Required: `outcome`, `ended_at`, `exit_code`, `cleanup_settled` (boolean), `logs`
(`{path, sha256}` list), `failure` (message or null). Outcomes: `completed`, `failed`,
`cancelled`, `timed_out`, `interrupted`, `launch_failed`. Completion needs exit zero,
settled cleanup and an end within the reservation. Every attempt counts against the
limit; the candidate's lifecycle follows its latest attempt, and earlier failures stay
recorded. Reservations only record deadlines; the caller supervises processes and
descendant cleanup. A published reservation must be settled before `seal` or `resume`.

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

### Eval evidence

[Eval](eval/README.md) result JSON is not a `check` or `benchmark` payload and is
never imported automatically. Keep the adapter and inputs with the candidate, retain
the result JSON under it before sealing and reference its bytes from a check's
`evidence`. Publish benchmark samples only under cases and a protocol frozen for them;
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

## Files and identities

```text
run-ID/
  run.json, inputs.json          fixed request and resolved inputs
  state.json                    replaceable run projection
  segments/segment-NNNN.json     bounds, counters and deadline
  segments/segment-NNNN/         agent_request.md and discovery/agent attempts
  events/event-NNNNNNNN.json     ordered events
  captured/                     retained workload files
  work/                         mutable build scratch
  candidates/candidate-NNNN/
    candidate.json              fixed proposal/configuration
    source/, source.manifest.json
    package/manifest.json       runnable files, build and launch identities
    attempts/attempt-NNNN/      reservation.json and result.json
    checks/, benchmarks/, logs/
    telemetry.json             active projection
    outcome.json, seal.json     final outcome and complete inventory
  chats/                       separate requests, replies and conversations
```

Records are UTF-8 JSON with unique keys, finite numbers, integer `schema_version: 1`,
the correct `record_kind`, owner IDs and timezone-aware UTC timestamps. Projections
have increasing integer revisions. Optional extensions go in `extensions`; invalid
versions/fields are rejected. Limits: **1 MiB per JSON file, 512 candidates per run,
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

## Recovery and Resume

A per-operation coordinator lease serializes writers. Immutable records publish
atomically and exclusively, projections by atomic replacement, with no cross-file
transaction; readers wait for consistent revisions. The lease names the owner's PID,
host and OS. A terminated `crit run` (SIGTERM) releases it; a killed one leaves it for
`recover`, which verifies a dead owner, keeps valid orphaned evidence and rebuilds
**active** projections, never rewriting seals, inventing records, resetting counters or
starting processes. Live or unproven owners and corrupt identities block it, and it
must run on the owner's host and OS (WSL and Windows process IDs differ).

Before sealing or resuming, settle every reservation and external process tree; a
vanished coordinator does not prove an agent stopped. `resume` keeps the request and
results, appends fresh bounds and requires sealed candidates plus cleanup confirmation.
Changing criteria, model or workload needs a new run.
