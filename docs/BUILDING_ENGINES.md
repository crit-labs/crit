# Build and publish an engine

Use [Core](../core/README.md) for runtime contracts, [Inference](../inference/README.md)
for techniques, [Kernels](../kernels/README.md) for kernels and [Eval](eval/README.md)
for common speed, quality and exactness checks. Agents start with
[AGENTS.md](../AGENTS.md). All model-specific source, dependencies, builds and results
belong in the run workspace (default `CRIT_ROOT/runs/`). [Run records](RUN_RECORDS.md)
defines every payload below; replace uppercase placeholders, and put global options
before the action.

## 1. Define the workload

Agree on model/encoding, hardware/OS, prompt/output sizes, concurrency, cold or
resident mode, correctness criteria, objectives, headroom, candidate budget, attempts
and provider spend. Preflight models, compilers, Eval requirements and dependencies.
`candidate_budget` counts complete candidates; there is no time budget, only a 30-day
safety window.

Every candidate serves Eval's [max context](eval/CONTEXT.md) (model positions, capped by
the KV cache that fits). Probe the device (`crit eval probe --out device.json`, or
`device` in `settings.toml`) and `create` freezes it as `kv_tokens`; set `kv_tokens`
only when the user asks.

```json
{
  "model_path": "ABSOLUTE_LOCAL_MODEL_PATH",
  "correctness_criteria": "REPLACE_WITH_FIXED_CRITERIA",
  "device": "ABSOLUTE_PATH_TO_DEVICE_JSON",
  "vram_headroom_gib": 1,
  "ram_headroom_gib": 2,
  "candidate_budget": 3,
  "attempt_limit": 32,
  "verification_level": "artifact_verified",
  "workload": {
    "prompt_tokens": 512, "output_tokens": 128, "concurrency": 1,
    "repetitions": 3, "warmup_runs": 1, "mode": "resident"
  },
  "targets": {}
}
```

These values are a template, not acceptance criteria; use `cases` for multiple workloads.

```text
crit run --project CRIT_ROOT --root RUN_ROOT create --record request.json
```

Read the returned run's `segments/segment-0001/agent_request.md`. Continue a supplied
run instead of creating a duplicate. Provider credentials and spend caps are
configured separately.

## 2. Freeze inputs and source

Retain workload files under `RUN_ROOT/RUN_ID/captured/` with nonempty hardware and
environment manifests and path/size/SHA-256 manifests for every case. The coordinator
verifies them and hashes the model.

```text
crit run --project CRIT_ROOT --root RUN_ROOT --id RUN_ID resolve --record inputs.json
crit run --project CRIT_ROOT --root RUN_ROOT --id RUN_ID candidate --source SOURCE_DIR --record proposal.json
```

The proposal gives label, rationale, strategy, configuration and Inference decisions
(`implemented`, `inherited`, `rejected`, `untested`) in `store_decisions`, plus
parent/origin IDs when reusing earlier work. The coordinator allocates IDs and copies
the source; build in the run's `work/`. Changed source, flags, dependencies or
configuration need a new candidate.

## 3. Build, check and compare

Reserve each process **before** launch (`reserve` consumes an attempt but launches
nothing), supervise the process tree to the returned deadline, then settle
descendants, logs and cleanup, failures included.

```text
crit run --root RUN_ROOT --id RUN_ID reserve --candidate CANDIDATE_ID --operation build --seconds PROCESS_CAP
crit run --root RUN_ROOT --id RUN_ID settle --reservation RESERVATION_PATH --record result.json
crit run --root RUN_ROOT --id RUN_ID package --candidate CANDIDATE_ID --source PACKAGE_DIR --record recipe.json
crit run --root RUN_ROOT --id RUN_ID check --candidate CANDIDATE_ID --record check.json
crit run --root RUN_ROOT --id RUN_ID benchmark --candidate CANDIDATE_ID --record benchmark.json
```

Run these from the checkout or add `--project CRIT_ROOT`; use the exact reservation
path returned. The package holds runnable files, dependencies and notices, without
`manifest.json` (crit creates it). Build/launch recipes are literal arguments, without
shell expansion.

| Check | What to retain |
| --- | --- |
| Baseline | Requests through Core admission, session, commit and retirement; exact source/package identities. When continuing earlier work, re-record the previous best: the same package can measure several percent differently in another session |
| Strategy | Measured bottleneck; alternatives with estimated ceilings; Inference code used and the hooks your engine supplies |
| Headroom | `crit eval status`: each KPI and the long-context curve against its floor or ceiling. Re-measure every candidate and on resume |
| Correctness | Separate arithmetic, functional and resource gates with fixed criteria; failures stay failed |
| Repeatability | Same output in a fresh process ([recorded selection](../inference/recorded-selection/README.md)) |
| Quality | When arithmetic changes, perplexity and KL divergence from a reference ([full-vocabulary comparison](eval/MEASUREMENTS.md#full-vocabulary-comparison)). Under `weights_policy: as_stored`, requantized or pruned weights don't qualify and each candidate declares `weights` ([run records](RUN_RECORDS.md)) |
| Performance | Matched inputs/protocol, warmups, every repetition, prefill/decode/request/load costs, physical memory, plus [`crit eval`](eval/README.md); tune text-dependent settings such as draft width on its mixed cases |
| Coverage | Representative contexts and request shapes; operator tests alone don't qualify a model |

Preserve representation, ownership and readiness invariants when specializing or fusing
technique code, and keep model policy out of Core (expert caches:
[expert residency](../inference/expert-residency/README.md#validation)). Measure cold and
changing requests, not only warm repeats; charge initialization, transfers,
synchronization and refresh to the measurement they serve; keep polling out of timed
paths; define aggregation first. One fast sample is not a result.

Choose implementations (kernel variants, library algorithms) from a table recorded in
the candidate configuration, not by timing at load: near-ties resolve differently per
process and can change outputs. For speculative decoding, record
`accepted_tokens_per_round` with decode speed.

### Run the common Eval suite

Retain an [Eval adapter](eval/ADAPTER.md) for the candidate's engine with its source.
Every candidate of a model uses the same prepared data, reference and pinned
continuation (`crit eval pin`, with the adapter's `replay`). Keep data and results in
the run workspace and run evaluation under the same reservation and settlement.

Pass the run's `quality_budget`, `exactness` and `kv_tokens` to `crit eval run` as
`--quality-budget`, `--exactness` and `--context` (`--context` also to `pin`, `ab` and
`kernels`). Run the full suite for the final comparison; retain raw results, settings,
inputs and the Eval revision; report skipped cases and overrides. Confirm small
differences with `crit eval ab`. Start kernel work from `crit eval kernels`, checking
[Kernels](../kernels/README.md) first.

Eval results are evidence, not run records, and Eval's drift budget does not replace
the agreed gates. See [Eval results](eval/README.md#results-and-status) and
[Eval evidence](RUN_RECORDS.md#eval-evidence).

## 4. Make the package servable and seal

Before sealing, implement the [resident adapter](SERVING.md#adapter-contract)
(`chat-json-v1`, `chat-resident-v1`, `create_engine(context)`, model allocations reused
across requests). Verify with `crit serve`: consecutive requests, a leading system
message, streaming, a stop string, A → B → A switching, cancellation and Ctrl+C, with
real tokenizer counts and synchronized phase times. Declare `chat-config-v1` so the host
rejects an adapter that drops or changes a configuration key. For `response_format`,
apply Core's [structured-output service](../core/src/constraints/README.md) while
sampling and declare `chat-constraints-v1`
([structured output](SERVING.md#structured-output-for-adapters)).

```text
crit run --root RUN_ROOT --id RUN_ID seal --candidate CANDIDATE_ID --lifecycle completed
```

Use `built` for a servable engine without benchmarks, or the actual failure lifecycle.
Sealing needs settled attempts and closes the candidate permanently. A failed gate
does not prevent serving, but prevents qualification.

## 5. Deliver

Plan each candidate as [AGENTS.md](../AGENTS.md) describes, aiming at every Eval goal.
The stop record accounts for each goal ([stop](RUN_RECORDS.md#stop)). Stop for the
objective, exhausted search, a blocker, cancellation or a used candidate budget
(`budget_exhausted` is refused while budgeted candidates remain); `resume` adds candidates.

```text
crit run --root RUN_ROOT --id RUN_ID stop --record stop.json
crit run --root RUN_ROOT --id RUN_ID resume --candidates N
crit serve --workspace RUN_ROOT --run RUN_ID --project CRIT_ROOT --write-launchers
```

Save `summary.md` at the run root (contents listed in [AGENTS.md](../AGENTS.md)) and name
the candidate to open, or say none qualified. Resume needs settled cleanup and keeps
prior attempts, criteria and sealed results ([recovery](RUN_RECORDS.md#recovery-and-resume)).
