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
After `create`, gather the model's sampling defaults, stop tokens, chat templates and
`config.json` fields, then finish the scaffolded `work/recon.md` from the model card and
reference code ([AGENTS.md](../AGENTS.md)):

```text
crit run --project CRIT_ROOT --root RUN_ROOT --id RUN_ID recon
```

`candidate` refuses a run whose model is a folder until recon has run; a single-file
model needs none. `work/recon.json` names every file it read with its SHA-256. Its `frozen` block
(`sampling`, `chat_template`, `template_variables`, `stop_token_ids`, `stop_strings`)
goes into each later candidate's configuration as `recon`; edit the block, with the
reason in `recon.md`, before the candidate it should change. Its `sampling` is the
exception: it is the model's own and stays so (a candidate refuses another). `recon.md`'s
"Eval sampling" section names the sampling crit Eval's sampled suite measures, from the
model's `generation_config.json` or the protocol default; only the user overrides it,
with `eval_sampling` at `create` (and `crit eval prepare --sampling` with the same
settings), and the served default then follows the override. Finish recon.md's drafting
plan for both decode modes before the first candidate. `--force` gathers again and keeps
`recon.md`. `recon.json` also carries the checkpoint's fingerprint (architecture, weight layout,
tokenizer and template, each hashed), which every candidate records; `recon --compare OTHER_MODEL`
compares the run's model with another checkpoint and says which sections differ and what that
means for the engine (an equal fingerprint: weight values only, so the engine loads it unchanged).
`candidate_budget` counts complete candidates; there is no time budget, only a 30-day
safety window.

To improve, port or adapt an engine that crit-hub already lists, pass its folder in a crit-hub
checkout to `create --from-engine ENGINE_DIR` ([origin engine](RUN_RECORDS.md#origin-engine)). The
run records the engine, its source pin and the checkpoint it was measured on, keeps its published Eval
results under `captured/origin/`, and `resolve` says whether the run's model is that checkpoint. Check
the source out at the pinned commit and make it the first candidate, with `"lineage": "origin-engine"`
in the proposal, measured before any change: that is the baseline every later Eval compares against.

Every candidate serves Eval's [max context](eval/CONTEXT.md): the context the user wants
(`context_tokens` in `settings.toml`, default 262,144), capped by the model's positions and
the KV cache that fits. Probe the device (`crit eval probe --out device.json`, or `device`
in `settings.toml`) and `create` freezes it as `kv_tokens`, with why it is that size in
`hardware_constraints.context` and recon.md's Required context section; set `kv_tokens` or
`context_tokens` only when the user asks.

```json
{
  "model_path": "ABSOLUTE_LOCAL_MODEL_PATH",
  "correctness_criteria": "REPLACE_WITH_FIXED_CRITERIA",
  "device": "ABSOLUTE_PATH_TO_DEVICE_JSON",
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
`model_path` is the model as the user has it: a Hugging Face model folder, or a GGUF file (or a
folder holding one), whose config, tokenizer and chat template crit derives from its metadata
([GGUF models](eval/README.md#gguf-models)).
Free-memory headroom comes from `settings.toml` (copy `settings.example.toml`; `vram_headroom_gib`,
`ram_headroom_gib`; defaults 1 and 2 GiB), and `create` freezes it. A value in the
request replaces the machine's setting, so add one only when the user asks for it.

```text
crit run --project CRIT_ROOT --root RUN_ROOT create --record request.json
```

Read the returned run's `segments/segment-0001/agent_request.md`. Continue a supplied
run instead of creating a duplicate. Provider credentials and spend caps are
configured separately.

## 2. Freeze inputs and source

Retain workload files under `RUN_ROOT/RUN_ID/captured/` with nonempty hardware and
environment manifests and path/size/SHA-256 manifests for every case. The coordinator
verifies them and hashes the model. Keep each case's request as a JSON file
`{"messages": [...], "tools": [...]}` (`tools` optional) so `crit run measure` can
benchmark it through `crit serve`. Its messages must render to the workload's
`prompt_tokens` as the candidate renders them:
`crit_core.text.load_model_text(MODEL_DIR, clock=None).encode_chat(messages, tools=tools,
add_generation_prompt=True, **chat_template_kwargs)`, with the candidate's configured
`chat_template_kwargs`, counts them as `ResidentChat` does.

```text
crit run --project CRIT_ROOT --root RUN_ROOT --id RUN_ID resolve --record inputs.json
crit run --project CRIT_ROOT --root RUN_ROOT --id RUN_ID plan --record plan.json
crit run --project CRIT_ROOT --root RUN_ROOT --id RUN_ID candidate --source SOURCE_DIR --record proposal.json
```

Before any code for a candidate, record its [plan](RUN_RECORDS.md#plan) after the parent's
Eval: whether decode and prefill are still needed, the current strategy's ceiling and the
best alternative's, the levers with each KPI's projected change, a forecast for every KPI
and how the gates hold. crit freezes the parent's newest Eval, computes the priorities and
refuses an incomplete plan; the run's first plan has no parent and may import a baseline as
is. The proposal names the plan's `plan_id`, gives each of its levers a fate in `lever_fates`
(built, pruned or deferred, with the reason unless built) and gives label, rationale, strategy,
configuration and Inference decisions (`implemented`, `inherited`, `rejected`, `untested`)
in `store_decisions`, plus parent/origin IDs when reusing earlier work. The coordinator allocates IDs and copies
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
crit run --root RUN_ROOT --id RUN_ID measure --candidate CANDIDATE_ID
crit run --root RUN_ROOT --id RUN_ID eval --candidate CANDIDATE_ID --data EVAL_DATA_DIR
```

Run these from the checkout or add `--project CRIT_ROOT`; use the exact reservation
path returned. `measure` and `eval` reserve, run and settle their own benchmark attempt.
`measure` serves the packaged candidate through `crit serve`, as `eval` does, and
records one benchmark per case from what a chat client receives: each candidate's own
greedy reply, memory of the resident host process and `crit serve`'s tokens per round.
A `benchmark --record` payload remains for workloads `measure` does not cover (cold
mode, concurrency above 1).
The package holds runnable files, dependencies and notices, without
`manifest.json` (crit creates it). Build/launch recipes are literal arguments, without
shell expansion.

| Check | What to retain |
| --- | --- |
| Baseline | Requests through Core admission, session, commit and retirement; exact source/package identities. When continuing earlier work, re-record the previous best: the same package can measure several percent differently in another session |
| Strategy | Measured bottleneck; alternatives with estimated ceilings; Inference code used and the hooks your engine supplies |
| Headroom | `crit eval status`: each KPI and the long-context curve against its floor or ceiling. Re-measure every candidate and on resume |
| Correctness | Separate arithmetic, functional and resource gates with fixed criteria; failures stay failed |
| Repeatability | Same output in a fresh process and after other requests, long ones included (Eval's request-history gate; [recorded selection](../inference/recorded-selection/README.md)) |
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
`accepted_tokens_per_round` with decode speed, greedy and sampled: Eval's headline decode samples at
the model's defaults ([Sampled speed](eval/MEASUREMENTS.md#sampled-speed)), so a drafter that only
helps greedy requests leaves most requests at one token per pass. Verify sampled drafts with
rejection sampling so the distribution stays exact, and sample on the device.

### Run the common Eval suite

Eval measures a candidate as a chat client gets it. `crit run ... eval --candidate ID
--data EVAL_DATA_DIR` reserves a benchmark attempt, serves the packaged candidate through
`crit serve` before it is sealed and runs every Eval part over the chat API
([Engines behind a server](eval/README.md#engines-behind-a-server)), quality through the
engine's score call against the run's model's scores of the same texts, which the
reference stores in the data once (`crit eval reference`, before the first candidate is
served). It applies the run's
`quality_budget`, `exactness` and `kv_tokens` and retains the result as
`candidates/ID/eval/ATTEMPT.json` for checks to reference. For a candidate built from a plan it
writes the plan's [outcome](RUN_RECORDS.md#outcome), the forecast next to the measurement, and it
prints the priorities the result's fractions give the next plan. The package therefore needs its
[resident adapter](#4-make-the-package-servable-and-seal) with `chat-stream-v1`,
`chat-logprobs-v1` and `chat-score-v1` from its first Eval on; `crit_core.serving.ResidentChat`
provides them (`score_tokens` runs the engine's prefill at every position and its decode forced
to a continuation; `crit_core.native.ResidentEngine` implements it for Core engines with query
scoring and replay).
Every candidate of a model uses the same prepared data, kept in the run workspace. `--only`
runs some parts, `--baseline` shows KPI changes against an earlier result.

Hardware telemetry stays in process, through an [Eval adapter](eval/ADAPTER.md) retained
with the candidate's source: tokens per round and acceptance on the pinned continuation
(`crit eval pin`, with the adapter's `replay`), `crit eval ab` to confirm small
differences and `crit eval kernels` to start kernel work, checking
[Kernels](../kernels/README.md) first. Pass the run's `kv_tokens` to `pin`, `ab` and
`kernels` as `--context`. An engine built as a Core
[engine library](../core/src/runtime/README.md#engine-libraries) uses the stock adapters in
`crit_core.native` for Eval and for serving. Run the full suite for the final comparison;
retain raw results, settings, inputs and the Eval revision; report skipped cases and
overrides.

Eval results are evidence, not run records, and Eval's drift budget does not replace
the agreed gates. See [Eval results](eval/README.md#results-and-status) and
[Eval evidence](RUN_RECORDS.md#eval-evidence).

## 4. Make the package servable and seal

Before the candidate's first Eval, implement the [resident adapter](SERVING.md#adapter-contract)
(`chat-json-v1`, `chat-resident-v1`, `chat-stream-v1`, `chat-logprobs-v1`,
`create_engine(context)`, model allocations reused across requests). Verify with `crit serve`: consecutive requests, a leading system
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

Plan each candidate as [AGENTS.md](../AGENTS.md) describes. crit orders the Eval goals in
each plan: decode and prefill are p1 and p2 by which is further from its limit (both p1
when their fractions are within the noise of each other; long-context decode counts toward
decode, long-context time to first token toward prefill), time to first token is p2 or p3
and never p1, and quality and the required context stay gates. Priorities order the work
and settle conflicts; every lever that costs no higher priority goes in.
The stop record accounts for each goal ([stop](RUN_RECORDS.md#stop)). Stop for the
objective, exhausted search, a blocker, cancellation or a used candidate budget
(`budget_exhausted` is refused while budgeted candidates remain); `resume` adds candidates.
When neither decode nor prefill is needed the run stops even with candidates left: record a
stop plan from the newest Eval, then stop with `search_exhausted`.

```text
crit run --root RUN_ROOT --id RUN_ID stop --record stop.json
crit run --root RUN_ROOT --id RUN_ID resume --candidates N
crit serve --workspace RUN_ROOT --run RUN_ID --project CRIT_ROOT --write-launchers
```

Save `summary.md` at the run root (contents listed in [AGENTS.md](../AGENTS.md)) and name
the candidate to open, or say none qualified. Resume needs settled cleanup and keeps
prior attempts, criteria and sealed results ([recovery](RUN_RECORDS.md#recovery-and-resume)).
