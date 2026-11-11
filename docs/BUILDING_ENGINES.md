# Building engines

The engineering details behind [AGENTS.md](../AGENTS.md): what a request, a captured case, a package and a
measurement must look like, and how to read what Eval reports. AGENTS.md says what to build and how to judge it;
`crit run ... next` says which step comes next; [Run records](RUN_RECORDS.md) defines every command and payload.
Commands below omit crit run's global options (`--project CRIT_ROOT --root RUN_ROOT --id RUN_ID`, before the action).

## 1. The request

Probe the device (`crit eval probe --out device.json`, or `device` in `settings.toml`) and `create` sizes the
context as Eval's [max context](https://github.com/crit-labs/crit-eval/blob/main/docs/CONTEXT.md): the context the
user wants (`context_tokens`, default 200,000) capped by the model's positions and the KV cache that fits, frozen as
`kv_tokens` with why it is that size in `hardware_constraints.context`. Free-memory headroom comes from
`settings.toml` (copy `settings.example.toml`); a value in the request replaces the machine's setting, so add one only
when the user asks.

```json
{
  "model_path": "ABSOLUTE_LOCAL_MODEL_PATH",
  "correctness_criteria": "REPLACE_WITH_FIXED_CRITERIA",
  "device": "ABSOLUTE_PATH_TO_DEVICE_JSON",
  "candidate_budget": 5,
  "workload": {
    "prompt_tokens": 512, "output_tokens": 128, "concurrency": 1,
    "repetitions": 3, "warmup_runs": 1, "mode": "resident"
  },
  "targets": {}
}
```

```text
crit run --project CRIT_ROOT --root RUN_ROOT create --record request.json [--from-engine ENGINE_DIR]
```

These values are a template, not acceptance criteria; `cases` holds several workloads. `model_path` is the model as
the user has it: a Hugging Face model folder, or a GGUF file whose config, tokenizer and chat template crit derives
from its metadata ([GGUF models](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md#gguf-models)). There
is no time budget, only a 30-day safety window. Provider credentials and spend caps are configured separately.

An engine built outside a run (a proof of a model's plugins over the library, say) is measured and served only as a
run's candidate. Create a run with `candidate_budget` 1, so its first candidate takes the full Eval, finish recon.md
(say under Decisions how and where the source was built), record a first plan with no levers (it imports a baseline
as is), and create the candidate from the package's `plugins/` folder.

## 2. Recon and captured cases

`recon`'s `frozen` block (`sampling`, `chat_template`, `template_variables`, `stop_token_ids`, `stop_strings`) goes
into every candidate's configuration as `recon`; edit it, with the reason in `recon.md`, before the candidate it
should change. `--force` gathers again and keeps `recon.md`. `recon.json` also fingerprints the checkpoint
(architecture, weight layout, tokenizer, template); `recon --compare OTHER_MODEL` says what another checkpoint
changes for the engine (an equal fingerprint: weight values only). `recon --drafter PATH` records a companion
drafter's pass bytes.

Each case's request is a JSON file under `captured/CASE/`, `{"messages": [...], "tools": [...],
"chat_template_kwargs": {...}}` (the last two optional), whose messages render to the workload's `prompt_tokens` as
the candidate renders them: `crit_core.text.load_model_text(MODEL_DIR, clock=None).encode_chat(messages,
tools=tools, add_generation_prompt=True, **chat_template_kwargs)` with recon's `template_variables` under the
request's own. A request that needs a template switch names it in its `chat_template_kwargs`. `capture --case
CASE_ID [--text FILE] --inputs inputs.json` writes one (a neutral passage, or the text given, cut and padded to the
exact count) and sets its manifest row; `resolve --record inputs.json` then freezes the model and input identities.

## 3. Build, check and compare

`build --candidate ID --record recipe.json [--drop GLOB]` copies the retained source to `work/builds/ID-ATTEMPT`, runs
the recipe's literal `build.command` there to the reservation's deadline (log `logs/build-ATTEMPT.log`), settles the
attempt and packages the folder: every file the build added or changed is bound as a dependency, and `--drop`
removes intermediates first. A recipe is its `build` only: Core's loader runs the package, so there is no launch
entry. An [external engine](RUN_RECORDS.md#external-engines) is the one exception: its configuration declares the
command that starts its own server, and `serve`, `eval`, `measure`, `ablate` and `kernels` start and stop it (whole
process tree) under each attempt. Build commands are literal arguments, without shell expansion. The package holds runnable files,
dependencies and notices; data the engine reads at start (a residency profile that fills a device expert cache)
belongs in it too, as part of the candidate's identity.

A build or check crit does not run itself is reserved before launch, supervised to the returned deadline (the whole
process tree) and settled with its logs and cleanup, failures included:

```text
reserve --candidate ID --operation build --seconds PROCESS_CAP
settle --reservation RESERVATION_PATH --record result.json
package --candidate ID --source PACKAGE_DIR --record recipe.json
check --candidate ID --record check.json
```

The library's plugins are meant to give a fast engine from candidate 1 with only the model plugin written for the
checkpoint ([the model plugin](PLUGINS.md#the-model-plugin)): decode rounds captured through Core's round driver,
short prompt chunks through Core's prompt graphs, the drafter over the model's device draft head, and the KV pool,
attention, projections, recurrence and experts plugins each in its stored format with crit's kernel (built for the
GPU, tuned by `crit kernels sweep`) or its torch fallback. A mixture of experts larger than the device needs no engine code for its experts' residency:
device-placement's `off_device` puts the routed experts' bundles in host memory, the expert source's `open_store`
holds them where the device reads them, and expert-residency's `cache(...)` keeps the hot ones in device slots. Make
the KV pool with `spare` bytes for the cache's slots and pass it as `lend`, so the cache also uses pages no sequence
holds; warm it from a residency profile and read the engine's `telemetry()` for its hit rate. A drafting model plugin
may let its rounds choose their draft width among several captured ones (`Rounds(widths=..., width_policy=...)`),
sizing every buffer for the widest.

In `preserve`, Eval's honesty check holds every reply token's log-probability and its alternatives within 0.001
nats of the same reply scored as fixed text. The reply comes from decode rows and the score from prompt rows, so the
two paths must give a token the same bits, and a prompt path that sums in another order (a GEMM for long prompt calls
against a GEMV for decode rows, a prompt reader against the split-KV reader) misses it by thousandths to tenths of a
nat. The library keeps one arithmetic on both paths with three settings: the attention plugin's `prompt_reader:
decode`; attention-state's `fresh_mib: 0`, so a prompt chunk over FP8 or INT8 pages reads its own keys and values
from the pages; and prompt calls through the projections' decode kernel (chunks of at most
`invariant_rows(projections)` tokens, or quantized-projections' `tile_rows`). Prefill then runs at the decode kernels'
rate: on a 33B NVFP4 model on one RTX 5090, 385 tokens a second through the GEMV and 1,024 through the tile, against
about 2,150 on the prompt path. A `speed` engine's honesty bound is looser, so the prompt reader, fresh chunk keys
and pinned prompt GEMMs fit it.

| Check | What to retain |
| --- | --- |
| Baseline | Requests through Core admission, session, commit and retirement; exact source and package identities |
| Strategy | Measured bottleneck; alternatives with estimated ceilings; the plugin in each slot, the Inference code it starts from and the facts your engine supplies |
| Correctness | Separate arithmetic, functional and resource gates with fixed criteria; failures stay failed |
| Repeatability | Same output in a fresh process and after other requests, long ones included (Eval's request-history gate) |
| Performance | Matched inputs and protocol, warmups, every repetition, prefill, decode, request and load costs, physical memory; text-dependent settings such as draft width tuned on mixed cases |
| Coverage | Representative contexts and request shapes; operator tests alone don't qualify a model |

Measure cold and changing requests, not only warm repeats; charge initialization, transfers, synchronization and
refresh to the measurement they serve; keep polling out of timed paths; define aggregation first. One fast sample is
not a result. Choose implementations (kernel variants, library algorithms) from a table recorded in the candidate's
configuration, not by timing at load: near-ties resolve differently per process and can change outputs.

`measure` serves the packaged candidate through `crit serve` and records one benchmark per case from what a chat
client receives; a case of concurrency above 1 sends its requests at once and adds their aggregate throughput. It
comes after `check --from-eval`: a benchmark qualifies only on the checks recorded before it
([measured benchmarks](RUN_RECORDS.md#measured-benchmarks)). A `benchmark --record` payload remains for workloads
`measure` does not cover (cold mode).

### Serve concurrent requests

Every engine serves several requests at once; one request at a time is a batch of one. With
[mixed batches](../core/src/runtime/README.md#mixed-token-work-batches) (`max_batch_tokens`) one step runs every
scheduled request's decode rows and prompt chunks and reads each weight once. Core supplies what every batched engine
needs:

| Need | Core |
| --- | --- |
| A step's row layout and logits rows; a WorkItem's rows | [Batched steps](../core/src/batch/README.md): `StepLayout`, `step_item` |
| Admission that keeps every admitted request able to finish | `PageReservations` |
| Batched rows that keep their serial bits | The [batch invariance rules](../core/src/batch/README.md#batch-invariance) |
| KV pages, prefix reuse, copy-on-write forks | [State cache](../core/src/state/README.md) |
| Speculative rounds per request in a shared step | [Speculative rows](../core/src/batch/README.md#speculative-rows) |
| Serving | Core's loader runs the model plugin on its [step engine](../core/src/runtime/README.md#step-engines); crit serve serves it as `chat-concurrent-v1` |

A request's tokens must not depend on the requests sharing its steps: every kernel on the step path follows the
invariance rules. Measure concurrency as a workload of its own (a case with `workload.concurrency` N) and through
Eval's [concurrency](https://github.com/crit-labs/crit-eval/blob/main/docs/MEASUREMENTS.md#concurrency) part, which
crit serve's `max_concurrent_requests` turns on.

### Run the common Eval suite

`crit run ... eval --candidate ID --data EVAL_DATA_DIR` serves the packaged candidate through `crit serve` and runs
every Eval part over the chat API, quality through the engine's score call against the model's scores of the same
texts (`crit eval reference` stores them in the data once). It applies the run's quality terms and `kv_tokens`,
retains the result as `candidates/ID/eval/ATTEMPT.json`, writes the plan's [outcome](RUN_RECORDS.md#outcome) and
prints the priorities the result gives and each KPI's fraction of its limit with its largest gaps: a low fraction is
the next plan's room, not a result. The model plugin declares `token_logprobs` with 20 alternatives, `scoring`,
`query_scoring` and `replay`, which Core's loader requires.

How far an Eval goes has one controller, the candidate's rung: crit passes it to Eval as a cap (none, stage 1,
stage 2, or certified), and within that cap Eval chooses the
[stages](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md#stages) whose requests fit its 45-minute
budget at the candidate's measured rates. Stage 1: each speed case once, short quality texts, the request history,
the memory the required context needs. Stage 2 adds repetitions, follow-up turns, the long quality text and the
long-context curve's first point. Stage 3 the sampled suite, exactness and the follow-up turns again. Only the
certified Eval serves the required context itself and reopens the engine at a lower context, which the capacity and
lower-context gates need.

Eval measures each speed case at steady state (crit-eval's `speed.steady_state`), so an engine that adapts to what
it serves, such as an expert cache, needs no freezing. Eval journals every request beside the result; an attempt that
stops goes on with `--resume` in a new attempt, answering the measured requests from the journal. A fix to the
candidate is a new package and a new Eval, not a resume. `crit eval journal FILE` says how far an attempt got.

Where recon's placement keeps weights off the device, Eval prices them at host memory's tier
([memory tiers](https://github.com/crit-labs/crit-eval/blob/main/docs/MEASUREMENTS.md#memory-tiers)), routed experts
at the hit rate the engine reports (`expert_reads` and `expert_hits` in its decode telemetry, or the expert cache's
reads and hits in `telemetry()`), the KV at `kv_bytes_per_token`. The engine's facts declare `host_bytes_per_pass`
(`{"link", "cpu"}`) where its own path differs from the faster of the link and the CPU.

Read the reports beyond the KPIs ([Measurements](https://github.com/crit-labs/crit-eval/blob/main/docs/MEASUREMENTS.md#limits)):

- **Prefill by term:** `crit eval status RESULT --kernels kernels.json` puts each prefill term (matmul per weight
  format, attention, recurrence, other kernels, the host's share of the first token) next to its floor, and splits
  a speculative round into its verify pass and draft steps.
- **Prefill ceilings by activation format:** declare the engine's `prompt_activations` (the model plugin's info);
  status prices the ceilings at BF16, FP16 accumulation, FP8 and the stored format, says which the quality evidence
  rules out and names the fastest shown passing as the reachable limit.
- **Quality during development:** `crit eval run --package P --only quality` scores in process (development
  evidence, never the gate; pass the run's `--quality-mode`). `--attribute-drift` gives each matrix group's share of
  the budget from the engine's `exact_variants`, with its spread over positions (`budget_spread`).
- **Time to first token by part**, the sampled cases' **standard errors** (`--adaptive-sampled` repeats a noisy
  case), acceptance beyond the built draft width (`wider_draft`) and the **device memory** left free at the required
  context.
- **The draft step's share of a round:** when a round costs about the same at every draft width, the draft steps
  dominate it; the draft head is then the lever (`draft-vocabulary-subset`, after its README's caution on held-out
  languages, and `draft-trees`). Declare `draft_bytes_per_step` so Eval prices the round with its draft steps.

Hardware telemetry stays in process, through Core's [eval-v1 adapter](https://github.com/crit-labs/crit-eval/blob/main/docs/ADAPTER.md)
over the package (`--package DIR`): `crit eval pin --package` for tokens per round and acceptance on the pinned
continuation, `crit eval ab --package` to confirm small differences, `crit eval kernels --package` to start kernel
work; pass `--context KV_TOKENS` to each. Class the engine's kernels in its model plugin's info `kernel_work`
(`{"match": NAME_PART, "term": matmul|attention|recurrent|other|draft, "format": nvfp4|fp8|bf16}`, or `crit eval
kernels --work RULES`) and keep draft kernels named `draft`, so status joins each term to its floor. Trace every case
the prefill score takes. Under WSL a profiler may record no kernels of CUDA graph replays or of `cuLaunchKernelEx`
launches (crit's CUDA kernels may be): time such work with CUDA events around the call and a spin kernel
(`torch.cuda._sleep`) before it, so the span holds no host gaps.

Eval results are evidence, not run records; they reach the run's checks only through `check --from-eval`
([Eval evidence](RUN_RECORDS.md#eval-evidence)).

## 4. Seal and deliver

A candidate measured by `measure` or a completed `eval` seals `completed`; `built` is a servable engine without
measurements; a failure seals with its failure lifecycle. Sealing needs settled attempts and closes the candidate for
good. A failed gate does not prevent serving, but prevents qualification. Once sealed, verify A → B → A switching with
`crit serve`.

```text
stop --record stop.json
resume --candidates N
crit serve --workspace RUN_ROOT --run RUN_ID --candidate CANDIDATE_ID --project CRIT_ROOT --write-launchers
```

`stop` records the reason, the critique and each goal's headroom ([stop](RUN_RECORDS.md#stop)); `resume` adds
candidates and keeps prior attempts, criteria and sealed results ([recovery](RUN_RECORDS.md#recovery-and-resume)).
