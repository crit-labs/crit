# The crit charter

This is the one place that says what crit is, what it is for and the rules it keeps. The other docs say how:
[AGENTS.md](../AGENTS.md) and [Building engines](BUILDING_ENGINES.md) walk a run, [Run records](RUN_RECORDS.md)
defines every record, [Plugins](PLUGINS.md) the engine's shape, [What belongs in crit](CONTRIBUTION_CRITERIA.md) the bar
for a change. When one of them disagrees with this charter, one of the two is wrong: fix it, don't pick a side
silently.

## What crit is

crit is an agent toolkit for inference engines. An agent uses it to improve an existing engine or to build one from
scratch for one LLM on one machine, to test every change in a correct, repeatable environment against that machine's
physical limits, and to serve the result. It is built for agents: building an engine by hand is not a supported
path.

It has four parts and one front door:

| Part | What it is |
| --- | --- |
| [Core](../core/README.md) | The plumbing every engine needs and none should rebuild: requests, sessions, admission, the concurrent step scheduler, stop rules, sampling control, chat templates, state-cache bookkeeping, model files, and the contracts crit serve and Eval speak with an engine. It knows no model and no device. |
| [Inference](../inference/README.md) | A curated library of techniques distilled from established engines, each a documented mechanism with code, a contract and, where it fills a slot, a plugin with a host reference and tests. |
| [Kernels](../kernels/README.md) | A kernel core of building blocks (format decoders, tensor-core multiplies per GPU generation, pipelines, reductions) and kernel techniques composed from them, as crit has Core and Inference: each technique has knobs an engine sweeps and tunes on its own device and shapes, timed beside the vendor and open libraries it is measured against, and a workbench to check and time a kernel. Engineered kernels are mined for mechanisms that become core blocks or new structures. |
| [Eval](https://github.com/crit-labs/crit-eval) | `crit-eval`, a separate package: one fixed suite that measures speed and quality through the same API a client uses, every score next to its physical limit on the device it ran on. |
| [CLI](../cli/README.md) | `crit`: `crit run` keeps a run's records (recon, plans, candidates, builds, checks, Evals, ablations, stop, cost), `crit serve` serves an engine, `crit core` builds Core's runtime, `crit smoke`, `crit plugins`, `crit kernels`, `crit models`, `crit inference`, `crit pull`, `crit bench`, `crit hub`. `crit mcp` offers `crit run`, `crit eval`, `crit kernels`, `crit models`, `crit inference`, `crit pull` and `crit serve` as MCP tools ([MCP](../mcp/README.md)). |

An engine is Core plus a selection of plugins, one per slot (`model`, `placement`, `projections`, `attention`,
`kv_cache`, `sampler`, and every slot the model plugin requires), and nothing else. A candidate is a sealed plugin
package; crit serve, Eval and `crit run` run nothing of it but its plugins ([Plugins](PLUGINS.md#every-engine-is-plugins)).

Sibling repositories:

| Repository | Role |
| --- | --- |
| [crit-eval](https://github.com/crit-labs/crit-eval) | The yardstick. Released on its own and before crit when both change. |
| [crit-hub](https://github.com/crit-labs/crit-hub) | A registry of engines by GPU and model, each pinned to its source and checkpoint with the same six stats. A registry, not a scoreboard. External engines are registered so they can be browsed: crit-hub links to them, and crit never codes with them. |
| [crit-engines](https://github.com/crit-labs/crit-engines) | The sealed source of crit-built engines that crit-hub lists. |

crit-kernels, which was to publish crit-labs kernels to the Hugging Face Kernel Hub, is folded into crit's
`kernels/`: everything lives in crit.

## What crit is not

- **Not an inference engine.** Core alone serves nothing; it has no layer arithmetic. The engine is what a run builds.
- **Not a collection of model ports.** crit holds no code path for one model family, one checkpoint or one GPU. The
  model's instance lives in the engine's plugins; crit holds the general mechanism behind it
  ([interfaces, not instances](CONTRIBUTION_CRITERIA.md#interfaces-not-instances)).
- **Not a kernel host.** crit hosts and distributes no kernel binaries. Its kernels are templates in its source that an
  engine builds and tunes for its device; the rest of the arithmetic comes from vendor and open libraries or the engine.
- **Not a benchmark that ranks.** Eval reports facts and limits; it does not grade speed, rank engines or prescribe a
  build order. crit-hub lists engines, it does not rank them.
- **Not a training tool.** Runs never train or fine-tune a drafter or the target. Weight transforms (PTQ,
  re-encoding) happen only where the run's `weights_policy` allows them, through `crit models`.
- **Not a kit for building engines by hand.** crit is an agent toolkit. Its records, checks and guides are written
  for an agent driving `crit run`; a person building an engine by hand is not a supported path.
- **Not tied to one vendor.** crit aims at every backend. CUDA is the one built today.
- **Not a scheduler of people's time.** A run's budget is a number of candidates, not hours.

## Vision

Nobody should have to wait for someone else to optimize an LLM for their hardware. crit serves two kinds of users:

- **A. Improve a jump-off engine.** A user who already has an engine, from crit-hub or from elsewhere, uses crit to
  make improvements and test each one in a pristine, correct and robust test environment: the same checks, the same
  Eval and the same records for every change, so a gain is real and a regression cannot hide. An external engine
  stays what it is: it is never rebuilt as plugins. It takes crit's techniques and kernel templates into its own code,
  and crit's checks and Eval measure it through the API it serves.
- **B. Build from scratch, quickly.** A user who has only a model and a machine gets a fast, correct, servable engine
  in hours, with the record of how it got there as evidence, not a recollection.

Four things make that possible, and they are what crit is for:

1. **The boring parts are done.** Requests, memory, text, sampling and serving are Core's, so a run spends its time on
   what makes this model fast on this machine.
2. **What works elsewhere is reusable.** The speedups established engines rely on are techniques in Inference and
   kernel templates in Kernels, each testable alone and measurable in place by ablation or a sweep.
3. **The distance to the limit is visible.** Eval turns the device's measured bandwidth and matmul rates into a limit
   for every KPI, so "fast" means a fraction of what the hardware allows, and quality loss cannot hide behind speed.
4. **The test environment is trustworthy.** Every candidate is sealed, checked and measured the same way, so results
   compare across changes, runs, agents and machines.

The scope is every backend (CUDA today; ROCm, Metal, Vulkan and CPU), one GPU or several, and engines that run on the
CPU alone. Multi-GPU and CPU-only engines are in scope and not fully built yet.

Engines a run produces feed crit-hub, so the next user starts from a working engine instead of from zero, and the
general lessons a run learns feed back into crit under the contribution criteria.

## Inputs

What a run takes. Each one is the user's; crit freezes it at `crit run create` and later records cite it.

| Input | Where it comes from | Default |
| --- | --- | --- |
| Model | A local Hugging Face folder (safetensors, `.bin`, ONNX, quantized formats) or a GGUF file | required |
| Device | `crit eval probe` (bandwidth, GEMM rates per format, tensor-core rates, link and host rates) | required, or `kv_tokens` |
| Context | `context_tokens`, capped by the model's positions and the KV cache that fits; frozen as `kv_tokens` | 200,000 |
| Quality mode | `speed` or `preserve` ([quality modes](RUN_RECORDS.md#quality-modes)) | `speed` |
| Quality budget | Perplexity ratio, mean KL, p99 KL against Eval's exact reference | the mode's: 1.4 / 0.4 / 4.0 (`speed`), 1.05 / 0.05 / 0.5 (`preserve`) |
| Weights policy | `as_stored` or `transformable` | `transformable` in `speed` |
| Exactness | `reported` or `required` | `reported` |
| Sampling | The model's `generation_config.json`; only the user overrides it (`eval_sampling`) | the model's |
| Workload and cases | Prompt and output sizes, concurrency, repetitions, mode | 512 / 128, concurrency 1 |
| Candidate budget | Candidates on the [candidate ladder](RUN_RECORDS.md#the-candidate-ladder) | 5 |
| Companion drafters | Up to 8 published drafter folders, or one the user produced with `crit models drafter` | none |
| Jump-off engine | A crit-built engine from crit-hub (`create --from-engine`), or the user's external engine, served behind an OpenAI-compatible API and recorded as an [external candidate](RUN_RECORDS.md#external-engines) with its launch command | none |
| Machine settings | `settings.toml`: memory headroom, `wsl_run_root`, defaults above | `settings.example.toml` |

## Outputs

| Output | What it is |
| --- | --- |
| Candidates | Each a sealed plugin package: source frozen by hash, configuration (the recon block included), dependencies and their notices. Sealed candidates never change. |
| Evidence | Recon (`recon.json`, `recon.md`), every plan with its forecast and outcome, smoke and serve checks, ablations, Eval results, kernel traces, benchmarks. Failures stay visible. |
| A stop record | Per KPI: the score at the start, the score reached, the limit, what was built for it and the physical cause of what is left. |
| `summary.md` and `cost.json` | The handoff: baseline against the recommended candidate, gains, trade-offs, missing coverage, toolkit feedback, and what the run cost per candidate. |
| A served engine | `crit serve` for the selected candidate: OpenAI-compatible chat (streaming, tools, stop strings, structured output, log-probabilities), plus Anthropic Messages, OpenAI Responses, `/api/chat` and Prometheus metrics, with `serve.cmd` and `serve.sh` launchers. |
| A crit-hub card | Optional: the sealed source exported to crit-engines and a card with its six stats from the sealed Eval. |

## How a run goes

Recon studies the model and names its placement scenario. Then candidates, one per round: plan from the limits and
the last result, build from plugins, check (plugin tests, `serve --check`, crit smoke, ablation), measure as far as the
candidate's rung allows, read the gaps, plan the next. The run stops when the budget is used, when neither decode nor
prefill can still gain beyond its noise, or when it is blocked; it then records the stop, selects, serves and hands
off. `crit run ... next` names the next step from the run's records at any point; [AGENTS.md](../AGENTS.md) is the full
procedure.

## Rules

Rules hold for every run, every model and every machine. Where crit enforces one, a record that breaks it is refused.

### Shape of crit

1. **Core is model- and hardware-agnostic.** No layer arithmetic, no kernels beyond the backends' runtime services,
   no number or rule true only for one model family, dtype, GPU or backend. What engines share enters Core as
   bookkeeping. Core and the plugin contract stay backend-neutral, so every backend can be added.
2. **Every engine crit builds is Core plus plugins; plugins are the truth.** One plugin per slot, every need met, no conflict, one
   memory budget; a slot's arithmetic lives in that slot's plugin, never inline in the model plugin. Core's loader
   checks this. There is no other engine shape: a doc, example or engine that describes one is out of date and gets
   rewritten or replaced. An engine may write every plugin itself instead of taking one from the library. An external
   engine is the exception that proves the rule: crit never reshapes it, it only lends it techniques and kernels and
   measures it from outside.
3. **Every engine batches.** The model plugin steps many requests' rows at once, and a row's tokens never depend on
   the rows beside it (row invariance).
4. **Interfaces, not instances.** A model's need becomes a general mechanism in crit and an instance in the engine.
   Shared knowledge (published formats, config conventions shared by families) lives once, in data.
5. **Kernels are templates, never hosted.** crit's kernels are curated templates in `kernels/` with knobs an engine
   sweeps on its own device and shapes, each checked against a reference and measured against the best vendor or open
   library at the same arithmetic. crit hosts no kernel builds. An engine's own kernel stays in the engine. A kernel's
   speed is evidence about the GPU it was measured on, never about another: its speed-only constants are knobs, its
   device limits are read from the device, an instruction one GPU generation has sits behind a feature check with a
   path for the others beside it, and a new way of computing an operation is added as a structure beside the ones it
   beat, which the sweep keeps choosing between on every GPU ([guarding against one
   GPU](../kernels/README.md#one-gpu)). A tuning belongs to the GPU it was swept on.
6. **One code path per concern.** When a general mechanism replaces a specific one, the specific one goes in the same
   change. A kernel's structures are not such copies: each is one way of computing its operation, kept while some GPU
   and shape may pick it.

### Measurement

7. **Eval reports facts; the agent reasons.** Eval gives each KPI its score, its physical limit, the gap, the noise and
   what hides a limit. It issues no verdicts, thresholds, tags or next steps. The only pass/fail is the gates.
8. **Limits are physical.** A limit never assumes the current implementation (not the current drafter's acceptance,
   not the current kernels). If an engine measures beyond a limit, Eval's model of the limit is wrong and gets fixed.
9. **No blended score.** The end-to-end score is reported as a fact; no KPI is traded against another through it.
10. **Compare like with like.** Same Eval, data, device and quality mode. Results from different modes never compare.
11. **The numbers you read are the numbers you get.** Eval measures every candidate through `crit serve`, over the
    chat API a client uses.

### Runs

These are the principles a run's procedure serves. The procedure itself lives in [AGENTS.md](../AGENTS.md) and
[Run records](RUN_RECORDS.md); this charter does not restate it.

12. **The budget is candidates, not time.** Effort, time and the candidate count are never a reason to leave a lever
    out, and never the cause of a remaining gap in the stop record ([budget counts](RUN_RECORDS.md#budget-counts)).
13. **Goals are the user's to set; the default is the limit.** A user may set targets. Without them the goal is
    to raise speed toward the physical limits while quality stays inside the run's drift budget.
14. **Candidate 1 is a proof of concept; every later candidate is an ambitious one-shot.** From candidate 2 on,
    everything expected to close the gaps goes in together; large rewrites and new kernels are expected.
15. **Plans are records.** Written before code, checked by crit for completeness and consistency with the facts they
    freeze (never for whether the reasoning is good), compared with their outcome; every lever gets a fate
    ([plan](RUN_RECORDS.md#plan), [checks](RUN_RECORDS.md#checks)).
16. **Priorities order work, never what is left out.** Quality and the required context are gates, not goals; how
    crit ranks decode, prefill and time to first token is in [AGENTS.md](../AGENTS.md#3-build-and-compare-candidates).
17. **Measure before you decide.** An estimate stays an estimate and is never raised to pass a check; pruning follows
    measurement.
18. **The context is a requirement.** Speed depends on the tokens held, not the capacity.
19. **Output differences are bugs until explained,** within what the run's quality mode judges as rounding
    ([quality modes](RUN_RECORDS.md#quality-modes)).
20. **The user's choices are the user's.** Quality mode, quality budget, weights policy, context, sampling and
    exactness change only on the user's word; a resume that amends a run's terms records who approved it.
21. **Records go through `crit run`.** Changed source, configuration or dependencies make a new candidate; sealed
    candidates and existing projects are never altered. A candidate's state, and what qualifies one, has one
    definition ([states](RUN_RECORDS.md#states)).

### Quality

22. **Quality is a budget to spend on speed.** Inside the budget, a lever counts in full; drift below what is needed is
    no gain. Both modes judge against the same exact reference: the checkpoint's stored weights with exact activations.

## Guardrails

Things no run, agent or contribution may do. Most are enforced by crit; the rest are checked in review.

- **Never fit the engine to Eval.** No content-dependent choice (draft-head or vocabulary subsets, n-gram tables,
  acceptance thresholds, routing) chosen by its Eval score or built from Eval's prompts or look-alike text; no special
  cases for Eval's token ids, lengths or cases. A content-dependent choice needs a general corpus unrelated to Eval and a
  held-out check in several languages. Eval's held-out prompts and drawn lengths exist to catch this.
- **Never train or fine-tune a drafter.** Use the checkpoint's, a published companion as released, or one the user
  produced with `crit models drafter`; the stored target verifies every token.
- **Never change the model's sampling.** Not even to a model card's recommendation; only the user overrides it.
- **Never invent measurements.** Without the target hardware, results are labelled unmeasured. Never claim a server
  was started on the user's machine.
- **Never generalize one GPU's kernel results.** A kernel measured faster on one GPU is faster there, on those shapes:
  no default, constant, removed structure or claim may rest on it for other GPUs. Every number names its device; a
  capability a kernel was only compiled for is "compile-checked", not "runs"; a structure leaves a kernel only when it
  loses on every GPU and shape it was measured on, more than one GPU among them, and another structure covers every
  GPU it ran on.
- **Never weaken a check to pass it.** Don't relax agreed criteria, demand byte-identical output where none is
  required, or bend required work into a speed lever.
- **Keep run data out of crit.** Model-specific source, models, dependencies, builds and results live in the run
  workspace; generated files, caches and run data never enter the repository.
- **Clean-room runs stay clean.** A request to build from crit alone keeps crit's providers and excludes earlier
  results: past runs, earlier engines for the model and records outside the run.
- **No engine names in crit.** crit's code, docs, tests and changelogs describe behaviour generically and never name
  other inference engines as references.
- **No one-off rules.** A heuristic or phrasing from one conversation or one run does not become a rule; rules hold for
  any run.

## How to improve crit

Every change meets all three [contribution criteria](CONTRIBUTION_CRITERIA.md): it improves the platform measurably
(a reproduced bug, a failing test that passes, a before/after measurement, a capability engines kept rebuilding), it
does not overfit to one model or machine, and it is valuable to almost any model and hardware. A change that misses one
belongs in an engine, a run record or crit-hub.

- **Fix causes in crit, not symptoms in a run.** When an agent settles for less than the limits allow, find what in
  crit let it settle and remove that.
- **Salvage from runs as mechanisms.** A lesson from one run goes into that run's `summary.md` and its crit-hub entry;
  only the general principle behind it reaches crit, and often only after a second model or machine confirms it.
- **Where a change goes:** a lifecycle or bookkeeping need of every engine into Core; a reusable technique into
  Inference as an entry and, where it fills a slot, a plugin tested against the slot's base; a missing kernel into
  `kernels/` as a technique, a mechanism mined from an engineered kernel into the kernel core or a technique's structures; a change to how engines are judged into crit-eval, keeping the suite, protocol, reference and timing
  comparable and stating the effect on existing results; a limit that an engine beat into Eval's limit model.
- **Docs describe what ships.** Update docs and examples with behaviour. A measured number in a doc is evidence: name
  its device, model and workload, and state the principle separately.
- **Check proportionately.** Run the tests that cover the change; native, quality and performance claims need their own
  tests ([Contributing](CONTRIBUTING.md#check-a-change)).
- **Keep this charter true.** A change that alters what crit is, its inputs, outputs, rules or guardrails updates this
  file in the same change.

## Not built yet

What the charter commits to that crit does not do today, so a reader knows the gap is known:

- **Backends beyond CUDA.** ROCm, Metal, Vulkan and CPU backends; plugins can already declare them.
- **Kernels run on more than one GPU.** Every kernel's tests, gates and measurements so far come from one RTX 5090
  (sm_120). CI compiles every kernel for every architecture it claims, so the other architectures are compile-checked,
  not run; Kernel Core's wgmma (sm_90) and tcgen05 (sm_100) wrappers are compile-checked and no technique uses them yet.
- **Multi-GPU and CPU-only engines,** end to end and with Eval's limits for them.
- **`crit serve` for a sealed external engine.** `crit run` records, checks, measures, ablates and traces an
  external engine started from its launch command, but `crit serve` runs plugin packages only: a sealed external
  candidate is served by its own launch command.
