# Build a bespoke engine with crit

Build a complete engine for the user's LLM and hardware, with recorded candidate
comparisons and a working OpenAI-compatible API (`crit serve`). You own the work from
investigation to delivery. Changes to crit itself must meet [what belongs in crit](docs/CONTRIBUTION_CRITERIA.md)
(general interfaces for any model and any CUDA hardware, never one model's or machine's code) and follow
[Contributing](docs/CONTRIBUTING.md).
Install from the repo root with `python -m pip install -e .`, then run `crit ...`.

## 1. Set up the project

- Establish model/encoding, hardware/OS, workload, quality criteria, performance
  objective, resource limits, attempt bounds and paid allowances. Reuse user
  decisions, ask about essential gaps, state routine assumptions. Without a user
  target, record one in `targets` from a speed-of-light estimate (section 3) before
  the first candidate.
- Probe the device (`crit eval probe`) and pass it as the request's `device` (or set
  `device` in `<crit-repo>/settings.toml`); `kv_tokens` then becomes Eval's max
  context for the model on it. Set `kv_tokens` only when the user asks. Preflight
  model, tools and dependencies before starting the clock.
- **Study the model before the first candidate.** After `create`, run
  `crit run --id RUN_ID recon` (`candidate` refuses a model folder without it): it reads `generation_config.json` (sampling defaults),
  the tokenizer files and chat templates (BOS/EOS and stop tokens, template switches
  such as `enable_thinking`) and `config.json` (rope scaling, MTP or draft heads,
  attention layout), writes the facts with file hashes to `work/recon.json` and
  scaffolds `work/recon.md`. Finish `recon.md` from the model card (recommended
  sampling per mode, e.g. thinking vs non-thinking, known issues, context limits) and
  the reference implementation in transformers (norm and attention quirks),
  citing sources. Where they override the files, edit the `frozen` block of
  `recon.json` and say why under Decisions. Every later candidate freezes that block
  (sampling, chat template and its variables, stop tokens) as `configuration["recon"]`
  instead of defaulting to greedy; Eval's speed and exactness cases stay greedy by protocol.
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

- **[Core](core/README.md):** Model/Session/Backend for requests, resources and
  lifetimes; libraries for sampling, state cache and model files. You supply model
  arithmetic, tokenizer and layouts.
- **[Inference](inference/README.md):** read relevant techniques and their code; adapt,
  combine or fuse them, preserving correctness and ownership invariants. Nothing is mandatory.
- **[Kernels](kernels/README.md):** search the Hugging Face Kernel Hub by operation
  before writing optimized arithmetic, unless the criteria exclude outside sources.
  Check operations, formats, shapes, OS/device, licenses, dependencies and the three
  checks Kernels lists.
- **[Models](models/README.md):** only when the run's `weights_policy` is `transformable`:
  quantize the source with `crit models ptq`, run the output through `crit models eval`
  against the source's Eval data, and keep both records with the candidate.
- **[Eval](docs/eval/README.md):** connect the engine through an eval adapter; use the
  same prepared data and reference for every engine of one model. A Core engine built
  as a shared [engine library](core/src/runtime/README.md#engine-libraries) gets the
  Eval and serve adapters from `crit_core.native`.

Retain Core revision/build inputs, dependency identities/notices and adaptations with
candidate source. Keep model-specific changes in the engine; record toolkit gaps in the summary. Salvage
from a run reaches crit only as a general mechanism that meets [the criteria](docs/CONTRIBUTION_CRITERIA.md).

## 3. Build and compare candidates

Follow [Building engines](docs/BUILDING_ENGINES.md) and [Run records](docs/RUN_RECORDS.md).
Work in rounds, one candidate per round:

1. **Recon** (`crit run ... recon`, section 1) before the first candidate. Cite the
   findings that bear on the design (attention layout, draft heads, rope, length
   thresholds, known issues) in each proposal's `strategy` and `rationale`.
2. **Plan** each candidate as an ambitious one-shot from the physical limits and the
   last round: `crit eval status` goals, gaps and failed gates, the profile's heaviest
   kernels, the parent's result. Include everything expected to reach every goal;
   effort is never a reason to leave something out.
3. **Build** it (`crit run ... candidate`); fix or prune what is slow or fails a gate
   (prune only after measuring) until the plan is implemented.
4. **Evaluate:** `crit eval run --baseline FIRST_RESULT`, `crit eval kernels`, then
   `crit eval status RESULT --baseline PARENT_RESULT`. A failed gate, an exactness drop
   or a repeat difference is a bug and the next candidate's first item.
5. **Revisit recon** when a result contradicts it, or when the engine switches behaviour
   at a threshold (prefetch, sparse attention, chunk sizes) Eval does not cross: update
   `recon.md` (and the frozen block if needed) and add a run case that crosses it
   before the next candidate.
6. **Repeat** from step 2 until the targets are met, the candidate budget (a number of
   candidates, not time) is used or the run is blocked. On resume, re-measure first.

- **Strategy first.** Working baseline, measured bottlenecks, then compare strategies,
  not only settings. Reach the highest-ceiling strategy in working form before tuning
  kernels; sessions can end early and tuning does not transfer between strategies.
- **Eval sets the goals.** `crit eval status` gives one goal per KPI: decode, prefill,
  time to first token, and long-context decode and time to first token, each from
  current score to physical limit. Pursue all at once; none trades against another.
  Never weigh by the end-to-end score.
- **The context is a requirement.** Build every candidate for `kv_tokens` and pass it to
  Eval as `--context`. Beyond weights and KV cache the engine gets `context_reserve_gib`
  (default 3 GiB) for activations, workspaces, graphs and speed techniques. Speed must
  depend on the tokens held, not the capacity: Eval fails a candidate that is slower
  when opened at a lower context.
- **Never fit the engine to Eval's suite.** Forbidden: sizing or choosing anything
  content-dependent (vocabulary or draft-head subsets, draft tables, n-gram or prompt
  caches, acceptance thresholds, per-content routing) by its Eval score; building such
  a choice from Eval's prompts, pinned continuation, held-out prompts or look-alike
  text; special-casing Eval's token ids, lengths or cases. Hardware/shape tuning (GEMM
  tables, tiles, launch geometry) is fine. A content-dependent choice needs a general
  corpus unrelated to Eval and a held-out check in several languages and domains, both
  stated in the proposal; when in doubt keep the general option (the full draft head).
  Pass `--baseline` (the run's first candidate) so Eval's generality gate applies, and
  check each candidate against its parent (`crit eval status RESULT --baseline PARENT_RESULT`).
- **Never train or fine-tune a drafter.** Use the checkpoint's drafter or a published
  companion draft model as released.
- **Ceilings, in milliseconds.** Decode = tokens per round ÷ round time. A model pass
  cannot beat bytes read ÷ bandwidth (weights read plus KV and recurrent state). A draft
  step has its own bytes floor; acceptance at a generous width projects every smaller
  width, and draft width, head size, vocabulary and encoding must be tested together.
  Prefill is bound by FLOPs at the GEMM rate per weight format, plus attention. Time
  to first token is at least the larger of the compute floor and one weight read.
  `crit eval probe` measures bandwidth and GEMM rates; status reports each term's floor.
- **Profile** each candidate with `crit eval kernels`; for the heaviest kernels per
  phase far from their floor, check Kernels before replacing them, and fix kernels
  that leave SMs idle.
- **Measure** with [`crit eval`](docs/eval/README.md) plus the run's cases
  (`stop_record_headroom` with `--baseline` seeds the stop record). Sweep settings on
  the mixed suite, not one prompt.
  Retain adapter, applied configuration, data, reference and results; report skipped
  cases. Eval results neither publish records nor qualify a candidate.
- **Output differences are bugs until explained.** A drop in plain-greedy equality versus
  an earlier candidate, or any output difference between repeated identical requests
  (Eval's `repeatable` and `request_history` gates), is a bug until a root cause is
  demonstrated: the differing kernel or state located and the difference reproduced.
  Never attribute it to rounding, FP8 KV or accumulation order without that evidence,
  even when exactness is only reported.
- **Record through `crit run`:** freeze inputs, create candidates, retain source and
  packages, publish each case's checks, then benchmark with `crit run measure`, which
  replays one pinned text per case for every candidate and binds the checks; seal
  (check telemetry's eligibility first: sealed candidates are immutable). Reserve
  before launch; settle results and cleanup.
  Record parent, changes, rationale, build, results, selection reasoning
  and technique decisions; keep failures visible. Changed source, configuration or
  dependencies need a new candidate.
- **Compare and select** on the same model and workload; explain missing metrics.
  Apply agreed criteria without relaxing them or demanding byte-identical output.
  Spend the `quality_budget` (default 5% perplexity) on speed; drift below it is no
  gain. Speculative output must equal plain greedy only when the run requires
  `exactness`. Failed required checks stay exploratory. Select the best qualified
  candidate and say whether it beats the baseline.

## 4. Deliver a served engine

Include serving unless the user asked for less. Before sealing, implement the
[resident adapter contract](docs/SERVING.md#adapter-contract): declare `chat-json-v1`
and `chat-resident-v1`, export `create_engine(context)`, retain the adapter (native
engines can use a Python wrapper). Load weights once; keep model allocations across
requests and switches; accept an optional leading system message; keep request state
private; honor cancellation and cleanup. Build the adapter on `crit_core.serving.ResidentChat`
so the chat template, tools, per-request sampling and streaming come from Core, and list every
sampling field the decoder applies; `crit run package` refuses an adapter without temperature
or, when the model's template renders tools, without tool calls. Verify with `crit serve`: consecutive requests, streaming, a stop
string, a tool call and its result, candidate switching and Ctrl+C. Report anything unverified.

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

Always give the actual candidate and its serve command:

```text
crit serve --workspace "RUN_ROOT" --run "RUN_ID" --candidate "CANDIDATE_ID" --project "CRIT_ROOT"
```

Clients connect to `http://127.0.0.1:8000/v1`. Also run it with `--write-launchers`
(writes `serve.cmd` and `serve.sh` into the run folder) and name them. Headless agents
deliver transferable artifacts and target-machine instructions. Without the target
hardware, label results unmeasured; never invent measurements or claim to have started
a server on the user's machine.
