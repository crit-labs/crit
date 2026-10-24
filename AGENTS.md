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
  context for the model on it: the user's wanted context (`context_tokens` in
  `settings.toml`, default 262,144), capped by the model's positions and the KV cache
  that fits. The context is the user's, like the sampling: set `kv_tokens` or
  `context_tokens` only when the user asks. Preflight model, tools and dependencies
  before starting the clock.
- **Study the model before the first candidate.** After `create`, run
  `crit run --id RUN_ID recon` (`candidate` refuses a model folder without it): it reads `generation_config.json` (sampling defaults),
  the tokenizer files and chat templates (BOS/EOS and stop tokens, template switches
  such as `enable_thinking`) and `config.json` (rope scaling, MTP or draft heads,
  attention layout; for a lone GGUF, the files its metadata derives), writes the facts with file hashes to `work/recon.json` and
  scaffolds `work/recon.md`. Finish `recon.md` from the model card (recommended
  sampling per mode, e.g. thinking vs non-thinking, known issues, context limits) and
  the reference implementation in transformers (norm and attention quirks),
  citing sources. Where they override the files, edit the `frozen` block of
  `recon.json` and say why under Decisions. Every later candidate freezes that block
  (sampling, chat template and its variables, stop tokens) as `configuration["recon"]`
  instead of defaulting to greedy. `recon.json` also fingerprints the checkpoint (architecture,
  weight layout, tokenizer, template), and every candidate records it; when the engine should
  run a related checkpoint (a finetune, a re-quantization, a sibling model), `crit run --id RUN_ID
  recon --compare OTHER_MODEL` says what differs for it, and an equal fingerprint means the weight
  values differ only.
- **Start from a crit-hub engine when one exists for the model.** `create --from-engine ENGINE_DIR`
  (the engine's folder in a crit-hub checkout) records it as the run's origin: its source pin, the
  checkpoint it was measured on and its published Eval results under `captured/origin/`. Check the
  source out at the pinned commit and make it the first candidate, with `"lineage": "origin-engine"`
  in the proposal, built and measured before any change: it is the baseline. `resolve` records
  whether your model is that exact checkpoint; if not, the first candidate absorbs what
  `recon --compare` says differs.
- **The sampling is the model's, not yours.** Recon records the sampling Eval measures under
  "Eval sampling" in `recon.md`: the model's `generation_config.json` (temperature, top_p,
  top_k, min_p when it samples by default), else the protocol default (temperature 1.0, no
  filters). The served default (`frozen.sampling`) follows it, and a candidate refuses a frozen
  sampling that differs. Never pick a temperature, not even one the model card recommends for a
  mode; write card recommendations down, but only the user overrides the sampling, with
  `eval_sampling` at run create. Pass `crit eval prepare --sampling` only when the run declares
  `eval_sampling`, and then exactly its settings. Write the drafting plan in `recon.md` for
  these settings before the first candidate.
- **Build for both decode modes.** Eval scores both, on the same six speed cases.
  Greedy (temperature 0): speculative output must stay token-exact against plain greedy
  (the exactness check). Sampled, at the Eval sampling: the headline decode, its goal and
  the stop record's decode; verify drafts by rejection sampling (accept a draft with
  probability min(1, p/q) under the processed target distribution p, sample from
  max(p - q, 0) on the first rejection; a greedy draft is q one-hot) so the output keeps the
  model's distribution (the `sampling` gate), and sample on the device inside the decode
  graph rather than copying logits to the host for every token
  ([token-sampling](inference/token-sampling/README.md)'s `enqueue_verify` feeds
  [device-speculative-round](inference/device-speculative-round/README.md); its host
  oracle is [mtp-decode](inference/mtp-decode/README.md)). A drafter that only helps
  greedy requests leaves the headline at one token per pass. Requests at other settings
  must stay correct too.
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
  quantize the source with `crit models ptq`, measure the served output with
  `crit eval run --url` against the source's Eval data, bind the result with
  `crit models eval`, and keep both records with the candidate.
- **[Eval](docs/eval/README.md):** measure each candidate with `crit run ... eval`: it
  scores the data's fixed quality texts on the model's checkpoint once (`crit eval
  reference`), serves the packaged candidate through `crit serve` and runs Eval over the
  chat API, quality through the engine's own score call. Use the same prepared data for
  every engine of one model. Hardware telemetry (tokens per round, acceptance, memory,
  kernels) comes from `crit run ... measure` and Eval's adapter commands through an eval
  adapter. A Core engine built as a shared
  [engine library](core/src/runtime/README.md#engine-libraries) gets the Eval and serve
  adapters from `crit_core.native`.

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
   kernels, the parent's result. Include every lever that helps a goal without costing
   a higher priority; effort is never a reason to leave something out. Record the plan
   with `crit run ... plan --record plan.json` after the parent's Eval and before
   writing code ([Run records](docs/RUN_RECORDS.md#plan)), and name its `plan_id` in the
   proposal. A plan proved wrong before its candidate exists is replaced by a new plan
   with `supersedes`.
3. **Build** it (`crit run ... candidate`); fix or prune what is slow or fails a gate
   until the plan is implemented. Prune a part only after measuring it against its
   floor (Eval's floors per term, the profile's kernels). The proposal gives every lever
   of its plan a fate in `lever_fates`: built, pruned or deferred, with the reason for a
   pruned or deferred one.
4. **Evaluate:** `crit eval run --baseline FIRST_RESULT`, `crit eval kernels`, then
   `crit eval status RESULT --baseline PARENT_RESULT`. A failed gate, an exactness drop
   or a repeat difference is a bug and the next candidate's first item.
5. **Revisit recon** when a result contradicts it, or when the engine switches behaviour
   at a threshold (prefetch, sparse attention, chunk sizes) Eval does not cross: update
   `recon.md` (and the frozen block if needed) and add a run case that crosses it
   before the next candidate.
6. **Repeat** from step 2 until the targets are met, the candidate budget (a number of
   candidates, not time) is used, neither decode nor prefill is needed or the run is
   blocked. When neither is needed the run stops even with candidates left: record a
   stop plan (`"decision": "stop"`) from the newest Eval, then stop with
   `search_exhausted`. On resume, re-measure first.

- **Strategy first.** Working baseline, measured bottlenecks, then compare strategies,
  not only settings. Reach the highest-ceiling strategy in working form before tuning
  kernels; sessions can end early and tuning does not transfer between strategies.
- **Eval sets the goals.** `crit eval status` gives one goal per KPI: decode, prefill,
  time to first token, and long-context decode and time to first token, each from
  current score to physical limit with its fraction of the limit and its noise. From
  crit-eval-v7 the decode goal is the sampled decode. Never weigh by the end-to-end score.
- **The priority rule orders them.** Decode and prefill are each *needed* while some
  lever can still raise it by more than its run-to-run noise; one stops being needed
  when the distance left is within the noise, or when every Eval gap on it above the
  noise has a physical cause or a measured rejection. Long-context decode counts toward
  decode and long-context time to first token toward prefill; a family's fraction of
  its limit is the lower of its members'. The needed one further from its limit is p1,
  the other p2; both are p1 when their fractions are within the noise of each other.
  Time to first token is p2 or p3: never p1 and never the reason for a candidate on its
  own. Priorities decide conflicts and order, never what is left out: every lever that
  costs no higher priority goes in. crit computes the priorities in each plan. When Eval
  gives decode or long-context decode no limit, crit computes one from the parent's Eval:
  the measured tokens per round over the pass floor (bytes per pass over bandwidth).
  Where it cannot, the plan states an estimated limit and how it was derived (for example a
  drafter's expected tokens per round over the pass floor). The current strategy's limit is
  Eval's, then crit's, then the estimate; crit ranks decode by the best strategy's ceiling,
  the higher of that limit and the alternative strategy's ceiling the plan gives in
  `strategy.alternative.limits` (its tokens per round and draft steps: crit computes the
  round floor), whichever strategy the plan chooses, and names the source of each fraction.
  In runs with `"plan_rules": 2` the alternative's tokens per round is at most its draft
  steps + 1; an unknown noise takes the largest noise the Eval knows (else 5%), a family's
  noise is the largest of its members', the two families' noise bands combine in quadrature
  for the near tie, and each gap and lever change is weighed against its own KPI's noise.
  There crit's computed limit and the alternative's ceiling follow Eval's cost model: tokens
  per round over the pass floor plus the draft steps at Eval's draft floor (where the draft
  width is unknown, the pass floor alone, a labelled upper bound). An alternative names the
  `drafter` it assumes (`measured` or `new`); its tokens per round stays within the measured
  drafter's acceptance bound at its draft steps (at most 8 or the parent's draft width) unless
  the plan cites another candidate's Eval attempt that measured the new drafter. The first
  plan's decision
  follows its needed flags like every other plan's. Near the limit (both families at 0.85 of
  it or above) a near tie puts only the lower one in p1, and a family marked not needed below
  0.85 answers why the rest of its limit is out of reach (`gaps` entry `limit`, answer `cause`).
- **Plans are checkable.** Each plan forecasts every KPI (low, expected, high), names the
  current strategy's ceiling and the best alternative's, groups levers that only pay
  together, and puts no lever below the noise into a candidate on its own. Work the rules
  require that Eval does not score (for example sampled speculation before Eval measures
  it) is a requirement lever (`"kind": "requirement"` with the rule it `serves`): it needs
  no projected change and never counts as the plan's lever for a KPI. After the
  candidate's Eval crit writes the plan's outcome next to the forecast; a drop in
  greedy decode, held-out decode or follow-up time to first token beyond its noise is
  flagged there, outside the priorities.
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
- **Never train or fine-tune a drafter.** Use the checkpoint's drafter, a published
  companion draft model as released, or a drafter the user produced with
  [`crit models drafter`](models/README.md#drafters) and handed you with its transform
  record: retain that record with the candidate's source, name the drafter's
  `output.manifest_sha256` in the configuration, and verify every draft with the stored
  target, so `weights` stays `as_stored`.
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
  Retain adapter, applied configuration, data and results; report skipped
  cases. Eval results neither publish records nor qualify a candidate.
- **Output differences are bugs until explained.** A drop in plain-greedy equality versus
  an earlier candidate, or any output difference between repeated identical requests
  (Eval's `repeatable` and `request_history` gates), is a bug until a root cause is
  demonstrated: the differing kernel or state located and the difference reproduced.
  Never attribute it to rounding, FP8 KV or accumulation order without that evidence,
  even when exactness is only reported.
- **Record through `crit run`:** freeze inputs, create candidates, retain source and
  packages, publish each case's checks, then benchmark with `crit run measure`, which
  streams each case's chat request to the candidate served through `crit serve`; seal
  (check telemetry's eligibility first: sealed candidates are immutable). Reserve
  before launch; settle results and cleanup.
  Record parent, changes, rationale, build, results, selection reasoning
  and technique decisions; keep failures visible. Changed source, configuration or
  dependencies need a new candidate.
- **Compare and select** on the same model and workload; explain missing metrics.
  Apply agreed criteria without relaxing them or demanding byte-identical output.
  Spend the `quality_budget` (defaults: perplexity ratio 1.05, mean KL 0.05, p99 KL 0.5;
  [Run records](docs/RUN_RECORDS.md#request)) on speed; drift below it is no
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
Record what the run cost with `crit run ... cost --transcript <your session transcript> --agent "<agent and
model>"`: it attributes your model requests to the candidates built in their spans and writes `cost.json`
(wall time, tokens, API-equivalent price per candidate), which crit-hub shows on the engine's card.

Always give the actual candidate and its serve command:

```text
crit serve --workspace "RUN_ROOT" --run "RUN_ID" --candidate "CANDIDATE_ID" --project "CRIT_ROOT"
```

Clients connect to `http://127.0.0.1:8000/v1`; Anthropic and Ollama clients (Claude Code,
local chat UIs) use `http://127.0.0.1:8000` ([other API dialects](docs/SERVING.md#other-api-dialects)),
and `GET /metrics` serves Prometheus metrics. Also run it with `--write-launchers`
(writes `serve.cmd` and `serve.sh` into the run folder) and name them. Headless agents
deliver transferable artifacts and target-machine instructions. Without the target
hardware, label results unmeasured; never invent measurements or claim to have started
a server on the user's machine.
