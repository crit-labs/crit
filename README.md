<div align="center">

<p align="center">
  <img src="docs/assets/wordmark-light.png#gh-light-mode-only" width="200" alt="crit" />
  <img src="docs/assets/wordmark.png#gh-dark-mode-only" width="200" alt="crit" />
  <br />
</p>

### Why wait for someone to optimize an LLM for your GPU when an AI agent can build a specialized inference engine in a few hours?

[![Version 0.10.0](docs/assets/badge-version.svg)](CHANGELOG.md)
[![License Apache-2.0](docs/assets/badge-license.svg)](LICENSE)
[![crit-hub engines](docs/assets/badge-crit-hub.svg)](https://github.com/crit-labs/crit-hub)
[![CI](https://github.com/crit-labs/crit/actions/workflows/ci.yml/badge.svg)](https://github.com/crit-labs/crit/actions/workflows/ci.yml)

[Summary](#summary) • [Quick start](#quick-start) • [What crit solves](#what-crit-solves) • [Case studies](#case-study) •
[How crit works](#how-crit-works) • [API](#api) • [CLI](#cli) • [crit-hub](#crit-hub) • [Limitations](#limitations) • [Documentation](#documentation)

</div>

---

<a id="summary"></a>

## Summary

**`crit` is a toolkit for optimizing or building inference engines tailored to your hardware and desired model.**
> [!IMPORTANT]
> Combines a runtime foundation, reusable techniques, kernel discovery
> and physics-based evaluation. In our first case study, a coding agent built an engine for
> Qwen3.8-27B on one RTX 5090 in about 4 hours and seven candidates,
> reaching 252 tok/s greedy decode with the model's full 262,144-token context.
> You get runnable source, measured candidate comparisons and an OpenAI-compatible API.
> 
> Follow the [building guide](docs/BUILDING_ENGINES.md) yourself, or give your coding
> agent the [agent instructions](AGENTS.md), target model and hardware.

---

<a id="quick-start"></a>

## Quick start

Give one of these prompts to your coding agent and specify your target LLM or target engine:
```bash
Use github.com/crit-labs/crit to build an optimized inference engine for <huggingface_url>.
```

```bash
Use github.com/crit-labs/crit to optimize my own inference engine in <engine_reference>.
```

---

<a id="what-crit-solves"></a>

## What `crit` solves

Less time building an engine. More performance from your hardware.

**The boring stuff has already been built ⏱️** 
> Requests, memory management and cleanup
> eat up development time before optimization even starts. `crit` supplies a clean
> foundation so you can get to the work that makes your engine fast.

**Curated high-impact inference strategies 📊** 
> Don't rebuild inference strategies from scratch. We studied established
> inference engines and distilled their core speedup strategies into reusable code, so you can
> apply what works to your model and hardware quickly.

**Physics-based evaluation suite 🔬** 
> A benchmark score alone doesn't tell
> you what's still left on the table. `crit` measures your hardware's limits and shows
> how close your engine gets, with quality and correctness checks along the way.

---

<a id="case-study"></a>

## Case Studies

> [!NOTE]
> Find the referenced engines in our [`crit-hub`](https://github.com/crit-labs/crit-hub) repository.

We gave `Claude Opus 5.5` a target LLM along with the `crit` repository and let it cook autonomously.

| What it shows | Evidence |
| --- | --- |
| **Fast engines** | Qwen3.8-27B on one RTX 5090: 252 tok/s greedy decode, 4x its own first working candidate, and 7,425 tok/s prefill, with the full 262,144-token context |
| **No quality traded for speed** | Perplexity within 0.1% of the reference, 5.8% of the drift budget; drafting reproduces plain greedy decoding token for token |
| **Hours, not weeks** | 4h 15m and seven candidates from the run's first command to the selected engine, $42 at API list prices |
| **Every step is evidence** | Each candidate is sealed with its exact source and its Eval result, so the table below is a record, not a recollection |
| **Honest about what is left** | Prefill runs at 50% of the GPU's measured compute ceiling; Eval reports that gap next to the score |
| **Next** | Gemma 4 31B (dense), and Qwen3.8-Flash-Next and Nemotron 3.5 Super 120B, mixture-of-experts models larger than VRAM |

---

<details>
<summary><strong>📋 Dense LLM, Qwen architecture</strong> → <a href="https://github.com/crit-labs/crit-hub/tree/main/engines/crit-labs/mtp7-device-confidence">Inspect engine</a></summary>

<br />

**Goal:** Get to a fast, working engine for a dense model as quickly as possible.
**Limit:** A fixed candidate budget.

<div align="center">
<table>
  <tr>
    <td align="center" width="16%">
      <strong>

4h 15m</strong>

Development time
    </td>
    <td align="center" width="16%">
      <strong>

252.4 tok/s</strong>

Decode
    </td>
    <td align="center" width="16%">
      <strong>

7,425 tok/s</strong>

Prefill
    </td>
    <td align="center" width="16%">
      <strong>

150 ms</strong>

TTFT
    </td>
    <td align="center" width="16%">
      <strong>

+0.1%</strong>

Perplexity vs reference
    </td>
    <td align="center" width="16%">
      <strong>

0.0028</strong>

Mean KL divergence
    </td>
  </tr>
</table>
</div>

<br />

[`mtp7-device-confidence`](https://github.com/crit-labs/crit-hub/tree/main/engines/crit-labs/mtp7-device-confidence)
runs Qwen3.8-27B in its original NVFP4 weights on one RTX 5090, with the model's full 262,144
tokens of context. One coding agent built it from scratch on crit 0.5.0, writing its own kernels
on Core without using another inference engine as a reference. It drafts up to 7 tokens per round
with the model's own built-in draft layer and stops once the drafter's confidence in its latest
draft falls below 0.5; the GPU makes that decision inside one CUDA graph per round, so the host
never waits mid-round.

<p align="center">
  <img src="docs/assets/case-qwen-candidates.svg" width="720" alt="Decode and prefill of each of the seven candidates over the run's 4h 15m: decode jumps from 62 to 225 tok/s when drafting arrives with candidate 2 and ends at 252 tok/s; prefill climbs from 5,176 to 7,425 tok/s" />
</p>

| Candidate | What changed | Decode | Prefill | TTFT | Dev time, cumulative | Agent tokens, cumulative | API equivalent, cumulative |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Working engine: reads the checkpoint as stored, 8-bit KV cache, no drafting | 62 tok/s | 5,176 tok/s | 210 ms | 0h 25m | 16.1M | $4.25 |
| 2 | Drafts 4 tokens ahead with the model's draft layer, tensor-core decode GEMVs | 225 tok/s | 5,584 tok/s | 187 ms | 1h 06m | 42.9M | $12.55 |
| 3 | 8-bit prompt attention | 227 tok/s | 6,010 tok/s | 184 ms | 1h 37m | 70.7M | $19.73 |
| 4 | cuBLASLt for the prompt's 4-bit and 8-bit matrix multiplies | 216 tok/s | 7,111 tok/s | 156 ms | 2h 23m | 108.0M | $29.68 |
| 5 | Drafts 7 tokens ahead | 244 tok/s | 7,408 tok/s | 151 ms | 3h 02m | 128.2M | $34.58 |
| 6 | Stops drafting once the drafter's confidence falls below 0.5, decided on the host (superseded: host waits made rounds jitter and failed the lower-context check) | 243 tok/s | 7,102 tok/s | 152 ms | 3h 59m | 154.6M | $41.96 |
| **7** | **The same stopping rule, decided on the GPU inside one CUDA graph per round** | **252 tok/s** | **7,425 tok/s** | **150 ms** | **4h 15m** | **155.7M** | **$42.30** |

The last three columns are running totals from the run's first command (01:16:25 UTC) to each
candidate's seal, and for candidate 7 to its selection, so its row is the whole run. Agent tokens
and API equivalent count the agent's messages in that span, from its Claude Code session transcript:
Claude Opus 5.5 for every message, priced at Anthropic's API list prices ($4/MTok input, $20/MTok
output, $8/MTok 1-hour cache writes, $0.20/MTok cache reads; 99% of the tokens are cache reads). The
agent writes the next candidate while the previous one is measured, so most of candidate 7 was built
before candidate 6 was sealed: from there to its selection added only 16 minutes and 1.1M tokens.

Measured with crit Eval (v6) at the full 262,144-token context. Decode is greedy (temperature 0), the
geometric mean over
six prompts, from a chat question to a 14K-token agent session; Eval decodes each engine's own
output, so candidates 3 and 4 have the same round time but accept different numbers of drafts.

- **Long context:** a prompt that fills all 262,144 tokens gets its first token after 79 s, then
  decodes at 244 tok/s. On prompts outside Eval's speed suite, in 8 languages and code, it decodes
  at 163 tok/s.
- **Quality:** perplexity stays within 0.1% of the reference, 5.8% of the drift budget, and
  drafting never changes the output: it is token for token what plain decoding produces.
- **Sampled requests:** this engine drafts only on greedy requests. With sampling (temperature 0.6 to
  1.0, including the defaults `crit serve` applies) it decodes one token per pass, at 59–60 tok/s on
  the same six cases, measured outside Eval. crit Eval v7 measures sampled decode too.

</details>

---

<details>
<summary><strong>📋 Dense LLM, Gemma architecture</strong> → TBD</summary>

<br />

**TBD.** We are rebuilding this case study with crit 0.5.0; its engine and figures follow shortly.

</details>

---

<details>
<summary><strong>📋 MoE LLM, VRAM overload, Qwen architecture</strong> → TBD</summary>

<br />

**TBD.** We are rebuilding this case study with crit 0.5.0; its engine and figures follow shortly.

</details>

---

<details>
<summary><strong>📋 MoE LLM, VRAM overload, Nemotron architecture</strong> → TBD</summary>

<br />

**TBD.** We are rebuilding this case study with crit 0.5.0; its engine and figures follow shortly.

</details>

---

<a id="how-crit-works"></a>

## How `crit` works

### 1. The workflow

An agent, or you, builds the engine as a series of candidates. Each candidate is a complete,
servable engine, measured the same way, and each next one is chosen from the last one's numbers.

```mermaid
flowchart LR
    recon["Reconnaissance<br/>model, hardware,<br/>crit guides"] --> limits["crit eval<br/>the hardware's limits"]
    limits --> c1["Candidate 1<br/>a working engine"]
    c1 --> measure["crit eval<br/>scores vs limits"]
    measure --> decide{"Decision<br/>algorithm"}
    decide -->|"build: the biggest<br/>gap first"| next["Candidate 2 … X"]
    next --> measure
    decide -->|"stop: nothing left<br/>beyond the noise"| final["Final candidate,<br/>served by crit serve"]
```

1. **Reconnaissance.** Gather everything the engine depends on before writing code: the user's
   model, hardware, workload, quality and context needs; the model's files (`crit run ... recon`
   reads its config, tokenizer, chat template, sampling defaults and draft layers); its model card
   and reference implementation; crit's [agent instructions](AGENTS.md), guides and docs; and any
   [crit-hub](https://github.com/crit-labs/crit-hub) engine to start from.
2. **Learn the hardware's limits.** `crit eval probe` measures the GPU, and Eval turns the
   measurements into the fastest decode, prefill and time to first token this model could reach on it.
3. **Build candidate 1:** a working engine on [Core](#core), measured by Eval. It sets the baseline.
4. **Decide what the next candidate changes.** The [decision algorithm](#decision-algorithm) reads
   the last Eval and names the KPI furthest from its limit, the levers that close the gap, and a
   forecast crit checks after the build.
5. **Build candidate 2, 3, …**, each measured by Eval and compared with its forecast.
6. **Stop at the final candidate** once neither decode nor prefill can still move beyond its noise.
   It is selected, sealed with its source and evidence, and served through an OpenAI-compatible API.

Every step is recorded through the [CLI](cli/README.md): each candidate freezes its source by hash,
each measurement points at that exact package, and the run's cost is counted from the agent's own
transcripts ([run records](docs/RUN_RECORDS.md)).

<a id="eval"></a>

### 2. crit Eval → [Full guide](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md)

Eval runs one fixed suite against every engine: the same prompts, output lengths, stop handling and
quality references, through the same `crit serve` API a client uses. Results from different runs,
agents and machines therefore compare.

**Limits, not just tok/s.** "252 tok/s" alone does not say whether an engine is good or how much is
left. Eval measures the GPU's memory bandwidth and matrix-multiply rates per weight format, and from
the model's files computes what each phase costs at best: decode reads the weights (and the KV cache)
once per pass, so bandwidth sets its floor; prefill is arithmetic, so the GEMM rates set its ceiling.
Every score is reported as a fraction of its limit, with its run-to-run noise, and the gaps say which
term (model pass, draft steps, draft width, drafter acceptance, prefill, first token) would give how
much at its floor. That is what makes "speed of light" a number an agent can aim at.

| KPI | What it measures | Its limit |
| --- | --- | --- |
| **Decode** | Tokens per second over six workloads (chat, code, math, tool calls, a long document, a 14K-token agent session), at the model's own sampling; greedy decode reported beside it | The weights read per pass at the measured bandwidth, and with drafting the best tokens per round the drafter allows |
| **Prefill** | Prompt tokens per second on the two long prompts | The prompt's arithmetic at the measured GEMM rates |
| **Time to first token** | Milliseconds over every workload, where short prompts expose fixed costs | The larger of the prompt's compute and one weight read |
| **Long-context decode and first token** | The same scores along a curve of prompts up to the required context, 262,144 tokens by default | As above, with the KV cache read at each length |
| **Quality** | The share of the drift budget the worst quality set uses: perplexity ratio, mean and p99 KL divergence | Zero drift |

Gates decide whether a result counts at all: output repeats, the same greedy requests give the same
tokens after a long request history, follow-up turns answer the same from a prompt cache, facts at the
start, middle and end of long prompts are retrieved, speculative output equals plain greedy when
required, sampled tokens follow the model's distribution, and quality stays inside the budget.

**Quality against the checkpoint you chose.** Eval measures drift against the checkpoint as stored,
for example an NVFP4 model, scored once in FP32 with the model's official code, not against its
full-precision original. The user chose that checkpoint and its quantization loss; the engine cannot
change it. Comparing with full precision would mix the publisher's loss with the engine's own, so a
careless engine could hide behind the quantization and a careful one would be charged for it. What
Eval reports is exactly what the engine adds, and the budget (perplexity ratio ≤ 1.05, mean KL
≤ 0.05, p99 KL ≤ 0.5 by default) is there to be spent on speed: lower drift than needed is not a gain.

<a id="decision-algorithm"></a>

### 3. The decision algorithm → [Plan records](docs/RUN_RECORDS.md#plan)

Before any code for the next candidate, the agent records a plan, and crit checks it against the last
Eval, which it freezes:

1. **Rank.** crit computes the priorities from the fractions of the limits: of decode and prefill, the
   one further from its limit comes first, both when they are within noise of each other, and time to
   first token never leads alone. Where Eval gives no limit, crit computes one with Eval's cost model.
2. **Compare strategies.** The plan sets the current strategy's ceiling against the best alternative's,
   for example plain decoding against drafting with the model's own draft layer, the alternative held
   to what its drafter is measured to accept.
3. **Name levers and a forecast.** Each lever states its projected change per KPI; the plan gives a
   low, expected and high forecast.
4. **Refuse bad plans.** crit rejects a plan that stops while a KPI is still needed, leaves a gap above
   the noise unanswered, forecasts no gain beyond the noise or only chases time to first token.
5. **Check the outcome.** After Eval, crit sets the forecast next to the measured result and records
   each lever as built, pruned or deferred.

The run stops only when neither decode nor prefill can still move beyond its noise. The agent cannot
drift, overclaim or quit early, and every candidate has a recorded reason to exist.

<a id="core"></a>

### 4. crit Core → [Full guide](core/README.md)

We studied about 30 inference engines and kept what every one of them has to build before it can be
fast: request scheduling, memory admission and tracking, output delivery, stop strings, sampling,
state and prefix caches, model files and config, structured output, and process cleanup after a crash. That minimum became Core, a native SDK with a Python package. Your engine owns
everything that makes it specific (the model, its kernels, its tensor layouts and its execution
strategy) and calls Core for the plumbing. Try the
[Core + Inference walkthrough](inference/examples/core-projection/README.md).

<a id="inference"></a>

### 5. crit Inference → [Full guide](inference/README.md)

We studied about 50 inference engines, from general-purpose runtimes to engines tuned for one model
on one GPU, and extracted what makes each of them fast into separate techniques: speculative decoding
with draft layers, trees and device-side rounds, expert caches and host-resident experts, quantized
projections and KV caches, paged and partitioned attention, prefix checkpoints, CUDA graph reuse and
more, 36 entries in all. Each says when it helps, what is tested and what the engine supplies, so an
agent can adopt it quickly instead of rediscovering it.

The collection keeps growing the same way. The engines stay pinned at exact revisions, and salvage
passes read what changed in them down to individual pull requests, merged or not. A finding enters
crit only with its mechanism and evidence, a crit consumer and the gap it closes, the smallest useful
adaptation and the three [contribution criteria](docs/CONTRIBUTION_CRITERIA.md): value that can be
shown, nothing tuned to one model or GPU, useful across architectures and GPUs. Upstream numbers are
never carried over as crit's, and rejected findings are kept with their reasons.

### More parts


<a id="kernels"></a>

<details>
<summary><strong>📘 Kernels</strong> → <a href="kernels/README.md">Full guide</a></summary>

<br />

**Kernels helps you find and assess existing kernels** for your data format and
hardware, including the [Hugging Face Kernel Hub](https://huggingface.co/kernels).
It checks declared compatibility, shows mismatches and gaps, and retains chosen
artifacts with exact versions and license notices. Your engine then measures
whether the kernel helps.

</details>

---

<a id="models"></a>

<details>
<summary><strong>📙 Models</strong> → <a href="models/README.md">Full guide</a></summary>

<br />

**Models produces transformed weights** when a run allows them, starting with
post-training quantization to FP8 or NVFP4 (round-to-nearest or GPTQ). It exports a
checkpoint engines can read, runs it in your engine through Eval against the original
model, and records the recipe, calibration data, seed, file hashes and cost.

</details>

---

<a id="api"></a>
<a id="serving"></a>

<details>
<summary><strong>🔌 API</strong> → <a href="api/README.md">Overview</a> · <a href="docs/SERVING.md">Full guide</a></summary>

<br />

`crit serve` loads a sealed candidate, keeps it resident and answers OpenAI-compatible
chat requests, including streaming, tool calls, stop strings and structured output. The same engine
also speaks the [Anthropic Messages, OpenAI Responses and Ollama APIs](docs/SERVING.md#other-api-dialects),
so Claude Code, Codex and local chat UIs need only a base URL, and `GET /metrics` serves
[Prometheus metrics](docs/SERVING.md#metrics). Eval measures every candidate through this same
server, so the numbers you read are the numbers you get. Engines implement the
[adapter contract](docs/SERVING.md#adapter-contract).

</details>

---

<a id="cli"></a>

<details>
<summary><strong>📓 CLI</strong> → <a href="cli/README.md">Overview</a></summary>

<br />

One `crit` command drives everything: `crit run` keeps the run's records (plans, candidates,
measurements, selection, cost), `crit eval` runs the common suite, `crit serve` serves an engine,
and `crit kernels`, `crit models` and `crit inference` reach the other parts. Records are written
only through it, so each plan, candidate, Eval and seal is checked as it is recorded.

</details>

---

## Miscellaneous

---

<a id="crit-hub"></a>

<details>
<summary><strong>🔭 crit-hub</strong> → <a href="https://github.com/crit-labs/crit-hub">Repository</a></summary>

<br />

[`crit-hub`](https://github.com/crit-labs/crit-hub) collects bespoke inference engines in one repository 
to make discovery super easy. Each engine pins its runnable source, the exact checkpoint it was built on, what 
was measured, its known limits and a link to the repository of whoever contributed it. 
Use them as jump-off point for further optimization.

</details>

---

<a id="limitations"></a>

<details>
<summary><strong>🚫 Limitations</strong></summary>

<br />

| Area | Current status |
| --- | --- |
| Multi-GPU setups | Expected to work with engine-specific partitioning and communication; not yet tested end to end. |
| macOS | Not yet tested. Current CI covers Windows and Linux. |
| AMD and Apple GPUs | No bundled ROCm or Metal backend. The current GPU backend uses CUDA. |
| Model formats | Hugging Face safetensors, PyTorch state dicts (`.bin`, `.pt`) and Optimum ONNX exports, unquantized or [quantized](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md#other-weight-formats-and-quantization-schemes) as FP8, NVFP4, compressed-tensors int, AWQ, GPTQ, MLX, bitsandbytes or MatMulNBits, and [GGUF](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md#gguf-models), a lone `.gguf` included: its config, tokenizer and chat template are derived from its metadata for the `llama`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35` and `qwen35moe` architectures and byte-level BPE tokenizers (others need the base model's files beside it), and Eval's reference reads those architectures. A new format is one `ModelFormat` in `crit_core.model`. Core's C++ model library reads safetensors only; engines read a GGUF's tensors through `crit_core.model`'s offsets. |

</details>

---

<a id="documentation"></a>

## Documentation

| To… | Read |
| --- | --- |
| Build and compare engines | [Building engines](docs/BUILDING_ENGINES.md) · [Agent instructions](AGENTS.md) · [Run records](docs/RUN_RECORDS.md) |
| Integrate the native runtime | [Core](core/README.md) · [Contracts](core/contracts/README.md) · [Core + Inference walkthrough](inference/examples/core-projection/README.md) |
| Choose or adapt a technique | [Inference](inference/README.md) |
| Find and adapt kernels | [Kernels](kernels/README.md) |
| Quantize weights and record the transform | [Models](models/README.md) |
| Compare speed, quality and speculative output | [Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md) |
| Serve an engine through the OpenAI, Anthropic or Ollama API | [API](api/README.md) · [Serving](docs/SERVING.md) · [Adapter contract](docs/SERVING.md#adapter-contract) |
| Find a `crit` command | [CLI](cli/README.md) |
| Change `crit` or run its tests | [What belongs in crit](docs/CONTRIBUTION_CRITERIA.md) · [Contributing](docs/CONTRIBUTING.md) · [Licensing](docs/LICENSES.md) |
| See release history | [Changelog](CHANGELOG.md) |
