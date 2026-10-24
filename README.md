<div align="center">

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/wordmark.png" />
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/wordmark-light.png" />
    <img src="docs/assets/wordmark.png" width="200" alt="wordmark" />
  </picture>
  <br />
</p>

### Why wait for someone to optimize an LLM for your GPU when an AI agent can build a specialized inference engine in a few hours?

[![Version 0.7.0](docs/assets/badge-version.svg)](CHANGELOG.md)
[![License Apache-2.0](docs/assets/badge-license.svg)](LICENSE)
[![crit-hub engines](docs/assets/badge-crit-hub.svg)](https://github.com/crit-labs/crit-hub)
[![CI](https://github.com/crit-labs/crit/actions/workflows/ci.yml/badge.svg)](https://github.com/crit-labs/crit/actions/workflows/ci.yml)

[Summary](#summary) • [Quick start](#quick-start) • [What crit solves](#what-crit-solves) • [Case studies](#case-study) •
[Methodology](#methodology) • [crit-hub](#crit-hub) • [Serving](#serving) • [Limitations](#limitations) • [Documentation](#documentation)

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

| What it shows               | Evidence |
|-----------------------------| --- |
| **`crit` engines are fast** | Qwen3.8-27B decodes greedily at 252 tok/s and prefills at 7,425 tok/s |
| **Dense and MoE models**    | Dense: Qwen3.8-27B; Gemma 4 31B TBD. Mixture-of-experts larger than VRAM: Qwen3.8-Flash-Next and Nemotron 3.5 Super 120B TBD |
| **Different architectures** | Qwen (gated DeltaNet and attention); Gemma and Nemotron TBD |
| **Hours, not weeks**        | Qwen3.8-27B in 4h 15m from the run's first command to the selected engine; the others TBD |

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

<a id="methodology"></a>

## Methodology

We studied working tailored engines and where specialization makes them fast. The
runtime every engine needs became Core. Techniques whose value depends on model and
hardware became Inference entries, with their assumptions made explicit.

---

<a id="core"></a>

<details>
<summary><strong>📗 Core</strong> → <a href="core/README.md">Full guide</a></summary>

<br />

**Core provides what every engine needs**: request scheduling, memory tracking,
output delivery and safe cleanup. Your engine owns the model, tokenizer and layouts,
and calls Core's libraries for sampling, state cache and model files. The native
runtime supports generation and scoring, with optional CUDA and concurrent requests.
Try the [Core + Inference walkthrough](inference/examples/core-projection/README.md).

</details>

---

<a id="inference"></a>

<details>
<summary><strong>📕 Inference</strong> → <a href="inference/README.md">Full guide</a></summary>

<br />

Inference holds **techniques with explanations, implementation code and worked
examples**. Start from a bottleneck:

- **Reuse prompt work.** [Prefix checkpoints](inference/checkpoint-storage/README.md)
  resume from stored state when a prompt shares an earlier prefix.
- **Keep useful weights close.** [Expert residency](inference/expert-residency/README.md)
  retains reusable expert weights within a memory budget.
- **Generate several tokens per round.** [Speculative decoding](inference/mtp-decode/README.md)
  proposes a continuation, verifies it and keeps the accepted tokens.
- **Reuse GPU execution.** Core's optional [CUDA graph harness](core/src/backends/cuda/README.md#graph-capture)
  captures repeated work for replay to cut launch overhead.

Each technique says when it helps, what is tested and what your engine supplies.

</details>

---

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

<a id="eval"></a>

<details>
<summary><strong>📚 Eval</strong> → <a href="docs/eval/README.md">Full guide</a></summary>

<br />

**Eval gives engines a common set of tests**: fixed prompts, output lengths and a
shared quality reference per model.

- **Speed** on chat, code, math, tool calling, long-document summarization and a long
  agent session, with decode, prefill and time-to-first-token scores against their
  physical ceilings.
- **Quality**: the engine's predictions against the reference on fixed conversations,
  across its execution paths.
- **Speculative checks**: drafting against plain greedy decoding, exact when the run requires it.
- **Repeatability**: one pinned decode text per model, and paired A/B runs for small changes.
- **Profiling**: kernels ranked per phase by GPU time and launch width.

Results keep samples, settings and skipped cases visible.

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

<a id="serving"></a>

<details>
<summary><strong>🔌 Serving</strong> → <a href="docs/SERVING.md">Full guide</a></summary>

<br />

`crit serve` loads a sealed candidate, keeps it resident and answers OpenAI-compatible
chat requests, including streaming, stop strings and structured output. Engines
implement the [adapter contract](docs/SERVING.md#adapter-contract).

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
| Model formats | Hugging Face safetensors, PyTorch state dicts (`.bin`, `.pt`) and Optimum ONNX exports, unquantized or [quantized](docs/eval/README.md#other-weight-formats-and-quantization-schemes) as FP8, NVFP4, compressed-tensors int, AWQ, GPTQ, MLX, bitsandbytes or MatMulNBits, and [GGUF](docs/eval/README.md#gguf-models), a lone `.gguf` included: its config, tokenizer and chat template are derived from its metadata for the `llama`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35` and `qwen35moe` architectures and byte-level BPE tokenizers (others need the base model's files beside it), and Eval's reference reads those architectures. A new format is one `ModelFormat` in `crit_core.model`. Core's C++ model library reads safetensors only; engines read a GGUF's tensors through `crit_core.model`'s offsets. |

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
| Compare speed, quality and speculative output | [Eval](docs/eval/README.md) |
| Serve an engine as an OpenAI-compatible API | [Serving](docs/SERVING.md) · [Adapter contract](docs/SERVING.md#adapter-contract) |
| Change `crit` or run its tests | [What belongs in crit](docs/CONTRIBUTION_CRITERIA.md) · [Contributing](docs/CONTRIBUTING.md) · [Licensing](docs/LICENSES.md) |
| See release history | [Changelog](CHANGELOG.md) |
