<div align="center">

<p align="center">
  <img src="docs/assets/wordmark.png" width="200" alt="wordmark" />
  <br />
</p>

### What if building a tailored, high-performance inference engine took hours instead of weeks?

[![Version 0.1.0](docs/assets/badge-version.svg)](CHANGELOG.md)
[![License Apache-2.0](docs/assets/badge-license.svg)](LICENSE)
[![crit-hub engines](docs/assets/badge-crit-hub.svg)](https://github.com/crit-labs/crit-hub)
[![CI](https://github.com/crit-labs/crit/actions/workflows/ci.yml/badge.svg)](https://github.com/crit-labs/crit/actions/workflows/ci.yml)

[Summary](#summary) • [General vs. tailored](#inference-engines) •
[Methodology](#methodology) •
[Core, Inference, Kernels, Eval](#components) • [Case study](#case-study) • [Limitations](#limitations) •
[Documentation](#documentation)

</div>

---

<a id="summary"></a>

## Summary

**A toolkit for developers and coding agents to quickly build inference engines
tailored to your model and hardware.**
> [!IMPORTANT]
> `crit` combines a runtime foundation, reusable techniques, kernel discovery
> and evaluation. In our first case study, a coding agent built an engine for
> Qwen3.8-27B on one RTX 5090 in about four hours and four candidates, reaching
> 190 tok/s decode with the card's full 244,736-token context.
> You get runnable source, measured candidate comparisons and an OpenAI-compatible API.
> Follow the [building guide](docs/BUILDING_ENGINES.md) yourself, or give your coding
> agent the [agent instructions](AGENTS.md), target model and hardware.

---

<a id="case-study"></a>

<details>
<summary><strong>📋 Case study 01: RTX5090, Qwen3.8-27b@NVFP4</strong> → <a href="https://github.com/crit-labs/crit-hub/tree/main/engines/rtx5090/qwen3.8-27b@NVFP4/mtp7-fp8kv">Inspect engine</a></summary>

<br />

<div align="center">
<table>
  <tr>
    <td align="center" width="16%">
      <strong>

4h 10m</strong>

Development to selection
    </td>
    <td align="center" width="16%">
      <strong>

190 tok/s</strong>

Decode
    </td>
    <td align="center" width="16%">
      <strong>

7,804 tok/s</strong>

Prefill
    </td>
    <td align="center" width="16%">
      <strong>

244,736 tok</strong>

Context
  </td>
    <td align="center" width="16%">
      <strong>

+0.6%</strong>

Perplexity vs reference
    </td>
    <td align="center" width="16%">
      <strong>

0.0098</strong>

Mean KL divergence
    </td>
  </tr>
</table>
</div>

<br />

[`mtp7-fp8kv`](https://github.com/crit-labs/crit-hub/tree/main/engines/rtx5090/qwen3.8-27b@NVFP4/mtp7-fp8kv)
runs Qwen3.8-27B in its original NVFP4 weights on one RTX 5090, with up to 244,736 tokens of
context, as much as fits on the card. One coding agent built it in a single `crit run`, writing
its own kernels on Core without using another inference engine as a reference. It speeds up
decoding with the model's own built-in draft layer, used as released.

| Candidate | What changed | Decode | Prefill | First token |
| --- | --- | ---: | ---: | ---: |
| 1 | Working engine: reads the checkpoint as stored, drafts 7 tokens ahead | 132 tok/s | 5,387 tok/s | 204 ms |
| 2 | Prompts processed in the checkpoint's own 4- and 8-bit formats | 188 tok/s | 7,696 tok/s | 148 ms |
| 3 | Fastest matrix-multiply variant chosen per shape | 190 tok/s | 7,811 tok/s | 142 ms |
| **4** | **Long context: 8-bit KV cache, 244,736 tokens** | **190 tok/s** | **7,804 tok/s** | **142 ms** |

Measured with crit Eval back to back at a 32K context. Decode is the geometric mean over six
prompts, from a chat question to a 14K-token agent session.

- **Long context:** candidate 4 is as fast when built for 244,736 tokens. A prompt that fills
  all of them gets its first token after 2 minutes, then decodes at 152 tok/s.
- **Quality:** perplexity stays within 0.6% of the reference, and drafting never changes the
  output: it is token for token what plain decoding produces.

</details>

---

<a id="inference-engines"></a>

## Inference engines: general vs. tailored

| | General-purpose engine | Tailored engine |
| --- | --- | --- |
| Target | Many models, machines, workloads | One model, machine, workload |
| Design | Shared paths, broad compatibility | Layouts, kernels and memory chosen for the target |
| Effort | Configure an existing engine | Build and test it; crit supplies the parts |
| Best fit | Broad support, ready to use | Maximum performance on a known target |

---

<a id="methodology"></a>

## Methodology

We studied working tailored engines and where specialization makes them fast. The
runtime every engine needs became Core; techniques whose value depends on model and
hardware became Inference entries, with their assumptions made explicit.

---

<a id="components"></a>

<a id="core"></a>

<details>
<summary><strong>📗 Core — the foundation</strong> → <a href="core/README.md">Full guide</a></summary>

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
<summary><strong>📕 Inference — curated techniques</strong> → <a href="inference/README.md">Full guide</a></summary>

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
<summary><strong>📘 Kernels — kernel discovery</strong> → <a href="kernels/README.md">Full guide</a></summary>

<br />

**Kernels helps you find and assess existing kernels** for your data format and
hardware, including the [Hugging Face Kernel Hub](https://huggingface.co/kernels).
It checks declared compatibility, shows mismatches and gaps, and retains chosen
artifacts with exact versions and license notices. Your engine then measures
whether the kernel helps.

</details>

---

<a id="eval"></a>

<details>
<summary><strong>📚 Eval — speed and quality</strong> → <a href="docs/eval/README.md">Full guide</a></summary>

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

<a id="serving"></a>

<details>
<summary><strong>🔌 Serving — OpenAI-compatible API</strong> → <a href="docs/SERVING.md">Full guide</a></summary>

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
| Compare speed, quality and speculative output | [Eval](docs/eval/README.md) |
| Serve an engine as an OpenAI-compatible API | [Serving](docs/SERVING.md) · [Adapter contract](docs/SERVING.md#adapter-contract) |
| Change `crit` or run its tests | [Contributing](docs/CONTRIBUTING.md) · [Licensing](docs/LICENSES.md) |
| See release history | [Changelog](CHANGELOG.md) |
