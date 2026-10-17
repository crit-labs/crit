<div align="center">

<p align="center">
  <img src="docs/assets/wordmark.png" width="200" alt="wordmark" />
  <br />
</p>

### What if building a tailored, high-performance inference engine took hours instead of weeks?

[![Version 0.2.0](docs/assets/badge-version.svg)](CHANGELOG.md)
[![License Apache-2.0](docs/assets/badge-license.svg)](LICENSE)
[![crit-hub engines](docs/assets/badge-crit-hub.svg)](https://github.com/crit-labs/crit-hub)
[![CI](https://github.com/crit-labs/crit/actions/workflows/ci.yml/badge.svg)](https://github.com/crit-labs/crit/actions/workflows/ci.yml)

[Summary](#summary) • [Case studies](#case-study) • [General vs. tailored](#inference-engines) •
[Methodology](#methodology) • [Limitations](#limitations) • [Documentation](#documentation)

</div>

---

<a id="summary"></a>

## Summary

**A toolkit for developers and coding agents to quickly build inference engines
tailored to your model and hardware.**
> [!IMPORTANT]
> `crit` combines a runtime foundation, reusable techniques, kernel discovery
> and evaluation. In our first case study, a coding agent built an engine for
> Qwen3.8-27B on one RTX 5090 in about 3.5 hours and five candidates,
> reaching 216 tok/s decode with the model's full 262,144-token context.
> You get runnable source, measured candidate comparisons and an OpenAI-compatible API.
> Follow the [building guide](docs/BUILDING_ENGINES.md) yourself, or give your coding
> agent the [agent instructions](AGENTS.md), target model and hardware.

---

<a id="case-study"></a>

## Case Studies

> [!NOTE]
> Find the referenced engines in our [`crit-hub`](https://github.com/crit-labs/crit-hub) repository.

We gave `Claude Opus 5.5` a target LLM along with the `crit` repository and let it cook autonomously. 
Here are the results.

---

<details>
<summary><strong>📋 Case study 01: RTX5090, 96GB RAM, Qwen3.8-27b@NVFP4</strong> → <a href="https://github.com/crit-labs/crit-hub/tree/main/engines/rtx5090/qwen3.8-27b@NVFP4/mtp7-fp8attn">Inspect engine</a></summary>

<br />

<div align="center">
<table>
  <tr>
    <td align="center" width="16%">
      <strong>

3h 33m</strong>

Development to selection
    </td>
    <td align="center" width="16%">
      <strong>

216 tok/s</strong>

Decode
    </td>
    <td align="center" width="16%">
      <strong>

8,917 tok/s</strong>

Prefill
    </td>
    <td align="center" width="16%">
      <strong>

262,144 tok</strong>

Context
  </td>
    <td align="center" width="16%">
      <strong>

+0.3%</strong>

Perplexity vs reference
    </td>
    <td align="center" width="16%">
      <strong>

0.0094</strong>

Mean KL divergence
    </td>
  </tr>
</table>
</div>

<br />

[`mtp7-fp8attn`](https://github.com/crit-labs/crit-hub/tree/main/engines/rtx5090/qwen3.8-27b@NVFP4/mtp7-fp8attn)
runs Qwen3.8-27B in its original NVFP4 weights on one RTX 5090, with the model's full 262,144
tokens of context. One coding agent built it from scratch in a single `crit run`, writing its
own kernels on Core without using another inference engine as a reference. It speeds up
decoding with the model's own built-in draft layer, used as released.

| Candidate | What changed | Decode | Prefill | First token |
| --- | --- | ---: | ---: | ---: |
| 1 | Working engine: reads the checkpoint as stored, 8-bit KV cache, drafts 4 tokens ahead | 174 tok/s | 5,177 tok/s | 181 ms |
| 2 | Draft layer in 4-bit, draft width adapted per request, fused decode passes | 218 tok/s | 7,530 tok/s | 138 ms |
| 3 | 8-bit prompt attention scores, fastest matrix-multiply variant recorded per shape | 217 tok/s | 8,113 tok/s | 127 ms |
| 4 | 8-bit decode attention, faster prompt epilogues and convolution | 216 tok/s | 8,674 tok/s | 120 ms |
| **5** | **Prompt attention fully in 8-bit; decode attention from 4 reverted** | **216 tok/s** | **8,917 tok/s** | **120 ms** |

Measured with crit Eval at the full 262,144-token context. Decode is the geometric mean over
six prompts, from a chat question to a 14K-token agent session.

- **Long context:** a prompt that fills all 262,144 tokens gets its first token after 69 s,
  then decodes at 217 tok/s. Candidate 4's 8-bit decode attention was slower in real rounds,
  so candidate 5 dropped it.
- **Quality:** perplexity stays within 0.3% of the reference, and drafting never changes the
  output: it is token for token what plain decoding produces.

</details>

---

<details>
<summary><strong>📋 Case study 02: RTX5090, 96GB RAM, Qwen3.8-Flash-Next@FP4-Local</strong> → <a href="https://github.com/crit-labs/crit-hub/tree/main/engines/rtx5090/qwen3.8-flash-next@FP4-Local/mtp3-expert-cache">Inspect engine</a></summary>

<br />

<div align="center">
<table>
  <tr>
    <td align="center" width="16%">
      <strong>

13h 56m</strong>

Development to selection
    </td>
    <td align="center" width="16%">
      <strong>

88 tok/s</strong>

Decode
    </td>
    <td align="center" width="16%">
      <strong>

1,818 tok/s</strong>

Prefill
    </td>
    <td align="center" width="16%">
      <strong>

262,144 tok</strong>

Context
    </td>
    <td align="center" width="16%">
      <strong>

+0.03%</strong>

Perplexity vs reference
    </td>
    <td align="center" width="16%">
      <strong>

0.0012</strong>

Mean KL divergence
    </td>
  </tr>
</table>
</div>

<br />

[`mtp3-expert-cache`](https://github.com/crit-labs/crit-hub/tree/main/engines/rtx5090/qwen3.8-flash-next@FP4-Local/mtp3-expert-cache)
runs Qwen3.8-Flash-Next, a 102 GB mixture-of-experts checkpoint with 512 NVFP4 experts per
layer, on one 32 GB RTX 5090 with the model's full 262,144 tokens of context. The weights are
used as stored: attention, the shared expert and the KV cache stay on the GPU, the routed
experts (about 61 GB) sit in pinned host memory, and a GPU cache of about 7,800 slots holds the
hot ones. The coding agent wrote it from scratch on Core, again without another inference engine
as a reference, and drafts 3 tokens ahead with the model's own MTP block, used as released.

| Candidate | What changed | Decode | Prefill | First token |
| --- | --- | ---: | ---: | ---: |
| 1 | Working engine: stored formats, experts in host memory behind a GPU cache | 36 tok/s | 1,098 tok/s | 2.19 s |
| 2 | Drafts 3 tokens ahead, decode in CUDA graphs, unused KV space lent to the expert cache | 82 tok/s | 1,916 tok/s | 1.63 s |
| 3 | Weights decoded once per pass, CPU share of short prompts' experts | 81 tok/s | 1,744 tok/s | 1.43 s |
| 4 | Second run, from candidate 2 with candidate 3's exact changes | 80 tok/s | 1,737 tok/s | 1.79 s |
| **5** | **Faster decode GEMVs and expert sums, 8-bit KV cache** | **88 tok/s** | **1,818 tok/s** | **1.60 s** |

Measured with crit Eval at the full 262,144-token context. The work took two `crit run`s back to
back, from 14:37 to 04:34. The first run's request asked for 1 GiB of free VRAM instead of the
machine's 0.3 GiB, so none of its candidates qualified, and the second continued from its
candidate 2. The selected record is candidate 5 recorded again with the same source, because
its first benchmarks were published before their checks.

- **Bottleneck:** decode is bound by expert misses over the host link (about 27 GB/s): each
  round reads 88 to 280 missed experts. A prompt that fills all 262,144 tokens gets its first
  token after 99 s, then decodes at 65 tok/s.
- **Quality:** perplexity stays within 0.03% of the reference. With the 8-bit KV cache,
  drafted output equals plain decoding on 3 of 6 cases; this run reported exactness rather
  than requiring it.

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
runtime every engine needs became Core. Techniques whose value depends on model and
hardware became Inference entries, with their assumptions made explicit.

---

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

<a id="models"></a>

<details>
<summary><strong>📙 Models — weight transforms</strong> → <a href="models/README.md">Full guide</a></summary>

<br />

**Models produces transformed weights** when a run allows them, starting with
post-training quantization to FP8 or NVFP4 (round-to-nearest or GPTQ). It exports a
checkpoint engines can read, runs it in your engine through Eval against the original
model, and records the recipe, calibration data, seed, file hashes and cost.

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

## Miscellaneous

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
| Quantize weights and record the transform | [Models](models/README.md) |
| Compare speed, quality and speculative output | [Eval](docs/eval/README.md) |
| Serve an engine as an OpenAI-compatible API | [Serving](docs/SERVING.md) · [Adapter contract](docs/SERVING.md#adapter-contract) |
| Change `crit` or run its tests | [Contributing](docs/CONTRIBUTING.md) · [Licensing](docs/LICENSES.md) |
| See release history | [Changelog](CHANGELOG.md) |
