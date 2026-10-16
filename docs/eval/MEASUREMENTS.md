# Eval measurements

[Overview](README.md) · [Measurements](MEASUREMENTS.md) · [Context](CONTEXT.md) · [Adapter](ADAPTER.md)

## Speed

[suite/speed.json](../../eval/suite/speed.json) holds six greedy cases, each generating 256 tokens:

| Case | Prompt | Why |
| --- | --- | --- |
| `chat` | short prose question | everyday chat |
| `code` | write a Python module | code drafts most easily |
| `math` | multi-step word problems | reasoning |
| `tool-call` | coding agent with tool definitions and one file read, asked for edit calls | tool calls with code payloads |
| `long-summary` | 3,584-token document, summarize | long-context prefill and decode |
| `agent-long` | 14,336-token agent session: tools, task, a grep, eight file reads, edit request | agent turns over a long history |

Agent cases render [tools.json](../../eval/suite/agent/tools.json) and frozen tool results
([suite/agent](../../eval/suite/agent)) with the model's chat template. `agent-long` needs
14,592 tokens of context; an engine with less skips it and its decode score is incomplete.

Stop tokens are disabled; output past the model's end token is flagged `stop_crossed`.
Defaults: 1 warmup, 3 repetitions ([protocol.json](../../eval/protocol.json)). Time to first token
gives prefill, the remaining tokens decode. The **decode score** is the geometric mean of the six
case medians; each case also reports tokens per round. The **prefill score** is the geometric mean of prompt tokens ÷ TTFT over `long-summary`
and `agent-long` (`prefill_cases`). The **TTFT score** is the geometric mean of TTFT in ms over
every case, where short prompts expose fixed costs.

## Pinned continuation

Slightly different arithmetic yields different text and draft acceptance. `crit eval pin` records one engine's plain greedy output of every speed case (it must
repeat exactly) as the data directory's `continuations.json`, identified by the sha256 of its
token ids; a data directory keeps one. When it exists and the adapter declares `replay`, `run`
measures every speed KPI by replaying that text through the engine's speculative loop, so every
engine decodes the same tokens. Two free generations per case still feed the repeatability gate
and exactness. `speed.decode_text` records which text was decoded.

## Limits

**Device ceilings** come from `crit eval probe` on the target machine. It builds [probe/device_probe.cu](../../eval/probe/device_probe.cu) with
`nvcc`, runs it three times and keeps medians: copy bandwidth (2 GiB device-to-device copies,
read plus written) and dense cuBLASLt GEMM rates with FP32 accumulation at prefill-like shapes
for BF16, FP8 E4M3 and NVFP4 (16-element E4M3 block scales), plus total and least free device
memory and host memory. Pass it with `run --device`; `--bandwidth GBS` overrides the bandwidth.

**Prefill ceiling.** `prepare` counts arithmetic per token from safetensors headers and config
([crit_eval/ceiling.py](../../eval/crit_eval/ceiling.py)): 2 × rows × columns for every
projection in its stored format (BF16, FP8 or NVFP4), excluding gathered embeddings, the vision
tower, draft layers and the output head; 4 × heads × head dim per query-key pair in each
full-attention layer; and the chunked linear-attention recurrence. The floor is matmul FLOPs at
each format's GEMM rate plus attention and recurrence at the BF16 rate.

**TTFT floor:** the larger of the prompt's compute floor and one weight read (`bytes_per_pass`,
else the language-model and output-head bytes) plus the prompt's KV writes at device bandwidth.

**Decode headroom.** Every case reports `round_ms`, decode time per target pass. Optional
adapter fields add the ceiling:

- `draft_width`: **saturation**, tokens per round ÷ (draft width + 1).
- `bytes_per_pass` (weights read, not gathered embeddings, plus recurrent state) and the
  bandwidth: the **pass floor**, compared with the measured one-token step. Each case adds the
  KV read at its context: BF16 K and V of every full-attention layer, or `kv_bytes_per_token`.
- `accepted` per round from `generate()`: tokens per round at every width up to the measured one.
  With the draft cost (`draft_ms`, else round time beyond one pass ÷ width) Eval projects the
  **best draft width at current costs**; with `draft_bytes_per_step` also the best width at floor
  costs and the **speed-of-light decode score**, the geometric mean of tokens per round ÷ (pass
  floor + width × draft floor). Measure at a generous width.
- A better drafter raises tokens per round, so Eval also projects one whose later drafts are
  accepted as often as its first (each case's depth-1 rate, up to width 8) at floor costs. The
  **decode ceiling** is the larger projection.

When `generate()` reports `verify_ms` and `draft_ms`, draft steps are compared with the drafter's
bandwidth floor and the verify pass with the ordinary pass; otherwise they form one term.

**Gaps** state what each term would add at its floor, all else as measured: model pass, draft
steps, verify pass at the ordinary pass cost, best draft width and drafter acceptance (on the
decode score), prefill at its ceiling and TTFT at its floor. Each gain is a fraction of its own
score. Results carry this under `headroom`, the summary under `status`.

## Quality

[suite/quality.json](../../eval/suite/quality.json) holds six fixed conversations (prose, code,
math, German, JSON, story; 1,434 scored tokens for Qwen3.8). The final assistant turn is scored
with teacher forcing on every path the engine has: `prefill`, `decode` (one token at a time) and
`verify`. Each path reports, against the reference, the perplexity ratio, KL(reference ‖ engine)
mean / p99 / max over the reference's top-16 tokens plus a lumped tail (a lower bound of the
full-vocabulary KL), and top-1 agreement.

The **drift budget** defaults to PPL ratio ≤ 1.05, mean KLD ≤ 0.05, p99 KLD ≤ 0.5.
`--quality-budget '{"ppl_ratio_max": 1.02}'` sets any subset;
`{"ppl_ratio_max": 1.0, "kld_mean_max": 0, "kld_p99_max": 0}` asks for no drift. Each path
reports `budget_used`, its largest share of any limit. The budget is there to be spent on speed:
lower drift than needed is not a gain.

The **reference** runs the model's official `transformers` code with FP32 activations, no TF32,
and stored weights dequantized exactly (unquantized, FP8 E4M3 with channel or tensor scales,
NVFP4), streaming layers onto the GPU one at a time.

## Exactness

For a speculative engine, Eval compares each measured speed case's greedy output with the same
engine's non-speculative output and reports whether they are identical, the first differing
token and the agreeing prefix. By default (`--exactness reported`) this is information, since
verify and decode may sum in different orders and the drift budget judges both;
`--exactness required` makes any difference fail. Teacher-forced scoring cannot see a lossy
acceptance rule (accepting drafts that are not the target's argmax), so such an engine records
the rule in its configuration and needs criteria that allow it.

## Held-out prompts and generality

[suite/heldout.json](../../eval/suite/heldout.json) holds ten other prompts: Chinese, Japanese,
Korean, Russian, Arabic, Hindi, German and Spanish prose, Rust code and SQL. No engine choice may
be tuned or selected on them ([AGENTS.md](../../AGENTS.md)). After the speed cases, each decodes
its own greedy output (256 tokens, twice); Eval reports per-prompt decode tok/s and tokens per
round and the **held-out decode score**, their geometric mean (`--only heldout` runs just this).

The **generality gate** compares with a baseline (`--baseline`, normally the run's first
candidate): the held-out score may not fall more than 5%, and no prompt may decode more than 10%
slower. For the most sensitive check also compare
with the parent (`crit eval status RESULT --baseline PARENT_RESULT`). Without a baseline, or with
data prepared before the held-out set, the gate is unmeasured.

## Full-vocabulary comparison

When a change trades arithmetic for speed (lower-precision activations, approximate kernels,
another accumulation order), measure the exact shift on full logit rows over a fixed corpus:

- **Host, FP64** ([metrics.py](../../eval/crit_eval/metrics.py)): `log_softmax(logits, valid)`
  over the first `valid` outputs, `nll`, `kl_divergence` (KL(reference ‖ candidate) in nats,
  clamped at zero), a streaming `Summary` (perplexity, KL mean, nearest-rank quantiles, maximum)
  and `compare(reference_rows, candidate_rows, targets)`.
- **Device reference** ([logprob_compare.cu](../../eval/probe/logprob_compare.cu),
  [logprob_compare.h](../../eval/probe/logprob_compare.h)): `enqueue_row_compare(RowCompare, stream)`
  computes each row's log-softmax over a strided FP32 logits matrix and writes the target NLL,
  stores a reference, or writes the KL from a stored one. Fixed-order reductions make it bitwise
  repeatable; an out-of-range target yields NaN. Compile it beside the forward path or fuse the
  same arithmetic; [logprob.py](../../eval/crit_eval/logprob.py) builds it with nvcc.

Per corpus window, run the reference path once with `store`, then every output-producing path
with `reference` on the same ids, accumulating on the host. Report the reference perplexity next
to the candidate's: the ratio measures the change, the KL tail rare moved positions. For chat
models score assistant-role tokens only.

## Served benchmark outcomes

[outcomes.py](../../eval/crit_eval/outcomes.py) counts only complete, valid responses.
`validate_completion` checks an OpenAI-style SSE stream: status, in-band errors, finish policy,
`[DONE]` framing and reported usage (request `stream_options.include_usage`). `validate_cohort`
disqualifies an aggregate with a missing, failed or duplicated request or too little
concurrency; its rate is completion tokens over the fixed wall interval. `telemetry_count` keeps
missing telemetry `None`.
