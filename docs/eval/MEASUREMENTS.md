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
Defaults: 1 warmup, 2 repetitions and a third only when the two disagree: their decode speed or time to
first token spread by more than 5%, or their outputs differ ([protocol.json](../../eval/protocol.json)). Time to first token
gives prefill, the remaining tokens decode. The **decode score** is the geometric mean of the six
case medians; each case also reports tokens per round. The **prefill score** is the geometric mean of prompt tokens ÷ TTFT over `long-summary`
and `agent-long` (`prefill_cases`). The **TTFT score** is the geometric mean of TTFT in ms over
every case, where short prompts expose fixed costs.

## Sampled speed

Real requests sample, and an engine that speculates only on greedy requests decodes them one token
per pass. From crit-eval-v7 the six speed cases run a second time at the model's own sampling, the
`sampling` that `crit eval prepare` freezes from its `generation_config.json`: temperature, top_p,
top_k and min_p when it sets `do_sample` (temperature 1.0 when it names none), temperature 1.0 with
no filters when it does not sample (`sampling_source` `generation_config` or `protocol_default`).
Only the user may measure at other settings: `crit eval prepare --sampling
temperature=0.6,top_p=0.95,top_k=20` (temperature required; top_p, top_k and min_p optional)
freezes them instead, as `sampling_source` `override`; results record the source next to the
settings, `status` and `report` label an override, and decodes at different settings or sources
are not compared. Engine builders never choose the sampling: a speed won at a lower temperature is
not the speed the model's requests get. Repetition, presence and frequency penalties stay neutral,
so every engine samples the same distribution. Each repetition has its own fixed seed (warmup 0,
repetitions 1, 2, 3), all three measured and no extra one: different seeds decode different texts,
whose tokens per round differ, so their speeds disagree by content, not noise. For the same reason a
disturbed device is judged on the time per target pass (`round_ms`) of the sampled repetitions, not
on their decode speed. Each decode score carries its relative standard error (`decode_se`: per
case, the spread of the repetitions' logarithms over the square root of their count, combined over
the cases), which for the sampled suite includes how much the drawn texts move it; `status` names a
baseline change within two standard errors of the difference as within the noise. The **sampled decode score**, the geometric mean of the case medians, is the decode KPI
from v7 on (`status.kpis.decode`, with `suite` "sampled"): its headroom and ceiling, the decode goal
and the stop record's decode follow it, and baselines compare it at the same sampling. The greedy
suite's decode score stays next to it (`status.kpis.greedy_decode`), and prefill, time to first
token, held-out prompts and exactness stay on the greedy suite. A result with no measured sampled
decode (v6, or an engine that cannot sample) keeps the decode KPI on the greedy suite.

Every sampled request names every sampling field, so no engine default changes the distribution
(an engine applies its own defaults, such as a model card's presence penalty, to the fields a
request leaves out): the prepared settings, the filters they leave unset off (`top_p` 1, `top_k` 0,
`min_p` 0), `repetition_penalty` 1, `presence_penalty` and `frequency_penalty` 0, and the seed. An
adapter's engine samples through `generate_sampled` ([Adapter](ADAPTER.md)); an engine behind a
server gets the fields in the `/chat/completions` body. An engine that lists the fields it applies
(crit serve's `supported_sampling_parameters`, an adapter's `sampling_fields`) gets only those, and
one that cannot apply a field the settings set, or cannot sample, has no sampled decode.

## Sampling check

A speculative engine must keep the distribution it samples from. Under greedy decoding the exactness
check compares tokens; under sampling there is no single right token, so Eval checks the
distribution. After the sampled suite, one more sampled reply per speed case (seed 7) returns each
emitted token's log-probability and its top 20 alternatives. For every position Eval rebuilds the
processed distribution from those raw log-probabilities (temperature, then top_k, top_p and min_p),
with the unreported rest of the vocabulary as one last alternative: it holds the mass the reported
tokens leave, and each of its tokens is at most as likely as the last reported one. A position is
judged when the reported alternatives determine that distribution: top_k keeps only reported tokens
(Qwen's top_k 20), or the top_p or min_p edge falls among them whatever the rest holds, or the
rest's processed mass is known within 0.1% (exactly at temperature 1 without filters). The others
are reported as `unresolved`, typically flat positions under top_p with top_k above 20 (Gemma's
64) or none. Every judged token gets a randomized probability integral transform:
u = (mass of the more likely alternatives, the unreported rest last) + V × p(token), with V
uniform from a seeded generator. When every token was drawn from its processed distribution, u is
uniform on [0, 1], so over n tokens z = (mean u − ½) / √(1 / 12n) stays near 0. A sampler that
favours likely tokens, for example one that accepts drafts by greedy match, pulls z far below 0 (a
sampler taking the likeliest token half the time gives about −19 over 1,500 tokens). The
`sampling` gate fails when |z| exceeds 4 or a token falls outside its processed distribution at a
judged position, and is unmeasured with fewer than 256 judged tokens; the Kolmogorov-Smirnov
distance is reported. The log-probabilities must be the model's own, before temperature and
filters, the values the quality honesty check verifies for greedy replies; a server names each
token and alternative by its bytes, as for the honesty check.

Correct speculative sampling keeps the distribution exactly: accept a draft d with probability
min(1, p(d) / q(d)), where p is the target's processed distribution and q the drafter's, and on the
first rejection sample from the normalized residual max(p − q, 0); a greedy draft is q one-hot.

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

**Prefill ceiling.** `prepare` counts arithmetic per token from safetensors or GGUF headers and config
([crit_eval/ceiling.py](../../eval/crit_eval/ceiling.py)): 2 × rows × columns for every
projection in its stored format (BF16, FP8 or NVFP4; a GGUF's block types at the FP8 rate, the
rate int8 tensor cores share, and its MXFP4/NVFP4 at the NVFP4 rate), excluding gathered embeddings, the vision
tower, draft layers and the output head; 4 × heads × head dim per query-key pair in each
full-attention layer; and the chunked linear-attention recurrence. The floor is matmul FLOPs at
each format's GEMM rate, attention at the faster of the BF16 and FP8 rates (FP8 Q·Kᵀ and P·V fit
the quality budget, so a BF16 attention floor can be beaten), and the recurrence at the BF16 rate.

**TTFT floor:** the larger of the prompt's compute floor and one weight read (`bytes_per_pass`,
else the language-model and output-head bytes) plus the prompt's KV writes at device bandwidth.

**Decode headroom.** Every case reports `round_ms`, decode time per target pass. Optional
adapter fields add the ceiling:

- `draft_width`: **saturation**, tokens per round ÷ (draft width + 1).
- `bytes_per_pass` (weights read, not gathered embeddings, plus recurrent state) and the
  bandwidth: the **pass floor**, compared with the measured one-token step. Each case adds the
  KV read at its context: BF16 K and V of every full-attention layer, or `kv_bytes_per_token`.
  An engine that reports no `bytes_per_pass`, such as one behind another server, gets the stored
  model's: the weights read on every pass but draft layers, the routed experts' share
  (`num_experts_per_tok` of the experts) and the fixed recurrent and sliding-window state
  (`headroom.bytes_per_pass_source` says `engine` or `model files`).
- `accepted` per round from `generate()`: tokens per round at every width up to the measured one.
  With the draft cost (`draft_ms`, else round time beyond one pass ÷ width) Eval projects the
  **best draft width at current costs**; with `draft_bytes_per_step` also the best width at floor
  costs and the **speed-of-light decode score**, the geometric mean of tokens per round ÷ (pass
  floor + width × draft floor). Measure at a generous width.
- A better drafter raises tokens per round, so Eval also projects one whose later drafts are
  accepted as often as its first (each case's depth-1 rate, up to width 8) at floor costs. The
  **decode ceiling** is the larger projection.
- From v7 these describe the sampled suite. Rejection sampling lets a drafter speculate under
  sampling, so the depth-1 rate of the projection is the better of the sampled and greedy suites'
  for each case; a sampled case that drafts nothing projects from its measured tokens per round.
  An engine that speculates only on greedy requests thus keeps the ceiling its drafter allows,
  not the one-token-per-pass floor.

When `generate()` reports `verify_ms` and `draft_ms`, draft steps are compared with the drafter's
bandwidth floor and the verify pass with the ordinary pass; otherwise they form one term.

**Gaps** state what each term would add at its floor, all else as measured: model pass, draft
steps, verify pass at the ordinary pass cost, best draft width and drafter acceptance (on the
decode score), from v7 sampled requests as fast as greedy ones (each sampled case at its greedy
case's decode, where that is faster), prefill at its ceiling and TTFT at its floor. Each gain is a fraction of its own
score. Results carry this under `headroom`, the summary under `status`.

**Noise.** Every goal KPI carries its run-to-run spread (`headroom.noise`: the spread of the
repetitions' scores, max − min over the median): decode from its own suite, prefill over the prefill
cases, TTFT over every case. The long-context curve runs once per point, so its decode and first
token take the speed suite's spreads (the greedy decode's and the prefill's). `status` prints each
goal's share of its limit and its noise, the noise beside every gap, and marks a baseline change no
larger than the larger of the two results' spreads as within the noise (decode, which has a standard
error, by two standard errors of the difference). Eval states the goals; ordering them is the run's.

## Quality

Quality is what the engine loses on top of the checkpoint the user chose, nothing else, measured
on fixed text through the paths the engine serves with.
[suite/quality.json](../../eval/suite/quality.json) holds six conversations (prose, code, math,
German, JSON, story), each with a written answer. Each quality text is a conversation rendered by
the model's chat template with its generation prompt, followed by its answer cut to 256 tokens
(`quality.continuation_tokens`); `prepare` freezes both as token ids.

`crit eval reference` scores every position of every text once, teacher forced on the checkpoint,
and stores each position's log-probability and the reference's top 16 under the data. The engine
scores the same ids through crit serve's score call (`chat-score-v1`): every prompt position and the
first answer token from one prefill of the prompt, as it prefills any request, and every later
answer token from its decode, each step forced to the answer (a speculative engine accepts drafts
only while they equal it and takes every correction and bonus token from it). Eval compares each
position: the perplexity ratio (the engine's perplexity of the scored tokens over the reference's),
KL(reference ‖ engine) mean / p99 / max over the tokens in both top-16 lists plus one bucket for the
rest (a lower bound of the full-vocabulary KL), and top-1 agreement (the share of positions where
both rank the same token first). Each set is judged on both paths the engine computes it with, each
by the drift budget on its own: its prompt positions from the prefill (`SET/prompt`) and the positions
it serves (`SET/served`), the answer, its first token from the prefill's last position and the rest
from decode. No request returns the prompt positions' logits, but they are the prefill's arithmetic
that every reply's first token comes from, a single position among a set's served ones: an engine
whose prefill drifts far more than its decode (lower-precision prefill activations, for one) is
judged on it. Both are also reported by position: absolute position in log2 buckets (1-511,
512-1023, 1024-2047, ...) for the prompt and offset in the answer (1-31, 32-63, 64-127, 128-255) for
the served positions, so a threshold where the engine switches kernels shows as a step (reported, not
judged).

**Long prompts.** A real request prefills a long prompt and then decodes, and decode reads what the
prefill left: prefetched experts and their scales, KV written by a different kernel, sparse-attention
indices. Three of the conversations (`quality.long` in [protocol.json](../../eval/protocol.json))
also go with suite text ([long.json](../../eval/suite/long.json)) before their first user message, so
they render to about 1,024, 4,096 and 32,768 tokens, past typical prefetch, prefill-kernel and
sparse-attention thresholds. The conversations are one set and each length another
(`conversations`, `long-1024`, `long-4096`, `long-32768`); a set whose texts do not fit
`max_context` is skipped.

**Honesty.** The scores must come from the paths the engine serves with. Eval also takes the
engine's greedy replies to the conversations and one 32K-token prompt (`explain-tides`; each long
prompt costs two more prefills, and one reaches the long serving path) over its chat API, with
`logprobs` and 16 `top_logprobs` (the server renders each prompt to the token count the model's chat
template renders and names each reply token by its bytes), and scores the same reply tokens through
the score call, without the prompt's rows, which the fixed texts already score: every token's
log-probability and every alternative in both lists must agree within `quality.honesty.tolerance`
(1e-3 nats), and the lists hold the same tokens but those within it of the 16th. `quality.honesty`
records the largest difference; a failure fails the quality gate.

`quality.timing` shows where the phase's time goes: for the fixed texts' score requests, the honesty
replies and the honesty rescores, the requests' seconds, the engine's own prefill and decode
seconds (`crit_telemetry`), the rest of each request's time (transfer and the server's own work),
the response bytes and the client's parse seconds, with each text's own row in `timing.texts`;
`timing.metrics_seconds` is the time the comparison with the reference took. Eval asks crit serve
for the scores in its packed format (int32 tokens and float32 log-probabilities, base64), a
fraction of the JSON rows' bytes, and reads JSON rows from a server that offers no packed format.

The **drift budget** defaults to PPL ratio ≤ 1.05, mean KLD ≤ 0.05, p99 KLD ≤ 0.5.
`--quality-budget '{"ppl_ratio_max": 1.02}'` sets any subset;
`{"ppl_ratio_max": 1.0, "kld_mean_max": 0, "kld_p99_max": 0}` asks for no drift. Each set
reports `budget_used`, its largest share of any limit. The budget is there to be spent on speed:
lower drift than needed is not a gain.

The **reference** is the checkpoint the data was prepared from. It runs the model's official
`transformers` code with FP32 activations, no TF32, and stored weights dequantized exactly
(unquantized, FP8 E4M3 with channel or tensor scales, NVFP4), streaming layers onto the GPU one at
a time, so a model larger than device memory still runs; it needs the GPU to itself, so stop the
engine first. The long prompts use the same arithmetic in blocks of 256 query rows, since a mask and score matrix over 32K tokens would not fit. Mamba-2
mixers take the dt limit config.json declares (`time_step_limit`, usually [0, inf]) as the
model's own kernels do; some transformers mixers clamp dt at `time_step_min` instead,
which makes long-memory heads forget (retrieval fails beyond about 64K tokens) and reads as
drift against an engine that follows the config.

## Exactness

Exactness covers the greedy speed suite; under sampling the [sampling check](#sampling-check) judges the
distribution instead. For a speculative engine, Eval compares each measured speed case's greedy output with the same
engine's non-speculative output and reports whether they are identical, the first differing
token and the agreeing prefix. An engine declaring `plain` decodes the non-speculative output
itself before it closes; otherwise Eval opens a second engine with `speculative` false
(`exactness.engine` is `shared` or `separate`). A served engine is speculative when crit serve's
`engine_facts` say so, and decodes plainly when its package declares `chat-plain-v1`; without it
exactness is unmeasured, since a second engine from the same server would be the same speculative
one. By default (`--exactness reported`) this is
information, since verify and decode may sum in different orders and the drift budget judges the
output; `--exactness required` makes any difference fail. The drift budget compares distributions,
so it does not judge a lossy acceptance rule (accepting drafts that are not the target's argmax);
top-1 agreement shows one, and such an engine records the rule in its configuration and needs
criteria that allow it.

A difference is information, not evidence of rounding. A drop in plain-greedy equality against an
earlier candidate (status notes it when given `--baseline`) or any difference between identical
requests is a bug until a root cause is demonstrated ([AGENTS.md](../../AGENTS.md)): stale or
uninitialized state shows up exactly this way, and quality scores over a few prompts can miss it.

## Request history

Right after the engine built for the required context opens, before any other request, Eval
generates the speed cases with the shortest and the longest prompt (`agent-long`, 14,336 tokens),
and again after everything else that engine served: speed, held-out prompts, quality, the
long-context curve up to the requirement and retrieval. The tokens must be identical. A
deterministic engine's greedy output does not depend on earlier requests, so this gate holds in
every exactness mode and constrains no speed; `repeatable` cannot see state that changed before
the first measured repetition. `history.served_between` lists what ran in between.

## Follow-up turns and the prompt cache

A conversation's follow-up turns resend its history. Right after the request history's first
requests, Eval sends each turn of [followup.json](../../eval/suite/followup.json) in order: the first
its user message after a document making the prompt about `followup.prompt_tokens` (8,192)
tokens, each later one the conversation so far with the suite's fixed assistant replies and the
next user message, `output_tokens` (64) each. Each turn may be served from a prompt cache, so an
engine that keeps the earlier turns prefills only the rest. Every turn reports its time to first
token, the prompt tokens the engine reports as cached
(`usage.prompt_tokens_details.cached_tokens`) and the prefill rate of the tokens it computed;
`followup.ttft_ms` and `followup.prefill_tps` are the geometric means over the turns after the
first. After every other part the turns run again, and every turn must answer as it did first
(the **prompt_cache** gate): a cached path must give the tokens the fresh one gave. An engine
without a prompt cache reports its fresh prefill and passes. The part is skipped when its last
turn does not fit the engine's context.

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
