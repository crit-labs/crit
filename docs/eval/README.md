# crit Eval

[Overview](README.md) · [Measurements](MEASUREMENTS.md) · [Context](CONTEXT.md) · [Adapter](ADAPTER.md)

Eval combines physics-based performance targets with common speed and quality tests.
Measured bandwidth and compute throughput set the limits each engine aims to approach.
[Core](../../core/README.md) provides the
runtime, [Inference](../../inference/README.md) supplies techniques,
[Kernels](../../kernels/README.md) helps select kernels, and Eval measures the resulting engine.
Every engine runs it the same way: `crit eval` fixes the prompts, output lengths, stop handling
and quality references, so results from different runs, agents and machines compare.

```text
crit eval probe --out device.json
crit eval context --model MODEL_DIR --device device.json
crit eval pin --adapter my_adapter.py --model MODEL_DIR --data EVAL_DATA_DIR [--context TOKENS]
crit eval run --adapter my_adapter.py --model MODEL_DIR --data EVAL_DATA_DIR --device device.json [--context TOKENS] --out result.json
crit eval run --url http://127.0.0.1:8000/v1 --model MODEL_DIR --data EVAL_DATA_DIR --context TOKENS --out result.json
crit eval reference --model MODEL_DIR --data EVAL_DATA_DIR --result result.json
crit eval status result.json [--baseline earlier.json]
crit eval compare a.json b.json
crit eval ab --adapter my_adapter.py --config A.json --config-b B.json --model MODEL_DIR --data EVAL_DATA_DIR --out ab.json
crit eval kernels --adapter my_adapter.py --model MODEL_DIR --data EVAL_DATA_DIR --device device.json --out kernels.json
```

Results report facts; Eval does not classify or rank. The builder reasons from them:

- **Gates**, the only pass/fail checks: every speed case ran and repeated its output; the same
  greedy requests give the same tokens on the fresh engine and after everything else it served
  (request history); the engine's own replies, scored on the model's checkpoint, stay within the
  drift budget in every quality set, including replies after 1K-, 4K- and 32K-token prompts; the
  engine serves a request at the required context, answers facts placed at the start, middle
  and end of long prompts up to it verbatim (retrieval) and is not slower at a lower context;
  with `--exactness required`, speculative output equals plain greedy; against a baseline, the
  held-out prompts keep up (generality). A failing gate blocks qualification whatever the speed.
- **Goals and KPIs:** decode, prefill, time to first token and the long-context scores, each
  against its physical [limit](MEASUREMENTS.md#limits); quality as the share of the drift budget
  the worst set uses. All goals hold at once for the next candidate.
- **Gaps:** what each term (model pass, draft steps, draft width, drafter acceptance, prefill,
  TTFT) would give at its floor, plus missing inputs that hide a limit.

## Setup

Eval ships in the source checkout, not the CLI wheel: `python -m pip install -e .`, or run
`python -B eval/crit_eval ...` on another host (for example WSL) with the complete `eval/`
directory, the project license and the adapter. `run` prepares missing data; the steps can also run
separately:

| Command | Does | Needs |
| --- | --- | --- |
| `probe` | Device ceilings: free memory, copy bandwidth, GEMM TFLOPS per weight format | CUDA toolkit (`nvcc`, cuBLASLt) on the target machine |
| `context` | The max context the engine must serve | the probe result, `config.json`, safetensors headers |
| `prepare --model M --data D` | Freeze the suite as token ids, with arithmetic per token | `tokenizers`, `jinja2` |
| `reference --model M --data D --result R` | Score a run's quality replies on the checkpoint `D` was prepared from and judge them | `torch`, `transformers`, `safetensors`, a GPU |
| `pin` | Pin the decode text (once per data directory) | the engine's dependencies |
| `run` | Evaluate an engine | Python standard library and the engine's dependencies |
| `kernels` | Rank kernels by GPU time per phase | Linux, a CUDA toolkit with CUPTI |

Use one data directory per model so every engine gets the same prompts. Data prepared for an
earlier major protocol version does not run: prepare it again into a new directory and pin it
again. Keep data and results outside product folders, for example in the run's workspace. The
reference needs a compatible CUDA GPU to itself; the engine under test can be CPU or GPU.
`reference`, `pin`, `run --adapter`, `ab` and `kernels` take the machine's engine lock, shared with
`crit run measure` and `crit serve`: one model-loading process at a time, the others wait and say who
holds it ([SERVING.md](../SERVING.md#start-the-server)). `run --url` takes none: the server loaded its engine.

## Engines behind a server

`--url` measures an engine behind any OpenAI-compatible server through its streaming
`/chat/completions`, with no adapter: `crit serve` and other servers alike. `--context` is the
context the server was started with, `--model` the model directory the data was prepared from
(its tokenizer encodes the replies), and `--served-model` names the model when the server lists
several. A key for the server goes in `CRIT_SERVER_KEY`.

Every prompt goes as the chat messages and tools `prepare` froze next to its token ids, and the
server applies its own chat template: the speed cases, held-out prompts, request history, the
long-context curve and retrieval. Prompt and output lengths are the server's `usage` counts,
which it must stream (`stream_options.include_usage`), and every sample records them as `served`
with the finish reason. No standard field turns end tokens off, so a reply that ends early,
typically at a tool call, is measured on the tokens it has. The server must prefill every
request: a reply whose usage reports cached prompt tokens fails, so turn its prompt cache off.
Prefill and first-token times include the server's HTTP and templating. A running server cannot
be opened at a lower context, so the lower-context gate stays unmeasured, except under
`crit run ... eval`: it serves a run's candidate itself and starts it again at the lower context
([Building engines](../BUILDING_ENGINES.md#run-the-common-eval-suite)).

Quality prompts go the same way, with `logprobs` and 16 `top_logprobs`
([Measurements](MEASUREMENTS.md#quality)). The server must render each to the token count the
model's chat template does (its own template, thinking off), report every reply token's bytes
with its log-probability and alternatives (`crit serve` does for packages declaring
`chat-logprobs-v1`, [Serving](../SERVING.md#token-log-probabilities)), and the result records the
replies. Stop the server, then `crit eval reference --result result.json` scores them on the
checkpoint the data was prepared from and writes the quality sets and the status into the result.
An adapter's engine gets no quality figures: quality is measured over the chat API.

## Protocol versions

Every prepared data directory and result records the protocol it ran under, `protocol` in
[protocol.json](../../eval/protocol.json): `crit-eval-v4` is version 4.0 and `crit-eval-v4.1`
is 4.1. Scores compare within one major version. A change that can move a score starts
the next major version: the speed, held-out, quality or long-context cases and their prompts,
output lengths, warmups and repetitions, aggregation, the quality thresholds or scoring rules,
and any harness change that measures differently. Any other change to the protocol (wording,
a reported-only field, a diagnostic) takes the next minor version. `compare` warns when results
come from different major versions. Data prepared under one major version runs under each of its
minor versions as long as it froze the suite Eval ships; otherwise `run` and `ab` ask to prepare
it again. A test pins every version to its protocol and suite, so an edit to either fails until
it has a new version.

## Results and status

`--only speed`, `--only quality` and repeated `--only` options help during development;
exactness needs speed output (`--only speed --only exactness`); `--only history` repeats its
two requests around whatever else runs. `--warmups` and
`--repetitions` override the protocol defaults; report them when comparing. Read coverage
with each score:

- `speed.complete` records whether every case ran; a partial decode score is not the full one.
- `context` records the required context and its source (`given` or `computed`), `max_context`,
  the curve and the lower-context comparison.
- Quality covers the sets whose prompts and replies fit the engine's context (`quality.skipped`
  names the others); it is unmeasured until `crit eval reference` has scored the replies.
- Exactness covers measured speed cases and is a gate only with `--exactness required`. A drop
  in plain-greedy equality against `--baseline` (status notes it) and any difference between
  repeated identical requests are bugs until a root cause is shown ([AGENTS.md](../../AGENTS.md)).
  `requirements` records the drift budget and exactness mode used.
- Physical RAM/VRAM use, cancellation and lifecycle need separate engine measurements.

A round stops early when a speed case's measured decode speeds spread by more than the protocol's
`disturbed_spread` (25%; quiet engines stay near 10% at most): something else was using the device.
The result keeps what was measured, with `aborted` and no status, and `run` exits with code 3.
Measure again when the device is quiet; `--keep-disturbed` finishes the round anyway. Requests
shorter than 100 ms are not judged, since timer jitter dominates them. Each speed case records
`host_other_cpu_cores`, the CPU cores other processes kept busy while it ran (host busy time minus
Eval's own process); the disturbed message and a failed `lower_context` gate name it, since engines
that compute on the host slow down with other programs.

`crit eval status result.json` (also printed by `run`, stored as `status`) reports the gates,
goals, each KPI against its limit, the gaps and missing inputs, with no verdict or ranking.
`--baseline earlier.json` adds each KPI's change; `--json` prints the structure.
`status.stop_record_headroom` holds each goal's start, reached score and limit in the form
[`crit run ... stop`](../RUN_RECORDS.md#stop) takes. The **end-to-end score** (output tokens over
whole requests) mixes KPIs by the suite's case mix: a fact, not a weighting.

Absolute scores drift a few percent over a day. Compare candidates with `crit eval ab`, which
runs both engines (two adapters, or one with two configurations) in one session in alternating
order (A B, then B A) and reports the paired ratio per round, its median and spread. Compare
only results with the same data, device file and Eval revision. `compare` warns when
model or suite identities differ but does not enforce matching hardware, and the
model identity does not hash every weight byte, so keep the exact identity in the run records.

A successful exit does not mean quality or exactness passed: inspect the fields. Follow
[Building engines](../BUILDING_ENGINES.md#run-the-common-eval-suite) and
[Eval evidence](../RUN_RECORDS.md#eval-evidence) to retain results.

## Kernel profile

`crit eval kernels` shows which kernels spend the GPU time. It runs each speed case once after
a warmup in a child process with a CUPTI injection library, recording every kernel (CUDA-graph
nodes included) without engine changes. It needs Linux (WSL works) and a CUDA toolkit with
CUPTI; `--case` limits the cases. Per phase (prefill, decode) it reports wall, kernel and GPU
idle time, then kernels ranked by total time with calls, average duration and launch width,
flagging launches with fewer blocks than SMs, plus each case's heaviest kernels. When `info()` declares `kernel_work`, matching kernels also show achieved
TFLOPS and GB/s against the `--device` ceilings:

```python
"kernel_work": [
    # FLOPs per prompt token summed over all calls of the kernel (prefill), and the matching format.
    {"match": "gemm_fp4", "flops_per_token": 2 * 56 * 3 * 5120 * 17408, "format": "nvfp4"},
    # Bytes one call reads and writes (weights, KV, state): memory-bound decode kernels.
    {"match": "gemv", "bytes_per_call": 89_128_960},
]
```

`match` is a substring of the demangled name; the first matching rule applies. Times are
exclusive: a kernel that starts before the previous one ends (programmatic dependent launch,
another stream) is charged only from that end, so a kernel waiting on its predecessor is not
charged the wait and the kernel total never exceeds wall time; `span_ms` keeps raw durations.
After each candidate, compare the heaviest kernels
with their floors and search [Kernels](../../kernels/README.md) before writing a replacement.
