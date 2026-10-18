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
crit eval status result.json [--baseline earlier.json]
crit eval compare a.json b.json
crit eval ab --adapter my_adapter.py --config A.json --config-b B.json --model MODEL_DIR --data EVAL_DATA_DIR --out ab.json
crit eval kernels --adapter my_adapter.py --model MODEL_DIR --data EVAL_DATA_DIR --device device.json --out kernels.json
```

Results report facts; Eval does not classify or rank. The builder reasons from them:

- **Gates**, the only pass/fail checks: every speed case ran and repeated its output; the same
  greedy requests give the same tokens on the fresh engine and after everything else it served
  (request history); quality stays within the drift budget on every scoring path, including
  decode after 1K-, 4K- and 32K-token prompts (`prefill_decode`, required of every engine); the
  engine serves a request at the required context, answers facts placed at the start, middle
  and end of long prompts up to it verbatim (retrieval) and is not slower at a lower context;
  with `--exactness required`, speculative output equals plain greedy; against a baseline, the
  held-out prompts keep up (generality). A failing gate blocks qualification whatever the speed.
- **Goals and KPIs:** decode, prefill, time to first token and the long-context scores, each
  against its physical [limit](MEASUREMENTS.md#limits); quality as the share of the drift budget
  the worst path uses. All goals hold at once for the next candidate.
- **Gaps:** what each term (model pass, draft steps, draft width, drafter acceptance, prefill,
  TTFT) would give at its floor, plus missing inputs that hide a limit.

## Setup

Eval ships in the source checkout, not the CLI wheel: `python -m pip install -e .`, or run
`python -B eval/crit_eval ...` on another host (for example WSL) with the complete `eval/`
directory, the project license and the adapter. `run` prepares missing data and, when quality is requested, builds a missing reference;
the steps can also run separately:

| Command | Does | Needs |
| --- | --- | --- |
| `probe` | Device ceilings: free memory, copy bandwidth, GEMM TFLOPS per weight format | CUDA toolkit (`nvcc`, cuBLASLt) on the target machine |
| `context` | The max context the engine must serve | the probe result, `config.json`, safetensors headers |
| `prepare --model M --data D` | Freeze the suite as token ids, with arithmetic per token | `tokenizers`, `jinja2` |
| `reference --model M --data D` | Reference log-probabilities | `torch`, `transformers`, `safetensors`, a GPU |
| `pin` | Pin the decode text (once per data directory) | the engine's dependencies |
| `run` | Evaluate an engine | Python standard library and the engine's dependencies |
| `kernels` | Rank kernels by GPU time per phase | Linux, a CUDA toolkit with CUPTI |

Use one data directory per model so every engine gets the same token ids and reference. Data
prepared for an earlier protocol (before `crit-eval-v4`) has no long prompts: prepare it again
into a new directory, then build its reference and pin it again. Keep
data, references and results outside product folders, for example in the run's workspace.
Reference generation needs a compatible CUDA GPU; the engine under test can be CPU or GPU.

## Results and status

`--only speed`, `--only quality` and repeated `--only` options help during development;
exactness needs speed output (`--only speed --only exactness`); `--only history` repeats its
two requests around whatever else runs. `--warmups` and
`--repetitions` override the protocol defaults; report them when comparing. Read coverage
with each score:

- `speed.complete` records whether every case ran; a partial decode score is not the full one.
- `context` records the required context and its source (`given` or `computed`), `max_context`,
  the curve and the lower-context comparison.
- Quality covers the declared scoring paths; `prefill_decode` is required, so an adapter without
  it fails the quality gate.
- Exactness covers measured speed cases and is a gate only with `--exactness required`. A drop
  in plain-greedy equality against `--baseline` (status notes it) and any difference between
  repeated identical requests are bugs until a root cause is shown ([AGENTS.md](../../AGENTS.md)).
  `requirements` records the drift budget and exactness mode used.
- Physical RAM/VRAM use, cancellation and lifecycle need separate engine measurements.

A round stops early when a speed case's measured decode speeds spread by more than the protocol's
`disturbed_spread` (25%; quiet engines stay near 10% at most): something else was using the device.
The result keeps what was measured, with `aborted` and no status, and `run` exits with code 3.
Measure again when the device is quiet; `--keep-disturbed` finishes the round anyway. Requests
shorter than 100 ms are not judged, since timer jitter dominates them.

`crit eval status result.json` (also printed by `run`, stored as `status`) reports the gates,
goals, each KPI against its limit, the gaps and missing inputs, with no verdict or ranking.
`--baseline earlier.json` adds each KPI's change; `--json` prints the structure.
`status.stop_record_headroom` holds each goal's start, reached score and limit in the form
[`crit run ... stop`](../RUN_RECORDS.md#stop) takes. The **end-to-end score** (output tokens over
whole requests) mixes KPIs by the suite's case mix: a fact, not a weighting.

Absolute scores drift a few percent over a day. Compare candidates with `crit eval ab`, which
runs both engines (two adapters, or one with two configurations) in one session in alternating
order (A B, then B A) and reports the paired ratio per round, its median and spread. Compare
only results with the same data, reference, device file and Eval revision. `compare` warns when
model or suite identities differ but does not enforce matching references or hardware, and the
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
