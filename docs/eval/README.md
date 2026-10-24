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
crit eval reference --model MODEL_DIR --data EVAL_DATA_DIR
crit eval run --adapter my_adapter.py --model MODEL_DIR --data EVAL_DATA_DIR --device device.json [--context TOKENS] --out result.json
crit eval run --url http://127.0.0.1:8000/v1 --model MODEL_DIR --data EVAL_DATA_DIR --context TOKENS --out result.json
crit eval status result.json [--baseline earlier.json]
crit eval compare a.json b.json
crit eval ab --adapter my_adapter.py --config A.json --config-b B.json --model MODEL_DIR --data EVAL_DATA_DIR --out ab.json
crit eval kernels --adapter my_adapter.py --model MODEL_DIR --data EVAL_DATA_DIR --device device.json --out kernels.json
```

Results report facts; Eval does not classify or rank. The builder reasons from them:

- **Gates**, the only pass/fail checks: every speed case ran and repeated its output; the same
  greedy requests give the same tokens on the fresh engine and after everything else it served
  (request history); the engine's scores of the served tokens of fixed quality texts (after its own
  prefill, through its decode) stay within the drift budget of the model's checkpoint's in every
  quality set, including texts of 1K-, 4K- and 32K-token prompts, and its served replies score
  alike (honesty); follow-up turns run again after everything else answer as they did first,
  whatever the engine served from a prompt cache (prompt_cache); the
  engine serves a request at the required context, answers facts placed at the start, middle
  and end of long prompts up to it verbatim (retrieval) and is not slower at a lower context;
  with `--exactness required`, speculative output equals plain greedy; sampled tokens follow the
  model's sampling distribution ([sampling](MEASUREMENTS.md#sampling-check)); against a baseline, the
  held-out prompts keep up (generality). A failing gate blocks qualification whatever the speed.
- **Goals and KPIs:** decode, prefill, time to first token and the long-context scores, each
  against its physical [limit](MEASUREMENTS.md#limits); quality as the share of the drift budget
  the worst set uses. Eval states each goal with the fraction of its limit reached and its run-to-run
  spread, and leaves their order to the run (`crit run`). From crit-eval-v7 the decode KPI,
  its goal and the stop record's decode are the [sampled decode](MEASUREMENTS.md#sampled-speed), at the
  model's own sampling, with the greedy decode reported next to it.
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
| `context` | The max context the engine must serve | the probe result, `config.json`, safetensors or GGUF headers |
| `prepare --model M --data D` | Freeze the suite as token ids, with arithmetic per token | `tokenizers`, `jinja2` |
| `reference --model M --data D` | Score the quality texts once on the checkpoint `D` was prepared from and store the scores in `D` | `torch`, `transformers`, `safetensors` (`gguf` for a GGUF), a GPU |
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

## GGUF models

`--model` takes a GGUF as it comes: one `.gguf` file, the first part of a `gguf-split` set, or a
folder holding one of them. Nothing else is needed. GGUF's converter writes the model's
description into the GGUF's metadata, and Core derives the files the rest of crit reads from it
([crit_core.model](../../core/python/crit_core/model/README.md#gguf)): `config.json` from the
architecture's shape and RoPE keys, `tokenizer.json` and `tokenizer_config.json` from the
vocabulary, merges and special tokens, the chat templates from `tokenizer.chat_template`, and
`generation_config.json` from the end-of-generation tokens and any `general.sampling.*` defaults.
crit runs no GGUF runtime for any of it; `reference` needs the `gguf` Python package to decode
the tensors.

```text
crit eval prepare --model ~/models/Qwen3.5-27B-IQ3_S.gguf --data DATA
```

The sampled speed suite runs at the model's own sampling defaults. A user who wants another
sampling measured passes it at prepare, `--sampling temperature=0.6,top_p=0.95,top_k=20`, and every
result from that data records and labels it as an override
([Sampled speed](MEASUREMENTS.md#sampled-speed)).

From a Hugging Face GGUF repository, download the one quantization you want and point at that
file: a folder with several quantizations holds several models and is refused.

Configs are derived for the `llama`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35` and `qwen35moe`
architectures (another architecture gets the generic shape keys when its metadata needs nothing
more, and is refused by name otherwise), tokenizers for byte-level BPE vocabularies with the
`qwen2`, `qwen35`, `llama-bpe` and `gpt-2` pre-tokenizers. For anything else, put the base model's
own files beside the GGUF in its folder. A folder's own files are used by group (the config; the
tokenizer with its templates; the generation config) and only the groups it lacks are derived.
The files it has must agree with the GGUF: layers, width, heads, head size, MLP and expert widths,
vocabulary, norm epsilon, RoPE base, every token string and BPE merge, so files from another model
are an error rather than a silent drift.

The context, the ceiling and the model identity count the GGUF's stored bytes per block type
(the identity also hashes the derived files). `reference` decodes every block type gguf-py
decodes (bit for bit as ggml's own dequantization) and undoes the converter's layout per
architecture before the official modeling code runs
([gguf_layout.py](../../eval/crit_eval/gguf_layout.py)), for the same six architectures; others
are refused by name. The GGUF is the reference: an engine that serves it is judged against the
quantized model as stored. A multimodal projector (`mmproj-*.gguf`) beside it is ignored.

## Other weight formats and quantization schemes

`--model` also takes PyTorch state dicts (`pytorch_model.bin`, sharded with its
`pytorch_model.bin.index.json`, or a `.pt`/`.pth` file) and ONNX graphs exported from a Hugging
Face checkpoint (Optimum, `torch.onnx`), with the model's description files beside them as
usual. Core reads a state dict's pickle without running it, so a file that would run code under
`torch.load` is refused
([crit_core.model](../../core/python/crit_core/model/README.md#pytorch-state-dicts)), and names an
ONNX graph's weights as the checkpoint it came from
([ONNX](../../core/python/crit_core/model/README.md#onnx)). Where a repository ships safetensors
beside either, the safetensors are the weights.

Quantized checkpoints are read as stored, whatever writes them
([schemes.py](../../eval/crit_eval/schemes.py)): FP8 (per tensor, per row or per block, compressed-
tensors or DeepSeek/Qwen `weight_scale_inv`), NVFP4 (compressed-tensors, ModelOpt), compressed-tensors
int (pack-quantized W4A16/W8A16, symmetric or with zero points, and int-quantized W8A8), AWQ GEMM,
GPTQ (2, 3, 4 and 8 bits, `gptq` and `gptq_v2`), MLX (affine of any width, mixed per-module widths,
`mxfp4`, `nvfp4`, `mxfp8`), bitsandbytes (NF4 and FP4, double-quantized or not, and LLM.int8) and
ONNX Runtime's MatMulNBits. `reference` decodes each to exactly the FP32 values the format's own
library produces (the tests check this against GPTQModel, AutoAWQ, mlx, bitsandbytes,
compressed-tensors and onnxruntime on models those tools wrote, and that the reference scores such
a model exactly as transformers scores its dequantized twin). The ceiling counts each matrix once
at its stored format's rate: weight-only integer codes at bf16 (engines widen them for the GEMM),
8-bit integer weights with per-row scales at fp8 (int8 tensor cores run at the FP8 rate), MXFP4 at
nvfp4. Refused by name: AWQ GEMV layouts, GPTQModel's planar 5/6/7-bit layout, ONNX graphs from
onnxruntime-genai's builder, ONNX quantized ops other than MatMulNBits, and EXL2 and EXL3,
which exllama decodes only in CUDA kernels, so there is no CPU dequantization to check against.

`reference` runs the text model's own transformers forward in FP32 and streams its decoder
layers onto the GPU one at a time ([reference.py](../../eval/crit_eval/reference.py)). One pass
per prompt records what the forward hands each layer, without running it; each layer is then
loaded once and run for every prompt, and the forward runs again on the last layer's output so
its own tail (final norm or mixer) gives the head its input. Whatever a model does around its
layers, such as embedding scales, repeated residual streams, multi-axis positions, per-layer
inputs and its masks, is therefore its own code. Checkpoint names reach the modeling code's
parameters through transformers' conversion mapping for the model type: per-expert matrices
become fused expert tensors (each expert decoded on its own first), a table stored in shards is
concatenated, and a text model stored under a multimodal prefix is found by its layer names.
Vision towers and draft (MTP) layers are never read. Each layer's stored tensors are read ahead
while the previous layer runs, in large reads in file order into one host buffer (one layer, at
most 8 GiB), and reach the GPU in one transfer, where they are dequantized. Embedding lookups
read and decode only the rows they need, by parallel reads of exactly their bytes (plain, FP8 and
NVFP4 tables, including a sharded table with one scale), so a hashed n-gram table larger than
host memory is never built. Experts
run transformers' eager per-expert loop. The long prompts use chunked attention with boolean
masks only where the modeling code needs one (for example a sparse-attention indexer). Where
modeling code loops in Python over every token, the reference binds a form that computes the same
at once, only to the code it was checked against ([indexers.py](../../eval/crit_eval/indexers.py):
qwen4_exp's sparse-attention indexer, which pools every complete key block once instead of once per
query; floating-point ties at its top-k edge are left to the device, as in the loop).

`reference` scores every position of every prepared quality text (a conversation's prompt
followed by its written answer) once per model, suite and reference version, and stores per
position the target log-probability and the reference's top 16 under `D/reference/`: one
safetensors file per text and `index.json` with the identity (the reference's version, the
transformers and torch versions, the GPU, the model and every text's ids) and each file's sha256,
about 16 MB for the suite. Every engine's scores of the same texts are judged against them, so a
candidate costs only its own engine time. A text runs in chunks of 8,192 tokens that resume the
cache the model's own forward builds (transformers' Cache: keys and values, sliding windows,
linear-attention or Mamba states), kept per decoder layer in scratch files until the text is
scored, so no mask of the whole text's length squared is built. The chunks run layer-major: a
layer, once loaded, runs every chunk it can before the next layer loads, with each chunk's hidden
states on the host. The forward reads the state of a few layers between chunks (usually the first
layer of each kind), so the layers up to those run chunk by chunk and the rest are loaded once;
the reference checks every read of another layer's state and of the cache's own attributes against
what the chunk-major order (each chunk a round through every layer) would see, and runs the texts
that way where they differ; either order gives the same scores. A model whose cache cannot be
resumed runs each text whole. Under `crit run ... eval` the reference scores the data's texts
before the first candidate is served, stopping at the reservation's deadline, which it checks
between layers and texts.

## Engines behind a server

`--url` measures an engine behind any OpenAI-compatible server through its streaming
`/chat/completions`, with no adapter: `crit serve` and other servers alike. `--context` is the
context the server was started with, `--model` the model directory the data was prepared from
(its tokenizer encodes the replies), and `--served-model` names the model when the server lists
several. A key for the server goes in `CRIT_SERVER_KEY`. Eval sets no time budget of its own:
each request may take the protocol's floor plus a slack times what the engine's earlier requests
in the same Eval say it should (`requests.timeout`: prefill time grows at most quadratically with
the prompt, a decode step at most linearly with the context), and a request past that stops the
Eval as hung.

The greedy cases send `temperature: 0`; the [sampled speed](MEASUREMENTS.md#sampled-speed) cases send the
model's own sampling (`temperature`, `top_p`, `top_k`, `min_p`) and a `seed`, and the sampling check asks
for `logprobs` and `top_logprobs` on its streamed replies.

Every prompt goes as the chat messages and tools `prepare` froze next to its token ids, and the
server applies its own chat template: the speed cases, held-out prompts, request history, the
long-context curve and retrieval. Prompt and output lengths are the server's `usage` counts,
which it must stream (`stream_options.include_usage`), and every sample records them as `served`
with the finish reason. No standard field turns end tokens off, so a reply that ends early,
typically at a tool call, is measured on the tokens it has. The server must prefill every
timed request: a reply whose usage reports cached prompt tokens
(`usage.prompt_tokens_details.cached_tokens`) fails. Retrieval requests, judged on their text
alone, may be served from a prompt cache.
Prefill and first-token times include the server's HTTP and templating. A running server cannot
be opened at a lower context, so the lower-context gate stays unmeasured, except under
`crit run ... eval`: it serves a run's candidate itself, first at the lower context, then builds it
again at the run's context in the same resident host, which keeps the model's files, device
weights and whatever host-side weights the adapter keeps in the process
([Building engines](../BUILDING_ENGINES.md#run-the-common-eval-suite)). From `crit serve` Eval also
reads the loaded engine's `engine_facts` (whether it speculates, its draft width and the bytes a
pass and a draft step read, [Serving](../SERVING.md#engine-facts-and-plain-decoding)) and each
reply's drafts accepted per round, so a served candidate gets the same decode limits as an
adapter's, and checks exactness on the same engine through plain decoding (`chat-plain-v1`).

Quality needs the engine's own scores of the fixed texts, from the paths it serves with: `POST
/v1/crit/score` on `crit serve` for packages declaring `chat-score-v1`
([Serving](../SERVING.md#fixed-text-scores)) returns every prompt position from one prefill of the
prompt and every continuation token from its decode, forced to the continuation
([Measurements](MEASUREMENTS.md#quality)). The honesty check then takes the engine's greedy replies
to the conversations and the 32K-token prompts as quality replies, with `logprobs` and 16
`top_logprobs` (the server renders each prompt to the token count the model's chat template does,
its own template, thinking off, and names every reply token by its bytes; `crit serve` does for
packages declaring `chat-logprobs-v1`, [Serving](../SERVING.md#token-log-probabilities)), and
scores the same tokens through the score call: the two must agree. Since that needs the model's
own chat template, a round that measures quality requires every prompt sent as its frozen messages
to render to its prepared token count from the first request on: a server on another template
(thinking on, a default system prompt) fails at once, not after the speed parts. Run `crit eval
reference --model M --data D` once before, with no engine on the device. An adapter's engine, or a
server without the score call, gets no quality figures.

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
it has a new version. A baseline compares only with results of its prepared data, suite and major
version.

## Results and status

`--only speed`, `--only quality` and repeated `--only` options help during development;
exactness needs speed output (`--only speed --only exactness`); `--only history` repeats its
two requests around whatever else runs. `--warmups` and
`--repetitions` override the protocol defaults; report them when comparing. Read coverage
with each score:

- `speed.complete` records whether every case ran; a partial decode score is not the full one.
- `context` records the required context and its source (`given` or `computed`), `max_context`,
  the curve and the lower-context comparison.
- Quality covers the sets whose texts fit the engine's context (`quality.skipped` names the
  others), each judged on its prompt positions (`SET/prompt`, from the prefill) and its served
  positions (`SET/served`), with `quality.buckets` by position and `quality.honesty`; it
  is unmeasured until `crit eval
  reference` has stored the reference scores in the data directory.
- Exactness covers measured speed cases and is a gate only with `--exactness required`. A drop
  in plain-greedy equality against `--baseline` (status notes it) and any difference between
  repeated identical requests are bugs until a root cause is shown ([AGENTS.md](../../AGENTS.md)).
  `requirements` records the drift budget and exactness mode used.
- `host` records the machine: OS (and WSL), CPU model and cores, physical RAM (the VM's under WSL)
  and each GPU's VRAM, driver and PCIe link. `pcie.gen_current` is sampled idle, when GPUs lower
  their link; `gen_max` is what the GPU and slot allow together, `gen_gpu_max` and `gen_host_max`
  each side's own. Unreadable values are null. Compare engines that move weights over PCIe or
  compute on the CPU only on matching hosts. Paths in results (`adapter`, with `adapter_sha256`,
  `native_libraries`, the probe's and kernel profile's nvcc) name no user: relative to the
  checkout, `~/...` in the home directory, the file name in another user directory, else absolute.
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
`--baseline earlier.json` adds each KPI's change; `--json` prints the structure. A result of
another protocol major version is refused: its status is the one stored in it when it was measured.
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
