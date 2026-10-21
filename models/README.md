# crit Models

Models produces transformed weights for an engine and records how they were made.
[Inference](../inference/README.md) executes encoded formats; Models chooses and produces
the weights. The first transform is post-training quantization (PTQ). Pruning and
quantization-aware distillation will reuse the same workflow and records.

A run allows transformed weights only with `weights_policy: transformable`
([Run records](../docs/RUN_RECORDS.md#request)); its `quality_budget` is the drift
the transform and the engine may spend together.

## Workflow

```text
crit models plan --recipe recipe.json --source SOURCE_DIR
crit models ptq  --recipe recipe.json --source SOURCE_DIR --out OUT_DIR --record transform.json [--calibration data.jsonl]
crit eval prepare   --model SOURCE_DIR --data DATA_DIR
crit eval run --url URL --model OUT_DIR --data DATA_DIR --context TOKENS --out result.json
crit models eval --record transform.json --data DATA_DIR --result result.json --evaluation evaluation.json
crit models verify --record transform.json
```

1. **Plan** resolves the recipe against the source's safetensors headers: which
   projections it quantizes and the bytes before and after. It reads no tensor data
   and needs only the standard library.
2. **PTQ** quantizes and exports a complete checkpoint into a new directory: every
   source shard under its own name, targets in their export tensors, every other
   tensor and file byte for byte, `config.json` with a compressed-tensors
   `quantization_config`. It writes the `models_transform` record. It needs `torch`
   and `safetensors`, and `transformers` for GPTQ; with `--device cuda` the GPU must
   be free of other measurements.
3. **Eval** measures the output in your engine: serve `OUT_DIR` with it on any
   OpenAI-compatible server and run `crit eval run --url` against data prepared from the
   **source** checkpoint ([Engines behind a server](../docs/eval/README.md#engines-behind-a-server)).
   Stop the server, then `crit models eval` verifies the output, scores the engine's quality
   replies on the source checkpoint (`crit eval reference`) and writes the `models_evaluation`
   record. It refuses data prepared from another model and a result measured on other data or
   through an adapter, so quality is always the engine's own replies against the original.
   Pass `--baseline` with the same engine's result on the source to see the speed change.
4. **Verify** re-hashes the output and source against the record.

The source must be unquantized (BF16, F16 or F32 projections, no `quantization_config`).
Keep sources, outputs, calibration data and records in the run's workspace, outside
product folders.

## Recipe

```json
{
  "schema_version": 1,
  "record_kind": "models_recipe",
  "method": "ptq",
  "format": "nvfp4",
  "algorithm": "gptq",
  "include": ["^layers\\.\\d+\\..+_proj$"],
  "exclude": [],
  "seed": 0,
  "calibration": {"samples": 128, "sequence_tokens": 2048},
  "gptq": {"dampening": 0.01, "block": 128}
}
```

| Field | Meaning / default |
| --- | --- |
| `format` | `fp8-channel` or `nvfp4` (below) |
| `algorithm` | `rtn` (round to nearest, no calibration) or `gptq` |
| `include`, `exclude` | Regular expressions searched in module names relative to the decoder-layer prefix (`layers.3.mlp.down_proj`); default include all `*_proj` in decoder layers, nothing excluded. Embeddings, norms, `lm_head` and tensors outside the decoder layers (for example MTP heads) stay as stored unless matched |
| `seed` | Orders the calibration documents; recorded with the output. Default 0 |
| `calibration` | GPTQ only: `samples` windows of `sequence_tokens` tokens. Default 128 x 2048 |
| `gptq` | GPTQ only: `dampening` (fraction of the mean Hessian diagonal, default 0.01) and `block` (columns per update, a multiple of 16, default 128) |

The record freezes the recipe with every default filled in.

### Formats

Both are compressed-tensors layouts that Eval's reference and common engines read. They
quantize weights only; activation precision stays an engine choice, judged by Eval's quality gates.

| Format | Tensors per projection | Arithmetic |
| --- | --- | --- |
| `fp8-channel` | `weight` E4M3FN `[rows, cols]`, `weight_scale` FP32 `[rows, 1]` | scale = row amax / 448 (1 for a zero row); codes round to nearest even, saturating |
| `nvfp4` | `weight_packed` U8 `[rows, cols/2]`, `weight_scale` E4M3FN `[rows, cols/16]`, `weight_global_scale` FP32 `[1]` | the Inference [NVFP4 codec](../inference/quantized-projections/README.md) in FP32: value = e2m1(code) x e4m3(scale) / global |

### Algorithms

- **RTN** quantizes each projection on its own. It is deterministic and needs no model code.
- **GPTQ** (Frantar et al., 2022) streams the decoder layers through the model's official
  `transformers` code with FP32 activations, as Eval's reference does, so a model larger
  than device memory still runs. Per layer it collects each target's input Hessian over
  the calibration windows, quantizes column by column while spreading the rounding error
  over the remaining columns, and feeds the next layer the quantized layer's outputs.
  NVFP4 block scales come from the error-updated columns. GPU arithmetic can change the
  output bytes between runs; the record hashes the bytes that were produced.

### Calibration data

A JSONL file with one `{"text": "..."}` document per line. Documents are shuffled by the
seed, tokenized with the source's tokenizer, joined with its first end token and cut
into the recipe's windows. Calibration text must be unrelated to Eval: `ptq` rejects a
file containing an excerpt of any Eval suite text (quality references, held-out prompts,
speed prompts, the long document), and the general rule against
[fitting the engine to Eval](../AGENTS.md#3-build-and-compare-candidates) applies. Use a
general corpus in several languages and domains.

## Records

`models_transform` (JSON, written once):

| Field | Holds |
| --- | --- |
| `recipe`, `recipe_sha256`, `seed` | The resolved recipe and its canonical hash |
| `source` | Path, file inventory (`path`, `bytes`, `sha256`), `manifest_sha256` and Eval's model identity |
| `calibration` | `null` for RTN; else file path, bytes and sha256, documents, windows, separator ids, `windows_sha256` of the token ids, and whether the suite-overlap check ran |
| `plan` | Decoder prefix, module count, `modules_sha256`, stored and transformed bytes |
| `output` | Path, file inventory, `manifest_sha256`, Eval's model identity, files copied byte for byte, skipped subdirectories |
| `errors` | Mean and worst `weight_error` (relative Frobenius) and, for GPTQ, `output_error` (Hessian-weighted) |
| `cost` | Wall seconds, seconds per phase, device and name, peak device and host bytes |
| `environment`, `crit_version`, `started_at`, `ended_at` | Python, platform, torch, CUDA, safetensors, transformers |

`models_evaluation` binds an Eval result to a transform: the transform record's sha256 and
output manifest, the server's URL and served model name, the prepared data's hash, the
result's sha256, and a summary of gates, quality and speed KPIs.

## Use in a run

In a `transformable` run, propose the candidate with `weights: "transformed"`, keep
`transform.json`, `evaluation.json` and the recipe in its source, and put the transform's
`output.manifest_sha256` in its configuration. The candidate's Eval evidence is its
result on the output against the source's data; a `crit run ... check` can reference
both records. Changing the recipe, calibration data or seed is a new transform and a new candidate.
