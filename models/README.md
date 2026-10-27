# crit Models

Models produces transformed weights for an engine and records how they were made.
[Inference](../inference/README.md) executes encoded formats; Models chooses and produces
the weights. The transforms are post-training quantization (PTQ) and
[drafter production](#drafters), a draft head distilled from the target for speculative
decoding. Pruning and quantization-aware distillation will reuse the same workflow and records.

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
   **source** checkpoint ([Engines behind a server](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md#engines-behind-a-server)),
   whose reference scores `crit eval reference --model SOURCE --data DATA` stored first. Stop
   the server, then `crit models eval` verifies the output and writes the `models_evaluation`
   record. It refuses data prepared from another model and a result measured on other data or
   through an adapter, so quality is always the engine's own scores against the original's.
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
| `format` | `fp8-channel`, `nvfp4` or `nvfp4-4over6` (below) |
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
| `nvfp4-4over6` | as `nvfp4` | `nvfp4` whose block scales also try mapping the block maximum to 4 instead of 6 and keep the smaller squared error (the codec's `four_over_six`); GPTQ chooses from the error-updated columns |

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

## Drafters

Most checkpoints ship no draft layer, and the run rules forbid the agent to train one. A drafter is
the user's transform instead: `crit models drafter` distills a draft head from the target itself,
records how (recipe, calibration data, seed, hashes, cost, the agreement it reached), and the result
rides along with the target like a published companion draft model. The target's weights never
change, so a candidate that verifies every draft with the stored target keeps the target's output
and declares `weights: as_stored` as before; it records the drafter's transform record with its
source and names the drafter's `output.manifest_sha256` in its configuration.

```text
crit models plan    --recipe drafter.json --source MODEL_DIR
crit models drafter --recipe drafter.json --source MODEL_DIR --out DRAFTER_DIR --record drafter-transform.json --calibration data.jsonl
crit models verify  --record drafter-transform.json
```

The recipe (`method: drafter`), every field optional:

| Field | Meaning / default |
| --- | --- |
| `kind` | `mtp-head` (the only kind yet) |
| `init_layer` | The target layer the drafter's layer starts from; `-1` (default) is the last |
| `calibration` | `samples` windows of `sequence_tokens` tokens, default 64 x 1024, from the JSONL as for GPTQ ([calibration data](#calibration-data)) |
| `heldout_fraction` | The share of windows held out to measure the drafter, default 0.1 |
| `train` | `steps` (1000), `windows_per_step` (4), `learning_rate` (1e-4), `warmup_steps` (50), `weight_decay` (0) |
| `dtype` | `bf16` (default) or `fp32` for the stored tensors |
| `seed` | Orders the windows and the training draws, default 0 |

The `mtp-head` drafter (`crit-drafter-v1`): at position t it takes the target's final hidden state
before the final norm and the embedding of token t+1, normalizes both with copies of the target's final
norm (`pre_fc_norm_hidden`, `pre_fc_norm_embedding`), fuses them with `fc` (2 x hidden to hidden, no
bias), runs one decoder layer of the target's own architecture (`layers.0`, initialized from
`init_layer`) and a copy of the final norm (`norm`), and predicts through the target's own output head,
which is not trained and not stored. It is trained to match the target's next-token distribution one
position ahead (the KL divergence from the target's distribution) over the calibration windows, with
the held-out windows reporting the greedy agreement (`top1_agreement`: how often the draft's argmax
equals the target's, the acceptance a greedy engine would see) and the KL left, before and after
training. The output folder holds `config.json`, the target's config with `num_hidden_layers: 1` and
a `crit_drafter` block that says what the tensors are and which target they belong to
(`eval_model_sha256`), and `model.safetensors` with the tensors under `drafter.`. The target pass is
the PTQ calibration pass (embeddings through the layers with the model's rotary embeddings and a
causal mask), so models whose forward does more around their layers are outside it, as for GPTQ.
Calibration text must be unrelated to Eval ([calibration data](#calibration-data)); the agreement is
measured on held-out windows of the same corpus, so a corpus in several languages and domains is the
honest one.

## Records

`models_transform` (JSON, written once):

| Field | Holds |
| --- | --- |
| `recipe`, `recipe_sha256`, `seed` | The resolved recipe and its canonical hash |
| `source` | Path, file inventory (`path`, `bytes`, `sha256`), `manifest_sha256` and Eval's model identity |
| `calibration` | `null` for RTN; else file path, bytes and sha256, documents, windows, separator ids, `windows_sha256` of the token ids, and whether the suite-overlap check ran |
| `plan` | PTQ: decoder prefix, module count, `modules_sha256`, stored and transformed bytes. Drafter: prefix, `layers`, `init_layer`, `parameters`, the stored tensors |
| `output` | Path, file inventory, `manifest_sha256`, Eval's model identity, files copied byte for byte, skipped subdirectories |
| `errors` | PTQ: mean and worst `weight_error` (relative Frobenius) and, for GPTQ, `output_error` (Hessian-weighted). Drafter: `train_loss` (start, end, steps) and the held-out `kl_mean` and `top1_agreement` before (`heldout_before`) and after (`heldout`) training |
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
