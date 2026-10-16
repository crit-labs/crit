# crit Eval

Eval is crit's common measurement component: [Core](../core/README.md) provides
the runtime, [Forge](../forge/README.md) supplies techniques, [Market](../market/README.md)
helps select kernels, and Eval measures the resulting engine.

One speed and quality suite that every engine runs the same way, so results from different
runs, agents and machines can be compared. Engines don't choose their own prompts, output
lengths, stop handling or quality references; `crit eval` fixes all of them.

```text
crit eval run --adapter my_adapter.py --model MODEL_DIR --data EVAL_DATA_DIR --out result.json
crit eval compare a.json b.json
```

Eval currently ships in the source checkout or full source snapshot, not the
installed CLI wheel. From a checkout, use `python -B tools/crit.py eval ...`
without installing crit. The harness can also run as `python -B eval/crit_eval ...`
on another host (for example WSL) with the complete `eval/` directory, including
its suite and protocol files, the project license and the engine adapter.

`run` prepares missing data and, when quality evaluation is requested, builds a
missing reference. The two steps can also run separately:

| Step | Command | Needs |
| --- | --- | --- |
| Freeze the suite as token ids for one model | `crit eval prepare --model M --data D` | `tokenizers`, `jinja2` |
| Reference log-probabilities | `crit eval reference --model M --data D` | `torch`, `transformers`, `safetensors`, a GPU |
| Evaluate an engine against prepared data and reference | `crit eval run ...` | Python standard library and the engine's dependencies |

Use one data directory per model. Every engine evaluated against it receives exactly the same
token ids and is judged against the same reference.
Keep preparation data, reference files and results outside product folders, such
as in an engine run's workspace. Reference generation requires a compatible CUDA
GPU and model implementation; the evaluator itself can call a CPU or GPU engine.

## What is measured

**Speed** ([suite/speed.json](suite/speed.json)): five greedy cases, each generating 256 tokens.

| Case | Prompt | Why it is there |
| --- | --- | --- |
| `chat` | short prose question | everyday chat |
| `code` | write a Python module | code is the easiest text to draft |
| `math` | multi-step word problems | reasoning |
| `german` | long German email | non-English text, which exposes draft-vocabulary shortcuts |
| `long-summary` | 3,584-token document, summarize | long-context prefill and decode |

Speculative decoding makes decode speed depend on the text, so one prompt can't stand for an
engine. With all five cases measured, the **decode score** is the geometric mean
of their medians, and every case
also reports tokens per round. Stop tokens are disabled so all engines produce the same
number of tokens; prompts ask for long answers, and an output that runs past the model's own
end token is flagged `stop_crossed`. Defaults: 1 warmup, 3 measured repetitions
([protocol.json](protocol.json)). The harness times tokens as the engine delivers them:
time to first token gives prefill speed, and the remaining tokens give decode speed.

**Headroom:** every case also reports `round_ms`, decode time per target pass (one token
per pass for ordinary decoding). Optional adapter fields add the ceiling:

- `draft_width` in `info()`: each case reports **saturation**, tokens per round ÷ (draft width
  + 1), with a warning at 80%.
- `bytes_per_pass` in `info()` (weights read, not gathered embeddings, plus KV and recurrent
  state at short context) and `--bandwidth GBS` from a copy test: the **pass floor**,
  compared with the ordinary one-token step measured during the exactness run (or the speed
  cases for an ordinary engine). At 90% of the floor the pass is polished.
- `accepted` from `generate()`, drafts accepted in each round: with greedy acceptance this
  gives tokens per round at every width up to the one measured. Combined with the draft cost
  (`draft_ms` in `info()`, else round time beyond one pass ÷ width) Eval projects the **best
  draft width at current costs**. With `draft_bytes_per_step` in `info()` it also projects
  the best width at floor costs and the **speed-of-light decode score**: the geometric mean
  over cases of tokens per round ÷ (pass floor + width × draft floor). Measure at a generous
  width so the projection can see beyond it.

Results carry this under `headroom`; `run` prints the fraction of speed of light and
`compare` shows it.

**Quality** ([suite/quality.json](suite/quality.json)): six fixed conversations (prose, code,
math, German, JSON, story; 1,434 scored tokens for Qwen3.8). The final assistant turn is scored
with teacher forcing on every path the engine has: `prefill`, `decode` (one token at a time)
and `verify` (speculative verify blocks). Each path is compared with the reference:

- perplexity ratio against the reference,
- KL(reference ‖ engine), mean / p99 / max, over the reference's top-16 tokens plus one lumped
  tail bucket (a lower bound of the full-vocabulary KL, exact when the tail is negligible),
- top-1 agreement with the reference.

Default thresholds: PPL ratio ≤ 1.01, mean KLD ≤ 0.01, p99 KLD ≤ 0.1.

**Exactness**: for a speculative engine, the greedy outputs of all speed cases must equal the
same engine's non-speculative greedy outputs token for token.
This is a separate check from agreement with the quality reference. It runs for
measured speed cases when the adapter declares speculative decoding; skipped
cases provide no exactness coverage.

The **reference** runs the model's official Hugging Face `transformers` modeling code with
FP32 activations, no TF32, and the stored weights dequantized exactly (unquantized, FP8 E4M3
with channel or tensor scales, NVFP4). Layers are streamed onto the GPU one at a
time. Memory needs and preparation time depend on the model and target hardware.

## The eval-v1 adapter

An engine joins by providing a Python file that exports `create_eval_engine(context)`.
[examples/mock_adapter.py](examples/mock_adapter.py) is a complete example.

`context` holds `model` (the directory passed to `--model`), `config` (the `--config` JSON),
and `speculative` (false when the harness wants the engine's plain one-token decoding for the
exactness check). The returned object provides:

| Method | Returns |
| --- | --- |
| `info()` | `name`, `configuration` (settings actually applied), `max_context`, `speculative`, `score_paths`; optional `draft_width`, `bytes_per_pass`, `draft_bytes_per_step` and `draft_ms` for headroom |
| `generate(prompt_ids, max_new_tokens, on_tokens)` | Greedy, stop tokens disabled, exactly `max_new_tokens` tokens. Call `on_tokens(list)` as soon as tokens are available on the host. Return `{"tokens": [...], "rounds": int or None}`, where rounds counts target-model passes after prefill; optional `accepted` lists drafts accepted per round |
| `score(ids, first_target, path, queries)` | For each target `ids[i]`, `i >= first_target`, on that path: `target_logprobs`, `query_logprobs` (log-probabilities of the ids in `queries[i - first_target]`) and optionally `argmax` |
| `close()` | Release the engine |

Log-probabilities are natural-log softmax over the token ids that exist. The adapter calls the
engine's own code; the harness computes every metric. A result file records the engine's
reported configuration, the host, the model-metadata and suite identities, every
sample and the generated tokens.

This adapter is separate from the [resident chat adapter](../docs/CHAT.md).
Both can use the same underlying engine, but Eval needs token-level generation
and scoring through this contract. Retain the adapter with the engine's source.

## Compare and retain results

Run the default complete suite for a common comparison. `--only speed`,
`--only quality` and repeated `--only` options are useful during development.
Exactness also needs speed output: use `--only speed --only exactness` when
omitting quality. `--warmups` and `--repetitions` override the protocol defaults;
report those settings when comparing results.

Read coverage alongside each score:

- Speed cases exceeding the adapter's declared context capacity are skipped.
  `speed.complete` records whether every case ran; the decode score uses only
  measured cases, so a partial score is not the full-suite score.
- Quality checks run the adapter's declared scoring paths. Unsupported paths
  are listed separately; a quality pass does not establish coverage of them.
- Speculative exactness is reported separately and covers the measured speed
  cases. Ordinary engines and runs without speed results have no exactness result.
- Eval records load time and token-delivery timings. Physical RAM/VRAM use,
  cancellation and other lifecycle checks need separate engine measurements.

Compare engines using the same prepared data, reference and Eval revision.
The comparison command warns when model-data or suite identities differ, but
does not enforce matching reference files, protocol overrides or hardware.
Eval's model identity covers configuration/tokenizer bytes and weight-file sizes;
it does not hash every weight byte. Keep the exact model identity in the engine's
run records and retain preparation/reference files with the result.

Eval writes standalone JSON and a comparison table. A successful command exit
does not mean that quality or exactness passed: inspect the result fields.
There is no automatic publication into candidate checks, benchmarks or the
terminal radar. Follow [Building engines](../docs/BUILDING_ENGINES.md#run-the-common-eval-suite)
and [Eval evidence](../docs/RUN_RECORDS.md#eval-evidence) to retain results and
publish evidence under the run's fixed criteria.
