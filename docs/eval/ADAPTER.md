# The eval-v1 adapter

[Overview](README.md) · [Measurements](MEASUREMENTS.md) · [Context](CONTEXT.md) · [Adapter](ADAPTER.md)

An engine joins with a Python file exporting `create_eval_engine(context)`;
[examples/mock_adapter.py](../../eval/examples/mock_adapter.py) is a complete example.

`context` holds `model` (the `--model` directory), `config` (the `--config` JSON),
`speculative` (false when the harness wants plain one-token decoding for the exactness check
from a second engine) and, with a [required context](CONTEXT.md), `context_tokens`. Size the KV
cache and every length-dependent table for it at load and report `max_context` of at least it. The returned
object provides:

| Method | Returns |
| --- | --- |
| `info()` | `name`, `configuration` (settings actually applied), `max_context`, `speculative`, `score_paths`; optional `draft_width`, `bytes_per_pass`, `draft_bytes_per_step`, `draft_ms`, `kv_bytes_per_token` (KV bytes per token at the stored precision) for headroom; `replay: true` when `replay()` exists; `plain: true` when `generate_plain()` exists; `native_libraries` (paths of loaded native libraries; Eval records sha256 and size); `kernel_work` (for `crit eval kernels`) |
| `generate(prompt_ids, max_new_tokens, on_tokens)` | Greedy, stop tokens disabled, exactly `max_new_tokens` tokens. Call `on_tokens(list)` as soon as tokens reach the host. Return `{"tokens": [...], "rounds": int or None}` (target passes after prefill); optional `accepted` (drafts accepted per round), `verify_ms` and `draft_ms` (device time in verify passes and draft steps) |
| `replay(prompt_ids, continuation, on_tokens)` | Optional. The speculative loop on a fixed continuation: drafts accepted while they equal its next tokens, each correction or bonus token taken from it, exactly its tokens emitted. Same return fields |
| `generate_plain(prompt_ids, max_new_tokens, on_tokens)` | Optional, speculative engines. Plain greedy on the loaded engine: one token per target pass, no drafting. Same contract as `generate`. Eval then runs the exactness check on this engine instead of opening a second one with `speculative` false |
| `score(ids, first_target, path, queries)` | For each target `ids[i]`, `i >= first_target`, on that path: `target_logprobs`, `query_logprobs` (for the ids in `queries[i - first_target]`) and optionally `argmax`. Run the whole sequence through the scored path (`decode` steps every position from the start) |
| `close()` | Release the engine |

Log-probabilities are natural-log softmax over the token ids that exist. A Core engine can serve
`score()` through `Engine::submit_score` with `ScoreQueries` (one row of query ids per target),
which returns target and query log-probabilities and the argmax. The adapter calls the engine's
code; the harness computes every metric and records the configuration, host, model and suite
identities, every sample and the generated tokens.

One `crit eval run` opens engines one after another in the same process, never two at once: at the
lower context, at the required context and, unless the engine declares `plain`, a plain one for
exactness. Loading dominates a round for engines that stream weights from host memory, so an
adapter may keep host-side weights (for example pinned expert records) in the process and hand them
to the next engine it opens for the same model. Build everything on the device fresh for each engine:
KV cache, expert or weight caches, workspaces and graphs. Device state carried over from an earlier
engine would bias the lower-context comparison.

A Core engine built as a shared library with `CRIT_EXPORT_ENGINE`
([engine libraries](../../core/src/runtime/README.md#engine-libraries)) needs no adapter code of
its own; the adapter file only points at the library:

```python
from pathlib import Path
from crit_core.native import eval_engine

def create_eval_engine(context):
    return eval_engine(context, Path(__file__).with_name("libengine.so"), defaults={"mode": "fast"})
```

It generates greedily with stops disabled, scores on the engine's paths, offers `replay` and
`generate_plain` when the engine declares `replay` or `plain`, and reports the engine's facts
(`draft_width`, `bytes_per_pass` and the rest).

This adapter is separate from the [resident adapter used by crit serve](../SERVING.md#adapter-contract);
both can wrap the same engine. Retain it with the engine's source.
