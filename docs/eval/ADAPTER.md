# The eval-v1 adapter

[Overview](README.md) · [Measurements](MEASUREMENTS.md) · [Context](CONTEXT.md) · [Adapter](ADAPTER.md)

An engine joins with a Python file exporting `create_eval_engine(context)`;
[examples/mock_adapter.py](../../eval/examples/mock_adapter.py) is a complete example.

`context` holds `model` (the `--model` directory), `config` (the `--config` JSON),
`speculative` (false when the harness wants plain one-token decoding for the exactness check)
and, with a [required context](CONTEXT.md), `context_tokens`. Size the KV cache and every
length-dependent table for it at load and report `max_context` of at least it. The returned
object provides:

| Method | Returns |
| --- | --- |
| `info()` | `name`, `configuration` (settings actually applied), `max_context`, `speculative`, `score_paths`; optional `draft_width`, `bytes_per_pass`, `draft_bytes_per_step`, `draft_ms`, `kv_bytes_per_token` (KV bytes per token at the stored precision) for headroom; `replay: true` when `replay()` exists; `native_libraries` (paths of loaded native libraries; Eval records sha256 and size); `kernel_work` (for `crit eval kernels`) |
| `generate(prompt_ids, max_new_tokens, on_tokens)` | Greedy, stop tokens disabled, exactly `max_new_tokens` tokens. Call `on_tokens(list)` as soon as tokens reach the host. Return `{"tokens": [...], "rounds": int or None}` (target passes after prefill); optional `accepted` (drafts accepted per round), `verify_ms` and `draft_ms` (device time in verify passes and draft steps) |
| `replay(prompt_ids, continuation, on_tokens)` | Optional. The speculative loop on a fixed continuation: drafts accepted while they equal its next tokens, each correction or bonus token taken from it, exactly its tokens emitted. Same return fields |
| `score(ids, first_target, path, queries)` | For each target `ids[i]`, `i >= first_target`, on that path: `target_logprobs`, `query_logprobs` (for the ids in `queries[i - first_target]`) and optionally `argmax`. Run the whole sequence through the scored path (`decode` steps every position from the start) |
| `close()` | Release the engine |

Log-probabilities are natural-log softmax over the token ids that exist. A Core engine can serve
`score()` through `Engine::submit_score` with `ScoreQueries` (one row of query ids per target),
which returns target and query log-probabilities and the argmax. The adapter calls the engine's
code; the harness computes every metric and records the configuration, host, model and suite
identities, every sample and the generated tokens.

This adapter is separate from the [resident adapter used by crit serve](../SERVING.md#adapter-contract);
both can wrap the same engine. Retain it with the engine's source.
