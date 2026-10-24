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
| `info()` | `name`, `configuration` (settings actually applied), `max_context`, `speculative`; optional `draft_width`, `bytes_per_pass`, `draft_bytes_per_step`, `draft_ms`, `kv_bytes_per_token` (KV bytes per token at the stored precision) for headroom; `replay: true` when `replay()` exists; `plain: true` when `generate_plain()` exists; `native_libraries` (paths of loaded native libraries; Eval records sha256 and size); `kernel_work` (for `crit eval kernels`) |
| `generate(prompt_ids, max_new_tokens, on_tokens)` | Greedy, stop tokens disabled, exactly `max_new_tokens` tokens. Call `on_tokens(list)` as soon as tokens reach the host. Return `{"tokens": [...], "rounds": int or None}` (target passes after prefill); optional `accepted` (drafts accepted per round), `verify_ms` and `draft_ms` (device time in verify passes and draft steps). An engine behind a chat API, which sees only text, returns `served` (`prompt_tokens` and `output_tokens` as its server counted them): its reply may end early, and `tokens` only identify the text |
| `replay(prompt_ids, continuation, on_tokens)` | Optional. The speculative loop on a fixed continuation: drafts accepted while they equal its next tokens, each correction or bonus token taken from it, exactly its tokens emitted. Same return fields |
| `generate_plain(prompt_ids, max_new_tokens, on_tokens)` | Optional, speculative engines. Plain greedy on the loaded engine: one token per target pass, no drafting. Same contract as `generate`. Eval then runs the exactness check on this engine instead of opening a second one with `speculative` false |
| `generate_sampled(prompt_ids, max_new_tokens, on_tokens, sampling, logprobs=None)` | Optional, from crit-eval-v7. Sample at `sampling`, which names every field (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `seed`; `top_k` 0 is no filter), stop tokens disabled; same return fields as `generate`. An engine that applies only some fields lists them as the attribute `sampling_fields`: Eval then sends only those, and needs every field whose value is not neutral With `logprobs` n, also return `logprobs`: per token `{"token", "logprob", "top": [[token, logprob], ...n]}`, the model's own log-probabilities before temperature and filters. Without it the engine has no sampled decode ([Sampled speed](MEASUREMENTS.md#sampled-speed)) |
| `close()` | Release the engine |

The adapter calls the engine's code; the harness computes every metric and records the
configuration, host, model and suite identities, every sample and the generated tokens. Quality is
measured through crit serve, on the engine's scores of fixed text from the paths it serves with
([Engines behind a server](README.md#engines-behind-a-server)): an adapter's engine gets no quality
figures.

One `crit eval run` opens engines one after another in the same process, never two at once: at the
lower context, at the required context and, unless the engine declares `plain`, a plain one for
exactness. Loading dominates a round for engines that stream weights from host memory, so an
adapter may keep host-side weights (for example pinned expert records) in the process and hand them
to the next engine it opens for the same model. Under `crit run ... eval` the served candidate's engines
are built in one resident host process for the same reason: the run's context is built again where
the lower context ran, and an adapter that keeps host-side weights in the process finds them there.
Build everything on the device fresh for each engine:
KV cache, expert or weight caches, workspaces and graphs. Device state carried over from an earlier
engine would bias the lower-context comparison.

Within one engine, nothing a request leaves behind may change a later result. Caches
may stay warm, but their contents must be complete and correct whenever they are read: Eval's
request-history gate compares outputs on the freshly opened engine with outputs after every
other part, long-context requests included.

A Core engine built as a shared library with `CRIT_EXPORT_ENGINE`
([engine libraries](../../core/src/runtime/README.md#engine-libraries)) needs no adapter code of
its own; the adapter file only points at the library:

```python
from pathlib import Path
from crit_core.native import eval_engine

def create_eval_engine(context):
    return eval_engine(context, Path(__file__).with_name("libengine.so"), defaults={"mode": "fast"})
```

It generates greedily with stops disabled, offers `replay` and `generate_plain` when the engine
declares `replay` or `plain`, `generate_sampled` when it samples (with log-probabilities when it
reports `token_logprobs`), and reports the engine's facts
(`draft_width`, `bytes_per_pass` and the rest).

An engine behind an OpenAI-compatible server needs no adapter:
[`crit eval run --url`](README.md#engines-behind-a-server) measures it through the server, quality
included.

This adapter is separate from the [resident adapter used by crit serve](../SERVING.md#adapter-contract);
both can wrap the same engine. Retain it with the engine's source.
