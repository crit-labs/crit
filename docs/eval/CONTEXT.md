# Eval context

[Overview](README.md) · [Measurements](MEASUREMENTS.md) · [Context](CONTEXT.md) · [Adapter](ADAPTER.md)

Every engine is built for one **required context**: the longest request, prompt plus output, it
must serve. Eval computes the largest the model and device allow; serving less may only make
the engine faster.

## The max context

`crit eval context --model MODEL_DIR --device device.json` prints it with its basis
([crit_eval/context.py](../../eval/crit_eval/context.py)): `max context = min(model limit, device limit)`.

- **Model limit:** the positions `config.json` declares (`max_position_embeddings` or another
  length key, from the text config of a multimodal model); with YaRN rope scaling, the original
  positions times the factor.
- **Device limit:** from the least free memory any `crit eval probe` run saw, subtract the VRAM
  headroom (default 1 GiB), the reserve for activations, workspaces and graphs, the weights read
  on every pass and the state that does not grow with context. Divide the rest by the KV bytes
  per token at the protocol's KV precision and round down to 1,024 tokens.

Weights read every pass are all tensors except gathered embedding rows (unless the output head
is tied to them), the vision tower and mixture-of-experts experts; draft layers count. Experts
are read sparsely and may live in host memory or on disk; at the max context they get only the
device memory the KV cache leaves, and the result says how much. Fixed state is the
linear-attention recurrence (FP32 state, BF16 convolution window) and sliding-window KV. The KV
cache stays on the device because decode reads all of it every pass.

The `context` section of [protocol.json](../../eval/protocol.json) fixes the KV precision (FP8),
the 3 GiB reserve and the rounding. `--context-reserve-gib` (or `context_reserve_gib` in
`settings.toml`) replaces the reserve and `--vram-headroom-gib` (`vram_headroom_gib`) the
headroom. A larger reserve trades context for speed techniques such as faster weight layouts,
draft copies and device-resident tables. A run freezes the max context as its `kv_tokens` when
its request names the device ([run records](../RUN_RECORDS.md#request)); an explicit
`kv_tokens` overrides it.

## Checks at the required context

`crit eval run --context TOKENS` gives the required context; without it Eval computes it from
`--model` and `--device`; with neither, engines open at their default and the context gates are
unmeasured.

- **One build:** the engine opens with `context_tokens` set to the requirement, and speed,
  quality and exactness all run on it.
- **Capacity gate:** `max_context` is at least the requirement and a request filling it completes.
- **Long-context curve:** after speed and quality, one greedy request per point fills the context
  exactly, prompt plus 256 output tokens, at powers of two from 32,768 below the requirement and
  at the requirement. Prompts are copies of the longest speed case's prompt, ending with a whole
  one. With a pinned continuation and a replaying adapter, each point replays that case's pinned
  text, so every engine decodes the same tokens (as in the speed cases); otherwise it decodes its
  own output, and `context_decode` compares only between results decoded on the same text.
  Prefilling the longest points can take minutes.
- **Long-context goals:** `context_decode` is the geometric mean of decode tok/s over the points
  against their floors (pass floor with the KV read at that context, plus draft steps at their
  floor, at the measured tokens per round). `context_ttft` is the geometric mean TTFT against
  the points' TTFT floors.
- **Lower-context gate:** just before the full build opens, the speed cases run on an engine
  opened at a quarter of the requirement (never below the longest speed case). A case fails when
  it decodes slower, or reaches its first token later, than on the full build by more than 5% or
  its own run-to-run spread, whichever is larger. Failing to open or run there fails too.

`--only context` runs the curve; `--only speed` also runs the lower-context gate.

## What this asks of an engine

Size the KV cache and every length-dependent table, workspace and graph for `context_tokens` at
load. Speed must depend on the tokens a request holds, never on capacity: a short chat on an
engine built for 262,144 tokens runs as fast as on one built for 32,768. Spend memory a lower
context frees where it helps; for experts that spill to host memory or disk, that is the device
expert cache.
