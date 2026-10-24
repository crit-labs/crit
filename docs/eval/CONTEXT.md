# Eval context

[Overview](README.md) · [Measurements](MEASUREMENTS.md) · [Context](CONTEXT.md) · [Adapter](ADAPTER.md)

Every engine is built for one **required context**: the longest request, prompt plus output, it
must serve. It is the context the user wants, as far as the model and device allow; serving less
may only make the engine faster.

## The max context

`crit eval context --model MODEL_DIR --device device.json` prints it with its basis
([crit_eval/context.py](../../eval/crit_eval/context.py)):
`max context = min(requested, model limit, device limit)`.

- **Requested:** the longest request the user wants served, `context_tokens` in `settings.toml`
  (default 262,144; `--requested-context` overrides it). When the model or the device allows
  less, they bound it and the result's `note` says so, e.g. "the requested 262,144 tokens exceed
  the model's 131,072 positions". It is the user's setting: engine builders do not change it.

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
  at the requirement. Each prompt is the retrieval check's archive, suite filler text
  ([long.json](../../eval/suite/long.json)) cut to fit with three retrieval facts in it (at 2%, 50%
  and 97%), and `retrieval.curve_user`'s request: list the facts, then summarize the archive in
  detail. Every prompt is chat messages that an engine behind a server is sent too. Each point
  times the engine's own reply, so `context_decode` compares between results of the same
  requirement, and judges retrieval on it: one prefill of each long prompt serves both. The facts
  are framed as part of the archive the user asks about: inside an agent's file reads a model may
  take a request for "secret identifiers" for a prompt injection and refuse. Prefilling the
  longest points can take minutes.
- **Retrieval gate:** every curve point is a retrieval point, judged on its reply. Each of
  4,096, 32,768 and 131,072 tokens below the requirement that the curve does not reach (4,096
  always) gets one greedy request of its own (64 output tokens) after the curve, prompt plus
  output filling it: suite text with the three facts at 2%, 50% and 97% of it, then a question
  asking for all of them. The output up to its first stop token, decoded with the token strings
  `prepare` records, must contain every answer verbatim. A retrieval request of its own is judged
  on its text, so it may be served from a prompt cache (its cached tokens are recorded); every
  timed request still prefills. Capacity shows the engine holds the context; retrieval that it
  still reads all of it correctly. A point it cannot serve fails.
- **Long-context goals:** `context_decode` is the geometric mean of decode tok/s over the points
  against their floors (pass floor with the KV read at that context, plus draft steps at their
  floor, at the measured tokens per round). `context_ttft` is the geometric mean TTFT against
  the points' TTFT floors.
- **Lower-context gate:** just before the full build opens, the speed cases run on an engine
  opened at a quarter of the requirement (never below the longest speed case). A case fails when
  it decodes slower, or reaches its first token later, than on the full build by more than 5% or
  its own run-to-run spread, whichever is larger. Failing to open or run there fails too.

`--only context` runs the curve and retrieval; `--only speed` also runs the lower-context gate.

## What this asks of an engine

Size the KV cache and every length-dependent table, workspace and graph for `context_tokens` at
load. Speed must depend on the tokens a request holds, never on capacity: a short chat on an
engine built for 262,144 tokens runs as fast as on one built for 32,768. Spend memory a lower
context frees where it helps; for experts that spill to host memory or disk, that is the device
expert cache.
