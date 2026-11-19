# Compose a model plugin from blocks

Write a model plugin by choosing and configuring tested blocks instead of writing the architecture and its serving
from scratch. The plugin `model-template` fills the `model` slot for every checkpoint one of its layouts covers; an
engine whose architecture none covers composes its own model plugin from the same code: a layout file, and a block
class for a layer part no block computes.

## When to use it

Use it for every engine whose checkpoint is a decoder of the kinds below, and as the base of an engine's own model
plugin otherwise. The serving contracts are where hand-written model plugins fail crit's checks most often (a prompt
step past the cancel grace at the full context, a prompt row that is not the decode row's bits, a decode row that
changes with the rows beside it, memory at a step's peak), and the template keeps all of them in one place.

Don't use it for a post-norm decoder, an encoder, or attention the attention slot does not compute; write those
parts as blocks of your own (below), or a model plugin of your own where the layer loop itself differs from both
residuals the template has.

## What it is made of

| Part | What it does | File |
| --- | --- | --- |
| Layouts | Data: which block each layer kind of the config runs (`layer_types`: full_attention, sliding_attention, linear_attention, or the config's own name of a layer type), the FFN of each layer (`auto`: a mixture of experts on the layers the config routes, `crit_core.model.routed_layers`, a SwiGLU MLP elsewhere), how the blocks join the residual (`residual`), blocks that add rows into the residual ahead of some layers (`before`), the norm offset (1 for zero-centred RMSNorms), an attention output gate, the draft layer's prefix, and the config values the family's reference assumes where config.json is silent (`defaults`) | [code/layouts](code/layouts) |
| Residuals | `pre-norm`: one stream, an RMSNorm before each block, the block's output added to the stream, a final norm. `streams`: several residual streams (hyper-connections); each block reads one row mixed from the streams' norms by weights the streams compute themselves (low-rank maps over all of them), its output is added to every stream by weights of its own, and a last mix gives the row the head reads. `sandwich`: one stream, each block between two norms (its normed output added to the stream), a scalar a layer, and the stream before the FFN's norm given to an FFN that reads it (`Context.residual`) | [code/residual.py](code/residual.py) |
| Blocks | The arithmetic of each layer part over the other slots' plugins: `attention` (grouped-query softmax attention: one stacked q, k, v matrix, biases, query/key norms where the checkpoint stores them, rotary positions from the config's rope table (its layer type's where the config keys its rope by layer type, the `proportional` type included), partial rotation, sliding windows, sinks, a sigmoid output gate; for families whose layer kinds differ in shape, a kind's own head width and key/value heads, a fixed scale, value norms, and values taken from the key projection's rows where a layer stores no value projection), `indexed-attention` (block-sparse attention: an indexer scores the complete blocks of positions before a token and the token reads its best ones, within a budget, and its own block; the index keys in a pool of keys only, the choice on crit's kernel sparse-indexer where it loads), `latent-attention` (multi-head latent attention in its absorbed form: one cached row a position, its latent and a rotary key every head shares, in a pool of keys only; low-rank queries multiplied into the latent space, the attention slot's `latent=`, the values' up-projection after it), `gated-delta` (the delta rule through the recurrence slot, either projection layout a checkpoint stores, the gated norm's activation by the config), `ngram-rows` (a `before` block: rows of a hashed n-gram table, held on the device or read where the checkpoint stores it, gated into every stream, with a dilated causal convolution through the recurrence slot), `swiglu` (the gated MLP, its gate SiLU or tanh-GELU by the config's activation), `moe` (the router slot's selection with the router's rule checked against the config, the experts slot over any experts plugin, expert-offload included, a shared expert and its gate), `parallel-moe` (a dense MLP and a mixture of experts side by side, each normed and summed, the router scoring the residual stream itself with a scale an expert; experts gated by GELU need an experts plugin that provides `experts:activations`) | [code/blocks.py](code/blocks.py) |
| The draft layer | The checkpoint's multi-token-prediction layer as the device draft head of Core's captured rounds: one decoder layer of the model's own blocks in its own pools, entered and left as the residual does it (pre-norm: fc over the normed embedding and the normed target hidden state, the final norm; streams: every stream of the target through fc_hidden, the normed embedding through fc_embedding added to each, the layer's own last mix) | [code/model.py](code/model.py) `MtpHead`, [code/residual.py](code/residual.py) |
| The serving contracts | Exact rows and prompt spans ([below](#the-serving-contracts)) | [code/rows.py](code/rows.py) |
| The step model | Core's model interface over the blocks: the placement plan of every device tensor, the pools, `step`, `commit`, `retire`, `score`, admission, Core's captured rounds (greedy and sampled, drafted), captured prompt chunks, Core's prompt cache, constrained, replayed and plain requests, info, facts and telemetry | [code/model.py](code/model.py) |
| compose() | The model plugin's object from a layout and the blocks it names | [code/compose.py](code/compose.py) |
| The checkpoint | Its safetensors files, or a GGUF file read under the same names ([below](#gguf-checkpoints)): the tensors the blocks read and the matrices as the projections and experts slots load them | [code/checkpoint.py](code/checkpoint.py), [code/gguf.py](code/gguf.py) |

The library's layouts:

| Layout | Model types | Tested against transformers on |
| --- | --- | --- |
| `decoder` | llama, qwen2, qwen3, qwen2_moe, qwen3_moe | Llama; Qwen3 with a sliding-window layer; Qwen3-MoE (experts-reference, and expert-offload) |
| `gated-hybrid` | qwen3_next, qwen3_5, qwen3_5_text, qwen3_5_moe, qwen3_5_moe_text | Qwen3-Next (gated delta, gated attention, a gated shared expert); Qwen3.5 text with an MTP layer, drafting |
| `latent` | deepseek_v3 | DeepSeek V3: latent attention with low-rank queries and interleaved rotary pairs, a dense MLP on the first layer, experts routed by sigmoid scores within the best group, a shared expert |
| `gated-streams` | qwen4_exp, qwen4_exp_text | Qwen4Exp: four residual streams, gated delta layers, hashed n-gram rows ahead of one of them, gated block-sparse attention below and past its budget, routed experts with a gated shared expert; its draft layer drafting (transformers runs no draft layer for this architecture: [below](#not-covered-yet)) |
| `sandwich` | gemma4, gemma4_text | Gemma 4 with its mixture-of-experts block: every block between two norms with a scalar a layer, sliding-window layers beside full layers of another head shape whose values are the key projection's rows and whose rotary table turns a quarter of each head, value norms, a GELU-gated MLP beside GELU-gated experts routed from the residual stream, scaled embeddings, capped logits (experts-reference, and expert-offload with the experts in host memory); a config with the experts switched off composes its layers dense |

## The serving contracts

**Exact rows.** Every matmul of a step goes through one function (`RowGroups.matmul`): each prompt chunk alone with
`prompt=True`, so a prompt row takes the prompt rows' arithmetic at every length, and one-token rows (decode rows, a
round's columns, draft steps, logits rows) in calls of at most `invariant_rows(projections)` rows, so each row has
the bits it has alone. Norms, rotary positions and gated products are the norms slot's. A request decodes the same
alone and beside others, and a step's decode rows never take the prompt path: with expert-offload a step's decode
rows and its prompt chunks are separate calls.

**Prompt spans within the cancel grace.** Core cancels a request between steps, and crit serve retires the engine
whose cancel does not settle within its grace (one second): `serve --check` fails a candidate whose slowest prompt
step at the full context takes longer. A prompt chunk ends at its span, a function of the position it starts at
only, so chunk boundaries never depend on what shares a step. At load the template times three prompt steps on a
sequence of its own (a short and a full `prefill_chunk` at position 0, a full one at the deepest position the engine
serves, its earlier KV written with zeros), fits `seconds = c + a * rows + b * rows * (position + rows)` and the peak
bytes a row takes, and gives each position the longest span within `span_seconds` (0.75 s by default: three quarters
of the grace, since a step's fixed part, such as an experts plugin's copies of a chunk's routed experts, is paid once
a step and a smaller budget pays it for fewer tokens; where that part alone passes the budget, the span that prefills
fastest, and the facts' `spans.over_budget` says so beside each step's predicted seconds and rate) and within the
device's room (its free memory less `headroom_mib`, where the experts plugin does not make room itself). The fit is a
[measured decision](../../docs/PLUGINS.md#measured-decisions): `crit plugins decisions` records it in the package so
every process spans its prompts alike. Fixed-text scores run in the same spans. The engine's facts carry the fit
(`spans`). The fit and the spans it gives are plain functions ([code/rows.py](code/rows.py)):

```python
from rows import Spans, fit

# Three prompt steps timed at load: a short and a full chunk at position 0, a full chunk at the deepest position.
costs = fit({(128, 0): 0.060, (512, 0): 0.210, (512, 200_000): 0.290}, peaks={512: 96 << 20})
spans = Spans(512, costs, seconds=0.75, deepest=200_000)
assert spans.rows(0) == 512 and spans.rows(200_000) == 512      # every step within the budget: whole chunks

slow = Spans(512, {**costs, "b": 2e-8}, seconds=0.75, deepest=200_000)
assert slow.rows(0) == 512 and slow.rows(200_000) == 160        # a deep position takes a shorter span
assert slow.predicted(160, 200_000) <= 0.75 < slow.predicted(176, 200_000)

# A fixed part past the budget: no span meets it, so the span is the one that prefills fastest, and the facts say so.
swept = Spans(512, {**costs, "c": 2.4}, seconds=0.75, deepest=200_000)
assert swept.rows(0) == 512 and "over_budget" in swept.facts()
# A pass ends a step between a chunk through one layer and the next: a span's time is that unit's.
assert "over_budget" not in Spans(512, {**costs, "c": 2.4}, 0.75, deepest=200_000, units=48).facts()
```

**Prompts layer by layer (passes).** Chunk by chunk, each prompt chunk runs through every layer before the next
chunk starts. An experts plugin that keeps routed experts off the device brings a layer's experts to it for every
chunk that reads them, and a chunk of a hundred tokens or more routes nearly all of a layer's experts: every chunk
moves every expert of every layer over the link, gigabytes a chunk whatever its size, so prefill runs at the link's
rate divided by the experts' size and a single chunk's step can pass the cancel grace by itself. Where the experts
plugin says it stages prompt chunks so (`stages_prompts()`: expert-offload with experts in host memory or a file), the
template runs a prompt's spans as a pass instead: every chunk of the span through layer 0, then every chunk through
layer 1, and so on, so each layer's experts cross the link once for the span, and the next layer's are staged while
the current one computes. A pass takes as many of Core's steps as its work takes time ([Core's prefill
passes](../../core/src/runtime/README.md#model-progress)): a step runs chunks through layers until `span_seconds` or
half the cancel grace is up (a cancel then also gives back what the pass holds), consumes nothing and reports the
span; the step in which the last layer ends consumes it. So no step runs past
the budget whatever the experts cost, a cancel settles between a pass's steps as between any others, and the spans
are sized by what a chunk takes through one layer (the fit's seconds over the layers), so a chunk stays as long as
the device computes well.

A pass changes no token. A chunk through a layer is the same call with the same inputs in either order: the
layer's keys, values, index keys and recurrent state of the earlier chunks are in its pools when a chunk reads them,
and what a layer leaves the next is each chunk's residual state (`begin`, `advance` and `end` of
[residual.py](code/residual.py)), which the pass keeps between layers in device memory the device's givers make room
for (`pass_mib` bounds it; an elastic expert cache gives the bytes and takes them back after the prompt). Scores run
their spans the same way. A pass ends where the prompt cache marks a place, as a chunk does, and a request cancelled
inside a pass has its sequence dropped rather than kept for later prompts, since its pools stand past the rows it
committed. Sliding-window layers run passes as the others do (a ring page is a position's, and each layer keeps
its own keys and values there). `passes: on` runs passes with any experts plugin (the
tests compare both orders bit for bit on every layout), `off` never; the facts' `passes` say which and why, and the
telemetry's `prompt_passes` count passes, their tokens, chunks and steps.

Measured on two checkpoints whose experts do not fit an RTX 5090 (32 GB; expert-offload at its defaults with the
experts in host memory, 131,072 tokens of context, 1,024-token chunks, fresh prompts, no drafter; prefill tokens a
second, chunk by chunk against layer by layer):

| Prompt tokens | Qwen3.8-Flash-Next (68 GB of NVFP4 experts, 52 GB off the device) | Gemma 4 26B (46 GB of BF16 experts, 29 GB off the device) |
| --- | --- | --- |
| 1,024 | 508 against 489 | 731 against 745 |
| 4,096 | 557 against 1,266 | 968 against 2,445 |
| 16,384 | 555 against 2,131 | 1,050 against 4,727 |
| 65,536 | 2,285 layer by layer | 4,323 layer by layer |
| Slowest prompt step | 2.18 s against 0.62 s | 1.39 s against 0.56 s |
| A cancel inside a long prompt settles after | 0.47 s against 0.61 s | 2.16 s against 0.16 to 0.54 s |

Replies (greedy and seeded sampled) and scores were the same bits in both orders on both checkpoints. One chunk
pays one sweep of the experts either way, so a prompt of a single chunk gains nothing; from the second chunk on the
sweep is shared.

**Memory.** Every tensor the model keeps on the device is planned through the placement plugin before it is made
(weights by name, the KV and state pools and the pools its blocks declare, `activations_mib` for a step's
temporaries); the routed experts are left out of the plan where the experts plugin places them itself
(`experts:offload`), which is begun with the device bytes the model still allocates and opened after its pools. A
lookup table a block reads rows of (the token embeddings, a hashed n-gram table) is planned as `tables.<name>`, which
a placement plugin may put off the device (device-placement's `off_device: "tables."`): the block then reads it where
the checkpoint stores it through an embeddings plugin with `embeddings:host` (lookup-table-rows). A prompt or a score
that ends hands torch's cached blocks back when the device has less than `release_below_mib` free.

**Drafting and batching.** The model declares `max_concurrency` (option `concurrency`), `max_row_tokens` and
`decode_token_work` of the widest round, `prefill_work_quantum` equal to its prefill chunk, with passes
`prefill_pass_tokens` and `prefill_pass_steps`, token log-probabilities
with 20 alternatives, fixed-text and query scores, replays and Core's sampling fields; with `mtp` and a drafter in the
selection it drafts from the checkpoint's MTP layer in captured rounds (a drafter that asks the model's draft head:
`greedy-head-drafter` for the layer's chain, `mtp-decode` with an `override` for long prompt copies over it; one that
never asks it, mtp-decode at its default, is refused), decodes plainly on request, and states
`draft_bytes_per_step`. Requests decode in Core's native rounds once their prompt is in; constrained requests, long
replays and rows the rounds cannot hold decode through `step()`. Its info names the round counters its rounds'
parts count (`round_counters`: an expert cache's `expert_reads` and `expert_hits`), which the round driver collects
from the parts themselves.

**Devices.** CUDA, and the CPU: there every block runs the same arithmetic over the slots' CPU forms, decoding goes
through `step()` (captured rounds and prompt chunks are a GPU's) and no span is timed. It is how the plugin's tests
run where there is no GPU, and a way to check a layout against its reference before a GPU is free.

## How to use it

An engine for a covered checkpoint takes the plugin as it is:

```sh
crit plugins add ENGINE/plugins model-template attention-state softmax-attention quantized-projections \
  token-sampling device-placement fused-norms memory-table-rows sequential-recurrence
```

```json
{"plugins": {"use": {"model": "model-template", "placement": "device-placement", "projections": "quantized-projections",
                     "attention": "softmax-attention", "kv_cache": "attention-state", "sampler": "token-sampling",
                     "norms": "fused-norms", "embeddings": "memory-table-rows", "recurrence": "sequential-recurrence"},
             "options": {"model-template": {"context": 262144, "concurrency": 4, "prefill_chunk": 1024}}}}
```

The manifest requires the slots each model type's layout calls (`requires_when`), so `crit plugins check` names a
missing recurrence or experts slot before anything loads. A mixture of experts takes a router and an experts plugin;
the router's options must state the checkpoint's rule (the template refuses one that does not, naming the options).

**An architecture no layout covers.** Write the layout (the format of [code/layouts](code/layouts): `model_types`,
`mixers` by layer kind, `ffn` (with `auto`, `routed` and `dense` may name the blocks of routed and dense layers),
`residual`, `norm_offset`, `draft`, and where the family has them `embed_scale` (`sqrt_hidden`), `logit_softcap`
(the config key of a tanh cap on the logits), `routed_if` (the config key that switches the family's experts on) and
`requires` (config values the layout's blocks compute for; any other value is refused at load))
beside the plugin and name it with the option `layout`. Where
a layer part needs arithmetic no block has, write the block in the engine's own model plugin and compose with it:

```text
# ENGINE/plugins/my-model/plugin.py; plugin.json: "sources": {"template": "../model-template/code",
#                                                             "rope": "../attention-state/code"}
from pathlib import Path
from .template.blocks import Attention
from .template.compose import compose, tuning

class WindowedAttention(Attention):          # a block: __init__(model, layer, prefix, spec), forward(h, ctx)
    def forward(self, h, ctx):
        ...                                  # ctx.mm for every matmul, model.norms for every norm

def create(context, options, parts):
    folder = Path(__file__).resolve().parent
    tuning(options, folder)
    return compose(context, options, parts, layout="layout.json", folder=folder,
                   blocks={"windowed-attention": WindowedAttention})
```

A mixer block has `pool` (the key of the pool its layer's state lives in: `"kv"`, `("kv", window)`, `"states"`, or a pool it declares, with `keeps_kv = False` where its layers keep no KV in the model's pools),
`forward(h, ctx)` and, where a draft layer may run it, `write(h, ctx)` (its rows' state written, nothing read); an
FFN block has `forward(h, ctx)`; a `before` block has `forward(residual, ctx)` and returns the residual with its rows
added. `ctx` is the step: its batch, positions, row groups, the slots each pool reserved, the tokens' ids and, for a
block that sets `token_window`, each token's window of ids (`ctx.windows`). A block that keeps state beside its
layer's pool declares it as a class method, `pools(model, spec, layers)` giving `[{key, kind: "keys" | "window",
layers, ...}]` (a pool of keys only, a row a position; the window of a causal depthwise convolution), and a lookup
table as `tables(model, prefix, spec)`: the model plans their bytes with every other tensor, makes the pools and
follows them through steps, rounds and the prompt cache as it does its KV and state pools. Every matmul through
`ctx.mm`, every norm and product through the norms slot, every attention and recurrence through their slots, and a
row sum of its own in one fixed order (`blocks.row_sum`): the contracts then hold for the new block too. Test the
composed model as this plugin's tests do: a small random checkpoint of the architecture with every part the real one
ships, scored and decoded against its reference implementation, batched against alone.

## GGUF checkpoints

A GGUF file (or a split set, or a folder holding one) is read as a safetensors checkpoint is: Core derives its
config.json from its metadata (or reads the base model's beside it), and `code/gguf.py` gives the blocks its tensors
under the Hugging Face names they read, from the file's own general.architecture and the convention every converter
names tensors by (`blk.N.attn_q`, `blk.N.ffn_up_exps`, `blk.N.ssm_out`, `blk.N.hc_attn_down`, `token_embd`, `output`,
...). What a converter changed on the way is undone exactly: zero-centred norms stored as 1 + w and A_log stored as
-exp(A_log) are inverted in FP32; Llama's permuted query and key rows and a gated delta net's tiled value heads are
moved back as whole rows, or along a matrix's inputs as whole blocks of its stored format, so a quantized matrix is
never decoded for it; merged tensors (attn_qkv_gate, ssm_beta_alpha, hc_*_down_inject) read as their halves, the
split indexer query and key rows as one; the n-gram table and its hash constants (moved to metadata) as the
checkpoint's; DeepSeek's kv_b_proj, which the converter splits by heads into attn_k_b (each head's key rows
transposed) and attn_v_b, joined back and decoded (latent attention reads it as a plain matrix). The draft (nextn)
block after the text layers is read as the layout's draft layer under `mtp.`: its decoder layer's parts as any
layer's, eh_proj as the MTP layer's fc or, for a layer that takes them apart, its input halves fc_embedding and
fc_hidden as whole blocks of their columns, enorm, hnorm and shared_head_norm as its norms (a draft block's own
embedding and head copies are not read: the template drafts with the model's own). Each architecture's rules are data
in `ARCHITECTURES`: llama, qwen2, qwen3, qwen2moe, qwen3moe, qwen35, qwen35moe, qwen4exp (Qwen3.8-Flash-Next's GGUFs),
gemma4 (its norms in pairs around each block, the layer scalar, the router's scale and each expert's scale as .scale
tensors, the experts one gate-up stack at the layer's own prefix) and deepseek2 (DeepSeek V2 and V3: latent attention's
low-rank matrices, the router's score correction as exp_probs_b.bias, the shared experts). A tensor that maps to no
name is refused at load, so a converter's new part fails there and not as a wrong answer.

Quantized matrices load as stored (`gguf:<TYPE>`, for [gguf-projections](../gguf-projections/README.md)); routed
experts reach an experts plugin that takes them as stored (`experts:gguf`: expert-offload) as each expert's rows of
the file's stacks with each layer's formats, and any other experts plugin as their stored values in BF16. A table
stored in a block format (an n-gram table IQ4_NL, token rows) is read through an embeddings plugin that decodes its
rows on lookup (`embeddings:gguf`: lookup-table-rows); so are quantized token embeddings, kept as the file's rows
on the device or read in place where the plan puts them off it. The file's dense tensors and its quantized tensors
that no matrix load or table takes (the hyper-connections' maps; the token embeddings through an embeddings plugin
without `embeddings:gguf`) are decoded on the host by Core's decoders.

Tested by `tests/test_gguf.py` (every tensor of each architecture's rules read back from a file written as the
converter writes it, by `tests/gguf_checkpoint.py`) and as whole engines against transformers on GGUF copies of the
harness's checkpoints (Llama here; Qwen3-MoE through expert-offload in host memory, Qwen3.5-MoE, Qwen3.5 drafting from
its MTP layer, Qwen4Exp with its draft layer, Gemma 4 with its experts and DeepSeek V3 in crit's
tests/test_model_template.py), their matrices Q8_0.

## Not covered yet

An indexer that scores every position's own key with weighted heads (`indexed-attention` is the block-pooled form:
unweighted heads over the mean keys of fixed blocks), a draft layer for the `latent` layout, latent attention's
per-head maps in a quantized format (they are plain matrices) or on a kernel (they run in torch, a token at a time
for decode rows), Gemma 4's smaller variants (layers that share another layer's keys and values, a gated per-layer
input, the double-wide MLP: the `sandwich` layout refuses a config that has them, `requires`), FP8 in blocks other than MXFP8's
(the projections slot loads FP8 per channel, per tensor and MXFP8), experts stored in other names than
`experts.N.{gate,up,down}_proj` or the fused `gate_up_proj` / `down_proj`, captured prompt chunks for a layout with
`before` blocks or a streams residual that drafts (their chunks run eager). A draft layer shipped in a GGUF of its own
(a converter's mtp-only file), a lone gemma4 or deepseek2 GGUF without its base model's config.json (Core derives no
config for those architectures yet), a Gemma 4 GGUF whose converter trimmed the token rows past the tokenizer's
vocabulary, and a gated delta net whose value heads the converter tiled in a matrix stored in a format whose blocks are
wider than a head (its inputs cannot be moved as stored: refused by name).

The `gated-streams` draft layer is read from the checkpoint's tensors alone: transformers ignores `mtp.*` for this
architecture, so there is no reference to score it against. Two things stand in for one. A test gives the layer
weights whose drafts are known (it passes the target's streams on, so every draft is the round's pending token), which
checks that the draft head reads the right position's feature through the right maps. And on Qwen3.8-Flash-Next the
layer's drafts are the target's tokens: 81% of a greedy reply's drafts accepted at width 3 (3.4 tokens a round), 58%
sampled at temperature 0.7. Judge another checkpoint the same way, by its acceptance (the request facts' `accepted`
beside `drafted`).

An indexer's cut between blocks that score exactly alike (several blocks at zero, where few score above it) keeps the
lower blocks, crit's canonical choice; a reference that breaks such ties another way reads other blocks there.

## Validation

The harness in `tests/test_device.py` (`Engines`, `Layout`) is also the way to test an engine's own model plugin or
layout: a class names a small random checkpoint and gets every check below. A small random model hides most
selection bugs, so the harness sharpens what a test must depend on: the output head always (a greedy choice no
rounding flips), `sharp_routes` for routed experts (a near-tie route no rounding flips) and `sharp_attention` for a
model that limits what a token attends (an indexer's blocks, a window). Without `sharp_attention` such a model
attends every position almost alike, and its scores follow the reference whatever the limit picks: an indexer that
took its lowest-scoring blocks passed every check here until the class set it.

`tests/test_plugin.py` (host): the layouts check and cover the model types the manifest needs, with the slots each
requires; the manifest's options are compose()'s defaults; spans are the longest within the budget and the room,
and the fastest where a step's fixed part passes the budget; row groups; matrices read as stored (MXFP8, NVFP4 with
compressed scales decoded). `tests/test_device.py` (transformers, Core's step library; CUDA, or the CPU where there is none): per
layout a whole engine through Core's loader against transformers' log-probabilities and greedy choices, requests
together as alone, seeded sampled requests repeatable and batched as alone, and on a GPU spans timed at load and
16-token spans scoring as whole chunks. crit's `tests/test_model_template.py` runs the same harness for the layouts
whose plugins a dense engine does not use: the router's rule checked, expert-offload in the experts slot in its
device, host-memory and file tiers (rounds counting each request's expert reads and hits), captured prompt chunks
decoding as eager ones, MTP drafting in captured rounds giving the plain greedy replies, DeepSeek V3's latent
attention and grouped sigmoid routing, and Qwen4Exp below and past its indexer's budget, its prompt in steps of other
sizes and with end tokens inside it.
