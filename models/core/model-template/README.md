# Compose a model plugin from blocks

Write a model plugin by choosing and configuring tested blocks instead of writing the architecture and its serving
from scratch. `model-template` is the models core: a plugin that fills no slot and that every model plugin extends.
A finished model plugin ([models/cache](../../cache/README.md)) is a layout over it; an engine whose architecture
none serves builds its own the same way: a layout file, and a block class for a layer part no block computes.

## When to use it

Use it for every engine whose checkpoint is a decoder of the kinds below. The serving contracts are where
hand-written model plugins fail crit's checks most often (a prompt step past the cancel grace at the full context, a
prompt row that is not the decode row's bits, a decode row that changes with the rows beside it, memory at a step's
peak), and the core keeps all of them in one place.

Don't use it for a post-norm decoder, an encoder, or attention the attention slot does not compute; write those
parts as blocks of your own (below), or a model plugin of your own where the layer loop itself differs from the
residuals the core has.

## What it is made of

| Part | What it does | File |
| --- | --- | --- |
| Layouts | Data: which block each layer kind of the config runs (`layer_types`: full_attention, sliding_attention, linear_attention, or the config's own name of a layer type), the FFN of each layer (`auto`: a mixture of experts on the layers the config routes, `crit_models.config.routed_layers`, a SwiGLU MLP elsewhere), how the blocks join the residual (`residual`), blocks that add rows into the residual ahead of some layers (`before`), the norm offset (1 for zero-centred RMSNorms), an attention output gate, the draft layer's prefix, and the config values the family's reference assumes where config.json is silent (`defaults`); a model plugin's `layout.json` | [models/cache](../../cache/README.md) |
| Residuals | `pre-norm`: one stream, an RMSNorm before each block, the block's output added to the stream, a final norm. `streams`: several residual streams (hyper-connections); each block reads one row mixed from the streams' norms by weights the streams compute themselves (low-rank maps over all of them), its output is added to every stream by weights of its own, and a last mix gives the row the head reads. `sandwich`: one stream, each block between two norms (its normed output added to the stream), a scalar a layer, and the stream before the FFN's norm given to an FFN that reads it (`Context.residual`). `block-attention`: attention residuals over blocks of layers (each block input an attention over the blocks' summaries and the row so far by a learned query, the summaries growing a block at a time), with a looped span of layers run several times, its later passes started again from the row before the span and taking messages from the pass before through gates; each a model technique, the protocol (begin, advance, end, carry) in residual.py | [models/techniques/residuals](../../techniques/README.md), [code/residual.py](code/residual.py) |
| Blocks | The arithmetic of each layer part over the other slots' plugins: `attention` (grouped-query softmax attention: one stacked q, k, v matrix, biases, query/key norms where the checkpoint stores them, rotary positions from the config's rope table (its layer type's where the config keys its rope by layer type, the `proportional` type included), partial rotation, sliding windows, sinks, a sigmoid output gate; for families whose layer kinds differ in shape, a kind's own head width and key/value heads, a fixed scale, value norms, and values taken from the key projection's rows where a layer stores no value projection), `indexed-attention` (block-sparse attention: an indexer scores the complete blocks of positions before a token and the token reads its best ones, within a budget, and its own block; the index keys in a pool of keys only, the choice on crit's kernel sparse-indexer where it loads), `latent-attention` (multi-head latent attention in its absorbed form: one cached row a position, its latent and a rotary key every head shares, in a pool of keys only; low-rank queries multiplied into the latent space, the attention slot's `latent=`, the values' up-projection after it), `gated-delta` (the delta rule through the recurrence slot, either projection layout a checkpoint stores, the gated norm's activation by the config), `ngram-rows` (a `before` block: rows of a hashed n-gram table, held on the device or read where the checkpoint stores it, gated into every stream, with a dilated causal convolution through the recurrence slot), `swiglu` (the gated MLP, its gate SiLU or tanh-GELU by the config's activation), `moe` (the router slot's selection with the router's rule checked against the config, the experts slot over any experts plugin, expert-offload included, a shared expert and its gate), `parallel-moe` (a dense MLP and a mixture of experts side by side, each normed and summed, the router scoring the residual stream itself with a scale an expert; experts gated by GELU need an experts plugin that provides `experts:activations`), `relative-attention` (softmax attention with a relative-position bias each token computes for the positions just behind it, the attention slot's `bias=`, learned sinks a token applied through the log-sum-exp, L2-clipped queries and keys, short convolutions over keys and values through the recurrence slot, dynamic ones with taps a token where the checkpoint stores them, a gated output, and the convolutions its residual runs after the attention and the FFN); each a model technique the core's manifest mounts as a source, found by `blocks.library()`, with the block contract and the helpers they share (`Context`, `row_sum`) in blocks.py | [models/techniques](../../techniques/README.md), [code/blocks.py](code/blocks.py) |
| The draft layer | The checkpoint's multi-token-prediction layer as the device draft head of Core's captured rounds: one decoder layer of the model's own blocks in its own pools, entered and left as the residual does it (pre-norm: fc over the normed embedding and the normed target hidden state, the final norm; streams: every stream of the target through fc_hidden, the normed embedding through fc_embedding added to each, the layer's own last mix) | [code/model.py](code/model.py) `MtpHead`, [code/residual.py](code/residual.py) |
| A drafter technique | A draft block stored beside the checkpoint (the layout's draft `{block, prefix}` and the technique's own keys, a block a model technique's `DRAFTERS` gives: [block-drafter](../../techniques/drafters/block-drafter/README.md)) as the device draft head instead of an MTP layer: the core asks it whether the checkpoint stores it, the layer applications it taps (the core keeps their entry rows for any residual that gives its running row: residual.py), its bytes and spare cache positions, and makes its head (`feature_head`), which reads each prompt's feature rows (the tapped rows, then the final-normed row) as its chunks run, eager or captured | [code/model.py](code/model.py) `draft_block`, `feature_head` |
| Kernels | A block asks the model for crit's kernel by its id (`model.kernel(id)`: built for this machine with the engine's tuning, or None and the block runs its torch form); the model plugin's plugin.json declares it, else the core's: the core declares [row-ops](../../../kernels/techniques/norms/row-ops/README.md), the row operations between the techniques' products (head clips, small products, sink-gated outputs, scaled adds, summary mixes, gated joins, span attention), whose FP64 references are the techniques' operations (their operations.json) | [code/model.py](code/model.py) `kernel` |
| The serving contracts | Exact rows and prompt spans ([below](#the-serving-contracts)) | [code/rows.py](code/rows.py) |
| The step model | Core's model interface over the blocks: the placement plan of every device tensor, the pools, `step`, `commit`, `retire`, `score`, admission, Core's captured rounds (greedy and sampled, drafted), captured prompt chunks, Core's prompt cache, constrained, replayed and plain requests, info, facts and telemetry | [code/model.py](code/model.py) |
| compose() | The model plugin's object from a layout and the blocks it names; its twin resolve(), the engine's KV before any load; `model_plugin()`, the three functions a model plugin's plugin.py gives Core's loader | [code/compose.py](code/compose.py) |
| The shape | The model before its weights: layer kinds, each layer's mixer, the residual, the draft layer, the decoder's tensor prefix and the KV pools (which layers share one, heads, widths, positions, page type), the same for a load and a dry resolve | [code/shape.py](code/shape.py) |
| The checkpoint | Its safetensors files, or a GGUF file read under the same names ([below](#gguf-checkpoints)): the tensors the blocks read and the matrices as the projections and experts slots load them | [code/checkpoint.py](code/checkpoint.py), [code/gguf.py](code/gguf.py) |

The finished model plugins ([models/cache](../../cache/README.md)):

| Model plugin | Model types (config.json `model_type`) | Its tests serve, against transformers |
| --- | --- | --- |
| [llama-qwen3-model](../../cache/llama-qwen3-model/README.md) | llama, qwen2, qwen3, qwen2_moe, qwen3_moe | Llama; Qwen3 with a sliding-window layer; Qwen3-MoE; the Llama checkpoint written as GGUF |
| [qwen3-5-model](../../cache/qwen3-5-model/README.md) | qwen3_next, qwen3_5, qwen3_5_text, qwen3_5_moe, qwen3_5_moe_text | Qwen3-Next and Qwen3.5-MoE (gated delta, gated attention, a gated shared expert); its MTP layer drafts |
| [qwen4-exp-model](../../cache/qwen4-exp-model/README.md) | qwen4_exp, qwen4_exp_text | Qwen4Exp: four residual streams, gated delta layers, hashed n-gram rows, gated block-sparse attention, routed experts with a gated shared expert, its draft layer |
| [deepseek-v3-model](../../cache/deepseek-v3-model/README.md) | deepseek_v3 | DeepSeek V3: latent attention, experts routed by sigmoid scores within the best groups, shared experts |
| [gemma4-model](../../cache/gemma4-model/README.md) | gemma4, gemma4_text | Gemma 4 with its mixture-of-experts block: blocks between two norms, sliding and full layers of two head shapes, a GELU-gated MLP beside GELU-gated experts |
| [inkling2-loop-model](../../cache/inkling2-loop-model/README.md) | inkling2_loop, inkling2_loop_text | Inkling 2 loop: a middle span of layers run several times, attention residuals over blocks of layers, relative attention, capped SwiGLU experts; drafting from the block drafter stored beside the checkpoint |

## The serving contracts

**Exact rows.** Every matmul of a step goes through one function (`RowGroups.matmul`): each prompt chunk alone with
`prompt=True`, so a prompt row takes the prompt rows' arithmetic at every length, and one-token rows (decode rows, a
round's columns, draft steps, logits rows) in calls of at most `invariant_rows(projections)` rows, so each row has
the bits it has alone. Norms, rotary positions and gated products are the norms slot's. A request decodes the same
alone and beside others, and a step's decode rows never take the prompt path: with expert-offload a step's decode
rows and its prompt chunks are separate calls. The one exception is a debug request that asks `decode_via_prompt`
([decode through the prompt path](../../../docs/PLUGINS.md#decode-through-the-prompt-path), crit smoke's rounding or
fault): its decode rows run as one-token prompt chunks in step(), after the step's other decode rows, never in a
round slot (info `decode_via_prompt: true`).

**Prompt spans within the cancel grace.** Core cancels a request between steps, and crit serve retires the engine
whose cancel does not settle within its grace (one second): `serve --check` fails a candidate whose slowest prompt
step at the full context takes longer. A prompt chunk ends at its span, which Core's prompt-step budget gives
(`crit_plugins.steps.StepBudget`, through the load context's `steps`): Core times every prompt step the engine runs,
the template says which chunks a step ran, and the budget keeps, per depth, what the steps there took over what the
load's form `seconds = c + a * rows + b * rows * (position + rows)` predicted (a slower step taken at once, faster ones
averaged in), so a deep step that runs past the bound shrinks the next chunks at that depth and never the shallow
ones, whatever the load predicted. The load's calibration gives the form: the template times three prompt steps on
a sequence of its own (a short and a full `prefill_chunk` at position 0, a full one at the deepest position the
engine serves, its earlier KV written with zeros), and Core's step plan settles them as the template's decision and
fits the form and the peak bytes a row takes (`steps.calibrated`), a
[measured decision](../../../docs/PLUGINS.md#measured-decisions) `crit plugins decisions` records in the package. The
bound is Core's (`steps.seconds`), `span_seconds` a tighter one where it is lower (0.75 s by default, Core's most: three
quarters of the grace, since a step's fixed part, such as an
experts plugin's copies of a chunk's routed experts, is paid once a step and a smaller one pays it for fewer tokens;
where that part alone passes the bound, the span that prefills fastest, and the facts' `spans.over_budget` says so),
within the device's room (its free memory less the run's VRAM headroom). Where the engine asks its prompt rows below
row-exact a chunk's boundaries change its rows' bits, so the spans stay the calibration's (`adaptive` false) and the
budget only counts the steps that ran past the bound. The engine's facts carry the calibration, the ratio per depth,
the steps timed and those past the bound (`spans`). Fixed-text scores run in the same spans.

```python
from crit_plugins.steps import StepBudget
from rows import fit

# Three prompt steps timed at load: a short and a full chunk at position 0, a full chunk at the deepest position.
costs = fit({(128, 0): 0.060, (512, 0): 0.210, (512, 200_000): 0.290}, peaks={512: 96 << 20})
spans = StepBudget(512, prior=costs, seconds=0.75, deepest=200_000)
assert spans.rows(0) == 512 and spans.rows(200_000) == 512      # every step within the bound: whole chunks

# The engine's deep steps take longer than the calibration said: the budget learns it from the steps it times.
for _ in range(12):
    rows = spans.rows(200_000)
    spans.ran(rows, 200_000)
    spans.observe(0.05 + 3e-4 * rows + 1e-8 * rows * (200_000 + rows))
assert spans.rows(200_000) < 512 and spans.rows(0) == 512 and spans.facts()["over_bound"] > 0
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
passes](../../../inference/core/src/runtime/README.md#model-progress)): a step runs chunks through layers until `span_seconds` or
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
the checkpoint stores it through an embeddings plugin with `embeddings:host` (lookup-table-rows, from the files;
memory-table-rows, read once into registered host memory). A prompt or a score that ends hands torch's cached blocks
back through Core's step plan (`LoadContext.steps.release()`) when the device has less free than the next step needs
(the headroom and the step arena; the configuration's `release_below_mib` overrides it)
([step settings](../../../docs/PLUGINS.md#step-settings)).

**Drafting and batching.** The model declares `max_concurrency`, the lanes Core's step plan gives it (`steps.lanes`:
at most option `concurrency`, within crit serve's `--max-concurrency`; its waiting places likewise, option `pending`),
and sizes its states and captured rounds for them, `max_row_tokens` and
`decode_token_work` of the widest round, `prefill_work_quantum` equal to its prefill chunk, with passes
`prefill_pass_tokens` and `prefill_pass_steps`, token log-probabilities
with 20 alternatives, fixed-text and query scores, replays and Core's sampling fields; with `mtp` and a drafter in the
selection it drafts from the checkpoint's MTP layer in captured rounds (a drafter that asks the model's draft head:
`greedy-head-drafter` for the layer's chain, `mtp-decode` with an `override` for long prompt copies over it; one that
never asks it, mtp-decode at its default, is refused), decodes plainly on request, and states
`draft_bytes_per_step`. Requests decode in Core's native rounds once their prompt is in, a greedy constrained
request at width 0 where the sampler's rounds mask (token-constraints: Core stages its allowed set between rounds);
other constrained requests, long replays and rows the rounds cannot hold decode through `step()`. Its info names the round counters its rounds'
parts count (`round_counters`: an expert cache's `expert_reads` and `expert_hits`), which the round driver collects
from the parts themselves.

**Devices.** CUDA, and the CPU: there every block runs the same arithmetic over the slots' CPU forms, decoding goes
through `step()` (captured rounds and prompt chunks are a GPU's) and no span is timed. It is how the plugin's tests
run where there is no GPU, and a way to check a layout against its reference before a GPU is free.

## How to use it

An engine for a checkpoint a finished model plugin serves takes that plugin (`crit plugins list --model MODEL
--device cuda` names it; `crit plugins add` brings the core with it):

```sh
crit plugins add ENGINE/plugins qwen3-5-model attention-state softmax-attention quantized-projections \
  token-constraints device-placement fused-norms memory-table-rows sequential-recurrence
```

```json
{"plugins": {"use": {"model": "qwen3-5-model", "placement": "device-placement", "projections": "quantized-projections",
                     "attention": "softmax-attention", "kv_cache": "attention-state", "sampler": "token-constraints",
                     "norms": "fused-norms", "embeddings": "memory-table-rows", "recurrence": "sequential-recurrence"},
             "options": {"qwen3-5-model": {"context": 262144, "concurrency": 4, "prefill_chunk": 1024}}}}
```

The model plugin's manifest requires the slots its layout calls (`requires`, and `requires_when` for the model types
that route experts), so `crit plugins check` names a missing recurrence or experts slot before anything loads. A
mixture of experts takes a router and an experts plugin; the router's options must state the checkpoint's rule (the
core refuses one that does not, naming the options). The options are the core's, set under the model plugin's id.

### A new model plugin

**An architecture no finished model plugin serves.** Start with `crit models inspect MODEL`: the model plugin that
serves it if one does, the model techniques that read its config keys and tensors, and what no technique reads. The
plugin is a folder in the engine beside the core `crit plugins add ENGINE/plugins model-template` put there:

```text
ENGINE/plugins/my-model/
  plugin.json     the manifest: fills model, the model types it serves, the slots it calls, the core it extends
  layout.json     the architecture: the technique of each layer kind
  plugin.py       the core's three functions over the layout; a class for a layer part no technique computes
  tests/          a small random checkpoint served by its engine, against the reference
```

```json
{"schema": "crit-plugin-v1", "id": "my-model", "title": "The model plugin of My checkpoints",
 "fills": ["model"], "needs": [["model_type:my_model", "model_type:my_model_text"]],
 "requires": ["norms", "embeddings"], "devices": ["cuda", "cpu"],
 "sources": {"core": "../model-template"}, "extends": "core",
 "test_with": ["attention-state", "device-placement", "fused-norms", "memory-table-rows", "projections-dense",
               "softmax-attention", "token-sampling", "torch-norms"],
 "entry": "plugin.py:create"}
```

`needs` names the config model_types it serves, `requires` the slots its layout calls beyond the core's own (add
`recurrence` for linear-attention layers, `router` and `experts` for a mixture of experts; `requires_when` where only
some model types have them), `provides` `draft:head` and `draft:head-device` where the layout has a draft layer, and
`kernels` each crit kernel a block of its layout loads (indexed attention's sparse-indexer). `extends` gives it the
core's options. The layout is the format of the finished plugins' layout.json
([models/cache](../../cache/README.md)):
`mixers` by layer kind, `ffn` (with `auto`, `routed` and `dense` may name the blocks of routed and dense layers),
`residual`, `norm_offset`, `draft`, and where the family has them `embed_scale` (`sqrt_hidden`), `logit_softcap`
(the config key of a tanh cap on the logits), `routed_if` (the config key that switches the family's experts on) and
`requires` (config values the layout's blocks compute for; any other value is refused at load), `names` (rules that rename a
checkpoint's own tensor names to the ones the blocks read: [pattern, replacement] regular expressions, a third entry
shifting the number the first group matched), `head_rows` (the config key of the rows the head predicts where it
stores more) and `expert_caps` (the config keys of a capped SwiGLU's two caps)).

```python
# ENGINE/plugins/my-model/plugin.py
from pathlib import Path
from .core.template.compose import model_plugin

resolve, memory, create = model_plugin(Path(__file__).resolve().parent)        # layout.json beside it
```

Where a layer part needs arithmetic no block has, write the block in the plugin and hand it to the core: the
library's techniques import from the core's sources (`.core.KEY.KEY`, the `source` key of each technique's entry).

```python
from pathlib import Path
from .core.attention.attention import Attention
from .core.template.compose import model_plugin

class WindowedAttention(Attention):          # a block: __init__(model, layer, prefix, spec), forward(h, ctx)
    def forward(self, h, ctx):
        ...                                  # ctx.mm for every matmul, model.norms for every norm

resolve, memory, create = model_plugin(Path(__file__).resolve().parent,
                                       blocks={"windowed-attention": WindowedAttention})
```

The plugin's module-level `resolve()` (compose's `resolve`) is the model before its weights (code/shape.py's
`Shape`: layer kinds, residual, draft layer, the decoder's prefix and the KV pools, each pool's layers, heads,
widths, positions and page type), over the checkpoint's headers alone, its pools priced by the kv_cache plugin's
`position_bytes`: `crit plugins resolve` prints its `kv_bytes_per_token` in seconds. The load is the same `Shape` and
makes exactly the pools it lists, and Core refuses a load whose facts are not what its plugins resolved.

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
plugin as the finished ones' tests do: a small random checkpoint of the architecture with every part the real one
ships, scored and decoded against its reference implementation, batched against alone: `crit models tiny MODEL --out
TINY` makes such a checkpoint from the reference's own modeling code, and `crit models diff --model TINY --plugin
ENGINE/plugins/my-model` scores it through the plugin's engine and the reference. Once it passes, `crit models cache
add ENGINE/plugins/my-model` keeps the plugin, so the next engine for the architecture takes it with `crit plugins
add my-model` ([keeping a plugin](../../cache/README.md#keeping-a-plugin-a-run-built)).

## GGUF checkpoints

A GGUF file (or a split set, or a folder holding one) is read as a safetensors checkpoint is: Core derives its
config.json from its metadata (or reads the base model's beside it), and `code/gguf.py` gives the blocks its tensors
under the Hugging Face names they read. The reader knows GGUF's common convention alone: the parts every converter
names alike (`blk.N.attn_q`, `blk.N.ffn_norm`, `blk.N.ffn_gate_inp`, `token_embd`, `output_norm`, `output`), a routed
layer's experts as `ffn_{gate,up,down}_exps` stacks (or one `ffn_gate_up_exps`), and a set of generic steps. Which
general.architectures a model plugin reads, and what each one's converter did, is data in the plugin's layout: `gguf`
maps each architecture to an entry whose `parts` ({GGUF part: name under the layer}, over the convention's), `outside`
({GGUF tensor: name}; `{key}` in a name is the one layer the config key lists), `draft` (the draft block's own parts,
under the layout's draft prefix), `experts` (the routed experts' prefix) and `constants` (integer arrays a converter
moves to metadata) are its tables, and whose `transforms` name the converter's steps with every parameter they take:
which tensors, by name endings, and which config keys give their sizes. The steps are undone exactly:
`zero-centred` (norms stored 1 + w) and `negated-exp` (stored -exp(w)) in FP32; `squeezed` (a unit axis dropped);
`permuted-rotary` (query and key rows interleaved per head) and `tiled-heads` (heads tiled where the Hugging Face
layout groups them) moved back as whole rows, or along a matrix's inputs as whole blocks of its stored format, so a
quantized matrix is never decoded for it; `joined-rows` and `joined-columns` (tensors the converter wrote as one, read
as each, whole blocks of their columns), `split-rows` (one tensor written as several parts, read joined) and
`split-heads` (a matrix split by heads into a key part, each head's rows transposed, and a value part: joined back and
decoded, as latent attention reads it). The reader names no architecture and no family's part
([models/cache](../../cache/README.md) lists what each finished plugin reads, and its layout.json how); the first draft
(nextn) block after the text layers is read as the layout's draft layer where the entry names its parts (its decoder
layer's parts as any layer's; its own copies of the embedding and the head are not read: the template drafts with the
model's own). A tensor that maps to no name is refused at load, so a converter's new part fails there and not as a
wrong answer.

Quantized matrices load as stored (`gguf:<TYPE>`, for [gguf-projections](../../../inference/techniques/projections/gguf-projections/README.md)); routed
experts reach an experts plugin that takes them as stored (`experts:gguf`: expert-offload) as each expert's rows of
the file's stacks with each layer's formats, and any other experts plugin as their stored values in BF16. A table
stored in a block format (an n-gram table IQ4_NL, token rows) is read through an embeddings plugin that decodes its
rows on lookup (`embeddings:gguf`: lookup-table-rows); so are quantized token embeddings, kept as the file's rows
on the device or read in place where the plan puts them off it. The file's dense tensors and its quantized tensors
that no matrix load or table takes (the hyper-connections' maps; the token embeddings through an embeddings plugin
without `embeddings:gguf`) are decoded on the host by Core's decoders.

Tested by `tests/test_gguf.py` (every tensor of each architecture a layout reads, read back from a file written as the
converter writes it by the same layout entry, `crit_models.testing.gguf_checkpoint`) and as whole engines against transformers on GGUF copies
of the model plugins' test checkpoints (Llama in llama-qwen3-model's tests; Qwen3-MoE through expert-offload in host memory, Qwen3.5-MoE, Qwen3.5 drafting from
its MTP layer, Qwen4Exp with its draft layer, Gemma 4 with its experts and DeepSeek V3 in crit's
tests/test_model_template.py), their matrices Q8_0.

## Not covered yet

inkling2-loop-model reads its relative attention through crit's split-KV kernel (one pipeline stage; no
flash structure, prompt reader or FlashInfer takes the bias), its capped SwiGLU on experts-reference and
grouped-experts (expert-offload refuses it by name), its MXFP8 experts as their BF16 values where the experts plugin
has no MXFP8 bundles (grouped-experts: twice their bytes), and leaves out the checkpoint's draft layer (a block
drafter) and vision tower. The bias and the attention residuals' sums are torch's products, row-exact on the CPU and
not yet held to the exact rows on a GPU. The released Inkling-2-Mini loads on an RTX 5090 in 42 s at 22.6 GB (context
8,192) and decodes about 46 tokens a second greedy over grouped-experts and quantized-projections.

An indexer that scores every position's own key with weighted heads (`indexed-attention` is the block-pooled form:
unweighted heads over the mean keys of fixed blocks), a draft layer for deepseek-v3-model, latent attention's
per-head maps in a quantized format (they are plain matrices) or on a kernel (they run in torch, a token at a time
for decode rows), Gemma 4's smaller variants (layers that share another layer's keys and values, a gated per-layer
input, the double-wide MLP: gemma4-model's layout refuses a config that has them, `requires`), FP8 in blocks other than MXFP8's
(the projections slot loads FP8 per channel, per tensor and MXFP8), experts stored in other names than
`experts.N.{gate,up,down}_proj` or the fused `gate_up_proj` / `down_proj`, captured prompt chunks for a layout with
`before` blocks or a streams residual that drafts (their chunks run eager). A draft layer shipped in a GGUF of its own
(a converter's mtp-only file), a lone gemma4 or deepseek2 GGUF without its base model's config.json (Core derives no
config for those architectures yet), a Gemma 4 GGUF whose converter trimmed the token rows past the tokenizer's
vocabulary, and a gated delta net whose value heads the converter tiled in a matrix stored in a format whose blocks are
wider than a head (its inputs cannot be moved as stored: refused by name).

qwen4-exp-model's draft layer is read from the checkpoint's tensors alone: transformers ignores `mtp.*` for this
architecture, so there is no reference to score it against. Two things stand in for one. A test gives the layer
weights whose drafts are known (it passes the target's streams on, so every draft is the round's pending token), which
checks that the draft head reads the right position's feature through the right maps. And on Qwen3.8-Flash-Next the
layer's drafts are the target's tokens: 81% of a greedy reply's drafts accepted at width 3 (3.4 tokens a round), 58%
sampled at temperature 0.7. Judge another checkpoint the same way, by its acceptance (the request facts' `accepted`
beside `drafted`).

An indexer's cut between blocks that score exactly alike (several blocks at zero, where few score above it) keeps the
lower blocks, crit's canonical choice; a reference that breaks such ties another way reads other blocks there.

## Validation

The harness a model plugin's tests run on is [crit_models.testing](../python/crit_models/testing/__init__.py)
(`Engines`, `Layout`): a class names the plugin's folder and a small random checkpoint and gets every check below,
for a finished plugin and for one an engine is building alike.

```python
# ENGINE/plugins/my-model/tests/test_device.py
from pathlib import Path
from crit_models import testing as t

class MyModelTests(t.Layout):
    plugin = Path(__file__).resolve().parents[1]

    @classmethod
    def checkpoint(cls, root):
        config = t.transformers.MyConfig(vocab_size=256, hidden_size=128, num_hidden_layers=2)
        return cls.save(t.transformers.MyForCausalLM(config), root)
```

A small random model hides most
selection bugs, so the harness sharpens what a test must depend on: the output head always (a greedy choice no
rounding flips), `sharp_routes` for routed experts (a near-tie route no rounding flips) and `sharp_attention` for a
model that limits what a token attends (an indexer's blocks, a window). Without `sharp_attention` such a model
attends every position almost alike, and its scores follow the reference whatever the limit picks: an indexer that
took its lowest-scoring blocks passed every check here until the class set it.

`tests/test_plugin.py` (host): the layout of every model plugin that extends the core checks; the manifest's options
are compose()'s defaults; spans are the longest within the budget and the room,
and the fastest where a step's fixed part passes the budget; row groups; matrices read as stored (MXFP8, NVFP4 with
compressed scales decoded). Each finished model plugin's `tests/test_device.py` (transformers, Core's step library;
CUDA, or the CPU where there is none): a whole engine through Core's loader against transformers' log-probabilities
and greedy choices, requests
together as alone, seeded sampled requests repeatable and batched as alone, and on a GPU spans timed at load and
16-token spans scoring as whole chunks. crit's `tests/test_model_template.py` varies those classes over the library
plugins a model plugin's own tests do not use: expert-offload in the experts slot in its
device, host-memory and file tiers (rounds counting each request's expert reads and hits), captured prompt chunks
decoding as eager ones, MTP drafting in captured rounds giving the plain greedy replies, DeepSeek V3's latent
attention and grouped sigmoid routing, and Qwen4Exp below and past its indexer's budget, its prompt in steps of other
sizes and with end tokens inside it.
