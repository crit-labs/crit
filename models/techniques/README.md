# Model techniques

The parts a model plugin composes an architecture from, one entry each, as Inference and Kernels keep theirs. A
technique is code a model plugin mounts as a source under the entry's `source` key (`code/KEY.py`), which gives the
blocks (`BLOCKS`), residuals (`RESIDUALS`) or draft blocks (`DRAFTERS`: a drafter stored beside the checkpoint, which
a layout's `draft` names beside the core's own `mtp`) a layout names. A technique's arithmetic on a device comes from
crit's kernel templates (`model.kernel(id)`: [row-ops](../../kernels/techniques/norms/row-ops/README.md) for the row
operations between its products, its knobs tuned by `crit kernels sweep`) with its own torch form as the fallback; an
operation a kernel computes for it is in its `operations.json` with its FP64 reference (`code/reference.py`). Its `entry.json` says what it provides, the sources it
imports beside it (`needs_sources`) and what it reads (`reads`: its tensor names, config keys and layout fields, as
its code names them: [what a technique reads](#what-a-technique-reads)), whose tensors and config keys
`crit models inspect` matches against a checkpoint.

| Category | Technique | Provides | What it computes |
| --- | --- | --- | --- |
| drafters | [block-drafter](drafters/block-drafter/README.md) | drafter: `block-drafter` | A drafter stored beside the checkpoint: a few bidirectional layers that fill a whole block of drafts in one pass over their own cache of the target's tapped rows, and a recurrent head that orders the block's drafts |
| ffn | [moe](ffn/moe/README.md) | block: `moe`, `parallel-moe` | A mixture of experts over the router and experts slots (any experts plugin, expert-offload included), a shared expert and its gate; parallel-moe: a dense MLP and the experts side by side |
| ffn | [swiglu](ffn/swiglu/README.md) | block: `swiglu` | The gated MLP: SiLU or tanh-GELU gate by the config's activation |
| inputs | [ngram-rows](inputs/ngram-rows/README.md) | block: `ngram-rows` | Rows of a hashed n-gram table gated into the residual ahead of a layer, with a dilated causal convolution |
| mixers | [attention](mixers/attention/README.md) | block: `attention` | Grouped-query softmax attention: a stacked q, k, v projection, biases, query/key norms, rotary positions (per layer type, partial rotation), sliding windows, sinks and a sigmoid output gate, through the attention slot |
| mixers | [gated-delta](mixers/gated-delta/README.md) | block: `gated-delta` | Gated delta-rule linear attention through the recurrence slot, either projection layout a checkpoint stores |
| mixers | [indexed-attention](mixers/indexed-attention/README.md) | block: `indexed-attention` | Block-sparse attention: an indexer scores the complete blocks of positions before a token and the token reads its best ones within a budget, and its own block |
| mixers | [latent-attention](mixers/latent-attention/README.md) | block: `latent-attention` | Multi-head latent attention in its absorbed form: one cached row a position (a latent and a shared rotary key), low-rank queries multiplied into the latent space |
| mixers | [relative-attention](mixers/relative-attention/README.md) | block: `relative-attention` | Softmax attention with a relative-position bias a token computes for the positions behind it, learned sinks a token, L2-clipped queries and keys, short and dynamic convolutions over keys and values, a gated output |
| residuals | [block-attention](residuals/block-attention/README.md) | residual: `block-attention` | Attention residuals over blocks of layers, with a looped span of layers run several times, its later passes restarted and gated by the last pass's messages |
| residuals | [pre-norm](residuals/pre-norm/README.md) | residual: `pre-norm` | One residual stream: an RMSNorm before each block, its output added, a final norm |
| residuals | [sandwich](residuals/sandwich/README.md) | residual: `sandwich` | One residual stream, each block between two norms, a scalar a layer |
| residuals | [streams](residuals/streams/README.md) | residual: `streams` | Several residual streams (hyper-connections) mixed by weights the streams compute themselves |

A technique holds no model's name and is built so any model's layout can compose it, even where one model uses it
today: it reads its tensors and config keys under its own names through the layout, and a model plugin that composes
it serves whole engines of tiny random checkpoints against the reference in two checkpoint layouts
([model-neutral code](../../docs/CONTRIBUTION_CRITERIA.md#model-neutral-code), `tests/test_model_neutral.py`). What one family arranges and spells its own way (which blocks its layers run, its
tensor names, the config values its reference assumes) is that family's layout: its finished model plugin's in
[models/cache](../cache/README.md), or the one an engine builds.

## What a technique reads

`entry.json`'s `reads` lists, each sorted, what the technique's code reads and nothing else:

- `tensors`: each tensor name its code reads, from where its own spelling starts: after the prefix its caller gives
  (the block's; a layer's or the decoder's for a residual, the draft prefix for a drafter), or after a part a layout
  field without a default gives. A layout field read with a default is spelled as that default (`experts.N.gate_proj`
  under moe's `experts`), a layer's or an expert's number as `N`.
- `config`: each config.json key its code reads by name (`model.setting(KEY)`, `model.config.get(KEY)`, the layout's
  `defaults` by key, the default of a layout field that names a config key); a key inside an entry it reads as
  `ENTRY.KEY` (`rope_parameters.factor`), inside an entry a layout field names as `*.KEY` (block-drafter's
  `*.block_size`). What it reads through the parsed config's typed fields (`config.moe`, `config.mla`,
  `config.linear`, `config.sliding_window`) is the config reader's, and not listed.
- `layout`: each layout field its code reads (its spec's and the layout's own), with the fields of its spec a
  models-core helper it hands the spec to reads (`kv_shape`: `head_dim`, `kv_heads`); the fields the core reads or
  sets on every block (`block`, `prefix`, `kind`) aside.

[tests/test_technique_reads.py](../../tests/test_technique_reads.py) derives the three lists from the code (its ast:
constants, loops and comprehensions, calls within the technique's own files, closures, attributes set on self) and
fails where an entry lacks a read or declares one its code never makes; `python tests/test_technique_reads.py --write`
writes them from the code. A class built on another technique's (`needs_sources`) reads that one's names under that
one's entry. `crit models inspect` matches `tensors` as suffixes of a checkpoint's names (after its layout's `names`
rules, `N` any number) and the top-level `config` keys against its config; `layout` it never matches, as a checkpoint
has no layout fields.

### Tensor vocabulary

A technique's tensor names are the transformers spelling of its layer kind where transformers has one, and its own
where it has none; a layout maps any other spelling onto them with `names` rules, and `reads` declares them. A
spelling one family brought is the vocabulary where it is transformers' name for that layer kind: what a technique
must not do is take one checkpoint's names where transformers spells the kind otherwise. Checked against
transformers 5.23:

- Shared conventions: attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm`, `sinks`), swiglu, moe's
  routed and shared experts (`gate`, `experts.*`, `gate_up_proj`, `e_score_correction_bias`: about 30 of its models),
  latent-attention (`q_a_proj`, `q_a_layernorm`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`: 17
  of its models spell multi-head latent attention so), gated-delta's `A_log`, `dt_bias`, `conv1d` and `norm` (the
  state-space layers' convention), pre-norm's and sandwich's layer norms.
- transformers' only spelling of the layer kind, which one family brought: parallel-moe (`router.proj`,
  `router.scale`, `router.per_expert_scale`, its paired feed-forward norms), moe's `shared_expert_gate`, sandwich's
  `layer_scalar`, indexed-attention (`indexer.index_qk_proj`, its query and key layernorms), ngram-rows, streams,
  block-attention's block queries (`attention_input`, `mlp_input`, `final_input`, `q_input_layernorm`, ...).
- Its own, where transformers spells none: block-drafter, block-attention's looped-pass messages (`message_input.*`,
  `*_message_gate.*`), the draft layers' `fc` and `pre_fc_norm_*` (pre-norm, streams).
- Not held to the rule yet: gated-delta's projections are one family line's (`in_proj_qkvz` and `in_proj_ba`, or
  `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`), while transformers also spells gated delta-rule layers as
  separate `q_proj`, `k_proj`, `v_proj`, `a_proj`, `b_proj`, `g_proj`, `o_proj`, which no rename reaches (the
  technique reads stacked matrices). relative-attention's names (`q_proj.linear`, `relative_position_bias.*`,
  `o_proj.gate_proj`, `k_proj.sconv.*`, `q_proj.norm.scale`) are its own, while transformers spells that layer kind
  `q_proj`, `r_proj`, `rel_logits_proj`, `gate_proj`, `k_sconv`, `v_sconv`, `q_l2_clip` and `rel_l2_clip`; renaming
  them changes its code and its layout's `names` rules together.

## Adding a technique

A folder `CATEGORY/ID/` with `entry.json` (schema `crit-model-technique-v1`), `README.md`, `code/KEY.py` giving
`BLOCKS`, `RESIDUALS` or `DRAFTERS`, and `tests/`. A draft block (`DRAFTERS`) is a class the core asks
`stored(checkpoint, prefix)`, then makes as `Class(model, prefix, spec)` (spec: the layout's draft entry) while the model loads: its `name` (the facts'
drafter_kind), `taps(applications)` (the layer applications whose entry rows it reads, which the core keeps for any residual that gives its running row: `row(state)`),
`spare` (cache positions its pass writes past the pending token), `weight_bytes(ck, vocabulary)`,
`device_bytes(ck, vocabulary, rows, tokens)` and `head(taps, rows, widest, tokens, head_matrix)`: the model's device
draft head, whose `prompt(rows, features)` reads each prompt's feature rows. Its code imports the core's helpers as `from ..template.blocks import Context,
row_sum, _tensor` and another technique's class as `from ..KEY.KEY import Class` (that key listed in
`needs_sources`). The model plugin mounts it by adding `"KEY": "<path to its code>"` to its manifest's `sources`;
`blocks.library()` then finds it. Its tensor names follow the [tensor vocabulary](#tensor-vocabulary) rule, and its
entry's `reads` comes from its code: `python tests/test_technique_reads.py --write` writes it, the test holds it.
