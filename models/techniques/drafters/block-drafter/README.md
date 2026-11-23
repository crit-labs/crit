# block-drafter

A drafter stored beside the checkpoint: a few bidirectional layers that fill a whole block of drafts in one pass over their own cache of the target's tapped rows, and a recurrent head that orders the block's drafts token by token, folded through the model's head.

## How it works

The drafter's config is the model config's entry the layout's draft `config` names, read under the drafter's own keys
(the draft's `keys` maps them to the config's): a decoder's standard ones (hidden_size, num_hidden_layers,
num_attention_heads, num_key_value_heads, head_dim, intermediate_size, rms_norm_eps, rope_theta) and block_size,
head_rank, tap_layers, local_layers and sliding_window. It reads the target's rows on entry to the layer applications
`tap_layers` names (counted over every pass of a looped span), then the final-normed row: the model's feature rows,
which the models core keeps for any residual that gives its running row (`row(state)`: pre-norm, sandwich,
block-attention). Per layer and per committed position its context keys and values come from a learned mix of those
rows; a block pass fills positions P .. P + block_size - 1 (P the pending token's) from the pending token's embedding
(normed by the residual's embedding norm where it has one) and `mask_embedding`, every row of the block seeing the
whole block and the cache before it (local layers their window); the recurrent head then orders the block's drafts
one at a time, each logit `cap * tanh((head(hidden) + head(out_proj(tanh(out)))) / cap)` with the second term folded
through the model's head once at load. Its tensors are read under the names in [entry.json](entry.json) (`reads`); a
checkpoint that spells them otherwise gets `names` rules in its layout. [code/block_drafter.py](code/block_drafter.py)'s docstring has the arithmetic in full.

The drafter only proposes: the target verifies every draft, so its precision (BF16 weights, FP32 sums, the
projections slot's draft encoding) changes which drafts are kept, never what a request emits.

## Use it

A layout names it as its draft block (`"draft": {"block": "block-drafter", "prefix": "<its tensors' prefix>"}`); the
model plugin mounts [code/block_drafter.py](code/block_drafter.py) as its source `block_drafter` and finds the block in
`DRAFTERS`: `BlockDrafter` answers what the models core asks a drafter technique (whether the checkpoint stores it,
the layer applications it taps, its device bytes, its spare cache positions) and makes its draft head (`BlockHead`:
the model's device draft head in Core's captured rounds, and the reader of each prompt's feature rows). What it reads
is in [entry.json](entry.json) (`reads`). On CUDA the block's attention over a span of cached positions runs
crit's kernel [row-ops](../../../../kernels/techniques/norms/row-ops/README.md) (`span_attention`), else torch's
attention with the span's mask; its FP64 reference is [code/reference.py](code/reference.py) (operations.json).
