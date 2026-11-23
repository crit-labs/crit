# attention

Grouped-query softmax attention: a stacked q, k, v projection, biases, query/key norms, rotary positions (per layer type, partial rotation), sliding windows, sinks and a sigmoid output gate, through the attention slot.

## How it works

Softmax attention over the kv_cache slot's paged pool, grouped query heads: one stacked q|k|v matrix (biases
where the checkpoint stores them), per-head query and key RMSNorms where it stores them, rotary positions (the
config's rope table, its layer type's where the config keys its rope by layer type; partial rotation), a sliding
window (the layer kind sliding_attention: the config's sliding_window, or the spec's `window` key), attention
sinks where the checkpoint stores `sinks`, and with the spec's `output_gate: sigmoid` a gated output (each head's
query and gate from one projection, the output times sigmoid of the gate). Spec keys for families whose layer
kinds differ in shape: `head_dim` and `kv_heads` name the config keys of this kind's head width and key/value
heads (a config's own per-layer entry first, the model's by default), `scale` is the attention's scale where it is not head_dim ** -0.5, and
`value_norm: true` norms each value head (an RMSNorm without a weight). A layer that stores no v_proj takes its
values from the key projection's rows, before the keys' norm and rotation.

## Use it

A layout names it as a mixer, an FFN or a `before` block (`attention`); the model plugin mounts [code/attention.py](code/attention.py) as its source `attention`. What it reads is in [entry.json](entry.json) (`reads`): the tensors after the block's prefix and the config keys, as its code names them.
