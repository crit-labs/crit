# gated-delta

Gated delta-rule linear attention through the recurrence slot, either projection layout a checkpoint stores.

## How it works

A gated delta-rule layer (linear attention) over the recurrence slot: q, k and v through the slot's causal
depthwise convolution, the decay -exp(A_log) * softplus(a + dt_bias) and beta = sigmoid(b) from the norms slot, the
delta rule in the recurrence slot's pool, the gated RMSNorm over SiLU(z) (or, with the spec's `norm_gate` naming a
config key, that key's activation: silu or sigmoid), the output projection. The checkpoint's projections are read
in either layout it stores: one grouped in_proj_qkvz and in_proj_ba (per key head: its q, k, then its value heads'
v and z; its value heads' b, then a) or separate in_proj_qkv, in_proj_z, in_proj_b and in_proj_a.

## Use it

A layout names it as a mixer, an FFN or a `before` block (`gated-delta`); the model plugin mounts [code/gated_delta.py](code/gated_delta.py) as its source `gated_delta`. What it reads is in [entry.json](entry.json) (`reads`): the tensors after the block's prefix and the config keys, as its code names them.
