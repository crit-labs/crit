# pre-norm

One residual stream: an RMSNorm before each block, its output added, a final norm.

## How it works

One pre-norm residual stream. Layout keys (beside `block`): pre_norm and post_norm, the layer norms' names
(input_layernorm and post_attention_layernorm by default; the layout's own pre_norm / post_norm are read too).

## Use it

A layout names it as its `residual` (`pre-norm`); the model plugin mounts [code/pre_norm.py](code/pre_norm.py) as its source `pre_norm`. What it reads is in [entry.json](entry.json) (`reads`): the tensors after the block's prefix and the config keys, as its code names them.
