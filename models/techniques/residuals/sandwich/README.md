# sandwich

One residual stream, each block between two norms, a scalar a layer.

## How it works

One residual stream, each block between two norms:

    x += norm(mixer(norm(x)))        input_layernorm, post_attention_layernorm
    x += norm(ffn(norm(x)))          pre_feedforward_layernorm, post_feedforward_layernorm
    x *= layer_scalar                where the layer stores one

The stream before the FFN's norm is the context's `residual` while the FFN runs, for an FFN that reads it beside
its normed input (a router that scores the stream itself). Layout keys (beside `block`): norms, the four norms'
names in that order, and scalar, the scalar's name (layer_scalar).

## Use it

A layout names it as its `residual` (`sandwich`); the model plugin mounts [code/sandwich.py](code/sandwich.py) as its source `sandwich`. What it reads is in [entry.json](entry.json) (`reads`): the tensors after the block's prefix and the config keys, as its code names them.
