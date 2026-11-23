# streams

Several residual streams (hyper-connections) mixed by weights the streams compute themselves.

## How it works

Several residual streams. A mix of the streams s (S of them, each the model's hidden width) is

    n = rms_norm(s)                                     each stream by its own weight (the layout's norm offset)
    g = sigmoid(up(silu(down(n) / S)))                  one gate a value: the low-rank maps read every stream
    h = mean over the streams of g * n                  the block's input row
    w = 2 * sigmoid(inject(n) / S)                      one weight a stream: s += w * block(h)

A layer mixes before its mixer and before its FFN; the last mix, without a block or weights, is the row the head
reads. The streams start as S copies of the embedding row. Layout keys (beside `block`): count (the config key of
S, hc_count by default), mixer, ffn and final (the three mixes' prefixes: attn_hyper_connection.,
mlp_hyper_connection., hyper_connection_mixer.), norm, down, up and inject (their tensors' names: hc_norm,
input_mix_weight_down, input_mix_weight_up, block_inject_weight). A draft layer's feature is the streams before
the last mix, [S * hidden] a row.

## Use it

A layout names it as its `residual` (`streams`); the model plugin mounts [code/streams.py](code/streams.py) as its source `streams`. What it reads is in [entry.json](entry.json) (`reads`): the tensors after the block's prefix and the config keys, as its code names them.
