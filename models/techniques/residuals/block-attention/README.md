# block-attention

Attention residuals over blocks of layers, with a looped span of layers run several times, its later passes restarted and gated by the last pass's messages.

## How it works

Attention residuals over blocks of layers, with the layers of a looped span run several times (layer passes).

The layers are grouped in blocks of `block` (the config's n_layers_per_attn_res_block): at a block's first layer
the row so far becomes one more block summary, and the layer's own sum starts again from its attention's output.
Each block input is an attention over the summaries and the row so far, all normed without a weight, by a learned
query (FP32):

    mix(blocks, h, query) = softmax(query . rms(s) / sqrt(hidden)) . rms(s)   s: the summaries, then h

    q_in = k_in = v_in = mix(blocks, h, attention_query)     (layer 0: h; a full layer has three queries)
    a = mixer(norm(q_in), norm(k_in), norm(v_in)); a = conv_after_attention(a)
    r = a at a block's first layer (h joins the summaries), else h + a
    y = conv_after_ffn(ffn(norm(mix(blocks, r, ffn_query))));  the layer's row: r + y

A looped span's later passes start again from the row and summaries before the span, and each of its layers
takes messages from the pass before: one attention over that pass's own summaries and last row by four learned
queries a layer (the query, key, value and FFN messages), each joined to its input by a gate,

    gate(x, m) = (1 - tanh(forget(g))) * x + tanh(gain(g)) * m,   g = down([rms(x), rms(m)])

The last row is the head's input: mix(blocks, h, final query), then the final norm. Each pass of a looped layer is
a layer of its own here (its own KV and convolution windows: shape.py's layer applications). Tensor names are
transformers' (a layout's names rules rename a checkpoint's own); the mixer is a relative-attention block, whose
post convolutions the residual runs. Layout keys (beside `block`): embed_norm (the embedding's norm, after the
lookup), block (the config key of the block size).

## Use it

A layout names it as its `residual` (`block-attention`); the model plugin mounts [code/block_attention.py](code/block_attention.py) as its source `block_attention`, beside the sources it imports (attention). What it reads is in [entry.json](entry.json) (`reads`): the tensors after the block's prefix and the config keys, as its code names them.
