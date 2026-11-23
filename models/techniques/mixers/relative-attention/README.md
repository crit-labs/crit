# relative-attention

Softmax attention with a relative-position bias a token computes for the positions behind it, learned sinks a token, L2-clipped queries and keys, short and dynamic convolutions over keys and values, a gated output.

## How it works

Softmax attention with a relative-position bias a token computes for itself, learned sinks a token, short
convolutions over the keys and values, and a gated output. Its rows: queries, keys and values may each come from
rows of their own (forward((q_in, k_in, v_in), ctx), each normed by the residual; a single row for all three):

    q = clip(q_proj(q_in) * q_scale)           each head shortened to the config's qk_l2_clip_threshold
    k = clip(conv(k_proj(k_in)))               the convolution: (taps over the last `kernel` rows + the row) /
    v = rms_norm(conv(v_proj(v_in)))           sqrt(kernel + 1); with stored static taps and a projection of
                                               the input (dynamic), each token's taps are the static ones plus
                                               its own, one set a key/value head, / sqrt(2 * kernel + 1)
    r = clip(relative_proj(q_in) * r_scale)    each head's relative state, d_rel values, clipped to the config's
    bias[d] = r . proj[:, d]                     rel_l2_clip_threshold: its score for the position d behind it
    out = sigmoid(lse - sink) * attention(q, k, v, bias)    sink = sink_proj(q_in), a logit with no value
    y = o_proj(out * sigmoid(gate_proj(q_in)))

With `log_scaling` (the full layers) the queries and the relative states are scaled after their clip by
1 + alpha * log(max((position + 1) / floor, 1)) (the config's log_scaling_alpha and log_scaling_n_floor). The
bias reaches `extent` positions back: the sliding window, or the config's rel_extent. The block also holds the
convolutions its layer's residual runs after the attention and after the FFN (`post_convolution(name, y, ctx)`):
each convolution keeps its window in a pool the block declares (the recurrence slot's convolve), one a layer, so a
looped layer's passes keep their own. Tensor names are transformers'; spec keys: heads, kv_heads and head_dim (the
config keys of this kind's shape), log_scaling, post (the residual's convolutions: {name: tensor prefix under the
layer}).

## Use it

A layout names it as a mixer, an FFN or a `before` block (`relative-attention`); the model plugin mounts [code/relative_attention.py](code/relative_attention.py) as its source `relative_attention`. What it reads is in [entry.json](entry.json) (`reads`): the tensors after the block's prefix and the config keys, as its code names them.
