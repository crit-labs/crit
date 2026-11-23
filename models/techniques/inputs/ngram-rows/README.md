# ngram-rows

Rows of a hashed n-gram table gated into the residual ahead of a layer, with a dilated causal convolution.

## How it works

Rows of a hashed n-gram table gated into every residual stream ahead of a layer (a `before` block over a
streams residual).

A token's n-grams, of 2 .. size tokens ending at it, are hashed to table rows, several heads an n-gram: the tokens
times one odd multiplier each, XORed, modulo each head's own prime size, at the head's offset in one table. An
n-gram never reaches past the end token before it: the tokens from that one back count as end tokens, as those
before the sequence do. The rows side by side are the token's embedding e, and with n the streams' norms

    key = rms_norm(key_proj(e))           one a stream            value = value_proj(e)
    g = key . rms_norm(stream) / sqrt(hidden),  then sign(g) * sqrt(max(|g|, 1e-6))
    v = sigmoid(g) * value                one a stream
    stream += v + silu(conv(rms_norm(v)))  a causal depthwise convolution over the token's sequence, dilated by
                                           the n-gram size (the recurrence slot's, its window a pool the block
                                           declares)

The table (hundreds of millions of rows where the n-gram vocabulary is large) is planned as a lookup table: on
the device where the placement plugin puts it there, else read where the checkpoint stores it (its shards'
files) through an embeddings plugin that provides embeddings:host; stored NVFP4 rows (their global scale
weight_scale_2) and FP8 rows (E4M3 codes with one multiplier for the table, weight_scale) are decoded on lookup. The
multipliers, sizes and offsets are the checkpoint's where it stores them, else derived as its reference derives
them. Config keys (the spec's `config` renames them): ngram_size, heads_per_ngram, ngram_vocab_size_base,
ple_embed_dim (the hidden width by default), ple_conv_kernel_size, seed, ple_layer_ids (the layers, for the
table's turn among a model's tables), and eos_token_id. Spec: embedding (ple_embedding.) and table
(ngram_embedding.), the tensors' prefixes under the block's.

## Use it

A layout names it as a mixer, an FFN or a `before` block (`ngram-rows`); the model plugin mounts [code/ngram_rows.py](code/ngram_rows.py) as its source `ngram_rows`. What it reads is in [entry.json](entry.json) (`reads`): the tensors after the block's prefix and the config keys, as its code names them.
