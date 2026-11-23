# latent-attention

Multi-head latent attention in its absorbed form: one cached row a position (a latent and a shared rotary key), low-rank queries multiplied into the latent space.

## How it works

Multi-head latent attention: every position caches one row, its latent (rank R) and one rotary key (width P)
that every head shares, and the latent is the value too.

    q      = q_b(rms_norm(q_a(h)))  (or q(h))     per head [N + P]: a part without positions, a rotary part
    c, k   = kv_a(h)                               c = rms_norm of its first R values, k the rest, rotated
    key_h  = [U_h c | k],  value_h = V_h c        kv_b's rows per head: U_h [N][R], V_h [values][R]
    out_h  = softmax(scale * q_h . key_h) values, then o_proj over the heads' outputs

run in its absorbed form: the query's first part is multiplied into the latent space (U_h^T q_h), the attention
slot attends the cached rows with latent=R and returns each head's mix of latents, and V_h maps it to the head's
values; no key or value per head is ever stored. The layers keep no KV in the model's pools: one pool of keys
only, a row [R + P] a position (`pools`). The per-head maps U_h and V_h are plain matrices (the checkpoint's
kv_b_proj, 16-bit or FP32), multiplied a token at a time for one-token rows and a chunk at a time for a prompt
chunk, so a decode row's bits are its own. Rotary pairs are interleaved where the config says so
(rope_interleave; the spec's `interleave` renames the key), and the scale is (N + P) ** -0.5, times the squared
YaRN magnitude scale where the config's rope parameters give mscale_all_dim. Config: kv_lora_rank,
qk_rope_head_dim, qk_nope_head_dim, v_head_dim, q_lora_rank (none: one q_proj).

## Use it

A layout names it as a mixer, an FFN or a `before` block (`latent-attention`); the model plugin mounts [code/latent_attention.py](code/latent_attention.py) as its source `latent_attention`. What it reads is in [entry.json](entry.json) (`reads`): the tensors after the block's prefix and the config keys, as its code names them.
