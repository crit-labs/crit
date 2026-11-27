# shared-kv-attention

A model technique of the [gemma4-model](../../README.md) package ([a family's own pieces](../../../README.md#a-familys-own-pieces)):
full-attention layers whose values are their keys' own rows (config.json `attention_k_eq_v`), each position held
once.

Such a layer stores no `v_proj`: its values are the key projection's rows `r` through the value norm (an RMSNorm
without a weight), `v = r / rms(r)`, and its keys the same rows through the key norm and the layer's partial rotation,
`k = rope(v * w_k)`. The keys' unrotated dimensions are the values' times `w_k`; only the dimensions the rotary table
turns carry anything else. So each KV head keeps one row a position, `[v | k_rot]`, and the attention slot's latent
attention (`attention:latent`, rank `R = D`) reads it:

| | |
| --- | --- |
| query row | `[q * w_k` on the unrotated dimensions, `0` on the rotated ones `| q_rot]` |
| score | `q_lat . v + q_rot . k_rot = q . k`, the layer's own score at its own scale |
| output | the mix of the rows' first `R` values: the mix of `v`, the layer's own output |

A position then takes `kv_heads x (D + rotated)` elements where attention keeps keys and values apart, `kv_heads x 2D`
(heads 512 wide with a quarter of them rotated: 640 against 1,024, 37.5% less KV a position, which every decode step
reads).

The layout names it with `"otherwise": "attention"`: where the config does not set `attention_k_eq_v`, or the
attention plugin provides no `attention:latent`, the layer runs the library's attention, which takes its values from
the key projection's rows itself (`serves()`; the engine's facts name the layer kinds that did so, `mixers_instead`).
Its pool is one of keys only, a head a row, at the model's KV page type (static FP8 pages become FP8 pages by a
scale a row, since a static key scale does not scale values). A drafter that reads the layer's keys and values calls
`read(q, batch)`.

| File | What it holds |
| --- | --- |
| [entry.json](entry.json) | What it provides and reads |
| [code/shared_kv_attention.py](code/shared_kv_attention.py) | The block (`BLOCKS`) |

Tested by the package's engine tests ([tests/test_device.py](../../tests/test_device.py)): the same log-probabilities
and greedy choices as transformers and as the plain attention path, in both checkpoint layouts, and the smaller KV.

Promoted from the 2026-11-25 Gemma 4 build's engine (the 2026-11-24 build wrote the same), reworked to the library's
standard: its reads of the attention plugin's internals (a fused launch of every KV head over one pool, a warm-up of
the plugin's latent kernels, a free-memory threshold for torch's cache) are gone; it calls the attention slot's
`forward(..., latent=D)` a KV head.
