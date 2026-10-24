# Attend over a latent KV cache (MLA)

Multi-head latent attention (DeepSeek-V2/V3, Kimi K2 and other DeepSeek-architecture
models) caches one compressed latent and one rope key per token and layer, shared by every
head. Absorbing the key up-projection into the query turns each head's attention into
multi-query attention over the shared 576-wide cache rows (512 + 64 for DeepSeek-V3), and
the value up-projection applies once to the head's 512-wide latent output. This entry
supplies that absorbed attention for decode, speculative verify and prompt chunks, over a
contiguous or paged latent cache.

## When to use it

For any model whose config declares `kv_lora_rank` (Core's `ModelConfig::mla`). The latent
cache holds `rank + rope` values per token and layer instead of `heads x (qk_head + v_head)`:
for DeepSeek-V3, 576 against 40,960, which is what lets a long context fit beside offloaded
experts on one GPU. Not for grouped-query models, and not the fastest prompt path: for long
prompts, decompressing keys and values and running ordinary flash attention does less than
a third of the absorbed form's work (head dimension 192 instead of 576); keep this kernel's
prompt mode for short chunks or as the oracle.

## How it works

**Absorption.** With `q_h = [q_nope ; q_rope]`, keys `k_hj = [W_UK_h c_j ; r_j]` and values
`v_hj = W_UV_h c_j`:

```text
score_hj = scale (q_nope . W_UK_h c_j + q_rope . r_j) = scale ([W_UK_h^T q_nope ; q_rope] . [c_j ; r_j])
out_h    = W_UV_h (sum_j p_hj c_j)
```

The engine forms the absorbed queries `[W_UK_h^T q_nope ; q_rope]` (a batched GEMM over
heads) and applies `W_UV_h` to the latent outputs; the kernel attends.

**Kernel.** A block owns eight heads of one query column and one key split: warp `w` is
head `w`, lane `j` scores key `j` of each 32-key tile (rows padded so lanes reading one
column hit distinct banks), the warp keeps an online softmax, and lane `l` accumulates
latent dimensions `l + 32 i`. 32-key tiles go to splits by position (tile `i` to split
`i mod splits`), and a combine adds the splits in order with their weights
`l_b e^{m_b - M}` (and an optional sink), so a column's result depends only on its
position and the split count: a column decoded alone equals the same column in a verify.
A block table reads Core's paged cache, a null table contiguous rows.

## What the code supplies

| File | Supplies |
| --- | --- |
| [latent_attention.h](code/latent_attention.h) / [.cu](code/latent_attention.cu) | `Problem`, `enqueue` (attention plus ordered split combine), `partial_floats`; contiguous or paged BF16 cache, optional sinks |
| [reference.py](code/reference.py) | Per-head (naive) attention, the absorbed query, the kernel's latent attention and cache sizes |

## What the engine supplies

The latent cache writes (the compressed `c` after its norm, and the rope key with RoPE
applied, packed `[c ; r]`), the absorbed queries, `W_UV` after attention, the softmax
scale (`1 / sqrt(qk_nope + qk_rope)`, with any YaRN multiplier the config declares), and
Core's paged storage when caches are shared.

## Worked example

From `code/`: the absorbed form equals per-head attention over decompressed keys and
values.

```python
import random
from reference import absorbed_query, latent_attention, matvec, naive

rng = random.Random(0)
g = lambda: rng.gauss(0, 1)
heads, rank, rope, nope, vdim, L = 2, 4, 2, 3, 3, 5
latents = [[g() for _ in range(rank)] for _ in range(L)]
ropes = [[g() for _ in range(rope)] for _ in range(L)]
w_uk = [[[g() for _ in range(rank)] for _ in range(nope)] for _ in range(heads)]
w_uv = [[[g() for _ in range(rank)] for _ in range(vdim)] for _ in range(heads)]
q_nope = [[g() for _ in range(nope)] for _ in range(heads)]
q_rope = [[g() for _ in range(rope)] for _ in range(heads)]
want = naive(q_nope, q_rope, latents, ropes, w_uk, w_uv, scale=0.5)

queries = [absorbed_query(q_nope[h], w_uk[h]) + q_rope[h] for h in range(heads)]   # what the kernel reads
cache = [c + r for c, r in zip(latents, ropes)]                                    # one row per token, all heads
latent = latent_attention(queries, cache, rank, scale=0.5)
got = [matvec(w_uv[h], latent[h]) for h in range(heads)]
assert all(abs(a - b) < 1e-9 for ra, rb in zip(want, got) for a, b in zip(ra, rb))
```

## Adapting it

Preserve: tile-to-split assignment by position and the ordered combine (column invariance),
the cache row layout `[c ; r]` the queries match, and RoPE on the rope key only. Adaptable:
tensor-core tiling of scores and outputs across heads (FlashMLA's approach), FP8 latent
pages with per-tile scales, split counts from the SM count, and a decompressed prompt path.

## Costs and validation

Each split block reads its tiles of `rank + rope` values once for eight heads; scores cost
`heads x (rank + rope)` and outputs `heads x rank` multiply-adds per key, so decode with many
heads is compute bound on SIMT. Reference speed; no speed is claimed.

[latent_attention_tests.cpp](../../tests/inference/latent_attention_tests.cpp) compares
DeepSeek-V3 widths (16 heads) and a smaller geometry, contiguous and paged, with and without
sinks, three verify columns over five splits against an FP64 reference, and each column
alone bit for bit against the verify. [test_latent_attention.py](../../tests/test_latent_attention.py)
checks the absorption identity and the cache arithmetic.
