"""Host reference for multi-head latent attention (MLA, as in DeepSeek-V2/V3 and Kimi K2) and its absorbed form.

Each position caches one latent c (kv_lora_rank values) and one rope key r (qk_rope_head_dim values, RoPE applied),
shared by every head. Head h reads keys k_h = [W_UK_h c ; r] and values v_h = W_UV_h c, so with a query
q_h = [q_nope ; q_rope]:

    score_hj = scale (q_nope . W_UK_h c_j + q_rope . r_j)          out_h = W_UV_h sum_j p_hj c_j

Absorbing W_UK_h into the query, q_lat = W_UK_h^T q_nope, the score is scale [q_lat ; q_rope] . [c_j ; r_j]: every head
attends over the same 576-wide key rows (512 + 64 for DeepSeek-V3) and the same 512-wide values (the latent part of
the key), like multi-query attention, and W_UV_h applies once to the head's latent output. The cache holds 576
values per position and layer instead of heads x (192 + 128).
"""
import math


def matvec(m, v):
    return [math.fsum(a * b for a, b in zip(row, v)) for row in m]


def transpose(m):
    return [list(col) for col in zip(*m)]


def naive(q_nope, q_rope, latents, ropes, w_uk, w_uv, scale):
    """Per-head attention over decompressed keys and values. q_nope[h], q_rope[h]; latents[j], ropes[j];
    w_uk[h] is [nope][rank], w_uv[h] is [v][rank]. Returns out[h] (v values)."""
    out = []
    for h in range(len(q_nope)):
        keys = [matvec(w_uk[h], c) for c in latents]
        values = [matvec(w_uv[h], c) for c in latents]
        scores = [scale * (math.fsum(a * b for a, b in zip(q_nope[h], k)) + math.fsum(a * b for a, b in zip(q_rope[h], r)))
                  for k, r in zip(keys, ropes)]
        top = max(scores)
        w = [math.exp(s - top) for s in scores]
        total = math.fsum(w)
        out.append([math.fsum(p * v[d] for p, v in zip(w, values)) / total for d in range(len(values[0]))])
    return out


def absorbed_query(q_nope, w_uk_h):
    """q_lat = W_UK_h^T q_nope: the head's query in latent space."""
    return matvec(transpose(w_uk_h), q_nope)


def latent_attention(queries, cache, rank, scale):
    """What the kernel computes: queries[h] = q_lat ; q_rope (rank + rope values), cache[j] = c_j ; r_j. Returns the
    latent outputs o_lat[h] = sum_j p_hj c_j (rank values)."""
    out = []
    for q in queries:
        scores = [scale * math.fsum(a * b for a, b in zip(q, row)) for row in cache]
        top = max(scores)
        w = [math.exp(s - top) for s in scores]
        total = math.fsum(w)
        out.append([math.fsum(p * row[d] for p, row in zip(w, cache)) / total for d in range(rank)])
    return out


def cache_values_per_token(rank, rope, layers):
    """MLA's cache per token: one latent and one rope key per layer, shared by every head."""
    return layers * (rank + rope)
