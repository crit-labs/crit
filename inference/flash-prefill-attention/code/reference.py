"""Host reference for causal grouped-query prompt attention and its flash-style (online softmax) evaluation.

`attention` is the direct definition: for query t at absolute position pos0 + t, softmax over the visible keys
0 .. pos0 + t of scale * q . k, then the weighted sum of values. `flash_attention` computes the same thing one
key tile at a time with a running row maximum m and sum l, never holding a full row of scores:

    m' = max(m, max_j s_j)      c = exp(m - m')      p_j = exp(s_j - m')
    l' = l * c + sum_j p_j      o' = o * c + sum_j p_j v_j      out = o / l   (after the last tile)

`round_p` applies a rounding function to each p_j before both the sum and the product, as a kernel does when it
feeds BF16 probabilities to a tensor core; using the rounded value in l as well keeps the weights normalized.
"""
import math
import struct


def bf16(x):
    """Round a float to the nearest BF16 value (ties to even)."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _check(q, k, v, pos0, heads, kv_heads):
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads:
        raise ValueError("heads must be a positive multiple of kv_heads")
    if pos0 < 0 or len(k) != pos0 + len(q) or len(v) != len(k):
        raise ValueError("the cache must hold pos0 + T positions of keys and values")


def attention(q, k, v, pos0, heads, kv_heads, scale):
    """q[t][h] and k[j][g], v[j][g] are head vectors; returns out[t][h]."""
    _check(q, k, v, pos0, heads, kv_heads)
    group = heads // kv_heads
    out = []
    for t, row in enumerate(q):
        heads_out = []
        for h, qv in enumerate(row):
            g = h // group
            scores = [scale * sum(a * b for a, b in zip(qv, k[j][g])) for j in range(pos0 + t + 1)]
            top = max(scores)
            weights = [math.exp(s - top) for s in scores]
            total = sum(weights)
            dim = len(v[0][g])
            heads_out.append([sum(w * v[j][g][d] for j, w in enumerate(weights)) / total for d in range(dim)])
        out.append(heads_out)
    return out


def flash_attention(q, k, v, pos0, heads, kv_heads, scale, key_tile=32, round_p=None):
    """The same attention, one key tile at a time with an online softmax."""
    _check(q, k, v, pos0, heads, kv_heads)
    group = heads // kv_heads
    out = []
    for t, row in enumerate(q):
        heads_out = []
        for h, qv in enumerate(row):
            g = h // group
            visible = pos0 + t + 1
            dim = len(v[0][g])
            m, total, acc = -math.inf, 0.0, [0.0] * dim
            for start in range(0, visible, key_tile):
                keys = range(start, min(visible, start + key_tile))
                scores = [scale * sum(a * b for a, b in zip(qv, k[j][g])) for j in keys]
                new_m = max(m, max(scores))
                correction = math.exp(m - new_m) if m != -math.inf else 0.0
                probabilities = [math.exp(s - new_m) for s in scores]
                if round_p:
                    probabilities = [round_p(p) for p in probabilities]
                total = total * correction + sum(probabilities)
                acc = [a * correction + sum(p * v[j][g][d] for p, j in zip(probabilities, keys)) for d, a in enumerate(acc)]
                m = new_m
            heads_out.append([a / total for a in acc])
        out.append(heads_out)
    return out


def max_difference(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for ha, hb in zip(ra, rb) for x, y in zip(ha, hb))
