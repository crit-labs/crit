"""Rotated low-bit KV reference (FP64, standard library).

R = H * diag(signs), H the normalized Walsh-Hadamard matrix (orthonormal and
symmetric), signs +-1 from a fixed seed. Keys and values are rotated before
symmetric per-group quantization; queries are rotated by the same R, and the
attention output is rotated back with R^T = diag(signs) * H:

    (R q) . (R k) = q . k                       (scores unchanged by rotation)
    R^T sum_i p_i (R v_i) = sum_i p_i v_i       (outputs unchanged by rotation)

These identities hold before quantization. Rotation can spread large channels
and reduce group maxima, but the resulting quantization error can improve or
worsen, even with smaller maxima. Compare both representations on the workload.
"""
import math
import random

GROUP = 32


def signs(dim, seed):
    if dim < 1 or dim & (dim - 1):
        raise ValueError("dimension must be a power of two")
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def fwht(x):
    """Normalized fast Walsh-Hadamard transform (its own inverse)."""
    n = len(x)
    if n < 1 or n & (n - 1):
        raise ValueError("length must be a power of two")
    y = [float(v) for v in x]
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(i, i + h):
                a, b = y[j], y[j + h]
                y[j], y[j + h] = a + b, a - b
        h *= 2
    scale = 1 / math.sqrt(n)
    return [v * scale for v in y]


def rotate(x, s):
    """R x = H (s * x)."""
    if len(x) != len(s):
        raise ValueError("vector and signs differ in length")
    return fwht([v * t for v, t in zip(x, s)])


def unrotate(y, s):
    """R^T y = s * (H y)."""
    return [v * t for v, t in zip(fwht(y), s)]


def quantize(x, bits=8, group=GROUP):
    """Symmetric per-group integers in [-(2^(bits-1) - 1), 2^(bits-1) - 1] with one
    scale per group; ties round to even (Python round)."""
    if bits not in (4, 8) or len(x) % group:
        raise ValueError("bits must be 4 or 8 and the length a multiple of the group")
    qmax = (1 << (bits - 1)) - 1
    codes, scales = [], []
    for g in range(0, len(x), group):
        block = x[g:g + group]
        amax = max(abs(v) for v in block)
        scale = amax / qmax if amax > 0 else 1.0
        scales.append(scale)
        codes.extend(max(-qmax, min(qmax, round(v / scale))) for v in block)
    return codes, scales


def dequantize(codes, scales, group=GROUP):
    return [c * scales[i // group] for i, c in enumerate(codes)]


def attention(q, keys, values):
    """Softmax(q . k_i / sqrt(d)) weighted sum of values (FP64)."""
    d = len(q)
    scores = [sum(a * b for a, b in zip(q, k)) / math.sqrt(d) for k in keys]
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = sum(weights)
    return [sum(w * v[j] for w, v in zip(weights, values)) / total for j in range(len(values[0]))]


def compressed_attention(q, keys, values, s=None, bits=8):
    """Attention over quantized keys/values; rotated by R when signs are given."""
    stored_k, stored_v = [], []
    for k, v in zip(keys, values):
        rk, rv = (rotate(k, s), rotate(v, s)) if s else (k, v)
        stored_k.append(dequantize(*quantize(rk, bits)))
        stored_v.append(dequantize(*quantize(rv, bits)))
    out = attention(rotate(q, s) if s else q, stored_k, stored_v)
    return unrotate(out, s) if s else out
