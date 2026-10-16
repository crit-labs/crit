"""Two-pass chunked gated-delta prefill, checked against the token-by-token recurrence in reference.py.

Within a chunk of C tokens with cumulative log decay g_t (inclusive) and normalized keys k_t and queries q_t,
the token-by-token update S_t = e^{l_t} S_{t-1} + k_t r_t^T, r_t = b_t (v_t - e^{l_t} S_{t-1}^T k_t), is solved
for the whole chunk at once. With L_ij = b_i e^{g_i - g_j} k_i . k_j (j < i) and T = (I + L)^{-1}:

    U = T diag(b) V - T diag(b e^g) K S0 = Ub - W S0          (the corrected values of the chunk)
    o_t = s (e^{g_t} q_t . S0 + sum_{i <= t} e^{g_t - g_i} (q_t . k_i) U_i)
    S_C = e^{g_C} S0 + sum_i e^{g_C - g_i} k_i U_i^T

T, W = T diag(b e^g) K, Ub = T diag(b) V and P_ti = e^{g_t - g_i} q_t . k_i do not depend on the state, so pass 1
computes them for every chunk in parallel; pass 2 walks the chunks in order with matrix products only (no
triangular solve), which suits tensor cores. The result equals the token-by-token recurrence up to rounding.
"""
import math

from reference import Geometry, Step, _advance, _normal  # noqa: F401  (Geometry/Step re-exported for callers)


def sequential(matrix, steps, geometry):
    """Token-by-token recurrence: (outputs, final matrix) with the matrix flattened as [keys][values]."""
    outputs = []
    for step in steps:
        matrix, output = _advance(tuple(matrix), step, geometry)
        outputs.append(output)
    return outputs, matrix


def prepare_chunk(steps, geometry):
    """Pass 1 for one chunk: everything that does not depend on the chunk's start state."""
    K = [_normal(s.key, geometry.epsilon, geometry.normalize_keys) for s in steps]
    Q = [_normal(s.query, geometry.epsilon, geometry.normalize_queries) for s in steps]
    beta = [s.beta for s in steps]
    g, total = [], 0.0
    for s in steps:
        total += s.log_decay
        g.append(total)
    n = len(steps)

    def dot(a, b):
        return math.fsum(x * y for x, y in zip(a, b))
    L = [[beta[i] * math.exp(g[i] - g[j]) * dot(K[i], K[j]) if j < i else 0.0 for j in range(n)] for i in range(n)]
    T = [[0.0] * n for _ in range(n)]
    for i in range(n):  # forward substitution: (I + L) T = I
        for j in range(n):
            T[i][j] = (1.0 if i == j else 0.0) - math.fsum(L[i][m] * T[m][j] for m in range(i))
    keys, values = geometry.keys, geometry.values
    W = [[math.fsum(T[i][m] * beta[m] * math.exp(g[m]) * K[m][k] for m in range(i + 1)) for k in range(keys)] for i in range(n)]
    Ub = [[math.fsum(T[i][m] * beta[m] * steps[m].value[v] for m in range(i + 1)) for v in range(values)] for i in range(n)]
    P = [[math.exp(g[t] - g[i]) * dot(Q[t], K[i]) if i <= t else 0.0 for i in range(n)] for t in range(n)]
    return {"K": K, "Q": Q, "g": g, "W": W, "Ub": Ub, "P": P}


def apply_chunk(matrix, chunk, geometry):
    """Pass 2 for one chunk: outputs and the state after it, from the chunk's start state."""
    keys, values = geometry.keys, geometry.values
    S = [list(matrix[k * values:(k + 1) * values]) for k in range(keys)]
    n = len(chunk["g"])
    U = [[chunk["Ub"][i][v] - math.fsum(chunk["W"][i][k] * S[k][v] for k in range(keys)) for v in range(values)]
         for i in range(n)]
    outputs = []
    for t in range(n):
        qs = [math.fsum(chunk["Q"][t][k] * S[k][v] for k in range(keys)) for v in range(values)]
        outputs.append(tuple(geometry.query_scale * (math.exp(chunk["g"][t]) * qs[v]
                                                     + math.fsum(chunk["P"][t][i] * U[i][v] for i in range(t + 1)))
                             for v in range(values)))
    gC = chunk["g"][-1]
    new = tuple(math.exp(gC) * S[k][v] + math.fsum(math.exp(gC - chunk["g"][i]) * chunk["K"][i][k] * U[i][v] for i in range(n))
                for k in range(keys) for v in range(values))
    return outputs, new


def two_pass(matrix, steps, geometry, chunk_size):
    """Chunk-parallel pass 1 for every chunk, then the sequential pass 2: (outputs, final matrix)."""
    if chunk_size <= 0:
        raise ValueError("positive chunk size required")
    chunks = [prepare_chunk(steps[i:i + chunk_size], geometry) for i in range(0, len(steps), chunk_size)]
    outputs = []
    for chunk in chunks:
        out, matrix = apply_chunk(matrix, chunk, geometry)
        outputs += out
    return outputs, matrix


def max_difference(a, b):
    """Largest absolute difference between two (outputs, matrix) results."""
    (out_a, m_a), (out_b, m_b) = a, b
    diffs = [abs(x - y) for oa, ob in zip(out_a, out_b) for x, y in zip(oa, ob)]
    diffs += [abs(x - y) for x, y in zip(m_a, m_b)]
    return max(diffs) if diffs else 0.0
