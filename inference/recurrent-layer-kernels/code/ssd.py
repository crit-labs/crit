"""Host references for the recurrent-layer kernels: the short causal convolution and Mamba-2's state-space recurrence
(token by token, and chunked as the kernels evaluate it). The gated-delta math lives in recurrent-state-recovery's
chunked_prefill.py, the oracle of the gated-delta kernels.

Mamba-2 per head, with state H[p][n], decay e^{dt A} (A < 0) and the group's B and C:

    H' = e^{dt A} H + dt x B^T          y = H' C + D x

Within a chunk with inclusive cumulative a_t = sum_{s <= t} dt_s A:

    y_t = e^{a_t} H0 C_t + sum_{s <= t} (C_t . B_s) e^{a_t - a_s} dt_s x_s + D x_t
    H_Q = e^{a_Q} H0 + sum_s e^{a_Q - a_s} dt_s x_s B_s^T

M_ts = (C_t . B_s) e^{a_t - a_s} dt_s does not depend on the state, so pass 1 builds it for every chunk in parallel
and pass 2 walks the chunks in order.
"""
import math


def conv(rows, weight, state, bias=0.0, silu=True):
    """One channel's causal convolution: (outputs, new state). weight[-1] multiplies the current input, state holds
    the previous len(weight) - 1 inputs, oldest first."""
    if len(state) != len(weight) - 1:
        raise ValueError("the state holds width - 1 inputs")
    window, out = list(state), []
    for x in rows:
        acc = bias + math.fsum(w * v for w, v in zip(weight, window + [x]))
        out.append(acc / (1 + math.exp(-acc)) if silu else acc)
        window = (window + [x])[1:]
    return out, window


def ssd_step(H, x, dt, A, B, C, D=0.0):
    """One token of one head: (new state, output)."""
    decay = math.exp(dt * A)
    H = [[decay * h + dt * xp * b for h, b in zip(row, B)] for row, xp in zip(H, x)]
    return H, [math.fsum(h * c for h, c in zip(row, C)) + D * xp for row, xp in zip(H, x)]


def ssd_sequential(H, steps, A, D=0.0):
    """Token-by-token: steps = [(x, dt, B, C)]; returns (outputs, final state)."""
    out = []
    for x, dt, B, C in steps:
        H, y = ssd_step(H, x, dt, A, B, C, D)
        out.append(y)
    return out, H


def ssd_chunked(H, steps, A, D=0.0, chunk=64):
    """The kernels' two-pass evaluation over chunks of `chunk` tokens."""
    if chunk <= 0:
        raise ValueError("positive chunk size required")
    out = []
    for c0 in range(0, len(steps), chunk):
        part = steps[c0:c0 + chunk]
        a, total = [], 0.0
        for _, dt, _, _ in part:
            total += dt * A
            a.append(total)
        n = len(part)

        def dot(u, v):
            return math.fsum(p * q for p, q in zip(u, v))
        M = [[dot(part[t][3], part[s][2]) * math.exp(a[t] - a[s]) * part[s][1] if s <= t else 0.0 for s in range(n)]
             for t in range(n)]
        for t in range(n):
            x_t, _, _, C_t = part[t]
            out.append([math.exp(a[t]) * dot(H[p], C_t) + math.fsum(M[t][s] * part[s][0][p] for s in range(t + 1)) + D * x_t[p]
                        for p in range(len(H))])
        aQ = a[-1]
        H = [[math.exp(aQ) * H[p][m] + math.fsum(math.exp(aQ - a[s]) * part[s][1] * part[s][0][p] * part[s][2][m] for s in range(n))
              for m in range(len(H[0]))] for p in range(len(H))]
    return out, H


def max_difference(a, b):
    (out_a, h_a), (out_b, h_b) = a, b
    diffs = [abs(x - y) for ra, rb in zip(out_a, out_b) for x, y in zip(ra, rb)]
    diffs += [abs(x - y) for ra, rb in zip(h_a, h_b) for x, y in zip(ra, rb)]
    return max(diffs) if diffs else 0.0
