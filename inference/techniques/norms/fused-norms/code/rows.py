"""The norms slot's host form: the elementwise work around a transformer's matmuls over rows of numbers, in FP64.

The oracle every norms plugin's device form is tested against (crit_plugins.slots.INTERFACES["norms"]). A row is a list
of floats: a token's hidden row, or one head (or one residual stream) of one token: a token's heads are rows of their
own. Everything is computed in double precision from the inputs as given; the only rounding the contract itself
names is the rotary angle, which is an FP32 product (below), as the checkpoints' reference code computes it.

- rms_norm_rows(rows, weight, eps, offset=0.0, residual=None): y = x / sqrt(mean(x^2) + eps) * (offset + weight),
  where x is the row, or residual + row when residual rows are given (then (y, sums) is returned, sums the x rows).
  weight: one row of the rows' width, a list of G such rows (row i of the rows takes weight row i % G: the rows of G
  heads or streams in token order, each group its own weight), or None (the normalized rows). offset: families that
  store weight - 1 (a zero-centred weight) use 1.
- gated_rms_norm_rows(rows, gates, weight, eps, offset=0.0, activation="silu"): rms_norm_rows(rows, weight, eps,
  offset) times activation(gate) element by element (silu or sigmoid), as gated linear-attention outputs do.
- rope_rows(rows, positions, freqs, rotary_dim=None, style="half", scale=1.0): each row (one head) at its position,
  its first rotary_dim values rotated in pairs and the rest passed through. freqs: rotary_dim / 2 inverse
  frequencies (the angle of pair j is fp32(fp32(position) * fp32(freqs[j]))), or a table of rows indexed by position,
  each [cos of every pair, then sin of every pair] (rotary_dim values). style "half": pair (j, j + rotary_dim / 2)
  (rotate_half); "interleaved": pair (2j, 2j + 1). scale multiplies cos and sin (an attention factor).
- act_mul_rows(gates, values, activation): activation(gate) * value element by element: silu, gelu (exact, erf),
  gelu_tanh (the tanh approximation) or sigmoid.
- l2_norm_rows(rows, eps=1e-6): x / sqrt(sum(x^2) + eps).
- stream_sum_rows(streams, gates=None, activation="none", scale=1.0): per token (streams[t]: its S rows),
  scale * the sum over its streams, in stream order, of activation(gate) * row: gates[t] S rows of gates (one per
  value) or S numbers (one per stream), None for the plain sum; activation none (the gate itself) or any of
  act_mul_rows'. Several residual streams mixed into one row.
- stream_add_rows(streams, rows, weights=None): per token, every stream s plus weights[t][s] * rows[t] (1 without
  weights): one row added into every residual stream. Returns the new streams.
- softplus_decay_rows(rows, bias, rate, gates=None): -rate[i] * softplus(x[i] + bias[i]) for each value of each row
  (softplus(v) = log(1 + exp(v))): a gated delta or state-space layer's discretized log-decay (rate exp(A_log)); with
  gate rows, also their sigmoids (its beta): (decays, betas).
"""
import math
import struct

ACTIVATIONS = ("silu", "gelu", "gelu_tanh", "sigmoid")
GATES = ("silu", "sigmoid")
STREAM_GATES = ("none",) + ACTIVATIONS
STYLES = ("half", "interleaved")


def _f32(value):
    """value rounded to the nearest FP32 (ties to even), as a Python float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def activate(name, x):
    """activation(x) in FP64: silu, gelu, gelu_tanh, sigmoid, or none (x itself)."""
    if name == "silu":
        return x * _sigmoid(x)
    if name == "gelu":
        return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
    if name == "gelu_tanh":
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    if name == "sigmoid":
        return _sigmoid(x)
    if name == "none":
        return x
    raise ValueError(f"activation {name!r} is none of {', '.join(STREAM_GATES)}")


def _width(rows):
    if not rows:
        return 0
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("rows of one positive width")
    return width


def _weights(weight, width):
    """The weight as a list of G rows (G 1 for one row), or None."""
    if weight is None:
        return None
    grouped = bool(weight) and isinstance(weight[0], (list, tuple))
    groups = [list(map(float, row)) for row in weight] if grouped else [list(map(float, weight))]
    if not groups or any(len(row) != width for row in groups):
        raise ValueError(f"a weight of rows of the rows' width {width} (one, or one per group)")
    return groups


def rms_norm_rows(rows, weight, eps, offset=0.0, residual=None):
    """Each row (residual + row with residual rows) over its root mean square, times offset + weight (row i of a
    grouped weight's G rows taking row i % G)."""
    width = _width(rows)
    groups = _weights(weight, width) if rows else None
    if residual is not None:
        if len(residual) != len(rows) or _width(residual) not in (0, width):
            raise ValueError("one residual row of the rows' width per row")
        rows = [[float(r) + float(x) for r, x in zip(rr, xr)] for rr, xr in zip(residual, rows)]
    out = []
    for i, row in enumerate(rows):
        inv = 1.0 / math.sqrt(math.fsum(float(v) * float(v) for v in row) / width + eps)
        w = None if groups is None else groups[i % len(groups)]
        out.append([float(v) * inv * (1.0 if w is None else offset + w[d]) for d, v in enumerate(row)])
    return out if residual is None else (out, [list(map(float, row)) for row in rows])


def gated_rms_norm_rows(rows, gates, weight, eps, offset=0.0, activation="silu"):
    """rms_norm_rows of each row times activation(gate), element by element."""
    if activation not in GATES:
        raise ValueError(f"a gated norm's activation is one of {', '.join(GATES)}, not {activation!r}")
    if len(gates) != len(rows) or _width(gates) != _width(rows):
        raise ValueError("one gate row of the rows' width per row")
    normed = rms_norm_rows(rows, weight, eps, offset)
    return [[y * activate(activation, float(g)) for y, g in zip(row, gate)]
            for row, gate in zip(normed, gates)]


def rope_rows(rows, positions, freqs, rotary_dim=None, style="half", scale=1.0):
    """Each row rotated at its position (rotary positional embedding), the values past rotary_dim passed through."""
    if style not in STYLES:
        raise ValueError(f"style is one of {', '.join(STYLES)}, not {style!r}")
    width = _width(rows)
    if len(positions) != len(rows):
        raise ValueError("one position per row")
    table = bool(freqs) and isinstance(freqs[0], (list, tuple))
    found = len(freqs[0]) if table else 2 * len(freqs)
    rotary_dim = found if rotary_dim is None else rotary_dim
    if rotary_dim != found or rotary_dim < 2 or rotary_dim % 2 or rotary_dim > width:
        raise ValueError(f"rotary_dim {rotary_dim}: an even count of at most the row's {width} values, which the "
                         f"frequencies give ({found})")
    half = rotary_dim // 2
    out = []
    for row, position in zip(rows, positions):
        if position < 0 or (table and position >= len(freqs)):
            raise ValueError(f"position {position} is outside the {'table' if table else 'nonnegative positions'}")
        if table:
            cos = [float(c) * scale for c in freqs[position][:half]]
            sin = [float(s) * scale for s in freqs[position][half:]]
        else:
            angles = [_f32(_f32(float(position)) * _f32(float(f))) for f in freqs]
            cos = [math.cos(a) * scale for a in angles]
            sin = [math.sin(a) * scale for a in angles]
        x = [float(v) for v in row]
        y = list(x)
        for j in range(half):
            a, b = (j, j + half) if style == "half" else (2 * j, 2 * j + 1)
            y[a] = x[a] * cos[j] - x[b] * sin[j]
            y[b] = x[b] * cos[j] + x[a] * sin[j]
        out.append(y)
    return out


def act_mul_rows(gates, values, activation):
    """activation(gate) * value, element by element."""
    if len(gates) != len(values) or _width(gates) != _width(values):
        raise ValueError("gates and values of one shape")
    return [[activate(activation, float(g)) * float(v) for g, v in zip(gate, value)]
            for gate, value in zip(gates, values)]


def l2_norm_rows(rows, eps=1e-6):
    """Each row over its Euclidean norm: x / sqrt(sum(x^2) + eps)."""
    _width(rows)
    return [[float(v) / math.sqrt(math.fsum(float(u) * float(u) for u in row) + eps) for v in row] for row in rows]


def _token_streams(streams):
    if not streams:
        return 0, 0
    count, width = len(streams[0]), _width(streams[0])
    if count < 1 or any(len(token) != count or _width(token) != width for token in streams):
        raise ValueError("every token has the same streams of one width")
    return count, width


def stream_sum_rows(streams, gates=None, activation="none", scale=1.0):
    """Per token, scale * the sum over its streams in stream order of activation(gate) * stream row."""
    if activation not in STREAM_GATES:
        raise ValueError(f"activation is one of {', '.join(STREAM_GATES)}, not {activation!r}")
    count, width = _token_streams(streams)
    if gates is not None and len(gates) != len(streams):
        raise ValueError("one token of gates per token of streams")
    out = []
    for t, token in enumerate(streams):
        total = [0.0] * width
        for s, row in enumerate(token):
            if gates is None:
                factors = [1.0] * width
            else:
                gate = gates[t][s]
                values = gate if isinstance(gate, (list, tuple)) else [gate] * width
                if len(values) != width:
                    raise ValueError("a gate per value of the stream, or one per stream")
                factors = [activate(activation, float(g)) for g in values]
            total = [acc + f * float(v) for acc, f, v in zip(total, factors, row)]
        out.append([scale * v for v in total])
    return out


def stream_add_rows(streams, rows, weights=None):
    """Per token, every stream plus weights[t][s] * rows[t] (1 without weights)."""
    count, width = _token_streams(streams)
    if len(rows) != len(streams) or (rows and _width(rows) != width):
        raise ValueError("one row of the streams' width per token")
    if weights is not None and (len(weights) != len(streams) or any(len(w) != count for w in weights)):
        raise ValueError("one weight per token and stream")
    return [[[float(h) + (1.0 if weights is None else float(weights[t][s])) * float(x) for h, x in zip(row, rows[t])]
             for s, row in enumerate(token)] for t, token in enumerate(streams)]


def _softplus(v):
    return max(v, 0.0) + math.log1p(math.exp(-abs(v)))


def softplus_decay_rows(rows, bias, rate, gates=None):
    """-rate[i] * softplus(x[i] + bias[i]) per value; with gate rows, (those rows, sigmoid of every gate)."""
    width = _width(rows)
    if rows and (len(bias) != width or len(rate) != width):
        raise ValueError("a bias and a rate per value of the rows")
    decays = [[-float(r) * _softplus(float(v) + float(b)) for v, b, r in zip(row, bias, rate)] for row in rows]
    if gates is None:
        return decays
    if len(gates) != len(rows) or (rows and _width(gates) != width):
        raise ValueError("gates of the rows' shape")
    return decays, [[_sigmoid(float(g)) for g in gate] for gate in gates]


class Rows:
    """The host form as one object (a norms plugin's host methods)."""

    rms_norm_rows = staticmethod(rms_norm_rows)
    gated_rms_norm_rows = staticmethod(gated_rms_norm_rows)
    rope_rows = staticmethod(rope_rows)
    act_mul_rows = staticmethod(act_mul_rows)
    l2_norm_rows = staticmethod(l2_norm_rows)
    stream_sum_rows = staticmethod(stream_sum_rows)
    stream_add_rows = staticmethod(stream_add_rows)
    softplus_decay_rows = staticmethod(softplus_decay_rows)
