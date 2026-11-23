"""The norms slot's device form in plain torch ops (crit_plugins.device.DEVICE_INTERFACES["norms"]): the base plugin
torch-norms, and fused-norms' fallback where crit's kernel fused-norms cannot be built.

Every operation reads its inputs in their own dtype (BF16, FP16 or FP32), computes in FP32 and rounds once, to
out_dtype (by default the dtype of the tensor it normalizes or gates). Each row's bits are independent of the rows
beside it: the elementwise steps are, and every reduction goes through torch.nn.functional.rms_norm, which reduces one
row at a time (torch's mean and sum over the last dimension do not: on a GPU a row of a few-row call sums in another
order than the same row in a long call). On the CPU, torch's sigmoid and SiLU compute a call's last few values by
another formula than the rest, so which values those are moves with the row count: there they are composed from exp,
add and divide, which do not (_sigmoid, _silu). Nothing synchronizes with the host or depends on the values, so a CUDA
graph captures each call; results are new tensors unless `out` is given (then written there and returned).
"""
import math

STYLES = ("half", "interleaved")
LAYOUTS = ("halves", "interleaved")
GATES = ("silu", "sigmoid")
APPROXIMATIONS = ("none", "tanh")


def _torch():
    import torch
    return torch


def _sigmoid(t):
    """sigmoid of an FP32 tensor: torch's on CUDA, 1 / (1 + exp(-t)) on the CPU (row-invariant there)."""
    torch = _torch()
    if t.device.type == "cuda":
        return torch.sigmoid(t)
    return 1.0 / (1.0 + torch.exp(-t))


def _silu(t):
    torch = _torch()
    return torch.nn.functional.silu(t) if t.device.type == "cuda" else t * _sigmoid(t)


def _gelu(t, approximate="none"):
    torch = _torch()
    if approximate == "tanh" and t.device.type != "cuda":
        return t * _sigmoid(1.5957691216057308 * (t + 0.044715 * t * t * t))   # 0.5 t (1 + tanh(u)) = t sigmoid(2u)
    return torch.nn.functional.gelu(t, approximate=approximate)


def _deliver(y, out):
    if out is None:
        return y
    if tuple(out.shape) != tuple(y.shape) or out.dtype != y.dtype:
        raise ValueError(f"out is {tuple(out.shape)} {out.dtype}; the result is {tuple(y.shape)} {y.dtype}")
    if out.dim() and out.shape[-1] > 1 and out.stride(-1) != 1:
        raise ValueError("out's last dimension is contiguous")
    out.copy_(y)
    return out


def _normalized(x, eps, weight=None, offset=0.0):
    """x in FP32 over each last-dimension row's root mean square, times offset + weight: weight [dim], or
    [groups][dim] for the rows at index g of x's second-to-last dimension (groups entries)."""
    torch = _torch()
    y = torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), eps=float(eps))
    if weight is None:
        return y
    grouped = weight.dim() == 2 and x.dim() >= 2 and tuple(weight.shape) == tuple(x.shape[-2:])
    if weight.shape != (x.shape[-1],) and not grouped:
        raise ValueError(f"a weight of shape {tuple(weight.shape)} for rows of shape {tuple(x.shape)}: [dim], or "
                         "[groups][dim] with groups x's second-to-last dimension")
    factor = weight.float()
    return y * (factor + offset if offset else factor)


def rms_norm(x, weight, eps, offset=0.0, residual=None, out_dtype=None, out=None):
    """y = x / sqrt(mean(x^2) + eps) * (offset + weight) per last-dimension row; with residual, the residual first
    becomes residual + x in place (in its own dtype) and its rows are normalized: (y, residual)."""
    if residual is not None:
        if residual.shape != x.shape:
            raise ValueError(f"a residual of shape {tuple(residual.shape)} for rows of shape {tuple(x.shape)}")
        residual.add_(x)
        x = residual
    y = _deliver(_normalized(x, eps, weight, offset).to(out_dtype or x.dtype), out)
    return y if residual is None else (y, residual)


def rms_norm_heads(x, weight, eps, offset=0.0, out_dtype=None, out=None):
    """rms_norm of each head ([tokens][heads][dim], any strides), weight [dim]."""
    return _deliver(_normalized(x, eps, weight, offset).to(out_dtype or x.dtype), out)


def _angles(positions, freqs, rotary_dim, scale):
    """(cos, sin) [tokens][rotary_dim / 2] FP32 for each token's position."""
    if freqs.dim() == 1:
        found = 2 * freqs.shape[0]
    elif freqs.dim() == 2:
        found = freqs.shape[1]
    else:
        raise ValueError("freqs is [rotary_dim / 2] inverse frequencies or a [positions][rotary_dim] cos|sin table")
    rotary_dim = found if rotary_dim is None else int(rotary_dim)
    if rotary_dim != found or rotary_dim < 2 or rotary_dim % 2:
        raise ValueError(f"rotary_dim {rotary_dim}, but freqs give {found} rotated values")
    half = rotary_dim // 2
    if freqs.dim() == 1:
        angle = positions.float()[:, None] * freqs.float()[None, :]
        cos, sin = angle.cos(), angle.sin()
    else:
        rows = freqs.float().index_select(0, positions.clamp(0, freqs.shape[0] - 1))
        cos, sin = rows[:, :half], rows[:, half:]
    if scale != 1.0:
        cos, sin = cos * scale, sin * scale
    return rotary_dim, cos, sin


def _rotate(x32, positions, freqs, rotary_dim, style, scale):
    """x32 [tokens][heads][dim] FP32 rotated in pairs on its first rotary_dim values."""
    torch = _torch()
    if style not in STYLES:
        raise ValueError(f"style is one of {', '.join(STYLES)}, not {style!r}")
    if positions.shape != (x32.shape[0],):
        raise ValueError(f"positions of shape {tuple(positions.shape)} for {x32.shape[0]} tokens")
    rotary_dim, cos, sin = _angles(positions, freqs, rotary_dim, scale)
    if rotary_dim > x32.shape[-1]:
        raise ValueError(f"rotary_dim {rotary_dim} is past the heads' {x32.shape[-1]} values")
    cos, sin = cos[:, None, :], sin[:, None, :]
    half = rotary_dim // 2
    if style == "half":
        a, b = x32[..., :half], x32[..., half:rotary_dim]
        turned = torch.cat([a * cos - b * sin, b * cos + a * sin], -1)
    else:
        a, b = x32[..., 0:rotary_dim:2], x32[..., 1:rotary_dim:2]
        turned = torch.stack([a * cos - b * sin, b * cos + a * sin], -1).flatten(-2)
    return turned if rotary_dim == x32.shape[-1] else torch.cat([turned, x32[..., rotary_dim:]], -1)


def _heads(x):
    if x.dim() == 2:
        return x[:, None, :]
    if x.dim() != 3:
        raise ValueError(f"rope rotates [tokens][heads][dim] or [tokens][dim], not {tuple(x.shape)}")
    return x


def rope(x, positions, freqs, rotary_dim=None, style="half", scale=1.0, out_dtype=None, out=None):
    """Each token's heads ([tokens][heads][dim] or [tokens][dim]) rotated at its position (positions [tokens]):
    freqs [rotary_dim / 2] inverse frequencies (angle fp32(position) * freq in FP32) or a [positions][rotary_dim]
    table (cos of every pair, then sin); style half (rotate_half) or interleaved; scale multiplies cos and sin."""
    y = _rotate(_heads(x).float(), positions, freqs, rotary_dim, style, scale).reshape(x.shape)
    return _deliver(y.to(out_dtype or x.dtype), out)


def rms_norm_rope(x, weight, eps, positions, freqs, offset=0.0, rotary_dim=None, style="half", scale=1.0,
                  out_dtype=None, out=None):
    """rms_norm_heads, then rope, in FP32 between them (one rounding, to out_dtype)."""
    normed = _normalized(_heads(x), eps, weight, offset)
    y = _rotate(normed, positions, freqs, rotary_dim, style, scale).reshape(x.shape)
    return _deliver(y.to(out_dtype or x.dtype), out)


def _gate(name, z32):
    if name == "silu":
        return _silu(z32)
    if name == "sigmoid":
        return _sigmoid(z32)
    raise ValueError(f"a gated norm's activation is one of {', '.join(GATES)}, not {name!r}")


def gated_rms_norm(x, z, weight, eps, offset=0.0, activation="silu", out_dtype=None, out=None):
    """rms_norm of each last-dimension row of x times activation(z) element by element (silu or sigmoid)."""
    if z.shape != x.shape:
        raise ValueError(f"gates of shape {tuple(z.shape)} for rows of shape {tuple(x.shape)}")
    y = _normalized(x, eps, weight, offset) * _gate(activation, z.float())
    return _deliver(y.to(out_dtype or x.dtype), out)


def _operands(x, up, layout):
    if up is not None:
        if up.shape != x.shape:
            raise ValueError(f"up of shape {tuple(up.shape)} for gates of shape {tuple(x.shape)}")
        return x, up
    if x.shape[-1] % 2:
        raise ValueError(f"gate and up halves of an even width, not {x.shape[-1]}")
    half = x.shape[-1] // 2
    if layout == "halves":
        return x[..., :half], x[..., half:]
    if layout == "interleaved":
        return x[..., 0::2], x[..., 1::2]
    raise ValueError(f"layout is one of {', '.join(LAYOUTS)}, not {layout!r}")


def silu_mul(x, up=None, layout="halves", out_dtype=None, out=None):
    """silu(gate) * up: x the gates and up the values, or x [..., 2n] holding both (halves: [gate | up];
    interleaved: gate at even, up at odd columns)."""
    gate, value = _operands(x, up, layout)
    return _deliver((_silu(gate.float()) * value.float()).to(out_dtype or x.dtype), out)


def gelu_mul(x, up=None, layout="halves", approximate="none", out_dtype=None, out=None):
    """gelu(gate) * up as silu_mul; approximate none (erf) or tanh."""
    if approximate not in APPROXIMATIONS:
        raise ValueError(f"approximate is one of {', '.join(APPROXIMATIONS)}, not {approximate!r}")
    gate, value = _operands(x, up, layout)
    y = _gelu(gate.float(), approximate) * value.float()
    return _deliver(y.to(out_dtype or x.dtype), out)


def sigmoid_mul(x, gate, out_dtype=None, out=None):
    """x * sigmoid(gate), element by element (an attention output's gate)."""
    if gate.shape != x.shape:
        raise ValueError(f"a gate of shape {tuple(gate.shape)} for x of shape {tuple(x.shape)}")
    return _deliver((x.float() * _sigmoid(gate.float())).to(out_dtype or x.dtype), out)


def l2_norm(x, eps=1e-6, out_dtype=None, out=None):
    """x / sqrt(sum(x^2) + eps) per last-dimension row (rms_norm with eps / dim, over sqrt(dim))."""
    width = x.shape[-1]
    y = _normalized(x, eps / width) * (1.0 / math.sqrt(width))
    return _deliver(y.to(out_dtype or x.dtype), out)


STREAM_GATES = ("none", "silu", "gelu", "gelu_tanh", "sigmoid")


def _activation(name, g32):
    if name == "none":
        return g32
    if name == "silu":
        return _silu(g32)
    if name == "gelu":
        return _gelu(g32)
    if name == "gelu_tanh":
        return _gelu(g32, "tanh")
    if name == "sigmoid":
        return _sigmoid(g32)
    raise ValueError(f"activation is one of {', '.join(STREAM_GATES)}, not {name!r}")


def stream_sum(x, gate=None, activation="none", scale=1.0, out_dtype=None, out=None):
    """y[t] = scale * the sum over streams, in stream order, of activation(gate[t][s]) * x[t][s]: x [tokens][streams]
    [dim], gate x's shape or [tokens][streams] (one per stream) or None (the plain sum). Elementwise adds in stream
    order, not a reduction kernel, so each token's bits are its own."""
    if x.dim() != 3:
        raise ValueError(f"x is [tokens][streams][dim], not {tuple(x.shape)}")
    tokens, streams, width = x.shape
    if activation not in STREAM_GATES:
        raise ValueError(f"activation is one of {', '.join(STREAM_GATES)}, not {activation!r}")
    terms = x.float()
    if gate is not None:
        if tuple(gate.shape) not in ((tokens, streams, width), (tokens, streams)):
            raise ValueError(f"a gate of shape {tuple(gate.shape)} for streams of shape {tuple(x.shape)}")
        g = _activation(activation, gate.float())
        terms = terms * (g if g.dim() == 3 else g[:, :, None])
    total = terms[:, 0]
    for s in range(1, streams):
        total = total + terms[:, s]
    if scale != 1.0:
        total = total * scale
    return _deliver(total.to(out_dtype or x.dtype), out)


def stream_add(streams, x, weights=None):
    """streams[t][s] += weights[t][s] * x[t] for every stream, in place (weights [tokens][streams] or None for 1)."""
    if streams.dim() != 3 or tuple(x.shape) != (streams.shape[0], streams.shape[2]):
        raise ValueError(f"streams [tokens][streams][dim] and x [tokens][dim], not {tuple(streams.shape)} and "
                         f"{tuple(x.shape)}")
    if weights is not None and tuple(weights.shape) != tuple(streams.shape[:2]):
        raise ValueError(f"weights of shape {tuple(weights.shape)}: one per token and stream")
    add = x.float()[:, None, :]
    if weights is not None:
        add = add * weights.float()[:, :, None]
    else:
        add = add.expand(streams.shape)
    streams.copy_(streams.float() + add)
    return streams


def _softplus(t):
    """softplus of an FP32 tensor: torch's on CUDA (v past 20, log1p(exp(v)) below); on the CPU the same formula from
    exp and log1p, which give every value one formula whatever the row count (torch's own CPU softplus does not)."""
    torch = _torch()
    if t.device.type == "cuda":
        return torch.nn.functional.softplus(t)
    return torch.where(t > 20.0, t, torch.log1p(torch.exp(t)))


def softplus_decay(x, bias, rate, gate=None, out_dtype=None, out=None, gate_out=None):
    """decay = -rate * softplus(x + bias) (x [tokens][n], bias and rate [n]) and with gate also sigmoid(gate):
    (decay, beta); FP32 inside, out_dtype FP32 by default."""
    torch = _torch()
    if x.dim() != 2 or tuple(bias.shape) != (x.shape[1],) or tuple(rate.shape) != (x.shape[1],):
        raise ValueError(f"x [tokens][n] with bias and rate [n], not {tuple(x.shape)}, {tuple(bias.shape)} and "
                         f"{tuple(rate.shape)}")
    if gate is not None and gate.shape != x.shape:
        raise ValueError(f"a gate of shape {tuple(gate.shape)} for x of shape {tuple(x.shape)}")
    out_dtype = out_dtype or torch.float32
    decay = _deliver((-rate.float() * _softplus(x.float() + bias.float())).to(out_dtype), out)
    if gate is None:
        return decay
    return decay, _deliver(_sigmoid(gate.float()).to(out_dtype), gate_out)


def _pair_shapes(q, k, positions):
    if q.dim() != 3 or k.dim() != 3 or k.shape[0] != q.shape[0] or k.shape[-1] != q.shape[-1]:
        raise ValueError(f"q [tokens][q_heads][dim] and k [tokens][k_heads][dim] of the same tokens and head size, "
                         f"not {tuple(q.shape)} and {tuple(k.shape)}")
    if positions.shape != (q.shape[0],):
        raise ValueError(f"positions of shape {tuple(positions.shape)} for {q.shape[0]} tokens")


def rms_norm_rope_pair(q, k, q_weight, k_weight, eps, positions, freqs, offset=0.0, rotary_dim=None, style="half",
                       scale=1.0, out_dtype=None, q_out=None, k_out=None):
    """rms_norm_rope of a layer's queries and of its keys at the same positions: (q, k)."""
    _pair_shapes(q, k, positions)
    if (q_weight is None) != (k_weight is None):
        raise ValueError("q and k both have a weight or neither")
    return (rms_norm_rope(q, q_weight, eps, positions, freqs, offset, rotary_dim, style, scale, out_dtype, q_out),
            rms_norm_rope(k, k_weight, eps, positions, freqs, offset, rotary_dim, style, scale, out_dtype, k_out))


def rope_pair(q, k, positions, freqs, rotary_dim=None, style="half", scale=1.0, out_dtype=None, q_out=None,
              k_out=None):
    """rope of a layer's queries and of its keys at the same positions: (q, k)."""
    _pair_shapes(q, k, positions)
    return (rope(q, positions, freqs, rotary_dim, style, scale, out_dtype, q_out),
            rope(k, positions, freqs, rotary_dim, style, scale, out_dtype, k_out))


def stream_add_norm(streams, x, weights, weight, eps, offset=0.0, out_dtype=None, out=None):
    """stream_add, then rms_norm of every stream as stored: (y, streams)."""
    stream_add(streams, x, weights)
    return rms_norm(streams, weight, eps, offset, out_dtype=out_dtype, out=out), streams


OPERATIONS = ("rms_norm", "rms_norm_heads", "rms_norm_rope", "gated_rms_norm", "rope", "silu_mul", "gelu_mul",
              "sigmoid_mul", "l2_norm", "stream_sum", "stream_add", "softplus_decay", "rms_norm_rope_pair",
              "rope_pair", "stream_add_norm")
