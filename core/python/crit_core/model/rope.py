"""Rotary inverse frequencies and the attention factor from a RopeConfig, derived as Hugging Face's ROPE_INIT_FUNCTIONS
derive them (transformers.modeling_rope_utils), so an engine reads its rotary table instead of re-deriving YaRN, Llama 3
or LongRoPE scaling. The C++ twin is crit::core::rope_table (crit/model_config.h); both compute in double."""
import math
from typing import NamedTuple

from .config import ConfigError

DEFAULT_THETA = 10000.0   # Hugging Face's default rope_theta when a config declares none
TYPES = ("default", "mrope", "linear", "dynamic", "yarn", "llama3", "longrope")


class RopeTable(NamedTuple):
    """inv_freq: one inverse frequency per rotated pair (rotary_dim / 2 of them); attention_factor: the scale the
    cos and sin tables are multiplied by (1 except for YaRN and LongRoPE)."""
    inv_freq: tuple
    attention_factor: float


def rotary_dim(rope, head_dim):
    """int(head_dim x partial_rotary_factor), the dimensions that rotate; even and at least 2."""
    if type(head_dim) is not int or head_dim < 1:
        raise ConfigError(f"rope: head dimension {head_dim!r} is not a positive integer")
    factor = 1.0 if rope.partial_rotary_factor is None else rope.partial_rotary_factor
    if not 0 < factor <= 1:
        raise ConfigError(f"rope: partial_rotary_factor {factor!r} is outside (0, 1]")
    dim = int(head_dim * factor)
    if dim < 2 or dim % 2:
        raise ConfigError(f"rope: rotary dimension {dim} (head_dim {head_dim} x {factor:g}) is not a positive even number")
    return dim


def rope_table(rope, head_dim, max_position_embeddings=None, sequence_length=None):
    """The RopeTable of `rope` (a RopeConfig) for `head_dim`. max_position_embeddings is the model's declared length
    (ModelConfig.max_positions): dynamic NTK needs it, and YaRN, Llama 3 and LongRoPE read it when the rope object
    leaves factor or original_max_position_embeddings out. sequence_length selects dynamic NTK's base and LongRoPE's
    long factors past the original length; None means the declared length, as at Hugging Face's init. Types other
    than TYPES, and missing parameters, raise ConfigError."""
    kind = rope.type or "default"
    if kind not in TYPES:
        raise ConfigError(f"rope: type {kind!r} has no table; supported: {', '.join(TYPES)}")
    dim = rotary_dim(rope, head_dim)
    base = DEFAULT_THETA if rope.theta is None else rope.theta
    if not base > 1:
        raise ConfigError(f"rope: rope_theta {base!r} must exceed 1")
    half = dim // 2
    exponents = [2 * i / dim for i in range(half)]

    def need(name, value):
        if value is None:
            raise ConfigError(f"rope: type {kind!r} needs {name}")
        return value

    def original():
        return need("original_max_position_embeddings (or max_position_embeddings)",
                    rope.original_max_position_embeddings or max_position_embeddings)

    def factor_or_ratio():
        if rope.factor is not None:
            return rope.factor
        return need("max_position_embeddings", max_position_embeddings) / original()

    plain = [1.0 / base ** e for e in exponents]
    if kind in ("default", "mrope"):
        return RopeTable(tuple(plain), 1.0)
    if kind == "linear":
        factor = need("factor", rope.factor)
        return RopeTable(tuple(f / factor for f in plain), 1.0)
    if kind == "dynamic":
        factor, declared = need("factor", rope.factor), need("max_position_embeddings", max_position_embeddings)
        if dim == 2:
            raise ConfigError("rope: dynamic NTK needs a rotary dimension above 2")
        length = max(sequence_length or declared, declared)
        scaled = base * ((factor * length / declared) - (factor - 1)) ** (dim / (dim - 2))
        return RopeTable(tuple(1.0 / scaled ** e for e in exponents), 1.0)
    if kind == "yarn":
        factor, start = factor_or_ratio(), original()
        attention = rope.attention_factor
        if attention is None:
            def mscale(scale, m=1.0):
                return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0
            # Hugging Face's rule: mscale and mscale_all_dim count only when both are nonzero.
            attention = (mscale(factor, rope.mscale) / mscale(factor, rope.mscale_all_dim)
                         if rope.mscale and rope.mscale_all_dim else mscale(factor))
        fast, slow = rope.beta_fast or 32.0, rope.beta_slow or 1.0

        def correction(rotations):
            return dim * math.log(start / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low, high = correction(fast), correction(slow)
        if rope.raw.get("truncate", True) is not False:
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001   # Hugging Face's guard against a zero-width ramp
        inv = []
        for i, f in enumerate(plain):
            extrapolate = 1.0 - min(max((i - low) / (high - low), 0.0), 1.0)
            inv.append(f / factor * (1.0 - extrapolate) + f * extrapolate)
        return RopeTable(tuple(inv), float(attention))
    if kind == "llama3":
        factor = need("factor", rope.factor)
        low = need("low_freq_factor", rope.low_freq_factor)
        high = need("high_freq_factor", rope.high_freq_factor)
        start = original()
        inv = []
        for f in plain:
            wavelength = 2 * math.pi / f
            if wavelength < start / high:
                inv.append(f)
            elif wavelength > start / low:
                inv.append(f / factor)
            else:
                smooth = (start / wavelength - low) / (high - low)
                inv.append((1 - smooth) * f / factor + smooth * f)
        return RopeTable(tuple(inv), 1.0)
    # longrope
    start = original()
    factor = factor_or_ratio()
    for name, values in (("short_factor", rope.short_factor), ("long_factor", rope.long_factor)):
        if len(values) != half:
            raise ConfigError(f"rope: {name} has {len(values)} entries, not rotary_dim / 2 = {half}")
    attention = rope.attention_factor
    if attention is None:
        attention = 1.0 if factor <= 1 else math.sqrt(1 + math.log(factor) / math.log(start))
    scales = rope.long_factor if sequence_length and sequence_length > start else rope.short_factor
    return RopeTable(tuple(f / s for f, s in zip(plain, scales)), float(attention))
