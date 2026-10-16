"""Portable grouped signed-integer representation and projection semantics."""
from dataclasses import dataclass
import math
import struct


def represented_scale(value):
    """Round a positive scale to finite FP32, rejecting overflow/underflow."""
    if isinstance(value, bool):
        raise ValueError("positive finite FP32 scale required")
    try:
        rounded = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, TypeError, struct.error) as exc:
        raise ValueError("scale is not representable in FP32") from exc
    if not math.isfinite(rounded) or rounded <= 0:
        raise ValueError("positive finite FP32 scale required")
    return rounded


@dataclass(frozen=True)
class QuantizedRows:
    rows: int
    width: int
    group: int
    bits: int
    codes: bytes
    scales: tuple[float, ...]

    def validate(self):
        if any(type(x) is not int or x <= 0 for x in (self.rows, self.width, self.group)):
            raise ValueError("positive integer geometry required")
        if self.bits not in (4, 8) or self.width % self.group:
            raise ValueError("4/8-bit complete groups required")
        if len(self.codes) != (self.rows * self.width * self.bits + 7) // 8:
            raise ValueError("encoded extent differs from geometry")
        if len(self.scales) != self.rows * self.width // self.group:
            raise ValueError("one represented scale per group required")
        if any(represented_scale(x) != x for x in self.scales):
            raise ValueError("scales must already contain their represented FP32 values")


def quantize(rows, group=32, bits=8):
    if bits not in (4, 8) or type(group) is not int or group <= 0:
        raise ValueError("unsupported representation")
    rows = tuple(tuple(row) for row in rows)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("nonempty rectangular rows required")
    width = len(rows[0])
    if width % group or any(not math.isfinite(v) for row in rows for v in row):
        raise ValueError("finite complete groups required")
    limit = (1 << (bits - 1)) - 1
    codes, scales = [], []
    for row in rows:
        for start in range(0, width, group):
            values = row[start:start + group]
            scale = max(map(abs, values)) / limit or 1.0
            scale = represented_scale(scale)
            scales.append(scale)
            codes.extend(max(-limit, min(limit, round(v / scale))) & ((1 << bits) - 1) for v in values)
    if bits == 8:
        payload = bytes(codes)
    else:
        payload = bytes(codes[i] | ((codes[i + 1] if i + 1 < len(codes) else 0) << 4)
                        for i in range(0, len(codes), 2))
    result = QuantizedRows(len(rows), width, group, bits, payload, tuple(scales))
    result.validate()
    return result


def dequantize(weights):
    weights.validate()
    values = []
    for index in range(weights.rows * weights.width):
        code = weights.codes[index] if weights.bits == 8 else (weights.codes[index // 2] >> (4 * (index % 2))) & 15
        if code >= 1 << (weights.bits - 1):
            code -= 1 << weights.bits
        values.append(code * weights.scales[index // weights.group])
    return tuple(tuple(values[i:i + weights.width]) for i in range(0, len(values), weights.width))


def project(weights, activation, backend=None):
    """An optimized backend consumes these exact codes/scales and accumulation contract."""
    weights.validate()
    activation = tuple(activation)
    if len(activation) != weights.width or any(not math.isfinite(x) for x in activation):
        raise ValueError("activation geometry/domain differs")
    if backend is not None:
        return backend(weights, activation)
    return tuple(sum(w * a for w, a in zip(row, activation)) for row in dequantize(weights))
