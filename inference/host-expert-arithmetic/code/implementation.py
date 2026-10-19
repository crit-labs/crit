"""Exact FP32 reference of a declared NVFP4 row order (host_nvfp4.h), standard library only.

Every operation rounds once to FP32, round to nearest even, as the C++ paths and the device kernel do: the
product e2m1(code) * scale, the FMA into the lane, each add of the reduction and the final multiply. The format
itself (E2M1 codes, E4M3 block scales, adjacent nibbles) comes from quantized-projections' nvfp4.py.
"""

import math
import struct
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quantized-projections" / "code"))
from nvfp4 import BLOCK, e2m1_value, e4m3_value  # noqa: E402

HALVING, SEQUENTIAL = "halving", "sequential"


def f32(x):
    """Round a double to FP32 (correct for products and sums of FP32 values: 53 >= 2 * 24 + 2)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _round_fraction(q, negative_zero):
    """Correctly round an exact rational to FP32 (normals and subnormals; finite inputs only)."""
    if q == 0:
        return -0.0 if negative_zero else 0.0
    sign = -1.0 if q < 0 else 1.0
    q = abs(q)
    exponent = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** exponent > q:
        exponent -= 1
    exponent = max(exponent, -126)  # subnormals keep the minimum exponent
    scaled = q / Fraction(2) ** (exponent - 23)  # mantissa in [2^23, 2^24) for normals
    whole, rest = divmod(scaled.numerator, scaled.denominator)
    half = Fraction(rest, scaled.denominator)
    if half > Fraction(1, 2) or (half == Fraction(1, 2) and whole % 2):
        whole += 1
    return sign * math.ldexp(whole, exponent - 23)


def fma32(a, b, c):
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    product_negative = math.copysign(1.0, a) * math.copysign(1.0, b) < 0
    return _round_fraction(exact, product_negative and math.copysign(1.0, c) < 0)


def codes_of(packed, cols):
    """Adjacent packing: byte i holds element 2i in its low nibble and 2i + 1 in its high nibble."""
    return [(packed[k // 2] >> (4 * (k % 2))) & 15 for k in range(cols)]


SCALED_ELEMENTS, BLOCK_DOT = "scaled_elements", "block_dot"


def dot_row(packed, scales, x, lanes=16, reduce=HALVING, post=1.0, product=SCALED_ELEMENTS):
    """The declared order. scaled_elements: element k -> lane k % lanes, fma(f32(e2m1 * scale), x, lane).
    block_dot: block b -> lane b % lanes, d = sequential fma(e2m1, x, d) over the block, fma(d, scale, lane).
    Then the lanes are reduced and multiplied by post."""
    cols = len(x)
    if (product == SCALED_ELEMENTS and cols % lanes) or cols % BLOCK or len(scales) != cols // BLOCK:
        raise ValueError("cols must be a multiple of 16 (and of the lanes for scaled_elements), one scale per 16")
    acc = [0.0] * lanes
    codes = codes_of(packed, cols)
    if product == BLOCK_DOT:
        for b in range(cols // BLOCK):
            d = 0.0
            for e in range(BLOCK):
                d = fma32(e2m1_value(codes[b * BLOCK + e]), x[b * BLOCK + e], d)
            acc[b % lanes] = fma32(d, scales[b], acc[b % lanes])
    else:
        for k, code in enumerate(codes):
            ws = f32(e2m1_value(code) * scales[k // BLOCK])
            acc[k % lanes] = fma32(ws, x[k], acc[k % lanes])
    if reduce == HALVING:
        width = lanes // 2
        while width:
            acc = [f32(acc[i] + acc[i + width]) for i in range(width)]
            width //= 2
        total = acc[0]
    else:
        total = acc[0]
        for value in acc[1:]:
            total = f32(total + value)
    return f32(total * post)


def standard_scales(scale_bytes):
    """The standard scale decode: E4M3FN bytes."""
    return [e4m3_value(b) for b in scale_bytes]


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]
