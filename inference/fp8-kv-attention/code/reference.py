"""Host reference for the FP8 KV cache: E4M3 encoding with one scale per head vector, and the values the attention
kernels read back.

A key or value vector x of one head is stored as codes c_i = e4m3(x_i / s) with s = max |x_i| / 448 (s = 1 for an
all-zero vector), so the largest element maps to E4M3's largest finite value. The kernels read element i as
bf16(decode(c_i) * s). Attention over these values (for example flash-prefill-attention's reference.py) is the
exact target the CUDA kernels approximate; FP8 storage itself is a quality setting the engine's quality budget judges.
"""
import math
import struct

E4M3_MAX = 448.0


def decode(code):
    """Value of one E4M3 byte (no infinities; 0x7F / 0xFF are NaN)."""
    sign = -1.0 if code & 0x80 else 1.0
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return math.nan
    if exponent == 0:
        return sign * math.ldexp(mantissa / 8.0, -6)
    return sign * math.ldexp(1.0 + mantissa / 8.0, exponent - 7)


_FINITE = sorted({decode(c) for c in range(256) if not math.isnan(decode(c)) and decode(c) >= 0.0})


def encode(x):
    """Nearest E4M3 code (ties to an even code), saturating to +-448."""
    if math.isnan(x):
        raise ValueError("NaN has no E4M3 code here")
    magnitude = min(abs(x), E4M3_MAX)
    lo, hi = 0, len(_FINITE) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if _FINITE[mid] <= magnitude else (lo, mid)
    below, above = _FINITE[lo], _FINITE[hi]
    if magnitude - below < above - magnitude or (magnitude - below == above - magnitude and lo % 2 == 0):
        index = lo
    else:
        index = hi
    return index | (0x80 if x < 0 and index else 0)


def bf16(x):
    """Round a float to the nearest BF16 value (ties to even)."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def quantize(vector):
    """(codes, scale) for one head vector."""
    amax = max((abs(x) for x in vector), default=0.0)
    scale = amax / E4M3_MAX if amax > 0.0 else 1.0
    return [encode(x / scale) for x in vector], scale


def dequantize(codes, scale):
    """The values the attention kernels read: bf16(code * scale)."""
    return [bf16(decode(c) * scale) for c in codes]
