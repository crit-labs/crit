"""Host reference for GGUF Q4_0, Q8_0, Q4_K, Q5_K and Q6_K blocks, equal bit for bit to gguf-py's quants.dequantize.

Each format stores a row of a weight matrix as consecutive blocks; `dequantize(kind, data)` returns the FP32 values
of every element, computed with gguf-py's operations in gguf-py's order, each rounded to FP32 (doubles hold every
product of two FP32 values exactly, and rounding a double sum of two FP32 values to FP32 equals FP32 addition).

    Q8_0   32 values: d (f16), 32 int8 codes                        x = d * q
    Q4_0   32 values: d (f16), 16 bytes, low nibbles first          x = d * (q - 8)
    Q4_K  256 values: d, dmin (f16), 12 bytes of 6-bit scales and mins, 128 bytes of nibbles
                      sub-block j (32 values) has scale sc_j and min m_j; x = (d * sc_j) * q - dmin * m_j
    Q5_K  256 values: as Q4_K plus 32 bytes of fifth bits (bit j of byte l belongs to element l of sub-block j)
    Q6_K  256 values: 128 bytes of low nibbles, 64 bytes of high 2-bit pairs, 16 int8 scales, d (f16)
                      x = (d * scale_(e // 16)) * (q - 32)
"""
import struct

FORMATS = {  # name: (values per block, bytes per block)
    "Q4_0": (32, 18), "Q8_0": (32, 34), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210),
}


def f32(x):
    """Round a double to the nearest FP32 value."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f16(data, at):
    return struct.unpack_from("<e", data, at)[0]


def scale_min(scales, j):
    """The 6-bit scale and min of sub-block j from the 12 packed bytes of a K-quant block."""
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4), (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)


def _q4_0(b):
    d = f16(b, 0)
    return [f32(d * ((b[2 + e] & 15) - 8)) for e in range(16)] + [f32(d * ((b[2 + e] >> 4) - 8)) for e in range(16)]


def _q8_0(b):
    d = f16(b, 0)
    return [f32(d * q) for q in struct.unpack_from("<32b", b, 2)]


def _k_scales(b):
    d, dmin, scales = f16(b, 0), f16(b, 2), b[4:16]
    return [(f32(d * sc), f32(dmin * m)) for sc, m in (scale_min(scales, j) for j in range(8))]


def _q4_k(b):
    out, qs = [], b[16:144]
    for j, (dj, mj) in enumerate(_k_scales(b)):
        i, h = divmod(j, 2)
        out += [f32(f32(dj * ((qs[32 * i + l] >> (4 * h)) & 15)) - mj) for l in range(32)]
    return out


def _q5_k(b):
    out, qh, qs = [], b[16:48], b[48:176]
    for j, (dj, mj) in enumerate(_k_scales(b)):
        i, h = divmod(j, 2)
        out += [f32(f32(dj * (((qs[32 * i + l] >> (4 * h)) & 15) | (((qh[l] >> j) & 1) << 4))) - mj) for l in range(32)]
    return out


def _q6_k(b):
    ql, qh, scales, d = b[0:128], b[128:192], struct.unpack_from("<16b", b, 192), f16(b, 208)
    out = []
    for e in range(256):
        row, col = divmod(e, 32)
        a, t = divmod(row, 4)
        low = (ql[64 * a + 32 * (t % 2) + col] >> (4 * (t // 2))) & 15
        high = (qh[32 * a + col] >> (2 * t)) & 3
        out.append(f32(f32(d * scales[e // 16]) * ((low | (high << 4)) - 32)))
    return out


_DECODE = {"Q4_0": _q4_0, "Q8_0": _q8_0, "Q4_K": _q4_k, "Q5_K": _q5_k, "Q6_K": _q6_k}


def dequantize(kind, data):
    """FP32 values of the blocks in `data` (bytes), block after block."""
    values, size = FORMATS[kind]
    data = bytes(data)
    if len(data) % size:
        raise ValueError(f"{kind} data must be whole {size}-byte blocks")
    out = []
    for at in range(0, len(data), size):
        out += _DECODE[kind](data[at:at + size])
    return out


def project(kind, data, rows, x):
    """y[r] = sum_k w[r][k] * x[k] for a rows x len(x) matrix stored as blocks, accumulated in double."""
    w = dequantize(kind, data)
    k = len(x)
    if len(w) != rows * k:
        raise ValueError("data, rows and activations disagree")
    return [sum(w[r * k + c] * x[c] for c in range(k)) for r in range(rows)]


def test_blocks(kind, count, seed):
    """Deterministic blocks for fixtures: bytes from a 64-bit LCG, with each FP16 field set to a finite value of
    moderate magnitude and random sign. The native tests generate the same bytes."""
    values, size = FORMATS[kind]
    fields = {"Q4_0": (0,), "Q8_0": (0,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,)}[kind]
    state, out = seed, bytearray()
    for _ in range(count):
        block = bytearray(size)
        for i in range(size):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            block[i] = state >> 56
        for at in fields:
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            bits = 0x1C00 + ((state >> 40) % 0x1400) | ((state >> 63) << 15)   # exponents 7..11: magnitudes 2^-8..2^-3
            struct.pack_into("<H", block, at, bits)
        out += block
    return bytes(out)
