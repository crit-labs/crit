"""NVFP4 weight codec and scalar projection reference (FP64, standard library).

An NVFP4 row stores 4-bit E2M1 codes, one E4M3FN scale per 16 consecutive values
and one FP32 global scale per tensor. With the divisor convention used by common
checkpoints, a value decodes as

    value = e2m1(code) * e4m3(block_scale) / global_scale

Packing here is "adjacent": byte i holds code 2i in its low nibble and code 2i+1
in its high nibble. Other layouts exist (for example interleaved half-blocks);
convert them explicitly, never by guessing from sizes.
"""
import math

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # magnitudes of codes 0..7; bit 3 is the sign
BLOCK = 16
E2M1_MAX, E4M3_MAX = 6.0, 448.0


def e2m1_value(code):
    if not 0 <= code <= 15:
        raise ValueError("E2M1 code outside 0..15")
    magnitude = E2M1[code & 7]
    return -magnitude if code & 8 else magnitude


def e4m3_value(byte):
    """E4M3FN: bias 7, no infinities, 0x7F/0xFF are NaN, maximum 448."""
    if not 0 <= byte <= 255:
        raise ValueError("E4M3 byte outside 0..255")
    sign = -1.0 if byte & 0x80 else 1.0
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if exponent == 15 and mantissa == 7:
        return math.nan
    if exponent == 0:
        return sign * mantissa / 8 * 2.0 ** -6
    return sign * (1 + mantissa / 8) * 2.0 ** (exponent - 7)


def _nearest(magnitude, table):
    """Index of the nearest table entry; ties go to the even index (round half to even)."""
    best = 0
    for i, v in enumerate(table):
        d, bd = abs(magnitude - v), abs(magnitude - table[best])
        if d < bd or (d == bd and i % 2 == 0 and best % 2 == 1):
            best = i
    return best


_E4M3_POSITIVE = tuple(e4m3_value(b) for b in range(0x7F))  # codes 0..126, finite, ascending


def e4m3_encode(x):
    """Round to nearest even, saturating at 448. NaN is rejected."""
    if math.isnan(x):
        raise ValueError("NaN scale")
    code = _nearest(min(abs(x), E4M3_MAX), _E4M3_POSITIVE)
    return code | (0x80 if x < 0 and code else 0)


def e2m1_encode(x):
    """Round to nearest even, saturating at 6."""
    if math.isnan(x):
        raise ValueError("NaN value")
    code = _nearest(min(abs(x), E2M1_MAX), E2M1)
    return code | (8 if x < 0 and code else 0)


def global_scale(values):
    """Divisor that maps the tensor maximum to the largest representable product."""
    amax = max((abs(v) for v in values), default=0.0)
    return E2M1_MAX * E4M3_MAX / amax if amax > 0 else 1.0


def _block(block, g, top):
    """Scale byte, codes and squared error (in scaled units) of one block whose maximum maps to `top`."""
    s = e4m3_encode(max(abs(v) for v in block) * g / top)
    decoded = e4m3_value(s)
    codes = [e2m1_encode(v * g / decoded) if decoded > 0 else 0 for v in block]
    error = sum((e2m1_value(c) * decoded - v * g) ** 2 for c, v in zip(codes, block))
    return s, codes, error


def quantize_row(row, g, four_over_six=False):
    """Return (codes, scale_bytes) for one row with global scale g. With four_over_six, each block also tries the
    scale that maps its maximum to 4 instead of 6 and keeps the one with the smaller squared error (6 on a tie)."""
    if len(row) % BLOCK or not math.isfinite(g) or g <= 0:
        raise ValueError("row length must be a multiple of 16 and the global scale positive")
    codes, scales = [], []
    for b in range(0, len(row), BLOCK):
        block = row[b:b + BLOCK]
        s, c, error = _block(block, g, E2M1_MAX)
        if four_over_six:
            s4, c4, error4 = _block(block, g, 4.0)
            if error4 < error:
                s, c = s4, c4
        scales.append(s)
        codes.extend(c)
    return codes, scales


def pack(codes):
    if len(codes) % 2:
        raise ValueError("an even number of codes is required")
    return bytes(codes[i] | codes[i + 1] << 4 for i in range(0, len(codes), 2))


def unpack(packed, count):
    if len(packed) * 2 < count:
        raise ValueError("packed row is too short")
    codes = []
    for byte in packed:
        codes += [byte & 15, byte >> 4]
    return codes[:count]


def dequantize_row(codes, scales, g):
    if len(codes) != len(scales) * BLOCK or not math.isfinite(g) or g <= 0:
        raise ValueError("one scale per 16 codes and a positive global scale required")
    return [e2m1_value(c) * e4m3_value(scales[i // BLOCK]) / g for i, c in enumerate(codes)]


class Matrix:
    """N rows of K values: packed codes [N][K/2], scale bytes [N][K/16], global scale."""

    def __init__(self, rows, cols, packed, scales, g):
        if cols % BLOCK or len(packed) != rows * cols // 2 or len(scales) != rows * cols // BLOCK:
            raise ValueError("NVFP4 geometry mismatch")
        if not math.isfinite(g) or g <= 0:
            raise ValueError("global scale must be positive and finite")
        self.rows, self.cols, self.packed, self.scales, self.g = rows, cols, bytes(packed), bytes(scales), g

    @classmethod
    def quantize(cls, dense, four_over_six=False):
        rows, cols = len(dense), len(dense[0])
        g = global_scale([v for row in dense for v in row])
        packed, scales = bytearray(), bytearray()
        for row in dense:
            codes, s = quantize_row(row, g, four_over_six)
            packed += pack(codes)
            scales += bytes(s)
        return cls(rows, cols, packed, scales, g)

    def row(self, n):
        codes = unpack(self.packed[n * self.cols // 2:(n + 1) * self.cols // 2], self.cols)
        return dequantize_row(codes, list(self.scales[n * self.cols // BLOCK:(n + 1) * self.cols // BLOCK]), self.g)


def project(matrix, tokens):
    """y[t][n] = sum_k W[n][k] * x[t][k]. Each 16-value block is decoded once and
    reused for every token row, which is what makes batched CPU projection cheap."""
    if any(len(x) != matrix.cols for x in tokens):
        raise ValueError("token width must equal the matrix columns")
    out = [[0.0] * matrix.rows for _ in tokens]
    for n in range(matrix.rows):
        row = matrix.row(n)
        for b in range(0, matrix.cols, BLOCK):
            block = row[b:b + BLOCK]
            for t, x in enumerate(tokens):
                out[t][n] += math.fsum(w * v for w, v in zip(block, x[b:b + BLOCK]))
    return out
