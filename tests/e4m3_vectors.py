"""The shared E4M3FN scale vectors of tests/inference/e4m3_scale_vectors.txt (also read by the C++ codec test)."""
from pathlib import Path

VECTORS = Path(__file__).resolve().parent / "inference" / "e4m3_scale_vectors.txt"


def e4m3_scale_vectors():
    """(FP32 bits, E4M3FN code) pairs, in file order."""
    pairs = []
    for line in VECTORS.read_text(encoding="utf-8").splitlines():
        fields = line.split("#", 1)[0].split()
        if fields:
            pairs.append((int(fields[0], 16), int(fields[1], 16)))
    return pairs


def e4m3_magnitude(byte):
    """|value| of an E4M3FN byte from its fields alone (bias 7, subnormals at exponent 0), None for NaN codes."""
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if exponent == 15 and mantissa == 7:
        return None
    return mantissa * 2.0 ** -9 if exponent == 0 else (8 + mantissa) * 2.0 ** (exponent - 10)
