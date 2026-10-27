"""Stored-tile sources from a checkpoint: the row-major codes, scales and global scales repack_fp8 / repack_fp4 take,
read from an NVFP4 (compressed-tensors or ModelOpt) or FP8 matrix as stored.

Core decides how a matrix is stored (crit_core.model.matrix_parts: which tensors, which scale convention) and reads
the bytes (crit_core.model.TensorReader); this module checks the matrix against the tile layout's requirements and
puts its scales in the form the launches take: the NVFP4 global scales as divisors (Fp4Weights) and the FP8 scale as
one FP32 value per row (the engine places it as the BF16 row-scale vector). Codes and NVFP4 scales are zero-copy
views of the checkpoint; nothing is re-encoded.
"""
from array import array
from dataclasses import dataclass
import struct
import sys
from pathlib import Path

try:
    from crit_core.model import SchemeError, TensorReader, matrix_parts
except ImportError:  # running from this checkout without the crit-core distribution
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "core" / "python"))
    from crit_core.model import SchemeError, TensorReader, matrix_parts

# Repack granularity: 16-row tiles of 32 FP8 or 64 NVFP4 columns (stored_tile_gemm.h).
TILE_ROWS = 16
TILE_COLUMNS = {"fp8": 32, "fp4": 64}


@dataclass(frozen=True)
class TileSource:
    """One matrix [n, k] ready for enqueue_repack_fp8 / enqueue_repack_fp4 (or their _rows forms).

    fp4: `codes` row-major [n][k / 2] E2M1 pairs, `scales` [n][k / 16] E4M3, `weight_global_scale` the divisor
    Fp4Weights takes, `input_global_scale` the checkpoint's static activation divisor (None when it declares none).
    fp8: `codes` row-major [n][k] E4M3 and `row_scale` n FP32 multipliers (a per-tensor scale repeated per row).
    `parts` is Core's StoredMatrix the source was read from."""
    format: str
    n: int
    k: int
    codes: memoryview
    scales: memoryview | None
    row_scale: array | None
    weight_global_scale: float | None
    input_global_scale: float | None
    parts: object


def _floats(reader, name):
    """The values of a small F32, BF16 or F16 tensor as Python floats."""
    tensor = reader.tensor(name)
    data = bytes(reader.map(tensor))
    if tensor.dtype == "F32":
        return list(struct.unpack(f"<{len(data) // 4}f", data))
    if tensor.dtype == "BF16":
        return [struct.unpack("<f", b"\0\0" + data[i:i + 2])[0] for i in range(0, len(data), 2)]
    if tensor.dtype == "F16":
        return list(struct.unpack(f"<{len(data) // 2}e", data))
    raise SchemeError(f"{name}: scale stored as {tensor.dtype}, expected F32, BF16 or F16")


def _scalar(reader, name):
    values = _floats(reader, name)
    if len(values) != 1:
        raise SchemeError(f"{name}: expected one value, found {len(values)}")
    return values[0]


def _divisor(value, divides, name):
    """A global scale as the divisor the kernels take. A ModelOpt multiplier m becomes 1 / m rounded to FP32 (the
    division by it then equals the multiplication up to that rounding; exact when m is a power of two)."""
    if not value > 0:
        raise SchemeError(f"{name}: global scale {value} is not positive")
    return value if divides else struct.unpack("<f", struct.pack("<f", 1.0 / value))[0]


def _expert_view(view, shape, expert, row_bytes):
    """Rows of one expert of a stacked [E, n, ...] tensor, or the whole [n, ...] tensor."""
    if len(shape) == 2:
        if expert is not None:
            raise SchemeError(f"expert {expert} of a matrix that is not stacked")
        return view
    if len(shape) != 3 or expert is None or not 0 <= expert < shape[0]:
        raise SchemeError(f"shape {list(shape)} needs an expert index in [0, {shape[0] if len(shape) == 3 else 0})"
                          if len(shape) == 3 else f"shape {list(shape)} is not a matrix or a stack of matrices")
    size = shape[1] * row_bytes
    return view[expert * size:(expert + 1) * size]


def tile_source(reader, base, quantization=None, *, expert=None):
    """The TileSource of matrix `base` in the checkpoint `reader` (a crit_core.model.TensorReader) reads, with the
    model's quantization config (crit_core.model.quantization_of). `expert` picks one matrix of a stacked [E, n, k]
    tensor. SchemeError when the matrix is not NVFP4 or E4M3 FP8 with tensor or row scales, or when n is not a
    multiple of 16 or k of 32 (FP8) or 64 (NVFP4)."""
    if not isinstance(reader, TensorReader):
        raise TypeError(f"expected a crit_core.model.TensorReader, not {type(reader).__name__}")
    parts = matrix_parts(reader.checkpoint.tensors, base, quantization)
    if parts is None:
        raise KeyError(f"no matrix {base!r} in {reader.checkpoint.source}")
    if parts.scheme not in ("nvfp4", "fp8"):
        raise SchemeError(f"{base}: stored as {parts.scheme}; stored tiles hold NVFP4 and FP8 matrices")
    fmt = "fp4" if parts.scheme == "nvfp4" else "fp8"
    n, k = parts.shape[-2:]
    if n % TILE_ROWS or k % TILE_COLUMNS[fmt]:
        raise SchemeError(f"{base}: [{n}, {k}] does not fill {TILE_ROWS} x {TILE_COLUMNS[fmt]} tiles")
    codes = _expert_view(reader.map(parts.codes), parts.shape, expert, k // 2 if fmt == "fp4" else k)
    if fmt == "fp4":
        scales = _expert_view(reader.map(parts.scales), parts.shape, expert, k // 16)
        weight = _divisor(_scalar(reader, parts.global_scale), parts.global_divides, parts.global_scale)
        activation = None if parts.input_scale is None else \
            _divisor(_scalar(reader, parts.input_scale), parts.input_divides, parts.input_scale)
        return TileSource(fmt, n, k, codes, scales, None, weight, activation, parts)
    if reader.tensor(parts.codes).dtype != "F8_E4M3":
        raise SchemeError(f"{base}: FP8 stored as {reader.tensor(parts.codes).dtype}; stored tiles hold E4M3")
    if parts.scale_granularity == "block":
        raise SchemeError(f"{base}: FP8 with {list(parts.block)} block scales; stored tiles take one scale per row")
    values = _floats(reader, parts.scales)
    if parts.scale_granularity == "tensor":
        rows = values * n
    else:   # one per row of every stacked matrix; the codes' view already checked `expert`
        rows = values if len(parts.shape) == 2 else values[expert * n:(expert + 1) * n]
    return TileSource(fmt, n, k, codes, None, array("f", rows), None, None, parts)
