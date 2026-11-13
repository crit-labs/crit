"""The projections slot's host form over GGUF block formats: a matrix as its stored blocks (uint8 rows, a GGUF type
id), its stored values from Core's exact decoders (crit_core.model.gguf_blocks), and the projection with exact sums
over them; a Q8_0 encoder of a dense matrix, so the slot's cases (rows of numbers) have a stored form.

    raw = encode_q8_0(rows)                         # uint8 [N][row bytes]: d = max |w| / 127 a 32-value block (FP16)
    w = GgufWeight(raw, TYPES["Q8_0"], k)
    y = project_rows(w, activations)                # [[out] per token], fsum over the stored values
"""
from dataclasses import dataclass
import math

from crit_core.model.gguf_blocks import TYPES, row_bytes, stored_values


@dataclass
class GgufWeight:
    blocks: object      # uint8 numpy [N][row bytes]
    type: int           # GGUF type id
    k: int              # inputs a row


def encode_q8_0(rows):
    """A dense matrix [N][K] (K a multiple of 32) as Q8_0 blocks, uint8 numpy [N][K / 32 * 34]: each 32 values' scale
    d = max |w| / 127 stored in FP16, the values round(w / d) in int8 (0 where d is 0), as the format quantizes."""
    import numpy as np
    w = np.asarray(rows, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % 32 or not w.size:
        raise ValueError("a nonempty [out][in] matrix whose rows are whole 32-value blocks")
    blocks = w.reshape(w.shape[0], -1, 32)
    d = (np.abs(blocks).max(axis=-1, keepdims=True) / 127).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d == 0, 0, np.round(blocks / np.where(d == 0, 1, d)))
    q = np.clip(q, -127, 127).astype(np.int8)
    raw = np.concatenate([d.astype(np.float16).view(np.uint8), q.view(np.uint8)], axis=-1)
    return np.ascontiguousarray(raw.reshape(w.shape[0], -1))


def dense_rows(weight):
    """The stored values [N][K] as Python floats (the values project_rows multiplies)."""
    return stored_values(weight.blocks, weight.type, weight.k).astype(float).tolist()


def project_rows(weight, activations):
    """y[t][n] = fsum over k of W[n][k] x[t][k], W the stored values: exact sums, one rounding."""
    w = dense_rows(weight)
    if any(len(x) != weight.k for x in activations):
        raise ValueError("activation width differs from the matrix's inputs")
    return [[math.fsum(a * b for a, b in zip(row, x)) for row in w] for x in activations]


def check_rows(weight):
    """ValueError unless the blocks are uint8 rows of the format's size for k values."""
    import numpy as np
    if weight.type not in TYPES.values():
        raise ValueError(f"GGUF type {weight.type} is not one of {', '.join(f'{n} ({t})' for n, t in TYPES.items())}")
    blocks = np.asarray(weight.blocks)
    if blocks.dtype != np.uint8 or blocks.ndim not in (2, 3) or blocks.shape[-1] != row_bytes(weight.type, weight.k):
        raise ValueError(f"a {weight.k}-value row of type {weight.type} holds {row_bytes(weight.type, weight.k)} bytes")
