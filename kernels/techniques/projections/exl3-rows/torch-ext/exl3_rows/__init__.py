"""EXL3 projections (the exl3 checkpoint scheme), crit's kernel exl3-rows, built on this GPU with the engine's tuning or
the default knobs: y = x W for W [in][out] = diag(su) H Q H diag(sv), Q the trellis tiles at their stored K bits a
value (1 to 8, or 1.5, 2.5, 3.5 with codebook 2), H the block-diagonal normalized 128-point Hadamard, su and sv the
matrix's sign vectors.

    w = Exl3Matrix(trellis, suh, svh, codebook)      # suh / svh: FP16 signs, or the bitfields' unpack_signs()
    y = gemv(x, w)                                   # 1 to 16 rows: the tiles decoded on the fly, FP32 sums
    dense = dequantize(w)                            # W^T [out][in] BF16 for a prompt's GEMM

gemv rotates the rows (x su H in FP32, stored in FP16), multiplies them by the tiles decoded once each on FP16 tensor
cores (a warp a 16-column strip over its split of the input tiles, FP32 sums: a row's bits never depend on the rows
beside it), and rotates the result (y' H sv) into x's dtype. decode gives the tiles' stored FP16 values, the bits of the format's own decoder.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from ._ops import ops

__all__ = ["CODEBOOKS", "Exl3Matrix", "MAX_ROWS", "decode", "dequantize", "gemv", "rotate", "splits", "unpack_signs"]

MAX_ROWS = 16
CODEBOOKS = {"3inst": 0, "mcg": 1, "mul1": 2}


@dataclass
class Exl3Matrix:
    trellis: torch.Tensor   # [in / 16][out / 16][16 K] int16
    suh: torch.Tensor       # [in] FP16 signs
    svh: torch.Tensor       # [out] FP16 signs
    codebook: int = 0       # 0 3INST, 1 MCG (a .mcg marker), 2 MUL1 (a .mul1 marker)

    @property
    def shape(self):
        """(in, out)."""
        return 16 * self.trellis.shape[0], 16 * self.trellis.shape[1]

    @property
    def bits(self):
        """K, the bits a value: whole, or a half (codebook 2)."""
        return self.trellis.shape[2] / 16


def unpack_signs(bitfield: torch.Tensor, size: int) -> torch.Tensor:
    """FP16 +-1 of a sign bitfield (int16 words, bit i of word w element 16 w + i, 1 for -1)."""
    words = bitfield.reshape(-1).to(torch.int32) & 0xFFFF
    bits = (words[:, None] >> torch.arange(16, device=bitfield.device)) & 1
    return (1 - 2 * bits.reshape(-1)[:size]).to(torch.float16).contiguous()


def rotate(rows: torch.Tensor, pre: Optional[torch.Tensor] = None, post: Optional[torch.Tensor] = None,
           out: Optional[torch.Tensor] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """rows [M][L] in 128-value blocks: times pre, the normalized Hadamard, times post (FP16 signs [L] or None)."""
    out = torch.empty(rows.shape, dtype=dtype or rows.dtype, device=rows.device) if out is None else out
    for lo in range(0, rows.shape[0], 65535):   # a launch's grid holds 65,535 rows
        ops.rotate(rows[lo:lo + 65535], pre, post, out[lo:lo + 65535])
    return out


def decode(w: Exl3Matrix) -> torch.Tensor:
    """Q [in][out] FP16: the tiles' stored values."""
    q = torch.empty(w.shape, dtype=torch.float16, device=w.trellis.device)
    ops.decode(w.trellis, int(w.codebook), q)
    return q


def gemv(x: torch.Tensor, w: Exl3Matrix, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """y [M][out] = x [M][in] (M 1 to 16, BF16, FP16 or FP32) times W, in x's dtype."""
    rows = x.shape[0]
    xr = rotate(x, pre=w.suh, dtype=torch.float16)   # the tensor cores' A operand
    yr = torch.empty(splits(w, rows), rows, w.shape[1], dtype=torch.float32, device=x.device)
    ops.tiles(w.trellis, int(w.codebook), xr, yr)
    out = torch.empty(rows, w.shape[1], dtype=x.dtype, device=x.device) if out is None else out
    ops.rotate(yr, None, w.svh, out)   # the splits' sums added in order, then rotated
    return out


_SPLITS = {}


def splits(w: Exl3Matrix, rows: int) -> int:
    """The splits of the input's tiles gemv takes for w at `rows` rows on its device (kept per matrix shape)."""
    key = (w.trellis.device, tuple(w.trellis.shape), int(w.codebook), rows)
    if key not in _SPLITS:
        _SPLITS[key] = int(ops.splits(w.trellis, int(w.codebook), int(rows)))
    return _SPLITS[key]


def dequantize(w: Exl3Matrix, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """W^T [out][in] in dtype (for a GEMM on many rows): Q H diag(sv) by rows, transposed, then H diag(su) by rows."""
    q = rotate(decode(w), post=w.svh, dtype=torch.float32)
    return rotate(q.t().contiguous(), post=w.suh, dtype=dtype)
