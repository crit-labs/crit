"""GGUF projections (a GGUF checkpoint's quantized matrices), crit's kernel gguf-rows, built on this GPU with the
engine's tuning or the default knobs: y = x W^T for W [N][K] in a GGUF block format (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4), read at
its stored bits.

    w = GgufMatrix(blocks, "Q4_K", k)           # blocks: uint8 [N][row bytes] as the file stores them
    y = gemv(x, w)                              # 1 to 8 rows: int8 activations, dp4a over the stored blocks
    y = gemv(x, experts, ids)                   # experts: GgufMatrix over [E][N][row bytes]; row m times expert ids[m]
    y = gemm(x, w)                              # any rows: the s8 tensor cores, each output gemv's bits
    dense = dequantize(w)                       # W [N][K] in BF16 (or FP32, exact)

gemv quantizes each row's 32-value blocks to int8 (d = max |x| / 127, q = rint(x / d)), takes each sub-block's exact
integer dot with the stored values and weights it in FP32, in the order the kernel declares (gguf_rows.h): a row's
bits never depend on the rows beside it, a gathered row equals the plain call on its expert, and bench.py's
reference reproduces them exactly.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from ._ops import ops

__all__ = ["BLOCKS", "FORMATS", "GgufMatrix", "MAX_ROWS", "dequantize", "gemm", "gemv", "matmul", "quantize",
           "splits"]

MAX_ROWS = 8
# GGUF type ids of the formats, and each one's (values, bytes) a block.
FORMATS = {"Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7, "Q8_0": 8, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14,
           "IQ2_XXS": 16, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ1_S": 19, "IQ4_NL": 20, "IQ3_S": 21, "IQ2_S": 22,
           "IQ4_XS": 23, "IQ1_M": 29, "MXFP4": 39}
BLOCKS = {2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144),
          13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
          21: (256, 110), 22: (256, 82), 23: (256, 136), 29: (256, 56), 39: (32, 17)}


@dataclass
class GgufMatrix:
    blocks: torch.Tensor    # uint8 [N][row bytes], or [experts][N][row bytes]
    type: int               # GGUF type id (FORMATS)
    k: int                  # inputs

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = FORMATS[self.type]

    @property
    def n(self):
        return self.blocks.shape[-2]


def quantize(x: torch.Tensor, by_subblock: bool = False):
    """(xq int8 [M][K], xd FP32 [M][K / 32], xs int32 [M][K / 32]): x's 32-value blocks as gemv multiplies them; by
    sub-block, as gemm reads them, xq [K / 32][M][32], xd and xs [K / 32][M]."""
    m, k = x.shape
    shape = (k // 32, m) if by_subblock else (m, k // 32)
    xq = torch.empty(*((k // 32, m, 32) if by_subblock else (m, k)), dtype=torch.int8, device=x.device)
    xd = torch.empty(*shape, dtype=torch.float32, device=x.device)
    xs = torch.empty(*shape, dtype=torch.int32, device=x.device)
    ops.quantize(x, xq, xd, xs, by_subblock)
    return xq, xd, xs


_SPLITS = {}


def splits(w: GgufMatrix) -> int:
    """The parts gemv splits a row of w into on its device (kept per shape): part of the sums' order."""
    key = (w.blocks.device, w.type, w.n, w.k)
    if key not in _SPLITS:
        _SPLITS[key] = int(ops.splits(w.blocks, int(w.type), int(w.k)))
    return _SPLITS[key]


def gemv(x: torch.Tensor, w: GgufMatrix, ids: Optional[torch.Tensor] = None,
         out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """y [M][N] = x [M][K] (M 1 to 8, BF16, FP16 or FP32) times W^T, in x's dtype; with ids (int32 [M]), row m times
    expert ids[m] of w's [experts][N][row bytes]."""
    rows = x.shape[0]
    xq, xd, xs = quantize(x)
    yp = torch.empty(splits(w), rows, w.n, dtype=torch.float32, device=x.device)
    ops.gemv(w.blocks, int(w.type), int(w.k), xq, xd, xs, ids, yp)
    out = torch.empty(rows, w.n, dtype=x.dtype, device=x.device) if out is None else out
    ops.sum_parts(yp, out)
    return out


def gemm(x: torch.Tensor, w: GgufMatrix, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """y [M][N] = x [M][K] (any M) times W^T in x's dtype, each output gemv's bits for its row: the prompt GEMM on the
    s8 tensor cores, the sub-blocks decoded into shared memory."""
    rows = x.shape[0]
    xq, xd, xs = quantize(x, by_subblock=True)
    out = torch.empty(rows, w.n, dtype=x.dtype, device=x.device) if out is None else out
    ops.gemm(w.blocks, int(w.type), int(w.k), xq, xd, xs, splits(w), out)
    return out


def matmul(x: torch.Tensor, w: GgufMatrix, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """gemv for up to MAX_ROWS rows, else gemm: the same bits either way."""
    return gemv(x, w, out=out) if x.shape[0] <= MAX_ROWS else gemm(x, w, out=out)


def dequantize(w: GgufMatrix, dtype: torch.dtype = torch.bfloat16, expert: int = 0) -> torch.Tensor:
    """W [N][K] in dtype: the stored values (FP32 exactly as the format's reference decoder gives them, BF16 and FP16
    rounded once)."""
    out = torch.empty(w.n, w.k, dtype=dtype, device=w.blocks.device)
    ops.dequantize(w.blocks, int(w.type), int(w.k), int(expert), out)
    return out
