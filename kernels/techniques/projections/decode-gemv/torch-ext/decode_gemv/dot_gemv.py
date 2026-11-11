"""A Triton FP8 W8A16 decode GEMV on tensor cores for 1 to 16 rows: a program multiplies a [16 rows x BK] block of the
activations by a [BN x BK] block of the E4M3 weights (exact in BF16) with tl.dot, FP32 sums, over its part of K; with
`split` parts the programs write FP32 partial sums and a second kernel adds them in part order and applies the row
scale. The program's shape never depends on the call's row count (rows past it are masked to zero), so a row's bits
are the same alone and beside up to 15 others; tile() picks the shape from the build's tuning for the matrix's
[N, K], or untuned from that shape and the device's multiprocessor count (rule()), so a process keeps one order a
matrix. Adapted from a crit-labs engine's Triton decode GEMV (Apache-2.0), as general code. The kernel is the
format-generic Triton GEMV's dot path (triton_any, structure gemv-triton) on FP8, the same operations as before:
structure gemv-fp8-dot is an alias of gemv-triton."""
import torch

from ._crit_tuning import knobs
from .triton_any import dot, dot_rule, partial_elements as _partial_elements

MAX_ROWS = 16


def tile(n: int, k: int, multiprocessors: int):
    """(BN, BK, warps, split) for an [N, K] FP8 matrix: the build's tuning for this shape (kernel.json's structure
    gemv-fp8-dot, one plan a matrix for every row count), or, untuned, rule()'s. A tuned BK and split that do not
    divide K give the rule's plan."""
    if k % 128:
        raise ValueError(f"the FP8 dot GEMV takes K a multiple of 128, not {k}")
    tuned = knobs("gemv-fp8-dot", {"N": n, "K": k}, 1)
    plan = (tuned["BN"], tuned["BK"], tuned["WARPS"], tuned["SPLIT"])
    if 0 in plan or k % (plan[1] * plan[3]):
        return rule(n, k, multiprocessors)
    return plan


def rule(n: int, k: int, multiprocessors: int):
    """(BN, BK, warps, split) from the matrix's shape and the multiprocessor count alone (triton_any.dot_rule)."""
    return dot_rule(n, k, multiprocessors)


def partial_elements(rows: int, n: int, split: int) -> int:
    """FP32 elements the partial sums of a call take (0 for one part)."""
    return _partial_elements(rows, n, split)


def gemv_fp8_dot(x: torch.Tensor, w: torch.Tensor, row_scale: torch.Tensor, out: torch.Tensor, plan, partial=None):
    """out [M][N] (any float dtype, unit column stride) = x [M][K] (M 1..16, BF16 or FP16, unit column stride) times
    w [N][K] E4M3 (float8_e4m3fn or uint8) with row_scale [N] FP32 after the sum; plan: tile()'s; partial: FP32 with
    partial_elements(M, N, split) elements or more when split > 1 (a call allocates nothing then)."""
    bn, bk, warps, split = plan
    M, K = x.shape
    N = w.shape[0]
    if not 1 <= M <= MAX_ROWS or w.shape[1] != K or K % (split * bk) or x.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError(f"gemv_fp8_dot: x [1..{MAX_ROWS}][K] for w [N][K] with K a multiple of split * BK "
                         f"({split} x {bk}), not x {tuple(x.shape)} and w {tuple(w.shape)}")
    if split > 1 and (partial is None or partial.numel() < partial_elements(M, N, split)
                      or partial.dtype != torch.float32):
        raise ValueError(f"gemv_fp8_dot: {split} parts of K take an FP32 partial buffer of "
                         f"{partial_elements(M, N, split)} elements or more")
    # BF16 operands for either activation type, as this GEMV had (gemv_any takes FP16 operands for FP16 activations)
    return dot(1, x, w.view(torch.uint8), None, None, row_scale, out, 0, plan, partial, dot_dtype=torch.bfloat16)
