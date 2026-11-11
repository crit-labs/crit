"""Projections of 1 to 64 activation rows on tensor cores in one fixed tile, from crit's quantized-projections
technique: an NVFP4, FP8, MXFP8 or BF16 matrix [N][K] times BF16 rows [M][K], the activations used as they are
(weight only: W4A16, W8A16) or quantized as a prompt's are (FP8 per row: W8A8; NVFP4 per 16-value block: W4A4), FP32
sums. Every value enters mma.m16n8k16 exactly (BF16 operands, FP32 accumulate) and each weight is read once a call.
The order of every output's sum depends on N, K and `splits` only: a row gets the same bits alone, in a decode call of
up to 16 rows or in a prompt's tile of up to 64. plan() picks `groups`, `warps` (speed only) and `splits` (the sums'
order) from the build's tuning for the matrix (kernel.json's structures decode-tile and fp4-tile), or untuned from the
matrix and the device, so a process keeps one plan per matrix.

Quantized activations go through a pre-pass into `scratch` (scratch_bytes); splits > 1 add their partial sums in
`partial` (partial_elements) with one int32 counter a row tile in `counters` (row_tiles(N, 1, warps): calls of more
than 32 rows take one group a warp; zeros, left zero). Calls that share those buffers run in stream order; nothing allocates and nothing waits for the host, so a CUDA graph captures a
call. The activations' quantizers alone (quantize_fp8_rows, quantize_nvfp4) give a provider's GEMM the same
activations (cuBLASLt's FP8 and NVFP4 block-scaled GEMMs for prompt rows).

Matrices as checkpoints store them: NVFP4 codes [N][K / 2] uint8 (element 2i in the low nibble of byte i), E4M3 block
scales [N][K / 16] row major or in cuBLASLt's 128 x 4 tiled layout (blocked), row_scale [N] FP32: one over each row's
global scale. FP8: E4M3 codes [N][K] and row_scale [N]. MXFP8: E4M3 codes [N][K] and E8M0 scales [N][K / 32]. BF16
[N][K]. INT4 with a zero point a group (int4_zp: AWQ, GPTQ in crit's rows, decode-gemv's repack_awq / repack_gptq):
codes [N][K / 2] uint8 (unsigned, element 2i in the low nibble of byte i), FP16 scales and uint8 zeros [N][K / group]
(the group a multiple of 32 dividing K), a value (code - zero) * scale: it does not fit BF16, so the decoder gives it as
two BF16 parts and each MMA runs twice (hi, then lo), every product exact; BF16 activations. K is a multiple of 32."""
import math

import torch

from ._crit_tuning import knobs
from ._ops import ops

WEIGHTS = {"nvfp4": 0, "fp8": 1, "mxfp8": 2, "bf16": 3, "int4_zp": 4}
ACTS = {"bf16": 0, "fp8_rows": 1, "nvfp4": 2, "nvfp4_mma": 3}
FP4_STEP = 128     # K the FP4 MMA's step covers (its splits are of these steps)
MAX_ROWS = 64     # activation rows a call takes
STEP = 32         # K is a multiple of it
VALUES = {name: int(ops.step_values(code)) for name, code in WEIGHTS.items()}   # K a step takes (the kernel's)
MAX_WARPS, MAX_SPLITS = 4, 32
WARPS = 4          # warps a CTA, each its own 16 x groups matrix rows
PROMPT_TOKENS = 64   # activation rows a CTA of mm_prompt takes, untuned (structure prompt-tile's PT)

__all__ = ["ACTS", "FP4_STEP", "MAX_ROWS", "PROMPT_TOKENS", "STEP", "VALUES", "WEIGHTS", "eligible", "fp4_mma",
           "mm_prompt", "mm_rows", "partial_elements", "plan", "quantize_fp8_rows", "quantize_nvfp4", "row_tiles",
           "scratch_bytes"]


def eligible(rows: int, columns: int) -> bool:
    """Whether mm_rows takes a matrix of this shape (K a multiple of STEP)."""
    return rows >= 1 and columns >= STEP and columns % STEP == 0


def plan(rows: int, columns: int, weights: str, device=None, acts: str = "bf16"):
    """(groups, warps, splits) for a rows x columns matrix of `weights` (with `acts`) on `device`: the build's tuning
    for the matrix (structure fp4-tile for acts nvfp4_mma, else decode-tile; one plan for every row count), its
    splits capped at the matrix's steps. Untuned (the default knobs): one 16-row group a warp, WARPS warps a CTA, and
    the splits of K (of VALUES[weights] steps, of FP4_STEP on the FP4 MMA, acts nvfp4_mma) whose CTAs fill the
    multiprocessors' resident CTAs of the one-row kernel best (decode_ctas: that kernel's occupancy there, the decode
    tile's or the FP4 MMA's), the fewer splits on a tie; splits of one step only while every CTA fits one wave. A
    decode call is a stream of weights: CTAs that overrun a whole number of waves leave a short last wave the call
    waits for (a 44032 x 7168 NVFP4 matrix in 688 CTAs for 680 resident ran 10% slower than in seven splits), and
    waves of one-step splits leave a CTA nothing to keep in flight (a 7168 x 7168 one in 14 splits, 12% slower than in
    6). A tuning's SPLITS 0 keeps that rule for its groups and warps."""
    if weights not in WEIGHTS:
        raise ValueError(f"weights are {', '.join(WEIGHTS)}, not {weights!r}")
    if not eligible(rows, columns):
        raise ValueError(f"mm_rows takes K a multiple of {STEP}, not {columns}")
    if acts == "nvfp4_mma" and (weights != "nvfp4" or columns % FP4_STEP):
        raise ValueError(f"the FP4 MMA takes NVFP4 weights with K a multiple of {FP4_STEP}, not {weights} x {columns}")
    fp4 = acts == "nvfp4_mma"
    if fp4:
        tuned = knobs("fp4-tile", {"N": rows, "K": columns}, 1)
        groups = 1
    else:
        tuned = knobs("decode-tile", {"weights": WEIGHTS[weights], "N": rows, "K": columns}, 1)
        groups = tuned["GROUPS"]
    warps = tuned["WARPS"]
    steps = columns // FP4_STEP if fp4 else math.ceil(columns / VALUES[weights])
    if tuned["SPLITS"]:
        return groups, warps, max(1, min(tuned["SPLITS"], steps, MAX_SPLITS))
    device = torch.device(device or "cuda")
    with torch.cuda.device(device):
        resident = int(ops.decode_ctas(WEIGHTS[weights], ACTS["nvfp4_mma" if fp4 else "bf16"]))
    slots = torch.cuda.get_device_properties(device).multi_processor_count * resident
    tiles = row_tiles(rows, groups, warps)

    def filled(splits):   # the share of the resident CTAs a call's waves keep busy
        waves = tiles * splits / slots
        return waves if waves <= 1 else waves / math.ceil(waves)
    choices = [s for s in range(1, min(MAX_SPLITS, steps) + 1) if steps // s >= 2 or tiles * s <= slots] or [1]
    return groups, warps, max(choices, key=lambda s: (filled(s), -s))


def row_tiles(rows: int, groups: int, warps: int) -> int:
    return math.ceil(rows / (16 * groups * warps))


def scratch_bytes(tokens: int, columns: int, acts: str = "fp8_rows") -> int:
    """Bytes of scratch quantized activations take for `tokens` rows of `columns` (acts nvfp4_mma: their codes and
    blocked scales, rows padded to 128)."""
    if acts == "nvfp4_mma":
        return int(ops.fp4_scratch_bytes(tokens, columns))
    return (tokens * columns * 2 + 15) // 16 * 16 + tokens * 4


def fp4_mma(device=None) -> bool:
    """Whether acts nvfp4_mma (the activations in NVFP4 on the block-scaled FP4 MMA) runs on `device` with this build:
    the sm_120 family (sm_120a, sm_121a code). The first call per device runs a probe kernel and waits for it."""
    device = torch.device(device or "cuda")
    return device.type == "cuda" and bool(ops.fp4_mma(torch.empty(0, device=device)))


def partial_elements(tokens: int, rows: int, splits: int) -> int:
    return splits * tokens * rows if splits > 1 else 0


def mm_rows(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, weights: str, scales=None, row_scale=None,
            acts: str = "bf16", input_scale: float = 1.0, blocked: bool = False, groups: int = 1, warps: int = WARPS,
            splits: int = 1, scratch=None, partial=None, counters=None, zeros=None, group: int = 0):
    """out [M][N] (FP32, BF16 or FP16) = x [M][K] (M 1..64, BF16, rows on 16-byte boundaries) times the matrix w.
    acts: bf16 (x as it is), fp8_rows (FP8 weights: each row in E4M3 by its own scale amax * fp32(1 / 448), at least
    1e-12; out = sum * row_scale * the row's scale), nvfp4 (NVFP4 weights: each 16-value block in E2M1 under
    input_scale, the block scale e4m3(min(amax * fp32(1 / 6) * input_scale, 448)), multiplied as BF16 values; out =
    sum * row_scale * fp32(1 / input_scale)), nvfp4_mma (the same NVFP4 activations multiplied as codes with their
    block scales on the FP4 MMA, NVFP4 weights with blocked scales, K a multiple of FP4_STEP, where fp4_mma(); its own
    order). groups, warps, splits: plan()'s (with the acts). int4_zp weights: scales FP16 and zeros uint8 [N][K /
    group]."""
    ops.mm_rows(x, w, scales, row_scale, WEIGHTS[weights], ACTS[acts], float(input_scale), blocked, groups, warps,
                splits, scratch, partial, counters, out, zeros, int(group))
    return out


def mm_prompt(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, weights: str, scales=None, row_scale=None,
              acts: str = "bf16", input_scale: float = 1.0, blocked: bool = False, splits: int = 1, scratch=None,
              zeros=None, group: int = 0):
    """out [M][N] = x [M][K] times the matrix w for any number of rows M >= 1 (the prompt tile), every element with
    the bits mm_rows gives it for the same matrix, acts and splits (plan()'s, the matrix's): a prompt row equals the
    same row decoded alone or batched, so a prompt's tokens score as they are served. Each decoded weight serves
    PROMPT_TOKENS rows a CTA (a GEMM's reuse); quantized activations take scratch_bytes(M, K) of scratch."""
    ops.mm_prompt(x, w, scales, row_scale, WEIGHTS[weights], ACTS[acts], float(input_scale), blocked, splits, scratch,
                  out, zeros, int(group))
    return out


def quantize_fp8_rows(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor):
    """q [M][K] (float8_e4m3fn or uint8) and scale [M] FP32: BF16 rows in E4M3, each by its own scale, as mm_rows'
    fp8_rows activations (any M and K)."""
    ops.quantize_fp8_rows(x, q.view(torch.uint8), scale)
    return q, scale


def quantize_nvfp4(x: torch.Tensor, input_scale: float, codes: torch.Tensor, scales: torch.Tensor):
    """codes [M][K / 2] (uint8 or float4_e2m1fn_x2) and scales (E4M3 bytes, cuBLASLt's 128 x 4 tiled layout of
    [M][K / 16], rows padded to 128 and blocks to 4, the padding zero): BF16 rows in NVFP4 under input_scale, as
    mm_rows' nvfp4 activations (K a multiple of 16)."""
    ops.quantize_nvfp4(x, float(input_scale), codes.view(torch.uint8), scales.view(torch.uint8))
    return codes, scales
