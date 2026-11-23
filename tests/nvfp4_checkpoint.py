"""A matrix stored as NVFP4 in ModelOpt's layout, for test checkpoints (torch): round to nearest, FP32 arithmetic.

`weight_global_scale` = 6 * 448 / amax(matrix), one E4M3FN scale per 16 row values from the block's amax, E2M1 codes
packed two to a byte (the even element in the low nibble); a value decodes as e2m1(code) * e4m3(scale) / global. The
same codec as inference/techniques/projections/quantized-projections/code/nvfp4.py, vectorized for a whole matrix.
"""
import torch

E4M3_MAX, E2M1_MAX, BLOCK = 448.0, 6.0, 16
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e2m1_codes(x):
    """4-bit codes (bit 3 the sign) of x rounded to the nearest E2M1 value, ties to the even code, saturating at 6."""
    m = x.abs()
    index = ((m > 0.25).to(torch.uint8) + (m >= 0.75) + (m > 1.25) + (m >= 1.75) + (m > 2.5) + (m >= 3.5) + (m > 5.0))
    return index.to(torch.uint8) | (((x < 0) & (index > 0)).to(torch.uint8) << 3)


def nvfp4(matrix):
    """{weight_packed, weight_scale (E4M3FN), weight_global_scale} of a [rows][cols] matrix, cols a multiple of 16."""
    matrix = matrix.float()
    rows, cols = matrix.shape
    amax = matrix.abs().amax()
    g = torch.where(amax > 0, (E2M1_MAX * E4M3_MAX) / amax, torch.ones_like(amax))
    blocks = matrix.reshape(rows, cols // BLOCK, BLOCK)
    scales = (blocks.abs().amax(dim=-1) * g / E2M1_MAX).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    decoded = scales.float()[..., None]
    codes = _e2m1_codes(torch.where(decoded > 0, blocks * g / torch.where(decoded > 0, decoded, 1.0), 0.0))
    codes = codes.reshape(rows, cols)
    return {"weight_packed": (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(),
            "weight_scale": scales.contiguous(), "weight_global_scale": g.reshape(1).contiguous()}
