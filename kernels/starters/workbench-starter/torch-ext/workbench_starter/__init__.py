"""The kernel workbench's starter matmul (crit kernels new --starter --from this folder): y[columns][N] = x[columns][K]
. w[N][K] in BF16 or E4M3 with per-row and per-column scales, FP32 out, one warp per output row, a fixed order per
column. A correct starting point, not a fast one."""
import torch

from ._ops import ops

__all__ = ["run_bf16", "run_fp8"]


def run_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return ops.run_bf16(x, w)


def run_fp8(x: torch.Tensor, x_scale: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    return ops.run_fp8(x, x_scale.reshape(-1).contiguous(), w, w_scale.reshape(-1).contiguous())
