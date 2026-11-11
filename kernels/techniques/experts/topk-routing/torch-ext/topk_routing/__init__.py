"""Top-k expert routing of router logits in one kernel, a warp per token row (crit's kernel topk-routing, which crit's
topk-router plugin loads, built on this GPU with the engine's tuning or the default knobs): softmax or sigmoid scores
before or after selection, a correction bias that only changes the selection, renormalization and scaling; ties to the
lower expert id; NaN logits rank below every number (-inf included), so a row of NaN logits still picks valid distinct
experts, with weights 0. One launch, no host synchronization.

    ids, weights = route(logits, top, bias=None, softmax=True, before=False, renormalize=False)
"""
import torch

from ._ops import ops

__all__ = ["MAX_EXPERTS", "MAX_TOP", "route"]

MAX_EXPERTS, MAX_TOP = 1024, 32


def route(logits, top, bias=None, softmax=True, before=False, renormalize=False, epsilon=0.0, floor=0.0, scaling=1.0,
          ids=None, weights=None):
    """(ids [tokens][top] int64 in selection order, weights [tokens][top] FP32) of logits [tokens][experts] FP32."""
    tokens = logits.shape[0]
    ids = torch.empty(tokens, top, dtype=torch.int64, device=logits.device) if ids is None else ids
    weights = torch.empty(tokens, top, dtype=torch.float32, device=logits.device) if weights is None else weights
    ops.route(logits, int(top), bias, bool(softmax), bool(before), bool(renormalize), float(epsilon), float(floor),
              float(scaling), ids, weights)
    return ids, weights
