"""Recurrent layers of hybrid models on CUDA: a short causal depthwise convolution with its carried window, gated
delta attention (Qwen3-Next / Qwen3.5 gated DeltaNet) and Mamba-2's state-space recurrence. Each has a chunked
two-pass prefill for prompts and a token-by-token scan for decode and speculative verify; the scan is column
invariant (a token decoded alone equals the same token inside a multi-token verify, bit for bit) and can write the
state after every token. All tensors are float32 on the GPU; the state is updated in place."""
from typing import Optional

import torch

from ._ops import ops

__all__ = ["causal_conv1d", "gated_delta_prefill", "gated_delta_scan", "ssd_prefill", "ssd_scan",
           "STASH", "CHECKPOINTS", "RECOMPUTE", "FINAL", "FLUSH", "REPLAY", "FLUSH_PENDING",
           "pool_blocks", "pool_half_floats", "pool_conv_blocks", "pool_delta", "pool_delta_replay", "pool_conv",
           "conv_segments", "pool_conv_prompt", "pool_conv_supported"]


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor, bias: Optional[torch.Tensor] = None,
                  silu: bool = True, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x [T, channels], weight [channels, width]; state [channels, width - 1] holds the window before row 0 and
    receives the window after the last row. Returns act(bias + sum_i weight[:, i] * x[t - width + 1 + i])."""
    out = torch.empty_like(x) if out is None else out
    ops.causal_conv1d(out, state, x, weight, bias, silu)
    return out


def _delta_out(v, out):
    return torch.empty_like(v) if out is None else out


def gated_delta_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, log_decay: torch.Tensor, beta: torch.Tensor,
                        state: torch.Tensor, scale: float = 1.0, out: Optional[torch.Tensor] = None,
                        tensor_cores: bool = False) -> torch.Tensor:
    """Chunked gated delta over a prompt. q, k [T, k_heads, dk] (normalized keys and queries), v [T, v_heads, dv],
    log_decay (<= 0) and beta [T, v_heads], state [v_heads, dk, dv] updated in place. Per token:
    delta = beta (v - e^l S^T k); S' = e^l S + k delta^T; out = scale S'^T q. tensor_cores: the chunked WY form on
    tensor cores (structure delta-prefill-tc: FP16 operands, FP32 accumulation and state, values and states within
    FP16's range; dk 64 or 128, dv a multiple of 16), else the FP32 two-pass prefill (dv a multiple of 64)."""
    out = _delta_out(v, out)
    (ops.gated_delta_prefill_tc if tensor_cores else ops.gated_delta_prefill)(out, state, q, k, v, log_decay, beta, scale)
    return out


def gated_delta_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, log_decay: torch.Tensor, beta: torch.Tensor,
                     state: torch.Tensor, scale: float = 1.0, checkpoints: Optional[torch.Tensor] = None,
                     out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The same recurrence token by token (decode and verify); checkpoints [T, v_heads, dk, dv] receives the state
    after every token when given."""
    out = _delta_out(v, out)
    ops.gated_delta_scan(out, state, q, k, v, log_decay, beta, scale, checkpoints)
    return out


def ssd_prefill(x: torch.Tensor, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor, A: torch.Tensor,
                state: torch.Tensor, D: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Chunked Mamba-2 over a prompt. x [T, heads, head_dim], dt [T, heads] (after softplus and bias), B, C
    [T, groups, d_state], A [heads] (< 0), D [heads], state [heads, head_dim, d_state] updated in place.
    Per token: H' = e^{dt A} H + dt x B^T; y = H' C + D x."""
    out = torch.empty_like(x) if out is None else out
    ops.ssd_prefill(out, state, x, dt, B, C, A, D)
    return out


def ssd_scan(x: torch.Tensor, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor, A: torch.Tensor, state: torch.Tensor,
             D: Optional[torch.Tensor] = None, checkpoints: Optional[torch.Tensor] = None,
             out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mamba-2 token by token; checkpoints [T, heads, head_dim, d_state] receives the state after every token."""
    out = torch.empty_like(x) if out is None else out
    ops.ssd_scan(out, state, x, dt, B, C, A, D, checkpoints)
    return out


# ------------------------------------------------------------------------------------------------- pools of sequences
# Gated delta and the causal convolution over a pool of sequences (recurrent_rows.h): the decode and verify rows of many
# sequences in one launch, each from its own sequence's state, and the device commit of a speculative round's accepted
# prefix's arithmetic (the commit's bookkeeping, writing pending counts or selecting a checkpoint, is the caller's).
# Row kinds:
STASH, CHECKPOINTS, RECOMPUTE, FINAL, FLUSH = 0, 1, 2, 3, 4
# and modes of pool_delta_replay:
REPLAY, FLUSH_PENDING = 2, 3
_COLUMNS, _CONV_THREADS = 32, 128


def pool_blocks(v_heads: int, dv: int) -> int:
    """Blocks of a layer's state per sequence: the pending counts (then as many stash parities) its control holds."""
    return v_heads * ((dv + _COLUMNS - 1) // _COLUMNS)


def pool_half_floats(k_heads: int, v_heads: int, dk: int, dv: int, capacity: int) -> int:
    """Floats of one stash half: key [capacity][k_heads][dk], decay [capacity][v_heads], delta [capacity][v_heads][dv]."""
    return capacity * (k_heads * dk + v_heads + v_heads * dv)


def pool_conv_blocks(channels: int) -> int:
    return (channels + _CONV_THREADS - 1) // _CONV_THREADS


def pool_delta(out, states, stash, control, scratch, q, k, v, log_decay, beta, meta=None, starts=None, slots=None,
               table=None, rows=0, kind=STASH, layer=0, capacity=1, epsilon=1e-6, scale=1.0, normalize_keys=True,
               normalize_queries=True, pipelined=False):
    """One layer's rows: each row continues its sequence's state (states[layer][sequence], after its pending count)
    over its tokens; out [T][v_heads][dv] FP32. q, k [T][k_heads][dk] and v [T][v_heads][dv] in FP32, BF16 or FP16
    (rows contiguous), log_decay and beta [T][v_heads] FP32. Rows from meta ([rows][5] int32: first token, tokens,
    sequence or -1, kind, scratch slot or -1) or a round's starts ([rows + 1] int32), slots ([rows] int64) and table
    ([slots] int32 -> sequence or -1) with every row of `kind`. scratch [slots][...]: checkpoints [capacity + 1][state],
    or the start or final state, by the meta row's scratch slot or the round slot; a FINAL, CHECKPOINTS or RECOMPUTE row
    without scratch, and a row of no sequence, write zero outputs and touch no state. pipelined: the pool-delta-pipe
    structure (the same bits on another schedule: state tiles staged ahead in shared memory, a block running several
    slots one after another). Capture-safe."""
    ops.pool_delta(out, states, stash, control, scratch, q, k, v, log_decay, beta, meta, starts, slots, table, rows, kind,
                   layer, capacity, epsilon, scale, normalize_keys, normalize_queries, pipelined)
    return out


def pool_delta_replay(states, stash, control, scratch, accepted, meta=None, slots=None, table=None, rows=0, columns=0,
                      capacity=1, key_heads=1, mode=REPLAY):
    """Every layer's replay for each row: REPLAY replays accepted[r] ([rows] int64 on the device, clamped to
    0 .. columns) stashed steps from the start state in scratch [layers][slots][state] into the live state (the
    recompute commit), FLUSH_PENDING applies and consumes each row's sequence's pending count (accepted and scratch
    unused). A row's pending count is control[l][sequence][:pool_blocks]: the caller writes it to commit deferred.
    Capture-safe."""
    ops.pool_delta_replay(states, stash, control, scratch, accepted, meta, slots, table, rows, columns, capacity, key_heads,
                          mode)


def pool_conv_supported(width: int, dilation: int = 1) -> bool:
    """Whether the pool convolutions run `width` taps at `dilation`: 2 .. 8 taps undilated, 2 .. 4 taps at dilation
    2 .. 4. A window holds the last (width - 1) * dilation inputs of each channel; the kernels read the dilation from
    its width."""
    if dilation == 1:
        return 2 <= width <= 8
    return 2 <= width <= 4 and 2 <= dilation <= 4


def pool_conv(out, window, stash, pending, scratch, x, weight, bias=None, meta=None, starts=None, slots=None, table=None,
              rows=0, kind=STASH, layer=0, layers=1, capacity=1, silu=True):
    """The causal depthwise convolution of rows over their sequences' windows (window[layer][sequence], after the
    pending count): x and out [T][channels] (x's dtype; out not x), weight [channels][width] FP32; window
    [layers][sequences][channels][(width - 1) * dilation] (its last dimension sets the dilation, pool_conv_supported):
    y[t] = act(bias + sum_i weight[i] x[t - (width - 1 - i) * dilation]). STASH rows keep
    their inputs for the deferred commit, FINAL rows write the window after their last input to scratch [slots][...],
    FLUSH rows (over `layers` layers from `layer`) apply the pending count only. Capture-safe."""
    ops.pool_conv(out, window, stash, pending, scratch, x, weight, bias, meta, starts, slots, table, rows, kind, layer,
                  layers, capacity, silu)
    return out


def conv_segments(out, window_in, window_out, x, weight, bias=None, silu=True):
    """One long row out of place from window_in ([channels][(width - 1) * dilation]: its width sets the dilation),
    the window after its last input to window_out: a thread per channel and 32 rows, the bits of pool_conv's."""
    ops.conv_segments(out, window_in, window_out, x, weight, bias, silu)
    return out


def pool_conv_prompt(out, window, stash, pending, x, weight, bias=None, meta=None, layer=0, capacity=1, silu=True):
    """One sequence's prompt chunk padded to x's rows (a captured graph's bucket): meta [5] int32 on the device (first,
    count, sequence, kind, slot; count and sequence read) says how many rows are real and whose window they continue.
    The real rows convolve from window[layer][sequence] after its pending count (applied and consumed), the padding
    rows output zeros, and the window after the real inputs is written in place; a sequence below 0 outputs zeros and
    touches no window. The bits of pool_conv's and conv_segments'. Capture-safe: the same launch for any chunk."""
    ops.pool_conv_prompt(out, window, stash, pending, x, weight, bias, meta, layer, capacity, silu)
    return out
