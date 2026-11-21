"""Latent attention (multi-head latent attention, absorbed): decode tokens and verify windows over a paged cache of
compressed rows, one row a position holding the latent (rank R) and then the rotary key (width P), shared by every
query head; the latent is each position's value too, so a piece of rows is read once for the scores and the values.

    write(latent, rope, slots, cache)                  # rows into their page slots: cache [pages][page][R + P]
    out = attend(q, cache, tables, rows, starts, firsts, columns, page, rank=R)
    # q [tokens][heads][R + P]: the query's latent part (the key up-projection absorbed) then its rotary part;
    # out [tokens][heads][R]: the attention output in the latent space (the value up-projection follows outside)

Row b of a launch is table row rows[b]'s sequence; its columns are tokens starts[b] .. starts[b] + counts[b] - 1 at
positions firsts[b] + column, each seeing the positions up to its own (causal inside the window, up to MAX_COLUMNS
columns). The scores are scale (q_latent . latent + q_rope . rope) in FP32 from BF16 or FP16 operands on the tensor
cores, the probabilities in the operands' dtype for the value product, the softmax in FP32. A row's positions split
into pieces of PIECE positions, piece i to partition i mod PARTS (both order knobs, one value a shape: a token's bits
depend on its own query and positions, the piece and the partition count, never on the rows or columns beside it),
each partition's state folded by the merge in partition order; attend(..., fused=True) folds them in the reading
program itself (the same bits, no partials: wide windows, and prompt chunks read as windows of MAX_COLUMNS columns
with their tokens' decode bits). sinks: FP32 [heads], each head's sink logit (natural
log, in the scaled scores' units; -inf: none), one more position with no value, folded in last. lse: FP32
[tokens][heads], the natural log-sum-exp of each token's scaled scores (the sink counted). Every allocation comes from
torch's caching allocator and nothing syncs with the host, so CUDA graphs capture a call once its shape has run.

Knobs (structure window, keyed by R, P, heads, page and the cache kind, KIND 0 16-bit or 1 E4M3): BLOCK_M (query rows a program: a column's heads), PIECE and
PARTS (order), WARPS and STAGES; PARTS 0 is the rule: about two programs a multiprocessor for a decode row, at most
32, from the heads and the device only.
"""
from typing import Optional

import torch
import triton

from . import _crit_tuning, _kernels

__all__ = ["MAX_COLUMNS", "PARTIAL_BYTES", "attend", "partitions", "prefill", "write"]

MAX_COLUMNS = 64
PARTIAL_BYTES = 512 << 20   # a split launch's partials past this take the fused launch (attend's fused=None)
FP8 = torch.float8_e4m3fn
MAX_PARTS = 128
_KNOBS = {}
_SMS = {}


def _knobs(key, rows, structure="window"):
    if getattr(_crit_tuning, "SELECTED", None):
        return _crit_tuning.knobs(structure, key, rows)
    at = (structure, rows, *key.values())
    if at not in _KNOBS:
        _KNOBS[at] = _crit_tuning.knobs(structure, key, rows)
    return _KNOBS[at]


def _hardware_fp8(device):
    """The GPU converts E4M3 itself (compute capability 8.9 on)."""
    return torch.cuda.get_device_capability(device) >= (8, 9)


def _multiprocessors(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SMS:
        _SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _SMS[index]


def partitions(device, heads, rank, rope, page, kind=0):
    """The partitions a launch over this geometry splits a row's positions into: the tuning's PARTS, else the rule
    (about two programs a multiprocessor for one decode token: 2 x SMs over the heads' 16-row tiles, 1 to 32)."""
    parts = int(_knobs({"R": rank, "P": rope, "H": heads, "PAGE": int(page), "KIND": kind}, 1)["PARTS"])
    if parts:
        return parts
    return max(1, min(MAX_PARTS, (2 * _multiprocessors(device)) // max(1, -(-heads // 16))))


def write(latent, rope, slots, cache, scales=None):
    """latent [N][R] and rope [N][P] into their slots (int64 [N], page * page_size + offset) of cache
    [pages][page][R + P], in the cache's dtype (device work only; capture-safe). An E4M3 cache (float8_e4m3fn) takes
    scales FP32 [pages][page][2]: each position's latent and rotary key coded at their own scale, max |x| / 448 (at
    least 1e-30), the codes the nearest E4M3 values of x / scale (IEEE divisions)."""
    rank = latent.shape[-1]
    if cache.shape[-1] != rank + rope.shape[-1] or latent.shape[0] != rope.shape[0] or slots.shape[0] != latent.shape[0]:
        raise ValueError("a cache row is the latent then the rotary key, one row a slot")
    slots = slots.long()
    if cache.dtype == FP8:
        if scales is None or tuple(scales.shape) != (*cache.shape[:2], 2) or scales.dtype != torch.float32:
            raise ValueError("an E4M3 cache takes scales FP32 [pages][page][2] (the latent's, the rotary key's)")
        parts, found = [], []
        for x in (latent.float(), rope.float()):
            amax = x.abs().amax(-1)
            s = (amax / torch.full_like(amax, 448.0)).clamp(min=1e-30)
            parts.append((x / s[:, None]).clamp(-448.0, 448.0).to(FP8))
            found.append(s)
        codes = torch.cat(parts, dim=-1).view(torch.uint8)   # index_copy_ takes bytes, not E4M3
        cache.view(torch.uint8).view(-1, cache.shape[-1]).index_copy_(0, slots, codes)
        scales.view(-1, 2).index_copy_(0, slots, torch.stack(found, dim=-1))
        return
    rows = torch.cat([latent.to(cache.dtype), rope.to(cache.dtype)], dim=-1)
    cache.view(-1, cache.shape[-1]).index_copy_(0, slots, rows)


def _check(q, cache, page, rank, sinks, scales):
    """(R, P, fp8) of a call, its inputs checked."""
    T, H, width = q.shape
    R, P = int(rank), width - int(rank)
    if cache.dim() != 3 or cache.shape[-1] != width or cache.shape[1] != page or not cache.is_contiguous():
        raise ValueError(f"the cache is contiguous [pages][{page}][{width}]: the latent then the rotary key")
    fp8 = cache.dtype == FP8
    if (q.dtype != cache.dtype and not fp8) or q.dtype not in (torch.bfloat16, torch.float16) or q.stride(-1) != 1:
        raise ValueError("q is BF16 or FP16 in the cache's dtype (either over an E4M3 cache), its last dimension "
                         "contiguous")
    if fp8 and (scales is None or tuple(scales.shape) not in ((*cache.shape[:2], 2), (*cache.shape[:2], 1)) or
                scales.dtype != torch.float32 or not scales.is_contiguous()):
        raise ValueError("an E4M3 cache takes its scales FP32 [pages][page][2] (or [pages][page][1]: one a row)")
    if R < 16 or P < 16 or R & (R - 1) or P & (P - 1) or R > 1024 or P > 256:
        raise ValueError("the rank and the rotary width are powers of two (16 to 1024 and 16 to 256)")
    if sinks is not None and (sinks.dtype != torch.float32 or tuple(sinks.shape) != (H,) or not sinks.is_contiguous()
                              or sinks.device != q.device):
        raise ValueError(f"sinks are contiguous FP32 [{H}] on q's device")
    return R, P, fp8


def prefill(q, cache, table, position, page, *, rank, scale=None, out: Optional[torch.Tensor] = None,
            lse: Optional[torch.Tensor] = None, sinks: Optional[torch.Tensor] = None,
            scales: Optional[torch.Tensor] = None) -> torch.Tensor:
    """A prompt chunk of one sequence over the latent cache (structure prompt): q [T][heads][R + P] at positions
    position .. position + T - 1 (their rows written), table [pages] int32 its page table, position an int or a
    one-element int64 device tensor read on the device (one CUDA graph for every position) -> out [T][heads][R] in q's
    dtype (and lse [T][heads] when given), causal. The chunk's tokens times their heads are packed into tiles of
    BLOCK_M rows, so each piece of cached rows is read once for BLOCK_M of them; the pieces go in order through one
    online softmax and the output is written directly (no partitions, no partials: a token's bits do not depend on the
    chunk's length; another reduction than attend()'s). Sinks, lse, scale and an E4M3 cache's scales as attend()."""
    T, H, width = q.shape
    R, P, fp8 = _check(q, cache, page, rank, sinks, scales)
    if T < 1 or table.dim() != 1:
        raise ValueError("a prompt chunk is one token or more over one page table [pages]")
    device = q.device
    knobs = _knobs({"R": R, "P": P, "H": H, "PAGE": int(page), "KIND": int(fp8)}, T, "prompt")
    vsplit = int(knobs["VSPLIT"])
    bm = int(knobs["BLOCK_M"])
    out = torch.empty(T, H, R, dtype=q.dtype, device=device) if out is None else out
    on_device = torch.is_tensor(position)
    codes = cache.view(torch.uint8) if fp8 else cache
    dummy = out if lse is None else lse
    grid = (triton.cdiv(T * H, bm) * vsplit,)
    _kernels._prompt_kernel[grid](q, codes, scales if fp8 else dummy, table, position if on_device else table, out,
                                  lse if lse is not None else dummy, sinks if sinks is not None else dummy, table,
                                  table, T, 0 if on_device else int(position), q.stride(0), q.stride(1), out.stride(0),
                                  out.stride(1), 0, float(scale if scale is not None else width ** -0.5), H=H, R=R, P=P,
                                  BLOCK_M=bm, PIECE=int(knobs["PIECE"]), PAGE=page, VSPLIT=vsplit, FP8=fp8,
                                  HW_FP8=_hardware_fp8(device), SCALES=int(scales.shape[-1]) if fp8 else 2,
                                  POS_T=on_device, LSE=lse is not None, SINKS=sinks is not None, RAGGED=False,
                                  num_warps=int(knobs["WARPS"]), num_stages=int(knobs["STAGES"]))
    return out


_ragged_plans = {}   # (device, BLOCK_M, heads, starts, counts, positions) -> (tile map, metadata)


def _ragged_plan(device, block_m, heads, starts, counts, positions):
    """The tile map ([tiles][2] int32: the sequence, its tile) and metadata ([sequences][3] int64: first token, tokens,
    first position) of a ragged prompt launch, made once a set of rows (every layer of a step takes the same) and kept
    for the last 64 sets."""
    key = (str(device), int(block_m), int(heads), tuple(starts), tuple(counts), tuple(positions))
    found = _ragged_plans.get(key)
    if found is None:
        tiles = [(r, m) for r, n in enumerate(counts) for m in range(triton.cdiv(int(n) * heads, block_m))]
        meta = torch.tensor([[s, n, p] for s, n, p in zip(starts, counts, positions)], dtype=torch.int64)
        found = (torch.tensor(tiles, dtype=torch.int32).to(device, non_blocking=True),
                 meta.to(device, non_blocking=True))
        if len(_ragged_plans) >= 64:
            _ragged_plans.pop(next(iter(_ragged_plans)))
        _ragged_plans[key] = found
    return found


def prefill_rows(q, cache, tables, starts, counts, positions, page, *, rank, scale=None,
                 out: Optional[torch.Tensor] = None, lse: Optional[torch.Tensor] = None,
                 sinks: Optional[torch.Tensor] = None, scales: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Several sequences' prompt chunks in one launch (structure prompt): row i's queries q[starts[i] : starts[i] +
    counts[i]] at positions positions[i] .. (host ints, their rows written), its page table tables[i] (int32 [rows]
    [width]), causal; out, lse, sinks, scale and scales as prefill(). Each row's tiles are the ones prefill() runs for
    the row alone, so a token has the bits prefill() gives it (the knobs are those of the call's tokens together: the
    order knob PIECE is one a shape). Not under a capture (the rows are host values)."""
    T, H, width = q.shape
    R, P, fp8 = _check(q, cache, page, rank, sinks, scales)
    starts, counts, positions = [int(x) for x in starts], [int(x) for x in counts], [int(x) for x in positions]
    if (not (len(starts) == len(counts) == len(positions) == tables.shape[0]) or tables.dim() != 2 or
            tables.dtype != torch.int32 or tables.stride(1) != 1):
        raise ValueError("tables are int32 [rows][width], one row a chunk, as many as starts, counts and positions")
    if not starts or any(x < 0 or n < 1 or x + n > T for x, n in zip(starts, counts)):
        raise ValueError("each row's tokens lie in q, one row or more")
    device = q.device
    knobs = _knobs({"R": R, "P": P, "H": H, "PAGE": int(page), "KIND": int(fp8)}, T, "prompt")
    vsplit, bm = int(knobs["VSPLIT"]), int(knobs["BLOCK_M"])
    tiles, meta = _ragged_plan(device, bm, H, starts, counts, positions)
    out = torch.empty(T, H, R, dtype=q.dtype, device=device) if out is None else out
    codes = cache.view(torch.uint8) if fp8 else cache
    dummy = out if lse is None else lse
    _kernels._prompt_kernel[(tiles.shape[0] * vsplit,)](
        q, codes, scales if fp8 else dummy, tables, tables, out, lse if lse is not None else dummy,
        sinks if sinks is not None else dummy, tiles, meta, T, 0, q.stride(0), q.stride(1), out.stride(0),
        out.stride(1), tables.stride(0), float(scale if scale is not None else width ** -0.5), H=H, R=R, P=P,
        BLOCK_M=bm, PIECE=int(knobs["PIECE"]), PAGE=page, VSPLIT=vsplit, FP8=fp8, HW_FP8=_hardware_fp8(device),
        SCALES=int(scales.shape[-1]) if fp8 else 2, POS_T=False, LSE=lse is not None, SINKS=sinks is not None,
        RAGGED=True, num_warps=int(knobs["WARPS"]), num_stages=int(knobs["STAGES"]))
    return out


def attend(q, cache, tables, rows, starts, firsts, columns, page, *, rank, counts=None, scale=None,
           out: Optional[torch.Tensor] = None, lse: Optional[torch.Tensor] = None,
           sinks: Optional[torch.Tensor] = None, scales: Optional[torch.Tensor] = None,
           fused: Optional[bool] = None) -> torch.Tensor:
    """Decode tokens and verify windows over the latent cache (see the module's description). q [tokens][heads][R + P]
    BF16 or FP16 in the cache's dtype, the last dimension contiguous; cache [pages][page][R + P] contiguous; tables
    [S][width] int32; rows, starts [B] int32; firsts [B] int64; counts [B] int32 (None: `columns` each); columns the
    widest row (a Python int, 1 to MAX_COLUMNS). Returns out [tokens][heads][R] in q's dtype (written at the rows'
    tokens); lse [tokens][heads] FP32 when given; sinks FP32 [heads]. scale defaults to (R + P) ** -0.5 (pass the
    model's: the softmax scale of its un-absorbed query and key width). An E4M3 cache takes its scales, with BF16 or
    FP16 queries: write()'s two a position, or one a position for the whole row ([pages][page][1], a KV pool's fp8:row
    pages of one head). fused: False, a program a partition and the merge over their partials (FP32, rank wide, a
    query head, column and partition: the launch for decode rows, whose partitions fill the device); True, a program
    a tile that reads its partitions in turn and folds them itself, the same bits with no partials (wide windows, and
    a prompt chunk read as windows of MAX_COLUMNS columns, whose tokens then have their decode bits); None, fused
    where the partials would take more than PARTIAL_BYTES."""
    T, H, width = q.shape
    R, P = int(rank), width - int(rank)
    B = int(starts.shape[0])
    if cache.dim() != 3 or cache.shape[-1] != width or cache.shape[1] != page or not cache.is_contiguous():
        raise ValueError(f"the cache is contiguous [pages][{page}][{width}]: the latent then the rotary key")
    fp8 = cache.dtype == FP8
    if (q.dtype != cache.dtype and not fp8) or q.dtype not in (torch.bfloat16, torch.float16) or q.stride(-1) != 1:
        raise ValueError("q is BF16 or FP16 in the cache's dtype (either over an E4M3 cache), its last dimension "
                         "contiguous")
    if fp8 and (scales is None or tuple(scales.shape) not in ((*cache.shape[:2], 2), (*cache.shape[:2], 1)) or
                scales.dtype != torch.float32 or not scales.is_contiguous()):
        raise ValueError("an E4M3 cache takes its scales FP32 [pages][page][2] (or [pages][page][1]: one a row)")
    if R < 16 or P < 16 or R & (R - 1) or P & (P - 1) or R > 1024 or P > 256:
        raise ValueError("the rank and the rotary width are powers of two (16 to 1024 and 16 to 256)")
    if not 1 <= columns <= MAX_COLUMNS or B == 0:
        raise ValueError(f"1..{MAX_COLUMNS} columns a row, one row or more")
    if sinks is not None and (sinks.dtype != torch.float32 or tuple(sinks.shape) != (H,) or not sinks.is_contiguous()
                              or sinks.device != q.device):
        raise ValueError(f"sinks are contiguous FP32 [{H}] on q's device")
    device = q.device
    key = {"R": R, "P": P, "H": H, "PAGE": int(page), "KIND": int(fp8)}
    knobs = _knobs(key, columns)
    parts = partitions(device, H, R, P, page, int(fp8))
    bm = max(16, min(int(knobs["BLOCK_M"]), triton.next_power_of_2(columns * H)))
    piece = int(knobs["PIECE"])
    out = torch.empty(T, H, R, dtype=q.dtype, device=device) if out is None else out
    partial = B * columns * H * parts
    scale = float(scale if scale is not None else width ** -0.5)
    count_ptr = counts if counts is not None else starts
    vsplit = int(knobs["VSPLIT"])
    codes = cache.view(torch.uint8) if fp8 else cache
    if fused is None:
        fused = partial * (R + 2) * 4 > PARTIAL_BYTES
    if fused:
        dummy = out if lse is None else lse
        _kernels._fused_kernel[(triton.cdiv(columns * H, bm) * vsplit, B)](
            q, codes, scales if fp8 else dummy, tables, rows, starts, count_ptr, firsts, out,
            lse if lse is not None else dummy, sinks if sinks is not None else dummy, q.stride(0), q.stride(1),
            tables.stride(0), tables.shape[1], parts, scale, columns, out.stride(0), out.stride(1), H=H, R=R, P=P,
            COLS=columns, BLOCK_M=bm, PIECE=piece, PAGE=page, COUNTS=counts is not None, VSPLIT=vsplit, FP8=fp8,
            HW_FP8=_hardware_fp8(device), SCALES=int(scales.shape[-1]) if fp8 else 2, LSE=lse is not None,
            SINKS=sinks is not None, num_warps=int(knobs["WARPS"]), num_stages=int(knobs["STAGES"]))
        return out
    po = torch.empty(partial * R, dtype=torch.float32, device=device)
    pm = torch.empty(partial, dtype=torch.float32, device=device)
    pl = torch.empty(partial, dtype=torch.float32, device=device)
    grid = (triton.cdiv(columns * H, bm) * vsplit, parts, B)
    _kernels._window_kernel[grid](q, codes, scales if fp8 else pm, tables, rows, starts, count_ptr, firsts, po, pm, pl, q.stride(0),
                                  q.stride(1), tables.stride(0), tables.shape[1], parts, scale, columns, H=H, R=R, P=P,
                                  COLS=columns, BLOCK_M=bm, PIECE=piece, PAGE=page, COUNTS=counts is not None,
                                  VSPLIT=vsplit, FP8=fp8, HW_FP8=_hardware_fp8(device),
                                  SCALES=int(scales.shape[-1]) if fp8 else 2, num_warps=int(knobs["WARPS"]), num_stages=int(knobs["STAGES"]))
    _kernels._merge_kernel[(columns, H, B)](po, pm, pl, out, lse if lse is not None else pm,
                                            sinks if sinks is not None else pm, starts, count_ptr, firsts, parts,
                                            out.stride(0), out.stride(1), columns, H=H, R=R, COLS=columns,
                                            PIECE=piece, COUNTS=counts is not None, LSE=lse is not None,
                                            SINKS=sinks is not None, CHUNK_P=min(8, triton.next_power_of_2(parts)),
                                            num_warps=4)
    return out
