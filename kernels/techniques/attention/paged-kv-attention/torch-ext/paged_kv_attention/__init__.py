"""Attention over paged KV caches in Triton: writes into pages, decode and speculative-verify windows split over KV
partitions merged by their log-sum-exp (with tree masks and block-sparse selection), and causal prompt chunks.

Pages. A cache layer is keys [pages][page][kv_heads][dim] and values [pages][page][kv_heads][value_dim]; position p of
a sequence lives at slot table[p // page] * page + p % page. Page kinds:

- "bf16", "fp16": 16-bit values;
- "fp8": E4M3 codes with one multiplier per layer for the keys and one for the values (value = code * multiplier);
- "fp8:row": E4M3 codes with an FP32 scale per position and KV head ([pages][page][kv_heads]): scale = max |x| / 448
  (at least 1e-30), code = x / scale rounded to the nearest E4M3 value, ties to even;
- "int8": INT8 codes with an FP32 scale per position and KV head: scale = max |x| / 127 (at least 1e-30), code = x / scale
  rounded to the nearest integer, ties to even.

Codes convert exactly to BF16 (or FP16) for the tensor cores; the keys' scale multiplies the scores and the values'
the probabilities. Softmax in FP32.

Windows (attend). A launch reads R rows; row r is a run of counts[r] <= columns tokens of one sequence (a decode token,
a verify window of a pending token and its drafts, a short prompt piece): tokens starts[r] .. of q, column c at position
firsts[r] + c, its keys and values written already, its page table tables[rows[r]]. Column c sees every position before
firsts[r] and the window's columns 0..c, or with `parents` (each token's parent column in its row, -1 for none: a draft
tree in topological order) the window columns among its ancestors and itself; with `blocks` ([tokens][k] KV block ids,
-1 for none, block_size positions each) only the positions of its listed blocks.

Keys are read in pieces of `piece` positions (piece_size); piece i belongs to partition i mod P (`partitions`), so
partition b walks pieces b, b + P, ... up to its row's last position: the work follows each row's length, read on the
device, not the cache's capacity, and one launch shape (a CUDA graph) serves every length. Each partition writes its
unnormalized output with its running maximum and sum (the partials: one row per column, rows' columns packed by
`offsets` or `columns` apart); a merge combines the partitions holding a column's keys in partition order.

A window launch's query tile (tile_for) holds the launch's columns of a KV head's query heads: up to 32 rows in two
warps, or 64 rows in four where the GPU multiplies four warps over 64 rows with mma.sync too (not sm_90's wgmma, not
sm_100's tcgen05: there such windows take 32-row tiles): on each GPU one instruction shape and one summation order for
every launch, and the row sums run on the tensor cores in a fixed order too. So a column's bits
depend on its own keys and query, the partitions, the piece and the page kind only: not on the rows beside it, not on
how many columns or rows its launch reads (one column decoded alone equals the same column inside a verify window).
A launch given another tile or warp count keeps that only where the GPU multiplies it with mma.sync too.

Knobs. The partitions rule, the piece size, the tiles' warps and stages, the merge's warps and the prompt chunks'
blocks are defaults measured on one sm_120 GPU (an RTX 5090); crit's sweep tunes them for a model for the engine's device
(kernel.json's structures, read through _crit_tuning.knobs() at each call): "split" (the piece size and the partition
count, order knobs: one per cache, the same in every launch over it), "window", "merge", "fused" and "sparse" (warps
and stages, the merge's warps and partitions a turn, a fused launch's value split: speed only, never the bits), "write"
(warps) and "prefill" (its blocks and warps are order knobs: a GPU with warpgroup MMA may multiply another tile with
another instruction). A knob of 0 keeps the rule described here; a warps knob never turns a tile of 64 rows or more
into four warps on a GPU whose tensor cores take that as warpgroup MMA. Past the first call of a shape nothing is
allocated but an `out` the caller does not give (the partials live in a workspace that grows outside a capture only,
or in the caller's), and nothing waits for the host.

Prompt chunks (prefill): one sequence's chunk of queries at positions position.. over its positions up to each query's
own, causal, through its page table; int8_qk quantizes the queries to INT8 per token and head for integer tensor
cores over INT8 pages (an approximation of the queries, twice the BF16 rate).
"""
import ctypes
import functools
import math

import torch
import triton

from crit_kcore import attention_softmax

from . import _crit_cuda, _crit_tuning, _flash, _kernels

__all__ = ["KINDS", "PIECE", "MAX_COLUMNS", "attend", "cuda_block", "cuda_chunk", "cuda_parts", "cuda_ramp",
           "cuda_splits", "flash_block", "flash_chunk", "flash_parts", "flash_tile",
           "partitions", "piece_size", "prefill", "tile_for", "workspace_floats", "write"]

KINDS = {"bf16": 0, "fp16": 0, "fp8": 1, "fp8:row": 2, "int8": 3}
PIECE = 32          # positions per piece of FP8 or INT8 pages at a 256-wide head (piece_size)
MAX_COLUMNS = 64    # columns of one row a launch reads (a tree's ancestor masks are 64-bit)

_workspaces = {}


@functools.lru_cache(maxsize=4096)
def _tuned(structure, key, rows, selected):
    return _crit_tuning.knobs(structure, None if key is None else dict(key), rows)


def _knobs(structure, key, rows):
    """The structure's knobs for a shape key (field -> int, None: no key) and row count from the build's tuning
    (kept per call shape; a sweep's selection, _crit_tuning.SELECTED, read at every call)."""
    selected = getattr(_crit_tuning, "SELECTED", {}).get(structure)
    return _tuned(structure, None if key is None else tuple(sorted(key.items())), int(rows), selected)


def _warps(knob, rule, rows, capability):
    """A tile's warps: the knob's, unless it would make a tile of 64 rows or more four warps (or a multiple) on a GPU
    whose tensor cores take that as warpgroup MMA (another summation order): there the rule's."""
    if not knob or (rows >= 64 and capability[0] in WARPGROUP_MMA and knob % 4 == 0):
        return rule
    return int(knob)


def _check_kind(kind, key_scales):
    if kind not in KINDS:
        raise ValueError(f"page kinds are {', '.join(KINDS)}, not {kind!r}")
    if (KINDS[kind] >= 2) != (key_scales is not None):
        raise ValueError(f"{kind} pages {'take' if KINDS[kind] >= 2 else 'have no'} scales per position and head")
    return KINDS[kind]


@functools.lru_cache(maxsize=None)
def _capability(device):
    return torch.cuda.get_device_capability(device), torch.cuda.get_device_properties(device).multi_processor_count


def _hardware_fp8(device):
    return _capability(device)[0] >= (8, 9)


def _codes(pages, kind, device):
    """Pages as the kernels load them: FP8 pages as their bytes (converted by the GPU where it converts E4M3)."""
    if KINDS[kind] in (1, 2):
        return pages.view(torch.uint8)
    return pages


_dummies = {}


def _dummy(device):
    """A one-float tensor for the pointers a launch does not read (no scales, no tree, no lse)."""
    found = _dummies.get(device)
    if found is None:
        found = _dummies[device] = torch.zeros(1, dtype=torch.float32, device=device)
    return found


def _split_key(head_dim, value_dim, kv_heads, kind):
    if head_dim is None or kv_heads is None:
        return None
    return {"D": int(head_dim), "DV": int(value_dim or head_dim), "HKV": int(kv_heads), "KIND": KINDS[kind]}


def partitions(device, kv_heads, kind, head_dim=None, value_dim=None):
    """The partition count for a cache of `kv_heads` heads of page kind `kind` on `device`: the tuning's (structure
    "split", knob PARTS, by the cache's head widths when given), else the rule for one long row: about one and a half
    programs per SM over converted pages (FP8, INT8), three quarters of one over 16-bit pages, whose pieces carry four
    times the bytes (a power of two within 8..128; measured best at 200K positions with 2, 4 and 8 KV heads of 256 and
    128 on an sm_120 GPU). A column's bits depend on it, so keep one count for every launch whose results must
    agree."""
    key = _split_key(head_dim, value_dim, kv_heads, kind)
    tuned = _knobs("split", key, 1)["PARTS"] if key else 0
    if tuned:
        return int(tuned)
    sms = _capability(torch.device(device))[1]
    programs = (3 * sms) // (2 if KINDS[kind] else 4)
    return max(8, min(128, triton.next_power_of_2(max(1, programs // max(1, kv_heads)))))


def piece_size(head_dim, value_dim, kind="bf16", kv_heads=None):
    """Positions per piece for heads of these widths and a page kind: the tuning's (structure "split", knob PIECE, by
    the cache's KV heads when given), else 16,384 head values for 16-bit pages, 8,192 for codes converted in registers
    (FP8, INT8), within 16..128 (measured best at 256- and 128-wide heads on an sm_120 GPU). A column's bits depend on
    it; it follows the cache's geometry and page kind only."""
    key = _split_key(head_dim, value_dim, kv_heads, kind)
    tuned = _knobs("split", key, 1)["PIECE"] if key else 0
    if tuned:
        return int(tuned)
    values = 16384 if KINDS[kind] == 0 else 8192
    return max(16, min(128, values // max(head_dim, value_dim)))


def write(keys, values, slots, key_pages, value_pages, kind, key_scales=None, value_scales=None,
          multipliers=(1.0, 1.0)):
    """A step's keys [tokens][kv_heads][dim] and values [tokens][kv_heads][value_dim] (any float dtype) into their
    slots ([tokens] int64; a slot below 0 is skipped) of one layer's pages (any view whose storage is [slots][kv_heads]
    [dim]), encoded for the page kind (multipliers: an "fp8" layer's key and value multipliers, codes = x / multiplier
    clamped to +-448; the scales of "fp8:row" and "int8" pages are written beside their codes), in one launch.
    values and value_pages None: keys only (a cache of keys alone, an indexer's)."""
    k_kind = _check_kind(kind, key_scales)
    T, hkv, d = keys.shape
    if T == 0:
        return
    if keys.stride(-1) != 1:
        keys = keys.contiguous()
    heads = 2 * hkv
    if values is None:
        values, value_pages, value_scales, heads = keys, key_pages, key_scales, hkv
    elif values.stride(-1) != 1:
        values = values.contiguous()
    dv = values.shape[-1]
    kp, vp = key_pages, value_pages
    if k_kind in (1, 2):
        kp, vp = kp.view(torch.uint8), vp.view(torch.uint8)
    ks = key_scales if key_scales is not None else _dummy(keys.device)
    vs = value_scales if value_scales is not None else ks
    bd = triton.next_power_of_2(max(d, dv))
    warps = _knobs("write", {"D": d, "DV": dv, "HKV": hkv, "KIND": k_kind}, T)["WARPS"] or (1 if bd <= 128 else 2)
    _kernels._write_kernel[(T, heads)](keys, values, slots, kp, vp, ks, vs, keys.stride(0), keys.stride(1),
                                       values.stride(0), values.stride(1), float(multipliers[0]),
                                       float(multipliers[1]), HKV=hkv, D=d, DV=dv, BD=bd, KIND=k_kind,
                                       num_warps=warps)


NARROW_ROWS = 32                 # rows of the tiles every GPU multiplies with mma.sync
WARPS, STAGES = 2, 3             # a tile of up to 32 rows
WIDE_ROWS, WIDE_WARPS, WIDE_STAGES = 64, 4, 4   # a 64-row tile, where four warps over it stay mma.sync
WARPGROUP_MMA = (9, 10, 11)      # major capabilities whose tensor cores take four warps over 64 rows as warpgroup MMA
FUSED_WARPS = 4                  # a fused launch's tile in four warps (two output sets),
FUSED_VSPLIT = 2                 # its value columns over two programs past 128 (no spills at 64 rows of 256-wide heads)


def tile_for(columns, group, capability, fused=False):
    """(rows, warps, stages) of a window launch's query tile, for `columns` columns over KV heads serving `group` query
    heads each on a GPU of `capability` (major, minor): the smallest power of two of 16 rows or more holding the
    columns' query rows, up to 32, in two warps; past 32 rows a 64-row tile in four warps (each piece's keys and values
    read once for its columns), except on GPUs whose tensor cores take four warps over 64 rows as warpgroup MMA (sm_90's
    wgmma, sm_100's and sm_110's tcgen05: another summation order), which read such windows in 32-row tiles; past 32
    heads a group, one column a tile of its power of two rows in two warps. So on each GPU every tile is multiplied with
    mma.sync, one instruction shape and summation order whatever the tile, and a column's bits are the same alone and
    in any window (the tests compile each choice for sm_80, sm_90, sm_100, sm_110 and sm_120 and check). Measured on
    one sm_120 GPU (an RTX 5090): 64 rows in four warps read windows of more than 32 query rows fastest there. A fused
    launch (its programs hold a partition's output and the folded one, each over half the value columns) takes the
    same rows in four warps."""
    if group > NARROW_ROWS:
        return triton.next_power_of_2(group), FUSED_WARPS if fused else WARPS, STAGES
    rows = columns * group
    if fused:
        if rows <= NARROW_ROWS or capability[0] in WARPGROUP_MMA:
            return max(16, triton.next_power_of_2(min(rows, NARROW_ROWS))), FUSED_WARPS, STAGES
        return WIDE_ROWS, FUSED_WARPS, STAGES
    if rows <= NARROW_ROWS or capability[0] in WARPGROUP_MMA:
        return max(16, triton.next_power_of_2(min(rows, NARROW_ROWS))), WARPS, STAGES
    return WIDE_ROWS, WIDE_WARPS, WIDE_STAGES


def workspace_floats(tokens, heads, partitions, value_dim):
    """FP32 floats of the partials a launch needs for `tokens` partial rows (attend's workspace): R x columns rows,
    or with offsets the rows' tokens packed."""
    return tokens * heads * partitions * (value_dim + 2)


def _workspace(device, tokens, heads, parts, dv, given=None):
    """Partials for tokens x heads x parts: outputs, maxima and sums, in `given` (one FP32 tensor of
    workspace_floats or more) or in a module workspace that grows outside a capture only."""
    need = tokens * heads * parts
    if given is not None:
        if given.dtype != torch.float32 or given.numel() < need * (dv + 2):
            raise ValueError(f"attend's workspace holds {given.numel()} floats; this launch needs {need * (dv + 2)} "
                             "(workspace_floats)")
        return given[:need * dv], given[need * dv:need * (dv + 1)], given[need * (dv + 1):need * (dv + 2)]
    found = _workspaces.get(device)
    if found is None or found[1].numel() < need:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("paged_kv_attention.attend: run a launch of this size once before capturing it (its "
                               "partials' workspace is allocated outside the capture)")
        size = max(need, 4096)
        found = (torch.empty(size * dv, dtype=torch.float32, device=device),
                 torch.empty(size, dtype=torch.float32, device=device),
                 torch.empty(size, dtype=torch.float32, device=device))
        _workspaces[device] = found
    po, pm, pl = found
    if po.numel() < need * dv:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("paged_kv_attention.attend: run a launch of this size once before capturing it")
        po = torch.empty(need * dv, dtype=torch.float32, device=device)
        _workspaces[device] = (po, pm, pl)
    return po, pm, pl


def attend(q, key_pages, value_pages, kind, tables, rows, starts, firsts, columns, page, *, counts=None,
           offsets=None, tokens=None, key_scales=None, value_scales=None, multipliers=(1.0, 1.0), scale=None,
           partitions=None, parents=None, blocks=None, block_size=None, out=None, lse=None, workspace=None, piece=None,
           fused=False, fresh=None, vsplit=None, tile=None, num_warps=None, num_stages=None, maxnreg=None,
           flash=False, chunk=None, ramp=None):
    """Decode and verify windows over paged KV (see the module's description).

    q [tokens][heads][dim] BF16 or FP16 (FP16 for "fp16" pages); key_pages/value_pages one layer's pages, key_scales/
    value_scales their scales for "fp8:row" and "int8"; tables [S][width] int32 page tables; rows [R] the table row of
    each batch row; starts [R] int32 its first token in q; firsts [R] int64 its column 0's position; counts [R] int32
    its columns (None: `columns` each; more than `columns` read as `columns`); columns the widest row (<= MAX_COLUMNS),
    a Python int. offsets [R] int32 and tokens: each row's first partial row (its tokens' rank among the launch's,
    nondecreasing, rows' counts apart at least) and the partial rows the launch takes from offsets[0] on (at most
    R x columns), so the partials follow the launch's tokens; without them row r's partials are rows r x columns ...
    parents [tokens] (or a view of them) each token's parent column, -1 for none; blocks [tokens][k] int32 KV block ids
    and block_size. Writes out [tokens][heads][value_dim] (q's dtype; a new tensor each call when not given) at the
    rows' tokens, and lse [tokens][heads] FP32 (natural log of each row's softmax sum over its scaled scores) when
    given; returns out. workspace: an FP32 tensor of workspace_floats(partial rows, heads, partitions, value_dim) or
    more for the partials (default: one the module keeps per device, grown outside a capture). piece: positions per
    piece (default piece_size). fused: each program reads every partition of its columns in turn and folds
    them as the merge does, writing the output itself: the same bits as the split launch, no workspace (a prompt
    chunk's rows, whose partials would not fit; no block lists). fresh: (keys [N][kv_heads][dim], values
    [N][kv_heads][value_dim] in q's dtype, shift, first): each row's positions from first on read from those rows
    instead of the pages (row = position - shift; a prompt chunk's own keys and values as the projections gave them),
    shift and first int64 tensors of one element a row, or of one element for every row (a captured chunk's, on the
    device). vsplit: a fused launch's programs per tile, each holding a part of the value columns (default
    FUSED_VSPLIT past 128 value columns, else 1: fewer registers, each recomputing the scores; the same bits). tile,
    num_warps, num_stages, maxnreg: another launch configuration than tile_for()'s (a measurement's),
    whose columns keep their bits only where the GPU multiplies it with mma.sync too.

    flash: the flash structures instead (structures "flash-window" and "flash-fused", _flash.py): a row's keys in
    contiguous splits of `chunk` positions (default flash_chunk(): one per cache, the bits depend on it), its columns
    times the KV head's query heads packed into tiles of up to 128 rows, the splits merged (or with fused folded in
    the program) by their states in split order; another reduction than the split launch's (its own one: a column's
    bits are the same alone and in any window of a flash launch over the cache, fused or not). partitions, piece and
    block lists do not apply (blocks: the split launch); its workspace takes flash_parts() splits a partial row.
    flash "cuda": the CUDA flash tile (cuda_window.cu), its splits cuda_chunk()'s positions with the cache's ramp
    (cuda_ramp(), or `ramp` (R, K)); its workspace takes cuda_parts() splits a partial row."""
    k_kind = _check_kind(kind, key_scales)
    T, hq, d = q.shape
    hkv, dv = key_pages.shape[-2], value_pages.shape[-1]
    R = int(starts.shape[0])
    if flash == "cuda" and blocks is None:
        return _attend_cuda(q, key_pages, value_pages, kind, k_kind, tables, rows, starts, firsts, columns, page,
                            counts, offsets, tokens, key_scales, value_scales, multipliers, scale, parents, out, lse,
                            workspace, fused, fresh, vsplit, chunk, num_warps, num_stages, ramp)
    if flash and blocks is None:
        return _attend_flash(q, key_pages, value_pages, kind, k_kind, tables, rows, starts, firsts, columns, page,
                             counts, offsets, tokens, key_scales, value_scales, multipliers, scale, parents, out, lse,
                             workspace, fused, fresh, vsplit, chunk, tile, num_warps, num_stages)
    if hq % hkv or not 1 <= columns <= MAX_COLUMNS or R == 0:
        raise ValueError("query heads a multiple of the KV heads, 1..64 columns a row, one row or more")
    if q.stride(-1) != 1:
        raise ValueError("q's last dimension is contiguous")
    if max(d, dv) > 4 * min(64, d, dv):
        raise ValueError("head dims are powers of two of 16 to 256 (keys and values within four 64-wide chunks)")
    if (offsets is None) != (tokens is None):
        raise ValueError("offsets come with the partial rows they take (tokens)")
    partial_rows = R * columns if offsets is None else int(tokens)
    g = hq // hkv
    device = q.device
    parts = int(partitions or globals()["partitions"](device, hkv, kind, d, dv))
    scale = float(scale if scale is not None else d ** -0.5)
    if fused and blocks is not None:
        raise ValueError("a fused launch reads windows without block lists")
    if fused:
        po = pm = pl = _dummy(device)
    else:
        po, pm, pl = _workspace(device, partial_rows, hq, parts, dv, workspace)
    out = torch.empty(T, hq, dv, dtype=q.dtype, device=device) if out is None else out
    kp, vp = _codes(key_pages, kind, device), _codes(value_pages, kind, device)
    ks = key_scales if key_scales is not None else _dummy(device)
    vs = value_scales if value_scales is not None else ks
    tree = parents is not None
    par = parents.reshape(-1) if tree else starts
    count_ptr = counts if counts is not None else starts
    off_ptr = offsets if offsets is not None else starts
    launch = {"maxnreg": int(maxnreg)} if maxnreg else {}
    piece = int(piece or piece_size(d, dv, kind, hkv))
    capability = _capability(device)[0]
    key = {"D": d, "DV": dv, "HQ": hq, "HKV": hkv, "PAGE": int(page), "KIND": k_kind}
    bm, warps, stages = tile_for(columns, g, capability, fused)
    if tile:
        bm = max(16, triton.next_power_of_2(max(int(tile), g)))
    if blocks is None:
        tuned = _knobs("fused" if fused else "window", key, columns)
        warps, stages = _warps(tuned["WARPS"], warps, bm, capability), tuned["STAGES"] or stages
    warps, stages = num_warps or warps, num_stages or stages
    common = dict(HQ=hq, HKV=hkv, G=g, D=d, DV=dv, PIECE=piece, PAGE=page, KIND=k_kind,
                  HW_FP8=_hardware_fp8(device), TREE=tree, COUNTS=counts is not None, OFFSETS=offsets is not None,
                  DC=min(64, d, dv))
    if blocks is None:
        if fresh is not None:
            fk, fv, fshift, ffirst = fresh
            if fk.shape[1:] != (hkv, d) or fv.shape[1:] != (hkv, dv) or fk.stride(-1) != 1 or fv.stride(-1) != 1 or \
                    fk.stride(1) != d or fv.stride(1) != dv or fshift.numel() != ffirst.numel() or \
                    fshift.numel() not in (1, R):
                raise ValueError(f"fresh rows are [N][{hkv}][dim] each, head rows contiguous, with one shift and first "
                                 "a row or one for every row")
            fstride = 0 if fshift.numel() == 1 else 1
        else:
            fk = fv = q
            fshift = ffirst = firsts
            fstride = 0
        cpt = bm // g
        dc = min(64, d, dv)
        split = 1
        if fused:
            split = int(vsplit or tuned["VSPLIT"] or (FUSED_VSPLIT if dv > 128 else 1))
            while split > 1 and (dv // split) % dc:
                split //= 2
        grid = (triton.cdiv(columns, cpt), split if fused else parts, R * hkv)
        _kernels._window_kernel[grid](q, kp, vp, ks, vs, tables, rows, starts, count_ptr, firsts, par, off_ptr, po, pm, pl,
                     out, lse if lse is not None else ks, fk, fv, fshift, ffirst, q.stride(0), q.stride(1),
                     out.stride(0), out.stride(1), fk.stride(0), fv.stride(0), fstride, tables.stride(0),
                     tables.shape[1], parts, scale, float(multipliers[0]), float(multipliers[1]), columns,
                     COLS=columns, CPT=cpt, BLOCK_M=bm, num_warps=warps, num_stages=stages, FUSED=bool(fused),
                     LSE=bool(fused) and lse is not None, FRESH=fresh is not None, DVP=dv // split, **common,
                     **launch)
        if fused:
            return out
        fixed = 0
    else:
        if not block_size or block_size < 1:
            raise ValueError("block-sparse attention takes the blocks' size in positions")
        if blocks.stride(-1) != 1:
            blocks = blocks.contiguous()
        k = int(blocks.shape[-1])
        bm = max(16, triton.next_power_of_2(g))   # one token's query heads a program
        tuned = _knobs("sparse", key, R * columns)
        warps = num_warps or _warps(tuned["WARPS"], WARPS, bm, capability)
        stages = num_stages or tuned["STAGES"] or STAGES
        grid = (parts, hkv, R * columns)
        _kernels._sparse_kernel[grid](q, kp, vp, ks, vs, tables, rows, starts, count_ptr, firsts, par, off_ptr,
                                      blocks, blocks.stride(0), k, int(block_size), po, pm, pl, q.stride(0),
                                      q.stride(1), tables.stride(0), tables.shape[1], parts, scale,
                                      float(multipliers[0]), float(multipliers[1]), columns, COLS=columns,
                                      BLOCK_M=bm, num_warps=warps, num_stages=stages, **common)
        fixed = triton.cdiv(k * int(block_size), piece)
    block_p = triton.next_power_of_2(parts)
    merge = _knobs("merge", key, columns)
    _kernels._merge_kernel[(columns, hq, R)](po, pm, pl, out, lse if lse is not None else pm, starts, count_ptr,
                                             firsts, off_ptr, parts, fixed, out.stride(0), out.stride(1), columns,
                                             COLS=columns, HQ=hq, DV=dv, PIECE=piece, COUNTS=counts is not None,
                                             OFFSETS=offsets is not None, LSE=lse is not None, BLOCK_P=block_p,
                                             CHUNK_P=min(merge["CHUNK"], block_p), num_warps=merge["WARPS"])
    return out


# -- the flash structures ----------------------------------------------------------------------------------------------
FLASH_CHUNK = 1024        # positions a split of a flash launch over a cache (flash_chunk's rule)
FLASH_WARPGROUP_TILE = (64, 4)   # the one flash tile on GPUs with warpgroup MMA (one instruction for every launch)


def _flash_key(d, dv, hq, hkv, page, k_kind):
    return {"D": int(d), "DV": int(dv), "HQ": int(hq), "HKV": int(hkv), "PAGE": int(page), "KIND": int(k_kind)}


def _flash_split_knobs(head_dim, value_dim, heads, kv_heads, page, kind):
    return _knobs("flash-window", _flash_key(head_dim, value_dim, heads, kv_heads, page, KINDS[kind]), 1)


def flash_block(head_dim, value_dim, heads, kv_heads, page, kind="bf16"):
    """Positions a step of a flash launch's online softmax over a cache of this geometry: the tuning's (structure
    "flash-window", knob BLOCK_N, an order knob), else 32."""
    tuned = _flash_split_knobs(head_dim, value_dim, heads, kv_heads, page, kind)["BLOCK_N"]
    return int(tuned) if tuned else 32


def flash_chunk(head_dim, value_dim, heads, kv_heads, page, kind="bf16"):
    """Positions a split of every flash launch over a cache of this geometry (decode, verify windows, fused prompt
    rows): the tuning's (structure "flash-window", knob CHUNK, an order knob), else FLASH_CHUNK; a multiple of the
    block (flash_block). A column's bits depend on it, on the block and on the tile (flash_tile): keep one of each for
    every launch whose results must agree."""
    tuned = _flash_split_knobs(head_dim, value_dim, heads, kv_heads, page, kind)["CHUNK"]
    block = flash_block(head_dim, value_dim, heads, kv_heads, page, kind)
    found = int(tuned) if tuned else FLASH_CHUNK
    return max(block, found // block * block)


def flash_parts(positions, chunk):
    """Splits a flash launch's partial row takes over page tables holding `positions` positions a row."""
    return max(1, -(-int(positions) // int(chunk)))


def flash_tile(head_dim, value_dim, heads, kv_heads, page, kind, capability):
    """(rows, warps) of the query tile of every flash window launch over a cache of this geometry on a GPU of
    `capability`: the tuning's (structure "flash-window", knobs ROWS and WARPS, order knobs), else 32 rows in four
    warps (two along the rows, two along the value columns: the accumulators fit the registers); on GPUs whose
    tensor cores take four warps over 64 rows as warpgroup MMA (sm_90, sm_100, sm_110) FLASH_WARPGROUP_TILE. A row's
    sum is the tile layout's reduction, the same for every row of every tile of one shape, and mma.sync products do not
    depend on the rows beside them: one tile per cache keeps a column's bits the same alone and in any window."""
    if capability[0] in WARPGROUP_MMA:
        return FLASH_WARPGROUP_TILE
    tuned = _flash_split_knobs(head_dim, value_dim, heads, kv_heads, page, kind)
    return int(tuned["ROWS"]) or 32, int(tuned["WARPS"]) or 4


def _flash_fresh(fresh, R, hkv, d, dv):
    fk, fv, fshift, ffirst = fresh
    if fk.shape[1:] != (hkv, d) or fv.shape[1:] != (hkv, dv) or fk.stride(-1) != 1 or fv.stride(-1) != 1 or \
            fk.stride(1) != d or fv.stride(1) != dv or fshift.numel() != ffirst.numel() or fshift.numel() not in (1, R):
        raise ValueError(f"fresh rows are [N][{hkv}][dim] each, head rows contiguous, with one shift and first a row or "
                         "one for every row")
    return fk, fv, fshift, ffirst, 0 if fshift.numel() == 1 else 1


def _pow2(n):
    return n > 0 and n & (n - 1) == 0


def _attend_flash(q, key_pages, value_pages, kind, k_kind, tables, rows, starts, firsts, columns, page, counts, offsets,
                  tokens, key_scales, value_scales, multipliers, scale, parents, out, lse, workspace, fused, fresh,
                  vsplit, chunk, tile, num_warps, num_stages):
    """attend(..., flash=True): windows through the flash kernel (split launch and merge, or fused)."""
    T, hq, d = q.shape
    hkv, dv = key_pages.shape[-2], value_pages.shape[-1]
    R = int(starts.shape[0])
    if hq % hkv or not 1 <= columns <= MAX_COLUMNS or R == 0 or q.stride(-1) != 1:
        raise ValueError("query heads a multiple of the KV heads, 1..64 columns a row, one row or more, q's last "
                         "dimension contiguous")
    if (offsets is None) != (tokens is None):
        raise ValueError("offsets come with the partial rows they take (tokens)")
    if d > 256 or dv > 256 or not (_pow2(d) and _pow2(dv)) or min(d, dv) < 16:
        raise ValueError("flash launches take head dims that are powers of two of 16 to 256")
    g = hq // hkv
    device = q.device
    scale = float(scale if scale is not None else d ** -0.5)
    key = _flash_key(d, dv, hq, hkv, page, k_kind)
    block_n = flash_block(d, dv, hq, hkv, page, kind)
    chunk = int(chunk or flash_chunk(d, dv, hq, hkv, page, kind))
    if chunk % block_n:
        raise ValueError(f"a flash split ({chunk} positions) is a multiple of the block ({block_n})")
    bm, warps = flash_tile(d, dv, hq, hkv, page, kind, _capability(device)[0])
    bm, warps = int(tile or bm), int(num_warps or warps)
    tuned = _knobs("flash-fused" if fused else "flash-window", key, columns)
    stages = int(num_stages or tuned["STAGES"] or (1 if max(d, dv) > 128 else 2))
    parts = flash_parts(tables.shape[1] * page, chunk)
    partial_rows = R * columns if offsets is None else int(tokens)
    if fused:
        po = pm = pl = _dummy(device)
        split = int(vsplit or tuned["VSPLIT"] or (2 if dv >= 128 else 1))
        while split > 1 and ((dv // split) < 16 or dv % split):
            split //= 2
    else:
        po, pm, pl = _workspace(device, partial_rows, hq, parts, dv, workspace)
        split = 1
    out = torch.empty(T, hq, dv, dtype=q.dtype, device=device) if out is None else out
    kp, vp = _codes(key_pages, kind, device), _codes(value_pages, kind, device)
    ks = key_scales if key_scales is not None else _dummy(device)
    vs = value_scales if value_scales is not None else ks
    tree = parents is not None
    par = parents.reshape(-1) if tree else starts
    count_ptr = counts if counts is not None else starts
    off_ptr = offsets if offsets is not None else starts
    if fresh is not None:
        fk, fv, fshift, ffirst, fstride = _flash_fresh(fresh, R, hkv, d, dv)
    else:
        fk = fv = q
        fshift = ffirst = firsts
        fstride = 0
    grid = (triton.cdiv(columns * g, bm), split if fused else parts, R * hkv)
    _flash._flash_kernel[grid](
        q, kp, vp, ks, vs, tables, rows, starts, count_ptr, firsts, par, off_ptr, po, pm, pl, out,
        lse if lse is not None else ks, fk, fv, fshift, ffirst, q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        fk.stride(0), fv.stride(0), fstride, tables.stride(0), tables.shape[1], parts, chunk,
        scale * attention_softmax.LOG2E, float(multipliers[0]), float(multipliers[1]), columns, 0, 0,
        HQ=hq, HKV=hkv, G=g, D=d, DV=dv, DVP=dv // split, BLOCK_M=bm, BLOCK_N=block_n, PAGE=page, KIND=k_kind,
        HW_FP8=_hardware_fp8(device), TREE=tree, TREE_COLS=columns if tree else 1, COUNTS=counts is not None,
        OFFSETS=offsets is not None, FUSED=bool(fused), ONE_SPLIT=False, LSE=lse is not None, FRESH=fresh is not None,
        ONE_ROW=False, POS_T=False, FIRST_T=False, DEV_CHUNK=False, INT8_QK=False, SKIP=False, num_warps=warps,
        num_stages=stages)
    if fused:
        return out
    merge = _knobs("merge", {"D": d, "DV": dv, "HQ": hq, "HKV": hkv, "PAGE": int(page), "KIND": k_kind}, columns)
    _flash._flash_merge_kernel[(columns, hq, R)](
        po, pm, pl, out, lse if lse is not None else pm, starts, count_ptr, firsts, off_ptr, parts, chunk, columns, 0,
        out.stride(0), out.stride(1), HQ=hq, DV=dv, BLOCK_N=block_n, COUNTS=counts is not None,
        OFFSETS=offsets is not None, LSE=lse is not None, ONE_ROW=False, POS_T=False, DEV_CHUNK=False,
        CHUNK_P=min(merge["CHUNK"], triton.next_power_of_2(parts)), num_warps=merge["WARPS"])
    return out


def _prefill_flash(q, kp, vp, kind, k_kind, table, position, page, ks, vs, multipliers, scale, out, lse, int8_qk,
                   fresh, first):
    """prefill(..., flash=True): the chunk as one row of T columns through the flash kernel, its positions split on
    the device over SPLITS programs a tile (merged), or with one split read in one range and written directly."""
    T, hq, d = q.shape
    hkv = kp.shape[-2]
    g = hq // hkv
    device = q.device
    if d > 256 or not _pow2(d) or d < 16:
        raise ValueError("flash launches take head dims that are powers of two of 16 to 256")
    tuned = _knobs("flash-prefill", {"D": d, "HQ": hq, "HKV": hkv, "PAGE": int(page), "KIND": k_kind,
                                     "INT8": int(bool(int8_qk))}, T)
    capability, sms = _capability(device)
    bm = int(tuned["BLOCK_M"]) or 32
    block_n = int(tuned["BLOCK_N"]) or 32
    warps = int(tuned["WARPS"]) or 4
    stages = int(tuned["STAGES"]) or (1 if d > 128 else 2)
    if capability[0] in WARPGROUP_MMA:
        bm, warps = FLASH_WARPGROUP_TILE
    tiles = triton.cdiv(T * g, bm)
    splits = int(tuned["SPLITS"])
    if not splits:
        # About two programs an SM: the chunk's tiles times its KV heads, split while they are fewer.
        splits = max(1, min(64, (2 * sms) // max(1, tiles * hkv)))
    on_device = torch.is_tensor(position)
    first_t = torch.is_tensor(first)
    one = splits == 1
    if one:
        po = pm = pl = _dummy(device)
    else:
        po, pm, pl = _workspace(device, T, hq, splits, d)
    if fresh is not None:
        fk, fv = fresh[:2]
    else:
        fk = fv = q
    pos = table if not on_device else position
    qk = float(scale if scale is not None else d ** -0.5) * attention_softmax.LOG2E
    _flash._flash_kernel[(tiles, 1 if one else splits, hkv)](
        q, kp, vp, ks, vs, table, table, table, table, pos, table, table, po, pm, pl, out,
        lse if lse is not None else ks, fk, fv, table, first if first_t else table, q.stride(0), q.stride(1),
        out.stride(0), out.stride(1), fk.stride(0), fv.stride(0), 0, 0, table.shape[0], splits, 0, qk,
        float(multipliers[0]), float(multipliers[1]), T, 0 if on_device else int(position),
        0 if first_t else int(first),
        HQ=hq, HKV=hkv, G=g, D=d, DV=d, DVP=d, BLOCK_M=bm, BLOCK_N=block_n, PAGE=page, KIND=k_kind,
        HW_FP8=_hardware_fp8(device), TREE=False, TREE_COLS=1, COUNTS=False, OFFSETS=False, FUSED=False,
        ONE_SPLIT=one, LSE=lse is not None, FRESH=fresh is not None, ONE_ROW=True, POS_T=on_device, FIRST_T=first_t,
        DEV_CHUNK=True, INT8_QK=bool(int8_qk), SKIP=True, num_warps=warps, num_stages=stages)
    if not one:
        _flash._flash_merge_kernel[(T, hq, 1)](
            po, pm, pl, out, lse if lse is not None else pm, table, table, pos, table, splits, 0, T,
            0 if on_device else int(position), out.stride(0), out.stride(1), HQ=hq, DV=d, BLOCK_N=block_n,
            COUNTS=False, OFFSETS=False, LSE=lse is not None, ONE_ROW=True, POS_T=on_device, DEV_CHUNK=True,
            CHUNK_P=min(16, triton.next_power_of_2(splits)), num_warps=4)
    return out


# -- the CUDA flash tile (cuda_window.cu, kernel.json "cuda": compiled by the build, _crit_cuda) ----------------------
CUDA_CHUNK = 512          # positions a split of a cuda launch over a cache where no GPU is visible (cuda_chunk)
CUDA_REACH = 90112        # cuda_chunk's rule: a decode row of this many positions spreads over two programs an SM
CUDA_BLOCK = 32           # positions a step of its online softmax over 8-bit pages (cuda_block's rule; 16-bit: 16)
CUDA_RAMP_SPLITS = 16     # splits a band of a ramp (cuda_ramp)
CUDA_WIDTHS = (64, 128, 256)
CUDA_FLAGS = {"counts": 1, "offsets": 2, "tree": 4, "lse": 8, "fresh": 16, "one_row": 32, "pos_t": 64, "first_t": 128,
              "dev_chunk": 256, "one_split": 512, "par64": 1024}
FUSED_GROUP = 16                  # splits a fused tile's program folds at once at most (scheduling only, never a bit)
FUSED_WORKSPACE = 64 << 20        # bytes a fused launch's ring of states may take before its groups shrink
FUSED_PARTIALS = 128 << 20        # workspace bytes (workspace_floats) a fused launch may take as a window launch
FUSED_DEPTH = 2                   # groups of slots a fused tile's ring holds (deeper rings measured slower: their
                                  # programs run ahead of the folds, which then trail the last split)
_CUDA_WINDOW, _CUDA_FUSED, _CUDA_PREFILL = 0, 1, 2


class _Params(ctypes.Structure):
    """cuda_window.cu's Params, field for field."""
    _fields_ = ([(n, ctypes.c_void_p) for n in ("q", "kp", "vp", "ks", "vs", "tables", "rows", "starts", "counts",
                                                "firsts", "parents", "offsets", "po", "pm", "pl", "out", "lse", "fk",
                                                "fv", "fshift", "ffirst", "run", "sync")]
                + [(n, ctypes.c_longlong) for n in ("stride_qt", "stride_qh", "stride_ot", "stride_oh", "stride_fk",
                                                    "stride_fv", "first0", "ffirst0")]
                + [(n, ctypes.c_int) for n in ("stride_fr", "tab_stride", "tab_width", "hq", "hkv", "g", "page",
                                               "n_parts", "chunk", "cols", "flags", "page_shift", "block", "group",
                                               "tiles", "merge", "ring", "ramp", "ramp_k", "pad")]
                + [(n, ctypes.c_float) for n in ("qk_scale", "k_mult", "v_mult", "pad2")])


_cuda_libs = {}
_fused_workspaces = {}


def _cuda_lib(device, **variant):
    """The CUDA flash tile's library for a variant (query dtype T, page kind, head widths, the caller's warps W, stages
    S and value split VS or 0), compiled by the build for this GPU on first use."""
    key = (str(device), tuple(sorted(variant.items())))
    lib = _cuda_libs.get(key)
    if lib is None:
        lib = _crit_cuda.library(device, **variant)
        lib.crit_cw_plan.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.POINTER(ctypes.c_int)]
        lib.crit_cw_plan.restype = ctypes.c_int
        lib.crit_cw_launch.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_Params), ctypes.c_void_p,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.crit_cw_launch.restype = ctypes.c_int
        lib.crit_cw_merge.argtypes = [ctypes.POINTER(_Params), ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int]
        lib.crit_cw_merge.restype = ctypes.c_int
        lib.crit_cw_error.restype = ctypes.c_char_p
        lib.crit_cw_error.argtypes = [ctypes.c_int]
        if lib.crit_cw_params_size() != ctypes.sizeof(_Params):
            raise RuntimeError("cuda_window.cu's Params and the package's _Params differ")
        _cuda_libs[key] = lib
    return lib


def _cuda_plan(lib, structure, hq, hkv, page, rows, fresh=False, length=-1):
    """(entry, warps, stages, block, value split) of a call: the build's tuning for its shape, rows and length (the
    positions its page tables hold: a tuning's length ranges, kernel.json "lengths"), the rules resolved for its
    variant, whether it reads fresh rows and the GPU's shared memory."""
    out = (ctypes.c_int * 8)()
    err = lib.crit_cw_plan(structure, hq, hkv, int(page), int(rows), int(bool(fresh)), int(length), out)
    if err == 1:
        raise RuntimeError("this build of paged-kv-attention holds no such cuda configuration (a sweep build of "
                           "another structure)")
    if err == 2:
        raise ValueError(f"a cuda tile of blocks of {out[3]} does not fit the GPU's shared memory")
    return tuple(out[:5])


def _cuda_launch(lib, structure, entry, params, device, grid):
    stream = torch.cuda.current_stream(device).cuda_stream
    err = lib.crit_cw_launch(structure, entry, ctypes.byref(params), ctypes.c_void_p(stream), *[int(x) for x in grid])
    if err:
        raise RuntimeError(f"paged-kv-attention cuda launch: {lib.crit_cw_error(err).decode()}")


def _cuda_merge(lib, params, device, grid):
    stream = torch.cuda.current_stream(device).cuda_stream
    err = lib.crit_cw_merge(ctypes.byref(params), ctypes.c_void_p(stream), *[int(x) for x in grid])
    if err:
        raise RuntimeError(f"paged-kv-attention cuda merge: {lib.crit_cw_error(err).decode()}")


def _cuda_split_knobs(head_dim, value_dim, heads, kv_heads, page, kind):
    return _knobs("cuda-window", _flash_key(head_dim, value_dim, heads, kv_heads, page, KINDS[kind]), 1)


def cuda_block(head_dim, value_dim, heads, kv_heads, page, kind="bf16"):
    """Positions a step of a cuda launch's online softmax over a cache of this geometry: the tuning's (structure
    "cuda-window", knob BLOCK_N, an order knob), else CUDA_BLOCK."""
    tuned = _cuda_split_knobs(head_dim, value_dim, heads, kv_heads, page, kind)["BLOCK_N"]
    return int(tuned) if tuned else (16 if KINDS[kind] == 0 else CUDA_BLOCK)


def cuda_ramp(head_dim, value_dim, heads, kv_heads, page, kind="bf16"):
    """(R, K) of every cuda launch over a cache of this geometry: the first positions of a row in R bands of K splits,
    band b's of cuda_chunk() / 2^(R - b) positions, the rest in splits of cuda_chunk() (the tuning's structure
    "cuda-window", knob RAMP, an order knob; K CUDA_RAMP_SPLITS); R lowered where the chunk would not split into whole
    blocks. (0, K): every split of cuda_chunk() positions."""
    ramp = int(_cuda_split_knobs(head_dim, value_dim, heads, kv_heads, page, kind)["RAMP"])
    chunk = cuda_chunk(head_dim, value_dim, heads, kv_heads, page, kind)
    block = cuda_block(head_dim, value_dim, heads, kv_heads, page, kind)
    while ramp and chunk % (block << ramp):
        ramp -= 1
    return ramp, CUDA_RAMP_SPLITS


def cuda_splits(positions, chunk, ramp=(0, CUDA_RAMP_SPLITS)):
    """Splits of a row of `positions` positions (at least one) under a split of `chunk` and a ramp (R, K), as
    cuda_window.cu's splits_to: a cuda launch's partials hold this many a row of its page tables' positions."""
    r, k = ramp
    positions = max(1, int(positions))
    if not r:
        return -(-positions // chunk)
    s0 = chunk >> r
    band = k * s0
    edge = band * ((1 << r) - 1)
    if positions > edge:
        return k * r + -(-(positions - edge) // chunk)
    last = positions - 1
    b = ((last + band) // band).bit_length() - 1
    return k * b + (last - band * ((1 << b) - 1)) // (s0 << b) + 1


def cuda_parts(head_dim, value_dim, heads, kv_heads, page, kind, positions, chunk=None):
    """Splits a cuda launch over this cache takes a row of `positions` table positions (its partials' count a row and
    head, workspace_floats): the cache's chunk (or `chunk`) and ramp (cuda_ramp)."""
    chunk = int(chunk or cuda_chunk(head_dim, value_dim, heads, kv_heads, page, kind))
    return cuda_splits(positions, chunk, cuda_ramp(head_dim, value_dim, heads, kv_heads, page, kind))


def cuda_chunk(head_dim, value_dim, heads, kv_heads, page, kind="bf16"):
    """Positions a split of every cuda launch over a cache of this geometry (decode, verify windows, fused prompt rows):
    the tuning's (structure "cuda-window", knob CHUNK, an order knob), else CUDA_CHUNK; a multiple of the block
    (cuda_block). A column's bits depend on it and on the block only (each warp owns 16 query rows and their softmax:
    the warps and the tile change no bit): keep one of each for every launch whose results must agree."""
    tuned = _cuda_split_knobs(head_dim, value_dim, heads, kv_heads, page, kind)["CHUNK"]
    block = cuda_block(head_dim, value_dim, heads, kv_heads, page, kind)
    found = int(tuned) if tuned else _cuda_chunk_rule(kv_heads)
    return max(block, found // block * block)


@functools.lru_cache(maxsize=64)
def _cuda_chunk_rule(kv_heads):
    """The untuned split, from the GPU: a decode row of CUDA_REACH positions over two programs an SM, its KV heads
    counted by their square root (a row's KV heads share its splits' pages and queries' tiles: four times the heads
    want about twice the split, measured), to the nearest 128 positions, 128 to 2,048. Short rows then take a few
    splits a KV head and long rows few enough that their merge stays short; fixed per cache (the GPU and the cache's
    KV heads), never by the batch."""
    if not torch.cuda.is_available():
        return CUDA_CHUNK
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    found = CUDA_REACH * math.sqrt(int(kv_heads)) / (2 * sms)
    return int(min(2048, max(128, round(found / 128) * 128)))


def _cuda_dtype(q, kind):
    want = torch.float16 if kind == "fp16" else torch.bfloat16
    if q.dtype not in (torch.bfloat16, torch.float16) or (kind in ("bf16", "fp16") and q.dtype != want):
        raise ValueError("cuda launches take BF16 queries (FP16 over fp16 pages)")
    return 1 if q.dtype == torch.float16 else 0


def _aligned16(*tensors):
    return all(t is None or t.data_ptr() % 16 == 0 for t in tensors)


def _cuda_params(q, kp, vp, ks, vs, out, lse, scale, multipliers, hq, hkv, page):
    P = _Params()
    P.q, P.kp, P.vp = q.data_ptr(), kp.data_ptr(), vp.data_ptr()
    P.ks = ks.data_ptr() if ks is not None else 0
    P.vs = vs.data_ptr() if vs is not None else 0
    P.out = out.data_ptr()
    P.lse = lse.data_ptr() if lse is not None else 0
    P.stride_qt, P.stride_qh = q.stride(0), q.stride(1)
    P.stride_ot, P.stride_oh = out.stride(0), out.stride(1)
    P.hq, P.hkv, P.g, P.page = hq, hkv, hq // hkv, int(page)
    P.page_shift = int(page).bit_length() - 1 if _pow2(int(page)) else -1
    P.qk_scale = scale * attention_softmax.LOG2E
    P.k_mult, P.v_mult = float(multipliers[0]), float(multipliers[1])
    return P


def _as(t, dtype):
    """t as a contiguous tensor of dtype (itself where it is one already, else a copy)."""
    return t if t.dtype == dtype and t.is_contiguous() else t.to(dtype).contiguous()


def _cuda_rows_ok(q, out):
    if q.stride(-1) != 1 or q.stride(0) % 8 or q.stride(1) % 8 or out.stride(-1) != 1 or out.stride(0) % 2 or \
            out.stride(1) % 2 or q.data_ptr() % 16 or out.data_ptr() % 4:
        raise ValueError("cuda launches take q rows 16-byte aligned and out rows 4-byte aligned")


def _fused_workspace(device, chains, rows, dvp, parts):
    """A fused launch's ring of split states (FUSED_DEPTH groups of `group` slots a tile: BF16 outputs, maxima, sums),
    the tiles' running states (FP32, needed only where a tile has more splits than a group) and its sync words (zero
    between launches: each launch leaves them so). The group and the ring's depth are scheduling only (the fold order
    is the splits'): a group of up to FUSED_GROUP splits, shrunk where the ring would not fit FUSED_WORKSPACE. Returns
    (group, depth, slots, run, sync), from a module workspace that grows outside a capture only."""
    group = max(1, min(FUSED_GROUP, 16, parts))   # the kernel folds 16 states a group at most
    depth = FUSED_DEPTH

    def floats(k, depth):
        ring = chains * depth * k * rows * (dvp // 2 + 2)
        return ring, (chains * rows * (dvp + 2) if parts > k else 0)
    while group > 1 and 4 * sum(floats(group, depth)) > FUSED_WORKSPACE:
        group //= 2
    ring, run = floats(group, depth)
    words = 2 + (1 + depth) * chains
    found = _fused_workspaces.get(device)
    if found is None or found[0].numel() < ring + run or found[1].numel() < words:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("paged_kv_attention.attend: run a fused launch of this size once before capturing it "
                               "(its workspace is allocated outside the capture)")
        size = max(ring + run, 4096, found[0].numel() if found else 0)
        found = (torch.empty(size, dtype=torch.float32, device=device),
                 torch.zeros(max(words, 1024, found[1].numel() if found else 0), dtype=torch.int32, device=device))
        _fused_workspaces[device] = found
    return group, depth, found[0][:ring], found[0][ring:ring + run], found[1]


def _fused_as_window(partial_rows, heads, parts, dv):
    """Whether a fused cuda launch runs as the window launch and its merge: where the partials of every split the
    page tables hold (partial_rows x heads x parts) fit FUSED_PARTIALS bytes."""
    return 4 * workspace_floats(partial_rows, heads, parts, dv) <= FUSED_PARTIALS


def _attend_cuda(q, key_pages, value_pages, kind, k_kind, tables, rows, starts, firsts, columns, page, counts, offsets,
                 tokens, key_scales, value_scales, multipliers, scale, parents, out, lse, workspace, fused, fresh,
                 vsplit, chunk, num_warps, num_stages, ramp=None):
    """attend(..., flash="cuda"): windows through the CUDA flash tile (split launch and merge, or fused)."""
    T, hq, d = q.shape
    hkv, dv = key_pages.shape[-2], value_pages.shape[-1]
    R = int(starts.shape[0])
    if hq % hkv or not 1 <= columns <= MAX_COLUMNS or R == 0:
        raise ValueError("query heads a multiple of the KV heads, 1..64 columns a row, one row or more")
    if (offsets is None) != (tokens is None):
        raise ValueError("offsets come with the partial rows they take (tokens)")
    if d not in CUDA_WIDTHS or dv not in CUDA_WIDTHS:
        raise ValueError("cuda launches take head widths of 64, 128 or 256")
    if not _aligned16(key_pages, value_pages):
        raise ValueError("cuda launches take pages 16-byte aligned")
    # The launch reads int32 tables, rows, starts, counts and offsets and int64 positions: others are converted.
    tables, rows, starts = _as(tables, torch.int32), _as(rows, torch.int32), _as(starts, torch.int32)
    firsts = _as(firsts, torch.int64)
    counts = _as(counts, torch.int32) if counts is not None else None
    offsets = _as(offsets, torch.int32) if offsets is not None else None
    t_kind = _cuda_dtype(q, kind)
    g = hq // hkv
    device = q.device
    scale = float(scale if scale is not None else d ** -0.5)
    block = cuda_block(d, dv, hq, hkv, page, kind)
    chunk = int(chunk or cuda_chunk(d, dv, hq, hkv, page, kind))
    if chunk % block:
        raise ValueError(f"a cuda split ({chunk} positions) is a multiple of the block ({block})")
    lib = _cuda_lib(device, T=t_kind, KIND=k_kind, D=d, DV=dv, W=int(num_warps or 0), S=int(num_stages or 0),
                    VS=int(vsplit or 0) if fused else 0)
    if fused and offsets is None and workspace is None and _fused_as_window(
            R * columns, hq, cuda_splits(tables.shape[1] * page, chunk,
                                         tuple(ramp) if ramp is not None else cuda_ramp(d, dv, hq, hkv, page, kind)),
            dv):
        # Few partials: the window launch and the staged merge (the same bits), whose merge spreads over the GPU
        # where a fused tile's folds run one after another.
        fused = False
        lib = _cuda_lib(device, T=t_kind, KIND=k_kind, D=d, DV=dv, W=int(num_warps or 0), S=int(num_stages or 0),
                        VS=0)
    structure = _CUDA_FUSED if fused else _CUDA_WINDOW
    entry, warps, _, built_block, split = _cuda_plan(lib, structure, hq, hkv, page, columns, fresh is not None,
                                                     tables.shape[1] * page)
    if built_block != block:
        raise RuntimeError(f"the build's cuda block ({built_block}) differs from cuda_block() ({block})")
    ramp = tuple(ramp) if ramp is not None else cuda_ramp(d, dv, hq, hkv, page, kind)
    if ramp[0] and chunk % (block << ramp[0]):
        raise ValueError(f"a cuda ramp of {ramp[0]} splits a chunk of {chunk} positions into whole blocks of {block}")
    parts = cuda_splits(tables.shape[1] * page, chunk, ramp)
    tiles = -(-columns * g // (16 * warps))
    out = torch.empty(T, hq, dv, dtype=q.dtype, device=device) if out is None else out
    _cuda_rows_ok(q, out)
    P = _cuda_params(q, key_pages, value_pages, key_scales, value_scales, out, lse, scale, multipliers, hq, hkv, page)
    if fused:
        dvp = dv // split
        group, depth, slots, run, sync = _fused_workspace(device, tiles * split * R * hkv, 16 * warps, dvp, parts)
        states = slots.numel() // (dvp // 2 + 2)   # one a row of a slot
        P.po = slots.data_ptr()
        P.pm = slots[states * (dvp // 2):].data_ptr()
        P.pl = slots[states * (dvp // 2 + 1):].data_ptr()
        P.run = run.data_ptr() if run.numel() else 0
        P.sync = sync.data_ptr()
        P.group, P.tiles, P.ring = group, tiles, depth
    else:
        partial_rows = R * columns if offsets is None else int(tokens)
        po, pm, pl = _workspace(device, partial_rows, hq, parts, dv, workspace)
        P.po, P.pm, P.pl = po.data_ptr(), pm.data_ptr(), pl.data_ptr()
    flags = 0
    P.tables, P.rows, P.starts, P.firsts = tables.data_ptr(), rows.data_ptr(), starts.data_ptr(), firsts.data_ptr()
    P.tab_stride, P.tab_width = tables.stride(0), tables.shape[1]
    if counts is not None:
        flags |= CUDA_FLAGS["counts"]
        P.counts = counts.data_ptr()
    if offsets is not None:
        flags |= CUDA_FLAGS["offsets"]
        P.offsets = offsets.data_ptr()
    if parents is not None:
        par = parents.reshape(-1)
        if par.stride(0) != 1 or par.dtype not in (torch.int32, torch.int64):
            raise ValueError("parents are contiguous int32 or int64")
        flags |= CUDA_FLAGS["tree"] | (CUDA_FLAGS["par64"] if par.dtype == torch.int64 else 0)
        P.parents = par.data_ptr()
    if lse is not None:
        flags |= CUDA_FLAGS["lse"]
    if fresh is not None:
        fk, fv, fshift, ffirst, fstride = _flash_fresh(fresh, R, hkv, d, dv)
        if fk.dtype != q.dtype or fv.dtype != q.dtype or fk.stride(0) % 8 or fv.stride(0) % 8 or not _aligned16(fk, fv):
            raise ValueError("fresh rows in q's dtype, 16-byte aligned")
        fshift, ffirst = _as(fshift, torch.int64), _as(ffirst, torch.int64)
        flags |= CUDA_FLAGS["fresh"]
        P.fk, P.fv, P.fshift, P.ffirst = fk.data_ptr(), fv.data_ptr(), fshift.data_ptr(), ffirst.data_ptr()
        P.stride_fk, P.stride_fv, P.stride_fr = fk.stride(0), fv.stride(0), fstride
    P.n_parts, P.chunk, P.cols, P.flags, P.block = parts, chunk, columns, flags, block
    P.ramp, P.ramp_k = ramp
    _cuda_launch(lib, structure, entry, P, device, (tiles, parts, R * hkv))
    if not fused:
        # The staged merge: deep batches where its programs fit the GPU at once, shallow ones where they do not.
        P.merge = 1 if columns * hq * R <= _capability(device)[1] else 2
        _cuda_merge(lib, P, device, (columns, hq, R))
    return out


def _prefill_cuda(q, kp, vp, kind, k_kind, table, position, page, ks, vs, multipliers, scale, out, lse, fresh, first):
    """prefill(..., flash="cuda"): the chunk as one row of T columns through the CUDA flash tile, its positions split on
    the device over SPLITS programs a tile (merged), or with one split read in one range and written directly."""
    T, hq, d = q.shape
    hkv = kp.shape[-2]
    g = hq // hkv
    device = q.device
    if d not in CUDA_WIDTHS or not _aligned16(kp, vp):
        raise ValueError("cuda launches take head widths of 64, 128 or 256 and pages 16-byte aligned")
    table = _as(table, torch.int32)
    position = _as(position, torch.int64) if torch.is_tensor(position) else position
    first = _as(first, torch.int64) if torch.is_tensor(first) else first
    t_kind = _cuda_dtype(q, kind)
    _cuda_rows_ok(q, out)
    tuned = _knobs("cuda-prefill", {"D": d, "HQ": hq, "HKV": hkv, "PAGE": int(page), "KIND": k_kind, "INT8": 0}, T)
    sms = _capability(device)[1]
    lib = _cuda_lib(device, T=t_kind, KIND=k_kind, D=d, DV=d, W=0, S=0, VS=0)
    entry, warps, _, block, _ = _cuda_plan(lib, _CUDA_PREFILL, hq, hkv, page, T)
    tiles = -(-T * g // (16 * warps))
    splits = int(tuned["SPLITS"])
    if not splits:
        # About two programs an SM: the chunk's tiles times its KV heads, split while they are fewer.
        splits = max(1, min(64, (2 * sms) // max(1, tiles * hkv)))
    one = splits == 1
    P = _cuda_params(q, kp, vp, ks if k_kind >= 2 else None, vs if k_kind >= 2 else None, out, lse,
                     float(scale if scale is not None else d ** -0.5), multipliers, hq, hkv, page)
    flags = CUDA_FLAGS["one_row"] | CUDA_FLAGS["dev_chunk"] | (CUDA_FLAGS["one_split"] if one else 0)
    P.tables, P.tab_stride, P.tab_width = table.data_ptr(), 0, table.shape[0]
    if torch.is_tensor(position):
        flags |= CUDA_FLAGS["pos_t"]
        P.firsts = position.data_ptr()
    else:
        P.first0 = int(position)
    if lse is not None:
        flags |= CUDA_FLAGS["lse"]
    if fresh is not None:
        fk, fv = fresh[:2]
        if fk.dtype != q.dtype or fv.dtype != q.dtype or not _aligned16(fk, fv) or fk.stride(0) % 8 or \
                fv.stride(0) % 8:
            raise ValueError("fresh rows in q's dtype, 16-byte aligned")
        flags |= CUDA_FLAGS["fresh"]
        P.fk, P.fv, P.stride_fk, P.stride_fv = fk.data_ptr(), fv.data_ptr(), fk.stride(0), fv.stride(0)
        if torch.is_tensor(first):
            flags |= CUDA_FLAGS["first_t"]
            P.ffirst = first.data_ptr()
        else:
            P.ffirst0 = int(first)
    if not one:
        po, pm, pl = _workspace(device, T, hq, splits, d)
        P.po, P.pm, P.pl = po.data_ptr(), pm.data_ptr(), pl.data_ptr()
    P.n_parts, P.chunk, P.cols, P.flags, P.block = splits, 0, T, flags, block
    _cuda_launch(lib, _CUDA_PREFILL, entry, P, device, (tiles, 1 if one else splits, hkv))
    if not one:
        _cuda_merge(lib, P, device, (T, hq, 1))
    return out


def prefill(q, key_pages, value_pages, kind, table, position, page, *, key_scales=None, value_scales=None,
            multipliers=(1.0, 1.0), scale=None, out=None, lse=None, int8_qk=False, fresh=None, flash=False):
    """A prompt chunk of one sequence: q [T][heads][dim] at positions position.. (their keys and values written),
    table [pages] int32 -> out [T][heads][dim] in q's dtype (and lse [T][heads] FP32 when given), causal. position: an
    int, or a one-element int64 device tensor read on the device (one launch for every position: a captured graph of
    the chunk replays at any; its keys reach position + T). fresh: (keys, values) [T][kv_heads][dim] in q's dtype, the
    chunk's own as the projections gave them, or (keys, values, first): the chunk then reads its own positions from
    `first` on (default: all of them; an int or a one-element int64 device tensor, position + T or more: none) from
    them, not from the pages (where they were stored at the page kind's precision), and the earlier positions from
    the pages.

    flash: the flash structure (structure "flash-prefill", _flash.py) instead: the chunk's tokens times a KV head's
    query heads packed into tiles of up to 128 rows, its positions split on the device over the tuning's splits
    (0: about two programs an SM) and merged by their states, blocks every row of a tile sees in full read without
    the causal mask, exp2 with the scale folded in. Another reduction than the split launch's and than this one's:
    a token's bits follow the chunk's length (its split), not only its own positions."""
    k_kind = _check_kind(kind, key_scales)
    if int8_qk and k_kind != 3:
        raise ValueError("int8_qk multiplies INT8 queries with INT8 keys: int8 pages only")
    T, hq, d = q.shape
    hkv = key_pages.shape[-2]
    if value_pages.shape[-1] != d:
        raise ValueError("prompt chunks read keys and values of one width")
    device = q.device
    out = torch.empty_like(q) if out is None else out
    kp, vp = _codes(key_pages, kind, device), _codes(value_pages, kind, device)
    ks = key_scales if key_scales is not None else _dummy(device)
    vs = value_scales if value_scales is not None else ks
    tuned = _knobs("prefill", {"D": d, "HQ": hq, "HKV": hkv, "PAGE": int(page), "KIND": k_kind,
                               "INT8": int(bool(int8_qk))}, T)
    block_m = tuned["BLOCK_M"]
    block_n = tuned["BLOCK_N"] or (64 if d <= 128 else 32)
    on_device = torch.is_tensor(position)
    first = 0
    if fresh is not None:
        fk, fv = fresh[:2]
        first = fresh[2] if len(fresh) > 2 else position
        if fk.shape != (T, hkv, d) or fv.shape != (T, hkv, d) or fk.stride(-1) != 1 or fv.stride(-1) != 1 or \
                fk.stride(1) != d or fv.stride(1) != d:
            raise ValueError(f"fresh keys and values are [{T}][{hkv}][{d}], each head row contiguous")
    else:
        fk = fv = q
    if flash == "cuda" and not int8_qk:
        return _prefill_cuda(q, kp, vp, kind, k_kind, table, position, page, ks, vs, multipliers, scale, out, lse,
                             fresh, first)
    if flash:
        return _prefill_flash(q, kp, vp, kind, k_kind, table, position, page, ks, vs, multipliers, scale, out, lse,
                              int8_qk, fresh, first)
    first_t = torch.is_tensor(first)
    _kernels._prefill_kernel[(triton.cdiv(T, block_m), hq)](
        q, kp, vp, ks, vs, table, out, lse if lse is not None else ks, position if on_device else table, fk, fv,
        first if first_t else table, T, 0 if on_device else int(position), 0 if first_t else int(first),
        float(scale if scale is not None else d ** -0.5), float(multipliers[0]), float(multipliers[1]), q.stride(0),
        q.stride(1), out.stride(0), out.stride(1), fk.stride(0), fv.stride(0), HQ=hq, HKV=hkv, G=hq // hkv, D=d,
        BLOCK_M=block_m, BLOCK_N=block_n, PAGE=page, KIND=k_kind, HW_FP8=_hardware_fp8(device), INT8_QK=int8_qk,
        LSE=lse is not None, POS_T=on_device, FRESH=fresh is not None, FIRST_T=first_t, num_warps=tuned["WARPS"],
        num_stages=tuned["STAGES"] or (3 if int8_qk else 2))
    return out
