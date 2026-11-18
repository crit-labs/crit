"""Sparse indexer: for each query token, a score over every compressed key it may see, then the positions of its top
k keys, which a sparse attention then reads.

    s = scores(q, keys, weights, ends)                # FP32 [tokens][keys]
    picked = topk(s, ends, k)                         # int32 [tokens][k]: ascending positions, -1 past the valid ones
    picked = select(q, keys, weights, ends, k)        # both, tokens in groups so the scores stay within a workspace
    s = scores(q, store, weights, ends, slots=slots)  # keys read in place: key j is store[slots[j]], no gathered copy

The score of token t over key s is the sum over its heads h of weights[t][h] x scale x ReLU(q[t][h] . keys[s]) (q
[tokens][heads][dim] BF16 or FP16; keys [n][dim] in q's dtype, or E4M3 codes as float8_e4m3fn with FP32 key_scales
[n], a key's value its code times its scale; weights FP32 [tokens][heads]; scale default dim ** -0.5), for s below
ends[t] (int32 [tokens]: the keys token t sees, a causal bound); keys at or past it score -inf. The products run on
the tensor cores with FP32 sums, ReLU, the head factor and the sum over heads in FP32.

With slots the keys are read where a pool stores them (paged state: a sequence's rows lie wherever its pages
are): keys is the store [rows][dim] (and key_scales FP32 [rows]), slots int32 or int64 [n], or [groups][n] with the
tokens in `groups` equal runs (a round's rows, each a sequence: token t reads slots[t // (tokens / groups)]), and
key j of a token is store[slots[...][j]]. Nothing is gathered: the kernel loads each key row from its slot, and a
token's scores have the bits the gathered keys give.

topk keeps, per token, the k largest scores among its valid keys (all of them where it has k or fewer), ties to the
lower position, a NaN below every number, and writes their positions in ascending order, then -1. Every allocation is
torch's caching allocator's and nothing syncs with the host, so CUDA graphs capture a call once its shape has run.

Knobs (keyed by heads, dim and the key kind, KIND 0 16-bit or 1 E4M3): structure scores TM (tokens a program), BN,
WARPS and STAGES, speed only (the head sum is an explicit tree: no tile changes a bit); structure topk BLOCK (scores a step), WARPS and SLICE (the
positions a program selects among, the rest by a second pass over the slices' picks: speed only, the same picks),
and CUDA (calls of few tokens select in the CUDA kernel, cuda_select.cu: a slice's scores copied to shared memory
once for the five passes the Triton program reads from global memory; the same picks).
"""
from typing import Optional

import torch
import triton

from . import _crit_cuda, _crit_tuning, _kernels

__all__ = ["scores", "select", "topk"]

FP8 = torch.float8_e4m3fn
_KNOBS = {}


def _knobs(structure, key, rows):
    if getattr(_crit_tuning, "SELECTED", None):
        return _crit_tuning.knobs(structure, key, rows)
    at = (structure, rows, *key.values())
    if at not in _KNOBS:
        _KNOBS[at] = _crit_tuning.knobs(structure, key, rows)
    return _KNOBS[at]


def _check_slots(slots, tokens, rows):
    """(n, tokens a group) of `slots`: int32 or int64 [n] or [groups][n] on the keys' device, contiguous rows."""
    if slots.dtype not in (torch.int32, torch.int64) or slots.dim() not in (1, 2) or slots.stride(-1) != 1:
        raise ValueError("slots are int32 or int64 [n] or [groups][n], each row contiguous")
    groups = slots.shape[0] if slots.dim() == 2 else 1
    if groups < 1 or tokens % groups:
        raise ValueError(f"{tokens} tokens are not {groups} equal groups (a group's tokens share its slots)")
    return slots.shape[-1], tokens // groups


def _check_scores(q, keys, weights, ends, key_scales):
    if q.dim() != 3 or q.dtype not in (torch.bfloat16, torch.float16) or q.stride(-1) != 1:
        raise ValueError("q is [tokens][heads][dim] BF16 or FP16, each head row contiguous")
    t, h, d = q.shape
    if d not in (64, 128, 256) or h & (h - 1) or h > 128:
        raise ValueError(f"heads a power of two up to 128 and a dim of 64, 128 or 256, not {h} x {d}")
    if keys.dim() != 2 or keys.shape[1] != d or keys.stride(-1) != 1:
        raise ValueError(f"keys are [n][{d}], each row contiguous")
    fp8 = keys.dtype == FP8
    if not fp8 and keys.dtype != q.dtype:
        raise ValueError("keys are in q's dtype, or E4M3 codes (float8_e4m3fn) with key_scales")
    if fp8 and (key_scales is None or key_scales.dtype != torch.float32 or key_scales.shape != (keys.shape[0],)):
        raise ValueError("E4M3 keys take key_scales FP32 [n]")
    if weights.shape != (t, h) or weights.dtype != torch.float32 or weights.stride(-1) != 1:
        raise ValueError(f"weights are FP32 [{t}][{h}]")
    if ends.shape != (t,) or ends.dtype != torch.int32:
        raise ValueError(f"ends are int32 [{t}]")
    return t, h, d, fp8


def scores(q, keys, weights, ends, *, key_scales=None, scale=None, out: Optional[torch.Tensor] = None,
           slots: Optional[torch.Tensor] = None):
    """Every token's score over keys 0 .. n - 1 (FP32 [tokens][n]; -inf at or past ends[t]). slots: the keys in
    place, key j the store's row slots[j] (the module's description); n is then the slots' count."""
    t, h, d, fp8 = _check_scores(q, keys, weights, ends, key_scales)
    n, group = (keys.shape[0], max(1, t)) if slots is None else _check_slots(slots, t, keys.shape[0])
    out = torch.empty(t, n, dtype=torch.float32, device=q.device) if out is None else out
    if out.shape[0] < t or out.shape[1] < n or out.dtype != torch.float32 or out.stride(-1) != 1:
        raise ValueError(f"out is FP32 [{t}][{n}] or wider, rows contiguous")
    if t == 0 or n == 0:
        return out
    k = _knobs("scores", {"H": h, "D": d, "KIND": int(fp8)}, t)
    tm = max(1, min(int(k["TM"]), 128 // h, t)) if h <= 128 else 1   # no tile rows for tokens the call has not
    while group % tm:      # a program's tokens are of one group (TM is a speed knob: no bit follows it)
        tm -= 1
    bn = int(k["BN"])
    grid = (triton.cdiv(t, tm), triton.cdiv(n, bn))
    _kernels.score_kernel[grid](
        q, keys.view(torch.uint8) if fp8 else keys, key_scales if fp8 else weights, weights, ends, out,
        slots if slots is not None else ends, t, n, group,
        float(scale if scale is not None else d ** -0.5), q.stride(0), q.stride(1), keys.stride(0), weights.stride(0),
        out.stride(0), slots.stride(0) if slots is not None and slots.dim() == 2 else 0,
        H=h, D=d, TM=tm, BN=bn, FP8=fp8, LOG_H=h.bit_length() - 1,
        HW_FP8=torch.cuda.get_device_capability(q.device) >= (8, 9), SLOTS=slots is not None,
        num_warps=int(k["WARPS"]), num_stages=int(k["STAGES"]))
    return out


def topk(s, ends, k, *, out: Optional[torch.Tensor] = None):
    """The positions of each token's k largest scores below its end (s FP32 [tokens][n]), ascending, then -1 (int32
    [tokens][k])."""
    if s.dim() != 2 or s.dtype != torch.float32 or s.stride(-1) != 1:
        raise ValueError("scores are FP32 [tokens][n], rows contiguous")
    t, n = s.shape
    if ends.shape != (t,) or ends.dtype != torch.int32:
        raise ValueError(f"ends are int32 [{t}]")
    if k < 1:
        raise ValueError("k is at least 1")
    out = torch.empty(t, k, dtype=torch.int32, device=s.device) if out is None else out
    if out.shape != (t, k) or out.dtype != torch.int32 or out.stride(-1) != 1:
        raise ValueError(f"out is int32 [{t}][{k}]")
    if t == 0:
        return out
    knobs = _knobs("topk", {"K": int(k)}, t)
    if int(knobs.get("CUDA", 1)) and n > 0 and 2 * t < _multiprocessors(s.device) and \
            _topk_cuda(s, ends, k, out, int(knobs["SLICE"])):
        return out
    return _topk_triton(s, ends, k, out, knobs)


def _topk_triton(s, ends, k, out, knobs):
    """topk in the Triton programs (every token count)."""
    t, n = s.shape
    block, warps, kb = int(knobs["BLOCK"]), int(knobs["WARPS"]), min(1024, triton.next_power_of_2(k))
    width = _slice(t, n, k, int(knobs["SLICE"]), s.device)
    if width >= n:
        _kernels.topk_kernel[(t, 1)](s, s, ends, out, s, n, k, n, s.stride(0), out.stride(0), BLOCK=block, KB=kb,
                                     POS=False, ENDS=True, SOUT=False, num_warps=warps)
        return out
    # Few tokens over many keys: each slice's k best in parallel (positions and scores, in position order), then the
    # k best of those candidates. A token's k best are among its slices' (the same order, ties to the lower
    # position), so the picks are the one pass's.
    parts = -(-n // width)
    pos = torch.empty(t, parts * k, dtype=torch.int32, device=s.device)
    best = torch.empty(t, parts * k, dtype=torch.float32, device=s.device)
    _kernels.topk_kernel[(t, parts)](s, s, ends, pos, best, n, k, width, s.stride(0), pos.stride(0), BLOCK=block,
                                     KB=kb, POS=False, ENDS=True, SOUT=True, num_warps=warps)
    _kernels.topk_kernel[(t, 1)](best, pos, ends, out, best, parts * k, k, parts * k, best.stride(0), out.stride(0),
                                 BLOCK=block, KB=kb, POS=True, ENDS=False, SOUT=False, num_warps=warps)
    return out


# -- the CUDA select (cuda_select.cu, kernel.json "cuda": compiled by the build, _crit_cuda) ----------------------------
_cuda = {}   # device -> (library, a slice's room in scores), or None where the sources do not compile there


def _cuda_lib(device):
    import ctypes
    key = str(device)
    if key not in _cuda:
        try:
            lib = _crit_cuda.library(device)
        except (RuntimeError, OSError):   # no nvcc, or a GPU the sources do not build for: the Triton programs
            _cuda[key] = None
            return None
        lib.crit_si_select.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_longlong] * 2 + \
            [ctypes.c_void_p]
        lib.crit_si_select.restype = ctypes.c_int
        lib.crit_si_slice_room.restype = ctypes.c_int
        with torch.cuda.device(device):
            room = int(lib.crit_si_slice_room())
        _cuda[key] = (lib, room) if room > 0 else None
    return _cuda[key]


def _topk_cuda(s, ends, k, out, knob):
    """topk of a few tokens in the CUDA kernel: one launch where a token's scores fit a block's shared memory, else
    the slices' k best (positions and scores) and a second launch over those candidates, as the Triton programs
    slice. False where the kernel is not there or the scores do not fit (the caller takes the Triton programs)."""
    found = _cuda_lib(s.device)
    if found is None:
        return False
    lib, room = found
    t, n = s.shape
    width = max(int(knob), 2 * k) if knob else max(4096, 4 * k)
    if not knob and n <= min(2 * width, room):   # a short row: one launch
        width = n
    parts = -(-n // width)
    if parts > 1 and 2 * parts * k > room:   # the candidates of the second launch: fewer, wider slices
        parts = room // (2 * k)
        width = -(-n // max(parts, 1))
        parts = -(-n // width) if parts else 0
    if parts < 1 or width > room:
        return False
    with torch.cuda.device(s.device):
        stream = torch.cuda.current_stream(s.device).cuda_stream
        if parts == 1:
            err = lib.crit_si_select(s.data_ptr(), None, ends.data_ptr(), out.data_ptr(), None, t, 1, n, k, width,
                                     s.stride(0), out.stride(0), stream)
        else:
            pos = torch.empty(t, parts * k, dtype=torch.int32, device=s.device)
            best = torch.empty(t, parts * k, dtype=torch.float32, device=s.device)
            err = lib.crit_si_select(s.data_ptr(), None, ends.data_ptr(), pos.data_ptr(), best.data_ptr(), t, parts,
                                     n, k, width, s.stride(0), pos.stride(0), stream)
            err = err or lib.crit_si_select(best.data_ptr(), pos.data_ptr(), None, out.data_ptr(), None, t, 1,
                                            parts * k, k, parts * k, best.stride(0), out.stride(0), stream)
    if err:
        raise RuntimeError(f"sparse-indexer's CUDA select failed ({err})")
    return True


def _slice(tokens, n, k, knob, device):
    """The positions a program of the select takes (n: one pass): the tuning's SLICE, else the rule: slices of
    max(4,096, 4 k) positions where the tokens leave most multiprocessors idle and a row holds two slices or more."""
    if knob:
        return max(int(knob), 2 * k)
    width = max(4096, 4 * k)
    if 2 * tokens >= _multiprocessors(device) or n < 2 * width:
        return n
    return width


_SMS = {}


def _multiprocessors(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SMS:
        _SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _SMS[index]


def select(q, keys, weights, ends, k, *, key_scales=None, scale=None, out: Optional[torch.Tensor] = None,
           workspace: Optional[torch.Tensor] = None, workspace_bytes=64 << 20,
           slots: Optional[torch.Tensor] = None):
    """topk(scores(...)) over groups of tokens whose scores fit the workspace (FP32; default a buffer of
    workspace_bytes), the same positions as one call (each token's scores and choice are its own). slots: the keys
    in place, as scores takes them (one list of slots for every token)."""
    t, h, d, _ = _check_scores(q, keys, weights, ends, key_scales)
    if slots is not None and slots.dim() != 1:
        raise ValueError("select takes one list of slots [n] (its tokens go in groups of its own): scores takes "
                         "[groups][n]")
    n = keys.shape[0] if slots is None else slots.shape[0]
    out = torch.empty(t, k, dtype=torch.int32, device=q.device) if out is None else out
    if t == 0:
        return out
    if n == 0:
        return out.fill_(-1)
    room = (workspace.numel() if workspace is not None else workspace_bytes // 4)
    group = max(1, min(t, room // n))
    if workspace is None:
        workspace = torch.empty(group * n, dtype=torch.float32, device=q.device)
    elif workspace.dtype != torch.float32 or workspace.numel() < n:
        raise ValueError(f"the workspace is FP32 with room for one token's {n} scores at least")
    for t0 in range(0, t, group):
        g = min(group, t - t0)
        s = workspace.view(-1)[:g * n].view(g, n)
        scores(q[t0:t0 + g], keys, weights[t0:t0 + g], ends[t0:t0 + g], key_scales=key_scales, scale=scale, out=s,
               slots=slots)
        topk(s, ends[t0:t0 + g], k, out=out[t0:t0 + g])
    return out
