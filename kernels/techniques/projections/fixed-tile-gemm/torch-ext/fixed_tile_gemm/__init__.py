"""A row-invariant prompt GEMM from crit's quantized-projections technique: x [M][K] times dense [N][K] (both BF16 or
both FP16) on tensor cores with FP32 accumulation, in fixed tiles with one K loop whatever M is (no autotuning, no
split of K), so a row's result is the same bits whatever rows share the GEMM. torch's GEMM picks its kernel by the row
count and so does not promise it; with this one, a model can put the prompts of several sequences into one GEMM and
still give each the result of a GEMM of its own.

Knobs (kernel.json's structure mm, tuned per shape and row count by crit kernels sweep): the tile (BLOCK_M, BLOCK_N),
num_warps, num_stages, and SMALL (take MM_SMALL's tile where the knobs' tile would not fill the device twice over) only
pick which outputs a program computes and how its loads are staged; BLOCK_K, the K step of the one K loop, is an order
knob, the same for every row count of a shape. The defaults are MM_TILE with SMALL on: an untuned build runs the tiles
it ran before knobs; at 1 to 32 rows (default_rows) a 16 x 64 tile in 4 warps and 6 stages, no switch.

mm_split (structure mm-split, row_invariant false): the same GEMM with K split over SPLIT programs a tile, each
writing its FP32 partial sums, then one pass adding the partials in split order and rounding once: more programs for
narrow matrices at a few rows (a matrix of 5,120 outputs has 40 to 80 tiles on a device of 170 multiprocessors). The
sums follow the split, which a tuning may choose by the row count: a row's bits may change with the rows beside it
(deterministic on a repeat). For prompts whose rows need not keep their bits (quantized-projections' prompt_gemm
native); mm stays the row-invariant GEMM (fixed_tiles, pinned).

mm_cuda / mm_cuda_split (structures cuda-mm, cuda-mm-split): the same GEMMs in CUDA (cuda_gemm.cu, compiled by the build
for this GPU, kernel.json "cuda"): mm_cuda row invariant (every output summed over K in k16 steps in one accumulator
whatever the configuration and the row count), mm_cuda_split with K split over CTAs (not row invariant). Rows 16-byte
aligned and K a multiple of 8 (cuda_ok); out in x's dtype or FP32."""
import torch
import triton
import triton.language as tl

from crit_kcore import load as _load   # the kernel core's grouped tile order (kernels/core)

from ._crit_tuning import knobs as _knobs
from . import _crit_cuda

__all__ = ["mm", "mm_split", "mm_cuda", "mm_cuda_split", "cuda_ok", "tile", "split_tile", "MM_TILE", "MM_SMALL"]


@triton.jit
def _mm_kernel(x_ptr, w_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    tm, tn = _load.grouped_tile(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)
    rm = tm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kmask = k0 + rk < K
        a = tl.load(x_ptr + rm[:, None] * stride_xm + (k0 + rk)[None, :], mask=(rm < M)[:, None] & kmask[None, :],
                    other=0.0)
        b = tl.load(w_ptr + rn[:, None] * stride_wn + (k0 + rk)[None, :], mask=(rn < N)[:, None] & kmask[None, :],
                    other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=(rm < M)[:, None] & (rn < N)[None, :])


@triton.jit
def _split_kernel(x_ptr, w_ptr, p_ptr, M, N, K, KC, stride_xm, stride_wn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """A tile's sums over K range [s KC, (s + 1) KC) (s = program_id(1)) into the partials p[s][M][N] (FP32)."""
    tm, tn = _load.grouped_tile(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)
    s = tl.program_id(1)
    rm = tm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    k_lo = s * KC
    k_hi = tl.minimum(K, k_lo + KC)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        kmask = k0 + rk < k_hi
        a = tl.load(x_ptr + rm[:, None] * stride_xm + (k0 + rk)[None, :], mask=(rm < M)[:, None] & kmask[None, :],
                    other=0.0)
        b = tl.load(w_ptr + rn[:, None] * stride_wn + (k0 + rk)[None, :], mask=(rn < N)[:, None] & kmask[None, :],
                    other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(p_ptr + s.to(tl.int64) * M * N + rm[:, None].to(tl.int64) * N + rn[None, :], acc,
             mask=(rm < M)[:, None] & (rn < N)[None, :])


@triton.jit
def _split_sum_kernel(p_ptr, out_ptr, M, N, stride_om, S: tl.constexpr, BLOCK: tl.constexpr):
    """out = the partials p[0] + p[1] + ... in split order, rounded once to out's dtype."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mn = M.to(tl.int64) * N
    m = i < mn
    acc = tl.load(p_ptr + i, mask=m, other=0.0)
    for s in tl.static_range(1, S):
        acc += tl.load(p_ptr + s * mn + i, mask=m, other=0.0)
    tl.store(out_ptr + (i // N) * stride_om + i % N, acc.to(out_ptr.dtype.element_ty), mask=m)


# Two tiles of one K loop (BLOCK_K 64, the same dot): the small one where the large one would not fill the device
# twice over. The tile picks which outputs a program computes, never the order an output's sum takes, so a row's
# result is the same in either (tests/test_fixed_tile_gemm.py checks it across the switch). These are the untuned
# build's tiles (the knobs' defaults); a tuning replaces MM_TILE by its knobs' tile, and MM_SMALL keeps the knobs'
# BLOCK_K.
MM_TILE = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 3}
MM_SMALL = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 4, "num_stages": 4}

_SMS = {}


def _sms(device):
    """The device's multiprocessor count."""
    if device not in _SMS:
        _SMS[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[device]


def tile(M, N, K, device):
    """The launch parameters of an [M][K] x [N][K] call: the tuning's knobs for (N, K) at M rows (the defaults,
    MM_TILE, without one), or MM_SMALL with the knobs' BLOCK_K where SMALL is on and the knobs' tile would not fill
    the device twice over."""
    k = _knobs("mm", {"N": N, "K": K}, M)
    t = {"BLOCK_M": k["BLOCK_M"], "BLOCK_N": k["BLOCK_N"], "BLOCK_K": k["BLOCK_K"], "GROUP_M": MM_TILE["GROUP_M"],
         "num_warps": k["WARPS"], "num_stages": k["STAGES"]}
    if k["SMALL"] and triton.cdiv(M, t["BLOCK_M"]) * triton.cdiv(N, t["BLOCK_N"]) < 2 * _sms(device):
        t = dict(MM_SMALL, BLOCK_K=t["BLOCK_K"])
    return t


SPLITS_MAX = 4            # mm-split's most splits of K (SPLIT 0, the untuned rule)
SPLIT_K_MIN = 1024        # and the fewest K values a split takes


def split_tile(M, N, K, device):
    """mm-split's launch parameters for an [M][K] x [N][K] call: the tuning's knobs for (N, K) at M rows, and the
    splits: SPLIT, or with SPLIT 0 as many (up to SPLITS_MAX, each at least SPLIT_K_MIN values of K) as bring the
    tiles to the device's multiprocessors."""
    k = _knobs("mm-split", {"N": N, "K": K}, M)
    t = {"BLOCK_M": k["BLOCK_M"], "BLOCK_N": k["BLOCK_N"], "BLOCK_K": k["BLOCK_K"], "GROUP_M": MM_TILE["GROUP_M"],
         "num_warps": k["WARPS"], "num_stages": k["STAGES"]}
    splits = k["SPLIT"]
    if not splits:
        tiles = triton.cdiv(M, t["BLOCK_M"]) * triton.cdiv(N, t["BLOCK_N"])
        splits = max(1, min(SPLITS_MAX, round(_sms(device) / tiles), K // SPLIT_K_MIN))
    t["SPLIT"] = splits
    return t


def _check(x, dense):
    M, K = x.shape
    N = dense.shape[0]
    if dense.shape[1] != K or dense.dtype != x.dtype or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"activations [{M}][{K}] {x.dtype} for a matrix [{N}][{dense.shape[1]}] {dense.dtype}: "
                         "both BF16 or both FP16")
    if x.stride(-1) != 1 or dense.stride(-1) != 1:
        raise ValueError("x and dense need unit column strides")
    return M, N, K


def mm_split(x: torch.Tensor, dense: torch.Tensor, dtype=None) -> torch.Tensor:
    """x [M][K] times dense [N][K] as mm, with K split over split_tile()'s SPLIT programs a tile (their FP32 partials
    added in split order, one rounding): NOT row invariant (the split may follow the row count); one split is mm's
    launch in the split tile."""
    M, N, K = _check(x, dense)
    out = torch.empty(M, N, dtype=dtype or x.dtype, device=x.device)
    if M == 0 or N == 0:
        return out
    t = split_tile(M, N, K, x.device)
    tiles = triton.cdiv(M, t["BLOCK_M"]) * triton.cdiv(N, t["BLOCK_N"])
    meta = {"BLOCK_M": t["BLOCK_M"], "BLOCK_N": t["BLOCK_N"], "BLOCK_K": t["BLOCK_K"], "GROUP_M": t["GROUP_M"],
            "num_warps": t["num_warps"], "num_stages": t["num_stages"]}
    if t["SPLIT"] == 1:
        _mm_kernel[(tiles,)](x, dense, out, M, N, K, x.stride(0), dense.stride(0), out.stride(0), **meta)
        return out
    splits = t["SPLIT"]
    chunk = triton.cdiv(triton.cdiv(K, splits), t["BLOCK_K"]) * t["BLOCK_K"]
    partials = torch.empty(splits * M * N, dtype=torch.float32, device=x.device)
    _split_kernel[(tiles, splits)](x, dense, partials, M, N, K, chunk, x.stride(0), dense.stride(0), **meta)
    _split_sum_kernel[(triton.cdiv(M * N, 2048),)](partials, out, M, N, out.stride(0), S=splits, BLOCK=2048,
                                                   num_warps=4)
    return out


def mm(x: torch.Tensor, dense: torch.Tensor, dtype=None) -> torch.Tensor:
    """x [M][K] times dense [N][K] (both BF16 or both FP16, unit column stride) -> out [M][N] in dtype (x's by
    default), FP32 accumulation, in the tile tile() picks (MM_TILE's or MM_SMALL's untuned) whatever M is."""
    M, N, K = _check(x, dense)
    out = torch.empty(M, N, dtype=dtype or x.dtype, device=x.device)
    t = tile(M, N, K, x.device)
    grid = (triton.cdiv(M, t["BLOCK_M"]) * triton.cdiv(N, t["BLOCK_N"]),)
    _mm_kernel[grid](x, dense, out, M, N, K, x.stride(0), dense.stride(0), out.stride(0), BLOCK_M=t["BLOCK_M"],
                     BLOCK_N=t["BLOCK_N"], BLOCK_K=t["BLOCK_K"], GROUP_M=t["GROUP_M"], num_warps=t["num_warps"],
                     num_stages=t["num_stages"])
    return out


# -- the CUDA GEMM (cuda_gemm.cu, kernel.json "cuda": compiled by the build, _crit_cuda) --------------------------------
_cuda_libs = {}


def _cuda_lib(device, dtype):
    import ctypes
    key = (str(device), dtype)
    lib = _cuda_libs.get(key)
    if lib is None:
        lib = _crit_cuda.library(device, T=0 if dtype == torch.bfloat16 else 1)
        lib.crit_ftg_plan.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_int)]
        lib.crit_ftg_plan.restype = ctypes.c_int
        lib.crit_ftg_mm.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p,
                                    ctypes.c_longlong, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_void_p]
        lib.crit_ftg_mm.restype = ctypes.c_int
        _cuda_libs[key] = lib
    return lib


def cuda_ok(x, dense):
    """Whether the CUDA GEMM takes these operands: both BF16 or both FP16 on a CUDA device, unit column strides, rows
    16-byte aligned (pointers and row strides) and K a multiple of 8."""
    if not (x.is_cuda and dense.is_cuda) or x.dtype != dense.dtype or x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if x.dim() != 2 or dense.dim() != 2 or x.stride(-1) != 1 or dense.stride(-1) != 1 or x.shape[1] != dense.shape[1]:
        return False
    return (x.shape[1] % 8 == 0 and x.stride(0) % 8 == 0 and dense.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
            and dense.data_ptr() % 16 == 0)


_ERRORS = {1: "this build holds no such configuration (a sweep build of another structure)",
           2: "operands not 16-byte aligned rows with K a multiple of 8", 3: "more shared memory than the GPU gives",
           4: "too many tiles", 5: "the launch failed"}


def _cuda_mm(structure, x, dense, dtype):
    import ctypes
    M, N, K = _check(x, dense)
    if not cuda_ok(x, dense):
        raise ValueError("the CUDA GEMM takes 16-byte aligned rows and K a multiple of 8 (cuda_ok)")
    dtype = dtype or x.dtype
    if dtype not in (x.dtype, torch.float32):
        raise ValueError(f"the CUDA GEMM writes {x.dtype} or float32, not {dtype}")
    out = torch.empty(M, N, dtype=dtype, device=x.device)
    if M == 0 or N == 0:
        return out
    lib = _cuda_lib(x.device, x.dtype)
    plan = (ctypes.c_int * 4)()
    if lib.crit_ftg_plan(structure, N, K, M, plan):
        raise RuntimeError(f"fixed-tile-gemm cuda: {_ERRORS[1]}")
    splits, chunk = plan[2], plan[3]
    part = torch.empty(splits * M * N, dtype=torch.float32, device=x.device) if splits > 1 else None
    stream = torch.cuda.current_stream(x.device).cuda_stream
    err = lib.crit_ftg_mm(structure, x.data_ptr(), x.stride(0), dense.data_ptr(), dense.stride(0), out.data_ptr(),
                          out.stride(0), int(dtype == torch.float32), M, N, K,
                          part.data_ptr() if part is not None else None, splits, chunk, stream)
    if err:
        raise RuntimeError(f"fixed-tile-gemm cuda: {_ERRORS.get(err, err)}")
    return out


def mm_cuda(x: torch.Tensor, dense: torch.Tensor, dtype=None) -> torch.Tensor:
    """x [M][K] times dense [N][K] (both BF16 or both FP16) -> out [M][N] in dtype (x's or FP32), FP32 accumulation
    over K in k16 steps in one accumulator: a row's bits independent of the call's other rows and of the configuration
    (structure cuda-mm, row invariant). Operands as cuda_ok()."""
    return _cuda_mm(0, x, dense, dtype)


def mm_cuda_split(x: torch.Tensor, dense: torch.Tensor, dtype=None) -> torch.Tensor:
    """mm_cuda with K split over CTAs a tile (structure cuda-mm-split: SPLIT, or the device-filling rule), the FP32
    partials added in split order and rounded once: NOT row invariant (the split may follow the row count);
    deterministic."""
    return _cuda_mm(1, x, dense, dtype)
