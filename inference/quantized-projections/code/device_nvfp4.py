"""Weight-only decode rows on a device (16-bit activations times NVFP4 (W4A16), FP8 or MXFP8 (W8A16) and BF16 weights),
the NVFP4 and MXFP8 decoders for prompt GEMMs, and a fixed-tile prompt GEMM: the projections slot's arithmetic that no
provider reachable from torch has.

crit hosts no kernel. No provider reachable from torch covers W4A16 NVFP4 (cuBLASLt's block-scaled GEMM takes NVFP4 on
both operands; FlashInfer's FP4 weight-only kernels sit inside its fused MoE), so the arithmetic comes from crit-labs'
kernels in the Hugging Face kernel registry (crit_plugins.kernels.registry_kernel, declared with their pins in the
plugin's plugin.json "kernels"), and where one is not there (not published yet, no build for this machine, offline)
from a plain torch fallback with the same contract, which the engine's facts then name under "kernels":

- crit-labs/decode-gemv: `gemv` (CUDA cores, NVFP4 W4A16, 1 to 16 rows), `gemv_fp8` (CUDA cores, FP8 E4M3 with a row
  scale or MXFP8's E8M0 scale per 32 inputs, W8A16, 1 to 16 rows), `gemv_bf16` (CUDA cores, BF16 weights, 1 to 16
  rows), `gemv_triton` (Triton, NVFP4 W4A16, 1 to 8 rows), `decode` and `decode_fp8` (an NVFP4 matrix's values for a
  provider's GEMM).
- crit-labs/fixed-tile-gemm: `mm`, a tensor-core GEMM with one K loop and fixed tiles whatever the row count, so a
  row's result does not depend on the rows beside it.
- crit-labs/mma-rows: `mm_rows`, 1 to 64 rows of any format on tensor cores in one fixed tile (the weights read once a
  call, every value exact in BF16, FP32 sums, the prompt rows' quantized activations computed by its pre-pass), and
  its quantizers; device_matrices calls it directly (_tile).

Every path keeps the projections contract: up to MAX_ROWS rows (TRITON_ROWS on the Triton GEMV), each row's bits are
those of the row alone. The
kernels reduce in a fixed order per row; the fallbacks multiply one row at a time (torch's GEMM picks its kernel by the
row count, never by which row it is). Their sums go in different orders, so a process keeps to one: a matrix takes its
path when it loads.

The matrix as the checkpoint stores it: codes [rows][columns / 2] bytes (code 2i in the low nibble of byte i), E4M3
block scales [rows][columns / 16] row major, and each output row's inverse global scale (nvfp4.py's divisor convention;
one per matrix, or one per row for matrices stacked by rows). A value is e2m1(code) * e4m3(scale) * inverse; e2m1 *
e4m3 has at most 6 significant bits, so it is exact in BF16 and FP16, and the fallback GEMV multiplies by those exact
values, sums in FP32 and applies the inverse once, as the kernels do. The fallback decoders give the kernels' bits:
one FP32 product e2m1 * scale * factor rounded once to the output type. An E4M3 value (times a power-of-two block scale
for MXFP8) is exact in BF16 too, so the FP8 fallback GEMV multiplies by exact values the same way and applies the row
scale once; the MXFP8 decoder (torch operations, no kernel) writes those exact values for a prompt GEMM.
"""
import torch

from . import device_matrices

MAX_ROWS = 16      # the CUDA GEMVs' and the fallbacks' most rows a call (device_matrices.GEMV_ROWS)
TRITON_ROWS = 8    # the Triton GEMV's
ROWS, TILES, MMA, PINNED = ("crit-labs/decode-gemv", "crit-labs/fixed-tile-gemm", "crit-labs/mma-rows",
                             "crit-labs/pinned-gemm")

# E2M1 values of codes 0..15 (bit 3 the sign), the fallback decoders' table.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

def kernel(name):
    """The registry kernel's module (crit_plugins.kernels.registry_kernel at the pin the plugin's plugin.json declares:
    device_matrices.KERNEL_FOLDER, which quantized-projections' plugin.py sets, and a plugin that runs this code for its
    own device form sets to its folder; crit_plugins.kernels caches the module and the manifest), or None: the
    fallbacks serve, and crit_plugins.kernels records why for the engine's facts. Without a plugin folder no kernel is
    asked for."""
    from crit_plugins.kernels import registry_kernel
    folder = device_matrices.KERNEL_FOLDER
    return registry_kernel(name, folder=folder) if folder is not None else None


# (kernel, shape) of each Triton launch that ran eagerly: Triton compiles (and the GEMV autotunes) on a kernel's first
# call for a shape, which a CUDA graph capture cannot hold.
_WARM = set()


def _warmed(key, what):
    """Refuse a Triton kernel's first call for `key` inside a capture, with what to do instead; note an eager one."""
    if torch.cuda.is_current_stream_capturing():
        if key not in _WARM:
            raise RuntimeError(f"quantized-projections: {what} compiles on its first call for this shape, which a "
                               "captured round cannot hold: run the step once eagerly before capturing it")
    else:
        _WARM.add(key)


def rows_kernel():
    """crit-labs/decode-gemv's module, or None (the torch fallbacks serve)."""
    return kernel(ROWS)


def tiles_kernel():
    """crit-labs/fixed-tile-gemm's module, or None (the row-at-a-time fallback serves)."""
    return kernel(TILES)


def mma_kernel():
    """crit-labs/mma-rows' module, or None (decode rows go through the GEMVs)."""
    return kernel(MMA)


def mma(device):
    """crit-labs/mma-rows (its tensor-core tile for 1 to 64 rows) for `device`, or None where it is not here."""
    if torch.device(device).type != "cuda":
        return None
    module = mma_kernel()
    return module if module is not None and callable(getattr(module, "mm_rows", None)) else None


def pinned(device):
    """crit-labs/pinned-gemm (cuBLASLt GEMMs pinned to one algorithm for every row count) for `device`, or None where
    it is not here."""
    if torch.device(device).type != "cuda":
        return None
    module = kernel(PINNED)
    return module if module is not None and callable(getattr(module, "pin", None)) else None


def dot(device):
    """crit-labs/decode-gemv's FP8 dot GEMV (Triton, tensor cores, 1 to 16 rows) for `device`, or None where it is not
    here."""
    if torch.device(device).type != "cuda":
        return None
    module = rows_kernel()
    return module if module is not None and callable(getattr(module, "gemv_fp8_dot", None)) else None


def native(device):
    """The CUDA GEMVs (crit-labs/decode-gemv's gemv and gemv_bf16) for `device`, or None where the kernel is not
    available here: gemv then runs the Triton GEMV where that is available, else the torch fallback."""
    if torch.device(device).type != "cuda":
        return None
    module = rows_kernel()
    return module if module is not None and callable(getattr(module, "gemv", None)) else None


def triton(device):
    """crit-labs/decode-gemv's Triton GEMV for `device`, or None (the torch fallback)."""
    if torch.device(device).type != "cuda":
        return None
    module = rows_kernel()
    return module if module is not None and callable(getattr(module, "gemv_triton", None)) else None


_TABLES = {}


def prepare(device):
    """The tables on `device`, made at load (here and device_matrices.constants): a table made on first use inside a
    captured graph would copy from the host during the capture."""
    key = device_matrices.device_key(device)
    if key not in _TABLES:
        _TABLES[key] = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=device)
        device_matrices.constants(device)
    return _TABLES[key]


def _table(device):
    return prepare(device)


def _values(codes, scales, factor, dtype):
    """e2m1(code) * e4m3(scale) [* factor[row]] in FP32, rounded once to dtype: [rows][columns]."""
    table = _table(codes.device)
    rows = codes.shape[0]
    values = torch.stack((table[(codes & 15).long()], table[(codes >> 4).long()]), dim=-1).view(rows, -1, 16)
    values = (values * scales.float()[..., None]).view(rows, -1)
    if factor is not None:
        values = values * factor.reshape(-1)[:, None]
    return values.to(dtype)


def _aligned(x):
    """x with unit column stride and rows on 16-byte boundaries (what the kernels read; a copy where it is not)."""
    x = x if x.stride(-1) == 1 else x.contiguous()
    if x.stride(0) % 8 or x.data_ptr() % 16:
        x = x.clone()
    return x


CHUNK_BYTES = 64 << 20   # the fallbacks decode a matrix this many FP32 bytes of rows at a time


def chunks(rows, columns):
    """The row ranges the fallbacks decode one at a time: fixed by the matrix's shape, never by the activation rows."""
    step = max(1, CHUNK_BYTES // (columns * 4))
    return [(n, min(rows, n + step)) for n in range(0, rows, step)]


def one_row_at_a_time(x, weight, dtype):
    """x [M][K] times weight [N][K] (same dtype) -> [M][N] in dtype: one torch GEMM a row, so each row's bits are those
    of the row alone (torch picks its kernel by the row count, the same for every row here)."""
    found = torch.empty(x.shape[0], weight.shape[0], dtype=dtype, device=x.device)
    for r in range(x.shape[0]):
        row = _aligned(x[r:r + 1])
        if x.dtype == torch.float32 or dtype == x.dtype:
            found[r:r + 1] = torch.mm(row, weight.t()).to(dtype)
        elif not x.is_cuda:   # torch has no CPU kernel for a 16-bit GEMM into FP32: the 16-bit values in FP32
            found[r:r + 1] = (row.float() @ weight.float().t()).to(dtype)
        else:
            found[r:r + 1] = torch.mm(row, weight.t(), out_dtype=dtype)
    return found


def gemv(x, codes, scales, inverse, out=None, dtype=None, native=None, triton=None, blocked=False):
    """x [M][K] (M <= 16, 8 on the Triton GEMV; 16-bit) times the NVFP4 matrix (codes [N][K / 2] uint8, scales [N][K / 16] E4M3, inverse
    [N] FP32: 1 / the row's global scale) -> out [M][N], FP32 accumulation, written in out's dtype (dtype, FP32 by
    default). native: decode-gemv's module for its CUDA GEMV (native()), which needs K a multiple of 32 and 16-bit
    activations; triton: the module for its Triton GEMV (triton()); neither: the torch fallback. blocked: the scales are
    in cuBLASLt's 128 x 4 tiled layout (device_matrices.to_blocked) instead of [N][K / 16] (the CUDA GEMV and the
    fallback read it; the Triton GEMV does not)."""
    M, K = x.shape
    N = codes.shape[0]
    if not 1 <= M <= (TRITON_ROWS if native is None and triton is not None else MAX_ROWS):
        raise ValueError(f"the NVFP4 GEMV takes 1 to {MAX_ROWS} rows ({TRITON_ROWS} in Triton), not {M}")
    if K % 16 or codes.shape[1] * 2 != K:
        raise ValueError(f"activations of width {K} for codes of width {codes.shape[1] * 2}")
    x = x if x.stride(-1) == 1 else x.contiguous()
    out = torch.empty(M, N, dtype=dtype or torch.float32, device=x.device) if out is None else out
    if native is not None:
        if K % 32 or x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"the CUDA NVFP4 GEMV takes 16-bit activations of a width that is a multiple of 32, not "
                             f"{x.dtype} of {K}")
        x = _aligned(x)
        if out.dtype in (torch.float32, x.dtype) and out.stride(-1) == 1:
            native.gemv(x, codes, scales, inverse, out, blocked=blocked)
        else:
            found = torch.empty(M, N, dtype=torch.float32, device=x.device)
            native.gemv(x, codes, scales, inverse, found, blocked=blocked)
            out.copy_(found)
        return out
    if triton is not None and not blocked:
        _warmed(("gemv", N, K, M, x.dtype, out.dtype), "the Triton GEMV")
        triton.gemv_triton(x, codes, scales, inverse, out)
        return out
    if blocked:
        from .device_matrices import from_blocked
        scales = from_blocked(scales, N, K // 16).view(torch.float8_e4m3fn)
    # The fallback: the exact values e2m1 * scale in x's type, a chunk of matrix rows at a time, each activation row
    # on its own, FP32 sums, the inverse once.
    kind = x.dtype if x.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.float32
    rows, inverse = x.to(kind), inverse.reshape(-1)
    for n0, n1 in chunks(N, K):
        sums = one_row_at_a_time(rows, _values(codes[n0:n1], scales[n0:n1], None, kind), torch.float32)
        out[:, n0:n1].copy_(sums * inverse[n0:n1])
    return out


def gemv_fp8(x, weight, scales, out, native=None):
    """x [M][K] (M <= 16, BF16 or FP16) times an FP8 matrix (weight [N][K] E4M3) with scales FP32 [N] (one per row,
    applied after the sum) or uint8 [N][K / 32] (MXFP8's E8M0 exponents, each value times 2^(e - 127)) -> out [M][N]
    (FP32 or x's dtype), FP32 sums: decode-gemv's CUDA GEMV where given, else the exact values in x's type a chunk of rows
    at a time, one activation row per torch GEMM, the row scale once."""
    M, K = x.shape
    N = weight.shape[0]
    if not 1 <= M <= MAX_ROWS or weight.shape[1] != K:
        raise ValueError(f"the FP8 GEMV takes 1 to {MAX_ROWS} rows of the matrix's width {weight.shape[1]}, not "
                         f"{tuple(x.shape)}")
    mx = scales.dtype == torch.uint8
    if native is not None and callable(getattr(native, "gemv_fp8", None)):
        found = out if out.dtype in (torch.float32, x.dtype) and out.stride(-1) == 1 else \
            torch.empty(M, N, dtype=torch.float32, device=x.device)
        native.gemv_fp8(_aligned(x), weight, scales, found)
        if found is not out:
            out.copy_(found)
        return out
    # The values in BF16 (exact: an E4M3 code, times a power of two for MXFP8) against BF16 rows; FP16 rows keep their
    # 11 bits against the values in FP32 (an MXFP8 value can be past FP16's range).
    kind = torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float32
    rows = x.to(kind)
    for n0, n1 in chunks(N, K):
        values = weight[n0:n1].float()
        if mx:
            values = mxfp8_values(values, scales[n0:n1])
        sums = one_row_at_a_time(rows, values.to(kind), torch.float32)
        out[:, n0:n1].copy_(sums if mx else sums * scales.reshape(-1)[n0:n1])
    return out


def mxfp8_values(values, exponents):
    """E4M3 values [N][K] (FP32) times 2^(e - 127) per 32 inputs (E8M0 exponents [N][K / 32]); 255 is NaN, as the
    kernel reads it."""
    e = exponents.int().repeat_interleave(32, dim=1)
    scaled = torch.ldexp(values, e - 127)
    return torch.where(e == 255, torch.full_like(scaled, float("nan")), scaled)


def decode_mxfp8(weight, exponents, out):
    """An MXFP8 matrix's values [N][K] (weight E4M3, exponents [N][K / 32] E8M0) in out's dtype (BF16: exact), a chunk
    of rows at a time, by torch operations."""
    N, K = out.shape
    for n0, n1 in chunks(N, K):
        out[n0:n1].copy_(mxfp8_values(weight[n0:n1].float(), exponents[n0:n1]).to(out.dtype))
    return out


def gemv_bf16(x, weight, out, native=None):
    """x [M][K] BF16 (M <= 16) times weight [N][K] BF16 -> out [M][N] (FP32 or BF16), FP32 sums: decode-gemv's CUDA
    GEMV where given (K a multiple of 8), else one row at a time through torch."""
    if native is not None:
        native.gemv_bf16(_aligned(x), weight, out)
        return out
    out.copy_(one_row_at_a_time(x, weight, out.dtype))
    return out


# A call's module argument when the caller holds none (it is looked up); a matrix passes the one it holds, None too.
LOOKUP = object()


def decode(codes, scales, inverse, out, module=LOOKUP):
    """The matrix's values [N][K] in out's dtype (16-bit): e2m1 * scale / global scale, rounded once. module:
    decode-gemv's as the matrix holds it from load (None: the fallback), looked up when not given."""
    module = rows_kernel() if module is LOOKUP else module
    if module is not None and callable(getattr(module, "decode", None)) and codes.is_cuda:
        _warmed(("decode", out.dtype), "the Triton decoder")
        module.decode(codes, scales, inverse, out)
        return out
    inverse = inverse.reshape(-1)
    for n0, n1 in chunks(*out.shape):
        out[n0:n1].copy_(_values(codes[n0:n1], scales[n0:n1], inverse[n0:n1], out.dtype))
    return out


def row_scales(scales, inverse):
    """Each row's FP8 scale: its largest representable value (6 times its largest block scale, over its global
    scale) at 448."""
    return (scales.float().amax(-1) * 6.0 * inverse / 448.0).clamp_min(1e-30)


def decode_fp8(codes, scales, factor, out, module=LOOKUP):
    """The matrix's values [N][K] as FP8 E4M3 in units of each row's scale: e2m1 * scale * factor, factor [N] = 1 /
    (global scale * row scale) (row_scales), rounded once (to nearest even, within +-448). module: as decode's."""
    module = rows_kernel() if module is LOOKUP else module
    if module is not None and callable(getattr(module, "decode_fp8", None)) and codes.is_cuda:
        _warmed(("decode_fp8",), "the Triton FP8 decoder")
        module.decode_fp8(codes, scales, factor, out)
        return out
    factor = factor.reshape(-1)
    for n0, n1 in chunks(*out.shape):
        out[n0:n1].copy_(_values(codes[n0:n1], scales[n0:n1], factor[n0:n1], torch.float32)
                         .clamp_(-448.0, 448.0).to(out.dtype))
    return out


def mm(x, dense, dtype=None, module=LOOKUP):
    """x [M][K] times dense [N][K] (both BF16 or both FP16) -> out [M][N] in dtype (x's by default), FP32
    accumulation, each row's result independent of the rows beside it: crit-labs/fixed-tile-gemm's fixed tiles, or
    one row at a time through torch where that kernel is not available (correct, much slower)."""
    M, K = x.shape
    N = dense.shape[0]
    if dense.shape[1] != K or dense.dtype != x.dtype:
        raise ValueError(f"activations [{M}][{K}] {x.dtype} for a matrix [{N}][{dense.shape[1]}] {dense.dtype}")
    module = tiles_kernel() if module is LOOKUP else module   # fixed-tile-gemm's, as the matrix holds it from load
    if module is not None and x.is_cuda:
        _warmed(("mm", N, K, M, x.dtype, dtype), "the fixed-tile GEMM")
        return module.mm(x if x.stride(-1) == 1 else x.contiguous(), dense, dtype or x.dtype)
    return one_row_at_a_time(x, dense, dtype or x.dtype)
