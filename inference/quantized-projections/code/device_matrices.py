"""Stored matrices on a device and their projection through cuBLASLt (torch's scaled matmul): the projections slot's
device form (crit_plugins.device).

Formats, by the tensors a checkpoint stores (roles, not names):

    bf16         weight [out][in] in BF16/FP16/FP32; y = x W^T in BF16
    fp8:channel  weight [out][in] E4M3, scale [out] (W = weight * scale per output row)
    fp8:tensor   weight [out][in] E4M3, scale [] (one for the matrix)
    mxfp8        weight [out][in] E4M3, scale [out][in / 32] E8M0 bytes (W = weight * 2^(scale - 127), MXFP8 as
                 modelopt stores it)
    nvfp4        weight [out][in / 2] packed E2M1 (code 2i in the low nibble of byte i), scale [out][in / 16] E4M3
                 block scales, global_scale [] or [out] (W = e2m1 * scale / global_scale, nvfp4.py's divisor
                 convention; one per output row for matrices stacked by rows) and input_scale [] (the activations'
                 global scale, the same convention; W4A4 only)

Decode rows (calls of up to GEMV_ROWS rows: decode steps, verify windows) keep each row's bits whatever rows share
the call. With `decode` prompt (the default) they take the prompt rows' activations, so a token decoded and the same
token in a prompt (a replay, a prompt cache) differ only in the order of their FP32 sums: FP8 matrices quantize each
activation row to E4M3 (fp8_rows) and W4A4 NVFP4 ones to NVFP4 (nvfp4_rows), and crit-labs' GEMV multiplies those
quantized values (exact in BF16) by the stored weights, the activation scales applied after; without the kernel,
cuBLASLt multiplies one row a call (it picks its kernel, and so a row's sums, by the row count). With `decode` bf16
they multiply the 16-bit activations themselves (weight only: W8A16, W4A16), closer to the unquantized model and further
from the prompt rows. MXFP8 and weight-only NVFP4 prompt rows take 16-bit activations, so their decode rows do too.

Prompt rows of FP8 matrices take each activation row in E4M3 with its own scale (its absolute maximum over 448), so a
row's codes never depend on the other rows, multiply with FP32 accumulation and scale each output by its row's and its
column's scales. MXFP8 prompt rows multiply 16-bit activations by the matrix's exact values decoded to BF16 (an E4M3
value times a power of two). NVFP4 matrices multiply prompt rows in one of two ways, as `activations` asks when they
load:

- nvfp4 (W4A4, as such checkpoints declare their activations): each block of 16 values of a row scaled to E2M1 by an
  E4M3 block scale under the input_scale, multiplied with the weight's codes by the block-scaled cuBLASLt kernel
  (scales in its 128 x 4 tiled layout, to_blocked) with FP32 accumulation, then divided by both global scales.
- bf16 (W4A16, weight only: the activations stay 16-bit, for preserve-mode quality): the matrix decoded to BF16 in a
  shared workspace and torch's GEMM, FP32 accumulation; with `gemm` fixed_tiles, a GEMM with fixed tiles instead of
  torch's, whose rows do not depend on the rows beside them (prompts of several sequences in one GEMM match GEMMs of
  their own).

With `gemm` pinned, every format's prompt GEMMs (W4A4, W8A8, BF16, and the decoded chunks of weight-only matrices) run
cuBLASLt with one split-K-free algorithm per GEMM shape and format, pinned at load and run for every row count
(crit-labs/pinned-gemm, _pin): a prompt row's bits are the same at every prompt length, in any chunk and in a padded
bucket, which torch's choice of algorithm by the row count does not keep.

crit hosts no kernel: the FP8 and NVFP4 x NVFP4 products are torch's scaled matmul (cuBLASLt) or crit-labs/pinned-gemm's
pinned cuBLASLt calls, the activation quantizers torch operations or crit-labs/mma-rows', and the W4A16 GEMV, decoders
and fixed-tile GEMM crit-labs' registry kernels with torch fallbacks (device_nvfp4.py).

Draft-only re-encoding (encode): a matrix only a drafter reads (an MTP layer, a draft head) may be re-encoded at load
from its stored bf16 or fp8 values to nvfp4 (E4M3 scale per 16 values, one FP32 global scale, nvfp4.py's rule) or fp8
(E4M3 codes, one FP32 scale per row), which cuts the bytes a draft step reads; the encoders are torch operations in
FP64 and give nvfp4.py's codes and scales exactly (reencode). The stored target weights still verify every draft.

The constant tables the activation quantizer reads live on the device once (no host copy per call), so a step that
multiplies through this module can be captured in a CUDA graph.
"""
import math

from . import nvfp4 as codec

# The folder of the plugin whose plugin.json declares the registry kernels device_nvfp4 loads (crit-labs/decode-gemv,
# crit-labs/fixed-tile-gemm) with their pins: quantized-projections' plugin.py sets it.
KERNEL_FOLDER = None

FORMATS = ("bf16", "fp8:channel", "fp8:tensor", "mxfp8", "nvfp4")
DECODES = ("prompt", "bf16")   # decode rows' activations: the prompt rows' (quantized like them), or 16-bit
GEMV_ROWS = 16    # the most rows a decode call takes; each row's bits are those of the row alone
TRITON_ROWS = 8   # the Triton NVFP4 GEMV's most rows
E4M3_MAX, E2M1_MAX = 448.0, 6.0
# E2M1 magnitudes of codes 0..7, the boundaries between them and those whose tie rounds up (to the even code).
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
TIES_UP = (0.75, 1.75, 3.5)


ACTIVATIONS = ("nvfp4", "bf16")
FP8_ACTIVATIONS = ("fp8", "bf16")   # FP8 matrices' prompt rows: W8A8, or W8A16 (the E4M3 weights exact in BF16)


class Matrix:
    def __init__(self, format, rows, columns, tensors, activations=None):
        self.format, self.rows, self.columns, self.tensors = format, rows, columns, tensors
        self.activations = activations


_CONSTANTS = {}


def device_key(device):
    """One name per device (cuda and cuda:0 alike), so a table made at load is the one a captured call finds."""
    import torch
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return str(device)


def constants(device):
    """The quantizer's tables on `device`, made once (at load): E2M1 bounds, the ties that round up, the decode table."""
    import torch
    key = device_key(device)
    if key not in _CONSTANTS:
        _CONSTANTS[key] = {"bounds": torch.tensor(BOUNDS, dtype=torch.float32, device=device),
                           "ties": torch.tensor(TIES_UP, dtype=torch.float32, device=device),
                           "table": torch.tensor([*E2M1, *(-v for v in E2M1)], dtype=torch.float32, device=device)}
    return _CONSTANTS[key]


def _scalar(tensor, what):
    values = tensor.float().reshape(-1)
    if values.numel() != 1:
        raise ValueError(f"an nvfp4 matrix has one {what}, not {values.numel()}")
    return float(values[0])


PROMPTS = ("bf16", "fp16_acc", "fp8")
GEMMS = ("provider", "fixed_tiles", "pinned", "tile")
GEMVS = ("auto", "mma", "native", "triton")
TILE_ROWS = 64       # crit-labs/mma-rows' rows a call: longer tile calls go 64 rows at a time
PROMPT_TILES = ("auto", 0, 64, 128, 192, 256)   # tile_rows: the most prompt rows a call sends through the tile
# tile_rows "auto": by a matrix's prompt arithmetic, the prompt rows the tile takes before its prompt path is the
# faster (measured on an RTX 5090 at 16 to 256 rows over 23 matrices of three models, 640 to 248320 rows: the tile
# reads the weights once, but its MMAs and decodes grow with the rows): weight-only prompts, whose prompt path decodes
# the matrix for a GEMM, up to 64 (the tile won there on every matrix, and at 96 on all but one); W4A4, W8A8 and BF16,
# whose prompt paths are cuBLASLt GEMMs on the stored or quantized values, none past the decode rows.
AUTO_TILES = {"weight_only": 64, "w4a4": 0, "w8a8": 0, "bf16": 0}

# Buffers every matrix on a device shares, made by the plugin's factory and declared by its memory(): the decoded
# rows of a prompt GEMM (a chunk of matrix rows at a time), mma-rows' quantized activations, its partial sums and its
# row tiles' counters. Calls run in stream order (a projection at a time), as a model's forward makes them.
WORKSPACE_BYTES = 64 << 20
SCRATCH_BYTES = (TILE_ROWS * 32768 * 2 + 15) // 16 * 16 + TILE_ROWS * 4   # tile activations of up to 32768 columns
PARTIAL_BYTES = 64 << 20   # a 64-row tile call's partial sums for the splits plan() gives up to N of ~65536
COUNTERS = 1 << 14
GEMM_BYTES = 32 << 20      # the pinned prompt GEMMs' cuBLASLt workspace
BUFFER_BYTES = WORKSPACE_BYTES + SCRATCH_BYTES + PARTIAL_BYTES + 4 * COUNTERS + GEMM_BYTES


def load(tensors, format, device=None, activations="nvfp4", prompt="bf16", gemm="provider", gemv="auto",
         decode="prompt", decode_rows=GEMV_ROWS, tile_rows="auto", fp8_activations="fp8"):
    """A Matrix on `device` (the weight's own unless given) from a checkpoint's tensors by role. Its decode rows (calls
    of up to its decode_rows: `decode_rows`, 8 to GEMV_ROWS, or fewer where its GEMV takes fewer) take `decode` (DECODES: prompt, the prompt rows' activations; bf16, 16-bit ones) through `gemv`
    (GEMVS: auto, by the matrix's arithmetic: FP8 16-bit decode rows on crit-labs/decode-gemv's dot GEMV, W4A4 ones on
    mma-rows' block-scaled FP4 MMA, where those are here, every other matrix as mma; mma, crit-labs/mma-rows'
    tensor-core tile where that kernel is here and takes the matrix (K a multiple of 32), else as native; native,
    crit-labs/decode-gemv's CUDA GEMVs where that kernel is here, its Triton GEMV for
    weight-only NVFP4 widths that are no multiple of 32; triton, the Triton NVFP4 GEMV; where no kernel is here,
    cuBLASLt one row a call for quantized activations and the torch fallbacks for 16-bit ones); matrix.path names it.
    On the tile, prompt calls of up to `tile_rows` rows (PROMPT_TILES) go through it too, 64 rows a call, where the
    prompt arithmetic is the tile's (all but weight-only NVFP4 prompts in fp16_acc or fp8, and fixed-tile or pinned
    prompt GEMMs, which keep every prompt row's bits their own way); a row then gets the same bits in any call of up
    to matrix.invariant_rows rows. An nvfp4 matrix's prompt rows multiply `activations`
    (ACTIVATIONS: nvfp4 for W4A4, bf16 for weight only), a weight-only one's in the `prompt` arithmetic (PROMPTS: the
    matrix decoded to BF16 with FP32 accumulation, to FP16 with FP16 accumulation, or to FP8 rows against FP8
    activation rows); bf16, mxfp8 and weight-only bf16-prompt nvfp4 ones through `gemm` (GEMMS: torch's, or the
    fixed-tile GEMM); with `gemm` pinned, every format's prompt GEMMs through pinned cuBLASLt algorithms (_pin;
    weight-only NVFP4 prompts in bf16 only); with `gemm` tile (gemv mma; weight-only NVFP4 prompts in bf16 only),
    every prompt row through mma-rows' prompt tile with the decode rows' activations and plan, so each output has the
    bits that row has as a decode row (matrix.gemm stays tile where the tile takes the matrix and the kernel has the
    prompt tile, else provider: _finish)."""
    import torch
    from . import device_nvfp4
    if format not in FORMATS:
        raise ValueError(f"a matrix is {', '.join(FORMATS)}, not {format!r}")
    if activations not in ACTIVATIONS:
        raise ValueError(f"nvfp4 activations are {', '.join(ACTIVATIONS)}, not {activations!r}")
    if prompt not in PROMPTS:
        raise ValueError(f"weight-only prompt GEMMs run {', '.join(PROMPTS)}, not {prompt!r}")
    if gemm not in GEMMS or (gemm == "fixed_tiles" and prompt != "bf16" and format not in ("bf16", "mxfp8") and
                             not (format.startswith("fp8") and fp8_activations == "bf16")):
        raise ValueError(f"weight-only prompt GEMMs go through {', '.join(GEMMS)} (fixed_tiles with bf16 prompts), "
                         f"not {gemm!r} with {prompt!r}")
    if gemm in ("pinned", "tile") and format == "nvfp4" and activations == "bf16" and prompt != "bf16":
        raise ValueError(f"{gemm} prompt GEMMs multiply weight-only NVFP4 prompts in bf16, not {prompt!r}")
    if gemm == "tile" and gemv != "mma":
        raise ValueError(f"tile prompt GEMMs give prompt rows the decode tile's bits: decode rows on gemv mma, not "
                         f"{gemv!r}")
    if decode not in DECODES:
        raise ValueError(f"decode rows take {', '.join(DECODES)} activations, not {decode!r}")
    if not 8 <= decode_rows <= GEMV_ROWS:
        raise ValueError(f"a decode call takes 8 to {GEMV_ROWS} rows, not {decode_rows}")
    if gemv not in GEMVS:
        raise ValueError(f"decode rows go through {', '.join(GEMVS)}, not {gemv!r}")
    if tile_rows not in PROMPT_TILES:
        raise ValueError(f"tile_rows is one of {PROMPT_TILES}, not {tile_rows}")
    if fp8_activations not in FP8_ACTIVATIONS:
        raise ValueError(f"fp8 activations are {', '.join(FP8_ACTIVATIONS)}, not {fp8_activations!r}")
    tile = gemv if gemv in ("auto", "mma") else None   # mma-rows (auto: the dot GEMV and the FP4 MMA where they are)
    gemv = "native" if tile else gemv
    weight = tensors["weight"]
    device = torch.device(device) if device is not None else weight.device
    device_nvfp4.prepare(device)   # the fallbacks' tables, before any call a graph may capture
    # The registry modules this matrix's calls use, held from load (a lookup a call, or a module imported on first use
    # inside a capture, is what this avoids).
    cuda = device.type == "cuda"
    modules = {"rows": device_nvfp4.rows_kernel() if cuda else None,
               "tiles": device_nvfp4.tiles_kernel() if cuda and gemm == "fixed_tiles" else None,
               "quantize": device_nvfp4.mma(device) if cuda else None,
               "pinned": device_nvfp4.pinned(device) if cuda and gemm == "pinned" else None}
    if cuda:
        buffers(device)
    if format == "bf16":
        w = weight.to(device=device, dtype=torch.bfloat16).contiguous()
        matrix = Matrix(format, w.shape[0], w.shape[1], {"weight": w})
        matrix.gemm, matrix.decode, matrix.decode_rows = gemm, "bf16", GEMV_ROWS
        matrix.native = device_nvfp4.native(device) if gemv == "native" and w.shape[1] % 8 == 0 else None
        matrix.path = "gemv" if matrix.native is not None else "torch"
        return _finish(matrix, decode_rows, device, tile, tile_rows, modules)
    if format == "mxfp8":
        w = weight.to(device).view(torch.float8_e4m3fn).contiguous()
        exponents = tensors["scale"].to(device).view(torch.uint8).contiguous()
        if w.shape[1] % 32 or tuple(exponents.shape) != (w.shape[0], w.shape[1] // 32):
            raise ValueError(f"an mxfp8 matrix of {w.shape[0]} x {w.shape[1]} has {w.shape[0]} x {w.shape[1] // 32} "
                             f"E8M0 scales (and a multiple of 32 columns), not {tuple(exponents.shape)}")
        matrix = Matrix(format, w.shape[0], w.shape[1], {"weight": w, "scale": exponents})
        matrix.gemm, matrix.decode, matrix.decode_rows = gemm, "bf16", GEMV_ROWS
        matrix.native = device_nvfp4.native(device)
        matrix.path = "gemv" if matrix.native is not None else "torch"
        return _finish(matrix, decode_rows, device, tile, tile_rows, modules)
    if format.startswith("fp8"):
        w = weight.to(device).view(torch.float8_e4m3fn).contiguous()
        scale = tensors["scale"].to(device=device, dtype=torch.float32).reshape(-1)
        if format == "fp8:channel" and scale.numel() != w.shape[0]:
            raise ValueError(f"an fp8:channel matrix has one scale per output row ({w.shape[0]}), not {scale.numel()}")
        if format == "fp8:tensor" and scale.numel() != 1:
            raise ValueError("an fp8:tensor matrix has one scale")
        if w.shape[0] % 16 or w.shape[1] % 16:
            raise ValueError(f"an fp8 matrix multiplies through cuBLASLt, which takes rows and columns in multiples of "
                             f"16, not {w.shape[0]} x {w.shape[1]}")
        matrix = Matrix(format, w.shape[0], w.shape[1], {"weight": w, "scale": scale.expand(w.shape[0]).contiguous()},
                        fp8_activations)
        matrix.decode, matrix.decode_rows = decode, GEMV_ROWS
        matrix.gemm = gemm if fp8_activations == "bf16" or gemm in ("pinned", "tile") else "provider"
        matrix.native = device_nvfp4.native(device)
        quantized = decode == "prompt" and fp8_activations == "fp8"
        matrix.path = "gemv" if matrix.native is not None else "cublaslt" if quantized else "torch"
        if tile == "auto" and not quantized and cuda and _dot_plan(matrix, device):
            tile = None   # one kernel for every decode row; prompt rows take the prompt path
        return _finish(matrix, decode_rows, device, tile, tile_rows, modules)
    codes = weight.to(device).view(torch.uint8).contiguous()
    rows, columns = codes.shape[0], codes.shape[1] * 2
    scales = tensors["scale"].to(device).view(torch.float8_e4m3fn)
    if columns % 16 or tuple(scales.shape) != (rows, columns // 16):
        raise ValueError(f"an nvfp4 matrix of {rows} x {columns} has {rows} x {columns // 16} block scales")
    g = tensors["global_scale"].to(device=device, dtype=torch.float32).reshape(-1)
    if g.numel() not in (1, rows):
        raise ValueError(f"an nvfp4 matrix of {rows} rows has one global scale or one per row, not {g.numel()}")
    constants(device)
    inverse = (1.0 / g).expand(rows).contiguous()
    if activations == "bf16":
        found = {"codes": codes, "scale": scales.contiguous(), "inverse": inverse,
                 "ones": torch.ones(rows, dtype=torch.float32, device=device)}
        if prompt == "fp8":
            found["row_scale"] = device_nvfp4.row_scales(found["scale"], inverse)
            found["fp8_factor"] = (inverse / found["row_scale"]).contiguous()
        matrix = Matrix(format, rows, columns, found, activations)
        matrix.prompt, matrix.gemm, matrix.decode, matrix.decode_rows = prompt, gemm, "bf16", GEMV_ROWS
        matrix.native = device_nvfp4.native(device) if gemv == "native" and columns % 32 == 0 else None
        matrix.triton = device_nvfp4.triton(device) if matrix.native is None else None
        matrix.path = "gemv" if matrix.native is not None else "triton" if matrix.triton is not None else "torch"
        if matrix.triton is not None:
            matrix.decode_rows = TRITON_ROWS
        return _finish(matrix, decode_rows, device, tile, tile_rows, modules)
    if "input_scale" not in tensors:
        raise ValueError("an nvfp4 matrix with nvfp4 activations needs the activations' input_scale")
    # W4A4: the codes as cuBLASLt reads them, the block scales once in its tiled layout (the decode GEMV reads that
    # layout too), the inverse global scales [1][rows] for the prompt rows' epilogue and [rows] for the GEMV.
    found = {"weight": codes.view(torch.float4_e2m1fn_x2), "codes": codes, "scale": to_blocked(scales),
             "inverse": inverse[None, :], "inverse_rows": inverse,
             "input_scale": _scalar(tensors["input_scale"], "input scale")}
    found["input_rcp"] = torch.tensor(1.0 / found["input_scale"], dtype=torch.float32, device=device)
    matrix = Matrix(format, rows, columns, found, activations)
    matrix.runs = _runs(found, rows, columns, device)
    matrix.gemm = gemm if gemm in ("pinned", "tile") else "provider"
    matrix.decode, matrix.decode_rows, matrix.triton = decode, GEMV_ROWS, None
    matrix.native = device_nvfp4.native(device) if gemv == "native" and columns % 32 == 0 else None
    matrix.path = "gemv" if matrix.native is not None else "cublaslt" if decode == "prompt" else "torch"
    return _finish(matrix, decode_rows, device, tile, tile_rows, modules)


def _runs(found, rows, columns, device):
    """A W4A4 matrix's runs of rows sharing one global scale (a stack of matrices: q/k/v, gate/up), each
    (first, end, its inverse global scale as a device scalar), for block-scaled GEMMs with tensor-wide factors; their
    block scales re-tiled per run (found["run_scales"][first]) where a run does not start on a 128-row tile. None
    for more than 8 runs or rows past a run that are no multiple of 16 (one GEMM, the scales after it)."""
    import torch
    inverse = found["inverse_rows"]
    edges = [0] + (torch.nonzero(inverse[1:] != inverse[:-1]).flatten() + 1).tolist() + [rows]
    if len(edges) > 9 or any((b - a) % 16 for a, b in zip(edges, edges[1:])):
        return None
    runs, found["run_scales"] = [], {}
    tile = math.ceil(columns / 64) * 512   # a 128-row tile's bytes of the tiled scales
    for a, b in zip(edges, edges[1:]):
        if a % 128 == 0:   # the run's tiles are a range of the matrix's (the last one's padding rows unread)
            found["run_scales"][a] = found["scale"][a // 128 * tile:math.ceil(b / 128) * tile]
        else:
            found["run_scales"][a] = to_blocked(from_blocked(found["scale"], rows, columns // 16)[a:b].contiguous())
        runs.append((a, b, inverse[a:a + 1].reshape(()).clone()))
        found.setdefault("run_weights", {})[a] = found["weight"][a:b]
        found.setdefault("run_alpha", {})[a] = (inverse[a:a + 1] * found["input_rcp"]).reshape(1).contiguous()
    return runs


def _finish(matrix, rows, device, tile, tile_rows, modules):
    """The matrix with its decode rows capped at `rows` (the decode_rows option: calls of more rows are prompt rows),
    on crit-labs/mma-rows' tile where `tile` (auto, mma) and the kernel take it (_tile_plan), and its invariant_rows:
    the most rows a call takes with each row's bits those of the row alone; its Triton kernels compiled
    (_warm_triton)."""
    matrix.decode_rows = min(matrix.decode_rows, rows)
    matrix.mma, matrix.tile_rows, matrix.modules = None, 0, modules
    if getattr(matrix, "path", None) == "dot":
        matrix.decode_rows = min(matrix.decode_rows, matrix.dot.DOT_ROWS)
    if tile:
        _tile_plan(matrix, device, tile_rows, tile == "auto")
    if getattr(matrix, "gemm", "provider") == "tile" and (
            matrix.mma is None or getattr(matrix.mma, "mm_prompt", None) is None):
        matrix.gemm = "provider"   # no decode tile for this matrix, or a kernel without the prompt tile: not exact
    matrix.pinned = _pin(matrix, device) if getattr(matrix, "gemm", "provider") == "pinned" else None
    same = matrix.mma is not None and matrix.mma_acts["decode"] == matrix.mma_acts["prompt"]
    matrix.invariant_rows = max(matrix.decode_rows, matrix.tile_rows) if same else matrix.decode_rows
    if device.type == "cuda":
        _warm_triton(matrix)
        _warm_paths(matrix)
    return matrix


# The Triton launches warmed in this process, by what selects their compiled specialization.
_WARMED = set()


def _warm_triton(m):
    """The Triton kernels this matrix's calls can reach, each run once (here, at load) per specialization class the
    calls will meet, so no request JIT-compiles one (and a captured chunk or round can hold them): the fixed-tile
    prompt GEMM over the widths it multiplies (the matrix's chunks, or the matrix), for each tile it picks and row
    counts divisible by 16 or not, into FP32 and the activations' dtype; the Triton GEMV at each row count it takes
    (its tile autotuned per shape). The prompt decoders are warmed with the weight-only matrices (_warm_decoders);
    mma-rows and decode-gemv's CUDA GEMVs are compiled ahead of time, cuBLASLt picks its kernels without compiling."""
    import torch
    from . import device_nvfp4
    if getattr(m, "gemm", "provider") == "fixed_tiles" and m.modules["tiles"] is not None:
        dtype = torch.bfloat16   # fixed-tile prompt GEMMs take BF16 rows
        weight = m.tensors["weight"] if m.format == "bf16" else None
        device = weight.device if weight is not None else next(t.device for t in m.tensors.values()
                                                                if isinstance(t, torch.Tensor))
        widths = {m.rows} if weight is not None else {n1 - n0 for n0, n1 in row_chunks(m)}
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        space = buffers(device)["workspace"].view(dtype)
        for width in sorted(widths):
            columns = -(-width // 128)
            need = -(-(2 * sms) // columns)   # 128-row tiles the large tile needs at this width (fixed-tile-gemm's rule)
            first = 128 * (need - 1) + 1      # the fewest rows the large tile takes
            for rows in sorted({17, 32, max(17, first), max(32, first + 15)}):
                big = -(-rows // 128) * columns >= 2 * sms
                for out in (torch.float32, dtype):
                    key = ("mm", width, m.columns, rows % 16 == 0, big, out, dtype)
                    if key in _WARMED:
                        continue
                    dense = weight if weight is not None else space[:width * m.columns].view(width, m.columns)
                    x = torch.zeros(rows, m.columns, dtype=dtype, device=dense.device)
                    device_nvfp4.mm(x, dense, out, module=m.modules["tiles"])
                    _WARMED.add(key)
    if getattr(m, "triton", None) is not None:
        t = m.tensors
        for rows in range(1, TRITON_ROWS + 1):
            for out in (torch.float32, torch.bfloat16):
                key = ("gemv", m.rows, m.columns, rows, out)
                if key in _WARMED:
                    continue
                x = torch.zeros(rows, m.columns, dtype=torch.bfloat16, device=t["codes"].device)
                device_nvfp4.gemv(x, t["codes"], t["scale"], t["inverse"], dtype=out, native=None, triton=m.triton)
                _WARMED.add(key)


# Row counts the warm-up runs a matrix's prompt route at past its tile: GEMM providers pick kernels by the row count
# (and load each on its first use), so a ladder of counts up to a prompt chunk's size meets them at load.
WARM_ROWS = (65, 100, 128, 129, 200, 256, 384, 512, 768, 1024, 1536, 2048)


def _warm_paths(m):
    """Every call path of the matrix run once at load on zero activations: decode rows at every count it takes (the
    tile's or the GEMVs' instances), the tile's prompt counts, and the prompt route over WARM_ROWS, into its default
    output and FP32, so a request meets no first use (a Triton compile, cuBLAS/cuBLASLt picking and loading a kernel
    for a shape, a CUDA module loaded on its first launch). Matrices of the same shape and arithmetic warm once."""
    import torch
    key = (m.format, m.activations, getattr(m, "decode", None), getattr(m, "prompt", None), getattr(m, "gemm", None),
           m.rows, m.columns, m.decode_rows, m.tile_rows, getattr(m, "path", None), getattr(m, "mma_plan", None),
           tuple((a, b) for a, b, _ in (getattr(m, "runs", None) or ())))
    if ("paths",) + key in _WARMED:
        return
    device = next(t.device for t in m.tensors.values() if isinstance(t, torch.Tensor))
    counts = sorted(set(range(1, m.decode_rows + 1)) | {r for r in (17, 32, 33, 48, 64) if r <= m.tile_rows}
                    | {r for r in WARM_ROWS if r > max(m.decode_rows, m.tile_rows)})
    x = torch.zeros(counts[-1], m.columns, dtype=torch.bfloat16, device=device)
    for rows in counts:
        for out in (None, torch.float32):
            matmul(x[:rows], m, out)
    _WARMED.add(("paths",) + key)


def _tile_plan(matrix, device, tile_rows, fp4):
    """Decode rows (and prompt rows up to tile_rows where the prompt arithmetic is the tile's) through mma-rows: every
    value exact in BF16 and FP32 sums, the prompt rows' quantized activations (decode prompt) or 16-bit ones, a sum's
    order fixed by the matrix and the device (mma-rows' plan, its K splits bounded by the shared partial buffer). With
    `fp4` (gemv auto), W4A4 decode rows (decode prompt) go on the block-scaled FP4 MMA where the device has it
    (mma-rows' nvfp4_mma: the activations' and the weights' E2M1 codes with their block scales, FP32 sums; K a
    multiple of 128), one plan for every decode row count, and the tile then takes no prompt rows."""
    import torch
    from . import device_nvfp4
    module = matrix.modules["quantize"]
    if module is None or not module.eligible(matrix.rows, matrix.columns):
        return
    weights = matrix.format if matrix.format in ("bf16", "mxfp8") else "fp8" if matrix.format.startswith("fp8") \
        else "nvfp4"
    quantized = (weights == "fp8" and matrix.activations == "fp8") or (weights == "nvfp4" and matrix.activations == "nvfp4")
    prompt = ("fp8_rows" if weights == "fp8" else "nvfp4") if quantized else "bf16"
    decode = prompt if matrix.decode == "prompt" else "bf16"
    fp4 = fp4 and decode == "nvfp4" and matrix.columns % getattr(module, "FP4_STEP", 1 << 30) == 0 and \
        module.fp4_mma(device)
    if fp4:
        decode = "nvfp4_mma"
    if "bf16" != prompt and max(module.scratch_bytes(TILE_ROWS, matrix.columns),
                                module.scratch_bytes(TILE_ROWS, matrix.columns, "nvfp4_mma") if fp4 else 0) \
            > SCRATCH_BYTES:
        return
    groups, warps, splits = module.plan(matrix.rows, matrix.columns, weights, device,
                                        acts="nvfp4_mma" if fp4 else "bf16")
    while splits > 1 and module.partial_elements(TILE_ROWS, matrix.rows, splits) * 4 > PARTIAL_BYTES:
        splits -= 1
    if module.row_tiles(matrix.rows, 1, warps) > COUNTERS:
        return
    matrix.mma, matrix.mma_plan, matrix.mma_weights = module, (groups, warps, splits), weights
    matrix.mma_acts = {"decode": decode, "prompt": prompt}
    weight_only_other = weights == "nvfp4" and not quantized and getattr(matrix, "prompt", "bf16") != "bf16"
    fixed = getattr(matrix, "gemm", "provider") in ("fixed_tiles", "pinned", "tile")
    if tile_rows == "auto":
        tile_rows = AUTO_TILES["bf16" if weights == "bf16" else "w8a8" if prompt == "fp8_rows" else
                               "w4a4" if prompt == "nvfp4" else "weight_only"]
    matrix.tile_rows = 0 if weight_only_other or fixed or fp4 else tile_rows
    matrix.path = "mma_fp4" if fp4 else "mma"


def _dot_plan(m, device):
    """An FP8 matrix's 16-bit decode rows on crit-labs/decode-gemv's dot GEMV (a Triton program of 16 rows by a block
    of the matrix on tensor cores, the E4M3 values exact in BF16, FP32 sums, a fixed split of K added in order, the row
    scale after): its program shape from the matrix's shape and the device (fp8_dot_tile), its parts of K cut to what
    the shared partial buffer holds at 16 rows. Faster than the tile from 8 rows on at the measured widths and level
    below; False where the kernel is not here or K is no multiple of 128."""
    import torch
    from . import device_nvfp4
    module = device_nvfp4.dot(device)
    if module is None or m.columns % 128:
        return False
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    bn, bk, warps, split = module.fp8_dot_tile(m.rows, m.columns, sms)
    while split > 1 and module.dot_partial_elements(GEMV_ROWS, m.rows, split) * 4 > PARTIAL_BYTES:
        split //= 2
    m.dot, m.dot_plan, m.path = module, (bn, bk, warps, split), "dot"
    return True


_PINNED = {}


def _pin(m, device):
    """The matrix's prompt GEMMs, each pinned to one cuBLASLt algorithm for every row count (crit-labs/pinned-gemm,
    the matrix's module; None where it is not here, and the prompt rows go one a call through the provider's GEMM):
    by output type, or by chunk width for a matrix decoded a chunk of rows at a time, or by run and output type for
    W4A4 runs. A shape and format is pinned once a device (the choice depends on them, the device and the cuBLASLt
    version only), the checks multiplying this matrix's stored weight where the GEMM reads it as stored."""
    module = m.modules["pinned"]
    if module is None:
        return None
    space, t = buffers(device)["gemm"], m.tensors

    def get(fmt, n, weight=None):
        key = (device_key(device), fmt, n, m.columns)
        if key not in _PINNED:
            _PINNED[key] = module.pin(fmt, n, m.columns, device, workspace=GEMM_BYTES, buffer=space, weight=weight)
        return _PINNED[key]

    if m.format == "bf16":
        return {torch_dtype: get(fmt, m.rows, (t["weight"], None))
                for torch_dtype, fmt in ((None, "bf16:bf16"), ("fp32", "bf16:fp32"))}
    if m.format == "mxfp8" or (m.format.startswith("fp8") and m.activations == "bf16") or m.activations == "bf16":
        return {n1 - n0: get("bf16:fp32", n1 - n0) for n0, n1 in row_chunks(m)}
    if m.format.startswith("fp8"):   # cuBLASLt lists no split-K-free algorithm with row scales: they go after
        return {"fp32": get("fp8e4m3:fp32", m.rows, (t["weight"], None))}
    if m.runs is None:
        return {"whole": get("nvfp4:fp32", m.rows, (t["weight"], t["scale"]))}
    return {(n0, out): get("nvfp4:" + out, n1 - n0, (t["run_weights"][n0], t["run_scales"][n0]))
            for n0, n1, _ in m.runs for out in ("bf16", "fp32")}


_BUFFERS = {}


def buffers(device):
    """The buffers every matrix on `device` shares (BUFFER_BYTES), made once: by the plugin's factory, which its
    memory() declares, or by a first load on a device the factory did not make them for."""
    import torch
    key = device_key(device)
    if key not in _BUFFERS:
        _BUFFERS[key] = {"workspace": torch.empty(WORKSPACE_BYTES // 2, dtype=torch.bfloat16, device=device),
                         "scratch": torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device=device),
                         "partial": torch.empty(PARTIAL_BYTES // 4, dtype=torch.float32, device=device),
                         "counters": torch.zeros(COUNTERS, dtype=torch.int32, device=device),
                         "gemm": torch.empty(GEMM_BYTES, dtype=torch.uint8, device=device)}
    return _BUFFERS[key]


def row_chunks(m):
    """The matrix-row ranges a prompt GEMM decodes one at a time into the workspace: multiples of 128 rows of
    WORKSPACE_BYTES at most (one range where the whole matrix fits), fixed by the matrix's shape."""
    step = max(128, WORKSPACE_BYTES // (2 * m.columns) // 128 * 128)
    return [(n, min(m.rows, n + step)) for n in range(0, m.rows, step)]


def _decoded(x, m, decode, gemm, dtype=None, finish=None, out_dtype=None, default=None):
    """The prompt rows' product with the matrix decoded chunk by chunk (row_chunks): decode(n0, n1, out) writes rows
    [n0, n1) into the workspace view `out` (of `dtype`), gemm(x, out, n0, n1) multiplies them and finish(y, n0, n1)
    applies the chunk's scales; each chunk's columns are written into the output (out_dtype, else `default`, else
    the GEMM's), one rounding."""
    import torch
    space = buffers(x.device)["workspace"]
    if dtype is not None:
        space = space.view(dtype)
    ranges, final = row_chunks(m), None
    for n0, n1 in ranges:
        dense = space[:(n1 - n0) * m.columns].view(n1 - n0, m.columns)
        decode(n0, n1, dense)
        y = gemm(x, dense, n0, n1)
        if finish is not None:
            y = finish(y, n0, n1)
        if len(ranges) == 1:
            return _out(y, out_dtype, default or y.dtype)
        if final is None:
            final = torch.empty(x.shape[0], m.rows, dtype=out_dtype or default or y.dtype, device=x.device)
        final[:, n0:n1].copy_(y)
    return final


def _tile(x, m, out_dtype, phase):
    """Rows through mma-rows (64 a call), in out_dtype (BF16 unless FP32 or FP16 is asked for), the activations as
    `phase` (decode or prompt) takes them."""
    import torch
    from . import device_nvfp4
    b = buffers(x.device)
    groups, warps, splits = m.mma_plan
    acts = m.mma_acts[phase]
    dtype = out_dtype if out_dtype in (torch.float32, torch.float16) else torch.bfloat16
    out = torch.empty(x.shape[0], m.rows, dtype=dtype, device=x.device)
    x16 = device_nvfp4._aligned(x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16))
    w, scales, row_scale, blocked, g = _tile_operands(m)
    for r0 in range(0, x.shape[0], TILE_ROWS):
        r1 = min(x.shape[0], r0 + TILE_ROWS)
        m.mma.mm_rows(x16[r0:r1], w, out[r0:r1], m.mma_weights, scales, row_scale, acts, g, blocked, groups, warps,
                      splits, b["scratch"], b["partial"], b["counters"])
    return out if out_dtype is None or out.dtype == out_dtype else out.to(out_dtype)


def matmul(x, m, out_dtype=None):
    """y [tokens][rows] = x [tokens][columns] W^T, in BF16 (FP16 for a weight-only matrix's fp16_acc prompt rows,
    the format they accumulate in), or in out_dtype when given (FP32 for logits: every path accumulates in FP32, so
    the sums are not rounded to 16 bits first). Up to the matrix's decode_rows rows are decode rows, more prompt rows;
    up to its invariant_rows rows, each row's bits are those of the row alone."""
    import torch
    if x.shape[-1] != m.columns:
        raise ValueError(f"activations of width {x.shape[-1]} for a matrix of {m.columns} columns")
    if x.shape[0] <= m.decode_rows:
        return _tile(x, m, out_dtype, "decode") if _tiled(x, m, "decode") else _decode_rows(x, m, out_dtype)
    if getattr(m, "gemm", "provider") == "tile":
        if _tiled(x, m, "decode"):
            return _prompt_tile(x, m, out_dtype)
        raise ValueError(f"prompt_gemm tile keeps a prompt row's bits those of the row decoded, on mma-rows' tiles, "
                         f"which take no {x.dtype} activations for this matrix: give it the activations its decode rows "
                         f"take, or choose another prompt_gemm")
    if x.shape[0] <= m.tile_rows and _tiled(x, m, "prompt"):
        return _tile(x, m, out_dtype, "prompt")
    return _prompt_rows(x, m, out_dtype)


def _prompt_tile(x, m, out_dtype):
    """Prompt rows through mma-rows' prompt tile (gemm tile): the decode rows' activations and the matrix's plan, so
    every output has the bits the row has as a decode row; in out_dtype (BF16 unless FP32 or FP16 is asked for).
    Quantized activations go through the shared scratch, as many rows a call as it holds (64 at least)."""
    import torch
    from . import device_nvfp4
    b = buffers(x.device)
    _, _, splits = m.mma_plan
    acts = m.mma_acts["decode"]
    dtype = out_dtype if out_dtype in (torch.float32, torch.float16) else torch.bfloat16
    out = torch.empty(x.shape[0], m.rows, dtype=dtype, device=x.device)
    x16 = device_nvfp4._aligned(x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16))
    w, scales, row_scale, blocked, g = _tile_operands(m)
    step = x.shape[0] if acts == "bf16" else max(TILE_ROWS, SCRATCH_BYTES // (m.columns * 2 + 4) // 16 * 16)
    for r0 in range(0, x.shape[0], step):
        r1 = min(x.shape[0], r0 + step)
        m.mma.mm_prompt(x16[r0:r1], w, out[r0:r1], m.mma_weights, scales, row_scale, acts, g, blocked, splits,
                        b["scratch"])
    return out if out_dtype is None or out.dtype == out_dtype else out.to(out_dtype)


def _tile_operands(m):
    """(w, scales, row_scale, blocked, input_scale) of a matrix on mma-rows' tiles."""
    t = m.tensors
    if m.mma_weights == "bf16":
        return t["weight"], None, None, False, 1.0
    if m.mma_weights == "fp8":
        return t["weight"], None, t["scale"], False, 1.0
    if m.mma_weights == "mxfp8":
        return t["weight"], t["scale"], None, False, 1.0
    if m.activations == "nvfp4":
        return t["codes"], t["scale"], t["inverse_rows"], True, t["input_scale"]
    return t["codes"], t["scale"], t["inverse"], False, 1.0


def _tiled(x, m, phase):
    """Whether the tile takes these activations: BF16 ones, or FP32 ones it multiplies in BF16 (16-bit activations,
    as the other paths take them); quantized activations of another type, and FP16 ones, take the other paths, which
    quantize or multiply them as they are."""
    import torch
    return m.mma is not None and (x.dtype == torch.bfloat16 or (x.dtype == torch.float32 and m.mma_acts[phase] == "bf16"))


def _out(y, out_dtype, dtype=None):
    import torch
    wanted = out_dtype or dtype or torch.bfloat16
    return y if y.dtype == wanted else y.to(wanted)


def _decode_rows(x, m, out_dtype):
    """Decode rows, each row in an order of its own: the GEMVs (on the prompt rows' quantized activations, or on the
    16-bit ones), or without the kernel cuBLASLt one row a call (quantized activations) or the torch fallbacks."""
    import torch
    from . import device_nvfp4
    if m.format == "bf16":
        return _bf16_rows(x.to(torch.bfloat16), m, out_dtype or torch.bfloat16)
    if m.path == "dot":
        x16 = x if x.dtype in (torch.bfloat16, torch.float16) else x.to(torch.bfloat16)
        x16 = x16 if x16.stride(-1) == 1 else x16.contiguous()
        out = torch.empty(x.shape[0], m.rows, dtype=torch.float32 if out_dtype == torch.float32 else torch.bfloat16,
                          device=x.device)
        device_nvfp4._warmed(("dot", m.rows, m.columns, x16.dtype, out.dtype), "the FP8 dot GEMV")
        m.dot.gemv_fp8_dot(x16, m.tensors["weight"], m.tensors["scale"], out, m.dot_plan, buffers(x.device)["partial"])
        return _out(out, out_dtype)
    if m.path == "cublaslt":
        return torch.cat([_prompt_rows(x[r:r + 1], m, out_dtype) for r in range(x.shape[0])])
    dtype = torch.float32 if out_dtype == torch.float32 else torch.bfloat16   # BF16 out unless FP32 is asked for
    t = m.tensors
    quantized = m.decode == "prompt" and m.activations in ("fp8", "nvfp4")
    if not quantized:
        x16 = x if x.dtype in (torch.bfloat16, torch.float16) else x.to(torch.bfloat16)
    if m.format == "nvfp4" and m.activations == "bf16":
        return _out(device_nvfp4.gemv(x16, t["codes"], t["scale"], t["inverse"], dtype=dtype, native=m.native,
                                      triton=m.triton), out_dtype)
    if m.format == "nvfp4":
        # W4A4: the activations as the prompt rows quantize them (e2m1 x block scale, exact in BF16, over the input
        # scale) or 16-bit; the weights' blocked scales as stored.
        rows = nvfp4_values(x, t["input_scale"]) if quantized else x16
        y = device_nvfp4.gemv(rows, t["codes"], t["scale"], t["inverse_rows"], dtype=torch.float32, native=m.native,
                              blocked=True)
        return _out(y / t["input_scale"] if quantized else y, out_dtype)
    out = torch.empty(x.shape[0], m.rows, dtype=torch.float32 if quantized else dtype, device=x.device)
    if quantized:   # FP8: each row in E4M3 by its own scale, as the prompt rows; the row scale after the sum
        q, row_scale = fp8_rows(x)
        device_nvfp4.gemv_fp8(q.to(torch.bfloat16), t["weight"], t["scale"], out, m.native)
        return _out(out * row_scale[:, None], out_dtype)
    device_nvfp4.gemv_fp8(x16, t["weight"], t["scale"], out, m.native)
    return _out(out, out_dtype)


def _pinned_rows(x, m, out_dtype):
    """Prompt rows of a BF16, W8A8 or W4A4 matrix through its pinned GEMMs (_pin): every row's bits the same at any
    row count, in any chunk and in a padded bucket. FP8 rows multiply into FP32, their row scales and the matrix's
    column scales after, rounded once into the output; W4A4 runs take one over both global scales as their epilogue
    factor and write their columns of one output. (Matrices decoded a chunk at a time take theirs in _chunk_gemm.)"""
    import torch
    t, p = m.tensors, m.pinned
    if m.format == "bf16":
        x16 = x.to(torch.bfloat16).contiguous()
        if out_dtype == torch.float32:
            return p["fp32"].mm(x16, t["weight"])
        return _out(p[None].mm(x16, t["weight"]), out_dtype)
    if m.format.startswith("fp8") and m.activations == "fp8":
        q, row_scale = quantized_fp8_rows(x, m.modules["quantize"])
        y = p["fp32"].mm(q, t["weight"]).mul_(row_scale[:, None])
        if out_dtype == torch.float32:
            return y.mul_(t["scale"][None, :])
        return torch.mul(y, t["scale"][None, :], out=torch.empty(y.shape, dtype=out_dtype or torch.bfloat16,
                                                                 device=y.device))
    if m.format == "nvfp4" and m.activations == "nvfp4":
        codes, scales = quantized_nvfp4_rows(x, t["input_scale"], m.modules["quantize"])
        if m.runs is None:
            y = p["whole"].mm(codes, t["weight"], scales, t["scale"])
            return _out(y.mul_(t["inverse"] / t["input_scale"]), out_dtype)
        out = "fp32" if out_dtype == torch.float32 else "bf16"
        y = torch.empty(x.shape[0], m.rows, dtype=torch.float32 if out == "fp32" else torch.bfloat16, device=x.device)
        for n0, n1, _ in m.runs:
            p[n0, out].mm(codes, t["run_weights"][n0], scales, t["run_scales"][n0], t["run_alpha"][n0],
                          out=y[:, n0:n1])
        return _out(y, out_dtype)
    raise AssertionError("decoded matrices multiply their chunks through _decoded")


def _one_row_a_call(x, m, out_dtype):
    """Prompt rows of a pinned matrix where crit-labs/pinned-gemm is not here: one row a call through the provider's
    path, whose GEMM then takes one algorithm for every row (correct, much slower)."""
    import torch
    return torch.cat([_provider_rows(x[r:r + 1], m, out_dtype) for r in range(x.shape[0])])


def _prompt_rows(x, m, out_dtype):
    pinned = getattr(m, "gemm", "provider") == "pinned"
    decoded = m.format == "mxfp8" or m.activations == "bf16"
    if pinned and not decoded:
        return _pinned_rows(x, m, out_dtype) if m.pinned is not None else _one_row_a_call(x, m, out_dtype)
    return _provider_rows(x, m, out_dtype)


def _chunk_gemm(m, out_dtype):
    """gemm(x, w, n0, n1) for a matrix decoded a chunk of rows at a time: the fixed-tile GEMM, the chunk's pinned GEMM
    (or one row a call without crit-labs/pinned-gemm), or torch's; into out_dtype, FP32 for pinned ones."""
    import torch
    from . import device_nvfp4
    gemm = getattr(m, "gemm", "provider")
    if gemm == "fixed_tiles":
        return lambda x, w, n0, n1: device_nvfp4.mm(x, w, out_dtype, module=m.modules["tiles"])
    if gemm == "pinned" and m.pinned is not None:
        return lambda x, w, n0, n1: m.pinned[n1 - n0].mm(x.contiguous(), w)
    if gemm == "pinned":
        return lambda x, w, n0, n1: device_nvfp4.one_row_at_a_time(x, w, torch.float32)
    return lambda x, w, n0, n1: _bf16_gemm(x, w, out_dtype)


def _provider_rows(x, m, out_dtype):
    import torch
    from . import device_nvfp4
    if m.format == "bf16" and getattr(m, "gemm", "provider") == "fixed_tiles":
        return device_nvfp4.mm(x.to(torch.bfloat16), m.tensors["weight"], out_dtype or torch.bfloat16,
                               module=m.modules["tiles"])
    if m.format == "bf16":
        return _bf16_gemm(x.to(torch.bfloat16), m.tensors["weight"], out_dtype)
    if m.format == "mxfp8":
        t, x16 = m.tensors, x.to(torch.bfloat16)
        fixed = getattr(m, "gemm", "provider") == "fixed_tiles"
        return _decoded(x16, m, lambda n0, n1, out: device_nvfp4.decode_mxfp8(t["weight"][n0:n1], t["scale"][n0:n1], out),
                        _chunk_gemm(m, out_dtype or torch.bfloat16 if fixed else out_dtype), out_dtype=out_dtype,
                        default=torch.bfloat16)
    if m.format.startswith("fp8") and m.activations == "bf16":
        # W8A16: the E4M3 codes widened exactly to BF16 a chunk of rows at a time, FP32 sums, the row scale after.
        t, x16 = m.tensors, x.to(torch.bfloat16)
        return _decoded(x16, m, lambda n0, n1, out: out.copy_(t["weight"][n0:n1]), _chunk_gemm(m, torch.float32),
                        finish=lambda y, n0, n1: y.mul_(t["scale"][None, n0:n1]), out_dtype=out_dtype,
                        default=torch.bfloat16)
    if m.format.startswith("fp8"):
        q, row_scale = quantized_fp8_rows(x, m.modules["quantize"])
        if out_dtype in (None, torch.bfloat16):   # the scales in cuBLASLt's epilogue, BF16 out
            return torch._scaled_mm(q, m.tensors["weight"].t(), scale_a=row_scale[:, None].contiguous(),
                                    scale_b=m.tensors["scale"][None, :], out_dtype=torch.bfloat16)
        one = torch.ones((), dtype=torch.float32, device=x.device)
        y = torch._scaled_mm(q, m.tensors["weight"].t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
        return _out(y * row_scale[:, None] * m.tensors["scale"][None, :], out_dtype)
    if m.activations == "bf16":
        t = m.tensors
        prompt = getattr(m, "prompt", "bf16")
        if prompt == "fp8":
            q, row_scale = fp8_rows(x)
            scale_a = row_scale[:, None].contiguous()
            return _decoded(q, m, lambda n0, n1, out: device_nvfp4.decode_fp8(t["codes"][n0:n1], t["scale"][n0:n1],
                                                                             t["fp8_factor"][n0:n1], out,
                                                                             m.modules["rows"]),
                            lambda q, w8, n0, n1: torch._scaled_mm(q, w8.t(), scale_a=scale_a,
                                                                   scale_b=t["row_scale"][n0:n1][None, :],
                                                                   out_dtype=out_dtype or torch.bfloat16),
                            torch.float8_e4m3fn, out_dtype=out_dtype)
        if prompt == "fp16_acc":
            flags = torch.backends.cuda.matmul
            before = flags.allow_fp16_accumulation
            flags.allow_fp16_accumulation = True
            try:
                return _decoded(x.to(torch.float16), m,
                                lambda n0, n1, out: device_nvfp4.decode(t["codes"][n0:n1], t["scale"][n0:n1],
                                                                        t["inverse"][n0:n1], out, m.modules["rows"]),
                                lambda x, w, n0, n1: torch.mm(x, w.t()), torch.float16, out_dtype=out_dtype,
                                default=torch.float16)
            finally:
                flags.allow_fp16_accumulation = before
        # e2m1 x block scale is exact in BF16: the matrix's values decoded without the inverse global scale, which
        # multiplies the FP32 sums after, as the decode GEMV applies it (decode and prompt rows differ in order only).
        return _decoded(x.to(torch.bfloat16), m,
                        lambda n0, n1, out: device_nvfp4.decode(t["codes"][n0:n1], t["scale"][n0:n1], t["ones"][n0:n1],
                                                                out, m.modules["rows"]),
                        _chunk_gemm(m, torch.float32),
                        finish=lambda y, n0, n1: y.mul_(t["inverse"][None, n0:n1]), out_dtype=out_dtype,
                        default=torch.bfloat16)
    t = m.tensors
    codes, scales = quantized_nvfp4_rows(x, t["input_scale"], m.modules["quantize"])
    if out_dtype in (None, torch.bfloat16) and m.runs is not None:
        # Each run of rows sharing a global scale: one block-scaled GEMM whose tensor-wide factors are the activations'
        # and the run's global scales, BF16 out.
        import torch.nn.functional as F
        block, tensor = F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise
        swizzled, plain = F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE
        def run(n0, n1, factor):
            return F.scaled_mm(codes, t["weight"][n0:n1].t(), [scales, t["input_rcp"]], [block, tensor],
                               [t["run_scales"][n0], factor], [block, tensor], swizzle_a=[swizzled, plain],
                               swizzle_b=[swizzled, plain], output_dtype=torch.bfloat16)
        if len(m.runs) == 1:
            return run(*m.runs[0])
        y = torch.empty(x.shape[0], m.rows, dtype=torch.bfloat16, device=x.device)
        for n0, n1, factor in m.runs:   # each run's output into its columns, freed before the next run's GEMM
            y[:, n0:n1].copy_(run(n0, n1, factor))
        return y
    y = torch._scaled_mm(codes, t["weight"].t(), scale_a=scales, scale_b=t["scale"], out_dtype=torch.float32)
    return _out(y * (t["inverse"] / t["input_scale"]), out_dtype)


def _bf16_gemm(x, w, out_dtype):
    """x [M][K] BF16 times w [N][K] BF16 through torch, in BF16 or out_dtype (FP32 sums either way; torch has no CPU
    kernel for a BF16 GEMM into FP32, so on the CPU the BF16 values multiply in FP32)."""
    import torch
    if out_dtype is None or out_dtype == torch.bfloat16:
        return torch.nn.functional.linear(x, w)
    if not x.is_cuda:
        return (x.float() @ w.float().t()).to(out_dtype)
    return torch.mm(x, w.t(), out_dtype=out_dtype)


def _bf16_rows(x, m, dtype):
    """Up to GEMV_ROWS rows times a BF16 matrix, each row's bits the same alone and beside the others: crit-labs/
    decode-gemv's BF16 GEMV where that kernel is here, else one row at a time through torch (whose GEMM picks its
    kernel by the row count)."""
    import torch
    from . import device_nvfp4
    native = getattr(m, "native", None)
    out = torch.empty(x.shape[0], m.rows, dtype=torch.float32 if dtype == torch.float32 else torch.bfloat16,
                      device=x.device)
    device_nvfp4.gemv_bf16(x, m.tensors["weight"], out, native)
    return out if out.dtype == dtype else out.to(dtype)


def dense(m):
    """The matrix's decoded values [rows][columns] in FP32 (what matmul multiplies the activations' codes with)."""
    import torch
    if m.format == "bf16":
        return m.tensors["weight"].float()
    if m.format == "mxfp8":
        return torch.ldexp(m.tensors["weight"].float(), m.tensors["scale"].int().repeat_interleave(32, dim=1) - 127)
    if m.format.startswith("fp8"):
        return m.tensors["weight"].float() * m.tensors["scale"][:, None]
    if m.activations == "bf16":
        codes, scales = m.tensors["codes"], m.tensors["scale"].float()
    else:
        codes = m.tensors["weight"].view(torch.uint8)
        scales = from_blocked(m.tensors["scale"], m.rows, m.columns // 16).float()
    values = decode_e2m1(codes)
    inverse = m.tensors["inverse"].reshape(-1)[:, None]
    return (values.view(m.rows, -1, 16) * scales[..., None]).reshape(m.rows, m.columns) * inverse


def quantized_fp8_rows(x, module):
    """fp8_rows' codes and scales, by crit-labs/mma-rows' quantizer in one kernel (module, the matrix's) where it is
    here and x is BF16 (the same bits), else by fp8_rows."""
    import torch
    from . import device_nvfp4
    module = module if x.dtype == torch.bfloat16 else None
    if module is None:
        return fp8_rows(x)
    x = device_nvfp4._aligned(x)
    q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    module.quantize_fp8_rows(x, q, scale)
    return q, scale


def quantized_nvfp4_rows(x, input_scale, module):
    """nvfp4_rows' codes and tiled block scales, by crit-labs/mma-rows' quantizer in one kernel (module, the matrix's)
    where it is here and x is BF16 (the same bits), else by nvfp4_rows."""
    import torch
    from . import device_nvfp4
    module = module if x.dtype == torch.bfloat16 else None
    if module is None:
        return nvfp4_rows(x, input_scale)
    x = device_nvfp4._aligned(x)
    tokens, columns = x.shape
    codes = torch.empty(tokens, columns // 2, dtype=torch.uint8, device=x.device)
    scales = torch.empty(math.ceil(tokens / 128) * 128 * math.ceil(columns / 64) * 4, dtype=torch.uint8,
                         device=x.device)
    module.quantize_nvfp4(x, input_scale, codes, scales)
    return codes.view(torch.float4_e2m1fn_x2), scales.view(torch.float8_e4m3fn)


def fp8_rows(x):
    """Activation rows in E4M3, each by its own scale (absolute maximum / 448): (codes, scales [tokens])."""
    import torch
    x = x.float()
    scale = (x.abs().amax(dim=-1) / E4M3_MAX).clamp(min=1e-12)
    return (x / scale[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), scale


def nvfp4_values(x, input_scale):
    """Activation rows as nvfp4_rows quantizes them, as values: e2m1(code) * e4m3(block scale) in BF16 (exact: at most 6
    significant bits), the value times input_scale ([tokens][columns])."""
    import torch
    codes, scale = _nvfp4_quantize(x, input_scale)
    table = constants(x.device)["table"]
    tokens, columns = codes.shape
    values = table[codes.long()].view(tokens, -1, 16) * scale.float()[..., None]
    return values.view(tokens, columns).to(torch.bfloat16)


def _nvfp4_quantize(x, input_scale):
    """E2M1 codes [tokens][columns] (uint8) and E4M3 block scales [tokens][columns / 16] of activation rows."""
    import torch
    x = x.float()
    tokens, columns = x.shape
    blocks = x.view(tokens, columns // 16, 16)
    scale = (blocks.abs().amax(dim=-1) / E2M1_MAX * input_scale).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    decoded = scale.float() / input_scale
    scaled = torch.where(decoded[..., None] > 0, blocks / decoded[..., None].clamp(min=1e-30), torch.zeros_like(blocks))
    return encode_e2m1(scaled.reshape(tokens, columns)), scale


def nvfp4_rows(x, input_scale):
    """Activation rows in NVFP4 under `input_scale`: (packed E2M1 codes [tokens][columns / 2], block scales in the
    tiled layout). Each block of 16 values gets the E4M3 scale amax / 6 * input_scale; values divide by its decoded
    value over input_scale and round to the nearest E2M1 magnitude, ties to the even code."""
    import torch
    codes, scale = _nvfp4_quantize(x, input_scale)
    return pack_e2m1(codes).view(torch.float4_e2m1fn_x2), to_blocked(scale)


def encode_e2m1(values):
    """E2M1 codes (uint8 0..15) of values already in E2M1's range: nearest magnitude, ties to the even code."""
    import torch
    magnitude = values.abs().clamp(max=E2M1_MAX)
    table = constants(values.device)
    code = torch.bucketize(magnitude, table["bounds"].to(magnitude.dtype), right=False)
    code = code + torch.isin(magnitude, table["ties"].to(magnitude.dtype)).to(code.dtype)
    return (code | ((values < 0).to(code.dtype) << 3)).to(torch.uint8)


def decode_e2m1(packed):
    """Values of packed E2M1 bytes ([rows][columns / 2] uint8): [rows][columns] FP32."""
    import torch
    table = constants(packed.device)["table"]
    low, high = (packed & 15).long(), (packed >> 4).long()
    return torch.stack((table[low], table[high]), dim=-1).reshape(packed.shape[0], -1)


def pack_e2m1(codes):
    """Codes [rows][columns] into bytes [rows][columns / 2]: code 2i in the low nibble of byte i."""
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def to_blocked(scales):
    """Block scales [rows][blocks] in the 128 x 4 tiled layout block-scaled cuBLASLt kernels read (zero padded)."""
    import torch
    rows, columns = scales.shape
    row_tiles, column_tiles = math.ceil(rows / 128), math.ceil(columns / 4)
    padded = torch.zeros(row_tiles * 128, column_tiles * 4, dtype=scales.dtype, device=scales.device)
    padded[:rows, :columns] = scales
    tiles = padded.view(row_tiles, 128, column_tiles, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def from_blocked(blocked, rows, columns):
    """The inverse of to_blocked: [rows][blocks]."""
    row_tiles, column_tiles = math.ceil(rows / 128), math.ceil(columns / 4)
    tiles = blocked.view(-1, 32, 4, 4).transpose(1, 2).reshape(row_tiles, column_tiles, 128, 4).permute(0, 2, 1, 3)
    return tiles.reshape(row_tiles * 128, column_tiles * 4)[:rows, :columns]


# Draft-only re-encoding: the formats a stored matrix may be re-encoded to, and from which stored formats.
ENCODINGS = {"nvfp4": ("bf16", "fp8:channel", "fp8:tensor"), "fp8": ("bf16",)}
ENCODE_BYTES = 256 << 20   # FP64 values the encoders hold at a time


def stored_bytes(tensors, format):
    """The bytes a matrix's stored tensors take (what a pass multiplying it reads)."""
    names = ("weight", "scale", "global_scale") if format == "nvfp4" else ("weight", "scale")
    return sum(tensors[n].numel() * tensors[n].element_size() for n in names if n in tensors)


_ENCODERS = {}


def _encoder_tables(device):
    """E4M3's 127 finite nonnegative values and E2M1's 8 magnitudes in FP64, ascending, with the midpoints between
    neighbours (exact in FP64), on `device`."""
    import torch
    key = str(device)
    if key not in _ENCODERS:
        e4m3 = torch.tensor([codec.e4m3_value(b) for b in range(0x7F)], dtype=torch.float64, device=device)
        e2m1 = torch.tensor(codec.E2M1, dtype=torch.float64, device=device)
        _ENCODERS[key] = {"e4m3": e4m3, "e4m3_mids": (e4m3[1:] + e4m3[:-1]) / 2,
                          "e2m1": e2m1, "e2m1_mids": (e2m1[1:] + e2m1[:-1]) / 2}
    return _ENCODERS[key]


def nearest(x, table, mids):
    """For x >= 0 (FP64): the index of the nearest entry of `table` (ascending), ties to the even index, values past
    the last entry saturating to it: nvfp4.py's _nearest elementwise. The two neighbours of x are at most a factor of
    2 apart, so their distances to x are exact in FP64 and a tie is exactly x == midpoint."""
    import torch
    index = torch.bucketize(x, mids, right=False)        # midpoints strictly below x
    at = mids[index.clamp(max=mids.numel() - 1)]
    return index + ((index < mids.numel()) & (at == x) & (index % 2 == 1)).to(index.dtype)


def _stored_values(tensors, format, device, first, last):
    """Rows first..last of a stored bf16 (or 16/32-bit) or fp8 matrix as FP64 values, exactly (an E4M3 code times an
    FP32 scale fits FP64)."""
    import torch
    weight = tensors["weight"][first:last]
    if format == "bf16":
        return weight.to(device=device, dtype=torch.float64)
    codes = weight.to(device).view(torch.float8_e4m3fn).to(torch.float64)
    scale = tensors["scale"].to(device=device, dtype=torch.float32).reshape(-1)
    if format == "fp8:channel":
        return codes * scale[first:last].to(torch.float64)[:, None]
    return codes * scale[0].to(torch.float64)


def reencode(tensors, format, encode, device=None):
    """A draft-only matrix's stored tensors re-encoded to `encode`: (tensors, format) as load() takes them.

    nvfp4: global scale g = FP32(6 * 448 / amax) over the matrix (1 for an all-zero one); per 16 values the E4M3 scale
    e4m3(amax(block) * g / 6) and E2M1 codes e2m1(v * g / scale), round to nearest even, saturating at 448 and 6,
    zero for a zero scale: nvfp4.py's quantize_row at g, code for code and byte for byte. fp8: per row the FP32 scale
    amax / 448 (1 for an all-zero row) and E4M3 codes e4m3(v / scale), the same rounding (fp8:channel). The stored
    values are taken exactly (FP64) and the encoders run on `device` (the weight's own unless given), a slice of rows
    at a time."""
    import torch
    if encode not in ENCODINGS:
        raise ValueError(f"a draft-only matrix re-encodes to {', '.join(ENCODINGS)}, not {encode!r}")
    if format not in ENCODINGS[encode]:
        raise ValueError(f"a stored {format} matrix does not re-encode to {encode} "
                         f"(from {', '.join(ENCODINGS[encode])}: re-encoding only makes a matrix smaller)")
    weight = tensors["weight"]
    device = torch.device(device) if device is not None else weight.device
    rows, columns = weight.shape
    if encode == "nvfp4" and columns % 16:
        raise ValueError(f"an nvfp4 matrix has a multiple of 16 columns, not {columns}")
    step = max(1, ENCODE_BYTES // (columns * 8))
    spans = [(r, min(rows, r + step)) for r in range(0, rows, step)]
    tables = _encoder_tables(device)
    e4m3, e4m3_mids, e2m1_mids = tables["e4m3"], tables["e4m3_mids"], tables["e2m1_mids"]
    if encode == "fp8":
        codes = torch.empty(rows, columns, dtype=torch.uint8, device=device)
        scale = torch.empty(rows, dtype=torch.float32, device=device)
        for first, last in spans:
            values = _stored_values(tensors, format, device, first, last)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("a matrix with non-finite values does not re-encode")
            amax = values.abs().amax(dim=-1)
            s = torch.where(amax > 0, amax / codec.E4M3_MAX, torch.ones_like(amax)).to(torch.float32)
            q = values / s.to(torch.float64)[:, None]
            code = nearest(q.abs().clamp(max=codec.E4M3_MAX), e4m3, e4m3_mids)
            code = code | (((q < 0) & (code > 0)).to(code.dtype) << 7)
            codes[first:last], scale[first:last] = code.to(torch.uint8), s
        return {"weight": codes.view(torch.float8_e4m3fn), "scale": scale}, "fp8:channel"
    amax = 0.0
    for first, last in spans:
        values = _stored_values(tensors, format, device, first, last)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("a matrix with non-finite values does not re-encode")
        amax = max(amax, float(values.abs().max()))
    g32 = torch.tensor(codec.global_scale([amax]), dtype=torch.float32)
    g = float(g32)
    codes = torch.empty(rows, columns // 2, dtype=torch.uint8, device=device)
    scales = torch.empty(rows, columns // 16, dtype=torch.uint8, device=device)
    for first, last in spans:
        blocks = _stored_values(tensors, format, device, first, last).view(last - first, -1, 16)
        s = nearest((blocks.abs().amax(dim=-1) * g / codec.E2M1_MAX).clamp(max=codec.E4M3_MAX), e4m3, e4m3_mids)
        decoded = e4m3[s][..., None]
        q = torch.where(decoded > 0, blocks * g / decoded.clamp(min=e4m3[1]), torch.zeros_like(blocks))
        code = nearest(q.abs().clamp(max=codec.E2M1_MAX), tables["e2m1"], e2m1_mids)
        code = (code | (((q < 0) & (code > 0)).to(code.dtype) << 3)).view(last - first, columns)
        codes[first:last] = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
        scales[first:last] = s.to(torch.uint8)
    return {"weight": codes, "scale": scales, "global_scale": g32.to(device)}, "nvfp4"
