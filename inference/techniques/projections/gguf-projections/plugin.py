"""gguf-projections: the projections slot over a GGUF checkpoint's matrices as stored (code/host.py's host form, Core's
decoders as the reference), on the device on crit's kernel gguf-rows.

A model plugin reads a matrix's stored bytes with crit_core.model.gguf_blocks.GgufTensors and loads it here as format
"gguf:<TYPE>" (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4, Q1_0, Q2_0) with tensors {"weight": uint8 [N][row bytes]}, or a stack of
experts [E][N][row bytes] (matmul_experts); a GGUF file's dense tensors (F32, F16, BF16) load as "bf16". Decode rows
(calls of up to DECODE_ROWS rows) run gguf-rows' gemv: int8 activations per 32 values against the stored blocks, each
row's bits its own. Prompt rows (longer calls, or prompt=True) take the `prompt` option: dequantize (the matrix
decoded to BF16 a call, cuBLAS's GEMM; fastest today) or exact (gguf-rows' gemm: every prompt row with the bits it has
as a decode row); auto, the default, is dequantize where the engine's configuration asks repeatable prompt rows or
nothing and exact where it asks more (its exactness, crit_plugins.kernels.asks). Where the kernel cannot be built
here, the stored values decoded on the host at load (Core's exact decoders) run as a BF16 matrix through torch, one
row a call for decode rows, and the facts say so.

Dense matrices (bf16, and the fallback's decoded ones) take the same two paths: with prompt dequantize, decode rows
one row a call through torch and prompt rows cuBLAS's GEMM (repeatable); with prompt exact, every row with the bits
it has as a decode row, on crit's kernel mma-rows where it builds and takes the matrix (decode rows its decode tile,
prompt rows its prompt tile with the decode tile's bits: the BF16 pair quantized-projections' prompt_gemm tile runs
for a checkpoint's stored BF16 matrices, both at the K splits mma-rows' plan gives the matrix), else one row a call
through torch for prompt rows too (the decode rows' call, so their bits; slower, and the facts' dense_paths say so)."""
from pathlib import Path

from .gguf.host import GgufWeight, check_rows, dense_rows, encode_q8_0, project_rows

from crit_plugins.device import DECODE_ROWS   # noqa: E402 - the slot's decode rows (Core's contract), one definition
# The decode tile's partial sums of a call (DECODE_ROWS rows) at a matrix's K splits are at most this many bytes; a
# matrix whose plan asks more takes fewer splits (the prompt tile the same, so its rows keep the decode rows' bits).
PARTIAL_BYTES = 16 << 20
PROMPTS = ("auto", "dequantize", "exact")


def _mm(x16, w):
    """x16 [M][K] times w [N][K]^T, both BF16, in FP32: cuBLAS's FP32 sums on CUDA; on the CPU the BF16 values (exact in
    FP32) multiplied in FP32."""
    import torch
    if x16.is_cuda:
        return torch.mm(x16, w.t(), out_dtype=torch.float32)
    return torch.mm(x16.float(), w.float().t())


class GgufMatrix:
    """A loaded matrix: format "gguf" with the kernel's matrix (`kernel`) or the host-decoded BF16 values (`dense`,
    the fallback), or format "bf16" with `dense`."""

    def __init__(self, format, rows, columns, kernel=None, dense=None, type=None):
        self.format, self.rows, self.columns = format, rows, columns
        self.kernel, self.dense, self.type = kernel, dense, type
        # A dense matrix's path (dense_path): "tile" (mma-rows' decode and prompt tiles, `tile` its groups, warps and
        # K splits),
        # "rows" (one row a call for decode and prompt rows) or "torch" (one row a call for decode rows, cuBLAS's
        # GEMM for prompt rows); None for a matrix on gguf-rows.
        self.path, self.tile = None, None


class GgufProjections:
    def __init__(self, context, options):
        import torch
        # the context's device, else CUDA where the host has it and the CPU (the fallback) where it does not
        self.device = getattr(context, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt = options.get("prompt", "auto")
        if self.prompt not in PROMPTS:
            raise ValueError(f"gguf-projections: prompt is one of {', '.join(PROMPTS)}, not {self.prompt!r}")
        from crit_plugins.kernels import asks, load_kernel
        if self.prompt == "auto":
            # The fastest prompt path that keeps the level the engine asks of its prompt rows: cuBLAS's GEMM over the
            # decoded matrix follows the row count (repeatable); gguf-rows' gemm has the decode rows' bits.
            self.prompt = "dequantize" if asks("prompt") in (None, "repeatable") else "exact"
        import torch
        # gguf-rows on a CUDA device; on the CPU the fallback (the host-decoded values through torch)
        cuda = torch.device(self.device).type == "cuda"
        self.kernel = load_kernel("gguf-rows", folder=Path(__file__).parent) if cuda else None
        # Dense matrices' exact rows on CUDA: mma-rows' BF16 decode tile and its prompt tile with the same bits.
        self.tile = load_kernel("mma-rows", folder=Path(__file__).parent) if cuda and self.prompt == "exact" else None
        self.formats = set()
        self.dense_paths = set()
        self.partial = self.counters = None   # the decode tile's partial sums and counters, shared by every matrix
        self.shapes = set()   # (N, K, format) of every matrix loaded, for crit kernels sweep --matrices

    # The host form (code/host.py): Q8_0 blocks of a dense matrix, exact sums over the stored values.
    def prepare(self, rows):
        k = len(rows[0]) if rows and rows[0] else 0
        return GgufWeight(encode_q8_0(rows), 8, k)

    def project(self, weight, activations):
        return project_rows(weight, activations)

    def dense(self, weight):
        return dense_rows(weight)

    # The device form.
    def load(self, tensors, format, role=None, encode=None):
        """A matrix on the device from its stored tensors: "gguf:<TYPE>" with weight uint8 [N][row bytes] (or
        [experts][N][row bytes]), or "bf16" with a 16- or 32-bit weight [out][in]."""
        import numpy as np
        import torch
        from crit_core.model.gguf_blocks import TYPES, stored_values
        if encode is not None:
            raise ValueError("gguf-projections keeps matrices as stored: re-encoding is quantized-projections'")
        weight = tensors["weight"]
        if format == "bf16":
            w = torch.as_tensor(weight).to(device=self.device, dtype=torch.bfloat16).contiguous()
            self.formats.add("bf16")
            return self.dense_path(GgufMatrix("bf16", w.shape[0], w.shape[1], dense=w))
        name = format.split(":", 1)[1] if format.startswith("gguf:") else None
        if name not in TYPES:
            raise ValueError(f"gguf-projections loads gguf:<TYPE> ({', '.join(TYPES)}) or bf16 matrices, not {format!r}")
        type_id = TYPES[name]
        raw = np.ascontiguousarray(np.asarray(weight.cpu() if torch.is_tensor(weight) else weight, dtype=np.uint8))
        from crit_core.model.gguf_blocks import BLOCKS
        values, size = BLOCKS[type_id]
        if raw.ndim not in (2, 3) or raw.shape[-1] % size or not raw.shape[-1]:
            raise ValueError(f"gguf-projections: {name} rows of {raw.shape[-1:]} bytes are no whole number of the "
                             f"format's {size}-byte blocks")
        k = raw.shape[-1] // size * values
        check_rows(GgufWeight(raw, type_id, k))
        self.formats.add(name)
        n = raw.shape[-2]
        self.shapes.add((n, k, name))
        if self.kernel is not None:
            blocks = torch.from_numpy(raw).to(self.device)
            return GgufMatrix("gguf", n, k, kernel=self.kernel.GgufMatrix(blocks, type_id, k), type=type_id)
        # The fallback: the stored values, decoded exactly on the host, as BF16 on the device.
        flat = raw.reshape(-1, raw.shape[-1])
        values = torch.from_numpy(stored_values(flat, type_id, k)).reshape(*raw.shape[:-1], k)
        return self.dense_path(GgufMatrix("gguf", n, k, type=type_id,
                                          dense=values.to(device=self.device, dtype=torch.bfloat16).contiguous()))

    def dense_path(self, matrix):
        """The matrix with its dense path: with prompt exact, mma-rows' tiles where the kernel is here and takes the
        shape (K a multiple of its step; an expert stack is not tiled), else one row a call; with dequantize, torch."""
        if self.prompt != "exact":
            matrix.path = "torch"
        elif self.tile is not None and matrix.dense.dim() == 2 and self.tile.eligible(matrix.rows, matrix.columns):
            # The plan's splits of K: one split leaves a small matrix's call a few CTAs (a 4 x 10240 decode row ran
            # 25.5 us on one split, 3.4 us on the plan's 32, torch's GEMV 3.7 us; ~240 dense calls a decode token)
            groups, warps, splits = self.tile.plan(matrix.rows, matrix.columns, "bf16", matrix.dense.device)
            while splits > 1 and self.tile.partial_elements(DECODE_ROWS, matrix.rows, splits) * 4 > PARTIAL_BYTES:
                splits -= 1
            matrix.path, matrix.tile = "tile", (groups, warps, splits)
            self._tile_buffers(matrix, splits, warps)
        else:
            matrix.path = "rows"
        self.dense_paths.add(matrix.path)
        return matrix

    def _tile_buffers(self, matrix, splits, warps):
        """The decode tile's partial sums and counters, made at load as large as the largest matrix's call needs (a
        call in a captured round must not allocate them); the kernel leaves its counters zero again."""
        import torch
        partial = max(1, self.tile.partial_elements(DECODE_ROWS, matrix.rows, splits))
        counters = self.tile.row_tiles(matrix.rows, 1, warps)
        device = matrix.dense.device
        if self.partial is None or self.partial.numel() < partial:
            self.partial = torch.empty(partial, dtype=torch.float32, device=device)
        if self.counters is None or self.counters.numel() < counters:
            self.counters = torch.zeros(counters, dtype=torch.int32, device=device)

    def _tiled(self, x16, matrix, decode):
        """Rows through mma-rows' BF16 tiles in FP32: decode rows on its decode tile, prompt rows on its prompt tile,
        whose outputs have the decode tile's bits for the same matrix and K splits."""
        import torch
        x16 = x16.contiguous()
        if x16.data_ptr() % 16:
            x16 = x16.clone()
        out = torch.empty(x16.shape[0], matrix.rows, dtype=torch.float32, device=x16.device)
        groups, warps, splits = matrix.tile
        if decode:
            return self.tile.mm_rows(x16, matrix.dense, out, "bf16", groups=groups, warps=warps, splits=splits,
                                     partial=self.partial, counters=self.counters)
        return self.tile.mm_prompt(x16, matrix.dense, out, "bf16", splits=splits)

    def matmul(self, x, matrix, out_dtype=None, *, prompt):
        """y = x W^T in BF16 (or out_dtype). prompt (required, the rows' phase): False, decode rows, at most
        DECODE_ROWS a call, each row's bits its own; True, a prompt chunk of any length."""
        import torch
        if x.shape[-1] != matrix.columns:
            raise ValueError(f"activations of width {x.shape[-1]} for a matrix of {matrix.columns} columns")
        dtype = out_dtype or torch.bfloat16
        decode = not prompt
        if decode:
            from crit_plugins.device import decode_call
            decode_call(x.shape[0], DECODE_ROWS, "gguf-projections")
        if matrix.kernel is not None:
            if decode:
                return self.kernel.gemv(x.to(torch.bfloat16), matrix.kernel).to(dtype)
            if self.prompt == "exact":
                return self.kernel.gemm(x.to(torch.bfloat16), matrix.kernel).to(dtype)
            dense = self.kernel.dequantize(matrix.kernel)
            return _mm(x.to(torch.bfloat16), dense).to(dtype)
        w, x16 = matrix.dense, x.to(torch.bfloat16)
        if matrix.path == "tile":
            return self._tiled(x16, matrix, decode).to(dtype)
        # One row a call for decode rows, and for prompt rows on the exact path: each row's bits its own, a prompt
        # row's those of the row decoded.
        if x.shape[0] > 1 and (decode or matrix.path == "rows"):
            return torch.cat([_mm(x16[r:r + 1], w).to(dtype) for r in range(x.shape[0])])
        return _mm(x16, w).to(dtype)

    def matmul_experts(self, x, matrix, ids, out_dtype=None):
        """Row t times expert ids[t] of a stack loaded from [experts][N][row bytes] (up to DECODE_ROWS rows), each
        row's bits those of the plain call on its expert."""
        import torch
        if x.shape[0] > DECODE_ROWS:
            raise ValueError(f"matmul_experts takes up to {DECODE_ROWS} rows, not {x.shape[0]}")
        dtype = out_dtype or torch.bfloat16
        if matrix.kernel is not None:
            ids32 = ids.to(device=x.device, dtype=torch.int32).contiguous()
            return self.kernel.gemv(x.to(torch.bfloat16), matrix.kernel, ids32).to(dtype)
        x16 = x.to(torch.bfloat16)
        return torch.cat([_mm(x16[t:t + 1], matrix.dense[int(e)]) for t, e in enumerate(ids.tolist())]).to(dtype)

    def exactness(self):
        """The level the loaded matrices' rows keep (crit_plugins.exactness; Core compares it with the level the
        configuration asks). On gguf-rows, decode rows are its gemv, the kernel's declared order, and prompt rows
        its gemm in the same order with prompt exact, or cuBLAS's GEMM over the decoded matrix with dequantize, whose
        sums follow the row count: repeatable. A dense matrix (bf16, and every matrix where the kernel does not
        build) keeps its decode rows' bits its own (row-exact: mma-rows' decode tile, or one row a call through
        torch), and its prompt rows those of the rows decoded with prompt exact (exact: mma-rows' prompt tile, or one
        row a call) or cuBLAS's GEMM's with dequantize (repeatable). Nothing before a matrix is loaded."""
        if not self.formats:
            return {}
        dense = bool(self.dense_paths)
        if self.prompt != "exact":
            prompt = "repeatable"
        else:
            prompt = "exact" if dense else "declared"
        return {"decode": "row-exact" if dense else "declared", "prompt": prompt}

    def facts(self):
        exact = self.prompt == "exact"
        return {"row_invariance": {"rows": DECODE_ROWS, "prompt_rows": exact, "prompt_exact": exact},
                "gguf_formats": sorted(self.formats), "prompt": self.prompt if self.kernel is not None else "fallback",
                **({"dense_paths": sorted(self.dense_paths)} if self.dense_paths else {}),
                **({"matrices": [{"N": n, "K": k, "format": f} for n, k, f in sorted(self.shapes)]}
                   if self.shapes else {})}


def create(context, options, parts):
    return GgufProjections(context, options)
