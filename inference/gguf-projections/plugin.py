"""gguf-projections: the projections slot over a GGUF checkpoint's matrices as stored (code/host.py's host form, Core's
decoders as the reference), on the device on crit's kernel gguf-rows.

A model plugin reads a matrix's stored bytes with crit_core.model.gguf_blocks.GgufTensors and loads it here as format
"gguf:<TYPE>" (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4) with tensors {"weight": uint8 [N][row bytes]}, or a stack of
experts [E][N][row bytes] (matmul_experts); a GGUF file's dense tensors (F32, F16, BF16) load as "bf16". Decode rows
(calls of up to DECODE_ROWS rows) run gguf-rows' gemv: int8 activations per 32 values against the stored blocks, each
row's bits its own. Prompt rows (longer calls, or prompt=True) take the `prompt` option: dequantize (the default: the
matrix decoded to BF16 a call, cuBLAS's GEMM; fastest today) or exact (gguf-rows' gemm: every prompt row with the bits
it has as a decode row). Where the kernel cannot be built here, the stored values decoded on the host at load (Core's
exact decoders) run as a BF16 matrix through torch, one row a call for decode rows, and the facts say so."""
from pathlib import Path

from .gguf.host import GgufWeight, check_rows, dense_rows, encode_q8_0, project_rows

DECODE_ROWS = 8
PROMPTS = ("dequantize", "exact")


class GgufMatrix:
    """A loaded matrix: format "gguf" with the kernel's matrix (`kernel`) or the host-decoded BF16 values (`dense`,
    the fallback), or format "bf16" with `dense`."""

    def __init__(self, format, rows, columns, kernel=None, dense=None, type=None):
        self.format, self.rows, self.columns = format, rows, columns
        self.kernel, self.dense, self.type = kernel, dense, type


class GgufProjections:
    def __init__(self, context, options):
        self.device = getattr(context, "device", None) or "cuda"
        self.prompt = options.get("prompt", "dequantize")
        if self.prompt not in PROMPTS:
            raise ValueError(f"gguf-projections: prompt is one of {', '.join(PROMPTS)}, not {self.prompt!r}")
        from crit_plugins.kernels import load_kernel
        self.kernel = load_kernel("gguf-rows", folder=Path(__file__).parent)
        self.formats = set()
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
            return GgufMatrix("bf16", w.shape[0], w.shape[1], dense=w)
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
        return GgufMatrix("gguf", n, k, dense=values.to(device=self.device, dtype=torch.bfloat16).contiguous(),
                          type=type_id)

    def matmul(self, x, matrix, out_dtype=None, prompt=False):
        """y = x W^T in BF16 (or out_dtype). Up to DECODE_ROWS rows (not a prompt chunk) each row's bits its own."""
        import torch
        if x.shape[-1] != matrix.columns:
            raise ValueError(f"activations of width {x.shape[-1]} for a matrix of {matrix.columns} columns")
        dtype = out_dtype or torch.bfloat16
        decode = x.shape[0] <= DECODE_ROWS and not prompt
        if matrix.kernel is not None:
            if decode:
                return self.kernel.gemv(x.to(torch.bfloat16), matrix.kernel).to(dtype)
            if self.prompt == "exact":
                return self.kernel.gemm(x.to(torch.bfloat16), matrix.kernel).to(dtype)
            dense = self.kernel.dequantize(matrix.kernel)
            return torch.mm(x.to(torch.bfloat16), dense.t(), out_dtype=torch.float32).to(dtype)
        w, x16 = matrix.dense, x.to(torch.bfloat16)
        if decode and x.shape[0] > 1:   # one row a call: each row's bits its own
            return torch.cat([self.matmul(x16[r:r + 1], matrix, out_dtype) for r in range(x.shape[0])])
        return torch.mm(x16, w.t(), out_dtype=torch.float32).to(dtype)

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
        return torch.cat([torch.mm(x16[t:t + 1], matrix.dense[int(e)].t(), out_dtype=torch.float32)
                          for t, e in enumerate(ids.tolist())]).to(dtype)

    def facts(self):
        exact = self.kernel is not None and self.prompt == "exact"
        return {"row_invariance": {"rows": DECODE_ROWS, "prompt_rows": exact, "prompt_exact": exact},
                "gguf_formats": sorted(self.formats), "prompt": self.prompt if self.kernel is not None else "fallback",
                **({"matrices": [{"N": n, "K": k, "format": f} for n, k, f in sorted(self.shapes)]}
                   if self.shapes else {})}


def create(context, options, parts):
    return GgufProjections(context, options)
