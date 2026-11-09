"""Block-scaled rows on the CPU in a declared order in the projections slot.

Host form: host-expert-arithmetic's dot_row (implementation.py) over quantized-projections' NVFP4 matrices (nvfp4.py),
in the order the options declare: the oracle.

Device form on the CPU (CPU torch tensors): a checkpoint's stored NVFP4 or INT4 matrices multiplied in a declared order,
each row of x on its own (activations quantized per row), so a row's bits never depend on the rows beside it at any row
count: weight only (activations fp32, host_nvfp4's 16-lane FP32 order), W4A4 (e2m1x2 under the matrix's input_scale) or
W4A8 (int8 per 32), host_int's integer block order; 16-bit matrices through torch a row at a time. The native rows
(host_capi.cpp, built at load by native.py) or, where they do not build, the same orders in torch (declared.py), the same
bits."""
import os
from pathlib import Path

from .arithmetic import implementation as arithmetic
from .quantized import nvfp4

FORMATS = ("bf16", "nvfp4", "int4")


class Matrix:
    def __init__(self, format, rows, columns, tensors, activations):
        self.format, self.rows, self.columns, self.tensors, self.activations = format, rows, columns, tensors, activations


def _bytes(tensor):
    import torch
    t = tensor.detach().cpu().contiguous()
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def _scalar(value):
    import torch
    values = torch.as_tensor(value).float().reshape(-1)
    if values.numel() != 1:
        raise ValueError("an input scale is one number")
    return float(values[0])


class HostProjections:
    def __init__(self, options, code, quantized):
        self.options = options
        self.code, self.quantized = code, quantized
        self.native, self.reason = None, None
        self.threads = options["threads"] if options["threads"] > 0 else max(1, (os.cpu_count() or 2) // 2)

    # ------------------------------------------------------------------ host form (the oracle)

    def prepare(self, rows):
        return nvfp4.Matrix.quantize([[float(v) for v in row] for row in rows])

    def project(self, weight, activations):
        o, cols = self.options, weight.cols
        rows = [(weight.packed[n * cols // 2:(n + 1) * cols // 2],
                 arithmetic.standard_scales(weight.scales[n * cols // nvfp4.BLOCK:(n + 1) * cols // nvfp4.BLOCK]))
                for n in range(weight.rows)]
        return [[arithmetic.dot_row(packed, scales, [arithmetic.f32(v) for v in x], o["lanes"], o["reduce"],
                                    arithmetic.f32(1.0 / weight.g), o["product"]) for packed, scales in rows]
                for x in activations]

    def dense(self, weight):
        return [weight.row(n) for n in range(weight.rows)]

    # ------------------------------------------------------------------ device form (CPU tensors)

    def _rows(self):
        """The native rows, or None (the torch path) where they do not build or the options say off."""
        if self.native is None and self.options["native"] != "off":
            from importlib import import_module
            native = import_module(f"{self.code.__name__}.native")
            rows = import_module(f"{self.code.__name__}.rows")
            try:
                self.native = rows.HostRows.build(self.quantized)
            except native.NativeUnavailable as error:
                if self.options["native"] == "on":
                    raise
                self.native, self.reason = False, str(error).splitlines()[0]
        return self.native or None

    def _declared(self):
        from importlib import import_module
        return import_module(f"{self.code.__name__}.declared")

    def load(self, tensors, format, role=None):
        """A matrix from a checkpoint's stored tensors: bf16 ({weight} 16-bit), nvfp4 or int4 ({weight, scale,
        global_scale, input_scale}, divisors; E4M3 scale per 16)."""
        import torch
        if format not in FORMATS:
            raise ValueError(f"host-expert-arithmetic's device form loads {', '.join(FORMATS)}, not {format!r}")
        o = self.options
        if format != "bf16" and (o["lanes"], o["reduce"], o["product"]) != (16, "halving", "scaled_elements"):
            raise ValueError("the device form runs the 16-lane halving scaled_elements order (the options' default)")
        weight = tensors["weight"]
        if format == "bf16":
            w = weight.detach().to("cpu", torch.bfloat16).contiguous()
            return Matrix(format, w.shape[0], w.shape[1], {"weight": w}, "bf16")
        codes = _bytes(weight)
        rows, columns = codes.shape[0], codes.shape[1] * 2
        scales = _bytes(tensors["scale"])
        if columns % 64 or tuple(scales.shape) != (rows, columns // 16):
            raise ValueError(f"a {format} matrix of {rows} x {columns} (a multiple of 64) has {rows} x {columns // 16} "
                             "block scales")
        g = tensors.get("global_scale")
        if g is None and format == "nvfp4":
            raise ValueError("an nvfp4 matrix has its global_scale")
        g = torch.ones(1) if g is None else torch.as_tensor(g).detach().cpu().float().reshape(-1)   # int4: optional
        if g.numel() not in (1, rows):
            raise ValueError(f"a matrix of {rows} rows has one global scale or one per row, not {g.numel()}")
        inverse = (1.0 / g).expand(rows).contiguous()
        activations = o["activations"]
        if format == "int4" and activations == "fp32":
            activations = "int8"   # INT4 rows have no FP32 order here: W4A8
        found = {"codes": codes, "scales": scales, "inverse": inverse}
        if activations == "e2m1x2":
            if tensors.get("input_scale") is None:
                raise ValueError("W4A4 (activations e2m1x2) needs the matrix's input_scale")
            found["input"] = _scalar(tensors["input_scale"])
        return Matrix(format, rows, columns, found, activations)

    def matmul(self, x, matrix, out_dtype=None):
        import torch
        if x.device.type != "cpu":
            raise ValueError("host-expert-arithmetic's device form runs on CPU tensors")
        tokens = x.shape[0]
        dtype = out_dtype or x.dtype
        if matrix.format == "bf16":
            w = matrix.tensors["weight"].float()
            xf = x.float()
            out = torch.empty(tokens, matrix.rows, dtype=torch.float32)
            for t in range(tokens):   # each row alone: torch's sums never see the rows beside it
                out[t] = (xf[t:t + 1] @ w.T)[0]
            return out.to(dtype)
        t = matrix.tensors
        native, declared = self._rows(), self._declared()
        xf = x.float().contiguous()
        weights = "int4" if matrix.format == "int4" else "e2m1x2"
        if matrix.activations == "fp32":
            if native:
                out = native.nvfp4_rows(t["codes"], t["scales"], t["inverse"], xf, threads=self.threads)
            else:
                out = declared.nvfp4_rows(t["codes"], t["scales"], t["inverse"], xf)
            return out.to(dtype)
        factor = (0.25 if matrix.activations == "e2m1x2" else 0.5) if weights == "e2m1x2" else \
            (0.5 if matrix.activations == "e2m1x2" else 1.0)
        if matrix.activations == "e2m1x2":
            values, scales = (native or declared).quantize_e2m1x2(xf, t["input"])
            block = 16
        else:
            values, scales = (native or declared).quantize_int8(xf, 32)
            block = 32
        if native:
            out = native.int_rows(t["codes"], t["scales"], t["inverse"], values, scales, block, weights, factor,
                                  threads=self.threads)
        else:
            out = declared.int_rows(t["codes"], t["scales"], t["inverse"], values, scales, block, weights, factor)
        return out.to(dtype)

    def facts(self):
        rows = self._rows()
        return {"row_invariance": {"decode_rows": True, "prompt_rows": True},
                "cpu_rows": f"native {rows.int_isa()}" if rows else
                "torch, the declared order" + (f" ({self.reason})" if self.reason else "")}


def create(context, options, parts):
    from . import arithmetic as code, quantized
    return HostProjections(options, code, Path(quantized.__path__[0]))
