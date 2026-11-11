"""pinned-gemm's sweep benches (crit_kernels.sweep): the model's matrices in each structure's formats, random operands
of the format (the package's operands(), one set a shape whose rows are sliced for every row count, so row 0 never
depends on the count), a GEMM pinned once a build, shape and knob setting with the bench's weight, and an FP64
reference of the dequantized operands for the first rows.

A shape is {N, K, IN, OUT, format}: format a pinned GEMM format (pin()'s), IN and OUT its input and output bits (the
tuning's key, the package's structure())."""
import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
MAX_ROWS = 4096     # the most rows the structures' tune_rows ask for
CHECKED = 64        # rows of each call checked against the FP64 reference
BITS = {"bf16": 16, "fp16": 16, "fp32": 32}


def _shape(n, k, fmt, a_bits):
    return {"N": int(n), "K": int(k), "IN": a_bits, "OUT": BITS[fmt.split(":")[1]], "format": fmt}


def _dequantized(f, x, scale, side):
    if f[side] == "e2m1":
        table = torch.tensor(E2M1, dtype=torch.float64, device=x.device)
        v = torch.stack((table[(x & 15).long()], table[(x >> 4).long()]), dim=-1).reshape(x.shape[0], -1)
    else:
        v = x.double()
    if f["scale"] == "tensor":
        v = v * scale.double()
    elif f["scale"] == "row":
        v = v * scale.double()[:, None]
    elif f["scale"] == "vec32":
        v = v * torch.pow(2.0, scale.double() - 127).repeat_interleave(32, dim=1)
    elif f["scale"] == "vec16":
        v = v * scale.view(torch.float8_e4m3fn).double().repeat_interleave(16, dim=1)
    return v


class _Pinned:
    structure = None
    formats = ()     # (model matrix format, pinned GEMM formats, a's bits)

    def __init__(self):
        self._data, self._pins, self._calls, self._references = {}, {}, {}, {}
        self._module, self._library = None, {}

    def shapes(self, model):
        found = []
        for m in model.matrices():
            for fmt_in, fmts, bits in self.formats:
                if m["format"] == fmt_in:
                    found += [_shape(m["N"], m["K"], fmt, bits) for fmt in fmts]
        return found

    def inputs(self, shape, rows, device):
        return {"shape": shape, "rows": int(rows), "device": device}

    def _operands(self, module, s):
        shape = s["shape"]
        key = (shape["format"], shape["N"], shape["K"])
        if key not in self._data:   # one set a shape, kept for every configuration the sweep times
            fmt, n, k = key
            f = module.parse(fmt)
            a, b, scales = module.operands(fmt, n, k, MAX_ROWS, s["device"], seed=n * 31 + k)
            sb = module.blocked_scales(scales["b"]) if f["block"] else scales["b"]
            self._data[key] = (f, a, b, scales, sb)
        return key, self._data[key]

    def _pin(self, module, key, data):
        fmt, n, k = key
        knobs = module.tuned(fmt, n, k)   # the configuration the sweep selected in this build
        pkey = (id(module), key, tuple(sorted(knobs.items())))
        if pkey not in self._pins:
            self._pins[pkey] = module.pin(fmt, n, k, data[1].device, weight=(data[2], data[4]))
        return self._pins[pkey]

    def run(self, module, s):
        self._module = module   # the yardstick takes the operands the same package made
        key, data = self._operands(module, s)
        f, a, b, scales, sb = data
        p = self._pin(module, key, data)
        rows = s["rows"]
        ckey = (key, rows)
        if ckey not in self._calls:
            sa = scales["a"]
            if f["scale"] == "row":
                sa = sa[:rows].contiguous()
            elif f["block"]:
                sa = module.blocked_scales(sa[:rows])
            out = torch.empty(rows, key[1], dtype=p.dtype, device=a.device)
            self._calls[ckey] = (a[:rows], sa, out)
        x, sa, out = self._calls[ckey]
        return p.mm(x, b, sa, sb, scales["alpha"], out=out)

    def yardstick(self, s):
        """cuBLASLt through torch on the same operands, unpinned: torch.mm (16-bit, FP32 output through out_dtype) or
        torch._scaled_mm (FP8 with unit tensor scales; NVFP4 with the same blocked scales), whose heuristic picks the
        algorithm by the row count, split-K included. NVFP4's global alpha is left out (a power of two in the
        epilogue). What pinning one algorithm a shape costs or saves against the library's own choice."""
        if self._module is None:
            raise RuntimeError("the yardstick takes the operands of a timed call; none ran")
        key, data = self._operands(self._module, s)
        f, a, b, scales, sb = data
        rows = s["rows"]
        out = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s["shape"]["format"].split(":")[1]]
        x = a[:rows]
        if f["block"]:
            lk = ("blocked", key, rows)
            if lk not in self._library:
                self._library[lk] = self._module.blocked_scales(scales["a"][:rows])
            sa = self._library[lk]
            x4, b4 = x.view(torch.float4_e2m1fn_x2), b.view(torch.float4_e2m1fn_x2)
            sa8, sb8 = sa.view(torch.float8_e4m3fn), sb.view(torch.float8_e4m3fn)
            return lambda: torch._scaled_mm(x4, b4.t(), sa8, sb8, out_dtype=out)
        if f["a"] in ("e4m3", "e5m2"):
            one = torch.ones((), dtype=torch.float32, device=x.device)
            if f["scale"] == "none":
                return lambda: torch._scaled_mm(x, b.t(), one, one, out_dtype=out)
            raise NotImplementedError(f"{f['name']}: the bench's formats are unscaled FP8")
        if out == x.dtype:
            return lambda: torch.matmul(x, b.t())
        return lambda: torch.mm(x, b.t(), out_dtype=out)

    def reference(self, s):
        shape = s["shape"]
        key = (shape["format"], shape["N"], shape["K"])
        if key not in self._references:   # the first CHECKED rows, the same for every row count
            f, a, b, scales, _ = self._data[key]
            sa = scales["a"][:CHECKED] if f["scale"] == "row" or f["block"] else scales["a"]
            x = _dequantized(f, a[:CHECKED], sa, "a")
            exact = torch.empty(CHECKED, key[1], dtype=torch.float64, device=a.device)
            step = 2048
            for n0 in range(0, key[1], step):   # the weight dequantized a slice at a time (FP64 is 8 bytes a value)
                sb = scales["b"]
                if f["scale"] == "row" or f["block"]:
                    sb = sb[n0:n0 + step]
                exact[:, n0:n0 + step] = x @ _dequantized(f, b[n0:n0 + step], sb, "b").t()
            if scales["alpha"] is not None:
                exact *= scales["alpha"].double()
            self._references[key] = exact
        return self._references[key]

    def check(self, got, want, s):
        rows = min(s["rows"], CHECKED)
        error = float((got[:rows].double() - want[:rows]).abs().max() / want[:rows].abs().max())
        # FP32 accumulation of the products (the pinned GEMMs' compute type), then one rounding to the output type
        bound = 2e-5 if s["shape"]["OUT"] == 32 else 1e-2
        return None if error < bound else f"relative error {error:.2e} beyond {bound:.0e}"

    def first_row(self, out):
        return out[0]

    def bytes(self, shape, rows):
        n, k, bits = int(shape["N"]), int(shape["K"]), int(shape["IN"])
        return (n + rows) * k * bits // 8 + rows * n * int(shape["OUT"]) // 8


class LtDense(_Pinned):
    structure = "lt-dense"
    formats = (("bf16", ("bf16:bf16", "bf16:fp32"), 16), ("f16", ("fp16:fp32",), 16))


class LtFp8(_Pinned):
    structure = "lt-fp8"
    formats = (("fp8", ("fp8e4m3:fp32",), 8),)   # FP8 into FP32, the row scales applied after (the plugin's path)


class LtBlock(_Pinned):
    structure = "lt-block"
    formats = (("nvfp4", ("nvfp4:bf16", "nvfp4:fp32"), 4),)
