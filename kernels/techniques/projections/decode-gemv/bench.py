"""decode-gemv's sweep benches (crit_kernels.sweep): the model's matrices in each structure's format, random codes and
scales of the stored layout, BF16 activations whose row i is seeded by i alone, an FP64 reference of the stored
values."""
import torch

from crit_kernels import contracts
from crit_kernels.weight_formats import INT4_GROUP, WEIGHTS, int4_matrix, int4_shapes, int4_values, int4_yardstick

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _activations(rows, k, device):
    return torch.stack([torch.randn(k, generator=torch.Generator().manual_seed(1000 + i)) for i in range(rows)]).to(
        device, torch.bfloat16)


class GemvNvfp4:
    """NVFP4 matrices, their activations and the FP64 reference (gemv-cuda's bench takes them for weights 0; the
    per-format structures they benched are retired)."""

    def __init__(self):
        self._matrices, self._references = {}, {}

    def shapes(self, model):
        return [m for m in model.matrices() if m["format"] == "nvfp4"]

    def valid(self, knobs):
        return knobs["WPR"] <= knobs["WARPS"]

    def inputs(self, shape, rows, device):
        n, k = int(shape["N"]), int(shape["K"])
        if (n, k) not in self._matrices:   # one matrix a shape, kept for every configuration the sweep times
            g = torch.Generator(device=device).manual_seed(n * 31 + k)
            codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8, device=device)
            scales = (torch.rand(n, k // 16, generator=g, device=device) * 0.4 + 0.1).to(torch.float8_e4m3fn)
            inverse = (torch.rand(n, generator=g, device=device) + 0.5) / 64
            self._matrices[(n, k)] = (codes, scales, inverse)
        codes, scales, inverse = self._matrices[(n, k)]
        x = _activations(rows, k, device)
        out = torch.empty(rows, n, device=device, dtype=torch.bfloat16)
        return {"x": x, "codes": codes, "scales": scales, "inverse": inverse, "out": out, "key": (n, k, rows)}

    def run(self, module, s):
        return module.gemv(s["x"], s["codes"], s["scales"], s["inverse"], s["out"])

    def reference(self, s):
        if s["key"] not in self._references:
            self._references[s["key"]] = self._reference(s)
        return self._references[s["key"]]

    def _reference(self, s):
        table = torch.tensor(E2M1, dtype=torch.float64, device=s["codes"].device)
        x = s["x"].double()
        values, magnitudes = [], []
        for lo in range(0, s["codes"].shape[0], 4096):   # 4096 matrix rows at a time: a vocabulary head fits
            c = s["codes"][lo:lo + 4096].long()
            w = torch.stack((table[c & 15], table[c >> 4]), dim=-1).reshape(c.shape[0], -1)
            w = (w.view(c.shape[0], -1, 16) * s["scales"][lo:lo + 4096].double()[..., None]).reshape(c.shape[0], -1)
            inv = s["inverse"][lo:lo + 4096].double()[None, :]
            values.append(x @ w.t() * inv)
            magnitudes.append(x.abs() @ w.abs().t() * inv)
        return torch.cat(values, dim=1), torch.cat(magnitudes, dim=1)

    def check(self, got, want, s):
        return _bf16_bound(got, want)

    def first_row(self, out):
        return out[0]

    def bytes(self, shape, rows):
        n, k = int(shape["N"]), int(shape["K"])
        return n * k // 2 + n * k // 16 + n * 4 + rows * k * 2 + rows * n * 2

    def yardstick(self, s):
        """flashinfer's W4A16 NVFP4 GEMM on the same stored bytes (_w4a16_yardstick)."""
        return _w4a16_yardstick(self._matrices, s)


def _chunked(x, rows_of, n, chunk=8192):
    """(x @ W.T, |x| @ |W|.T) in FP64 over the matrix's rows in chunks: rows_of(lo, hi) gives W[lo:hi] in FP64 (a
    vocabulary head's FP64 values would not fit the device at once)."""
    xd = x.double()
    value = torch.empty(x.shape[0], n, dtype=torch.float64, device=x.device)
    magnitude = torch.empty_like(value)
    for lo in range(0, n, chunk):
        w = rows_of(lo, min(n, lo + chunk))
        value[:, lo:lo + w.shape[0]] = xd @ w.t()
        magnitude[:, lo:lo + w.shape[0]] = xd.abs() @ w.abs().t()
    return value, magnitude


def _bf16_bound(got, want):
    """A projection's tolerance against its FP64 reference (value, magnitudes): FP32 sums of exact products (a relative
    2^-20 or so of the magnitudes), then one BF16 rounding of the result (crit_kernels.contracts.Sums, every projection
    operation's)."""
    return contracts.check("nvfp4-projection", got, want)


class _Cached:
    """Matrices kept per shape and references per (shape, rows) across the sweep's calls."""

    def __init__(self):
        self._matrices, self._references = {}, {}

    def reference(self, s):
        if s["key"] not in self._references:
            self._references[s["key"]] = self._reference(s)
        return self._references[s["key"]]

    def check(self, got, want, s):
        return _bf16_bound(got, want)

    def first_row(self, out):
        return out[0]


def _fp8_codes(n, k, device):
    g = torch.Generator(device=device).manual_seed(n * 37 + k)
    codes = torch.randint(0, 0x7E, (n, k), generator=g, dtype=torch.uint8, device=device)   # no NaN codes
    codes |= torch.randint(0, 2, (n, k), generator=g, dtype=torch.uint8, device=device) << 7
    row_scale = torch.rand(n, generator=g, device=device) / 64 + 1e-3
    return codes, row_scale


def _mx_codes(n, k, device):
    """MXFP8 rows: the FP8 codes of _fp8_codes and an E8M0 exponent a 32 inputs (scales 2^-9 .. 2^0)."""
    codes, _ = _fp8_codes(n, k, device)
    g = torch.Generator(device=device).manual_seed(n * 43 + k)
    exps = torch.randint(118, 128, (n, k // 32), generator=g, dtype=torch.uint8, device=device)
    return codes, exps


def _fp8_yardstick(cache, s):
    """No library multiplies FP8 weights by 16-bit activations (cuBLASLt's FP8 GEMMs want FP8 on both sides, which
    quantizes the activations: other arithmetic), so the yardstick is cuBLAS on the stored values times their row
    scales widened to BF16 once (an E4M3 value is exact in BF16, its product with the scale rounded once): what an
    engine without a W8A16 kernel runs. It reads twice the weight bytes."""
    codes, scales = s["codes"], s["scales"]
    n, k = codes.shape
    if ("wide", n, k) not in cache:
        wide = torch.empty(n, k, dtype=torch.bfloat16, device=codes.device)
        for lo in range(0, n, 8192):
            wide[lo:lo + 8192] = (codes[lo:lo + 8192].view(torch.float8_e4m3fn).float() *
                                  scales[lo:lo + 8192, None]).to(torch.bfloat16)
        cache[("wide", n, k)] = wide
    wide, x = cache[("wide", n, k)], s["x"]
    return lambda: torch.matmul(x, wide.t())


class GemvFp8(_Cached):
    """The model's FP8 matrices with row scales (MXFP8 runs the same kernel with one more byte a step): gemv-cuda's
    bench takes their codes and reference for weights 1 and 2."""

    def shapes(self, model):
        return [m for m in model.matrices() if m["format"] == "fp8"]

    def inputs(self, shape, rows, device):
        n, k = int(shape["N"]), int(shape["K"])
        if (n, k) not in self._matrices:
            self._matrices[(n, k)] = _fp8_codes(n, k, device)
        codes, row_scale = self._matrices[(n, k)]
        return {"x": _activations(rows, k, device), "codes": codes, "scales": row_scale,
                "out": torch.empty(rows, n, device=device, dtype=torch.bfloat16), "key": (n, k, rows)}

    def run(self, module, s):
        return module.gemv_fp8(s["x"], s["codes"], s["scales"], s["out"])

    def yardstick(self, s):
        return _fp8_yardstick(self._matrices, s)

    def _reference(self, s):
        codes, scales = s["codes"], s["scales"]
        return _chunked(s["x"], lambda lo, hi: codes[lo:hi].view(torch.float8_e4m3fn).double() *
                        scales[lo:hi].double()[:, None], codes.shape[0])

    def bytes(self, shape, rows):
        n, k = int(shape["N"]), int(shape["K"])
        return n * k + n * 4 + rows * k * 2 + rows * n * 2


class GemvBf16(_Cached):
    """BF16 matrices: gemv-cuda's bench takes their sums inputs and reference for weights 3."""

    def shapes(self, model):
        return [m for m in model.matrices() if m["format"] == "bf16"]

    def inputs(self, shape, rows, device):
        n, k = int(shape["N"]), int(shape["K"])
        if (n, k) not in self._matrices:
            g = torch.Generator(device=device).manual_seed(n * 41 + k)
            self._matrices[(n, k)] = (torch.randn(n, k, generator=g, device=device) / 16).to(torch.bfloat16)
        return {"x": _activations(rows, k, device), "w": self._matrices[(n, k)],
                "out": torch.empty(rows, n, device=device, dtype=torch.bfloat16), "key": (n, k, rows)}

    def integer_inputs(self, shape, rows, device):
        """The gates' sums inputs (crit_kernels.gates): integers 0 to 15 on both sides, the matrix's second half the
        first half negated plus a small integer at its end, so partial sums pass FP16's range and BF16's precision
        while every FP32 sum is exact and each output is a small integer."""
        n, k = int(shape["N"]), int(shape["K"])
        g = torch.Generator(device="cpu").manual_seed(n * 43 + k)
        x = torch.randint(0, 16, (rows, k), generator=g).float()
        w = torch.randint(0, 16, (n, k), generator=g).float()
        w[:, k // 2:k // 2 * 2] = -w[:, :k // 2]
        w[:, -1] += torch.arange(n).float() % 7
        return {"x": x.to(device, torch.bfloat16), "w": w.to(device, torch.bfloat16),
                "out": torch.empty(rows, n, device=device, dtype=torch.bfloat16), "key": ("integer", n, k, rows)}

    def integer_reference(self, s):
        return s["x"].double() @ s["w"].double().t()

    def run(self, module, s):
        return module.gemv_bf16(s["x"], s["w"], s["out"])

    def yardstick(self, s):
        """cuBLAS's GEMM on the same BF16 matrix (torch.matmul: its heuristic picks the kernel by the row count)."""
        w, x = s["w"], s["x"]
        return lambda: torch.matmul(x, w.t())

    def _reference(self, s):
        w = s["w"]
        return _chunked(s["x"], lambda lo, hi: w[lo:hi].double(), w.shape[0])

    def bytes(self, shape, rows):
        n, k = int(shape["N"]), int(shape["K"])
        return n * k * 2 + rows * k * 2 + rows * n * 2


def _blocked(scales):
    """[rows][blocks] one-byte scales in the 128 x 4 tiled layout of cuBLASLt and flashinfer (rows padded to 128,
    blocks to 4), flat, the dtype kept."""
    rows, cols = scales.shape
    pr, pc = -(-rows // 128) * 128, -(-cols // 4) * 4
    padded = torch.zeros(pr, pc, dtype=torch.uint8, device=scales.device)
    padded[:rows, :cols] = scales.contiguous().view(torch.uint8)
    tiles = padded.view(pr // 128, 128, pc // 4, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).view(scales.dtype)


def _w4a16_yardstick(cache, s):
    """flashinfer's BF16 x NVFP4 GEMM (mm_bf16_fp4, backend cute-dsl-native on sm_120: the stored codes as they are,
    the E4M3 block scales put in its 128 x 4 layout once), the same W4A16 arithmetic on the same bytes; the per-row
    inverse global scale is left out (flashinfer takes one alpha a matrix). None where flashinfer is not installed."""
    try:
        import flashinfer
    except ImportError:
        return None
    codes, x = s["codes"], s["x"]
    n, k = codes.shape[0], codes.shape[1] * 2
    if ("blocked", n, k) not in cache:
        cache[("blocked", n, k)] = _blocked(s["scales"])
    sb = cache[("blocked", n, k)]
    return lambda: flashinfer.mm_bf16_fp4(x, codes, sb, None, backend="cute-dsl-native")


class GemvCuda(_Cached):
    """gemv-cuda on the model's matrices of every format it takes (NVFP4, FP8, BF16 by model.matrices(); AWQ and GPTQ
    INT4 by their qweight tensors; MXFP8 from a shapes file), each checked against an FP64 reference of the stored
    values."""
    structure = "gemv-cuda"

    def shapes(self, model):
        found = [dict(m, weights=WEIGHTS[m["format"]]) for m in model.matrices()
                 if m["format"] in ("nvfp4", "fp8", "bf16")]
        return found + int4_shapes(model)

    def valid(self, knobs):
        return knobs["WPR"] <= knobs["WARPS"]

    def inputs(self, shape, rows, device):
        w, n, k = int(shape["weights"]), int(shape["N"]), int(shape["K"])
        group = int(shape.get("group") or INT4_GROUP)
        key = (w, n, k, group)
        if key not in self._matrices:
            if w == 0:
                held = GemvNvfp4().inputs({"N": n, "K": k}, 1, device)
                self._matrices[key] = (held["codes"], held["scales"], held["inverse"])
            elif w == 1:
                self._matrices[key] = _fp8_codes(n, k, device)
            elif w == 2:
                self._matrices[key] = _mx_codes(n, k, device)
            elif w == 3:
                g = torch.Generator(device=device).manual_seed(n * 41 + k)
                self._matrices[key] = (torch.randn(n, k, generator=g, device=device) / 16).to(torch.bfloat16)
            else:
                self._matrices[key] = int4_matrix(
                    n, k, group, torch.Generator(device=device).manual_seed(n * 47 + k + group))
        held = self._matrices[key]
        s = {"weights": w, "x": _activations(rows, k, device), "group": group, "key": (w, n, k, group, rows),
             "out": torch.empty(rows, n, device=device, dtype=torch.bfloat16)}
        if w == 0:
            s.update(codes=held[0], scales=held[1], inverse=held[2])
        elif w in (1, 2):
            s.update(codes=held[0], scales=held[1])
        elif w == 3:
            s.update(w=held)
        else:
            s.update(codes=held[0], scales=held[1], zeros=held[2])
        return s

    def integer_inputs(self, shape, rows, device):
        """The gates' sums inputs for BF16 weights (GemvBf16's), None for the stored formats."""
        if int(shape["weights"]) != 3:
            return None
        return dict(GemvBf16().integer_inputs(shape, rows, device), weights=3)

    def integer_reference(self, s):
        return GemvBf16().integer_reference(s)

    def run(self, module, s):
        w = s["weights"]
        if w == 0:
            return module.gemv(s["x"], s["codes"], s["scales"], s["inverse"], s["out"])
        if w in (1, 2):
            return module.gemv_fp8(s["x"], s["codes"], s["scales"], s["out"])
        if w == 3:
            return module.gemv_bf16(s["x"], s["w"], s["out"])
        return module.gemv_int4(s["x"], s["codes"], s["scales"], s["zeros"], s["out"], s["group"])

    def _reference(self, s):
        w = s["weights"]
        if w == 0:
            return GemvNvfp4()._reference(s)
        if w == 1:
            return GemvFp8()._reference(s)
        if w == 2:
            codes, exps = s["codes"], s["scales"]
            return _chunked(s["x"], lambda lo, hi: codes[lo:hi].view(torch.float8_e4m3fn).double() *
                            torch.exp2(exps[lo:hi].double() - 127).repeat_interleave(32, 1), codes.shape[0])
        if w == 3:
            return GemvBf16()._reference(s)
        return _chunked(s["x"], lambda lo, hi: int4_values(s["codes"], s["scales"], s["zeros"], s["group"], lo, hi),
                        s["codes"].shape[0])

    def yardstick(self, s):
        w = s["weights"]
        if w == 0:
            return _w4a16_yardstick(self._matrices, s)
        if w == 1:
            return _fp8_yardstick(self._matrices, s)
        if w == 2:
            codes, exps, x = s["codes"], s["scales"], s["x"]
            n, k = codes.shape
            if ("mxwide", n, k) not in self._matrices:
                self._matrices[("mxwide", n, k)] = (codes.view(torch.float8_e4m3fn).float() *
                                                    torch.exp2(exps.float() - 127).repeat_interleave(32, 1)
                                                    ).to(torch.bfloat16)
            wide = self._matrices[("mxwide", n, k)]
            return lambda: torch.matmul(x, wide.t())
        if w == 3:
            weight, x = s["w"], s["x"]
            return lambda: torch.matmul(x, weight.t())
        return int4_yardstick(self._matrices, s["x"], s["codes"], s["scales"], s["zeros"], s["group"])

    def bytes(self, shape, rows):
        w, n, k = int(shape["weights"]), int(shape["N"]), int(shape["K"])
        group = int(shape.get("group") or INT4_GROUP)
        weights = {0: n * k // 2 + n * k // 16 + n * 4, 1: n * k + n * 4, 2: n * k + n * k // 32, 3: n * k * 2,
                   4: n * k // 2 + 3 * n * (k // group)}[w]
        return weights + rows * k * 2 + rows * n * 2


class GemvTriton(GemvCuda):
    """gemv-triton, the format-generic Triton GEMV, on gemv-cuda's matrices of every format (the same references and
    yardsticks): each configuration a multiply path (MUL) and its tile."""
    structure = "gemv-triton"

    def valid(self, knobs):
        """The configurations a sweep times: the rule; fma with the autotuner or 4-32 rows a program, 256 or 512 values
        a step, 4 or 8 warps; dot with 32-128 rows a program, BK 128-512 within a stage's shared memory, 4 or 8 warps,
        1-8 parts of K."""
        mul, bn, bk, warps, stages, split = (knobs[k] for k in ("MUL", "BN", "BK", "WARPS", "STAGES", "SPLIT"))
        if mul == 0:   # the rule, every knob from it
            return bn == bk == warps == stages == split == 0
        if stages or (mul == 1 and split):
            return False
        if mul == 1:   # fma: BN 0 is the autotuner, whose tile and warps are its own
            if bn == 0:
                return bk == 0 and warps == 0
            return 4 <= bn <= 32 and bk in (256, 512) and warps in (4, 8)
        return (bn >= 32 and bk >= 128 and warps in (4, 8) and split >= 1 and bn * bk <= 32768)

    def valid_for(self, knobs, shape):
        """The dot path where the matrix takes it (K a multiple of 128 and of split * BK; BF16 weights: a stage's
        [BN x BK] block within shared memory)."""
        if knobs["MUL"] != 2:
            return True
        k = int(shape["K"])
        if k % 128 or k % (knobs["BK"] * knobs["SPLIT"]):
            return False
        return int(shape["weights"]) != 3 or knobs["BN"] * knobs["BK"] <= 16384

    def run(self, module, s):
        if "partial" not in s:   # once a state (the first, untimed call): the timed calls allocate nothing
            n = s["out"].shape[1]
            s["partial"] = torch.empty(module.dot_partial_elements(16, n, 8), dtype=torch.float32,
                                       device=s["x"].device)
        w = s["weights"]
        if w == 0:
            return module.gemv_any(s["x"], 0, s["codes"], s["out"], scales=s["scales"], factor=s["inverse"],
                                   partial=s["partial"])
        if w == 1:
            return module.gemv_any(s["x"], 1, s["codes"], s["out"], factor=s["scales"], partial=s["partial"])
        if w == 2:
            return module.gemv_any(s["x"], 2, s["codes"], s["out"], scales=s["scales"], partial=s["partial"])
        if w == 3:
            return module.gemv_any(s["x"], 3, s["w"], s["out"], partial=s["partial"])
        return module.gemv_any(s["x"], 4, s["codes"], s["out"], scales=s["scales"], zeros=s["zeros"],
                               group=s["group"], partial=s["partial"])


# ---- rows-int-block and rows-fp8-block: the declared decode rows, on the benches of the kernels that alias them ----

def _bench_of(relative):
    """The sweep bench module of another kernel of the library (its path from the families' folder)."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location("crit_kernel_bench_" + path.parent.name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Declared:
    """decode-gemv's declared rows under the names int-block-experts' and fp8-block-rows' benches call."""

    def __init__(self, module, kind):
        self._m, self._kind = module, kind
        self.row_jobs = module.declared_row_jobs
        self.matrix_table = module.int_block_table if kind == "int" else module.fp8_block_table
        self.rows = module.rows_int_block if kind == "int" else module.rows_fp8_block


_INT_BENCH = _bench_of("experts/int-block-experts/bench.py")
_FP8_BENCH = _bench_of("projections/fp8-block-rows/bench.py")


class RowsIntBlock(_INT_BENCH.Rows):
    """int-block-experts' rows bench on decode-gemv's rows-int-block: the same matrices, activations and yardstick, a
    configuration kept where every output has the untuned build's bits."""
    structure = "rows-int-block"
    kernel = "decode-gemv"

    def _call(self, module, s, out):
        _Declared(module, "int").rows(s["table"], s["jobs"], s["n"], s["values"], s["scales"], _INT_BENCH.BLOCK, out)


class RowsFp8Block(_FP8_BENCH.Rows):
    """fp8-block-rows' rows bench on decode-gemv's rows-fp8-block."""
    structure = "rows-fp8-block"

    def run(self, module, s):
        return super().run(_Declared(module, "fp8"), s)
