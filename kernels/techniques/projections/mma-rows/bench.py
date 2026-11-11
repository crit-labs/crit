"""mma-rows' sweep benches (crit_kernels.sweep): the model's matrices in their format (NVFP4, FP8 with row scales,
BF16) with random codes and scales of the stored layout, BF16 activations whose row i is seeded by i alone, the plan
from the build's plan() (the plan a plugin takes), and an FP64 reference of the stored values. INT4 with a zero point a group (AWQ and GPTQ checkpoints, int4_zp, as
decode-gemv's repack_awq / repack_gptq leave them) is found by the checkpoint's qweight tensors."""
import math

import torch

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
WEIGHTS = {"nvfp4": 0, "fp8": 1, "bf16": 3, "int4_zp": 4}
INT4_GROUP = 128   # a shape's group where it names none (AWQ's and GPTQ's usual)
INPUT_SCALE = 500.0   # NVFP4 activations' input scale: randn blocks' scales well inside E4M3's range


def _activations(rows, k, device):
    return torch.stack([torch.randn(k, generator=torch.Generator().manual_seed(1000 + i)) for i in range(rows)]).to(
        device, torch.bfloat16)


def _blocked(scales):
    """[rows][blocks] scale bytes in cuBLASLt's 128 x 4 tiled layout (rows padded to 128, blocks to 4)."""
    rows, cols = scales.shape
    rt, ct = math.ceil(rows / 128), math.ceil(cols / 4)
    padded = torch.zeros(rt * 128, ct * 4, dtype=scales.dtype, device=scales.device)
    padded[:rows, :cols] = scales
    tiles = padded.view(rt, 128, ct, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _row_major(blocked, rows, cols):
    rt, ct = math.ceil(rows / 128), math.ceil(cols / 4)
    tiles = blocked.view(-1, 32, 4, 4).transpose(1, 2).reshape(rt, ct, 128, 4).permute(0, 2, 1, 3)
    return tiles.reshape(rt * 128, ct * 4)[:rows, :cols]


def _nvfp4_values(codes, scales, lo, hi):
    """Rows lo..hi of an NVFP4 matrix (codes [N][K / 2], row-major E4M3 scales) in FP64, without the row scale."""
    table = torch.tensor(E2M1, dtype=torch.float64, device=codes.device)
    c = codes[lo:hi].long()
    w = torch.stack((table[c & 15], table[c >> 4]), dim=-1).reshape(c.shape[0], -1)
    return (w.view(c.shape[0], -1, 16) * scales[lo:hi].double()[..., None]).reshape(c.shape[0], -1)


def int4_shapes(model):
    """The model's AWQ or GPTQ matrices (qweight int32: AWQ [K][N / 8], GPTQ [K / 8][N]) as int4_zp keys."""
    quant = model.quantization or {}
    method = str(quant.get("quant_method", "")).lower()
    if method not in ("awq", "gptq") or int(quant.get("bits", 4)) != 4:
        return []
    group = int(quant.get("group_size") or INT4_GROUP)
    found = []
    for name, (dtype, shape) in model.tensors.items():
        if name.endswith(".qweight") and dtype == "I32" and len(shape) == 2:
            n, k = (shape[1] * 8, shape[0]) if method == "awq" else (shape[1], shape[0] * 8)
            if group > 0 and group % 32 == 0 and k % group == 0:
                found.append({"name": name[:-8], "weights": WEIGHTS["int4_zp"], "N": n, "K": k, "group": group})
    return found


def _int4_yardstick(cache, s):
    """torch's W4A16 INT4 group GEMM (torch._weight_int4pack_mm, tinygemm): the same codes repacked once, a BF16 scale
    and offset a group ((code - 8) * scale + offset, offset (8 - zero) * scale). None where torch lacks it or refuses
    the shape."""
    if not hasattr(torch, "_weight_int4pack_mm") or not hasattr(torch, "_convert_weight_to_int4pack"):
        return None
    codes, scales, x = s["stored"][0], s["stored"][1], s["x"]
    zeros, group = s["zeros"], s["group"]
    n, k = codes.shape[0], codes.shape[1] * 2
    if ("int4pack", n, k, group) not in cache:
        try:
            packed = torch._convert_weight_to_int4pack(codes.contiguous(), 8)
            sz = torch.stack((scales.float(), (8 - zeros.float()) * scales.float()), -1)
            sz = sz.transpose(0, 1).contiguous().to(torch.bfloat16)
            torch._weight_int4pack_mm(x, packed, group, sz)
        except (RuntimeError, TypeError):
            return None
        cache[("int4pack", n, k, group)] = (packed, sz)
    packed, sz = cache[("int4pack", n, k, group)]
    return lambda: torch._weight_int4pack_mm(x, packed, group, sz)


def _chunked(x, rows_of, n, chunk=8192):
    """(x @ W.T, |x| @ |W|.T) in FP64, the matrix's rows in chunks (rows_of(lo, hi): W[lo:hi] in FP64)."""
    xd = x.double()
    value = torch.empty(x.shape[0], n, dtype=torch.float64, device=x.device)
    magnitude = torch.empty_like(value)
    for lo in range(0, n, chunk):
        w = rows_of(lo, min(n, lo + chunk))
        value[:, lo:lo + w.shape[0]] = xd @ w.t()
        magnitude[:, lo:lo + w.shape[0]] = xd.abs() @ w.abs().t()
    return value, magnitude


class DecodeTile:
    structure = "decode-tile"
    acts = "bf16"
    blocked = False

    def __init__(self):
        self._matrices, self._references = {}, {}

    def shapes(self, model):
        return [dict(m, weights=WEIGHTS[m["format"]]) for m in model.matrices() if m["format"] in WEIGHTS
                and m["K"] % 32 == 0] + (int4_shapes(model) if self.acts == "bf16" else [])

    def valid(self, knobs):
        return knobs.get("SW", 0) > 0 or not knobs.get("SPF", 0)   # SPF is the staged tile's

    def _format(self, shape):
        if "weights" not in shape:   # a shapes file may give the checkpoint's format instead of its code
            return shape["format"]
        return {code: name for name, code in WEIGHTS.items()}[int(shape["weights"])]

    def _matrix(self, shape, device):
        n, k, fmt = int(shape["N"]), int(shape["K"]), self._format(shape)
        if (fmt, n, k) not in self._matrices:   # one matrix a shape, kept for every configuration the sweep times
            g = torch.Generator(device=device).manual_seed(n * 31 + k + WEIGHTS[fmt])
            if fmt == "int4_zp":
                group = int(shape.get("group") or INT4_GROUP)
                codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8, device=device)
                scales = ((torch.rand(n, k // group, generator=g, device=device) + 0.5) / 64).to(torch.float16)
                zeros = torch.randint(0, 17, (n, k // group), generator=g, dtype=torch.uint8, device=device)
                stored = (codes, scales, None, zeros, group)

                def rows_of(lo, hi):
                    c = codes[lo:hi].long()
                    q = torch.stack((c & 15, c >> 4), dim=-1).reshape(c.shape[0], -1).double()
                    return ((q - zeros[lo:hi].double().repeat_interleave(group, 1)) *
                            scales[lo:hi].double().repeat_interleave(group, 1))
            elif fmt == "nvfp4":
                codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8, device=device)
                scales = (torch.rand(n, k // 16, generator=g, device=device) * 0.4 + 0.1).to(torch.float8_e4m3fn)
                inverse = (torch.rand(n, generator=g, device=device) + 0.5) / 64
                stored = (codes, (_blocked(scales) if self.blocked else scales).view(torch.uint8), inverse)
                rows_of = lambda lo, hi: _nvfp4_values(codes, scales, lo, hi) * inverse[lo:hi].double()[:, None]
            elif fmt == "fp8":
                codes = torch.randint(0, 0x7E, (n, k), generator=g, dtype=torch.uint8, device=device)
                codes |= torch.randint(0, 2, (n, k), generator=g, dtype=torch.uint8, device=device) << 7
                scale = torch.rand(n, generator=g, device=device) / 64 + 1e-3
                stored = (codes, None, scale)
                rows_of = lambda lo, hi: codes[lo:hi].view(torch.float8_e4m3fn).double() * scale[lo:hi].double()[:, None]
            else:
                w = (torch.randn(n, k, generator=g, device=device) / 16).to(torch.bfloat16)
                stored = (w, None, None)
                rows_of = lambda lo, hi: w[lo:hi].double()
            self._matrices[(fmt, n, k)] = (stored, rows_of)
        return self._matrices[(fmt, n, k)]

    def inputs(self, shape, rows, device):
        n, k, fmt = int(shape["N"]), int(shape["K"]), self._format(shape)
        stored, rows_of = self._matrix(shape, device)
        extra = stored[3:] if len(stored) > 3 else (None, 0)
        return {"x": _activations(rows, k, device), "stored": stored[:3], "rows_of": rows_of, "weights": fmt,
                "zeros": extra[0], "group": extra[1],
                "out": torch.empty(rows, n, device=device, dtype=torch.bfloat16), "key": (fmt, n, k, rows)}

    def _buffers(self, module, s):
        rows, k = s["x"].shape
        n = s["out"].shape[1]
        groups, warps, splits = s["plan"]
        s["scratch"] = torch.empty(max(16, module.scratch_bytes(rows, k, self.acts)), dtype=torch.uint8,
                                   device=s["x"].device)
        s["partial"] = torch.empty(max(1, module.partial_elements(rows, n, splits)), dtype=torch.float32,
                                   device=s["x"].device)
        s["counters"] = torch.zeros(module.row_tiles(n, 1, warps), dtype=torch.int32, device=s["x"].device)

    def run(self, module, s):
        w, scales, row_scale = s["stored"]
        if "plan" not in s:   # once a state (the first, untimed call): the timed calls launch the kernels alone
            s["plan"] = module.plan(w.shape[0], s["x"].shape[1], s["weights"], s["x"].device, acts=self.acts)
            self._buffers(module, s)
        groups, warps, splits = s["plan"]
        return module.mm_rows(s["x"], w, s["out"], s["weights"], scales, row_scale, self.acts, INPUT_SCALE,
                              self.blocked, groups, warps, splits, s["scratch"], s["partial"], s["counters"],
                              s["zeros"], s["group"])

    def yardstick(self, s):
        """The best library at the matrix's W*A16 arithmetic. NVFP4: flashinfer's BF16 x NVFP4 GEMM (mm_bf16_fp4,
        backend cute-dsl-native on sm_120) on the stored codes, the block scales put in its 128 x 4 layout once, the
        per-row inverse scale left out (it takes one alpha a matrix); None where flashinfer is not installed. BF16:
        cuBLAS (torch.matmul) on the same matrix. FP8: no library takes FP8 weights against BF16 activations
        (cuBLASLt's FP8 GEMMs quantize the activations too), so cuBLAS on the values times their row scales widened
        to BF16 once, what an engine without the tile runs: it reads twice the weight bytes."""
        fmt, n, k = s["weights"], s["out"].shape[1], s["x"].shape[1]
        if fmt == "int4_zp":
            return _int4_yardstick(self._matrices, s)
        if fmt == "nvfp4":
            try:
                import flashinfer
            except ImportError:
                return None
            codes, scales, _ = s["stored"]
            if ("blocked", n, k) not in self._matrices:
                self._matrices[("blocked", n, k)] = _blocked(scales.view(torch.float8_e4m3fn))
            sb, x = self._matrices[("blocked", n, k)], s["x"]
            return lambda: flashinfer.mm_bf16_fp4(x, codes, sb, None, backend="cute-dsl-native")
        if ("wide", fmt, n, k) not in self._matrices:
            wide = torch.empty(n, k, dtype=torch.bfloat16, device=s["x"].device)
            for lo in range(0, n, 8192):
                wide[lo:lo + 8192] = s["rows_of"](lo, min(n, lo + 8192)).to(torch.bfloat16)
            self._matrices[("wide", fmt, n, k)] = wide
        wide, x = self._matrices[("wide", fmt, n, k)], s["x"]
        return lambda: torch.matmul(x, wide.t())

    def reference(self, s):
        if s["key"] not in self._references:
            self._references[s["key"]] = _chunked(self._reference_input(s), s["rows_of"], s["out"].shape[1])
        return self._references[s["key"]]

    def _reference_input(self, s):
        return s["x"]

    def check(self, got, want, s):
        value, magnitude = want
        error = (got.double() - value).abs()
        # FP32 sums of exact products on the MMA (a relative 2e-5 of the magnitudes at most), one BF16 rounding
        bound = magnitude * 2e-5 + value.abs() * 2.0 ** -8 + 1e-6
        bad = int((error > bound).sum())
        return None if bad == 0 else f"{bad} outputs beyond FP32 accumulation and one BF16 rounding"

    def first_row(self, out):
        return out[0]

    def bytes(self, shape, rows):
        n, k, fmt = int(shape["N"]), int(shape["K"]), self._format(shape)
        group = int(shape.get("group") or INT4_GROUP)
        weights = {"nvfp4": n * k // 2 + n * k // 16 + n * 4, "fp8": n * k + n * 4, "bf16": n * k * 2,
                   "int4_zp": n * k // 2 + 3 * n * (k // group)}[fmt]
        return weights + rows * k * 2 + rows * n * 2


class Fp4Tile(DecodeTile):
    """NVFP4 matrices (K a multiple of 128) with blocked scales on the FP4 MMA, the activations quantized by its
    pre-pass; the reference multiplies the quantized activations (the quantize_nvfp4 op's codes and scales) in FP64.
    Devices without the MMA (fp4_mma() false) drop every configuration."""
    structure = "fp4-tile"
    acts = "nvfp4_mma"
    blocked = True

    def shapes(self, model):
        return [{"name": m["name"], "N": m["N"], "K": m["K"], "weights": 0} for m in model.matrices()
                if m["format"] == "nvfp4" and m["K"] % 128 == 0]

    def _format(self, shape):
        return "nvfp4"

    def _reference_input(self, s):
        x, module = s["x"], s["module"]
        rows, k = x.shape
        codes = torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device)
        scales = torch.zeros(math.ceil(rows / 128) * 128 * math.ceil(k / 64) * 4, dtype=torch.uint8, device=x.device)
        module.quantize_nvfp4(x, INPUT_SCALE, codes, scales)
        values = _nvfp4_values(codes, _row_major(scales, rows, k // 16).view(torch.float8_e4m3fn), 0, rows)
        # out = (sum * row_scale) * fp32(1 / input_scale): the reference takes the activations' values over the scale
        return values * float(torch.tensor(1.0 / INPUT_SCALE, dtype=torch.float32))

    def run(self, module, s):
        s["module"] = module
        return super().run(module, s)

    def yardstick(self, s):
        """cuBLASLt's block-scaled NVFP4 GEMM (torch._scaled_mm on float4_e2m1fn_x2 with UE4M3 scales in its blocked
        layout), the same W4A4 arithmetic on the same weight bytes. The activations are quantized once outside the
        timed call (random codes and scales of the layout: the arithmetic does not depend on them), so the library's
        time leaves out the pre-pass the tile's includes (about 1 us); the per-row inverse scale is left out too."""
        w, scales, _ = s["stored"]
        rows, k = s["x"].shape
        key = ("acts", rows, k)
        if key not in self._matrices:
            g = torch.Generator(device=s["x"].device).manual_seed(rows * 7 + k)
            codes = torch.randint(0, 256, (rows, k // 2), generator=g, dtype=torch.uint8, device=s["x"].device)
            block = (torch.rand(rows, k // 16, generator=g, device=s["x"].device) * 0.4 + 0.1).to(torch.float8_e4m3fn)
            self._matrices[key] = (codes.view(torch.float4_e2m1fn_x2), _blocked(block))
        a, sa = self._matrices[key]
        b, sb = w.view(torch.float4_e2m1fn_x2), scales.view(torch.float8_e4m3fn)
        return lambda: torch._scaled_mm(a, b.t(), sa, sb, out_dtype=torch.bfloat16)


class PromptTile(DecodeTile):
    """mm_prompt on the decode tile's matrices with the matrix's plan (its splits), 64 to 2,048 rows; the check takes
    the first, a middle and the last row against FP64 (a vocabulary head's FP64 outputs for 2,048 rows would not fit
    the device; the tests check every row against the decode tile bit for bit)."""
    structure = "prompt-tile"

    def _reference_input(self, s):
        return s["x"][self._sampled(s)]

    def _sampled(self, s):
        rows = s["x"].shape[0]
        return sorted({0, rows // 2, rows - 1})

    def check(self, got, want, s):
        return super().check(got[self._sampled(s)], want, s)

    def run(self, module, s):
        w, scales, row_scale = s["stored"]
        if "plan" not in s:
            s["plan"] = module.plan(w.shape[0], s["x"].shape[1], s["weights"], s["x"].device)
        return module.mm_prompt(s["x"], w, s["out"], s["weights"], scales, row_scale, "bf16", INPUT_SCALE, False,
                                s["plan"][2], None, s["zeros"], s["group"])
