"""fp8-block-rows' sweep benches (crit_kernels.sweep): the model's FP8 matrices as MXFP8-scaled E4M3 codes (E8M0 bytes
per 32 inputs, the scale layout every one of the kernel's layouts shares the arithmetic of), E4M3 activation codes with
power-of-two FP32 group scales whose row i is seeded by i alone (one set a K, sliced for every token count), and an
FP64 reference of the stored values for the first tokens."""
import math

import torch

from crit_kernels.gates import shared_memory_optin

MAX_TOKENS = 4096   # the most tokens the structures' tune_rows ask for
CHECKED = 64        # tokens of each call checked against the FP64 reference
GROUP = 32


class _Fp8Block:
    structure = None

    def __init__(self):
        self._matrices, self._activations, self._tables, self._jobs, self._outs, self._references = {}, {}, {}, {}, {}, {}

    def shapes(self, model):
        return [{"N": m["N"], "K": m["K"]} for m in model.matrices() if m["format"] == "fp8" and m["K"] % 32 == 0]

    def inputs(self, shape, rows, device):
        n, k = int(shape["N"]), int(shape["K"])
        if (n, k) not in self._matrices:   # one matrix a shape, kept for every configuration the sweep times
            g = torch.Generator().manual_seed(n * 31 + k)
            codes = torch.randint(0, 0x7E, (n, k), dtype=torch.uint8, generator=g)
            codes |= (torch.randint(0, 2, (n, k), generator=g) * 0x80).to(torch.uint8)
            scales = torch.randint(120, 134, (n, k // 32), dtype=torch.uint8, generator=g)   # E8M0 bytes
            self._matrices[(n, k)] = (codes.to(device), scales.to(device))
        if k not in self._activations:
            codes, scales = [], []
            for i in range(MAX_TOKENS):
                g = torch.Generator().manual_seed(1000 + i)
                c = torch.randint(0, 0x7E, (k,), dtype=torch.uint8, generator=g)
                codes.append(c | (torch.randint(0, 2, (k,), generator=g) * 0x80).to(torch.uint8))
                scales.append(torch.pow(2.0, torch.randint(-12, -4, (k // GROUP,), generator=g).float()))
            self._activations[k] = (torch.stack(codes).to(device), torch.stack(scales).to(device))
        x, xs = self._activations[k]
        return {"key": (n, k), "rows": int(rows), "x": x[:rows], "xs": xs[:rows], "device": device}

    def _table(self, module, s):
        if s["key"] not in self._tables:   # holds the matrix's pointers; the matrix stays in self._matrices
            codes, scales = self._matrices[s["key"]]
            self._tables[s["key"]] = module.matrix_table([(codes, scales, 1, 32, 0.5)])
        return self._tables[s["key"]]

    def _out(self, s):
        if (s["key"], s["rows"]) not in self._outs:
            self._outs[(s["key"], s["rows"])] = torch.zeros(s["rows"], s["key"][0], device=s["x"].device)
        return self._outs[(s["key"], s["rows"])]

    def yardstick(self, s):
        """cuBLASLt's MXFP8 GEMM (torch._scaled_mm on E4M3 codes with E8M0 scales per 32 in its blocked layout, FP32
        output), the same arithmetic on the same bytes: the activations' power-of-two FP32 group scales are E8M0 bytes
        exactly (converted once outside the timed call), the weights' E8M0 bytes are the stored ones; the global 0.5
        is left out (a power of two in the epilogue)."""
        (n, k), rows = s["key"], s["rows"]
        if ("library", n, k) not in self._tables:
            codes, scales = self._matrices[(n, k)]
            self._tables[("library", n, k)] = (codes.view(torch.float8_e4m3fn), _blocked(scales))
        if ("library", k, rows) not in self._outs:
            exponents = (torch.log2(s["xs"]).round() + 127).to(torch.uint8)
            self._outs[("library", k, rows)] = (s["x"].view(torch.float8_e4m3fn), _blocked(exponents))
        (b, sb), (a, sa) = self._tables[("library", n, k)], self._outs[("library", k, rows)]
        return lambda: torch._scaled_mm(a, b.t(), sa, sb, out_dtype=torch.float32)

    def reference(self, s):
        if s["key"] not in self._references:   # the first CHECKED tokens, the same for every token count
            codes, scales = self._matrices[s["key"]]
            x, xs = self._activations[s["key"][1]]
            w = codes.view(torch.float8_e4m3fn).double() * torch.pow(2.0, scales.double() - 127).repeat_interleave(
                32, dim=1)
            a = x[:CHECKED].view(torch.float8_e4m3fn).double() * xs[:CHECKED].double().repeat_interleave(GROUP, dim=1)
            self._references[s["key"]] = (a @ w.t() * 0.5, a.abs() @ w.abs().t() * 0.5)
        return self._references[s["key"]]

    def check(self, got, want, s):
        value, magnitude = want
        rows = min(s["rows"], CHECKED)
        error = (got[:rows].double() - value[:rows]).abs()
        # FP32 block sums and FMAs in the declared order: within 2e-7 of the products' magnitude on the targets measured
        bad = int((error > magnitude[:rows] * 2e-6 + 1e-30).sum())
        return None if bad == 0 else f"{bad} outputs beyond FP32 accumulation of the products"

    def first_row(self, out):
        return out[0]

    def bytes(self, shape, rows):
        n, k = int(shape["N"]), int(shape["K"])
        return n * k + n * k // 32 + rows * k + rows * k // GROUP * 4 + rows * n * 4


def _blocked(scales):
    """[rows][blocks] E8M0 bytes in cuBLASLt's 128 x 4 tiled layout (rows padded to 128, blocks to 4), as
    float8_e8m0fnu."""
    rows, cols = scales.shape
    rt, ct = math.ceil(rows / 128), math.ceil(cols / 4)
    padded = torch.full((rt * 128, ct * 4), 127, dtype=torch.uint8, device=scales.device)
    padded[:rows, :cols] = scales
    tiles = padded.view(rt, 128, ct, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).view(torch.float8_e8m0fnu)


class Rows(_Fp8Block):
    structure = "rows"

    def run(self, module, s):
        table, out = self._table(module, s), self._out(s)
        jk = ("rows", s["key"], s["rows"])
        if jk not in self._jobs:
            self._jobs[jk] = module.row_jobs([(0, list(range(s["rows"])), list(range(s["rows"])))], s["x"].device)
        module.rows(table, self._jobs[jk], s["key"][0], s["x"], s["xs"], GROUP, out)
        return out


class Gemm(_Fp8Block):
    structure = "gemm"

    def run(self, module, s):
        table, out = self._table(module, s), self._out(s)
        jk = ("gemm", s["key"], s["rows"])
        if jk not in self._jobs:
            self._jobs[jk] = module.gemm_jobs([(0, 0, s["rows"])], s["x"].device)
        module.gemm(table, self._jobs[jk], s["key"][0], s["rows"], s["x"], s["xs"], GROUP, out)
        return out


class GemmStaged(Gemm):
    structure = "gemm-staged"

    def valid(self, knobs):
        """A warp tile of whole 16 x 8 MMA tiles, at least 2 warps, a CTA's registers within an SM's (an output holds
        GROUP running lanes and the partial sums still needed: 5 registers, 6 with GROUP 4; about 48 more a thread),
        and the ring within a CTA's shared memory (the GPU's opt-in limit)."""
        bm, bn, wm, wn, group = knobs["BM"], knobs["BN"], knobs["WM"], knobs["WN"], knobs["GROUP"]
        if bm % (16 * wm) or bn % (8 * wn) or wm * wn < 2:
            return False
        outputs = 4 * (bm // wm // 16) * (bn // wn // 8)
        stage = group * knobs["RUN"] * ((bm + bn) * 36 + 4 * bm)
        regs = (outputs * (6 if group == 4 else 5) + 48 + 7) // 8 * 8
        return regs <= 208 and 32 * wm * wn * regs <= 65536 and knobs["STAGES"] * stage <= shared_memory_optin()

    def check(self, got, want, s):
        """Within FP32 accumulation of the FP64 reference, and the bits of the untuned build's `gemm` (the declared
        order's) for every output."""
        bad = super().check(got, want, s)
        if bad:
            return bad
        key = ("bits", s["key"], s["rows"])
        if key not in self._references:
            bits = torch.zeros_like(got)
            self._gemm(self.default, s, bits)
            self._references[key] = bits
        differ = int((got.view(torch.int32) != self._references[key].view(torch.int32)).sum())
        return None if differ == 0 else f"{differ} outputs differ from gemm's bits (the declared order)"

    def _gemm(self, module, s, out):
        jk = ("gemm", s["key"], s["rows"])
        if jk not in self._jobs:
            self._jobs[jk] = module.gemm_jobs([(0, 0, s["rows"])], s["x"].device)
        module.gemm(self._table(module, s), self._jobs[jk], s["key"][0], s["rows"], s["x"], s["xs"], GROUP, out)

    def run(self, module, s):
        table, out = self._table(module, s), self._out(s)
        jk = ("gemm", s["key"], s["rows"])
        if jk not in self._jobs:
            self._jobs[jk] = module.gemm_jobs([(0, 0, s["rows"])], s["x"].device)
        module.gemm_staged(table, self._jobs[jk], s["key"][0], s["rows"], s["x"], s["xs"], GROUP, out)
        return out


class GemmNative(Gemm):
    """The prompt GEMM accumulated in the tensor cores: not the declared order, so checked against the FP64 reference
    only (the base check: within FP32 accumulation of the products' magnitude), not against gemm's bits. The bench's
    activation scales are powers of two (MXFP8's), so the call says mx and an sm_120a GPU runs the block-scaled MMAs."""
    structure = "gemm-native"

    def valid(self, knobs):
        """Warp tiles of whole 16 x 16 tiles with at most 32 accumulator tiles (128 FP32 registers) a thread, 4 or 8
        warps (a multiple of BK / 2 threads: the tile copies' rows), and the ring within a CTA's shared memory (the GPU's opt-in limit)."""
        tm, tn, wm, wn, bk = knobs["TM"], knobs["TN"], knobs["WM"], knobs["WN"], knobs["BK"]
        if tm % (16 * wm) or tn % (16 * wn) or (tm // wm // 16) * (tn // wn // 8) > 32 or not 4 <= wm * wn <= 8:
            return False
        if (32 * wm * wn) % (bk // 2):
            return False
        kb = bk // 32
        scales = max(tm * kb * 4 + tn * 4, (tm + tn) * kb * 4)
        stage = (tm * bk + tn * bk + scales + 127) // 128 * 128
        return knobs["STAGES"] * stage <= shared_memory_optin()

    def check(self, got, want, s):
        """Within 1e-5 of the products' magnitude of the FP64 reference: the block-scaled MMA's own sum of a block is
        not FP32 round to nearest (up to 3.3e-6 of the magnitude measured on an RTX 5090 at K = 2560)."""
        value, magnitude = want
        rows = min(s["rows"], CHECKED)
        error = (got[:rows].double() - value[:rows]).abs()
        bad = int((error > magnitude[:rows] * 1e-5 + 1e-30).sum())
        return None if bad == 0 else f"{bad} outputs beyond 1e-5 of the products' magnitude"

    def run(self, module, s):
        table, out = self._table(module, s), self._out(s)
        jk = ("gemm", s["key"], s["rows"])
        if jk not in self._jobs:
            self._jobs[jk] = module.gemm_jobs([(0, 0, s["rows"])], s["x"].device)
        module.gemm_native(table, self._jobs[jk], s["key"][0], s["rows"], s["x"], s["xs"], GROUP, out, mx=True)
        return out


# The projections an engine stacks into one prompt GEMM (rows summed, K shared): a checkpoint's separate q, k, v (and
# a gated-delta layer's q|k|v and z, an MLP's gate and up) are one matrix of the summed rows in the engine.
_STACKS = (("q_proj", "k_proj", "v_proj"), ("in_proj_qkv", "in_proj_z"), ("gate_proj", "up_proj"))
CHANNEL_MAX_TOKENS = 8192   # the most tokens gemm-native-channel's tune_rows ask for


class GemmNativeChannel:
    """The prompt GEMM of an FP8 checkpoint's per-channel matrices (one FP32 scale a weight row) against activations
    with one FP32 scale a token (amax / 448, not a power of two), BF16 out, on the shapes an engine multiplies: the
    model's FP8 matrices with the projections it stacks summed (_STACKS). Checked against the FP64 product: within
    1e-5 of the products' magnitude (the tensor cores' own block sums) plus BF16's rounding of the value. Yardstick:
    cuBLASLt's rowwise-scaled FP8 GEMM (torch._scaled_mm, scale_a [tokens, 1], scale_b [1, rows], BF16 out)."""
    structure = "gemm-native-channel"

    def __init__(self):
        self._matrices, self._activations, self._tables, self._jobs, self._outs, self._references = {}, {}, {}, {}, {}, {}

    def shapes(self, model):
        found, seen = {}, set()
        for m in model.matrices():
            if m["format"] != "fp8" or m["K"] % 32:
                continue
            base, _, last = m["name"].rpartition(".")
            stack = next((s for s in _STACKS if last in s), None)
            key = (base, stack) if stack else (m["name"], None)
            n, k = found.get(key, (0, m["K"]))
            found[key] = (n + m["N"], k)
        shapes = []
        for n, k in found.values():
            if (n, k) not in seen:
                seen.add((n, k))
                shapes.append({"N": n, "K": k})
        return sorted(shapes, key=lambda s: (s["K"], s["N"]))

    def inputs(self, shape, rows, device):
        n, k = int(shape["N"]), int(shape["K"])
        if (n, k) not in self._matrices:
            g = torch.Generator().manual_seed(n * 31 + k)
            codes = torch.randint(0, 0x7E, (n, k), dtype=torch.uint8, generator=g)
            codes |= (torch.randint(0, 2, (n, k), generator=g) * 0x80).to(torch.uint8)
            scales = (torch.rand(n, 1, generator=g) * 0.02 + 0.001).float()
            self._matrices[(n, k)] = (codes.to(device), scales.to(device))
        if k not in self._activations:   # one set a K, sliced for every token count: row i the same in every slice
            g = torch.Generator().manual_seed(1000 + k)
            codes = torch.randint(0, 0x7E, (CHANNEL_MAX_TOKENS, k), dtype=torch.uint8, generator=g)
            codes |= (torch.randint(0, 2, (CHANNEL_MAX_TOKENS, k), generator=g) * 0x80).to(torch.uint8)
            scales = (torch.rand(CHANNEL_MAX_TOKENS, 1, generator=g) * 0.05 + 0.002).float()
            self._activations[k] = (codes.to(device), scales.to(device))
        x, xs = self._activations[k]
        return {"key": (n, k), "rows": int(rows), "x": x[:rows], "xs": xs[:rows], "device": device}

    def _table(self, module, s):
        if s["key"] not in self._tables:
            codes, scales = self._matrices[s["key"]]
            self._tables[s["key"]] = module.matrix_table([(codes, scales, 1, s["key"][1], 1.0)])
        return self._tables[s["key"]]

    def _out(self, s, tag="crit"):
        key = (tag, s["key"], s["rows"])
        if key not in self._outs:
            self._outs.clear() if len(self._outs) > 4 else None   # outputs of 248320-row heads are GBs: keep few
            self._outs[key] = torch.zeros(s["rows"], s["key"][0], dtype=torch.bfloat16, device=s["x"].device)
        return self._outs[key]

    def run(self, module, s):
        table, out = self._table(module, s), self._out(s)
        jk = (s["key"], s["rows"])
        if jk not in self._jobs:
            self._jobs[jk] = module.gemm_jobs([(0, 0, s["rows"])], s["x"].device)
        module.gemm_native_channel(table, self._jobs[jk], s["key"][0], s["rows"], s["x"], s["xs"], out)
        return out

    def yardstick(self, s):
        """cuBLASLt's FP8 GEMM with row-wise scales (torch._scaled_mm, E4M3 operands, scale_a [tokens, 1] and
        scale_b [1, rows] FP32, BF16 output): the same arithmetic on the same bytes."""
        (n, k), rows = s["key"], s["rows"]
        codes, scales = self._matrices[(n, k)]
        b, sb = codes.view(torch.float8_e4m3fn), scales.reshape(1, n)
        a, sa = s["x"].view(torch.float8_e4m3fn), s["xs"].contiguous()
        out = self._out(s, "library")
        return lambda: torch._scaled_mm(a, b.t(), sa, sb, out_dtype=torch.bfloat16, out=out)

    def reference(self, s):
        if s["key"] not in self._references:   # the first CHECKED tokens, the same for every token count
            codes, scales = self._matrices[s["key"]]
            x, xs = self._activations[s["key"][1]]
            a = x[:CHECKED].view(torch.float8_e4m3fn).double() * xs[:CHECKED].double()
            value, magnitude = [], []
            for r0 in range(0, codes.shape[0], 16384):   # the head's 248320 rows in FP64 a slice at a time
                w = codes[r0:r0 + 16384].view(torch.float8_e4m3fn).double() * scales[r0:r0 + 16384].double()
                value.append(a @ w.t())
                magnitude.append(a.abs() @ w.abs().t())
            self._references[s["key"]] = (torch.cat(value, 1), torch.cat(magnitude, 1))
        return self._references[s["key"]]

    def check(self, got, want, s):
        """Within 1e-5 of the products' magnitude (the tensor cores' own block sums are not FP32 round to nearest)
        plus BF16's rounding of the value (half an ulp: 2^-9 of it), and no NaN."""
        value, magnitude = want
        rows = min(s["rows"], CHECKED)
        g = got[:rows].double()
        error = (g - value[:rows]).abs()
        bad = int((error > magnitude[:rows] * 1e-5 + value[:rows].abs() * 2.0 ** -8 + 1e-30).sum()) + \
            int(g.isnan().sum())
        return None if bad == 0 else f"{bad} outputs beyond 1e-5 of the products' magnitude and BF16's rounding"

    def first_row(self, out):
        return out[0]

    def bytes(self, shape, rows):
        n, k = int(shape["N"]), int(shape["K"])
        return n * k + n * 4 + rows * k + rows * 4 + rows * n * 2

    def valid(self, knobs):
        """Warp tiles of whole 16 x 16 tiles with at most 32 accumulator tiles (128 FP32 registers) a thread, 4 or 8
        warps (a multiple of BK / 2 threads: the tile copies' rows), and the ring within a CTA's shared memory (the GPU's opt-in limit)."""
        tm, tn, wm, wn, bk = knobs["TM"], knobs["TN"], knobs["WM"], knobs["WN"], knobs["BK"]
        if tm % (16 * wm) or tn % (16 * wn) or (tm // wm // 16) * (tn // wn // 8) > 32 or not 4 <= wm * wn <= 8:
            return False
        if (32 * wm * wn) % (bk // 2):
            return False
        return knobs["STAGES"] * (tm + tn) * bk <= shared_memory_optin()


class GemmNativeBlocks(GemmNativeChannel):
    """The prompt GEMM of an FP8 block checkpoint (128 x 128 FP32 weight block scales) against activations quantized
    per group of 128 with FP32 scales (amax / 448, not powers of two), BF16 out, through gemm_native(mx=False): the
    blocks GEMM, each scale span's MMAs summed in the tensor cores, one FMA a span. Shapes: the model's FP8 matrices
    with K a multiple of 128. Yardstick: cuBLASLt has no FP8 GEMM with FP32 block scales on sm_120 (torch._scaled_mm's
    1 x 128 / 128 x 128 scaling: CUBLAS_STATUS_NOT_SUPPORTED with torch 2.14.1 on an RTX 5090), so the yardstick is its
    MXFP8 GEMM (E8M0 scales per 32 on both operands, applied by the tensor cores) on the same codes, BF16 out:
    block-scaled FP8 at the library's best rate."""
    structure = "gemm-native-blocks"
    BLOCK = 128

    def shapes(self, model):
        seen, shapes = set(), []
        for m in model.matrices():
            if m["format"] == "fp8" and m["K"] % self.BLOCK == 0 and (m["N"], m["K"]) not in seen:
                seen.add((m["N"], m["K"]))
                shapes.append({"N": m["N"], "K": m["K"]})
        return sorted(shapes, key=lambda s: (s["K"], s["N"]))

    def inputs(self, shape, rows, device):
        n, k = int(shape["N"]), int(shape["K"])
        if (n, k) not in self._matrices:
            g = torch.Generator().manual_seed(n * 31 + k)
            codes = torch.randint(0, 0x7E, (n, k), dtype=torch.uint8, generator=g)
            codes |= (torch.randint(0, 2, (n, k), generator=g) * 0x80).to(torch.uint8)
            scales = (torch.rand(-(-n // self.BLOCK), k // self.BLOCK, generator=g) * 0.02 + 0.001).float()
            self._matrices[(n, k)] = (codes.to(device), scales.to(device))
        if k not in self._activations:
            g = torch.Generator().manual_seed(1000 + k)
            codes = torch.randint(0, 0x7E, (CHANNEL_MAX_TOKENS, k), dtype=torch.uint8, generator=g)
            codes |= (torch.randint(0, 2, (CHANNEL_MAX_TOKENS, k), generator=g) * 0x80).to(torch.uint8)
            scales = (torch.rand(CHANNEL_MAX_TOKENS, k // self.BLOCK, generator=g) * 0.05 + 0.002).float()
            self._activations[k] = (codes.to(device), scales.to(device))
        x, xs = self._activations[k]
        return {"key": (n, k), "rows": int(rows), "x": x[:rows], "xs": xs[:rows], "device": device}

    def _table(self, module, s):
        if s["key"] not in self._tables:
            codes, scales = self._matrices[s["key"]]
            self._tables[s["key"]] = module.matrix_table([(codes, scales, self.BLOCK, self.BLOCK, 1.0)])
        return self._tables[s["key"]]

    def run(self, module, s):
        table, out = self._table(module, s), self._out(s)
        jk = (s["key"], s["rows"])
        if jk not in self._jobs:
            self._jobs[jk] = module.gemm_jobs([(0, 0, s["rows"])], s["x"].device)
        module.gemm_native(table, self._jobs[jk], s["key"][0], s["rows"], s["x"], s["xs"], self.BLOCK, out, mx=False)
        return out

    def yardstick(self, s):
        """cuBLASLt's MXFP8 GEMM (torch._scaled_mm, E4M3 with blocked E8M0 scales per 32 on both operands, BF16 out) on
        the same codes, the E8M0 scales made once outside the timed call."""
        (n, k), rows = s["key"], s["rows"]
        if ("library weights", n, k) not in self._tables:
            codes, _ = self._matrices[(n, k)]
            g = torch.Generator().manual_seed(n + k)
            e8m0 = torch.randint(120, 134, (n, k // 32), dtype=torch.uint8, generator=g).to(codes.device)
            self._tables[("library weights", n, k)] = (codes.view(torch.float8_e4m3fn), _blocked(e8m0))
        if ("library activations", k, rows) not in self._tables:
            g = torch.Generator().manual_seed(k + rows)
            e8m0 = torch.randint(120, 134, (rows, k // 32), dtype=torch.uint8, generator=g).to(s["x"].device)
            self._tables[("library activations", k, rows)] = (s["x"].view(torch.float8_e4m3fn), _blocked(e8m0))
        (b, sb), (a, sa) = self._tables[("library weights", n, k)], self._tables[("library activations", k, rows)]
        out = self._out(s, "library")
        return lambda: torch._scaled_mm(a, b.t(), sa, sb, out_dtype=torch.bfloat16, out=out)

    def reference(self, s):
        if s["key"] not in self._references:
            codes, scales = self._matrices[s["key"]]
            x, xs = self._activations[s["key"][1]]
            a = x[:CHECKED].view(torch.float8_e4m3fn).double() * xs[:CHECKED].double().repeat_interleave(
                self.BLOCK, 1)
            value, magnitude = [], []
            for r0 in range(0, codes.shape[0], 16384):
                w_scales = scales[r0 // self.BLOCK:(r0 + 16384) // self.BLOCK].double()
                w_scales = w_scales.repeat_interleave(self.BLOCK, 0).repeat_interleave(self.BLOCK, 1)
                w = codes[r0:r0 + 16384].view(torch.float8_e4m3fn).double()
                w = w * w_scales[:w.shape[0]]
                value.append(a @ w.t())
                magnitude.append(a.abs() @ w.abs().t())
            self._references[s["key"]] = (torch.cat(value, 1), torch.cat(magnitude, 1))
        return self._references[s["key"]]

    def valid(self, knobs):
        """GemmNativeChannel's with the span sums beside the outputs' (at most 16 accumulator tiles of each a thread)
        and the scale ring (STAGES + 1 slots of FP32 words for every token and row of a stage) within a CTA's shared
        memory (the GPU's opt-in limit)."""
        tm, tn, wm, wn, bk, stages = knobs["TM"], knobs["TN"], knobs["WM"], knobs["WN"], knobs["BK"], knobs["STAGES"]
        if tm % (16 * wm) or tn % (16 * wn) or (tm // wm // 16) * (tn // wn // 8) > 16 or not 4 <= wm * wn <= 8:
            return False
        if (32 * wm * wn) % (bk // 2):
            return False
        return stages * (tm + tn) * bk + (stages + 1) * (tm + tn) * (bk // 32) * 4 <= shared_memory_optin()
