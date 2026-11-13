"""int-block-experts' sweep benches (crit_kernels.sweep): the model's expert matrices (gate/up [2 inter][hidden], down
[hidden][inter]) as random E2M1 codes with E4M3 scales, INT8 activations per 32 whose row i is seeded by i alone. No
knob changes an output's operations or their order, so a configuration must give the untuned build's bits (the
declared order's, which tests/int_block_tests.cu checks against a scalar oracle) for every output."""
import struct

import torch

from crit_kernels.gates import shared_memory_optin

from crit_kernels import build as kbuild

BLOCK = 32
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_MATRIX = struct.Struct("<QQiifi")       # Matrix: codes, scales, rows, cols, post, int4
_ROW_JOB = struct.Struct("<ii8i8i")      # RowJob: matrix, tokens, act_row[8], out_row[8]
_GEMM_JOB = struct.Struct("<iii")        # GemmJob: matrix, first, count


def _device(packed, device):
    return torch.frombuffer(bytearray(packed), dtype=torch.uint8).to(device)


def _widened(codes, scales, post):
    """The matrix's values in BF16 [n][k]: the doubled E2M1 code (an integer, as the kernel takes it) * E4M3 block
    scale * the matrix's post scale."""
    table = torch.tensor(E2M1 + [-v for v in E2M1], device=codes.device)
    n = codes.shape[0]
    w = table[torch.stack((codes & 15, codes >> 4), -1).long()].reshape(n, -1, 16)
    return (w * scales.view(torch.float8_e4m3fn).float()[..., None] * (2 * post)).reshape(n, -1).bfloat16()


def _blocked(scales):
    """E4M3 block scales [rows][k / 16] in cuBLASLt's 128 x 4 tiled layout (rows padded to 128, columns to 4)."""
    rows, cols = scales.shape
    padded = torch.zeros(-(-rows // 128) * 128, -(-cols // 4) * 4, dtype=scales.dtype, device=scales.device)
    padded[:rows, :cols] = scales
    r, c = padded.shape[0] // 128, padded.shape[1] // 4
    blocks = padded.view(r, 128, c, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _nvfp4(x):
    """x [rows][k] quantized to NVFP4 (round to nearest E2M1, E4M3 scale amax / 6 a 16-block): (codes, scales)."""
    rows, k = x.shape
    blocks = x.float().view(rows, k // 16, 16)
    scale = (blocks.abs().amax(-1) / 6).clamp(min=2.0 ** -9).to(torch.float8_e4m3fn)
    q = blocks / scale.float()[..., None]
    mag = torch.bucketize(q.abs(), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device))
    code = (mag + 8 * (q < 0)).to(torch.uint8).view(rows, k // 2, 2)
    return code[..., 0] | (code[..., 1] << 4), scale.view(torch.uint8)


def expert_shapes(model):
    hidden = model.get("hidden_size")
    inter = model.get("moe_intermediate_size", "expert_intermediate_size")
    if not hidden or not inter:
        return []
    out = [{"N": 2 * int(inter), "K": int(hidden)}, {"N": int(hidden), "K": int(inter)}]
    return [s for s in out if s["K"] % 64 == 0]


class _Bench:
    kernel = "int-block-experts"

    def __init__(self):
        self._matrices, self._acts, self._references, self._default, self._library = {}, {}, {}, None, {}

    def shapes(self, model):
        return expert_shapes(model)

    def _matrix(self, n, k, device):
        if (n, k) not in self._matrices:   # one matrix a shape, kept for every configuration the sweep times
            g = torch.Generator(device=device).manual_seed(n * 31 + k)
            codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8, device=device)
            scales = torch.randint(0x20, 0x5F, (n, k // 16), generator=g, device=device).to(torch.uint8)
            table = _device(_MATRIX.pack(codes.data_ptr(), scales.data_ptr(), n, k, 0.5, 0), device)
            self._matrices[(n, k)] = (codes, scales, table)
        return self._matrices[(n, k)]

    def _activations(self, rows, k, device):
        if (rows, k) not in self._acts:
            values, scales = [], []
            for i in range(rows):
                g = torch.Generator().manual_seed(1000 + i)
                values.append(torch.randint(-127, 128, (k,), generator=g, dtype=torch.int8))
                scales.append(torch.rand(k // BLOCK, generator=g) * 0.02 + 0.001)
            self._acts[(rows, k)] = (torch.stack(values).to(device), torch.stack(scales).to(device))
        return self._acts[(rows, k)]

    def inputs(self, shape, rows, device):
        n, k = int(shape["N"]), int(shape["K"])
        codes, scales, table = self._matrix(n, k, device)
        values, act_scales = self._activations(rows, k, device)
        return {"n": n, "k": k, "rows": rows, "table": table, "jobs": self._jobs(rows, device), "values": values,
                "scales": act_scales, "out": torch.zeros(rows, n, device=device), "key": (n, k, rows),
                "matrix": (codes, scales)}

    def reference(self, s):
        if s["key"] not in self._references:
            if self._default is None:
                self._default = kbuild.load(self.kernel)   # the untuned build
            want = torch.zeros_like(s["out"])
            self._call(self._default, s, want)
            self._references[s["key"]] = want
        return self._references[s["key"]]

    def run(self, module, s):
        self._call(module, s, s["out"])
        return s["out"]

    def check(self, got, want, s):
        bad = int((got.view(torch.int32) != want.view(torch.int32)).sum())
        return None if bad == 0 else f"{bad} outputs differ from the untuned build's bits"

    def first_row(self, out):
        return out[0]

    def bytes(self, shape, rows):
        n, k = int(shape["N"]), int(shape["K"])
        return n * k // 2 + n * k // 16 + rows * k + rows * k // BLOCK * 4 + rows * n * 4


class Rows(_Bench):
    structure = "rows"

    def yardstick(self, s):
        """cuBLAS (torch.matmul) BF16 on the weights widened to BF16 and the INT8 activations dequantized to BF16, both
        once: no library multiplies E2M1 weights by INT8 activations in blocks of 32."""
        n, k = s["n"], s["k"]
        if ("wide", n, k) not in self._library:
            self._library[("wide", n, k)] = _widened(*s["matrix"], 0.5)
        x = (s["values"].float() * s["scales"].repeat_interleave(BLOCK, 1)).bfloat16()
        wide = self._library[("wide", n, k)]
        return lambda: torch.matmul(x, wide.t())

    def _jobs(self, rows, device):
        if rows > 8:
            raise ValueError("a decode job has 1 to 8 tokens")
        pad = [0] * (8 - rows)
        return _device(_ROW_JOB.pack(0, rows, *range(rows), *pad, *range(rows), *pad), device)

    def _call(self, module, s, out):
        module.rows(s["table"], s["jobs"], s["n"], s["values"], s["scales"], BLOCK, out)


class Gemm(_Bench):
    structure = "gemm"

    def yardstick(self, s):
        """cuBLASLt's NVFP4 block-scaled GEMM (torch._scaled_mm, FP4 tensor cores) on the stored codes and scales, the
        INT8 activations re-quantized to NVFP4 once (no library takes INT8 activations against 4-bit weights in blocks
        of 32). The bench's post scale 0.5 cancels the kernel's doubled E2M1 integers, so the values are the same."""
        n, k = s["n"], s["k"]
        if ("fp4", n, k) not in self._library:
            codes, scales = s["matrix"]
            self._library[("fp4", n, k)] = (codes.view(torch.float4_e2m1fn_x2),
                                           _blocked(scales).view(torch.float8_e4m3fn))
        w, w_scales = self._library[("fp4", n, k)]
        x, x_scales = _nvfp4(s["values"].float() * s["scales"].repeat_interleave(BLOCK, 1))
        x, x_scales = x.view(torch.float4_e2m1fn_x2), _blocked(x_scales).view(torch.float8_e4m3fn)
        return lambda: torch._scaled_mm(x, w.t(), x_scales, w_scales, out_dtype=torch.bfloat16)

    def _jobs(self, rows, device):
        return _device(_GEMM_JOB.pack(0, 0, rows), device)

    def _call(self, module, s, out):
        module.gemm(s["table"], s["jobs"], s["n"], s["rows"], s["values"], s["scales"], BLOCK, out)


class GemmStaged(Gemm):
    """The staged GEMM: every output must give the untuned build's bits, which are its `gemm` structure's (the same
    declared order) and the scalar oracle's."""
    structure = "gemm-staged"

    def valid(self, knobs):
        """A warp tile of whole 16 x 8 MMA tiles, at least 2 warps, a CTA's registers within an SM's (an output holds
        GROUP running lanes and the partial sums still needed: 5 registers, 6 with GROUP 4; about 48 more a thread),
        and the ring within a CTA's shared memory (the GPU's opt-in limit)."""
        bm, bn, wm, wn, group = knobs["BM"], knobs["BN"], knobs["WM"], knobs["WN"], knobs["GROUP"]
        if bm % (16 * wm) or bn % (8 * wn) or wm * wn < 2:
            return False
        outputs = 4 * (bm // wm // 16) * (bn // wn // 8)
        stage = group * knobs["RUN"] * (32 * bm + 20 * bn)
        regs = (outputs * (6 if group == 4 else 5) + 48 + 7) // 8 * 8
        return regs <= 208 and 32 * wm * wn * regs <= 65536 and knobs["STAGES"] * stage <= shared_memory_optin()

    def reference(self, s):
        if s["key"] not in self._references:
            if self._default is None:
                self._default = kbuild.load(self.kernel)   # the untuned build
            want = torch.zeros_like(s["out"])
            Gemm._call(self, self._default, s, want)      # its register GEMM: the declared order's bits
            self._references[s["key"]] = want
        return self._references[s["key"]]

    def _call(self, module, s, out):
        module.gemm_staged(s["table"], s["jobs"], s["n"], s["rows"], s["values"], s["scales"], BLOCK, out)


class GemmNative:
    """The prompt GEMM accumulated in the tensor cores (gemm_native) on NVFP4 activations, as quantized-projections and
    host-expert-arithmetic call it: the model's expert matrices, or a dense model's NVFP4 MLP matrices (gate|up fused
    [2 inter][hidden], down [hidden][inter]), random E2M1 codes with E4M3 scales; BF16 activations whose row i is
    seeded by i alone, quantized once by the untuned build's quantize_nvfp4 (global 2, so a block's value is the plain
    NVFP4 value E2M1 x E4M3); every scale byte in cuBLASLt's tiled layout on GPUs whose path takes it (tiled_supported)
    and the output in BF16. Not the declared order: checked against an FP64 reference of the stored values (within FP32
    accumulation of the products' magnitude and one BF16 rounding), not against the untuned build's bits."""
    kernel = "int-block-experts"
    structure = "gemm-native"
    CHECKED = 64
    GLOBAL = 2.0

    def __init__(self):
        self._matrices, self._acts, self._references, self._library = {}, {}, {}, {}
        self.default = None

    def shapes(self, model):
        found = expert_shapes(model)
        if found:
            return found
        hidden, inter = model.get("hidden_size"), model.get("intermediate_size")
        if not hidden or not inter or not any(m["format"] == "nvfp4" for m in model.matrices()):
            return []
        out = [{"N": 2 * int(inter), "K": int(hidden)}, {"N": int(hidden), "K": int(inter)}]
        return [s for s in out if s["K"] % 64 == 0]

    def _cc12(self):
        """Whether the GPU is compute capability 12, without initializing CUDA in this process (the sweep forks its
        build workers after valid()): nvidia-smi's compute_cap, else torch's."""
        if getattr(self, "_capability", None) is None:
            import subprocess
            try:
                found = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=30).stdout.split()
                self._capability = int(float(found[0])) if found else torch.cuda.get_device_capability()[0]
            except (OSError, ValueError, subprocess.SubprocessError):
                self._capability = torch.cuda.get_device_capability()[0]
        return self._capability == 12

    def valid(self, knobs):
        """On the warp-specialized path (compute capability 12), with the producer warp beside them: 4 or 8 consumer
        warps (255 or 168 registers a thread) holding their accumulators and two sets of fragments, or 12 (128
        registers, the B fragments streamed) of 32 x 64 on 192 x 128 tiles; as many stages as fit in shared memory
        (99 KB with EPI's staging boxes; 2 with 128-byte stages); EPI 0 or 1 (2 measures as 1), SPLIT 0 or 2 (1 runs
        the same tiles as 0 on these shapes) on tiles of 128 tokens or more; smaller tiles (for small matrices,
        valid_for) with EPI 0 and SPLIT 0. Elsewhere the cp.async path's limits, SPLIT 0 (it does not split K)."""
        tm, tn, wm, wn, bk = knobs["TM"], knobs["TN"], knobs["WM"], knobs["WN"], knobs["BK"]
        if tm % (16 * wm) or tn % (16 * wn) or wm * wn not in (4, 8, 12):
            return False
        mt, nt = tm // wm // 16, tn // wn // 8
        if self._cc12():
            nc = wm * wn
            if tm > 192 or tn > 128 or knobs["EPI"] == 2 or knobs["SPLIT"] == 1 or (tm == 192) != (nc == 12 and tn == 128):
                return False
            if nc == 12:
                regs, cap = 4 * mt * nt + 10 * mt + 12 + 24, 128
            else:
                regs, cap = 4 * mt * nt + 2 * (5 * mt + 3 * nt) + 24, 255 if nc == 4 else 168
            wtn = tn // wn
            box = (16 * wtn * 2 + 1023) // 1024 * 1024
            epi = nc * knobs["EPI"] * box if wtn in (16, 32, 64) else None
            stage = ((tm + tn) * bk + ((tm + 64 + 127) // 128 if tm % 128 else max(1, tm // 128)) * (bk // 32) * 512
                     + (tn + 127) // 128 * (bk // 32) * 512 + 1023) // 1024 * 1024
            if regs > cap or epi is None:
                return False
            fit = min((101376 - 1024 - 256 - epi) // stage, 5)   # int_block.cu's kFit: sm_120's 99 KB, this path's only GPU
            if knobs["STAGES"] != fit or fit < 2:
                return False
            if tm < 128 or tn < 128:
                return knobs["EPI"] == 0 and knobs["SPLIT"] == 0
            return True
        if mt * nt > 32 or (32 * wm * wn) % (bk // 2) or knobs["SPLIT"] != 0 or knobs["STAGES"] not in (3, 4):
            return False
        stage = ((tm + tn) * (bk + bk // 8) + 127) // 128 * 128
        return knobs["STAGES"] * stage <= shared_memory_optin()

    def valid_for(self, knobs, shape):
        """Tiles under 128 tokens or rows only for matrices under 4,096 rows or inputs (an expert's): on larger ones
        they read too many bytes from L2 a product (RTX 5090, 34816 x 5120: 128 x 64 tiles 0.65 of 128 x 128's speed)."""
        small = int(shape["N"]) < 4096 or int(shape["K"]) < 4096
        return small or (knobs["TM"] >= 128 and knobs["TN"] >= 128) or not self._cc12()

    def _matrix(self, n, k, device):
        if (n, k) not in self._matrices:   # one matrix a shape, kept for every configuration the sweep times
            g = torch.Generator(device=device).manual_seed(n * 31 + k)
            codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8, device=device)
            scales = torch.randint(0x20, 0x5F, (n, k // 16), generator=g, device=device).to(torch.uint8)
            tiled = _blocked(scales).view(torch.uint8)
            self._matrices[(n, k)] = (codes, scales, tiled)
        return self._matrices[(n, k)]

    def _activations(self, rows, k, device):
        if (rows, k) not in self._acts:
            x = torch.empty(rows, k, dtype=torch.bfloat16, device=device)
            for i in range(rows):   # row i seeded by i alone
                x[i] = torch.randn(k, generator=torch.Generator(device=device).manual_seed(1000 + i),
                                   device=device).bfloat16()
            if self.default is None:
                self.default = kbuild.load(self.kernel)
            codes, scales = self.default.quantize_nvfp4(x, self.GLOBAL)                      # row major
            _, tiled = self.default.quantize_nvfp4(x, self.GLOBAL, tiled=True)
            self._acts[(rows, k)] = (codes, scales, tiled)
        return self._acts[(rows, k)]

    def inputs(self, shape, rows, device):
        n, k = int(shape["N"]), int(shape["K"])
        codes, scales, tiled = self._matrix(n, k, device)
        x, xs, xt = self._activations(rows, k, device)
        if self.default is None:
            self.default = kbuild.load(self.kernel)
        use_tiled = self.default.tiled_supported() if hasattr(self.default, "tiled_supported") else False
        table = self.default.matrix_table([(codes, tiled if use_tiled else scales, 0.5, False)])
        jobs = _device(_GEMM_JOB.pack(0, 0, rows), device)
        return {"n": n, "k": k, "rows": rows, "table": table, "jobs": jobs, "x": x, "xs": xt if use_tiled else xs,
                "x_rows": xs, "tiled": 3 if use_tiled else 0, "out": torch.zeros(rows, n, dtype=torch.bfloat16,
                                                                              device=device),
                "matrix": (codes, scales, tiled), "key": (n, k, rows), "x_tiled": xt}

    def yardstick(self, s):
        """cuBLASLt's NVFP4 block-scaled GEMM (torch._scaled_mm) on the same codes and scale bytes as the kernel (the
        tiled layout it reads), BF16 output like the kernel's."""
        codes, _, tiled = s["matrix"]
        w, ws = codes.view(torch.float4_e2m1fn_x2), tiled.view(torch.float8_e4m3fn)
        x, xs = s["x"].view(torch.float4_e2m1fn_x2), s["x_tiled"].view(torch.float8_e4m3fn).flatten()
        return lambda: torch._scaled_mm(x, w.t(), xs, ws, out_dtype=torch.bfloat16)

    def reference(self, s):
        key = ("fp64",) + s["key"]
        if key not in self._references:
            rows = min(s["rows"], self.CHECKED)
            codes, scales, _ = s["matrix"]
            w, a = _values64(codes, scales, 1.0), _values64(s["x"][:rows], s["x_rows"][:rows], 0.5)
            self._references[key] = (a @ w.t() * 0.5, a.abs() @ w.abs().t() * 0.5)   # post 0.5
        return self._references[key]

    def run(self, module, s):
        module.gemm_native(s["table"], s["jobs"], s["n"], s["rows"], s["x"], s["xs"], self.GLOBAL, s["out"],
                           tiled=s["tiled"])
        return s["out"]

    def check(self, got, want, s):
        value, magnitude = want
        rows = value.shape[0]
        error = (got[:rows].double() - value).abs()
        bad = int((error > magnitude * 2e-6 + value.abs() * 2.0 ** -8 + 1e-30).sum())
        return None if bad == 0 else f"{bad} outputs beyond FP32 accumulation of the products and a BF16 rounding"

    def first_row(self, out):
        return out[0]

    def bytes(self, shape, rows):
        n, k = int(shape["N"]), int(shape["K"])
        return n * k // 2 + n * k // 16 + rows * k // 2 + rows * k // 16 + rows * n * 2


def _values64(codes, scales, half):
    """[rows][k / 2] E2M1 codes and E4M3 bytes per 16 as FP64 values: 2 x E2M1 x scale x half (the E2M1 x 2 integer)."""
    table = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float64, device=codes.device)
    n = codes.shape[0]
    v = table[torch.stack((codes & 15, codes >> 4), -1).long()].reshape(n, -1, 16)
    return (2 * v * scales.view(torch.float8_e4m3fn).double()[..., None] * half).reshape(n, -1)
