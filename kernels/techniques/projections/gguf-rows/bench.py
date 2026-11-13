"""gguf-rows' sweep bench (crit_kernels.sweep) and its reference: the model's GGUF matrices (their N, K and format;
seeded blocks of that layout with finite scales), BF16 activations whose row i is seeded by i alone, and the declared
order of gguf_rows.h computed in torch from Core's host decoders of the stored blocks (crit_core.model.gguf_blocks'
subblocks: each 32-value sub-block's integer codes and FP32 scales, the stored values fsub(fmul(s, q), m) bit for bit,
tests/ checking them against the format's own reference decoder). The check is bit equality."""
import numpy as np
import torch

from crit_core.model.gguf_blocks import (BLOCKS, HAS_MIN, SCALE_BYTES, TWO_MINS, TWO_SCALES,  # noqa: F401
                                         TYPES as FORMATS, stored_values, subblocks)


def quantize_rows(x):
    """gguf_rows.h step 1 in torch FP32: (q int64 [M][J][32], d FP32 [M][J], sum int64 [M][J])."""
    x = x.float().cpu().reshape(x.shape[0], -1, 32)
    d = x.abs().amax(-1) / 127.0
    q = torch.where(d[..., None] == 0, torch.zeros_like(x), torch.round(x / d[..., None]).clamp(-127, 127))
    q = q.to(torch.int64)
    return q, d, q.sum(-1)


def product(raw, fmt, k, x, parts):
    """y [M][N] FP32: gguf_rows.h's declared order exactly, in torch FP32 on the CPU."""
    q, s0, s1, m0, m1 = (torch.from_numpy(a) for a in subblocks(raw, fmt, k, halves=True))
    xq, xd, xs = quantize_rows(x)
    q = q.to(torch.int64)
    lo = torch.einsum("njs,mjs->mnj", q[..., :16], xq[..., :16])
    hi = torch.einsum("njs,mjs->mnj", q[..., 16:], xq[..., 16:])
    d = xd[:, None, :]
    if fmt in TWO_SCALES:
        t = (s0[None] * d) * lo.float() + (s1[None] * d) * hi.float()
    else:
        t = (s0[None] * d) * (lo + hi).float()
    if fmt == 10:   # Q2_K's integer form: fmul(dx, fsub(fmul(dw, isum), fmul(dmin, imin)))
        dw, dmin, sc, mn = q2k_integers(raw, k)
        isum = torch.einsum("njs,mjs->mnj", q * sc, xq)
        first = xq[..., :16].sum(-1)[:, None, :]
        imin = mn[None, ..., 0] * first + mn[None, ..., 1] * (xs[:, None, :] - first)
        t = d * ((dw[None] * isum.float()) - (dmin[None] * imin.float()))
    elif fmt in HAS_MIN:
        t = t - (m0[None] * d) * xs[:, None, :].float()
    j = t.shape[-1]
    out = None
    for p in range(parts):
        part = t[..., j * p // parts:j * (p + 1) // parts]
        steps = -(-part.shape[-1] // 32)
        part = torch.nn.functional.pad(part, (0, 32 * steps - part.shape[-1])).reshape(*part.shape[:-1], steps, 32)
        acc = torch.zeros(part.shape[:-2] + (32,))
        for s in range(steps):
            acc = acc + part[..., s, :]
        lanes = torch.arange(32)
        for off in (16, 8, 4, 2, 1):
            acc = acc + acc[..., lanes ^ off]
        out = acc[..., 0] if out is None else out + acc[..., 0]
    return out


def q2k_integers(raw, k):
    """Q2_K's integer form a sub-block: (d, dmin FP32 [N][J], the 4-bit scales int64 [N][J][32] a value, the 4-bit
    minimums int64 [N][J][2] a half)."""
    n = raw.shape[0]
    b = torch.from_numpy(np.ascontiguousarray(raw).reshape(n, k // 256, 84).astype(np.int64))
    f16 = np.ascontiguousarray(raw).reshape(n, k // 256, 84)
    d = torch.from_numpy(f16[..., 80:82].copy().view(np.float16)[..., 0].astype(np.float32))
    dmin = torch.from_numpy(f16[..., 82:84].copy().view(np.float16)[..., 0].astype(np.float32))
    sc = b[..., :16].reshape(n, -1, 2)                                     # [N][J][half]
    scale = (sc & 15).repeat_interleave(16, dim=-1)                         # [N][J][32]
    return (d.repeat_interleave(8, dim=-1), dmin.repeat_interleave(8, dim=-1), scale, sc >> 4)


def blocks(fmt, n, k, seed, experts=None):
    """Seeded stored blocks uint8 [N][row bytes] (or [experts][N][row bytes]) of format fmt, their FP16 scales finite
    (normal, magnitudes 2^-12 to 2^-4; IQ1_M's spread over its scale words' top nibbles) and every other byte random."""
    values, size = BLOCKS[fmt]
    g = np.random.default_rng(seed)
    shape = (experts or 1, n, k // values, size)
    raw = g.integers(0, 256, size=shape, dtype=np.uint8)
    for at in SCALE_BYTES[fmt]:
        scale = (g.uniform(1, 2, size=shape[:-1]) * 2.0 ** g.integers(-12, -4, size=shape[:-1])).astype(np.float16)
        raw[..., at:at + 2] = scale.view(np.uint8).reshape(*shape[:-1], 2)
    if fmt == 29:   # IQ1_M keeps its FP16 d in the top nibbles of its four scale words
        d = (g.uniform(1, 2, size=shape[:-1]) * 2.0 ** g.integers(-12, -4, size=shape[:-1])).astype(np.float16)
        bits = d.view(np.uint16)
        for u in range(4):
            nibble = ((bits >> (4 * u)) & 15).astype(np.uint8)
            raw[..., 48 + 2 * u + 1] = (raw[..., 48 + 2 * u + 1] & 15) | (nibble << 4)
    if fmt == 39:   # MXFP4's E8M0 byte: scales 2^-13 to 2^-3 (the values doubled)
        raw[..., 0] = g.integers(115, 126, size=shape[:-1], dtype=np.uint8)
    raw = raw.reshape(experts or 1, n, -1)
    return raw if experts else raw[0]


def activations(rows, k, device, dtype=torch.bfloat16):
    return torch.stack([torch.randn(k, generator=torch.Generator().manual_seed(1000 + i)) for i in range(rows)]).to(
        device, dtype)


class Gemv:
    structure = "gemv"

    def __init__(self):
        self._matrices, self._references = {}, {}

    def shapes(self, model):
        return [m for m in model.matrices() if m["format"].upper() in FORMATS]

    def matrix(self, package, shape, device):
        fmt = FORMATS[shape["format"].upper()]
        n, k = int(shape["N"]), int(shape["K"])
        key = (fmt, n, k)
        if key not in self._matrices:
            raw = blocks(fmt, n, k, seed=n * 31 + k + fmt)
            self._matrices[key] = (raw, package.GgufMatrix(torch.from_numpy(raw).to(device), fmt, k))
        return self._matrices[key]

    def inputs(self, shape, rows, device):
        return {"shape": shape, "x": activations(rows, int(shape["K"]), device),
                "key": (shape["format"].upper(), int(shape["N"]), int(shape["K"]), rows)}

    def run(self, module, s):
        s["raw"], s["w"] = self.matrix(module, s["shape"], s["x"].device)
        s["module"] = module
        return module.gemv(s["x"], s["w"])

    def reference(self, s):
        if s["key"] not in self._references:
            if "w" not in s:
                s["raw"], s["w"] = self.matrix(s["module"], s["shape"], s["x"].device)
            w = s["w"]
            y = product(s["raw"], w.type, w.k, s["x"], s["module"].splits(w))
            self._references[s["key"]] = y.to(s["x"].dtype).to(s["x"].device)
        return self._references[s["key"]]

    def check(self, got, want, s):
        bad = int((got.view(torch.int16) != want.view(torch.int16)).sum())
        return None if bad == 0 else f"{bad} values differ from the declared order's bits"

    def first_row(self, out):
        return out[0]

    def bytes(self, shape, rows):
        fmt = FORMATS[shape["format"].upper()]
        values, size = BLOCKS[fmt]
        n, k = int(shape["N"]), int(shape["K"])
        return n * (k // values) * size + rows * (k + n) * 2


class Gemm(Gemv):
    """The prompt rows (structure gemm): the same declared order and bit equality. The reference is the untuned build's
    gemv, 8 rows a call (the declared order bit for bit, tests/ checking it against product): product's host
    arithmetic over thousands of rows would not fit in memory."""
    structure = "gemm"

    def valid(self, knobs):
        # 2 to 8 warps a CTA: one warp leaves the tile too small, 16 cannot hold a CTA's registers
        return 2 <= knobs["WN"] * knobs["WT"] <= 8

    def run(self, module, s):
        s["raw"], s["w"] = self.matrix(module, s["shape"], s["x"].device)
        s["module"] = module
        return module.gemm(s["x"], s["w"])

    def reference(self, s):
        default = getattr(self, "default", None)
        if default is None:
            return super().reference(s)
        if s["key"] not in self._references:
            _, w = self.matrix(default, s["shape"], s["x"].device)
            x = s["x"]
            self._references[s["key"]] = torch.cat([default.gemv(x[i:i + 8], w) for i in range(0, x.shape[0], 8)])
        return self._references[s["key"]]
