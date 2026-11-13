"""The grouped experts against an FP64 reference of the same decoded weights in every format, and the bits a route
keeps: the same alone and beside other tokens, in decode-width (8) and prompt-width (32) items, through the one-block
plan and the sorted one, on a repeat, in a captured CUDA graph, and in a build with other knobs. The bundles are packed
here in crit's experts:bundles layout (crit's inference/grouped-experts code/bundles.py is the definition)."""
import math

import pytest
import torch

from crit_kernels.build import load

gx = load("grouped-experts")
dev = torch.device("cuda")
FORMATS = {"bf16": 0, "fp8:channel": 1, "fp8:block": 2, "nvfp4": 3, "int4": 4, "mxfp4": 5, "int4_zp": 6}
GROUP = 64   # int4_zp's inputs a scale and zero point (AWQ, GPTQ: 32 to 128)
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_SIGNED = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float64)


def cdiv(a, b):
    return -(-a // b)


def components(fmt, H, I):
    if fmt == "bf16":
        return [("gate_up", 4 * I * H), ("down", 2 * H * I)]
    if fmt.startswith("fp8"):
        gs, ds = ((8 * I, 4 * H) if fmt == "fp8:channel" else   # fp8:block: gate's grid, then up's
                  (4 * 2 * cdiv(I, 128) * cdiv(H, 128), 4 * cdiv(H, 128) * cdiv(I, 128)))
        return [("gate_up", 2 * I * H), ("gate_up_scale", gs), ("down", H * I), ("down_scale", ds)]
    if fmt in ("nvfp4", "int4"):
        return [("gate_up", I * H), ("gate_up_scale", I * H // 8), ("gate_up_inverse", 8 * I), ("down", H * I // 2),
                ("down_scale", H * I // 16), ("down_inverse", 4 * H), ("input", 16)]
    if fmt == "int4_zp":   # FP16 scales, then the zero points (the inverse's place), [rows][K / GROUP] each
        return [("gate_up", I * H), ("gate_up_scale", 4 * I * H // GROUP), ("gate_up_inverse", 2 * I * H // GROUP),
                ("down", H * I // 2), ("down_scale", 2 * H * I // GROUP), ("down_inverse", H * I // GROUP)]
    return [("gate_up", I * H), ("gate_up_scale", I * H // 16), ("down", H * I // 2), ("down_scale", H * I // 32)]


def offsets(fmt, H, I):
    found, at = {}, 0
    for name, size in components(fmt, H, I):
        found[name] = at
        at += cdiv(size, 16) * 16
    return found, at


def layout(fmt, H, I):
    o, size = offsets(fmt, H, I)
    keys = ("gate_up", "gate_up_scale", "gate_up_inverse", "down", "down_scale", "down_inverse", "input")
    return [FORMATS[fmt], size] + [o.get(k, -1) for k in keys] + [H, I] + ([GROUP] if fmt == "int4_zp" else [])


def e4m3_bytes_of(values):
    return values.float().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)


def matrix(fmt, rows, cols, g):
    """(component tensors, decoded FP64 values) of one random matrix."""
    if fmt == "bf16":
        w = (torch.randn(rows, cols, generator=g) * 0.05).bfloat16()
        return {"": w.view(torch.uint8).reshape(-1)}, w.double()
    if fmt.startswith("fp8"):
        codes = e4m3_bytes_of(torch.randn(rows, cols, generator=g) * 64)
        v = codes.view(torch.float8_e4m3fn).double()
        if fmt == "fp8:channel":
            s = torch.rand(rows, generator=g) * 0.01 + 0.0005
            return {"": codes.reshape(-1), "scale": s.float()}, v * s.double()[:, None]
        s = torch.rand(cdiv(rows, 128), cdiv(cols, 128), generator=g) * 0.01 + 0.0005
        full = s.double().repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :cols]
        return {"": codes.reshape(-1), "scale": s.float()}, v * full
    codes = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g)
    lo, hi = (codes & 15).long(), (codes >> 4).long()
    if fmt == "int4_zp":   # (code - zero) * scale, FP16 scales and zeros 0..16 a group
        q = torch.stack([lo, hi], -1).reshape(rows, cols).double()
        s = ((torch.rand(rows, cols // GROUP, generator=g) + 0.5) / 64).half()
        z = torch.randint(0, 17, (rows, cols // GROUP), generator=g).to(torch.uint8)
        decoded = (q - z.double().repeat_interleave(GROUP, 1)) * s.double().repeat_interleave(GROUP, 1)
        return {"": codes.reshape(-1), "scale": s, "inverse": z}, decoded
    if fmt == "int4":
        int4 = torch.tensor([float(c if c < 8 else c - 16) for c in range(16)], dtype=torch.float64)
        vals = torch.stack([int4[lo], int4[hi]], -1).reshape(rows, cols)
    else:
        vals = torch.stack([E2M1_SIGNED[lo], E2M1_SIGNED[hi]], -1).reshape(rows, cols)
    if fmt == "mxfp4":
        s = torch.randint(120, 127, (rows, cols // 32), dtype=torch.uint8, generator=g)
        return {"": codes.reshape(-1), "scale": s}, vals * torch.pow(2.0, s.double() - 127).repeat_interleave(32, 1)
    exp, man = torch.randint(5, 9, (rows, cols // 16), generator=g), torch.randint(0, 8, (rows, cols // 16), generator=g)
    s = (exp * 8 + man).to(torch.uint8)
    inverse = torch.rand(rows, generator=g) * 0.05 + 0.01
    decoded = vals * s.view(torch.float8_e4m3fn).double().repeat_interleave(16, 1) * inverse.double()[:, None]
    return {"": codes.reshape(-1), "scale": s, "inverse": inverse.float()}, decoded


def bundle(fmt, H, I, g, inputs=(0.0, 0.0)):
    """One expert's bundle (uint8 on the device) and its decoded FP64 gate/up and down matrices."""
    o, size = offsets(fmt, H, I)
    out = torch.zeros(size, dtype=torch.uint8)
    if fmt == "fp8:block":   # gate and up each with its own 128 x 128 scale grid
        (gate, gate_values), (up, up_values) = matrix(fmt, I, H, g), matrix(fmt, I, H, g)
        gu = {k: torch.cat([gate[k], up[k]]) for k in gate}
        gu_values = torch.cat([gate_values, up_values])
    else:
        gu, gu_values = matrix(fmt, 2 * I, H, g)
    dn, dn_values = matrix(fmt, H, I, g)
    for prefix, parts in (("gate_up", gu), ("down", dn)):
        for suffix, t in parts.items():
            name = prefix + (f"_{suffix}" if suffix else "")
            raw = t.contiguous().view(torch.uint8).reshape(-1)
            out[o[name]:o[name] + raw.numel()] = raw
    if "input" in o:
        a, b = inputs
        values = torch.tensor([a, b, 1 / a if a else 0.0, 1 / b if b else 0.0], dtype=torch.float32)
        out[o["input"]:o["input"] + 16] = values.view(torch.uint8)
    return out, gu_values, dn_values


def quantize(v, g):
    """crit's declared NVFP4 quantizer as values (e2m1 * decoded / g), FP32 steps, per row of 16-blocks."""
    signed, decoded = quantize_codes(v, g)
    return (signed.double() * decoded.double()[..., None] / g).reshape(v.shape)


def quantize_codes(v, g):
    """(signed E2M1 magnitudes [rows][blocks][16] FP32, decoded block scales [rows][blocks] FP32)."""
    rows, cols = v.shape
    blocks = v.float().reshape(rows, cols // 16, 16)
    amax = blocks.abs().amax(-1)
    wanted = (amax * g) / 6.0
    wanted = torch.where(torch.isfinite(wanted), wanted, torch.full_like(wanted, 448.0))
    decoded = wanted.clamp(max=448.0).to(torch.float8_e4m3fn).float()
    q = (blocks * g) / decoded[..., None].clamp(min=1e-38)
    a = q.abs()
    code = sum((a > b).int() if strict else (a >= b).int()
               for b, strict in ((0.25, 1), (0.75, 0), (1.25, 1), (1.75, 0), (2.5, 1), (3.5, 0), (5.0, 1)))
    mag = torch.tensor(E2M1, dtype=torch.float32)[code.long()]
    signed = torch.where((q < 0) & (code > 0), -mag, mag)
    return torch.where(decoded[..., None] > 0, signed, torch.zeros_like(signed)), decoded


class Experts:
    def __init__(self, fmt, E=6, H=256, I=128, seed=0, a4=False):
        g = torch.Generator().manual_seed(seed)
        self.fmt, self.E, self.H, self.I, self.a4 = fmt, E, H, I, a4
        self.inputs = [(float(torch.rand(1, generator=g)) * 200 + 300, float(torch.rand(1, generator=g)) * 40 + 20)
                       for _ in range(E)]
        made = [bundle(fmt, H, I, g, self.inputs[e]) for e in range(E)]
        self.store = torch.stack([m[0] for m in made]).to(dev)
        self.gate_up, self.down = [m[1] for m in made], [m[2] for m in made]
        self.layout = layout(fmt, H, I)

    def reference(self, x, ids, weights, slots=None):
        x = x.bfloat16().double().cpu()
        out = torch.zeros(x.shape[0], self.H, dtype=torch.float64)
        for t in range(x.shape[0]):
            for j in range(ids.shape[1]):
                e = int(ids[t, j])
                if e < 0 or (slots is not None and int(slots[e]) < 0):
                    continue
                xin = quantize(x[t:t + 1].float(), self.inputs[e][0])[0] if self.a4 else x[t]
                gu = self.gate_up[e] @ xin
                h = torch.nn.functional.silu(gu[:self.I]) * gu[self.I:]
                h = quantize(h[None].float(), self.inputs[e][1])[0] if self.a4 else h.float().bfloat16().double()
                out[t] += float(weights[t, j]) * (self.down[e] @ h)
        return out

    def forward(self, x, ids, weights, slots=None, width=None):
        ws = gx.Workspace(ids.shape[0], ids.shape[1], self.E, self.H, self.I, dev, width)
        return gx.forward(x, ids, weights, self.store, slots, self.layout, ws, self.a4)

    def rows(self, x, ids, slots=None, width=None):
        ws = gx.Workspace(ids.shape[0], ids.shape[1], self.E, self.H, self.I, dev, width)
        return gx.route_outputs(x, ids, self.store, slots, self.layout, ws, self.a4).clone()


def routes(T, top, E, g, holes=True):
    ids = torch.stack([torch.randperm(E, generator=g)[:top] for _ in range(T)])
    if holes:
        ids[torch.rand(T, top, generator=g) < 0.1] = -1
    w = torch.rand(T, top, generator=g)
    return ids.to(dev), w.to(dev)


CASES = [("bf16", False), ("fp8:channel", False), ("fp8:block", False), ("nvfp4", False), ("nvfp4", True),
         ("int4", False), ("mxfp4", False), ("int4_zp", False)]


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fmt,a4", CASES)
def test_every_format_matches_the_fp64_reference(fmt, a4):
    experts = Experts(fmt, a4=a4)
    g = torch.Generator().manual_seed(3)
    x = (torch.randn(13, experts.H, generator=g)).bfloat16().to(dev)
    ids, w = routes(13, 3, experts.E, g)
    out = experts.forward(x, ids, w).double().cpu()
    ref = experts.reference(x, ids.cpu(), w.cpu())
    scale = ref.abs().max()
    assert (out - ref).abs().max() <= (3e-2 if a4 else 5e-3) * scale, (fmt, a4, float((out - ref).abs().max() / scale))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fmt,a4", [("bf16", False), ("nvfp4", True), ("fp8:block", False), ("mxfp4", False),
                                    ("int4_zp", False)])
def test_a_route_keeps_its_bits_whatever_shares_its_call(fmt, a4):
    experts = Experts(fmt, E=16, a4=a4, seed=1)
    g = torch.Generator().manual_seed(5)
    T, top = 300, 4   # 1200 routes: the sorted plan; the slices below take the one-block plan
    x = torch.randn(T, experts.H, generator=g).bfloat16().to(dev)
    ids, w = routes(T, top, experts.E, g)
    full = experts.rows(x, ids)
    valid = (ids.reshape(-1) >= 0).cpu()
    for width in (8, 32, 64):
        for first, count in ((0, 1), (17, 1), (40, 7), (100, 200)):
            part = experts.rows(x[first:first + count].clone(), ids[first:first + count].clone(), width=width)
            sel = valid[first * top:(first + count) * top]
            assert torch.equal(part[sel], full[first * top:(first + count) * top][sel]), (fmt, width, first, count)
    again = experts.rows(x, ids)
    assert torch.equal(again[valid], full[valid])
    out = experts.forward(x, ids, w)
    assert torch.equal(experts.forward(x[17:18].clone(), ids[17:18].clone(), w[17:18].clone())[0], out[17])


@pytest.mark.kernels_ci
def test_slots_place_experts_anywhere_and_negative_slots_are_skipped():
    experts = Experts("nvfp4", E=8, seed=2)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(20, experts.H, generator=g).bfloat16().to(dev)
    ids, w = routes(20, 3, experts.E, g, holes=False)
    base = experts.forward(x, ids, w)
    perm = torch.randperm(experts.E, generator=g)
    store = torch.zeros(experts.E + 3, experts.store.shape[1], dtype=torch.uint8, device=dev)
    store[perm + 3] = experts.store
    slots = (perm + 3).int().to(dev)
    ws = gx.Workspace(20, 3, experts.E, experts.H, experts.I, dev)
    moved = gx.forward(x, ids, w, store, slots, experts.layout, ws)
    assert torch.equal(moved, base)
    slots[2] = -2   # expert 2 on another tier: its routes skipped here
    skipped = gx.forward(x, ids, w, store, slots, experts.layout, ws)
    ref = experts.reference(x, ids.cpu(), w.cpu(), slots.cpu())
    assert (skipped.double().cpu() - ref).abs().max() <= 5e-3 * ref.abs().max()
    y = gx.route_outputs(x, ids, store, slots, experts.layout, ws).clone()
    full = experts.rows(x, ids)
    tier = (ids.reshape(-1) == 2)
    y[tier] = full[tier]   # the other tier's rows, then the declared combine without slots
    assert torch.equal(gx.combine(y, ids, w, None, experts.E), base)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("tokens", [1, 8, 120])
def test_a_captured_call_replays_new_routes(tokens):
    experts = Experts("nvfp4", E=32, seed=4, a4=True)
    g = torch.Generator().manual_seed(9)
    x = torch.randn(tokens, experts.H, generator=g).bfloat16().to(dev)
    ids, w = routes(tokens, 4, experts.E, g)
    ws = gx.Workspace(tokens, 4, experts.E, experts.H, experts.I, dev)
    out = torch.empty(tokens, experts.H, device=dev)
    gx.forward(x, ids, w, experts.store, None, experts.layout, ws, True, out)   # the first call, outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gx.forward(x, ids, w, experts.store, None, experts.layout, ws, True, out)
    for seed in (11, 12):
        g2 = torch.Generator().manual_seed(seed)
        x.copy_(torch.randn(tokens, experts.H, generator=g2).bfloat16())
        new_ids, new_w = routes(tokens, 4, experts.E, g2)
        ids.copy_(new_ids)
        w.copy_(new_w)
        graph.replay()
        eager = experts.forward(x, ids, w)
        assert torch.equal(out, eager)


@pytest.mark.kernels_ci
def test_the_quantizer_is_the_declared_one():
    g = torch.Generator().manual_seed(13)
    x = torch.randn(9, 256, generator=g) * torch.rand(9, 1, generator=g) * 10
    x[0, :16] = 0.0
    x[1, 3] = float("nan")
    values, scales = gx.quantize_e2m1x2(x.to(dev), 37.5)
    signed, decoded = quantize_codes(torch.nan_to_num(x, nan=0.0), 37.5)
    want = (2 * signed).to(torch.int8).reshape(x.shape)
    want_scales = decoded / torch.tensor(37.5)
    want[1, :16] = 0                      # the NaN's block: zero codes and a NaN scale, so its products carry it
    want_scales[1, 0] = float("nan")
    assert torch.equal(values.cpu(), want)
    assert torch.equal(scales.cpu().nan_to_num(nan=-1.0), want_scales.nan_to_num(nan=-1.0))
    assert bool(scales[1, 0].isnan())


# ------------------------------------------------------------------------------------- the declared arithmetic
# An exact oracle of crit's integer block order (host-expert-arithmetic host_int.h, the CPU tier's): every FP32 step
# rounded once, fmaf through exact rationals.

from fractions import Fraction  # noqa: E402


def f32(value):
    if isinstance(value, Fraction):
        return _round(value)
    return float(torch.tensor(value, dtype=torch.float64).float())


def _round(q):
    if q == 0:
        return 0.0
    sign = -1.0 if q < 0 else 1.0
    q = abs(q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    e = max(e, -126)
    scaled = q / Fraction(2) ** (e - 23)
    whole, rest = divmod(scaled.numerator, scaled.denominator)
    half = Fraction(rest, scaled.denominator)
    if half > Fraction(1, 2) or (half == Fraction(1, 2) and whole % 2):
        whole += 1
    return sign * math.ldexp(whole, e - 23)


def fma32(a, b, c):
    return _round(Fraction(a) * Fraction(b) + Fraction(c))


def add32(a, b):
    return _round(Fraction(a) + Fraction(b))


def mul32(a, b):
    return _round(Fraction(a) * Fraction(b))


def div32(a, b):
    return _round(Fraction(a) / Fraction(b))


def exp_declared(x):
    x = 88.0 if x > 88.0 else -87.0 if x < -87.0 else x
    n = float(round(mul32(x, 1.44269504)))   # rint: ties to even (Python's round)
    r = fma32(n, f32(-0.693145752), x)
    r = fma32(n, f32(-1.42860677e-06), r)
    p = f32(1.98412698e-04)
    for c in (1.38888889e-03, 8.33333333e-03, 4.16666667e-02, 1.66666667e-01, 0.5, 1.0, 1.0):
        p = fma32(p, r, f32(c))
    return mul32(p, math.ldexp(1.0, int(n)))


def swiglu_declared(gate, up):
    return mul32(div32(gate, add32(1.0, exp_declared(-gate))), up)


def declared_row(weights, wscales, acts, ascales, ratio, post):
    lanes = [0.0] * 16
    for b in range(len(wscales)):
        s = sum(int(w) * int(a) for w, a in zip(weights[16 * b:16 * b + 16], acts[16 * b:16 * b + 16]))
        lanes[b % 16] = fma32(float(s), mul32(wscales[b], ascales[b // ratio]), lanes[b % 16])
    w = 8
    while w >= 1:
        lanes = [add32(lanes[i], lanes[i + w]) if i < w else lanes[i] for i in range(16)]
        w //= 2
    return mul32(lanes[0], post)


def quantize_int8(v, block=32):
    out, scales = [], []
    for b in range(len(v) // block):
        seg = [f32(x) for x in v[block * b:block * b + block]]
        d = div32(max(abs(x) for x in seg), 127.0)
        scales.append(d)
        for x in seg:
            q = float(round(div32(x, d))) if d > 0 else 0.0
            out.append(int(max(-127.0, min(127.0, q))))
    return out, scales


def declared_reference(experts, x, ids, fmt):
    """y [routes][hidden] of the declared arithmetic, route by route."""
    H, I = experts.H, experts.I
    store = experts.store.cpu()
    o, _ = offsets(fmt, H, I)
    int4 = fmt == "int4"

    def weights_of(e, name, rows, cols):
        codes = store[e, o[name]:o[name] + rows * cols // 2].view(rows, cols // 2)
        lo, hi = (codes & 15).long(), (codes >> 4).long()
        table = [c if c < 8 else c - 16 for c in range(16)] if int4 else \
            [2 * v for v in E2M1] + [-2 * v for v in E2M1]
        vals = torch.tensor(table)[torch.stack([lo, hi], -1).reshape(rows, cols)]
        sc = store[e, o[name + "_scale"]:o[name + "_scale"] + rows * cols // 16].view(rows, cols // 16)
        wscale = sc.view(torch.float8_e4m3fn).float().tolist()
        inv = store[e, o[name + "_inverse"]:o[name + "_inverse"] + 4 * rows].view(torch.float32).tolist()
        post = [inv[n] if int4 else mul32(0.25, inv[n]) for n in range(rows)]
        return vals.tolist(), wscale, post

    y = {}
    xs = x.float().cpu()
    for t in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            if e < 0:
                continue
            g_in, g_dn = experts.inputs[e]
            if int4:
                acts, ascales = quantize_int8(xs[t].tolist())
            else:
                signed, decoded = quantize_codes(xs[t:t + 1], f32(g_in))
                acts = (2 * signed).reshape(-1).int().tolist()
                ascales = (decoded / torch.tensor(f32(g_in))).reshape(-1).tolist()
            gw, gs, gp = weights_of(e, "gate_up", 2 * I, H)
            ratio = 2 if int4 else 1
            gu = [declared_row(gw[n], gs[n], acts, ascales, ratio, gp[n]) for n in range(2 * I)]
            h = [swiglu_declared(gu[k], gu[I + k]) for k in range(I)]
            if int4:
                hv, hsc = quantize_int8(h)
            else:
                signed, decoded = quantize_codes(torch.tensor([h], dtype=torch.float32), f32(g_dn))
                hv = (2 * signed).reshape(-1).int().tolist()
                hsc = (decoded / torch.tensor(f32(g_dn))).reshape(-1).tolist()
            dw, dsc, dp = weights_of(e, "down", H, I)
            y[t * ids.shape[1] + j] = [declared_row(dw[n], dsc[n], hv, hsc, ratio, dp[n]) for n in range(H)]
    return y


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fmt", ["nvfp4", "int4"])
def test_the_declared_arithmetic_is_the_integer_block_order_bit_for_bit(fmt):
    experts = Experts(fmt, E=3, H=128, I=64, seed=21)
    g = torch.Generator().manual_seed(22)
    x = torch.randn(3, experts.H, generator=g).bfloat16().to(dev)
    ids, w = routes(3, 2, experts.E, g, holes=False)
    ws = gx.Workspace(3, 2, experts.E, experts.H, experts.I, dev)
    y = gx.route_outputs(x, ids, experts.store, None, experts.layout, ws, arithmetic="declared").clone()
    expected = declared_reference(experts, x, ids.cpu(), fmt)
    for route, row in expected.items():
        assert y[route].cpu().tolist() == row, (fmt, route)
    wide = gx.Workspace(3, 2, experts.E, experts.H, experts.I, dev, width=32)
    again = gx.route_outputs(x, ids, experts.store, None, experts.layout, wide, arithmetic="declared")
    assert torch.equal(again, y)
    ref = experts.reference(x, ids.cpu(), w.cpu()) if fmt == "nvfp4" else None
    if ref is not None:   # W4A4 against the FP64 block of the same quantized operands
        out = gx.combine(y, ids, w, None, experts.E).double().cpu()
        experts.a4 = True
        ref = experts.reference(x, ids.cpu(), w.cpu())
        assert (out - ref).abs().max() <= 3e-2 * ref.abs().max()


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fmt,a4", [("nvfp4", False), ("nvfp4", True), ("bf16", False), ("fp8:block", False)])
def test_long_reductions_give_the_same_bits_in_decode_and_prompt(fmt, a4):
    # hidden 2048 and inter 1152: long k loops through the decode items' prefetch rings and the prompt items' staging
    # halves give the same bits.
    experts = Experts(fmt, E=8, H=2048, I=1152, seed=31, a4=a4)
    g = torch.Generator().manual_seed(32)
    T, top = 40, 2
    x = torch.randn(T, experts.H, generator=g).bfloat16().to(dev)
    ids, w = routes(T, top, experts.E, g)
    prompt_rows = experts.rows(x, ids, width=32)
    valid = (ids.reshape(-1) >= 0).cpu()
    for first, count in ((0, 1), (7, 4), (30, 10)):
        part = experts.rows(x[first:first + count].clone(), ids[first:first + count].clone(), width=8)
        sel = valid[first * top:(first + count) * top]
        assert torch.equal(part[sel], prompt_rows[first * top:(first + count) * top][sel]), (fmt, first)
    out = experts.forward(x[:6], ids[:6], w[:6])
    ref = experts.reference(x[:6], ids[:6].cpu(), w[:6].cpu())
    assert (out.double().cpu() - ref).abs().max() <= (3e-2 if a4 else 5e-3) * ref.abs().max()


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fmt,a4", CASES)
def test_partial_128_blocks_give_decode_the_prompt_bits_and_match_the_fp64_reference(fmt, a4):
    # hidden 320 and inter 192 (64 mod 128): the last 128-k block of each matrix and the last fp8:block scale rows of
    # gate and up are half blocks.
    experts = Experts(fmt, E=8, H=320, I=192, seed=5, a4=a4)
    g = torch.Generator().manual_seed(6)
    T, top = 300, 4
    x = torch.randn(T, experts.H, generator=g).bfloat16().to(dev)
    ids, w = routes(T, top, experts.E, g)
    valid = (ids.reshape(-1) >= 0).cpu()
    full = experts.rows(x, ids)   # the sorted plan, 32-route items
    for width in (8, 32, 64):
        for first, count in ((0, 1), (3, 5), (50, 250)):
            part = experts.rows(x[first:first + count].clone(), ids[first:first + count].clone(), width=width)
            sel = valid[first * top:(first + count) * top]
            assert torch.equal(part[sel], full[first * top:(first + count) * top][sel]), (fmt, width, first, count)
    out = experts.forward(x[:12], ids[:12], w[:12]).double().cpu()
    ref = experts.reference(x[:12], ids[:12].cpu(), w[:12].cpu())
    assert (out - ref).abs().max() <= (3e-2 if a4 else 5e-3) * ref.abs().max(), fmt


def e8m0(byte):
    """An E8M0 scale byte's value as the format defines it: 2^(byte - 127), byte 255 NaN."""
    return float("nan") if byte == 255 else math.ldexp(1.0, byte - 127)


def mxfp4_values(store, e, offset, scale_offset, rows, cols):
    codes = store[e, offset:offset + rows * cols // 2].view(rows, cols // 2)
    vals = torch.stack([E2M1_SIGNED[(codes & 15).long()], E2M1_SIGNED[(codes >> 4).long()]], -1).reshape(rows, cols)
    s = store[e, scale_offset:scale_offset + rows * cols // 32].view(rows, cols // 32)
    scale = torch.tensor([[e8m0(int(b)) for b in row] for row in s.tolist()], dtype=torch.float64)
    return vals * scale.repeat_interleave(32, 1)


@pytest.mark.kernels_ci
def test_mxfp4_scale_bytes_0_and_255_decode_as_the_format_defines():
    experts = Experts("mxfp4", E=2, H=128, I=64, seed=51)
    H, I = experts.H, experts.I
    o, _ = offsets("mxfp4", H, I)
    store = experts.store.cpu()
    scales = store[:, o["gate_up_scale"]:o["gate_up_scale"] + 2 * I * H // 32].view(2, 2 * I, H // 32)
    scales[0, :I] = 0     # expert 0: every gate weight times 2^-127 (BF16 subnormals and their neighbours)
    scales[0, I:] = 127   # and its up weights times 1
    scales[1, 1, 1] = 255   # expert 1: one block of gate row 1 NaN
    experts.store = store.to(dev)
    g = torch.Generator().manual_seed(52)
    x = (torch.randn(2, H, generator=g) * 2.0 ** 20).bfloat16().to(dev)   # products of 2^-127 weights stay normal
    ids = torch.tensor([[0], [1]], device=dev)
    w = torch.ones(2, 1, device=dev)
    out = experts.forward(x, ids, w).double().cpu()
    gate_up = mxfp4_values(store, 0, o["gate_up"], o["gate_up_scale"], 2 * I, H)
    down = mxfp4_values(store, 0, o["down"], o["down_scale"], H, I)
    gu = gate_up @ x[0].double().cpu()
    h = (torch.nn.functional.silu(gu[:I]) * gu[I:]).float().bfloat16().double()
    ref = down @ h
    assert bool(ref.abs().max() > 0) and (out[0] - ref).abs().max() <= 5e-3 * ref.abs().max()
    assert bool(torch.isnan(out[1]).all())


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fmt", ["nvfp4", "int4"])
def test_the_declared_arithmetic_with_several_blocks_a_lane_bit_for_bit(fmt):
    # hidden 512: 32 gate/up blocks, two on each of the 16 lanes; inter 256: 16 down blocks. Then 300 tokens batched
    # (32-route items) against one alone.
    experts = Experts(fmt, E=4, H=512, I=256, seed=41)
    g = torch.Generator().manual_seed(42)
    x = (torch.randn(2, experts.H, generator=g) * 3).bfloat16().to(dev)
    ids, _ = routes(2, 2, experts.E, g, holes=False)
    ws = gx.Workspace(2, 2, experts.E, experts.H, experts.I, dev)
    y = gx.route_outputs(x, ids, experts.store, None, experts.layout, ws, arithmetic="declared").clone()
    for route, row in declared_reference(experts, x, ids.cpu(), fmt).items():
        assert y[route].cpu().tolist() == row, (fmt, route)
    g2 = torch.Generator().manual_seed(43)
    X = torch.randn(300, experts.H, generator=g2).bfloat16().to(dev)
    ids2, _ = routes(300, 2, experts.E, g2, holes=False)
    big = gx.route_outputs(X, ids2, experts.store, None, experts.layout,
                           gx.Workspace(300, 2, experts.E, experts.H, experts.I, dev), arithmetic="declared").clone()
    one = gx.route_outputs(X[123:124].clone(), ids2[123:124].clone(), experts.store, None, experts.layout,
                           gx.Workspace(1, 2, experts.E, experts.H, experts.I, dev), arithmetic="declared")
    assert torch.equal(one, big[246:248])


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fmt,tokens", [("nvfp4", 1), ("nvfp4", 8), ("int4", 8)])
def test_a_captured_declared_call_replays_new_routes(fmt, tokens):
    experts = Experts(fmt, E=16, seed=61)
    g = torch.Generator().manual_seed(62)
    x = torch.randn(tokens, experts.H, generator=g).bfloat16().to(dev)
    ids, w = routes(tokens, 3, experts.E, g)
    ws = gx.Workspace(tokens, 3, experts.E, experts.H, experts.I, dev)
    out = torch.empty(tokens, experts.H, device=dev)
    a4 = fmt == "nvfp4"
    gx.forward(x, ids, w, experts.store, None, experts.layout, ws, a4, out, "declared")   # outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gx.forward(x, ids, w, experts.store, None, experts.layout, ws, a4, out, "declared")
    for seed in (63, 64):
        g2 = torch.Generator().manual_seed(seed)
        x.copy_(torch.randn(tokens, experts.H, generator=g2).bfloat16())
        new_ids, new_w = routes(tokens, 3, experts.E, g2)
        ids.copy_(new_ids)
        w.copy_(new_w)
        graph.replay()
        fresh = gx.Workspace(tokens, 3, experts.E, experts.H, experts.I, dev)
        eager = gx.forward(x, ids, w, experts.store, None, experts.layout, fresh, a4, None, "declared")
        assert torch.equal(out, eager)


@pytest.mark.kernels_ci
def test_one_expert_limit_for_both_plans():
    for tokens in (1, 2048):   # the one-block plan and the sorted one
        gx.Workspace(tokens, 1, gx.MAX_EXPERTS, 64, 64, dev)
        with pytest.raises(ValueError, match="experts"):
            gx.Workspace(tokens, 1, gx.MAX_EXPERTS + 1, 64, 64, dev)


OTHER = {"decode": {"WARPS": 8, "TILES": 4, "DEPTH": 2}, "prompt": {"WARPS": 2, "TILES": 4, "DEPTH": 1, "CHUNK": 64},
         "prompt-wide": {"WARPS": 8, "TILES": 2, "DEPTH": 3, "CHUNK": 64}}


@pytest.mark.kernels_ci
def test_other_knobs_give_every_route_the_same_bits():
    other = load("grouped-experts", knobs=OTHER)
    for fmt, a4 in CASES:
        experts = Experts(fmt, E=5, H=256, I=192, seed=7, a4=a4)
        g = torch.Generator().manual_seed(9)
        x = torch.randn(37, experts.H, generator=g).bfloat16().to(dev)
        ids, _ = routes(37, 3, experts.E, g, holes=False)
        for width in (8, 32, 64):
            want = experts.rows(x, ids, width=width)
            ws = other.Workspace(37, 3, experts.E, experts.H, experts.I, dev, width)
            got = other.route_outputs(x, ids, experts.store, None, experts.layout, ws, a4)
            assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (fmt, a4, width)
