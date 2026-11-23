"""host_int.h's declared integer block order in torch, on any device, with the same bits as the C++ rows and the device
kernels that declare it: the torch fallback of the CPU rows and the CPU expert tier, and an oracle for both.

Every operation is one IEEE FP32 operation, as the C++ writes it. torch has no fused multiply-add, so `fma32` computes
a * b + c rounded once: for FP32 a and b the product is exact in FP64, TwoSum gives the FP64 sum and its exact error,
and the FP64 sum rounded to FP32 is the FMA unless it is exactly halfway between two FP32 values, when the error's sign
picks the side. The declared constants are the FP32 values of their decimal literals, rounded once (not through FP64).

The order (host_int.h): a block's integer dot (exact), lane b % 16 = fma(float(sum), wscale_b * ascale_b, lane) with
the product rounded, blocks ascending, the halving tree over the 16 lanes, times the row's post. quantize_e2m1x2,
quantize_int8, exp / silu / swiglu_declared and the route-order combine as host_int.cpp and tier_capi.cpp write them.
"""
from fractions import Fraction
import math
import struct

E2M1_X2 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
INT4 = (0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1)
E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
LANES = 16


def f32_literal(text):
    """The FP32 value of a decimal literal rounded once to nearest even, as a C++ compiler reads `1.5f`."""
    q = Fraction(text)
    if q == 0:
        return 0.0
    sign, q = (-1.0 if q < 0 else 1.0), abs(q)
    exponent = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** exponent > q:
        exponent -= 1
    exponent = max(exponent, -126)
    scaled = q / Fraction(2) ** (exponent - 23)
    whole, rest = divmod(scaled.numerator, scaled.denominator)
    if Fraction(rest, scaled.denominator) > Fraction(1, 2) or (Fraction(rest, scaled.denominator) == Fraction(1, 2)
                                                               and whole % 2):
        whole += 1
    value = sign * math.ldexp(whole, exponent - 23)
    return struct.unpack("<f", struct.pack("<f", value))[0]


LOG2E = f32_literal("1.44269504")
LN2_HI, LN2_LO = f32_literal("-0.693145752"), f32_literal("-1.42860677e-06")
POLY = [f32_literal(t) for t in ("1.98412698e-04", "1.38888889e-03", "8.33333333e-03", "4.16666667e-02",
                                  "1.66666667e-01", "0.5", "1.0", "1.0")]


def _torch():
    import torch
    return torch


_CONSTANTS = {}


def _constant(name, values, dtype, device):
    """A constant table on `device`, made once (a host-to-device copy cannot be captured in a CUDA graph)."""
    torch = _torch()
    key = (name, str(dtype), str(torch.device(device) if device is not None else "cpu"))
    if key not in _CONSTANTS:
        _CONSTANTS[key] = torch.tensor(values, dtype=dtype, device=device)
    return _CONSTANTS[key]


def fma32(a, b, c):
    """a * b + c rounded once to FP32 (a, b, c FP32 tensors, broadcast)."""
    torch = _torch()
    ab = a.double() * b.double()
    cd = c.double()
    hi = ab + cd
    bb = hi - ab
    lo = (ab - (hi - bb)) + (cd - bb)
    r = hi.float()
    rd = r.double()
    up = torch.nextafter(r, torch.full_like(r, math.inf))
    down = torch.nextafter(r, torch.full_like(r, -math.inf))
    above = (hi > rd) & ((hi - rd) * 2 == up.double() - rd) & (lo > 0)
    below = (hi < rd) & ((rd - hi) * 2 == rd - down.double()) & (lo < 0)
    return torch.where(above, up, torch.where(below, down, r))


def prepare(device):
    """Every constant table the declared order reads, on `device`, now: a capture that follows copies nothing from the
    host."""
    torch = _torch()
    e4m3_table(device)
    _constant("e2m1x2", E2M1_X2[:8], torch.int64, device)
    for weights in ("e2m1x2", "int4"):
        _constant(weights, INT4 if weights == "int4" else E2M1_X2, torch.int8, device)
    _constant("e2m1", E2M1, torch.float32, device)


def e4m3_table(device=None):
    """The 256 E4M3FN values by byte (0x7F and 0xFF NaN), crit::nvfp4::e4m3's floats."""
    torch = _torch()
    return _constant("e4m3", _e4m3_values(), torch.float32, device)


def _e4m3_values():
    values = []
    for byte in range(256):
        exponent, mantissa = (byte >> 3) & 15, byte & 7
        sign = -1.0 if byte & 0x80 else 1.0
        if exponent == 15 and mantissa == 7:
            values.append(math.nan)
        elif exponent == 0:
            values.append(sign * math.ldexp(mantissa / 8.0, -6))
        else:
            values.append(sign * math.ldexp(1.0 + mantissa / 8.0, exponent - 7))
    return values


def _e4m3_encode(m, table):
    """crit::nvfp4::e4m3_encode of magnitudes m (FP32, >= 0, saturated at 448): the nearest of codes 0..126, ties to the
    even code; distances in FP32 as the C++ computes them."""
    torch = _torch()
    positive = table[:127]
    hi = torch.searchsorted(positive, m.contiguous()).clamp(1, 126)
    lo = hi - 1
    d_lo = (m - positive[lo]).abs()
    d_hi = (m - positive[hi]).abs()
    take_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    return torch.where(take_hi, hi, lo)


def _e2m1_code(q):
    """e2m1_encode's magnitude code 0..7 of |q| saturated at 6 (nearest, ties to the even code), with its sign."""
    torch = _torch()
    m = q.abs().clamp(max=6.0)
    code = torch.zeros_like(m, dtype=torch.int64)
    for i, mid in enumerate(E2M1_MIDS):
        code = code + ((m > mid) | ((m == mid) & (i % 2 == 1))).long()
    return code, (q < 0) & (code > 0)


def _amax(blocks):
    torch = _torch()
    a = blocks.abs()
    return torch.where(torch.isnan(a), torch.zeros_like(a), a).amax(dim=-1)


def quantize_e2m1x2(x, global_scale):
    """x [tokens][n] FP32 -> (values int8 [tokens][n] = 2 * e2m1, scales FP32 [tokens][n / 16]) as host_int's."""
    torch = _torch()
    tokens, n = x.shape
    blocks = x.float().reshape(tokens, n // 16, 16)
    g = (global_scale.to(device=x.device, dtype=torch.float32).reshape(()) if torch.is_tensor(global_scale)
         else torch.full((), float(global_scale), dtype=torch.float32, device=x.device))
    # Divisors are tensors: torch's CUDA division by a Python number multiplies by its reciprocal, another rounding.
    wanted = (_amax(blocks) * g) / torch.full((), 6.0, dtype=torch.float32, device=x.device)
    wanted = torch.where(torch.isfinite(wanted), wanted, torch.full_like(wanted, 448.0)).clamp(max=448.0)
    table = e4m3_table(x.device)
    decoded = table[_e4m3_encode(wanted, table)]
    q = (blocks * g) / decoded[..., None]
    code, negative = _e2m1_code(q)
    values = _constant("e2m1x2", E2M1_X2[:8], torch.int64, x.device)[code]
    values = torch.where(negative, -values, values)
    live = (decoded[..., None] > 0) & (q == q)
    values = torch.where(live, values, torch.zeros_like(values)).to(torch.int8).reshape(tokens, n)
    return values, decoded / g


def quantize_int8(x, block=32):
    """x [tokens][n] FP32 -> (values int8, scales FP32 [tokens][n / block]) as host_int's quantize_int8."""
    torch = _torch()
    tokens, n = x.shape
    blocks = x.float().reshape(tokens, n // block, block)
    d = _amax(blocks) / torch.full((), 127.0, dtype=torch.float32, device=x.device)
    q = torch.where(d[..., None] > 0, torch.round(blocks / d[..., None]), torch.zeros_like(blocks))
    q = torch.where(q == q, q, torch.zeros_like(q)).clamp(-127.0, 127.0)
    return q.to(torch.int8).reshape(tokens, n), d


def weight_values(codes, weights="e2m1x2"):
    """codes [rows][cols / 2] uint8 -> the integer weight values [rows][cols] int8 (low nibble the even element)."""
    torch = _torch()
    table = _constant(weights, INT4 if weights == "int4" else E2M1_X2, torch.int8, codes.device)
    low, high = (codes & 15).long(), (codes >> 4).long()
    return torch.stack([table[low], table[high]], dim=-1).reshape(codes.shape[0], codes.shape[1] * 2)


def int_rows(codes, scales, post, values, act_scales, block, weights="e2m1x2", factor=1.0, row_chunk=256):
    """The declared integer rows: codes [N][K / 2] uint8, scales [N][K / 16] uint8 E4M3, post [N] FP32 or None, values
    [T][K] int8, act_scales [T][K / block] FP32 -> out [T][N] FP32, each row times factor * post[n] (or factor)."""
    torch = _torch()
    n_rows, k = codes.shape[0], codes.shape[1] * 2
    tokens, blocks, ratio = values.shape[0], k // 16, block // 16
    table = e4m3_table(codes.device)
    f = torch.full((), float(factor), dtype=torch.float32, device=codes.device)
    a_scale = act_scales.float().repeat_interleave(ratio, dim=1)[:, :blocks]          # [T][B]
    v = values.to(torch.int32).reshape(tokens, 1, blocks, 16)
    out = torch.empty(tokens, n_rows, dtype=torch.float32, device=codes.device)
    for r0 in range(0, n_rows, row_chunk):
        r1 = min(n_rows, r0 + row_chunk)
        w = weight_values(codes[r0:r1], weights).to(torch.int32).reshape(1, r1 - r0, blocks, 16)
        sums = (w * v).sum(dim=-1, dtype=torch.int32).float()                           # [T][R][B], exact
        p = table[scales[r0:r1].long()][None] * a_scale[:, None, :]                     # [T][R][B], rounded once
        acc = torch.zeros(tokens, r1 - r0, LANES, dtype=torch.float32, device=codes.device)
        for b0 in range(0, blocks, LANES):
            width = min(LANES, blocks - b0)
            lanes = fma32(sums[..., b0:b0 + width], p[..., b0:b0 + width], acc[..., :width])
            acc = torch.cat([lanes, acc[..., width:]], dim=-1) if width < LANES else lanes
        width = LANES // 2
        while width:
            acc = acc[..., :width] + acc[..., width:2 * width]
            width //= 2
        total = acc[..., 0]
        if post is None:
            out[:, r0:r1] = total * f
        else:
            out[:, r0:r1] = (total * 1.0) * (f * post[r0:r1].float())[None]
    return out


def exp_declared(x):
    """exp_declared: n = rint(x * log2 e), r = x - n ln2 by two FMAs, a degree-7 polynomial by FMAs, times 2^n built from
    the exponent bits; x clamped to [-87, 88]."""
    torch = _torch()
    c = lambda v: torch.full_like(x, v, dtype=torch.float32)  # noqa: E731
    x = x.float().clamp(-87.0, 88.0)
    n = torch.round(x * c(LOG2E))
    r = fma32(n, c(LN2_HI), x)
    r = fma32(n, c(LN2_LO), r)
    p = c(POLY[0])
    for coefficient in POLY[1:]:
        p = fma32(p, r, c(coefficient))
    scale = ((n.to(torch.int32) + 127) << 23).view(torch.float32)
    return p * scale


def silu_declared(x):
    return x / (1.0 + exp_declared(-x))


def swiglu_declared(gate, up):
    return silu_declared(gate.float()) * up.float()


def combine(y, ids, weights):
    """out[t] = acc, acc = fma(weights[t][j], y[t * top + j], acc) for j ascending over ids[t][j] >= 0, acc from 0."""
    torch = _torch()
    tokens, top = ids.shape
    y = y.reshape(tokens, top, -1).float()
    acc = torch.zeros(tokens, y.shape[-1], dtype=torch.float32, device=y.device)
    for j in range(top):
        step = fma32(weights[:, j:j + 1].float(), y[:, j], acc)
        acc = torch.where((ids[:, j] >= 0)[:, None], step, acc)
    return acc


def quantize(x, kind, global_scale=1.0, block=32):
    """The declared quantizer by kind: e2m1x2 (16 a block under global_scale) or int8 (per block)."""
    if kind == "e2m1x2":
        return quantize_e2m1x2(x, global_scale)
    return quantize_int8(x, block)


def expert_rows(x, gate_up, down, weights="e2m1x2", factor=0.25, input_kind="e2m1x2", input_global=1.0,
                inter_kind="e2m1x2", inter_global=1.0, block=32):
    """One expert's unweighted outputs for token rows x [T][H] FP32: gate_up / down are (codes, scales, inverse) as a
    bundle stores them; host-expert-tier's composition (inputs quantized, gate/up rows, declared SwiGLU, the intermediate
    quantized, down rows), each row's post factor * inverse."""
    inter = gate_up[0].shape[0] // 2
    v, s = quantize(x.float(), input_kind, input_global, block)
    gu = int_rows(gate_up[0], gate_up[1], gate_up[2], v, s, 16 if input_kind == "e2m1x2" else block, weights, factor)
    h = swiglu_declared(gu[:, :inter], gu[:, inter:])
    hv, hs = quantize(h, inter_kind, inter_global, block)
    return int_rows(down[0], down[1], down[2], hv, hs, 16 if inter_kind == "e2m1x2" else block, weights, factor)


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def nvfp4_rows(codes, scales, post, x, row_chunk=128):
    """host_nvfp4.h's declared FP32 order (scaled_elements, 16 lanes, halving) against FP32 x [T][K]: element k to lane
    k % 16, lane = fma(f32(e2m1 * scale), x[k], lane) in increasing k, the halving tree, times post[n] (or 1)."""
    torch = _torch()
    n_rows, k = codes.shape[0], codes.shape[1] * 2
    tokens = x.shape[0]
    table = e4m3_table(codes.device)
    e2m1 = _constant("e2m1", E2M1, torch.float32, codes.device)
    x = x.float().reshape(tokens, 1, k // LANES, LANES)
    out = torch.empty(tokens, n_rows, dtype=torch.float32, device=codes.device)
    for r0 in range(0, n_rows, row_chunk):
        r1 = min(n_rows, r0 + row_chunk)
        c = codes[r0:r1]
        nib = torch.stack([(c & 15).long(), (c >> 4).long()], dim=-1).reshape(r1 - r0, k)
        w = e2m1[nib] * table[scales[r0:r1].long()].repeat_interleave(16, dim=1)    # rounded once
        w = w.reshape(1, r1 - r0, k // LANES, LANES)
        acc = torch.zeros(tokens, r1 - r0, LANES, dtype=torch.float32, device=codes.device)
        for g in range(k // LANES):
            acc = fma32(w[:, :, g], x[:, :, g], acc)
        width = LANES // 2
        while width:
            acc = acc[..., :width] + acc[..., width:2 * width]
            width //= 2
        total = acc[..., 0]
        out[:, r0:r1] = total if post is None else (total * 1.0) * post[r0:r1].float()[None]
    return out
