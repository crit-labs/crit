"""The grouped experts in plain torch: what the experts plugin runs where the registry kernel is not there (no published
build for the machine, a CPU device). It computes what the kernel's tensor arithmetic computes, the same operands and
factors, with torch's batched matrix products in place of the kernel's tensor-core tiles, so its sums differ from the
kernel's in their last bits but never in meaning:

- each route's expert bundle is gathered from the store by its slot and decoded exactly (bundles.operands): the
  weights' codes times their block scales as BF16 operands, the per-row factors (1 / global scale, channel scale, 1 /
  input scale) applied to the FP32 sums; fp8:block's block scales on each 128-k block's FP32 sum;
- with W4A4 the inputs and the intermediate are quantized per route by crit's declared NVFP4 quantizer at the expert's
  input scales, as the kernel quantizes them; the SwiGLU is crit's declared one;
- every product has BF16 operands and FP32 sums on CUDA (FP64 sums on the CPU), chosen per call: no global precision
  flag is read or set;
- routes go through in chunks of a fixed size (padded), so every call of any token count runs the same products of
  the same shapes: a route's bits are the same alone and batched, in decode rows and in prompt chunks, and a call at a
  fixed token count has fixed shapes, no host synchronization and no data-dependent Python, so it can be captured.

The declared arithmetic (crit's integer block order) is the kernel's alone: its bits are what it is for.

Slow (each route decodes its expert's whole matrices): a correct fallback, not a fast path; the engine's facts say
when it runs.
"""
import math

try:
    from . import bundles
except ImportError:   # loaded as a file of its own, its folder on the path
    import bundles

CHUNK_BYTES = 256 << 20   # the decoded operands of one chunk of routes
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_CONSTANTS = {}


def _constants(device):
    import torch
    key = str(device)
    if key not in _CONSTANTS:
        _CONSTANTS[key] = {"e2m1": torch.tensor(E2M1, dtype=torch.float32, device=device)}
    return _CONSTANTS[key]


def chunk_routes(found):
    """Routes per chunk for a layout: the decoded operands and their temporaries within CHUNK_BYTES, 1 to 64."""
    H, I = found["hidden"], found["inter"]
    per = 3 * H * I * 12 + found["bundle_bytes"]
    return max(1, min(64, CHUNK_BYTES // per))


def exp_declared(x):
    """crit's declared exp (host_int exp_declared) in FP32 torch operations, one rounding each."""
    import torch
    x = x.clamp(-87.0, 88.0)
    n = torch.round(x * 1.44269504)
    r = x - n * 0.693145752   # torch rounds the product and the difference: close to, not, the declared FMAs
    r = r - n * 1.42860677e-06
    p = torch.full_like(r, 1.98412698e-04)
    for c in (1.38888889e-03, 8.33333333e-03, 4.16666667e-02, 1.66666667e-01, 0.5, 1.0, 1.0):
        p = p * r + c
    return p * torch.ldexp(torch.ones_like(n), n)


def swiglu(gate, up):
    return gate / (1.0 + exp_declared(-gate)) * up


def _quantized(v, g):
    """The declared NVFP4 quantizer (host_int quantize_e2m1x2) per row at each row's global g ([rows] FP32): (signed
    E2M1 magnitudes [rows][blocks][16], decoded block scales [rows][blocks]), each step rounded to FP32."""
    import torch
    rows, cols = v.shape
    blocks = v.float().reshape(rows, cols // 16, 16)
    amax = torch.nan_to_num(blocks.abs(), nan=0.0).amax(-1)
    g = g.float().reshape(rows, 1)
    wanted = (amax * g) / 6.0
    wanted = torch.where(torch.isfinite(wanted), wanted, torch.full_like(wanted, 448.0))
    decoded = wanted.clamp(max=448.0).to(torch.float8_e4m3fn).float()
    q = (blocks * g[..., None]) / decoded[..., None].clamp(min=1e-38)
    a = q.abs()
    code = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int() + (a > 2.5).int()
            + (a >= 3.5).int() + (a > 5.0).int())
    magnitude = _constants(v.device)["e2m1"][code.long()]
    signed = torch.where((q < 0) & (code > 0), -magnitude, magnitude)
    signed = torch.where(decoded[..., None] > 0, signed, torch.zeros_like(signed))
    return signed, decoded


def quantize(v, g):
    """The operand values e2m1(code) * decoded scale (exact in BF16) of the declared quantizer, FP32 [rows][cols]."""
    signed, decoded = _quantized(v, g)
    return (signed * decoded[..., None]).reshape(v.shape)


def _products(weights, vectors):
    """weights [C][rows][cols] times vectors [C][cols] -> [C][rows] FP32, both exact in BF16: BF16 operands with FP32
    sums on CUDA; FP64 sums rounded to FP32 on the CPU or where torch has no BF16 product with FP32 sums. Explicit per
    call: no global precision flag (TF32, reduced-precision CPU matmuls) changes them."""
    import torch
    if weights.is_cuda:
        a, b = weights.to(torch.bfloat16), vectors.to(torch.bfloat16)[:, :, None]
        try:
            return torch.bmm(a, b, out_dtype=torch.float32)[..., 0]
        except (TypeError, RuntimeError):
            pass
    return torch.bmm(weights.double(), vectors.double()[:, :, None])[..., 0].float()


def _block_products(values, blocks, vectors):
    """fp8:block: each 128-k block's sum (_products) times its block scale, added into the running sum in k order."""
    import torch
    C, rows, cols = values.shape
    kb = blocks.shape[2]
    pad = kb * bundles.BLOCK - cols
    if pad:
        values = torch.nn.functional.pad(values, (0, pad))
        vectors = torch.nn.functional.pad(vectors, (0, pad))
    a = values.view(C, rows, kb, bundles.BLOCK).transpose(1, 2).reshape(C * kb, rows, bundles.BLOCK)
    part = _products(a, vectors.reshape(C * kb, bundles.BLOCK)).view(C, kb, rows)
    total = torch.zeros(C, rows, dtype=torch.float32, device=values.device)
    for k in range(kb):
        total = torch.addcmul(total, part[:, k], blocks[:, :, k])
    return total


def _matrix(operand, vectors):
    values, factor, blocks = operand
    out = _products(values, vectors) if blocks is None else _block_products(values, blocks, vectors)
    return out if factor is None else out * factor


def route_outputs(x, ids, store, slots, found, a4, y):
    """Each route's unweighted output into y [tokens * top][hidden] FP32 (skipped routes keep y's rows): x [tokens]
    [hidden], ids [tokens][top], store uint8 [S][bytes], slots int32 [experts] or None (expert e in slot e). a4: NVFP4
    inputs and intermediate (W4A4)."""
    import torch
    tokens, top = ids.shape
    n = tokens * top
    device = x.device
    flat = ids.reshape(-1).long()
    experts = slots.numel() if slots is not None else store.shape[0]
    valid = (flat >= 0) & (flat < experts)
    expert = flat.clamp(0, experts - 1)
    slot = slots.long()[expert] if slots is not None else expert
    valid = valid & (slot >= 0)
    slot = torch.where(valid, slot, torch.zeros_like(slot))
    token = torch.arange(n, device=device) // top
    C = chunk_routes(found)
    padded = math.ceil(n / C) * C
    if padded > n:
        slot = torch.cat([slot, torch.zeros(padded - n, dtype=slot.dtype, device=device)])
        token = torch.cat([token, torch.zeros(padded - n, dtype=token.dtype, device=device)])
    xb = x.to(torch.bfloat16).float()
    inter = found["inter"]
    for c0 in range(0, n, C):
        b = store.index_select(0, slot[c0:c0 + C])
        (gu, gu_f, gu_b), (dn, dn_f, dn_b) = bundles.operands(b, found)
        xin = xb.index_select(0, token[c0:c0 + C])
        if a4:
            scales = bundles.inputs(b, found)
            xin = quantize(xin, scales[:, 0])
            gu_f, dn_f = gu_f * scales[:, 2:3], dn_f * scales[:, 3:4]
        both = _matrix((gu, gu_f, gu_b), xin)
        h = swiglu(both[:, :inter], both[:, inter:])
        h = quantize(h, scales[:, 1]) if a4 else h.to(torch.bfloat16).float()
        out = _matrix((dn, dn_f, dn_b), h)
        count = min(C, n - c0)
        keep = valid[c0:c0 + count, None]
        y[c0:c0 + count] = torch.where(keep, out[:count], y[c0:c0 + count])
    return y


def combine(y, ids, weights, slots=None, out=None):
    """out [tokens][hidden] FP32: each token's routes in order, weight times row added to the sum, over the routes
    with id >= 0 (and slots[id] >= 0 when slots are given)."""
    import torch
    tokens, top = ids.shape
    rows = y[:tokens * top].view(tokens, top, -1)
    keep = ids >= 0
    if slots is not None:
        keep = keep & (slots.long()[ids.long().clamp(0, slots.numel() - 1)] >= 0) & (ids < slots.numel())
    w = torch.where(keep, weights.float(), torch.zeros_like(weights, dtype=torch.float32))
    total = torch.zeros(tokens, rows.shape[2], dtype=torch.float32, device=y.device)
    for j in range(top):
        total = total + torch.where(keep[:, j:j + 1], w[:, j:j + 1] * rows[:, j], torch.zeros_like(total))
    if out is not None:
        out.copy_(total)
        return out
    return total
