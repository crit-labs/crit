"""Expert bundles: one expert's gate/up and down matrices in their stored format as one contiguous block of bytes, the
unit a slot store holds (a device expert cache's slots, a host or file store) and the grouped experts read.

A bundle keeps the matrices row major exactly as a checkpoint stores them, so a CPU tier reads them in place: gate/up
is [2 * inter][hidden] with the gate rows first, down is [hidden][inter]. Components, in this order, each starting on a
16-byte boundary (layout() gives their offsets):

    bf16         gate_up bfloat16 [2I][H]; down bfloat16 [H][I]
    fp8:channel  gate_up float8_e4m3fn [2I][H]; gate_up_scale float32 [2I]; down [H][I]; down_scale float32 [H]
                 (each row's codes times its scale)
    fp8:block    as fp8:channel, the scales one per 128 x 128 block: gate_up_scale gate's grid then up's, [2 *
                 ceil(I / 128)][ceil(H / 128)] (so an inter of 64 mod 128 keeps gate and up blocks apart);
                 down_scale [ceil(H / 128)][ceil(I / 128)]
    nvfp4        gate_up uint8 [2I][H / 2] (E2M1, element 2k in the low nibble of byte k); gate_up_scale uint8 [2I][H / 16]
                 (E4M3); gate_up_inverse float32 [2I] (1 / the row's global scale, crit's divisor convention); down,
                 down_scale, down_inverse alike; input float32 [4]: the gate/up and down input scales (divisors) and
                 their inverses, 0 when the checkpoint has none
    int4         as nvfp4, the codes INT4 two's complement nibbles; *_inverse 1 unless a global scale is given
    mxfp4        gate_up uint8 [2I][H / 2] (E2M1); gate_up_scale uint8 [2I][H / 32] (E8M0: 2^(byte - 127), byte 255
                 NaN); down, down_scale

pack() takes what an experts plugin's load() takes: per matrix a bfloat16 tensor (bf16) or the stored tensors by role,
{weight, scale, global_scale, input_scale}, as the projections slot's load takes them. It copies bytes as they are and
casts nothing: a bf16 matrix is bfloat16, a byte component (codes, E4M3 or E8M0 scales) comes as raw bytes (uint8,
int8) or in its own dtype (float8_e4m3fn, float4_e2m1fn_x2, float8_e8m0fnu).
"""
import math

FORMATS = ("bf16", "fp8:channel", "fp8:block", "nvfp4", "int4", "mxfp4")
KERNEL_FORMATS = {"bf16": 0, "fp8:channel": 1, "fp8:block": 2, "nvfp4": 3, "int4": 4, "mxfp4": 5}
ALIGN = 16
BLOCK = 128   # fp8:block's scale block

_KEYS = ("gate_up", "gate_up_scale", "gate_up_inverse", "down", "down_scale", "down_inverse", "input")


def _check(format, hidden, inter):
    if format not in FORMATS:
        raise ValueError(f"an expert bundle is {', '.join(FORMATS)}, not {format!r}")
    if not (isinstance(hidden, int) and isinstance(inter, int)) or hidden < 64 or inter < 64 or hidden % 64 or inter % 64:
        raise ValueError(f"bundles take hidden and inter multiples of 64, not {hidden} and {inter}")


def _components(format, hidden, inter):
    H, I = hidden, inter
    if format == "bf16":
        return [("gate_up", "bfloat16", (2 * I, H)), ("down", "bfloat16", (H, I))]
    if format.startswith("fp8"):
        if format == "fp8:channel":
            gs, ds = (2 * I,), (H,)
        else:   # gate's grid, then up's
            gs, ds = (2 * math.ceil(I / BLOCK), math.ceil(H / BLOCK)), (math.ceil(H / BLOCK), math.ceil(I / BLOCK))
        return [("gate_up", "float8_e4m3fn", (2 * I, H)), ("gate_up_scale", "float32", gs),
                ("down", "float8_e4m3fn", (H, I)), ("down_scale", "float32", ds)]
    if format in ("nvfp4", "int4"):
        return [("gate_up", "uint8", (2 * I, H // 2)), ("gate_up_scale", "uint8", (2 * I, H // 16)),
                ("gate_up_inverse", "float32", (2 * I,)), ("down", "uint8", (H, I // 2)),
                ("down_scale", "uint8", (H, I // 16)), ("down_inverse", "float32", (H,)), ("input", "float32", (4,))]
    return [("gate_up", "uint8", (2 * I, H // 2)), ("gate_up_scale", "uint8", (2 * I, H // 32)),
            ("down", "uint8", (H, I // 2)), ("down_scale", "uint8", (H, I // 32))]


_SIZES = {"bfloat16": 2, "float8_e4m3fn": 1, "float32": 4, "uint8": 1}
# What a byte component's stored tensor may be, besides raw bytes (uint8, int8): its own dtype.
_BYTE_DTYPES = {("fp8", "gate_up"): "float8_e4m3fn", ("fp8", "down"): "float8_e4m3fn",
                ("e2m1", "gate_up"): "float4_e2m1fn_x2", ("e2m1", "down"): "float4_e2m1fn_x2",
                ("e4m3", "gate_up_scale"): "float8_e4m3fn", ("e4m3", "down_scale"): "float8_e4m3fn",
                ("e8m0", "gate_up_scale"): "float8_e8m0fnu", ("e8m0", "down_scale"): "float8_e8m0fnu"}
_KINDS = {"fp8:channel": ("fp8",), "fp8:block": ("fp8",), "nvfp4": ("e2m1", "e4m3"), "int4": ("e4m3",),
          "mxfp4": ("e2m1", "e8m0")}
_LAYOUTS = {}
_INTERS = {}


def layout(format, hidden, inter):
    """{bundle_bytes, format, hidden, inter, components: {name: (offset, dtype, shape)}} of one expert's bundle: the
    experts:bundles layout (the module docstring gives each component's meaning)."""
    key = (format, hidden, inter)
    if key not in _LAYOUTS:
        _check(format, hidden, inter)
        components, at = {}, 0
        for name, dtype, shape in _components(format, hidden, inter):
            components[name] = (at, dtype, shape)
            at += math.ceil(math.prod(shape) * _SIZES[dtype] / ALIGN) * ALIGN
        _LAYOUTS[key] = {"bundle_bytes": at, "format": format, "hidden": hidden, "inter": inter,
                         "components": components}
    found = _LAYOUTS[key]
    return {**found, "components": dict(found["components"])}


def kernel_layout(found):
    """The layout as the grouped experts kernel takes it: [format, bytes, offsets of _KEYS (-1 absent), hidden, inter]."""
    components = found["components"]
    return ([KERNEL_FORMATS[found["format"]], found["bundle_bytes"]]
            + [components[k][0] if k in components else -1 for k in _KEYS] + [found["hidden"], found["inter"]])


def inter_of(format, hidden, bundle_bytes):
    """The inter width whose bundles of `format` at `hidden` are bundle_bytes long (a store's row), or ValueError."""
    key = (format, hidden, bundle_bytes)
    if key not in _INTERS:
        inter = 64
        while True:
            size = layout(format, hidden, inter)["bundle_bytes"]
            if size == bundle_bytes:
                break
            if size > bundle_bytes:
                raise ValueError(f"no {format} bundle at hidden {hidden} is {bundle_bytes} bytes")
            inter += 64
        _INTERS[key] = inter
    return _INTERS[key]


def _stored(t):
    return t if isinstance(t, dict) else {"weight": t}


def _scalar(t, what):
    import torch
    values = torch.as_tensor(t).float().reshape(-1)
    if values.numel() != 1:
        raise ValueError(f"an expert matrix has one {what}, not {values.numel()}")
    if not (bool(torch.isfinite(values).all()) and float(values[0]) > 0):
        raise ValueError(f"an expert matrix's {what} is a positive number, not {float(values[0])}")
    return values[:1]


def _rows_of(value, rows, what):
    """A global scale (one, or one per row) as each row's inverse [rows] FP32."""
    import torch
    g = torch.as_tensor(value).float().reshape(-1)
    if g.numel() not in (1, rows):
        raise ValueError(f"an expert matrix of {rows} rows has one {what} or one per row, not {g.numel()}")
    return (1.0 / g).expand(rows).contiguous()


def _byte_dtypes(format, name):
    """The dtypes a byte component's tensor may come in: raw bytes, or the component's own dtype."""
    import torch
    allowed = {torch.uint8, torch.int8}
    for kind in _KINDS.get(format, ()):
        own = getattr(torch, _BYTE_DTYPES.get((kind, name), ""), None)
        if own is not None:
            allowed.add(own)
    return allowed


def pack(gate_up, down, format, device=None):
    """One expert's bundle: uint8 [bundle_bytes] on `device` (the weight's own by default) from load()'s arguments.
    Bytes are copied as they are, nothing cast: a bf16 matrix must be bfloat16, a byte component raw bytes or its own
    dtype. Input scales are optional (0 in the bundle: W4A16 only); an experts plugin running W4A4 refuses their
    lack."""
    import torch
    gu, dn = _stored(gate_up), _stored(down)
    weight = torch.as_tensor(gu["weight"])
    device = torch.device(device) if device is not None else weight.device
    if format in ("nvfp4", "int4", "mxfp4"):
        hidden, inter = weight.shape[1] * 2, weight.shape[0] // 2
    else:
        hidden, inter = weight.shape[1], weight.shape[0] // 2
    found = layout(format, hidden, inter)
    out = torch.zeros(found["bundle_bytes"], dtype=torch.uint8, device=device)

    def put(name, value):
        offset, dtype, shape = found["components"][name]
        value = torch.as_tensor(value)
        if dtype in ("uint8", "float8_e4m3fn"):   # codes or byte scales: their bytes as they are
            allowed = _byte_dtypes(format, name)
            if value.dtype not in allowed:
                names = ", ".join(sorted(str(d).replace("torch.", "") for d in allowed))
                raise ValueError(f"{format} {name} comes as {names}, not {str(value.dtype).replace('torch.', '')}: "
                                 "pack casts nothing")
            raw = value.contiguous().view(torch.uint8)
        elif dtype == "bfloat16":
            if value.dtype != torch.bfloat16:
                raise ValueError(f"format bf16 takes bfloat16 matrices, not {str(value.dtype).replace('torch.', '')}: "
                                 "pack casts nothing")
            raw = value.contiguous().view(torch.uint8)
        else:   # FP32 factors: channel and block scales, inverse global scales, input scales
            if not value.is_floating_point():
                raise ValueError(f"{format} {name} is a float tensor, not {str(value.dtype).replace('torch.', '')}")
            raw = value.float().contiguous().view(torch.uint8)
        if tuple(value.shape) != tuple(shape) if dtype != "float32" else value.numel() != math.prod(shape):
            raise ValueError(f"{format} {name} is {list(shape)}, not {list(value.shape)}")
        raw = raw.reshape(-1).to(device)
        out[offset:offset + raw.numel()] = raw

    for prefix, stored, rows in (("gate_up", gu, 2 * inter), ("down", dn, hidden)):
        put(prefix, stored["weight"])
        if format.startswith("fp8"):
            scale = torch.as_tensor(stored["scale"])
            if format == "fp8:channel":
                put(prefix + "_scale", scale.reshape(-1))
            else:
                shape = found["components"][prefix + "_scale"][2]
                if tuple(scale.shape) != tuple(shape):
                    split = (" (gate's 128 x 128 grid, then up's: a grid over the fused rows cannot keep gate and up "
                             "apart at an inter of 64 mod 128)") if prefix == "gate_up" else ""
                    raise ValueError(f"fp8:block {prefix}_scale is {list(shape)}, not {list(scale.shape)}{split}")
                put(prefix + "_scale", scale)
        elif format in ("nvfp4", "int4", "mxfp4"):
            put(prefix + "_scale", stored["scale"])
            if format != "mxfp4":
                g = stored.get("global_scale")
                if g is None and format == "nvfp4":
                    raise ValueError("an nvfp4 expert matrix needs its global_scale")
                put(prefix + "_inverse", _rows_of(1.0 if g is None else g, rows, "global scale"))
    if format in ("nvfp4", "int4"):
        scales = []
        for stored in (gu, dn):
            s = stored.get("input_scale")
            scales.append(float(_scalar(s, "input scale")) if s is not None else 0.0)
        a, b = scales
        put("input", torch.tensor([a, b, 1.0 / a if a else 0.0, 1.0 / b if b else 0.0], dtype=torch.float32))
    return out


def input_counts(store, found, slots=None):
    """(named, carrying): how many bundles of store [S][bytes] (or one bundle [bytes]) slots names (each distinct slot
    >= 0; every row without slots) and how many of them carry both input scales, so W4A4 runs on them. Reads the
    bundles' input components: a host synchronization, for an eager call."""
    import torch
    if "input" not in found["components"]:
        return 0, 0
    offset = found["components"]["input"][0]
    rows = store if store.dim() == 2 else store[None]
    values = rows[:, offset:offset + 16].contiguous().view(torch.float32)
    ok = (values[:, :2] > 0).all(-1)
    if slots is not None:
        ok = ok[torch.unique(slots[slots >= 0].long())]
    return int(ok.numel()), int(ok.sum())


# ------------------------------------------------------------------------------------------- decoding (torch)

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_TABLES = {}


def _table(device, which):
    """Decode tables as FP32: E2M1 codes, INT4 codes, E8M0 bytes (2^(byte - 127), byte 255 NaN, byte 0 2^-127)."""
    import torch
    key = (str(device), which)
    if key not in _TABLES:
        if which == "e2m1":
            values = E2M1
        elif which == "int4":
            values = [float(c if c < 8 else c - 16) for c in range(16)]
        else:
            values = [math.ldexp(1.0, b - 127) for b in range(255)] + [float("nan")]
        _TABLES[key] = torch.tensor(values, dtype=torch.float32, device=device)
    return _TABLES[key]


def _view(bundles, found, name):
    """Component `name` of bundles [count][bytes] as [count, *shape] in its dtype."""
    import torch
    offset, dtype, shape = found["components"][name]
    size = math.prod(shape) * _SIZES[dtype]
    return bundles[:, offset:offset + size].contiguous().view(getattr(torch, dtype)).view(bundles.shape[0], *shape)


def block_rows(prefix, rows, inter, device=None):
    """fp8:block: each row's row of its matrix's scale grid, [rows] int64 (gate/up: gate's grid, then up's)."""
    import torch
    r = torch.arange(rows, device=device)
    if prefix == "gate_up":
        return torch.where(r < inter, r // BLOCK, math.ceil(inter / BLOCK) + (r - inter) // BLOCK)
    return r // BLOCK


def operands(bundles, found):
    """What the grouped experts multiply, from bundles [count][bytes], per matrix (gate_up, then down): (values
    [count][rows][cols] FP32, each exact in BF16: the codes times their block scales, or fp8's codes alone; factors
    [count][rows] or None: fp8 channel scales or 1 / global scales, on each row's FP32 sum; blocks
    [count][rows][ceil(cols / 128)] or None: fp8:block's scales, on each 128-k block's sum)."""
    import torch
    fmt = found["format"]
    out = []
    for prefix in ("gate_up", "down"):
        w = _view(bundles, found, prefix)
        factor = blocks = None
        if fmt == "bf16":
            values = w.float()
        elif fmt.startswith("fp8"):
            values = w.float()
            s = _view(bundles, found, prefix + "_scale")
            if fmt == "fp8:channel":
                factor = s
            else:
                blocks = s.index_select(1, block_rows(prefix, values.shape[1], found["inter"], s.device))
        else:
            table = _table(w.device, "int4" if fmt == "int4" else "e2m1")
            codes = torch.stack([table[(w & 15).long()], table[(w >> 4).long()]], -1).reshape(w.shape[0], w.shape[1], -1)
            s = _view(bundles, found, prefix + "_scale")
            if fmt == "mxfp4":
                scale = _table(w.device, "e8m0")[s.long()].repeat_interleave(32, 2)
            else:
                scale = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, 2)
                factor = _view(bundles, found, prefix + "_inverse")
            values = codes * scale
        out.append((values, factor, blocks))
    return tuple(out)


def matrices(bundles, found):
    """The decoded values of bundles [count][bytes]: (gate_up [count][2I][H], down [count][H][I]) FP32, each weight
    its stored value (operands times their block scales and row factors)."""
    out = []
    for values, factor, blocks in operands(bundles, found):
        if blocks is not None:
            values = values * blocks.repeat_interleave(BLOCK, 2)[:, :, :values.shape[2]]
        if factor is not None:
            values = values * factor[:, :, None]
        out.append(values)
    return tuple(out)


def inputs(bundles, found):
    """[count][4] FP32: each bundle's gate/up and down input scales and their inverses (nvfp4, int4)."""
    return _view(bundles, found, "input")
