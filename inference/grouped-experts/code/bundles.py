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
    int4_zp:G    INT4 with a zero point a group of G inputs (AWQ and GPTQ in crit's rows, the grouped experts kernel's
                 format 6): gate_up uint8 [2I][H / 2] (unsigned codes, element 2k in the low nibble of byte k);
                 gate_up_scale float16 [2I][H / G]; gate_up_inverse uint8 [2I][H / G] (the zero points, 0 to 16);
                 down, down_scale, down_inverse alike over I; a value (code - zero) * scale. G a multiple of 16
                 dividing H and I

pack() takes what an experts plugin's load() takes: per matrix a bfloat16 tensor (bf16) or the stored tensors by role,
{weight, scale, global_scale, input_scale}, as the projections slot's load takes them; for int4_zp:G {weight (the codes),
scale, zeros} in crit's rows, or a checkpoint's AWQ or GPTQ tensors with format awq or gptq ({qweight, qzeros, scales},
GPTQ's g_idx and zero_offset as the projections slot takes them; gate/up as one matrix of 2I outputs, gate's first, or
a (gate, up) pair), repacked exactly into an int4_zp:G bundle (bundle_format names it; act order is refused: a
bundle's inputs are the activations' order). It copies bytes as they are and
casts nothing: a bf16 matrix is bfloat16, a byte component (codes, E4M3 or E8M0 scales) comes as raw bytes (uint8,
int8) or in its own dtype (float8_e4m3fn, float4_e2m1fn_x2, float8_e8m0fnu).
"""
import math

FORMATS = ("bf16", "fp8:channel", "fp8:block", "nvfp4", "int4", "mxfp4", "int4_zp")   # int4_zp as int4_zp:<group>
KERNEL_FORMATS = {"bf16": 0, "fp8:channel": 1, "fp8:block": 2, "nvfp4": 3, "int4": 4, "mxfp4": 5, "int4_zp": 6}
STORED_INT4 = ("awq", "gptq")   # checkpoint formats pack() repacks into int4_zp:<group> bundles
ALIGN = 16
BLOCK = 128   # fp8:block's scale block

_KEYS = ("gate_up", "gate_up_scale", "gate_up_inverse", "down", "down_scale", "down_inverse", "input")


def parse_format(format):
    """(base format, group): int4_zp:<group> -> ("int4_zp", group); any other format -> (format, None)."""
    base, _, group = str(format).partition(":")
    if base == "int4_zp":
        if not group.isdigit() or int(group) <= 0:
            raise ValueError(f"an int4_zp bundle names its group, int4_zp:<group> (e.g. int4_zp:128), not {format!r}")
        return base, int(group)
    return format, None


def known(format):
    """Whether `format` names a bundle layout (int4_zp with its group)."""
    try:
        base, group = parse_format(format)
    except ValueError:
        return False
    return base in FORMATS and (base != "int4_zp" or group is not None) and format != "int4_zp"


def _check(format, hidden, inter):
    if not known(format):
        raise ValueError(f"an expert bundle is {', '.join(f for f in FORMATS if f != 'int4_zp')} or "
                         f"int4_zp:<group>, not {format!r}")
    if not (isinstance(hidden, int) and isinstance(inter, int)) or hidden < 64 or inter < 64 or hidden % 64 or inter % 64:
        raise ValueError(f"bundles take hidden and inter multiples of 64, not {hidden} and {inter}")
    _, group = parse_format(format)
    if group is not None and (group % 16 or hidden % group or inter % group):
        raise ValueError(f"an int4_zp bundle's group is a multiple of 16 dividing hidden and inter, not {group} at "
                         f"{hidden} and {inter}")


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
    base, G = parse_format(format)
    if base == "int4_zp":
        return [("gate_up", "uint8", (2 * I, H // 2)), ("gate_up_scale", "float16", (2 * I, H // G)),
                ("gate_up_inverse", "uint8", (2 * I, H // G)), ("down", "uint8", (H, I // 2)),
                ("down_scale", "float16", (H, I // G)), ("down_inverse", "uint8", (H, I // G))]
    if format in ("nvfp4", "int4"):
        return [("gate_up", "uint8", (2 * I, H // 2)), ("gate_up_scale", "uint8", (2 * I, H // 16)),
                ("gate_up_inverse", "float32", (2 * I,)), ("down", "uint8", (H, I // 2)),
                ("down_scale", "uint8", (H, I // 16)), ("down_inverse", "float32", (H,)), ("input", "float32", (4,))]
    return [("gate_up", "uint8", (2 * I, H // 2)), ("gate_up_scale", "uint8", (2 * I, H // 32)),
            ("down", "uint8", (H, I // 2)), ("down_scale", "uint8", (H, I // 32))]


_SIZES = {"bfloat16": 2, "float8_e4m3fn": 1, "float32": 4, "uint8": 1, "float16": 2}
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
    base, group = parse_format(found["format"])
    return ([KERNEL_FORMATS[base], found["bundle_bytes"]]
            + [components[k][0] if k in components else -1 for k in _KEYS] + [found["hidden"], found["inter"]]
            + ([group] if group is not None else []))


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


AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)   # nibble p of an AWQ word holds output AWQ_ORDER[p]


def _nibbles(words, order=None):
    """int32 words [..., j] -> values 0..15 [..., 8 j + e], nibble p holding element order[p] (in sequence without)."""
    import torch
    w = torch.as_tensor(words).to(torch.int64) & 0xFFFFFFFF
    parts = [(w >> (4 * p)) & 15 for p in range(8)]
    if order is not None:
        placed = [None] * 8
        for p, e in enumerate(order):
            placed[e] = parts[p]
        parts = placed
    return torch.stack(parts, -1).reshape(*w.shape[:-1], -1)


def repack(stored, format):
    """One AWQ or GPTQ matrix ({qweight, qzeros, scales}, GPTQ's g_idx and zero_offset: 1 by default, for checkpoints
    that store zero - 1) in crit's INT4 rows, exactly: {weight: codes uint8 [out][in / 2] (element 2k in the low
    nibble of byte k), scale: FP16 [out][in / group], zeros: uint8 [out][in / group]} and the group. GPTQ's act order
    (a g_idx other than k // group) is refused: an expert's inputs are the activations' order (the kernel reads them
    as they are) and the down matrix's inputs are the kernel's own intermediate."""
    import torch
    qweight, qzeros = torch.as_tensor(stored["qweight"]), torch.as_tensor(stored["qzeros"])
    scales = torch.as_tensor(stored["scales"])
    if scales.dtype != torch.float16:
        raise ValueError(f"{format} scales are float16, not {str(scales.dtype).replace('torch.', '')}: pack casts "
                         "nothing")
    k, n = (qweight.shape[0], qweight.shape[1] * 8) if format == "awq" else (qweight.shape[0] * 8, qweight.shape[1])
    groups = scales.shape[0]
    if groups < 1 or k % groups or tuple(scales.shape) != (groups, n) or tuple(qzeros.shape) != (groups, n // 8):
        raise ValueError(f"an {format} matrix of {n} x {k} has scales [{k} / group][{n}] and qzeros [{k} / group]"
                         f"[{n // 8}], not {list(scales.shape)} and {list(qzeros.shape)}")
    group = k // groups
    if format == "awq":
        q, zeros = _nibbles(qweight, AWQ_ORDER), _nibbles(qzeros, AWQ_ORDER)
    else:
        offset = stored.get("zero_offset", 1)
        offset = int(torch.as_tensor(offset).reshape(-1)[0])
        q, zeros = _nibbles(qweight.t().contiguous()).t(), _nibbles(qzeros) + offset
        g_idx = stored.get("g_idx")
        if g_idx is not None and not torch.equal(torch.as_tensor(g_idx).to(q.device, torch.int64),
                                                 torch.arange(k, device=q.device) // group):
            raise ValueError("a gptq expert matrix with act order (g_idx other than input // group) is not packed: "
                             "the grouped experts read their inputs in the activations' order")
    if int(zeros.min()) < 0 or int(zeros.max()) > 16:
        raise ValueError(f"an {format} matrix's zero points are 0 to 16, not {int(zeros.min())} to {int(zeros.max())}")
    rows = q.t().contiguous()
    codes = (rows[:, 0::2] | (rows[:, 1::2] << 4)).to(torch.uint8)
    return {"weight": codes, "scale": scales.t().contiguous(), "zeros": zeros.t().contiguous().to(torch.uint8)}, group


def _repacked(matrix, format):
    """repack() of a matrix, or of a (gate, up) pair stacked by rows (gate's first)."""
    import torch
    if isinstance(matrix, (tuple, list)):
        parts = [repack(m, format) for m in matrix]
        if len({g for _, g in parts}) != 1:
            raise ValueError(f"an {format} expert's gate and up matrices share one group, not {[g for _, g in parts]}")
        return {name: torch.cat([p[name] for p, _ in parts]) for name in ("weight", "scale", "zeros")}, parts[0][1]
    return repack(matrix, format)


def bundle_format(gate_up, down, format):
    """The bundle format pack() writes for these matrices: int4_zp:<group> for awq and gptq ones (their scales give
    the group), else `format`."""
    import torch
    if format not in STORED_INT4:
        return format
    groups = set()
    for matrix in (*(gate_up if isinstance(gate_up, (tuple, list)) else (gate_up,)), down):
        qweight, scales = torch.as_tensor(matrix["qweight"]), torch.as_tensor(matrix["scales"])
        k = qweight.shape[0] if format == "awq" else qweight.shape[0] * 8
        groups.add(k // max(1, scales.shape[0]))
    if len(groups) != 1:
        raise ValueError(f"an {format} expert's matrices share one group, not {sorted(groups)}")
    return f"int4_zp:{groups.pop()}"


def pack(gate_up, down, format, device=None):
    """One expert's bundle: uint8 [bundle_bytes] on `device` (the weight's own by default) from load()'s arguments.
    Bytes are copied as they are, nothing cast: a bf16 matrix must be bfloat16, a byte component raw bytes or its own
    dtype. Input scales are optional (0 in the bundle: W4A16 only); an experts plugin running W4A4 refuses their
    lack. awq and gptq matrices are repacked (repack) into an int4_zp:<group> bundle (bundle_format)."""
    import torch
    if format in STORED_INT4:
        named = bundle_format(gate_up, down, format)
        (gate_up, _), (down, _) = _repacked(gate_up, format), _repacked(down, format)
        format = named
    base, _ = parse_format(format)
    gu, dn = _stored(gate_up), _stored(down)
    weight = torch.as_tensor(gu["weight"])
    device = torch.device(device) if device is not None else weight.device
    if base in ("nvfp4", "int4", "mxfp4", "int4_zp"):
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
        elif dtype == "float16":   # int4_zp's scales as the checkpoint stores them
            if value.dtype != torch.float16:
                raise ValueError(f"{format} {name} is float16, not {str(value.dtype).replace('torch.', '')}: "
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
        elif base == "int4_zp":
            put(prefix + "_scale", stored["scale"])
            put(prefix + "_inverse", stored["zeros"])
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
    [count][rows][cols] FP32, each exact in BF16 (int4_zp's (code - zero) * scale in two BF16 parts): the codes times
    their block scales, or fp8's codes alone; factors [count][rows] or None: fp8 channel scales or 1 / global scales,
    on each row's FP32 sum; blocks
    [count][rows][ceil(cols / 128)] or None: fp8:block's scales, on each 128-k block's sum)."""
    import torch
    fmt = found["format"]
    out = []
    for prefix in ("gate_up", "down"):
        w = _view(bundles, found, prefix)
        factor = blocks = None
        if fmt == "bf16":
            values = w.float()
        elif fmt.startswith("int4_zp"):   # (code - zero) * scale, exact in FP32 (two BF16 parts: fallback's split)
            group = parse_format(fmt)[1]
            q = torch.stack([(w & 15).float(), (w >> 4).float()], -1).reshape(w.shape[0], w.shape[1], -1)
            z = _view(bundles, found, prefix + "_inverse").float().repeat_interleave(group, 2)
            values = (q - z) * _view(bundles, found, prefix + "_scale").float().repeat_interleave(group, 2)
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
