"""Expert bundles in the experts slot's layout (experts:bundles in docs/PLUGINS.md, the experts plugin's layout()):
one expert's gate/up and down matrices in one contiguous block, components 16-byte aligned, row-major as the
checkpoint stores them, so a CPU tier reads a bundle in place where a bundle store keeps it. The formats the CPU tier
computes:

    nvfp4  gate_up uint8 [2I][H / 2] (E2M1, element 2k in the low nibble of byte k); gate_up_scale uint8 [2I][H / 16]
           (E4M3); gate_up_inverse float32 [2I] (1 / each row's global scale, crit's divisor convention); down uint8
           [H][I / 2]; down_scale uint8 [H][I / 16]; down_inverse float32 [H]; input float32 [4] = (gate_up input_scale,
           down input_scale, 1 / each), 0 when the checkpoint has none
    int4   as nvfp4 with INT4 two's complement nibbles; the inverses 1 unless a global scale is given

The arithmetic the CPU tier runs on them (the declared order, the device's bits): nvfp4 W4A4 (inputs E2M1 x 2 under
input[0], the intermediate under input[1], post 0.25 * inverse per row); int4 W4A8 (INT8 per 32, post inverse).
"""
import math

ALIGN = 16
FORMATS = ("nvfp4", "int4")
ORDER = ("gate_up", "gate_up_scale", "gate_up_inverse", "down", "down_scale", "down_inverse", "input")
OFFSETS = ORDER[:6]   # the tier's set_expert order


def _up(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def layout(format, hidden, inter):
    """{"bundle_bytes", "format", "hidden", "inter", "components": {name: (offset, dtype, shape)}}."""
    if format not in FORMATS:
        raise ValueError(f"the CPU tier's bundles are {', '.join(FORMATS)}, not {format!r}")
    if hidden % 64 or inter % 64 or hidden < 64 or inter < 64:
        raise ValueError("hidden and inter are multiples of 64")
    shapes = {"gate_up": ("uint8", (2 * inter, hidden // 2)), "gate_up_scale": ("uint8", (2 * inter, hidden // 16)),
              "gate_up_inverse": ("float32", (2 * inter,)), "down": ("uint8", (hidden, inter // 2)),
              "down_scale": ("uint8", (hidden, inter // 16)), "down_inverse": ("float32", (hidden,)),
              "input": ("float32", (4,))}
    components, offset = {}, 0
    for name in ORDER:
        dtype, shape = shapes[name]
        components[name] = (offset, dtype, shape)
        offset = _up(offset + math.prod(shape) * (4 if dtype == "float32" else 1))
    return {"bundle_bytes": offset, "format": format, "hidden": hidden, "inter": inter, "components": components}


def _bytes(tensor):
    import torch
    t = tensor.detach().cpu().contiguous()
    if t.dtype != torch.uint8:
        t = t.view(torch.uint8) if t.element_size() == 1 else t
    return t


def _scalar(value):
    import torch
    if value is None:
        return None
    values = torch.as_tensor(value).float().reshape(-1)
    if values.numel() != 1:
        raise ValueError("an input scale is one number")
    return float(values[0])


def _inverse(matrix, rows, format):
    import torch
    g = matrix.get("global_scale")
    if g is None:
        if format == "nvfp4":
            raise ValueError("an nvfp4 matrix has its global_scale")
        return torch.ones(rows, dtype=torch.float32)   # int4's global scale is optional (the format's definition)
    g = torch.as_tensor(g).detach().cpu().float().reshape(-1)
    if g.numel() not in (1, rows):
        raise ValueError(f"a matrix of {rows} rows has one global scale or one per row, not {g.numel()}")
    return (1.0 / g).expand(rows).contiguous()


def pack(gate_up, down, format):
    """The CPU uint8 bundle of one expert from its stored tensors by role ({weight, scale, global_scale, input_scale},
    as projections' load takes them; global and input scales divisors)."""
    import torch
    if not isinstance(gate_up, dict) or not isinstance(down, dict):
        raise ValueError(f"a {format} expert is its stored tensors by role ({{weight, scale, global_scale, input_scale}})")
    gu_codes, dn_codes = _bytes(gate_up["weight"]), _bytes(down["weight"])
    rows2, hidden = gu_codes.shape[0], gu_codes.shape[1] * 2
    inter = rows2 // 2
    plan = layout(format, hidden, inter)
    parts = {"gate_up": gu_codes, "gate_up_scale": _bytes(gate_up["scale"]),
             "gate_up_inverse": _inverse(gate_up, 2 * inter, format), "down": dn_codes,
             "down_scale": _bytes(down["scale"]), "down_inverse": _inverse(down, hidden, format)}
    gi, di = _scalar(gate_up.get("input_scale")), _scalar(down.get("input_scale"))
    parts["input"] = torch.tensor([gi or 0.0, di or 0.0, 1.0 / gi if gi else 0.0, 1.0 / di if di else 0.0],
                                  dtype=torch.float32)
    bundle = torch.zeros(plan["bundle_bytes"], dtype=torch.uint8)
    for name, (offset, dtype, shape) in plan["components"].items():
        part = parts[name]
        if tuple(part.shape) != tuple(shape):
            raise ValueError(f"{format} {name} is {list(shape)}, not {list(part.shape)}")
        raw = part.contiguous().view(torch.uint8).reshape(-1)
        bundle[offset:offset + raw.numel()] = raw
    return bundle


def view(bundle, plan, name):
    """A component of a CPU bundle as a tensor of its dtype and shape (a view of the bundle's bytes)."""
    import torch
    offset, dtype, shape = plan["components"][name]
    size = math.prod(shape) * (4 if dtype == "float32" else 1)
    raw = bundle[offset:offset + size]
    return (raw.view(torch.float32) if dtype == "float32" else raw).reshape(shape)


def arithmetic(format):
    """The declared arithmetic the CPU tier runs on a format: weights, factor, input / intermediate quantizers."""
    if format == "nvfp4":
        return {"weights": "e2m1x2", "factor": 0.25, "input_kind": "e2m1x2", "inter_kind": "e2m1x2", "block": 16}
    if format == "int4":
        return {"weights": "int4", "factor": 1.0, "input_kind": "int8", "inter_kind": "int8", "block": 32}
    raise ValueError(f"the CPU tier computes {', '.join(FORMATS)}, not {format!r}")


def globals_of(bundle, plan):
    """(gate_up input global, down input global) of an nvfp4 bundle: input[0], input[1]."""
    values = view(bundle, plan, "input")
    return float(values[0]), float(values[1])
