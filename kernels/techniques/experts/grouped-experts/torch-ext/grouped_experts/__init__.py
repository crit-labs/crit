"""Routed SwiGLU experts over a slot store of expert bundles, for decode rows and prompt chunks alike: crit's kernel
grouped-experts, which crit's grouped-experts plugin (inference/grouped-experts) loads, built on this GPU with the
engine's tuning or the default knobs.

An expert is one bundle in crit's experts:bundles layout; a call names each bundle's components by byte offset (a
layout: [format, bytes, gate_up, gate_up_scale, gate_up_inverse, down, down_scale, down_inverse, input, hidden,
inter], for int4_zp its group after them (INT4 with a zero point a group, AWQ and GPTQ in crit's rows, FP16 scales in
*_scale and the zero points in *_inverse), and for a block not gated by SiLU the group (0 for none) and then its
activation (ACTIVATIONS), the store holding the bundles ([slots][bytes] uint8) and each expert's slot (slots [experts] int32: >= 0 a
slot, negative skipped). The routes are grouped by expert on the device (no host synchronization), every expert's
gate/up and down rows run on BF16 tensor cores with the weights decoded exactly, and each token's routes are summed in
route order. A route's result is the same bits whatever rows, chunk or item size share its call, so decode rows and
prompt chunks compute one function, and a call at a fixed token count can be captured in a CUDA graph once its
Workspace exists. The tuning picks warps, row tiles, weight blocks in flight and the staged chunk per item width,
format, pass, shape and token count; no knob changes a route's bits.

    ws = Workspace(tokens, top, experts, hidden, inter, x.device)
    out = forward(x, ids, weights, store, slots, layout, ws)          # [tokens][hidden] FP32
    y = route_outputs(x, ids, store, slots, layout, ws)               # unweighted rows, [tokens * top][hidden]
    out = combine(y, ids, weights, slots, experts)                    # fmaf(weight, row, sum) in route order

arithmetic="fp4" (nvfp4 bundles with input scales, hidden and inter multiples of 128, where fp4_available()): W4A4 on
the block-scaled FP4 tensor cores, every route's inputs and intermediate quantized by crit's declared NVFP4 quantizer
to E2M1 codes and UE4M3 scales, one MMA chain an output over K in order: decode and prompt items again give a route
the same bits (structures fp4-decode and fp4-prompt), another arithmetic than the BF16 path's a4. arithmetic="fp8"
(fp8:channel and fp8:block bundles, hidden and inter multiples of 128, where fp8_available()): W8A8 on the E4M3 tensor
cores, every route's inputs and intermediate quantized per 128 values of a row in E4M3 at amax / 448 (crit's declared
group quantizer), each 128-k block's sum folded by the activation's and the weight's scales in block order (structures
fp8-decode and fp8-prompt: again a route's bits whatever its item). arithmetic="fp4-row" and "fp4-row2" (nvfp4
bundles with or without input scales): the same FP4 MMAs at a scale of each route's own, g = 6 * 448 / amax of its row
(the inputs', then the intermediate's, which passes between the two as BF16): "fp4-row" one FP4 term a value (W4A4),
"fp4-row2" two (the first term's remainder at the same g, about 8 bits a value, twice the MMAs over the same weights).
"""
import math

import torch

from ._ops import ops

__all__ = ["ACTIVATIONS", "ARITHMETICS", "caps_layout", "FORMATS", "MAX_EXPERTS", "MAX_ITEMS", "SMALL_PLAN", "Workspace",
           "combine", "forward", "fp4_available", "fp8_available", "quantize_e2m1x2", "route_outputs", "width_for"]

FORMATS = {"bf16": 0, "fp8:channel": 1, "fp8:block": 2, "nvfp4": 3, "int4": 4, "mxfp4": 5, "int4_zp": 6, "mxfp8": 7}
# A block's gated product, a layout's 13th integer (after the int4_zp group, 0 for none): silu(gate) * up, crit's
# declared SwiGLU and the default, gelu(gate) * up with GELU's tanh form, or SwiGLU with its linear gate factor and
# its up capped by a tanh each, the two caps a layout's 14th and 15th integers (caps_layout) (the tensor arithmetic
# only).
ACTIVATIONS = {"silu": 0, "gelu_tanh": 1, "softcapped_silu": 2}


def caps_layout(caps):
    """The layout integers of softcapped_silu's (gate cap, up cap): each an FP32's bits."""
    import struct
    return [struct.unpack("<i", struct.pack("<f", float(cap)))[0] for cap in caps]
SMALL_PLAN = 1024   # routes the one-block plan takes; larger calls group their routes from a stable sort
MAX_ITEMS = 65535   # a call's items (the launches' second grid dimension)
MAX_EXPERTS = 65535   # the experts a call may name, in either plan


def width_for(routes, experts):
    """Routes per item: 32 once a call averages 8 or more routes per expert (prompt chunks), else 8 (decode rows); 64
    is a choice of the caller's (Workspace(width=64); FP4 items of 64 run fp4-prompt, or fp4-tile with tile=True). The
    width changes the speed only, never a route's bits."""
    return 32 if routes >= 8 * experts else 8


class Workspace:
    """The buffers of calls of `tokens` tokens with `top` routes each over `experts` experts of a [hidden][inter]
    shape: the items, the intermediate h and the unweighted route rows y. Allocated once; reuse it for every call of
    the shape (and capture after one call). rows: where its row buffers come from, rows(name, count, width, dtype)
    -> [count][width] of dtype on the device (any contents), for a caller whose workspaces of every shape share
    their rows (calls of one stream run one after another); by default each workspace has its own. tile: FP4 items
    of 64 routes run the block tile (structure fp4-tile) in place of fp4-prompt's 64-route form, the same bits."""

    def __init__(self, tokens, top, experts, hidden, inter, device, width=None, rows=None, tile=False):
        routes = tokens * top
        if not 1 <= experts <= MAX_EXPERTS:
            raise ValueError(f"a call names 1 to {MAX_EXPERTS} experts, not {experts}")
        self.tokens, self.top, self.experts, self.routes = tokens, top, experts, routes
        self.width = width or width_for(routes, experts)
        self.tile = bool(tile) and self.width == 64
        if self.width not in (8, 32, 64):
            raise ValueError("items hold 8, 32 or 64 routes")
        self.capacity = math.ceil(routes / self.width) + min(experts, routes)
        if self.capacity > MAX_ITEMS:
            raise ValueError(f"{routes} routes make more than {MAX_ITEMS} items: split the call")
        ints = dict(dtype=torch.int32, device=device)
        self.item_slot = torch.zeros(self.capacity, **ints)
        self.item_routes = torch.full((self.capacity, self.width), -1, **ints)
        self.item_count = torch.zeros(1, **ints)
        self._rows = rows or (lambda name, count, width, dtype: torch.zeros(count, width, dtype=dtype, device=device))
        self.h = self._rows("h", routes, inter, torch.bfloat16)
        self.y = self._rows("y", routes, hidden, torch.float32)
        self.ones = torch.ones(routes, **ints) if routes > SMALL_PLAN else None
        self.hidden, self.inter, self.device, self._declared, self.xa = hidden, inter, device, None, None
        self._fp4 = self._fp8 = None
        self._fp4_rows = {}

    def declared(self, layout):
        """The declared arithmetic's buffers (made at its first call): quantized inputs and intermediate, gate/up rows."""
        if self._declared is None:
            block = 32 if layout[0] == FORMATS["int4"] else 16
            n, H, I, dev = self.routes, self.hidden, self.inter, self.device
            self._declared = {"x": self._rows("declared.x", n, H, torch.int8),
                              "xs": self._rows("declared.xs", n, H // block, torch.float32),
                              "gu": self._rows("declared.gu", n, 2 * I, torch.float32),
                              "h": self._rows("declared.h", n, I, torch.int8),
                              "hs": self._rows("declared.hs", n, I // block, torch.float32)}
        return self._declared

    def fp4(self):
        """The FP4 arithmetic's buffers (made at its first call): the routes' input codes and scales, the
        intermediate's."""
        if self._fp4 is None:
            n, H, I = self.routes, self.hidden, self.inter
            self._fp4 = {"x": self._rows("fp4.x", n, H // 2, torch.uint8),
                         "xs": self._rows("fp4.xs", n, H // 16, torch.uint8),
                         "h": self._rows("fp4.h", n, I // 2, torch.uint8),
                         "hs": self._rows("fp4.hs", n, I // 16, torch.uint8)}
        return self._fp4

    def fp4_rows(self, terms):
        """The FP4 row-scale arithmetic's buffers (made at its first call): the routes' input codes, scales and 1 / g
        (terms a row), the intermediate's (h itself is the workspace's)."""
        if terms not in self._fp4_rows:
            n, H, I = self.routes, self.hidden, self.inter
            name = f"fp4r{terms}"
            self._fp4_rows[terms] = {"x": self._rows(name + ".x", n, terms * H // 2, torch.uint8),
                                     "xs": self._rows(name + ".xs", n, terms * H // 16, torch.uint8),
                                     "xr": self._rows(name + ".xr", n, 1, torch.float32).view(-1),
                                     "h": self._rows(name + ".h", n, terms * I // 2, torch.uint8),
                                     "hs": self._rows(name + ".hs", n, terms * I // 16, torch.uint8),
                                     "hr": self._rows(name + ".hr", n, 1, torch.float32).view(-1)}
        return self._fp4_rows[terms]

    def fp8(self):
        """The FP8 arithmetic's buffers (made at its first call): the routes' input codes and group scales, the
        intermediate's (h itself is the workspace's)."""
        if self._fp8 is None:
            n, H, I = self.routes, self.hidden, self.inter
            self._fp8 = {"x": self._rows("fp8.x", n, H, torch.uint8), "xs": self._rows("fp8.xs", n, H // 128, torch.float32),
                         "h": self._rows("fp8.h", n, I, torch.uint8), "hs": self._rows("fp8.hs", n, I // 128, torch.float32)}
        return self._fp8


def _plan(ids, slots, ws):
    flat = ids.reshape(-1)
    if flat.numel() != ws.routes:
        raise ValueError(f"the workspace takes {ws.routes} routes, not {flat.numel()}")
    if ws.routes <= SMALL_PLAN:
        ops.plan(flat, slots, ws.experts, ws.item_slot, ws.item_routes, ws.item_count)
        return
    experts = ws.experts
    valid = (flat >= 0) & (flat < experts)
    if slots is not None:
        valid &= slots[flat.clamp(0, experts - 1)] >= 0
    keys = torch.where(valid, flat.long(), experts)
    order = torch.argsort(keys, stable=True)
    counts = torch.zeros(experts + 1, dtype=torch.int32, device=flat.device).scatter_add_(0, keys, ws.ones)
    starts = torch.cumsum(counts, 0, dtype=torch.int32) - counts
    item_end = torch.cumsum((counts[:experts] + ws.width - 1) // ws.width, 0, dtype=torch.int32)
    ops.items(order, counts, starts, item_end, slots, experts, ws.item_slot, ws.item_routes, ws.item_count)


ARITHMETICS = ("tensor", "declared", "fp4", "fp4-row", "fp4-row2", "fp8")
ROW_TERMS = {"fp4-row": 1, "fp4-row2": 2}   # FP4 at each route's own scale: its operands' terms
_probed = {}


def _probe(kind, device):
    device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    key = (kind, str(device))
    if key not in _probed:
        with torch.cuda.device(device):
            _probed[key] = bool(ops.fp4_available() if kind == "fp4" else ops.fp8_available())
    return _probed[key]


def fp4_available(device=None):
    """Whether this build multiplies on the device's block-scaled FP4 tensor cores (arithmetic "fp4": its code for
    the device has mma.sync kind::mxf4nvf4, the sm_120 family's specific targets)."""
    return _probe("fp4", device)


def fp8_available(device=None):
    """Whether this build multiplies on the device's E4M3 tensor cores (arithmetic "fp8": 8.9 on)."""
    return _probe("fp8", device)


def route_outputs(x, ids, store, slots, layout, ws, a4=False, arithmetic="tensor"):
    """Each planned route's unweighted output row, y [tokens * top][hidden] FP32 (ws.y; rows of skipped routes keep
    what they held). x [tokens][hidden] BF16 contiguous; ids [tokens][top]; store uint8 [slots][bytes]; slots int32
    [experts] or None (expert e in slot e); a4: nvfp4 W4A4 at the bundles' input scales. arithmetic: "tensor" (BF16
    tensor cores, every format), "declared" (crit's integer block order, the CPU tier's bits: nvfp4 W4A4, int4 W4A8)
    "fp4" (nvfp4 W4A4 on the FP4 tensor cores, a4 implied) or "fp8" (fp8 W8A8 on the E4M3 tensor cores): see the
    module's docstring."""
    if ids.shape != (ws.tokens, ws.top):
        raise ValueError(f"the workspace takes ids [{ws.tokens}][{ws.top}], not {list(ids.shape)}")
    if arithmetic not in ARITHMETICS:
        raise ValueError(f"arithmetic is {' or '.join(ARITHMETICS)}, not {arithmetic!r}")
    if arithmetic.startswith("fp4") and (layout[0] != FORMATS["nvfp4"] or ws.hidden % 128 or ws.inter % 128):
        raise ValueError(f"the {arithmetic} arithmetic takes nvfp4 bundles (fp4: with input scales), hidden and inter "
                         "multiples of 128")
    if arithmetic.startswith("fp4") and not fp4_available(x.device):
        raise ValueError("the fp4 arithmetic runs on the block-scaled FP4 tensor cores (the sm_120 family): this "
                         "device's code has none")
    if arithmetic == "fp8" and (layout[0] not in (FORMATS["fp8:channel"], FORMATS["fp8:block"]) or ws.hidden % 128 or
                                ws.inter % 128):
        raise ValueError("the fp8 arithmetic takes fp8:channel and fp8:block bundles, hidden and inter multiples of 128")
    if arithmetic == "fp8" and not fp8_available(x.device):
        raise ValueError("the fp8 arithmetic runs on the E4M3 tensor cores (8.9 on): this device's code has none")
    _plan(ids, slots, ws)
    items = (ws.item_slot, ws.item_routes, ws.item_count)
    if arithmetic == "fp4":
        f = ws.fp4()
        ops.fp4_inputs(x, ws.top, store, list(layout), *items, f["x"], f["xs"])
        ops.fp4_gate_up(f["x"], f["xs"], store, list(layout), *items, f["h"], f["hs"], ws.tokens, int(ws.tile))
        ops.fp4_down(f["h"], f["hs"], store, list(layout), *items, ws.y, ws.tokens, int(ws.tile))
        return ws.y
    if arithmetic in ROW_TERMS:   # each route's own scale: the inputs, then h, quantized as they come
        terms, f = ROW_TERMS[arithmetic], ws.fp4_rows(ROW_TERMS[arithmetic])
        ops.fp4_rows(x, ws.top, *items, terms, f["x"], f["xs"], f["xr"])
        ops.fp4_gate_up_rows(f["x"], f["xs"], f["xr"], terms, store, list(layout), *items, ws.h, ws.tokens,
                             int(ws.tile))
        ops.fp4_rows(ws.h, 1, *items, terms, f["h"], f["hs"], f["hr"])
        ops.fp4_down_rows(f["h"], f["hs"], f["hr"], terms, store, list(layout), *items, ws.y, ws.tokens,
                          int(ws.tile))
        return ws.y
    if arithmetic == "fp8":
        f = ws.fp8()
        ops.fp8_inputs(x, ws.top, *items, f["x"], f["xs"])
        ops.fp8_gate_up(f["x"], f["xs"], store, list(layout), *items, ws.h, ws.tokens)
        ops.fp8_inputs(ws.h, 1, *items, f["h"], f["hs"])
        ops.fp8_down(f["h"], f["hs"], store, list(layout), *items, ws.y, ws.tokens)
        return ws.y
    if arithmetic == "declared":
        if len(layout) >= 13 and layout[12]:
            raise ValueError("the declared arithmetic is SwiGLU's (its host twin computes no other gate): a block "
                             "gated by GELU runs the tensor arithmetic")
        d = ws.declared(layout)
        ops.declared_inputs(x, ws.top, store, list(layout), *items, d["x"], d["xs"])
        ops.declared_rows(d["x"], d["xs"], store, list(layout), True, *items, d["gu"])
        ops.declared_intermediate(d["gu"], store, list(layout), *items, d["h"], d["hs"])
        ops.declared_rows(d["h"], d["hs"], store, list(layout), False, *items, ws.y)
        return ws.y
    if a4:   # the W4A4 operands, quantized once per route at its expert's input scale
        if ws.xa is None:
            ws.xa = ws._rows("xa", ws.routes, ws.hidden, torch.bfloat16)
        ops.a4_inputs(x, ws.top, store, list(layout), *items, ws.xa)
        ops.gate_up(ws.xa, 1, store, list(layout), True, *items, ws.h, ws.tokens)
    else:
        ops.gate_up(x, ws.top, store, list(layout), bool(a4), *items, ws.h, ws.tokens)
    ops.down(ws.h, store, list(layout), bool(a4), *items, ws.y, ws.tokens)
    return ws.y


def combine(y, ids, weights, slots, experts, skip_negative=True, out=None):
    """out [tokens][hidden] FP32: for each token, sum = fmaf(weights[t][j], y[t * top + j], sum) over its routes in
    order, those with 0 <= id < experts (and slots[id] >= 0 when skip_negative and slots are given)."""
    tokens = ids.shape[0]
    out = torch.empty(tokens, y.shape[1], dtype=torch.float32, device=y.device) if out is None else out
    ops.combine(y, ids, weights, slots, experts, bool(skip_negative), out)
    return out


def forward(x, ids, weights, store, slots, layout, ws, a4=False, out=None, arithmetic="tensor"):
    """The routed SwiGLU block: out [tokens][hidden] FP32, routes to negative slots skipped."""
    y = route_outputs(x, ids, store, slots, layout, ws, a4, arithmetic)
    return combine(y, ids, weights, slots, ws.experts, True, out)


def quantize_e2m1x2(x, global_scale):
    """crit's declared NVFP4 activation quantizer (host_int quantize_e2m1x2): x [rows][cols] FP32 -> (values int8,
    2 * e2m1; scales FP32 [rows][cols / 16], decoded / global)."""
    x = x.float().contiguous()
    values = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty(x.shape[0], x.shape[1] // 16, dtype=torch.float32, device=x.device)
    ops.quantize_e2m1x2(x, float(global_scale), values, scales)
    return values, scales
