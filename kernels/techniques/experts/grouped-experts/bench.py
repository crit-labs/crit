"""grouped-experts' sweep benches (crit_kernels.sweep): one bench per item width (decode 8, prompt 32, prompt-wide 64),
each timing one pass (gate/up or down) of the model's experts in their stored format: a store of random bundles in
crit's experts:bundles layout (as tests/test_grouped_experts.py packs them), BF16 inputs whose token t is seeded by t
alone and routed to experts seeded by t alone. The routes are planned and a down pass's input computed by the untuned
build (the plan has no knobs). No knob changes a route's operations or their order, so a configuration must give the
untuned build's bits for every route (the build tests/test_grouped_experts.py checks against an FP64 reference)."""
import torch

from crit_kernels import build as kbuild

FORMATS = {"bf16": 0, "fp8:channel": 1, "fp8:block": 2, "nvfp4": 3, "int4": 4, "mxfp4": 5, "int4_zp": 6}
GROUP = 128   # int4_zp's inputs a scale and zero point where a shape names none (AWQ's and GPTQ's usual)
NAMES = {v: k for k, v in FORMATS.items()}
STORE_BYTES = 2 << 30   # at most this much of bundles (fewer experts than the model's beyond it)
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def cdiv(a, b):
    return -(-a // b)


def components(fmt, H, I, group=GROUP):
    """(name, bytes, kind) of a bundle's components (crit's experts:bundles layout)."""
    if fmt == "int4_zp":   # FP16 scales, then the zero points in the inverse's place, [rows][K / group] each
        return [("gate_up", I * H, "u8"), ("gate_up_scale", 4 * I * H // group, "f16"),
                ("gate_up_inverse", 2 * I * H // group, "zero"), ("down", H * I // 2, "u8"),
                ("down_scale", 2 * H * I // group, "f16"), ("down_inverse", H * I // group, "zero")]
    if fmt == "bf16":
        return [("gate_up", 4 * I * H, "bf16"), ("down", 2 * H * I, "bf16")]
    if fmt.startswith("fp8"):
        gs, ds = ((8 * I, 4 * H) if fmt == "fp8:channel" else
                  (4 * 2 * cdiv(I, 128) * cdiv(H, 128), 4 * cdiv(H, 128) * cdiv(I, 128)))
        return [("gate_up", 2 * I * H, "e4m3"), ("gate_up_scale", gs, "f32"), ("down", H * I, "e4m3"),
                ("down_scale", ds, "f32")]
    if fmt in ("nvfp4", "int4"):
        return [("gate_up", I * H, "u8"), ("gate_up_scale", I * H // 8, "scale"), ("gate_up_inverse", 8 * I, "f32"),
                ("down", H * I // 2, "u8"), ("down_scale", H * I // 16, "scale"), ("down_inverse", 4 * H, "f32"),
                ("input", 16, "input")]
    return [("gate_up", I * H, "u8"), ("gate_up_scale", I * H // 16, "e8m0"), ("down", H * I // 2, "u8"),
            ("down_scale", H * I // 32, "e8m0")]


def offsets(fmt, H, I, group=GROUP):
    found, at = {}, 0
    for name, size, kind in components(fmt, H, I, group):
        found[name] = (at, size, kind)
        at += cdiv(size, 16) * 16
    return found, at


def layout(fmt, H, I, group=GROUP):
    o, size = offsets(fmt, H, I, group)
    keys = ("gate_up", "gate_up_scale", "gate_up_inverse", "down", "down_scale", "down_inverse", "input")
    return ([FORMATS[fmt], size] + [o[k][0] if k in o else -1 for k in keys] + [H, I] +
            ([group] if fmt == "int4_zp" else []))


def _fill(view, kind, g):
    """Random valid bytes of a component for every expert at once (view [experts][bytes] uint8)."""
    shape, device = view.shape, view.device
    if kind == "bf16":
        v = (torch.randn(shape[0], shape[1] // 2, generator=g, device=device) * 0.05).bfloat16().view(torch.uint8)
    elif kind == "e4m3":
        v = (torch.randn(shape, generator=g, device=device) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
        v = v.view(torch.uint8)
    elif kind == "f32":
        v = (torch.rand(shape[0], shape[1] // 4, generator=g, device=device) * 0.01 + 0.0005).view(torch.uint8)
    elif kind == "scale":   # E4M3 block scales, exponents 5..8
        v = (torch.randint(5, 9, shape, generator=g, device=device) * 8 +
             torch.randint(0, 8, shape, generator=g, device=device)).to(torch.uint8)
    elif kind == "f16":   # FP16 group scales
        v = ((torch.rand(shape[0], shape[1] // 2, generator=g, device=device) + 0.5) / 64).half().view(torch.uint8)
    elif kind == "zero":   # zero points 0..16
        v = torch.randint(0, 17, shape, generator=g, device=device).to(torch.uint8)
    elif kind == "e8m0":
        v = torch.randint(120, 127, shape, generator=g, device=device).to(torch.uint8)
    elif kind == "input":   # gate/up and down input scales, their inverses
        a = torch.rand(shape[0], generator=g, device=device) * 200 + 300
        b = torch.rand(shape[0], generator=g, device=device) * 40 + 20
        v = torch.stack((a, b, 1 / a, 1 / b), 1).float().contiguous().view(torch.uint8)
    else:
        v = torch.randint(0, 256, shape, generator=g, device=device).to(torch.uint8)
    view.copy_(v.reshape(shape))


def widen(store, fmt, H, I, part, chunk=16):
    """[experts][rows][cols] BF16: a pass's weights (gate/up [2 inter][hidden], down [hidden][inter]) decoded as the
    kernel decodes them, every expert of the store."""
    o, _ = offsets(fmt, H, I)
    rows, cols = (2 * I, H) if part == "gate_up" else (H, I)
    E, device = store.shape[0], store.device
    at, n, _ = o[part]
    out = torch.empty(E, rows, cols, dtype=torch.bfloat16, device=device)
    signed = torch.tensor([float(c if c < 8 else c - 16) for c in range(16)] if fmt == "int4" else
                          E2M1 + [-v for v in E2M1], device=device)
    for lo in range(0, E, chunk):
        b = store[lo:lo + chunk]
        e = b.shape[0]
        raw = b[:, at:at + n].contiguous()
        if fmt == "bf16":
            out[lo:lo + e] = raw.view(torch.bfloat16).view(e, rows, cols)
            continue
        sa, sn, _ = o[part + "_scale"]
        scale = b[:, sa:sa + sn].contiguous()
        if fmt == "int4_zp":
            za, zn, _ = o[part + "_inverse"]
            zero = b[:, za:za + zn].contiguous().float().view(e, rows, cols // GROUP).repeat_interleave(GROUP, 2)
            codes = raw.view(e, rows, cols // 2)
            q = torch.stack((codes & 15, codes >> 4), -1).reshape(e, rows, cols).float()
            s = scale.view(torch.float16).float().view(e, rows, cols // GROUP).repeat_interleave(GROUP, 2)
            out[lo:lo + e] = ((q - zero) * s).bfloat16()
            continue
        if fmt.startswith("fp8"):
            v = raw.view(torch.float8_e4m3fn).float().view(e, rows, cols)
            sc = scale.view(torch.float32)
            if fmt == "fp8:channel":
                v = v * sc.view(e, rows, 1)
            else:   # 128 x 128 grids: gate/up keeps gate's grid, then up's
                grids = [(I, H)] * 2 if part == "gate_up" else [(H, I)]
                full, at_g = [], 0
                for r, c in grids:
                    g = sc[:, at_g:at_g + cdiv(r, 128) * cdiv(c, 128)].view(e, cdiv(r, 128), cdiv(c, 128))
                    at_g += cdiv(r, 128) * cdiv(c, 128)
                    full.append(g.repeat_interleave(128, 1)[:, :r].repeat_interleave(128, 2)[:, :, :c])
                v = v * torch.cat(full, 1)
        else:
            codes = raw.view(e, rows, cols // 2)
            v = signed[torch.stack((codes & 15, codes >> 4), -1).long()].reshape(e, rows, cols)
            if fmt == "mxfp4":
                v = v * torch.exp2(scale.float() - 127).view(e, rows, cols // 32).repeat_interleave(32, 2)
            else:
                ia, inn, _ = o[part + "_inverse"]
                inverse = b[:, ia:ia + inn].contiguous().view(torch.float32).view(e, rows, 1)
                v = v * scale.view(torch.float8_e4m3fn).float().view(e, rows, cols // 16).repeat_interleave(16, 2)
                v = v * inverse
        out[lo:lo + e] = v.bfloat16()
    return out


def expert_shapes(model):
    """The routed experts' shapes in the model: format (of the expert weights), a4 (nvfp4 experts with input scales:
    both W4A16 and W4A4), hidden, inter, experts, top."""
    hidden = model.get("hidden_size")
    inter = model.get("moe_intermediate_size", "expert_intermediate_size")
    experts = model.get("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts")
    top = model.get("num_experts_per_tok", "moe_top_k", "top_k")
    if not (hidden and inter and experts and top) or hidden % 64 or inter % 64:
        return []
    formats = {m["format"] for m in model.matrices() if ".experts." in m["name"]}
    quant = model.quantization or {}
    zp = (str(quant.get("quant_method", "")).lower() in ("awq", "gptq") and int(quant.get("bits", 4)) == 4 and
          any(n.endswith(".qweight") and ".experts." in n for n in model.tensors))
    if zp and hidden % GROUP == 0 and inter % GROUP == 0 and int(quant.get("group_size") or GROUP) == GROUP:
        fmt = "int4_zp"
    elif "nvfp4" in formats:
        fmt = "nvfp4"
    elif "fp8" in formats:
        fmt = "fp8:block" if quant.get("weight_block_size") else "fp8:channel"
    elif str(quant.get("quant_method", "")).lower() == "mxfp4":
        fmt = "mxfp4"
    else:
        fmt = "bf16"
    a4s = [0]
    if fmt == "nvfp4" and any(n.endswith(".input_scale") and ".experts." in n for n in model.tensors):
        a4s.append(1)
    return [{"format": FORMATS[fmt], "a4": a4, "gate_up": gate_up, "hidden": int(hidden), "inter": int(inter),
             "experts": int(experts), "top": int(top)} for a4 in a4s for gate_up in (1, 0)]


class _Bench:
    width = 8
    default = None   # the untuned build: the sweep sets it before timing, else loaded on first use

    def __init__(self):
        self._stores, self._x, self._ids, self._plans, self._inputs, self._references = {}, {}, {}, {}, {}, {}
        self._top, self._wide = 1, {}

    def shapes(self, model):
        return expert_shapes(model)

    def valid(self, knobs):
        return knobs["DEPTH"] * knobs["TILES"] <= 8   # weights in flight beyond this spill registers

    def untuned(self):
        if self.default is None:
            self.default = kbuild.load("grouped-experts")   # the untuned build: plans and reference outputs
        return self.default

    def _store(self, shape, device):
        fmt, H, I = NAMES[int(shape["format"])], int(shape["hidden"]), int(shape["inter"])
        key = (fmt, H, I, int(shape["experts"]))
        if key not in self._stores:   # one store a shape, kept for every configuration the sweep times
            o, size = offsets(fmt, H, I)
            experts = max(1, min(int(shape["experts"]), STORE_BYTES // size))
            store = torch.empty(experts, size, dtype=torch.uint8, device=device)
            g = torch.Generator(device=device).manual_seed(H * 7 + I)
            for name, (at, n, kind) in o.items():
                _fill(store[:, at:at + n], kind, g)
            self._stores[key] = (store, layout(fmt, H, I))
        return self._stores[key]

    def _routes(self, experts, top, rows, device):
        key = (experts, top, rows)
        if key not in self._ids:   # token t's experts seeded by t alone
            ids = torch.stack([torch.randperm(experts, generator=torch.Generator().manual_seed(2000 + t))[:top]
                               for t in range(rows)])
            self._ids[key] = (ids.to(device), int(ids.unique().numel()))
        return self._ids[key][0]

    def inputs(self, shape, rows, device):
        store, lay = self._store(shape, device)
        H, I, top, a4 = int(shape["hidden"]), int(shape["inter"]), int(shape["top"]), bool(int(shape["a4"]))
        experts = store.shape[0]
        ids = self._routes(experts, top, rows, device)
        self._top = top
        gx = self.untuned()
        key = (tuple(lay), experts, top, rows, a4)
        if key not in self._plans:
            if (H, rows) not in self._x:
                self._x[(H, rows)] = torch.stack([torch.randn(H, generator=torch.Generator().manual_seed(1000 + t))
                                                  for t in range(rows)]).bfloat16().to(device)
            x = self._x[(H, rows)]
            ws = gx.Workspace(rows, top, experts, H, I, device, width=self.width)
            gx.route_outputs(x, ids, store, None, lay, ws, a4)   # plans the items, h of the untuned build
            if a4:
                xin, xtop = ws.xa.clone(), 1
            else:
                xin, xtop = x, top
            self._plans[key] = (ws, xin, xtop, ws.h.clone())
        ws, xin, xtop, h = self._plans[key]
        items = (ws.item_slot, ws.item_routes, ws.item_count)
        gate_up = bool(int(shape["gate_up"]))
        out = torch.zeros(ws.routes, I if gate_up else H, dtype=torch.bfloat16 if gate_up else torch.float32,
                          device=device)
        return {"gate_up": gate_up, "x": xin, "top": xtop, "h": h, "store": store, "layout": lay, "a4": a4,
                "items": items, "out": out, "tokens": rows, "top_k": top, "key": key + (gate_up,), "ids": ids,
                "fmt": NAMES[int(shape["format"])], "hidden": H, "inter": I}

    def _call(self, module, s, out):
        if s["gate_up"]:
            module.ops.gate_up(s["x"], s["top"], s["store"], s["layout"], s["a4"], *s["items"], out, s["tokens"])
        else:
            module.ops.down(s["h"], s["store"], s["layout"], s["a4"], *s["items"], out, s["tokens"])

    def run(self, module, s):
        self._call(module, s, s["out"])
        return s["out"]

    def reference(self, s):
        if s["key"] not in self._references:
            want = torch.zeros_like(s["out"])
            self._call(self.untuned(), s, want)
            self._references[s["key"]] = want
        return self._references[s["key"]]

    def check(self, got, want, s):
        view = torch.int16 if got.dtype == torch.bfloat16 else torch.int32
        bad = int((got.view(view) != want.view(view)).any(1).sum())
        return None if bad == 0 else f"{bad} routes differ from the untuned build's bits"

    def yardstick(self, s):
        """FlashInfer's grouped GEMM (cuDNN MoE backend, grouped_mm_bf16) on the pass's weights widened to BF16 once:
        no library takes every stored format against BF16 inputs one pass at a time. The timed call gathers the
        routes' inputs in expert order, runs the grouped GEMM and (gate/up) the SwiGLU; outputs stay in expert order."""
        try:
            from flashinfer.grouped_mm import grouped_mm_bf16
        except ImportError:
            return None
        part = "gate_up" if s["gate_up"] else "down"
        H, I, store = s["hidden"], s["inter"], s["store"]
        wkey = (tuple(s["layout"]), store.shape[0], part)
        if wkey not in self._wide:
            self._wide.clear()   # one pass's widened weights at a time
            self._wide[wkey] = widen(store, s["fmt"], H, I, part)
        w = self._wide[wkey]
        flat = s["ids"].reshape(-1)
        order = torch.argsort(flat, stable=True)
        indptr = torch.zeros(w.shape[0] + 1, dtype=torch.int32, device=flat.device)
        indptr[1:] = torch.bincount(flat, minlength=w.shape[0]).cumsum(0)
        if s["gate_up"]:
            x, src = s["x"], order // s["top"]

            def call():
                y = grouped_mm_bf16(x.index_select(0, src), w, indptr)
                return torch.nn.functional.silu(y[:, :I]) * y[:, I:]
            return call
        h = s["h"]
        return lambda: grouped_mm_bf16(h.index_select(0, order), w, indptr, out_dtype=torch.float32)

    def first_row(self, out):
        return out[:self._top]   # token 0's routes (first_row follows a run of the shape inputs() made last)

    def bytes(self, shape, rows):
        fmt, H, I = NAMES[int(shape["format"])], int(shape["hidden"]), int(shape["inter"])
        o, _ = offsets(fmt, H, I)
        part = "gate_up" if int(shape["gate_up"]) else "down"
        per_expert = sum(n for name, (_, n, _) in o.items() if name.startswith(part))
        store, _ = self._store(shape, "cuda")
        touched = self._ids.get((store.shape[0], int(shape["top"]), rows), (None, store.shape[0]))[1]
        return per_expert * touched


class Decode(_Bench):
    structure = "decode"
    width = 8


class Prompt(_Bench):
    structure = "prompt"
    width = 32


class PromptWide(_Bench):
    structure = "prompt-wide"
    width = 64
