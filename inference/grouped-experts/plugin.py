"""Routed SwiGLU experts over expert bundles in the experts slot. The host form is expert-streamed-prefill's FP64 block
(as experts-reference's), on the expert_source slot's weights. The device form keeps each expert as one bundle in its
stored format (code/bundles.py: bf16, fp8:channel, fp8:block, nvfp4, int4, mxfp4, int4_zp:<group> from AWQ and
GPTQ experts) and runs a layer's routes over its
own loaded bundles (forward) or over a residency cache's slot store and slot table (forward_bundles): through crit's
kernel grouped-experts (BF16 tensor cores, every format, the routes grouped by expert on the device; built on this GPU,
tuned or default knobs), or, where it does not build, through the same arithmetic in torch (code/fallback.py). Either
way a route's bits do not depend on the rows beside it, on chunking or on decode against prompt, and a call at a fixed
token count captures in a CUDA graph (after one eager call at that count); every workspace a capture used is kept for
the plugin's life. In a captured prompt chunk (prompts:experts) the padding's routes, ids -1 as the model masks them,
read no expert and add nothing. With activations nvfp4 each NVFP4 layer runs as its stored matrices declare: W4A4
where they carry input_scale, W4A16 where they do not (the facts name each layer's)."""
from pathlib import Path

from .grouped import bundles, fallback
from .prefill import reference

HERE = Path(__file__).resolve().parent
KERNEL = "grouped-experts"
ANY_EXPERT = (1 << 30) - 1   # the id bound of a combine without a slot table
PROMPT_KEEP = 2              # workspaces of prompt-sized calls kept, besides decode shapes and those a capture used
DECLARED = ("nvfp4", "int4")  # the formats crit's integer block order computes (W4A4, W4A8)
MAX_ITEMS = 65535            # route items in one kernel call (the kernel's MAX_ITEMS)
PLAN_BYTES = 40              # a route's plan temporaries above 1,024 routes (sort keys and order, counts)
KNOWN = tuple(f for f in bundles.FORMATS if f != "int4_zp") + ("int4_zp:<group>",)


def check_routes(x, ids, weights):
    if len(ids) != len(x) or len(weights) != len(x):
        raise ValueError("one ids row and one weights row per token")
    top = len(ids[0]) if ids else 0
    if any(len(row) != top for row in ids) or any(len(row) != top for row in weights):
        raise ValueError("every token has the same number of routes and one weight per route")
    return top


def capturing():
    """Whether the current CUDA stream is capturing a graph."""
    import torch
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class GroupedExperts:
    def __init__(self, context, options, parts):
        self.source = parts.get("expert_source")
        self.activations, self.arithmetic = options["activations"], options["arithmetic"]
        self.workspace_bytes = int(options["workspace_mib"]) << 20
        self.device = getattr(context, "device", None)
        self.layers = {}         # layer -> {"format", "bytes", "bundles": {expert: uint8}, "store", "captured"}
        self.workspaces = {}
        self.prompt_shapes = []  # evictable prompt-sized workspace keys, oldest first
        self.captured = set()    # workspace keys a CUDA graph capture used: kept for the plugin's life
        self.declared = {}       # nvfp4 layer -> True (its stored matrices carry input scales: W4A4) or False (W4A16)
        self.paths = {}          # (layer, store) -> W4A4 or not, read from the store's bundles at its first eager call
        self.kernel = None
        if options["kernel"] and self._cuda():
            from crit_plugins.kernels import load_kernel
            self.kernel = load_kernel(KERNEL, folder=HERE)
        if self.arithmetic == "declared" and self.kernel is None:
            raise ValueError(f"grouped-experts: arithmetic declared runs only on crit's kernel {KERNEL}, which is "
                             f"not here ({self._no_kernel(options)}); it exists to give the CPU expert tier's bits, which "
                             "the torch arithmetic does not")

    def _cuda(self):
        try:
            import torch
        except ImportError:
            return False
        device = self.device
        return torch.cuda.is_available() and (device is None or str(device).startswith("cuda"))

    def _no_kernel(self, options):
        if not options["kernel"]:
            return "option kernel is false"
        if not self._cuda():
            return "no CUDA device"
        from crit_plugins.kernels import loaded
        return loaded().get(KERNEL, {}).get("reason") or "it did not load"

    # -- host form ------------------------------------------------------------------------------------------------------
    def run(self, layer, x, ids, weights):
        check_routes(x, ids, weights)
        used = [e for row in ids for e in row if e >= 0]
        if not used:
            return [[0.0] * len(row) for row in x]
        cache = {}

        def matrices(expert):
            if expert not in cache:
                cache[expert] = self.source.matrices(layer, expert)
            return cache[expert]

        return reference.moe_block(x, ids, weights, lambda e: matrices(e)[0], lambda e: matrices(e)[1],
                                   len(matrices(used[0])[0]) // 2)

    # -- bundles --------------------------------------------------------------------------------------------------------
    def layout(self, format, hidden, inter):
        """The experts:bundles layout of one expert (code/bundles.py's layout, the one definition): {bundle_bytes,
        format, hidden, inter, components: {name: (offset, dtype, shape)}}."""
        return bundles.layout(format, hidden, inter)

    def pack(self, gate_up, down, format, layer=None):
        """One expert's bundle (uint8 [bundle_bytes] on the tensors' device) for a store this plugin will read: refused
        as load() refuses it. With `layer`, the bundle declares that layer's activations as a load does (W4A4 where the
        stored matrices carry input_scale), and a store forward_bundles reads for the layer must match it."""
        w4a4 = self._refuse(bundles.bundle_format(gate_up, down, format), gate_up, down)
        if layer is not None and w4a4 is not None:
            self._declare(layer, w4a4)
        return bundles.pack(gate_up, down, format)

    def _refuse(self, format, gate_up, down):
        """Refuses what this plugin does not compute; for an nvfp4 expert under activations nvfp4, whether it runs W4A4
        (both matrices carry input_scale, as the checkpoint declares them) or W4A16 (neither does), else None."""
        if not bundles.known(format):
            raise ValueError(f"grouped-experts computes {', '.join(KNOWN)}, not {format!r}")
        if self.arithmetic == "declared" and format not in DECLARED:
            raise ValueError(f"grouped-experts: arithmetic declared computes {' and '.join(DECLARED)} experts, not "
                             f"{format}")
        if self.arithmetic == "declared" and format == "nvfp4" and self.activations != "nvfp4":
            raise ValueError("grouped-experts: arithmetic declared runs NVFP4 experts W4A4: set activations nvfp4")
        if self.activations != "nvfp4" or format != "nvfp4":
            return None
        carried = [isinstance(stored, dict) and stored.get("input_scale") is not None for stored in (gate_up, down)]
        if carried[0] != carried[1]:
            raise ValueError("grouped-experts: an nvfp4 expert's gate/up and down matrices both carry input_scale (W4A4) "
                             "or neither does (W4A16), not one of them")
        if self.arithmetic == "declared" and not carried[0]:
            raise ValueError("grouped-experts: arithmetic declared runs NVFP4 experts W4A4, which needs each matrix's "
                             "input_scale")
        return carried[0]

    def _declare(self, layer, w4a4):
        """A layer's activations, as its experts' stored matrices declare them: every expert of a layer alike."""
        known = self.declared.setdefault(layer, w4a4)
        if known != w4a4:
            raise ValueError(f"grouped-experts: layer {layer} mixes nvfp4 experts with input_scale (W4A4) and without "
                             "(W4A16): one kernel call runs a layer one way")

    def load(self, layer, expert, gate_up, down, format="bf16"):
        """An expert's gate/up and down matrices (bfloat16 tensors, or the stored tensors by role) as one bundle on the
        engine's device. A load after the layer's first forward writes the bundle into the layer's store in place.
        awq and gptq matrices (format awq or gptq, the checkpoint's tensors; gate/up one matrix or a (gate, up) pair)
        load as int4_zp:<group> bundles, the layer's format from then on."""
        stored = format
        format = bundles.bundle_format(gate_up, down, format)
        w4a4 = self._refuse(format, gate_up, down)
        if w4a4 is not None:
            self._declare(layer, w4a4)
        first = gate_up[0] if isinstance(gate_up, (tuple, list)) else gate_up
        weight = first.get("weight", first.get("qweight")) if isinstance(first, dict) else first
        device = self.device if self.device is not None else weight.device
        bundle = bundles.pack(gate_up, down, stored, device)
        entry = self.layers.setdefault(layer, {"format": format, "bytes": bundle.numel(), "bundles": {}, "store": None,
                                               "captured": False})
        if entry["format"] != format or entry["bytes"] != bundle.numel():
            raise ValueError(f"layer {layer} holds {entry['format']} experts of {entry['bytes']} bytes, not {format} "
                             f"of {bundle.numel()}")
        expert = int(expert)
        store = entry["store"]
        if store is not None and expert < store.shape[0]:
            store[expert].copy_(bundle)   # in place: the store, and any graph that captured it, keep their memory
            return
        if entry["captured"]:
            raise ValueError(f"layer {layer}'s store is in a captured CUDA graph: load its experts before the capture")
        entry["bundles"][expert] = bundle
        entry["store"] = None

    def _own(self, layer, device=None):
        """The layer's loaded bundles as a store [experts][bytes], expert e in slot e, on `device` (the activations'):
        built once after its loads (and moved once, when the experts were loaded where the engine names no device);
        the loaded bundles then live on as its rows."""
        import torch
        entry = self.layers.get(layer)
        if entry is None:
            raise KeyError(f"no experts of layer {layer} are loaded")
        store = entry["store"]
        if store is None or (device is not None and store.device != torch.device(device)):
            if store is None:
                found = entry["bundles"]
                count = max(found) + 1
                missing = sorted(set(range(count)) - set(found))
                if missing:
                    raise KeyError(f"layer {layer} lacks experts {missing[:8]}")
                store = torch.stack([found[e] for e in range(count)])
            if device is not None and store.device != torch.device(device):
                store = store.to(device)
            entry["store"] = store
            entry["bundles"] = {e: store[e] for e in range(store.shape[0])}   # views: no second copy
        if not entry["captured"] and capturing():
            entry["captured"] = True
        return store, entry["format"]

    # -- device form ----------------------------------------------------------------------------------------------------
    def forward(self, layer, x, ids, weights):
        """out [tokens][hidden] FP32: the routed SwiGLU block over the layer's loaded bundles."""
        store, format = self._own(layer, x.device)
        return self.forward_bundles(layer, x, ids, weights, store, None, format)

    def _named_format(self, layer, format):
        """A store's bundle format: awq and gptq stand for the int4_zp:<group> bundles the layer's loads packed."""
        if format in bundles.STORED_INT4:
            entry = self.layers.get(layer)
            if entry is None or not str(entry["format"]).startswith("int4_zp"):
                raise ValueError(f"grouped-experts: {format} experts are int4_zp:<group> bundles: name the bundles' "
                                 "format (bundles.bundle_format), or load the layer's experts first")
            return entry["format"]
        return format

    def forward_bundles(self, layer, x, ids, weights, store, slots, format):
        """out [tokens][hidden] FP32: the routed SwiGLU block over a store [S][bundle_bytes] of `format` bundles, each
        expert in its slot (slots [experts] int32; None: expert e in slot e); routes to negative slots skipped. A long
        prompt chunk runs in parts (route_outputs' parts), each part's routes combined into its tokens' rows as it
        goes, so no call holds every route's row at once (a token's sum is over its own routes in route order: the
        same bits whole or in parts)."""
        import torch
        format = self._named_format(layer, format)
        tokens = ids.shape[0]
        step = self._part_tokens(layer, x, ids, store, slots, format)
        if tokens > step:
            out = torch.empty(tokens, x.shape[1], dtype=torch.float32, device=x.device)
            for t0 in range(0, tokens, step):
                part = slice(t0, t0 + step)
                y = self.route_outputs(layer, x[part], ids[part], store, slots, format)
                out[part] = self.combine(y, ids[part], weights[part], slots)
            return out
        y = self.route_outputs(layer, x, ids, store, slots, format)
        return self.combine(y, ids, weights, slots)

    def _part_tokens(self, layer, x, ids, store, slots, format):
        """Tokens per part of a call of these routes over this store (_step's)."""
        if not bundles.known(format):
            raise ValueError(f"grouped-experts computes {', '.join(KNOWN)}, not {format!r}")
        hidden = x.shape[1]
        inter = bundles.inter_of(format, hidden, store.shape[1])
        found = bundles.layout(format, hidden, inter)
        experts = slots.numel() if slots is not None else store.shape[0]
        a4 = self._a4(layer, store, None if slots is None else slots.int().contiguous(), found)
        return self._step(ids.shape[1], hidden, inter, experts, a4, self.arithmetic == "declared", format)

    def route_outputs(self, layer, x, ids, store, slots, format):
        """Each route's unweighted output row, [tokens * top][hidden] FP32 (a workspace: valid until the next call of
        the shape); routes with id -1 or a negative slot keep the rows they had."""
        import torch
        format = self._named_format(layer, format)
        if not bundles.known(format):
            raise ValueError(f"grouped-experts computes {', '.join(KNOWN)}, not {format!r}")
        tokens, top = ids.shape
        hidden = x.shape[1]
        inter = bundles.inter_of(format, hidden, store.shape[1])
        found = bundles.layout(format, hidden, inter)
        experts = slots.numel() if slots is not None else store.shape[0]
        declared = self.arithmetic == "declared"
        if declared and format not in DECLARED:
            raise ValueError(f"grouped-experts: arithmetic declared computes {' and '.join(DECLARED)} experts, not "
                             f"{format}")
        slots = None if slots is None else (slots if slots.dtype == torch.int32 else slots.int()).contiguous()
        a4 = self._a4(layer, store, slots, found)
        step = self._step(top, hidden, inter, experts, a4, declared, format)
        if tokens > step:   # a long prompt chunk in parts: every route keeps its bits whatever part it falls in
            y = torch.empty(tokens * top, hidden, dtype=torch.float32, device=x.device)
            for t0 in range(0, tokens, step):
                part = self.route_outputs(layer, x[t0:t0 + step], ids[t0:t0 + step], store, slots, format)
                y[t0 * top:(t0 + step) * top] = part
            return y
        ws = self._workspace(tokens, top, experts, hidden, inter, x.device)
        if self.kernel is not None and x.is_cuda:
            xb = x if x.dtype == torch.bfloat16 and x.is_contiguous() else x.to(torch.bfloat16).contiguous()
            return self.kernel.route_outputs(xb, ids.contiguous(), store, slots, bundles.kernel_layout(found), ws, a4,
                                             self.arithmetic)
        if declared:
            raise ValueError(f"grouped-experts: arithmetic declared runs on crit's kernel {KERNEL}: CUDA "
                             f"activations, not {x.device}")
        return fallback.route_outputs(x, ids, store, slots, found, a4, ws.y)

    def combine(self, y, ids, weights, slots=None):
        """out [tokens][hidden] FP32: each token's routes in route order, fmaf(weight, row, sum), over the routes with
        id >= 0 (and a slot >= 0 when slots are given)."""
        import torch
        weights = weights if weights.dtype == torch.float32 and weights.is_contiguous() else weights.float().contiguous()
        if self.kernel is not None and y.is_cuda:
            experts = slots.numel() if slots is not None else ANY_EXPERT
            s = None if slots is None else (slots if slots.dtype == torch.int32 else slots.int()).contiguous()
            return self.kernel.combine(y, ids.contiguous(), weights, s, experts, True)
        return fallback.combine(y, ids, weights, slots)

    def _a4(self, layer, store, slots, found):
        """Whether a call over `store` runs the layer W4A4 (activations nvfp4 and nvfp4 bundles that carry their input
        scales) or W4A16. The first eager call over a (layer, store) reads the input scales of the bundles its slot table
        names: all of them carry theirs or none does, and a layer whose loads or packs declared its activations gets
        bundles that match (a W4A4 layer's bundles packed without their scales would give zeros). Captures replay
        what that eager call decided."""
        if self.activations != "nvfp4" or found["format"] != "nvfp4":
            return False
        key = (layer, store.data_ptr(), tuple(store.shape), str(store.device))
        if key in self.paths:
            return self.paths[key]
        known = self.declared.get(layer)
        if capturing():
            if known is None:
                raise ValueError(f"grouped-experts: layer {layer}'s activations are not known inside a capture: run one "
                                 "eager call over its store first")
            return known
        named, carrying = bundles.input_counts(store, found, slots)
        if 0 < carrying < named:
            raise ValueError(f"grouped-experts: layer {layer}'s bundles mix input scales ({carrying} of the {named} "
                             "its slot table names carry them): one kernel call runs a layer W4A4 or W4A16")
        if named == 0:   # nothing routed through this store yet: no bundle to read
            return bool(known)
        carried = carrying == named
        if known is not None and known != carried:
            raise ValueError(f"grouped-experts: layer {layer}'s experts declare "
                             f"{'input scales (W4A4)' if known else 'no input scales (W4A16)'}, but the store's bundles "
                             f"{'lack' if known else 'carry'} them: pack the layer's bundles from its stored tensors")
        self.paths[key] = carried
        return carried

    def _step(self, top, hidden, inter, experts, a4, declared, format):
        """Tokens per part of a long call: the route rows workspace_mib holds and at most MAX_ITEMS kernel items,
        whole multiples of 256."""
        per_route = hidden * 4 + inter * 2 + (hidden * 2 if a4 else 0) + PLAN_BYTES   # y FP32, h BF16, W4A4 operands
        if declared:   # quantized inputs and their scales, gate/up rows FP32, the quantized intermediate and its scales
            block = 32 if format == "int4" else 16
            per_route += hidden + hidden // block * 4 + 8 * inter + inter + inter // block * 4
        by_bytes = self.workspace_bytes // (per_route * top)
        by_items = max(0, MAX_ITEMS - experts) * 8 // top   # items: ceil(routes / width) + min(experts, routes)
        return max(256, min(by_bytes, by_items) // 256 * 256)

    def _workspace(self, tokens, top, experts, hidden, inter, device):
        key = (tokens, top, experts, hidden, inter, str(device))
        found = self.workspaces.get(key)
        if found is None and tokens * top > 1024 and not capturing():
            # A prompt-sized call of fewer tokens than a kept prompt workspace of its shape runs in that one's buffers
            # (the leading rows; its items' capacity and width cover it), so a chunk and its MTP pass, or a part and the
            # last part, keep one workspace rather than one each.
            found = self._within(tokens, key)
        if found is None:
            if self.kernel is not None and str(device).startswith("cuda"):
                found = self.kernel.Workspace(tokens, top, experts, hidden, inter, device)
            else:
                found = _TorchWorkspace(tokens, top, hidden, device)
            self.workspaces[key] = found
            if tokens * top > 1024:
                self.prompt_shapes.append(key)
        if key not in self.captured and capturing():
            self.captured.add(key)   # a graph replays into it: never dropped
            if key in self.prompt_shapes:
                self.prompt_shapes.remove(key)
        while len(self.prompt_shapes) > PROMPT_KEEP:
            self.workspaces.pop(self.prompt_shapes.pop(0), None)
        return found

    def _within(self, tokens, key):
        """A view of the smallest kept prompt workspace of key's shape with at least `tokens` tokens, or None."""
        top, experts = key[1], key[2]
        fits = sorted((k for k in self.prompt_shapes if k[1:] == key[1:] and k[0] >= tokens), key=lambda k: k[0])
        for k in fits:
            parent = self.workspaces.get(k)
            if parent is None:
                continue
            if isinstance(parent, _TorchWorkspace):
                return _TorchWorkspace.within(parent, tokens * top)
            width = getattr(self.kernel, "width_for", None)
            if width is not None and width(tokens * top, experts) != parent.width:
                continue
            return _KernelWithin(parent, tokens)
        return None

    def facts(self):
        """The plugin's work, each layer's included: {layer: {format, activations}} (nvfp4 or bf16 activations) for every
        layer it has loaded, packed for or read a store of."""
        layers = {}
        for layer, entry in self.layers.items():
            layers[str(layer)] = {"format": entry["format"], "activations": self._named(entry["format"], layer)}
        for layer, w4a4 in self.declared.items():
            layers.setdefault(str(layer), {"format": "nvfp4", "activations": "nvfp4" if w4a4 else "bf16"})
        for (layer, *_), w4a4 in self.paths.items():
            layers.setdefault(str(layer), {"format": "nvfp4", "activations": "nvfp4" if w4a4 else "bf16"})
        return {"experts": {"arithmetic": self.arithmetic, "activations": self.activations, "layers": layers,
                            "path": "kernel" if self.kernel is not None else "torch",
                            "row_invariance": {"rows": True, "prompt_rows": True}}}

    def _named(self, format, layer):
        if format != "nvfp4" or self.activations != "nvfp4":
            return "bf16"
        return "nvfp4" if self.declared.get(layer) else "bf16"


class _KernelWithin:
    """The kernel's Workspace for `tokens` tokens over the leading rows of a larger one's buffers: the same
    items (capacity covers fewer routes at the same width), h, y, ones and the W4A4 operands sliced to its routes."""

    def __init__(self, parent, tokens):
        self.parent, self.tokens, self.top = parent, int(tokens), parent.top
        self.experts, self.routes = parent.experts, self.tokens * parent.top
        self.width, self.capacity = parent.width, parent.capacity
        self.item_slot, self.item_routes, self.item_count = parent.item_slot, parent.item_routes, parent.item_count
        self.h, self.y = parent.h[:self.routes], parent.y[:self.routes]
        self.ones = parent.ones[:self.routes] if parent.ones is not None else None
        self.hidden, self.inter, self.device = parent.hidden, parent.inter, parent.device
        self._xa = None

    @property
    def xa(self):
        if self._xa is None and self.parent.xa is not None:
            return self.parent.xa[:self.routes]
        return self._xa

    @xa.setter
    def xa(self, value):
        self._xa = value

    def declared(self, layout):
        return {k: v[:self.routes] for k, v in self.parent.declared(layout).items()}


class _TorchWorkspace:
    def __init__(self, tokens, top, hidden, device):
        import torch
        self.y = torch.zeros(tokens * top, hidden, dtype=torch.float32, device=device)

    @classmethod
    def within(cls, parent, routes):
        found = cls.__new__(cls)
        found.y = parent.y[:routes]
        return found


def create(context, options, parts):
    return GroupedExperts(context, options, parts)
