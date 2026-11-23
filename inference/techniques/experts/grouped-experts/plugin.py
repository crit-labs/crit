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
import json
from pathlib import Path

from .grouped import bundles, fallback
from .prefill import reference

HERE = Path(__file__).resolve().parent
KERNEL = "grouped-experts"
ANY_EXPERT = (1 << 30) - 1   # the id bound of a combine without a slot table
PROMPT_KEEP = 2              # workspaces of prompt-sized calls kept, besides decode shapes and those a capture used
DECLARED = ("nvfp4", "int4")  # the formats crit's integer block order computes (W4A4, W4A8)
# The arithmetics on the formats' own tensor cores: (formats, what they need of the GPU)
ROW_SCALED = ("fp4-row", "fp4-row2")   # nvfp4 on the FP4 tensor cores at each route's own scale: 1 or 2 terms
TENSOR_CORES = {"fp4": (("nvfp4",), "block-scaled FP4 tensor cores (the sm_120 family)"),
                "fp4-row": (("nvfp4",), "block-scaled FP4 tensor cores (the sm_120 family)"),
                "fp4-row2": (("nvfp4",), "block-scaled FP4 tensor cores (the sm_120 family)"),
                "fp8": (("fp8:channel", "fp8:block"), "E4M3 tensor cores (8.9 on)")}
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
        from crit_plugins.kernels import quality_mode
        speed = quality_mode() == "speed"
        # auto: by the quality mode the engine states, the stored activations (W4A4 where the matrices carry their
        # input scales) and each format's own tensor cores in speed (own_cores: the arithmetic a layer takes where
        # its format, activations and shape allow, else tensor's BF16 tensor cores), 16-bit ones and tensor otherwise.
        self.activations = options["activations"] if options["activations"] != "auto" else (
            "nvfp4" if speed else "bf16")
        self.arithmetic, self.own_cores, self.chosen = options["arithmetic"], frozenset(), {}
        self.prompt_chosen = {}   # layer -> the arithmetic its prompt rows ran where it may differ (prompt_arithmetic)
        # prompt rows' arithmetic where it differs (the calls a model marks prompt: decode rows keep `arithmetic`);
        # auto in speed lets an NVFP4 layer without input scales take fp4-row for prompt rows where the engine asks
        # its prompt rows no more than row-exact (their bits then their own, not the decoded route's)
        self.prompt_arithmetic = options.get("prompt_arithmetic", "auto")
        if self.prompt_arithmetic == self.arithmetic and self.arithmetic != "auto":
            self.prompt_arithmetic = "same"
        self.speed = speed
        # FP4 prompt calls' items: fp4-prompt's 32 routes, fp4-tile's 64 (the block tile), or auto: the faster of the
        # two in the engine's sweep report on this device, fp4-prompt where there is none (_prompt_items)
        self.prompt_items = options.get("prompt_items", "auto")
        if self.prompt_items not in PROMPT_ITEMS:
            raise ValueError(f"grouped-experts: prompt_items is one of {', '.join(PROMPT_ITEMS)}, not "
                             f"{self.prompt_items!r}")
        self.items_chosen = {}   # (hidden, inter, tokens) -> the structure an FP4 prompt call's items run, and why
        self.activation, self.caps = "silu", None   # the blocks' gate (activate)
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
        if "auto" in (self.arithmetic, self.prompt_arithmetic) and speed and self.kernel is not None:
            # fp4-row and fp4-row2 run on the FP4 MMAs
            self.own_cores = frozenset(a for a in TENSOR_CORES if getattr(
                self.kernel, a.split("-")[0] + "_available", lambda device: False)(self.device))
        for name, asked in (("arithmetic", self.arithmetic), ("prompt_arithmetic", self.prompt_arithmetic)):
            if asked not in TENSOR_CORES:
                continue
            probe = getattr(self.kernel, asked.split("-")[0] + "_available", None) if self.kernel is not None else None
            if probe is None or not probe(self.device):
                raise ValueError(f"grouped-experts: {name} {asked} runs only on crit's kernel {KERNEL} on a GPU with "
                                 f"{TENSOR_CORES[asked][1]}, which is not here")
        if self.prompt_arithmetic in ("declared",):
            raise ValueError("grouped-experts: prompt_arithmetic declared would give prompt rows the CPU tier's bits "
                             "and decode rows others: set arithmetic declared")
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

    def activate(self, name, caps=None):
        """The gate of the blocks' product, act(gate) * up: silu (SwiGLU, the default), gelu_tanh or softcapped_silu
        with caps (gate, up) (experts:activations), for every layer; before the first forward. The declared
        arithmetic is SwiGLU's."""
        if (name == "softcapped_silu") != (caps is not None) or (caps is not None and (len(caps) != 2
                                                                                         or min(caps) <= 0)):
            raise ValueError("grouped-experts: softcapped_silu, and only it, takes two positive caps: (gate, up)")
        if name not in bundles.ACTIVATIONS:
            raise ValueError(f"grouped-experts gates a block by {' or '.join(bundles.ACTIVATIONS)}, not {name!r}")
        if name != "silu" and self.arithmetic == "declared":
            raise ValueError("grouped-experts: arithmetic declared computes SwiGLU blocks (the CPU expert tier's "
                             f"bits); a block gated by {name} runs arithmetic tensor")
        self.activation, self.caps = name, None if caps is None else (float(caps[0]), float(caps[1]))

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
                                   len(matrices(used[0])[0]) // 2, activation=self.activation, caps=self.caps)

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
        if self.arithmetic == "fp8" and format not in TENSOR_CORES["fp8"][0]:
            raise ValueError(f"grouped-experts: arithmetic fp8 runs FP8 experts W8A8 ({', '.join(TENSOR_CORES['fp8'][0])}), "
                             f"not {format}")
        if self.arithmetic == "fp4" and (format != "nvfp4" or self.activations != "nvfp4"):
            raise ValueError("grouped-experts: arithmetic fp4 runs NVFP4 experts W4A4: nvfp4 bundles with activations "
                             f"nvfp4, not {format} with activations {self.activations}")
        if self.arithmetic == "declared" and format == "nvfp4" and self.activations != "nvfp4":
            raise ValueError("grouped-experts: arithmetic declared runs NVFP4 experts W4A4: set activations nvfp4")
        if self.activations != "nvfp4" or format != "nvfp4":
            return None
        carried = [isinstance(stored, dict) and stored.get("input_scale") is not None for stored in (gate_up, down)]
        if carried[0] != carried[1]:
            raise ValueError("grouped-experts: an nvfp4 expert's gate/up and down matrices both carry input_scale (W4A4) "
                             "or neither does (W4A16), not one of them")
        if self.arithmetic == "fp4" and not carried[0]:
            raise ValueError("grouped-experts: arithmetic fp4 runs NVFP4 experts W4A4, which needs each matrix's "
                             "input_scale")
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
        if layer not in self.layers:
            # A new layer's first load: the layers loaded before it become their stores now, while the device still
            # has the room for one layer's copy (built at its first forward, the copy would meet the engine's KV).
            for other, found in self.layers.items():
                if found["store"] is None and found["bundles"]:
                    self._own(other)
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
    def forward(self, layer, x, ids, weights, prompt=None, rows=None):
        """out [tokens][hidden] FP32: the routed SwiGLU block over the layer's loaded bundles. prompt: whether the rows
        are a prompt chunk's (experts:phases: the model calls decode rows and prompt rows apart), which
        prompt_arithmetic computes where it is set; rows: not read (an expert cache's per-row counts)."""
        store, format = self._own(layer, x.device)
        return self.forward_bundles(layer, x, ids, weights, store, None, format, prompt=bool(prompt))

    def _named_format(self, layer, format):
        """A store's bundle format: awq and gptq stand for the int4_zp:<group> bundles the layer's loads packed."""
        if format in bundles.STORED_INT4:
            entry = self.layers.get(layer)
            if entry is None or not str(entry["format"]).startswith("int4_zp"):
                raise ValueError(f"grouped-experts: {format} experts are int4_zp:<group> bundles: name the bundles' "
                                 "format (bundles.bundle_format), or load the layer's experts first")
            return entry["format"]
        return format

    def forward_bundles(self, layer, x, ids, weights, store, slots, format, prompt=False):
        """out [tokens][hidden] FP32: the routed SwiGLU block over a store [S][bundle_bytes] of `format` bundles, each
        expert in its slot (slots [experts] int32; None: expert e in slot e); routes to negative slots skipped. A long
        prompt chunk runs in parts (route_outputs' parts), each part's routes combined into its tokens' rows as it
        goes, so no call holds every route's row at once (a token's sum is over its own routes in route order: the
        same bits whole or in parts)."""
        import torch
        format = self._named_format(layer, format)
        tokens = ids.shape[0]
        step = self._part_tokens(layer, x, ids, store, slots, format, prompt)
        if tokens > step:
            out = torch.empty(tokens, x.shape[1], dtype=torch.float32, device=x.device)
            for t0 in range(0, tokens, step):
                part = slice(t0, t0 + step)
                y = self.route_outputs(layer, x[part], ids[part], store, slots, format, prompt=prompt)
                out[part] = self.combine(y, ids[part], weights[part], slots)
            return out
        y = self.route_outputs(layer, x, ids, store, slots, format, prompt=prompt)
        return self.combine(y, ids, weights, slots)

    def _part_tokens(self, layer, x, ids, store, slots, format, prompt=False):
        """Tokens per part of a call of these routes over this store (_step's)."""
        if not bundles.known(format):
            raise ValueError(f"grouped-experts computes {', '.join(KNOWN)}, not {format!r}")
        hidden = x.shape[1]
        inter = bundles.inter_of(format, hidden, store.shape[1])
        found = bundles.layout(format, hidden, inter)
        experts = slots.numel() if slots is not None else store.shape[0]
        a4 = self._a4(layer, store, None if slots is None else slots.int().contiguous(), found)
        arithmetic = self._arithmetic(format, a4, hidden, inter, prompt)
        return self._step(ids.shape[1], hidden, inter, experts, a4, arithmetic, format)

    def _arithmetic(self, format, a4, hidden, inter, prompt=False):
        """The arithmetic a call over a layer of `format` runs: prompt_arithmetic for prompt rows where it names one,
        else arithmetic where it names one; auto, the format's own tensor cores where own_cores has them (speed mode on
        a GPU that has them: fp4 for W4A4 NVFP4 experts, fp8 for FP8 ones, hidden and inter multiples of 128, as those
        take), and for an NVFP4 layer without input scales fp4-row on prompt rows where _prompts_may_split allows it (decode
        rows, bound by their weights' bytes, keep tensor: FP4 operands save them nothing), else tensor. A row-scaled FP4
        arithmetic computes nvfp4 bundles of hidden and inter multiples of 128, other layers tensor."""
        fits = hidden % 128 == 0 and inter % 128 == 0
        asked = self.prompt_arithmetic if prompt and self.prompt_arithmetic not in ("same", "auto") else self.arithmetic
        if asked in ROW_SCALED:
            return asked if format == "nvfp4" and fits else "tensor"
        if asked != "auto":
            return asked
        own = "fp4" if format == "nvfp4" and a4 else "fp8" if format in TENSOR_CORES["fp8"][0] else None
        if own in self.own_cores and fits:
            return own
        if (prompt and self.prompt_arithmetic == "auto" and format == "nvfp4" and fits and "fp4-row" in self.own_cores
                and self._prompts_may_split()):
            return "fp4-row"
        return "tensor"

    def route_outputs(self, layer, x, ids, store, slots, format, prompt=False):
        """Each route's unweighted output row, [tokens * top][hidden] FP32 (a workspace: valid until the next call of
        the shape); routes with id -1 or a negative slot keep the rows they had. prompt: a prompt chunk's rows
        (prompt_arithmetic)."""
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
        arithmetic = self._arithmetic(format, a4, hidden, inter, prompt)
        if arithmetic == "fp4" and not a4:
            raise ValueError("grouped-experts: arithmetic fp4 runs NVFP4 experts W4A4: this layer's bundles carry no "
                             "input scales (fp4-row takes each route's own scale)")
        if self.arithmetic == "auto" and not prompt:
            self.chosen[layer] = arithmetic
        elif prompt and (self.prompt_arithmetic not in ("same", "auto") or self.arithmetic == "auto"):
            self.prompt_chosen[layer] = arithmetic
        step = self._step(top, hidden, inter, experts, a4, arithmetic, format)
        if tokens > step:   # a long prompt chunk in parts: every route keeps its bits whatever part it falls in
            y = torch.empty(tokens * top, hidden, dtype=torch.float32, device=x.device)
            for t0 in range(0, tokens, step):
                part = self.route_outputs(layer, x[t0:t0 + step], ids[t0:t0 + step], store, slots, format,
                                          prompt=prompt)
                y[t0 * top:(t0 + step) * top] = part
            return y
        ws = self._workspace(tokens, top, experts, hidden, inter, x.device, arithmetic)
        if self.kernel is not None and x.is_cuda:
            xb = x if x.dtype == torch.bfloat16 and x.is_contiguous() else x.to(torch.bfloat16).contiguous()
            return self.kernel.route_outputs(xb, ids.contiguous(), store, slots,
                                             bundles.kernel_layout(found, self.activation, self.caps), ws,
                                             a4 and arithmetic not in ROW_SCALED, arithmetic)
        if declared or arithmetic in ROW_SCALED:
            raise ValueError(f"grouped-experts: arithmetic {arithmetic} runs on crit's kernel {KERNEL}: CUDA "
                             f"activations, not {x.device}")
        return fallback.route_outputs(x, ids, store, slots, found, a4, ws.y, self.activation, self.caps)

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

    def _step(self, top, hidden, inter, experts, a4, arithmetic, format):
        """Tokens per part of a long call in `arithmetic`: the route rows workspace_mib holds and at most MAX_ITEMS
        kernel items, whole multiples of 256."""
        per_route = hidden * 4 + inter * 2 + (hidden * 2 if a4 else 0) + PLAN_BYTES   # y FP32, h BF16, W4A4 operands
        if arithmetic == "declared":   # quantized inputs and scales, gate/up rows FP32, the quantized intermediate
            block = 32 if format == "int4" else 16
            per_route += hidden + hidden // block * 4 + 8 * inter + inter + inter // block * 4
        if arithmetic == "fp4":   # the inputs' and the intermediate's codes and scale bytes
            per_route += (hidden + inter) // 2 + (hidden + inter) // 16
        if arithmetic in ROW_SCALED:   # each term's codes and scale bytes, and the routes' 1 / g
            terms = 2 if arithmetic == "fp4-row2" else 1
            per_route += terms * ((hidden + inter) // 2 + (hidden + inter) // 16) + 8
        if arithmetic == "fp8":   # the inputs' and the intermediate's codes and FP32 group scales
            per_route += (hidden + inter) + (hidden + inter) // 32
        by_bytes = self.workspace_bytes // (per_route * top)
        by_items = max(0, MAX_ITEMS - experts) * 8 // top   # items: ceil(routes / width) + min(experts, routes)
        return max(256, min(by_bytes, by_items) // 256 * 256)

    def _prompt_items(self, tokens, top, experts, hidden, inter, arithmetic):
        """(width, tile) of a call's items: None, False (the kernel's width_for) but for an FP4 prompt-sized call
        (8 routes an expert or more, an FP4 tensor-core arithmetic), whose items prompt_items picks: fp4-prompt
        (32), fp4-tile (64 through the block tile), or auto: the one the engine's sweep report on this device timed
        faster at the call's tokens over both passes (crit_kernels.select.fastest), fp4-prompt without a report."""
        if arithmetic not in FP4_MMA or tokens * top < 8 * experts:
            return None, False
        chosen, why = self.prompt_items, "the prompt_items option"
        if chosen == "auto":
            chosen, why = "fp4-prompt", "no sweep report of both structures on this device"
            report = _report(self.device)
            if report is not None:
                from crit_kernels import select

                def both(structure):
                    passes = [select.times(report, structure, {"gate_up": g, "hidden": hidden, "inter": inter})
                              for g in (1, 0)]
                    return {r: passes[0][r] + passes[1][r] for r in set(passes[0]) & set(passes[1])}
                found = select.fastest([("fp4-prompt", both("fp4-prompt")), ("fp4-tile", both("fp4-tile"))], tokens,
                                       default=None)
                if found is not None:
                    chosen, why = found, "the sweep report's faster at these tokens"
        self.items_chosen[(hidden, inter, tokens)] = {"structure": chosen, "by": why}
        return (64, True) if chosen == "fp4-tile" else (32, False)

    def _workspace(self, tokens, top, experts, hidden, inter, device, arithmetic=None):
        """The kept workspace of a call's shape, made at its first call, its items as _prompt_items gives the call's
        arithmetic."""
        key = (tokens, top, experts, hidden, inter, str(device))
        found = self.workspaces.get(key)
        if found is None and tokens * top > 1024 and not capturing():
            # A prompt-sized call of fewer tokens than a kept prompt workspace of its shape runs in that one's buffers
            # (the leading rows; its items' capacity and width cover it), so a chunk and its MTP pass, or a part and the
            # last part, keep one workspace rather than one each.
            found = self._within(tokens, key, arithmetic)
        if found is None:
            if self.kernel is not None and str(device).startswith("cuda"):
                width, tile = self._prompt_items(tokens, top, experts, hidden, inter, arithmetic)
                found = self.kernel.Workspace(tokens, top, experts, hidden, inter, device, width=width, tile=tile)
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

    def _within(self, tokens, key, arithmetic=None):
        """A view of the smallest kept prompt workspace of key's shape with at least `tokens` tokens, or None."""
        top, experts = key[1], key[2]
        fits = sorted((k for k in self.prompt_shapes if k[1:] == key[1:] and k[0] >= tokens), key=lambda k: k[0])
        for k in fits:
            parent = self.workspaces.get(k)
            if parent is None:
                continue
            if isinstance(parent, _TorchWorkspace):
                return _TorchWorkspace.within(parent, tokens * top)
            width, tile = self._prompt_items(tokens, top, experts, key[3], key[4], arithmetic)
            width = width or self.kernel.width_for(tokens * top, experts)
            if (width, tile) != (parent.width, getattr(parent, "tile", False)):
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
        for layer, arithmetic in self.chosen.items():   # what arithmetic auto ran each layer in
            layers.setdefault(str(layer), {})["arithmetic"] = arithmetic
        for layer, arithmetic in self.prompt_chosen.items():   # ... and its prompt rows, where they may differ
            if arithmetic != self.chosen.get(layer, arithmetic) or self._prompts_apart():
                layers.setdefault(str(layer), {})["prompt_arithmetic"] = arithmetic
        return {"experts": {"arithmetic": self.arithmetic, "activations": self.activations, "layers": layers,
                            **({"prompt_arithmetic": self.prompt_arithmetic} if self._prompts_apart() else {}),
                            # what FP4 prompt calls' items ran (prompt_items), by hidden x inter @ tokens, and why
                            **({"prompt_items": {f"{h}x{i}@{t}": v for (h, i, t), v in sorted(self.items_chosen.items())}}
                               if self.items_chosen else {}),
                            "path": "kernel" if self.kernel is not None else "torch",
                            "row_invariance": {"rows": True, "prompt_rows": True}}}

    def _prompts_may_split(self):
        """Whether auto may give prompt rows another arithmetic than decode rows': in quality mode speed where the
        engine asks its prompt rows no more than row-exact (the engine's policy, active around each call)."""
        from crit_plugins.kernels import asks
        return self.speed and asks("prompt") in (None, "row-exact", "repeatable")

    def _prompts_apart(self):
        """Whether prompt rows run another arithmetic than decode rows: prompt_arithmetic names one, or auto gives
        fp4-row to an NVFP4 layer without input scales (_prompts_may_split)."""
        if self.prompt_arithmetic not in ("same", "auto"):
            return True
        return (self.prompt_arithmetic == "auto" and self._prompts_may_split() and "fp4-row" in self.own_cores and
                (any(not w4a4 for w4a4 in self.declared.values()) or
                 any(v in ROW_SCALED for v in self.prompt_chosen.values())))

    def exactness(self):
        """The level its rows keep (crit_plugins.exactness; Core compares it with the level the configuration asks).
        arithmetic declared is crit's integer block order, the CPU expert tier's bits, for every row. arithmetic
        tensor: decode rows row-exact and a prompt row the bits of the same route decoded (exact). On the kernel the
        prompt structures have the decode structure's bits; the torch arithmetic is one computation for both, which
        keeps a route's bits at every row count. Nothing before a layer is loaded, packed for or read."""
        if not (self.layers or self.declared or self.paths):
            return {}
        if self.arithmetic == "declared":
            return {"decode": "declared", "prompt": "declared"}
        if self._prompts_apart():   # a prompt row's bits its own, not the decoded route's
            return {"decode": "row-exact", "prompt": "row-exact"}
        return {"decode": "row-exact", "prompt": "exact"}   # tensor and fp4: prompt structures with the decode's bits

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
        self.width, self.capacity, self.tile = parent.width, parent.capacity, getattr(parent, "tile", False)
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

    def fp4(self):
        return {k: v[:self.routes] for k, v in self.parent.fp4().items()}

    def fp4_rows(self, terms):
        return {k: v[:self.routes] for k, v in self.parent.fp4_rows(terms).items()}

    def fp8(self):
        return {k: v[:self.routes] for k, v in self.parent.fp8().items()}


class _TorchWorkspace:
    def __init__(self, tokens, top, hidden, device):
        import torch
        self.y = torch.zeros(tokens * top, hidden, dtype=torch.float32, device=device)

    @classmethod
    def within(cls, parent, routes):
        found = cls.__new__(cls)
        found.y = parent.y[:routes]
        return found


PROMPT_ITEMS = ("auto", "fp4-prompt", "fp4-tile")
FP4_MMA = ("fp4", "fp4-row", "fp4-row2")   # the arithmetics on the FP4 tensor cores (fp4-prompt or fp4-tile items)
_REPORTS = {}


def _report(device):
    """The sweep report of grouped-experts beside the engine's tuning (crit kernels sweep writes <id>.json and
    <id>.report.json side by side) where it was swept on this device's model, else None."""
    from crit_plugins.kernels import tuning_file
    path = tuning_file(KERNEL)
    if path is None:
        return None
    if str(path) not in _REPORTS:
        try:
            _REPORTS[str(path)] = (json.loads(path.read_text(encoding="utf-8")),
                                   json.loads(path.with_name(f"{KERNEL}.report.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            _REPORTS[str(path)] = None
    found = _REPORTS[str(path)]
    if found is None:
        return None
    import torch
    swept = found[0].get("device")
    here = torch.cuda.get_device_name(device) if device is not None and torch.cuda.is_available() else None
    return found[1] if swept in (None, here) else None


def create(context, options, parts):
    return GroupedExperts(context, options, parts)
