"""What a checkpoint, its layout and the plugin's options make of the model before any weight is read: Shape.

The layer kinds and each layer's mixer, the decoder's tensor prefix, whether the checkpoint's MTP layer drafts, and the
KV pools the model will make: which layers share a pool (a sliding window's layers apart), each pool's heads, widths,
positions and page type (option kv; auto is FP8 where the checkpoint stores static k_scale and v_scale for every
attention layer, else fp8:row in quality mode speed and BF16 otherwise: precision()), the draft layer's pool (mtp_kv)
and the pools blocks declare beside their layers (an
indexer's keys, a latent attention's rows). ComposedModel is a Shape and makes those pools; the plugin's resolve()
is a Shape alone over the checkpoint's headers, so the KV an engine will take is known before it loads, and Core's
loader holds the load to it.

    shape = Shape(config, options, layout, blocks, checkpoint, context_tokens=None, drafter=False)
    shape.planned_pools()      # [{key, kind, layers, kv_heads, head_dim, value_dim, dtype, tokens, sliding}]
    shape.kv_facts(price)      # {"kv_bytes_per_token": ...}: price, the kv_cache plugin's position_bytes
"""
import json

from crit_plugins.kernels import quality_mode

from .blocks import library
from .residual import DEFAULT

PAGE = 16             # KV page tokens

# The precision options whose auto follows the quality mode the engine states (crit_plugins.kernels.quality_mode):
# in speed the 8- or 4-bit one the mode's arithmetic starts from, otherwise the stored one. kv's auto is resolved
# with the checkpoint (its static scales: Shape), the others here.
SPEED_PRECISION = {"mtp_encode": ("nvfp4", "bf16"), "mtp_kv": ("fp8:row", "bf16")}


def precision(options):
    """options with the draft layer's precision options left on auto resolved by the quality mode (SPEED_PRECISION:
    (speed's, the others')): its matrices NVFP4 and its KV pages fp8:row in speed, as stored and BF16 otherwise. The
    drafter's precision changes no published token, since the target verifies every draft: in speed the smaller
    reads only cost acceptance where they cost any."""
    speed = quality_mode() == "speed"
    return {**options, **{name: chosen[0] if speed else chosen[1] for name, chosen in SPEED_PRECISION.items()
                          if options.get(name) == "auto"}}


def _own_price(layers, kv_heads, head_dim, dtype="bf16", value_dim=None):
    """A position's bytes in a pool as the model counts them where the kv_cache plugin prices none: keys and values at
    the page type's width, a row-scaled type's FP32 scales beside them."""
    element = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1, "fp8:row": 1, "int8": 1}[dtype]
    values = head_dim if value_dim is None else int(value_dim)
    found = layers * kv_heads * (head_dim + values) * element
    if dtype in ("fp8:row", "int8"):
        found += 2 * layers * kv_heads * 4
    return found


class Shape:
    def __init__(self, config, options, layout, blocks, checkpoint, *, residuals=None, context_tokens=None,
                 drafter=False, steps=None):
        """config: the checkpoint's parsed config (crit_core.model.parse_model_config, read with its mechanisms:
        crit_models.config.architecture); options: the plugin's over
        compose's DEFAULTS; checkpoint: its headers (`name in checkpoint`, names()); drafter: whether the selection
        fills the drafter slot; steps: Core's step plan (LoadContext.steps; Core's defaults without a loader), whose
        lanes and waiting places the model sizes for, its options concurrency and pending their bounds."""
        from crit_models.config import architecture
        config = architecture(config)   # the generic fields and the mechanisms' sections and rules (crit_models)
        self.options, self.layout = precision(options), layout
        options = self.options
        found, joins, drafts = library()   # the model techniques the core mounts, then the model plugin's own
        self.blocks, self.residuals = {**found, **(blocks or {})}, {**joins, **(residuals or {})}
        self.drafters = drafts
        self.config = config
        # The layers a token runs, in order: one a checkpoint layer, or for a looped decoder (the config's layer
        # passes) a looped layer once a pass. `sources[i]` is the checkpoint layer layer i runs: the layers of one
        # source share its weights and keep their own state (KV, convolution windows), each its own layer here.
        self.sources = config.layer_applications()
        kinds = config.layer_kinds()
        self.kinds = tuple(kinds[s] for s in self.sources)
        # The config's own names of its layer types, which a layout may key its mixers by (a name Core's kinds do
        # not know is "other" there).
        types = tuple(config.layer_types) if len(config.layer_types) == len(kinds) else kinds
        self.types = tuple(types[s] for s in self.sources)
        self.layers, self.hidden = len(self.kinds), config.hidden_size
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.effective_head_dim()
        self.eps, self.vocabulary = config.rms_norm_eps or 1e-6, config.vocab_size
        if layout.get("head_rows"):   # a head stored with padded rows: the config key of the rows it predicts
            self.vocabulary = int(self.setting(layout["head_rows"]) or self.vocabulary)
        self.norm_offset = float(layout.get("norm_offset", 0.0))
        res = dict(layout.get("residual") or {"block": DEFAULT})
        if res.get("block") not in self.residuals:
            raise ValueError(f"model-template: no residual {res.get('block')!r} (residuals: "
                             f"{', '.join(sorted(self.residuals))})")
        self.context = int(context_tokens or options["context"])
        from crit_plugins.steps import Steps
        self.step_plan = steps if steps is not None else Steps()
        self.concurrency = self.step_plan.lanes(int(options["concurrency"]))
        self.pending_max = self.step_plan.pending(int(options["pending"]), self.concurrency)
        self.chunk = int(options["prefill_chunk"])
        self.sequences = self.concurrency + 1   # every request decoding at once, and a score's
        self.kept = int(options["prompt_cache"])
        self.ck = ck = checkpoint
        self.residual = self.residuals[res["block"]](self, res)
        draft = layout.get("draft") or {}
        self.draft_prefix = draft.get("prefix")
        kind = draft.get("block", "mtp")
        wanted = bool(options["mtp"]) and drafter and self.draft_prefix is not None
        self.mtp = wanted and kind == "mtp" and self.residual.draft_stored(self.draft_prefix)
        # A drafter technique's draft block beside the model (DRAFTERS: its own layers and cache, no MTP pool)
        self.draft_block = None
        self.taps = ()   # the layer applications whose entry rows a drafter technique reads (residual.py)
        if wanted and kind != "mtp":
            if kind not in self.drafters:
                raise ValueError(f"model-template: the layout drafts from {kind!r}, which no mounted drafter "
                                 f"technique gives ({', '.join(sorted(self.drafters)) or 'none'})")
            if self.drafters[kind].stored(checkpoint, self.draft_prefix):
                self.draft_block = self.drafters[kind](self, self.draft_prefix, draft)
        self.width = int(options["width"]) if self.mtp or self.draft_block is not None else 0
        # The decoder's tensor prefix: where layer 0's anchor tensor is, outside the draft layer.
        anchor = "layers.0." + self.residual.anchor()
        prefixes = [n[:-len(anchor)] for n in ck.names() if n.endswith(anchor)
                    and not (self.draft_prefix and n.startswith(self.draft_prefix))]
        if not prefixes:
            raise ValueError("model-template: no decoder layer 0 in the checkpoint")
        self.prefix = sorted(prefixes, key=len)[0]   # "model.language_model." or "model."
        # The attention layers whose keys and values the model's KV pools hold (a mixer may keep its own: keeps_kv),
        # and the pages' type.
        self.full = [i for i, k in enumerate(self.kinds) if k != "linear_attention"
                     and getattr(self._block(self._layer_spec(i)["block"]), "keeps_kv", True)]
        scales = bool(self.full) and all(f"{self.layer_prefix(i)}self_attn.k_scale" in ck for i in self.full)
        self.kv_dtype = options["kv"]
        if self.kv_dtype == "auto":
            self.kv_dtype = "fp8" if scales else "fp8:row" if quality_mode() == "speed" else "bf16"
        if self.kv_dtype == "fp8" and not scales:
            raise ValueError("model-template: fp8 KV pages take the checkpoint's static k_scale and v_scale, which it "
                             "does not store: choose kv bf16 or fp8:row")
        self.kv_scales = self.kv_dtype == "fp8"
        self.kv_tokens = self.context + int(options["kv_margin"])
        # Each window's layers share a pool, so they share a head shape (a family whose layer kinds differ in shape
        # differs in window too: its full layers a pool, its sliding ones another).
        self.windows = {}
        for i in self.full:
            self.windows.setdefault(self._window(i), []).append(i)
        self.shapes = {}
        for window, layers in self.windows.items():
            found = {self.kv_shape(i, self._layer_spec(i)) for i in layers}
            if len(found) > 1:
                raise ValueError(f"model-template: the attention layers of window {window} have key/value heads x "
                                 f"widths {sorted(found)}: layers that share a KV pool share a head shape")
            self.shapes[window] = found.pop()
        self.declared = self._declared()

    # -- the KV pools ----------------------------------------------------------------------------------------------------
    def planned_pools(self):
        """The KV pools the model makes, in the order it makes them: each window's layers ("kv", or ("kv", window) for
        a sliding window, a ring of pages a sequence), the draft layer's ("mtp", at option mtp_kv) and the pools of
        keys blocks declare (BF16, value_dim 0). The recurrence's state pools are not KV: none is listed."""
        found = []
        for window, layers in self.windows.items():
            heads, dim = self.shapes[window]
            found.append({"key": ("kv", window) if window else "kv", "kind": "kv", "layers": list(layers),
                          "kv_heads": heads, "head_dim": dim, "value_dim": None, "dtype": self.kv_dtype,
                          "tokens": self.sequences * (window + self.chunk + 2 * PAGE) if window else self.kv_tokens,
                          "sliding": window})
        if self.mtp:
            found.append({"key": "mtp", "kind": "mtp", "layers": [self.layers], "kv_heads": self.kv_heads,
                          "head_dim": self.head_dim, "value_dim": None, "dtype": self.options["mtp_kv"],
                          "tokens": self.kv_tokens, "sliding": 0})
        for d in self.declared:
            if d["kind"] == "keys":
                found.append({"key": d["key"], "kind": "keys", "layers": list(d["layers"]), "kv_heads": d["heads"],
                              "head_dim": d["dim"], "value_dim": 0, "dtype": "bf16", "tokens": self.kv_tokens,
                              "sliding": 0})
        return found

    @staticmethod
    def _price(pool, price):
        return int((price or _own_price)(len(pool["layers"]), pool["kv_heads"], pool["head_dim"], pool["dtype"],
                                         pool["value_dim"]))

    def kv_device_bytes(self, price=None):
        """The device bytes the KV pools take, whole pages (the placement plan's "kv"): price, the kv_cache plugin's
        position_bytes, else the model's own count of a position."""
        return sum(self._price(pool, price) * -(-pool["tokens"] // PAGE) * PAGE for pool in self.planned_pools())

    def kv_facts(self, price):
        """The KV facts the engine's kv_cache plugin will state of the pools made (its facts()), priced by its own
        position_bytes: kv_bytes_per_token over every pool but a sliding window's (a ring a sequence, whatever its
        length)."""
        return {"kv_bytes_per_token": sum(self._price(pool, price) for pool in self.planned_pools()
                                          if not pool["sliding"])}

    def layer_prefix(self, layer):
        """The tensor prefix of the checkpoint layer that layer `layer` runs (its source)."""
        return f"{self.prefix}layers.{self.sources[layer]}."

    # -- the layout ------------------------------------------------------------------------------------------------------
    def _spec(self, kind, name=None):
        """The layout's mixer of a layer kind (Core's: full_attention, sliding_attention, linear_attention), looked up
        first by the config's own name of the layer type where it has another."""
        mixers = self.layout.get("mixers") or {}
        spec = mixers.get(name) if name else None
        spec = mixers.get(kind) if spec is None else spec
        if spec is None:
            raise ValueError(f"model-template: the layout has no block for {name or kind} layers (its mixers: "
                             f"{', '.join(sorted(mixers)) or 'none'})")
        return {**spec, "kind": kind}

    def _layer_spec(self, layer):
        return self._spec(self.kinds[layer], self.types[layer])

    def setting(self, key):
        """The config's value of `key`, else what the family's reference assumes where config.json is silent (the
        layout's `defaults`)."""
        found = self.config.get(key)
        return (self.layout.get("defaults") or {}).get(key) if found is None else found

    def _before_layers(self, spec):
        """The layers (numbered from 1, as configs list them) a `before` block runs ahead of."""
        layers = spec["layers"]
        return set(self.config.get(layers) or ()) if isinstance(layers, str) else set(layers)

    def _before(self, layer):
        return [spec for spec in self.layout.get("before") or []
                if self.sources[layer] + 1 in self._before_layers(spec)]

    def _declared(self):
        """The pools the layout's blocks keep beside their layers' (a block class's pools(model, spec, layers)): one
        declaration a key, for the layers that run the block; the draft layer's mixer declares its own."""
        groups = {}
        for i in range(self.layers):
            spec = self._layer_spec(i)
            groups.setdefault(json.dumps(spec, sort_keys=True), (spec, []))[1].append(i)
        groups = list(groups.values())
        for spec in self.layout.get("before") or []:
            layers = [i for i in range(self.layers) if spec in self._before(i)]
            if layers:
                groups.append((spec, layers))
        if self.mtp:
            groups.append(({**self._spec("full_attention"), "window": 0, "pool": "mtp"}, [self.layers]))
        found = {}
        for spec, layers in groups:
            declare = getattr(self._block(spec["block"]), "pools", None)
            for d in (declare(self, spec, layers) if declare is not None else []):
                if d["key"] in found or d["key"] in ("kv", "states", "mtp") or d["kind"] not in ("keys", "window"):
                    raise ValueError(f"model-template: block {spec['block']} declares the pool {d['key']!r} "
                                     f"({d['kind']}), which is taken or of no kind (keys, window)")
                found[d["key"]] = d
        return list(found.values())

    def kv_shape(self, layer, spec):
        """(key/value heads, head width) of a layer's attention: what the config states for the layer itself (its
        per_layer_config entry, by the layer's number or its type), else the config keys the mixer's spec names
        (`kv_heads`, `head_dim`), else the model's."""
        per = self.config.get("per_layer_config") or {}
        source = self.sources[layer] if layer < len(self.sources) else layer
        own = per.get(str(source)) or per.get(self.types[layer] if layer < len(self.types) else None) or {}
        kv_heads = own.get("num_key_value_heads") or (self.setting(spec["kv_heads"]) if spec.get("kv_heads") else None)
        head_dim = own.get("head_dim") or (self.setting(spec["head_dim"]) if spec.get("head_dim") else None)
        return int(kv_heads or self.kv_heads), int(head_dim or self.head_dim)

    def _block(self, name):
        found = self.blocks.get(name)
        if found is None:
            raise ValueError(f"model-template: no block {name!r} (blocks: {', '.join(sorted(self.blocks))}); an engine "
                             "passes its own to compose()")
        return found

    def _window(self, layer):
        spec = self._layer_spec(layer)
        window = spec.get("window")
        if window is None and self.kinds[layer] == "sliding_attention":
            window = self.config.sliding_window
        return int(window or 0)
