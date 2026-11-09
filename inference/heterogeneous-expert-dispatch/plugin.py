"""Routed experts split between the device and the CPU in the experts slot.

Host form: heterogeneous-expert-dispatch's partition (implementation.py) with async-expert-contributions'
ContributionGroup (implementation.py), each route's contribution from expert-streamed-prefill's reference block
(reference.py): the oracle, bitwise the reference whichever side ran a route.

Device form on CUDA (device_dispatch.py): nvfp4 / int4 experts as bundles; the routes whose slot is -2 go to
host-expert-tier's CPU tier, posted and joined inside the stream through host-work-mailbox (one post and one join a
layer), the rest to the device part, and every token's routes are summed in route order by the device part's combine.
Both parts compute host-expert-arithmetic's declared integer block order, so a token's bits are the same whichever part
computed which route. Without a store and slot table of its own (forward(), not forward_bundles()), the plugin keeps the
first `resident` experts of each layer in a device store and sends the others to the CPU."""
from pathlib import Path

from .contributions import implementation as contributions
from .dispatch import implementation as dispatch
from .prefill import reference


def check_routes(x, ids, weights):
    """x [tokens][hidden], ids and weights [tokens][top] with one top for every token; returns top."""
    if len(ids) != len(x) or len(weights) != len(x):
        raise ValueError("one ids row and one weights row per token")
    top = len(ids[0]) if ids else 0
    if any(len(row) != top for row in ids) or any(len(row) != top for row in weights):
        raise ValueError("every token has the same number of routes and one weight per route")
    return top


class Matrices:
    """An expert_source's matrices for one call, read once per expert."""

    def __init__(self, source, layer):
        self.source, self.layer, self.cache = source, layer, {}

    def __call__(self, expert):
        if expert not in self.cache:
            self.cache[expert] = self.source.matrices(self.layer, expert)
        return self.cache[expert]

    def gate_up(self, expert):
        return self(expert)[0]

    def down(self, expert):
        return self(expert)[1]

    def inter(self, expert):
        return len(self(expert)[0]) // 2


class Settled:
    """The receipt of a contribution this host reference computed when it was submitted."""

    def __init__(self, value):
        self.value = value

    def done(self):
        return True

    def result(self):
        return self.value


class SplitExperts:
    def __init__(self, source, residency, options, code, built):
        self.source, self.residency, self.device_slots = source, residency, options["device_slots"]
        self.options, self.code, self.built = options, code, built
        self.routes = {"device": 0, "host": 0}
        self.loaded = {}         # (layer, expert) -> host bundle
        self.shape = None        # (format, hidden, inter)
        self.engine = None       # the device dispatch, made at the first forward
        self.stores = {}         # layer -> (device store, slots table)
        self.quotas = {}         # (tokens, misses) -> (SplitTimings, quota table)
        self.binding = None      # (source, layers, experts) of bind_source()
        self.quota_tensors = {}  # (tokens, misses, device) -> int32 [33] on the device

    # ------------------------------------------------------------------ host form (the oracle)

    def run(self, layer, x, ids, weights):
        top = check_routes(x, ids, weights)
        routes = [dispatch.Route(t * top + j, e, float(w)) for t in range(len(x))
                  for j, (e, w) in enumerate(zip(ids[t], weights[t])) if e >= 0]
        distinct = list(dict.fromkeys(route.expert for route in routes))
        plan = self.residency.plan(layer, distinct)
        device, host = dispatch.partition(routes, {e for e, on in zip(distinct, plan) if on}, self.device_slots)
        self.routes["device"] += len(device)
        self.routes["host"] += len(host)
        side = {route.position: "device" for route in device}
        matrices = Matrices(self.source, layer)
        groups = {}
        for route in routes:
            t = route.position // top
            if t not in groups:
                groups[t] = contributions.ContributionGroup(r.position for r in routes if r.position // top == t)
        for route in device + host:
            t = route.position // top
            vector = reference.moe_block([x[t]], [[route.expert]], [[route.weight]], matrices.gate_up,
                                         matrices.down, matrices.inter(route.expert))[0]
            groups[t].accept(route.position, Settled(vector), side.get(route.position, "host"))
        return [list(groups[t].settle()) if t in groups else [0.0] * len(x[t]) for t in range(len(x))]

    # ------------------------------------------------------------------ device form (CUDA)

    def layout(self, format, hidden, inter):
        return self.code["bundles"].layout(format, hidden, inter)

    def pack(self, gate_up, down, format):
        return self.code["bundles"].pack(gate_up, down, format)

    def load(self, layer, expert, gate_up, down, format="nvfp4"):
        """An expert's stored tensors by role in nvfp4 or int4 (the declared order's formats; others are refused): its
        bundle in host memory for the CPU tier and, for the first `resident` of its layer, in the device store."""
        bundles = self.code["bundles"]
        if format not in bundles.FORMATS:
            raise ValueError(f"heterogeneous-expert-dispatch splits {', '.join(bundles.FORMATS)} experts, not {format!r}")
        if self.engine is not None or self.binding is not None:
            raise ValueError("experts load before the first forward, and not beside a bound source")
        bundle = bundles.pack(gate_up, down, format)
        shape = (format, gate_up["weight"].shape[1] * 2, gate_up["weight"].shape[0] // 2)
        if self.shape not in (None, shape):
            raise ValueError(f"every expert is {self.shape}, not {shape}")
        self.shape = shape
        self.loaded[(int(layer), int(expert))] = bundle

    def bind_source(self, source, layers, experts, format, hidden, inter):
        """Read the CPU routes' experts in place from an expert cache's bundle store in host memory (W2's BundleStore:
        bundle_bytes, region() -> (host uint8 tensor, a CUDA tensor over its device address), address(layer, expert) ->
        the bundle's device address, None where the store holds none): no second host copy of the experts. The store's
        bundles must be this format's plain bundles (bundle_bytes and, where the store names it, its layout's
        components equal to layout(format, hidden, inter)): a store of another encoding (a CSF program) is refused.
        Then forward_bundles() takes the cache's store and slot table."""
        if self.engine is not None or self.loaded:
            raise ValueError("bind a source before the first forward, instead of loading experts")
        plan = self.code["bundles"].layout(format, int(hidden), int(inter))
        if int(getattr(source, "bundle_bytes", -1)) != plan["bundle_bytes"]:
            raise ValueError(f"the store's bundles are {getattr(source, 'bundle_bytes', None)} bytes, a {format} bundle "
                             f"of {hidden} x {inter} is {plan['bundle_bytes']}: a store of another encoding")
        theirs = getattr(source, "layout", None)
        if theirs is not None:
            components = theirs.get("components") if isinstance(theirs, dict) else None
            mine = {name: (offset, dtype, tuple(shape)) for name, (offset, dtype, shape) in plan["components"].items()}
            if components is None or {name: (int(o), str(d), tuple(sh)) for name, (o, d, sh) in components.items()} != mine:
                raise ValueError(f"the store's bundle layout is not the {format} layout the CPU tier reads in place")
        self.shape = (format, int(hidden), int(inter))
        self.binding = (source, list(layers), int(experts))

    def _host_bundles(self):
        """(stride, {(layer, expert): host bundle}) from the loaded experts or the bound source's region, each a view
        of the region's bytes (read in place) inside its bounds."""
        if self.binding is None:
            return max(e for _, e in self.loaded) + 1, self.loaded
        source, layers, experts = self.binding
        host, device = source.region()   # the device side: a CUDA tensor over the region's device address
        flat = host.view(-1)             # a view: the region itself, never a copy
        base = device.data_ptr()
        size = self.code["bundles"].layout(*self.shape)["bundle_bytes"]
        found = {}
        for layer in layers:
            for expert in range(experts):
                address = source.address(layer, expert)
                if address is None:
                    continue
                offset = int(address) - int(base)
                if offset < 0 or offset + size > flat.numel():
                    raise ValueError(f"expert {expert} of layer {layer} lies outside the store's region "
                                     f"(offset {offset}, {flat.numel()} bytes)")
                found[(layer, expert)] = flat[offset:offset + size]
        return experts, found

    def _engine(self, device):
        if self.engine is None:
            import torch
            if self.shape is None:
                raise KeyError("no expert is loaded")
            format, hidden, inter = self.shape
            o, c = self.options, self.code
            stride, host_bundles = self._host_bundles()
            native = {"on": True, "off": False}.get(o["native"])
            tier = c["cpu_tier"].CpuTier(hidden, inter, format, c["arithmetic"], c["quantized"], native=native,
                                         workers=o["workers"], pin=o["pin"], spin_us=o["spin_us"],
                                         spin_max_us=o["spin_max_us"], rows_per_task=o["rows_per_task"])
            for (layer, expert), bundle in host_bundles.items():
                tier.add(layer * stride + expert, bundle)
            device = torch.device(device)
            part = c["dispatch"].DeclaredRoutes(c["declared"], c["bundles"], format, hidden, inter, device)
            join = None if o["join"] == "auto" else o["join"]
            self.engine = c["dispatch"].DeviceDispatch(
                device, hidden, inter, format, tier, part, self.built["ext"], stride, {l for l, _ in host_bundles},
                max_tokens=o["max_tokens"], join=join, idle_after_us=o["idle_after_us"],
                idle_sleep_us=o["idle_sleep_us"], joins=self.built["joins"])
            for layer in sorted({layer for layer, _ in self.loaded}):
                experts = sorted(e for l, e in self.loaded if l == layer)
                keep = experts if o["resident"] < 0 else experts[:o["resident"]]
                slots = torch.full((stride,), -2, dtype=torch.int32)
                rows = []
                for slot, e in enumerate(keep):
                    slots[e] = slot
                    rows.append(self.loaded[(layer, e)])
                store = torch.stack(rows).to(device) if rows else torch.zeros(1, c["bundles"].layout(
                    format, hidden, inter)["bundle_bytes"], dtype=torch.uint8, device=device)
                self.stores[layer] = (store, slots.to(device))
        return self.engine

    def quota_table(self, tokens=1, misses=(1, 2, 3, 4), repeats=5, device="cuda"):
        """Link misses per miss count 0..32 (split.py's quota table, -1 where unmeasured) for `tokens` tokens an expert,
        measured on this machine with the loaded experts through captured layers of posts, link copies and joins: what
        an expert cache's admission keeps on the link, the rest of a layer's misses marked -2 for the CPU tier.
        Measured once per token count. Provisional (the facts say so): it times copy-engine copies, not the cache's own
        fills behind its decided() hook."""
        key = (int(tokens), tuple(misses))
        if key not in self.quotas:
            engine = self._engine(device)
            stride, host = self._host_bundles()
            order = sorted(host)
            bundles = [host[k] if host[k].is_pinned() else host[k].pin_memory() for k in order[:64]]
            # Layers whose experts 0..most hold bundles: the measurement posts experts 0..j-1, expert `most` resident.
            layers = [layer for layer in sorted({l for l, _ in order})
                      if all((layer, e) in host for e in range(max(misses) + 1))]
            if not layers:
                raise ValueError(f"no layer holds experts 0..{max(misses)} to measure the split with")
            self.quotas[key] = engine.calibrate(self.code["split"], bundles, layers, int(tokens), misses, repeats)
        return self.quotas[key][1]

    def quota_tensor(self, tokens=1, device="cuda", misses=(1, 2, 3, 4)):
        """quota_table(tokens) as an int32 device tensor [33], made once per token count: the table an expert cache's
        plan reads for that decode bucket (its address captured with the plan's graph, so it is never rewritten)."""
        key = (int(tokens), tuple(misses), str(device))
        if key not in self.quota_tensors:
            import torch
            self.quota_tensors[key] = torch.tensor(self.quota_table(tokens, misses, device=device), dtype=torch.int32,
                                                   device=device)
        return self.quota_tensors[key]

    def decider(self, layer, x, ids):
        """The callback an expert cache's plan(layer, ids, decided=...) calls between deciding the slot table and
        enqueueing its fills: it posts the layer's CPU routes then, so the CPU computes while the link copies; the
        forward_bundles() that follows only joins."""
        return lambda slots: self.post(layer, x, ids, slots)

    def post(self, layer, x, ids, slots=None):
        """Post the layer's CPU routes now (right after the slot table is known, before the link fills), so the CPU
        computes while the stream copies; forward() then joins them."""
        engine = self._engine(x.device)
        engine.post(layer, x, ids, self.stores[layer][1] if slots is None else slots)

    def forward(self, layer, x, ids, weights):
        engine = self._engine(x.device)
        store, slots = self.stores[layer]
        return engine.forward(layer, x, ids, weights, store, slots)

    def forward_bundles(self, layer, x, ids, weights, store, slots, format=None):
        return self._engine(x.device).forward(layer, x, ids, weights, store, slots)

    def facts(self):
        """What serves, known when the engine loads: the CPU rows (native or torch, and why), the join (or the host path,
        and why), whether a capture can join the CPU routes, the device part, and the quota's state."""
        b = self.built
        rows = f"native {b['isa']}" if b["native"] else f"torch, the declared order ({b['native_reason']})"
        if b["ext"] is not None and b["native"]:
            join = self.options["join"] if self.options["join"] != "auto" else b["joins"]["choice"]
        else:
            join = "host, after a synchronization (" + (b["ext_reason"] or "no native CPU tier") + ")"
        return {"cpu_expert_rows": rows, "cpu_join": join, "cpu_routes_capture": b["ext"] is not None and b["native"],
                "device_expert_rows": "torch, the declared order (baseline device part)",
                "cpu_quota": "provisional: measured with copy-engine copies, not the expert cache's fills"}

    def telemetry(self):
        if self.engine is None:
            return {}
        quotas = {str(tokens): table for (tokens, _), (_, table) in self.quotas.items()}
        return {"cpu_tier": self.engine.tier.stats(), "cpu_routes": self.engine.status(),
                "cpu_quota": {"provisional": True, "by_tokens": quotas}}

    def close(self):
        if self.engine is not None:
            self.engine.close()        # the mailbox answers, the streams drain, the mailbox goes
            self.engine.tier.close()   # then the tier its handler called
            self.engine = None


def build(code, options):
    """At load: the native CPU tier's library and the CUDA mailbox extension (built and cached by content), and the
    joins measured on the engine's device, so the facts say what serves before the first forward."""
    built = {"native": False, "native_reason": "", "isa": "", "ext": None, "ext_reason": None, "joins": None}
    if options["native"] != "off":
        native = __import__(f"{code['arithmetic'].__name__}.native", fromlist=["native"])
        try:
            lib = code["cpu_tier"].build_library(code["arithmetic"], code["quantized"])
            built["native"] = True
            built["isa"] = code["cpu_tier"].library_isa(lib)
        except native.NativeUnavailable as error:
            if options["native"] == "on":
                raise
            built["native_reason"] = str(error).splitlines()[0]
    else:
        built["native_reason"] = "option native off"
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if cuda and built["native"]:
        built["ext"], built["ext_reason"] = code["dispatch"].extension(code["mailbox"])
        if built["ext"] is not None:
            built["joins"] = built["ext"].measure_joins(torch.cuda.current_device(), 32, 50)
    elif not cuda:
        built["ext_reason"] = "no CUDA device"
    return built


def create(context, options, parts):
    from importlib import import_module
    from . import arithmetic, dispatch as dispatch_code, mailbox, quantized, split, tier
    code = {"bundles": import_module(f"{tier.__name__}.bundles"), "cpu_tier": import_module(f"{tier.__name__}.cpu_tier"),
            "declared": import_module(f"{arithmetic.__name__}.declared"), "arithmetic": arithmetic,
            "quantized": Path(quantized.__path__[0]), "mailbox": Path(mailbox.__path__[0]),
            "dispatch": import_module(f"{dispatch_code.__name__}.device_dispatch"),
            "split": import_module(f"{split.__name__}.split")}
    return SplitExperts(parts["expert_source"], parts["expert_residency"], options, code, build(code, options))
