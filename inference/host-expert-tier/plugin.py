"""Routed experts on the CPU tier in the experts slot.

Host form: host-expert-tier's chunks and device-order FP32 sum (implementation.py), each route's row from
expert-streamed-prefill's reference block (reference.py): the oracle.

Device form on the CPU (CPU torch tensors): nvfp4 and int4 experts packed as bundles (bundles.py) and computed by the CPU
tier (cpu_tier.py) in host-expert-arithmetic's declared integer block order, every route's unweighted row, then summed
in route order by FMAs (host-expert-arithmetic's combine): the native tier where it builds, else the same order in torch,
so a token's bits never depend on the rows beside it nor on which path ran."""
from pathlib import Path

from .prefill import reference
from .tier import bundles, cpu_tier
from .tier import implementation as tier


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


class HostTierExperts:
    def __init__(self, source, options, arithmetic, quantized):
        self.source, self.max_tokens, self.options = source, options["max_tokens"], options
        self.arithmetic, self.quantized = arithmetic, quantized
        self.rows = self._build()
        self.chunks = 0
        self.tiers = {}          # (format, hidden, inter, use) -> CpuTier; use "load" or "store"
        self.keys = {}           # (layer, expert) -> table key of the load() tier
        self.tables = {}         # layer -> int32 tensor [experts] of keys (-1: not loaded)
        self.formats = {}        # layer -> (format, hidden, inter)
        self.inters = {}         # (format, hidden, store row bytes) -> inter

    def _build(self):
        """The native tier's library, built (or found built) when the plugin is created: what serves is known at load."""
        if self.options["native"] == "off":
            return "torch, the declared order (option native off)"
        native = __import__(f"{self.arithmetic.__name__}.native", fromlist=["native"])
        try:
            return f"native {cpu_tier.library_isa(cpu_tier.build_library(self.arithmetic, self.quantized))}"
        except native.NativeUnavailable as error:
            if self.options["native"] == "on":
                raise
            return f"torch, the declared order ({str(error).splitlines()[0]})"

    # ------------------------------------------------------------------ host form (the oracle)

    def run(self, layer, x, ids, weights):
        check_routes(x, ids, weights)
        routes = [(t, e, w) for t in range(len(x)) for e, w in zip(ids[t], weights[t]) if e >= 0]
        matrices = Matrices(self.source, layer)
        rows = [None] * len(routes)
        for expert, chunk in tier.chunks([e for _, e, _ in routes], self.max_tokens):
            self.chunks += 1
            for route in chunk:
                t = routes[route][0]
                row = reference.moe_block([x[t]], [[expert]], [[1.0]], matrices.gate_up, matrices.down,
                                          matrices.inter(expert))[0]
                rows[route] = [tier.f32(v) for v in row]
        scaled = [tier.f32(w) for _, _, w in routes]
        out = []
        for t in range(len(x)):
            plan = [route for route, (token, _, _) in enumerate(routes) if token == t]
            out.append(tier.device_sum(plan, scaled, rows) if plan else [0.0] * len(x[t]))
        return out

    # ------------------------------------------------------------------ device form (CPU tensors)

    def _tier(self, format, hidden, inter, use="load"):
        key = (format, hidden, inter, use)
        if key not in self.tiers:
            o = self.options
            native = {"on": True, "off": False}.get(o["native"])
            self.tiers[key] = cpu_tier.CpuTier(hidden, inter, format, self.arithmetic, self.quantized, native=native,
                                               workers=o["workers"], pin=o["pin"], spin_us=o["spin_us"],
                                               rows_per_task=o["rows_per_task"], spin_max_us=o["spin_max_us"])
        return self.tiers[key]

    def load(self, layer, expert, gate_up, down, format="nvfp4"):
        """An expert's stored tensors by role ({weight, scale, global_scale, input_scale}) in nvfp4 or int4, packed as a
        bundle the CPU tier reads; 16-bit matrices and other formats are refused (no declared integer order for them)."""
        if format not in bundles.FORMATS:
            raise ValueError(f"host-expert-tier computes {', '.join(bundles.FORMATS)} experts, not {format!r}")
        bundle = bundles.pack(gate_up, down, format)
        hidden, inter = gate_up["weight"].shape[1] * 2, gate_up["weight"].shape[0] // 2
        shape = (format, hidden, inter)
        if self.formats.setdefault(layer, shape) != shape:
            raise ValueError(f"layer {layer}'s experts are {self.formats[layer]}, not {shape}")
        key = self.keys.setdefault((layer, int(expert)), len(self.keys))
        self._tier(*shape).add(key, bundle)
        self.tables.pop(layer, None)

    def _table(self, layer):
        import torch
        if layer not in self.tables:
            experts = [e for (l, e) in self.keys if l == layer]
            table = torch.full((max(experts) + 1 if experts else 0,), -1, dtype=torch.int32)
            for e in experts:
                table[e] = self.keys[(layer, e)]
            self.tables[layer] = table
        return self.tables[layer]

    def _routes(self, ids, keys_of):
        """(keys, tokens, routes) of the routed (id >= 0) positions of ids [tokens][top], keys from keys_of(ids)."""
        import torch
        flat = ids.reshape(-1).long()
        routes = (flat >= 0).nonzero(as_tuple=True)[0]
        keys = keys_of(flat[routes])
        return keys.to(torch.int32), (routes // ids.shape[1]).to(torch.int32), routes

    def forward(self, layer, x, ids, weights):
        import torch
        if layer not in self.formats:
            raise KeyError(f"no expert of layer {layer} is loaded")
        tier_ = self._tier(*self.formats[layer])
        table = self._table(layer)

        def keys_of(experts):
            if experts.numel() and (int(experts.max()) >= table.numel() or bool((table[experts] < 0).any())):
                missing = sorted(set(experts.tolist()) - {e for (l, e) in self.keys if l == layer})
                raise KeyError(f"experts {missing} of layer {layer} are not loaded")
            return table[experts]

        keys, tokens, routes = self._routes(ids, keys_of)
        y = torch.zeros(ids.numel(), tier_.hidden, dtype=torch.float32)
        y[routes] = tier_.compute(keys, tokens, x.bfloat16().float())   # the declared order reads bf16(x)
        return tier_.combine(y, ids, weights.float())

    # experts:bundles: a store of bundles in host memory (a CPU uint8 tensor [S][bundle_bytes]) read in place.
    def layout(self, format, hidden, inter):
        return bundles.layout(format, hidden, inter)

    def pack(self, gate_up, down, format):
        return bundles.pack(gate_up, down, format)

    def forward_bundles(self, layer, x, ids, weights, store, slots, format="nvfp4"):
        import torch
        hidden = x.shape[1]
        inter = self._inter(format, hidden, store.shape[1])
        tier_ = self._tier(format, hidden, inter, "store")
        slot_of = slots.long()

        def keys_of(experts):
            picked = slot_of[experts]
            if bool((picked < 0).any()):
                raise ValueError("a routed expert has no slot in the store")
            for s in sorted(set(picked.tolist())):
                tier_.add(s, store[s])
            return picked

        keys, tokens, routes = self._routes(ids, keys_of)
        y = torch.zeros(ids.numel(), hidden, dtype=torch.float32)
        y[routes] = tier_.compute(keys, tokens, x.bfloat16().float())   # the declared order reads bf16(x)
        return tier_.combine(y, ids, weights.float())

    def _inter(self, format, hidden, row_bytes):
        """The intermediate width of a store whose rows are `row_bytes` bundles of `format` at `hidden`."""
        key = (format, hidden, row_bytes)
        if key not in self.inters:
            at = lambda i: bundles.layout(format, hidden, i)["bundle_bytes"]  # noqa: E731 - linear in inter but alignment
            guess = 64 * max(1, round((row_bytes - at(64)) / (at(128) - at(64)) + 1))
            found = [i for i in (guess - 64, guess, guess + 64) if i >= 64 and at(i) == row_bytes]
            if not found:
                raise ValueError(f"a store row of {row_bytes} bytes is no {format} bundle of hidden {hidden}")
            self.inters[key] = found[0]
        return self.inters[key]

    def facts(self):
        return {"cpu_expert_rows": self.rows}

    def telemetry(self):
        return {"cpu_tier": {f"{f}:{h}x{i}:{u}": t.stats() for (f, h, i, u), t in self.tiers.items()}}

    def close(self):
        for t in self.tiers.values():
            t.close()
        self.tiers = {}


def create(context, options, parts):
    from . import arithmetic, quantized
    return HostTierExperts(parts["expert_source"], options, arithmetic, Path(quantized.__path__[0]))
