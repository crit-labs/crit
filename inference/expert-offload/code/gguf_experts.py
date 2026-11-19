"""Routed experts as a GGUF checkpoint stores them (format "gguf"), in expert-offload's tiers: each expert one bundle of
its stored rows, cached and streamed over the link as any bundle, and computed on crit's kernel gguf-rows straight
from the slot it is in.

A GGUF file quantizes each layer's expert stacks in a format of its own (one layer IQ2_XXS, another IQ2_S, a down
projection Q2_0 or IQ4_NL), so the bundle layout is the layers' formats ({layer: (gate_up type, down type)}): an
expert's gate rows and up rows (one format: the model plugin stacks them) and, at the next 64-byte boundary, its down
rows, all as the file stores them. The store's and the cache's slots are of one size, the bundle most layers need:
the smallest layer bundle past which the layers left over (outsized: a few layers stored in a wider format) hold at
most OUTSIZED of every expert's bytes. A smaller layer's bundle is padded to the slot; an outsized layer's bundle is
stored in pieces of a slot each (its first piece at its own key, the others in extra store layers after the model's),
and the engine keeps such a layer whole on the device at its own size, outside the cache.

    found = layout({0: ("IQ2_XXS", "Q2_0"), 1: ("IQ2_S", "IQ4_NL")}, hidden, inter)
    bundle = pack(gate_up, down, found, layer)      # gate_up/down: {"weight": uint8 [rows][row bytes], "type": ...}
    part = GgufRoutes(gguf_rows, found, activation="silu")
    y = part.route_outputs(layer, x, ids, store, slots)   # store uint8 [slots][bundle_bytes], slots: expert -> slot
    out = part.combine(y, ids, weights)

The arithmetic is gguf-rows' alone: a decode row's routes on its gemv over the slots (int8 activations a 32 values,
each stored sub-block's exact integer dot, FP32 in the kernel's declared order; a route's bits are the plain call's
on its expert whatever rows share the call), the gate's activation and the product in FP32 rounded to BF16 as the
down projection reads it, and each token's routes added in route order in FP32. A prompt chunk's routes are grouped
by slot and multiplied by gguf-rows' grouped GEMM (gemm_experts: each slot's rows times its expert, the bits of its
gemm and so of its gemv), so a prompt row has the bits it has as a decode row.

GgufCpuTier is the CPU tier over the same bundles: each route's expert in gguf-rows' declared order on the host (Core's
declared_rows, the kernel's splits on the device), the bits a slot gives it, in torch: joined after a synchronization,
outside a capture."""

ALIGN = 64
MAX_ROWS = 8          # gguf-rows' gemv takes up to 8 rows a call
DECODE_ROUTES = 128   # a call of up to this many routes (or a captured one) runs the gathered gemv; the same bits
OUTSIZED = 1 / 16     # the most of every expert's bytes the layers past the slot size may hold (kept whole on the device)


def _row_bytes(type_name, k):
    from crit_core.model.gguf_blocks import TYPES, row_bytes
    if type_name not in TYPES:
        raise ValueError(f"expert-offload computes GGUF experts in {', '.join(TYPES)}, not {type_name!r}")
    return row_bytes(TYPES[type_name], k)


def _up(n):
    return -(-n // ALIGN) * ALIGN


def slot_bytes(sizes, share=OUTSIZED):
    """The slot size for layer bundles of `sizes` ({layer: bytes}): the smallest of them past which the larger layers
    hold at most `share` of all the bytes (every layer's experts are as many)."""
    total = sum(sizes.values())
    for candidate in sorted(set(sizes.values())):
        if sum(size for size in sizes.values() if size > candidate) <= share * total:
            return candidate
    return max(sizes.values())


def layout(formats, hidden, inter, share=OUTSIZED):
    """{bundle_bytes, format "gguf", hidden, inter, formats: {layer: [gate_up type, down type]}, offsets: {layer:
    down offset}, sizes: {layer: its own bundle bytes}, pieces: {outsized layer: [the extra store layers of its
    pieces past the first]}, store_layers, components}: the slot layout of every routed layer's bundles."""
    if not formats:
        raise ValueError("GGUF experts take each layer's formats: begin(..., formats={layer: (gate_up, down)})")
    found, offsets, sizes = {}, {}, {}
    for layer, (gate_up, down) in sorted((int(k), v) for k, v in dict(formats).items()):
        gu = 2 * inter * _row_bytes(gate_up, hidden)
        offsets[layer] = _up(gu)
        sizes[layer] = offsets[layer] + _up(hidden * _row_bytes(down, inter))
        found[layer] = [str(gate_up), str(down)]
    size = slot_bytes(sizes, share)
    layers = max(sizes) + 1
    pieces, extra = {}, layers
    for layer in sorted(sizes):
        if sizes[layer] > size:
            count = -(-sizes[layer] // size)
            pieces[layer] = list(range(extra, extra + count - 1))
            extra += count - 1
    fits = [layer for layer in sizes if layer not in pieces]
    widest = max(offsets[layer] for layer in fits)
    lowest = min(offsets[layer] for layer in fits)
    return {"bundle_bytes": size, "format": "gguf", "hidden": int(hidden), "inter": int(inter), "formats": found,
            "offsets": offsets, "sizes": sizes, "pieces": pieces, "store_layers": extra,
            "components": {"gate_up": (0, "uint8", (widest,)), "down": (lowest, "uint8", (size - lowest,))}}


def pieces(bundle, found, layer):
    """[(store layer, piece uint8 [bundle_bytes])] of a layer's bundle (pack's): one for a layer that fits the slot,
    a slot's bytes each for an outsized layer (its first at its own layer, the rest at its extra store layers)."""
    import torch
    layer, size = int(layer), found["bundle_bytes"]
    at = [layer] + list(found["pieces"].get(layer, ()))
    padded = torch.zeros(len(at) * size, dtype=torch.uint8)
    padded[:bundle.numel()] = bundle
    return [(where, padded[i * size:(i + 1) * size]) for i, where in enumerate(at)]


def pack(gate_up, down, found, layer):
    """One expert's bundle, uint8 [its layer's size]: its gate and up rows as stored, then at its layer's offset its
    down rows (pieces() cuts it into the store's slots)."""
    import torch
    layer = int(layer)
    if layer not in found["offsets"]:
        raise ValueError(f"layer {layer} has no GGUF formats in the layout (begin(..., formats=...))")
    want = found["formats"][layer]
    for part, name, rows, k in ((gate_up, want[0], 2 * found["inter"], found["hidden"]),
                                (down, want[1], found["hidden"], found["inter"])):
        if part.get("type") != name:
            raise ValueError(f"layer {layer}'s experts are {want[0]} / {want[1]}, not {gate_up.get('type')} / "
                             f"{down.get('type')}")
        weight = part["weight"]
        if tuple(weight.shape) != (rows, _row_bytes(name, k)):
            raise ValueError(f"a {name} part of {rows} rows of {k} values is [{rows}][{_row_bytes(name, k)}] bytes, "
                             f"not {list(weight.shape)}")
    bundle = torch.zeros(max(found["sizes"][layer], found["bundle_bytes"]), dtype=torch.uint8)
    gu = torch.as_tensor(gate_up["weight"]).reshape(-1)
    dn = torch.as_tensor(down["weight"]).reshape(-1)
    at = found["offsets"][layer]
    bundle[:gu.numel()] = gu.cpu()
    bundle[at:at + dn.numel()] = dn.cpu()
    return bundle


class GgufRoutes:
    """The device part over GGUF bundles in slots (expert-offload's part interface: prepare, route_outputs, combine)."""

    decode_arithmetic = "declared"

    def __init__(self, kernel, found, activation="silu"):
        from crit_core.model.gguf_blocks import TYPES
        if activation not in ("silu", "gelu_tanh"):
            raise ValueError(f"GGUF experts are gated by silu or gelu_tanh, not {activation!r}")
        self.kernel, self.found, self.activation = kernel, found, activation
        self.hidden, self.inter = found["hidden"], found["inter"]
        self.types = {layer: (TYPES[g], TYPES[d]) for layer, (g, d) in found["formats"].items()}
        self.a4 = False

    def prepare(self, tokens, top, experts, device, arithmetic="declared"):
        """Nothing to make ahead: the kernel's buffers are a call's own."""
        return None

    def close(self):
        pass

    def _views(self, layer, store):
        """The layer's gate-up and down stacks over every slot of `store` (uint8 [slots][slot bytes]: the cache's
        slots, or an outsized layer's own [experts][its size]), in place."""
        g, d = self.types[layer]
        rows_gu = _row_bytes_id(g, self.hidden)
        rows_dn = _row_bytes_id(d, self.inter)
        slots, size = store.shape
        flat = store.view(-1)
        gate_up = flat.as_strided((slots, 2 * self.inter, rows_gu), (size, rows_gu, 1), 0)
        down = flat.as_strided((slots, self.hidden, rows_dn), (size, rows_dn, 1), self.found["offsets"][layer])
        return (self.kernel.GgufMatrix(gate_up, g, self.hidden), self.kernel.GgufMatrix(down, d, self.inter))

    def _gate(self, both):
        import torch
        gate, up = both.float().split(self.inter, dim=-1)
        if self.activation == "silu":
            act = torch.nn.functional.silu(gate)
        else:
            act = torch.nn.functional.gelu(gate, approximate="tanh")
        return (act * up).to(torch.bfloat16)

    def route_outputs(self, layer, x, ids, store, slots, arithmetic="declared"):
        """y [tokens * top][hidden] BF16: each route's expert block, its row t * top + j for route j of token t."""
        import torch
        layer = int(layer)
        tokens, top = ids.shape
        routes = tokens * top
        gate_up, down = self._views(layer, store)
        # A route to no expert (id -1) reads store slot 0 and combine() leaves it out. Slot 0 itself, not expert 0's
        # slot: slots[0] is -1 where expert 0 has no slot, and the gathered gemv reads any slot it is given, -1 too (an
        # illegal address before the store). A real route keeps its own lookup.
        flat = ids.reshape(-1).to(torch.long)
        found = slots.to(torch.long)[flat.clamp(min=0)]
        slot = torch.where(flat >= 0, found, torch.zeros_like(found)).to(torch.int32).contiguous()
        xr = x.to(torch.bfloat16).repeat_interleave(top, dim=0)
        if routes <= DECODE_ROUTES or torch.cuda.is_current_stream_capturing():
            # Decode rows (a round's columns): the gathered gemv, MAX_ROWS routes a call.
            both = torch.cat([self.kernel.gemv(xr[a:a + MAX_ROWS], gate_up, slot[a:a + MAX_ROWS])
                              for a in range(0, routes, MAX_ROWS)])
            h = self._gate(both)
            return torch.cat([self.kernel.gemv(h[a:a + MAX_ROWS], down, slot[a:a + MAX_ROWS])
                              for a in range(0, routes, MAX_ROWS)])
        # A prompt chunk: its routes sorted by slot, each slot's rows times its expert's matrices in gguf-rows' grouped
        # GEMM (one staging and one GEMM launch a batch of experts; each row the bits of the plain call on its expert)
        order = torch.argsort(slot, stable=True)
        counts = torch.bincount(slot.to(torch.long), minlength=store.shape[0])
        used = counts.nonzero().flatten()
        experts, rows = used.tolist(), counts.index_select(0, used).tolist()   # one synchronization
        h = self._gate(self.kernel.gemm_experts(xr.index_select(0, order).contiguous(), gate_up, experts, rows))
        y = torch.empty(routes, self.hidden, dtype=torch.bfloat16, device=x.device)
        y.index_copy_(0, order, self.kernel.gemm_experts(h, down, experts, rows))
        return y

    def combine(self, y, ids, weights):
        """out [tokens][hidden] FP32: each token's routes weighted and added in route order (a route to no expert,
        id -1, adds nothing)."""
        import torch
        tokens, top = ids.shape
        rows = y.view(tokens, top, self.hidden).float()
        # A route to no expert adds +0 whatever its row holds (slot 0's bytes, not a finite row of its own).
        keep = (ids >= 0)[..., None]
        zero = torch.zeros((), device=y.device)
        out = torch.where(keep[:, 0], rows[:, 0] * weights[:, 0:1].float(), zero)
        for j in range(1, top):
            out = out + torch.where(keep[:, j], rows[:, j] * weights[:, j:j + 1].float(), zero)
        return out


class GgufCpuTier:
    """The CPU tier over GGUF bundles (host-expert-tier's CpuTier interface: add, compute, entries, stats): each route's
    expert computed on the host in gguf-rows' declared order (Core's declared_rows over the bundle's stored rows, read
    in place where the store keeps it), with the splits gguf-rows takes on the device, so a route has the bits it has
    in a slot: the gate-up rows rounded to BF16 as the kernel writes them, the gate in FP32 rounded to BF16, the down
    rows rounded to BF16. It runs in torch (not native): its routes join the device's after a synchronization, so a
    round with misses on it is not captured.

        tier = GgufCpuTier(found, experts, splits, activation="silu")   # splits: {(type id, rows, k): parts}
        tier.add(layer * experts + e, bundle)                             # a CPU uint8 bundle (kept alive here)
        y = tier.compute(keys, token, x)                                  # [routes][hidden] FP32 (BF16 values)
    """

    native = False

    def __init__(self, found, experts, splits, activation="silu"):
        from crit_core.model.gguf_blocks import TYPES
        if activation not in ("silu", "gelu_tanh"):
            raise ValueError(f"GGUF experts are gated by silu or gelu_tanh, not {activation!r}")
        self.found, self.experts, self.splits, self.activation = found, int(experts), dict(splits), activation
        self.hidden, self.inter = found["hidden"], found["inter"]
        self.types = {int(layer): (TYPES[g], TYPES[d]) for layer, (g, d) in found["formats"].items()}
        self.entries = {}
        self.reset_stats()

    def add(self, key, bundle):
        layer = int(key) // self.experts
        if layer not in self.types:
            raise ValueError(f"layer {layer} has no GGUF formats in the layout")
        self.entries[int(key)] = bundle

    def _parts(self, layer, bundle):
        """(gate-up rows, down rows) of a bundle, uint8 numpy views [rows][row bytes], and their types."""
        g, d = self.types[layer]
        rows_gu, rows_dn = _row_bytes_id(g, self.hidden), _row_bytes_id(d, self.inter)
        raw = bundle.numpy()
        at = self.found["offsets"][layer]
        gate_up = raw[:2 * self.inter * rows_gu].reshape(2 * self.inter, rows_gu)
        down = raw[at:at + self.hidden * rows_dn].reshape(self.hidden, rows_dn)
        return (gate_up, g), (down, d)

    def expert_rows(self, key, x):
        """One expert's rows of x [rows][hidden] (BF16 values), [rows][hidden] FP32 holding BF16 values."""
        import torch
        from crit_core.model.gguf_blocks import declared_rows
        layer = int(key) // self.experts
        (gu, g), (dn, d) = self._parts(layer, self.entries[int(key)])
        both = declared_rows(gu, g, self.hidden, x, self.splits[(g, 2 * self.inter, self.hidden)])
        gate, up = both.to(torch.bfloat16).float().split(self.inter, dim=-1)
        act = (torch.nn.functional.silu(gate) if self.activation == "silu"
               else torch.nn.functional.gelu(gate, approximate="tanh"))
        h = (act * up).to(torch.bfloat16).float()
        return declared_rows(dn, d, self.inter, h, self.splits[(d, self.hidden, self.inter)]).to(torch.bfloat16).float()

    def compute(self, keys, token, x, out=None):
        """Each route's unweighted row: keys and token [routes] int32 (table keys, rows of x), x [tokens][hidden]."""
        import time
        import torch
        started = time.perf_counter()
        keys, token = keys.to(torch.int64).reshape(-1), token.to(torch.int64).reshape(-1)
        x = x.to(torch.bfloat16).float()
        out = torch.zeros(keys.numel(), self.hidden, dtype=torch.float32) if out is None else out
        for key in sorted(set(keys.tolist())):
            picked = (keys == key).nonzero(as_tuple=True)[0]
            out[picked] = self.expert_rows(key, x[token[picked]])
        self._stats["jobs"] += 1
        self._stats["routes"] += int(keys.numel())
        self._stats["ms_total"] += (time.perf_counter() - started) * 1e3
        return out

    def stats(self):
        return dict(self._stats, native=False)

    def reset_stats(self):
        self._stats = {"jobs": 0, "routes": 0, "ms_total": 0.0}

    def close(self):
        self.entries = {}


def tier_splits(kernel, found, device):
    """{(type id, rows, k): parts} gguf-rows splits each layer's gate-up and down rows into on `device`: part of the
    sums' order, which the CPU tier follows."""
    import torch
    out = {}
    for layer, (g, d) in found["formats"].items():
        from crit_core.model.gguf_blocks import TYPES
        for type_id, rows, k in ((TYPES[g], 2 * found["inter"], found["hidden"]),
                                 (TYPES[d], found["hidden"], found["inter"])):
            if (type_id, rows, k) not in out:
                blocks = torch.zeros(rows, _row_bytes_id(type_id, k), dtype=torch.uint8, device=device)
                out[type_id, rows, k] = int(kernel.splits(kernel.GgufMatrix(blocks, type_id, k)))
    return out


def _row_bytes_id(type_id, k):
    from crit_core.model.gguf_blocks import row_bytes
    return row_bytes(type_id, k)
