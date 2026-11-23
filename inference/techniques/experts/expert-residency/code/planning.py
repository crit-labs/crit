"""Host plans for exact expert caching; allocation, copies and fences are external."""
from dataclasses import dataclass


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"nonempty {label} required")


@dataclass(frozen=True)
class CacheAdmission:
    phase: str
    mode: str  # populate, reuse (no replacement-policy touch), or bypass
    maximum_unique: int


def cache_admission(phase, tokens, experts_per_token, expert_count, slots, *, populate_limit=32):
    """Prefill reuses without admitting; bounded decode may populate complete bundles.

    The distinct-demand bound is conservative. Reuse/bypass never drops misses:
    the engine executes each original expert through its ordinary exact path.
    Actual protected/loading slots can still force fallback after admission.
    """
    if not isinstance(phase, str) or phase not in {"prefill", "decode"}:
        raise ValueError("explicit prefill or decode phase required")
    for value, name in ((tokens, "tokens"), (slots, "slots")):
        _integer(value, name)
    for value, name in ((experts_per_token, "experts per token"), (expert_count, "expert count"),
                        (populate_limit, "populate limit")):
        _integer(value, name, 1)
    if experts_per_token > expert_count:
        raise ValueError("selected experts exceed the layer's expert count")
    maximum = min(tokens * experts_per_token, expert_count)
    if not tokens or not slots:
        mode = "bypass"
    elif phase == "decode" and tokens <= populate_limit and maximum <= slots:
        mode = "populate"
    else:
        mode = "reuse"
    return CacheAdmission(phase, mode, maximum)


@dataclass(frozen=True)
class BankComponent:
    name: str
    encoding: str  # includes packing/scales/layout distinctions beyond shape
    shape: tuple[int, ...]  # shape of one expert component
    expert_bytes: int  # physical expert stride, including per-expert padding
    alignment: int = 1
    guard_bytes: int = 0  # backend's initialized tail-read allowance

    def validate(self):
        _text(self.name, "component name")
        _text(self.encoding, "component encoding/layout")
        if not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError("immutable nonempty expert shape required")
        for extent in self.shape:
            _integer(extent, "shape extent", 1)
        _integer(self.expert_bytes, "expert stride", 1)
        _integer(self.alignment, "bank alignment", 1)
        _integer(self.guard_bytes, "bank guard bytes")


@dataclass(frozen=True)
class LayerExperts:
    revision: str
    layer: int
    expert_count: int
    components: tuple[BankComponent, ...]
    minimum_slots: int = 1

    def validate(self):
        _text(self.revision, "model revision")
        _integer(self.layer, "layer")
        _integer(self.expert_count, "expert count", 1)
        _integer(self.minimum_slots, "minimum slots", 1)
        if self.minimum_slots > self.expert_count:
            raise ValueError("minimum slots exceed a layer's expert count")
        if not isinstance(self.components, tuple) or not self.components:
            raise ValueError("immutable complete expert components required")
        for component in self.components:
            if not isinstance(component, BankComponent):
                raise ValueError("bank component descriptor required")
            component.validate()
        if len({c.name for c in self.components}) != len(self.components):
            raise ValueError("duplicate expert component")


@dataclass(frozen=True)
class BankSpan:
    component: BankComponent
    offset: int
    nbytes: int


@dataclass(frozen=True)
class BankAllocation:
    revision: str
    layers: tuple[int, ...]
    expert_count: int
    slots: int
    spans: tuple[BankSpan, ...]
    nbytes: int


@dataclass(frozen=True)
class BankPlan:
    banks: tuple[BankAllocation, ...]
    nbytes: int
    unspent_bytes: int


def _spans(components, slots):
    if not slots:
        return (), 0
    spans, offset = [], 0
    for component in components:
        alignment = component.alignment
        offset = (offset + alignment - 1) // alignment * alignment
        payload = slots * component.expert_bytes + component.guard_bytes
        size = (payload + alignment - 1) // alignment * alignment
        spans.append(BankSpan(component, offset, size))
        offset += size
    return tuple(spans), offset


def size_banks(layers, budget_bytes):
    """Group identical layouts/revisions and split a byte budget by host payload size.

    Each group gets a separate backing allocation. Bases must satisfy every span
    alignment. Skipped groups spend zero; unused shares are not redistributed.
    This deterministic feasibility baseline does not model workload profitability.
    Reserve/reconcile actual allocations, maps and staging separately in Core.
    """
    _integer(budget_bytes, "bank budget")
    layers, groups, identities = tuple(layers), {}, set()
    for layer in layers:
        if not isinstance(layer, LayerExperts):
            raise ValueError("layer expert descriptor required")
        layer.validate()
        identity = (layer.revision, layer.layer)
        if identity in identities:
            raise ValueError("duplicate model/layer identity")
        identities.add(identity)
        groups.setdefault((layer.revision, layer.expert_count, layer.components), []).append(layer)
    weights = {key: len(group) * key[1] * sum(c.expert_bytes for c in key[2])
               for key, group in groups.items()}
    total = sum(weights.values())
    banks = []
    for (revision, experts, components), group in groups.items():
        share = budget_bytes * weights[(revision, experts, components)] // total
        low, high = 0, experts * len(group)
        while low < high:
            middle = (low + high + 1) // 2
            if _spans(components, middle)[1] <= share:
                low = middle
            else:
                high = middle - 1
        slots = low if low >= max(layer.minimum_slots for layer in group) else 0
        spans, size = _spans(components, slots)
        banks.append(BankAllocation(revision, tuple(layer.layer for layer in group), experts,
                                    slots, spans, size))
    used = sum(bank.nbytes for bank in banks)
    return BankPlan(tuple(banks), used, budget_bytes - used)


@dataclass(frozen=True)
class CopyRun:
    first_expert: int
    first_slot: int
    count: int
    source_offset: int
    destination_offset: int
    nbytes: int


def copy_runs(transfers, component, *, expert_count, slots):
    """One layer/component's copies: merge only dual-contiguous expert/slot pairs.

    Source and destination must use this same encoded stride. For repacking or
    mixed representations use a conversion kernel instead of these byte copies.
    Runs carry no owner or readiness; protect all hits and retain each ticket,
    component, source and destination through operation-specific completion.
    """
    if not isinstance(component, BankComponent):
        raise ValueError("bank component descriptor required")
    component.validate()
    _integer(expert_count, "source expert count", 1)
    _integer(slots, "destination slots")
    pairs = []
    for pair in transfers:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("expert/slot pairs required")
        expert, slot = pair
        _integer(expert, "source expert")
        _integer(slot, "destination slot")
        if expert >= expert_count or slot >= slots:
            raise ValueError("copy exceeds declared expert/slot extent")
        pairs.append((expert, slot))
    if len({e for e, _ in pairs}) != len(pairs) or len({s for _, s in pairs}) != len(pairs):
        raise ValueError("each demanded expert and destination slot must be unique")
    pairs.sort()
    runs = []
    for expert, slot in pairs:
        if runs and expert == runs[-1][0] + runs[-1][2] and slot == runs[-1][1] + runs[-1][2]:
            first_expert, first_slot, count = runs[-1]
            runs[-1] = (first_expert, first_slot, count + 1)
        else:
            runs.append((expert, slot, 1))
    stride = component.expert_bytes
    return tuple(CopyRun(e, s, n, e * stride, s * stride, n * stride) for e, s, n in runs)


@dataclass(frozen=True)
class CacheObservation:
    phase: str
    mode: str
    route_hits: int
    route_misses: int
    unique_hits: int
    unique_misses: int
    uploaded_bytes: int
    transfer_calls: int
    synchronization_ns: int


def observe_cache(admission, expert_ids, hit_ids, *, uploaded_bytes=0, transfer_calls=0,
                  synchronization_ns=0):
    """Count original route demands; physical cost fields are measured by the caller.

    Keep observations grouped by phase AND mode. Uploaded bytes do not include
    device-to-device reuse unless it was actually an upload. Duplicate routes
    count as multiple uses but only one unique materialization demand.
    """
    if (not isinstance(admission, CacheAdmission) or admission.phase not in {"prefill", "decode"}
            or admission.mode not in {"populate", "reuse", "bypass"}):
        raise ValueError("valid cache admission required")
    routes, hits = tuple(expert_ids), frozenset(hit_ids)
    for expert in routes + tuple(hits):
        _integer(expert, "original expert ID")
    unique = set(routes)
    if not hits <= unique:
        raise ValueError("cache hits must identify original demands")
    if admission.mode == "bypass" and hits:
        raise ValueError("bypassed cache cannot report cache hits")
    for value, name in ((uploaded_bytes, "uploaded bytes"), (transfer_calls, "transfer calls"),
                        (synchronization_ns, "synchronization nanoseconds")):
        _integer(value, name)
    route_hits = sum(e in hits for e in routes)
    return CacheObservation(admission.phase, admission.mode, route_hits, len(routes) - route_hits,
                            len(hits), len(unique - hits), uploaded_bytes, transfer_calls,
                            synchronization_ns)
