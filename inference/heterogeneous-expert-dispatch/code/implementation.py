"""Route-position-preserving CPU/device partition; executors belong to the caller."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Route:
    position: int
    expert: int
    weight: float


@dataclass(frozen=True)
class ResidencyGroup:
    """One indivisible allocation group in caller-defined placement order."""
    key: str
    device_bytes: int


@dataclass(frozen=True)
class ResidencyPlan:
    host_keys: tuple[str, ...]
    device_keys: tuple[str, ...]
    device_bytes: int
    headroom_bytes: int


def prefix_residency_plans(groups, available_bytes, fixed_device_bytes, headroom_bytes=0):
    """Enumerate feasible host prefixes, reserving nonexpert state and headroom.

    Extents include padding, repacking and duplicated aliases where applicable.
    This is a snapshot feasibility policy, not an allocation or a speed model.
    Callers measure candidate latency and acquire core reservations separately.
    """
    groups = tuple(groups)
    for value in (available_bytes, fixed_device_bytes, headroom_bytes):
        if type(value) is not int or value < 0:
            raise ValueError("nonnegative integer byte extents required")
    if any(not isinstance(g.key, str) or not g.key or type(g.device_bytes) is not int
           or g.device_bytes <= 0 for g in groups):
        raise ValueError("named positive allocation extents required")
    if len({g.key for g in groups}) != len(groups):
        raise ValueError("duplicate allocation group")
    if fixed_device_bytes + headroom_bytes > available_bytes:
        raise ValueError("fixed device state and headroom are infeasible")
    remaining = sum(g.device_bytes for g in groups)
    plans = []
    for count in range(len(groups) + 1):
        device_bytes = fixed_device_bytes + remaining
        if device_bytes + headroom_bytes <= available_bytes:
            plans.append(ResidencyPlan(tuple(g.key for g in groups[:count]),
                                       tuple(g.key for g in groups[count:]),
                                       device_bytes, headroom_bytes))
        if count < len(groups):
            remaining -= groups[count].device_bytes
    return tuple(plans)


def partition(routes, resident_experts, device_slots):
    import math
    routes = tuple(routes)
    if type(device_slots) is not int or device_slots < 0:
        raise ValueError("nonnegative device slot limit required")
    if len({r.position for r in routes}) != len(routes):
        raise ValueError("route positions must be unique")
    if any(type(r.position) is not int or r.position < 0 or type(r.expert) is not int or r.expert < 0
           or not math.isfinite(r.weight) for r in routes):
        raise ValueError("invalid route")
    device, host = [], []
    for route in routes:
        (device if route.expert in resident_experts and len(device) < device_slots else host).append(route)
    return tuple(device), tuple(host)


def combine(routes, contributions):
    """Contributions are already weighted; identify them by occurrence, not expert ID."""
    routes = tuple(routes)
    expected = {r.position for r in routes}
    if len(expected) != len(routes):
        raise ValueError("duplicate route positions")
    values = {}
    width = None
    for position, vector in contributions:
        vector = tuple(vector)
        if position not in expected or position in values:
            raise ValueError("unknown or duplicate contribution")
        width = len(vector) if width is None else width
        if not width or len(vector) != width:
            raise ValueError("contribution geometry differs")
        values[position] = vector
    if set(values) != expected or width is None:
        raise ValueError("every route occurrence must settle")
    return tuple(sum(values[r.position][column] for r in routes) for column in range(width))


@dataclass(frozen=True)
class ReadySlot:
    slot: int
    generation: int


@dataclass(frozen=True)
class MixedRoute:
    route: Route
    source: str  # cache, scratch or exact stock
    index: int
    generation: int = 0  # cache content generation; scratch is not published yet


@dataclass(frozen=True)
class MixedSourcePlan:
    routes: tuple[MixedRoute, ...]
    demands: tuple[tuple[int, int], ...]  # unique (expert, scratch slot)
    capacity: int

    def validate(self):
        if type(self.capacity) is not int or self.capacity < 0 or not self.routes:
            raise ValueError("nonempty routes and explicit cache capacity required")
        positions, experts, slot_owners, scratch = set(), {}, {}, {}
        for resolved in self.routes:
            route = resolved.route
            if (type(route.position) is not int or route.position < 0 or route.position in positions
                    or type(route.expert) is not int or route.expert < 0 or not math.isfinite(route.weight)):
                raise ValueError("unique valid route occurrences required")
            positions.add(route.position)
            if (resolved.source not in {"cache", "scratch", "stock"}
                    or type(resolved.index) is not int or resolved.index < 0
                    or type(resolved.generation) is not int):
                raise ValueError("invalid exact source selection")
            if resolved.source == "stock":
                if resolved.index != route.expert or resolved.generation != 0:
                    raise ValueError("stock fallback must retain the original expert identity")
            else:
                if resolved.index >= self.capacity:
                    raise ValueError("source slot exceeds declared capacity")
                if ((resolved.source == "cache" and resolved.generation <= 0)
                        or (resolved.source == "scratch" and resolved.generation != 0)):
                    raise ValueError("invalid source content generation")
                if resolved.index in slot_owners and slot_owners[resolved.index] != route.expert:
                    raise ValueError("different routed experts cannot alias a slot")
                slot_owners[resolved.index] = route.expert
                if resolved.source == "scratch":
                    scratch[route.expert] = resolved.index
            identity = (resolved.source, resolved.index, resolved.generation)
            if route.expert in experts and experts[route.expert] != identity:
                raise ValueError("repeated expert occurrences must resolve consistently")
            experts[route.expert] = identity
        if (any(type(expert) is not int or expert < 0 or type(slot) is not int or not 0 <= slot < self.capacity
                for expert, slot in self.demands)
                or len(dict(self.demands)) != len(self.demands) or dict(self.demands) != scratch):
            raise ValueError("one exact staging demand per scratch expert required")


def mixed_source_plan(routes, ready_slots, capacity, scratch_slots=()):
    """Plan one projection without splitting route occurrences into two graphs.

    READY snapshots must be protected by leases for the whole operation. Scratch
    belongs to this operation and cannot overlap any READY slot. If scratch runs
    out, preserve the original expert as exact stock fallback; never wrap slots.
    Native backends can perform this resolution on device instead of host readback.
    """
    routes, ready_slots, scratch_slots = tuple(routes), dict(ready_slots), tuple(scratch_slots)
    if type(capacity) is not int or capacity < 0:
        raise ValueError("nonnegative explicit cache capacity required")
    occupied = set()
    for expert, ready in ready_slots.items():
        if (type(expert) is not int or expert < 0 or not isinstance(ready, ReadySlot)
                or type(ready.slot) is not int or not 0 <= ready.slot < capacity
                or type(ready.generation) is not int or ready.generation <= 0
                or ready.slot in occupied):
            raise ValueError("READY sources require distinct slots and content generations")
        occupied.add(ready.slot)
    if (any(type(slot) is not int or not 0 <= slot < capacity or slot in occupied for slot in scratch_slots)
            or len(set(scratch_slots)) != len(scratch_slots)):
        raise ValueError("scratch must be distinct from every protected READY slot")
    demands, selected = {}, []
    for route in routes:
        if route.expert in ready_slots:
            ready = ready_slots[route.expert]
            selected.append(MixedRoute(route, "cache", ready.slot, ready.generation))
        else:
            if route.expert not in demands and len(demands) < len(scratch_slots):
                demands[route.expert] = scratch_slots[len(demands)]
            if route.expert in demands:
                selected.append(MixedRoute(route, "scratch", demands[route.expert]))
            else:
                selected.append(MixedRoute(route, "stock", route.expert))
    plan = MixedSourcePlan(tuple(selected), tuple(demands.items()), capacity)
    plan.validate()
    return plan


def execute_mixed(plan, activations, quantize, project):
    """Quantize once and submit one source-aware projection through explicit hooks.

    project(encoded_activations, plan) must stage complete scratch bundles before
    reading, keep exact stock fallback, and preserve route positions and weights.
    Its returned owners/readiness follow the consumer backend contract. This host
    recipe supplies no quantized kernel, CUDA launch or physical completion proof.
    """
    plan.validate()
    return project(quantize(activations), plan)
