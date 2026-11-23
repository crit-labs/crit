"""Measured host/device split planning; no probing, allocation or execution.

Curves describe a caller-declared workload class, including its shapes and load.
Missing measurements are unavailable, never interpolated. All occurrences of an
expert stay on one side; READY device experts stay on the device. The search is
over prefixes of an explicit miss order, not over every possible subset.
"""
from dataclasses import dataclass
import math
from statistics import median


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"nonnegative integer {name} required")


def _duration(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("finite nonnegative milliseconds required")


@dataclass(frozen=True)
class CalibrationKey:
    """Exact caller identities, not paths, marketing names or inferred defaults.

    workload includes geometry, batch/route multiplicity, residency and concurrent
    load; implementation includes kernel/configuration/thread settings; hardware
    includes devices, CPU/NUMA placement and links. Retain the inputs behind them.
    """
    model: str
    representation: str
    workload: str
    implementation: str
    hardware: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in vars(self).values()):
            raise ValueError("complete nonempty calibration identities required")


@dataclass(frozen=True)
class TimingCurve:
    """Immutable raw samples indexed by exact work count (or bytes for uploads)."""
    samples: tuple[tuple[int, tuple[float, ...]], ...]

    def __post_init__(self):
        if not isinstance(self.samples, tuple):
            raise ValueError("immutable timing samples required")
        seen = set()
        for point in self.samples:
            if not isinstance(point, tuple) or len(point) != 2:
                raise ValueError("count and samples required")
            count, times = point
            _integer(count, "work count")
            if count in seen or not isinstance(times, tuple) or not times:
                raise ValueError("unique work counts and nonempty immutable samples required")
            seen.add(count)
            for time in times:
                _duration(time)
                if count == 0 and time != 0:
                    raise ValueError("an inactive lane must have zero cost")
            _duration(median(times))
        if 0 not in seen:
            raise ValueError("explicit zero-work point required")

    @classmethod
    def from_samples(cls, samples):
        return cls(tuple((count, tuple(times)) for count, times in samples.items()))

    def at(self, count):
        """Median milliseconds, or None for an unmeasured work count."""
        _integer(count, "work count")
        return next((median(times) for n, times in self.samples if n == count), None)


@dataclass(frozen=True)
class DispatchCalibration:
    identity: CalibrationKey
    cpu: TimingCurve       # complete CPU lane, indexed by route occurrences
    upload: TimingCurve    # complete weight upload, indexed by physical bytes
    device: TimingCurve    # complete device arithmetic lane, including READY hits
    common_samples_ms: tuple[float, ...]  # common preparation and final join
    overlap: str           # serial, or cpu-device (CPU overlaps upload + device)

    def __post_init__(self):
        if not isinstance(self.identity, CalibrationKey):
            raise ValueError("calibration identity required")
        if any(not isinstance(c, TimingCurve) for c in (self.cpu, self.upload, self.device)):
            raise ValueError("measured CPU, upload and device curves required")
        if not isinstance(self.common_samples_ms, tuple) or not self.common_samples_ms:
            raise ValueError("immutable common-overhead samples required")
        for time in self.common_samples_ms:
            _duration(time)
        _duration(median(self.common_samples_ms))
        if self.overlap not in {"serial", "cpu-device"}:
            raise ValueError("explicit serial or cpu-device overlap required")


@dataclass(frozen=True)
class DispatchCandidate:
    host: tuple
    device: tuple
    uploads: tuple[tuple[int, int], ...]  # distinct expert, complete physical bytes
    estimated_ms: float | None
    missing_measurements: tuple[tuple[str, int], ...]
    host_memory_bytes: tuple[int, int, int] | None = None  # anonymous, pinned, clean file working set
    memory_admitted: bool = True
    memory_reason: str = "host memory admission not requested"

    @property
    def staging_bytes(self):
        return sum(nbytes for _, nbytes in self.uploads)


@dataclass(frozen=True)
class DispatchDecision:
    identity: CalibrationKey
    candidates: tuple[DispatchCandidate, ...]  # feasible prefix splits, device-first

    @property
    def best(self):
        """Best measured feasible split; None means use the engine's exact fallback."""
        measured = (c for c in self.candidates if c.estimated_ms is not None and c.memory_admitted)
        return min(measured, key=lambda c: c.estimated_ms, default=None)


def plan_dispatch(routes, resident_experts, upload_bytes, calibration, *, identity,
                  staging_budget_bytes, max_host_routes=None, max_device_routes=None,
                  miss_order=None, memory_costs=None, admit_host=None):
    """Rank feasible prefix splits using measured lane curves and explicit overlap.

    Inputs are existing Route objects. Nonresident experts must be executable on
    both paths; upload_bytes covers their complete bundles, including padding.
    Capacities bound route occurrences and operation-private upload storage, not
    cache size. Resident sources must remain protected until device completion.
    An optional miss_order changes the prefix family without changing reduction
    order. A decision is metadata, not a reservation or a completion receipt.
    """
    if not isinstance(calibration, DispatchCalibration) or identity != calibration.identity:
        raise ValueError("calibration does not match the current execution identity")
    if (memory_costs is None) != (admit_host is None):
        raise ValueError("memory costs and admission hook must be supplied together")
    if memory_costs is not None and (not isinstance(memory_costs, HostMemoryCosts)
            or memory_costs.identity != identity or not callable(admit_host)):
        raise ValueError("current memory-cost identity and admission hook required")
    _integer(staging_budget_bytes, "staging budget")
    for limit in (max_host_routes, max_device_routes):
        if limit is not None:
            _integer(limit, "route capacity")
    routes = tuple(routes)
    if not routes:
        raise ValueError("nonempty routes required")
    positions = set()
    for route in routes:
        _integer(route.position, "route position")
        _integer(route.expert, "expert")
        try:
            finite_weight = math.isfinite(route.weight)
        except (TypeError, OverflowError):
            finite_weight = False
        if route.position in positions or not finite_weight:
            raise ValueError("unique positions and finite route weights required")
        positions.add(route.position)
    resident_experts = tuple(resident_experts)
    for expert in resident_experts:
        _integer(expert, "resident expert")
    resident_experts = frozenset(resident_experts)
    missing = tuple(dict.fromkeys(r.expert for r in routes if r.expert not in resident_experts))
    order = missing if miss_order is None else tuple(miss_order)
    for expert in order:
        _integer(expert, "ordered expert")
    if len(order) != len(missing) or set(order) != set(missing):
        raise ValueError("miss order must contain every nonresident expert exactly once")
    extents = {}
    for expert in missing:
        if expert not in upload_bytes:
            raise ValueError("every nonresident expert needs a complete upload extent")
        nbytes = upload_bytes[expert]
        _integer(nbytes, "upload extent")
        if not nbytes:
            raise ValueError("positive upload extents required")
        extents[expert] = nbytes
    candidates = []
    for count in range(len(order) + 1):
        host_experts = set(order[:count])
        host = tuple(r for r in routes if r.expert in host_experts)
        device = tuple(r for r in routes if r.expert not in host_experts)
        uploads = tuple((e, extents[e]) for e in order[count:])
        nbytes = sum(size for _, size in uploads)
        if (nbytes > staging_budget_bytes
                or (max_host_routes is not None and len(host) > max_host_routes)
                or (max_device_routes is not None and len(device) > max_device_routes)):
            continue
        work = (("cpu", len(host)), ("upload", nbytes), ("device", len(device)))
        values = [getattr(calibration, name).at(n) for name, n in work]
        absent = tuple(w for w, v in zip(work, values) if v is None)
        estimate = None
        if not absent:
            cpu, upload, gpu = values
            device_lane = upload + gpu
            lanes = max(cpu, device_lane) if calibration.overlap == "cpu-device" else cpu + device_lane
            estimate = lanes + median(calibration.common_samples_ms)
            _duration(estimate)
        demand, allowed, reason = None, True, "host memory admission not requested"
        if memory_costs is not None:
            demand = memory_costs.requirement(host, uploads)
            allowed, reason = ((False, "missing host memory cost") if demand is None
                               else _admit(demand, admit_host))
        candidates.append(DispatchCandidate(host, device, uploads, estimate, absent,
                                             demand, allowed, reason))
    return DispatchDecision(calibration.identity, tuple(candidates))


@dataclass(frozen=True)
class HostMemoryCosts:
    """Exact incremental peak costs, bound to the dispatch calibration identity.

    Scratch is indexed by host route count; pinned staging by total upload bytes.
    Expert entries are (id, anonymous loader bytes, additional clean file pages).
    Upload-source entries have the same shape and account for pages touched while
    staging device uploads. Every used expert/source needs an explicit estimate,
    including zero. Unique CPU experts are charged once. Loader bytes sum across
    the chosen experts and sources: callers using bounded batches may supply a different
    planner, but must not understate simultaneously committed bytes here.
    Common anonymous bytes apply even to a device-only plan. All tuples immutable.
    """
    identity: CalibrationKey
    scratch: tuple[tuple[int, int], ...]
    staging: tuple[tuple[int, int], ...]
    experts: tuple[tuple[int, int, int], ...]
    common_anonymous_bytes: int = 0
    upload_sources: tuple[tuple[int, int, int], ...] = ()  # anonymous loader bytes, clean source pages

    def __post_init__(self):
        if not isinstance(self.identity, CalibrationKey):
            raise ValueError("memory-cost calibration identity required")
        _integer(self.common_anonymous_bytes, "common host bytes")
        for table, width in ((self.scratch, 2), (self.staging, 2), (self.experts, 3), (self.upload_sources, 3)):
            if not isinstance(table, tuple):
                raise ValueError("immutable memory-cost tables required")
            seen = set()
            for row in table:
                if not isinstance(row, tuple) or len(row) != width:
                    raise ValueError("invalid memory-cost row")
                for value in row:
                    _integer(value, "memory-cost field")
                if row[0] in seen:
                    raise ValueError("duplicate memory-cost key")
                seen.add(row[0])
            if width == 2 and dict(table).get(0) != 0:
                raise ValueError("inactive lane must have an explicit zero memory cost")

    def requirement(self, host, uploads):
        scratch = dict(self.scratch).get(len(host))
        pinned = dict(self.staging).get(sum(n for _, n in uploads))
        experts = {e: (anonymous, pages) for e, anonymous, pages in self.experts}
        sources = {e: (anonymous, pages) for e, anonymous, pages in self.upload_sources}
        unique = tuple(dict.fromkeys(r.expert for r in host))
        if (scratch is None or pinned is None or any(e not in experts for e in unique)
                or any(e not in sources for e, _ in uploads)):
            return None
        return (self.common_anonymous_bytes + scratch + sum(experts[e][0] for e in unique)
                + sum(sources[e][0] for e, _ in uploads),
                pinned, sum(experts[e][1] for e in unique) + sum(sources[e][1] for e, _ in uploads))


def _admit(demand, admit_host):
    decision = admit_host(anonymous_bytes=demand[0], pinned_bytes=demand[1], file_bytes=demand[2])
    if type(getattr(decision, "admitted", None)) is not bool or not isinstance(getattr(decision, "reason", None), str):
        raise ValueError("host admission must return a boolean decision and reason")
    return decision.admitted, decision.reason


def revalidate_memory(candidate, admit_host):
    """Call with a fresh observation just before allocation/submission.

    False requires another feasible measured candidate or exact fallback. An
    allocation can still fail after True; retain partial owners and do not launch
    work until all required reservations/allocations have succeeded.
    """
    if not isinstance(candidate, DispatchCandidate) or candidate.host_memory_bytes is None:
        raise ValueError("memory-accounted candidate required")
    return _admit(candidate.host_memory_bytes, admit_host)[0]
