"""Offline byte-budget cache-policy comparison over complete demand groups.

This does not change the CUDA directory or simulate asynchronous copies. Hits are
pinned before admission; misses always fetch the exact requested bundle, even
when admission is bypassed. Miss costs are additive screening estimates, not
latency or throughput. Only farthest-next-use can inspect future requests.
"""
from collections import deque
from dataclasses import dataclass
import math


POLICIES = ("lru", "lfu", "clock", "farthest-next-use")
ADMISSIONS = ("always", "second-touch")


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"nonnegative integer {name} required")


@dataclass(frozen=True)
class Bundle:
    """A complete physical extent and optional measured cost of one exact miss."""
    nbytes: int
    miss_ms: float | None = None

    def __post_init__(self):
        _integer(self.nbytes, "bundle bytes")
        if not self.nbytes:
            raise ValueError("positive bundle bytes required")
        if self.miss_ms is not None:
            try:
                valid = (type(self.miss_ms) in (int, float)
                         and math.isfinite(self.miss_ms) and self.miss_ms >= 0)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("finite nonnegative miss milliseconds required")


@dataclass(frozen=True)
class Demand:
    keys: tuple[str, ...]
    phase: str = "decode"
    traffic: str = "demand"  # demand, accepted-draft, rejected-draft, prefetch


@dataclass(frozen=True)
class Capacity:
    nbytes: int


@dataclass(frozen=True)
class Decay:
    numerator: int = 1
    denominator: int = 2  # integer floor of historical frequency * numerator / denominator


@dataclass(frozen=True)
class Pin:
    identity: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class Release:
    identity: str


@dataclass(frozen=True)
class TrafficStats:
    phase: str
    traffic: str
    accesses: int
    hits: int
    misses: int
    fetch_bytes: int


@dataclass(frozen=True)
class ReplayResult:
    policy: str
    admission: str
    capacity_bytes: int
    warmup_groups: int
    scored_groups: int
    demands: int              # distinct bundles per group, not route occurrences
    hits: int
    misses: int
    fetch_bytes: int          # one complete backing-store fetch per distinct miss
    admissions: int
    bypasses: int             # exact misses served without retaining the bundle
    evictions: int
    estimated_miss_ms: float | None
    peak_resident_bytes: int  # includes warmup; excludes external staging/scratch
    final_keys: tuple[str, ...]
    final_capacity_bytes: int = 0
    capacity_rejections: int = 0
    traffic: tuple[TrafficStats, ...] = ()
    wasted_prefetch_bytes: int = 0  # fetched then evicted without useful demand
    unused_prefetch_bytes: int = 0  # still resident at end; later usefulness unknown

    @property
    def hit_rate(self):
        return self.hits / self.demands if self.demands else 0.0


def _inputs(trace, bundles, capacity_bytes, warmup_groups):
    _integer(capacity_bytes, "cache capacity")
    _integer(warmup_groups, "warmup groups")
    bundles = dict(bundles)
    if any(not isinstance(k, str) or not k or not isinstance(v, Bundle)
           for k, v in bundles.items()):
        raise ValueError("nonempty content/representation keys and Bundle values required")
    groups, demands = [], 0
    for group in trace:
        if isinstance(group, Capacity):
            _integer(group.nbytes, "changed cache capacity")
            groups.append(group)
            continue
        if isinstance(group, Decay):
            _integer(group.numerator, "decay numerator")
            _integer(group.denominator, "decay denominator")
            if not group.denominator or group.numerator > group.denominator:
                raise ValueError("decay factor must be in [0, 1]")
            groups.append(group)
            continue
        if isinstance(group, Release):
            if not isinstance(group.identity, str) or not group.identity:
                raise ValueError("pin identity required")
            groups.append(group)
            continue
        event = group if isinstance(group, (Demand, Pin)) else Demand(group)
        if isinstance(event, Pin):
            if not isinstance(event.identity, str) or not event.identity:
                raise ValueError("pin identity required")
        elif event.phase not in {"prefill", "decode"} or event.traffic not in {
                "demand", "accepted-draft", "rejected-draft", "prefetch"}:
            raise ValueError("explicit known phase and traffic class required")
        group = event.keys
        if isinstance(group, (str, bytes)):
            raise ValueError("a demand group must be a sequence of keys")
        keys = tuple(group)
        if any(not isinstance(k, str) or k not in bundles for k in keys):
            raise ValueError("every demand must identify a declared exact bundle")
        keys = tuple(dict.fromkeys(keys))
        groups.append(Pin(event.identity, keys) if isinstance(event, Pin)
                      else Demand(keys, event.phase, event.traffic))
        demands += isinstance(event, Demand)
    if warmup_groups > demands:
        raise ValueError("warmup exceeds the trace")
    return tuple(groups), bundles


def replay(trace, bundles, capacity_bytes, *, policy="lru", admission="always", warmup_groups=0,
           prefill_populates=False, protect_decode_groups=0):
    """Replay from an empty cache; warmup builds state but is excluded from scores.

    Keys include model/content revision, layer/expert and representation. A group
    names all bundles used together: existing hits and newly admitted misses stay
    pinned until it ends. Repeated occurrences fetch once. Oversized bundles or
    groups can use exact uncached service; that staging memory is engine-owned.

    LFU counts decode-demand accesses with explicit optional decay and breaks
    ties by recency. Second-touch needs two retained decode-demand counts.
    Farthest-next-use is a hindsight
    eviction reference; with unequal sizes or groups it is not an optimal bound.
    """
    if policy not in POLICIES or admission not in ADMISSIONS:
        raise ValueError("unknown replacement or admission policy")
    groups, bundles = _inputs(trace, bundles, capacity_bytes, warmup_groups)
    return _replay(groups, bundles, capacity_bytes, policy, admission, warmup_groups,
                   prefill_populates, protect_decode_groups)


def compare_policies(trace, bundles, capacity_bytes, *, policies=POLICIES,
                     admission="always", warmup_groups=0, prefill_populates=False,
                     protect_decode_groups=0):
    """Independent runs of the same trace, catalog, budget and warmup, in input order."""
    policies = tuple(policies)
    if (not policies or len(set(policies)) != len(policies)
            or any(p not in POLICIES for p in policies) or admission not in ADMISSIONS):
        raise ValueError("distinct supported policies and a supported admission required")
    groups, bundles = _inputs(trace, bundles, capacity_bytes, warmup_groups)
    return tuple(_replay(groups, bundles, capacity_bytes, p, admission, warmup_groups,
                         prefill_populates, protect_decode_groups) for p in policies)


def _replay(groups, bundles, capacity, policy, admission, warmup, prefill_populates, protect_decode_groups):
    if type(prefill_populates) is not bool:
        raise ValueError("explicit prefill population policy required")
    _integer(protect_decode_groups, "recent decode protection")
    # Only the hindsight policy gets the future. Online policies use prefix state.
    future = {}
    if policy == "farthest-next-use":
        for index, group in enumerate(groups):
            if not isinstance(group, Demand):
                continue
            for key in group.keys:
                future.setdefault(key, deque()).append(index)
    resident, frequency, recency = {}, {}, {}
    clock = deque()
    used = peak = stamp = 0
    hits = misses = fetch_bytes = admissions = bypasses = evictions = 0
    cost, complete_cost = 0.0, True
    initial_capacity, group_count, decode_count = capacity, 0, 0
    pins, recent, prefetched, stats = {}, {}, {}, {}
    capacity_rejections = wasted = 0

    def victim(pinned):
        available = [k for k in resident if k not in pinned]
        if not available:
            raise ValueError("no unprotected victim")
        if policy == "clock":
            while True:
                key = clock.popleft()
                if key in pinned:
                    clock.append(key)
                elif resident[key]:
                    resident[key] = False
                    clock.append(key)
                else:
                    return key
        if policy == "lru":
            return min(available, key=lambda k: recency[k])
        if policy == "lfu":
            return min(available, key=lambda k: (frequency.get(k, 0), recency[k]))
        return max(available, key=lambda k: future[k][0] if future[k] else math.inf)

    def evict(key, scored):
        nonlocal used, evictions, wasted
        del resident[key]
        used -= bundles[key].nbytes
        evictions += int(scored)
        wasted += prefetched.pop(key, 0)

    for event in groups:
        scored = group_count >= warmup
        protected = set().union(*pins.values()) if pins else set()
        if isinstance(event, Pin):
            if event.identity in pins or not event.keys or any(k not in resident for k in event.keys):
                raise ValueError("pin must uniquely identify currently resident content")
            pins[event.identity] = set(event.keys)
            continue
        if isinstance(event, Release):
            if event.identity not in pins:
                raise ValueError("unknown or already released pin")
            del pins[event.identity]
            continue
        if isinstance(event, Decay):
            frequency = {k: v * event.numerator // event.denominator for k, v in frequency.items()}
            continue
        if isinstance(event, Capacity):
            if sum(bundles[k].nbytes for k in protected) > event.nbytes:
                capacity_rejections += int(scored)  # Refuse atomically; no partial eviction.
                continue
            while used > event.nbytes:
                evict(victim(protected), scored)
            capacity = event.nbytes
            continue
        keys = event.keys
        group_count += 1
        promotes = event.phase == "decode" and event.traffic == "demand"
        if promotes:
            decode_count += 1
        elif protect_decode_groups:
            protected |= {k for k, at in recent.items() if decode_count - at < protect_decode_groups}
        present = set(keys) & resident.keys()
        pinned = present | protected
        for key in keys:
            stamp += 1
            if promotes:
                frequency[key] = frequency.get(key, 0) + 1
                recency[key] = stamp
                recent[key] = decode_count
            if key in present and promotes:
                resident[key] = True
            if key in present and event.traffic in {"demand", "accepted-draft"}:
                prefetched.pop(key, None)  # Outcome affects accounting, never replacement.
            if future:
                future[key].popleft()
        bucket = stats.setdefault((event.phase, event.traffic), [0, 0, 0, 0])
        if scored:
            hits += len(present)
            bucket[0] += len(keys)
            bucket[1] += len(present)
        for key in keys:
            if key in present:
                continue
            bundle = bundles[key]
            if scored:
                misses += 1
                fetch_bytes += bundle.nbytes
                bucket[2] += 1
                bucket[3] += bundle.nbytes
                if bundle.miss_ms is None:
                    complete_cost = False
                else:
                    cost += bundle.miss_ms
                    if not math.isfinite(cost):
                        raise ValueError("aggregate miss cost overflow")
            reclaimable = sum(bundles[k].nbytes for k in resident if k not in pinned)
            if ((event.phase == "prefill" and not prefill_populates)
                    or (admission == "second-touch" and frequency.get(key, 0) < 2)
                    or bundle.nbytes > capacity - used + reclaimable):
                bypasses += int(scored)
                if scored and event.traffic == "prefetch":
                    wasted += bundle.nbytes
                continue  # no partial eviction when the group cannot admit it
            while used + bundle.nbytes > capacity:
                evict(victim(pinned), scored)
            resident[key] = True
            recency.setdefault(key, stamp)
            if not promotes:
                recency[key] = stamp
            if event.traffic == "prefetch":
                prefetched[key] = bundle.nbytes if scored else 0
            if policy == "clock":
                clock.append(key)
            pinned.add(key)
            used += bundle.nbytes
            peak = max(peak, used)
            admissions += int(scored)
    return ReplayResult(policy, admission, initial_capacity, warmup, group_count - warmup,
                        hits + misses, hits, misses, fetch_bytes, admissions,
                        bypasses, evictions, cost if complete_cost else None,
                        peak, tuple(sorted(resident)), capacity, capacity_rejections,
                        tuple(TrafficStats(*key, *values) for key, values in sorted(stats.items()) if values[0]),
                        wasted, sum(prefetched.values()))
