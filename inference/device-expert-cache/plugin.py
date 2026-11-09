"""Device expert slots in the expert_residency slot. Host form: device-expert-cache's clock directory (device_cache.py)
under its clock policy, or under its RankedSchedule (ranked_residency.py), as the `policy` option chooses. Device form
(the "experts:cache" extension): cache(...) -> expert_cache.ExpertCache, the directory on the device, misses filled from
a bundle store in device-readable host memory, prompt chunks through a staging ring, a KV pool's unused pages lent."""
from .cache import device_cache, expert_cache, ranked_residency


class _Device:
    """The device form beside either host form: every cache the plugin made, their telemetry and facts."""

    def _setup(self, options):
        self.options, self.caches = options, []

    def cache(self, layers, experts, layout, slots, device, source, lend=None):
        """ExpertCache over `layers` x `experts` bundles of `layout` (the experts plugin's layout()), `slots` own
        slots on `device`, misses from `source` (an expert_source's open_store(): region() or read()), and lend: a KV
        pool to borrow unused pages from (attention-state's KVPool with spare bytes)."""
        o = self.options
        cache = expert_cache.ExpertCache(
            layers, experts, layout, slots, device, source, lend, policy=o["policy"], admit_quota=o["admit_quota"],
            quota_tables=quota_tables(o["quota_table"]), second_touch=o["second_touch"], cpu_tier=o["cpu_tier"],
            ring_slots=None if o["ring_slots"] < 0 else o["ring_slots"], ring=o["ring"],
            stage_routes=o["stage_routes"] or None, fill_blocks=o["fill_blocks"], fill_vector=int(o["fill_vector"]),
            decay=o["decay"], period=o["period"], max_swaps=o["max_swaps"], burst_swaps=o["burst_swaps"],
            kernel=o["native"], rows=o["round_rows"])
        self.caches.append(cache)
        return cache

    def telemetry(self):
        """{"expert_cache": the cache's telemetry} (summed over caches when the model made several)."""
        if not self.caches:
            return {}
        found = [cache.telemetry() for cache in self.caches]
        if len(found) == 1:
            return {"expert_cache": found[0]}
        total = {k: sum(t[k] for t in found) for k in found[0] if isinstance(found[0][k], int)
                 and not isinstance(found[0][k], bool)}
        return {"expert_cache": total}

    def round_problems(self):
        """Why a captured round over these caches cannot run, or [] (a file tier's misses are read on the host)."""
        return [problem for cache in self.caches for problem in cache.round_problems()]

    def facts(self):
        if not self.caches:
            return {}
        cache = self.caches[0]
        return {"expert_cache": {"directory": "cuda" if cache.directory.kernel is not None else "torch",
                                 "policy": cache.policy, "ring": cache.ring_mode, "ring_slots": len(cache.ring_slots),
                                 "lent": cache.lend is not None}}

    def close(self):
        for cache in self.caches:
            cache.close()
        self.caches = []


def quota_tables(text):
    """The quota_table option: "q0,q1,..." (one table for every plan) or "tokens:q0,q1;tokens:q0,..." (a table per
    token bucket: a plan of t tokens reads the largest bucket <= t) -> {tokens: [quota by miss count]} or None."""
    text = text.strip()
    if not text:
        return None
    tables = {}
    for part in text.split(";"):
        bucket, _, quotas = part.rpartition(":")
        tables[int(bucket) if bucket.strip() else 1] = [int(v) for v in quotas.split(",") if v.strip()]
    return tables


class ClockExpertCache(_Device):
    def __init__(self, options):
        self._setup(options)
        self.directory = device_cache.ClockDirectory(
            slots=options["slots"], groups=options["layers"], experts=options["experts"],
            admit_quota=options["admit_quota"], second_touch=options["second_touch"])
        self.stats = {"hits": 0, "misses": 0, "bypassed": 0}

    def plan(self, layer, experts):
        experts = list(experts)
        if len(set(experts)) != len(experts):
            raise ValueError("the experts of one plan must be distinct")
        limit = min(32, self.directory.slots)
        planned = self.directory.plan(layer, experts[:limit]) if experts else device_cache.Plan([], [])
        for name in self.stats:
            self.stats[name] += getattr(planned, name)
        return [isinstance(slot, int) for slot in planned.slots] + [False] * max(0, len(experts) - limit)

    def resident(self, layer):
        experts = self.directory.experts
        return {key - layer * experts for key in self.directory.key_slot if key // experts == layer}


class RankedExpertCache(_Device):
    def __init__(self, options):
        self._setup(options)
        self.directory = device_cache.ClockDirectory(slots=options["slots"], groups=options["layers"],
                                                     experts=options["experts"], admit_quota=0, count_routes=True)
        ranked = ranked_residency.RankedOptions(decay=options["decay"], period=options["period"],
                                                max_swaps=options["max_swaps"],
                                                burst_swaps=max(options["burst_swaps"], options["max_swaps"]))
        policy = ranked_residency.RankedPolicy(options["layers"] * options["experts"], options["slots"], ranked)
        self.schedule = ranked_residency.RankedSchedule(policy, self.directory)
        self.last = None

    def plan(self, layer, experts):
        experts = list(experts)
        if len(set(experts)) != len(experts):
            raise ValueError("the experts of one plan must be distinct")
        if self.last is not None and layer <= self.last:
            self.schedule.run_window([])      # the round ended: count the window, swap at a boundary
        self.last = layer
        limit = min(32, self.directory.slots)
        planned = self.directory.plan(layer, experts[:limit]) if experts else device_cache.Plan([], [])
        self.schedule.hits += planned.hits
        self.schedule.misses += planned.misses
        return [isinstance(slot, int) for slot in planned.slots] + [False] * max(0, len(experts) - limit)

    def resident(self, layer):
        experts = self.directory.experts
        return {key - layer * experts for key in self.directory.key_slot if key // experts == layer}


def create(context, options, parts):
    return (RankedExpertCache if options["policy"] == "ranked" else ClockExpertCache)(options)
