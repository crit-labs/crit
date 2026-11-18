"""A cache's residency policies replayed over a routing trace in seconds (numpy): what a cache of S slots hits under
a policy and its options, before any engine is rebuilt.

    recent   recent_residency.RecentResidency, the rule the device runs (expert_cache._Recent): every miss copied in
             beside the round, serving from the next, into the slot of the expert routed longest ago.
    ranked   ranked_residency.RankedPolicy's rule as the device runs it (expert_cache._Ranked): plans count routes and
             admit nothing; at every `period`-th round each key's score decays and gains its routes since the last
             boundary, the top S keys are wanted, and those not resident are swapped in for empty slots, then for the
             lowest-scored residents whose score is lower, at most max_swaps a boundary (burst_swaps when more than
             burst_fraction of the wanted keys is missing). With side_swaps a boundary's swaps serve from the next
             round.
    static   the slots holding fixed keys (a profile, or the trace's own most routed keys: no profile can beat it).

A trace is the cache's own record (ExpertCache.routes(), the record_routes option): [(layer, [experts])], one row per
token and routed layer in plan order, a row (-1, []) at every round's start. A trace without those marks is cut where
the layer falls, a run of the trace's highest layer alone (a draft layer's plans before the verify pass) joined to the
round after it.

    rounds = windows(trace)                               # [[(layer, experts array), ...] per round]
    out = recent(rounds, layers, experts, slots=7000)      # {"hit_rate", "route_hit_rate", "misses_per_round", ...}
    out = ranked(rounds, layers, experts, slots=7000, decay=0.85, period=3)
    rows = sweep(rounds, layers, experts, slots=(5000, 7000, 9000), swaps=(64, 256))
    best = static(rounds, layers, experts, slots=7000)     # the same slots holding the trace's own top keys, fixed
"""
import numpy as np

MARK = -1


def windows(trace, token_rows=False):
    """The trace as rounds: each a list of plans (layer, int64 array of the plan's route ids, every row's); with
    token_rows each plan is (layer, [one array per token row]) instead, as a bench replays them."""
    rows = [(int(layer), np.asarray(ids, dtype=np.int64)) for layer, ids in trace]
    marked = any(layer == MARK for layer, _ in rows)
    top = max((layer for layer, _ in rows), default=0)
    rounds, plans, last = [], [], None
    for layer, ids in rows:
        if layer == MARK:
            if plans:
                rounds.append(plans)
            plans, last = [], None
            continue
        if last is not None and layer == last:   # the next token row of the same plan
            plans[-1] = (layer, plans[-1][1] + [ids]) if token_rows else (layer, np.concatenate([plans[-1][1], ids]))
            continue
        if not marked and last is not None and layer < last and not all(p[0] == top for p in plans):
            rounds.append(plans)
            plans = []
        plans.append((layer, [ids] if token_rows else ids))
        last = layer
    if plans:
        rounds.append(plans)
    return rounds


def _keys(plan, experts):
    layer, ids = plan
    ids = ids[ids >= 0]
    return layer * experts + ids


def ranked(rounds, layers, experts, slots, decay=0.85, period=3, max_swaps=96, burst_swaps=384, burst_fraction=0.25,
           side_swaps=True, preload=None, warmup=0):
    """The ranked policy over `rounds` with `slots` slots, from an empty cache or `preload` (keys, hottest first: a
    residency profile). The first `warmup` rounds build state and are not scored. Returns {"slots", "rounds",
    "hit_rate" (distinct experts a plan, the cache's own reads and hits), "route_hit_rate" (every route), "reads",
    "misses", "misses_per_round", "swaps", "bursts", "final_resident"}."""
    K = layers * experts
    slots = int(min(slots, K))
    resident = np.zeros(K, dtype=bool)
    if preload is not None:
        first = np.asarray(list(preload), dtype=np.int64)
        first = first[(first >= 0) & (first < K)][:slots]
        resident[first] = True
    scores = np.zeros(K, dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)
    last = np.zeros(K, dtype=np.int64)
    pending = np.zeros(0, dtype=np.int64)
    reads = hits = routes = route_hits = swaps = bursts = 0
    order_tie = np.arange(K)
    for r, plans in enumerate(rounds):
        if pending.size:                       # last boundary's side-stream swaps serve from this round
            resident[pending] = True
            pending = np.zeros(0, dtype=np.int64)
        if r > 0 and r % period == 0:
            scores = scores * decay + (counts - last)
            last = counts.copy()
            n = min(slots, int((scores > 0).sum()))
            if n:
                top = np.lexsort((order_tie, -scores))[:n]            # score descending, ties to the lower key
                admits = top[~resident[top]]
                burst = admits.size > burst_fraction * slots
                bursts += bool(burst)
                limit = burst_swaps if burst else max_swaps
                empty = slots - int(resident.sum())
                wanted = np.zeros(K, dtype=bool)
                wanted[top] = True
                losing = np.flatnonzero(resident & ~wanted)
                losing = losing[np.lexsort((losing, scores[losing]))]  # score ascending, ties to the lower key
                took, e, l = [], 0, 0
                for k in admits[:limit]:
                    if e < empty:
                        e += 1
                    elif l < losing.size and scores[losing[l]] < scores[k]:
                        resident[losing[l]] = False
                        l += 1
                    else:
                        break
                    took.append(k)
                took = np.asarray(took, dtype=np.int64)
                swaps += took.size
                if side_swaps:
                    pending = took
                else:
                    resident[took] = True
        for plan in plans:
            keys = _keys(plan, experts)
            if not keys.size:
                continue
            np.add.at(counts, keys, 1)
            if r < warmup:
                continue
            distinct = np.unique(keys)
            reads += distinct.size
            hits += int(resident[distinct].sum())
            routes += keys.size
            route_hits += int(resident[keys].sum())
    scored = max(1, len(rounds) - warmup)
    return {"policy": "ranked", "slots": slots, "rounds": len(rounds), "hit_rate": hits / reads if reads else 0.0,
            "route_hit_rate": route_hits / routes if routes else 0.0, "reads": reads, "misses": reads - hits,
            "misses_per_round": (reads - hits) / scored, "swaps": swaps, "swaps_per_round": swaps / max(1, len(rounds)),
            "bursts": bursts, "final_resident": int(resident.sum()), "decay": decay, "period": period,
            "max_swaps": max_swaps, "side_swaps": bool(side_swaps)}


def recent(rounds, layers, experts, slots, swaps=256, preload=None, warmup=0, pinned=None):
    """Recent residency (recent_residency.RecentResidency, the device cache's policy "recent") over `rounds` with
    `slots` slots and at most `swaps` copies a round, from an empty cache or `preload`. pinned: keys held for good
    in slots of their own (the cache's pinned core, ExpertCache.pin): they always hit and the rule runs in the
    slots left. Returns ranked()'s fields, and "pinned"."""
    try:
        from .recent_residency import RecentResidency
    except ImportError:   # run from code/
        from recent_residency import RecentResidency
    K = layers * experts
    slots = min(int(slots), K)
    core = np.zeros(K, dtype=bool)
    if pinned is not None:
        core[np.asarray(list(pinned), dtype=np.int64)[:max(0, slots - 1)]] = True
    held = int(core.sum())
    cache = RecentResidency(K, slots - held, swaps)
    if preload is not None:
        cache.preload([int(k) for k in preload if not core[int(k)]])
    routes = route_hits = core_reads = 0
    for r, plans in enumerate(rounds):
        if r == warmup:
            cache.reads = cache.hits = 0
        cache.start_round()
        for layer, ids in plans:
            ids = ids[ids >= 0]
            if not ids.size:
                continue
            keys = layer * experts + ids
            mine = core[keys]
            rest = keys[~mine]
            if r >= warmup:
                routes += ids.size
                route_hits += int(mine.sum()) + int((cache.key_slot[rest] >= 0).sum())
                core_reads += np.unique(keys[mine]).size
            if rest.size:
                _, at = np.unique(rest, return_index=True)
                cache.plan(rest[np.sort(at)])   # the distinct experts in the order they are first routed
    scored = max(1, len(rounds) - warmup)
    reads, hits = cache.reads + core_reads, cache.hits + core_reads
    return {"policy": "recent", "slots": slots, "rounds": len(rounds), "hit_rate": hits / reads if reads else 0.0,
            "route_hit_rate": route_hits / routes if routes else 0.0, "reads": reads, "misses": reads - hits,
            "misses_per_round": (reads - hits) / scored, "swaps": cache.copies,
            "swaps_per_round": cache.copies / max(1, len(rounds)),
            "final_resident": held + int(cache.resident().size), "max_swaps": swaps, "pinned": held}


def static(rounds, layers, experts, slots, keys=None, warmup=0):
    """`slots` slots holding fixed keys and never swapping: `keys` (hottest first: a profile), or by default the
    trace's own most routed keys, which no profile built from other text can beat. Returns ranked()'s fields."""
    K = layers * experts
    slots = int(min(slots, K))
    if keys is None:
        counts = np.zeros(K, dtype=np.int64)
        for plans in rounds:
            for plan in plans:
                np.add.at(counts, _keys(plan, experts), 1)
        keys = np.lexsort((np.arange(K), -counts))[:slots]
        keys = keys[counts[keys] > 0]
    found = ranked(rounds, layers, experts, slots, max_swaps=0, burst_swaps=0, preload=keys, warmup=warmup)
    return {**found, "policy": "static"}


def sweep(rounds, layers, experts, slots, swaps=(256,), policy="recent", decay=(0.85,), period=(3,), preload=None,
          warmup=0):
    """A policy over every combination of the given slot counts and options (recent: swaps; ranked: swaps, decay and
    period), best hit rate first within a slot count."""
    rows = []
    for s in slots:
        if policy == "recent":
            found = [recent(rounds, layers, experts, s, swaps=m, preload=preload, warmup=warmup) for m in swaps]
        else:
            found = [ranked(rounds, layers, experts, s, decay=d, period=p, max_swaps=m, burst_swaps=4 * m,
                            preload=preload, warmup=warmup) for d in decay for p in period for m in swaps]
        rows += sorted(found, key=lambda row: -row["hit_rate"])
    return rows
