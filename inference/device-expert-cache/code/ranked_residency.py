"""Background ranked residency, the host policy (ranked_residency.h is its twin): the cache adapts off the decode step's
critical path.

Plans count every route and admit no misses (admission quota 0: misses go to the CPU tier or the store). Every `period`
windows, at a fixed window boundary, the policy decays each key's score, adds the routes since the last boundary, ranks
every key of every layer together and swaps the top keys that are not resident in for the lowest-ranked resident ones:
begun at that boundary (the victims unmapped, their copies started on a side stream) and committed at the next one. The
placement at every window is a function of the route history alone, never of timing.

    policy = RankedPolicy(keys=groups * experts, slots=slots)
    schedule = RankedSchedule(policy, directory)          # a device_cache.ClockDirectory with admit_quota=0
    for window in windows:                                 # each window: the plans of one decode or verify round
        schedule.run_window(window)
"""
from dataclasses import dataclass

try:
    from .device_cache import RESERVED   # imported as part of a package (a crit plugin's source)
except ImportError:
    from device_cache import RESERVED


@dataclass(frozen=True)
class RankedOptions:
    decay: float = 0.85          # a score's weight per boundary: recent routes outweigh old ones
    period: int = 3              # windows between boundaries
    max_swaps: int = 96          # swaps per boundary
    burst_swaps: int = 384       # swaps per boundary when the ranking has moved (a topic change)
    burst_fraction: float = 0.25  # ... that is, when this fraction of the top `slots` keys is not resident


class RankedPolicy:
    def __init__(self, keys, slots, options=RankedOptions()):
        o = options
        if keys < 1 or slots < 1 or not 0 < o.decay <= 1 or o.period < 1 or o.max_swaps < 0 or \
                o.burst_swaps < o.max_swaps or not 0 <= o.burst_fraction <= 1:
            raise ValueError("ranked residency: bad geometry or options")
        self.options, self.slots = o, slots
        self.scores = [0.0] * keys
        self._last = [0] * keys
        self.bursts = 0

    def boundary(self, window):
        """Window `window` (from 0) is a boundary; the first is at window `period`."""
        return window > 0 and window % self.options.period == 0

    def decide(self, counts, slot_key):
        """counts: cumulative route counts per key (as the device keeps them); slot_key: each slot's key (None empty,
        RESERVED filling). Returns the swaps [(slot, key)] to begin. Ties rank by key."""
        o = self.options
        for k, c in enumerate(counts):
            delta = (c - self._last[k]) & 0xFFFFFFFF
            self._last[k] = c
            self.scores[k] = self.scores[k] * o.decay + delta
        order = sorted(range(len(self.scores)), key=lambda k: (-self.scores[k], k))[:self.slots]
        resident = {k for k in slot_key if k is not None and k != RESERVED}
        wanted, admits = set(), []
        for k in order:
            if self.scores[k] <= 0:
                break
            wanted.add(k)
            if k not in resident:
                admits.append(k)
        empty = [s for s, k in enumerate(slot_key) if k is None]
        losing = sorted((s for s, k in enumerate(slot_key) if k is not None and k != RESERVED and k not in wanted),
                        key=lambda s: (self.scores[slot_key[s]], slot_key[s]))
        burst = len(admits) > o.burst_fraction * self.slots
        self.bursts += burst
        limit = o.burst_swaps if burst else o.max_swaps
        out, e, l = [], 0, 0
        for k in admits:
            if len(out) == limit:
                break
            if e < len(empty):
                out.append((empty[e], k))
                e += 1
            elif l < len(losing) and self.scores[slot_key[losing[l]]] < self.scores[k]:
                out.append((losing[l], k))
                l += 1
            else:
                break
        return out


class RankedSchedule:
    """Drives a ClockDirectory (device_cache.py, admit_quota 0, count_routes on) window by window: the plans of a
    window, then at a boundary the previous boundary's swaps committed, the new ones decided from the counts and begun.
    Reports hits, misses and swaps; copies happen only at boundaries."""

    def __init__(self, policy, directory):
        if not directory.count_routes:
            raise ValueError("the directory must count routes")
        self.policy, self.directory = policy, directory
        self.window = 0
        self.pending = []
        self.hits = self.misses = self.swapped = 0

    def run_window(self, plans):
        for group, experts in plans:
            p = self.directory.plan(group, experts)
            self.hits += p.hits
            self.misses += p.misses
        self.window += 1
        if self.policy.boundary(self.window):
            self.directory.commit_swaps(self.pending)
            counts = [self.directory.route_counts.get(k, 0) for k in range(len(self.policy.scores))]
            self.pending = self.policy.decide(counts, self.directory.slot_key)
            self.directory.begin_swaps(self.pending)
            self.swapped += len(self.pending)
