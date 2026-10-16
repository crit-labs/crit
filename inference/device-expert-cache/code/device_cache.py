"""Host reference of the device expert cache directory (same rules as device_cache.cu).

A plan for one routed group pins every resident expert first, then admits each miss
in route order with a clock: starting at the hand, a slot pinned by this plan is
skipped, a referenced slot loses its bit, and the first slot that is neither becomes
the victim. A lookahead pool (two halves by group parity) can hold predicted bundles;
a miss found there is copied from the pool instead of the store.
"""
from dataclasses import dataclass, field


@dataclass
class Plan:
    slots: list          # slot per route, or None for a rejected route
    fills: list          # (source, key, slot): source is "store" or ("pool", index)
    hits: int = 0
    misses: int = 0


@dataclass
class ClockDirectory:
    slots: int
    groups: int
    experts: int
    pool_slots: int = 0
    hand: int = 0
    stamp: int = 0
    key_slot: dict = field(default_factory=dict)
    slot_key: list = None
    slot_ref: list = None
    slot_pin: list = None
    pool_tags: list = None

    def __post_init__(self):
        if self.slots < 1 or self.groups < 1 or self.experts < 1 or self.pool_slots < 0:
            raise ValueError("directory geometry must be positive")
        self.slot_key = [None] * self.slots
        self.slot_ref = [0] * self.slots
        self.slot_pin = [-1] * self.slots
        self.pool_tags = [[None] * self.pool_slots, [None] * self.pool_slots]

    def key(self, group, expert):
        if not (0 <= group < self.groups and 0 <= expert < self.experts):
            raise ValueError("route outside the directory")
        return group * self.experts + expert

    def plan(self, group, experts):
        if len(experts) > min(32, self.slots):
            raise ValueError("at most 32 routes and no more than the slot count per plan")
        keys = [self.key(group, e) for e in experts]
        if len(set(keys)) != len(keys):
            raise ValueError("routes in one plan must be distinct")
        self.stamp += 1
        out = Plan([None] * len(keys), [])
        for i, k in enumerate(keys):
            s = self.key_slot.get(k)
            if s is not None:
                self.slot_ref[s] = 1
                self.slot_pin[s] = self.stamp
                out.slots[i] = s
                out.hits += 1
        for i, k in enumerate(keys):
            if out.slots[i] is not None:
                continue
            out.misses += 1
            s = self._victim()
            if self.slot_key[s] is not None:
                del self.key_slot[self.slot_key[s]]
            self.slot_key[s] = k
            self.key_slot[k] = s
            self.slot_ref[s] = 1
            self.slot_pin[s] = self.stamp
            tags = self.pool_tags[group & 1]
            source = ("pool", tags.index(k)) if k in tags else "store"
            out.fills.append((source, k, s))
            out.slots[i] = s
        return out

    def _victim(self):
        for _ in range(4 * self.slots):
            c = self.hand
            self.hand = (self.hand + 1) % self.slots
            if self.slot_pin[c] == self.stamp:
                continue
            if self.slot_ref[c]:
                self.slot_ref[c] = 0
                continue
            return c
        raise RuntimeError("no victim: every slot is pinned by this plan")

    def prefetch(self, group, predicted):
        """Tags the pool half for `group`; returns (key, pool index) copies to make."""
        if not self.pool_slots:
            raise ValueError("no prefetch pool")
        tags = self.pool_tags[group & 1]
        copies = []
        for p in range(self.pool_slots):
            tags[p] = None
            if p >= len(predicted):
                continue
            e = predicted[p]
            if not (0 <= group < self.groups and 0 <= e < self.experts):
                continue
            k = group * self.experts + e
            if k in self.key_slot or k in tags[:p]:
                continue
            tags[p] = k
            copies.append((k, p))
        return copies


def simulate(directory, routes):
    """Hit rate of a sequence of (group, experts) plans."""
    hits = misses = 0
    for group, experts in routes:
        p = directory.plan(group, experts)
        hits += p.hits
        misses += p.misses
    return hits / max(1, hits + misses)
