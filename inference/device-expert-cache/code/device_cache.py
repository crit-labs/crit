"""Host reference of the device expert cache directory (same rules as device_cache.cu).

A plan for one routed group pins every resident expert first, then admits each miss
in route order with a clock: starting at the hand, a slot pinned by this plan is
skipped, a referenced slot loses its bit, and the first slot that is neither becomes
the victim. A lookahead pool (two halves by group parity) can hold predicted bundles;
a miss found there is copied from the pool instead of the store.

Admission limits (off by default) bypass some misses instead of admitting them:
admit_quota admits at most that many misses per plan in route order, admit_by_misses
gives that quota by the plan's miss count (split.py), and second_touch admits a key
only on its second miss. A bypassed miss takes no victim; its slot is "store" (the
engine reads the store bundle mapped or computes it on the CPU). preload maps a
residency profile's keys to the first slots before the first plan.
"""
from dataclasses import dataclass, field


@dataclass
class Plan:
    slots: list          # slot per route, "store" for a bypassed miss, or None for a rejected route
    fills: list          # (source, key, slot): source is "store" or ("pool", index)
    hits: int = 0
    misses: int = 0
    bypassed: int = 0


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
    admit_quota: int = -1           # misses admitted per plan, -1 for all
    admit_by_misses: list = None    # quota by the plan's miss count (33 entries), overrides admit_quota
    second_touch: bool = False
    seen: set = field(default_factory=set)

    def __post_init__(self):
        if self.slots < 1 or self.groups < 1 or self.experts < 1 or self.pool_slots < 0:
            raise ValueError("directory geometry must be positive")
        if self.admit_by_misses is not None and len(self.admit_by_misses) != 33:
            raise ValueError("admit_by_misses needs one quota per miss count 0-32")
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
        missing = sum(1 for slot in out.slots if slot is None)
        quota = self.admit_by_misses[missing] if self.admit_by_misses is not None else self.admit_quota
        admitted = 0
        for i, k in enumerate(keys):
            if out.slots[i] is not None:
                continue
            out.misses += 1
            admit = quota < 0 or admitted < quota
            if self.second_touch and k not in self.seen:
                self.seen.add(k)
                admit = False
            if not admit:
                out.slots[i] = "store"
                out.bypassed += 1
                continue
            admitted += 1
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

    def preload(self, keys):
        """Warm start: the first valid, distinct keys not yet resident take slots 0, 1, ...
        unreferenced; returns the (key, slot) store copies. The clock starts after them."""
        copies = []
        for k in keys:
            if len(copies) == self.slots:
                break
            if not 0 <= k < self.groups * self.experts or k in self.key_slot:
                continue
            n = len(copies)
            if self.slot_key[n] is not None:
                del self.key_slot[self.slot_key[n]]
            self.slot_key[n] = k
            self.key_slot[k] = n
            self.slot_ref[n] = 0
            copies.append((k, n))
        self.hand = len(copies) % self.slots
        return copies

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
        """Tags the pool half for `group`; returns (key, pool index) copies to make.
        Pool indices go to the nonresident, valid, distinct predictions in order."""
        if not self.pool_slots:
            raise ValueError("no prefetch pool")
        tags = self.pool_tags[group & 1]
        tags[:] = [None] * self.pool_slots
        copies = []
        for e in predicted:
            if len(copies) == self.pool_slots:
                break
            if not (0 <= group < self.groups and 0 <= e < self.experts):
                continue
            k = group * self.experts + e
            if k in self.key_slot or k in tags:
                continue
            tags[len(copies)] = k
            copies.append((k, len(copies)))
        return copies


def simulate(directory, routes):
    """Hit rate of a sequence of (group, experts) plans."""
    hits = misses = 0
    for group, experts in routes:
        p = directory.plan(group, experts)
        hits += p.hits
        misses += p.misses
    return hits / max(1, hits + misses)
