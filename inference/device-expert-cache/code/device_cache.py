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

Background ranked residency (ranked_residency.py): with count_routes every plan counts its
routes; begin_swaps unmaps and reserves slots (RESERVED: no plan uses or evicts them) and
commit_swaps maps their new keys at the next window boundary.

plan_layer is the rule the device form follows (expert_cache.py, its registry kernel and its
torch fallback, bit for bit): one plan per routed layer over all of a step's routes, ids
[tokens * top] with duplicates and -1 (not routed); the distinct experts in the order they are
first routed; hits pinned; the misses decided in that order (quota, quota by miss count,
second touch; a miss not admitted goes to the CPU tier, CPU); admitted misses take clock
victims, several in one window of the scan, never a slot pinned by the plan, RESERVED or
UNAVAILABLE (memory the cache cannot use now: reclaimed lent memory, a borrowed prompt ring).
A victim search scans at most 2 * slots slots (two passes find a victim if one exists); a miss
without one goes to the CPU tier and sets the failure bit 2. prompt_layer reads hits in place
and gives the misses ring slots in route order, never admitting.
"""
from dataclasses import dataclass, field

RESERVED = "reserved"   # slot_key of a slot a background swap is filling (kReserved on the device)
UNAVAILABLE = "unavailable"   # slot_key of a slot the cache cannot use now (reclaimed lent memory, a borrowed ring)
NOT_ROUTED, CPU = -1, -2      # slot table values besides a slot: not routed in this call; computed by the CPU tier


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
    count_routes: bool = False      # route_counts[key] += 1 per valid route (ranked residency's input)
    route_counts: dict = field(default_factory=dict)

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
        if self.count_routes:
            for k in keys:
                self.route_counts[k] = self.route_counts.get(k, 0) + 1
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
            if self.slot_pin[c] == self.stamp or self.slot_key[c] in (RESERVED, UNAVAILABLE):
                continue
            if self.slot_ref[c]:
                self.slot_ref[c] = 0
                continue
            return c
        raise RuntimeError("no victim: every slot is pinned by this plan")

    def begin_swaps(self, swaps):
        """At a window boundary: each (slot, key) slot's key is unmapped and the slot reserved; returns the (key,
        slot) store copies to run on a side stream."""
        copies = []
        for slot, key in swaps:
            old = self.slot_key[slot]
            if old is not None and old not in (RESERVED, UNAVAILABLE):
                del self.key_slot[old]
            self.slot_key[slot] = RESERVED
            self.slot_ref[slot] = 0
            copies.append((key, slot))
        return copies

    def commit_swaps(self, swaps):
        """At the next boundary, after the copies: maps each key to its reserved slot; a key that became resident
        another way meanwhile leaves its slot empty."""
        for slot, key in swaps:
            if self.slot_key[slot] != RESERVED:
                continue
            if key in self.key_slot:
                self.slot_key[slot] = None
                continue
            self.slot_key[slot] = key
            self.key_slot[key] = slot
            self.slot_ref[slot] = 0

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


@dataclass
class LayerPlan:
    table: list          # [experts]: slot, NOT_ROUTED or CPU
    fills: list          # (key, slot) copies, in admission order
    reads: int = 0       # distinct experts routed
    hits: int = 0
    cpu: int = 0         # misses sent to the CPU tier (quota, second touch, no victim)
    failure: int = 0     # bit 0: a route outside the experts; bit 1: a miss without a victim
    rows: dict = None    # row -> [reads, hits]: each row's distinct routed experts and those resident before the plan


class LayerDirectory(ClockDirectory):
    """ClockDirectory with the device form's plans (plan_layer, prompt_layer), slots the cache cannot use now
    (UNAVAILABLE: set_unavailable / set_available) and the CPU tier for misses it does not admit. admit_by_misses, when
    given, has one quota per miss count 0..experts (the last entry serves every larger count)."""

    def __post_init__(self):
        if self.admit_by_misses is not None and not self.admit_by_misses:
            raise ValueError("admit_by_misses needs at least one quota")
        table, self.admit_by_misses = self.admit_by_misses, None
        super().__post_init__()
        self.admit_by_misses = table
        self.cpu_tier = False

    def _quota(self, misses):
        if self.admit_by_misses is not None:
            return self.admit_by_misses[min(misses, len(self.admit_by_misses) - 1)]
        return self.admit_quota

    @staticmethod
    def _distinct(experts, ids):
        order, failure = {}, 0
        for i, e in enumerate(ids):
            if e == NOT_ROUTED:
                continue
            if not 0 <= e < experts:
                failure |= 1
                continue
            order.setdefault(e, i)
        return list(order), failure

    def _row_counts(self, ids, top, rows, resident):
        out = {}
        for i, e in enumerate(ids):
            if not 0 <= e < self.experts:
                continue
            r, j = rows[i // top], i - 1
            first = True
            while j >= 0 and rows[j // top] == r:
                if ids[j] == e:
                    first = False
                    break
                j -= 1
            if first:
                counts = out.setdefault(r, [0, 0])
                counts[0] += 1
                counts[1] += e in resident
        return out

    def plan_layer(self, layer, ids, top=1, rows=None):
        """One routed layer's plan over a step's routes (ids flat, tokens * top, -1 not routed; rows: the row of each
        token, for the per-row counters)."""
        if not 0 <= layer < self.groups:
            raise ValueError("layer outside the directory")
        distinct, failure = self._distinct(self.experts, ids)
        self.stamp += 1
        out = LayerPlan([NOT_ROUTED] * self.experts, [], reads=len(distinct), failure=failure)
        resident = set()
        for e in distinct:
            k = layer * self.experts + e
            if self.count_routes:
                self.route_counts[k] = self.route_counts.get(k, 0) + 1
            s = self.key_slot.get(k)
            if s is not None:
                self.slot_ref[s] = 1
                self.slot_pin[s] = self.stamp
                out.table[e] = s
                resident.add(e)
        out.hits = len(resident)
        misses = [e for e in distinct if e not in resident]
        quota, admitted, admit = self._quota(len(misses)), 0, []
        for e in misses:
            k = layer * self.experts + e
            ok = quota < 0 or admitted < quota
            if self.second_touch and k not in self.seen:
                self.seen.add(k)
                ok = False
            if ok:
                admitted += 1
                admit.append(e)
            else:
                out.table[e] = CPU
                out.cpu += 1
        for e in admit:
            s = self._search()
            if s is None:
                out.table[e] = CPU
                out.cpu += 1
                out.failure |= 2
                continue
            k = layer * self.experts + e
            if self.slot_key[s] is not None:
                del self.key_slot[self.slot_key[s]]
            self.slot_key[s], self.key_slot[k] = k, s
            self.slot_ref[s], self.slot_pin[s] = 1, self.stamp
            out.fills.append((k, s))
            out.table[e] = s
        if rows is not None:
            out.rows = self._row_counts(ids, top, rows, resident)
        return out

    def prompt_layer(self, layer, ids, ring):
        """A prompt chunk's plan: hits read in place (no reference bit, no pin), misses copied into ring slots in route
        order, nothing admitted. Returns (table, copies [(key, ring slot)], reads, hits)."""
        distinct, _ = self._distinct(self.experts, ids)
        table, copies, hits = [NOT_ROUTED] * self.experts, [], 0
        for e in distinct:
            s = self.key_slot.get(layer * self.experts + e)
            if s is not None:
                table[e] = s
                hits += 1
            else:
                if len(copies) == len(ring):
                    raise ValueError("more prompt misses than ring slots")
                table[e] = ring[len(copies)]
                copies.append((layer * self.experts + e, ring[len(copies)]))
        return table, copies, len(distinct), hits

    def _search(self):
        """The clock victim of one miss: at most 2 * slots slots scanned (the hand keeps moving), None without one."""
        for _ in range(2 * self.slots):
            c = self.hand
            self.hand = (self.hand + 1) % self.slots
            if self.slot_pin[c] == self.stamp or self.slot_key[c] in (RESERVED, UNAVAILABLE):
                continue
            if self.slot_ref[c]:
                self.slot_ref[c] = 0
                continue
            return c
        return None

    def set_unavailable(self, slots):
        """The cache cannot use these slots now: their keys are dropped and no plan takes them."""
        for s in slots:
            k = self.slot_key[s]
            if k is not None and k not in (RESERVED, UNAVAILABLE):
                del self.key_slot[k]
            self.slot_key[s], self.slot_ref[s], self.slot_pin[s] = UNAVAILABLE, 0, -1

    def set_available(self, slots):
        """Unavailable slots come back empty."""
        for s in slots:
            if self.slot_key[s] == UNAVAILABLE:
                self.slot_key[s], self.slot_ref[s], self.slot_pin[s] = None, 0, -1

    def preload(self, keys):
        """Warm start: the first valid, distinct keys not yet resident take the available slots in slot order,
        unreferenced (an empty slot or one holding a key, which is dropped); returns the (key, slot) copies. The clock
        starts after the last one."""
        copies, slots = [], [s for s in range(self.slots) if self.slot_key[s] not in (RESERVED, UNAVAILABLE)]
        for k in keys:
            if len(copies) == len(slots):
                break
            if not 0 <= k < self.groups * self.experts or k in self.key_slot:
                continue
            s = slots[len(copies)]
            if self.slot_key[s] is not None:
                del self.key_slot[self.slot_key[s]]
            self.slot_key[s], self.key_slot[k], self.slot_ref[s] = k, s, 0
            copies.append((k, s))
        if copies:
            self.hand = (copies[-1][1] + 1) % self.slots
        return copies


def simulate(directory, routes):
    """Hit rate of a sequence of (group, experts) plans."""
    hits = misses = 0
    for group, experts in routes:
        p = directory.plan(group, experts)
        hits += p.hits
        misses += p.misses
    return hits / max(1, hits + misses)
