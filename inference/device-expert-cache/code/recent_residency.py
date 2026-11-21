"""Recent residency, the rule on the host (numpy). The device runs the same rule in expert_cache._Recent and the
directory's plan (crit's kernel expert-directory, device_directory.py); trace_replay.recent replays it over a routing trace.

What is resident is what was routed last. Every expert a plan misses is computed by the CPU tier and copied into the
cache beside the round, so it serves from the next round. It takes a slot from the round's victim list: the `swaps`
open slots whose expert was routed longest ago, listed at the round's start (empty slots first, ties to the lower
slot), each taken once, and never one whose expert was routed in this round. A miss that finds the list used up stays
on the CPU tier and is copied in by a later round that misses it again. A profile's experts start in the slots in order
and leave before any routed expert does, the profile's coldest first.

    r = RecentResidency(keys=layers * experts, slots=slots, swaps=256)
    r.preload(profile_keys)                 # hottest first
    for plans in rounds:                    # one decode or verify round
        r.start_round()
        for keys in plans:                  # one routed layer: its distinct keys in the order they are first routed
            hits = r.plan(keys)             # bool per key: resident (the device computes it) or the CPU tier's
    r.reads, r.hits, r.copies, r.rounds
    r.close_slots(slots); r.open_slots(slots)   # slots lent away (a prompt's ring) and back, empty
"""
import numpy as np

EMPTY, COPYING = -1, -2
NEVER = -(1 << 31)


class RecentResidency:
    def __init__(self, keys, slots, swaps=256):
        if keys < 1 or slots < 1 or swaps < 1:
            raise ValueError("recent residency: keys, slots and copies a round")
        self.keys, self.slots, self.swaps = int(keys), int(slots), int(swaps)
        self.slot_key = np.full(self.slots, EMPTY, dtype=np.int64)
        self.slot_last = np.full(self.slots, NEVER, dtype=np.int64)
        self.key_slot = np.full(self.keys, EMPTY, dtype=np.int64)     # a slot, EMPTY, or COPYING
        self.open = np.ones(self.slots, dtype=bool)                   # False: a slot the cache cannot use now
        self.pending = []                                             # the round's copies (key, slot)
        self.victims = np.zeros(0, dtype=np.int64)
        self.cursor = 0
        self.round = 0
        self.reads = self.hits = self.copies = self.rounds = 0

    def preload(self, keys):
        """A profile's keys (hottest first) into the open slots in slot order; returns the keys placed."""
        slots = [s for s in range(self.slots) if self.open[s] and self.slot_key[s] == EMPTY]
        placed = []
        for key in keys:
            key = int(key)
            if len(placed) == len(slots):
                break
            if 0 <= key < self.keys and self.key_slot[key] == EMPTY:
                slot = slots[len(placed)]
                self.slot_key[slot], self.key_slot[key] = key, slot
                self.slot_last[slot] = -1 - len(placed)
                placed.append(key)
        return placed

    def settle(self):
        """The copies under way land and serve, each stamped with the round it was missed in."""
        for key, slot in self.pending:
            self.slot_key[slot], self.key_slot[key], self.slot_last[slot] = key, slot, self.round
        self.pending = []

    def close_slots(self, slots):
        """Slots the cache cannot use now (a prompt's ring, memory given back): their experts are dropped."""
        self.settle()
        for slot in slots:
            if self.slot_key[slot] >= 0:
                self.key_slot[self.slot_key[slot]] = EMPTY
            self.slot_key[slot], self.slot_last[slot], self.open[slot] = EMPTY, NEVER, False

    def open_slots(self, slots):
        """Closed slots come back empty."""
        for slot in slots:
            self.open[slot] = True

    def start_round(self):
        """The copies of the round before land and serve; the round's victims are listed."""
        self.settle()
        self.round += 1
        self.rounds += 1
        usable = np.flatnonzero(self.open & (self.slot_key >= EMPTY))
        order = usable[np.lexsort((usable, self.slot_last[usable]))]
        self.victims, self.cursor = order[:self.swaps], 0

    def plan(self, keys):
        """One routed layer's distinct keys, in the order they are first routed: whether each is resident. Misses
        whose copy is not under way take the round's next victims."""
        keys = np.asarray(keys, dtype=np.int64)
        where = self.key_slot[keys]
        hit = where >= 0
        self.slot_last[where[hit]] = self.round
        self.reads += keys.size
        self.hits += int(hit.sum())
        for key in keys[where == EMPTY]:
            slot = -1
            while self.cursor < self.victims.size:
                v = self.victims[self.cursor]
                self.cursor += 1
                if self.slot_key[v] >= EMPTY and self.open[v] and self.slot_last[v] != self.round:
                    slot = v
                    break
            if slot < 0:
                break
            if self.slot_key[slot] >= 0:
                self.key_slot[self.slot_key[slot]] = EMPTY
            self.slot_key[slot], self.key_slot[key] = COPYING, COPYING
            self.pending.append((int(key), int(slot)))
            self.copies += 1
        return hit

    def resident(self):
        """The resident keys."""
        return np.flatnonzero(self.key_slot >= 0)
