"""The device expert cache's directory in device tensors, and its operations: the plan of one routed layer (decode,
admitting misses), the plan of a prompt chunk (misses into ring slots, never admitting), the fill of a plan's copies,
and the host-driven changes (slots made unavailable or available again, preload, background swaps).

The rule is device_cache.LayerDirectory's, bit for bit. Plans and fills run directory_ops.cu (built at load,
native.py) and otherwise the torch ops below: fixed shapes, no host synchronization in plan and prompt plans
(capture-safe), the clock's victims by cumulative sums over the slots in scan order. A torch fill under capture copies
the plan's most possible copies (the rest into the scratch slot); outside capture it reads the copy count and copies
only those. The host-driven changes are torch ops either way.

Tensors (int32 unless said): key_slot [keys + 1] (slot or -1; the last entry a dummy), slot_key [slots + 1] (key, -1
empty, -2 reserved by a swap, -3 unavailable), slot_ref uint8 [slots + 1], slot_pin [slots + 1] (stamp of the last plan
using the slot), state [4] (clock hand, plan stamp, failure bits: 1 a route outside the experts, 2 a miss without a
victim), counters int64 [8] (COUNTERS), seen uint8 [keys + 1], route_counts int64 [keys + 1] (or empty), quota tables
int32 [q] by miss count, one per token bucket (a plan of t tokens reads the table of the largest bucket <= t; below
every bucket, or with none, admit_quota), per layer the slot table [experts], copies [experts + 1][2] (key, slot; the last a dummy)
and copy count [1].
"""
EMPTY, RESERVED, UNAVAILABLE = -1, -2, -3
NOT_ROUTED, CPU = -1, -2
COUNTERS = ("reads", "hits", "fills", "cpu", "prompt_reads", "prompt_hits", "prompt_copies", "spare")


def _kernel(device):
    try:
        from . import native
    except ImportError:   # run from code/
        import native
    return native.directory_ops(device)


class Program:
    """A store layout's fill program (expert_store.csf_layout): store bundles of `store_bytes` rebuilt into slot
    bundles op by op (COPY a component, CSF decode a compressed scale matrix, then its exceptions)."""

    def __init__(self, store_layout, device):
        import torch
        self.store_bytes = int(store_layout["bundle_bytes"])
        self.ops = [list(map(int, op)) for op in store_layout["program"]]
        self.tensor = torch.tensor(self.ops, dtype=torch.int64, device=device)
        self.shared = max([16 * (1 + op[4] // 2) for op in self.ops if op[0] == 1] or [16])

    def fill(self, directory, store, source, jobs, count, scratch, limit, blocks=0):
        torch = directory.torch
        if directory.kernel is not None:
            if not blocks:   # its slab decodes wait on the link: twice the multiprocessors (measured best of 0.5-2x)
                blocks = 2 * torch.cuda.get_device_properties(store.device).multi_processor_count
            directory.kernel.fill_program(source, store, jobs, count, self.tensor, self.shared, int(blocks))
            return
        if directory.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            m = max(1, min(int(limit), jobs.shape[0]))
            live = torch.arange(m, device=directory.device) < count
            keys = torch.where(live, jobs[:m, 0].long(), 0)
            slots = torch.where(live, jobs[:m, 1].long(), scratch)
            self._rebuild(store, source.index_select(0, keys), slots)
            return
        g = int(count.reshape(-1)[0].item())
        if g:
            pairs = jobs[:g].long()
            self._rebuild(store, source.index_select(0, pairs[:, 0]), pairs[:, 1])

    def _rebuild(self, store, raw, slots):
        """Slots `slots` of store from raw store bundles [m][store_bytes], by the program, in torch."""
        import torch
        try:
            from . import expert_store
        except ImportError:
            import expert_store
        m = raw.shape[0]
        for op, src, dst, a, b, c, d, _ in self.ops:
            if op == 0:
                store[:, dst:dst + a].index_copy_(0, slots, raw[:, src:src + a])
                continue
            fixed = raw[:, src:src + a * (1 + b // 2)].reshape(m, a // 16, 16 * (1 + b // 2))
            words = raw[:, c:c + 4 * d].contiguous().view(torch.int32).reshape(m, d)
            scales = expert_store.csf_decode(fixed, words).reshape(m, a * b)
            store[:, dst:dst + a * b].index_copy_(0, slots, scales)


class DeviceDirectory:
    def __init__(self, layers, experts, slots, device, second_touch=False, count_routes=False, admit_quota=-1,
                 quota=None, kernel=True):
        import torch
        if layers < 1 or experts < 1 or slots < 1:
            raise ValueError("a directory has layers, experts and slots")
        self.torch, self.device = torch, torch.device(device)
        self.layers, self.experts, self.slots = layers, experts, slots
        self.keys = layers * experts
        i32 = dict(dtype=torch.int32, device=self.device)
        self.key_slot = torch.full((self.keys + 1,), EMPTY, **i32)
        self.slot_key = torch.full((slots + 1,), EMPTY, **i32)
        self.slot_ref = torch.zeros(slots + 1, dtype=torch.uint8, device=self.device)
        self.slot_pin = torch.full((slots + 1,), -1, **i32)
        self.state = torch.zeros(4, **i32)
        self.counters = torch.zeros(len(COUNTERS), dtype=torch.int64, device=self.device)
        self.seen = torch.zeros(self.keys + 1, dtype=torch.uint8, device=self.device)
        self.second_touch = bool(second_touch)
        self.route_counts = torch.zeros(self.keys + 1 if count_routes else 0, dtype=torch.int64, device=self.device)
        self.admit_quota = int(admit_quota)
        # quota: {tokens: [quota by miss count]} (a table per token bucket), each a device tensor made once
        self.quotas = sorted((int(t), torch.tensor(list(q), **i32)) for t, q in (quota or {}).items())
        if any(t < 1 or not q.numel() for t, q in self.quotas):
            raise ValueError("a quota table is a token bucket of 1 or more and one quota per miss count")
        self._no_quota = torch.zeros(0, **i32)
        self._admit = torch.tensor(self.admit_quota, dtype=torch.int64, device=self.device)   # made once: capture
        self.tables = torch.full((layers, experts), NOT_ROUTED, **i32)
        self.copies = torch.zeros(layers, experts + 1, 2, **i32)
        self.copy_count = torch.zeros(layers, **i32)
        self._ar_e = torch.arange(experts, dtype=torch.int64, device=self.device)
        self._ar_s = torch.arange(slots, dtype=torch.int64, device=self.device)
        self._empty_i64 = torch.zeros(0, dtype=torch.int64, device=self.device)
        self._empty_i32 = torch.zeros(0, **i32)
        self._work = {}
        self.kernel = _kernel(self.device) if kernel and self.device.type == "cuda" else None

    # -- the plans ---------------------------------------------------------------------------------------------------
    def plan(self, layer, ids, top, rows=None, row_counters=None, ring=None):
        """Plans layer `layer` over ids ([tokens][top] or flat, int64 or int32, on the device): the slot table goes to
        tables[layer], the copies to copies[layer] and their count to copy_count[layer]. ring (an int32 device tensor
        of ring slots) makes it a prompt plan. rows ([tokens] int64) and row_counters ([rows][2] int64) add each row's
        reads and hits."""
        if not 0 <= layer < self.layers:
            raise ValueError(f"layer {layer} outside the directory's {self.layers}")
        flat = ids.reshape(-1)
        quota = self.quota_for(flat.numel() // int(top))
        if self.kernel is not None:
            self.kernel.plan(self.key_slot, self.slot_key, self.slot_ref, self.slot_pin, self.state, self.counters,
                             self.seen if self.second_touch else self._empty_u8(), self.route_counts, quota,
                             self.admit_quota, layer, self.experts, flat, int(top),
                             self._empty_i64 if rows is None else rows,
                             self._empty_i64 if row_counters is None else row_counters,
                             self._empty_i32 if ring is None else ring,
                             self.tables[layer], self.copies[layer], self.copy_count[layer:layer + 1])
            return
        if ring is None:
            self._plan_torch(layer, flat.long(), int(top), rows, row_counters, quota)
        else:
            self._prompt_torch(layer, flat.long(), ring)

    def quota_for(self, tokens):
        """The quota table of a plan of `tokens` tokens (empty: admit_quota)."""
        found = self._no_quota
        for bucket, table in self.quotas:
            if bucket <= tokens:
                found = table
        return found

    def _empty_u8(self):
        return self.seen[:0]

    def _first(self, flat):
        """Each expert's first route index ([experts], n where not routed) and the routes outside the experts."""
        torch = self.torch
        n, E = flat.numel(), self.experts
        valid = (flat >= 0) & (flat < E)
        bad = ((flat < -1) | (flat >= E)).any()
        self.state[2:3].bitwise_or_(bad.to(torch.int32))
        first = torch.full((E + 1,), n, dtype=torch.int64, device=self.device)
        first.scatter_reduce_(0, torch.where(valid, flat, E), torch.arange(n, device=self.device), reduce="amin")
        return first[:E], valid

    def _plan_torch(self, layer, flat, top, rows, row_counters, table):
        torch = self.torch
        E, S, K, n = self.experts, self.slots, self.keys, flat.numel()
        first, valid = self._first(flat)
        routed = first < n
        keys = layer * E + self._ar_e
        slot_of = self.key_slot[keys].long()
        hit = routed & (slot_of >= 0)
        miss = routed & ~hit
        self.state[1:2].add_(1)
        stamp = self.state[1].clone()
        hs = torch.where(hit, slot_of, S)
        self.slot_ref.index_fill_(0, hs, 1)
        self.slot_pin.index_copy_(0, hs, stamp.expand(E).contiguous())
        if self.route_counts.numel():
            self.route_counts.index_add_(0, torch.where(routed, keys, K), routed.long())
        order = torch.argsort(torch.where(miss, first, n), stable=True)
        misses = miss.sum()
        if table.numel():   # index_select: a 0-dim index would read the device on the host
            quota = table.index_select(0, misses.clamp(max=table.numel() - 1).view(1)).squeeze(0).long()
        else:
            quota = self._admit
        if self.second_touch:
            eligible = miss & self.seen[keys].bool()
            self.seen.index_fill_(0, torch.where(miss, keys, K), 1)
        else:
            eligible = miss
        eligible_sorted = eligible[order]
        admitted_sorted = eligible_sorted & ((quota < 0) | (torch.cumsum(eligible_sorted.long(), 0) <= quota))
        admitted = torch.zeros_like(miss)
        admitted[order] = admitted_sorted
        rank = torch.empty_like(order)
        rank[order] = torch.cumsum(admitted_sorted.long(), 0) - 1
        count = admitted_sorted.sum()
        victims, found = self._victims(count, stamp)
        ok = admitted & (rank < found)
        victim = torch.where(ok, victims[rank.clamp(0, E)], -1)
        vs = torch.where(ok, victim, S)
        old = self.slot_key[vs].long()
        self.key_slot.index_fill_(0, torch.where(ok & (old >= 0), old, K), EMPTY)
        self.slot_key.index_copy_(0, vs, torch.where(ok, keys, EMPTY).int())
        self.key_slot.index_copy_(0, torch.where(ok, keys, K), vs.int())
        self.slot_ref.index_fill_(0, vs, 1)
        self.slot_pin.index_copy_(0, vs, stamp.expand(E).contiguous())
        self.key_slot[K:].fill_(EMPTY)
        self.slot_key[S:].fill_(EMPTY)
        self.slot_ref[S:].fill_(0)
        self.slot_pin[S:].fill_(-1)
        table = torch.where(hit, slot_of, torch.where(ok, victim, torch.where(miss, CPU, NOT_ROUTED)))
        self.tables[layer].copy_(table)
        jobs = self.copies[layer]
        jobs.index_copy_(0, torch.where(ok, rank, E), torch.stack([keys, victim], 1).int())
        self.copy_count[layer:layer + 1].copy_(found.view(1))
        self.state[2:3].bitwise_or_(((count > found).int() * 2).view(1))
        self.counters[0:4].add_(torch.stack([routed.sum(), hit.sum(), found, misses - found]))
        if rows is not None and row_counters is not None:
            self._row_counts(flat, valid, hit, top, rows, row_counters)

    def _victims(self, count, stamp):
        """The clock's victims for `count` admitted misses: victims [experts + 1] (slot by rank, the last a dummy) and
        how many exist; the reference bits and the hand advance as a scan one slot at a time would leave them."""
        torch = self.torch
        E, S = self.experts, self.slots
        hand = self.state[0].long()
        o = (hand + self._ar_s) % S
        key = self.slot_key[:S]
        blocked = ((self.slot_pin[:S] == stamp) | (key == RESERVED) | (key == UNAVAILABLE))[o]
        referenced = self.slot_ref[:S].bool()[o]
        free = ~blocked & ~referenced
        second = ~blocked & referenced
        cf = torch.cumsum(free.long(), 0)
        cr = torch.cumsum(second.long(), 0)
        F, R = cf[-1], cr[-1]
        victims = torch.full((E + 1,), -1, dtype=torch.int64, device=self.device)
        victims.scatter_(0, torch.where(free & (cf <= count), cf - 1, E).clamp(max=E), o)
        victims.scatter_(0, torch.where(second & (F + cr <= count), F + cr - 1, E).clamp(max=E), o)
        victims[E:].fill_(-1)
        found = torch.minimum(count, F + R)
        p1 = torch.searchsorted(cf, count.view(1)).squeeze(0)
        cleared = second & (count > 0) & ((count > F) | (self._ar_s < p1))
        self.slot_ref.index_fill_(0, torch.where(cleared, o, S), 0)
        p2 = torch.searchsorted(cr, (count - F).clamp(min=1).view(1)).squeeze(0)
        last_second = torch.searchsorted(cr, R.clamp(min=1).view(1)).squeeze(0)
        last_free = torch.searchsorted(cf, F.clamp(min=1).view(1)).squeeze(0)
        last = torch.where(R > 0, last_second, last_free)
        step = torch.where(count <= F, p1 + 1, torch.where(count <= F + R, p2 + 1, last + 1))
        step = torch.where((count == 0) | (F + R == 0), torch.zeros_like(step), step)
        self.state[0:1].copy_(((hand + step) % S).int().view(1))
        return victims, found

    def _row_counts(self, flat, valid, hit, top, rows, row_counters):
        torch = self.torch
        E, n = self.experts, flat.numel()
        tokens = rows.numel()
        pos = torch.arange(n, device=self.device)
        row = rows.long()[pos // top]
        boundary = torch.ones(tokens, dtype=torch.int64, device=self.device)
        if tokens > 1:
            boundary[1:] = (rows[1:] != rows[:-1]).long()
        run = (torch.cumsum(boundary, 0) - 1)[pos // top]
        bucket = torch.where(valid, run * E + flat.clamp(0, E - 1), tokens * E)
        first = torch.full((tokens * E + 1,), n, dtype=torch.int64, device=self.device)
        first.scatter_reduce_(0, bucket, pos, reduce="amin")
        first_in = valid & (first[bucket] == pos)
        hits = first_in & hit[flat.clamp(0, E - 1)]
        row_counters[:, 0].index_add_(0, row, first_in.long())
        row_counters[:, 1].index_add_(0, row, hits.long())

    def _prompt_torch(self, layer, flat, ring):
        torch = self.torch
        E, n = self.experts, flat.numel()
        first, _ = self._first(flat)
        routed = first < n
        keys = layer * E + self._ar_e
        slot_of = self.key_slot[keys].long()
        hit = routed & (slot_of >= 0)
        miss = routed & ~hit
        order = torch.argsort(torch.where(miss, first, n), stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.cumsum(miss[order].long(), 0) - 1
        ok = miss & (rank < ring.numel())
        dest = ring.long()[rank.clamp(0, ring.numel() - 1)]
        table = torch.where(hit, slot_of, torch.where(ok, dest, torch.where(miss, CPU, NOT_ROUTED)))
        self.tables[layer].copy_(table)
        self.copies[layer].index_copy_(0, torch.where(ok, rank, E), torch.stack([keys, dest], 1).int())
        copied = ok.sum()
        self.copy_count[layer:layer + 1].copy_(copied.view(1))
        self.state[2:3].bitwise_or_(((miss.sum() > copied).int() * 2).view(1))
        self.counters[4:7].add_(torch.stack([routed.sum(), hit.sum(), copied]))

    # -- fills -------------------------------------------------------------------------------------------------------
    def fill(self, layer, store, source, scratch, limit, blocks=0, vector=16, program=None):
        """The copies of layer's last plan: bundles of `source` (a device-readable uint8 tensor [keys][bundle_bytes]:
        a mapped host store) into `store` ([slots][bundle_bytes] on the device). limit: the most copies the plan can
        have (its distinct routes, at most the experts). program: a Program (store bundles in another layout, CSF
        scales: expert_store.csf_layout) to rebuild each slot by."""
        self.fill_counted(self.copies[layer], self.copy_count[layer:layer + 1], store, source, scratch, limit, blocks,
                          vector, program)

    def fill_counted(self, jobs, count, store, source, scratch, limit, blocks=0, vector=16, program=None):
        """fill() over given copies: jobs [n + 1][2] (key, slot) int32 and their count [1] int32, on the device."""
        if program is not None:
            program.fill(self, store, source, jobs, count, scratch, limit, blocks)
            return
        torch = self.torch
        if self.kernel is not None:
            self.kernel.fill(source, store, jobs, count, int(blocks), int(vector))
            return
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            m = max(1, min(int(limit), jobs.shape[0] - 1))
            live = torch.arange(m, device=self.device) < count
            keys = torch.where(live, jobs[:m, 0].long(), 0)
            slots = torch.where(live, jobs[:m, 1].long(), scratch)
            store.index_copy_(0, slots, source.index_select(0, keys))
            return
        g = int(count.reshape(-1)[0].item())
        for key, slot in jobs[:g].tolist():
            store[slot].copy_(source[key])

    def fill_jobs(self, pairs, store, source, scratch, program=None, blocks=0, vector=16):
        """Copies of (key, slot) pairs outside a plan (preload, swaps) through the same fill: the CUDA op with a
        program, a per-bundle copy without one is the caller's."""
        torch = self.torch
        jobs = torch.tensor([[k, s] for k, s in pairs] or [[0, scratch]], dtype=torch.int32, device=self.device)
        count = torch.tensor([len(pairs)], dtype=torch.int32, device=self.device)
        if program is not None:
            program.fill(self, store, source, jobs, count, scratch, len(pairs), blocks)
        elif self.kernel is not None:
            self.kernel.fill(source, store, jobs, count, int(blocks), int(vector))
        else:
            for key, slot in pairs:
                store[slot].copy_(source[key])

    # -- host-driven changes (between steps, never captured) ---------------------------------------------------------
    def _list(self, values):
        return self.torch.tensor(list(values), dtype=self.torch.int64, device=self.device)

    def set_unavailable(self, slots):
        """Slots the cache cannot use now: their keys are dropped and no plan takes them."""
        if not slots:
            return
        s = self._list(slots)
        old = self.slot_key[s].long()
        self.key_slot.index_fill_(0, self.torch.where(old >= 0, old, self.keys), EMPTY)
        self.key_slot[self.keys:].fill_(EMPTY)
        self.slot_key.index_fill_(0, s, UNAVAILABLE)
        self.slot_ref.index_fill_(0, s, 0)
        self.slot_pin.index_fill_(0, s, -1)

    def set_available(self, slots):
        """Unavailable slots come back empty."""
        if not slots:
            return
        s = self._list(slots)
        back = self.slot_key[s] == UNAVAILABLE
        s = self.torch.where(back, s, self.slots)
        self.slot_key.index_fill_(0, s, EMPTY)
        self.slot_ref.index_fill_(0, s, 0)
        self.slot_pin.index_fill_(0, s, -1)
        self.slot_key[self.slots:].fill_(EMPTY)

    def snapshot(self):
        """(key_slot [keys], slot_key [slots]) as host lists: synchronizes."""
        return self.key_slot[:self.keys].tolist(), self.slot_key[:self.slots].tolist()

    def assign(self, pairs, referenced=False):
        """Maps each (key, slot): the slot's old key and the key's old slot are dropped (preload, swap commits)."""
        if not pairs:
            return
        keys = self._list(k for k, _ in pairs)
        slots = self._list(s for _, s in pairs)
        old = self.slot_key[slots].long()
        self.key_slot.index_fill_(0, self.torch.where(old >= 0, old, self.keys), EMPTY)
        held = self.key_slot[keys].long()
        self.slot_key.index_fill_(0, self.torch.where(held >= 0, held, self.slots), EMPTY)
        self.slot_key.index_copy_(0, slots, keys.int())
        self.key_slot.index_copy_(0, keys, slots.int())
        self.slot_ref.index_fill_(0, slots, 1 if referenced else 0)
        self.key_slot[self.keys:].fill_(EMPTY)
        self.slot_key[self.slots:].fill_(EMPTY)

    def reserve(self, slots):
        """Background swaps begin: each slot's key is dropped and the slot reserved (no plan uses or evicts it)."""
        if not slots:
            return
        s = self._list(slots)
        old = self.slot_key[s].long()
        self.key_slot.index_fill_(0, self.torch.where(old >= 0, old, self.keys), EMPTY)
        self.key_slot[self.keys:].fill_(EMPTY)
        self.slot_key.index_fill_(0, s, RESERVED)
        self.slot_ref.index_fill_(0, s, 0)

    def set_hand(self, hand):
        self.state[0:1].fill_(int(hand) % self.slots)
