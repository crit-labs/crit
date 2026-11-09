"""The device expert cache a model plugin calls per routed layer (crit_plugins.device DEVICE_EXTENSIONS "experts:cache"):
slots of bundles on the device in one store tensor, the directory on the device (device_directory.py), misses filled
from a bundle store in device-readable host memory (expert_store.BundleStore), prompt chunks streamed through a staging
ring that never admits, and the slots of a KV pool's unused page tails lent to it.

    cache = ExpertCache(layers, experts, layout, slots, device, source, lend=kv_pool, policy="clock")
    slots = cache.plan(layer, ids)                 # decode / verify rows: admits by the policy, fills, capture-safe
    out = experts.forward_bundles(layer, x, ids, w, cache.store, slots, layout["format"])
    store, slots = cache.prompt(layer, ids)        # prompt chunks: never admits
    cache.end_prompt()                             # at the end of every step that called prompt()
    cache.telemetry()                              # {slots, resident, lent, reads, hits, fills, fill_bytes, ...}

The store holds every slot the experts forward reads: the cache's own slots, its prompt ring and, with a KV pool to
borrow from (lend=pool: attention-state's KVPool made with spare bytes for the own slots), the pool's pages no sequence
holds now. Slot table values: >= 0 a slot of store; -1 not routed; -2 the CPU tier computes it (an admission quota or
second touch sent it there, which needs cpu_tier). Without a CPU tier the cache keeps own slots for a layer's every
expert beside its ring and its swaps, so a plan always finds a victim (an eager plan raises if one did not). A row a
round holds (it emits nothing) has its ids set to -1 by the model before plan(), so it neither reads nor admits.

A step plans every routed layer with plan() or every one with prompt(), on one stream. Prompt chunks: a chunk of at least
`stage_routes` routes (by default as many as the layer has experts) routes nearly every expert of a layer, so the host
stages every expert its residency snapshot (read once per prompt step) says is not resident, by the copy engine on a
side stream into ring slots, the next layer under the current one; a ring segment is reused only after the layer that
read it ran (events). A shorter chunk copies its routed misses into ring slots on the device. The ring is borrowed from
the own slots for a prompt step and given back by end_prompt() at its end (ring="borrow"), or kept apart
(ring="fixed"). A decode plan outside a capture ends a prompt step the model left open.

Rounds (crit_plugins.rounds): start_round(view) at a round's start keeps ranked residency on the device (every
`period`-th round the decayed route counts of every key are ranked and up to max_swaps of the top keys swapped in,
burst_swaps after a topic change, copied in the round like misses); count_round(view) adds each row's reads and hits
into the round's counters. plan() outside a capture calls start_round itself when it sees a new round.
"""
from collections import deque

try:
    from . import device_directory as dd
    from . import host_memory
except ImportError:   # run from code/
    import device_directory as dd
    import host_memory


class _Segment:
    """Ring slots holding one layer's staged bundles: positions in the ring, expert -> slot, the copy stream's event
    after its copies (staged) and the compute stream's event after the layer that read it (consumed)."""
    __slots__ = ("layer", "positions", "ring_of", "staged", "consumed")

    def __init__(self, layer, positions):
        self.layer, self.positions, self.ring_of = layer, positions, {}
        self.staged = self.consumed = None


class ExpertCache:
    def __init__(self, layers, experts, layout, slots, device, source, lend=None, policy="clock", admit_quota=-1,
                 quota_tables=None, second_touch=False, cpu_tier=False, ring_slots=None, ring="borrow",
                 stage_routes=None, fill_blocks=0, fill_vector=16, decay=0.85, period=3, max_swaps=96,
                 burst_swaps=384, burst_fraction=0.25, kernel=True, raw_chunk=32, rows=64):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.layers, self.experts = int(layers), int(experts)
        self.bundle_bytes = int(layout["bundle_bytes"])
        if self.layers < 1 or self.experts < 1 or self.bundle_bytes < 16 or self.bundle_bytes % 16:
            raise ValueError("a cache has layers, experts and bundles of a multiple of 16 bytes")
        if policy not in ("clock", "ranked"):
            raise ValueError(f"policy is clock or ranked, not {policy!r}")
        if ring not in ("borrow", "fixed"):
            raise ValueError(f"ring is borrow or fixed, not {ring!r}")
        if (admit_quota >= 0 or quota_tables or second_touch) and not cpu_tier:
            raise ValueError("an admission quota or second touch sends misses to the CPU tier: give cpu_tier=True with "
                             "a CPU half beside the experts plugin")
        self.policy, self.cpu_tier, self.ring_mode = policy, bool(cpu_tier), ring
        # -- the source --------------------------------------------------------------------------------------------
        self.source = source
        if source.keys != self.layers * self.experts:
            raise ValueError(f"the store holds {source.keys} bundles, the cache {self.layers} x {self.experts}")
        self.program = dd.Program(source.layout, self.device) if "program" in source.layout else None
        slot_bytes = source.layout["slot_bytes"] if self.program is not None else source.bundle_bytes
        if slot_bytes != self.bundle_bytes:
            raise ValueError(f"the store's bundles fill slots of {slot_bytes} bytes, the cache's layout has "
                             f"{self.bundle_bytes}")
        if self.program is not None and cpu_tier:
            raise ValueError("a CPU tier reads bundles in place in the slot layout; a store in another layout (CSF "
                             "scales) needs a plain store for it")
        self._region = self._source_region(source)
        self._raw, self.raw_chunk = None, max(1, int(raw_chunk))   # prompt staging of a program store, by chunks
        B = self.bundle_bytes
        # -- slot memory: own slots (the ring among them), a scratch slot; lent page tails --------------------------
        self.lend = lend
        self._threshold = {}      # lent slot -> the floor up to which it is free (its first page)
        if lend is None:
            total = int(slots) + 1
            self.memory = torch.empty(total * B, dtype=torch.uint8, device=self.device)
            base, own = 0, list(range(total))
        else:
            info = lend.lendable()
            self.memory = info["memory"]
            first, end = info["spare"]
            base = first % B          # the store's first byte: own slots start exactly at the spare bytes
            total = (self.memory.numel() - base) // B
            own = [s for s in range(total) if first <= base + s * B and base + (s + 1) * B <= end]
            self._pages = info["regions"][0][2] if info["regions"] else 0
            for begin, page_bytes, pages in info["regions"]:
                stop = begin + pages * page_bytes
                for s in range(max(0, -(-(begin - base) // B)), total):
                    if base + (s + 1) * B > stop:
                        break
                    self._threshold[s] = (base + s * B - begin) // page_bytes
            if len(own) < int(slots) + 1:
                raise ValueError(f"the pool's spare bytes hold {len(own)} bundles; the cache needs its {int(slots)} own "
                                 f"slots and a scratch slot there: make the pool with spare = (slots + 1) * bundle_bytes")
            own = own[:int(slots) + 1]
        if total < 2:
            raise ValueError("a cache needs at least one slot of its own")
        self.store = self.memory[base:base + total * B].view(total, B)
        self.total = total
        self.scratch, self.own = own[-1], own[:-1]
        if not self.own:
            raise ValueError("a cache needs at least one slot of its own")
        swaps = max(burst_swaps, max_swaps) if policy == "ranked" else 0
        keep = 0 if cpu_tier else self.experts + swaps     # own slots a plan may need beside the ring
        if ring_slots is None:
            ring_slots = next((r for r in (2 * self.experts, self.experts) if len(self.own) - r >= keep), 0)
        ring_slots = int(ring_slots)
        if ring_slots and ring_slots < self.experts:
            raise ValueError(f"a prompt ring holds a layer's every expert: {self.experts} slots or more, not "
                             f"{ring_slots} (or 0: no prompts)")
        if not cpu_tier and len(self.own) - ring_slots < keep:
            raise ValueError(f"without a CPU tier every miss takes a slot: the cache's {len(self.own)} own slots less "
                             f"its ring of {ring_slots} and {swaps} swaps leave fewer than the {self.experts} experts "
                             "of a layer")
        self.ring_slots = self.own[len(self.own) - ring_slots:] if ring_slots else []
        if policy == "ranked":
            admit_quota = 0 if cpu_tier else -1   # ranked without a CPU tier: the clock admits misses as well
        tables = None
        if quota_tables:
            tables = quota_tables if isinstance(quota_tables, dict) else {1: quota_tables}
        self.directory = dd.DeviceDirectory(self.layers, self.experts, total, self.device, second_touch=second_touch,
                                            count_routes=policy == "ranked", admit_quota=admit_quota, quota=tables,
                                            kernel=kernel)
        self.floor = info["floor"] if lend is not None else None
        available = set(self.own) | {s for s, t in self._threshold.items() if t >= (self.floor or 0)}
        if ring == "fixed":
            available -= set(self.ring_slots)
        self.directory.set_unavailable([s for s in range(total) if s not in available])
        self._lent = {s for s in self._threshold if s in available}
        self.fill_blocks, self.fill_vector = int(fill_blocks), int(fill_vector)
        self.stage_routes = int(stage_routes) if stage_routes else self.experts
        self.copy_stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        self._row_counters = torch.zeros(max(1, int(rows)), 2, dtype=torch.int64, device=self.device)
        # prompt state
        self._prompt_open = ring == "fixed"
        self._snapshot = None
        self._prompt_layer = None
        self._ring = deque()        # live segments, oldest first
        self._ring_head = self._ring_used = 0
        self._staged = {}           # layer -> segment staged ahead
        self._reading = []          # the segments the last prompt() call's table reads
        cuda = self.device.type == "cuda"
        self._tables = torch.full((4, self.experts), dd.NOT_ROUTED, dtype=torch.int32, device=self.device)
        self._pinned = [torch.zeros(self.experts, dtype=torch.int32, pin_memory=cuda) for _ in range(4)]
        self._table_events = [None] * 4
        self._table_next = 0
        self.staged_copies = 0
        self._last_layer = None
        self._ranked = None
        if policy == "ranked":
            self._ranked = _Ranked(self, decay, period, max_swaps, max(burst_swaps, max_swaps), burst_fraction)
        if lend is not None:
            lend.on_reserve(self._lend_changed)

    # -- the source ----------------------------------------------------------------------------------------------
    def _source_region(self, source):
        if not source.mapped:   # a file tier: misses go through read() on the host
            return None
        host, device = source.region()
        keys = self.layers * self.experts
        if device is None and self.device.type == "cpu":
            device = host
        return host.view(keys, -1), (device.view(keys, -1) if device is not None else None)

    def _mapped(self):
        return self._region is not None and self._region[1] is not None

    def round_problems(self):
        """Why a round that plans through this cache cannot be captured, or [] (crit_plugins.rounds)."""
        return [] if self._mapped() else ["the expert source is a file tier: its misses are read on the host"]

    # -- decode --------------------------------------------------------------------------------------------------
    def plan(self, layer, ids, rows=None, decided=None):
        """The slot table of one routed layer ([experts] int32 on the device) for ids ([tokens][top], the router's):
        hits pinned, misses admitted by the policy and copied in before anything later on the stream reads the store.
        rows: the batch's row of each token ([tokens] int64, RoundBatch.rows_t), to count each row's reads and hits
        for count_round(). decided(slots), when given, runs between the plan and its copies (a dispatch posting the
        CPU tier's routes before the link is busy). Capture-safe with a mapped store, after one call at the shape
        outside capture."""
        torch = self.torch
        capturing = self.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        if not capturing:
            if self._prompt_open and self.ring_mode == "borrow":
                self.end_prompt()
            if self._last_layer is not None and layer <= self._last_layer:
                self.start_round(None)
            self._last_layer = layer
            self._snapshot = None
            if rows is not None and rows.numel() and int(rows.max()) >= self._row_counters.shape[0]:
                raise ValueError(f"a row past the cache's {self._row_counters.shape[0]} row counters (rows=)")
        top = ids.shape[-1] if ids.dim() > 1 else 1
        self.directory.plan(layer, ids, top, rows, self._row_counters if rows is not None else None)
        table = self.directory.tables[layer]
        if decided is not None:
            decided(table)
        self._fill(layer, min(ids.numel(), self.experts), capturing)
        if not capturing and not self.cpu_tier and int(self.directory.state[2]) & 2:
            raise RuntimeError("a plan found no victim for a miss and no CPU tier computes it: the cache lost a route")
        return table

    def _fill(self, layer, limit, capturing):
        if self._mapped():
            self.directory.fill(layer, self.store, self._region[1], self.scratch, limit, self.fill_blocks,
                                self.fill_vector, self.program)
            return
        if capturing:
            raise RuntimeError("the expert source has no device-readable region (a file tier): plan() reads its misses "
                               "on the host and cannot be captured; give the cache a mapped store for captured rounds")
        count = int(self.directory.copy_count[layer].item())
        if count:
            self._copy_bundles([tuple(job) for job in self.directory.copies[layer][:count].tolist()])

    def row_counters(self):
        """The per-row reads and hits plans with rows= add ([rows][2] int64 on the device: rows the constructor's)."""
        return self._row_counters

    def count_round(self, view):
        """A captured round's counters (crit_plugins.rounds; docs/PLUGINS.md, Captured rounds): each row's
        distinct expert reads and device hits of the round's plans (made with rows=) ADDED into
        view.counter("expert_reads") and view.counter("expert_hits"), and the cache's own row counters zeroed.
        Capture-safe."""
        reads, hits = view.counter("expert_reads"), view.counter("expert_hits")
        rows = reads.shape[0]
        if rows > self._row_counters.shape[0]:
            raise ValueError(f"the round has {rows} rows, the cache's row counters {self._row_counters.shape[0]}")
        reads.add_(self._row_counters[:rows, 0])
        hits.add_(self._row_counters[:rows, 1])
        self._row_counters.zero_()

    def start_round(self, view):
        """A decode or verify round starts (crit_plugins.rounds calls it inside the captured round; plan() outside a
        capture when it sees a new round): ranked residency's boundary work on the device, capture-safe."""
        if self._ranked is not None:
            self._ranked.round()
        if not (self.device.type == "cuda" and self.torch.cuda.is_current_stream_capturing()):
            self._last_layer = None   # the round's first plan does not start another

    # -- prompts -------------------------------------------------------------------------------------------------
    def prompt(self, layer, ids, stream=None):
        """(store, slot table) for a prompt chunk's routes of one layer: resident experts read in place, the others from
        ring slots; nothing admitted. Host-driven, not captured; end_prompt() ends the step."""
        torch = self.torch
        if not self.ring_slots:
            raise ValueError("the cache has no prompt ring (ring_slots 0): it serves decode plans only")
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prompt() is host-driven (residency snapshots, ring segments, copy-engine staging) and "
                               "cannot be captured: a prompt forward that calls it runs eagerly (the cache provides no "
                               "prompts:expert_residency; name expert_residency among the prompt forward's slots)")
        stream = stream or (torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None)
        if not self._prompt_open:
            self._open_prompt()
        if self._prompt_layer is not None and layer <= self._prompt_layer:
            self._snapshot = None      # a new prompt step: rounds may have run since (captured ones too)
        self._prompt_layer = layer
        for segment in self._reading:  # the layer before read its segments: everything on the stream so far
            segment.consumed = self._event(stream) or True
        self._reading = []
        if ids.numel() >= self.stage_routes and self._region is not None and self.copy_stream is not None:
            return self.store, self._prompt_staged(layer, stream)
        return self.store, self._prompt_routed(layer, ids, stream)

    def _event(self, stream):
        if stream is None:
            return None
        event = self.torch.cuda.Event()
        event.record(stream)
        return event

    def _open_prompt(self):
        self.directory.set_unavailable(self.ring_slots)
        self._snapshot = None
        self._prompt_open = True

    def end_prompt(self):
        """The prompt step is over (the model calls it at the end of every step that called prompt()): staged copies
        land before anything later on the stream, later staging waits for the layers that read the ring, a borrowed
        ring goes back to the cache."""
        if not self._prompt_open:
            return
        cuda = self.device.type == "cuda"
        stream = self.torch.cuda.current_stream(self.device) if cuda else None
        for segment in list(self._ring):
            if segment.staged is not None and stream is not None:
                stream.wait_event(segment.staged)
        if cuda:
            self.copy_stream.wait_event(self._event(stream))
        self._ring.clear()
        self._staged.clear()
        self._reading = []
        self._ring_head = self._ring_used = 0
        self._prompt_layer = None
        self._snapshot = None
        if self.ring_mode == "borrow":
            self.directory.set_available(self.ring_slots)
            self._prompt_open = False

    def _residency(self, stream):
        if self._snapshot is None:
            if stream is not None:
                stream.synchronize()
            self._snapshot = self.directory.snapshot()
        return self._snapshot

    def _allocate(self, layer, count, waiter):
        """A ring segment of `count` slots for `layer`, reusing the oldest consumed segments' slots (`waiter`, a stream,
        waits for the layers that read them). None when segments not yet read hold the room."""
        size = len(self.ring_slots)
        if count > size:
            raise ValueError(f"{count} prompt misses of one layer do not fit the ring's {size} slots")
        while size - self._ring_used < count:
            if not self._ring or self._ring[0].consumed is None:
                return None
            old = self._ring.popleft()
            if waiter is not None and old.consumed is not True:
                waiter.wait_event(old.consumed)
            self._ring_used -= len(old.positions)
        positions = [(self._ring_head + i) % size for i in range(count)]
        self._ring_head = (self._ring_head + count) % size
        self._ring_used += count
        segment = _Segment(layer, positions)
        self._ring.append(segment)
        return segment

    def _stage(self, layer, missing):
        """The bundles of `missing` experts of `layer` copied into a new ring segment on the copy stream (consecutive
        keys into consecutive ring slots in one copy), or None when the ring has no room."""
        torch = self.torch
        segment = self._allocate(layer, len(missing), self.copy_stream)
        if segment is None:
            return None
        ring = [self.ring_slots[p] for p in segment.positions]
        keys = [layer * self.experts + e for e in missing]
        segment.ring_of = dict(zip(missing, ring))
        host = self._region[0]
        with torch.cuda.stream(self.copy_stream):
            if self.program is not None:
                self._stage_by_program(keys, ring)
            else:
                i = 0
                while i < len(keys):
                    j = i + 1
                    while j < len(keys) and keys[j] == keys[j - 1] + 1 and ring[j] == ring[j - 1] + 1:
                        j += 1
                    self.store[ring[i]:ring[i] + j - i].copy_(host[keys[i]:keys[i] + j - i], non_blocking=True)
                    i = j
            segment.staged = self._event(self.copy_stream)
        self.staged_copies += len(keys)
        return segment

    def _stage_by_program(self, keys, ring):
        """A store in another layout (CSF scales), on the copy stream: chunks of its bundles copied by the copy engine
        into a device buffer (consecutive keys in one copy), each rebuilt into its ring slots by the program fill
        before the next chunk reuses the buffer (one stream: in order)."""
        torch = self.torch
        if self._raw is None:
            self._raw = torch.empty(self.raw_chunk, self.program.store_bytes, dtype=torch.uint8, device=self.device)
        host = self._region[0]
        for at in range(0, len(keys), self.raw_chunk):
            part, slots = keys[at:at + self.raw_chunk], ring[at:at + self.raw_chunk]
            i = 0
            while i < len(part):
                j = i + 1
                while j < len(part) and part[j] == part[j - 1] + 1:
                    j += 1
                self._raw[i:j].copy_(host[part[i]:part[i] + j - i], non_blocking=True)
                i = j
            self.directory.fill_jobs(list(enumerate(slots)), self.store, self._raw, self.scratch, self.program,
                                     self.fill_blocks)

    def _missing(self, layer, snapshot, experts=None):
        base, key_slot = layer * self.experts, snapshot[0]
        return [e for e in (range(self.experts) if experts is None else experts) if key_slot[base + e] < 0]

    def _prompt_staged(self, layer, stream):
        torch = self.torch
        snapshot = self._residency(stream)
        missing = self._missing(layer, snapshot)
        segments = []
        staged = self._staged.pop(layer, None)
        for other in list(self._staged.values()):   # staged ahead for another layer than the one asked: let it go
            stream.wait_event(other.staged)
            other.consumed = self._event(stream)
        self._staged.clear()
        if staged is not None:
            segments.append(staged)
            missing = [e for e in missing if e not in staged.ring_of]   # not resident any more since it was staged
        if missing or staged is None:
            segment = self._stage(layer, missing)
            if segment is None:
                raise RuntimeError("the prompt ring is held by segments not yet read: give it twice the experts of a "
                                   "layer")
            segments.append(segment)
        base, key_slot = layer * self.experts, snapshot[0]
        table = []
        for e in range(self.experts):
            slot = key_slot[base + e]
            if slot < 0:
                slot = next(s.ring_of[e] for s in segments if e in s.ring_of)
            table.append(slot)
        for segment in segments:
            stream.wait_event(segment.staged)
        i = self._table_next
        self._table_next = (i + 1) % len(self._pinned)
        if self._table_events[i] is not None:
            self._table_events[i].synchronize()
        self._pinned[i].copy_(torch.tensor(table, dtype=torch.int32))
        with torch.cuda.stream(stream):
            self._tables[i].copy_(self._pinned[i], non_blocking=True)
        self._table_events[i] = self._event(stream)
        self._reading = segments
        following = layer + 1    # the next layer of this step, staged under this one
        if following < self.layers and following not in self._staged:
            ahead = self._stage(following, self._missing(following, snapshot))
            if ahead is not None:
                self._staged[following] = ahead
        return self._tables[i]

    def _prompt_routed(self, layer, ids, stream):
        torch = self.torch
        for segment in self._staged.values():   # staged ahead for a long chunk; this step reads its routes only
            if stream is not None:
                stream.wait_event(segment.staged)
            segment.consumed = self._event(stream) or True
        self._staged.clear()
        snapshot = self._residency(stream)
        routed = sorted({int(e) for e in ids.reshape(-1).tolist() if 0 <= int(e) < self.experts})
        segment = self._allocate(layer, max(1, len(self._missing(layer, snapshot, routed))), stream)
        if segment is None:
            raise RuntimeError("the prompt ring is held by segments not yet read")
        ring = torch.tensor([self.ring_slots[p] for p in segment.positions], dtype=torch.int32, device=self.device)
        context = torch.cuda.stream(stream) if stream is not None else _Nothing()
        with context:
            self.directory.plan(layer, ids, ids.shape[-1] if ids.dim() > 1 else 1, ring=ring)
            self._fill(layer, ring.numel(), False)
        self._reading = [segment]
        return self.directory.tables[layer]

    # -- warm start ----------------------------------------------------------------------------------------------
    def preload(self, keys):
        """Maps keys (layer * experts + expert, a residency profile's order) to the available slots in slot order,
        copies their bundles and waits; the clock starts after the last. Between requests, never captured."""
        self.end_prompt()
        key_slot, slot_key = self.directory.snapshot()
        slots = [s for s in range(self.total) if slot_key[s] not in (dd.RESERVED, dd.UNAVAILABLE)]
        pairs, seen = [], set()
        for k in keys:
            if len(pairs) == len(slots):
                break
            k = int(k)
            if not 0 <= k < self.layers * self.experts or key_slot[k] >= 0 or k in seen:
                continue
            seen.add(k)
            pairs.append((k, slots[len(pairs)]))
        self.directory.assign(pairs)
        if pairs:
            self.directory.set_hand(pairs[-1][1] + 1)
        self._copy_bundles(pairs)
        if self.device.type == "cuda":
            self.torch.cuda.current_stream(self.device).synchronize()
        self._snapshot = None
        return pairs

    def warm(self, profile):
        """preload() from a residency profile: a list of keys, a dict with "keys", or a profile file
        (residency_profile.load_profile, checked against this cache's geometry)."""
        if isinstance(profile, (str, bytes)) or hasattr(profile, "__fspath__"):
            try:
                from . import residency_profile
            except ImportError:
                import residency_profile
            profile = residency_profile.load_profile(profile, self.layers, self.experts, len(self.own))
        keys = profile["keys"] if isinstance(profile, dict) else profile
        return self.preload(keys)

    def _copy_bundles(self, pairs, stream=None):
        """(key, slot) bundles copied from the source by the copy engine on `stream` (the current one by default)."""
        torch = self.torch
        if not pairs:
            return
        cuda = self.device.type == "cuda"
        stream = stream or (torch.cuda.current_stream(self.device) if cuda else None)
        if self.program is not None:
            self._copy_by_program(pairs, stream)
            return
        if self._region is None:
            staging = torch.empty(len(pairs), self.bundle_bytes, dtype=torch.uint8, pin_memory=cuda)
            self.source.read([k for k, _ in pairs], staging)
            with (torch.cuda.stream(stream) if cuda else _Nothing()):
                for i, (_, slot) in enumerate(pairs):
                    self.store[slot].copy_(staging[i], non_blocking=True)
            if cuda:
                stream.synchronize()   # the staging buffer is read before it goes
            return
        host = self._region[0]
        with (torch.cuda.stream(stream) if cuda else _Nothing()):
            for key, slot in pairs:
                self.store[slot].copy_(host[key], non_blocking=True)

    def _copy_by_program(self, pairs, stream):
        """(key, slot) bundles of a store in another layout, rebuilt by the program on the device: from the mapped
        store, or (a file tier) read into memory and copied up first."""
        torch = self.torch
        cuda = self.device.type == "cuda"
        with (torch.cuda.stream(stream) if cuda else _Nothing()):
            if self._mapped():
                self.directory.fill_jobs(pairs, self.store, self._region[1], self.scratch, self.program,
                                         self.fill_blocks)
                return
            raw = torch.empty(len(pairs), self.program.store_bytes, dtype=torch.uint8, pin_memory=cuda)
            self.source.read([k for k, _ in pairs], raw)
            raw = raw.to(self.device, non_blocking=True)
            self.directory.fill_jobs([(i, s) for i, (_, s) in enumerate(pairs)], self.store, raw, self.scratch,
                                     self.program, self.fill_blocks)
            if cuda:
                stream.synchronize()   # the read buffer goes once the fill has used it

    # -- lent memory ---------------------------------------------------------------------------------------------
    def _lend_changed(self, floor):
        """The KV pool's floor moved (pages [0, floor) may hold KV): lent slots below it are reclaimed before those
        pages are written, slots above it lent again. Returns (low, high): call again once the floor falls to low or a
        page at or past high is handed out. Host work between steps, never inside a capture."""
        if self.device.type == "cuda" and self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the KV pool changed its pages inside a capture: lent slots are reclaimed between steps")
        reclaim = sorted(s for s in self._lent if self._threshold[s] < floor)
        relend = sorted(s for s, t in self._threshold.items() if t >= floor and s not in self._lent)
        if reclaim:
            self.directory.set_unavailable(reclaim)
            self._lent.difference_update(reclaim)
            if self._snapshot is not None:
                key_slot, slot_key = self._snapshot
                for s in reclaim:
                    if slot_key[s] >= 0:
                        key_slot[slot_key[s]] = -1
                    slot_key[s] = dd.UNAVAILABLE
        if relend:
            self.directory.set_available(relend)
            self._lent.update(relend)
            if self._snapshot is not None:
                for s in relend:
                    self._snapshot[1][s] = dd.EMPTY
        self.floor = floor
        return self._lend_bounds()

    def _lend_bounds(self):
        held = [self._threshold[s] for s in self._lent]
        free = [t for s, t in self._threshold.items() if s not in self._lent]
        return (max(free) if free else -1), (min(held) if held else self._pages)

    # -- telemetry -----------------------------------------------------------------------------------------------
    def telemetry(self, sync=False):
        """{slots, resident, lent, reads, hits, fills, fill_bytes, cpu, prompt_reads, prompt_hits, prompt_copies,
        failure, swaps, kernel}: decode plans' distinct experts read and hit, copies and their bytes (decode fills,
        swaps and prompt staging), misses sent to the CPU tier. Without sync it never waits for the device: the counters
        of the last snapshot that has landed (a copy on a stream of its own to pinned memory, started by the call
        before, in relaxed capture mode so a capture on another thread goes on; the first call waits for one; inside a
        capture on this thread, the last snapshot). sync: the counters now."""
        values = self._snapshot_counters(sync)
        names = dd.COUNTERS + ("failure", "slots", "resident", "swaps")
        c = dict(zip(names, values)) if values else dict.fromkeys(names, 0)
        prompt_copies = c["prompt_copies"] + self.staged_copies
        fills = c["fills"] + c["swaps"] + prompt_copies
        return {"slots": c["slots"], "resident": c["resident"], "lent": len(self._lent), "reads": c["reads"],
                "hits": c["hits"], "fills": c["fills"], "fill_bytes": fills * self.bundle_bytes, "cpu": c["cpu"],
                "prompt_reads": c["prompt_reads"], "prompt_hits": c["prompt_hits"], "prompt_copies": prompt_copies,
                "failure": c["failure"], "swaps": c["swaps"], "kernel": self.directory.kernel is not None}

    def _gather(self, d):
        torch = self.torch
        key = d.slot_key[:self.total]
        swaps = self._ranked.swapped if self._ranked is not None else torch.zeros(1, dtype=torch.int64,
                                                                                  device=self.device)
        return torch.cat([d.counters, d.state[2:3].long(), (key >= dd.EMPTY).sum().view(1),
                          (key >= 0).sum().view(1), swaps.view(1)])

    def _snapshot_counters(self, sync):
        torch, d = self.torch, self.directory
        if self.device.type != "cuda" or sync:
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            return self._gather(d).tolist()
        if torch.cuda.is_current_stream_capturing():
            return getattr(self, "_tele", {}).get("values")
        if not hasattr(self, "_tele"):
            self._tele = {"stream": torch.cuda.Stream(self.device), "event": None, "values": None,
                          "host": torch.zeros(len(dd.COUNTERS) + 4, dtype=torch.int64, pin_memory=True)}
        t = self._tele
        with host_memory.relaxed_capture():
            if t["event"] is not None and (t["values"] is None or t["event"].query()):
                t["event"].synchronize()   # landed already, or the first snapshot: waits for that copy alone
                t["values"], t["event"] = t["host"].tolist(), None
            if t["event"] is None:
                with torch.cuda.stream(t["stream"]):
                    t["host"].copy_(self._gather(d), non_blocking=True)
                    t["event"] = torch.cuda.Event()
                    t["event"].record(t["stream"])
            if t["values"] is None:
                t["event"].synchronize()
                t["values"], t["event"] = t["host"].tolist(), None
        return t["values"]

    def close(self):
        if self.lend is not None:
            self.lend.on_reserve(None)
            self.lend = None
        self._staged.clear()
        self._ring.clear()


class _Ranked:
    """Ranked residency on the device (ranked_residency.RankedPolicy's rule, applied in the round): every round
    counts; at every `period`-th, each key's score decays and gains its routes since the last boundary, every key of
    every layer is ranked (ties to the lower key), and the top keys that are not resident are swapped into the slots
    that may take them (not the ring, not reclaimed or reserved ones): empty slots first in slot order, then the
    lowest-ranked resident keys (score, then key) while their score is lower, at most max_swaps (burst_swaps when more
    than burst_fraction of the slots' worth of top keys is missing). The swaps map at once and their bundles are copied
    in the round like misses. Torch ops: fixed shapes, capture-safe."""

    def __init__(self, cache, decay, period, max_swaps, burst_swaps, fraction):
        torch = cache.torch
        d, dev = cache.directory, cache.device
        if not 0 < decay <= 1 or period < 1 or max_swaps < 0 or not 0 <= fraction <= 1:
            raise ValueError("ranked residency: decay in (0, 1], a period, swaps and a burst fraction in [0, 1]")
        self.cache, self.decay, self.period = cache, float(decay), int(period)
        self.max_swaps, self.burst_swaps, self.fraction = int(max_swaps), int(burst_swaps), float(fraction)
        K, S = d.keys, d.slots
        self.scores = torch.zeros(K, dtype=torch.float64, device=dev)
        self.last = torch.zeros(K, dtype=torch.int64, device=dev)
        self.rounds = torch.zeros(1, dtype=torch.int64, device=dev)
        self.swapped = torch.zeros(1, dtype=torch.int64, device=dev)
        self.bursts = torch.zeros(1, dtype=torch.int64, device=dev)
        never = set(cache.ring_slots) | {cache.scratch}
        self.eligible = torch.tensor([s not in never for s in range(S)], dtype=torch.bool, device=dev)
        M = max(1, self.burst_swaps)
        self.jobs = torch.zeros(M + 1, 2, dtype=torch.int32, device=dev)
        self.count = torch.zeros(1, dtype=torch.int32, device=dev)
        self._ar_m = torch.arange(M, device=dev)
        self._ar_k = torch.arange(K, device=dev)
        self._ar_s = torch.arange(S, device=dev)

    def round(self):
        torch, cache = self.cache.torch, self.cache
        d = cache.directory
        K, S, M = d.keys, d.slots, self.jobs.shape[0] - 1
        boundary = (self.rounds > 0) & (self.rounds % self.period == 0)          # [1]
        self.rounds.add_(1)
        counts = d.route_counts[:K]
        scores = torch.where(boundary, self.scores * self.decay + (counts - self.last).double(), self.scores)
        self.last.copy_(torch.where(boundary, counts, self.last))
        self.scores.copy_(scores)
        key = d.slot_key[:S].long()
        open_slot = self.eligible & (key >= dd.EMPTY)          # neither reclaimed, reserved, nor the ring or scratch
        n_slots = open_slot.sum()
        order = torch.sort(scores, descending=True, stable=True).indices    # ties: the lower key first
        wanted_sorted = (torch.arange(K, device=scores.device) < n_slots) & (scores[order] > 0)
        wanted = torch.zeros(K, dtype=torch.bool, device=scores.device)
        wanted[order] = wanted_sorted
        resident = d.key_slot[:K] >= 0          # anywhere: a key in a slot swaps may not take is still resident
        admit_sorted = wanted_sorted & ~resident[order]
        n_admit = admit_sorted.sum()
        admit_rank = torch.cumsum(admit_sorted.long(), 0) - 1
        admits = torch.full((M + 1,), K, dtype=torch.int64, device=scores.device)
        admits.scatter_(0, torch.where(admit_sorted & (admit_rank < M), admit_rank, M), order)
        admits = admits[:M]
        empty = open_slot & (key == dd.EMPTY)
        n_empty = empty.sum()
        empties = torch.full((M + 1,), S, dtype=torch.int64, device=scores.device)
        e_rank = torch.cumsum(empty.long(), 0) - 1
        empties.scatter_(0, torch.where(empty & (e_rank < M), e_rank, M), self._ar_s)
        empties = empties[:M]
        losing = open_slot & (key >= 0) & ~wanted[key.clamp(0, K - 1)]
        n_losing = losing.sum()
        lose_score = torch.where(losing, scores[key.clamp(0, K - 1)], float("inf"))
        by_key = torch.sort(torch.where(losing, key, K), stable=True).indices          # slots by key
        by_score = by_key[torch.sort(lose_score[by_key], stable=True).indices]       # then by score: (score, key)
        losers = by_score[:M] if S >= M else torch.cat([by_score, by_score.new_full((M - S,), S)])
        burst = n_admit.double() > self.fraction * n_slots.double()
        limit = torch.where(burst, self.burst_swaps, self.max_swaps)
        i = self._ar_m
        j = (i - n_empty).clamp(min=0)
        admit_score = scores[admits.clamp(max=K - 1)]
        loser = losers[j.clamp(max=M - 1)]
        loser_score = lose_score[loser.clamp(max=S - 1)]
        ok = (i < limit) & (i < n_admit) & ((i < n_empty) | ((j < n_losing) & (loser_score < admit_score)))
        ok = torch.cumprod(ok.long(), 0).bool() & boundary
        victim = torch.where(i < n_empty, empties, loser)
        found = ok.sum()
        # apply: the victims' keys unmapped, the admitted keys mapped, their bundles copied in the round
        vs = torch.where(ok, victim, S)
        new = torch.where(ok, admits, K)
        old = d.slot_key[vs].long()
        d.key_slot.index_fill_(0, torch.where(ok & (old >= 0), old, K), dd.EMPTY)
        d.slot_key.index_copy_(0, vs, torch.where(ok, new, dd.EMPTY).int())
        d.key_slot.index_copy_(0, new, vs.int())
        d.slot_ref.index_fill_(0, vs, 0)
        d.key_slot[K:].fill_(dd.EMPTY)
        d.slot_key[S:].fill_(dd.EMPTY)
        self.jobs.index_copy_(0, torch.where(ok, i, M), torch.stack([new, vs], 1).int())
        self.count.copy_(found.view(1))
        self.swapped.add_(found)
        self.bursts.add_((burst & boundary & (found > 0)).long())
        if cache._mapped():
            d.fill_counted(self.jobs, self.count, cache.store, cache._region[1], cache.scratch, M,
                           cache.fill_blocks, cache.fill_vector, cache.program)
        elif not (cache.device.type == "cuda" and torch.cuda.is_current_stream_capturing()):
            g = int(found)
            if g:
                cache._copy_bundles([tuple(job) for job in self.jobs[:g].tolist()])
        else:
            raise RuntimeError("ranked swaps over a file tier read on the host: no captured rounds")


class _Nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
