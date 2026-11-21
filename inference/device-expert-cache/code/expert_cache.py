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
read it ran (events). A shorter chunk copies its routed misses into ring slots on the device, and one of a few
times stage_routes routes that leaves experts of its layer unrouted has only those it routes staged
(_routes_few, _partial_table). The ring is
borrowed from the own slots for a prompt step and given back by end_prompt() at its end (ring="borrow": the whole
ring for a staged chunk, as many slots as it has routes for a shorter one, so a short prompt costs the cache few
experts), or kept apart (ring="fixed"). A decode plan outside a capture ends a prompt step the model left open.

A staged layer serves every chunk that reads it. A model that runs a span of a prompt layer by layer calls
prompt(layer, ...) once a chunk for the same layer: the calls after the first read the layer's experts where the
first found them, and nothing is copied again, so the layer's experts cross the link once for the whole span instead
of once a chunk. Such a span takes several steps: end_prompt(hold=True) ends a step and keeps the ring and what is
staged in it, decode plans (captured rounds too) run meanwhile in the slots beside it, and the next step's first
prompt() reads the residency again and stages only what left the cache since. end_prompt() gives the ring back.

A short chunk's misses may go to the CPU tier instead (prompt_cpu(), with a CPU tier): its table marks the experts
that are not resident -2, as a plan's that sends a miss to the CPU tier, nothing is copied and no ring is borrowed.
The caller gives what it measured, a missed route on the CPU tier and a bundle over the link, and the chunk takes
the CPU where its missed routes cost less there than their experts over the link.

Rounds (crit_plugins.rounds): start_round(view) at a round's start keeps the policy's residency on the device;
count_round(view) adds each row's reads and hits into the round's counters. plan() outside a capture calls start_round
itself when it sees a new round.

    recent   (with a CPU tier) every expert a plan misses is computed by the CPU tier and copied in beside the round
             (a side stream), serving from the next round; it takes an empty slot, else the slot of the expert routed
             longest ago, from the max_swaps such slots listed at the round's start (never one routed this round).
             What is resident is what was routed last: the rule trace_replay.recent replays.
    ranked   every `period`-th round the decayed route counts of every key are ranked and up to max_swaps of the top
             keys swapped in, burst_swaps after a topic change, copied in the round like misses or beside it
             (side_swaps).
    clock    every miss a plan admits takes the clock's victim and is copied in before the layer reads it.
"""
from collections import deque

try:
    from . import device_directory as dd
    from . import host_memory
except ImportError:   # run from code/
    import device_directory as dd
    import host_memory


# A prompt chunk of stage_routes routes or more has every non-resident expert of its layer staged: a long chunk
# routes nearly all of them. One of fewer than ROUTED_CHECK times that many routes may still leave some unrouted
# (a few hundred tokens of ordinary text keep to the experts their topic uses), so its routes are looked at first:
# where it routes at most ROUTED_SHARE of the layer's non-resident experts, only the experts it routes are staged
# (_partial_table), and a later chunk of the span that reads the layer has the ones it routes beyond them staged
# beside. Staging a whole layer has the next layer's copies run under this layer's rows, which routes known a layer
# at a time cannot: that pays where the rows take as long as the copies they hide, so a chunk that routes nearly
# all the experts anyway is staged whole (a chunk this short computes in a fraction of its copies' time: the share
# is high), and so are the later layers of a span whose chunks together came to route more than that share.
ROUTED_CHECK = 8
ROUTED_SHARE = 0.85


def residency_profile_keys(path, layers, experts):
    """Every key a residency profile file names, hottest first (residency_profile.load_profile)."""
    try:
        from . import residency_profile
    except ImportError:
        import residency_profile
    return residency_profile.load_profile(path, layers, experts, layers * experts)


def ring_for(room, experts, keep=0, elastic=False):
    """The prompt ring a cache of `room` own slots gives itself: two layers' experts, or one's, or none, as far as
    `keep` slots stay beside it (and one more in an elastic cache)."""
    return next((r for r in (2 * experts, experts) if room - r >= keep and (not elastic or room > r)), 0)


class _Segment:
    """Ring slots holding one layer's staged bundles: positions in the ring, expert -> slot, the copy stream's event
    after its copies (staged) and the compute stream's event after the layer that read it (consumed)."""
    __slots__ = ("layer", "positions", "ring_of", "staged", "consumed")

    def __init__(self, layer, positions):
        self.layer, self.positions, self.ring_of = layer, positions, {}
        self.staged = self.consumed = None


class ExpertCache:
    # The round counters count_round adds to: the round driver has them without the model naming them
    # (crit_plugins.rounds.Rounds).
    round_counters = ("expert_reads", "expert_hits")

    def __init__(self, layers, experts, layout, slots, device, source, lend=None, policy="clock", admit_quota=-1,
                 quota_tables=None, second_touch=False, cpu_tier=False, ring_slots=None, ring="borrow",
                 stage_routes=None, fill_blocks=0, fill_vector=16, decay=0.85, period=3, max_swaps=96,
                 burst_swaps=384, burst_fraction=0.25, kernel=True, raw_chunk=32, rows=64, record_routes=0,
                 side_swaps=False, max_slots=0, owner="device-expert-cache", link_layers=(), link_slots=0):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        # A ring of the last `record_routes` decode routes, one row per token and layer: [layer, ids...] on the device,
        # written inside plan() (captured rounds too), read by routes() for crit plugins offload --trace.
        self._trace = self._trace_pos = None
        if record_routes:
            self._trace_rows = int(record_routes)
            self._trace_pos = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.layers, self.experts = int(layers), int(experts)
        self.bundle_bytes = int(layout["bundle_bytes"])
        # The bytes of a slot its rows use: the bundle, or an opaque row of fewer bytes padded to the slot (row_cache)
        self.row_bytes = int(layout.get("row_bytes", self.bundle_bytes))
        if not 0 < self.row_bytes <= self.bundle_bytes:
            raise ValueError(f"a row of {self.row_bytes} bytes does not fit a slot of {self.bundle_bytes}")
        if self.layers < 1 or self.experts < 1 or self.bundle_bytes < 16 or self.bundle_bytes % 16:
            raise ValueError("a cache has layers, experts and bundles of a multiple of 16 bytes")
        if policy not in ("clock", "ranked", "recent"):
            raise ValueError(f"policy is clock, ranked or recent, not {policy!r}")
        if policy == "recent" and not cpu_tier:
            raise ValueError("recent residency copies a miss in beside the round while the CPU tier computes it: give "
                             "cpu_tier=True with a CPU half beside the experts plugin")
        if ring not in ("borrow", "fixed"):
            raise ValueError(f"ring is borrow or fixed, not {ring!r}")
        if (admit_quota >= 0 or quota_tables or second_touch) and not cpu_tier:
            raise ValueError("an admission quota or second touch sends misses to the CPU tier: give cpu_tier=True with "
                             "a CPU half beside the experts plugin")
        self.policy, self.cpu_tier, self.ring_mode = policy, bool(cpu_tier), ring
        # Link layers: layers whose experts the CPU tier does not compute (a W4A16 draft layer beside W4A4 ones) in a
        # recent cache: their misses are admitted at once from a reserve of `link_slots` victims a round (what one
        # decode round can route in them) and copied before they are read, never sent to the CPU tier.
        self.link_layers = frozenset(int(layer) for layer in link_layers)
        self.link_slots = min(int(link_slots), int(experts)) if self.link_layers else 0
        if self.link_layers and (policy != "recent" or not self.link_slots):
            raise ValueError("link layers are a recent cache's (beside the CPU tier), with link_slots a round")
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
        # max_slots: the slots live in an elastic range of the device's memory (crit_plugins.memory, listed there
        # as owner/slots) reserved for that many, backed for `slots` now; grow()/shrink() back more or fewer while
        # every slot's address stays. The scratch slot and the ring sit at the range's start, so the backed prefix
        # always holds them.
        self.elastic = None
        if lend is None and int(max_slots) > int(slots):
            from crit_plugins.memory import of
            total = int(max_slots) + 1
            self.elastic = of(self.device).elastic(owner, "slots", total * B)
            self.memory = self.elastic.tensor
            base, own = 0, list(range(1, total)) + [0]   # the scratch slot is the last of own: slot 0
        elif lend is None:
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
        # own slots a plan may need beside the ring: a layer's experts without a CPU tier, the link layers' reserve
        keep = (self.link_slots if cpu_tier else self.experts + swaps)
        self.plan_slots = keep   # which a shrink leaves too: without a CPU tier a miss with no slot is a lost route
        room = int(slots) if self.elastic is not None else len(self.own)   # an elastic cache: the slots backed first
        if ring_slots is None:
            ring_slots = ring_for(room, self.experts, keep, self.elastic is not None)
            if not ring_slots:   # a cache asked to serve prompts that could stage none fails at its first prompt
                need = self.experts + keep + (1 if self.elastic is not None else 0)
                raise ValueError(
                    f"a cache of {room} own slots holds no prompt ring: a ring takes a layer's {self.experts} experts "
                    + ((f"beside the {keep} slots its link layers' misses may take a round, "
                        if self.link_layers else
                        f"beside the {keep} slots a plan may need without a CPU tier (every miss takes a slot), ")
                       if keep else "")
                    + f"so prompts take {need} slots or more ({need * B / 2**30:.2f} GiB of bundles); give the "
                    "cache that much device memory"
                    + (", or the CPU tier the misses" if keep and not self.link_layers else "")
                    + ", or make a decode-only cache with ring_slots 0")
        ring_slots = int(ring_slots)
        if ring_slots and ring_slots < self.experts:
            raise ValueError(f"a prompt ring holds a layer's every expert: {self.experts} slots or more, not "
                             f"{ring_slots} (or 0: no prompts)")
        if self.link_layers and room - ring_slots < keep:
            raise ValueError(f"the link layers' misses take slots: the cache's {room} own slots less its ring of "
                             f"{ring_slots} leave fewer than the {keep} a round may route in them")
        if not cpu_tier and room - ring_slots < keep:
            raise ValueError(f"without a CPU tier every miss takes a slot: the cache's {room} own slots less "
                             f"its ring of {ring_slots} and {swaps} swaps leave fewer than the {self.experts} experts "
                             "of a layer")
        if self.elastic is not None:   # the ring at the front, in the prefix that stays backed
            self.ring_slots = self.own[:ring_slots] if ring_slots else []
        else:
            self.ring_slots = self.own[len(self.own) - ring_slots:] if ring_slots else []
        if policy == "recent":
            admit_quota, quota_tables = 0, None
        if policy == "ranked":
            admit_quota = 0 if cpu_tier else -1   # ranked without a CPU tier: the clock admits misses as well
        tables = None
        if quota_tables:
            tables = quota_tables if isinstance(quota_tables, dict) else {1: quota_tables}
        self.directory = dd.DeviceDirectory(self.layers, self.experts, total, self.device, second_touch=second_touch,
                                            count_routes=policy != "clock", admit_quota=admit_quota, quota=tables,
                                            kernel=kernel)
        self.floor = info["floor"] if lend is not None else None
        available = set(self.own) | {s for s, t in self._threshold.items() if t >= (self.floor or 0)}
        if self.elastic is not None:   # backed: slot 0 (scratch), then `slots` own slots, the ring their first
            if int(slots) <= len(self.ring_slots):
                raise ValueError(f"an elastic cache of {int(slots)} slots holds no slot beside its ring of "
                                 f"{len(self.ring_slots)}")
            self.backed = 1 + int(slots)
            self.elastic.grow(self.backed * B)
            available &= set(range(self.backed))
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
        self._ring_now = list(self.ring_slots) if ring == "fixed" else []   # the ring slots this prompt step holds
        self._snapshot = None
        self._prompt_layer = None
        self._table = None          # the staged slot table of _prompt_layer, which every chunk of that layer reads
        self.holding = False        # the ring is kept past a step's end (end_prompt(hold=True)): a span of many steps
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
        self.prompt_cpu_routes = 0   # prompt routes whose missed expert the CPU tier was given (prompt_cpu)
        self._few = {}               # this prompt step's chunk sizes whose routed misses are copied alone (_routes_few)
        self._needed = None          # the layer staged for its chunks' routes alone: the experts they routed so far
        self._whole = False          # a span's chunks together came to route most of a layer: later layers whole
        self.routed_chunks = 0       # the prompt steps' chunk sizes decided so
        self._last_layer = None
        self._ranked = self._recent = None
        self.pinned = {}            # slot -> key: the pinned core (pin()), which no plan, shrink or preload takes
        if policy == "ranked":
            self._ranked = _Ranked(self, decay, period, max_swaps, max(burst_swaps, max_swaps), burst_fraction,
                                   side=bool(side_swaps))
        if policy == "recent":
            self._recent = _Recent(self, max_swaps, self.link_slots)
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

    def backed_slots(self):
        """Own slots backed now (an elastic cache), else every own slot; the prompt ring's are among them."""
        return self.backed - 1 if self.elastic is not None else len(self.own)

    def grow(self, slots):
        """An elastic cache backs `slots` own slots (more than now): the new ones come empty and available. Between
        steps, never captured. Returns the slots backed."""
        if self.elastic is None:
            return len(self.own)
        want = min(self.total, 1 + int(slots))
        if want <= self.backed:
            return self.backed_slots()
        got = self.elastic.grow(want * self.bundle_bytes) // self.bundle_bytes
        new = [s for s in range(self.backed, min(want, got))]
        self.backed = self.backed + len(new)
        if new:
            self.directory.set_available(new)
            self._snapshot = None
        return self.backed_slots()

    def shrink(self, slots):
        """An elastic cache gives back own slots past `slots`. Under the ranked policy the experts in those slots
        that outrank the coldest experts kept first move into those experts' slots (device to device), so a shrink
        loses the coldest experts, not whatever the last slots held. It keeps its ring and, without a CPU tier, the
        own slots a plan may need beside it (a layer's every expert and its swaps: a miss finds a slot or the route
        is lost), so a cache asked for more than it can give gives what it can. Between steps."""
        if self.elastic is None:
            return len(self.own)
        want = max(1 + len(self.ring_slots) + max(1, self.plan_slots), 1 + int(slots),
                   max(self.pinned, default=0) + 2)
        if want >= self.backed:
            return self.backed_slots()
        self.torch.cuda.synchronize(self.device)
        if self._recent is not None:
            self._recent.settle()
        if self._ranked is not None and self._ranked.pending is not None:
            # swaps copied on the side stream into slots that go now are dropped, never mapped
            self._ranked.join()
            p = self._ranked.pending
            gone = p[:, 1] >= want
            p[:, 0].masked_fill_(gone, self.directory.keys)
            p[:, 1].masked_fill_(gone, self.directory.slots)
        self._compact(want)
        self.directory.set_unavailable(list(range(want, self.backed)))
        self._snapshot = None      # a prompt step's residency names slots that are gone or hold another expert now
        granule = self.elastic.gran
        keep = -(-want * self.bundle_bytes // granule) * granule   # whole granules: a slot past `want` may stay backed
        self.elastic.shrink(keep)
        self.backed = want
        return self.backed_slots()

    def _compact(self, want):
        """The experts in the slots from `want` on that outrank the coldest experts (or empty slots) below it take
        those slots: the hottest leaving expert against the coldest kept one, while it ranks higher (ranked: its
        score; recent: the round it was last routed in)."""
        if self._ranked is None and self._recent is None:
            return 0
        torch, d = self.torch, self.directory
        keys = d.slot_key[:self.backed].long()
        last = d.slot_last[:self.backed]
        rank = last.double() if self._recent is not None else self._ranked.scores[keys.clamp(min=0)]
        rank = torch.where(keys >= 0, rank, torch.full_like(rank, float("-inf")))
        if self.pinned:   # the pinned core stays where it is
            rank[torch.tensor(sorted(self.pinned), device=self.device)] = float("inf")
        at = torch.arange(self.backed, device=self.device)
        leaving = torch.nonzero((at >= want) & (keys >= 0)).flatten()
        kept = torch.nonzero((at >= 1) & (at < want) & (keys >= dd.EMPTY)).flatten()   # slot 0 is the scratch slot
        if not leaving.numel() or not kept.numel():
            return 0
        leaving = leaving[torch.argsort(rank[leaving], descending=True, stable=True)]
        kept = kept[torch.argsort(rank[kept], stable=True)]
        n = min(leaving.numel(), kept.numel())
        moves = int(torch.cumprod((rank[leaving[:n]] > rank[kept[:n]]).long(), 0).sum())
        if not moves:
            return 0
        src, dst = leaving[:moves], kept[:moves]
        for i in range(0, moves, 32):
            self.store.index_copy_(0, dst[i:i + 32], self.store.index_select(0, src[i:i + 32]))
        d.assign(list(zip(keys[src].tolist(), dst.tolist())), last=last[src].tolist())
        return moves

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
            if self._prompt_open and self.ring_mode == "borrow" and not self.holding:
                self.end_prompt()
            if (self._last_layer is not None and layer <= self._last_layer) or \
                    (self._recent is not None and not self._recent.listed):
                self.start_round(None)
            self._last_layer = layer
            self._snapshot = None
            if rows is not None and rows.numel() and int(rows.max()) >= self._row_counters.shape[0]:
                raise ValueError(f"a row past the cache's {self._row_counters.shape[0]} row counters (rows=)")
        top = ids.shape[-1] if ids.dim() > 1 else 1
        if self._trace_pos is not None:
            self._record(layer, ids.reshape(-1, top), capturing)
        recent = self._recent.call(capturing) if self._recent is not None else None
        # A store that keeps what the cache evicts (a file tier's host tier) is told which slots held what before
        # the plan gave them away.
        prior = (self.directory.slot_key.clone()
                 if not capturing and self.program is None and getattr(self.source, "retains", False) else None)
        link = recent is not None and layer in self.link_layers
        self.directory.plan(layer, ids, top, rows, self._row_counters if rows is not None else None, recent=recent,
                            link=link)
        table = self.directory.tables[layer]
        if decided is not None:
            decided(table)
        if link:   # its misses copied in now, before the layer reads them: the CPU tier does not compute them
            self._fill(layer, min(ids.numel(), self.experts), capturing, prior)
            if not capturing and int(self.directory.state[2]) & 2:
                raise RuntimeError(f"a plan of link layer {layer} found no victim for a miss: the cache lost a route "
                                   f"(its reserve of {self.link_slots} slots a round is smaller than what it routed)")
            return table
        if recent is not None:
            self._recent.fill(recent[3], capturing)   # beside the round: the CPU tier computes these misses now
            return table
        self._fill(layer, min(ids.numel(), self.experts), capturing, prior)
        if not capturing and not self.cpu_tier and int(self.directory.state[2]) & 2:
            raise RuntimeError("a plan found no victim for a miss and no CPU tier computes it: the cache lost a route")
        return table

    def _record(self, layer, ids, capturing):
        torch = self.torch
        top = ids.shape[1]
        if self._trace is None:
            if capturing:
                raise RuntimeError("plan once outside a capture before capturing a recording cache")
            self._trace = torch.full((self._trace_rows, 1 + top), -1, dtype=torch.int32, device=self.device)
            self._trace_steps = {}
        if self._trace.shape[1] != 1 + top:
            raise ValueError(f"a recording cache records {self._trace.shape[1] - 1} routes a token, not {top}")
        n = ids.shape[0]
        step = self._trace_steps.get(n)
        if step is None:
            if capturing:
                raise RuntimeError(f"plan {n} tokens once outside a capture before capturing a recording cache")
            step = self._trace_steps[n] = torch.arange(n, dtype=torch.int64, device=self.device)
        at = (self._trace_pos + step) % self._trace_rows
        row = torch.cat([torch.full((n, 1), int(layer), dtype=torch.int32, device=self.device), ids.to(torch.int32)], 1)
        self._trace.index_copy_(0, at, row)
        self._trace_pos.add_(n)

    def routes(self):
        """The recorded decode routes, oldest first: [(layer, [experts])], one per token and routed layer (ids -1, rows a
        round holds, left out); synchronizes. Empty without record_routes."""
        if self._trace is None:
            return []
        total, rows = int(self._trace_pos.item()), self._trace_rows
        data = self._trace.cpu().tolist()
        order = range(total) if total <= rows else range(total - rows, total)
        out = []
        for i in order:
            layer, *ids = data[i % rows]
            kept = [e for e in ids if e >= 0]
            if kept:
                out.append((layer, kept))
        return out

    def _fill(self, layer, limit, capturing, prior=None, ring=False):
        """The copies of the layer's plan. prior: the slots' keys before the plan (a store that retains evicted
        bundles gets them back first); ring: the copies go to a prompt's ring slots, which nothing keeps."""
        if self._mapped():
            self.directory.fill(layer, self.store, self._region[1], self.scratch, limit, self.fill_blocks,
                                self.fill_vector, self.program)
            return
        if capturing:
            raise RuntimeError("the expert source has no device-readable region (a file tier): plan() reads its misses "
                               "on the host and cannot be captured; give the cache a mapped store for captured rounds")
        count = int(self.directory.copy_count[layer].item())
        if count:
            pairs = [tuple(job) for job in self.directory.copies[layer][:count].tolist()]
            if prior is not None:
                self._retain(pairs, prior)
            self._copy_bundles(pairs, keep=ring)

    def _retain(self, pairs, prior):
        """The bundles the plan's copies replace go back to the store on the host before their slots are written."""
        torch = self.torch
        slots = torch.tensor([slot for _, slot in pairs], dtype=torch.int64, device=self.device)
        held = prior[slots].tolist()
        evicted = [(key, slot) for key, (_, slot) in zip(held, pairs) if key >= 0]
        if evicted:
            rows = self.store[torch.tensor([slot for _, slot in evicted], dtype=torch.int64, device=self.device)]
            self.source.retain([key for key, _ in evicted], rows.cpu())

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
        for policy in (self._ranked, self._recent):
            if policy is not None:
                policy.join()   # a side stream's copies rejoin inside the round

    def start_round(self, view):
        """A decode or verify round starts (crit_plugins.rounds calls it inside the captured round; plan() outside a
        capture when it sees a new round): ranked residency's boundary work on the device, capture-safe."""
        for policy in (self._ranked, self._recent):
            if policy is not None:
                policy.round()
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
        if self._prompt_open and self._table is not None and layer == self._prompt_layer:
            # The layer is staged: another chunk of a span run layer by layer reads its experts where the first
            # chunk found them. After a step's end (a held ring) the residency is read again: decode plans ran
            # meanwhile, and what left the cache since is staged beside what the ring holds. A layer staged for its
            # chunks' routes alone has what this chunk routes beyond them staged beside.
            if self._needed is not None:
                self._table = self._partial_table(layer, ids, stream)
            elif self._snapshot is None:
                self._table = self._restaged(layer, stream)
            return self.store, self._table
        staged = ids.numel() >= self.stage_routes and self._region is not None and self.copy_stream is not None
        want = len(self.ring_slots) if staged else min(len(self.ring_slots), self.experts, max(1, ids.numel()))
        if self._prompt_open and want > len(self._ring_now):
            self.end_prompt()          # a longer chunk than the step began with: the ring is borrowed again, larger
        if not self._prompt_open:
            self._open_prompt(want)
        if self._prompt_layer is not None and layer <= self._prompt_layer:
            self._snapshot = None      # a new prompt step: rounds may have run since (captured ones too)
        self._prompt_layer, self._table = layer, None
        for segment in self._reading:  # the layer before read its segments: everything on the stream so far
            segment.consumed = self._event(stream) or True
        self._reading = []
        self._needed = None
        if (staged and ids.numel() < ROUTED_CHECK * self.stage_routes and not self._whole
                and self._routes_few(layer, ids, stream)):
            self._needed = [False] * self.experts
            self._table = self._partial_table(layer, ids, stream)
            return self.store, self._table
        if staged:
            self._table = self._prompt_staged(layer, stream)
            return self.store, self._table
        return self.store, self._prompt_routed(layer, ids, stream)

    def _routed(self, ids):
        """[experts] booleans on the host: the experts a chunk's routes name (an id below 0 names none)."""
        routed = self.torch.zeros(self.experts + 1, dtype=self.torch.bool, device=ids.device)
        routed[ids.reshape(-1).long().clamp(-1, self.experts - 1)] = True   # an id of -1: the spare last entry
        return routed[:self.experts].cpu().tolist()

    def _routes_few(self, layer, ids, stream):
        """Whether a prompt chunk of a few times stage_routes routes is one for which only the experts it routes
        are staged: it routes at most ROUTED_SHARE of the layer's experts that are not resident. Decided at a
        prompt step's first such layer for the chunks of that size in the step (a text routes its layers alike, and
        a staged layer's copies for the next would be thrown away where the next decided otherwise)."""
        size = int(ids.numel())
        found = self._few.get(size)
        if found is None:
            missing = self._missing(layer, self._residency(stream))
            routed = self._routed(ids)
            need = sum(1 for e in missing if routed[e])
            found = self._few[size] = bool(missing) and need <= ROUTED_SHARE * len(missing)
            self.routed_chunks += found
        return found

    def _partial_table(self, layer, ids, stream):
        """The slot table of a layer staged for its chunks' routes alone, with this chunk's: the experts the layer's
        chunks routed so far (_needed) each at its cache slot where resident, else at a ring slot it was staged
        into; what this chunk routes beyond them, and what left the cache since (a held ring: decode plans ran
        between a span's steps), is staged beside. An expert no chunk routed has no slot: no row reads it."""
        routed, needed = self._routed(ids), self._needed
        fresh = not any(needed)
        new = [e for e in range(self.experts) if routed[e] and not needed[e]]
        if not new and not fresh and self._snapshot is not None:
            return self._table   # nothing beyond what the layer's chunks before it had staged
        for e in new:
            needed[e] = True
        for segment in self._staged.values():   # staged ahead for a whole layer: this one reads its routes only
            stream.wait_event(segment.staged)
            segment.consumed = self._event(stream) or True
        self._staged.clear()
        if self._recent is not None and self._snapshot is None:
            self._recent.settle()
        snapshot = self._residency(stream)
        base, key_slot = layer * self.experts, snapshot[0]
        segments = [] if fresh else self._reading
        missing = [e for e in range(self.experts) if needed[e] and key_slot[base + e] < 0
                   and not any(e in s.ring_of for s in segments)]
        if missing:
            segment = self._stage(layer, missing)
            if segment is None:
                raise RuntimeError("the prompt ring is held by segments not yet read: give it twice the experts of a "
                                   "layer")
            segments.append(segment)
        table, away, staged = [], 0, 0
        for e in range(self.experts):
            slot = key_slot[base + e]
            if slot < 0:
                away += 1
                slot = next((s.ring_of[e] for s in segments if e in s.ring_of), dd.NOT_ROUTED)
                staged += slot != dd.NOT_ROUTED
            table.append(slot)
        if not fresh and staged > ROUTED_SHARE * away:   # a span's chunks together route them: later layers whole
            self._whole = True
        for segment in segments:
            stream.wait_event(segment.staged)
        self._reading = segments
        return self._upload(table, stream)

    def prompt_cpu(self, layer, ids, cpu_us, link_us, wins=1.0, stream=None):
        """The slot table of a short prompt chunk whose misses the CPU tier computes: each resident expert's slot and
        -2 for the others ([experts] int32, as a plan's that sends a miss to the CPU tier), nothing copied, nothing
        admitted, no ring borrowed. None where the link serves the chunk sooner, and the caller then asks prompt():
        cpu_us is a missed route on the CPU tier and link_us a bundle over the link, as the caller measured them, and
        the CPU takes the chunk where its missed routes cost under `wins` of their experts' copies (a route of an
        expert the tier has just read costs it less than the first, so the CPU's side is an upper bound). None also
        for a chunk of stage_routes routes or more (it routes nearly every expert of the layer, each for many rows:
        the link's), for a layer that is staged, for a link layer (the CPU tier does not compute its experts), and
        without a CPU tier. Host-driven, not captured."""
        torch = self.torch
        if (not self.cpu_tier or layer in self.link_layers or ids.numel() >= self.stage_routes
                or (self.device.type == "cuda" and torch.cuda.is_current_stream_capturing())
                or (self._prompt_open and self._table is not None and layer == self._prompt_layer)):
            return None
        stream = stream or (torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None)
        if self._prompt_layer is None or layer <= self._prompt_layer:
            self._snapshot = None      # a new prompt step: rounds may have run since (captured ones too)
        if self._recent is not None and self._snapshot is None:
            self._recent.settle()
        snapshot = self._residency(stream)
        routes = [int(e) for e in ids.reshape(-1).tolist() if 0 <= int(e) < self.experts]
        missing = set(self._missing(layer, snapshot, sorted(set(routes))))
        missed = sum(1 for e in routes if e in missing)
        if missed * float(cpu_us) >= float(wins) * len(missing) * float(link_us) and missing:
            return None
        self._prompt_layer, self._table = layer, None
        for segment in self._reading:  # the layer before read its segments: everything on the stream so far
            segment.consumed = self._event(stream) or True
        self._reading = []
        base, key_slot = layer * self.experts, snapshot[0]
        self.prompt_cpu_routes += missed
        return self._upload([key_slot[base + e] if key_slot[base + e] >= 0 else dd.CPU for e in range(self.experts)],
                            stream)

    def _upload(self, table, stream):
        """A host-made slot table on the device, through a pinned buffer no copy still reads."""
        torch = self.torch
        if self.device.type != "cuda":
            return torch.tensor(table, dtype=torch.int32, device=self.device)
        i = self._table_next
        self._table_next = (i + 1) % len(self._pinned)
        if self._table_events[i] is not None:
            self._table_events[i].synchronize()
        self._pinned[i].copy_(torch.tensor(table, dtype=torch.int32))
        with torch.cuda.stream(stream):
            self._tables[i].copy_(self._pinned[i], non_blocking=True)
        self._table_events[i] = self._event(stream)
        return self._tables[i]

    def _event(self, stream):
        if stream is None:
            return None
        event = self.torch.cuda.Event()
        event.record(stream)
        return event

    def _open_prompt(self, count):
        """A prompt step borrows the ring's first `count` slots: their experts are dropped until end_prompt()."""
        if self._recent is not None:
            self._recent.settle()
        self._ring_now = self.ring_slots[:count]
        self.directory.set_unavailable(self._ring_now)
        self._snapshot = None
        self._prompt_open = True
        self._few, self._needed, self._whole = {}, None, False

    def end_prompt(self, hold=False):
        """The prompt step is over (the model calls it at the end of every step that called prompt()): staged copies
        land before anything later on the stream, later staging waits for the layers that read the ring, a borrowed
        ring goes back to the cache. hold: the prompt goes on in a later step (a span run layer by layer over several
        steps), so the ring stays borrowed with what is staged in it; the residency is read again by the next step's
        first prompt(), since decode plans may run in the slots beside the ring meanwhile."""
        self._snapshot = None      # the next step reads the residency again, whatever this one read
        if not self._prompt_open:
            return
        cuda = self.device.type == "cuda"
        stream = self.torch.cuda.current_stream(self.device) if cuda else None
        for segment in list(self._ring):
            if segment.staged is not None and stream is not None:
                stream.wait_event(segment.staged)
        if cuda:
            self.copy_stream.wait_event(self._event(stream))
        self.holding = bool(hold) and self.ring_mode == "borrow"
        if self.holding:
            return
        self._table = self._needed = None
        self._ring.clear()
        self._staged.clear()
        self._reading = []
        self._ring_head = self._ring_used = 0
        self._prompt_layer = None
        self._snapshot = None
        if self.ring_mode == "borrow":
            self.directory.set_available(self._ring_now)
            self._ring_now = []
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
        size = len(self._ring_now)
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
        ring = [self._ring_now[p] for p in segment.positions]
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
        table = self._layer_table(layer, snapshot, segments, stream)
        self._reading = segments
        following = layer + 1    # the next layer of this step, staged under this one
        if following < self.layers and following not in self._staged:
            ahead = self._stage(following, self._missing(following, snapshot))
            if ahead is not None:
                self._staged[following] = ahead
        return table

    def _layer_table(self, layer, snapshot, segments, stream):
        """The slot table of a staged layer on the device: each expert's cache slot where the snapshot has it
        resident, else its ring slot in `segments`, read once their copies landed."""
        base, key_slot = layer * self.experts, snapshot[0]
        table = []
        for e in range(self.experts):
            slot = key_slot[base + e]
            if slot < 0:
                slot = next(s.ring_of[e] for s in segments if e in s.ring_of)
            table.append(slot)
        for segment in segments:
            stream.wait_event(segment.staged)
        return self._upload(table, stream)

    def _restaged(self, layer, stream):
        """The staged layer's table after the residency moved (a held ring: decode plans ran since the layer was
        staged): the experts that left the cache meanwhile are staged beside the layer's segments, in the ring
        slots of the next layer, which was staged ahead and is staged again when its turn comes."""
        if self._recent is not None:
            self._recent.settle()
        snapshot = self._residency(stream)
        segments = self._reading
        missing = [e for e in self._missing(layer, snapshot) if not any(e in s.ring_of for s in segments)]
        if missing:
            # The ring's newest segment is the next layer's, staged ahead, which nothing read: its slots are taken
            # back first, so the layer's own segments stay the ring's newest and what left the cache always finds
            # room beside them (the copy stream runs in order: what is staged there now lands after it).
            for other in self._staged.values():
                if self._ring and self._ring[-1] is other:
                    self._ring.pop()
                    self._ring_head = (self._ring_head - len(other.positions)) % len(self._ring_now)
                    self._ring_used -= len(other.positions)
            self._staged.clear()
            segment = self._stage(layer, missing)
            if segment is None:
                raise RuntimeError("the prompt ring has no room for the experts that left the cache while a layer "
                                   "was held staged: give it twice the experts of a layer")
            segments.append(segment)
        return self._layer_table(layer, snapshot, segments, stream)

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
        ring = torch.tensor([self._ring_now[p] for p in segment.positions], dtype=torch.int32, device=self.device)
        context = torch.cuda.stream(stream) if stream is not None else _Nothing()
        with context:
            self.directory.plan(layer, ids, ids.shape[-1] if ids.dim() > 1 else 1, ring=ring)
            self._fill(layer, ring.numel(), False, ring=True)
        self._reading = [segment]
        return self.directory.tables[layer]

    # -- warm start ----------------------------------------------------------------------------------------------
    def preload(self, keys):
        """Maps keys (layer * experts + expert, a residency profile's order) to the available slots in slot order,
        copies their bundles and waits; the clock starts after the last. Between requests, never captured."""
        self.end_prompt()
        if self._recent is not None:
            self._recent.settle()
        key_slot, slot_key = self.directory.snapshot()
        slots = [s for s in range(self.total) if slot_key[s] not in (dd.RESERVED, dd.UNAVAILABLE)
                 and s not in self.pinned]
        pairs, seen = [], set()
        for k in keys:
            if len(pairs) == len(slots):
                break
            k = int(k)
            if not 0 <= k < self.layers * self.experts or key_slot[k] >= 0 or k in seen:
                continue
            seen.add(k)
            pairs.append((k, slots[len(pairs)]))
        # recent residency: a preloaded expert leaves before any routed one, the profile's coldest first
        self.directory.assign(pairs, last=[-1 - i for i in range(len(pairs))])
        if pairs:
            self.directory.set_hand(pairs[-1][1] + 1)
        self._copy_bundles(pairs)
        if self.device.type == "cuda":
            self.torch.cuda.current_stream(self.device).synchronize()
        self._snapshot = None
        return pairs

    def pin(self, keys):
        """The pinned core: these keys' bundles in slots of their own for good. No plan lists such a slot as a
        victim, no shrink gives it back (the cache never shrinks below its core) and no preload or prompt ring
        takes it, so an expert of the core never misses and its bundle is never read from the store again: a store
        that left it out (leave()) holds each of them once, on the device. Under the recent policy, before anything
        is resident (right after the cache is made); each bundle is read with the store's read(). Returns the
        (key, slot) pairs."""
        torch = self.torch
        if self._recent is None:
            raise ValueError("a pinned core needs the recent policy: its rule lists victims among the other slots")
        if self.pinned:
            raise ValueError("the cache has its pinned core already")
        keys = list(dict.fromkeys(int(k) for k in keys))
        if any(not 0 <= k < self.layers * self.experts for k in keys):
            raise ValueError("a key to pin is outside the cache's layers and experts")
        self.end_prompt()
        self._recent.settle()
        key_slot, slot_key = self.directory.snapshot()
        ring = set(self.ring_slots)
        slots = [s for s in range(self.total) if slot_key[s] == dd.EMPTY and s != self.scratch and s not in ring]
        if any(key_slot[k] >= 0 for k in keys) or len(keys) > len(slots) - 1:
            raise ValueError(f"a pinned core of {len(keys):,} experts needs as many empty slots beside the ring and "
                             f"one more for the experts that come and go: {len(slots):,} are empty (pin before any "
                             "preload)")
        pairs = list(zip(keys, slots))
        cuda = self.device.type == "cuda"
        for first in range(0, len(pairs), 64):   # through read(): a store that left them out reads its file
            part = pairs[first:first + 64]
            staging = torch.empty(len(part), self.bundle_bytes, dtype=torch.uint8, pin_memory=cuda)
            self.source.read([k for k, _ in part], staging)
            for i, (_, slot) in enumerate(part):
                self.store[slot].copy_(staging[i], non_blocking=True)
            if cuda:
                torch.cuda.current_stream(self.device).synchronize()   # the staging buffer is read before it goes
        self.directory.assign(pairs)
        self.pinned = {slot: key for key, slot in pairs}
        if pairs:
            self._recent.eligible[torch.tensor([slot for _, slot in pairs], device=self.device)] = False
        self._snapshot = None
        return pairs

    def pinned_intact(self):
        """Whether every slot of the pinned core still holds its key (synchronizes): a check for tests and reports."""
        if not self.pinned:
            return True
        slots = self.torch.tensor(sorted(self.pinned), device=self.device)
        keys = self.torch.tensor([self.pinned[s] for s in sorted(self.pinned)], dtype=self.directory.slot_key.dtype,
                                 device=self.device)
        return bool((self.directory.slot_key[slots] == keys).all())

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

    def _copy_bundles(self, pairs, stream=None, keep=False):
        """(key, slot) bundles copied from the source by the copy engine on `stream` (the current one by default).
        keep: the slots are a prompt's ring, so a store with a host tier keeps its copy."""
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
            if keep and getattr(self.source, "retains", False):
                self.source.read([k for k, _ in pairs], staging, keep=True)
            else:
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
        if self._recent is not None:
            self._recent.settle()
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
    def rows(self):
        """The store's slots as rows of the caller's own bytes ([S][row_bytes] uint8, a view of store without the
        padding): a slot table's slot s holds its key's row at rows()[s]."""
        return self.store[:, :self.row_bytes]

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
                "prompt_cpu_routes": self.prompt_cpu_routes, "routed_chunks": self.routed_chunks,
                "failure": c["failure"], "swaps": c["swaps"], "kernel": self.directory.kernel is not None,
                "pinned": len(self.pinned)}

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
        if getattr(self, "elastic", None) is not None:
            self.torch.cuda.synchronize(self.device)
            self.elastic.close()
            self.elastic = None
        if self.lend is not None:
            self.lend.on_reserve(None)
            self.lend = None
        self._staged.clear()
        self._ring.clear()


class _Recent:
    """Recent residency on the device (the rule trace_replay.recent replays). A plan sends every miss to the CPU tier
    and gives it the next slot of the round's victim list that is still open and was not routed this round (the
    directory's plan); its bundle is copied into the slot on a side stream beside the round, the slot and the key
    reserved meanwhile. A round's start maps the copies of the round before (each expert stamped with the round it
    was missed in) and lists the victims: the `swaps` open slots whose expert was routed longest ago, empty slots
    first, ties to the lower slot. Fixed shapes, capture-safe; the copies need the directory's kernel in a capture."""

    def __init__(self, cache, swaps, reserve=0):
        torch = cache.torch
        d, dev = cache.directory, cache.device
        if int(swaps) < 1:
            raise ValueError("recent residency copies at least one miss a round (max_swaps)")
        self.cache = cache
        S = d.slots
        # the round's victims: `swaps` for the CPU tier's copies beside the round, then the link layers' reserve
        R = self.swaps = min(int(swaps) + int(reserve), S)
        i32 = dict(dtype=torch.int32, device=dev)
        self.victims = torch.full((R + 1,), S, **i32)        # the last a dummy
        # the round, the next victim, the victims, the copies, the link layers' next victim (from the end), the reserve
        self.state = torch.zeros(6, **i32)
        self.state[5] = int(reserve)
        self.pending = torch.zeros(R + 1, 2, **i32)          # the round's copies (key, slot); the last a dummy
        self.ranges = torch.zeros(4 * cache.layers, 2, **i32)   # each plan call's range of pending (first, count)
        self.calls = 0
        fixed = set(cache.ring_slots) if cache.ring_mode == "fixed" else set()
        self.eligible = torch.tensor([s != cache.scratch and s not in fixed for s in range(S)], dtype=torch.bool,
                                     device=dev)
        self._ar_r = torch.arange(R, device=dev)
        self._ar_s = torch.arange(S, device=dev)
        self.side = torch.cuda.Stream(dev) if dev.type == "cuda" else None
        self._forked = False
        self.listed = False      # the victims are listed for the round under way

    def call(self, capturing):
        """The directory's recent tensors for one plan call: (victims, state, pending, this call's range)."""
        if self.calls >= self.ranges.shape[0]:
            raise ValueError(f"a round plans at most {self.ranges.shape[0]} times through a recent cache "
                             "(start_round() at each round's start)")
        span = self.ranges[self.calls]
        self.calls += 1
        return self.victims, self.state, self.pending, span

    def fill(self, span, capturing):
        """A plan call's copies, beside the round: on the side stream, after the plan that listed them."""
        cache, torch = self.cache, self.cache.torch
        d = cache.directory
        if cache._mapped() and d.kernel is not None:
            if self.side is None:
                d.fill_counted(self.pending, span[1:2], cache.store, cache._region[1], cache.scratch, self.swaps,
                               cache.fill_blocks, cache.fill_vector, first=span[0:1])
                return
            self.side.wait_stream(torch.cuda.current_stream(cache.device))
            with torch.cuda.stream(self.side):
                d.fill_counted(self.pending, span[1:2], cache.store, cache._region[1], cache.scratch, self.swaps,
                               cache.fill_blocks, cache.fill_vector, first=span[0:1])
            self._forked = True
            return
        if capturing:
            raise RuntimeError("recent residency copies its misses by the directory's kernel from a mapped store in a "
                               "captured round; here they are copied on the host")
        first, count = span.tolist()
        if count:
            cache._copy_bundles([tuple(job) for job in self.pending[first:first + count].tolist()])

    def join(self):
        """The side stream's copies land before anything later on the current stream."""
        if self._forked:
            self.cache.torch.cuda.current_stream(self.cache.device).wait_stream(self.side)
            self._forked = False

    def commit(self):
        """The copies that landed are mapped: each pending (key, slot) resident, stamped with the round it missed in."""
        torch, d = self.cache.torch, self.cache.directory
        K, S, R = d.keys, d.slots, self.swaps
        live = self._ar_r < self.state[3]
        keys = torch.where(live, self.pending[:R, 0].long(), K)
        slots = torch.where(live, self.pending[:R, 1].long(), S)
        d.slot_key.index_copy_(0, slots, keys.int())
        d.key_slot.index_copy_(0, keys, slots.int())
        d.slot_last.index_copy_(0, slots, self.state[0].expand(R).contiguous())
        d.key_slot[K:].fill_(dd.EMPTY)
        d.slot_key[S:].fill_(dd.EMPTY)
        d.slot_last[S:].fill_(dd.NEVER)
        self.state[3:4].zero_()

    def settle(self):
        """Before a host-driven change of the slots (a prompt's ring, a shrink, a preload): the copies under way land
        and are mapped; the next plan lists the victims again. Between steps, never captured."""
        self.join()
        self.commit()
        self.listed = False

    def round(self):
        torch, d = self.cache.torch, self.cache.directory
        S, R = d.slots, self.swaps
        self.join()
        self.commit()
        self.state[0:1].add_(1)
        open_slot = self.eligible & (d.slot_key[:S] >= dd.EMPTY)      # not reserved, reclaimed or a prompt's ring now
        stamp = (d.slot_last[:S].long() + (1 << 31)) * (1 << 21) + self._ar_s
        first = torch.sort(torch.where(open_slot, stamp, 1 << 62)).values[:R]
        found = first < (1 << 62)
        self.victims[:R].copy_(torch.where(found, first % (1 << 21), S).int())
        self.state[1:2].zero_()
        self.state[2:3].copy_(found.sum().int().view(1))
        self.state[4:5].copy_(self.state[2:3])   # the link layers take victims from the list's end
        self.calls = 0
        self.listed = True


class _Ranked:
    """Ranked residency on the device (ranked_residency.RankedPolicy's rule, applied in the round): every round
    counts; at every `period`-th, each key's score decays and gains its routes since the last boundary, every key of
    every layer is ranked (ties to the lower key), and the top keys that are not resident are swapped into the slots
    that may take them (not the ring, not reclaimed or reserved ones): empty slots first in slot order, then the
    lowest-ranked resident keys (score, then key) while their score is lower, at most max_swaps (burst_swaps when more
    than burst_fraction of the slots' worth of top keys is missing). The swaps map at once and their bundles are copied
    in the round like misses. Torch ops: fixed shapes, capture-safe."""

    def __init__(self, cache, decay, period, max_swaps, burst_swaps, fraction, side=False):
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
        # side: a boundary's bundles are copied on a side stream beside the round, the
        # victims unavailable meanwhile; the round joins it at its end (count_round) and the next round maps them.
        self.side = torch.cuda.Stream(dev) if side and dev.type == "cuda" else None
        self.pending = torch.stack([torch.full((M + 1,), K, dtype=torch.int64, device=dev),
                                    torch.full((M + 1,), S, dtype=torch.int64, device=dev)], 1) if self.side else None
        self._forked = False

    def join(self):
        """The side stream's swap copies land before anything later on the current stream."""
        if self._forked:
            self.cache.torch.cuda.current_stream(self.cache.device).wait_stream(self.side)
            self._forked = False

    def _apply_pending(self):
        """Last boundary's copied swaps mapped: each pending (key, slot) now resident, sentinels after."""
        d, K, S = self.cache.directory, self.cache.directory.keys, self.cache.directory.slots
        new, vs = self.pending[:, 0], self.pending[:, 1]
        d.slot_key.index_copy_(0, vs, torch_where(self.cache.torch, new < K, new, dd.EMPTY).int())
        d.key_slot.index_copy_(0, new, vs.int())
        d.key_slot[K:].fill_(dd.EMPTY)
        d.slot_key[S:].fill_(dd.EMPTY)
        self.pending[:, 0].fill_(K)
        self.pending[:, 1].fill_(S)

    def round(self):
        torch, cache = self.cache.torch, self.cache
        d = cache.directory
        if self.side is not None:
            self.join()
            self._apply_pending()
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
        if self.side is not None:   # the victims out of use until their new bundles land (the next round maps them)
            d.slot_key.index_copy_(0, vs, torch.where(ok, dd.UNAVAILABLE, dd.EMPTY).int())
            self.pending.index_copy_(0, torch.where(ok, i, M), torch.stack([new, vs], 1))
        else:
            d.slot_key.index_copy_(0, vs, torch.where(ok, new, dd.EMPTY).int())
            d.key_slot.index_copy_(0, new, vs.int())
        d.slot_ref.index_fill_(0, vs, 0)
        d.key_slot[K:].fill_(dd.EMPTY)
        d.slot_key[S:].fill_(dd.EMPTY)
        self.jobs.index_copy_(0, torch.where(ok, i, M), torch.stack([new, vs], 1).int())
        self.count.copy_(found.view(1))
        self.swapped.add_(found)
        self.bursts.add_((burst & boundary & (found > 0)).long())
        if cache._mapped() and self.side is not None:
            self.side.wait_stream(torch.cuda.current_stream(cache.device))
            with torch.cuda.stream(self.side):
                d.fill_counted(self.jobs, self.count, cache.store, cache._region[1], cache.scratch, M,
                               cache.fill_blocks, cache.fill_vector, cache.program)
            self._forked = True
        elif cache._mapped():
            d.fill_counted(self.jobs, self.count, cache.store, cache._region[1], cache.scratch, M,
                           cache.fill_blocks, cache.fill_vector, cache.program)
        elif not (cache.device.type == "cuda" and torch.cuda.is_current_stream_capturing()):
            g = int(found)
            if g:
                cache._copy_bundles([tuple(job) for job in self.jobs[:g].tolist()])
        else:
            raise RuntimeError("ranked swaps over a file tier read on the host: no captured rounds")


def torch_where(torch, condition, a, b):
    return torch.where(condition, a, b)


class _Nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
