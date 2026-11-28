"""device-expert-cache's device form against its host twin: the directory's plans (CUDA ops and torch ops) give
LayerDirectory's slots, copies, counters, reference bits and hand bit for bit over random route streams (duplicates,
-1, routes outside the experts, unavailable slots, quotas, second touch, failures); the cache's slots hold their
experts' bundles byte for byte; a plan and its fill capture in a CUDA graph and replay (the torch ops too, with a quota
table and second touch); the prompt ring never hands a layer a segment its copies have not reached and never
overwrites one a layer has not read; ranked residency swaps inside captured rounds by RankedPolicy's decisions; recent
residency gives RecentResidency's hits, copies and slots plan for plan (CUDA ops and torch ops, eager and in a
captured round, across a prompt's borrowed ring); a KV pool's unused pages serve as slots and are reclaimed before KV
lands there, whatever the bundle size; an elastic cache grows and shrinks with its addresses fixed and keeps its
hottest experts; a cache that could lose a route refuses to be made."""
from pathlib import Path
import random
import sys

from crit_plugins.testing import PluginTestCase

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import device_cache as dc  # noqa: E402
import device_directory as dd  # noqa: E402
import ranked_residency  # noqa: E402
import recent_residency  # noqa: E402


def _twin_of(cache):
    """The host twin of a cache's directory: its slot positions, the scratch slot (and a fixed ring) unavailable."""
    twin = dc.LayerDirectory(slots=cache.total, groups=cache.layers, experts=cache.experts)
    twin.set_unavailable([s for s in range(cache.total) if cache.directory.slot_key[s].item() == dd.UNAVAILABLE])
    return twin


def _twin_state(twin, keys):
    key_slot = [twin.key_slot.get(k, -1) for k in range(keys)]
    slot_key = [-1 if v is None else -2 if v == dc.RESERVED else -3 if v == dc.UNAVAILABLE else v
                for v in twin.slot_key]
    return key_slot, slot_key, list(twin.slot_ref)


def _recent_twin(cache, swaps):
    """The host twin of a recent cache: its slot positions, the slots it cannot use closed."""
    twin = recent_residency.RecentResidency(cache.layers * cache.experts, cache.total, swaps)
    twin.close_slots([s for s in range(cache.total) if cache.directory.slot_key[s].item() == dd.UNAVAILABLE])
    return twin


class _View:
    """A round's counters as crit_plugins.rounds' RoundView gives them (counter(name) -> [rows] int64)."""

    def __init__(self, torch, device, rows):
        self.counter_names = ("accepted", "expert_reads", "expert_hits")
        self.counters = torch.zeros(rows, 3, dtype=torch.int64, device=device)

    def counter(self, name):
        return self.counters[:, self.counter_names.index(name)]


class _Retaining:
    """A file tier as the cache sees one: no device mapping, read() on the host, and retains set, so the cache hands
    back the bundles it evicts. It keeps each bundle once, as such a store does: a bundle read for a cache slot
    leaves `held`, one read for a prompt's ring (keep) stays."""

    mapped, retains = False, True

    def __init__(self, torch, layers, experts, bundle=64, seed=3):
        g = torch.Generator().manual_seed(seed)
        self.keys, self.bundle_bytes, self.layout = layers * experts, bundle, {"bundle_bytes": bundle}
        self.all = torch.randint(0, 256, (self.keys, bundle), dtype=torch.uint8, generator=g)
        self.held = {}                      # key -> the bytes the cache handed back
        self.file_reads, self.host_hits, self.kept = [], [], []

    def read(self, keys, into, keep=False):
        for i, key in enumerate(keys):
            if key in self.held:
                into[i].copy_(self.held[key] if keep else self.held.pop(key))
                self.host_hits.append(key)
            else:
                into[i].copy_(self.all[key])
                self.file_reads.append(key)
            if keep:
                self.kept.append(key)

    def retain(self, keys, rows):
        for key, row in zip(keys, rows):
            self.held[int(key)] = row.clone()


class DeviceTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch") if self.device() else None

    def kernels(self):
        return (True, False) if self.device() == "cuda" else (False,)

    def store(self, layers, experts, bundle=256, seed=0):
        import expert_store
        g = self.torch.Generator().manual_seed(seed)
        store = expert_store.BundleStore({"bundle_bytes": bundle}, layers, experts, self.device())
        self.addCleanup(store.close)
        for layer in range(layers):
            for expert in range(experts):
                store.put(layer, expert, self.torch.randint(0, 256, (bundle,), dtype=self.torch.uint8, generator=g))
        return store

    def check_slots(self, cache, store, layer, table, routed=None):
        for e, slot in enumerate(table):
            if slot >= 0 and (routed is None or e in routed):
                self.assertTrue(self.torch.equal(cache.store[slot].cpu(), store.bundle(layer, e)),
                                f"layer {layer} expert {e} in slot {slot}")

    # -- the directory ------------------------------------------------------------------------------------------
    def run_stream(self, seed, kernel, layers=3, experts=16, slots=20, quota=-1, table=None, second=False, steps=250,
                   unavailable=()):
        torch, device = self.torch, self.device()
        rng = random.Random(seed)
        d = dd.DeviceDirectory(layers, experts, slots, device, second_touch=second, admit_quota=quota,
                               quota={1: table} if table else None, kernel=kernel)
        if kernel:
            self.assertIsNotNone(d.kernel, "the directory's CUDA ops did not build")
        twin = dc.LayerDirectory(slots=slots, groups=layers, experts=experts, admit_quota=quota,
                                 admit_by_misses=table, second_touch=second)
        d.set_unavailable(list(unavailable))
        twin.set_unavailable(list(unavailable))
        counters = torch.zeros(8, 2, dtype=torch.int64, device=device)
        rows_twin, totals, failure = {}, [0, 0, 0, 0], 0
        for step in range(steps):
            layer, tokens, top = rng.randrange(layers), rng.choice([1, 2, 3, 5]), rng.choice([1, 2, 3])
            ids = [rng.randrange(experts) if rng.random() > 0.05 else -1 for _ in range(tokens * top)]
            if rng.random() < 0.02:
                ids[0] = experts + 3
            rows = sorted(rng.randrange(4) for _ in range(tokens))
            d.plan(layer, torch.tensor(ids, device=device).view(tokens, top), top,
                   torch.tensor(rows, device=device), counters)
            plan = twin.plan_layer(layer, ids, top, rows)
            for r, (a, b) in plan.rows.items():
                rows_twin.setdefault(r, [0, 0])
                rows_twin[r][0] += a
                rows_twin[r][1] += b
            for i, v in enumerate((plan.reads, plan.hits, len(plan.fills), plan.cpu)):
                totals[i] += v
            failure |= plan.failure
            self.assertEqual(d.tables[layer].tolist(), plan.table, f"step {step}")
            count = int(d.copy_count[layer])
            self.assertEqual([tuple(j) for j in d.copies[layer][:count].tolist()], plan.fills, f"step {step}")
            if step % 10 == 0:
                key_slot, slot_key = d.snapshot()
                self.assertEqual((key_slot, slot_key, d.slot_ref[:slots].tolist()),
                                 _twin_state(twin, layers * experts), f"step {step}")
                self.assertEqual(d.state[:3].tolist(), [twin.hand, twin.stamp, failure], f"step {step}")
                self.assertEqual(d.counters[:4].tolist(), totals, f"step {step}")
            if rng.random() < 0.03:
                some = rng.sample(range(slots), 3)
                change = ("set_unavailable", "set_available")[rng.random() < 0.5]
                getattr(d, change)(some)
                getattr(twin, change)(some)
        self.assertEqual({r: v for r, v in enumerate(counters.tolist()) if v != [0, 0]}, rows_twin)

    def test_the_device_directory_plans_as_its_host_twin(self):
        cases = [{}, {"slots": 7}, {"quota": 1}, {"table": [0, 1, 1, 2, 0]}, {"second": True},
                 {"unavailable": (0, 5, 6)}, {"slots": 40, "experts": 64, "layers": 2}, {"slots": 33}]
        for kernel in self.kernels():
            for case in cases:
                with self.subTest(kernel=kernel, **{k: str(v) for k, v in case.items()}):
                    for seed in range(2):
                        self.run_stream(seed, kernel, **case)

    def test_quota_tables_go_by_token_bucket(self):
        torch, device = self.torch, self.device()
        for kernel in self.kernels():
            d = dd.DeviceDirectory(1, 16, 20, device, admit_quota=-1, quota={1: [9], 4: [0]}, kernel=kernel)
            d.plan(0, torch.tensor([[1, 2, 3]], device=device), 3)            # 1 token: every miss admitted
            self.assertTrue(all(s >= 0 for s in d.tables[0][[1, 2, 3]].tolist()))
            d.plan(0, torch.tensor([[4], [5], [6], [7]], device=device), 1)  # 4 tokens: quota 0
            self.assertEqual(d.tables[0][[4, 5, 6, 7]].tolist(), [dc.CPU] * 4)

    def test_prompt_plans_read_hits_in_place_and_copy_misses_into_the_ring(self):
        torch, device = self.torch, self.device()
        for kernel in self.kernels():
            rng = random.Random(4)
            d = dd.DeviceDirectory(2, 12, 30, device, kernel=kernel)
            twin = dc.LayerDirectory(slots=30, groups=2, experts=12)
            for _ in range(20):
                ids = [rng.randrange(12) for _ in range(4)]
                d.plan(0, torch.tensor(ids, device=device), 1)
                twin.plan_layer(0, ids)
            ring = list(range(20, 30))
            for _ in range(10):
                ids = [rng.randrange(12) if rng.random() > 0.1 else -1 for _ in range(15)]
                d.plan(0, torch.tensor(ids, device=device).view(5, 3), 3,
                       ring=torch.tensor(ring, dtype=torch.int32, device=device))
                table, copies, _, _ = twin.prompt_layer(0, ids, ring)
                self.assertEqual(d.tables[0].tolist(), table)
                count = int(d.copy_count[0])
                self.assertEqual([tuple(j) for j in d.copies[0][:count].tolist()], copies)
            self.assertEqual(d.snapshot()[0], _twin_state(twin, 24)[0])   # prompts admit nothing

    # -- the cache ----------------------------------------------------------------------------------------------
    def cache(self, store, layers, experts, slots, layout=None, link=None, **options):
        plugin = self.assemble(options={"native": options.pop("native", True), **options})["expert_residency"]
        layout = layout or {"bundle_bytes": store.bundle_bytes}
        extra = {"link_layers": link[0], "link_slots": link[1]} if link else {}
        return plugin.cache(layers, experts, layout, slots, self.device(), store, **extra)

    def test_slots_hold_their_experts_bundles_and_telemetry_counts_them(self):
        torch, device = self.torch, self.device()
        store = self.store(3, 24)
        for native in self.kernels():
            plugin = self.assemble(options={"native": native, "ring_slots": 0})["expert_residency"]
            cache = plugin.cache(3, 24, {"bundle_bytes": 256}, 40, device, store)
            twin = _twin_of(cache)
            rng = random.Random(1)
            reads = hits = fills = 0
            for _ in range(60):
                layer, tokens = rng.randrange(3), rng.choice([1, 2, 4])
                ids = [rng.randrange(24) for _ in range(tokens * 3)]
                table = cache.plan(layer, torch.tensor(ids, device=device).view(tokens, 3)).tolist()
                plan = twin.plan_layer(layer, ids, 3)
                reads, hits, fills = reads + plan.reads, hits + plan.hits, fills + len(plan.fills)
                self.assertEqual(table, plan.table)
                self.check_slots(cache, store, layer, table)
            self.assertEqual(set(plugin.telemetry()["expert_cache"]), set(cache.telemetry(sync=True)))
            seen = cache.telemetry(sync=True)
            self.assertEqual((seen["slots"], seen["failure"], seen["reads"], seen["hits"], seen["fills"],
                              seen["fill_bytes"]), (40, 0, reads, hits, fills, fills * 256))
            self.assertEqual(seen["resident"], len(twin.key_slot))
            self.assertEqual(plugin.facts()["expert_cache"]["directory"], "cuda" if native else "torch")

    def test_a_store_that_retains_gets_every_evicted_bundle_back_and_holds_each_expert_once(self):
        torch, device = self.torch, self.device()
        layers, experts, slots = 4, 24, 30   # fewer slots than the 96 experts, a prompt's ring among them
        for native in self.kernels():
            with self.subTest(native=native):
                source = _Retaining(torch, layers, experts)
                plugin = self.assemble(options={"native": native, "ring_slots": experts})["expert_residency"]
                cache = plugin.cache(layers, experts, source.layout, slots + experts, device, source)
                self.assertTrue(cache.round_problems())            # a file tier: read on the host, not captured
                rng = random.Random(8)
                for step in range(120):
                    layer = rng.randrange(layers)
                    ids = torch.tensor([rng.randrange(experts) for _ in range(6)], device=device).view(2, 3)
                    table = cache.plan(layer, ids).tolist()
                    for e in set(ids.reshape(-1).tolist()):
                        self.assertTrue(torch.equal(cache.store[table[e]].cpu(), source.all[layer * experts + e]))
                    _, slot_key = cache.directory.snapshot()
                    resident = {key for key in slot_key if key >= 0}
                    # Each expert is in a cache slot or in the store's hands, never both, and what came back is
                    # the expert's own bytes.
                    self.assertFalse(resident & set(source.held), f"step {step}")
                    for key, row in source.held.items():
                        self.assertTrue(torch.equal(row, source.all[key]))
                # Every expert that was ever read and is not resident came back: nothing was dropped on the way.
                self.assertEqual(set(source.file_reads) - resident, set(source.held))
                self.assertTrue(source.host_hits)                  # and evicted experts came up again from the host
                self.assertEqual(len(set(source.file_reads)), len(source.file_reads))   # the file once an expert
                # A prompt chunk's ring slots keep nothing: the store keeps its copy of what it hands them.
                held = dict(source.held)
                ids = torch.tensor(sorted(key % experts for key in held if key < experts)[:3], device=device).view(1, -1)
                if ids.numel():
                    _, table = cache.prompt(0, ids)
                    self.assertEqual(sorted(source.kept), ids.reshape(-1).tolist())
                    self.assertEqual(set(source.held), set(held))
                    cache.end_prompt()
                cache.close()

    def test_a_plan_and_its_fill_capture_and_replay(self):
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        torch = self.torch
        store = self.store(2, 32, bundle=512)
        cases = [{"native": native, "ring_slots": 0} for native in self.kernels()]
        cases.append({"native": False, "ring_slots": 0, "cpu_tier": True, "quota_table": "0,2,1", "second_touch": True})
        for options in cases:
            with self.subTest(**options):
                cache = self.cache(store, 2, 32, 40, **options)
                twin = _twin_of(cache)
                twin.admit_by_misses = [0, 2, 1] if "quota_table" in options else None
                twin.second_touch = options.get("second_touch", False)
                ids = torch.zeros(4, 2, dtype=torch.int64, device="cuda")
                rng = random.Random(7)
                first = [rng.randrange(32) for _ in range(8)]
                ids.copy_(torch.tensor(first).view(4, 2))
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    cache.plan(1, ids)            # once outside capture at the shape
                    twin.plan_layer(1, first, 2)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        table = cache.plan(1, ids)
                for _ in range(12):
                    routes = [rng.randrange(32) for _ in range(8)]
                    ids.copy_(torch.tensor(routes).view(4, 2))
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = twin.plan_layer(1, routes, 2).table
                    self.assertEqual(table.tolist(), expected)
                    self.check_slots(cache, store, 1, expected)

    def test_a_recording_cache_keeps_its_last_decode_routes_in_order(self):
        torch, device = self.torch, self.device()
        store = self.store(2, 16)
        cache = self.cache(store, 2, 16, 20, ring_slots=0, record_routes=5)
        self.assertEqual(self.cache(store, 2, 16, 20, ring_slots=0).routes(), [])
        cache.plan(0, torch.tensor([[1, 2], [3, -1]], device=device))
        cache.plan(1, torch.tensor([[4, 5], [-1, -1]], device=device))   # a held row records nothing
        self.assertEqual(cache.routes(), [(0, [1, 2]), (0, [3]), (1, [4, 5])])
        cache.plan(0, torch.tensor([[6, 7], [8, 9]], device=device))   # past the ring of 5 rows: the oldest goes
        self.assertEqual(cache.routes(), [(0, [3]), (1, [4, 5]), (0, [6, 7]), (0, [8, 9])])
        if device == "cuda":
            ids = torch.tensor([[10, 11]], device=device)
            cache.plan(1, ids)   # once at the shape outside a capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                cache.plan(1, ids)
            ids.copy_(torch.tensor([[12, 13]], device=device))
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(cache.routes()[-2:], [(1, [10, 11]), (1, [12, 13])])

    def test_count_round_adds_each_rows_reads_and_hits_into_the_rounds_counters(self):
        torch, device = self.torch, self.device()
        store = self.store(1, 16)
        cache = self.cache(store, 1, 16, 20, ring_slots=0, round_rows=4)
        view = _View(torch, device, 3)
        view.counter("expert_reads").fill_(5)
        view.counter("expert_hits").fill_(1)
        cache.plan(0, torch.tensor([[1, 2], [2, 3]], device=device), rows=torch.tensor([0, 2], device=device))
        cache.plan(0, torch.tensor([[2, 9], [1, 1]], device=device), rows=torch.tensor([0, 0], device=device))
        self.assertEqual(cache.row_counters()[:3].tolist(), [[5, 2], [0, 0], [2, 0]])
        cache.count_round(view)
        self.assertEqual((view.counter("expert_reads").tolist(), view.counter("expert_hits").tolist()),
                         ([5 + 2 + 3, 5, 5 + 2], [1 + 0 + 2, 1, 1]))
        self.assertEqual(cache.row_counters().sum().item(), 0)
        self.assertEqual(tuple(cache.row_counters().shape), (4, 2))   # made once, at the round rows
        with self.assertRaisesRegex(ValueError, "row counters"):
            cache.plan(0, torch.tensor([[1]], device=device), rows=torch.tensor([4], device=device))
        quiet = cache.telemetry()   # no wait for the device: a snapshot, in the end the same counters
        self.assertEqual(set(quiet), set(cache.telemetry(sync=True)))

    def test_decided_sees_the_slots_before_the_copies(self):
        torch, device = self.torch, self.device()
        store = self.store(1, 16)
        cache = self.cache(store, 1, 16, 20, ring_slots=0, admit_quota=1, cpu_tier=True)
        seen = []
        cache.plan(0, torch.tensor([[3, 4, 5]], device=device), decided=lambda slots: seen.append(slots.tolist()))
        self.assertEqual(len(seen), 1)
        self.assertEqual([seen[0][3] >= 0, seen[0][4], seen[0][5]], [True, dc.CPU, dc.CPU])

    def test_a_quota_sends_misses_to_the_cpu_tier_and_needs_one(self):
        torch, device = self.torch, self.device()
        store = self.store(1, 16)
        with self.assertRaisesRegex(ValueError, "cpu_tier"):
            self.cache(store, 1, 16, 20, admit_quota=1)
        cache = self.cache(store, 1, 16, 20, admit_quota=1, cpu_tier=True, ring_slots=0)
        table = cache.plan(0, torch.tensor([[3, 4, 5]], device=device)).tolist()
        self.assertEqual([table[3] >= 0, table[4], table[5]], [True, dc.CPU, dc.CPU])
        self.assertEqual(cache.telemetry(sync=True)["cpu"], 2)

    def test_a_cache_that_could_lose_a_route_or_misread_its_store_is_refused(self):
        import expert_store
        store = self.store(1, 16)
        with self.assertRaisesRegex(ValueError, "without a CPU tier"):
            self.cache(store, 1, 16, 12, ring_slots=0)           # fewer own slots than a layer's experts
        with self.assertRaisesRegex(ValueError, "without a CPU tier"):
            self.cache(store, 1, 16, 24, ring_slots=16)          # the ring leaves too few
        with self.assertRaisesRegex(ValueError, "ring holds a layer's every expert"):
            self.cache(store, 1, 16, 40, ring_slots=8)
        # Its own ring: a cache too small for a layer's experts beside the slots a plan needs is refused when it is
        # made, naming the slots prompts take, not at its first prompt; with a CPU tier the same slots hold a ring.
        with self.assertRaisesRegex(ValueError, "holds no prompt ring.*prompts take 32 slots"):
            self.cache(store, 1, 16, 20)
        self.assertEqual(len(self.cache(store, 1, 16, 20, cpu_tier=True).ring_slots), 16)
        self.assertEqual(len(self.cache(store, 1, 16, 12, cpu_tier=True, ring_slots=0).own), 12)
        other = expert_store.BundleStore({"bundle_bytes": 512}, 1, 16, self.device())
        self.addCleanup(other.close)
        with self.assertRaisesRegex(ValueError, "512 bytes"):
            self.cache(other, 1, 16, 20, {"bundle_bytes": 256}, ring_slots=0)   # the store's bundles: another size
        with self.assertRaisesRegex(ValueError, "16 bundles"):
            self.cache(store, 2, 16, 20, ring_slots=0)           # the store holds another geometry
        csf = expert_store.csf_layout({"bundle_bytes": 256, "components": {"a": (0, "uint8", (16, 8)),
                                                                           "b": (128, "uint8", (128,))}}, {"a": 4})
        packed = expert_store.BundleStore(csf, 1, 16, self.device())
        self.addCleanup(packed.close)
        with self.assertRaisesRegex(ValueError, "CPU tier reads bundles in place"):
            self.cache(packed, 1, 16, 20, {"bundle_bytes": 256}, ring_slots=0, cpu_tier=True)

    def test_preload_maps_a_profile_and_the_clock_starts_after_it(self):
        torch, device = self.torch, self.device()
        store = self.store(2, 16)
        cache = self.cache(store, 2, 16, 20, ring_slots=0)
        twin = _twin_of(cache)
        keys = [17, 3, 3, 40, 5]
        self.assertEqual(cache.warm({"keys": keys}), twin.preload(keys))
        for layer in (1, 0):
            for e in range(16):
                key = layer * 16 + e
                slot = cache.directory.snapshot()[0][key]
                if slot >= 0:
                    self.assertTrue(torch.equal(cache.store[slot].cpu(), store.bundle(layer, e)))
        ids = [3, 7, 8]
        self.assertEqual(cache.plan(0, torch.tensor(ids, device=device)).tolist(), twin.plan_layer(0, ids).table)
        with self.assertRaises(KeyError):
            cache.warm({"slots": [1]})          # a profile dict names its keys
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as folder:   # a profile file, checked against the cache's geometry
            path = Path(folder) / "profile.json"
            path.write_text(json.dumps({"kind": "device-expert-cache-profile", "groups": 2, "experts": 16,
                                        "slots": 4, "keys": [30, 31]}), encoding="utf-8")
            fresh = self.cache(store, 2, 16, 20, ring_slots=0)
            self.assertEqual(fresh.warm(path), [(30, 0), (31, 1)])
            path.write_text(json.dumps({"kind": "device-expert-cache-profile", "groups": 3, "experts": 16,
                                        "slots": 4, "keys": [1]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "geometry"):
                fresh.warm(path)

    def test_prompt_chunks_read_staged_bundles_and_admit_nothing(self):
        if self.device() != "cuda":
            self.skipTest("the copy engine")
        torch = self.torch
        layers, experts = 4, 24
        store = self.store(layers, experts, bundle=4096, seed=3)
        expected = torch.stack([store.bundle(l, e).view(torch.int32).sum(dtype=torch.int64)
                                for l in range(layers) for e in range(experts)]).view(layers, experts).cuda()
        for native in self.kernels():
            for ring in ("borrow", "fixed"):
                cache = self.cache(store, layers, experts, 80, native=native, ring_slots=2 * experts, ring=ring)
                rng = random.Random(2)
                sums = torch.zeros(4, layers, experts, dtype=torch.int64, device="cuda")
                for chunk in range(4):   # the last in another layer order: what was staged ahead is let go
                    for _ in range(3):   # decode rounds between the prompt steps move residency
                        for layer in range(layers):
                            cache.plan(layer, torch.tensor([rng.randrange(experts) for _ in range(6)], device="cuda"))
                    before = cache.directory.snapshot()[0]
                    for layer in (range(layers) if chunk < 3 else (2, 0, 3, 1)):
                        routes = 30 if chunk != 1 else 4    # long chunks stage every miss, a short one its routed misses
                        ids = torch.tensor([rng.randrange(experts) for _ in range(routes)], device="cuda").view(-1, 2)
                        store_t, table = cache.prompt(layer, ids)
                        torch.cuda._sleep(200_000)   # a slow layer: staging of the next one runs under it
                        routed = torch.zeros(experts, dtype=torch.bool, device="cuda")
                        routed[ids.flatten()] = True
                        slots = table.long().clamp(min=0)
                        got = store_t[slots].view(experts, -1).view(torch.int32).sum(1, dtype=torch.int64)
                        sums[chunk, layer] = torch.where(routed, got, expected[layer])
                    if chunk == 0:
                        self.assertEqual(cache.telemetry(sync=True)["slots"], 80 - 2 * experts)
                    # a borrowed ring: the whole of it for long chunks, a slot a route for the short one
                    held = cache._ring_now
                    self.assertEqual(len(held), 4 if chunk == 1 and ring == "borrow" else 2 * experts)
                    before = [-1 if slot in set(held) else slot for slot in before]   # the borrowed slots empty
                    self.assertEqual(cache.directory.snapshot()[0], before)   # nothing admitted, nothing else evicted
                    cache.end_prompt()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(sums, expected.expand(4, -1, -1)), (native, ring))
                seen = cache.telemetry(sync=True)
                self.assertGreater(seen["prompt_copies"], 0)
                self.assertEqual(seen["slots"], 80 if ring == "borrow" else 80 - 2 * experts)

    def test_a_staged_layer_serves_every_chunk_of_a_span_across_steps(self):
        # A span run layer by layer: every chunk asks prompt() for the same layer before the next layer comes. The
        # layer's experts are staged once; a step's end keeps the ring (hold), decode plans move the residency in the
        # slots beside it, and the next step's chunk reads every expert's bundle still, only what left the cache
        # since staged again. Chunk by chunk the same chunks stage every layer once a chunk.
        if self.device() != "cuda":
            self.skipTest("the copy engine")
        torch = self.torch
        layers, experts, chunks = 4, 24, 5
        store = self.store(layers, experts, bundle=4096, seed=5)
        expected = torch.stack([store.bundle(l, e).view(torch.int32).sum(dtype=torch.int64)
                                for l in range(layers) for e in range(experts)]).view(layers, experts).cuda()

        def read(cache, layer, routes, rng):
            ids = torch.tensor([rng.randrange(experts) for _ in range(routes)], device="cuda").view(-1, 2)
            held, table = cache.prompt(layer, ids)
            self.assertTrue(bool((table >= 0).all()), "a staged layer's table names a slot for every expert")
            found = held[table.long()].view(experts, -1).view(torch.int32).sum(1, dtype=torch.int64)
            self.assertTrue(torch.equal(found, expected[layer]), f"layer {layer}")

        def decode(cache, rng, rounds=2):
            for _ in range(rounds):
                for layer in range(layers):
                    cache.plan(layer, torch.tensor([rng.randrange(experts) for _ in range(6)], device="cuda"))

        copies = {}
        for native in self.kernels():
            for order in ("chunks", "layers", "held"):
                cache = self.cache(store, layers, experts, 80, native=native, ring_slots=2 * experts)
                rng = random.Random(7)
                decode(cache, rng, 4)
                missing = sum(int(slot < 0) for slot in cache.directory.snapshot()[0])
                long = 8 * experts   # routes of a chunk that is staged whole, whatever it routes
                if order == "chunks":
                    for chunk in range(chunks):
                        for layer in range(layers):
                            read(cache, layer, long, rng)
                else:
                    for layer in range(layers):
                        for chunk in range(chunks):
                            read(cache, layer, long if chunk < chunks - 1 else 4, rng)   # the last chunk a short one
                            if order == "held" and chunk in (1, 3):   # a step ends inside the layer
                                cache.end_prompt(hold=True)
                                self.assertTrue(cache.holding and len(cache._ring_now) == 2 * experts)
                                decode(cache, rng)   # plans beside the held ring: the residency moves
                                self.assertTrue(cache.holding, "a decode plan leaves a held ring where it is")
                cache.end_prompt()
                self.assertFalse(cache.holding)
                seen = cache.telemetry(sync=True)
                self.assertEqual(seen["slots"], 80)   # the ring went back
                copies[order] = seen["prompt_copies"]
                if order == "layers":   # each layer's missing experts once, however many chunks read them
                    self.assertLessEqual(copies[order], missing + 2 * experts)
            self.assertGreater(copies["chunks"], 3 * copies["layers"], copies)
            # held across steps: what the plans in between pushed out of the cache is staged again, nothing more
            self.assertLess(copies["held"], copies["chunks"] // 2, copies)

    def test_a_chunk_that_leaves_experts_unrouted_has_only_its_spans_routes_staged(self):
        # A chunk of a few times stage_routes routes that routes a part of its layer: only the experts it routes
        # cross the link. A later chunk of the span reads the layer's experts where the first found them and has
        # what it routes beyond them staged beside; an expert no chunk routed has no slot. Once the span's chunks
        # together route nearly the whole layer the later layers are staged whole, and the next prompt step starts
        # over.
        if self.device() != "cuda":
            self.skipTest("the copy engine")
        torch = self.torch
        layers, experts = 3, 64
        store = self.store(layers, experts, bundle=4096, seed=9)
        expected = torch.stack([store.bundle(l, e).view(torch.int32).sum(dtype=torch.int64)
                                for l in range(layers) for e in range(experts)]).view(layers, experts).cuda()

        def read(cache, layer, routed, times):
            ids = torch.tensor(list(routed) * times, device="cuda").view(-1, 2)
            held, table = cache.prompt(layer, ids)
            torch.cuda.synchronize()
            picked = torch.tensor(list(routed), device="cuda")
            self.assertTrue(bool((table[picked] >= 0).all()), "every routed expert has a slot")
            found = held[table[picked].long()].view(len(routed), -1).view(torch.int32).sum(1, dtype=torch.int64)
            self.assertTrue(torch.equal(found, expected[layer][picked]), f"layer {layer}")
            return table.clone()

        for native in self.kernels():
            cache = self.cache(store, layers, experts, 200, native=native, ring_slots=2 * experts)
            copies = lambda: cache.telemetry(sync=True)["prompt_copies"]   # noqa: E731
            first = read(cache, 0, range(16), 4)            # 64 routes of 16 experts, none resident
            self.assertEqual(copies(), 16)
            self.assertEqual(int((first >= 0).sum()), 16)   # an expert the chunk does not route has no slot
            second = read(cache, 0, range(8, 32), 3)        # the span's next chunk: 16 experts beyond the first's
            self.assertEqual(copies(), 32)
            self.assertTrue(torch.equal(second[:16], first[:16]), "the first chunk's experts stay where they are")
            read(cache, 0, range(8, 32), 3)                 # nothing new: nothing copied
            self.assertEqual(copies(), 32)
            read(cache, 0, range(60), 2)                    # together nearly the whole layer
            self.assertEqual(copies(), 60)
            whole = read(cache, 1, range(16), 4)            # so the next layer is staged whole, the one after ahead
            self.assertTrue(bool((whole >= 0).all()))
            self.assertEqual(copies(), 60 + 2 * experts)
            cache.end_prompt()
            again = read(cache, 0, range(16), 4)            # another prompt step: its routes alone again
            self.assertEqual(int((again >= 0).sum()), 16)
            self.assertEqual(copies(), 60 + 2 * experts + 16)
            self.assertEqual(cache.telemetry(sync=True)["routed_chunks"], 2)
            cache.end_prompt()
            self.assertEqual(cache.telemetry(sync=True)["slots"], 200)

    def test_an_elastic_cache_grows_and_shrinks_with_its_addresses_fixed(self):
        device = self.device()
        if device != "cuda":
            self.skipTest("an elastic range is the CUDA driver's")
        from crit_plugins.memory import of
        if not of(device).elastic_ranges:
            self.skipTest("the driver's virtual memory API is not reachable here")
        layers, experts = 2, 32
        store = self.store(layers, experts, bundle=1 << 20)   # 1 MiB bundles: a 2 MiB granule holds two
        for policy, side in (("ranked", False), ("ranked", True), ("recent", True)):
            with self.subTest(policy=policy, side_swaps=side):
                self.elastic_case(store, layers, experts, policy, side)

    def elastic_case(self, store, layers, experts, policy, side):
        torch, device = self.torch, self.device()
        cache = self.cache(store, layers, experts, 4, policy=policy, cpu_tier=True, ring_slots=0, max_slots=48,
                           max_swaps=8, burst_swaps=8, period=1, side_swaps=side)
        base = cache.store.data_ptr()
        self.assertEqual(cache.backed_slots(), 4)
        rng = random.Random(3)

        def plans(rounds):
            for _ in range(rounds):
                cache.start_round(None)
                for layer in range(layers):
                    ids = torch.tensor([rng.randrange(experts) for _ in range(6)], device=device).view(2, 3)
                    table = cache.plan(layer, ids).tolist()
                    torch.cuda.synchronize()
                    self.check_slots(cache, store, layer, table)
                    self.assertTrue(all(s < cache.backed for s in table if s >= 0))
                for held in (cache._ranked, cache._recent):
                    if held is not None:
                        held.join()
        plans(8)
        self.assertEqual(cache.grow(40), 40)
        self.assertEqual(cache.store.data_ptr(), base)   # the addresses stay
        plans(12)
        resident = sum(int(x >= 0) for x in cache.directory.key_slot[:layers * experts].tolist())
        self.assertGreater(resident, 4)
        self.assertEqual(cache.shrink(10), 10)
        plans(8)
        self.assertEqual(cache.grow(30), 30)
        plans(6)
        self.assertEqual(cache.shrink(6), 6)   # right after a boundary that copied swaps on the side stream
        plans(6)
        # a shrink keeps the hottest experts: those in the leaving slots take the coldest kept experts' slots
        self.assertEqual(cache.grow(30), 30)
        plans(10)
        torch.cuda.synchronize()
        if cache._recent is not None:
            cache._recent.settle()
        elif cache._ranked.pending is not None:
            cache._ranked.join()
            cache._ranked._apply_pending()

        def ranks():   # each resident expert's rank under the policy: its score, or the round it was last routed in
            keys = cache.directory.slot_key[:cache.backed].tolist()
            if cache._recent is not None:
                last = cache.directory.slot_last[:cache.backed].tolist()
                return [last[s] for s, k in enumerate(keys) if k >= 0]
            scores = cache._ranked.scores.tolist()
            return [scores[k] for k in keys if k >= 0]
        held = sorted(ranks(), reverse=True)
        self.assertEqual(cache.shrink(8), 8)
        after = ranks()
        self.assertEqual(sorted(after, reverse=True), held[:len(after)])
        self.assertEqual(len(after), min(8, len(held)))
        for slot, key in enumerate(cache.directory.slot_key[:cache.backed].tolist()):
            if key >= 0:   # a moved expert's slot holds its bundle
                self.assertTrue(torch.equal(cache.store[slot].cpu(), store.bundle(key // experts, key % experts)))
        plans(4)
        torch.cuda.synchronize()
        cache.close()

    def test_recent_residency_plans_and_copies_by_its_rule(self):
        cuda = self.device() == "cuda"
        for native in self.kernels():
            for captured in ((False, True) if cuda and native else (False,)):
                with self.subTest(native=native, captured=captured):
                    self.recent_case(native, captured)

    def recent_case(self, native, captured, layers=3, experts=24, slots=20, swaps=6, rounds=40):
        torch, device = self.torch, self.device()
        store = self.store(layers, experts, bundle=1024)
        cache = self.cache(store, layers, experts, slots, native=native, policy="recent", cpu_tier=True, ring_slots=0,
                           max_swaps=swaps)
        twin = _recent_twin(cache, swaps)
        profile = [5, 30, 31, 60, 7, 5, 900]
        self.assertEqual([k for k, _ in cache.preload(profile)], twin.preload(profile))
        tokens, top = 3, 4
        ids = [torch.full((tokens, top), -1, dtype=torch.int64, device=device) for _ in range(layers)]
        view = _View(torch, device, 4)

        def one_round():
            cache.start_round(view)
            tables = [cache.plan(layer, ids[layer]).clone() for layer in range(layers)]
            cache.count_round(view)
            return tables

        graph = tables = None
        if captured:
            one_round()   # the shapes once outside capture: a round without routes
            twin.start_round()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    tables = one_round()
        rng = random.Random(21)
        hot = [rng.sample(range(experts), 8) for _ in range(layers)]
        for r in range(rounds):
            if r == rounds // 2:   # the routing moves on
                hot = [rng.sample(range(experts), 8) for _ in range(layers)]
            routes = [[[rng.choice(hot[layer]) if rng.random() < 0.8 else rng.choice([-1, rng.randrange(experts)])
                        for _ in range(top)] for _ in range(tokens)] for layer in range(layers)]
            for layer in range(layers):
                ids[layer].copy_(torch.tensor(routes[layer], device=device))
            if graph is not None:
                graph.replay()
            else:
                tables = one_round()
            if device == "cuda":
                torch.cuda.synchronize()
            twin.start_round()
            for layer in range(layers):
                flat = [e for row in routes[layer] for e in row if e >= 0]
                distinct = list(dict.fromkeys(flat))
                hits = twin.plan([layer * experts + e for e in distinct])
                expected = [-1] * experts
                for e, hit in zip(distinct, hits):
                    expected[e] = int(twin.key_slot[layer * experts + e]) if hit else -2
                self.assertEqual(tables[layer].tolist(), expected, (r, layer))
                self.check_slots(cache, store, layer, expected)   # a hit's slot holds its bundle: its copy landed
            count = int(cache._recent.state[3])
            self.assertEqual([tuple(job) for job in cache._recent.pending[:count].tolist()], twin.pending, r)
        seen = cache.telemetry(sync=True)
        self.assertEqual((seen["reads"], seen["hits"], seen["fills"]), (twin.reads, twin.hits, twin.copies))
        self.assertEqual(seen["cpu"], twin.reads - twin.hits)
        self.assertGreater(twin.copies, swaps)
        self.assertGreater(twin.hits, twin.reads // 3)

    def test_a_link_layers_misses_are_admitted_now_from_the_rounds_reserve(self):
        # Beside CPU-tier layers (misses -2, copied beside the round) a link layer, whose experts the CPU tier does not
        # compute, gets a slot for every miss in the same plan, its bundle copied before the table is read, from the
        # round's victims above the reserve's floor; the CPU tier's copies take those below. Kernel and torch ops alike,
        # plan for plan.
        torch, device = self.torch, self.device()
        layers, experts, slots, swaps, reserve = 3, 24, 30, 4, 8
        found = {}
        for native in self.kernels():
            store = self.store(layers, experts, bundle=1024)
            cache = self.cache(store, layers, experts, slots, native=native, policy="recent", cpu_tier=True,
                               ring_slots=0, max_swaps=swaps, link=((2,), reserve))
            self.assertEqual(cache._recent.swaps, swaps + reserve)
            rng = random.Random(9)
            seen = []
            for r in range(30):
                view = _View(torch, device, 4)
                cache.start_round(view)
                tables = []
                for layer in range(layers):
                    routes = [[rng.randrange(experts) for _ in range(2)] for _ in range(3)]
                    tables.append(cache.plan(layer, torch.tensor(routes, device=device)).clone())
                cache.count_round(view)
                if device == "cuda":
                    torch.cuda.synchronize()
                link = tables[2].tolist()
                self.assertNotIn(-2, link, r)                  # every routed expert of the link layer has a slot
                self.check_slots(cache, store, 2, link)        # ... holding its bundle when the plan returns
                self.assertLessEqual(int(cache._recent.state[3]), swaps)   # the CPU tier's copies stay below the floor
                seen.append([t.tolist() for t in tables])
            found[native] = seen
            telemetry = cache.telemetry(sync=True)
            self.assertGreater(telemetry["fills"], 0)
            self.assertGreater(telemetry["cpu"], 0)            # the other layers' misses went to the CPU tier
        if len(found) == 2:
            self.assertEqual(found[True], found[False])        # the kernel's rule is the torch ops'
        # A ring of a layer's 8 experts beside a reserve of 8 takes 16 slots: 12 are refused, as is a decode-only
        # cache whose 6 slots hold fewer than the reserve.
        with self.assertRaisesRegex(ValueError, "holds no prompt ring.*link layers' misses may take"):
            self.cache(self.store(layers, 8), layers, 8, 12, policy="recent", cpu_tier=True, max_swaps=swaps,
                       link=((2,), 8))
        with self.assertRaisesRegex(ValueError, "link layers' misses take slots"):
            self.cache(self.store(layers, 8), layers, 8, 6, policy="recent", cpu_tier=True, max_swaps=swaps,
                       ring_slots=0, link=((2,), 8))

    def test_a_link_plan_whose_reserve_the_rounds_hits_took_goes_below_it(self):
        # A step decodes the decoder's layers before the draft layer in one round: their hits stamp the slots they
        # read, some of them among the reserve's victims (the list's last), which the link layer's plan then cannot
        # take. It goes on below the reserve, down to the CPU tier's next victim, and loses no route while the list
        # has open victims (a round's CPU-tier copies and hits take at most the decoder's distinct experts). The
        # reserve alone lost routes here. Kernel and torch ops alike, plan for plan.
        torch, device = self.torch, self.device()
        layers, experts, slots, swaps, reserve, tokens, top = 2, 16, 16, 8, 6, 3, 2
        found = {}
        for native in self.kernels():
            store = self.store(layers, experts, bundle=1024)
            cache = self.cache(store, layers, experts, slots, native=native, policy="recent", cpu_tier=True,
                               ring_slots=0, max_swaps=swaps, link=((1,), reserve))
            rng = random.Random(4)
            seen, below = [], 0
            for r in range(80):
                view = _View(torch, device, 4)
                cache.start_round(view)
                floor = int(cache._recent.state[2]) - reserve
                tables = []
                for layer in range(layers):   # the decoder's layer first, the link layer last
                    routes = [rng.sample(range(experts), top) for _ in range(tokens)]
                    tables.append(cache.plan(layer, torch.tensor(routes, device=device)).clone())
                below += int(cache._recent.state[4]) < floor
                cache.count_round(view)
                if device == "cuda":
                    torch.cuda.synchronize()
                link = tables[1].tolist()
                self.assertNotIn(-2, link, r)
                self.check_slots(cache, store, 1, link)
                seen.append([t.tolist() for t in tables])
            self.assertGreater(below, 0)                       # the reserve alone was short in some rounds
            self.assertEqual(cache.telemetry(sync=True)["failure"], 0)
            found[native] = seen
        if len(found) == 2:
            self.assertEqual(found[True], found[False])

    def test_recent_residency_lends_a_prompt_its_ring_and_takes_it_back(self):
        torch, device = self.torch, self.device()
        layers, experts, slots, swaps = 2, 8, 20, 8   # 16 experts over 4 slots beside the ring's 16
        store = self.store(layers, experts, bundle=1024)
        cache = self.cache(store, layers, experts, slots, policy="recent", cpu_tier=True, max_swaps=swaps)
        self.assertEqual(len(cache.ring_slots), 2 * experts)
        twin = _recent_twin(cache, swaps)
        rng = random.Random(4)

        def decode(rounds):
            for _ in range(rounds):
                cache.start_round(None)
                twin.start_round()
                for layer in range(layers):
                    routed = rng.sample(range(experts), 5)
                    table = cache.plan(layer, torch.tensor(routed, device=device)).tolist()
                    hits = twin.plan([layer * experts + e for e in routed])
                    for e, hit in zip(routed, hits):
                        self.assertEqual(table[e], int(twin.key_slot[layer * experts + e]) if hit else -2)
                    self.check_slots(cache, store, layer, table)
                cache._recent.join()
        decode(6)   # the cache fills every slot it has, the ring's among them
        self.assertTrue(any(cache.directory.slot_key[s].item() >= 0 for s in cache.ring_slots))
        for layer in range(layers):   # a prompt chunk routing every expert: staged through the ring, nothing admitted
            ids = torch.arange(experts, device=device).repeat(4, 1)
            held, table = cache.prompt(layer, ids)
            if device == "cuda":
                torch.cuda.synchronize()
            for e, slot in enumerate(table.tolist()):
                self.assertTrue(torch.equal(held[slot].cpu(), store.bundle(layer, e)))
        twin.close_slots(cache.ring_slots)
        cache.end_prompt()
        twin.open_slots(cache.ring_slots)
        decode(6)   # the ring's slots serve experts again
        self.assertTrue(any(cache.directory.slot_key[s].item() >= 0 for s in cache.ring_slots))
        before = sum(int(k >= 0) for k in cache.directory.slot_key[:cache.total].tolist())
        for layer in range(layers):   # a short chunk borrows as many ring slots as it has routes, not the ring
            held, table = cache.prompt(layer, torch.tensor([[0, 3, 5]], device=device))
            if device == "cuda":
                torch.cuda.synchronize()
            for e in (0, 3, 5):
                self.assertTrue(torch.equal(held[table[e].item()].cpu(), store.bundle(layer, e)))
        self.assertEqual(cache._ring_now, cache.ring_slots[:3])
        self.assertGreaterEqual(sum(int(k >= 0) for k in cache.directory.slot_key[:cache.total].tolist()), before - 3)
        twin.close_slots(cache.ring_slots[:3])
        cache.end_prompt()
        twin.open_slots(cache.ring_slots[:3])
        decode(4)
        self.assertEqual(sorted(cache.directory.key_slot[:layers * experts].tolist()),
                         sorted(int(v) if v >= 0 else -1 for v in twin.key_slot))

    def test_a_pinned_core_keeps_its_slots_and_its_bundles_leave_host_memory(self):
        # Experts in one place: the store leaves the core's bundles out of host memory (a store on a file reads them
        # from it again), the cache holds them in slots no plan, preload, prompt or shrink takes.
        import tempfile
        import expert_store
        torch, device = self.torch, self.device()
        layers, experts, B, swaps = 2, 16, 1 << 16, 6    # bundles of 16 pages
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        store = expert_store.BundleStore({"bundle_bytes": B}, layers, experts, device,
                                         path=Path(folder.name) / "experts.bin", identity={"test": 1})
        self.addCleanup(store.close)
        g = torch.Generator().manual_seed(5)
        bundles = torch.randint(1, 256, (layers * experts, B), dtype=torch.uint8, generator=g)
        for key in range(layers * experts):
            store.put(key // experts, key % experts, bundles[key])
        core = [3, 17, 4, 30, 9, 21]
        self.assertEqual(store.leave(core), frozenset(core))
        from crit_plugins.memory import of
        elastic = device == "cuda" and bool(of(device).elastic_ranges)
        cache = self.cache(store, layers, experts, 28, policy="recent", cpu_tier=True, max_swaps=swaps,
                           ring_slots=experts, **({"max_slots": 40} if elastic else {}))
        host, _ = store.region()
        self.assertEqual(store.left_bytes, len(core) * B)
        for key in range(layers * experts):   # the host holds every other bundle, and the filler where the core was
            held = host.view(layers * experts, B)[key]
            self.assertEqual(torch.equal(held, bundles[key]), key not in core)
            self.assertEqual(int(held.max()) == 0, key in core)
            self.assertTrue(torch.equal(store.bundle(key // experts, key % experts), bundles[key]))
        with self.assertRaisesRegex(ValueError, "pin before any preload|empty"):
            cache.pin(range(layers * experts))
        pairs = cache.pin(core)
        self.assertEqual([k for k, _ in pairs], core)
        self.assertFalse({s for _, s in pairs} & (set(cache.ring_slots) | {cache.scratch}))
        with self.assertRaisesRegex(ValueError, "has its pinned core already"):
            cache.pin([1])

        def intact():
            if device == "cuda":
                torch.cuda.synchronize()
            self.assertTrue(cache.pinned_intact())
            for key, slot in pairs:
                self.assertEqual(int(cache.directory.key_slot[key]), slot)
                self.assertTrue(torch.equal(cache.store[slot].cpu(), bundles[key]))
        intact()
        loaded = cache.preload([5, 3, 6, 17, 40])    # a profile beside the core: its keys take other slots
        self.assertEqual([k for k, _ in loaded], [5, 6])
        self.assertFalse({s for _, s in loaded} & set(cache.pinned))
        intact()
        rng = random.Random(8)

        def decode(rounds):
            for _ in range(rounds):
                cache.start_round(None)
                for layer in range(layers):
                    routed = rng.sample(range(experts), 9)
                    table = cache.plan(layer, torch.tensor(routed, device=device)).tolist()
                    for e in routed:
                        if layer * experts + e in core:
                            self.assertGreaterEqual(table[e], 0)    # an expert of the core never misses
                    self.check_slots(cache, store, layer, table)
                cache._recent.join()
        decode(30)    # every other slot turns over many times
        intact()
        for layer in range(layers):   # a prompt chunk routing every expert: the core is read in place
            held, table = cache.prompt(layer, torch.arange(experts, device=device).repeat(4, 1))
            if device == "cuda":
                torch.cuda.synchronize()
            for e, slot in enumerate(table.tolist()):
                self.assertTrue(torch.equal(held[slot].cpu(), bundles[layer * experts + e]))
                if layer * experts + e in core:
                    self.assertEqual(slot, dict(pairs)[layer * experts + e])
        cache.end_prompt()
        decode(6)
        intact()
        if elastic:   # a shrink never reaches the core, and compaction leaves it where it is
            self.assertEqual(cache.grow(40), 40)
            decode(10)
            kept = cache.shrink(1)
            self.assertGreater(kept, max(cache.pinned) - 1)
            self.assertLess(kept, 40)
            intact()
            decode(6)
            intact()
        self.assertEqual(cache.telemetry(sync=True)["pinned"], len(core))
        cache.close()

    def test_side_stream_swaps_land_before_a_plan_reads_them_in_captured_rounds(self):
        torch, device = self.torch, self.device()
        if device != "cuda":
            self.skipTest("a side stream is CUDA's")
        layers, experts, slots = 2, 16, 6
        store = self.store(layers, experts, bundle=4096)
        cache = self.cache(store, layers, experts, slots, policy="ranked", cpu_tier=True, period=2, ring_slots=0,
                           max_swaps=3, burst_swaps=4)
        side = self.cache(store, layers, experts, slots, policy="ranked", cpu_tier=True, period=2, ring_slots=0,
                          max_swaps=3, burst_swaps=4, side_swaps=True)
        ids = [torch.zeros(3, dtype=torch.int64, device=device) for _ in range(layers)]
        view = _View(torch, device, 4)

        def one_round(c):
            c.start_round(view)
            tables = [c.plan(layer, ids[layer]).clone() for layer in range(layers)]
            c.count_round(view)
            return tables

        graphs = {}
        for c in (cache, side):
            one_round(c)   # the shapes once outside capture
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=stream):
                    tables = one_round(c)
            graphs[id(c)] = (g, tables)
        rng = random.Random(9)
        resident = {}
        for r in range(16):
            hot = [[3, 9, rng.randrange(experts)], [1, 4, rng.randrange(experts)]] if r < 8 else                 [[11, 12, rng.randrange(experts)], [13, 14, rng.randrange(experts)]]
            for layer in range(layers):
                ids[layer].copy_(torch.tensor(hot[layer], device=device))
            for c in (cache, side):
                g, tables = graphs[id(c)]
                g.replay()
                torch.cuda.synchronize()
                for layer in range(layers):
                    self.check_slots(c, store, layer, tables[layer].tolist())   # every slot a plan names holds its bundle
                resident[id(c)] = sum(int(x >= 0) for x in c.directory.key_slot[:layers * experts].tolist())
        self.assertGreater(int(side._ranked.swapped.item()), 0)
        self.assertEqual(resident[id(side)], resident[id(cache)])   # the same residency, one round later at most

    def test_ranked_residency_swaps_inside_captured_rounds_by_its_policy(self):
        torch, device = self.torch, self.device()
        layers, experts, slots = 2, 16, 6
        store = self.store(layers, experts)
        cache = self.cache(store, layers, experts, slots, policy="ranked", cpu_tier=True, period=2, ring_slots=0,
                           max_swaps=3, burst_swaps=4)
        policy = ranked_residency.RankedPolicy(layers * experts, slots, ranked_residency.RankedOptions(
            decay=0.85, period=2, max_swaps=3, burst_swaps=4))
        host_slot_key = [None] * slots
        counts = [0] * (layers * experts)
        ids = [torch.zeros(3, dtype=torch.int64, device=device) for _ in range(layers)]
        rng = random.Random(5)

        def one_round():
            cache.start_round(None)
            for layer in range(layers):
                cache.plan(layer, ids[layer])

        graph = None
        if device == "cuda":
            one_round()   # the shapes once outside capture
            completed = 1
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    one_round()
        else:
            completed = 0
        for layer in range(layers):   # the warm-up round's routes (zeros)
            if graph is not None:
                counts[layer * experts] += 1
        for r in range(12):
            if completed > 0 and completed % 2 == 0:   # the host policy at the same boundary, applied at once
                for i, key in policy.decide(counts, host_slot_key):
                    host_slot_key[i] = key
            hot = [[3, 9, rng.randrange(experts)], [1, 4, rng.randrange(experts)]] if r < 6 else \
                [[11, 12, rng.randrange(experts)], [13, 14, rng.randrange(experts)]]
            for layer in range(layers):
                ids[layer].copy_(torch.tensor(hot[layer]))
                for e in set(hot[layer]):
                    counts[layer * experts + e] += 1
            if graph is not None:
                graph.replay()
            else:
                one_round()
            completed += 1
            slot_key = cache.directory.snapshot()[1][:slots]
            self.assertEqual(slot_key, [-1 if k is None else k for k in host_slot_key], f"round {r}")
        self.assertGreater(cache.telemetry(sync=True)["swaps"], 0)
        key_slot = cache.directory.snapshot()[0]
        for layer in range(layers):
            for e in range(experts):
                slot = key_slot[layer * experts + e]
                if slot >= 0:
                    self.assertTrue(torch.equal(cache.store[slot].cpu(), store.bundle(layer, e)))


class LendingTests(PluginTestCase):
    def run_lending(self, B):
        torch = __import__("torch")
        device = self.device()
        from crit_plugins import StepBatch
        from crit_plugins.device import PREFILL
        import expert_store
        layers, experts = 2, 8
        store = expert_store.BundleStore({"bundle_bytes": B}, layers, experts, device)
        self.addCleanup(store.close)
        g = torch.Generator().manual_seed(0)
        for l in range(layers):
            for e in range(experts):
                store.put(l, e, torch.randint(0, 256, (B,), dtype=torch.uint8, generator=g))
        kv = self.assemble(use={"kv_cache": "attention-state"},
                           use_options={"attention-state": {"page": 16}})["kv_cache"]
        own = 11
        pool = kv.pool([0, 1], 1, 64, 1024, device, "bf16", spare=(own + 1) * B)   # 2 KB pages per layer and K/V
        plugin = self.assemble(options={"ring_slots": 0})["expert_residency"]
        cache = plugin.cache(layers, experts, {"bundle_bytes": B}, own, device, store, lend=pool)
        base = cache.store.data_ptr() - pool.memory.data_ptr()
        self.assertEqual(cache.own[0] * B + base, pool.spare[0])      # the own slots start at the spare bytes
        lent = cache.telemetry()["lent"]
        self.assertGreater(lent, 0)
        for l in range(layers):   # fill every slot, lent ones too
            cache.plan(l, torch.arange(experts, device=device))
        pool.open("a")
        keys = torch.randn(400, 1, 64, generator=g).to(device)
        batch = StepBatch.build([("a", 0, 400, PREFILL)], device)
        pool.write(0, pool.reserve(batch), keys, keys)
        self.assertLess(cache.telemetry()["lent"], lent)
        for begin, page_bytes, pages in pool.regions:
            for s in range(cache.total):
                key = cache.directory.slot_key[s].item()
                if key >= 0 and base + s * B < begin + pool.floor * page_bytes and base + (s + 1) * B > begin:
                    self.fail(f"slot {s} holds an expert over pages in use")
        for l in range(layers):
            table = cache.plan(l, torch.arange(experts, device=device)).tolist()
            for e, slot in enumerate(table):
                self.assertTrue(torch.equal(cache.store[slot].cpu(), store.bundle(l, e)))
        self.assertTrue(torch.equal(pool.gather(0, "a", 400)[0].float().cpu(), keys.bfloat16().float().cpu()))
        pool.close("a")
        self.assertEqual(pool.floor, 0)
        self.assertEqual(cache.telemetry()["lent"], lent)

    def test_a_pools_unused_pages_are_slots_until_a_sequence_needs_them(self):
        for B in (4096, 4096 + 16 * 7):   # a bundle that divides the pages' bytes, and one that does not
            with self.subTest(bundle=B):
                self.run_lending(B)
