"""device-expert-cache's device form against its host twin: the directory's plans (CUDA ops and torch ops) give
LayerDirectory's slots, copies, counters, reference bits and hand bit for bit over random route streams (duplicates,
-1, routes outside the experts, unavailable slots, quotas, second touch, failures); the cache's slots hold their
experts' bundles byte for byte; a plan and its fill capture in a CUDA graph and replay (the torch ops too, with a quota
table and second touch); the prompt ring never hands a layer a segment its copies have not reached and never
overwrites one a layer has not read; ranked residency swaps inside captured rounds by RankedPolicy's decisions; a KV
pool's unused pages serve as slots and are reclaimed before KV lands there, whatever the bundle size; a cache that
could lose a route refuses to be made."""
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


class _View:
    """A round's counters as crit_plugins.rounds' RoundView gives them (counter(name) -> [rows] int64)."""

    def __init__(self, torch, device, rows):
        self.counter_names = ("accepted", "expert_reads", "expert_hits")
        self.counters = torch.zeros(rows, 3, dtype=torch.int64, device=device)

    def counter(self, name):
        return self.counters[:, self.counter_names.index(name)]


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
    def cache(self, store, layers, experts, slots, layout=None, **options):
        plugin = self.assemble(options={"native": options.pop("native", True), **options})["expert_residency"]
        layout = layout or {"bundle_bytes": store.bundle_bytes}
        return plugin.cache(layers, experts, layout, slots, self.device(), store)

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
                    ring_slots = set(cache.ring_slots)
                    before = [-1 if slot in ring_slots else slot for slot in before]   # the borrowed ring empties
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
                    self.assertEqual(cache.directory.snapshot()[0], before)   # nothing admitted, nothing evicted
                    cache.end_prompt()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(sums, expected.expand(4, -1, -1)), (native, ring))
                seen = cache.telemetry(sync=True)
                self.assertGreater(seen["prompt_copies"], 0)
                self.assertEqual(seen["slots"], 80 if ring == "borrow" else 80 - 2 * experts)

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
        batch = StepBatch.build([("a", 0, 400)], device)
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
