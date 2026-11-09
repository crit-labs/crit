"""heterogeneous-expert-dispatch's device form on CUDA: any split of a layer's routes between the device part and the CPU
tier (slot -2) gives the all-device bits, the CPU routes posted and joined inside the stream also in a captured graph
replayed with new routes and splits, and every window answered (served or skipped, never failed)."""
import random
import unittest

from crit_plugins.testing import PluginTestCase

try:
    import torch
except ImportError:
    torch = None

USE = {"expert_source": "expert-tensors", "expert_residency": "device-expert-cache"}
HIDDEN, INTER, EXPERTS, LAYERS = 256, 128, 8, 2


def stored(generator, format):
    def matrix(rows, cols, input_scale):
        return {"weight": torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator),
                "scale": torch.randint(0x28, 0x48, (rows, cols // 16), dtype=torch.uint8, generator=generator),
                "global_scale": torch.rand(rows, generator=generator) * 100 + 50, "input_scale": torch.tensor(input_scale)}
    return matrix(2 * INTER, HIDDEN, 10.0), matrix(HIDDEN, INTER, 2.5)


def bits(t):
    return t.contiguous().view(torch.int32)


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "the device form runs on CUDA")
class DispatchDeviceTests(PluginTestCase):
    def dispatch(self, resident=-1, format="nvfp4", seed=4, native="auto"):
        parts = self.assemble(USE, {"resident": resident, "workers": 3, "pin": False, "max_tokens": 16,
                                    "native": native})
        experts, generator = parts["experts"], torch.Generator().manual_seed(seed)
        for layer in range(LAYERS):
            for e in range(EXPERTS):
                experts.load(layer, e, *stored(generator, format), format)
        return experts

    def routes(self, rng, tokens=4, top=4):
        ids = torch.tensor([rng.sample(range(EXPERTS), top) for _ in range(tokens)], dtype=torch.int64)
        ids[rng.randrange(tokens), rng.randrange(top)] = -1
        weights = torch.tensor([[rng.random() for _ in range(top)] for _ in range(tokens)], dtype=torch.float32)
        x = torch.tensor([[rng.gauss(0.0, 1.0) for _ in range(HIDDEN)] for _ in range(tokens)]).bfloat16()
        return x.cuda(), ids.cuda(), weights.cuda()

    def test_it_has_the_device_form(self):
        self.assertImplements(self.assemble(USE))

    def test_any_split_gives_the_all_device_bits(self):
        for format in ("nvfp4", "int4"):
            with self.subTest(format=format):
                rng = random.Random(1)
                everything, split = self.dispatch(-1, format), self.dispatch(3, format)
                for layer in range(LAYERS):
                    x, ids, weights = self.routes(rng)
                    want = everything.forward(layer, x, ids, weights)
                    self.assertTrue(torch.equal(bits(split.forward(layer, x, ids, weights)), bits(want)))
                    store, slots = everything.stores[layer]
                    for _ in range(6):   # random admission quotas: any expert may be sent to the CPU
                        table = slots.clone()
                        table[torch.tensor([e for e in range(EXPERTS) if rng.random() < 0.5], dtype=torch.long)] = -2
                        got = everything.forward_bundles(layer, x, ids, weights, store, table)
                        self.assertTrue(torch.equal(bits(got), bits(want)))
                torch.cuda.synchronize()
                status = everything.engine.status()
                if status["in_stream"]:
                    self.assertFalse(status["failed"], status["error"])
                    self.assertGreater(status["served"], 0)
                    self.assertEqual(status["overflow"], [0, 0])

    def test_a_token_alone_gives_its_bits_beside_other_rows(self):
        experts = self.dispatch(3)
        x, ids, weights = self.routes(random.Random(5), tokens=6)
        batched = experts.forward(0, x, ids, weights)
        for t in range(6):
            alone = experts.forward(0, x[t:t + 1].clone(), ids[t:t + 1].clone(), weights[t:t + 1].clone())
            self.assertTrue(torch.equal(bits(alone[0]), bits(batched[t])), f"token {t}")

    def test_cpu_routes_join_inside_a_captured_graph(self):
        experts = self.dispatch(-1)
        rng = random.Random(7)
        x, ids, weights = self.routes(rng)
        experts.forward(0, x, ids, weights)   # the engine, its stores and its buffers at this shape, outside capture
        store, slots = experts.stores[0]
        table = slots.clone()
        experts.forward_bundles(0, x, ids, weights, store, table)
        if not experts.engine.in_stream:
            self.skipTest("the CPU routes' mailbox did not build here")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = experts.forward_bundles(0, x, ids, weights, store, table)
        served = experts.engine.box.served()
        for replay in range(8):
            nx, nids, nw = self.routes(rng)
            x.copy_(nx)
            ids.copy_(nids)
            weights.copy_(nw)
            table.copy_(slots)
            table[torch.tensor([e for e in range(EXPERTS) if rng.random() < 0.6], dtype=torch.long)] = -2
            graph.replay()
            torch.cuda.synchronize()
            want = experts.forward_bundles(0, x.clone(), ids.clone(), weights.clone(), store, slots)
            self.assertTrue(torch.equal(bits(out), bits(want)), f"replay {replay}")
        torch.cuda.synchronize()
        self.assertGreater(experts.engine.box.served(), served)
        self.assertFalse(experts.engine.status()["failed"])

    def test_a_caches_host_store_read_in_place_gives_the_loaded_bits(self):
        loaded = self.dispatch(-1)
        rng = random.Random(3)
        x, ids, weights = self.routes(rng)
        want = loaded.forward(1, x, ids, weights)
        order = sorted(loaded.loaded)
        region = torch.stack([loaded.loaded[k] for k in order]).pin_memory()
        size = region.shape[1]
        mirror = torch.empty(region.numel(), dtype=torch.uint8, device="cuda")   # stands for its device address range

        class Source:   # an expert cache's bundle store: bundle_bytes, region() and address() (experts:store)
            bundle_bytes = size

            def region(self):
                return region, mirror

            def address(self, layer, expert):
                return mirror.data_ptr() + order.index((layer, expert)) * size

        bound = self.assemble(USE, {"workers": 2, "pin": False, "max_tokens": 16})["experts"]
        bound.bind_source(Source(), range(LAYERS), EXPERTS, "nvfp4", HIDDEN, INTER)
        store, slots = loaded.stores[1]
        table = slots.clone()
        table[:5] = -2
        self.assertTrue(torch.equal(bits(bound.forward_bundles(1, x, ids, weights, store, table)), bits(want)))

    def test_the_cpu_share_is_measured_here(self):
        experts = self.dispatch(-1)
        table = experts.quota_table(tokens=1, misses=(1, 2), repeats=2)
        self.assertEqual(len(table), 33)
        self.assertTrue(all(-1 <= k <= m for m, k in enumerate(table)))
        self.assertIn(table[1], (0, 1))

    def test_a_post_from_the_caches_decided_hook_is_joined_by_the_forward(self):
        experts = self.dispatch(-1)
        rng = random.Random(11)
        x, ids, weights = self.routes(rng)
        want = experts.forward(0, x, ids, weights)
        store, slots = experts.stores[0]
        table = slots.clone()
        table[:4] = -2
        posts = experts.engine.status()["jobs"]
        decided = experts.decider(0, x, ids)
        decided(table)                          # the cache's plan calls it before its fills
        got = experts.forward_bundles(0, x, ids, weights, store, table)
        self.assertEqual(experts.engine.status()["jobs"], posts + 1)   # posted once, joined by the forward
        self.assertTrue(torch.equal(bits(got), bits(want)))

    def test_a_quota_tensor_is_made_once_per_token_count(self):
        experts = self.dispatch(-1)
        one = experts.quota_tensor(1, misses=(1, 2))
        self.assertEqual((one.dtype, one.numel(), one.device.type), (torch.int32, 33, "cuda"))
        self.assertIs(experts.quota_tensor(1, misses=(1, 2)), one)
        self.assertEqual(one.tolist(), experts.quota_table(1, misses=(1, 2)))

    def test_its_facts_say_what_serves_when_it_loads(self):
        facts = self.assemble(USE)["experts"].facts()
        self.assertIn(facts["cpu_join"], ("device_poll", "stream_wait"))
        self.assertTrue(facts["cpu_routes_capture"])
        self.assertTrue(facts["cpu_expert_rows"].startswith("native"))
        self.assertIn("provisional", facts["cpu_quota"])

    def test_fp32_or_bf16_x_give_the_same_bits(self):
        experts = self.dispatch(3)
        x, ids, weights = self.routes(random.Random(2))
        xf = x.float() + 1e-3 * torch.randn(x.shape, device="cuda")   # FP32 values that are no BF16
        a = experts.forward(0, xf, ids, weights)
        b = experts.forward(0, xf.bfloat16(), ids, weights)
        self.assertTrue(torch.equal(bits(a), bits(b)))

    def test_a_prompt_chunk_past_max_tokens_gives_the_all_device_bits(self):
        everything = self.dispatch(-1)   # max_tokens 16
        x, ids, weights = self.routes(random.Random(8), tokens=37)
        want = everything.forward(0, x, ids, weights)   # no -2: the mailbox is skipped
        jobs = everything.engine.status()["jobs"]
        store, slots = everything.stores[0]
        table = slots.clone()
        table[:3] = -2
        got = everything.forward_bundles(0, x, ids, weights, store, table)   # in pieces of 16, 16 and 5 tokens
        self.assertTrue(torch.equal(bits(got), bits(want)))
        self.assertEqual(everything.engine.status()["jobs"], jobs + 3)

    def test_a_slot_table_of_another_length_is_refused(self):
        experts = self.dispatch(-1)
        x, ids, weights = self.routes(random.Random(3))
        experts.forward(0, x, ids, weights)
        store, slots = experts.stores[0]
        longer = torch.full((2 * EXPERTS,), -2, dtype=torch.int32, device="cuda")
        longer[:EXPERTS] = slots
        with self.assertRaisesRegex(ValueError, "one entry per expert"):
            experts.forward_bundles(0, x, ids, weights, store, longer)

    def test_a_cpu_tier_failure_raises_at_the_next_call_and_shows_in_telemetry(self):
        loaded = self.dispatch(-1)
        order = sorted(loaded.loaded)
        region = torch.stack([loaded.loaded[k] for k in order]).pin_memory()
        size = region.shape[1]
        mirror = torch.empty(region.numel(), dtype=torch.uint8, device="cuda")

        class Missing:   # a store that holds every expert but expert 5 of layer 0
            bundle_bytes = size

            def region(self):
                return region, mirror

            def address(self, layer, expert):
                return None if (layer, expert) == (0, 5) else mirror.data_ptr() + order.index((layer, expert)) * size

        bound = self.assemble(USE, {"workers": 2, "pin": False, "max_tokens": 16})["experts"]
        bound.bind_source(Missing(), range(LAYERS), EXPERTS, "nvfp4", HIDDEN, INTER)
        x, ids, weights = self.routes(random.Random(5))
        loaded.forward(0, x, ids, weights)
        store, slots = loaded.stores[0]
        ids[0, 0] = 5
        table = slots.clone()
        table[5] = -2   # the cache sends expert 5 to the CPU tier, which does not hold it
        bound.forward_bundles(0, x, ids, weights, store, table)
        torch.cuda.synchronize()
        status = bound.telemetry()["cpu_routes"]
        self.assertTrue(status["failed"])
        self.assertIn("expert id out of range", status["error"])
        with self.assertRaisesRegex(RuntimeError, "CPU tier failed"):
            bound.forward_bundles(0, x, ids, weights, store, table)

    def test_a_capture_without_the_in_stream_join_raises(self):
        experts = self.dispatch(-1, native="off")
        self.assertFalse(experts.facts()["cpu_routes_capture"])
        x, ids, weights = self.routes(random.Random(4))
        experts.forward(0, x, ids, weights)
        store, slots = experts.stores[0]
        table = slots.clone()
        table[:2] = -2
        want = experts.forward_bundles(0, x, ids, weights, store, table)   # the host path: the same bits
        self.assertTrue(torch.equal(bits(want), bits(self.dispatch(-1).forward(0, x, ids, weights))))
        graph = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(RuntimeError, "in-stream mailbox"):
            with torch.cuda.graph(graph):
                experts.forward_bundles(0, x, ids, weights, store, table)

    def test_a_store_of_another_encoding_or_outside_its_region_is_refused(self):
        experts = self.assemble(USE)["experts"]

        class Other:
            bundle_bytes = 1234

        with self.assertRaisesRegex(ValueError, "another encoding"):
            experts.bind_source(Other(), range(1), 1, "nvfp4", HIDDEN, INTER)
        size = experts.layout("nvfp4", HIDDEN, INTER)["bundle_bytes"]
        region = torch.zeros(size, dtype=torch.uint8)
        mirror = torch.empty(size, dtype=torch.uint8, device="cuda")

        class Short:
            bundle_bytes = size

            def region(self):
                return region, mirror

            def address(self, layer, expert):
                return mirror.data_ptr() + size   # past the region's end

        experts.bind_source(Short(), range(1), 1, "nvfp4", HIDDEN, INTER)
        x, ids, weights = self.routes(random.Random(6), top=1)
        with self.assertRaisesRegex(ValueError, "outside the store's region"):
            experts.forward_bundles(0, x, ids.clamp(max=0), weights, mirror.view(1, -1), torch.full((1,), -2, dtype=torch.int32,
                                                                                                    device="cuda"))

    def test_the_device_counts_the_jobs_a_replayed_graph_posts(self):
        experts = self.dispatch(-1)
        x, ids, weights = self.routes(random.Random(9))
        experts.forward(0, x, ids, weights)
        store, slots = experts.stores[0]
        table = slots.clone()
        table[:4] = -2
        experts.forward_bundles(0, x, ids, weights, store, table)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            experts.forward_bundles(0, x, ids, weights, store, table)
        before = experts.engine.status()["jobs"]
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(experts.engine.status()["jobs"], before + 5)

    def test_it_refuses_a_format_without_a_declared_integer_order(self):
        experts = self.assemble(USE)["experts"]
        with self.assertRaisesRegex(ValueError, "nvfp4, int4"):
            experts.load(0, 0, torch.zeros(2 * INTER, HIDDEN), torch.zeros(HIDDEN, INTER), "bf16")
