"""expert-offload's device form on CUDA: each tier and miss path gives the bits of every expert on the device in the
same arithmetic (tensor where misses go over the link, the CPU's declared order where the CPU computes them), decode
rows and prompt chunks alike, captured rounds included, its facts and telemetry say what it decided, and an elastic
cache gives a prompt step its room and grows back with every route's bits unchanged."""
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

from crit_plugins.testing import PluginTestCase

try:
    import torch
except ImportError:
    torch = None

USE = {"expert_source": "expert-tensors"}
HIDDEN, INTER, EXPERTS, LAYERS = 256, 128, 8, 2


def stored(generator):
    def matrix(rows, cols, input_scale):
        return {"weight": torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator),
                "scale": torch.randint(0x28, 0x48, (rows, cols // 16), dtype=torch.uint8, generator=generator),
                "global_scale": torch.rand(rows, generator=generator) * 100 + 50, "input_scale": torch.tensor(input_scale)}
    return matrix(2 * INTER, HIDDEN, 10.0), matrix(HIDDEN, INTER, 2.5)


def weight_only(matrices):
    """The same stored matrices without their input scales: a W4A16 expert (a draft layer's, say)."""
    return tuple({k: v for k, v in m.items() if k != "input_scale"} for m in matrices)


def bits(t):
    return t.contiguous().view(torch.int32)


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "the device form runs on CUDA")
class OffloadDeviceTests(PluginTestCase):
    def offload(self, count=EXPERTS, **options):
        use, context = dict(options.pop("use_options", {})), options.pop("context", None)
        experts = self.assemble(USE, {"workers": 3, "pin": False, "max_tokens": 16, **options}, use_options=use,
                                context=context)
        experts = experts["experts"]
        self.addCleanup(experts.close)
        experts.begin(LAYERS, count, "nvfp4", HIDDEN, INTER)
        generator = torch.Generator().manual_seed(4)
        for layer in range(LAYERS):
            for e in range(count):
                experts.load(layer, e, *stored(generator), "nvfp4")
        return experts

    def kv_pool(self, tokens):
        """A KV pool of attention-state's whose pages are backed as sequences come to hold them (elastic)."""
        from crit_plugins.assemble import Registry, assemble
        registry = Registry.discover(Path(__file__).resolve().parents[2])
        made = assemble({"use": {"kv_cache": "attention-state"}, "options": {"attention-state": {"elastic": True}}},
                        registry, None, ())
        self.addCleanup(made.close)
        return made["kv_cache"].pool([0, 1], 2, 64, tokens, "cuda", "bf16")

    def routes(self, rng, tokens=4, top=4):
        ids = torch.tensor([rng.sample(range(EXPERTS), top) for _ in range(tokens)], dtype=torch.int64)
        ids[rng.randrange(tokens), rng.randrange(top)] = -1
        weights = torch.tensor([[rng.random() for _ in range(top)] for _ in range(tokens)], dtype=torch.float32)
        x = torch.tensor([[rng.gauss(0.0, 1.0) for _ in range(HIDDEN)] for _ in range(tokens)]).bfloat16()
        return x.cuda(), ids.cuda(), weights.cuda()

    def all_device(self, offload, layer, x, ids, weights, arithmetic):
        """The layer with every expert on the device in one store, through the plugin's own device part."""
        store = torch.stack([offload.store.bundle(layer, e) for e in range(EXPERTS)]).cuda()
        slots = torch.arange(EXPERTS, dtype=torch.int32, device="cuda")
        return offload.part.combine(offload.part.route_outputs(layer, x, ids, store, slots, arithmetic), ids, weights)

    def test_experts_that_fit_stay_on_the_device(self):
        fits = self.offload()
        self.assertEqual(fits.tier, "device")
        x, ids, weights = self.routes(random.Random(1))
        out = fits.forward(0, x, ids, weights)
        self.assertEqual(tuple(out.shape), (4, HIDDEN))
        self.assertEqual(fits.facts()["expert_offload"]["tier"], "device")

    def test_keep_routes_leaves_each_tokens_last_routes_out_in_every_tier(self):
        whole = self.offload()
        cut = self.offload(keep_routes=2)                                     # every expert on the device
        offloaded = self.offload(tier="ram", misses="link", margin_mib=0, keep_routes=2)
        offloaded.open()
        rng = random.Random(12)
        for tokens in (3, 24):                                                # decode rows, then a prompt chunk
            x, ids, weights = self.routes(rng, tokens=tokens)
            want = whole.forward(1, x, ids[:, :2].contiguous(), weights[:, :2].contiguous())
            self.assertTrue(torch.equal(bits(cut.forward(1, x, ids, weights)), bits(want)))
            self.assertTrue(torch.equal(bits(offloaded.forward(1, x, ids, weights)), bits(want)))
            self.assertFalse(torch.equal(bits(whole.forward(1, x, ids, weights)), bits(want)))
            offloaded.end_step(False)
        self.assertEqual(cut.facts()["expert_offload"]["keep_routes"], 2)
        self.assertNotIn("keep_routes", whole.facts()["expert_offload"])

    def test_misses_over_the_link_give_the_all_device_bits(self):
        fits, offloaded = self.offload(), self.offload(tier="ram", misses="link", margin_mib=0)
        offloaded.open()
        self.assertEqual(offloaded.misses, "link")
        rng = random.Random(2)
        for _ in range(6):
            layer = rng.randrange(LAYERS)
            x, ids, weights = self.routes(rng)
            self.assertTrue(torch.equal(bits(offloaded.forward(layer, x, ids, weights)),
                                        bits(fits.forward(layer, x, ids, weights))))
        x, ids, weights = self.routes(rng, tokens=37)   # a prompt chunk: staged through the ring
        self.assertTrue(torch.equal(bits(offloaded.forward(0, x, ids, weights)), bits(fits.forward(0, x, ids, weights))))
        offloaded.end_step()
        line = offloaded.telemetry()["expert_offload"]["diagnosis"]
        self.assertIn("ram tier", line)
        # Every row the kernel's tensor arithmetic: a prompt row has the bits of the same route decoded.
        self.assertEqual(offloaded.exactness(), {"decode": "row-exact", "prompt": "exact"})
        self.assertEqual(fits.exactness(), {"decode": "row-exact", "prompt": "exact"})

    def test_misses_on_the_cpu_give_the_declared_all_device_bits(self):
        offloaded = self.offload(tier="ram", misses="cpu", margin_mib=0)
        offloaded.open()
        if offloaded.misses != "cpu":
            self.skipTest(f"no CPU tier here: {offloaded.reasons.get('misses')}")
        rng = random.Random(3)
        for _ in range(16):   # rounds of every layer: a miss is the CPU's, then its copy serves on the device
            for layer in range(LAYERS):
                x, ids, weights = self.routes(rng)
                want = self.all_device(offloaded, layer, x, ids, weights, "declared")
                self.assertTrue(torch.equal(bits(offloaded.forward(layer, x, ids, weights)), bits(want)))
        facts = offloaded.facts()["expert_offload"]
        self.assertEqual((facts["misses"], facts["decode_arithmetic"], facts["policy"]), ("cpu", "declared", "recent"))
        self.assertEqual(offloaded.exactness(), {"decode": "declared", "prompt": "declared"})
        seen = offloaded.cache.telemetry(sync=True)
        self.assertGreater(seen["hits"], 0)            # experts the link copied in beside a round
        self.assertGreater(seen["cpu"], 0)             # and experts the CPU tier computed
        self.assertGreater(seen["fills"], 0)

    def test_a_cache_too_small_for_link_misses_serves_prompts_with_the_cpu_tier_or_is_refused_at_open(self):
        # Over the link a decode plan keeps a layer's experts beside the prompt ring's: 12 slots for layers of 8
        # experts hold no ring then. misses auto takes the CPU tier, which needs no slots beside the ring, and a
        # prompt chunk runs; with misses pinned to the link the cache is refused at open(), naming what prompts take.
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / "offload.json"
            settings.write_text(json.dumps({"options": {"expert-offload": {"slots": 12}}}), encoding="utf-8")
            auto = self.offload(tier="ram", misses="auto", settings=str(settings), reserve_mib=0, margin_mib=0,
                                elastic=False)
            try:
                auto.open()
            except ValueError:
                if auto.misses != "cpu":
                    self.skipTest(f"no CPU tier here: {auto.reasons.get('misses')}")
                raise
            self.assertEqual(auto.misses, "cpu")
            self.assertIn("hold no prompt ring", auto.reasons["misses_small"])
            self.assertEqual(len(auto.cache.ring_slots), EXPERTS)
            x, ids, weights = self.routes(random.Random(5), tokens=37)
            want = self.all_device(auto, 0, x, ids, weights, "declared")
            self.assertTrue(torch.equal(bits(auto.forward(0, x, ids, weights)), bits(want)))
            auto.end_step()
            link = self.offload(tier="ram", misses="link", settings=str(settings), reserve_mib=0, margin_mib=0,
                                elastic=False)
            with self.assertRaisesRegex(ValueError, "holds no prompt ring.*prompts take 16 slots"):
                link.open()

    def test_a_short_prompt_chunks_misses_go_where_they_cost_less_by_what_open_measured(self):
        # Where the CPU tier computes decode misses and prompt rows keep its order, a prompt chunk of fewer routes
        # than a layer has experts sends its missed routes to the CPU tier when they cost less there than their
        # experts over the link, by the times open() measured (also with the misses pinned to the CPU tier): the
        # bits are the same either way, nothing is admitted, and the CPU's way borrows no ring.
        offloaded = self.offload(tier="ram", misses="cpu", margin_mib=0)
        offloaded.open()
        if offloaded.misses != "cpu":
            self.skipTest(f"no CPU tier here: {offloaded.reasons.get('misses')}")
        self.assertEqual(set(offloaded.miss_times), {"cpu", "link", "compute", "handoff"})
        self.assertAlmostEqual(offloaded.miss_times["cpu"],
                               offloaded.miss_times["compute"] + offloaded.miss_times["handoff"])
        self.assertIn("short", offloaded.facts()["expert_offload"]["prompts"])
        rng, cache = random.Random(9), offloaded.cache

        def chunk(cpu, link):
            offloaded.miss_times = {"cpu": cpu, "link": link}
            before = (cache.prompt_cpu_routes, cache.telemetry(sync=True)["prompt_copies"])
            x, ids, weights = self.routes(rng, tokens=1)   # four routes: fewer than the layer's eight experts
            want = self.all_device(offloaded, 1, x, ids, weights, "declared")
            self.assertTrue(torch.equal(bits(offloaded.forward(1, x, ids, weights, prompt=True)), bits(want)))
            held = len(cache._ring_now)
            offloaded.end_step()
            after = (cache.prompt_cpu_routes, cache.telemetry(sync=True)["prompt_copies"])
            return after[0] - before[0], after[1] - before[1], held
        on_cpu, copies, ring = chunk(1.0, 1000.0)       # the CPU tier far the sooner
        self.assertEqual((on_cpu > 0, copies, ring), (True, 0, 0))
        on_cpu, copies, ring = chunk(1000.0, 1.0)       # the link far the sooner: its experts into the ring
        self.assertEqual((on_cpu, copies > 0, ring > 0), (0, True, True))
        self.assertEqual(cache.telemetry(sync=True)["resident"], 0)   # a prompt admits nothing either way

    def test_prompt_arithmetic_auto_follows_the_level_the_engine_asks_of_its_prompt_rows(self):
        from crit_plugins import kernels
        self.addCleanup(kernels.use_exactness, None)
        for asked, arithmetic, level in ((None, "declared", "declared"), ("repeatable", "tensor", "row-exact"),
                                         ("row-exact", "tensor", "row-exact"), ("exact", "declared", "declared")):
            kernels.use_exactness(asked and {"decode": "row-exact", "prompt": asked})
            offloaded = self.offload(tier="ram", misses="cpu", margin_mib=0)
            offloaded.open()
            if offloaded.misses != "cpu":
                self.skipTest(f"no CPU tier here: {offloaded.reasons.get('misses')}")
            self.assertEqual(offloaded.facts()["expert_offload"]["prompt_arithmetic"], arithmetic, asked)
            self.assertEqual(offloaded.exactness(), {"decode": "declared", "prompt": level}, asked)
            x, ids, weights = self.routes(random.Random(5), tokens=37)   # a prompt chunk in that arithmetic
            want = self.all_device(offloaded, 0, x, ids, weights, arithmetic)
            self.assertTrue(torch.equal(bits(offloaded.forward(0, x, ids, weights)), bits(want)), asked)
            offloaded.end_step()
        kernels.use_exactness("exact")   # an option set to one arithmetic is that arithmetic, whatever is asked
        pinned = self.offload(tier="ram", misses="cpu", margin_mib=0, prompt_arithmetic="tensor")
        pinned.open()
        self.assertEqual(pinned.facts()["expert_offload"]["prompt_arithmetic"], "tensor")
        self.assertEqual(pinned.exactness(), {"decode": "declared", "prompt": "row-exact"})

    def test_a_pinned_core_holds_its_experts_on_the_device_alone_with_every_routes_bits(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            settings, profile = Path(folder) / "offload.json", Path(folder) / "profile.json"
            settings.write_text(json.dumps({"options": {"expert-offload": {"slots": 14}}}), encoding="utf-8")
            profile.write_text(json.dumps({"kind": "expert-offload-routing", "layers": LAYERS, "experts": EXPERTS,
                                           "routes": 1, "keys": [9, 2, 14, 5, 0, 11]}), encoding="utf-8")
            options = dict(tier="ram", misses="cpu", policy="recent", store_dir=folder, settings=str(settings),
                           profile=str(profile), reserve_mib=0, margin_mib=0, elastic=False)
            plain = self.offload(**options)       # the same cache without a core, from the same store file
            plain.open()
            if plain.misses != "cpu":
                self.skipTest(f"no CPU tier here: {plain.reasons.get('misses')}")
            self.assertEqual((plain.core, plain.cache.telemetry(sync=True)["pinned"]), ([], 0))
            plain.close()
            auto = self.offload(**{**options, "misses": "auto", "pinned": 0.3})   # a core needs misses: cpu named
            auto.open()
            self.assertEqual(auto.core, [])
            self.assertIn("misses: cpu", auto.facts()["expert_offload"]["reasons"]["pinned"])
            auto.close()
            pinned = self.offload(**{**options, "pinned": 0.3})
            pinned.open()
            self.assertEqual(pinned.core, [9, 2, 14, 5])                          # 0.3 of the 14 slots
            held = pinned.telemetry()["expert_offload"]
            self.assertEqual((held["expert_cache"]["pinned"], held["host_store"]["left"]), (4, 4))
            size = int(pinned.bundle_layout["bundle_bytes"])
            self.assertGreater(held["host_store"]["left_bytes"], 4 * (size - 2 * 4096) - 1)
            self.assertIn("held on the device alone", pinned.facts()["expert_offload"]["reasons"]["pinned"])
            host, _ = pinned.store.region()
            for key in pinned.core:   # the host's bytes there are the filler's (whole pages inside the bundle)
                first = -(-key * size // 4096) * 4096
                self.assertEqual(int(host[first:(key + 1) * size // 4096 * 4096].max()), 0)
            rng = random.Random(3)
            for tokens in (4, 4, 4, 4, 4, 4, 24, 4, 4, 4):   # decode rounds, a prompt chunk between them
                for layer in range(LAYERS):
                    x, ids, weights = self.routes(rng, tokens=tokens)
                    want = self.all_device(pinned, layer, x, ids, weights, "declared")
                    self.assertTrue(torch.equal(bits(pinned.forward(layer, x, ids, weights)), bits(want)))
                if tokens > 4:
                    pinned.end_step(False)
            torch.cuda.synchronize()
            self.assertTrue(pinned.cache.pinned_intact())
            seen = pinned.cache.telemetry(sync=True)
            self.assertGreater(seen["cpu"], 0)             # the other experts still miss and come in
            self.assertGreater(seen["fills"], 0)
            pinned.close()

    def test_an_elastic_cache_gives_a_prompt_step_its_room_and_grows_back(self):
        from crit_plugins import memory
        device = memory.of("cuda")
        if not device.elastic_ranges:
            self.skipTest("the driver's virtual memory API is not reachable here")
        device.load()   # an engine starts to load: until serve() the cache keeps the size it opened with
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / "offload.json"   # fewer slots than experts, so the cache has room to grow into
            settings.write_text(json.dumps({"options": {"expert-offload": {"slots": 12}}}), encoding="utf-8")
            offloaded = self.offload(tier="ram", misses="cpu", settings=str(settings), reserve_mib=0, margin_mib=0)
            offloaded.open()
            if offloaded.misses != "cpu":
                self.skipTest(f"no CPU tier here: {offloaded.reasons.get('misses')}")
            self.assertTrue(offloaded.facts()["expert_offload"]["elastic"])
            self.assertEqual(offloaded.cache.backed_slots(), 12)
            rng = random.Random(8)

            def decode(rounds):
                for _ in range(rounds):
                    for layer in range(LAYERS):
                        x, ids, weights = self.routes(rng)
                        want = self.all_device(offloaded, layer, x, ids, weights, "declared")
                        self.assertTrue(torch.equal(bits(offloaded.forward(layer, x, ids, weights)), bits(want)))
            decode(3)
            device.serve()   # Core's call after the warm requests: the load is over, the givers take what is left
            self.assertEqual(offloaded.cache.backed_slots(), LAYERS * EXPERTS)   # the free memory holds every expert
            decode(3)
            # A prompt step whose chunk was measured to take more than the device has free: the cache gives slots back
            # before it, serves the chunk through its ring, and grows back after the prompt's last step.
            offloaded._peaks = {24: device.free() + (64 << 20)}
            offloaded.begin_step(24)
            self.assertLess(offloaded.cache.backed_slots(), LAYERS * EXPERTS)
            self.assertGreaterEqual(offloaded.shrinks, 1)
            x, ids, weights = self.routes(rng, tokens=24)
            for layer in range(LAYERS):
                want = self.all_device(offloaded, layer, x, ids, weights, "declared")
                self.assertTrue(torch.equal(bits(offloaded.forward(layer, x, ids, weights)), bits(want)))
            offloaded._peaks = {}                                   # what the step took is measured at its end
            offloaded.end_step(False)
            self.assertEqual(list(offloaded._peaks), [24])
            self.assertEqual(offloaded.cache.backed_slots(), LAYERS * EXPERTS)
            decode(3)                                               # the experts lost come back as they are missed
            # A prompt step of a size seen finds its measured room free: the cache gives nothing back for it, nor
            # for a step without prompt tokens.
            shrinks = offloaded.shrinks
            for tokens in (24, 0):
                offloaded.begin_step(tokens)
                if tokens:
                    x, ids, weights = self.routes(rng, tokens=tokens)
                    for layer in range(LAYERS):
                        want = self.all_device(offloaded, layer, x, ids, weights, "declared")
                        self.assertTrue(torch.equal(bits(offloaded.forward(layer, x, ids, weights)), bits(want)))
                offloaded.end_step(False)
                self.assertEqual((offloaded.shrinks, offloaded.cache.backed_slots()), (shrinks, LAYERS * EXPERTS))
            decode(2)
            seen = offloaded.telemetry()["expert_offload"]["elastic"]
            self.assertEqual((seen["slots"], list(seen["prompt_bytes"])), (LAYERS * EXPERTS, [24]))

    def test_a_kv_pool_takes_its_pages_from_an_elastic_cache_on_a_full_device(self):
        # While the engine serves, an elastic cache holds every byte the device has free but what it keeps. A KV pool
        # sized for a long context backs its pages as a sequence grows: the ledger has the cache give the bytes back
        # (its coldest experts first), the pool gets its pages, and every route keeps its bits.
        from crit_plugins import StepBatch, memory
        device = memory.of("cuda")
        if not device.elastic_ranges:
            self.skipTest("the driver's virtual memory API is not reachable here")
        device.load()
        count = 96   # about ten megabytes of bundles: the cache spans several of the device's granules
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / "offload.json"   # fewer slots than experts at open(): the cache is elastic
            # a decode-only cache (ring_slots 0): two layers over the link hold a prompt ring only with every expert
            settings.write_text(json.dumps({"options": {"expert-offload": {"slots": 3 * count // 2, "ring_slots": 0}}}),
                                encoding="utf-8")
            offloaded = self.offload(count, tier="ram", misses="link", settings=str(settings), reserve_mib=0,
                                     margin_mib=0)
            offloaded.open()
        self.assertTrue(offloaded.facts()["expert_offload"]["elastic"])
        device.serve()
        self.assertEqual(offloaded.cache.backed_slots(), LAYERS * count)
        pool = self.kv_pool(1 << 20)
        before, held = offloaded.cache.backed_slots(), offloaded.cache.elastic.mapped
        # The device counts as full: the cache keeps free exactly what is free now, so every byte the pool backs
        # is one the cache gives back.
        offloaded.options["margin_mib"] = device.free() >> 20
        pool.open("a")
        slots = pool.reserve(StepBatch.build([("a", 0, 32768)], "cuda"))
        self.assertEqual(int(slots.numel()), 32768)
        self.assertGreater(pool.elastic.bytes, 8 << 20)
        self.assertLess(offloaded.cache.backed_slots(), before)
        self.assertLess(offloaded.cache.elastic.mapped, held)
        self.assertGreaterEqual(offloaded.shrinks, 1)
        # Asked for more than it holds, the cache keeps a layer's experts in slots of its own: over the link a miss
        # needs one (a cache shrunk below that lost the route and failed the request).
        self.assertGreaterEqual(offloaded.cache.backed_slots(), count)
        rng = random.Random(21)
        ids = torch.tensor([rng.sample(range(count), 4) for _ in range(4)], dtype=torch.int64).cuda()
        weights = torch.rand(4, 4).cuda()
        x = torch.randn(4, HIDDEN).bfloat16().cuda()
        store = torch.stack([offloaded.store.bundle(1, e) for e in range(count)]).cuda()
        table = torch.arange(count, dtype=torch.int32, device="cuda")
        want = offloaded.part.combine(offloaded.part.route_outputs(1, x, ids, store, table, "tensor"), ids, weights)
        self.assertTrue(torch.equal(bits(offloaded.forward(1, x, ids, weights)), bits(want)))
        pool.close("a")
        offloaded.options["margin_mib"] = 0

    def test_a_prompt_step_that_cannot_reuse_torchs_kept_blocks_keeps_the_margin(self):
        # Blocks torch's allocator keeps from a prompt's earlier steps are room only where a step reuses them: a
        # tensor larger than any kept block (attention's grow with the context) takes new device memory beside them.
        # The cache gives a step the device bytes such steps took, so the margin holds at the step's peak; counting
        # the kept blocks as room in use left 24 MiB less than the margin free here.
        from crit_plugins import memory
        device = memory.of("cuda")
        if not device.elastic_ranges:
            self.skipTest("the driver's virtual memory API is not reachable here")
        device.load()
        count, mib = 2048, 1 << 20
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / "offload.json"   # fewer slots than experts at open(): the cache is elastic
            settings.write_text(json.dumps({"options": {"expert-offload": {"slots": 3 * count // 2, "ring_slots": 0}}}),
                                encoding="utf-8")
            offloaded = self.offload(count, tier="ram", misses="link", settings=str(settings), reserve_mib=0,
                                     margin_mib=0)
            offloaded.open()
        device.serve()
        try:
            offloaded.options["margin_mib"] = device.free(release=True) >> 20   # the device counts as full
            margin = offloaded._margin()
            offloaded.begin_step(24)            # a prompt's first step: 32 MiB of temporaries
            offloaded._prompt_forward(24)
            first = torch.empty(32 * mib, dtype=torch.uint8, device="cuda")
            del first
            offloaded.end_step(True)            # the prompt goes on: torch keeps the block
            self.assertGreaterEqual(device.torch()["unused"], 32 * mib)
            offloaded.begin_step(24)            # its next step: a larger tensor, which the kept block cannot hold
            offloaded._prompt_forward(24)
            larger = torch.empty(56 * mib, dtype=torch.uint8, device="cuda")
            self.assertGreaterEqual(device.free(), margin)
            del larger
            offloaded.end_step(False)
        finally:
            offloaded.options["margin_mib"] = 0
            offloaded.close()

    def test_a_prompt_step_that_outgrows_its_room_keeps_the_margin_at_every_look(self):
        # A step's room is a forecast from the steps before it. A step that takes more (deeper in the context than
        # any before, a plugin's temporaries outside the ledger) is looked at before each routed layer: the cache
        # keeps a reserve free beyond the margin and gives the step's next stride before the step takes it. With the
        # room alone the fifth of these allocations left 10 MiB less than the margin free.
        from crit_plugins import memory
        device = memory.of("cuda")
        if not device.elastic_ranges:
            self.skipTest("the driver's virtual memory API is not reachable here")
        device.load()
        count, mib = 3072, 1 << 20
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / "offload.json"   # fewer slots than experts at open(): the cache is elastic
            settings.write_text(json.dumps({"options": {"expert-offload": {"slots": 3 * count // 2, "ring_slots": 0}}}),
                                encoding="utf-8")
            offloaded = self.offload(count, tier="ram", misses="link", settings=str(settings), reserve_mib=0,
                                     margin_mib=0)
            offloaded.open()
        device.serve()
        try:
            offloaded.options["margin_mib"] = device.free(release=True) >> 20   # the device counts as full
            margin = offloaded._margin()
            offloaded.begin_step(24)            # a prompt step of this size takes 16 MiB: its room from now on
            offloaded._prompt_forward(24)
            first = torch.empty(16 * mib, dtype=torch.uint8, device="cuda")
            del first
            offloaded.end_step(False)
            offloaded.begin_step(24)            # the next takes six times that, 12 MiB a layer
            offloaded._prompt_forward(24)
            held = []
            for layer in range(8):
                held.append(torch.empty(12 * mib, dtype=torch.uint8, device="cuda"))
                self.assertGreaterEqual(device.free(), margin, layer)
                offloaded._prompt_forward(24)   # the next routed layer: a look
            self.assertGreaterEqual(offloaded.extends, 3)
            found = offloaded.telemetry()["expert_offload"]["elastic"]
            self.assertEqual((found["looks"], found["extends"]), (offloaded.looks, offloaded.extends))
            self.assertGreaterEqual(found["reserve_bytes"], 64 * mib)
            del held
            offloaded.end_step(False)
        finally:
            offloaded.options["margin_mib"] = 0
            offloaded.close()

    def test_a_cache_that_keeps_its_size_leaves_elastic_pools_their_room(self):
        # A cache that cannot give memory back (elastic off, no elastic ranges here, or one that holds every expert)
        # must not take what the model's elastic pools back later: open() leaves their unbacked bytes free and says
        # so. A cache with fewer slots than experts is elastic and gives them back as the pools ask.
        from crit_plugins import memory
        if not memory.of("cuda").elastic_ranges:
            self.skipTest("the driver's virtual memory API is not reachable here")
        pool = self.kv_pool(1 << 22)
        reserved = pool.elastic.range.size - pool.elastic.range.mapped
        self.assertGreater(reserved, 1 << 30)
        fixed = self.offload(tier="ram", misses="link", elastic=False, margin_mib=0)
        self.assertEqual(fixed._pools_unbacked(), reserved)
        fixed.open()
        self.assertFalse(fixed.facts()["expert_offload"]["elastic"])
        self.assertIn("left free for the model's pools", fixed.facts()["expert_offload"]["reasons"]["pools"])
        whole = self.offload(tier="ram", misses="link", margin_mib=0)
        whole.open()
        self.assertFalse(whole.facts()["expert_offload"]["elastic"])
        self.assertIn("left free for the model's pools", whole.facts()["expert_offload"]["reasons"]["pools"])
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / "offload.json"
            settings.write_text(json.dumps({"options": {"expert-offload": {"slots": 3 * EXPERTS // 2,
                                                                           "ring_slots": 0}}}),   # decode only
                                encoding="utf-8")
            elastic = self.offload(tier="ram", misses="link", settings=str(settings), margin_mib=0)
            elastic.open()
        self.assertTrue(elastic.facts()["expert_offload"]["elastic"])
        self.assertNotIn("pools", elastic.facts()["expert_offload"]["reasons"])

    def test_a_disk_tier_reads_misses_from_its_file(self):
        with tempfile.TemporaryDirectory() as folder:
            fits = self.offload()
            disk = self.offload(tier="disk", store_dir=folder, misses="link")
            disk.open()
            self.assertEqual((disk.tier, disk.misses), ("disk", "link"))
            self.assertTrue(disk.round_problems())   # misses read on the host: no captured rounds
            x, ids, weights = self.routes(random.Random(5))
            self.assertTrue(torch.equal(bits(disk.forward(1, x, ids, weights)), bits(fits.forward(1, x, ids, weights))))
            disk.close()

    def test_the_hosts_cache_keeps_the_store_between_loads(self):
        # No store_dir: the store goes to the host's plugin cache (context.cache, which crit run gives every run), so
        # the next load reopens it and needs no expert loads. It is keyed by the package's plugin code: after a
        # change there the store is built again and the old file removed. A disk that would keep less than the
        # floor of free space keeps none, and the reason says so. The machine's ledger is the test's own.
        from types import SimpleNamespace
        from unittest import mock
        x, ids, weights = self.routes(random.Random(5))
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {
                "CRIT_PLUGIN_CACHE_LEDGER": str(Path(folder) / "ledger.json"), "CRIT_DISK_FLOOR_GIB": "0"}):
            package = Path(folder) / "package"
            (package / "plugins" / "model").mkdir(parents=True)
            code = package / "plugins" / "model" / "plugin.py"
            code.write_text("READS = 1\n", encoding="utf-8")
            context = SimpleNamespace(cache=Path(folder) / "cache", package=package, config={"layers": LAYERS},
                                      model="checkpoint")
            stores = context.cache / "expert-offload"

            def load():
                experts = self.offload(tier="ram", misses="link", context=context)
                complete = experts.complete
                experts.open()
                return experts, complete
            first, complete = load()
            self.assertFalse(complete)
            self.assertIn("later loads reopen it", first.facts()["expert_offload"]["reasons"]["store"])
            want = bits(first.forward(1, x, ids, weights))
            first.close()
            (built,) = sorted(stores.glob("experts-*.bin"))
            again, complete = load()
            self.assertTrue(complete)
            self.assertIn("reopened", again.facts()["expert_offload"]["reasons"]["store"])
            self.assertTrue(torch.equal(bits(again.forward(1, x, ids, weights)), want))
            again.close()
            code.write_text("READS = 2\n", encoding="utf-8")
            changed, complete = load()
            self.assertFalse(complete)
            self.assertTrue(torch.equal(bits(changed.forward(1, x, ids, weights)), want))
            changed.close()
            (rebuilt,) = sorted(stores.glob("experts-*.bin"))
            self.assertNotEqual(rebuilt, built)
            code.write_text("READS = 3\n", encoding="utf-8")
            with mock.patch("shutil.disk_usage", return_value=SimpleNamespace(free=1 << 20)), \
                    mock.patch.dict(os.environ, {"CRIT_DISK_FLOOR_GIB": "1"}):
                none, complete = load()
            self.assertFalse(complete)
            self.assertIn("GiB floor", none.facts()["expert_offload"]["reasons"]["store"])
            self.assertEqual(sorted(stores.glob("experts-*.bin")), [])   # the checkpoint's old store went first
            self.assertTrue(torch.equal(bits(none.forward(1, x, ids, weights)), want))
            none.close()

    def mixed(self, **options):
        """Layers 0 and 1 W4A4, layer 2 W4A16 (weight-only: its experts carry no input scales), as a checkpoint whose
        draft layer's experts are stored otherwise than its decoder's."""
        use = dict(options.pop("use_options", {}))
        experts = self.assemble(USE, {"workers": 3, "pin": False, "max_tokens": 16, **options},
                                use_options=use)["experts"]
        self.addCleanup(experts.close)
        experts.begin(LAYERS + 1, EXPERTS, "nvfp4", HIDDEN, INTER)
        generator = torch.Generator().manual_seed(4)
        for layer in range(LAYERS + 1):
            for e in range(EXPERTS):
                found = stored(generator)
                experts.load(layer, e, *(found if layer < LAYERS else weight_only(found)), "nvfp4")
        return experts

    def test_a_weight_only_layer_beside_w4a4_layers_loads_in_every_tier(self):
        # The CPU tier computes W4A4 experts only. With the experts in host memory and no store file to leave the
        # weight-only layer out of, open() added that layer's experts to the tier, which refused them: the engine did
        # not load. In a file tier the same checkpoint was refused outright. The layer stays whole on the device
        # where the CPU tier may serve misses (it is left out of the store before the timing that picks who does:
        # with misses auto it stays there whoever is picked) and is cached as the others where only the link can.
        fits = self.mixed()
        rng = random.Random(9)
        calls = [(layer, *self.routes(rng)) for layer in (2, 0, 2, 1, 2)]
        want = [bits(fits.forward(layer, x, ids, weights)) for layer, x, ids, weights in calls]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            for options in (dict(tier="ram", misses="cpu"), dict(tier="ram", misses="auto"),
                            dict(tier="ram", misses="link"), dict(tier="disk", misses="auto", store_dir=folder)):
                offloaded = self.mixed(margin_mib=0, **options)
                offloaded.open()
                served = "cpu" if options["misses"] == "cpu" else offloaded.misses
                if options["misses"] == "cpu" and offloaded.misses != "cpu":
                    self.skipTest(f"no CPU tier here: {offloaded.reasons.get('misses')}")
                whole = options["tier"] == "ram" and options["misses"] != "link" and offloaded._cpu_possible("nvfp4")
                self.assertEqual(sorted(offloaded.resident), [2] if whole else [], options)
                for (layer, x, ids, weights), expected in zip(calls, want):
                    found = bits(offloaded.forward(layer, x, ids, weights))
                    if layer == 2 or served != "cpu":   # the kernel's tensor arithmetic: the device tier's bits
                        self.assertTrue(torch.equal(found, expected), (options, layer))
                    else:                               # the CPU's order on the W4A4 layers
                        declared = self.all_device(offloaded, layer, x, ids, weights, "declared")
                        self.assertTrue(torch.equal(found, bits(declared)), (options, layer))
                if served == "cpu":
                    self.assertNotIn(2, offloaded.dispatch.layers)
                    self.assertTrue(all(key // EXPERTS != 2 for key in offloaded.dispatch.tier.entries))
                offloaded.close()

    def test_a_store_with_compressed_scales_sends_its_misses_over_the_link(self):
        # The CPU tier reads bundles in place in the slot layout; a store that keeps scale components compressed
        # (the expert source's csf_scales) has another. On auto the misses go over the link, with the reason; asked
        # for the CPU by name, the engine is refused before the tier reads a byte of the wrong layout.
        csf = {"use_options": {"expert-tensors": {"csf_scales": "gate_up_scale:4096,down_scale:2048"}}}
        fits = self.offload()
        offloaded = self.offload(tier="ram", misses="auto", margin_mib=0, **csf)
        offloaded.open()
        self.assertEqual(offloaded.misses, "link")
        self.assertIn("compressed", offloaded.reasons["misses"])
        x, ids, weights = self.routes(random.Random(11))
        self.assertTrue(torch.equal(bits(offloaded.forward(1, x, ids, weights)), bits(fits.forward(1, x, ids, weights))))
        refused = self.offload(tier="ram", misses="cpu", margin_mib=0, **csf)
        with self.assertRaisesRegex(ValueError, "csf_scales"):
            refused.open()

    def test_recorded_routes_are_the_decode_routes(self):
        offloaded = self.offload(tier="ram", misses="link", margin_mib=0, record_routes=16)
        x, ids, weights = self.routes(random.Random(6), tokens=2)
        offloaded.forward(1, x, ids, weights)
        want = [(1, [e for e in row if e >= 0]) for row in ids.tolist()]
        self.assertEqual(offloaded.routes(), want)


if __name__ == "__main__":
    unittest.main()


GGUF_FORMATS = {0: ("Q8_0", "Q2_0"), 1: ("IQ2_XXS", "Q8_0")}   # each layer its own formats, as GGUF files have


def gguf_part(generator, type_name, rows, k):
    """Seeded stored rows uint8 [rows][row bytes] of a GGUF format with modest finite FP16 scales."""
    import numpy as np
    from crit_core.model.gguf_blocks import BLOCKS, SCALE_BYTES, TYPES
    values, size = BLOCKS[TYPES[type_name]]
    raw = torch.randint(0, 256, (rows, k // values, size), dtype=torch.uint8, generator=generator).numpy()
    for at in SCALE_BYTES[TYPES[type_name]]:
        scale = (np.full(raw.shape[:2], 2.0 ** -7) * (1 + torch.rand(raw.shape[:2], generator=generator).numpy()))
        raw[..., at:at + 2] = scale.astype(np.float16).view(np.uint8).reshape(*raw.shape[:2], 2)
    return {"weight": torch.from_numpy(np.ascontiguousarray(raw.reshape(rows, -1))), "type": type_name}


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "the device form runs on CUDA")
class GgufOffloadTests(PluginTestCase):
    """Experts as a GGUF checkpoint stores them (format gguf: each layer's formats): every tier gives the bits of every
    expert on the device, prompt chunks the bits of the same routes decoded, all of it the blocks of the stored values
    (Core's decoders) within the int8 activations' rounding."""

    def offload(self, formats=GGUF_FORMATS, **options):
        experts = self.assemble(USE, {"workers": 3, "pin": False, "max_tokens": 16, **options})["experts"]
        self.addCleanup(experts.close)
        experts.begin(len(formats), EXPERTS, "gguf", HIDDEN, INTER, formats=formats)
        generator = torch.Generator().manual_seed(9)
        self.parts = {}
        for layer, (gate_up, down) in formats.items():
            for e in range(EXPERTS):
                found = (gguf_part(generator, gate_up, 2 * INTER, HIDDEN), gguf_part(generator, down, HIDDEN, INTER))
                self.parts[layer, e] = found
                experts.load(layer, e, *found, "gguf")
        return experts

    def routes(self, rng, tokens=4, top=3):
        ids = torch.tensor([rng.sample(range(EXPERTS), top) for _ in range(tokens)], dtype=torch.int64)
        ids[rng.randrange(tokens), rng.randrange(top)] = -1
        weights = torch.rand(tokens, top, generator=torch.Generator().manual_seed(rng.randrange(1 << 30)))
        x = torch.randn(tokens, HIDDEN, generator=torch.Generator().manual_seed(rng.randrange(1 << 30))).bfloat16()
        return x.cuda(), ids.cuda(), weights.cuda()

    def reference(self, layer, x, ids, weights):
        """The block over the stored values in FP32 (Core's decoders), each route's gate SiLU."""
        from crit_core.model.gguf_blocks import TYPES, stored_values
        out = torch.zeros(x.shape[0], HIDDEN)
        for t in range(x.shape[0]):
            for j, e in enumerate(ids[t].tolist()):
                if e < 0:
                    continue
                gate_up, down = self.parts[layer, e]
                gu = torch.from_numpy(stored_values(gate_up["weight"].numpy(), TYPES[gate_up["type"]], HIDDEN))
                dn = torch.from_numpy(stored_values(down["weight"].numpy(), TYPES[down["type"]], INTER))
                both = gu @ x[t].float().cpu()
                h = torch.nn.functional.silu(both[:INTER]) * both[INTER:]
                out[t] += float(weights[t, j]) * (dn @ h)
        return out

    def test_every_tier_gives_the_device_tiers_bits_and_the_stored_values_block(self):
        fits = self.offload()
        offloaded = self.offload(tier="ram", misses="link", margin_mib=0)
        offloaded.open()
        self.assertEqual((fits.tier, offloaded.tier, offloaded.misses), ("device", "ram", "link"))
        rng = random.Random(5)
        for tokens in (1, 4, 40):   # decode rows, a round's columns, a prompt chunk (grouped by slot)
            for layer in range(LAYERS):
                x, ids, weights = self.routes(rng, tokens=tokens)
                want = fits.forward(layer, x, ids, weights)
                self.assertTrue(torch.equal(bits(offloaded.forward(layer, x, ids, weights)), bits(want)))
                if tokens > 4:
                    offloaded.end_step(False)
                ref = self.reference(layer, x, ids, weights)
                scale = ref.abs().max().item()
                self.assertLess((want.cpu() - ref).abs().max().item(), 0.05 * scale + 1e-3, (tokens, layer))
        # a prompt chunk's rows have the bits of the same routes decoded one token at a time
        x, ids, weights = self.routes(rng, tokens=24)
        chunk = fits.forward(0, x, ids, weights)
        rows = torch.cat([fits.forward(0, x[t:t + 1], ids[t:t + 1], weights[t:t + 1]) for t in range(24)])
        self.assertTrue(torch.equal(bits(chunk), bits(rows)))
        facts = offloaded.facts()["expert_offload"]
        self.assertEqual((facts["format"], facts["misses"]), ("gguf", "link"))
        self.assertIn("misses pinned to the link", offloaded.reasons["misses"])

    def test_a_layer_stored_wider_than_the_rest_stays_whole_on_the_device(self):
        # Twenty layers, one stored IQ2_S where the rest are IQ2_XXS (a twentieth of the bytes): the slot is the narrow
        # layers' bundle, the wide layer's bundles are stored in pieces of a slot and kept whole on the device at their
        # own size; every layer gives the bits of every expert on the device.
        formats = {layer: ("IQ2_XXS", "Q2_0") for layer in range(19)}
        formats[19] = ("IQ2_S", "Q2_0")
        fits = self.offload(formats)
        offloaded = self.offload(formats, tier="ram", misses="link", margin_mib=0)
        offloaded.open()
        found = offloaded.bundle_layout
        self.assertEqual(found["bundle_bytes"], found["sizes"][0])
        self.assertGreater(found["sizes"][19], found["bundle_bytes"])
        self.assertEqual((found["pieces"], found["store_layers"]), ({19: [20]}, 21))
        self.assertEqual(list(offloaded.resident), [19])
        self.assertEqual(tuple(offloaded.resident[19].shape), (EXPERTS, found["sizes"][19]))
        self.assertIn("kept whole on the device", offloaded.reasons["outsized"])
        rng = random.Random(11)
        for tokens in (1, 40):
            for layer in (0, 7, 19):
                x, ids, weights = self.routes(rng, tokens=tokens)
                got = offloaded.forward(layer, x, ids, weights)
                if tokens > 16:
                    offloaded.end_step(False)
                self.assertTrue(torch.equal(bits(got), bits(fits.forward(layer, x, ids, weights))), (layer, tokens))

    def test_the_cpu_tier_computes_gguf_misses_in_the_devices_bits(self):
        # The CPU tier computes a GGUF expert in gguf-rows' declared order on the host (Core's declared_rows, the
        # kernel's splits): a decode row served from it has the bits of every expert on the device, and a cache too
        # small for a prompt ring beside a layer's experts still serves prompts with every miss on the CPU tier.
        fits = self.offload().open()
        offloaded = self.offload(tier="ram", misses="cpu", margin_mib=0, settings="")
        offloaded.open()
        self.assertEqual((offloaded.misses, offloaded.dispatch.in_stream), ("cpu", False))
        self.assertEqual(type(offloaded.dispatch.tier).__name__, "GgufCpuTier")
        self.assertTrue(any("synchronization" in r for r in offloaded.round_problems()))
        self.assertEqual(offloaded.exactness(), {"decode": "declared", "prompt": "declared"})
        rng = random.Random(7)
        tier = offloaded.dispatch.tier
        for layer in range(LAYERS):
            x, ids, weights = self.routes(rng, tokens=2)
            # the tier's rows of each route against the device part's over every expert in its own slot
            keys = torch.tensor([layer * EXPERTS + e for e in range(EXPERTS)], dtype=torch.int32)
            tokens = torch.zeros(EXPERTS, dtype=torch.int32)
            host = tier.compute(keys, tokens, x[:1].float().cpu())
            store = fits.stores[layer]
            slots = torch.arange(EXPERTS, dtype=torch.int32, device="cuda")
            device = fits.part.route_outputs(layer, x[:1].repeat(EXPERTS, 1),
                                             torch.arange(EXPERTS, device="cuda").view(EXPERTS, 1), store, slots)
            self.assertTrue(torch.equal(bits(host.bfloat16()), bits(device.cpu())), layer)
            for tokens in (1, 3, 40):
                x, ids, weights = self.routes(rng, tokens=tokens)
                got = offloaded.forward(layer, x, ids, weights)
                if tokens > 16:
                    offloaded.end_step(False)
                self.assertTrue(torch.equal(bits(got), bits(fits.forward(layer, x, ids, weights))), (layer, tokens))
        self.assertGreater(tier.stats()["routes"], 0)
