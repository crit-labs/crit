"""expert-offload's device form on CUDA: each tier and miss path gives the bits of every expert on the device in the
same arithmetic (tensor where misses go over the link, the CPU's declared order where the CPU computes them), decode
rows and prompt chunks alike, captured rounds included, its facts and telemetry say what it decided, and an elastic
cache gives a prompt step its room and grows back with every route's bits unchanged."""
import json
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


def bits(t):
    return t.contiguous().view(torch.int32)


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "the device form runs on CUDA")
class OffloadDeviceTests(PluginTestCase):
    def offload(self, **options):
        experts = self.assemble(USE, {"workers": 3, "pin": False, "max_tokens": 16, **options})
        experts = experts["experts"]
        self.addCleanup(experts.close)
        experts.begin(LAYERS, EXPERTS, "nvfp4", HIDDEN, INTER)
        generator = torch.Generator().manual_seed(4)
        for layer in range(LAYERS):
            for e in range(EXPERTS):
                experts.load(layer, e, *stored(generator), "nvfp4")
        return experts

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

    def test_recorded_routes_are_the_decode_routes(self):
        offloaded = self.offload(tier="ram", misses="link", margin_mib=0, record_routes=16)
        x, ids, weights = self.routes(random.Random(6), tokens=2)
        offloaded.forward(1, x, ids, weights)
        want = [(1, [e for e in row if e >= 0]) for row in ids.tolist()]
        self.assertEqual(offloaded.routes(), want)


if __name__ == "__main__":
    unittest.main()
