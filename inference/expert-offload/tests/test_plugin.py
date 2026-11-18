"""expert-offload's host form against the reference, and its decision tree on given sizes, rates and quotas."""
import importlib.util
from pathlib import Path

from crit_plugins.testing import PluginTestCase

USE = {"expert_source": "expert-tensors"}
CODE = Path(__file__).resolve().parents[1] / "code" / "offload.py"
spec = importlib.util.spec_from_file_location("expert_offload_code", CODE)
offload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offload)
GB = 10 ** 9


class ExpertOffloadTests(PluginTestCase):
    def test_it_has_the_experts_interface(self):
        self.assertImplements(self.assemble(USE))

    def test_its_host_form_is_the_reference(self):
        self.assertMatchesReference(USE, {}, atol=0.0, rtol=0.0)

    def test_prompt_arithmetic_auto_follows_the_level_the_engine_asks_of_its_prompt_rows(self):
        from crit_plugins import kernels
        self.addCleanup(kernels.use_exactness, None)
        for asked, arithmetic in ((None, "declared"), ("repeatable", "tensor"), ("row-exact", "tensor"),
                                  ("exact", "declared"), ("declared", "declared")):
            kernels.use_exactness(asked and {"decode": "row-exact", "prompt": asked})
            experts = self.assemble(USE)["experts"]
            self.assertEqual(experts._asked_arithmetic(), arithmetic, asked)
            self.assertEqual(experts.exactness(), {})   # before open() it states no level
        kernels.use_exactness("exact")   # an option set to one arithmetic is that arithmetic, whatever is asked
        self.assertEqual(self.assemble(USE, {"prompt_arithmetic": "tensor"})["experts"]._asked_arithmetic(), "tensor")

    def test_the_tier_follows_the_memory_each_holds(self):
        tier = offload.choose_tier
        self.assertEqual(tier(20 * GB, 32 * GB, 6 * GB, 90 * GB, GB, 2 * GB), "device")
        self.assertEqual(tier(68 * GB, 32 * GB, 6 * GB, 90 * GB, GB, 2 * GB), "ram")
        self.assertEqual(tier(68 * GB, 32 * GB, 6 * GB, 60 * GB, GB, 2 * GB), "disk")
        self.assertEqual(tier(20 * GB, None, 0, 90 * GB, GB, 2 * GB), "ram")      # no device memory to read
        self.assertEqual(tier(20 * GB, 32 * GB, 6 * GB, 90 * GB, GB, 2 * GB, "disk"), "disk")

    def test_the_cache_takes_every_byte_past_the_margin(self):
        self.assertEqual(offload.cache_slots(24 * GB + 1000, 2 * GB, 2780176), (22 * GB + 1000) // 2780176 - 1)
        self.assertEqual(offload.cache_slots(GB, 2 * GB, 2780176), 0)

    def test_misses_go_to_the_cpu_only_where_it_computes_them(self):
        choose = offload.choose_misses
        self.assertEqual(choose("nvfp4", True), "cpu")
        self.assertEqual(choose("nvfp4", False), "link")          # no native tier or join here
        self.assertEqual(choose("fp8:block", True), "link")       # not a CPU tier format
        self.assertEqual(choose("int4", True, "cpu"), "cpu")
        self.assertEqual(choose("int4", True, "link"), "link")
        # auto with both timed at load: the CPU tier where it takes under CPU_WINS of the link's time a missed expert
        self.assertEqual(choose("nvfp4", True, "auto", {"cpu": 110.0, "link": 105.0}), "link")
        self.assertEqual(choose("nvfp4", True, "auto", {"cpu": 60.0, "link": 105.0}), "cpu")
        self.assertEqual(choose("nvfp4", True, "cpu", {"cpu": 110.0, "link": 105.0}), "cpu")     # pinned
        self.assertEqual(choose("nvfp4", False, "auto", {"cpu": 60.0, "link": 105.0}), "link")   # no tier here

    def test_the_disk_tiers_store_keeps_each_expert_once_and_reads_the_file_at_a_depth(self):
        import tempfile
        import torch
        layers, experts, size = 2, 10, 5000                           # bundles that end off a block of the file
        g = torch.Generator().manual_seed(2)
        bundles = torch.randint(0, 256, (layers * experts, size), dtype=torch.uint8, generator=g)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            for host_slots in (4, 0):
                with self.subTest(host_slots=host_slots):
                    store = offload.FileStore({"bundle_bytes": size}, layers, experts, Path(folder) / f"e{host_slots}.bin",
                                              host_bytes=host_slots * size + 17)
                    self.addCleanup(store.close)
                    for key in range(layers * experts):
                        store.put(key // experts, key % experts, bundles[key])
                    self.assertEqual((store.slots, store.retains, store.mapped), (host_slots, host_slots > 0, False))
                    into = torch.empty(6, size, dtype=torch.uint8)
                    keys = [0, 7, 19, 3, 11, 12]
                    store.read(keys, into)                             # six reads of the file, at once
                    self.assertTrue(torch.equal(into, bundles[keys]))
                    seen = store.stats()
                    self.assertEqual((seen["file_reads"], seen["host_hits"]), (6, 0))
                    self.assertEqual((seen["reads"]["reads"], seen["reads"]["depth"]), (6, 8))
                    if not host_slots:
                        self.assertFalse(seen["reads"]["direct"])      # the page cache is the host copy
                        store.retain([0], bundles[[0]])                # nothing to keep them in
                        self.assertEqual(store.stats()["host_held"], 0)
                        store.close()
                        continue
                    # The device cache evicts five: four fit, the one kept longest leaves for the fifth.
                    store.retain([0, 7, 19, 3, 11], bundles[[0, 7, 19, 3, 11]])
                    seen = store.stats()
                    self.assertEqual((seen["retained"], seen["dropped"], seen["host_held"]), (5, 1, 4))
                    two = torch.empty(2, size, dtype=torch.uint8)
                    store.read([7, 0], two)                            # 7 from the host tier, 0 from the file again
                    self.assertTrue(torch.equal(two, bundles[[7, 0]]))
                    seen = store.stats()
                    self.assertEqual((seen["host_hits"], seen["file_reads"], seen["host_held"]), (1, 7, 3))
                    store.read([19], two[:1], keep=True)               # a prompt's ring slot: the host copy stays
                    self.assertTrue(torch.equal(two[0], bundles[19]))
                    self.assertEqual(store.stats()["host_held"], 3)
                    store.retain([19], bundles[[19]])                  # held already
                    self.assertEqual(store.stats()["retained"], 5)
                    store.put(0, 5, bundles[9])                        # a load after reads: the readers open again
                    store.read([5], two[:1])
                    self.assertTrue(torch.equal(two[0], bundles[9]))
                    store.close()

    def test_a_prompt_steps_bytes_follow_the_steps_measured(self):
        need = offload.prompt_bytes
        self.assertEqual(need({}, 1024), 0)
        self.assertEqual(need({37: 700}, 16), 700)                    # the part that does not grow with the tokens
        self.assertEqual(need({37: 700}, 74), 1400)                   # one size: in proportion past it
        seen = {37: 700, 601: 2956}                                   # 4 bytes a token over 700
        self.assertEqual(need(seen, 37), 700)
        self.assertEqual(need(seen, 319), 1828)
        self.assertEqual(need(seen, 1024), 2956 * 1024 // 601)        # past the largest: in proportion, for a step
        # Peaks of the load are first steps (their captures' temporaries): one stands until a served step of its
        # size or larger took less; a larger size's stays, and served peaks keep their most.
        peaks, first = {}, set()
        for tokens, peak in ((7, 358), (37, 234), (601, 350)):
            offload.learn(peaks, first, tokens, peak, serving=False)
        offload.learn(peaks, first, 40, 128, serving=True)
        self.assertEqual((peaks, first), ({40: 128, 601: 350}, {601}))
        offload.learn(peaks, first, 40, 110, serving=True)
        offload.learn(peaks, first, 601, 400, serving=True)
        self.assertEqual((peaks, first), ({40: 128, 601: 400}, set()))
        self.assertEqual(need({37: 900, 601: 400}, 602), 900)         # never less than a smaller chunk took

    def test_the_diagnosis_names_the_hit_rate_and_the_next_lever(self):
        line = offload.diagnosis({"tier": "ram", "misses": "cpu", "expert_cache": {"reads": 1000, "hits": 620,
                                  "slots": 6000, "fills": 40}, "cpu_routes": {"routes": 300}})
        self.assertIn("hits 62.0%", line)
        self.assertIn("300 routes on the CPU", line)
        self.assertIn("next lever", line)
        self.assertNotIn("GiB", line)                              # no device shares given: none claimed
        shared = offload.diagnosis({"tier": "ram", "misses": "cpu", "expert_cache": {"reads": 1000, "hits": 900,
                                    "slots": 7000, "fills": 90}, "cpu_routes": {"routes": 100},
                                    "device": {"total": 32 << 30, "cache": 19 << 30, "model": 9 << 30,
                                               "free": 1 << 30, "bundle": 1 << 21,
                                               "holders": [["projections", 6 << 30], ["the-model", 3 << 29]],
                                               "tensors": [{"plugin": "the-model", "bytes": 3 << 29,
                                                            "shape": [250000, 3072], "dtype": "bfloat16"}]}})
        self.assertIn("of the device's 32.0 GiB the cache holds 19.0, the model's weights, pools and workspaces 9.0 "
                      "(projections 6.0, the-model 1.5), 1.0 are free (a GiB more for the cache is 512 slots)", shared)
        self.assertIn("its largest tensors: [250000][3072] bfloat16 1.5 of the-model", shared)   # a table, by shape
        self.assertNotIn("next lever", shared)                     # 90% hits: the shares say enough
        self.assertIn("device", offload.diagnosis({"tier": "device"}))
