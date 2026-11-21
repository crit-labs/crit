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

    def test_the_store_is_keyed_by_the_code_that_packs_it_not_by_every_plugin(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            plugins = Path(temp) / "plugins"

            def plugin(folder, name, sources=None, code="x = 1\n"):
                folder = plugins / folder
                (folder / "code").mkdir(parents=True)
                (folder / "plugin.json").write_text(json.dumps({"id": name, "sources": sources or {}}), "utf-8")
                (folder / "code" / "main.py").write_text(code, encoding="utf-8")
                (folder / "tests").mkdir()
                (folder / "tests" / "test_it.py").write_text("# a test\n", encoding="utf-8")
            plugin("toy-model", "toy-model")
            plugin("toy-experts", "toy-experts", {"offload": "code", "cache": "../toy-cache/code"})
            plugin("toy-cache", "toy-cache")
            plugin("base/toy-source", "toy-source")
            plugin("toy-sampler", "toy-sampler")
            use = {"plugins": {"use": {"model": "toy-model", "experts": "toy-experts",
                                       "expert_source": "toy-source", "sampler": "toy-sampler"}}}
            folders = offload.packing_folders(plugins, use)
            self.assertEqual([f.relative_to(plugins.resolve()).as_posix() for f in folders],
                             ["toy-model", "base/toy-source", "toy-experts", "toy-experts/code", "toy-cache/code"])
            context = type("Context", (), {"package": temp, "configuration": use})()
            experts = offload.ExpertOffload.__new__(offload.ExpertOffload)
            experts.context = context
            before = experts._code()
            (plugins / "toy-sampler" / "code" / "main.py").write_text("x = 2\n", encoding="utf-8")
            (plugins / "toy-experts" / "tests" / "test_it.py").write_text("# changed\n", encoding="utf-8")
            self.assertEqual(experts._code(), before)    # another slot's plugin, a test: the same store
            (plugins / "toy-cache" / "code" / "main.py").write_text("x = 3\n", encoding="utf-8")
            self.assertNotEqual(experts._code(), before)  # code the experts plugin packs with: another store
            # a selection that does not name the three: every plugin's code, as before
            self.assertIsNone(offload.packing_folders(plugins, {"plugins": {"use": {"model": "toy-model"}}}))

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


GGUF = importlib.util.spec_from_file_location("expert_offload_gguf", CODE.with_name("gguf_experts.py"))
gguf_experts = importlib.util.module_from_spec(GGUF)
GGUF.loader.exec_module(gguf_experts)


class GgufBundleTests(PluginTestCase):
    """GGUF experts' bundles: each layer's formats, the down rows at the layer's 64-byte boundary after its gate-up rows,
    the slot the bundle most layers need, a few wide layers (a sixteenth of the bytes at most) in pieces of a slot."""

    def test_the_slot_leaves_only_a_few_wide_layers_outside(self):
        self.assertEqual(gguf_experts.slot_bytes({0: 100, 1: 100, 2: 300}), 300)          # a third of the bytes: no
        self.assertEqual(gguf_experts.slot_bytes({i: 100 for i in range(31)} | {31: 150}), 100)
        self.assertEqual(gguf_experts.slot_bytes({i: 100 + i for i in range(20)}), 118)  # the widest outside
        formats = {i: ("IQ2_XXS", "Q2_0") for i in range(20)}
        formats[20] = ("IQ2_S", "Q2_0")
        found = gguf_experts.layout(formats, 256, 128)
        self.assertEqual((found["bundle_bytes"], found["sizes"][20]), (26112, 30208))
        self.assertEqual((found["pieces"], found["store_layers"]), ({20: [21]}, 22))
        import torch
        bundle = torch.arange(30208, dtype=torch.int64).to(torch.uint8)
        cut = gguf_experts.pieces(bundle, found, 20)
        self.assertEqual([(where, piece.numel()) for where, piece in cut], [(20, 26112), (21, 26112)])
        self.assertTrue(torch.equal(torch.cat([piece for _, piece in cut])[:30208], bundle))
        formats[20] = ("Q8_0", "Q8_0")   # a sixth of the bytes: every slot that layer's size
        self.assertEqual(gguf_experts.layout(formats, 256, 128)["pieces"], {})

    def test_the_layout_holds_every_layers_formats_in_bundles_of_one_size(self):
        found = gguf_experts.layout({0: ("IQ2_XXS", "Q2_0"), 1: ("IQ2_S", "IQ4_NL")}, 2560, 640)
        gate_up = {0: 2 * 640 * 660, 1: 2 * 640 * 820}            # 2560 values a row: 10 blocks of 66 / 82 bytes
        down = {0: 2560 * 180, 1: 2560 * 360}                      # 640 values a row: 10 Q2_0 / 20 IQ4_NL blocks
        self.assertEqual(found["offsets"], {k: -(-v // 64) * 64 for k, v in gate_up.items()})
        self.assertEqual(found["bundle_bytes"], max(found["offsets"][k] + -(-down[k] // 64) * 64 for k in down))
        self.assertEqual(found["formats"], {0: ["IQ2_XXS", "Q2_0"], 1: ["IQ2_S", "IQ4_NL"]})
        self.assertEqual(found["bundle_bytes"] % 16, 0)   # a bundle store's rows are whole 16-byte units

    def test_a_bundle_is_the_stored_rows_at_their_offsets(self):
        import torch
        found = gguf_experts.layout({0: ("Q8_0", "Q2_0"), 1: ("Q4_K", "Q8_0")}, 256, 128)
        g = torch.Generator().manual_seed(2)
        gate_up = torch.randint(0, 256, (256, 272), dtype=torch.uint8, generator=g)     # Q8_0: 8 blocks of 34
        down = torch.randint(0, 256, (256, 36), dtype=torch.uint8, generator=g)         # Q2_0: 2 blocks of 18
        bundle = gguf_experts.pack({"weight": gate_up, "type": "Q8_0"}, {"weight": down, "type": "Q2_0"}, found, 0)
        self.assertEqual(bundle.numel(), found["bundle_bytes"])
        self.assertTrue(torch.equal(bundle[:gate_up.numel()], gate_up.reshape(-1)))
        at = found["offsets"][0]
        self.assertTrue(torch.equal(bundle[at:at + down.numel()], down.reshape(-1)))
        with self.assertRaisesRegex(ValueError, "layer 1's experts are Q4_K / Q8_0"):
            gguf_experts.pack({"weight": gate_up, "type": "Q8_0"}, {"weight": down, "type": "Q2_0"}, found, 1)
        with self.assertRaisesRegex(ValueError, r"is \[256\]\[36\] bytes"):
            gguf_experts.pack({"weight": gate_up, "type": "Q8_0"}, {"weight": down[:, :20], "type": "Q2_0"}, found, 0)

    def test_a_store_folder_not_there_yet_is_made(self):
        import os
        import tempfile
        if os.name == "nt":
            self.skipTest("the ram tier's store is anonymous host memory through libc, which Windows does not have")
        with tempfile.TemporaryDirectory() as folder:
            store_dir = Path(folder) / "not" / "there"
            experts = self.assemble(USE, {"store_dir": str(store_dir), "tier": "ram"})["experts"]
            experts.begin(2, 4, "gguf", 256, 128, formats={0: ("Q8_0", "Q8_0"), 1: ("Q8_0", "Q8_0")}, device="cpu")
            self.assertEqual(Path(experts.store.path).parent, store_dir)
            experts.close()

    def test_begin_takes_every_layers_formats(self):
        experts = self.assemble(USE)["experts"]
        with self.assertRaisesRegex(ValueError, "formats of every layer 0 .. 1"):
            experts.begin(2, 4, "gguf", 256, 128, formats={0: ("Q8_0", "Q8_0")}, device="cpu")
        with self.assertRaisesRegex(ValueError, "formats are a GGUF checkpoint's"):
            self.assemble(USE)["experts"].begin(2, 4, "bf16", 256, 128, formats={}, device="cpu")
