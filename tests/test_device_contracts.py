"""crit_plugins' device contracts beyond each slot's device form: the extensions a plugin offers by the facts it provides
(DEVICE_EXTENSIONS), checked when an engine assembles, and the round batch captured decode rounds compute over."""
from pathlib import Path
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core" / "python"))

from crit_plugins import PluginError, Registry, assemble  # noqa: E402
from crit_plugins.device import DEVICE_EXTENSIONS, RoundBatch, StepBatch, extension_missing  # noqa: E402
from support.plugins import write_plugins  # noqa: E402


class ExtensionTests(unittest.TestCase):
    def test_every_extension_names_a_fact_and_documents_its_methods(self):
        for fact, methods in DEVICE_EXTENSIONS.items():
            self.assertRegex(fact, r"^[a-z_]+:[a-z_-]+$")
            for name, text in methods.items():
                self.assertTrue(text.startswith(name + "("), f"{fact}.{name}: {text}")

    def test_a_method_counts_on_the_plugin_or_an_object_its_device_form_made(self):
        class Pool:
            def bind(self, slot, sequence, capacity): pass
            def unbind(self, slot): pass
        self.assertEqual(extension_missing("rounds:kv_cache", object(), (Pool(),)), ["commit_round"])
        self.assertEqual(extension_missing("no:such-fact", object()), [])

    def test_an_engine_refuses_a_plugin_that_provides_an_extension_without_its_methods(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_plugins(Path(temp) / "plugins", {
                "drafts": {"fills": ["drafter"], "provides": ["draft:device"], "devices": ["host"]}})
            registry = Registry.discover(root)
            with self.assertRaisesRegex(PluginError, "provides draft:device but lacks draft, commit_round"):
                assemble({"use": {"drafter": "drafts"}}, registry)


class KernelLoaderTests(unittest.TestCase):
    """crit_plugins.kernels: crit's kernels a plugin's manifest declares, built from crit's library with the engine's
    tuning or the default knobs, and recorded fallbacks."""

    def setUp(self):
        from unittest.mock import patch
        from crit_plugins import kernels
        self.kernels = kernels
        kernels.forget()
        kernels.use_tuning(None)
        self.addCleanup(kernels.forget)
        self.addCleanup(kernels.use_tuning, None)
        self.saved = {name: sys.modules.get(name) for name in ("crit_kernels", "crit_kernels.build")}
        self.addCleanup(self._restore)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        self.root = self.temp / "plugins"
        # the tests name their own tuning: a CRIT_KERNEL_TUNING of the run must not reach them
        environ = patch.dict("os.environ")
        environ.start()
        self.addCleanup(environ.stop)
        os.environ.pop("CRIT_KERNEL_TUNING", None)

    def _restore(self):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def fake(self, available):
        """crit_kernels.build with load() returning available[id] (a failed build raises), and its calls."""
        import json
        import types
        package, module = types.ModuleType("crit_kernels"), types.ModuleType("crit_kernels.build")
        calls = []

        def load(kernel_id, tuning=None, **options):
            calls.append((kernel_id, tuning))
            if kernel_id not in available:
                raise RuntimeError(f"Error building extension '{kernel_id}'\nnvcc: not found")
            return available[kernel_id]
        module.load = load
        module.read_tuning = lambda path: json.loads(Path(path).read_text())
        package.build = module
        sys.modules["crit_kernels"], sys.modules["crit_kernels.build"] = package, module
        return calls

    def plugin(self, plugin_id, *kernels):
        """A host plugin whose manifest declares `kernels` ({name, fallback[, backend]})."""
        import json
        write_plugins(self.root, {plugin_id: {"fills": ["drafter"], "devices": ["host"]}})
        path = self.root / plugin_id / "plugin.json"
        manifest = json.loads(path.read_text())
        manifest["kernels"] = [dict({"fallback": "torch"}, **k) for k in kernels]
        path.write_text(json.dumps(manifest))
        return self.root / plugin_id

    def test_a_kernel_is_built_once_with_its_default_knobs_and_shared_by_plugins(self):
        sentinel = object()
        calls = self.fake({"grouped-experts": sentinel})
        moe = self.plugin("moe", {"name": "grouped-experts"})
        other = self.plugin("other", {"name": "grouped-experts"})
        self.assertIs(self.kernels.load_kernel("grouped-experts", folder=moe), sentinel)
        self.assertIs(self.kernels.load_kernel("grouped-experts", folder=other), sentinel)
        self.assertEqual(calls, [("grouped-experts", None)])
        self.assertEqual(self.kernels.loaded()["grouped-experts"], {"source": "built", "plugins": ["moe", "other"]})

    def test_the_engines_tuning_builds_a_tuned_kernel(self):
        import json
        calls = self.fake({"recurrent-scans": object()})
        tuning = self.temp / "tuning"
        tuning.mkdir()
        (tuning / "recurrent-scans.json").write_text(json.dumps({"schema": "crit-kernel-tuning-v1"}))
        recur = self.plugin("recur", {"name": "recurrent-scans"})
        self.kernels.use_tuning(tuning)
        self.kernels.load_kernel("recurrent-scans", folder=recur)
        self.assertEqual(calls, [("recurrent-scans", {"schema": "crit-kernel-tuning-v1"})])
        record = self.kernels.loaded()["recurrent-scans"]
        self.assertEqual((record["source"], record["tuning"]), ("tuned", str(tuning / "recurrent-scans.json")))

    def test_a_plugin_may_ask_for_a_kernel_with_its_default_knobs_beside_the_tuned_one(self):
        import json
        tuned, untuned = object(), object()
        calls = []
        self.fake({})
        build = sys.modules["crit_kernels.build"]

        def load(kernel_id, tuning=None, **options):
            calls.append((kernel_id, tuning))
            return tuned if tuning else untuned
        build.load = load
        tuning = self.temp / "tuning"
        tuning.mkdir()
        (tuning / "fp8-block-rows.json").write_text(json.dumps({"schema": "crit-kernel-tuning-v1"}))
        proj = self.plugin("proj", {"name": "fp8-block-rows"})
        self.kernels.use_tuning(tuning)
        self.assertIs(self.kernels.load_kernel("fp8-block-rows", folder=proj), tuned)
        self.assertIs(self.kernels.load_kernel("fp8-block-rows", folder=proj, tuned=False), untuned)
        self.assertIs(self.kernels.load_kernel("fp8-block-rows", folder=proj, tuned=False), untuned)   # built once
        self.assertEqual(calls, [("fp8-block-rows", {"schema": "crit-kernel-tuning-v1"}), ("fp8-block-rows", None)])
        found = self.kernels.loaded()
        self.assertEqual(found["fp8-block-rows"]["source"], "tuned")
        self.assertEqual(found["fp8-block-rows" + self.kernels.UNTUNED], {"source": "built", "plugins": ["proj"]})

    def test_a_kernel_that_cannot_be_built_falls_back_with_its_reason(self):
        self.fake({})
        proj = self.plugin("proj", {"name": "fp8-block-rows"})
        self.assertIsNone(self.kernels.load_kernel("fp8-block-rows", folder=proj))
        record = self.kernels.loaded()["fp8-block-rows"]
        self.assertEqual(record["source"], "fallback")
        self.assertEqual(record["reason"], "RuntimeError: nvcc: not found")
        self.assertEqual(self.kernels.fallbacks(), {"fp8-block-rows": record["reason"]})

    def test_without_crits_kernels_package_every_kernel_falls_back(self):
        sys.modules["crit_kernels"] = sys.modules["crit_kernels.build"] = None   # imports raise ImportError
        moe = self.plugin("moe", {"name": "grouped-experts"})
        self.assertIsNone(self.kernels.load_kernel("grouped-experts", folder=moe))
        self.assertEqual(self.kernels.loaded()["grouped-experts"]["reason"],
                         "crit's kernels package (crit_kernels) is not installed")

    def test_an_undeclared_kernel_and_bad_entries_are_refused(self):
        moe = self.plugin("moe", {"name": "grouped-experts"})
        with self.assertRaisesRegex(PluginError, "moe loads the kernel other, which its plugin.json does not declare"):
            self.kernels.load_kernel("other", folder=moe)
        for bad, message in (({"name": "crit-labs/grouped"}, "is the id of one of crit's kernels"),
                             ({"name": "a", "version": 1}, r"is \{name, fallback\}"),
                             ({"name": "a", "fallback": ""}, "without it"),
                             ({"name": "a", "backend": "tpu"}, "backend is cuda, rocm, metal, xpu or cpu")):
            self.plugin("moe", bad)
            with self.assertRaisesRegex(PluginError, message):
                Registry.discover(self.root)

    def test_the_engine_facts_carry_the_kernels_of_its_plugins(self):
        from crit_core.serving import engine_facts
        self.fake({})
        self.kernels.load_kernel("grouped-experts", folder=self.plugin("moe", {"name": "grouped-experts"}))
        self.kernels.load_kernel("recurrent-scans", folder=self.plugin("elsewhere", {"name": "recurrent-scans"}))
        assembly = assemble({"use": {"drafter": "moe"}}, Registry.discover(self.root))
        facts = assembly.facts()
        self.assertEqual(list(facts["kernels"]), ["grouped-experts"])
        engine = type("Engine", (), {"facts": facts})()
        self.assertEqual(engine_facts(engine)["kernels"]["grouped-experts"]["source"], "fallback")


class ReleaseCachedTests(unittest.TestCase):
    def test_cached_blocks_go_back_only_below_the_free_threshold(self):
        import types
        from unittest import mock
        from crit_plugins.device import release_cached_under
        state = {"free": 4 << 30, "reserved": 3 << 30, "emptied": 0}

        def empty():   # torch's cache of 3 GiB, 1 GiB of it in use
            state["emptied"] += 1
            state["free"] += state["reserved"] - (1 << 30)
            state["reserved"] = 1 << 30
        cuda = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, device_count=lambda: 1,
                                     mem_get_info=lambda device: (state["free"], 32 << 30), empty_cache=empty,
                                     memory_allocated=lambda i: 1 << 30, memory_reserved=lambda i: state["reserved"])
        fake = types.SimpleNamespace(cuda=cuda, device=lambda name: types.SimpleNamespace(type=str(name).split(":")[0]))
        with mock.patch.dict(sys.modules, {"torch": fake}):
            self.assertEqual(release_cached_under(1 << 30), 0)         # 4 GiB free: the cache stays
            self.assertEqual(state["emptied"], 0)
            self.assertEqual(release_cached_under(8 << 30), 2 << 30)   # under 8 GiB: the 2 GiB nothing uses
            self.assertEqual((state["emptied"], state["free"]), (1, 6 << 30))
            self.assertIsNone(release_cached_under(8 << 30, "cpu"))


class RoundBatchTests(unittest.TestCase):
    def test_a_round_fills_its_tokens_positions_and_rows_in_fixed_buffers(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        batch = RoundBatch(4, 3, "cpu")
        addresses = batch._tokens.data_ptr(), batch._positions.data_ptr()
        frontier = torch.tensor([10, 20, 30, 40])
        columns = torch.arange(16).view(4, 4)
        batch.fill(2, 2, frontier, columns)
        self.assertTrue(batch.round and not StepBatch.round)
        self.assertEqual((batch.rows, batch.width, batch.tokens), (2, 2, 6))
        self.assertEqual(batch.tokens_t.tolist(), [0, 1, 2, 4, 5, 6])
        self.assertEqual(batch.positions_t.tolist(), [10, 11, 12, 20, 21, 22])
        self.assertEqual(batch.rows_t.tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(batch.starts_t.tolist(), [0, 3, 6])
        self.assertEqual(batch.slots_t.tolist(), [0, 1])
        self.assertEqual(batch.frontier_t.tolist(), [10, 20])
        self.assertEqual((batch.tokens_t.data_ptr(), batch.positions_t.data_ptr()), addresses)
        with self.assertRaises(ValueError):
            batch.fill(5, 1, frontier, columns)
        with self.assertRaises(ValueError):
            batch.row(0)


if __name__ == "__main__":
    unittest.main()
