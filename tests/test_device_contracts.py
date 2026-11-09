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


class KernelRegistryTests(unittest.TestCase):
    """crit_plugins.kernels: the registry kernels a plugin's manifest declares, loaded at their pins, local builds, and
    recorded fallbacks."""

    def setUp(self):
        from unittest.mock import patch
        from crit_plugins import kernels
        self.kernels = kernels
        kernels.forget()
        self.addCleanup(kernels.forget)
        self.saved = sys.modules.get("kernels")
        self.addCleanup(self._restore)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "plugins"
        # the tests name their own local builds: a LOCAL_KERNELS of the run (local kernel builds) must not reach them
        environ = patch.dict("os.environ")
        environ.start()
        self.addCleanup(environ.stop)
        os.environ.pop("LOCAL_KERNELS", None)

    def _restore(self):
        if self.saved is None:
            sys.modules.pop("kernels", None)
        else:
            sys.modules["kernels"] = self.saved

    def fake(self, available):
        import types
        module = types.ModuleType("kernels")
        calls = []

        def get_kernel(repo, **options):
            calls.append((repo, options))
            if repo not in available:
                raise FileNotFoundError(f"Cannot find a build variant for {repo}\nmore lines")
            return available[repo]
        module.get_kernel = get_kernel
        sys.modules["kernels"] = module
        return calls

    def plugin(self, plugin_id, *kernels):
        """A host plugin whose manifest declares `kernels` ({name, version, revision, fallback[, backend]})."""
        import json
        write_plugins(self.root, {plugin_id: {"fills": ["drafter"], "devices": ["host"]}})
        path = self.root / plugin_id / "plugin.json"
        manifest = json.loads(path.read_text())
        manifest["kernels"] = [dict({"version": 1, "revision": None, "fallback": "torch"}, **k) for k in kernels]
        path.write_text(json.dumps(manifest))
        return self.root / plugin_id

    def test_a_published_kernel_comes_from_the_registry_at_its_pinned_revision(self):
        sentinel = object()
        calls = self.fake({"crit-labs/grouped-experts": sentinel})
        moe = self.plugin("moe", {"name": "crit-labs/grouped-experts", "revision": "abc1234"})
        other = self.plugin("other", {"name": "crit-labs/grouped-experts", "revision": "abc1234"})
        self.assertIs(self.kernels.registry_kernel("crit-labs/grouped-experts", folder=moe), sentinel)
        self.assertEqual(calls, [("crit-labs/grouped-experts", {"trust_remote_code": ["crit-labs/grouped-experts"],
                                                               "revision": "abc1234"})])
        self.assertIs(self.kernels.registry_kernel("crit-labs/grouped-experts", folder=other), sentinel)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.kernels.loaded()["crit-labs/grouped-experts"],
                         {"source": "hub", "version": 1, "revision": "abc1234", "plugins": ["moe", "other"]})

    def test_an_unpublished_kernel_is_asked_for_by_its_version(self):
        calls = self.fake({"crit-labs/grouped-experts": object()})
        self.kernels.registry_kernel("crit-labs/grouped-experts",
                                     folder=self.plugin("moe", {"name": "crit-labs/grouped-experts"}))
        self.assertEqual(calls[0][1], {"trust_remote_code": ["crit-labs/grouped-experts"], "version": 1})

    def test_a_local_build_named_in_local_kernels_is_recorded_as_local(self):
        from unittest.mock import patch
        self.fake({"crit-labs/recurrent-scans": object()})
        recur = self.plugin("recur", {"name": "crit-labs/recurrent-scans"})
        with patch.dict("os.environ", {"LOCAL_KERNELS": "crit-labs/recurrent-scans=/tmp/recurrent-scans"}):
            self.kernels.registry_kernel("crit-labs/recurrent-scans", folder=recur)
        self.assertEqual(self.kernels.loaded()["crit-labs/recurrent-scans"]["source"], "local")

    def test_an_unavailable_kernel_falls_back_with_its_reason(self):
        self.fake({})
        proj = self.plugin("proj", {"name": "crit-labs/fp8-block-rows"})
        self.assertIsNone(self.kernels.registry_kernel("crit-labs/fp8-block-rows", folder=proj))
        record = self.kernels.loaded()["crit-labs/fp8-block-rows"]
        self.assertEqual(record["source"], "fallback")
        self.assertEqual(record["reason"], "FileNotFoundError: Cannot find a build variant for crit-labs/fp8-block-rows")
        self.assertEqual(self.kernels.fallbacks(), {"crit-labs/fp8-block-rows": record["reason"]})

    def test_without_the_kernels_package_every_kernel_falls_back(self):
        sys.modules["kernels"] = None   # import kernels raises ImportError
        moe = self.plugin("moe", {"name": "crit-labs/grouped-experts"})
        self.assertIsNone(self.kernels.registry_kernel("crit-labs/grouped-experts", folder=moe))
        self.assertEqual(self.kernels.loaded()["crit-labs/grouped-experts"]["reason"],
                         "the kernels package is not installed")

    def test_an_undeclared_kernel_and_bad_pins_are_refused(self):
        moe = self.plugin("moe", {"name": "crit-labs/grouped-experts"})
        with self.assertRaisesRegex(PluginError, "moe loads the registry kernel crit-labs/other, which its "
                                                 "plugin.json does not declare"):
            self.kernels.registry_kernel("crit-labs/other", folder=moe)
        for bad, message in (({"name": "Grouped"}, "is ORG/NAME"), ({"name": "a/b", "version": 0}, "major version"),
                             ({"name": "a/b", "revision": "main"}, "pinned to"),
                             ({"name": "a/b", "version": None}, "names its version"),
                             ({"name": "a/b", "fallback": ""}, "without it"),
                             ({"name": "a/b", "backend": "tpu"}, "backend is cuda, rocm, metal, xpu or cpu")):
            self.plugin("moe", bad)
            with self.assertRaisesRegex(PluginError, message):
                Registry.discover(self.root)

    def test_a_cpu_kernel_beside_a_device_torch_loads_its_cpu_build(self):
        calls = self.fake({"crit-labs/host-rows": object()})
        tier = self.plugin("tier", {"name": "crit-labs/host-rows", "backend": "cpu", "fallback": "torch on the CPU"})
        self.assertIsNotNone(self.kernels.registry_kernel("crit-labs/host-rows", folder=tier))
        self.assertEqual(calls[0][1], {"trust_remote_code": ["crit-labs/host-rows"], "backend": "cpu", "version": 1})

    def test_the_engine_facts_carry_the_kernels_of_its_plugins(self):
        from crit_core.serving import engine_facts
        self.fake({})
        self.kernels.registry_kernel("crit-labs/grouped-experts",
                                     folder=self.plugin("moe", {"name": "crit-labs/grouped-experts"}))
        self.kernels.registry_kernel("crit-labs/recurrent-scans",
                                     folder=self.plugin("elsewhere", {"name": "crit-labs/recurrent-scans"}))
        assembly = assemble({"use": {"drafter": "moe"}}, Registry.discover(self.root))
        facts = assembly.facts()
        self.assertEqual(list(facts["kernels"]), ["crit-labs/grouped-experts"])
        engine = type("Engine", (), {"facts": facts})()
        self.assertEqual(engine_facts(engine)["kernels"]["crit-labs/grouped-experts"]["source"], "fallback")


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
