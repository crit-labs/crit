"""Plugins built on library plugins (crit_plugins.compose, the manifest's extends): an engine's fused routing plugin
subclasses topk-router instead of copying it. In an engine that crit plugins add gave topk-router, the extension's
manifest takes topk-router's options beside its own, its code imports topk-router's modules as topk-router does (its
own sources nested), it selects bit for bit as topk-router's torch form over random logits with ties and NaN rows,
leaves the other rules and the host reference to topk-router's code, and a wrapper creates topk-router's object with
the options it declares. A manifest that extends no plugin's folder, or goes round in a circle, is refused."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core" / "python"))
sys.path.insert(0, str(ROOT / "cli"))

from crit_plugins import PluginError, Registry, assemble  # noqa: E402
from crit_plugins.assemble import load_module  # noqa: E402
from crit_plugins.manifest import load_manifest  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover - torch is in crit's test environment
    torch = None

FUSED = ROOT / "tests" / "support" / "extending_plugins" / "fused-router"


class Engine:
    """A temporary engine's plugins folder: topk-router added from the library, the fused extension beside it."""

    def setUp(self):
        from crit.plugins import add
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.plugins = Path(temp.name) / "plugins"
        add(self.plugins, ["topk-router"])
        shutil.copytree(FUSED, self.plugins / "fused-router")
        self.registry = Registry.discover(self.plugins)

    def router(self, plugin_id, **options):
        assembly = assemble({"use": {"router": plugin_id}, "options": {plugin_id: options}}, self.registry)
        self.addCleanup(assembly.close)
        return assembly["router"]


class ExtendsTests(Engine, unittest.TestCase):
    def test_the_extension_takes_its_bases_options_beside_its_own(self):
        base, fused = self.registry["topk-router"], self.registry["fused-router"]
        self.assertEqual(fused["extends"], {"source": "base", "id": "topk-router"})
        self.assertEqual(set(fused["options"]), set(base["options"]) | {"fused"})
        self.assertEqual(fused["options"]["scoring"], base["options"]["scoring"])
        self.assertEqual(fused["provides"], [], "the base's facts are not the extension's")

    def test_its_own_option_replaces_a_bases_of_the_same_name(self):
        data = json.loads((self.plugins / "fused-router" / "plugin.json").read_text(encoding="utf-8"))
        data["options"]["scaling"] = {"type": "float", "default": 2.0}
        (self.plugins / "fused-router" / "plugin.json").write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(load_manifest(self.plugins / "fused-router")["options"]["scaling"]["default"], 2.0)

    def test_a_manifest_extends_a_plugins_folder_among_its_sources(self):
        folder = self.plugins / "fused-router"
        data = json.loads((folder / "plugin.json").read_text(encoding="utf-8"))

        def refused(change, message):
            (folder / "plugin.json").write_text(json.dumps({**data, **change}), encoding="utf-8")
            with self.assertRaisesRegex(PluginError, message):
                load_manifest(folder)
        refused({"extends": "other"}, "extends names one of the plugin's sources")
        refused({"sources": {**data["sources"], "code": "../_sources/heterogeneous-expert-dispatch/code"},
                 "extends": "code"}, "holds no plugin.json")
        refused({"extends": ["base"]}, "extends names one of the plugin's sources")
        # topk-router extending its extension: a circle
        base = self.plugins / "base" / "topk-router" / "plugin.json"
        original = json.loads(base.read_text(encoding="utf-8"))
        base.write_text(json.dumps({**original, "sources": {**original["sources"], "ext": "../../fused-router"},
                                    "extends": "ext"}), encoding="utf-8")
        refused({}, "circle")

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_the_fused_selection_is_topk_routers_bit_for_bit(self):
        fused = self.router("fused-router", kernel=False)
        plain = self.router("topk-router", kernel=False)
        self.assertEqual(type(fused).__name__, "FusedRouter")
        self.assertIn("TopKRouter", [c.__name__ for c in type(fused).__mro__], "a subclass, not a copy")
        g = torch.Generator().manual_seed(5)
        for tokens, experts, top in ((1, 8, 2), (7, 16, 4), (5, 64, 8)):
            logits = torch.randn(tokens, experts, generator=g)
            logits[0, :3] = logits[0, 3]                 # ties go to the lower id
            logits[-1] = float("nan")                    # a junk row still routes, weights 0
            if tokens > 2:
                logits[1, 2] = float("-inf")
            for scaling in (1.0, 2.5):
                fused.routing = plain.routing = replace(plain.routing, scaling=scaling)
                with mock.patch.object(type(fused).__mro__[1], "select", side_effect=AssertionError("not fused")):
                    ids, weights = fused.select(logits, top)
                want_ids, want_weights = plain.select(logits, top)
                self.assertTrue(torch.equal(ids, want_ids), (tokens, experts, top))
                self.assertTrue(torch.equal(weights, want_weights), (tokens, experts, top))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_other_rules_and_the_host_reference_stay_topk_routers(self):
        fused = self.router("fused-router", kernel=False, scoring="sigmoid", score_before_selection=True)
        plain = self.router("topk-router", kernel=False, scoring="sigmoid", score_before_selection=True)
        logits = torch.randn(4, 16, generator=torch.Generator().manual_seed(1))
        bias = torch.linspace(0, 1, 16)
        for got, want in zip(fused.select(logits, 3, bias), plain.select(logits, 3, bias)):
            self.assertTrue(torch.equal(got, want))
        self.assertEqual(fused.route(logits.tolist(), 3), plain.route(logits.tolist(), 3))

    def test_a_wrapper_creates_its_base_with_the_options_the_base_declares(self):
        from crit_plugins.compose import base_manifest, create_base
        module = load_module(self.registry["fused-router"])
        self.assertEqual(base_manifest(module.__name__)["id"], "topk-router")
        inner = create_base(module.__name__, None, {"fused": True, "scoring": "sigmoid", "kernel": False}, {})
        self.assertEqual((type(inner).__name__, inner.routing.scoring), ("TopKRouter", "sigmoid"))
        with self.assertRaisesRegex(PluginError, "no plugin's folder"):
            create_base(module, None, {}, {}, source="nothing")
        with self.assertRaisesRegex(PluginError, "no module of a plugin"):
            base_manifest(__name__)

    def test_crit_plugins_add_brings_the_plugin_an_extension_extends(self):
        from crit.plugins import closure
        registry = Registry.discover(self.plugins)
        self.assertEqual(closure(registry, ["fused-router"]),
                         [("fused-router", "named"), ("topk-router", "the reference of fused-router")])
        data = json.loads((self.plugins / "fused-router" / "plugin.json").read_text(encoding="utf-8"))
        del data["reference"]
        (self.plugins / "fused-router" / "plugin.json").write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(closure(Registry.discover(self.plugins), ["fused-router"]),
                         [("fused-router", "named"), ("topk-router", "fused-router extends it")])


if __name__ == "__main__":
    unittest.main()
