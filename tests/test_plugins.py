"""crit plugins: manifests, a selection's slots, needs, conflicts and memory, the assembly an engine loads, each plugin's
own tests and the variants an ablation serves (offline, CPU)."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core" / "python"))
sys.path.insert(0, str(ROOT / "cli"))

from crit_plugins import PluginError, Registry, assemble, check, model_problems  # noqa: E402
from crit_plugins.assemble import plugins_root  # noqa: E402
from crit_plugins.ablation import blame, describe, engine_facts, summary, toggles, variants  # noqa: E402
from crit_plugins.testing import run_tests  # noqa: E402
from crit.plugins import main  # noqa: E402
from support.plugins import PLUGINS, write_plugins  # noqa: E402

FULL = {"use": {"attention": "attn-fp8", "kv_cache": "kv-fp8", "drafter": "draft-eagle"},
        "base": {"attention": "attn-plain", "kv_cache": "kv-bf16", "drafter": None},
        "options": {"draft-eagle": {"width": 4}}}


class PluginFixture:
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = write_plugins(Path(temp.name) / "plugins")
        self.registry = Registry.discover(self.root)

    def manifest(self, plugin_id, **changes):
        path = self.root / plugin_id / "plugin.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(changes)
        for key, value in changes.items():
            if value is None:
                del data[key]
        path.write_text(json.dumps(data), encoding="utf-8")


class LibraryPlaceTests(unittest.TestCase):
    def test_a_checkout_s_crit_uses_the_checkout_s_library(self):
        # An installed crit's is the copy its wheel ships, crit/library/inference (tests/test_wheel_install.py).
        import crit.inference
        import crit.plugins as plugins
        self.assertEqual((crit.inference.ROOT, crit.inference.LIBRARY), (ROOT, ROOT / "inference"))
        self.assertEqual(plugins.INFERENCE, ROOT / "inference")

    def test_a_crit_without_its_library_is_a_broken_install(self):
        from unittest.mock import patch
        import crit.inference
        import crit.plugins as plugins
        with tempfile.TemporaryDirectory() as root:
            missing = Path(root) / "inference"
            out = io.StringIO()
            with patch.object(plugins, "INFERENCE", missing):
                with patch.object(plugins.library, "__defaults__", (missing,)), contextlib.redirect_stderr(out):
                    self.assertEqual(main(["list"]), 1)
            with contextlib.redirect_stderr(out):
                self.assertEqual(crit.inference.main(["list"], Path(root)), 1)
        self.assertEqual(out.getvalue().count(f"crit's plugin library is missing ({missing} is no folder): this crit "
                                              "install is broken; reinstall crit"), 2, out.getvalue())


class ManifestTests(PluginFixture, unittest.TestCase):
    def test_the_fixture_plugins_load(self):
        self.assertEqual(list(self.registry.manifests), sorted(PLUGINS))
        fused = self.registry["attn-kv-fused"]
        self.assertEqual(fused["reference"], ["attn-fp8", "kv-fp8"])
        self.assertEqual(self.registry.reference_use("attn-kv-fused"), {"attention": "attn-fp8", "kv_cache": "kv-fp8"})
        self.assertEqual(self.registry["attn-plain"]["needs"], [["kv:bf16", "kv:fp8"]])

    def test_a_bad_manifest_is_refused_with_what_to_change(self):
        cases = [
            ({"id": "other"}, "id is the folder's name"),
            ({"fills": []}, "fills is a nonempty list"),
            ({"fills": ["Attention"]}, "fills holds 'Attention'"),
            ({"entry": "missing.py:create"}, "no file of the plugin"),
            ({"entry": "plugin.py"}, "entry is FILE.py:FUNCTION"),
            ({"schema": "v0"}, "schema is crit-plugin-v1"),
            ({"title": None}, "missing title"),
            ({"speed": "fast"}, "unknown speed"),
            ({"options": {"block": {"type": "int", "default": 8, "min": 16}}}, "option block is int in \\[16, inf\\]"),
            ({"options": {"mode": {"type": "choice", "default": "a"}}}, "lists its choices"),
            ({"reference": ["kv-fp8"]}, "they must fill the same slots, each once"),
            ({"reference": ["attn-plain"]}, "not its own reference"),
            ({"test_with": ["attn-plain"]}, "test_with names the other plugins its tests use"),
            ({"test_with": ["kv-nothing"]}, "its tests use kv-nothing, which is no plugin"),
            ({"sources": {"Code": "."}}, "source 'Code' needs a lowercase identifier"),
            ({"sources": {"plugin": "."}}, "source 'plugin' needs a lowercase identifier that names no file"),
            ({"sources": {"code": "../missing"}}, "source code names ../missing, which is no folder"),
            ({"sources": {"code": "C:\\code"}}, "relative folder path with forward slashes"),
            ({"provides": ["kv:fp8"], "conflicts": ["kv:fp8"]}, "both provided and in conflicts"),
        ]
        for changes, message in cases:
            with self.subTest(changes=changes):
                write_plugins(self.root)
                self.manifest("attn-plain", **changes)
                with self.assertRaisesRegex(PluginError, message):
                    Registry.discover(self.root)

    def test_a_fused_plugins_reference_fills_its_slots_once(self):
        self.manifest("attn-kv-fused", reference=["attn-fp8", "attn-plain"])
        with self.assertRaisesRegex(PluginError, "must fill the same slots, each once"):
            Registry.discover(self.root)
        self.manifest("attn-kv-fused", reference=["attn-fp8", "kv-nothing"])
        with self.assertRaisesRegex(PluginError, "kv-nothing, which is no plugin"):
            Registry.discover(self.root)

    def test_a_plugin_of_one_slot_names_the_plugin_its_tests_compare_it_with(self):
        self.manifest("attn-plain", reference=["attn-fp8"])
        registry = Registry.discover(self.root)
        self.assertEqual(registry.reference_use("attn-plain"), {"attention": "attn-fp8"})
        with self.assertRaisesRegex(PluginError, "kv-fp8 names no reference"):
            registry.reference_use("kv-fp8")

    def test_a_plugin_imports_the_code_of_its_sources_as_subpackages(self):
        code = Path(self.root.parent) / "technique" / "code"
        code.mkdir(parents=True)
        (code / "helper.py").write_text("from .shared import TWO\n\ndef double(x):\n    return TWO * x\n",
                                        encoding="utf-8")
        (code / "shared.py").write_text("TWO = 2\n", encoding="utf-8")
        self.manifest("kv-fp8", sources={"technique": "../../technique/code"})
        plugin = self.root / "kv-fp8" / "plugin.py"
        plugin.write_text(plugin.read_text(encoding="utf-8").replace(
            "        return base * self.scale", "        from .technique.helper import double\n"
                                                "        return double(base) * self.scale"), encoding="utf-8")
        registry = Registry.discover(self.root)
        self.assertEqual(registry["kv-fp8"]["sources"]["technique"]["folder"], code.resolve())
        assembly = assemble({"use": {"kv_cache": "kv-fp8"}}, registry)
        self.addCleanup(assembly.close)
        self.assertEqual(assembly["kv_cache"].value(1.5), 3.0)

    def test_a_folder_of_plugins_keeps_its_base_plugins_in_base(self):
        (self.root / "base").mkdir()
        for plugin_id in ("attn-plain", "kv-bf16"):
            (self.root / plugin_id).rename(self.root / "base" / plugin_id)
        registry = Registry.discover(self.root)
        self.assertEqual(sorted(registry.manifests), sorted(PLUGINS))
        self.assertEqual(Path(registry["kv-bf16"]["folder"]).parent.name, "base")
        for plugin_id in ("kv-bf16", "kv-fp8"):
            self.assertEqual(plugins_root(registry[plugin_id]["folder"]), self.root.resolve())
        write_plugins(self.root / "base", {"base": PLUGINS["kv-bf16"]})
        with self.assertRaisesRegex(PluginError, "base is the folder of a folder's base plugins"):
            Registry.discover(self.root)

    def test_two_plugins_of_one_id_are_refused(self):
        second = Path(self.root.parent) / "more"
        write_plugins(second, {"kv-fp8": PLUGINS["kv-fp8"]})
        with self.assertRaisesRegex(PluginError, "two plugins are named kv-fp8"):
            Registry.discover(self.root, second)


class SelectionTests(PluginFixture, unittest.TestCase):
    def test_plugins_are_created_after_the_plugins_whose_facts_they_need(self):
        use, order, options = check(FULL, self.registry, facts=["taps:hidden"])
        self.assertEqual(order, ["kv-fp8", "attn-fp8", "draft-eagle"])
        self.assertEqual(options["draft-eagle"], {"width": 4})
        self.assertEqual(use, FULL["use"])

    def test_a_need_no_plugin_or_engine_fact_provides_is_refused(self):
        with self.assertRaisesRegex(PluginError, "draft-eagle needs taps:hidden, which no other plugin"):
            check(FULL, self.registry)
        selection = {"use": {"attention": "attn-fp8", "kv_cache": "kv-bf16"}}
        with self.assertRaisesRegex(PluginError, "attn-fp8 needs kv:fp8"):
            check(selection, self.registry)
        # Any one of a need's alternatives will do.
        self.assertEqual(check({"use": {"attention": "attn-plain", "kv_cache": "kv-bf16"}}, self.registry)[1],
                         ["kv-bf16", "attn-plain"])

    def test_a_fused_plugin_fills_every_slot_it_names(self):
        with self.assertRaisesRegex(PluginError, "attn-kv-fused fills attention, kv_cache: name it in kv_cache too"):
            check({"use": {"attention": "attn-kv-fused", "kv_cache": "kv-fp8"}}, self.registry)
        with self.assertRaisesRegex(PluginError, "kv-fp8 fills kv_cache; it does not fill attention"):
            check({"use": {"attention": "kv-fp8", "kv_cache": "kv-fp8"}}, self.registry)
        use, order, _ = check({"use": {"attention": "attn-kv-fused", "kv_cache": "attn-kv-fused"}}, self.registry)
        self.assertEqual(order, ["attn-kv-fused"])

    def test_conflicts_and_two_providers_of_a_fact_are_refused(self):
        selection = {"use": {"drafter": "draft-eagle", "sampler": "sampler-host"}}
        with self.assertRaisesRegex(PluginError, "draft-eagle conflicts with sampler:host, which sampler-host provides"):
            check(selection, self.registry, facts=["taps:hidden"])
        self.manifest("attn-fp8", provides=["kv:fp8"])
        registry = Registry.discover(self.root)
        with self.assertRaisesRegex(PluginError, "both provide kv:fp8"):
            check({"use": {"attention": "attn-fp8", "kv_cache": "kv-fp8"}}, registry)

    def test_options_and_the_selection_are_checked(self):
        cases = [
            ({**FULL, "options": {"draft-eagle": {"width": 99}}}, "option width is int in \\[1, 16\\], not 99"),
            ({**FULL, "options": {"draft-eagle": {"depth": 2}}}, "draft-eagle has no option depth \\(it has width\\)"),
            ({**FULL, "options": {"attn-kv-fused": {}}}, "options names attn-kv-fused, which the selection does not use"),
            ({**FULL, "base": {"router": None}}, "base names router, which plugins.use does not"),
            ({**FULL, "base": {"attention": "kv-bf16"}}, "plugins.base.attention: kv-bf16 does not fill attention"),
            ({"use": {"attention": "flash"}}, "no plugin flash"),
            ({"use": {"attention": "attn-fp8"}, "other": {}}, "unknown other"),
        ]
        for selection, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(PluginError, message):
                check(selection, self.registry, facts=["taps:hidden"])

    def test_a_cycle_of_needs_is_refused(self):
        self.manifest("kv-fp8", needs=["attn:x"])
        self.manifest("attn-fp8", provides=["attn:x"])
        with self.assertRaisesRegex(PluginError, "need each other in a cycle: kv-fp8 -> attn-fp8 -> kv-fp8"):
            check({"use": {"kv_cache": "kv-fp8", "attention": "attn-fp8"}}, Registry.discover(self.root))


class AssemblyTests(PluginFixture, unittest.TestCase):
    def test_the_assembly_creates_each_plugin_once_with_the_ones_it_needs(self):
        context = {}
        assembly = assemble(FULL, self.registry, context, facts=["taps:hidden"])
        self.assertEqual(context["created"], ["kv-fp8", "attn-fp8", "draft-eagle"])
        self.assertIs(assembly["attention"].kv, assembly["kv_cache"])
        self.assertEqual(assembly["attention"].value(3), 6.0)
        self.assertEqual(assembly["drafter"].options, {"width": 4})
        self.assertEqual(assembly.facts(), {"plugins": FULL["use"]})
        self.assertEqual(assembly.memory["kv-fp8"], {"device": 1 << 30, "host": 0})
        fused = assemble({"use": {"attention": "attn-kv-fused", "kv_cache": "attn-kv-fused"}}, self.registry, {})
        self.assertIs(fused["attention"], fused["kv_cache"])
        with self.assertRaises(KeyError):
            fused["drafter"]
        self.assertIsNone(fused.get("drafter"))

    def test_memory_over_the_budget_is_refused_before_any_plugin_is_created(self):
        context = {}
        selection = {"use": {"attention": "attn-plain", "kv_cache": "kv-bf16"}}
        with self.assertRaisesRegex(PluginError, "need 2.00 GiB of device memory, the budget is 1.50 GiB \\(kv-bf16 "
                                                 "2.00 GiB\\)"):
            assemble(selection, self.registry, context, budget={"device": 3 << 29})
        self.assertNotIn("created", context)
        assemble(selection, self.registry, context, budget={"device": 2 << 30})

    def test_a_failed_creation_closes_the_plugins_already_created(self):
        closed = []
        selection = {"use": {"attention": "attn-plain", "kv_cache": "kv-fp8"},
                     "options": {"attn-plain": {"fail_create": True}}}
        from crit_plugins.assemble import load_module
        module = load_module(self.registry["kv-fp8"])
        original = module.Plugin.close
        module.Plugin.close = lambda plugin: closed.append("kv-fp8")
        self.addCleanup(setattr, module.Plugin, "close", original)
        with self.assertRaisesRegex(RuntimeError, "attn-plain failed to create"):
            assemble(selection, self.registry, {})
        self.assertEqual(closed, ["kv-fp8"])

    def test_close_runs_every_plugins_close_and_raises_the_first_error(self):
        assembly = assemble({"use": {"attention": "attn-plain", "kv_cache": "kv-fp8"},
                             "options": {"attn-plain": {"fail_close": True}}}, self.registry, {})
        with self.assertRaisesRegex(RuntimeError, "attn-plain failed to close"):
            assembly.close()
        self.assertEqual(assembly.instances, {})


# A whole engine's plugins: a model whose architecture uses a recurrence and provides a draft head, and a plugin in
# every slot an engine fills, with host references and a plugin for another device beside them.
ENGINE = {
    "model-x": {"fills": ["model"], "needs": [["model_type:x", "model_type:x_text"]], "requires": ["recurrence"],
                "provides": ["draft:head"], "devices": ["cuda"]},
    "place": {"fills": ["placement"], "devices": ["host"]},
    "proj": {"fills": ["projections"], "devices": ["cuda"]},
    "attn": {"fills": ["attention"], "devices": ["cuda", "rocm"]},
    "attn-rocm": {"fills": ["attention"], "devices": ["rocm"]},
    "kv": {"fills": ["kv_cache"], "devices": ["cuda"]},
    "samp": {"fills": ["sampler"], "devices": ["cuda"]},
    "rec": {"fills": ["recurrence"], "devices": ["cuda"]},
    "rec-reference": {"fills": ["recurrence"]},
    "draft": {"fills": ["drafter"], "needs": ["draft:head"], "devices": ["host"]},
}
WHOLE = {"use": {"model": "model-x", "placement": "place", "projections": "proj", "attention": "attn",
                 "kv_cache": "kv", "sampler": "samp", "recurrence": "rec", "drafter": "draft"}}


class EngineSelectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = write_plugins(Path(temp.name) / "plugins", ENGINE)
        self.registry = Registry.discover(self.root)

    def selection(self, **use):
        return {"use": {**WHOLE["use"], **use}}

    def test_the_model_is_created_last_after_every_slot_its_step_calls(self):
        # The drafter needs the model's draft head, a fact of the engine's: it orders nothing.
        use, order, _ = check(WHOLE, self.registry, facts=["model_type:x"], device="cuda")
        self.assertEqual(order[-1], "model-x")
        self.assertEqual(sorted(order[:-1]), sorted(set(WHOLE["use"].values()) - {"model-x"}))
        context = {}
        assembly = assemble(WHOLE, self.registry, context, facts=["model_type:x_text"], device="cuda")
        self.addCleanup(assembly.close)
        self.assertEqual(context["created"][-1], "model-x")

    def test_a_need_that_comes_with_a_fact_holds_only_where_the_fact_does(self):
        # A plugin that decodes a stored format only with a device feature: needs_when.
        root = write_plugins(self.root.parent / "when", {**ENGINE, "proj": {**ENGINE["proj"],
                                                                           "needs_when": {"stored:nvfp4": ["cuda:fp4"]}}})
        registry = Registry.discover(root)
        self.assertEqual(registry["proj"]["needs_when"], {"stored:nvfp4": [["cuda:fp4"]]})
        check(WHOLE, registry, facts=["model_type:x"], device="cuda")
        with self.assertRaisesRegex(PluginError, "proj needs cuda:fp4 where the engine provides stored:nvfp4"):
            check(WHOLE, registry, facts=["model_type:x", "stored:nvfp4"], device="cuda")
        check(WHOLE, registry, facts=["model_type:x", "stored:nvfp4", "cuda:fp4"], device="cuda")
        from crit_plugins.assemble import fit, fitting
        self.assertEqual(fit(registry["proj"], ["stored:nvfp4"], "cuda", registry), ["needs cuda:fp4"])
        self.assertEqual(fit(registry["attn-rocm"], [], "cuda", registry), ["runs on rocm"])
        self.assertEqual(fit(registry["rec-reference"], [], "cuda", registry)[0][:16], "a host reference")
        # The drafter's draft head is the model plugin's: another plugin of the library provides it, so it fits.
        self.assertEqual(fit(registry["draft"], [], "cuda", registry), [])
        found = fitting(registry, ["model_type:x"], "cuda")
        self.assertEqual(found["attention"], [("attn", []), ("attn-rocm", ["runs on rocm"])])
        self.assertEqual(found["embeddings"], [])

    def test_a_plugin_on_a_device_speaks_its_slots_device_form(self):
        # A host form on a device is an oracle no engine serves: assembling on cuda refuses it as it is created.
        root = write_plugins(self.root.parent / "host-only", {**ENGINE, "proj": {**ENGINE["proj"], "device_form": False}})
        context = {}
        with self.assertRaisesRegex(PluginError, "proj in projections lacks load, matmul of the projections device "
                                                 "form: an engine on cuda serves only plugins that compute there"):
            assemble(WHOLE, Registry.discover(root), context, facts=["model_type:x"], device="cuda")
        self.assertNotIn("model-x", context["created"])

    def test_the_model_runs_only_on_the_architecture_it_names(self):
        with self.assertRaisesRegex(PluginError, "model-x needs model_type:x or model_type:x_text"):
            check(WHOLE, self.registry, facts=["model_type:llama"], device="cuda")

    def test_an_engine_fills_every_required_slot(self):
        for slot in ("placement", "projections", "attention", "kv_cache", "sampler"):
            with self.subTest(slot=slot), self.assertRaisesRegex(PluginError, f"leaves {slot} empty"):
                check(self.selection(**{slot: None}), self.registry, facts=["model_type:x"], device="cuda")
        selection = {"use": {k: v for k, v in WHOLE["use"].items() if k not in ("model", "drafter")}}
        with self.assertRaisesRegex(PluginError, "an engine fills model, placement, .*leaves model empty"):
            check(selection, self.registry, device="cuda")

    def test_a_plugin_requires_the_slots_its_step_calls(self):
        with self.assertRaisesRegex(PluginError, "model-x requires recurrence, which the selection leaves empty"):
            check(self.selection(recurrence=None), self.registry, facts=["model_type:x"], device="cuda")
        # Also in a part of an engine: the requirement is the plugin's own.
        with self.assertRaisesRegex(PluginError, "model-x requires recurrence"):
            check({"use": {"model": "model-x"}}, self.registry, facts=["model_type:x"])

    def test_an_engine_serves_no_host_reference_and_no_plugin_of_another_device(self):
        with self.assertRaisesRegex(PluginError, "rec-reference is a host reference, the oracle device plugins are "
                                                 "tested against: serve a plugin that runs on cuda in recurrence"):
            check(self.selection(recurrence="rec-reference"), self.registry, facts=["model_type:x"], device="cuda")
        with self.assertRaisesRegex(PluginError, "attn-rocm runs on rocm, not on cuda"):
            check(self.selection(attention="attn-rocm"), self.registry, facts=["model_type:x"], device="cuda")
        # A policy plugin runs on the host beside any device; a part of an engine may hold host references.
        check(self.selection(attention="attn-rocm", projections="proj"), self.registry, facts=["model_type:x"])
        check({"use": {"recurrence": "rec-reference"}}, self.registry)

    def test_a_model_or_device_manifest_that_breaks_the_rules_is_refused(self):
        cases = [
            ("model-x", {"fills": ["model", "attention"]}, "a model plugin fills only model"),
            ("model-x", {"requires": ["model"]}, "requires names the other slots the plugin calls, not its own"),
            ("attn", {"devices": ["tpu"]}, "devices holds tpu"),
            ("attn", {"devices": ["host"]}, "attention computes on the device the engine serves on"),
        ]
        for plugin_id, changes, message in cases:
            with self.subTest(message=message):
                path = self.root / plugin_id / "plugin.json"
                saved = path.read_text(encoding="utf-8")
                self.addCleanup(path.write_text, saved, encoding="utf-8")
                path.write_text(json.dumps({**json.loads(saved), **changes}), encoding="utf-8")
                with self.assertRaisesRegex(PluginError, message):
                    Registry.discover(self.root)
                path.write_text(saved, encoding="utf-8")

    def test_core_runs_a_model_that_batches(self):
        class Model:
            def __init__(self, **info):
                self._info = info

            def info(self):
                return self._info

        for name in ("open", "step", "commit", "retire", "score"):
            setattr(Model, name, lambda self, *args: None)
        self.assertEqual(model_problems(Model(vocabulary=32, context=64, max_concurrency=4)), [])
        self.assertEqual(model_problems(Model(vocabulary=32, context=64, max_concurrency=1)),
                         ["info() declares max_concurrency 1: every engine batches, so its step runs at least 2 rows "
                          "of different requests together"])
        self.assertEqual(model_problems(Model(context=64, max_concurrency=2)), ["info() names no vocabulary"])
        del Model.score
        self.assertEqual(model_problems(Model(vocabulary=32, context=64, max_concurrency=4)),
                         ["it lacks score of the model interface"])


class DeviceFormTests(unittest.TestCase):
    def test_a_gpu_plugin_has_its_slots_device_form_beside_the_host_form(self):
        from crit_plugins.device import DEVICE_INTERFACES, device_missing

        class Attention:
            def attend(self, sequence, queries, positions, scale=None, pending=False):
                return []

        self.assertEqual(device_missing("attention", Attention()), ["forward"])
        Attention.forward = lambda self, layer, queries, pool, batch, scale=None: queries
        self.assertEqual(device_missing("attention", Attention()), [])
        self.assertEqual(device_missing("placement", object()), [])   # a policy slot has no device form
        for slot in ("projections", "kv_cache", "attention", "recurrence", "sampler"):
            self.assertIn(slot, DEVICE_INTERFACES)

    def test_a_kv_pool_names_what_the_attention_slot_reads(self):
        # softmax-attention reads gather() and value_dim of whatever kv_cache plugin's pool it is given.
        from crit_plugins.device import KV_POOL, pool_missing

        class Pool:
            kv_heads, head_dim = 2, 8

        for name in set(KV_POOL) - {"gather"}:
            setattr(Pool, name, lambda self, *args: None)
        self.assertEqual(pool_missing(Pool()), ["gather", "value_dim"])
        Pool.gather, Pool.value_dim = (lambda self, layer, sequence, count: (None, None)), 8
        self.assertEqual(pool_missing(Pool()), [])

    def test_a_step_batch_packs_rows_of_one_sequence_each(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("a step batch's tensors are torch's")
        from crit_plugins import StepBatch
        batch = StepBatch.build([("a", 0, 3), ("b", 7, 1), ("c", 4, 2)], "cpu")
        self.assertEqual((batch.rows, batch.tokens, batch.starts), (3, 6, [0, 3, 4, 6]))
        self.assertEqual(batch.positions, [0, 1, 2, 7, 4, 5])
        self.assertEqual(batch.rows_t.tolist(), [0, 0, 0, 1, 2, 2])
        self.assertEqual(batch.row(2), ("c", 4, 2, 4))
        with self.assertRaisesRegex(ValueError, "one or more tokens"):
            StepBatch.build([("a", 0, 0)], "cpu")


PASSING = '''from crit_plugins.testing import PluginTestCase


class FusedTests(PluginTestCase):
    def test_it_equals_its_reference(self):
        fused, reference = self.assemble(), self.reference()
        for x in (0.5, 1.0, 3.0):
            self.assertClose(fused["attention"].value(x), reference["attention"].value(x))
'''
FAILING = '''from crit_plugins.testing import PluginTestCase


class PlainTests(PluginTestCase):
    def test_bias(self):
        attention = self.assemble({"kv_cache": "kv-bf16"}, {"bias": 0.5})["attention"]
        self.assertClose(attention.value(1.0), 2.0)
'''


class PluginTestRunnerTests(unittest.TestCase):
    def test_each_plugin_runs_its_own_tests_and_one_without_tests_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_plugins(Path(temp) / "plugins", {k: PLUGINS[k] for k in
                                                          ("attn-fp8", "kv-fp8", "attn-kv-fused", "attn-plain", "kv-bf16")},
                                 tests={"attn-kv-fused": {"test_fused.py": PASSING},
                                        "attn-plain": {"test_plain.py": FAILING}})
            items = {i["name"]: i for i in run_tests(root, log=lambda line: None)}
        self.assertEqual(items["attn-kv-fused"]["state"], "passed", items["attn-kv-fused"])
        self.assertEqual(items["attn-kv-fused"]["detail"], "1 tests")
        self.assertEqual(items["attn-plain"]["state"], "failed")
        self.assertIn("2.5 differs from 2.0", items["attn-plain"]["detail"])
        self.assertEqual(items["kv-fp8"]["state"], "failed")
        self.assertIn("no tests", items["kv-fp8"]["detail"])


class PluginTestCacheTests(unittest.TestCase):
    def test_a_plugin_whose_tests_passed_unchanged_is_reported_cached_and_a_change_runs_them_again(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = write_plugins(Path(temp) / "plugins", {k: PLUGINS[k] for k in
                                                          ("attn-fp8", "kv-fp8", "attn-kv-fused", "attn-plain", "kv-bf16")},
                                 tests={"attn-kv-fused": {"test_fused.py": PASSING},
                                        "attn-plain": {"test_plain.py": FAILING}})
            with patch.dict("os.environ", {"CRIT_PLUGIN_TEST_CACHE": str(Path(temp) / "cache")}):
                first = {i["name"]: i for i in run_tests(root, only={"attn-kv-fused", "attn-plain"}, cache=True,
                                                         log=lambda line: None)}
                again = {i["name"]: i for i in run_tests(root, only={"attn-kv-fused", "attn-plain"}, cache=True,
                                                         log=lambda line: None)}
                self.assertNotIn("cached", first["attn-kv-fused"])
                self.assertTrue(again["attn-kv-fused"]["cached"])
                self.assertEqual(again["attn-kv-fused"]["detail"], "1 tests")
                # A failure is never cached, and a changed file runs the tests again.
                self.assertEqual(again["attn-plain"]["state"], "failed")
                folder = Path(temp) / "plugins" / "attn-kv-fused"
                (folder / "notes.txt").write_text("changed", encoding="utf-8")
                changed = {i["name"]: i for i in run_tests(root, only={"attn-kv-fused"}, cache=True,
                                                           log=lambda line: None)}
                self.assertNotIn("cached", changed["attn-kv-fused"])


class AblationTests(PluginFixture, unittest.TestCase):
    def test_variants_switch_each_plugin_off_and_on_alone(self):
        self.assertEqual(toggles(FULL), ["attn-fp8", "kv-fp8", "draft-eagle"])
        found = {v["name"]: v for v in variants(FULL, self.registry, facts=["taps:hidden"])}
        self.assertEqual(list(found), ["full", "base", "without:attn-fp8", "without:kv-fp8", "without:draft-eagle",
                                       "only:attn-fp8", "only:kv-fp8", "only:draft-eagle"])
        self.assertEqual(found["base"]["plugins"]["use"], {"attention": "attn-plain", "kv_cache": "kv-bf16",
                                                            "drafter": None})
        self.assertEqual(found["without:draft-eagle"]["plugins"]["use"]["drafter"], None)
        # A variant without a plugin leaves out its options, so the engine's own assembly takes the variant.
        self.assertNotIn("draft-eagle", found["without:draft-eagle"]["plugins"]["options"])
        self.assertEqual(found["full"]["plugins"]["options"], {"draft-eagle": {"width": 4}})
        for variant in found.values():
            if "error" not in variant:
                check(variant["plugins"], self.registry, facts=["taps:hidden"])
        # attn-fp8 reads FP8 KV: without the FP8 cache, or on its own over the plain cache, it has none.
        self.assertIn("attn-fp8 needs kv:fp8", found["without:kv-fp8"]["error"])
        self.assertIn("attn-fp8 needs kv:fp8", found["only:attn-fp8"]["error"])
        self.assertNotIn("error", found["only:kv-fp8"])
        # crit run ablate names no facts: the engine's own are what the candidate's plugins need and none provides.
        self.assertEqual(engine_facts(FULL, self.registry), ["taps:hidden"])
        inferred = {v["name"]: v.get("error") for v in variants(FULL, self.registry)}
        self.assertEqual(inferred, {name: v.get("error") for name, v in found.items()})
        self.assertIn("draft-eagle needs taps:hidden", variants(FULL, self.registry, facts=[])[0]["error"])
        one = {"use": {"attention": "attn-kv-fused", "kv_cache": "attn-kv-fused"},
               "base": {"attention": "attn-plain", "kv_cache": "kv-bf16"}}
        self.assertEqual([v["name"] for v in variants(one, self.registry)], ["full", "base"])
        self.assertEqual([v["name"] for v in variants(FULL, names={"full", "base"})], ["full", "base"])
        with self.assertRaisesRegex(PluginError, "no variant only:x"):
            variants(FULL, names={"only:x"})

    def test_blame_names_what_each_plugin_adds_costs_and_breaks(self):
        def measured(name, decode, prefill, failed=()):
            return {"name": name, "state": "measured", "kpis": {"decode_tps": decode, "prefill_tps": prefill,
                                                                 "ttft_ms": 1000 / prefill},
                    "smoke": {"failed": list(failed)}}
        report = [measured("full", 150, 900, ["causal"]), measured("base", 100, 1000),
                  measured("without:attn-fp8", 150, 1000), measured("without:kv-fp8", 140, 905, ["causal"]),
                  measured("without:draft-eagle", 101, 899, ["causal"]),
                  measured("only:attn-fp8", 99, 905, ["causal"]), measured("only:kv-fp8", 101, 1001),
                  measured("only:draft-eagle", 149, 1000)]
        rows = {row["plugin"]: row for row in blame(report, FULL)}
        self.assertEqual(rows["attn-fp8"]["verdict"], "breaks")
        self.assertEqual(rows["attn-fp8"]["breaks"], ["causal"])
        self.assertAlmostEqual(rows["attn-fp8"]["in_engine"]["prefill_tps"], -0.1)
        self.assertEqual(rows["draft-eagle"]["verdict"], "pays")
        self.assertAlmostEqual(rows["draft-eagle"]["in_engine"]["decode_tps"], 150 / 101 - 1)
        self.assertAlmostEqual(rows["draft-eagle"]["alone"]["decode_tps"], 0.49)
        self.assertEqual(rows["kv-fp8"]["verdict"], "pays")
        self.assertIn("attn-fp8: breaks; in the engine decode +0.0%, prefill -10.0%", describe(rows["attn-fp8"]))
        self.assertIn("fails causal with it and passes without it", describe(rows["attn-fp8"]))
        # With one toggle, without:ID is the base engine itself.
        one = {"use": {"attention": "attn-fp8"}, "base": {"attention": "attn-plain"}}
        self.assertEqual(variants(one)[1]["also"], ["without:attn-fp8"])
        slow = [measured("full", 90, 1000), {**measured("base", 100, 1000), "also": ["without:attn-fp8"]}]
        (row,) = blame(slow, one)
        self.assertEqual(row["verdict"], "slows")
        self.assertAlmostEqual(row["alone"]["decode_tps"], -0.1)
        # Better on one KPI and worse on another beyond the noise: a trade the plan has to weigh.
        mixed = [measured("full", 150, 800), {**measured("base", 100, 1000), "also": ["without:attn-fp8"]}]
        (row,) = blame(mixed, one)
        self.assertEqual(row["verdict"], "trades")

    def test_blame_compares_the_round_time_where_both_variants_draft(self):
        # A Qwen3.8-27B engine's own projections decode 1.9% faster than cuBLASLt's at 3.20 tokens a round against
        # 3.76, which is a round 20% faster.
        def drafting(name, decode, accepted):
            kpis = summary({"case-0001": {"decode_tps": decode, "prefill_tps": 8000.0, "ttft_ms": 62.0,
                                          "accepted_tokens_per_round": accepted}})
            return {"name": name, "state": "measured", "kpis": kpis, "smoke": {"failed": []}}
        full = drafting("full", 154.3, 3.20)
        self.assertAlmostEqual(full["kpis"]["round_ms"], 1000 * 3.20 / 154.3)
        self.assertEqual(full["kpis"]["accepted_tokens_per_round"], 3.20)
        one = {"use": {"attention": "attn-fp8"}, "base": {"attention": "attn-plain"}}
        (row,) = blame([full, {**drafting("base", 151.4, 3.76), "also": ["without:attn-fp8"]}], one)
        self.assertAlmostEqual(row["in_engine"]["decode_tps"], 154.3 / 151.4 - 1)
        self.assertAlmostEqual(row["in_engine"]["round_ms"], (3.76 / 151.4) / (3.20 / 154.3) - 1)
        self.assertEqual(row["verdict"], "pays")
        self.assertIn("decode +1.9%, prefill +0.0%, ttft +0.0%, round +19.8%", describe(row))
        # A drafter against none: its rounds are longer by design, so only the tokens per second count.
        (row,) = blame([full, {**drafting("base", 67.1, 1.01), "also": ["without:attn-fp8"]}], one)
        self.assertNotIn("round_ms", row["in_engine"])
        self.assertEqual(row["verdict"], "pays")
        self.assertIsNone(summary({"case-0001": {"decode_tps": 100.0}})["round_ms"])

    def test_where_both_variants_draft_a_plugin_outside_the_drafter_is_judged_by_its_round_time(self):
        # A Qwen3.8-27B engine's recurrence plugin: decode 11.6% lower at 2.6 tokens a round against 2.9, in rounds
        # 0.3% longer: the tokens the short bench accepted moved, the plugin's own cost did not.
        def drafting(name, decode, accepted):
            kpis = summary({"case-0001": {"decode_tps": decode, "prefill_tps": 6000.0, "ttft_ms": 88.0,
                                          "accepted_tokens_per_round": accepted}})
            return {"name": name, "state": "measured", "kpis": kpis, "smoke": {"failed": []}}
        recurrence = {"use": {"recurrence": "scan-two-pass"}, "base": {"recurrence": "scan-plain"}}
        full, base = drafting("full", 132.6, 2.57), drafting("base", 150.0, 2.9)
        (row,) = blame([full, {**base, "also": ["without:scan-two-pass"]}], recurrence)
        self.assertLess(row["in_engine"]["decode_tps"], -0.1)
        self.assertGreater(row["in_engine"]["round_ms"], -0.01)
        self.assertEqual(row["judged"], ["prefill_tps", "round_ms", "ttft_ms"])
        self.assertEqual(row["verdict"], "neutral")
        # The drafter's own toggle is judged by the tokens per second its drafts give, rounds of any length.
        drafter = {"use": {"drafter": "draft-wide"}, "base": {"drafter": "draft-narrow"}}
        (row,) = blame([full, {**base, "also": ["without:draft-wide"]}], drafter)
        self.assertEqual(row["judged"], ["decode_tps", "prefill_tps", "ttft_ms"])
        self.assertEqual(row["verdict"], "slows")


class CliTests(PluginFixture, unittest.TestCase):
    def run_cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            code = main([str(a) for a in args])
        return code, out.getvalue() + err.getvalue()

    def test_list_check_and_variants(self):
        code, out = self.run_cli("list", self.root)
        self.assertEqual(code, 0)
        self.assertIn("attn-kv-fused", out)
        self.assertIn("fused; reference attn-fp8, kv-fp8", out)
        selection = Path(self.root.parent) / "configuration.json"
        selection.write_text(json.dumps({"plugins": FULL}), encoding="utf-8")
        code, out = self.run_cli("check", self.root, "--selection", selection, "--fact", "taps:hidden")
        self.assertEqual(code, 0, out)
        self.assertIn("created in the order kv-fp8, attn-fp8, draft-eagle", out)
        code, out = self.run_cli("check", self.root, "--selection", selection)
        self.assertEqual(code, 1)
        self.assertIn("draft-eagle needs taps:hidden", out)
        code, out = self.run_cli("variants", self.root, "--selection", selection, "--fact", "taps:hidden")
        self.assertEqual(code, 0, out)
        self.assertIn("without:kv-fp8", out)
        self.assertIn("(refused: attn-fp8 needs kv:fp8", out)
        code, out = self.run_cli("variants", self.root, "--selection", selection)
        self.assertEqual(code, 0, out)
        self.assertNotIn("draft-eagle needs", out)


if __name__ == "__main__":
    unittest.main()
