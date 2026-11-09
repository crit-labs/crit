"""crit's library of plugins, its Inference folder: every technique that fills a slot is one plugin of its id and passes
its own tests against its reference, the base plugins (inference/base) give each slot its plain plugin, every entry is a
plugin, lends one its code or is an engine service, and crit plugins add copies plugins into an engine where their tests
pass again (offline, CPU)."""
import ast
import contextlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core" / "python"))
sys.path.insert(0, str(ROOT / "cli"))

from crit_plugins import SLOTS, PluginError, Registry  # noqa: E402
from crit_plugins.slots import CASES, INTERFACES  # noqa: E402
from crit_plugins.testing import run_tests  # noqa: E402
from crit.plugins import ENGINE_SERVICES, SOURCES, add, library, main  # noqa: E402

INFERENCE = ROOT / "inference"
ENTRIES = sorted(p.parent.name for p in INFERENCE.glob("*/entry.json"))
BASE = ["chain-verifier", "device-placement", "eager-graphs", "expert-tensors", "experts-reference", "greedy-head-drafter",
        "memory-table-rows", "projections-dense", "sequential-recurrence", "softmax-attention", "topk-router",
        "torch-norms"]


def kernel_requests(paths):
    """What Python modules can ask crit_plugins.kernels.registry_kernel for: whether any calls it, and the string
    constants of the modules' code (not their docstrings or comments), where a kernel's name must be."""
    calls, strings = False, set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                      and node.body and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                calls |= (f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)) == "registry_kernel"
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                strings.add(node.value)
    return calls, strings


def tests_of(manifest):
    """A plugin's test code, less the lines that only expect an error message."""
    return "\n".join(line for path in sorted(Path(manifest["tests"]).glob("*.py"))
                     for line in path.read_text(encoding="utf-8").splitlines() if "assertRaisesRegex" not in line)


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = library()

    def test_every_library_plugin_passes_its_own_tests(self):
        items = run_tests(self.registry, log=lambda line: None)
        failed = {i["name"]: i["detail"] for i in items if i["state"] != "passed"}
        self.assertEqual(failed, {})
        self.assertEqual(len(items), len(self.registry))

    def test_every_plugin_fills_slots_crit_names_and_is_held_to_their_interface(self):
        for manifest in self.registry:
            with self.subTest(plugin=manifest["id"]):
                self.assertTrue(set(manifest["fills"]) <= set(SLOTS), manifest["fills"])
                self.assertTrue(set(manifest["fills"]) <= set(INTERFACES), manifest["fills"])
                self.assertIn("assertImplements", tests_of(manifest))
                self.assertTrue(manifest["description"])

    def test_a_plugin_with_a_reference_is_compared_with_it(self):
        compared = [m for m in self.registry if m["reference"]]
        self.assertGreater(len(compared), 10)
        for manifest in compared:
            with self.subTest(plugin=manifest["id"]):
                self.assertRegex(tests_of(manifest), r"assertMatchesReference\(|self\.reference\(")
        for slot in CASES:
            with self.subTest(slot=slot):
                self.assertTrue(any(slot in m["fills"] for m in self.registry), f"no library plugin fills {slot}")

    def test_test_with_names_every_other_library_plugin_a_plugins_tests_use(self):
        ids = set(self.registry.manifests)
        for manifest in self.registry:
            with self.subTest(plugin=manifest["id"]):
                named = {m for m in re.findall(r'"([a-z0-9][a-z0-9-]*)"', tests_of(manifest)) if m in ids}
                expected = named - {manifest["id"]} - set(manifest["reference"] or [])
                self.assertEqual(set(manifest["test_with"]), expected)

    def test_a_technique_is_its_entry_and_builds_on_its_own_code(self):
        for manifest in self.registry:
            folder = Path(manifest["folder"])
            if folder.parent.name == "base":
                continue
            with self.subTest(plugin=manifest["id"]):
                self.assertEqual(folder.parent, INFERENCE)
                self.assertTrue((folder / "entry.json").is_file())
                self.assertEqual(json.loads((folder / "entry.json").read_text(encoding="utf-8"))["id"], manifest["id"])
                paths = [source["path"] for source in manifest["sources"].values()]
                self.assertEqual(paths[0], "code", "a technique's first source is its own code")
                for source in manifest["sources"].values():
                    code = Path(source["folder"])
                    self.assertEqual((code.parent.parent, code.name), (INFERENCE, "code"), source["path"])

    def test_every_declared_registry_kernel_is_one_the_plugin_loads_and_none_lives_in_crit(self):
        """plugin.json "kernels" lists what the plugin's code asks crit_plugins.kernels.registry_kernel for, by name
        (crit_plugins.kernels refuses an undeclared one at run time), and no kernel-builder project lives in crit. The
        plugin's modules (its tests aside) must call registry_kernel and hold each declared name as a string of their
        code, a docstring or a comment naming it not counting; which call takes which name is the plugin's tests' to
        show (they run on the registry path and on the fallback)."""
        declared = 0
        for manifest in self.registry:
            folders = [Path(manifest["folder"])] + [Path(s["folder"]) for s in manifest["sources"].values()]
            code = [p for folder in folders for p in folder.rglob("*.py")
                    if Path(manifest["tests"]) not in p.parents]
            calls, strings = kernel_requests(code)
            for kernel in manifest["kernels"]:
                declared += 1
                with self.subTest(plugin=manifest["id"], kernel=kernel["name"]):
                    self.assertTrue(calls, "declares kernels, but the plugin's code never calls registry_kernel")
                    self.assertIn(kernel["name"], strings, "declared, but the plugin's code never asks for it")
                    self.assertTrue(kernel["fallback"].strip())
        self.assertGreater(declared, 0)
        self.assertEqual([p.relative_to(ROOT).as_posix() for p in ROOT.glob("**/build.toml") if ".git" not in p.parts],
                         [], "kernel-builder projects live in crit-kernels")

    def test_a_kernel_name_in_a_docstring_or_comment_is_not_a_request(self):
        with tempfile.TemporaryDirectory() as directory:
            module = Path(directory) / "code.py"
            module.write_text('"""Uses "crit-labs/a" (docs)."""\nfrom crit_plugins.kernels import registry_kernel\n'
                              '# crit-labs/b\nNAME = "crit-labs/c"\n\n\ndef f():\n    """crit-labs/d"""\n'
                              '    return registry_kernel(NAME, folder=None)\n', encoding="utf-8", newline="\n")
            self.assertEqual(kernel_requests([module]), (True, {"crit-labs/c"}))
            module.write_text('NAME = "crit-labs/c"\n', encoding="utf-8", newline="\n")
            self.assertEqual(kernel_requests([module]), (False, {"crit-labs/c"}))

    def test_every_slot_with_cases_has_one_base_plugin_and_it_has_no_entry(self):
        base = [m for m in self.registry if Path(m["folder"]).parent == INFERENCE / "base"]
        self.assertEqual(sorted(m["id"] for m in base), BASE)
        slots = [slot for m in base for slot in m["fills"]]
        self.assertEqual(len(slots), len(set(slots)), "one base plugin per slot")
        self.assertTrue(set(CASES) - {"sampler", "kv_cache"} <= set(slots))
        for manifest in base:
            with self.subTest(plugin=manifest["id"]):
                self.assertIsNone(manifest["reference"])
                self.assertFalse((Path(manifest["folder"]) / "entry.json").exists())
                for source in manifest["sources"].values():
                    self.assertEqual(Path(source["folder"]).parent.parent, INFERENCE, source["path"])

    def test_every_inference_entry_is_a_plugin_or_lends_one_code_or_is_an_engine_service(self):
        techniques = {Path(m["folder"]).name for m in self.registry if Path(m["folder"]).parent == INFERENCE}
        sourced = {Path(s["folder"]).parent.name for m in self.registry for s in m["sources"].values()}
        self.assertEqual(set(ENGINE_SERVICES) & (techniques | sourced), set())
        self.assertEqual(sorted(techniques | sourced | set(ENGINE_SERVICES)), ENTRIES)
        self.assertEqual(sorted(sourced - techniques), ["async-expert-contributions", "expert-streamed-prefill",
                                                        "host-work-mailbox"])
        self.assertEqual((len(techniques), len(ENTRIES)), (21, 31))
        self.assertEqual(list(INFERENCE.glob("*/plugins/*/plugin.json")), [])


class AddTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dest = Path(temp.name) / "engine" / "plugins"

    def test_an_update_copies_the_code_its_plugins_import_again(self):
        add(self.dest, ["heterogeneous-expert-dispatch"])
        stale = self.dest / SOURCES / "host-work-mailbox" / "code"
        files = sorted(p for p in stale.rglob("*.py"))
        self.assertTrue(files)
        files[0].write_text("# a stale copy\n", encoding="utf-8")
        report = add(self.dest, ["heterogeneous-expert-dispatch"], update=True)
        self.assertIn(f"{SOURCES}/host-work-mailbox/code", report["sources"])
        self.assertNotEqual(files[0].read_text(encoding="utf-8"), "# a stale copy\n")

    def test_added_plugins_bring_their_references_tests_and_code_and_pass_there(self):
        report = add(self.dest, ["heterogeneous-expert-dispatch", "host-expert-arithmetic", "softmax-attention"])
        added = dict(report["added"])
        self.assertEqual(added["heterogeneous-expert-dispatch"], "named")
        self.assertEqual(added["experts-reference"], "the reference of heterogeneous-expert-dispatch")
        self.assertEqual(added["device-expert-cache"], "the tests of heterogeneous-expert-dispatch use it")
        self.assertEqual(added["quantized-projections"], "the reference of host-expert-arithmetic")
        self.assertEqual(added["projections-dense"], "the reference of quantized-projections")
        self.assertEqual(added["attention-state"], "the tests of softmax-attention use it")
        self.assertEqual(sorted(report["sources"]), [f"{SOURCES}/async-expert-contributions/code",
                                                     f"{SOURCES}/expert-streamed-prefill/code",
                                                     f"{SOURCES}/host-expert-tier/code",
                                                     f"{SOURCES}/host-work-mailbox/code",
                                                     f"{SOURCES}/lookup-table-rows/code"])   # memory-table-rows' rows
        engine = Registry.discover(self.dest)
        self.assertEqual(set(engine.manifests), set(added))
        self.assertTrue((self.dest / "host-expert-arithmetic" / "README.md").is_file())   # a technique comes whole
        self.assertTrue((self.dest / "base" / "softmax-attention" / "plugin.json").is_file())

        def sources(folder):
            return json.loads((self.dest / folder / "plugin.json").read_text(encoding="utf-8"))["sources"]

        self.assertEqual(sources("heterogeneous-expert-dispatch"), {
            "dispatch": "code", "contributions": f"../{SOURCES}/async-expert-contributions/code",
            "prefill": f"../{SOURCES}/expert-streamed-prefill/code", "tier": f"../{SOURCES}/host-expert-tier/code",
            "arithmetic": "../host-expert-arithmetic/code", "quantized": "../quantized-projections/code",
            "mailbox": f"../{SOURCES}/host-work-mailbox/code", "split": "../device-expert-cache/code"})
        self.assertEqual(sources("host-expert-arithmetic"), {"arithmetic": "code",
                                                             "quantized": "../quantized-projections/code"})
        self.assertEqual(sources("base/softmax-attention"), {"state": "../../attention-state/code"})
        self.assertEqual(sources("base/experts-reference"),
                         {"prefill": f"../../{SOURCES}/expert-streamed-prefill/code"})
        for manifest in engine:
            for source in manifest["sources"].values():
                Path(source["folder"]).relative_to(self.dest.resolve())
        self.assertFalse(list(self.dest.rglob("__pycache__")))
        items = run_tests(engine, log=lambda line: None)
        self.assertEqual({i["name"]: i["detail"] for i in items if i["state"] != "passed"}, {})

    def test_the_engines_own_plugins_and_code_are_kept(self):
        add(self.dest, ["attention-state"])
        (self.dest / "attention-state" / "plugin.py").write_text("# the engine's own\n" + (
            self.dest / "attention-state" / "plugin.py").read_text(encoding="utf-8"), encoding="utf-8")
        report = add(self.dest, ["softmax-attention"])
        self.assertEqual(report["added"], [("softmax-attention", "named")])
        self.assertEqual(report["kept"], [("attention-state", "the tests of softmax-attention use it")])
        self.assertEqual(report["sources"], [])    # it reads the engine's attention-state code
        own = (self.dest / "attention-state" / "plugin.py").read_text(encoding="utf-8")
        self.assertTrue(own.startswith("# the engine"))
        with self.assertRaisesRegex(PluginError, "has attention-state already"):
            add(self.dest, ["attention-state"])
        with self.assertRaisesRegex(PluginError, "no plugin no-such-plugin"):
            add(self.dest, ["no-such-plugin"])

    def test_update_replaces_a_library_copy_an_engine_plugin_reads(self):
        add(self.dest, ["softmax-attention"])
        own = self.dest / "attention-state" / "plugin.py"
        own.write_text("# an old copy\n" + own.read_text(encoding="utf-8"), encoding="utf-8")
        # an engine plugin of its own that reads attention-state's code and names it as its reference
        mine = self.dest / "my-attention"
        mine.mkdir()
        (mine / "plugin.json").write_text(json.dumps({
            "schema": "crit-plugin-v1", "id": "my-attention", "title": "The engine's attention",
            "description": "Reads attention-state's code.", "fills": ["attention"], "needs": [["kv:rows", "kv:pages"]],
            "devices": ["cuda"], "reference": ["softmax-attention"], "sources": {"state": "../attention-state/code"},
            "entry": "plugin.py:create"}), encoding="utf-8")
        (mine / "plugin.py").write_text("def create(context, options, parts):\n    return None\n", encoding="utf-8")
        report = add(self.dest, ["attention-state"], update=True)
        self.assertEqual(report["replaced"], ["attention-state"])
        self.assertFalse(own.read_text(encoding="utf-8").startswith("# an old copy"))
        self.assertIn("my-attention", Registry.discover(self.dest))
        with self.assertRaisesRegex(PluginError, "my-attention is the engine's own plugin"):
            add(self.dest, ["my-attention"], update=True)

    def test_needs_no_plugin_of_the_engine_provides_are_reported(self):
        report = add(self.dest, ["draft-trees", "experts-reference"])
        self.assertEqual(report["unmet"]["draft:head"], [])
        self.assertNotIn("experts:source", report["unmet"])    # its tests bring expert-tensors
        report = add(self.dest, ["heterogeneous-expert-dispatch"])
        self.assertEqual(report["unmet"], {})

    def test_the_library_lists_what_fits_a_checkpoint_and_a_device(self):
        # An NVFP4 checkpoint: dense projections cannot load it, quantized ones only on a GPU with FP4 tensor cores.
        model = Path(self.dest).parent / "nvfp4-model"
        model.mkdir(parents=True)
        (model / "config.json").write_text(json.dumps({"model_type": "llama", "quantization_config": {
            "quant_method": "modelopt", "quant_algo": "NVFP4"}}), encoding="utf-8")
        device = Path(self.dest).parent / "device.json"
        for capability, fits in (("12.0", True), ("8.6", False)):
            device.write_text(json.dumps({"compute_capability": capability}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = main(["list", "--model", str(model), "--device", "cuda", "--device-file", str(device), "--all"])
            text = out.getvalue()
            self.assertEqual(code, 0, text)
            with self.subTest(capability=capability):
                self.assertIn("stored:nvfp4", text.splitlines()[0])
                self.assertIn("not projections-dense: conflicts with stored:nvfp4", text)
                self.assertEqual("projections (required): quantized-projections" in text, fits, text)
                self.assertEqual("not quantized-projections: needs cuda:fp4" in text, not fits, text)
                # No library plugin implements llama yet: the model plugin is the one to write.
                self.assertIn("nothing fits model", text)

    def test_the_cli_lists_the_library_and_adds_from_it(self):
        def run(*args):
            with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
                code = main([str(a) for a in args])
            return code, out.getvalue() + err.getvalue()

        code, out = run("list")
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"recurrent-state-recovery +technique +recurrence ")
        self.assertRegex(out, r"sequential-recurrence +base +recurrence ")
        self.assertIn("(reference sequential-recurrence)", out)
        self.assertRegex(out, r"draft-trees +technique +drafter, verifier .*\(fused; reference greedy-head-drafter, "
                              r"chain-verifier\)")
        code, out = run("add", self.dest, "draft-trees")
        self.assertEqual(code, 0, out)
        self.assertIn("added   chain-verifier", out)
        self.assertIn(f"copied  {SOURCES}/mtp-decode/code", out)
        code, out = run("add", self.dest, "draft-trees")
        self.assertEqual(code, 1)
        self.assertIn("has draft-trees already", out)
        manifest = json.loads((self.dest / "draft-trees" / "plugin.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sources"], {"trees": "code", "mtp": f"../{SOURCES}/mtp-decode/code"})
        manifest = json.loads((self.dest / "base" / "chain-verifier" / "plugin.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sources"], {"mtp": f"../../{SOURCES}/mtp-decode/code"})


if __name__ == "__main__":
    unittest.main()
