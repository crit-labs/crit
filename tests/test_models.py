"""crit models: the model techniques and finished model plugins it lists, a checkpoint inspected against them, tiny
configs, a model plugin checked, kept in the user's cache and added to another engine by its id; each technique's
own tests; and, with torch and transformers, a tiny random checkpoint whose engine scores as its reference does, with
the finished model plugin and with one being built that brings a technique of its own."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("inference/core/python", "cli", "kernels"):
    sys.path.insert(0, str(ROOT / _folder))

from crit import models  # noqa: E402

try:
    import torch
    import transformers
except ImportError:
    torch = transformers = None


class CatalogTests(unittest.TestCase):
    def test_every_technique_and_model_plugin_lists_and_checks(self):
        found = models.techniques()
        self.assertTrue(found)
        provided = {name for entry in found for names in entry["provides"].values() for name in names} | {"auto"}
        shipped = models.model_plugins([("library", models.CACHE)])
        self.assertTrue(shipped)
        for plugin in shipped:
            with self.subTest(plugin=plugin["id"]):
                layout = plugin["layout"]
                self.assertTrue(plugin["model_types"])
                used = ({spec["block"] for spec in layout["mixers"].values()}
                        | {spec["block"] for spec in layout.get("before", [])}
                        | {layout.get("residual", {}).get("block", "pre-norm"), layout.get("ffn", {}).get("block", "auto")})
                own = {name for piece in plugin["pieces"] for names in piece["provides"].values() for name in names}
                # a shipped plugin composes library techniques and its family's own
                self.assertEqual(used - provided - own, set())
                self.assertFalse(own & provided)
        served = [t for plugin in shipped for t in plugin["model_types"]]
        self.assertEqual(len(served), len(set(served)))   # one finished plugin a model type
        self.assertIn("pre-norm", provided)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(models.main(["list", "--json"]), 0)
        rows = json.loads(out.getvalue())
        self.assertEqual(len(rows["techniques"]), len(found))
        self.assertEqual({row["id"] for row in rows["plugins"] if row["where"] == "library"},
                         {plugin["id"] for plugin in shipped})

    def test_each_technique_s_own_tests_pass(self):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "cli"), str(ROOT / "inference/core/python"),
                                                            os.environ.get("PYTHONPATH", "")])}
        for entry in models.techniques():
            with self.subTest(technique=entry["id"]):
                done = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-t",
                                       "tests"], cwd=entry["folder"], env=env, capture_output=True, text=True,
                                      timeout=120)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_a_technique_holds_no_model_name(self):
        # The model plugins' model_types are the only names: none appears in a technique's files.
        names = {t for plugin in models.model_plugins([("library", models.CACHE)]) for t in plugin["model_types"]}
        for path in models.TECHNIQUES.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".json", ".md"):
                text = path.read_text(encoding="utf-8").lower()
                for name in names:
                    self.assertNotIn(name, text, path)


OWN_MIXER = """from pathlib import Path

from .core.attention.attention import Attention
from .core.template.compose import model_plugin


class MyMixer(Attention):
    pass


resolve, memory, create = model_plugin(Path(__file__).resolve().parent, blocks={"my-mixer": MyMixer})
"""


class ModelPluginTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        patcher = mock.patch.object(models, "user_cache", return_value=self.root / "user")
        patcher.start()
        self.addCleanup(patcher.stop)
        from crit import plugins
        patcher = mock.patch.object(plugins, "user_models", side_effect=lambda: {
            path.parent.name: path.parent for path in sorted((self.root / "user").glob("*/plugin.json"))})
        patcher.start()
        self.addCleanup(patcher.stop)

    def built(self, model_type="tiny_test_decoder", mixer=None):
        """An engine whose agent built the model plugin `mine`: the models core beside it as crit plugins add puts
        it, the plugin's own manifest, layout and code (mixer: a block of its own for the sliding-window layers)."""
        from crit.plugins import add
        plugins = self.root / "engine" / "plugins"
        add(plugins, ["model-template", "softmax-attention"])
        shipped = models.CACHE / "llama-qwen3-model"
        folder = plugins / "mine"
        folder.mkdir()
        manifest = json.loads((shipped / "plugin.json").read_text(encoding="utf-8"))
        manifest.update(id="mine", title="A test's own model plugin", needs=[[f"model_type:{model_type}"]],
                        sources={"core": "../model-template"}, test_with=["softmax-attention"])
        (folder / "plugin.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        layout = json.loads((shipped / "layout.json").read_text(encoding="utf-8"))
        if mixer:
            layout["mixers"]["sliding_attention"]["block"] = "my-mixer"
        (folder / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
        (folder / "plugin.py").write_text(mixer or (shipped / "plugin.py").read_text(encoding="utf-8"),
                                          encoding="utf-8")
        return folder

    def test_a_model_plugin_is_kept_once_and_added_to_another_engine_by_its_id(self):
        folder = self.built()
        self.assertEqual(models.check_plugin(folder), [])
        kept = models.cache_add(folder)
        self.assertEqual(kept, self.root / "user" / "mine")
        with self.assertRaisesRegex(ValueError, "--force"):
            models.cache_add(folder)
        mine = next(plugin for plugin in models.model_plugins() if plugin["id"] == "mine")
        self.assertEqual((mine["where"], mine["model_types"]), ("cache", ["tiny_test_decoder"]))
        self.assertEqual(models.plugin_for(["tiny_test_decoder"])["id"], "mine")
        # Another engine takes it by its id, with the core it extends and the plugin its tests use.
        from crit.plugins import add
        from crit_plugins import Registry
        other = self.root / "other" / "plugins"
        report = add(other, ["mine"])
        added = dict(report["added"])
        self.assertEqual(added["mine"], "named, from your cache of model plugins")
        self.assertEqual((added["model-template"], added["softmax-attention"]), ("mine uses it", "mine uses it"))
        self.assertEqual((other / "mine" / "plugin.json").read_bytes(), (folder / "plugin.json").read_bytes())
        registry = Registry.discover(other)
        self.assertEqual(registry["mine"]["extends"]["id"], "model-template")
        self.assertEqual(registry["mine"]["options"].keys(), registry["model-template"]["options"].keys())
        with self.assertRaisesRegex(ValueError, "the engine has mine already"):
            add(other, ["mine"])

    def test_a_kept_plugin_goes_into_the_library_with_the_library_s_sources(self):
        folder = self.built()
        self.assertEqual(models.library_sources(folder), {"core": "../../core/model-template"})
        shipped = json.loads((models.CACHE / "llama-qwen3-model" / "plugin.json").read_text(encoding="utf-8"))
        self.assertEqual(shipped["sources"], models.library_sources(folder))

    def test_a_folder_that_is_no_model_plugin_is_refused(self):
        folder = self.built()
        plugins = folder.parent
        problems = models.check_plugin(plugins / "attention" / "softmax-attention")
        self.assertTrue(any("a model plugin fills model" in p for p in problems), problems)
        self.assertTrue(any("extends the models core" in p for p in problems), problems)
        self.assertTrue(any("no layout.json" in p for p in problems), problems)
        self.assertTrue(any("sits at ENGINE/plugins/ID" in p for p in problems), problems)
        self.assertTrue(any("no plugin.json" in p for p in models.check_plugin(plugins / "nothing")))
        with self.assertRaisesRegex(ValueError, "no plugin.json"):
            models.cache_add(plugins / "nothing")


class ConfigTests(unittest.TestCase):
    def test_a_tiny_config_keeps_the_structure_and_shrinks_the_sizes(self):
        config = {"model_type": "x", "hidden_size": 4096, "num_attention_heads": 32, "num_hidden_layers": 12,
                  "layer_types": ["full_attention", "linear_attention"] * 6, "num_experts": 128,
                  "num_experts_per_tok": 8, "loop_layer_index_start": 2, "tie_word_embeddings": False,
                  "text_config": {"hidden_size": 2048, "num_hidden_layers": 12}}
        small = models.tiny_config(config)
        self.assertEqual((small["hidden_size"], small["num_attention_heads"], small["num_experts"]), (64, 4, 8))
        self.assertEqual(small["layer_types"], ["full_attention", "linear_attention"] * 2)
        self.assertEqual((small["num_hidden_layers"], small["loop_layer_index_start"]), (4, 2))
        self.assertEqual(small["text_config"]["hidden_size"], 64)
        self.assertIs(small["tie_word_embeddings"], False)

    def test_a_tensor_pattern_numbers_layers_and_experts_alike(self):
        self.assertEqual(models.pattern("model.layers.12.mlp.experts.3.up_proj.weight"),
                         "model.layers.N.mlp.experts.N.up_proj.weight")
        self.assertEqual(models.pattern("model.attn_res.queries.7"), "model.attn_res.queries.N")
        self.assertEqual(models.rename("a.wq.weight", [["\\.wq\\.", ".q_proj."]]), "a.q_proj.weight")


@unittest.skipIf(torch is None, "torch and transformers")
class FamilyReadsTests(unittest.TestCase):
    def test_inspect_counts_a_familys_own_techniques_and_never_a_companion_drafters_reads(self):
        # a finished model plugin's own techniques count as the library's do; a technique whose reads are a companion
        # drafter folder's (reads_of companion: Gemma 4's assistant drafter) is not matched against the target
        import tempfile
        from support.tiny_checkpoints import checkpoint
        with tempfile.TemporaryDirectory() as temp:
            gemma = checkpoint("windows", Path(temp) / "gemma")
            if gemma is None:
                self.skipTest("transformers lacks the Gemma 4 architecture")
            found = models.inspect(gemma)
        families = {entry["id"]: entry for entry in models.family_techniques(ROOT / "models/cache/gemma4-model")}
        self.assertEqual(families["assistant-drafter"]["reads_of"], "companion")
        self.assertNotIn("assistant-drafter", found["techniques"])
        listed = [entry["id"] for entry in models.family_techniques(ROOT / "models/cache/gemma4-model")]
        self.assertIn("shared-kv-attention", listed)


class TinyTests(unittest.TestCase):
    def test_a_tiny_checkpoint_inspects_and_its_composed_engine_scores_as_its_reference(self):
        if os.environ.get("CRIT_CORE_STEP_LIBRARY") is None and not (ROOT / "build").is_dir():
            from crit_plugins.testing import unchecked
            raise unchecked("Core's runtime (crit core build), which the composed engine runs on")
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "config-only"
            transformers.Qwen3Config(vocab_size=2048, hidden_size=256, intermediate_size=512, num_hidden_layers=6,
                                     num_attention_heads=8, num_key_value_heads=4, head_dim=32).save_pretrained(source)
            tiny = models.tiny(source, Path(folder) / "tiny")
            found = models.inspect(tiny)
            self.assertEqual(found["plugins"], ["llama-qwen3-model"])
            self.assertIn("attention", found["techniques"])
            compared = models.diff(tiny)
            self.assertIsNone(compared["first_divergent"], compared)
            # A model plugin being built, with a technique of its own, served from its engine's plugins.
            building = ModelPluginTests("built")
            building.setUp()
            self.addCleanup(building.doCleanups)
            compared = models.diff(tiny, plugin=building.built("qwen3", mixer=OWN_MIXER))
            self.assertIsNone(compared["first_divergent"], compared)


@unittest.skipIf(transformers is None or not hasattr(transformers, "Gemma4AssistantForCausalLM"),
                 "transformers with Gemma4AssistantForCausalLM")
class TinyCompanionTests(unittest.TestCase):
    def test_tiny_shrinks_the_companion_beside_it_and_diff_plans_it_as_the_draft_head(self):
        if os.environ.get("CRIT_CORE_STEP_LIBRARY") is None and not (ROOT / "build").is_dir():
            from crit_plugins.testing import unchecked
            raise unchecked("Core's runtime (crit core build), which the composed engine runs on")
        text = dict(vocab_size=512, vocab_size_per_layer_input=512, hidden_size=256, intermediate_size=512,
                    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                    global_head_dim=64, num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=24,
                    layer_types=["sliding_attention", "full_attention"] * 2, enable_moe_block=False,
                    hidden_size_per_layer_input=0, max_position_embeddings=2048, tie_word_embeddings=True,
                    pad_token_id=0, eos_token_id=1, bos_token_id=2,
                    rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                                        "rope_type": "proportional"},
                                     "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}})
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "downloads" / "model"
            transformers.Gemma4TextConfig(**text).save_pretrained(source)
            drafter = dict(text, hidden_size=128, num_hidden_layers=2, layer_types=["sliding_attention",
                                                                                    "full_attention"],
                           num_kv_shared_layers=2, vocab_size_per_layer_input=0)
            transformers.Gemma4AssistantConfig(text_config=transformers.Gemma4TextConfig(**drafter),
                                               backbone_hidden_size=256).save_pretrained(source.parent / "assistant")
            self.assertEqual(models.companions_near(source), [source.parent / "assistant"])
            tiny = models.tiny(source, Path(folder) / "tiny" / "model")
            shrunk = json.loads((tiny.parent / "model-companion" / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(shrunk["backbone_hidden_size"], models.TINY["hidden_size"])
            self.assertEqual(shrunk["model_type"], "gemma4_assistant")
            compared = models.diff(tiny)
            self.assertIsNone(compared["first_divergent"], compared)
            self.assertEqual(Path(compared["companion"]["folder"]).resolve(), (tiny.parent / "model-companion")
                             .resolve())
            self.assertGreater(compared["companion"]["bytes_per_step"], 0)
            # --no-companion: none written
            plain = models.tiny(source, Path(folder) / "plain" / "model", companion=False)
            self.assertFalse((plain.parent / "model-companion").exists())


if __name__ == "__main__":
    unittest.main()
