"""crit diff on a tiny random checkpoint: an engine of the library's plugins stays within the 16-bit baseline at every
layer, and an engine whose checkpoint has one layer's FFN weight altered is named at that layer, its blocks and the
plugins they call."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "cli"), str(ROOT / "inference/core/python"), str(ROOT / "models/core/python")]

try:
    import torch
    import transformers
    from safetensors.torch import load_file, save_file
except ImportError:   # the reference and the engine need both
    torch = transformers = None


@unittest.skipIf(torch is None, "torch and transformers")
class LayerDiffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from crit import diff, models
        from crit.plugins import add
        from crit_plugins.testing import core_runtime
        core_runtime()
        cls.diff, cls.temp = diff, tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        config = transformers.LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                                          rms_norm_eps=1e-6, tie_word_embeddings=False)
        torch.manual_seed(0)
        reference = transformers.LlamaForCausalLM(config).to(torch.bfloat16)
        cls.model = root / "model"
        reference.save_pretrained(cls.model, safe_serialization=True)
        # The same checkpoint with layer 2's down projection tripled: the engine reads it, the reference the original
        cls.faulty = root / "faulty"
        shutil.copytree(cls.model, cls.faulty)
        (shard,) = sorted(cls.faulty.glob("*.safetensors"))
        tensors = load_file(str(shard))
        tensors["model.layers.2.mlp.down_proj.weight"] = tensors["model.layers.2.mlp.down_proj.weight"] * 3
        save_file(tensors, str(shard), metadata={"format": "pt"})
        use = models.selection_for(cls.model)
        cls.package = root / "package"
        add(cls.package / "plugins", sorted(set(use.values())))
        (cls.package / "configuration.json").write_text(json.dumps(
            {"plugins": {"use": use, "options": {use["model"]: dict(models.SMALL)}}}), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_a_correct_engine_stays_within_the_16_bit_baseline(self):
        found = self.diff.layer_diff(self.package, self.model, tokens=24, device="cpu")
        self.assertEqual([p["name"] for p in found["places"]], ["layer 0", "layer 1", "layer 2", "layer 3", "final"])
        self.assertEqual(found["verdict"], "format-sized", self.diff.render(found))
        self.assertIsNone(found["first_divergent"])

    def test_a_text_longer_than_a_prompt_chunk_is_tapped_chunk_after_chunk(self):
        # layer_rows runs a prompt's chunks one after another (each reading the earlier chunks' KV), so a text past
        # one chunk is diffed whole and stays within the baseline
        from crit import models
        chunked = Path(self.temp.name) / "chunked"
        if not chunked.exists():
            shutil.copytree(self.package, chunked)
            configuration = json.loads((chunked / "configuration.json").read_text(encoding="utf-8"))
            configuration["plugins"]["options"][configuration["plugins"]["use"]["model"]] = {
                **models.SMALL, "prefill_chunk": 16}
            (chunked / "configuration.json").write_text(json.dumps(configuration), encoding="utf-8")
        found = self.diff.layer_diff(chunked, self.model, tokens=40, device="cpu")
        self.assertEqual(found["tokens"], 40)
        self.assertEqual(found["verdict"], "format-sized", self.diff.render(found))

    def test_a_fault_in_one_layer_is_named_at_that_layer(self):
        found = self.diff.layer_diff(self.package, self.faulty, tokens=24, device="cpu", reference=self.model)
        self.assertEqual(found["verdict"], "bug-sized", self.diff.render(found))
        first = found["first_divergent"]
        self.assertEqual((first["name"], first["made_by"], first["layer"]), ("layer 3", "layer 2", 2))
        self.assertTrue(first["mixer"] and first["ffn"])   # the blocks layer 2 runs
        self.assertIn("projections-dense", first["plugins"])
        # the places before it are rounding only
        self.assertEqual({p["sized"] for p in found["places"][:3]}, {"format-sized"})
        self.assertIn("first bug-sized gap: on entry to layer 3, made by layer 2", self.diff.render(found))


    def test_a_declared_activation_format_widens_the_bound_by_its_own_gap(self):
        # A checkpoint that declares its matrices' input activations (here every one, FP8 per token, by patching
        # crit-eval's reading of the checkpoint) gets the declared-format allowance beside the 16-bit baseline: the
        # FP32 reference with those inputs quantized, FACTOR times the larger of the two the bound
        from unittest import mock
        with mock.patch("crit_eval.limits.schemes.activations", lambda checkpoint, base: ("fp8_token", None)):
            found = self.diff.layer_diff(self.package, self.model, tokens=24, device="cpu")
        places = found["places"]
        self.assertTrue(all(p["declared"] is not None for p in places))
        self.assertGreater(places[-1]["declared"], places[-1]["baseline"])   # FP8 inputs move more than BF16
        for p in places:
            self.assertEqual(p["bound"], max(self.diff.FACTOR * max(p["baseline"], p["declared"]), self.diff.FLOOR))
        self.assertIn("declared formats", self.diff.render(found))
        # without declared formats there is none, and the bound is the baseline's
        plain = self.diff.layer_diff(self.package, self.model, tokens=24, device="cpu")
        self.assertTrue(all(p["declared"] is None for p in plain["places"]))


    def test_a_model_plugin_that_offers_no_tap_is_told_the_contract(self):
        from crit_core.native import StepEngine
        engine = StepEngine.__new__(StepEngine)
        engine.model = object()
        with self.assertRaisesRegex(ValueError, "offers no layer rows: a model plugin gives crit diff its tap with "
                                                "layer_rows"):
            engine.layer_rows([1, 2])
        self.assertIsNone(engine.layer_parts())


if __name__ == "__main__":
    unittest.main()
