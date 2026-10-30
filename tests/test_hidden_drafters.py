"""Hidden-state drafters: Core's reader of the published drafter formats (speculators EAGLE-3/DFlash/DSpark, native
EAGLE-3, native DFlash and DSpark, and crit's own crit models drafter heads) with one tap convention, and the drafting
reference's catch-up, block, mapping and truncation rules."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core" / "python"))

from crit_core.model.config import ConfigError  # noqa: E402
from crit_core.model.drafters import BLOCK, EAGLE3, read_drafter, read_drafter_file  # noqa: E402


def drafting():
    name = "crit_test_hidden_drafters_drafting"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "inference" / "hidden-state-drafters" / "code" / "drafting.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DrafterFormatTests(unittest.TestCase):
    def test_speculators_eagle3(self):
        d = read_drafter({
            "speculators_model_type": "eagle3", "draft_vocab_size": 32000, "target_hidden_size": 4096,
            "eagle_aux_hidden_state_layer_ids": [2, 18, 33],
            "transformer_layer_config": {"hidden_size": 4096, "num_hidden_layers": 1},
            "speculators_config": {"proposal_methods": [{"speculative_tokens": 3}], "verifier": {"name_or_path": "org/target"}},
        }, target_layers=36)
        self.assertEqual((d.kind, d.taps, d.draft_vocab_size, d.layers, d.speculative_tokens, d.verifier),
                         (EAGLE3, (2, 18, 33), 32000, 1, 3, "org/target"))
        self.assertEqual(d.drafts_per_pass, 1)

    def test_tap_conventions_agree_across_block_formats(self):
        # speculators spells taps; native DFlash and DSpark name the layer whose output is read (tap = id + 1)
        spec = read_drafter({"speculators_model_type": "dflash", "aux_hidden_state_layer_ids": [2, 18, 33],
                             "mask_token_id": 151669, "block_size": 16, "transformer_layer_config": {"num_hidden_layers": 5}})
        native = read_drafter({"architectures": ["DFlashDraftModel"], "block_size": 16, "num_hidden_layers": 5,
                               "dflash_config": {"target_layer_ids": [1, 17, 32], "mask_token_id": 151669}})
        dspark = read_drafter({"architectures": ["Qwen3DSparkModel"], "block_size": 7, "mask_token_id": 5,
                               "target_layer_ids": [1, 17, 32], "enable_confidence_head": True})
        self.assertEqual(spec.taps, (2, 18, 33))
        self.assertEqual(native.taps, spec.taps)
        self.assertEqual(dspark.taps, spec.taps)
        self.assertEqual((native.kind, native.block_size, native.mask_token_id, native.causal), (BLOCK, 16, 151669, False))
        self.assertEqual(native.drafts_per_pass, 15)
        # DFlash2 keeps block_size inside dflash_config, next to its taps and mask
        dflash2 = read_drafter({"architectures": ["DFlash2DraftModel"], "num_hidden_layers": 3,
                                "dflash_config": {"block_size": 8, "mask_token_id": 0, "target_layer_ids": [1, 25, 49]}},
                               target_layers=52)
        self.assertEqual((dflash2.block_size, dflash2.mask_token_id, dflash2.taps), (8, 0, (2, 26, 50)))
        self.assertTrue(dspark.confidence_head)
        anchor = read_drafter({"speculators_model_type": "dspark", "aux_hidden_state_layer_ids": [3], "mask_token_id": 1,
                               "block_size": 7, "sample_from_anchor": True})
        self.assertEqual(anchor.drafts_per_pass, 7)

    def test_native_eagle3_defaults_and_rejections(self):
        d = read_drafter({"architectures": ["LlamaForCausalLMEagle3"], "draft_vocab_size": 32000}, target_layers=32)
        self.assertEqual(d.taps, (2, 16, 29))
        with self.assertRaises(ConfigError):
            read_drafter({"architectures": ["Eagle3LlamaForCausalLM"]})   # default taps need the target's depth
        with self.assertRaises(ConfigError):
            read_drafter({"speculators_model_type": "eagle3", "eagle_aux_hidden_state_layer_ids": [2, 40]}, target_layers=36)
        with self.assertRaises(ConfigError):
            read_drafter({"architectures": ["DFlashDraftModel"], "dflash_config": {"target_layer_ids": [1]}})  # no block
        self.assertIsNone(read_drafter({"architectures": ["LlamaForCausalLM"]}))
        self.assertIsNone(read_drafter({"speculators_model_type": "medusa"}))

    def test_crits_own_drafters(self):
        d = read_drafter({"num_hidden_layers": 1, "hidden_size": 64, "crit_drafter": {
            "format": "crit-drafter-v1", "kind": "mtp-head", "target_layers": 28, "target": {"path": "/models/target"}}},
            target_layers=28)
        self.assertEqual((d.kind, d.taps, d.drafts_per_pass, d.verifier, d.source), ("mtp-head", (28,), 1, "/models/target",
                                                                                     "crit:crit-drafter-v1"))
        with self.assertRaises(ConfigError):
            read_drafter({"crit_drafter": {"kind": "mtp-head", "target_layers": 28}}, target_layers=32)
        with self.assertRaises(ConfigError):
            read_drafter({"crit_drafter": {"kind": "unknown", "target_layers": 28}})

    def test_reads_a_drafter_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "config.json").write_text(json.dumps({"architectures": ["Eagle3Qwen3ForCausalLM"],
                                                               "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]}}))
            self.assertEqual(read_drafter_file(folder).taps, (1, 2, 3))


class DraftingRuleTests(unittest.TestCase):
    def test_rules(self):
        m = drafting()
        self.assertEqual(m.eagle_inputs([11, 12], 4), [(0, 11), (1, 12), (2, m.PAD), (3, m.PAD)])
        self.assertEqual(m.block_input(9, 99, 4), [9, 99, 99, 99])
        self.assertEqual(m.block_drafts([5, 6, 7, 8]), [6, 7, 8])
        self.assertEqual(m.block_drafts([5, 6, 7, 8], sample_from_anchor=True), [5, 6, 7, 8])
        self.assertEqual(m.map_tokens([0, 1, 2, -1], [0, 4, 7]), [0, 5, 9, -1])
        self.assertEqual(m.scatter_law([0.5, 0.25, 0.25], [0, 4, 7], 10), [0.5, 0, 0, 0, 0, 0.25, 0, 0, 0, 0.25])
        self.assertEqual(m.truncate([1, 2, 3, 4], [0.9, 0.6, 0.4, 0.8], 0.5), [1, 2, m.PAD, m.PAD])
        with self.assertRaises(ValueError):
            m.block_input(1, 2, 1)
        with self.assertRaises(ValueError):
            m.eagle_inputs([1, 2, 3], 2)


if __name__ == "__main__":
    unittest.main()
