"""crit_core.model.rope_table against Hugging Face's ROPE_INIT_FUNCTIONS (when transformers is installed) and against
fixed values, for default, linear, dynamic NTK, YaRN (with mscale and attention_factor), Llama 3 and LongRoPE."""
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core" / "python"))

from crit_core.model import ConfigError, parse_model_config, rope_table  # noqa: E402

BASE = {"model_type": "llama", "hidden_size": 512, "num_attention_heads": 4, "num_hidden_layers": 2,
        "max_position_embeddings": 32768}

# (name, config keys over BASE, sequence lengths to check)
CASES = [
    ("default", {"rope_theta": 1e6}, (None,)),
    ("partial", {"rope_theta": 1e4, "partial_rotary_factor": 0.5}, (None,)),
    ("linear", {"rope_theta": 1e4, "rope_scaling": {"rope_type": "linear", "factor": 4.0}}, (None,)),
    ("dynamic", {"rope_theta": 1e4, "rope_scaling": {"rope_type": "dynamic", "factor": 2.0}}, (None, 65536)),
    ("yarn", {"rope_theta": 1e6, "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                                                  "original_max_position_embeddings": 8192}}, (None,)),
    ("yarn-mscale", {"rope_theta": 1e4, "rope_scaling": {"rope_type": "yarn", "factor": 40.0, "mscale": 1.0,
                                                         "mscale_all_dim": 0.707, "beta_fast": 32, "beta_slow": 1,
                                                         "original_max_position_embeddings": 4096}}, (None,)),
    ("yarn-attention", {"rope_theta": 1e6, "rope_scaling": {"rope_type": "yarn", "factor": 128.0,
                                                            "attention_factor": 1.4852,
                                                            "original_max_position_embeddings": 4096}}, (None,)),
    ("yarn-untruncated", {"rope_theta": 1e6, "rope_scaling": {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096,
                                                        "truncate": False}}, (None,)),
    ("llama3", {"rope_theta": 5e5, "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                                    "high_freq_factor": 4.0,
                                                    "original_max_position_embeddings": 8192}}, (None,)),
    ("longrope", {"rope_theta": 1e4, "original_max_position_embeddings": 4096,
                  "rope_scaling": {"rope_type": "longrope", "short_factor": [1.0 + 0.01 * i for i in range(64)],
                                   "long_factor": [1.0 + 0.5 * i for i in range(64)]}}, (None, 2048, 8192)),
]


def config(keys):
    return parse_model_config(dict(BASE, **keys))


class RopeTableTests(unittest.TestCase):
    def test_tables_match_hugging_face(self):
        try:
            import torch  # noqa: F401
            from transformers import LlamaConfig
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        except ImportError:
            self.skipTest("transformers and torch are not installed")
        for name, keys, lengths in CASES:
            parsed = config(keys)
            hf = LlamaConfig(**dict(BASE, **keys))
            kind = parsed.rope.type or "default"
            for length in lengths:
                with self.subTest(name=name, length=length):
                    table = rope_table(parsed.rope, parsed.effective_head_dim(), parsed.max_positions, length)
                    if kind == "default":
                        dim = int(parsed.effective_head_dim() * (parsed.rope.partial_rotary_factor or 1.0))
                        theta = parsed.rope.theta
                        want = [1.0 / theta ** (i / dim) for i in range(0, dim, 2)], 1.0
                    else:
                        inv, attention = ROPE_INIT_FUNCTIONS[kind](hf, "cpu", length)
                        want = inv.double().tolist(), float(attention)
                    self.assertEqual(len(table.inv_freq), len(want[0]))
                    for got, expected in zip(table.inv_freq, want[0]):
                        self.assertTrue(math.isclose(got, expected, rel_tol=2e-6), (got, expected))
                    self.assertTrue(math.isclose(table.attention_factor, want[1], rel_tol=1e-9))

    def test_rules_without_transformers(self):
        plain = rope_table(config({"rope_theta": 1e4}).rope, 4)
        self.assertEqual(plain, ((1.0, 0.01), 1.0))
        # A declared attention_factor is the final scale: never multiplied by YaRN's own mscale (SGLang #33558).
        declared = config(CASES[6][1])
        self.assertEqual(rope_table(declared.rope, 128, declared.max_positions).attention_factor, 1.4852)
        # YaRN without factor uses max_position_embeddings / original_max_position_embeddings.
        ratio = config({"rope_theta": 1e6, "rope_scaling": {"rope_type": "yarn", "original_max_position_embeddings": 4096}})
        explicit = config({"rope_theta": 1e6, "rope_scaling": {"rope_type": "yarn", "factor": 8.0,
                                                               "original_max_position_embeddings": 4096}})
        self.assertEqual(rope_table(ratio.rope, 128, 32768), rope_table(explicit.rope, 128, 32768))
        # LongRoPE switches to its long factors only past the original length.
        long = config(CASES[-1][1])
        short, at, past = (rope_table(long.rope, 128, long.max_positions, n) for n in (None, 4096, 4097))
        self.assertEqual(short, at)
        self.assertNotEqual(short.inv_freq, past.inv_freq)
        self.assertAlmostEqual(short.attention_factor, math.sqrt(1 + math.log(8) / math.log(4096)))

    def test_unsupported_types_and_missing_parameters_are_refused(self):
        refused = [
            ({"rope_scaling": {"rope_type": "su", "factor": 2.0}}, 128, "has no table"),
            ({"rope_scaling": {"rope_type": "linear"}}, 128, "needs factor"),
            ({"rope_scaling": {"rope_type": "llama3", "factor": 8.0, "original_max_position_embeddings": 8192}}, 128,
             "needs low_freq_factor"),
            ({"rope_scaling": {"rope_type": "longrope", "short_factor": [1.0], "long_factor": [1.0],
                               "original_max_position_embeddings": 4096}}, 128, "short_factor has 1 entries"),
            ({"partial_rotary_factor": 0.3}, 10, "not a positive even number"),
            ({}, 0, "not a positive integer"),
        ]
        for keys, head_dim, message in refused:
            with self.subTest(keys=keys), self.assertRaisesRegex(ConfigError, message):
                rope_table(config(keys).rope, head_dim, 32768)
