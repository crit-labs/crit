"""Random geometry (crit.preflight.geometry): checkpoints of sampled shapes resolve the library-default engine's spec
and score through the models core as their reference does, on the CPU. SEEDS span the head widths (32 to 512, partial
rotary tables over 96- and 320-wide heads), the vocabularies to 262k and the layer kinds; CRIT_GEOMETRY_SEEDS=A:B
draws more (each seed one checkpoint of every family)."""
import os
from pathlib import Path
import tempfile
import unittest

import support.runs  # noqa: F401 - the product folders on sys.path

try:
    import torch
    import transformers
except ImportError:   # pragma: no cover - the toolkit's venv has both
    torch = transformers = None

SEEDS = (1, 3, 5, 11)


def seeds():
    given = os.environ.get("CRIT_GEOMETRY_SEEDS")
    if not given:
        return SEEDS
    start, end = (int(n) for n in given.split(":"))
    return tuple(range(start, end))


@unittest.skipIf(torch is None, "torch and transformers")
class RandomGeometryTests(unittest.TestCase):
    def test_sampled_shapes_cover_the_ranges(self):
        from crit.preflight import geometry
        drawn = [geometry.sample(seed, family) for seed in range(40) for family in geometry.FAMILIES]
        widths = {c["head_dim"] for c in drawn} | {c["global_head_dim"] for c in drawn if "global_head_dim" in c}
        self.assertEqual(widths, set(geometry.HEAD_WIDTHS))
        self.assertEqual({c["vocab_size"] for c in drawn}, set(geometry.VOCABULARIES))
        self.assertTrue(any(set(c.get("layer_types", ())) == {"sliding_attention", "full_attention"} for c in drawn))
        self.assertEqual(geometry.sample(7, "routed"), geometry.sample(7, "routed"))   # one seed, one geometry

    def test_each_geometry_resolves_and_scores_as_its_reference(self):
        if os.environ.get("CRIT_CORE_STEP_LIBRARY") is None and not (Path(__file__).parents[1] / "build").is_dir():
            from crit_plugins.testing import unchecked
            raise unchecked("Core's runtime (crit core build), which the composed engine runs on")
        from crit import models, spec
        from crit.preflight import geometry
        for seed in seeds():
            for family in geometry.FAMILIES:
                config = geometry.sample(seed, family)
                with self.subTest(geometry.geometry(config)), tempfile.TemporaryDirectory() as temp:
                    folder = geometry.write(config, Path(temp) / "model", seed=seed)
                    if folder is None:
                        self.skipTest(f"this transformers lacks {config['model_type']}")
                    self.assertIsNotNone(spec.library_spec(folder))   # the library-default engine resolves
                    compared = models.diff(folder)
                    self.assertIsNone(compared["first_divergent"], compared)


if __name__ == "__main__":
    unittest.main()
