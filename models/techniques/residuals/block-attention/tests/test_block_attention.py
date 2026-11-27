"""block-attention: its code mounts beside model-template's and registers what entry.json says it provides."""
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]


class TechniqueTests(unittest.TestCase):
    def test_it_registers_what_it_provides(self):
        from crit.models import mounted   # crit's CLI: the technique as model-template mounts it
        entry = json.loads((HERE / "entry.json").read_text(encoding="utf-8"))
        module = mounted(entry["source"])
        for kind, names in entry["provides"].items():
            self.assertEqual(sorted(getattr(module, kind.upper())), sorted(names))


if __name__ == "__main__":
    unittest.main()


class PhaseTests(unittest.TestCase):
    """A looped layer's gates take the fused kernel for every decode row, a step's beside prompt rows too (they took
    the GEMMs whenever the step held a prompt row, so a decode row's bits followed what shared its step)."""

    def test_the_fused_gates_take_every_decode_row_before_the_first_prompt_chunk(self):
        from types import SimpleNamespace
        from crit.models import mounted
        module = mounted("block_attention")
        residual = module.BlockAttention.__new__(module.BlockAttention)
        residual.ops = object()
        messages, gates = SimpleNamespace(shape=(20,)), {"down": None}

        def ctx(groups):
            return SimpleNamespace(prompt=any(p for _, _, p in groups), groups=groups)
        self.assertEqual(residual._fused_rows(messages, gates, ctx([(0, 8, False), (8, 20, False)])), 20)
        self.assertEqual(residual._fused_rows(messages, gates, ctx([(0, 3, False), (3, 4, True), (4, 20, True)])), 3)
        self.assertEqual(residual._fused_rows(messages, gates, ctx([(0, 20, True)])), 0)
        self.assertEqual(residual._fused_rows(None, gates, ctx([(0, 20, False)])), 0)
        residual.ops = None
        self.assertEqual(residual._fused_rows(messages, gates, ctx([(0, 20, False)])), 0)
