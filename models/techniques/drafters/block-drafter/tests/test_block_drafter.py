"""block-drafter: its code mounts beside model-template's and registers what entry.json says it provides (its span
attention on a GPU is crit's kernel row-ops', tested there against code/reference.py)."""
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


class RingTests(unittest.TestCase):
    """A local layer keeps a doubled ring of R = window + block columns: positions written in chunks, catch-up
    columns and block rows, any window of up to R positions before the frontier reads as one span equal to the full
    cache's positions."""

    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch")
        from crit.models import mounted
        self.torch, self.d = torch, mounted("block_drafter")

    def test_rings_only_for_local_layers_smaller_than_the_context(self):
        cfg = {"num_hidden_layers": 3, "local_layers": [0, 2], "sliding_window": 1024, "block_size": 8,
               "num_key_value_heads": 8, "head_dim": 128}
        self.assertEqual(self.d.rings(cfg, 200_000), [1032, 0, 1032])
        self.assertEqual(self.d.rings(cfg, 2000), [0, 0, 0])
        self.assertEqual(self.d.rings({**cfg, "sliding_window": 0}, 200_000), [0, 0, 0])
        full = 3 * 2 * 2 * 200_000 * 8 * 128 * 2
        self.assertEqual(self.d.cache_bytes(cfg, 2, 200_000), full // 3 + 2 * 2 * 2 * 2065 * 8 * 128 * 2)

    def test_every_window_reads_back(self):
        torch, d = self.torch, self.d
        r, tokens, total = 13, 400, 300
        full = torch.full((tokens,), -1.0)
        ring = torch.full((2 * r + 1,), -1.0)

        def put(positions, values):
            full[positions] = values
            for at in d.spots(torch, r, tokens, positions):
                ring[at] = values

        g = torch.Generator().manual_seed(5)
        at = 0
        while at < total:   # chunks, a round's block rows then its catch-up (some columns unused)
            n = int(torch.randint(1, 9, (1,), generator=g))
            block = torch.arange(at, at + 8)
            put(block, torch.rand(8, generator=g))
            unused = torch.full((8 - n,), tokens - 1)
            put(torch.cat([torch.arange(at, at + n), unused]), torch.rand(8, generator=g))
            at += n
            last = at + 7
            put(torch.arange(at, at + 8), torch.rand(8, generator=g))   # the next block pass's rows
            first = max(0, last - r + 1)
            a, b = d.span(r, torch.tensor(first), torch.tensor(last))
            self.assertTrue(torch.equal(ring[int(a):int(b) + 1], full[first:last + 1]), at)


if __name__ == "__main__":
    unittest.main()
