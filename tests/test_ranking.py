"""crit_plugins.ranking.top_ranked: the first k entries of a stable descending sort, bit for bit, at a top-k's cost."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference/core/python"))

try:
    import torch
except ImportError:   # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class TopRanked(unittest.TestCase):
    def sorted_first(self, values, k):
        ranked, order = values.sort(dim=-1, descending=True, stable=True)
        return ranked[..., :k], order[..., :k]

    def same(self, values, k):
        from crit_plugins.ranking import top_ranked
        got, want = top_ranked(values, k), self.sorted_first(values, k)
        self.assertTrue(torch.equal(got[1], want[1]), (k, got[1], want[1]))
        self.assertTrue(torch.equal(got[0].view(torch.int32), want[0].view(torch.int32)), k)

    def test_it_is_the_stable_sorts_first_k_with_ties_at_and_above_the_kth_value(self):
        g = torch.Generator().manual_seed(5)
        values = torch.randn(64, 997, generator=g)
        values[3, [10, 400, 900]] = 9.0                    # ties above the k-th value
        values[7, :] = values[7, :].round(decimals=0)      # ties everywhere, at the k-th value too
        values[9, [1, 2, 3, 4, 5]] = values[9].max()       # more ties at the top than k
        values[11, :] = 0.0                                # one value: the lowest indices
        values[12, 500:] = float("-inf")                   # a masked half
        values[13, :] = float("-inf")
        values[14, 0] = -0.0
        values[15, 6] = float("nan")                       # the sort puts NaN first
        for k in (1, 5, 20, 996, 997):
            self.same(values, k)
        self.same(values[7], 20)                           # one row

    def test_half_precision_and_a_large_vocabulary(self):
        g = torch.Generator().manual_seed(6)
        values = torch.randn(4, 248_320, generator=g).to(torch.bfloat16)   # bf16: many exact ties
        from crit_plugins.ranking import top_ranked
        got, want = top_ranked(values, 20), self.sorted_first(values, 20)
        self.assertTrue(torch.equal(got[1], want[1]))
        self.assertTrue(torch.equal(got[0], want[0]))

    def test_no_temporary_is_wider_than_the_rows_masks(self):
        # A long score ranks 128 logit rows of a large vocabulary at once: a temporary of int64 a value (a count of
        # the masks, 256 MiB at 262,144 tokens) was the allocation that failed on the device, past its headroom.
        from torch.utils._python_dispatch import TorchDispatchMode
        from crit_plugins.ranking import top_ranked

        class Widest(TorchDispatchMode):
            widest = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                for t in out if isinstance(out, (tuple, list)) else (out,):
                    if isinstance(t, torch.Tensor):
                        Widest.widest = max(Widest.widest, t.numel() * t.element_size())
                return out
        values = torch.randn(32, 50_000, generator=torch.Generator().manual_seed(7))
        values[5, :] = values[5].round(decimals=0)          # a row with ties at the k-th value
        with Widest():
            top_ranked(values, 20)
        self.assertLessEqual(Widest.widest, values.numel())  # a bool mask: a byte a value
        self.same(values, 20)


if __name__ == "__main__":
    unittest.main()
