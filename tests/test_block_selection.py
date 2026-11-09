"""crit_plugins.device.topk_blocks: block-sparse attention's canonical selection (ties to the lower block id, ids in
ascending order, -1 past the readable blocks), the same on the CPU and CUDA, in a batch and alone, in a CUDA graph."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "python"))

from crit_plugins.device import topk_blocks  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def reference(row, k):
    finite = [(-v, i) for i, v in enumerate(row) if not (math.isnan(v) or v == -math.inf)]
    chosen = sorted(i for _, i in sorted(finite)[:k])
    return chosen + [-1] * (k - len(chosen))


@unittest.skipIf(torch is None, "torch is not installed")
class TopkBlocksTests(unittest.TestCase):
    def devices(self):
        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def scores(self, tokens, blocks, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.randint(-3, 4, (tokens, blocks), generator=g).float()      # many ties
        x[torch.rand(tokens, blocks, generator=g) < 0.1] = float("-inf")
        x[torch.rand(tokens, blocks, generator=g) < 0.05] = float("nan")
        x[0, :3] = -0.0                                                     # -0.0 ties with +0.0
        x[0, 3:6] = 0.0
        x[1] = float("-inf")                                                # a token that may read nothing
        x[2, blocks - 1] = float("inf")
        return x

    def test_it_picks_the_highest_scores_with_ties_to_the_lower_id_in_id_order(self):
        for device in self.devices():
            for blocks, k in ((50, 8), (9, 9), (5, 12), (300, 64)):
                x = self.scores(6, blocks, blocks + k)
                found = topk_blocks(x.to(device), k).cpu().tolist()
                self.assertEqual(found, [reference(row, k) for row in x.tolist()], (device, blocks, k))

    def test_fp64_scores_rank_as_they_are_and_16_bit_ones_too(self):
        for device in self.devices():
            # 1 + 1e-12 rounds to 1.0 in FP32: compared as FP64 it ranks above 1.0, no new tie
            x = torch.tensor([[1.0, 1.0 + 1e-12, 0.5, float("nan"), float("-inf")]], dtype=torch.float64)
            self.assertEqual(topk_blocks(x.to(device), 1).tolist(), [[1]], device)
            self.assertEqual(topk_blocks(x.to(device), 4).tolist(), [[0, 1, 2, -1]], device)
            for dtype in (torch.float64, torch.bfloat16, torch.float16):
                y = self.scores(6, 50, 4).to(dtype)
                want = [reference(row, 8) for row in y.double().tolist()]
                self.assertEqual(topk_blocks(y.to(device), 8).cpu().tolist(), want, (device, dtype))

    def test_rows_select_alone_as_in_a_batch_and_alike_on_every_device(self):
        x = self.scores(12, 1000, 3)
        together = topk_blocks(x, 37)
        for t in range(12):
            self.assertTrue(torch.equal(topk_blocks(x[t:t + 1].clone(), 37)[0], together[t]))
        for device in self.devices():
            self.assertTrue(torch.equal(topk_blocks(x.to(device), 37).cpu(), together))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA graphs need CUDA")
    def test_it_selects_inside_a_cuda_graph(self):
        x = self.scores(4, 512, 5).cuda()
        topk_blocks(x, 16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = topk_blocks(x, 16)
        y = self.scores(4, 512, 6).cuda()
        x.copy_(y)
        graph.replay()
        self.assertEqual(out.cpu().tolist(), [reference(row, 16) for row in y.cpu().tolist()])

    def test_with_positions_the_complete_blocks_compete_and_the_tail_is_always_read(self):
        bs, k = 4, 3
        g = torch.Generator().manual_seed(8)
        scores = torch.randn(6, 40, generator=g)
        positions = torch.tensor([2, 3, 11, 13, 30, 159])
        for device in self.devices():
            found = topk_blocks(scores.to(device), k, positions.to(device), bs).cpu().tolist()
            for row, p, got in zip(scores.tolist(), positions.tolist(), found):
                complete = (p + 1) // bs
                chosen = reference([v if b < complete else float("-inf") for b, v in enumerate(row)], k)
                tail = [p // bs] if (p + 1) % bs else []
                want = sorted([b for b in chosen if b >= 0] + tail)
                self.assertEqual(got, want + [-1] * (k + 1 - len(want)), (device, p))
        # Below the budget every complete block is read, and the tail: plain causal attention.
        for position in (9, 10, 11):     # blocks 0 and 1 complete with tail 2, or (11) blocks 0..2 complete
            self.assertEqual(topk_blocks(scores[:1], k, torch.tensor([position]), bs)[0].tolist(), [0, 1, 2, -1])


if __name__ == "__main__":
    unittest.main()
