"""token-sampling's staged device form draws what sample() draws, row for row, for every rule mixed in one step, and
replays from a CUDA graph with new rules set between replays."""
import random

from crit_plugins.testing import PluginTestCase

RULES = [{"temperature": 0.0}, {"temperature": 0.8, "top_p": 0.9, "seed": 3}, {"temperature": 1.3, "top_k": 4, "seed": 5},
         {"temperature": 0.6, "min_p": 0.2, "seed": 9, "stream": 1}, {"temperature": 1.0, "top_p": 0.5, "top_k": 6, "seed": 11},
         {"temperature": 1.0, "seed": (1 << 63) + 12345}]


class StagedSamplerTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("the staged form runs on CUDA")
        self.torch = __import__("torch")
        self.sampler = self.assemble()["sampler"]
        rng = random.Random(7)
        rows = [[round(rng.uniform(-3, 3), 3) for _ in range(300)] for _ in range(len(RULES))]
        rows[0][4] = rows[0][9] = max(rows[0]) + 1   # a tie: the lower id ranks first
        self.logits = self.torch.tensor(rows, dtype=self.torch.float32, device="cuda")

    def test_a_staged_step_draws_what_sample_draws(self):
        staged = self.sampler.staged(8, "cuda")
        for position in range(40):
            params = [dict(rule, position=position + 1000 * r) for r, rule in enumerate(RULES)]
            staged.set(params)
            self.assertEqual(staged.sample(self.logits).tolist(), self.sampler.sample(self.logits, params).tolist(),
                             position)

    def test_a_captured_step_replays_with_the_rules_set_before_it(self):
        torch = self.torch
        staged = self.sampler.staged(len(RULES), "cuda")
        staged.set([dict(rule, position=0) for rule in RULES])
        staged.sample(self.logits)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tokens = staged.sample(self.logits)
        for position in (5, 17, 99):
            params = [dict(rule, position=position) for rule in RULES]
            staged.set(params)
            graph.replay()
            self.assertEqual(tokens.tolist(), self.sampler.sample(self.logits, params).tolist(), position)
