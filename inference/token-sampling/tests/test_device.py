"""token-sampling's device form draws what its host form draws, row for row, at every rule."""
import random

from crit_plugins.testing import PluginTestCase

RULES = [{"temperature": 0.0}, {"temperature": 0.8, "top_p": 0.9, "seed": 3}, {"temperature": 1.3, "top_k": 4, "seed": 5},
         {"temperature": 0.6, "min_p": 0.2, "seed": 9, "stream": 1}, {"temperature": 1.0, "top_p": 0.5, "top_k": 6, "seed": 11},
         {"temperature": 1.0, "seed": 2, "allowed": [1, 3, 4, 8]}]


class DeviceSamplerTests(PluginTestCase):
    def test_every_row_draws_what_the_host_form_draws(self):
        torch = __import__("torch") if self.device() else None
        sampler = self.assemble()["sampler"]
        rng = random.Random(7)
        rows = [[round(rng.uniform(-3, 3), 3) for _ in range(23)] for _ in range(8)]
        rows[2][4] = rows[2][9] = max(rows[2]) + 1   # a tie: the lower id ranks first
        logits = torch.tensor(rows, dtype=torch.float32, device=self.device())
        for rule in RULES:
            for position in range(12):
                params = [dict(rule, position=position + r) for r in range(len(rows))]
                with self.subTest(rule=rule, position=position):
                    device = sampler.sample(logits, params).tolist()
                    host = sampler.select([[float(v) for v in row] for row in logits.cpu().tolist()], params)
                    self.assertEqual(device, host)

    def test_a_constrained_row_draws_only_what_its_mask_allows(self):
        from crit_plugins import Allowed
        torch = __import__("torch") if self.device() else None
        sampler = self.assemble()["sampler"]
        rng = random.Random(3)
        rows = [[round(rng.uniform(-3, 3), 3) for _ in range(70)] for _ in range(4)]
        logits = torch.tensor(rows, dtype=torch.float32, device=self.device())
        allowed = [Allowed.of([2, 33, 34, 65], 70), Allowed.of([69], 70), None, Allowed.of(range(0, 70, 3), 70)]
        for rule in RULES[:5]:
            params = [dict(rule, position=r, **({} if a is None else {"allowed": a})) for r, a in enumerate(allowed)]
            with self.subTest(rule=rule):
                device = sampler.sample(logits, params).tolist()
                self.assertEqual(device, sampler.select(rows, params))
                self.assertTrue(all(a is None or t in a for t, a in zip(device, allowed)))

    def test_one_rule_for_every_row_and_a_row_without_a_choice(self):
        torch = __import__("torch") if self.device() else None
        sampler = self.assemble()["sampler"]
        logits = torch.tensor([[0.0, 2.0, 1.0], [5.0, 5.0, -1.0]], device=self.device())
        self.assertEqual(sampler.sample(logits, {"temperature": 0.0}).tolist(), [1, 0])
        with self.assertRaisesRegex(ValueError, "no selectable token"):
            sampler.sample(torch.full((1, 3), float("-inf"), device=self.device()), {"temperature": 1.0})
