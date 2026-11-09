"""Draft widths chosen round by round (crit_plugins.rounds: FixedWidth, StepWidth, RateWidth): each round's width is
picked among the driver's captured widths from the rounds before it, on the device inside the round, and the choice
never changes what a row emits: greedy rounds emit plain greedy decoding at any policy, sampled rounds are the host
forms' rounds at the widths they ran (so the law is the rejection rule's at every width) and keep the target's law
over many seeds, eagerly and replayed from CUDA graphs."""
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "core" / "python"))
sys.path.insert(0, str(ROOT))
from test_device_drafting import DEVICE, GRAPHS, HEAD, Head, Target, host_decode, plugins, torch  # noqa: E402

WIDTHS = (1, 2, 3, 4, 6)


def driver(rows, policy, widths=WIDTHS, widest=None):
    from crit_plugins.rounds import SpeculativeRounds
    rounds = SpeculativeRounds(None, rows, widest or max(widths), DEVICE, widths=widths, width_policy=policy)
    rounds.start["allowance"][:rows] = 100
    return rounds


def judge(rounds, width, matched, hold=(), allowance=None):
    """One round's acceptance as the policy reads it inside a round: rows of n_matched, held rows, start allowance."""
    rows = len(matched)
    view = rounds.view(rows, width, "greedy")
    rounds.n_matched[:rows] = torch.tensor(matched)
    rounds.hold[:rows] = torch.tensor([1 if r in hold else 0 for r in range(rows)])
    if allowance is not None:
        rounds.start["allowance"][:rows] = torch.tensor(allowance)
    rounds._choose(view)
    rounds.hold.zero_()
    return rounds.width_table.tolist()


@unittest.skipIf(torch is None, "torch is not installed")
class PolicyTests(unittest.TestCase):
    def test_the_step_policy_steps_one_width_up_after_a_round_that_kept_every_draft_and_down_after_fewer(self):
        from crit_plugins.rounds import StepWidth
        rounds = driver(2, StepWidth(start=3))
        self.assertEqual(rounds.width(1), 3)
        self.assertEqual(judge(rounds, 3, [3, 3]), [4, 4])
        self.assertEqual(judge(rounds, 4, [4, 2]), [3, 3], "one row kept fewer: down")
        self.assertEqual(judge(rounds, 3, [3, 0], hold=(1,)), [4, 4], "a held row is not judged")
        self.assertEqual(judge(rounds, 4, [4, 0], allowance=[100, 3]), [6, 6], "nor a row whose allowance ran out")
        self.assertEqual(judge(rounds, 6, [6, 6]), [6, 6], "the widest stays")
        self.assertEqual(judge(rounds, 1, [0, 0]), [1, 1], "the narrowest stays")
        self.assertEqual(judge(rounds, 2, [0, 0], hold=(0, 1)), [1, 1], "a round of held rows changes nothing")
        self.assertEqual(judge(rounds, 5, [5, 5]), [1, 1], "a width outside the policy's changes nothing")
        rounds.width_table[1] = -1   # a row count native rounds leave to step()
        self.assertEqual(judge(rounds, 1, [1, 1]), [2, -1])

    def test_a_band_between_down_and_up_keeps_the_width(self):
        from crit_plugins.rounds import StepWidth
        rounds = driver(4, StepWidth(start=3, up=0.95, down=0.5))
        self.assertEqual(judge(rounds, 3, [3, 3, 3, 2]), [3] * 4, "11 of 12 kept: in the band")
        self.assertEqual(judge(rounds, 3, [3, 3, 3, 3]), [4] * 4)
        self.assertEqual(judge(rounds, 4, [1, 1, 2, 3]), [3] * 4, "7 of 16 kept: down")

    def test_the_rate_policy_widens_with_acceptance_narrows_without_and_weighs_each_row_counts_cost(self):
        from crit_plugins.rounds import RateWidth
        policy = RateWidth(decay=0.9, column_cost=0.02)
        rounds = driver(2, policy)
        self.assertEqual(rounds.width(1), 1)
        for _ in range(30):
            table = judge(rounds, 6, [6, 6])
        self.assertEqual(table, [6, 6], "every draft kept: the widest pays")
        self.assertGreater(float(policy.rates()[-1]), 0.9)
        for _ in range(60):
            table = judge(rounds, 6, [0, 0])
        self.assertEqual(table, [1, 1], "drafts rejected at once: the narrowest")
        for _ in range(60):
            table = judge(rounds, 6, [4, 4])
        self.assertIn(table[0], (3, 4, 6))
        # Rounds of two rows past two drafts that cost ten times as much: a narrower width for two rows than for one.
        policy.set_costs([[1.0 + 0.02 * (w + 1) for w in WIDTHS], [1.0, 1.0, 10.0, 10.0, 10.0]])
        table = judge(rounds, 6, [4, 4])
        self.assertLess(table[1], table[0])
        with self.assertRaises(ValueError):
            policy.set_costs([[1.0] * len(WIDTHS)])

    def test_policies_take_only_the_drivers_widths_and_name_themselves(self):
        from crit_plugins.rounds import FixedWidth, StepWidth, width_policy
        self.assertEqual(driver(1, None).width(1), 6, "fixed at the widest by default")
        self.assertEqual(driver(1, FixedWidth(2)).width(1), 2)
        for policy in (FixedWidth(5), StepWidth(start=5)):
            with self.assertRaises(ValueError):
                driver(1, policy)
        with self.assertRaises(ValueError):
            driver(1, None, widths=(2, 9), widest=8)
        self.assertIsInstance(width_policy("step", up=1.0), StepWidth)
        with self.assertRaises(ValueError):
            width_policy("random")
        rounds = driver(1, StepWidth())
        self.assertEqual(rounds.width_facts(), {"draft_widths": list(WIDTHS), "width_policy": "step"})


def policies():
    from crit_plugins.rounds import FixedWidth, RateWidth, StepWidth
    return {"fixed": FixedWidth(3), "step": StepWidth(start=2), "rate": RateWidth(column_cost=0.02)}


@unittest.skipIf(torch is None, "torch is not installed")
class DecodeTests(unittest.TestCase):
    """Rounds through Core's round driver at the widths a policy chose, against the host forms at those widths."""

    def decode(self, graphs, mode, rules, policy, prompts, counts, stops=(5,), vocab=40, head_weight=1.2):
        from crit_plugins.rounds import Rounds
        widest = max(WIDTHS)
        assembly = plugins({"drafter": "greedy-head-drafter", "verifier": "chain-verifier", "sampler": "token-sampling"},
                           facts=HEAD)
        target = Target(vocab, 21)
        head = Head(vocab, widest, 22).like(target, head_weight)
        rows = len(prompts)
        rounds = Rounds(target, assembly["sampler"], rows=rows, widest=widest, device=DEVICE,
                        drafter=assembly["drafter"], head=head, verifier=assembly["verifier"], stops=2,
                        vocabulary=vocab, sampled=mode == "sampled", widths=WIDTHS, width_policy=policy)
        if graphs:
            for width in WIDTHS:
                rounds.capture(rows, width, mode)
        sampler = assembly["sampler"]
        firsts = [sampler.select([target.host(p[-1], len(p) - 1)], dict(rules[r], position=0))[0]
                  for r, p in enumerate(prompts)]
        for slot, (prompt, rule) in enumerate(zip(prompts, rules)):
            rounds.bind(slot, ("sequence", slot), len(prompt), firsts[slot], counts[slot] - 1,
                        len(prompt) + counts[slot] + widest + 1, stops=stops, params=rule,
                        tokens=list(prompt) + [firsts[slot]])
        outputs = [[f] for f in firsts]
        done = [len(o) >= c or o[-1] in stops for o, c in zip(outputs, counts)]
        ran = [[] for _ in prompts]
        for _ in range(400):
            if all(done):
                break
            held = [r for r in range(rows) if done[r]]
            width = rounds.width(rows)
            rounds.launch(rows, width, mode, hold=held)
            for r, result in enumerate(rounds.read()):
                if r in held:
                    continue
                ran[r].append(width)
                outputs[r] += result["tokens"]
                done[r] = outputs[r][-1] in stops or len(outputs[r]) >= counts[r]
        return outputs, (assembly, target, head, prompts, firsts, counts, rules, stops), ran, rounds

    PROMPTS = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

    def test_greedy_rounds_emit_plain_greedy_decoding_at_every_policy(self):
        # Three rows, and one (its rounds narrower than the widest write a strided slice of the predictions).
        for prompts, counts in ((self.PROMPTS, [60, 41, 52]), (self.PROMPTS[:1], [60])):
            rules = [{"temperature": 0.0}] * len(prompts)
            for (name, policy), graphs in ((p, g) for p in policies().items() for g in GRAPHS):
                outputs, host, ran, rounds = self.decode(graphs, "greedy", rules, policy, prompts, counts)
                plain = host_decode(*host[:6], [[0]] * len(prompts), host[6], host[7], {}, {})
                with self.subTest(policy=name, graphs=graphs, rows=len(prompts)):
                    self.assertEqual(outputs, plain)
                    by_width = rounds.telemetry()["drafts"]["rounds_by_width"]
                    self.assertEqual(sum(by_width.values()), max(len(r) for r in ran))
                    if name == "fixed":
                        self.assertEqual(set(by_width), {"3"})
                    else:
                        self.assertGreater(len(by_width), 1, ("the policy chose more than one width", by_width))

    def test_sampled_rounds_are_the_host_forms_rounds_at_the_widths_they_ran(self):
        rules = [{"temperature": 0.8, "top_k": 12, "seed": 3}, {"temperature": 1.0, "seed": 9},
                 {"temperature": 1.1, "top_p": 0.9, "seed": 17, "stream": 1}]
        counts = [40, 33, 37]
        for name, policy in policies().items():
            for graphs in GRAPHS:
                outputs, host, ran, rounds = self.decode(graphs, "sampled", rules, policy, self.PROMPTS, counts)
                with self.subTest(policy=name, graphs=graphs):
                    self.assertEqual(outputs, host_decode(*host[:6], ran, host[6], host[7], {}, {}))
                    if name != "fixed":
                        self.assertGreater(len({w for r in ran for w in r}), 1)

    def test_sampled_rounds_keep_the_targets_law_whatever_widths_the_batch_chose(self):
        """64 rows of different seeds a decode, 3 tokens each after a fixed first one, widths chosen by the step
        policy from the whole batch's acceptance: the 3-token sequences' frequencies against the target's law
        (chi-square, cells of expected count 5 or more merged)."""
        from crit_plugins.rounds import StepWidth
        vocab, rows, runs, prompt, first = 5, 64, 24, [1, 2], 3
        target = Target(vocab, 31)
        seen = {}
        for run in range(runs):
            rules = [{"temperature": 1.0, "seed": 1000 * run + r} for r in range(rows)]
            outputs = self.law_decode(target, rows, rules, prompt, first, StepWidth(start=1))
            for o in outputs:
                seen[tuple(o)] = seen.get(tuple(o), 0) + 1
        law = self.target_law(target, prompt, first, 3)
        n = rows * runs
        cells = sorted(law, key=law.get, reverse=True)
        chi, pooled_p, pooled_seen, dof = 0.0, 0.0, 0, -1
        for cell in cells:
            expected = n * law[cell]
            if expected >= 5:
                chi += (seen.get(cell, 0) - expected) ** 2 / expected
                dof += 1
            else:
                pooled_p, pooled_seen = pooled_p + law[cell], pooled_seen + seen.get(cell, 0)
        if n * pooled_p >= 5:
            chi += (pooled_seen - n * pooled_p) ** 2 / (n * pooled_p)
            dof += 1
        # Wilson-Hilferty: the chi-square quantile at p = 0.001 (z = 3.09).
        bound = dof * (1 - 2 / (9 * dof) + 3.09 * math.sqrt(2 / (9 * dof))) ** 3
        self.assertLess(chi, bound, (chi, dof))

    def law_decode(self, target, rows, rules, prompt, first, policy):
        from crit_plugins.rounds import Rounds
        widths = (1, 2, 3)
        assembly = plugins({"drafter": "greedy-head-drafter", "verifier": "chain-verifier",
                            "sampler": "token-sampling"}, facts=HEAD)
        head = Head(target.table.shape[0], 3, 7).like(target, 1.5)
        rounds = Rounds(target, assembly["sampler"], rows=rows, widest=3, device=DEVICE, drafter=assembly["drafter"],
                        head=head, verifier=assembly["verifier"], stops=1, vocabulary=target.table.shape[0],
                        sampled=True, widths=widths, width_policy=policy)
        for slot in range(rows):
            rounds.bind(slot, slot, len(prompt), first, 3, len(prompt) + 8, params=rules[slot],
                        tokens=list(prompt) + [first])
        outputs = [[] for _ in range(rows)]
        for _ in range(8):
            if all(len(o) >= 3 for o in outputs):
                break
            rounds.launch(rows, rounds.width(rows), "sampled", hold=[r for r in range(rows) if len(outputs[r]) >= 3])
            for r, result in enumerate(rounds.read()):
                outputs[r] += result["tokens"]
        return outputs

    @staticmethod
    def target_law(target, prompt, first, count):
        """The target's law of the `count` tokens after `first` (at position len(prompt)), temperature 1."""
        law = {(): 1.0}
        for step in range(count):
            grown = {}
            for path, p in law.items():
                token = path[-1] if path else first
                row = torch.tensor(target.host(token, len(prompt) + step), dtype=torch.float64)
                probabilities = torch.softmax(row, 0).tolist()
                for t, q in enumerate(probabilities):
                    grown[path + (t,)] = p * q
            law = grown
        return law


if __name__ == "__main__":
    unittest.main()
