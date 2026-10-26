"""Host references for device speculative rounds; no device execution."""
import random
import unittest

from test_techniques import load


class RoundBudgetTests(unittest.TestCase):
    def setUp(self):
        self.m = load("device-speculative-round", "round_budget.py")

    def test_ratio_tokens_speedup_and_default_threshold(self):
        budget = self.m.RoundBudget()
        self.assertEqual(budget.threshold, 1.3)
        budget.observe_step([self.m.DeviceTiming("forward", 0, 1000)])
        budget.observe_step([500, 500, 1000])  # mean ordinary step: 1500 ns
        budget.observe_round([1500, 300], emitted=4)
        budget.observe_round([1200, 0], emitted=2)
        report = budget.report()
        self.assertAlmostEqual(report.step_ns, 1500)
        self.assertAlmostEqual(report.round_ns, 1500)
        self.assertEqual(report.tokens_per_round, 3)
        self.assertAlmostEqual(report.ratio, 1.0)
        self.assertAlmostEqual(report.speedup, 3.0)
        self.assertTrue(report.passed)

    def test_threshold_boundary_and_failure(self):
        at = self.m.RoundBudget(threshold=1.5)
        at.observe_step([1000])
        at.observe_round([1500], emitted=1)
        self.assertTrue(at.report().passed)
        over = self.m.RoundBudget()
        over.observe_step([1000])
        over.observe_round([1301], emitted=5)
        report = over.report()
        self.assertFalse(report.passed)
        self.assertGreater(report.speedup, 1)  # many tokens do not excuse the ratio

    def test_invalid_observations_are_rejected(self):
        budget = self.m.RoundBudget()
        for bad in (0, float("inf"), -1):
            with self.subTest(threshold=bad), self.assertRaises(ValueError):
                self.m.RoundBudget(threshold=bad)
        with self.assertRaises(ValueError):
            budget.report()
        with self.assertRaises(ValueError):
            budget.observe_step([0])
        with self.assertRaises(ValueError):
            budget.observe_step([-5])
        with self.assertRaises(ValueError):
            budget.observe_round([100], emitted=-1)
        with self.assertRaises(ValueError):
            budget.observe_round([1.5], emitted=1)


class DrafterCatchUpTests(unittest.TestCase):
    def setUp(self):
        self.m = load("device-speculative-round", "catch_up.py")

    def test_plan_positions_and_validation(self):
        plan = self.m.catch_up_plan(frontier=10, emitted=2, width=4, draft_tokens=3)
        self.assertEqual(plan.positions, (8, 9, 10, 11))
        self.assertEqual(plan.valid, (True, True, False, False))
        self.assertEqual((plan.source, plan.chain_positions), (1, (10, 11)))
        for bad in ((10, 5, 4, 3), (1, 2, 4, 3), (10, 1, 0, 3), (10, 1, 4, 0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.m.catch_up_plan(*bad)

    def test_fixed_width_catch_up_equals_exact_ingestion(self):
        import random
        rng = random.Random(5)
        for draft_tokens in (1, 3, 5):
            width = draft_tokens + 1
            fixed, exact = self.m.PositionalDrafter(97), self.m.PositionalDrafter(97)
            for p in range(6):  # shared prefill history
                f, t = rng.randrange(1000), rng.randrange(97)
                fixed.step(p, f, t); exact.step(p, f, t)
            frontier = 6  # drafter memory covers positions < frontier before each round
            for _ in range(40):
                emitted = rng.randint(1, width)
                frontier += emitted
                features = [rng.randrange(1000) for _ in range(width)]
                tokens = [rng.randrange(97) for _ in range(width)]
                plan = self.m.catch_up_plan(frontier, emitted, width, draft_tokens)
                got = fixed.propose_fixed(plan, features, tokens, draft_tokens)
                want = exact.propose_exact(frontier, emitted, features, tokens, draft_tokens)
                self.assertEqual(got, want)

    def test_widths_chosen_per_round_catch_up_the_widest_width(self):
        # The previous round emitted up to its own K + 1 tokens, so with K chosen per round the catch-up keeps the
        # widest K + 1 columns; a round narrower than the one before cannot hold that count in its own K + 1.
        rng = random.Random(8)
        widest = 6
        width = widest + 1
        fixed, exact = self.m.PositionalDrafter(97), self.m.PositionalDrafter(97)
        for p in range(6):
            f, t = rng.randrange(1000), rng.randrange(97)
            fixed.step(p, f, t); exact.step(p, f, t)
        frontier, previous, narrower = 6, widest, 0
        for _ in range(80):
            emitted = rng.randint(1, previous + 1)
            frontier += emitted
            draft_tokens = rng.randint(1, widest)
            features = [rng.randrange(1000) for _ in range(width)]
            tokens = [rng.randrange(97) for _ in range(width)]
            plan = self.m.catch_up_plan(frontier, emitted, width, draft_tokens)
            self.assertEqual(fixed.propose_fixed(plan, features, tokens, draft_tokens),
                             exact.propose_exact(frontier, emitted, features, tokens, draft_tokens))
            if emitted > draft_tokens + 1:
                narrower += 1
                with self.assertRaises(ValueError):
                    self.m.catch_up_plan(frontier, emitted, draft_tokens + 1, draft_tokens)
            previous = draft_tokens
        self.assertGreater(narrower, 10)


class ColumnInvarianceTests(unittest.TestCase):
    def setUp(self):
        self.m = load("device-speculative-round", "column_invariance.py")

    def test_fixed_order_reductions_are_column_invariant_and_width_splits_are_not(self):
        rng = random.Random(4)
        weights = [[rng.uniform(-1, 1) * 10 ** rng.randint(-4, 4) for _ in range(32)] for _ in range(3)]
        columns = [[rng.uniform(-1, 1) * 10 ** rng.randint(-4, 4) for _ in range(32)] for _ in range(8)]
        filler = [[rng.uniform(-1, 1) for _ in range(32)] for _ in range(3)]
        self.assertEqual(self.m.check(self.m.fixed_order_dot(weights), columns, (1, 2, 5, 8), filler), [])
        differing = self.m.check(self.m.width_split_dot(weights), columns, (1, 2, 5, 8), filler)
        self.assertTrue(differing)
        self.assertNotIn(1, {width for _, width in differing})  # width 1 is the one-column result itself
        with self.assertRaises(ValueError):
            self.m.check(self.m.fixed_order_dot(weights), columns, (0,), filler)


class DraftSubsetTests(unittest.TestCase):
    def setUp(self):
        self.m = load("draft-vocabulary-subset", "subset.py")

    def test_frequency_choice_required_tokens_and_ties(self):
        counts = {0: 5, 1: 9, 2: 9, 3: 1, 4: 7}
        self.assertEqual(self.m.choose_subset(counts, 3), (1, 2, 4))
        self.assertEqual(self.m.choose_subset(counts, 3, required=(3,)), (1, 2, 3))
        self.assertEqual(self.m.choose_subset([4, 4, 4, 4], 2), (0, 1))  # ties prefer lower IDs
        for bad in ((counts, 0, ()), (counts, 1, (0, 3)), (counts, 9, ()), ({-1: 2}, 1, ())):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.m.choose_subset(*bad)

    def test_prior_subset_and_contiguous_ranges(self):
        # Without statistics: the lowest IDs plus required control tokens at the end.
        rows = self.m.prior_subset(10, 5, required=(9, 8))
        self.assertEqual(rows, (0, 1, 2, 8, 9))
        self.assertEqual(self.m.SubsetHead(rows, 10).ranges(), ((0, 3), (8, 10)))
        self.assertEqual(self.m.SubsetHead((4,), 10).ranges(), ((4, 5),))
        self.assertEqual(self.m.prior_subset(4, 4), (0, 1, 2, 3))
        for bad in ((10, 1, (8, 9)), (10, 0, ()), (10, 11, ()), (10, 2, (10,)), (0, 1, ())):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.m.prior_subset(*bad)

    def test_index_mapping_and_validation(self):
        head = self.m.SubsetHead((2, 5, 9), vocabulary=10)
        self.assertEqual([head.token(i) for i in range(len(head))], [2, 5, 9])
        self.assertEqual((head.index(5), head.index(4)), (1, None))
        for rows in ((), (3, 3), (5, 2), (0, 10)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.m.SubsetHead(rows, vocabulary=10)

    def test_gathered_argmax_equals_target_whenever_target_is_inside(self):
        rng = random.Random(4)
        vocabulary, hidden = 60, 8
        head = [[rng.gauss(0, 1) for _ in range(hidden)] for _ in range(vocabulary)]
        head[41] = list(head[7])  # exact tie: the lower ID wins in both heads
        subset = self.m.SubsetHead(self.m.choose_subset(
            {t: rng.randrange(100) for t in range(vocabulary)}, 15, required=(7, 41)), vocabulary)
        inside = 0
        for trial in range(400):
            state = list(head[7]) if trial == 0 else [rng.gauss(0, 1) for _ in range(hidden)]
            target = self.m.target_argmax(head, state)
            draft = self.m.draft_argmax(head, state, subset)
            self.assertIsNotNone(subset.index(draft))
            if subset.index(target) is not None:
                self.assertEqual(draft, target)
                inside += 1
        self.assertEqual(self.m.target_argmax(head, head[7]), 7)
        self.assertGreater(inside, 50)

    def test_speculative_output_with_subset_drafts_equals_ordinary_greedy(self):
        rng = random.Random(11)
        vocabulary, hidden = 40, 6
        head = [[rng.gauss(0, 1) for _ in range(hidden)] for _ in range(vocabulary)]
        embed = [[rng.gauss(0, 1) for _ in range(hidden)] for _ in range(vocabulary)]

        def state(history):  # synthetic context: decayed sum of embeddings
            out = [0.0] * hidden
            for token in history:
                out = [0.6 * a + b for a, b in zip(out, embed[token])]
            return out

        ordinary = [3]
        for _ in range(60):
            ordinary.append(self.m.target_argmax(head, state(ordinary)))
        counts = {}
        for token in ordinary:
            counts[token] = counts.get(token, 0) + 1
        subset = self.m.SubsetHead(self.m.choose_subset(counts, min(6, len(counts))), vocabulary)
        history, rounds = [3], 0
        while len(history) < len(ordinary):
            drafts, scratch = [], list(history)
            for _ in range(3):
                drafts.append(self.m.draft_argmax(head, state(scratch), subset))
                scratch.append(drafts[-1])
            accepted = []
            for draft in drafts:  # verification always uses the full head
                target = self.m.target_argmax(head, state(history + accepted))
                if draft != target:
                    accepted.append(target)
                    break
                accepted.append(draft)
            else:
                accepted.append(self.m.target_argmax(head, state(history + accepted)))
            history += accepted[: len(ordinary) - len(history)]
            rounds += 1
        self.assertEqual(history, ordinary)
        self.assertLess(rounds, len(ordinary) - 1)

    def test_cost_coverage_and_expected_tokens(self):
        cost = self.m.draft_head_cost(vocabulary=1000, hidden=64, size=100, draft_tokens=3, bytes_per_weight=0.5)
        self.assertEqual(cost.full_bytes, 3 * 1000 * 64 // 2)
        self.assertEqual(cost.subset_bytes, 3 * 100 * (32 + 4))
        self.assertEqual(cost.saved_bytes, cost.full_bytes - cost.subset_bytes)
        self.assertAlmostEqual(cost.ratio, 10800 / 96000)
        subset = self.m.SubsetHead((1, 2), vocabulary=4)
        self.assertAlmostEqual(self.m.coverage({0: 1, 1: 2, 2: 1}, subset), 0.75)
        self.assertEqual(self.m.expected_tokens(1.0, 3), 4.0)
        self.assertAlmostEqual(self.m.expected_tokens(0.5, 3), 1.875)
        with self.assertRaises(ValueError):
            self.m.draft_head_cost(10, 4, 11, 1, 1)

    def test_size_is_chosen_from_measured_rounds_not_coverage(self):
        best, rates = self.m.choose_size({100: (2.0, 10.0), 200: (2.6, 12.0), 400: (2.7, 13.5)})
        self.assertEqual(best, 200)
        self.assertAlmostEqual(rates[400], 2.7 / 13.5)
        self.assertEqual(self.m.choose_size({100: (2.0, 10.0), 200: (4.0, 20.0)})[0], 100)  # tie: smaller size
        for bad in ({}, {0: (2.0, 1.0)}, {10: (0.5, 1.0)}, {10: (2.0, 0.0)}):
            with self.assertRaises(ValueError):
                self.m.choose_size(bad)


if __name__ == "__main__":
    unittest.main()
