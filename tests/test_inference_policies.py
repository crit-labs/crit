"""Exact languages, economical restore decisions and bounded loop observations."""
import math
import random
from pathlib import Path
import sys
import unittest
from test_techniques import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/python"))


class ChoiceTests(unittest.TestCase):
    def setUp(self):
        from crit_core.constraints import choices
        self.m = choices

    def test_segmentation_prefix_choices_utf8_and_private_rollback(self):
        vocab = (b"a", b"b", b"ab", b"", b"x", b"\xc3", b"\xa9")
        grammar = self.m.ChoiceGrammar((b"a", b"ab", "é".encode()), vocab, 3)
        state = self.m.ChoiceState(grammar)
        self.assertEqual(state.allowed(), (0, 2, 5))
        branch = state.fork()
        branch.append(0)
        self.assertEqual(branch.allowed(), (1, 3))
        branch.append(3)
        self.assertTrue(branch.ended)
        with self.assertRaises(ValueError):
            branch.append(3)
        branch.rewind(1)
        branch.append(1)
        self.assertEqual(branch.allowed(), (3,))
        self.assertEqual(state.allowed(), (0, 2, 5))
        state.append(5)
        self.assertEqual(state.allowed(), (6,))
        state.append(6)
        state.append(3)
        self.assertTrue(state.ended)

    def test_invalid_dead_end_and_empty_choice(self):
        state = self.m.ChoiceState(self.m.ChoiceGrammar((b"", b"a"), (b"", b"a", b""), 0))
        self.assertEqual(state.allowed(), (0, 1))
        with self.assertRaises(ValueError):
            state.append(2)
        with self.assertRaises(ValueError):
            state.append(-1)
        with self.assertRaises(ValueError):
            state.rewind(1)
        state = self.m.ChoiceState(self.m.ChoiceGrammar((b"a",), (b"x", b""), 1))
        with self.assertRaises(ValueError):
            state.allowed()
        for args in (((), (b"",), 0), ((b"a",), (b"eos",), 0), (("a",), (b"",), 0)):
            with self.assertRaises(ValueError):
                self.m.ChoiceGrammar(*args)


class CheckpointPolicyTests(unittest.TestCase):
    def setUp(self):
        self.m = load("checkpoint-storage", "planning.py")
        self.query = tuple(range(100))
        self.cost = self.m.PrefillCost(16, .001, .0001, .0000001)

    def source(self, name, n, **kwargs):
        return self.m.PrefixSource(name, self.query[:n], "model+conditioning", "codec", {"kv", "recurrent"}, **kwargs)

    def choose(self, sources, **kwargs):
        return self.m.choose_prefix(self.query, sources, self.cost,
                                    fingerprint="model+conditioning", codec="codec",
                                    components={"kv", "recurrent"}, **kwargs)

    def test_short_resident_prefix_beats_long_host_and_scratch_beats_expensive(self):
        resident = self.source("device", 60)
        host = self.source("host", 90, transfers=(self.m.TransferCost(1000000, 1000000),))
        plan = self.choose((host, resident))
        self.assertIs(plan.source, resident)
        self.assertGreater(plan.estimates[0].total_seconds, plan.scratch_seconds)
        plan = self.choose((host,))
        self.assertIsNone(plan.source)
        self.assertEqual(plan.predicted_seconds, plan.scratch_seconds)

    def test_identity_sibling_pending_provisional_and_feasibility_are_hard_gates(self):
        from dataclasses import replace
        valid = self.source("device", 90)
        changes = ({"fingerprint": "different image"}, {"codec": "different"},
                   {"components": frozenset({"kv"})}, {"ready": False}, {"committed": False},
                   {"feasible": False}, {"prefix": (999,)}, {"prefix": self.query + (100,)})
        for change in changes:
            with self.subTest(change=change):
                plan = self.choose((replace(valid, **change),))
                self.assertIsNone(plan.source)
                self.assertIsNotNone(plan.estimates[0].rejection)

    def test_total_transfer_and_attention_math_ties_and_numeric_limits(self):
        self.assertAlmostEqual(self.cost.seconds(10, 17), .002 + .0017 + 323 * .0000001)
        source = self.source("disk", 90, transfers=(self.m.TransferCost(100, 10000, 2, .001),
                                                     self.m.TransferCost(100, 100000, 1, .001)))
        self.assertAlmostEqual(self.choose((source,)).estimates[0].restore_seconds, .014)
        zero = self.m.PrefillCost(16, 0, 0)
        self.assertIsNone(self.m.choose_prefix(self.query, (self.source("device", 90),), zero,
                                              fingerprint="model+conditioning", codec="codec", components={"kv", "recurrent"}).source)
        for value in (math.nan, math.inf, -1, True):
            with self.assertRaises(ValueError):
                self.m.PrefillCost(16, value, .001)
        with self.assertRaises(ValueError):
            self.m.TransferCost(100, 0)
        with self.assertRaises(ValueError):
            self.choose((source, source))

    def test_random_decisions_equal_independent_minimum(self):
        rng = random.Random(19)
        for _ in range(200):
            sources = tuple(self.source(str(i), rng.randrange(1, 101),
                                       transfers=(self.m.TransferCost(rng.randrange(100), 10000),))
                            for i in range(8))
            plan = self.choose(sources)
            expected = min([plan.scratch_seconds] + [e.total_seconds for e in plan.estimates])
            self.assertEqual(plan.predicted_seconds, expected)

    def test_capture_semantics_exclusions_budget_and_chunk_grid(self):
        m = self.m
        hints = (m.CaptureHint(40, 0), m.CaptureHint(45, 0), m.CaptureHint(80, 0), m.CaptureHint(200, 0))
        plan = m.plan_captures(300, 0, hints, chunk_tokens=64, min_gap=16,
                               max_captures=3, exclusions=((70, 90),), ladder_tokens=100)
        self.assertEqual(tuple(h.position for h in plan), (40, 200, 256))
        self.assertTrue(plan[0].exact)
        self.assertFalse(plan[-1].exact)
        self.assertEqual(m.plan_captures(300, 0, hints, max_captures=0), ())
        self.assertEqual(m.plan_captures(1, 0), ())
        for kwargs in ({"base": 301}, {"exclusions": ((20, 10),)}, {"chunk_tokens": 0}):
            args = dict(base=0)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                m.plan_captures(300, **args)


class ReasoningGuardTests(unittest.TestCase):
    def guard(self, **kwargs):
        m = load("session-scheduling", "reasoning_guard.py")
        config = dict(passage_words=3, repeats=3, window_words=12, history_words=30, threshold=.5)
        config.update(kwargs)
        return m.ReasoningLoopGuard(**config)

    def test_split_unicode_and_whole_text_have_identical_coverage(self):
        text = "Évaluer ces trois cas. évaluer ces trois cas. ÉVALUER ces trois cas. "
        whole, streamed = self.guard(), self.guard()
        whole.feed(text, final=True)
        for char in text:
            streamed.feed(char)
        streamed.feed("", final=True)
        self.assertEqual(whole.observe(), streamed.observe())
        self.assertTrue(whole.observe().triggered)

    def test_unique_text_warmup_aging_and_transactional_overflow(self):
        guard = self.guard(max_word_chars=5)
        guard.feed("a b c a b c ")
        self.assertFalse(guard.observe().triggered)
        guard.feed("a b c a b c ")
        self.assertTrue(guard.observe().triggered)
        before = guard.observe()
        with self.assertRaises(ValueError):
            guard.feed("x y toolong")
        self.assertEqual(before, guard.observe())
        guard.feed(" ".join(str(i) for i in range(40)), final=True)
        self.assertEqual(guard.observe().history_words, 30)
        self.assertEqual(guard.observe().coverage, 0)

    def test_overlap_counts_union_not_sum_and_bounds(self):
        guard = self.guard()
        guard.feed("a " * 12)
        self.assertEqual(guard.observe().coverage, 1)
        for kwargs in ({"repeats": 1}, {"threshold": math.nan}, {"window_words": 31}, {"passage_words": 0}):
            with self.assertRaises(ValueError):
                self.guard(**kwargs)


if __name__ == "__main__":
    unittest.main()
