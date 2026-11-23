"""Core's prompt-step budget (crit_plugins.steps): a prompt chunk's rows by the position it starts at, within the
run's bound, learned from the steps the engine runs (the load's calibration only the first guess)."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference" / "core" / "python"))

from crit_plugins.steps import GRACE, QUANTUM, SECONDS, StepBudget, Steps  # noqa: E402


def fit(timings):
    """The calibration's fit for three steps (a model plugin's, model-template's rows.fit without the bytes)."""
    (r1, _), (r2, _), (_, p) = sorted(timings, key=lambda k: (k[1], k[0]))
    t1, t2, t3 = timings[(r1, 0)], timings[(r2, 0)], timings[(r2, p)]
    b = max(0.0, (t3 - t2) / (r2 * p))
    a = max(0.0, (t2 - t1) / (r2 - r1) - b * (r1 + r2))
    return {"c": max(0.0, t1 - a * r1 - b * r1 * r1), "a": a, "b": b}


class SpanTests(unittest.TestCase):
    def test_spans_are_the_longest_within_the_bound_at_each_position(self):
        costs = fit({(128, 0): 0.02, (512, 0): 0.06, (512, 200_000): 0.9})
        spans = StepBudget(512, prior=costs, seconds=0.5)
        for position in (0, 50_000, 120_000, 200_000):
            found = spans.rows(position)
            self.assertEqual(found % QUANTUM, 0)
            self.assertLessEqual(spans.predicted(found, position), 0.5)
            if found < 512:
                self.assertGreater(spans.predicted(found + QUANTUM, position), 0.5)
        self.assertEqual(spans.rows(0), 512)
        self.assertLess(spans.rows(200_000), 512)
        self.assertEqual(spans.stop(1000), 1000 + spans.rows(1000))

    def test_a_span_leaves_the_room_s_bytes_and_one_without_a_calibration_is_the_chunk(self):
        costs = {"c": 0.001, "a": 1e-5, "b": 0.0}
        self.assertEqual(StepBudget(64, prior=costs, seconds=10.0, room=40 << 20, per_row=1 << 20).rows(0), 32)
        self.assertEqual(StepBudget(64).rows(10 ** 6), 64)

    def test_where_no_span_meets_the_bound_spans_are_the_ones_that_prefill_fastest(self):
        fixed = {"c": 0.6, "a": 2e-4, "b": 0.0}
        spans = StepBudget(1024, prior=fixed, seconds=0.5, deepest=200_000)
        self.assertEqual([spans.rows(p) for p in (0, 100_000, 200_000)], [1024, 1024, 1024])
        self.assertIn("0.60 s of it whatever its span", spans.facts()["over_budget"])
        staged = {"c": 0.45, "a": 4.8e-4, "b": 0.0}
        self.assertEqual(SECONDS, 0.75 * GRACE)
        self.assertEqual((StepBudget(1024, prior=staged, seconds=SECONDS).rows(0),
                          StepBudget(1024, prior=staged, seconds=0.5).rows(0)), (624, 96))


class LoopTests(unittest.TestCase):
    """The budget follows what the engine's steps take: a calibration that underestimates the deep steps (zeroed KV,
    a cache that held more at load) is corrected by the steps that ran past the bound."""

    def serve(self, spans, truth, position, steps=12):
        for _ in range(steps):
            rows = spans.rows(position)
            spans.ran(rows, position)
            spans.observe(truth(rows, position))
        return spans.rows(position)

    def test_deep_steps_that_take_longer_than_the_calibration_said_shrink_their_spans(self):
        prior = {"c": 0.05, "a": 2e-4, "b": 1e-9}            # the load's guess: 0.25 s for 1,024 rows at 200k
        def truth(rows, position):                            # what the steps take: three times the depth cost
            return 0.05 + 2e-4 * rows + 3e-9 * rows * (position + rows)
        spans = StepBudget(1024, prior=prior, seconds=SECONDS, deepest=200_000)
        self.assertEqual(spans.rows(200_000), 1024)            # the guess: the whole chunk fits
        found = self.serve(spans, truth, 200_000)
        self.assertLess(found, 1024)
        self.assertLessEqual(truth(found, 200_000), SECONDS * 1.05)   # the steps end within the bound again
        self.assertGreater(spans.over_bound, 0)
        self.assertGreater(spans.facts()["ratios"][200_000 // 8192 * 8192], 1.0)
        self.assertEqual(spans.rows(0), 1024)                  # shallow steps still take the whole chunk

    def test_spans_whose_boundaries_change_bits_stay_the_calibration_s_and_count_what_overran(self):
        prior = {"c": 0.05, "a": 2e-4, "b": 1e-9}
        spans = StepBudget(1024, prior=prior, seconds=SECONDS, adaptive=False)
        before = spans.rows(200_000)
        self.serve(spans, lambda rows, position: 2.0, 200_000)
        self.assertEqual((spans.rows(200_000), spans.costs, spans.ratios), (before, prior, {}))
        self.assertEqual(spans.over_bound, 12)

    def test_a_step_of_a_pass_counts_the_units_it_ran(self):
        prior = {"c": 0.0, "a": 1e-3, "b": 0.0}               # a chunk through every layer: 1 ms a row
        spans = StepBudget(512, prior=prior, seconds=SECONDS, units=48)
        spans.ran(512, 0, units=4)                             # four layers of the chunk in this step
        spans.observe(4 * 512e-3 / 48 * 2)                     # twice what the form says for four layers
        self.assertAlmostEqual(spans.ratio(0), 2.0)

    def test_the_engine_s_steps_tell_every_budget_and_a_step_without_chunks_teaches_nothing(self):
        steps = Steps()
        first = steps.budget(256, prior={"c": 0.1, "a": 1e-4, "b": 0.0})
        steps.observe(0.5)                                     # no chunk ran: nothing learned
        self.assertEqual(first.observed, 0)
        first.ran(256, 0)
        steps.observe(0.9)
        self.assertEqual((first.observed, first.over_bound), (1, 1))



class StepPlanTests(unittest.TestCase):
    """Core's step plan (Steps): the lanes, the waiting places, a step's bound and the activation reserve are Core's,
    the model's options their bounds, the caller's cap over both; Core calibrates the spans as the plugin's decision."""

    def test_lanes_and_waiting_places_are_cores_within_the_bounds_and_the_cap(self):
        from crit_plugins.steps import LANES, PENDING
        self.assertEqual((Steps().lanes(), Steps().pending()), (LANES, PENDING))
        self.assertEqual((Steps().lanes(8), Steps().pending(5)), (8, 5))   # the model's bound where it names one
        capped = Steps(max_concurrent_requests=5)
        self.assertEqual((capped.lanes(8), capped.pending(4, 5)), (5, 0))
        self.assertEqual(Steps(max_concurrent_requests=6).pending(4, 4), 2)
        self.assertEqual(Steps(max_concurrent_requests=1).lanes(4), 2)     # every engine batches: Core's runtime caps

    def test_a_steps_bound_is_at_most_cores(self):
        self.assertEqual((Steps().seconds(None), Steps().seconds(0), Steps().seconds(5.0)), (SECONDS, SECONDS, SECONDS))
        self.assertEqual(Steps().seconds(0.25), 0.25)
        from crit_plugins.steps import ACTIVATIONS
        self.assertEqual((Steps().activations(), Steps().activations(1 << 20)), (ACTIVATIONS, 1 << 20))

    def test_core_calibrates_the_spans_as_the_plugins_recorded_decision(self):
        from crit_plugins import decisions, policy
        measured = []

        def measure():
            measured.append(1)
            return {"timings": [[64, 0, 0.02], [256, 0, 0.05], [256, 4096, 0.09]], "peaks": [[256, 256 << 10]]}
        engine = policy.Policy(load=2)
        with engine.active():
            steps = Steps()
            budget = steps.calibrated("a-model", 256, key={"shape": 1}, measure=measure, seconds=0.5, deepest=4096)
            again = steps.calibrated("a-model", 256, key={"shape": 1}, measure=measure)
            row = decisions.settled("a-model", "spans")
        self.assertEqual(len(measured), 1)                       # settled once, replayed after
        self.assertEqual(budget.costs, again.costs)
        self.assertEqual(row["choice"]["m"], 1024.0)             # the most bytes a row took
        self.assertEqual((budget.seconds, again.seconds), (0.5, SECONDS))
        self.assertIn(budget, steps.budgets)


if __name__ == "__main__":
    unittest.main()
