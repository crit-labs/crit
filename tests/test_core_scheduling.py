"""Core schedules an engine's requests (crit_core.native.StepEngine over Core's runtime): waiting requests are
admitted by priority, the oldest first among equals (next_request, the rule inference/core/src/runtime/admission.cpp applies);
the cap on the requests the runtime holds at once is Core's setting (cap_concurrency), and so is how long a request
may wait for a lane."""
from pathlib import Path
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("inference/core/python", "tests"):
    sys.path.insert(0, str(ROOT / _folder))

from crit_core.native import NO_DEADLINE_MS, NativeError, StepEngine, StepResult, cap_concurrency, next_request  # noqa: E402
from support.engines import STEP_LIBRARY  # noqa: E402


class RuleTests(unittest.TestCase):
    def test_the_lowest_priority_value_goes_first_then_arrival(self):
        self.assertEqual(next_request([5, 0, -1, 0]), 2)
        self.assertEqual(next_request([0, 0, 0]), 0)
        self.assertEqual(next_request([3, 1, 1]), 1)
        self.assertIsNone(next_request([]))

    def test_the_cap_takes_lanes_first_then_waiting_places(self):
        model = {"max_concurrency": 4, "max_pending": 2}
        self.assertEqual(cap_concurrency(model, None), model)
        self.assertEqual(cap_concurrency(model, 5), {"max_concurrency": 4, "max_pending": 1})
        self.assertEqual(cap_concurrency(model, 3), {"max_concurrency": 3, "max_pending": 0})
        self.assertEqual(cap_concurrency({}, 3), {"max_concurrency": 3, "max_pending": 0})   # Core's defaults: 8, 8
        with self.assertRaises(NativeError):
            cap_concurrency(model, 0)


class Session:
    def __init__(self, request):
        self.first, self.output, self.made = request.prompt[0], request.output_tokens, 0


class Model:
    """One lane unless told otherwise; admission waits while `ready` is unset; open() records the order."""

    def __init__(self, lanes=1, pending=4):
        self.lanes, self.pending = lanes, pending
        self.ready, self.opened = threading.Event(), []

    def info(self):
        return {"implementation": "scheduling toy", "vocabulary": 64, "context": 256, "max_concurrency": self.lanes,
                "max_pending": self.pending, "max_batch_tokens": 0}

    def admission(self, request):
        return "ready" if self.ready.is_set() else "deferred"

    def open(self, request):
        self.opened.append(request.prompt[0])
        return Session(request)

    def step(self, rows):
        out = []
        for row in rows:
            row.session.made += 1
            out.append(StepResult(0 if row.prefill_complete else row.prefill_tokens, [row.session.first], True))
        return out


@unittest.skipUnless(STEP_LIBRARY, "Core's runtime (crit_core_steps) is not built")
class EngineTests(unittest.TestCase):
    def submitted(self, engine):
        """A counter of the requests Core's runtime took (crit_engine_request_open returned)."""
        count, changed, opened = [0], threading.Condition(), engine.lib.crit_engine_request_open

        def counting(*args):
            found = opened(*args)
            with changed:
                count[0] += 1
                changed.notify_all()
            return found
        engine.lib.crit_engine_request_open = counting

        def wait(n):
            with changed:
                self.assertTrue(changed.wait_for(lambda: count[0] >= n, 5), f"{n} requests did not reach Core")
        return wait

    def test_waiting_requests_are_admitted_by_priority_then_arrival(self):
        model = Model()
        engine = StepEngine(model, STEP_LIBRARY)
        self.addCleanup(engine.close)
        wait, threads = self.submitted(engine), []
        for n, (first, priority) in enumerate(((1, 5), (2, 0), (3, -1), (4, 0)), 1):
            threads.append(threading.Thread(target=engine.generate, args=([first], 1),
                                            kwargs={"stop_tokens": [], "priority": priority}))
            threads[-1].start()
            wait(n)   # each waits in Core's queue before the next arrives
        model.ready.set()
        for thread in threads:
            thread.join(5)
        self.assertEqual(model.opened, [3, 2, 4, 1])

    def test_the_cap_is_cores_runtime_setting(self):
        engine = StepEngine(Model(lanes=4, pending=2), STEP_LIBRARY, max_concurrent_requests=3)
        self.addCleanup(engine.close)
        self.assertEqual((engine.info()["max_concurrency"], engine.info()["max_concurrent_requests"]), (3, 3))

    def test_a_request_waits_for_a_lane_as_long_as_the_caller_allows(self):
        engine = StepEngine(Model(), STEP_LIBRARY, pending_timeout=0.05)
        self.addCleanup(engine.close)
        with self.assertRaisesRegex(NativeError, "pending request deadline"):
            engine.generate([1], 1, stop_tokens=[])
        self.assertGreater(NO_DEADLINE_MS, 30_000)   # without one, no deadline Core's old default (30 s) would hit


if __name__ == "__main__":
    unittest.main()
