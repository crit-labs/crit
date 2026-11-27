"""crit run serve: a packaged candidate served through crit serve before it is sealed, under a check attempt that
settles when the server stops."""
import json
import threading
import unittest
import urllib.request
from unittest.mock import patch

from test_run_measure import MeasureFixture
from crit.run import served
from crit.run.records import read_json


class RunServeTests(MeasureFixture, unittest.TestCase):
    evaluated = False   # serving takes a check attempt on a packaged candidate
    def serve_in_thread(self, candidate):
        ready, stop, out = threading.Event(), threading.Event(), {}

        def bound(port):
            out["port"] = port
            ready.set()

        def run():
            try:
                out["result"] = served.serve(self.store, self.run_id, candidate, port=0, ready=bound, stop=stop)
            except BaseException as exc:  # reported by the test
                out["error"] = exc
                ready.set()
        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(ready.wait(120), "the server never listened")
        self.assertNotIn("error", out)
        return thread, stop, out

    def chat(self, port, content, stream=False):
        body = json.dumps({"messages": [{"role": "user", "content": content}], "max_tokens": 4,
                           "stream": stream}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode()

    def test_serves_the_unsealed_candidate_until_stopped_and_settles_its_check_attempt(self):
        candidate = self.served()
        thread, stop, out = self.serve_in_thread(candidate)
        try:
            reply = json.loads(self.chat(out["port"], "hello"))
            self.assertEqual(reply["choices"][0]["message"]["content"], "xxxx")
            self.assertIn("data:", self.chat(out["port"], "hello again", stream=True))
        finally:
            stop.set()
            thread.join(120)
        self.assertNotIn("error", out)
        reservation = read_json(self.folder / "attempts/attempt-0001/reservation.json")
        self.assertEqual(reservation["operation"], "check")
        result = self.attempt()
        self.assertEqual((result["outcome"], result["cleanup_settled"]), ("completed", True))
        self.assertEqual(out["result"], result)
        log = self.directory / result["logs"][0]["path"]
        self.assertIn("not sealed", log.read_text(encoding="utf-8"))

    def test_a_terminated_serve_settles_as_interrupted_and_exits(self):
        """crit run turns SIGTERM into SystemExit (run/cli.py): the check attempt settles as interrupted."""
        candidate = self.served()
        with patch.object(served.Served, "open", side_effect=SystemExit(143)):
            with self.assertRaises(SystemExit):
                served.serve(self.store, self.run_id, candidate, port=0)
        result = self.attempt()
        self.assertEqual((result["outcome"], result["cleanup_settled"]), ("interrupted", True))

    def test_refuses_a_sealed_or_unpackaged_candidate(self):
        unpackaged = self.candidate()["candidate_id"]
        with self.assertRaisesRegex(ValueError, "Package the candidate"):
            served.serve(self.store, self.run_id, unpackaged, port=0)
        candidate = self.served()
        self.store.seal(self.run_id, candidate, "built")
        with self.assertRaisesRegex(ValueError, "sealed and takes no attempt: crit serve serves it .crit serve "
                                                "--workspace .* --candidate candidate-"):
            served.serve(self.store, self.run_id, candidate, port=0)



class CoreItemsTests(unittest.TestCase):
    """crit run serve --check's items from Core's counters over every load the check served: a compile after load
    fails, and so does a drafting engine that kept none of its drafts."""

    def items(self, *counts):
        from crit.run.checks import core_items
        return {item["name"]: item for item in core_items(list(counts), lambda line: None)}

    def test_a_compile_after_load_fails_the_check(self):
        clean = {"compiles_after_load": {"count": 0, "first": []}}
        self.assertEqual(self.items(clean, clean)["compiles_after_load"]["state"], "passed")
        compiled = {"compiles_after_load": {"count": 2, "first": ["Triton compile of k.prompt(x: *bf16) constexprs "
                                                                  "BLOCK=64", "Triton compile of k.prompt(...)"]}}
        item = self.items(clean, compiled)["compiles_after_load"]
        self.assertEqual(item["state"], "failed")
        self.assertIn("2 kernel compiles or extension builds after load", item["detail"])
        self.assertIn("k.prompt(x: *bf16) constexprs BLOCK=64", item["detail"])
        self.assertEqual(self.items({})["compiles_after_load"]["state"], "unverified")   # an engine Core did not load

    def test_a_drafting_engine_that_keeps_none_of_its_drafts_fails_the_check(self):
        def drafts(rounds, accepted, drafted=None):
            return {"drafts": {"requests": 3, "rounds": rounds, "accepted": accepted, "drafted": drafted}}
        self.assertEqual(self.items({})["drafts_accepted"]["state"], "passed")   # no request drafted
        dead = self.items(drafts(200, 0, 600), drafts(85, 0, 255))["drafts_accepted"]
        self.assertEqual(dead["state"], "failed")
        self.assertIn("none of the drafts of 285 decode rounds was accepted (855 proposed)", dead["detail"])
        self.assertEqual(self.items(drafts(10, 0))["drafts_accepted"]["state"], "unverified")   # too few to judge
        working = self.items(drafts(100, 81, 300))["drafts_accepted"]
        self.assertEqual(working["state"], "passed")
        self.assertIn("81 drafts accepted over 100 decode rounds (27% of 300 proposed)", working["detail"])
        self.assertIn("81 drafts accepted over 140 decode rounds",
                      self.items(drafts(100, 81, 300), drafts(40, 0))["drafts_accepted"]["detail"])


    def test_a_step_past_the_arena_or_the_headroom_fails_the_check(self):
        def steps(**counts):
            return {"steps": {"arena": 800 << 20, "steps": 40, "refused": 0, "overflows": 0, "breaches": 0,
                              "least_free": 600 << 20, **counts}}
        self.assertEqual(self.items({})["step_memory"]["state"], "unverified")
        self.assertEqual(self.items(steps(unaccounted="no Core allocator here"))["step_memory"]["state"],
                         "unverified")
        clean = self.items(steps(), steps())["step_memory"]
        self.assertEqual(clean["state"], "passed")
        self.assertIn("80 steps, arena 800 MiB, least free 600 MiB", clean["detail"])
        broken = self.items(steps(), steps(overflows=2, breaches=1, least_free=100 << 20))["step_memory"]
        self.assertEqual(broken["state"], "failed")
        self.assertIn("2 segments past the step arena", broken["detail"])
        self.assertIn("1 segments that left less free than the VRAM headroom", broken["detail"])
        self.assertIn("least free 100 MiB", broken["detail"])
        self.assertIn("refused", self.items(steps(refused=1))["step_memory"]["detail"])
        # A step past the arena names the most each kind of step took, largest first, over every load.
        kinds = self.items(steps(overflows=1, most={"decode": 100 << 20, "prompt": 900 << 20}),
                           steps(most={"prompt": 1200 << 20, "score": 300 << 20}))["step_memory"]["detail"]
        self.assertIn("the most a step took, by kind: prompt 1,200 MiB, score 300 MiB, decode 100 MiB", kinds)
        self.assertNotIn("by kind", clean["detail"])

    def test_step_memory_sets_aside_memory_other_programs_took(self):
        # Another program took 200 MiB of the card while the engine served: the device read 237 MiB left after a
        # segment, under the 256 MiB headroom, but by the engine's own use it left 437 MiB.
        mib = 1 << 20

        def steps(segments, at_serve=300 * mib, breaches=None):
            return {"steps": {"headroom": 256 * mib, "arena": 800 << 20, "steps": 40, "refused": 0, "overflows": 0,
                              "breaches": sum(1 for left, _ in segments if left < 256 * mib) if breaches is None
                              else breaches, "least_free": min(left for left, _ in segments),
                              "segment_memory": [[left, other] for left, other in segments],
                              "outside_at_serve": at_serve}}
        other_program = self.items(steps([(600 * mib, 300 * mib), (237 * mib, 500 * mib)]))["step_memory"]
        self.assertEqual(other_program["state"], "passed", other_program["detail"])
        self.assertIn("least free 437 MiB", other_program["detail"])
        self.assertIn("memory outside the engine held 300 to 500 MiB", other_program["detail"])
        self.assertIn("1 segments left less than the headroom by the device's reading", other_program["detail"])
        # the outside memory flat: the engine's own segment broke the headroom, as before
        own = self.items(steps([(600 * mib, 300 * mib), (237 * mib, 300 * mib)]))["step_memory"]
        self.assertEqual(own["state"], "failed")
        self.assertIn("1 segments that left less free than the VRAM headroom", own["detail"])
        self.assertIn("least free 237 MiB", own["detail"])
        # a Core that keeps no readings (or cut them): judged by the device's counts
        old = steps([(237 * mib, 500 * mib)])
        old["steps"].pop("segment_memory")
        self.assertEqual(self.items(old)["step_memory"]["state"], "failed")
        cut = steps([(237 * mib, 500 * mib)])
        cut["steps"]["segment_memory_cut"] = True
        self.assertEqual(self.items(cut)["step_memory"]["state"], "failed")


class CloseItemTests(unittest.TestCase):
    """close_returns_memory: every engine close a check made in the resident host, as crit serve reports each one
    (last_close), read into Served.closes once per close."""

    def test_a_close_that_kept_memory_fails_the_check_and_one_judged_close_passes_it(self):
        from crit.run.checks import close_item
        passed = {"returned": True, "judged": True, "message": "returned", "close": 1}
        kept = {"returned": False, "judged": True, "close": 2,
                "message": "4 captured graph(s) alive (core/graphs), held by StepEngine.rounds"}
        self.assertEqual(close_item([passed], lambda line: None)["state"], "passed")
        item = close_item([passed, kept], lambda line: None)
        self.assertEqual(item["state"], "failed")
        self.assertIn("1 of 2 engine closes did not return their memory; the first: 4 captured graph(s)", item["detail"])
        unjudged = {"returned": None, "judged": False, "message": "not judged: 1 other engine(s) are open", "close": 1}
        self.assertEqual(close_item([unjudged], lambda line: None)["state"], "unverified")
        self.assertEqual(close_item([], lambda line: None)["state"], "unverified")

    def test_served_reads_each_close_once_and_logs_one_that_kept_memory(self):
        logged = []
        found = served.Served.__new__(served.Served)
        found.closes, found.log = [], logged.append

        class Engine:
            close = None

            def last_close(self):
                return self.close
        engine = Engine()
        self.assertIsNone(found.closed(engine))   # no close yet
        engine.close = {"returned": True, "judged": True, "message": "returned", "close": 1}
        found.closed(engine, 8192)
        found.closed(engine, 4096)   # a reload that failed before it closed the engine: the same close
        engine.close = {"returned": False, "judged": True, "message": "elastic range a/b is open", "close": 2}
        found.closed(engine)
        self.assertEqual([(c["close"], c["reopened_at"]) for c in found.closes], [(1, 8192), (2, None)])
        self.assertEqual(logged, ["crit serve: the closed engine did not return its memory: elastic range a/b is open"])


class HostTests(unittest.TestCase):
    def test_crit_run_reloads_each_engine_in_a_host_of_its_own_unless_asked_for_one_host(self):
        # one engine, one process where crit run reloads engines; --same-host keeps one host for the command
        from unittest import mock
        from crit.run import cli
        self.assertIs(served.Served.one_process, True)
        self.addCleanup(setattr, served.Served, "one_process", True)
        with mock.patch.object(cli, "act", lambda command, args: 0):
            cli.main(["--same-host", "--id", "run-x", "show"])
        self.assertIs(served.Served.one_process, False)


class ReopenTests(unittest.TestCase):
    def test_a_reopen_closes_the_engine_first_also_at_the_same_context(self):
        # The host keeps an engine whose spec it already holds, so a reopen at the same context and configuration
        # (the preflight's reopen_same_context) closes it first: Core judges that close, and the reload builds it again.
        import threading
        from crit_api.engine import ServedEngine
        calls = []

        class Runtime:
            def status(self):
                return {"engine": {"candidate_id": "c"}}

            def call(self, action, payload, stop, timeout, **_):
                calls.append(action)
                return {"status": "completed", "error": None}
        engine = ServedEngine.__new__(ServedEngine)
        engine.lock, engine.closed, engine._stopping = threading.RLock(), False, threading.Event()
        engine.attempt, engine.specs, engine.runtime, engine.timeout = "attempt-0001", {}, Runtime(), 5
        engine.context_tokens = engine.configuration = None
        engine.spec = lambda candidate_id=None: {"candidate_id": "c"}
        engine._load_default = lambda spec, stop: calls.append("load")
        engine.reopen(None)
        self.assertEqual(calls, ["close", "load"])


class EvalCloseTests(unittest.TestCase):
    def test_evals_close_of_a_served_engine_closes_it_in_the_host_and_returns_cores_report(self):
        # crit run eval: Eval's lower-context round reads the close of the engine it opened (context.lower.close),
        # which crit serve's host makes and Core judges; an external candidate's close says nothing.
        import threading
        from crit.run.evaluation import EvalEngines, Opening
        logged = []
        found = EvalEngines.__new__(EvalEngines)
        found.closes, found.log, found.lock = [], logged.append, threading.Lock()
        kept = {"returned": False, "judged": True, "message": "4 captured graph(s) alive", "close": 1}

        class Engine:
            closed_in_host = 0

            def close_engine(self):
                self.closed_in_host += 1
                return kept

            def last_close(self):
                return kept if self.closed_in_host else None
        engine = Engine()
        found.current = (engine, object())
        report = found.close_report()
        self.assertEqual(engine.closed_in_host, 1)
        self.assertEqual((report["returned"], report["message"]), (False, "4 captured graph(s) alive"))
        self.assertEqual(found.closes, [report])
        self.assertIn("did not return its memory", logged[-1])
        found.current = None
        self.assertIsNone(found.close_report())   # nothing served: nothing closed
        self.assertIsNone(Opening.close_report(found))


class PlanKvTests(unittest.TestCase):
    """A plan's kv_bytes_per_token is checked against the engine's facts at its first serve check."""

    def test_a_plan_that_budgets_another_kv_is_refused_and_a_plan_states_a_positive_figure(self):
        from crit.run import plans
        from crit.run.checks import plan_kv
        required = dict.fromkeys(plans.FIELDS)
        for wrong in (0, 12288.5, "12288"):
            with self.assertRaisesRegex(ValueError, "kv_bytes_per_token is the device bytes"):
                plans._fields({**required, "kv_bytes_per_token": wrong})
        # The build's recon priced FP8 at 12,288 bytes a token; the engine served BF16 and indexer keys at 29,952.
        state, detail = plan_kv(12288, {"kv_bytes_per_token": 29952})
        self.assertEqual(state, "failed")
        self.assertIn("29,952 bytes a token, the candidate's plan budgeted 12,288 (2.44x)", detail)
        self.assertEqual(plan_kv(29952, {"kv_bytes_per_token": 29952})[0], "passed")
        self.assertEqual(plan_kv(29952, {})[0], "failed")


if __name__ == "__main__":
    unittest.main()
