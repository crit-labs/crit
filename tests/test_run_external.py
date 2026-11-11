"""crit run on an external candidate: an engine crit did not assemble from plugins, declared with its launch command
(configuration.engine, crit.run.external) and recorded like any candidate. The engine is support/external_server.py,
a tiny OpenAI-compatible server started by the launch command; it starts a child process of its own, so every test
also checks that crit stops the engine's whole process tree."""
import json
import os
from pathlib import Path
import shutil
import sys
import time
import unittest
from unittest.mock import patch

from test_run_measure import MeasureFixture
from support.serve import RECIPE
from crit.process.ownership import process_identity
from crit.run import ablate, evaluation, external, measure, served
from crit.run.records import read_json

SERVER = Path(__file__).resolve().parent / "support" / "external_server.py"
FAST = {"fast": {"flag": "--fast", "switch": True, "default": False}}


class ExternalFixture(MeasureFixture):
    def external(self, options=None, render="chars", child=True, context=True, extra=(), checks=True):
        """A packaged external candidate whose launch command starts the toy server; with `checks`, its checks
        recorded and passed, as crit run ... check --from-eval leaves them before measure."""
        if not (self.directory / "inputs.json").exists():
            self.resolve()
        source = self.root / "external-source"
        source.mkdir(exist_ok=True)
        shutil.copyfile(SERVER, source / "server.py")
        self.pids = self.root / "pids.txt"
        command = [sys.executable, "-B", "server.py", "--port", "{port}", "--pids", str(self.pids), "--render", render,
                   *(["--child"] if child else []), *(["--context", "{context_tokens}"] if context else []), *extra]
        self.engine = {"kind": "external", "launch": {"command": command, "ready_seconds": 60},
                       **({"options": options} if options else {})}
        self.record = self.store.candidate(self.run_id, source, {
            "label": "External engine", "rationale": "An engine with its own server", "strategy": "external",
            "configuration": {"engine": self.engine}, "store_decisions": {},
            "plan_id": self.first_plan(engine="external"), "lever_fates": {}})
        candidate = self.record["candidate_id"]
        self.manifest = self.store.package(self.run_id, candidate, source, RECIPE)
        if checks:
            self.checks(self.record, self.manifest)
        self.folder = self.directory / "candidates" / candidate
        return candidate

    def started(self):
        """Every process the engine's launches started, by role: {"server": [...], "child": [...]}, and the
        contexts they were started at."""
        found = {"server": [], "child": [], "context": [], "fast": []}
        for line in self.pids.read_text(encoding="utf-8").splitlines():
            role, value = line.split(" ", 1)
            found[role].append(value)
        return found

    def assert_tree_gone(self):
        started = self.started()
        pids = [int(p) for p in started["server"] + started["child"]]
        self.assertTrue(pids)
        deadline = time.monotonic() + 10
        while any(process_identity(pid) is not None for pid in pids) and time.monotonic() < deadline:
            time.sleep(.1)
        self.assertEqual([pid for pid in pids if process_identity(pid) is not None], [],
                         "a process of the engine's tree outlived its attempt")


class ExternalCandidateTests(ExternalFixture, unittest.TestCase):
    def test_serve_check_measure_and_cleanup_run_the_engine_from_its_launch_command(self):
        candidate = self.external()
        result = served.serve(self.store, self.run_id, candidate, port=0, check=True)
        self.assert_tree_gone()
        report = read_json(self.directory / f"candidates/{candidate}/logs/serve-check-attempt-0001.json")["checks"]
        states = {item["name"]: item["state"] for item in report}
        for name in ("consecutive", "system_message", "streaming", "stop_string", "cancel", "stop_engine"):
            self.assertEqual(states[name], "passed", (name, report))
        self.assertFalse(any(name.startswith("plugin_test") for name in states), "an external engine has no plugins")
        self.assertIn("process tree", next(i["detail"] for i in report if i["name"] == "stop_engine"))
        (check,) = result["checks"]
        self.assertEqual((check["gate"], check["status"]), ("functional", "passed"))
        attempt = read_json(self.folder / "attempts/attempt-0001/result.json")
        self.assertEqual((attempt["outcome"], attempt["cleanup_settled"]), ("completed", True))
        logs = [row["path"] for row in attempt["logs"]]
        self.assertIn(f"candidates/{candidate}/logs/engine-attempt-0001.log", logs)
        self.assertIn("toy engine: serving", (self.directory / logs[logs.index(
            f"candidates/{candidate}/logs/engine-attempt-0001.log")]).read_text(encoding="utf-8"))
        self.assertEqual(self.started()["context"], ["8192"] * len(self.started()["server"]))

        (record,) = measure.measure(self.store, self.run_id, candidate)
        self.assertEqual(record["extensions"]["served_by"], external.SERVED_BY)
        self.assertEqual(len(record["metrics"]["decode_tps"]["samples"]), 3)
        self.assertIn("accepted_tokens_per_round", record["missing_metrics"])
        self.assert_tree_gone()
        server = int(self.started()["server"][-1])
        self.assertNotEqual(server, os.getpid())

    def test_eval_measures_the_engine_over_its_url_and_starts_it_again_at_another_context(self):
        import crit_eval
        sys.path.append(str(Path(crit_eval.__file__).resolve().parents[1] / "tests"))
        from test_eval_server import Text, write_data
        from crit_eval.run import chat
        from crit_eval.quality import prepare
        for patcher in (patch.object(chat, "load_text", lambda model, chat_template=None: Text()),
                        patch.object(chat, "ids_by_bytes",
                                     lambda tokenizer: {chr(i).encode(): i for i in range(1, 0xD800) if i != 0xFFFD})):
            patcher.start()
            self.addCleanup(patcher.stop)
        candidate = self.external(render="text", checks=False)
        data = self.root / "eval-data"
        data.mkdir()
        write_data(data)
        prepared = json.loads((data / "prepared.json").read_text())
        prepared["model"] = prepare.model_identity(self.model)
        prepared["speed"] = [case for case in prepared["speed"] if not case["tools"]]
        (data / "prepared.json").write_text(json.dumps(prepared))
        measured = evaluation.evaluate(self.store, self.run_id, candidate, data, ("speed",))
        result = read_json(self.directory / measured["result"]["path"])
        self.assertEqual(result["server"], f"external {self.run_id}/{candidate}")
        self.assertTrue(result["speed"], result.keys())
        self.assert_tree_gone()

        # Eval's lower-context gate asks for another context: the engine is started again at it.
        from crit.run.attempt import Attempt
        attempt = Attempt(self.store, self.run_id, candidate, "check", None, "context", "context", "context")
        with attempt.running() as log:
            engines = evaluation.eval_engines(self.store, self.run_id, candidate, attempt, 8192, prepared, log,
                                              attempt.deadline_at)
            with engines.until(attempt.deadline_at):
                engines.open(8192)
                engines.open(8192)   # the engine already running at it
                engines.open(4096)
        attempt.settle()
        attempt.finish()
        self.assertEqual(self.started()["context"][-2:], ["8192", "4096"])
        self.assert_tree_gone()

    def test_ablation_serves_the_declared_options_and_says_which_pays(self):
        candidate = self.external(options=FAST, extra=("--delay-ms", "8"))
        found = ablate.ablate(self.store, self.run_id, candidate, repetitions=2, smoke=False)
        names = [row["name"] for row in found["variants"]]
        self.assertEqual(names, ["full", "set:fast=true"])
        self.assertEqual({row["state"] for row in found["variants"]}, {"measured"})
        (verdict,) = found["blame"]
        # The option halves the time per token: decode is what it changes (a request's first token moves with the
        # machine's load as much).
        self.assertEqual(verdict["plugin"], "set:fast=true")
        self.assertIn(verdict["verdict"], ("pays", "trades"), verdict)
        self.assertGreater(verdict["in_engine"]["decode_tps"], 0.3, verdict)
        self.assertEqual(self.started()["fast"], ["False", "True"])
        self.assert_tree_gone()
        report = read_json(self.directory / found["report"])
        self.assertEqual((report["engine"], report["options"]), ("external", FAST))

        # --set names only declared options; another one is refused, the run goes on.
        found = ablate.ablate(self.store, self.run_id, candidate, repetitions=1, smoke=False, sets=["fast=true",
                                                                                                    "width=3"])
        states = {row["name"]: row["state"] for row in found["variants"]}
        self.assertEqual(states, {"full": "measured", "set:fast=true": "measured", "set:width=3": "refused"})

    def test_an_engine_that_declares_no_option_has_no_ablation_and_its_report_says_so(self):
        candidate = self.external()
        found = ablate.ablate(self.store, self.run_id, candidate, smoke=False)
        self.assertEqual(found["ablation"], "unavailable")
        self.assertIn("declares no option", found["reason"])
        self.assertEqual(found["variants"], [])
        report = read_json(self.directory / found["report"])
        self.assertEqual(report["ablation"], "unavailable")
        self.assertFalse(self.pids.exists(), "nothing was started")

    def test_kernels_trace_the_engine_or_say_why_they_cannot(self):
        candidate = self.external()
        with patch.object(external, "_toolkit", lambda nvcc=None: ("no CUDA toolkit with CUPTI found", None)):
            found = external.kernels(self.store, self.run_id, candidate)
        self.assertEqual((found["available"], found["reason"]), (False, "no CUDA toolkit with CUPTI found"))
        report = read_json(self.directory / found["report"])
        self.assertEqual(report["available"], False)
        attempt = read_json(self.folder / "attempts/attempt-0001/result.json")
        self.assertEqual(attempt["outcome"], "completed")
        if not sys.platform.startswith("linux"):
            self.assertIn("Linux", external._toolkit()[0])

    def test_an_engine_that_exits_or_outlives_its_reservation_leaves_no_process(self):
        candidate = self.external(extra=("--exit", "3"))
        with self.assertRaisesRegex(ValueError, "(?s)exited 3 before /v1/models answered.*refusing to start"):
            served.serve(self.store, self.run_id, candidate, port=0, check=True)
        attempt = read_json(self.folder / "attempts/attempt-0001/result.json")
        self.assertEqual((attempt["outcome"], attempt["cleanup_settled"]), ("failed", True))

        # crit run serve without --check serves until its reservation's deadline, then stops the whole tree.
        self.pids.unlink(missing_ok=True)
        candidate = self.external(checks=False)
        with self.assertRaisesRegex(ValueError, "timed_out: crit run serve passed its reserved deadline"):
            served.serve(self.store, self.run_id, candidate, port=0, seconds=8)
        attempt = read_json(self.directory / f"candidates/{candidate}/attempts/attempt-0001/result.json")
        self.assertEqual((attempt["outcome"], attempt["cleanup_settled"]), ("timed_out", True))
        self.assert_tree_gone()

    def test_refusals_keep_the_kinds_apart(self):
        self.resolve()
        source = self.root / "mixed"
        (source / "plugins").mkdir(parents=True)
        (source / "server.py").write_text("print()", encoding="utf-8")
        engine = {"kind": "external", "launch": {"command": ["python", "server.py", "--port", "{port}"]}}
        proposal = {"label": "x", "rationale": "x", "strategy": "x", "store_decisions": {}, "lever_fates": {},
                    "plan_id": self.first_plan(engine="external"), "configuration": {"engine": engine}}
        with self.assertRaisesRegex(ValueError, "holds plugins/"):
            self.store.candidate(self.run_id, source, proposal)
        shutil.rmtree(source / "plugins")
        with self.assertRaisesRegex(ValueError, "takes \\{port\\}"):
            self.store.candidate(self.run_id, source, {**proposal, "configuration": {"engine": {
                "kind": "external", "launch": {"command": ["python", "server.py"]}}}})
        with self.assertRaisesRegex(ValueError, "plans a plugin package"):
            self.store.candidate(self.run_id, source, {**proposal, "plan_id": self.first_plan()})
        with self.assertRaisesRegex(ValueError, 'engine is "external"'):
            self.first_plan(engine="another")


if __name__ == "__main__":
    unittest.main()
