"""crit run capture, build, check --from-eval and serve --check: the workflow steps every run repeats."""
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_run_measure as run_measure
from support.runs import RunFixture, with_plugins
from crit.run import served
from crit.run.build import build
from crit.run.capture import capture
from crit.run.checks import from_eval
from crit.run.measure import check_case_prompts
from crit.run.records import digest, file_sha, now, read_json


class CaptureTests(RunFixture, unittest.TestCase):
    model_files = {"README.md": "# Fixture model\n"}

    def test_a_captured_case_renders_to_the_workload_and_resolves(self):
        inputs = self.root / "inputs.json"
        inputs.write_text(json.dumps({"hardware_manifest": {"device": "host"},
                                      "environment_manifest": {"benchmark": "synthetic"}}), encoding="utf-8")
        result = capture(self.store, self.run_id, "case-0001", system="Answer briefly.", inputs=inputs)
        self.assertEqual(result["prompt_tokens"], 512)
        path = self.directory / result["row"]["path"]
        self.assertEqual(result["row"], {"path": "captured/case-0001/request.json", "bytes": path.stat().st_size,
                                         "sha256": file_sha(path)})
        request = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(request["messages"][0], {"role": "system", "content": "Answer briefly."})
        payload = read_json(inputs)
        self.assertEqual(payload["case_manifests"], {"case-0001": [result["row"]]})
        # resolve renders it exactly as candidates serve it
        check_case_prompts(self.directory, self.run, payload["case_manifests"], result["template_variables"])
        self.assertIn("case_manifests", self.store.freeze_inputs(self.run_id, payload))
        with self.assertRaisesRegex(ValueError, "inputs are resolved"):
            capture(self.store, self.run_id, "case-0001")

    def test_a_given_text_too_short_or_an_unknown_case_is_refused(self):
        with self.assertRaisesRegex(ValueError, "renders to \\d+ prompt tokens, the workload has 512; give a longer"):
            capture(self.store, self.run_id, "case-0001", text="three short words")
        with self.assertRaisesRegex(ValueError, "no case case-0002"):
            capture(self.store, self.run_id, "case-0002")


BUILD = """import pathlib, sys
pathlib.Path("out").mkdir(exist_ok=True)
pathlib.Path("out/kernels.bin").write_bytes(b"built " + pathlib.Path("kernel.cu").read_bytes()[:8])
pathlib.Path("tmp").mkdir(exist_ok=True)
pathlib.Path("tmp/object.o").write_bytes(b"intermediate")
print("compiled")
sys.exit(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
"""
KERNEL = "__global__ void toy() {}\n"


class BuildTests(RunFixture, unittest.TestCase):
    def built_candidate(self):
        if not (self.directory / "inputs.json").exists():
            self.resolve()
        source = with_plugins(self.root / "engine-source")
        (source / "build.py").write_text(BUILD, encoding="utf-8")
        (source / "kernel.cu").write_text(KERNEL, encoding="utf-8")
        return self.store.candidate(self.run_id, source, self.proposal())["candidate_id"]

    def recipe(self, *args):
        return {"build": {"toolchain": {"python": sys.version.split()[0]},
                          "command": [sys.executable, "-B", "build.py", *args]}}

    def test_a_build_settles_its_attempt_and_packages_its_outputs(self):
        candidate = self.built_candidate()
        result = build(self.store, self.run_id, candidate, self.recipe(), drop=["tmp"], log=lambda m: None)
        attempt = result["attempt"]
        self.assertEqual((attempt["outcome"], attempt["exit_code"], attempt["cleanup_settled"]), ("completed", 0, True))
        log = self.directory / attempt["logs"][0]["path"]
        self.assertEqual(attempt["logs"][0]["path"], f"candidates/{candidate}/logs/build-attempt-0001.log")
        self.assertIn("compiled", log.read_text())
        self.assertEqual(result["build_folder"], f"work/builds/{candidate}-attempt-0001")
        package = result["package"]
        built = self.directory / "candidates" / candidate / "package/out/kernels.bin"
        self.assertEqual(package["build"]["dependencies"], [
            {"name": "out/kernels.bin", "version": "built by attempt-0001", "path": "out/kernels.bin",
             "sha256": file_sha(built)}])
        self.assertFalse({"launch", "capabilities"} & set(package))   # what it serves is Core's to say
        self.assertEqual(package["build"]["command"], self.recipe()["build"]["command"])
        self.assertNotIn("tmp/object.o", [row["path"] for row in package["files"]])
        self.assertEqual(self.store.get(self.run_id)["errors"], [])

    def test_a_build_command_reads_the_candidates_configuration(self):
        # The build folder holds the candidate's configuration.json beside its plugins (a check of the selection, a
        # build reading an option), and the package carries the same file, bound as no build output.
        candidate = self.built_candidate()
        reader = [sys.executable, "-B", "-c", "import json; json.load(open('configuration.json'))['plugins']['use']"]
        result = build(self.store, self.run_id, candidate, {"build": {**self.recipe()["build"], "command": reader}},
                       log=lambda m: None)
        self.assertEqual(result["attempt"]["outcome"], "completed")
        configuration = read_json(self.directory / "candidates" / candidate / "candidate.json")["configuration"]
        packaged = self.directory / "candidates" / candidate / "package/configuration.json"
        self.assertEqual(read_json(packaged), configuration)
        self.assertNotIn("configuration.json", [d["path"] for d in result["package"]["build"]["dependencies"]])

    def test_a_source_that_carries_its_configuration_as_written_builds_into_the_candidates(self):
        # A source keeps the configuration.json the candidate was made from, before crit completed it (the exactness
        # of its rows, the quality mode): the build folder gets the candidate's, so the package carries that one and
        # nothing has to be dropped. A source whose copy names other values is still refused, with what to do.
        if not (self.directory / "inputs.json").exists():
            self.resolve()
        proposal = self.proposal()
        source = with_plugins(self.root / "engine-source")
        (source / "build.py").write_text(BUILD, encoding="utf-8")
        (source / "kernel.cu").write_text(KERNEL, encoding="utf-8")
        (source / "configuration.json").write_text(json.dumps(proposal["configuration"]), encoding="utf-8")
        candidate = self.store.candidate(self.run_id, source, proposal)["candidate_id"]
        configuration = read_json(self.directory / "candidates" / candidate / "candidate.json")["configuration"]
        self.assertNotEqual(configuration, proposal["configuration"])   # crit completed it
        result = build(self.store, self.run_id, candidate, self.recipe(), drop=["tmp"], log=lambda m: None)
        self.assertEqual(result["attempt"]["outcome"], "completed")
        self.assertEqual(read_json(self.directory / "candidates" / candidate / "package/configuration.json"),
                         configuration)
        (source / "configuration.json").write_text(json.dumps({**proposal["configuration"], "batch": 9}),
                                                   encoding="utf-8")
        other = self.store.candidate(self.run_id, source, self.proposal(configuration={**proposal["configuration"],
                                                                                        "batch": 2}))["candidate_id"]
        with self.assertRaisesRegex(ValueError, "differs from the candidate configuration in batch.*"
                                                "--drop configuration.json"):
            build(self.store, self.run_id, other, self.recipe(), drop=["tmp"], log=lambda m: None)

    def test_a_failed_build_is_settled_failed_with_its_log_and_not_packaged(self):
        candidate = self.built_candidate()
        with self.assertRaisesRegex(ValueError, "Build failed: .* exited 3 \\(log candidates/.*build-attempt-0001.log\\)"
                                                "\\ncompiled"):
            build(self.store, self.run_id, candidate, self.recipe("3"), log=lambda m: None)
        result = read_json(self.directory / "candidates" / candidate / "attempts/attempt-0001/result.json")
        self.assertEqual((result["outcome"], result["exit_code"]), ("failed", 3))
        self.assertFalse((self.directory / "candidates" / candidate / "package").exists())

    def test_a_recipe_names_its_build_only(self):
        """A launch entry or capabilities are no recipe's: Core's loader runs the package and says what it serves."""
        candidate = self.built_candidate()
        for extra in ({"launch": ["engine.py"]}, {"capabilities": ["chat-json-v1"]}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "Unknown fields"):
                build(self.store, self.run_id, candidate, {**self.recipe(), **extra}, log=lambda m: None)
        self.assertFalse((self.directory / "candidates" / candidate / "attempts").exists())


def gates(**states):
    names = ("coverage", "repeatable", "request_history", "prompt_cache", "quality", "sampling", "capacity",
             "retrieval", "lower_context", "concurrency", "generality")
    return [{"name": n, "state": states.get(n, "pass"), "detail": n + " detail"} for n in names]


class FromEvalTests(RunFixture, unittest.TestCase):
    def evaluated(self, gate_list, free=2 * 2**30, requirements=None):
        candidate = self.candidate()
        self.package(candidate)
        candidate = candidate["candidate_id"]
        reservation = self.store.reserve(self.run_id, candidate, "eval")
        path = self.directory / "candidates" / candidate / "eval" / (reservation["attempt_id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        # The run's terms: its request names no quality mode, so speed and Eval's budget for it.
        path.write_text(json.dumps({"requirements": requirements or {"quality_budget": {
            "ppl_ratio_max": 1.4, "kld_mean_max": 0.4, "kld_p99_max": 4.0}, "exactness": "reported",
            "quality_mode": "speed"},
            "context": {"device_memory": {"free_bytes": free}}, "status": {"gates": gate_list}}), encoding="utf-8")
        self.store.settle(self.run_id, reservation, {"outcome": "completed", "ended_at": now(), "exit_code": 0,
                                                     "cleanup_settled": True, "logs": [], "failure": None})
        return candidate, path

    def test_each_gate_is_judged_by_the_eval_gates_its_criteria_name(self):
        candidate, path = self.evaluated(gates(generality="fail"))
        checks = from_eval(self.store, self.run_id, candidate)
        self.assertEqual([(c["gate"], c["status"]) for c in checks],
                         [("arithmetic", "passed"), ("functional", "passed"), ("resources", "passed")])
        self.assertEqual(checks[0]["evidence"], [{"path": path.relative_to(self.directory).as_posix(),
                                                  "sha256": file_sha(path)}])
        self.assertIn("quality pass: quality detail; sampling pass", checks[0]["rationale"])
        self.assertIn("GiB free at the required context", checks[2]["rationale"])
        view = self.store.get(self.run_id)
        self.assertEqual(view["errors"], [])
        self.assertEqual(view["candidates"][0]["gates"], {"arithmetic": "passed", "functional": "passed",
                                                          "resources": "passed"})

    def test_a_failed_or_unmeasured_gate_and_low_memory_carry_over(self):
        candidate, _ = self.evaluated(gates(sampling="fail", retrieval="unmeasured"), free=2**20)
        checks = {c["gate"]: c["status"] for c in from_eval(self.store, self.run_id, candidate)}
        self.assertEqual(checks, {"arithmetic": "failed", "functional": "unverified", "resources": "failed"})

    def test_a_result_under_other_quality_terms_is_refused(self):
        candidate, _ = self.evaluated(gates(), requirements={"quality_budget": {"ppl_ratio_max": 1.5}, "exactness": "reported"})
        with self.assertRaisesRegex(ValueError, "ran under quality terms .* evaluate the candidate again"):
            from_eval(self.store, self.run_id, candidate)
        # The run's budget judged in the other mode, or in no stated mode, is refused: the modes' forks, honesty bounds
        # and prefill limits differ.
        budget = {"ppl_ratio_max": 1.4, "kld_mean_max": 0.4, "kld_p99_max": 4.0}
        for requirements in ({"quality_budget": budget, "exactness": "reported", "quality_mode": "preserve"},
                             {"quality_budget": budget, "exactness": "reported"}):
            # Each in a run of its own: every candidate's first plan starts a run without a candidate on the ladder.
            self.run_id = self.store.create(self.data)["run"]["run_id"]
            self.directory = self.store.directory(self.run_id)
            other, _ = self.evaluated(gates(), requirements=requirements)
            with self.assertRaisesRegex(ValueError, "the run's are .*'quality_mode': 'speed'"):
                from_eval(self.store, self.run_id, other)
        with self.assertRaisesRegex(ValueError, "no retained Eval result for attempt-0009"):
            from_eval(self.store, self.run_id, other, "attempt-0009")

    def test_the_cli_records_from_eval_or_a_record_but_not_neither(self):
        import contextlib
        import io
        from crit.run.cli import main
        candidate, _ = self.evaluated(gates())
        prefix = ["--root", str(self.store.root), "--id", self.run_id, "check", "--candidate", candidate]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(prefix + ["--from-eval"]), 0)
        self.assertEqual([c["gate"] for c in json.loads(out.getvalue())], ["arithmetic", "functional", "resources"])
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            main(prefix)
        self.assertIn("check needs --record or --from-eval", err.getvalue())


class RunCacheTests(unittest.TestCase):
    def test_a_runs_engines_get_the_runs_plugin_cache_unless_the_user_named_one(self):
        # The engine's host inherits the environment: crit run names a folder of the run there for files plugins can
        # build again (an expert store), a later run its own, and a value the user set stays (empty: no cache).
        import os
        from pathlib import Path
        from crit_core.engine import PLUGIN_CACHE
        with patch.dict(os.environ), patch.object(served.run_cache, "given", None):
            os.environ.pop(PLUGIN_CACHE, None)
            served.run_cache(Path("runs") / "run-a")
            self.assertEqual(os.environ[PLUGIN_CACHE], str(Path("runs") / "run-a" / ".plugin-cache"))
            served.run_cache(Path("runs") / "run-b")
            self.assertEqual(os.environ[PLUGIN_CACHE], str(Path("runs") / "run-b" / ".plugin-cache"))
            for own in ("", "/srv/stores"):
                os.environ[PLUGIN_CACHE] = own
                served.run_cache(Path("runs") / "run-c")
                self.assertEqual(os.environ[PLUGIN_CACHE], own)


class ServeCheckTests(run_measure.MeasureFixture, unittest.TestCase):
    def test_serve_check_runs_the_serving_checks_and_records_a_functional_check(self):
        candidate = self.served()
        result = served.serve(self.store, self.run_id, candidate, port=0, check=True)
        self.assertEqual(result["attempt"]["outcome"], "completed")
        (check,) = result["checks"]
        self.assertEqual((check["gate"], check["status"]), ("functional", "passed"), check["rationale"])
        report = self.directory / check["evidence"][0]["path"]
        self.assertEqual(check["evidence"][0]["path"], f"candidates/{candidate}/logs/serve-check-attempt-0001.json")
        items = {i["name"]: i["state"] for i in read_json(report)["checks"]}
        smoke = {name: state for name, state in items.items() if name.startswith("smoke_")}
        self.assertTrue(smoke)
        self.assertNotIn("failed", smoke.values(), smoke)
        # The package's plugin tests run first, one item per plugin.
        tested = {name: state for name, state in items.items() if name.startswith("plugin_test_")}
        self.assertTrue(tested)
        self.assertEqual(set(tested.values()), {"passed"}, tested)
        self.assertEqual({k: v for k, v in items.items() if k not in smoke and k not in tested},
                         {"consecutive": "passed", "system_message": "passed", "streaming": "passed",
                          "stop_string": "passed", "cancel": "passed", "draft_bytes": "passed", "stop_engine": "passed",
                          # The candidate's plan declared its prompt activations, and this engine has none to differ.
                          "plan_activations": "passed",
                          # A scripted adapter's steps are not Core's, so nothing probes them, no compile watch runs
                          # and no request drafts.
                          "full_context_prompt_step": "unverified", "full_context_device_memory": "unverified",
                          "compiles_after_load": "unverified", "drafts_accepted": "passed", "step_memory": "unverified"})
        self.assertEqual(check["evidence"][1]["path"], f"candidates/{candidate}/logs/serve-attempt-0001.log")
        self.assertEqual(check["package_sha256"],
                         digest(read_json(self.directory / "candidates" / candidate / "package/manifest.json")))
        self.assertEqual(self.store.get(self.run_id)["errors"], [])

    def test_an_engine_that_serves_other_prompt_activations_than_its_plan_declared_fails_the_check(self):
        from crit.run import checks

        def run(planned, served):
            class Client:
                def __init__(self, port, timeout):
                    pass

                def chat(self, body):
                    return {"choices": [{"message": {"content": "a plain reply"}, "finish_reason": "stop"}],
                            "usage": {"completion_tokens": 3}}

                def stream(self, body, cancel_after_first=False):
                    return "a plain reply", "stop"

                def models(self):
                    return [{"id": "m", "engine_facts": {"prompt_activations": served} if served else {}}]
            with patch.object(checks, "Client", Client):
                items = checks.serve_checks(0, [], {}, 60, lambda line: None, planned_activations=planned)
            return {i["name"]: i for i in items}.get("plan_activations")
        # One activation for every quantized format, or one a format, against what the engine's facts declare.
        self.assertEqual(run("stored", {"mxfp8": "stored", "nvfp4": "stored"})["state"], "passed")
        self.assertEqual(run({"mxfp8": "bf16", "nvfp4": "stored"}, {"mxfp8": "bf16", "nvfp4": "stored"})["state"],
                         "passed")
        found = run("stored", {"mxfp8": "bf16", "nvfp4": "stored"})
        self.assertEqual(found["state"], "failed")
        self.assertIn('"mxfp8": "bf16"', found["detail"])
        self.assertEqual(run("stored", None)["state"], "passed")   # no quantized format's prompt GEMMs
        self.assertIsNone(run(None, {"mxfp8": "bf16"}))            # no plan declared any: nothing to hold it to

    def test_the_full_context_pass_judges_the_slowest_prompt_step_and_the_least_free_memory(self):
        from crit.run import checks
        asked = []

        class Client:
            def __init__(self, port, timeout):
                pass

            def models(self):
                return [{"id": "m"}]

            def chat(self, body):
                copies = body["messages"][0]["content"].count(checks.FILLER)
                asked.append((copies, body["max_tokens"]))
                return {"usage": {"prompt_tokens": 7 + 30 * copies}}

        def run(probe, headroom=2 << 30):
            # Before the request the engine had probed one slow warm-up step of each kind, which is not judged.
            warm = {kind: {"steps": 1, "recent": [[5.0, 1]]} for kind in ("prompt", "decode")}
            after = None if probe is None else {
                kind: {"steps": 1 + value["steps"], "recent": [[5.0, 1]] + [[value["slowest_seconds"],
                                                                             value["min_free_device_bytes"]]] *
                       value["steps"]} for kind, value in probe.items()}
            reads = iter([warm if probe is not None else None, after])
            engine = SimpleNamespace(telemetry=lambda: {"core": {"probe": next(reads)}})
            with patch.object(checks, "Client", Client):
                return {i["name"]: i for i in checks.full_context_checks(0, engine, 1000, headroom, 1, 60,
                                                                           lambda line: None)}
        fits = {"prompt": {"steps": 4, "slowest_seconds": 0.4, "min_free_device_bytes": 3 << 30},
                "decode": {"steps": 16, "slowest_seconds": 0.01, "min_free_device_bytes": 5 << 30}}
        found = run(fits)
        # Sized from one copy (37 tokens) and two (67): 1 + (1000 - 16 - 37) // 30 copies, then 16 tokens decoded.
        self.assertEqual(asked, [(1, 8), (2, 8), (32, 16)])
        self.assertEqual((found["full_context_prompt_step"]["state"], found["full_context_device_memory"]["state"]),
                         ("passed", "passed"), found)
        self.assertIn("up to a 967-token prompt, took 0.400 s", found["full_context_prompt_step"]["detail"])
        slow = {**fits, "prompt": {**fits["prompt"], "slowest_seconds": 1.3}}
        self.assertEqual(run(slow)["full_context_prompt_step"]["state"], "failed")
        self.assertIn("prefill in shorter steps", run(slow)["full_context_prompt_step"]["detail"])
        # Below the headroom on a prompt step, or reporting no probe at all.
        self.assertEqual(run(fits, headroom=4 << 30)["full_context_device_memory"]["state"], "failed")
        self.assertEqual({i["state"] for i in run(None).values()}, {"unverified"})

    def test_the_full_context_memory_is_judged_by_the_engine_not_by_what_others_on_the_card_hold(self):
        # Core's probe samples [seconds, free at the step's peak, memory outside the engine]: another program that
        # takes memory during a step lowers the device's reading but not the engine's verdict; an engine that leaves
        # too little with the outside memory at its least still fails.
        from crit.run import checks
        gib, mib = 1 << 30, 1 << 20

        class Client:
            def __init__(self, port, timeout):
                pass

            def models(self):
                return [{"id": "m"}]

            def chat(self, body):
                return {"usage": {"prompt_tokens": 7 + 30 * body["messages"][0]["content"].count(checks.FILLER)}}

        def run(prompt, decode, headroom=1 * gib):
            # A warm-up step of each kind before the request (outside memory at 1 GiB then): not judged.
            warm = {kind: {"steps": 1, "recent": [[5.0, 0, 1 * gib]]} for kind in ("prompt", "decode")}
            after = {kind: {"steps": 1 + len(samples), "recent": warm[kind]["recent"] + samples}
                     for kind, samples in (("prompt", prompt), ("decode", decode))}
            reads = iter([warm, after])
            engine = SimpleNamespace(telemetry=lambda: {"core": {"probe": next(reads)}})
            with patch.object(checks, "Client", Client):
                found = {i["name"]: i for i in checks.full_context_checks(0, engine, 1000, headroom, 1, 60,
                                                                            lambda line: None)}
            return found["full_context_device_memory"]
        decode = [[0.01, 3 * gib // 2, 2300 * mib]] * 4
        # The outside memory swings by 600 MiB during the prompt: the card reads 0.9 GiB free at that step, under the
        # 1 GiB headroom, but the engine itself held only 50 MiB more than at its other steps.
        swings = [[0.4, 3 * gib // 2, 2300 * mib], [0.4, 3 * gib // 2 - 650 * mib, 2900 * mib],
                  [0.4, 3 * gib // 2, 2300 * mib]]
        found = run(swings, decode)
        self.assertEqual(found["state"], "passed", found)
        self.assertIn("prompt 1.45 GiB", found["detail"])
        self.assertIn("held 2,300 to 2,900 MiB during the request", found["detail"])
        self.assertIn("600 MiB above its least was subtracted", found["detail"])
        self.assertIn("the device read 0.87 GiB free there", found["detail"])
        # The same swing at another step, the engine leaving 0.9 GiB at its own peak with the outside memory at its
        # least: the engine breaks the headroom.
        breaks = swings + [[0.4, 9 * gib // 10, 2300 * mib]]
        found = run(breaks, decode)
        self.assertEqual(found["state"], "failed", found)
        self.assertIn("prompt 0.90 GiB", found["detail"])
        self.assertIn("0 MiB above its least was subtracted", found["detail"])
        # A Core that samples no outside memory is judged by the device's reading, as before.
        found = run([[0.4, 3 * gib // 2 - 600 * mib]], [[0.01, 3 * gib // 2]])
        self.assertEqual(found["state"], "failed", found)
        self.assertNotIn("outside the engine", found["detail"])

    def test_the_full_context_wait_fails_a_prefill_that_cannot_finish_in_the_time_left(self):
        import threading
        from crit.run import checks
        clock, counted, lines = [0.0], [0], []
        engine = SimpleNamespace(telemetry=lambda: {"core": {"prompt_tokens": counted[0], "steps": 1}})

        def polls(rate, left, seconds, engine=engine):
            # The wait's polls, every five seconds, of an engine prefilling `rate` tokens a second.
            clock[0], counted[0] = 0.0, 1000   # tokens of earlier requests are not this prompt's
            watch = checks._prefill_watch(engine, 1000, 200_000, lambda: left - clock[0], lines.append,
                                          clock=lambda: clock[0])
            for second in range(5, seconds + 1, 5):
                clock[0], counted[0] = float(second), 1000 + rate * second
                watch()
        # 37 tokens a second: the 200,000-token prompt needs an hour and a half, the reservation has 45 minutes. The
        # wait says so once the prompt steps it timed span 30 seconds, not at the deadline.
        with self.assertRaisesRegex(checks.PrefillTooSlow, r"prefill too slow, about 37 tok/s: 1,295 of about "
                                                           r"200,000 prompt tokens are in, the rest needs \d+ min"):
            polls(37, 2700, 600)
        self.assertEqual(clock[0], 35.0)
        # 1,700 tokens a second fits: the wait logs the rate every minute and lets the request run.
        polls(1700, 2700, 100)
        self.assertEqual(len(lines), 1)
        self.assertIn("about 1,700 tok/s", lines[0])
        # An engine that reports no count is not judged.
        polls(37, 2700, 600, engine=SimpleNamespace(telemetry=lambda: {"core": {"steps": 1}}))
        # What the watch raises closes the request's connection (crit serve cancels it) and is the wait's error.
        closed = threading.Event()
        client = SimpleNamespace(abort=closed.set)

        def hung():
            closed.wait(5)
            raise OSError("connection closed")

        def slow():
            raise checks.PrefillTooSlow("prefill too slow, about 37 tok/s")
        with self.assertRaisesRegex(checks.PrefillTooSlow, "about 37 tok/s"):
            checks._progressing(hung, client, engine, stall=5, poll=0.02, watch=slow)
        self.assertTrue(closed.is_set())

    def test_the_full_context_request_waits_while_the_engine_steps_and_fails_when_it_stalls(self):
        import threading
        import time
        from crit.run import checks

        class Steps:
            # Core's step count: moving while `moving` is set.
            def __init__(self):
                self.count, self.moving = 0, threading.Event()
                self.moving.set()

            def telemetry(self):
                if self.moving.is_set():
                    self.count += 1
                return {"core": {"steps": self.count}}

        class Waited:
            # The client: abort closes the connection its request waits on.
            def __init__(self):
                self.aborted, self.closed = False, threading.Event()

            def abort(self):
                self.aborted = True
                self.closed.set()

            def hung(self):
                self.closed.wait(5)
                raise OSError("connection closed")

        # A request lasting many times `stall` returns while the steps keep moving.
        engine, client = Steps(), Waited()

        def slow():
            time.sleep(0.3)
            return "answered"
        self.assertEqual(checks._progressing(slow, client, engine, stall=0.1, poll=0.02), "answered")
        self.assertFalse(client.aborted)
        # A request whose engine stops stepping is aborted (crit serve cancels it) after `stall` without a step.
        engine.moving.clear()

        started = time.perf_counter()
        with self.assertRaisesRegex(TimeoutError, "no engine step finished in 0.1 s"):
            checks._progressing(client.hung, client, engine, stall=0.1, poll=0.02)
        self.assertTrue(client.aborted)
        self.assertLess(time.perf_counter() - started, 2)

        # The request's own error is its error.
        def refused():
            raise ValueError("crit serve answered 500")
        with self.assertRaisesRegex(ValueError, "answered 500"):
            checks._progressing(refused, Waited(), engine, stall=0.1, poll=0.02)
        # An engine that reports no step count is waited for `stall` seconds in all.
        quiet = Waited()
        with self.assertRaisesRegex(TimeoutError, "reports no step count"):
            checks._progressing(quiet.hung, quiet, SimpleNamespace(telemetry=lambda: None), stall=0.1, poll=0.02)
        self.assertTrue(quiet.aborted)

    def test_serve_check_runs_crit_smoke_at_the_run_context_and_a_quarter_of_it(self):
        # Smoke's pass at the run's context comes right after the serving checks' short requests (before the full
        # context's prompt grows the engine's pools) and its pass at the lower context after the gate pre-check,
        # where the pre-check left the engine: one reopen for both.
        candidate = self.served()
        calls = []

        def smoke(served_, port, log, reference=None, full=True, lower=True):
            calls.append(("run" if full else "") + ("lower" if lower else ""))
            return ([{"name": "smoke_causal", "state": "passed", "detail": "13 prefixes"}] if full else []) + (
                [{"name": "smoke_causal@49152", "state": "failed", "detail": "prefix of 17 tokens"}] if lower else [])

        def precheck(*args):
            calls.append("precheck")
            return []

        def full_context(*args, **options):
            calls.append("full context")
            return []
        with patch.object(served, "_smoke", smoke), patch.object(served, "_precheck", precheck),                 patch("crit.run.checks.full_context_checks", full_context):
            result = served.serve(self.store, self.run_id, candidate, port=0, check=True, data=self.directory)
        (check,) = result["checks"]
        self.assertEqual(calls, ["run", "full context", "precheck", "lower"])
        self.assertEqual(check["status"], "failed")
        self.assertIn("smoke_causal@49152 failed", check["rationale"])
        calls.clear()
        with patch.object(served, "_smoke", smoke):
            result = served.serve(self.store, self.run_id, candidate, port=0, check=True, smoke=False)
        self.assertEqual(calls, [])
        self.assertEqual(result["checks"][0]["status"], "passed", result["checks"][0]["rationale"])

    def test_smoke_starts_at_the_context_the_engine_is_open_at_and_is_not_opened_again_to_go_back(self):
        opened = []
        fake = SimpleNamespace(kv_tokens=262144, opened_at=65536, timeout=60,
                               open=lambda tokens=None: opened.append(tokens) or (None, None, 9))
        ran = []
        with patch("crit_smoke.run", lambda client, log=None, reference=None: ran.append(reference) or
                   [{"name": "causal", "state": "passed", "detail": ""}]):
            items = served._smoke(fake, 9, lambda line: None, {"texts": "recon's"})
        self.assertEqual([item["name"] for item in items], ["smoke_causal@65536", "smoke_causal"])
        self.assertEqual(ran, [{"texts": "recon's"}] * 2)   # recon's smoke reference judges both passes
        self.assertEqual(opened, [262144])                 # once, to the run's context; not back to the lower one
        fake.opened_at, opened[:] = 262144, []
        with patch("crit_smoke.run", lambda client, log=None, reference=None: [{"name": "causal", "state": "passed",
                                                                               "detail": ""}]):
            items = served._smoke(fake, 9, lambda line: None)
        self.assertEqual([item["name"] for item in items], ["smoke_causal", "smoke_causal@65536"])
        self.assertEqual(opened, [65536])
        # serve --check's two passes: the run's context where the engine is, then the lower one, opened once
        fake.opened_at, opened[:] = 262144, []
        with patch("crit_smoke.run", lambda client, log=None, reference=None: [{"name": "causal", "state": "passed",
                                                                               "detail": ""}]):
            first = served._smoke(fake, 9, lambda line: None, lower=False)
            self.assertEqual(opened, [])
            fake.opened_at = 65536     # the pre-check left it at the lower context
            second = served._smoke(fake, 9, lambda line: None, full=False)
        self.assertEqual([item["name"] for item in first + second], ["smoke_causal", "smoke_causal@65536"])
        self.assertEqual(opened, [])

    def test_serve_check_with_data_adds_the_gate_precheck(self):
        candidate = self.served()
        with self.assertRaisesRegex(ValueError, "--data is for crit run serve --check"):
            served.serve(self.store, self.run_id, candidate, port=0, data=self.directory)
        failed = [{"name": "precheck_long_honesty", "state": "failed", "detail": "curve@32768 differs"}]
        with patch.object(served, "_precheck", lambda *args: list(failed)):
            result = served.serve(self.store, self.run_id, candidate, port=0, check=True, data=self.directory)
        (check,) = result["checks"]
        self.assertEqual(check["status"], "failed")
        self.assertIn("precheck_long_honesty failed", check["rationale"])

    def test_the_gate_precheck_judges_drift_by_the_runs_budget(self):
        # The pre-check's quality item judges as crit run eval does: the run's budget over its mode's, not Eval's
        # defaults.
        from crit.run.records import load_run, quality_terms
        root = self.store.directory(self.run_id)
        data = root / "eval-data"
        data.mkdir()
        (data / "prepared.json").write_text("{}", encoding="utf-8")
        asked = {}
        with patch("crit_eval.run.precheck.run_precheck", lambda *args, **options: asked.update(options) or []):
            served._precheck(SimpleNamespace(kv_tokens=4096, server=None), load_run(root), root, data, 9,
                             lambda line: None)
        run = load_run(root)
        self.assertEqual(asked["quality_budget"], quality_terms(run["spec"]["correctness"])[0])
        self.assertIn("kld_mean_max", asked["quality_budget"])


if __name__ == "__main__":
    unittest.main()
