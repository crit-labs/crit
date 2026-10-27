"""crit run capture, build, check --from-eval and serve --check: the workflow steps every run repeats."""
import json
import sys
import unittest

import test_run_measure as run_measure
import test_run_records as run_records
from crit.run import served
from crit.run.build import build
from crit.run.capture import capture
from crit.run.checks import from_eval
from crit.run.measure import check_case_prompts
from crit.run.records import digest, file_sha, now, read_json


class CaptureTests(run_records.RunFixture, unittest.TestCase):
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
pathlib.Path("out/kernels.bin").write_bytes(b"built " + pathlib.Path("engine.py").read_bytes()[:8])
pathlib.Path("tmp").mkdir(exist_ok=True)
pathlib.Path("tmp/object.o").write_bytes(b"intermediate")
print("compiled")
sys.exit(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
"""
ADAPTER = """def declared_capabilities(model_path):
    return ["prefill", "decode"]
"""


class BuildTests(run_records.RunFixture, unittest.TestCase):
    def built_candidate(self):
        if not (self.directory / "inputs.json").exists():
            self.resolve()
        source = self.root / "engine-source"
        source.mkdir(exist_ok=True)
        (source / "build.py").write_text(BUILD, encoding="utf-8")
        (source / "engine.py").write_text(ADAPTER, encoding="utf-8")
        return self.store.candidate(self.run_id, source, self.proposal())["candidate_id"]

    def recipe(self, *args):
        return {"build": {"toolchain": {"python": sys.version.split()[0]},
                          "command": [sys.executable, "-B", "build.py", *args]}, "launch": ["engine.py"]}

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
        self.assertEqual(package["capabilities"], ["prefill", "decode"])   # from the adapter
        self.assertEqual(package["build"]["command"], self.recipe()["build"]["command"])
        self.assertNotIn("tmp/object.o", [row["path"] for row in package["files"]])
        self.assertEqual(self.store.get(self.run_id)["errors"], [])

    def test_a_failed_build_is_settled_failed_with_its_log_and_not_packaged(self):
        candidate = self.built_candidate()
        with self.assertRaisesRegex(ValueError, "Build failed: .* exited 3 \\(log candidates/.*build-attempt-0001.log\\)"
                                                "\\ncompiled"):
            build(self.store, self.run_id, candidate, self.recipe("3"), log=lambda m: None)
        result = read_json(self.directory / "candidates" / candidate / "attempts/attempt-0001/result.json")
        self.assertEqual((result["outcome"], result["exit_code"]), ("failed", 3))
        self.assertFalse((self.directory / "candidates" / candidate / "package").exists())

    def test_capabilities_named_in_the_recipe_are_kept(self):
        candidate = self.built_candidate()
        recipe = {**self.recipe(), "capabilities": ["prefill"]}
        self.assertEqual(build(self.store, self.run_id, candidate, recipe, log=lambda m: None)["package"]["capabilities"],
                         ["prefill"])


def gates(**states):
    names = ("coverage", "repeatable", "request_history", "prompt_cache", "quality", "sampling", "capacity",
             "retrieval", "lower_context", "generality")
    return [{"name": n, "state": states.get(n, "pass"), "detail": n + " detail"} for n in names]


class FromEvalTests(run_records.RunFixture, unittest.TestCase):
    def evaluated(self, gate_list, free=2 * 2**30, requirements=None):
        candidate = self.candidate()
        self.package(candidate)
        candidate = candidate["candidate_id"]
        reservation = self.store.reserve(self.run_id, candidate, "eval")
        path = self.directory / "candidates" / candidate / "eval" / (reservation["attempt_id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"requirements": requirements or {"quality_budget": {
            "ppl_ratio_max": 1.05, "kld_mean_max": 0.05, "kld_p99_max": 0.5}, "exactness": "reported"},
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
        with self.assertRaisesRegex(ValueError, "no retained Eval result for attempt-0009"):
            from_eval(self.store, self.run_id, candidate, "attempt-0009")

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
        self.assertEqual(items, {"consecutive": "passed", "system_message": "passed", "streaming": "passed",
                                 "stop_string": "passed", "cancel": "passed", "stop_engine": "passed"})
        self.assertEqual(check["evidence"][1]["path"], f"candidates/{candidate}/logs/serve-attempt-0001.log")
        self.assertEqual(check["package_sha256"],
                         digest(read_json(self.directory / "candidates" / candidate / "package/manifest.json")))
        self.assertEqual(self.store.get(self.run_id)["errors"], [])


if __name__ == "__main__":
    unittest.main()
