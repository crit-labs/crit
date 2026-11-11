"""The run's explicit state: crit run ... next names the one next step from the records, and plan --draft writes the
next plan's skeleton with the facts crit computes for it."""
import contextlib
import io
import json
import unittest

from support.runs import RunFixture
from crit.run.cli import main
from crit.run.records import digest, file_sha, now, read_json


class NextStepTests(RunFixture, unittest.TestCase):
    ladder_scopes = True

    def step(self):
        return self.store.next(self.run_id)

    def test_next_walks_candidate_1_up_its_rung_and_drafts_the_next_plan(self):
        self.assertEqual(self.step()["step"], "capture")
        self.resolve()
        found = self.step()
        self.assertEqual((found["step"], found["command"].split(f"--id {self.run_id} ")[1]),
                         ("plan", "plan --draft --record plan.json"))
        plan_id = self.first_plan()
        self.assertEqual(self.step()["step"], "candidate")
        self.assertIn(plan_id, self.step()["why"])
        candidate = self.store.candidate(self.run_id, self.root, self.proposal(plan_id=plan_id)) \
            if False else self.candidate(plan_id=plan_id, lever_fates={})
        cid = candidate["candidate_id"]
        self.assertEqual((self.step()["step"], self.step()["candidate"]), ("build", cid))
        package = self.package(candidate)
        self.assertEqual(self.step()["step"], "serve_check")
        path = self.directory / "candidates" / cid / "logs" / "serve-check-fixture.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"checks": []}', encoding="utf-8")
        self.store.append(self.run_id, cid, "check", {
            "case_id": "case-0001", "package_sha256": digest(package), "gate": "functional",
            "criteria_sha256": digest(self.run["spec"]["correctness"]), "status": "passed", "rationale": "fixture",
            "evidence": [{"path": path.relative_to(self.directory).as_posix(), "sha256": file_sha(path)}],
            "evidence_level": "artifact_verified", "recorded_at": now(),
            "extensions": {"recorded_by": "crit run serve --check"}})
        found = self.step()
        self.assertEqual((found["step"], found["command"].split(f"--id {self.run_id} ")[1]),
                         ("seal", f"seal --candidate {cid} --lifecycle built"))
        self.assertEqual(found["state"]["candidates"][0]["state"], "serve_checked")
        self.store.seal(self.run_id, cid, "built")
        self.assertEqual(self.step()["step"], "plan")
        # plan --draft writes the skeleton of candidate 2's plan, the plugin round from candidate 1.
        target = self.root / "plan.json"
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--root", str(self.store.root), "--id", self.run_id, "plan", "--draft", "--record",
                         str(target)])
        self.assertEqual(code, 0)
        drafted = json.loads(output.getvalue())
        self.assertEqual(read_json(target), drafted["plan"])
        self.assertEqual(drafted["plan"]["parent_candidate_id"], cid)
        self.assertEqual((drafted["facts"]["objective"]["candidate"], drafted["facts"]["objective"]["name"]),
                         (2, "plugin_round"))
        with self.assertRaisesRegex(ValueError, "exists"):
            self.store.plan_draft(self.run_id, target)
        # The CLI prints the step on stderr and the JSON on stdout.
        errors, output = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.assertEqual(main(["--root", str(self.store.root), "--id", self.run_id, "next"]), 0)
        self.assertTrue(errors.getvalue().startswith("next: plan"))
        self.assertEqual(json.loads(output.getvalue())["step"], "plan")

    def test_a_failed_build_is_a_draft_to_seal(self):
        self.resolve()
        cid = self.candidate(plan_id=self.first_plan(), lever_fates={})["candidate_id"]
        reservation = self.store.reserve(self.run_id, cid, "build")
        self.assertEqual(self.step()["step"], "wait")
        self.store.settle(self.run_id, reservation, {"outcome": "failed", "ended_at": now(), "exit_code": 1,
                                                     "cleanup_settled": True, "logs": [], "failure": "fixture"})
        found = self.step()
        self.assertEqual((found["step"], found["state"]["candidates"][0]["state"]), ("seal", "build_failed"))
        self.assertIn("--lifecycle build_failed", found["command"])


if __name__ == "__main__":
    unittest.main()
