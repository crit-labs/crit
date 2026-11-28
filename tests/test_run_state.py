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


class TransitionTableTests(RunFixture, unittest.TestCase):
    """The one transition table (crit.run.state.OPERATIONS): every state has a legal action, next names only commands
    the table allows in the candidate's state, and every operation refuses outside its states with the table's reason;
    an interrupted attempt (an ablate's, unsettled) holds every other operation until it is settled."""

    def test_every_state_has_a_legal_action_and_each_step_is_one_the_table_allows(self):
        from crit.run import state
        for name in state.CANDIDATE_STATES:
            with self.subTest(state=name):
                legal = [op for op, states in state.OPERATIONS.items() if name in states]
                # running waits or settles, sealed plans from it: neither starts an operation on the candidate
                self.assertTrue(legal or name in state.NEVER, name)
                self.assertIn(name, state.WHY)
                row = {"candidate": "candidate-0009", "state": name, "detail": "fixture"}
                for op in state.OPERATIONS:
                    refused = state.refusal(op, row)
                    self.assertEqual(refused is None, name in state.OPERATIONS[op], (op, name))
                    if refused:
                        self.assertIn(state.WHY[name].split(":")[0].format(cid="candidate-0009", detail="fixture"),
                                      refused)
        for step, op in state.STEP_OPERATION.items():
            self.assertIn(op, state.OPERATIONS, step)

    def test_next_names_a_command_the_table_allows_on_the_open_candidate(self):
        from crit.run import state
        self.resolve()
        cid = self.candidate(plan_id=self.first_plan(), lever_fates={})["candidate_id"]
        found = self.store.next(self.run_id)
        row = found["state"]["candidates"][0]
        self.assertEqual((found["step"], row["state"]), ("build", "created"))
        self.assertIsNone(state.refusal(state.STEP_OPERATION[found["step"]], row))
        # show lists the same state and the operations the table allows in it
        self.assertEqual(self.store.get(self.run_id)["candidate_states"][cid],
                         {"state": "created", "allowed": ["build", "seal"]})

    def test_reserve_holds_every_caller_to_the_phase_order(self):
        # No Eval before a passed serve --check, no measurement before a completed Eval, for every caller of reserve
        self.resolve()
        cid = self.candidate(plan_id=self.first_plan(), lever_fates={})["candidate_id"]
        with self.assertRaisesRegex(ValueError, "is created, where crit run's eval does not start: it is not built"):
            self.store.reserve(self.run_id, cid, "eval")
        self.package({"candidate_id": cid})
        with self.assertRaisesRegex(ValueError, "is packaged, where crit run's eval does not start: its serve --check "
                                                "has not passed"):
            self.store.reserve(self.run_id, cid, "eval")
        self.serve_checked(cid)
        with self.assertRaisesRegex(ValueError, "is serve_checked, where crit run's benchmark does not start: it has "
                                                "no completed Eval"):
            self.store.reserve(self.run_id, cid, "benchmark")
        self.eval_completed(cid)
        self.store.reserve(self.run_id, cid, "benchmark")   # evaluated: a measurement starts

    def test_an_unsettled_attempt_holds_every_operation_with_the_tables_reason(self):
        # 2026-11-23: an ablation interrupted before it settled left its attempt reserved; nothing may start beside
        # it, and next says to wait for it or settle it.
        self.resolve()
        cid = self.candidate(plan_id=self.first_plan(), lever_fates={})["candidate_id"]
        reservation = self.store.reserve(self.run_id, cid, "build")
        self.assertEqual(self.store.next(self.run_id)["step"], "wait")
        for op in ("build", "check", "eval", "benchmark"):
            with self.subTest(operation=op), self.assertRaisesRegex(ValueError, "is running, where crit run's "
                                                                                f"{op} does not start: .*settle"):
                self.store.reserve(self.run_id, cid, op)
        with self.assertRaisesRegex(ValueError, "is running, where crit run's seal does not start"):
            self.store.seal(self.run_id, cid, "build_failed", failure="fixture")
        self.store.settle(self.run_id, reservation, {"outcome": "interrupted", "ended_at": now(), "exit_code": 130,
                                                     "cleanup_settled": True, "logs": [], "failure": "fixture"})
        self.store.reserve(self.run_id, cid, "build")   # settled: the table's build starts again


class EvaluatedStepTests(unittest.TestCase):
    """An evaluated candidate's next step is one its command takes: where its checks keep its benchmarks exploratory,
    measure refuses it and a completed seal needs them, so next names seal built, and the refusals name it too."""

    def test_an_evaluated_candidate_whose_benchmarks_cannot_qualify_is_sealed_built(self):
        from test_run_plans import PlanFixture, plan, status

        class Fixture(PlanFixture, unittest.TestCase):
            def runTest(self):
                pass
        f = Fixture()
        f.setUp()
        self.addCleanup(f.doCleanups)
        f.resolve()
        cid = f.candidate(plan_id=f.store.plan(f.run_id, plan(levers=[], forecast=None))["plan_id"])["candidate_id"]
        package = f.package(read_json(f.directory / "candidates" / cid / "candidate.json"))
        path = f.directory / "candidates" / cid / "logs" / "check-fixture.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"checks": []}', encoding="utf-8")

        def check(status_, by):
            f.store.append(f.run_id, cid, "check", {
                "case_id": "case-0001", "package_sha256": digest(package), "gate": "functional",
                "criteria_sha256": digest(f.run["spec"]["correctness"]), "status": status_, "rationale": "fixture",
                "evidence": [{"path": path.relative_to(f.directory).as_posix(), "sha256": file_sha(path)}],
                "evidence_level": "artifact_verified", "recorded_at": now(), "extensions": {"recorded_by": by}})
        check("passed", "crit run serve --check")
        f.evaluated(cid, status())
        self.assertEqual(f.store.next(f.run_id)["step"], "record_checks")
        check("unverified", "crit run check --from-eval")   # Eval could not verify its functional check
        found = f.store.next(f.run_id)
        self.assertEqual((found["step"], found["state"]["candidates"][-1]["state"]), ("seal", "evaluated"))
        self.assertIn(f"seal --candidate {cid} --lifecycle built", found["command"])
        self.assertIn("cannot qualify", found["why"])
        from crit.run.measure import refusal, unqualified   # what crit run measure refuses it with
        said = refusal(cid, unqualified(f.directory, f.run, cid))
        self.assertRegex(said, f"seal it built .*seal --candidate {cid} --lifecycle built")
        with self.assertRaisesRegex(ValueError, "seal it built"):
            f.store.seal(f.run_id, cid, "completed")
        f.store.seal(f.run_id, cid, "built")   # the step next named is one the store takes
        self.assertEqual(f.store.next(f.run_id)["step"], "plan")


if __name__ == "__main__":
    unittest.main()
