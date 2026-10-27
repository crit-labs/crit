"""Replays: a run as it stood after one candidate's Eval, rendered with today's Eval, recording nothing."""
import contextlib
from datetime import datetime, timezone
import io
import json
import os
import sys
import time
import unittest
from unittest.mock import patch

from test_run_plans import PlanFixture, plan, status
from crit.run import replay
from crit.run.cli import main
from crit.run.records import read_json
from crit.run.store import RunStore

LATER = 3600   # seconds after the cutoff at which the fixture's later records were written


def later(*paths):
    """Date the files (and every file under the folders) as written an hour after now."""
    stamp = time.time() + LATER
    for path in paths:
        for item in ([path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]):
            os.utime(item, (stamp, stamp))


class ReplayTests(PlanFixture, unittest.TestCase):
    plan_rules = 3

    def setUp(self):
        super().setUp()
        self.parent = self.first()["candidate_id"]
        (self.directory / "work" / "early.txt").write_text("measured before the cutoff", encoding="utf-8")
        (self.directory / "work" / "engine.so").write_bytes(b"\x7fELF build product")
        (self.directory / "work" / "plan-draft.json").write_text(json.dumps(plan(self.parent)), encoding="utf-8")
        # After candidate-0001's Eval the agent planned and built candidate-0002, and wrote more evidence.
        recorded = self.store.plan(self.run_id, plan(self.parent))
        self.candidate(plan_id=recorded["plan_id"], parent_candidate_id=self.parent)
        (self.directory / "work" / "later.txt").write_text("measured after the cutoff", encoding="utf-8")
        later(self.directory / "plans" / recorded["plan_id"], self.directory / "candidates" / "candidate-0002",
              self.directory / "work" / "later.txt", *sorted((self.directory / "events").glob("event-*.json"))[-3:])
        self.out = self.root / "replays"

    def rendered(self, **kpis):
        """crit-eval's status of a result rendered again: the stored fixture status with other KPI values."""
        return (patch("crit_eval.status.status_of", lambda path, protocol, baseline=None: status(**kpis)),
                patch("crit_eval.status.render", lambda value: "status rendered again"))

    def make(self, count=1, **kpis):
        renders = self.rendered(**kpis)
        with renders[0], renders[1]:
            return replay.make(self.store, self.run_id, self.parent, self.out, count)

    def test_a_replay_is_the_run_as_it_stood_when_the_candidates_eval_finished(self):
        made = self.make(count=2)
        self.assertEqual([r["root"] for r in made["replays"]], [str(self.out / "replay-1"), str(self.out / "replay-2")])
        folder = self.out / "replay-1" / self.run_id
        self.assertEqual(sorted(p.name for p in (folder / "plans").iterdir()), ["plan-0001"])
        self.assertEqual(sorted(p.name for p in (folder / "candidates").iterdir()), [self.parent])
        self.assertTrue((folder / "work" / "early.txt").is_file())
        self.assertFalse((folder / "work" / "later.txt").exists())
        self.assertFalse((folder / "work" / "engine.so").exists())   # build products stay behind
        self.assertFalse((folder / "work" / "plan-draft.json").exists())   # so does a draft of the next plan
        marker = read_json(folder / replay.MARKER)
        self.assertEqual((marker["run_id"], marker["at"]), (self.run_id, self.parent))
        self.assertEqual(marker["statuses"][marker["result"]]["text"], f"replay/status/{self.parent}.txt")
        self.assertEqual((folder / "replay" / "status" / f"{self.parent}.txt").read_text(encoding="utf-8"),
                         "status rendered again\n")
        brief = " ".join((folder / "replay" / "agent_request.md").read_text(encoding="utf-8").split())
        for line in (f"Replay of crit run {self.run_id} at {self.parent}", "A replay builds nothing",
                     f"--id {self.run_id} plan --record {folder}/replay/plan.json --check", "`next.md`",
                     "only this folder, crit and crit-eval"):
            self.assertIn(line, brief)
        view = RunStore(self.out / "replay-1", self.store.project).get(self.run_id)
        self.assertEqual((view["errors"], view["state"]["phase"]), ([], "awaiting_agent"))
        self.assertEqual([c["candidate_id"] for c in view["candidates"]], [self.parent])
        # The run itself is untouched, and a second replay into the same folder is refused.
        self.assertEqual(sorted(p.name for p in (self.directory / "plans").iterdir()), ["plan-0001", "plan-0002"])
        with self.assertRaisesRegex(ValueError, "exists: replay into a new folder"):
            self.make()

    def test_a_replay_before_a_plan_has_the_work_done_toward_it_but_not_the_plan(self):
        # The agent worked ten minutes after candidate-0001's Eval and recorded plan-0002 just before the later files.
        path = self.directory / "plans" / "plan-0002" / "plan.json"
        value = read_json(path)
        value["created_at"] = datetime.fromtimestamp(time.time() + LATER - 1, timezone.utc).isoformat()
        os.chmod(path, 0o644)
        path.write_text(json.dumps(value), encoding="utf-8")
        later(path)
        between = self.directory / "work" / "tilings.txt"
        between.write_text("measured toward plan-0002", encoding="utf-8")
        stamp = time.time() + 600
        os.utime(between, (stamp, stamp))
        renders = self.rendered()
        with renders[0], renders[1]:
            made = replay.make(self.store, self.run_id, None, self.out, before="plan-0002")
        self.assertEqual((made["at"], made["before"]), (self.parent, "plan-0002"))
        folder = self.out / "replay-1" / self.run_id
        self.assertTrue((folder / "work" / "tilings.txt").is_file())
        for left in ("work/later.txt", "work/plan-draft.json", "plans/plan-0002", "candidates/candidate-0002"):
            self.assertFalse((folder / left).exists(), left)
        self.assertEqual(read_json(folder / replay.MARKER)["before"], "plan-0002")
        brief = " ".join((folder / "replay" / "agent_request.md").read_text(encoding="utf-8").split())
        self.assertIn(f"Replay of crit run {self.run_id} before plan-0002", brief)
        self.assertIn("just before plan-0002 was recorded", brief)
        # At the candidate itself the work after its Eval is not there.
        with renders[0], renders[1]:
            replay.make(self.store, self.run_id, self.parent, self.root / "at")
        self.assertFalse((self.root / "at" / "replay-1" / self.run_id / "work" / "tilings.txt").exists())
        with self.assertRaisesRegex(ValueError, "at a candidate .* or before a plan"):
            replay.make(self.store, self.run_id, self.parent, self.root / "both", before="plan-0002")
        with self.assertRaisesRegex(ValueError, "starts from no candidate's Eval"):
            replay.make(self.store, self.run_id, None, self.root / "first", before="plan-0001")

    def test_a_replay_records_nothing_and_checks_plans_against_the_status_rendered_again(self):
        # Rendered again, decode is the family further from its limit; the stored status had prefill further.
        self.make(decode=50.0, prefill=3000.0)
        replica = RunStore(self.out / "replay-1", self.store.project)
        with self.assertRaisesRegex(ValueError, "is a replay of .* and records nothing"):
            replica.plan(self.run_id, plan(self.parent))
        with self.assertRaisesRegex(ValueError, "records nothing"):
            replica.candidate(self.run_id, self.directory / "work", self.proposal())
        with self.assertRaisesRegex(ValueError, "records nothing"):
            replay.make(replica, self.run_id, self.parent, self.root / "again")
        checked = replica.plan(self.run_id, plan(self.parent), check=True)
        self.assertEqual((checked["checked"], checked["recorded"]), (True, False))
        self.assertEqual(checked["priorities"]["p1"], ["decode"])
        self.assertEqual(sorted(p.name for p in (self.out / "replay-1" / self.run_id / "plans").iterdir()), ["plan-0001"])
        # The same plan checked in the run itself judges the status stored in its result.
        self.assertEqual(self.store.plan(self.run_id, plan(self.parent), check=True)["priorities"]["p1"], ["prefill"])

    def test_collect_lists_each_replays_checked_plan(self):
        self.make(count=2)
        path = self.out / "replay-1" / self.run_id / "replay" / "plan.json"
        path.write_text(json.dumps(plan(self.parent)), encoding="utf-8")
        found = replay.collect(self.out, self.run_id, RunStore, self.store.project)
        first, second = found["replays"]
        self.assertEqual((first["decision"], first["priorities"]["p1"]), ("build", ["prefill"]))
        self.assertEqual([lever["id"] for lever in first["levers"]], ["graphs", "chunked-prefill"])
        self.assertEqual(second, {"replay": "replay-2", "next": False, "plan": None, "error": "no plan.json"})
        text = replay.render_collected(found)
        self.assertIn("replay-1: build; p1 prefill, p2 decode", text)
        self.assertIn("chunked-prefill (free text): prefill +50%, ttft +20%", text)

    def test_the_cli_makes_replays_and_checks_plans(self):
        renders = self.rendered()
        out = io.StringIO()
        base = ["--project", str(self.store.project), "--root", str(self.store.root), "--id", self.run_id]
        with renders[0], renders[1], contextlib.redirect_stdout(out):
            self.assertEqual(main(base + ["replay", "--at", self.parent, "--out", str(self.out)]), 0)
        self.assertEqual(json.loads(out.getvalue())["replays"][0]["root"], str(self.out / "replay-1"))
        path = self.root / "plan.json"
        path.write_text(json.dumps(plan(self.parent)), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(base + ["plan", "--record", str(path), "--check"]), 0)
        self.assertEqual(json.loads(out.getvalue())["recorded"], False)
        self.assertFalse((self.directory / "plans" / "plan-0003").exists())

    def test_any_harness_runs_on_every_replay_through_agent(self):
        # A stand-in harness: it reads the brief it is given and writes the plan of its replay.
        script = self.root / "harness.py"
        script.write_text("import json, os, pathlib, sys\n"
                          "brief, number = sys.argv[1], sys.argv[2]\n"
                          "assert pathlib.Path(brief).read_text().startswith('# Replay of crit run')\n"
                          "folder = pathlib.Path(os.environ['CRIT_REPLAY_FOLDER'])\n"
                          "(folder / 'replay' / 'plan.json').write_text(os.environ['PLAN'])\n"
                          "print('harness ran replay', number)\n", encoding="utf-8")
        renders = self.rendered()
        out = io.StringIO()
        base = ["--project", str(self.store.project), "--root", str(self.store.root), "--id", self.run_id]
        with renders[0], renders[1], patch.dict(os.environ, {"PLAN": json.dumps(plan(self.parent))}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(base + ["replay", "--at", self.parent, "--out", str(self.out), "--count", "2",
                                          "--agent", f"{sys.executable} {script} {{brief}} {{number}}"]), 0)
        found = json.loads(out.getvalue())
        self.assertEqual([r["agent"]["exit_code"] for r in found["replays"]], [0, 0])
        self.assertEqual([r["decision"] for r in found["collected"]["replays"]], ["build", "build"])
        folder = self.out / "replay-2" / self.run_id / "replay"
        self.assertEqual((folder / "agent.log").read_text(encoding="utf-8").strip(), "harness ran replay 2")
        self.assertEqual(read_json(folder / "agent.json")["exit_code"], 0)
        with self.assertRaisesRegex(ValueError, "placeholders"):
            replay.launch({"replays": [{"brief": "b", "folder": str(folder.parent), "root": "r"}]}, "agent {plan}")


if __name__ == "__main__":
    unittest.main()
