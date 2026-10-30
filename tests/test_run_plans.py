"""Candidate plans: recorded after the parent's Eval and before any code, checked against the facts they freeze."""
import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

from test_run_records import RunFixture
from crit.run import plans
from crit.run.cli import main
from crit.run.records import file_sha, read_json, validate_spec, write_json


def status(decode=100.0, prefill=1000.0, ttft=200.0, limits=(200.0, 4000.0, 100.0), noise=0.02, context=None,
           gaps=(), greedy=120.0, heldout=90.0, followup=50.0):
    """An Eval result's stored status as crit_eval.reports.status.build writes it, with the fields plans read."""
    rows = [("decode", decode, limits[0], decode / limits[0]), ("prefill", prefill, limits[1], prefill / limits[1]),
            ("ttft", ttft, limits[2], limits[2] / ttft)]
    if context:
        (cd, cd_limit), (ct, ct_floor) = context
        rows += [("context_decode", cd, cd_limit, cd / cd_limit), ("context_ttft", ct, ct_floor, ct_floor / ct)]
    return {"kpis": {"decode": {"score": decode, "ceiling": limits[0], "fraction_of_ceiling": decode / limits[0]},
                     "prefill": {"score": prefill, "ceiling": limits[1], "fraction_of_ceiling": prefill / limits[1]},
                     "ttft": {"score_ms": ttft, "floor_ms": limits[2], "fraction_of_floor": limits[2] / ttft},
                     "end_to_end": {"decode_noise": noise, "ttft_noise": noise},
                     "greedy_decode": {"score": greedy}, "heldout": {"decode_score": heldout},
                     "followup": {"ttft_ms": followup},
                     "context": {"decode": {"score": context[0][0], "ceiling_score": context[0][1]},
                                 "ttft": {"score_ms": context[1][0], "floor_ms": context[1][1]}} if context else None},
            "goals": [{"kpi": kpi, "score": score, "limit": limit, "fraction_of_limit": fraction, "noise": noise}
                      for kpi, score, limit, fraction in rows],
            "gaps": [{"kpi": kpi, "name": name, "projected_gain": gain} for kpi, name, gain in gaps]}


def limitless_decode(value):
    """The status of an engine Eval gives no decode limit, such as a plain engine served through crit serve."""
    value["kpis"]["decode"]["ceiling"] = value["kpis"]["decode"]["fraction_of_ceiling"] = None
    value["goals"][0]["limit"] = value["goals"][0]["fraction_of_limit"] = None
    return value


def headroom(tokens_per_round=(2.0, 2.0), decode_tps=100.0, bytes_per_pass=10e9, bandwidth_gbs=1000.0,
             kv_bytes_per_token=0):
    """An Eval attempt's headroom and speed sections: one pass reads bytes_per_pass at bandwidth_gbs, so with no KV the
    pass floor is 10 ms and each case's limit is its tokens per round x 100 tok/s."""
    cases = {f"case-{i}": {"decode_tps": decode_tps, "tokens_per_round": tpr} for i, tpr in enumerate(tokens_per_round)}
    return {"headroom": {"bandwidth_gbs": bandwidth_gbs, "bytes_per_pass": bytes_per_pass,
                         "kv_bytes_per_token": kv_bytes_per_token, "cases": cases},
            "speed": {"cases": {cid: {"prompt_tokens": 1000, "output_tokens": 256} for cid in cases}}}


def limit_causes(*families):
    """Rules 2: the gaps entries that answer why a family marked not needed below NEAR_LIMIT stops short of its limit."""
    return [{"kpi": name, "gap": "limit", "answer": "cause", "detail": "fixture: the rest of the limit is out of reach"}
            for name in families]


def lever(lever_id="graphs", changes=None, group=None):
    return {"id": lever_id, "technique": "free text", "description": "What the lever builds",
            "changes": changes if changes is not None else {"decode": {"change": 0.3, "source": "estimate"}}, "group": group, "confidence": "medium",
            "provider": "lib:fixture"}


def strategy():
    return {"current": {"name": "EAGLE drafter", "ceiling": "306 tok/s (Eval decode ceiling)", "evidence": "status"},
            "alternative": {"name": "MTP heads", "ceiling": "280 tok/s (acceptance at width 4)", "evidence": "recon"},
            "choice": "current", "reason": "Higher ceiling"}


def plan(parent=None, decision="build", decode=True, prefill=True, **updates):
    return {"parent_candidate_id": parent, "decision": decision,
            "needed": {"decode": {"needed": decode, "evidence": "gap above noise"},
                       "prefill": {"needed": prefill, "evidence": "gap above noise"}},
            "gaps": [], "strategy": strategy(),
            "levers": [lever(), lever("chunked-prefill", {"prefill": {"change": 0.5, "source": "estimate"},
                                                          "ttft": {"change": 0.2, "source": "estimate"}})],
            "rejected": [{"name": "smaller vocabulary", "reason": "content-dependent"}],
            "forecast": {"decode": {"low": 110.0, "expected": 130.0, "high": 150.0},
                         "prefill": {"low": 1100.0, "expected": 1500.0, "high": 1800.0},
                         "ttft": {"low": 150.0, "expected": 170.0, "high": 200.0}},
            "gates": "Quality stays inside the budget; the KV cache is planned for kv_tokens", **updates}


class PlanFixture(RunFixture):
    new_run_plans = "required"
    # The plan rules the fixture's runs are created with; PlanRules3Tests set 3, the default of new runs.
    plan_rules = 2

    def setUp(self):
        super().setUp()
        # A candidate-budget run, as runs are created by default.
        self.data = {key: value for key, value in self.data.items() if key != "budget_minutes"} | {"candidate_budget": 3}
        with patch("crit.run.records.NEW_RUN_PLAN_RULES", self.plan_rules):
            self.view = self.store.create(self.data)
        self.run = self.view["run"]
        self.run_id = self.run["run_id"]
        self.directory = self.store.directory(self.run_id)

    def candidate(self, **updates):
        """A candidate from its plan; unless the test gives lever_fates, every lever of the plan was built."""
        stored = self.directory / "plans" / str(updates.get("plan_id")) / "plan.json"
        if "lever_fates" not in updates and stored.is_file():
            stored = read_json(stored)
            updates["lever_fates"] = {lever["id"]: {"fate": "built"} for lever in stored["levers"]}
        return super().candidate(**updates)

    def evaluated(self, candidate_id, value, completed=True, **result):
        """A retained Eval attempt of the candidate, as crit run eval leaves it; `result` adds sections beside the
        status (headroom, speed)."""
        reservation = self.store.reserve(self.run_id, candidate_id, "eval")
        path = self.directory / "candidates" / candidate_id / "eval" / (reservation["attempt_id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": value, **result}), encoding="utf-8")
        self.store.settle(self.run_id, reservation, {"outcome": "completed" if completed else "failed",
                          "ended_at": reservation["reserved_at"], "exit_code": 0 if completed else 1,
                          "cleanup_settled": True, "logs": [], "failure": None if completed else "fixture"})
        return reservation["attempt_id"]

    def first(self):
        """The run's first candidate, from a first plan, with an Eval."""
        recorded = self.store.plan(self.run_id, plan(levers=[], forecast=None))
        candidate = self.candidate(plan_id=recorded["plan_id"])
        self.evaluated(candidate["candidate_id"], status())
        return candidate


class PlanRecordTests(PlanFixture, unittest.TestCase):
    def test_a_new_run_requires_a_plan_for_every_candidate(self):
        self.assertEqual(self.run["spec"]["plans"], "required")
        with self.assertRaisesRegex(ValueError, "requires a plan for every candidate"):
            self.candidate()
        recorded = self.store.plan(self.run_id, plan(levers=[], forecast=None))
        self.assertEqual(recorded["plan_id"], "plan-0001")
        self.assertIsNone(recorded["priorities"])
        self.assertIn("no Eval yet", recorded["priorities_reason"])
        self.assertIsNone(recorded["inputs"]["eval"])
        candidate = self.candidate(plan_id="plan-0001")
        self.assertEqual(candidate["plan_id"], "plan-0001")
        view = self.store.get(self.run_id)
        self.assertEqual(view["errors"], [])
        self.assertEqual(view["state"]["plans"], [{"plan_id": "plan-0001", "segment_id": "segment-0001",
                         "parent_candidate_id": None, "decision": "build", "supersedes": None, "superseded_by": None,
                         "candidate_id": "candidate-0001"}])
        events = [read_json(path) for path in sorted((self.directory / "events").glob("event-*.json"))]
        recorded_event = next(e for e in events if e["event_type"] == "plan_recorded")
        self.assertEqual(recorded_event["reference"]["path"], "plans/plan-0001/plan.json")
        self.assertEqual(recorded_event["reference"]["sha256"], file_sha(self.directory / "plans/plan-0001/plan.json"))
        with self.assertRaisesRegex(ValueError, "already linked to candidate-0001"):
            self.candidate(plan_id="plan-0001")
        # The run has a candidate now: a plan names its parent.
        with self.assertRaisesRegex(ValueError, "null only while the run has no candidate"):
            self.store.plan(self.run_id, plan(levers=[], forecast=None))

    def test_a_plan_freezes_the_parents_newest_eval_and_crit_computes_its_priorities(self):
        parent = self.first()["candidate_id"]
        self.evaluated(parent, status(decode=50.0), completed=False)  # a failed attempt is not frozen
        newest = self.evaluated(parent, status(decode=150.0, prefill=1000.0))
        recorded = self.store.plan(self.run_id, plan(parent, forecast={**plan()["forecast"],
                                   "decode": {"low": 150.0, "expected": 170.0, "high": 190.0}}))
        path = self.directory / "candidates" / parent / "eval" / (newest + ".json")
        self.assertEqual(recorded["inputs"]["eval"], {"path": path.relative_to(self.directory).as_posix(), "sha256": file_sha(path)})
        self.assertEqual(recorded["inputs"]["kpis"]["decode"], {"score": 150.0, "limit": 200.0, "fraction": 0.75})
        self.assertEqual(recorded["inputs"]["noise"]["decode"], 0.02)
        # Prefill (25% of its limit) is further from its limit than decode (75%).
        self.assertEqual(recorded["priorities"], {"p1": ["prefill"], "p2": ["decode"], "p3": ["ttft"], "estimated": [],
                                                  "sources": {"decode": "eval", "prefill": "eval"}})
        self.assertIsNone(recorded["priorities_reason"])
        stored = read_json(self.directory / "plans" / recorded["plan_id"] / "plan.json")
        self.assertEqual(stored, recorded)
        self.assertEqual(stored["record_kind"], "plan")
        with self.assertRaises(FileExistsError):  # crit never changes a stored plan
            write_json(self.directory / "plans" / recorded["plan_id"] / "plan.json", {}, immutable=True)

    def test_the_agent_request_states_the_priority_rule_and_the_plan(self):
        prompt = " ".join((self.directory / "agent_request.md").read_text(encoding="utf-8").split())
        for line in ("time to first token is p2 or p3, never p1 and never the reason for a candidate on its own",
                     "Priorities decide conflicts and order, never what is left out",
                     "When neither decode nor prefill is needed, the run stops",
                     "crit run ... plan --record plan.json", "needs a stop plan from the newest Eval first"):
            self.assertIn(line, prompt)
        self.assertNotIn("none trades against another", prompt)
        with patch("crit.run.records.NEW_RUN_PLANS", None):
            older = self.store.create(self.data)["run"]["run_id"]
        prompt = (self.store.directory(older) / "agent_request.md").read_text(encoding="utf-8")
        self.assertNotIn("crit run ... plan --record", prompt)

    def test_the_first_plans_decision_follows_its_needed_flags(self):
        for value, message in ((plan(decode=False, prefill=False, levers=[], forecast=None), "record a stop plan"),
                               (plan(decision="stop", prefill=False, levers=[], forecast=None),
                                "requires both decode and prefill not needed: decode is marked needed")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.store.plan(self.run_id, value)
        self.assertEqual(self.store.plan(self.run_id, plan(prefill=False, levers=[], forecast=None))["plan_id"], "plan-0001")
        # A run created before rules 2 skips check 3 on its first plan.
        with patch("crit.run.records.NEW_RUN_PLAN_RULES", None):
            older = self.store.create(self.data)["run"]["run_id"]
        self.assertEqual(self.store.plan(older, plan(decode=False, prefill=False, levers=[], forecast=None))["decision"],
                         "build")

    def test_the_cli_records_a_plan(self):
        path = self.root / "plan.json"
        path.write_text(json.dumps(plan(levers=[], forecast=None)), encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--root", str(self.store.root), "--id", self.run_id, "plan", "--record", str(path)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["plan_id"], "plan-0001")

    def test_each_check_refuses_with_what_to_fix(self):
        parent = self.first()["candidate_id"]
        self.evaluated(parent, status(context=((40.0, 100.0), (4000.0, 1000.0)),
                                      gaps=[("decode", "model pass at its bandwidth floor", 0.25),
                                            ("context_decode", "KV cache at fp8", 0.01)]))
        forecast = {**plan()["forecast"], "context_decode": {"low": 40.0, "expected": 50.0, "high": 60.0},
                    "context_ttft": {"low": 3000.0, "expected": 3500.0, "high": 4000.0}}
        good = plan(parent, forecast=forecast)
        cases = [
            # 1. fields, types and text limits; unknown keys
            ({**good, "vibes": 1}, "Unknown fields"),
            ({**good, "decision": "maybe"}, "decision is build or stop"),
            ({**good, "gates": " "}, "gates"),
            ({**good, "levers": [lever(changes={"decode": {"change": "big", "source": "estimate"}})]}, "decode change"),
            ({**good, "levers": [lever(changes={"decode": {"change": 0.3, "source": "a hunch"}})]}, "eval-gap:<gap name> or estimate"),
            ({**good, "levers": [lever(), lever()]}, "appears twice"),
            ({**good, "levers": [lever(changes={"decode": {"change": 0.3, "source": "eval-gap:imaginary"}})]},
             "names no decode gap"),
            # 2. the parent exists
            (plan("candidate-0009", forecast=forecast), "is not a candidate of this run"),
            # 3. the decision follows the needed flags
            ({**good, "decision": "stop"}, "requires both decode and prefill not needed"),
            (plan(parent, decode=False, prefill=False, forecast=forecast), "record a stop plan"),
            # 4. a gap above the noise on a KPI marked not needed is answered
            (plan(parent, decode=False, forecast=forecast), "model pass at its bandwidth floor"),
            # 5. levers, each with a KPI change
            ({**good, "levers": []}, "names its levers"),
            ({**good, "levers": [lever(), lever("empty", {})]}, "names no KPI change"),
            # 6. a lever below the noise is grouped with the lever it pays together with
            ({**good, "levers": good["levers"] + [lever("tiny", {"decode": {"change": 0.01, "source": "estimate"}})]},
             "tiny changes no KPI by more than its noise"),
            ({**good, "levers": good["levers"] + [lever("tiny", {"decode": {"change": 0.01, "source": "estimate"}}, "alone")]},
             "tiny changes no KPI"),
            # 7. a needed p1 KPI has a lever on its family
            ({**good, "levers": [lever()]}, "prefill is p1 but no lever improves it"),
            # 8. no plan targets time to first token alone
            (plan(parent, prefill=False, forecast=forecast, gaps=limit_causes("prefill"),
                  levers=[lever(changes={"ttft": {"change": 0.3, "source": "estimate"}})]), "never the reason"),
            # 9. the forecast covers every measured KPI and improves decode or prefill beyond the noise
            ({**good, "forecast": plan()["forecast"]}, "forecast lacks context_decode, context_ttft"),
            ({**good, "forecast": {**forecast, "decode": {"low": 140.0, "expected": 130.0, "high": 150.0}}},
             "low ≤ expected ≤ high"),
            ({**good, "forecast": {**forecast, "decode": {"low": 0.0, "expected": 130.0, "high": 150.0}}}, "forecast decode low"),
            ({**good, "forecast": None}, "forecasts decode, prefill, ttft, context_decode, context_ttft"),
            ({**good, "forecast": {**forecast, "decode": {"low": 90.0, "expected": 101.0, "high": 110.0},
                                   "prefill": {"low": 900.0, "expected": 1010.0, "high": 1100.0}}},
             "improves neither decode nor prefill beyond its noise"),
            # 10. the strategy names the current one and the best alternative
            ({**good, "strategy": {**strategy(), "alternative": {"name": "", "ceiling": "x", "evidence": "y"}}},
             "strategy.alternative.name"),
            ({**good, "strategy": {k: v for k, v in strategy().items() if k != "alternative"}}, "alternative"),
            # 11. supersedes names an existing plan
            ({**good, "supersedes": "plan-0099"}, "Unknown plan plan-0099"),
        ]
        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.store.plan(self.run_id, copy.deepcopy(value))
        self.assertEqual(sorted(path.name for path in (self.directory / "plans").iterdir()), ["plan-0001"])
        # The same plan with the gap answered and the small lever grouped passes.
        answered = plan(parent, decode=False, forecast={**forecast, "prefill": {"low": 1100.0, "expected": 1500.0, "high": 2000.0}},
                        gaps=[{"kpi": "decode", "gap": "model pass at its bandwidth floor", "answer": "rejected",
                               "detail": "measured: the graph was slower"}] + limit_causes("decode"),
                        levers=[lever("chunked-prefill", {"prefill": {"change": 0.5, "source": "estimate"}}, "prefill"),
                                lever("tiny", {"prefill": {"change": 0.01, "source": "estimate"}}, "prefill")])
        recorded = self.store.plan(self.run_id, answered)
        self.assertEqual(recorded["priorities"], {"p1": ["prefill"], "p2": ["ttft"], "p3": [], "estimated": [],
                                                  "sources": {}})

    def test_a_parent_without_eval_is_refused_once_the_run_has_one(self):
        first = self.first()["candidate_id"]
        recorded = self.store.plan(self.run_id, plan(first))
        second = self.candidate(parent_candidate_id=first, plan_id=recorded["plan_id"])["candidate_id"]
        with self.assertRaisesRegex(ValueError, f"{second} has no completed Eval attempt"):
            self.store.plan(self.run_id, plan(second))

    def test_revisions_supersede_the_newest_plan_before_its_candidate(self):
        parent = self.first()["candidate_id"]
        old = self.store.plan(self.run_id, plan(parent))["plan_id"]
        new = self.store.plan(self.run_id, plan(parent, supersedes=old))["plan_id"]
        with self.assertRaisesRegex(ValueError, f"already superseded by {new}"):
            self.store.plan(self.run_id, plan(parent, supersedes=old))
        with self.assertRaisesRegex(ValueError, f"superseded by {new}"):
            self.candidate(parent_candidate_id=parent, plan_id=old)
        self.candidate(parent_candidate_id=parent, plan_id=new)
        with self.assertRaisesRegex(ValueError, "change course with a new candidate"):
            self.store.plan(self.run_id, plan(parent, supersedes=new))
        state = {row["plan_id"]: row for row in self.store.get(self.run_id)["state"]["plans"]}
        self.assertEqual(state[old]["superseded_by"], new)  # the old plan stays in state
        self.assertTrue((self.directory / "plans" / old / "plan.json").is_file())
        self.assertEqual(state[new]["candidate_id"], "candidate-0002")

    def test_a_candidate_links_only_an_unlinked_build_plan_of_its_parent(self):
        parent = self.first()["candidate_id"]
        with self.assertRaisesRegex(ValueError, "Unknown plan plan-0042"):
            self.candidate(parent_candidate_id=parent, plan_id="plan-0042")
        recorded = self.store.plan(self.run_id, plan(parent))
        with self.assertRaisesRegex(ValueError, f"starts from {parent}, the proposal from None"):
            self.candidate(plan_id=recorded["plan_id"])
        stop = self.store.plan(self.run_id, plan(parent, decision="stop", decode=False, prefill=False, levers=[],
                                                 forecast=None, gaps=limit_causes("decode", "prefill")))
        self.assertIsNone(stop["priorities"])
        with self.assertRaisesRegex(ValueError, "is a stop plan"):
            self.candidate(parent_candidate_id=parent, plan_id=stop["plan_id"])
        self.assertEqual(list((self.directory / "candidates").glob("candidate-0002")), [])

    def test_an_older_run_keeps_plan_id_optional_and_checks_it_when_given(self):
        self.assertEqual(self.run["spec"]["plans"], "required")
        with patch("crit.run.records.NEW_RUN_PLANS", None):
            older = self.store.create({key: value for key, value in self.data.items()})
        self.assertNotIn("plans", older["run"]["spec"])
        self.run_id, self.directory = older["run"]["run_id"], self.store.directory(older["run"]["run_id"])
        self.assertNotIn("plan_id", self.candidate())
        with self.assertRaisesRegex(ValueError, "Unknown plan"):
            self.candidate(plan_id="plan-0001")

    def test_the_outcome_compares_the_forecast_with_the_measured_eval(self):
        parent = self.first()["candidate_id"]
        recorded = self.store.plan(self.run_id, plan(parent))
        child = self.candidate(parent_candidate_id=parent, plan_id=recorded["plan_id"])["candidate_id"]
        # Decode lands inside its range, prefill above it; greedy decode fell 20% (noise 2%), held-out decode rose,
        # the follow-up first token is 1% slower: inside the noise.
        attempt = self.evaluated(child, status(decode=125.0, prefill=2000.0, ttft=170.0, greedy=96.0, heldout=95.0,
                                               followup=50.5))
        value = self.store.plan_outcome(self.run_id, child, attempt)
        path = self.directory / "plans" / recorded["plan_id"] / "outcomes" / (attempt + ".json")
        self.assertEqual(read_json(path), value)
        self.assertEqual(value["eval"]["path"], f"candidates/{child}/eval/{attempt}.json")
        self.assertEqual(value["kpis"]["decode"]["measured"], 125.0)
        self.assertTrue(value["kpis"]["decode"]["inside"])
        self.assertAlmostEqual(value["kpis"]["decode"]["error"], 125 / 130 - 1)
        self.assertFalse(value["kpis"]["prefill"]["inside"])
        self.assertEqual([flag["kpi"] for flag in value["flags"]], ["greedy_decode"])
        self.assertAlmostEqual(value["flags"][0]["change"], -0.2)
        self.assertNotIn("verdict", value)
        # The first candidate's plan forecast nothing: its outcome holds no KPI and no parent to compare with.
        self.assertEqual(self.store.plan_outcome(self.run_id, parent, attempt_of(self, parent))["kpis"], {})

    def test_an_estimated_limit_fills_a_limit_eval_does_not_give(self):
        parent = self.first()["candidate_id"]
        self.evaluated(parent, limitless_decode(status(decode=100.0)))  # a plain served engine: no decode limit
        estimate = {"limit": 125.0, "derivation": "3.5 accepted tokens per round over a 28 ms round floor"}
        for estimates, message in (
                ({}, "gives decode no limit, and the priorities compare decode with prefill: state estimated_limits.decode"),
                ({"decode": {**estimate, "limit": 90.0}}, "90 is below the measured score 100"),
                ({"decode": {**estimate, "derivation": " "}}, "estimated_limits.decode.derivation"),
                ({"decode": estimate, "prefill": {"limit": 5000.0, "derivation": "x"}}, "Eval's limit stands"),
                ({"decode": estimate, "context_decode": {"limit": 90.0, "derivation": "x"}}, "did not measure"),
                ({"decode": estimate, "ttft": {"limit": 250.0, "derivation": "x"}}, "Eval's limit stands")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.store.plan(self.run_id, plan(parent, estimated_limits=estimates))
        recorded = self.store.plan(self.run_id, plan(parent, estimated_limits={"decode": estimate}))
        # Decode at 80% of its estimated ceiling is closer to its limit than prefill at 25%.
        self.assertEqual(recorded["priorities"], {"p1": ["prefill"], "p2": ["decode"], "p3": ["ttft"], "estimated": ["decode"],
                                                  "sources": {"decode": "estimate", "prefill": "eval"}})
        self.assertEqual(recorded["inputs"]["estimated"], ["decode"])
        self.assertEqual(recorded["inputs"]["kpis"]["decode"], {"score": 100.0, "limit": None, "fraction": 0.8})
        self.assertEqual(recorded["estimated_limits"], {"decode": estimate})
        # With only decode needed nothing is compared, so no estimate is needed.
        only = self.store.plan(self.run_id, plan(parent, prefill=False, levers=[lever()], gaps=limit_causes("prefill")))
        self.assertEqual(only["priorities"], {"p1": ["decode"], "p2": ["ttft"], "p3": [], "estimated": [], "sources": {}})
        # The outcome sets the estimate next to the measurement and, once Eval gives one, its real limit.
        child = self.candidate(parent_candidate_id=parent, plan_id=recorded["plan_id"])["candidate_id"]
        value = self.store.plan_outcome(self.run_id, child, self.evaluated(child, status(decode=130.0)))
        self.assertEqual(value["estimates"]["decode"]["estimate"], 125.0)
        self.assertEqual(value["estimates"]["decode"]["measured"], 130.0)
        self.assertTrue(value["estimates"]["decode"]["exceeded"])  # the estimate was too low
        self.assertEqual(value["estimates"]["decode"]["eval_limit"], 200.0)
        self.assertAlmostEqual(value["estimates"]["decode"]["error"], 125 / 200 - 1)

    def test_crit_computes_a_decode_limit_eval_does_not_give(self):
        parent = self.first()["candidate_id"]
        # Eval states no decode limit, but the attempt holds the facts: 2 tokens per round over a 10 ms pass floor.
        self.evaluated(parent, limitless_decode(status(decode=100.0)), **headroom())
        recorded = self.store.plan(self.run_id, plan(parent))
        computed = recorded["inputs"]["computed_limits"]["decode"]
        self.assertAlmostEqual(computed["limit"], 200.0)
        self.assertTrue(computed["used"])
        self.assertEqual(computed["unit"], "tok/s")
        self.assertEqual(computed["inputs"]["bytes_per_pass"], "headroom.bytes_per_pass")
        self.assertEqual(computed["inputs"]["tokens_per_round"], "headroom.cases.<case>.tokens_per_round")
        self.assertIn("tokens_per_round x 1000 / pass_floor_ms", computed["formula"])
        self.assertAlmostEqual(recorded["inputs"]["kpis"]["decode"]["fraction"], 0.5)
        self.assertEqual(recorded["inputs"]["fraction_sources"], {"decode": "computed", "prefill": "eval", "ttft": "eval"})
        # Decode at 50% of crit's limit is closer to it than prefill at 25%; no estimate was needed.
        self.assertEqual(recorded["priorities"], {"p1": ["prefill"], "p2": ["decode"], "p3": ["ttft"], "estimated": [],
                                                  "sources": {"decode": "computed", "prefill": "eval"}})
        # An agent's estimate for a limit crit computed is kept, with its difference, and not used.
        estimate = {"limit": 300.0, "derivation": "full acceptance at width 4 over the round floor"}
        recorded = self.store.plan(self.run_id, plan(parent, estimated_limits={"decode": estimate}))
        self.assertEqual(recorded["inputs"]["estimates_not_used"]["decode"]["estimate"], 300.0)
        self.assertAlmostEqual(recorded["inputs"]["estimates_not_used"]["decode"]["difference"], 0.5)
        self.assertAlmostEqual(recorded["inputs"]["kpis"]["decode"]["fraction"], 0.5)
        self.assertEqual(recorded["priorities"]["sources"]["decode"], "computed")

    def test_a_computed_limit_below_the_score_falls_back_to_the_estimate(self):
        parent = self.first()["candidate_id"]
        # One token per round over the 10 ms floor gives 100 tok/s, below the measured 150: not used.
        self.evaluated(parent, limitless_decode(status(decode=150.0)), **headroom(tokens_per_round=(1.0,), decode_tps=150.0))
        with self.assertRaisesRegex(ValueError, "state estimated_limits.decode"):
            self.store.plan(self.run_id, plan(parent, forecast={**plan()["forecast"],
                            "decode": {"low": 150.0, "expected": 170.0, "high": 190.0}}))
        recorded = self.store.plan(self.run_id, plan(parent, forecast={**plan()["forecast"],
                                   "decode": {"low": 150.0, "expected": 170.0, "high": 190.0}},
                                   estimated_limits={"decode": {"limit": 300.0, "derivation": "fixture"}}))
        computed = recorded["inputs"]["computed_limits"]["decode"]
        self.assertFalse(computed["used"])
        self.assertIn("below the measured score", computed["note"])
        self.assertEqual(recorded["priorities"]["sources"]["decode"], "estimate")

    def test_a_run_created_before_computed_limits_keeps_the_agents_estimate(self):
        with patch("crit.run.records.NEW_RUN_PLAN_LIMITS", None):
            self.data = {key: value for key, value in self.data.items()}
            self.view = self.store.create(self.data)
        self.run, self.run_id = self.view["run"], self.view["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)
        self.assertNotIn("plan_limits", self.run["spec"])
        parent = self.first()["candidate_id"]
        self.evaluated(parent, limitless_decode(status(decode=100.0)), **headroom())
        with self.assertRaisesRegex(ValueError, "state estimated_limits.decode"):
            self.store.plan(self.run_id, plan(parent))
        recorded = self.store.plan(self.run_id, plan(parent, estimated_limits={"decode": {"limit": 125.0, "derivation": "x"}}))
        self.assertNotIn("computed_limits", recorded["inputs"])
        self.assertNotIn("sources", recorded["priorities"])

    def test_computed_limits_cover_the_long_context_curve(self):
        result = headroom()
        result["headroom"]["kv_bytes_per_token"] = 1000
        result["headroom"]["context"] = {"points": {"65536": {"decode_tps": 50.0, "tokens_per_round": 1.0,
                                                              "prompt_tokens": 65000, "output_tokens": 256}}}
        found = plans.computed_limits(result, self.model)
        # 10e9 + 1000 x 65,128 bytes at 1,000 GB/s: 10.065 ms per pass.
        self.assertAlmostEqual(found["context_decode"]["limit"], 1000 / ((10e9 + 1000 * 65128) / 1e12 * 1000))
        self.assertEqual(found["context_decode"]["inputs"]["kv_bytes_per_token"], "headroom.kv_bytes_per_token")
        self.assertEqual(plans.computed_limits({"headroom": {"cases": {}}}, self.model), {})

    def test_a_requirement_lever_serves_a_rule_without_a_scored_change(self):
        parent = self.first()["candidate_id"]
        sampled = {"id": "sampled-speculation", "technique": "token-sampling", "description": "Rejection-sampled drafts",
                   "changes": {}, "group": None, "confidence": "high", "kind": "requirement",
                   "serves": "AGENTS.md: sampled requests speculate"}
        # Alone and with no change, as a speed lever it is refused (check 5); as a requirement lever it passes.
        with self.assertRaisesRegex(ValueError, "sampled-speculation names no KPI change"):
            self.store.plan(self.run_id, plan(parent, levers=plan()["levers"] + [
                {k: v for k, v in sampled.items() if k not in ("kind", "serves")}]))
        with self.assertRaisesRegex(ValueError, "changes no KPI by more than its noise"):
            self.store.plan(self.run_id, plan(parent, levers=plan()["levers"] + [
                {**{k: v for k, v in sampled.items() if k not in ("kind", "serves")},
                 "changes": {"decode": {"change": 0.0, "source": "estimate"}}}]))
        recorded = self.store.plan(self.run_id, plan(parent, levers=plan()["levers"] + [sampled]))
        self.assertEqual(recorded["levers"][-1]["kind"], "requirement")
        for broken, message in (({**sampled, "serves": " "}, "serves"),
                                ({**sampled, "kind": "maybe"}, "kind is speed or requirement"),
                                ({**lever("extra"), "serves": "x"}, "only a requirement lever")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.store.plan(self.run_id, plan(parent, levers=plan()["levers"] + [broken]))
        # Requirement levers never count as a KPI lever: alone they make no candidate, and they reach no p1 (check 7).
        big = {**sampled, "changes": {"prefill": {"change": 0.5, "source": "estimate"}}}
        with self.assertRaisesRegex(ValueError, "requirement levers alone"):
            self.store.plan(self.run_id, plan(parent, levers=[big]))
        with self.assertRaisesRegex(ValueError, "prefill is p1 but no lever improves it"):
            self.store.plan(self.run_id, plan(parent, levers=[lever(), big]))
        child = self.candidate(parent_candidate_id=parent, plan_id=recorded["plan_id"])["candidate_id"]
        value = self.store.plan_outcome(self.run_id, child, self.evaluated(child, status()))
        self.assertEqual(value["requirements"], [{"id": "sampled-speculation", "serves": "AGENTS.md: sampled requests speculate"}])

    def test_every_lever_of_the_plan_gets_its_fate(self):
        parent = self.first()["candidate_id"]
        self.assertEqual(self.run["spec"]["lever_fates"], "required")
        plan_id = self.store.plan(self.run_id, plan(parent))["plan_id"]
        for fates, message in (
                ({"graphs": {"fate": "built"}}, "gives no fate to chunked-prefill"),
                ({"graphs": {"fate": "built"}, "chunked-prefill": {"fate": "built"}, "vibes": {"fate": "built"}},
                 "names vibes, which plan-0002 does not plan"),
                ({"graphs": {"fate": "built"}, "chunked-prefill": {"fate": "pruned"}}, "chunked-prefill detail"),
                ({"graphs": {"fate": "built"}, "chunked-prefill": {"fate": "maybe"}}, "built, pruned or deferred"),
                ({"graphs": {"fate": "built"}, "chunked-prefill": {"fate": "deferred", "detail": "x", "when": "later"}},
                 "Unknown fields")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.candidate(parent_candidate_id=parent, plan_id=plan_id, lever_fates=fates)
        fates = {"graphs": {"fate": "built"},
                 "chunked-prefill": {"fate": "pruned", "detail": "measured 3% slower than the fused path at 8K tokens"}}
        child = self.candidate(parent_candidate_id=parent, plan_id=plan_id, lever_fates=fates)
        self.assertEqual(child["lever_fates"], fates)
        self.assertEqual(self.store.get(self.run_id)["errors"], [])
        value = self.store.plan_outcome(self.run_id, child["candidate_id"], self.evaluated(child["candidate_id"], status()))
        self.assertEqual([(row["id"], row["fate"]) for row in value["levers"]],
                         [("graphs", {"fate": "built"}), ("chunked-prefill", fates["chunked-prefill"])])
        self.assertEqual(value["levers"][0]["changes"], {"decode": {"change": 0.3, "source": "estimate"}})

    def test_an_older_run_checks_fates_only_when_given(self):
        with patch("crit.run.records.NEW_RUN_LEVER_FATES", None):
            self.view = self.store.create(self.data)
        self.run, self.run_id = self.view["run"], self.view["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)
        self.assertNotIn("lever_fates", self.run["spec"])
        parent = self.first()["candidate_id"]
        plan_id = self.store.plan(self.run_id, plan(parent))["plan_id"]
        with self.assertRaisesRegex(ValueError, "names vibes"):
            self.candidate(parent_candidate_id=parent, plan_id=plan_id, lever_fates={"vibes": {"fate": "built"}})
        child = self.candidate(parent_candidate_id=parent, plan_id=plan_id, lever_fates={"graphs": {"fate": "built"}})
        self.assertEqual(child["lever_fates"], {"graphs": {"fate": "built"}})
        # Without fates the candidate is created as before.
        plan_id = self.store.plan(self.run_id, plan(parent))["plan_id"]
        self.assertNotIn("lever_fates", super(PlanFixture, self).candidate(parent_candidate_id=parent, plan_id=plan_id))

    def test_the_outcome_flags_a_forecast_whose_gain_falls_on_p2(self):
        parent = self.first()["candidate_id"]
        # Prefill is p1, but the forecast puts +100% on decode (p2) and +5% on prefill.
        forecast = {**plan()["forecast"], "decode": {"low": 150.0, "expected": 200.0, "high": 250.0},
                    "prefill": {"low": 1000.0, "expected": 1050.0, "high": 1100.0}}
        lopsided = self.store.plan(self.run_id, plan(parent, forecast=forecast))
        self.assertEqual(lopsided["priorities"]["p1"], ["prefill"])
        child = self.candidate(parent_candidate_id=parent, plan_id=lopsided["plan_id"])["candidate_id"]
        flag = self.store.plan_outcome(self.run_id, child, self.evaluated(child, status()))["forecast_on_p2"]
        self.assertEqual((flag["p1"], flag["p2"]), (["prefill"], ["decode"]))
        self.assertAlmostEqual(flag["p1_gain"], 0.05)
        self.assertAlmostEqual(flag["p2_gain"], 1.0)
        # The default forecast puts +50% on prefill (p1) and +30% on decode: no flag.
        balanced = self.store.plan(self.run_id, plan(child))
        self.assertIsNone(plans.priority_flag(balanced))

    def baseline(self):
        """Point 2 of run 3141e655 in miniature: a plain engine at 62.2 tok/s, one token per pass, 19.26 GB per pass at
        1437.7 GB/s (crit computes ~74.6 tok/s, 83% of it), prefill at 33.5% of its ceiling."""
        parent = self.first()["candidate_id"]
        result = headroom(tokens_per_round=(1.0, 1.0), decode_tps=62.2, bytes_per_pass=19.26e9, bandwidth_gbs=1437.7)
        result["headroom"]["draft_bytes_per_step"] = 0.9e9
        self.evaluated(parent, limitless_decode(status(decode=62.2, prefill=1340.0)), **result)
        return parent

    def mtp(self, tokens_per_round=3.5, draft_steps=3, drafter="new"):
        """The alternative MTP strategy: a new drafter for the plain baseline (no drafter field: rules 1)."""
        return {**strategy(), "alternative": {**strategy()["alternative"], "limits": {"decode": {
            "tokens_per_round": tokens_per_round, "draft_steps": draft_steps, **({"drafter": drafter} if drafter else {}),
            "derivation": "checkpoint MTP chained 3 deep at the per-step acceptance its recon reports"}}}}

    def drafted(self, accepted=(1,) * 8 + (0,) * 2, width=7):
        """A speculative parent Eval gives no decode limit (crit computes 2 tokens per round over the 10 ms pass floor:
        200 tok/s), with its drafter measured at draft_width and these per-round accepted counts in each case:
        the first draft accepted in 8 rounds of 10, a = 0.8."""
        parent = self.first()["candidate_id"]
        result = headroom()
        result["headroom"] |= {"draft_width": width, "projection": {"draft_floor_ms": 1.0}}
        result["speed"]["cases"] = {cid: {**case, "status": "measured", "samples": [{"accepted": list(accepted)}]}
                                    for cid, case in result["speed"]["cases"].items()}
        self.evaluated(parent, limitless_decode(status(decode=100.0)), **result)
        return parent

    def test_the_measured_drafter_stays_within_evals_acceptance_bound(self):
        parent = self.drafted()
        # At 7 draft steps and a = 0.8: (1 - 0.8^8) / 0.2 = 4.161 tokens per round at most.
        bound = (1 - 0.8 ** 8) / 0.2
        for strategy_, message in (
                (self.mtp(5.1, 7, "measured"), f"tokens_per_round 5.1 is above {bound:.3f}, the tokens per round the "
                                               "measured drafter reaches at 7 draft steps"),
                (self.mtp(4.0, 9, "measured"), "draft_steps 9 is wider than Eval projects a drafter \\(8 draft steps"),
                (self.mtp(4.0, 9, "new"), "draft_steps 9 is wider than Eval projects a drafter \\(8 draft steps"),
                (self.mtp(4.0, 7, None), "drafter names the drafter the ceiling assumes: measured")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.store.plan(self.run_id, plan(parent, strategy=strategy_))
        entry = self.store.plan(self.run_id, plan(parent, strategy=self.mtp(4.0, 7, "measured")))["inputs"][
            "alternative_limits"]["decode"]
        self.assertEqual(entry["drafter"], "measured")
        self.assertAlmostEqual(entry["acceptance_bound"]["tokens_per_round"], bound)
        self.assertEqual(entry["acceptance_bound"]["rates"], {"case-0": 0.8, "case-1": 0.8})
        self.assertEqual((entry["acceptance_bound"]["draft_steps"], entry["acceptance_bound"]["widest"]), (7, 8))
        # The claim's implied rate: 4 tokens a round at 7 steps is a = 0.778.
        self.assertAlmostEqual(entry["implied_acceptance"], plans.implied_rate(4.0, 7))
        self.assertAlmostEqual((1 - entry["implied_acceptance"] ** 8) / (1 - entry["implied_acceptance"]), 4.0)

    def test_a_new_drafter_accepts_like_the_measured_one_unless_the_run_measured_it(self):
        parent = self.drafted()
        with self.assertRaisesRegex(ValueError, "tokens_per_round 6.5 is above 4.161.*or cite another candidate's Eval "
                                                "attempt that measured it"):
            self.store.plan(self.run_id, plan(parent, strategy=self.mtp(6.5, 7, "new")))
        # Another candidate of the run built the new drafter, and its Eval measured the first draft accepted in 19
        # rounds of 20 (a = 0.95): at 7 steps (1 - 0.95^8) / 0.05 = 6.73 tokens a round at most.
        prototype = self.store.plan(self.run_id, plan(parent))["plan_id"]
        other = self.candidate(parent_candidate_id=parent, plan_id=prototype)["candidate_id"]
        result = headroom()
        result["headroom"] |= {"draft_width": 7, "projection": {"draft_floor_ms": 1.0}}
        result["speed"]["cases"] = {cid: {**case, "status": "measured", "samples": [{"accepted": [1] * 19 + [0]}]}
                                    for cid, case in result["speed"]["cases"].items()}
        attempt = self.evaluated(other, status(), **result)
        cited = self.mtp(6.5, 7, "new")
        evidence = f"candidates/{other}/eval/{attempt}.json"
        cited["alternative"]["limits"]["decode"]["acceptance_evidence"] = evidence
        entry = self.store.plan(self.run_id, plan(parent, strategy=cited))["inputs"]["alternative_limits"]["decode"]
        self.assertEqual(entry["acceptance_evidence"]["path"], evidence)
        self.assertEqual(entry["acceptance_evidence"]["sha256"], file_sha(self.directory / evidence))
        self.assertAlmostEqual(entry["acceptance_bound"]["tokens_per_round"], (1 - 0.95 ** 8) / 0.05)
        cited["alternative"]["limits"]["decode"]["tokens_per_round"] = 6.8
        with self.assertRaisesRegex(ValueError, "tokens_per_round 6.8 is above 6.732"):
            self.store.plan(self.run_id, plan(parent, strategy=cited))
        cited["alternative"]["limits"]["decode"]["tokens_per_round"] = 6.5
        parents = sorted((self.directory / "candidates" / parent / "eval").glob("attempt-*.json"))[-1].name
        (self.directory / "work").mkdir(exist_ok=True)
        (self.directory / "work" / "acceptance.json").write_text("{}", encoding="utf-8")
        for path, message in (("run.json", "names another candidate's Eval attempt"),
                              ("work/acceptance.json", "names another candidate's Eval attempt"),
                              (f"candidates/{parent}/eval/{parents}", "names another candidate's Eval attempt"),
                              (f"candidates/{other}/eval/attempt-0099.json", "is not a completed Eval attempt")):
            cited["alternative"]["limits"]["decode"]["acceptance_evidence"] = path
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, message):
                    self.store.plan(self.run_id, plan(parent, strategy=cited))
        measured = self.mtp(4.0, 7, "measured")
        measured["alternative"]["limits"]["decode"]["acceptance_evidence"] = evidence
        with self.assertRaisesRegex(ValueError, "acceptance_evidence is for a new drafter"):
            self.store.plan(self.run_id, plan(parent, strategy=measured))

    def test_the_long_context_bound_comes_from_the_long_context_points(self):
        parent = self.first()["candidate_id"]
        result = headroom()
        result["headroom"] |= {"draft_width": 4, "projection": {"draft_floor_ms": 1.0}, "context": {"points": {
            "65536": {"decode_tps": 30.0, "tokens_per_round": 2.5, "prompt_tokens": 65000, "output_tokens": 256}}}}
        self.evaluated(parent, status(context=((40.0, 100.0), (4000.0, 1000.0))), **result)
        forecast = {**plan()["forecast"], "context_decode": {"low": 40.0, "expected": 50.0, "high": 60.0},
                    "context_ttft": {"low": 3000.0, "expected": 3500.0, "high": 4000.0}}

        def claim(tokens_per_round, draft_steps, drafter="measured"):
            return {**strategy(), "alternative": {**strategy()["alternative"], "limits": {"context_decode": {
                "tokens_per_round": tokens_per_round, "draft_steps": draft_steps, "drafter": drafter,
                "derivation": "the drafter at a wider draft"}}}}
        # 2.5 tokens a round at width 4 is a rate of 0.5605; at 8 steps it reaches 2.27 / 0.4395 = 2.27... tokens.
        rate = plans.implied_rate(2.5, 4)
        bound = (1 - rate ** 9) / (1 - rate)
        with self.assertRaisesRegex(ValueError, f"above {bound:.3f}"):
            self.store.plan(self.run_id, plan(parent, forecast=forecast, strategy=claim(bound + 0.01, 8)))
        entry = self.store.plan(self.run_id, plan(parent, forecast=forecast, strategy=claim(bound, 8)))["inputs"][
            "alternative_limits"]["context_decode"]
        self.assertAlmostEqual(entry["acceptance_bound"]["rates"]["65536"], rate)
        self.assertIn("long-context points", entry["acceptance_bound"]["source"])
        # A width the engine did not report: the points' measured tokens per round is the cap.
        result["headroom"]["draft_width"] = None
        self.evaluated(parent, status(context=((40.0, 100.0), (4000.0, 1000.0))), **result)
        with self.assertRaisesRegex(ValueError, "above 2.500"):
            self.store.plan(self.run_id, plan(parent, forecast=forecast, strategy=claim(2.6, 4)))

    def test_without_a_measured_drafter_the_physical_bound_applies(self):
        parent = self.baseline()  # a plain engine: no draft width, no accepted counts
        entry = self.store.plan(self.run_id, plan(parent, strategy=self.mtp(3.5, 3, "measured")))["inputs"][
            "alternative_limits"]["decode"]
        self.assertIsNone(entry["acceptance_bound"])
        self.assertIn("measured no drafter acceptance", entry["note"])
        with self.assertRaisesRegex(ValueError, "above draft_steps \\+ 1 = 4"):
            self.store.plan(self.run_id, plan(parent, strategy=self.mtp(4.5, 3, "measured")))
        with self.assertRaisesRegex(ValueError, "draft_steps 9 is wider than Eval projects a drafter \\(8"):
            self.store.plan(self.run_id, plan(parent, strategy=self.mtp(9.0, 9, "new")))
        # A run created before rules 2 has no drafter field.
        with patch("crit.run.records.NEW_RUN_PLAN_RULES", None):
            self.view = self.store.create(self.data)
        self.run, self.run_id = self.view["run"], self.view["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)
        parent = self.baseline()
        with self.assertRaisesRegex(ValueError, "drafter is a field of plan rules 2"):
            self.store.plan(self.run_id, plan(parent, strategy=self.mtp(3.5, 3, "measured")))

    def test_rules_2_bound_every_alternative_by_its_tokens_per_round(self):
        parent = self.first()["candidate_id"]
        self.evaluated(parent, limitless_decode(status(decode=62.2, prefill=1340.0)))  # no round floor facts
        for drafter in ("measured", "new"):
            given = {**strategy(), "alternative": {**strategy()["alternative"], "limits": {"decode": {
                "limit": 215.0, "derivation": "x", "drafter": drafter}}}}
            with self.subTest(drafter=drafter):
                with self.assertRaisesRegex(ValueError, "gives a bare limit: in plan rules 2"):
                    self.store.plan(self.run_id, plan(parent, strategy=given,
                                                      estimated_limits={"decode": {"limit": 75.0, "derivation": "x"}}))
        # tokens_per_round without the floor's facts: the refusal names the way left, not the bare limit refused above
        with self.assertRaisesRegex(ValueError, "leave decode out of strategy.alternative.limits"):
            self.store.plan(self.run_id, plan(parent, strategy=self.mtp(3.5, 3, "new"),
                                              estimated_limits={"decode": {"limit": 75.0, "derivation": "x"}}))

    def test_rules_2_price_a_companion_drafter_from_recon(self):
        # A plain parent of a checkpoint without draft layers: Eval measured no draft floor, so a separate drafter's
        # round floor comes from the drafter recon recorded (recon --drafter), one pass for a whole block of drafts.
        parent = self.first()["candidate_id"]
        result = headroom(tokens_per_round=(1.0, 1.0), decode_tps=62.2, bytes_per_pass=19.26e9, bandwidth_gbs=1437.7)
        self.evaluated(parent, limitless_decode(status(decode=62.2, prefill=1340.0)), **result)
        with self.assertRaisesRegex(ValueError, "record the drafter with recon"):
            self.store.plan(self.run_id, plan(parent, strategy=self.mtp(2.0, 7, "new")))
        recon = self.directory / "work" / "recon.json"
        recon.parent.mkdir(parents=True, exist_ok=True)
        recon.write_text(json.dumps({"drafters": [{"path": "/models/dflash2", "kind": "block", "drafts_per_pass": 7,
                                                   "pass_bytes": 4.35e9}]}), encoding="utf-8")
        entry = self.store.plan(self.run_id, plan(parent, strategy=self.mtp(2.0, 7, "new")))["inputs"]["alternative_limits"]["decode"]
        # 2 tokens over 19.26 GB + one 4.35 GB drafter pass at 1,437.7 GB/s: ~121.8 tok/s
        self.assertAlmostEqual(entry["limit"], 2.0 * 1000 / ((19.26e9 + 4.35e9) / 1437.7e6), places=3)
        self.assertEqual(entry["computed_by"], "crit")
        self.assertIn("recon companion drafter /models/dflash2", entry["inputs"]["draft_floor_ms"])
        with self.assertRaisesRegex(ValueError, "companion names no drafter"):
            given = self.mtp(2.0, 7, "new")
            given["alternative"]["limits"]["decode"]["companion"] = "/models/other"
            self.store.plan(self.run_id, plan(parent, strategy=given))

    def test_the_cleanroom_claims_meet_their_measured_acceptance(self):
        # Per case acceptance rates of cleanroom candidates 0027-0029 (Eval's acceptance-bound rates, 4 decimals): at
        # 7 draft steps they reach 5.30, 5.55 and 5.43 tokens a round, below the 7.5, 6.9 and 5.6 that flipped p1
        # to decode there when a claim was bounded by draft_steps + 1 only.
        for rates, bound, flipped in (
                ((0.72, 0.8095, 0.898, 0.9756, 0.8413, 1.0), 5.304, 7.52),
                ((0.7113, 0.9016, 0.8776, 0.9268, 0.9155, 1.0), 5.552, 6.93),
                ((0.7158, 0.873, 0.8776, 0.9524, 0.8714, 1.0), 5.426, 5.62)):
            reached = plans.acceptance_bound({"rates": dict(enumerate(rates))}, 7)
            self.assertAlmostEqual(reached, bound, places=2)
            self.assertLess(reached, flipped)

    def test_the_best_strategys_ceiling_ranks_decode(self):
        parent = self.baseline()
        # Without the alternative's ceiling a plain baseline looks 83% done on decode: prefill is p1.
        current = self.store.plan(self.run_id, plan(parent))
        self.assertEqual(current["priorities"]["p1"], ["prefill"])
        self.assertIsNone(current["inputs"]["alternative_limits"] or None)
        # MTP: 3.5 tokens per round over 13.4 ms + 3 draft steps at the engine's 0.9 GB per step: ~229 tok/s, 27% of
        # it: decode is p1. The plain baseline's computed limit has no draft width: the labelled upper bound.
        recorded = self.store.plan(self.run_id, plan(parent, strategy=self.mtp()))
        entry = recorded["inputs"]["alternative_limits"]["decode"]
        self.assertAlmostEqual(entry["limit"], 3.5 * 1000 / (19.26e9 / 1437.7e6 + 3 * 0.9e9 / 1437.7e6), places=3)
        self.assertAlmostEqual(entry["current_limit"], 1437.7e6 / 19.26e9 * 1000, places=3)
        self.assertEqual((entry["computed_by"], entry["current_source"], entry["won"]), ("crit", "computed", True))
        self.assertEqual(entry["bound"], "Eval's round floor, draft steps at Eval's draft floor")
        self.assertEqual(entry["inputs"]["draft_floor_ms"], "headroom.draft_bytes_per_step")
        self.assertEqual(recorded["inputs"]["computed_limits"]["decode"]["bound"], "upper bound, draft cost excluded")
        self.assertEqual(recorded["priorities"]["p1"], ["decode"])
        self.assertEqual(recorded["priorities"]["sources"]["decode"], "alternative")
        self.assertEqual(recorded["strategy"]["choice"], "current")  # used whatever the plan chooses
        # An alternative below the current strategy's limit changes nothing.
        lower = self.store.plan(self.run_id, plan(parent, strategy=self.mtp(tokens_per_round=1.0, draft_steps=0)))
        self.assertFalse(lower["inputs"]["alternative_limits"]["decode"]["won"])
        self.assertEqual(lower["priorities"]["p1"], ["prefill"])
        self.assertEqual(lower["priorities"]["sources"]["decode"], "computed")

    def test_rules_2_charge_draft_steps_at_evals_draft_floor(self):
        parent = self.first()["candidate_id"]
        # A drafting engine at width 2 and 2 tokens per round; Eval's draft floor is 1 ms a step (10 ms pass floor).
        result = headroom()
        result["headroom"] |= {"draft_width": 2, "draft_bytes_per_step": 5e9, "projection": {"draft_floor_ms": 1.0}}
        self.evaluated(parent, limitless_decode(status(decode=100.0)), **result)
        recorded = self.store.plan(self.run_id, plan(parent, strategy=self.mtp(tokens_per_round=2.0, draft_steps=2)))
        computed = recorded["inputs"]["computed_limits"]["decode"]
        # Eval's draft floor wins over the 5 GB the attempt reports: 2 x 1000 / (10 + 2 x 1) ms.
        self.assertAlmostEqual(computed["limit"], 2000 / 12)
        self.assertEqual((computed["bound"], computed["draft_steps"], computed["draft_floor_ms"]),
                         ("Eval's round floor, draft steps at Eval's draft floor", 2, 1.0))
        self.assertEqual(computed["inputs"]["draft_floor_ms"], "headroom.projection.draft_floor_ms")
        # The same strategy at the same tokens per round gets the same limit from either side, and does not win.
        entry = recorded["inputs"]["alternative_limits"]["decode"]
        self.assertAlmostEqual(entry["limit"], computed["limit"])
        self.assertFalse(entry["won"])

    def test_rules_1_charge_the_alternative_its_draft_steps(self):
        with patch("crit.run.records.NEW_RUN_PLAN_RULES", None):
            self.view = self.store.create(self.data)
        self.run, self.run_id = self.view["run"], self.view["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)
        parent = self.baseline()
        # 3.5 tokens per round over 13.4 ms + 3 draft steps of 0.9 GB: ~229 tok/s.
        entry = self.store.plan(self.run_id, plan(parent, strategy=self.mtp(drafter=None)))["inputs"]["alternative_limits"]["decode"]
        self.assertAlmostEqual(entry["limit"], 3.5 * 1000 / (19.26e9 / 1437.7e6 + 3 * 0.9e9 / 1437.7e6), places=3)
        self.assertEqual(entry["inputs"]["draft_bytes_per_step"], "headroom.draft_bytes_per_step")
        self.assertNotIn("bound", entry)

    def test_the_alternatives_ceiling_is_refused_where_crit_or_the_plan_cannot_give_it(self):
        parent = self.baseline()
        given = {**strategy(), "alternative": {**strategy()["alternative"],
                                               "limits": {"decode": {"limit": 215.0, "derivation": "x", "drafter": "new"}}}}
        with self.assertRaisesRegex(ValueError, "gives a bare limit"):
            self.store.plan(self.run_id, plan(parent, strategy=given))
        for limits, message in (({"decode": {"derivation": "x"}}, "not both"),
                                ({"decode": {"tokens_per_round": 3.0, "limit": 2.0, "derivation": "x"}}, "not both"),
                                ({"decode": {"tokens_per_round": 0.5, "derivation": "x"}}, "tokens_per_round"),
                                ({"ttft": {"limit": 50.0, "derivation": "x"}}, "maps decode and context_decode"),
                                ({"decode": {"tokens_per_round": 3.0, "derivation": " "}}, "derivation")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.store.plan(self.run_id, plan(parent, strategy={
                        **strategy(), "alternative": {**strategy()["alternative"], "limits": limits}}))

    def test_the_alternatives_tokens_per_round_stay_within_its_draft_steps(self):
        self.assertEqual(self.run["spec"]["plan_rules"], 2)
        prompt = " ".join((self.directory / "agent_request.md").read_text(encoding="utf-8").split())
        self.assertIn("tokens per round is at most its draft steps + 1", prompt)
        parent = self.baseline()
        # Three draft steps emit at most four tokens a round; zero draft steps one.
        for tokens_per_round, draft_steps, bound in ((4.5, 3, 4), (2.0, 0, 1)):
            with self.subTest(tokens_per_round=tokens_per_round):
                with self.assertRaisesRegex(ValueError, f"tokens_per_round {tokens_per_round:g} is above draft_steps \\+ 1 = {bound}"):
                    self.store.plan(self.run_id, plan(parent, strategy=self.mtp(tokens_per_round, draft_steps)))
        recorded = self.store.plan(self.run_id, plan(parent, strategy=self.mtp(4.0, 3)))
        self.assertEqual(recorded["inputs"]["alternative_limits"]["decode"]["tokens_per_round"], 4.0)
        # A run created before the bound keeps the rules it was created with.
        with patch("crit.run.records.NEW_RUN_PLAN_RULES", None):
            self.view = self.store.create(self.data)
        self.run, self.run_id = self.view["run"], self.view["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)
        self.assertNotIn("plan_rules", self.run["spec"])
        parent = self.baseline()
        self.assertEqual(self.store.plan(self.run_id, plan(parent, strategy=self.mtp(4.5, 3, drafter=None)))
                         ["inputs"]["alternative_limits"]["decode"]["tokens_per_round"], 4.5)

    def test_rules_2_check_each_gap_and_lever_against_its_own_noise(self):
        parent = self.first()["candidate_id"]
        # Prefill (5% noise) sets its family; long-context first token has 1% noise and a +2% gap.
        value = status(context=((40.0, 100.0), (1000.0, 900.0)), gaps=[("context_ttft", "KV reads at fp8", 0.02)])
        for goal in value["goals"]:
            goal["noise"] = {"prefill": 0.05, "context_ttft": 0.01}.get(goal["kpi"], 0.02)
        self.evaluated(parent, value)
        forecast = {**plan()["forecast"], "context_decode": {"low": 40.0, "expected": 50.0, "high": 60.0},
                    "context_ttft": {"low": 900.0, "expected": 950.0, "high": 1000.0}}
        decode_only = plan(parent, prefill=False, forecast=forecast, levers=[lever()], gaps=limit_causes("prefill"))
        with self.assertRaisesRegex(ValueError, "context_ttft gap 'KV reads at fp8' \\(\\+2.0%\\) is above the noise"):
            self.store.plan(self.run_id, decode_only)
        answered = {**decode_only, "gaps": [{"kpi": "context_ttft", "gap": "KV reads at fp8", "answer": "rejected",
                                             "detail": "measured: fp8 KV failed the quality gate"}] + limit_causes("prefill")}
        recorded = self.store.plan(self.run_id, answered)
        self.assertEqual(recorded["inputs"]["noise_default"], {"noise": 0.05, "source": "prefill"})
        # A lever of +0.1% on a KPI of unknown noise is below the default noise (rules 1 counted any positive change).
        value = status()
        value["goals"][0]["noise"] = None
        value["kpis"]["end_to_end"] = {"ttft_noise": 0.02}
        self.evaluated(parent, value)
        tiny = lever("tiny", {"decode": {"change": 0.001, "source": "estimate"}})
        with self.assertRaisesRegex(ValueError, "tiny changes no KPI by more than its noise"):
            self.store.plan(self.run_id, plan(parent, levers=plan()["levers"] + [tiny]))

    def test_rules_2_answer_why_a_family_short_of_its_limit_is_not_needed(self):
        parent = self.first()["candidate_id"]
        # Prefill at 25% of its limit, marked not needed with no Eval gap above its noise.
        decode_only = plan(parent, prefill=False, levers=[lever()])
        with self.assertRaisesRegex(ValueError, 'prefill is marked not needed at 25% of its limit, below 85%: add a gaps '
                                                'entry {"kpi": "prefill", "gap": "limit", "answer": "cause"'):
            self.store.plan(self.run_id, decode_only)
        for gaps, message in (([{**limit_causes("prefill")[0], "answer": "rejected"}], "answers cause"),
                              ([{**limit_causes("prefill")[0], "kpi": "context_ttft"}], "names a family")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.store.plan(self.run_id, {**decode_only, "gaps": gaps})
        recorded = self.store.plan(self.run_id, {**decode_only, "gaps": limit_causes("prefill")})
        self.assertEqual(recorded["gaps"], limit_causes("prefill"))
        # At or above NEAR_LIMIT no cause is asked: prefill at 90%.
        self.evaluated(parent, status(prefill=3600.0))
        self.assertEqual(self.store.plan(self.run_id, decode_only)["priorities"]["p1"], ["decode"])

    def test_plan_rules_need_computed_limits(self):
        spec = copy.deepcopy(self.run["spec"])
        for broken in ({**spec, "plan_rules": 5}, {key: value for key, value in spec.items() if key != "plan_limits"}):
            with self.subTest(broken=broken.get("plan_rules")):
                with self.assertRaisesRegex(ValueError, "plan_rules is 2, 3 or 4 when present"):
                    validate_spec(broken)

    def test_without_the_round_floors_facts_the_plan_gives_the_alternatives_ceiling(self):
        # Rules 2 refuse a bare limit (test_rules_2_bound_every_alternative_by_its_tokens_per_round); rules 1 take it.
        with patch("crit.run.records.NEW_RUN_PLAN_RULES", None):
            self.view = self.store.create(self.data)
        self.run, self.run_id = self.view["run"], self.view["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)
        parent = self.first()["candidate_id"]
        self.evaluated(parent, limitless_decode(status(decode=62.2, prefill=1340.0)))
        estimate = {"limit": 75.0, "derivation": "one pass per token"}
        with self.assertRaisesRegex(ValueError, "crit cannot compute the round floor"):
            self.store.plan(self.run_id, plan(parent, estimated_limits={"decode": estimate}, strategy=self.mtp(drafter=None)))
        given = {**strategy(), "alternative": {**strategy()["alternative"],
                                               "limits": {"decode": {"limit": 215.0, "derivation": "MTP at 3.5 tokens"}}}}
        recorded = self.store.plan(self.run_id, plan(parent, estimated_limits={"decode": estimate}, strategy=given,
                                                     forecast={**plan()["forecast"], "decode": {"low": 90.0, "expected": 130.0, "high": 170.0}}))
        entry = recorded["inputs"]["alternative_limits"]["decode"]
        self.assertEqual((entry["limit"], entry["computed_by"], entry["current_source"], entry["won"]),
                         (215.0, "plan", "estimate", True))
        # 62.2 / 215 = 0.29 against prefill's 0.335 and its 2% noise band: decode p1.
        self.assertEqual(recorded["priorities"]["p1"], ["decode"])
        # The outcome compares the alternative's ceiling with the measurement once the plan builds it.
        chosen = self.store.plan(self.run_id, plan(parent, estimated_limits={"decode": estimate},
                                                   strategy={**given, "choice": "alternative"},
                                                   forecast={**plan()["forecast"], "decode": {"low": 90.0, "expected": 130.0, "high": 170.0}}))
        child = self.candidate(parent_candidate_id=parent, plan_id=chosen["plan_id"])["candidate_id"]
        value = self.store.plan_outcome(self.run_id, child, self.evaluated(child, status(decode=180.0)))
        self.assertEqual(value["alternative"]["decode"]["limit"], 215.0)
        self.assertEqual(value["alternative"]["decode"]["measured"], 180.0)
        self.assertFalse(value["alternative"]["decode"]["exceeded"])

    def test_an_older_run_ranks_by_the_current_strategy_only(self):
        with patch("crit.run.records.NEW_RUN_PLAN_LIMITS", None):
            self.view = self.store.create(self.data)
        self.run, self.run_id = self.view["run"], self.view["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)
        parent = self.first()["candidate_id"]
        self.evaluated(parent, status())
        with self.assertRaisesRegex(ValueError, "ranks by the current strategy's limits only"):
            self.store.plan(self.run_id, plan(parent, strategy=self.mtp()))

    def test_an_early_stop_is_a_stop_plan_from_the_newest_eval(self):
        parent = self.first()["candidate_id"]
        account = {"kpis": [{"kpi": kpi, "start": 1.0, "reached": 1.0, "limit": 2.0, "built": "fixture", "cause": "fixture"}
                            for kpi in ("decode", "prefill", "ttft")]}
        with self.assertRaisesRegex(ValueError, "requires a stop plan"):
            self.store.stop(self.run_id, "search_exhausted", critique="critique", headroom=account)
        stop = self.store.plan(self.run_id, plan(parent, decision="stop", decode=False, prefill=False, levers=[],
                                                 forecast=None, gaps=limit_causes("decode", "prefill")))
        # A newer Eval of the newest candidate makes the stop plan stale.
        self.evaluated(parent, status(context=((40.0, 100.0), (4000.0, 1000.0))))
        with self.assertRaisesRegex(ValueError, "not the newest Eval"):
            self.store.stop(self.run_id, "search_exhausted", critique="critique", headroom=account)
        self.store.plan(self.run_id, plan(parent, decision="stop", decode=False, prefill=False, levers=[],
                                          forecast=None, supersedes=stop["plan_id"], gaps=limit_causes("decode", "prefill")))
        # That Eval measured the long-context curve: the account covers it too.
        with self.assertRaisesRegex(ValueError, "account of context_decode, context_ttft"):
            self.store.stop(self.run_id, "search_exhausted", critique="critique", headroom=account)
        account["kpis"] += [{**account["kpis"][0], "kpi": kpi} for kpi in plans.CONTEXT]
        result = self.store.stop(self.run_id, "search_exhausted", critique="critique", headroom=account)
        self.assertEqual(result["state"]["stopping_reason"], "search_exhausted")

    def test_with_the_budget_used_no_stop_plan_is_needed(self):
        parent = self.first()["candidate_id"]
        for _ in range(2):
            recorded = self.store.plan(self.run_id, plan(parent))
            self.candidate(parent_candidate_id=parent, plan_id=recorded["plan_id"])
        account = {"kpis": [{"kpi": kpi, "start": 1.0, "reached": 1.0, "limit": 2.0, "built": "fixture", "cause": "fixture"}
                            for kpi in ("decode", "prefill", "ttft")]}
        self.assertEqual(self.store.stop(self.run_id, "search_exhausted", critique="critique", headroom=account)
                         ["state"]["stopping_reason"], "search_exhausted")


def attempt_of(test, candidate_id):
    return sorted((test.directory / "candidates" / candidate_id / "eval").glob("attempt-*.json"))[0].stem


class PlanRules3Tests(PlanFixture, unittest.TestCase):
    """Plan rules 3, the default of new runs: forecasts against the priorities, measured rejections below half the
    limit, and recon's prefill arithmetic plan."""
    plan_rules = 3

    def test_new_runs_check_plans_with_rules_3(self):
        self.assertEqual(self.run["spec"]["plan_rules"], 3)

    def test_a_forecast_that_gains_more_on_p2_or_lets_p1_fall_says_why(self):
        parent = self.first()["candidate_id"]
        # Prefill at 25% of its limit is p1, decode at 50% p2.
        self.evaluated(parent, status())
        p2_heavy = plan(parent, forecast={**plan()["forecast"], "decode": {"low": 120.0, "expected": 160.0, "high": 180.0},
                                          "prefill": {"low": 1000.0, "expected": 1100.0, "high": 1200.0}})
        with self.assertRaisesRegex(ValueError, "expects more gain on p2 \\(decode: \\+60.0%\\) than on p1 \\(prefill: "
                                                "\\+10.0%\\).*forecast_justification"):
            self.store.plan(self.run_id, p2_heavy)
        falling = plan(parent, forecast={**plan()["forecast"], "prefill": {"low": 800.0, "expected": 900.0, "high": 1000.0}})
        with self.assertRaisesRegex(ValueError, "forecasts p1 prefill to fall \\(1000 -> 900\\)"):
            self.store.plan(self.run_id, falling)
        why = "the drafter rework lands first; the prefill GEMMs follow it next candidate"
        recorded = self.store.plan(self.run_id, {**p2_heavy, "forecast_justification": why})
        self.assertEqual(recorded["forecast_justification"], why)
        self.assertEqual(len(recorded["inputs"]["forecast_checks"]), 1)
        self.assertEqual(self.store.plan(self.run_id, plan(parent))["inputs"]["forecast_checks"], [])

    def evidence(self, name, value):
        path = self.directory / "work" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return f"work/{name}"

    def test_a_family_below_half_its_limit_is_not_needed_only_on_measured_rejections(self):
        parent = self.first()["candidate_id"]
        self.evaluated(parent, status(gaps=[("prefill", "prefill at its compute ceiling", 3.0)]))
        decode_only = plan(parent, prefill=False, levers=[lever()])
        answered = [{"kpi": "prefill", "gap": "prefill at its compute ceiling", "answer": "cause",
                     "detail": "quality budget"}] + limit_causes("prefill")
        with self.assertRaisesRegex(ValueError, "below 50%: gaps entry 'prefill at its compute ceiling' needs the evidence"):
            self.store.plan(self.run_id, {**decode_only, "gaps": answered})
        measured = self.evidence("prefill-gemm-ab.json", {"note": "fixture measurement"})
        with_evidence = [{**row, "evidence": measured} for row in answered]
        with self.assertRaisesRegex(ValueError, "answers cause: below half the limit an Eval gap is answered by a measured "
                                                "rejection"):
            self.store.plan(self.run_id, {**decode_only, "gaps": with_evidence})
        rejected = [{**with_evidence[0], "answer": "rejected"}, with_evidence[1]]
        recorded = self.store.plan(self.run_id, {**decode_only, "gaps": rejected})
        self.assertEqual(recorded["inputs"]["gap_evidence"]["prefill:limit"]["path"], measured)
        with self.assertRaisesRegex(ValueError, "is no file of this run"):
            self.store.plan(self.run_id, {**decode_only, "gaps": [{**row, "evidence": "work/missing.json"} for row in rejected]})

    def test_prefill_below_half_its_limit_rejects_every_faster_activation_format_outside_the_quality_noise(self):
        parent = self.first()["candidate_id"]
        value = status()
        value["kpis"]["prefill"]["activations"] = {
            "formats": ["nvfp4"], "declared": {"ceiling_score": 2000.0, "state": "passes"},
            "modes": {"bf16": {"ceiling_score": 1500.0, "state": None}, "fp8": {"ceiling_score": 3000.0, "state": None},
                      "stored": {"ceiling_score": 4000.0, "state": "fails"}}}
        self.evaluated(parent, value)
        measured = self.evidence("limit.json", {})
        decode_only = plan(parent, prefill=False, levers=[lever()],
                           gaps=[{**limit_causes("prefill")[0], "evidence": measured}])
        # Stored is ruled out by Eval's own evidence and BF16 is slower than the engine's own: FP8 needs a rejection.
        with self.assertRaisesRegex(ValueError, "prices prompt activations fp8 at 3,000 tok/s, above the engine's own "
                                                "2,000: reject it with a quality measurement"):
            self.store.plan(self.run_id, decode_only)

        def quality(activations, p05, spread=True):
            sets = {"long-4096/prompt": {"budget_used": 1.3, **({"budget_spread": {"budget_used": {
                "p05": p05, "p50": 1.3, "p95": 1.6}}} if spread else {})}}
            label = activations if isinstance(activations, str) else "shares"
            return self.evidence(f"fp8-{p05}-{spread}-{label}.json",
                                 {"engine": {"prompt_activations": activations}, "quality": {"sets": sets}})

        def with_fp8(path):
            return {**decode_only, "gaps": decode_only["gaps"] + [
                {"kpi": "prefill", "gap": "activation:fp8", "answer": "rejected", "evidence": path,
                 "detail": "FP8 activations on the MLP"}]}
        for path, message in ((quality("fp8", 0.9), "inside the quality noise, so not a rejection"),
                              (quality("fp8", 1.1, spread=False), "no quality spread over positions"),
                              (quality("bf16", 1.1), "measured prompt activations 'bf16', not 'fp8'")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.store.plan(self.run_id, with_fp8(path))
        recorded = self.store.plan(self.run_id, with_fp8(quality({"nvfp4": "fp8"}, 1.1)))
        self.assertEqual(recorded["inputs"]["gap_evidence"]["prefill:activation:fp8"]["over_budget_sets"],
                         ["long-4096/prompt"])
        value["kpis"]["prefill"]["activations"]["declared"] = None
        self.evaluated(parent, value)
        with self.assertRaisesRegex(ValueError, "the engine declares no prompt_activations"):
            self.store.plan(self.run_id, with_fp8(quality("fp8", 1.1)))

class PlanRules4Tests(PlanFixture, unittest.TestCase):
    """Plan rules 4, the default of new runs: an objective per candidate, critique rounds naming the parent's two
    weakest terms, and a +10% floor on p1 for candidates 2 to 5."""
    plan_rules = 4

    def test_new_runs_check_plans_with_rules_4(self):
        from crit.run import records
        self.assertEqual(records.NEW_RUN_PLAN_RULES, 4)
        self.assertEqual(self.run["spec"]["plan_rules"], 4)

    def next_candidate(self, parent, **updates):
        recorded = self.store.plan(self.run_id, plan(parent, **updates))
        candidate = self.candidate(parent_candidate_id=parent, plan_id=recorded["plan_id"])["candidate_id"]
        self.evaluated(candidate, status())
        return recorded, candidate

    def evidence(self, name):
        path = self.directory / "work" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return f"work/{name}"

    def test_each_plan_records_its_candidates_objective(self):
        first = self.store.plan(self.run_id, plan(levers=[], forecast=None))
        self.assertEqual(first["inputs"]["objective"]["name"], "proof_of_concept")
        self.assertEqual(first["inputs"]["objective"]["candidate"], 1)
        parent = self.candidate(plan_id=first["plan_id"])["candidate_id"]
        self.evaluated(parent, status())
        second, parent = self.next_candidate(parent)
        self.assertEqual((second["inputs"]["objective"]["name"], second["inputs"]["objective"]["min_gain"]),
                         ("crit_round", 0.1))
        third, parent = self.next_candidate(parent)
        self.assertEqual(third["inputs"]["objective"]["candidate"], 3)
        self.assertEqual(third["inputs"]["objective"]["name"], "crit_round")

    def test_candidates_2_to_5_forecast_at_least_ten_percent_on_p1(self):
        parent = self.first()["candidate_id"]
        # Prefill at 25% of its limit is p1: +5% on it is refused however much decode gains.
        small = plan(parent, forecast={**plan()["forecast"], "decode": {"low": 120.0, "expected": 160.0, "high": 180.0},
                                       "prefill": {"low": 1000.0, "expected": 1050.0, "high": 1100.0}},
                     forecast_justification="decode first")
        with self.assertRaisesRegex(ValueError, "Candidate 2 forecasts \\+5.0% on p1 \\(prefill\\): candidates 2 to 5 "
                                                "are planned for at least \\+10%"):
            self.store.plan(self.run_id, small)
        enough = plan(parent, forecast={**plan()["forecast"], "decode": {"low": 100.0, "expected": 105.0, "high": 110.0},
                                        "prefill": {"low": 1050.0, "expected": 1100.0, "high": 1200.0}})
        self.assertEqual(self.store.plan(self.run_id, enough)["inputs"]["objective"]["candidate"], 2)

    def test_a_critique_round_names_the_parents_two_weakest_terms_and_other_levers(self):
        parent = self.first()["candidate_id"]
        for _ in range(2):
            _, parent = self.next_candidate(parent)
        trace = self.evidence("kernels-c3.json")
        weakest = [{"kpi": "prefill", "term": "gated delta 927 ms vs 13 ms floor", "evidence": trace,
                    "levers": ["chunked-prefill"]},
                   {"kpi": "decode", "term": "verify pass 16.5 vs 13.4 ms floor", "evidence": trace, "levers": ["graphs"]}]
        with self.assertRaisesRegex(ValueError, "Candidate 4 is a critique round: name the parent's two weakest terms"):
            self.store.plan(self.run_id, plan(parent))
        with self.assertRaisesRegex(ValueError, "add other levers beside the ones for its two weakest terms"):
            self.store.plan(self.run_id, plan(parent, weakest=weakest))
        levers = plan()["levers"] + [lever("fused-norms", {"decode": {"change": 0.1, "source": "estimate"}})]
        with self.assertRaisesRegex(ValueError, "graphs is no speed lever of this plan that improves prefill"):
            self.store.plan(self.run_id, plan(parent, levers=levers, weakest=[weakest[0], {**weakest[1], "kpi": "prefill"}]))
        with self.assertRaisesRegex(ValueError, "is no file of this run"):
            self.store.plan(self.run_id, plan(parent, levers=levers, weakest=[weakest[0], {**weakest[1], "evidence": "work/x"}]))
        recorded = self.store.plan(self.run_id, plan(parent, levers=levers, weakest=weakest))
        self.assertEqual(recorded["inputs"]["objective"]["name"], "critique_round")
        self.assertEqual(recorded["weakest"], weakest)


    def test_every_speed_lever_names_its_provider_and_own_code_its_yardstick(self):
        parent = self.first()["candidate_id"]
        bare = plan(parent)
        bare["levers"] = [{k: v for k, v in row.items() if k != "provider"} if row["id"] == "graphs" else row
                          for row in bare["levers"]]
        with self.assertRaisesRegex(ValueError, "graphs names no provider: inference:, hub:, lib:, own"):
            self.store.plan(self.run_id, bare)
        with self.assertRaisesRegex(ValueError, "provider starts with inference:, hub:, lib:, own"):
            self.store.plan(self.run_id, plan(parent, levers=[{**lever(), "provider": "somewhere"}, plan()["levers"][1]]))
        own = {**lever(), "provider": "own: a warp-specialized NVFP4 GEMM"}
        with self.assertRaisesRegex(ValueError, "is the engine's own code: give its yardstick"):
            self.store.plan(self.run_id, plan(parent, levers=[own, plan()["levers"][1]]))
        measured = {"evidence": self.evidence("cublaslt-fp4.json"), "detail": "cuBLASLt NVFP4 GEMM 1,050 TFLOPS on our shapes"}
        recorded = self.store.plan(self.run_id, plan(parent, levers=[{**own, "yardstick": measured}, plan()["levers"][1]]))
        self.assertEqual(recorded["levers"][0]["yardstick"], measured)
        with self.assertRaisesRegex(ValueError, "is no file of this run"):
            self.store.plan(self.run_id, plan(parent, levers=[{**own, "yardstick": {**measured, "evidence": "work/x"}},
                                                              plan()["levers"][1]]))

    def test_the_next_plan_reviews_the_parents_plan_against_its_measured_eval(self):
        parent = self.first()["candidate_id"]
        planned = self.store.plan(self.run_id, plan(parent))
        child = self.candidate(parent_candidate_id=parent, plan_id=planned["plan_id"],
                               lever_fates={"graphs": {"fate": "built"},
                                            "chunked-prefill": {"fate": "deferred", "detail": "tiling variants slower"}})
        child = child["candidate_id"]
        # Decode 130 lands inside its forecast (110-150); prefill 1050 misses its low of 1100.
        attempt = self.evaluated(child, status(decode=130.0, prefill=1050.0))
        self.store.plan_outcome(self.run_id, child, attempt)
        trace = self.evidence("kernels-c2.json")
        with self.assertRaisesRegex(ValueError, f"Review {planned['plan_id']} first: .*\\(prefill\\)"):
            self.store.plan(self.run_id, plan(child))
        review = {"plan_id": planned["plan_id"], "lesson": "the prompt GEMM is the term to rewrite",
                  "kpis": {"prefill": {"cause": "the GEMM stayed at 270 TFLOPS", "evidence": trace}},
                  "levers": {"graphs": {"result": "decode +30%"}, "chunked-prefill": {"result": "deferred"}}}
        with self.assertRaisesRegex(ValueError, "prefill missed its forecast"):
            self.store.plan(self.run_id, plan(child, review={**review, "kpis": {}}))
        with self.assertRaisesRegex(ValueError, "say what graphs of"):
            self.store.plan(self.run_id, plan(child, review={**review, "levers": {"chunked-prefill": {"result": "-"}}}))
        # chunked-prefill was the plan's largest p1 lever and was deferred: planned again, or rejected on a measurement
        # of another structure.
        without = [lever(), lever("attention", {"prefill": {"change": 0.3, "source": "estimate"}})]
        with self.assertRaisesRegex(ValueError, "chunked-prefill was deferred, the largest p1 lever"):
            self.store.plan(self.run_id, plan(child, levers=without, review=review))
        rejected = {**review, "levers": {**review["levers"], "chunked-prefill": {
            "result": "deferred", "evidence": trace, "structure": "cuBLASLt FP8 GEMM on the same shapes"}}}
        recorded = self.store.plan(self.run_id, plan(child, levers=without, review=rejected))
        self.assertEqual(recorded["inputs"]["review"]["missed"], ["prefill"])
        self.assertEqual(recorded["inputs"]["review"]["largest_p1_lever"], "chunked-prefill")
        self.assertEqual(self.store.plan(self.run_id, plan(child, review=review))["review"], review)


class ReconPlanRules3Tests(PlanFixture, unittest.TestCase):
    """Plan rules 3 with a model folder: recon's plans are written before the first candidate."""
    plan_rules = 3
    model_files = {"README.md": "# Fixture model\n"}

    def test_recon_needs_a_prefill_arithmetic_plan_beside_the_drafting_plan(self):
        from crit.run import recon
        self.store.recon(self.run_id)
        notes = self.directory / recon.RECON_MD
        self.assertIn("## Prefill arithmetic plan\n\nTODO.", notes.read_text(encoding="utf-8"))
        self.assertEqual(recon.unfinished(self.directory), list(recon.PLANS))
        recorded = self.store.plan(self.run_id, plan(levers=[], forecast=None))
        with self.assertRaisesRegex(ValueError, "Finish work/recon.md before the candidate: Drafting plan for both decode "
                                                "modes, Prefill arithmetic plan"):
            self.candidate(plan_id=recorded["plan_id"])
        text = notes.read_text(encoding="utf-8")
        for title in recon.PLANS:
            text = text.replace(f"## {title}\n\nTODO.", f"## {title}\n\nWritten: the fixture's plan.")
        notes.write_text(text, encoding="utf-8")
        self.assertEqual(recon.unfinished(self.directory), [])
        self.assertEqual(self.candidate(plan_id=recorded["plan_id"])["plan_id"], recorded["plan_id"])


class PriorityTests(unittest.TestCase):
    def facts(self, decode, prefill, noise=None, context=None):
        value = status(decode=decode * 200, prefill=prefill * 4000, noise=noise or 0, context=context)
        if noise is None:
            for goal in value["goals"]:
                goal["noise"] = None
            value["kpis"]["end_to_end"] = {}
        return plans.extract(value)

    def needed(self, decode=True, prefill=True):
        return {"decode": {"needed": decode}, "prefill": {"needed": prefill}}

    def test_the_priority_table(self):
        facts = self.facts(0.75, 0.25, 0.02)
        for needed, expected in (
                (self.needed(), {"p1": ["prefill"], "p2": ["decode"], "p3": ["ttft"]}),
                (self.needed(prefill=False), {"p1": ["decode"], "p2": ["ttft"], "p3": []}),
                (self.needed(decode=False), {"p1": ["prefill"], "p2": ["ttft"], "p3": []})):
            self.assertEqual(plans.priorities(needed, facts), (expected, None))
        self.assertEqual(plans.priorities(self.needed(), self.facts(0.25, 0.75, 0.02))[0],
                         {"p1": ["decode"], "p2": ["prefill"], "p3": ["ttft"]})
        order, reason = plans.priorities(self.needed(False, False), facts)
        self.assertIsNone(order)
        self.assertIn("stop plan", reason)
        self.assertIn("no Eval yet", plans.priorities(self.needed(), None)[1])

    def test_a_near_tie_puts_both_in_p1(self):
        # |0.50 - 0.45| = 0.05 is within 0.12 × 0.50 = 0.06, the larger noise band of the two families.
        self.assertEqual(plans.priorities(self.needed(), self.facts(0.50, 0.45, 0.12))[0],
                         {"p1": ["decode", "prefill"], "p2": ["ttft"], "p3": []})
        # With 2% noise the same fractions are apart: prefill is further from its limit.
        self.assertEqual(plans.priorities(self.needed(), self.facts(0.50, 0.45, 0.02))[0]["p1"], ["prefill"])
        # Unknown noise counts as 0: only equal fractions tie.
        self.assertEqual(plans.priorities(self.needed(), self.facts(0.50, 0.45))[0]["p1"], ["prefill"])
        self.assertEqual(plans.priorities(self.needed(), self.facts(0.50, 0.50))[0]["p1"], ["decode", "prefill"])

    def test_families_take_their_members_lower_fraction(self):
        # Long-context decode at 20% of its ceiling pulls the decode family below prefill's 45%.
        facts = self.facts(0.75, 0.45, 0.02, context=((20.0, 100.0), (1000.0, 900.0)))
        self.assertEqual(plans.family(facts, "decode")[0], 0.2)
        self.assertEqual(plans.priorities(self.needed(), facts)[0]["p1"], ["decode"])
        # Long-context first token at 10% of its floor pulls the prefill family lower still.
        facts = self.facts(0.75, 0.45, 0.02, context=((20.0, 100.0), (1000.0, 100.0)))
        self.assertEqual(plans.priorities(self.needed(), facts)[0]["p1"], ["prefill"])

    def noisy(self, decode, prefill, noise, context=None):
        """Facts with these fractions and a relative noise per KPI (missing: unknown); context is the
        (context_decode, context_ttft) fractions."""
        value = status(decode=decode * 200, prefill=prefill * 4000, noise=0,
                       context=((context[0] * 100, 100.0), (1000.0, context[1] * 1000)) if context else None)
        value["kpis"]["end_to_end"] = {}
        value["goals"] = [{**goal, "noise": noise.get(goal["kpi"])} for goal in value["goals"]]
        return plans.extract(value)

    def test_rules_2_rank_by_the_largest_noise_of_each_family(self):
        # The smallest monotonicity break of rules 1: long-context first token, of unknown noise, moves from above
        # prefill's 0.65 to below it, and prefill falls from p1 to p2 although it moved further from its limit.
        noise = {"decode": 0.01, "prefill": 0.08, "context_decode": 0.01}
        before, after = self.noisy(0.60, 0.65, noise, (0.99, 0.67)), self.noisy(0.60, 0.65, noise, (0.99, 0.62))
        self.assertEqual(plans.priorities(self.needed(), before)[0]["p1"], ["decode", "prefill"])
        self.assertEqual(plans.priorities(self.needed(), after)[0]["p1"], ["decode"])
        # Rules 2: prefill's noise is the larger of its members' (context_ttft's unknown takes 8%, the largest known).
        self.assertEqual(plans.family(after, "prefill", 2), (0.62, 0.08))
        for facts in (before, after):
            self.assertEqual(plans.priorities(self.needed(), facts, 2)[0]["p1"], ["decode", "prefill"])

    def test_rules_2_give_unknown_noise_a_default(self):
        # Cleanroom candidate-0030: decode 0.531 (1.5% noise), prefill 0.542 of unknown noise, ttft 4.6%.
        facts = self.noisy(0.5312, 0.5421, {"decode": 0.0154, "ttft": 0.0456})
        self.assertEqual(plans.priorities(self.needed(), facts)[0]["p1"], ["decode"])
        self.assertEqual(plans.noise_default(facts), (0.0456, "ttft"))
        # √((0.0154 × 0.531)² + (0.0456 × 0.542)²) = 0.026 ≥ 0.011: a near tie.
        self.assertEqual(plans.priorities(self.needed(), facts, 2)[0]["p1"], ["decode", "prefill"])
        # With no noise known at all, ±1% flips rules 1 from decode to prefill; rules 2 take 5% and tie.
        for decode, first in ((0.495, "decode"), (0.505, "prefill")):
            facts = self.noisy(decode, 0.5, {})
            self.assertEqual(plans.noise_default(facts), (0.05, "default"))
            self.assertEqual(plans.priorities(self.needed(), facts)[0]["p1"], [first])
            self.assertEqual(plans.priorities(self.needed(), facts, 2)[0]["p1"], ["decode", "prefill"])

    def test_rules_2_rank_one_family_when_both_are_near_their_limits(self):
        self.assertEqual(plans.NEAR_LIMIT, 0.85)
        noise = {kpi: 0.03 for kpi in plans.KPIS}
        for decode, prefill, first in ((0.90, 0.92, "decode"), (0.92, 0.90, "prefill"), (0.90, 0.90, "decode")):
            with self.subTest(decode=decode, prefill=prefill):
                facts = self.noisy(decode, prefill, noise)
                self.assertEqual(plans.priorities(self.needed(), facts)[0]["p1"], ["decode", "prefill"])
                self.assertEqual(plans.priorities(self.needed(), facts, 2)[0],
                                 {"p1": [first], "p2": ["prefill" if first == "decode" else "decode"], "p3": ["ttft"]})
        # Below NEAR_LIMIT (one of them at 0.84) the near tie still puts both in p1.
        self.assertEqual(plans.priorities(self.needed(), self.noisy(0.84, 0.86, noise), 2)[0]["p1"], ["decode", "prefill"])

    def test_a_missing_fraction_leaves_the_priorities_open(self):
        value = status()
        value["kpis"]["prefill"]["fraction_of_ceiling"] = None
        value["goals"][1]["fraction_of_limit"] = None
        order, reason = plans.priorities(self.needed(), plans.extract(value))
        self.assertIsNone(order)
        self.assertIn("no fraction of the limit for prefill", reason)

    def test_results_before_goal_fractions_read_the_kpis(self):
        value = status(decode=150.0, noise=0.03)
        del value["goals"]
        value["kpis"]["decode"]["standard_error"] = 0.01
        facts = plans.extract(value)
        self.assertEqual(facts["kpis"]["decode"]["fraction"], 0.75)
        self.assertEqual(facts["noise"]["decode"], 0.02)  # two standard errors
        self.assertEqual(facts["noise"]["ttft"], 0.03)
        self.assertIsNone(facts["noise"]["prefill"])  # unknown, never an error
        self.assertEqual(plans.extract({}), {"kpis": {kpi: {"score": None, "limit": None, "fraction": None}
                                                       for kpi in plans.HEADLINE},
                                             "noise": {kpi: None for kpi in plans.HEADLINE},
                                             "untiered": {"greedy_decode": {"score": None, "noise": None},
                                                          "heldout_decode": {"score": None, "noise": None},
                                                          "followup_ttft": {"score": None, "noise": None}},
                                             "gaps": []})


if __name__ == "__main__":
    unittest.main()
