"""Candidate plans: recorded after the parent's Eval and before any code, checked against the facts they freeze."""
import contextlib
import copy
import io
import json
import unittest

from support.runs import RunFixture
from crit.run import plans
from crit.run.cli import main
from crit.run.records import digest, file_sha, now, read_json, write_json

# Eval's budget for each quality mode, a run's unless it sets its own.
PRESERVE_BUDGET = {"ppl_ratio_max": 1.05, "kld_mean_max": 0.05, "kld_p99_max": 0.5}
SPEED_BUDGET = {"ppl_ratio_max": 1.4, "kld_mean_max": 0.4, "kld_p99_max": 4.0}


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
    """The gaps entries that answer why a family marked not needed below NEAR_LIMIT stops short of its limit, each
    resting on the family's round account (plan() adds one where a plan gives none) and its bound term."""
    return [{"kpi": name, "gap": "limit", "answer": "cause", "detail": "fixture: the rest of the limit is out of reach",
             "cause_terms": ["weights"], "evidence": "run.json"} for name in families]


def round_account(families):
    """A round account per family whose one term runs at 90% of its floor's rate, evidence the run's run.json."""
    return {name: {"total_ms": 10.0, "idle_ms": 0.0, "terms": [{"term": "weights", "ms": 10.0, "floor_ms": 9.0}],
                   "evidence": "run.json"} for name in families}


def lever(lever_id="graphs", changes=None, group=None):
    return {"id": lever_id, "technique": "free text", "description": "What the lever builds",
            "changes": changes if changes is not None else {"decode": {"change": 0.3, "source": "estimate"}}, "group": group, "confidence": "medium",
            "provider": "lib:fixture"}


def strategy():
    return {"current": {"name": "EAGLE drafter", "ceiling": "306 tok/s (Eval decode ceiling)", "evidence": "status"},
            "alternative": {"name": "MTP heads", "ceiling": "280 tok/s (acceptance at width 4)", "evidence": "recon"},
            "choice": "current", "reason": "Higher ceiling"}


def plan(parent=None, decision="build", decode=True, prefill=True, **updates):
    """A plan payload. A family marked not needed, or named by a limit gap, gets the fixture's round account unless the
    plan gives one (round None: none), and a limit gap without cause_terms names its one term."""
    families = sorted(({row["kpi"] for row in updates.get("gaps") or [] if row.get("gap") == "limit"} |
                       {name for name, flag in (("decode", decode), ("prefill", prefill)) if not flag}) &
                      {"decode", "prefill"})
    if "round" in updates and updates["round"] is None:
        updates = {key: value for key, value in updates.items() if key != "round"}
    elif families and "round" not in updates:
        updates = {**updates, "round": round_account(families)}
    if updates.get("gaps"):
        updates["gaps"] = [{"cause_terms": ["weights"], **row} if row.get("gap") == "limit" and row.get("kpi") in
                           ("decode", "prefill") and updates.get("round") else row for row in updates["gaps"]]
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
    # The run's quality mode: preserve, where the prompt-activation check asks nothing, so the other tests plan
    # without prompt_activations; PromptActivationTests judge speed, and None leaves the request without one
    # (the default, speed).
    quality_mode = "preserve"

    def setUp(self):
        super().setUp()
        # A candidate-budget run, as runs are created by default.
        self.data = {key: value for key, value in self.data.items() if key != "budget_minutes"} | {"candidate_budget": 3}
        if self.quality_mode:
            self.data["quality_mode"] = self.quality_mode
        self.view = self.store.create(self.data)
        self.run = self.view["run"]
        self.run_id = self.run["run_id"]
        self.directory = self.store.directory(self.run_id)

    def proposal(self, **updates):
        """A proposal as given: these tests record each plan themselves."""
        value = super().proposal(**{"plan_id": "unset", "lever_fates": {}, **updates})
        for key in ("plan_id", "lever_fates"):
            if key not in updates:
                del value[key]
        return value

    def recreate(self, **data):
        """Switch the test to a new run of this request with `data` over it."""
        self.view = self.store.create({**self.data, **data})
        self.run, self.run_id = self.view["run"], self.view["run"]["run_id"]
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
    def test_every_candidate_requires_a_plan(self):
        self.resolve()
        with self.assertRaisesRegex(ValueError, "Every candidate has a plan"):
            self.store.candidate(self.run_id, self.root, self.proposal())
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
        with self.assertRaisesRegex(ValueError, "candidate-0001, a draft that has not reached the candidate ladder"):
            self.candidate(plan_id="plan-0001")
        # The run has a candidate on the ladder now: a plan names its parent.
        self.evaluated("candidate-0001", status())
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

    def test_the_first_plans_decision_follows_its_needed_flags(self):
        for value, message in ((plan(decode=False, prefill=False, levers=[], forecast=None), "record a stop plan"),
                               (plan(decision="stop", prefill=False, levers=[], forecast=None),
                                "requires both decode and prefill not needed: decode is marked needed")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.store.plan(self.run_id, value)
        self.assertEqual(self.store.plan(self.run_id, plan(prefill=False, levers=[], forecast=None))["plan_id"], "plan-0001")

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
            ({**good, "levers": [lever(changes={"decode": {"change": 0.3, "source": "a hunch"}})]}, "eval-gap:<gap name>, measured:"),
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
                                   "prefill": {"low": 900.0, "expected": 1010.0, "high": 1100.0},
                                   "context_decode": {"low": 39.0, "expected": 40.0, "high": 41.0},
                                   "context_ttft": {"low": 3900.0, "expected": 4000.0, "high": 4100.0}}},
             "improves neither decode, prefill, context_decode nor context_ttft beyond its noise"),
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
                               "detail": "measured: the graph was slower", "evidence": "run.json"}]
                        + limit_causes("decode"),
                        levers=[lever("chunked-prefill", {"prefill": {"change": 0.5, "source": "estimate"}}, "prefill"),
                                lever("tiny", {"prefill": {"change": 0.01, "source": "estimate"}}, "prefill")])
        recorded = self.store.plan(self.run_id, answered)
        self.assertEqual(recorded["priorities"], {"p1": ["prefill"], "p2": ["ttft"], "p3": [], "estimated": [],
                                                  "sources": {}})

    def test_a_parent_without_eval_is_refused_once_the_run_has_one(self):
        first = self.first()["candidate_id"]
        recorded = self.store.plan(self.run_id, plan(first))
        second = self.candidate(parent_candidate_id=first, plan_id=recorded["plan_id"])["candidate_id"]
        with self.assertRaisesRegex(ValueError, f"{second} is not on the candidate ladder yet"):
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

    def test_the_outcome_flags_a_forecast_whose_gain_falls_on_p2(self):
        parent = self.first()["candidate_id"]
        # Prefill is p1, but the forecast puts +100% on decode (p2) and +5% on prefill.
        forecast = {**plan()["forecast"], "decode": {"low": 150.0, "expected": 200.0, "high": 250.0},
                    "prefill": {"low": 1000.0, "expected": 1050.0, "high": 1100.0}}
        lopsided = self.store.plan(self.run_id, plan(parent, forecast=forecast, forecast_justification="fixture"))
        self.assertEqual(lopsided["priorities"]["p1"], ["prefill"])
        child = self.candidate(parent_candidate_id=parent, plan_id=lopsided["plan_id"])["candidate_id"]
        flag = self.store.plan_outcome(self.run_id, child, self.evaluated(child, status()))["forecast_on_p2"]
        self.assertEqual((flag["p1"], flag["p2"]), (["prefill"], ["decode"]))
        self.assertAlmostEqual(flag["p1_gain"], 0.05)
        self.assertAlmostEqual(flag["p2_gain"], 1.0)
        # The default forecast puts +50% on prefill (p1) and +30% on decode: no flag.
        self.assertIsNone(plans.priority_flag({**lopsided, "forecast": plan()["forecast"]}))

    def baseline(self):
        """Point 2 of run 3141e655 in miniature: a plain engine at 62.2 tok/s, one token per pass, 19.26 GB per pass at
        1437.7 GB/s (crit computes ~74.6 tok/s, 83% of it), prefill at 33.5% of its ceiling."""
        parent = self.first()["candidate_id"]
        result = headroom(tokens_per_round=(1.0, 1.0), decode_tps=62.2, bytes_per_pass=19.26e9, bandwidth_gbs=1437.7)
        result["headroom"]["draft_bytes_per_step"] = 0.9e9
        self.evaluated(parent, limitless_decode(status(decode=62.2, prefill=1340.0)), **result)
        return parent

    def mtp(self, tokens_per_round=3.5, draft_steps=3, drafter="new"):
        """The alternative MTP strategy: a new drafter for the plain baseline (drafter None: none named)."""
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

    def test_bound_every_alternative_by_its_tokens_per_round(self):
        parent = self.first()["candidate_id"]
        self.evaluated(parent, limitless_decode(status(decode=62.2, prefill=1340.0)))  # no round floor facts
        for drafter in ("measured", "new"):
            given = {**strategy(), "alternative": {**strategy()["alternative"], "limits": {"decode": {
                "limit": 215.0, "derivation": "x", "drafter": drafter}}}}
            with self.subTest(drafter=drafter):
                with self.assertRaisesRegex(ValueError, "gives a bare limit"):
                    self.store.plan(self.run_id, plan(parent, strategy=given,
                                                      estimated_limits={"decode": {"limit": 75.0, "derivation": "x"}}))
        # tokens_per_round without the floor's facts: the refusal names the way left, not the bare limit refused above
        with self.assertRaisesRegex(ValueError, "leave decode out of strategy.alternative.limits"):
            self.store.plan(self.run_id, plan(parent, strategy=self.mtp(3.5, 3, "new"),
                                              estimated_limits={"decode": {"limit": 75.0, "derivation": "x"}}))

    def test_price_a_companion_drafter_from_recon(self):
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
        entry = self.store.plan(self.run_id, plan(parent, strategy=self.mtp(2.0, 7, "new"),
                                                  forecast_justification="fixture"))["inputs"]["alternative_limits"]["decode"]
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
        current = self.store.plan(self.run_id, plan(parent, forecast_justification="fixture"))
        self.assertEqual(current["priorities"]["p1"], ["prefill"])
        self.assertIsNone(current["inputs"]["alternative_limits"] or None)
        # MTP: 3.5 tokens per round over 13.4 ms + 3 draft steps at the engine's 0.9 GB per step: ~229 tok/s, 27% of
        # it: decode is p1. The plain baseline's computed limit has no draft width: the labelled upper bound.
        recorded = self.store.plan(self.run_id, plan(parent, strategy=self.mtp(), forecast_justification="fixture"))
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
        lower = self.store.plan(self.run_id, plan(parent, strategy=self.mtp(tokens_per_round=1.0, draft_steps=0), forecast_justification="fixture"))
        self.assertFalse(lower["inputs"]["alternative_limits"]["decode"]["won"])
        self.assertEqual(lower["priorities"]["p1"], ["prefill"])
        self.assertEqual(lower["priorities"]["sources"]["decode"], "computed")

    def test_charge_draft_steps_at_evals_draft_floor(self):
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
        parent = self.baseline()
        # Three draft steps emit at most four tokens a round; zero draft steps one.
        for tokens_per_round, draft_steps, bound in ((4.5, 3, 4), (2.0, 0, 1)):
            with self.subTest(tokens_per_round=tokens_per_round):
                with self.assertRaisesRegex(ValueError, f"tokens_per_round {tokens_per_round:g} is above draft_steps \\+ 1 = {bound}"):
                    self.store.plan(self.run_id, plan(parent, strategy=self.mtp(tokens_per_round, draft_steps)))
        recorded = self.store.plan(self.run_id, plan(parent, strategy=self.mtp(4.0, 3)))
        self.assertEqual(recorded["inputs"]["alternative_limits"]["decode"]["tokens_per_round"], 4.0)

    def test_check_each_gap_and_lever_against_its_own_noise(self):
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
                                             "detail": "measured: fp8 KV failed the quality gate",
                                             "evidence": "run.json"}] + limit_causes("prefill")}
        recorded = self.store.plan(self.run_id, answered)
        self.assertEqual(recorded["inputs"]["noise_default"], {})
        # A lever of +0.1% on a KPI of unknown noise is below the default noise .
        value = status()
        value["goals"][0]["noise"] = None
        value["kpis"]["end_to_end"] = {"ttft_noise": 0.02}
        self.evaluated(parent, value)
        tiny = lever("tiny", {"decode": {"change": 0.001, "source": "estimate"}})
        with self.assertRaisesRegex(ValueError, "tiny changes no KPI by more than its noise"):
            self.store.plan(self.run_id, plan(parent, levers=plan()["levers"] + [tiny]))

    def test_answer_why_a_family_short_of_its_limit_is_not_needed(self):
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
            child = self.candidate(parent_candidate_id=parent, plan_id=recorded["plan_id"])["candidate_id"]
            self.evaluated(child, status())   # on the ladder
        account = {"kpis": [{"kpi": kpi, "start": 1.0, "reached": 1.0, "limit": 2.0, "built": "fixture", "cause": "fixture"}
                            for kpi in ("decode", "prefill", "ttft")]}
        self.assertEqual(self.store.stop(self.run_id, "search_exhausted", critique="critique", headroom=account)
                         ["state"]["stopping_reason"], "search_exhausted")


def attempt_of(test, candidate_id):
    return sorted((test.directory / "candidates" / candidate_id / "eval").glob("attempt-*.json"))[0].stem


class MeasuredRejectionTests(PlanFixture, unittest.TestCase):
    """Forecasts against the priorities, and measured rejections below half the limit."""

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

        def quality(activations, p05, spread=True, budget=PRESERVE_BUDGET):
            sets = {"long-4096/prompt": {"budget_used": 1.3, **({"budget_spread": {"budget_used": {
                "p05": p05, "p50": 1.3, "p95": 1.6}}} if spread else {})}}
            label = activations if isinstance(activations, str) else "shares"
            # Judged in preserve, this run's mode, with preserve's budget.
            return self.evidence(f"fp8-{p05}-{spread}-{label}-{budget['kld_mean_max']}.json",
                                 {"engine": {"prompt_activations": activations}, "quality": {"sets": sets},
                                  "requirements": {"quality_mode": "preserve", "quality_budget": budget}})

        def with_fp8(path):
            return {**decode_only, "gaps": decode_only["gaps"] + [
                {"kpi": "prefill", "gap": "activation:fp8", "answer": "rejected", "evidence": path,
                 "detail": "FP8 activations on the MLP"}]}
        for path, message in ((quality("fp8", 0.9), "inside the quality noise, so not a rejection"),
                              (quality("fp8", 1.1, spread=False), "no quality spread over positions"),
                              (quality("bf16", 1.1), "measured prompt activations 'bf16', not 'fp8'"),
                              # A share of a looser budget than the run's measures nothing about the run's.
                              (quality("fp8", 1.1, budget={**PRESERVE_BUDGET, "kld_mean_max": 0.1}),
                               "judged in quality mode preserve with the budget .*0.1.*, this run's terms are "
                               "quality mode preserve with the budget")):
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

class ObjectiveTests(PlanFixture, unittest.TestCase):
    """An objective per candidate, critique rounds naming the parent's two weakest terms, gate and check repairs,
    providers, the review, and a warning below +10% on p1 for candidates 2 to 5."""

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
                         ("plugin_round", 0.1))
        third, parent = self.next_candidate(parent)
        self.assertEqual(third["inputs"]["objective"]["candidate"], 3)
        self.assertEqual(third["inputs"]["objective"]["name"], "crit_round")

    def test_candidates_2_to_5_forecast_at_least_ten_percent_on_p1(self):
        parent = self.first()["candidate_id"]
        # Prefill at 25% of its limit is p1: +5% on it is refused however much decode gains.
        small = plan(parent, forecast={**plan()["forecast"], "decode": {"low": 120.0, "expected": 160.0, "high": 180.0},
                                       "prefill": {"low": 1000.0, "expected": 1050.0, "high": 1100.0}},
                     forecast_justification="decode first")
        warned = self.store.plan(self.run_id, small)["inputs"]["warnings"]
        self.assertTrue(any("candidate 2 forecasts +5.0% on p1 (prefill): candidates 2 to 5 are for at least +10%" in w
                            for w in warned), warned)
        enough = plan(parent, forecast={**plan()["forecast"], "decode": {"low": 100.0, "expected": 105.0, "high": 110.0},
                                        "prefill": {"low": 1050.0, "expected": 1100.0, "high": 1200.0}})
        self.assertEqual(self.store.plan(self.run_id, enough)["inputs"]["objective"]["candidate"], 2)

    def test_a_gate_repair_needs_no_floor_on_p1_but_names_a_gate_the_parent_failed(self):
        recorded = self.store.plan(self.run_id, plan(levers=[], forecast=None))
        parent = self.candidate(plan_id=recorded["plan_id"])["candidate_id"]
        self.evaluated(parent, {**status(), "gates": [{"name": "lower_context", "state": "fail"},
                                                      {"name": "quality", "state": "pass"}]})
        repair = {"id": "warm-at-load", "kind": "requirement", "serves": "gate:lower_context", "technique": "warm-up",
                  "description": "every request path warm at load", "changes": {}, "group": None,
                  "confidence": "high", "provider": "own: load-time warm-up"}
        small = {**plan()["forecast"], "decode": {"low": 95.0, "expected": 101.0, "high": 105.0},
                 "prefill": {"low": 1000.0, "expected": 1020.0, "high": 1040.0}}
        self.assertTrue(any("+10%" in w for w in self.store.plan(self.run_id, plan(parent, forecast=small))
                            ["inputs"]["warnings"]))
        with self.assertRaisesRegex(ValueError, "names gates the parent's Eval did not fail: quality"):
            self.store.plan(self.run_id, plan(parent, forecast=small,
                                              levers=plan()["levers"] + [{**repair, "serves": "gate:quality"}]))
        recorded = self.store.plan(self.run_id, plan(parent, forecast=small, levers=plan()["levers"] + [repair]))
        self.assertEqual(recorded["inputs"]["objective"]["repairs"], ["lower_context"])
        # A repair alone needs no speed lever, nor a forecast that improves decode or prefill (checks 5 to 9) or
        # reaches toward the limit (check 15).
        same = {**plan()["forecast"], "decode": {"low": 95.0, "expected": 100.0, "high": 105.0},
                "prefill": {"low": 950.0, "expected": 1000.0, "high": 1050.0}}
        with self.assertRaisesRegex(ValueError, "requirement levers alone do not make a candidate"):
            self.store.plan(self.run_id, plan(parent, forecast=same,
                                              levers=[{**repair, "serves": "AGENTS.md: warm paths"}]))
        alone = self.store.plan(self.run_id, plan(parent, forecast=same, levers=[repair]))
        self.assertEqual(alone["inputs"]["objective"]["repairs"], ["lower_context"])

    # A requirement lever repairing the resources check, and a forecast that stays where the parent is.
    CHECK_REPAIR = {"id": "headroom", "kind": "requirement", "serves": "check:resources", "technique": "KV pool sizing",
                    "description": "size the KV pool so the run's VRAM headroom stays free at the required context",
                    "changes": {}, "group": None, "confidence": "high", "provider": "own: KV pool sizing"}
    SAME = {**plan()["forecast"], "decode": {"low": 95.0, "expected": 100.0, "high": 105.0},
            "prefill": {"low": 950.0, "expected": 1000.0, "high": 1050.0}}

    def packaged_parent(self):
        """The run's first candidate, packaged and through an Eval whose every gate passed; its Eval attempt id."""
        recorded = self.store.plan(self.run_id, plan(levers=[], forecast=None))
        candidate = self.candidate(plan_id=recorded["plan_id"])
        self.package(candidate)
        attempt = self.evaluated(candidate["candidate_id"], {**status(), "gates": [
            {"name": "capacity", "state": "pass"}, {"name": "quality", "state": "pass"}]})
        return candidate["candidate_id"], attempt

    def check(self, candidate_id, attempt, gate, state):
        """A check of the candidate's package from its Eval attempt, as crit run ... check --from-eval records it."""
        package = read_json(self.directory / "candidates" / candidate_id / "package/manifest.json")
        result = f"candidates/{candidate_id}/eval/{attempt}.json"
        return self.store.append(self.run_id, candidate_id, "check", {
            "case_id": "case-0001", "package_sha256": digest(package), "gate": gate,
            "criteria_sha256": digest(self.run["spec"]["correctness"]), "status": state,
            "rationale": f"crit Eval {attempt}: fixture", "evidence": [{"path": result,
                                                                        "sha256": file_sha(self.directory / result)}],
            "evidence_level": "artifact_verified", "recorded_at": now(),
            "extensions": {"recorded_by": "crit run check --from-eval"}})["check_id"]

    def test_a_check_repair_names_a_check_the_parent_failed_and_needs_no_speed(self):
        # Every gate of the parent's Eval passed, but crit run's resources check failed (free device memory at the
        # required context below the run's headroom): select refuses it, and its child only repairs that.
        parent, attempt = self.packaged_parent()
        repair = self.CHECK_REPAIR
        with self.assertRaisesRegex(ValueError, "A check repair names checks candidate-0001 did not fail: resources "
                                                "\\(no resources check recorded\\)\\. check:NAME repairs a check whose "
                                                "newest record of candidate-0001 failed; record its checks first "
                                                "\\(crit run \\.\\.\\. check --candidate candidate-0001 --from-eval\\)"):
            self.store.plan(self.run_id, plan(parent, forecast=self.SAME, levers=[repair]))
        self.check(parent, attempt, "arithmetic", "passed")
        self.check(parent, attempt, "functional", "passed")
        failed = self.check(parent, attempt, "resources", "failed")
        # Without the repair: requirement levers alone make no candidate; as a gate repair, resources is no Eval gate.
        with self.assertRaisesRegex(ValueError, "requirement levers alone do not make a candidate"):
            self.store.plan(self.run_id, plan(parent, forecast=self.SAME,
                                              levers=[{**repair, "serves": "resources gate: VRAM headroom"}]))
        with self.assertRaisesRegex(ValueError, "A gate repair names gates the parent's Eval did not fail: resources "
                                                "\\(failed: none\\)\\. resources is a check of crit run, not a gate of "
                                                "Eval's status: a lever repairing a failed one serves check:resources"):
            self.store.plan(self.run_id, plan(parent, forecast=self.SAME, levers=[{**repair, "serves": "gate:resources"}]))
        with self.assertRaisesRegex(ValueError, "A check repair names checks candidate-0001 did not fail: arithmetic "
                                                "\\(its newest, check-0001 for case-0001, passed\\)\\. check:NAME "
                                                "repairs a check whose newest record of candidate-0001 failed$"):
            self.store.plan(self.run_id, plan(parent, forecast=self.SAME,
                                              levers=[repair, {**repair, "id": "x", "serves": "check:arithmetic"}]))
        with self.assertRaisesRegex(ValueError, "A check repair names checks crit run does not require: capacity "
                                                "\\(required: arithmetic, functional, resources\\); a gate of the "
                                                "parent's Eval is repaired as gate:NAME"):
            self.store.plan(self.run_id, plan(parent, forecast=self.SAME, levers=[{**repair, "serves": "check:capacity"}]))
        # The repair alone, with the parent's forecast: no speed lever, no +10% on p1, no improvement (checks 5 to 9).
        recorded = self.store.plan(self.run_id, plan(parent, forecast=self.SAME, levers=[repair]))
        path = f"candidates/{parent}/checks/{failed}.json"
        self.assertEqual(recorded["inputs"]["objective"]["check_repairs"], {"resources": [
            {"path": path, "sha256": file_sha(self.directory / path), "check_id": failed, "case_id": "case-0001",
             "recorded_by": "crit run check --from-eval"}]})
        self.assertNotIn("repairs", recorded["inputs"]["objective"])
        # The plan changes nothing about the parent: its resources gate is failed and select refuses it.
        self.store.seal(self.run_id, parent, "built")
        candidate = next(c for c in self.store.get(self.run_id)["candidates"] if c["candidate_id"] == parent)
        self.assertEqual((candidate["gates"]["resources"], candidate["eligible"]), ("failed", False))
        with self.assertRaisesRegex(ValueError, "candidate-0001 is not eligible: required gate resources is failed"):
            self.store.select(self.run_id, parent)

    def test_a_check_repair_follows_the_parents_newest_check(self):
        # The run's first plan has no parent whose check it could repair.
        with self.assertRaisesRegex(ValueError, "A check repair names a failed check of the parent: the run's first "
                                                "plan has none"):
            self.store.plan(self.run_id, plan(levers=[self.CHECK_REPAIR], forecast=None))
        parent, attempt = self.packaged_parent()
        self.check(parent, attempt, "resources", "failed")
        passed = self.check(parent, attempt, "resources", "passed")
        with self.assertRaisesRegex(ValueError, f"did not fail: resources \\(its newest, {passed} for case-0001, "
                                                "passed\\)"):
            self.store.plan(self.run_id, plan(parent, forecast=self.SAME, levers=[self.CHECK_REPAIR]))
        failed = self.check(parent, attempt, "resources", "failed")
        recorded = self.store.plan(self.run_id, plan(parent, forecast=self.SAME, levers=[self.CHECK_REPAIR]))
        self.assertEqual(recorded["inputs"]["objective"]["check_repairs"]["resources"][0]["check_id"], failed)

    def test_a_critique_round_names_the_parents_two_weakest_terms_and_other_levers(self):
        self.recreate(candidate_budget=8)
        parent = self.first()["candidate_id"]
        for _ in range(4):
            _, parent = self.next_candidate(parent)
        trace = self.evidence("kernels-c3.json")
        weakest = [{"kpi": "prefill", "term": "gated delta 927 ms vs 13 ms floor", "evidence": trace,
                    "levers": ["chunked-prefill"]},
                   {"kpi": "decode", "term": "verify pass 16.5 vs 13.4 ms floor", "evidence": trace, "levers": ["graphs"]}]
        with self.assertRaisesRegex(ValueError, "Candidate 6 is a critique round: name the parent's two weakest terms"):
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
        with self.assertRaisesRegex(ValueError, "graphs names no provider: inference:, kernel:, hub:, lib:, own"):
            self.store.plan(self.run_id, bare)
        with self.assertRaisesRegex(ValueError, "provider starts with inference:, kernel:, hub:, lib:, own"):
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


class CandidateLadderTests(PlanFixture, unittest.TestCase):
    """The candidate ladder: a candidate's number in the run says how far it goes, from
    candidate 1's build, plugin tests and serve check with no Eval, through candidate 2's small Eval and candidates 3
    and 4's medium one, to the full Eval from candidate 5 on; a draft is fixed from the same plan with its number."""
    ladder_scopes = True
    account = {"kpis": [{"kpi": kpi, "start": 1.0, "reached": 1.0, "limit": 2.0, "built": "fixture", "cause": "fixture"}
                        for kpi in ("decode", "prefill", "ttft")]}

    def setUp(self):
        super().setUp()
        # A run at the default budget: the ladder's five rungs.
        self.create()

    def create(self, **updates):
        data = {key: value for key, value in self.data.items() if key != "candidate_budget"} | updates
        self.run = self.store.create(data)["run"]
        self.run_id = self.run["run_id"]
        self.directory = self.store.directory(self.run_id)

    def served(self, candidate_id, state="passed"):
        """The candidate packaged, with the functional check crit run ... serve --check records."""
        package = self.package({"candidate_id": candidate_id})
        path = self.directory / "candidates" / candidate_id / "logs" / "serve-check-fixture.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checks": []}), encoding="utf-8")
        self.store.append(self.run_id, candidate_id, "check", {
            "case_id": "case-0001", "package_sha256": digest(package), "gate": "functional",
            "criteria_sha256": digest(self.run["spec"]["correctness"]), "status": state,
            "rationale": "crit run serve --check: fixture",
            "evidence": [{"path": path.relative_to(self.directory).as_posix(), "sha256": file_sha(path)}],
            "evidence_level": "artifact_verified", "recorded_at": now(),
            "extensions": {"recorded_by": "crit run serve --check"}})

    def built(self, plan_id, parent=None):
        """A candidate from the plan that passed its serve check and was sealed built, skipping its rung's Eval."""
        candidate = self.candidate(plan_id=plan_id, **({"parent_candidate_id": parent} if parent else {}))["candidate_id"]
        self.served(candidate)
        self.store.seal(self.run_id, candidate, "built")
        return candidate

    def critique(self, parent):
        """A critique round's plan from the parent: its two weakest terms and another lever beside them."""
        path = self.directory / "work" / "kernels.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        weakest = [{"kpi": "prefill", "term": "attention 9 ms vs 3 ms floor", "evidence": "work/kernels.json",
                    "levers": ["chunked-prefill"]},
                   {"kpi": "decode", "term": "verify pass 16 vs 13 ms floor", "evidence": "work/kernels.json",
                    "levers": ["graphs"]}]
        levers = plan()["levers"] + [lever("fused-norms", {"decode": {"change": 0.1, "source": "estimate"}})]
        return plan(parent, levers=levers, weakest=weakest)

    @staticmethod
    def rung(recorded):
        goal = recorded["inputs"]["objective"]
        return goal["candidate"], goal["name"], goal["eval"]["scope"]

    def test_new_runs_climb_the_ladder_at_five_candidates(self):
        from crit.run import records
        self.assertEqual(records.DEFAULT_CANDIDATE_BUDGET, 5)
        self.assertNotIn("budget_counts", self.run["spec"])
        self.assertEqual(self.store.get(self.run_id)["segment"]["limits"]["candidate_budget"], 5)
        prompt = (self.directory / "agent_request.md").read_text(encoding="utf-8")
        self.assertIn("candidate budget: 5 candidates on the candidate ladder", prompt)
        self.assertIn(" next\n", prompt)

    def test_each_candidate_goes_as_far_as_its_rung(self):
        first = self.store.plan(self.run_id, plan(levers=[], forecast=None))
        self.assertEqual(self.rung(first), (1, "proof_of_concept", "none"))
        draft = self.candidate(plan_id=first["plan_id"])["candidate_id"]
        # Candidate 1 has no Eval, and no plan starts from it before it stands on its rung.
        with self.assertRaisesRegex(ValueError, "first rung: no Eval"):
            self.store.reserve(self.run_id, draft, "eval")
        with self.assertRaisesRegex(ValueError, f"{draft} is not on the candidate ladder yet"):
            self.store.plan(self.run_id, plan(draft))
        # Sealed built without a passed serve check it is a draft: its fix keeps its plan and its number.
        self.package({"candidate_id": draft})
        self.store.seal(self.run_id, draft, "built")
        with self.assertRaisesRegex(ValueError, f"{draft} is not on the candidate ladder yet"):
            self.store.plan(self.run_id, plan(draft))
        self.assertEqual(self.rung(self.store.plan(self.run_id, plan(levers=[], forecast=None), check=True)),
                         (1, "proof_of_concept", "none"))
        c1 = self.built(first["plan_id"])
        with self.assertRaisesRegex(ValueError, f"already linked to {c1}, which is on the candidate ladder"):
            self.candidate(plan_id=first["plan_id"])
        with self.assertRaisesRegex(ValueError, "null only while the run has no candidate on the candidate ladder"):
            self.store.plan(self.run_id, plan(levers=[], forecast=None))
        # Candidate 2, the plugin round: its plan freezes no Eval, its Eval is the small one.
        second = self.store.plan(self.run_id, plan(c1))
        self.assertEqual(self.rung(second), (2, "plugin_round", "small"))
        self.assertEqual(second["inputs"]["objective"]["eval"]["stage"], 1)
        self.assertIsNone(second["inputs"]["eval"])
        c2 = self.candidate(plan_id=second["plan_id"], parent_candidate_id=c1)["candidate_id"]
        self.evaluated(c2, status())
        # Candidates 3 and 4: the medium Eval, whether the parent ran its Eval or skipped it.
        third = self.store.plan(self.run_id, plan(c2))
        self.assertEqual(self.rung(third), (3, "crit_round", "medium"))
        self.assertEqual(third["inputs"]["eval"]["path"].split("/")[1], c2)
        c3 = self.built(third["plan_id"], c2)
        fourth = self.store.plan(self.run_id, plan(c3))
        self.assertEqual(self.rung(fourth), (4, "crit_round", "medium"))
        self.assertIsNone(fourth["inputs"]["eval"])
        c4 = self.built(fourth["plan_id"], c3)
        # Candidate 5: the full Eval, which it cannot skip: sealed built without it, it is a draft.
        fifth = self.store.plan(self.run_id, plan(c4))
        self.assertEqual(self.rung(fifth), (5, "crit_round", "full"))
        skipped = self.built(fifth["plan_id"], c4)
        with self.assertRaisesRegex(ValueError, f"{skipped} is not on the candidate ladder yet"):
            self.store.plan(self.run_id, plan(skipped))
        with self.assertRaisesRegex(ValueError, "Only 4 of 5 budgeted candidates are on the candidate ladder"):
            self.store.stop(self.run_id, "budget_exhausted", critique="critique", headroom=self.account)
        c5 = self.candidate(plan_id=fifth["plan_id"], parent_candidate_id=c4)["candidate_id"]
        self.evaluated(c5, status())
        # The budget is used: a new candidate is refused, the stop accounts for it, and a resume adds critique rounds.
        sixth = self.store.plan(self.run_id, self.critique(c5))
        with self.assertRaisesRegex(ValueError, "candidate budget \\(5 candidates on the candidate ladder\\) is used"):
            self.candidate(plan_id=sixth["plan_id"], parent_candidate_id=c5)
        self.store.stop(self.run_id, "budget_exhausted", critique="critique", headroom=self.account)
        self.store.seal(self.run_id, c2, "cancelled", {"message": "fixture"})
        self.store.seal(self.run_id, c5, "cancelled", {"message": "fixture"})
        self.store.resume(self.run_id, attempt_limit=64, external_cleanup_confirmed=True, candidates=1)
        later = self.store.plan(self.run_id, self.critique(c5))
        self.assertEqual(self.rung(later)[1:], ("critique_round", "full"))
        self.assertEqual(self.store.get(self.run_id)["errors"], [])

    def test_the_segments_last_budgeted_candidate_gets_the_full_eval(self):
        self.create(candidate_budget=2)
        first = self.store.plan(self.run_id, plan(levers=[], forecast=None))
        self.assertEqual(self.rung(first)[2], "none")
        c1 = self.built(first["plan_id"])
        self.assertEqual(self.rung(self.store.plan(self.run_id, plan(c1)))[1:], ("plugin_round", "full"))
        # A single candidate is its own proof of concept and the full Eval.
        self.create(candidate_budget=1)
        self.assertEqual(self.rung(self.store.plan(self.run_id, plan(levers=[], forecast=None))),
                         (1, "proof_of_concept", "full"))

    def test_crit_run_eval_measures_as_far_as_the_rung(self):
        from crit.run import evaluation
        from crit.run.ladder import eval_options
        with self.assertRaisesRegex(ValueError, "first rung: no Eval"):
            eval_options("none", None, False, "candidate-0001")
        self.assertEqual(eval_options("small", None, False, "c"), (1, False))
        self.assertEqual(eval_options("medium", None, False, "c"), (2, False))
        self.assertEqual(eval_options("medium", 1, False, "c"), (1, False))
        self.assertEqual(eval_options("full", None, False, "c"), (None, True))
        self.assertEqual(eval_options(None, 3, False, "c"), (3, False))  # outside the ladder, as asked
        for name, stage, certify in (("small", 2, False), ("small", None, True), ("small", "full", False),
                                     ("medium", 3, False), ("medium", None, True)):
            with self.assertRaisesRegex(ValueError, "the full Eval \\(--certify\\) is for candidate 5 on"):
                eval_options(name, stage, certify, "c")
        with self.assertRaisesRegex(ValueError, "drop --stage"):
            eval_options("full", 2, True, "c")
        # crit run eval refuses before it reserves or reads the data.
        first = self.store.plan(self.run_id, plan(levers=[], forecast=None))
        c1 = self.candidate(plan_id=first["plan_id"])["candidate_id"]
        self.served(c1)
        with self.assertRaisesRegex(ValueError, "first rung: no Eval"):
            evaluation.evaluate(self.store, self.run_id, c1, self.root / "no-data")
        self.store.seal(self.run_id, c1, "built")
        second = self.store.plan(self.run_id, plan(c1))
        c2 = self.candidate(plan_id=second["plan_id"], parent_candidate_id=c1)["candidate_id"]
        self.package({"candidate_id": c2})
        with self.assertRaisesRegex(ValueError, f"{c2} gets at most a small Eval"):
            evaluation.evaluate(self.store, self.run_id, c2, self.root / "no-data", certify=True)
        self.assertEqual(list((self.directory / "candidates" / c2 / "attempts").glob("attempt-*")), [])


class ReachTests(PlanFixture, unittest.TestCase):
    """The strategy's ceiling is the limit: a forecast whose best case stays below half of a needed family's limit
    gets a warning unless it names the measured bound that keeps the rest out of reach."""

    def test_a_forecast_far_below_the_limit_names_the_measured_bound(self):
        parent = ObjectiveTests.first(self)["candidate_id"]
        # The fixture's Eval: decode 100 of 200 tok/s, prefill 1,000 of 4,000. A prefill high of 1,800 is 45% of it.
        warned = self.store.plan(self.run_id, plan(parent))["inputs"]["warnings"]
        self.assertTrue(any("the forecast's best case for prefill (1800) is 45% of its limit 4000 (the parent's Eval "
                            "measured 25%), below 50%" in w for w in warned), warned)
        cause = {"kpi": "prefill", "gap": "limit", "answer": "cause", "detail": "experts stream from host memory"}
        self.assertTrue(self.store.plan(self.run_id, plan(parent, gaps=[cause]))["inputs"]["warnings"])  # no measurement
        with self.assertRaisesRegex(ValueError, "is no file of this run"):
            self.store.plan(self.run_id, plan(parent, gaps=[{**cause, "evidence": "work/none.json"}]))
        measured = ObjectiveTests.evidence(self, "host-bound.json")
        recorded = self.store.plan(self.run_id, plan(parent, gaps=[{**cause, "evidence": measured}]))
        reach = recorded["inputs"]["reach"]
        self.assertEqual((reach["prefill"]["limit"], reach["prefill"]["source"], reach["prefill"]["fraction"]),
                         (4000.0, "eval", 0.25))
        self.assertAlmostEqual(reach["prefill"]["reach"], 0.45)
        self.assertEqual(reach["prefill"]["evidence"]["path"], measured)
        self.assertAlmostEqual(reach["decode"]["reach"], 0.75)   # 150 of 200: no cause asked
        self.assertNotIn("evidence", reach["decode"])

    def test_a_forecast_reaching_half_the_limit_needs_no_cause(self):
        parent = ObjectiveTests.first(self)["candidate_id"]
        forecast = {**plan()["forecast"], "prefill": {"low": 1100.0, "expected": 1500.0, "high": 2000.0}}
        recorded = self.store.plan(self.run_id, plan(parent, forecast=forecast))
        self.assertEqual(sorted(recorded["inputs"]["reach"]), ["decode", "prefill"])
        # A family marked not needed is not asked.
        self.assertNotIn("prefill", self.store.plan(self.run_id, plan(parent, prefill=False, gaps=[
            {"kpi": "prefill", "gap": "limit", "answer": "cause", "detail": "out of reach",
             "evidence": ObjectiveTests.evidence(self, "prefill.json")}], levers=[lever()],
            forecast={**plan()["forecast"], "prefill": {"low": 1000.0, "expected": 1000.0, "high": 1000.0}}))
            ["inputs"]["reach"])


class PromptActivationTests(PlanFixture, unittest.TestCase):
    """A speed run's build plans declare prompt_activations, no quantized
    format starts at BF16 without a measured rejection of the stored format, and a candidate serves what its plan
    declared."""
    quality_mode = "speed"
    stored = {"nvfp4": "stored", "fp8": "fp8"}

    def evidence(self, name, value):
        return MeasuredRejectionTests.evidence(self, name, value)

    def bf16_rejection(self, p05=1.1, activations="stored", mode="speed", budget=None):
        sets = {"long-4096/prompt": {"budget_used": 1.3, "budget_spread": {"budget_used": {"p05": p05}}}}
        budget = budget or (SPEED_BUDGET if mode == "speed" else PRESERVE_BUDGET)
        path = self.evidence(f"stored-{p05}-{activations}-{mode}-{budget['kld_mean_max']}.json",
                             {"engine": {"prompt_activations": activations}, "quality": {"sets": sets},
                              "requirements": {"quality_mode": mode, "quality_budget": budget}})
        return {"kpi": "prefill", "gap": "activation:stored", "answer": "rejected", "evidence": path,
                "detail": "NVFP4 prompt activations on the experts"}

    def test_candidate_1_starts_at_the_checkpoints_activation_format(self):
        first = plan(levers=[], forecast=None)
        with self.assertRaisesRegex(ValueError, "build plan declares prompt_activations"):
            self.store.plan(self.run_id, first)
        for exact in ("bf16", {"nvfp4": "bf16", "fp8": "fp8"}, {"nvfp4": {"stored": 0.5, "bf16": 0.5}}):
            with self.subTest(exact=exact):
                with self.assertRaisesRegex(ValueError, "at bf16: in speed, a quantized "
                                                        "format's prompt GEMMs start at its stored format's "
                                                        "activations .*crit eval run --package P"):
                    self.store.plan(self.run_id, {**first, "prompt_activations": exact})
        with self.assertRaisesRegex(ValueError, "prompt_activations is what the planned engine declares"):
            self.store.plan(self.run_id, {**first, "prompt_activations": {"nvfp4": "int4"}})
        recorded = self.store.plan(self.run_id, {**first, "prompt_activations": self.stored})
        self.assertEqual(recorded["prompt_activations"], self.stored)
        self.assertNotIn("activation_evidence", recorded["inputs"])

    def test_bf16_stands_on_a_measured_rejection_of_the_stored_format(self):
        first = {**plan(levers=[], forecast=None), "prompt_activations": {"nvfp4": "bf16", "fp8": "bf16"}}
        with self.assertRaisesRegex(ValueError, "inside the quality noise, so not a rejection"):
            self.store.plan(self.run_id, {**first, "gaps": [self.bf16_rejection(p05=0.9)]})
        with self.assertRaisesRegex(ValueError, "measured prompt activations 'fp8', not 'stored'"):
            self.store.plan(self.run_id, {**first, "gaps": [self.bf16_rejection(activations="fp8")]})
        # Judged by preserve's budget, which rejects what the run's speed mode admits: no rejection here; nor is
        # one judged in the run's mode by a tighter budget than the run's.
        for other in (self.bf16_rejection(mode="preserve"),
                      self.bf16_rejection(budget={**SPEED_BUDGET, "kld_p99_max": 0.5})):
            with self.assertRaisesRegex(ValueError, "this run's terms are quality mode speed with the budget"):
                self.store.plan(self.run_id, {**first, "gaps": [other]})
        rejection = self.bf16_rejection()
        recorded = self.store.plan(self.run_id, {**first, "gaps": [rejection]})
        self.assertEqual(recorded["inputs"]["activation_evidence"]["path"], rejection["evidence"])
        self.assertEqual(recorded["inputs"]["activation_evidence"]["over_budget_sets"], ["long-4096/prompt"])

    def test_a_candidate_serves_what_its_plan_declared(self):
        recorded = self.store.plan(self.run_id, {**plan(levers=[], forecast=None), "prompt_activations": self.stored})
        parent = self.candidate(plan_id=recorded["plan_id"])["candidate_id"]
        self.evaluated(parent, status(), engine={"prompt_activations": {"nvfp4": "bf16", "fp8": "bf16"}})
        forecast = {**plan()["forecast"], "prefill": {"low": 1100.0, "expected": 1500.0, "high": 2000.0}}
        follow = {**plan(parent, forecast=forecast), "prompt_activations": self.stored}
        with self.assertRaisesRegex(ValueError, f"{parent} served prompt activations .*bf16.*, its plan plan-0001 "
                                                "declared .*stored"):
            self.store.plan(self.run_id, follow)
        self.evaluated(parent, status(), engine={"prompt_activations": self.stored})
        self.assertEqual(self.store.plan(self.run_id, follow)["prompt_activations"], self.stored)

    def test_one_planned_activation_holds_for_every_served_format(self):
        recorded = self.store.plan(self.run_id, {**plan(levers=[], forecast=None), "prompt_activations": "stored"})
        parent = self.candidate(plan_id=recorded["plan_id"])["candidate_id"]
        self.evaluated(parent, status(), engine={"prompt_activations": {"nvfp4": "stored", "fp8": "stored"}})
        forecast = {**plan()["forecast"], "prefill": {"low": 1100.0, "expected": 1500.0, "high": 2000.0}}
        follow = {**plan(parent, forecast=forecast), "prompt_activations": "stored"}
        self.assertEqual(self.store.plan(self.run_id, follow)["prompt_activations"], "stored")


class PromptActivationDefaultModeTests(PromptActivationTests):
    """A request that names no quality mode is in speed: check 16 asks its plans."""
    quality_mode = None

    def test_the_run_records_speed(self):
        self.assertEqual(self.run["spec"]["correctness"]["quality_mode"], "speed")


class PromptActivationPreserveTests(PromptActivationTests):
    """Check 16 asks only speed runs."""
    quality_mode = "preserve"

    def test_a_preserve_run_is_not_asked(self):
        recorded = self.store.plan(self.run_id, {**plan(levers=[], forecast=None), "prompt_activations": "bf16"})
        self.assertEqual(recorded["prompt_activations"], "bf16")
        self.assertEqual(self.store.plan(self.run_id, plan(levers=[], forecast=None))["plan_id"], "plan-0002")


    test_candidate_1_starts_at_the_checkpoints_activation_format = None
    test_bf16_stands_on_a_measured_rejection_of_the_stored_format = None
    test_a_candidate_serves_what_its_plan_declared = None


class RoundAccountTests(PlanFixture, unittest.TestCase):
    """Unknown noise takes its own family's, a plan freezes the newest Eval
    attempt that measured speed, a limit claim rests on a round account, and a rejection bound by a term reopens once
    that term shrinks."""
    reaching = {**plan()["forecast"], "prefill": {"low": 1100.0, "expected": 1500.0, "high": 2000.0}}

    def evidence(self, name):
        return ObjectiveTests.evidence(self, name)

    def account(self, terms, idle=0.0, total=None, name="round.json"):
        terms = [{"term": term, "ms": ms, "floor_ms": floor} for term, ms, floor in terms]
        total = total if total is not None else sum(t["ms"] for t in terms) + idle
        return {"total_ms": total, "idle_ms": idle, "terms": terms, "evidence": self.evidence(name)}

    def limited(self, parent, cause_terms, account, **updates):
        """A plan marking prefill not needed, its limit claim resting on the prefill round account."""
        gap = {"kpi": "prefill", "gap": "limit", "answer": "cause", "detail": "copies bound the chunk",
               "evidence": self.evidence("bound.json"), **({"cause_terms": cause_terms} if cause_terms else {})}
        return plan(parent, prefill=False, gaps=[gap], levers=[lever()],
                    forecast={**plan()["forecast"], "prefill": {"low": 1000.0, "expected": 1000.0, "high": 1000.0}},
                    round={"prefill": account} if account else None, **updates)

    def test_a_check_repair_skips_the_speed_checks(self):
        # One run's candidate 2 passed every gate of its Eval and failed crit run's resources check. Its child, planned
        # only to repair that, was refused both ways: as a gate repair and as a speed plan.
        parent, attempt = ObjectiveTests.packaged_parent(self)
        for gate, state in (("arithmetic", "passed"), ("functional", "passed"), ("resources", "failed")):
            ObjectiveTests.check(self, parent, attempt, gate, state)
        repair, same = ObjectiveTests.CHECK_REPAIR, ObjectiveTests.SAME
        with self.assertRaisesRegex(ValueError, "did not fail: resources \\(failed: none\\)\\. resources is a check of "
                                                "crit run, not a gate of Eval's status"):
            self.store.plan(self.run_id, plan(parent, forecast=same, levers=[lever(), {**repair, "serves": "gate:resources"}]))
        with self.assertRaisesRegex(ValueError, "prefill is p1 but no lever improves it"):
            self.store.plan(self.run_id, plan(parent, forecast=same,
                                              levers=[lever(), {**repair, "serves": "resources: VRAM headroom"}]))
        recorded = self.store.plan(self.run_id, plan(parent, forecast=same, levers=[lever(), repair]))
        self.assertEqual(recorded["inputs"]["objective"]["check_repairs"]["resources"][0]["check_id"], "check-0003")
        self.assertNotIn("reach", recorded["inputs"])   # check 15 does not apply to it either

    def test_an_unknown_noise_takes_its_own_familys(self):
        # Candidate 13 of one run: decode's noise unknown, a 151% first-token spread. gave decode 151%.
        facts = PriorityTests.noisy(PriorityTests(), 0.5, 0.6, {"ttft": 1.51, "prefill": 0.04})
        self.assertEqual(plans.noise_default(facts), (1.51, "ttft"))
        self.assertEqual(plans.noise_default(facts, "decode"), (0.05, "default"))
        self.assertEqual(plans.noise_default(facts, "context_ttft"), (0.04, "prefill"))
        self.assertEqual(plans.noises(facts)["decode"], 0.05)
        self.assertEqual(plans.noises(facts)["ttft"], 1.51)
        parent = ObjectiveTests.first(self)["candidate_id"]
        recorded = self.store.plan(self.run_id, plan(parent, forecast=self.reaching))
        # The fixture measures no long context: those two take their families' measured noise.
        self.assertEqual(recorded["inputs"]["noise_default"], {"context_decode": {"noise": 0.02, "source": "decode"},
                                                                "context_ttft": {"noise": 0.02, "source": "prefill"}})

    def test_a_plan_freezes_the_newest_attempt_that_measured_speed(self):
        parent = ObjectiveTests.first(self)["candidate_id"]
        partial = status()
        partial["kpis"]["decode"]["score"] = partial["kpis"]["prefill"]["score"] = None
        speed = self.directory / "candidates" / parent / "eval"
        newest = self.evaluated(parent, partial)   # a quality-only rerun
        recorded = self.store.plan(self.run_id, plan(parent, forecast=self.reaching))
        self.assertNotIn(newest, recorded["inputs"]["eval"]["path"])
        self.assertTrue((speed.parent.parent.parent / recorded["inputs"]["eval"]["path"]).is_file())
        self.assertEqual(recorded["inputs"]["kpis"]["decode"]["score"], 100.0)

    def test_a_limit_claim_rests_on_a_round_account(self):
        parent = ObjectiveTests.first(self)["candidate_id"]
        with self.assertRaisesRegex(ValueError, "give the parent's round account"):
            self.store.plan(self.run_id, self.limited(parent, None, None))
        with self.assertRaisesRegex(ValueError, "add up to 50 ms, the measured round is 60 ms"):
            self.store.plan(self.run_id, self.limited(parent, ["expert copies"], self.account(
                [("expert copies", 28.0, 26.0), ("dense", 22.0, 20.0)], total=60.0)))
        with self.assertRaisesRegex(ValueError, "cause_terms gpu copies are no terms"):
            self.store.plan(self.run_id, self.limited(parent, ["gpu copies"], self.account(
                [("expert copies", 28.0, 26.0)])))
        # Dense at 27% of its floor is not at its floor: candidate 14's stop record, as a round account.
        slow = self.account([("expert copies", 28.0, 26.0), ("dense", 10.4, 2.8), ("rest", 13.3, 12.0)], idle=8.3)
        with self.assertRaisesRegex(ValueError, "dense takes 10.4 of 60 ms at 27% of its floor"):
            self.store.plan(self.run_id, self.limited(parent, ["expert copies"], slow))
        with self.assertRaisesRegex(ValueError, "the device idles 8.3 of 60 ms"):
            self.store.plan(self.run_id, self.limited(parent, ["expert copies", "dense"], slow))
        recorded = self.store.plan(self.run_id, self.limited(parent, ["expert copies", "dense", "idle"], slow))
        self.assertAlmostEqual(recorded["inputs"]["round"]["prefill"]["parts_ms"], 60.0)
        # A small term may trail its floor: dense at 5 of 60 ms is below the 10% that matters.
        small = self.account([("expert copies", 50.0, 48.0), ("dense", 5.0, 1.0), ("rest", 5.0, 4.0)])
        self.store.plan(self.run_id, self.limited(parent, ["expert copies"], small))

    def test_a_rejection_reopens_once_its_binding_term_shrinks(self):
        parent = ObjectiveTests.first(self)["candidate_id"]
        bound = {"family": "decode", "term": "expert copies"}
        before = self.account([("expert copies", 28.0, 26.0), ("dense", 30.0, 28.0)], name="before.json")
        rejected = [{"name": "cpu-tier", "reason": "measured slower", "bound_by": bound,
                     "evidence": self.evidence("cpu-slower.json")}]
        with self.assertRaisesRegex(ValueError, "rejected cpu-tier is bound by 'link'"):
            self.store.plan(self.run_id, plan(parent, forecast=self.reaching, round={"decode": before},
                                              rejected=[{**rejected[0], "bound_by": {**bound, "term": "link"}}]))
        self.store.plan(self.run_id, plan(parent, forecast=self.reaching, round={"decode": before}, rejected=rejected))
        after = self.account([("expert copies", 12.0, 11.0), ("dense", 30.0, 28.0)], name="after.json")
        with self.assertRaisesRegex(ValueError, "rejected 'cpu-tier' while expert copies took 28 ms of the decode "
                                                "round; it is now 12 ms"):
            self.store.plan(self.run_id, plan(parent, forecast=self.reaching, round={"decode": after}))
        # Rejected again on the same measurement: still refused; on a new one, or planned again: recorded.
        with self.assertRaisesRegex(ValueError, "Plan it again"):
            self.store.plan(self.run_id, plan(parent, forecast=self.reaching, round={"decode": after}, rejected=rejected))
        again = [{**rejected[0], "evidence": self.evidence("cpu-again.json")}]
        recorded = self.store.plan(self.run_id, plan(parent, forecast=self.reaching, round={"decode": after},
                                                     rejected=again))
        self.assertEqual(recorded["inputs"]["reopened"][0]["name"], "cpu-tier")
        planned = plan(parent, forecast=self.reaching, round={"decode": after},
                       levers=plan()["levers"] + [lever("cpu-tier")], rejected=[])
        self.assertIn("cpu-tier", [row["id"] for row in self.store.plan(self.run_id, planned)["levers"]])
        # The same term, barely smaller: the rejection still holds.
        close = self.account([("expert copies", 25.0, 24.0), ("dense", 30.0, 28.0)], name="close.json")
        self.store.plan(self.run_id, plan(parent, forecast=self.reaching, round={"decode": close}))

    def test_candidate_1_starts_with_the_cpu_tier_where_the_cpu_beats_the_link(self):
        recon = self.directory / "work" / "recon.json"
        recon.parent.mkdir(parents=True, exist_ok=True)
        placement = {"scenario": "sparse_device_host", "plain_pass_floor_ms": {"link": 20.0, "cpu": 12.0, "split": 10.0}}
        recon.write_text(json.dumps({"gathered": {"placement": placement}}), encoding="utf-8")
        first = plan(levers=[], forecast=None)
        with self.assertRaisesRegex(ValueError, "candidate 1 starts with the CPU expert tier"):
            self.store.plan(self.run_id, first)
        tier = {**lever("cpu-tier"), "kind": "requirement", "serves": "recon's starting design", "changes": {}}
        recorded = self.store.plan(self.run_id, {**first, "levers": [{**tier, "provider": "inference:host-expert-tier"}]})
        self.assertEqual(recorded["inputs"]["cpu_tier"]["floors_ms"], {"link": 20.0, "cpu": 12.0, "split": 10.0})
        rejected = {"kpi": "decode", "gap": "placement:cpu", "answer": "rejected", "detail": "layers measured slower",
                    "evidence": self.evidence("cpu-layers.json")}
        self.assertIn("rejection", self.store.plan(self.run_id, {**first, "gaps": [rejected]})["inputs"]["cpu_tier"])
        # A placement whose link is the faster source asks nothing.
        placement["plain_pass_floor_ms"] = {"link": 10.0, "cpu": 12.0, "split": 11.0}
        recon.write_text(json.dumps({"gathered": {"placement": placement}}), encoding="utf-8")
        self.assertNotIn("cpu_tier", self.store.plan(self.run_id, first)["inputs"])

class LongContextAndEvidenceTests(unittest.TestCase):
    """A candidate planned for the long-context KPIs alone is a candidate, and a lever change measured before the plan
    names its evidence (measured:PATH)."""

    @staticmethod
    def plan(forecast):
        from types import SimpleNamespace
        facts = {"kpis": {"decode": {"score": 100.0}, "prefill": {"score": 1000.0}, "ttft": {"score": 200.0},
                          "context_decode": {"score": 20.0}, "context_ttft": {"score": 5000.0}}}
        return SimpleNamespace(decision="build", first=False, repairing=False, facts=facts,
                               noise={"decode": 0.02, "prefill": 0.02, "context_decode": 0.02, "context_ttft": 0.02},
                               forecast={kpi: {"low": v, "expected": v, "high": v} for kpi, v in forecast.items()})

    def test_a_candidate_for_long_context_decode_alone_is_planned(self):
        # One run's long-context requests decoded outside its captured rounds at 20 tok/s; the fix changed neither
        # decode nor prefill, and a plan was once refused until it added a speed lever nobody had measured.
        from crit.run.plans import _check_forecast
        flat = {"decode": 100.0, "prefill": 1000.0, "ttft": 200.0, "context_decode": 150.0, "context_ttft": 5000.0}
        _check_forecast(self.plan(flat))
        faster_first_token = dict(flat, context_decode=20.0, context_ttft=4000.0)
        _check_forecast(self.plan(faster_first_token))
        with self.assertRaisesRegex(ValueError, "context_decode nor context_ttft beyond its noise"):
            _check_forecast(self.plan(dict(flat, context_decode=20.0)))

    def test_a_measured_lever_change_names_its_evidence_in_the_run(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from crit.run.plans import _check_lever_sources
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "work").mkdir()
            (Path(root) / "work" / "ab.json").write_text("{}", encoding="utf-8")
            def plan(source):
                return SimpleNamespace(root=Path(root), inputs={}, data={"levers": [
                    {"id": "copies", "changes": {"decode": {"change": -0.09, "source": source}}}]})
            found = plan("measured:work/ab.json")
            _check_lever_sources(found)
            self.assertEqual(found.inputs["lever_evidence"]["copies"]["decode"]["path"], "work/ab.json")
            with self.assertRaisesRegex(ValueError, "no file of this run"):
                _check_lever_sources(plan("measured:work/none.json"))


class RoundAccountSpeedTests(RoundAccountTests):
    """The round account under quality mode speed: every build plan declares prompt_activations and each parent has
    served them, so these plans declare the checkpoint's stored format and their candidates serve it."""
    quality_mode = None

    def setUp(self):
        super().setUp()
        self.assertEqual(self.run["spec"]["correctness"]["quality_mode"], "speed")
        self.undeclared = self.store.plan

        def declaring(run_id, value, *args, **kwargs):
            return self.undeclared(run_id, {"prompt_activations": PromptActivationTests.stored, **value}, *args, **kwargs)
        self.store.plan = declaring

    def evaluated(self, candidate_id, value, completed=True, **result):
        """An Eval attempt whose engine served the prompt activations its plan declared."""
        return super().evaluated(candidate_id, value, completed,
                                 **{"engine": {"prompt_activations": PromptActivationTests.stored}, **result})

    def test_check_16_asks_a_plan_that_declares_no_prompt_activations(self):
        with self.assertRaisesRegex(ValueError, "build plan declares prompt_activations"):
            self.undeclared(self.run_id, plan(levers=[], forecast=None))


class ReconMeasuredRejectionTests(PlanFixture, unittest.TestCase):
    """With a model folder, recon's plans are written before the first candidate."""
    finish_recon = False
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
    def test_eval_states_each_kpis_fraction_of_its_limit_and_largest_gaps(self):
        value = status(decode=22.0, gaps=(("decode", "model pass at its bandwidth floor", 0.7),
                                          ("decode", "draft steps at their bandwidth floor", 0.01),
                                          ("decode", "verify pass at the ordinary pass cost", 0.65)))
        value["kpis"]["decode"]["pass_efficiency"] = 0.16
        lines = plans.standing(value)
        self.assertEqual(lines[0], "decode 22.0 tok/s is 11% of its limit 200.0 tok/s; pass at 16% of its floor; "
                                   "largest gaps: model pass at its bandwidth floor +70%, "
                                   "verify pass at the ordinary pass cost +65%")
        self.assertEqual(lines[1], "prefill 1,000.0 tok/s is 25% of its limit 4,000.0 tok/s")
        self.assertEqual(lines[2], "ttft 200.0 ms is 50% of its floor 100.0 ms")
        self.assertEqual(len(plans.standing(limitless_decode(value))), 2)   # no limit, no line

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

    def test_rank_by_the_largest_noise_of_each_family(self):
        # Long-context first token, of unknown noise, moves from above prefill's 0.65 to below it: prefill stays in p1,
        # since its noise is the larger of its members' (context_ttft's unknown takes its family's 8%).
        noise = {"decode": 0.01, "prefill": 0.08, "context_decode": 0.01}
        before, after = self.noisy(0.60, 0.65, noise, (0.99, 0.67)), self.noisy(0.60, 0.65, noise, (0.99, 0.62))
        self.assertEqual(plans.family(after, "prefill"), (0.62, 0.08))
        for facts in (before, after):
            self.assertEqual(plans.priorities(self.needed(), facts)[0]["p1"], ["decode", "prefill"])

    def test_give_unknown_noise_a_default(self):
        # Cleanroom candidate-0030: decode 0.531 (1.5% noise), prefill 0.542 of unknown noise, ttft 4.6%.
        facts = self.noisy(0.5312, 0.5421, {"decode": 0.0154, "ttft": 0.0456})
        # Prefill's family knows no noise: 5%, never time to first token's.
        self.assertEqual(plans.noise_default(facts, "prefill"), (0.05, "default"))
        # √((0.0154 × 0.531)² + (0.05 × 0.542)²) = 0.028 ≥ 0.011: a near tie.
        self.assertEqual(plans.priorities(self.needed(), facts)[0]["p1"], ["decode", "prefill"])
        # With no noise known at all, 5% each: a near tie on either side of 0.5.
        for decode in (0.495, 0.505):
            facts = self.noisy(decode, 0.5, {})
            self.assertEqual(plans.noise_default(facts), (0.05, "default"))
            self.assertEqual(plans.priorities(self.needed(), facts)[0]["p1"], ["decode", "prefill"])

    def test_rank_one_family_when_both_are_near_their_limits(self):
        self.assertEqual(plans.NEAR_LIMIT, 0.85)
        noise = {kpi: 0.03 for kpi in plans.KPIS}
        for decode, prefill, first in ((0.90, 0.92, "decode"), (0.92, 0.90, "prefill"), (0.90, 0.90, "decode")):
            with self.subTest(decode=decode, prefill=prefill):
                facts = self.noisy(decode, prefill, noise)
                self.assertEqual(plans.priorities(self.needed(), facts)[0],
                                 {"p1": [first], "p2": ["prefill" if first == "decode" else "decode"], "p3": ["ttft"]})
        # Below NEAR_LIMIT (one of them at 0.84) the near tie still puts both in p1.
        self.assertEqual(plans.priorities(self.needed(), self.noisy(0.84, 0.86, noise))[0]["p1"], ["decode", "prefill"])

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
