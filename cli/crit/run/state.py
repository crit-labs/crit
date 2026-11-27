"""The state of a run as one explicit model, and the one next step it allows (crit run ... next).

A run's records are files; this module reads them once and names where the run stands, so an agent never has to
infer the order of the commands from prose or from refusals. Each candidate is in exactly one of CANDIDATE_STATES,
the run in exactly one of RUN_STEPS, and next_step() turns them into the command to run now and why.

Candidate states, in the order a candidate moves through them (docs/RUN_RECORDS.md#the-candidate-ladder):

| State | Means | Next |
| --- | --- | --- |
| created | source retained, no build yet | build |
| build_failed | its newest build attempt failed | seal build_failed, fix as a new candidate from the same plan |
| packaged | built and packaged | serve --check |
| check_failed | its newest serve --check failed | seal check_failed, fix as a new candidate from the same plan |
| serve_checked | serve --check passed | its rung: seal built (rung 1), eval (or seal built, rungs 2 to 4), eval (full) |
| eval_stopped | its newest Eval attempt stopped before the end | eval --resume |
| evaluated | a completed Eval | check --from-eval, measure, seal completed; seal built where its checks keep its benchmarks exploratory (measure refuses it) |
| running | an attempt is running | wait for it (or settle it when its process is gone) |
| sealed | closed with a lifecycle | nothing: the run plans from it |

With the run's rung unreachable (crit.run.ladder.unreachable: Eval refused the run's model), a serve_checked candidate
on any rung is sealed built, and the hand-off says that no candidate is certified and why.

A sealed candidate is a draft when it is not on the candidate ladder (crit.run.ladder.counted); its plan builds the
fix.

Which operations may start in each state is one table (OPERATIONS) that next_step(), RunStore.reserve and seal read:
the docs' table of the run's states and this module's agree with it, and tests/test_run_state.py walks every state.

A step named here is one its command takes: where a command judges a candidate by a rule of its own (measure's
qualification, seal's measured cases), the step reads that rule (measure.unqualified), so no state is left without
a legal action. A build's c2 was told to measure, which refused (its checks keep its benchmarks exploratory), and
seal completed refused without a measurement.
"""
from . import ladder, plans, recon
from .replay import MARKER as REPLAY_MARKER
from .records import load_run, read_json, safe_path, segment, timestamp
from ..records import utc_now

CANDIDATE_STATES = ("created", "build_failed", "packaged", "check_failed", "serve_checked", "eval_stopped",
                    "evaluated", "running", "sealed")
RUN_STEPS = ("resume", "preflight", "recon", "finish_recon", "capture", "resolve", "plan", "candidate", "build", "serve_check",
             "seal", "eval", "eval_resume", "record_checks", "measure", "wait", "stop", "hand_off", "done")
FROM_EVAL = "crit run check --from-eval"

# The one transition table of a candidate: the operations a command takes on it (an attempt it reserves, or a seal)
# and the states each may start in. Every command that reserves an attempt asks it (RunStore.reserve: refusal()), and
# next_step() names only commands it allows, so no state is left without a legal action and no command starts where
# the table refuses it. An operation's other states refuse it with WHY.
PACKAGED = ("packaged", "check_failed", "serve_checked", "eval_stopped", "evaluated")
OPERATIONS = {
    "build": ("created", "build_failed"),
    "check": PACKAGED,       # serve (--check), ablate, profile, record-routes, switch-check
    # Eval only after a passed serve --check (and its resume, or another Eval of an evaluated candidate); measure only
    # on a completed Eval (measure's own qualification judges the rest: measure.unqualified). The phase order is
    # the table's, so RunStore.reserve refuses an Eval or a measurement out of it for every caller.
    "eval": ("serve_checked", "eval_stopped", "evaluated"),
    "benchmark": ("evaluated",),
    "seal": ("created", "build_failed") + PACKAGED,
}
WHY = {
    "created": "it is not built: crit run ... build --candidate {cid} first",
    "build_failed": "its newest build failed: seal it build_failed and build the fix as a new candidate",
    "packaged": "its serve --check has not passed: crit run ... serve --candidate {cid} --check first",
    "check_failed": "its newest serve --check failed: seal it check_failed and build the fix as a new candidate",
    "serve_checked": "it has no completed Eval: crit run ... eval --candidate {cid} first",
    "eval_stopped": "its Eval stopped before the end: crit run ... eval --candidate {cid} --resume",
    "evaluated": "its Eval is complete: record its checks (check --from-eval), measure it or seal it",
    "running": "an attempt of it is running ({detail}): wait for it, or settle it with its real outcome when its "
               "process is gone (crit run ... settle)",
    "sealed": "it is sealed ({detail}): a sealed candidate is immutable; allocate a new candidate",
}
# The operation each of next_step()'s steps takes, where it takes one (the others are records or the agent's work).
STEP_OPERATION = {"build": "build", "serve_check": "check", "eval": "eval", "eval_resume": "eval",
                  "measure": "benchmark", "seal": "seal"}


# The states no operation starts in, whoever asks (refusal(strict=False): seal's own check, which judges a seal's
# lifecycle by rules of its own); RunStore.reserve asks the whole table.
NEVER = ("running", "sealed")


def refusal(operation, row, strict=True):
    """None where the table allows `operation` in the candidate's state (`row`: candidate_state's), else why not.
    strict False: only the states no operation starts in (NEVER)."""
    if operation not in OPERATIONS or row["state"] in OPERATIONS[operation]:
        return None
    if not strict and row["state"] not in NEVER:
        return None
    return (f"{row['candidate']} is {row['state']}, where crit run's {operation} does not start: "
            + WHY[row["state"]].format(cid=row["candidate"], detail=row.get("detail")))
# A sealed candidate takes no attempt, so what the next plan wants measured on this engine is measured before its seal.
SWEEP_FIRST = ("A sealed candidate takes no attempt: sweep the options the next plan needs measured on this engine "
               "(ablate --candidate {cid} --set ...) before its seal")


def replay_of(root):
    """Whether the run folder is a replay (crit.run.replay), which records nothing, a preflight neither."""
    return (root / REPLAY_MARKER).is_file()


def _attempts(folder):
    """The candidate's attempts in order: (attempt_id, operation, outcome or None while it runs)."""
    found = []
    for path in sorted(folder.glob("attempts/attempt-*/reservation.json")):
        result = path.with_name("result.json")
        outcome = read_json(result).get("outcome") if result.is_file() else None
        found.append((path.parent.name, read_json(path).get("operation"), outcome))
    return found


def _checks(folder, recorded_by):
    """The candidate's check records that `recorded_by` wrote, oldest first."""
    found = []
    for path in sorted(folder.glob("checks/check-*.json")):
        value = read_json(path)
        if (value.get("extensions") or {}).get("recorded_by") == recorded_by:
            found.append(value)
    return found


def candidate_state(root, run, candidate_id):
    """Where one candidate stands: {candidate, state, scope, on_ladder, plan_id, detail, allowed}: allowed, the
    operations the transition table lets start in its state (OPERATIONS), which next and show print."""
    found = _candidate_state(root, run, candidate_id)
    return {**found, "allowed": [op for op, states in OPERATIONS.items() if found["state"] in states]}


def _candidate_state(root, run, candidate_id):
    folder = safe_path(root, f"candidates/{candidate_id}")
    sealed = ladder.sealed_lifecycle(root, candidate_id)
    scope = ladder.scope_of(root, candidate_id)
    row = {"candidate": candidate_id, "scope": scope, "on_ladder": candidate_id in ladder.counted(root),
           "plan_id": read_json(folder / "candidate.json").get("plan_id"), "detail": None}
    if sealed is not None:
        return row | {"state": "sealed", "detail": sealed}
    attempts = _attempts(folder)
    running = [a for a in attempts if a[2] is None]
    if running:
        return row | {"state": "running", "detail": f"{running[-1][0]} ({running[-1][1]})"}
    builds = [a for a in attempts if a[1] == "build"]
    packaged = (folder / "package" / "manifest.json").is_file()
    if not packaged:
        failed = builds and builds[-1][2] != "completed"
        return row | {"state": "build_failed" if failed else "created",
                      "detail": f"{builds[-1][0]} {builds[-1][2]}" if failed else None}
    serve = _checks(folder, ladder.SERVE_CHECK)
    if not serve:
        return row | {"state": "packaged"}
    if serve[-1].get("status") != "passed":
        return row | {"state": "check_failed", "detail": serve[-1].get("rationale")}
    evals = [a for a in attempts if a[1] == "eval"]
    if not evals:
        return row | {"state": "serve_checked"}
    if not plans.completed_evals(root, candidate_id) or evals[-1][2] != "completed":
        return row | {"state": "eval_stopped", "detail": f"{evals[-1][0]} {evals[-1][2]}"}
    return row | {"state": "evaluated"}


def _measured(folder, run):
    """The run's cases without a benchmark of the candidate."""
    telemetry = folder / "telemetry.json"
    series = read_json(telemetry).get("series") or [] if telemetry.is_file() else []
    done = {s.get("case_id") for s in series}
    return [c["case_id"] for c in run["spec"]["cases"] if c["case_id"] not in done]


def next_step(store, run_id):
    """The one next step of the run: {step, command, why, candidate, state}. `command` is the crit command to run
    (the run's global options included), or None where the step is the agent's own work (finishing recon.md, writing
    source, a plan)."""
    root = store.directory(run_id)
    run = load_run(root)
    state = read_json(safe_path(root, "state.json"))
    bounds = segment(root, run, state["current_segment_id"])
    base = f'crit run --project "{store.project}" --root "{store.root}" --id {run_id}'
    candidates = sorted(p.name for p in safe_path(root, "candidates").glob("candidate-*"))
    rows = [candidate_state(root, run, c) for c in candidates]

    from ..preflight.records import brief
    preflight = brief(root, run["spec"]["model_path"]) if not replay_of(root) else None

    def step(name, command, why, candidate=None):
        return {"step": name, "command": command and f"{base} {command}", "why": why, "candidate": candidate,
                "state": {"segment": bounds["segment_id"], "candidates": rows, "preflight": preflight}}

    uncertified = ladder.uncertified(root)
    if state["stopping_reason"] is not None:
        if not (root / "summary.md").is_file():
            late = plans.retroactive(root, run)
            return step("hand_off", None, f"The run stopped ({state['stopping_reason']}). Write summary.md, record the "
                        "cost (crit run ... cost --transcript ...), run crit serve --write-launchers for the selected "
                        "candidate and hand off (AGENTS.md section 5)"
                        + (f". Retroactive plans, recorded after their change was measured: {', '.join(late)}; "
                           "summary.md names them" if late else "")
                        + (f". {uncertified}: summary.md says so" if uncertified else ""))
        return step("done", None, f"The run stopped ({state['stopping_reason']}) and summary.md is written. Resume "
                    "with crit run ... resume --candidates N only when the user asks for more"
                    + (f". {uncertified}" if uncertified else ""))
    if utc_now() >= timestamp(bounds["deadline_at"], "deadline"):
        return step("resume", "resume --candidates N --external-agent-stopped",
                    "The segment's safety window has passed: only the user authorizes a new segment")
    if preflight is not None and not preflight["current"] and not rows:
        # Before the first candidate: the library at its defaults on the run's own checkpoint, so the toolkit's bugs
        # this checkpoint's geometry exposes show up in minutes, not in the build's candidates.
        return step("preflight", "preflight", "Run the preflight first: the library's plugins at their defaults on a "
                    "cut of the run's checkpoint, sized from the run, in minutes (no report for this crit and "
                    "checkpoint yet" + (f"; the newest is {preflight['key']}" if preflight["report"] else "") + "). "
                    "A failed scenario is toolkit feedback, not a gate: the build may go on")
    from .store import described
    if not (root / recon.RECON_JSON).is_file() and described(run["spec"]["model_path"]):
        return step("recon", "recon", "Study the model before the first candidate: recon writes work/recon.json and "
                    "scaffolds work/recon.md")
    unfinished = recon.unfinished(root)
    if unfinished:
        return step("finish_recon", None, f"Finish {recon.RECON_MD}: " + ", ".join(unfinished) + " (still TODO or "
                    "missing), from the model card and the transformers reference")
    if not (root / "inputs.json").is_file():
        missing = [c["case_id"] for c in run["spec"]["cases"]
                   if not (root / "captured" / c["case_id"] / "request.json").is_file()]
        if missing:
            return step("capture", f"capture --case {missing[0]} --inputs inputs.json",
                        f"Capture each case's chat request first ({', '.join(missing)})")
        return step("resolve", "resolve --record inputs.json", "Freeze the model and input identities")
    open_rows = [r for r in rows if r["state"] != "sealed"]
    if open_rows:
        return _candidate_step(step, root, run, open_rows[-1])
    return _plan_step(step, root, run, bounds)


def _candidate_step(step, root, run, row):
    """The next step of the run's open candidate."""
    cid, name = row["candidate"], row["state"]
    if name == "running":
        return step("wait", None, f"{row['detail']} is running: wait for it. If its process is gone, settle it "
                    "(crit run ... settle) with its real outcome", cid)
    if name == "created":
        return step("build", f"build --candidate {cid} --record recipe.json", "Build and package the retained source",
                    cid)
    if name in ("build_failed", "check_failed"):
        return step("seal", f'seal --candidate {cid} --lifecycle {name} --failure "WHAT STOPPED IT"',
                    f"{cid} is a draft ({row['detail']}): seal it with what stopped it, then build the fix as a new "
                    f"candidate from the same plan ({row['plan_id']}), at no cost to the budget", cid)
    if name == "packaged":
        return step("serve_check", f"serve --candidate {cid} --check --data EVAL_DATA",
                    "The one check before any Eval or seal: a plugin package's plugin tests, serving checks, crit smoke and Eval's "
                    "pre-check", cid)
    if name == "eval_stopped":
        return step("eval_resume", f"eval --candidate {cid} --data EVAL_DATA --resume",
                    f"Its Eval stopped ({row['detail']}): go on from its journal, never from the start", cid)
    if name == "serve_checked":
        blocked = ladder.unreachable(root)
        if blocked is not None and row["scope"] != "none":
            return step("seal", f"seal --candidate {cid} --lifecycle built",
                        f"The run records {ladder.state_text(blocked)}: seal it built, and the next plan starts from "
                        f"it. {SWEEP_FIRST.format(cid=cid)}", cid)
        if row["scope"] == "none":
            return step("seal", f"seal --candidate {cid} --lifecycle built",
                        "Rung 1 takes no Eval: seal it built, and the next plan starts from it", cid)
        if row["scope"] == "full":
            return step("eval", f"eval --candidate {cid} --data EVAL_DATA --baseline FIRST_RESULT",
                        "Its rung takes the full Eval, which certifies it", cid)
        return step("eval", f"eval --candidate {cid} --data EVAL_DATA --baseline FIRST_RESULT",
                    f"Its rung takes at most a {row['scope']} Eval; or seal it built to skip it "
                    f"(seal --candidate {cid} --lifecycle built). {SWEEP_FIRST.format(cid=cid)}", cid)
    folder = safe_path(root, f"candidates/{cid}")
    if not _checks(folder, FROM_EVAL):
        return step("record_checks", f"check --candidate {cid} --from-eval",
                    "Read crit eval status (with --kernels) for the next plan, and record the run's checks from the "
                    "Eval result", cid)
    from .measure import unqualified
    found = unqualified(root, run, cid)
    if found["lasting"] or found["missing"]:
        # measure refuses it (its benchmarks would stay exploratory) and a completed seal needs its benchmarks: built
        # is the seal it takes, and the next plan repairs what keeps it from qualifying.
        repairs = ("; the next plan repairs it with a requirement lever serving "
                   + ", ".join(f"check:{gate}" for gate in sorted(found["failed"])) if found["failed"] else "")
        return step("seal", f"seal --candidate {cid} --lifecycle built",
                    "Its benchmarks cannot qualify (" + "; ".join(found["lasting"] + found["missing"]) + "): crit run "
                    "measure refuses it and a completed seal needs every case's benchmark. Seal it built" + repairs
                    + f". {SWEEP_FIRST.format(cid=cid)}", cid)
    missing = _measured(folder, run)
    if missing:
        return step("measure", f"measure --candidate {cid}", f"Benchmark the run's cases ({', '.join(missing)})", cid)
    return step("seal", f"seal --candidate {cid} --lifecycle completed", "Its Eval, checks and benchmarks are "
                f"recorded: seal it completed. {SWEEP_FIRST.format(cid=cid)}", cid)


def _plan_step(step, root, run, bounds):
    """With every candidate sealed: a candidate from a ready plan, a new plan, or the stop."""
    counted = ladder.counted(root)
    budget = bounds["limits"].get("candidate_budget")
    used = sum(found == bounds["segment_id"] for found in counted.values())
    summaries = [s for s in plans.summaries(root, run)
                 if s["segment_id"] == bounds["segment_id"] and s["superseded_by"] is None]
    parent = max(counted) if counted else None
    # With the rung unreachable no candidate qualifies, so the stop selects none.
    uncertified = ladder.uncertified(root)
    select = (f"{uncertified}, so select none" if uncertified
              else "select the best eligible candidate (crit run ... select)")
    if summaries and summaries[-1]["decision"] == "stop":
        return step("stop", "stop --record stop.json", f"The newest plan is a stop plan: {select}, then stop with "
                    "search_exhausted")
    ready = [s for s in summaries if s["decision"] == "build" and (s["candidate_id"] is None
                                                                   or s["candidate_id"] not in counted)]
    if ready:
        plan_id = ready[-1]["plan_id"]
        return step("candidate", "candidate --source SOURCE_DIR --record proposal.json",
                    f"{plan_id} is recorded: write its source, name plan_id {plan_id} (parent "
                    f"{ready[-1]['parent_candidate_id']}) and every lever's fate in the proposal")
    if budget is not None and used >= budget:
        return step("stop", "stop --record stop.json", f"The segment's {budget} budgeted candidates are on the ladder: "
                    f"{select}, then stop with budget_exhausted")
    target = f"from {parent}" if parent else "for the first candidate"
    return step("plan", "plan --draft --record plan.json",
                f"Plan the next candidate {target} before any code: plan --draft writes the plan's skeleton with "
                "crit's computed facts; fill it, check it with plan --check, then plan --record")


def render(found):
    """next_step's result as the few lines crit run ... next prints on stderr."""
    lines = [f"next: {found['step']}" + (f" ({found['candidate']})" if found["candidate"] else ""), found["why"]]
    if found["command"]:
        lines.append(found["command"])
    preflight = (found.get("state") or {}).get("preflight")
    if preflight and preflight.get("report"):
        lines.append(f"preflight {preflight['key']}" + ("" if preflight["current"] else " (another crit or checkpoint)")
                     + f": {preflight['passed']} passed, {preflight['failed']} failed, {preflight['skipped']} skipped"
                     + (" (at the library's defaults: " + ", ".join(preflight["failed_scenarios"][:8]) + ")"
                        if preflight["failed"] else "") + f"; {preflight['report']}")
    return "\n".join(lines)

