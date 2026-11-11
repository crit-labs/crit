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
| evaluated | a completed Eval | check --from-eval, measure, seal completed |
| running | an attempt is running | wait for it (or settle it when its process is gone) |
| sealed | closed with a lifecycle | nothing: the run plans from it |

A sealed candidate is a draft when it is not on the candidate ladder (crit.run.ladder.counted); its plan builds the
fix.
"""
from . import ladder, plans, recon
from .records import load_run, read_json, safe_path, segment, timestamp
from ..records import utc_now

CANDIDATE_STATES = ("created", "build_failed", "packaged", "check_failed", "serve_checked", "eval_stopped",
                    "evaluated", "running", "sealed")
RUN_STEPS = ("resume", "recon", "finish_recon", "capture", "resolve", "plan", "candidate", "build", "serve_check",
             "seal", "eval", "eval_resume", "record_checks", "measure", "wait", "stop", "hand_off", "done")
FROM_EVAL = "crit run check --from-eval"


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
    """Where one candidate stands: {candidate, state, scope, on_ladder, plan_id, detail}."""
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

    def step(name, command, why, candidate=None):
        return {"step": name, "command": command and f"{base} {command}", "why": why, "candidate": candidate,
                "state": {"segment": bounds["segment_id"], "candidates": rows}}

    if state["stopping_reason"] is not None:
        if not (root / "summary.md").is_file():
            return step("hand_off", None, f"The run stopped ({state['stopping_reason']}). Write summary.md, record the "
                        "cost (crit run ... cost --transcript ...), run crit serve --write-launchers for the selected "
                        "candidate and hand off (AGENTS.md section 5)")
        return step("done", None, f"The run stopped ({state['stopping_reason']}) and summary.md is written. Resume "
                    "with crit run ... resume --candidates N only when the user asks for more")
    if utc_now() >= timestamp(bounds["deadline_at"], "deadline"):
        return step("resume", "resume --candidates N --external-agent-stopped",
                    "The segment's safety window has passed: only the user authorizes a new segment")
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
        if row["scope"] == "none":
            return step("seal", f"seal --candidate {cid} --lifecycle built",
                        "Rung 1 takes no Eval: seal it built, and the next plan starts from it", cid)
        if row["scope"] == "full":
            return step("eval", f"eval --candidate {cid} --data EVAL_DATA --baseline FIRST_RESULT",
                        "Its rung takes the full Eval, which certifies it", cid)
        return step("eval", f"eval --candidate {cid} --data EVAL_DATA --baseline FIRST_RESULT",
                    f"Its rung takes at most a {row['scope']} Eval; or seal it built to skip it "
                    f"(seal --candidate {cid} --lifecycle built)", cid)
    folder = safe_path(root, f"candidates/{cid}")
    if not _checks(folder, FROM_EVAL):
        return step("record_checks", f"check --candidate {cid} --from-eval",
                    "Read crit eval status (with --kernels) for the next plan, and record the run's checks from the "
                    "Eval result", cid)
    missing = _measured(folder, run)
    if missing:
        return step("measure", f"measure --candidate {cid}", f"Benchmark the run's cases ({', '.join(missing)})", cid)
    return step("seal", f"seal --candidate {cid} --lifecycle completed", "Its Eval, checks and benchmarks are "
                "recorded: seal it completed", cid)


def _plan_step(step, root, run, bounds):
    """With every candidate sealed: a candidate from a ready plan, a new plan, or the stop."""
    counted = ladder.counted(root)
    budget = bounds["limits"].get("candidate_budget")
    used = sum(found == bounds["segment_id"] for found in counted.values())
    summaries = [s for s in plans.summaries(root, run)
                 if s["segment_id"] == bounds["segment_id"] and s["superseded_by"] is None]
    parent = max(counted) if counted else None
    if summaries and summaries[-1]["decision"] == "stop":
        return step("stop", "stop --record stop.json", "The newest plan is a stop plan: select the best eligible "
                    "candidate (crit run ... select), then stop with search_exhausted")
    ready = [s for s in summaries if s["decision"] == "build" and (s["candidate_id"] is None
                                                                   or s["candidate_id"] not in counted)]
    if ready:
        plan_id = ready[-1]["plan_id"]
        return step("candidate", "candidate --source SOURCE_DIR --record proposal.json",
                    f"{plan_id} is recorded: write its source, name plan_id {plan_id} (parent "
                    f"{ready[-1]['parent_candidate_id']}) and every lever's fate in the proposal")
    if budget is not None and used >= budget:
        return step("stop", "stop --record stop.json", f"The segment's {budget} budgeted candidates are on the ladder: "
                    "select the best eligible candidate (crit run ... select), then stop with budget_exhausted")
    target = f"from {parent}" if parent else "for the first candidate"
    return step("plan", "plan --draft --record plan.json",
                f"Plan the next candidate {target} before any code: plan --draft writes the plan's skeleton with "
                "crit's computed facts; fill it, check it with plan --check, then plan --record")


def render(found):
    """next_step's result as the few lines crit run ... next prints on stderr."""
    lines = [f"next: {found['step']}" + (f" ({found['candidate']})" if found["candidate"] else ""), found["why"]]
    if found["command"]:
        lines.append(found["command"])
    return "\n".join(lines)

