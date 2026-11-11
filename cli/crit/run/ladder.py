"""The candidate ladder: how far each candidate of a run goes, by its number in the run, and what a segment's candidate
budget counts (docs/RUN_RECORDS.md#the-candidate-ladder).

| Candidate | Built | Eval |
| --- | --- | --- |
| 1 | builds and passes crit run ... serve --check (its plugins' tests included) | none (small for an origin engine) |
| 2 | every plugin crit's library and crit-hub offer that fits the checkpoint and the GPU | at most small (stage 1) |
| 3, 4 | as candidate 2, improved with crit's toolkit | at most medium (stage 2) |
| 5 on | as candidate 4 | full (--certify) |

The segment's last budgeted candidate gets the full Eval whatever its number, so a run with a smaller budget still ends
on a certified candidate. A candidate is on the ladder once an Eval attempt was reserved for it, or, on a rung below the
full Eval, once it is sealed built or completed with a passed crit run ... serve --check; any other candidate is a
draft, fixed as a new candidate from the same plan at no cost and with its number. A run started from a crit-hub engine
(create --from-engine) gives its first candidate, the engine as imported, a small Eval, so the run has a measured
baseline before any change.
"""
from .records import evaluated_candidates, ident, read_json, safe_path

# The ladder's rungs: the default candidate budget of a ladder run.
RUNGS = 5
# Each rung's Eval: none, at most Eval's stage 1, at most stage 2, or the certified Eval.
SCOPES = {
    "none": "no Eval: crit run ... eval is refused. The candidate is sealed built once crit run ... serve --check "
            "passes (it runs the plugins' tests first)",
    "small": "at most a small Eval: crit run ... eval measures Eval's stage 1 (each speed case once, quality on the "
             "short texts, the request history)",
    "medium": "at most a medium Eval: crit run ... eval measures up to Eval's stage 2 (stage 1, then the repetitions, "
              "the follow-up turns, the long quality texts and the long-context curve's first point)",
    "full": "the full Eval: crit run ... eval certifies (stage 3 whatever the budget, one request at the required "
            "context and the lower-context gate), which qualification needs",
}
STAGES = {"small": 1, "medium": 2}
# The lifecycles a candidate below the full Eval is sealed with to stand on the ladder without an Eval.
SEALED = ("built", "completed")
SERVE_CHECK = "crit run serve --check"


def scope(number, last=False, origin=False):
    """The Eval of the run's candidate `number` (1 for the first); `last`: it takes its segment's last budgeted
    candidate, which gets the full Eval; `origin`: it is a crit-hub engine as imported, which gets a small one."""
    if last or number >= RUNGS:
        return "full"
    return ("small" if origin else "none") if number == 1 else "small" if number == 2 else "medium"


def rung(number, last=False, origin=False):
    """What a plan records of its candidate's rung (inputs.objective.eval): the Eval and its rule."""
    name = scope(number, last, origin)
    return {"scope": name, "rule": SCOPES[name], **({"stage": STAGES[name]} if name in STAGES else {})}


def _candidate(root, candidate_id):
    return read_json(safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}/candidate.json"))


def scope_of(root, candidate_id, found=None):
    """The Eval of the candidate's rung, as its plan recorded it; for a candidate without one, by its number among the
    candidates on the ladder before it (`found`: those counted so far, else every one)."""
    plan_id = _candidate(root, candidate_id).get("plan_id")
    if plan_id is not None:
        path = safe_path(root, f"plans/{ident(plan_id, 'plan')}/plan.json")
        recorded = ((read_json(path).get("inputs") or {}).get("objective") or {}).get("eval") if path.is_file() else None
        if recorded:
            return recorded["scope"]
    earlier = [c for c in (counted(root) if found is None else found) if c < candidate_id]
    return scope(len(earlier) + 1)


def served_check(root, candidate_id):
    """Whether crit run ... serve --check passed on the candidate: a functional check it recorded with status passed."""
    folder = safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}")
    for path in sorted(folder.glob("checks/check-*.json")):
        check = read_json(path)
        if (check.get("gate") == "functional" and check.get("status") == "passed"
                and (check.get("extensions") or {}).get("recorded_by") == SERVE_CHECK):
            return True
    return False


def sealed_lifecycle(root, candidate_id):
    """The lifecycle a sealed candidate closed with, or None while it is open."""
    folder = safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}")
    if not (folder / "seal.json").is_file() or not (folder / "outcome.json").is_file():
        return None
    return read_json(folder / "outcome.json").get("lifecycle")


def counted(root):
    """candidate_id -> the segment it counts in, for every candidate on the ladder: those an Eval attempt was reserved
    for (the segment of the first), and those below the full Eval sealed built or completed with a passed serve
    check (the segment that created them)."""
    found = dict(evaluated_candidates(root))
    for folder in sorted(safe_path(root, "candidates").glob("candidate-*")):
        name = folder.name
        if name in found or sealed_lifecycle(root, name) not in SEALED or not served_check(root, name):
            continue
        if scope_of(root, name, found) != "full":
            found[name] = _candidate(root, name).get("segment_id")
    return dict(sorted(found.items()))


def eval_options(name, stage, certify, candidate_id):
    """The stage and certify crit run ... eval measures the candidate with on its rung `name` (None outside the
    ladder: as asked). Refuses what the rung does not allow."""
    if name is None:
        return stage, certify
    if name == "none":
        raise ValueError(f"{candidate_id} is on the ladder's first rung: {SCOPES['none']}")
    if name == "full":
        if stage is not None:
            raise ValueError(f"{candidate_id} gets {SCOPES['full']}: drop --stage")
        return None, True
    cap = STAGES[name]
    if certify or stage == "full" or (stage is not None and stage > cap):
        raise ValueError(f"{candidate_id} gets {SCOPES[name]}: the full Eval (--certify) is for candidate {RUNGS} on "
                         "and the segment's last budgeted candidate")
    return stage or cap, False
