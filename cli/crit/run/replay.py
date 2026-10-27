"""Replays: a run as it stood when one candidate's Eval finished, for an agent to say what it would do next.

`crit run --root ROOT --id RUN_ID replay --at CANDIDATE --out DIR --count N` copies the run N times into
DIR/replay-1 ... DIR/replay-N (each a run root holding RUN_ID): every file written by the moment the candidate's newest
completed Eval attempt ended (its result and plan outcome included), nothing later. The run's records and candidates
come whole; work/ leaves out build products (binaries and files over MAX_WORK_BYTES). The run's state and unsealed
candidates' projections are rebuilt from the copied records. The Eval status of every copied candidate is rendered again with today's crit-eval
(replay/status/), and plan checks in the replay judge the parent's Eval by it, so a replay shows what an agent decides
from today's reports at that point of the run. `--before PLAN` replays the run as it stood just before that plan was
recorded instead: with the work its agent did after the parent's Eval toward it. Drafts of the next plan (work/ JSON
with a plan's fields, written after the parent's Eval) are always left out.

A replay builds nothing: crit refuses every command that records or changes a record in it (refuse). Its agent reads
replay/agent_request.md, writes the plan it would record next to replay/plan.json, checks it with `plan --check`, and
says why in replay/next.md; `replay --collect` puts the replays' plans side by side. crit starts no agent of its own:
`--agent CMD` runs any harness's command once per replay, all at once (launch).
"""
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from .records import file_sha, ident, load_run, now, read_json, safe_path, timestamp, write_json

MARKER = "replay.json"     # in a replay's run folder: what it replays
FOLDER = "replay"          # the replay's own files: brief, statuses rendered again, the agent's plan and notes
MARGIN = timedelta(seconds=5)   # crit's own records of the Eval command (plan outcome, events) land just after it ends
MAX_WORK_BYTES = 16 << 20
BUILD_SUFFIXES = {".so", ".o", ".a", ".obj", ".lib", ".dll", ".pyd", ".exe", ".cubin", ".fatbin", ".ptx", ".pt",
                  ".bin", ".safetensors", ".gguf", ".whl"}
REBUILT = "state.json"   # the run's projection, rebuilt from the copied records (as unsealed candidates' are)


def refuse(root):
    """Raise when `root` is a replay: it records nothing."""
    path = Path(root) / MARKER
    if path.is_file():
        value = read_json(path)
        raise ValueError(f"This run is a replay of {value['run_id']} at {value['at']} and records nothing: write the plan "
                         f"you would record next to {FOLDER}/plan.json and check it with plan --record "
                         f"{FOLDER}/plan.json --check")


def status_for(root):
    """{Eval result path in the run: its status rendered again} of a replay; {} for any other run."""
    path = Path(root) / MARKER
    if not path.is_file():
        return {}
    statuses = read_json(path).get("statuses") or {}
    return {result: read_json(safe_path(root, entry["status"])) for result, entry in statuses.items()
            if entry.get("status")}


def cutoff(root, candidate_id):
    """When the candidate's newest completed Eval attempt ended (its plan outcome and events within MARGIN), as a
    datetime, with that result's path in the run."""
    from .plans import completed_evals
    found = completed_evals(root, ident(candidate_id, "candidate"))
    if not found:
        raise ValueError(f"{candidate_id} has no completed Eval attempt to replay from")
    result = found[-1]
    attempt = read_json(result.parent.parent / "attempts" / result.stem / "result.json")
    ended = timestamp(attempt["ended_at"], "attempt end")
    outcomes = list((root / "plans").glob(f"plan-*/outcomes/{result.stem}.json"))
    for path in outcomes:
        if read_json(path).get("candidate_id") == candidate_id:
            ended = max(ended, datetime.fromtimestamp(path.stat().st_mtime, timezone.utc))
    return ended + MARGIN, result.relative_to(root).as_posix()


PLAN_FIELDS = {"parent_candidate_id", "decision", "needed", "levers"}


def _draft(path):
    """Whether a work/ file is a plan payload: a JSON object with a plan's fields."""
    if path.suffix.lower() != ".json" or path.stat().st_size > 1 << 20:
        return False
    try:
        value = read_json(path)
    except (ValueError, OSError):
        return False
    return isinstance(value, dict) and PLAN_FIELDS <= set(value)


def _kept(relative, path, limit, drafts_after):
    """Whether a file of the run belongs in a replay at `limit` (a POSIX timestamp); plan drafts written after
    `drafts_after` (the parent's Eval) stay out."""
    parts = relative.split("/")
    if any(part.startswith(".") for part in parts) or relative == REBUILT:
        return False
    stat = path.stat()
    if stat.st_mtime > limit:
        return False
    if parts[0] == "work" and (stat.st_size > MAX_WORK_BYTES or path.suffix.lower() in BUILD_SUFFIXES):
        return False
    if parts[0] == "work" and stat.st_mtime > drafts_after and _draft(path):
        return False
    return True


def snapshot(source, destination, limit, drafts_after):
    """Copy the files of run folder `source` kept at `limit` (a datetime) into `destination`, times preserved; plan
    drafts written after `drafts_after` (a datetime) stay out."""
    stamp, drafts = limit.timestamp(), drafts_after.timestamp()
    copied, left = 0, 0
    for directory, folders, files in os.walk(source):
        folders[:] = [f for f in folders if not f.startswith(".")]
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                left += 1
                continue
            relative = path.relative_to(source).as_posix()
            if not _kept(relative, path, stamp, drafts):
                left += 1
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    for name in ("segments", "events", "candidates", "work", "captured", "plans"):
        (destination / name).mkdir(exist_ok=True)
    return copied, left


def _render(root, statuses, at):
    """Every copied candidate's newest completed Eval status, rendered again with today's crit-eval against the
    baseline its result was measured with: {result path: {status, text, baseline} or {error}}."""
    from crit_eval import EVAL_DIR, load_json
    from crit_eval.status import render, status_of
    from .plans import completed_evals, read_status
    protocol = load_json(EVAL_DIR / "protocol.json")
    newest = {}
    for folder in sorted((root / "candidates").glob("candidate-*")):
        found = completed_evals(root, folder.name)
        if found:
            newest[folder.name] = found[-1]
    by_label = {read_json(path).get("label"): path for path in newest.values()}
    out = root / FOLDER / "status"
    out.mkdir(parents=True, exist_ok=True)
    for candidate_id, path in newest.items():
        relative = path.relative_to(root).as_posix()
        label = (read_status(path).get("baseline") or {}).get("label")
        baseline = by_label.get(label) if label and label != read_json(path).get("label") else None
        try:
            status = status_of(path, protocol, baseline)
        except SystemExit as exc:   # another protocol series: its stored status stands
            statuses[relative] = {"error": str(exc)}
            continue
        write_json(out / f"{candidate_id}.json", status)
        (out / f"{candidate_id}.txt").write_text(render(status) + "\n", encoding="utf-8", newline="\n")
        statuses[relative] = {"status": f"{FOLDER}/status/{candidate_id}.json", "text": f"{FOLDER}/status/{candidate_id}.txt",
                              "baseline": baseline.relative_to(root).as_posix() if baseline else None}
    if not any(path.endswith(f"{at}.txt") for entry in statuses.values() for path in [entry.get("text") or ""]):
        raise ValueError(f"{at}'s Eval status could not be rendered again: "
                         + "; ".join(e["error"] for e in statuses.values() if "error" in e))


def brief(run_id, at, limit, root, project, segment_id, statuses, before=None):
    status = next(entry["text"] for entry in statuses.values() if (entry.get("text") or "").endswith(f"{at}.txt"))
    command = f"crit run --project {project} --root {root.parent} --id {run_id}"
    point = (f"just before {before} was recorded ({limit.isoformat()}), after {at}'s Eval: with the work done since that "
             "Eval toward the next plan, but no draft of the plan itself" if before
             else f"when {at}'s Eval finished ({limit.isoformat()})")
    title = f"before {before}" if before else f"at {at}"
    return f"""# Replay of crit run {run_id} {title}

This folder ({root}) is crit run {run_id} as it stood {point}: its records, its
candidates and their Eval results, and the files of work/ written by then (without build products). Nothing written
later is here. crit and crit-eval are today's versions: the Eval status of every copied candidate is rendered again with
them in {FOLDER}/status/ ({at}: {status}), and crit's plan checks here judge the parent's Eval by those statuses.

You are the run's agent at that point. Follow AGENTS.md (in {project}) and the run's brief
(segments/{segment_id}/agent_request.md) as you would from here, up to recording the next plan, and stop
there: decide what the next candidate should be, or whether the run should stop. A replay builds nothing. Do not create
candidates, package, measure, serve, run Eval, record plans or use the GPU, and do not change crit or crit-eval; crit
refuses every record in a replay. Read the run's records, work/recon.md, plans, candidates, Eval results and work/
evidence, and use read-only commands on them (`crit eval status RESULT` with its options, `{command} show`). Read
nothing else about this run or this model: not the original run folder, other runs, notes or write-ups; only this
folder, crit and crit-eval.

Write two files in {FOLDER}/:

1. `plan.json`: the plan you would record next, in the format `crit run ... plan` records (docs/RUN_RECORDS.md in
   crit), checked with `{command} plan --record {root}/{FOLDER}/plan.json --check` until crit accepts it.
2. `next.md`: what you would do next and why: the levers in the order you would build them, the evidence each rests
   on, and what you would measure first.
"""


def make(store, run_id, at, out, count=1, before=None):
    """`count` replays of run `run_id` under `out` (replay-1 ... replay-N, each a run root): at candidate `at`, or
    just before plan `before` was recorded."""
    source = store.directory(run_id)
    refuse(source)
    if (at is None) == (before is None):
        raise ValueError("A replay starts at a candidate (--at) or before a plan (--before)")
    if before is not None:
        from .plans import load
        plan = load(source, load_run(source), ident(before, "plan"))
        at = plan["parent_candidate_id"]
        if at is None or not (plan.get("inputs") or {}).get("eval"):
            raise ValueError(f"{before} starts from no candidate's Eval: replay a plan that has a parent")
    at = ident(at, "candidate")
    if not (source / "candidates" / at).is_dir():
        raise ValueError(f"{at} is not a candidate of {run_id}")
    if not 1 <= count <= 16:
        raise ValueError("A replay is made 1 to 16 times")
    limit, result = cutoff(source, at)
    drafts_after = limit - MARGIN
    if before is not None:
        # The plan froze the parent's Eval it was checked against; the replay ends where its record was made.
        limit, result = timestamp(plan["created_at"], "plan creation"), plan["inputs"]["eval"]["path"]
    out = Path(out).expanduser().absolute()
    made = []
    for number in range(1, count + 1):
        replay_root = out / f"replay-{number}"
        if replay_root.exists():
            raise ValueError(f"{replay_root} exists: replay into a new folder")
        root = replay_root / run_id
        copied, left = snapshot(source, root, limit, drafts_after)
        replica = type(store)(replay_root, store.project)
        run = load_run(root)
        segments = sorted((root / "segments").glob("segment-*.json"))
        if not segments:
            raise ValueError("The run had no segment by then")
        segment_id = segments[-1].stem
        statuses = {}
        _render(root, statuses, at)
        write_json(root / MARKER, {"schema_version": 1, "record_kind": "replay", "run_id": run_id, "at": at,
                                   "before": before, "cutoff": limit.isoformat(), "result": result,
                                   "result_sha256": file_sha(root / result), "created_at": now(),
                                   "copied_files": copied, "left_out_files": left, "statuses": statuses})
        # Projections from the copied records alone, as the run's coordinator had them at the cutoff; a sealed
        # candidate's is the one its seal names.
        for folder in sorted((root / "candidates").glob("candidate-*")):
            if not (folder / "seal.json").is_file():
                replica._refresh(root, folder.name)
        replica._state(root, current_segment_id=segment_id, phase="awaiting_agent")
        text = brief(run["run_id"], at, limit, root, store.project, segment_id, statuses, before)
        (root / FOLDER / "agent_request.md").write_text(text, encoding="utf-8", newline="\n")
        made.append({"root": str(replay_root), "folder": str(root), "brief": str(root / FOLDER / "agent_request.md"),
                     "copied_files": copied, "left_out_files": left})
    return {"run_id": run_id, "at": at, "before": before, "cutoff": limit.isoformat(), "replays": made}


PLACEHOLDERS = ("brief", "folder", "root", "number")


def split_command(command):
    """A command line as arguments, as the platform's shell splits it: POSIX rules, or on Windows whitespace and quotes
    only, so backslashes in paths stay (POSIX rules read them as escapes)."""
    if os.name != "nt":
        return shlex.split(command)
    return [part[1:-1] if len(part) > 1 and part[0] == part[-1] and part[0] in "\"'" else part
            for part in shlex.split(command, posix=False)]


def launch(made, command):
    """Run `command` (a shell-quoted template; {brief}, {folder}, {root} and {number} name the replay's brief, run folder,
    run root and number) once per replay of `made`, all at once, each in its run folder with CRIT_REPLAY_BRIEF and
    CRIT_REPLAY_FOLDER set and its output in replay/agent.log; wait for all and record each exit in replay/agent.json."""
    template = split_command(command)
    if not template:
        raise ValueError("--agent names the command that starts an agent on a replay")
    running = []
    for number, entry in enumerate(made["replays"], 1):
        values = {"brief": entry["brief"], "folder": entry["folder"], "root": entry["root"], "number": str(number)}
        try:
            argv = [part.format(**values) for part in template]
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"--agent takes the placeholders {', '.join('{' + p + '}' for p in PLACEHOLDERS)} "
                             f"(and {{{{ }}}} for a literal brace): {exc}") from None
        folder = Path(entry["folder"])
        log = (folder / FOLDER / "agent.log").open("wb")
        env = dict(os.environ, CRIT_REPLAY_BRIEF=entry["brief"], CRIT_REPLAY_FOLDER=entry["folder"])
        started = now()
        try:
            process = subprocess.Popen(argv, cwd=folder, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT)
        except OSError as exc:
            log.close()
            raise ValueError(f"--agent could not start {argv[0]}: {exc}") from None
        running.append((entry, folder, process, log, argv, started))
    for entry, folder, process, log, argv, started in running:
        code = process.wait()
        log.close()
        entry["agent"] = {"command": argv, "started_at": started, "ended_at": now(), "exit_code": code,
                          "log": f"{FOLDER}/agent.log"}
        write_json(folder / FOLDER / "agent.json", entry["agent"])
    return made


def collect(out, run_id, store_type, project):
    """Each replay's plan under `out`, checked as `plan --check` checks it: decision, needed flags, priorities, levers
    and forecast; or why it has none."""
    rows = []
    for replay_root in sorted(Path(out).expanduser().absolute().glob("replay-*"),
                              key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0):
        root = replay_root / run_id
        plan_path = root / FOLDER / "plan.json"
        row = {"replay": replay_root.name, "next": (root / FOLDER / "next.md").is_file()}
        if not plan_path.is_file():
            rows.append({**row, "plan": None, "error": "no plan.json"})
            continue
        try:
            value = store_type(replay_root, project).plan(run_id, read_json(plan_path), check=True)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            rows.append({**row, "plan": f"{FOLDER}/plan.json", "error": str(exc)})
            continue
        rows.append({**row, "plan": f"{FOLDER}/plan.json", "decision": value["decision"],
                     "needed": {name: entry["needed"] for name, entry in value["needed"].items()},
                     "priorities": value["priorities"],
                     "levers": [{"id": lever["id"], "technique": lever["technique"],
                                 "changes": {kpi: change["change"] for kpi, change in lever["changes"].items()}}
                                for lever in value["levers"]],
                     "rejected": [entry["name"] for entry in value["rejected"]],
                     "forecast": {kpi: entry["expected"] for kpi, entry in (value["forecast"] or {}).items()}})
    return {"run_id": run_id, "replays": rows}


def render_collected(found):
    """The collected replays as text, one block per replay."""
    lines = []
    for row in found["replays"]:
        if row.get("error"):
            lines.append(f"{row['replay']}: {row['error']}")
            continue
        order = row["priorities"] or {}
        lines.append(f"{row['replay']}: {row['decision']}; p1 {', '.join(order.get('p1') or []) or '-'}, "
                     f"p2 {', '.join(order.get('p2') or []) or '-'}; forecast "
                     + ", ".join(f"{kpi} {value:,.0f}" for kpi, value in row["forecast"].items()))
        for lever in row["levers"]:
            lines.append(f"  {lever['id']} ({lever['technique']}): "
                         + ", ".join(f"{kpi} {change:+.0%}" for kpi, change in lever["changes"].items()))
        if row["rejected"]:
            lines.append("  rejected: " + "; ".join(row["rejected"]))
    return "\n".join(lines)
