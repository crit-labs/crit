"""First-class run/candidate publication commands for local agents."""
import argparse
import json
from pathlib import Path
import signal
import sys
import tomllib

from .records import exploratory_reason, ineligibility, read_json, safe_path
from .store import ROOT, RunStore

SETTINGS = {"wsl_run_root": str, "ram_headroom_gib": (int, float), "vram_headroom_gib": (int, float),
            "context_reserve_gib": (int, float), "device": str}


def load_settings(project):
    """Machine settings from PROJECT/settings.toml; a missing file means none."""
    path = Path(project) / "settings.toml"
    if not path.is_file():
        return {}
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    for key, item in value.items():
        if key not in SETTINGS:
            raise ValueError(f"{path}: unknown setting {key!r}; known: {', '.join(SETTINGS)}")
        if isinstance(item, bool) or not isinstance(item, SETTINGS[key]):
            raise ValueError(f"{path}: {key} has the wrong type")
    return value


def _candidate(store, run_id, candidate_id):
    return next((c for c in store.get(run_id)["candidates"] if c["candidate_id"] == candidate_id), None)


def warn_exploratory(store, run_id, candidate_id, benchmark_id):
    """A benchmark that does not qualify keeps the candidate ineligible; say so when it is recorded. The run view
    shows every series of an unsealed candidate as exploratory, so read the candidate's own projection."""
    path = safe_path(store.directory(run_id), f"candidates/{candidate_id}/telemetry.json")
    candidate = read_json(path) if path.is_file() else _candidate(store, run_id, candidate_id)
    series = next((s for s in (candidate or {}).get("series", []) if s["benchmark_id"] == benchmark_id), None)
    if series and series["qualification"] != "qualified":
        print(f"warning: {benchmark_id} is exploratory and cannot make {candidate_id} eligible: {exploratory_reason(series)}",
              file=sys.stderr)


def warn_ineligible(store, run_id, candidate_id):
    view = store.get(run_id)
    candidate = next((c for c in view["candidates"] if c["candidate_id"] == candidate_id), None)
    if candidate and not candidate["eligible"]:
        reasons = ineligibility(view["run"], candidate)
        print(f"warning: {candidate_id} is sealed but not eligible for select" + (": " + "; ".join(reasons) if reasons else ""),
              file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT if (ROOT / "core/include/crit/core.h").is_file() else Path.cwd())
    parser.add_argument("--root", type=Path, help="Run root; defaults to PROJECT/runs")
    parser.add_argument("--id", help="Existing run identity")
    subs = parser.add_subparsers(dest="action", required=True)
    for name in ("create", "resolve", "check", "benchmark", "stop"):
        sub = subs.add_parser(name)
        sub.add_argument("--record", type=Path, required=True)
        if name in {"check", "benchmark"}:
            sub.add_argument("--candidate", required=True)
    sub = subs.add_parser("candidate")
    sub.add_argument("--source", type=Path, required=True)
    sub.add_argument("--record", type=Path, required=True, help="label, rationale, strategy, configuration, store_decisions, parent/origin IDs")
    sub = subs.add_parser("recon", help="Gather the model's sampling, stop tokens, chat template and config into work/recon.json")
    sub.add_argument("--force", action="store_true", help="Gather again over an existing work/recon.json")
    sub = subs.add_parser("package")
    sub.add_argument("--candidate", required=True)
    sub.add_argument("--source", type=Path, required=True)
    sub.add_argument("--record", type=Path, required=True, help="build, launch and capabilities")
    sub = subs.add_parser("reserve")
    sub.add_argument("--candidate", help="Candidate owner; omit only for discovery")
    sub.add_argument("--operation", choices=("discovery", "build", "check", "benchmark"), required=True)
    sub.add_argument("--seconds", type=float)
    sub = subs.add_parser("settle")
    sub.add_argument("--reservation", required=True, help="Reservation path relative to run")
    sub.add_argument("--record", type=Path, required=True)
    sub = subs.add_parser("measure", help="Benchmark the run's cases on a candidate package through its Eval adapter")
    sub.add_argument("--candidate", required=True)
    sub.add_argument("--adapter", required=True, help="The package's eval-v1 adapter, relative to the package")
    sub.add_argument("--model", type=Path, help="Model directory for the adapter (default: the run's model)")
    sub.add_argument("--config", metavar="JSON", help="Configuration overrides passed to the adapter")
    sub.add_argument("--seconds", type=float, help="Reservation length (default: the benchmark process cap)")
    sub = subs.add_parser("seal")
    sub.add_argument("--candidate", required=True)
    sub.add_argument("--lifecycle", required=True)
    sub.add_argument("--failure", help="Retained failure message")
    sub = subs.add_parser("select")
    sub.add_argument("--candidate", required=True)
    sub = subs.add_parser("resume")
    sub.add_argument("--candidates", type=int, help="Candidates this segment builds (no working time budget)")
    sub.add_argument("--minutes", type=int, help="Time-budget segment instead, or the safety window with --candidates")
    sub.add_argument("--attempt-limit", type=int, default=64)
    sub.add_argument("--external-agent-stopped", action="store_true")
    sub.add_argument("--quality-budget", metavar="JSON", help="Amend the drift budget: any of ppl_ratio_max, kld_mean_max, kld_p99_max")
    sub.add_argument("--exactness", choices=["reported", "required"], help="Amend the speculative exactness mode")
    sub.add_argument("--approved-by", help="Who approved the amendment (required with --quality-budget or --exactness)")
    subs.add_parser("show")
    subs.add_parser("recover", help="Explicit dead-owner lease recovery and projection rebuild; never resets attempts/deadlines")
    args = parser.parse_args(argv)
    if hasattr(signal, "SIGTERM") and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        # A terminated command (for example a timeout) unwinds and releases its lease.
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    if args.action != "create" and not args.id:
        parser.error("--id is required for an existing run")
    try:
        store = RunStore(args.root or args.project / "runs", args.project)
        payload = read_json(args.record) if hasattr(args, "record") else None
        if args.action == "create":
            settings = load_settings(args.project)
            if isinstance(payload, dict):
                # Request values win; the run records the headroom it actually used.
                filled = {key: settings[key] for key in ("ram_headroom_gib", "vram_headroom_gib", "context_reserve_gib")
                          if key in settings}
                if "device" in settings:
                    # This machine's probe result: the run is built for Eval's max context on it.
                    device = Path(settings["device"]).expanduser()
                    filled["device"] = str(device if device.is_absolute() else args.project / device)
                payload = {**filled, **payload}
            result = store.create(payload)
        elif args.action == "resolve":
            result = store.freeze_inputs(args.id, payload)
        elif args.action == "candidate":
            result = store.candidate(args.id, args.source, payload)
        elif args.action == "recon":
            result = store.recon(args.id, args.force)
        elif args.action == "package":
            result = store.package(args.id, args.candidate, args.source, payload)
        elif args.action in {"check", "benchmark"}:
            result = store.append(args.id, args.candidate, args.action, payload)
            if args.action == "benchmark":
                warn_exploratory(store, args.id, args.candidate, result["benchmark_id"])
        elif args.action == "reserve":
            result = store.reserve(args.id, args.candidate, args.operation, args.seconds)
        elif args.action == "settle":
            reservation = read_json(safe_path(store.directory(args.id), args.reservation))
            result = store.settle(args.id, reservation, payload)
        elif args.action == "measure":
            from .measure import measure
            result = measure(store, args.id, args.candidate, args.adapter, args.model,
                             json.loads(args.config) if args.config else None, args.seconds)
            for record in result:
                warn_exploratory(store, args.id, args.candidate, record["benchmark_id"])
        elif args.action == "seal":
            failure = {"message": args.failure} if args.failure else None
            result = store.seal(args.id, args.candidate, args.lifecycle, failure)
            warn_ineligible(store, args.id, args.candidate)
        elif args.action == "select":
            result = store.select(args.id, args.candidate)
        elif args.action == "resume":
            amend = {key: value for key, value in (("quality_budget", json.loads(args.quality_budget) if args.quality_budget else None),
                                                   ("exactness", args.exactness), ("approved_by", args.approved_by)) if value is not None}
            result = store.resume(args.id, args.minutes, args.attempt_limit, args.external_agent_stopped, args.candidates,
                                  amend or None)
        elif args.action == "stop":
            result = store.stop(args.id, **payload)
        elif args.action == "recover":
            result = store.recover(args.id)
        else:
            result = store.get(args.id)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1
