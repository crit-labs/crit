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
            "context_reserve_gib": (int, float), "device": str, "context_tokens": int,
            "eval_packages": str, "quality_mode": str}


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


def json_argument(value, option):
    """An option's JSON text, or a file holding it."""
    if Path(value).is_file():
        return read_json(Path(value))
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option} is neither a file nor JSON: {exc}") from None

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT if (ROOT / "core/include/crit/core.h").is_file() else Path.cwd())
    parser.add_argument("--root", type=Path, help="Run root; defaults to PROJECT/runs")
    parser.add_argument("--id", help="Existing run identity")
    subs = parser.add_subparsers(dest="action", required=True)
    for name in ("create", "resolve", "plan", "check", "benchmark", "stop"):
        sub = subs.add_parser(name)
        sub.add_argument("--record", type=Path, required=name != "check")
        if name in {"check", "benchmark"}:
            sub.add_argument("--candidate", required=True)
    subs.choices["check"].add_argument("--from-eval", nargs="?", const="newest", metavar="ATTEMPT",
                                       help="Instead of --record: record the arithmetic, functional and resources check "
                                            "of every case from the candidate's retained Eval result (default its newest)")
    subs.choices["plan"].add_argument("--check", action="store_true",
                                      help="Run the plan's checks and print the record crit would store, recording nothing "
                                           "(the plan command of a replay)")
    subs.choices["plan"].description = ("Record the next candidate's plan after the parent's Eval and before any code: "
                                        "needed flags, strategy, levers, forecast and gates; crit freezes the parent's "
                                        "newest Eval and computes the priorities")
    subs.choices["create"].add_argument("--from-engine", type=Path, metavar="ENGINE_DIR",
                                        help="Start from a crit-hub engine: its engines/<owner>/<name>/ folder (card.yaml, "
                                             "card.lock.json, evidence/) is recorded as the run's origin_engine")
    sub = subs.add_parser("candidate")
    sub.add_argument("--source", type=Path, required=True)
    sub.add_argument("--record", type=Path, required=True, help="label, rationale, strategy, configuration, store_decisions, parent/origin IDs")
    sub = subs.add_parser("recon", help="Gather the model's sampling, stop tokens, chat template and config into work/recon.json")
    sub.add_argument("--force", action="store_true", help="Gather again over an existing work/recon.json")
    sub.add_argument("--drafter", type=Path, action="append", metavar="PATH", help="A companion drafter to record "
                     "(EAGLE-3, DFlash, DSpark or crit models drafter output); repeatable")
    sub.add_argument("--compare", type=Path, metavar="MODEL", help="Instead: compare the run's model fingerprint with "
                     "another checkpoint's (architecture, weight layout, tokenizer, template) and print what differs")
    sub = subs.add_parser("capture", help="Write a case's chat request (captured/CASE/request.json) that renders to "
                                          "the workload's prompt tokens with the template variables candidates serve")
    sub.add_argument("--case", default="case-0001")
    sub.add_argument("--text", type=Path, help="File of the user message's text (default: a neutral passage); it is "
                                               "cut to whole words and padded to the exact count")
    sub.add_argument("--instruction", default=None, help="Line after the text (default: a summary request; '' for none)")
    sub.add_argument("--system", help="A leading system message")
    sub.add_argument("--template-kwargs", metavar="JSON", help="chat_template_kwargs the request renders with, over "
                                                               "recon's frozen template_variables (text or a file)")
    sub.add_argument("--inputs", type=Path, metavar="INPUTS_JSON", help="Set the case's manifest row in this resolve "
                                                                        "payload (created when missing)")
    sub = subs.add_parser("build", help="Build a candidate's retained source with the recipe's build.command under a "
                                        "build attempt (log kept, attempt settled), then package the result")
    sub.add_argument("--candidate", required=True)
    sub.add_argument("--record", type=Path, required=True, help="Recipe: build {toolchain, command[, dependencies]}, "
                     "launch[, capabilities]; the build's outputs are bound as dependencies, capabilities default to "
                     "the adapter's declared_capabilities(model_path)")
    sub.add_argument("--seconds", type=float, help="Reservation length (default: the build process cap)")
    sub.add_argument("--drop", action="append", default=[], metavar="GLOB",
                     help="Remove these build-folder paths (intermediates) before packaging; repeatable")
    sub.add_argument("--no-package", action="store_true", help="Build and settle only")
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
    sub = subs.add_parser("measure", help="Benchmark the run's cases on a packaged candidate through crit serve")
    sub.add_argument("--candidate", required=True)
    sub.add_argument("--seconds", type=float, help="Reservation length (default: the benchmark process cap)")
    sub = subs.add_parser("eval", help="Run Eval on a packaged candidate through crit serve, quality against the "
                                       "run's model's scores of the fixed texts (scored once into the data)")
    sub.add_argument("--candidate", required=True)
    sub.add_argument("--data", type=Path, required=True, help="Eval data directory prepared from the run's model")
    sub.add_argument("--only", action="append", choices=["speed", "quality", "exactness", "context", "heldout", "history", "followup"],
                     help="Run only these Eval parts (repeatable); default all")
    sub.add_argument("--baseline", type=Path, metavar="RESULT", help="Earlier Eval result to show KPI changes against")
    sub.add_argument("--label")
    sub.add_argument("--seconds", type=float, help="Reservation length (default: to the segment's deadline; each "
                                                   "request is bounded by the rates the engine measured)")
    sub = subs.add_parser("serve", help="Serve a packaged candidate before it is sealed through crit serve, under a "
                                        "check attempt, until Ctrl+C: for the checks a candidate passes before sealing")
    sub.add_argument("--candidate", required=True)
    sub.add_argument("--host", default="127.0.0.1")
    sub.add_argument("--port", type=int, default=8000)
    sub.add_argument("--seconds", type=float, help="Reservation length (default: the check process cap)")
    sub.add_argument("--check", action="store_true", help="Run the standard serving checks (consecutive requests, "
                     "system message, streaming, stop string, cancelling, tool call, logprobs, template switch, "
                     "stopping), keep their report and record a functional check per case, then stop")
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
    sub = subs.add_parser("cost", help="Account the run's build cost from the agent's transcripts: wall time, tokens and "
                                       "API price per candidate, written to cost.json")
    sub.add_argument("--transcript", type=Path, action="append", required=True, metavar="PATH",
                     help="A Claude Code session, Codex rollout or plain usage JSONL (repeatable)")
    sub.add_argument("--agent", help="The agent and model that built the run, as the record names it")
    sub.add_argument("--prices", help="JSON object or file of {model: {input, cache_read, cache_write_5m, cache_write_1h, "
                                      "output}} in USD per million tokens, over the list prices")
    sub.add_argument("--until", help="Count requests up to this UTC timestamp (default: the run's stop, else now)")
    sub = subs.add_parser("replay", help="Copy the run as it stood when a candidate's Eval finished, for an agent to say "
                                         "what it would do next with today's crit and crit-eval (records nothing)")
    sub.add_argument("--at", metavar="CANDIDATE", help="The candidate whose newest completed Eval the replays start from")
    sub.add_argument("--before", metavar="PLAN", help="Instead: the run as it stood just before this plan was recorded "
                     "(with the work done toward it since its parent's Eval, without drafts of the plan)")
    sub.add_argument("--out", type=Path, required=True, help="Folder of the replays: replay-1 ... replay-N, each a run root")
    sub.add_argument("--count", type=int, default=1, help="Replays to make, one per agent (default 1)")
    sub.add_argument("--agent", metavar="CMD", help="Start an agent on every replay at once with this command (any "
                     "harness; {brief}, {folder}, {root} and {number} name the replay's), wait for all, then collect")
    sub.add_argument("--collect", action="store_true", help="Instead: check and list the plans the replays under --out wrote")
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
        payload = read_json(args.record) if getattr(args, "record", None) else None
        if args.action == "create":
            settings = load_settings(args.project)
            if isinstance(payload, dict):
                # Request values win; the run records the headroom it actually used.
                filled = {key: settings[key] for key in ("ram_headroom_gib", "vram_headroom_gib", "context_reserve_gib",
                                                         "context_tokens", "quality_mode") if key in settings}
                if "device" in settings:
                    # This machine's probe result: the run is built for Eval's max context on it.
                    device = Path(settings["device"]).expanduser()
                    filled["device"] = str(device if device.is_absolute() else args.project / device)
                payload = {**filled, **payload}
            result = store.create(payload, args.from_engine)
        elif args.action == "resolve":
            result = store.freeze_inputs(args.id, payload)
        elif args.action == "plan":
            result = store.plan(args.id, payload, args.check)
        elif args.action == "replay" and args.collect:
            from .replay import collect, render_collected
            result = collect(args.out, args.id, RunStore, args.project)
            print(render_collected(result), file=sys.stderr)
        elif args.action == "replay":
            if bool(args.at) == bool(args.before):
                parser.error("replay needs --at CANDIDATE or --before PLAN (or --collect)")
            from .replay import collect, launch, make, render_collected
            result = make(store, args.id, args.at, args.out, args.count, args.before)
            if args.agent:
                result = launch(result, args.agent)
                result["collected"] = collect(args.out, args.id, RunStore, args.project)
                print(render_collected(result["collected"]), file=sys.stderr)
        elif args.action == "candidate":
            result = store.candidate(args.id, args.source, payload)
        elif args.action == "recon" and args.compare:
            result = store.recon_compare(args.id, args.compare)
        elif args.action == "recon":
            result = store.recon(args.id, args.force, args.drafter or ())
        elif args.action == "package":
            result = store.package(args.id, args.candidate, args.source, payload)
        elif args.action == "check" and args.from_eval:
            if args.record:
                parser.error("check takes --record or --from-eval, not both")
            from .checks import from_eval
            result = from_eval(store, args.id, args.candidate, None if args.from_eval == "newest" else args.from_eval)
        elif args.action == "check" and not args.record:
            parser.error("check needs --record or --from-eval")
        elif args.action == "capture":
            from .capture import INSTRUCTION, capture
            result = capture(store, args.id, args.case,
                             args.text.read_text(encoding="utf-8") if args.text else None,
                             INSTRUCTION if args.instruction is None else args.instruction, args.system,
                             json_argument(args.template_kwargs, "--template-kwargs") if args.template_kwargs else None,
                             args.inputs)
        elif args.action == "build":
            from .build import build
            result = build(store, args.id, args.candidate, payload, args.seconds, args.drop, not args.no_package)
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
            result = measure(store, args.id, args.candidate, args.seconds)
            for record in result:
                warn_exploratory(store, args.id, args.candidate, record["benchmark_id"])
        elif args.action == "eval":
            from .evaluation import evaluate
            result = evaluate(store, args.id, args.candidate, args.data, args.only,
                              read_json(args.baseline) if args.baseline else None, args.seconds, args.label)
            from crit_eval.reports.status import render
            from .plans import extract, priorities, rules
            status = result.pop("status")
            print(render(status), file=sys.stderr)
            # Eval states the goals without an order; the run's priority rule orders them for the next plan.
            order, reason = priorities({"decode": {"needed": True}, "prefill": {"needed": True}}, extract(status),
                                       rules(store.get(args.id)["run"]))
            print("priorities while decode and prefill are both needed: " + (
                "; ".join(f"{tier} {', '.join(order[tier])}" for tier in ("p1", "p2", "p3") if order[tier])
                if order else reason) + " (crit run ... plan computes them from the plan's needed flags)", file=sys.stderr)
        elif args.action == "serve":
            from .served import serve
            result = serve(store, args.id, args.candidate, args.host, args.port, args.seconds, check=args.check)
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
        elif args.action == "cost":
            result = store.cost(args.id, args.transcript, args.agent, args.prices, args.until)
        elif args.action == "recover":
            result = store.recover(args.id)
        else:
            result = store.get(args.id)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1
