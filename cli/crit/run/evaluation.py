"""crit run eval: Eval of a packaged candidate through crit serve, under a benchmark reservation.

crit serve's resident host loads the candidate's resident chat adapter, packaged and not yet sealed, and Eval
measures it over the chat API like any served engine (docs/eval/README.md#engines-behind-a-server): speed, request
history, held-out prompts, the long-context curve, retrieval and the quality replies. The server then stops and the
run's model scores the replies (crit eval reference), so the quality gate judges the engine's own replies against
the model's checkpoint. The result is retained under the candidate's eval/ folder for checks to reference; it
publishes no check or benchmark.
"""
from pathlib import Path
import sys
import time

from .records import file_sha, ident, load_run, now, quality_terms, read_json, safe_path, timestamp
from .served import Expired, SERVED, Served, remaining
from .store import amendments

LOGPROBS = "chat-logprobs-v1"  # the quality replies need every token's log-probability


def evaluate(store, run_id, candidate_id, data, parts=None, baseline=None, seconds=None, label=None):
    """Reserve, serve the candidate, run Eval on it over the chat API, stop the server, score its quality replies on
    the run's model and settle. Returns the retained result (path relative to the run, sha256) and its status."""
    from crit_eval import load_json, machine_settings
    from crit_eval.prepare import PREPARED, model_identity
    from crit_eval.runner import PARTS
    root = store.directory(run_id)
    run = load_run(root)
    folder = safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}")
    if (folder / "seal.json").exists():
        raise ValueError("Sealed candidate is immutable; allocate a new candidate")
    manifest_path = folder / "package" / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Package the candidate before measuring it")
    parts = tuple(parts or PARTS)
    declared = set(read_json(manifest_path)["capabilities"])
    missing = [c for c in SERVED + ((LOGPROBS,) if "quality" in parts else ()) if c not in declared]
    if missing:
        raise ValueError("Eval measures the candidate through crit serve: its package must declare "
                         + ", ".join(missing) + " (docs/SERVING.md)")
    model, data = run["spec"]["model_path"], data.absolute()
    if not (data / PREPARED).is_file():
        raise ValueError(f"{data} holds no {PREPARED}: prepare it from the run's model first "
                         f"(crit eval prepare --model {model} --data {data})")
    if load_json(data / PREPARED)["model"]["sha256"] != model_identity(model)["sha256"]:
        raise ValueError(f"{data} was prepared from another model than the run's")
    constraints = run["spec"]["hardware_constraints"]
    basis = constraints.get("context") or {}
    # The probe the run's context was derived from gives Eval its physical limits while its bytes are unchanged.
    probe = Path(basis["device_file"]) if basis.get("device_file") else None
    device = read_json(probe) if probe and probe.is_file() and file_sha(probe) == basis["device_sha256"] else None
    settings = machine_settings()
    reserve = settings.get("context_reserve_gib")
    reservation = store.reserve(run_id, candidate_id, "benchmark", seconds)
    attempt = reservation["attempt_id"]
    budget, exactness = quality_terms(run["spec"]["correctness"], amendments(root, reservation["segment_id"]))
    options = dict(label=label, device=device, baseline=baseline, quality_budget=budget, exactness=exactness,
                   context_tokens=constraints["kv_tokens"],
                   vram_headroom_bytes=round(settings.get("vram_headroom_gib", 1.0) * (1 << 30)),
                   context_reserve_bytes=round(reserve * (1 << 30)) if reserve is not None else None)
    out = folder / "eval" / f"{attempt}.json"
    log_path = folder / "logs" / f"eval-{attempt}.log"
    outcome, failure, settled, interrupted = "completed", None, True, None
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            def log(message):
                print(message, file=sys.stderr, flush=True)
                stream.write(message + "\n")
                stream.flush()
            result = _measure(store, run_id, candidate_id, attempt, reservation["deadline_at"], model, data, out,
                              parts, options, log)
    except BaseException as exc:
        settled = getattr(exc, "settled", True)
        if isinstance(exc, (Expired, TimeoutError)):
            outcome, failure = "timed_out", str(exc) or "Eval passed its reserved deadline"
        elif isinstance(exc, KeyboardInterrupt):
            outcome, failure, interrupted = "interrupted", "Interrupted", exc
        else:
            outcome, failure = "failed", (type(exc).__name__ + ": " + str(exc))[:8000]
    ended = now()
    if outcome == "completed" and timestamp(ended, "ended") > timestamp(reservation["deadline_at"], "deadline"):
        outcome, failure = "timed_out", "Eval passed its reserved deadline"
    logs = ([{"path": log_path.relative_to(root).as_posix(), "sha256": file_sha(log_path)}]
            if settled and log_path.is_file() else [])
    # Eval and the server ran in this process; the resident host's exit is part of the cleanup.
    store.settle(run_id, reservation, {"outcome": outcome, "ended_at": ended,
                                       "exit_code": 0 if outcome == "completed" else None,
                                       "cleanup_settled": settled, "logs": logs, "failure": failure})
    if interrupted is not None:
        raise interrupted
    if outcome != "completed":
        raise ValueError(f"Eval {outcome}: {failure} (log {log_path.relative_to(root).as_posix()})")
    return {"result": {"path": out.relative_to(root).as_posix(), "sha256": file_sha(out)}, "status": result["status"]}


class EvalEngines(Served):
    """Eval's engines for a packaged candidate (crit_eval.runner.run takes it as its adapter): every engine Eval opens
    is the candidate served through crit serve at the context Eval asks for, the run's or the lower-context gate's,
    and stops when Eval closes it."""

    def __init__(self, store, run_id, candidate_id, attempt, kv_tokens, prepared, log):
        super().__init__(store, run_id, candidate_id, attempt, kv_tokens, log, "Eval")
        self.prepared = prepared

    def create_eval_engine(self, context):
        from crit_eval import chat
        _, api, port = self.open(context["context_tokens"])
        served = chat.Server("http://127.0.0.1:%d/v1" % port, self.prepared, api.model).create_eval_engine(context)
        served.close = self.stop
        return served


def _measure(store, run_id, candidate_id, attempt, deadline_at, model, data, out, parts, options, log):
    """Measure through crit serve, then score; a failed engine cleanup raises with `settled` False."""
    from crit_core.machine import engine_lock
    from crit_eval import load_json
    from crit_eval.prepare import PREPARED
    from crit_eval.reference import score_result
    from crit_eval.runner import run as run_eval
    served = EvalEngines(store, run_id, candidate_id, attempt, options["context_tokens"], load_json(data / PREPARED),
                         log)
    with engine_lock(f"crit run eval {run_id} {candidate_id}", deadline=time.monotonic() + remaining(deadline_at)):
        with served.until(deadline_at):
            # Eval closes every engine it opens, also when a part fails.
            result = run_eval(served, model, data, out, {}, parts, log=log, **options)
        if result.get("aborted"):
            raise ValueError("Eval aborted the round: " + result["aborted"]["detail"])
        if (result.get("quality") or {}).get("status") == "replied":
            if remaining(deadline_at) <= 0:
                raise Expired("Eval passed its reserved deadline before scoring")
            result = score_result(model, data, out, options["baseline"], log)
    return result
