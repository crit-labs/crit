"""crit run eval: Eval of a packaged candidate through crit serve, under one eval reservation.

crit serve's resident host loads the candidate's resident chat adapter, packaged and not yet sealed, and Eval
measures it over the chat API like any served engine (crit-eval's docs/README.md#engines-behind-a-server): speed, request
history, held-out prompts, the long-context curve, retrieval and quality, the engine's scores of the fixed quality
texts through its serving paths (chat-score-v1) against the reference scores of the run's model stored under the
data. When the data holds none yet, the run's model scores the texts first (crit eval reference), once, before the
candidate is served. The result is retained under the candidate's eval/ folder for checks to reference; it
publishes no check or benchmark.

Every part runs in one attempt on one loaded engine (the lower-context gate opens its own, at its context). Eval
measures in stages (crit-eval's protocol stages): as far as its budget allows at the candidate's measured rates,
and the run's kv_tokens itself only with certify, for the candidate that will ship. The attempt has no fixed process
cap, since Eval's work scales with the model and the device: it may run to the
segment's deadline, and each request is bounded by the rates the engine measured in it (crit_eval.run.deadline), so a
hung engine stops the attempt.
"""
from pathlib import Path
import sys
import time

from .records import file_sha, ident, load_run, now, quality_mode, quality_terms, read_json, safe_path, timestamp
from .served import Expired, SERVED, Served, remaining
from .store import amendments

# Quality needs the score call and, for its honesty check, every served token's log-probability.
QUALITY = ("chat-score-v1", "chat-logprobs-v1")


def check_sampling(run, prepared, data, model):
    """The data's sampled-suite sampling is the model's own unless the user overrode it for this run (eval_sampling at
    create); then the data must be prepared with exactly that override."""
    wanted = (run["spec"].get("eval_sampling") or {}).get("settings")
    source = prepared.get("sampling_source")
    if wanted is not None and (source != "override" or prepared.get("sampling") != wanted):
        flag = ",".join(f"{key}={value}" for key, value in wanted.items())
        raise ValueError(f"This run measures sampled decode at the user's sampling {wanted}: prepare the data with it "
                         f"(crit eval prepare --model {model} --data NEW_DIR --sampling {flag})")
    if wanted is None and source == "override":
        raise ValueError(f"{data} was prepared with a sampling override {prepared.get('sampling')}, which only the user "
                         "sets, for a run that declares it (eval_sampling at create): prepare the data again without "
                         "--sampling")


def evaluate(store, run_id, candidate_id, data, parts=None, baseline=None, seconds=None, label=None, stage=None,
             certify=False, budget_seconds=None):
    """Reserve, score the data's quality texts on the run's model unless stored, serve the candidate, run Eval on it
    over the chat API, stop the server and settle. stage, certify and budget_seconds: how far Eval measures
    (crit_eval.run.runner.run). Returns the retained result (path relative to the run, sha256)
    and its status."""
    from crit_eval import load_json, machine_settings
    from crit_eval.quality.prepare import PREPARED, model_identity, prepared_template
    from crit_eval.run.runner import PARTS
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
    missing = [c for c in SERVED + (QUALITY if "quality" in parts else ()) if c not in declared]
    if missing:
        raise ValueError("Eval measures the candidate through crit serve: its package must declare "
                         + ", ".join(missing) + " (docs/SERVING.md)")
    model, data = run["spec"]["model_path"], data.absolute()
    if not (data / PREPARED).is_file():
        raise ValueError(f"{data} holds no {PREPARED}: prepare it from the run's model first "
                         f"(crit eval prepare --model {model} --data {data})")
    prepared = load_json(data / PREPARED)
    if prepared["model"]["sha256"] != model_identity(model, prepared_template(prepared))["sha256"]:
        raise ValueError(f"{data} was prepared from another model than the run's")
    check_sampling(run, prepared, data, model)
    constraints = run["spec"]["hardware_constraints"]
    basis = constraints.get("context") or {}
    # The probe the run's context was derived from gives Eval its physical limits while its bytes are unchanged.
    probe = Path(basis["device_file"]) if basis.get("device_file") else None
    device = read_json(probe) if probe and probe.is_file() and file_sha(probe) == basis["device_sha256"] else None
    settings = machine_settings()
    reserve = settings.get("context_reserve_gib")
    reservation = store.reserve(run_id, candidate_id, "eval", seconds)
    attempt = reservation["attempt_id"]
    budget, exactness = quality_terms(run["spec"]["correctness"], amendments(root, reservation["segment_id"]))
    options = dict(label=label, device=device, baseline=baseline, quality_budget=budget, exactness=exactness,
                   quality_mode=quality_mode(run["spec"]["correctness"]),
                   context_tokens=constraints["kv_tokens"], context_basis=basis,
                   vram_headroom_bytes=round(settings.get("vram_headroom_gib", 1.0) * (1 << 30)),
                   context_reserve_bytes=round(reserve * (1 << 30)) if reserve is not None else None,
                   stage=stage, certify=certify, budget_seconds=budget_seconds)
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
    # A candidate created from a plan gets the plan's outcome: its forecast next to what Eval measured.
    planned = store.plan_outcome(run_id, candidate_id, attempt)
    return {"result": {"path": out.relative_to(root).as_posix(), "sha256": file_sha(out)}, "status": result["status"],
            **({"plan_outcome": f"plans/{planned['plan_id']}/outcomes/{attempt}.json"} if planned else {})}


class EvalEngines(Served):
    """Eval's engines for a packaged candidate (crit_eval.run.runner.run takes it as its adapter): every engine Eval opens
    is the candidate served through crit serve at the context Eval asks for, the run's or the lower-context gate's,
    and stops when Eval closes it."""

    def __init__(self, store, run_id, candidate_id, attempt, kv_tokens, prepared, log, deadline_at):
        # crit serve's own bound on a request is the attempt's deadline: Eval bounds each request by what it measured.
        super().__init__(store, run_id, candidate_id, attempt, kv_tokens, log, "Eval",
                         timeout=min(86400, max(1.0, remaining(deadline_at))))
        self.prepared = prepared

    def create_eval_engine(self, context):
        from crit_eval.run import chat
        _, api, port = self.open(context["context_tokens"])
        served = chat.Server("http://127.0.0.1:%d/v1" % port, self.prepared, api.model).create_eval_engine(context)
        # Eval opens engines one after another: the next one is built in the same resident host (Served.open), and
        # the server stops when Eval's attempt ends (Served.until).
        served.close = lambda: None
        return served


def _measure(store, run_id, candidate_id, attempt, deadline_at, model, data, out, parts, options, log):
    """Score the data's quality texts unless stored, then measure through crit serve; a failed engine cleanup raises
    with `settled` False."""
    from crit_core.machine import engine_lock
    from crit_eval import load_json
    from crit_eval.quality.prepare import PREPARED
    from crit_eval.quality.reference import score_data
    from crit_eval.run.runner import run as run_eval
    served = EvalEngines(store, run_id, candidate_id, attempt, options["context_tokens"], load_json(data / PREPARED),
                         log, deadline_at)
    with engine_lock(f"crit run eval {run_id} {candidate_id}", deadline=time.monotonic() + remaining(deadline_at)):
        if "quality" in parts:
            def check():
                # Between the reference's layers and texts: scoring stops at the reservation's deadline.
                if remaining(deadline_at) <= 0:
                    raise Expired("Eval passed its reserved deadline while the reference scored the quality texts")
            from crit_eval.quality.fixed import ALLOWANCE, STORE
            from crit_eval.quality.packages import fetch, source
            mode = options["quality_mode"]
            # A published package's scores (with the as_quantized allowance beside them), when the source has them.
            if source() is not None and (not (data / STORE / "index.json").is_file()
                                         or mode == "as_quantized" and not (data / STORE / ALLOWANCE).is_file()):
                fetch(model, data, source(), log)
            how = score_data(model, data, log, check)
            log(f"quality: reference scores of {how['positions']:,} positions {how['method']}"
                + (f" in {how['seconds']:.0f} s" if how["method"] == "scored" else ""))
        with served.until(deadline_at):
            # Eval closes every engine it opens, also when a part fails.
            result = run_eval(served, model, data, out, {}, parts, log=log, **options)
        if result.get("aborted"):
            raise ValueError("Eval aborted the round: " + result["aborted"]["detail"])
    return result
