"""crit run eval: Eval of a packaged candidate through crit serve, under one eval reservation.

crit serve's resident host loads the candidate's resident chat adapter, packaged and not yet sealed, and Eval
measures it over the chat API like any served engine (crit-eval's docs/README.md#engines-behind-a-server): speed, request
history, held-out prompts, the long-context curve, retrieval and quality, the engine's scores of the fixed quality
texts through its serving paths (chat-score-v1) against the reference scores of the run's model stored under the
data. When the data holds none yet, the run's model scores the texts first (crit eval reference), once, before the
candidate is served. The result is retained under the candidate's eval/ folder for checks to reference; it
publishes no check or benchmark.

Every part runs in one attempt on one loaded engine (the lower-context gate opens its own, at its context). The
attempt journals every engine it opens and every request served beside its result (eval/attempt-N.journal.jsonl,
crit-eval's crit_eval.run.journal), each line synced as it completes; an attempt that stops for any reason
(interrupted, failed, past its deadline, a disturbed or hung engine) keeps its journal and what it measured
(eval/attempt-N.json, `aborted`), and `--resume` measures in a new attempt from where it stopped: the journal's
requests are answered again at their recorded times, the engine loads only for the requests left. Eval
measures in stages (crit-eval's protocol stages): as far as its budget allows at the candidate's measured rates,
and the run's kv_tokens itself only with certify, for the candidate that will ship. On the candidate ladder
(crit.run.ladder) the candidate's rung sets how far: none on candidate 1, at most stage 1 (the small Eval) on candidate
2, at most stage 2 (the medium Eval) on candidates 3 and 4, and the certified Eval (the full one) from candidate 5 on
and on the segment's last budgeted candidate. The attempt has no fixed process
cap, since Eval's work scales with the model and the device: it may run to the
segment's deadline, and each request is bounded by the rates the engine measured in it (crit_eval.run.deadline), so a
hung engine stops the attempt.
"""
import contextlib
import os
from pathlib import Path
import time

from . import ladder
from .attempt import Attempt, Expired, packaged, remaining, unsealed
from .records import file_sha, ident, load_run, now, quality_mode, quality_terms, read_json, safe_path
from .served import Served
from .store import amendments

# Allocator settings the reference may set for its own process (expandable segments) and that crit serve's resident
# host, started from this process, must not inherit: the engine runs under the allocator its user started crit with.
ALLOCATOR_ENV = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")


# Placement scenarios (crit_eval.limits.context.SCENARIOS) in which a pass reads weights from host memory.
OFF_DEVICE = ("sparse_device_host", "sparse_device_host_disk", "dense_device_host")


def summary(store, run_id, outcome):
    """What crit run ... eval prints once Eval has run (outcome: evaluate's, whose status it takes): Eval's status,
    the priorities the run's rule gives while decode and prefill are both needed, how far each KPI stands from its
    limit, and the host_bytes_warning."""
    from crit_eval.reports.status import render
    from .plans import extract, priorities, rules, standing
    status = outcome.pop("status")
    lines = [render(status)]
    # Eval states the goals without an order; the run's priority rule orders them for the next plan.
    order, reason = priorities({"decode": {"needed": True}, "prefill": {"needed": True}}, extract(status),
                               rules(store.get(run_id)["run"]))
    lines.append("priorities while decode and prefill are both needed: " + (
        "; ".join(f"{tier} {', '.join(order[tier])}" for tier in ("p1", "p2", "p3") if order[tier])
        if order else reason) + " (crit run ... plan computes them from the plan's needed flags)")
    # How far each prioritized KPI stands from its limit: a low fraction is the next plan's room, not a result.
    lines += ["  " + line for line in standing(status)]
    path = (outcome.get("result") or {}).get("path")
    root = store.directory(run_id)
    warning = host_bytes_warning(root, read_json(safe_path(root, path))) if path else None
    return lines + ([warning] if warning else [])


def host_bytes_warning(root, result):
    """What crit run eval prints when recon's placement keeps weights off the device and the Eval result's engine
    declares no host_bytes_per_pass: a note when Eval priced its floors by the placement's memory tiers (crit-eval's
    headroom.tiers), at the engine's measured expert cache hit rate, or with the misses beyond the device's capacity
    unmeasured (an optimistic floor) while the engine reports no expert reads and hits, else a warning that Eval priced
    its pass floor as if every byte came from device memory, a floor it cannot reach. None otherwise."""
    from .recon import RECON_JSON
    facts = safe_path(root, RECON_JSON)
    scenario = (((read_json(facts).get("gathered") or {}).get("placement") or {}).get("scenario")
                if facts.is_file() else None)
    declared = ((result or {}).get("engine") or {}).get("host_bytes_per_pass")
    if scenario not in OFF_DEVICE or isinstance(declared, dict) and (declared.get("link") or declared.get("cpu")):
        return None
    priced = (((result or {}).get("headroom") or {}).get("host_pass") or {})
    if priced.get("source") == "placement":
        rate = priced.get("hit_rate")
        rate = f"a {rate:.0%}" if isinstance(rate, (int, float)) else "an unmeasured"
        return (f"NOTE: recon's placement is {scenario} and the candidate declares no host_bytes_per_pass, so Eval priced "
                f"its floors by the placement's memory tiers at {rate} expert cache hit rate "
                f"({priced.get('hit_rate_source')}). Report expert_reads and expert_hits in the decode telemetry for the "
                "cache's own hit rate, or host_bytes_per_pass ({\"link\", \"cpu\"} bytes per pass) for the engine's own "
                "path.")
    return (f"WARNING: recon's placement is {scenario} (weights read from host memory), but the candidate declares no "
            "host_bytes_per_pass, so Eval priced its pass floor as if every weight were on the device. Report "
            "{\"link\": bytes copied over the host link, \"cpu\": bytes the CPU computes from} per pass in the "
            "engine's facts: Eval then prices a reachable floor by source and keeps the all-on-device fraction "
            "(fixed_decode) beside it.")


@contextlib.contextmanager
def own_allocator_env():
    """Restore ALLOCATOR_ENV as it was when the block began: a reference that exports them (crit-eval before its
    allocator change) leaves the served engine's environment as it found it."""
    before = {key: os.environ.get(key) for key in ALLOCATOR_ENV}
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def resumable(store, run_id, root, folder, attempt=None):
    """The journal of the candidate's Eval attempt a new attempt resumes: `attempt`'s, else the newest that has one.
    An attempt that completed has nothing left to measure."""
    from crit_eval.run.journal import SUFFIX
    found = sorted((folder / "eval").glob("attempt-*" + SUFFIX))
    if attempt is not None:
        found = [p for p in found if p.name == ident(attempt, "attempt") + SUFFIX]
    if not found:
        raise ValueError(f"{folder.name} has no Eval journal" + (f" for {attempt}" if attempt else "")
                         + " to resume: run crit run ... eval without --resume")
    path = found[-1]
    name = path.name[:-len(SUFFIX)]
    outcome = folder / "attempts" / name / "result.json"
    if not outcome.is_file():
        settle_abandoned(store, run_id, root, folder, name, path)
    if read_json(outcome)["outcome"] == "completed":
        raise ValueError(f"{name} completed: its result stands, nothing is left to resume")
    return path


# How long a journal stays untouched before an unsettled attempt's process counts as gone (with the lock free).
QUIET_SECONDS = 60


def settle_abandoned(store, run_id, root, folder, name, journal):
    """Settle an Eval attempt whose process ended without settling it (killed, its session or the machine stopped) as
    interrupted, with its journal and what it measured as evidence. Eval holds the machine's engine lock while it
    measures: the lock free and the journal untouched for a minute prove no process measures for the attempt any more."""
    from crit_core.machine import engine_lock, lock_path
    if lock_path() is None:
        raise ValueError(f"{name} is not settled and the machine's engine lock is off, so nothing proves its process "
                         "ended: settle it (crit run ... settle) before resuming its journal")
    if time.time() - journal.stat().st_mtime < QUIET_SECONDS:
        raise ValueError(f"{name} is not settled and its journal was written in the last minute: its Eval may still "
                         "run; wait for it, or stop it")
    try:
        with engine_lock(f"crit run eval --resume {name}", wait=False):
            pass
    except RuntimeError as busy:
        raise ValueError(f"{name} is not settled and {busy}: its Eval may still run; stop it first") from None
    reservation = read_json(folder / "attempts" / name / "reservation.json")
    kept = [folder / "logs" / f"eval-{name}.log", journal, folder / "eval" / f"{name}.json"]
    store.settle(run_id, reservation, {
        "outcome": "interrupted", "ended_at": now(), "exit_code": None, "cleanup_settled": True,
        "logs": [{"path": p.relative_to(root).as_posix(), "sha256": file_sha(p)} for p in kept if p.is_file()],
        "failure": "The Eval process ended without settling its attempt (settled by --resume: the machine's engine "
                   "lock was free and the journal untouched)"})


def evaluate(store, run_id, candidate_id, data, parts=None, baseline=None, seconds=None, label=None, stage=None,
             certify=False, budget_seconds=None, resume=None):
    """Reserve, score the data's quality texts on the run's model unless stored, serve the candidate, run Eval on it
    over the chat API, stop the server and settle. stage, certify and budget_seconds: how far Eval measures
    (crit_eval.run.runner.run). resume: True or an attempt id: go on from that Eval attempt's journal (the newest's
    with True), which must have measured this package with the same options. Returns the retained result (path
    relative to the run, sha256) and its status."""
    from crit_eval import load_json, machine_settings
    from crit_eval.quality.prepare import PREPARED, model_identity, prepared_template
    from crit_eval.run.journal import path_for
    from crit_eval.run.runner import PARTS
    root = store.directory(run_id)
    run = load_run(root)
    folder = unsealed(root, candidate_id)
    parts = tuple(parts or PARTS)
    packaged(folder, "measuring", "Eval measures")
    # The candidate ladder: the rung says how far Eval measures the candidate, and refuses more.
    if ladder.ladder(run):
        stage, certify = ladder.eval_options(ladder.scope_of(root, candidate_id), stage, certify, candidate_id)
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
    earlier = resumable(store, run_id, root, folder, None if resume is True else resume) if resume else None
    attempt = Attempt(store, run_id, candidate_id, "eval", seconds, "eval", "Eval", "Eval")
    reservation = attempt.reservation
    budget, exactness = quality_terms(run["spec"]["correctness"], amendments(root, reservation["segment_id"]))
    options = dict(label=label, device=device, baseline=baseline, quality_budget=budget, exactness=exactness,
                   quality_mode=quality_mode(run["spec"]["correctness"]),
                   context_tokens=constraints["kv_tokens"], context_basis=basis,
                   vram_headroom_bytes=round(settings.get("vram_headroom_gib", 1.0) * (1 << 30)),
                   context_reserve_bytes=round(reserve * (1 << 30)) if reserve is not None else None,
                   stage=stage, certify=certify, budget_seconds=budget_seconds,
                   # What makes this attempt's measurements the ones a resumed attempt may answer from.
                   identity={"run_id": run_id, "candidate_id": candidate_id, "package_sha256": reservation["package_sha256"],
                             "parts": sorted(parts), "label": label},
                   resume=earlier)
    out = folder / "eval" / f"{attempt.id}.json"
    journal = path_for(out)
    store.eval_started(run_id, reservation, journal, earlier)
    with attempt.running() as log:
        out.parent.mkdir(parents=True, exist_ok=True)
        result = _measure(store, run_id, candidate_id, attempt.id, attempt.deadline_at, model, data, out, parts,
                          options, log)
    # Eval and the server ran in this process; the resident host's exit is part of the cleanup. The journal and the
    # result (complete, or what an attempt that stopped measured) are evidence beside the log.
    attempt.settle(evidence=(journal, out))
    attempt.finish(f"; its journal keeps what it measured: crit run --id {run_id} eval --candidate {candidate_id} "
                   f"--resume {attempt.id} goes on from there" if journal.is_file() else "")
    # A candidate created from a plan gets the plan's outcome: its forecast next to what Eval measured.
    planned = store.plan_outcome(run_id, candidate_id, attempt.id)
    return {"result": {"path": out.relative_to(root).as_posix(), "sha256": file_sha(out)}, "status": result["status"],
            **({"plan_outcome": f"plans/{planned['plan_id']}/outcomes/{attempt.id}.json"} if planned else {})}


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
            from crit_eval.quality.fixed import STORE
            from crit_eval.quality.packages import fetch, source
            # A published package's scores, when the data holds none and the source has them. Both quality modes judge
            # against the same scores: no run reads the allowance a package may carry beside them (protocol
            # quality.allowance), so its absence is no reason to fetch again.
            if source() is not None and not (data / STORE / "index.json").is_file():
                fetch(model, data, source(), log)
            with own_allocator_env():
                how = score_data(model, data, log, check)
            log(f"quality: reference scores of {how['positions']:,} positions {how['method']}"
                + (f" in {how['seconds']:.0f} s" if how["method"] == "scored" else ""))
        with served.until(deadline_at):
            # Eval closes every engine it opens, also when a part fails.
            result = run_eval(served, model, data, out, {}, parts, log=log, **options)
        if result.get("aborted"):
            raise ValueError("Eval aborted the round: " + result["aborted"]["detail"])
    return result
