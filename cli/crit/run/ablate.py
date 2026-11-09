"""crit run ... ablate: a candidate built from plugins (docs/PLUGINS.md) served once through crit serve and built
again in the same resident host with each variant of its plugins: as built, at its plain base, without each plugin
and with each alone (crit_plugins.ablation.variants). Every variant serves the run's cases for a short bench (crit run
measure's requests, warmed until the engine runs steady) and runs crit smoke, and the report says what each plugin
adds, costs or breaks before Eval spends a round on the engine. Development evidence under a check attempt: it
records no check and no benchmark.
"""
from pathlib import Path
import statistics
import sys
import time

from .attempt import Attempt, Expired, packaged, remaining, unsealed
from .measure import Sampler, _case, _wave, case_body, case_request, run_gpus
from .records import inputs, load_run, read_json, write_json
from .served import Served

ROOT = Path(__file__).resolve().parents[3]
# A variant's bench: the run's cases, warmed (_warm), at this many repetitions, of which it keeps the median. With
# two the median is their mean, and one cold request moves the variant.
REPETITIONS = 3
# An engine built again in the host runs its first requests cold: one variant of a Qwen3.8-27B engine (RTX 5090)
# decoded at 91, 98 and 130 tok/s over its three repetitions after a single warmup. So a variant's warmup repeats each
# case until two requests in a row take the same time per decode round, within STEADY, and stops at WARMUPS requests.
STEADY, WARMUPS = 0.03, 8
FOLDER = "plugins"   # where a package keeps its plugins, as crit_plugins.Registry finds them


def _import(name):
    try:
        return __import__(name, fromlist=["_"])
    except ImportError:   # a checkout installed before the folder existed
        sys.path.insert(0, str(ROOT / {"crit_plugins": "core/python", "crit_smoke": "smoke"}[name.split(".")[0]]))
        return __import__(name, fromlist=["_"])


def ablate(store, run_id, candidate_id, seconds=None, names=None, repetitions=REPETITIONS, smoke=True,
           smoke_tests=None, noise=None):
    """Serve each variant of the candidate's plugins and measure it; keep the report as
    candidates/ID/logs/ablation-ATTEMPT.json and return {attempt, report, variants, blame}."""
    ablation = _import("crit_plugins.ablation")
    assemble = _import("crit_plugins.assemble")
    noise = ablation.NOISE if noise is None else noise
    root = store.directory(run_id)
    run = load_run(root)
    folder = unsealed(root, candidate_id, "The candidate is sealed: ablate its plugins before sealing it")
    packaged(folder, "ablating", "crit run ablate serves")
    configuration = read_json(folder / "candidate.json")["configuration"]
    selection = configuration.get("plugins")
    if selection is None:
        raise ValueError("The candidate's configuration has no plugins: an engine built from plugins names the one "
                         "filling each slot in configuration.plugins (docs/PLUGINS.md)")
    plugins = folder / "package" / FOLDER
    registry = assemble.Registry.discover(plugins) if plugins.is_dir() else None
    found = ablation.variants(selection, registry, names=set(names) if names else None)
    resolved = inputs(root, run)
    cases = [{"case_id": case["case_id"],
              "request": case_request(root, resolved["case_manifests"][case["case_id"]]),
              **{key: case["workload"][key] for key in ("prompt_tokens", "output_tokens", "concurrency")},
              "warm": case["workload"]["warmup_runs"] > 0, "warmup_runs": 0, "repetitions": repetitions}
             for case in run["spec"]["cases"] if case["workload"]["mode"] == "resident"]
    kv_tokens = run["spec"]["hardware_constraints"]["kv_tokens"]
    attempt = Attempt(store, run_id, candidate_id, "check", seconds, "ablate", "crit run ablate", "The ablation")
    rows = []
    with attempt.running() as log:
        log(f"crit run ablate: {len(found)} variants of {', '.join(ablation.toggles(selection)) or 'no plugin'}")
        from crit_core.machine import engine_lock
        served = Served(store, run_id, candidate_id, attempt.id, kv_tokens, log, "The ablation",
                        timeout=min(86400, max(1.0, remaining(attempt.deadline_at))))
        with engine_lock(f"crit run ablate {run_id} {candidate_id}",
                         deadline=time.monotonic() + remaining(attempt.deadline_at)):
            with served.until(attempt.deadline_at):
                for variant in found:
                    rows.append(_variant(store, run_id, served, variant, cases, smoke, smoke_tests, log))
    blamed = ablation.blame(rows, selection, noise) if rows else []
    report = folder / "logs" / f"ablation-{attempt.id}.json"
    write_json(report, {"attempt_id": attempt.id, "plugins": selection, "repetitions": repetitions, "noise": noise,
                        "variants": rows, "blame": blamed}, immutable=True)
    for row in blamed:
        print("crit run ablate: " + ablation.describe(row), file=sys.stderr)
    result = attempt.settle(evidence=(report,))
    attempt.finish()
    return {"attempt": result, "report": report.relative_to(root).as_posix(),
            "variants": [{k: row.get(k) for k in ("name", "state", "kpis", "detail")} for row in rows],
            "blame": blamed}


def _round_seconds(wave):
    """A wave's median time per decode round (per token where crit serve reports no rounds), None when no reply
    decoded: a plugin's speed, apart from the tokens a draft gets accepted."""
    times = []
    for _, first, last, usage, telemetry in wave:
        tokens = usage["completion_tokens"] - 1
        if tokens > 0 and last > first:
            times.append((last - first) / tokens * (telemetry.get("tokens_per_round") or 1))
    return statistics.median(times) if times else None


def _warm(port, case, timeout):
    """Send the case until two requests in a row take the same time per decode round (within STEADY), at most
    WARMUPS of them; returns how many it sent. A case whose workload has no warmup is not warmed."""
    if not case["warm"]:
        return 0
    body, previous = case_body(case), None
    for sent in range(1, WARMUPS + 1):
        current = _round_seconds(_wave(port, body, timeout, case.get("concurrency", 1)))
        if current is None or previous is not None and abs(current - previous) <= STEADY * previous:
            return sent
        previous = current
    return WARMUPS


def _median(metrics, name):
    values = (metrics.get(name) or {}).get("samples") or []
    return statistics.median(values) if values else None


def _variant(store, run_id, served, variant, cases, smoke, smoke_tests, log):
    """One variant served, checked to run the plugins it names, benched and smoke-tested."""
    use = variant["plugins"]["use"]
    row = {"name": variant["name"], "use": use, **({"also": variant["also"]} if variant.get("also") else {})}
    shown = ", ".join(f"{slot}={plugin_id or '-'}" for slot, plugin_id in use.items())
    if "error" in variant:
        log(f"crit run ablate: {variant['name']} ({shown}) is refused by its plugins: {variant['error']}")
        return dict(row, state="refused", detail=variant["error"])
    log(f"crit run ablate: {variant['name']}: {shown}")
    try:
        started = time.perf_counter()
        engine, _, port = served.open(configuration={"plugins": variant["plugins"]})
        load_ms = (time.perf_counter() - started) * 1000
        smoke_module = _import("crit_smoke")
        client = smoke_module.Client(f"http://127.0.0.1:{port}", timeout=min(served.timeout, 120))
        reported = client.facts().get("plugins")
        if reported != use:
            raise ValueError(f"the engine reports the plugins {reported}, the variant asked for {use}: assemble "
                             "configuration.plugins with crit_plugins.assemble and report assembly.facts() in the "
                             "engine's facts")
        sampler = Sampler(engine.host_pid(), run_gpus(store, run_id))
        measured, warmups = {}, {}
        try:
            for case in cases:
                warmups[case["case_id"]] = _warm(port, case, served.timeout)
                log(f"{case['case_id']}: warm after {warmups[case['case_id']]} requests")
                measured[case["case_id"]] = _case(port, case, load_ms, sampler, served.timeout, log, engine.capacity())
        finally:
            sampler.stop()
        per_case = {case_id: {"decode_tps": _median(m["metrics"], "decode_tps"),
                              "prefill_tps": _median(m["metrics"], "prefill_tps"),
                              "ttft_ms": _median(m["metrics"], "ttft_ms"),
                              "accepted_tokens_per_round": _median(m["metrics"], "accepted_tokens_per_round"),
                              "warmups": warmups[case_id]}
                    for case_id, m in measured.items()}
        kpis = _import("crit_plugins.ablation").summary(per_case)
        row.update(state="measured", load_ms=round(load_ms, 1), cases=per_case, kpis=kpis)
        if smoke:
            items = smoke_module.run(client, set(smoke_tests or ()), log=lambda line: log("crit smoke: " + line.strip()))
            row["smoke"] = {"passed": sum(i["state"] == "passed" for i in items),
                            "failed": [i["name"] for i in items if i["state"] == "failed"],
                            "skipped": sum(i["state"] == "skipped" for i in items),
                            "items": items}
    except Exception as error:   # a variant that cannot load or serve fails, and the next one is served again
        if served.expired or isinstance(error, Expired):
            raise
        log(f"crit run ablate: {variant['name']} failed: {type(error).__name__}: {error}")
        return dict(row, state="failed", detail=f"{type(error).__name__}: {error}"[:2000])
    shown = ", ".join(f"{kpi} {value:.1f}" for kpi, value in row["kpis"].items() if value is not None)
    log(f"crit run ablate: {variant['name']}: {shown or 'no speed measured'}" + (
        f"; smoke {row['smoke']['passed']} passed, {len(row['smoke']['failed'])} failed"
        + (f" ({', '.join(row['smoke']['failed'])})" if row["smoke"]["failed"] else "") if "smoke" in row else ""))
    return row
