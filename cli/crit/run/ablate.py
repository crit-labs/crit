"""crit run ... ablate: a candidate built from plugins (docs/PLUGINS.md) served once through crit serve and built
again in the same resident host with each variant of its plugins: as built, at its plain base, without each plugin
and with each alone (crit_plugins.ablation.variants). Every variant serves the run's cases for a short bench (crit run
measure's requests, warmed until the engine runs steady) and runs crit smoke, and the report says what each plugin
adds, costs or breaks before Eval spends a round on the engine. With crit-eval's prepared data (--data) each variant
also decodes Eval's speed cases and held-out prompts at the model's sampling, each warmed as the run's cases are, then
one request per seed: a plugin whose
effect depends on the text (drafts, copies, draft widths) is measured on the text Eval scores, not on the run's cases
alone. Option variants (--set PLUGIN.OPTION=VALUE) serve the full selection with options changed: an in-process A/B on
the same cases; exactness=LEVEL (or exactness.prompt=LEVEL, exactness.decode=LEVEL) among them serves the engine
asking that level of its rows, so the report says what a level costs or gains. With --rounds N every variant is
served N times, in alternating order (as listed, then reversed), and each comparison is the median of its rounds'
paired changes, as crit eval ab pairs two engines: the device's drift between variants served far apart cancels, so a
change of 1% is told from noise where every round shows it (crit_plugins.ablation.PAIRED_NOISE). Development evidence
under a check attempt: it records no check and no benchmark.

An external candidate (crit.run.external) has no plugins to switch: it is ablated only on the options its
configuration declares (configuration.engine.options), each a flag of its launch command. --set NAME=VALUE serves it
with those options beside it as built; without --set each declared switch is served flipped. An engine that declares
no option has no ablation, and its report says so.
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
# Seeds of each Eval case (--data): a sampled reply's tokens per round move with what it samples, so a drafting
# engine's decode on one seed moves by several percent between seeds.
SEEDS = 3
# An engine built again in the host runs its first requests cold: one variant of a dense 27B engine (RTX 5090)
# decoded at 91, 98 and 130 tok/s over its three repetitions after a single warmup. So a variant's warmup repeats each
# case until two requests in a row take the same time per decode round, within STEADY, and stops at WARMUPS requests.
STEADY, WARMUPS = 0.03, 8
FOLDER = "plugins"   # where a package keeps its plugins, as crit_plugins.Registry finds them


def _import(name):
    try:
        return __import__(name, fromlist=["_"])
    except ImportError:   # a checkout installed before the folder existed
        sys.path.insert(0, str(ROOT / {"crit_plugins": "inference/core/python", "crit_smoke": "smoke"}[name.split(".")[0]]))
        return __import__(name, fromlist=["_"])


def eval_cases(data, seeds=None):
    """crit-eval's speed cases and held-out prompts from its prepared data (DATA/prepared.json) as ablation cases:
    each request its messages (and tools) at the prepared sampling, `seeds` measured requests with seeds 1 to
    `seeds`, its output tokens as Eval decodes them, named eval:ID (eval:heldout:ID). Their prompt tokens are crit
    serve's rendering of the messages, not checked against Eval's."""
    import json
    seeds = SEEDS if seeds is None else seeds
    prepared = json.loads((Path(data) / "prepared.json").read_text(encoding="utf-8"))
    # Data prepared before crit-eval-v7 names no sampling: the protocol default (temperature 1.0), never greedy.
    sampling = dict(prepared.get("sampling") or {"temperature": 1.0})
    out = []
    for prefix, rows in (("eval:", prepared.get("speed") or []), ("eval:heldout:", prepared.get("heldout") or [])):
        for row in rows:
            if not row.get("messages"):
                continue
            request = {"messages": row["messages"], **({"tools": row["tools"]} if row.get("tools") else {}), **sampling}
            # Warmed as the run's cases are (_warm): an engine's first requests of a prompt it has not served run
            # cold (expert residency follows the content), so unwarmed, the first seed of every case measured slow (in
            # an expert-offload build's ablations rep 1 decoded at 0.87 of its case's median, rep 3 at 1.11).
            out.append({"case_id": prefix + row["id"], "request": request, "prompt_tokens": None,
                        "output_tokens": int(row["output_tokens"]), "concurrency": 1, "warm": True,
                        "warmup_runs": 0, "repetitions": seeds, "seeds": list(range(1, seeds + 1))})
    if not out:
        raise ValueError(f"{data}: prepared.json holds no speed case or held-out prompt with messages")
    return out


def ablate(store, run_id, candidate_id, seconds=None, names=None, repetitions=REPETITIONS, smoke=True,
           smoke_tests=None, noise=None, data=None, seeds=None, sets=None, rounds=1, quality=False):
    """Serve each variant of the candidate's plugins and measure it; keep the report as
    candidates/ID/logs/ablation-ATTEMPT.json and return {attempt, report, variants, blame}. data: crit-eval's
    prepared data, whose speed cases and held-out prompts each variant also decodes (eval_cases, `seeds` seeds each).
    sets: option variants (crit_plugins.ablation.option_variants), served with the full selection and, unless `names`
    asks for others, instead of the plugin variants. rounds: every variant served this many times in alternating
    order (_rounds), crit smoke in the first round alone; noise defaults to ablation.PAIRED_NOISE then. quality:
    with `data`, each variant's quality texts also scored against the reference by the run's drift budget, in the
    first round (_quality)."""
    ablation = _import("crit_plugins.ablation")
    assemble = _import("crit_plugins.assemble")
    rounds = _check_rounds(rounds)
    root = store.directory(run_id)
    run = load_run(root)
    folder = unsealed(root, candidate_id, "The candidate is sealed and takes no attempt: ablate a candidate before "
                      "its seal (crit run ... next says so at the step before it). To sweep this engine's options "
                      "now, create the next candidate from its plan with this one's source and ablate that one "
                      "before its Eval")
    packaged(folder, "ablating", "crit run ablate serves")
    from .external import candidate_spec
    spec = candidate_spec(folder)
    if spec is not None:
        return _external(store, run_id, candidate_id, spec, seconds, names, repetitions, smoke, smoke_tests, noise,
                         data, seeds, sets, rounds)
    configuration = read_json(folder / "candidate.json")["configuration"]
    selection = configuration.get("plugins")
    if selection is None:
        raise ValueError("The candidate's configuration has no plugins: an engine built from plugins names the one "
                         "filling each slot in configuration.plugins (docs/PLUGINS.md)")
    plugins = folder / "package" / FOLDER
    registry = assemble.Registry.discover(plugins) if plugins.is_dir() else None
    if sets:
        found = ablation.variants(selection, registry, names=set(names) if names else {"full"})
        found += ablation.option_variants(selection, sets, registry)
        # A variant that sets the engine's exactness is served asking those levels over the candidate's own (which a
        # candidate recorded before crit stated them takes from the run's quality mode).
        if any(variant.get("exactness") for variant in found):
            from crit_plugins.exactness import setting
            from .records import quality_mode
            own = setting(configuration.get("exactness"), quality_mode(run["spec"]["correctness"]))
            for variant in found:
                if variant.get("exactness"):
                    variant["exactness"] = setting({**own, **variant["exactness"]})
    else:
        found = ablation.variants(selection, registry, names=set(names) if names else None)
    cases = _cases(root, run, repetitions, data, seeds)
    kv_tokens = run["spec"]["hardware_constraints"]["kv_tokens"]
    if quality and data is None:
        raise ValueError("--quality scores Eval's quality texts: give --data, crit-eval's prepared data with the "
                         "reference's scores (crit eval reference)")
    attempt = Attempt(store, run_id, candidate_id, "check", seconds, "ablate", "crit run ablate", "The ablation")
    judge = _quality_terms(store, root, run, attempt, data, kv_tokens) if quality else None
    measured = {}
    with attempt.running() as log:
        log(f"crit run ablate: {len(found)} variants of {', '.join(ablation.toggles(selection)) or 'no plugin'}"
            + (f", {rounds} rounds in alternating order" if rounds > 1 else ""))
        from crit_core.machine import engine_lock
        served = Served(store, run_id, candidate_id, attempt.id, kv_tokens, log, "The ablation",
                        timeout=min(86400, max(1.0, remaining(attempt.deadline_at))))
        with engine_lock(f"crit run ablate {run_id} {candidate_id}",
                         deadline=time.monotonic() + remaining(attempt.deadline_at)):
            with served.until(attempt.deadline_at):
                _rounds(found, rounds, measured, log, lambda variant, first: _variant(
                    store, run_id, served, variant, cases, smoke and first, smoke_tests, log, judge if first else None))
    report = folder / "logs" / f"ablation-{attempt.id}.json"
    rows, blamed = [], []

    def write():
        nonlocal rows, blamed, noise
        rows = [ablation.combine(measured[variant["name"]]) for variant in found if measured.get(variant["name"])]
        noise = _noise(ablation, measured, noise)
        # Option variants alone (--set without --variant) serve no plugin variant, so no plugin is blamed.
        plugins_blamed = ablation.blame(rows, selection, noise) if rows and (names or not sets) else []
        blamed = plugins_blamed + (ablation.settings(rows, noise) if rows else [])
        write_json(report, {"attempt_id": attempt.id, "plugins": selection, "repetitions": repetitions,
                            "noise": noise, "rounds": rounds,
                            **({"eval_data": str(data), "seeds": seeds or SEEDS} if data is not None else {}),
                            "variants": rows, "blame": blamed}, immutable=True)
        for row in blamed:
            print("crit run ablate: " + ablation.describe(row), file=sys.stderr)
        _one_pass(rounds, blamed)

    result = _settled(attempt, write, report)
    return {"attempt": result, "report": report.relative_to(root).as_posix(),
            "variants": [{k: row.get(k) for k in ("name", "state", "kpis", "detail")} for row in rows],
            "blame": blamed}


def _settled(attempt, write, report):
    """Write the report (write()), then settle the attempt and finish it, whatever the writing raises: an attempt left
    unsettled holds the run, and every later serve of it is refused until it settles. A report that could not be
    written fails the attempt with its error and keeps no evidence of it."""
    try:
        write()
    except BaseException as exc:   # recorded and settled, then raised
        if attempt.outcome == "completed":
            attempt.ended(exc)
        attempt.settle()
        attempt.finish()
        raise
    result = attempt.settle(evidence=(report,))
    attempt.finish()
    return result


def _one_pass(rounds, blamed):
    """After a single pass that judged a change: say it is not confirmed. One pass is judged against NOISE, and an
    engine that warms slowly (an expert cache filling, a drafter's acceptance settling) moves by more between passes."""
    changed = [row["plugin"] for row in blamed if row.get("verdict") in ("pays", "slows", "trades")]
    if rounds == 1 and changed:
        print(f"crit run ablate: one pass judged {', '.join(changed)}; a pass of a slowly warming engine can differ "
              "from the next by more than the noise it is judged against, so confirm a change the next plan acts on "
              "with --rounds 2 (--variant to serve just those)", file=sys.stderr)


def _check_rounds(rounds):
    rounds = int(rounds)
    if rounds < 1:
        raise ValueError("--rounds is at least 1")
    return rounds


def _noise(ablation, measured, noise):
    """The noise the verdicts read unless given: PAIRED_NOISE where variants were compared over two rounds or more,
    else NOISE (also for an attempt with --rounds that ran out of time in its first round)."""
    if noise is not None:
        return noise
    return ablation.PAIRED_NOISE if any(len(rows) > 1 for rows in measured.values()) else ablation.NOISE


def _rounds(found, rounds, measured, log, serve):
    """Serve every variant `rounds` times, as crit eval ab alternates two engines: in the listed order, then reversed,
    and so on, so no variant always follows the same one. Each served row goes to measured[name] as it is made (an
    attempt that runs out of time keeps the rounds it served); serve(variant, first) serves one, `first` in the first
    round alone. A variant refused or failed in a round is not served again: its row says why."""
    for index in range(rounds):
        order = found if index % 2 == 0 else found[::-1]
        if rounds > 1:
            log(f"crit run ablate: round {index + 1} of {rounds}: " + ", ".join(variant["name"] for variant in order))
        for variant in order:
            rows = measured.get(variant["name"]) or []
            if rows and rows[-1].get("state") != "measured":
                continue
            # The row goes in once served: a variant interrupted while served (Ctrl+C, SIGTERM) leaves no empty
            # entry, which the report could not combine into a row.
            row = serve(variant, index == 0)
            measured.setdefault(variant["name"], []).append(row)


def _cases(root, run, repetitions, data, seeds):
    """A variant's bench: the run's resident cases, and with `data` Eval's speed cases and held-out prompts."""
    resolved = inputs(root, run)
    cases = [{"case_id": case["case_id"],
              "request": case_request(root, resolved["case_manifests"][case["case_id"]]),
              **{key: case["workload"][key] for key in ("prompt_tokens", "output_tokens", "concurrency")},
              "warm": case["workload"]["warmup_runs"] > 0, "warmup_runs": 0, "repetitions": repetitions}
             for case in run["spec"]["cases"] if case["workload"]["mode"] == "resident"]
    return cases + (eval_cases(data, seeds) if data is not None else [])


def option_sets(spec, sets):
    """An external engine's variants: as built ("full"), then one per --set text (NAME=VALUE pairs joined by ";",
    values JSON or else strings), or, without --set, each declared switch flipped. A variant naming an option the
    engine does not declare carries its `error`."""
    import json
    out = [{"name": "full", "options": {}}]
    if not sets:
        sets = [f"{name}={json.dumps(not option.get('default'))}" for name, option in spec["options"].items()
                if option.get("switch")]
    for text in sets:
        variant, given = {"name": f"set:{text}", "options": {}}, {}
        for pair in [part.strip() for part in text.split(";") if part.strip()]:
            key, sep, value = pair.partition("=")
            if not sep or not key.strip():
                raise ValueError(f"--set {pair!r}: give OPTION=VALUE, an option the engine declares")
            try:
                given[key.strip()] = json.loads(value)
            except ValueError:
                given[key.strip()] = value
        unknown = sorted(set(given) - set(spec["options"]))
        if unknown:
            variant["error"] = (f"the engine declares no option {', '.join(unknown)} (configuration.engine.options: "
                                + (", ".join(sorted(spec["options"])) or "none") + ")")
        variant["options"] = given
        out.append(variant)
    return out


def _external(store, run_id, candidate_id, spec, seconds, names, repetitions, smoke, smoke_tests, noise, data, seeds,
              sets, rounds=1):
    """crit run ablate on an external candidate: its declared options served A/B (option_sets), or, when it declares
    none, a report that it has no ablation."""
    ablation = _import("crit_plugins.ablation")
    rounds = _check_rounds(rounds)
    root = store.directory(run_id)
    run = load_run(root)
    folder = root / "candidates" / candidate_id
    attempt = Attempt(store, run_id, candidate_id, "check", seconds, "ablate", "crit run ablate", "The ablation")
    report = folder / "logs" / f"ablation-{attempt.id}.json"
    head = {"attempt_id": attempt.id, "engine": "external", "options": spec["options"], "repetitions": repetitions,
            "noise": noise, "rounds": rounds,
            **({"eval_data": str(data), "seeds": seeds or SEEDS} if data is not None else {})}
    measured, found, blamed = {}, [], []
    with attempt.running() as log:
        if not spec["options"]:
            reason = ("the external engine declares no option (configuration.engine.options), so crit cannot switch "
                      "any of its levers: this candidate has no ablation")
            log("crit run ablate: " + reason)
            head.update(ablation="unavailable", reason=reason)
        else:
            found = option_sets(spec, sets)
            if names:
                found = [variant for variant in found if variant["name"] in set(names)]
            log(f"crit run ablate: {len(found)} variants of the options {', '.join(spec['options'])}"
                + (f", {rounds} rounds in alternating order" if rounds > 1 else ""))
            cases = _cases(root, run, repetitions, data, seeds)
            from crit_core.machine import engine_lock
            from .external import serving
            served = serving(store, run_id, candidate_id, attempt, run["spec"]["hardware_constraints"]["kv_tokens"],
                             log, "The ablation", timeout=min(86400, max(1.0, remaining(attempt.deadline_at))))
            with engine_lock(f"crit run ablate {run_id} {candidate_id}",
                             deadline=time.monotonic() + remaining(attempt.deadline_at)):
                with served.until(attempt.deadline_at):
                    _rounds(found, rounds, measured, log, lambda variant, first: _external_variant(
                        store, run_id, served, variant, cases, smoke and first, smoke_tests, log))
    rows = []

    def write():
        nonlocal rows, blamed, noise
        rows = [ablation.combine(measured[variant["name"]]) for variant in found if measured.get(variant["name"])]
        noise = _noise(ablation, measured, noise)
        if rows:
            blamed = ablation.settings(rows, noise)
        write_json(report, {**head, "noise": noise, "variants": rows, "blame": blamed}, immutable=True)
        for row in blamed:
            print("crit run ablate: " + ablation.describe(row), file=sys.stderr)
        _one_pass(rounds, blamed)

    result = _settled(attempt, write, report)
    return {"attempt": result, "report": report.relative_to(root).as_posix(),
            **({"ablation": "unavailable", "reason": head["reason"]} if "reason" in head else {}),
            "variants": [{k: row.get(k) for k in ("name", "state", "kpis", "detail")} for row in rows],
            "blame": blamed}


def _external_variant(store, run_id, served, variant, cases, smoke, smoke_tests, log):
    """One option variant of an external engine, started from its launch command with those options, benched and
    smoke-tested."""
    row = {"name": variant["name"], "options": variant["options"]}
    if "error" in variant:
        log(f"crit run ablate: {variant['name']} is refused: {variant['error']}")
        return dict(row, state="refused", detail=variant["error"])
    log(f"crit run ablate: {variant['name']}")
    try:
        started = time.perf_counter()
        engine, _, port = served.open(configuration={"options": variant["options"]})
        load_ms = (time.perf_counter() - started) * 1000
        return _bench(store, run_id, served, engine, port, load_ms, cases, smoke, smoke_tests, log, row)
    except Exception as error:   # a variant that cannot start or serve fails, and the next one is started again
        if served.expired or isinstance(error, Expired):
            raise
        log(f"crit run ablate: {variant['name']} failed: {type(error).__name__}: {error}")
        return _recheck(served, dict(row, state="failed", detail=f"{type(error).__name__}: {error}"[:2000]), log)


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
    body, previous = case_body(case, port), None
    for sent in range(1, WARMUPS + 1):
        current = _round_seconds(_wave(port, body, timeout, case.get("concurrency", 1)))
        if current is None or previous is not None and abs(current - previous) <= STEADY * previous:
            return sent
        previous = current
    return WARMUPS


def _median(metrics, name):
    values = (metrics.get(name) or {}).get("samples") or []
    return statistics.median(values) if values else None


def _served_sampling(cases, supported):
    """The Eval cases' requests with the sampling fields the served engine applies (its card's
    supported_sampling_parameters; temperature always): a package that declares no seed decodes each repetition as an
    unseeded draw, and one without top_p or top_k samples at the fields it has. Other cases are unchanged."""
    if supported is None:
        return cases
    supported = set(supported) | {"temperature"}
    out = []
    for case in cases:
        if not case.get("seeds"):
            out.append(case)
            continue
        request = {key: value for key, value in case["request"].items()
                   if key in ("messages", "tools") or key in supported}
        out.append({**case, "request": request, **({} if "seed" in supported else {"seeds": None})})
    return out


def _variant(store, run_id, served, variant, cases, smoke, smoke_tests, log, judge=None):
    """One variant served, checked to run the plugins it names, benched and smoke-tested, and with `judge`
    (_quality_terms) its quality texts scored (_quality)."""
    use = variant["plugins"]["use"]
    row = {"name": variant["name"], "use": use, **({"also": variant["also"]} if variant.get("also") else {}),
           **({"approximates": variant["approximates"]} if variant.get("approximates") else {})}
    for found in variant.get("approximates") or ():
        log(f"crit run ablate: {variant['name']} sets {found['plugin']}.{found['option']}={found['value']!r}, which "
            f"approximates the model ({found['what']}): its speed here does not judge it; its drift against the "
            "reference scores does (Eval's quality, the pre-check's drift item)")
    shown = ", ".join(f"{slot}={plugin_id or '-'}" for slot, plugin_id in use.items())
    if "error" in variant:
        log(f"crit run ablate: {variant['name']} ({shown}) is refused by its plugins: {variant['error']}")
        return dict(row, state="refused", detail=variant["error"])
    log(f"crit run ablate: {variant['name']}: {shown}")
    closes = len(served.closes)
    try:
        started = time.perf_counter()
        # A variant is asked no prompt activations: it measures what other plugins or options serve (its facts say
        # which), where the candidate's configuration asks its plan's.
        engine, api, port = served.open(configuration={"plugins": variant["plugins"], "prompt_activations": None, **(
            {"exactness": variant["exactness"]} if variant.get("exactness") else {})})
        load_ms = (time.perf_counter() - started) * 1000
        smoke_module = _import("crit_smoke")
        client = smoke_module.Client(f"http://127.0.0.1:{port}", timeout=min(served.timeout, 120))
        facts = client.facts()
        reported = facts.get("plugins")
        if facts.get("exactness"):   # what the variant was asked and what each of its plugins keeps
            row["exactness"] = facts["exactness"]
        if reported != use:
            raise ValueError(f"the engine reports the plugins {reported}, the variant asked for {use}: assemble "
                             "configuration.plugins with crit_plugins.assemble and report assembly.facts() in the "
                             "engine's facts")
        row = _bench(store, run_id, served, engine, port, load_ms, cases, smoke, smoke_tests, log, row, client)
        if judge is not None:
            row["quality"] = _quality(judge, port, api.model, log, row["name"])
    except Exception as error:   # a variant that cannot load or serve fails, and the next one is served again
        if served.expired or isinstance(error, Expired):
            raise
        log(f"crit run ablate: {variant['name']} failed: {type(error).__name__}: {error}")
        failed = dict(row, state="failed", detail=f"{type(error).__name__}: {error}"[:2000])
        if row["name"] in ("full", "base"):
            return _closed(served, closes, failed)
        return _recheck(served, _closed(served, closes, failed), log)
    return _closed(served, closes, row)


def _recheck(served, row, log):
    """A variant that failed: the engine as built is served again right after it. Where that fails too, the failure was
    not the option's (the machine, the device, the host: an infrastructure failure), so the row says so
    (`infrastructure`) and the verdicts blame nothing on it (crit_plugins.ablation: unmeasured)."""
    try:
        served.open()
    except Exception as error:   # noqa: BLE001 - recorded: the variant's failure is then not the option's doing
        if served.expired or isinstance(error, Expired):
            raise
        row["infrastructure"] = (f"the engine as built failed right after it too ({type(error).__name__}: "
                                 f"{error})")[:2000]
        log(f"crit run ablate: {row['name']}: the engine as built failed right after it too, so its failure is not "
            "blamed on it")
    return row


def _closed(served, closes, row):
    """A variant's row with how the engine served before it closed (crit serve's last_close, read by Served.open):
    "previous_close" {returned, judged, message}. A close that kept memory refuses the variant's load (Core), so its
    row fails with the report, and the previous variant is the one that leaked."""
    if len(served.closes) > closes:
        row["previous_close"] = {key: served.closes[closes].get(key) for key in ("returned", "judged", "message")}
    return row


def _quality_terms(store, root, run, attempt, data, kv_tokens):
    """What scoring a variant's quality texts needs: the protocol, the prepared data, the run's model, quality mode and
    drift budget (its segment's amendments included) and context."""
    from crit_eval import EVAL_DIR, load_json
    from crit_eval.quality.prepare import PREPARED
    from .records import quality_mode, user_quality_budget
    from .store import amendments
    # As crit run eval: only the limits the user set, so the checkpoint's stored allowance applies where none is set.
    budget = user_quality_budget(run["spec"]["correctness"], amendments(root, attempt.reservation["segment_id"])) or None
    return {"protocol": load_json(Path(EVAL_DIR) / "protocol.json"), "prepared": load_json(Path(data) / PREPARED),
            "data": Path(data), "model": run["spec"]["model_path"], "budget": budget, "kv_tokens": kv_tokens,
            "mode": quality_mode(run["spec"]["correctness"])}


def _quality(judge, port, model_name, log, name):
    """The served variant's quality texts scored against the reference by the run's drift budget, as the gate
    pre-check scores them (crit_eval.run.precheck.quality_item): {state, detail, budget_used, worst_set, baseline: what
    a plain 16-bit engine of the model uses of the same budget on that set, reported, never judged}. An option
    that approximates the model (fewer routed experts, a narrower KV) is judged by this, not by its speed."""
    from crit_eval.run.adapters import open_engine
    from crit_eval.run.chat import Server
    from crit_eval.run.precheck import quality_item
    module = Server(f"http://127.0.0.1:{port}/v1", judge["prepared"], model_name)
    engine, info = open_engine(module, judge["model"], {}, speculative=True, context_tokens=judge["kv_tokens"],
                               exact_prompts=True)
    item = quality_item(engine, info, judge["prepared"], judge["protocol"], judge["data"], judge["mode"],
                        judge["budget"], lambda line: log("crit run ablate: " + line))
    used = item.get("budget_used")
    sixteen = (item.get("baseline") or {}).get("budget_used")
    log(f"crit run ablate: {name}: quality {item['state']}"
        + (f", {used:.0%} of the drift budget (worst {item.get('worst_set')})" if used is not None else "")
        + (f", the 16-bit baseline {sixteen:.0%} of it there" if sixteen is not None else ""))
    return {key: item[key] for key in ("state", "detail", "budget_used", "worst_set", "baseline") if key in item}


def _bench(store, run_id, served, engine, port, load_ms, cases, smoke, smoke_tests, log, row, client=None):
    """A served variant's short bench of `cases` and crit smoke, into `row`."""
    smoke_module = _import("crit_smoke")
    client = client or smoke_module.Client(f"http://127.0.0.1:{port}", timeout=min(served.timeout, 120))
    cases = _served_sampling(cases, client.card().get("supported_sampling_parameters"))
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
        from .recon import smoke_reference_of
        items = smoke_module.run(client, set(smoke_tests or ()), log=lambda line: log("crit smoke: " + line.strip()),
                                 reference=smoke_reference_of(store.directory(run_id)))
        row["smoke"] = {"passed": sum(i["state"] == "passed" for i in items),
                        "failed": [i["name"] for i in items if i["state"] == "failed"],
                        "skipped": sum(i["state"] == "skipped" for i in items),
                        "items": items}
    shown = ", ".join(f"{kpi} {value:.1f}" for kpi, value in row["kpis"].items() if value is not None)
    log(f"crit run ablate: {row['name']}: {shown or 'no speed measured'}" + (
        f"; smoke {row['smoke']['passed']} passed, {len(row['smoke']['failed'])} failed"
        + (f" ({', '.join(row['smoke']['failed'])})" if row["smoke"]["failed"] else "") if "smoke" in row else ""))
    return row
