"""Run one engine through the suite: `crit eval run --adapter FILE --model DIR --data DIR --out FILE`."""
import datetime
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

from . import CONTRACT, EVAL_DIR, load_json, write_json
from . import headroom, metrics, status
from .ceiling import model_ceiling
from .context import GIB, curve_points, filled_prompt, lower_context, max_context
from .pin import CONTINUATIONS, load_continuations
from .prepare import PREPARED
from .reference import REFERENCE


def load_adapter(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"crit_eval_adapter_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    # A retained package is inventoried byte for byte: importing its adapter (and the modules it imports) must
    # not write __pycache__ into it.
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    if not hasattr(module, "create_eval_engine"):
        raise SystemExit(f"{path} does not export create_eval_engine(context); see docs/eval/ADAPTER.md")
    return module


def open_engine(module, model, config, speculative, context_tokens=None):
    """context_tokens: the context (prompt plus output) the engine is built for, when one is required."""
    context = {"contract": CONTRACT, "model": str(model), "config": dict(config), "speculative": speculative}
    if context_tokens:
        context["context_tokens"] = int(context_tokens)
    engine = module.create_eval_engine(context)
    info = engine.info()
    for key in ("name", "configuration", "max_context"):
        if key not in info:
            raise SystemExit(f"engine.info() is missing {key!r}")
    return engine, info


def _timed(call, prompt, output_tokens, expected=None):
    deliveries = []

    def on_tokens(tokens):
        deliveries.append((time.perf_counter(), list(tokens)))
    started = time.perf_counter()
    result = call(on_tokens)
    tokens = list(result["tokens"])
    streamed = [t for _, chunk in deliveries for t in chunk]
    if streamed != tokens:
        raise RuntimeError("tokens passed to on_tokens differ from the returned tokens")
    if len(tokens) != output_tokens:
        raise RuntimeError(f"engine returned {len(tokens)} tokens, expected {output_tokens} (stop tokens must be disabled)")
    if expected is not None and tokens != list(expected):
        diverge = next(i for i, (a, b) in enumerate(zip(tokens, expected)) if a != b)
        raise RuntimeError(f"replay emitted a token different from the pinned continuation at position {diverge}")
    first = next(t for t, chunk in deliveries if chunk)
    sample = metrics.speed_sample(len(prompt), tokens, started, first, deliveries[-1][0], result.get("rounds"))
    sample["engine_timing"] = result.get("timing")
    sample["accepted"] = result.get("accepted")
    # Optional engine-measured parts of the decode: time in verify passes and in draft steps (ms, whole request).
    for key in ("verify_ms", "draft_ms"):
        value = result.get(key)
        sample[key] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return tokens, sample


def timed_generate(engine, prompt, output_tokens):
    return _timed(lambda on_tokens: engine.generate(list(prompt), output_tokens, on_tokens), prompt, output_tokens)


def timed_plain(engine, prompt, output_tokens):
    """Plain one-token greedy decoding on a speculative engine that declares it (info "plain")."""
    return _timed(lambda on_tokens: engine.generate_plain(list(prompt), output_tokens, on_tokens), prompt, output_tokens)


def timed_replay(engine, prompt, continuation):
    """The engine's speculative loop on a fixed continuation: drafts are accepted while they match it and
    each round's correction is its next token, so every engine decodes the same text."""
    continuation = list(continuation)
    return _timed(lambda on_tokens: engine.replay(list(prompt), list(continuation), on_tokens), prompt,
                  len(continuation), expected=continuation)


class Disturbed(Exception):
    """A speed case's repetitions disagree far beyond any quiet engine's run-to-run spread: something else
    is using the device. The round stops instead of measuring the rest on a disturbed device."""


DISTURBED_MIN_REQUEST_MS = 100  # shorter requests vary with timer and scheduler jitter, not with the device


def check_disturbed(case_id, samples, limit):
    """Raise Disturbed when the measured decode speeds of one case spread (max - min over the median) by more
    than `limit`. Quiet rounds recorded at most about 10%; a disturbed one 37%."""
    measured = [s for s in samples if not s.get("warmup")]
    if not limit or not measured or metrics.median([s["request_ms"] for s in measured]) < DISTURBED_MIN_REQUEST_MS:
        return
    spread = headroom.spread([s["decode_tps"] for s in measured if s.get("decode_tps")])
    if spread is not None and spread > limit:
        raise Disturbed(f"speed {case_id}: decode varied {spread:.0%} between repetitions (limit {limit:.0%}); "
                        "the device was disturbed. Measure again when nothing else uses it")


SUMMARY_KEYS = ("decode_tps", "prefill_tps", "ttft_ms", "request_ms", "tokens_per_round", "round_ms")


def _summary(samples):
    return {key: metrics.median([s[key] for s in samples if s[key] is not None]) for key in SUMMARY_KEYS}


def decode_text(info, continuations):
    """Which text the decode KPIs are measured on: the pinned continuation when the data directory has
    one and the adapter can replay it, else each engine's own greedy output."""
    if continuations and info.get("replay") and info.get("speculative"):
        return {"source": "pinned", "sha256": continuations["sha256"], "pinned_by": continuations.get("engine")}
    reason = ("no pinned continuation in the data directory (crit eval pin)" if not continuations
              else "the adapter does not declare replay" if not info.get("replay") else "ordinary decoding")
    return {"source": "generated", "reason": reason}


def run_speed(engine, info, prepared, protocol, warmups, repetitions, log, continuations=None, disturbed_spread=None):
    """Speed cases. With a pinned continuation and a replaying adapter, every speed KPI comes from replays
    of the pinned text (identical for every engine) and two free generations record the engine's own
    output for repeatability and exactness; otherwise the KPIs come from free generation.
    disturbed_spread: stop the round (Disturbed) when a case's decode speeds spread by more than this."""
    cases = {}
    stop_ids = set(prepared["stop_ids"])
    text = decode_text(info, continuations)
    pinned = (continuations or {}).get("cases", {}) if text["source"] == "pinned" else {}
    for case in prepared["speed"]:
        prompt, n = case["prompt_ids"], case["output_tokens"]
        if len(prompt) + n > info["max_context"]:
            cases[case["id"]] = {"status": "skipped", "reason": f"needs {len(prompt) + n} tokens of context"}
            log(f"speed {case['id']}: skipped (context)")
            continue
        samples, outputs = [], []
        continuation = pinned.get(case["id"])
        if pinned and (continuation is None or len(continuation) != n):
            raise SystemExit(f"{CONTINUATIONS} has no {n}-token continuation for speed case {case['id']}")
        if continuation is not None:
            for _ in range(2):
                tokens, _ = timed_generate(engine, prompt, n)
                outputs.append(tokens)
            for i in range(warmups + repetitions):
                _, sample = timed_replay(engine, prompt, continuation)
                sample["warmup"] = i < warmups
                samples.append(sample)
        else:
            for i in range(warmups + repetitions):
                tokens, sample = timed_generate(engine, prompt, n)
                sample["warmup"] = i < warmups
                samples.append(sample)
                if i >= warmups:
                    outputs.append(tokens)
        check_disturbed(case["id"], samples, disturbed_spread)
        summary = _summary([s for s in samples if not s["warmup"]])
        crossed = [i for i, t in enumerate(outputs[0]) if t in stop_ids]
        cases[case["id"]] = {
            "status": "measured", "kind": case["kind"], "prompt_tokens": len(prompt), "output_tokens": n, **summary,
            "repeatable": all(o == outputs[0] for o in outputs),
            "stop_crossed": bool(crossed), "first_stop_index": crossed[0] if crossed else None,
            "tokens": outputs[0], "samples": samples,
        }
        log(f"speed {case['id']}: decode {summary['decode_tps']:.1f} tok/s"
            + (" (pinned continuation)" if continuation is not None else "")
            + f", prefill {summary['prefill_tps']:.0f} tok/s, tokens/round {summary['tokens_per_round'] or 0:.2f}, "
            f"round {summary['round_ms'] or 0:.2f} ms")
    measured = [c for c in cases.values() if c["status"] == "measured"]
    prefill = [cases[i]["prefill_tps"] for i in protocol["speed"].get("prefill_cases", [])
               if cases.get(i, {}).get("status") == "measured" and cases[i].get("prefill_tps")]
    return {
        "warmup_runs": warmups, "repetitions": repetitions, "decode_text": text,
        "decode_score": metrics.geometric_mean([c["decode_tps"] for c in measured]),
        "prefill_score": metrics.geometric_mean(prefill)
        if prefill and len(prefill) == len(protocol["speed"].get("prefill_cases", [])) else None,
        "ttft_score_ms": metrics.geometric_mean([c["ttft_ms"] for c in measured if c.get("ttft_ms")]),
        "complete": len(measured) == len(cases),
        "cases": cases,
    }


def run_heldout(engine, info, prepared, protocol, log):
    """The held-out prompts (protocol "heldout"): each decodes its own greedy output on the engine built for the
    required context. Data prepared before the held-out set has none; it is reported as missing."""
    settings = protocol["heldout"]
    if "heldout" not in prepared:
        log("held-out: not in the prepared data (prepare it again with crit eval prepare into a new data directory)")
        return {"status": "missing", "reason": "the prepared data has no held-out prompts; prepare it again"}
    prompts = {}
    for prompt in prepared["heldout"]:
        ids, n = prompt["prompt_ids"], prompt["output_tokens"]
        if len(ids) + n > info["max_context"]:
            prompts[prompt["id"]] = {"status": "skipped", "reason": f"needs {len(ids) + n} tokens of context"}
            continue
        samples = []
        for _ in range(int(settings["repetitions"])):
            _, sample = timed_generate(engine, ids, n)
            samples.append(sample)
        summary = _summary(samples)
        prompts[prompt["id"]] = {"status": "measured", "language": prompt["language"], "kind": prompt["kind"],
                                 "prompt_tokens": len(ids), "output_tokens": n, **summary, "samples": samples}
        log(f"held-out {prompt['id']} ({prompt['language']}): decode {summary['decode_tps']:.1f} tok/s, "
            f"tokens/round {summary['tokens_per_round'] or 0:.2f}")
    measured = [p for p in prompts.values() if p["status"] == "measured"]
    return {"status": "measured", "prompts": prompts,
            "decode_score": metrics.geometric_mean([p["decode_tps"] for p in measured]),
            "tokens_per_round": metrics.geometric_mean([p["tokens_per_round"] for p in measured if p.get("tokens_per_round")]),
            "complete": len(measured) == len(prompts)}


def native_identity(info):
    """sha256 and size of each native library the adapter declares it loaded (`native_libraries`)."""
    out = []
    for path in info.get("native_libraries") or []:
        entry = {"path": str(path)}
        try:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            entry.update(sha256=digest.hexdigest(), bytes=Path(path).stat().st_size)
        except OSError as error:
            entry["error"] = str(error)
        out.append(entry)
    return out


QUALITY_KEYS = ("ppl_ratio_max", "kld_mean_max", "kld_p99_max")
EXACTNESS_MODES = ("reported", "required")


def quality_thresholds(protocol, budget=None):
    """The protocol's drift budget, with a run's own limits (partial is fine) laid over it."""
    budget = budget or {}
    unknown = set(budget) - set(QUALITY_KEYS)
    if unknown:
        raise SystemExit(f"--quality-budget keys are {', '.join(QUALITY_KEYS)}; got {', '.join(sorted(unknown))}")
    for key, value in budget.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < (1.0 if key == "ppl_ratio_max" else 0.0):
            raise SystemExit(f"--quality-budget {key} must be a number of at least {1.0 if key == 'ppl_ratio_max' else 0.0}")
    return {**protocol["quality"]["thresholds"], **{k: float(v) for k, v in budget.items()}}


def run_quality(engine, info, prepared, reference, protocol, log, thresholds=None):
    paths = [p for p in protocol["quality"]["paths"] if p in info.get("score_paths", [])]
    by_id = {c["id"]: c["targets"] for c in reference["conversations"]}
    out = {"paths": {}, "unsupported": [p for p in protocol["quality"]["paths"] if p not in paths]}
    thresholds = thresholds or protocol["quality"]["thresholds"]
    out["thresholds"] = thresholds
    for path in paths:
        started = time.time()
        reference_rows, engine_rows = [], []
        for conversation in prepared["quality"]:
            targets = by_id[conversation["id"]]
            scored = engine.score(conversation["ids"], conversation["first_target"], path, [t["top_ids"] for t in targets])
            argmax = scored.get("argmax") or [None] * len(targets)
            for target, lp, query, best in zip(targets, scored["target_logprobs"], scored["query_logprobs"], argmax):
                engine_rows.append({"target_logprob": lp, "top_logprobs": query, "argmax": best})
            reference_rows += targets
        result = metrics.quality_path(reference_rows, engine_rows)
        failures = metrics.judge(result, thresholds)
        result.update(seconds=time.time() - started, passed=not failures, failures=failures,
                      budget_used=metrics.budget_used(result, thresholds))
        out["paths"][path] = result
        log(f"quality {path}: ppl ratio {result['ppl_ratio']:.5f}, KLD mean {result['kld_mean']:.2e}, "
            f"p99 {result['kld_p99']:.2e}, top-1 {result['top1_agreement']:.3f}, {result['budget_used']:.1%} of the drift budget "
            f"({'pass' if not failures else 'FAIL'})")
    out["passed"] = bool(out["paths"]) and all(p["passed"] for p in out["paths"].values())
    return out


def plain_on_engine(engine, info):
    """Whether the speculative engine itself decodes plain one-token greedy (info "plain", generate_plain),
    so exactness needs no second engine."""
    return bool(info.get("speculative") and info.get("plain") and callable(getattr(engine, "generate_plain", None)))


def run_exactness(module, model, config, prepared, speed, log, mode="reported", context_tokens=None, engine=None,
                  info=None):
    """Plain greedy output of every measured speed case against the speculative output. With `engine` (a
    speculative engine whose plain_on_engine holds) the plain requests run on it; otherwise on a second
    engine opened with speculative false."""
    shared = engine is not None
    if not shared:
        engine, info = open_engine(module, model, config, speculative=False, context_tokens=context_tokens)
    generate = timed_plain if shared else timed_generate
    try:
        cases, samples, steps = {}, [], {}
        for case in prepared["speed"]:
            measured = speed["cases"].get(case["id"], {})
            if measured.get("status") != "measured":
                continue
            tokens, sample = generate(engine, case["prompt_ids"], case["output_tokens"])
            samples.append(sample)
            steps[case["id"]] = headroom.step_ms_from([sample])
            diverge = next((i for i, (a, b) in enumerate(zip(tokens, measured["tokens"])) if a != b), None)
            agreeing = len(tokens) if diverge is None else diverge
            cases[case["id"]] = {"identical": diverge is None, "first_difference": diverge,
                                 "agreeing_prefix": agreeing / max(len(tokens), 1)}
            log(f"exactness {case['id']}: {'identical' if diverge is None else f'differs at token {diverge}'}")
        return {"mode": mode, "baseline": info["configuration"], "engine": "shared" if shared else "separate",
                "cases": cases,
                "identical": all(c["identical"] for c in cases.values()),
                "passed": mode == "reported" or all(c["identical"] for c in cases.values()),
                "step_ms": headroom.step_ms_from(samples), "steps_ms": steps}
    finally:
        if not shared:
            engine.close()


def context_requirement(model, device, context_tokens, vram_headroom_bytes, settings):
    """The context every engine is built and measured for: the given one (a run's kv_tokens) or Eval's
    max context for the model on the device. `tokens` is None when neither is known."""
    if context_tokens:
        return {"tokens": int(context_tokens), "source": "given", "missing": [], "reason": None}
    return {**max_context(model, device, vram_headroom_bytes, settings), "source": "computed"}


def context_text(info, continuations, prepared, output_tokens):
    """The text the long-context points decode, and the replayed tokens: the longest speed case's pinned
    continuation (its first output_tokens), since every point's prompt ends with that case's whole prompt,
    when the data has one and the adapter replays; else each engine's own greedy output."""
    text = decode_text(info, continuations)
    if text["source"] != "pinned":
        return text, None
    longest = max(prepared["speed"], key=lambda case: len(case["prompt_ids"]))
    pinned = continuations["cases"].get(longest["id"]) or []
    if len(pinned) < output_tokens:
        return {"source": "generated", "reason": f"the pinned continuation of {longest['id']} has {len(pinned)} tokens, "
                f"fewer than the {output_tokens} each point decodes"}, None
    return {**text, "case": longest["id"]}, list(pinned[:output_tokens])


def run_context(engine, info, prepared, settings, required, log, continuations=None):
    """One greedy request per curve point, each filling the context exactly (prompt plus output equal
    the point). The prompt is copies of the longest speed case's prompt ending with a whole one, so the
    engine decodes an answer to it: a replay of that case's pinned continuation when there is one (the
    same text for every engine, as in the speed cases), else the engine's own output. A request the
    engine cannot serve is recorded as failed. Returns (points, decode text)."""
    base = max((case["prompt_ids"] for case in prepared["speed"]), key=len)
    n = int(settings["output_tokens"])
    text, continuation = context_text(info, continuations, prepared, n)
    points = {}
    for point in curve_points(required, settings):
        if point > info["max_context"]:
            points[str(point)] = {"status": "skipped", "context": point,
                                  "reason": f"the engine's max_context {info['max_context']:,} is below {point:,}"}
            log(f"context {point:,}: skipped (max_context {info['max_context']:,})")
            continue
        if point <= n:
            points[str(point)] = {"status": "skipped", "context": point, "reason": f"no room for a prompt before {n} output tokens"}
            continue
        prompt = filled_prompt(base, point - n)
        samples = []
        try:
            for _ in range(int(settings.get("repetitions", 1))):
                if continuation is not None:
                    _, sample = timed_replay(engine, prompt, continuation)
                else:
                    _, sample = timed_generate(engine, prompt, n)
                samples.append(sample)
        except Exception as error:  # an engine that cannot hold the context fails here, however it fails
            points[str(point)] = {"status": "failed", "context": point, "error": f"{type(error).__name__}: {error}"[:2000]}
            log(f"context {point:,}: failed ({type(error).__name__}: {error})")
            continue
        summary = _summary(samples)
        points[str(point)] = {"status": "measured", "context": point, "prompt_tokens": len(prompt), "output_tokens": n,
                              **summary, "samples": samples}
        log(f"context {point:,}: decode {summary['decode_tps'] or 0:.1f} tok/s, first token after {summary['ttft_ms']:,.0f} ms, "
            f"tokens/round {summary['tokens_per_round'] or 0:.2f}")
    return points, text


def _spread(case, key):
    return headroom.spread([s[key] for s in case.get("samples", []) if not s.get("warmup") and s.get(key)]) or 0.0


def compare_lower(at_max, at_lower, lower, settings):
    """Each speed case at a lower context against the same case at the required context. A case is slower
    when its decode falls, or its first token comes later, by more than the tolerance: the protocol's, or the
    case's own run-to-run spread in either measurement when that is larger."""
    cases, slower = {}, {}
    for case_id, high in at_max["cases"].items():
        low = at_lower["cases"].get(case_id) or {}
        if high.get("status") != "measured" or low.get("status") != "measured":
            continue
        row = {"decode_tps": low["decode_tps"], "decode_tps_at_required": high["decode_tps"],
               "ttft_ms": low["ttft_ms"], "ttft_ms_at_required": high["ttft_ms"],
               "decode_tolerance": max(float(settings["tolerance"]), _spread(high, "decode_tps"), _spread(low, "decode_tps")),
               "ttft_tolerance": max(float(settings["tolerance"]), _spread(high, "ttft_ms"), _spread(low, "ttft_ms"))}
        row["decode_change"] = (low["decode_tps"] / high["decode_tps"] - 1
                                if low.get("decode_tps") and high.get("decode_tps") else None)
        row["ttft_change"] = low["ttft_ms"] / high["ttft_ms"] - 1 if low.get("ttft_ms") and high.get("ttft_ms") else None
        kinds = []
        if row["decode_change"] is not None and row["decode_change"] < -row["decode_tolerance"]:
            kinds.append("decode")
        if row["ttft_change"] is not None and row["ttft_change"] > row["ttft_tolerance"]:
            kinds.append("ttft")
        if kinds:
            slower[case_id] = kinds
        cases[case_id] = row
    unmatched = [cid for cid, c in at_max["cases"].items() if c.get("status") == "measured" and cid not in cases]
    return {"context": lower, "cases": cases, "slower": slower, "unmeasured": unmatched,
            "decode_score": at_lower.get("decode_score"), "decode_score_at_required": at_max.get("decode_score"),
            "ttft_score_ms": at_lower.get("ttft_score_ms"), "ttft_score_ms_at_required": at_max.get("ttft_score_ms"),
            "decode_text": at_lower.get("decode_text"),
            "passed": bool(cases) and not slower and not unmatched}


def run_lower_context(module, model, config, prepared, protocol, required, warmups, repetitions, log,
                      continuations=None, disturbed_spread=None):
    """The speed cases on an engine opened at a lower context. It runs just before the engine built for the
    required context, so the two speed runs sit side by side and ahead of the long-context curve's heat.
    Lowering the context frees device memory and must never make a case slower (compare_lower)."""
    settings = protocol["context"]
    longest = max(len(case["prompt_ids"]) + case["output_tokens"] for case in prepared["speed"])
    lower = lower_context(required, longest, settings)
    if lower is None:
        return {"context": None, "passed": None,
                "reason": f"the required {required:,} tokens leave no lower context that holds the longest speed case ({longest:,})"}
    log(f"lower context: opening the engine at {lower:,} tokens")
    try:
        engine, info = open_engine(module, model, config, speculative=True, context_tokens=lower)
        try:
            measured = run_speed(engine, info, prepared, protocol, warmups, repetitions,
                                 lambda message: log("lower context: " + message), continuations, disturbed_spread)
        finally:
            engine.close()
    except Disturbed:
        raise
    except Exception as error:  # an engine that cannot run at a lower context fails the gate, however it fails
        log(f"lower context: failed ({type(error).__name__}: {error})")
        return {"context": lower, "passed": False, "error": f"{type(error).__name__}: {error}"[:2000]}
    return {"context": lower, "engine_max_context": info["max_context"], "speed": measured}


PARTS = ("speed", "quality", "exactness", "context", "heldout")


def run(adapter, model, data, out, config=None, parts=PARTS, warmups=None,
        repetitions=None, label=None, log=print, bandwidth_gbs=None, device=None, baseline=None,
        quality_budget=None, exactness=None, context_tokens=None, vram_headroom_bytes=None, context_reserve_bytes=None,
        abort_disturbed=True):
    """device: `crit eval probe` result (bandwidth_gbs, tflops, free memory); bandwidth_gbs overrides its bandwidth.
    baseline: an earlier result to show KPI changes against.
    quality_budget: the run's drift limits over the protocol's (ppl_ratio_max, kld_mean_max, kld_p99_max).
    exactness: "reported" (default) or "required" (any difference from plain greedy fails the gate).
    context_tokens: the context every engine is built for (a run's kv_tokens); without it, Eval's max context
    for the model on the device, with vram_headroom_bytes (default 1 GiB) left free. The `context` part
    measures the long-context curve up to it and the speed cases again at a lower context. The `heldout` part
    decodes the held-out prompts; with a baseline that measured them, the generality gate judges them.
    abort_disturbed: stop the round when a speed case's repetitions show a disturbed device (protocol
    speed.disturbed_spread); the result then holds what was measured and `aborted`, and no status."""
    protocol = load_json(EVAL_DIR / "protocol.json")
    thresholds = quality_thresholds(protocol, quality_budget)
    exactness = exactness or protocol["exactness"].get("default", "required")
    if exactness not in EXACTNESS_MODES:
        raise SystemExit(f"exactness must be one of {', '.join(EXACTNESS_MODES)}")
    prepared = load_json(Path(data) / PREPARED)
    if prepared.get("protocol") != protocol["protocol"]:
        raise SystemExit(f"{data} was prepared for {prepared.get('protocol')}, not {protocol['protocol']}: "
                         "prepare it again (crit eval prepare) into a new data directory")
    device = device or {}
    bandwidth_gbs = bandwidth_gbs or device.get("bandwidth_gbs")
    settings = protocol["context"]
    if context_reserve_bytes is not None:  # the machine's own reserve for workspaces and speed techniques
        settings = {**settings, "reserve_gib": context_reserve_bytes / GIB}
    requirement = context_requirement(model, device, context_tokens,
                                      GIB if vram_headroom_bytes is None else vram_headroom_bytes, settings)
    required = requirement["tokens"]
    ceiling = prepared.get("ceiling")
    if ceiling is None:
        try:
            ceiling = model_ceiling(model)
        except (OSError, ValueError, KeyError):
            ceiling = None
    reference = load_json(Path(data) / REFERENCE) if "quality" in parts else None
    if reference and reference["prepared_model_sha256"] != prepared["model"]["sha256"]:
        raise SystemExit("reference.json was built from different prepared data")
    continuations = load_continuations(data, prepared)
    module = load_adapter(adapter)
    result = {
        "contract": CONTRACT, "protocol": protocol["protocol"], "label": label,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_sha256": prepared["model"]["sha256"], "suite_sha256": prepared["suite_sha256"],
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "adapter": str(Path(adapter).resolve()),
        "device": {k: device.get(k) for k in ("device", "bandwidth_gbs", "tflops", "memory_total_bytes",
                                               "memory_free_bytes", "measured_at")} if device else None,
        "bandwidth_gbs": bandwidth_gbs,
        "requirements": {"quality_budget": thresholds, "exactness": exactness},
        "context": requirement,
    }
    if required:
        log(f"context: {required:,} tokens ({requirement['source']}"
            + (f", {requirement['bound']} bound" if requirement.get("bound") else "") + ")")
    else:
        log("context: unknown (" + "; ".join(requirement["missing"] or [requirement["reason"] or "no input"])
            + "); engines use their own")
    warmups = protocol["speed"]["warmup_runs"] if warmups is None else warmups
    repetitions = protocol["speed"]["repetitions"] if repetitions is None else repetitions
    disturbed = protocol["speed"].get("disturbed_spread") if abort_disturbed else None
    try:
        return _measure(result, module, model, config or {}, prepared, protocol, parts, warmups, repetitions, log,
                        continuations, required, settings, reference, thresholds, exactness, disturbed, bandwidth_gbs,
                        device, ceiling, baseline, out)
    except Disturbed as error:
        log(f"aborted: {error}")
        result["aborted"] = {"reason": "disturbed", "detail": str(error)}
        result["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(out, result)
        return result


def _measure(result, module, model, config, prepared, protocol, parts, warmups, repetitions, log, continuations,
             required, settings, reference, thresholds, exactness, disturbed, bandwidth_gbs, device, ceiling, baseline,
             out):
    lower = None
    if "context" in parts and "speed" in parts and required:
        lower = run_lower_context(module, model, config, prepared, protocol, required, warmups, repetitions, log,
                                  continuations, disturbed)
    started = time.perf_counter()
    engine, info = open_engine(module, model, config, speculative=True, context_tokens=required)
    result["load_ms"] = (time.perf_counter() - started) * 1000
    result["engine"] = info
    result["context"]["engine_max_context"] = info["max_context"]
    if info.get("native_libraries"):
        result["native_libraries"] = native_identity(info)
    log(f"loaded {info['name']} in {result['load_ms'] / 1000:.1f} s")
    try:
        if "speed" in parts:
            result["speed"] = run_speed(engine, info, prepared, protocol, warmups, repetitions, log, continuations,
                                        disturbed)
        if "heldout" in parts:
            result["heldout"] = run_heldout(engine, info, prepared, protocol, log)
        if "quality" in parts:
            result["quality"] = run_quality(engine, info, prepared, reference, protocol, log, thresholds)
        if "context" in parts and required:
            result["context"]["points"], result["context"]["decode_text"] = run_context(
                engine, info, prepared, settings, required, log, continuations)
        if "exactness" in parts and "speed" in result and plain_on_engine(engine, info):
            # The engine decodes plain greedy itself: no second engine to load.
            result["exactness"] = run_exactness(module, model, config, prepared, result["speed"], log, exactness,
                                                engine=engine, info=info)
    finally:
        engine.close()
    if "exactness" in parts and "speed" in result and info.get("speculative") and "exactness" not in result:
        result["exactness"] = run_exactness(module, model, config, prepared, result["speed"], log, exactness,
                                            context_tokens=required)
    if lower is not None and "speed" in lower:
        result["context"]["lower"] = {**compare_lower(result["speed"], lower["speed"], lower["context"], settings),
                                      "engine_max_context": lower["engine_max_context"]}
        for case_id, kinds in result["context"]["lower"]["slower"].items():
            log(f"lower context: {case_id} is slower at {lower['context']:,} tokens ({', '.join(kinds)})")
    elif lower is not None:
        result["context"]["lower"] = lower
    if "speed" in result:
        if info.get("speculative"):
            step_ms = result.get("exactness", {}).get("step_ms")
            steps = result.get("exactness", {}).get("steps_ms")
        else:
            measured = {cid: c for cid, c in result["speed"]["cases"].items() if c["status"] == "measured"}
            step_ms = headroom.step_ms_from([s for c in measured.values() for s in c["samples"] if not s["warmup"]])
            steps = {cid: headroom.step_ms_from([s for s in c["samples"] if not s["warmup"]]) for cid, c in measured.items()}
        result["headroom"] = headroom.summarize(info, result["speed"], step_ms, bandwidth_gbs, protocol["headroom"],
                                                ceiling=ceiling, tflops=device.get("tflops"), steps=steps,
                                                prefill_cases=protocol["speed"].get("prefill_cases", []),
                                                context=result["context"], context_settings=settings)
        for warning in result["headroom"]["warnings"]:
            log("headroom: " + warning)
    result["status"] = status.build(result, protocol, baseline)
    result["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(out, result)
    return result
