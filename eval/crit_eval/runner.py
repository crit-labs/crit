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
from .hostload import HostLoad
from . import headroom, metrics, status
from .ceiling import model_ceiling
from .chat import Server
from .context import (GIB, curve_points, curve_prompt, lower_context, max_context, retrieval_points, retrieval_prompt,
                      token_text)
from .pin import CONTINUATIONS, load_continuations
from .prepare import PREPARED, check_prepared


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
    # An engine behind a chat API streams text: the server counts its tokens, and its reply may end early.
    served = result.get("served")
    streamed = [t for _, chunk in deliveries for t in chunk]
    if served is None and streamed != tokens:
        raise RuntimeError("tokens passed to on_tokens differ from the returned tokens")
    count = len(tokens) if served is None else int(served["output_tokens"])
    if count > output_tokens or count < output_tokens and served is None:
        raise RuntimeError(f"engine returned {count} tokens, expected {output_tokens} (stop tokens must be disabled)")
    if expected is not None and tokens != list(expected):
        diverge = next(i for i, (a, b) in enumerate(zip(tokens, expected)) if a != b)
        raise RuntimeError(f"replay emitted a token different from the pinned continuation at position {diverge}")
    first = next(t for t, chunk in deliveries if chunk)
    prompt_tokens = len(prompt) if served is None else int(served["prompt_tokens"])
    sample = metrics.speed_sample(prompt_tokens, count, started, first, deliveries[-1][0], result.get("rounds"))
    if served is not None:
        sample["served"] = served
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


def check_disturbed(case_id, samples, limit, other_cores=None):
    """Raise Disturbed when the measured decode speeds of one case spread (max - min over the median) by more
    than `limit`. Quiet rounds recorded at most about 10%; a disturbed one 37%. other_cores: CPU cores other
    processes kept busy during the case (HostLoad), named in the message."""
    measured = [s for s in samples if not s.get("warmup")]
    if not limit or not measured or metrics.median([s["request_ms"] for s in measured]) < DISTURBED_MIN_REQUEST_MS:
        return
    spread = headroom.spread([s["decode_tps"] for s in measured if s.get("decode_tps")])
    if spread is not None and spread > limit:
        load = f"; other processes kept {other_cores:.1f} CPU cores busy" if other_cores is not None else ""
        raise Disturbed(f"speed {case_id}: decode varied {spread:.0%} between repetitions (limit {limit:.0%}){load}; "
                        "the device was disturbed. Measure again when nothing else uses it")


SUMMARY_KEYS = ("decode_tps", "prefill_tps", "ttft_ms", "request_ms", "tokens_per_round", "round_ms")


def _summary(samples):
    return {key: metrics.median([s[key] for s in samples if s[key] is not None]) for key in SUMMARY_KEYS}


def rate(value, digits):
    """A speed for the log; a reply streamed in one piece has no decode time, and one before any time no prefill."""
    return "unmeasured" if value is None else f"{value:.{digits}f} tok/s"


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
        load = HostLoad().start()
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
        other = load.cores()
        check_disturbed(case["id"], samples, disturbed_spread, other)
        summary = _summary([s for s in samples if not s["warmup"]])
        crossed = [i for i, t in enumerate(outputs[0]) if t in stop_ids]
        cases[case["id"]] = {
            "status": "measured", "kind": case["kind"], "prompt_tokens": len(prompt), "output_tokens": n, **summary,
            "repeatable": all(o == outputs[0] for o in outputs),
            "stop_crossed": bool(crossed), "first_stop_index": crossed[0] if crossed else None,
            "tokens": outputs[0], "samples": samples,
            # CPU cores other processes kept busy during the case: engines computing on the host slow down with them.
            "host_other_cpu_cores": None if other is None else round(other, 2),
        }
        log(f"speed {case['id']}: decode {rate(summary['decode_tps'], 1)}"
            + (" (pinned continuation)" if continuation is not None else "")
            + f", prefill {rate(summary['prefill_tps'], 0)}, tokens/round {summary['tokens_per_round'] or 0:.2f}, "
            f"round {summary['round_ms'] or 0:.2f} ms"
            + (f", other processes {other:.1f} CPU cores" if other is not None and other >= 0.5 else ""))
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
        log(f"held-out {prompt['id']} ({prompt['language']}): decode {rate(summary['decode_tps'], 1)}, "
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


def run_quality(engine, info, prepared, protocol, log):
    """protocol quality: the engine's greedy reply to every quality prompt, with the log-probabilities it reports
    (an engine's reply(prompt, max_new_tokens, top_k): an engine behind a server, crit_eval.chat). `crit eval
    reference` scores them on the prepared model's checkpoint (reference.score_result). A set with a prompt that
    does not fit the engine's max_context with reply_tokens is skipped."""
    rule = protocol["quality"]
    out = {"top_k": rule["top_k"], "reply_tokens": rule["reply_tokens"]}
    if not callable(getattr(engine, "reply", None)):
        log("quality: not measured (an adapter's engine; quality is measured over the chat API)")
        return {**out, "status": "unsupported",
                "reason": "quality is measured over the chat API: serve the engine and run crit eval run --url"}
    started, replies, skipped = time.time(), [], {}
    for prompt in prepared["quality"]:
        need = max(len(p["prompt_ids"]) for p in prepared["quality"] if p["set"] == prompt["set"]) + rule["reply_tokens"]
        if need > info["max_context"]:
            skipped[prompt["set"]] = f"needs {need:,} tokens of context"
            continue
        replies.append({"id": prompt["id"], "set": prompt["set"],
                        **engine.reply(prompt["prompt_ids"], rule["reply_tokens"], rule["top_k"])})
    for name, reason in skipped.items():
        log(f"quality {name}: skipped ({reason})")
    if not replies:
        return {**out, "status": "skipped", "skipped": skipped,
                "reason": f"no quality prompt fits the engine's {info['max_context']:,}-token context"}
    log(f"quality: {len(replies)} replies, {sum(len(r['tokens']) for r in replies):,} tokens, await the reference")
    return {**out, "status": "replied", "replies": replies, "skipped": skipped, "seconds": time.time() - started}


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
    """The text the long-context points decode, and the replayed tokens: the curve case's pinned continuation
    (its first output_tokens), since every point's prompt ends with that case's request, when the data has one
    and the adapter replays; else each engine's own greedy output."""
    text = decode_text(info, continuations)
    if text["source"] != "pinned":
        return text, None
    case = prepared["context"]["case"]
    pinned = continuations["cases"].get(case) or []
    if len(pinned) < output_tokens:
        return {"source": "generated", "reason": f"the pinned continuation of {case} has {len(pinned)} tokens, "
                f"fewer than the {output_tokens} each point decodes"}, None
    return {**text, "case": case}, list(pinned[:output_tokens])


def run_context(engine, info, prepared, settings, required, log, continuations=None):
    """One greedy request per curve point, each filling the context exactly (prompt plus output equal
    the point). The prompt is the longest speed case with its document and filler text after it
    (curve_prompt), so the engine decodes an answer to that case: a replay of its pinned continuation when
    there is one (the same text for every engine, as in the speed cases), else the engine's own output. A
    request the engine cannot serve is recorded as failed. Returns (points, decode text)."""
    pieces, filler = prepared["context"], prepared["retrieval"]["filler_ids"]
    n = int(settings["output_tokens"])
    text, continuation = context_text(info, continuations, prepared, n)
    points = {}
    for point in curve_points(required, settings):
        if point > info["max_context"]:
            points[str(point)] = {"status": "skipped", "context": point,
                                  "reason": f"the engine's max_context {info['max_context']:,} is below {point:,}"}
            log(f"context {point:,}: skipped (max_context {info['max_context']:,})")
            continue
        prompt = curve_prompt(pieces, filler, point - n)
        if prompt is None:
            points[str(point)] = {"status": "skipped", "context": point,
                                  "reason": f"no room for the {pieces['case']} prompt before {n} output tokens"}
            continue
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


def run_retrieval(engine, info, prepared, settings, required, log):
    """Facts near the start, middle and end of filler text, asked for at the end (protocol context.retrieval):
    one greedy request per point, the last filling the required context. A point passes when the output up to
    its first stop token contains every fact's answer verbatim; one the engine cannot serve fails."""
    pieces, rule = prepared.get("retrieval"), settings["retrieval"]
    if not pieces:
        log("retrieval: not in the prepared data (prepare it again with crit eval prepare into a new data directory)")
        return {"status": "missing", "reason": "the prepared data has no retrieval prompts; prepare it again"}
    n, stops, points = int(rule["output_tokens"]), prepared["stop_ids"], {}
    for point in retrieval_points(required, rule):
        prompt = retrieval_prompt(pieces, point - n, rule["positions"])
        if prompt is None:
            points[str(point)] = {"status": "skipped", "context": point, "reason": "too short to hold the facts"}
            continue
        if point > info["max_context"]:
            points[str(point)] = {"status": "failed", "context": point,
                                  "error": f"the engine's max_context {info['max_context']:,} is below {point:,}"}
            log(f"retrieval {point:,}: not served (max_context {info['max_context']:,})")
            continue
        try:
            tokens, sample = timed_generate(engine, prompt, n)
        except Exception as error:  # an engine that cannot hold the context fails here, however it fails
            points[str(point)] = {"status": "failed", "context": point, "error": f"{type(error).__name__}: {error}"[:2000]}
            log(f"retrieval {point:,}: failed ({type(error).__name__}: {error})")
            continue
        text = token_text(tokens, pieces["token_text"], stops)
        found = {fact["id"]: fact["answer"] in text for fact in pieces["facts"]}
        points[str(point)] = {"status": "measured", "context": point, "prompt_tokens": len(prompt), "found": found,
                              "passed": all(found.values()), "text": text[:2000], "tokens": tokens,
                              "ttft_ms": sample["ttft_ms"]}
        log(f"retrieval {point:,}: " + ", ".join(f"{k} {'found' if v else 'MISSING'}" for k, v in found.items()))
    judged_points = [p for p in points.values() if p["status"] != "skipped"]
    return {"status": "measured", "points": points,
            "passed": bool(judged_points) and all(p.get("passed") for p in judged_points)}


def history_cases(prepared, info):
    """The speed cases the request-history check repeats (protocol history): those with the shortest and the
    longest prompt the engine holds."""
    held = [c for c in prepared["speed"] if len(c["prompt_ids"]) + c["output_tokens"] <= info["max_context"]]
    if not held:
        return []
    ends = (min(held, key=lambda c: len(c["prompt_ids"])), max(held, key=lambda c: len(c["prompt_ids"])))
    return [c for c in held if any(c is end for end in ends)]


def history_outputs(engine, cases):
    return {case["id"]: timed_generate(engine, case["prompt_ids"], case["output_tokens"])[0] for case in cases}


def compare_history(fresh, after, served):
    """Each case's output on the freshly opened engine against its output after everything `served` names."""
    cases = {}
    for case_id, first in fresh.items():
        later = after.get(case_id) or []
        diverge = next((i for i, (a, b) in enumerate(zip(first, later)) if a != b), None)
        if diverge is None and len(first) != len(later):
            diverge = min(len(first), len(later))
        cases[case_id] = {"identical": diverge is None, "first_difference": diverge, "fresh": first, "after": later}
    return {"cases": cases, "served_between": served,
            "passed": bool(cases) and all(c["identical"] for c in cases.values())}


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
               "ttft_tolerance": max(float(settings["tolerance"]), _spread(high, "ttft_ms"), _spread(low, "ttft_ms")),
               "host_other_cpu_cores": low.get("host_other_cpu_cores"),
               "host_other_cpu_cores_at_required": high.get("host_other_cpu_cores")}
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


PARTS = ("speed", "quality", "exactness", "context", "heldout", "history")


def run(adapter, model, data, out, config=None, parts=PARTS, warmups=None,
        repetitions=None, label=None, log=print, bandwidth_gbs=None, device=None, baseline=None,
        quality_budget=None, exactness=None, context_tokens=None, vram_headroom_bytes=None, context_reserve_bytes=None,
        abort_disturbed=True, url=None, served_model=None):
    """adapter: the eval-v1 adapter file, or an object that opens engines as an adapter module does
    (create_eval_engine), its `server` naming them in the result (crit run eval serves a candidate per engine); with
    url instead, the engine behind that OpenAI-compatible server (crit_eval.chat), served_model naming its model.
    device: `crit eval probe` result (bandwidth_gbs, tflops, free memory); bandwidth_gbs overrides its bandwidth.
    baseline: an earlier result to show KPI changes against.
    quality_budget: the run's drift limits over the protocol's (ppl_ratio_max, kld_mean_max, kld_p99_max). The
    `quality` part records the engine's replies to the quality prompts; `crit eval reference` scores them and judges
    every set by these limits.
    exactness: "reported" (default) or "required" (any difference from plain greedy fails the gate).
    context_tokens: the context every engine is built for (a run's kv_tokens); without it, Eval's max context
    for the model on the device, with vram_headroom_bytes (default 1 GiB) left free. The `context` part
    measures the long-context curve and the retrieval check up to it and the speed cases again at a lower context.
    The `heldout` part decodes the held-out prompts; with a baseline that measured them, the generality gate judges
    them. The `history` part repeats protocol history's cases right after the engine opens and after every other
    part ran on it.
    abort_disturbed: stop the round when a speed case's repetitions show a disturbed device (protocol
    speed.disturbed_spread); the result then holds what was measured and `aborted`, and no status."""
    protocol = load_json(EVAL_DIR / "protocol.json")
    thresholds = quality_thresholds(protocol, quality_budget)
    exactness = exactness or protocol["exactness"].get("default", "required")
    if exactness not in EXACTNESS_MODES:
        raise SystemExit(f"exactness must be one of {', '.join(EXACTNESS_MODES)}")
    prepared = load_json(Path(data) / PREPARED)
    check_prepared(prepared, protocol, data)
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
    continuations = load_continuations(data, prepared)
    opener = None if url or isinstance(adapter, (str, Path)) else adapter
    module = Server(url, prepared, served_model) if url else opener or load_adapter(adapter)
    result = {
        "contract": CONTRACT, "protocol": protocol["protocol"], "label": label,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_sha256": prepared["model"]["sha256"], "suite_sha256": prepared["suite_sha256"],
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "adapter": None if url or opener else str(Path(adapter).resolve()), "server": url or getattr(opener, "server", None),
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
                        continuations, required, settings, exactness, disturbed, bandwidth_gbs,
                        device, ceiling, baseline, out)
    except Disturbed as error:
        log(f"aborted: {error}")
        result["aborted"] = {"reason": "disturbed", "detail": str(error)}
        result["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(out, result)
        return result


def _measure(result, module, model, config, prepared, protocol, parts, warmups, repetitions, log, continuations,
             required, settings, exactness, disturbed, bandwidth_gbs, device, ceiling, baseline,
             out):
    lower = None
    if "context" in parts and "speed" in parts and required and isinstance(module, Server):
        lower = {"context": None, "passed": None, "reason": "a served engine cannot be opened at a lower context"}
    elif "context" in parts and "speed" in parts and required:
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
        fresh = None
        if "history" in parts:
            # Before any other request: what this engine answers fresh, repeated once everything else has run.
            cases = history_cases(prepared, info)
            fresh = history_outputs(engine, cases)
            log("history: fresh outputs of " + (", ".join(fresh) or "no case (none fits the engine)"))
        if "speed" in parts:
            result["speed"] = run_speed(engine, info, prepared, protocol, warmups, repetitions, log, continuations,
                                        disturbed)
        if "heldout" in parts:
            result["heldout"] = run_heldout(engine, info, prepared, protocol, log)
        if "quality" in parts:
            result["quality"] = run_quality(engine, info, prepared, protocol, log)
        if "context" in parts and required:
            result["context"]["points"], result["context"]["decode_text"] = run_context(
                engine, info, prepared, settings, required, log, continuations)
            result["context"]["retrieval"] = run_retrieval(engine, info, prepared, settings, required, log)
        if "exactness" in parts and "speed" in result and plain_on_engine(engine, info):
            # The engine decodes plain greedy itself: no second engine to load.
            result["exactness"] = run_exactness(module, model, config, prepared, result["speed"], log, exactness,
                                                engine=engine, info=info)
        if fresh is not None:
            served = [part for part in ("speed", "heldout", "quality", "context", "exactness") if part in result
                      and (part != "context" or "points" in result["context"])
                      and (part != "quality" or result["quality"].get("replies"))
                      and (part != "exactness" or result["exactness"].get("engine") == "shared")]
            result["history"] = compare_history(fresh, history_outputs(engine, cases), served)
            between = ", ".join(served) or "nothing else"
            for case_id, row in result["history"]["cases"].items():
                log(f"history {case_id}: " + (f"identical after {between}" if row["identical"] else
                                              f"differs at token {row['first_difference']} after {between}"))
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
