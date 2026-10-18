"""What an engine measures against its physical limits: `crit eval status RESULT.json`.

Reads one result file and reports facts. It does not classify or rank; the agent reasons from
the numbers about how to build.

1. Gates, the only pass/fail: speed coverage, repeatable outputs, request history (the same greedy
   requests give the same tokens on the fresh engine and after everything else it served), quality
   against the reference's drift budget on every path (prefill_decode, decode after long prompts,
   required), capacity (the engine serves a request at the required context), retrieval (facts at the
   start, middle and end of long prompts up to the required context are answered verbatim), lower
   context (no speed case is slower on an engine opened at a lower context), generality (against a
   baseline, no held-out prompt and not the held-out score slows down) and, when the run requires it,
   speculative exactness. Exactness in its default `reported` mode is a note: the quality budget
   judges output; request history is a gate in every mode.
2. Goals for the next candidate: each of decode, prefill and time to first token, and decode and
   time to first token over the long-context curve, from its current score to its physical limit.
   All of them hold at once; none is ranked or optional.
3. KPIs, each next to its physical limit: the decode score against its ceiling (the larger of
   this drafter's speed of light and a drafter accepting later drafts as often as the first),
   the model pass against its bandwidth floor, the prefill score against its compute ceiling,
   time to first token against its floor, the share of the drift budget the worst path uses,
   and the end-to-end score as a fact.
4. Gaps: what each term would give its own KPI at its floor, in a fixed order.
5. Missing inputs that hide a limit, and notes.

`run` writes the same status into the result and prints it. With --baseline, the KPIs also
show the change from an earlier result of the same model and suite.
"""
from . import load_json, metrics


def _gates(result):
    speed, quality, exact = result.get("speed"), result.get("quality"), result.get("exactness")
    speculative = (result.get("engine") or {}).get("speculative")
    gates = []
    if speed:
        cases = speed["cases"]
        measured = [cid for cid, c in cases.items() if c["status"] == "measured"]
        skipped = {cid: c.get("reason") for cid, c in cases.items() if c["status"] != "measured"}
        gates.append({"name": "coverage", "state": "pass" if not skipped else "fail",
                      "detail": f"{len(measured)}/{len(cases)} speed cases measured"
                                + "".join(f"; {cid} skipped: {why}" for cid, why in skipped.items())})
        unrepeatable = [cid for cid in measured if not cases[cid].get("repeatable", True)]
        gates.append({"name": "repeatable", "state": "fail" if unrepeatable else "pass",
                      "detail": (f"output changes between repetitions: {', '.join(unrepeatable)}" if unrepeatable
                                 else f"{len(measured)}/{len(measured)} cases repeat their output")})
    else:
        gates.append({"name": "coverage", "state": "unmeasured", "detail": "speed not run"})
    gates += _history_gate(result)
    if quality is None:
        gates.append({"name": "quality", "state": "unmeasured", "detail": "quality not run"})
    elif not quality["paths"]:
        gates.append({"name": "quality", "state": "unmeasured",
                      "detail": "the adapter declares no score paths (score_paths)"})
    else:
        failures = [f"{p}: {', '.join(q['failures'])}" for p, q in quality["paths"].items() if not q["passed"]]
        # A required path (protocol quality.required_paths) that was not scored fails: every engine prefills and decodes.
        failures += [f"{p}: not scored ({(quality.get('missing') or {}).get(p) or 'the adapter does not declare it in score_paths'})"
                     for p in quality.get("required", []) if p not in quality["paths"]]
        detail = ("paths " + ", ".join(quality["paths"]) if not failures else "; ".join(failures))
        skipped = [f"{int(length):,}" for q in quality["paths"].values()
                   for length, row in (q.get("lengths") or {}).items() if row.get("status") == "skipped"]
        if skipped:
            detail += f" (prefill_decode skipped at {', '.join(skipped)} prompt tokens: above max_context)"
        if quality.get("unsupported"):
            detail += f" (unscored: {', '.join(quality['unsupported'])})"
        gates.append({"name": "quality", "state": "fail" if failures else "pass", "detail": detail})
    gates += _context_gates(result)
    if speculative and _exactness_mode(result) == "required":
        if exact is None:
            gates.append({"name": "exactness", "state": "unmeasured", "detail": "exactness not run"})
        else:
            differ = {cid: c["first_difference"] for cid, c in exact["cases"].items() if not c["identical"]}
            gates.append({"name": "exactness", "state": "fail" if differ else "pass",
                          "detail": (f"{len(exact['cases']) - len(differ)}/{len(exact['cases'])} cases equal plain greedy"
                                     + "".join(f"; {cid} differs at token {i}" for cid, i in differ.items()))})
    return gates


LEGACY_PROTOCOLS = ("crit-eval-v1", "crit-eval-v2", "crit-eval-v3")


def _since_v4(result):
    """Results of crit-eval-v4 and later have the request-history and retrieval gates; earlier ones are shown as
    they were judged."""
    return result.get("protocol") is not None and result["protocol"] not in LEGACY_PROTOCOLS


def _history_gate(result):
    """The same greedy requests on the freshly opened engine and after everything else it served. A gate in
    every exactness mode: a deterministic engine repeats a request whatever came before."""
    history = result.get("history")
    if history is None:
        return [{"name": "request_history", "state": "unmeasured", "detail": "history not run"}] if _since_v4(result) else []
    cases = history.get("cases") or {}
    if not cases:
        return [{"name": "request_history", "state": "unmeasured", "detail": "no history case fits the engine"}]
    differ = {cid: c["first_difference"] for cid, c in cases.items() if not c["identical"]}
    served = ", ".join(history.get("served_between") or []) or "nothing else"
    return [{"name": "request_history", "state": "fail" if differ else "pass",
             "detail": (f"{len(cases) - len(differ)}/{len(cases)} cases repeat their fresh-engine output after {served}"
                        + "".join(f"; {cid} differs at token {i}" for cid, i in differ.items()))}]


def _retrieval_gate(result, required):
    context = result.get("context") or {}
    retrieval = context.get("retrieval")
    if retrieval is None:
        return [{"name": "retrieval", "state": "unmeasured", "detail": "context checks not run"}] if _since_v4(result) else []
    if retrieval.get("status") != "measured":
        return [{"name": "retrieval", "state": "unmeasured", "detail": retrieval.get("reason") or "retrieval not run"}]
    problems = []
    for key, point in retrieval["points"].items():
        if point["status"] == "failed":
            problems.append(f"{int(key):,}: {point['error']}")
        elif point["status"] == "measured" and not point["passed"]:
            missing = [fact for fact, found in point["found"].items() if not found]
            problems.append(f"{int(key):,}: {', '.join(missing)} fact not found in {point['text'][:200]!r}")
    judged = [k for k, p in retrieval["points"].items() if p["status"] != "skipped"]
    if not judged:
        return [{"name": "retrieval", "state": "unmeasured", "detail": "no point holds the facts"}]
    return [{"name": "retrieval", "state": "fail" if problems else "pass",
             "detail": ("; ".join(problems) if problems else
                        f"every fact found at {', '.join(f'{int(k):,}' for k in judged)} tokens (up to the required {required:,})")}]


def _context_gates(result):
    """Capacity, retrieval and lower context; results from before the context checks have none of them."""
    context = result.get("context")
    if context is None:
        return []
    required = context.get("tokens")
    if not required:
        why = context.get("reason") or "; ".join(context.get("missing") or []) or "no required context"
        detail = f"max context unknown: {why} (pass --context, or --device with a crit eval probe result)"
        return [{"name": "capacity", "state": "unmeasured", "detail": detail}] + (
            [{"name": "retrieval", "state": "unmeasured", "detail": detail}] if _since_v4(result) else []) + [
            {"name": "lower_context", "state": "unmeasured", "detail": detail}]
    points, lower = context.get("points"), context.get("lower")
    if points is None:
        capacity = {"name": "capacity", "state": "unmeasured", "detail": "context checks not run"}
    else:
        top = points.get(str(required)) or {}
        origin = (context.get("source") or "given") + (f", {context['bound']} bound" if context.get("bound") else "")
        capacity = ({"name": "capacity", "state": "pass",
                     "detail": f"served {required:,} tokens of prompt plus output, the required context ({origin})"}
                    if top.get("status") == "measured" else
                    {"name": "capacity", "state": "fail",
                     "detail": f"the required {required:,} tokens ({origin}): {top.get('error') or top.get('reason') or 'not measured'}"})
    if lower is None:
        lowered = {"name": "lower_context", "state": "unmeasured", "detail": "context checks or speed cases not run"}
    elif lower.get("context") is None:
        lowered = {"name": "lower_context", "state": "unmeasured", "detail": lower.get("reason")}
    elif lower.get("error"):
        lowered = {"name": "lower_context", "state": "fail",
                   "detail": f"the engine opened at {lower['context']:,} tokens failed: {lower['error']}"}
    elif lower["passed"]:
        lowered = {"name": "lower_context", "state": "pass",
                   "detail": f"no speed case is slower at {lower['context']:,} tokens than at {required:,}: decode score "
                             f"{_change(lower.get('decode_score'), lower.get('decode_score_at_required'))}, ttft score "
                             f"{_change(lower.get('ttft_score_ms'), lower.get('ttft_score_ms_at_required'))}"}
    else:
        rows = lower.get("cases", {})
        parts = [f"{cid} {kind} {rows[cid][kind + '_change']:+.1%} (tolerance {rows[cid][kind + '_tolerance']:.1%})"
                 for cid, kinds in lower.get("slower", {}).items() for kind in kinds]
        parts += [f"{cid} not measured" for cid in lower.get("unmeasured", [])]
        lowered = {"name": "lower_context", "state": "fail",
                   "detail": f"slower at {lower['context']:,} tokens than at {required:,}: " + "; ".join(parts)}
    return [capacity] + _retrieval_gate(result, required) + [lowered]


def _generality_gate(result, protocol, baseline):
    """Held-out prompts against a baseline that measured them: the held-out decode score may not fall by more than
    the tolerance and no held-out prompt may slow down by more than prompt_tolerance, whatever the suite gained.
    How far the held-out change trails the suite's is reported, not judged: speculation gains less on text that
    drafts poorly, so a trailing held-out score alone is not proof of tuning to the suite."""
    held = result.get("heldout")
    if held is None:
        return []
    if held.get("status") != "measured":
        return [{"name": "generality", "state": "unmeasured", "detail": held.get("reason") or "held-out prompts not run"}]
    settings = protocol.get("heldout") or {}
    tolerance, prompt_tolerance = float(settings.get("tolerance", 0.05)), float(settings.get("prompt_tolerance", 0.1))
    before = (baseline or {}).get("heldout") or {}
    score = held.get("decode_score")
    if before.get("status") != "measured" or not before.get("decode_score") or not score:
        return [{"name": "generality", "state": "unmeasured",
                 "detail": f"held-out decode {_num(score, '.1f')} tok/s over {len(held.get('prompts', {}))} prompts; "
                           "judged against a baseline that measured them (--baseline)"}]
    compared = changes(result, baseline)
    if not compared["comparable"]:
        return [{"name": "generality", "state": "unmeasured", "detail": f"baseline not comparable: {compared['reason']}"}]
    held_change = score / before["decode_score"] - 1
    suite_change = compared.get("decode_score")
    failures = []
    if held_change < -tolerance:
        failures.append(f"held-out decode {held_change:+.1%} (more than {tolerance:.0%} slower)")
    for pid, row in held.get("prompts", {}).items():
        old = (before.get("prompts") or {}).get(pid) or {}
        if row.get("status") == "measured" and old.get("status") == "measured" and old.get("decode_tps"):
            change = row["decode_tps"] / old["decode_tps"] - 1
            if change < -prompt_tolerance:
                failures.append(f"{pid} ({row.get('language')}) {change:+.1%}")
    suite = "suite decode not compared (different decode text)" if suite_change is None else f"suite {suite_change:+.1%}"
    detail = f"held-out decode {held_change:+.1%}, {suite} against {compared['label']}"
    return [{"name": "generality", "state": "fail" if failures else "pass",
             "detail": detail + ("; " + "; ".join(failures) if failures else "; no held-out prompt slower by more than "
                                 f"{prompt_tolerance:.0%}")}]


def _change(now, before):
    return "–" if not now or not before else f"{now / before - 1:+.1%}"


def _exactness_mode(result):
    """Results from before exactness modes existed were judged as required."""
    return ((result.get("exactness") or {}).get("mode")
            or (result.get("requirements") or {}).get("exactness") or "required")


def _exactness_note(result):
    exact = result.get("exactness")
    if not exact or _exactness_mode(result) == "required":
        return None
    cases = exact["cases"]
    differ = {cid: c["first_difference"] for cid, c in cases.items() if not c["identical"]}
    return (f"exactness reported: {len(cases) - len(differ)}/{len(cases)} cases equal plain greedy"
            + "".join(f"; {cid} differs at token {i}" for cid, i in differ.items())
            + ("; the quality budget judges the output" if differ else ""))


def _exactness_drop(result, baseline):
    """Fewer cases equal plain greedy than in the baseline: a bug until a root cause is shown (AGENTS.md), never
    put down to rounding without evidence."""
    now, before = (result.get("exactness") or {}).get("cases"), ((baseline or {}).get("exactness") or {}).get("cases")
    if not now or not before:
        return None
    shared = [cid for cid in now if cid in before]
    equal_now = sum(now[cid]["identical"] for cid in shared)
    equal_before = sum(before[cid]["identical"] for cid in shared)
    if equal_now >= equal_before:
        return None
    lost = [cid for cid in shared if before[cid]["identical"] and not now[cid]["identical"]]
    label = baseline.get("label") or (baseline.get("engine") or {}).get("name") or "the baseline"
    return (f"plain-greedy equality fell from {equal_before}/{len(shared)} ({label}) to {equal_now}/{len(shared)}"
            f" ({', '.join(lost)} now differ): treat it as a bug until a root cause is demonstrated, not as rounding")


def _quality_kpi(quality, thresholds):
    if not quality or not quality.get("paths"):
        return None
    thresholds = quality.get("thresholds") or thresholds
    paths = quality["paths"]
    used = max(((metrics.budget_used(p, thresholds), name) for name, p in paths.items()),
               key=lambda v: -1 if v[0] is None else v[0])
    worst = {key: max(((p[key], name) for name, p in paths.items()), key=lambda v: v[0])
             for key in ("kld_mean", "kld_p99", "ppl_ratio")}
    lowest_top1 = min(((p["top1_agreement"], name) for name, p in paths.items()), key=lambda v: v[0])
    return {"passed": quality["passed"], "paths": list(paths), "unsupported": quality.get("unsupported", []),
            "worst_kld_mean": worst["kld_mean"][0], "worst_kld_mean_path": worst["kld_mean"][1],
            "worst_kld_p99": worst["kld_p99"][0], "worst_kld_p99_path": worst["kld_p99"][1],
            "worst_ppl_ratio": worst["ppl_ratio"][0], "worst_ppl_ratio_path": worst["ppl_ratio"][1],
            "lowest_top1": lowest_top1[0], "lowest_top1_path": lowest_top1[1],
            "budget_used": used[0], "budget_used_path": used[1],
            "limits": {"kld_mean": thresholds.get("kld_mean_max"), "kld_p99": thresholds.get("kld_p99_max"),
                       "ppl_ratio": thresholds.get("ppl_ratio_max")}}


def _context_kpi(result):
    """The required context and decode and time to first token over the long-context curve."""
    context = result.get("context")
    if context is None:
        return None
    room = (result.get("headroom") or {}).get("context") or {}
    lower = context.get("lower") or {}
    return {"required": context.get("tokens"), "source": context.get("source"), "bound": context.get("bound"),
            "model_limit": context.get("model_limit"), "device_limit": context.get("device_limit"),
            "engine_max_context": context.get("engine_max_context"),
            "decode": room.get("decode") or {}, "ttft": room.get("ttft") or {},
            "points": {key: {k: row.get(k) for k in ("context", "decode_tps", "ceiling_tps", "ttft_ms")}
                       | {"ttft_floor_ms": (row.get("ttft_floor") or {}).get("total_ms")}
                       for key, row in (room.get("points") or {}).items()},
            "not_measured": {key: point.get("error") or point.get("reason")
                             for key, point in (context.get("points") or {}).items() if point.get("status") != "measured"},
            "lower": {k: lower.get(k) for k in ("context", "decode_score", "decode_score_at_required", "ttft_score_ms",
                                                "ttft_score_ms_at_required")} if lower.get("context") else None}


def build(result, protocol, baseline=None):
    room = result.get("headroom") or {}
    speed = result.get("speed") or {}
    projection = room.get("projection") or {}
    prefill = room.get("prefill") or {}
    decode = {"score": speed.get("decode_score"), "complete": speed.get("complete"),
              "speed_of_light": projection.get("speed_of_light_score"),
              "acceptance_bound": projection.get("acceptance_bound_score"),
              # Results from before the acceptance bound use the drafter's speed of light as the ceiling.
              "ceiling": projection.get("ceiling_score") or projection.get("speed_of_light_score"),
              "fraction_of_ceiling": projection.get("fraction_of_ceiling") or projection.get("fraction_of_speed_of_light"),
              "pass_ms": room.get("step_ms"), "pass_floor_ms": room.get("pass_floor_ms"),
              "pass_efficiency": room.get("pass_efficiency")}
    prefill_kpi = {"score": prefill.get("score"), "ceiling": prefill.get("ceiling_score"),
                   "fraction_of_ceiling": prefill.get("fraction_of_ceiling"),
                   "cases": {cid: {k: row.get(k) for k in ("prompt_tokens", "ttft_ms", "prefill_tps", "ceiling_tps",
                                                            "efficiency")}
                             | {"floor_ms": (row.get("floor") or {}).get("total_ms")}
                             for cid, row in prefill.get("cases", {}).items()}}
    ttft = room.get("ttft") or {}
    ttft_kpi = {"score_ms": ttft.get("score_ms") or speed.get("ttft_score_ms"), "floor_ms": ttft.get("floor_ms"),
                "fraction_of_floor": ttft.get("fraction_of_floor"),
                "cases": {cid: {"prompt_tokens": row.get("prompt_tokens"), "ttft_ms": row.get("ttft_ms"),
                                "floor_ms": (row.get("floor") or {}).get("total_ms"),
                                "bound": (row.get("floor") or {}).get("bound"), "efficiency": row.get("efficiency")}
                          for cid, row in ttft.get("cases", {}).items()}}
    e2e = room.get("end_to_end") or {}
    spreads = room.get("noise") or {}
    end_to_end = {"score": e2e.get("score"), "ceiling": e2e.get("ceiling_score"),
                  "fraction_of_ceiling": e2e.get("fraction_of_ceiling"), "noise": spreads.get("end_to_end"),
                  "decode_noise": spreads.get("decode"), "ttft_noise": spreads.get("ttft")}
    held = result.get("heldout") or {}
    heldout_kpi = ({"decode_score": held.get("decode_score"), "tokens_per_round": held.get("tokens_per_round"),
                    "prompts": {pid: {k: row.get(k) for k in ("language", "decode_tps", "tokens_per_round")}
                                for pid, row in held.get("prompts", {}).items() if row.get("status") == "measured"}}
                   if held.get("status") == "measured" else None)
    kpis = {"end_to_end": end_to_end, "decode": decode, "prefill": prefill_kpi, "ttft": ttft_kpi,
            "context": _context_kpi(result), "heldout": heldout_kpi,
            "quality": _quality_kpi(result.get("quality"), protocol["quality"]["thresholds"])}
    missing = list(room.get("missing", []))
    quality = result.get("quality")
    if quality and quality.get("unsupported"):
        missing.append("scoring on paths " + ", ".join(quality["unsupported"]) + " (adapter score_paths)")
    # Older results call the gaps levers, tag them and add end-to-end gains; none of that is shown.
    gaps = [{k: v for k, v in gap.items() if k not in ("polished_kpi", "below_noise", "end_to_end_gain")}
            for gap in room.get("gaps", room.get("levers", []))]
    notes = [f"{cid}: the model's end token appears at output token {c['first_stop_index']} (stops are disabled)"
             for cid, c in speed.get("cases", {}).items() if c.get("stop_crossed")]
    exactness_note = _exactness_note(result)
    if exactness_note:
        notes.append(exactness_note)
    drop = _exactness_drop(result, baseline)
    if drop:
        notes.append(drop)
    if ((result.get("context") or {}).get("points") or {}):
        ctext = (result.get("context") or {}).get("decode_text") or {"source": "generated"}
        notes.append(f"long-context points replay {ctext.get('case')}'s pinned continuation {ctext['sha256'][:12]}, after copies "
                     "of its prompt" if ctext["source"] == "pinned" else
                     "long-context points decode each engine's own output, after copies of the longest speed case's prompt"
                     + (f" ({ctext['reason']})" if ctext.get("reason") else ""))
    text = speed.get("decode_text")
    if text:
        notes.append(f"decode measured on the pinned continuation {text['sha256'][:12]}" if text["source"] == "pinned"
                     else f"decode measured on the engine's own output ({text.get('reason')})")
    comparable = baseline and changes(result, baseline)
    status = {"kpis": kpis, "gates": _gates(result) + _generality_gate(result, protocol, baseline), "goals": goals(result),
              "gaps": gaps, "missing": missing,
              "notes": notes,
              "stop_record_headroom": stop_record_headroom(result, baseline if comparable and comparable["comparable"] else None)}
    if baseline:
        status["baseline"] = comparable
    return status


GOAL_UNITS = {"decode": "tok/s", "prefill": "tok/s", "ttft": "ms", "context_decode": "tok/s", "context_ttft": "ms"}


def _limits(result):
    """(kpi, score, limit) for decode, prefill and time to first token, and once the long-context curve
    ran, decode and time to first token over it."""
    room = result.get("headroom") or {}
    speed = result.get("speed") or {}
    projection = room.get("projection") or {}
    prefill, ttft = room.get("prefill") or {}, room.get("ttft") or {}
    rows = [("decode", speed.get("decode_score"), projection.get("ceiling_score") or projection.get("speed_of_light_score")),
            ("prefill", prefill.get("score"), prefill.get("ceiling_score")),
            ("ttft", ttft.get("score_ms") or speed.get("ttft_score_ms"), ttft.get("floor_ms"))]
    if (result.get("context") or {}).get("points"):
        long = room.get("context") or {}
        decode, first = long.get("decode") or {}, long.get("ttft") or {}
        rows += [("context_decode", decode.get("score"), decode.get("ceiling_score")),
                 ("context_ttft", first.get("score_ms"), first.get("floor_ms"))]
    return rows


def goals(result):
    """The next candidate's goals: every KPI from its current score to its physical limit, as the
    change that closes it (time to first token falls, the scores rise). They hold together; Eval
    does not rank them or mark any as enough."""
    return [{"kpi": kpi, "score": score, "limit": limit, "unit": GOAL_UNITS[kpi],
             "change": limit / score - 1 if score and limit else None}
            for kpi, score, limit in _limits(result)]


def stop_record_headroom(result, baseline=None):
    """The goals in the shape `crit run ... stop` takes as headroom: each KPI's score at the start
    (from the baseline, when given), the score reached and its limit. The agent adds what it built
    for each KPI (`built`) and the physical cause of any distance left to the limit (`cause`)."""
    rows = _limits(result)
    if not any(score for _, score, _ in rows):
        return None
    start = {kpi: score for kpi, score, _ in _limits(baseline)} if baseline else {}

    def rounded(value):
        return round(value, 3) if value else None
    return {"kpis": [{"kpi": kpi, "start": rounded(start.get(kpi)), "reached": rounded(score), "limit": rounded(limit),
                      "built": "", "cause": ""}
                     for kpi, score, limit in rows]}


def changes(result, baseline):
    """Relative change of each headline KPI from a baseline result."""
    if baseline.get("model_sha256") != result.get("model_sha256") or baseline.get("suite_sha256") != result.get("suite_sha256"):
        return {"comparable": False, "reason": "different prepared data or suite"}

    def ratio(now, before):
        return now / before - 1 if now and before else None
    q_now, q_before = result.get("quality") or {}, baseline.get("quality") or {}
    worst = lambda q: max((p["kld_mean"] for p in q.get("paths", {}).values()), default=None)  # noqa: E731
    same_text = decode_text_identity(result) == decode_text_identity(baseline)
    same_context = (result.get("context") or {}).get("tokens") == (baseline.get("context") or {}).get("tokens")
    same_context_text = same_context and context_text_identity(result) == context_text_identity(baseline)
    long_now = (result.get("headroom") or {}).get("context") or {}
    long_before = (baseline.get("headroom") or {}).get("context") or {}
    return {"comparable": True, "label": baseline.get("label") or (baseline.get("engine") or {}).get("name"),
            # Decode depends on the text decoded: scores on different texts are not compared.
            "decode_text_differs": not same_text,
            "decode_score": ratio((result.get("speed") or {}).get("decode_score"),
                                  (baseline.get("speed") or {}).get("decode_score")) if same_text else None,
            "prefill_score": ratio(((result.get("headroom") or {}).get("prefill") or {}).get("score"),
                                   ((baseline.get("headroom") or {}).get("prefill") or {}).get("score")),
            # Faster first tokens count as a gain, like the other scores.
            "ttft_score": ratio((baseline.get("speed") or {}).get("ttft_score_ms"),
                                (result.get("speed") or {}).get("ttft_score_ms")),
            # Long-context scores compare only at the same required context, and decode only on the same text.
            "context_decode": ratio(long_now.get("decode", {}).get("score"), long_before.get("decode", {}).get("score"))
            if same_context_text else None,
            "context_ttft": ratio(long_before.get("ttft", {}).get("score_ms"), long_now.get("ttft", {}).get("score_ms"))
            if same_context else None,
            "worst_kld_mean": ratio(worst(q_now), worst(q_before)),
            "heldout_decode": ratio((result.get("heldout") or {}).get("decode_score"),
                                    (baseline.get("heldout") or {}).get("decode_score"))}


def context_text_identity(result):
    """The long-context points' pinned continuation sha256, or "generated" (own output; results from before
    the points replayed one)."""
    text = (result.get("context") or {}).get("decode_text") or {"source": "generated"}
    return text.get("sha256") if text.get("source") == "pinned" else "generated"


def decode_text_identity(result):
    """The pinned continuation's sha256, or "generated" when each engine decoded its own output (results from
    before pinned continuations decoded their own output)."""
    text = (result.get("speed") or {}).get("decode_text") or {"source": "generated"}
    return text.get("sha256") if text.get("source") == "pinned" else "generated"


def _pct(value):
    return "–" if value is None else f"{value:.0%}"


def _num(value, spec):
    return "–" if value is None else format(value, spec)


def render(status):
    k = status["kpis"]
    d, p, t, q = k["decode"], k["prefill"], k["ttft"], k["quality"]
    marks = {"pass": "✓", "fail": "✗", "unmeasured": "?"}
    lines = ["gates: " + " · ".join(f"{g['name']} {marks[g['state']]}" for g in status["gates"])]
    for g in status["gates"]:
        if g["state"] != "pass":
            lines.append(f"  {g['name']}: {g['detail']}")
    if status.get("goals"):
        lines.append("goals for the next candidate, all of them at once: each KPI to its physical limit")
        for g in status["goals"]:
            spec = ".1f" if g["kpi"] in ("decode", "context_decode") else ",.0f"
            if g["change"] is None:
                lines.append(f"  {g['kpi']:<8} {_num(g['score'], spec)} {g['unit']}, limit unknown (see missing)")
            else:
                lines.append(f"  {g['kpi']:<8} {format(g['score'], spec)} -> {format(g['limit'], spec)} {g['unit']} "
                             f"({g['change']:+.0%})")
    lines.append("KPIs against their physical limits:")
    e = k.get("end_to_end") or {}
    decode = f"  decode   {_num(d['score'], '.1f')} tok/s"
    if d["ceiling"]:
        decode += (f", {_pct(d['fraction_of_ceiling'])} of its ceiling {d['ceiling']:.1f} "
                   f"(this drafter at floor costs {_num(d['speed_of_light'], '.1f')}"
                   + (f", later drafts accepted like the first {d['acceptance_bound']:.1f}" if d.get("acceptance_bound") else "")
                   + ")")
    if d["pass_efficiency"] is not None:
        decode += f"; model pass {d['pass_ms']:.2f} ms, {_pct(d['pass_efficiency'])} of its {d['pass_floor_ms']:.2f} ms floor"
    if e.get("decode_noise") is not None:
        decode += f"; run-to-run spread {e['decode_noise']:.1%}"
    if d["complete"] is False:
        decode += "; incomplete (skipped cases)"
    lines.append(decode)
    prefill = f"  prefill  {_num(p['score'], ',.0f')} tok/s"
    if p["ceiling"]:
        prefill += f", {_pct(p['fraction_of_ceiling'])} of its compute ceiling {p['ceiling']:,.0f}"
    for cid, row in p["cases"].items():
        if row.get("ttft_ms"):
            prefill += f"; {cid} {row['ttft_ms']:,.0f} ms to first token"
            if row.get("floor_ms"):
                prefill += f" vs {row['floor_ms']:,.0f} ms floor"
    lines.append(prefill)
    ttft = f"  ttft     {_num(t['score_ms'], ',.0f')} ms geometric mean over {len(t['cases'])} cases"
    if t["floor_ms"]:
        ttft += f", {_pct(t['fraction_of_floor'])} of its {t['floor_ms']:,.0f} ms floor"
        ttft += "".join(f"; {cid} {row['ttft_ms']:,.0f} ms vs {row['floor_ms']:,.1f} ms ({row['bound']} bound)"
                        for cid, row in t["cases"].items() if row.get("floor_ms"))
    if e.get("ttft_noise") is not None:
        ttft += f"; run-to-run spread {e['ttft_noise']:.1%}"
    lines.append(ttft)
    c = k.get("context")
    if c and c.get("required"):
        line = (f"  context  {c['required']:,} tokens required ({c['source']}"
                + (f", {c['bound']} bound" if c.get("bound") else "") + ")")
        if c.get("engine_max_context") is not None:
            line += f"; engine holds {c['engine_max_context']:,}"
        cd, ct = c.get("decode") or {}, c.get("ttft") or {}
        if cd.get("score"):
            line += f"; decode {cd['score']:.1f} tok/s over {len(c['points'])} points"
            if cd.get("ceiling_score"):
                line += f", {_pct(cd['fraction_of_ceiling'])} of its ceiling {cd['ceiling_score']:.1f}"
        if ct.get("score_ms"):
            line += f"; first token {ct['score_ms']:,.0f} ms"
            if ct.get("floor_ms"):
                line += f", {_pct(ct['fraction_of_floor'])} of its {ct['floor_ms']:,.0f} ms floor"
        line += "".join(f"; {int(key):,} not measured: {why}" for key, why in (c.get("not_measured") or {}).items())
        low = c.get("lower")
        if low:
            line += (f"; at {low['context']:,} tokens decode {_change(low.get('decode_score'), low.get('decode_score_at_required'))}, "
                     f"ttft {_change(low.get('ttft_score_ms'), low.get('ttft_score_ms_at_required'))}")
        lines.append(line)
    elif c is not None:
        lines.append("  context  unmeasured (no required context)")
    h = k.get("heldout")
    if h:
        lines.append(f"  heldout  {_num(h['decode_score'], '.1f')} tok/s decode over {len(h['prompts'])} held-out prompts, "
                     f"{_num(h['tokens_per_round'], '.2f')} tokens/round (geometric means): "
                     + ", ".join(f"{pid} {_num(row['decode_tps'], '.0f')}" for pid, row in h["prompts"].items()))
    if q:
        used = "" if q.get("budget_used") is None else f"{q['budget_used']:.1%} of the drift budget used ({q['budget_used_path']}); "
        lines.append(f"  quality  {'pass' if q['passed'] else 'FAIL'}: {used}worst KLD mean {q['worst_kld_mean']:.1e} "
                     f"({q['worst_kld_mean_path']}, limit {q['limits']['kld_mean']}), p99 {q['worst_kld_p99']:.1e} "
                     f"(limit {q['limits']['kld_p99']}), PPL ×{q['worst_ppl_ratio']:.4f} (limit {q['limits']['ppl_ratio']}), "
                     f"top-1 {q['lowest_top1']:.3f}")
    else:
        lines.append("  quality  unmeasured")
    if e.get("score"):
        line = f"  end-to-end {e['score']:.1f} tok/s over whole requests"
        if e.get("ceiling"):
            line += f", {_pct(e['fraction_of_ceiling'])} of its ceiling {e['ceiling']:.1f}"
        if e.get("noise") is not None:
            line += f"; run-to-run spread {e['noise']:.1%}"
        lines.append(line + " (mixes the KPIs by the suite's cases)")
    if status["gaps"]:
        lines.append("gaps: what each term would give its own KPI at its floor, everything else as measured:")
        lines += [f"  {gap['kpi']:<8} +{gap['projected_gain']:.0%} {gap['kpi']} score  {gap['name']}"
                  + (f": {gap['detail']}" if gap.get("detail") else "") for gap in status["gaps"]]
    if status["missing"]:
        lines.append("missing: " + "; ".join(status["missing"]))
    lines += [f"note: {n}" for n in status["notes"]]
    base = status.get("baseline")
    if base:
        if base["comparable"]:
            decode = ("not compared (decoded a different text)" if base.get("decode_text_differs")
                      else _signed(base["decode_score"]))
            long = "".join(f", {name} {_signed(base[key])}" for key, name in (("context_decode", "context decode"),
                                                                              ("context_ttft", "context ttft"))
                           if base.get(key) is not None)
            held = f", held-out decode {_signed(base['heldout_decode'])}" if base.get("heldout_decode") is not None else ""
            lines.append(f"vs {base['label']}: decode {decode}, prefill {_signed(base['prefill_score'])}, "
                         f"ttft {_signed(base['ttft_score'])} (speed){long}{held}, worst KLD mean {_signed(base['worst_kld_mean'])}")
        else:
            lines.append(f"baseline not comparable: {base['reason']}")
    return "\n".join(lines)


def _signed(value):
    return "–" if value is None else f"{value:+.1%}"


def status_of(path, protocol, baseline_path=None):
    result = load_json(path)
    if result.get("aborted"):
        raise SystemExit(f"{path} is an aborted round with no status: {result['aborted']['detail']}")
    baseline = load_json(baseline_path) if baseline_path else None
    return build(result, protocol, baseline)
