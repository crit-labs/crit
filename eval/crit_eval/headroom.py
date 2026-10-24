"""Distance to the physical limits, and where the time goes against each floor.

Decode speed is tokens per round divided by round time. A full model pass (one ordinary
step, or a verify of a few columns while it stays memory bound) cannot beat bytes read per
pass divided by achievable bandwidth; a draft step cannot beat its own bytes divided by
bandwidth. Bytes per pass are the adapter's short-context figure plus the KV cache read at
each case's context. With greedy acceptance, the drafts accepted at width K also tell the
tokens per round at every smaller width, so one run at a generous width projects the curve:

    tokens(k)   = 1 + sum_{j<=k} P(accepted >= j)
    round(k)    = pass + k * draft
    speed(k)    = tokens(k) / round(k)

At measured pass and draft costs this gives the best width now; at the floors it gives the
drafter-bound speed of light: how fast this drafter could go. A better drafter raises tokens per
round, so the decode ceiling also projects a drafter whose later drafts are accepted as often as
its first (the same conditional rate at every depth, up to a wider width):

    tokens*(k)  = 1 + c1 + c1^2 + ... + c1^k      with c1 = P(accepted >= 1)

The decode ceiling is the larger of the two at floor costs. An engine that does not speculate
decodes one token per model pass, so its ceiling is every pass at its floor: the geometric mean of
1 / pass floor over the cases.

Prefill is compute bound: its floor is the model's FLOPs over the device's GEMM rates per
weight format (see ceiling.py). The prefill score is the geometric mean of prefill tok/s over
the protocol's prefill cases, and its ceiling the same mean at the floor. Time to first token
is measured on every case against the larger of its compute floor and one read of the weights
at the device bandwidth; the TTFT score is the geometric mean in ms (lower is better).

The end-to-end score is per case output tokens over the whole request (time to first token
plus decode), as a geometric mean over the cases. Its ceiling puts time to first token at its
floor and decode at its ceiling. It mixes the KPIs by the suite's case mix, so it is reported as
a fact and never used to weigh one KPI against another.

Long context: Eval measures one request at each point of a curve up to the required context (see
context.py). Each point's round has a floor of its pass floor, with the KV cache read at that
context, plus the draft steps at their floor; tokens per round as measured. Its decode score is the
geometric mean of decode tok/s over the points against the same mean at the floors, and its time to
first token the geometric mean over the points against their TTFT floors.

A round is a verify pass over the drafts plus the draft steps. When generate() reports the engine's own
time in verify passes and in draft steps (`verify_ms`, `draft_ms`), the two are separate gaps: draft steps
against the drafter's bandwidth floor, and the verify pass against the ordinary one-token pass (while
the verify stays memory bound, its extra columns read the same weights, so their floor adds nothing).
Without them, round time beyond one ordinary pass per draft is one combined term.

From crit-eval-v7 the decode terms (cases, projection, decode gaps, decode noise and the end-to-end
score) describe the sampled speed suite, the decode real requests get (decode_suite); prefill and
time to first token stay on the greedy suite's cases. A result without a measured sampled suite
(v6, or an engine that cannot sample) keeps them on the greedy suite. Rejection sampling lets a
drafter speculate under sampling too, so the sampled decode's acceptance bound takes each case's
first-draft acceptance from the better of the two suites, and a sampled case that drafts nothing
projects from its measured tokens per round. One more decode gap compares the suites directly:
every sampled case as fast as its greedy one.

Gaps state what one term would give its own KPI at its floor, with everything else as measured.
They are facts in a fixed order (decode terms, prefill, time to first token), not a ranking:
Eval does not judge or classify speed. Noise is the spread of each score across the measured
repetitions.
"""
from . import ceiling as model_ceiling
from . import metrics


def decode_suite(result):
    """(name, section) of the suite the decode KPI, its headroom, the goals and stop records follow: the sampled
    speed suite when it measured a decode score (crit-eval-v7), else the greedy speed suite."""
    sampled = result.get("sampled_speed") or {}
    if sampled.get("status") == "measured" and sampled.get("decode_score"):
        return "sampled", sampled
    return "greedy", result.get("speed") or {}


def pass_floor_ms(bytes_per_pass, bandwidth_gbs):
    if not bytes_per_pass or not bandwidth_gbs:
        return None
    return bytes_per_pass / (bandwidth_gbs * 1e9) * 1000


def tokens_by_width(accepted, width):
    """Expected tokens per round for widths 0..width from per-round accepted-draft counts."""
    if not accepted:
        return None
    rounds = len(accepted)
    curve, total = [1.0], 1.0
    for j in range(1, width + 1):
        total += sum(1 for a in accepted if a >= j) / rounds
        curve.append(total)
    return curve


def conditional_acceptance(accepted, width):
    """P(accepted >= j | accepted >= j - 1) for depths j = 1..width."""
    rates, reached = [], len(accepted)
    for j in range(1, width + 1):
        count = sum(1 for a in accepted if a >= j)
        rates.append(count / reached if reached else 0.0)
        reached = count
    return rates


def tokens_at_rate(rate, width):
    """Tokens per round for widths 0..width when every depth is accepted at the same conditional rate."""
    curve, reach = [1.0], 1.0
    for _ in range(width):
        reach *= rate
        curve.append(curve[-1] + reach)
    return curve


def best_width(curves, pass_ms, draft_ms):
    """Width maximizing the geometric mean over cases, and that mean (tok/s). pass_ms is one
    value or one per curve."""
    passes = pass_ms if isinstance(pass_ms, (list, tuple)) else [pass_ms] * len(curves)
    best = (0, None)
    for k in range(len(min(curves, key=len))):
        score = metrics.geometric_mean([1000 * c[k] / (p + k * draft_ms) for c, p in zip(curves, passes)])
        if score and (best[1] is None or score > best[1]):
            best = (k, score)
    return best


def _context(case):
    return (case.get("prompt_tokens") or 0) + (case.get("output_tokens") or 0) / 2


def _projected_gain(cases, score, new_round):
    """Decode-score gain when each case's round time becomes new_round(case_id, row), and each
    case's decode speed-up factor."""
    if not score:
        return None, {}
    speeds, factors = [], {}
    for case_id, row in cases.items():
        rnd = row.get("round_ms")
        changed = new_round(case_id, row) if rnd else None
        factors[case_id] = rnd / changed if changed and changed > 0 else 1.0
        speeds.append(row["decode_tps"] * factors[case_id])
    projected = metrics.geometric_mean(speeds)
    return (projected / score - 1 if projected else None), factors


def _rows(speed):
    """Measured cases with the parts of a request: time to first token and decode time."""
    rows = {}
    for case_id, case in (speed or {}).get("cases", {}).items():
        n = case.get("output_tokens")
        if case.get("status") != "measured" or not case.get("ttft_ms") or not case.get("decode_tps") or not n:
            continue
        rows[case_id] = {"output_tokens": n, "ttft_ms": case["ttft_ms"], "decode_ms": (n - 1) * 1000 / case["decode_tps"]}
    return rows


def _e2e_score(rows, ttft=None, decode=None):
    """Geometric mean of output tokens per second of whole request; ttft/decode override per case."""
    ttft, decode = ttft or {}, decode or {}
    return metrics.geometric_mean([r["output_tokens"] * 1000 / (ttft.get(cid, r["ttft_ms"]) + decode.get(cid, r["decode_ms"]))
                                   for cid, r in rows.items()])


def spread(values):
    """(max - min) / median of per-repetition scores; None with fewer than two."""
    values = [v for v in values if v]
    if len(values) < 2:
        return None
    middle = metrics.median(values)
    return (max(values) - min(values)) / middle if middle else None


def noise(speed, prefill_cases=()):
    """Run-to-run spread of the decode, TTFT, end-to-end and prefill scores across measured repetitions (the
    prefill score over prefill_cases)."""
    cases = {cid: c for cid, c in (speed or {}).get("cases", {}).items() if c.get("status") == "measured"}
    reps = {cid: [s for s in c.get("samples", []) if not s.get("warmup")] for cid, c in cases.items()}
    if not reps or min(len(r) for r in reps.values()) < 2:
        return {"decode": None, "ttft": None, "end_to_end": None, "prefill": None}
    count = min(len(r) for r in reps.values())
    decode, ttft, e2e, prefill = [], [], [], []
    for i in range(count):
        row = [r[i] for r in reps.values()]
        decode.append(metrics.geometric_mean([s["decode_tps"] for s in row if s.get("decode_tps")]))
        ttft.append(metrics.geometric_mean([s["ttft_ms"] for s in row if s.get("ttft_ms")]))
        e2e.append(metrics.geometric_mean([s["output_tokens"] * 1000 / s["request_ms"] for s in row
                                           if s.get("request_ms") and s.get("output_tokens")]))
        prefill.append(metrics.geometric_mean([reps[cid][i]["prefill_tps"] for cid in prefill_cases
                                               if cid in reps and reps[cid][i].get("prefill_tps")]))
    return {"decode": spread(decode), "ttft": spread(ttft), "end_to_end": spread(e2e),
            "prefill": spread(prefill) if all(cid in reps for cid in prefill_cases) and prefill_cases else None}


def _noises(speed, decode_speed, prefill_cases):
    """Each KPI's run-to-run spread. The long-context curve runs once per point, so its decode and first token take
    the device's spread from the speed suite's repetitions: the greedy decode's and the prefill's."""
    greedy = noise(speed, prefill_cases)
    return {**greedy, "decode": noise(decode_speed)["decode"], "end_to_end": noise(decode_speed)["end_to_end"],
            "context_decode": greedy["decode"], "context_ttft": greedy["prefill"]}


def end_to_end_summary(speed, ttft, decode_score, speed_of_light):
    """End-to-end score now and at its ceiling (first token at its floor, decode at speed of light)."""
    rows = _rows(speed)
    score = _e2e_score(rows) if rows else None
    ceiling = None
    floors = {cid: ((ttft or {}).get("cases", {}).get(cid, {}).get("floor") or {}).get("total_ms") for cid in rows}
    if score and speed_of_light and decode_score and all(floors.values()):
        ceiling = _e2e_score(rows, floors, {cid: r["decode_ms"] * decode_score / speed_of_light for cid, r in rows.items()})
    return {"score": score, "ceiling_score": ceiling, "fraction_of_ceiling": score / ceiling if score and ceiling else None,
            "cases": {cid: {**r, "tps": r["output_tokens"] * 1000 / (r["ttft_ms"] + r["decode_ms"])} for cid, r in rows.items()}}


def prefill_summary(speed, ceiling, tflops, prefill_cases):
    cases, missing, speeds, ceilings = {}, [], [], []
    for case_id in prefill_cases:
        case = (speed or {}).get("cases", {}).get(case_id)
        if not case or case.get("status") != "measured":
            missing.append(f"a measurement of prefill case {case_id}")
            continue
        row = {"prompt_tokens": case["prompt_tokens"], "ttft_ms": case.get("ttft_ms"), "prefill_tps": case.get("prefill_tps")}
        floor, lacking = model_ceiling.prefill_floor_ms(ceiling, tflops, case["prompt_tokens"])
        missing += [m for m in lacking if m not in missing]
        if floor:
            row["floor"] = floor
            row["ceiling_tps"] = case["prompt_tokens"] / floor["total_ms"] * 1000
            if case.get("ttft_ms"):
                row["efficiency"] = floor["total_ms"] / case["ttft_ms"]
            ceilings.append(row["ceiling_tps"])
        if case.get("prefill_tps"):
            speeds.append(case["prefill_tps"])
        cases[case_id] = row
    score = metrics.geometric_mean(speeds) if speeds and len(speeds) == len(prefill_cases) else None
    ceiling_score = metrics.geometric_mean(ceilings) if ceilings and len(ceilings) == len(prefill_cases) else None
    fraction = score / ceiling_score if score and ceiling_score else None
    return {"cases": cases, "score": score, "ceiling_score": ceiling_score, "fraction_of_ceiling": fraction,
            "missing": missing}


def ttft_summary(info, speed, ceiling, tflops, bandwidth_gbs):
    cases, missing, measured, floors = {}, [], [], []
    for case_id, case in (speed or {}).get("cases", {}).items():
        if case.get("status") != "measured" or not case.get("ttft_ms"):
            continue
        row = {"prompt_tokens": case["prompt_tokens"], "ttft_ms": case["ttft_ms"]}
        floor, lacking = model_ceiling.ttft_floor_ms(ceiling, tflops, bandwidth_gbs, case["prompt_tokens"],
                                                    info.get("bytes_per_pass"))
        missing += [m for m in lacking if m not in missing]
        if floor:
            row["floor"] = floor
            row["efficiency"] = floor["total_ms"] / case["ttft_ms"]
            floors.append(floor["total_ms"])
        measured.append(case["ttft_ms"])
        cases[case_id] = row
    score = metrics.geometric_mean(measured)
    floor_score = metrics.geometric_mean(floors) if floors and len(floors) == len(measured) else None
    efficiency = floor_score / score if score and floor_score else None
    return {"cases": cases, "score_ms": score, "floor_ms": floor_score, "fraction_of_floor": efficiency,
            "missing": missing}


def summarize(info, speed, step_ms, bandwidth_gbs, thresholds, ceiling=None, tflops=None, steps=None,
              prefill_cases=(), context=None, context_settings=None, decode_speed=None, decode_name="greedy"):
    """info: adapter info(); speed: run_speed() result; step_ms: median ordinary one-token step
    in ms (None when unmeasured); bandwidth_gbs: achievable device bandwidth in GB/s; ceiling:
    the prepared model arithmetic (ceiling.model_ceiling); tflops: GEMM rates by format; steps:
    ordinary step ms by case; prefill_cases: case ids in the prefill score; context: the result's
    context section (the requirement and the measured curve); context_settings: the protocol's;
    decode_speed: the suite the decode terms describe (decode_suite), named decode_name; speed by default."""
    decode_speed = decode_speed or speed
    width = info.get("draft_width")
    bandwidth_gbs = bandwidth_gbs or info.get("bandwidth_gbs")
    floor = pass_floor_ms(info.get("bytes_per_pass"), bandwidth_gbs)
    draft_floor = pass_floor_ms(info.get("draft_bytes_per_step"), bandwidth_gbs)
    kv_per_token = info.get("kv_bytes_per_token") or (ceiling or {}).get("kv_bytes_per_token") or 0
    steps = steps or {}
    cases, warnings, curves, case_floors, missing, bound_curves = {}, [], [], [], [], []
    bound_width = max(width or 0, thresholds.get("acceptance_ceiling_width", 8))
    for case_id, case in (decode_speed or {}).get("cases", {}).items():
        if case.get("status") != "measured":
            continue
        row = {"round_ms": case.get("round_ms"), "tokens_per_round": case.get("tokens_per_round"),
               "decode_tps": case.get("decode_tps"), "step_ms": steps.get(case_id) or step_ms}
        if width and case.get("tokens_per_round"):
            row["saturation"] = case["tokens_per_round"] / (width + 1)
        if floor:
            row["pass_floor_ms"] = pass_floor_ms(info["bytes_per_pass"] + kv_per_token * _context(case), bandwidth_gbs)
            if case.get("round_ms"):
                row["round_over_pass_floor"] = case["round_ms"] / row["pass_floor_ms"]
        measured_samples = [s for s in case.get("samples", []) if not s.get("warmup")]
        parts = [(s["verify_ms"] / s["rounds"], s["draft_ms"] / (s["rounds"] * width)) for s in measured_samples
                 if width and s.get("rounds") and s.get("verify_ms") is not None and s.get("draft_ms") is not None]
        if parts and len(parts) == len(measured_samples):
            row["verify_ms"] = metrics.median([v for v, _ in parts])
            row["draft_step_ms"] = metrics.median([d for _, d in parts])
        accepted = [a for s in measured_samples for a in (s.get("accepted") or [])]
        greedy = (speed or {}).get("cases", {}).get(case_id) or {} if decode_speed is not speed else {}
        greedy_accepted = [a for s in greedy.get("samples", []) if not s.get("warmup") for a in (s.get("accepted") or [])]
        if width and (accepted or greedy_accepted):
            row["tokens_by_width"] = (tokens_by_width(accepted, width) if accepted
                                      else [case.get("tokens_per_round") or 1.0] * (width + 1))
            rate = 0.0
            if accepted:
                row["conditional_acceptance"] = conditional_acceptance(accepted, width)
                rate = row["conditional_acceptance"][0]
            if greedy_accepted:
                rate = max(rate, conditional_acceptance(greedy_accepted, width)[0])
            curves.append(row["tokens_by_width"])
            bound_curves.append(tokens_at_rate(rate, bound_width))
            case_floors.append(row.get("pass_floor_ms"))
        cases[case_id] = row

    efficiency = floor / step_ms if floor and step_ms else None
    if efficiency is None:
        lacking = [name for name, value in (("bytes_per_pass from the adapter", info.get("bytes_per_pass")),
                                            ("device bandwidth (--device)", bandwidth_gbs),
                                            ("an ordinary step (exactness run or plain decoding)", step_ms)) if not value]
        warnings.append("pass efficiency unmeasured: missing " + ", ".join(lacking))
        missing += lacking
    else:
        warnings.append(f"model pass {step_ms:.2f} ms is {efficiency:.0%} of its {floor:.2f} ms bandwidth floor")

    # Draft cost per step: reported by the adapter, else what a round costs beyond one pass.
    rounds = [c["round_ms"] for c in cases.values() if c.get("round_ms")]
    draft_ms = info.get("draft_ms")
    if draft_ms is None and width and step_ms and rounds:
        draft_ms = max(0.0, (metrics.median(rounds) - step_ms) / width)
    projection = None
    score = (decode_speed or {}).get("decode_score")
    if curves and step_ms and draft_ms is not None:
        now_width, now_score = best_width(curves, step_ms, draft_ms)
        projection = {"draft_ms": draft_ms, "draft_floor_ms": draft_floor,
                      "best_width_now": now_width, "score_at_best_width_now": now_score}
        if floor and draft_floor is not None:
            floors = case_floors if all(case_floors) else floor
            sol_width, sol_score = best_width(curves, floors, draft_floor)
            bound_at, bound_score = best_width(bound_curves, floors, draft_floor)
            decode_ceiling = max(sol_score, bound_score or 0)
            projection.update(best_width_at_floor=sol_width, speed_of_light_score=sol_score,
                              fraction_of_speed_of_light=score / sol_score if score else None,
                              acceptance_bound_score=bound_score, acceptance_bound_width=bound_at,
                              ceiling_score=decode_ceiling, fraction_of_ceiling=score / decode_ceiling if score else None)
            if score:
                warnings.append(f"{decode_name} decode score {score:.1f} is {score / decode_ceiling:.0%} of its ceiling {decode_ceiling:.1f} "
                                f"(this drafter at floor costs: {sol_score:.1f} at width {sol_width}; later drafts "
                                f"accepted like the first: {bound_score:.1f} at width {bound_at})")
        if width in (projection.get("best_width_now"), projection.get("best_width_at_floor")):
            warnings.append(f"best projected width is the widest measured ({width}); "
                            "measure acceptance at a wider draft to see beyond it")
            missing.append(f"acceptance beyond draft width {width} (run with a wider draft)")
    elif width:
        warnings.append("speed-of-light projection needs per-round `accepted` counts from generate() and an ordinary step")
        if not curves:
            missing.append("per-round `accepted` counts from generate()")
    elif not info.get("speculative") and floor and cases:
        # A plain engine decodes one token per model pass: its ceiling is every pass at its bandwidth floor.
        ceiling_score = metrics.geometric_mean([1000 / row["pass_floor_ms"] for row in cases.values()])
        projection = {"basis": "one token per pass", "ceiling_score": ceiling_score,
                      "fraction_of_ceiling": score / ceiling_score if score else None}
        if score:
            warnings.append(f"{decode_name} decode score {score:.1f} is {score / ceiling_score:.0%} of its ceiling "
                            f"{ceiling_score:.1f} (one token per model pass at its bandwidth floor)")
    if width and info.get("draft_bytes_per_step") is None:
        missing.append("draft_bytes_per_step from the adapter")
    if width and cases and not all(row.get("verify_ms") is not None for row in cases.values()):
        missing.append("verify_ms and draft_ms from generate() (separates the draft steps from the verify pass)")

    gaps = decode_gaps(cases, score, width, draft_floor, projection)
    gaps += acceptance_gaps(cases, score, width, bound_width, step_ms, draft_ms)
    if decode_speed is not speed:
        gaps += sampling_gaps(decode_speed, speed, score)
    prefill = prefill_summary(speed, ceiling, tflops, prefill_cases)
    if prefill["fraction_of_ceiling"] is not None:
        parts = [r["floor"] for r in prefill["cases"].values() if r.get("floor")]
        detail = "; ".join(
            f"{cid} {r['ttft_ms']:.0f} ms vs {r['floor']['total_ms']:.0f} ms floor (matmul {r['floor']['matmul_ms']:.0f}, "
            f"attention {r['floor']['attention_ms']:.0f}, recurrent {r['floor']['recurrent_ms']:.0f})"
            for cid, r in prefill["cases"].items() if r.get("floor") and r.get("ttft_ms"))
        if parts and prefill["fraction_of_ceiling"] < 1:
            gaps.append({"kpi": "prefill", "name": "prefill at its compute ceiling",
                         "projected_gain": 1 / prefill["fraction_of_ceiling"] - 1, "detail": detail,
                         "cases": list(prefill["cases"])})
        warnings.append(f"prefill score {prefill['score']:.0f} tok/s is {prefill['fraction_of_ceiling']:.0%} "
                        f"of its compute ceiling {prefill['ceiling_score']:.0f} tok/s")
    missing += [m for m in prefill["missing"] if m not in missing]
    ttft = ttft_summary(info, speed, ceiling, tflops, bandwidth_gbs)
    if ttft["fraction_of_floor"] is not None:
        slowest = sorted(((cid, r) for cid, r in ttft["cases"].items() if r.get("efficiency")),
                         key=lambda item: item[1]["efficiency"])[:3]
        if ttft["fraction_of_floor"] < 1:
            gaps.append({"kpi": "ttft", "name": "time to first token at its floor",
                         "projected_gain": 1 / ttft["fraction_of_floor"] - 1, "cases": list(ttft["cases"]),
                         "detail": "furthest from the floor: " + ", ".join(
                             f"{cid} {r['ttft_ms']:,.0f} ms vs {r['floor']['total_ms']:,.0f} ms ({r['floor']['bound']} bound)"
                             for cid, r in slowest)})
        warnings.append(f"time to first token {ttft['score_ms']:,.0f} ms (geometric mean) is {ttft['fraction_of_floor']:.0%} "
                        f"of its {ttft['floor_ms']:,.0f} ms floor")
    missing += [m for m in ttft["missing"] if m not in missing]
    decode_ceiling = (projection or {}).get("ceiling_score")
    end_to_end = end_to_end_summary(decode_speed, ttft, score, decode_ceiling)
    long_context = None
    if (context or {}).get("points"):
        long_context = context_summary(info, context, ceiling, tflops, bandwidth_gbs, draft_floor, kv_per_token,
                                       context_settings or {})
        gaps += long_context.pop("gaps")
        missing += [m for m in long_context["missing"] if m not in missing]
        if long_context["decode"]["fraction_of_ceiling"] is not None:
            warnings.append(f"long-context decode {long_context['decode']['score']:.1f} tok/s is "
                            f"{long_context['decode']['fraction_of_ceiling']:.0%} of its ceiling "
                            f"{long_context['decode']['ceiling_score']:.1f}")
    return {
        "decode_suite": decode_name, "draft_width": width, "bytes_per_pass": info.get("bytes_per_pass"), "bandwidth_gbs": bandwidth_gbs,
        "kv_bytes_per_token": kv_per_token or None,
        "pass_floor_ms": floor, "step_ms": step_ms, "pass_efficiency": efficiency,
        "projection": projection, "cases": cases, "prefill": prefill, "ttft": ttft, "gaps": gaps,
        "end_to_end": end_to_end, "noise": _noises(speed, decode_speed, prefill_cases), "context": long_context,
        "missing": missing, "warnings": warnings,
    }


def context_summary(info, context, ceiling, tflops, bandwidth_gbs, draft_floor, kv_per_token, settings):
    """Decode and time to first token over the long-context curve against their floors, and the gaps:
    rounds at their floor, the KV cache at the protocol's precision (with everything else as measured)
    and the first token at its floor."""
    width = info.get("draft_width") or 0
    points, missing, gaps = {}, [], []
    for key, point in sorted(context["points"].items(), key=lambda item: item[1]["context"]):
        if point.get("status") != "measured":
            continue
        row = {k: point.get(k) for k in ("context", "prompt_tokens", "output_tokens", "decode_tps", "ttft_ms",
                                          "prefill_tps", "tokens_per_round", "round_ms")}
        reading = point["prompt_tokens"] + point["output_tokens"] / 2
        pass_floor = pass_floor_ms(info["bytes_per_pass"] + kv_per_token * reading, bandwidth_gbs) if info.get(
            "bytes_per_pass") else None
        if pass_floor is not None and (not width or draft_floor is not None):
            row["round_floor_ms"] = pass_floor + width * (draft_floor or 0.0)
            row["ceiling_tps"] = (point.get("tokens_per_round") or 1.0) * 1000 / row["round_floor_ms"]
        floor, lacking = model_ceiling.ttft_floor_ms(ceiling, tflops, bandwidth_gbs, point["prompt_tokens"],
                                                    info.get("bytes_per_pass"))
        missing += [m for m in lacking if m not in missing]
        if floor:
            row["ttft_floor"] = floor
        points[key] = row
    if width and draft_floor is None:
        missing.append("draft_bytes_per_step from the adapter (long-context round floor)")
    if not info.get("kv_bytes_per_token"):
        missing.append("kv_bytes_per_token from the adapter (the KV bytes each token adds at the engine's KV precision)")
    complete = bool(points) and len(points) == len(context["points"])
    decode = [r["decode_tps"] for r in points.values() if r.get("decode_tps")]
    score = metrics.geometric_mean(decode) if decode and len(decode) == len(points) else None
    ceilings = [r["ceiling_tps"] for r in points.values() if r.get("ceiling_tps")]
    ceiling_score = metrics.geometric_mean(ceilings) if ceilings and len(ceilings) == len(points) else None
    ttfts = [r["ttft_ms"] for r in points.values() if r.get("ttft_ms")]
    ttft_score = metrics.geometric_mean(ttfts) if ttfts and len(ttfts) == len(points) else None
    floors = [r["ttft_floor"]["total_ms"] for r in points.values() if r.get("ttft_floor")]
    floor_score = metrics.geometric_mean(floors) if floors and len(floors) == len(points) else None
    if score and ceiling_score and ceiling_score > score:
        gaps.append({"kpi": "context_decode", "name": "long-context rounds at their floor",
                     "projected_gain": ceiling_score / score - 1,
                     "detail": ", ".join(f"{r['context']:,} tokens {r['decode_tps']:.1f} vs {r['ceiling_tps']:.1f} tok/s"
                                         for r in points.values() if r.get("ceiling_tps"))})
    # The KV cache at the protocol's precision: each round reads fewer KV bytes, everything else as measured,
    # and no round faster than its floor at that precision.
    target = (ceiling or {}).get("kv_bytes_per_token")
    target = target / 2 * float(settings.get("kv_bytes_per_element", 1)) if target else None
    if (score and target and info.get("kv_bytes_per_token") and kv_per_token > target and bandwidth_gbs
            and all(r.get("round_ms") for r in points.values())):
        speeds = []
        for r in points.values():
            reading = r["prompt_tokens"] + r["output_tokens"] / 2
            saved = (kv_per_token - target) * reading / (bandwidth_gbs * 1e9) * 1000
            floor = 0.0
            if info.get("bytes_per_pass"):
                floor = pass_floor_ms(info["bytes_per_pass"] + target * reading, bandwidth_gbs) + width * (draft_floor or 0.0)
            projected = max(r["round_ms"] - saved, floor)
            if projected <= 0:
                speeds = []
                break
            speeds.append(r["decode_tps"] * r["round_ms"] / projected)
        gain = metrics.geometric_mean(speeds) / score - 1 if speeds else 0.0
        if gain > 0:
            gaps.append({"kpi": "context_decode", "name": f"KV cache at {settings.get('kv_precision', 'the protocol precision')}",
                         "projected_gain": gain,
                         "detail": f"{kv_per_token:,.0f} -> {target:,.0f} bytes per token read at every point"})
    if ttft_score and floor_score and ttft_score > floor_score:
        gaps.append({"kpi": "context_ttft", "name": "long-context time to first token at its floor",
                     "projected_gain": ttft_score / floor_score - 1,
                     "detail": ", ".join(f"{r['context']:,} tokens {r['ttft_ms']:,.0f} vs {r['ttft_floor']['total_ms']:,.0f} ms "
                                         f"({r['ttft_floor']['bound']} bound)" for r in points.values() if r.get("ttft_floor"))})
    return {"points": points, "complete": complete,
            "decode": {"score": score, "ceiling_score": ceiling_score,
                       "fraction_of_ceiling": score / ceiling_score if score and ceiling_score else None},
            "ttft": {"score_ms": ttft_score, "floor_ms": floor_score,
                     "fraction_of_floor": floor_score / ttft_score if ttft_score and floor_score else None},
            "missing": missing, "gaps": gaps}


def decode_gaps(cases, score, width, draft_floor, projection):
    """Decode-score gains if one term were at its floor, with everything else as measured."""
    gaps = []
    timed = {cid: row for cid, row in cases.items() if row.get("round_ms") and row.get("decode_tps")}
    if not score or not timed:
        return gaps
    if all(row.get("pass_floor_ms") and row.get("step_ms") for row in timed.values()):
        slow = {cid: row["step_ms"] - row["pass_floor_ms"] for cid, row in timed.items()}
        gain, factors = _projected_gain(timed, score, lambda cid, row: row["round_ms"] - max(0.0, slow[cid]))
        if gain and gain > 0:
            detail = ", ".join(f"{cid} {row['step_ms']:.2f} vs {row['pass_floor_ms']:.2f} ms"
                               for cid, row in timed.items())
            gaps.append({"kpi": "decode", "name": "model pass at its bandwidth floor", "projected_gain": gain,
                         "detail": "ordinary step vs floor: " + detail, "case_speedup": factors})
    split = width and all(row.get("verify_ms") is not None and row.get("draft_step_ms") is not None
                          for row in timed.values())
    if split and draft_floor is not None and any(row["draft_step_ms"] > draft_floor for row in timed.values()):
        gain, factors = _projected_gain(timed, score, lambda cid, row: row["round_ms"] - width * max(
            0.0, row["draft_step_ms"] - draft_floor))
        if gain and gain > 0:
            gaps.append({"kpi": "decode", "name": "draft steps at their bandwidth floor", "projected_gain": gain,
                         "detail": f"draft step {metrics.median([r['draft_step_ms'] for r in timed.values()]):.2f} ms "
                                   f"vs {draft_floor:.2f} ms floor (engine-measured)",
                         "case_speedup": factors})
    if split and all(row.get("step_ms") for row in timed.values()) and any(
            row["verify_ms"] > row["step_ms"] for row in timed.values()):
        gain, factors = _projected_gain(timed, score, lambda cid, row: row["round_ms"] - max(
            0.0, row["verify_ms"] - row["step_ms"]))
        if gain and gain > 0:
            detail = ", ".join(f"{cid} {row['verify_ms']:.2f} vs {row['step_ms']:.2f} ms" for cid, row in timed.items())
            gaps.append({"kpi": "decode", "name": f"verify pass of {width + 1} columns at the ordinary pass cost",
                         "projected_gain": gain, "detail": "verify pass vs ordinary step: " + detail,
                         "case_speedup": factors})
    if not split and width and draft_floor is not None and all(row.get("step_ms") for row in timed.values()):
        def at_floor(cid, row):
            draft = max(0.0, (row["round_ms"] - row["step_ms"]) / width)
            return row["round_ms"] - width * max(0.0, draft - draft_floor)
        gain, factors = _projected_gain(timed, score, at_floor)
        if gain and gain > 0:
            drafts = [max(0.0, (r["round_ms"] - r["step_ms"]) / width) for r in timed.values()]
            gaps.append({"kpi": "decode", "name": "draft steps and verify columns at the draft floor",
                         "projected_gain": gain,
                         "detail": f"(round - ordinary step) / width {metrics.median(drafts):.2f} ms vs {draft_floor:.2f} ms "
                                   "draft floor; this includes the verify pass's cost beyond an ordinary step",
                         "case_speedup": factors})
    if projection and projection.get("best_width_now") not in (None, width) and projection.get("score_at_best_width_now"):
        gain = projection["score_at_best_width_now"] / score - 1
        if gain > 0:
            gaps.append({"kpi": "decode", "name": f"draft width {width} -> {projection['best_width_now']}",
                         "projected_gain": gain, "detail": "at current pass and draft costs",
                         "case_speedup": {cid: 1 + gain for cid in timed}})
    return gaps


def sampling_gaps(sampled, greedy, score):
    """Sampled requests as fast as greedy ones: each sampled case at its greedy case's decode, where that is faster
    (speculation kept under sampling by rejection sampling)."""
    pairs = {cid: (case, (greedy or {}).get("cases", {}).get(cid) or {}) for cid, case in (sampled or {}).get("cases", {}).items()
             if case.get("status") == "measured" and case.get("decode_tps")}
    if not score or not pairs or not all(g.get("decode_tps") for _, g in pairs.values()):
        return []
    factors = {cid: max(1.0, g["decode_tps"] / s["decode_tps"]) for cid, (s, g) in pairs.items()}
    projected = metrics.geometric_mean([s["decode_tps"] * factors[cid] for cid, (s, _) in pairs.items()])
    gain = projected / score - 1 if projected else 0.0
    if gain <= 0:
        return []
    detail = ", ".join(f"{cid} {s['decode_tps']:.1f} vs {g['decode_tps']:.1f} tok/s"
                       + (f" ({s.get('tokens_per_round') or 0:.2f} vs {g.get('tokens_per_round') or 0:.2f} tokens/round)")
                       for cid, (s, g) in pairs.items())
    return [{"kpi": "decode", "name": "sampled requests decoded as fast as greedy ones (drafts verified by rejection sampling)",
             "projected_gain": gain, "detail": "sampled vs greedy: " + detail, "case_speedup": factors}]


def acceptance_gaps(cases, score, width, bound_width, step_ms, draft_ms):
    """A better drafter: later drafts accepted as often as the first, at current pass and draft costs."""
    timed = {cid: row for cid, row in cases.items() if row.get("conditional_acceptance") and row.get("decode_tps")}
    if not score or not width or not timed or not step_ms or draft_ms is None:
        return []
    passes = [row.get("step_ms") or step_ms for row in timed.values()]
    modeled = metrics.geometric_mean([1000 * row["tokens_by_width"][width] / (p + width * draft_ms)
                                      for row, p in zip(timed.values(), passes)])
    bound_at, bound_score = best_width([tokens_at_rate(row["conditional_acceptance"][0], bound_width)
                                        for row in timed.values()], passes, draft_ms)
    if not modeled or not bound_score or bound_score <= modeled:
        return []
    gain = bound_score / modeled - 1
    depth = [metrics.median([row["conditional_acceptance"][j] for row in timed.values()]) for j in range(width)]
    return [{"kpi": "decode", "name": "drafter acceptance and width: later drafts accepted as often as the first",
             "projected_gain": gain, "case_speedup": {cid: 1 + gain for cid in timed},
             "detail": "conditional acceptance by depth " + ", ".join(f"{d:.2f}" for d in depth)
                       + f"; at width {bound_at} with current costs"}]


def step_ms_from(samples):
    """Median ordinary one-token step from plain greedy speed samples."""
    return metrics.median([1000 / s["decode_tps"] for s in samples if s.get("decode_tps")])
