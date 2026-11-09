"""Candidate plans: written after the parent's Eval and before any code, frozen and checked by crit, and compared with
their outcome after the candidate's Eval.

crit checks a plan's completeness and its consistency with the facts it freezes (the parent's Eval), never whether
its reasoning is good, the line headroom_record holds for stop records. The priorities are crit's: decode and
prefill are p1 or p2 by which family is further from its limit, time to first token is p2 or p3 and never p1, and a
plan in which neither decode nor prefill is needed is a stop plan."""
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

from . import ladder
from .records import (counts_evaluated, digest, evaluated_candidates, file_sha, header, ident, keys, load_run, now,
                      number, quality_mode, quality_terms, read_json, record, safe_path, text)

KPIS = ("decode", "prefill", "ttft", "context_decode", "context_ttft")
HEADLINE = ("decode", "prefill", "ttft")
CONTEXT = ("context_decode", "context_ttft")
# Long-context decode counts toward decode, long-context first token toward prefill.
FAMILIES = {"decode": ("decode", "context_decode"), "prefill": ("prefill", "context_ttft")}
FAMILY = {member: family for family, members in FAMILIES.items() for member in members}
# Numbers outside the priorities whose drop beyond their noise an outcome flags, and whether lower is better.
UNTIERED = {"greedy_decode": False, "heldout_decode": False, "followup_ttft": True}
MS = {"ttft", "context_ttft"}
DECISIONS = {"build", "stop"}
CONFIDENCE = {"low", "medium", "high"}
# A speed lever is planned for its KPI changes; a requirement lever for a rule or gate it serves (`serves`), such as
# work the rules require that Eval does not score: it needs no change above the noise and never counts as a KPI lever.
LEVER_KINDS = {"speed", "requirement"}
MAX_ROWS = 200
# Eval results hold every case and quality row; they are read once, when a plan freezes one or an outcome compares.
EVAL_JSON_BYTES = 256 * 1024 * 1024


def rules(run):
    """The version of the plan rules a run was created with: 2 to 7 from `"plan_rules"` in its spec, else 1."""
    return run["spec"].get("plan_rules", 1)


def _get(value, *path):
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _positive(value):
    return value if type(value) in (int, float) and value > 0 else None


def extract(status):
    """The facts a plan freezes from an Eval result's stored status: per KPI the score, limit and fraction of the
    limit, the relative run-to-run noise per KPI (None when Eval does not report it), the untiered numbers and the
    gaps. Read defensively: a missing field is unknown, never an error."""
    status = status if isinstance(status, dict) else {}
    kpis = dict(status.get("kpis") or {})
    # crit-eval names the headline decode decode_sampled (else decode_greedy); earlier results call it decode and the
    # greedy suite's beside it greedy_decode.
    if "decode" not in kpis and ("decode_sampled" in kpis or "decode_greedy" in kpis):
        sampled = kpis.get("decode_sampled")
        kpis["decode"] = sampled or kpis.get("decode_greedy")
        kpis["greedy_decode"] = kpis.get("decode_greedy") if sampled else None
    end_to_end = kpis.get("end_to_end") or {}
    context = kpis.get("context") or {}
    rows = {"decode": (_get(kpis, "decode", "score"), _get(kpis, "decode", "ceiling"),
                       _get(kpis, "decode", "fraction_of_ceiling")),
            "prefill": (_get(kpis, "prefill", "score"), _get(kpis, "prefill", "ceiling"),
                        _get(kpis, "prefill", "fraction_of_ceiling")),
            "ttft": (_get(kpis, "ttft", "score_ms"), _get(kpis, "ttft", "floor_ms"), _get(kpis, "ttft", "fraction_of_floor"))}
    if _positive(_get(context, "decode", "score")) or _positive(_get(context, "ttft", "score_ms")):
        rows |= {"context_decode": (_get(context, "decode", "score"), _get(context, "decode", "ceiling_score"),
                                    _get(context, "decode", "fraction_of_ceiling")),
                 "context_ttft": (_get(context, "ttft", "score_ms"), _get(context, "ttft", "floor_ms"),
                                  _get(context, "ttft", "fraction_of_floor"))}
    # Each goal states its fraction of the limit and its noise; results measured before it fall back to the KPIs.
    goals = {goal.get("kpi"): goal for goal in status.get("goals") or [] if isinstance(goal, dict)}
    error = _positive(_get(kpis, "decode", "standard_error"))
    fallback = {"decode": 2 * error if error else _positive(end_to_end.get("decode_noise")),
                "prefill": _positive(_get(kpis, "prefill", "noise")),
                "ttft": _positive(end_to_end.get("ttft_noise")),
                "context_decode": _positive(_get(context, "noise", "decode")),
                "context_ttft": _positive(_get(context, "noise", "ttft"))}
    noise = {kpi: _positive(_get(goals, kpi, "noise")) or value for kpi, value in fallback.items()}
    rows = {kpi: (score, limit, _get(goals, kpi, "fraction_of_limit") or fraction)
            for kpi, (score, limit, fraction) in rows.items()}
    untiered = {"greedy_decode": {"score": _positive(_get(kpis, "greedy_decode", "score")),
                                  "noise": _positive(end_to_end.get("decode_noise"))},
                "heldout_decode": {"score": _positive(_get(kpis, "heldout", "decode_score")), "noise": None},
                "followup_ttft": {"score": _positive(_get(kpis, "followup", "ttft_ms")),
                                  "noise": _positive(end_to_end.get("ttft_noise"))}}
    gaps = [{"kpi": gap["kpi"], "name": gap["name"], "projected_gain": gap.get("projected_gain")}
            for gap in status.get("gaps") or []
            if isinstance(gap, dict) and gap.get("kpi") in KPIS and isinstance(gap.get("name"), str)]
    return {"kpis": {kpi: {"score": _positive(score), "limit": _positive(limit),
                           "fraction": fraction if type(fraction) in (int, float) and fraction > 0 else None}
                     for kpi, (score, limit, fraction) in rows.items()},
            "noise": {kpi: noise[kpi] for kpi in rows}, "untiered": untiered, "gaps": gaps}


UNITS = {"decode": "tok/s", "prefill": "tok/s", "ttft": "ms", "context_decode": "tok/s", "context_ttft": "ms"}


def standing(status, kpis=HEADLINE):
    """Lines saying how far each of `kpis` stands from its limit in an Eval result's status, and its largest gaps: the
    fraction a plan's priorities rank by, stated so a low one cannot be missed ("decode 46.6 tok/s is 11% of its limit
    419.5 tok/s; pass at 16% of its floor; largest gaps: ..."). A KPI Eval gives no limit is left out."""
    facts = extract(status)
    decode = (status or {}).get("kpis", {}).get("decode_sampled") or (status or {}).get("kpis", {}).get("decode_greedy")         or (status or {}).get("kpis", {}).get("decode") or {}
    lines = []
    for kpi in kpis:
        row = facts["kpis"].get(kpi) or {}
        if not row.get("score") or not row.get("limit") or not row.get("fraction"):
            continue
        unit = UNITS[kpi]
        line = (f"{kpi} {row['score']:,.1f} {unit} is {row['fraction']:.0%} of its "
                f"{'floor' if unit == 'ms' else 'limit'} {row['limit']:,.1f} {unit}")
        if kpi == "decode" and _positive(decode.get("pass_efficiency")):
            line += f"; pass at {decode['pass_efficiency']:.0%} of its floor"
        gaps = sorted((gap for gap in facts["gaps"] if gap["kpi"] == kpi and _positive(gap.get("projected_gain"))),
                      key=lambda gap: -gap["projected_gain"])[:2]
        if gaps:
            line += "; largest gaps: " + ", ".join(f"{gap['name']} +{gap['projected_gain']:.0%}" for gap in gaps)
        lines.append(line)
    return lines


# The KPIs whose limit crit computes when Eval states none, and an alternative strategy's ceiling may raise.
COMPUTED = ("decode", "context_decode")


def _geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _bytes_per_pass(room, model):
    """Bytes one decode pass reads: the Eval attempt's own figure (the engine's or the model files'), else the run's
    model files read as Eval reads them (crit_eval.limits.context.pass_bytes), else unknown."""
    found = _positive(room.get("bytes_per_pass"))
    if found:
        return found, "headroom.bytes_per_pass"
    try:
        from crit_eval import EVAL_DIR
        from crit_eval.limits.context import pass_bytes
        settings = json.loads((EVAL_DIR / "protocol.json").read_text(encoding="utf-8"))["context"]
        found = _positive(pass_bytes(Path(model), float(settings["kv_bytes_per_element"])))
    except Exception:  # no Eval checkout, no weight headers, or a model this host cannot read: unknown
        found = None
    return (found, "model files (crit_eval.limits.context.pass_bytes)") if found else (None, None)


def _draft_bytes(room, model):
    """Bytes one draft step reads: the Eval attempt's figure, else the checkpoint's own draft layers and the head they
    read (crit_eval.limits.context.draft_step_bytes), else unknown."""
    found = _positive(room.get("draft_bytes_per_step"))
    if found:
        return found, "headroom.draft_bytes_per_step"
    try:
        from crit_eval.limits.context import draft_step_bytes
        found = _positive(draft_step_bytes(Path(model)))
    except Exception:  # no Eval checkout, no draft layers, or a model this host cannot read: unknown
        found = None
    return (found, "model files (crit_eval.limits.context.draft_step_bytes)") if found else (None, None)


# Plan rules 2: crit's computed limits and the alternative's ceiling use Eval's cost model, the round floor with the
# draft steps at Eval's draft floor; where the draft width is unknown, the pass floor alone is an upper bound.
ROUND_FLOOR = "Eval's round floor, draft steps at Eval's draft floor"
# Plan rules 2: the drafter an alternative's ceiling assumes, the one the parent's Eval measured or a new one.
DRAFTERS = ("measured", "new")
UPPER_BOUND = "upper bound, draft cost excluded"


def _draft_floor(result, model):
    """Plan rules 2: the time one draft step costs at its bytes floor, as Eval charges it, and where it comes from: Eval's
    own draft floor (headroom.projection.draft_floor_ms), else the engine's draft_bytes_per_step over the bandwidth,
    else the checkpoint's draft layers (crit_eval.limits.context.draft_step_bytes) over it; (None, None) when none is known."""
    room = result.get("headroom") or {}
    floor = (room.get("projection") or {}).get("draft_floor_ms")
    if type(floor) in (int, float) and floor >= 0:
        return floor, "headroom.projection.draft_floor_ms"
    bandwidth = _positive(room.get("bandwidth_gbs"))
    if not bandwidth:
        return None, None
    for found, source in ((room.get("draft_bytes_per_step"), "headroom.draft_bytes_per_step"),
                          ((result.get("engine") or {}).get("draft_bytes_per_step"), "engine.draft_bytes_per_step")):
        if _positive(found):
            return found / (bandwidth * 1e9) * 1000, source
    found, source = _draft_bytes({}, model)
    return (found / (bandwidth * 1e9) * 1000, source) if found else (None, None)


def _companion(root, row, name):
    """The companion drafter whose draft cost an alternative's round floor takes where Eval measured none: the one
    drafter work/recon.json records with its pass_bytes (recon --drafter), or the one the row names by its recorded
    path (companion) when recon recorded several. None without one."""
    path = root / "work" / "recon.json"
    drafters = [d for d in ((read_json(path).get("drafters") or []) if path.is_file() else [])
                if isinstance(d, dict) and d.get("pass_bytes") and d.get("drafts_per_pass")]
    if "companion" in row:
        found = [d for d in drafters if d.get("path") == row["companion"]]
        if not found:
            raise ValueError(f"{name}.companion names no drafter work/recon.json records with its pass bytes")
        return found[0]
    return drafters[0] if len(drafters) == 1 else None


def _width(result):
    """The parent engine's draft width as Eval read it, or None (a plain engine, or a drafter that reports none)."""
    width = (result.get("headroom") or {}).get("draft_width")
    return width if type(width) is int and width >= 1 else None


def alternative_limit(result, model, kpi, tokens_per_round, draft_steps, version=1, companion=None):
    """The alternative strategy's ceiling for decode or context_decode on the parent's Eval cases (points): the
    plan's tokens per round over a round floor crit computes, the pass floor plus draft_steps draft steps at their
    bytes floor (rules 2: at Eval's draft floor, _draft_floor), geometric mean. Rules 2 without a draft floor take a
    companion drafter's (_companion: the one recon recorded): ceil(draft_steps / drafts_per_pass) drafter passes at
    its pass_bytes over the bandwidth, a block drafter's one pass yielding its whole block. None when the attempt and
    the model files lack the pass floor's facts, or the draft step's."""
    room = result.get("headroom") or {}
    bandwidth = _positive(room.get("bandwidth_gbs"))
    per_pass, source = _bytes_per_pass(room, model)
    if not bandwidth or not per_pass:
        return None
    kv = _positive(room.get("kv_bytes_per_token")) or 0
    if version >= 2:
        step_ms, draft_source = _draft_floor(result, model) if draft_steps else (None, None)
        if draft_steps and step_ms is None and companion:
            passes = math.ceil(draft_steps / companion["drafts_per_pass"])
            step_ms = passes * companion["pass_bytes"] / (bandwidth * 1e9) * 1000 / draft_steps
            draft_source = f"recon companion drafter {companion['path']} ({passes} pass x pass_bytes)"
        if draft_steps and step_ms is None:
            return None
        step_ms = step_ms or 0.0
    else:
        draft, draft_source = _draft_bytes(room, model) if draft_steps else (None, None)
        if draft_steps and not draft:
            return None
        step_ms = (draft or 0) / (bandwidth * 1e9) * 1000
    held = _held(result, kpi)
    if not held:
        return None

    def floor(tokens):
        return (per_pass + kv * tokens) / (bandwidth * 1e9) * 1000 + draft_steps * step_ms
    return {"limit": _geomean([tokens_per_round * 1000 / floor(tokens) for tokens in held]),
            "round_floor_ms": _geomean([floor(tokens) for tokens in held]),
            "inputs": {"bytes_per_pass": source, "bandwidth_gbs": "headroom.bandwidth_gbs",
                       "kv_bytes_per_token": "headroom.kv_bytes_per_token" if kv else None,
                       ("draft_floor_ms" if version >= 2 else "draft_bytes_per_step"): draft_source}}


def _held(result, kpi):
    """Tokens each measured decode case (long-context point) holds on average while it decodes."""
    from crit_eval.limits.headroom import decode_suite
    room = result.get("headroom") or {}
    if kpi == "decode":
        suite = decode_suite(result)[1]
        return [((suite.get("cases") or {}).get(cid) or {}).get("prompt_tokens", 0) +
                (((suite.get("cases") or {}).get(cid) or {}).get("output_tokens") or 0) / 2
                for cid, row in (room.get("cases") or {}).items() if _positive(row.get("decode_tps"))]
    return [(row.get("prompt_tokens") or 0) + (row.get("output_tokens") or 0) / 2
            for row in ((room.get("context") or {}).get("points") or {}).values() if _positive(row.get("decode_tps"))]


def acceptance_rates(result):
    """Per decode case, the per-token acceptance rate Eval's acceptance bound uses (crit_eval.limits.headroom.summarize): the
    first draft's conditional acceptance over the decode suite's measured rounds, the greedy suite's where it is higher
    and the decode suite is the sampled one. {} without a draft width or per-round accepted counts."""
    from crit_eval.limits.headroom import conditional_acceptance, decode_suite
    width = (result.get("headroom") or {}).get("draft_width")
    if type(width) is not int or width < 1:
        return {}
    name, suite = decode_suite(result)
    greedy_suite = (result.get("speed") or {}).get("cases") or {} if name == "sampled" else {}
    rates = {}
    for cid, case in ((suite or {}).get("cases") or {}).items():
        if case.get("status") != "measured":
            continue
        accepted = [a for s in case.get("samples", []) if not s.get("warmup") for a in (s.get("accepted") or [])]
        greedy = greedy_suite.get(cid) or {}
        greedy_accepted = [a for s in greedy.get("samples", []) if not s.get("warmup") for a in (s.get("accepted") or [])]
        if not accepted and not greedy_accepted:
            continue
        rate = conditional_acceptance(accepted, width)[0] if accepted else 0.0
        if greedy_accepted:
            rate = max(rate, conditional_acceptance(greedy_accepted, width)[0])
        rates[cid] = rate
    return rates


def _widest(result):
    """Plan rules 2: the widest draft an alternative may claim, as wide as Eval projects its acceptance bound:
    max(the parent's draft width, the protocol's acceptance_ceiling_width)."""
    from crit_eval import EVAL_DIR
    protocol = json.loads((EVAL_DIR / "protocol.json").read_text(encoding="utf-8"))
    return max(_width(result) or 0, protocol["headroom"].get("acceptance_ceiling_width", 8))


def implied_rate(tokens_per_round, draft_steps):
    """The constant per-token acceptance rate a at which draft_steps draft steps emit tokens_per_round tokens a round,
    (1 - a^(k+1)) / (1 - a) = tokens_per_round (crit_eval.limits.headroom.tokens_at_rate inverted by bisection): 0 at one
    token, 1 at draft_steps + 1."""
    from crit_eval.limits.headroom import tokens_at_rate
    if draft_steps < 1 or tokens_per_round <= 1:
        return 0.0
    if tokens_per_round >= draft_steps + 1:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (middle, high) if tokens_at_rate(middle, draft_steps)[draft_steps] < tokens_per_round else (low, middle)
    return (low + high) / 2


# Plan rules 2: tokens per round a long-context point must exceed for its engine to count as drafting where the draft
# width is unknown (crit serve counts a plain engine's n tokens over n - 1 passes, slightly above 1).
DRAFTING_TOKENS_PER_ROUND = 1.1


def measured_acceptance(result, kpi):
    """Plan rules 2: the per-token acceptance of the drafter the parent's Eval measured, for decode or context_decode.
    decode: per case the rate Eval's acceptance bound uses (acceptance_rates). context_decode: per long-context point its
    measured tokens per round at the engine's draft width inverted to a rate (implied_rate); where the width is unknown,
    a cap at the points' measured tokens per round (geometric mean). None when Eval measured no drafter."""
    if kpi == "decode":
        rates = acceptance_rates(result)
        return {"rates": rates, "source": "decode cases: the rates of Eval's acceptance bound"} if rates else None
    points = {key: row["tokens_per_round"] for key, row in
              (((result.get("headroom") or {}).get("context") or {}).get("points") or {}).items()
              if _positive(row.get("decode_tps")) and _positive(row.get("tokens_per_round"))}
    width = _width(result)
    if not points or (width is None and _geomean(list(points.values())) <= DRAFTING_TOKENS_PER_ROUND):
        return None
    if width is None:
        return {"cap": _geomean(list(points.values())),
                "source": "long-context points: their measured tokens per round (draft width unknown)"}
    return {"rates": {key: implied_rate(tokens, width) for key, tokens in points.items()},
            "source": f"long-context points: their measured tokens per round at draft width {width}, as a rate"}


def acceptance_bound(acceptance, draft_steps):
    """Plan rules 2: the most tokens per round the measured drafter reaches at draft_steps draft steps with every later
    draft accepted as often as its first, as Eval projects its acceptance bound (crit_eval.limits.headroom.tokens_at_rate):
    per case (point) (1 - a^(k+1)) / (1 - a) at its rate a, geometric mean; or the cap measured_acceptance gives."""
    from crit_eval.limits.headroom import tokens_at_rate
    if "cap" in acceptance:
        return min(acceptance["cap"], draft_steps + 1)
    return _geomean([tokens_at_rate(rate, draft_steps)[draft_steps] for rate in acceptance["rates"].values()])


def computed_limits(result, model, version=1):
    """The decode limits crit computes from an Eval attempt's facts: per measured decode case (and long-context point)
    the measured tokens per round (1 where none is measured) over its pass floor, the bytes one pass reads (the
    pass's weights and state plus the KV of the tokens held) over the device's memory bandwidth, and their geometric
    mean. Deterministic: measured acceptance, never full acceptance. In rules 1 the draft steps' own cost is left out,
    an upper bound at the measured acceptance; rules 2 charge each round the engine's draft width at Eval's draft floor
    (Eval's round floor), and keep the upper bound, labelled, where the width or the draft floor is unknown. {} when the
    attempt lacks the bandwidth or the bytes per pass."""
    from crit_eval.limits.headroom import decode_suite
    room = result.get("headroom") or {}
    bandwidth = _positive(room.get("bandwidth_gbs"))
    per_pass, source = _bytes_per_pass(room, model)
    if not bandwidth or not per_pass:
        return {}
    kv = _positive(room.get("kv_bytes_per_token")) or 0
    paths = {"bytes_per_pass": source, "bandwidth_gbs": "headroom.bandwidth_gbs",
             "kv_bytes_per_token": "headroom.kv_bytes_per_token" if kv else None}
    width, step_ms, label = _width(result), None, {}
    if version >= 2:
        step_ms, draft_source = _draft_floor(result, model) if width else (None, None)
        if width and step_ms is not None:
            label = {"bound": ROUND_FLOOR, "draft_steps": width, "draft_floor_ms": step_ms}
            paths = {**paths, "draft_width": "headroom.draft_width", "draft_floor_ms": draft_source}
        else:
            label = {"bound": UPPER_BOUND}
    draft_ms = width * step_ms if label.get("bound") == ROUND_FLOOR else 0.0
    draft_formula = " + draft_width x draft_floor_ms" if draft_ms else ""

    def rate(tokens_per_round, held):
        floor_ms = (per_pass + kv * held) / (bandwidth * 1e9) * 1000 + draft_ms
        return (_positive(tokens_per_round) or 1.0) * 1000 / floor_ms

    found = {}
    suite_name, suite = decode_suite(result)
    suite_path = "sampled_speed" if suite_name == "sampled" else "speed"
    rows = [(row, (suite.get("cases") or {}).get(cid) or {}) for cid, row in (room.get("cases") or {}).items()
            if _positive(row.get("decode_tps"))]
    if rows:
        found["decode"] = {
            "limit": _geomean([rate(row.get("tokens_per_round"),
                                    (case.get("prompt_tokens") or 0) + (case.get("output_tokens") or 0) / 2)
                               for row, case in rows]),
            "unit": "tok/s",
            "formula": "geometric mean over the decode cases of tokens_per_round x 1000 / pass_floor_ms, with "
                       "pass_floor_ms = (bytes_per_pass + kv_bytes_per_token x (prompt_tokens + output_tokens / 2)) / "
                       "(bandwidth_gbs x 1e9) x 1000" + draft_formula + "; tokens_per_round 1 where none is measured",
            **label,
            "inputs": {**paths, "tokens_per_round": "headroom.cases.<case>.tokens_per_round",
                       "prompt_tokens": f"{suite_path}.cases.<case>.prompt_tokens",
                       "output_tokens": f"{suite_path}.cases.<case>.output_tokens"}}
    points = [row for row in ((room.get("context") or {}).get("points") or {}).values()
              if _positive(row.get("decode_tps"))]
    if points:
        found["context_decode"] = {
            "limit": _geomean([rate(row.get("tokens_per_round"),
                                    (row.get("prompt_tokens") or 0) + (row.get("output_tokens") or 0) / 2)
                               for row in points]),
            "unit": "tok/s",
            "formula": "geometric mean over the long-context points of tokens_per_round x 1000 / pass_floor_ms, the "
                       "same pass floor with the point's tokens held" + draft_formula,
            **label,
            "inputs": {**paths, "tokens_per_round": "headroom.context.points.<point>.tokens_per_round",
                       "prompt_tokens": "headroom.context.points.<point>.prompt_tokens",
                       "output_tokens": "headroom.context.points.<point>.output_tokens"}}
    return found


# Plan rules 2: a family at this fraction of its limit or above is near it. Near the limit a near tie no longer puts
# both families in p1, and a family further from its limit is marked not needed only with the cause in gaps.
NEAR_LIMIT = 0.85
# Plan rules 2: the gaps entry that answers why a family's own limit is not reachable.
LIMIT_GAP = "limit"
# Plan rules 2: the relative noise of a KPI whose noise the Eval leaves unknown, when it knows none of its KPIs'.
DEFAULT_NOISE = 0.05
# Plan rules 3: a family below this fraction of its limit is marked not needed only on measured rejections: every
# gaps entry on it carries the evidence, an Eval gap answers rejected, and (prefill) every faster prompt activation
# format Eval reports is rejected by a quality measurement outside the quality noise.
MEASURED_ONLY = 0.5
# Plan rules 5: a build plan whose forecast's best case leaves a needed family below this fraction of its limit says,
# with a measurement in the run, what keeps the rest out of reach.
REACH = 0.5
# Plan rules 3: a gaps entry naming a prompt activation format Eval prices (crit-eval's ceilings by activation format).
ACTIVATION_GAP = "activation:"


def noise_default(facts, kpi=None, version=2):
    """Plan rules 2: the noise an unknown noise takes, the largest relative noise the Eval knows among its KPIs, else
    DEFAULT_NOISE, with where it comes from. Plan rules 7, given the KPI: the largest known noise of its own family
    (time to first token is a family of its own), else DEFAULT_NOISE: a first token's spread says nothing about decode's
    noise, and borrowing it once set a decode lever's bar at 151%."""
    known = {name: noise for name, noise in ((facts or {}).get("noise") or {}).items() if noise is not None}
    if version >= 7 and kpi is not None:
        peers = FAMILIES.get(FAMILY.get(kpi), (kpi,))
        known = {name: noise for name, noise in known.items() if name in peers}
    if not known:
        return DEFAULT_NOISE, "default"
    kpi = max(known, key=known.get)
    return known[kpi], kpi


def noises(facts, version):
    """Each KPI's relative noise as the plan rules use it: as frozen in rules 1 (unknown stays None), with unknown
    noise at noise_default in rules 2 (its family's in rules 7)."""
    frozen = (facts or {}).get("noise") or {}
    if version < 2:
        return dict(frozen)
    return {kpi: frozen.get(kpi) if frozen.get(kpi) is not None else noise_default(facts, kpi, version)[0]
            for kpi in KPIS}


def family(facts, name, version=1):
    """A family's fraction of its limit (the lower of its members') and its noise: in rules 1 the noise of the member
    that sets the fraction, in rules 2 the largest noise of its members with a fraction (unknown at its default)."""
    members = [(facts["kpis"][kpi]["fraction"], kpi) for kpi in FAMILIES[name]
               if kpi in facts["kpis"] and facts["kpis"][kpi]["fraction"] is not None]
    if not members:
        return None, None
    fraction, kpi = min(members)
    if version < 2:
        return fraction, facts["noise"].get(kpi)
    noise = noises(facts, version)
    return fraction, max(noise[member] for _, member in members)


def priorities(needed, facts, version=1):
    """The plan's priorities from its needed flags and the frozen fractions, or None with the reason. Decode and
    prefill are p1 and p2 by which is further from its limit, both p1 when the distance between their fractions is
    within the noise: in rules 1 the larger of (relative noise × fraction) of the two families, in rules 2 the two
    combined in quadrature, except (rules 2) when both are at NEAR_LIMIT or above, where the lower is p1 alone. Time
    to first token is never p1."""
    decode, prefill = needed["decode"]["needed"], needed["prefill"]["needed"]
    if not decode and not prefill:
        return None, "neither decode nor prefill is needed: the plan is a stop plan"
    if facts is None:
        return None, "the run has no Eval yet: priorities follow from the first Eval's fractions of the limits"
    if decode and prefill:
        (f_decode, n_decode), (f_prefill, n_prefill) = family(facts, "decode", version), family(facts, "prefill", version)
        missing = [name for name, fraction in (("decode", f_decode), ("prefill", f_prefill)) if fraction is None]
        if missing:
            return None, "the parent's Eval reports no fraction of the limit for " + " and ".join(missing)
        if version >= 2:
            band = math.hypot(n_decode * f_decode, n_prefill * f_prefill)
        else:
            band = max((n_decode or 0) * f_decode, (n_prefill or 0) * f_prefill)
        near = version >= 2 and f_decode >= NEAR_LIMIT and f_prefill >= NEAR_LIMIT
        if abs(f_decode - f_prefill) <= band and not near:
            return {"p1": ["decode", "prefill"], "p2": ["ttft"], "p3": []}, None
        # Apart, or both near their limits (rules 2): the family further from its limit alone, decode on an exact tie.
        first, second = ("prefill", "decode") if f_prefill < f_decode else ("decode", "prefill")
        return {"p1": [first], "p2": [second], "p3": ["ttft"]}, None
    return {"p1": ["decode" if decode else "prefill"], "p2": ["ttft"], "p3": []}, None


def completed_evals(root, candidate_id):
    """The candidate's retained Eval results whose attempt completed, oldest first."""
    folder = safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}")
    found = []
    for path in sorted(folder.glob("eval/attempt-*.json")):
        attempt = ident(path.stem, "attempt")
        result = folder / "attempts" / attempt / "result.json"
        if result.is_file() and read_json(result).get("outcome") == "completed":
            found.append(path)
    return found


def newest_eval(root):
    """The newest completed Eval attempt of the run's newest candidate that has one, or None."""
    for folder in sorted(safe_path(root, "candidates").glob("candidate-*"), reverse=True):
        found = completed_evals(root, folder.name)
        if found:
            return found[-1]
    return None


def read_status(path):
    return read_json(path, max_bytes=EVAL_JSON_BYTES).get("status") or {}


def plan_paths(root):
    return sorted(safe_path(root, "plans").glob("plan-*/plan.json"))


def load(root, run, plan_id):
    path = safe_path(root, f"plans/{ident(plan_id, 'plan')}/plan.json")
    if not path.is_file():
        raise ValueError(f"Unknown plan {plan_id}: record it with crit run ... plan --record plan.json")
    value = read_json(path)
    header(value, "plan", run["run_id"], set(FIELDS) | CRIT_FIELDS, OPTIONAL)
    if value["plan_id"] != plan_id or value["request_sha256"] != run["request_sha256"]:
        raise ValueError(f"{plan_id} does not belong to this run's request")
    return value


def links(root):
    """plan_id -> the candidate created from it."""
    linked = {}
    for path in sorted(safe_path(root, "candidates").glob("candidate-*/candidate.json")):
        plan_id = read_json(path).get("plan_id")
        if plan_id is not None:
            linked[plan_id] = path.parent.name
    return linked


def summaries(root, run):
    """The run's plans for state.json: superseded plans stay listed with what replaced them."""
    plans = [load(root, run, path.parent.name) for path in plan_paths(root)]
    replaced = {plan["supersedes"]: plan["plan_id"] for plan in plans if plan.get("supersedes")}
    linked = links(root)
    return [{"plan_id": plan["plan_id"], "segment_id": plan["segment_id"], "parent_candidate_id": plan["parent_candidate_id"],
             "decision": plan["decision"], "supersedes": plan.get("supersedes"), "superseded_by": replaced.get(plan["plan_id"]),
             "candidate_id": linked.get(plan["plan_id"])} for plan in plans]


FIELDS = {"parent_candidate_id", "decision", "needed", "gaps", "strategy", "levers", "rejected", "forecast", "gates"}
CRIT_FIELDS = {"plan_id", "segment_id", "created_at", "request_sha256", "inputs", "priorities", "priorities_reason"}


OPTIONAL = {"supersedes", "estimated_limits", "forecast_justification", "weakest", "review", "prompt_activations",
            "round"}

# Plan rules 4: what each candidate of a run is for, by its number in the run (objective()). The proof of concept
# works end to end; the crit rounds raise it with crit's toolkit; each critique round names the parent's two weakest
# terms and improves them, with other levers beside them.
OBJECTIVES = {
    "proof_of_concept": "the first engine that works end to end: it serves, passes the functional gates and the "
                        "quality gate at the required context, built in its placement scenario's starting design "
                        "(recon.md's placement plan)",
    "crit_round": "improve the parent as much as possible with crit's toolkit: Inference techniques and Kernels Hub "
                  "kernels, each measured against its floor and fixed or replaced where it trails",
    "critique_round": "name the parent's two weakest terms (the largest measured distances to their floors, from Eval "
                      "and the kernel trace) in weakest, improve both, and add other levers beside them",
}
CRIT_ROUNDS = 3        # candidates 2 and 3
# On the candidate ladder (crit.run.ladder) candidate 1 builds and serves with no Eval, candidate 2 is the plugin
# round, candidates 3 to LADDER_CRIT_ROUNDS are crit rounds and critique rounds start after the ladder's full Eval.
LADDER_OBJECTIVES = {
    "proof_of_concept": "the first engine that builds, passes its plugins' tests (crit plugins test) and crit run ... "
                        "serve --check, built in its placement scenario's starting design (recon.md's placement plan)",
    "plugin_round": "the parent with every plugin crit's library and crit-hub offer that fits the checkpoint and the "
                    "GPU (crit plugins list --model MODEL --device cuda) in the slots it fills, each plugin's tests "
                    "passing against its reference (crit plugins test); crit run ... ablate says which plugin pays, "
                    "slows or breaks the engine",
}
LADDER_CRIT_ROUNDS = ladder.RUNGS   # candidates 3 to 5
# Plan rules 4: what supplies a speed lever's code (`provider`): an Inference entry, a Kernel Hub repository, a library
# (kernels/SOURCES.md), or the engine's own code, which carries a `yardstick`: the measurement of the best existing
# provider for the operation, or the providers it checked and why none fits.
PROVIDERS = ("inference:", "hub:", "lib:", "own")
# Candidates 2 to FLOOR_UNTIL forecast at least MIN_GAIN of expected gain on a p1 KPI over the parent's Eval.
MIN_GAIN, FLOOR_UNTIL = 0.10, 5
# A requirement lever that `serves` "gate:NAME" repairs that gate of the parent's Eval (state "fail"): the plan is a
# gate repair, which a candidate needs whatever it forecasts, so the floor on p1 and a critique round's weakest terms
# do not apply to it.
GATE_REPAIR = "gate:"
# A requirement lever that `serves` "check:NAME" repairs a check crit run requires (arithmetic, functional,
# resources) that the parent failed: its newest check record of that gate for some case, as crit run ... check
# --from-eval or serve --check recorded it, has status "failed". Such a check keeps the parent from being eligible
# even when every gate of its Eval passed (resources also judges the device memory left free at the required
# context), and the plan is exempt as a gate repair is.
CHECK_REPAIR = "check:"


def _missed(kpi, row):
    """An outcome row whose measurement fell short of its forecast's low (above its high for a time)."""
    measured = row.get("measured")
    if measured is None:
        return False
    return measured > row["forecast"]["high"] if kpi in MS else measured < row["forecast"]["low"]


def _largest(plan, p1):
    """The id of the plan's speed lever with the largest forecast change on a p1 family, or None."""
    best, found = 0.0, None
    for lever in speed_levers(plan["levers"]):
        change = max((row["change"] for kpi, row in lever["changes"].items() if FAMILY.get(kpi, kpi) in p1), default=0.0)
        if change > best:
            best, found = change, lever["id"]
    return found


def review_facts(root, run, parent, eval_ref):
    """Plan rules 4, check 14: what the plan for the parent's child reviews: the parent's plan, its outcome at the Eval
    attempt this plan freezes, the KPIs that missed their forecast and the levers built, pruned or deferred. None when
    the parent has no plan or no outcome at that attempt."""
    planned = [s for s in summaries(root, run) if s["candidate_id"] == parent]
    if not planned or eval_ref is None:
        return None
    plan = load(root, run, planned[0]["plan_id"])
    attempt = eval_ref["path"].rsplit("/", 1)[-1][:-len(".json")]
    path = root / "plans" / plan["plan_id"] / "outcomes" / f"{attempt}.json"
    if not path.is_file():
        return None
    outcome_value = read_json(path)
    fates = {row["id"]: (row.get("fate") or {}).get("fate") for row in outcome_value["levers"]}
    p1 = (plan.get("priorities") or {}).get("p1") or []
    largest = _largest(plan, p1)
    return {"plan_id": plan["plan_id"], "outcome": {"path": path.relative_to(root).as_posix(), "sha256": file_sha(path)},
            "missed": sorted(kpi for kpi, row in outcome_value["kpis"].items() if _missed(kpi, row)),
            "built": sorted(i for i, fate in fates.items() if fate == "built"),
            "dropped": sorted(i for i, fate in fates.items() if fate in ("pruned", "deferred")),
            "largest_p1_lever": largest, "largest_p1_fate": fates.get(largest) if largest else None}


def objective(number, rungs=False, last=False):
    """Plan rules 4: the objective of the run's candidate `number` (1 for the first). rungs: the run counts its budget
    on the candidate ladder, where the objective also names the rung's Eval (`eval`, crit.run.ladder.rung), the full
    one when `last` (the plan's candidate takes its segment's last budgeted candidate)."""
    if rungs:
        name = ("proof_of_concept" if number == 1 else "plugin_round" if number == 2
                else "crit_round" if number <= LADDER_CRIT_ROUNDS else "critique_round")
        return {"candidate": number, "name": name, "rule": LADDER_OBJECTIVES.get(name, OBJECTIVES.get(name)),
                **({"min_gain": MIN_GAIN} if 2 <= number <= FLOOR_UNTIL else {}), "eval": ladder.rung(number, last)}
    name = ("proof_of_concept" if number == 1 else "crit_round" if number <= CRIT_ROUNDS else "critique_round")
    return {"candidate": number, "name": name, "rule": OBJECTIVES[name],
            **({"min_gain": MIN_GAIN} if 2 <= number <= FLOOR_UNTIL else {})}


def _fields(data):
    """Check 1: fields, types and text limits; unknown keys are refused."""
    keys(data, FIELDS, OPTIONAL)
    if "prompt_activations" in data:
        from crit_core.serving import prompt_activations
        if not prompt_activations(data["prompt_activations"]):
            raise ValueError("prompt_activations is what the planned engine declares to Eval: one of bf16, fp16_acc, fp8 "
                             "or stored for every quantized format, or per stored format (fp8, nvfp4) one of them or "
                             "{activation: share}")
    if "forecast_justification" in data:
        text(data["forecast_justification"], "forecast_justification", 8000)
    if "review" in data:
        review = data["review"]
        keys(review, {"plan_id", "kpis", "levers", "lesson"})
        ident(review["plan_id"], "plan")
        text(review["lesson"], "review lesson (what this plan changes because of the review)", 8000)
        if not isinstance(review["kpis"], dict) or not set(review["kpis"]) <= set(KPIS):
            raise ValueError("review.kpis maps the KPIs that missed their forecast to {cause, evidence}")
        for kpi, row in review["kpis"].items():
            keys(row, {"cause", "evidence"})
            text(row["cause"], f"review {kpi} cause (why it missed, from the profile)", 8000)
            text(row["evidence"], f"review {kpi} evidence (a path in the run)", 400)
        if not isinstance(review["levers"], dict):
            raise ValueError("review.levers maps the parent plan's levers to what they delivered")
        for lever_id, row in review["levers"].items():
            keys(row, {"result"}, {"evidence", "structure"})
            text(row["result"], f"review {lever_id} result (what it delivered, measured)", 8000)
            for key in ("evidence", "structure"):
                if key in row:
                    text(row[key], f"review {lever_id} {key}", 400 if key == "evidence" else 4000)
    if "weakest" in data:
        if not isinstance(data["weakest"], list) or len(data["weakest"]) != 2:
            raise ValueError("weakest names the parent's two weakest terms")
        for row in data["weakest"]:
            keys(row, {"kpi", "term", "evidence", "levers"})
            if row["kpi"] not in KPIS:
                raise ValueError("weakest entries name a KPI (" + ", ".join(KPIS) + ")")
            text(row["term"], "weakest term (what is slow, with its measured time and floor)", 400)
            text(row["evidence"], f"weakest evidence for {row['term']} (a path in the run)", 400)
            if not isinstance(row["levers"], list) or not row["levers"] or not all(isinstance(i, str) for i in row["levers"]):
                raise ValueError(f"weakest {row['term']}: levers lists the ids of the levers that improve it")
    if data["parent_candidate_id"] is not None:
        ident(data["parent_candidate_id"], "candidate")
    if data.get("supersedes") is not None:
        ident(data["supersedes"], "plan")
    if data["decision"] not in DECISIONS:
        raise ValueError("decision is build or stop")
    keys(data["needed"], set(FAMILIES))
    for name in FAMILIES:
        keys(data["needed"][name], {"needed", "evidence"})
        if type(data["needed"][name]["needed"]) is not bool:
            raise ValueError(f"needed.{name}.needed is true or false")
        text(data["needed"][name]["evidence"], f"needed.{name}.evidence", 8000)
    for name in ("gaps", "levers", "rejected"):
        if not isinstance(data[name], list) or len(data[name]) > MAX_ROWS:
            raise ValueError(f"{name} is a list of up to {MAX_ROWS} entries")
    for row in data["gaps"]:
        keys(row, {"kpi", "gap", "answer", "detail"}, {"evidence", "cause_terms"})
        if "evidence" in row:
            text(row["evidence"], f"gaps evidence for {row['gap']} (a path in the run)", 400)
        if "cause_terms" in row and (not isinstance(row["cause_terms"], list) or not row["cause_terms"]
                                     or not all(isinstance(term, str) for term in row["cause_terms"])):
            raise ValueError(f"gaps {row['gap']} cause_terms lists the terms of the plan's round account that bound it")
        if row["kpi"] not in KPIS or row["answer"] not in ("cause", "rejected"):
            raise ValueError("gaps entries name a KPI (" + ", ".join(KPIS) + ") and answer cause or rejected")
        text(row["gap"], "gaps.gap (the Eval gap's name)", 400)
        text(row["detail"], f"gaps detail for {row['gap']}", 8000)
    strategy = data["strategy"]
    keys(strategy, {"current", "alternative", "choice", "reason"})
    for side in ("current", "alternative"):  # check 10: both present and nonempty
        keys(strategy[side], {"name", "ceiling", "evidence"}, {"limits"} if side == "alternative" else ())
        text(strategy[side]["name"], f"strategy.{side}.name", 400)
        text(strategy[side]["ceiling"], f"strategy.{side}.ceiling (a number with its unit and where it comes from)", 400)
        text(strategy[side]["evidence"], f"strategy.{side}.evidence", 8000)
    if strategy["choice"] not in ("current", "alternative"):
        raise ValueError("strategy.choice is current or alternative")
    limits = strategy["alternative"].get("limits", {})
    if not isinstance(limits, dict) or not set(limits) <= set(COMPUTED):
        raise ValueError("strategy.alternative.limits maps decode and context_decode to the alternative's ceiling: "
                         "{tokens_per_round, draft_steps, derivation}, or {limit, derivation} where crit cannot "
                         "compute the round floor")
    for kpi, row in limits.items():
        keys(row, {"derivation"}, {"tokens_per_round", "draft_steps", "limit", "drafter", "acceptance_evidence",
                                   "companion"})
        if "companion" in row:
            text(row["companion"], f"strategy.alternative.limits.{kpi}.companion (a drafter path recon recorded)", 1000)
        text(row["derivation"], f"strategy.alternative.limits.{kpi}.derivation", 8000)
        if row.get("drafter", "measured") not in DRAFTERS:
            raise ValueError(f"strategy.alternative.limits.{kpi}.drafter is measured (the drafter the parent's Eval "
                             "measured) or new")
        if ("limit" in row) == ("tokens_per_round" in row):
            raise ValueError(f"strategy.alternative.limits.{kpi} gives tokens_per_round (and draft_steps), or a limit "
                             "where crit cannot compute the round floor, not both")
        if "tokens_per_round" in row:
            number(row["tokens_per_round"], f"strategy.alternative.limits.{kpi}.tokens_per_round", 1, 64)
            number(row.get("draft_steps", 0), f"strategy.alternative.limits.{kpi}.draft_steps", 0, 64, integer=True)
        else:
            number(row["limit"], f"strategy.alternative.limits.{kpi}.limit (tok/s)", 0.000001, 1e9)
    text(strategy["reason"], "strategy.reason", 8000)
    seen = set()
    for lever in data["levers"]:
        keys(lever, {"id", "technique", "description", "changes", "group", "confidence"},
             {"kind", "serves", "provider", "yardstick"})
        text(lever["id"], "lever id", 100)
        if lever.get("kind", "speed") not in LEVER_KINDS:
            raise ValueError(f"{lever['id']} kind is speed or requirement")
        if lever.get("kind") == "requirement":
            text(lever.get("serves"), f"{lever['id']} serves (the rule or gate the requirement lever is for)", 400)
        elif "serves" in lever:
            raise ValueError(f"{lever['id']} names what it serves, which only a requirement lever does: set kind requirement")
        if lever["id"] in seen:
            raise ValueError(f"Lever id {lever['id']} appears twice")
        seen.add(lever["id"])
        text(lever["technique"], f"{lever['id']} technique", 400)
        if "provider" in lever:
            provider = text(lever["provider"], f"{lever['id']} provider", 200)
            if not provider.startswith(PROVIDERS):
                raise ValueError(f"{lever['id']} provider starts with " + ", ".join(PROVIDERS) + ": what supplies its code")
        if "yardstick" in lever:
            keys(lever["yardstick"], {"evidence", "detail"})
            text(lever["yardstick"]["evidence"], f"{lever['id']} yardstick evidence (a path in the run)", 400)
            text(lever["yardstick"]["detail"], f"{lever['id']} yardstick detail", 4000)
        text(lever["description"], f"{lever['id']} description", 8000)
        if lever["group"] is not None:
            text(lever["group"], f"{lever['id']} group", 100)
        if lever["confidence"] not in CONFIDENCE:
            raise ValueError(f"{lever['id']} confidence is low, medium or high")
        if not isinstance(lever["changes"], dict) or not set(lever["changes"]) <= set(KPIS):
            raise ValueError(f"{lever['id']} changes map KPIs (" + ", ".join(KPIS) + ") to {change, source}")
        for kpi, change in lever["changes"].items():
            keys(change, {"change", "source"})
            number(change["change"], f"{lever['id']} {kpi} change (relative, in the KPI's improving direction)", -100, 1000)
            source = text(change["source"], f"{lever['id']} {kpi} source", 500)
            if source != "estimate" and not source.startswith("eval-gap:"):
                raise ValueError(f"{lever['id']} {kpi} source is eval-gap:<gap name> or estimate")
    for row in data["rejected"]:
        keys(row, {"name", "reason"}, {"bound_by", "evidence"})
        text(row["name"], "rejected name", 400)
        text(row["reason"], f"rejected {row['name']} reason", 8000)
        if "bound_by" in row:
            keys(row["bound_by"], {"family", "term"})
            if row["bound_by"]["family"] not in FAMILIES:
                raise ValueError(f"rejected {row['name']} bound_by names a family (decode or prefill) and a term of its "
                                 "round account")
            text(row["bound_by"]["term"], f"rejected {row['name']} bound_by term", 200)
        if "evidence" in row:
            text(row["evidence"], f"rejected {row['name']} evidence (a path in the run)", 400)
    if "round" in data:
        if not isinstance(data["round"], dict) or not data["round"] or not set(data["round"]) <= set(FAMILIES):
            raise ValueError("round maps decode and prefill to the parent's round account: {total_ms, idle_ms, terms, "
                             "evidence}")
        for name, account in data["round"].items():
            keys(account, {"total_ms", "idle_ms", "terms", "evidence"})
            number(account["total_ms"], f"round.{name}.total_ms (one measured round or chunk)", 0.000001, 1e9)
            number(account["idle_ms"], f"round.{name}.idle_ms (the device idle within it)", 0, 1e9)
            text(account["evidence"], f"round.{name}.evidence (a path in the run)", 400)
            if not isinstance(account["terms"], list) or not account["terms"] or len(account["terms"]) > MAX_ROWS:
                raise ValueError(f"round.{name}.terms lists the round's parts: {{term, ms, floor_ms}}")
            names = set()
            for term in account["terms"]:
                keys(term, {"term", "ms", "floor_ms"})
                label = text(term["term"], f"round.{name} term", 200)
                if label in names or label == IDLE:
                    raise ValueError(f"round.{name} names {label!r} twice" if label in names else
                                     f"round.{name}: {IDLE!r} is the account's idle_ms, not a term")
                names.add(label)
                number(term["ms"], f"round.{name}.{label}.ms", 0, 1e9)
                number(term["floor_ms"], f"round.{name}.{label}.floor_ms (its bytes or operations at the device's "
                                         "rate)", 0, 1e9)
    if data["forecast"] is not None:
        if not isinstance(data["forecast"], dict) or not set(data["forecast"]) <= set(KPIS):
            raise ValueError("forecast maps KPIs (" + ", ".join(KPIS) + ") to {low, expected, high}")
        for kpi, row in data["forecast"].items():
            keys(row, {"low", "expected", "high"})
            for key in ("low", "expected", "high"):
                number(row[key], f"forecast {kpi} {key}", 0.000001, 1e12)
            if not row["low"] <= row["expected"] <= row["high"]:
                raise ValueError(f"forecast {kpi} requires low ≤ expected ≤ high")
    text(data["gates"], "gates (how quality stays inside the budget and the required context is met)", 8000)
    estimates = data.get("estimated_limits", {})
    if not isinstance(estimates, dict) or not set(estimates) <= set(KPIS):
        raise ValueError("estimated_limits maps KPIs (" + ", ".join(KPIS) + ") to {limit, derivation}")
    for kpi, row in estimates.items():
        keys(row, {"limit", "derivation"})
        number(row["limit"], f"estimated_limits.{kpi}.limit ({'ms' if kpi in MS else 'tok/s'})", 0.000001, 1e12)
        text(row["derivation"], f"estimated_limits.{kpi}.derivation (how the limit was derived)", 8000)


def _alternative(run, root, facts, result, data, sources):
    """The limits the priorities rank decode and context_decode by: the best strategy's ceiling, the higher of the
    current strategy's limit (Eval's, crit's computed, the plan's estimate) and the alternative strategy's ceiling.
    crit computes the alternative's round floor wherever the attempt or the model files hold its facts, so only the
    tokens per round are the plan's; elsewhere the plan gives the ceiling. Used whatever the plan chooses."""
    limits = data["strategy"]["alternative"].get("limits", {})
    if run["spec"].get("plan_limits") != "computed":
        if limits:
            raise ValueError("strategy.alternative.limits rank a run that computes plan limits; this run ranks by the "
                             "current strategy's limits only")
        return None
    found = {}
    for kpi, row in limits.items():
        current = (facts or {}).get("kpis", {}).get(kpi) if facts else None
        if not current or current["score"] is None:
            raise ValueError(f"strategy.alternative.limits names {kpi}, which the parent's Eval did not measure")
        model = run["spec"]["model_path"]
        steps = row.get("draft_steps", 0)
        if rules(run) >= 2 and "tokens_per_round" in row and row["tokens_per_round"] > steps + 1:
            raise ValueError(f"strategy.alternative.limits.{kpi}.tokens_per_round {row['tokens_per_round']:g} is above "
                             f"draft_steps + 1 = {steps + 1}: a round of {steps} draft steps emits at most {steps + 1} "
                             "tokens (the drafts and the target's own token)")
        drafted = _drafter(run, root, data["parent_candidate_id"], result, kpi, row, steps)
        # Rules 2 charge the draft steps at Eval's draft floor, as computed_limits does: one cost model, Eval's.
        computed = (alternative_limit(result, model, kpi, row["tokens_per_round"], steps, rules(run),
                                      _companion(root, row, name=f"strategy.alternative.limits.{kpi}"))
                    if "tokens_per_round" in row else None)
        if "tokens_per_round" in row and computed is None:
            if rules(run) >= 2:  # a bare limit is refused there too (_drafter): say the one way that is left
                raise ValueError(f"crit cannot compute the round floor for strategy.alternative.limits.{kpi} (no bytes "
                                 "per pass, bandwidth or draft step cost: a separate drafter no Eval has measured is "
                                 "priced from recon --drafter's pass bytes): record the drafter with recon, or in plan "
                                 f"rules 2 leave {kpi} out of strategy.alternative.limits, so the alternative does not "
                                 "rank it; its strategy, forecast and evidence still stand")
            raise ValueError(f"crit cannot compute the round floor for strategy.alternative.limits.{kpi} (no bytes per "
                             "pass, bandwidth or draft step bytes): give its limit and how it was derived instead")
        if "limit" in row and alternative_limit(result, model, kpi, 1.0, 0) is not None:
            raise ValueError(f"crit computes the round floor for {kpi} from the parent's Eval: give "
                             f"strategy.alternative.limits.{kpi}.tokens_per_round (and draft_steps), not a limit")
        limit = computed["limit"] if computed else row["limit"]
        entry = {"limit": limit, "unit": "tok/s", "derivation": row["derivation"],
                 "computed_by": "crit" if computed else "plan",
                 **({"tokens_per_round": row["tokens_per_round"], "draft_steps": row.get("draft_steps", 0),
                     "round_floor_ms": computed["round_floor_ms"], "inputs": computed["inputs"],
                     "formula": "geometric mean over the parent's decode cases (long-context points) of tokens_per_round "
                                "x 1000 / round_floor_ms, round_floor_ms = the pass floor with the tokens held plus "
                                "draft_steps x " + ("draft_floor_ms" if rules(run) >= 2 else "draft_bytes_per_step / bandwidth"),
                     **({"bound": ROUND_FLOOR} if rules(run) >= 2 else {})} if computed else {}), **drafted}
        current_limit = (current["score"] / current["fraction"]) if current["fraction"] else None
        entry |= {"current_limit": current_limit, "current_source": sources.get(kpi),
                  "won": current_limit is None or limit > current_limit}
        if entry["won"]:
            current["fraction"] = current["score"] / limit
            sources[kpi] = "alternative"
        found[kpi] = entry
    return found


def _drafter(run, root, parent, result, kpi, row, steps):
    """Plan rules 2: the drafter an alternative's ceiling assumes and the bound on its claim. Every claim gives its
    tokens per round (never a bare limit) and stays within Eval's widest projected draft (_widest) and draft_steps + 1
    tokens a round. The drafter the parent's Eval measured (measured) and a new one (new) both stay within the measured
    drafter's acceptance bound at the claimed steps (acceptance_bound): a new drafter is assumed to accept like the
    measured one, unless the plan cites another candidate's completed Eval attempt retained in the run
    (acceptance_evidence), whose measured acceptance bounds it instead. Without a measured drafter the physical bound
    applies, noted. Returns the entry's drafter fields with the claim's implied acceptance rate; refuses a claim above
    its bound."""
    name = f"strategy.alternative.limits.{kpi}"
    if rules(run) < 2:
        for key in ("drafter", "acceptance_evidence"):
            if key in row:
                raise ValueError(f"{name}.{key} is a field of plan rules 2; this run checks plans with the first rules")
        return {}
    if "drafter" not in row:
        raise ValueError(f"{name}.drafter names the drafter the ceiling assumes: measured (the drafter the parent's Eval "
                         "measured) or new")
    drafter, evidence = row["drafter"], row.get("acceptance_evidence")
    if evidence is not None and drafter != "new":
        raise ValueError(f"{name}.acceptance_evidence is for a new drafter: the measured drafter's acceptance is its Eval's")
    if "tokens_per_round" not in row:
        raise ValueError(f"{name} gives a bare limit: in plan rules 2 an alternative's ceiling is its tokens_per_round "
                         "and draft_steps over a round floor crit computes, so it can be bounded; where crit cannot "
                         "compute the floor (no bandwidth or bytes per pass), the alternative does not rank the families")
    measured = None
    if evidence is not None:
        measured = _evidence(root, parent, kpi, text(evidence, f"{name}.acceptance_evidence", 400), name)
    widest = max(_widest(result or {}), _widest(measured[0]) if measured else 0)
    if steps > widest:
        raise ValueError(f"{name}.draft_steps {steps} is wider than Eval projects a drafter ({widest} draft steps: the "
                         "parent's draft width or the protocol's acceptance_ceiling_width): claim at most that")
    fields = {"drafter": drafter, "implied_acceptance": implied_rate(row["tokens_per_round"], steps)}
    if measured:
        fields |= {"acceptance_evidence": measured[1]}
    acceptance = measured[2] if measured else measured_acceptance(result or {}, kpi)
    if acceptance is None:
        return fields | {"acceptance_bound": None,
                         "note": "the parent's Eval measured no drafter acceptance: the physical bound, draft_steps + 1"}
    bound = acceptance_bound(acceptance, steps)
    if row["tokens_per_round"] > bound + 1e-9:
        raise ValueError(f"{name}.tokens_per_round {row['tokens_per_round']:g} is above {bound:.3f}, the tokens per round "
                         f"the measured drafter reaches at {steps} draft steps with every later draft accepted as often "
                         "as its first ((1 - a^(k+1)) / (1 - a) at its measured rate a per case, the projection of "
                         "Eval's acceptance bound; geometric mean). Claim at most that"
                         + (", or cite another candidate's Eval attempt that measured it (acceptance_evidence)"
                            if drafter == "new" and not measured else ""))
    return fields | {"acceptance_bound": {"tokens_per_round": bound, "draft_steps": steps, "widest": widest,
                                          "source": acceptance["source"],
                                          **({"rates": acceptance["rates"]} if "rates" in acceptance
                                             else {"cap": acceptance["cap"]})}}


EVIDENCE = re.compile(r"candidates/(candidate-[0-9]{4,8})/eval/(attempt-[0-9]{4,8})\.json")


def _evidence(root, parent, kpi, path, name):
    """Plan rules 2: a new drafter's acceptance measurement, another candidate's completed Eval attempt retained in the
    run (candidates/<candidate>/eval/<attempt>.json, not the parent's): (its result, {path, sha256}, its measured
    acceptance for the KPI). Refused when it is anything else or measured no drafter acceptance."""
    match = EVIDENCE.fullmatch(path)
    if not match or match[1] == parent:
        raise ValueError(f"{name}.acceptance_evidence names another candidate's Eval attempt retained in this run, "
                         "candidates/<candidate>/eval/<attempt>.json, not the parent's: the measurement of the new drafter")
    found = safe_path(root, path)
    attempt = safe_path(root, f"candidates/{match[1]}/attempts/{match[2]}/result.json")
    if not found.is_file() or not attempt.is_file() or read_json(attempt).get("outcome") != "completed":
        raise ValueError(f"{name}.acceptance_evidence {path} is not a completed Eval attempt of this run")
    result = read_json(found, max_bytes=EVAL_JSON_BYTES)
    acceptance = measured_acceptance(result, kpi)
    if acceptance is None:
        raise ValueError(f"{name}.acceptance_evidence {path} measured no drafter acceptance for {kpi} (no draft width "
                         "and per-round accepted counts)")
    return result, {"path": path, "sha256": file_sha(found), "source": acceptance["source"]}, acceptance


def _limits(facts, data, compared, computed):
    """Fill the fractions Eval leaves without a limit: from the limit crit computed from the attempt's facts, else
    from the plan's estimated limits. Every member of a family the priorities compare needs one of them; an estimate
    never replaces a limit Eval gives, and is physically consistent with the frozen score (a ceiling at or above a
    rate, a floor at or below a time). An estimate for a KPI crit computed is recorded but not used. Returns the
    source of each fraction and the estimates not used."""
    estimates = data.get("estimated_limits", {})
    if facts is None:
        if estimates:
            raise ValueError("estimated_limits fill limits a parent's Eval lacks; this plan freezes no Eval")
        return {}, {}
    sources = {kpi: "eval" for kpi, row in facts["kpis"].items() if row["fraction"] is not None}
    for kpi, entry in (computed or {}).items():
        row = facts["kpis"][kpi]
        entry["used"] = row["score"] is not None and entry["limit"] >= row["score"]
        if entry["used"]:
            row["fraction"] = row["score"] / entry["limit"]
            sources[kpi] = "computed"
        elif row["score"] is not None:
            entry["note"] = "below the measured score: the limit model misses a term, so it is not used"
    for kpi in estimates:
        row = facts["kpis"].get(kpi)
        if not row or row["score"] is None:
            raise ValueError(f"estimated_limits names {kpi}, which the parent's Eval did not measure")
        if sources.get(kpi) == "eval":
            raise ValueError(f"The parent's Eval gives {kpi} a limit ({row['limit']}); Eval's limit stands, so drop "
                             f"estimated_limits.{kpi}")
    for kpi in compared:
        row = facts["kpis"].get(kpi)
        if row and row["score"] is not None and kpi not in sources and kpi not in estimates:
            unit = "ms" if kpi in MS else "tok/s"
            raise ValueError(f"The parent's Eval gives {kpi} no limit, and the priorities compare decode with prefill: "
                             f"state estimated_limits.{kpi} with the limit in {unit} and its derivation (for example "
                             "a drafter's expected tokens per round over the pass floor)")
    unused = {}
    for kpi, estimate in estimates.items():
        row, limit = facts["kpis"][kpi], estimate["limit"]
        if limit < row["score"] if kpi not in MS else limit > row["score"]:
            raise ValueError(f"estimated_limits.{kpi}.limit {limit:g} is {'below' if kpi not in MS else 'above'} the "
                             f"measured score {row['score']:g}: a {'floor is at most' if kpi in MS else 'ceiling is at least'} "
                             "what the engine already reaches")
        if sources.get(kpi) == "computed":
            unused[kpi] = {"estimate": limit, "computed": computed[kpi]["limit"],
                           "difference": limit / computed[kpi]["limit"] - 1}
            continue
        row["fraction"] = limit / row["score"] if kpi in MS else row["score"] / limit
        sources[kpi] = "estimate"
    return sources, unused


def speed_levers(levers):
    """The levers planned for KPI changes: every lever but the requirement levers."""
    return [lever for lever in levers if lever.get("kind", "speed") == "speed"]


def _above(lever, noise):
    """Whether a lever has a change above its KPI's noise; a positive change on a KPI of unknown noise counts (in rules
    2 no noise is unknown: `noise` comes from noises())."""
    return any(row["change"] > (noise.get(kpi) or 0) if noise.get(kpi) is not None else row["change"] > 0
               for kpi, row in lever["changes"].items())


def build(root, run, bounds, data, status_for=None):
    """Validate a plan payload against the run's facts and return the record crit stores (without plan_id).
    status_for: the Eval status to judge the parent's result by, by its path in the run (a replay's statuses rendered
    again, replay.status_for); by default the status stored in the result. The plan passes CHECKS, those of the run's
    plan rules and the ones before them."""
    _fields(data)
    version = rules(run)
    parent = data["parent_candidate_id"]
    first = parent is None
    candidates = sorted(path.name for path in safe_path(root, "candidates").glob("candidate-*"))
    rungs = ladder.ladder(run)
    # The run's candidates by number: on the candidate ladder those on it, with budget_counts evaluated those that went
    # through Eval (a draft's fix plans the same candidate again).
    counted = ladder.counted(root) if rungs else None
    numbered = (sorted(counted) if rungs else sorted(evaluated_candidates(root)) if counts_evaluated(run)
                else candidates)
    # Check 2: the parent exists and the plan freezes its newest Eval attempt.
    if first and numbered:
        raise ValueError("parent_candidate_id names the candidate whose Eval the plan starts from; null only while the "
                         "run has no candidate" + (" on the candidate ladder" if rungs else " that went through Eval"
                                                   if counts_evaluated(run) else ""))
    eval_ref, facts, computed, result = None, None, None, None
    if not first:
        if parent not in candidates:
            raise ValueError(f"parent_candidate_id {parent} is not a candidate of this run")
        found = completed_evals(root, parent)
        if rungs:
            # On the ladder a parent stands on its rung before a plan starts from it: through its Eval, or below the
            # full Eval sealed built or completed with a passed serve check, when it skipped its Eval.
            if parent not in counted:
                raise ValueError(f"{parent} is not on the candidate ladder yet: run its Eval (crit run ... eval), or, "
                                 "below the full Eval, seal it built (crit run ... seal --lifecycle built) once its "
                                 "plugins' tests and crit run ... serve --check pass, before planning from it")
            if not found and (parent in evaluated_candidates(root) or ladder.scope_of(root, parent) == "full"):
                raise ValueError(f"{parent} has no completed Eval attempt: run crit run ... eval --candidate {parent} "
                                 "(--resume goes on from a stopped one) before planning from it")
        elif not found and newest_eval(root) is not None:
            raise ValueError(f"{parent} has no completed Eval attempt: run crit run ... eval --candidate {parent} before "
                             "planning from it")
        # Rules 7: the newest attempt that measured decode and prefill, so a partial rerun (quality alone) leaves the
        # attempt the plan's gaps, forecast and review refer to in place.
        for path in reversed(found):
            eval_ref = {"path": path.relative_to(root).as_posix(), "sha256": file_sha(path)}
            result = read_json(path, max_bytes=EVAL_JSON_BYTES)
            status = (status_for or {}).get(eval_ref["path"])
            facts = extract(status if status is not None else result.get("status") or {})
            if version < 7 or path == found[0] or all(
                    (facts["kpis"].get(kpi) or {}).get("score") is not None for kpi in ("decode", "prefill")):
                break
        if found:
            if run["spec"].get("plan_limits") == "computed":
                # The limits Eval leaves out that crit can compute from the attempt (and the run's model files).
                computed = {kpi: entry for kpi, entry in computed_limits(result, run["spec"]["model_path"], version).items()
                            if kpi in facts["kpis"] and facts["kpis"][kpi]["fraction"] is None}
    needed = {name: data["needed"][name]["needed"] for name in FAMILIES}
    # A limit Eval leaves out is the plan's estimate when the priorities compare the two families.
    compared = [kpi for name in FAMILIES for kpi in FAMILIES[name]] if all(needed.values()) else []
    sources, unused = _limits(facts, data, compared, computed)
    estimated = [kpi for kpi in KPIS if sources.get(kpi) == "estimate"]
    alternative = _alternative(run, root, facts, result, data, sources)
    recon_path = root / "work" / "recon.json"
    inputs = {"eval": eval_ref, "recon_sha256": file_sha(recon_path) if recon_path.is_file() else None,
              **(facts or {"kpis": None, "noise": None, "untiered": None, "gaps": None}), "estimated": estimated,
              **({"computed_limits": computed, "fraction_sources": sources, "estimates_not_used": unused,
                  "alternative_limits": alternative} if run["spec"].get("plan_limits") == "computed" else {}),
              **({"noise_default": dict(zip(("noise", "source"), noise_default(facts)))}
                 if 2 <= version < 7 and facts is not None else {}),
              # Rules 7: each unknown noise's default, from its own family.
              **({"noise_default": {kpi: dict(zip(("noise", "source"), noise_default(facts, kpi, version)))
                                    for kpi in KPIS if (facts.get("noise") or {}).get(kpi) is None}}
                 if version >= 7 and facts is not None else {})}
    if version >= 4 or rungs:
        # The run's candidates so far, this plan's candidate the next one (a revision plans the same one again). On the
        # candidate ladder the segment's last budgeted candidate gets the full Eval.
        budget = bounds["limits"].get("candidate_budget")
        last = (rungs and budget is not None
                and sum(found == bounds["segment_id"] for found in counted.values()) + 1 >= budget)
        inputs["objective"] = objective(len(numbered) + 1, rungs, last)
    plan = SimpleNamespace(
        root=root, run=run, bounds=bounds, data=data, version=version, parent=parent, first=first, eval_ref=eval_ref,
        facts=facts, computed=computed, result=result, needed=needed, compared=compared, sources=sources,
        estimated=estimated, inputs=inputs, decision=data["decision"], forecast=data["forecast"], order=None, reason=None,
        # The noise checks 4, 6 and 9 use: in rules 2 an unknown noise takes its default.
        noise=noises(facts, version),
        # Requirement levers are exempt from checks 5 (a numeric change) and 6, and never count toward 7 or 8.
        levers=speed_levers(data["levers"]))
    # A gate repair (rules 4): requirement levers serving gate:NAME for gates the parent's Eval failed, or check:NAME
    # for checks crit run requires that the parent failed. Either exempts the plan from the speed checks.
    plan.repairs = _gate_repairs(plan) if version >= 4 and data["decision"] == "build" else []
    plan.check_repairs = _check_repairs(plan) if version >= 4 and data["decision"] == "build" else {}
    plan.repairing = bool(plan.repairs or plan.check_repairs)
    for minimum, check in CHECKS:
        if version >= minimum:
            check(plan)
    return {**data, "supersedes": data.get("supersedes"), "segment_id": bounds["segment_id"], "created_at": now(),
            "request_sha256": run["request_sha256"], "inputs": inputs, "priorities": plan.order,
            "priorities_reason": plan.reason}


def _check_decision(plan):
    """Check 3: the decision follows the needed flags (in rules 2 the run's first plan too)."""
    if not plan.first or plan.version >= 2:
        if plan.decision == "stop" and any(plan.needed.values()):
            raise ValueError("A stop plan requires both decode and prefill not needed: "
                             + ", ".join(name for name, flag in plan.needed.items() if flag) + " is marked needed")
        if plan.decision == "build" and not any(plan.needed.values()):
            raise ValueError("neither decode nor prefill is needed: record a stop plan (decision stop)")


def _check_gap_names(plan):
    """Every gaps entry and every lever change sourced from an Eval gap names a gap the parent's Eval reports, beside
    the entries rules 2 (limit) and 3 (activation:FORMAT) add."""
    facts, version = plan.facts, plan.version
    gap_names = {(gap["kpi"], gap["name"]) for gap in (facts or {}).get("gaps") or []}
    for row in plan.data["gaps"]:
        if version >= 3 and row["gap"].startswith(ACTIVATION_GAP):
            if row["kpi"] != "prefill" or row["answer"] != "rejected" or "evidence" not in row:
                raise ValueError(f"gaps entry {row['gap']!r} names a prompt activation format: kpi prefill, answer "
                                 "rejected, with the evidence of its quality measurement")
            continue
        if version >= 7 and row["gap"] == CPU_TIER_GAP:
            if row["kpi"] != "decode" or row["answer"] != "rejected" or "evidence" not in row:
                raise ValueError(f"gaps entry {CPU_TIER_GAP!r} rejects the CPU expert tier: kpi decode, answer rejected, "
                                 "with the evidence of its measurement")
            continue
        if version >= 2 and row["gap"] == LIMIT_GAP:
            if row["kpi"] not in FAMILIES or row["answer"] != "cause":
                raise ValueError(f"gaps entry {LIMIT_GAP!r} names a family (decode or prefill) and answers cause: why "
                                 "the rest of that family's limit is not reachable")
            continue
        if facts is not None and (row["kpi"], row["gap"]) not in gap_names:
            raise ValueError(f"gaps names {row['kpi']} gap {row['gap']!r}, which the parent's Eval does not report")
    for lever in plan.data["levers"]:
        for kpi, change in lever["changes"].items():
            if change["source"].startswith("eval-gap:") and (facts is None or (kpi, change["source"][9:]) not in gap_names):
                raise ValueError(f"{lever['id']} {kpi} source {change['source']} names no {kpi} gap of the parent's Eval")


def _check_answered_gaps(plan):
    """Check 4: every gap above the noise on a KPI marked not needed is answered."""
    facts = plan.facts
    if plan.first or facts is None:
        return
    answered = {(row["kpi"], row["gap"]) for row in plan.data["gaps"]}
    for name, flag in plan.needed.items():
        if flag:
            continue
        threshold = family(facts, name)[1]
        for gap in facts["gaps"]:
            gain = gap["projected_gain"]
            # Rules 1 compare with the noise of the member that sets the family, rules 2 with the gap's own KPI.
            limit = plan.noise.get(gap["kpi"]) if plan.version >= 2 else threshold
            if FAMILY.get(gap["kpi"]) == name and type(gain) in (int, float) and gain > (limit or 0) \
                    and (gap["kpi"], gap["name"]) not in answered:
                raise ValueError(f"{name} is marked not needed, but the {gap['kpi']} gap {gap['name']!r} "
                                 f"(+{gain:.1%}) is above the noise: answer it in gaps with its physical cause or "
                                 "its measured rejection")


def _check_limit_gaps(plan):
    """Check 4, rules 2: a family marked not needed below NEAR_LIMIT answers why its own limit is not reachable."""
    if plan.facts is None:
        return
    limits = {row["kpi"] for row in plan.data["gaps"] if row["gap"] == LIMIT_GAP}
    for name, flag in plan.needed.items():
        fraction = family(plan.facts, name, plan.version)[0]
        if not flag and fraction is not None and fraction < NEAR_LIMIT and name not in limits:
            raise ValueError(f"{name} is marked not needed at {fraction:.0%} of its limit, below {NEAR_LIMIT:.0%}: "
                             f"add a gaps entry {{\"kpi\": \"{name}\", \"gap\": \"{LIMIT_GAP}\", \"answer\": "
                             "\"cause\", \"detail\": ...} with the physical cause that keeps the rest of the limit "
                             "out of reach")


def _check_measured_only(plan):
    """Rules 3: a family far below its limit is not needed only on measured rejections (_measured_only)."""
    if plan.facts is not None:
        plan.inputs["gap_evidence"] = _measured_only(plan.root, plan.data, plan.facts, plan.result, plan.needed,
                                                     plan.version)


def _prioritize(plan):
    """The priorities the plan's needed flags give, with the estimates and limit sources the ranking rests on."""
    plan.order, plan.reason = priorities(plan.data["needed"], plan.facts, plan.version)
    if plan.order is not None:
        # The ranking rests on these estimates where it compares the families.
        plan.order["estimated"] = [kpi for kpi in plan.estimated if kpi in plan.compared]
        if plan.run["spec"].get("plan_limits") == "computed":
            # Where each compared fraction comes from: Eval's limit, crit's computed limit or the plan's estimate.
            plan.order["sources"] = {kpi: plan.sources[kpi] for kpi in plan.compared if kpi in plan.sources}


def _check_levers(plan):
    """Checks 5 to 8 of a build plan: its levers name KPI changes above the noise (or pay in a group), one improves
    more than time to first token, and every needed p1 family has one."""
    if plan.decision != "build":
        return
    levers, noise, first, order = plan.levers, plan.noise, plan.first, plan.order
    for lever in levers:
        if not lever["changes"]:
            raise ValueError(f"{lever['id']} names no KPI change")
    if plan.repairing:
        return   # a gate repair is built for the gate whatever its speed levers change (checks 5 to 8)
    # Check 5: levers, each naming a KPI change.
    if not levers and not first:
        raise ValueError("A build plan names its levers: each with the KPIs it changes (requirement levers alone "
                         "do not make a candidate)")
    for lever in levers:
        if not lever["changes"]:
            raise ValueError(f"{lever['id']} names no KPI change")
    # Check 6: a lever below the noise joins a group with one above it.
    if not first:
        paying = {lever["group"] for lever in levers if lever["group"] is not None and _above(lever, noise)}
        for lever in levers:
            if not _above(lever, noise) and (lever["group"] is None or lever["group"] not in paying
                                             or sum(other["group"] == lever["group"] for other in levers) < 2):
                raise ValueError(f"{lever['id']} changes no KPI by more than its noise: put it in a group with the "
                                 "lever it pays together with, or leave it out")
    # Check 8: no plan targets time to first token alone.
    if levers and not any(row["change"] > 0 for lever in levers for kpi, row in lever["changes"].items() if kpi != "ttft"):
        raise ValueError("No lever improves decode, prefill, context_decode or context_ttft: time to first token is "
                         "never the reason for a candidate on its own")
    # Check 7: every needed p1 KPI has a lever on its family.
    if not first and order is not None:
        for name in order["p1"]:
            if plan.needed.get(name) and not any(row["change"] > 0 for lever in levers
                                                 for kpi, row in lever["changes"].items() if FAMILY.get(kpi) == name):
                raise ValueError(f"{name} is p1 but no lever improves it ({', '.join(FAMILIES[name])})")


def _check_forecast(plan):
    """Check 9: the forecast covers every KPI the parent's Eval measured and improves decode or prefill."""
    forecast, facts = plan.forecast, plan.facts
    measured = tuple(HEADLINE + (CONTEXT if facts is not None and "context_decode" in facts["kpis"] else ()))
    if forecast is None:
        if plan.decision == "build" and not plan.first:
            raise ValueError("A build plan forecasts " + ", ".join(measured) + ": {low, expected, high} each")
        return
    missing = [kpi for kpi in measured if kpi not in forecast]
    if missing:
        raise ValueError("forecast lacks " + ", ".join(missing))
    extra = [kpi for kpi in forecast if kpi not in measured]
    if extra:
        raise ValueError("forecast names " + ", ".join(extra) + ", which the parent's Eval did not measure")
    if plan.decision == "build" and not plan.first and facts is not None and not plan.repairing:
        scored = [kpi for kpi in ("decode", "prefill") if facts["kpis"][kpi]["score"]]
        if scored and not any(forecast[kpi]["expected"] / facts["kpis"][kpi]["score"] - 1 > (plan.noise.get(kpi) or 0)
                              for kpi in scored):
            raise ValueError("The forecast improves neither decode nor prefill beyond its noise over the parent's "
                             "Eval: a candidate is planned for at least one of them")


def _check_forecast_priorities(plan):
    """Check 12 (rules 3): a forecast that puts its gain on p2 or lets a p1 KPI fall says why."""
    if plan.decision == "build" and plan.forecast is not None and plan.facts is not None and plan.order is not None:
        plan.inputs["forecast_checks"] = _forecast_checks(plan.order, plan.forecast, plan.facts)
        if plan.inputs["forecast_checks"] and "forecast_justification" not in plan.data:
            raise ValueError("The forecast " + "; ".join(plan.inputs["forecast_checks"]) + ": give forecast_justification "
                             "(why this candidate is worth building anyway), which crit records, or plan for p1")


def _check_reach(plan):
    """Check 15 (rules 5): the strategy's ceiling is the limit, not the engine's own measurement."""
    # A gate repair is built for its gate, not for reach: the forecast may stay where the parent is.
    if plan.decision == "build" and plan.forecast is not None and plan.facts is not None and not plan.repairing:
        plan.inputs["reach"] = _reach(plan.root, plan.data, plan.facts, plan.computed, plan.forecast, plan.needed)


def _failed_gates(result):
    """The gates the frozen Eval attempt's status reports failed ("fail"), by name."""
    gates = ((result or {}).get("status") or {}).get("gates") or []
    return {g.get("name") for g in gates if isinstance(g, dict) and g.get("state") == "fail"}


def _gate_repairs(plan):
    """The parent's failed gates the plan's requirement levers repair (serves "gate:NAME"); a named gate the parent's
    Eval did not fail is refused."""
    named = {lever["serves"][len(GATE_REPAIR):] for lever in plan.data["levers"]
             if lever.get("kind") == "requirement" and str(lever.get("serves", "")).startswith(GATE_REPAIR)}
    if not named:
        return []
    failed = _failed_gates(plan.result)
    unfailed = sorted(named - failed)
    if unfailed:
        checks = [name for name in unfailed if name in plan.run["spec"]["correctness"]["required_gates"]]
        raise ValueError("A gate repair names gates the parent's Eval did not fail: " + ", ".join(unfailed) +
                         " (failed: " + (", ".join(sorted(failed)) or "none") + ")" +
                         (f". {', '.join(checks)} {'is a check' if len(checks) == 1 else 'are checks'} of crit run, "
                          "not a gate of Eval's status: a lever repairing a failed one serves "
                          + ", ".join(CHECK_REPAIR + name for name in checks) if checks else ""))
    return sorted(named)


def _newest_checks(root, candidate_id, gate):
    """The candidate's newest check record of `gate` per case (crit run ... check --from-eval, serve --check, check
    --record), {case_id: (path, record)}, the records in the order crit numbered them."""
    newest = {}
    for path in sorted(safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}/checks").glob("check-*.json")):
        value = read_json(path)
        if value.get("gate") == gate:
            newest[value.get("case_id")] = (path, value)
    return newest


def _check_repairs(plan):
    """The checks crit run requires that the plan's requirement levers repair (serves "check:NAME"): {NAME: the
    parent's newest NAME check of each case that failed, {path, sha256, check_id, case_id, recorded_by}}. A named
    check that is not a required one, that the parent has no record of, or whose newest records all did not fail is
    refused."""
    named = {lever["serves"][len(CHECK_REPAIR):] for lever in plan.data["levers"]
             if lever.get("kind") == "requirement" and str(lever.get("serves", "")).startswith(CHECK_REPAIR)}
    if not named:
        return {}
    required = plan.run["spec"]["correctness"]["required_gates"]
    unknown = sorted(named - set(required))
    if unknown:
        raise ValueError("A check repair names checks crit run does not require: " + ", ".join(unknown) +
                         " (required: " + ", ".join(required) + "); a gate of the parent's Eval is repaired as gate:NAME")
    if plan.first:
        raise ValueError("A check repair names a failed check of the parent: the run's first plan has none")
    found, unfailed, unrecorded = {}, [], False
    for name in sorted(named):
        newest = _newest_checks(plan.root, plan.parent, name)
        failed = [(case, path, value) for case, (path, value) in sorted(newest.items()) if value.get("status") == "failed"]
        if not failed:
            unrecorded |= not newest
            unfailed.append(f"{name} (" + ("; ".join(f"its newest, {path.stem} for {case}, {value.get('status')}"
                                                   for case, (path, value) in sorted(newest.items()))
                                           or f"no {name} check recorded") + ")")
            continue
        found[name] = [{"path": path.relative_to(plan.root).as_posix(), "sha256": file_sha(path),
                        "check_id": path.stem, "case_id": case,
                        "recorded_by": (value.get("extensions") or {}).get("recorded_by")} for case, path, value in failed]
    if unfailed:
        raise ValueError(f"A check repair names checks {plan.parent} did not fail: " + "; ".join(unfailed) +
                         f". {CHECK_REPAIR}NAME repairs a check whose newest record of {plan.parent} failed"
                         + (f"; record its checks first (crit run ... check --candidate {plan.parent} --from-eval)"
                            if unrecorded else ""))
    return found


def _check_objective(plan):
    """Check 13 (rules 4): the objective's requirements."""
    if plan.decision != "build":
        return
    root, data, levers, forecast, facts, order = plan.root, plan.data, plan.levers, plan.forecast, plan.facts, plan.order
    repairs = plan.repairing
    if plan.repairs:
        plan.inputs["objective"]["repairs"] = plan.repairs
    if plan.check_repairs:
        plan.inputs["objective"]["check_repairs"] = plan.check_repairs
    # Every speed lever names what supplies its code; the engine's own code is measured against what exists.
    for lever in levers:
        if "provider" not in lever:
            raise ValueError(f"{lever['id']} names no provider: " + ", ".join(PROVIDERS) + " (kernels/SOURCES.md "
                             "lists where kernels come from)")
        if lever["provider"].startswith("own"):
            if "yardstick" not in lever:
                raise ValueError(f"{lever['id']} is the engine's own code: give its yardstick, the measured best "
                                 "existing provider for the operation or the providers checked and why none fits")
            _evidence_file(root, lever["yardstick"]["evidence"], f"{lever['id']} yardstick")
    goal = plan.inputs["objective"]
    if goal["name"] == "critique_round" and not repairs:
        if "weakest" not in data:
            raise ValueError(f"Candidate {goal['candidate']} is a critique round: {goal['rule']}")
        ids = {lever["id"]: lever for lever in levers}
        for row in data["weakest"]:
            _evidence_file(root, row["evidence"], f"weakest {row['term']}")
            for lever_id in row["levers"]:
                lever = ids.get(lever_id)
                if lever is None or not any(change["change"] > 0 for kpi, change in lever["changes"].items()
                                            if FAMILY.get(kpi, kpi) == FAMILY.get(row["kpi"], row["kpi"])):
                    raise ValueError(f"weakest {row['term']}: {lever_id} is no speed lever of this plan that "
                                     f"improves {row['kpi']}")
        if not set(ids) - {i for row in data["weakest"] for i in row["levers"]}:
            raise ValueError(f"Candidate {goal['candidate']} is a critique round: add other levers beside the ones "
                             "for its two weakest terms")
    if "min_gain" in goal and not repairs and forecast is not None and facts is not None and order is not None:
        best = max((_gain(kpi, forecast[kpi]["expected"], (facts["kpis"].get(kpi) or {}).get("score"))
                    for name in order["p1"] for kpi in FAMILIES.get(name, (name,)) if kpi in forecast), default=0.0)
        if best < goal["min_gain"]:
            raise ValueError(f"Candidate {goal['candidate']} forecasts {best:+.1%} on p1 ({', '.join(order['p1'])}): "
                             f"candidates 2 to {FLOOR_UNTIL} are planned for at least +{goal['min_gain']:.0%} on a "
                             "p1 KPI; plan the levers that close more of its gap")


def _check_rules_4_fields(plan):
    """weakest (critique rounds) and review belong to plan rules 4."""
    if plan.version < 4:
        if "weakest" in plan.data:
            raise ValueError("weakest belongs to plan rules 4 (critique rounds)")
        if "review" in plan.data:
            raise ValueError("review belongs to plan rules 4, once the parent's plan has an outcome")


def _check_review(plan):
    """Check 14 (rules 4): the plan reviews its parent's plan against the parent's measured Eval."""
    root, data = plan.root, plan.data
    facts_review = review_facts(root, plan.run, plan.parent, plan.eval_ref) if not plan.first else None
    if facts_review is None:
        if "review" in data:
            raise ValueError("review belongs to plan rules 4, once the parent's plan has an outcome")
        return
    plan.inputs["review"] = facts_review
    review = data.get("review")
    if review is None:
        raise ValueError(f"Review {facts_review['plan_id']} first: review gives its plan_id, the cause and evidence of "
                         "each KPI that missed its forecast (" + (", ".join(facts_review["missed"]) or "none") + "), "
                         "what each lever delivered (" + ", ".join(facts_review["built"] + facts_review["dropped"])
                         + ") and the lesson this plan draws")
    if review["plan_id"] != facts_review["plan_id"]:
        raise ValueError(f"review names {review['plan_id']}: the parent's plan is {facts_review['plan_id']}")
    for kpi in facts_review["missed"]:
        if kpi not in review["kpis"]:
            raise ValueError(f"review: {kpi} missed its forecast in {facts_review['outcome']['path']}: give its cause "
                             "and evidence")
    for kpi, row in review["kpis"].items():
        _evidence_file(root, row["evidence"], f"review {kpi}")
    for lever_id in facts_review["built"] + facts_review["dropped"]:
        if lever_id not in review["levers"]:
            raise ValueError(f"review: say what {lever_id} of {facts_review['plan_id']} delivered")
    # Gate: the parent's largest p1 lever, pruned or deferred, is planned again or rejected on a measurement of
    # another kernel structure or a vendor library at the same arithmetic, never on variants of one kernel.
    largest = facts_review["largest_p1_lever"]
    if largest and facts_review["largest_p1_fate"] in ("pruned", "deferred") and \
            largest not in {lever["id"] for lever in data["levers"]}:
        row = review["levers"][largest]
        if "evidence" not in row or "structure" not in row:
            raise ValueError(f"{largest} was {facts_review['largest_p1_fate']}, the largest p1 lever of "
                             f"{facts_review['plan_id']}: plan it again (same id) or give review.levers.{largest} "
                             "evidence and structure: the other kernel structure or vendor library measured on it")
        _evidence_file(root, row["evidence"], f"review {largest}")


def _check_supersedes(plan):
    """Check 11: a revision replaces the newest plan of its chain before its candidate exists."""
    if plan.data.get("supersedes") is None:
        return
    old = load(plan.root, plan.run, plan.data["supersedes"])
    summary = next(s for s in summaries(plan.root, plan.run) if s["plan_id"] == old["plan_id"])
    if old["segment_id"] != plan.bounds["segment_id"] or old["parent_candidate_id"] != plan.parent:
        raise ValueError(f"supersedes names {old['plan_id']}, a plan of another segment or parent")
    if summary["candidate_id"] is not None:
        raise ValueError(f"{old['plan_id']} is linked to {summary['candidate_id']}: change course with a new candidate "
                         "and a new plan")
    if summary["superseded_by"] is not None:
        raise ValueError(f"{old['plan_id']} is already superseded by {summary['superseded_by']}")


# Plan rules 6: in a speed run a prompt GEMM over quantized weights starts at a quantized
# activation format.
EXACT_ACTIVATION = "bf16"
STORED_GAP = ACTIVATION_GAP + "stored"


def _bf16_formats(declared):
    """The stored formats whose prompt activations `declared` (prompt_activations) puts at BF16, in part or whole;
    "*" for a declaration that names no format."""
    if isinstance(declared, str):
        return ["*"] if declared == EXACT_ACTIVATION else []
    found = []
    for name, mix in (declared or {}).items():
        if mix == EXACT_ACTIVATION or isinstance(mix, dict) and (mix.get(EXACT_ACTIVATION) or 0) > 0:
            found.append(name)
    return found


def _check_activations(plan):
    """Check 16, rules 6, speed runs: a build plan declares the prompt activations its candidate takes, and
    no quantized format's prompt GEMMs start at BF16. speed judges against the same exact reference as
    preserve, with a budget that admits the stored format's activations without evidence per group (the mode trades
    drift for speed), so a candidate reads prompts at the stored format (or fp8 / fp16_acc) from the first one on;
    BF16 stands only beside a gaps entry activation:stored whose evidence is a crit-eval result that measured the
    stored format outside the quality noise (_quality_rejection). The parent must have served what its plan
    declared."""
    if plan.decision != "build" or quality_mode(plan.run["spec"]["correctness"]) != "speed":
        return
    declared = plan.data.get("prompt_activations")
    if declared is None:
        raise ValueError("A speed run's build plan declares prompt_activations, the formats its candidate's "
                         "prompt GEMMs take per stored format (as the engine declares them to Eval), for example "
                         "{\"nvfp4\": \"stored\", \"fp8\": \"fp8\"}")
    rejected = next((row for row in plan.data["gaps"] if row["gap"] == STORED_GAP), None)
    exact = _bf16_formats(declared)
    if exact:
        label = f"prompt_activations puts {', '.join(exact)} at bf16"
        if rejected is None:
            raise ValueError(f"{label}: in speed, a quantized format's prompt GEMMs start at "
                             "its stored format's activations (stored), or fp8 / fp16_acc, from candidate 1 on. Declare "
                             "that, or measure the stored format first (crit eval run --package P --only quality "
                             "--attribute-drift) and add a gaps entry {\"kpi\": \"prefill\", \"gap\": "
                             f"\"{STORED_GAP}\", \"answer\": \"rejected\", \"evidence\": <that result>, \"detail\": ...}}")
        plan.inputs["activation_evidence"] = {
            **_evidence_file(plan.root, rejected["evidence"], f"gaps entry {STORED_GAP}"),
            "over_budget_sets": _quality_rejection(plan.root, rejected["evidence"], "stored", f"gaps entry {STORED_GAP}")}
    if plan.first or plan.result is None:
        return
    parent_plan = next((plan_id for plan_id, candidate in links(plan.root).items() if candidate == plan.parent), None)
    planned = load(plan.root, plan.run, parent_plan).get("prompt_activations") if parent_plan else None
    served = (plan.result.get("engine") or {}).get("prompt_activations")
    if planned is not None and served != planned and rejected is None:
        raise ValueError(f"{plan.parent} served prompt activations {served!r}, its plan {parent_plan} declared "
                         f"{planned!r}: build what the plan declares, or record the measurement that rejected it "
                         f"(gaps entry {STORED_GAP})")


# Plan rules 7: the round account. A term of at least ROUND_SHARE of the round that runs slower than FLOOR_RATE of its
# floor is not at its floor, nor is idle time of that share; terms and idle add up to the round within ACCOUNT_SLACK.
# A rejection bound by a term reopens once that term falls by STALE.
IDLE = "idle"
ROUND_SHARE, FLOOR_RATE, ACCOUNT_SLACK, STALE = 0.10, 0.5, 0.10, 0.25


def _account_terms(account):
    return {term["term"]: term for term in account["terms"]}


def _check_round(plan):
    """Check 17 (rules 7): a family's limit claim rests on a round account. A gaps entry `limit` names the terms of the
    parent's round (plan field `round`, per family) that bound it in `cause_terms`; every other term large enough to
    matter runs at least at half its floor's rate, the device idles less than that share, and the parts add up to the
    measured round. A stop record once called a decode round "at its bandwidth floor but for the expert copies" while
    its dense matrices ran at 27% of their floor and the device idled 8 of 60 ms."""
    data = plan.data
    if plan.version < 7:
        if "round" in data or any("cause_terms" in row for row in data["gaps"]) or \
                any("bound_by" in row or "evidence" in row for row in data["rejected"]):
            raise ValueError("round, gaps cause_terms and rejected bound_by / evidence belong to plan rules 7")
        return
    rounds = data.get("round") or {}
    recorded = {}
    for name, account in rounds.items():
        evidence = _evidence_file(plan.root, account["evidence"], f"round.{name}")
        parts = sum(term["ms"] for term in account["terms"]) + account["idle_ms"]
        if abs(parts - account["total_ms"]) > ACCOUNT_SLACK * account["total_ms"]:
            raise ValueError(f"round.{name}: its terms and idle time add up to {parts:g} ms, the measured round is "
                             f"{account['total_ms']:g} ms: account for the whole round (within {ACCOUNT_SLACK:.0%}), "
                             "the device's idle time included")
        recorded[name] = {"evidence": evidence, "parts_ms": parts}
    for row in data["gaps"]:
        if row["gap"] != LIMIT_GAP:
            if "cause_terms" in row:
                raise ValueError(f"gaps {row['gap']}: cause_terms belong to a {LIMIT_GAP!r} entry")
            continue
        name = row["kpi"]
        account = rounds.get(name)
        if account is None or "cause_terms" not in row:
            raise ValueError(f"gaps entry {LIMIT_GAP!r} on {name} claims the rest of its limit is out of reach: give the "
                             f"parent's round account (round.{name}: total_ms, idle_ms, terms [{{term, ms, floor_ms}}], "
                             "evidence) and name the terms that bind it in cause_terms")
        terms = _account_terms(account)
        unknown = [term for term in row["cause_terms"] if term not in terms and term != IDLE]
        if unknown:
            raise ValueError(f"gaps {LIMIT_GAP} on {name}: cause_terms {', '.join(unknown)} are no terms of round.{name}")
        share = ROUND_SHARE * account["total_ms"]
        for label, term in terms.items():
            if label in row["cause_terms"] or term["ms"] < share:
                continue
            rate = term["floor_ms"] / term["ms"] if term["ms"] else 1.0
            if rate < FLOOR_RATE:
                raise ValueError(f"round.{name}: {label} takes {term['ms']:g} of {account['total_ms']:g} ms at "
                                 f"{rate:.0%} of its floor ({term['floor_ms']:g} ms): it is not at its floor, so the "
                                 f"limit is not out of reach yet. Plan a lever for it, or name it in cause_terms with "
                                 "the measured reason it cannot come closer")
        if account["idle_ms"] >= share and IDLE not in row["cause_terms"]:
            raise ValueError(f"round.{name}: the device idles {account['idle_ms']:g} of {account['total_ms']:g} ms: "
                             "plan a lever that fills it (host work beside the device, captured launches), or name "
                             f"{IDLE!r} in cause_terms with the measured reason")
    for row in data["rejected"]:
        if "bound_by" in row:
            account = rounds.get(row["bound_by"]["family"])
            if account is None or row["bound_by"]["term"] not in _account_terms(account):
                raise ValueError(f"rejected {row['name']} is bound by {row['bound_by']['term']!r}: name a term of "
                                 f"round.{row['bound_by']['family']}")
            if "evidence" not in row:
                raise ValueError(f"rejected {row['name']} names what bound it: give the evidence of its measurement")
        if "evidence" in row:
            _evidence_file(plan.root, row["evidence"], f"rejected {row['name']}")
    if recorded:
        plan.inputs["round"] = recorded


def _check_stale_rejections(plan):
    """Check 18 (rules 7): a rejection measured while a term bound the round holds only while that term does. Once an
    earlier plan's rejection is bound by a term this plan's round account shows STALE smaller (or gone), the lever is
    planned again (a lever whose id is its name) or rejected again here on new evidence. "CPU expert compute: slower"
    measured while link copies bound the round says nothing once they no longer do."""
    rounds = plan.data.get("round") or {}
    if not rounds:
        return
    current = {row["name"]: row for row in plan.data["rejected"]}
    planned = {lever["id"] for lever in plan.data["levers"]}
    reopened = []
    for summary in summaries(plan.root, plan.run):
        earlier = load(plan.root, plan.run, summary["plan_id"])
        for row in earlier.get("rejected") or []:
            bound = row.get("bound_by")
            account = rounds.get((bound or {}).get("family"))
            then = ((earlier.get("round") or {}).get((bound or {}).get("family")) or {})
            if not bound or account is None or not then:
                continue
            before = _account_terms(then).get(bound["term"])
            now_term = _account_terms(account).get(bound["term"])
            if before is None or (now_term is not None and now_term["ms"] > (1 - STALE) * before["ms"]):
                continue
            name = row["name"]
            reopened.append({"plan_id": earlier["plan_id"], "name": name, "term": bound["term"],
                             "then_ms": before["ms"], "now_ms": now_term["ms"] if now_term else 0.0})
            again = current.get(name)
            if name in planned:
                continue
            if again is None or again.get("evidence") in (None, row.get("evidence")):
                shown = f"{now_term['ms']:g} ms" if now_term else "gone"
                raise ValueError(f"{earlier['plan_id']} rejected {name!r} while {bound['term']} took "
                                 f"{before['ms']:g} ms of the {bound['family']} round; it is now {shown}. Plan it again "
                                 f"(a lever with id {name!r}) or reject it again here on a new measurement (rejected "
                                 "with bound_by and evidence)")
    if reopened:
        plan.inputs["reopened"] = reopened


# Plan rules 7: where recon's placement prices misses computed on the CPU (alone or split with the link) below misses
# copied over the link, candidate 1 starts with the CPU expert tier, or the plan names the measurement that rejects it.
CPU_TIER = "inference:host-expert-tier"
CPU_TIER_GAP = "placement:cpu"


def _cpu_wins(root):
    """The placement's plain-pass floors when recon.json prices the CPU (alone or split) below the link, else None."""
    from .recon import RECON_JSON
    facts = safe_path(root, RECON_JSON)
    if not facts.is_file():
        return None
    placement = (read_json(facts).get("gathered") or {}).get("placement") or {}
    floors = placement.get("plain_pass_floor_ms") or {}
    cpu = min((floors[k] for k in ("cpu", "split") if floors.get(k)), default=None)
    if placement.get("scenario") not in ("sparse_device_host", "sparse_device_host_disk") or not cpu or             not floors.get("link") or cpu >= floors["link"]:
        return None
    return {key: floors[key] for key in ("link", "cpu", "split") if key in floors}


def _check_cpu_tier(plan):
    """Check 19 (rules 7): candidate 1 of a placement whose CPU beats the link starts with the CPU expert tier."""
    if not plan.first or plan.decision != "build":
        return
    floors = _cpu_wins(plan.root)
    if floors is None:
        return
    plan.inputs["cpu_tier"] = {"floors_ms": floors}
    if any(lever.get("provider", "").startswith(CPU_TIER) for lever in plan.data["levers"]):
        return
    rejected = next((row for row in plan.data["gaps"] if row["gap"] == CPU_TIER_GAP), None)
    if rejected is None or rejected["answer"] != "rejected" or "evidence" not in rejected:
        raise ValueError("recon's placement prices misses computed on the CPU below misses copied over the link "
                         f"(plain-pass floors {floors} ms): candidate 1 starts with the CPU expert tier. Add a lever "
                         f"(speed or requirement) with provider {CPU_TIER!r}, or a gaps entry {{\"kpi\": \"decode\", "
                         f"\"gap\": \"{CPU_TIER_GAP}\", \"answer\": \"rejected\", \"evidence\": <a measurement in the "
                         "run>, \"detail\": ...} with the layer times that reject it on this machine")
    plan.inputs["cpu_tier"]["rejection"] = _evidence_file(plan.root, rejected["evidence"], f"gaps entry {CPU_TIER_GAP}")


# The checks build runs on a plan, in order, each with the first version of the plan rules it belongs to: a check
# raises ValueError for a plan that breaks it, and may record what crit derived in the plan's inputs.
CHECKS = ((1, _check_decision), (1, _check_gap_names), (1, _check_answered_gaps), (2, _check_limit_gaps),
          (3, _check_measured_only), (1, _prioritize), (1, _check_levers), (1, _check_forecast),
          (3, _check_forecast_priorities), (5, _check_reach), (4, _check_objective), (1, _check_rules_4_fields),
          (4, _check_review), (6, _check_activations), (1, _check_round), (7, _check_stale_rejections),
          (7, _check_cpu_tier), (1, _check_supersedes))


def _reach(root, data, facts, computed, forecast, needed):
    """Plan rules 5, check 15: how far the forecast's best case reaches toward each needed family's limit (Eval's, else
    the one crit computed), recorded beside the strategy as {limit, fraction, forecast_high, reach} with the limit's
    source. Below REACH the plan answers why the rest is out of reach: a gaps entry `limit` (answer cause) with the
    evidence, a file in the run that measured the bound; an engine's own speed alone is no bound."""
    found = {}
    causes = {row["kpi"]: row for row in data["gaps"] if row["gap"] == LIMIT_GAP}
    for name, flag in needed.items():
        row = facts["kpis"].get(name) or {}
        limit, source = row.get("limit"), "eval"
        if not limit and _positive(((computed or {}).get(name) or {}).get("limit")):
            limit, source = computed[name]["limit"], "computed"
        if not flag or not limit or name not in forecast:
            continue
        high = forecast[name]["high"]
        found[name] = {"limit": limit, "source": source, "fraction": row.get("fraction"), "forecast_high": high,
                       "reach": high / limit}
        if high / limit >= REACH:
            continue
        cause = causes.get(name)
        label = (f"The forecast's best case for {name} ({high:g}) is {high / limit:.0%} of its limit {limit:g}"
                 + (f" (the parent's Eval measured {row['fraction']:.0%})" if row.get("fraction") else ""))
        if cause is None or "evidence" not in cause:
            raise ValueError(f"{label}, below {REACH:.0%}: the strategy's ceiling is that limit, not the engine's own "
                             "measurement. Plan the levers that close more of it, or add a gaps entry {\"kpi\": "
                             f"\"{name}\", \"gap\": \"{LIMIT_GAP}\", \"answer\": \"cause\", \"detail\": ..., "
                             "\"evidence\": ...} naming the measured physical bound that keeps the rest out of reach (an "
                             "engine reading weights from host memory: the floor Eval prices from its memory tiers at "
                             "its expert cache hit rate, or from its declared host_bytes_per_pass), the evidence a file "
                             "in the run")
        found[name]["evidence"] = _evidence_file(root, cause["evidence"], label)
    return found


def _forecast_checks(order, forecast, facts):
    """Plan rules 3, check 12: what the forecast does against the priorities: more expected gain on p2 than on p1
    (priority_flag), or a p1 KPI forecast below the parent's score (above it for a time)."""
    found = []
    flag = priority_flag({"priorities": order, "forecast": forecast, "inputs": {"kpis": facts["kpis"]}})
    if flag:
        found.append(f"expects more gain on p2 ({', '.join(flag['p2'])}: {flag['p2_gain']:+.1%}) than on p1 "
                     f"({', '.join(flag['p1'])}: {flag['p1_gain']:+.1%})")
    for name in order["p1"]:
        for kpi in FAMILIES.get(name, (name,)):
            score = (facts["kpis"].get(kpi) or {}).get("score")
            if kpi in forecast and score and (forecast[kpi]["expected"] > score if kpi in MS
                                              else forecast[kpi]["expected"] < score):
                found.append(f"forecasts p1 {kpi} to fall ({score:g} -> {forecast[kpi]['expected']:g})")
    return found


def _evidence_file(root, path, name):
    """A gaps entry's evidence: a file retained in the run, {path, sha256}."""
    found = safe_path(root, path)
    if not found.is_file():
        raise ValueError(f"{name} evidence {path} is no file of this run: retain the measurement in the run (work/ or a "
                         "candidate's eval/) and name its path")
    return {"path": path, "sha256": file_sha(found)}


def _quality_rejection(root, path, activation, name):
    """Plan rules 3: whether the crit-eval result at `path` measured `activation` (its engine's prompt_activations)
    and its quality outside the noise: some set's share of the budget above the budget at the 5th percentile of its
    spread over positions (crit-eval's budget_spread). A rejection inside the quality noise is not a rejection, and
    neither is one judged under other quality terms than the run's: its budget share is relative to its own budget,
    so preserve's budget rejects what the speed mode's admits. The result's quality mode (which it states) and budget
    must be the run's, every resume's amendment applied."""
    from .store import amendments
    result = read_json(safe_path(root, path), max_bytes=EVAL_JSON_BYTES)
    correctness = load_run(root)["spec"]["correctness"]
    latest = max(p.stem for p in (root / "segments").glob("segment-*.json"))
    mode, budget = quality_mode(correctness), quality_terms(correctness, amendments(root, latest))[0]
    required = result.get("requirements") or {}
    judged, judged_budget = required.get("quality_mode"), required.get("quality_budget")
    if judged != mode or judged_budget != budget:
        raise ValueError(f"{name}: {path} was judged in quality mode {judged} with the budget {judged_budget}, this "
                         f"run's terms are quality mode {mode} with the budget {budget}: measure it again with crit "
                         f"eval run ... --quality-mode {mode} --quality-budget '{json.dumps(budget)}'")
    declared = (result.get("engine") or {}).get("prompt_activations")
    if declared != activation and not (isinstance(declared, dict) and declared and set(declared.values()) == {activation}):
        raise ValueError(f"{name}: {path} measured prompt activations {declared!r}, not {activation!r} (the engine's "
                         "prompt_activations in that result)")
    sets = ((result.get("quality") or {}).get("sets")) or {}
    if not sets:
        raise ValueError(f"{name}: {path} scored no quality sets")
    if not any("budget_spread" in s for s in sets.values()):
        raise ValueError(f"{name}: {path} has no quality spread over positions (budget_spread, which crit-eval 0.1.0 "
                         "reports): measure it again so the rejection can be told from the quality noise")
    over = [set_name for set_name, s in sets.items()
            if ((s.get("budget_spread") or {}).get("budget_used") or {}).get("p05", 0) > 1.0]
    if not over:
        raise ValueError(f"{name}: {path} keeps every set's budget share within its spread over positions (5th "
                         "percentile at or below the budget): inside the quality noise, so not a rejection")
    return over


def _measured_only(root, data, facts, result, needed, version):
    """Plan rules 3: a family marked not needed below MEASURED_ONLY of its limit stands only on measured rejections:
    every gaps entry on it names its evidence (a file in the run), every Eval gap on it answers rejected, and for
    prefill every prompt activation format Eval prices faster than the engine's own is rejected, by Eval's own quality
    evidence or by a gaps entry `activation:<format>` whose evidence is a crit-eval result that measured that format
    outside the quality noise (_quality_rejection). Returns the evidence recorded, per gap."""
    recorded = {}
    for name, flag in needed.items():
        fraction = family(facts, name, version)[0]
        if flag or fraction is None or fraction >= MEASURED_ONLY:
            continue
        for row in data["gaps"]:
            if FAMILY.get(row["kpi"], row["kpi"]) != name:
                continue
            label = f"{name} is marked not needed at {fraction:.0%} of its limit, below {MEASURED_ONLY:.0%}: gaps entry {row['gap']!r}"
            if "evidence" not in row:
                raise ValueError(f"{label} needs the evidence of its measured rejection (a path in the run); a cause "
                                 "alone does not stand below half the limit")
            if row["gap"] not in (LIMIT_GAP,) and not row["gap"].startswith(ACTIVATION_GAP) and row["answer"] != "rejected":
                raise ValueError(f"{label} answers {row['answer']}: below half the limit an Eval gap is answered by a "
                                 "measured rejection")
            recorded[f"{row['kpi']}:{row['gap']}"] = _evidence_file(root, row["evidence"], label)
        if name == "prefill":
            activations = (((result or {}).get("status") or {}).get("kpis") or {}).get("prefill", {}).get("activations")
            if activations:
                own = (activations.get("declared") or {}).get("ceiling_score")
                if own is None:
                    raise ValueError(f"prefill is marked not needed at {fraction:.0%} of its limit, below "
                                     f"{MEASURED_ONLY:.0%}, and the engine declares no prompt_activations: declare them, so "
                                     "Eval names the format in use and the faster ones to reject")
                answered = {row["gap"][len(ACTIVATION_GAP):]: row for row in data["gaps"]
                            if row["gap"].startswith(ACTIVATION_GAP)}
                for mode, row in (activations.get("modes") or {}).items():
                    if (row.get("ceiling_score") or 0) <= own or row.get("state") == "fails":
                        continue   # no faster than the engine's own, or ruled out by Eval's own quality evidence
                    if mode not in answered:
                        raise ValueError(f"prefill is marked not needed at {fraction:.0%} of its limit, and Eval prices "
                                         f"prompt activations {mode} at {row['ceiling_score']:,.0f} tok/s, above the "
                                         f"engine's own {own:,.0f}: reject it with a quality measurement, gaps entry "
                                         f"{{\"kpi\": \"prefill\", \"gap\": \"{ACTIVATION_GAP}{mode}\", \"answer\": "
                                         "\"rejected\", \"evidence\": <its crit-eval result>, \"detail\": ...}")
                    sets = _quality_rejection(root, answered[mode]["evidence"], mode, f"gaps entry {ACTIVATION_GAP}{mode}")
                    recorded[f"prefill:{ACTIVATION_GAP}{mode}"]["over_budget_sets"] = sets
    return recorded


FATES = {"built", "pruned", "deferred"}


def check_fates(plan, fates, required):
    """A proposal's lever_fates against its plan: {lever_id: {fate, detail}}, fate built, pruned or deferred, a detail
    for pruned and deferred, no unknown lever; with `required`, every lever of the plan has exactly one fate."""
    if not isinstance(fates, dict):
        raise ValueError("lever_fates maps each lever id of the plan to {fate, detail}")
    ids = [lever["id"] for lever in plan["levers"]]
    unknown = sorted(set(fates) - set(ids))
    if unknown:
        raise ValueError("lever_fates names " + ", ".join(unknown) + f", which {plan['plan_id']} does not plan")
    for lever_id, row in fates.items():
        keys(row, {"fate"}, {"detail"})
        if row["fate"] not in FATES:
            raise ValueError(f"{lever_id} fate is built, pruned or deferred")
        if row["fate"] != "built" or "detail" in row:
            text(row.get("detail"), f"{lever_id} detail (why it was {row['fate']})", 4000)
    missing = [lever_id for lever_id in ids if lever_id not in fates]
    if required and missing:
        raise ValueError("lever_fates gives no fate to " + ", ".join(missing) + f" of {plan['plan_id']}: built, "
                         "pruned or deferred, each with a detail unless built")
    return fates


def check_link(root, run, plan_id, parent):
    """A proposal's plan: it exists, is the newest of its chain, builds, is unlinked and starts from the same parent.
    With budget_counts evaluated a plan whose candidate never reached Eval (a draft, sealed with what stopped it) links
    again: the new candidate is the draft's fix."""
    plan = load(root, run, plan_id)
    summary = next(s for s in summaries(root, run) if s["plan_id"] == plan_id)
    if summary["superseded_by"] is not None:
        raise ValueError(f"{plan_id} is superseded by {summary['superseded_by']}: name the newest plan")
    linked = summary["candidate_id"]
    if linked is not None:
        rungs = ladder.ladder(run)
        drafts = counts_evaluated(run) or rungs
        if not drafts or linked in (ladder.counted(root) if rungs else evaluated_candidates(root)):
            raise ValueError(f"{plan_id} is already linked to {linked}" + (
                ", which is on the candidate ladder" if rungs else ", which went through Eval" if drafts else "")
                + ": record a new plan")
        if not (safe_path(root, f"candidates/{linked}") / "seal.json").is_file():
            raise ValueError(f"{plan_id} built {linked}, a draft that has not reached " + (
                "the candidate ladder" if rungs else "Eval") + ": seal it with what stopped it (crit run ... seal "
                f"--candidate {linked} --lifecycle build_failed or check_failed --failure ...) before building its fix "
                "from the plan")
    if plan["decision"] != "build":
        raise ValueError(f"{plan_id} is a stop plan; a candidate needs a build plan")
    if plan["parent_candidate_id"] != parent:
        raise ValueError(f"{plan_id} starts from {plan['parent_candidate_id']}, the proposal from {parent}")
    return plan


def require_stop_plan(root, run, bounds):
    """search_exhausted with budgeted candidates left: the segment's newest unsuperseded plan is a stop plan that
    froze the run's newest Eval."""
    plans = [s for s in summaries(root, run) if s["segment_id"] == bounds["segment_id"] and s["superseded_by"] is None]
    newest = newest_eval(root)
    expected = newest.relative_to(root).as_posix() if newest is not None else None
    hint = ("record a stop plan (decision stop, decode and prefill not needed) from " + (expected or "the run's state")
            + " with crit run ... plan, or build the next candidate")
    if not plans or plans[-1]["decision"] != "stop":
        raise ValueError("search_exhausted with candidates left in the budget requires a stop plan: " + hint)
    plan = load(root, run, plans[-1]["plan_id"])
    if (plan["inputs"]["eval"] or {}).get("path") != expected:
        raise ValueError(f"{plan['plan_id']} froze {(plan['inputs']['eval'] or {}).get('path')}, not the newest Eval "
                         f"{expected}: " + hint)
    return plan


def _gain(kpi, expected, score):
    """The forecast's expected relative gain on a KPI against the parent's score, in its improving direction."""
    if not expected or not score:
        return 0.0
    return max(0.0, score / expected - 1 if kpi in MS else expected / score - 1)


def priority_flag(plan):
    """When most of the forecast's expected gain falls on a p2 KPI instead of p1: the gain summed per tier (decode and
    prefill by family, ttft alone), or None."""
    order, forecast, facts = plan.get("priorities"), plan.get("forecast"), plan["inputs"].get("kpis")
    if not order or not forecast or not facts:
        return None
    tier = {name: key for key in ("p1", "p2", "p3") for name in order[key]}
    totals = {"p1": 0.0, "p2": 0.0}
    for kpi, row in forecast.items():
        key = tier.get(FAMILY.get(kpi, kpi))
        if key in totals:
            totals[key] += _gain(kpi, row["expected"], (facts.get(kpi) or {}).get("score"))
    if totals["p2"] > totals["p1"]:
        return {"p1": order["p1"], "p2": order["p2"], "p1_gain": totals["p1"], "p2_gain": totals["p2"]}
    return None


def outcome(run, plan, candidate_id, attempt_id, eval_ref, status, fates=None):
    """The plan's forecast next to the candidate's measured Eval, and the untiered numbers that fell against the
    parent's Eval by more than their noise (at all when the noise is unknown). Facts only, no verdict."""
    now_facts, before = extract(status), plan["inputs"]
    kpis = {}
    for kpi, row in (plan["forecast"] or {}).items():
        measured = (now_facts["kpis"].get(kpi) or {}).get("score")
        kpis[kpi] = {"forecast": row, "measured": measured,
                     "inside": row["low"] <= measured <= row["high"] if measured is not None else None,
                     "error": measured / row["expected"] - 1 if measured is not None else None}
    flags = []
    for name, lower_better in UNTIERED.items():
        parent = ((before.get("untiered") or {}).get(name) or {}).get("score")
        measured = now_facts["untiered"][name]["score"]
        if parent is None or measured is None:
            continue
        noises = [n for n in (((before.get("untiered") or {}).get(name) or {}).get("noise"),
                              now_facts["untiered"][name]["noise"]) if n is not None]
        noise = max(noises) if noises else None
        change = parent / measured - 1 if lower_better else measured / parent - 1
        if change < -(noise or 0):
            flags.append({"kpi": name, "parent": parent, "measured": measured, "change": change, "noise": noise})
    estimates = {}
    for kpi in before.get("estimated") or []:
        estimate = plan["estimated_limits"][kpi]["limit"]
        row = now_facts["kpis"].get(kpi) or {}
        measured, limit = row.get("score"), row.get("limit")
        estimates[kpi] = {"estimate": estimate, "measured": measured,
                          # A rate above its estimated ceiling, or a time below its estimated floor: it was too low.
                          "exceeded": (measured < estimate if kpi in MS else measured > estimate) if measured is not None else None,
                          "eval_limit": limit, "error": estimate / limit - 1 if limit is not None else None}
    requirements = [{"id": lever["id"], "serves": lever["serves"]} for lever in plan["levers"]
                    if lever.get("kind") == "requirement"]
    # The alternative strategy's ceiling next to the measurement, when the plan built the alternative.
    alternative = {}
    if plan["strategy"]["choice"] == "alternative":
        for kpi, entry in (before.get("alternative_limits") or {}).items():
            measured = (now_facts["kpis"].get(kpi) or {}).get("score")
            alternative[kpi] = {"limit": entry["limit"], "measured": measured,
                                "fraction": measured / entry["limit"] if measured else None,
                                "exceeded": measured > entry["limit"] if measured is not None else None}
    # Each lever's fate from the candidate's proposal next to what it was forecast to change; null when none was given.
    levers = [{"id": lever["id"], "kind": lever.get("kind", "speed"), "changes": lever["changes"],
               "fate": (fates or {}).get(lever["id"])} for lever in plan["levers"]]
    return record("plan_outcome", run["run_id"], plan_id=plan["plan_id"], candidate_id=candidate_id, attempt_id=attempt_id,
                  recorded_at=now(), plan_sha256=digest(plan), eval=eval_ref, kpis=kpis, estimates=estimates, flags=flags,
                  requirements=requirements, levers=levers, forecast_on_p2=priority_flag(plan), alternative=alternative)
