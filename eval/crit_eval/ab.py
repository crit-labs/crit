"""Paired speed runs: `crit eval ab --adapter A --config CA --config-b CB --model M --data D --out FILE`.

A shared device drifts between sessions (other processes, clocks, temperature) by more than the
run-to-run spread inside one session, so two results measured apart cannot confirm a small change.
`ab` measures both engines in one session: each round runs the speed suite for both, in alternating
order (A B, then B A), loading and closing each engine in turn. It reports every round's scores and the
paired ratio B / A per round; the median and spread of those ratios are facts about the pair, measured
under the same conditions.
"""
import datetime
import time
from pathlib import Path

from . import EVAL_DIR, load_json, metrics, write_json
from .pin import load_continuations
from .prepare import PREPARED, check_prepared
from .runner import load_adapter, open_engine, run_speed

KPIS = ("decode_score", "prefill_score", "ttft_score_ms")


def run_ab(adapter_a, config_a, adapter_b, config_b, model, data, out, rounds=2, warmups=None, repetitions=None,
           log=print, context_tokens=None):
    protocol = load_json(EVAL_DIR / "protocol.json")
    prepared = load_json(Path(data) / PREPARED)
    check_prepared(prepared, protocol, data)
    if rounds < 1:
        raise SystemExit("--rounds must be at least 1")
    continuations = load_continuations(data, prepared)
    warmups = protocol["speed"]["warmup_runs"] if warmups is None else warmups
    repetitions = protocol["speed"]["repetitions"] if repetitions is None else repetitions
    engines = {"a": (load_adapter(adapter_a), dict(config_a or {})), "b": (load_adapter(adapter_b), dict(config_b or {}))}
    result = {"started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "model_sha256": prepared["model"]["sha256"], "suite_sha256": prepared["suite_sha256"],
              "warmup_runs": warmups, "repetitions": repetitions, "context_tokens": context_tokens, "rounds": [],
              "engines": {}}
    for r in range(rounds):
        order = ("a", "b") if r % 2 == 0 else ("b", "a")
        row = {"order": list(order)}
        for key in order:
            module, config = engines[key]
            started = time.perf_counter()
            engine, info = open_engine(module, model, config, speculative=True, context_tokens=context_tokens)
            try:
                speed = run_speed(engine, info, prepared, protocol, warmups, repetitions,
                                  lambda message, k=key: log(f"round {r + 1} {k}: {message}"), continuations)
            finally:
                engine.close()
            result["engines"].setdefault(key, {"name": info["name"], "configuration": info["configuration"]})
            row[key] = {kpi: speed.get(kpi) for kpi in KPIS} | {"complete": speed["complete"],
                                                                "decode_text": speed["decode_text"],
                                                                "seconds": time.perf_counter() - started}
        result["rounds"].append(row)
    result["paired"] = paired(result["rounds"])
    result["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(out, result)
    return result


def paired(rounds):
    """Per KPI: the B / A ratio in each round, their median and spread (max - min). For time to first
    token the ratio is A / B, so a ratio above 1 means B is faster on every KPI."""
    out = {}
    for kpi in KPIS:
        ratios = []
        for row in rounds:
            a, b = row["a"].get(kpi), row["b"].get(kpi)
            if a and b:
                ratios.append(a / b if kpi == "ttft_score_ms" else b / a)
        out[kpi] = {"ratios": ratios, "median": metrics.median(ratios),
                    "spread": (max(ratios) - min(ratios)) if len(ratios) > 1 else None}
    return out


def render(result):
    names = {k: v["name"] for k, v in result["engines"].items()}
    lines = [f"A = {names.get('a')}, B = {names.get('b')}; {len(result['rounds'])} rounds, order alternating"]
    for i, row in enumerate(result["rounds"], 1):
        cells = []
        for key in ("a", "b"):
            r = row[key]
            cells.append(f"{key.upper()}: decode {_fmt(r['decode_score'], '.1f')}, prefill {_fmt(r['prefill_score'], ',.0f')}, "
                         f"ttft {_fmt(r['ttft_score_ms'], ',.0f')} ms")
        lines.append(f"round {i} ({' '.join(k.upper() for k in row['order'])}): " + "; ".join(cells))
    for kpi, label in (("decode_score", "decode"), ("prefill_score", "prefill"), ("ttft_score_ms", "ttft (speed)")):
        p = result["paired"][kpi]
        if p["median"] is not None:
            lines.append(f"B vs A {label}: {p['median'] - 1:+.1%} median over rounds"
                         + (f", spread {p['spread']:.1%}" if p["spread"] is not None else ""))
    return "\n".join(lines)


def _fmt(value, spec):
    return "–" if value is None else format(value, spec)
