"""Side-by-side comparison of result files: `crit eval compare A.json B.json ...`."""
from pathlib import Path

from . import load_json, protocol_version
from .status import decode_text_identity


def _fmt(value, spec):
    return "–" if value is None else format(value, spec)


def _major(result):
    try:
        return protocol_version(result.get("protocol"))[0]
    except ValueError:
        return None


def compare(paths):
    results = [load_json(p) for p in paths]
    names = [r.get("label") or r["engine"]["name"] or Path(p).stem for r, p in zip(results, paths)]
    rows = [("Engine", [r["engine"]["name"] for r in results]),
            ("Host", [r["host"]["platform"] for r in results]),
            ("Model data", [r["model_sha256"][:12] for r in results])]
    warn = []
    if len({r["model_sha256"] for r in results}) > 1 or len({r["suite_sha256"] for r in results}) > 1:
        warn.append("Results use different prepared data or suites; they are not comparable.")
    if len({_major(r) for r in results}) > 1:
        warn.append("Results use different Eval protocol major versions; their scores are not comparable.")
    texts = [decode_text_identity(r) for r in results]
    rows.append(("Decode text", [("pinned " + t[:12]) if t != "generated" else "own output" for t in texts]))
    if len(set(texts)) > 1:
        warn.append("Results decoded different texts; their decode scores are not comparable.")
    if any(r.get("native_libraries") for r in results):
        rows.append(("Native libraries", [", ".join(f"{Path(e['path']).name} {e.get('sha256', '?')[:12]}"
                                                    for e in r.get("native_libraries") or []) or "–" for r in results]))
    case_ids = []
    for r in results:
        for case in r.get("speed", {}).get("cases", {}):
            if case not in case_ids:
                case_ids.append(case)
    if any(r.get("status") for r in results):
        rows.append(("**Gates**", [_gates((r.get("status") or {}).get("gates")) for r in results]))
    if case_ids:
        if any((r.get("sampled_speed") or {}).get("decode_score") for r in results):
            # crit-eval-v7: the decode at the model's own sampling, which the decode ceiling below describes.
            rows.append(("**Sampled decode score** (geomean tok/s)",
                         [_fmt((r.get("sampled_speed") or {}).get("decode_score"), ".1f") for r in results]))
            rows.append(("Sampling", [_sampling(r.get("sampled_speed") or {}) for r in results]))
        rows.append(("**Decode score** (greedy, geomean tok/s)", [_fmt(r.get("speed", {}).get("decode_score"), ".1f") for r in results]))
        cells = []
        for r in results:
            p = (r.get("headroom") or {}).get("prefill") or {}
            cell = _fmt(p.get("score") or r.get("speed", {}).get("prefill_score"), ",.0f")
            if p.get("ceiling_score"):
                cell += f" ({p['fraction_of_ceiling']:.0%} of {p['ceiling_score']:,.0f} ceiling)"
            cells.append(cell)
        rows.append(("**Prefill score** (geomean tok/s)", cells))
        cells = []
        for r in results:
            t = (r.get("headroom") or {}).get("ttft") or {}
            cell = _fmt(t.get("score_ms") or r.get("speed", {}).get("ttft_score_ms"), ",.0f")
            if t.get("floor_ms"):
                cell += f" ({t['fraction_of_floor']:.0%} of {t['floor_ms']:,.0f} ms floor)"
            cells.append(cell)
        rows.append(("**TTFT** (geomean ms)", cells))
        for case in case_ids:
            cells = []
            for r in results:
                c = r.get("speed", {}).get("cases", {}).get(case)
                if not c or c["status"] != "measured":
                    cells.append("skipped" if c else "–")
                    continue
                cell = f"{c['decode_tps']:.1f}"
                if c.get("tokens_per_round"):
                    cell += f" ({c['tokens_per_round']:.2f}/round"
                    room = r.get("headroom", {})
                    # From v7 the headroom cases are the sampled suite's; these rows are the greedy suite's.
                    saturation = (room.get("cases", {}).get(case, {}).get("saturation")
                                  if room.get("decode_suite", "greedy") == "greedy" else None)
                    cell += f", {saturation:.0%} of cap)" if saturation is not None else ")"
                if c.get("stop_crossed"):
                    cell += " ⚠stop"
                cells.append(cell)
            rows.append((f"Decode tok/s: {case}", cells))
        for case in case_ids:
            if all(r.get("speed", {}).get("cases", {}).get(case, {}).get("prompt_tokens", 0) >= 2048 for r in results):
                rows.append((f"Prefill tok/s: {case}",
                             [_fmt(r["speed"]["cases"][case].get("prefill_tps"), ".0f") for r in results]))
                cells = []
                for r in results:
                    cell = _fmt(r["speed"]["cases"][case].get("ttft_ms"), ",.0f")
                    floor = (((r.get("headroom") or {}).get("prefill") or {}).get("cases", {}).get(case) or {}).get("floor")
                    cells.append(cell + (f" (floor {floor['total_ms']:,.0f})" if floor else ""))
                rows.append((f"TTFT ms: {case}", cells))
    if any(r.get("headroom") for r in results):
        cells = []
        for r in results:
            h = r.get("headroom") or {}
            cells.append("–" if h.get("pass_efficiency") is None else
                         f"{h['step_ms']:.2f} ms vs {h['pass_floor_ms']:.2f} ms floor ({h['pass_efficiency']:.0%})")
        rows.append(("Model pass vs bandwidth floor", cells))
        cells = []
        for r in results:
            p = (r.get("headroom") or {}).get("projection") or {}
            ceiling = p.get("ceiling_score") or p.get("speed_of_light_score")
            fraction = p.get("fraction_of_ceiling") or p.get("fraction_of_speed_of_light")
            cells.append("–" if not ceiling else f"{ceiling:.1f}" + (f" (at {fraction:.0%})" if fraction else ""))
        rows.append(("Decode ceiling" + (" (sampled)" if any((r.get("headroom") or {}).get("decode_suite") == "sampled"
                                                            for r in results) else ""), cells))
    sets_seen = []
    for r in results:
        for name in (r.get("quality") or {}).get("sets") or {}:
            if name not in sets_seen:
                sets_seen.append(name)
    for name in sets_seen:
        cells = []
        for r in results:
            q = ((r.get("quality") or {}).get("sets") or {}).get(name)
            cells.append("–" if q is None else
                         f"PPL ×{q['ppl_ratio']:.4f}, KLD {q['kld_mean']:.1e} (p99 {q['kld_p99']:.1e}), "
                         f"top-1 {q['top1_agreement']:.3f} {'✓' if q['passed'] else '✗'}")
        rows.append((f"Quality: {name}", cells))
    if any("exactness" in r for r in results):
        cells = []
        for r in results:
            e = r.get("exactness")
            if e is None:
                cells.append("–")
            else:
                same = sum(c["identical"] for c in e["cases"].values())
                cells.append(f"{same}/{len(e['cases'])} cases identical {'✓' if e['passed'] else '✗'}")
        rows.append(("Speculative = plain greedy", cells))
    gap_names = []
    for r in results:
        for gap in _gaps(r):
            if gap["name"] not in gap_names:
                gap_names.append(gap["name"])
    for name in gap_names:
        cells = []
        for r in results:
            gap = next((g for g in _gaps(r) if g["name"] == name), None)
            cells.append("–" if not gap else f"+{gap['projected_gain']:.0%} {gap['kpi']}")
        rows.append((f"Gap: {name}", cells))
    lines = ["| | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    lines += ["| " + label + " | " + " | ".join(cells) + " |" for label, cells in rows]
    return "\n".join(warn + lines)


def _gaps(result):
    room = result.get("headroom") or {}
    return room.get("gaps", room.get("levers", []))


def _gates(gates):
    if not gates:
        return "–"
    failed = [g["name"] for g in gates if g["state"] == "fail"]
    unmeasured = [g["name"] for g in gates if g["state"] == "unmeasured"]
    return ("fail: " + ", ".join(failed) if failed else "unmeasured: " + ", ".join(unmeasured) if unmeasured else "pass")


def _sampling(sampled):
    """The sampled suite's settings, marked when they are not the model's own."""
    if not sampled.get("sampling"):
        return "–"
    text = ", ".join(f"{key} {value}" for key, value in sampled["sampling"].items())
    return text + {"override": " (user override)", "protocol_default": " (protocol default)"}.get(
        sampled.get("sampling_source"), "")
