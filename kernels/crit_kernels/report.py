"""One report over crit's kernels: every structure's gates and its speed against its yardstick library, on the shapes of
the models given.

crit kernels report --model A --model B ... takes each structure's shapes from every model (crit_kernels.model, the
bench's shapes()), keeps the distinct ones by the structure's key and runs at most `most` of them (the smallest, the
middle and the largest by the key's product), so a structure several models share is not run once a model. On each:

- gates   crit_kernels.gates (nonfinite, repeat, edges, sums; limits once a kernel with --limits)
- speed   the default build's call and the yardstick's (kernel.json's structure "yardstick", the bench's yardstick())
          on the structure's first and last tune_rows, each timed as the sweep times (crit_kernels.sweep.Timer: L2
          flushed by a write, a GPU spin before the call, a CUDA graph replay where it can be captured, the median)

The JSON report holds every row; markdown() gives one line a kernel (gates passed or the failures, the slowest and
fastest speed against the library) and one a structure."""
import json
import math
import statistics

from . import build as kbuild


def _size(shape, key):
    return math.prod(int(shape[k]) for k in key if isinstance(shape.get(k), (int, float)) and shape.get(k)) or 0


def chosen_shapes(bench, spec, models, most=3):
    """The bench's distinct shapes over the models (by the structure's key), at most `most`: the smallest, the middle
    and the largest by the key's product. Returns (shapes, errors)."""
    seen, errors = {}, []
    key = spec.get("key", [])
    for m in models:
        try:
            for shape in bench.shapes(m) or []:
                seen.setdefault(json.dumps({f: shape.get(f) for f in key}, sort_keys=True, default=str), shape)
        except Exception as error:  # noqa: BLE001 - a model a bench cannot read is a fact to report
            errors.append(f"{m.folder.name}: {type(error).__name__}: {str(error)[:200]}")
    found = sorted(seen.values(), key=lambda s: _size(s, key))
    if len(found) <= most:
        return found, errors
    picks = sorted({round(i * (len(found) - 1) / (most - 1)) for i in range(most)}) if most > 1 else [len(found) - 1]
    return [found[i] for i in picks], errors


def _speed(kernel_id, folder, manifest, chosen, timer, log):
    """{structure: [{shape, rows, us, timed, library, library_us, vs_library | library_note}]}: the default build
    against the yardstick, at each chosen shape's first and last tune_rows."""
    import torch
    from .sweep import _bench_class
    module = kbuild.load(kernel_id)
    out = {}
    for s, shapes in chosen.items():
        spec = manifest["structures"][s]
        bench = _bench_class(folder, spec["bench"])
        bench.default = module
        tune = spec.get("tune_rows") or [1]
        rows_list = sorted({tune[0], tune[-1]})
        library = (spec.get("yardstick") or {}).get("library")
        entries = []
        for shape in shapes:
            for rows in rows_list:
                entry = {"shape": {f: shape.get(f) for f in spec.get("key", [])}, "rows": rows, "library": library}
                try:
                    state = bench.inputs(shape, rows, "cuda")
                    def call_mine(state=state):
                        return bench.run(module, state)
                    mine = timer.graph_of(call_mine)
                    entry["timed"] = "eager" if mine is call_mine else "graph"
                    entry["us"] = round(timer(mine), 3)
                    call = bench.yardstick(state) if library and hasattr(bench, "yardstick") else None
                    if call is not None:
                        theirs = timer.graph_of(call)
                        entry["library_us"] = round(timer(theirs), 3)
                        entry["us"] = round(min(entry["us"], timer(mine)), 3)   # paired with the library's time
                        entry["vs_library"] = round(entry["library_us"] / entry["us"], 3)
                    elif library:
                        entry["library_note"] = "not installed here"
                    torch.cuda.synchronize()
                except Exception as error:  # noqa: BLE001 - a shape a call cannot take is a fact to report
                    torch.cuda.synchronize()
                    entry["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:200] if str(error) else ''}"
                entries.append(entry)
                log(f"{kernel_id}/{s} {entry['shape']} rows {rows}: " + (
                    f"{entry.get('us')} us" + (f", {library} {entry['library_us']} us ({entry['vs_library']}x)"
                                               if "library_us" in entry else "") if "error" not in entry
                    else entry["error"]))
        out[s] = entries
    return out


def report(model_folders, kernels=None, most=3, gates_on=True, speed=True, limits=False, repeat=25, log=print):
    """{"models", "kernels": {id: {"shapes": {structure: [shape]}, "gates", "speed", "passed", "notes"}}}."""
    from .gates import gates
    from .model import Model
    from .sweep import Timer, _bench_class
    models = [Model(f) for f in model_folders]
    ids = kernels or sorted(kbuild.manifests())
    timer = None
    result = {"models": [str(f) for f in model_folders], "kernels": {}}
    for kernel_id in ids:
        folder, manifest = kbuild.find(kernel_id, None)
        chosen, notes = {}, []
        for s, spec in manifest.get("structures", {}).items():
            if not spec.get("bench"):
                notes.append(f"{s}: no bench")
                continue
            shapes, errors = chosen_shapes(_bench_class(folder, spec["bench"]), spec, models, most)
            notes += [f"{s}: {e}" for e in errors]
            if shapes:
                chosen[s] = shapes
            else:
                notes.append(f"{s}: no shape in these models")
        entry = {"shapes": {s: [{f: x.get(f) for f in manifest["structures"][s].get("key", [])} for x in v]
                            for s, v in chosen.items()}, "notes": notes}
        try:
            if gates_on and (chosen or limits):
                g = gates(kernel_id, shapes=chosen, structures=list(chosen), limits=limits, log=log)
                entry["gates"] = {"structures": g["structures"], **({"limits": g["limits"]} if "limits" in g else {})}
                entry["passed"] = g["passed"]
            if speed and chosen:
                if timer is None:
                    import torch
                    timer = Timer(torch, 3, repeat)
                entry["speed"] = _speed(kernel_id, folder, manifest, chosen, timer, log)
        except Exception as error:  # noqa: BLE001 - one kernel that cannot build does not stop the report
            entry["error"] = f"{type(error).__name__}: {str(error)[:500]}"
            entry["passed"] = False
        result["kernels"][kernel_id] = entry
    return result


def _name(folder):
    """A checkpoint's name: its folder's, or its parent's where the folder is a revision (a 40-hex commit)."""
    parts = str(folder).replace("\\", "/").rstrip("/").split("/")
    last = parts[-1]
    return parts[-2] if len(parts) > 1 and len(last) == 40 and all(c in "0123456789abcdef" for c in last) else last


def markdown(result):
    """The report as markdown: a line a kernel, then a table of its structures (gates, speed against the library)."""
    lines = ["# crit kernels report", "", "Models: " + ", ".join(_name(f) for f in result["models"]), ""]
    lines += ["| Kernel | Gates | Structures | Speed against the library (default knobs) |", "|---|---|---|---|"]
    detail = []
    for kid, e in result["kernels"].items():
        fails = [f"{s} {r['shape']}: {g['gate']} ({g.get('reason', '')[:80]})"
                 for s, rows in (e.get("gates") or {}).get("structures", {}).items() for r in rows for g in r["gates"]
                 if g["verdict"] == "fail"]
        lim = (e.get("gates") or {}).get("limits")
        if lim and lim.get("verdict") == "fail":
            fails.append("limits")
        gate_text = "error: " + e["error"][:80] if "error" in e else (
            "not run" if "passed" not in e else "pass" if not fails else f"{len(fails)} failing")
        ratios = [x["vs_library"] for rows in (e.get("speed") or {}).values() for x in rows if "vs_library" in x]
        speed = (f"{min(ratios):.2f}x .. {max(ratios):.2f}x (median {statistics.median(ratios):.2f}x, "
                 f"{len(ratios)} points)") if ratios else "no library timed"
        lines.append(f"| {kid} | {gate_text} | {len(e['shapes'])} | {speed} |")
        detail.append(f"## {kid}")
        libraries = sorted({x["library"] for rows in (e.get("speed") or {}).values() for x in rows if x.get("library")})
        detail += [f"- library: {name}" for name in libraries]
        detail += [f"- gate failure: {f}" for f in fails]
        detail += [f"- note: {n}" for n in e.get("notes", [])]
        for s, rows in (e.get("speed") or {}).items():
            for x in rows:
                text = x.get("error") or (f"{x['us']} us" + (
                    f", the library {x['library_us']} us: {x['vs_library']}x" if "library_us" in x else
                    f" ({x['library_note']})" if "library_note" in x else ""))
                detail.append(f"- {s} {x['shape']} rows {x['rows']}: {text}")
        detail.append("")
    return "\n".join(lines + [""] + detail)
