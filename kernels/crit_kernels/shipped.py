"""Tuning tables a kernel carries in its folder: what sweeps measured on one compute capability, applied there only.

A kernel folder may hold tunings/<capability>.json (tunings/12.0.json: swept on GPUs of compute capability 12.0), a
tuning (crit-kernel-tuning-v1, so check_tuning reads it) keyed by each structure's shape key ("key" in kernel.json:
weights, N, K), never by a model. It is written by `crit kernels ship ID --tuning SWEEP/ID.json` from a finished
sweep's tuning and report (merge()), never by hand, and records where each entry came from:

    {"schema": "crit-kernel-tuning-v1", "kernel": ID, "capability": "12.0",
     "sweeps": [{"source": the kernel source digest swept, "device": the GPU's name, "capability", "date"}, ...],
     "structures": {S: {"configs": [knob dicts, the defaults first],
                        "table": [{"key", "rows", "config", "sweep": index into sweeps}, ...],
                        "report": [the sweep report's rows for those keys, each with "sweep"]}}}

How it is used (fill(), from crit_kernels.build.build):

- Only on a GPU of the table's compute capability (device_capability(); none to ask: never). On any other GPU, or for a
  kernel without a table, nothing changes: the default knobs, and the sweep runs as before.
- Per structure and shape key, the engine's own tuning first (given explicitly or found in its tuning folder), then the
  table, then the defaults: a tuning that lacks a shape gets the table's entries for that shape, one that has the shape
  keeps its own. The defaults' own rules (kernel.json "default_rows", entries for every key) come after the table's.
- `crit kernels sweep` leaves out the shapes the table covers for a structure (they are written into the tuning it
  outputs from the table, and their report rows into its report, marked "shipped"), unless --no-shipped; a sweep
  whose shapes are all covered builds nothing.
- The kernel's source may have changed since an entry was swept. As for an engine's tuning: a retired structure's
  entries move to the structure that replaced it (check_tuning, retire), a knob added since takes its default, and a
  knob or value removed since makes the table fail check_tuning: the kernel's tests check every table against its
  kernel.json (kernels/crit_kernels/tests/test_shipped.py), and a build that meets a table that fails ignores it with a
  warning. `crit kernels ship` drops such entries from the table it rewrites.

A table is a measurement on the named capability, not a default: no other GPU inherits it, and it never replaces a
knob's default in kernel.json (kernels/README.md, guarding against one GPU)."""
import datetime
import json
import sys
from pathlib import Path

from . import build as kbuild

FOLDER = "tunings"
_cache = {}      # path -> (mtime_ns, size, table)
_warned = set()  # paths whose table failed check_tuning, reported once


def path(folder, capability):
    """The table a kernel folder carries for a compute capability ('12.0')."""
    return Path(folder) / FOLDER / f"{capability}.json"


def _read(found):
    stat = found.stat()
    known = _cache.get(str(found))
    if known and known[:2] == (stat.st_mtime_ns, stat.st_size):
        return known[2]
    data = json.loads(found.read_text(encoding="utf-8"))
    _cache[str(found)] = (stat.st_mtime_ns, stat.st_size, data)
    return data


def table(kernel_id, capability=None, root=None):
    """(table, path) the kernel carries for `capability` (None: the current CUDA device's), checked against its
    kernel.json, or (None, None): no device to ask, no table, or one that fails check_tuning (said once on stderr)."""
    capability = capability or kbuild.device_capability()
    if not capability:
        return None, None
    folder, manifest = kbuild.find(kernel_id, root)
    found = path(folder, capability)
    if not found.is_file():
        return None, None
    data = _read(found)
    try:
        if str(data.get("capability")) != str(capability):
            raise ValueError(f"it says capability {data.get('capability')!r}")
        kbuild.check_tuning(manifest, data)
    except (ValueError, KeyError, TypeError) as error:
        if str(found) not in _warned:
            _warned.add(str(found))
            print(f"crit: kernel {kernel_id}: its shipped tuning {found} is not used ({error}); `crit kernels ship` "
                  "rewrites it from a sweep.", file=sys.stderr)
        return None, None
    return data, found


def _key_text(key):
    return None if key is None else json.dumps({f: int(v) for f, v in key.items()}, sort_keys=True)


def shape_key(spec, shape):
    """A shape's key as the tables hold it (the structure's key fields), as text."""
    return _key_text({f: shape[f] for f in spec.get("key", [])})


def keys(tuning, structure):
    """The shape keys a tuning's table names for a structure (text, entries for every key left out)."""
    t = ((tuning or {}).get("structures") or {}).get(structure) or {}
    return {_key_text(e["key"]) for e in t.get("table") or [] if e.get("key") is not None}


def fill(manifest, tuning, shipped, only=None):
    """The tuning (None: the kernel's default tuning) checked (check_tuning), with the shipped table's entries for
    every shape key the tuning's table lacks, per structure, placed after the tuning's own keyed entries and before
    its entries for every key. only: {structure: shape keys (text)} to take, every key where None. Returns (tuning,
    {structure: shape keys taken})."""
    tuning = kbuild.check_tuning(manifest, tuning) if tuning else kbuild.default_tuning(manifest)
    table = kbuild.check_tuning(manifest, shipped)
    taken = {}
    structures = {}
    for sid, t in tuning["structures"].items():
        own = keys(tuning, sid)
        wanted = None if only is None else set(only.get(sid) or ())
        source = table["structures"].get(sid) or {"configs": [], "table": []}
        configs, entries, added = [dict(c) for c in t["configs"]], [], set()
        for e in source["table"]:
            k = _key_text(e["key"])
            if k is None or k in own or (wanted is not None and k not in wanted):
                continue
            config = source["configs"][e["config"]]
            if config not in configs:
                configs.append(dict(config))
            entries.append(dict(e, config=configs.index(config)))
            added.add(k)
        if added:
            keyed = [e for e in t["table"] if e["key"] is not None]
            rest = [e for e in t["table"] if e["key"] is None]
            structures[sid] = {"configs": configs, "table": keyed + entries + rest}
            taken[sid] = added
        else:
            structures[sid] = t
    return dict(tuning, structures=structures), taken


def apply(kernel_id, manifest, tuning, root=None, capability=None):
    """fill() with the table the kernel carries for this GPU, where it carries one: (tuning, table path or None,
    {structure: shape keys taken}). The tuning comes back checked, or the default tuning for None."""
    shipped, where = table(kernel_id, capability, root)
    if shipped is None:
        return (kbuild.check_tuning(manifest, tuning) if tuning else kbuild.default_tuning(manifest)), None, {}
    filled, taken = fill(manifest, tuning, shipped)
    return filled, where, taken


def report_rows(shipped, structure, spec, wanted):
    """The shipped table's report rows for a structure's shape keys `wanted` (text), each marked shipped, with the
    name of the structure's yardstick library (kernel.json's, which the table does not repeat a row)."""
    t = ((shipped or {}).get("structures") or {}).get(structure) or {}
    library = (spec.get("yardstick") or {}).get("library")
    return [dict(r, library=library, shipped=True) for r in t.get("report") or []
            if shape_key(spec, r["shape"]) in wanted]


# merge(): a sweep's tuning into a kernel's table.

def _provenance(tuning, date):
    missing = [f for f in ("source", "device", "capability") if not tuning.get(f)]
    if missing:
        raise ValueError(f"the tuning names no {', '.join(missing)}: a shipped table takes a sweep's tuning (crit kernels "
                         "sweep writes them), not one written by hand")
    when = tuning.get("date") or date
    if not when:
        raise ValueError("the tuning names no date: give --date (the day it was swept)")
    datetime.date.fromisoformat(str(when))
    return {"source": tuning["source"], "device": tuning["device"], "capability": str(tuning["capability"]),
            "date": str(when)}


def _by_key(manifest, data, sweeps, spec_report=None):
    """{structure: {key text: {"entries": [(rows, length, knobs)], "sweep": provenance, "report": rows}}} of a tuning
    or table (sweeps: its provenance list, or one provenance for every entry)."""
    data = kbuild.retire(manifest, data)
    out = {}
    for sid, t in (data.get("structures") or {}).items():
        spec = manifest["structures"].get(sid)
        if spec is None:
            continue
        base = kbuild.defaults(manifest, sid)
        configs = [dict(base, **c) for c in t.get("configs") or []]
        if not configs or configs[0] != base:
            configs, shift = [base] + configs, 1
        else:
            shift = 0
        found = out.setdefault(sid, {})
        for e in t.get("table") or []:
            k = _key_text(e.get("key"))
            if k is None:
                continue
            where = sweeps if isinstance(sweeps, dict) else sweeps[e.get("sweep", 0)]
            slot = found.setdefault(k, {"entries": [], "sweep": where, "report": []})
            slot["entries"].append((list(e["rows"]), list(e["length"]) if e.get("length") else None,
                                    configs[e["config"] + shift]))
        rows = (spec_report or {}).get(sid) if spec_report is not None else t.get("report")
        for r in rows or []:
            k = shape_key(spec, r["shape"])
            if k in found:
                found[k]["report"].append({f: v for f, v in r.items() if f not in ("sweep", "shipped", "library")})
    return out


def _newer(new, old, current):
    """Whether an entry swept as `new` replaces one swept as `old`: one on the kernel's current source wins over one on
    an earlier source, else the later date (the same day: the one being merged)."""
    if (new["source"] == current) != (old["source"] == current):
        return new["source"] == current
    return new["date"] >= old["date"]


def _usable(manifest, sid, knobs):
    spec = manifest["structures"][sid].get("knobs", {})
    return all(n in spec and v in spec[n]["values"] for n, v in knobs.items())


def merge(manifest, existing, tuning, report=None, date=None, current=None):
    """The kernel's table for the tuning's capability with a finished sweep's entries merged in: (table, {structure:
    {"added", "replaced", "kept", "dropped"}}). Per structure and shape key the sweep's entries replace the table's
    where they are newer (_newer), with the sweep report's rows for the key; the table's entries whose knobs no longer
    exist in kernel.json are dropped. existing: the table as it is (None: a new one)."""
    kbuild.check_tuning(manifest, tuning)   # refused as an engine's tuning would be
    made = _provenance(tuning, date)
    if existing is not None and str(existing.get("capability")) != made["capability"]:
        raise ValueError(f"the tuning was swept on compute capability {made['capability']}, the table is for "
                         f"{existing.get('capability')}")
    old = _by_key(manifest, existing, existing.get("sweeps") or []) if existing else {}
    new = _by_key(manifest, tuning, made, (report or {}).get("structures") or {})
    for sid, taken in (tuning.get("shipped") or {}).items():   # the sweep copied these from a table: not its own
        for key in taken:
            new.get(sid, {}).pop(_key_text(key), None)
    stats = {}
    merged = {}
    for sid in manifest.get("structures", {}):
        here, there = old.get(sid, {}), new.get(sid, {})
        if not here and not there:
            continue
        counts = {"added": 0, "replaced": 0, "kept": 0, "dropped": 0}
        out = {}
        for k, slot in here.items():
            if all(_usable(manifest, sid, knobs) for _, _, knobs in slot["entries"]):
                out[k] = slot
            else:
                counts["dropped"] += 1
        for k, slot in there.items():
            if k not in out:
                counts["added"] += 1
                out[k] = slot
            elif _newer(slot["sweep"], out[k]["sweep"], current):
                counts["replaced"] += 1
                out[k] = slot
        counts["kept"] = len(out) - counts["added"] - counts["replaced"]
        merged[sid] = out
        stats[sid] = counts
    sweeps = sorted({json.dumps(s["sweep"], sort_keys=True) for slots in merged.values() for s in slots.values()},
                    key=lambda s: (json.loads(s)["date"], s))
    index = {s: i for i, s in enumerate(sweeps)}
    structures = {}
    for sid, slots in merged.items():
        base = kbuild.defaults(manifest, sid)
        configs, table_, rows = [base], [], []
        for k in sorted(slots, key=lambda k: tuple(json.loads(k).values())):
            slot = slots[k]
            i = index[json.dumps(slot["sweep"], sort_keys=True)]
            for span, length, knobs in sorted(slot["entries"], key=lambda e: (e[1] or [0], e[0])):
                if knobs not in configs:
                    configs.append(knobs)
                entry = {"key": json.loads(k), "rows": span, "config": configs.index(knobs), "sweep": i}
                if length:
                    entry["length"] = length
                table_.append(entry)
            rows += [dict(r, sweep=i) for r in slot["report"]]
        structures[sid] = {"configs": configs, "table": table_, "report": rows}
    out = {"schema": kbuild.TUNING_SCHEMA, "kernel": manifest["id"], "capability": made["capability"],
           "sweeps": [json.loads(s) for s in sweeps], "structures": structures}
    kbuild.check_tuning(manifest, out)
    return out, stats


def ship(kernel_id, tuning_path, report_path=None, date=None, root=None):
    """Merge the sweep's tuning at tuning_path (its report beside it, <id>.report.json, unless report_path names
    another) into the table the kernel's folder carries for its capability, and write it. Returns {"table", "stats"}."""
    folder, manifest = kbuild.find(kernel_id, root)
    tuning_path = Path(tuning_path)
    tuning = kbuild.read_tuning(tuning_path)
    if tuning.get("kernel") != kernel_id:
        raise ValueError(f"{tuning_path} is a tuning of {tuning.get('kernel')!r}, not {kernel_id!r}")
    report_path = Path(report_path) if report_path else tuning_path.with_name(f"{kernel_id}.report.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    if not tuning.get("date") and not date:
        date = datetime.date.fromtimestamp(tuning_path.stat().st_mtime).isoformat()
    if not tuning.get("capability"):
        raise ValueError(f"{tuning_path} names no capability: a shipped table takes a sweep's tuning")
    target = path(folder, tuning["capability"])
    existing = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else None
    table_, stats = merge(manifest, existing, tuning, report, date, kbuild.source_digest(kernel_id, root))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(table_, indent=1) + "\n", encoding="utf-8", newline="\n")
    _cache.pop(str(target), None)
    return {"table": str(target), "stats": stats}


__all__ = ["FOLDER", "apply", "fill", "keys", "merge", "path", "report_rows", "shape_key", "ship", "table"]
