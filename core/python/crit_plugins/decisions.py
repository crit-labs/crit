"""Decisions a plugin takes from a measurement: recorded once, replayed after.

A plugin sometimes chooses between two ways to do one thing by timing both on this machine as it loads: who computes
a missed expert, where a prompt chunk's knee lies, how a layer's work splits between the host and the device. Two
loads of the same package then resolve a near-tie differently, and where the two ways do not give the same bits (an
arithmetic that follows the choice, a kernel structure below `exact`) the same request gets another text in another
process. A single-process repeat check cannot see it. So nothing an engine computes may depend on how fast something
ran in this process: a measured choice is a decision, settled through this module.

    from crit_plugins import decisions
    misses = decisions.settle(
        "my-plugin", "misses",                                   # the plugin and the decision's name
        key={"device": name, "hidden": 4096, "format": "nvfp4"},  # what the choice depends on besides the timing
        measure=lambda: {"cpu_us": 131.0, "link_us": 188.0},     # runs only when no record answers
        decide=lambda found: "cpu" if found["cpu_us"] < 0.8 * found["link_us"] else "link")

settle() returns the recorded choice where the engine's record holds this decision with the same key, and neither
measures nor decides again. Otherwise it measures, decides and keeps the result for the report. The record is a JSON
file the package carries: its configuration's "decisions" names it inside the package ("decisions":
"decisions.json"), written by `crit plugins decisions PACKAGE --model MODEL` from one load of the engine, and Core's
loader applies it before any plugin is made, so the package Eval measured decides as the package served does.
Without it CRIT_DECISIONS names a file for a quick try, and without either every load measures, as before; the
report then says so.

The engine's telemetry carries the report under "decisions": every decision with its key, its choice, where the
choice came from ("recorded" or "measured"), the measurement, and one digest over all the choices. Two processes
whose texts part show different digests when a decision is why.

A choice is JSON (a string, a number, a list or an object of those), and so is a key. A decision that only changes
speed and never a bit (a spin time, a thread count) needs none of this.
"""
import hashlib
import json
import os
import threading
from pathlib import Path

VERSION = 1
SOURCES = ("recorded", "measured")

_lock = threading.Lock()
_record = None     # the file use_record() named
_recorded = {}     # (plugin, name) -> {"key", "choice", "measurement"} read from the record
_read = None       # the file _recorded was read from
_settled = {}      # (plugin, name) -> the decision as report() gives it


class DecisionError(ValueError):
    pass


def _plain(value, what):
    """value through JSON, so a recorded value and a fresh one compare equal: tuples as lists, keys as text."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise DecisionError(f"{what} is JSON (text, numbers, lists and objects of them): {error}") from None


def use_record(path):
    """Settle decisions from the record at path (a package's decisions file). None: CRIT_DECISIONS, else no record.
    Forgets what the process settled before, as a new engine's load starts."""
    global _record, _read
    with _lock:
        _record = None if path is None else Path(path)
        _read = None
        _recorded.clear()
        _settled.clear()


def record_file():
    """The record this process settles from, or None."""
    if _record is not None:
        return _record
    return Path(os.environ["CRIT_DECISIONS"]) if os.environ.get("CRIT_DECISIONS") else None


def read(path):
    """A record file's decisions as {(plugin, name): {"key", "choice", "measurement"}}; DecisionError for a file
    that is not one."""
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DecisionError(f"{path} is no decisions record: {error}") from None
    rows = value.get("decisions") if isinstance(value, dict) else None
    if not isinstance(value, dict) or value.get("version") != VERSION or not isinstance(rows, list):
        raise DecisionError(f"{path} is no decisions record of version {VERSION} (crit plugins decisions writes one)")
    found = {}
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k] for k in ("plugin", "name")) \
                or "choice" not in row or not isinstance(row.get("key"), dict):
            raise DecisionError(f"{path}: a decision is {{plugin, name, key, choice}}, not {row!r}")
        found[(row["plugin"], row["name"])] = {"key": row["key"], "choice": row["choice"],
                                               "measurement": row.get("measurement")}
    return found


def _load():
    global _read
    path = record_file()
    if path == _read:
        return
    _recorded.clear()
    if path is not None and path.is_file():
        _recorded.update(read(path))
    _read = path


def settle(plugin, name, *, key, measure, decide):
    """The choice of decision `name` of `plugin`: the record's where it holds this decision with an equal key, else
    decide(measure()). key: a dict of everything the choice depends on besides the measurement (the device, the
    shapes, the format, the options that bound it); a record with another key is not this decision and is measured
    again. The second call of a process returns the first's choice without measuring."""
    if not isinstance(key, dict):
        raise DecisionError(f"{plugin}.{name}: key is a dict of what the choice depends on")
    key = _plain(key, f"{plugin}.{name}'s key")
    with _lock:
        _load()
        known = _settled.get((plugin, name))
        if known is not None and known["key"] == key:
            return known["choice"]
        recorded = _recorded.get((plugin, name))
    if recorded is not None and recorded["key"] == key:
        row = {"plugin": plugin, "name": name, "key": key, "choice": recorded["choice"], "source": "recorded",
               "measurement": recorded["measurement"]}
    else:
        measurement = _plain(measure(), f"{plugin}.{name}'s measurement")
        row = {"plugin": plugin, "name": name, "key": key,
               "choice": _plain(decide(measurement), f"{plugin}.{name}'s choice"), "source": "measured",
               "measurement": measurement}
        if recorded is not None:
            row["stale"] = "the record holds this decision for another key"
    with _lock:
        _settled[(plugin, name)] = row
    return row["choice"]


def settled(plugin, name):
    """The decision `name` of `plugin` as this process settled it ({key, choice, source, measurement}), or None."""
    with _lock:
        row = _settled.get((plugin, name))
    return None if row is None else dict(row)


def digest(rows):
    """One hash over the choices of rows (each decision's plugin, name, key and choice, in order of plugin and name):
    equal in two processes that decided alike."""
    text = json.dumps([[r["plugin"], r["name"], r["key"], r["choice"]]
                       for r in sorted(rows, key=lambda r: (r["plugin"], r["name"]))], sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def report():
    """{"record": the file or None, "decisions": [each settled decision], "digest", "measured": how many of them this
    process measured}: what the engine's telemetry carries. measured above 0 means another load may decide
    otherwise."""
    with _lock:
        rows = [dict(row) for _, row in sorted(_settled.items())]
    path = record_file()
    return {"record": None if path is None else str(path), "decisions": rows, "digest": digest(rows),
            "measured": sum(row["source"] == "measured" for row in rows)}


def write(path, rows=None):
    """Write the settled decisions (or rows) as a record at path, and return how many it holds."""
    rows = report()["decisions"] if rows is None else rows
    kept = [{k: row[k] for k in ("plugin", "name", "key", "choice", "measurement") if k in row} for row in rows]
    text = json.dumps({"version": VERSION, "decisions": kept}, indent=2, sort_keys=True) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    return len(kept)
