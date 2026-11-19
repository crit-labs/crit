"""A machine-wide bound on the files plugins keep in their caches (crit_core.engine.LoadContext.cache).

Each crit run gives its engines a cache folder of its own, and a run that is never stopped (an agent killed, a build
abandoned) keeps its files there for good: an expert store is tens of gigabytes, and a few of them fill a disk. Every
file a plugin keeps is therefore entered in one ledger for the machine (the OS instance: under WSL, the distribution),
whatever run or clone it belongs to. admit() makes room before a new file is written: it removes the least recently
used files of the ledger until those kept stay under a cap and the disk keeps a floor of free space after the write,
and refuses when it cannot; the plugin then builds in memory, as it would without a cache.

CRIT_PLUGIN_CACHE_GIB caps the bytes kept (default 100), CRIT_DISK_FLOOR_GIB is the free space every write leaves
(default 50), and CRIT_PLUGIN_CACHE_LEDGER names the ledger (default ~/.cache/crit/plugin-cache.json).
"""
import json
import os
from pathlib import Path
import shutil
import time

from crit_core.machine import try_lock, unlock

GIB = 1 << 30
CAP_ENV = "CRIT_PLUGIN_CACHE_GIB"
FLOOR_ENV = "CRIT_DISK_FLOOR_GIB"
LEDGER_ENV = "CRIT_PLUGIN_CACHE_LEDGER"
DEFAULT_CAP_GIB = 100.0
DEFAULT_FLOOR_GIB = 50.0
PENDING_S = 6 * 3600   # how long an admitted file that is not written yet keeps its room


def ledger_path():
    value = os.environ.get(LEDGER_ENV)
    from crit_core import cache_home
    return Path(value) if value else cache_home("plugin-cache.json")


def _gib(name, default):
    try:
        return float(os.environ.get(name) or default) * GIB
    except ValueError:
        return default * GIB


def cap_bytes():
    return _gib(CAP_ENV, DEFAULT_CAP_GIB)


def floor_bytes():
    return _gib(FLOOR_ENV, DEFAULT_FLOOR_GIB)


class _Ledger:
    """The ledger, read and written under its lock file: {path: {bytes, used, files}}, the entries whose file is
    gone dropped on reading."""

    def __enter__(self):
        path = ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.handle = path, open(str(path) + ".lock", "a+b")
        while not try_lock(self.handle):
            time.sleep(0.05)
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            entries = {}
        # An admitted file is written after admit() returns: its entry counts while it is young, before it exists.
        now = time.time()
        self.entries = {key: value for key, value in entries.items()
                        if Path(key).is_file() or now - float(value.get("used", 0)) < PENDING_S}
        return self

    def __exit__(self, *exc):
        try:
            if exc[0] is None:
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.entries, indent=1, sort_keys=True), encoding="utf-8")
                os.replace(temporary, self.path)
        finally:
            unlock(self.handle)
            self.handle.close()

    def remove(self, key):
        """Delete an entry's files; False where one is held open and cannot go (Windows), and it stays listed."""
        entry = self.entries[key]
        for name in entry.get("files") or [key]:
            try:
                Path(name).unlink(missing_ok=True)
            except OSError:
                return False
        del self.entries[key]
        return True


def admit(path, nbytes, files=None):
    """Make room for a `nbytes` file a plugin is about to keep at `path` (with `files` beside it, all removed together
    when it is evicted) and enter it in the ledger. Returns (True, None), or (False, reason) where the cap or the free
    space floor cannot be kept even with every other kept file removed; nothing is entered then."""
    path, nbytes = str(Path(path).resolve()), int(nbytes)
    folder = Path(path).parent
    cap, floor = cap_bytes(), floor_bytes()
    if nbytes > cap:
        return False, f"a {nbytes / GIB:.1f} GiB file exceeds the plugin cache's {cap / GIB:.0f} GiB cap ({CAP_ENV})"
    with _Ledger() as ledger:
        ledger.entries.pop(path, None)

        def short():
            kept = sum(int(entry["bytes"]) for entry in ledger.entries.values())
            return kept + nbytes > cap or shutil.disk_usage(folder).free - nbytes < floor
        for key in sorted(ledger.entries, key=lambda key: ledger.entries[key]["used"]):
            if not short():
                break
            ledger.remove(key)
        if short():
            free = shutil.disk_usage(folder).free
            return False, (f"{folder} has {free / GIB:.1f} GiB free, and a {nbytes / GIB:.1f} GiB file would leave "
                           f"less than the {floor / GIB:.0f} GiB floor ({FLOOR_ENV})")
        ledger.entries[path] = {"bytes": nbytes, "used": time.time(),
                                "files": [path] + [str(Path(name).resolve()) for name in files or ()]}
    return True, None


def touch(path, files=None):
    """Mark a kept file used now (it is reopened), entering it where the ledger does not list it yet."""
    path = str(Path(path).resolve())
    with _Ledger() as ledger:
        entry = ledger.entries.get(path) or {"bytes": Path(path).stat().st_size,
                                             "files": [path] + [str(Path(name).resolve()) for name in files or ()]}
        entry["used"] = time.time()
        ledger.entries[path] = entry


def forget(path):
    """Drop a file from the ledger and the disk (its owner removed it as stale, or a build did not finish)."""
    path = str(Path(path).resolve())
    with _Ledger() as ledger:
        if path in ledger.entries:
            ledger.remove(path)
