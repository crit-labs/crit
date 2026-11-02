"""The single-writer lease over a folder that crit run's coordinator (each run) and crit serve's runtime ownership take:
a kernel lock on a guard file that is never unlinked, and an owner lease file, created exclusively, that names the
writer's PID, host and OS, so writers on Windows and in WSL exclude each other too. A lease whose owner exited stays
until an explicit recovery on the owner's host and OS proves it gone."""
from contextlib import contextmanager
import json
import os
import platform
import socket

from crit_core.machine import try_lock, unlock

from crit.process.identity import process_identity
from crit.records import keys, now, number, safe_path


def lease_owner():
    """Identity written into the lease: a PID is only meaningful on its own host and OS."""
    return {"pid": os.getpid(), "host": socket.gethostname(), "system": platform.system(), "started_at": now()}


def read_lease_owner(path):
    """Lease owner from the lock file; a legacy lock holds only the PID."""
    raw = path.read_text(encoding="utf-8").strip()
    if raw.isdigit():
        return {"pid": int(raw), "host": None, "system": None, "started_at": None}
    value = json.loads(raw)
    keys(value, {"pid", "host", "system", "started_at"})
    number(value["pid"], "lease owner PID", 1, 2 ** 32, integer=True)
    return value


@contextmanager
def lease(root, *, recover=False):
    """Pin ownership while checking/removing a dead lease and throughout every write."""
    # Never unlink this guard: replacing its inode would split the kernel lock.
    # The separate owner lease still excludes writers across Windows/WSL.
    with safe_path(root, ".coordinator.guard").open("a+b") as guard:
        if not try_lock(guard):
            raise ValueError("Run coordinator is busy; another writer or recoverer holds its guard")
        try:
            if recover:
                release_dead_lease(root)
            with exclusive_lease(root):
                yield
        finally:
            unlock(guard)


@contextmanager
def exclusive_lease(root):
    """An abandoned exclusive lease blocks writers until explicit dead-owner recovery."""
    path = safe_path(root, ".coordinator.lock")
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            owner = read_lease_owner(path)
            held = f" (held by PID {owner['pid']}" + (f" on {owner['host']}, {owner['system']}, since {owner['started_at']})" if owner["host"] else ")")
        except (OSError, ValueError):
            held = ""
        raise ValueError(f"Run coordinator is busy{held}; if that process has exited, run recover on the same host and OS") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(lease_owner()))
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        path.unlink()


def release_dead_lease(root):
    """Called only while holding the persistent coordinator guard: remove the lease of an owner that has exited."""
    lock_path = safe_path(root, ".coordinator.lock")
    if not lock_path.exists():
        return
    owner = read_lease_owner(lock_path)
    if owner["host"] is not None and (owner["host"], owner["system"]) != (socket.gethostname(), platform.system()):
        # Another OS (for example WSL versus Windows) has its own PID space:
        # a missing PID here proves nothing about the owner.
        raise ValueError(f"Lease owner ran on {owner['host']} ({owner['system']}); recover from that host and OS")
    # An exited process can still be opened while another process retains its handle; its identity is None only
    # once it has terminated. "unknown" (no permission, no /proc) proves nothing.
    found = process_identity(owner["pid"])
    if found == "unknown":
        raise ValueError("Cannot prove coordinator owner exited")
    if found is not None:
        raise ValueError("Coordinator owner is still alive; recovery blocked")
    lock_path.unlink()
