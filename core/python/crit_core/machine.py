"""One engine-loading process at a time on a machine.

crit eval (run, pin, ab, kernels, reference), crit run measure and crit serve each load a model: often most of the
GPU and, for models whose weights live in host memory, tens of GB of pinned RAM that cannot be swapped. Two of them
side by side can exhaust the host and freeze it. engine_lock() serializes them through an advisory lock file and
says who holds it while it waits.

CRIT_ENGINE_LOCK names the lock file (default: crit-engine.lock in the system temp directory, one per OS instance;
under WSL that is the distribution); "off" disables the lock. A process holding it exports CRIT_ENGINE_LOCK_HELD,
so the processes it starts (an Eval adapter's engine, crit run measure's children) never wait for their parent.

File locks do not cross OS instances, but Windows and its WSL distributions share one GPU and one RAM. The holder
therefore also keeps a lease file every OS instance of the machine can read and refreshes it while it holds the lock;
the others wait while the lease is fresh. CRIT_ENGINE_LEASE names it (default: crit/engine.lease in %ProgramData% on
Windows, /mnt/c/ProgramData/crit/engine.lease under WSL, none elsewhere, and next to the lock file when
CRIT_ENGINE_LOCK names one); "off" disables it.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import json
import tempfile
import threading
import time
import uuid

LOCK_ENV = "CRIT_ENGINE_LOCK"
HELD_ENV = "CRIT_ENGINE_LOCK_HELD"
LEASE_ENV = "CRIT_ENGINE_LEASE"
LEASE_REFRESH_S = 5.0
LEASE_STALE_S = 30.0   # a holder that stopped refreshing (crashed, machine asleep) no longer blocks after this


def lock_path():
    value = os.environ.get(LOCK_ENV)
    if value == "off":
        return None
    return Path(value) if value else Path(tempfile.gettempdir()) / "crit-engine.lock"


def _try_lock(handle):
    if os.name == "nt":
        import msvcrt
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(handle):
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _under_wsl():
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text(encoding="ascii").lower()
    except OSError:
        return False


def lease_path():
    """The lease file every OS instance of this machine reads, or None."""
    value = os.environ.get(LEASE_ENV)
    if value == "off":
        return None
    if value:
        return Path(value)
    lock = os.environ.get(LOCK_ENV)
    if lock and lock != "off":
        return Path(lock + ".lease")
    if os.name == "nt":
        return Path(os.environ.get("ProgramData") or "C:/ProgramData") / "crit" / "engine.lease"
    if _under_wsl() and Path("/mnt/c").is_dir():
        return Path("/mnt/c/ProgramData/crit/engine.lease")
    return None


def _read_lease(path):
    try:
        lease = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return lease if isinstance(lease, dict) and isinstance(lease.get("at"), (int, float)) else None


def _write_lease(path, lease):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{lease['id']}.tmp")
        temporary.write_text(json.dumps(dict(lease, at=time.time())), encoding="utf-8")
        os.replace(temporary, path)
        return True
    except OSError:
        return False


class _Lease:
    """This holder's lease on the machine: taken after the file lock, refreshed by a thread, removed on release."""

    def __init__(self, path, label):
        self.path, self.stop, self.thread = path, threading.Event(), None
        self.lease = {"id": uuid.uuid4().hex, "pid": os.getpid(), "label": label,
                      "system": "wsl" if _under_wsl() else os.name}

    def other(self):
        """The fresh lease another holder keeps, or None."""
        lease = _read_lease(self.path)
        if lease and lease.get("id") != self.lease["id"] and time.time() - lease["at"] < LEASE_STALE_S:
            return lease
        return None

    def take(self):
        """True once this process owns the lease. Two OS instances can write at once, so it reads the lease back
        after a moment; an unwritable location means there is no shared lease to honor."""
        if self.other():
            return False
        if not _write_lease(self.path, self.lease):
            return True
        time.sleep(0.2)
        lease = _read_lease(self.path)
        if lease and lease.get("id") != self.lease["id"]:
            return False
        self.thread = threading.Thread(target=self._refresh, daemon=True)
        self.thread.start()
        return True

    def _refresh(self):
        while not self.stop.wait(LEASE_REFRESH_S):
            _write_lease(self.path, self.lease)

    def release(self):
        self.stop.set()
        if self.thread:
            self.thread.join()
        lease = _read_lease(self.path)
        if lease and lease.get("id") == self.lease["id"]:
            try:
                self.path.unlink()
            except OSError:
                pass


def _describe(lease):
    return f"{lease.get('pid')} {lease.get('label')} ({lease.get('system')})" if lease else "another OS instance"


def holder(path=None):
    """The holder line ("pid label") the current holder wrote, or None."""
    path = path or lock_path()
    if path is None:
        return None
    try:
        return Path(str(path) + ".holder").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


@contextmanager
def engine_lock(label, wait=True, poll_s=2.0, log=None):
    """Hold the machine's engine lock for the block. Waits (printing the holder once) unless wait is False, in which
    case a busy lock raises RuntimeError."""
    path = lock_path()
    if path is None or os.environ.get(HELD_ENV):
        yield
        return
    log = log or (lambda line: print(line, file=sys.stderr, flush=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        if not _try_lock(handle):
            who = holder(path) or "another process"
            if not wait:
                raise RuntimeError(f"another engine is loaded on this machine ({who}); {LOCK_ENV}={path}")
            log(f"waiting for the engine lock held by {who} ({path}); one engine-loading process at a time")
            while not _try_lock(handle):
                time.sleep(poll_s)
        shared = lease_path()
        lease = _Lease(shared, label) if shared else None
        try:
            if lease and not lease.take():
                if not wait:
                    raise RuntimeError(f"another engine is loaded on this machine ({_describe(lease.other())}); "
                                       f"{LEASE_ENV}={shared}")
                log(f"waiting for the engine lease held by {_describe(lease.other())} ({shared}); "
                    "one engine-loading process per machine")
                while not lease.take():
                    time.sleep(poll_s)
        except BaseException:
            _unlock(handle)
            raise
        Path(str(path) + ".holder").write_text(f"{os.getpid()} {label}\n", encoding="utf-8")
        os.environ[HELD_ENV] = "1"
        try:
            yield
        finally:
            os.environ.pop(HELD_ENV, None)
            if lease:
                lease.release()
            try:
                Path(str(path) + ".holder").write_text("", encoding="utf-8")
            except OSError:
                pass
            _unlock(handle)
    finally:
        handle.close()
