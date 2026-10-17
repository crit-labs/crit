"""Durable per-user runtime ownership; stale/failed cleanup never grants admission."""
import getpass
import hashlib
import os
from pathlib import Path
import signal
import re
import shutil
import tempfile
import time
import uuid

from crit.run.records import keys, now, number, read_json, safe_path, write_json
from crit.run.store import lease


def process_identity(pid):
    if not isinstance(pid, int) or pid <= 0:
        return None
    if os.name == "nt":
        import ctypes as c
        from ctypes import wintypes as w
        api = c.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes, api.OpenProcess.restype = [w.DWORD, w.BOOL, w.DWORD], w.HANDLE
        api.GetProcessTimes.argtypes = [w.HANDLE] + [c.POINTER(w.FILETIME)] * 4
        api.CloseHandle.argtypes = [w.HANDLE]
        api.GetExitCodeProcess.argtypes = [w.HANDLE, c.POINTER(w.DWORD)]
        handle = api.OpenProcess(0x1000, False, pid)
        if not handle:
            return None if c.get_last_error() in {87, 1168} else "unknown"
        try:
            code = w.DWORD()
            if not api.GetExitCodeProcess(handle, c.byref(code)):
                return "unknown"
            if code.value != 259:
                return None
            values = [w.FILETIME() for _ in range(4)]
            if not api.GetProcessTimes(handle, *(c.byref(v) for v in values)):
                return "unknown"
            return str((values[0].dwHighDateTime << 32) | values[0].dwLowDateTime)
        finally:
            api.CloseHandle(handle)
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return boot + ":" + fields[19]
    except FileNotFoundError:
        if Path("/proc").is_dir():
            return None
    except (OSError, IndexError):
        return "unknown"
    try:
        os.kill(pid, 0)
        return "unknown"
    except ProcessLookupError:
        return None
    except PermissionError:
        return "unknown"


def group_alive(pgid):
    if Path("/proc/self/stat").exists():
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                    return True
            except FileNotFoundError:
                continue
            except (OSError, ValueError, IndexError):
                return True  # An unobservable member cannot prove settlement.
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def windows_job_settled(name):
    import ctypes as c
    from ctypes import wintypes as w
    if not name:
        raise ValueError("No retained Windows job identity; cleanup cannot be proven")
    api = c.WinDLL("kernel32", use_last_error=True)
    api.OpenJobObjectW.argtypes, api.OpenJobObjectW.restype = [w.DWORD, w.BOOL, w.LPCWSTR], w.HANDLE
    api.QueryInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p]
    api.CloseHandle.argtypes = [w.HANDLE]
    handle = api.OpenJobObjectW(4, False, name)  # JOB_OBJECT_QUERY
    if not handle:
        if c.get_last_error() == 2:  # Destroyed only after the owned processes exit.
            return True
        raise c.WinError(c.get_last_error())
    try:
        class Accounting(c.Structure):
            _fields_ = [("times", c.c_int64 * 4), ("faults", w.DWORD), ("total", w.DWORD),
                        ("active", w.DWORD), ("terminated", w.DWORD)]
        value = Accounting()
        if not api.QueryInformationJobObject(handle, 1, c.byref(value), c.sizeof(value), None):
            raise c.WinError(c.get_last_error())
        return value.active == 0
    finally:
        api.CloseHandle(handle)


# How long a killed process tree may take to exit before cleanup counts as failed. Exit is not immediate: a process
# that pinned tens of GB of host memory or holds a GPU context releases it after SIGKILL (several seconds here).
RETIREMENT_SECONDS = 60


def default_directory():
    user = str(os.getuid()) if hasattr(os, "getuid") else hashlib.sha256(getpass.getuser().encode()).hexdigest()[:16]
    return Path(os.environ.get("CRIT_RUNTIME_DIRECTORY", str(Path(tempfile.gettempdir()) / ("crit-runtime-" + user))))


def clean_snapshots(directory, runtime_id):
    """Remove a settled owner's private model and provisional-text snapshots."""
    if runtime_id is None:
        return
    if not isinstance(runtime_id, str) or not re.fullmatch(r"[0-9a-f]{32}", runtime_id):
        raise ValueError("Invalid private runtime identity")
    root = safe_path(directory, "runtimes/" + runtime_id + "/models")
    if root.exists():
        for path in root.rglob("*"):
            safe_path(root, path.relative_to(root).as_posix())
            if path.is_file():
                path.chmod(0o600)
        # Job retirement can precede release of a mapped file's Windows handle.
        # Retry only transient sharing/permission errors on this checked private
        # directory; persistent failure must retain runtime ownership.
        deadline = time.monotonic() + .25
        while True:
            try:
                shutil.rmtree(root)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in {5, 32, 33} or time.monotonic() >= deadline:
                    raise
                time.sleep(.005)
    stream = safe_path(directory, "runtimes/" + runtime_id + "/stream.json")
    stream.unlink(missing_ok=True)


class RuntimeOwnership:
    def __init__(self, directory=None):
        self.directory = (Path(directory) if directory is not None else default_directory()).expanduser().absolute()
        self.path = safe_path(self.directory, "owner.json")
        self.token = None
        self.failed_tree = None
        self.blocked = ""

    def acquire(self):
        if self.token:
            value = self.read_owned()
            if self.blocked or value["state"] == "blocked":
                raise ValueError("Runtime cleanup is unresolved. Unload/recover it before another request.")
            return
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        token = uuid.uuid4().hex
        value = {"schema_version": 1, "token": token, "owner_pid": os.getpid(),
                 "owner_identity": process_identity(os.getpid()), "process": None,
                 "state": "reserved", "updated_at": now(), "error": None}
        try:
            write_json(self.path, value, immutable=True)
        except FileExistsError:
            raise ValueError("Another CRIT runtime owns execution or left unresolved cleanup. Unload it or run crit serve --recover-runtime.") from None
        self.token = token

    def read_owned(self):
        # Atomic replacement can briefly deny an open on Windows. Retry only
        # that access failure, within a fixed deadline; never use stale data or
        # infer idle/unowned state from an unreadable ownership record.
        deadline = time.monotonic() + .25
        while True:
            try:
                value = read_json(self.path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.005)
        if not self.token or value.get("token") != self.token:
            raise ValueError("Runtime ownership changed")
        return value

    def update(self, **changes):
        value = self.read_owned()
        write_json(self.path, value | changes | {"updated_at": now()})

    def bind(self, tree):
        if not tree.guardian or process_identity(tree.guardian["pid"]) != tree.guardian["identity"]:
            raise ValueError("Launch guardian identity must be verified before binding")
        self.update(process=tree.guardian | {"job_name": tree.job_name}, state="running")

    def block(self, error):
        if self.token:
            self.blocked = str(error) or "Unresolved runtime cleanup"
            try:
                self.update(state="blocked", error=self.blocked)
            except (ValueError, OSError) as exc:
                # The existing exclusive guard still blocks other clients. Keep
                # local quarantine even if the diagnostic update cannot persist.
                self.blocked += "; could not record cleanup failure: " + str(exc)

    def release(self, settled=False):
        if self.token:
            value = self.read_owned()
            if not settled and (self.blocked or value["state"] == "blocked"):
                raise ValueError("Prove runtime cleanup before releasing ownership")
            self.path.unlink()
            self.token = None
            self.blocked = ""

    def recover(self):
        """Explicit dead-owner recovery; live owners and PID reuse block it."""
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with lease(self.directory):
            if not self.path.exists():
                return {"status": "idle"}
            value = read_json(self.path)
            keys(value, {"schema_version", "token", "owner_pid", "owner_identity", "process", "state", "updated_at", "error"}, {"runtime_id"})
            if value["schema_version"] != 1 or value["state"] not in {"reserved", "running", "blocked"}:
                raise ValueError("Invalid runtime ownership record")
            number(value["owner_pid"], "runtime owner PID", 1, 2 ** 32, integer=True)
            if not isinstance(value["token"], str) or not re.fullmatch(r"[0-9a-f]{32}", value["token"]):
                raise ValueError("Invalid runtime ownership token")
            if value["process"] is not None:
                keys(value["process"], {"pid", "identity"}, {"job_name"})
                number(value["process"]["pid"], "runtime process PID", 1, 2 ** 32, integer=True)
            owner = process_identity(value["owner_pid"])
            if owner is not None and (owner == "unknown" or owner == value["owner_identity"]):
                raise ValueError("The runtime owner is still alive; use its Unload action")
            child = value.get("process")
            if child:
                current = process_identity(child["pid"])
                if current is not None and current != child["identity"]:
                    raise ValueError("Process identity changed; refusing ambiguous cleanup")
                if os.name == "nt":
                    # Windows job ownership closes on owner death. Do not kill an
                    # arbitrary process outside that job or infer that it settled.
                    if current is not None or not windows_job_settled(child.get("job_name")):
                        raise ValueError("Owned Windows job has not finished closing; retry recovery after settlement")
                else:
                    if child["identity"] in {None, "unknown"}:
                        raise ValueError("Process birth identity is unavailable; refusing ambiguous recovery")
                    boot_path = Path("/proc/sys/kernel/random/boot_id")
                    if boot_path.exists() and not child["identity"].startswith(boot_path.read_text().strip() + ":"):
                        raise ValueError("Runtime belongs to a different boot; verify and archive its stale guard explicitly")
                    if current is None and group_alive(child["pid"]):
                        raise ValueError("Live group has no matching birth-identified guardian; refusing ambiguous recovery")
                    if group_alive(child["pid"]):
                        os.killpg(child["pid"], signal.SIGKILL)
                    end = time.monotonic() + RETIREMENT_SECONDS
                    while group_alive(child["pid"]) and time.monotonic() < end:
                        time.sleep(.02)
                    if group_alive(child["pid"]):
                        raise ValueError("Owned process group has not settled")
            clean_snapshots(self.directory, value.get("runtime_id"))
            # Before bind, the guardian cannot launch engine code. After bind,
            # recovery has checked the recorded private process group/job.
            recovery = value | {"recovered_at": now(), "recovery": "dead owner; owned process settlement verified"}
            write_json(self.directory / ("recovery-" + uuid.uuid4().hex + ".json"), recovery, immutable=True)
            self.path.unlink()
            return {"status": "recovered"}
