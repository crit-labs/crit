"""Bounded local command adapter; Windows children are owned before execution."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import uuid

from crit.run.records import keys, read_json, write_json

from .ownership import group_alive, process_identity


class ProcessTree:
    def __init__(self, command, cwd, log, job_name=None, env=None):
        self.process = None
        self.closed = False
        self.identity = None
        self.job = None
        self.job_name = job_name
        self.guardian = None
        if os.name == "nt":
            self._windows_start(command, cwd, log, env)
        else:
            self.process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True, env=env)
            self.identity = process_identity(self.process.pid)

    def _windows_start(self, command, cwd, log, env):
        import ctypes as c
        from ctypes import wintypes as w

        class Limits(c.Structure):
            _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64), ("flags", w.DWORD),
                        ("min_working_set", c.c_size_t), ("max_working_set", c.c_size_t),
                        ("active_limit", w.DWORD), ("affinity", c.c_size_t),
                        ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class Extended(c.Structure):
            _fields_ = [("basic", Limits), ("io", c.c_uint64 * 6),
                        ("process_memory", c.c_size_t), ("job_memory", c.c_size_t),
                        ("peak_process", c.c_size_t), ("peak_job", c.c_size_t)]

        class Thread(c.Structure):
            _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("id", w.DWORD),
                        ("owner", w.DWORD), ("base_priority", w.LONG),
                        ("delta_priority", w.LONG), ("flags", w.DWORD)]

        api = c.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([c.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "OpenProcess": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "CreateToolhelp32Snapshot": ([w.DWORD, w.DWORD], w.HANDLE),
            "Thread32First": ([w.HANDLE, c.POINTER(Thread)], w.BOOL),
            "Thread32Next": ([w.HANDLE, c.POINTER(Thread)], w.BOOL),
            "OpenThread": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "ResumeThread": ([w.HANDLE], w.DWORD), "CloseHandle": ([w.HANDLE], w.BOOL)}
        signatures.update({"TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "QueryInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p], w.BOOL)})
        for name, (arguments, result) in signatures.items():
            function = getattr(api, name)
            function.argtypes, function.restype = arguments, result
        self.api = api
        self.job = api.CreateJobObjectW(None, self.job_name)
        if not self.job:
            raise c.WinError(c.get_last_error())
        try:
            limits = Extended()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not api.SetInformationJobObject(self.job, 9, c.byref(limits), c.sizeof(limits)):
                raise c.WinError(c.get_last_error())
            # CREATE_SUSPENDED prevents a launcher spawning children before assignment.
            self.process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, creationflags=0x4 | 0x08000000, env=env)
            handle = api.OpenProcess(0x0100 | 0x0001, False, self.process.pid)
            if not handle:
                raise c.WinError(c.get_last_error())
            try:
                if not api.AssignProcessToJobObject(self.job, handle):
                    raise c.WinError(c.get_last_error())
            finally:
                api.CloseHandle(handle)
            snapshot = api.CreateToolhelp32Snapshot(0x4, 0)
            if snapshot == c.c_void_p(-1).value:
                raise c.WinError(c.get_last_error())
            resumed = False
            try:
                entry = Thread()
                entry.size = c.sizeof(entry)
                available = api.Thread32First(snapshot, c.byref(entry))
                while available:
                    if entry.owner == self.process.pid:
                        thread = api.OpenThread(0x0002, False, entry.id)
                        if not thread:
                            raise c.WinError(c.get_last_error())
                        try:
                            if api.ResumeThread(thread) == 0xFFFFFFFF:
                                raise c.WinError(c.get_last_error())
                            resumed = True
                        finally:
                            api.CloseHandle(thread)
                        break
                    available = api.Thread32Next(snapshot, c.byref(entry))
            finally:
                api.CloseHandle(snapshot)
            if not resumed:
                raise OSError("Could not resume the owned runner thread")
        except BaseException:
            self.close()
            raise

    def identify_guardian(self, pid, identity):
        """Verify the handshake names a live member of this owned process tree."""
        if (type(pid) is not int or pid <= 0 or not isinstance(identity, str) or
                identity == "unknown" or process_identity(pid) != identity):
            raise ValueError("Launch guardian has no matching live process identity")
        if os.name == "nt":
            import ctypes as c
            from ctypes import wintypes as w
            if not self.job or self.closed:
                raise ValueError("Launch guardian has no owned Windows job")
            self.api.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)]
            self.api.IsProcessInJob.restype = w.BOOL
            handle = self.api.OpenProcess(0x1000, False, pid)
            if not handle:
                raise c.WinError(c.get_last_error())
            try:
                member = w.BOOL()
                if not self.api.IsProcessInJob(handle, self.job, c.byref(member)):
                    raise c.WinError(c.get_last_error())
                if not member.value:
                    raise ValueError("Launch guardian is outside the owned Windows job")
            finally:
                self.api.CloseHandle(handle)
        elif (pid != self.process.pid or identity != self.identity or
              os.getpgid(pid) != self.process.pid):
            # POSIX Python execs in place; the guardian is also the group leader
            # whose identity is retained for crash recovery.
            raise ValueError("Launch guardian is outside the owned process group")
        if process_identity(pid) != identity:
            raise ValueError("Launch guardian identity changed during verification")
        self.guardian = {"pid": pid, "identity": identity}

    def close(self):
        if self.closed:
            return
        if os.name == "nt" and self.job:
            import ctypes as c
            from ctypes import wintypes as w
            class Accounting(c.Structure):
                _fields_ = [("user", c.c_int64), ("kernel", c.c_int64), ("period_user", c.c_int64),
                            ("period_kernel", c.c_int64), ("faults", w.DWORD), ("total", w.DWORD),
                            ("active", w.DWORD), ("terminated", w.DWORD)]
            end = time.monotonic() + 5
            # Job accounting can reach zero before asynchronous termination has
            # settled a descendant's I/O. Retain process handles before killing
            # the job, then wait for those actual processes as well as the job.
            self.api.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
            self.api.WaitForSingleObject.restype = w.DWORD
            self.api.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)]
            self.api.IsProcessInJob.restype = w.BOOL
            capacity, handles = 16, []
            try:
                while True:
                    class Members(c.Structure):
                        _fields_ = [("assigned", w.DWORD), ("count", w.DWORD),
                                    ("ids", c.c_size_t * capacity)]
                    members = Members()
                    if self.api.QueryInformationJobObject(self.job, 3, c.byref(members), c.sizeof(members), None):
                        break
                    if c.get_last_error() != 234:  # ERROR_MORE_DATA
                        raise c.WinError(c.get_last_error())
                    if time.monotonic() >= end:
                        raise subprocess.TimeoutExpired("Windows job member snapshot", 5)
                    capacity = max(capacity * 2, members.assigned)
                for pid in members.ids[:members.count]:
                    handle = self.api.OpenProcess(0x00100000 | 0x1000, False, pid)
                    if not handle:
                        error = c.get_last_error()
                        if process_identity(pid) is None:
                            continue  # It exited between the snapshot and open.
                        raise c.WinError(error)
                    handles.append(handle)
                    member = w.BOOL()
                    if not self.api.IsProcessInJob(handle, self.job, c.byref(member)):
                        raise c.WinError(c.get_last_error())
                    if not member.value:  # A recycled PID is not our process.
                        self.api.CloseHandle(handles.pop())
                if not self.api.TerminateJobObject(self.job, 1):
                    raise c.WinError(c.get_last_error())
                while True:
                    value = Accounting()
                    if not self.api.QueryInformationJobObject(self.job, 1, c.byref(value), c.sizeof(value), None):
                        raise c.WinError(c.get_last_error())
                    if not value.active:
                        break
                    if time.monotonic() >= end:
                        raise subprocess.TimeoutExpired("Windows job retirement", 5)
                    time.sleep(.02)
                for handle in handles:
                    remaining = max(0, int((end - time.monotonic()) * 1000))
                    status = self.api.WaitForSingleObject(handle, remaining)
                    if status == 258:  # WAIT_TIMEOUT
                        raise subprocess.TimeoutExpired("Windows process retirement", 5)
                    if status != 0:  # WAIT_OBJECT_0
                        raise c.WinError(c.get_last_error())
            finally:
                for handle in handles:
                    self.api.CloseHandle(handle)
            if not self.api.CloseHandle(self.job):
                raise c.WinError(c.get_last_error())
            self.job = None
        elif os.name != "nt" and self.process:
            current = process_identity(self.process.pid)
            if current is not None and current != self.identity:
                raise OSError("Process birth identity changed; refusing ambiguous tree cleanup")
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.process:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=5)
            if os.name == "nt":
                # Popen otherwise retains its kernel handle until garbage
                # collection. Release this owner at the explicit close boundary;
                # wait()/poll() still use the cached return code afterwards.
                self.process._handle.Close()
            if os.name != "nt":
                end = time.monotonic() + 5
                while group_alive(self.process.pid) and time.monotonic() < end:
                    time.sleep(.02)
                if group_alive(self.process.pid):
                    raise subprocess.TimeoutExpired("Process group retirement", 5)
        self.closed = True


def guarded_tree(command, cwd, log, ownership, folder):
    folder.mkdir(parents=True, exist_ok=True)
    launch = folder / ("launch-" + uuid.uuid4().hex + ".json")
    write_json(launch, {"owner_path": str(ownership.path), "token": ownership.token,
               "command": command, "cwd": str(cwd)}, immutable=True)
    interpreter, environment = sys.executable, None
    if os.name == "nt" and getattr(sys, "_base_executable", sys.executable) != sys.executable:
        # A venv's python.exe is a launcher that starts the base interpreter, and Windows activates a Microsoft
        # Store base outside the launcher's job. Start the base directly, naming the venv as the launcher does.
        interpreter, environment = sys._base_executable, dict(os.environ, __PYVENV_LAUNCHER__=sys.executable)
    tree = ProcessTree([interpreter, "-B", str(Path(__file__).with_name("guardian.py")), str(launch)], cwd, log,
                       job_name="Local\\crit-runtime-" + ownership.token, env=environment)
    tree.launch_error_path = launch.with_suffix(".error.json")
    try:
        ready_path = launch.with_suffix(".ready.json")
        deadline = time.monotonic() + 10
        while not ready_path.exists():
            if tree.process.poll() is not None:
                raise ValueError("Launch guardian exited before reporting its identity")
            if time.monotonic() >= deadline:
                raise ValueError("Launch guardian did not report its identity before the deadline")
            time.sleep(.01)
        ready = read_json(ready_path)
        keys(ready, {"token", "pid", "identity"})
        if ready["token"] != ownership.token:
            raise ValueError("Launch guardian ownership token changed")
        tree.identify_guardian(ready["pid"], ready["identity"])
        ownership.bind(tree)
    except BaseException:
        try:
            tree.close()
        except BaseException:
            ownership.failed_tree = tree
            ownership.block("Launch ownership could not be published or settled")
        raise
    return tree
