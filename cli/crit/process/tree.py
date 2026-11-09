"""Bounded local command adapter; Windows children are owned before execution."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import uuid

from crit.records import keys, read_json, write_json

from .ownership import RETIREMENT_SECONDS, group_alive, process_identity

if os.name == "nt":
    import _winapi


class OwnedProcess:
    """What a tree's callers use of subprocess.Popen (pid, poll, wait, kill, returncode, the _handle it closes), for
    the Windows process a tree creates itself (ProcessTree._windows_start), with Popen's semantics."""

    def __init__(self, handle, pid, args):
        self._handle, self.pid, self.args, self.returncode = subprocess.Handle(handle), pid, args, None

    def poll(self):
        return self._wait(0)

    def wait(self, timeout=None):
        code = self._wait(_winapi.INFINITE if timeout is None else int(timeout * 1000))
        if code is None:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return code

    def _wait(self, milliseconds):
        if self.returncode is None:
            if _winapi.WaitForSingleObject(self._handle, milliseconds) == _winapi.WAIT_TIMEOUT:
                return None
            self.returncode = _winapi.GetExitCodeProcess(self._handle)
        return self.returncode

    def kill(self):
        if self.returncode is not None:
            return
        try:
            _winapi.TerminateProcess(self._handle, 1)
        except PermissionError:  # denied once the process has exited
            code = _winapi.GetExitCodeProcess(self._handle)
            if code == _winapi.STILL_ACTIVE:
                raise
            self.returncode = code


class ProcessTree:
    def __init__(self, command, cwd, log, job_name=None, env=None, errors=None):
        """LOG takes the tree's output, and its errors too unless ERRORS names another file."""
        self.process = None
        self.closed = False
        self.identity = None
        self.job = None
        self.job_name = job_name
        self.guardian = None
        if os.name == "nt":
            if command and command[0] == sys.executable and getattr(sys, "_base_executable", sys.executable) != \
                    sys.executable:
                # A venv's python.exe is a launcher that starts the base interpreter, which a Microsoft Store Python
                # activates under Windows' default policy, so the base's own children would leave the job. Start the
                # base directly, naming the venv as the launcher does.
                command = [sys._base_executable, *command[1:]]
                env = dict(os.environ if env is None else env, __PYVENV_LAUNCHER__=sys.executable)
            self._windows_start(command, cwd, log, env, errors)
        else:
            self.process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=errors or subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True, env=env)
            self.identity = process_identity(self.process.pid)

    def _windows_start(self, command, cwd, log, env, errors=None):
        """Create the command's process suspended, assign it to the tree's job, then let it run, so it starts no child
        outside the job. It is created with a desktop app policy that keeps its children in the job too: a Microsoft
        Store Python is a packaged app, and Windows starts the children of a packaged process, and a packaged
        interpreter that a venv's launcher starts, outside the job of the process that starts them unless that
        process's policy says otherwise. Without the policy the job held only the tree's first process there, and
        closing the tree neither ended nor waited for the others, which kept running with the tree's log files open.
        subprocess.Popen cannot pass the policy, so the tree creates its process itself."""
        import ctypes as c
        from ctypes import wintypes as w
        import msvcrt

        class Limits(c.Structure):
            _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64), ("flags", w.DWORD),
                        ("min_working_set", c.c_size_t), ("max_working_set", c.c_size_t),
                        ("active_limit", w.DWORD), ("affinity", c.c_size_t),
                        ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class Extended(c.Structure):
            _fields_ = [("basic", Limits), ("io", c.c_uint64 * 6),
                        ("process_memory", c.c_size_t), ("job_memory", c.c_size_t),
                        ("peak_process", c.c_size_t), ("peak_job", c.c_size_t)]

        class Startup(c.Structure):  # STARTUPINFOEXW
            _fields_ = [("size", w.DWORD), ("reserved", w.LPWSTR), ("desktop", w.LPWSTR), ("title", w.LPWSTR),
                        ("x", w.DWORD), ("y", w.DWORD), ("width", w.DWORD), ("height", w.DWORD),
                        ("columns", w.DWORD), ("rows", w.DWORD), ("fill", w.DWORD), ("flags", w.DWORD),
                        ("show", w.WORD), ("reserved_size", w.WORD), ("reserved_data", c.c_void_p),
                        ("stdin", w.HANDLE), ("stdout", w.HANDLE), ("stderr", w.HANDLE), ("attributes", c.c_void_p)]

        class Created(c.Structure):  # PROCESS_INFORMATION
            _fields_ = [("process", w.HANDLE), ("thread", w.HANDLE), ("pid", w.DWORD), ("tid", w.DWORD)]

        api = c.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([c.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "OpenProcess": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "InitializeProcThreadAttributeList": ([c.c_void_p, w.DWORD, w.DWORD, c.POINTER(c.c_size_t)], w.BOOL),
            "UpdateProcThreadAttribute": ([c.c_void_p, w.DWORD, c.c_size_t, c.c_void_p, c.c_size_t, c.c_void_p,
                                           c.c_void_p], w.BOOL),
            "DeleteProcThreadAttributeList": ([c.c_void_p], None),
            "CreateProcessW": ([w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p, w.BOOL, w.DWORD, c.c_void_p,
                                w.LPCWSTR, c.c_void_p, c.c_void_p], w.BOOL),
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
            this = _winapi.GetCurrentProcess()

            def inheritable(file):
                return _winapi.DuplicateHandle(this, msvcrt.get_osfhandle(file.fileno()), this, 0, True,
                                               _winapi.DUPLICATE_SAME_ACCESS)

            size = c.c_size_t()
            api.InitializeProcThreadAttributeList(None, 2, 0, c.byref(size))
            attributes = c.create_string_buffer(size.value)
            if not api.InitializeProcThreadAttributeList(attributes, 2, 0, c.byref(size)):
                raise c.WinError(c.get_last_error())
            standard = []  # input, output, errors
            try:
                with open(os.devnull, "rb") as devnull:
                    standard += [inheritable(devnull), inheritable(log)]
                standard.append(inheritable(errors) if errors else standard[1])
                # The process inherits its standard handles and no others (PROC_THREAD_ATTRIBUTE_HANDLE_LIST), and
                # takes the policy (PROC_THREAD_ATTRIBUTE_DESKTOP_APP_POLICY,
                # PROCESS_CREATION_DESKTOP_APP_BREAKAWAY_DISABLE_PROCESS_TREE), which Windows before 10 1703 refuses:
                # it has no packaged apps.
                inherited = (w.HANDLE * len(set(standard)))(*set(standard))
                if not api.UpdateProcThreadAttribute(attributes, 0, 0x20002, inherited, c.sizeof(inherited), None,
                                                     None):
                    raise c.WinError(c.get_last_error())
                policy = w.DWORD(0x2)
                api.UpdateProcThreadAttribute(attributes, 0, 0x20012, c.byref(policy), c.sizeof(policy), None, None)
                startup = Startup(size=c.sizeof(Startup), flags=0x100, stdin=standard[0], stdout=standard[1],
                                  stderr=standard[2], attributes=c.addressof(attributes))  # STARTF_USESTDHANDLES
                environment = None
                if env is not None:
                    variables = sorted(env.items(), key=lambda item: item[0].upper())
                    environment = c.create_unicode_buffer("".join(f"{k}={v}\0" for k, v in variables) + "\0")
                created = Created()
                # CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW
                if not api.CreateProcessW(None, c.create_unicode_buffer(subprocess.list2cmdline(command)), None,
                                          None, True, 0x4 | 0x400 | 0x80000 | 0x08000000, environment,
                                          None if cwd is None else os.fspath(cwd), c.byref(startup),
                                          c.byref(created)):
                    raise c.WinError(c.get_last_error())
            finally:
                api.DeleteProcThreadAttributeList(attributes)
                for handle in set(standard):
                    _winapi.CloseHandle(handle)
            self.process = OwnedProcess(created.process, created.pid, command)
            try:
                if not api.AssignProcessToJobObject(self.job, created.process):
                    raise c.WinError(c.get_last_error())
                if api.ResumeThread(created.thread) == 0xFFFFFFFF:
                    raise c.WinError(c.get_last_error())
            finally:
                api.CloseHandle(created.thread)
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
            end = time.monotonic() + RETIREMENT_SECONDS
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
                        raise subprocess.TimeoutExpired("Windows job member snapshot", RETIREMENT_SECONDS)
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
                        raise subprocess.TimeoutExpired("Windows job retirement", RETIREMENT_SECONDS)
                    time.sleep(.02)
                for handle in handles:
                    remaining = max(0, int((end - time.monotonic()) * 1000))
                    status = self.api.WaitForSingleObject(handle, remaining)
                    if status == 258:  # WAIT_TIMEOUT
                        raise subprocess.TimeoutExpired("Windows process retirement", RETIREMENT_SECONDS)
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
            self.process.wait(timeout=RETIREMENT_SECONDS)
            if os.name == "nt":
                # Popen otherwise retains its kernel handle until garbage
                # collection. Release this owner at the explicit close boundary;
                # wait()/poll() still use the cached return code afterwards.
                self.process._handle.Close()
            if os.name != "nt":
                end = time.monotonic() + RETIREMENT_SECONDS
                while group_alive(self.process.pid) and time.monotonic() < end:
                    time.sleep(.02)
                if group_alive(self.process.pid):
                    raise subprocess.TimeoutExpired("Process group retirement", RETIREMENT_SECONDS)
        self.closed = True


def guarded_tree(command, cwd, log, ownership, folder, env=None, deadline=None):
    ownership.launch_settled = True
    folder.mkdir(parents=True, exist_ok=True)
    launch = folder / ("launch-" + uuid.uuid4().hex + ".json")
    write_json(launch, {"owner_path": str(ownership.path), "token": ownership.token,
               "command": command, "cwd": str(cwd)}, immutable=True)
    ownership.launch_settled = False
    tree = ProcessTree([sys.executable, "-B", str(Path(__file__).with_name("guardian.py")), str(launch)], cwd, log,
                       job_name="Local\\crit-runtime-" + ownership.token, env=env)
    tree.launch_error_path = launch.with_suffix(".error.json")
    try:
        ready_path = launch.with_suffix(".ready.json")
        deadline = min(time.monotonic() + 10, deadline) if deadline is not None else time.monotonic() + 10
        while not ready_path.exists():
            if tree.process.poll() is not None:
                raise ValueError("Launch guardian exited before reporting its identity")
            if time.monotonic() >= deadline:
                raise TimeoutError("Launch guardian did not report its identity before the deadline")
            time.sleep(.01)
        ready = read_json(ready_path)
        keys(ready, {"token", "pid", "identity"})
        if ready["token"] != ownership.token:
            raise ValueError("Launch guardian ownership token changed")
        tree.identify_guardian(ready["pid"], ready["identity"])
        if time.monotonic() >= deadline:
            raise TimeoutError("Launch guardian passed its deadline before engine admission")
        ownership.bind(tree)
    except BaseException:
        try:
            tree.close()
            ownership.launch_settled = True
        except BaseException:
            ownership.failed_tree = tree
            ownership.block("Launch ownership could not be published or settled")
        raise
    return tree
