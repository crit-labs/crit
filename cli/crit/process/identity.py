"""A process's birth identity, so a recorded PID never names another process after reuse: its start time (Windows) or
its boot and start ticks (Linux), None once it has exited, "unknown" when that cannot be told. Standard library only;
crit's coordinator lease and runtime ownership both ask it."""
import os
from pathlib import Path


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
