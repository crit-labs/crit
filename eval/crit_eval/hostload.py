"""CPU time other processes take during a measurement.

Engines that compute on the host (CPU expert paths, host-side samplers) slow down when other programs run, and a
speed case can then fail the disturbed or lower_context checks without anything wrong in the engine. HostLoad
reports how many CPU cores the rest of the machine kept busy over an interval: the host's busy CPU time minus this
process's own, over the wall time. None where the host gives no total (other platforms) or on any read error.
"""
import os
import time


def _host_busy_seconds():
    """Busy CPU seconds of the whole host since boot, or None."""
    try:
        if os.name == "nt":
            import ctypes
            idle, kernel, user = (ctypes.c_ulonglong() for _ in range(3))
            if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                return None
            # kernel time includes idle time; units of 100 ns
            return (kernel.value - idle.value + user.value) / 1e7
        with open("/proc/stat", encoding="ascii") as f:
            fields = f.readline().split()
        if fields[0] != "cpu":
            return None
        values = [int(v) for v in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)   # idle + iowait
        busy = sum(values[:8]) - idle                               # user..steal, guests already counted in user
        return busy / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _own_seconds():
    t = os.times()
    return t.user + t.system


class HostLoad:
    """start(), then cores() after the interval: average CPU cores other processes used, or None."""

    def start(self):
        self.wall, self.host, self.own = time.perf_counter(), _host_busy_seconds(), _own_seconds()
        return self

    def cores(self):
        host = _host_busy_seconds()
        if host is None or self.host is None:
            return None
        wall = time.perf_counter() - self.wall
        if wall <= 0:
            return None
        return max(0.0, (host - self.host) - (_own_seconds() - self.own)) / wall
