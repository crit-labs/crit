"""Read-only host memory observations and an admission check.

A logical byte budget does not prove the OS can back an allocation. This module
reports three independent observations, each possibly unknown:

- physical_available: memory obtainable without paging (Linux MemAvailable,
  Windows available physical memory);
- commit_available: commit headroom where the OS enforces commit (Windows
  available page file; Linux only with vm.overcommit_memory = 2);
- cgroup_available: the tightest ancestor cgroup headroom on Linux, counting
  only clean file pages as reclaimable (dirty and writeback pages are not).

`admit` compares a request plus an explicit reserve with the tightest known
observation. Standard library only; nothing is allocated or changed.

This is the Python host reference. Native engines call Core's
crit::core::observe_host_memory and crit::core::admit_host (crit/host_memory.h),
which apply these observation rules and also turn malformed values into unknowns;
admit_working_set follows admit_host.
"""
from dataclasses import dataclass, field
import os
import sys


@dataclass(frozen=True)
class Observation:
    physical_available: int | None = None
    commit_available: int | None = None
    cgroup_available: int | None = None
    notes: tuple = field(default_factory=tuple)  # why an observation is unknown

    def limits(self):
        return {k: v for k, v in (("physical", self.physical_available), ("commit", self.commit_available),
                                  ("cgroup", self.cgroup_available)) if v is not None}


@dataclass(frozen=True)
class Decision:
    admitted: bool
    binding: str | None   # observation that set the limit
    limit: int | None     # bytes available after the reserve
    reason: str


def parse_meminfo(text):
    """/proc/meminfo as bytes per key (values given in kB)."""
    values = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if not parts or not parts[0].isdigit():
            continue
        scale = 1024 if parts[1:2] == ["kB"] else 1
        values[name.strip()] = int(parts[0]) * scale
    return values


def parse_stat(text):
    """cgroup memory.stat: 'key value' per line."""
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            values[parts[0]] = int(parts[1])
    return values


def _read(path):
    try:
        with open(path, encoding="ascii") as f:
            return f.read()
    except OSError:
        return None


def _join(root, absolute):
    return os.path.join(root, absolute.lstrip("/"))


def cgroup_headroom(root="/"):
    """Tightest headroom over this process's cgroup and its ancestors, or (None, note).
    v2: memory.max - memory.current + clean file pages; v1: the memory controller's
    hierarchical limit with cache minus dirty/writeback as reclaimable."""
    membership = _read(_join(root, "/proc/self/cgroup"))
    if membership is None:
        return None, "cgroup membership unreadable"
    v2_path = v1_path = None
    for line in membership.splitlines():
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy == "0" and controllers == "":
            v2_path = path
        elif "memory" in controllers.split(","):
            v1_path = path
    best = None
    if v2_path is not None:
        current = v2_path
        while True:
            base = _join(_join(root, "/sys/fs/cgroup"), current)
            limit_text, usage_text = _read(os.path.join(base, "memory.max")), _read(os.path.join(base, "memory.current"))
            if limit_text is not None and usage_text is not None and limit_text.strip() != "max":
                stat = parse_stat(_read(os.path.join(base, "memory.stat")) or "")
                clean = max(0, stat.get("file", 0) - stat.get("file_dirty", 0) - stat.get("file_writeback", 0))
                headroom = max(0, int(limit_text) - int(usage_text) + clean)
                best = headroom if best is None else min(best, headroom)
            if current in ("/", ""):
                break
            current = current.rsplit("/", 1)[0] or "/"
        if best is not None or v1_path is None:
            return best, None if best is not None else "no cgroup v2 memory limit"
    # No limited v2 level: a v1 memory hierarchy (systemd hybrid hosts) still applies.
    if v1_path is not None:
        base = _join(_join(root, "/sys/fs/cgroup/memory"), v1_path)
        stat = parse_stat(_read(os.path.join(base, "memory.stat")) or "")
        usage_text = _read(os.path.join(base, "memory.usage_in_bytes"))
        limit = stat.get("hierarchical_memory_limit")
        if limit is None or usage_text is None or limit >= 1 << 62:
            return None, "no cgroup v1 memory limit"
        clean = max(0, stat.get("total_cache", stat.get("cache", 0)) - stat.get("total_dirty", stat.get("dirty", 0))
                    - stat.get("total_writeback", stat.get("writeback", 0)))
        return max(0, limit - int(usage_text) + clean), None
    return None, "no memory cgroup"


def observe_linux(root="/"):
    notes = []
    info = parse_meminfo(_read(_join(root, "/proc/meminfo")) or "")
    physical = info.get("MemAvailable")
    if physical is None:
        notes.append("MemAvailable unavailable")
    commit = None
    mode = (_read(_join(root, "/proc/sys/vm/overcommit_memory")) or "").strip()
    if mode == "2" and "CommitLimit" in info and "Committed_AS" in info:
        commit = max(0, info["CommitLimit"] - info["Committed_AS"])
    else:
        notes.append("commit not enforced (overcommit_memory=%s)" % (mode or "unknown"))
    cgroup, note = cgroup_headroom(root)
    if note:
        notes.append(note)
    return Observation(physical, commit, cgroup, tuple(notes))


def observe_windows():
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return Observation(notes=("GlobalMemoryStatusEx failed",))
    # ullAvailPageFile is the commit headroom (commit limit minus committed).
    return Observation(int(status.ullAvailPhys), int(status.ullAvailPageFile), None, ("no cgroups on Windows",))


def observe():
    if sys.platform.startswith("linux"):
        return observe_linux()
    if sys.platform == "win32":
        return observe_windows()
    return Observation(notes=("unsupported platform " + sys.platform,))


def admit(request, observation, reserve=0, require_known=True):
    """Admit `request` bytes when request + reserve fits the tightest known
    observation. With no known observation, deny unless require_known is False."""
    if request < 0 or reserve < 0:
        raise ValueError("request and reserve must be non-negative")
    limits = observation.limits()
    if not limits:
        return Decision(not require_known, None, None, "no observation: " + "; ".join(observation.notes))
    binding = min(limits, key=lambda k: (limits[k], k))
    available = limits[binding] - reserve
    if request <= available:
        return Decision(True, binding, available, "fits %s headroom" % binding)
    return Decision(False, binding, max(0, available), "%d bytes exceed %s headroom of %d after a %d-byte reserve"
                    % (request, binding, max(0, available), reserve))


def admit_working_set(*, anonymous_bytes, pinned_bytes, file_bytes, observation,
                     reserve=0, require_physical=True):
    """Check incremental peak bytes against each applicable OS memory domain.

    The three extents are disjoint: anonymous excludes pinned; file_bytes is the
    additional clean file-backed working set, not the file's total mapping size.
    Physical/cgroup headroom pays all three; commit pays anonymous + pinned.
    Already resident/charged pages must not be counted again. The observation
    already subtracts other allocations; reserve covers uncertainty/concurrency.
    This is admission evidence, not an allocation or a guarantee against races.
    """
    for value in (anonymous_bytes, pinned_bytes, file_bytes, reserve):
        if type(value) is not int or value < 0:
            raise ValueError("nonnegative integer memory extents/reserve required")
    if type(require_physical) is not bool:
        raise ValueError("explicit physical-memory policy required")
    limits = observation.limits()
    if any(type(v) is not int or v < 0 for v in limits.values()):
        raise ValueError("invalid memory observation")
    physical = anonymous_bytes + pinned_bytes + file_bytes
    if not physical:
        return Decision(True, None, None, "no incremental host working set")
    if not limits or (require_physical and "physical" not in limits):
        return Decision(False, None, None, "required host memory observation is unknown")
    needs = {"physical": physical, "cgroup": physical, "commit": anonymous_bytes + pinned_bytes}
    # Choose the tightest remaining margin, since demands differ by domain.
    binding = min(limits, key=lambda k: (limits[k] - reserve - needs[k], k))
    available = max(0, limits[binding] - reserve)
    allowed = all(needs[k] + reserve <= v for k, v in limits.items())
    return Decision(allowed, binding, available,
                    ("fits " if allowed else "exceeds ") + binding + " working-set headroom")
