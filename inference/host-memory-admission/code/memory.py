"""Read-only host memory observations and an admission check.

A logical byte budget does not prove the OS can back an allocation. This module
reports three independent observations, each possibly unknown:

- physical_available: memory obtainable without paging (Linux MemAvailable,
  Windows available physical memory);
- commit_available: commit headroom where the OS enforces commit (Windows
  available page file; Linux only with vm.overcommit_memory = 2);
- cgroup_available: the tightest ancestor cgroup headroom on Linux, counting
  only clean page cache as reclaimable (tmpfs/shmem, dirty and writeback pages
  are not).

`admit` compares a request plus an explicit reserve with the tightest known
observation. Standard library only; nothing is allocated or changed.

This is the Python host reference. Native engines call Core's
crit::core::observe_host_memory and crit::core::admit_host (crit/host_memory.h),
which apply the same observation rules, including malformed values becoming
unknowns (or no reclaimable pages) with a note; admit_working_set follows admit_host.
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
    """cgroup memory.stat: 'key value' per line; a malformed value is None."""
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            values[parts[0]] = _u64(parts[1])
    return values


_U64_MAX = (1 << 64) - 1


def _u64(text):
    """Decimal digits only, without sign, below 2**64; otherwise None."""
    if not (text.isascii() and text.isdigit()):
        return None
    value = int(text)
    return value if value <= _U64_MAX else None


def _read(path):
    try:
        with open(path, encoding="ascii") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def _join(root, absolute):
    return os.path.join(root, absolute.lstrip("/"))


def clean_pages(stat, cache, dirty, writeback, totals, source, notes):
    """Clean page cache: the file LRUs (active_file + inactive_file, which exclude
    tmpfs/shmem pages) when listed, else cache - shmem; minus dirty and writeback.
    v1 (totals) prefers each key's hierarchical total_ form. A malformed entry
    counts nothing as reclaimable, as in Core."""
    def key(name):
        return "total_" + name if totals and "total_" + name in stat else name
    active, inactive = key("active_file"), key("inactive_file")
    lru = active in stat or inactive in stat
    a, b, d, w = (stat.get(k, 0) for k in ((active, inactive) if lru else (key(cache), key("shmem")))
                  + (key(dirty), key(writeback)))
    if None in (a, b, d, w):
        notes.append(source + " malformed; no file pages counted as reclaimable")
        return 0
    return max(0, (min(a + b, _U64_MAX) if lru else max(0, a - b)) - d - w)


def cgroup_headroom(root="/"):
    """Tightest headroom over this process's cgroup and its ancestors, and a note
    (None when there is nothing to say). v2: min(memory.max, memory.high) -
    memory.current + clean page cache; v1: the memory controller's hierarchical
    limit with the same reclaimable rule. Follows Core's observe_host_memory_linux."""
    notes = []
    membership = _read(_join(root, "/proc/self/cgroup"))
    if membership is None:
        return None, "cgroup membership unreadable"
    v2_path = v1_path = None
    for line in membership.splitlines():
        hierarchy, sep1, rest = line.partition(":")
        controllers, sep2, path = rest.partition(":")
        if not (sep1 and sep2):
            continue
        if hierarchy == "0" and controllers == "":
            v2_path = path
        elif "memory" in controllers.split(","):
            v1_path = path

    def result(value):
        return value, "; ".join(notes) or None

    if v2_path is not None:
        best = None
        current = v2_path
        while True:
            base = _join(_join(root, "/sys/fs/cgroup"), current)
            # Past memory.high the kernel throttles and reclaims, so a level limited
            # only by memory.high is still limited. Unreadable reads as "max".
            max_text, high_text = (("max" if t is None else t).strip()
                                   for t in (_read(os.path.join(base, n)) for n in ("memory.max", "memory.high")))
            usage_text = _read(os.path.join(base, "memory.current"))
            if usage_text is not None and (max_text != "max" or high_text != "max"):
                limits = [_U64_MAX if t == "max" else _u64(t) for t in (max_text, high_text)]
                used = _u64(usage_text.strip())
                if None in limits or used is None:
                    notes.append("cgroup %s memory.max, memory.high or memory.current malformed" % base)
                else:
                    stat_path = os.path.join(base, "memory.stat")
                    stat = parse_stat(_read(stat_path) or "")
                    clean = clean_pages(stat, "file", "file_dirty", "file_writeback", False, stat_path, notes)
                    headroom = max(0, min(min(limits) + clean, _U64_MAX) - used)
                    best = headroom if best is None else min(best, headroom)
            if current in ("/", ""):
                break
            current = current.rsplit("/", 1)[0] or "/"
        if best is not None:
            return result(best)
    # No limited v2 level: a v1 memory hierarchy (systemd hybrid hosts) still applies.
    if v1_path is not None:
        base = _join(_join(root, "/sys/fs/cgroup/memory"), v1_path)
        stat_path = os.path.join(base, "memory.stat")
        stat = parse_stat(_read(stat_path) or "")
        usage_text = _read(os.path.join(base, "memory.usage_in_bytes"))
        if "hierarchical_memory_limit" not in stat or usage_text is None:
            notes.append("no cgroup v1 memory limit")
        elif stat["hierarchical_memory_limit"] is None:
            notes.append("cgroup %s hierarchical_memory_limit malformed" % base)
        elif stat["hierarchical_memory_limit"] >= 1 << 62:
            notes.append("no cgroup v1 memory limit")
        elif _u64(usage_text.strip()) is None:
            notes.append("cgroup %s memory.usage_in_bytes malformed" % base)
        else:
            clean = clean_pages(stat, "cache", "dirty", "writeback", True, stat_path, notes)
            return result(max(0, min(stat["hierarchical_memory_limit"] + clean, _U64_MAX) - _u64(usage_text.strip())))
        if v2_path is not None:
            notes.append("no cgroup v2 memory limit")
        return result(None)
    notes.append("no cgroup v2 memory limit" if v2_path is not None else "no memory cgroup")
    return result(None)


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
