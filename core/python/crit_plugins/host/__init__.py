"""The host's processors, one object a process: who runs where, the worker pool, and the work the host does inside a
device's stream.

An engine's threads share one machine: the workers that compute what the device does not hold, the thread that
answers the device's doorbells, the loop that drives the device. Left to each plugin, every one starts its own pool
and picks its own cores, and the threads of an engine outnumber them. So the host is one object, as a device's
memory is (crit_plugins.memory):

    host = current()                  # the process's host; LoadContext.host
    host.cores                        # the physical cores this process may run on: the first logical CPU of each
    host.plan()                       # {"service": cpu, "workers": [cpu, ...]}: who runs where
    pool = host.pool()                # the worker pool: one a process, for Core and every plugin
    pool.run(tasks, fn)               # fn(i) for i in range(tasks), from Python (tests and slow paths)
    entry, handle = pool.native       # what a plugin's native library runs its batches on (pool.h's PoolRef)
    handoff = host.handoff(device, owner, name, slots, payload_bytes, serve, context)
    reads = host.reads(owner, path)   # one file read at a queue depth, past the page cache (files.py)
    host.report()                     # an engine's telemetry "host"

The plan. The first physical core is the service's: the thread that answers the device's doorbells and runs their
jobs, which is also the caller of every batch such a job hands to the pool. Each other core has one worker, pinned
to it (Linux). Nothing else of the engine computes on the host while the device waits for a job, so a job has every
core, each once.

The pool (pool.h). One batch at a time, claimed by the workers and the caller without a lock; batches of different
callers run one after the other. Its size is the plan's: the first pool() may size it (workers, pin; without pin
the service thread is not pinned either), and what it is serves everyone after that, since any split of a batch
gives the same result. Native code built apart from Core
runs on it through the two addresses of `native`.

Handoffs (handoff.h, handoff_device.cuh). Work the device decides and the host does inside one CUDA stream, captured
graphs included: a kernel writes a job into a slot's payload (mapped pinned memory) and raises its doorbell, the
service thread runs the handler (a C function of a host library) and answers, and the stream's join waits for the
answer where the device reads the result. The join is the cheaper capturable one measured on the device
(joins(device)): the driver's stream wait, or a one-thread poll kernel. A plugin's kernels post and read through
handoff_device.cuh (INCLUDE is its folder) with the slot's device addresses, words(slot). A stream's wait has no
clock, so a job that never returned would hang the device for good: a second thread of the service watches, and a
job that runs past the job budget (JOB_BUDGET_MS, job_budget()) fails its handoff. Its waits are answered at once,
failed() says why, and the owner stops serving instead of hanging.

File reads (files.py). A tier that keeps bytes of a file larger than host memory reads it through reads(): positioned
reads from a few threads at once, each with its own handle, opened past the OS's page cache where the file system
allows, so the drive works at a queue depth and nothing is held twice. Those threads wait in the kernel and compute
nothing: they are no workers of the pool and are pinned nowhere.

Backends. The plan and the pool know no device. A handoff's device side (the mapped memory, the post and the join)
is a runtime service of the device's backend: CUDA is the one built today, handoff() raises NativeUnavailable for a
device of any other, and another backend brings its own behind the same call.

Core's native code for all of this is built on this machine at first use (crit_plugins.native): the pool by the
system's C++ compiler, the handoff by torch's extension builder for the device's architecture. Where the pool does
not build, pool() raises NativeUnavailable and the plan is empty; where the handoff does not, handoff() raises it.
"""
import ctypes
import os
import threading
from pathlib import Path

from ..native import NativeUnavailable, built, extension, library
from .files import Reads, write_memory

HERE = Path(__file__).resolve().parent
INCLUDE = HERE   # pool.h (PoolRef) and handoff_device.cuh, for a plugin's native code
SPIN_US, SPIN_MAX_US = 100, 3000          # a worker spins as long as the recent gaps between batches, within these
IDLE_AFTER_US, IDLE_SLEEP_US = 2000, 50   # the service spins this long after a job, then sleeps this long a pass
JOB_BUDGET_MS = 20000                     # a handoff's job that runs longer fails the handoff: the device waits no more
MAX_CPUS = 4096
TASK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t)


def memory_bytes():
    """The host's memory in bytes: {"total", "available"} (what a process can take without paging: Linux's
    MemAvailable, Windows' available physical memory); 0 where the OS does not say."""
    out = {"total": 0, "available": 0}
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                name, _, rest = line.partition(":")
                if name == "MemTotal":
                    out["total"] = int(rest.split()[0]) * 1024
                elif name == "MemAvailable":
                    out["available"] = int(rest.split()[0]) * 1024
        return out
    except OSError:
        pass
    if os.name == "nt":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                        ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                        ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            out = {"total": int(status.total), "available": int(status.available)}
    return out


def _declare(lib):
    lib.crit_pool_create.restype = ctypes.c_void_p
    lib.crit_pool_create.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.crit_pool_destroy.restype = None
    lib.crit_pool_destroy.argtypes = [ctypes.c_void_p]
    lib.crit_pool_run.restype = ctypes.c_int
    lib.crit_pool_run.argtypes = [ctypes.c_void_p, ctypes.c_size_t, TASK, ctypes.c_void_p]
    lib.crit_pool_stats.restype = None
    lib.crit_pool_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.crit_host_core_leaders.restype = ctypes.c_int
    lib.crit_host_core_leaders.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.crit_host_pin_thread.restype = ctypes.c_int
    lib.crit_host_pin_thread.argtypes = [ctypes.c_int]
    return lib


class Pool:
    """The process's worker pool (pool.h): one worker per CPU of `cpus`, pinned to it where it is >= 0."""

    def __init__(self, lib, cpus, spin_us=SPIN_US, spin_max_us=SPIN_MAX_US):
        self.lib, self.cpus = lib, [int(c) for c in cpus]
        array = (ctypes.c_int * max(1, len(self.cpus)))(*self.cpus)
        self.handle = lib.crit_pool_create(array, len(self.cpus), int(spin_us), int(spin_us), int(spin_max_us))
        if not self.handle:
            raise NativeUnavailable("the host's worker pool did not start its threads")

    @property
    def workers(self):
        return len(self.cpus)

    @property
    def native(self):
        """(the address of the pool's C entry, the pool): pool.h's PoolRef, for native code built apart from Core."""
        return ctypes.cast(self.lib.crit_pool_run, ctypes.c_void_p).value, int(self.handle)

    def run(self, tasks, task):
        """task(i) for i in range(tasks) on the workers and the calling thread; returns when all have finished and
        raises the first exception a task raised. Python tasks hold the interpreter's lock one at a time: this is
        for tests and paths where the tasks release it (native calls)."""
        errors = []

        def one(_, index):
            try:
                task(index)
            except BaseException as error:   # noqa: BLE001 - raised below, on the caller's thread
                errors.append(error)

        self.lib.crit_pool_run(self.handle, int(tasks), TASK(one), None)
        if errors:
            raise errors[0]

    def stats(self):
        values = (ctypes.c_uint64 * 3)()
        self.lib.crit_pool_stats(self.handle, values)
        return {"workers": int(values[2]), "cpus": [c for c in self.cpus if c >= 0], "batches": int(values[0]),
                "wakes": int(values[1])}

    def close(self):
        if self.handle:
            self.lib.crit_pool_destroy(self.handle)
            self.handle = None


class Handoff:
    """Host work joined inside a device's stream (handoff.h): `slots` slots of `payload_bytes` bytes in mapped
    pinned memory, a handler that serves a posted slot on the host's service thread, and the join the device's
    stream waits in. Every call that touches the stream uses torch's current stream of the handoff's device, and
    but for an eager join each can be captured into a graph."""

    def __init__(self, host, native, device, label):
        self.host, self.native, self.device, self.label = host, native, device, label
        self.join_kind = native.join_kind()
        self.slots, self.payload_bytes, self.bytes = native.slots(), native.payload_bytes(), native.bytes()

    def words(self, slot):
        """The slot's device addresses (ready, done, status, payload): handoff_device.cuh's DeviceSlot, for the
        kernels that post its jobs and read their results."""
        return self.native.words(int(slot))

    def post(self, slot):
        """Raise the slot's doorbell from the stream (a one-thread kernel); a plugin's own kernel usually posts."""
        self.native.post(int(slot))

    def skip(self, slot):
        """The device answers the slot itself: no host work this time."""
        self.native.skip(int(slot))

    def join(self, slot):
        """The stream waits here until the slot is answered."""
        self.native.join(int(slot))

    def release(self):
        """Answer every pending and later doorbell as released (a shutdown, a stuck window): outputs of windows
        that saw it are invalid; rearm() starts the next window."""
        self.native.release()

    def rearm(self, timeout_ms=5000):
        return self.native.rearm(int(timeout_ms))

    def failed(self):
        """A handler that failed fails the handoff for good: every later doorbell is answered as failed."""
        return self.native.failed()

    def error(self):
        return self.native.error()

    def served(self):
        return self.native.served()

    def status(self):
        return {"join": self.join_kind, "slots": self.slots, "bytes": self.bytes, "served": self.served(),
                "failed": self.failed()}

    def close(self):
        """Answer every pending doorbell, let the device's streams drain, then give the slots back. Before the
        handler's library goes."""
        if self.native is None:
            return
        import torch
        self.native.release()
        torch.cuda.synchronize(self.device)
        self.host._left(self)
        self.native = None


class Host:
    """The host's processors. cores: the first logical CPU of each physical core this process may run on (given
    for a test; read from the OS through the pool's library otherwise)."""

    def __init__(self, cores=None):
        self._lock = threading.RLock()
        self._cores = None if cores is None else [int(c) for c in cores]
        self._lib = self._why = None
        self._pool = None
        self._sized = None        # (workers, pin) the pool was made with
        self._module = None
        self._joins = {}          # device index -> the measured joins
        self._handoffs = {}       # owner/name -> Handoff
        self._reads = {}          # owner: path -> Reads
        self._budget_ms, self._service_pin = JOB_BUDGET_MS, 0
        self._service = None

    # -- the cores and the plan --------------------------------------------------------------------------------------
    def _library(self):
        """Core's host library (pool.cpp), built at first use; NativeUnavailable where it does not build."""
        with self._lock:
            if self._lib is None:
                if self._why is not None:
                    raise NativeUnavailable(self._why)
                try:
                    self._lib = _declare(library("crit_host", [HERE / "pool.cpp"], [HERE]))
                except NativeUnavailable as error:
                    self._why = str(error)
                    raise
            return self._lib

    @property
    def cores(self):
        with self._lock:
            if self._cores is None:
                try:
                    lib = self._library()
                    found = (ctypes.c_int * MAX_CPUS)()
                    self._cores = list(found[:min(MAX_CPUS, lib.crit_host_core_leaders(found, MAX_CPUS))])
                except NativeUnavailable:
                    self._cores = []
            return list(self._cores)

    def plan(self, workers=None, pin=None):
        """Who runs where: {"service": the CPU of the thread that answers the device's doorbells (None: not
        pinned), "workers": the pool's workers' CPUs (-1: a worker that is not pinned)}. workers: how many (None
        or -1: one per physical core but the service's); pin False pins nothing. Where the OS does not name the
        cores, nothing is pinned and the workers are the hardware threads halved, less one."""
        cores = self.cores
        pin = True if pin is None else bool(pin)
        if workers is None or int(workers) < 0:
            workers = len(cores) - 1 if cores else max(0, (os.cpu_count() or 2) // 2 - 1)
        workers = max(0, int(workers))
        free = cores[1:] if pin else []
        return {"service": cores[0] if cores and pin else None,
                "workers": [free[i] if i < len(free) else -1 for i in range(workers)]}

    # -- the pool ----------------------------------------------------------------------------------------------------
    def pool(self, workers=None, pin=None):
        """The process's worker pool, made at the first call with the plan (workers, pin size it then). A later
        call gets the pool as it is, whatever it asks: one pool serves the process."""
        with self._lock:
            if self._pool is None:
                lib = self._library()
                self._sized = (workers, pin)
                self._pool = Pool(lib, self.plan(workers, pin)["workers"])
            return self._pool

    # -- handoffs ----------------------------------------------------------------------------------------------------
    def _handoff_module(self):
        with self._lock:
            if self._module is None:
                self._module = extension("crit_host_handoff", [HERE / "handoff_module.cu", HERE / "handoff.cu"], [HERE])
            return self._module

    def joins(self, device):
        """The capturable joins of `device`, measured once a process: {"mapped", "stream_wait" (the driver's stream
        wait works in a captured graph), "stream_wait_us", "device_poll_us" (one post and join, None where not
        supported), "choice" (the cheaper: stream_wait, device_poll, or host_eager when neither), "detail"}."""
        import torch
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        with self._lock:
            if index not in self._joins:
                self._joins[index] = dict(self._handoff_module().measure_joins(index, 32, 50))
            return dict(self._joins[index])

    def handoff(self, device, owner, name, slots, payload_bytes, serve, context=0, join=None):
        """A handoff on `device`, listed as owner/name: `slots` slots of `payload_bytes` bytes, each served by
        serve, the address of a C function int(void* context, uint64 slot, void* payload, uint64 bytes,
        char* error, uint64 error_bytes) that returns 0 or 1 with the error's text. join: stream_wait, device_poll
        or host_eager; None takes the cheaper capturable one measured on the device. Raises NativeUnavailable where
        Core's handoff does not build."""
        import torch
        device = torch.device(device)
        if device.type != "cuda":
            raise NativeUnavailable(f"a handoff joins inside a CUDA stream: {device} has none")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        with self._lock:
            module = self._handoff_module()
            kind = join or self.joins(device)["choice"]
            if self._service is None:   # the service thread's CPU and idle times, before its first handoff
                service = self.plan(*(self._sized or (None, None)))["service"]   # pinned as the pool's workers are
                pin = 0
                if service is not None:
                    try:
                        pin = ctypes.cast(self._library().crit_host_pin_thread, ctypes.c_void_p).value
                    except NativeUnavailable:
                        service = None
                self._service = -1 if service is None else service
                self._service_pin = pin
                module.service(self._service, pin, IDLE_AFTER_US, IDLE_SLEEP_US, self._budget_ms)
            label, n = f"{owner}/{name}", 1
            while label in self._handoffs:
                n += 1
                label = f"{owner}/{name}#{n}"
            native = module.Handoff(device.index, int(slots), int(payload_bytes), kind, int(serve), int(context))
            found = self._handoffs[label] = Handoff(self, native, device, label)
            return found

    def job_budget(self, milliseconds):
        """The longest a handoff's job may run before the service's watch fails its handoff and releases the
        device's waits (JOB_BUDGET_MS by default; 0: no watch). For every handoff of the process, from now on."""
        with self._lock:
            self._budget_ms = int(milliseconds)
            if self._service is not None:
                self._handoff_module().service(self._service, self._service_pin, IDLE_AFTER_US, IDLE_SLEEP_US,
                                               self._budget_ms)

    def failures(self):
        """{owner/name: why} of the handoffs that failed (a handler that failed, a job past the job budget): what
        their windows computed since is invalid, and an engine that owns one stops serving."""
        with self._lock:
            handoffs = list(self._handoffs.items())
        return {label: handoff.native.error() for label, handoff in handoffs
                if handoff.native is not None and handoff.failed()}

    def _left(self, handoff):
        with self._lock:
            self._handoffs.pop(handoff.label, None)

    # -- file reads --------------------------------------------------------------------------------------------------
    def reads(self, owner, path, depth=None, direct=True):
        """Reads of the file at `path` for `owner` (files.py's Reads): positioned reads at a queue depth (8 by
        default), past the OS's page cache where the file system allows and direct asks. The owner closes it; the
        host's report lists it until then."""
        found = Reads(path, depth or 8, direct)
        with self._lock:
            self._reads[f"{owner}: {path}"] = found
        return found

    # -- what serves -------------------------------------------------------------------------------------------------
    def report(self):
        """The host as an engine's telemetry reads it: its cores and memory, the pool's workers and counters, the
        service thread's CPU, each handoff's join and jobs, the measured joins, and what was built on this machine
        for it and for the plugins (crit_plugins.native's record)."""
        with self._lock:
            out = {"cores": len(self.cores), "cpus": os.cpu_count() or 0, "memory": memory_bytes()}
            if self._pool is not None:
                out["pool"] = self._pool.stats()
            elif self._why is not None:
                out["pool"] = {"unavailable": self._why.splitlines()[0][:300]}
            if self._service is not None:
                out["service"] = {"cpu": self._service if self._service >= 0 else None,
                                  "job_budget_ms": self._budget_ms}
            if self._handoffs:
                out["handoffs"] = {label: handoff.status() for label, handoff in self._handoffs.items()}
            if self._joins:
                out["joins"] = {f"cuda:{index}": dict(found) for index, found in self._joins.items()}
            self._reads = {label: found for label, found in self._reads.items() if not found.closed}
            if self._reads:
                out["reads"] = {label: found.stats() for label, found in self._reads.items()}
            native = built()
            if native:
                out["native"] = native
            return out

    def close(self):
        """Stop the pool (a test's host): handoffs and whatever runs on the pool are their owners' to close first.
        The process's host is never closed: its threads end with the process."""
        with self._lock:
            if self._pool is not None:
                self._pool.close()
                self._pool = None


_current = None
_current_lock = threading.Lock()


def current():
    """The process's host."""
    global _current
    with _current_lock:
        if _current is None:
            _current = Host()
        return _current


__all__ = ["Handoff", "Host", "INCLUDE", "NativeUnavailable", "Pool", "Reads", "current", "memory_bytes",
           "write_memory"]
