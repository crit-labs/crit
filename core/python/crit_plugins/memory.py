"""The device's memory: one object a device, for Core and for every plugin.

    memory = crit_plugins.memory.of("cuda:0")

Every byte of a device is somebody's: a plugin's tensors in torch's allocator, an elastic range, blocks torch holds
and nothing uses, the driver's and other processes' (a desktop's among them), or free. Core's loader and the plugins
read and move them here, so one place knows who holds what and what may be taken.

Readings, in bytes (None where the device's memory cannot be read: no CUDA device)

    memory.total                 the device's memory
    memory.free()                what the driver has free now; free(release=True) first hands back torch's unused blocks
    memory.torch()               {"allocated", "reserved", "unused"}: torch's allocator here; unused are the whole
                                 blocks nothing uses (what release() hands back to the driver: the empty segments
                                 of captured graphs' own pools are not among them, a graph's replays write there)
    memory.release(below=None)   hand them back; with `below`, only when fewer than that many bytes are free
    memory.mark(), memory.peak() the most torch's allocator held (reserved) since the mark: what a step took at its peak
    memory.made()                the allocations torch's allocator has made here so far (a count): unchanged over a
                                 step that only replays captured graphs

Elastic ranges: one reserved virtual address range, backed granule by granule through the driver's virtual memory
API, so a tier grows and shrinks while every address in it stays the same (captured graphs and kernel arguments keep
pointing at it). The driver is reached through ctypes (libcuda on Linux, nvcuda on Windows): nothing links against it.

    r = memory.elastic("expert-offload", "slots", max_bytes)   # reserves address space, backs its first granule
    r.grow(n)                    # backs [0, n) (whole granules), zeroed
    r.tensor                     # uint8 [max_bytes] over the whole range: only backed bytes may be read or written
    r.shrink(m)                  # gives the backing past m back to the device
    r.back(first, end)           # backs the granules holding [first, end): a range backed in places
    r.back(first, end, host=True)    # the same from host memory: the device reads it through the range, over the link
    r.release(first, end)        # gives back the granules wholly inside [first, end)
    r.close()

memory.elastic_ranges says whether the device and driver have the API; where they do not, a tier keeps one fixed
allocation. A tier that fills from its start (an expert cache's slots) grows and shrinks; one whose live parts lie
apart (a paged pool: the first pages of every layer's tensors) backs and releases them where they are:

    p = memory.paged("attention-state", "pages", nbytes, planes)   # planes: [(first byte, bytes a unit, units)]
    p.tensor                     # the range's tensor; p.backed units of every plane are backed
    p.back(units)                # backs units [0, units) of every plane, the givers making room first: False when
                                 # the device has no memory for them (nothing is backed then)
    p.trim(units)                # gives back what lies past them; the givers grow into it

A pool's units are what its sequences take as they come: a KV pool's pages, a state pool's sequences. It takes
device memory as they grow and gives it back as they close, instead of holding its whole size from the load on.

A workspace made for each call shape (a prompt chunk's size) is one buffer at a fixed address that grows to its
largest user, where each shape would keep its own for good:

    g = memory.growing("expert-offload", "routes.y", max_bytes)
    g.take(nbytes)               # backs [0, nbytes), the givers making room first: that prefix, uint8

Scratch is what a call computes in and nothing reads once it has returned: a kernel's partial sums, a workspace
of route rows, a decoded tile. An engine's calls run one after another on its stream, so they all compute in the
same bytes:

    scratch = memory.scratch()               # an engine's: Core's loader makes it, a plugin finds it as context.scratch
    rows = scratch.take(nbytes)              # the first nbytes, uint8: taken by each call, every time it runs
    y, h = scratch.take(a, b)                # a call's buffers together, one after another (each 64-byte aligned)

It is one growing buffer, so its address is fixed and a captured graph keeps it, and it costs the engine its
largest call's bytes where a workspace a plugin would cost their sum. A call holds nothing there across calls:
what the next call finds, another plugin's or its own, is undefined, and one take is all a call has (a second
take is the same bytes). It grows outside captures only: a call's eager run (a capture's warm-up) takes its
size first. With CRIT_SCRATCH_POISON=1 every take fills what it hands out with 0xFF bytes (NaN as a float, -1
as an index), inside captured graphs too, so a call that reads what an earlier one left fails where it can be
seen: the plugin tests run that way.

The ledger makes the free memory one budget. A tier that can give memory back (an expert cache: its coldest slots)
is a giver and says what it keeps free (`keep`: bytes, or a callable read when asked, so it can follow the requests
in flight). Whoever is about to take device memory it cannot do without (KV pages a request reserves) calls
need(bytes) first: the givers release what the device lacks to hold those bytes with what they keep still free.
settle() lets them grow back down to what they keep. Nothing is taken from a giver while the device has room, so a
tier sized to every free byte costs no request its memory.

    memory.give("expert-offload", release=lambda n: freed_bytes, regrow=lambda room: taken_bytes, keep=margin)
    memory.need(nbytes)          # before the allocation, outside a capture
    memory.settle()              # the device is quiet: each giver regrows into free memory, less what it keeps
    memory.relieve()             # between steps: another program took memory, the givers hand theirs back now

The device is not the engine's alone: a desktop, a browser tab or a second engine takes memory while this one
serves, and nothing in the engine asks for it. Where the driver would then move allocations to system memory
(WDDM), every kernel that reads them slows down and nothing fails. So Core calls relieve() between an engine's
steps (at most every RELIEVE_S): where less is free than the givers keep, they release the difference, as for a
need of the engine's own. When what other programs hold has grown since the last look, the givers also wait
HOLD_S before they grow back, so a program whose memory comes and goes does not make them give and take with it.
A tier of copies (an expert cache) loses speed by this and never a token. Rounds that Core launches without the
engine's Python run no step of it: watch(ask) starts a thread of Core's that reads the device's free memory every
RELIEVE_S and calls ask() when it fell under what the givers keep, and the engine sends its next decode step
through step() once, where relieve() runs (crit_core.native's NativeRounds.ask_step).

An engine loads, then serves. While it loads, what its plugins left free at their creation is the load's (graphs,
workspaces, first prompts), and settle() does nothing; Core calls serve() once the warm requests ran, and the givers
take what is left. Under WDDM (Windows, WSL) the driver does not refuse an allocation past the device's memory: it
moves it to system memory, and every kernel that reads it slows down. A giver therefore never grows past the free
memory it is told, and free memory is the driver's: blocks torch holds unused count as taken.

The census says who holds torch's bytes: census(plugins) walks the plugin objects and counts each live tensor
storage of this device for the plugin whose code made the object holding it. report() gives the whole device for
an engine's telemetry: free, torch's allocator, each elastic range, each plugin, and what is neither (the CUDA
context, other processes).

A captured graph's temporaries live in a memory pool of its own, which torch keeps for the graph's replays.
pool(owner, name) makes one and lists it as owner/name, so the device's readings say which graph holds what; a
driver captures inside capturing(pool). After record(), the device also notes which code made each allocation:
every block of torch's allocator carries the code that made it (blocks()), and each pool the allocations made in it
while its graphs were captured (`recorded`), which crit_plugins.audit reads."""
import ctypes
import os
import sys
import threading
import time
from bisect import bisect_right
from collections import Counter, deque
from contextlib import contextmanager

_CU = None
_HANDLE_NONE, _LOCATION_DEVICE, _LOCATION_HOST, _ACCESS_RW = 0, 1, 2, 3
_OUT_OF_MEMORY = 2   # CUDA_ERROR_OUT_OF_MEMORY
SCRATCH_BYTES = 16 << 30        # the address space of an engine's scratch: what one call may take
POISON = "CRIT_SCRATCH_POISON"  # "1": every take of a scratch is filled with 0xFF bytes first (the plugin tests)
RECORDED = 1 << 17   # allocation events torch keeps while a device records: several captures' worth


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _Prop(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int), ("location", _Location),
                ("win32HandleMetaData", ctypes.c_void_p), ("allocFlags", ctypes.c_ubyte * 8)]


class _Access(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


class _Interface:
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {"shape": (int(n),), "typestr": "|u1", "data": (int(ptr), False),
                                         "version": 3}


def _driver():
    global _CU
    if _CU is None:
        names = ("nvcuda.dll",) if os.name == "nt" else ("libcuda.so.1", "libcuda.so")
        for name in names:
            try:
                _CU = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            _CU = False
    return _CU or None


def _check(code, what):
    if code != 0:
        raise RuntimeError(f"elastic range: {what} failed (CUresult {code})")


def _prop(index):
    p = _Prop()
    p.type, p.requestedHandleTypes, p.location = 1, _HANDLE_NONE, _Location(_LOCATION_DEVICE, int(index))
    return p


def granularity(index=0):
    """The device's mapping granule in bytes, 0 where the virtual memory API is not reachable."""
    cu = _driver()
    if cu is None:
        return 0
    g = ctypes.c_size_t()
    if cu.cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(_prop(index)), 0) != 0:
        return 0
    return int(g.value)


class ElasticRange:
    """One reserved address range of a CUDA device, backed granule by granule (the module's description). Made by
    DeviceMemory.elastic(), which lists it in the device's census until it is closed."""

    def __init__(self, max_bytes, device="cuda", initial=0):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        torch.cuda.init()
        torch.zeros(1, device=self.device)   # the context exists before the driver calls
        self.cu = _driver()
        self.gran = granularity(self.index)
        if not self.cu or not self.gran:
            raise RuntimeError("elastic range: the driver's virtual memory API is not reachable here")
        self.size = -(-int(max_bytes) // self.gran) * self.gran
        va = ctypes.c_uint64()
        _check(self.cu.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(self.size), ctypes.c_size_t(0),
                                           ctypes.c_uint64(0), ctypes.c_uint64(0)), "cuMemAddressReserve")
        self.base = int(va.value)
        self.handles = {}            # granule -> its physical allocation
        self.host = set()            # the granules backed from host memory, which the device reads over the link
        self.tensor = None
        self.closed = None           # called once when the range closes (the census drops it)
        self.planes = None           # a paged range's planes [(first byte, bytes a unit, units)]
        self.workspace = False       # a growing buffer: nothing in it outlives a call
        # torch reads the pointer's attributes when it wraps it, so the range's start is backed first
        self._map(0, max(int(initial), self.gran))
        if 0 not in self.handles:
            self.close()
            raise MemoryError("elastic range: the device has no memory to back the range's first granule")
        self.tensor = torch.as_tensor(_Interface(self.base, self.size), device=self.device)
        self.tensor[:self.prefix].zero_()

    @property
    def mapped(self):
        """The device bytes backed, wherever they lie."""
        return (len(self.handles) - len(self.host)) * self.gran

    @property
    def host_mapped(self):
        """The bytes of the range backed from host memory."""
        return len(self.host) * self.gran

    @property
    def prefix(self):
        """The bytes backed from the range's start without a gap."""
        g = 0
        while g in self.handles:
            g += 1
        return g * self.gran

    def grow(self, nbytes):
        """Back [0, nbytes) (rounded up to whole granules, at most the range), zeroed; what the device has no memory
        for stays unbacked. Returns the bytes backed from the range's start."""
        self.back(0, nbytes, strict=False)
        return self.prefix

    def back(self, first, end, strict=True, host=False):
        """Back the granules that hold bytes [first, end), zeroed where they were not backed. Returns the bytes newly
        backed. strict: a device out of memory raises MemoryError (what was backed stays); else it backs what it can.
        host: the new granules are host memory the driver pins and maps into the range, so kernels read and write
        them through the range's addresses as they do the rest, at the host link's rate and at no device memory."""
        new = self._map(first, end, host)
        end_granule = -(-min(self.size, int(end)) // self.gran)
        if strict and any(g not in self.handles for g in range(int(first) // self.gran, end_granule)):
            raise MemoryError(f"elastic range: the device has no memory to back {int(end) - int(first):,} more bytes")
        i = 0
        while i < len(new):   # zero each run of new granules
            j = i + 1
            while j < len(new) and new[j] == new[j - 1] + 1:
                j += 1
            if self.tensor is not None:
                self.tensor[new[i] * self.gran:(new[j - 1] + 1) * self.gran].zero_()
            i = j
        return len(new) * self.gran

    def _map(self, first, end, host=False):
        """Back the granules holding [first, end) that are not backed, until the device (host: the host) is out of
        memory: the granules newly backed. Any other failure raises (a thread without the device's context, say),
        never a short range."""
        prop, access = _prop(self.index), _Access(_Location(_LOCATION_DEVICE, int(self.index)), _ACCESS_RW)
        if host:
            prop.location = _Location(_LOCATION_HOST, 0)
        new = []
        for g in range(max(0, int(first)) // self.gran, -(-min(self.size, int(end)) // self.gran)):
            if g in self.handles:
                continue
            h = ctypes.c_uint64()
            code = self.cu.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(self.gran), ctypes.byref(prop),
                                       ctypes.c_uint64(0))
            if code == _OUT_OF_MEMORY:   # keep what was mapped
                break
            _check(code, "cuMemCreate")
            at = ctypes.c_uint64(self.base + g * self.gran)
            code = self.cu.cuMemMap(at, ctypes.c_size_t(self.gran), ctypes.c_size_t(0), h, ctypes.c_uint64(0))
            if code == 0:
                self.handles[g] = int(h.value)
                if host:
                    self.host.add(g)
                code = self.cu.cuMemSetAccess(at, ctypes.c_size_t(self.gran), ctypes.byref(access), ctypes.c_size_t(1))
            else:
                self.cu.cuMemRelease(h)
            _check(code, "mapping a granule")
            new.append(g)
        return new

    def release(self, first, end, closing=False):
        """Give the granules wholly inside [first, end) back to the device; the caller has made sure nothing reads
        them (the device is synchronized first). The range's first granule stays until close(). Returns the bytes
        given back."""
        granules = [g for g in range(-(-max(0, int(first)) // self.gran), min(self.size, int(end)) // self.gran)
                    if g in self.handles and (g or closing)]
        if not granules:
            return 0
        self.torch.cuda.synchronize(self.device)
        for g in granules:
            at = ctypes.c_uint64(self.base + g * self.gran)
            _check(self.cu.cuMemUnmap(at, ctypes.c_size_t(self.gran)), "cuMemUnmap")
            _check(self.cu.cuMemRelease(ctypes.c_uint64(self.handles.pop(g))), "cuMemRelease")
            self.host.discard(g)
        return len(granules) * self.gran

    def shrink(self, nbytes):
        """Give the backing past nbytes (rounded up to whole granules) back to the device; the caller has made sure
        nothing reads it (the device is synchronized first). Returns the bytes backed from the range's start."""
        self.release(-(-int(nbytes) // self.gran) * self.gran, self.size)
        return self.prefix

    def close(self):
        if self.cu and self.base:
            self.release(0, self.size, closing=True)
            self.cu.cuMemAddressFree(ctypes.c_uint64(self.base), ctypes.c_size_t(self.size))
            self.base = 0
            closed, self.closed = self.closed, None
            if closed is not None:
                closed()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Paged:
    """An elastic range of planes whose units are backed as they come into use (the module's description): plane p
    holds `units` units of `per` bytes from byte `first`; `kept`: byte ranges backed from the start on (a dump page,
    a spare sequence). Made by DeviceMemory.paged()."""

    def __init__(self, memory, elastic, planes, kept=(), host_bytes=0):
        self.memory, self.range, self.tensor = memory, elastic, elastic.tensor
        self.host_bytes = int(host_bytes)   # the most of the range that may be backed from host memory
        self.planes = elastic.planes = [(int(first), int(per), int(units)) for first, per, units in planes]
        self.units = max((units for _, _, units in self.planes), default=0)
        self.backed = 0
        for first, end in kept:
            elastic.back(first, end)

    @property
    def bytes(self):
        """The device bytes the range takes now."""
        return self.range.mapped

    def back(self, units, host=False):
        """Back units [0, units) of every plane. The device's givers release what its free memory lacks first (and
        keep free what they keep), so the growth never overcommits the device; False when the device has no memory
        for it: nothing is backed then (a driver that would move the units to system memory is not asked). host:
        the new units come from host memory instead, within the range's host_bytes and what the host has available
        beside HOST_HEADROOM: the device reads them through the range over the link, which is slower and exact.
        False where that does not hold them either. Host work, never captured."""
        units = min(self.units, int(units))
        if units <= self.backed:
            return True
        grow = sum((min(units, most) - min(self.backed, most)) * per for _, per, most in self.planes)
        if host:
            from .host import memory_bytes
            if (self.range.host_mapped + grow > self.host_bytes
                    or memory_bytes()["available"] < grow + HOST_HEADROOM):
                return False
        else:
            self.memory.need(grow)
            if self.memory.free() < grow:
                return False
        for first, per, most in self.planes:
            if per and self.backed < most:
                self.range.back(first + self.backed * per, first + min(units, most) * per, host=host)
        self.backed = units
        return True

    def trim(self, units):
        """Give the backing of the units from `units` on back to the device (whole granules: the nothing-reads-them
        promise is the caller's), and let the givers grow into it. Returns the bytes given back."""
        units = max(0, int(units))
        if units >= self.backed:
            return 0
        freed = 0
        for first, per, most in self.planes:
            if per and units < most:
                freed += self.range.release(first + units * per, first + min(self.backed, most) * per)
        self.backed = units
        if freed:
            self.memory.settle()
        return freed

    def close(self):
        self.range.close()


class Growing:
    """A buffer at a fixed address that holds what its largest user asked for (DeviceMemory.growing): take(nbytes)
    backs [0, nbytes) of its elastic range, the device's givers making room first, and returns that prefix as a
    uint8 tensor. Users of every size read the same bytes, one after another on a stream: a workspace made for
    each call shape (a prompt chunk's size) costs the largest shape's bytes and not their sum, and a pointer a
    captured graph holds stays good while the buffer grows."""

    def __init__(self, memory, elastic):
        self.memory, self.range = memory, elastic
        elastic.workspace = True

    @property
    def bytes(self):
        return self.range.prefix

    def take(self, nbytes):
        nbytes = int(nbytes)
        if nbytes > self.range.size:
            raise ValueError(f"the buffer holds {self.range.size:,} bytes at most, not {nbytes:,}")
        missing = nbytes - self.range.prefix
        if missing > 0:
            self.memory.need(missing)
            self.range.back(0, nbytes)
        return self.range.tensor[:nbytes]

    def close(self):
        self.range.close()


class Scratch:
    """An engine's scratch on a device (the module's description; DeviceMemory.scratch()): take(nbytes) gives the
    first nbytes of one growing buffer, take(a, b, ...) a buffer of each size from its start on, 64-byte aligned.
    Where the device has no elastic ranges a take of each size class is a buffer of its own (fixed addresses too,
    at the sum's cost). poison: fill every take with 0xFF bytes (by default where the environment asks)."""

    def __init__(self, memory, owner="core", poison=None):
        self.memory, self.owner = memory, owner
        self.poison = os.environ.get(POISON) == "1" if poison is None else bool(poison)
        self.buffer, self.fixed, self.most = None, {}, 0   # most: the largest take

    @property
    def bytes(self):
        """The device bytes the scratch takes now."""
        return self.buffer.bytes if self.buffer is not None else sum(self.fixed)

    def take(self, *sizes):
        import torch
        ends, at = [], 0
        for size in sizes:
            at = -(-at // 64) * 64 + int(size)
            ends.append(at)
        capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if self.memory.elastic_ranges:
            if self.buffer is None:
                self.buffer = self.memory.growing(self.owner, "scratch", SCRATCH_BYTES)
            if at > self.buffer.bytes and capturing:
                raise RuntimeError(f"a captured call takes {at:,} bytes of scratch and the engine's has "
                                   f"{self.buffer.bytes:,}: the scratch grows outside a capture, so run the call "
                                   "once at this size before capturing it")
            found = self.buffer.take(at)
        else:
            size = 1 << max(12, (at - 1).bit_length())
            if size not in self.fixed:
                if capturing:
                    raise RuntimeError(f"a captured call takes {at:,} bytes of scratch for the first time: run the "
                                       "call once at this size before capturing it")
                self.fixed[size] = torch.empty(size, dtype=torch.uint8, device=self.memory.device)
            found = self.fixed[size][:at]
        self.most = max(self.most, at)
        if self.poison:
            found.fill_(255)
        if len(sizes) == 1:
            return found
        return [found[end - int(size):end] for size, end in zip(sizes, ends)]

    def close(self):
        buffer, self.buffer, self.fixed = self.buffer, None, {}
        if buffer is not None:
            buffer.close()


class _Torch:
    """A CUDA device's readings through torch: what DeviceMemory reads and releases (a test gives its own)."""

    def __init__(self, index):
        self.index = index
        # The whole segments of captured graphs' own pools that hold nothing: torch's counters list them with the
        # blocks nothing uses, but release() cannot hand them back and no step can take them (a graph's replays
        # write there). Read at every release() and segments().
        self.pooled = 0

    @property
    def total(self):
        import torch
        return int(torch.cuda.get_device_properties(self.index).total_memory)

    def free(self):
        import torch
        return int(torch.cuda.mem_get_info(self.index)[0])

    def torch(self):
        import torch
        stats = torch.cuda.memory_stats(self.index)
        reserved, allocated = stats.get("reserved_bytes.all.current", 0), stats.get("allocated_bytes.all.current", 0)
        unused = reserved - stats.get("active_bytes.all.current", 0) - stats.get("inactive_split_bytes.all.current", 0)
        return {"allocated": int(allocated), "reserved": int(reserved), "unused": max(0, int(unused) - self.pooled)}

    def release(self):
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            self.segments()

    def made(self):
        import torch
        return int(torch.cuda.memory_stats(self.index).get("allocation.all.allocated", 0))

    def mark(self):
        import torch
        torch.cuda.reset_peak_memory_stats(self.index)

    def peak(self):
        import torch
        return int(torch.cuda.max_memory_reserved(self.index))

    def segments(self):
        """The address ranges of torch's allocator on this device, sorted: [(first, end, allocated bytes, whether
        the range is a captured graph's own pool and not the allocator's shared one)]."""
        import torch
        found = sorted((s["address"], s["address"] + s["total_size"], s["allocated_size"],
                        tuple(s.get("segment_pool_id") or (0, 0)) != (0, 0), tuple(s.get("segment_pool_id") or (0, 0)))
                       for s in torch.cuda.memory_snapshot() if s.get("device") == self.index)
        self.pooled = sum(end - first for first, end, allocated, private, _ in found if private and not allocated)
        return found

    def pool(self):
        import torch
        return tuple(torch.cuda.graph_pool_handle())

    def record(self, entries):
        import torch
        torch.cuda.memory._record_memory_history(enabled="all" if entries else None, context="all", stacks="python",
                                                 max_entries=int(entries) or 1)

    def blocks(self):
        """Every block of torch's allocator on this device, sorted by address: (first, end, state, pool, frames).
        state: "active_allocated" (a tensor's), "inactive" (free in its pool: in a graph's pool, the graph's
        temporaries), "active_pending_free"; frames: the stack that made an allocated block, while recording."""
        import torch
        found = []
        for segment in torch.cuda.memory._snapshot()["segments"]:
            if segment.get("device") == self.index:
                pool = tuple(segment.get("segment_pool_id") or (0, 0))
                found.extend((block["address"], block["address"] + block["size"], block["state"], pool,
                              block.get("frames")) for block in segment["blocks"])
        return sorted(found, key=lambda block: block[0])

    def allocations(self, pool, after=0):
        """The allocations made in `pool` that the recorded history still holds, those later than `after` (an
        earlier one's time): [(bytes, frames, time)]."""
        import torch
        traces = torch.cuda.memory._snapshot()["device_traces"]
        return [(event["size"], event.get("frames"), event.get("time_us", 0))
                for event in (traces[self.index] if self.index < len(traces) else ())
                if event["action"] == "alloc" and tuple(event.get("pool_id") or ()) == tuple(pool)
                and event.get("time_us", 0) > after]


class _Unreadable:
    """A device whose memory cannot be read (the CPU, a machine without CUDA): nothing to count, take or give."""
    total = None

    def free(self):
        return None

    def torch(self):
        return None

    def release(self):
        pass

    def made(self):
        return None

    def mark(self):
        pass

    def peak(self):
        return None

    def segments(self):
        return []

    def blocks(self):
        return []

    def pool(self):
        return None


HOST_HEADROOM = 2 << 30   # host memory left available when a range backs units from it
RELIEVE_S = 0.2        # relieve() reads the device at most this often
HOLD_S = 5.0           # after another program's memory grew, the givers wait this long before they grow back
OUTSIDE_SLACK = 32 << 20   # growth of what others hold below this is the process's own (a workspace, a context)


class DeviceMemory:
    """One device's memory (the module's description). of(device) gives the process's object for a device; a test
    makes its own over readings of its own (total, free(), torch(), release(), mark(), peak(), segments())."""

    def __init__(self, device, readings=None):
        self.device = str(device)
        self.readings = readings if readings is not None else _Unreadable()
        self.ranges = {}       # "owner/name" -> its ElasticRange
        self.givers = {}       # name -> (release, regrow, keep)
        self.held = None       # the last census: {"plugins": {plugin: bytes}, "graphs": bytes, "unattributed": bytes}
        self.serving = False   # the engine on this device serves (its load is over): settle() lets the givers grow
        self.pools = {}        # a captured graph's memory pool (torch's id) -> "owner/name"
        self.recorded = None   # while recording: {pool: Counter{(bytes, the code that made it): allocations,
                               # "read": the last one's time}}
        self.clock = time.monotonic
        self.outside = {"relieved": 0, "bytes": 0, "held": 0}   # relieve()'s releases, their bytes, its holds
        self._looked = self._hold = float("-inf")   # relieve()'s last reading; the givers grow back after _hold
        self._others = None    # the least other programs (and the driver) held at a reading
        self._floor = self._seen = None   # what the givers kept free, and the free bytes, at the engine's last look
        self._watch = None     # (thread, its stop event) of watch()

    # -- readings -----------------------------------------------------------------------------------------------------
    @property
    def total(self):
        return self.readings.total

    def free(self, release=False):
        if release:
            self.readings.release()
        return self.readings.free()

    def torch(self):
        return self.readings.torch()

    def release(self, below=None):
        """Hand the blocks torch's allocator holds and nothing uses back to the driver; with `below`, only when the
        device has fewer than that many bytes free (each release waits for the device, and the allocations after
        it go back to the driver). Returns the bytes released, None where the device's memory cannot be read."""
        before = self.readings.torch()
        if before is None:
            return None
        if below is not None and self.readings.free() >= below:
            return 0
        self.readings.release()
        return max(0, before["reserved"] - self.readings.torch()["reserved"])

    def made(self):
        """The allocations torch's allocator has made on this device so far (a count; None where it cannot be
        read). A captured graph's replay makes none: its temporaries were allocated when it was captured."""
        made = getattr(self.readings, "made", None)
        return made() if made is not None else None

    def mark(self):
        """From here on peak() is the most torch's allocator held on this device (its reserved bytes). One mark a
        device: whoever marks next moves it."""
        self.readings.mark()

    def peak(self):
        return self.readings.peak()

    # -- elastic ranges -----------------------------------------------------------------------------------------------
    @property
    def elastic_ranges(self):
        """Whether elastic ranges work here: a CUDA device whose driver has the virtual memory API."""
        return isinstance(self.readings, _Torch) and granularity(self.readings.index) > 0

    def elastic(self, owner, name, max_bytes, initial=0):
        """An elastic range of `max_bytes` of address space for `owner` (a plugin's name), listed in the census as
        owner/name (numbered when the owner makes several of a name) until it is closed."""
        made = ElasticRange(max_bytes, self.device, initial)
        key, n = f"{owner}/{name}", 1
        while key in self.ranges:
            n += 1
            key = f"{owner}/{name}#{n}"
        self.ranges[key] = made
        made.closed = lambda: self.ranges.pop(key, None)
        return made

    def paged(self, owner, name, nbytes, planes, kept=(), host_bytes=0):
        """An elastic range of `nbytes` whose planes' units are backed as they come into use (Paged): planes
        [(first byte, bytes a unit, units)], kept [(first, end)] byte ranges backed from the start on, host_bytes
        the most of it that may be backed from host memory when the device has none (Paged.back's host)."""
        return Paged(self, self.elastic(owner, name, nbytes), planes, kept, host_bytes)

    def growing(self, owner, name, max_bytes):
        """An elastic range of `max_bytes` of address space that holds what its largest user asked for (Growing)."""
        return Growing(self, self.elastic(owner, name, max_bytes))

    def scratch(self, owner="core", poison=None):
        """A scratch on this device (Scratch): Core's loader makes an engine's and closes it with the engine; code
        that runs without an engine (a test, a bench) makes its own."""
        return Scratch(self, owner, poison)

    # -- captured graphs' pools ---------------------------------------------------------------------------------------
    def pool(self, owner, name, handle=None):
        """A memory pool of its own for captured graphs (torch's handle, for torch.cuda.graph(pool=)), listed as
        owner/name in the device's readings; `handle` names a pool made elsewhere (a torch.cuda.MemPool's id)."""
        handle = tuple(handle) if handle is not None else self.readings.pool()
        if handle is not None:
            self.pools[handle] = f"{owner}/{name}"
        return handle

    @contextmanager
    def capturing(self, pool):
        """A graph is captured into `pool` inside the block. While the device records (record()), the allocations
        the capture made there are kept by the code that made them (`recorded`): once a graph is captured its
        temporaries are free blocks of its pool, and nothing else says what they are."""
        try:
            yield
        finally:
            if self.recorded is not None:   # the pool's allocations since its last capture
                found = self.recorded.setdefault(tuple(pool), Counter())
                for nbytes, frames, at in self.readings.allocations(pool, found.get("read", 0)):
                    found[(int(nbytes), where(frames))] += 1
                    found["read"] = max(found["read"], at)

    def record(self, entries=RECORDED):
        """From now on torch's allocator notes the code that makes each allocation here (its stack, the last
        `entries` events): blocks() names every allocated block by it and capturing() every capture's temporaries.
        Each allocation then costs a stack walk, so only a tool turns it on (the memory audit), before the engine
        loads; record(0) ends it."""
        self.recorded = {} if entries else None
        self.readings.record(entries)

    def blocks(self):
        """Every block of torch's allocator here, sorted by address: (first, end, state, pool, frames)."""
        return self.readings.blocks()

    # -- the ledger ---------------------------------------------------------------------------------------------------
    def give(self, name, release, regrow=None, keep=0):
        """A giver: release(nbytes) frees at least nbytes if it can and returns the bytes it freed; regrow(room) may
        take up to room bytes back and returns the bytes it took; keep: the free bytes it leaves the device (a
        number, or a callable giving it now)."""
        self.givers[name] = (release, regrow, keep if callable(keep) else int(keep))

    def ungive(self, name):
        self.givers.pop(name, None)

    def need(self, nbytes):
        """Make sure `nbytes` of device memory can be taken with what the givers keep free still free: the givers
        release what is missing, and torch's allocator hands back the blocks it holds and nothing uses if that was
        not enough. Returns the bytes the givers released (0 when the device had room). Call it before an
        allocation, outside a capture."""
        free = self.readings.free()
        if free is None:
            return 0
        givers = list(self.givers.values())
        kept = max((_kept(keep) for _, _, keep in givers), default=0)
        missing = int(nbytes) + kept - free
        released = 0
        for release, _, _ in givers:
            if missing <= released:
                break
            released += int(release(missing - released) or 0)
        if missing > released:
            self.readings.release()
        self._floor, self._seen = kept, (free + released if released else free)
        return released

    def _other(self):
        """What neither torch's allocator nor an elastic range of this process holds and is not free: the driver's
        and other programs'. None where the device cannot be read."""
        state, free = self.readings.torch(), self.readings.free()
        if state is None or free is None:
            return None
        return max(0, self.readings.total - free - state["reserved"] - sum(r.mapped for r in self.ranges.values()))

    def relieve(self):
        """Between an engine's steps, outside a capture: where another program left less free than the givers
        keep, they release the difference now. Reads the device at most every RELIEVE_S and does nothing while
        the engine loads. When what others hold grew since the last look, settle() lets no giver grow back for
        HOLD_S. Returns the bytes released."""
        now = self.clock()
        if not self.serving or not self.givers or now - self._looked < RELIEVE_S:
            return 0
        self._looked = now
        other = self._other()
        if other is None:
            return 0
        grew = self._others is not None and other - self._others > OUTSIDE_SLACK
        self._others = other if self._others is None or grew else min(self._others, other)
        released = self.need(0)
        if released:
            self.outside["relieved"] += 1
            self.outside["bytes"] += released
            if grew:
                self.outside["held"] += 1
                self._hold = now + HOLD_S
        return released

    def settle(self):
        """The device is quiet: each giver may regrow into the free memory less what it keeps free. Returns the
        bytes taken; nothing while the engine still loads (serve() ends the load), and nothing for HOLD_S after
        relieve() gave memory to another program."""
        taken = 0
        if self.serving and self.clock() >= self._hold:
            for _, regrow, keep in list(self.givers.values()):
                free = self.readings.free()
                room = 0 if free is None else free - _kept(keep)
                if regrow is not None and room > 0:
                    taken += int(regrow(room) or 0)
            if self.givers:
                self._floor = max(_kept(keep) for _, _, keep in self.givers.values())
                self._seen = self.readings.free()
        return taken

    def watch(self, ask, every=RELIEVE_S, also=None):
        """A thread of Core's that calls ask() when another program left the device less free memory than the
        givers keep: every `every` seconds it reads the free bytes and compares them with what the engine's thread
        last saw (what the givers kept then, and the bytes free then; it calls nothing of a giver's). ask() makes
        the engine run a step of its own, where relieve() gives the memory back; it is called once per fall, not
        while the device stays short. also: another reason the engine must run a step of its own, a callable read
        each time (a host handoff that failed: the engine's next step ends it). unwatch() ends the thread. One
        watch a device."""
        self.unwatch()
        stop = threading.Event()

        def run():
            asked = None
            while not stop.wait(every):
                if also is not None and also():
                    ask()
                floor, seen = self._floor, self._seen
                if not self.serving or not self.givers or floor is None or seen is None:
                    continue
                try:
                    free = self.readings.free()
                except Exception:   # the device went with its engine
                    continue
                if free is None or free >= floor or free >= seen:
                    asked = None            # room again, or as short as the engine left it
                elif asked is None or free < asked:
                    asked = free
                    ask()
        thread = threading.Thread(target=run, name="crit-memory-watch", daemon=True)
        self._watch = (thread, stop)
        thread.start()

    def unwatch(self):
        if self._watch is not None:
            thread, stop = self._watch
            self._watch = None
            stop.set()
            thread.join()

    def load(self):
        """An engine starts to load on this device: until serve(), what its plugins leave free is the load's."""
        self.serving, self.held = False, None

    def serve(self):
        """The load is over (Core calls it once the warm requests ran): torch's unused blocks go back to the driver
        and the givers take what is free, less what they keep. Returns the bytes they took."""
        self.readings.release()
        self.serving = True
        self._hold, self._others = float("-inf"), self._other()
        return self.settle()

    # -- the census ---------------------------------------------------------------------------------------------------
    def holdings(self, plugins):
        """Every device tensor that can be reached from an engine's plugins, each once: (plugin, name, tensor).
        plugins: {plugin id: its object} in creation order. The walk starts at the plugin objects and then at the
        globals of the plugins' own modules (a buffer a module keeps for every object of its plugin). A tensor is
        the plugin's whose code made the object holding it (the object's class lies in that plugin's package; an
        object of any other class belongs to whoever holds it), and its name says where it is held: the nearest
        object's class, the attribute and the keys below it ("StatePool.windows", "Matrix.tensors[weight]")."""
        from .observe import PLUGIN_MODULES, _entries, unwrap
        torch = sys.modules["torch"]
        packages = {(type(made).__module__ or "").split(".")[0]: plugin for plugin, made in plugins.items()}
        packages = {package: plugin for package, plugin in packages.items() if package.startswith(PLUGIN_MODULES)}
        queue = deque((made, plugin, "", False) for plugin, made in plugins.items())
        for name, module in sorted(sys.modules.items()):
            if module is not None and name.split(".")[0] in packages:
                queue.append((vars(module), packages[name.split(".")[0]], name.rpartition(".")[2], True))
        seen = set()
        while queue:
            holder, owner, name, attributes = queue.popleft()
            holder = unwrap(holder)   # past the loader's counting proxy, to the plugin's own object
            if id(holder) in seen or isinstance(holder, _SCALARS):
                continue
            seen.add(id(holder))
            if isinstance(holder, torch.Tensor):
                if holder.is_cuda:
                    yield owner, name, holder
                continue
            if isinstance(holder, (type, _MODULE)):
                continue
            plain = isinstance(holder, _CONTAINERS)
            if not plain:
                owner = packages.get((type(holder).__module__ or "").split(".")[0], owner)
                name = type(holder).__name__
            try:
                entries = _entries(holder)
            except Exception:  # noqa: BLE001 - an object whose attributes cannot be read holds nothing countable
                continue
            for key, value, _ in entries:
                if isinstance(value, _SCALARS):
                    continue
                if not plain and key is None and isinstance(value, dict):   # the object's own attributes
                    queue.append((value, owner, name, True))
                elif isinstance(key, (str, int)) and not isinstance(key, bool):
                    queue.append((value, owner, f"{name}.{key}" if attributes or not plain else f"{name}[{key}]", False))
                else:
                    queue.append((value, owner, name, False))

    def census(self, plugins):
        """Who holds torch's bytes on this device: {"plugins": {plugin: bytes}, "tensors": [...], "graphs": bytes,
        "pools": {owner/name: bytes}, "unattributed": bytes}, kept as `held` for report(). plugins: {plugin id: its object} in creation order.
        tensors: the largest storages (LARGEST of them, largest first), each {"plugin", "bytes", "shape", "dtype"}
        (the shape and dtype of the tensor it was reached by), so a table or a pool that could live elsewhere is
        seen by its shape and not only as a share of its plugin's total. Each live tensor
        storage of torch's allocator here that can be reached from a plugin (holdings) counts once, for the plugin
        whose code made the object holding it. graphs: the device memory of captured graphs' own pools beyond those
        tensors: torch keeps a pool for its graph's replays (the temporaries of the rounds and prompt chunks they
        replay live at its addresses), in use or not, so it is part of torch's reserved bytes that nothing else can
        take; pools: each pool's whole bytes by the name its driver gave it (pool()). unattributed: what torch has
        allocated beyond the plugins' tensors and the pools' (tensors no plugin object holds). None where the
        device's memory cannot be read."""
        state = self.readings.torch()
        if state is None:
            self.held = None
            return None
        segments = [(tuple(s) + (0, False))[:4] for s in self.readings.segments()]
        firsts = [s[0] for s in segments]
        pools = sum(end - first for first, end, _, private in segments if private)       # the graphs' pools
        pooled = sum(allocated for _, _, allocated, private in segments if private)      # what is allocated in them
        named = Counter()
        for found in self.readings.segments():
            if len(found) > 4 and found[3]:
                named[self.pools.get(found[4], "unnamed")] += found[1] - found[0]
        held, storages, largest = Counter(), set(), []
        for owner, _, tensor in self.holdings(plugins):
            storage = tensor.untyped_storage()
            at = storage.data_ptr()
            i = bisect_right(firsts, at) - 1
            if at and at not in storages and i >= 0 and at < segments[i][1]:
                storages.add(at)
                held[owner] += storage.nbytes()
                largest.append((storage.nbytes(), owner, tuple(tensor.shape), str(tensor.dtype)))
                if segments[i][3]:   # made during a capture and kept: the plugin's, not the graph's
                    pools, pooled = pools - storage.nbytes(), pooled - storage.nbytes()
        largest = sorted(largest, key=lambda found: -found[0])[:LARGEST]
        self.held = {"plugins": {plugin: int(held[plugin]) for plugin in plugins if held[plugin]},
                     "tensors": [{"plugin": plugin, "bytes": int(nbytes), "shape": list(shape),
                                  "dtype": dtype.removeprefix("torch.")} for nbytes, plugin, shape, dtype in largest],
                     "graphs": max(0, int(pools)), "pools": dict(named),
                     "unattributed": max(0, state["allocated"] - sum(held.values()) - max(0, int(pooled)))}
        return self.held

    def report(self):
        """The device now, for an engine's telemetry: {"device", "total", "free", "torch": {allocated, reserved,
        unused}, "elastic": {owner/name: {"backed", "reserved"}}, "other"} and, after a census, "plugins", "tensors",
        "graphs", "pools" and "unattributed". other: what neither torch's allocator nor an elastic range of this process
        holds and is not free (the CUDA context, other processes, a desktop). None where the device's memory cannot
        be read."""
        state = self.readings.torch()
        if state is None:
            return None
        elastic = {key: {"backed": made.mapped, "reserved": made.size,
                         **({"host": made.host_mapped} if made.host_mapped else {})}
                   for key, made in self.ranges.items()}
        free, total = self.readings.free(), self.readings.total
        found = {"device": self.device, "total": total, "free": free, "torch": state, "elastic": elastic,
                 "other": max(0, total - free - state["reserved"] - sum(e["backed"] for e in elastic.values()))}
        if self.outside["relieved"]:
            found["outside"] = dict(self.outside)
        if self.held is not None:
            found.update(self.held)
        return found


LARGEST = 8   # the storages a census names one by one


def where(frames, most=4):
    """The code that made an allocation, from its recorded stack: its innermost frames outside torch, innermost
    first, each "folder/file.py:line function"; "" without a stack."""
    own = [frame for frame in frames or () if "/torch/" not in frame["filename"].replace("\\", "/")]
    return " < ".join("%s:%d %s" % ("/".join(frame["filename"].replace("\\", "/").split("/")[-2:]), frame["line"],
                                    frame["name"]) for frame in own[:most])
_SCALARS = (type(None), bool, int, float, complex, str, bytes, bytearray, range, memoryview)
_CONTAINERS = (dict, list, deque, tuple, set, frozenset)
_MODULE = type(sys)
_DEVICES = {}


def _kept(keep):
    return int(keep() if callable(keep) else keep)


def of(device="cuda"):
    """The process's DeviceMemory for `device` (a name or a torch.device): one object a CUDA device, whoever asks;
    an object that reads nothing and gives nothing for any other device."""
    name = str(device)
    if name.startswith("cuda"):
        torch = sys.modules.get("torch")
        if torch is None:
            import torch
        if torch.cuda.is_available():
            index = torch.device(name).index
            if index is None:   # "cuda": the current device, without initializing CUDA for the question
                index = torch.cuda.current_device() if torch.cuda.is_initialized() else 0
            if index not in _DEVICES:
                _DEVICES[index] = DeviceMemory(f"cuda:{index}", _Torch(index))
            return _DEVICES[index]
    return _DEVICES.setdefault(name, DeviceMemory(name))


# -- the process's torch memory, every CUDA device together: what Core's loader holds an engine's plans to ------------
def device_bytes(device="cuda"):
    """The bytes torch holds allocated on every CUDA device of the process, or None where Core cannot read them (another
    device, no CUDA, torch not loaded by a plugin). It never initializes CUDA."""
    torch = sys.modules.get("torch")
    if device != "cuda" or torch is None or not torch.cuda.is_available():
        return None
    return sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))


def device_memory(device="cuda"):
    """{"allocated": bytes, "reserved": bytes} torch holds on every CUDA device of the process (reserved: what its
    caching allocator took from the device, allocated within it), or None where they cannot be read, as device_bytes.
    It never initializes CUDA."""
    torch = sys.modules.get("torch")
    if device != "cuda" or torch is None or not torch.cuda.is_available():
        return None
    devices = range(torch.cuda.device_count())
    return {"allocated": sum(torch.cuda.memory_allocated(i) for i in devices),
            "reserved": sum(torch.cuda.memory_reserved(i) for i in devices)}


def release_cached(device="cuda"):
    """Hand the device back what torch's caching allocator holds and nothing uses (a load's temporaries: checkpoint
    conversions, warm-ups), where torch is loaded and the device is CUDA: returns the bytes released, or None where the
    device's memory cannot be read. Live tensors stay where they are."""
    before = device_memory(device)
    if before is None:
        return None
    torch = sys.modules["torch"]
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    return max(0, before["reserved"] - device_memory(device)["reserved"])
