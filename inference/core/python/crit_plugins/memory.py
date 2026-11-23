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

    memory.give("expert-offload", release=lambda n: freed_bytes, regrow=lambda room: taken_bytes, rank=COPIES)
    memory.need(nbytes)          # before the allocation, outside a capture
    memory.settle()              # the device is quiet: each giver regrows into free memory, less what it keeps;
                                 # for SETTLE_S at most, and memory.owed says more is left for a later settle
    memory.relieve()             # between steps: another program took memory, the givers hand theirs back now

Every byte an engine's steps take is Core's to hand out: Core's step engine runs each step inside the device's
step pool (step_pool(): a torch memory pool whose segments come from Core's allocator, allocator.cpp through torch's
pluggable allocator), which makes a segment where the device has the bytes free and logs it, and the ledger counts
the log at every step frame (natively: no Python runs inside torch's allocation). What the givers keep free is
the device's to say, not a giver's: the headroom (memory.headroom, the run's VRAM headroom, which Core's loader sets
from the configuration's vram_headroom_bytes) and the step arena (memory.arena: the most a step took while the
engine loaded, whose warm requests run the largest prompt and score steps, or the model's own word for its deepest
step, set once by the loader), less what the steps hold of
it now. A step's temporaries live in the arena, which is free whenever no step holds it: nothing is forecast,
nothing is looked at mid-step, and no plugin asks for room before a temporary. A segment past the arena is lent
from free memory and counted (overflows), one that leaves less than the headroom free is counted (breaches), one the
device does not have is refused (torch raises its out-of-memory error): the serving checks read the counts. Once no
prompt is in flight, or a long prompt's next chunk could not take a chunk's most from the pool within the arena,
the step pool retires and a new one takes its place, so its segments go back to the driver and each prompt's are
its own; a tensor a step made and a plugin keeps (a workspace made at its first call) holds
its segment, counted as the engine's (retired). A giver only releases and regrows; its own keep, where it gives
one, only adds to the device's.

The givers form one eviction order, the device's, not a model's admission's: each has a rank (give(..., rank=)),
and need() asks the lowest first. IDLE holds what nothing in flight needs (a prompt kept past its use: the prompt
cache's sequences no prompt reused for a minute), COPIES holds copies whose loss costs speed (an expert cache's
slots), WORK holds work a later request reuses (a prompt kept for a conversation's next turn). A giver that grows back
says whether it would take more (hungry=): where a settle finds no room for it, the IDLE givers give it a slice, so a
long prompt kept past its use does not hold memory an expert cache's decode is short of. Each giver's releases and
regrowth are counted (givings(), in the device's report), and a caller reads a request's share as the difference of
two reports.

    memory.headroom = nbytes     # Core's loader, before any plugin is made
    with memory.step_pool():     # Core's step engine, around each step: the step's tensors from Core's allocator
        ...
    step = memory.measure(run, "prompt")   # a plugin's step at load no warm request reaches (a calibration), its
                                 # room made by the givers where it ran out; step.took: what it took from Core's
                                 # allocator, counted in the arena (measured(): the same frame as a with block)
    memory.size_arena(declared)  # Core's loader, once the warm requests ran
    memory.steps()               # {headroom, arena, granted, retired, most, steps, refused, overflows, breaches, ...}

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

A test may take a share of the device away (a ballast), so that an engine loads and serves as it would on a device
that much smaller, or beside another program that holds it: an elastic range owned by "ballast", backed from the
device and never given back, which every reading of free memory (this object's, the driver's, the card's) counts as
taken, as it counts another program's memory. CRIT_VRAM_BALLAST makes one in each process when the process first
asks for the device's object: "leave:SIZE" holds all but SIZE of what the device has free then, "take:SIZE" holds
SIZE (SIZE in bytes, or with KiB, MiB or GiB). memory.ballast(take=None, leave=None) makes one in code, and its
close() gives the bytes back.

A captured graph's temporaries live in a memory pool of its own, which torch keeps for the graph's replays.
pool(owner, name) makes one and lists it as owner/name, so the device's readings say which graph holds what; a
driver captures inside capturing(pool). The steps' own pools (step_pool()) are listed as core/steps. After record(), the device also notes which code made each allocation:
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
BALLAST = "CRIT_VRAM_BALLAST"   # "leave:SIZE" or "take:SIZE": a test's share of the device taken away (ballast)
_UNITS = {"": 1, "b": 1, "kib": 1 << 10, "mib": 1 << 20, "gib": 1 << 30}


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

    def unbacked(self, spans):
        """The bytes backing `spans` ([(first, end)] byte ranges) would map: the whole granules holding them that are
        not backed yet, each counted once."""
        granules = set()
        for first, end in spans:
            granules.update(range(max(0, int(first)) // self.gran, -(-min(self.size, int(end)) // self.gran)))
        return len(granules - set(self.handles)) * self.gran

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

        def spans():
            return [(first + self.backed * per, first + min(units, most) * per)
                    for first, per, most in self.planes if per and self.backed < most]
        # what backing them maps: whole granules a plane, so up to one more a plane than the units' own bytes
        grow = self.range.unbacked(spans())
        if host:
            from .host import memory_bytes
            if (self.range.host_mapped + grow > self.host_bytes
                    or memory_bytes()["available"] < grow + HOST_HEADROOM):
                return False
        else:
            self.memory.need(grow)
            # A giver may have given from this range itself (a prompt cache closing kept sequences of this pool, which
            # trims its backing): what is left to back is measured again from where the backing now ends.
            grow = self.range.unbacked(spans())
            if self.memory.free() < grow:
                return False
        for first, end in spans():
            self.range.back(first, end, host=host)
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


_NVML = {}   # the management library and each device's handle, found once (None: not there)


class _NvmlMemory(ctypes.Structure):   # nvmlMemory_v2_t
    _fields_ = [("version", ctypes.c_uint), ("total", ctypes.c_uint64), ("reserved", ctypes.c_uint64),
                ("free", ctypes.c_uint64), ("used", ctypes.c_uint64)]


def card_free(index):
    """The bytes CUDA device `index` has left with every program's memory and the driver's reserve taken off, as
    the card's management library counts them (NVML, which nvidia-smi reads: through ctypes, the device found by
    its UUID); None where it cannot be asked."""
    if "lib" not in _NVML:
        _NVML["lib"] = None
        for name in ("libnvidia-ml.so.1", "nvml.dll"):
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            if lib.nvmlInit_v2() == 0:
                _NVML["lib"] = lib
            break
    lib = _NVML["lib"]
    if lib is None:
        return None
    if index not in _NVML:
        handle = ctypes.c_void_p()
        found = 1
        try:
            import torch
            uuid = ("GPU-" + str(torch.cuda.get_device_properties(index).uuid)).encode()
            found = lib.nvmlDeviceGetHandleByUUID(ctypes.c_char_p(uuid), ctypes.byref(handle))
        except Exception:   # a torch without the property: the card of the same number
            found = 1
        if found != 0 and lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle)) != 0:
            handle = None
        _NVML[index] = handle
    handle = _NVML[index]
    if handle is None:
        return None
    info = _NvmlMemory(version=ctypes.sizeof(_NvmlMemory) | (2 << 24))
    try:
        if lib.nvmlDeviceGetMemoryInfo_v2(handle, ctypes.byref(info)) == 0:
            return int(info.free)
    except AttributeError:           # an older library
        pass
    old = (ctypes.c_uint64 * 3)()    # total, free, used
    if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(old)) != 0:
        return None
    return int(old[1])


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
        """The bytes this process may still take: the driver's reading, and never more than the card has left. On
        some drivers (Windows, and Linux under WSL) the driver's reading is this process's own budget and does not
        fall when another program takes memory, so what the card has left is read from the card as well (NVML,
        card_free) and the lesser is the answer; where the card cannot be read, the driver's reading alone."""
        import torch
        free, left = int(torch.cuda.mem_get_info(self.index)[0]), card_free(self.index)
        return free if left is None else min(free, left)

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
# A settle grows the givers back for at most this long, SETTLE_SLICE bytes at a time: backing device memory costs
# the driver time by the granule (half a millisecond each under WSL, a second for two gigabytes), a settle runs
# inside a step's frames, and a step must end inside the server's cancel grace. What is left is owed: the next
# settles go on, and Core's watch asks the engine for a step of its own while memory is owed.
SETTLE_S = 0.05
SETTLE_SLICE = 64 << 20
HOLD_S = 5.0           # after another program's memory grew, the givers wait this long before they grow back
# The device's one eviction order: a giver's rank, the lowest asked first. IDLE holds what nothing in flight needs and
# no request is likely to (a prompt kept past its use): it gives first, and also to a giver that grows back (settle);
# COPIES holds copies whose loss costs speed only (an expert cache's slots); WORK holds work a later request reuses
# (a prompt kept for its conversation's next turn), given last.
IDLE, COPIES, WORK = 0, 1, 2
OUTSIDE_SLACK = 32 << 20   # growth of what others hold below this is the process's own (a workspace, a context)
SEGMENT_SAMPLES = 4096     # the serving segments whose (free left, outside) readings steps() reports, the first ones


# The headroom: device memory the givers leave free at every moment, the run's VRAM headroom (crit Eval's device
# memory checks read it at a step's peak). Core's loader sets it from the configuration's vram_headroom_bytes, which
# crit run writes into every candidate; HEADROOM where none is named: the one default (0.25 GiB), which crit run,
# crit pull, crit bench and crit Eval take from here when settings.toml names no vram_headroom_gib.
HEADROOM = 256 << 20


def kv_fit(free_bytes, kv_bytes_per_token, *, headroom=HEADROOM, reserve=0, held=(), granularity=1):
    """What fits the device: {"budget": the bytes the KV cache may take (free_bytes less the headroom, the reserve
    and each amount of `held`, in that order; 0 or less where nothing fits), "tokens": the KV positions that fit in
    it, rounded down to a multiple of granularity (0 where the budget is 0 or less; None where nothing grows with
    the context, kv_bytes_per_token 0 or None)}. free_bytes: the device's free memory; headroom: what stays free at
    every moment (the run's VRAM headroom, HEADROOM where none is named); reserve: what the engine's steps take
    beyond weights and KV (activations, workspaces, graphs); held: what stays on the device whatever the context
    (the weights read on every pass, the state that does not grow with it, companion drafters). The one arithmetic
    of the KV cache that fits: crit Eval's max context, from which crit run sizes a run, calls it."""
    budget = free_bytes - headroom - reserve
    for amount in held:
        budget -= amount
    if not kv_bytes_per_token:
        return {"budget": budget, "tokens": None}
    return {"budget": budget, "tokens": 0 if budget <= 0 else budget // kv_bytes_per_token // granularity * granularity}
# The step arena: the device bytes an engine's steps take for their temporaries, kept free beside the headroom while
# nothing uses them. Core's loader sets it once the warm requests ran: the most a step took then, or a model's own
# word for its deepest step (info's step_bytes), whichever is more, ARENA_SLACK beyond. A step that takes more is let
# have it where the device has the memory, and counted (overflows); one that leaves less than the headroom free is
# counted too (breaches): the serving checks fail on either, so an arena too small cannot hide.
ARENA_SLACK = 1.125
ALLOCATOR_SOURCE = "allocator.cpp"
MEASURE_ATTEMPTS = 4   # a measured step that ran out of device memory runs again this often, the givers making room
STEP_POOLS_NAMED = 64  # step pools named in `pools` before the names of those holding nothing any more go


def _out_of_memory(error):
    """Whether `error` is torch's out-of-memory error (a segment Core's allocator refused)."""
    import sys
    torch = sys.modules.get("torch")
    kind = getattr(torch, "OutOfMemoryError", None) if torch is not None else None
    return (kind is not None and isinstance(error, kind)) or "out of memory" in str(error).lower()


class _Segment(ctypes.Structure):   # allocator.cpp's Event: one entry of a device's log
    _fields_ = [("kind", ctypes.c_int32), ("reserved", ctypes.c_int32), ("bytes", ctypes.c_uint64),
                ("free", ctypes.c_uint64)]


MADE, REFUSED, FREED = 1, 2, 3       # a segment made, refused or freed (allocator.cpp's Kind)
_UNKNOWN = (1 << 64) - 1             # a free reading the allocator could not take
_EVENTS = 256                        # the log's entries read at a time


class _Allocator:
    """Core's device allocator, one a process: the library torch's pluggable allocator calls for a new segment of an
    engine's step pool (allocator.cpp). It decides and logs natively and calls no Python: a Python hook inside
    torch's allocation needs the GIL under torch's allocator lock, and a thread that holds the GIL and reads torch's
    allocator (torch.cuda.memory_stats) then waits for that lock forever. Each device's ledger reads the log
    (events) at its step frames."""

    _made = None

    def __init__(self):
        import torch
        from pathlib import Path
        from .native import library
        self.library = library("crit_alloc", [Path(__file__).with_name(ALLOCATOR_SOURCE)])
        self.library.crit_alloc_card.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self.library.crit_alloc_card.restype = None
        self.library.crit_alloc_events.argtypes = [ctypes.c_int, ctypes.POINTER(_Segment), ctypes.c_int]
        self.library.crit_alloc_events.restype = ctypes.c_int
        self.pluggable = torch.cuda.memory.CUDAPluggableAllocator(self.library._name, "crit_alloc", "crit_free")

    def card(self, index):
        """The allocator makes a segment on device `index` only where the card has the bytes left too: the card's
        NVML handle, found here (torch is asked for the device's UUID, outside any allocation)."""
        card_free(index)
        handle = _NVML.get(index)
        self.library.crit_alloc_card(int(index), handle.value if handle is not None else None)

    def events(self, index):
        """The segments made, refused and freed on device `index` since the last call, oldest first: [(kind, bytes,
        free)], free None where the allocator could not read it (a freed segment's)."""
        found, buffer = [], (_Segment * _EVENTS)()
        while True:
            n = self.library.crit_alloc_events(int(index), buffer, _EVENTS)
            found.extend((int(e.kind), int(e.bytes), None if e.free == _UNKNOWN else int(e.free)) for e in buffer[:n])
            if n < _EVENTS:
                return found

    @classmethod
    def get(cls):
        """The process's allocator, made at its first use; None where it cannot be built here (no compiler: the
        steps then allocate through torch's own, unaccounted, and the device's report says so)."""
        if cls._made is None:
            try:
                cls._made = cls()
            except Exception as error:   # noqa: BLE001 - NativeUnavailable, a torch without the API
                cls._made = error
        return None if isinstance(cls._made, Exception) else cls._made


class Measured:
    """What DeviceMemory.measured() measured: `took`, the bytes the step's temporaries took from Core's allocator."""

    took = 0


class DeviceMemory:
    """One device's memory (the module's description). of(device) gives the process's object for a device; a test
    makes its own over readings of its own (total, free(), torch(), release(), mark(), peak(), segments())."""

    def __init__(self, device, readings=None):
        self.device = str(device)
        self.readings = readings if readings is not None else _Unreadable()
        self.ranges = {}       # "owner/name" -> its ElasticRange
        self.givers = {}       # name -> (release, regrow, keep)
        self.ranks = {}        # name -> its rank in the eviction order (a number, or a callable giving it now)
        self.given = {}        # name -> {"released", "regrown", "releases"}: what each giver gave and took back
        self._needing = False  # inside need(): the givers release, nothing settles
        self.hungers = {}      # name -> a callable: whether that giver would take more than it holds
        self.held = None       # the last census: {"plugins": {plugin: bytes}, "graphs": bytes, "unattributed": bytes}
        self.serving = False   # the engine on this device serves (its load is over): settle() lets the givers grow
        self.owed = False      # a settle ran out of time with free memory the givers may still take
        self.pools = {}        # a captured graph's memory pool (torch's id) -> "owner/name"
        self.recorded = None   # while recording: {pool: Counter{(bytes, the code that made it): allocations,
                               # "read": the last one's time}}
        self.clock = time.monotonic
        self.outside = {"relieved": 0, "bytes": 0, "held": 0}   # relieve()'s releases, their bytes, its holds
        self._looked = self._hold = float("-inf")   # relieve()'s last reading; the givers grow back after _hold
        self._others = None    # the least other programs (and the driver) held at a reading
        self._floor = self._seen = None   # what the givers kept free, and the free bytes, at the engine's last look
        self._watch = None     # (thread, its stop event) of watch()
        # The steps' memory (Core's allocator): the headroom, the arena, the step pool and what its segments hold.
        self.headroom = HEADROOM
        self.arena = 0         # the bytes the steps' temporaries may take, kept free while nothing uses them
        self.granted = 0       # the bytes of the step pool's live segments, the current pool's and the retired ones'
        self.retired = 0       # of them, those of pools retired with tensors still alive (the engine's to keep)
        self._pool = None      # the current step pool (torch.cuda.MemPool over Core's allocator), None: not made yet
        self._pool_bytes = 0   # the current pool's live segments
        self._step_pools = []  # the step pools named core/steps in `pools`, oldest first (_name_step_pool)
        self._step = None      # the step in flight: [kind, the current pool's bytes at its start, its most]
        self.most = {}         # {"prompt", "score", "decode": the most bytes one step of that kind took}
        self.counts = {"steps": 0, "refused": 0, "overflows": 0, "breaches": 0, "least_free": None}
        # each segment made while serving: (the free bytes it left, the memory outside the engine then: other()), and
        # that outside memory as the engine began serving; a check judges the engine's own use by them
        self.segment_memory, self.segment_memory_cut, self.outside_at_serve = [], False, None
        self.last_refused = None   # the bytes of the last segment Core's allocator refused (measure())
        self.loaded = None     # the load's counts once the engine serves (serve())
        self.unaccounted = None   # why steps allocate through torch's own allocator here (no Core allocator)
        self._unsettled = False   # a prompt step ended and the givers have not grown back since
        self._allocator = None    # Core's allocator once a step pool was made here: its log is this ledger's
        self._draining = threading.Lock()   # one thread counts the log at a time, in its order

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

    # -- a ballast: a test's share of the device taken away ---------------------------------------------------------------
    def ballast(self, take=None, leave=None):
        """An elastic range owned by "ballast" holding `take` bytes, or all but `leave` of the bytes free now (whole
        granules), which no giver is asked for and every free-memory reading counts as taken. Its close() gives them
        back. Refused where the device has no elastic ranges or too little free."""
        if (take is None) == (leave is None):
            raise ValueError("a ballast takes `take` bytes or leaves `leave` bytes free")
        if not self.elastic_ranges:
            raise ValueError(f"a ballast is an elastic range, which {self.device} does not have")
        free = self.free(release=True)
        nbytes = int(take) if take is not None else free - int(leave)
        if nbytes <= 0 or nbytes > free:
            raise ValueError(f"a ballast of {nbytes / 2**30:.2f} GiB on a device with {free / 2**30:.2f} GiB free")
        found = self.elastic("ballast", "test", nbytes, initial=nbytes)
        if found.mapped < nbytes - found.gran:
            found.close()
            raise MemoryError(f"the device backed {found.mapped / 2**30:.2f} of the ballast's {nbytes / 2**30:.2f} GiB")
        return found

    def _ballast_from(self, spec):
        """The ballast CRIT_VRAM_BALLAST asks for ("leave:SIZE" or "take:SIZE"), made once a process."""
        kind, _, size = spec.strip().partition(":")
        number = size.strip().lower()
        unit = number.lstrip("0123456789.")
        if kind not in ("leave", "take") or unit not in _UNITS or not number[:len(number) - len(unit)]:
            raise ValueError(f"{BALLAST} is leave:SIZE or take:SIZE (SIZE in bytes, KiB, MiB or GiB), not {spec!r}")
        nbytes = int(float(number[:len(number) - len(unit)]) * _UNITS[unit])
        return self.ballast(**{kind: nbytes})

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
    def give(self, name, release, regrow=None, keep=0, rank=COPIES, hungry=None):
        """A giver: release(nbytes) frees at least nbytes if it can and returns the bytes it freed; regrow(room) may
        take up to room bytes back and returns the bytes it took; keep: the free bytes it leaves the device beyond
        what the device keeps (a number, or a callable giving it now); rank: its place in the device's one eviction
        order (a number, or a callable giving it now): IDLE, COPIES or WORK, the lowest asked first (the module's
        description); hungry: whether it would take more than it holds now (a callable: an expert cache that holds
        fewer slots than experts), so a settle that found no room has the IDLE givers give to it."""
        self.givers[name] = (release, regrow, keep if callable(keep) else int(keep))
        self.ranks[name] = rank if callable(rank) else int(rank)
        if hungry is not None:
            self.hungers[name] = hungry
        else:
            self.hungers.pop(name, None)
        self.given.setdefault(name, {"released": 0, "regrown": 0, "releases": 0})

    def ungive(self, name):
        self.givers.pop(name, None)
        self.ranks.pop(name, None)
        self.hungers.pop(name, None)

    def rank(self, name):
        """A giver's place in the eviction order now (its rank)."""
        found = self.ranks.get(name, COPIES)
        return int(found()) if callable(found) else int(found)

    def _ordered(self):
        """The givers in the device's eviction order: the lowest rank first, then as they were given."""
        names = list(self.givers)
        return [(name, self.givers[name]) for name in sorted(names, key=lambda n: (self.rank(n), names.index(n)))]

    def _released(self, name, release, nbytes):
        got = int(release(nbytes) or 0)
        if got:
            counted = self.given.setdefault(name, {"released": 0, "regrown": 0, "releases": 0})
            counted["released"] += got
            counted["releases"] += 1
        return got

    def need(self, nbytes):
        """Make sure `nbytes` of device memory can be taken with what the givers keep free still free: the givers
        release what is missing, and torch's allocator hands back the blocks it holds and nothing uses if that was
        not enough. Returns the bytes the givers released (0 when the device had room). Call it before an
        allocation, outside a capture."""
        free = self.readings.free()
        if free is None:
            return 0
        if self._needing:   # a giver's release asked again (a range it trims settles): the outer need decides
            return 0
        givers = self._ordered()
        kept = self.keeps() if givers else 0
        missing = int(nbytes) + kept - free
        released = 0
        self._needing = True   # no settle while the givers release: they would grow back into what is being freed
        try:
            for name, (release, _, _) in givers:   # the device's eviction order: the lowest rank gives first
                if missing <= released:
                    break
                released += self._released(name, release, missing - released)
        finally:
            self._needing = False
        if missing > released:
            self.readings.release()
        self._floor, self._seen = kept, (free + released if released else free)
        return released

    def other(self, free=None, state=None):
        """The device memory outside the engine: what neither torch's allocator nor an elastic range of this process
        holds and is not free (the CUDA driver's, this process's memory made past both, other programs'). `free` and
        `state` (torch's reading): readings already taken, so the figure is of the same moment. None where the device
        cannot be read."""
        state = self.readings.torch() if state is None else state
        free = self.readings.free() if free is None else free
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
        other = self.other()
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

    def settle(self, seconds=SETTLE_S):
        """The device is quiet: each giver may regrow into the free memory less what it keeps free, SETTLE_SLICE
        bytes at a time for `seconds` at most (None: until nothing is left to take). Returns the bytes taken;
        nothing while the engine still loads (serve() ends the load), and nothing for HOLD_S after relieve() gave
        memory to another program. `owed` says whether it ran out of time with memory a giver may still take: the
        next settle goes on from there."""
        taken = 0
        if self.serving and self.clock() >= self._hold and not self._needing:
            deadline = None if seconds is None else time.perf_counter() + float(seconds)
            self.owed = False
            room = 0
            for name, (_, regrow, keep) in self._ordered()[::-1]:   # the highest rank grows back first
                while regrow is not None and not self.owed:
                    free = self.readings.free()
                    room = 0 if free is None else free - max(_kept(keep), self.kept())
                    got = int(regrow(min(room, SETTLE_SLICE)) or 0) if room > 0 else 0
                    if not got and room > SETTLE_SLICE:   # a giver whose unit is larger than a slice
                        got = int(regrow(room) or 0)
                    taken += got
                    if got:
                        self.given.setdefault(name, {"released": 0, "regrown": 0, "releases": 0})["regrown"] += got
                        room -= got
                    if not got:
                        break
                    self.owed = deadline is not None and time.perf_counter() >= deadline and room - got > 0
            free = self.readings.free()
            short = free is None or free - self.keeps() < SETTLE_SLICE
            if short and not self.owed and any(_hungry() for _hungry in list(self.hungers.values())):
                # A giver that grows back would take more, and no room is left: memory an idle giver holds (a prompt
                # kept past its use) is worth more in it, so the idle ones give a slice and the next settle grows
                # into it.
                for name, (release, regrow, _) in self._ordered():
                    if self.rank(name) > IDLE:
                        break
                    if self._released(name, release, SETTLE_SLICE):
                        self.owed = True   # Core's watch asks for the step whose settle takes it
                        break
            if self.givers:
                self._floor = self.keeps()
                self._seen = self.readings.free()
        return taken

    def watch(self, ask, every=RELIEVE_S, also=None):
        """A thread of Core's that calls ask() when another program left the device less free memory than the
        givers keep: every `every` seconds it reads the free bytes and compares them with what the engine's thread
        last saw (what the givers kept then, and the bytes free then; it calls nothing of a giver's). ask() makes
        the engine run a step of its own, where relieve() gives the memory back; it is called once per fall, not
        while the device stays short. also: another reason the engine must run a step of its own, a callable read
        each time (a host handoff that failed: the engine's next step ends it). While a settle left memory owed
        to the givers it asks too, every time it looks, so the givers grow back a settle a step although Core
        launches the engine's rounds without it. unwatch() ends the thread. One watch a device."""
        self.unwatch()
        stop = threading.Event()

        def run():
            asked = None
            while not stop.wait(every):
                if (also is not None and also()) or (self.owed and self.serving and self.clock() >= self._hold):
                    ask()   # memory owed to the givers: the engine's own step settles some more
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
        """An engine starts to load on this device: until serve(), what its plugins leave free is the load's. Its
        steps' arena and counts start anew."""
        self._drain()
        self.serving, self.held = False, None
        self.arena, self.most, self.loaded = 0, {}, None
        self.counts = {"steps": 0, "refused": 0, "overflows": 0, "breaches": 0, "least_free": None}
        self.segment_memory, self.segment_memory_cut, self.outside_at_serve = [], False, None

    def serve(self):
        """The load is over (Core calls it once the warm requests ran): torch's unused blocks go back to the driver
        and the givers take what is free, less what they keep, all of it at once (no step waits on this settle).
        The counts start anew for the requests (the load's are kept: steps()' "load"), so a check reads what its
        requests did. Returns the bytes they took."""
        self.readings.release()
        self._drain()   # the load's segments, the load's counts
        self.serving = True
        self.loaded = {key: value for key, value in self.counts.items() if key != "least_free"}
        self.counts = {"steps": 0, "refused": 0, "overflows": 0, "breaches": 0, "least_free": None}
        self._hold, self._others = float("-inf"), self.other()
        self.segment_memory, self.segment_memory_cut, self.outside_at_serve = [], False, self._others
        return self.settle(None)

    # -- the steps' memory: Core's allocator, the arena, the headroom -------------------------------------------------
    def keeps(self):
        """What the givers leave free now: the device's own (kept()) or a giver's own keep, the most of them."""
        return max([self.kept()] + [_kept(keep) for _, _, keep in self.givers.values()])

    def kept(self):
        """What the givers leave free on the device now: the headroom, and the part of the step arena the steps'
        segments do not hold now (while the engine serves: the load's steps live on what the givers left free when
        they were made). Nothing here is a forecast: the arena was measured at load, and the segments are counted as
        Core's allocator hands them out."""
        if not self.serving:
            return int(self.headroom)
        self._drain()
        return int(self.headroom) + max(0, int(self.arena) - self._pool_bytes)

    @contextmanager
    def step_pool(self):
        """Core's step frame for the allocations of one step: every tensor torch makes in the block on this thread
        comes from the step pool, whose segments Core's allocator makes and logs for this ledger (_drain). A device
        without Core's allocator (no compiler, no CUDA) runs the block on torch's own."""
        pool = self._current_pool()
        if pool is None:
            yield
            return
        import torch
        with torch.cuda.use_mem_pool(pool, device=self._index()):
            yield

    def _index(self):
        return int(self.device.split(":")[1]) if ":" in self.device else 0

    def _current_pool(self):
        if self._pool is None and isinstance(self.readings, _Torch):
            allocator = _Allocator.get()
            if allocator is None:
                self.unaccounted = f"no Core allocator here: {_Allocator._made}"
                return None
            import torch
            if self._allocator is None:
                allocator.card(self.readings.index)   # the card's handle, found here: the allocator asks no torch
                self._allocator = allocator
            self._drain()   # what the last pool's segments did is counted with that pool gone
            self._pool, self._pool_bytes = torch.cuda.MemPool(allocator.pluggable.allocator()), 0
            self._name_step_pool(tuple(self._pool.id))
        return self._pool

    def _name_step_pool(self, handle):
        """The step pool listed as core/steps in `pools`, as pool() lists a graph's, so the device's readings (and
        crit_plugins.audit) name its segments, also those a retired pool keeps (a workspace a plugin made at its first
        call). Each pool is a new torch pool, so every STEP_POOLS_NAMED of them the names of those the allocator holds
        no segment of any more go: the list stays as short as the pools that still hold memory."""
        self.pools[handle] = "core/steps"
        self._step_pools.append(handle)
        if len(self._step_pools) > STEP_POOLS_NAMED:
            held = {found[4] for found in self.readings.segments() if len(found) > 4}
            kept = [pool for pool in self._step_pools if pool in held or pool == handle]
            for pool in set(self._step_pools) - set(kept):
                self.pools.pop(pool, None)
            self._step_pools = kept

    def retire_pool(self):
        """The step pool goes, and with it every segment no live tensor holds (the allocator frees them as torch
        drops the pool); a segment a tensor still holds (a workspace a plugin made at its first call and keeps)
        stays counted as the engine's (retired). The next step takes a new pool, so its segments are its own and
        none a larger, earlier step left."""
        self._drain()
        self.retired += self._pool_bytes
        self._pool, self._pool_bytes = None, 0
        self._drain()   # the segments torch freed with the pool: the retired ones'

    def _drain(self):
        """What Core's allocator logged for this device since the last look (allocator.cpp: each segment made,
        refused and freed), counted in its order: at every step frame and every reading of the steps' memory, never
        inside torch's allocation."""
        allocator = self._allocator
        if allocator is None:
            return
        with self._draining:
            outside = False   # the memory outside the engine, read once a drain that holds a segment made
            for kind, nbytes, free in allocator.events(self._index()):
                if kind == MADE:
                    if outside is False:
                        outside = self.other() if self.serving else None
                    self._made(nbytes, free, outside)
                elif kind == REFUSED:
                    self.counts["refused"] += 1
                    self.last_refused = nbytes
                else:
                    self._freed(nbytes)

    def _admit(self, nbytes, free=None):
        """The rule Core's allocator applies to a new segment of `nbytes` (natively, allocator.cpp), for a device
        read through readings of its own (a test's): made where the device has the bytes free (`free`, or the
        readings' free bytes), and counted as the allocator's log is (_made), or refused (torch raises its
        out-of-memory error). Returns whether it was made."""
        free = self.readings.free() if free is None else int(free)
        if free is not None and free < int(nbytes):
            self.counts["refused"] += 1
            self.last_refused = int(nbytes)
            return False
        self._made(nbytes, free, self.other(free) if self.serving else None)
        return True

    def _made(self, nbytes, free, outside=None):
        """A segment of `nbytes` was made for the step pool, the device having had `free` bytes free before it (None:
        not read). What the arena does not hold the device lent from its free memory: counted where the pool passes
        the arena (overflows) or the segment left less than the headroom free (breaches). While serving, the free
        bytes it left are kept with `outside`, the memory outside the engine read with it (other(): the driver,
        other programs on the card; None: not read), so a check can judge the engine's own use (segment_memory)."""
        nbytes = int(nbytes)
        self.granted += nbytes
        self._pool_bytes += nbytes
        if self._step is not None:
            self._step[2] = max(self._step[2], self._pool_bytes - self._step[1])
        if self.serving:
            if self.arena and self._pool_bytes > self.arena:
                self.counts["overflows"] += 1
            if free is not None:
                left = free - nbytes
                least = self.counts["least_free"]
                self.counts["least_free"] = left if least is None else min(least, left)
                if left < self.headroom:
                    self.counts["breaches"] += 1
                if len(self.segment_memory) < SEGMENT_SAMPLES:
                    self.segment_memory.append((left, None if outside is None else int(outside)))
                else:
                    self.segment_memory_cut = True

    def _freed(self, nbytes):
        """Core's allocator freed a segment (torch dropped a retired pool's blocks): a retired one's first, as the
        pools that hold tensors still alive are the retired ones."""
        nbytes = int(nbytes)
        self.granted = max(0, self.granted - nbytes)
        if self._pool is None or self.retired >= nbytes:
            self.retired = max(0, self.retired - nbytes)
        else:
            self._pool_bytes = max(0, self._pool_bytes - nbytes)

    @contextmanager
    def measured(self, kind="prompt"):
        """A step a plugin runs at load to measure it (a calibration's prompt chunk at a position no warm request
        reaches): Core's step frame around it, its temporaries from the step pool, so what it took counts as a step
        of `kind` ("prompt", "score" or "decode") and the step arena holds it. Yields a Measured whose `took` is the
        bytes the step pool's segments grew by during it, set once the block ends (0 where Core allocates nothing
        here: no CUDA, no allocator). Not inside another step."""
        if self._step is not None:
            raise RuntimeError("a measured step runs between steps, not inside one")
        found = Measured()
        self._drain()
        self._step = [kind, self._pool_bytes, 0]
        self.counts["steps"] += 1
        try:
            with self.step_pool():
                yield found
        finally:
            self._drain()   # the step's segments, counted while it is the step in flight
            step, self._step = self._step, None
            found.took = int(step[2])
            self.most[kind] = max(self.most.get(kind, 0), found.took)
            self.retire_pool()   # its segments go back to the driver: the next measured step's are its own

    def measure(self, run, kind="prompt", attempts=MEASURE_ATTEMPTS):
        """run() as Core's measured step (measured()), with room made for it: while the engine loads the givers keep
        only the headroom free (an expert cache opened before a model's calibration takes every other byte), and a
        step's temporaries come from Core's allocator, which cannot ask a giver for a segment (it calls no Python
        inside torch's allocation). So where the allocator refused one of run()'s segments (torch's out-of-memory
        error), the givers release room (need(): twice the refused segment the first time, twice what was asked
        each time after, at least what the step took before it failed) and run() runs again, up to `attempts`
        times; the room stays free for the load's other steps, as nothing grows back before the engine serves. A
        failure for any other reason, or with no giver left to give, is raised. run() must be repeatable (a
        calibration: the same rows at the same positions). Returns the Measured of the run that finished."""
        asked = 0
        for attempt in range(int(attempts)):
            refused = self.counts["refused"]
            try:
                with self.measured(kind) as step:
                    run()
                return step
            except Exception as error:   # noqa: BLE001 - only the allocator's refusal is retried; the rest raises
                self._drain()
                if self.counts["refused"] == refused or attempt + 1 == int(attempts) or not _out_of_memory(error):
                    raise
                asked = max(2 * int(self.last_refused or 0), 2 * asked, int(step.took))
                free = self.readings.free()
                if not self.need(asked) and (free is None or free < asked + self.keeps()):
                    raise   # no giver gave, and the device does not have the room
        raise RuntimeError("unreachable")   # pragma: no cover

    def begin_step(self, prompt_tokens, score=False):
        """Core's step frame (before every plugin's): a step of `prompt_tokens` prompt tokens (0: decode rows alone)
        or a score begins. Its temporaries come from the step pool (step_pool), within the arena the givers keep
        free: nothing is made room for, since the arena is there whenever no step holds it."""
        self._drain()
        self._step = ["score" if score else "prompt" if prompt_tokens else "decode", self._pool_bytes, 0]
        self.counts["steps"] += 1

    def end_step(self, prompts_open=False):
        """Core's step frame (after every plugin's): the step returned (prompts_open: a row is still prefilling).
        What it took is kept by its kind (most). After a prompt or score step the step pool retires (its segments go
        back to the driver) once no prompt is in flight, and also while one is when another step of the kind could
        not take its most from the pool within the arena: torch reuses the pool's cached blocks for a like step (a
        long prompt's next chunk), but a deeper chunk needs larger blocks than the pool caches and adds segments of
        its own, so a pool kept through a whole long prompt grew past the arena (an engine offloading experts, its
        200,000-token prompt: 1.17 GB of segments against an arena of 0.79 GB), and one retired after every chunk
        makes every chunk's segments anew (3.5% of that prompt's time). Once no prompt is in flight the givers grow back into
        the free memory, less what they keep, a bounded settle at a time."""
        self._drain()   # the step's segments, counted while it is the step in flight
        step, self._step = self._step, None
        if step is not None:
            kind = step[0]
            self.most[kind] = max(self.most.get(kind, 0), step[2])
            if kind != "decode":
                self._unsettled = True
                if not prompts_open or self._pool_bytes + self.most[kind] > self.arena:
                    self.retire_pool()
        if not prompts_open and self._unsettled:
            self._unsettled = False
            if self.serving and self.givers:
                self.readings.release()
                self.settle()
        elif not prompts_open and self.serving and self.givers and self.owed:
            self.settle()

    def size_arena(self, declared=0):
        """Core's loader, once the warm requests ran: the arena is the most a step took then (most), or the model's
        own word for its deepest step (`declared`: info's step_bytes), whichever is more, ARENA_SLACK beyond."""
        measured = max(self.most.values(), default=0)
        self.arena = int(max(measured, int(declared or 0)) * ARENA_SLACK)
        return self.arena

    def steps(self):
        """The steps' memory for the device's report and a giver's telemetry: the headroom, the arena, the bytes the
        step pool's segments hold (granted, and retired: kept by the engine past their pool), the most a step of
        each kind took, and the counts the serving checks read: steps, segments refused, overflows past the arena,
        breaches of the headroom and the least free memory a segment left; and each serving segment's [free bytes left,
        memory outside the engine then or None] ("segment_memory", the first SEGMENT_SAMPLES; "segment_memory_cut"
        where more were made) with the outside memory as serving began ("outside_at_serve"), by which a check judges
        the engine's own use, other programs' allocations set aside."""
        self._drain()
        found = {"headroom": int(self.headroom), "arena": int(self.arena), "granted": int(self.granted),
                 "retired": int(self.retired), "most": dict(self.most), **dict(self.counts),
                 "segment_memory": [list(pair) for pair in self.segment_memory],
                 "outside_at_serve": self.outside_at_serve}
        if self.segment_memory_cut:
            found["segment_memory_cut"] = True
        if self.loaded is not None:   # the counts are the served requests'; the load's own, apart
            found["load"] = dict(self.loaded)
        if self.unaccounted:
            found["unaccounted"] = self.unaccounted
        return found

    def givings(self):
        """Each giver's rank now and what it gave back and took again so far ({name: {"rank", "released",
        "regrown", "releases"}}): the device's report carries it, and a caller reads a request's share as the
        difference of two reports (Eval's variance, a check's memory)."""
        return {name: {"rank": self.rank(name), **dict(self.given.get(name) or {})} for name in self.givers}

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
                 "other": self.other(free, state)}
        if self.outside["relieved"]:
            found["outside"] = dict(self.outside)
        found["steps"] = self.steps()   # the headroom, the arena, what the steps took, the counted invariants
        if self.givers:
            found["givers"] = self.givings()
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
                made = _DEVICES[index] = DeviceMemory(f"cuda:{index}", _Torch(index))
                if os.environ.get(BALLAST):   # a test's share of the device taken away, before anyone plans
                    made.ballasted = made._ballast_from(os.environ[BALLAST])
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
