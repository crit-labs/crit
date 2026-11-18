"""Host memory a device reads: a file mapped read-only or anonymous memory, registered with the CUDA runtime as mapped
memory (read-only where the device allows it), its device address, and tensors over it (a CPU tensor over the host
bytes; a CUDA tensor over the device address, through __cuda_array_interface__, that torch's kernels read over the
link). Host code over libc and the CUDA runtime through ctypes; no kernel.

Registering a mapped file takes seconds where allocating and filling tens of GB of pinned memory took minutes on one
machine. The device address of registered memory is cudaHostGetDevicePointer's: with unified addressing it is the host
address, but not on every system (under WSL it differs), so callers use device_address, never the host pointer, on the
device. A copy-engine copy from registered memory (torch's copy_ from the CPU tensor, non_blocking) reads the host
address."""
import ctypes
import os

PROT_READ, PROT_WRITE = 0x1, 0x2
MAP_SHARED, MAP_PRIVATE, MAP_FIXED, MAP_ANONYMOUS, MAP_NORESERVE = 0x01, 0x02, 0x10, 0x20, 0x4000
FILLER_BYTES = 4 << 20   # the shared memory mapped over ranges a mapping leaves out, piece by piece
MADV_RANDOM, MADV_WILLNEED, MADV_HUGEPAGE = 1, 3, 14
REGISTER_PORTABLE, REGISTER_MAPPED, REGISTER_READ_ONLY = 0x01, 0x02, 0x08
CAPTURE_RELAXED = 2   # cudaStreamCaptureModeRelaxed
ATTRIBUTE_READ_ONLY = 113   # cudaDevAttrHostRegisterReadOnlySupported

_libc = None
_cudart = None


class HostMemoryUnavailable(OSError):
    """Host memory regions need libc's mmap and madvise, which Windows does not have."""


def _c():
    global _libc
    if _libc is None:
        if os.name == "nt":
            raise HostMemoryUnavailable("host memory regions map files and anonymous memory through libc (mmap, "
                                        "madvise), which Windows does not have: keep the table on the device")
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_long)
        libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        _libc = libc
    return _libc


class Mapping:
    """[address, address + size) mapped: a file (read-only unless writable) or anonymous memory. close() unmaps."""

    def __init__(self, address, size, path=None, writable=False):
        self.address, self.size, self.path, self.writable = address, size, path, writable

    def advise(self, advice):
        if self.size:
            _c().madvise(self.address, self.size, advice)

    def close(self):
        if self.address:
            _c().munmap(self.address, self.size)
            self.address = 0


def map_file(path, size=None, writable=False):
    """The file mapped shared: read-only (pages stay the page cache's) unless writable."""
    flags = os.O_RDWR if writable else os.O_RDONLY
    fd = os.open(path, flags)
    try:
        size = os.fstat(fd).st_size if size is None else int(size)
        if not size:
            raise ValueError(f"{path} is empty")
        prot = PROT_READ | (PROT_WRITE if writable else 0)
        address = _c().mmap(None, size, prot, MAP_SHARED, fd, 0)
        if address in (None, ctypes.c_void_p(-1).value):
            error = ctypes.get_errno()
            raise OSError(error, f"mmap of {path}: {os.strerror(error)}")
    finally:
        os.close(fd)
    return Mapping(address, size, path, writable)


def map_anonymous(size):
    """size bytes of anonymous memory, zero, writable, reserved only as written."""
    address = _c().mmap(None, int(size), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0)
    if address in (None, ctypes.c_void_p(-1).value):
        error = ctypes.get_errno()
        raise OSError(error, f"mmap of {size} anonymous bytes: {os.strerror(error)}")
    mapping = Mapping(address, int(size), None, True)
    mapping.advise(MADV_HUGEPAGE)
    return mapping


def whole_pages(ranges, size):
    """The whole pages inside byte ranges [first, end) of `size` bytes, as merged spans [(first, end)] in order."""
    page = os.sysconf("SC_PAGE_SIZE")
    spans, last = [], None
    for first, end in sorted((int(first), int(end)) for first, end in ranges):
        first, end = -(-max(0, first) // page) * page, min(end, int(size)) // page * page
        if last is not None and first <= last[1]:
            last[1] = max(last[1], end)
        elif end > first:
            last = [first, end]
            spans.append(last)
    return [tuple(span) for span in spans]


def read_in(mapping, spans):
    """A mapped file's pages outside `spans` (whole_pages: what leave_out took out of the mapping) read into the
    OS's page cache, and none of the others. A mapping's faults read ahead, megabytes at a time, in blocks that
    lie across the left-out ranges, and the OS will not drop part of such a block again; so the kept ranges are
    asked for first (POSIX_FADV_WILLNEED, in pieces of the device's read-ahead size: a request for more is cut to
    it) and the mapping's own read-ahead is turned off (MADV_RANDOM). The faults of a registration then find every
    kept page there and read nothing beside it. Where the OS cannot be asked the faults read each page alone:
    slower, the same bytes."""
    if not spans or mapping.path is None or not hasattr(os, "posix_fadvise"):
        return
    mapping.advise(MADV_RANDOM)
    fd = os.open(mapping.path, os.O_RDONLY)
    try:
        device = os.fstat(fd).st_dev
        try:
            with open(f"/sys/class/bdi/{os.major(device)}:{os.minor(device)}/read_ahead_kb") as found:
                piece = int(found.read()) << 10
        except (OSError, ValueError):
            piece = 0
        piece = max(piece, 128 << 10)
        at = 0
        for first, end in list(spans) + [(mapping.size, mapping.size)]:
            for start in range(at, first, piece):
                os.posix_fadvise(fd, start, min(piece, first - start), os.POSIX_FADV_WILLNEED)
            at = end
    finally:
        os.close(fd)


def leave_out(mapping, ranges):
    """Takes byte ranges [first, end) of a mapping out of host memory: the whole pages inside each are mapped anew,
    onto one small piece of shared memory (FILLER_BYTES, zero) again and again. The mapping stays one range of
    addresses, so it is registered once and read through one device address, and the ranges cost the filler alone:
    a file's pages there are never read (read_in reads the others in) and anonymous pages there go back. What they
    held is gone from the mapping, so the caller keeps it elsewhere (a device cache that holds those bundles for
    good). Before the mapping is registered: a registration holds the pages it found. Returns the bytes left
    out."""
    libc, spans = _c(), whole_pages(ranges, mapping.size)
    if not spans:
        return 0
    fd = os.memfd_create("crit-left-out")
    try:
        os.ftruncate(fd, FILLER_BYTES)
        prot = PROT_READ | (PROT_WRITE if mapping.writable else 0)
        for first, end in spans:
            for at in range(first, end, FILLER_BYTES):
                size = min(FILLER_BYTES, end - at)
                found = libc.mmap(mapping.address + at, size, prot, MAP_SHARED | MAP_FIXED, fd, 0)
                if found in (None, ctypes.c_void_p(-1).value):
                    error = ctypes.get_errno()
                    raise OSError(error, f"mmap of a left-out range ({len(spans):,} ranges; the kernel's "
                                         f"vm.max_map_count bounds a process's mappings): {os.strerror(error)}")
    finally:
        os.close(fd)
    read_in(mapping, spans)
    return sum(end - first for first, end in spans)


def host_tensor(address, size):
    """A CPU uint8 tensor over [address, address + size) (not owning it: keep the memory alive while it is used)."""
    import numpy
    import torch
    array = numpy.ctypeslib.as_array((ctypes.c_uint8 * int(size)).from_address(int(address)))
    return torch.from_numpy(array)


def _runtime():
    """The CUDA runtime torch loaded (its libcudart), through ctypes."""
    global _cudart
    if _cudart is None:
        import torch
        torch.cuda.init()
        path = None
        with open("/proc/self/maps") as maps:
            for line in maps:
                if "libcudart" in line:
                    path = line.split()[-1]
                    break
        lib = ctypes.CDLL(path or "libcudart.so")
        lib.cudaHostRegister.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint)
        lib.cudaHostUnregister.argtypes = (ctypes.c_void_p,)
        lib.cudaHostGetDevicePointer.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint)
        lib.cudaDeviceGetAttribute.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int)
        lib.cudaThreadExchangeStreamCaptureMode.argtypes = (ctypes.POINTER(ctypes.c_int),)
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        lib.cudaGetLastError.restype = ctypes.c_int
        _cudart = lib
    return _cudart


def _check(error, what):
    if error:
        lib = _runtime()
        lib.cudaGetLastError()   # clear the sticky-free error so later calls do not see it
        raise RuntimeError(f"{what}: {lib.cudaGetErrorString(error).decode()} ({error})")


def read_only_supported(device=0):
    value = ctypes.c_int(0)
    error = _runtime().cudaDeviceGetAttribute(ctypes.byref(value), ATTRIBUTE_READ_ONLY, int(device))
    return not error and bool(value.value)


def register(address, size, read_only=False):
    """Registers [address, address + size) as mapped memory for every device of the process (portable; read-only: the
    device never writes it, the flag a read-only file mapping needs); returns its device address."""
    lib = _runtime()
    flags = REGISTER_PORTABLE | REGISTER_MAPPED | (REGISTER_READ_ONLY if read_only else 0)
    _check(lib.cudaHostRegister(address, size, flags), "cudaHostRegister")
    pointer = ctypes.c_void_p()
    error = lib.cudaHostGetDevicePointer(ctypes.byref(pointer), address, 0)
    if error:
        lib.cudaHostUnregister(address)
        _check(error, "cudaHostGetDevicePointer")
    return pointer.value


def unregister(address):
    _check(_runtime().cudaHostUnregister(address), "cudaHostUnregister")


class _DeviceView:
    def __init__(self, address, size):
        self.__cuda_array_interface__ = {"shape": (int(size),), "typestr": "|u1", "data": (int(address), False),
                                         "version": 3, "strides": None}


def device_tensor(address, size, device):
    """A CUDA uint8 tensor over device address [address, address + size) (registered host memory read over the link
    by torch's kernels). Not owning it: keep the registration alive while it is used."""
    import torch
    device = torch.device(device)
    with torch.cuda.device(device):
        return torch.as_tensor(_DeviceView(address, size), device=device)


class relaxed_capture:
    """While it is entered, this thread's CUDA calls run in relaxed stream-capture mode: another thread capturing a
    graph in global mode does not fail them (a telemetry read beside a capture). The previous mode comes back after."""

    def __enter__(self):
        self._mode = ctypes.c_int(CAPTURE_RELAXED)
        _check(_runtime().cudaThreadExchangeStreamCaptureMode(ctypes.byref(self._mode)), "capture mode")
        return self

    def __exit__(self, *exc):
        _runtime().cudaThreadExchangeStreamCaptureMode(ctypes.byref(self._mode))
        return False
