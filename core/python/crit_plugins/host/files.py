"""Reads of one file at a queue depth, past the OS's page cache: the host's reader threads (Host.reads).

A file read by one thread through the page cache keeps a drive at queue depth 1, and what it reads is held twice
once the reader keeps it too: in the page cache and in the reader's own memory. A tier that holds its bytes itself
(a bounded copy of a store larger than host memory) wants neither. Reads issues positioned reads from `depth`
threads, each with a handle of its own, and with direct=True opens the file past the page cache (O_DIRECT; on
Windows FILE_FLAG_NO_BUFFERING): each read then covers the aligned blocks around its range in the thread's own
aligned buffer and copies the range out. Where the file system refuses direct access the reads are buffered and
stats() says why.

    reads = host.reads(path, depth=8)             # crit_plugins.host.current().reads
    reads.read([(offset, buffer), ...])           # every buffer filled; the calling thread reads too
    reads.stats()                                 # {"reads", "bytes", "direct", "fallback", "depth"}
    reads.close()

The threads wait in the kernel and compute nothing, so they are not workers of the host's pool and are pinned
nowhere. A buffer is anything writable with the buffer protocol (a numpy array, a memoryview of a tensor's bytes).
"""
import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor

DEPTH = 8
ALIGN = 4096


class _Handle:
    """One thread's handle of the file and its aligned buffer."""

    def __init__(self, path, direct):
        self.direct, self.reason = False, None
        self.fd = self.win = None
        self.buffer, self.address = None, 0
        if direct:
            try:
                self._open_direct(path)
                self.direct = True
            except OSError as error:
                self.reason = f"{type(error).__name__}: {error}"
        if not self.direct:
            self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))

    def _open_direct(self, path):
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateFileW.restype = wintypes.HANDLE
            kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                           wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
            handle = kernel.CreateFileW(str(path), 0x80000000, 0x1 | 0x2 | 0x4, None, 3,
                                        0x20000000 | 0x10000000, None)   # read; share all; existing; no buffering
            if handle in (None, wintypes.HANDLE(-1).value):
                raise ctypes.WinError(ctypes.get_last_error())

            class Overlapped(ctypes.Structure):
                _fields_ = [("internal", ctypes.c_void_p), ("internal_high", ctypes.c_void_p),
                            ("offset", wintypes.DWORD), ("offset_high", wintypes.DWORD), ("event", wintypes.HANDLE)]
            kernel.ReadFile.restype = wintypes.BOOL
            kernel.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(Overlapped)]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.win, self._kernel, self._overlapped = handle, kernel, Overlapped
        else:
            flag = getattr(os, "O_DIRECT", None)
            if flag is None:
                raise OSError("this platform has no O_DIRECT")
            self.fd = os.open(path, os.O_RDONLY | flag)

    def _room(self, nbytes):
        if self.buffer is None or len(self.buffer) < nbytes:
            import ctypes
            self.buffer = mmap.mmap(-1, nbytes)   # anonymous memory starts on a page
            self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.buffer))
        return self.buffer

    def read(self, offset, into):
        """Fill `into` (a memoryview of bytes) from `offset`; OSError for a short read."""
        need = len(into)
        if not self.direct:
            got, view = 0, into
            while got < need:
                n = self._pread_into(view[got:], offset + got)
                if not n:
                    raise OSError(f"unexpected end of file at {offset + got}")
                got += n
            return
        first = offset - offset % ALIGN
        span = -(-(offset + need - first) // ALIGN) * ALIGN
        buffer = self._room(span)
        got = self._read_aligned(first, span)
        lead = offset - first
        if got < lead + need:
            raise OSError(f"unexpected end of file at {first + got}")
        into[:] = memoryview(buffer)[lead:lead + need]

    def _pread_into(self, view, offset):
        if hasattr(os, "preadv"):
            return os.preadv(self.fd, [view], offset)
        data = os.pread(self.fd, len(view), offset) if hasattr(os, "pread") else self._seek_read(len(view), offset)
        view[:len(data)] = data
        return len(data)

    def _seek_read(self, nbytes, offset):
        os.lseek(self.fd, offset, os.SEEK_SET)   # this handle is one thread's
        return os.read(self.fd, nbytes)

    def _read_aligned(self, offset, nbytes):
        if self.win is None:
            got = 0
            while got < nbytes:   # a direct read may return fewer whole blocks
                n = os.preadv(self.fd, [memoryview(self.buffer)[got:nbytes]], offset + got)
                got += n
                if not n or got % ALIGN:   # the end of the file: a part of a block
                    break
            return got
        import ctypes
        from ctypes import wintypes
        got = 0
        while got < nbytes:
            at, done = offset + got, wintypes.DWORD(0)
            where = self._overlapped(None, None, at & 0xFFFFFFFF, at >> 32, None)
            ok = self._kernel.ReadFile(self.win, self.address + got, min(nbytes - got, 1 << 30), ctypes.byref(done),
                                       ctypes.byref(where))
            if not ok and ctypes.get_last_error() != 38:   # 38: the end of the file
                raise ctypes.WinError(ctypes.get_last_error())
            got += done.value
            if not done.value or got % ALIGN:   # the end of the file: a part of a block
                break
        return got

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if self.win is not None:
            self._kernel.CloseHandle(self.win)
            self.win = None
        self.buffer = None


class Reads:
    """Positioned reads of `path` from `depth` threads (the caller's among them), direct where the file system
    allows it and `direct` asks."""

    def __init__(self, path, depth=DEPTH, direct=True):
        if depth < 1:
            raise ValueError("a queue depth of 1 or more")
        self.path, self.depth, self.direct = str(path), int(depth), bool(direct)
        self._local, self._handles, self._lock = threading.local(), [], threading.Lock()
        self._threads = ThreadPoolExecutor(self.depth - 1, thread_name_prefix="crit-read") if self.depth > 1 else None
        self._counted = {"reads": 0, "bytes": 0}
        self.fallback = None
        self.closed = False

    def _handle(self):
        found = getattr(self._local, "handle", None)
        if found is None:
            found = self._local.handle = _Handle(self.path, self.direct)
            with self._lock:
                self._handles.append(found)
                if self.direct and not found.direct:
                    self.fallback = found.reason
        return found

    def _one(self, request):
        offset, buffer = request
        view = memoryview(buffer).cast("B")
        self._handle().read(int(offset), view)
        return len(view)

    def read(self, requests):
        """Fill each (offset, buffer) of `requests` from the file. Returns the bytes read; the first failure is
        raised once every read has ended."""
        requests = list(requests)
        if self.closed:
            raise ValueError(f"{self.path}: its reads are closed")
        if not requests:
            return 0
        if self._threads is None or len(requests) == 1:
            done = [self._one(request) for request in requests]
        else:
            futures = [self._threads.submit(self._one, request) for request in requests[1:]]
            failure, done = None, []
            try:
                done.append(self._one(requests[0]))
            except Exception as error:   # the others still write into the caller's buffers: wait for them
                failure = error
            for future in futures:
                try:
                    done.append(future.result())
                except Exception as error:
                    failure = failure or error
            if failure is not None:
                raise failure
        with self._lock:
            self._counted["reads"] += len(done)
            self._counted["bytes"] += sum(done)
        return sum(done)

    def stats(self):
        with self._lock:
            direct = self.direct and self.fallback is None
            return {**self._counted, "depth": self.depth, "direct": direct, "fallback": self.fallback}

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self._threads is not None:
            self._threads.shutdown(wait=True)
        with self._lock:
            for handle in self._handles:
                handle.close()
            self._handles.clear()
