"""Positioned file reads and writes for the library's plugins on every platform crit runs on: os.pread and os.pwrite
where the platform has them (POSIX); Windows has neither, so there a read or write seeks first, under a lock per
descriptor, as crit_core.model.tensors reads: threads sharing a descriptor take turns. Open the descriptor in binary
mode (os.O_BINARY on Windows, where os.open defaults to text)."""
import os
import threading

_LOCKS = {}
_GUARD = threading.Lock()


def _lock(fd):
    with _GUARD:
        return _LOCKS.setdefault(fd, threading.Lock())


def pread(fd, size, offset):
    """Up to `size` bytes of descriptor `fd` from byte `offset` (fewer at the end of the file), as os.pread."""
    if hasattr(os, "pread"):
        return os.pread(fd, size, offset)
    with _lock(fd):
        os.lseek(fd, offset, os.SEEK_SET)
        return os.read(fd, size)


def pwrite(fd, data, offset):
    """Write `data` at byte `offset` of descriptor `fd`; the bytes written, as os.pwrite."""
    if hasattr(os, "pwrite"):
        return os.pwrite(fd, data, offset)
    with _lock(fd):
        os.lseek(fd, offset, os.SEEK_SET)
        return os.write(fd, data)
