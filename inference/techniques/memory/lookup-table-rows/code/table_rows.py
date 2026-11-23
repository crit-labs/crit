"""Rows of a file-backed lookup table (token or hashed n-gram embeddings) gathered on demand.

A decode pass reads a few rows of such a table per token, so the table can stay on disk. The cost is the OS, not the
rows: a page fault or a buffered read on a file the kernel thinks is read sequentially pulls in its whole read-ahead
window (often several MB) per row. This module opens the table for random access (POSIX_FADV_RANDOM on the file, or
MADV_RANDOM on a mapping), reads only the rows asked for with positioned reads in parallel, keeps an optional LRU of
hot rows, and returns them in request order with counters for what it read. RowPrefetch starts a token's rows
as soon as the token exists and lets the step that reads them wait only for rows still in flight.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import mmap
import os
import threading

from crit_plugins.files import pread


@dataclass(frozen=True)
class TableRegion:
    """A table stored as `rows` contiguous rows of `row_bytes` each, starting `offset` bytes into `path`."""
    path: str
    offset: int
    rows: int
    row_bytes: int


def advise_random(fd, offset, length):
    """Tell the OS reads of this file region are random: no read-ahead. Returns the advice applied, or "none" where
    the platform has no file advice (then read with positioned reads, never through a sequential stream)."""
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_RANDOM)
        return "fadvise_random"
    return "none"


def advise_mapping_random(mapping):
    """The same advice for a memory mapping of the table (mmap.MADV_RANDOM): page faults then bring in single pages
    instead of a read-ahead window. Returns the advice applied, or "none"."""
    if hasattr(mmap, "MADV_RANDOM"):
        mapping.madvise(mmap.MADV_RANDOM)
        return "madvise_random"
    return "none"


class RowTable:
    """Gather rows of one TableRegion. workers: parallel reads for the rows a gather misses (cold rows cost a disk
    round trip each, and SSDs serve many at once); cache_rows: hot rows kept in host memory, least recently used out."""

    def __init__(self, region, workers=8, cache_rows=0):
        if region.rows <= 0 or region.row_bytes <= 0 or region.offset < 0:
            raise ValueError("A table needs rows, a row size and a non-negative offset")
        if workers < 1 or cache_rows < 0:
            raise ValueError("workers is at least 1 and cache_rows not negative")
        self.region = region
        self._fd = os.open(region.path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        end = region.offset + region.rows * region.row_bytes
        if os.fstat(self._fd).st_size < end:
            os.close(self._fd)
            raise ValueError(f"{region.path} holds fewer bytes than the table's {end}")
        self.advice = advise_random(self._fd, region.offset, end - region.offset)
        self._pool = ThreadPoolExecutor(workers) if workers > 1 else None
        self._lock = threading.Lock()   # seek-and-read where the platform has no positioned read
        self._cache, self._cache_rows = OrderedDict(), cache_rows
        self.reads = self.bytes_read = self.hits = 0

    def _read(self, row):
        at, size = self.region.offset + row * self.region.row_bytes, self.region.row_bytes
        if hasattr(os, "pread"):
            data = pread(self._fd, size, at)
        else:
            with self._lock:
                os.lseek(self._fd, at, os.SEEK_SET)
                data = os.read(self._fd, size)
        if len(data) != size:
            raise OSError(f"short read of row {row}: {len(data)} of {size} bytes")
        return data

    def gather(self, rows):
        """The bytes of each row in `rows`, in request order; a row asked for twice is read once."""
        rows = list(rows)
        if any(not 0 <= r < self.region.rows for r in rows):
            raise IndexError("row outside the table")
        found, missing = {}, []
        for r in dict.fromkeys(rows):
            if r in self._cache:
                self._cache.move_to_end(r)
                found[r] = self._cache[r]
                self.hits += 1
            else:
                missing.append(r)
        read = list(self._pool.map(self._read, missing)) if self._pool and len(missing) > 1 else \
            [self._read(r) for r in missing]
        for r, data in zip(missing, read):
            found[r] = data
            self.reads += 1
            self.bytes_read += len(data)
            if self._cache_rows:
                self._cache[r] = data
                if len(self._cache) > self._cache_rows:
                    self._cache.popitem(last=False)
        return [found[r] for r in rows]

    def close(self):
        if self._pool:
            self._pool.shutdown()
        os.close(self._fd)


class RowPrefetch:
    """Draft-time prefetch over a RowTable: start(rows) begins reading a token's rows the moment the token exists (a
    draft step's output) and returns a ticket; wait(ticket) returns the rows, blocking only for rows still in flight.
    One service thread takes every queued request at once and gathers them in one batch, so consecutive draft tokens
    share the disk's parallelism. Requests finish in order; the table is this prefetcher's alone while it lives."""

    def __init__(self, table):
        self.table = table
        self._cv = threading.Condition()
        self._queue, self._done, self._failed = [], {}, {}
        self._issued = self._finished = 0
        self._stop = False
        self.requests = self.batches = self.ready_when_waited = 0
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def start(self, rows):
        rows = list(rows)
        if any(not 0 <= r < self.table.region.rows for r in rows):
            raise IndexError("row outside the table")
        with self._cv:
            self._issued += 1
            self._queue.append((self._issued, rows))
            self.requests += 1
            self._cv.notify_all()
            return self._issued

    def ready(self, ticket):
        with self._cv:
            return ticket <= self._finished

    def wait(self, ticket):
        """The ticket's rows in request order; raises its batch's read error."""
        with self._cv:
            if ticket > self._issued or (ticket <= self._finished and ticket not in self._done
                                          and ticket not in self._failed):
                raise KeyError("no such prefetch, or already waited for")
            if ticket <= self._finished:
                self.ready_when_waited += 1
            self._cv.wait_for(lambda: ticket <= self._finished)
            if ticket in self._failed:
                raise self._failed.pop(ticket)
            return self._done.pop(ticket)

    def _serve(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._stop or self._queue)
                if not self._queue:
                    return
                batch, self._queue = self._queue, []
                self.batches += 1
            try:
                rows = self.table.gather([r for _, ids in batch for r in ids])
                error = None
            except Exception as e:  # noqa: BLE001 - handed to the waiter
                rows, error = [], e
            with self._cv:
                at = 0
                for ticket, ids in batch:
                    if error is None:
                        self._done[ticket] = rows[at:at + len(ids)]
                    else:
                        self._failed[ticket] = error
                    at += len(ids)
                    self._finished = max(self._finished, ticket)
                self._cv.notify_all()

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join()
