"""Caller-paced opaque checkpoint transport with bounded staging and receipts.

Control-plane calls and callbacks must be serialized. A receipt's done() proves
physical settlement; successful primary result() returns the exact byte count.
This transports private state. Bundle identity, semantic boundaries, destination
allocation and atomic publication belong to the enclosing checkpoint transaction.
"""
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from itertools import islice
import re

_ACTIVE = {}
_MAX_ACTIVE = 16


@dataclass(frozen=True)
class TransferPart:
    name: str
    nbytes: int
    sha256: str

    def __post_init__(self):
        if (not isinstance(self.name, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.name) is None
                or type(self.nbytes) is not int or not 0 <= self.nbytes <= 1 << 40
                or not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None):
            raise ValueError("named bounded component extent and SHA256 required")


@dataclass(frozen=True)
class TransferLimits:
    window_bytes: int
    staging_bytes: int
    windows: int
    total_bytes: int

    def __post_init__(self):
        if (any(type(x) is not int for x in (self.window_bytes, self.staging_bytes, self.windows, self.total_bytes))
                or not 0 < self.window_bytes <= self.staging_bytes <= 1 << 30
                or not 0 < self.windows <= 64 or not 0 <= self.total_bytes <= 1 << 40):
            raise ValueError("explicit staging, window-count and total-transfer bounds required")


class PendingTransfer(RuntimeError):
    pass


class TransferFailure(RuntimeError):
    def __init__(self, transfer, reason):
        super().__init__(str(reason))
        self.transfer = transfer


@dataclass
class _Work:
    receipt: object
    primary: bool
    settled: bool = False


class TransferWindow:
    """One private staging owner; backend hooks register work before submission."""
    def __init__(self, part, offset, nbytes):
        self.part, self.offset, self.nbytes = part, offset, nbytes
        self._buffer = bytearray(nbytes)
        self._owners, self._work = [], []
        self._primary = None
        self._phase = "new"
        self._uncertain = False
        self._registering = False

    @property
    def buffer(self):
        """Writable during read; read-only for write and later physical readers."""
        view = memoryview(self._buffer)
        return view if self._phase == "reading" else view.toreadonly()

    def retain(self, owner):
        if not self._registering or len(self._owners) >= 64:
            raise ValueError("retain bounded owners inside a submission callback")
        self._owners.append(owner)
        return owner

    def track_work(self, receipt):
        if (not self._registering or len(self._work) >= 64
                or not callable(getattr(receipt, "done", None))
                or not callable(getattr(receipt, "result", None))):
            raise ValueError("register a bounded physical-work receipt before submission")
        if not any(w.receipt is receipt for w in self._work):
            self._work.append(_Work(receipt, False))
        return receipt

    def record(self, receipt):
        """Primary read/write receipt; result() must equal this window's nbytes."""
        self.track_work(receipt)
        if self._primary is not None and self._primary is not receipt:
            raise ValueError("primary receipt cannot change during submission")
        self._primary = receipt
        next(w for w in self._work if w.receipt is receipt).primary = True
        return receipt

    def _submit(self, callback, phase):
        self._phase, self._uncertain, self._registering = phase, True, True
        try:
            receipt = callback(self)
            if receipt is None or receipt is not self._primary:
                raise ValueError("callback must return its preregistered primary receipt")
            self._uncertain = False
        finally:
            self._registering = False

    def _settled(self, *, check_result):
        pending = self._uncertain
        for work in self._work:
            if work.settled:
                continue
            done = work.receipt.done()
            if type(done) is not bool:
                raise ValueError("done() must prove physical settlement with a boolean")
            if not done:
                pending = True
                continue
            work.settled = True
            if check_result:
                result = work.receipt.result()
                if work.primary and (type(result) is not int or result != self.nbytes):
                    raise ValueError("short or invalid checkpoint transfer")
        return not pending


def retained_transfers():
    """Active/failed pumps stay alive even when their caller loses the handle."""
    return tuple(_ACTIVE.values())


class BoundedTransfer:
    """Incremental read/hash/write pipeline; never allocates a whole component.

    read(window) and write(window) return preregistered settlement receipts. The
    source must be a leased immutable snapshot, and destinations must be private.
    owners retains snapshot/destination/reservation leases for the entire batch.
    Native registrations/allocations must be separately budgeted by the engine.
    """
    def __init__(self, parts, limits, read, write, *, owners=()):
        # Reject oversized/lazy inventories without first materializing all of
        # them; only payload extents, not caller iterable length, may be large.
        parts, owners = tuple(islice(parts, 65)), tuple(islice(owners, 65))
        if (not isinstance(limits, TransferLimits) or not 0 < len(parts) <= 64
                or any(not isinstance(p, TransferPart) for p in parts)
                or len({p.name for p in parts}) != len(parts)
                or sum(p.nbytes for p in parts) > limits.total_bytes
                or not callable(read) or not callable(write) or len(owners) > 64):
            raise ValueError("bounded unique parts, callbacks and retained owners required")
        self.parts, self.limits = parts, limits
        self._read, self._write, self._owners = read, write, owners
        self._windows = []
        self._part, self._offset, self._hashed = 0, 0, 0
        self._hash = hashlib.sha256()
        self._busy = False
        self.state, self.failure = "created", None
        self.read_bytes = self.written_bytes = self.peak_staging_bytes = 0

    @property
    def staging_bytes(self):
        return sum(w.nbytes for w in self._windows)

    @contextmanager
    def _guard(self):
        if self._busy:
            raise RuntimeError("reentrant checkpoint transfer")
        self._busy = True
        try:
            yield
        finally:
            self._busy = False

    def _fail(self, error):
        self.state, self.failure = "failed", error
        raise TransferFailure(self, error) from error

    def _observe(self):
        # Hash in source order even when read completions arrive out of order.
        # A ready window retains its bytes until every writer/extra reader ends.
        for window in self._windows:
            if window._phase == "reading" and window.offset == self._hashed:
                if window._settled(check_result=True):
                    self._hash.update(window.buffer)
                    self._hashed += window.nbytes
                    window._phase = "ready"
                    window._work, window._primary = [], None
            elif window._phase == "writing":
                if window._settled(check_result=True):
                    window._phase = "finished"
                    self.written_bytes += window.nbytes
        self._windows = [w for w in self._windows if w._phase != "finished"]
        while self._part < len(self.parts) and not self._windows:
            part = self.parts[self._part]
            if self._offset != part.nbytes:
                break
            if self._hash.hexdigest() != part.sha256:
                raise ValueError("checkpoint component checksum differs")
            self._part += 1
            self._offset = self._hashed = 0
            self._hash = hashlib.sha256()
        if self._part == len(self.parts):
            self.state = "complete"
            _ACTIVE.pop(id(self), None)
            self._read, self._write, self._owners = None, None, ()

    def pump(self, *, max_bytes, max_submissions=1):
        """Poll once and submit bounded work; no waits or background threads.

        Both read and write bytes count against max_bytes. Zero only polls;
        otherwise the quota must fit one configured window to avoid starvation.
        Return bytes submitted during this call, not bytes physically completed.
        """
        with self._guard():
            if (type(max_bytes) is not int or max_bytes < 0
                    or 0 < max_bytes < self.limits.window_bytes
                    or type(max_submissions) is not int or not 0 <= max_submissions <= 128):
                raise ValueError("bounded pump quota must fit one window (or be zero)")
            if self.state == "complete":
                return 0
            if self.state not in {"created", "active"}:
                raise ValueError("failed/cancelled transfer must be settled and discarded")
            if self.state == "created":
                if len(_ACTIVE) >= _MAX_ACTIVE:
                    raise ValueError("retained transfer capacity exhausted; settle existing work")
                _ACTIVE[id(self)] = self
                self.state = "active"
            submitted, calls = 0, 0
            try:
                self._observe()
                # Prioritize writes so ready buffers do not monopolize staging.
                for window in self._windows:
                    if (window._phase == "ready" and calls < max_submissions
                            and window.nbytes <= max_bytes - submitted):
                        window._submit(self._write, "writing")
                        submitted += window.nbytes
                        calls += 1
                while self._part < len(self.parts) and calls < max_submissions:
                    part = self.parts[self._part]
                    size = min(self.limits.window_bytes, part.nbytes - self._offset,
                               self.limits.staging_bytes - self.staging_bytes, max_bytes - submitted)
                    if size <= 0 or len(self._windows) >= self.limits.windows:
                        break
                    window = TransferWindow(part, self._offset, size)
                    self._windows.append(window)
                    self.peak_staging_bytes = max(self.peak_staging_bytes, self.staging_bytes)
                    window._submit(self._read, "reading")
                    self._offset += size
                    self.read_bytes += size
                    submitted += size
                    calls += 1
            except BaseException as error:
                self._fail(error)
            return submitted

    def done(self):
        """Receipt interface: observe settlement, never submit more work."""
        with self._guard():
            if self.state in {"created", "discarded"}:
                return self.state == "discarded"
            if self.state == "active":
                try:
                    self._observe()
                except BaseException as error:
                    self.state, self.failure = "failed", error
            if self.state == "complete":
                return True
            if self.state in {"failed", "cancelled"}:
                try:
                    return all(w._settled(check_result=False) for w in self._windows)
                except BaseException:
                    return False
            return False

    def result(self):
        if not self.done():
            raise PendingTransfer("checkpoint transfer still owns physical work")
        if self.state != "complete":
            raise TransferFailure(self, self.failure or RuntimeError("transfer discarded"))
        return self.written_bytes

    def cancel(self):
        with self._guard():
            if self.state not in {"created", "active", "failed", "cancelled"}:
                raise ValueError("transfer is already consumed")
            self.state = "cancelled"
            self.failure = self.failure or RuntimeError("checkpoint transfer cancelled")

    def discard(self, settle=None):
        """Stop scheduling; release only after every reader settles.

        settle(window)->True explicitly proves ALL work touching that window has
        stopped, including unknown work after an ambiguous submission failure.
        Pending/throwing/nonboolean proofs retain the entire batch.
        """
        with self._guard():
            if self.state == "discarded":
                raise ValueError("transfer is already discarded")
            if self.state != "complete":
                self.state = "cancelled" if self.state != "failed" else "failed"
                self.failure = self.failure or RuntimeError("checkpoint transfer discarded")
            pending = False
            for window in self._windows:
                try:
                    settled = window._settled(check_result=False)
                except BaseException:
                    settled = False
                if not settled and settle is not None:
                    try:
                        settled = settle(window) is True
                        if settled:
                            window._uncertain = False
                            for work in window._work:
                                work.settled = True
                    except BaseException:
                        settled = False
                pending |= not settled
            if pending:
                raise PendingTransfer("checkpoint owners retained until physical settlement")
            _ACTIVE.pop(id(self), None)
            self._windows, self._owners = [], ()
            self._read, self._write, self.failure = None, None, None
            self.state = "discarded"


def _file_window(window, file, base, write):
    if type(base) is not int or base < 0:
        raise ValueError("nonnegative file component offset required")
    window.retain(file)
    receipt = window.record(Future())
    try:
        file.seek(base + window.offset)
        count = file.write(window.buffer) if write else file.readinto(window.buffer)
        receipt.set_result(count)
    except BaseException as error:
        receipt.set_exception(error)
    return receipt


def read_file_window(window, file, *, base=0):
    """Synchronous seek/readinto adapter; no component-sized temporary allocation."""
    return _file_window(window, file, base, False)


def write_file_window(window, file, *, base=0):
    """Synchronous private-file adapter; enclosing commit must flush/fsync/publish."""
    return _file_window(window, file, base, True)
