"""Bounded host residency tiers with leased, generation-checked state migration.

Uses the existing paced transport. Blobs represent complete immutable encoded
state; model codecs/semantic identities are supplied by the engine. Control
calls are serialized. Host copies here do not implement device offload.
"""
from concurrent.futures import Future
from dataclasses import dataclass
import hashlib


@dataclass(eq=False)
class _Entry:
    key: str
    identity: str
    generation: int
    data: bytes
    pins: int = 0
    current: bool = True


class TierLease:
    def __init__(self, tier, entry):
        self._tier, self._entry = tier, entry
        entry.pins += 1

    @property
    def data(self):
        if self._entry is None:
            raise ValueError("released tier lease")
        return self._entry.data

    @property
    def current(self):
        return self._entry is not None and self._entry.current

    def close(self):
        if self._entry is None:
            raise ValueError("tier lease already released")
        entry, self._entry = self._entry, None
        entry.pins -= 1
        self._tier._collect(entry)


class MemoryTier:
    """Byte-capacity host store; replaced pinned generations remain charged."""
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("nonnegative tier capacity required")
        self.capacity, self.used, self._generation = capacity, 0, 0
        self._entries, self._retired = {}, set()

    def _reserve(self, size):
        if size > self.capacity - self.used:
            raise ValueError("tier capacity exhausted")
        self.used += size

    def _collect(self, entry):
        if not entry.current and not entry.pins:
            self._retired.discard(entry)
            self.used -= len(entry.data)

    def generation(self, key):
        entry = self._entries.get(key)
        return entry.generation if entry else None

    def _publish(self, key, identity, data):
        # The caller already reserved the new bytes; old pinned readers keep
        # their generation. All potentially allocating work precedes mutation.
        entry = _Entry(key, identity, self._generation + 1, data)
        old = self._entries.get(key)
        if old is not None:
            self._retired.add(old)
        self._entries[key] = entry
        self._generation += 1
        if old is not None:
            old.current = False
            self._collect(old)
        return entry.generation

    def put(self, key, identity, data):
        if not isinstance(key, str) or not key or not isinstance(identity, str) or not identity or type(data) is not bytes:
            raise ValueError("named complete immutable state and identity required")
        self._reserve(len(data))
        try:
            return self._publish(key, identity, data)
        except BaseException:
            self.used -= len(data)
            raise

    def acquire(self, key, identity):
        entry = self._entries[key]
        if entry.identity != identity:
            raise ValueError("incompatible state identity")
        return TierLease(self, entry)

    def evict(self, key, generation):
        entry = self._entries[key]
        if entry.generation != generation:
            raise ValueError("stale tier eviction")
        self._retired.add(entry)
        del self._entries[key]
        entry.current = False
        self._collect(entry)


class TierMove:
    """Copy privately, publish once after success, optionally evict source.

    Pass the sibling transfers module as transport (no hidden import state).
    Optional read/write hooks receive (move, window), register receipts before
    enqueue, and obey BoundedTransfer's physical-settlement contract. Unknown
    failures retain the move and both tier charges until discard proves safety.
    """
    def __init__(self, source, destination, key, identity, limits, *, transport,
                 read=None, write=None):
        if source is destination or not isinstance(source, MemoryTier) or not isinstance(destination, MemoryTier):
            raise ValueError("distinct residency tiers required")
        self.source, self.destination, self.key, self.identity = source, destination, key, identity
        self.lease = source.acquire(key, identity)
        self._source_generation = source.generation(key)
        self._destination_generation = destination.generation(key)
        self._size, self._reserved, self._closed = len(self.lease.data), False, False
        try:
            destination._reserve(self._size)
            self._reserved = True
            self._private = bytearray(self._size)
            part = transport.TransferPart("state", self._size, hashlib.sha256(self.lease.data).hexdigest())
            self._digest = part.sha256
            self.transfer = transport.BoundedTransfer((part,), limits,
                lambda window: self._read(window) if read is None else read(self, window),
                lambda window: self._write(window) if write is None else write(self, window), owners=(self,))
        except BaseException:
            if self._reserved:
                destination.used -= self._size
            self.lease.close()
            raise

    def _read(self, window):
        receipt = window.record(Future())
        window.buffer[:] = memoryview(self.lease.data)[window.offset:window.offset + window.nbytes]
        receipt.set_result(window.nbytes)
        return receipt

    def _write(self, window):
        receipt = window.record(Future())
        self._private[window.offset:window.offset + window.nbytes] = window.buffer
        receipt.set_result(window.nbytes)
        return receipt

    def publish(self, *, evict_source=False):
        if self._closed:
            raise ValueError("tier move already consumed")
        self.transfer.result()
        if hashlib.sha256(self._private).hexdigest() != self._digest:
            raise ValueError("private tier destination checksum differs")
        if (not self.lease.current or self.source.generation(self.key) != self._source_generation
                or self.destination.generation(self.key) != self._destination_generation):
            raise ValueError("state changed while migration was in flight")
        # Converting the private destination creates an immutable allocation.
        # Charge this transient second copy too; failure leaves a retryable move.
        self.destination._reserve(self._size)
        try:
            data = bytes(self._private)
            generation = self.destination._publish(self.key, self.identity, data)
        except BaseException:
            self.destination.used -= self._size
            raise
        self._private = None
        self.destination.used -= self._size
        self._reserved = False
        self._closed = True
        # Source eviction is an independent generation-checked operation. A
        # failure cannot roll back the now-complete destination publication.
        try:
            if evict_source:
                self.source.evict(self.key, self._source_generation)
        finally:
            self.lease.close()
        return generation

    def discard(self, settle=None):
        if self._closed:
            raise ValueError("tier move already consumed")
        self.transfer.discard(settle)
        self._private = None
        if self._reserved:
            self.destination.used -= self._size
            self._reserved = False
        self.lease.close()
        self._closed = True
