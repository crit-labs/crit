"""Stable metadata slots, immutable reader snapshots and incremental byte patches.

Serial host reference. Work scales with live rows and changed payloads, never the
declared row capacity. Native replicas must keep old physical readers immutable.
"""
from dataclasses import dataclass
import heapq


def _positive(value):
    return type(value) is int and value > 0


def _key(key):
    if (not isinstance(key, tuple) or len(key) != 2
            or not all(_positive(v) for v in key)):
        raise ValueError("positive request and generation pair required")


@dataclass(frozen=True)
class MetadataRow:
    slot: int
    key: tuple[int, int]
    version: int
    payload: bytes


@dataclass(frozen=True)
class MetadataSnapshot:
    domain: object
    sequence: int
    capacity: int
    row_bytes: int
    rows: tuple[MetadataRow, ...]

    def diff(self, previous):
        if (self.domain is not previous.domain or self.sequence <= previous.sequence
                or (self.capacity, self.row_bytes) != (previous.capacity, previous.row_bytes)):
            raise ValueError("ordered snapshots from the same metadata table required")
        before = {r.slot: r for r in previous.rows}
        after = {r.slot: r for r in self.rows}
        removed = tuple(slot for slot in before if slot not in after)
        changed = tuple(r for r in self.rows if before.get(r.slot) != r)
        return MetadataDelta(previous, self.sequence, removed, changed,
                             tuple(r.slot for r in self.rows))


@dataclass(frozen=True)
class MetadataDelta:
    base: MetadataSnapshot
    sequence: int
    removed: tuple[int, ...]
    changed: tuple[MetadataRow, ...]
    order: tuple[int, ...]

    @property
    def payload_bytes(self):
        return sum(len(r.payload) for r in self.changed)


class MetadataTable:
    def __init__(self, capacity, row_bytes):
        if not _positive(capacity) or not _positive(row_bytes):
            raise ValueError("positive row and byte bounds required")
        self.capacity, self.row_bytes = capacity, row_bytes
        self._domain, self._sequence, self._version = object(), 0, 0
        self._rows, self._free, self._next = {}, [], 0
        self.empty = MetadataSnapshot(self._domain, 0, capacity, row_bytes, ())

    def set(self, key, payload):
        _key(key)
        if (not isinstance(payload, (bytes, bytearray, memoryview))
                or (payload.nbytes if isinstance(payload, memoryview) else len(payload)) > self.row_bytes):
            raise ValueError("bounded byte payload required")
        payload = bytes(payload)
        if len(payload) > self.row_bytes:
            raise ValueError("payload extent exceeds row bytes")
        old = self._rows.get(key)
        if old is not None and old.payload == payload:
            return old.slot
        if old is None and not self._free and self._next == self.capacity:
            raise ValueError("metadata capacity exhausted")
        slot = old.slot if old else (self._free[0] if self._free else self._next)
        row = MetadataRow(slot, key, self._version + 1, payload)
        self._rows[key] = row
        self._version += 1
        if old is None:
            if self._free:
                heapq.heappop(self._free)
            else:
                self._next += 1
        return slot

    def release(self, key):
        row = self._rows[key]
        heapq.heappush(self._free, row.slot)
        del self._rows[key]

    def snapshot(self, keys=None):
        keys = tuple(self._rows) if keys is None else tuple(keys)
        if len(keys) > self.capacity or len(set(keys)) != len(keys):
            raise ValueError("bounded unique gather keys required")
        rows = tuple(self._rows[k] for k in keys)
        self._sequence += 1
        return MetadataSnapshot(self._domain, self._sequence, self.capacity, self.row_bytes, rows)


class MetadataReplica:
    """Atomically apply a delta; earlier snapshot objects retain their byte owners."""
    def __init__(self, empty):
        if not isinstance(empty, MetadataSnapshot) or empty.sequence != 0 or empty.rows:
            raise ValueError("start a replica from the table's empty snapshot")
        self.snapshot = empty

    def apply(self, delta):
        old = self.snapshot
        # Structural equality allows producer and replica snapshots to be
        # different objects; domain and every owned byte still have to agree.
        if delta.base != old or delta.sequence <= old.sequence:
            raise ValueError("stale or foreign metadata delta")
        rows = {r.slot: r for r in old.rows}
        if len(set(delta.removed)) != len(delta.removed):
            raise ValueError("duplicate removal")
        for slot in delta.removed:
            if slot not in rows:
                raise ValueError("unknown removed slot")
            del rows[slot]
        changed = set()
        for row in delta.changed:
            _key(row.key)
            if (type(row.slot) is not int or not 0 <= row.slot < old.capacity
                    or not _positive(row.version) or type(row.payload) is not bytes
                    or len(row.payload) > old.row_bytes or row.slot in changed or row.slot in delta.removed
                    or (row.slot in rows and row.version <= rows[row.slot].version)):
                raise ValueError("invalid metadata patch")
            rows[row.slot] = row
            changed.add(row.slot)
        if (len(rows) > old.capacity or any(type(slot) is not int for slot in delta.order)
                or len(set(delta.order)) != len(delta.order)
                or set(delta.order) != set(rows)
                or len({r.key for r in rows.values()}) != len(rows)):
            raise ValueError("invalid metadata gather membership")
        result = MetadataSnapshot(old.domain, delta.sequence, old.capacity, old.row_bytes,
                                  tuple(rows[s] for s in delta.order))
        self.snapshot = result
        return result
