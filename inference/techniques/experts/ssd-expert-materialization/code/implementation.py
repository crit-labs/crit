"""Immutable file-backed expert demand and optional read-ahead, with explicit representation, and an optional
bounded host tier in front of the file."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading


@dataclass(frozen=True)
class ExpertRegion:
    offset: int
    nbytes: int
    encoding: str
    sha256: str


@dataclass(frozen=True)
class Materialization:
    owner: object
    encoded_source: bytes
    readiness: object


class HostExpertTier:
    """A bounded host copy of experts between the file and the device cache, least recently used first out. It holds
    what demand read from the file and what the device cache evicts (retain), so a later miss is served from host
    memory instead of a reread. `capacity` is bytes the engine's host-memory admission granted, not extra memory
    beside it. An expert the current route selects (`protected`) is never evicted to make room."""

    def __init__(self, capacity):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("a positive host tier capacity in bytes is required")
        self.capacity = capacity
        self.entries = OrderedDict()
        self.held = 0
        self.lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "admitted": 0, "evicted": 0, "refused": 0}

    def get(self, key):
        with self.lock:
            data = self.entries.get(key)
            self.stats["hits" if data is not None else "misses"] += 1
            if data is not None:
                self.entries.move_to_end(key)
            return data

    def admit(self, key, data, protected=()):
        """Hold `data` for `key`, evicting the least recently used unprotected experts; False when it cannot fit."""
        data = bytes(data)
        with self.lock:
            if key in self.entries:
                self.entries.move_to_end(key)
                return True
            victims, room = [], self.capacity - self.held
            for held, value in self.entries.items():
                if room >= len(data):
                    break
                if held not in protected:
                    victims.append(held)
                    room += len(value)
            if room < len(data):
                self.stats["refused"] += 1
                return False
            for held in victims:
                self.held -= len(self.entries.pop(held))
                self.stats["evicted"] += 1
            self.entries[key] = data
            self.held += len(data)
            self.stats["admitted"] += 1
            return True


class FileExpertSource:
    def __init__(self, path, regions, tier=None):
        self.path = Path(path)
        self.regions = dict(regions)
        self.tier = tier
        self.pending = {}
        self.failed_uploads = []
        self.closed = False
        size = self.path.stat().st_size
        for region in self.regions.values():
            if type(region.offset) is not int or type(region.nbytes) is not int or region.offset < 0 or region.nbytes <= 0:
                raise ValueError("positive encoded expert extents required")
            if region.offset + region.nbytes > size or not region.encoding:
                raise ValueError("expert region exceeds immutable source")
            if len(region.sha256) != 64 or any(x not in "0123456789abcdef" for x in region.sha256):
                raise ValueError("canonical expert checksum required")

    def read(self, key, protected=()):
        """The expert's checked bytes: from the host tier when it holds them, else from the file, then held there."""
        if self.closed:
            raise ValueError("closed expert source")
        region = self.regions[key]
        if self.tier is not None:
            held = self.tier.get(key)
            if held is not None:
                return held
        with self.path.open("rb") as source:
            source.seek(region.offset)
            data = source.read(region.nbytes)
        self._check(region, data)
        if self.tier is not None:
            self.tier.admit(key, data, protected)
        return data

    def retain(self, key, data, protected=()):
        """Hold an expert the device cache evicts in the host tier, from its device copy, instead of rereading the
        file on its next miss. Copy the slot to host before the slot is overwritten; the bytes must be the encoded
        expert exactly. False without a tier or when it cannot make room without evicting `protected`."""
        if self.closed or key not in self.regions:
            raise ValueError("unavailable expert source")
        self._check(self.regions[key], data)
        return self.tier is not None and self.tier.admit(key, data, protected)

    @staticmethod
    def _check(region, data):
        if len(data) != region.nbytes or hashlib.sha256(data).hexdigest() != region.sha256:
            raise ValueError("encoded expert changed or is truncated")

    def prefetch(self, key, executor):
        if self.closed or key not in self.regions:
            raise ValueError("unavailable expert source")
        if key not in self.pending:
            self.pending[key] = executor.submit(self.read, key)
        return self.pending[key]

    def materialize(self, key, upload):
        """upload(encoding, bytes) returns a physical owner plus its readiness."""
        if self.closed:
            raise ValueError("closed expert source")
        data = self.pending.pop(key).result() if key in self.pending else self.read(key)
        try:
            owner, readiness = upload(self.regions[key].encoding, data)
        except BaseException:
            # A failed enqueue may still read staging. Keep bytes and the upload
            # service alive; that service must retain any partial destination.
            self.failed_uploads.append((data, upload))
            raise
        return Materialization(owner, data, readiness)

    def recover_uploads(self, quiesce_uploads):
        """Explicitly prove physical readers stopped; read quiescence is separate."""
        if self.failed_uploads:
            if quiesce_uploads(tuple(self.failed_uploads)) is not True:
                raise ValueError("failed upload quiescence is not proven")
            self.failed_uploads.clear()

    def quiesce(self):
        errors = []
        for receipt in tuple(self.pending.values()):
            try:
                receipt.result()
            except BaseException as exc:
                errors.append(exc)
        self.pending.clear()
        self.closed = True
        if errors:
            raise errors[0]
