"""Immutable file-backed expert demand and optional read-ahead, with explicit representation."""
from dataclasses import dataclass
import hashlib
from pathlib import Path


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


class FileExpertSource:
    def __init__(self, path, regions):
        self.path = Path(path)
        self.regions = dict(regions)
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

    def read(self, key):
        if self.closed:
            raise ValueError("closed expert source")
        region = self.regions[key]
        with self.path.open("rb") as source:
            source.seek(region.offset)
            data = source.read(region.nbytes)
        if len(data) != region.nbytes or hashlib.sha256(data).hexdigest() != region.sha256:
            raise ValueError("encoded expert changed or is truncated")
        return data

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
