"""Validated immutable byte-region loading; allocation is a caller service."""
from dataclasses import dataclass
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class TensorRegion:
    name: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    encoding: str
    sha256: str


@dataclass(frozen=True)
class Materialization:
    owner: object
    source_bytes: bytes
    readiness: object


class ArtifactReader:
    def __init__(self, path, regions):
        self.path = Path(path)
        self.regions = {}
        size = self.path.stat().st_size
        ranges = []
        for region in regions:
            if not region.name or region.name in self.regions:
                raise ValueError("region names must be unique and nonempty")
            if type(region.offset) is not int or type(region.nbytes) is not int:
                raise ValueError("byte extents must be integers")
            if region.offset < 0 or region.nbytes < 0 or region.offset + region.nbytes > size:
                raise ValueError("region exceeds the artifact")
            if not region.encoding or any(type(d) is not int or d <= 0 for d in region.shape):
                raise ValueError("explicit encoding and positive dimensions required")
            if len(region.sha256) != 64 or any(c not in "0123456789abcdef" for c in region.sha256):
                raise ValueError("expected SHA-256 must be canonical")
            ranges.append((region.offset, region.offset + region.nbytes))
            self.regions[region.name] = region
        ranges.sort()
        if any(right > next_left for (_, right), (next_left, _) in zip(ranges, ranges[1:])):
            raise ValueError("overlapping regions require a separate alias contract")

    def read(self, name):
        region = self.regions[name]
        with self.path.open("rb") as source:
            source.seek(region.offset)
            payload = source.read(region.nbytes)
        if len(payload) != region.nbytes or hashlib.sha256(payload).hexdigest() != region.sha256:
            raise ValueError("artifact region is truncated or changed")
        return payload

    def materialize(self, name, allocate, copy, retain_failed):
        """copy returns readiness; result retains both endpoints until settled."""
        payload = self.read(name)
        owner = allocate(len(payload))
        try:
            readiness = copy(owner, payload)
        except BaseException:
            # copy failure may still have submitted work. Its service must retain
            # the allocation until quiescence; do not infer settlement here.
            retain_failed((owner, payload))
            raise
        return Materialization(owner, payload, readiness)
