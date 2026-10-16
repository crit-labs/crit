"""Atomic, checksum-bound byte-prefix checkpoints with codec-owned opaque state."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

HEADER = struct.Struct("<8sIII32s")
MAGIC = b"CRITCHK1"


@dataclass(frozen=True)
class Checkpoint:
    prefix: bytes
    payload: bytes


class CheckpointStore:
    def __init__(self, root, fingerprint, codec, max_bytes=1 << 30):
        if not fingerprint or not codec or type(max_bytes) is not int or not 0 < max_bytes < 1 << 32:
            raise ValueError("explicit state identity, codec and bounded storage required")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fingerprint, self.codec, self.max_bytes = fingerprint, codec, max_bytes

    def _path(self, prefix):
        identity = json.dumps([self.fingerprint, self.codec], separators=(",", ":")).encode()
        return self.root / (hashlib.sha256(identity + b"\0" + prefix).hexdigest() + ".hck")

    def _decode(self, path):
        size = path.stat().st_size
        if not HEADER.size <= size <= self.max_bytes:
            raise ValueError("checkpoint size exceeds declared limits")
        with path.open("rb") as source:
            header = source.read(HEADER.size)
            if len(header) != HEADER.size:
                raise ValueError("truncated checkpoint header")
            magic, nm, np, ns, digest = HEADER.unpack(header)
            if magic != MAGIC or nm > 65536 or HEADER.size + nm + np + ns != size:
                raise ValueError("incompatible or malformed checkpoint envelope")
            body = source.read(size - HEADER.size)
        if len(body) != size - HEADER.size or hashlib.sha256(body).digest() != digest:
            raise ValueError("checkpoint checksum differs")
        metadata = json.loads(body[:nm])
        if metadata != {"fingerprint": self.fingerprint, "codec": self.codec}:
            raise ValueError("checkpoint state representation is incompatible")
        return Checkpoint(body[nm:nm + np], body[nm + np:])

    def put(self, prefix, payload):
        if not isinstance(prefix, bytes) or not isinstance(payload, bytes):
            raise ValueError("exact byte prefix and encoded state required")
        metadata = json.dumps({"fingerprint": self.fingerprint, "codec": self.codec}, sort_keys=True).encode()
        body = metadata + prefix + payload
        if len(metadata) > 65536 or any(len(x) >= 1 << 32 for x in (prefix, payload)):
            raise ValueError("checkpoint fields exceed envelope limits")
        encoded = HEADER.pack(MAGIC, len(metadata), len(prefix), len(payload), hashlib.sha256(body).digest()) + body
        path = self._path(prefix)
        if path.exists():
            existing = self._decode(path)
            if existing != Checkpoint(prefix, payload):
                raise ValueError("same state identity/prefix has a different payload")
            return path
        used = sum(p.stat().st_size for p in self.root.glob("*.hck"))
        if len(encoded) > self.max_bytes - used:
            raise ValueError("checkpoint storage budget exceeded")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, prefix=".crit-", delete=False) as output:
                temporary = Path(output.name)
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            # Reference contract requires a single writer; replace is atomic on
            # this filesystem. A production multi-writer service needs locking.
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return path

    def exact(self, prefix):
        checkpoint = self._decode(self._path(prefix))
        if checkpoint.prefix != prefix:
            raise ValueError("checkpoint prefix identity differs")
        return checkpoint

    def longest_prefix(self, prefix):
        if not isinstance(prefix, bytes):
            raise ValueError("byte prefix required")
        best = None
        for path in self.root.glob("*.hck"):
            candidate = self._decode(path)
            if prefix.startswith(candidate.prefix) and (best is None or len(candidate.prefix) > len(best.prefix)):
                best = candidate
        return best
