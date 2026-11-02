"""Strict records and retained bytes, as every part of crit keeps them: JSON read and written strictly and atomically,
SHA-256 identities of values, files and file inventories, canonical relative paths that never cross a link, and the
field checks records are validated with. crit run's records (crit.run.records), crit serve, crit pull, process
ownership, Kernels and Models all build on these; nothing here knows a run.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import time

from crit_core import jsonfile

MAX_JSON_BYTES = 1_048_576
MAX_FILES = 100_000
HASH_PARALLEL_BYTES = 256 * 1024 * 1024
GIB = 1024 ** 3
# Windows access denied, sharing and lock violations: what a reader's open handle, or a mapped file's handle not yet
# released, briefly causes.
SHARING_ERRORS = {5, 32, 33}
_verification_cache = ContextVar("crit_verification_cache", default=None)


@contextmanager
def verification_scope():
    """Deduplicate hashes only within one boundary check, observing file changes."""
    if _verification_cache.get() is not None:
        yield
        return
    token = _verification_cache.set({})
    try:
        yield
    finally:
        _verification_cache.reset(token)


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def retry_denied(action, any_denial=False):
    """action()'s result, retried for a quarter of a second while Windows briefly denies it (a PermissionError with
    a SHARING_ERRORS winerror, or any PermissionError with `any_denial`); a denial that outlasts it reaches the
    caller."""
    deadline = time.monotonic() + .25
    while True:
        try:
            return action()
        except PermissionError as exc:
            if not (any_denial or getattr(exc, "winerror", None) in SHARING_ERRORS) or time.monotonic() >= deadline:
                raise
            time.sleep(.005)


def file_identity(info):
    """A file's identity in its stat: device, file id, size and change times."""
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_sha(path, cache=False):
    # Windows may retain stale timestamps while another handle is open.
    # Metadata cannot safely identify unchanged bytes there, even within a scope.
    if cache is False:
        cache = _verification_cache.get() if os.name != "nt" else None
    before = path.stat()
    identity = (str(path), *file_identity(before))
    if cache is not None and identity in cache:
        return cache[identity]
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    after = path.stat()
    if file_identity(before) != file_identity(after):
        raise ValueError("Artifact changed while hashing")
    value = hasher.hexdigest()
    if cache is not None and len(cache) < MAX_FILES:
        cache[identity] = value
    return value


def text(value, name, limit=4096):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\0" in value:
        raise ValueError(f"{name} requires nonempty text (up to {limit} characters)")
    return value


def number(value, name, low=0, high=1_000_000, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} requires a finite {'integer' if integer else 'number'} between {low} and {high}")
    return value


def sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} requires a lowercase SHA-256")
    return value


def timestamp(value, name):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.utcoffset().total_seconds() != 0:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"{name} requires a UTC timestamp with timezone") from None
    return result


def keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys():
        raise ValueError("Missing required fields: " + ", ".join(sorted(set(required) - (value.keys() if isinstance(value, dict) else set()))))
    unknown = value.keys() - set(required) - set(optional) - {"extensions"}
    if unknown:
        raise ValueError("Unknown fields; use extensions: " + ", ".join(sorted(unknown)))
    if "extensions" in value and not isinstance(value["extensions"], dict):
        raise ValueError("extensions must be an object")
    json.dumps(value, allow_nan=False)
    return value


def safe_path(root, relative):
    text(relative, "record path")
    parts = PurePosixPath(relative).parts
    if ("\\" in relative or ":" in relative or relative.startswith("/") or
            any(p in {"", ".", ".."} for p in relative.split("/"))):
        raise ValueError("References require canonical relative forward-slash paths")
    root = Path(root).absolute()
    if root.resolve() != root:
        raise ValueError("Reference escapes its retained directory")
    cursor = root
    # Resolve the root once. Every child is lexical and every existing component
    # is inspected without following links; repeated full ancestor resolution
    # made large sealed source trees prohibitively slow on Windows.
    for part in (None, *parts):
        if part is not None:
            cursor = cursor / part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/reparse references are not permitted")
    return cursor


def read_json(path, *, max_bytes=MAX_JSON_BYTES):
    """A retained JSON record: a regular file, not a link, of at most max_bytes, read as strictly as Core reads model
    files (crit_core.jsonfile: no repeated keys, no non-finite numbers)."""
    limit = "1 MiB" if max_bytes == MAX_JSON_BYTES else str(max_bytes) + " bytes"
    if not path.exists() and not path.is_symlink():
        # A relative path resolves against the current directory, not the run.
        raise ValueError(f"JSON file not found: {path} (from {Path.cwd()})" if not path.is_absolute()
                         else f"JSON file not found: {path}")
    if not path.is_file() or path.is_symlink() or path.stat().st_size > max_bytes:
        raise ValueError("JSON must be a regular file under " + limit)
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError("JSON must be under " + limit)
    return jsonfile.loads(data, str(path), max_bytes=max_bytes)


def write_json(path, value, immutable=False, *, max_bytes=MAX_JSON_BYTES, temporary_directory=None):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > max_bytes:
        raise ValueError("JSON projection exceeds " + ("1 MiB" if max_bytes == MAX_JSON_BYTES else str(max_bytes) + " bytes"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=temporary_directory or path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            # Exclusive hard-link publication is atomic and never replaces retained bytes.
            os.link(temporary, path)
        else:
            # Windows readers can briefly deny replacement while holding an
            # ordinary file handle. Keep atomic publication and bound retries;
            # a persistent permission error must still reach the caller.
            retry_denied(lambda: os.replace(temporary, path))
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def inventory(root, exclude=()):
    root = Path(root)
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        safe_path(root, relative)
        if path.is_file() and relative not in exclude:
            files.append((relative, path))
            if len(files) > MAX_FILES:
                raise ValueError("Artifact exceeds file inventory limit")
    # Large artifacts (model checkpoints) hash files concurrently; hashlib releases
    # the GIL while digesting, so threads overlap reads and hashing. Row order and
    # values are identical to a sequential pass.
    cache = _verification_cache.get() if os.name != "nt" else None
    total = sum(path.stat().st_size for _, path in files)
    workers = min(8, os.cpu_count() or 1, len(files)) if total >= HASH_PARALLEL_BYTES else 1
    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            digests = list(pool.map(lambda item: file_sha(item[1], cache), files))
    else:
        digests = [file_sha(path, cache) for _, path in files]
    return [{"path": relative, "bytes": path.stat().st_size, "sha256": value} for (relative, path), value in zip(files, digests)]


def verify_inventory(root, rows, exclude=()):
    if not isinstance(rows, list) or not rows or len(rows) > MAX_FILES:
        raise ValueError("Artifact requires a nonempty bounded file inventory")
    for row in rows:
        keys(row, {"path", "bytes", "sha256"})
        safe_path(root, row["path"])
        number(row["bytes"], "file bytes", 0, 2 ** 60, integer=True)
        sha(row["sha256"], "file sha256")
    if rows != inventory(root, exclude):
        raise ValueError("Retained artifact bytes differ from manifest")
