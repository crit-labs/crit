"""Model files read by relative name from a directory, an in-memory mapping or a reader callback.

Every successful read is recorded as a manifest row (path, bytes, sha256), so identity covers exactly the
files a loader used. Rows have the same shape as crit run-record model manifests.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading

# Windows device names, refused as the stem (text before the first dot) of any name component on every platform.
# Windows also takes the superscript digits one, two and three (U+00B9, U+00B2, U+00B3) as COM and LPT numbers.
RESERVED_DEVICE = re.compile("(?:CON|PRN|AUX|NUL|COM[0-9\u00b9\u00b2\u00b3]|LPT[0-9\u00b9\u00b2\u00b3])")
# O_NONBLOCK keeps a FIFO swapped in after the regular-file check from blocking the open; regular files ignore it.
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)


class ModelFilesError(ValueError):
    """An invalid model file name or source, or a file that exists but cannot be read as one."""


@dataclass(frozen=True)
class FileRecord:
    path: str
    bytes: int
    sha256: str

    def row(self):
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


def file_name(name):
    """Return a canonical relative '/' name. Refused on every platform: absolute names, empty, '.' and '..'
    components, backslashes, NUL, ':' (Windows drives and NTFS streams) and Windows device names (CON, NUL.txt,
    COM1, and COM or LPT with a superscript digit)."""
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name or ":" in name:
        raise ModelFilesError(f"invalid model file name {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") or RESERVED_DEVICE.fullmatch(
            part.split(".", 1)[0].rstrip(" ").upper()) for part in name.split("/")):
        raise ModelFilesError(f"invalid model file name {name!r}")
    return str(path)


def _max_bytes(value):
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f"max_bytes must be a positive integer or None, not {value!r}")
    return value


def _too_large(name, size, max_bytes):
    return ModelFilesError(f"{name}: {size} bytes exceeds the {max_bytes}-byte limit")


def _read_regular(path, name, max_bytes):
    """The bytes of the regular file at `path` (symlinks followed; at most max_bytes + 1 of them when max_bytes is
    set), or None when nothing exists there. Anything else that cannot be opened as a regular file (a dangling or
    looping link, a link through a file, a directory, a FIFO, no permission) raises ModelFilesError naming `name`."""
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):   # nothing by that name, or a path through a regular file
        return None
    except OSError as error:
        raise ModelFilesError(f"{name}: open: {error.strerror}") from None
    try:
        if not stat.S_ISREG(os.stat(path).st_mode):
            raise ModelFilesError(f"{name}: not a regular file")
        handle = open(os.open(path, _OPEN_FLAGS), "rb")
    except OSError as error:
        raise ModelFilesError(f"{name}: open: {error.strerror}") from None
    with handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ModelFilesError(f"{name}: not a regular file")
        if max_bytes is None:
            return handle.read()
        if info.st_size > max_bytes:
            raise _too_large(name, info.st_size, max_bytes)
        return handle.read(max_bytes + 1)   # a file that grew past the limit fails the caller's length check


def _overrides(cls, hook, fallback):
    """Whether `hook` is defined at least as far down cls's MRO as `fallback`, so that a subclass overriding only
    `fallback` (for example _read below a class that defines _read_limited) is still consulted."""
    for klass in cls.__mro__:
        if hook in vars(klass):
            return True
        if fallback in vars(klass):
            return False
    return False


class ModelFiles:
    """Base class: subclasses implement _read(name) -> bytes-like or None and _names() -> iterable of names.

    A source that can do better may also implement _read_limited(name, max_bytes), which may refuse a file above
    max_bytes before reading it, and _names_under(prefix), which may list less than everything. read() and names()
    use them unless a subclass overrides _read or _names further down, so subclasses of DirectoryFiles that
    override _read or _names are consulted. read() and names() check and filter what the hooks return either way.
    An override of read() itself must accept max_bytes: the model and text loaders pass it.
    """

    def __init__(self):
        self._records = {}
        self._lock = threading.Lock()

    def read(self, name, max_bytes=None):
        """Return the file's bytes, or None when it is absent. A file above max_bytes raises ModelFilesError before
        it is hashed or recorded (a directory checks the size before reading)."""
        name = file_name(name)
        max_bytes = _max_bytes(max_bytes)
        if _overrides(type(self), "_read_limited", "_read"):
            data = self._read_limited(name, max_bytes)
        else:
            data = self._read(name)
        if data is None:
            return None
        data = bytes(data)
        if max_bytes is not None and len(data) > max_bytes:
            raise _too_large(name, len(data), max_bytes)
        record = FileRecord(name, len(data), hashlib.sha256(data).hexdigest())
        with self._lock:
            previous = self._records.setdefault(name, record)
        if previous != record:
            raise ModelFilesError(f"{name} changed between reads")
        return data

    def names(self, prefix=""):
        """Sorted names that start with `prefix` (for example 'additional_chat_templates/'). Names that file_name
        refuses are skipped, so an odd file elsewhere never breaks a listing."""
        if not isinstance(prefix, str):
            raise TypeError(f"prefix must be a string, not {type(prefix).__name__}")
        result = set()
        listed = self._names_under(prefix) if _overrides(type(self), "_names_under", "_names") else self._names()
        for name in listed:
            if isinstance(name, str) and name.startswith(prefix):
                try:
                    result.add(file_name(name))
                except ModelFilesError:
                    continue
        return sorted(result)

    @property
    def records(self):
        """Manifest rows for every file read so far, sorted by path."""
        with self._lock:
            return tuple(sorted(self._records.values(), key=lambda record: record.path))

    def _read(self, name):
        raise NotImplementedError

    def _names(self):
        return ()

    def _read_limited(self, name, max_bytes):
        return self._read(name)

    def _names_under(self, prefix):
        return self._names()


class DirectoryFiles(ModelFiles):
    """Files under a model directory. Symlinked files (Hugging Face cache snapshots) are followed. A name is absent
    only when nothing exists under it; anything else that cannot be opened as a regular file (a dangling or looping
    link, a directory by that name, no permission) raises ModelFilesError."""

    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        if not self.root.is_dir():
            raise ModelFilesError(f"{self.root} is not a directory")

    def _read(self, name):
        return _read_regular(self.root / name, name, None)

    def _read_limited(self, name, max_bytes):
        return _read_regular(self.root / name, name, max_bytes)

    def _names(self):
        return self._walk("")

    def _names_under(self, prefix):
        return self._walk(prefix)

    def _walk(self, prefix):
        # Walk only the folder the prefix names (the whole tree for a prefix without '/').
        folder = prefix.rpartition("/")[0]
        if folder:
            try:
                file_name(folder)
            except ModelFilesError:
                return ()   # no valid name can start with it
        base = self.root / folder
        if not base.is_dir():
            return ()
        return [path.relative_to(self.root).as_posix() for path in base.rglob("*") if path.is_file()]


class MappingFiles(ModelFiles):
    """Files held in memory as {name: bytes}, for tests and already-loaded snapshots."""

    def __init__(self, files):
        super().__init__()
        self._files = {file_name(name): bytes(data) for name, data in dict(files).items()}

    def _read(self, name):
        return self._files.get(name)

    def _names(self):
        return self._files.keys()


class CallableFiles(ModelFiles):
    """Files served by a callback returning bytes, a memoryview or None (for example a verified snapshot)."""

    def __init__(self, read, names=()):
        super().__init__()
        self._reader = read
        self._listed = tuple(names)

    def _read(self, name):
        return self._reader(name)

    def _names(self):
        return self._listed


def model_files(source):
    """A ModelFiles source as given, or DirectoryFiles for a str or os.PathLike directory path (ModelFilesError when
    it is not a directory). Anything else raises TypeError: objects that only look like a ModelFiles are refused."""
    if isinstance(source, ModelFiles):
        return source
    if isinstance(source, (str, os.PathLike)):
        return DirectoryFiles(source)
    raise TypeError(f"model files must be a directory path or a crit_core.files.ModelFiles source, not "
                    f"{type(source).__name__}")
