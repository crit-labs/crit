"""Tensor bytes of an inspected safetensors Checkpoint: Core's Checkpoint::map and Checkpoint::read in Python.

inspect_model validates the headers and never reads tensor bytes; TensorReader reads them from the same files. Each
shard is opened on first use and must still be the file the header was read from (its descriptor's identity equals
ShardInfo.file_identity), so a checkpoint replaced or rewritten after inspection is refused instead of read.
"""
import mmap
import os
from pathlib import Path
import stat

from .safetensors import Checkpoint, SafetensorsError, TensorInfo, _identity


class TensorReader:
    """Read-only access to the tensor bytes of `checkpoint` (crit_core.model.Checkpoint, not header_only).

    map(tensor) is a zero-copy read-only memoryview into the shard's mapping; the view keeps its mapping alive after
    close(), as the owner of a native View does. read(tensor, out) copies exactly the tensor's bytes into a writable
    buffer (pinned host memory, a numpy array), without mapping. `tensor` is a name or one of checkpoint.tensors'
    TensorInfo. On POSIX a file truncated under a mapping raises SIGBUS: pair map with verify_unchanged() and
    controlled storage, as with the native reader. Use as a context manager or call close()."""

    def __init__(self, checkpoint):
        if not isinstance(checkpoint, Checkpoint):
            raise TypeError(f"expected a crit_core.model.Checkpoint, not {type(checkpoint).__name__}")
        if checkpoint.header_only:
            raise SafetensorsError(f"{checkpoint.source}: checkpoint was inspected header_only; tensor bytes are "
                                   "unavailable")
        self.checkpoint = checkpoint
        self._shards = {shard.path: shard for shard in checkpoint.shards}
        self._open = {}   # shard path -> (descriptor, mapping or None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def tensor(self, tensor):
        """The checkpoint's TensorInfo for a name or a TensorInfo; KeyError for a name it lacks, ValueError for a
        record that is not its own."""
        if isinstance(tensor, str):
            found = self.checkpoint.tensors.get(tensor)
            if found is None:
                raise KeyError(f"{self.checkpoint.source}: no tensor {tensor!r}")
            return found
        if not isinstance(tensor, TensorInfo):
            raise TypeError(f"expected a tensor name or TensorInfo, not {type(tensor).__name__}")
        if self.checkpoint.tensors.get(tensor.name) != tensor:
            raise ValueError(f"{self.checkpoint.source}: tensor {tensor.name!r} does not belong to this checkpoint")
        return tensor

    def _descriptor(self, path):
        if self._open is None:
            raise ValueError("TensorReader is closed")
        entry = self._open.get(path)
        if entry is not None:
            return entry
        shard = self._shards[path]
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or _identity(opened) != shard.file_identity:
                raise SafetensorsError(f"{path}: file changed since its header was inspected")
        except BaseException:
            os.close(fd)
            raise
        entry = self._open[path] = [fd, None]
        return entry

    def map(self, tensor):
        """A read-only memoryview of the tensor's bytes, without copying (empty for a zero-byte tensor)."""
        t = self.tensor(tensor)
        if not t.nbytes:
            return memoryview(b"")
        entry = self._descriptor(t.shard)
        if entry[1] is None:
            entry[1] = mmap.mmap(entry[0], 0, access=mmap.ACCESS_READ)
        return memoryview(entry[1])[t.data_begin:t.data_end]

    def read(self, tensor, out):
        """Copy the tensor's bytes into `out`, a writable buffer of exactly its size; returns `out`."""
        t = self.tensor(tensor)
        view = memoryview(out)
        if view.readonly:
            raise ValueError("destination buffer is read-only")
        view = view.cast("B") if view.format != "B" or view.ndim != 1 else view
        if view.nbytes != t.nbytes:
            raise ValueError(f"destination of {view.nbytes} bytes does not match tensor {t.name!r} of {t.nbytes} "
                             "bytes")
        if not t.nbytes:
            return out
        fd = self._descriptor(t.shard)[0]
        done = 0
        while done < t.nbytes:
            count = _read_at(fd, view[done:], t.data_begin + done)
            if not count:
                raise SafetensorsError(f"{t.shard}: truncated while reading tensor {t.name!r}")
            done += count
        return out

    def verify_unchanged(self):
        """Raise SafetensorsError when an opened shard no longer has the identity it was inspected with."""
        for path, (fd, _) in (self._open or {}).items():
            if _identity(os.fstat(fd)) != self._shards[path].file_identity:
                raise SafetensorsError(f"{path}: file changed since its header was inspected")

    def close(self):
        """Close every descriptor and drop the mappings; views handed out keep their mapping alive."""
        entries, self._open = self._open or {}, None
        for fd, mapping in entries.values():
            if mapping is not None:
                try:
                    mapping.close()
                except BufferError:   # a view still exports it: it closes when the last view goes
                    pass
            os.close(fd)


def _read_at(fd, view, offset):
    """Bytes read into `view` from file offset `offset` (0 at end of file)."""
    if hasattr(os, "preadv"):
        return os.preadv(fd, [view], offset)
    # Windows has no preadv: seek, then read a bounded chunk (the descriptor is private to the reader).
    os.lseek(fd, offset, os.SEEK_SET)
    data = os.read(fd, min(len(view), 1 << 26))
    view[:len(data)] = data
    return len(data)


def open_tensors(path, **options):
    """inspect_model(path, **options) and a TensorReader over it."""
    from .safetensors import inspect_model
    return TensorReader(inspect_model(Path(path), **options))
