"""Strict safetensors header and index reader: validates layout and metadata and never reads tensor bytes.

Files are read unbuffered, so no read request extends past the 8-byte length prefix and the header it declares.

A shard is checked field by field (exact record keys, integer-only JSON without duplicate keys, known dtypes,
overflow-bounded extents, payload covered contiguously to the end of the file). An index is checked against every
shard header it names; reals may sit in keys the compatible policy does not interpret. Header and index JSON nest at
most JSON_DEPTH levels and hold no unpaired surrogate escapes, as in the native reader. Results are frozen
dataclasses with absolute file offsets for each tensor.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
import hashlib
import itertools
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType

from .. import jsonfile
from ..files import RESERVED_DEVICE

INDEX_SUFFIX = ".safetensors.index.json"
INDEX_NAME = "model" + INDEX_SUFFIX
SINGLE_NAME = "model.safetensors"
RECORD_KEYS = frozenset({"dtype", "shape", "data_offsets"})
SHARD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors")
SHARD_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SHARD_DEPTH = 8     # path components of an index shard ("tensors/model-00001-of-00002.safetensors" has 2)
UNKNOWN_SHOWN = 8   # unknown index keys named in one error or note
JSON_DEPTH = 64     # nesting limit of headers and indexes (the native reader's at both call sites)

# Storage width in bits. Values are never decoded here; sub-byte dtypes must fill whole bytes.
DTYPE_BITS = MappingProxyType({
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8,
    "U16": 16, "I16": 16, "F16": 16, "BF16": 16,
    "U32": 32, "I32": 32, "F32": 32,
    "U64": 64, "I64": 64, "F64": 64,
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
})


class SafetensorsError(ValueError):
    """A safetensors file or index that breaks the format contract or a reader limit."""


class IndexPolicy:
    """How an index's metadata is read; every policy checks each tensor against the shard headers.

    STRICT allows only the keys metadata and weight_map and only metadata.total_size, which must equal the header
    bytes. COMPATIBLE (the default) also allows metadata.total_parameters and unknown keys, and records unknown keys
    and a total_size disagreement in Checkpoint.notes instead of failing.
    """
    STRICT = "strict"
    COMPATIBLE = "compatible"
    ALL = (STRICT, COMPATIBLE)


@dataclass(frozen=True)
class Limits:
    """Bounds checked before memory is committed; every limit is a positive integer below 2**63."""
    max_header_bytes: int = 100_000_000
    max_index_bytes: int = 64 << 20
    max_tensors: int = 1 << 20
    max_shards: int = 4096
    max_rank: int = 16
    max_dimension: int = 1 << 48
    max_file_bytes: int = 1 << 42
    max_total_bytes: int = 1 << 46
    max_name_bytes: int = 4096

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value <= 0 or value > (1 << 63) - 1:
                raise SafetensorsError(f"limit {name} must be an integer in [1, 2**63 - 1], not {value!r}")

    @classmethod
    def production(cls):
        """The defaults: 100 MB headers, 64 MiB indexes, 2**20 tensors, 4096 shards, rank 16, dimensions up to
        2**48, 4096-byte names, 4 TiB files and 64 TiB checkpoints."""
        return cls()


def _rebuild(cls, values):
    return cls(**{name: MappingProxyType(value) if isinstance(value, dict) else value
                  for name, value in values.items()})


class _Picklable:
    """Pickling, copy and deepcopy for frozen results whose MappingProxyType fields (one level deep) cannot be
    pickled."""

    def __reduce__(self):
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return _rebuild, (type(self), {name: dict(value) if isinstance(value, MappingProxyType) else value
                                       for name, value in values.items()})


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple
    shard: Path       # absolute path of the shard as named (a symlink stays a symlink path)
    data_begin: int   # absolute file byte offsets
    data_end: int

    @property
    def nbytes(self):
        return self.data_end - self.data_begin

    @property
    def bits(self):
        return DTYPE_BITS[self.dtype]

    @property
    def elements(self):
        return math.prod(self.shape)

    @property
    def element_aligned(self):
        """Whether the file offset is a multiple of the element size (always true below one byte per element)."""
        return self.data_begin % max(1, self.bits // 8) == 0


@dataclass(frozen=True)
class ShardInfo(_Picklable):
    path: Path
    size: int
    header_sha256: str
    file_identity: tuple     # (st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns) of the opened file
    tensors: tuple           # header order
    header_length: int = 0
    metadata: object = field(default_factory=lambda: MappingProxyType({}))   # __metadata__, str -> str
    header_only: bool = False

    @property
    def payload_offset(self):
        return 8 + self.header_length


@dataclass(frozen=True)
class Checkpoint(_Picklable):
    source: Path             # the index or the single .safetensors file
    tensors: object          # read-only {name: TensorInfo}, sorted by name
    shards: tuple            # index shards sorted by name, then extra files in the order given
    index_sha256: str | None = None
    notes: tuple = ()        # compatible-policy findings
    index_policy: str = IndexPolicy.COMPATIBLE
    header_only: bool = False

    @property
    def tensor_bytes(self):
        return sum(tensor.nbytes for tensor in self.tensors.values())


def _fail(message):
    raise SafetensorsError(message)


def _limits(limits):
    if limits is None:
        return Limits()
    if not isinstance(limits, Limits):
        _fail(f"limits must be a Limits instance, not {type(limits).__name__}")
    return limits


def _within_64_bits(value, label):
    if type(value) is int and not -(1 << 63) <= value < (1 << 64):
        _fail(f"{label}: integer exceeds 64 bits")


def _integer(value, label, maximum):
    _within_64_bits(value, label)
    if type(value) is not int:
        _fail(f"{label} must be an integer, not {_short(value)}")
    if value < 0:
        _fail(f"{label} {_short(value)} is negative")
    if value > maximum:
        _fail(f"{label} {_short(value)} exceeds {maximum}")
    return value


def _offset(value, label, limit, header_only):
    _within_64_bits(value, f"{label}: data offset")
    if type(value) is not int:
        _fail(f"{label}: data offset must be an integer, not {_short(value)}")
    if value < 0:
        _fail(f"{label}: data offset {_short(value)} is negative")
    if value > limit:
        _fail(f"{label}: data offset {_short(value)} exceeds the {limit}-byte file limit" if header_only else
              f"{label}: data offset {_short(value)} is past the {limit}-byte payload")
    return value


def _short(value):
    if value is None:
        return "null"
    if value is True or value is False:
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return "an object" if isinstance(value, dict) else "an array"
    if isinstance(value, int) and value.bit_length() > 256:
        return "a very large integer"
    text = repr(value)
    return text if len(text) <= 48 else text[:45] + "..."


def _keys(keys):
    """A bounded listing of sorted keys for messages: at most UNKNOWN_SHOWN, then the count."""
    shown = "[" + ", ".join(_short(key) for key in keys[:UNKNOWN_SHOWN])
    return shown + "]" if len(keys) <= UNKNOWN_SHOWN else shown + f", ...] ({len(keys)} in all)"


def _unknown(keys, what, path, strict, notes):
    # Strict fails; compatible records one note however many keys there are.
    if not keys:
        return
    if strict:
        _fail(f"{path}: unsupported {what} keys {_keys(keys)}")
    notes.append(f"{what} key {_short(keys[0])} is not interpreted" if len(keys) == 1 else
                 f"{what} keys {_keys(keys)} are not interpreted")


def _name(value, label, limits):
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string, not {_short(value)}")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        _fail(f"{label} {_short(value)} is not valid UTF-8")
    if len(encoded) > limits.max_name_bytes:
        _fail(f"{label} of {len(encoded)} bytes exceeds the {limits.max_name_bytes}-byte limit")
    if "\0" in value:
        _fail(f"{label} {_short(value)} contains NUL")
    return value


def _shard_name(value, label, limits):
    _name(value, label, limits)
    if not SHARD_NAME.fullmatch(value):
        _fail(f"{label} {_short(value)} is not a portable flat .safetensors file name")
    if RESERVED_DEVICE.fullmatch(value.split(".", 1)[0].upper()):
        _fail(f"{label} {_short(value)} is a reserved device name")
    return value


def _shard_path(value, label, limits):
    """An index shard: a relative "/"-separated path of portable components ending in a flat shard name, as
    checkpoints that keep their shards in a subfolder (tensors/...) list them. Every component starts with a letter
    or digit, so none is "." or "..", and nothing leaves the index's directory but through a link."""
    _name(value, label, limits)
    parts = value.split("/")
    if (len(parts) > SHARD_DEPTH or not SHARD_NAME.fullmatch(parts[-1]) or
            not all(SHARD_COMPONENT.fullmatch(part) for part in parts[:-1])):
        _fail(f"{label} {_short(value)} is not a portable relative .safetensors path")
    for part in parts:
        if RESERVED_DEVICE.fullmatch(part.split(".", 1)[0].upper()):
            _fail(f"{label} {_short(value)} is a reserved device name")
    return value


def _shape(value, dtype, limits, label):
    if len(value) > limits.max_rank:
        _fail(f"{label}: rank {len(value)} exceeds the {limits.max_rank} limit")
    dims = tuple(_integer(dim, f"{label}: dimension", limits.max_dimension) for dim in value)
    bits = DTYPE_BITS[dtype]
    elements = 0 if 0 in dims else 1
    maximum = limits.max_file_bytes * 8 // bits
    for dim in dims:
        if elements and dim > maximum // elements:
            _fail(f"{label}: shape {list(dims)} of {dtype} exceeds the {limits.max_file_bytes}-byte file limit")
        elements *= dim
    if elements * bits % 8:
        _fail(f"{label}: {elements} elements of {dtype} do not fill whole bytes")
    return dims, elements * bits // 8


def _json(data, label, max_bytes, allow_real):
    try:
        value = jsonfile.loads(data, label, max_bytes=max_bytes, allow_real=allow_real, max_depth=JSON_DEPTH)
    except jsonfile.JsonError as error:
        raise SafetensorsError(str(error)) from None
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _path_identity(info):
    # Windows answers a path stat from directory metadata that can lag the open file's own timestamps and size,
    # so only the file ID ties a path stat to a descriptor there; descriptor stats still compare in full.
    return (info.st_dev, info.st_ino) if os.name == "nt" else _identity(info)


@contextmanager
def _regular_file(path, follow_symlinks):
    # Yields a raw descriptor: a buffered stream would fill its buffer past the header with tensor bytes.
    # Portable shard paths plus this check reject path escapes; without follow_symlinks also links. This detects
    # changes during the read; it is neither a sandbox against a hostile filesystem nor a payload digest.
    before = path.stat() if follow_symlinks else path.lstat()
    if not stat.S_ISREG(before.st_mode):
        _fail(f"{path}: expected a regular file" if follow_symlinks else
              f"{path}: expected a regular file that is not a symlink")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    if not follow_symlinks:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or _path_identity(opened) != _path_identity(before):
            _fail(f"{path}: file changed while opening")
        yield fd, opened
        if _identity(os.fstat(fd)) != _identity(opened):
            _fail(f"{path}: file changed during metadata read")
    finally:
        os.close(fd)


def _read_up_to(fd, count):
    """Exactly `count` bytes from the descriptor's position, or fewer at end of file; never asks for more."""
    chunks, remaining = [], count
    while remaining:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def inspect_safetensors(path, limits=None, *, header_only=False, follow_symlinks=True):
    """Validate one .safetensors file from its length prefix and header; returns a ShardInfo.

    header_only skips the checks against the file's payload (offsets within it, no trailing bytes) for files that
    carry only a header; the offsets must still cover a contiguous extent from 0 without holes or overlaps.
    follow_symlinks=False refuses a symlink in place of the file.
    """
    limits = _limits(limits)
    path = Path(path)
    if path.suffix != ".safetensors":
        _fail(f"{path}: expected a .safetensors file")
    with _regular_file(path, follow_symlinks) as (fd, info):
        size = info.st_size
        if size > limits.max_file_bytes:
            _fail(f"{path}: {size} bytes exceed the {limits.max_file_bytes}-byte file limit")
        prefix = _read_up_to(fd, 8)
        if len(prefix) != 8:
            _fail(f"{path}: truncated header length")
        length = int.from_bytes(prefix, "little")
        if not length:
            _fail(f"{path}: header length is zero")
        if length > limits.max_header_bytes:
            _fail(f"{path}: header length {length} exceeds the {limits.max_header_bytes}-byte limit")
        if length > size - 8:
            _fail(f"{path}: header length {length} exceeds the {size - 8} bytes after the length prefix")
        data = _read_up_to(fd, length)
        if len(data) != length:
            _fail(f"{path}: truncated header")
        if not data.startswith(b"{"):
            _fail(f"{path}: header must start with '{{'")
        header = _json(data, f"{path}: header", limits.max_header_bytes, allow_real=False)
        metadata = header.get("__metadata__", {})
        if not isinstance(metadata, dict) or any(not isinstance(value, str) for value in metadata.values()):
            _fail(f"{path}: __metadata__ must map strings to strings")
        payload_start = 8 + length
        payload_size = size - payload_start
        offset_limit = limits.max_file_bytes if header_only else payload_size
        tensors = []
        for name, record in header.items():
            if name == "__metadata__":
                continue
            _name(name, f"{path}: tensor name", limits)
            if len(tensors) >= limits.max_tensors:
                _fail(f"{path}: more than {limits.max_tensors} tensors")
            label = f"{path}: tensor {name!r}"
            if not isinstance(record, dict) or set(record) != RECORD_KEYS:
                _fail(f"{label}: expected exactly the keys dtype, shape and data_offsets")
            dtype = record["dtype"]
            if not isinstance(dtype, str) or dtype not in DTYPE_BITS:
                _fail(f"{label}: unsupported dtype {_short(dtype)}")
            if not isinstance(record["shape"], list):
                _fail(f"{label}: shape must be an array")
            shape, nbytes = _shape(record["shape"], dtype, limits, label)
            offsets = record["data_offsets"]
            if not isinstance(offsets, list) or len(offsets) != 2:
                _fail(f"{label}: data_offsets must be [begin, end]")
            begin, end = (_offset(offset, label, offset_limit, header_only) for offset in offsets)
            if end < begin:
                _fail(f"{label}: data_offsets end {end} precedes begin {begin}")
            if end - begin != nbytes:
                _fail(f"{label}: data_offsets [{begin}, {end}) hold {end - begin} bytes but shape {list(shape)} "
                      f"of {dtype} needs {nbytes}")
            tensors.append(TensorInfo(name, dtype, shape, path.absolute(), payload_start + begin, payload_start + end))
        cursor = payload_start
        for tensor in sorted(tensors, key=lambda t: (t.data_begin, t.data_end, t.name)):
            if tensor.data_begin != cursor:
                kind = "hole" if tensor.data_begin > cursor else "overlap"
                _fail(f"{path}: {kind} at payload offset {cursor - payload_start} before tensor {tensor.name!r}")
            cursor = tensor.data_end
        if not header_only and cursor != size:
            _fail(f"{path}: payload has {size - cursor} unindexed trailing bytes")
        return ShardInfo(path.absolute(), size, hashlib.sha256(data).hexdigest(), _identity(info), tuple(tensors),
                         length, MappingProxyType(dict(metadata)), bool(header_only))


def _resolve_directory(directory):
    for name in (INDEX_NAME, SINGLE_NAME):
        candidate = directory / name
        if candidate.is_file() or candidate.is_symlink():
            return candidate
    candidates = sorted(path for path in directory.iterdir()
                        if path.suffix == ".safetensors" and (path.is_file() or path.is_symlink()))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        _fail(f"{directory}: no {INDEX_NAME}, {SINGLE_NAME} or other .safetensors file")
    names = ", ".join(path.name for path in candidates[:8]) + (", ..." if len(candidates) > 8 else "")
    _fail(f"{directory}: {len(candidates)} .safetensors files and no index; name one: {names}")


def _extra_names(extra_files, limits):
    if isinstance(extra_files, (str, bytes, os.PathLike)):
        _fail("extra_files must be a sequence of file names, not one name")
    names = list(itertools.islice(extra_files, limits.max_shards + 1))   # bounded even for an endless iterable
    if len(names) > limits.max_shards:
        _fail(f"extra_files lists more than {limits.max_shards} files, the shard limit")
    seen = set()
    for name in names:
        _shard_name(name, "extra file name", limits)
        if name in seen:
            _fail(f"extra file {name!r} is listed twice")
        seen.add(name)
    return names


def _extent(shard):
    # Bytes a shard occupies: its size, or with header_only the end of its last declared tensor when that is larger.
    return max(shard.size, max((tensor.data_end for tensor in shard.tensors), default=0))


def _read_index(path, limits, follow_symlinks):
    with _regular_file(path, follow_symlinks) as (fd, info):
        if not info.st_size or info.st_size > limits.max_index_bytes:
            _fail(f"{path}: index size {info.st_size} is not in [1, {limits.max_index_bytes}] bytes")
        data = _read_up_to(fd, info.st_size + 1)   # one byte more detects a file that grew
        if len(data) != info.st_size:
            _fail(f"{path}: index changed or truncated while reading")
    return data, _json(data, str(path), limits.max_index_bytes, allow_real=True)   # typed fields check themselves


def inspect_model(path, limits=None, *, index_policy=IndexPolicy.COMPATIBLE, header_only=False,
                  follow_symlinks=True, extra_files=()):
    """Inspect a .safetensors file, a .safetensors.index.json or a model directory; returns a Checkpoint.

    A directory resolves to model.safetensors.index.json, else model.safetensors, else its only .safetensors file;
    anything else is an error that names the candidates. extra_files are flat names beside the source (for example
    a draft-layer file next to an index) whose tensors join the checkpoint; no tensor may appear twice.
    max_total_bytes bounds the index plus every file's size or, with header_only, its declared extent when larger.
    """
    limits = _limits(limits)
    if index_policy not in IndexPolicy.ALL:
        _fail(f"index_policy must be 'strict' or 'compatible', not {_short(index_policy)}")
    strict = index_policy == IndexPolicy.STRICT
    extras = _extra_names(extra_files, limits)
    path = Path(path)
    if path.is_dir():
        path = _resolve_directory(path)
    notes, index_sha256 = [], None
    if path.suffix == ".safetensors":
        shard = inspect_safetensors(path, limits, header_only=header_only, follow_symlinks=follow_symlinks)
        total = _extent(shard)
        if total > limits.max_total_bytes:
            _fail(f"{path}: {total} bytes exceed the {limits.max_total_bytes}-byte checkpoint limit")
        shards, members = [shard], {path.name}
        tensors = {tensor.name: tensor for tensor in shard.tensors}
    elif path.name.endswith(INDEX_SUFFIX):
        data, index = _read_index(path, limits, follow_symlinks)
        index_sha256 = hashlib.sha256(data).hexdigest()
        if "weight_map" not in index:
            _fail(f"{path}: index has no weight_map")
        _unknown(sorted(set(index) - {"metadata", "weight_map"}), "index", path, strict, notes)
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            _fail(f"{path}: weight_map must be a non-empty object")
        if len(weight_map) > limits.max_tensors:
            _fail(f"{path}: weight_map lists {len(weight_map)} tensors, more than {limits.max_tensors}")
        for key, name in weight_map.items():
            _name(key, f"{path}: weight_map tensor name", limits)
            _shard_path(name, f"{path}: weight_map shard for {key!r}", limits)
        names = sorted(set(weight_map.values()))
        if len(names) > limits.max_shards:
            _fail(f"{path}: {len(names)} shards exceed the {limits.max_shards}-shard limit")
        metadata = index.get("metadata", {})
        if not isinstance(metadata, dict):
            _fail(f"{path}: index metadata must be an object")
        known = {"total_size"} if strict else {"total_size", "total_parameters"}
        _unknown(sorted(set(metadata) - known), "index metadata", path, strict, notes)
        total_size = metadata.get("total_size")
        if "total_size" in metadata:
            _integer(total_size, f"{path}: index metadata total_size", limits.max_total_bytes)
        if "total_parameters" in metadata and not strict:
            _integer(metadata["total_parameters"], f"{path}: index metadata total_parameters", (1 << 63) - 1)
        shards, tensors, total, members = [], {}, len(data), set(names)
        for name in names:
            shard = inspect_safetensors(path.parent / name, limits, header_only=header_only,
                                        follow_symlinks=follow_symlinks)
            total += _extent(shard)
            if total > limits.max_total_bytes:
                _fail(f"{path}: checkpoint files exceed the {limits.max_total_bytes}-byte limit")
            for tensor in shard.tensors:
                declared = weight_map.get(tensor.name)
                if tensor.name in tensors:
                    _fail(f"{path}: tensor {tensor.name!r} appears in shards {tensors[tensor.name].shard.name!r} "
                          f"and {name!r}")
                if declared is None:
                    _fail(f"{path}: shard {name!r} holds tensor {tensor.name!r}, which weight_map does not list")
                if declared != name:
                    _fail(f"{path}: weight_map puts tensor {tensor.name!r} in {declared!r} but {name!r} holds it")
                tensors[tensor.name] = tensor
            shards.append(shard)
        missing = sorted(set(weight_map) - set(tensors))
        if missing:
            _fail(f"{path}: {len(missing)} weight_map tensors are absent from the shard headers, "
                  f"such as {missing[0]!r}")
        header_bytes = sum(tensor.nbytes for tensor in tensors.values())
        if total_size is not None and total_size != header_bytes:
            message = (f"index metadata total_size {total_size} differs from the {header_bytes} tensor bytes "
                       f"in the shards")
            if strict:
                _fail(f"{path}: {message}")
            notes.append(message)
    else:
        _fail(f"{path}: expected a .safetensors file, a {INDEX_SUFFIX} index or a model directory")
    for name in extras:
        if name in members:
            _fail(f"{path}: extra file {name!r} is already part of the checkpoint")
        if len(shards) >= limits.max_shards:
            _fail(f"{path}: {len(shards) + 1} files exceed the {limits.max_shards}-shard limit")
        shard = inspect_safetensors(path.parent / name, limits, header_only=header_only,
                                    follow_symlinks=follow_symlinks)
        total += _extent(shard)
        if total > limits.max_total_bytes:
            _fail(f"{path}: checkpoint files exceed the {limits.max_total_bytes}-byte limit")
        for tensor in shard.tensors:
            if tensor.name in tensors:
                _fail(f"{path}: tensor {tensor.name!r} appears in {tensors[tensor.name].shard.name!r} and "
                      f"extra file {name!r}")
            tensors[tensor.name] = tensor
            if len(tensors) > limits.max_tensors:
                _fail(f"{path}: more than {limits.max_tensors} tensors across files")
        shards.append(shard)
    ordered = MappingProxyType({name: tensors[name] for name in sorted(tensors)})
    return Checkpoint(path.absolute(), ordered, tuple(shards), index_sha256, tuple(notes), index_policy,
                      bool(header_only))
