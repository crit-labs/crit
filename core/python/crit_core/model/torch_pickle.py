"""Strict reader for PyTorch state dicts (torch.save's zip format: pytorch_model.bin, *.pt, *.pth), standard
library only: the tensor table comes from the pickle without running it and tensor bytes are never read.

torch.save writes a zip archive of STORED (uncompressed) entries:

    <name>/data.pkl     a pickle of the state dict: {name: torch._utils._rebuild_tensor_v2(storage, offset, size,
                        stride, ...)}, each storage a persistent id ("storage", storage type, key, location, numel)
    <name>/data/<key>   the bytes of storage <key>
    <name>/byteorder    "little" (absent in files from before torch 1.10, which are little-endian)

Unpickling an arbitrary pickle runs arbitrary code. This reader never imports what the pickle names: its unpickler
resolves only the globals a state dict uses (collections.OrderedDict, torch's tensor and parameter rebuild
functions, storage types, dtypes and torch.Size) to inert stand-ins, and refuses any other global, so a file that
would run code when torch.load()s it is refused instead. Tensors must be contiguous views of their storage; tied
weights share one. The legacy (pre-1.6, non-zip) format is refused. Results reuse crit_core.model's TensorInfo
with safetensors dtype names (BF16, F32, I64, F8_E4M3, ...) and absolute file offsets, so everything that counts
safetensors tensors counts these the same way.
"""
from collections import OrderedDict
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import pickle
import struct
from types import MappingProxyType
import zipfile

from .safetensors import DTYPE_BITS, Limits, TensorInfo, _identity, _short

SUFFIXES = (".bin", ".pt", ".pth")
INDEX_SUFFIX = ".bin.index.json"
INDEX_NAME = "pytorch_model" + INDEX_SUFFIX
SINGLE_NAME = "pytorch_model.bin"
MAX_PICKLE_BYTES = 256 << 20
ZIP_MAGIC = b"PK\x03\x04"
LEGACY_MAGIC = b"\x80\x02\x8a\x0a\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19"   # torch.save before 1.6
NOT_WEIGHTS = ("training_args", "optimizer", "scheduler", "rng_state")   # what a trainer saves beside a model

# torch storage classes and dtypes -> safetensors dtype names.
STORAGE_DTYPES = {
    "FloatStorage": "F32", "HalfStorage": "F16", "BFloat16Storage": "BF16", "DoubleStorage": "F64",
    "LongStorage": "I64", "IntStorage": "I32", "ShortStorage": "I16", "CharStorage": "I8", "ByteStorage": "U8",
    "BoolStorage": "BOOL",
}
TORCH_DTYPES = {
    "float32": "F32", "float": "F32", "float16": "F16", "half": "F16", "bfloat16": "BF16", "float64": "F64",
    "double": "F64", "int64": "I64", "long": "I64", "int32": "I32", "int": "I32", "int16": "I16", "short": "I16",
    "int8": "I8", "uint8": "U8", "bool": "BOOL", "float8_e4m3fn": "F8_E4M3", "float8_e5m2": "F8_E5M2",
    "float8_e8m0fnu": "F8_E8M0", "uint16": "U16", "uint32": "U32", "uint64": "U64",
}


class TorchFileError(ValueError):
    """A PyTorch file that is not a plain state dict in torch.save's zip format, or breaks a reader limit."""


def _fail(message):
    raise TorchFileError(message)


@dataclass(frozen=True)
class _StorageType:
    dtype: str | None   # None: an UntypedStorage, whose tensors name their dtype


@dataclass(frozen=True)
class _Dtype:
    name: str


@dataclass(frozen=True)
class _Storage:
    dtype: str | None
    key: str
    numel: int


@dataclass(frozen=True)
class _Tensor:
    storage: _Storage
    offset: int
    shape: tuple
    stride: tuple
    dtype: str


def _rebuild_v2(storage, offset, size, stride, *_):
    if not isinstance(storage, _Storage) or storage.dtype is None:
        _fail("a tensor's storage is not a typed storage")
    return _Tensor(storage, offset, tuple(size), tuple(stride), storage.dtype)


def _rebuild_v3(storage, offset, size, stride, requires_grad, hooks, dtype, *_):
    if not isinstance(storage, _Storage) or not isinstance(dtype, _Dtype):
        _fail("a tensor's storage or dtype is not one a state dict holds")
    return _Tensor(storage, offset, tuple(size), tuple(stride), dtype.name)


def _rebuild_parameter(data, *_):
    return data


_GLOBALS = {
    ("collections", "OrderedDict"): OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_v2,
    ("torch._utils", "_rebuild_tensor_v3"): _rebuild_v3,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
    ("torch._utils", "_rebuild_parameter_with_state"): _rebuild_parameter,
    ("torch", "Size"): tuple,
    ("torch.storage", "UntypedStorage"): _StorageType(None),
}


class _StateDictUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        found = _GLOBALS.get((module, name))
        if found is not None:
            return found
        if module == "torch" and name in STORAGE_DTYPES:
            return _StorageType(STORAGE_DTYPES[name])
        if module == "torch" and name in TORCH_DTYPES:
            return _Dtype(TORCH_DTYPES[name])
        _fail(f"the pickle references {module}.{name}, which a state dict does not; crit does not run it")

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):
            _fail(f"unknown persistent id {_short(pid)}")
        _, kind, key, _location, numel = pid
        if not isinstance(kind, _StorageType) or not isinstance(key, str) or type(numel) is not int or numel < 0:
            _fail(f"malformed storage {_short(pid)}")
        return _Storage(kind.dtype, key, numel)


@dataclass(frozen=True)
class TorchFile:
    path: Path
    size: int
    file_identity: tuple
    tensors: tuple       # TensorInfo in pickle order


def _entry_offset(handle, info):
    """Absolute offset of a STORED zip entry's bytes (past its local header)."""
    handle.seek(info.header_offset)
    header = handle.read(30)
    if len(header) != 30 or header[:4] != b"PK\x03\x04":
        _fail(f"{info.filename}: bad zip local header")
    name_length, extra_length = struct.unpack("<HH", header[26:30])
    return info.header_offset + 30 + name_length + extra_length


def inspect_torch(path, limits=None):
    """Validate one torch.save zip file from its directory and pickle; returns a TorchFile."""
    limits = Limits() if limits is None else limits
    path = Path(path)
    info = os.stat(path)
    with open(path, "rb") as handle:
        start = handle.read(len(LEGACY_MAGIC))
        if start == LEGACY_MAGIC:
            _fail(f"{path}: torch.save's legacy format (before torch 1.6) is not read; re-save it with a current "
                  "torch or convert it to safetensors")
        if not start.startswith(ZIP_MAGIC):
            _fail(f"{path}: not a file torch.save wrote")
        try:
            archive = zipfile.ZipFile(handle)
        except zipfile.BadZipFile as error:
            _fail(f"{path}: {error}")
        entries = {item.filename: item for item in archive.infolist()}
        pickles = [name for name in entries if name.endswith("/data.pkl") and name.count("/") == 1]
        if len(pickles) != 1:
            _fail(f"{path}: expected one <name>/data.pkl, found {len(pickles)}")
        root = pickles[0][:-len("data.pkl")]
        order = entries.get(root + "byteorder")
        if order is not None and archive.read(order) != b"little":
            _fail(f"{path}: big-endian tensors are not read")
        if entries[pickles[0]].file_size > MAX_PICKLE_BYTES:
            _fail(f"{path}: data.pkl exceeds {MAX_PICKLE_BYTES} bytes")
        try:
            state = _StateDictUnpickler(io.BytesIO(archive.read(pickles[0]))).load()
        except TorchFileError as error:
            _fail(f"{path}: {error}")
        except (pickle.UnpicklingError, EOFError, TypeError, ValueError, AttributeError, IndexError) as error:
            _fail(f"{path}: data.pkl: {error}")
        if not isinstance(state, dict):
            _fail(f"{path}: the pickle is a {type(state).__name__}, not a state dict of tensors")
        if len(state) > limits.max_tensors:
            _fail(f"{path}: {len(state)} tensors, more than {limits.max_tensors}")
        tensors = []
        for name, tensor in state.items():
            if not isinstance(name, str) or not name:
                _fail(f"{path}: tensor name {_short(name)} is not a non-empty string")
            if not isinstance(tensor, _Tensor):
                _fail(f"{path}: {name!r} is not a tensor; a state dict holds only tensors")
            if tensor.dtype not in DTYPE_BITS:
                _fail(f"{path}: {name!r} has dtype {tensor.dtype}")
            shape, stride = tensor.shape, tensor.stride
            if any(type(n) is not int or n < 0 for n in shape + stride) or len(shape) != len(stride):
                _fail(f"{path}: {name!r} has shape {list(shape)} and stride {list(stride)}")
            expected, step = [], 1
            for n in reversed(shape):
                expected.insert(0, step)
                step *= n
            if any(n > 1 and s != e for n, s, e in zip(shape, stride, expected)):
                _fail(f"{path}: {name!r} is a non-contiguous view of its storage (stride {list(stride)})")
            item = DTYPE_BITS[tensor.dtype] // 8
            entry = entries.get(f"{root}data/{tensor.storage.key}")
            if entry is None:
                _fail(f"{path}: storage {tensor.storage.key!r} of {name!r} is missing")
            if entry.compress_type != zipfile.ZIP_STORED:
                _fail(f"{path}: storage {tensor.storage.key!r} is compressed")
            elements = math.prod(shape)
            begin = tensor.offset * item
            if begin + elements * item > entry.file_size:
                _fail(f"{path}: {name!r} ends past its storage's {entry.file_size} bytes")
            start = _entry_offset(handle, entry)
            tensors.append(TensorInfo(name, tensor.dtype, shape, path.absolute(), start + begin,
                                      start + begin + elements * item))
    return TorchFile(path.absolute(), info.st_size, _identity(info), tuple(tensors))


@dataclass(frozen=True)
class TorchCheckpoint:
    source: Path
    tensors: object      # read-only {name: TensorInfo}, sorted by name
    shards: tuple        # TorchFile per file, index order

    @property
    def tensor_bytes(self):
        return sum(t.nbytes for t in self.tensors.values())


def is_state_dict_file(path):
    """Whether `path` is a file torch.save wrote (or a .bin.index.json over such files) and not a trainer's state.
    .bin is a suffix other formats use too (an engine's own weights), so a .bin file must start as torch.save's do."""
    path = Path(path)
    if path.name.startswith(NOT_WEIGHTS):
        return False
    if path.name.endswith(INDEX_SUFFIX) or path.suffix in (".pt", ".pth"):
        return True
    try:
        with open(path, "rb") as handle:
            start = handle.read(len(LEGACY_MAGIC))
    except OSError:
        return False
    return start.startswith(ZIP_MAGIC) or start == LEGACY_MAGIC


def weight_names(folder):
    return sorted(p.name for p in Path(folder).iterdir()
                  if p.name.endswith(SUFFIXES + (INDEX_SUFFIX,)) and (p.is_file() or p.is_symlink())
                  and is_state_dict_file(p))


def inspect_torch_model(path, limits=None):
    """A torch state-dict checkpoint: a file, a .bin.index.json, or a folder (its pytorch_model.bin.index.json, else
    pytorch_model.bin, else its only state-dict file)."""
    limits = Limits() if limits is None else limits
    path = Path(path)
    if path.is_dir():
        names = weight_names(path)
        for name in (INDEX_NAME, SINGLE_NAME):
            if name in names:
                path = path / name
                break
        else:
            files = [n for n in names if not n.endswith(INDEX_SUFFIX)]
            if len(files) != 1:
                _fail(f"{path}: {len(files)} PyTorch weight files and no index; name one: {', '.join(files[:8])}")
            path = path / files[0]
    if path.name.endswith(INDEX_SUFFIX):
        index = json.loads(path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            _fail(f"{path}: index has no weight_map")
        files = sorted(set(weight_map.values()))
        if any(not isinstance(f, str) or "/" in f or "\\" in f or f.startswith(".") for f in files):
            _fail(f"{path}: weight_map names a file outside the folder")
        shards = [inspect_torch(path.parent / f, limits) for f in files]
    else:
        shards = [inspect_torch(path, limits)]
        weight_map = None
    tensors = {}
    for shard in shards:
        for tensor in shard.tensors:
            if tensor.name in tensors:
                _fail(f"{path}: tensor {tensor.name!r} appears in two files")
            if weight_map is not None and weight_map.get(tensor.name) != shard.path.name:
                _fail(f"{path}: weight_map does not put {tensor.name!r} in {shard.path.name}")
            tensors[tensor.name] = tensor
    if weight_map is not None and set(weight_map) - set(tensors):
        missing = sorted(set(weight_map) - set(tensors))
        _fail(f"{path}: {len(missing)} weight_map tensors are in no file, first {missing[0]!r}")
    return TorchCheckpoint(path.absolute(), MappingProxyType(dict(sorted(tensors.items()))), tuple(shards))
