"""Write files in torch.save's zip format with the standard library, for tests that must not need torch.

torch_bytes(tensors) -> bytes, tensors a list of (name, dtype, shape, data) with dtype a safetensors name and data
the tensor's bytes (zeros when None). Tensors that name the same storage key share it (tied weights):
(name, dtype, shape, data, key, offset) with offset in elements. pickle_bytes(...) emits the pickle opcodes torch
writes, so a test can also build pickles torch never would (a foreign global, a non-tensor value)."""
import io
import math
import struct
import zipfile

STORAGES = {"F32": "FloatStorage", "F16": "HalfStorage", "BF16": "BFloat16Storage", "I64": "LongStorage",
            "I32": "IntStorage", "I8": "CharStorage", "U8": "ByteStorage", "BOOL": "BoolStorage"}
SIZES = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1}
TORCH_DTYPES = {"F8_E4M3": "float8_e4m3fn"}


class Global:
    def __init__(self, module, name):
        self.module, self.name = module, name


class Call:
    def __init__(self, function, *args):
        self.function, self.args = function, args


class PersistentId:
    def __init__(self, value):
        self.value = value


def _emit(out, value):
    if isinstance(value, Global):
        out.write(b"c" + value.module.encode() + b"\n" + value.name.encode() + b"\n")
    elif isinstance(value, Call):
        _emit(out, value.function)
        _emit(out, tuple(value.args))
        out.write(b"R")
    elif isinstance(value, PersistentId):
        _emit(out, value.value)
        out.write(b"Q")
    elif value is None:
        out.write(b"N")
    elif value is True or value is False:
        out.write(b"\x88" if value else b"\x89")
    elif isinstance(value, int):
        if 0 <= value < 1 << 31:
            out.write(b"J" + struct.pack("<i", value))
        else:
            data = value.to_bytes((value.bit_length() + 8) // 8, "little", signed=True)
            out.write(b"\x8a" + bytes([len(data)]) + data)
    elif isinstance(value, str):
        data = value.encode()
        out.write(b"X" + struct.pack("<I", len(data)) + data)
    elif isinstance(value, tuple):
        out.write(b"(")
        for item in value:
            _emit(out, item)
        out.write(b"t")
    elif isinstance(value, dict):
        out.write(b"}(")
        for key, item in value.items():
            _emit(out, key)
            _emit(out, item)
        out.write(b"u")
    else:
        raise TypeError(type(value))


def pickle_bytes(value):
    out = io.BytesIO()
    out.write(b"\x80\x02")
    _emit(out, value)
    out.write(b".")
    return out.getvalue()


def strides(shape):
    result, step = [], 1
    for n in reversed(shape):
        result.insert(0, step)
        step *= n
    return tuple(result)


def tensor_call(dtype, shape, key, offset=0, numel=None, stride=None):
    """What torch pickles for one tensor: _rebuild_tensor_v2 over a typed storage, or _rebuild_tensor_v3 over an
    untyped one for the dtypes without a storage class (float8)."""
    numel = math.prod(shape) if numel is None else numel
    stride = strides(shape) if stride is None else stride
    hooks = Global("collections", "OrderedDict")   # backward hooks: an empty OrderedDict
    if dtype in STORAGES:
        storage = PersistentId(("storage", Global("torch", STORAGES[dtype]), key, "cpu", numel))
        return Call(Global("torch._utils", "_rebuild_tensor_v2"), storage, offset, tuple(shape), stride, False,
                    Call(hooks))
    storage = PersistentId(("storage", Global("torch.storage", "UntypedStorage"), key, "cpu", numel * SIZES[dtype]))
    return Call(Global("torch._utils", "_rebuild_tensor_v3"), storage, offset, tuple(shape), stride, False,
                Call(hooks), Global("torch", TORCH_DTYPES[dtype]))


def torch_bytes(tensors, root="archive", state=None):
    """A torch.save zip of the state dict `tensors` (see the module docstring); `state` replaces the pickled
    value."""
    storages, entries = {}, {}
    for index, item in enumerate(tensors):
        name, dtype, shape, data = item[:4]
        key, offset = (item[4], item[5]) if len(item) > 4 else (str(index), 0)
        size = math.prod(shape) * SIZES[dtype]
        data = bytes(size) if data is None else data
        if key in storages:
            entries[name] = tensor_call(dtype, shape, key, offset, len(storages[key]) // SIZES[dtype])
            continue
        storages[key] = data
        entries[name] = tensor_call(dtype, shape, key, offset)
    value = Call(Global("collections", "OrderedDict"), tuple((k, v) for k, v in entries.items())) \
        if state is None else state
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr(f"{root}/data.pkl", pickle_bytes(value))
        archive.writestr(f"{root}/byteorder", "little")
        for key, data in storages.items():
            archive.writestr(f"{root}/data/{key}", data)
        archive.writestr(f"{root}/version", "3\n")
    return out.getvalue()
