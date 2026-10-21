"""A small GGUF writer for fixtures (standard library only): metadata of every value type and tensors whose bytes
are given or zero, laid out as GGUF converters write them unless a test asks for a broken layout."""
import math
import struct

from crit_core.model.gguf import TYPE_BY_NAME

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
_FORMATS = {UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h", UINT32: "<I", INT32: "<i", FLOAT32: "<f",
            BOOL: "<B", UINT64: "<Q", INT64: "<q", FLOAT64: "<d"}


def _string(text):
    data = text.encode("utf-8") if isinstance(text, str) else text
    return struct.pack("<Q", len(data)) + data


def _kind(value):
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, int):
        return UINT32 if 0 <= value < 1 << 32 else INT64
    if isinstance(value, float):
        return FLOAT32
    if isinstance(value, (str, bytes)):
        return STRING
    return ARRAY


def _value(kind, value, element=None):
    if kind == STRING:
        return _string(value)
    if kind == ARRAY:
        items = list(value)
        element = element if element is not None else (_kind(items[0]) if items else UINT32)
        body = b"".join(_value(element, item) for item in items)
        return struct.pack("<IQ", element, len(items)) + body
    return struct.pack(_FORMATS.get(kind, "<I"), value)


def tensor_bytes(dtype, shape):
    """Bytes of a tensor of ggml type `dtype` with row-major `shape`."""
    ggml = TYPE_BY_NAME[dtype]
    return shape[-1] // ggml.block * ggml.type_size * math.prod(shape[:-1])


def gguf_bytes(metadata=(), tensors=(), *, version=3, alignment=None, offsets=None, trailing=b"", magic=b"GGUF"):
    """metadata: [(key, value)] or [(key, value, type)] or [(key, value, ARRAY, element type)]; tensors:
    [(name, dtype, row-major shape)] or [(name, dtype, shape, data)]. offsets overrides the tensor offsets."""
    entries = []
    for entry in metadata:
        key, value = entry[0], entry[1]
        kind = entry[2] if len(entry) > 2 else _kind(value)
        element = entry[3] if len(entry) > 3 else None
        entries.append(_string(key) + struct.pack("<I", kind) + _value(kind, value, element))
    align = alignment or 32
    table, payload, cursor = [], [], 0
    for i, tensor in enumerate(tensors):
        name, dtype, shape = tensor[:3]
        nbytes = tensor_bytes(dtype, shape)
        data = tensor[3] if len(tensor) > 3 else bytes(nbytes)
        cursor = -(-cursor // align) * align
        offset = offsets[i] if offsets else cursor
        ne = tuple(reversed(shape))
        table.append(_string(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", n) for n in ne)
                     + struct.pack("<IQ", TYPE_BY_NAME[dtype].id, offset))
        payload.append((cursor, data))
        cursor += len(data)
    head = (magic + struct.pack("<IQQ", version, len(table), len(entries)) + b"".join(entries) + b"".join(table))
    head += bytes(-len(head) % align)
    body = bytearray()
    for start, data in payload:
        body += bytes(start - len(body)) + data
    body += bytes(-len(body) % align)
    return head + bytes(body) + trailing
