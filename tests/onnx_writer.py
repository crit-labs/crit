"""Write ONNX model files with the standard library, for tests that must not need the onnx package.

onnx_bytes(initializers, nodes) -> bytes of a ModelProto whose graph holds them. initializers: (name, dtype, dims,
data) with dtype an ONNX TensorProto.DataType number and data the raw bytes, or (name, dtype, dims, None,
{"location": ..., "offset": ..., "length": ...}) for external data. nodes: (op_type, name, inputs, {attribute:
integer})."""
import struct


def _varint(value):
    value &= (1 << 64) - 1
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _field(number, payload):
    if isinstance(payload, int):
        return _varint(number << 3) + _varint(payload)
    payload = payload.encode() if isinstance(payload, str) else payload
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def tensor_bytes(name, dtype, dims, data, external=None):
    out = b"".join(_field(1, d) for d in dims) + _field(2, dtype) + _field(8, name)
    if external is not None:
        out += b"".join(_field(13, _field(1, k) + _field(2, str(v))) for k, v in external.items()) + _field(14, 1)
    else:
        out += _field(9, data)
    return out


def node_bytes(op, name, inputs, attributes=None):
    out = b"".join(_field(1, i) for i in inputs) + _field(2, name + "_out") + _field(3, name) + _field(4, op)
    for key, value in (attributes or {}).items():
        out += _field(5, _field(1, key) + _field(3, value) + _field(20, 2))
    return out


def onnx_bytes(initializers, nodes):
    graph = b"".join(_field(1, node_bytes(*node)) for node in nodes)
    graph += b"".join(_field(5, tensor_bytes(*tensor)) for tensor in initializers)
    return _field(1, 8) + _field(7, graph + _field(2, "g"))


def floats(*values):
    return struct.pack(f"<{len(values)}f", *values)
