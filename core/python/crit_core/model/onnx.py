"""Strict reader for ONNX models (model.onnx, with its external data files), standard library only: the graph's
protobuf is walked without the onnx package and tensor bytes are never read.

An ONNX model is a graph whose weights are initializers. Exporters keep a module parameter's name when the graph
uses it as stored (embeddings, norms, biases), but the weight of a linear layer usually reaches its MatMul as a
transposed copy under a generated name (onnx::MatMul_123); the module it belongs to survives in the consuming
node's name (/model/layers.0/self_attn/q_proj/MatMul). This reader names every weight the way the Hugging Face
checkpoint the graph was exported from does, so Eval's ceiling and reference read it like that checkpoint:

    named initializer                       its own name, as stored
    generated MatMul B (onnx::*)            <module>.weight, stored [in, out] (OnnxTensor.transposed)
    Gemm B                                  <module>.weight, transposed unless transB
    MatMulNBits B, scales, zero_points,     <module>.nbits_weight, .nbits_scales, .nbits_zero_points, .nbits_g_idx,
    g_idx                                   with the node's K, N, bits and block_size (OnnxTensor.attributes)

where <module> is the node's name without its leading "/" and its last part, "/" read as "." (the scopes
torch.onnx gives nodes, as Optimum exports them). Other generated initializers (shape constants) are not weights.
Graphs from onnxruntime-genai's model builder (genai_config.json beside them) fuse and rename layers and are
refused, as are weights of quantized ops other than MatMulNBits (UNREAD_OPS). Initializers may hold their bytes
inline (raw_data) or in external data files beside the model (location,
offset, length); both are bounds-checked. Weights stored as varint fields, sparse initializers and 4-bit element
types are refused.
"""
from dataclasses import dataclass, field
import mmap
import os
from pathlib import Path
from types import MappingProxyType

from .safetensors import DTYPE_BITS, Limits, TensorInfo, _identity

SUFFIX = ".onnx"
# TensorProto.DataType -> safetensors dtype names
DTYPES = {1: "F32", 2: "U8", 3: "I8", 4: "U16", 5: "I16", 6: "I32", 7: "I64", 9: "BOOL", 10: "F16", 11: "F64",
          12: "U32", 13: "U64", 16: "BF16", 17: "F8_E4M3", 19: "F8_E5M2"}
NBITS = ("nbits_weight", "nbits_scales", "nbits_zero_points", "nbits_g_idx")   # MatMulNBits inputs 1..4
# Ops whose weight operands are quantized in ways this reader does not name (yet): refused, never misread.
UNREAD_OPS = ("MatMulInteger", "MatMulBnb4", "QLinearMatMul", "DynamicQuantizeMatMul", "GatherBlockQuantized",
              "DequantizeLinear", "MatMulIntegerToFloat")
GENAI_CONFIG = "genai_config.json"
MAX_GRAPH_BYTES = 1 << 31   # protobuf's own 2 GiB limit


class OnnxError(ValueError):
    """An ONNX file crit cannot read: a broken protobuf, weights outside their files, or a layout it refuses."""


def _fail(message):
    raise OnnxError(message)


@dataclass(frozen=True)
class OnnxTensor(TensorInfo):
    source_name: str = ""       # the initializer's own name in the graph
    transposed: bool = False    # stored [in, out]: the module's weight is its transpose
    attributes: object = field(default_factory=lambda: MappingProxyType({}))   # MatMulNBits K, N, bits, block_size


@dataclass(frozen=True)
class OnnxModel:
    path: Path                  # the .onnx file
    size: int                   # bytes of the .onnx file and its external data files
    file_identity: tuple
    tensors: tuple              # OnnxTensor in graph order, Hugging Face names
    data_files: tuple = ()      # external data files the weights live in, by name

    @property
    def shards(self):
        return (self,)

    @property
    def tensor_bytes(self):
        return sum(t.nbytes for t in self.tensors)


class _Reader:
    """Protobuf wire format over a memory-mapped file, within [start, end)."""

    def __init__(self, data, start, end):
        self.data, self.at, self.end = data, start, end

    def varint(self):
        result, shift = 0, 0
        while True:
            if self.at >= self.end:
                _fail("truncated protobuf")
            byte = self.data[self.at]
            self.at += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7
            if shift > 63:
                _fail("varint longer than 64 bits")

    def fixed(self, size):
        if self.at + size > self.end:
            _fail("truncated protobuf")
        value = int.from_bytes(self.data[self.at:self.at + size], "little")
        self.at += size
        return value

    def fields(self):
        """(field number, wire type, value): varints and fixed values as integers, length-delimited fields as
        (start, length) without reading them."""
        while self.at < self.end:
            key = self.varint()
            number, wire = key >> 3, key & 7
            if wire == 0:
                yield number, wire, self.varint()
            elif wire == 1:
                yield number, wire, self.fixed(8)
            elif wire == 5:
                yield number, wire, self.fixed(4)
            elif wire == 2:
                length = self.varint()
                if self.at + length > self.end:
                    _fail("a length-delimited field runs past its message")
                yield number, wire, (self.at, length)
                self.at += length
            else:
                _fail(f"unsupported protobuf wire type {wire}")

    def bytes_at(self, span, limit=1 << 20):
        start, length = span
        if length > limit:
            _fail(f"a string field of {length} bytes")
        return bytes(self.data[start:start + length])

    def sub(self, span):
        return _Reader(self.data, span[0], span[0] + span[1])


def _text(reader, span):
    try:
        return reader.bytes_at(span).decode("utf-8")
    except UnicodeDecodeError:
        _fail("a name that is not UTF-8")


def _signed(value):
    return value - (1 << 64) if value >= 1 << 63 else value


def _packed_varints(reader, span):
    sub, values = reader.sub(span), []
    while sub.at < sub.end:
        values.append(_signed(sub.varint()))
    return values


def _tensor(reader, span):
    """A TensorProto: name, dims, dtype and where its bytes are."""
    found = {"dims": [], "external": {}, "location": 0, "name": "", "dtype": 0, "raw": None, "floats": None,
             "varints": False}
    for number, wire, value in reader.sub(span).fields():
        if number == 1:
            found["dims"] += _packed_varints(reader, value) if wire == 2 else [_signed(value)]
        elif number == 2:
            found["dtype"] = value
        elif number == 8:
            found["name"] = _text(reader, value)
        elif number == 9:
            found["raw"] = value
        elif number == 4 and wire == 2:   # packed float_data: contiguous little-endian F32
            found["floats"] = value
        elif number in (5, 7, 11) or (number == 4 and wire != 2):
            found["varints"] = True
        elif number == 13:
            key = val = ""
            for inner, _, item in reader.sub(value).fields():
                if inner == 1:
                    key = _text(reader, item)
                elif inner == 2:
                    val = _text(reader, item)
            found["external"][key] = val
        elif number == 14:
            found["location"] = value
        elif number == 3:
            _fail(f"initializer {found['name'] or '?'} is segmented")
    return found


def _node(reader, span):
    node = {"inputs": [], "name": "", "op": "", "attributes": {}}
    for number, wire, value in reader.sub(span).fields():
        if number == 1:
            node["inputs"].append(_text(reader, value))
        elif number == 3:
            node["name"] = _text(reader, value)
        elif number == 4:
            node["op"] = _text(reader, value)
        elif number == 5:
            name, integer = "", None
            for inner, _, item in reader.sub(value).fields():
                if inner == 1:
                    name = _text(reader, item)
                elif inner == 3:
                    integer = _signed(item)
            if integer is not None:
                node["attributes"][name] = integer
    return node


def _graph(reader):
    """The model's (initializers, nodes)."""
    graph = None
    for number, _, value in reader.fields():
        if number == 7:
            graph = value
    if graph is None:
        _fail("no graph")
    initializers, nodes = [], []
    for number, _, value in reader.sub(graph).fields():
        if number == 5:
            initializers.append(_tensor(reader, value))
        elif number == 1:
            nodes.append(_node(reader, value))
        elif number == 15:
            _fail("sparse initializers are not read")
    return initializers, nodes


def _module(node_name):
    parts = node_name.strip("/").split("/")
    return ".".join(parts[:-1]) if len(parts) > 1 else None


def _names(initializers, nodes, path):
    """{initializer name: (Hugging Face name, transposed, attributes)} for the weights (see the module docstring)."""
    stored = {t["name"] for t in initializers}
    mapped = {}

    def claim(initializer, name, transposed=False, attributes=None):
        if initializer not in stored:
            return
        if initializer in mapped and mapped[initializer][0] != name:
            _fail(f"{path}: initializer {initializer!r} feeds both {mapped[initializer][0]} and {name}")
        mapped[initializer] = (name, transposed, MappingProxyType(dict(attributes or {})))

    for node in nodes:
        module = _module(node["name"])
        inputs = node["inputs"]
        if node["op"] in UNREAD_OPS and any(name in stored for name in inputs):
            _fail(f"{path}: {node['op']} ({node['name']}) holds weights crit does not read; crit reads float "
                  "weights and MatMulNBits")
        if node["op"] == "MatMulNBits" and module:
            for index, suffix in enumerate(NBITS, start=1):
                if index < len(inputs) and inputs[index]:
                    claim(inputs[index], f"{module}.{suffix}", attributes=node["attributes"])
        elif node["op"] in ("MatMul", "Gemm") and module and len(inputs) > 1 and inputs[1].startswith("onnx::"):
            transposed = node["op"] == "MatMul" or not node["attributes"].get("transB", 0)
            claim(inputs[1], f"{module}.weight", transposed)
    for tensor in initializers:
        if tensor["name"] not in mapped and not tensor["name"].startswith("onnx::"):
            mapped[tensor["name"]] = (tensor["name"], False, MappingProxyType({}))
    targets = {}
    for initializer, (name, _, _) in mapped.items():
        if name in targets:
            _fail(f"{path}: {targets[name]!r} and {initializer!r} both name {name}")
        targets[name] = initializer
    return mapped


def inspect_onnx(path, limits=None):
    """Validate one .onnx file and its external data; returns an OnnxModel."""
    limits = Limits() if limits is None else limits
    path = Path(path)
    info = os.stat(path)
    if info.st_size > MAX_GRAPH_BYTES:
        _fail(f"{path}: {info.st_size} bytes, more than protobuf's 2 GiB")
    sizes = {path.name: info.st_size}
    if info.st_size == 0:
        _fail(f"{path}: empty file")
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        try:
            initializers, nodes = _graph(_Reader(data, 0, info.st_size))
        except OnnxError as error:
            _fail(f"{path}: {error}")
    names = _names(initializers, nodes, path)
    if len(names) > limits.max_tensors:
        _fail(f"{path}: {len(names)} weights, more than {limits.max_tensors}")
    tensors = []
    for tensor in initializers:
        if tensor["name"] not in names:
            continue
        name, transposed, attributes = names[tensor["name"]]
        dtype = DTYPES.get(tensor["dtype"])
        if dtype is None or dtype not in DTYPE_BITS:
            _fail(f"{path}: {tensor['name']!r} has ONNX element type {tensor['dtype']}, which crit does not read")
        shape = tuple(tensor["dims"])
        if any(n < 0 for n in shape) or len(shape) > limits.max_rank:
            _fail(f"{path}: {tensor['name']!r} has shape {list(shape)}")
        nbytes = DTYPE_BITS[dtype] // 8
        for n in shape:
            nbytes *= n
        if tensor["location"] == 1:
            external = tensor["external"]
            location = external.get("location", "")
            if not location or Path(location).name != location or location.startswith("."):
                _fail(f"{path}: {tensor['name']!r} keeps its data in {location!r}, not a file beside the model")
            data = path.parent / location
            if location not in sizes:
                if not data.is_file():
                    _fail(f"{path}: external data file {location} is missing")
                sizes[location] = data.stat().st_size
            begin = int(external.get("offset", 0) or 0)
            length = int(external.get("length", nbytes) or nbytes)
            if length != nbytes or begin < 0 or begin + length > sizes[location]:
                _fail(f"{path}: {tensor['name']!r} needs bytes [{begin}, {begin + length}) of {location}, which "
                      f"has {sizes[location]}; its shape needs {nbytes}")
            shard = data.absolute()
        elif tensor["raw"] is not None or tensor["floats"] is not None:
            begin, length = tensor["raw"] if tensor["raw"] is not None else tensor["floats"]
            if tensor["raw"] is None and dtype != "F32":
                _fail(f"{path}: {tensor['name']!r} stores {dtype} values as float_data")
            if length != nbytes:
                _fail(f"{path}: {tensor['name']!r} holds {length} bytes; its shape needs {nbytes}")
            shard = path.absolute()
        elif tensor["varints"]:
            _fail(f"{path}: weight {tensor['name']!r} is stored as varint fields, which crit does not read")
        else:
            _fail(f"{path}: {tensor['name']!r} has no data")
        tensors.append(OnnxTensor(name, dtype, shape, shard, begin, begin + nbytes, tensor["name"], transposed,
                                  attributes))
    data_files = tuple(sorted(name for name in sizes if name != path.name))
    return OnnxModel(path.absolute(), sum(sizes.values()), _identity(info), tuple(tensors), data_files)


def onnx_names(folder):
    return sorted(p.name for p in Path(folder).iterdir() if p.name.endswith(SUFFIX) and (p.is_file() or p.is_symlink()))


def find_onnx(path):
    """The .onnx file of a folder (its only one) or the file itself."""
    path = Path(path)
    if not path.is_dir():
        found = path
    else:
        names = onnx_names(path)
        if len(names) != 1:
            _fail(f"{path}: {len(names)} .onnx files; name the one to read: {', '.join(names[:8])}")
        found = path / names[0]
    if (found.parent / GENAI_CONFIG).exists():
        _fail(f"{found}: built by onnxruntime-genai's model builder ({GENAI_CONFIG}), whose graph fuses and renames "
              "the model's layers; crit reads ONNX exported from a Hugging Face checkpoint (Optimum, torch.onnx)")
    return found


def description_folder(path):
    """Where an ONNX model's description files are: its folder, or the repository above an onnx/ folder (the
    layout of exported repositories: config.json at the root, onnx/model.onnx beside its variants)."""
    folder = Path(path) if Path(path).is_dir() else Path(path).parent
    if folder.name == "onnx" and (folder.parent / "config.json").is_file() and not (folder / "config.json").is_file():
        return folder.parent
    return folder
