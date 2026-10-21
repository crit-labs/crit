"""Strict GGUF reader: validates the header, metadata and tensor table and never decodes tensor bytes.

A GGUF file (versions 2 and 3, little-endian) is

    magic "GGUF" | version u32 | tensor count u64 | metadata count u64
    metadata     key string, value type u32, value          (strings: u64 length + UTF-8)
    tensor table name string, n_dims u32, ne u64[n_dims], ggml type u32, offset u64
    padding      to general.alignment (default 32)
    data         each tensor at its offset from the data start, padded to the alignment

`ne` lists dimensions innermost first; TensorInfo.shape is the reverse (row-major, as torch and safetensors
order it), so a projection of R rows and C columns has shape (R, C). Block types store ne[0] in blocks of
`block` elements of `type_size` bytes each. The file is mapped read-only and only the pages up to the data
start are touched, so no read reaches tensor bytes except the page the header ends in.

A model split by gguf-split names its files NAME-00001-of-00003.gguf and records split.no and
split.count; inspect_gguf_model joins the parts in order. Results are frozen dataclasses with absolute file
offsets for each tensor; type names (TensorInfo.dtype) are ggml's (F32, BF16, Q4_K, IQ3_S, ...).
"""
from dataclasses import dataclass, field
import hashlib
import math
import mmap
import os
from pathlib import Path
import re
import struct
from types import MappingProxyType

from .safetensors import Limits, _Picklable, _identity, _regular_file, _short

MAGIC = b"GGUF"
VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32
MAX_DIMS = 4          # GGML_MAX_DIMS
ARRAY_DEPTH = 2       # arrays of arrays nest at most this deep
SUFFIX = ".gguf"
SPLIT_NAME = re.compile(r"(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf")
# Multimodal projector files (mmproj files) sit beside a text model; they are another model, not a shard.
PROJECTOR = re.compile(r"(^|[-_.])mmproj([-_.]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class GgmlType:
    id: int
    name: str
    block: int       # elements per block
    type_size: int   # bytes per block

    @property
    def quantized(self):
        return self.block > 1


# ggml's storage types (ggml.h ggml_type, gguf-py GGML_QUANT_SIZES). Values are never decoded here.
GGML_TYPES = MappingProxyType({t.id: t for t in (
    GgmlType(0, "F32", 1, 4), GgmlType(1, "F16", 1, 2), GgmlType(2, "Q4_0", 32, 18), GgmlType(3, "Q4_1", 32, 20),
    GgmlType(6, "Q5_0", 32, 22), GgmlType(7, "Q5_1", 32, 24), GgmlType(8, "Q8_0", 32, 34),
    GgmlType(9, "Q8_1", 32, 36), GgmlType(10, "Q2_K", 256, 84), GgmlType(11, "Q3_K", 256, 110),
    GgmlType(12, "Q4_K", 256, 144), GgmlType(13, "Q5_K", 256, 176), GgmlType(14, "Q6_K", 256, 210),
    GgmlType(15, "Q8_K", 256, 292), GgmlType(16, "IQ2_XXS", 256, 66), GgmlType(17, "IQ2_XS", 256, 74),
    GgmlType(18, "IQ3_XXS", 256, 98), GgmlType(19, "IQ1_S", 256, 50), GgmlType(20, "IQ4_NL", 32, 18),
    GgmlType(21, "IQ3_S", 256, 110), GgmlType(22, "IQ2_S", 256, 82), GgmlType(23, "IQ4_XS", 256, 136),
    GgmlType(24, "I8", 1, 1), GgmlType(25, "I16", 1, 2), GgmlType(26, "I32", 1, 4), GgmlType(27, "I64", 1, 8),
    GgmlType(28, "F64", 1, 8), GgmlType(29, "IQ1_M", 256, 56), GgmlType(30, "BF16", 1, 2),
    GgmlType(34, "TQ1_0", 256, 54), GgmlType(35, "TQ2_0", 256, 66), GgmlType(39, "MXFP4", 32, 17),
    GgmlType(40, "NVFP4", 64, 36), GgmlType(41, "Q1_0", 128, 18), GgmlType(42, "Q2_0", 64, 18),
)})
TYPE_BY_NAME = MappingProxyType({t.name: t for t in GGML_TYPES.values()})

# Metadata value types: id -> (name, struct format or None for string/array).
_VALUE_TYPES = {0: ("UINT8", "<B"), 1: ("INT8", "<b"), 2: ("UINT16", "<H"), 3: ("INT16", "<h"),
                4: ("UINT32", "<I"), 5: ("INT32", "<i"), 6: ("FLOAT32", "<f"), 7: ("BOOL", "<B"),
                8: ("STRING", None), 9: ("ARRAY", None), 10: ("UINT64", "<Q"), 11: ("INT64", "<q"),
                12: ("FLOAT64", "<d")}


class GgufError(ValueError):
    """A GGUF file or split set that breaks the format contract or a reader limit."""


def _fail(message):
    raise GgufError(message)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str        # ggml type name
    shape: tuple      # row-major (reversed ne)
    shard: Path       # absolute path of the file as named
    data_begin: int   # absolute file byte offsets
    data_end: int

    @property
    def nbytes(self):
        return self.data_end - self.data_begin

    @property
    def ggml_type(self):
        return TYPE_BY_NAME[self.dtype]

    @property
    def ne(self):
        return tuple(reversed(self.shape))

    @property
    def elements(self):
        return math.prod(self.shape)


@dataclass(frozen=True)
class GgufFile(_Picklable):
    path: Path
    size: int
    version: int
    header_sha256: str        # SHA-256 of every byte before the data section
    file_identity: tuple
    metadata: object          # read-only {key: value}; arrays are tuples
    value_types: object       # read-only {key: type name}, ARRAY[element type] for arrays
    tensors: tuple            # tensor-table order
    alignment: int = DEFAULT_ALIGNMENT
    data_offset: int = 0

    @property
    def architecture(self):
        return self.metadata.get("general.architecture")


@dataclass(frozen=True)
class GgufModel(_Picklable):
    source: Path              # the file given, or the first part of a split set
    files: tuple              # GgufFile per part, in split order
    tensors: object           # read-only {name: TensorInfo}, sorted by name
    metadata: object          # the first part's metadata (converters write the model's keys there)

    @property
    def architecture(self):
        return self.metadata.get("general.architecture")

    @property
    def tensor_bytes(self):
        return sum(tensor.nbytes for tensor in self.tensors.values())

    def get(self, key, default=None):
        """Metadata key `key`, or with a leading "." the architecture's key (".block_count" is
        "<architecture>.block_count")."""
        if key.startswith("."):
            key = f"{self.architecture}{key}"
        return self.metadata.get(key, default)


class _Cursor:
    def __init__(self, view, path, limit):
        self.view, self.path, self.limit, self.at = view, path, limit, 0

    def take(self, count, what):
        end = self.at + count
        if end > self.limit:
            _fail(f"{self.path}: {what} at byte {self.at} runs past "
                  + (f"the {self.limit}-byte header limit" if self.limit < len(self.view) else "the end of the file"))
        start, self.at = self.at, end
        return start

    def unpack(self, fmt, what):
        start = self.take(struct.calcsize(fmt), what)
        return struct.unpack_from(fmt, self.view, start)[0]

    def string(self, what, max_bytes):
        length = self.unpack("<Q", f"{what} length")
        if length > max_bytes:
            _fail(f"{self.path}: {what} of {length} bytes exceeds the {max_bytes}-byte limit")
        start = self.take(length, what)
        try:
            return bytes(self.view[start:start + length]).decode("utf-8")
        except UnicodeDecodeError:
            _fail(f"{self.path}: {what} at byte {start} is not valid UTF-8")


def _value(cursor, kind, label, depth, limits):
    name, fmt = _VALUE_TYPES.get(kind, (None, None))
    if name is None:
        _fail(f"{cursor.path}: {label} has unknown value type {kind}")
    if fmt:
        value = cursor.unpack(fmt, label)
        if name == "BOOL":
            if value > 1:
                _fail(f"{cursor.path}: {label} is a BOOL of value {value}")
            return name, bool(value)
        return name, value
    if name == "STRING":
        return name, cursor.string(label, limits.max_header_bytes)
    if depth >= ARRAY_DEPTH:
        _fail(f"{cursor.path}: {label} nests arrays deeper than {ARRAY_DEPTH}")
    element = cursor.unpack("<I", f"{label} element type")
    count = cursor.unpack("<Q", f"{label} length")
    # Every element takes at least one byte, so a count beyond the header limit cannot be real.
    if count > cursor.limit - cursor.at:
        _fail(f"{cursor.path}: {label} declares {count} elements, more than the bytes left in the header")
    element_name, element_fmt = _VALUE_TYPES.get(element, (None, None))
    if element_name is None:
        _fail(f"{cursor.path}: {label} has unknown element type {element}")
    if element_fmt and element_name != "BOOL":
        start = cursor.take(struct.calcsize(element_fmt) * count, label)
        values = struct.unpack_from(f"<{count}{element_fmt[1]}", cursor.view, start)
        return f"ARRAY[{element_name}]", values
    values, inner = [], element_name
    for i in range(count):
        inner, item = _value(cursor, element, f"{label}[{i}]", depth + 1, limits)
        values.append(item)
    return f"ARRAY[{inner if element_name == 'ARRAY' else element_name}]", tuple(values)


def inspect_gguf(path, limits=None, *, follow_symlinks=True):
    """Validate one .gguf file from its header and tensor table; returns a GgufFile.

    Checked: magic, version 2 or 3 (a byte-swapped version is refused as big-endian), unique non-empty UTF-8 keys,
    known value types, general.alignment (a power of two), unique tensor names, 1 to 4 dimensions, known ggml types,
    ne[0] a multiple of the type's block, offsets on the alignment, and tensors that tile the data section in
    offset order with nothing between them but padding to the alignment. limits.max_header_bytes bounds the
    header (metadata and tensor table)."""
    limits = Limits() if limits is None else limits
    if not isinstance(limits, Limits):
        _fail(f"limits must be a Limits instance, not {type(limits).__name__}")
    path = Path(path)
    if path.suffix != SUFFIX:
        _fail(f"{path}: expected a .gguf file")
    with _regular_file(path, follow_symlinks) as (fd, info):
        size = info.st_size
        if size > limits.max_file_bytes:
            _fail(f"{path}: {size} bytes exceed the {limits.max_file_bytes}-byte file limit")
        if size < 24:
            _fail(f"{path}: truncated header")
        with mmap.mmap(fd, 0, access=mmap.ACCESS_READ) as mapped:
            view = memoryview(mapped)
            try:
                return _parse(path, view, size, info, limits)
            finally:
                view.release()


def _parse(path, view, size, info, limits):
    cursor = _Cursor(view, path, min(size, limits.max_header_bytes))
    if bytes(view[:4]) != MAGIC:
        _fail(f"{path}: not a GGUF file (magic {bytes(view[:4])!r})")
    cursor.at = 4
    version = cursor.unpack("<I", "version")
    if version not in VERSIONS:
        if int.from_bytes(struct.pack("<I", version), "big") in VERSIONS:
            _fail(f"{path}: big-endian GGUF is not supported")
        _fail(f"{path}: GGUF version {version} is not supported (expected 2 or 3)")
    tensor_count = cursor.unpack("<Q", "tensor count")
    if tensor_count > limits.max_tensors:
        _fail(f"{path}: {tensor_count} tensors exceed the {limits.max_tensors}-tensor limit")
    kv_count = cursor.unpack("<Q", "metadata count")
    if kv_count > cursor.limit:
        _fail(f"{path}: metadata count {kv_count} exceeds the header")
    metadata, value_types = {}, {}
    for i in range(kv_count):
        key = cursor.string(f"metadata key {i}", limits.max_name_bytes)
        if not key:
            _fail(f"{path}: metadata key {i} is empty")
        if key in metadata:
            _fail(f"{path}: metadata key {key!r} appears twice")
        kind = cursor.unpack("<I", f"metadata {key!r} type")
        value_types[key], metadata[key] = _value(cursor, kind, f"metadata {key!r}", 0, limits)
    alignment = DEFAULT_ALIGNMENT
    if "general.alignment" in metadata:
        alignment = metadata["general.alignment"]
        if value_types["general.alignment"] != "UINT32" or alignment <= 0 or alignment & (alignment - 1):
            _fail(f"{path}: general.alignment must be a UINT32 power of two, not {_short(alignment)}")
    tensors, names = [], set()
    for i in range(tensor_count):
        name = cursor.string(f"tensor {i} name", limits.max_name_bytes)
        if not name:
            _fail(f"{path}: tensor {i} has an empty name")
        if name in names:
            _fail(f"{path}: tensor {name!r} appears twice")
        names.add(name)
        label = f"{path}: tensor {name!r}"
        dims = cursor.unpack("<I", f"tensor {name!r} rank")
        if not 1 <= dims <= MAX_DIMS:
            _fail(f"{label}: rank {dims} is not in [1, {MAX_DIMS}]")
        ne = tuple(cursor.unpack("<Q", f"tensor {name!r} dimension") for _ in range(dims))
        if any(not 1 <= n <= limits.max_dimension for n in ne):
            _fail(f"{label}: dimensions {list(ne)} are not all in [1, {limits.max_dimension}]")
        kind = cursor.unpack("<I", f"tensor {name!r} type")
        ggml = GGML_TYPES.get(kind)
        if ggml is None:
            _fail(f"{label}: unknown ggml type {kind}")
        if ne[0] % ggml.block:
            _fail(f"{label}: ne[0] {ne[0]} is not a multiple of {ggml.name}'s {ggml.block}-element block")
        nbytes = ne[0] // ggml.block * ggml.type_size * math.prod(ne[1:])
        if nbytes > limits.max_file_bytes:
            _fail(f"{label}: {list(ne)} of {ggml.name} exceeds the {limits.max_file_bytes}-byte file limit")
        offset = cursor.unpack("<Q", f"tensor {name!r} offset")
        if offset % alignment:
            _fail(f"{label}: offset {offset} is not a multiple of the {alignment}-byte alignment")
        tensors.append((name, ggml.name, tuple(reversed(ne)), offset, nbytes))
    data_offset = -(-cursor.at // alignment) * alignment
    if not tensors and cursor.at <= size < data_offset:
        data_offset = size   # a vocabulary-only file (a converter's --vocab-only) ends with its header, unpadded
    if data_offset > size:
        _fail(f"{path}: the data section would start at byte {data_offset}, past the {size}-byte file")
    infos, end = [], 0
    for name, dtype, shape, offset, nbytes in sorted(tensors, key=lambda t: (t[3], t[0])):
        padded = -(-end // alignment) * alignment
        if offset != padded:
            kind = "hole" if offset > padded else "overlap"
            _fail(f"{path}: {kind} at data offset {padded} before tensor {name!r}")
        end = offset + nbytes
        if data_offset + end > size:
            _fail(f"{path}: tensor {name!r} ends at byte {data_offset + end}, past the {size}-byte file")
    padded_end = data_offset + -(-end // alignment) * alignment
    if size > max(padded_end, data_offset + end):
        _fail(f"{path}: {size - max(padded_end, data_offset + end)} trailing bytes after the last tensor")
    for name, dtype, shape, offset, nbytes in tensors:
        infos.append(TensorInfo(name, dtype, shape, path.absolute(), data_offset + offset,
                                data_offset + offset + nbytes))
    digest = hashlib.sha256(view[:data_offset]).hexdigest()
    return GgufFile(path.absolute(), size, version, digest, _identity(info), MappingProxyType(metadata),
                    MappingProxyType(value_types), tuple(infos), alignment, data_offset)


def _gguf_files(directory):
    return sorted(path for path in directory.iterdir()
                  if path.suffix == SUFFIX and not PROJECTOR.search(path.stem) and (path.is_file() or path.is_symlink()))


def find_gguf(directory):
    """The model's GGUF in DIRECTORY: its only .gguf file, or the first part of its only split set (mmproj
    projector files are left out), or None without any. Several models in one folder are an error."""
    directory = Path(directory)
    candidates = _gguf_files(directory)
    if not candidates:
        return None
    stems = {}
    for path in candidates:
        match = SPLIT_NAME.fullmatch(path.name)
        stems.setdefault(match["stem"] if match else path.stem, []).append(path)
    if len(stems) > 1:
        names = ", ".join(sorted(path.name for path in candidates)[:8])
        _fail(f"{directory}: {len(candidates)} .gguf files of {len(stems)} models; keep one model per folder: {names}")
    return sorted(next(iter(stems.values())))[0]


def inspect_gguf_model(path, limits=None, *, follow_symlinks=True):
    """Inspect a .gguf file, the first part of a split set, or a model directory (find_gguf); returns a GgufModel.
    A split set needs every part, each numbered by split.no/split.count as its name says, and no tensor twice."""
    limits = Limits() if limits is None else limits
    path = Path(path)
    if path.is_dir():
        found = find_gguf(path)
        if found is None:
            _fail(f"{path}: no .gguf file")
        path = found
    first = inspect_gguf(path, limits, follow_symlinks=follow_symlinks)
    match = SPLIT_NAME.fullmatch(path.name)
    count = first.metadata.get("split.count")
    if count is None:
        if match and int(match["count"]) != 1:
            _fail(f"{path}: named as part of a {int(match['count'])}-part split but has no split.count")
        files = [first]
    else:
        if not match:
            _fail(f"{path}: split.count {count} but the name is not NAME-00001-of-{count:05d}.gguf")
        if first.metadata.get("split.no") != 0 or int(match["no"]) != 1 or int(match["count"]) != count:
            _fail(f"{path}: not the first part of a {count}-part split")
        if count > limits.max_shards:
            _fail(f"{path}: {count} parts exceed the {limits.max_shards}-shard limit")
        files = [first]
        for no in range(1, count):
            part = path.with_name(f"{match['stem']}-{no + 1:05d}-of-{count:05d}{SUFFIX}")
            if not (part.is_file() or part.is_symlink()):
                _fail(f"{path}: split part {part.name} is missing")
            shard = inspect_gguf(part, limits, follow_symlinks=follow_symlinks)
            if shard.metadata.get("split.no") != no or shard.metadata.get("split.count") != count:
                _fail(f"{part}: split.no/split.count do not say part {no + 1} of {count}")
            files.append(shard)
    tensors, total = {}, 0
    for shard in files:
        total += shard.size
        if total > limits.max_total_bytes:
            _fail(f"{path}: model files exceed the {limits.max_total_bytes}-byte limit")
        for tensor in shard.tensors:
            if tensor.name in tensors:
                _fail(f"{path}: tensor {tensor.name!r} appears in {tensors[tensor.name].shard.name!r} and "
                      f"{shard.path.name!r}")
            tensors[tensor.name] = tensor
    declared = first.metadata.get("split.tensors.count")
    if declared is not None and declared != len(tensors):
        _fail(f"{path}: split.tensors.count {declared} but the parts hold {len(tensors)} tensors")
    ordered = MappingProxyType({name: tensors[name] for name in sorted(tensors)})
    return GgufModel(path.absolute(), tuple(files), ordered, first.metadata)


# A GGUF folder that also holds the base model's config.json or tokenizer files ----------------------------------

@dataclass(frozen=True)
class GgufCheck:
    errors: tuple   # disagreements that change the model's arithmetic or its tokens
    notes: tuple    # differences that do not (a declared context length, the chat template's source)

    def raise_errors(self, directory):
        if self.errors:
            _fail(f"{directory}: the GGUF and the folder's files disagree: " + "; ".join(self.errors))


def _array_or(value):
    return tuple(value) if isinstance(value, tuple) else (value,)


def _same_float(a, b):
    # GGUF stores FLOAT32; config.json the decimal that rounds to it.
    return struct.unpack("<f", struct.pack("<f", float(a)))[0] == struct.unpack("<f", struct.pack("<f", float(b)))[0]


def _vocab(tokenizer):
    """id -> token string of a tokenizer.json (model vocab plus added tokens)."""
    model = tokenizer.get("model") or {}
    vocab = model.get("vocab")
    tokens = {}
    if isinstance(vocab, dict):
        tokens.update({int(i): t for t, i in vocab.items()})
    elif isinstance(vocab, list):   # Unigram: [[piece, score], ...]
        tokens.update({i: entry[0] for i, entry in enumerate(vocab)})
    for added in tokenizer.get("added_tokens") or ():
        tokens[int(added["id"])] = added["content"]
    return tokens


def _merges(tokenizer):
    merges = (tokenizer.get("model") or {}).get("merges")
    if merges is None:
        return None
    return tuple(" ".join(m) if isinstance(m, list) else m for m in merges)


def _folder_template(directory):
    path = directory / "chat_template.jinja"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    path = directory / "tokenizer_config.json"
    if path.is_file():
        import json
        value = json.loads(path.read_text(encoding="utf-8")).get("chat_template")
        if isinstance(value, list):   # named templates: the default one
            value = next((t.get("template") for t in value if t.get("name") == "default"), None)
        return value if isinstance(value, str) else None
    return None


def check_gguf_folder(model, config, directory):
    """Compare a GgufModel with the config.json (crit_core.model.ModelConfig) and tokenizer.json beside it, when the
    folder holds them (otherwise crit_core.model.gguf_files derives them from the GGUF).

    The folder's files describe the model to everything that is not the weights (Eval's context and ceiling rules,
    the reference's modeling code, the tokenizer and chat template), so they must be the base model's: the shapes
    the GGUF declares (layers, width, heads, head size, MLP width, experts, vocabulary), its norm epsilon and RoPE
    base, and every token string and merge must agree. A different declared context length and a chat template
    that differs from the GGUF's tokenizer.chat_template are notes: the folder's files are used."""
    import json
    directory = Path(directory)
    errors, notes = [], []
    get = model.get

    def compare(label, gguf_value, config_value):
        if gguf_value is None or config_value is None:
            return
        values = {v for v in _array_or(gguf_value) if v}   # 0 marks a layer without the part (recurrent layers)
        if values and values != {config_value}:
            errors.append(f"{label}: GGUF {sorted(values) if len(values) > 1 else next(iter(values))}, "
                          f"config.json {config_value}")

    drafts = get(".nextn_predict_layers") or 0
    layers = get(".block_count")
    if layers is not None and config.num_hidden_layers is not None and layers - drafts != config.num_hidden_layers:
        errors.append(f"layers: GGUF {layers - drafts}" + (f" (+{drafts} draft)" if drafts else "")
                      + f", config.json {config.num_hidden_layers}")
    compare("hidden size", get(".embedding_length"), config.hidden_size)
    compare("attention heads", get(".attention.head_count"), config.num_attention_heads)
    compare("key/value heads", get(".attention.head_count_kv"), config.num_key_value_heads)
    compare("head size", get(".attention.key_length"), config.head_dim)
    if config.moe is None:
        compare("MLP width", get(".feed_forward_length"), config.intermediate_size)
    else:
        compare("experts", get(".expert_count"), config.moe.experts)
        compare("experts per token", get(".expert_used_count"), config.moe.experts_per_token)
        compare("expert MLP width", get(".expert_feed_forward_length"), config.moe.moe_intermediate_size)
    for label, key, value in (("RMS norm epsilon", ".attention.layer_norm_rms_epsilon", config.rms_norm_eps),
                              ("RoPE base", ".rope.freq_base", config.rope.theta)):
        found = get(key)
        if found is not None and value is not None and not _same_float(found, value):
            errors.append(f"{label}: GGUF {found}, config.json {value}")
    embedding = model.tensors.get("token_embd.weight")
    if embedding is not None and config.vocab_size is not None and embedding.shape[0] != config.vocab_size:
        errors.append(f"vocabulary rows: GGUF token_embd {embedding.shape[0]}, config.json {config.vocab_size}")
    limit, _ = config.position_limit()
    declared = get(".context_length")
    if declared is not None and limit is not None and declared != limit:
        notes.append(f"the GGUF declares {declared} positions, config.json {limit}; Eval uses config.json's")

    tokens = get("tokenizer.ggml.tokens")
    path = directory / "tokenizer.json"
    if tokens is not None and path.is_file():
        tokenizer = json.loads(path.read_text(encoding="utf-8"))
        vocab = _vocab(tokenizer)
        for i, token in enumerate(tokens):
            expected = vocab.get(i)
            # Converters pad a vocabulary shorter than the embedding with [PAD<i>] tokens.
            if expected is None and token.startswith("[PAD"):
                continue
            # The converter writes user-defined added tokens with SentencePiece's "\u2581" as a space.
            if expected != token and (expected is None or expected.replace("\u2581", " ") != token):
                errors.append(f"token {i}: GGUF {_short(token)}, tokenizer.json {_short(expected)}")
                break
        extra = sorted(i for i in vocab if i >= len(tokens))
        if extra:
            errors.append(f"tokenizer.json has {len(extra)} tokens past the GGUF's {len(tokens)}, from id {extra[0]}")
        merges, found = _merges(tokenizer), get("tokenizer.ggml.merges")
        if merges is not None and found is not None and tuple(found) != merges:
            first = next((i for i, (a, b) in enumerate(zip(found, merges)) if a != b), min(len(found), len(merges)))
            errors.append(f"BPE merges differ from merge {first} (GGUF {len(found)}, tokenizer.json {len(merges)})")
    template, embedded = _folder_template(directory), get("tokenizer.chat_template")
    if embedded and template is None:
        notes.append("the folder has no chat template; the GGUF's tokenizer.chat_template is not used")
    elif embedded and template != embedded:
        notes.append("the folder's chat template differs from the GGUF's tokenizer.chat_template; "
                     "the folder's is used")
    return GgufCheck(tuple(errors), tuple(notes))
