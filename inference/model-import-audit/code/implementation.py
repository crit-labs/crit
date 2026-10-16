"""Bounded Safetensors metadata audits; no tensor values or model execution.

Headers and indexes are read by Core's strict reader (crit_core.model.safetensors) under the audit policy: the
strict index policy, no symbolic links, tighter default limits and byte-addressable storage dtypes only. This module
adds the engine's role, capability and packed-geometry checks. Architecture roles, codec identities and actual
inference remain engine responsibilities.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Callable, Mapping

try:
    from crit_core.model import safetensors as _safetensors
except ImportError:   # a checkout without the installed crit distribution
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "core" / "python"))
    from crit_core.model import safetensors as _safetensors


class AuditError(ValueError):
    """The supported metadata contract was not established."""


# Deliberately byte-addressable subset. This table describes storage size only;
# FP8 numerical decoding and all sub-byte formats remain outside this audit.
DTYPE_BYTES = MappingProxyType({
    "BOOL": 1, "U8": 1, "U16": 2, "U32": 4, "U64": 8,
    "I8": 1, "I16": 2, "I32": 4, "I64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
})

# Core's results: TensorInfo(name, dtype, shape, shard, data_begin, data_end) with absolute offsets,
# ShardInfo(path, size, header_sha256, file_identity, tensors, ...) and Checkpoint(source, tensors, shards,
# index_sha256, ...), whose read-only tensors mapping is sorted by name.
TensorInfo = _safetensors.TensorInfo
ShardInfo = _safetensors.ShardInfo
ModelInventory = _safetensors.Checkpoint


@contextmanager
def _audit_errors():
    try:
        yield
    except _safetensors.SafetensorsError as exc:
        raise AuditError(str(exc)) from exc


@dataclass(frozen=True)
class Limits(_safetensors.Limits):
    """Core's reader limits with the audit's tighter defaults."""
    max_header_bytes: int = 1 << 20
    max_index_bytes: int = 1 << 20
    max_tensors: int = 8192
    max_shards: int = 32
    max_rank: int = 8
    max_dimension: int = (1 << 31) - 1
    max_file_bytes: int = 64 << 30
    max_total_bytes: int = 512 << 30
    max_name_bytes: int = 1024

    def __post_init__(self):
        with _audit_errors():
            super().__post_init__()


def _integer(value, label, maximum):
    if type(value) is not int or value < 0 or value > maximum:
        raise AuditError(f"Invalid {label}")
    return value


def _name(value, label, limits):
    if not isinstance(value, str) or not value:
        raise AuditError(f"Invalid {label}")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AuditError(f"Invalid UTF-8 {label}") from exc
    if len(encoded) > limits.max_name_bytes or "\x00" in value:
        raise AuditError(f"Invalid {label} length or NUL")
    return value


def _dtype(value):
    if not isinstance(value, str) or value not in DTYPE_BYTES:
        raise AuditError(f"Unsupported storage dtype: {value!r}")
    return value


def _shape(value, dtype, limits):
    if not isinstance(value, (tuple, list)) or len(value) > limits.max_rank:
        raise AuditError("Invalid tensor rank")
    dims = tuple(_integer(v, "dimension", limits.max_dimension) for v in value)
    elements = 0 if 0 in dims else 1
    maximum = limits.max_file_bytes // DTYPE_BYTES[dtype]
    for dim in dims:
        if elements and dim > maximum // elements:
            raise AuditError("Tensor byte extent exceeds limit")
        elements *= dim
    return dims, elements * DTYPE_BYTES[dtype]


def _byte_addressable(tensors):
    for tensor in tensors:
        _dtype(tensor.dtype)


def inspect_safetensors(path, limits=Limits()):
    """Read only the eight-byte prefix and a bounded UTF-8 header (Core's reader, no symbolic links)."""
    with _audit_errors():
        shard = _safetensors.inspect_safetensors(path, limits, follow_symlinks=False)
    _byte_addressable(shard.tensors)
    return shard


def inspect_model(path, limits=Limits()):
    """Inspect one container or a strict flat-shard .safetensors.index.json (Core's reader, strict policy)."""
    path = Path(path)
    if path.suffix != ".safetensors" and not path.name.endswith(_safetensors.INDEX_SUFFIX):
        raise AuditError("Unsupported model container/index format")
    with _audit_errors():
        inventory = _safetensors.inspect_model(path, limits, index_policy=_safetensors.IndexPolicy.STRICT,
                                               follow_symlinks=False)
    _byte_addressable(inventory.tensors.values())
    return inventory


@dataclass(frozen=True)
class TensorRole:
    role: str
    key: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class CapabilityRequirement:
    name: str
    roles: tuple[str, ...]
    requested: bool = True


@dataclass(frozen=True)
class PackedMatrix:
    """Geometry of already-repacked row-major K words and local group scales.

    `encoding` names the engine's exact codec; values/codebook/scales are NOT read.
    Interior K slices must align to both group and packing-word boundaries.
    Only the final slice can have a partial group/word.
    """
    encoding: str
    tensor_prefix: str
    data_role: str
    scale_role: str
    rows: int
    full_k: int
    group_size: int
    bits: int = 4
    k_begin: int = 0
    k_end: int | None = None


@dataclass(frozen=True)
class ModelSchema:
    roles: tuple[TensorRole, ...]
    capabilities: tuple[CapabilityRequirement, ...] = ()
    packed_matrices: tuple[PackedMatrix, ...] = ()
    reject_unexpected: bool = False


@dataclass(frozen=True)
class AuditResult:
    inventory: ModelInventory
    bindings: Mapping[str, TensorInfo]
    capability_status: Mapping[str, str]  # metadata_ready or not_requested
    codecs: tuple[PackedMatrix, ...]


def _packed(matrix, bindings, limits):
    _name(matrix.encoding, "codec identity", limits)
    prefix = _name(matrix.tensor_prefix, "matrix tensor prefix", limits)
    _name(matrix.data_role, "packed data role", limits)
    _name(matrix.scale_role, "packed scale role", limits)
    if matrix.data_role == matrix.scale_role:
        raise AuditError("Packed data and scale roles must differ")
    try:
        data, scale = bindings[matrix.data_role], bindings[matrix.scale_role]
    except KeyError as exc:
        raise AuditError("Packed matrix refers to an unknown tensor role") from exc
    if not all(t.name.startswith(prefix + ".") for t in (data, scale)):
        raise AuditError("Packed data/scales disagree with declared matrix namespace")
    rows = _integer(matrix.rows, "packed rows", limits.max_dimension)
    full = _integer(matrix.full_k, "packed full K", limits.max_dimension)
    group = _integer(matrix.group_size, "group size", limits.max_dimension)
    begin = _integer(matrix.k_begin, "K begin", full)
    end = full if matrix.k_end is None else _integer(matrix.k_end, "K end", full)
    if not rows or not full or not group or begin >= end or type(matrix.bits) is not int or matrix.bits not in (4, 8):
        raise AuditError("Invalid packed matrix geometry")
    if data.dtype not in ("U8", "I32") or scale.dtype not in ("F16", "BF16", "F32", "F64"):
        raise AuditError("Unsupported packed word or scale dtype")
    per_word = DTYPE_BYTES[data.dtype] * 8 // matrix.bits
    if begin % group or begin % per_word or (end != full and (end % group or end % per_word)):
        raise AuditError("K slice crosses a group or packed-word boundary")
    local = end - begin
    expected_data = (rows, (local + per_word - 1) // per_word)
    expected_scales = (rows, (local + group - 1) // group)
    if data.shape != expected_data or scale.shape != expected_scales:
        raise AuditError("Packed local data/scale geometry mismatch (unsharded scales are invalid)")


def audit_model(inventory, schema, limits=Limits()):
    """Bind exact engine roles to actual headers; never certify inference."""
    if not isinstance(inventory, ModelInventory) or not isinstance(schema, ModelSchema):
        raise AuditError("Expected inspected ModelInventory and ModelSchema")
    if any(not isinstance(values, (tuple, list)) for values in
           (schema.roles, schema.capabilities, schema.packed_matrices)):
        raise AuditError("Schema collections must be tuples/lists")
    if type(schema.reject_unexpected) is not bool or len(schema.roles) > limits.max_tensors:
        raise AuditError("Invalid schema bounds/options")
    if len(schema.capabilities) > limits.max_tensors or len(schema.packed_matrices) > limits.max_tensors:
        raise AuditError("Schema collection exceeds limit")
    bindings = {}
    for requirement in schema.roles:
        if not isinstance(requirement, TensorRole):
            raise AuditError("Invalid tensor role declaration")
        role = _name(requirement.role, "role name", limits)
        key = _name(requirement.key, "required tensor key", limits)
        dtype = _dtype(requirement.dtype)
        shape, _ = _shape(requirement.shape, dtype, limits)
        if role in bindings:
            raise AuditError("Duplicate role name")
        if key not in inventory.tensors:
            raise AuditError(f"Missing required tensor {key!r} for role {role!r}")
        actual = inventory.tensors[key]
        if actual.dtype != dtype or actual.shape != shape:
            raise AuditError(f"Actual header dtype/shape mismatch for role {role!r}")
        bindings[role] = actual
    if schema.reject_unexpected and set(inventory.tensors) != {t.name for t in bindings.values()}:
        raise AuditError("Unexpected tensor keys")
    capabilities = {}
    for capability in schema.capabilities:
        if not isinstance(capability, CapabilityRequirement):
            raise AuditError("Invalid capability declaration")
        name = _name(capability.name, "capability name", limits)
        if name in capabilities or type(capability.requested) is not bool or not isinstance(capability.roles, (tuple, list)) or not capability.roles:
            raise AuditError("Duplicate or invalid capability")
        if len(capability.roles) > limits.max_tensors:
            raise AuditError("Too many capability roles")
        for role in capability.roles:
            _name(role, "capability role", limits)
        if len(set(capability.roles)) != len(capability.roles) or any(role not in bindings for role in capability.roles):
            raise AuditError("Capability refers to missing/unknown required roles")
        capabilities[name] = "metadata_ready" if capability.requested else "not_requested"
    for matrix in schema.packed_matrices:
        if not isinstance(matrix, PackedMatrix):
            raise AuditError("Invalid packed matrix declaration")
        _packed(matrix, bindings, limits)
    return AuditResult(inventory, MappingProxyType(bindings), MappingProxyType(capabilities),
                       tuple(schema.packed_matrices))


@dataclass(frozen=True)
class ProposerProbe:
    capability: str
    output: tuple[int, ...]
    status: str  # invoked_with_proposals or invoked_without_proposals


def probe_proposer(audit, capability, proposer: Callable, prompt, *, vocab_size, max_proposals=1):
    """Invoke an engine-supplied synchronous toy/probe hook once, separately.

    Observation of invocation/token bounds is not trained arithmetic, target
    equivalence, state repair or performance evidence. Hook timing is unbounded.
    """
    if not isinstance(audit, AuditResult) or not isinstance(capability, str) or not callable(proposer):
        raise AuditError("Invalid audit/capability/proposer")
    if audit.capability_status.get(capability) != "metadata_ready":
        raise AuditError("Capability was not requested and metadata-audited")
    _integer(vocab_size, "vocabulary size", (1 << 31) - 1)
    _integer(max_proposals, "proposal bound", 64)
    if not vocab_size or not max_proposals or not isinstance(prompt, (tuple, list)) or len(prompt) > 4096:
        raise AuditError("Invalid bounded proposer input")
    for token in prompt:
        _integer(token, "prompt token", vocab_size - 1)
    output = proposer(tuple(prompt), max_proposals)
    if not isinstance(output, (tuple, list)) or len(output) > max_proposals:
        raise AuditError("Proposer must return a bounded token tuple/list")
    for token in output:
        _integer(token, "proposed token", vocab_size - 1)
    return ProposerProbe(capability, tuple(output),
                         "invoked_with_proposals" if output else "invoked_without_proposals")
