"""Checked opaque component bundles and private, settlement-bound restoration.

All control-plane calls are serialized. Backend callbacks register every owner
and physical-work receipt BEFORE a potentially ambiguous submission can throw.
Receipts' done() means physical settlement, not merely host dispatch completion.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import re
import struct

_HEADER = struct.Struct("<8sI")
_MAGIC = b"CRITBND1"
_MAX_MANIFEST = 65536
_MAX_COMPONENTS = 64
_MAX_RETAINED = 16
_ACTIVE_RESTORES = {}
_ACTIVE_WRITES = {}


def _text(value, maximum):
    return isinstance(value, str) and 0 < len(value.encode("utf-8")) <= maximum


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    codec: str
    max_bytes: int

    def __post_init__(self):
        if (not isinstance(self.name, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.name) is None
                or not _text(self.codec, 128) or type(self.max_bytes) is not int
                or not 0 <= self.max_bytes <= 1 << 30):
            raise ValueError("bounded component name, codec and byte extent required")


@dataclass(frozen=True)
class BundleSpec:
    fingerprint: str
    codec: str
    components: tuple
    max_payload_bytes: int = 1 << 26

    def __post_init__(self):
        components = tuple(self.components)
        if (not _text(self.fingerprint, 256) or not _text(self.codec, 128)
                or not 0 < len(components) <= _MAX_COMPONENTS
                or any(not isinstance(c, ComponentSpec) for c in components)
                or type(self.max_payload_bytes) is not int
                or not 0 <= self.max_payload_bytes <= 1 << 30):
            raise ValueError("explicit immutable bounded bundle identity required")
        if len({c.name for c in components}) != len(components):
            raise ValueError("duplicate required component")
        object.__setattr__(self, "components", tuple(sorted(components, key=lambda c: c.name)))


@dataclass(frozen=True)
class Component:
    name: str
    codec: str
    offset: int
    nbytes: int
    sha256: str


@dataclass(frozen=True)
class RestorePlan:
    spec: BundleSpec
    prefix: bytes
    payload: bytes
    components: tuple

    def source(self, component):
        if component not in self.components:
            raise ValueError("component is outside the validated restore plan")
        return memoryview(self.payload)[component.offset:component.offset + component.nbytes]


def encode_bundle(spec, values):
    values = dict(values)
    if set(values) != {c.name for c in spec.components}:
        raise ValueError("the exact required component set must be supplied")
    offset, descriptors, parts = 0, [], []
    for required in spec.components:
        part = values[required.name]
        if not isinstance(part, bytes) or len(part) > required.max_bytes:
            raise ValueError("immutable component bytes exceed the declared codec extent")
        if len(part) > spec.max_payload_bytes - offset:
            raise ValueError("bundle payload budget exceeded")
        descriptors.append({"name": required.name, "codec": required.codec,
                            "offset": offset, "nbytes": len(part),
                            "sha256": hashlib.sha256(part).hexdigest()})
        parts.append(part)
        offset += len(part)
    manifest = json.dumps({"format": "crit-components-v1", "fingerprint": spec.fingerprint,
                           "codec": spec.codec, "components": descriptors},
                          sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(manifest) > _MAX_MANIFEST:
        raise ValueError("bundle manifest budget exceeded")
    return _HEADER.pack(_MAGIC, len(manifest)) + manifest + b"".join(parts)


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate bundle manifest field")
        result[key] = value
    return result


def decode_bundle(spec, payload, *, prefix=b""):
    if (not isinstance(payload, bytes) or not isinstance(prefix, bytes)
            or not _HEADER.size <= len(payload) <= _HEADER.size + _MAX_MANIFEST + spec.max_payload_bytes):
        raise ValueError("bounded immutable checkpoint bytes required")
    magic, manifest_bytes = _HEADER.unpack_from(payload)
    if magic != _MAGIC or not 0 < manifest_bytes <= _MAX_MANIFEST or manifest_bytes > len(payload) - _HEADER.size:
        raise ValueError("malformed bundle header or manifest extent")
    try:
        manifest = json.loads(payload[_HEADER.size:_HEADER.size + manifest_bytes], object_pairs_hook=_unique_json)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("malformed bundle manifest") from error
    if (type(manifest) is not dict or set(manifest) != {"format", "fingerprint", "codec", "components"}
            or manifest["format"] != "crit-components-v1" or manifest["fingerprint"] != spec.fingerprint
            or manifest["codec"] != spec.codec or type(manifest["components"]) is not list
            or len(manifest["components"]) != len(spec.components)):
        raise ValueError("checkpoint bundle identity or complete component set differs")
    data_start = _HEADER.size + manifest_bytes
    data_bytes = len(payload) - data_start
    if data_bytes > spec.max_payload_bytes:
        raise ValueError("bundle payload budget exceeded")
    offset, components = 0, []
    for required, item in zip(spec.components, manifest["components"]):
        if (type(item) is not dict or set(item) != {"name", "codec", "offset", "nbytes", "sha256"}
                or item["name"] != required.name or item["codec"] != required.codec
                or type(item["offset"]) is not int or item["offset"] != offset
                or type(item["nbytes"]) is not int or not 0 <= item["nbytes"] <= required.max_bytes
                or item["nbytes"] > data_bytes - offset
                or not isinstance(item["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None):
            raise ValueError("component codec, contiguous extent or digest is invalid")
        absolute = data_start + offset
        view = memoryview(payload)[absolute:absolute + item["nbytes"]]
        if hashlib.sha256(view).hexdigest() != item["sha256"]:
            raise ValueError("checkpoint component checksum differs")
        components.append(Component(required.name, required.codec, absolute, item["nbytes"], item["sha256"]))
        offset += item["nbytes"]
    if offset != data_bytes:
        raise ValueError("unclaimed checkpoint component bytes")
    return RestorePlan(spec, prefix, payload, tuple(components))


class BundleStore:
    """Layer on a caller-supplied CheckpointStore; preserve its single-writer rule."""
    def __init__(self, store, spec):
        self.store, self.spec = store, spec
        self._compatible()

    def _compatible(self):
        if self.store.fingerprint != self.spec.fingerprint or self.store.codec != self.spec.codec:
            raise ValueError("outer checkpoint and component bundle identities differ")

    def put(self, prefix, values):
        self._compatible()
        return self.store.put(prefix, encode_bundle(self.spec, values))

    def exact(self, prefix):
        self._compatible()
        checkpoint = self.store.exact(prefix)
        return decode_bundle(self.spec, checkpoint.payload, prefix=checkpoint.prefix)

    def longest_prefix(self, prefix):
        self._compatible()
        checkpoint = self.store.longest_prefix(prefix)
        return None if checkpoint is None else decode_bundle(self.spec, checkpoint.payload, prefix=checkpoint.prefix)


class PendingRestore(RuntimeError):
    pass


class RestoreFailure(RuntimeError):
    def __init__(self, transaction, reason):
        super().__init__(str(reason))
        self.transaction = transaction


@dataclass
class _Work:
    receipt: object
    settled: bool = False
    succeeded: bool = False


class CopyOperation:
    def __init__(self, component, source):
        self.component, self.source = component, source
        self.destination = None
        self.owners, self.work = [], []
        self.copy_receipt = None
        self.submitted = False
        self.uncertain = False

    def retain(self, owner):
        """Retain allocator/registration/reader owners before backend submission."""
        self.owners.append(owner)
        return owner

    def adopt(self, destination):
        self.retain(destination)
        if destination is None or self.destination is not None and self.destination is not destination:
            raise ValueError("each component requires one private destination owner")
        self.destination = destination
        return destination

    def track_work(self, receipt):
        """Track any physical reader/copy work that can still access these owners."""
        if not callable(getattr(receipt, "done", None)) or not callable(getattr(receipt, "result", None)):
            self.retain(receipt)
            raise ValueError("physical-settlement receipt required")
        if not any(w.receipt is receipt for w in self.work):
            self.work.append(_Work(receipt))
        return receipt

    def record_copy(self, receipt):
        self.track_work(receipt)
        if self.copy_receipt is not None and self.copy_receipt is not receipt:
            raise ValueError("copy receipt changed after registration")
        self.copy_receipt = receipt
        return receipt


@dataclass(frozen=True)
class PublishedBundle:
    spec: BundleSpec
    prefix: bytes
    destinations: tuple


def retained_restores():
    """Strongly retained active/failed transactions; explicit settlement retires them."""
    return tuple(_ACTIVE_RESTORES.values())


class RestoreTransaction:
    def __init__(self, plan):
        # RestorePlan has a public constructor. Revalidate all bytes before any
        # destination allocation, rather than trusting a forged descriptor list.
        self.plan = decode_bundle(plan.spec, plan.payload, prefix=plan.prefix)
        self.operations = tuple(CopyOperation(c, self.plan.source(c)) for c in self.plan.components)
        self.state = "created"
        self.failure = None
        self._callbacks = ()
        self._busy = False

    @contextmanager
    def _guard(self):
        if self._busy:
            raise RuntimeError("reentrant checkpoint restoration")
        self._busy = True
        try:
            yield
        finally:
            self._busy = False

    def _fail(self, error):
        self.state, self.failure = "failed", error
        raise RestoreFailure(self, error) from error

    def submit(self, allocate, enqueue):
        with self._guard():
            if self.state != "created" or not callable(allocate) or not callable(enqueue):
                raise ValueError("single-use restore submission and backend callbacks required")
            if len(_ACTIVE_RESTORES) >= _MAX_RETAINED:
                raise ValueError("retained restore capacity exhausted; settle existing work first")
            _ACTIVE_RESTORES[id(self)] = self
            self._callbacks = (allocate, enqueue)
            self.state = "submitting"
            try:
                # Allocate all PRIVATE destinations before submitting the first
                # copy. An allocation callback must retain any partial owner
                # in the operation before throwing.
                for operation in self.operations:
                    operation.uncertain = True
                    destination = allocate(operation)
                    operation.adopt(destination)
                    if any(other is not operation and other.destination is destination
                           for other in self.operations):
                        raise ValueError("distinct private component destination views required")
                    operation.uncertain = False
                for operation in self.operations:
                    operation.submitted = True
                    operation.uncertain = True
                    receipt = enqueue(operation)
                    operation.record_copy(receipt)
                    operation.uncertain = False
            except BaseException as error:
                self._fail(error)
            self.state = "submitted"
            return self

    def _observe(self):
        if self.state not in {"submitted", "failed"}:
            raise ValueError("restore has no submitted work")
        failure = self.failure
        pending = False
        for operation in self.operations:
            pending |= operation.uncertain
            for work in operation.work:
                if work.settled:
                    continue
                try:
                    done = work.receipt.done()
                    if type(done) is not bool:
                        raise ValueError("receipt done() must prove physical settlement")
                    if not done:
                        pending = True
                        continue
                    work.settled = True
                    work.receipt.result()
                    work.succeeded = True
                except BaseException as error:
                    failure = failure or error
        if failure is not None:
            self._fail(failure)
        return not pending and all(o.copy_receipt is not None for o in self.operations)

    def poll(self):
        with self._guard():
            return self._observe()

    def publish(self):
        with self._guard():
            if not self._observe():
                raise PendingRestore("complete checkpoint restore is still pending")
            result = PublishedBundle(self.plan.spec, self.plan.prefix,
                                     tuple((o.component.name, o.destination) for o in self.operations))
            self.state = "published"
            _ACTIVE_RESTORES.pop(id(self))
            self.plan, self.operations, self._callbacks = None, (), ()
            return result

    def discard(self, settle=None):
        """Drop private state only after all physical work is settled.

        settle(operation)->True is an explicit backend proof for uncertain
        submission/allocation or unregistered work; it must cover EVERY retained
        owner and reader. False/pending/throwing proofs retain the whole bundle.
        """
        with self._guard():
            if self.state == "created":
                self.state, self.plan, self.operations = "discarded", None, ()
                return
            if self.state not in {"submitted", "failed"}:
                raise ValueError("restore is already consumed")
            unresolved = False
            for operation in self.operations:
                pending = operation.uncertain
                for work in operation.work:
                    if work.settled:
                        continue
                    try:
                        done = work.receipt.done()
                        if type(done) is not bool:
                            raise ValueError("receipt settlement proof differs")
                        work.settled = done
                        pending |= not done
                    except BaseException:
                        pending = True
                if pending and settle is not None:
                    try:
                        proof = settle(operation)
                        if type(proof) is not bool:
                            raise ValueError("backend settlement proof must be boolean")
                        pending = not proof
                        if proof:
                            operation.uncertain = False
                            for work in operation.work:
                                work.settled = True
                    except BaseException:
                        pending = True
                unresolved |= pending
            if unresolved:
                raise PendingRestore("private checkpoint owners retained until complete settlement")
            _ACTIVE_RESTORES.pop(id(self))
            self.state, self.plan, self.operations, self._callbacks = "discarded", None, (), ()
            self.failure = None


@dataclass(frozen=True)
class RangeFragment:
    lease: object
    begin: int
    end: int

    def __post_init__(self):
        if (not isinstance(self.lease, ActiveWriteLease) or self.lease.source is None
                or type(self.begin) is not int or type(self.end) is not int
                or not 0 <= self.begin <= self.end <= len(self.lease.source)):
            raise ValueError("fragment must name a bounded active physical source range")

    def split(self, at):
        if type(at) is not int or not self.begin < at < self.end:
            raise ValueError("split must be strictly inside the physical source range")
        return RangeFragment(self.lease, self.begin, at), RangeFragment(self.lease, at, self.end)

    def source(self):
        if self.lease.source is None:
            raise ValueError("write range lease has retired")
        return memoryview(self.lease.source)[self.begin:self.end]


class ActiveWriteLease:
    """One physical immutable source lease shared by every later range fragment."""
    def __init__(self, source, receipt, *, max_bytes=1 << 26):
        if (not isinstance(source, bytes) or type(max_bytes) is not int or not 0 <= max_bytes <= 1 << 30
                or len(source) > max_bytes or not callable(getattr(receipt, "done", None))
                or not callable(getattr(receipt, "result", None))):
            raise ValueError("bounded immutable write source and settlement receipt required")
        if len(_ACTIVE_WRITES) >= _MAX_RETAINED:
            raise ValueError("active write lease capacity exhausted")
        self._source, self._receipt = source, receipt
        _ACTIVE_WRITES[id(self)] = self

    @property
    def source(self):
        return self._source

    @property
    def receipt(self):
        return self._receipt

    def fragment(self):
        if self.source is None:
            raise ValueError("write range lease has retired")
        return RangeFragment(self, 0, len(self.source))

    def retire(self, settle=None):
        if self.source is None:
            raise ValueError("write range lease already retired")
        try:
            done = self.receipt.done()
            if type(done) is not bool:
                raise ValueError("write settlement proof must be boolean")
        except BaseException:
            if settle is None:
                raise
            done = False
        explicit = not done
        if not done:
            proof = settle(self) if settle is not None else False
            if type(proof) is not bool or not proof:
                raise PendingRestore("all split source fragments remain pinned during the write")
        # done() proves physical settlement even if result() reports an I/O
        # error. Release exactly this receipt's physical owner, then propagate.
        receipt = self.receipt
        _ACTIVE_WRITES.pop(id(self))
        self._source, self._receipt = None, None
        if not explicit:
            receipt.result()


def retained_writes():
    return tuple(_ACTIVE_WRITES.values())
