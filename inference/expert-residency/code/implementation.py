"""Representation-keyed expert cache with leases and settled-only eviction."""
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class ExpertKey:
    revision: str
    layer: int
    expert: int
    encoding: str

    def validate(self):
        if not self.revision or not self.encoding or any(type(x) is not int or x < 0 for x in (self.layer, self.expert)):
            raise ValueError("complete expert representation identity required")


@dataclass
class Resident:
    owner: object
    nbytes: int
    generation: int
    pins: int = 0
    uses: list = field(default_factory=list)

    def settled(self):
        self.uses[:] = [use for use in self.uses if not use.done()]
        return not self.uses


class ExpertCache:
    def __init__(self, capacity, release):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("positive byte capacity required")
        self.capacity, self.release = capacity, release
        self.entries = {}
        self.live = 0
        self.generation = 0

    def acquire(self, key, nbytes, load):
        key.validate()
        if type(nbytes) is not int or not 0 < nbytes <= self.capacity:
            raise ValueError("expert exceeds declared cache capacity")
        if key in self.entries:
            entry = self.entries.pop(key)
            self.entries[key] = entry  # least recently acquired first
            if nbytes != entry.nbytes:
                raise ValueError("same expert identity has a different extent")
        else:
            while self.live + nbytes > self.capacity:
                candidate = next((k for k, v in self.entries.items() if not v.pins and v.settled()), None)
                if candidate is None:
                    raise ValueError("all eviction candidates are pinned or in flight")
                old = self.entries[candidate]
                self.release(old.owner)  # exception retains entry/accounting
                del self.entries[candidate]
                self.live -= old.nbytes
            owner = load(key, nbytes)  # service owns partial failures; returns a ready physical owner
            self.generation += 1
            entry = Resident(owner, nbytes, self.generation)
            self.entries[key] = entry
            self.live += nbytes
        entry.pins += 1
        return ExpertLease(entry)


class ExpertLease:
    def __init__(self, resident):
        self.resident, self.closed = resident, False

    @property
    def owner(self):
        if self.closed:
            raise ValueError("closed expert lease")
        return self.resident.owner

    @property
    def generation(self):
        return self.resident.generation

    def close(self, completion=None):
        if self.closed:
            return
        if completion is not None:
            self.resident.uses.append(completion)
        self.closed = True
        self.resident.pins -= 1


@dataclass(frozen=True)
class BundleComponent:
    """One slot-indexed component; encoding describes its actual tensor layout."""
    name: str
    encoding: str
    nbytes: int


@dataclass(frozen=True)
class BundleTicket:
    slot: int
    generation: int
    key: ExpertKey


@dataclass(frozen=True)
class BundleReceipt:
    """Completion bound to this upload, not an earlier recording of an event."""
    ticket: BundleTicket
    component: str
    encoding: str
    nbytes: int
    completion: object  # done() proves settlement; result() reports copy success


@dataclass(frozen=True)
class BundleUse:
    slot: int
    generation: int
    key: ExpertKey
    identity: object = field(default_factory=object)


@dataclass
class _BundleCopy:
    source_owner: object
    completion: object = None


@dataclass
class _BundleSlot:
    key: object = None
    generation: int = 0
    tick: int = 0
    pins: int = 0
    pending: object = None
    copies: dict = field(default_factory=dict)
    failed_enqueue: bool = False


class BundleDirectory:
    """Serialized metadata for complete bundles in caller-owned fixed buffers.

    Acquire every existing route hit before choosing any replacement. Components
    become visible together only after successful operation-specific completion.
    This is a host protocol, not a transfer engine or device publication kernel.
    """
    def __init__(self, capacity, components, backing_owners):
        components, backing_owners = tuple(components), tuple(backing_owners)
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("positive slot capacity required")
        if not components or any(not isinstance(c, BundleComponent)
                or not isinstance(c.name, str) or not c.name
                or not isinstance(c.encoding, str) or not c.encoding
                or type(c.nbytes) is not int or c.nbytes <= 0 for c in components):
            raise ValueError("complete encoded component descriptors required")
        if len({c.name for c in components}) != len(components):
            raise ValueError("duplicate bundle component")
        if not backing_owners or any(owner is None for owner in backing_owners):
            raise ValueError("reserved backing owners required")
        self.components = MappingProxyType({c.name: c for c in components})
        self.backing_owners = backing_owners
        self.slots = [_BundleSlot() for _ in range(capacity)]
        self.uses = {}
        self.tick = 0
        self.closed = False

    def _open(self):
        if self.closed:
            raise ValueError("closed bundle directory")

    def _pending(self, ticket):
        self._open()
        if not isinstance(ticket, BundleTicket) or type(ticket.slot) is not int or not 0 <= ticket.slot < len(self.slots):
            raise ValueError("invalid bundle ticket")
        slot = self.slots[ticket.slot]
        if slot.pending is not ticket or slot.generation != ticket.generation:
            raise ValueError("foreign, stale or already completed bundle ticket")
        return slot

    def ready_slots(self):
        self._open()
        return {slot.key: (index, slot.generation) for index, slot in enumerate(self.slots) if slot.key is not None}

    def acquire(self, key, *, touch=True):
        self._open()
        key.validate()
        if type(touch) is not bool:
            raise ValueError("explicit boolean replacement-policy touch required")
        for index, slot in enumerate(self.slots):
            if slot.key == key:
                if touch:
                    self.tick += 1
                    slot.tick = self.tick
                slot.pins += 1
                use = BundleUse(index, slot.generation, key)
                self.uses[use.identity] = use
                return use
        return None

    def settle(self, use, completion):
        self._open()
        if not isinstance(use, BundleUse) or self.uses.get(use.identity) is not use:
            raise ValueError("foreign, stale or duplicate bundle use")
        if not completion.done():
            raise ValueError("bundle read has not physically settled")
        self.slots[use.slot].pins -= 1
        del self.uses[use.identity]

    def begin_upload(self, key):
        self._open()
        key.validate()
        if any(slot.key == key or (slot.pending is not None and slot.pending.key == key) for slot in self.slots):
            raise ValueError("bundle is already ready or loading")
        candidates = [(index, slot) for index, slot in enumerate(self.slots) if not slot.pins and slot.pending is None]
        if not candidates:
            raise ValueError("all slots are protected or loading; use exact source fallback")
        index, slot = min(candidates, key=lambda pair: (pair[1].key is not None, pair[1].tick))
        # An overwrite invalidates prior content. Aborting cannot resurrect it.
        slot.key = None
        slot.generation += 1
        slot.pending = BundleTicket(index, slot.generation, key)
        return slot.pending

    def submit(self, ticket, component, source_owner, enqueue):
        slot = self._pending(ticket)
        if component not in self.components or component in slot.copies or source_owner is None or slot.failed_enqueue:
            raise ValueError("invalid, duplicate or failed bundle component submission")
        copy = _BundleCopy(source_owner)
        slot.copies[component] = copy  # retain source before a possibly partial enqueue
        try:
            receipt = enqueue(ticket, self.components[component], source_owner)
            if (not isinstance(receipt, BundleReceipt) or receipt.ticket is not ticket
                    or receipt.component != component
                    or receipt.encoding != self.components[component].encoding
                    or type(receipt.nbytes) is not int or receipt.nbytes != self.components[component].nbytes
                    or not callable(getattr(receipt.completion, "done", None))
                    or not callable(getattr(receipt.completion, "result", None))):
                raise ValueError("completion must identify this bundle and component")
            copy.completion = receipt.completion
        except BaseException:
            slot.failed_enqueue = True  # work may exist without a usable receipt
            raise

    def _clear(self, slot):
        slot.pending = None
        slot.copies.clear()
        slot.failed_enqueue = False

    def publish(self, ticket):
        slot = self._pending(ticket)
        if slot.failed_enqueue or set(slot.copies) != set(self.components) or any(copy.completion is None for copy in slot.copies.values()):
            raise ValueError("complete successfully enqueued bundle required")
        if not all(copy.completion.done() for copy in slot.copies.values()):
            raise ValueError("all component copies must physically settle")
        errors = []
        for copy in slot.copies.values():
            try:
                copy.completion.result()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            self._clear(slot)  # all known readers settled; leave overwritten slot empty
            raise errors[0]
        self.tick += 1
        slot.key, slot.tick = ticket.key, self.tick
        self._clear(slot)
        return ticket.slot, ticket.generation

    def abort(self, ticket, quiesce=None):
        slot = self._pending(ticket)
        receipts = tuple(copy.completion for copy in slot.copies.values() if copy.completion is not None)
        if (slot.failed_enqueue or any(copy.completion is None for copy in slot.copies.values())) and quiesce is None:
            raise ValueError("failed enqueue requires explicit backend quiescence")
        if quiesce is not None:
            owners = self.backing_owners + tuple(copy.source_owner for copy in slot.copies.values())
            if quiesce(owners, receipts) is not True:
                raise ValueError("physical quiescence is not proven")
        if not all(receipt.done() for receipt in receipts):
            raise ValueError("submitted copies still access bundle owners")
        self._clear(slot)

    def close(self):
        self._open()
        if self.uses or any(slot.pending is not None for slot in self.slots):
            raise ValueError("bundle uses and uploads must drain before close")
        self.slots.clear()
        self.backing_owners = ()  # caller remains responsible for physical release
        self.closed = True
