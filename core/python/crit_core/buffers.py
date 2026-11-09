"""Fixed buffer plans for Python engines: the twin of Core's plan_buffers and BufferArena (crit/device_memory.h).

An engine names and sizes every device buffer once at load, packs them into one allocation with a scratch region and
as many equal slots (cache pages, expert slots) as the rest of the device budget holds after a reserve, and allocates
that once. It then needs no framework allocator at run time, whose temporaries need gigabytes of headroom, so the
reserve shrinks to what lies outside the plan (the driver, captured graphs). Memory a smaller plan no longer needs (a
lower context's KV pages, say) becomes slots when the engine plans again.

    plan = plan_buffers([BufferRequest("kv", kv_bytes, 4096), BufferRequest("logits", logit_bytes)],
                        available=free_bytes, reserve=reserve, scratch_bytes=scratch, slot_bytes=expert_bytes)
    block = torch.empty(plan.total_bytes, dtype=torch.uint8, device="cuda")   # the one allocation
    kv = plan.view(block, "kv")                                              # block[offset:offset + bytes]
    slots = [plan.slot(block, i) for i in range(plan.slots)]
    scratch = ScratchArena(plan.scratch(block))                              # mark / allocate / rewind per step

Views are slices of whatever the engine allocated (a torch uint8 tensor, a bytearray), so this module needs no
framework.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class BufferRequest:
    name: str
    bytes: int
    alignment: int = 256


@dataclass(frozen=True)
class PlannedBuffer:
    name: str
    offset: int
    bytes: int


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _power_of_two(value):
    return value > 0 and value & (value - 1) == 0


@dataclass
class BufferPlan:
    buffers: List[PlannedBuffer] = field(default_factory=list)
    scratch_offset: int = 0
    scratch_bytes: int = 0
    slots_offset: int = 0
    slot_bytes: int = 0
    slot_stride: int = 0
    slots: int = 0
    total_bytes: int = 0      # the one allocation
    reserve_bytes: int = 0
    unused_bytes: int = 0     # available - reserve - total: under one slot unless max_slots bound

    def buffer(self, name):
        for b in self.buffers:
            if b.name == name:
                return b
        raise KeyError(f"buffer plan has no buffer {name}")

    def view(self, block, name):
        b = self.buffer(name)
        return block[b.offset:b.offset + b.bytes]

    def slot(self, block, i):
        if not 0 <= i < self.slots:
            raise IndexError(f"slot {i} past {self.slots}")
        start = self.slots_offset + i * self.slot_stride
        return block[start:start + self.slot_bytes]

    def scratch(self, block):
        if not self.scratch_bytes:
            raise ValueError("the plan has no scratch region")
        return block[self.scratch_offset:self.scratch_offset + self.scratch_bytes]


def plan_buffers(requests, available, reserve=0, scratch_bytes=0, slot_bytes=0, slot_alignment=256, min_slots=0,
                 max_slots=None):
    """Fixed buffers in request order at their alignment, then the scratch region (256-aligned), then slots: as many
    as fit available - reserve, at most max_slots. Raises ValueError for an empty or repeated name, a zero-byte buffer
    or an alignment that is not a power of two, and MemoryError naming the shortfall when the buffers, scratch and
    min_slots do not fit."""
    plan, at, names = BufferPlan(), 0, set()
    for r in requests:
        if not r.name:
            raise ValueError("a buffer needs a name")
        if r.bytes <= 0:
            raise ValueError(f"{r.name} has no bytes")
        if not _power_of_two(r.alignment):
            raise ValueError(f"{r.name}'s alignment is not a power of two")
        if r.name in names:
            raise ValueError(f"{r.name} is named twice")
        names.add(r.name)
        at = _align(at, r.alignment)
        plan.buffers.append(PlannedBuffer(r.name, at, r.bytes))
        at += r.bytes
    if not _power_of_two(slot_alignment):
        raise ValueError("slot alignment is not a power of two")
    if max_slots is not None and min_slots > max_slots:
        raise ValueError("min_slots above max_slots")
    if scratch_bytes:
        at = _align(at, 256)
        plan.scratch_offset, plan.scratch_bytes = at, scratch_bytes
        at += scratch_bytes
    plan.reserve_bytes = reserve
    room = max(0, available - reserve)
    plan.slot_bytes = slot_bytes
    if slot_bytes:
        at = _align(at, slot_alignment)
        plan.slots_offset, plan.slot_stride = at, _align(slot_bytes, slot_alignment)
        plan.slots = max(0, (room - at) // plan.slot_stride) if room > at else 0
        if max_slots is not None:
            plan.slots = min(plan.slots, max_slots)
    else:
        plan.slots_offset = at
    needed = at + (min_slots * plan.slot_stride if slot_bytes else 0)
    if needed > room:
        raise MemoryError(f"{needed} bytes of buffers, scratch and slots do not fit {room} bytes (available less "
                          f"reserve); short by {needed - room}")
    plan.total_bytes = at + plan.slots * plan.slot_stride
    plan.unused_bytes = room - plan.total_bytes
    return plan


class ScratchArena:
    """A per-step bump allocator over a scratch view: allocate(bytes, alignment) returns a slice, mark() and rewind()
    reset it between steps, peak sizes the next plan's scratch. Offsets are relative to the view's start, so align the
    view itself (the plan's scratch region is 256-aligned within the block)."""

    def __init__(self, view):
        self.view, self.capacity = view, len(view)
        self.offset = self.peak = 0

    def allocate(self, nbytes, alignment=256):
        if not _power_of_two(alignment):
            raise ValueError("arena alignment must be a power of two")
        start = _align(self.offset, alignment)
        if start + nbytes > self.capacity:
            raise MemoryError("arena capacity exceeded")
        self.offset = start + nbytes
        self.peak = max(self.peak, self.offset)
        return self.view[start:start + nbytes]

    def mark(self):
        return self.offset

    def rewind(self, mark):
        if mark > self.offset:
            raise ValueError("arena cannot rewind to a future mark")
        self.offset = mark
