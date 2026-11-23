"""Heterogeneous byte layout and complete-state reusable-prefix intersection."""
from bisect import bisect_right
from dataclasses import dataclass
from itertools import islice
from math import lcm

_MAX = (1 << 63) - 1


def _uint(n):
    return type(n) is int and 0 <= n <= _MAX


@dataclass(frozen=True)
class StateLayout:
    name: str
    page_bytes: int
    pages: int
    alignment: int = 1


@dataclass(frozen=True)
class StateRegion:
    name: str
    offset: int
    stride: int
    pages: int

    @property
    def nbytes(self):
        return self.stride * self.pages


def plan_state_layout(components, byte_budget):
    """Keep each component's page size; align offsets/strides without an LCM.

    Offsets assume a backing allocation aligned to the largest alignment.
    Components describe physical pages, not logical tokens or model semantics.
    """
    components = tuple(islice(components, 65))
    if not _uint(byte_budget) or not 0 < len(components) <= 64:
        raise ValueError("bounded component inventory and byte budget required")
    result, names, end = [], set(), 0
    for c in components:
        if (not isinstance(c, StateLayout) or not isinstance(c.name, str) or not c.name
                or c.name in names or not _uint(c.page_bytes) or not c.page_bytes
                or not _uint(c.pages) or not _uint(c.alignment) or not c.alignment
                or c.alignment & (c.alignment - 1)):
            raise ValueError("unique components with bounded pages and power-of-two alignment required")
        stride = (c.page_bytes + c.alignment - 1) // c.alignment * c.alignment
        offset = (end + c.alignment - 1) // c.alignment * c.alignment
        end = offset + stride * c.pages
        if stride > _MAX or end > byte_budget:
            raise ValueError("state layout exceeds byte budget")
        result.append(StateRegion(c.name, offset, stride, c.pages))
        names.add(c.name)
    return tuple(result), end


@dataclass(frozen=True)
class PrefixFrontier:
    name: str
    identity: str
    maximum: int
    quantum: int = 1
    checkpoints: tuple[int, ...] | None = None

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name
                or not isinstance(self.identity, str) or not self.identity or not _uint(self.maximum)
                or not _uint(self.quantum) or not self.quantum):
            raise ValueError("named compatible frontier with bounded extent required")
        if self.checkpoints is not None:
            if (not isinstance(self.checkpoints, tuple)
                    or any(not _uint(v) or not 0 < v <= self.maximum or v % self.quantum
                           for v in self.checkpoints)
                    or tuple(sorted(set(self.checkpoints))) != self.checkpoints):
                raise ValueError("sorted unique checkpoint boundaries required")

    def predecessor(self, limit):
        value = min(limit, self.maximum)
        if self.checkpoints is None:
            return value // self.quantum * self.quantum
        index = bisect_right(self.checkpoints, value)
        return self.checkpoints[index - 1] if index else 0


def common_prefix(components, maximum, *, identity):
    """Greatest boundary supported by EVERY required state component.

    Dense/block-aligned frontiers and non-downward-closed checkpoint sets can be
    combined. Zero means recompute. A boundary alone is not a restore lease:
    revalidate inventories and acquire complete immutable bundles before use.
    """
    components = tuple(islice(components, 65))
    if (not _uint(maximum) or not identity or not 0 < len(components) <= 64
            or any(not isinstance(c, PrefixFrontier) or c.identity != identity for c in components)
            or len({c.name for c in components}) != len(components)):
        raise ValueError("complete unique compatible component frontiers required")
    # Collapse aligned dense frontiers first. Without this, coprime large page
    # quanta could require a token-by-token descent over an enormous context.
    quantum = lcm(*(c.quantum for c in components))
    candidate = min(maximum, *(c.maximum for c in components))
    candidate = candidate // quantum * quantum
    while candidate:
        narrowed = min(c.predecessor(candidate) for c in components)
        if narrowed == candidate:
            return candidate
        candidate = narrowed // quantum * quantum
    return 0
