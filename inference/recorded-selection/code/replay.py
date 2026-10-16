"""Portable eager/full/partial replay selection over measured backend offers."""
from dataclasses import dataclass
from itertools import islice


def _identity(value):
    return type(value) in (str, bytes, int) or (type(value) is tuple and all(_identity(v) for v in value))


@dataclass(frozen=True)
class ExecutionShape:
    phase: str
    tokens: int
    rows: int
    # Complete engine-supplied compatibility key: representation, backend,
    # physical allocation/binding generations and relevant operation semantics.
    identity: tuple

    def __post_init__(self):
        if (not isinstance(self.phase, str) or not self.phase
                or any(type(v) is not int or v <= 0 for v in (self.tokens, self.rows))
                or type(self.identity) is not tuple or not self.identity or not _identity(self.identity)):
            raise ValueError("explicit phase, positive shape and compatibility identity required")
        hash(self.identity)


@dataclass(frozen=True)
class ReplayOffer:
    name: str
    mode: str
    shape: ExecutionShape
    run_ns: int
    workspace_bytes: int = 0
    resident_bytes: int = 0
    setup_ns: int = 0
    setup_peak_bytes: int = 0
    ready: bool = False

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name or self.mode not in {"eager", "full", "partial"}
                or not isinstance(self.shape, ExecutionShape) or type(self.ready) is not bool
                or any(type(n) is not int or n < 0 for n in (
                    self.run_ns, self.workspace_bytes, self.resident_bytes,
                    self.setup_ns, self.setup_peak_bytes))):
            raise ValueError("valid execution mode and nonnegative measured costs required")
        if self.mode == "eager" and (self.resident_bytes or self.setup_ns or self.setup_peak_bytes):
            raise ValueError("eager offer cannot conceal capture costs")


@dataclass(frozen=True)
class ReplayChoice:
    offer: ReplayOffer
    total_ns: int
    rejected: tuple[tuple[str, str], ...]


def select_replay(shape, offers, *, modes, free_bytes, expected_uses=1, allow_padding=False):
    """Choose minimum run*uses + uncached setup cost, with stable input-order ties.

    Costs belong to the offered executed shape, including padding. Free bytes
    exclude already-resident captures; cold offers include both capture peak and
    resident+workspace demand. Engine must revalidate before capture/launch.
    """
    offers, modes = tuple(islice(offers, 257)), frozenset(modes)
    if (not isinstance(shape, ExecutionShape) or not offers or len(offers) > 256
            or any(not isinstance(o, ReplayOffer) for o in offers)
            or len({o.name for o in offers}) != len(offers)
            or not modes <= {"eager", "full", "partial"}
            or type(free_bytes) is not int or free_bytes < 0
            or type(expected_uses) is not int or expected_uses <= 0
            or type(allow_padding) is not bool):
        raise ValueError("bounded unique offers, capabilities and resource limits required")
    candidates, rejected = [], []
    for index, offer in enumerate(offers):
        reason = None
        offered = offer.shape
        exact = (offered.tokens, offered.rows) == (shape.tokens, shape.rows)
        memory = offer.workspace_bytes if offer.ready else max(
            offer.setup_peak_bytes, offer.resident_bytes + offer.workspace_bytes)
        if offer.mode not in modes:
            reason = "unsupported mode"
        elif offered.phase != shape.phase or offered.identity != shape.identity:
            reason = "incompatible phase or identity"
        elif (not exact and (not allow_padding or offer.mode == "eager"
                            or offered.tokens < shape.tokens or offered.rows < shape.rows)):
            reason = "unsupported shape"
        elif memory > free_bytes:
            reason = "insufficient memory"
        if reason:
            rejected.append((offer.name, reason))
        else:
            total = offer.run_ns * expected_uses + (0 if offer.ready else offer.setup_ns)
            candidates.append((total, index, offer))
    if not candidates:
        raise ValueError("no compatible execution offer fits the budget")
    total, _, offer = min(candidates, key=lambda candidate: candidate[:2])
    return ReplayChoice(offer, total, tuple(rejected))
