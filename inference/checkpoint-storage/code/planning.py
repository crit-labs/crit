"""Pure checkpoint admission/placement policies; storage and leases are external."""
from dataclasses import dataclass
import math


def _count(value, name, maximum=1 << 40):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a bounded nonnegative integer")
    return value


def _cost(value, name, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _tokens(values):
    values = tuple(values)
    if len(values) > 1048576 or any(type(x) is not int or not 0 <= x < 1 << 31 for x in values):
        raise ValueError("bounded signed-32-bit token IDs required")
    return values


@dataclass(frozen=True)
class PrefillCost:
    chunk_tokens: int
    call_seconds: float
    token_seconds: float
    attention_pair_seconds: float = 0

    def __post_init__(self):
        _count(self.chunk_tokens, "chunk_tokens", 1048576)
        if not self.chunk_tokens:
            raise ValueError("positive chunk size required")
        for name in ("call_seconds", "token_seconds", "attention_pair_seconds"):
            _cost(getattr(self, name), name)

    def seconds(self, base, tokens):
        _count(base, "base", 1048576)
        _count(tokens, "tokens", 1048576)
        calls = (tokens + self.chunk_tokens - 1) // self.chunk_tokens
        pairs = base * tokens + tokens * (tokens + 1) // 2
        result = calls * self.call_seconds + tokens * self.token_seconds + pairs * self.attention_pair_seconds
        return _cost(result, "predicted prefill")


@dataclass(frozen=True)
class TransferCost:
    nbytes: int
    bytes_per_second: float
    batches: int = 1
    batch_seconds: float = 0

    def __post_init__(self):
        _count(self.nbytes, "nbytes")
        _count(self.batches, "batches", 1048576)
        _cost(self.bytes_per_second, "bytes_per_second", True)
        _cost(self.batch_seconds, "batch_seconds")
        if self.nbytes and not self.batches:
            raise ValueError("nonempty transfer requires a batch")

    def seconds(self):
        # Zero bytes may still incur an explicitly declared fixed restore operation.
        return _cost(self.nbytes / self.bytes_per_second + self.batches * self.batch_seconds, "transfer cost")


@dataclass(frozen=True)
class PrefixSource:
    name: str
    prefix: tuple
    fingerprint: str
    codec: str
    components: frozenset
    transfers: tuple = ()  # serialized SSD->RAM->device edges; no assumed overlap
    ready: bool = True
    committed: bool = True
    feasible: bool = True

    def __post_init__(self):
        if any(type(x) is not str or not x or len(x) > 256 for x in (self.name, self.fingerprint, self.codec)):
            raise ValueError("bounded source/representation identities required")
        components, transfers = frozenset(self.components), tuple(self.transfers)
        if (not components or len(components) > 64
                or any(type(x) is not str or not x or len(x) > 64 for x in components)
                or len(transfers) > 16 or any(not isinstance(x, TransferCost) for x in transfers)
                or any(type(x) is not bool for x in (self.ready, self.committed, self.feasible))):
            raise ValueError("complete component names, bounded transfers and boolean state required")
        object.__setattr__(self, "prefix", _tokens(self.prefix))
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "transfers", transfers)


@dataclass(frozen=True)
class SourceEstimate:
    name: str
    frontier: int
    restore_seconds: float | None
    total_seconds: float | None
    rejection: str | None


@dataclass(frozen=True)
class AdmissionPlan:
    source: PrefixSource | None
    predicted_seconds: float
    scratch_seconds: float
    estimates: tuple


def choose_prefix(query, sources, cost, *, fingerprint, codec, components):
    """Rank only exact, complete, feasible committed snapshots against recomputation.

    A returned source is a suggestion, not a lease or readiness proof. Revalidate
    its identity/generation and acquire ownership before performing restoration.
    """
    query, sources, components = _tokens(query), tuple(sources), frozenset(components)
    if (not isinstance(cost, PrefillCost) or len(sources) > 4096
            or any(not isinstance(s, PrefixSource) for s in sources)
            or len({s.name for s in sources}) != len(sources)
            or any(type(x) is not str or not x or len(x) > 256 for x in (fingerprint, codec))
            or not components or len(components) > 64
            or any(type(x) is not str or not x or len(x) > 64 for x in components)):
        raise ValueError("bounded unique sources, cost and complete request identity required")
    scratch = best = cost.seconds(0, len(query))
    selected, estimates, best_restore = None, [], 0
    for source in sources:
        rejection = None
        if source.fingerprint != fingerprint or source.codec != codec:
            rejection = "representation or conditioning differs"
        elif source.components != components:
            rejection = "complete sibling state differs"
        elif not source.ready or not source.committed or not source.feasible:
            rejection = "source is pending, provisional or infeasible"
        elif not source.prefix or query[:len(source.prefix)] != source.prefix:
            rejection = "exact input prefix differs"
        if rejection:
            estimates.append(SourceEstimate(source.name, len(source.prefix), None, None, rejection))
            continue
        restore = _cost(sum(edge.seconds() for edge in source.transfers), "complete restore")
        total = _cost(restore + cost.seconds(len(source.prefix), len(query) - len(source.prefix)), "total reuse cost")
        estimates.append(SourceEstimate(source.name, len(source.prefix), restore, total, None))
        # Keep scratch on equal cost; among winning sources prefer less transfer.
        if total < best or (total == best and selected is not None and restore < best_restore):
            selected, best, best_restore = source, total, restore
    return AdmissionPlan(selected, best, scratch, tuple(estimates))


@dataclass(frozen=True)
class CaptureHint:
    position: int
    priority: int = 1  # lower wins; explicit/semantic boundaries normally use 0
    exact: bool = True

    def __post_init__(self):
        _count(self.position, "position", 1048576)
        _count(self.priority, "priority", 255)
        if type(self.exact) is not bool:
            raise ValueError("exact must be boolean")


def plan_captures(prompt_tokens, base, hints=(), *, existing=(), exclusions=(),
                  chunk_tokens=2048, max_captures=8, min_gap=64, ladder_tokens=4096):
    """Prefer semantic boundaries; put optional history captures at chunk ends.

    Exclusions are spans with no legal interior checkpoint (e.g. image regions).
    Exact hints inside them are dropped, never silently moved. The last prompt
    token is left for recomputation of the next prediction. Only plan metadata;
    captures must be performed at physically completed, committed frontiers.
    """
    for name, value in (("prompt_tokens", prompt_tokens), ("base", base),
                        ("chunk_tokens", chunk_tokens), ("max_captures", max_captures),
                        ("min_gap", min_gap), ("ladder_tokens", ladder_tokens)):
        _count(value, name, 1048576)
    hints, existing, exclusions = tuple(hints), tuple(existing), tuple(exclusions)
    if (base > prompt_tokens or not chunk_tokens or not min_gap or max_captures > 64
            or len(hints) > 4096 or len(existing) > 4096 or len(exclusions) > 4096
            or any(not isinstance(h, CaptureHint) or h.position > prompt_tokens for h in hints)):
        raise ValueError("bounded capture inputs required")
    for position in existing:
        _count(position, "existing frontier", prompt_tokens)
    for span in exclusions:
        if (type(span) not in (tuple, list) or len(span) != 2
                or any(type(p) is not int for p in span) or not 0 <= span[0] < span[1] <= prompt_tokens):
            raise ValueError("valid excluded token spans required")
    candidates = list(hints)
    if ladder_tokens:
        distance = ladder_tokens
        while distance < prompt_tokens - base:
            candidates.append(CaptureHint(prompt_tokens - distance, 3, False))
            distance *= 2
    if prompt_tokens > base + 1:
        candidates.append(CaptureHint(prompt_tokens - 1, 2, False))
    accepted = []
    for hint in sorted(candidates, key=lambda h: (h.priority, not h.exact, h.position)):
        position = hint.position
        if not hint.exact:
            position = base + (position - base) // chunk_tokens * chunk_tokens
        if not base < position < prompt_tokens or any(a < position < b for a, b in exclusions):
            continue
        if any(abs(position - p) < min_gap for p in (*existing, *(h.position for h in accepted))):
            continue
        if len(accepted) == max_captures:
            break
        accepted.append(CaptureHint(position, hint.priority, hint.exact))
    return tuple(sorted(accepted, key=lambda h: h.position))
