"""Sequence ownership and stable attention reduction; no transport or model dependency."""
from dataclasses import dataclass
import math


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"integer >= {minimum} required")
    return value


def _vector(values, width=None):
    values = tuple(values)
    if not values or (width is not None and len(values) != width):
        raise ValueError("nonempty consistent vector geometry required")
    if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in values):
        raise ValueError("finite numeric vector required")
    return values


@dataclass(frozen=True)
class SequencePartition:
    world: int
    rank: int
    page_size: int = 256

    def __post_init__(self):
        _integer(self.world, 1)
        _integer(self.rank)
        _integer(self.page_size, 1)
        if self.rank >= self.world:
            raise ValueError("rank outside partition group")

    def owner(self, position):
        return _integer(position) % self.world

    def local(self, position):
        if self.owner(position) != self.rank:
            raise ValueError("position belongs to another rank")
        return position // self.world

    def global_position(self, local):
        return _integer(local) * self.world + self.rank

    def length(self, total):
        total = _integer(total)
        return total // self.world + int(self.rank < total % self.world)

    def pages(self, capacity):
        count = self.length(capacity)
        return (count + self.page_size - 1) // self.page_size

    def address(self, position):
        """Local (page, row); global causal/position transforms still use position."""
        return divmod(self.local(position), self.page_size)


@dataclass(frozen=True)
class AttentionPartial:
    output: tuple[float, ...]
    log_normalizer: float

    def __post_init__(self):
        output = _vector(self.output)
        object.__setattr__(self, "output", output)
        lse = self.log_normalizer
        if isinstance(lse, bool) or not isinstance(lse, (int, float)) or math.isnan(lse) or lse == math.inf:
            raise ValueError("finite log normalizer or explicit empty -inf required")
        if lse == -math.inf and any(x != 0 for x in output):
            raise ValueError("empty shard must supply a zero output")


def attention_partial(query, rows, *, value_width, query_position, scale=None,
                      selected_positions=None):
    """rows contains (global_position, key, value); position supplies the causal mask.

    Sparse selection, when supplied, names global positions selected before
    partitioning. Missing selected positions may live on other ranks. Query/key
    position transforms are already applied by the caller.
    """
    query = _vector(query)
    _integer(value_width, 1)
    _integer(query_position)
    scale = 1 / math.sqrt(len(query)) if scale is None else scale
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("positive finite scale required")
    selected = None
    if selected_positions is not None:
        positions = tuple(selected_positions)
        for p in positions:
            _integer(p)
        selected = set(positions)
        if len(selected) != len(positions):
            raise ValueError("duplicate selected position")
    seen, logits, values = set(), [], []
    for position, key, value in rows:
        _integer(position)
        if position in seen:
            raise ValueError("duplicate shard position")
        seen.add(position)
        key, value = _vector(key, len(query)), _vector(value, value_width)
        if position > query_position or (selected is not None and position not in selected):
            continue
        score = math.fsum(q * k for q, k in zip(query, key)) * scale
        if not math.isfinite(score):
            raise ValueError("attention score overflow")
        logits.append(score)
        values.append(value)
    if not logits:
        return AttentionPartial((0.0,) * value_width, -math.inf)
    maximum = max(logits)
    weights = [math.exp(x - maximum) for x in logits]
    total = math.fsum(weights)
    output = tuple(math.fsum((w / total) * row[c] for w, row in zip(weights, values))
                   for c in range(value_width))
    return AttentionPartial(output, maximum + math.log(total))


def merge_partials(partials):
    """Merge disjoint shards for one identical query/head/scale/selection.

    A normalizer is sufficient: full logits/KV never need to cross the group.
    All-empty input is rejected; invalid nonempty shards are never discarded.
    """
    partials = tuple(partials)
    if not partials or any(not isinstance(p, AttentionPartial) for p in partials):
        raise ValueError("attention partials required")
    width = len(partials[0].output)
    if any(len(p.output) != width for p in partials):
        raise ValueError("partial output geometry differs")
    maximum = max(p.log_normalizer for p in partials)
    if maximum == -math.inf:
        raise ValueError("attention has no visible keys")
    weights = [math.exp(p.log_normalizer - maximum) for p in partials]
    total = math.fsum(weights)
    output = tuple(math.fsum((w / total) * p.output[c] for w, p in zip(weights, partials))
                   for c in range(width))
    return AttentionPartial(output, maximum + math.log(total))


def combine_and_scatter(shards):
    """Host equivalent of normalizer all-gather plus weighted reduce-scatter.

    Input: [rank][query][head] AttentionPartial. Output: [rank][local_head][query]
    AttentionPartial, including the combined normalizer for hierarchical reduction.
    Each rank supplies the same ordered queries/heads, over disjoint token shards.
    """
    shards = tuple(tuple(tuple(row) for row in shard) for shard in shards)
    if not shards or not shards[0] or not shards[0][0]:
        raise ValueError("nonempty rank/query/head geometry required")
    world, queries, heads = len(shards), len(shards[0]), len(shards[0][0])
    if heads % world or any(len(s) != queries or any(len(r) != heads for r in s) for s in shards):
        raise ValueError("matching query/head geometry and divisible head count required")
    flat = [p for shard in shards for row in shard for p in row]
    if any(not isinstance(p, AttentionPartial) for p in flat) or any(
        len(p.output) != len(flat[0].output) for p in flat
    ):
        raise ValueError("matching partial value width required")
    merged = tuple(tuple(merge_partials(shards[r][q][h] for r in range(world))
                         for q in range(queries)) for h in range(heads))
    per_rank = heads // world
    return tuple(merged[r * per_rank:(r + 1) * per_rank] for r in range(world))
