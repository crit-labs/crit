"""Draft against an indexed row subset of the model's output head; host reference.

The subset is a list of token IDs whose rows of the existing head are used as-is
(a row gather, no requantization). Verification always uses the full head, so
greedy output is unchanged; the subset only changes which drafts are proposed.
"""
from dataclasses import dataclass
import math


def choose_subset(counts, size, required=()):
    """Return `size` ascending token IDs: `required` first, then by count.

    `counts` is a mapping {token: count} or a sequence indexed by token. Ties in
    count prefer the lower token ID. The size is the engine's choice.
    """
    items = dict(enumerate(counts)) if not isinstance(counts, dict) else dict(counts)
    if any(not isinstance(t, int) or t < 0 for t in items) or any(c < 0 for c in items.values()):
        raise ValueError("token IDs and counts must be nonnegative")
    required = tuple(dict.fromkeys(required))
    if not isinstance(size, int) or size < len(required) or size <= 0:
        raise ValueError("subset size must be positive and hold every required token")
    chosen = list(required)
    for token, _ in sorted(items.items(), key=lambda item: (-item[1], item[0])):
        if len(chosen) == size:
            break
        if token not in chosen:
            chosen.append(token)
    if len(chosen) < size:
        raise ValueError("fewer known tokens than the requested subset size")
    return tuple(sorted(chosen))


def prior_subset(vocabulary, size, required=()):
    """Subset without workload statistics: `required` plus the lowest token IDs.

    For merge-ordered vocabularies (byte-pair encodings), lower IDs were merged
    earlier, i.e. were more frequent in the tokenizer's corpus, so the ID prefix
    is a reasonable prior. Special and control tokens usually sit at the end of
    the vocabulary: pass the ones the target emits (stops, turn delimiters,
    format markers) as `required`, or a prefix subset can never draft them.
    """
    if not isinstance(vocabulary, int) or vocabulary <= 0:
        raise ValueError("positive vocabulary required")
    required = tuple(dict.fromkeys(required))
    if any(not isinstance(t, int) or not 0 <= t < vocabulary for t in required):
        raise ValueError("required tokens must be in the vocabulary")
    if not isinstance(size, int) or size < len(required) or not 0 < size <= vocabulary:
        raise ValueError("subset size must be positive, within the vocabulary and hold every required token")
    chosen = set(required)
    token = 0
    while len(chosen) < size:
        chosen.add(token)
        token += 1
    return tuple(sorted(chosen))


class SubsetHead:
    """Map subset index <-> token ID. Rows ascend so index order is ID order."""

    def __init__(self, rows, vocabulary):
        rows = tuple(rows)
        if not rows or any(not isinstance(r, int) for r in rows) or rows[0] < 0 or rows[-1] >= vocabulary:
            raise ValueError("nonempty in-vocabulary row IDs required")
        if any(a >= b for a, b in zip(rows, rows[1:])):
            raise ValueError("subset rows must be strictly ascending")
        self.rows, self.vocabulary = rows, vocabulary
        self._index = {token: i for i, token in enumerate(rows)}

    def __len__(self):
        return len(self.rows)

    def token(self, index):
        return self.rows[index]

    def index(self, token):
        return self._index.get(token)

    def ranges(self):
        """Contiguous half-open row runs [(begin, end), ...]. A kernel can read a
        few runs as plain row blocks of the head, with no index array; subset
        index i is still the i-th row across the runs in order."""
        runs, begin = [], self.rows[0]
        for previous, row in zip(self.rows, self.rows[1:]):
            if row != previous + 1:
                runs.append((begin, previous + 1))
                begin = row
        runs.append((begin, self.rows[-1] + 1))
        return tuple(runs)


def logit(row, hidden):
    """One row's logit with a fixed summation order (the same for every caller)."""
    total = 0.0
    for w, h in zip(row, hidden, strict=True):
        total += w * h
    return total


def argmax(values):
    """First maximum, the ordinary greedy tie rule."""
    best = 0
    for i, value in enumerate(values):
        if value > values[best]:
            best = i
    return best


def target_argmax(head, hidden):
    return argmax([logit(row, hidden) for row in head])


def draft_argmax(head, hidden, subset):
    """Gather subset rows of the same head and map the winner back to a token ID."""
    return subset.token(argmax([logit(head[t], hidden) for t in subset.rows]))


def coverage(counts, subset):
    """Fraction of observed target tokens inside the subset: an upper bound on
    per-position draft acceptance for this workload."""
    items = dict(enumerate(counts)) if not isinstance(counts, dict) else counts
    total = sum(items.values())
    return sum(items.get(t, 0) for t in subset.rows) / total if total else 0.0


def expected_tokens(acceptance, draft_tokens):
    """Expected tokens per round with independent per-position acceptance a."""
    a = acceptance
    if not 0 <= a <= 1:
        raise ValueError("acceptance is a probability")
    return float(draft_tokens + 1) if a == 1 else (1 - a ** (draft_tokens + 1)) / (1 - a)


@dataclass(frozen=True)
class DraftHeadCost:
    full_bytes: int     # head bytes read per round when drafting with the full head
    subset_bytes: int   # head bytes read per round with the subset (rows + index)
    saved_bytes: int
    ratio: float        # subset / full draft head cost


def draft_head_cost(vocabulary, hidden, size, draft_tokens, bytes_per_weight, index_bytes=4):
    """K draft positions each read S rows of H weights plus an S-entry index,
    instead of V rows. Verification reads the full head either way."""
    if not 0 < size <= vocabulary or hidden <= 0 or draft_tokens < 0 or bytes_per_weight <= 0:
        raise ValueError("positive geometry with size <= vocabulary required")
    full = draft_tokens * vocabulary * hidden * bytes_per_weight
    subset = draft_tokens * size * (hidden * bytes_per_weight + index_bytes)
    full, subset = int(math.ceil(full)), int(math.ceil(subset))
    return DraftHeadCost(full, subset, full - subset, subset / full if full else 0.0)


def choose_size(measured):
    """The subset size with the most tokens per millisecond of round time, from measurements of the engine at
    each candidate size on a mixed workload: measured = {size: (tokens_per_round, round_ms)}. Coverage is only
    an upper bound on acceptance: a size that covers almost every token can still lose rounds, so choose by
    measured rounds. Returns (size, {size: tokens per ms}); ties prefer the smaller size."""
    if not measured:
        raise ValueError("at least one measured size is required")
    rates = {}
    for size, (tokens, round_ms) in measured.items():
        if size <= 0 or tokens < 1 or round_ms <= 0:
            raise ValueError("sizes are positive, a round emits at least one token and takes positive time")
        rates[size] = tokens / round_ms
    best = min(rates, key=lambda size: (-rates[size], size))
    return best, rates
