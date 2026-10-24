"""Host reference for batched paged attention: block-table addressing, a ragged batch and the split-key combine.

A sequence's key j lives at physical row table[j // page_tokens] * page_tokens + j % page_tokens. Sequence s with
n queries and kv_len keys places query i at position kv_len - n + i; it sees keys 0 .. that position.

Decode splits a sequence's keys into 128-key pieces; block b of `blocks` merges pieces b, b + blocks, ... with an
online softmax and keeps (m_b, l_b, o_b / l_b). The combine weighs block b by w_b = l_b exp(m_b - M), M = max m_b,
and adds the blocks in block order: out = sum_b w_b (o_b / l_b) / sum_b w_b. Which piece a block merges depends only
on key positions, so a query's result depends only on its own position and `blocks`.

Variants: `window` (query position p sees keys p - window + 1 .. p), `sink` (a logit whose weight exp(sink) joins
the combine's denominator after the blocks), `softcap` (score = softcap * tanh(score / softcap) after the scale) and
`trees` (per query a bitmask over the round's columns: query i sees round column j only when bit j is set, and its
window is measured from its depth, popcount - 1).
"""
import math

PIECE = 128


def physical_row(table, page_tokens, key):
    """The storage row of a sequence's key `key` through its block table."""
    return table[key // page_tokens] * page_tokens + key % page_tokens


def _softmax_part(q, rows, keys, scale, softcap=0.0):
    """(max, sum, unnormalized output) of q over the given (key, value) rows; (-inf, 0, zeros) when empty."""
    dim = len(rows[-1][1]) if rows else 0  # the newest row: rows before a window are None
    if not keys:
        return -math.inf, 0.0, [0.0] * dim
    scores = [scale * sum(a * b for a, b in zip(q, rows[j][0])) for j in keys]
    if softcap > 0:
        scores = [softcap * math.tanh(s / softcap) for s in scores]
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    out = [sum(w * rows[j][1][d] for w, j in zip(weights, keys)) for d in range(dim)]
    return top, sum(weights), out


def attend(storage, page_tokens, table, kv_len, queries, scale, blocks=1, window=0, sink=None, softcap=0.0, trees=None):
    """Single-head attention of a sequence's queries over its paged keys, evaluated as decode does: per-block online
    softmax over strided 128-key pieces, then the ordered combine. storage[row] = (key, value). Keys before the
    first query's window are never read (their pages may have left the window table)."""
    n = len(queries)
    if n > kv_len:
        raise ValueError("a sequence holds at least one key per query")
    base = kv_len - n
    first = max(0, base - window + 1) if window > 0 else 0
    rows = [storage[physical_row(table, page_tokens, j)] if j >= first else None for j in range(kv_len)]
    out = []
    for i, q in enumerate(queries):
        position = base + i
        depth = bin(trees[i]).count("1") - 1 if trees else i
        lo = max(0, base + depth - window + 1) if window > 0 else 0

        def sees(j):
            return lo <= j <= position and (not trees or j < base or (trees[i] >> (j - base)) & 1)

        parts = []
        for b in range(blocks):
            keys = [j for j in range(position + 1) if (j // PIECE) % blocks == b and sees(j)]
            parts.append(_softmax_part(q, rows, keys, scale, softcap))
        top = max([m for m, _, _ in parts] + ([sink] if sink is not None else []))
        weights = [0.0 if m == -math.inf else l * math.exp(m - top) for m, l, _ in parts]
        total = sum(weights) + (math.exp(sink - top) if sink is not None else 0.0)
        dim = len(rows[-1][1])
        # o_b / l_b, weighted by w_b: the l_b cancels, as in the kernel's normalized partials
        out.append([sum(w * (o[d] / l) for w, (_, l, o) in zip(weights, parts) if w) / total for d in range(dim)])
    return out


def batch(storage, page_tokens, sequences, scale, blocks=1, **variants):
    """A ragged batch: sequences = [(table, kv_len, queries)]; returns the output rows in query order."""
    out = []
    for table, kv_len, queries in sequences:
        out.extend(attend(storage, page_tokens, table, kv_len, queries, scale, blocks, **variants))
    return out
