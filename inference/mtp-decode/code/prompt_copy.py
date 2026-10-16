"""Bounded exact-window lookup over accepted tokens; proposals still need verification."""
from collections import deque
from dataclasses import dataclass


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _tokens(values):
    result = tuple(values)
    if any(type(t) is not int or not 0 <= t < 1 << 31 for t in result):
        raise ValueError("tokens must be nonnegative signed-32-bit IDs")
    return result


class PromptCopyIndex:
    """Serial callers; bounded context and recent occurrences per exact token window.

    Tuple dictionary keys compare actual tokens even when their hashes collide.
    Evicting older occurrences can reduce proposal quality, never bypass target checks.
    Rewind rebuilds the index so previously evicted occurrences can become visible again.
    """

    def __init__(self, prompt=(), *, window=4, history_window=8, max_draft=8,
                 max_matches=64, max_tokens=32768):
        self.window = _integer(window, "window", 1)
        self.history_window = _integer(history_window, "history_window", self.window)
        self.max_draft = _integer(max_draft, "max_draft", 1)
        self.max_matches = _integer(max_matches, "max_matches", 1)
        self.max_tokens = _integer(max_tokens, "max_tokens", 1)
        self._tokens = []
        self._positions = {}
        self.append(prompt)
        self._prompt_end = len(self._tokens)

    @property
    def tokens(self):
        return tuple(self._tokens)

    def __len__(self):
        return len(self._tokens)

    def _add_window(self, end):
        start = end - self.window
        if start < 0:
            return
        key = tuple(self._tokens[start:end])
        bucket = self._positions.get(key)
        if bucket is None:
            bucket = self._positions[key] = deque(maxlen=self.max_matches)
        bucket.append(start)

    def append(self, accepted):
        accepted = _tokens(accepted)
        if len(self._tokens) + len(accepted) > self.max_tokens:
            raise ValueError("accepted context exceeds index capacity")
        old = len(self._tokens)
        self._tokens.extend(accepted)
        for end in range(old, len(self._tokens)):
            self._add_window(end)

    def rewind(self, length):
        _integer(length, "length")
        if length > len(self._tokens):
            raise ValueError("rewind cannot extend accepted context")
        if length == len(self._tokens):
            return
        del self._tokens[length:]
        self._prompt_end = min(self._prompt_end, length)
        self._positions.clear()
        for end in range(self.window, length):
            self._add_window(end)

    def propose(self, remaining=None, *, vocabulary=None, stop_tokens=()):
        limit = self.max_draft if remaining is None else min(
            self.max_draft, _integer(remaining, "remaining"))
        if vocabulary is not None:
            _integer(vocabulary, "vocabulary", 1)
        stops = frozenset(_tokens(stop_tokens))
        length, width = len(self._tokens), self.window
        if not limit or length <= width:
            return ()
        suffix = tuple(self._tokens[-width:])
        best = ()
        for start in reversed(self._positions.get(suffix, ())):
            end = start + width
            if end >= length:  # The suffix itself has no known continuation.
                continue
            if start >= self._prompt_end:
                required = self.history_window
                if end < required or length < required or (
                    self._tokens[end-required:end] != self._tokens[length-required:length]
                ):
                    continue
            candidate = []
            for token in self._tokens[end:min(length, end + limit)]:
                if vocabulary is not None and token >= vocabulary:
                    break
                candidate.append(token)
                if token in stops:
                    break
            if len(candidate) > len(best):
                best = tuple(candidate)
            if len(best) == limit:
                break
        return best


@dataclass(frozen=True)
class CopyBatch:
    proposals: tuple[tuple[int, ...], ...]
    width: int
    verification_rows: int


def propose_batch(indices, row_budget, *, bucket=1, remaining=None,
                  vocabulary=None, stop_tokens=()):
    """Charge every row for the shared draft width plus one target prediction.

    Proposals stay ragged: the engine must mask padding and reconcile each row's
    actual prefix. Bucketing does not invent accepted tokens or target predictions.
    """
    indices = tuple(indices)
    _integer(row_budget, "row_budget")
    _integer(bucket, "bucket", 1)
    if not indices:
        return CopyBatch((), 0, 0)
    if row_budget < len(indices):
        raise ValueError("budget cannot cover one target prediction per request")
    stop_tokens = _tokens(stop_tokens)
    remaining = (None,) * len(indices) if remaining is None else tuple(remaining)
    if len(remaining) != len(indices):
        raise ValueError("one remaining output budget per request required")
    limit = row_budget // len(indices) - 1
    proposals = tuple(index.propose(limit if r is None else min(limit, _integer(r, "remaining")),
                                    vocabulary=vocabulary, stop_tokens=stop_tokens)
                      for index, r in zip(indices, remaining))
    width = max(map(len, proposals))
    width = min(limit, ((width + bucket - 1) // bucket) * bucket)
    return CopyBatch(proposals, width, len(indices) * (width + 1))
