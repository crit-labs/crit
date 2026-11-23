"""Encode a prompt that extends a recent one from near the seam only, not from its start.

A multi-turn chat prompt is usually the previous turn's prompt plus a short tail, and encoding the whole rendered
prompt again costs time in proportion to its length. PromptTokenCache keeps the ids of recent prompts. For a prompt
that starts with a cached one it backs off `overlap` tokens from the cached prompt's end, encodes the text from
there, and splices that onto the cached ids before it, but only when the first `resync` new ids equal the cached ids
over the same span: the tokenizer has come back into step with the cached prompt there. A tokenizer that encodes
text differently at its start (one that prefixes the first word with a space marker) can miss at one token and meet
at an earlier one, so up to `attempts` back-off points are tried, one token apart. Otherwise it encodes the whole
prompt. Standard library only, over crit_core.text's Tokenizer.
"""
from collections import OrderedDict
from dataclasses import dataclass
import threading

PROBE_CHARS = 256


@dataclass(frozen=True)
class _Entry:
    text: str
    ids: tuple
    splits: tuple      # (token, character it starts at): where a prompt extending `text` may be encoded from, best first
    probe: str         # the last PROBE_CHARS characters of `text`


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class PromptTokenCache:
    """Ids of recent prompts, so encode(text) re-encodes only the tail of a prompt that extends one of them.

    encode(text) returns tokenizer.encode(text, add_special_tokens=False, split_special_tokens=False), the call
    ModelText.encode_chat makes. Prompts shorter than `min_chars` are encoded in full and not kept. A prompt that
    extends a kept one replaces it, so a conversation holds one entry however many turns it has; eviction is least
    recently used, bounded by `entries` and by `max_chars` characters of kept text (the ids take about as much
    memory again). Entries are found among those sharing the prompt's first `min_chars` characters. Thread-safe.

    The splice keeps every id before the back-off point and asks the tokenizer to agree on `resync` ids after it, so
    it can differ from a full encode only where a token before the back-off point depends on text more than
    `overlap` tokens after it. `hits`, `misses` and `resync_failures` count prompts spliced, encoded in full, and
    encoded in full because the tokenizer did not come back into step.
    """

    def __init__(self, tokenizer, *, entries=64, max_chars=64 << 20, min_chars=4096, overlap=64, resync=32,
                 attempts=16):
        self.tokenizer = tokenizer
        self.entries = _positive(entries, "entries")
        self.max_chars = _positive(max_chars, "max_chars")
        self.min_chars = _positive(min_chars, "min_chars")
        self.overlap = _positive(overlap, "overlap")
        self.resync = _positive(resync, "resync")
        self.attempts = _positive(attempts, "attempts")
        if self.resync > self.overlap:
            raise ValueError("resync must not exceed overlap")
        self._lock = threading.Lock()
        self._kept = OrderedDict()   # id -> _Entry, least recently used first
        self._buckets = {}           # hash of the first min_chars characters -> {id: _Entry}
        self._chars = 0
        self._next = 0
        self.hits = self.misses = self.resync_failures = 0

    def __len__(self):
        """Prompts kept."""
        with self._lock:
            return len(self._kept)

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("text to encode must be a string")
        if len(text) < self.min_chars:
            return self._encode(text)[0]
        found = self._lookup(text)
        ids = None
        if found is not None:
            entry = found[1]
            for token, char in entry.splits:
                tail, offsets = self._encode(text[char:])
                cached = entry.ids[token:]
                span = min(self.resync, len(tail), len(cached))
                if span and tuple(tail[:span]) == cached[:span]:
                    ids = list(entry.ids[:token]) + tail
                    base_token, base_char = token, char
                    break
        if ids is None:
            tail, offsets = self._encode(text)
            ids, base_token, base_char = tail, 0, 0
        last = len(ids) - self.overlap
        first = max(last - self.attempts + 1, base_token, 1)
        splits = tuple((token, base_char + offsets[token - base_token][0]) for token in range(last, first - 1, -1))
        with self._lock:
            if found is None:
                self.misses += 1
            elif base_token == 0:
                self.misses += 1
                self.resync_failures += 1
            else:
                self.hits += 1
            if found is not None:
                self._remove(found[0])
            if splits:
                self._insert(text, ids, splits)
        return ids

    def _encode(self, text):
        return self.tokenizer.encode_offsets(text, add_special_tokens=False, split_special_tokens=False)

    def _bucket(self, text):
        return hash(text[:self.min_chars])

    def _lookup(self, text):
        """The longest kept prompt that `text` starts with, marked as just used; None when there is none."""
        with self._lock:
            candidates = list(self._buckets.get(self._bucket(text), {}).items())
        best = None
        for key, entry in candidates:   # entries are immutable: compare outside the lock
            size = len(entry.text)
            if (size <= len(text) and (best is None or size > len(best[1].text))
                    and text[size - len(entry.probe):size] == entry.probe and text.startswith(entry.text)):
                best = (key, entry)
        if best is not None:
            with self._lock:
                if best[0] in self._kept:
                    self._kept.move_to_end(best[0])
        return best

    def _insert(self, text, ids, splits):
        bucket = self._buckets.setdefault(self._bucket(text), {})
        if any(entry.text == text for entry in bucket.values()):
            return
        key, self._next = self._next, self._next + 1
        entry = _Entry(text, tuple(ids), splits, text[-PROBE_CHARS:])
        self._kept[key] = bucket[key] = entry
        self._chars += len(text)
        while self._kept and (len(self._kept) > self.entries or self._chars > self.max_chars):
            self._remove(next(iter(self._kept)))

    def _remove(self, key):
        entry = self._kept.pop(key, None)
        if entry is None:
            return
        self._chars -= len(entry.text)
        bucket = self._buckets[self._bucket(entry.text)]
        bucket.pop(key, None)
        if not bucket:
            del self._buckets[self._bucket(entry.text)]
