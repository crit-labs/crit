"""Opt-in observation of repeated passages in committed reasoning text."""
from collections import Counter, deque
from dataclasses import dataclass
import math
import unicodedata


@dataclass(frozen=True)
class LoopObservation:
    coverage: float
    window_words: int
    history_words: int
    triggered: bool


class ReasoningLoopGuard:
    """Serial, per-request state; feed accepted reasoning text, never draft text.

    Exact word tuples prevent hash collisions from creating repetition. Unicode
    letters, marks, numbers and underscores form words; punctuation is separate.
    Thresholds are workload policy, not a correctness guarantee or stop command.
    """
    def __init__(self, *, passage_words=12, repeats=3, window_words=2000,
                 history_words=30000, threshold=.25, max_word_chars=4096):
        values = (passage_words, repeats, window_words, history_words, max_word_chars)
        if (any(type(x) is not int or not 1 <= x <= 65536 for x in values)
                or passage_words > 64 or max_word_chars > 4096
                or repeats < 2 or not passage_words <= window_words <= history_words
                or type(threshold) not in (float, int) or not math.isfinite(threshold)
                or not 0 < threshold <= 1):
            raise ValueError("bounded consistent guard limits required")
        self.passage_words, self.repeats = passage_words, repeats
        self.window_words, self.history_words = window_words, history_words
        self.threshold, self.max_word_chars = threshold, max_word_chars
        self._words, self._partial = deque(maxlen=history_words), ""

    def feed(self, text, *, final=False):
        if type(text) is not str or len(text) > 1048576 or type(final) is not bool:
            raise ValueError("bounded decoded text and boolean final required")
        # Validate on local copies so overflow never half-mutates observations.
        words, partial = self._words.copy(), self._partial
        for char in text:
            word = char == "_" or unicodedata.category(char)[0] in "LMN"
            if word:
                if len(partial) == self.max_word_chars:
                    raise ValueError("reasoning word exceeds guard capacity")
                partial += char
            else:
                if partial:
                    words.append(partial.casefold())
                    partial = ""
                if not char.isspace():
                    words.append(char)
        if final and partial:
            words.append(partial.casefold())
            partial = ""
        self._words, self._partial = words, partial

    def observe(self):
        words = tuple(self._words)
        if len(words) < self.window_words:
            return LoopObservation(0, min(len(words), self.window_words), len(words), False)
        width = self.passage_words
        counts = Counter(words[i:i + width] for i in range(len(words) - width + 1))
        start = len(words) - self.window_words
        delta = [0] * (self.window_words + 1)
        for i in range(max(0, start - width + 1), len(words) - width + 1):
            if counts[words[i:i + width]] >= self.repeats:
                delta[max(i, start) - start] += 1
                delta[min(i + width, len(words)) - start] -= 1
        depth = covered = 0
        for change in delta[:-1]:
            depth += change
            covered += depth > 0
        coverage = covered / self.window_words
        return LoopObservation(coverage, self.window_words, len(words), coverage >= self.threshold)
