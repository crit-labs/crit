"""Detect a reply that has settled into repeating one span of tokens, so the engine can stop it before its cap.

Standard library only. A period p is a candidate when the last `k` tokens occurred p tokens earlier; its run counts
the consecutive positions, ending at the newest token, whose token equals the one p positions before it. A run of at
least max(min_tokens, min_repeats x p) tokens is a loop: the same span, exactly, min_repeats times and over at least
min_tokens tokens, with nothing else in between. Near-repetition (a counter that changes, an edited copy) is not a
loop here; the defaults keep a long list of identical items legitimate until it runs past min_tokens.
"""


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class LoopDetector:
    """Fed a reply's tokens in order (push), reports whether its tail repeats one span exactly.

    `period` and `run` describe the loop once found (`looped`): the span's length in tokens and how many trailing
    tokens repeat it. At most `candidates` periods are followed at once, the longest runs kept. O(k + candidates)
    per token; memory grows with the reply (its tokens and one entry per distinct k-gram)."""

    def __init__(self, *, min_repeats=8, min_tokens=1024, max_period=8192, k=8, candidates=8):
        self.min_repeats = _positive(min_repeats, "min_repeats")
        self.min_tokens = _positive(min_tokens, "min_tokens")
        self.max_period = _positive(max_period, "max_period")
        self.k = _positive(k, "k")
        self.candidates = _positive(candidates, "candidates")
        self._tokens = []
        self._seen = {}     # k-gram -> the position of its last token, last time it ended there
        self._runs = {}     # period -> run length ending at the newest token
        self.looped = False
        self.period = self.run = 0

    def push(self, tokens):
        """Add tokens; True once the reply loops (and from then on)."""
        for token in tokens:
            if self.looped:
                break
            self._push(token)
        return self.looped

    def _push(self, token):
        tokens, t = self._tokens, len(self._tokens)
        tokens.append(token)
        runs = {p: run + 1 for p, run in self._runs.items() if tokens[t - p] == token}
        if t + 1 >= self.k:
            gram = tuple(tokens[t + 1 - self.k:])
            last = self._seen.get(gram)
            if last is not None and 0 < t - last <= self.max_period and t - last not in runs:
                runs[t - last] = self.k
            self._seen[gram] = t
        if len(runs) > self.candidates:
            runs = dict(sorted(runs.items(), key=lambda item: -item[1])[:self.candidates])
        self._runs = runs
        for p, run in runs.items():
            if run >= max(self.min_tokens, self.min_repeats * p):
                self.looped, self.period, self.run = True, p, run
                return
