"""Stop strings over streamed text: the first completed stop wins, with minimal holdback (standard library only).

StopMatcher mirrors Core's C++ crit::core::StopMatcher. Python strings are matched by character, which is the same
as byte-exact matching over complete UTF-8 text; offsets here count characters. One Aho-Corasick automaton scans
every character exactly once, so the work is linear in the output length whatever the chunking.
"""
from .errors import TextError

MAX_STOPS = 16          # same limits as RequestOptions::stop_strings in Core
MAX_STOP_BYTES = 256


def _stop_list(stop_strings):
    if isinstance(stop_strings, (str, bytes, bytearray)):
        raise TextError("stop strings must be a list of strings, not one string")
    try:
        stops = tuple(stop_strings)
    except TypeError:
        raise TextError("stop strings must be a list of strings") from None
    if len(stops) > MAX_STOPS:
        raise TextError(f"at most {MAX_STOPS} stop strings are allowed, got {len(stops)}")
    for index, stop in enumerate(stops):
        if not isinstance(stop, str) or not stop:
            raise TextError(f"stop string {index} must be a non-empty string")
        try:
            size = len(stop.encode("utf-8"))
        except UnicodeEncodeError:
            raise TextError(f"stop string {index} is not valid Unicode text") from None
        if size > MAX_STOP_BYTES:
            raise TextError(f"stop string {index} has {size} UTF-8 bytes; the limit is {MAX_STOP_BYTES}")
    return stops


class StopMatcher:
    """Finds the first stop string to complete in text pushed in arbitrary chunks.

    Rules (identical to Core): the stop whose last character arrives first wins; among stops completing at the
    same character the one starting earliest (the longest) wins, then the lowest index. Text before the stop is
    released, the stop and everything after it is dropped. Without a match, only the longest suffix that is a
    proper prefix of some stop is held back; finish() releases it. Results never depend on how text is chunked.

    After a stop, push() and advance() return "" and ignore their input. After finish(), push() raises
    RuntimeError. `stopped` is the matched stop string (None until a match), `stop_index` its index and
    `stop_offset` the character offset where it starts in the whole text pushed.
    """

    def __init__(self, stop_strings=()):
        self.stops = _stop_list(stop_strings)
        goto, fail, depth, best = [{}], [0], [0], [None]
        for index, stop in enumerate(self.stops):
            node = 0
            for char in stop:
                child = goto[node].get(char)
                if child is None:
                    child = len(goto)
                    goto[node][char] = child
                    goto.append({})
                    fail.append(0)
                    depth.append(depth[node] + 1)
                    best.append(None)
                node = child
            if best[node] is None:
                best[node] = index
        queue = list(goto[0].values())
        for node in queue:
            for char, child in goto[node].items():
                link = fail[node]
                while link and char not in goto[link]:
                    link = fail[link]
                fail[child] = goto[link].get(char, 0)
                if best[child] is None:
                    best[child] = best[fail[child]]  # longest stop ending here, through the suffix links
                queue.append(child)
        self._goto, self._fail, self._depth, self._best = goto, fail, depth, best
        self._node = 0
        self._held = ""
        self._seen = 0
        self._finished = False
        self.released = 0
        self.stopped = None
        self.stop_index = None
        self.stop_offset = None

    @property
    def held(self):
        """Text received but not yet released: the longest suffix that could still begin a stop."""
        return self._held

    @property
    def finished(self):
        return self._finished

    def push(self, delta):
        """Feed newly generated text; return the text that can no longer be part of a stop."""
        if not isinstance(delta, str):
            raise TextError("stop matcher input must be a string")
        if self._finished:
            raise RuntimeError("stop matcher is finished")
        if self.stopped is not None or not delta:
            return ""
        goto, fail, best = self._goto, self._fail, self._best
        node = self._node
        for position, char in enumerate(delta):
            while node and char not in goto[node]:
                node = fail[node]
            node = goto[node].get(char, 0)
            index = best[node]
            if index is not None:
                stop = self.stops[index]
                end = self._seen + position + 1
                pending = self._held + delta[:position + 1]
                released = pending[:len(pending) - len(stop)]
                self.stopped, self.stop_index, self.stop_offset = stop, index, end - len(stop)
                self._seen, self._held, self._node = end, "", 0
                self.released += len(released)
                return released
        self._seen += len(delta)
        self._node = node
        pending = self._held + delta
        cut = len(pending) - self._depth[node]
        self._held = pending[cut:]
        self.released += cut
        return pending[:cut]

    def finish(self):
        """End of text: release the held suffix when no stop matched. The matcher is finished afterwards."""
        if self._finished:
            raise RuntimeError("stop matcher is finished")
        self._finished = True
        held, self._held = self._held, ""
        if self.stopped is not None:
            return ""
        self.released += len(held)
        return held

    def advance(self, cumulative, final=False):
        """Cumulative-text form used by crit serve: `cumulative` is all text so far and must extend the text given
        before; returns the newly releasable text and, with final=True, also releases the held suffix.

        Semantic change from crit serve's former StopScanner: that scanner cut at the earliest stop *start* found
        anywhere in the cumulative text and always held back max(len(stop)) - 1 characters, so its output depended
        on chunk sizes. This method applies the rules above: the first stop to complete wins and only a possible
        stop prefix is held (StopScanner(["STOP"]).advance("hello S") gave "hell"; this gives "hello ").
        """
        if not isinstance(cumulative, str):
            raise TextError("cumulative text must be a string")
        if not isinstance(final, bool):
            raise TextError("final must be True or False")
        if self._finished or self.stopped is not None:
            return ""
        if len(cumulative) < self._seen or cumulative[self._seen - len(self._held):self._seen] != self._held:
            raise TextError("cumulative text must extend the text already given")
        released = self.push(cumulative[self._seen:])
        if final and self.stopped is None:
            released += self.finish()
        return released
