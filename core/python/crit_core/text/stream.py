"""One reply's text: stop ids, incremental detokenization and stop strings, fed one token at a time."""
from dataclasses import dataclass

from .detokenize import IncrementalDetokenizer
from .errors import TextError
from .stops import StopMatcher


@dataclass(frozen=True)
class TextDelta:
    """Text released by one push() or finish().

    finish_reason is "stop" once a stop id or stop string ended the reply, "length" from finish() when nothing
    stopped it, otherwise None. `stop` is the matched stop string or stop id. `consumed` counts the ids of this
    push() that belong to the reply: all of them, or up to and including the one that completed a stop.
    """
    text: str
    finish_reason: str | None = None
    stop: str | int | None = None
    consumed: int = 0


class TextStream:
    """Turns generated ids into reply text, applying stop ids and stop strings with Core's rules.

    Ids are processed one at a time. A stop id is consumed and never decoded: the text before it is flushed
    and the reply ends. Stop strings are matched across token boundaries on the text the detokenizer releases,
    which it releases as soon as it is stable, so the token whose bytes complete a stop's last character is the
    last consumed token (like a stop id, it stays in the reply's tokens). The stop string and anything after it
    are dropped from the text, unless include_stop_str is True, which appends the matched stop string. Text the
    detokenizer still holds (a trailing U+FFFD) is matched when it is released; when the flush at a stop id
    completes a stop string, the earlier ids completed it and the stop id is not consumed. After a stop, push()
    consumes nothing. finish() flushes held text and closes the stream; any error closes it.
    """

    def __init__(self, tokenizer, prompt_ids=(), *, stop_strings=(), stop_token_ids=(), skip_special_tokens=True,
                 include_stop_str=False, context_tokens=5):
        if not isinstance(include_stop_str, bool):
            raise TextError("include_stop_str must be True or False")
        self._tokenizer = tokenizer
        self._stop_ids = frozenset(tokenizer.validate(stop_token_ids))
        self._matcher = StopMatcher(stop_strings)
        self._decoder = IncrementalDetokenizer(tokenizer, prompt_ids, skip_special_tokens=skip_special_tokens,
                                               context_tokens=context_tokens)
        self._include = include_stop_str
        self.stop = None
        self.consumed = 0
        self.closed = False

    def _stopped(self, released, consumed):
        if self._include and isinstance(self.stop, str):
            released += self.stop
        self.consumed += consumed
        return TextDelta(released, "stop", self.stop, consumed)

    def push(self, ids):
        """Feed generated ids; return the released text and how many of these ids the reply consumed."""
        if self.closed:
            raise RuntimeError("text stream is closed")
        try:
            ids = self._tokenizer.validate(ids)
            if self.stop is not None:
                return TextDelta("", "stop", self.stop, 0)
            pieces = []
            for index, token in enumerate(ids):
                if token in self._stop_ids:
                    pieces.append(self._matcher.push(self._decoder.finish()))
                    if self._matcher.stopped is not None:  # completed by held text of the ids before the stop id
                        self.stop = self._matcher.stopped
                        return self._stopped("".join(pieces), index)
                    pieces.append(self._matcher.finish())
                    self.stop = token
                    return self._stopped("".join(pieces), index + 1)
                piece = self._decoder.push((token,))
                if piece:
                    pieces.append(self._matcher.push(piece))
                    if self._matcher.stopped is not None:
                        self.stop = self._matcher.stopped
                        return self._stopped("".join(pieces), index + 1)
            self.consumed += len(ids)
            return TextDelta("".join(pieces), None, None, len(ids))
        except BaseException:
            self.closed = True
            raise

    def finish(self):
        """End of generation: flush held text (it may still complete a stop string) and close the stream."""
        if self.closed:
            raise RuntimeError("text stream is closed")
        self.closed = True
        if self.stop is not None:
            return TextDelta("", "stop", self.stop, 0)
        released = self._matcher.push(self._decoder.finish())
        if self._matcher.stopped is not None:
            self.stop = self._matcher.stopped
            return self._stopped(released, 0)
        return TextDelta(released + self._matcher.finish(), "length", None, 0)
