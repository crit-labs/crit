"""Delimiter identity shared by reasoning and tool parsing; ordinary text never acquires token provenance."""
from dataclasses import dataclass

from .errors import TextError
from .stream import TextStream


@dataclass(frozen=True)
class TokenText:
    """The decoded spelling of one delimiter token, with its original id. Plain strings carry no token identity."""
    text: str
    token_id: int


def marker_ids(tokenizer, markers, delimiters="auto"):
    """Resolve each delimiter independently: a single special token requires that id; others match text.

    `text` explicitly retains spelling-based formats even when a tokenizer has special tokens for their markers.
    Without a tokenizer, auto preserves the standalone parsers' text API.
    """
    if delimiters not in ("auto", "text"):
        raise TextError("delimiters must be 'auto' or 'text'")
    if tokenizer is None or delimiters == "text":
        return {}
    result = {}
    for marker in markers:
        ids = tokenizer.encode(marker, add_special_tokens=False, split_special_tokens=False)
        if len(ids) == 1 and ids[0] in tokenizer.special_ids:
            result[marker] = ids[0]
    return result


def prompt_ends_with(tokenizer, ids, marker, token_ids):
    """Whether the prompt ends in this delimiter followed only by whitespace, using the parser's boundary rule."""
    token_id = token_ids.get(marker)
    if token_id is None:
        return tokenizer.decode(ids).rstrip().endswith(marker)
    for index in range(len(ids) - 1, -1, -1):
        if ids[index] == token_id:
            return not tokenizer.decode(ids[index + 1:]).strip()
    return False


class MarkerDecoder:
    """Incremental detokenization with TokenText events for the selected {id: spelling} delimiters.

    The same decoder consumes every id, including delimiters: UTF-8 tails and continuation spaces retain their
    context. A delimiter must end the text released by its own token; an incompatible decoder fails explicitly
    rather than attributing an ordinary substring to a control token. No text is searched for marker spellings.
    """
    def __init__(self, tokenizer, prompt_ids=(), *, markers, skip_special_tokens=True, stop_token_ids=(), stream=None):
        self._tokenizer, self._markers = tokenizer, dict(markers)
        self._stream = (stream if stream is not None else
                        TextStream(tokenizer, prompt_ids, skip_special_tokens=skip_special_tokens,
                                   stop_token_ids=stop_token_ids))
        self.closed = False

    def push(self, ids):
        if self.closed:
            raise RuntimeError("text decoder is closed")
        parts, pending = [], []
        try:
            if not self._markers:
                text = self._stream.push(ids).text
                return [text] if text else []
            for token in self._tokenizer.validate(ids):
                delta = self._stream.push((token,))
                text = delta.text
                marker = self._markers.get(token) if delta.finish_reason is None else None
                if marker is not None:
                    if not marker or not text.endswith(marker):
                        raise TextError(f"delimiter token {token} did not decode to its marker {marker!r}")
                    if text[:-len(marker)]:
                        pending.append(text[:-len(marker)])
                    if pending:
                        parts.append("".join(pending))
                        pending.clear()
                    parts.append(TokenText(marker, token))
                elif text:
                    pending.append(text)
        except BaseException:
            self.closed = True
            raise
        if pending:
            parts.append("".join(pending))
        return parts

    def finish(self):
        if self.closed:
            raise RuntimeError("text decoder is closed")
        self.closed = True
        text = self._stream.finish().text
        return [text] if text else []


class MarkerBuffer:
    """Parser text plus offsets of intact TokenText events; slices retain identity only for whole tokens."""
    def __init__(self, text="", tokens=()):
        self.text, self.tokens = text, list(tokens)

    def append(self, value):
        if isinstance(value, TokenText):
            self.tokens.append((len(self.text), value))
            value = value.text
        self.text += value

    def slice(self, start=0, end=None):
        end = len(self.text) if end is None else end
        return MarkerBuffer(self.text[start:end], ((offset - start, value) for offset, value in self.tokens
                                                  if start <= offset and offset + len(value.text) <= end))

    def lstrip(self):
        return self.slice(len(self.text) - len(self.text.lstrip()))

    def find(self, marker, token_ids):
        token_id = token_ids.get(marker)
        if token_id is None:
            return self.text.find(marker)
        return next((offset for offset, value in self.tokens
                     if value.token_id == token_id and value.text == marker), -1)

    def held(self, marker, token_ids):
        if marker not in token_ids:
            for size in range(min(len(self.text), len(marker) - 1), 0, -1):
                if marker.startswith(self.text[-size:]):
                    return size
        return 0

    def parts(self):
        start = 0
        for offset, value in self.tokens:
            if start < offset:
                yield self.text[start:offset]
            yield value
            start = offset + len(value.text)
        if start < len(self.text):
            yield self.text[start:]
