"""Incremental detokenization with prefix and read offsets: text is released once it is stable and never retracted.

Mirrors Core's TextDecoder contract: push()/finish(), ids validated, closed after finish() or any error. Decoding a
token alone is lossy (byte-level splits give U+FFFD, SentencePiece decoders drop the leading space of a
continuation), so every step decodes a short window that starts a few tokens back and releases only what the new
tokens add. tokenizers.decoders.DecodeStream is deliberately not used: it re-emits its prefill when that ends
mid-character and it cannot flush an incomplete tail.
"""
from .errors import TextError

REPLACEMENT = "�"
HOLD_LIMIT = 16   # ids without a clean character boundary after which the window is cut to its last ids
MAX_PARTIAL = 3   # ids that can carry the bytes of one incomplete UTF-8 character (at most 3 bytes)
MIN_CONTEXT = 2   # released ids the window keeps as context, at least


def _shared(first, second):
    """Length of the common prefix of two strings."""
    limit = min(len(first), len(second))
    index = 0
    while index < limit and first[index] == second[index]:
        index += 1
    return index


class IncrementalDetokenizer:
    """Streams the text of generated ids.

    Each step releases what the window now decodes to beyond the text already released, except a trailing run of
    U+FFFD: that run may be a character whose remaining bytes have not arrived (byte-level merges that straddle
    characters, SentencePiece byte fallback), so it is held until later text follows it or finish() flushes it.
    The window is seeded with the last `context_tokens` prompt ids, moved back to an id that starts a character.
    Prompt text is never released: when the prompt ends inside a character, the output starts with the character
    it completes (the prompt's complete characters, such as the "g" of a byte-level token b"g\\xc3", stay prompt
    text), and U+FFFD that the output leaves in place (a literal U+FFFD, invalid bytes) stays prompt text too.

    For the tokenizer's own encodings, the concatenated push() and finish() results equal decode(prompt + ids)
    without the prompt's complete characters, so decode(ids) without a prompt. Memory is bounded: after HOLD_LIMIT
    ids without a clean boundary the window is cut to its last ids, keeping those that may carry an incomplete
    character, and a long U+FFFD run is released except its last MAX_PARTIAL characters.

    `anomalies` counts the times text already released or accounted for changed (a SentencePiece byte run that a
    later invalid or incomplete byte turns into U+FFFD as a whole): the pending ids are then decoded on their own,
    without repeating what was released from them; text is never retracted. With skip_special_tokens (the default)
    added tokens marked special are consumed without producing text.
    """

    def __init__(self, tokenizer, prompt_ids=(), *, skip_special_tokens=True, context_tokens=5):
        if not isinstance(skip_special_tokens, bool):
            raise TextError("skip_special_tokens must be True or False")
        if isinstance(context_tokens, bool) or not isinstance(context_tokens, int) or context_tokens < 0:
            raise TextError("context_tokens must be a non-negative integer")
        prompt = tokenizer.validate(prompt_ids)
        self._tokenizer = tokenizer
        self._skip = skip_special_tokens
        self._ignored = tokenizer.special_ids if skip_special_tokens else frozenset()
        prompt = [token for token in prompt if token not in self._ignored] if self._ignored else prompt
        self._keep = max(context_tokens, MIN_CONTEXT)
        self._window = self._seed(prompt, context_tokens)
        self._read = len(self._window)   # window[:read] is context, window[read:] the ids since the last anchor
        self._clean = []                 # window offsets where the decode ended on a clean boundary
        decoded = self._decode(self._window)
        self._head = decoded.rstrip(REPLACEMENT)   # leading text of the window decode already released or prompt
        self._base = self._head                     # head when window[read:] began
        self._prompt = decoded if self._head != decoded else None   # a prompt decode whose final U+FFFD run is open
        self._prompt_heads = self._complete_heads() if self._prompt is not None else ()
        self._outputs = 0
        self.closed = False
        self.anomalies = 0
        self.emitted = 0

    def _decode(self, ids):
        return self._tokenizer.backend.decode(ids, skip_special_tokens=self._skip)

    def _seed(self, prompt, context_tokens):
        """The prompt's last `context_tokens` ids, moved back by up to MAX_PARTIAL ids to one that starts a character
        (the decode does not start with U+FFFD once up to MAX_PARTIAL final ids, which may hold an incomplete
        character, are set aside). When no candidate does (byte-level merges that straddle characters, a byte run
        that ends in a literal U+FFFD), the start whose decode holds the fewest U+FFFD plus ids set aside, the latest
        of equals: a start inside a byte-fallback character turns the run into one U+FFFD per byte. The U+FFFD the
        chosen decode starts with stays prompt text."""
        end = len(prompt)
        if not context_tokens or not end:
            return []
        plain = max(end - context_tokens, 0)
        best, best_score = plain, None
        for start in range(plain, max(plain - MAX_PARTIAL, 0) - 1, -1):
            if start == 0:
                return list(prompt)
            for partial in range(min(MAX_PARTIAL, end - start - 1) + 1):
                text = self._decode(prompt[start:end - partial])
                if not text:
                    continue
                if not text.startswith(REPLACEMENT):
                    return prompt[start:]
                score = text.count(REPLACEMENT) + partial
                if best_score is None or score < best_score:
                    best, best_score = start, score
        return prompt[best:]

    def _complete_heads(self):
        """Decodes of the prompt window without 1 to MAX_PARTIAL final ids, with that count, longest first: the
        prompt's complete characters where an incomplete one turns a whole SentencePiece byte run into U+FFFD."""
        window = self._window
        heads = [(self._decode(window[:len(window) - partial]), partial)
                 for partial in range(1, min(MAX_PARTIAL, len(window)) + 1)]
        return sorted(heads, key=lambda head: (-len(head[0]), head[0], head[1]))

    def _resolve(self, full, force):
        """Once the output shows how the prompt's final U+FFFD run continues (text follows it, or the decode ends
        cleanly), or when `force` is set, account for the longest prompt text the window decode starts with: the part
        of the prompt decode it keeps (U+FFFD left in place stays prompt text), or a longer decode of the prompt's
        complete characters. The character the output completes is then released with the output. A decode of the
        prompt without its last ids is exact when the rest of the window decodes on its own to the rest of `full`; one
        that ends with U+FFFD (a literal U+FFFD before an incomplete character) counts only when exact. The prompt
        decode's final U+FFFD run is a byte run turned into U+FFFD as a whole by the prompt's incomplete character when
        an exact decode shows that the prompt decode is not that decode plus the decode of the ids after it, or, when
        the window decode does not keep all of the prompt decode, when those ids decode to one U+FFFD each, alone and
        together (one per byte). Then the U+FFFD the kept part ends with are not prompt text, except those an exact
        decode shows. Otherwise they are complete characters of the prompt (a literal U+FFFD or invalid bytes before its
        incomplete character), also when no exact decode shows them (a merge that straddles a literal U+FFFD and the
        next character)."""
        if not force and len(full.rstrip(REPLACEMENT)) <= len(self._head) and full.endswith(REPLACEMENT):
            return False
        window, read = self._window, self._read

        def exact(text, partial):
            return full.startswith(text) and text + self._decode(window[read - partial:]) == full

        def per_byte(tail, partial):
            return tail == REPLACEMENT * partial and all(self._decode([token]) == REPLACEMENT
                                                         for token in window[read - partial:read])

        kept = _shared(full, self._prompt)
        whole = kept == len(self._prompt)
        for text, partial in self._prompt_heads:
            if exact(text, partial):
                tail = self._decode(window[read - partial:read])
                if text + tail != self._prompt or (not whole and per_byte(tail, partial)):
                    kept = len(self._prompt[:kept].rstrip(REPLACEMENT))
                    break
        head = self._prompt[:kept] if kept >= len(self._head) else self._head
        for text, partial in self._prompt_heads:
            if len(text) > len(head) and full.startswith(text) and (not text.endswith(REPLACEMENT)
                                                                     or exact(text, partial)):
                head = text
                break
        self._head = self._base = head
        self._prompt = None
        return True

    def push(self, ids):
        """Feed generated ids; return the newly stable text ('' when none)."""
        if self.closed:
            raise RuntimeError("text decoder is closed")
        try:
            pieces = []
            for token in self._tokenizer.validate(ids):
                if token in self._ignored:
                    continue
                self._window.append(token)
                self._outputs += 1
                pieces.append(self._step(self._decode(self._window)))
        except BaseException:
            self.closed = True
            raise
        text = "".join(pieces)
        self.emitted += len(text)
        return text

    def _step(self, full):
        pending = len(self._window) - self._read
        if self._prompt is not None and not self._resolve(full, pending >= HOLD_LIMIT):
            return ""
        head = self._head
        if full.startswith(head):
            tail = full[len(head):]
            stable = tail.rstrip(REPLACEMENT)
            self._head = head + stable
            if len(stable) == len(tail):
                self._anchor()
                return stable
            if pending >= HOLD_LIMIT:
                return stable + self._bound(full)
            return stable
        if full.endswith(REPLACEMENT) and pending < HOLD_LIMIT:
            return ""  # a SentencePiece byte run is still open: the text released from it may come back
        return self._diverge(full)

    def _anchor(self):
        """The window decode ends cleanly and is fully accounted for: its ids become context. Context beyond `keep`
        ids is dropped from the latest clean boundary that leaves at least `keep` ids."""
        window = self._window
        self._read = len(window)
        self._base = self._head
        self._clean.append(len(window))
        cut = 0
        for offset in self._clean:
            if offset > len(window) - self._keep:
                break
            cut = offset
        if cut >= self._keep:
            del window[:cut]
            self._clean = [offset - cut for offset in self._clean if offset > cut]
            self._read -= cut
            self._head = self._base = self._decode(window)

    def _bound(self, full):
        """HOLD_LIMIT ids arrived without a clean boundary (merges that keep straddling characters, invalid bytes).
        Release the trailing U+FFFD run except its last MAX_PARTIAL characters, which may still be one incomplete
        character, and cut the window to its last ids, keeping the ids that may carry that character."""
        held = full[len(self._head):]
        released = ""
        if len(held) > MAX_PARTIAL:
            released = held[:len(held) - MAX_PARTIAL]
            held = held[len(released):]
            self._head += released
        window = self._window
        for keep in range(self._keep + MAX_PARTIAL, len(window)):
            text = self._decode(window[len(window) - keep:])
            if text.endswith(held):
                del window[:len(window) - keep]
                self._head = text[:len(text) - len(held)]
                break
        else:
            if len(window) > self._keep + MAX_PARTIAL:  # no shorter window keeps the held run: release it
                self.anomalies += 1
                released += held
                del window[:len(window) - self._keep]
                self._head = self._decode(window)
        self._clean = []
        self._read = len(window)
        self._base = self._head
        return released

    def _diverge(self, full):
        """Text already released or accounted for changed: release what the ids since the last anchor decode to on
        their own, without repeating text already released from them, and continue from the new decode."""
        self.anomalies += 1
        alone = self._decode(self._window[self._read:])
        text = alone[_shared(alone, self._head[len(self._base):]):]
        self._head = full
        self._anchor()
        return text

    def finish(self):
        """Flush the held tail (it may end with U+FFFD) and close the decoder."""
        if self.closed:
            raise RuntimeError("text decoder is closed")
        self.closed = True
        if not self._outputs:
            return ""
        full = self._decode(self._window)
        if self._prompt is not None:
            self._resolve(full, True)
        if full.startswith(self._head):
            text = full[len(self._head):]
        else:
            self.anomalies += 1
            alone = self._decode(self._window[self._read:])
            text = alone[_shared(alone, self._head[len(self._base):]):]
        self.emitted += len(text)
        return text
