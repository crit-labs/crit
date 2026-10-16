"""The byte table a constraint needs: the exact decoded bytes of every token id (standard library only).

`byte_vocabulary(tokenizer, eos)` reads a `crit_core.text.Tokenizer`'s tokenizer.json decoder and returns a
`ByteVocabulary` whose pieces are what the detokenizer appends to the output for each id, the input of the native
`crit::core::ByteVocabulary` and of `make_compile_request`/`PreparationService.request`.
"""
from dataclasses import dataclass
import hashlib
import json
import re

from crit_core.text.errors import TextError

# GPT-2's byte <-> printable character table, used by byte-level BPE vocabularies.
_PRINTABLE = [*range(ord("!"), ord("~") + 1), *range(ord("\xa1"), ord("\xac") + 1), *range(ord("\xae"), ord("\xff") + 1)]
_BYTE_LEVEL = {}
for _byte in range(256):
    _BYTE_LEVEL[chr(_byte) if _byte in _PRINTABLE else chr(256 + sum(b not in _PRINTABLE for b in range(_byte)))] = _byte
_FALLBACK = re.compile(r"<0x([0-9A-Fa-f]{2})>")


@dataclass(frozen=True)
class ByteVocabulary:
    """`pieces[id]` is the bytes id adds to the output; `pieces[eos]` and every forbidden id (special and control
    tokens) are b"". `identity` is a sha256 over the pieces and EOS, the tokenizer identity of compile requests."""
    pieces: tuple
    eos: int
    identity: str

    @property
    def request_fields(self):
        """Keyword arguments for make_compile_request, CompilerCache.compile and PreparationService.request."""
        return {"tokenizer_identity": self.identity, "pieces": self.pieces, "eos": self.eos}


def _identity(pieces, eos):
    digest = hashlib.sha256(b"crit-byte-vocabulary-v1\0%d\0%d\0" % (len(pieces), eos))
    for piece in pieces:
        digest.update(len(piece).to_bytes(4, "little"))
        digest.update(piece)
    return "sha256:" + digest.hexdigest()


def from_pieces(pieces, eos):
    """A ByteVocabulary from explicit pieces (bytes) and an EOS id whose piece is empty."""
    pieces = tuple(pieces)
    if (not pieces or any(type(p) is not bytes for p in pieces) or type(eos) is not int or not 0 <= eos < len(pieces)
            or pieces[eos]):
        raise TextError("a byte vocabulary needs bytes pieces and an EOS id with an empty piece")
    return ByteVocabulary(pieces, eos, _identity(pieces, eos))


def _decoder_steps(decoder):
    """The per-token rewrite a tokenizer.json decoder applies: ('bytelevel',) or replacements plus byte fallback.

    Decoders that only act at the start of the whole text (Strip of a leading space, Metaspace's prepended space) do
    not change a continuation's bytes and are accepted; anything that joins or rewrites tokens otherwise is refused.
    """
    if decoder is None:
        raise TextError("constraints need a tokenizer.json decoder; this tokenizer has none")
    kind = decoder.get("type")
    if kind == "ByteLevel":
        return {"byte_level": True, "replace": [], "fallback": False}
    if kind == "Metaspace":
        return {"byte_level": False, "replace": [(decoder.get("replacement", "▁"), " ")], "fallback": False}
    if kind == "Sequence":
        steps = {"byte_level": False, "replace": [], "fallback": False}
        for part in decoder.get("decoders") or []:
            sub = part.get("type")
            if sub == "Replace" and isinstance(part.get("pattern"), dict) and set(part["pattern"]) == {"String"}:
                steps["replace"].append((part["pattern"]["String"], part.get("content", "")))
            elif sub == "ByteFallback":
                steps["fallback"] = True
            elif sub == "Fuse" or (sub == "Strip" and part.get("stop", 0) == 0):
                continue  # Fuse joins tokens unchanged; a start-only Strip affects only the start of the text.
            elif sub == "Metaspace":
                steps["replace"].append((part.get("replacement", "▁"), " "))
            elif sub == "ByteLevel" and len(decoder["decoders"]) == 1:
                return {"byte_level": True, "replace": [], "fallback": False}
            else:
                raise TextError(f"constraints do not support the tokenizer.json decoder step {sub!r}")
        return steps
    raise TextError(f"constraints do not support the tokenizer.json decoder {kind!r}")


def byte_vocabulary(tokenizer, eos=None):
    """The ByteVocabulary of a crit_core.text.Tokenizer. `eos` (default: the tokenizer's eos_token_id) is the one
    token that ends constrained output; every other special added token gets an empty piece and is never allowed.
    Non-special added tokens (e.g. <think>) are their content. Raises TextError for an unknown EOS or a decoder
    whose per-token bytes cannot be derived exactly."""
    eos = tokenizer.eos_token_id if eos is None else eos
    if type(eos) is not int or not 0 <= eos < tokenizer.vocab_size:
        raise TextError("byte_vocabulary needs an EOS token id inside the vocabulary")
    steps = _decoder_steps(json.loads(tokenizer.backend.to_str()).get("decoder"))
    pieces = []
    for token in range(tokenizer.vocab_size):
        added = tokenizer.added_tokens.get(token)
        if token == eos or (added is not None and added.special):
            pieces.append(b"")
            continue
        if added is not None:
            pieces.append(added.content.encode("utf-8"))
            continue
        text = tokenizer.id_to_token(token)
        if steps["byte_level"]:
            try:
                pieces.append(bytes(_BYTE_LEVEL[char] for char in text))
            except KeyError:
                raise TextError(f"token {token} is not a byte-level BPE spelling") from None
            continue
        fallback = _FALLBACK.fullmatch(text) if steps["fallback"] else None
        if fallback:
            pieces.append(bytes([int(fallback[1], 16)]))
            continue
        for old, new in steps["replace"]:
            text = text.replace(old, new)
        pieces.append(text.encode("utf-8"))
    return from_pieces(pieces, eos)
