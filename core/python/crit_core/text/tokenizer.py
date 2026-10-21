"""Tokenizer over a Hugging Face tokenizer.json: explicit special-token handling, validated ids, never-mutated backend.

Differences from calling `tokenizers` directly: `encode` requires `add_special_tokens` as a keyword; `decode`
defaults to `skip_special_tokens=False` (as transformers does; `tokenizers` defaults to True); out-of-range ids
raise TokenError instead of decoding to "" or None; truncation and padding stored in tokenizer.json are switched
off at load, before the backend is shared, so text is never cut silently.
"""
from dataclasses import dataclass
import json
import operator
from pathlib import Path
import threading
from types import MappingProxyType

import tokenizers

from .errors import MAX_FILE_BYTES, ModelTextError, TextError, TokenError, model_files, read_bytes, read_json_object

SPECIAL_TOKEN_NAMES = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")


@dataclass(frozen=True)
class AddedToken:
    id: int
    content: str
    special: bool


def _token_string(value, source, name):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    raise ModelTextError(f"{source}: {name} must be a string, an object with a content string, or null")


def special_tokens_from(config, files):
    """Special-token strings the way transformers 5.19 resolves them: tokenizer_config.json values (a string or an
    AddedToken object); when that config has no added_tokens_decoder (the legacy layout), special_tokens_map.json
    is also read and its entries take precedence. Returns {name: string} for the names in SPECIAL_TOKEN_NAMES."""
    values = {}
    for name in SPECIAL_TOKEN_NAMES:
        token = _token_string(config.get(name), "tokenizer_config.json", name)
        if token is not None:
            values[name] = token
    if "added_tokens_decoder" not in config:
        legacy = read_json_object(files, "special_tokens_map.json") or {}
        for name in SPECIAL_TOKEN_NAMES:
            if name in legacy:
                token = _token_string(legacy[name], "special_tokens_map.json", name)
                if token is None:
                    values.pop(name, None)
                else:
                    values[name] = token
    return values


def byte_alphabet():
    """The characters byte-level BPE vocabularies spell bytes with (GPT-2's bytes_to_unicode): {character: byte}."""
    printable = {*range(ord("!"), ord("~") + 1), *range(ord("\u00a1"), ord("\u00ac") + 1), *range(ord("\u00ae"), 256)}
    extra = iter(range(256, 512))
    return {chr(byte) if byte in printable else chr(next(extra)): byte for byte in range(256)}


def _byte_level(backend):
    """Whether tokenizer.json decodes with a ByteLevel decoder, on its own or inside a Sequence."""
    decoder = json.loads(backend.to_str()).get("decoder") or {}
    return decoder.get("type") == "ByteLevel" or any(d.get("type") == "ByteLevel" for d in decoder.get("decoders") or [])


def _backend(text):
    try:
        backend = tokenizers.Tokenizer.from_str(text)
    except Exception as error:  # the Rust parser reports every malformed file as a plain Exception
        raise ModelTextError(f"tokenizer.json is not a valid tokenizer: {error}") from None
    backend.no_truncation()
    backend.no_padding()
    backend.encode_special_tokens = False
    return backend


class Tokenizer:
    """A loaded tokenizer.json plus the special-token strings of its model.

    `vocab_size` is max id + 1 over the vocabulary with added tokens; loading fails unless ids are contiguous
    (equal to get_vocab_size(True)). `added_tokens` maps id -> AddedToken, `special_ids` holds the ids of added
    tokens marked special, `special_tokens` maps bos_token/eos_token/... to their strings. `backend` is the
    `tokenizers.Tokenizer`; it is safe to share across threads only because nothing mutates it after load, so
    callers must not either. split_special_tokens=True makes encode() default to splitting control-token text
    such as "<|im_end|>" into ordinary pieces, through a second backend built on first use.
    """

    def __init__(self, data, *, special_tokens=None, split_special_tokens=False):
        if isinstance(data, (bytes, bytearray, memoryview)):
            try:
                text = bytes(data).decode("utf-8")
            except UnicodeDecodeError as error:
                raise ModelTextError(f"tokenizer.json is not UTF-8 ({error.reason} at byte {error.start})") from None
        elif isinstance(data, str):
            text = data
        else:
            raise TextError("tokenizer data must be the bytes or text of a tokenizer.json")
        if not isinstance(split_special_tokens, bool):
            raise TextError("split_special_tokens must be True or False")
        special = dict(special_tokens or {})
        for name, value in special.items():
            if name not in SPECIAL_TOKEN_NAMES:
                raise TextError(f"unknown special token name {name!r}; expected one of {', '.join(SPECIAL_TOKEN_NAMES)}")
            if not isinstance(value, str):
                raise TextError(f"special token {name} must be a string")
        backend = _backend(text)
        vocab = backend.get_vocab(with_added_tokens=True)
        if not vocab:
            raise ModelTextError("tokenizer.json has an empty vocabulary")
        size = max(vocab.values()) + 1
        count = backend.get_vocab_size(with_added_tokens=True)
        distinct = len(set(vocab.values()))
        if size != count or distinct != size:
            raise ModelTextError(f"tokenizer.json ids are not contiguous: {distinct} distinct ids for {count} "
                                 f"tokens, largest id {size - 1}")
        added = backend.get_added_tokens_decoder()
        self.backend = backend
        self.vocab_size = size
        self.added_tokens = MappingProxyType({index: AddedToken(index, token.content, bool(token.special))
                                              for index, token in sorted(added.items())})
        self.special_ids = frozenset(index for index, token in self.added_tokens.items() if token.special)
        self.special_tokens = MappingProxyType(special)
        self.bos_token_id = backend.token_to_id(special["bos_token"]) if "bos_token" in special else None
        self.eos_token_id = backend.token_to_id(special["eos_token"]) if "eos_token" in special else None
        self.split_special_tokens = split_special_tokens
        self._split = None
        self._bytes = None
        self._lock = threading.Lock()

    @classmethod
    def from_bytes(cls, data, *, special_tokens=None, split_special_tokens=False):
        return cls(data, special_tokens=special_tokens, split_special_tokens=split_special_tokens)

    @classmethod
    def from_file(cls, path, *, special_tokens=None, split_special_tokens=False):
        return cls(Path(path).read_bytes(), special_tokens=special_tokens, split_special_tokens=split_special_tokens)

    @classmethod
    def from_files(cls, files, *, split_special_tokens=False, tokenizer_config=None):
        """Load tokenizer.json (required) with the special tokens of tokenizer_config.json/special_tokens_map.json
        from a ModelFiles or a directory path. `tokenizer_config` is the already parsed tokenizer_config.json, when
        the caller has read it."""
        files = model_files(files)
        data = read_bytes(files, "tokenizer.json", MAX_FILE_BYTES)
        if data is None:
            raise ModelTextError("tokenizer.json is missing")
        if tokenizer_config is None:
            tokenizer_config = read_json_object(files, "tokenizer_config.json") or {}
        special = special_tokens_from(tokenizer_config, files)
        return cls(data, special_tokens=special, split_special_tokens=split_special_tokens)

    def _split_backend(self):
        with self._lock:
            if self._split is None:
                backend = _backend(self.backend.to_str())
                backend.encode_special_tokens = True
                self._split = backend
            return self._split

    def encode(self, text, *, add_special_tokens, split_special_tokens=None):
        """Token ids of `text`. add_special_tokens=True applies the tokenizer.json post-processor (for example a
        BOS); chat text rendered from a template passes False. Control-token text inside `text` is parsed as
        the control token unless split_special_tokens (default: the constructor's flag) is True."""
        if not isinstance(text, str):
            raise TextError("text to encode must be a string")
        if not isinstance(add_special_tokens, bool):
            raise TextError("add_special_tokens must be True or False")
        if split_special_tokens is None:
            split_special_tokens = self.split_special_tokens
        elif not isinstance(split_special_tokens, bool):
            raise TextError("split_special_tokens must be True or False")
        backend = self._split_backend() if split_special_tokens else self.backend
        try:
            return list(backend.encode(text, add_special_tokens=add_special_tokens).ids)
        except Exception as error:  # lone surrogates and similar input the Rust side rejects
            raise TextError(f"tokenizer cannot encode the text: {error}") from None

    def validate(self, ids):
        """Return `ids` as a list of ints, raising TokenError unless each is an integer in [0, vocab_size)."""
        if isinstance(ids, (str, bytes, bytearray)):
            raise TokenError("token ids must be a sequence of integers")
        try:
            iterator = iter(ids)
        except TypeError:
            raise TokenError("token ids must be a sequence of integers") from None
        size = self.vocab_size
        result = []
        for position, value in enumerate(iterator):
            token = None
            if not isinstance(value, bool):
                try:
                    token = operator.index(value)
                except TypeError:
                    pass
            if token is None or not 0 <= token < size:
                raise TokenError(f"token id {value!r} at position {position} is not an integer in [0, {size})")
            result.append(token)
        return result

    def decode(self, ids, *, skip_special_tokens=False):
        """Text of `ids`. skip_special_tokens drops added tokens marked special (non-special added tokens stay)."""
        if not isinstance(skip_special_tokens, bool):
            raise TextError("skip_special_tokens must be True or False")
        return self.backend.decode(self.validate(ids), skip_special_tokens=skip_special_tokens)

    def id_to_token(self, token_id):
        """The token string of one id; TokenError when the id is out of range."""
        return self.backend.id_to_token(self.validate((token_id,))[0])

    def token_to_id(self, token):
        """The id of one token string, or None when it is not in the vocabulary."""
        if not isinstance(token, str):
            raise TextError("token must be a string")
        return self.backend.token_to_id(token)

    def get_vocab(self):
        """A new {token: id} dict of the whole vocabulary, added tokens included."""
        return dict(self.backend.get_vocab(with_added_tokens=True))

    def token_bytes(self):
        """The bytes every id stands for in text, read from the vocabulary, as a tuple indexed by id: byte-level
        pieces through the byte alphabet, SentencePiece <0xNN> pieces as that byte, other pieces as UTF-8 with
        "\u2581" as a space, added tokens as their content. Servers name tokens by these bytes (logprobs)."""
        with self._lock:
            if self._bytes is None:
                alphabet = byte_alphabet() if _byte_level(self.backend) else None
                out = []
                for token in range(self.vocab_size):
                    added = self.added_tokens.get(token)
                    piece = added.content if added is not None else self.backend.id_to_token(token)
                    if added is not None:
                        out.append(piece.encode())
                    elif alphabet is not None:
                        out.append(bytes(alphabet[c] for c in piece) if all(c in alphabet for c in piece)
                                   else piece.encode())
                    elif len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
                        out.append(bytes([int(piece[3:5], 16)]))
                    else:
                        out.append(piece.replace("\u2581", " ").encode())
                self._bytes = tuple(out)
            return self._bytes
