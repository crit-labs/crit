"""Offline fixtures for the Core text-layer tests: tiny tokenizers built in-process and self-written chat templates.

Nothing here is downloaded or vendored. The byte-level BPE is trained on a small deterministic multilingual and
emoji corpus (Llama 3 / Qwen layout); the SentencePiece-style BPE keeps <0x00>..<0xFF> as ordinary vocabulary
entries with byte fallback (Llama 2 / Gemma layout); the straddling BPE has merges that join the end of one
character with the start of the next, so runs of its tokens never end on a character boundary. ByteText computes
which characters a prefix of ids completes from the vocabulary's bytes, independently of decode(). The templates
are written for these tests in the style of those model families.
"""
import bisect
import functools
import itertools
import json
from pathlib import Path

from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

CORPUS = (
    "Hello world, this is a tiny corpus for testing tokenizers.",
    "The quick brown fox jumps over the lazy dog.",
    "Grüße aus München! Ça va? Fußgängerübergänge sind breit.",
    "日本語のテキストです。東京は大きい。",
    "Привет, мир! Как дела?",
    "def f(x):\n    return x * 2\n",
    "emoji 😀 and more 👍🏽 and flags 🇩🇪 plus 👩‍👩‍👧",
    "Ελληνικά γράμματα και العربية",
    "  leading spaces and\ttabs\n\nnew lines",
)

BYTE_SPECIALS = ("<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
                 "<|im_start|>", "<|im_end|>")
BYTE_PLAIN = ("<think>", "</think>")          # added tokens that are not special
SP_SPECIALS = ("<unk>", "<s>", "</s>", "<start_of_turn>", "<end_of_turn>")


@functools.lru_cache(maxsize=None)
def byte_level_json():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=420, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=list(BYTE_SPECIALS), show_progress=False)
    tok.train_from_iterator(list(CORPUS) * 8, trainer)
    tok.add_tokens([AddedToken(content, special=False, normalized=False) for content in BYTE_PLAIN])
    bos = tok.token_to_id("<|begin_of_text|>")
    tok.post_processor = processors.TemplateProcessing(single="<|begin_of_text|> $A",
                                                       pair="<|begin_of_text|> $A <|begin_of_text|> $B",
                                                       special_tokens=[("<|begin_of_text|>", bos)])
    return tok.to_str()


@functools.lru_cache(maxsize=None)
def sentencepiece_json():
    trained = Tokenizer(models.BPE(unk_token="<unk>"))
    trained.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    trained.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never", split=True)
    alphabet = sorted(set("▁abcdefghijklmnopqrstuvwxyzHTGMCdef,.!?:()*2\n\tü"))  # the rest falls back to bytes
    trainer = trainers.BpeTrainer(vocab_size=300, initial_alphabet=alphabet, limit_alphabet=len(alphabet),
                                  special_tokens=["<unk>", "<s>", "</s>"], show_progress=False)
    trained.train_from_iterator(list(CORPUS) * 8, trainer)
    model = json.loads(trained.to_str())["model"]
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for byte in range(256):
        vocab[f"<0x{byte:02X}>"] = len(vocab)
    for piece, _ in sorted(model["vocab"].items(), key=lambda item: item[1]):
        vocab.setdefault(piece, len(vocab))
    merges = [tuple(merge) if isinstance(merge, list) else tuple(merge.split(" ")) for merge in model["merges"]]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=True, unk_token="<unk>", fuse_unk=True))
    tok.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    tok.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(),
                                     decoders.Strip(" ", 1, 0)])
    tok.add_special_tokens([AddedToken(content, special=True, normalized=False) for content in SP_SPECIALS])
    tok.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    return tok.to_str()


def _byte_chars():
    """The byte-level alphabet: byte -> the printable character that stands for it in vocabularies."""
    printable = {*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)}
    chars, extra = {}, 0
    for byte in range(256):
        if byte in printable:
            chars[byte] = chr(byte)
        else:
            chars[byte] = chr(256 + extra)
            extra += 1
    return chars


BYTE_CHARS = _byte_chars()
CHAR_BYTES = {char: byte for byte, char in BYTE_CHARS.items()}
STRADDLES = ((0xB3, 0xCE), (0xBB, 0xCE), (0xB4, 0xD0), (0x80, 0xF0), (0xA4, 0x67),  # end of γ λ д 😀 ä + next start
             (0xBD, 0xC3))                                                    # end of a literal U+FFFD + start of ä
STRADDLE_CORPUS = ("γ", "λ", "д", "😀", "ä", "g", "a", " ", "Ελλάδα", "дд")


@functools.lru_cache(maxsize=None)
def straddling_json():
    """A byte-level BPE over the 256 byte tokens plus merges that straddle characters: b"\xb3\xce" ends one "γ"
    and starts the next Greek letter, b"\xa4g" ends "ä" and adds "g", and second-level merges append one more byte,
    so "γγγ..." encodes to tokens that each end inside a character."""
    vocab = {BYTE_CHARS[byte]: byte for byte in range(256)}
    merges = []
    for first, second in STRADDLES:
        pair = (BYTE_CHARS[first], BYTE_CHARS[second])
        merges.append(pair)
        vocab["".join(pair)] = len(vocab)
    for first, second in STRADDLES[:3]:
        pair = (BYTE_CHARS[first] + BYTE_CHARS[second], BYTE_CHARS[first])
        merges.append(pair)
        vocab["".join(pair)] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken("<|eot|>", special=True, normalized=False)])
    return tok.to_str()


def is_sentencepiece(tokenizer):
    return tokenizer.token_to_id("<0x41>") is not None


def token_bytes(tokenizer, ids, skip_special_tokens):
    """The bytes each id adds to the decoded text, read from the vocabulary rather than from decode(): byte-level
    tokens map their characters through the byte alphabet, SentencePiece <0xNN> tokens are one byte and other
    pieces are UTF-8 with "▁" as a space; added tokens are their content, or b"" when special and skipped."""
    sentencepiece = is_sentencepiece(tokenizer)
    pieces = []
    for token in ids:
        added = tokenizer.added_tokens.get(token)
        if added is not None:
            pieces.append(b"" if skip_special_tokens and added.special else added.content.encode())
            continue
        piece = tokenizer.id_to_token(token)
        if not sentencepiece:
            pieces.append(bytes(CHAR_BYTES[char] for char in piece))
        elif len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            pieces.append(bytes([int(piece[3:5], 16)]))
        else:
            pieces.append(piece.replace("▁", " ").encode())
    return pieces


class ByteText:
    """decode(ids) for ids whose bytes form valid UTF-8, with the byte offsets of every id and character, so a test
    can tell which characters the first n ids complete without asking the detokenizer's rules."""

    def __init__(self, tokenizer, ids, skip_special_tokens=True):
        pieces = token_bytes(tokenizer, ids, skip_special_tokens)
        data = b"".join(pieces)
        strip = 1 if is_sentencepiece(tokenizer) and data.startswith(b" ") else 0  # the Strip decoder
        self.text = data[strip:].decode("utf-8")
        self.id_ends = [max(end - strip, 0) for end in itertools.accumulate((len(p) for p in pieces), initial=0)]
        self.char_ends = list(itertools.accumulate(len(char.encode()) for char in self.text))

    def complete(self, count):
        """The leading characters whose bytes all come from the first `count` ids."""
        return self.text[:bisect.bisect_right(self.char_ends, self.id_ends[count])]

    def boundary(self, count):
        """True when the bytes of the first `count` ids end on a character boundary."""
        end = self.id_ends[count]
        return end == 0 or end in self.char_ends


def gapped_json():
    """The SentencePiece-style tokenizer with one vocabulary entry removed, leaving a hole in the id space."""
    data = json.loads(sentencepiece_json())
    del data["model"]["vocab"]["<0x00>"]
    return json.dumps(data)


def truncating_json():
    """The byte-level tokenizer saved with truncation to 4 tokens and padding enabled."""
    tok = Tokenizer.from_str(byte_level_json())
    tok.enable_truncation(4)
    tok.enable_padding(length=64)
    return tok.to_str()


LLAMA3 = """{{- bos_token }}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "01 Jan 2000" %}
    {%- endif %}
{%- endif %}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] | trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}
{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}
{{- "Today Date: " + date_string + "\\n\\n" }}
{%- if tools is not none %}
    {{- "Tools:\\n" }}
    {%- for tool in tools %}
        {{- tool | tojson(indent=2) + "\\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}
{%- for message in messages %}
    {%- if message['role'] == 'assistant' %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
        {%- generation %}
            {{- message['content'] | trim + '<|eot_id|>' }}
        {%- endgeneration %}
    {%- else %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
{%- endif %}
"""

QWEN3 = """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools><|im_end|>\\n" }}
{%- elif messages[0].role == 'system' %}
    {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message.role == 'user' or (message.role == 'system' and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
    {%- elif message.role == 'assistant' %}
        {%- set reasoning = message.reasoning_content if message.reasoning_content is string else '' %}
        {{- '<|im_start|>assistant\\n' }}
        {%- if loop.last and reasoning %}
            {{- '<think>\\n' + reasoning + '\\n</think>\\n\\n' }}
        {%- endif %}
        {{- message.content }}
        {%- for call in message.tool_calls or [] %}
            {%- if call.function %}
                {%- set call = call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{"name": "' + call.name + '", "arguments": ' }}
            {%- if call.arguments is string %}
                {{- call.arguments }}
            {%- else %}
                {{- call.arguments | tojson }}
            {%- endif %}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == 'tool' %}
        {{- '<|im_start|>user\\n<tool_response>\\n' + message.content + '\\n</tool_response><|im_end|>\\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}
"""

GEMMA = """{{- bos_token }}
{%- if messages[0]['role'] == 'system' %}
    {%- set first_user_prefix = messages[0]['content'] + '\\n\\n' %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set first_user_prefix = "" %}
    {%- set loop_messages = messages %}
{%- endif %}
{%- for message in loop_messages %}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{- raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif %}
    {%- set role = "model" if message['role'] == 'assistant' else message['role'] %}
    {{- '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string %}
        {{- message['content'] | trim }}
    {%- else %}
        {%- for part in message['content'] %}
            {%- if part['type'] == 'text' %}
                {{- part['text'] | trim }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- '<end_of_turn>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<start_of_turn>model\\n' }}
{%- endif %}
"""

LOOPS = """{%- for message in messages %}
    {%- if message.role == 'stop' %}{% break %}{% endif %}
    {%- if message.role == 'skip' %}{% continue %}{% endif %}
    {{- message.content + ';' }}
{%- endfor %}"""


def file_bytes(value):
    """Bytes of one fixture file given as bytes, text or a JSON value."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, ensure_ascii=False, indent=1).encode("utf-8")


def write_files(directory, files):
    """Write {relative name: str | bytes | JSON value} under directory; returns the directory as a Path."""
    root = Path(directory)
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file_bytes(value))
    return root


def llama_files(**extra):
    """A byte-level Llama-3-style model: tokenizer files, template, generation and model config."""
    files = {
        "tokenizer.json": byte_level_json(),
        "tokenizer_config.json": {"bos_token": "<|begin_of_text|>",
                                  "eos_token": {"content": "<|eot_id|>", "special": True},
                                  "added_tokens_decoder": {}},
        "chat_template.jinja": LLAMA3,
        "generation_config.json": {"bos_token_id": 0, "eos_token_id": [1, 4], "do_sample": True, "temperature": 0.6,
                                   "top_p": 0.9, "custom_key": {"kept": True}},
        "config.json": {"eos_token_id": 1, "text_config": {"eos_token_id": 4, "vocab_size": 420}},
    }
    files.update(extra)
    return files
