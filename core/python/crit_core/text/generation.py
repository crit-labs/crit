"""Generation defaults from generation_config.json and config.json, with transformers' precedence (standard library).

GenerationConfig keeps "unset" distinct from transformers' implicit fallbacks (top_k=50, temperature=1.0, ...):
a field is None unless a file sets it. legacy_eos_ids reproduces crit Eval's stop_ids rule for well-formed files.
Both files are read as the native loaders read them: at most 16 MiB, NaN and Infinity literals allowed where no typed
field reads them, integers checked against the same ranges (eos ids in [0, 2^31), bos and pad any 32-bit id, counts
32-bit unsigned, max_length included, nothing beyond 64 bits).
"""
from dataclasses import dataclass, field
import math
from types import MappingProxyType

from .errors import MAX_CONFIG_BYTES, ModelTextError, model_files, read_json_object

GENERATION_FILE = "generation_config.json"
MODEL_FILE = "config.json"
SAMPLING_FIELDS = ("do_sample", "temperature", "top_p", "top_k", "min_p", "repetition_penalty")
SUBCONFIGS = ("decoder", "generator", "text_config")   # searched in this order, as transformers does
TOKEN_MIN, TOKEN_MAX = -(1 << 31), (1 << 31) - 1
U32_MAX = (1 << 32) - 1


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _within_64_bits(value, source, name):
    # The native reader keeps integers outside [-2^63, 2^64) only where no typed field reads them.
    if _is_int(value) and not -(1 << 63) <= value < (1 << 64):
        raise ModelTextError(f"{source}: {name}: integer exceeds 64 bits")


def _token_ids(value, source, name, vocab_size):
    """eos ids: an id or a list of ids in [0, 2^31), inside the tokenizer vocabulary when one is given."""
    if value is None:
        return ()
    values = value if isinstance(value, list) else [value]
    for item in values:
        _within_64_bits(item, source, name)
        if not _is_int(item) or not 0 <= item <= TOKEN_MAX:
            raise ModelTextError(f"{source}: {name} must be a token id in [0, {TOKEN_MAX}] or a list of them")
        if vocab_size is not None and item >= vocab_size:
            raise ModelTextError(f"{source}: {name} {item} is outside the tokenizer vocabulary of {vocab_size}")
    return tuple(values)


def _token_id(value, source, name, vocab_size):
    """bos and pad: one id in [-2^31, 2^31) (pad -1 is common); only a non-negative id must be inside the vocabulary."""
    if value is None:
        return None
    if isinstance(value, list):
        raise ModelTextError(f"{source}: {name} must be one token id")
    _within_64_bits(value, source, name)
    if not _is_int(value) or not TOKEN_MIN <= value <= TOKEN_MAX:
        raise ModelTextError(f"{source}: {name} must be a token id in [{TOKEN_MIN}, {TOKEN_MAX}]")
    if vocab_size is not None and value >= vocab_size:
        raise ModelTextError(f"{source}: {name} {value} is outside the tokenizer vocabulary of {vocab_size}")
    return value


def _number(value, source, name, valid, expectation):
    if value is None:
        return None
    _within_64_bits(value, source, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ModelTextError(f"{source}: {name} must be a finite number")
    if not valid(value):
        raise ModelTextError(f"{source}: {name} {value} must be {expectation}")
    return value


def _integer(value, source, name):
    if value is None:
        return None
    _within_64_bits(value, source, name)
    if not _is_int(value) or not 0 <= value <= U32_MAX:
        raise ModelTextError(f"{source}: {name} must be an integer in [0, {U32_MAX}]")
    return value


def _flag(value, source, name):
    if value is None or isinstance(value, bool):
        return value
    raise ModelTextError(f"{source}: {name} must be true, false or null")


def _strings(value, source, name):
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
        raise ModelTextError(f"{source}: {name} must be a non-empty string or a list of them")
    return tuple(values)


def _section(data, name, source):
    value = data.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ModelTextError(f"{source}: {name} must be an object")
    return value


def model_generation_view(config):
    """config.json as transformers' GenerationConfig.from_model_config reads it: top-level values that are not
    null, then fields still unset filled from the first non-empty of decoder, generator and text_config."""
    view = {key: value for key, value in config.items() if value is not None}
    for name in SUBCONFIGS:
        section = config.get(name)
        if section:
            if not isinstance(section, dict):
                raise ModelTextError(f"{MODEL_FILE}: {name} must be an object")
            for key, value in section.items():
                if view.get(key) is None and value is not None:
                    view[key] = value
            break
    return view


def _unique(values):
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class GenerationConfig:
    """A model's generation defaults.

    Fields come from generation_config.json when that file exists, else from config.json (see
    model_generation_view), as transformers loads them. eos_token_ids falls back to config.json when
    generation_config.json leaves eos_token_id unset. stop_token_ids is the ordered union of eos_token_ids, the
    eos_token_id of both files and their text_config, and the tokenizer's eos token. `raw` is
    generation_config.json as read, unknown keys included ({} without the file). max_length (prompt plus generated
    tokens, a count like max_new_tokens) is kept for parity with the native GenerationConfig.
    """
    eos_token_ids: tuple = ()
    stop_token_ids: tuple = ()
    bos_token_id: int | None = None
    pad_token_id: int | None = None
    do_sample: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int | None = None
    stop_strings: tuple = ()
    raw: MappingProxyType = field(default_factory=lambda: MappingProxyType({}), hash=False)
    max_length: int | None = None

    @classmethod
    def from_files(cls, files, tokenizer=None):
        """Read generation_config.json and config.json from a ModelFiles or a directory path. With a tokenizer,
        token ids must lie inside its vocabulary and its eos token joins stop_token_ids."""
        files = model_files(files)
        generation = _read_config(files, GENERATION_FILE)
        config = _read_config(files, MODEL_FILE)
        view = model_generation_view(config) if config is not None else {}
        source, name = (generation, GENERATION_FILE) if generation is not None else (view, MODEL_FILE)
        vocab = tokenizer.vocab_size if tokenizer is not None else None
        if generation is not None and generation.get("eos_token_id") is not None:
            eos = _token_ids(generation["eos_token_id"], GENERATION_FILE, "eos_token_id", vocab)
        else:
            eos = _token_ids(view.get("eos_token_id"), MODEL_FILE, "eos_token_id", vocab)
        stops = list(eos)
        for data, label in ((generation, GENERATION_FILE), (config, MODEL_FILE)):
            if data is not None:
                stops += _token_ids(data.get("eos_token_id"), label, "eos_token_id", vocab)
                stops += _token_ids(_section(data, "text_config", label).get("eos_token_id"), label,
                                    "text_config.eos_token_id", vocab)
        if tokenizer is not None and tokenizer.eos_token_id is not None:
            stops.append(tokenizer.eos_token_id)
        return cls(
            eos_token_ids=_unique(eos),
            stop_token_ids=_unique(stops),
            bos_token_id=_token_id(source.get("bos_token_id"), name, "bos_token_id", vocab),
            pad_token_id=_token_id(source.get("pad_token_id"), name, "pad_token_id", vocab),
            do_sample=_flag(source.get("do_sample"), name, "do_sample"),
            temperature=_number(source.get("temperature"), name, "temperature", lambda v: v >= 0, "at least 0"),
            top_p=_number(source.get("top_p"), name, "top_p", lambda v: 0 <= v <= 1, "between 0 and 1"),
            top_k=_integer(source.get("top_k"), name, "top_k"),
            min_p=_number(source.get("min_p"), name, "min_p", lambda v: 0 <= v <= 1, "between 0 and 1"),
            repetition_penalty=_number(source.get("repetition_penalty"), name, "repetition_penalty",
                                       lambda v: v > 0, "greater than 0"),
            max_new_tokens=_integer(source.get("max_new_tokens"), name, "max_new_tokens"),
            max_length=_integer(source.get("max_length"), name, "max_length"),
            stop_strings=_strings(source.get("stop_strings"), name, "stop_strings"),
            raw=MappingProxyType(dict(generation or {})),
        )

    def sampling_defaults(self):
        """Only the sampling fields a file set explicitly: never transformers' implicit top_k=50 and the like."""
        return {name: getattr(self, name) for name in SAMPLING_FIELDS if getattr(self, name) is not None}


def _read_config(files, name):
    return read_json_object(files, name, max_bytes=MAX_CONFIG_BYTES, allow_nonfinite=True)


def legacy_eos_ids(files):
    """crit Eval's stop_ids rule: the sorted union of eos_token_id (an id or a list) at the top level and in
    text_config (null reads as absent) of generation_config.json and config.json. Malformed values raise
    ModelTextError, and so does a generation_config.json that exists but cannot be read (a dangling link), which
    Eval skips."""
    files = model_files(files)
    ids = set()
    for name in (GENERATION_FILE, MODEL_FILE):
        data = _read_config(files, name)
        if data is None:
            continue
        ids.update(_token_ids(data.get("eos_token_id"), name, "eos_token_id", None))
        ids.update(_token_ids(_section(data, "text_config", name).get("eos_token_id"), name,
                              "text_config.eos_token_id", None))
    return sorted(ids)
