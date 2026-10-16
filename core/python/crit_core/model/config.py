"""Model config.json rules: the text section, typed fields, position limit, attention shape, layer kinds and state math.

The rule functions take a parsed config mapping and give the results of crit-eval-v2's own rules (ceiling.text_config,
attention_shape; context.model_limit, kv_elements_per_token, fixed_state_bytes; Eval now calls these) for every
well-typed config. A rule reads values in
eval's order and only where eval evaluates them. A value of the wrong type that it reads raises ConfigError instead of
producing a number, and so does an input on which eval fails (except a null text_config, which reads as absent).
load_model_config adds strict JSON, the native reader's depth limit and typed fields. Every typed field, the length
keys and YaRN included, refuses an integer outside [-2^63, 2^64), as the native reader does; untyped keys keep it.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import math
import numbers
import os
from pathlib import Path
from types import MappingProxyType

from .. import jsonfile
from ..files import ModelFilesError, _read_regular, model_files

CONFIG_NAME = "config.json"
MAX_CONFIG_BYTES = jsonfile.MAX_CONFIG_BYTES   # 16 MiB, shared with crit_core.text and the native loaders
MAX_CONFIG_DEPTH = 128   # nested objects and arrays, the root included (the native JSON reader's default)
MAX_LAYERS = 1 << 16
MAX_PATH_SHOWN = 200   # characters of a nested key path named in an error
U32_MAX = (1 << 32) - 1
TOKEN_MIN, TOKEN_MAX = -(1 << 31), (1 << 31) - 1
INT64_MIN, UINT64_END = -(1 << 63), 1 << 64   # the integers the native reader can type
LENGTH_KEYS = ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "max_sequence_length",
               "model_max_length")
EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
NORM_EPS_KEYS = ("rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon")

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
LINEAR_ATTENTION = "linear_attention"
OTHER = "other"
LAYER_KINDS = MappingProxyType({"full_attention": FULL_ATTENTION, "attention": FULL_ATTENTION,
                                "sliding_attention": SLIDING_ATTENTION, "linear_attention": LINEAR_ATTENTION})


class ConfigError(ValueError):
    """A missing, malformed or wrongly typed model config; messages read '<file>: key 'x': ...'."""


def _empty():
    return MappingProxyType({})


def _plain(value):
    # The pickled form: read-only mappings become dicts (they hold no other dicts), recursively through tuples.
    if isinstance(value, MappingProxyType):
        return {key: _plain(item) for key, item in value.items()}
    if type(value) is tuple:
        return tuple(_plain(item) for item in value)
    return value


def _read_only(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _read_only(item) for key, item in value.items()})
    if type(value) is tuple:
        return tuple(_read_only(item) for item in value)
    return value


def _rebuild(cls, values):
    return cls(**{name: _read_only(value) for name, value in values.items()})


class _Picklable:
    """Pickling, copy and deepcopy for frozen results whose MappingProxyType fields cannot be pickled."""

    def __reduce__(self):
        return _rebuild, (type(self), {item.name: _plain(getattr(self, item.name)) for item in fields(self)})


@dataclass(frozen=True)
class RopeConfig(_Picklable):
    """Rotary position parameters from rope_scaling (else rope_parameters) plus rope_theta; type "" when none."""
    theta: float | None = None
    partial_rotary_factor: float | None = None
    type: str = ""
    factor: float | None = None
    attention_factor: float | None = None
    beta_fast: float | None = None
    beta_slow: float | None = None
    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    mscale: float | None = None
    mscale_all_dim: float | None = None
    original_max_position_embeddings: int | None = None
    short_factor: tuple = ()
    long_factor: tuple = ()
    mrope_section: tuple = ()
    raw: Mapping = field(default_factory=_empty)


@dataclass(frozen=True)
class RopeLayer:
    """One entry of a rope object keyed by layer type, such as "sliding_attention" (transformers v5 Gemma 3 style):
    its own rope_type, scaling fields, rope_theta and partial_rotary_factor; nothing falls back to the section."""
    layer_type: str = ""
    rope: RopeConfig = field(default_factory=RopeConfig)


@dataclass(frozen=True)
class MoeConfig:
    """Mixture-of-experts fields; experts_key names the key the expert count came from."""
    experts: int = 0
    experts_key: str = ""
    experts_per_token: int = 0
    shared_experts: int | None = None
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    first_k_dense_replace: int | None = None
    decoder_sparse_step: int | None = None
    n_group: int | None = None
    topk_group: int | None = None
    mlp_only_layers: tuple = ()
    norm_topk_prob: bool | None = None
    routed_scaling_factor: float | None = None
    scoring_func: str = ""


@dataclass(frozen=True)
class MlaConfig:
    """Multi-head latent attention: present when kv_lora_rank is positive; absent sizes read as 0."""
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
    q_lora_rank: int | None = None


@dataclass(frozen=True)
class LinearAttentionConfig:
    """linear_* recurrence sizes: present when any of them is set."""
    num_key_heads: int | None = None
    num_value_heads: int | None = None
    key_head_dim: int | None = None
    value_head_dim: int | None = None
    conv_kernel_dim: int | None = None


@dataclass(frozen=True)
class QuantizationConfig(_Picklable):
    """quantization_config: quant_method ("" when absent) and the raw subtree, never interpreted here."""
    method: str = ""
    raw: Mapping = field(default_factory=_empty)


@dataclass(frozen=True)
class ModelConfig(_Picklable):
    """Typed view of one config.json. `section` is the text section (text_config keys over the root) that every
    typed field and rule reads; `raw` is the root and `text` the text_config object (the root when absent). All
    mappings are read-only all the way down (arrays become tuples); to_dict() returns a plain copy of the root.
    intermediate_size is the integer form; a per-layer array (Gemma 3n) leaves it None and fills
    intermediate_sizes. rope_by_layer_type holds one RopeLayer per entry of a rope_parameters (or rope_scaling)
    object keyed by layer type, in file order. Pickling and deep copies rebuild an equal object, except that a NaN
    kept in raw, text or section makes a pickled copy compare unequal (NaN never equals a new NaN)."""
    source: str
    raw: Mapping
    text: Mapping
    section: Mapping
    architectures: tuple = ()
    model_type: str = ""
    text_model_type: str = ""
    vocab_size: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    max_positions: int | None = None
    max_positions_key: str = ""
    rms_norm_eps: float | None = None
    norm_eps_key: str = ""
    tie_word_embeddings: bool | None = None
    dtype: str = ""
    rope: RopeConfig = field(default_factory=RopeConfig)
    sliding_window: int | None = None
    full_attention_interval: int | None = None
    layer_types: tuple = ()
    moe: MoeConfig | None = None
    mla: MlaConfig | None = None
    linear: LinearAttentionConfig | None = None
    num_nextn_predict_layers: int | None = None
    eos_token_ids: tuple = ()
    bos_token_id: int | None = None
    pad_token_id: int | None = None
    quantization: QuantizationConfig | None = None
    intermediate_sizes: tuple = ()
    rope_by_layer_type: tuple = ()

    def get(self, key, default=None):
        """A key of the text section, falling back to the root (the text_config merge rule)."""
        return self.section.get(key, default)

    def require_int(self, key):
        """A positive integer key of the text section; ConfigError when it is missing, zero or mistyped."""
        return _positive(self.section, key, self.source)

    def to_dict(self):
        """A plain, independent dict copy of the root object."""
        return _thawed(self.raw)

    def position_limit(self):
        return position_limit(self.section, source=self.source)

    def attention_shape(self):
        return attention_shape(self.section, source=self.source)

    def layer_kinds(self):
        return layer_kinds(self.section, source=self.source)

    def kv_elements_per_token(self):
        return kv_elements_per_token(self.section, source=self.source)

    def fixed_state_bytes(self, kv_bytes_per_element):
        return fixed_state_bytes(self.section, kv_bytes_per_element, source=self.source)

    def effective_head_dim(self):
        return effective_head_dim(self.section, source=self.source)

    def effective_kv_heads(self):
        return effective_kv_heads(self.section, source=self.source)


# Typed reads. Each returns None for an absent key or JSON null and raises ConfigError for any other wrong value.

def _describe(value):
    if value is None:
        return "null"
    if value is True or value is False:
        return "true" if value else "false"
    if isinstance(value, Mapping):
        return "an object"
    if isinstance(value, (list, tuple)):
        return "an array"
    if isinstance(value, int) and value.bit_length() > 256:
        return "a very large integer"
    text = repr(value)
    return text if len(text) <= 48 else text[:45] + "..."


def _fail(source, key, reason):
    raise ConfigError(f"{source}: key '{key}': {reason}")


def _fail_at(source, path, reason):
    if len(path) > MAX_PATH_SHOWN:   # nested keys are caller data; keep the message bounded
        path = path[:MAX_PATH_SHOWN - 3] + "..."
    raise ConfigError(f"{source}: key '{path}': {reason}" if path else f"{source}: {reason}")


def _mapping_of(config, source):
    if not isinstance(config, Mapping):
        raise ConfigError(f"{source}: the config must be a JSON object, not {_describe(config)}")
    return config


def _is_int(value):
    """A non-boolean integer, numpy integers included (crit eval computes with them)."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _eval_int(value):
    # crit eval's isinstance(value, int) test for lengths and YaRN, without booleans
    return isinstance(value, int) and not isinstance(value, bool)


def _beyond_64_bits(value):
    return _is_int(value) and not INT64_MIN <= value < UINT64_END


def _check_64_bits(value, source, key):
    if _beyond_64_bits(value):
        _fail(source, key, "integer exceeds 64 bits")


def _finite_real(value):
    """A non-boolean real number as a finite float, else None (integers beyond a double's range included)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _utf8(text):
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:   # a lone surrogate, which an escaped \ud800 in JSON text can produce
        return False
    return True


def _check_int(value, source, key, minimum=0, maximum=U32_MAX):
    _check_64_bits(value, source, key)
    if not _is_int(value) or not minimum <= value <= maximum:
        _fail(source, key, f"expected an integer in [{minimum}, {maximum}], got {_describe(value)}")
    return int(value)


def _integer(config, key, source, label=None, maximum=U32_MAX):
    value = config.get(key)
    return None if value is None else _check_int(value, source, label or key, 0, maximum)


def _positive(config, key, source, why=""):
    value = _integer(config, key, source)
    if value is None:
        _fail(source, key, "missing" + why)
    if value == 0:
        _fail(source, key, "expected a positive integer, got 0")
    return value


def _real(config, key, source, label=None):
    value = config.get(key)
    if value is None:
        return None
    _check_64_bits(value, source, label or key)
    number = _finite_real(value)
    if number is None:
        _fail(source, label or key, f"expected a finite number, got {_describe(value)}")
    return number


def _string(config, key, source, label=None):
    value = config.get(key)
    if value is not None and not isinstance(value, str):
        _fail(source, label or key, f"expected a string, got {_describe(value)}")
    return value


def _boolean(config, key, source, label=None):
    value = config.get(key)
    if value is not None and type(value) is not bool:
        _fail(source, label or key, f"expected true or false, got {_describe(value)}")
    return value


def _array(config, key, source, label):
    value = config.get(key)
    if value is not None and not isinstance(value, (list, tuple)):
        _fail(source, label, f"expected an array, got {_describe(value)}")
    return value


def _strings(config, key, source, label=None):
    value = _array(config, key, source, label or key)
    if value is None:
        return None
    for index, item in enumerate(value):
        if not isinstance(item, str):
            _fail(source, label or key, f"entry {index} is {_describe(item)}, not a string")
    return tuple(value)


def _integers(config, key, source, label=None):
    value = _array(config, key, source, label or key)
    if value is None:
        return ()
    return tuple(_check_int(item, source, f"{label or key}[{index}]") for index, item in enumerate(value))


def _reals(config, key, source, label=None):
    value = _array(config, key, source, label or key)
    if value is None:
        return ()
    for index, item in enumerate(value):
        _check_64_bits(item, source, f"{label or key}[{index}]")
    numbers = tuple(_finite_real(item) for item in value)
    for index, number in enumerate(numbers):
        if number is None:
            _fail(source, f"{label or key}[{index}]", f"expected a finite number, got {_describe(value[index])}")
    return numbers


def _object(config, key, source, label=None):
    value = config.get(key)
    if value is not None and not isinstance(value, Mapping):
        _fail(source, label or key, f"expected an object, got {_describe(value)}")
    return value


def _token_ids(config, key, source):
    value = config.get(key)
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(_check_int(item, source, f"{key}[{index}]", TOKEN_MIN, TOKEN_MAX)
                     for index, item in enumerate(value))
    return (_check_int(value, source, key, TOKEN_MIN, TOKEN_MAX),)


def _token_id(config, key, source):
    value = config.get(key)
    return None if value is None else _check_int(value, source, key, TOKEN_MIN, TOKEN_MAX)


def _first(config, keys, read, source):
    for key in keys:
        value = read(config, key, source)
        if value is not None:
            return value, key
    return None, ""


def _frozen(value, source, path="", depth=1):
    """A read-only copy of a JSON value (objects become MappingProxyType, arrays tuples, numbers plain int or float)
    held to JSON: string keys, UTF-8 strings and at most MAX_CONFIG_DEPTH nested containers, the native reader's
    default. Non-finite floats are kept (json.loads reads them in configs such as Mamba2's time_step_limit); every
    typed real field rejects them. The depth bound also bounds this recursion and stops at a cycle in a caller's
    dict."""
    if isinstance(value, (Mapping, list, tuple)) and depth > MAX_CONFIG_DEPTH:
        raise ConfigError(f"{source}: nesting exceeds the depth limit of {MAX_CONFIG_DEPTH}")
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _fail_at(source, path, f"object key {_describe(key)} is not a string")
            if not _utf8(key):
                _fail_at(source, path, f"object key {_describe(key)} is not valid UTF-8")
            result[key] = _frozen(item, source, f"{path}.{key}" if path else key, depth + 1)
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple([_frozen(item, source, f"{path}[{index}]", depth + 1) for index, item in enumerate(value)])
    if isinstance(value, str):
        if not _utf8(value):
            _fail_at(source, path, f"{_describe(value)} is not valid UTF-8")
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        try:
            return float(value)
        except (OverflowError, ValueError):
            _fail_at(source, path, f"{_describe(value)} is beyond a double's range")
    _fail_at(source, path, f"{_describe(value)} is not a JSON value")


def _thawed(value):
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thawed(item) for item in value]
    return value


# Rules shared with crit eval. Each value is read in eval's order and only where eval evaluates it.

def text_section(config, source=CONFIG_NAME):
    """The config with text_config's keys over the root's (a missing or null text_config leaves the root)."""
    config = _mapping_of(config, source)
    text = _object(config, "text_config", source)
    return {**config, **(text or {})}


def position_limit(config, source=CONFIG_NAME):
    """(positions the config declares, where the number comes from), or (None, None).

    The first positive integer among LENGTH_KEYS, unless rope_scaling (else rope_parameters) declares YaRN with a
    positive original_max_position_embeddings and factor: then their product, described in the source string.
    Values of other types are skipped, as in crit eval, and booleans are never numbers here. An integer outside
    [-2^63, 2^64) among the values read, a product of 2^64 or more and a product that is not a finite number (where
    crit eval fails) raise ConfigError; crit eval returns the larger integers.
    """
    config = _mapping_of(config, source)
    value, where = _declared_positions(config, source)
    rope_key, rope = "rope_scaling", config.get("rope_scaling")
    if not rope:
        rope_key, rope = "rope_parameters", config.get("rope_parameters") or {}
    if isinstance(rope, Mapping) and (rope.get("rope_type") or rope.get("type")) == "yarn":
        original, factor = rope.get("original_max_position_embeddings"), rope.get("factor")
        _check_64_bits(original, source, f"{rope_key}.original_max_position_embeddings")
        _check_64_bits(factor, source, f"{rope_key}.factor")
        if _eval_int(original) and isinstance(factor, (int, float)) and not isinstance(factor, bool) \
                and original > 0 and factor > 0:
            try:   # exact for integers, as in crit eval; an infinite product or a factor beyond a double fails
                limit = int(original * factor)
            except (OverflowError, ValueError):
                _fail(source, f"{rope_key}.factor", f"{_describe(original)} original positions x factor "
                                                    f"{_describe(factor)} is not a finite number of positions")
            if limit >= UINT64_END:
                shown = factor if _eval_int(factor) else f"{factor:g}"
                _fail(source, f"{rope_key}.factor", f"yarn position limit {original} x {shown} exceeds 2^64 - 1")
            return limit, f"yarn rope scaling: {original} original positions x factor {factor:g}"
    return value, where


def _declared_positions(config, source):
    # crit eval's rule (the first positive integer), except that an integer beyond 64 bits it reads is an error
    for key in LENGTH_KEYS:
        item = config.get(key)
        if _eval_int(item):
            _check_64_bits(item, source, key)
            if item > 0:
                return int(item), key
    return None, None


def attention_shape(config, source=CONFIG_NAME):
    """(full-attention layers, query heads, head dim, KV heads), or None without layers, heads or a head dim.

    Full layers: layer_types entries "full_attention" or "attention", else layers // full_attention_interval,
    else every layer. Head dim: head_dim, else hidden_size // heads. KV heads: num_key_value_heads, else heads.
    """
    config = _mapping_of(config, source)
    layers = _integer(config, "num_hidden_layers", source)
    if not layers:
        return None
    types = _strings(config, "layer_types", source)
    if types:
        full = sum(1 for kind in types if kind in ("full_attention", "attention"))
    else:
        interval = _integer(config, "full_attention_interval", source)
        full = layers // interval if interval else layers
    heads = _integer(config, "num_attention_heads", source)
    head_dim = _head_dim(config, source)
    if not heads or not head_dim:
        return None
    return full, heads, head_dim, _integer(config, "num_key_value_heads", source) or heads


def _head_dim(config, source):
    # head_dim, else hidden_size // num_attention_heads; heads and hidden size are read only without a head_dim
    head_dim = _integer(config, "head_dim", source)
    if head_dim:
        return head_dim
    heads = _integer(config, "num_attention_heads", source)
    hidden = _integer(config, "hidden_size", source) if heads else None
    return hidden // heads if heads and hidden else None


def kv_elements_per_token(config, source=CONFIG_NAME):
    """K and V elements one token adds to the cache over every layer whose cache grows with the context, or None.

    Multi-head latent attention (kv_lora_rank) caches the latent plus one shared rope key per layer.
    """
    config = _mapping_of(config, source)
    layers = _integer(config, "num_hidden_layers", source)
    rank = _integer(config, "kv_lora_rank", source)
    if rank and layers:
        return layers * (rank + (_integer(config, "qk_rope_head_dim", source) or 0))
    shape = attention_shape(config, source)
    if not shape:
        return None
    full, _, head_dim, kv_heads = shape
    return 2 * kv_heads * head_dim * full


def fixed_state_bytes(config, kv_bytes_per_element, source=CONFIG_NAME):
    """Bytes of state that does not grow with the context: the linear-attention recurrence (FP32 state, BF16
    convolution window of linear_conv_kernel_dim - 1, default kernel 4) and the KV cache of sliding_attention layers
    at kv_bytes_per_element. Layer kinds come from layer_types only; the linear_* sizes are read only for
    linear_attention layers and sliding_window only for sliding_attention layers."""
    element = kv_bytes_per_element
    if _finite_real(element) is None or element < 0:
        raise ValueError(f"kv_bytes_per_element must be a finite non-negative number, not {_describe(element)}")
    config = _mapping_of(config, source)
    types = _strings(config, "layer_types", source) or ()
    total = 0
    linear = types.count("linear_attention")
    hv = _integer(config, "linear_num_value_heads", source) if linear else None
    dk = _integer(config, "linear_key_head_dim", source) if hv else None
    dv = _integer(config, "linear_value_head_dim", source) if dk else None
    if dv:
        hk = _integer(config, "linear_num_key_heads", source) or hv
        conv = max(0, (_integer(config, "linear_conv_kernel_dim", source) or 4) - 1) * (2 * hk * dk + hv * dv)
        total += linear * (hv * dk * dv * 4 + conv * 2)
    sliding = types.count("sliding_attention")
    head_dim = _head_dim(config, source)   # crit eval computes it for every config
    window = _integer(config, "sliding_window", source) if sliding else None
    if window and head_dim:
        kv_heads = _integer(config, "num_key_value_heads", source) or _integer(config, "num_attention_heads", source)
        if kv_heads is None:
            _fail(source, "num_attention_heads", "missing; sliding_attention layers need a KV head count")
        cache = sliding * 2 * kv_heads * head_dim * window * element   # exact for an integer or rational element
        try:
            total += int(cache)
        except (OverflowError, ValueError):   # an infinite product
            raise ValueError(f"kv_bytes_per_element {_describe(element)} makes the sliding_attention cache size "
                             f"overflow") from None
    return total


# Strict helpers (not in crit eval): errors instead of None.

def layer_kinds(config, source=CONFIG_NAME):
    """One kind per layer: layer_types mapped ("attention" counts as full, unknown names as OTHER) and required to
    have num_hidden_layers entries; else every full_attention_interval-th layer full and the rest linear attention
    (the layer_types Hugging Face derives from that key); else all full. The FULL_ATTENTION count always equals
    attention_shape's full layers."""
    config = _mapping_of(config, source)
    layers = _positive(config, "num_hidden_layers", source)
    if layers > MAX_LAYERS:
        _fail(source, "num_hidden_layers", f"{layers} layers exceed the {MAX_LAYERS}-layer limit")
    types = _strings(config, "layer_types", source)
    if types:
        if len(types) != layers:
            _fail(source, "layer_types", f"{len(types)} entries for {layers} layers")
        return tuple(LAYER_KINDS.get(kind, OTHER) for kind in types)
    interval = _integer(config, "full_attention_interval", source)
    if interval:
        return tuple(FULL_ATTENTION if (index + 1) % interval == 0 else LINEAR_ATTENTION for index in range(layers))
    return (FULL_ATTENTION,) * layers


def effective_head_dim(config, source=CONFIG_NAME):
    """head_dim, else hidden_size / num_attention_heads, which must divide exactly."""
    config = _mapping_of(config, source)
    head_dim = _integer(config, "head_dim", source)
    if head_dim:
        return head_dim
    heads = _positive(config, "num_attention_heads", source, " (no head_dim)")
    hidden = _positive(config, "hidden_size", source, " (no head_dim)")
    if hidden % heads:
        _fail(source, "hidden_size", f"{hidden} is not divisible by num_attention_heads {heads}")
    return hidden // heads


def effective_kv_heads(config, source=CONFIG_NAME):
    """num_key_value_heads, else num_attention_heads; the KV heads must divide the query heads."""
    config = _mapping_of(config, source)
    heads = _positive(config, "num_attention_heads", source)
    kv_heads = _integer(config, "num_key_value_heads", source) or heads
    if heads % kv_heads:
        _fail(source, "num_key_value_heads", f"{kv_heads} does not divide num_attention_heads {heads}")
    return kv_heads


# Typed parsing.

def _rope_fields(rope, prefix, source):
    """rope_type (else type) and the scaling fields of one rope object, as RopeConfig keywords."""
    def real(key):
        return _real(rope, key, source, f"{prefix}.{key}")

    kind = _string(rope, "rope_type", source, f"{prefix}.rope_type") \
        or _string(rope, "type", source, f"{prefix}.type") or ""
    return dict(type=kind, factor=real("factor"), attention_factor=real("attention_factor"),
                beta_fast=real("beta_fast"), beta_slow=real("beta_slow"), low_freq_factor=real("low_freq_factor"),
                high_freq_factor=real("high_freq_factor"), mscale=real("mscale"), mscale_all_dim=real("mscale_all_dim"),
                original_max_position_embeddings=_integer(rope, "original_max_position_embeddings", source,
                                                          f"{prefix}.original_max_position_embeddings",
                                                          maximum=UINT64_END - 1),
                short_factor=_reals(rope, "short_factor", source, f"{prefix}.short_factor"),
                long_factor=_reals(rope, "long_factor", source, f"{prefix}.long_factor"),
                mrope_section=_integers(rope, "mrope_section", source, f"{prefix}.mrope_section"),
                raw=rope or _empty())   # already read-only: the section is frozen


def _rope(section, source):
    """(RopeConfig, rope_by_layer_type). A rope object whose members are all objects is keyed by layer type: each
    entry is read like a rope object of its own, with its own rope_theta and partial_rotary_factor."""
    scaling = _object(section, "rope_scaling", source) or {}
    parameters = _object(section, "rope_parameters", source) or {}
    rope_key, rope = ("rope_scaling", scaling) if scaling else ("rope_parameters", parameters)
    values = _rope_fields(rope, rope_key, source)
    layers = []
    if rope and all(isinstance(entry, Mapping) for entry in rope.values()):
        for layer_type, entry in rope.items():
            prefix = f"{rope_key}.{layer_type}"
            fields_read = _rope_fields(entry, prefix, source)
            layers.append(RopeLayer(layer_type, RopeConfig(
                theta=_real(entry, "rope_theta", source, f"{prefix}.rope_theta"),
                partial_rotary_factor=_real(entry, "partial_rotary_factor", source, f"{prefix}.partial_rotary_factor"),
                **fields_read)))
    theta = _real(section, "rope_theta", source)
    if theta is None:
        theta = _real(parameters, "rope_theta", source, "rope_parameters.rope_theta")
    if theta is None:
        theta = _real(scaling, "rope_theta", source, "rope_scaling.rope_theta")
    partial = _real(section, "partial_rotary_factor", source)
    if partial is None:
        partial = _real(rope, "partial_rotary_factor", source, f"{rope_key}.partial_rotary_factor")
    if values["original_max_position_embeddings"] is None:
        values["original_max_position_embeddings"] = _integer(section, "original_max_position_embeddings", source,
                                                              maximum=UINT64_END - 1)
    return RopeConfig(theta=theta, partial_rotary_factor=partial, **values), tuple(layers)


def _moe(section, source):
    experts, key = _first(section, EXPERT_KEYS, _integer, source)
    if not experts:
        return None
    shared, _ = _first(section, ("n_shared_experts", "num_shared_experts"), _integer, source)
    return MoeConfig(experts=experts, experts_key=key,
                     experts_per_token=_integer(section, "num_experts_per_tok", source) or 0,
                     shared_experts=shared,
                     moe_intermediate_size=_integer(section, "moe_intermediate_size", source),
                     shared_expert_intermediate_size=_integer(section, "shared_expert_intermediate_size", source),
                     first_k_dense_replace=_integer(section, "first_k_dense_replace", source),
                     decoder_sparse_step=_integer(section, "decoder_sparse_step", source),
                     n_group=_integer(section, "n_group", source),
                     topk_group=_integer(section, "topk_group", source),
                     mlp_only_layers=_integers(section, "mlp_only_layers", source),
                     norm_topk_prob=_boolean(section, "norm_topk_prob", source),
                     routed_scaling_factor=_real(section, "routed_scaling_factor", source),
                     scoring_func=_string(section, "scoring_func", source) or "")


def _mla(section, source):
    rank = _integer(section, "kv_lora_rank", source)
    if not rank:
        return None
    return MlaConfig(kv_lora_rank=rank, qk_rope_head_dim=_integer(section, "qk_rope_head_dim", source) or 0,
                     qk_nope_head_dim=_integer(section, "qk_nope_head_dim", source) or 0,
                     v_head_dim=_integer(section, "v_head_dim", source) or 0,
                     q_lora_rank=_integer(section, "q_lora_rank", source))


def _linear(section, source):
    values = {name: _integer(section, "linear_" + name, source)
              for name in ("num_key_heads", "num_value_heads", "key_head_dim", "value_head_dim", "conv_kernel_dim")}
    return LinearAttentionConfig(**values) if any(value is not None for value in values.values()) else None


def _sizes(config, key, source):
    """(integer, ()) or, for a per-layer array such as Gemma 3n's intermediate_size, (None, tuple of integers)."""
    value = config.get(key)
    if isinstance(value, (list, tuple)):
        return None, tuple(_check_int(item, source, f"{key}[{index}]") for index, item in enumerate(value))
    if value is not None and not _is_int(value):
        _fail(source, key, f"expected an integer or an array of integers, got {_describe(value)}")
    return _integer(config, key, source), ()


def _quantization(section, source):
    value = _object(section, "quantization_config", source)
    if value is None:
        return None
    method = _string(value, "quant_method", source, "quantization_config.quant_method") or ""
    return QuantizationConfig(method=method, raw=value)


def parse_model_config(config, source=CONFIG_NAME):
    """A ModelConfig from an already parsed config object; `source` names the file in error messages. The object is
    copied read-only and held to strict JSON: string keys, UTF-8 strings, finite numbers, MAX_CONFIG_DEPTH levels."""
    root = _frozen(_mapping_of(config, source), source)
    text = _object(root, "text_config", source)
    section = MappingProxyType(text_section(root, source))
    max_positions, max_positions_key = _declared_positions(section, source)
    norm_eps, norm_eps_key = _first(section, NORM_EPS_KEYS, _real, source)
    dtype, _ = _first(section, ("dtype", "torch_dtype"), _string, source)
    intermediate_size, intermediate_sizes = _sizes(section, "intermediate_size", source)
    rope, rope_by_layer_type = _rope(section, source)
    return ModelConfig(
        source=source, raw=root, text=text if text is not None else root, section=section,
        architectures=_strings(section, "architectures", source) or (),
        model_type=_string(root, "model_type", source) or "",
        text_model_type=_string(section, "model_type", source) or "",
        vocab_size=_integer(section, "vocab_size", source),
        hidden_size=_integer(section, "hidden_size", source),
        intermediate_size=intermediate_size,
        num_hidden_layers=_integer(section, "num_hidden_layers", source),
        num_attention_heads=_integer(section, "num_attention_heads", source),
        num_key_value_heads=_integer(section, "num_key_value_heads", source),
        head_dim=_integer(section, "head_dim", source),
        max_positions=max_positions, max_positions_key=max_positions_key or "",
        rms_norm_eps=norm_eps, norm_eps_key=norm_eps_key,
        tie_word_embeddings=_boolean(section, "tie_word_embeddings", source),
        dtype=dtype or "",
        rope=rope,
        sliding_window=_integer(section, "sliding_window", source),
        full_attention_interval=_integer(section, "full_attention_interval", source),
        layer_types=_strings(section, "layer_types", source) or (),
        moe=_moe(section, source), mla=_mla(section, source), linear=_linear(section, source),
        num_nextn_predict_layers=_integer(section, "num_nextn_predict_layers", source),
        eos_token_ids=_token_ids(section, "eos_token_id", source),
        bos_token_id=_token_id(section, "bos_token_id", source),
        pad_token_id=_token_id(section, "pad_token_id", source),
        quantization=_quantization(section, source), intermediate_sizes=intermediate_sizes,
        rope_by_layer_type=rope_by_layer_type)


def _parse_json(read):
    try:
        return read()
    except jsonfile.JsonError as error:
        raise ConfigError(str(error)) from None


def load_model_config(source, *, max_bytes=MAX_CONFIG_BYTES):
    """Read config.json from a model directory, a config file path or a ModelFiles source as a ModelConfig.

    Strict JSON, except that the literals NaN, Infinity and -Infinity load where no typed field reads them (as
    transformers writes them); at most max_bytes, checked before the file is read (a ModelFiles source other than
    DirectoryFiles checks what its reader returns). A missing file is a ConfigError, and so is a config.json that
    exists but cannot be opened as a regular file (a dangling or looping link, a directory, no permission). A
    source of any other type raises TypeError.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError(f"max_bytes must be a positive integer, not {_describe(max_bytes)}")
    if not isinstance(source, (str, os.PathLike)):
        files = model_files(source)
        try:
            data = files.read(CONFIG_NAME, max_bytes=max_bytes)
        except ModelFilesError as error:
            raise ConfigError(str(error)) from None
        if data is None:
            raise ConfigError(f"{CONFIG_NAME}: missing from the model files")
        config = _parse_json(lambda: jsonfile.loads(data, CONFIG_NAME, max_bytes=max_bytes, allow_nonfinite=True))
        return parse_model_config(config, CONFIG_NAME)
    path = Path(source)
    if path.is_dir():
        path = path / CONFIG_NAME
    label = str(path)
    try:   # the reader behind DirectoryFiles: absent only when nothing exists by that name
        data = _read_regular(path, label, max_bytes)
    except ModelFilesError as error:
        raise ConfigError(str(error)) from None
    if data is None:
        raise ConfigError(f"{label}: missing")
    config = _parse_json(lambda: jsonfile.loads(data, label, max_bytes=max_bytes, allow_nonfinite=True))
    return parse_model_config(config, label)
