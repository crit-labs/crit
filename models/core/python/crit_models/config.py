"""A decoder's mechanisms as its config.json states them: layer kinds, looped spans, routed experts, latent attention,
recurrent state and rotary positions, and the rules over them that Eval's limits and the model plugins share.

Core reads config.json's generic fields (crit_core.model.config: the text section, sizes, heads, positions, token
ids, quantization); this module reads what a mechanism adds on top of them, and the rules that need it: the layer
kinds a config states (layer_kinds), the layers a token runs through a looped span (layer_applications), the layers
whose MLP routes experts (routed_layers), the attention layers' geometry (attention_shape, attention_geometry), the
KV elements a token adds (kv_elements_per_token) and the state that does not grow with the context
(fixed_state_bytes). architecture(config) gives a ModelConfig together with these: its fields and these sections and
rules as one object. A family that spells a field its own way adds its key to config_keys.json, never a branch in a
rule. Every rule reads values only where it evaluates them and raises ConfigError for a value of the wrong type.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType

from crit_core.model.config import (CONFIG_NAME, MAX_LAYERS, UINT64_END, ConfigError, ModelConfig, _Picklable,
                                    _boolean, _describe, _empty, _fail, _finite_real, _first, _head_dim, _integer,
                                    _integers, _layer_count, _mapping_of, _object, _positive, _real, _reals, _string,
                                    _strings, parse_model_config)

# Which keys spell each mechanism field, in order: config_keys.json.
_KEYS = json.loads((Path(__file__).with_name("config_keys.json")).read_text(encoding="utf-8"))

EXPERT_KEYS = tuple(_KEYS["experts"])
EXPERTS_PER_TOKEN_KEYS = tuple(_KEYS["experts_per_token"])
FULL_KV_HEAD_KEYS = tuple(_KEYS["full_attention_kv_heads"])
FULL_HEAD_DIM_KEYS = tuple(_KEYS["full_attention_head_dim"])
FULL_KEYS_EQUAL_VALUES_KEYS = tuple(_KEYS["full_attention_keys_equal_values"])
LAYER_TYPE_KEYS = tuple(_KEYS["layer_types"])
SLIDING_LAYER_ID_KEYS = tuple(_KEYS["sliding_layer_ids"])
SLIDING_WINDOW_KEYS = tuple(_KEYS["sliding_window"])
ROUTED_SCALING_KEYS = tuple(_KEYS["routed_scaling_factor"])
LAYER_PASS_KEYS = tuple(MappingProxyType(dict(entry)) for entry in _KEYS["layer_passes"])
LAYER_TYPE_NAMES = MappingProxyType(dict(_KEYS["layer_type_names"]))
FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
LINEAR_ATTENTION = "linear_attention"
OTHER = "other"
LAYER_KINDS = MappingProxyType({"full_attention": FULL_ATTENTION, "attention": FULL_ATTENTION,
                                "sliding_attention": SLIDING_ATTENTION, "linear_attention": LINEAR_ATTENTION})


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
    """One entry of a rope object keyed by layer type, such as "sliding_attention" (transformers v5's per-layer-type rope):
    its own rope_type, scaling fields, rope_theta and partial_rotary_factor; nothing falls back to the section."""
    layer_type: str = ""
    rope: RopeConfig = field(default_factory=RopeConfig)


@dataclass(frozen=True)
class MoeConfig:
    """Mixture-of-experts fields; experts_key names the key the expert count came from. A field config.json leaves
    out is None, 0 or empty here (norm_topk_prob None, scoring_func ""): Core fills in no family's default, so a
    model plugin states what its reference assumes for it (the reference's own config class has the default)."""
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
    topk_method: str = ""


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


def _layer_types(config, source):
    # The first LAYER_TYPE_KEYS entry the config has (layers_block_type: "mamba", "moe", "mlp", "attention"); else,
    # where the config lists the layers that attend a sliding window (SLIDING_LAYER_ID_KEYS), "sliding_attention" for
    # those and "full_attention" for the others.
    key = next((key for key in LAYER_TYPE_KEYS if key in config), None)
    if key:   # a family's own names of the types read as the names they stand for (config_keys.json)
        found = _strings(config, key, source)
        return None if found is None else tuple(LAYER_TYPE_NAMES.get(name, name) for name in found)
    key = next((key for key in SLIDING_LAYER_ID_KEYS if key in config), None)
    ids = _integers(config, key, source) if key else None
    if ids is None:
        return None
    layers = _integer(config, "num_hidden_layers", source)
    if not layers:
        _fail(source, key, "lists sliding-window layers of a config without num_hidden_layers")
    if any(index >= layers for index in ids) or len(set(ids)) != len(ids):
        _fail(source, key, f"names layers outside 0..{layers - 1} or one twice")
    sliding = set(ids)
    return tuple(SLIDING_ATTENTION if index in sliding else FULL_ATTENTION for index in range(layers))


def _sliding_window(config, source):
    # The first SLIDING_WINDOW_KEYS entry the config has.
    return _first(config, SLIDING_WINDOW_KEYS, _integer, source)[0]


def _layer_passes(config, source):
    """((start, end, passes), ...): the spans of layers a looped decoder runs more than once (LAYER_PASS_KEYS), in
    order, without overlaps and inside the layers; () where the config has none."""
    spans = []
    for spec in LAYER_PASS_KEYS:
        if spec["object"] is None:   # the config's own keys
            value, label = (config, "") if spec["start"] in config else (None, "")
        else:
            value, label = _object(config, spec["object"], source), spec["object"] + "."
        if value is None:
            continue
        start = _integer(value, spec["start"], source, label + spec["start"])
        end = _integer(value, spec["end"], source, label + spec["end"])
        passes = _integer(value, spec["passes"], source, label + spec["passes"])
        if start is None or end is None or passes is None:
            _fail(source, spec["object"] or spec["start"], f"needs {spec['start']}, {spec['end']} and "
                                                           f"{spec['passes']}")
        spans.append((start, end, passes))
    if not spans:
        return ()
    layers = _layer_count(config, source)
    spans.sort()
    for i, (start, end, passes) in enumerate(spans):
        if not (0 <= start < end) or (layers and end > layers) or passes < 1 or \
                (i and start < spans[i - 1][1]):
            _fail(source, "layer passes", f"span [{start}, {end}) x {passes} is empty, overlaps another or lies "
                                          f"outside the {layers} layers")
    return tuple(spans)


def _passes_of(config, source):
    # How many times each layer runs a token: 1, or a looped span's passes.
    layers = _layer_count(config, source) or 0
    runs = [1] * layers
    for start, end, passes in _layer_passes(config, source):
        for index in range(start, min(end, layers)):
            runs[index] = passes
    return runs


def attention_shape(config, source=CONFIG_NAME):
    """(full-attention layers, query heads, head dim, KV heads), or None without layers, heads or a head dim.

    Full layers: layer_types entries "full_attention" or "attention", else layers // full_attention_interval,
    else every layer. Head dim and KV heads are the full-attention layers' (attention_geometry): a
    full_attention_head_dim / full_attention_kv_heads key (config_keys.json) where the config declares one, else
    head_dim (else hidden_size // heads) and num_key_value_heads (else heads).
    """
    config = _mapping_of(config, source)
    layers = _layer_count(config, source)
    if not layers:
        return None
    types = _layer_types(config, source)
    if types:
        full = sum(1 for kind in types if kind in ("full_attention", "attention"))
    else:
        interval = _integer(config, "full_attention_interval", source)
        full = layers // interval if interval else layers
    heads = _integer(config, "num_attention_heads", source)
    head_dim = _head_dim(config, source)
    if not heads or not head_dim:
        return None
    kv_heads = _integer(config, "num_key_value_heads", source) or heads
    full_kv_heads, full_head_dim, _ = _full_geometry(config, source, kv_heads, head_dim)
    return full, heads, full_head_dim, full_kv_heads


def _full_geometry(config, source, kv_heads, head_dim):
    # (KV heads, head dim, cached tensors) of full-attention layers: their own keys where declared, else the others'
    own_heads = _first(config, FULL_KV_HEAD_KEYS, _integer, source)[0]
    own_dim = _first(config, FULL_HEAD_DIM_KEYS, _integer, source)[0]
    keys_only = _first(config, FULL_KEYS_EQUAL_VALUES_KEYS, _boolean, source)[0]
    return own_heads or kv_heads, own_dim or head_dim, 1 if keys_only else 2


def attention_geometry(config, source=CONFIG_NAME):
    """{layer kind: (KV heads, head dim, cached tensors per token)} for FULL_ATTENTION and SLIDING_ATTENTION layers,
    or None without query heads or a head dim. Sliding layers: num_key_value_heads (else num_attention_heads) and
    head_dim (else hidden_size // heads), keys and values cached apart (2 tensors). Full layers: the same, except where
    the config declares their own (config_keys.json full_attention_kv_heads, full_attention_head_dim), and one cached
    tensor where full_attention_keys_equal_values is true (keys derived from the cached values)."""
    config = _mapping_of(config, source)
    heads = _integer(config, "num_attention_heads", source)
    head_dim = _head_dim(config, source)
    if not heads or not head_dim:
        return None
    kv_heads = _integer(config, "num_key_value_heads", source) or heads
    return {FULL_ATTENTION: _full_geometry(config, source, kv_heads, head_dim),
            SLIDING_ATTENTION: (kv_heads, head_dim, 2)}


def kv_elements_per_token(config, source=CONFIG_NAME):
    """K and V elements one token adds to the cache over every layer whose cache grows with the context, or None.

    Multi-head latent attention (kv_lora_rank) caches the latent plus one shared rope key per layer. Full-attention
    layers cache their own geometry (attention_geometry), one tensor where their keys equal their values. A layer a
    looped decoder runs several times (layer_passes) keeps a cache for each pass.
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
    runs = _passes_of(config, source)
    types = _layer_types(config, source)
    if types and any(count > 1 for count in runs):
        full = sum(count for kind, count in zip(types, runs) if kind in ("full_attention", "attention"))
    elif any(count > 1 for count in runs):
        full = sum(runs) if full == len(runs) else full
    tensors = attention_geometry(config, source)[FULL_ATTENTION][2]
    return tensors * kv_heads * head_dim * full


def fixed_state_bytes(config, kv_bytes_per_element, source=CONFIG_NAME, sliding=True):
    """Bytes of state that does not grow with the context: the linear-attention recurrence (FP32 state, BF16
    convolution window of linear_conv_kernel_dim - 1, default kernel 4), the Mamba-2 recurrence of "mamba" layers
    (FP32 state, BF16 convolution window of conv_kernel - 1) and, with `sliding`, the KV cache of sliding_attention
    layers at kv_bytes_per_element (one window; without it, a caller that prices the rings itself: an EngineSpec's
    kv_ring_bytes). Layer kinds come from layer_types (or the sliding layers' ids); the linear_* sizes are
    read only for linear_attention layers and the sliding window only for sliding_attention layers. A layer a looped
    decoder runs several times (layer_passes) keeps its state once a pass."""
    element = kv_bytes_per_element
    if _finite_real(element) is None or element < 0:
        raise ValueError(f"kv_bytes_per_element must be a finite non-negative number, not {_describe(element)}")
    config = _mapping_of(config, source)
    types = _layer_types(config, source) or ()
    runs = _passes_of(config, source)
    runs = runs if len(runs) == len(types) else [1] * len(types)

    def count(kind):
        return sum(n for name, n in zip(types, runs) if name == kind)

    total = 0
    mamba = count("mamba")
    m_heads = _integer(config, "mamba_num_heads", source) if mamba else None
    m_dim = _integer(config, "mamba_head_dim", source) if m_heads else None
    m_state = _integer(config, "ssm_state_size", source) if m_dim else None
    if m_state:
        # Mamba-2: FP32 state [heads][head_dim][state], BF16 window of conv_kernel - 1 over x, B and C.
        channels = m_heads * m_dim + 2 * (_integer(config, "n_groups", source) or 1) * m_state
        conv = max(0, (_integer(config, "conv_kernel", source) or 4) - 1) * channels
        total += mamba * (m_heads * m_dim * m_state * 4 + conv * 2)
    linear = count("linear_attention")
    hv = _integer(config, "linear_num_value_heads", source) if linear else None
    dk = _integer(config, "linear_key_head_dim", source) if hv else None
    dv = _integer(config, "linear_value_head_dim", source) if dk else None
    if dv:
        hk = _integer(config, "linear_num_key_heads", source) or hv
        conv = max(0, (_integer(config, "linear_conv_kernel_dim", source) or 4) - 1) * (2 * hk * dk + hv * dv)
        total += linear * (hv * dk * dv * 4 + conv * 2)
    sliding = count("sliding_attention") if sliding else 0
    head_dim = _head_dim(config, source)   # crit eval computes it for every config
    window = _sliding_window(config, source) if sliding else None
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


def layer_kinds(config, source=CONFIG_NAME):
    """One kind per layer: layer_types mapped ("attention" counts as full, unknown names as OTHER) and required to
    have num_hidden_layers entries; else every full_attention_interval-th layer full and the rest linear attention
    (the layer_types Hugging Face derives from that key); else all full. layer_types may be the list of the layers that
    attend a sliding window (config_keys.json sliding_layer_ids). The FULL_ATTENTION count always equals
    attention_shape's full layers."""
    config = _mapping_of(config, source)
    layers = _layer_count(config, source) if "num_hidden_layers" not in config else None
    if not layers:
        layers = _positive(config, "num_hidden_layers", source)
    if layers > MAX_LAYERS:
        _fail(source, "num_hidden_layers", f"{layers} layers exceed the {MAX_LAYERS}-layer limit")
    types = _layer_types(config, source)
    if types:
        if len(types) != layers:
            _fail(source, "layer_types", f"{len(types)} entries for {layers} layers")
        return tuple(LAYER_KINDS.get(kind, OTHER) for kind in types)
    interval = _integer(config, "full_attention_interval", source)
    if interval:
        return tuple(FULL_ATTENTION if (index + 1) % interval == 0 else LINEAR_ATTENTION for index in range(layers))
    return (FULL_ATTENTION,) * layers


def routed_layers(config, source=CONFIG_NAME):
    """The indices of the layers whose MLP routes experts, in order (() for a model without routed experts): every
    layer from first_k_dense_replace on whose (index + 1) is a multiple of decoder_sparse_step (default 1), less the
    mlp_only_layers, as the checkpoints' reference code places its sparse blocks."""
    config = _mapping_of(config, source)
    if _moe(config, source) is None:
        return ()
    layers = _layer_count(config, source) if "num_hidden_layers" not in config else None
    if not layers:
        layers = _positive(config, "num_hidden_layers", source)
    dense = _integer(config, "first_k_dense_replace", source) or 0
    step = _integer(config, "decoder_sparse_step", source) or 1
    dense_only = set(_integers(config, "mlp_only_layers", source) or ())
    return tuple(i for i in range(dense, layers) if (i + 1) % step == 0 and i not in dense_only)


def layer_applications(config, source=CONFIG_NAME):
    """The checkpoint layer each layer application of a token runs, in order: range(num_hidden_layers) for a plain
    decoder; for a looped one (layer_passes) each looped span repeated its passes in a row, so a span [6, 18) of 3
    passes runs 0..5, 6..17, 6..17, 6..17, 18.. . The applications of one layer share its weights and keep their own
    state (KV, convolution windows) each."""
    config = _mapping_of(config, source)
    layers = _layer_count(config, source) if "num_hidden_layers" not in config else None
    if not layers:
        layers = _positive(config, "num_hidden_layers", source)
    order, index = [], 0
    for start, end, passes in _layer_passes(config, source):
        order.extend(range(index, start))
        order.extend(list(range(start, end)) * passes)
        index = end
    order.extend(range(index, layers))
    return tuple(order)


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
                     experts_per_token=_first(section, EXPERTS_PER_TOKEN_KEYS, _integer, source)[0] or 0,
                     shared_experts=shared,
                     moe_intermediate_size=_integer(section, "moe_intermediate_size", source),
                     shared_expert_intermediate_size=_integer(section, "shared_expert_intermediate_size", source),
                     first_k_dense_replace=_integer(section, "first_k_dense_replace", source),
                     decoder_sparse_step=_integer(section, "decoder_sparse_step", source),
                     n_group=_integer(section, "n_group", source),
                     topk_group=_integer(section, "topk_group", source),
                     mlp_only_layers=_integers(section, "mlp_only_layers", source),
                     norm_topk_prob=_boolean(section, "norm_topk_prob", source),
                     routed_scaling_factor=_first(section, ROUTED_SCALING_KEYS, _real, source)[0],
                     scoring_func=_string(section, "scoring_func", source) or "",
                     topk_method=_string(section, "topk_method", source) or "")


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


# -- one object --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Architecture:
    """A ModelConfig (`config`) with the mechanism sections its text section states, and the rules over them. Every
    field of the ModelConfig reads through it (architecture.hidden_size is config.hidden_size). rope_by_layer_type
    holds one RopeLayer per entry of a rope object keyed by layer type, in file order; layer_passes the looped spans
    ((start, end, passes), ...)."""
    config: ModelConfig
    rope: RopeConfig = field(default_factory=RopeConfig)
    rope_by_layer_type: tuple = ()
    sliding_window: int | None = None
    full_attention_interval: int | None = None
    layer_types: tuple = ()
    moe: MoeConfig | None = None
    mla: MlaConfig | None = None
    linear: LinearAttentionConfig | None = None
    num_nextn_predict_layers: int | None = None
    layer_passes: tuple = ()

    def __getattr__(self, name):
        if name.startswith("__") or name == "config":
            raise AttributeError(name)
        return getattr(self.config, name)

    def attention_shape(self):
        return attention_shape(self.section, source=self.source)

    def attention_geometry(self):
        return attention_geometry(self.section, source=self.source)

    def layer_kinds(self):
        return layer_kinds(self.section, source=self.source)

    def kv_elements_per_token(self):
        return kv_elements_per_token(self.section, source=self.source)

    def fixed_state_bytes(self, kv_bytes_per_element, sliding=True):
        return fixed_state_bytes(self.section, kv_bytes_per_element, source=self.source, sliding=sliding)

    def routed_layers(self):
        return routed_layers(self.section, source=self.source)

    def layer_applications(self):
        return layer_applications(self.section, source=self.source)


def architecture(config, source=CONFIG_NAME):
    """The Architecture of a ModelConfig, an Architecture (returned as it is) or a parsed config mapping."""
    if isinstance(config, Architecture):
        return config
    if not isinstance(config, ModelConfig):
        config = parse_model_config(config, source)
    section, where = config.section, config.source
    rope, rope_by_layer_type = _rope(section, where)
    return Architecture(config=config, rope=rope, rope_by_layer_type=rope_by_layer_type,
                        sliding_window=_sliding_window(section, where),
                        full_attention_interval=_integer(section, "full_attention_interval", where),
                        layer_types=_layer_types(section, where) or (),
                        moe=_moe(section, where), mla=_mla(section, where), linear=_linear(section, where),
                        num_nextn_predict_layers=_integer(section, "num_nextn_predict_layers", where),
                        layer_passes=_layer_passes(section, where))


__all__ = ["Architecture", "ConfigError", "FULL_ATTENTION", "LINEAR_ATTENTION", "LinearAttentionConfig", "MlaConfig",
           "MoeConfig", "OTHER", "RopeConfig", "RopeLayer", "SLIDING_ATTENTION", "architecture", "attention_geometry",
           "attention_shape", "fixed_state_bytes", "kv_elements_per_token", "layer_applications", "layer_kinds",
           "routed_layers"]
