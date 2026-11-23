"""What a GGUF's metadata must agree with in a config.json beside it, for the mechanisms crit_models reads: the
attention layers' KV heads and head sizes by layer kind, the experts and the rotary base. Core's check of a GGUF folder
(crit_core.model.gguf.check_gguf_folder) compares the generic shape and runs these through crit_core.registry
(gguf_checks), which this package registers when it is imported and, installed, by its entry point."""
from crit_core.model.gguf import _array_or, _same_float

from .config import FULL_ATTENTION, SLIDING_ATTENTION, ConfigError, architecture


def compare_attention(model, config, drafts, compare, errors):
    """KV heads and head sizes of the GGUF's attention layers against config.json's per layer kind
    (Architecture.attention_geometry): a per-layer head_count_kv list is compared layer by layer (0 marks a layer
    without attention, draft blocks are not compared), and key_length_swa, where the GGUF has it, is the sliding
    layers' head size beside key_length, the full layers'."""
    get = model.get
    config = architecture(config)
    geometry = config.attention_geometry()
    try:
        kinds = config.layer_kinds() if geometry else None
    except ConfigError:
        kinds = None
    kv = get(".attention.head_count_kv")
    if isinstance(kv, tuple) and kinds and len(kv) in (len(kinds), len(kinds) + drafts):
        for index, (found, kind) in enumerate(zip(kv, kinds)):
            expected = geometry[kind][0] if kind in geometry else None
            if found and expected is not None and found != expected:
                errors.append(f"key/value heads: GGUF {found} at layer {index} ({kind}), config.json {expected}")
                break
    elif geometry and kinds:
        expected = {geometry[kind][0] for kind in set(kinds) if kind in geometry}
        values = {v for v in _array_or(kv) if v} if kv is not None else set()
        if values and expected and not values <= expected:
            errors.append(f"key/value heads: GGUF {sorted(values) if len(values) > 1 else next(iter(values))}, "
                          f"config.json {sorted(expected) if len(expected) > 1 else next(iter(expected))}")
    else:
        compare("key/value heads", kv, config.num_key_value_heads)
    declared = config.head_dim is not None
    full_dim = geometry[FULL_ATTENTION][1] if geometry and (declared or geometry[FULL_ATTENTION][1] != geometry[
        SLIDING_ATTENTION][1]) else config.head_dim
    swa = get(".attention.key_length_swa")
    compare("head size", get(".attention.key_length"), full_dim)
    if swa is not None:
        compare("sliding head size", swa, geometry[SLIDING_ATTENTION][1] if geometry and declared else None)


def compare_mlp(model, config, drafts, compare, errors):
    """The MLP width of a dense decoder, or the experts' count, per-token count and width of a mixture of experts."""
    get, config = model.get, architecture(config)
    if config.moe is None:
        compare("MLP width", get(".feed_forward_length"), config.intermediate_size)
    else:
        compare("experts", get(".expert_count"), config.moe.experts)
        compare("experts per token", get(".expert_used_count"), config.moe.experts_per_token)
        compare("expert MLP width", get(".expert_feed_forward_length"), config.moe.moe_intermediate_size)


def compare_rope(model, config, drafts, compare, errors):
    """The rotary base."""
    found, value = model.get(".rope.freq_base"), architecture(config).rope.theta
    if found is not None and value is not None and not _same_float(found, value):
        errors.append(f"RoPE base: GGUF {found}, config.json {value}")


# crit_core.registry gguf_checks: run in name order.
CHECKS = {"attention": compare_attention, "mlp": compare_mlp, "rope": compare_rope}
