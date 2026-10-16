"""crit-eval-v2's own config.json and stop-id rules, frozen as written before Eval adopted crit_core (protocol
crit-eval-v3). Core's ports (crit_core.model's position_limit, attention_shape, kv_elements_per_token,
fixed_state_bytes and crit_core.text.legacy_eos_ids) are tested against them, so Eval's numbers stay what they were
wherever the rules agree. Test oracle only: nothing in crit imports this module."""
import json
from pathlib import Path

LENGTH_KEYS = ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "max_sequence_length",
               "model_max_length")


def text_config(model):
    path = Path(model) / "config.json"
    if not path.is_file():
        return {}
    config = json.loads(path.read_text(encoding="utf-8"))
    return {**config, **config.get("text_config", {})}


def stop_ids(model):
    ids = set()
    for name in ("generation_config.json", "config.json"):
        path = Path(model) / name
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            for source in (data, data.get("text_config", {})):
                value = source.get("eos_token_id")
                ids.update(value if isinstance(value, list) else [value] if value is not None else [])
    return sorted(ids)


def attention_shape(config):
    layers = config.get("num_hidden_layers")
    if not layers:
        return None
    types = config.get("layer_types")
    if types:
        full = sum(1 for t in types if t in ("full_attention", "attention"))
    elif config.get("full_attention_interval"):
        full = layers // config["full_attention_interval"]
    else:
        full = layers
    heads = config.get("num_attention_heads")
    head_dim = config.get("head_dim") or (config["hidden_size"] // heads if heads and config.get("hidden_size") else None)
    if not heads or not head_dim:
        return None
    return full, heads, head_dim, config.get("num_key_value_heads") or heads


def model_limit(config):
    value, source = None, None
    for key in LENGTH_KEYS:
        item = config.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item > 0:
            value, source = item, key
            break
    rope = config.get("rope_scaling") or config.get("rope_parameters") or {}
    if isinstance(rope, dict) and (rope.get("rope_type") or rope.get("type")) == "yarn":
        original, factor = rope.get("original_max_position_embeddings"), rope.get("factor")
        if isinstance(original, int) and isinstance(factor, (int, float)) and original > 0 and factor > 0:
            return int(original * factor), f"yarn rope scaling: {original} original positions x factor {factor:g}"
    return value, source


def kv_elements_per_token(config):
    layers = config.get("num_hidden_layers")
    if config.get("kv_lora_rank") and layers:
        return layers * (config["kv_lora_rank"] + (config.get("qk_rope_head_dim") or 0))
    shape = attention_shape(config)
    if not shape:
        return None
    full, _, head_dim, kv_heads = shape
    return 2 * kv_heads * head_dim * full


def fixed_state_bytes(config, kv_bytes_per_element):
    types = config.get("layer_types") or []
    total = 0
    linear = types.count("linear_attention")
    hv, hk = config.get("linear_num_value_heads"), config.get("linear_num_key_heads") or config.get("linear_num_value_heads")
    dk, dv = config.get("linear_key_head_dim"), config.get("linear_value_head_dim")
    if linear and hv and dk and dv:
        conv = max(0, (config.get("linear_conv_kernel_dim") or 4) - 1) * (2 * hk * dk + hv * dv)
        total += linear * (hv * dk * dv * 4 + conv * 2)
    sliding = types.count("sliding_attention")
    window = config.get("sliding_window")
    heads = config.get("num_attention_heads")
    head_dim = config.get("head_dim") or (config["hidden_size"] // heads if heads and config.get("hidden_size") else None)
    if sliding and window and head_dim:
        kv_heads = config.get("num_key_value_heads") or heads
        total += int(sliding * 2 * kv_heads * head_dim * window * kv_bytes_per_element)
    return total
