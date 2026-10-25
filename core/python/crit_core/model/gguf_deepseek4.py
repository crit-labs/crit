"""config.json for a DeepSeek V4 GGUF (general.architecture "deepseek4") from its metadata alone (standard library).

The inverse of DeepseekV4Model.set_gguf_parameters in GGUF's reference converter (conversion/deepseek.py), written in
the keys of DeepSeek's released config.json, which the converter reads and transformers' DeepseekV4Config folds into
its own: compress_ratios (one per layer: 0 sliding window, 4 compressed sparse, 128 heavily compressed attention)
into layer_types, num_hash_layers into mlp_layer_types, qk_rope_head_dim into partial_rotary_factor, rope_scaling
into the "compress" rope (the compressed layers' YaRN at compress_rope_theta; sliding layers rotate at rope_theta).

What transformers' modeling code fixes for the architecture, the GGUF must say too, or it is refused: one KV head
shared by every query head, one shared expert, top-k weights always renormalised, one SwiGLU clamp for routed and
shared experts. So is metadata no rule here reads (a key a later converter adds means something this does not say).
The converter does not export DeepSeek's MTP layers by default; a GGUF that holds them (nextn_predict_layers)
declares num_nextn_predict_layers, which transformers never instantiates.
"""
from .gguf import GgufError
from .gguf_files import _base

# deepseek4.expert_gating_func (gguf-py ExpertGatingFuncType) -> config.json scoring_func.
SCORING = {1: "softmax", 2: "sigmoid", 4: "sqrtsoftplus"}
# compress_ratios values transformers maps to layer types.
RATIOS = (0, 4, 128)
# Every deepseek4.* key set_gguf_parameters writes (and the generic ones it might): any other is refused.
KNOWN = {
    ".block_count", ".context_length", ".embedding_length", ".embedding_length_out", ".feed_forward_length",
    ".vocab_size", ".nextn_predict_layers",
    ".attention.head_count", ".attention.head_count_kv", ".attention.key_length", ".attention.value_length",
    ".attention.layer_norm_rms_epsilon", ".attention.q_lora_rank", ".attention.sliding_window",
    ".attention.output_group_count", ".attention.output_lora_rank", ".attention.compress_ratios",
    ".attention.compress_rope_freq_base", ".attention.indexer.head_count", ".attention.indexer.key_length",
    ".attention.indexer.top_k",
    ".rope.freq_base", ".rope.dimension_count", ".rope.scaling.type", ".rope.scaling.factor",
    ".rope.scaling.original_context_length", ".rope.scaling.yarn_beta_fast", ".rope.scaling.yarn_beta_slow",
    ".rope.scaling.yarn_attn_factor",
    ".expert_count", ".expert_used_count", ".expert_gating_func", ".expert_feed_forward_length",
    ".expert_shared_count", ".expert_weights_scale", ".expert_weights_norm", ".swiglu_clamp_exp",
    ".swiglu_clamp_shexp",
    ".hyper_connection.count", ".hyper_connection.sinkhorn_iterations", ".hyper_connection.epsilon",
    ".hash_layer_count",
}


def _rope_scaling(get):
    """rope_scaling as DeepSeek's config.json writes it, or None for no scaling."""
    kind = get(".rope.scaling.type")
    if kind in (None, "none"):
        return None
    if kind != "yarn":
        raise GgufError(f"deepseek4.rope.scaling.type {kind!r}: transformers' DeepSeek V4 scales its compressed "
                        "layers' rope with YaRN only")
    scaling = {"type": "yarn", "factor": get.require(".rope.scaling.factor", "YaRN factor"),
               "original_max_position_embeddings": get.require(".rope.scaling.original_context_length",
                                                               "YaRN original context")}
    for key, name in ((".rope.scaling.yarn_beta_fast", "beta_fast"), (".rope.scaling.yarn_beta_slow", "beta_slow"),
                      (".rope.scaling.yarn_attn_factor", "attention_factor")):
        if get(key) is not None:
            scaling[name] = get(key)
    return scaling


def _array(get, key, label):
    value = get(key)
    if value is None:
        raise GgufError(f"the GGUF has no deepseek4{key}; config.json cannot be derived without its {label}")
    return list(value) if isinstance(value, tuple) else [value]


def _clamp(get):
    """swiglu_limit: the one clamp every layer's routed and shared experts take (transformers has one)."""
    values = set(_array(get, ".swiglu_clamp_exp", "routed-expert SwiGLU clamp"))
    values |= set(_array(get, ".swiglu_clamp_shexp", "shared-expert SwiGLU clamp"))
    if len(values) != 1 or not next(iter(values)) > 0:
        raise GgufError(f"deepseek4.swiglu_clamp_exp/_shexp hold {sorted(values)}; transformers' DeepSeek V4 clamps "
                        "every routed and shared expert at one positive swiglu_limit")
    return values.pop()


def deepseek4_config(model, get):
    """config.json for a deepseek4 GgufModel (get: crit_core.model.gguf_files._Reader over it), as a dict."""
    unknown = sorted(key for key in model.metadata
                     if key.startswith(f"{model.architecture}.") and key[len(model.architecture):] not in KNOWN)
    if unknown:
        raise GgufError(f"the GGUF has DeepSeek V4 metadata the config rules do not state ({', '.join(unknown)})")
    config, _, drafts = _base(model, get)
    experts = get.require(".expert_feed_forward_length", "expert MLP width")
    # transformers aliases intermediate_size to moe_intermediate_size: a dense width would overwrite it.
    dense = config.pop("intermediate_size", None)
    if dense is not None and dense != experts:
        raise GgufError(f"deepseek4.feed_forward_length {dense} differs from the expert width {experts}; DeepSeek V4 "
                        "has no dense MLP")
    if config["num_key_value_heads"] != 1:
        raise GgufError(f"deepseek4.attention.head_count_kv {config['num_key_value_heads']}: transformers' DeepSeek V4 "
                        "shares one key/value head among all query heads")
    if get.require(".expert_shared_count", "shared expert count") != 1:
        raise GgufError("transformers' DeepSeek V4 has exactly one shared expert; the GGUF declares "
                        f"{get('.expert_shared_count')}")
    if get.require(".expert_weights_norm", "expert weight normalisation") is not True:
        raise GgufError("the GGUF does not renormalise the top-k expert weights; transformers' DeepSeek V4 routers "
                        "always do")
    gating = get.require(".expert_gating_func", "expert gating function")
    if gating not in SCORING:
        raise GgufError(f"deepseek4.expert_gating_func {gating} has no transformers scoring_func")
    count = config["num_hidden_layers"]
    ratios = _array(get, ".attention.compress_ratios", "compress ratios")
    if len(ratios) < count or set(ratios[:count]) - set(RATIOS):
        raise GgufError(f"deepseek4.attention.compress_ratios {ratios} does not give each of {count} layers one of "
                        f"{RATIOS}")
    streams = get.require(".hyper_connection.count", "hyper-connection streams")
    out_width = get(".embedding_length_out")
    if out_width is not None and out_width != config["hidden_size"] * streams:
        raise GgufError(f"deepseek4.embedding_length_out {out_width} is not {streams} streams of "
                        f"{config['hidden_size']}")
    hashed = get.require(".hash_layer_count", "hash-routed layer count")
    if hashed > count:
        raise GgufError(f"deepseek4.hash_layer_count {hashed} exceeds the {count} layers")
    out = {"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4", **config,
           "attention_bias": False,
           "q_lora_rank": get.require(".attention.q_lora_rank", "query LoRA rank"),
           "qk_rope_head_dim": get.require(".rope.dimension_count", "rotary dimensions"),
           "o_groups": get.require(".attention.output_group_count", "output groups"),
           "o_lora_rank": get.require(".attention.output_lora_rank", "output LoRA rank"),
           "sliding_window": get.require(".attention.sliding_window", "sliding window"),
           "compress_ratios": ratios,
           "index_n_heads": get.require(".attention.indexer.head_count", "indexer heads"),
           "index_head_dim": get.require(".attention.indexer.key_length", "indexer head size"),
           "index_topk": get.require(".attention.indexer.top_k", "indexer top-k"),
           "rope_theta": get.require(".rope.freq_base", "RoPE base"),
           "compress_rope_theta": get.require(".attention.compress_rope_freq_base", "compressed RoPE base"),
           "n_routed_experts": get.require(".expert_count", "expert count"),
           "num_experts_per_tok": get.require(".expert_used_count", "experts per token"),
           "moe_intermediate_size": experts, "n_shared_experts": 1, "norm_topk_prob": True,
           "scoring_func": SCORING[gating],
           "routed_scaling_factor": get.require(".expert_weights_scale", "routed scaling factor"),
           "swiglu_limit": _clamp(get), "num_hash_layers": hashed,
           "hc_mult": streams,
           "hc_sinkhorn_iters": get.require(".hyper_connection.sinkhorn_iterations", "Sinkhorn iterations"),
           "hc_eps": get.require(".hyper_connection.epsilon", "hyper-connection epsilon")}
    scaling = _rope_scaling(get)
    if scaling is not None:
        out["rope_scaling"] = scaling
    if drafts:
        out["num_nextn_predict_layers"] = drafts
    return out
