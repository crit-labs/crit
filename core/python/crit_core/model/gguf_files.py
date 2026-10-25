"""The description files of a GGUF, derived from its own metadata: config.json, tokenizer.json, tokenizer_config.json,
the chat templates and generation_config.json, so a lone .gguf is a whole model to crit (standard library only).

GGUF's reference converter writes everything a runtime needs into the GGUF: the shapes and hyperparameters under
<architecture>.*, the vocabulary, merges and special tokens under tokenizer.ggml.*, the chat templates under
tokenizer.chat_template*. Each file here is the Hugging Face form of exactly that, for the GGUF's meaning as
its reference runtime defines it (where that runtime fixes a behaviour for an architecture, such as Qwen3 MoE always
renormalising its top-k expert weights, the config states it):

    config.json             CONFIG_RULES per general.architecture: the inverse of the converter's
                            set_gguf_parameters for that architecture, with model_type and architectures for
                            transformers. Another architecture gets the generic shape keys (layers, width, heads,
                            head size, MLP width, experts, context, RoPE base, vocabulary) when its metadata holds
                            nothing the generic keys cannot say (sliding windows, latent attention, recurrent
                            layers, ...), and is refused otherwise.
    tokenizer.json          byte-level BPE (tokenizer.ggml.model "gpt2") with the pre-tokenizer PRE_TOKENIZERS
                            gives tokenizer.ggml.pre: the regex, normalizer and ignore_merges of the Hugging Face
                            tokenizer.json the converter read. Other tokenizer models are refused.
    tokenizer_config.json   special tokens and add_bos/add_eos.
    chat_template.jinja     tokenizer.chat_template; named templates (tokenizer.chat_template.<name>) go to
                            additional_chat_templates/<name>.jinja.
    generation_config.json  BOS, padding, the end-of-generation ids (EOS, EOT, EOM and the control tokens the runtime
                            ends generation on) and general.sampling.* defaults.

FLOAT32 values become the shortest decimal that rounds to the stored float, which is the decimal config.json held.
Every derivation is tested against the original files of models converted by the reference converter, and the
tokenizer against its own vocabulary test sets (Hugging Face tokenizations of hard inputs).
"""
import json
import math
import struct

from .gguf import GgufError

# tokenizer.ggml.token_type values (the reference runtime's llama_token_type).
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6

_QWEN2 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
          r"\s+(?!\S)|\s+")
_QWEN35 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|"
           r"\s*[\r\n]+|\s+(?!\S)|\s+")
_LLAMA3 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
           r"\s*[\r\n]+|\s+(?!\S)|\s+")
# DeepSeek V3's splits in turn (digits in threes, CJK runs, then words and punctuation), which DeepSeek V4 shares.
_DEEPSEEK3 = (r"\p{N}{1,3}", r"[一-龥぀-ゟ゠-ヿ]+",
              r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|"""
              r" ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
# tokenizer.ggml.pre -> (Split regex, a tuple of Split regexes applied in turn, or None for GPT-2's own ByteLevel
# regex; normalizer; ignore_merges), as the reference runtime's llama_vocab::load groups the names.
PRE_TOKENIZERS = {
    "qwen2": (_QWEN2, "NFC", False), "deepseek-r1-qwen": (_QWEN2, "NFC", False), "megrez": (_QWEN2, "NFC", False),
    "qwen35": (_QWEN35, None, False),
    "llama-bpe": (_LLAMA3, None, True), "llama3": (_LLAMA3, None, True), "llama-v3": (_LLAMA3, None, True),
    "deepseek-v3": (_DEEPSEEK3, None, False), "joyai-llm": (_DEEPSEEK3, None, False),
    "gpt-2": (None, None, False),
}
# Control tokens the reference runtime ends generation on by their text (llama_vocab::load), when the file has them.
END_OF_GENERATION = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<end_of_turn>",
                     "<|end|>")
SAMPLING = (("general.sampling.temp", "temperature"), ("general.sampling.top_k", "top_k"),
            ("general.sampling.top_p", "top_p"), ("general.sampling.min_p", "min_p"),
            ("general.sampling.penalty_repeat", "repetition_penalty"))


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def decimal(value, value_type="FLOAT32"):
    """A GGUF number as config.json writes it: FLOAT32 as the shortest decimal that rounds to it (1e-06, not
    9.999999974752427e-07)."""
    if value_type != "FLOAT32" or not isinstance(value, float) or not math.isfinite(value):
        return value
    for digits in range(1, 10):
        candidate = float(f"{value:.{digits}g}")
        if _f32(candidate) == _f32(value):
            return candidate
    return value


class _Reader:
    """The GGUF's metadata with its value types, read as config.json values."""

    def __init__(self, model):
        self.model = model
        self.arch = model.architecture
        self.types = model.files[0].value_types if model.files else {}

    def __call__(self, key, default=None):
        full = f"{self.arch}{key}" if key.startswith(".") else key
        value = self.model.metadata.get(full)
        if value is None:
            return default
        kind = self.types.get(full, "")
        if isinstance(value, tuple):
            element = kind[len("ARRAY["):-1] if kind.startswith("ARRAY[") else ""
            return tuple(decimal(v, element) for v in value)
        return decimal(value, kind)

    def has(self, key):
        return (f"{self.arch}{key}" if key.startswith(".") else key) in self.model.metadata

    def one(self, key, default=None, label=None):
        """A per-model value that may be written per layer: the one value its layers share (0 marks a layer without
        the part, such as attention heads in a recurrent layer)."""
        value = self(key, default)
        if not isinstance(value, tuple):
            return value
        values = {v for v in value if v}
        if len(values) != 1:
            raise GgufError(f"{self.arch}{key} differs between layers ({sorted(values)}); the config rules take "
                            f"one {label or key[1:]} for every layer")
        return values.pop()

    def require(self, key, label=None):
        value = self.one(key, label=label)
        if value is None:
            raise GgufError(f"the GGUF has no {self.arch}{key}; config.json cannot be derived without it")
        return value


def _tensor_f32(tensor):
    """The values of a small F32 tensor (rope_freqs), read from its file."""
    if tensor.dtype != "F32" or tensor.nbytes > 1 << 20:
        raise GgufError(f"{tensor.name}: expected a small F32 tensor, found {tensor.dtype} of {tensor.nbytes} bytes")
    with open(tensor.shard, "rb") as handle:
        handle.seek(tensor.data_begin)
        data = handle.read(tensor.nbytes)
    if len(data) != tensor.nbytes:
        raise GgufError(f"{tensor.name}: the file ends inside the tensor")
    return struct.unpack(f"<{tensor.nbytes // 4}f", data)


def _llama3_rope(factors, theta, head_dim):
    """The rope_parameters of Llama 3's frequency scaling whose per-dimension factors (the GGUF's rope_freqs tensor,
    written by LlamaModel.generate_extra_tensors) are `factors`: the factor is their largest value; the original
    context and the low and high frequency factors are the released values that reproduce every factor."""
    factor = max(factors)
    freqs = [1.0 / (theta ** (i / head_dim)) for i in range(0, head_dim, 2)]
    for original in (8192, 4096, 2048, 16384, 32768, 65536, 131072):
        for low in (1.0, 2.0, 0.5):
            for high in (4.0, 8.0, 2.0, 1.0):
                if high <= low:
                    continue
                expected = []
                for freq in freqs:
                    wavelen = 2 * math.pi / freq
                    if wavelen < original / high:
                        expected.append(1.0)
                    elif wavelen > original / low:
                        expected.append(factor)
                    else:
                        smooth = (original / wavelen - low) / (high - low)
                        expected.append(1 / ((1 - smooth) / factor + smooth))
                if len(expected) == len(factors) and all(math.isclose(a, b, rel_tol=1e-5)
                                                         for a, b in zip(expected, factors)):
                    return {"rope_type": "llama3", "factor": decimal(factor), "low_freq_factor": low,
                            "high_freq_factor": high, "original_max_position_embeddings": original}
    raise GgufError("rope_freqs.weight holds frequency factors no Llama 3 rope scaling reproduces; the config rules "
                    "cannot state them")


def _rope(get, model, head_dim, rotary_dim=None):
    """rope_parameters from <arch>.rope.*: the base, and linear, YaRN or (with a rope_freqs tensor) Llama 3 scaling."""
    rope = {"rope_type": "default", "rope_theta": get(".rope.freq_base", 10000.0)}
    kind = get(".rope.scaling.type")
    if kind in ("linear", "yarn"):
        rope.update(rope_type=kind, factor=get(".rope.scaling.factor"))
        if kind == "yarn":
            rope["original_max_position_embeddings"] = get(".rope.scaling.original_context_length")
            for key, name in ((".rope.scaling.yarn_beta_fast", "beta_fast"), (".rope.scaling.yarn_beta_slow",
                              "beta_slow"), (".rope.scaling.yarn_attn_factor", "attention_factor"),
                              (".rope.scaling.yarn_ext_factor", "extrapolation_factor"),
                              (".rope.scaling.yarn_truncate", "truncate")):
                if get(key) is not None:
                    rope[name] = get(key)
    elif kind not in (None, "none"):
        raise GgufError(f"{model.architecture}.rope.scaling.type {kind!r} has no config rule")
    freqs = model.tensors.get("rope_freqs.weight")
    if freqs is not None:
        if kind not in (None, "none"):
            raise GgufError("the GGUF has rope_freqs.weight and rope.scaling.type; the config rules take one of them")
        rope.update(_llama3_rope(_tensor_f32(freqs), rope["rope_theta"], rotary_dim or head_dim))
    return rope


def _has(model, part, layers):
    """Whether every one of the `layers` blocks has tensor blk.<i>.<part>, or none does (None when only some do)."""
    found = {f"blk.{i}.{part}" in model.tensors for i in layers}
    return found.pop() if len(found) == 1 else None


def _same(model, part, layers, label):
    flag = _has(model, part, layers)
    if flag is None:
        raise GgufError(f"only some layers have {part}; config.json has one {label} flag for every layer")
    return flag


def _vocab_rows(model, get):
    embedding = model.tensors.get("token_embd.weight")
    if embedding is not None:
        return embedding.shape[0]
    tokens = get("tokenizer.ggml.tokens")
    return get(".vocab_size") or (len(tokens) if tokens else None)


def _base(model, get):
    """The keys every decoder config shares, from the GGUF's shape metadata."""
    drafts = get(".nextn_predict_layers") or 0
    blocks = get.require(".block_count", "block count")
    heads = get.require(".attention.head_count", "attention head count")
    hidden = get.require(".embedding_length", "width")
    head_dim = get.one(".attention.key_length") or hidden // heads
    value_dim = get.one(".attention.value_length")
    if value_dim is not None and value_dim != head_dim:
        raise GgufError(f"key length {head_dim} and value length {value_dim} differ; the config rules take one "
                        "head size")
    config = {
        "vocab_size": _vocab_rows(model, get),
        "hidden_size": hidden,
        "num_hidden_layers": blocks - drafts,
        "num_attention_heads": heads,
        "num_key_value_heads": get.one(".attention.head_count_kv") or heads,
        "head_dim": head_dim,
        "max_position_embeddings": get(".context_length"),
        "rms_norm_eps": get(".attention.layer_norm_rms_epsilon"),
        "hidden_act": "silu",
        "tie_word_embeddings": "output.weight" not in model.tensors,
    }
    if get(".feed_forward_length") is not None:
        config["intermediate_size"] = get.one(".feed_forward_length", label="MLP width")
    for key, name in (("tokenizer.ggml.bos_token_id", "bos_token_id"), ("tokenizer.ggml.eos_token_id", "eos_token_id")):
        if get(key) is not None:
            config[name] = get(key)
    return {key: value for key, value in config.items() if value is not None}, range(blocks - drafts), drafts


def _llama(model, get):
    config, layers, _ = _base(model, get)
    rotary = get(".rope.dimension_count")
    if rotary is not None and rotary != config["head_dim"]:
        raise GgufError(f"llama rotates {rotary} of {config['head_dim']} head dimensions; Llama rotates all")
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama", **config,
            "rope_parameters": _rope(get, model, config["head_dim"]),
            "attention_bias": _same(model, "attn_q.bias", layers, "attention bias"),
            "mlp_bias": _same(model, "ffn_up.bias", layers, "MLP bias")}


def _qwen2(model, get):
    config, layers, _ = _base(model, get)
    if not _same(model, "attn_q.bias", layers, "attention bias"):
        raise GgufError("a Qwen2 GGUF without query/key/value biases is not Qwen2 (its projections always have them)")
    return {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", **config,
            "rope_parameters": _rope(get, model, config["head_dim"]), "use_sliding_window": False}


def _qwen3(model, get):
    config, layers, _ = _base(model, get)
    return {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", **config,
            "rope_parameters": _rope(get, model, config["head_dim"]),
            "attention_bias": _same(model, "attn_q.bias", layers, "attention bias"), "use_sliding_window": False}


def _experts(model, get, layers):
    """The routed-expert keys, and the dense (MLP-only) layers among `layers`."""
    dense = [i for i in layers if f"blk.{i}.ffn_gate_inp.weight" not in model.tensors]
    return {"num_experts": get.require(".expert_count", "expert count"),
            "num_experts_per_tok": get.require(".expert_used_count", "experts per token"),
            "moe_intermediate_size": get.require(".expert_feed_forward_length", "expert MLP width")}, dense


def _qwen3moe(model, get):
    config, layers, _ = _base(model, get)
    experts, dense = _experts(model, get, layers)
    # The reference runtime's qwen3moe graph always renormalises the top-k weights (build_moe_ffn norm_w).
    return {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", **config, **experts,
            "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": dense,
            "rope_parameters": _rope(get, model, config["head_dim"]),
            "attention_bias": _same(model, "attn_q.bias", layers, "attention bias"), "use_sliding_window": False}


def _qwen35(model, get, moe=False):
    config, layers, drafts = _base(model, get)
    count = config["num_hidden_layers"]
    recurrent = get(".attention.recurrent_layers")
    if recurrent is not None:
        kinds = ["linear_attention" if flag else "full_attention" for flag in recurrent[:count]]
    else:
        interval = get(".full_attention_interval", 4)
        kinds = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(count)]
    state, value_heads = get.require(".ssm.state_size"), get.require(".ssm.time_step_rank")
    inner = get.require(".ssm.inner_size")
    rotary = get(".rope.dimension_count")
    rope = _rope(get, model, config["head_dim"])
    rope["partial_rotary_factor"] = rotary / config["head_dim"] if rotary else 0.25
    sections = get(".rope.dimension_sections")
    if sections:
        rope["mrope_section"] = list(sections[:3])
        rope["mrope_interleaved"] = True
    out = {"architectures": ["Qwen3_5MoeForCausalLM" if moe else "Qwen3_5ForCausalLM"],
           "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text", **config,
           "layer_types": kinds, "full_attention_interval": get(".full_attention_interval", 4),
           "linear_conv_kernel_dim": get.require(".ssm.conv_kernel"), "linear_key_head_dim": state,
           "linear_value_head_dim": inner // value_heads, "linear_num_key_heads": get.require(".ssm.group_count"),
           "linear_num_value_heads": value_heads, "rope_parameters": rope, "attn_output_gate": True,
           "attention_bias": False, "mtp_num_hidden_layers": drafts}
    if moe:
        experts, dense = _experts(model, get, layers)
        if dense:
            raise GgufError(f"Qwen3.5 MoE layers {dense} have no router; every layer of the architecture has experts")
        out.update(experts)
        shared = get(".expert_shared_feed_forward_length")
        if shared is not None:
            out["shared_expert_intermediate_size"] = shared
        out.pop("intermediate_size", None)
    return out


def _deepseek4(model, get):
    from .gguf_deepseek4 import deepseek4_config
    return deepseek4_config(model, get)


def _qwen4exp(model, get):
    from .gguf_qwen4exp import qwen4exp_config
    return qwen4exp_config(model, get, _qwen35(model, get, moe=True))


# general.architecture -> the inverse of the converter's set_gguf_parameters for it (crit_eval.gguf_layout holds
# the inverse of its tensor transforms for the same architectures).
CONFIG_RULES = {
    "llama": _llama,
    "qwen2": _qwen2,
    "qwen3": _qwen3,
    "qwen3moe": _qwen3moe,
    "qwen35": _qwen35,
    "qwen35moe": lambda model, get: _qwen35(model, get, moe=True),
    "deepseek4": _deepseek4,
    "qwen4exp": _qwen4exp,
}
# Metadata the generic keys cannot say: an architecture with any of it needs its own rule.
_BEYOND_GENERIC = (".attention.sliding_window", ".attention.sliding_window_pattern", ".attention.kv_lora_rank",
                   ".attention.q_lora_rank", ".attention.recurrent_layers", ".full_attention_interval",
                   ".ssm.conv_kernel", ".ssm.state_size", ".attention.shared_kv_layers", ".rope.dimension_sections",
                   ".expert_shared_count", ".leading_dense_block_count", ".wkv.head_size", ".rope.scaling.type")


def derive_config(model):
    """config.json for a GgufModel (crit_core.model.gguf), as a dict."""
    get = _Reader(model)
    rule = CONFIG_RULES.get(model.architecture)
    if rule is not None:
        return rule(model, get)
    beyond = [f"{model.architecture}{key}" for key in _BEYOND_GENERIC if get.has(key)]
    if beyond:
        raise GgufError(f"GGUF architecture {model.architecture!r} has no config rules in crit_core.model.gguf_files "
                        f"(supported: {', '.join(sorted(CONFIG_RULES))}) and its metadata needs them "
                        f"({', '.join(beyond)}); put the base model's config.json beside the GGUF")
    config, layers, _ = _base(model, get)
    rotary = get(".rope.dimension_count")
    rope = {"rope_type": "default", "rope_theta": get(".rope.freq_base", 10000.0)}
    if rotary and rotary != config["head_dim"]:
        rope["partial_rotary_factor"] = rotary / config["head_dim"]
    config["rope_parameters"] = rope
    if get(".expert_count"):
        config.update(_experts(model, get, layers)[0])
    return config


# tokenizer.json -------------------------------------------------------------------------------------------------

def _byte_level():
    return {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}


def derive_tokenizer(model):
    """tokenizer.json for a GgufModel with a byte-level BPE vocabulary, as a dict."""
    get = _Reader(model)
    kind = get("tokenizer.ggml.model")
    if kind != "gpt2":
        raise GgufError(f"tokenizer.json is derived for byte-level BPE vocabularies (tokenizer.ggml.model 'gpt2'); "
                        f"this GGUF's is {kind!r}; put the base model's tokenizer.json beside the GGUF")
    pre = get("tokenizer.ggml.pre")
    if pre not in PRE_TOKENIZERS:
        raise GgufError(f"tokenizer.ggml.pre {pre!r} has no pre-tokenizer rule in crit_core.model.gguf_files "
                        f"(supported: {', '.join(sorted(PRE_TOKENIZERS))}); put the base model's tokenizer.json "
                        "beside the GGUF")
    regex, normalizer, ignore_merges = PRE_TOKENIZERS[pre]
    tokens = get("tokenizer.ggml.tokens") or ()
    types = get("tokenizer.ggml.token_type") or (NORMAL,) * len(tokens)
    if len(types) != len(tokens):
        raise GgufError(f"tokenizer.ggml.token_type has {len(types)} entries for {len(tokens)} tokens")
    vocab, added = {}, []
    for i, (token, kind_) in enumerate(zip(tokens, types)):
        if kind_ == UNUSED:
            continue   # the converter's [PAD<i>] fillers up to the embedding's rows, which the tokenizer never had
        if kind_ == BYTE:
            raise GgufError(f"token {i} is a byte token; byte-level BPE has none")
        if token in vocab:
            raise GgufError(f"token {token!r} appears twice in tokenizer.ggml.tokens ({vocab[token]} and {i})")
        # Added tokens are in the model's vocabulary too, so every id below the largest one exists (an added token
        # outside it would take a new id after it).
        vocab[token] = i
        if kind_ in (CONTROL, USER_DEFINED):
            added.append({"id": i, "content": token, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": kind_ == CONTROL})
    merges = [merge.split(" ", 1) for merge in get("tokenizer.ggml.merges") or ()]
    if any(len(merge) != 2 for merge in merges):
        raise GgufError("tokenizer.ggml.merges holds an entry that is not two tokens")
    splitters = [{"type": "Split", "pattern": {"Regex": one}, "behavior": "Isolated", "invert": False}
                 for one in ((regex,) if isinstance(regex, str) else regex or ())]
    byte_level = _byte_level() if regex else {**_byte_level(), "use_regex": True}
    post = _byte_level()
    bos = get("tokenizer.ggml.bos_token_id")
    if get("tokenizer.ggml.add_bos_token", pre in ("llama-bpe", "llama3", "llama-v3")) and bos is not None:
        content = tokens[bos]
        post = {"type": "Sequence", "processors": [_byte_level(), {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": content, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": content, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                     {"SpecialToken": {"id": content, "type_id": 1}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {content: {"id": content, "ids": [bos], "tokens": [content]}}}]}
    return {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": normalizer} if normalizer else None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [*splitters, byte_level]} if splitters else byte_level,
        "post_processor": post, "decoder": _byte_level(),
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False,
                  "ignore_merges": ignore_merges, "vocab": vocab, "merges": merges},
    }


def _token(get, key):
    tokens = get("tokenizer.ggml.tokens") or ()
    index = get(key)
    return tokens[index] if isinstance(index, int) and 0 <= index < len(tokens) else None


def derive_tokenizer_config(model):
    get = _Reader(model)
    config = {"tokenizer_class": "PreTrainedTokenizerFast"}
    for key, name in (("bos", "bos_token"), ("eos", "eos_token"), ("padding", "pad_token"), ("unknown", "unk_token")):
        token = _token(get, f"tokenizer.ggml.{key}_token_id")
        if token is not None:
            config[name] = token
    for key in ("add_bos_token", "add_eos_token"):
        if get(f"tokenizer.ggml.{key}") is not None:
            config[key] = get(f"tokenizer.ggml.{key}")
    if get(".context_length") is not None:
        config["model_max_length"] = get(".context_length")
    return config


def derive_templates(model):
    """{file name: template source}: chat_template.jinja and additional_chat_templates/<name>.jinja."""
    get = _Reader(model)
    out = {}
    if isinstance(get("tokenizer.chat_template"), str):
        out["chat_template.jinja"] = get("tokenizer.chat_template")
    for name in get("tokenizer.chat_templates") or ():
        source = get(f"tokenizer.chat_template.{name}")
        if isinstance(source, str) and name != "default":
            out[f"additional_chat_templates/{name}.jinja"] = source
        elif isinstance(source, str):
            out.setdefault("chat_template.jinja", source)
    return out


def derive_generation(model):
    get = _Reader(model)
    tokens = get("tokenizer.ggml.tokens") or ()
    types = get("tokenizer.ggml.token_type") or ()
    eos = [get(f"tokenizer.ggml.{key}_token_id") for key in ("eos", "eot", "eom")]
    eos += [i for i, token in enumerate(tokens) if token in END_OF_GENERATION and i < len(types) and types[i] == CONTROL]
    eos = list(dict.fromkeys(i for i in eos if isinstance(i, int)))
    config = {}
    if get("tokenizer.ggml.bos_token_id") is not None:
        config["bos_token_id"] = get("tokenizer.ggml.bos_token_id")
    if eos:
        config["eos_token_id"] = eos[0] if len(eos) == 1 else eos
    if get("tokenizer.ggml.padding_token_id") is not None:
        config["pad_token_id"] = get("tokenizer.ggml.padding_token_id")
    sampling = {name: get(key) for key, name in SAMPLING if get(key) is not None}
    if sampling:
        config.update(do_sample=True, **sampling)
    return config


def _json(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


# File groups: a folder that holds any file of a group supplies the whole group (crit_core.model.source).
GROUPS = {
    "config": ("config.json",),
    "tokenizer": ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                  "chat_template.json", "additional_chat_templates/"),
    "generation": ("generation_config.json",),
}


def derived_files(model):
    """{group: {file name: zero-argument callable returning the file's bytes}} for a GgufModel. Each file is built
    when first read, so a model whose tokenizer cannot be derived still has its config."""
    def templates():
        return {name: source.encode("utf-8") for name, source in derive_templates(model).items()}
    tokenizer = {"tokenizer.json": lambda: _json(derive_tokenizer(model)),
                 "tokenizer_config.json": lambda: _json(derive_tokenizer_config(model))}
    for name in derive_templates(model):
        tokenizer[name] = (lambda name=name: templates()[name])
    return {"config": {"config.json": lambda: _json(derive_config(model))},
            "tokenizer": tokenizer,
            "generation": {"generation_config.json": lambda: _json(derive_generation(model))}}
