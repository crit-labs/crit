"""The description files of a GGUF, derived from its own metadata: config.json, tokenizer.json, tokenizer_config.json,
the chat templates and generation_config.json, so a lone .gguf is a whole model to crit (standard library only).

GGUF's reference converter writes everything a runtime needs into the GGUF: the shapes and hyperparameters under
<architecture>.*, the vocabulary, merges and special tokens under tokenizer.ggml.*, the chat templates under
tokenizer.chat_template*. Each file here is the Hugging Face form of exactly that, for the GGUF's meaning as
its reference runtime defines it:

    config.json             the config rule registered for general.architecture (crit_core.registry gguf_config:
                            the inverse of the converter's set_gguf_parameters for that architecture, with
                            model_type and architectures for transformers; crit-eval's architectures give the ones
                            its reference reads), built on the generic keys below (base_config, expert_config,
                            layer_flag). An architecture without a rule gets the generic shape keys (layers, width,
                            heads, head size, MLP width, experts, context, RoPE base, vocabulary) when its metadata
                            holds nothing the generic keys cannot say (sliding windows, latent attention, recurrent
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
Every derivation is tested against the original files of models converted by the reference converter (the config
rules with them, in crit-eval), and the tokenizer against its own vocabulary test sets (Hugging Face tokenizations
of hard inputs).
"""
import json
import math
import struct

from .. import registry
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


class MetadataReader:
    """The GGUF's metadata with its value types, read as config.json values: get(".key") is <architecture>.key,
    get("key") the key as named. Config rules take one as `get`."""

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


def _has(model, part, layers):
    """Whether every one of the `layers` blocks has tensor blk.<i>.<part>, or none does (None when only some do)."""
    found = {f"blk.{i}.{part}" in model.tensors for i in layers}
    return found.pop() if len(found) == 1 else None


def layer_flag(model, part, layers, label):
    """Whether the `layers` blocks have tensor blk.<i>.<part> (such as attn_q.bias): every one or none, since
    config.json has one `label` flag for every layer; GgufError when only some do."""
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


def base_config(model, get):
    """(config, text layers, draft layers): the keys every decoder config shares, from the GGUF's shape metadata, the
    range of the blocks that are text layers and the count of draft (nextn) blocks after them."""
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


def expert_config(model, get, layers):
    """(keys, dense layers): the routed-expert keys, and the dense (MLP-only) layers among `layers`."""
    dense = [i for i in layers if f"blk.{i}.ffn_gate_inp.weight" not in model.tensors]
    return {"num_experts": get.require(".expert_count", "expert count"),
            "num_experts_per_tok": get.require(".expert_used_count", "experts per token"),
            "moe_intermediate_size": get.require(".expert_feed_forward_length", "expert MLP width")}, dense


# Metadata the generic keys cannot say: an architecture with any of it needs its own rule.
_BEYOND_GENERIC = (".attention.sliding_window", ".attention.sliding_window_pattern", ".attention.kv_lora_rank",
                   ".attention.q_lora_rank", ".attention.recurrent_layers", ".full_attention_interval",
                   ".ssm.conv_kernel", ".ssm.state_size", ".attention.shared_kv_layers", ".rope.dimension_sections",
                   ".expert_shared_count", ".leading_dense_block_count", ".wkv.head_size", ".rope.scaling.type")


def derive_config(model):
    """config.json for a GgufModel (crit_core.model.gguf), as a dict: by the rule registered for its architecture
    (crit_core.registry gguf_config), else from the generic keys."""
    get = MetadataReader(model)
    rules = registry.entries("gguf_config")
    rule = rules.get(model.architecture)
    if rule is not None:
        return rule(model, get)
    beyond = [f"{model.architecture}{key}" for key in _BEYOND_GENERIC if get.has(key)]
    if beyond:
        raise GgufError(f"GGUF architecture {model.architecture!r} has no config rule (crit_core.registry gguf_config; "
                        f"registered: {', '.join(sorted(rules)) or 'none'}) and its metadata needs one "
                        f"({', '.join(beyond)}); put the base model's config.json beside the GGUF")
    config, layers, _ = base_config(model, get)
    rotary = get(".rope.dimension_count")
    rope = {"rope_type": "default", "rope_theta": get(".rope.freq_base", 10000.0)}
    if rotary and rotary != config["head_dim"]:
        rope["partial_rotary_factor"] = rotary / config["head_dim"]
    config["rope_parameters"] = rope
    if get(".expert_count"):
        config.update(expert_config(model, get, layers)[0])
    return config


# tokenizer.json -------------------------------------------------------------------------------------------------

def _byte_level():
    return {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}


def derive_tokenizer(model):
    """tokenizer.json for a GgufModel with a byte-level BPE vocabulary, as a dict."""
    get = MetadataReader(model)
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
    get = MetadataReader(model)
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
    get = MetadataReader(model)
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
    get = MetadataReader(model)
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
