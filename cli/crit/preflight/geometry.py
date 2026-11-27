"""Random geometry: checkpoints of sampled shapes, so a shape the library has not met is found before a model has it.

    crit models random --seed 3 --out DIR [--family dense|routed|windows] [--tokenizer FOLDER]

sample() draws a config of one of FAMILIES (transformers' own architectures with a finished model plugin) whose
shapes are drawn from the ranges real checkpoints span (HEAD_WIDTHS 32 to 512, VOCABULARIES to 300k, sliding
windows, mixed layer kinds, grouped-query ratios, tied heads), at a hidden size and depth small enough for a CPU.
write() gives it random weights with the reference's own modeling code and saves it as safetensors, beside a
tokenizer's files where one is given (crit run preflight serves the folder like a downloaded checkpoint).

What reads them: tests/test_random_geometry.py resolves the library-default engine's spec (crit.spec.library_spec)
and scores each through the models core against its reference (crit models diff) on the CPU; the library's device
plugins over the same folders are `crit run preflight` on a run of each, a GPU check."""
import json
from pathlib import Path
import random
import shutil

HEAD_WIDTHS = (32, 64, 80, 96, 128, 192, 256, 320, 512)
VOCABULARIES = (256, 1024, 32000, 151936, 262144, 300000)
WINDOWS = (8, 24, 128, 1024, 4096)
FAMILIES = ("dense", "routed", "windows")
HIDDEN = 64
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                   "generation_config.json")


def sample(seed, family=None):
    """A config.json (a dict) of random geometry: `family` (FAMILIES; drawn from the seed where None) with its head
    widths, head counts, vocabulary, depth, tied head and (windows) sliding window and layer kinds drawn from the
    seed."""
    rng = random.Random(seed)
    family = family or rng.choice(FAMILIES)
    kv = rng.choice((1, 2, 4))
    common = {"vocab_size": rng.choice(VOCABULARIES), "hidden_size": HIDDEN, "head_dim": rng.choice(HEAD_WIDTHS),
              "num_key_value_heads": kv, "num_attention_heads": kv * rng.choice((1, 2, 4)),
              "num_hidden_layers": rng.choice((2, 3, 4)), "tie_word_embeddings": rng.random() < 0.5,
              "max_position_embeddings": 262144, "rope_theta": rng.choice((10000.0, 1000000.0)),
              "pad_token_id": 0, "eos_token_id": 1, "bos_token_id": 2, "intermediate_size": 128}
    if family == "dense":
        return {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], **common}
    if family == "routed":
        experts = rng.choice((4, 8, 16))
        return {"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"], **common,
                "moe_intermediate_size": 32, "num_experts": experts, "num_experts_per_tok": rng.choice((1, 2, 4)),
                "norm_topk_prob": rng.random() < 0.5, "decoder_sparse_step": 1, "mlp_only_layers": []}
    if family == "windows":
        layers = common["num_hidden_layers"]
        kinds = [rng.choice(("sliding_attention", "full_attention")) for _ in range(layers)]
        kinds[-1] = "full_attention"   # a windowed checkpoint of this kind ends on a full layer
        common.pop("rope_theta")
        return {"model_type": "gemma4_text", "architectures": ["Gemma4ForCausalLM"], **common,
                "vocab_size_per_layer_input": common["vocab_size"], "hidden_size_per_layer_input": 0,
                "global_head_dim": rng.choice(HEAD_WIDTHS), "num_global_key_value_heads": rng.choice((1, kv)),
                "attention_k_eq_v": rng.random() < 0.5, "sliding_window": rng.choice(WINDOWS),
                "layer_types": kinds, "enable_moe_block": False,
                "rope_parameters": {"full_attention": {"partial_rotary_factor": rng.choice((0.25, 0.5, 1.0)),
                                                       "rope_theta": 1000000.0, "rope_type": "proportional"},
                                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}}
    raise ValueError(f"no family {family!r} (one of {', '.join(FAMILIES)})")


def write(config, out, tokenizer=None, seed=0):
    """The checkpoint of `config` (sample()'s) in `out`: random weights from the reference's own modeling code
    (transformers), saved as BF16 safetensors, and `tokenizer`'s files (TOKENIZER_FILES) where a folder is given.
    Returns out, or None where this transformers lacks the architecture."""
    import torch
    import transformers
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    try:
        built = transformers.AutoConfig.from_pretrained(str(out))
    except (KeyError, ValueError):   # an architecture this transformers does not know
        shutil.rmtree(out, ignore_errors=True)
        return None
    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_config(built)
    model.to(torch.bfloat16).save_pretrained(str(out), safe_serialization=True)
    if tokenizer is not None:
        for name in TOKENIZER_FILES:
            if (Path(tokenizer) / name).is_file():
                shutil.copy2(Path(tokenizer) / name, out / name)
    return out


def geometry(config):
    """The shapes a sampled config draws, in one line for a report: family, head widths, heads, vocabulary, layers."""
    widths = [config["head_dim"]] + ([config["global_head_dim"]] if "global_head_dim" in config else [])
    kinds = config.get("layer_types")
    return (f"{config['model_type']} heads {config['num_attention_heads']}/{config['num_key_value_heads']} x "
            f"{'/'.join(str(w) for w in widths)}, vocab {config['vocab_size']}, {config['num_hidden_layers']} layers"
            + (f" ({', '.join(k.split('_')[0] for k in kinds)}, window {config['sliding_window']})" if kinds else ""))
