"""Small random checkpoints of real architectures for the preflight's tests: each a transformers model saved as
safetensors (crit_models.testing's torch and transformers), beside the byte-level tokenizer files of
core_text_fixtures, so crit run create, recon and capture read them as they read a downloaded checkpoint. None where
transformers lacks the architecture.

| Name | Architecture | What the preflight reads of it |
| --- | --- | --- |
| dense | Qwen3 | full attention, no routed experts, no draft layer |
| routed | Qwen3-MoE | routed experts in every layer |
| windows | Gemma 4 text | sliding-window layers (rings) beside full ones, `window` positions wide |
| draft | Qwen3-Next | linear attention, routed experts and an MTP draft layer (mtp.* tensors in a shard of their own) |
"""
import json
from pathlib import Path
import struct


def _tokenizer(folder):
    from core_text_fixtures import llama_files, write_files
    files = llama_files()
    files.pop("config.json")
    write_files(folder, files)


def _save(model, folder):
    import torch
    model.to(torch.bfloat16).save_pretrained(folder, safe_serialization=True)
    _tokenizer(folder)
    return Path(folder)


def _shard(path, tensors):
    """A safetensors file of BF16 zeros of the given shapes ({name: shape})."""
    entries, at = {}, 0
    for name, shape in tensors.items():
        size = 2
        for n in shape:
            size *= n
        entries[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [at, at + size]}
        at += size
    header = json.dumps(entries).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(at))


def checkpoint(name, folder, window=24):
    """Write the tiny checkpoint `name` into `folder` (a new folder); returns it, or None where transformers lacks it."""
    from crit_models import testing as t
    transformers = t.transformers
    if transformers is None:
        return None
    folder = Path(folder)
    common = dict(vocab_size=256, hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                  max_position_embeddings=262144, tie_word_embeddings=False, pad_token_id=0, eos_token_id=1,
                  bos_token_id=2)
    if name == "dense":
        return _save(transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
            intermediate_size=128, num_hidden_layers=2, **common)), folder)
    if name == "routed":
        return _save(transformers.Qwen3MoeForCausalLM(transformers.Qwen3MoeConfig(
            intermediate_size=128, moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
            num_hidden_layers=2, **common)), folder)
    if name == "windows":
        if not hasattr(transformers, "Gemma4ForCausalLM"):
            return None
        config = transformers.Gemma4TextConfig(
            vocab_size=256, vocab_size_per_layer_input=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
            num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=window,
            layer_types=["sliding_attention", "sliding_attention", "full_attention", "full_attention"],
            enable_moe_block=False, hidden_size_per_layer_input=0, max_position_embeddings=262144,
            tie_word_embeddings=False, pad_token_id=0, eos_token_id=1, bos_token_id=2,
            rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                                "rope_type": "proportional"},
                             "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}})
        return _save(transformers.Gemma4ForCausalLM(config), folder)
    if name == "draft":
        if not hasattr(transformers, "Qwen3NextForCausalLM"):
            return None
        config = transformers.Qwen3NextConfig(
            intermediate_size=128, moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=4,
            num_experts_per_tok=2, num_hidden_layers=4, full_attention_interval=4, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
            decoder_sparse_step=1, num_nextn_predict_layers=1, **common)
        folder = _save(transformers.Qwen3NextForCausalLM(config), folder)
        # the MTP layer: transformers' model has none, the published checkpoints store it beside the decoder
        _shard(folder / "mtp.safetensors", {"mtp.fc.weight": (64, 128), "mtp.norm.weight": (64,),
                                            "mtp.layers.0.input_layernorm.weight": (64,)})
        return folder
    raise ValueError(f"no tiny checkpoint {name!r}")
