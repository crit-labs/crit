"""Core model files: config.json rules checked against crit-eval-v2's, and the strict safetensors header and index reader."""
import copy
import dataclasses
import errno
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "eval", "kernels", "cli"): sys.path.insert(0, str(ROOT / _folder))
from crit_core.files import DirectoryFiles, MappingFiles
from crit_core.model import config as mc
from crit_core.model import safetensors as st
import eval_v2_rules as v2   # crit-eval-v2's rules, the oracle Core's config rules were ported from

AUDIT_LIMITS = st.Limits(max_header_bytes=1 << 20, max_index_bytes=1 << 20, max_tensors=8192, max_shards=32,
                         max_rank=8, max_dimension=(1 << 31) - 1, max_file_bytes=64 << 30,
                         max_total_bytes=512 << 30, max_name_bytes=1024)
STRICT = {"index_policy": st.IndexPolicy.STRICT, "follow_symlinks": False}


# Config fixtures: shapes of real checkpoint families, small enough to read.
HYBRID = {
    "architectures": ["HybridForConditionalGeneration"], "model_type": "hybrid_vl", "tie_word_embeddings": False,
    "torch_dtype": "bfloat16", "eos_token_id": 151645, "num_hidden_layers": 99,
    "text_config": {
        "model_type": "hybrid_text", "num_hidden_layers": 8, "hidden_size": 2048, "num_attention_heads": 16,
        "num_key_value_heads": 2, "head_dim": 256, "max_position_embeddings": 262144, "vocab_size": 151936,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
        "full_attention_interval": 4, "linear_num_value_heads": 32, "linear_num_key_heads": 16,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512, "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": [],
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25},
        "rms_norm_eps": 1e-6, "eos_token_id": [151643, 151645],
    },
    "vision_config": {"depth": 2, "hidden_size": 1152},
}
LLAMA = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336,
    "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "max_position_embeddings": 131072,
    "rope_theta": 500000.0, "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                             "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
    "rms_norm_eps": 1e-5, "tie_word_embeddings": False, "torch_dtype": "bfloat16", "vocab_size": 128256,
    "bos_token_id": 128000, "eos_token_id": [128001, 128008, 128009], "pad_token_id": None, "head_dim": None,
}
DEEPSEEK = {
    "architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3", "hidden_size": 7168, "vocab_size": 129280,
    "num_hidden_layers": 61, "num_attention_heads": 128, "num_key_value_heads": 128, "kv_lora_rank": 512,
    "q_lora_rank": 1536, "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 128,
    "n_routed_experts": 256, "n_shared_experts": 1, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
    "first_k_dense_replace": 3, "n_group": 8, "topk_group": 4, "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
    "norm_topk_prob": True, "max_position_embeddings": 163840, "rope_theta": 10000,
    "rope_scaling": {"type": "yarn", "factor": 40, "original_max_position_embeddings": 4096, "beta_fast": 32,
                     "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0},
    "num_nextn_predict_layers": 1, "rms_norm_eps": 1e-6, "bos_token_id": 0, "eos_token_id": 1,
    "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]},
    "tie_word_embeddings": False, "torch_dtype": "bfloat16",
}
MISTRAL = {
    "architectures": ["MistralForCausalLM"], "model_type": "mistral", "hidden_size": 4096, "num_hidden_layers": 32,
    "num_attention_heads": 32, "num_key_value_heads": 8, "sliding_window": 4096, "max_position_embeddings": 32768,
    "rope_theta": 10000.0, "rms_norm_eps": 1e-5, "vocab_size": 32000, "bos_token_id": 1, "eos_token_id": 2,
}
GEMMA = {
    "architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text", "hidden_size": 1152, "num_hidden_layers": 6,
    "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256, "sliding_window": 512,
    "layer_types": ["sliding_attention"] * 5 + ["full_attention"], "max_position_embeddings": 32768,
    "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
                        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
    "dtype": "bfloat16", "torch_dtype": "float32", "layer_norm_eps": 1e-6, "vocab_size": 262144,
    "tie_word_embeddings": True, "eos_token_id": [1, 106], "pad_token_id": 0, "bos_token_id": 2,
}
GEMMA3N = {   # Gemma3nTextConfig saves intermediate_size as a per-layer array
    "architectures": ["Gemma3nForConditionalGeneration"], "model_type": "gemma3n", "text_config": {
        "model_type": "gemma3n_text", "num_hidden_layers": 4, "hidden_size": 2048, "num_attention_heads": 8,
        "num_key_value_heads": 2, "head_dim": 256, "max_position_embeddings": 32768,
        "intermediate_size": [16384, 16384, 8192, 8192], "sliding_window": 512, "vocab_size": 262400,
        "layer_types": ["sliding_attention"] * 3 + ["full_attention"], "rms_norm_eps": 1e-6}}
QWEN3_NEXT = {   # published Qwen3-Next configs declare full_attention_interval and no layer_types
    "architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next", "num_hidden_layers": 48,
    "full_attention_interval": 4, "hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 2,
    "head_dim": 256, "linear_num_value_heads": 32, "linear_num_key_heads": 16, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "max_position_embeddings": 262144,
    "num_experts": 512, "num_experts_per_tok": 10, "rope_theta": 10000000, "partial_rotary_factor": 0.25}
MAMBA2 = {   # transformers' json.dumps wrote time_step_limit's float("inf") as a bare Infinity
    "architectures": ["Mamba2ForCausalLM"], "model_type": "mamba2", "num_hidden_layers": 4, "hidden_size": 4096,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 8192,
    "time_step_limit": [0.0, math.inf], "time_step_floor": 1e-4, "vocab_size": 32768}
YARN_PARAMETERS = {
    "num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 2, "hidden_size": 512,
    "max_position_embeddings": 40960, "rope_scaling": None,
    "rope_parameters": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768,
                        "rope_theta": 1000000.0},
}
INTERVAL = {"num_hidden_layers": 48, "full_attention_interval": 4, "num_attention_heads": 16, "num_key_value_heads": 2,
            "head_dim": 256, "linear_num_value_heads": 32, "linear_key_head_dim": 128, "linear_value_head_dim": 128}
EVAL_HYBRID = {"text_config": {"num_hidden_layers": 4, "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                               "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
                               "linear_num_value_heads": 2, "linear_key_head_dim": 4, "linear_value_head_dim": 4}}
EVAL_FIXED = {"num_hidden_layers": 5, "layer_types": ["linear_attention"] * 3 + ["full_attention", "sliding_attention"],
              "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8, "sliding_window": 128,
              "linear_num_value_heads": 2, "linear_key_head_dim": 4, "linear_value_head_dim": 4}
REFERENCE_CONFIGS = {
    "hybrid with text_config": HYBRID, "llama": LLAMA, "llama with head_dim": {**LLAMA, "head_dim": 128},
    "deepseek mla with yarn": DEEPSEEK, "mistral sliding window": MISTRAL, "gemma layer types": GEMMA,
    "yarn through rope_parameters": YARN_PARAMETERS, "full_attention_interval": INTERVAL,
    "eval hybrid fixture": EVAL_HYBRID, "eval fixed-state fixture": EVAL_FIXED,
    "eval latent fixture": {"num_hidden_layers": 4, "kv_lora_rank": 512, "qk_rope_head_dim": 64},
    "latent without rope dim": {"num_hidden_layers": 4, "kv_lora_rank": 512, "num_attention_heads": 8, "head_dim": 64},
    "yarn with a fractional factor": {"max_position_embeddings": 4096, "rope_scaling": {
        "type": "yarn", "factor": 2.5, "original_max_position_embeddings": 32768}},
    "yarn without original positions": {"max_position_embeddings": 4096, "rope_scaling": {"type": "yarn", "factor": 2}},
    "linear rope scaling": {"max_position_embeddings": 4096, "rope_scaling": {"type": "linear", "factor": 4}},
    "rope_type default over type yarn": {"max_position_embeddings": 4096, "rope_scaling": {
        "rope_type": "default", "type": "yarn", "factor": 4, "original_max_position_embeddings": 32768}},
    "rope_type yarn over type default": {"max_position_embeddings": 4096, "rope_scaling": {
        "rope_type": "yarn", "type": "default", "factor": 4, "original_max_position_embeddings": 32768}},
    "rope_parameters rope_type default over type yarn": {"max_position_embeddings": 4096, "rope_parameters": {
        "rope_type": "default", "type": "yarn", "factor": 4, "original_max_position_embeddings": 32768}},
    "rope_parameters rope_type yarn over type linear": {"max_position_embeddings": 4096, "rope_parameters": {
        "type": "linear", "rope_type": "yarn", "factor": 2, "original_max_position_embeddings": 8192}},
    "empty rope_type falls back to type": {"max_position_embeddings": 4096, "rope_scaling": {
        "rope_type": "", "type": "yarn", "factor": 4, "original_max_position_embeddings": 32768}},
    "gemma 3n per-layer intermediate sizes": GEMMA3N,
    "interval hybrid without layer_types": QWEN3_NEXT,
    "later length key": {"max_position_embeddings": 0, "n_positions": 2048, "seq_length": 4096},
    "boolean and text lengths skipped": {"max_position_embeddings": True, "max_seq_len": "8192", "seq_length": 1.5,
                                         "model_max_length": 1024},
    "zero head_dim falls back": {"num_hidden_layers": 2, "head_dim": 0, "hidden_size": 512, "num_attention_heads": 8},
    "null kv heads": {"num_hidden_layers": 2, "head_dim": 64, "num_attention_heads": 8, "num_key_value_heads": None},
    "heads without a head size": {"num_hidden_layers": 4, "num_attention_heads": 8},
    "layers only": {"num_hidden_layers": 4},
    "empty": {},
    "text_config overrides the root": {"num_hidden_layers": 64, "max_position_embeddings": 2048, "text_config": {
        "num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 256, "max_position_embeddings": 8192}},
    "sliding layers with linear state": {"num_hidden_layers": 4, "layer_types": [
        "sliding_attention", "linear_attention", "full_attention", "sliding_attention"], "sliding_window": 1024,
        "hidden_size": 1024, "num_attention_heads": 8, "linear_num_value_heads": 4, "linear_num_key_heads": 2,
        "linear_key_head_dim": 32, "linear_value_head_dim": 64, "linear_conv_kernel_dim": 1},
}
ELEMENTS = (1, 2, 0.5, 1.5)
KINDS = ("full_attention", "attention", "linear_attention", "sliding_attention", "mamba", "chunked_attention")
COUNT_KEYS = {"num_hidden_layers": (0, 1, 2, 4, 7, 48), "num_attention_heads": (0, 1, 2, 8, 32),
              "num_key_value_heads": (0, 1, 2, 8), "head_dim": (0, 64, 128), "hidden_size": (0, 100, 512, 4096),
              "full_attention_interval": (0, 1, 3, 4), "kv_lora_rank": (0, 512), "qk_rope_head_dim": (0, 64),
              "linear_num_value_heads": (0, 2, 32), "linear_num_key_heads": (0, 2, 16),
              "linear_key_head_dim": (0, 4, 128), "linear_value_head_dim": (0, 4, 128),
              "linear_conv_kernel_dim": (0, 1, 4), "sliding_window": (0, 128, 4096)}


def write_config(directory, config):
    path = Path(directory) / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class _Unreadable:
    """A file whose bytes must not be read (it can still be stat-ed); see unreadable_open."""

    def __init__(self, handle):
        self.handle = handle

    def __enter__(self):
        return self

    def __exit__(self, *details):
        self.handle.close()

    def fileno(self):
        return self.handle.fileno()

    def read(self, *details):
        raise AssertionError("read before the size check")


def unreadable_open(file, mode="r", *details, **options):
    """Patched in for open() in crit_core.files and crit_core.jsonfile: opens the file but fails any read."""
    return _Unreadable(open(file, mode, *details, **options))


def eval_rules(directory, config):
    """crit eval's own answers for a config.json, each a value or the exception type it raised."""
    write_config(directory, config)
    section = v2.text_config(directory)
    results = {"section": section}
    for name, rule in (("limit", v2.model_limit), ("shape", v2.attention_shape),
                       ("kv", v2.kv_elements_per_token)):
        results[name] = attempt(rule, section)
    for element in ELEMENTS:
        results[f"fixed {element}"] = attempt(v2.fixed_state_bytes, section, element)
    return results


def core_rules(config):
    section = mc.text_section(config)
    results = {"section": section}
    for name, rule in (("limit", mc.position_limit), ("shape", mc.attention_shape), ("kv", mc.kv_elements_per_token)):
        results[name] = attempt(rule, section)
    for element in ELEMENTS:
        results[f"fixed {element}"] = attempt(mc.fixed_state_bytes, section, element)
    return results


def model_rules(model):
    results = {"limit": attempt(model.position_limit), "shape": attempt(model.attention_shape),
               "kv": attempt(model.kv_elements_per_token)}
    for element in ELEMENTS:
        results[f"fixed {element}"] = attempt(model.fixed_state_bytes, element)
    return results


def attempt(function, *args):
    try:
        return function(*args)
    except Exception as error:  # the type is the result being compared
        return type(error)


def random_rope(rng):
    choice = rng.random()
    if choice < 0.1:
        return rng.choice((None, [], "yarn", 0))
    rope = {}
    kinds = ("yarn", "yarn", "linear", "dynamic", "default", "", None)
    if rng.random() < 0.35:   # both keys, often disagreeing: rope_type is read first
        rope["rope_type"], rope["type"] = rng.choice(kinds), rng.choice(kinds)
    else:
        rope[rng.choice(("rope_type", "type"))] = rng.choice(kinds)
    for key, values in (("factor", (None, 0, -2, 4, 2.5, 40, "4", 1e-3)),
                        ("original_max_position_embeddings", (None, 0, -1, 4096, 32768, "4096", 4096.0))):
        if rng.random() < 0.85:
            rope[key] = rng.choice(values)
    return rope


def random_config(rng):
    config = {}
    for key, values in COUNT_KEYS.items():
        draw = rng.random()
        if draw < 0.65:
            config[key] = rng.choice(values)
        elif draw < 0.72:
            config[key] = None
    if rng.random() < 0.6:
        config["layer_types"] = [rng.choice(KINDS) for _ in range(rng.randrange(0, 9))]
    for key in mc.LENGTH_KEYS:
        if rng.random() < 0.25:
            config[key] = rng.choice((0, -1, True, False, 2048, 32768, "4096", 1.5, None))
    for key in ("rope_scaling", "rope_parameters"):
        if rng.random() < 0.4:
            config[key] = random_rope(rng)
    if rng.random() < 0.3:
        text = {}
        for key in list(config):
            if rng.random() < 0.5:
                text[key] = config[key] if rng.random() < 0.7 else config.pop(key)
                if key in config and rng.random() < 0.5 and key in COUNT_KEYS:
                    config[key] = rng.choice(COUNT_KEYS[key])
        config["text_config"] = text
    return config


class ConfigRuleParityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_rule_functions_equal_crit_eval_for_every_reference_config(self):
        for name, config in REFERENCE_CONFIGS.items():
            with self.subTest(config=name):
                expected = eval_rules(self.root, config)
                self.assertEqual(core_rules(config), expected)
                self.assertFalse(any(isinstance(value, type) for value in expected.values()), expected)

    def test_loaded_configs_answer_like_crit_eval(self):
        for name, config in REFERENCE_CONFIGS.items():
            with self.subTest(config=name):
                expected = eval_rules(self.root, config)
                model = mc.load_model_config(self.root)
                expected.pop("section")
                self.assertEqual(model_rules(model), expected)
                self.assertEqual(model.to_dict(), config)
                self.assertEqual(mc.text_section(model.to_dict()), v2.text_config(self.root))

    def test_randomized_configs_equal_crit_eval_and_fail_only_where_eval_fails(self):
        rng = random.Random(20261015)
        loaded = 0
        for case in range(3000):
            config = random_config(rng)
            expected = eval_rules(self.root, config)
            actual = core_rules(config)
            for key, value in expected.items():
                with self.subTest(case=case, rule=key):
                    if isinstance(value, type):
                        self.assertIs(actual[key], mc.ConfigError, (config, value))
                    else:
                        self.assertEqual(actual[key], value, config)
            try:
                model = mc.parse_model_config(config)
            except mc.ConfigError:
                continue
            loaded += 1
            for key, value in model_rules(model).items():
                self.assertEqual(value, actual[key], (case, key, config))
        self.assertGreater(loaded, 1500)

    def test_nemotron_h_llm_config(self):
        # Multimodal wrapper: the text section is llm_config; layers_block_type stands in for layer_types.
        types = ["mamba", "moe", "mamba", "attention"]
        text = {"layers_block_type": types, "num_attention_heads": 32,
                "num_key_value_heads": 2, "head_dim": 128, "max_position_embeddings": 1048576, "mamba_num_heads": 8,
                "mamba_head_dim": 4, "ssm_state_size": 16, "n_groups": 2, "conv_kernel": 4}
        config = {"model_type": "nemotron_h_omni", "llm_config": text}
        section = mc.text_section(config)
        self.assertEqual(mc.attention_shape(section), (1, 32, 128, 2))
        self.assertEqual(mc.position_limit(section), (1048576, "max_position_embeddings"))
        state = 8 * 4 * 16 * 4 + 3 * (8 * 4 + 2 * 2 * 16) * 2
        self.assertEqual(mc.fixed_state_bytes(section, 1), 2 * state)
        parsed = mc.parse_model_config(config)
        self.assertEqual(parsed.layer_types, tuple(types))
        self.assertEqual(parsed.num_hidden_layers, 4)   # derived from layers_block_type, as NemotronHConfig does
        self.assertEqual(parsed.num_attention_heads, 32)
        self.assertEqual(mc.layer_kinds(section), (mc.OTHER, mc.OTHER, mc.OTHER, mc.FULL_ATTENTION))
        # An explicit text_config (even null) wins over llm_config.
        self.assertEqual(mc.text_section({"llm_config": text, "text_config": None}).get("layers_block_type"), None)

    def test_crit_eval_fixture_values_hold(self):
        self.assertEqual(mc.position_limit({"max_position_embeddings": 262144}), (262144, "max_position_embeddings"))
        yarn = {"max_position_embeddings": 32768,
                "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}
        self.assertEqual(mc.position_limit(yarn), (131072, "yarn rope scaling: 32768 original positions x factor 4"))
        self.assertEqual(mc.position_limit({}), (None, None))
        self.assertEqual(mc.kv_elements_per_token(EVAL_FIXED), 2 * 1 * 8)
        recurrent = 2 * 4 * 4 * 4 + 3 * (2 * 2 * 4 + 2 * 4) * 2
        self.assertEqual(mc.fixed_state_bytes(EVAL_FIXED, 1), 3 * recurrent + 2 * 1 * 8 * 128)
        self.assertEqual(mc.kv_elements_per_token({"num_hidden_layers": 4, "kv_lora_rank": 512, "qk_rope_head_dim": 64}),
                         4 * (512 + 64))
        self.assertEqual(mc.attention_shape(mc.text_section(EVAL_HYBRID)), (1, 2, 8, 1))
        self.assertEqual(mc.position_limit(DEEPSEEK), (163840, "yarn rope scaling: 4096 original positions x factor 40"))
        self.assertEqual(mc.position_limit(YARN_PARAMETERS)[0], 131072)
        self.assertEqual(mc.attention_shape(INTERVAL), (12, 16, 256, 2))
        self.assertEqual(mc.attention_shape(MISTRAL), (32, 32, 128, 8))   # sliding_window alone stays full attention
        self.assertEqual(mc.fixed_state_bytes(MISTRAL, 1), 0)

    def test_yarn_skips_booleans_and_fails_like_eval_where_the_product_is_not_finite(self):
        for original, factor in ((True, 4.0), (32768, True), (32768, math.nan)):   # eval took booleans as 1
            with self.subTest(original=original, factor=factor):
                config = {"max_position_embeddings": 4096, "rope_scaling": {
                    "type": "yarn", "factor": factor, "original_max_position_embeddings": original}}
                self.assertEqual(mc.position_limit(config), (4096, "max_position_embeddings"))
        for rope_key in ("rope_scaling", "rope_parameters"):
            for original, factor in ((32768, math.inf), (4096, 1e306), (4096, 1e308), ((1 << 64) - 1, 1e300)):
                config = {"max_position_embeddings": 4096, rope_key: {
                    "rope_type": "yarn", "factor": factor, "original_max_position_embeddings": original}}
                with self.subTest(rope_key=rope_key, original=original, factor=factor):
                    with self.assertRaises((OverflowError, ValueError)):
                        v2.model_limit(config)
                    with self.assertRaisesRegex(mc.ConfigError, f"^config.json: key '{rope_key}.factor': .* original "
                                                                f"positions x factor .* is not a finite number"):
                        mc.position_limit(config)
        # Like the native reader: no integer beyond 64 bits, and no product of 2^64 or more (crit eval returns both).
        for original, factor, key, message in (
                (4096, 10 ** 400, "factor", "integer exceeds 64 bits"),
                (10 ** 400, 2.0, "original_max_position_embeddings", "integer exceeds 64 bits"),
                (-(1 << 63) - 1, 2.0, "original_max_position_embeddings", "integer exceeds 64 bits"),
                (1 << 32, 1 << 32, "factor", "yarn position limit 4294967296 x 4294967296 exceeds 2^64 - 1"),
                (4096, 1e300, "factor", "yarn position limit 4096 x 1e+300 exceeds 2^64 - 1"),
                (1 << 63, 2.0, "factor", "yarn position limit 9223372036854775808 x 2 exceeds 2^64 - 1")):
            config = {"rope_scaling": {"type": "yarn", "factor": factor, "original_max_position_embeddings": original}}
            with self.subTest(original=original, factor=factor), \
                    self.assertRaisesRegex(mc.ConfigError, "^" + re.escape(f"config.json: key 'rope_scaling.{key}': "
                                                                           f"{message}") + "$"):
                mc.position_limit(config)
        self.assertEqual(v2.model_limit({"rope_scaling": {"type": "yarn", "factor": 1 << 32,
                                                               "original_max_position_embeddings": 1 << 32}})[0],
                         1 << 64)
        for factor, original in ((1e10, 4096), ((1 << 32) - 1, 1 << 32), (3.0, 1 << 62)):   # exact, as in crit eval
            config = {"rope_scaling": {"type": "yarn", "factor": factor, "original_max_position_embeddings": original}}
            with self.subTest(factor=factor):
                self.assertEqual(mc.position_limit(config), v2.model_limit(config))

    def test_loaded_config_with_an_overflowing_yarn_factor_fails_at_the_position_limit(self):
        path = write_config(self.root, {"max_position_embeddings": 4096, "rope_scaling": {
            "type": "yarn", "factor": 1e306, "original_max_position_embeddings": 4096}})
        model = mc.load_model_config(self.root)
        self.assertEqual(model.rope.factor, 1e306)
        with self.assertRaisesRegex(mc.ConfigError, "^" + re.escape(f"{path}: key 'rope_scaling.factor': 4096 original "
                                                                    f"positions x factor 1e+306 is not a finite")):
            model.position_limit()

    def test_rules_read_values_only_where_crit_eval_evaluates_them(self):
        base = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096}
        unused = ({"sliding_window": 4096.0}, {"sliding_window": "4096", "layer_types": ["full_attention"] * 32},
                  {"linear_num_value_heads": 2.5}, {"linear_key_head_dim": "128", "linear_value_head_dim": True},
                  {"linear_num_key_heads": "16", "linear_conv_kernel_dim": -1},
                  {"layer_types": ["linear_attention"] * 32, "linear_num_value_heads": 0, "linear_key_head_dim": "128"},
                  {"layer_types": ["linear_attention"] * 32, "linear_num_value_heads": 4, "linear_key_head_dim": 0,
                   "linear_value_head_dim": 1.5},
                  {"head_dim": 128, "hidden_size": "4096"}, {"num_key_value_heads": 0, "kv_lora_rank": 0},
                  {"layer_types": ["full_attention"] * 32, "full_attention_interval": "4"})
        for extra in unused:
            config = {**base, **extra}
            with self.subTest(config=extra):
                for element in ELEMENTS:
                    self.assertEqual(mc.fixed_state_bytes(config, element), v2.fixed_state_bytes(config, element))
                self.assertEqual(mc.attention_shape(config), v2.attention_shape(config))
                self.assertEqual(mc.kv_elements_per_token(config), v2.kv_elements_per_token(config))
        # Documented deviations: wrong types that eval reads are errors, not numbers or silent defaults.
        for extra, key in (({"layer_types": False}, "layer_types"), ({"layer_types": ""}, "layer_types"),
                           ({"layer_types": ["full_attention"] * 31 + [None]}, "layer_types"),
                           ({"layer_types": ["sliding_attention"], "sliding_window": 4096.0}, "sliding_window"),
                           ({"layer_types": ["linear_attention"], "linear_num_value_heads": 2.5,
                             "linear_key_head_dim": 4, "linear_value_head_dim": 4}, "linear_num_value_heads")):
            with self.subTest(deviation=extra), self.assertRaisesRegex(mc.ConfigError, f"^config.json: key '{key}'"):
                mc.fixed_state_bytes({**base, **extra}, 1)

    def test_null_values_read_as_absent(self):
        self.assertEqual(mc.attention_shape({"num_hidden_layers": 2, "head_dim": None, "hidden_size": 4096,
                                             "num_attention_heads": 32, "num_key_value_heads": None}), (2, 32, 128, 32))
        self.assertEqual(mc.text_section({"text_config": None, "a": 1}), {"text_config": None, "a": 1})
        model = mc.parse_model_config({"text_config": None, "sliding_window": None, "rope_scaling": None,
                                       "eos_token_id": None, "quantization_config": None, "architectures": None})
        self.assertEqual((model.sliding_window, model.eos_token_ids, model.quantization, model.architectures),
                         (None, (), None, ()))
        self.assertEqual(model.text, model.raw)


class ModelConfigTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_typed_fields_of_a_nested_hybrid_mixture_of_experts(self):
        model = mc.load_model_config(write_config(self.root, HYBRID))
        self.assertEqual(model.source, str(self.root / "config.json"))
        self.assertEqual((model.architectures, model.model_type, model.text_model_type),
                         (("HybridForConditionalGeneration",), "hybrid_vl", "hybrid_text"))
        self.assertEqual((model.num_hidden_layers, model.hidden_size, model.num_attention_heads,
                          model.num_key_value_heads, model.head_dim, model.vocab_size), (8, 2048, 16, 2, 256, 151936))
        self.assertEqual((model.max_positions, model.max_positions_key), (262144, "max_position_embeddings"))
        self.assertEqual((model.rms_norm_eps, model.norm_eps_key), (1e-6, "rms_norm_eps"))
        self.assertEqual((model.tie_word_embeddings, model.dtype), (False, "bfloat16"))
        self.assertEqual((model.rope.theta, model.rope.partial_rotary_factor, model.rope.type),
                         (10000000.0, 0.25, "default"))
        self.assertEqual(model.moe, mc.MoeConfig(experts=512, experts_key="num_experts", experts_per_token=10,
                                                 moe_intermediate_size=512, shared_expert_intermediate_size=512,
                                                 decoder_sparse_step=1, norm_topk_prob=True))
        self.assertEqual(model.linear, mc.LinearAttentionConfig(16, 32, 128, 128, 4))
        self.assertIsNone(model.mla)
        self.assertEqual(model.eos_token_ids, (151643, 151645))   # the text section's list wins over the root's id
        self.assertEqual(model.layer_types, tuple(HYBRID["text_config"]["layer_types"]))
        self.assertEqual(model.text["model_type"], "hybrid_text")
        self.assertEqual(model.raw["num_hidden_layers"], 99)
        self.assertEqual(model.get("vision_config")["depth"], 2)
        self.assertEqual(model.layer_kinds(), (mc.LINEAR_ATTENTION,) * 3 + (mc.FULL_ATTENTION,) +
                         (mc.LINEAR_ATTENTION,) * 3 + (mc.FULL_ATTENTION,))
        self.assertEqual((model.effective_head_dim(), model.effective_kv_heads()), (256, 2))

    def test_latent_attention_experts_yarn_and_quantization(self):
        model = mc.parse_model_config(DEEPSEEK)
        self.assertEqual(model.mla, mc.MlaConfig(512, 64, 128, 128, 1536))
        self.assertEqual((model.moe.experts, model.moe.experts_key, model.moe.experts_per_token, model.moe.shared_experts,
                          model.moe.first_k_dense_replace, model.moe.n_group, model.moe.topk_group,
                          model.moe.routed_scaling_factor, model.moe.scoring_func),
                         (256, "n_routed_experts", 8, 1, 3, 8, 4, 2.5, "sigmoid"))
        self.assertEqual((model.rope.type, model.rope.factor, model.rope.original_max_position_embeddings,
                          model.rope.beta_fast, model.rope.mscale_all_dim, model.rope.theta),
                         ("yarn", 40.0, 4096, 32.0, 1.0, 10000.0))
        self.assertEqual(model.quantization.method, "fp8")
        self.assertEqual(model.quantization.raw["weight_block_size"], (128, 128))
        self.assertEqual((model.num_nextn_predict_layers, model.bos_token_id, model.eos_token_ids), (1, 0, (1,)))
        self.assertEqual(model.max_positions, 163840)
        self.assertEqual(model.position_limit()[0], 163840)
        self.assertEqual(model.kv_elements_per_token(), 61 * (512 + 64))

    def test_llama_mistral_and_gemma_fields(self):
        llama = mc.parse_model_config(LLAMA)
        self.assertEqual((llama.rope.type, llama.rope.factor, llama.rope.low_freq_factor, llama.rope.high_freq_factor,
                          llama.rope.original_max_position_embeddings, llama.rope.theta),
                         ("llama3", 8.0, 1.0, 4.0, 8192, 500000.0))
        self.assertEqual((llama.head_dim, llama.effective_head_dim(), llama.effective_kv_heads()), (None, 128, 8))
        self.assertEqual((llama.eos_token_ids, llama.bos_token_id, llama.pad_token_id), ((128001, 128008, 128009),
                                                                                         128000, None))
        self.assertEqual(llama.position_limit(), (131072, "max_position_embeddings"))   # llama3 scaling: declared
        mistral = mc.parse_model_config(MISTRAL)
        self.assertEqual((mistral.sliding_window, mistral.layer_kinds()), (4096, (mc.FULL_ATTENTION,) * 32))
        self.assertIsNone(mistral.moe)
        gemma = mc.parse_model_config(GEMMA)
        self.assertEqual((gemma.dtype, gemma.norm_eps_key, gemma.tie_word_embeddings), ("bfloat16", "layer_norm_eps",
                                                                                        True))
        self.assertEqual(gemma.layer_kinds(), (mc.SLIDING_ATTENTION,) * 5 + (mc.FULL_ATTENTION,))
        self.assertEqual((gemma.rope.type, gemma.rope.theta), ("", None))   # keyed per layer type: raw only
        self.assertEqual(gemma.rope.raw["full_attention"]["rope_theta"], 1000000.0)
        self.assertEqual(mc.parse_model_config({"torch_dtype": "float16"}).dtype, "float16")

    def test_rope_sources_follow_the_position_rule(self):
        both = {"rope_scaling": {"type": "linear", "factor": 2.0},
                "rope_parameters": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096,
                                    "rope_theta": 5e5}, "max_position_embeddings": 8192}
        model = mc.parse_model_config(both)
        self.assertEqual((model.rope.type, model.rope.factor, model.rope.theta), ("linear", 2.0, 500000.0))
        self.assertEqual(model.position_limit(), (8192, "max_position_embeddings"))
        self.assertEqual(model.position_limit(), v2.model_limit(both))
        model = mc.parse_model_config({**both, "rope_scaling": {}})
        self.assertEqual((model.rope.type, model.position_limit()[0]), ("yarn", 16384))
        phi = mc.parse_model_config({"original_max_position_embeddings": 4096, "rope_scaling": {
            "type": "longrope", "short_factor": [1, 1.5], "long_factor": [2.0, 3]}})
        self.assertEqual((phi.rope.short_factor, phi.rope.long_factor, phi.rope.original_max_position_embeddings),
                         ((1.0, 1.5), (2.0, 3.0), 4096))
        vl = mc.parse_model_config({"rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]}})
        self.assertEqual(vl.rope.mrope_section, (16, 24, 24))

    def test_layer_kinds_cover_types_intervals_and_defaults(self):
        self.assertEqual(mc.layer_kinds({"num_hidden_layers": 4, "layer_types": [
            "attention", "mamba", "chunked_attention", "linear_attention"]}),
            (mc.FULL_ATTENTION, mc.OTHER, mc.OTHER, mc.LINEAR_ATTENTION))
        self.assertEqual(mc.layer_kinds({"num_hidden_layers": 6, "full_attention_interval": 3}),
                         (mc.LINEAR_ATTENTION, mc.LINEAR_ATTENTION, mc.FULL_ATTENTION) * 2)
        self.assertEqual(mc.layer_kinds({"num_hidden_layers": 2}), (mc.FULL_ATTENTION,) * 2)
        for name, config in REFERENCE_CONFIGS.items():
            shape = mc.attention_shape(mc.text_section(config))
            if shape:
                with self.subTest(config=name):
                    kinds = mc.layer_kinds(mc.text_section(config))
                    self.assertEqual(kinds.count(mc.FULL_ATTENTION), shape[0])
        for config, message in (({}, "key 'num_hidden_layers': missing"),
                                ({"num_hidden_layers": 0}, "expected a positive integer, got 0"),
                                ({"num_hidden_layers": 3, "layer_types": ["full_attention"]}, "1 entries for 3 layers"),
                                ({"num_hidden_layers": mc.MAX_LAYERS + 1}, "layer limit")):
            with self.subTest(config=config), self.assertRaisesRegex(mc.ConfigError, message):
                mc.layer_kinds(config)

    def test_effective_head_dim_and_kv_heads_validate_the_division(self):
        self.assertEqual(mc.effective_head_dim({"head_dim": 80, "hidden_size": 4096, "num_attention_heads": 32}), 80)
        self.assertEqual(mc.effective_head_dim({"head_dim": 0, "hidden_size": 4096, "num_attention_heads": 32}), 128)
        self.assertEqual(mc.effective_kv_heads({"num_attention_heads": 32, "num_key_value_heads": 0}), 32)
        for function, config, message in (
                (mc.effective_head_dim, {"hidden_size": 4097, "num_attention_heads": 32},
                 "key 'hidden_size': 4097 is not divisible by num_attention_heads 32"),
                (mc.effective_head_dim, {}, "key 'num_attention_heads': missing \\(no head_dim\\)"),
                (mc.effective_head_dim, {"num_attention_heads": 8}, "key 'hidden_size': missing"),
                (mc.effective_head_dim, {"num_attention_heads": 0, "hidden_size": 8},
                 "key 'num_attention_heads': expected a positive integer, got 0"),
                (mc.effective_kv_heads, {"num_attention_heads": 32, "num_key_value_heads": 5},
                 "key 'num_key_value_heads': 5 does not divide num_attention_heads 32"),
                (mc.effective_kv_heads, {"num_key_value_heads": 8}, "key 'num_attention_heads': missing")):
            with self.subTest(config=config), self.assertRaisesRegex(mc.ConfigError, "^config.json: " + message):
                function(config)

    def test_wrong_types_raise_config_errors_naming_the_file_and_key(self):
        rules = (
            (mc.attention_shape, {"num_hidden_layers": True}, "key 'num_hidden_layers': expected an integer in "
                                                               "\\[0, 4294967295\\], got true"),
            (mc.attention_shape, {"num_hidden_layers": 4, "layer_types": "full_attention"},
             "key 'layer_types': expected an array"),
            (mc.attention_shape, {"num_hidden_layers": 4, "layer_types": ["full_attention", 3]},
             "key 'layer_types': entry 1 is 3, not a string"),
            (mc.attention_shape, {"num_hidden_layers": 4, "num_attention_heads": "8", "head_dim": 8},
             "key 'num_attention_heads'"),
            (mc.kv_elements_per_token, {"num_hidden_layers": 4, "kv_lora_rank": 512.0}, "key 'kv_lora_rank'"),
            (mc.kv_elements_per_token, {"num_hidden_layers": -4}, "key 'num_hidden_layers': .*got -4"),
            (mc.text_section, {"text_config": []}, "key 'text_config': expected an object, got an array"),
            (mc.text_section, [], "the config must be a JSON object, not an array"),
        )
        for rule, config, message in rules:
            with self.subTest(config=config), self.assertRaisesRegex(mc.ConfigError, "^config.json: " + message):
                rule(config)
        with self.assertRaisesRegex(mc.ConfigError, "^config.json: key 'num_attention_heads': missing"):
            mc.fixed_state_bytes({"layer_types": ["sliding_attention"], "sliding_window": 8, "head_dim": 8}, 1)
        for value in (True, -1, math.inf, "1"):
            with self.subTest(element=value), self.assertRaisesRegex(ValueError, "kv_bytes_per_element"):
                mc.fixed_state_bytes({}, value)
        parsed = (
            ({"rope_scaling": {"factor": "4"}}, "key 'rope_scaling.factor': expected a finite number, got '4'"),
            ({"rope_scaling": False}, "key 'rope_scaling': expected an object, got false"),
            ({"rope_parameters": {"rope_theta": True}}, "key 'rope_parameters.rope_theta'"),
            ({"rope_scaling": {"type": "yarn", "original_max_position_embeddings": 4096.0}},
             "key 'rope_scaling.original_max_position_embeddings'"),
            ({"rope_scaling": {"type": "longrope", "short_factor": [1, None]}}, "key 'rope_scaling.short_factor\\[1\\]'"),
            ({"eos_token_id": [1, True]}, "key 'eos_token_id\\[1\\]'"),
            ({"eos_token_id": 1 << 31}, "key 'eos_token_id': expected an integer in \\[-2147483648, 2147483647\\]"),
            ({"bos_token_id": "1"}, "key 'bos_token_id'"),
            ({"architectures": "LlamaForCausalLM"}, "key 'architectures': expected an array"),
            ({"tie_word_embeddings": 1}, "key 'tie_word_embeddings': expected true or false, got 1"),
            ({"rms_norm_eps": math.nan}, "key 'rms_norm_eps': expected a finite number, got nan"),
            ({"layer_norm_epsilon": "1e-5"}, "key 'layer_norm_epsilon'"),
            ({"quantization_config": "fp8"}, "key 'quantization_config': expected an object"),
            ({"quantization_config": {"quant_method": 3}}, "key 'quantization_config.quant_method'"),
            ({"hidden_size": 1 << 32}, "key 'hidden_size': expected an integer in \\[0, 4294967295\\]"),
            ({"num_local_experts": -1}, "key 'num_local_experts'"),
            ({"num_experts": 8, "mlp_only_layers": [0, "1"]}, "key 'mlp_only_layers\\[1\\]'"),
            ({"num_experts": 8, "norm_topk_prob": "true"}, "key 'norm_topk_prob'"),
            ({"kv_lora_rank": 512, "q_lora_rank": 1.5}, "key 'q_lora_rank'"),
            ({"linear_conv_kernel_dim": False}, "key 'linear_conv_kernel_dim'"),
            ({"model_type": 3}, "key 'model_type': expected a string, got 3"),
            ({"dtype": ["bfloat16"]}, "key 'dtype': expected a string, got an array"),
            ({"text_config": {"head_dim": 1 << 4000}}, "key 'head_dim': integer exceeds 64 bits$"),
            ({"hidden_size": 1 << 64}, "key 'hidden_size': integer exceeds 64 bits$"),
            ({"pad_token_id": -(1 << 63) - 1}, "key 'pad_token_id': integer exceeds 64 bits$"),
            ({"eos_token_id": [1, 1 << 64]}, "key 'eos_token_id\\[1\\]': integer exceeds 64 bits$"),
            ({"intermediate_size": [1, 1 << 70]}, "key 'intermediate_size\\[1\\]': integer exceeds 64 bits$"),
            ({"max_position_embeddings": "x", "n_positions": 1 << 64}, "key 'n_positions': integer exceeds 64 bits$"),
            ({"max_position_embeddings": -(1 << 64)}, "key 'max_position_embeddings': integer exceeds 64 bits$"),
            ({"rope_scaling": {"type": "longrope", "short_factor": [1, -(1 << 70)]}},
             "key 'rope_scaling.short_factor\\[1\\]': integer exceeds 64 bits$"),
            ({"layer_types": [None]}, "key 'layer_types': entry 0 is null, not a string"),
        )
        for config, message in parsed:
            with self.subTest(config=repr(config)[:80]), \
                    self.assertRaisesRegex(mc.ConfigError, "^config.json: " + message):
                mc.parse_model_config(config)
        with self.assertRaisesRegex(mc.ConfigError, "^weights/config.json: key 'vocab_size'"):
            mc.parse_model_config({"vocab_size": -1}, "weights/config.json")

    def test_sources_directory_path_and_model_files(self):
        path = write_config(self.root, LLAMA)
        for source in (self.root, str(self.root), path, str(path)):
            with self.subTest(source=source):
                self.assertEqual(mc.load_model_config(source).num_hidden_layers, 32)
        data = json.dumps(LLAMA).encode("utf-8")
        files = MappingFiles({"config.json": data})
        model = mc.load_model_config(files)
        self.assertEqual((model.source, model.num_key_value_heads), ("config.json", 8))
        self.assertEqual([record.row() for record in files.records],
                         [{"path": "config.json", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}])
        self.assertEqual(mc.load_model_config(DirectoryFiles(self.root)).vocab_size, 128256)
        with self.assertRaisesRegex(mc.ConfigError, "^config.json: missing from the model files"):
            mc.load_model_config(MappingFiles({}))
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(mc.ConfigError, "config.json: missing$"):
            mc.load_model_config(empty)
        with self.assertRaisesRegex(mc.ConfigError, "absent.json: missing$"):
            mc.load_model_config(self.root / "absent.json")
        with self.assertRaisesRegex(ValueError, "max_bytes"):
            mc.load_model_config(self.root, max_bytes=0)

    def test_malformed_json_is_a_config_error_naming_the_file(self):
        cases = ((b'{"a": 1, "a": 2}', "duplicate key 'a'"), (b'{"a": 1e400}', "non-finite number 1e400"),
                 (b'{"a": -NaN}', "Expecting value"), (b"[1]", "the config must be a JSON object"),
                 (b'{"a": 1', "Expecting"), (b'{"a": "\xff"}', "not UTF-8"),
                 (b'{"a": ' + b"1" * 5000 + b"}", "digits"), (b'{"a": [' * 200000 + b"]" * 200000 + b"}", "nesting"))
        for data, message in cases:
            with self.subTest(data=data[:24]):
                path = self.root / "config.json"
                path.write_bytes(data)
                with self.assertRaisesRegex(mc.ConfigError, "^" + re.escape(str(path)) + ": .*" + message):
                    mc.load_model_config(self.root)
                with self.assertRaisesRegex(mc.ConfigError, "^config.json: .*" + message):
                    mc.load_model_config(MappingFiles({"config.json": data}))
        path.write_text(json.dumps({"pad": "x" * 100}), encoding="utf-8")
        with self.assertRaisesRegex(mc.ConfigError, "exceeds the 64-byte limit"):
            mc.load_model_config(path, max_bytes=64)
        with self.assertRaisesRegex(mc.ConfigError, "exceeds the 64-byte limit"):
            mc.load_model_config(MappingFiles({"config.json": path.read_bytes()}), max_bytes=64)

    def test_model_config_is_read_only_all_the_way_down(self):
        model = mc.load_model_config(write_config(self.root, DEEPSEEK))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            model.hidden_size = 1
        for mapping in (model.raw, model.section, model.raw["rope_scaling"], model.quantization.raw, model.rope.raw):
            with self.subTest(mapping=mapping), self.assertRaises(TypeError):
                mapping["new"] = 1
        self.assertIsInstance(model.raw["quantization_config"]["weight_block_size"], tuple)
        copy = model.to_dict()
        copy["rope_scaling"]["factor"] = 1
        self.assertEqual(model.rope.factor, 40.0)
        self.assertEqual(json.loads(json.dumps(model.to_dict())), DEEPSEEK)

    def test_integers_beyond_64_bits_in_real_valued_keys_are_config_errors(self):
        huge = "1" + "0" * 400
        cases = (('{"rope_theta": %s}', "rope_theta"), ('{"rms_norm_eps": %s}', "rms_norm_eps"),
                 ('{"layer_norm_epsilon": %s}', "layer_norm_epsilon"),
                 ('{"partial_rotary_factor": %s}', "partial_rotary_factor"),
                 ('{"rope_scaling": {"type": "linear", "factor": %s}}', "rope_scaling.factor"),
                 ('{"rope_parameters": {"rope_type": "yarn", "factor": %s, "original_max_position_embeddings": 4096}}',
                  "rope_parameters.factor"),
                 ('{"rope_parameters": {"rope_theta": %s}}', "rope_parameters.rope_theta"),
                 ('{"rope_scaling": {"type": "longrope", "short_factor": [1, %s]}}', "rope_scaling.short_factor[1]"),
                 ('{"rope_scaling": {"type": "longrope", "long_factor": [%s]}}', "rope_scaling.long_factor[0]"),
                 ('{"num_experts": 8, "routed_scaling_factor": %s}', "routed_scaling_factor"),
                 ('{"text_config": {"rope_scaling": {"type": "yarn", "beta_fast": -%s}}}', "rope_scaling.beta_fast"))
        for template, key in cases:
            data = (template % huge).encode()
            message = f"key '{key}': integer exceeds 64 bits"
            with self.subTest(key=key):
                with self.assertRaisesRegex(mc.ConfigError, "^" + re.escape(f"config.json: {message}") + "$"):
                    mc.load_model_config(MappingFiles({"config.json": data}))
                (self.root / "config.json").write_bytes(data)
                with self.assertRaisesRegex(mc.ConfigError, re.escape(message)):
                    mc.load_model_config(self.root)
        self.assertEqual(mc.parse_model_config({"rope_theta": (1 << 64) - 1}).rope.theta, float((1 << 64) - 1))
        self.assertEqual(mc.parse_model_config({"rope_theta": -(1 << 63)}).rope.theta, -float(1 << 63))
        self.assertEqual(mc.load_model_config(MappingFiles({"config.json": b'{"rope_theta": 1e300}'})).rope.theta,
                         1e300)
        for value in (10 ** 300, 1 << 64, -(1 << 63) - 1):
            with self.subTest(value=value), self.assertRaisesRegex(mc.ConfigError, "^config.json: key 'rope_theta': "
                                                                                   "integer exceeds 64 bits$"):
                mc.parse_model_config({"rope_theta": value})
        for value in (10 ** 400, -(10 ** 400), 1 << 1024):
            with self.subTest(element=value), self.assertRaisesRegex(ValueError, "kv_bytes_per_element must be"):
                mc.fixed_state_bytes(EVAL_FIXED, value)
        with self.assertRaisesRegex(ValueError, "kv_bytes_per_element 1e\\+306 makes the sliding_attention cache"):
            mc.fixed_state_bytes(EVAL_FIXED, 1e306)
        self.assertEqual(mc.fixed_state_bytes(EVAL_FIXED, 10 ** 300), v2.fixed_state_bytes(EVAL_FIXED, 10 ** 300))

    def test_nesting_beyond_the_native_depth_limit_is_a_config_error(self):
        def nested(depth):   # the root object holding depth - 1 nested arrays
            inner = []
            for _ in range(depth - 2):
                inner = [inner]
            return {"a": inner}
        limit = mc.MAX_CONFIG_DEPTH
        deepest = nested(limit)
        data = json.dumps(deepest).encode()
        self.assertEqual(mc.load_model_config(MappingFiles({"config.json": data})).to_dict(), json.loads(data))
        self.assertEqual(mc.parse_model_config(deepest).to_dict(), deepest)
        for depth in (limit + 1, 499, 500, 1000, 5000):
            config = nested(depth)
            data = ('{"a": ' + "[" * (depth - 1) + "]" * (depth - 1) + "}").encode()
            with self.subTest(depth=depth):
                with self.assertRaisesRegex(mc.ConfigError, "^config.json: nesting"):
                    mc.load_model_config(MappingFiles({"config.json": data}))
                (self.root / "config.json").write_bytes(data)
                with self.assertRaisesRegex(mc.ConfigError, "^" + re.escape(str(self.root / "config.json")) +
                                            ": nesting"):
                    mc.load_model_config(self.root)
                with self.assertRaisesRegex(mc.ConfigError, f"^config.json: nesting exceeds the depth limit of {limit}$"):
                    mc.parse_model_config(config)
        cyclic = {"num_hidden_layers": 2}
        cyclic["self"] = [cyclic]
        with self.assertRaisesRegex(mc.ConfigError, "nesting exceeds the depth limit"):
            mc.parse_model_config(cyclic)

    def test_config_values_are_held_to_strict_json(self):
        cases = (('{"model_type": "\\ud800"}', "config.json: key 'model_type': '\\ud800' is not valid UTF-8"),
                 ('{"architectures": ["X\\udc00"]}', "key 'architectures[0]': 'X\\udc00' is not valid UTF-8"),
                 ('{"vision_config": {"name": ["\\udfff"]}}', "key 'vision_config.name[0]': '\\udfff' is not"),
                 ('{"a": {"\\ud800": 1}}', "config.json: key 'a': object key '\\ud800' is not valid UTF-8"),
                 ('{"\\ud800": 1}', "config.json: object key '\\ud800' is not valid UTF-8"),
                 ('{"a": "\\ud83d\\u0041"}', "config.json: key 'a': '\\ud83dA' is not valid UTF-8"))
        for text, message in cases:
            # Files fail in the JSON reader, as in the native one; parsed objects in the copy.
            with self.subTest(text=text), self.assertRaises(mc.ConfigError) as caught:
                mc.load_model_config(MappingFiles({"config.json": text.encode()}))
            self.assertEqual(str(caught.exception), "config.json: unpaired surrogate escape")
            with self.subTest(text=text), self.assertRaises(mc.ConfigError) as caught:
                mc.parse_model_config(json.loads(text))
            self.assertIn(message, str(caught.exception))
            str(caught.exception).encode("utf-8")   # the message itself stays printable
        self.assertEqual(mc.load_model_config(MappingFiles({"config.json": b'{"model_type": "\\ud83d\\ude00"}'}))
                         .model_type, "\U0001F600")
        for config, message in (({1: 2}, "config.json: object key 1 is not a string"),
                                ({"a": {1, 2}}, "config.json: key 'a': {1, 2} is not a JSON value"),
                                ({"a": [b"x"]}, "config.json: key 'a[0]': b'x' is not a JSON value"),
                                ({"a": [Fraction(10 ** 400)]}, "config.json: key 'a[0]': Fraction(1000"),
                                ({"rms_norm_eps": math.inf}, "config.json: key 'rms_norm_eps': expected a finite"),
                                ({"x" * 5000: {"y": b""}}, "config.json: key '" + "x" * 197 + "...': b'' is not")):
            with self.subTest(config=config), self.assertRaisesRegex(mc.ConfigError, "^" + re.escape(message)):
                mc.parse_model_config(config)

    def test_directory_files_sources_are_size_checked_before_reading(self):
        path = self.root / "config.json"
        with open(path, "wb") as handle:
            handle.truncate(64 << 20)   # sparse, four times the default limit
        files = DirectoryFiles(self.root)
        with mock.patch("crit_core.files.open", unreadable_open, create=True), \
                mock.patch("crit_core.jsonfile.open", unreadable_open, create=True):
            with self.assertRaisesRegex(mc.ConfigError, "^config.json: 67108864 bytes exceeds the 16777216-byte limit$"):
                mc.load_model_config(files)
            with self.assertRaisesRegex(mc.ConfigError, "^" + re.escape(f"{path}: 67108864 bytes exceeds the "
                                                                        "16777216-byte limit") + "$"):
                mc.load_model_config(self.root)
        self.assertEqual(files.records, ())
        write_config(self.root, {"pad": "x" * 100})
        with self.assertRaisesRegex(mc.ConfigError, "exceeds the 64-byte limit"):
            mc.load_model_config(DirectoryFiles(self.root), max_bytes=64)
        self.assertEqual(mc.load_model_config(files).get("pad"), "x" * 100)

    def test_token_ids_accept_negative_values_within_32_bits(self):
        model = mc.parse_model_config({"pad_token_id": -1, "eos_token_id": [-1, 2], "bos_token_id": -(1 << 31)})
        self.assertEqual((model.pad_token_id, model.eos_token_ids, model.bos_token_id), (-1, (-1, 2), -(1 << 31)))
        self.assertEqual(mc.parse_model_config({"eos_token_id": -1}).eos_token_ids, (-1,))
        with self.assertRaisesRegex(mc.ConfigError, "key 'pad_token_id': expected an integer in \\[-2147483648"):
            mc.parse_model_config({"pad_token_id": -(1 << 31) - 1})

    def test_typed_fields_fall_back_in_the_documented_order(self):
        theta = {"rope_theta": 1.0, "rope_parameters": {"rope_theta": 2.0}, "rope_scaling": {"rope_theta": 3.0}}
        self.assertEqual(mc.parse_model_config(theta).rope.theta, 1.0)
        self.assertEqual(mc.parse_model_config({**theta, "rope_theta": None}).rope.theta, 2.0)
        self.assertEqual(mc.parse_model_config({"rope_scaling": {"type": "linear", "factor": 2, "rope_theta": 5e5}})
                         .rope.theta, 500000.0)
        self.assertEqual(mc.parse_model_config({"num_experts": 8, "num_shared_experts": 2}).moe.shared_experts, 2)
        self.assertEqual(mc.parse_model_config({"num_experts": 8, "n_shared_experts": 1, "num_shared_experts": 2})
                         .moe.shared_experts, 1)
        nested = mc.parse_model_config({"architectures": ["Root"], "tie_word_embeddings": False, "text_config": {
            "architectures": ["Text"], "tie_word_embeddings": True}})
        self.assertEqual((nested.architectures, nested.tie_word_embeddings), (("Text",), True))
        root_only = mc.parse_model_config({"architectures": ["Root"], "tie_word_embeddings": False,
                                           "text_config": {"hidden_size": 8}})
        self.assertEqual((root_only.architectures, root_only.tie_word_embeddings), (("Root",), False))
        self.assertEqual(mc.parse_model_config({"linear_num_value_heads": 0}).linear,
                         mc.LinearAttentionConfig(num_value_heads=0))
        self.assertIsNone(mc.parse_model_config({"linear_num_value_heads": None}).linear)

    def test_require_int_names_missing_and_wrong_keys(self):
        model = mc.parse_model_config(LLAMA)
        self.assertEqual(model.require_int("num_hidden_layers"), 32)
        with self.assertRaisesRegex(mc.ConfigError, "key 'num_experts': missing"):
            model.require_int("num_experts")
        with self.assertRaisesRegex(mc.ConfigError, "key 'rope_theta': expected an integer"):
            model.require_int("rope_theta")

    def test_interval_hybrids_mark_the_layers_between_full_ones_as_linear_attention(self):
        model = mc.load_model_config(write_config(self.root, QWEN3_NEXT))
        kinds = model.layer_kinds()
        self.assertEqual((kinds.count(mc.LINEAR_ATTENTION), kinds.count(mc.FULL_ATTENTION)), (36, 12))
        self.assertEqual(kinds[:4], (mc.LINEAR_ATTENTION,) * 3 + (mc.FULL_ATTENTION,))
        self.assertEqual(model.linear, mc.LinearAttentionConfig(16, 32, 128, 128, 4))
        self.assertEqual(model.attention_shape(), (12, 16, 256, 2))
        rng = random.Random(4)
        for _ in range(200):   # the layer_types Hugging Face derives from full_attention_interval
            layers, interval = rng.randrange(1, 70), rng.randrange(1, 9)
            derived = ["linear_attention" if (index + 1) % interval else "full_attention" for index in range(layers)]
            with self.subTest(layers=layers, interval=interval):
                self.assertEqual(mc.layer_kinds({"num_hidden_layers": layers, "full_attention_interval": interval}),
                                 mc.layer_kinds({"num_hidden_layers": layers, "layer_types": derived}))
        self.assertEqual(mc.layer_kinds({"num_hidden_layers": 4, "full_attention_interval": 0}),
                         (mc.FULL_ATTENTION,) * 4)

    def test_get_reads_the_text_section_before_the_root(self):
        model = mc.parse_model_config(HYBRID)
        self.assertEqual((model.get("num_hidden_layers"), model.raw["num_hidden_layers"]), (8, 99))
        self.assertEqual(model.get("eos_token_id"), (151643, 151645))
        self.assertEqual((model.get("vision_config")["depth"], model.get("absent", 7)), (2, 7))
        self.assertEqual(model.text["num_hidden_layers"], 8)

    def test_per_layer_intermediate_sizes_and_untyped_non_finite_values_load(self):
        gemma = mc.load_model_config(write_config(self.root, GEMMA3N))
        self.assertEqual((gemma.intermediate_size, gemma.intermediate_sizes), (None, (16384, 16384, 8192, 8192)))
        self.assertEqual(gemma.layer_kinds(), (mc.SLIDING_ATTENTION,) * 3 + (mc.FULL_ATTENTION,))
        self.assertEqual((gemma.attention_shape(), gemma.position_limit()), ((1, 8, 256, 2),
                                                                             (32768, "max_position_embeddings")))
        self.assertEqual((mc.parse_model_config(LLAMA).intermediate_size, mc.parse_model_config(LLAMA)
                          .intermediate_sizes), (14336, ()))
        for value, message in (("16384", "key 'intermediate_size': expected an integer or an array of integers, "
                                         "got '16384'"),
                               ([16384, -1], "key 'intermediate_size\\[1\\]': expected an integer in"),
                               ([1.5], "key 'intermediate_size\\[0\\]'"),
                               ({"a": 1}, "key 'intermediate_size': .*got an object")):
            with self.subTest(value=value), self.assertRaisesRegex(mc.ConfigError, "^config.json: " + message):
                mc.parse_model_config({"intermediate_size": value})
        # A parsed config keeps non-finite numbers where no typed field reads them, as json.loads gives them.
        path = write_config(self.root, MAMBA2)
        self.assertIn(b"Infinity", path.read_bytes())
        parsed = json.loads(path.read_text(encoding="utf-8"))
        mamba = mc.parse_model_config(parsed)
        self.assertEqual(mamba.get("time_step_limit"), (0.0, math.inf))
        self.assertEqual(model_rules(mamba), {key: value for key, value in eval_rules(self.root, MAMBA2).items()
                                              if key != "section"})
        nan = mc.parse_model_config({"extra": [math.nan], "rope_scaling": {"type": "yarn", "factor": 2,
                                                                           "original_max_position_embeddings": 64}})
        self.assertTrue(math.isnan(nan.get("extra")[0]))
        for config, key in (({"rms_norm_eps": math.inf}, "rms_norm_eps"),
                            ({"rope_scaling": {"type": "linear", "factor": -math.inf}}, "rope_scaling.factor"),
                            ({"rope_theta": math.nan}, "rope_theta"),
                            ({"rope_scaling": {"type": "longrope", "long_factor": [1.0, math.inf]}},
                             "rope_scaling.long_factor\\[1\\]")):
            with self.subTest(config=config), self.assertRaisesRegex(mc.ConfigError, f"^config.json: key '{key}': "
                                                                                     f"expected a finite number"):
                mc.parse_model_config(config)
        # config.json files load the bare literals too (only config.json and generation_config.json opt in).
        for source in (self.root, path, DirectoryFiles(self.root), MappingFiles({"config.json": path.read_bytes()})):
            with self.subTest(source=source):
                loaded = mc.load_model_config(source)
                self.assertEqual(loaded.get("time_step_limit"), (0.0, math.inf))
                self.assertEqual(model_rules(loaded), model_rules(mamba))
        nonfinite = b'{"a": [NaN, -Infinity], "rms_norm_eps": 1e-5}'
        loaded = mc.load_model_config(MappingFiles({"config.json": nonfinite}))
        self.assertTrue(math.isnan(loaded.get("a")[0]))
        # Loads of one file and deep copies compare equal despite the NaN; a pickled copy does not (documented).
        self.assertEqual(mc.load_model_config(MappingFiles({"config.json": nonfinite})), loaded)
        self.assertEqual(copy.deepcopy(loaded), loaded)
        self.assertNotEqual(pickle.loads(pickle.dumps(loaded)), loaded)
        for text, key in (('{"rms_norm_eps": Infinity}', "rms_norm_eps"), ('{"rope_theta": NaN}', "rope_theta"),
                          ('{"rope_scaling": {"type": "linear", "factor": -Infinity}}', "rope_scaling.factor")):
            with self.subTest(text=text), self.assertRaisesRegex(mc.ConfigError, f"^config.json: key '{key}': "
                                                                                 f"expected a finite number"):
                mc.load_model_config(MappingFiles({"config.json": text.encode()}))

    def test_numeric_subclasses_give_crit_eval_results(self):
        class Count(int):
            pass

        class Real(float):
            pass

        sliding = {"num_hidden_layers": 2, "layer_types": ["sliding_attention", "full_attention"], "head_dim": 64,
                   "num_attention_heads": 8, "num_key_value_heads": 2, "sliding_window": 8}
        cases = [({"max_position_embeddings": 32768, "rope_scaling": {
                      "rope_type": "yarn", "factor": factor, "original_max_position_embeddings": original}}, None)
                 for factor in (Real(4.0), Fraction(4), 4, Count(2)) for original in (32768, Count(4096))]
        cases += [({"max_position_embeddings": Count(4096)}, None), ({**sliding, "head_dim": Count(64)}, None),
                  ({**sliding, "num_hidden_layers": Count(2), "sliding_window": Count(8)}, None)]
        cases += [(sliding, element) for element in (Real(2.0), Fraction(1, 2), Fraction(3, 7), Count(3), 0, 0.0)]
        numpy = None
        try:
            import numpy
        except ImportError:
            pass
        if numpy is not None:
            cases += [({"max_position_embeddings": 32768, "rope_scaling": {
                "rope_type": "yarn", "factor": numpy.float64(4.0), "original_max_position_embeddings": 32768}}, None),
                      ({"max_position_embeddings": 32768, "rope_scaling": {   # not an int: skipped, as in crit eval
                          "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": numpy.int64(32768)}},
                       None),
                      ({"max_position_embeddings": numpy.int64(4096)}, None),
                      ({**sliding, "num_key_value_heads": numpy.int64(4), "head_dim": numpy.int32(32)}, None)]
            cases += [(sliding, element) for element in (numpy.float64(2.0), numpy.float32(0.5), numpy.int64(2))]
        for config, element in cases:
            with self.subTest(config=config, element=element):
                if element is None:
                    for core, crit_eval in ((mc.position_limit, v2.model_limit),
                                            (mc.attention_shape, v2.attention_shape),
                                            (mc.kv_elements_per_token, v2.kv_elements_per_token)):
                        self.assertEqual(core(config), crit_eval(config))
                    self.assertEqual(mc.fixed_state_bytes(config, 1), v2.fixed_state_bytes(config, 1))
                else:
                    self.assertEqual(mc.fixed_state_bytes(config, element), v2.fixed_state_bytes(config, element))
        self.assertEqual(mc.position_limit(cases[0][0]), (131072, "yarn rope scaling: 32768 original positions x "
                                                                  "factor 4"))
        self.assertEqual(mc.position_limit(cases[2][0]), (32768, "max_position_embeddings"))   # Fraction: skipped
        model = mc.parse_model_config({"hidden_size": Count(4096), "rms_norm_eps": Real(1e-5), "rope_theta":
                                       Fraction(1, 4), "eos_token_id": [Count(2)]})
        self.assertEqual((model.hidden_size, model.rms_norm_eps, model.rope.theta, model.eos_token_ids),
                         (4096, 1e-5, 0.25, (2,)))
        self.assertEqual([type(value) for value in (model.hidden_size, model.raw["hidden_size"],
                                                    model.raw["eos_token_id"][0], model.raw["rope_theta"])],
                         [int, int, int, float])   # the frozen tree holds plain JSON numbers
        shape = mc.attention_shape({"num_hidden_layers": Count(2), "num_attention_heads": Count(8),
                                    "head_dim": Count(64)})
        self.assertEqual([type(value) for value in shape], [int] * 4)
        for element in (True, "2", None, complex(1, 0), Fraction(10 ** 400)):
            with self.subTest(element=element), self.assertRaisesRegex(ValueError, "kv_bytes_per_element must be"):
                mc.fixed_state_bytes(sliding, element)
        with self.assertRaisesRegex(mc.ConfigError, "key 'hidden_size': expected an integer"):
            mc.parse_model_config({"hidden_size": Real(4096.0)})

    def test_results_pickle_and_copy_as_equal_read_only_objects(self):
        for config in (HYBRID, DEEPSEEK, GEMMA, LLAMA, {}):
            model = mc.parse_model_config(config, "weights/config.json")
            for copied in (pickle.loads(pickle.dumps(model)), copy.deepcopy(model), copy.copy(model)):
                with self.subTest(config=config.get("model_type"), copied=type(copied).__name__):
                    self.assertEqual(copied, model)
                    self.assertEqual(copied.to_dict(), config)
                    self.assertEqual((copied.source, copied.rope, copied.quantization), (model.source, model.rope,
                                                                                         model.quantization))
                    for mapping in (copied.raw, copied.section, copied.text, copied.rope.raw):
                        self.assertIsInstance(mapping, types.MappingProxyType)
                        with self.assertRaises(TypeError):
                            mapping["new"] = 1
                    self.assertEqual(copied.position_limit(), model.position_limit())
        deepseek = pickle.loads(pickle.dumps(mc.parse_model_config(DEEPSEEK)))
        self.assertIsInstance(deepseek.quantization.raw, types.MappingProxyType)
        self.assertEqual(deepseek.quantization.raw["weight_block_size"], (128, 128))
        self.assertIsInstance(deepseek.raw["rope_scaling"], types.MappingProxyType)

    def test_documented_fallback_orders_and_defaults_hold(self):
        experts = {"num_experts": 8, "num_local_experts": 16, "n_routed_experts": 32}
        for config, expected in ((experts, (8, "num_experts")),
                                 ({**experts, "num_experts": None}, (16, "num_local_experts")),
                                 ({**experts, "num_experts": None, "num_local_experts": None},
                                  (32, "n_routed_experts"))):
            with self.subTest(config=config):
                moe = mc.parse_model_config(config).moe
                self.assertEqual((moe.experts, moe.experts_key), expected)
        eps = {"rms_norm_eps": 1e-6, "layer_norm_eps": 1e-5, "layer_norm_epsilon": 1e-4}
        for config, expected in ((eps, (1e-6, "rms_norm_eps")),
                                 ({**eps, "rms_norm_eps": None}, (1e-5, "layer_norm_eps")),
                                 ({"layer_norm_epsilon": 1e-4}, (1e-4, "layer_norm_epsilon"))):
            with self.subTest(config=config):
                model = mc.parse_model_config(config)
                self.assertEqual((model.rms_norm_eps, model.norm_eps_key), expected)
        partial = {"partial_rotary_factor": 0.5,
                   "rope_parameters": {"rope_type": "default", "partial_rotary_factor": 0.25}}
        self.assertEqual(mc.parse_model_config(partial).rope.partial_rotary_factor, 0.5)
        self.assertEqual(mc.parse_model_config({**partial, "partial_rotary_factor": None}).rope.partial_rotary_factor,
                         0.25)
        for rope, expected in (({"rope_type": "yarn", "type": "default"}, "yarn"),
                               ({"type": "linear", "rope_type": "dynamic"}, "dynamic"),
                               ({"rope_type": "", "type": "linear"}, "linear"), ({"type": "mrope"}, "mrope")):
            for key in ("rope_scaling", "rope_parameters"):
                with self.subTest(rope=rope, key=key):
                    self.assertEqual(mc.parse_model_config({key: rope}).rope.type, expected)
        self.assertEqual(mc.parse_model_config({"kv_lora_rank": 512}).mla, mc.MlaConfig(kv_lora_rank=512))
        self.assertEqual(mc.parse_model_config({"kv_lora_rank": 512}).mla.qk_rope_head_dim, 0)
        self.assertEqual(mc.fixed_state_bytes(EVAL_FIXED, 0), v2.fixed_state_bytes(EVAL_FIXED, 0))
        self.assertEqual(mc.fixed_state_bytes({"layer_types": ["sliding_attention"], "sliding_window": 8,
                                               "head_dim": 8, "num_attention_heads": 1}, 0), 0)

    def test_integers_beyond_64_bits_stay_in_untyped_keys_and_fail_in_typed_ones(self):
        # The native reader's rule: the document loads, untyped keys keep the value, typed fields refuse it.
        data = b'{"x": 123456789012345678901, "y": [-99999999999999999999], "vocab_size": 10}'
        model = mc.load_model_config(MappingFiles({"config.json": data}))
        self.assertEqual((model.get("x"), model.get("y"), model.vocab_size), (123456789012345678901,
                                                                             (-99999999999999999999,), 10))
        self.assertEqual(model.to_dict()["x"], 123456789012345678901)
        unset = 1000000000000000019884624838656   # what transformers writes for an unset model_max_length
        with self.assertRaisesRegex(mc.ConfigError, "^config.json: key 'model_max_length': integer exceeds 64 bits$"):
            mc.load_model_config(MappingFiles({"config.json": json.dumps({"model_max_length": unset}).encode()}))
        with self.assertRaisesRegex(mc.ConfigError, "^config.json: key 'model_max_length': integer exceeds 64 bits$"):
            mc.position_limit({"model_max_length": unset})
        # Length keys are read in crit eval's order and only until the first positive integer.
        earlier = mc.parse_model_config({"max_position_embeddings": 4096, "model_max_length": unset})
        self.assertEqual((earlier.max_positions, earlier.position_limit()), (4096, (4096, "max_position_embeddings")))
        self.assertEqual(mc.parse_model_config({"seq_length": (1 << 64) - 1}).max_positions, (1 << 64) - 1)
        self.assertEqual(mc.parse_model_config({"seq_length": -(1 << 63), "n_positions": 8}).max_positions, 8)
        self.assertEqual(v2.model_limit({"model_max_length": unset}), (unset, "model_max_length"))

    def test_rope_parameters_keyed_by_layer_type_fill_rope_by_layer_type(self):
        rope = {"full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1000000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.5,
                                      "original_max_position_embeddings": 4096, "mrope_section": [1, 2]}}
        model = mc.load_model_config(write_config(self.root, {"model_type": "gemma3_text", "rope_parameters": rope,
                                                              "rope_theta": 5.0}))
        self.assertEqual([layer.layer_type for layer in model.rope_by_layer_type],
                         ["full_attention", "sliding_attention"])
        full, sliding = (layer.rope for layer in model.rope_by_layer_type)
        self.assertEqual((full.type, full.factor, full.theta, full.partial_rotary_factor), ("linear", 8.0, 1e6, None))
        self.assertEqual((sliding.type, sliding.factor, sliding.theta, sliding.partial_rotary_factor,
                          sliding.original_max_position_embeddings, sliding.mrope_section),
                         ("default", None, 10000.0, 0.5, 4096, (1, 2)))
        self.assertEqual(dict(sliding.raw), {**rope["sliding_attention"], "mrope_section": (1, 2)})
        self.assertEqual((model.rope.type, model.rope.factor, model.rope.theta), ("", None, 5.0))
        self.assertIsInstance(model.rope_by_layer_type[0], mc.RopeLayer)
        self.assertEqual(pickle.loads(pickle.dumps(model)), model)
        self.assertEqual(copy.deepcopy(model).rope_by_layer_type, model.rope_by_layer_type)
        scaling = mc.parse_model_config({"rope_scaling": {"a": {"type": "yarn", "factor": 2}}})
        self.assertEqual([(layer.layer_type, layer.rope.type, layer.rope.factor)
                          for layer in scaling.rope_by_layer_type], [("a", "yarn", 2.0)])
        bare = mc.parse_model_config({"rope_theta": 5.0, "partial_rotary_factor": 0.25,
                                      "original_max_position_embeddings": 4096, "rope_scaling": {"a": {}},
                                      "rope_parameters": {"b": {"rope_type": "linear", "rope_theta": 7.0}}})
        self.assertEqual([layer.layer_type for layer in bare.rope_by_layer_type], ["a"])   # rope_scaling first
        layer = bare.rope_by_layer_type[0].rope   # nothing falls back to the section or the other rope object
        self.assertEqual((layer.type, layer.theta, layer.partial_rotary_factor, layer.original_max_position_embeddings),
                         ("", None, None, None))
        self.assertEqual((bare.rope.theta, bare.rope.partial_rotary_factor, bare.rope.original_max_position_embeddings),
                         (5.0, 0.25, 4096))
        for config in ({"rope_parameters": {"rope_type": "linear", "factor": 2.0}}, {"rope_parameters": {}},
                       {"rope_parameters": {"a": {}, "b": 1}},
                       {"rope_scaling": {"factor": 2.0}, "rope_parameters": rope}, {}):
            with self.subTest(config=config):
                self.assertEqual(mc.parse_model_config(config).rope_by_layer_type, ())
        cases = (({"rope_type": "linear", "factor": True}, "factor", "expected a finite number, got true"),
                 ({"rope_theta": "1e4"}, "rope_theta", "expected a finite number, got '1e4'"),
                 ({"partial_rotary_factor": None, "type": 3}, "type", "expected a string, got 3"),
                 ({"original_max_position_embeddings": -1}, "original_max_position_embeddings",
                  "expected an integer in"),
                 ({"long_factor": [1, 1 << 64]}, "long_factor[1]", "integer exceeds 64 bits"))
        for entry, key, message in cases:
            config = {"rope_parameters": {"sliding_attention": {}, "full_attention": entry}}
            expected = f"config.json: key 'rope_parameters.full_attention.{key}': {message}"
            with self.subTest(entry=entry), self.assertRaisesRegex(mc.ConfigError, "^" + re.escape(expected)):
                mc.parse_model_config(config)

    def test_dangling_or_non_regular_config_json_fails_loudly(self):
        (self.root / "config.json").mkdir()
        message = "not a regular file"
        for source in (self.root, DirectoryFiles(self.root)):   # a directory path names a model directory
            with self.subTest(source=source), self.assertRaisesRegex(mc.ConfigError, message):
                mc.load_model_config(source)
        (self.root / "config.json").rmdir()
        try:
            os.symlink(self.root / "missing-blob", self.root / "config.json")
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"no symlinks here: {error}")
        for source in (self.root, self.root / "config.json"):
            with self.subTest(source=source), self.assertRaisesRegex(
                    mc.ConfigError, "^" + re.escape(str(self.root / "config.json")) + ": open: "):
                mc.load_model_config(source)
        with self.assertRaisesRegex(mc.ConfigError, "^config.json: open: "):
            mc.load_model_config(DirectoryFiles(self.root))
        # A link that exists but resolves to nothing fails the same way: a loop, or a target through a file.
        (self.root / "file").write_text("{}")
        for target in ("config.json", "file/inner"):
            (self.root / "config.json").unlink()
            os.symlink(target, self.root / "config.json")
            for source in (self.root, self.root / "config.json"):
                with self.subTest(target=target, source=source), self.assertRaisesRegex(
                        mc.ConfigError, "^" + re.escape(str(self.root / "config.json")) + ": open: "):
                    mc.load_model_config(source)
            with self.subTest(target=target), self.assertRaisesRegex(mc.ConfigError, "^config.json: open: "):
                mc.load_model_config(DirectoryFiles(self.root))
        # So does one that cannot be opened (no permission); a path through a regular file is missing.
        (self.root / "config.json").unlink()
        write_config(self.root, LLAMA)
        denied = PermissionError(errno.EACCES, os.strerror(errno.EACCES))
        for source in (self.root, DirectoryFiles(self.root)):
            with self.subTest(source=source), mock.patch("crit_core.files.os.open", side_effect=denied), \
                    self.assertRaisesRegex(mc.ConfigError, "config.json: open: " + os.strerror(errno.EACCES) + "$"):
                mc.load_model_config(source)
        with self.assertRaisesRegex(mc.ConfigError, "^" + re.escape(str(self.root / "file" / "config.json")) +
                                    ": missing$"):
            mc.load_model_config(self.root / "file" / "config.json")

    def test_sources_follow_the_shared_model_files_rule(self):
        class LookAlike:   # read and names, but not a ModelFiles
            def read(self, name, max_bytes=None):
                return b"{}"

            def names(self, prefix=""):
                return []

        for source in (42, LookAlike(), b"/model", None):
            with self.subTest(source=source), self.assertRaisesRegex(TypeError, "directory path or a crit_core.files"):
                mc.load_model_config(source)
        write_config(self.root, LLAMA)
        shared = DirectoryFiles(self.root)
        self.assertEqual(mc.load_model_config(shared).vocab_size, 128256)
        write_config(self.root, {**LLAMA, "vocab_size": 7})
        with self.assertRaisesRegex(mc.ConfigError, "^config.json changed between reads$"):
            mc.load_model_config(shared)
        with self.assertRaisesRegex(mc.ConfigError, "^config.json: 9 bytes exceeds the 8-byte limit$"):
            mc.load_model_config(MappingFiles({"config.json": b'{"a": 10}'}), max_bytes=8)
        self.assertEqual(MappingFiles({}).records, ())


def layout(checkpoint):
    return {name: (t.dtype, t.shape, t.data_begin, t.data_end) for name, t in checkpoint.tensors.items()}


def header_bytes(header):
    return header if isinstance(header, bytes) else json.dumps(header).encode("utf-8")


class SafetensorsFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def raw(self, header, payload=b"", name="model.safetensors", declared_length=None, folder=None):
        data = header_bytes(header)
        path = (folder or self.root) / name
        length = len(data) if declared_length is None else declared_length
        path.write_bytes(struct.pack("<Q", length) + data + payload)
        return path

    def tensors(self, specs, name="model.safetensors", metadata=None, folder=None):
        header, payload = {}, bytearray()
        if metadata is not None:
            header["__metadata__"] = metadata
        for key, dtype, shape in specs:
            length = math.prod(shape) * st.DTYPE_BITS[dtype] // 8
            begin = len(payload)
            payload.extend(bytes(length))
            header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, len(payload)]}
        return self.raw(header, bytes(payload), name, folder=folder)

    def index(self, weights, metadata=None, extra=None, folder=None):
        value = {"weight_map": weights}
        if metadata is not None:
            value["metadata"] = metadata
        value.update(extra or {})
        path = (folder or self.root) / "model.safetensors.index.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def strict(self, path, limits=AUDIT_LIMITS, **options):
        return st.inspect_model(path, limits, **{**STRICT, **options})

    def both(self, path, limits=AUDIT_LIMITS):
        """Run strict (the audit's rules) and compatible; both must agree on rejection here."""
        return self.strict(path, limits), st.inspect_model(path, limits)

    def rejects(self, path, message, limits=AUDIT_LIMITS, policies=("strict", "compatible"), **options):
        for policy in policies:
            arguments = {"index_policy": policy, "follow_symlinks": policy != "strict", **options}
            with self.subTest(policy=policy), self.assertRaisesRegex(st.SafetensorsError, message):
                st.inspect_model(path, limits, **arguments)


class SafetensorsHeaderTests(SafetensorsFixture):
    def test_duplicate_json_fields_rejected_at_every_level(self):
        headers = (
            b'{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}',
            b'{"a":{"dtype":"U8","dtype":"I8","shape":[1],"data_offsets":[0,1]}}',
            b'{"__metadata__":{"claim":"a","claim":"b"}}',
        )
        for header in headers:
            with self.subTest(header=header):
                self.rejects(self.raw(header, b"\0\0"), "header: duplicate key")
        path = self.root / "model.safetensors.index.json"
        path.write_text('{"weight_map":{"a":"one.safetensors","a":"two.safetensors"}}')
        self.rejects(path, "duplicate key 'a'")

    def test_empty_scalar_and_out_of_order_records_are_supported(self):
        path = self.raw({
            "later": {"dtype": "BF16", "shape": [], "data_offsets": [1, 3]},
            "zero": {"dtype": "F32", "shape": [0, 100], "data_offsets": [1, 1]},
            "first": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        }, b"\0\0\0")
        for checkpoint in self.both(path):
            self.assertEqual(checkpoint.tensors["later"].nbytes, 2)
            self.assertEqual(checkpoint.tensors["zero"].nbytes, 0)
            self.assertEqual(list(checkpoint.tensors), ["first", "later", "zero"])
            self.assertEqual([t.name for t in checkpoint.shards[0].tensors], ["later", "zero", "first"])

    def test_truncated_huge_or_nonobject_headers_and_unsupported_containers(self):
        path = self.root / "model.safetensors"
        path.write_bytes(b"tiny")
        self.rejects(path, "truncated header length")
        for header, length, message in ((b"{}", 0, "header length is zero"),
                                        (b"{}", 100, "header length 100 exceeds the 2 bytes after the length prefix"),
                                        (b"{}", (1 << 64) - 1, "exceeds the 1048576-byte limit"),
                                        (b"[]", None, "header must start with '{'"),
                                        (b" {}", None, "header must start with '{'"),
                                        (b"{bad", None, "header: Expecting property name"),
                                        (b"{\xff}", None, "header: not UTF-8")):
            with self.subTest(header=header, length=length):
                self.rejects(self.raw(header, declared_length=length), message)
        self.rejects(self.raw({}, name="model.gguf"), "expected a .safetensors file, a .safetensors.index.json")
        with self.assertRaisesRegex(st.SafetensorsError, "model.gguf: expected a .safetensors file$"):
            st.inspect_safetensors(self.root / "model.gguf")

    def test_trailing_space_padding_is_accepted_and_hashed(self):
        data = json.dumps({"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode() + b"   "
        path = self.raw(data, b"\0")
        for checkpoint in self.both(path):
            shard = checkpoint.shards[0]
            self.assertEqual((shard.header_length, shard.payload_offset), (len(data), 8 + len(data)))
            self.assertEqual(shard.header_sha256, hashlib.sha256(data).hexdigest())
            self.assertEqual(checkpoint.tensors["a"].data_begin, 8 + len(data))

    def test_nonfinite_float_boolean_negative_and_overflow_dimensions_rejected(self):
        for value, message in ((True, "dimension must be an integer, not true"), (-1, "dimension -1 is negative"),
                               (1.0, "real number 1.0 where only integers"), (math.nan, "non-finite number NaN"),
                               (math.inf, "non-finite number Infinity"),
                               ((1 << 64) - 1, "dimension 18446744073709551615 exceeds"),
                               (1 << 64, "dimension: integer exceeds 64 bits")):
            with self.subTest(value=value):
                self.rejects(self.raw({"a": {"dtype": "U8", "shape": [value], "data_offsets": [0, 0]}}), message)
        self.rejects(self.raw({"a": {"dtype": "F64", "shape": [1 << 30, 1 << 30], "data_offsets": [0, 0]}}),
                     "exceeds the 68719476736-byte file limit")
        self.rejects(self.raw({"a": {"dtype": "F64", "shape": [1 << 30, 1 << 30], "data_offsets": [0, 0]}}),
                     "exceeds the 4398046511104-byte file limit", limits=None)
        self.rejects(self.raw({"a": {"dtype": "F64", "shape": [1] * 9, "data_offsets": [0, 0]}}),
                     "rank 9 exceeds the 8 limit")

    def test_bad_offsets_overlap_holes_tail_and_storage_mismatch(self):
        record = {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}
        for offsets, message in (([True, 2], "data offset must be an integer, not true"),
                                 ([-1, 1], "data offset -1 is negative"),
                                 ([0, 3], "data offset 3 is past the 2-byte payload"),
                                 ([0, (1 << 64) - 1], "past the 2-byte payload"),
                                 ([2, 0], "data_offsets end 0 precedes begin 2"),
                                 ([0], "data_offsets must be \\[begin, end\\]"),
                                 ("02", "data_offsets must be \\[begin, end\\]")):
            with self.subTest(offsets=offsets):
                self.rejects(self.raw({"a": dict(record, data_offsets=offsets)}, b"\0\0"), message)
        for header, payload, message in (({"a": record, "b": record}, b"\0\0",
                                          "overlap at payload offset 2 before tensor 'b'"),
                                         ({"a": dict(record, data_offsets=[1, 3])}, b"\0\0\0",
                                          "hole at payload offset 0"),
                                         ({"a": record}, b"\0\0\0", "payload has 1 unindexed trailing bytes"),
                                         ({"a": dict(record, dtype="F32")}, b"\0\0",
                                          "data_offsets \\[0, 2\\) hold 2 bytes but shape \\[2\\] of F32 needs 8"),
                                         ({"a": record}, b"\0", "data offset 2 is past the 1-byte payload")):
            with self.subTest(header=header):
                self.rejects(self.raw(header, payload), message)

    def test_malformed_tensor_metadata_and_unsupported_dtype_rejected(self):
        for dtype, message in (("F4", "1 elements of F4 do not fill whole bytes"), ("F6", "unsupported dtype 'F6'"),
                               ("F8_UNKNOWN", "unsupported dtype"), ([], "unsupported dtype an array"),
                               ("bad", "unsupported dtype 'bad'")):
            with self.subTest(dtype=dtype):
                self.rejects(self.raw({"a": {"dtype": dtype, "shape": [1], "data_offsets": [0, 1]}}, b"\0"), message)
        for metadata, message in (([], "__metadata__ must map strings to strings"),
                                  ({"claim": True}, "__metadata__ must map strings to strings"),
                                  ({"claim": {"dtype": "BF16"}}, "__metadata__ must map strings to strings"),
                                  ({"claim": "\ud800"}, "header: unpaired surrogate escape"),
                                  ({"\udc00": "x"}, "header: unpaired surrogate escape"),
                                  (None, "__metadata__ must map")):
            with self.subTest(metadata=repr(metadata)):
                self.rejects(self.raw({"__metadata__": metadata}), message)
        for record in ([], {"dtype": "U8", "shape": [1]},
                       {"dtype": "U8", "shape": [1], "data_offsets": [0, 1], "other": 0}):
            with self.subTest(record=record):
                self.rejects(self.raw({"a": record}, b"\0"), "expected exactly the keys dtype, shape and data_offsets")
        self.rejects(self.raw({"a": {"dtype": "U8", "shape": 1, "data_offsets": [0, 1]}}, b"\0"),
                     "shape must be an array")
        for name, message in (("", "tensor name must be a non-empty string"), ("a\0b", "contains NUL"),
                              ("\ud800", "header: unpaired surrogate escape")):
            with self.subTest(name=repr(name)):
                self.rejects(self.raw({name: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}, b"\0"), message)
        checkpoint = st.inspect_model(self.tensors((("a", "U8", (1,)),), metadata={"format": "pt", "é": "ü"}))
        self.assertEqual(dict(checkpoint.shards[0].metadata), {"format": "pt", "é": "ü"})

    def test_limits_bound_files_names_rank_and_header_before_payload_read(self):
        path = self.tensors((("a", "BF16", (2,)),))
        for limits, message in ((dataclasses.replace(AUDIT_LIMITS, max_header_bytes=8), "exceeds the 8-byte limit"),
                                (dataclasses.replace(AUDIT_LIMITS, max_file_bytes=8), "exceed the 8-byte file limit"),
                                (dataclasses.replace(AUDIT_LIMITS, max_dimension=1), "dimension 2 exceeds 1"),
                                (dataclasses.replace(AUDIT_LIMITS, max_total_bytes=8), "exceed the 8-byte checkpoint")):
            with self.subTest(limits=limits):
                self.rejects(path, message, limits)
        path = self.tensors((("a", "U8", (1,)), ("b", "U8", (1,))))
        self.rejects(path, "more than 1 tensors", dataclasses.replace(AUDIT_LIMITS, max_tensors=1))
        path = self.tensors((("long", "U8", (1,)),))
        self.rejects(path, "tensor name of 4 bytes exceeds the 3-byte limit",
                     dataclasses.replace(AUDIT_LIMITS, max_name_bytes=3))
        for kwargs in ({"max_rank": 0}, {"max_shards": True}, {"max_file_bytes": 1 << 64}, {"max_tensors": 1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(st.SafetensorsError, "limit max_"):
                st.Limits(**kwargs)
        with self.assertRaisesRegex(st.SafetensorsError, "limits must be a Limits instance"):
            st.inspect_model(path, {"max_tensors": 1})
        with self.assertRaisesRegex(st.SafetensorsError, "index_policy must be 'strict' or 'compatible'"):
            st.inspect_model(path, index_policy="lenient")

    def test_reader_never_reads_tensor_payload_and_detects_metadata_read_mutation(self):
        path = self.tensors((("a", "BF16", (2,)),))
        boundary = 8 + int.from_bytes(path.read_bytes()[:8], "little")
        real_read = st.os.read
        requests = []
        mutate = False

        def header_only_read(fd, count):   # every OS-level read the reader issues passes through here
            position = os.lseek(fd, 0, os.SEEK_CUR)
            requests.append((position, count))
            if position + count > boundary:
                raise AssertionError("a metadata read asked for tensor data")
            result = real_read(fd, count)
            if mutate and position + len(result) == boundary:
                info = path.stat()
                os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
            return result

        with mock.patch.object(st.os, "read", side_effect=header_only_read):
            st.inspect_model(path)
            self.assertEqual(requests, [(0, 8), (8, boundary - 8)])
            mutate = True
            with self.assertRaisesRegex(st.SafetensorsError, "file changed during metadata read"):
                st.inspect_model(path)

    def test_operating_system_reads_stop_at_the_header(self):
        # Counts bytes at the system-call level, so a buffered stream reading ahead into the payload is caught.
        def bytes_read():
            with open("/proc/self/io", encoding="ascii") as handle:
                return next(int(line.split()[1]) for line in handle if line.startswith("rchar:"))
        try:
            bytes_read()
        except (OSError, StopIteration, ValueError):
            self.skipTest("Host has no readable /proc/self/io")
        header = {"w": {"dtype": "U8", "shape": [1 << 20], "data_offsets": [0, 1 << 20]}}
        path = self.raw(header, bytes(1 << 20))
        expected = 8 + len(header_bytes(header))
        spans, overheads = [], []
        for _ in range(5):   # minimums, in case another thread of this process reads at the same time
            before = bytes_read()
            st.inspect_safetensors(path)
            spans.append(bytes_read() - before)
            first = bytes_read()
            overheads.append(bytes_read() - first)   # what one look at /proc/self/io adds
        self.assertGreaterEqual(min(spans), expected)
        self.assertLessEqual(min(spans) - min(overheads), expected + 64)   # slack for the counter's digit count

    def test_short_operating_system_reads_are_completed_and_early_ends_are_truncation(self):
        path = self.tensors((("a", "BF16", (2,)), ("b", "U8", (3,))))
        index = self.index({"a": "model.safetensors", "b": "model.safetensors"})
        real_read = st.os.read
        expected = layout(st.inspect_model(index))
        with mock.patch.object(st.os, "read", side_effect=lambda fd, count: real_read(fd, min(count, 3))):
            self.assertEqual(layout(st.inspect_model(index)), expected)
        def ends_after(cut):   # each file reads as if it ended after `cut` bytes
            consumed = {}

            def read(fd, count):
                taken = consumed.get(fd, 0)
                result = real_read(fd, max(0, min(count, cut - taken)))
                consumed[fd] = taken + len(result)
                return result
            return read
        for source, cut, message in ((path, 4, "model.safetensors: truncated header length"),
                                     (path, 8 + 5, "model.safetensors: truncated header$"),
                                     (index, 10, "index changed or truncated while reading")):
            with self.subTest(source=source.name, cut=cut), \
                    mock.patch.object(st.os, "read", side_effect=ends_after(cut)), \
                    self.assertRaisesRegex(st.SafetensorsError, message):
                st.inspect_model(source)
        grown = []

        def grows(fd, count):   # the index gains a byte after its size was taken
            if not grown:
                grown.append(True)
                with open(index, "ab") as handle:
                    handle.write(b" ")
            return real_read(fd, count)
        with mock.patch.object(st.os, "read", side_effect=grows), \
                self.assertRaisesRegex(st.SafetensorsError, "index changed or truncated while reading"):
            st.inspect_model(index)

    def test_special_files_are_refused_before_opening(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("Host has no named pipes")
        fifo = self.root / "pipe.safetensors"
        os.mkfifo(fifo)
        for follow in (True, False):
            with self.subTest(follow=follow), self.assertRaisesRegex(st.SafetensorsError, "expected a regular file"):
                st.inspect_safetensors(fifo, follow_symlinks=follow)

    def test_dtype_widths_and_sub_byte_extents(self):
        self.assertEqual(dict(st.DTYPE_BITS), {
            "BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8, "U16": 16, "I16": 16, "F16": 16,
            "BF16": 16, "U32": 32, "I32": 32, "F32": 32, "U64": 64, "I64": 64, "F64": 64, "F4": 4, "F6_E2M3": 6,
            "F6_E3M2": 6})
        specs = [(f"t.{dtype}", dtype, (2, 4)) for dtype in st.DTYPE_BITS]
        checkpoint = st.inspect_model(self.tensors(specs))
        for dtype, bits in st.DTYPE_BITS.items():
            with self.subTest(dtype=dtype):
                tensor = checkpoint.tensors[f"t.{dtype}"]
                self.assertEqual((tensor.bits, tensor.elements, tensor.nbytes), (bits, 8, bits))
        for dtype, shape, nbytes in (("F4", (2, 3), 3), ("F6_E2M3", (4,), 3), ("F6_E3M2", (2, 2), 3),
                                     ("F8_E8M0", (5,), 5), ("F4", (0, 3), 0), ("F4", (), None), ("F4", (3,), None),
                                     ("F6_E2M3", (2,), None), ("F6_E3M2", (5,), None)):
            with self.subTest(dtype=dtype, shape=shape):
                size = 1 if nbytes is None else nbytes
                header = {"a": {"dtype": dtype, "shape": list(shape), "data_offsets": [0, size]}}
                path = self.raw(header, bytes(size))
                if nbytes is None:
                    self.rejects(path, f"elements of {dtype} do not fill whole bytes", limits=None,
                                 policies=("compatible",))
                else:
                    self.assertEqual(st.inspect_model(path).tensors["a"].nbytes, nbytes)

    def test_element_alignment_uses_absolute_file_offsets(self):
        header = {"byte": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
                  "half": {"dtype": "BF16", "shape": [1], "data_offsets": [1, 3]},
                  "nibbles": {"dtype": "F4", "shape": [2], "data_offsets": [3, 4]},
                  "word": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}
        data = json.dumps(header).encode()
        data += b" " * (-(8 + len(data)) % 8)   # payload starts on an 8-byte boundary
        checkpoint = st.inspect_model(self.raw(data, bytes(8)))
        aligned = {name: tensor.element_aligned for name, tensor in checkpoint.tensors.items()}
        self.assertEqual(aligned, {"byte": True, "half": False, "nibbles": True, "word": True})

    def test_production_limits_accept_large_real_world_layouts(self):
        self.assertEqual(st.Limits(), st.Limits.production())
        self.assertEqual(dataclasses.asdict(st.Limits.production()), {
            "max_header_bytes": 100_000_000, "max_index_bytes": 64 << 20, "max_tensors": 1 << 20, "max_shards": 4096,
            "max_rank": 16, "max_dimension": 1 << 48, "max_file_bytes": 1 << 42, "max_total_bytes": 1 << 46,
            "max_name_bytes": 4096})
        pad = "x" * 120
        specs = [(f"model.layers.{index}.{pad}", "BF16", (1,)) for index in range(9000)]
        path = self.tensors(specs)
        self.assertGreater(8 + int.from_bytes(path.read_bytes()[:8], "little"), 1 << 20)
        self.assertEqual(len(st.inspect_model(path).tensors), 9000)
        self.rejects(path, "exceeds the 1048576-byte limit", policies=("strict",))
        rank = self.tensors((("deep", "F32", (1,) * 16),), name="rank.safetensors")
        self.assertEqual(st.inspect_model(rank).tensors["deep"].shape, (1,) * 16)
        self.rejects(self.raw({"a": {"dtype": "U8", "shape": [1] * 17, "data_offsets": [0, 1]}}, b"\0"),
                     "rank 17 exceeds the 16 limit", limits=None)
        self.rejects(self.raw(b"{}", declared_length=100_000_001), "header length 100000001 exceeds the "
                                                                  "100000000-byte limit", limits=None)
        folder = self.root / "sharded"
        folder.mkdir()
        weights = {}
        for shard in range(40):
            name = f"model-{shard + 1:05d}-of-00040.safetensors"
            group = specs[shard::40]
            self.tensors(group, name=name, folder=folder)
            weights.update({key: name for key, _, _ in group})
        index = self.index(weights, {"total_size": 18000, "total_parameters": 9000}, folder=folder)
        self.assertGreater(index.stat().st_size, 1 << 20)
        checkpoint = st.inspect_model(folder)
        self.assertEqual((len(checkpoint.shards), len(checkpoint.tensors), checkpoint.tensor_bytes, checkpoint.notes),
                         (40, 9000, 18000, ()))
        self.assertEqual(checkpoint.index_sha256, hashlib.sha256(index.read_bytes()).hexdigest())
        self.rejects(index, "index size .* is not in \\[1, 1048576\\] bytes", policies=("strict",))

    def test_header_only_files_are_read_only_on_request(self):
        header = {"lm_head.weight": {"dtype": "BF16", "shape": [64, 64], "data_offsets": [0, 8192]},
                  "norm.weight": {"dtype": "BF16", "shape": [64], "data_offsets": [8192, 8320]}}
        path = self.raw(header)
        self.rejects(path, "data offset 8192 is past the 0-byte payload", limits=None)
        checkpoint = st.inspect_model(path, header_only=True)
        self.assertTrue(checkpoint.header_only and checkpoint.shards[0].header_only)
        self.assertEqual(checkpoint.tensor_bytes, 8320)
        self.assertEqual(checkpoint.tensors["norm.weight"].data_end, checkpoint.shards[0].payload_offset + 8320)
        self.assertFalse(st.inspect_model(self.raw(header, bytes(8320), name="full.safetensors")).header_only)
        self.assertEqual(len(st.inspect_model(self.raw(header, b"\0" * 9000, name="tail.safetensors"),
                                              header_only=True).tensors), 2)
        for broken, message in (({"a": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]}},
                                 "hole at payload offset 0"),
                                ({"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                                  "b": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}, "overlap"),
                                ({"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 3]}}, "hold 3 bytes"),
                                ({"a": {"dtype": "U8", "shape": [1 << 43], "data_offsets": [0, 1 << 43]}},
                                 "exceeds the 4398046511104-byte file limit")):
            with self.subTest(header=broken):
                self.rejects(self.raw(broken, name="broken.safetensors"), message, limits=None, header_only=True)
        huge = self.raw({"a": {"dtype": "U8", "shape": [1 << 32], "data_offsets": [0, 1 << 32]}}, name="huge.safetensors")
        self.assertEqual(st.inspect_model(huge, header_only=True).tensors["a"].nbytes, 1 << 32)
        self.rejects(huge, "dimension 4294967296 exceeds 2147483647", header_only=True)
        folder = self.root / "planning"
        folder.mkdir()
        self.raw(header, name="model-00001-of-00001.safetensors", folder=folder)
        self.index({name: "model-00001-of-00001.safetensors" for name in header}, {"total_size": 8320}, folder=folder)
        self.assertEqual(len(st.inspect_model(folder, header_only=True, index_policy="strict").tensors), 2)

    def test_header_only_declared_extents_count_toward_the_checkpoint_limit(self):
        folder = self.root / "declared"
        folder.mkdir()
        names = [f"model-{shard + 1:05d}-of-00003.safetensors" for shard in range(3)]
        for shard, name in enumerate(names):
            self.raw({f"w{shard}": {"dtype": "U8", "shape": [1 << 42], "data_offsets": [0, 1 << 42]}}, name=name,
                     folder=folder)
        two_tib = st.Limits(max_total_bytes=1 << 43)
        index = self.index({"w0": names[0]}, folder=folder)
        self.assertEqual(st.inspect_model(index, two_tib, header_only=True).tensor_bytes, 1 << 42)
        with self.assertRaisesRegex(st.SafetensorsError, "checkpoint files exceed the 8796093022208-byte limit"):
            st.inspect_model(index, two_tib, header_only=True, extra_files=names[1:])
        index = self.index({f"w{shard}": name for shard, name in enumerate(names)}, folder=folder)
        for policy in st.IndexPolicy.ALL:
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(st.SafetensorsError, "checkpoint files exceed the 8796093022208-byte limit"):
                    st.inspect_model(folder, two_tib, header_only=True, index_policy=policy)
                checkpoint = st.inspect_model(folder, st.Limits(max_total_bytes=(3 << 42) + 4096), header_only=True,
                                              index_policy=policy)
                self.assertEqual(checkpoint.tensor_bytes, 3 << 42)
        with self.assertRaisesRegex(st.SafetensorsError, f"{names[0]}: \\d+ bytes exceed the 4398046511104-byte "
                                                         f"checkpoint limit"):
            st.inspect_model(folder / names[0], st.Limits(max_total_bytes=1 << 42), header_only=True)
        self.assertLessEqual(st.inspect_model(folder, header_only=True).tensor_bytes, st.Limits().max_total_bytes)

    def test_files_replaced_between_check_and_open_are_refused(self):
        path = self.tensors((("a", "U8", (1,)),))
        other = self.tensors((("b", "U8", (2,)),), name="other.safetensors")
        index = self.index({"a": "model.safetensors"})
        real_open = st.os.open

        def swap(target, flags, *args, **kwargs):
            if Path(target) == path and other.exists():
                os.replace(other, path)
            return real_open(target, flags, *args, **kwargs)

        for source in (path, index):
            with self.subTest(source=source.name), mock.patch.object(st.os, "open", side_effect=swap), \
                    self.assertRaisesRegex(st.SafetensorsError, "model.safetensors: file changed while opening"):
                st.inspect_model(source)
            other = self.tensors((("b", "U8", (2,)),), name="other.safetensors")
            self.tensors((("a", "U8", (1,)),))
        real_fstat = st.os.fstat

        def special(fd):
            info = real_fstat(fd)
            return types.SimpleNamespace(st_mode=stat_fifo, st_dev=info.st_dev, st_ino=info.st_ino,
                                         st_size=info.st_size, st_mtime_ns=info.st_mtime_ns,
                                         st_ctime_ns=info.st_ctime_ns)
        stat_fifo = 0o010644
        with mock.patch.object(st.os, "fstat", side_effect=special), \
                self.assertRaisesRegex(st.SafetensorsError, "file changed while opening"):
            st.inspect_safetensors(path)
        self.assertEqual(list(st.inspect_model(index).tensors), ["a"])


class SafetensorsIndexTests(SafetensorsFixture):
    def test_multishard_index_reads_actual_header_names_and_total_size(self):
        self.tensors((("a", "BF16", (2,)),), "first.safetensors")
        self.tensors((("b", "I32", (3,)),), "second.safetensors")
        path = self.index({"a": "first.safetensors", "b": "second.safetensors"}, {"total_size": 16})
        for checkpoint in self.both(path):
            self.assertEqual(set(checkpoint.tensors), {"a", "b"})
            self.assertEqual([shard.path.name for shard in checkpoint.shards],
                             ["first.safetensors", "second.safetensors"])
            self.assertEqual(checkpoint.tensors["b"].dtype, "I32")
            self.assertEqual(checkpoint.tensors["b"].shard, (self.root / "second.safetensors").absolute())
            self.assertEqual(checkpoint.index_sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual((checkpoint.source, checkpoint.notes, checkpoint.tensor_bytes), (path.absolute(), (), 16))
        self.assertEqual(self.strict(path).index_policy, "strict")

    def test_index_shards_in_a_subfolder(self):
        (self.root / "tensors").mkdir()
        self.tensors((("a", "BF16", (2,)),), "first.safetensors", folder=self.root / "tensors")
        self.tensors((("b", "I32", (3,)),), "second.safetensors", folder=self.root / "tensors")
        path = self.index({"a": "tensors/first.safetensors", "b": "tensors/second.safetensors"})
        for checkpoint in self.both(path):
            self.assertEqual(set(checkpoint.tensors), {"a", "b"})
            self.assertEqual(checkpoint.tensors["b"].shard, (self.root / "tensors" / "second.safetensors").absolute())

    def test_index_missing_undeclared_duplicate_and_wrong_shard_tensors(self):
        self.tensors((("a", "BF16", (2,)),), "first.safetensors")
        self.tensors((("b", "I32", (3,)),), "second.safetensors")
        for weights, message in (({"a": "first.safetensors", "missing": "first.safetensors"},
                                  "1 weight_map tensors are absent from the shard headers, such as 'missing'"),
                                 ({"a": "second.safetensors", "b": "first.safetensors"},
                                  "weight_map puts tensor 'a' in 'second.safetensors' but 'first.safetensors' holds it"),
                                 ({"b": "first.safetensors"},
                                  "shard 'first.safetensors' holds tensor 'a', which weight_map does not list")):
            with self.subTest(weights=weights):
                self.rejects(self.index(weights), message)
        self.tensors((("a", "I32", (1,)),), "second.safetensors")
        self.rejects(self.index({"a": "first.safetensors", "b": "second.safetensors"}),
                     "tensor 'a' appears in shards 'first.safetensors' and 'second.safetensors'")

    def test_index_path_escape_and_portability_rejection(self):
        for name, message in (("../outside.safetensors", "not a portable relative"), ("/tmp/a.safetensors", "not a portable"),
                              ("C:\\a.safetensors", "not a portable"), ("sub/../a.safetensors", "not a portable"),
                              ("sub//a.safetensors", "not a portable"), ("sub/.hidden/a.safetensors", "not a portable"),
                              ("CON/a.safetensors", "reserved device name"), ("a/" * 8 + "a.safetensors", "not a portable"),
                              ("a\\b.safetensors", "not a portable"), ("file:stream.safetensors", "not a portable"),
                              ("CON.safetensors", "reserved device name"), ("lpt1.x.safetensors", "reserved device"),
                              ("a.gguf", "not a portable"), (".safetensors", "not a portable"),
                              (3, "must be a non-empty string")):
            with self.subTest(name=name):
                self.rejects(self.index({"a": name}), message)

    def test_symlinks_are_followed_by_default_and_refused_on_request(self):
        blobs, snapshot = self.root / "blobs", self.root / "snapshots" / "rev"
        blobs.mkdir()
        snapshot.mkdir(parents=True)
        actual = self.tensors((("a", "U8", (1,)),), "0f1e2d", folder=blobs)
        index = self.index({"a": "model-00001-of-00001.safetensors"}, {"total_size": 1}, folder=blobs)
        blob_index = blobs / "3c4b5a"
        index.rename(blob_index)
        try:
            (snapshot / "model-00001-of-00001.safetensors").symlink_to(actual)
            (snapshot / "model.safetensors.index.json").symlink_to(blob_index)
        except OSError as exc:
            self.skipTest(f"Host does not permit symlink creation: {exc}")
        checkpoint = st.inspect_model(snapshot)
        self.assertEqual(checkpoint.source, (snapshot / "model.safetensors.index.json").absolute())
        self.assertEqual(checkpoint.tensors["a"].shard, (snapshot / "model-00001-of-00001.safetensors").absolute())
        self.assertEqual(checkpoint.shards[0].file_identity[1], actual.stat().st_ino)
        self.assertEqual(len(self.strict(snapshot, follow_symlinks=True).tensors), 1)
        self.rejects(snapshot, "expected a regular file that is not a symlink", follow_symlinks=False,
                     policies=("compatible",))
        self.rejects(snapshot, "expected a regular file that is not a symlink", policies=("strict",))
        shard_only = self.root / "flat"
        shard_only.mkdir()
        self.index({"a": "link.safetensors"}, folder=shard_only)
        (shard_only / "link.safetensors").symlink_to(actual)
        self.assertEqual(len(st.inspect_model(shard_only).tensors), 1)
        self.rejects(shard_only, "link.safetensors: expected a regular file that is not a symlink", policies=("strict",))
        dangling = self.root / "dangling"
        dangling.mkdir()
        (dangling / "model.safetensors").symlink_to(self.root / "absent")
        with self.assertRaises(FileNotFoundError):
            st.inspect_model(dangling)

    def test_index_metadata_and_bounds(self):
        self.tensors((("a", "BF16", (2,)),), "first.safetensors")
        self.tensors((("b", "BF16", (2,)),), "second.safetensors")
        for metadata, message in (({"total_size": True}, "total_size must be an integer, not true"),
                                  ({"total_size": None}, "total_size must be an integer, not null"),
                                  ({"total_size": -1}, "total_size -1 is negative"),
                                  ([], "index metadata must be an object"),
                                  ({"total_size": 1 << 60}, "total_size 1152921504606846976 exceeds")):
            with self.subTest(metadata=metadata):
                self.rejects(self.index({"a": "first.safetensors"}, metadata), message)
        stale = self.index({"a": "first.safetensors"}, {"total_size": 99})
        self.rejects(stale, "index metadata total_size 99 differs from the 4 tensor bytes", policies=("strict",))
        self.assertEqual(st.inspect_model(stale).notes,
                         ("index metadata total_size 99 differs from the 4 tensor bytes in the shards",))
        unknown = self.index({"a": "first.safetensors"}, {"unknown": 1, "total_parameters": 2, "total_size": 4})
        self.rejects(unknown, "unsupported index metadata keys \\['total_parameters', 'unknown'\\]", policies=("strict",))
        self.assertEqual(st.inspect_model(unknown).notes, ("index metadata key 'unknown' is not interpreted",))
        bad_parameters = self.index({"a": "first.safetensors"}, {"total_parameters": "2"})
        self.rejects(bad_parameters, "total_parameters must be an integer", policies=("compatible",))
        for weights, message in (({}, "weight_map must be a non-empty object"), ([], "weight_map must be a non-empty"),
                                 ({"a": None}, "weight_map shard for 'a' must be a non-empty string, not null"),
                                 ({"": "first.safetensors"}, "weight_map tensor name must be a non-empty")):
            with self.subTest(weights=weights):
                self.rejects(self.index(weights), message)
        extra = self.index({"a": "first.safetensors"}, extra={"config": {}})
        self.rejects(extra, "unsupported index keys \\['config'\\]", policies=("strict",))
        self.assertEqual(st.inspect_model(extra).notes, ("index key 'config' is not interpreted",))
        missing = self.root / "model.safetensors.index.json"
        missing.write_text('{"metadata": {}}')
        self.rejects(missing, "index has no weight_map")
        missing.write_text("")
        self.rejects(missing, "index size 0 is not in")
        path = self.index({"a": "first.safetensors", "b": "second.safetensors"})
        for limits, message in ((dataclasses.replace(AUDIT_LIMITS, max_shards=1), "2 shards exceed the 1-shard limit"),
                                (dataclasses.replace(AUDIT_LIMITS, max_tensors=1), "weight_map lists 2 tensors"),
                                (dataclasses.replace(AUDIT_LIMITS, max_index_bytes=8), "index size .* is not in"),
                                (dataclasses.replace(AUDIT_LIMITS, max_total_bytes=8), "checkpoint files exceed")):
            with self.subTest(limits=limits):
                self.rejects(path, message, limits)
        with self.assertRaises(FileNotFoundError):
            st.inspect_model(self.index({"a": "first.safetensors", "c": "absent.safetensors"}))

    def test_directory_resolution_never_guesses(self):
        def folder(name, files):
            path = self.root / name
            path.mkdir()
            for file in files:
                self.tensors(((file + ".t", "U8", (1,)),), file, folder=path)
            return path
        sharded = folder("sharded", ("model-00001-of-00001.safetensors", "stray.safetensors"))
        self.index({"model-00001-of-00001.safetensors.t": "model-00001-of-00001.safetensors"}, folder=sharded)
        self.assertEqual(list(st.inspect_model(sharded).tensors), ["model-00001-of-00001.safetensors.t"])
        single = folder("single", ("model.safetensors", "model_mtp.safetensors"))
        self.assertEqual(st.inspect_model(single).source, (single / "model.safetensors").absolute())
        only = folder("only", ("consolidated.safetensors",))
        (only / "notes.txt").write_text("not a shard")
        self.assertEqual(list(st.inspect_model(only).tensors), ["consolidated.safetensors.t"])
        with self.assertRaisesRegex(st.SafetensorsError, "no model.safetensors.index.json, model.safetensors or other"):
            st.inspect_model(folder("empty", ()))
        both = folder("both", ("model.safetensors", "model-00001-of-00001.safetensors"))
        self.index({"model-00001-of-00001.safetensors.t": "model-00001-of-00001.safetensors"}, folder=both)
        checkpoint = st.inspect_model(both)
        self.assertEqual((checkpoint.source, list(checkpoint.tensors)),
                         ((both / "model.safetensors.index.json").absolute(), ["model-00001-of-00001.safetensors.t"]))
        many = folder("many", [f"part{index}.safetensors" for index in range(10)])
        with self.assertRaisesRegex(st.SafetensorsError, "10 .safetensors files and no index; name one: "
                                                         "part0.safetensors, part1.safetensors, .*, \\.\\.\\.$"):
            st.inspect_model(many)

    def test_extra_files_join_the_checkpoint_without_duplicates(self):
        self.tensors((("model.layers.0.w", "BF16", (2,)),), "model-00001-of-00001.safetensors")
        self.tensors((("mtp.layers.0.w", "BF16", (2,)),), "model_mtp.safetensors")
        index = self.index({"model.layers.0.w": "model-00001-of-00001.safetensors"}, {"total_size": 4})
        checkpoint = st.inspect_model(self.root, extra_files=["model_mtp.safetensors"], index_policy="strict")
        self.assertEqual(list(checkpoint.tensors), ["model.layers.0.w", "mtp.layers.0.w"])
        self.assertEqual([shard.path.name for shard in checkpoint.shards],
                         ["model-00001-of-00001.safetensors", "model_mtp.safetensors"])
        self.assertEqual(st.inspect_model(self.root).tensors.keys(), {"model.layers.0.w"})
        single = st.inspect_model(self.root / "model_mtp.safetensors", extra_files=("model-00001-of-00001.safetensors",))
        self.assertEqual(len(single.tensors), 2)
        self.tensors((("model.layers.0.w", "BF16", (2,)),), "copy.safetensors")
        for extras, message in ((["copy.safetensors"], "tensor 'model.layers.0.w' appears in "
                                                        "'model-00001-of-00001.safetensors' and extra file 'copy"),
                                (["model-00001-of-00001.safetensors"], "is already part of the checkpoint"),
                                (["../model_mtp.safetensors"], "extra file name '../model_mtp.safetensors' is not"),
                                ("model_mtp.safetensors", "extra_files must be a sequence of file names"),
                                (["model_mtp.safetensors"] * 2, "listed twice")):
            with self.subTest(extras=extras), self.assertRaisesRegex(st.SafetensorsError, message):
                st.inspect_model(index, extra_files=extras)
        with self.assertRaisesRegex(st.SafetensorsError, "2 files exceed the 1-shard limit"):
            st.inspect_model(index, dataclasses.replace(AUDIT_LIMITS, max_shards=1),
                             extra_files=["model_mtp.safetensors"])

    def test_extra_files_count_toward_tensor_byte_and_shard_limits(self):
        shard = self.tensors((("model.w", "BF16", (2,)),), "model-00001-of-00001.safetensors")
        extra = self.tensors((("mtp.a", "BF16", (2,)), ("mtp.b", "BF16", (2,))), "model_mtp.safetensors")
        index = self.index({"model.w": "model-00001-of-00001.safetensors"})
        two = dataclasses.replace(AUDIT_LIMITS, max_tensors=2)
        self.assertEqual(len(st.inspect_model(index, two).tensors), 1)
        self.assertEqual(len(st.inspect_model(extra, two).tensors), 2)
        with self.assertRaisesRegex(st.SafetensorsError, "more than 2 tensors across files"):
            st.inspect_model(index, two, extra_files=["model_mtp.safetensors"])
        files = index.stat().st_size + shard.stat().st_size
        for limit, accepted in ((files, True), (files + extra.stat().st_size - 1, False),
                                (files + extra.stat().st_size, True)):
            limits = dataclasses.replace(AUDIT_LIMITS, max_total_bytes=limit)
            with self.subTest(limit=limit):
                if accepted:
                    st.inspect_model(index, limits, extra_files=["model_mtp.safetensors"] if limit > files else ())
                else:
                    with self.assertRaisesRegex(st.SafetensorsError, f"checkpoint files exceed the {limit}-byte limit"):
                        st.inspect_model(index, limits, extra_files=["model_mtp.safetensors"])
        def endless():   # an iterable is read only one name past the shard limit
            for number in itertools.count():
                if number > st.Limits().max_shards:
                    raise AssertionError("extra_files was read past the shard limit")
                yield f"x{number}.safetensors"
        with self.assertRaisesRegex(st.SafetensorsError, "extra_files lists more than 4096 files, the shard limit"):
            st.inspect_model(index, extra_files=endless())
        with self.assertRaisesRegex(st.SafetensorsError, "extra_files lists more than 32 files"):
            st.inspect_model(self.root / "absent.safetensors", AUDIT_LIMITS,
                             extra_files=[f"x{number}.safetensors" for number in range(40000)])

    def test_results_are_frozen_and_read_only(self):
        self.tensors((("b", "U8", (1,)), ("a", "U8", (2,))))
        checkpoint = st.inspect_model(self.root)
        self.assertEqual(list(checkpoint.tensors), ["a", "b"])
        with self.assertRaises(TypeError):
            checkpoint.tensors["c"] = checkpoint.tensors["a"]
        with self.assertRaises(TypeError):
            checkpoint.shards[0].metadata["c"] = "d"
        for value, attribute in ((checkpoint, "notes"), (checkpoint.shards[0], "size"),
                                 (checkpoint.tensors["a"], "name")):
            with self.subTest(attribute=attribute), self.assertRaises(dataclasses.FrozenInstanceError):
                setattr(value, attribute, None)
        self.assertIsInstance(checkpoint.shards[0].tensors, tuple)
        self.assertEqual(len(checkpoint.shards[0].file_identity), 5)

    def test_results_pickle_and_copy_as_equal_read_only_objects(self):
        self.tensors((("a", "BF16", (2,)),), "first.safetensors", metadata={"format": "pt"})
        self.tensors((("b", "U8", (3,)),), "second.safetensors")
        checkpoint = st.inspect_model(self.index({"a": "first.safetensors", "b": "second.safetensors"},
                                                 {"total_size": 99, "x": 1}))
        self.assertEqual(len(checkpoint.notes), 2)
        for copied in (pickle.loads(pickle.dumps(checkpoint)), copy.deepcopy(checkpoint), copy.copy(checkpoint)):
            with self.subTest(copied=copied is checkpoint):
                self.assertEqual(copied, checkpoint)
                self.assertEqual(list(copied.tensors), ["a", "b"])
                self.assertEqual(copied.shards[0].metadata, {"format": "pt"})
                for mapping in (copied.tensors, copied.shards[0].metadata):
                    self.assertIsInstance(mapping, types.MappingProxyType)
                    with self.assertRaises(TypeError):
                        mapping["c"] = None
        shard = checkpoint.shards[0]
        self.assertEqual(pickle.loads(pickle.dumps(shard)), shard)
        self.assertEqual(pickle.loads(pickle.dumps(checkpoint.tensors["a"])), checkpoint.tensors["a"])

    def test_elements_and_paths_of_results(self):
        path = self.tensors((("empty", "F32", (0, 3)), ("scalar", "F32", ()), ("matrix", "U8", (2, 3))))
        tensors = st.inspect_model(path).tensors
        self.assertEqual({name: (t.elements, t.nbytes) for name, t in tensors.items()},
                         {"empty": (0, 0), "scalar": (1, 4), "matrix": (6, 6)})
        index = self.index({"empty": "model.safetensors", "scalar": "model.safetensors", "matrix": "model.safetensors"})
        previous = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        for source, expected in ((Path("model.safetensors"), path), ("model.safetensors.index.json", index),
                                 (".", index)):
            with self.subTest(source=source):
                checkpoint = st.inspect_model(source)
                self.assertTrue(checkpoint.source.is_absolute())
                self.assertEqual(checkpoint.source, expected.absolute())
                self.assertTrue(all(t.shard.is_absolute() and t.shard == path.absolute()
                                    for t in checkpoint.tensors.values()))
                self.assertEqual(checkpoint.shards[0].path, path.absolute())
        self.assertTrue(st.inspect_safetensors("model.safetensors").path.is_absolute())

    def test_diagnostics_stay_bounded_for_huge_values_and_many_unknown_keys(self):
        huge = int("9" * 4000)
        for header, message in (({"a": {"dtype": "U8", "shape": [huge], "data_offsets": [0, 1]}},
                                 "dimension: integer exceeds 64 bits"),
                                ({"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, huge]}},
                                 "data offset: integer exceeds 64 bits"),
                                ({"a": {"dtype": "U8", "shape": [1], "data_offsets": [-huge, 1]}},
                                 "data offset: integer exceeds 64 bits")):
            with self.subTest(message=message), self.assertRaises(st.SafetensorsError) as caught:
                st.inspect_model(self.raw(header, b"\0"))
            self.assertIn(message, str(caught.exception))
            self.assertLess(len(str(caught.exception)), 300)
        self.tensors((("a", "U8", (1,)),), "first.safetensors")
        index = self.index({"a": "first.safetensors"}, {"total_size": huge})
        with self.assertRaisesRegex(st.SafetensorsError, "total_size: integer exceeds 64 bits$"):
            st.inspect_model(index)
        keys = {f"key{number:05d}" + "x" * 200: number for number in range(5000)}
        for extra, metadata, what in ((keys, None, "index"), (None, {"total_size": 1, **keys}, "index metadata")):
            index = self.index({"a": "first.safetensors"}, metadata, extra)
            with self.subTest(what=what):
                with self.assertRaises(st.SafetensorsError) as caught:
                    st.inspect_model(index, index_policy="strict")   # over the 1 MiB audit limits
                message = str(caught.exception)
                self.assertRegex(message, f"unsupported {what} keys \\['key00000x+\\.\\.\\., .*, \\.\\.\\.\\] "
                                          f"\\(5000 in all\\)$")
                self.assertLess(len(message), 1000)
                notes = st.inspect_model(index).notes
                self.assertEqual(len(notes), 1)
                self.assertRegex(notes[0], f"^{what} keys \\[.*\\] \\(5000 in all\\) are not interpreted$")
                self.assertLess(len(notes[0]), 1000)
        two = self.index({"a": "first.safetensors"}, {"total_size": 1, "b": 1, "a": 2}, {"z": 1, "y": 2})
        self.rejects(two, "unsupported index keys \\['y', 'z'\\]$", policies=("strict",))
        self.assertEqual(st.inspect_model(two).notes, ("index keys ['y', 'z'] are not interpreted",
                                                       "index metadata keys ['a', 'b'] are not interpreted"))


    def test_index_json_follows_the_native_reader_reals_depth_and_surrogates(self):
        self.tensors((("w", "U8", (23,)),), "a.safetensors")
        weights = {"w": "a.safetensors"}
        # Reals may sit in keys the compatible policy does not interpret; typed fields still need integers.
        meta = self.index(weights, {"total_size": 23, "ratio": 0.5})
        self.assertEqual(st.inspect_model(meta).notes, ("index metadata key 'ratio' is not interpreted",))
        self.rejects(meta, "unsupported index metadata keys \\['ratio'\\]$", policies=("strict",))
        top = self.index(weights, None, {"version": 1.0})
        self.assertEqual(st.inspect_model(top).notes, ("index key 'version' is not interpreted",))
        self.rejects(top, "unsupported index keys \\['version'\\]$", policies=("strict",))
        self.rejects(self.index(weights, {"total_size": 23.0}), "total_size must be an integer, not 23.0")
        self.rejects(self.index(weights, {"total_size": 1 << 64}), "total_size: integer exceeds 64 bits$")
        self.assertEqual(st.inspect_model(self.index(weights, None, {"big": 1 << 70})).notes,
                         ("index key 'big' is not interpreted",))
        path = self.root / "model.safetensors.index.json"
        for depth, ok in ((st.JSON_DEPTH, True), (st.JSON_DEPTH + 1, False), (70, False), (5000, False)):
            nested = "[" * (depth - 1) + "]" * (depth - 1)   # under the root object
            path.write_text('{"deep": ' + nested + ', "weight_map": {"w": "a.safetensors"}}', encoding="utf-8")
            with self.subTest(depth=depth):
                if ok:
                    self.assertEqual(st.inspect_model(path).notes, ("index key 'deep' is not interpreted",))
                else:
                    self.rejects(path, f"nesting exceeds the depth limit of {st.JSON_DEPTH}$")
        for text in ('{"\\ud800": 1, "weight_map": {"w": "a.safetensors"}}',
                     '{"x": "\\udc00", "weight_map": {"w": "a.safetensors"}}',
                     '{"x": ["\\ud800\\u0041"], "weight_map": {"w": "a.safetensors"}}'):
            path.write_text(text, encoding="utf-8")
            with self.subTest(text=text):
                self.rejects(path, "index.json: unpaired surrogate escape$")
        path.write_text('{"x": "\\ud83d\\ude00", "weight_map": {"w": "a.safetensors"}}', encoding="utf-8")
        self.assertEqual(st.inspect_model(path).notes, ("index key 'x' is not interpreted",))
        deep_header = {"__metadata__": {"a": "b"}, "w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
        data = header_bytes(deep_header).replace(b'"b"', b'"b", "c": ' + b"[" * 70 + b"]" * 70)
        self.root.joinpath("deep.safetensors").write_bytes(struct.pack("<Q", len(data)) + data + b"\0")
        self.rejects(self.root / "deep.safetensors", f"header: nesting exceeds the depth limit of {st.JSON_DEPTH}$")


class SafetensorsOracleTests(SafetensorsFixture):
    """Randomized layouts against an independent computation and, when present, the reader this module ports."""

    def random_layout(self, rng, widths):
        header, payload, records = {}, 0, []
        if rng.random() < 0.3:
            header["__metadata__"] = {"format": rng.choice(("pt", "np"))}
        for index in range(rng.randrange(0, 7)):
            dtype = rng.choice(sorted(widths))
            shape = [rng.randrange(0, 4) for _ in range(rng.randrange(0, 4))]
            bits = math.prod(shape) * widths[dtype]
            while bits % 8:
                shape.append(2)
                bits *= 2
            offsets = [payload, payload + bits // 8]
            records.append((f"t{index}", {"dtype": dtype, "shape": shape, "data_offsets": offsets}))
            payload += bits // 8
        rng.shuffle(records)
        header.update(records)
        return header, payload

    def test_random_valid_layouts_match_an_independent_computation(self):
        rng = random.Random(7)
        for case in range(300):
            header, payload = self.random_layout(rng, st.DTYPE_BITS)
            data = header_bytes(header)
            path = self.raw(data, bytes(payload), name=f"case{case % 7}.safetensors")
            shard = st.inspect_safetensors(path)
            start = 8 + len(data)
            expected = {name: (record["dtype"], tuple(record["shape"]), start + record["data_offsets"][0],
                               start + record["data_offsets"][1])
                        for name, record in header.items() if name != "__metadata__"}
            actual = {t.name: (t.dtype, t.shape, t.data_begin, t.data_end) for t in shard.tensors}
            self.assertEqual(actual, expected)
            for tensor in shard.tensors:
                self.assertEqual(tensor.nbytes * 8, math.prod(tensor.shape) * st.DTYPE_BITS[tensor.dtype])
            self.assertEqual(sum(t.nbytes for t in shard.tensors), payload)
            self.assertEqual((shard.size, shard.header_sha256), (start + payload, hashlib.sha256(data).hexdigest()))

class PackageTests(unittest.TestCase):
    def test_model_package_imports_only_the_standard_library(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import crit_core.model; "
                "loaded = {name.split('.')[0] for name in sys.modules}; "
                "assert not loaded & {'tokenizers', 'jinja2', 'numpy', 'torch', 'safetensors'}, loaded; "
                "assert 'crit_core.text' not in sys.modules")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT / "core/python")],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_core_sources_never_name_the_technique_store(self):
        for path in (ROOT / "core/python/crit_core/model").rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                self.assertNotRegex(path.read_text(encoding="utf-8").lower(), r"inference/|crit_inference", path)


if __name__ == "__main__":
    unittest.main()
