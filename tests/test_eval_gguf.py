"""Eval over a GGUF, alone or in a folder: ceiling, max context and model identity from the GGUF's tensors and
derived files, and the reference's GGUF layout (crit_eval.gguf_layout).

The conformance tests convert tiny Hugging Face models with GGUF's reference converter and need a checkout of it
(CRIT_LLAMA_CPP=/path/to/the/checkout) plus torch, transformers, safetensors, tokenizers and gguf; without them they
are skipped."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "eval", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_core.model import GgufError  # noqa: E402
from crit_eval import ceiling, context, prepare  # noqa: E402
from gguf_writer import gguf_bytes  # noqa: E402

HAS_TEXT = all(importlib.util.find_spec(name) for name in ("tokenizers", "jinja2"))
HAS_LAYOUT = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "gguf", "numpy"))
LLAMA_CPP = os.environ.get("CRIT_LLAMA_CPP")
HAS_CONVERTER = HAS_LAYOUT and bool(LLAMA_CPP) and all(
    importlib.util.find_spec(name) for name in ("safetensors", "tokenizers"))

CONFIG = {"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text", "hidden_size": 256,
          "intermediate_size": 512, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
          "head_dim": 64, "max_position_embeddings": 8192, "vocab_size": 16, "rms_norm_eps": 1e-6,
          "layer_types": ["linear_attention", "full_attention"], "linear_num_key_heads": 2,
          "linear_num_value_heads": 4, "linear_key_head_dim": 64, "linear_value_head_dim": 64,
          "linear_conv_kernel_dim": 4, "mtp_num_hidden_layers": 1}
VOCAB = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] + [chr(c) for c in range(ord("a"), ord("a") + 12)] + ["ab"]
TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
METADATA = [("general.architecture", "qwen35"), ("qwen35.block_count", 3), ("qwen35.nextn_predict_layers", 1),
            ("qwen35.embedding_length", 256), ("qwen35.attention.head_count", 4),
            ("qwen35.attention.head_count_kv", 2), ("qwen35.context_length", 8192),
            ("qwen35.feed_forward_length", 512), ("qwen35.attention.key_length", 64),
            ("qwen35.attention.value_length", 64), ("qwen35.attention.layer_norm_rms_epsilon", 1e-6),
            ("qwen35.rope.freq_base", 1e7), ("qwen35.rope.dimension_count", 16),
            ("qwen35.rope.dimension_sections", [3, 3, 2, 0]), ("qwen35.full_attention_interval", 2),
            ("qwen35.ssm.conv_kernel", 4), ("qwen35.ssm.state_size", 64), ("qwen35.ssm.group_count", 2),
            ("qwen35.ssm.time_step_rank", 4), ("qwen35.ssm.inner_size", 256),
            ("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "qwen35"), ("tokenizer.ggml.tokens", VOCAB),
            ("tokenizer.ggml.token_type", [3, 3, 3] + [1] * 13, 9, 5), ("tokenizer.ggml.merges", ["a b"]),
            ("tokenizer.ggml.eos_token_id", 2), ("tokenizer.ggml.padding_token_id", 0),
            ("tokenizer.chat_template", TEMPLATE)]
TENSORS = [("token_embd.weight", "Q6_K", (16, 256)), ("output.weight", "Q6_K", (16, 256)),
           ("output_norm.weight", "F32", (256,)),
           ("blk.0.attn_qkv.weight", "IQ3_S", (512, 256)), ("blk.0.ssm_conv1d.weight", "F32", (512, 4)),
           ("blk.0.ssm_out.weight", "IQ3_S", (256, 256)), ("blk.0.ffn_up.weight", "Q4_K", (512, 256)),
           ("blk.1.attn_q.weight", "F16", (512, 256)), ("blk.1.ffn_down_exps.weight", "Q4_K", (2, 256, 512)),
           ("blk.2.attn_q.weight", "BF16", (512, 256)), ("blk.2.nextn.eh_proj.weight", "F16", (256, 512))]


def gguf_folder(directory, config=CONFIG, tensors=TENSORS, metadata=METADATA):
    directory = Path(directory)
    (directory / "model-IQ3_S.gguf").write_bytes(gguf_bytes(metadata, tensors))
    if config is not None:
        (directory / "config.json").write_text(json.dumps(config))
    return directory


class GgufEvalTests(unittest.TestCase):
    def test_ceiling_counts_stored_bytes_by_block_format(self):
        with tempfile.TemporaryDirectory() as directory:
            model = gguf_folder(directory)
            result = ceiling.model_ceiling(model)
        # Block types at the 8-bit rate, F16 at bf16; the head, the convolution kernel and the draft block are out.
        self.assertEqual(result["matmul_flops_per_token"], {"fp8": 2 * (512 * 256 + 256 * 256 + 512 * 256 + 2 * 256 * 512),
                                                            "bf16": 2 * 512 * 256})
        self.assertEqual(result["matrices"], {"fp8": 5, "bf16": 1})
        stored = {"IQ3_S": 110 / 256, "Q4_K": 144 / 256, "Q6_K": 210 / 256, "F16": 2, "F32": 4}
        read = (512 * 256 + 256 * 256) * stored["IQ3_S"] + 512 * 256 * stored["Q4_K"] + 16 * 256 * stored["Q6_K"] \
            + 256 * 4 + 512 * 4 * 4 + 512 * 256 * 2 + 2 * 256 * 512 * stored["Q4_K"]
        self.assertEqual(result["weight_bytes"], read)

    def test_weight_split_keeps_the_draft_block_resident(self):
        with tempfile.TemporaryDirectory() as directory:
            model = gguf_folder(directory)
            split = context.weight_split(model, ceiling.model_config(model))
        self.assertEqual(split["draft"], 512 * 256 * 2 + 256 * 512 * 2)
        self.assertEqual(split["experts"], 2 * 256 * 512 * 144 // 256)
        self.assertEqual(split["embedding"], 16 * 256 * 210 // 256)

    def test_a_lone_gguf_is_a_whole_model(self):
        from crit_core.model import open_model
        with tempfile.TemporaryDirectory() as directory:
            model = gguf_folder(directory, config=None) / "model-IQ3_S.gguf"
            source = open_model(model)
            self.assertEqual(source.derived_names, ("chat_template.jinja", "config.json", "generation_config.json",
                                                    "tokenizer.json", "tokenizer_config.json"))
            config = ceiling.model_config(model).to_dict()
            # Everything the folder's config.json said, from the GGUF's metadata alone.
            for key, value in CONFIG.items():
                self.assertEqual(config.get(key), value, key)
            self.assertEqual(config.get("rope_parameters"), {"rope_type": "default", "rope_theta": 1e7,
                                                             "partial_rotary_factor": 0.25, "mrope_section": [3, 3, 2],
                                                             "mrope_interleaved": True})
            value = context.max_context(model, {"memory_free_bytes": 1 << 34}, 0,
                                        {"kv_bytes_per_element": 1, "reserve_gib": 1})
            self.assertEqual((value["tokens"], value["model_limit"]), (8192, 8192))
            self.assertEqual(ceiling.model_ceiling(model), ceiling.model_ceiling(gguf_folder(directory)))
            if HAS_TEXT:   # Core's text layer needs tokenizers and jinja2
                from crit_core.text import load_model_text
                text = load_model_text(source.files, clock=None)
                self.assertEqual(text.tokenizer.encode("abc", add_special_tokens=False), [15, 5])   # merge "a b"
                ids = text.encode_chat([{"role": "user", "content": "abc"}], add_generation_prompt=True)
                # <|im_start|>user ... <|im_end|> <|im_start|>assistant: the template's markers are single tokens.
                self.assertEqual((ids[0], ids.count(1), ids.count(2)), (1, 2, 1))
                self.assertEqual(text.generation.stop_token_ids, (2, 0))
            identity = prepare.model_identity(model)
            self.assertEqual(sorted(identity["files"]), ["chat_template.jinja", "config.json", "generation_config.json",
                                                         "tokenizer.json", "tokenizer_config.json", "weight_files"])
            self.assertEqual(identity["files"]["weight_files"], {model.name: model.stat().st_size})

    def test_a_folder_that_disagrees_with_the_gguf_is_refused(self):
        from crit_core.model import ModelSourceError
        with tempfile.TemporaryDirectory() as directory:
            model = gguf_folder(directory, config={**CONFIG, "hidden_size": 512})
            with self.assertRaisesRegex(GgufError, "hidden size: GGUF 256, config.json 512"):
                ceiling.model_ceiling(model)
            (Path(directory) / "model.safetensors").write_bytes(b"")
            with self.assertRaisesRegex(ModelSourceError, "holds gguf and safetensors weights"):
                ceiling.model_ceiling(model)

    def test_model_identity_covers_the_gguf_but_not_a_projector(self):
        with tempfile.TemporaryDirectory() as directory:
            model = gguf_folder(directory)
            before = prepare.model_identity(model)
            (model / "mmproj-model-f16.gguf").write_bytes(gguf_bytes())
            self.assertEqual(prepare.model_identity(model), before)
            size = (model / "model-IQ3_S.gguf").stat().st_size
            self.assertEqual(before["files"]["weight_files"], {"model-IQ3_S.gguf": size})


@unittest.skipUnless(HAS_LAYOUT, "needs torch, transformers, numpy and gguf")
class GgufLayoutTests(unittest.TestCase):
    """The inverse steps against the converter's forward steps, copied from its conversion package."""

    def test_value_heads_return_to_grouped_order(self):
        import torch
        from crit_eval.gguf_layout import _swap_heads

        def reorder_v_heads(tensor, dim, num_k_heads, num_v_per_k, head_dim):   # _LinearAttentionVReorderBase
            shape = list(tensor.shape)
            if dim < 0:
                dim += len(shape)
            new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
            tensor = tensor.reshape(*new_shape)
            perm = list(range(len(new_shape)))
            perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
            return tensor.permute(*perm).contiguous().reshape(*shape)

        weight = torch.randn(2 * 3 * 4, 5)
        for dim, tensor in ((0, weight), (1, weight.T.contiguous())):
            tiled = reorder_v_heads(tensor, dim, 2, 3, 4)
            self.assertFalse(torch.equal(tiled, tensor))
            self.assertTrue(torch.equal(_swap_heads(tiled, dim, 3, 2, 4), tensor))

    def test_llama_rotary_rows_are_unpermuted(self):
        import torch
        from crit_eval.gguf_layout import LlamaLayout

        def permute(weights, n_head, n_head_kv):   # LlamaModel.permute
            if n_head_kv is not None and n_head != n_head_kv:
                n_head = n_head_kv
            return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                    .swapaxes(1, 2).reshape(weights.shape))

        for heads, kv in ((4, 4), (4, 2)):
            weight = torch.randn(kv * 8, 6)
            self.assertTrue(torch.equal(LlamaLayout._unpermute(permute(weight, heads, kv), heads, kv), weight))

    def test_an_architecture_without_rules_is_refused(self):
        from crit_eval.gguf_layout import GgufCheckpoint, GgufLayoutError
        with tempfile.TemporaryDirectory() as directory:
            model = gguf_folder(directory, config=None, metadata=[("general.architecture", "gemma3")],
                                tensors=[("token_embd.weight", "F32", (4, 8))])
            from crit_core.model import inspect_gguf_model
            with self.assertRaisesRegex(GgufLayoutError, "'gemma3' has no layout rules"):
                GgufCheckpoint(inspect_gguf_model(model), object())


SMALL = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
DELTA_NET = dict(layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2, linear_num_value_heads=4,
                 linear_key_head_dim=16, linear_value_head_dim=16)
EXPERTS = dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32)
CONVERSIONS = {
    "llama": ("llama", dict(SMALL, head_dim=16)),
    "llama-tied": ("llama", dict(SMALL, head_dim=16, tie_word_embeddings=True)),
    "llama3-rope": ("llama", dict(SMALL, head_dim=16, max_position_embeddings=131072, rope_parameters={
        "rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192})),
    "qwen2": ("qwen2", dict(SMALL)),
    "qwen3": ("qwen3", dict(SMALL, head_dim=16)),
    # The reference runtime's qwen3moe always renormalises the top-k weights, as the released Qwen3 MoE models do.
    "qwen3moe": ("qwen3_moe", dict(SMALL, head_dim=16, **EXPERTS, norm_topk_prob=True)),
    "qwen35": ("qwen3_5_text", dict(SMALL, head_dim=16, **DELTA_NET)),
    "qwen35moe": ("qwen3_5_moe_text", dict(SMALL, head_dim=16, **DELTA_NET, **EXPERTS, shared_expert_intermediate_size=48)),
}
MROPE = ("mrope_section", "mrope_interleaved")
# The converter's elementwise transforms (1 + w, -exp(A_log)) round once each way.
ROUNDED = ("norm.weight", ".A_log")
CONVERT = """import runpy, sys
llama = sys.argv[1]
sys.path[:0] = [llama, llama + "/gguf-py"]
import conversion.base as base
base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen2"   # tiny test vocabularies have no known hash
sys.argv = ["convert_hf_to_gguf.py"] + sys.argv[2:]
runpy.run_path(llama + "/convert_hf_to_gguf.py", run_name="__main__")
"""


@unittest.skipUnless(HAS_CONVERTER, "needs CRIT_LLAMA_CPP (a converter checkout), torch, transformers, safetensors, "
                                    "tokenizers and gguf")
class GgufConformanceTests(unittest.TestCase):
    """A tiny model of each architecture with LAYOUTS rules, converted by the reference converter to an F32 GGUF, comes
    back tensor for tensor, the lone .gguf derives the original's config.json and tokenizer, and the reference scores
    it exactly as it scores the original."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "convert.py").write_text(CONVERT)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def tokenizer(self, out):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=298, special_tokens=["<|endoftext|>", "<|im_start|>"],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tokenizer.train_from_iterator(["the quick brown fox jumps over the lazy dog " * 4, "gguf models"], trainer)
        tokenizer.save(str(out / "tokenizer.json"))
        (out / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast",
                                                               "eos_token": "<|endoftext|>"}))

    def hf_model(self, name, model_type, options):
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import AutoConfig, AutoModelForCausalLM
        out = self.root / f"hf-{name}"
        out.mkdir()
        self.tokenizer(out)
        torch.manual_seed(0)
        # eos_token_id: the tokenizer's <|endoftext|> (LlamaConfig's default 2 is another token here).
        model = AutoModelForCausalLM.from_config(AutoConfig.for_model(model_type, vocab_size=320, eos_token_id=0,
                                                                      **options),
                                                 dtype=torch.float32)
        with torch.no_grad():
            for key, parameter in model.named_parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.2 + (0.5 if "norm" in key else 0)
                                - (1.0 if "A_log" in key else 0))
        model.save_pretrained(out)
        if model_type.startswith("qwen3_5"):
            # Qwen3.5 checkpoints ship an MTP block, which the converter expects: a copy of the attention layer.
            tensors = load_file(out / "model.safetensors")
            for key, value in list(tensors.items()):
                if key.startswith("model.layers.1."):
                    tensors["mtp.layers.0." + key[len("model.layers.1."):]] = value.clone()
            tensors["mtp.fc.weight"] = torch.randn(64, 128)
            for part in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
                tensors[f"mtp.{part}.weight"] = torch.randn(64)
            save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
            config = json.loads((out / "config.json").read_text())
            (out / "config.json").write_text(json.dumps({**config, "mtp_num_hidden_layers": 1}))
        return out

    def convert(self, name, hf):
        """The lone .gguf the reference converter writes, in a folder of its own (no file of the original beside it)."""
        folder = self.root / f"gguf-{name}"
        folder.mkdir()
        out = folder / f"{name}-F32.gguf"
        subprocess.run([sys.executable, str(self.root / "convert.py"), LLAMA_CPP, str(hf), "--outfile",
                        str(out), "--outtype", "f32"], check=True, capture_output=True)
        return out

    def assert_described_alike(self, gguf, hf):
        """The config.json and tokenizer the GGUF derives say what the original files say."""
        from tokenizers import Tokenizer
        from crit_core.model import open_model
        files = open_model(gguf).files
        derived, original = json.loads(files.read("config.json")), json.loads((hf / "config.json").read_text())
        for key, value in derived.items():
            if key in original and original[key] is not None:
                if key == "rope_parameters":   # the converter always writes Qwen3.5's (default) MRoPE sections
                    value = {k: v for k, v in value.items() if k in original[key] or k not in MROPE}
                self.assertEqual(value, original[key], key)
        tokenizers = [Tokenizer.from_str(files.read("tokenizer.json").decode()),
                      Tokenizer.from_file(str(hf / "tokenizer.json"))]
        for text in ("the quick brown fox", "<|im_start|>gguf models<|endoftext|>", "  lazy\n\ndog's 123"):
            ids = [t.encode(text, add_special_tokens=False).ids for t in tokenizers]
            self.assertEqual(ids[0], ids[1], text)

    @staticmethod
    def hf_tensor(tensors, key):
        import torch
        if key in tensors:
            return tensors[key]
        base = key.rsplit(".", 1)[0]
        count = sum(1 for k in tensors if k.startswith(base + ".") and k.endswith(".down_proj.weight"))
        if key.endswith("experts.gate_up_proj"):   # transformers saves fused experts one by one
            return torch.stack([torch.cat([tensors[f"{base}.{e}.gate_proj.weight"],
                                           tensors[f"{base}.{e}.up_proj.weight"]]) for e in range(count)])
        return torch.stack([tensors[f"{base}.{e}.down_proj.weight"] for e in range(count)])

    def test_round_trips(self):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoModelForCausalLM
        from crit_core.model import open_model
        from crit_eval.gguf_layout import GgufCheckpoint
        from crit_eval.reference import StreamingReference, _text_config
        torch.manual_seed(1)
        sequences = [(torch.randint(0, 300, (40,)).tolist(), 5), (torch.randint(0, 300, (23,)).tolist(), 3)]
        for name, (model_type, options) in CONVERSIONS.items():
            with self.subTest(name):
                hf = self.hf_model(name, model_type, options)
                folder = self.convert(name, hf)
                self.assert_described_alike(folder, hf)
                stored = load_file(hf / "model.safetensors")
                config = _text_config(folder)
                checkpoint = GgufCheckpoint(open_model(folder).weights, config)
                with torch.device("meta"):
                    skeleton = AutoModel.from_config(config)
                shapes = {"model." + k: v.shape for k, v in skeleton.state_dict().items()}
                if "lm_head.weight" in stored:
                    shapes["lm_head.weight"] = stored["lm_head.weight"].shape
                for key, shape in shapes.items():
                    expected = self.hf_tensor(stored, key).float()
                    got = checkpoint.fit(key, checkpoint.raw(key), shape)
                    if key.endswith(ROUNDED):
                        torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-7, msg=key)
                    else:
                        self.assertTrue(torch.equal(got, expected), key)
                gguf_scores = StreamingReference(folder, device="cpu").score(sequences, 5, None, log=lambda *_: None)
                model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32, attn_implementation="eager")
                with torch.no_grad():
                    for (ids, first), rows in zip(sequences, gguf_scores):
                        logprobs = torch.log_softmax(model(torch.tensor([ids])).logits[0].double(), -1)
                        self.assertEqual([row["logprob"] for row in rows],
                                         [logprobs[i - 1, ids[i]].item() for i in range(first, len(ids))])


if __name__ == "__main__":
    unittest.main()
