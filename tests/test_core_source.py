"""Core's model source (crit_core.model.open_model) and the description files a GGUF derives (gguf_files).

The tokenizer conformance tests tokenize the GGUF converter's vocabulary test sets (Hugging Face tokenizations of hard
inputs) with the derived tokenizer.json; they need its checkout (CRIT_LLAMA_CPP) and `tokenizers`."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_core.model import (GgufError, ModelSourceError, SafetensorsError, inspect_gguf_model,  # noqa: E402
                             open_model)
from crit_core.model.gguf_files import (PRE_TOKENIZERS, decimal, derive_config, derive_generation,  # noqa: E402
                                        derive_tokenizer)
from gguf_writer import gguf_bytes  # noqa: E402

LLAMA_CPP = os.environ.get("CRIT_LLAMA_CPP")
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None
TOKENS = ["<|endoftext|>", "<|im_end|>", "<think>", "a", "b", "ab", "[PAD6]", "[PAD7]"]
TYPES = [3, 3, 4, 1, 1, 1, 5, 5]


def metadata(architecture="qwen3", **overrides):
    values = {"general.architecture": architecture, f"{architecture}.block_count": 2,
              f"{architecture}.embedding_length": 32, f"{architecture}.feed_forward_length": 64,
              f"{architecture}.attention.head_count": 4, f"{architecture}.attention.head_count_kv": 2,
              f"{architecture}.attention.key_length": 8, f"{architecture}.context_length": 4096,
              f"{architecture}.attention.layer_norm_rms_epsilon": 1e-6, f"{architecture}.rope.freq_base": 1e6,
              "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2", "tokenizer.ggml.tokens": TOKENS,
              "tokenizer.ggml.token_type": TYPES, "tokenizer.ggml.merges": ["a b"],
              "tokenizer.ggml.eos_token_id": 1, "tokenizer.ggml.padding_token_id": 0}
    values.update(overrides)
    return [(key, value, 9, 5) if key == "tokenizer.ggml.token_type" else (key, value)
            for key, value in values.items() if value is not None]


TENSORS = [("token_embd.weight", "F16", (8, 32)), ("output.weight", "Q8_0", (8, 32)),
           ("blk.0.attn_q.weight", "Q8_0", (32, 32)), ("blk.1.attn_q.weight", "Q8_0", (32, 32))]


def safetensors_bytes(tensors):
    header, offset = {}, 0
    for name, shape in tensors:
        size = 2
        for n in shape:
            size *= n
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    data = json.dumps(header).encode()
    return struct.pack("<Q", len(data)) + data + bytes(offset)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def gguf(self, name="model-Q8_0.gguf", meta=None, tensors=TENSORS):
        path = self.root / name
        path.write_bytes(gguf_bytes(metadata() if meta is None else meta, tensors))
        return path

    def test_a_safetensors_folder_reads_its_own_files(self):
        (self.root / "model.safetensors").write_bytes(safetensors_bytes([("model.embed_tokens.weight", (8, 4))]))
        (self.root / "model-mtp.safetensors").write_bytes(safetensors_bytes([("mtp.fc.weight", (4, 4))]))
        (self.root / "config.json").write_text(json.dumps({"num_hidden_layers": 3}))
        source = open_model(self.root)
        self.assertEqual((source.format, source.folder, source.derived_names), ("safetensors", self.root, ()))
        self.assertEqual(source.config().num_hidden_layers, 3)
        self.assertEqual([shard.path.name for shard in source.weight_files()], ["model.safetensors"])
        self.assertEqual([shard.path.name for shard in source.weight_files(drafts=True)],
                         ["model.safetensors", "model-mtp.safetensors"])
        self.assertEqual(source.files.read("config.json"), (self.root / "config.json").read_bytes())

    def test_weights_are_validated_when_first_used(self):
        (self.root / "model.safetensors").write_bytes(b"short")
        (self.root / "config.json").write_text("{}")
        source = open_model(self.root)   # the description files do not need the weights
        self.assertEqual(source.files.read("config.json"), b"{}")
        with self.assertRaises(SafetensorsError):
            source.weight_files()

    def test_folders_and_files_crit_cannot_read(self):
        self.assertIsNone(open_model(self.root).format)   # a folder of description files only
        (self.root / "model.safetensors").write_bytes(safetensors_bytes([("w", (2, 2))]))
        self.gguf()
        with self.assertRaisesRegex(ModelSourceError, "holds gguf and safetensors weights"):
            open_model(self.root)
        (self.root / "weights.bin").write_bytes(b"")   # an engine's own format: no weights or files crit reads
        source = open_model(self.root / "weights.bin")
        self.assertEqual((source.format, source.folder, source.weight_files(), source.derived_names),
                         (None, None, (), ()))
        with self.assertRaisesRegex(ModelSourceError, "no such model"):
            open_model(self.root / "absent")

    def test_a_lone_gguf_derives_every_description_file(self):
        path = self.gguf()
        (self.root / "config.json").write_text("{}")   # beside the file, but a file path means the file alone
        source = open_model(path)
        self.assertIsNone(source.folder)
        self.assertEqual(source.derived_names, ("config.json", "generation_config.json", "tokenizer.json",
                                                "tokenizer_config.json"))
        config = json.loads(source.files.read("config.json"))
        self.assertEqual({k: config[k] for k in ("model_type", "num_hidden_layers", "rms_norm_eps", "vocab_size",
                                                 "tie_word_embeddings", "rope_parameters")},
                         {"model_type": "qwen3", "num_hidden_layers": 2, "rms_norm_eps": 1e-06, "vocab_size": 8,
                          "tie_word_embeddings": False,
                          "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0}})
        self.assertEqual(json.loads(source.files.read("tokenizer_config.json"))["eos_token"], "<|im_end|>")
        # EOS, plus the control tokens the reference runtime ends generation on.
        self.assertEqual(json.loads(source.files.read("generation_config.json"))["eos_token_id"], [1, 0])
        self.assertEqual([f.path.name for f in source.weight_files()], [path.name])

    def test_a_folder_supplies_whole_groups_and_the_gguf_the_rest(self):
        self.gguf()
        tokenizer = derive_tokenizer(inspect_gguf_model(self.root))
        (self.root / "tokenizer.json").write_text(json.dumps(tokenizer))
        source = open_model(self.root)
        # The folder's tokenizer group (tokenizer.json, so also tokenizer_config.json) is its own; config is derived.
        self.assertEqual(source.derived_names, ("config.json", "generation_config.json"))
        self.assertIsNone(source.files.read("tokenizer_config.json"))
        self.assertEqual(json.loads(source.files.read("tokenizer.json")), tokenizer)

    def test_a_folder_config_must_agree_with_the_gguf(self):
        self.gguf()
        (self.root / "config.json").write_text(json.dumps({"hidden_size": 64, "num_hidden_layers": 2}))
        with self.assertRaisesRegex(GgufError, "hidden size: GGUF 32, config.json 64"):
            open_model(self.root)


class DerivationTests(unittest.TestCase):
    def model(self, meta, tensors=TENSORS):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.gguf"
            path.write_bytes(gguf_bytes(meta, tensors))
            return inspect_gguf_model(path)

    def test_float32_values_take_their_shortest_decimal(self):
        stored = struct.unpack("<f", struct.pack("<f", 1e-6))[0]
        self.assertNotEqual(stored, 1e-6)
        self.assertEqual((decimal(stored), decimal(stored, "FLOAT64")), (1e-6, stored))
        self.assertEqual(decimal(struct.unpack("<f", struct.pack("<f", 0.1))[0]), 0.1)

    def test_per_layer_values_and_flags(self):
        config = derive_config(self.model(metadata(**{"qwen3.attention.head_count_kv": [2, 2]})))
        self.assertEqual((config["num_key_value_heads"], config["attention_bias"]), (2, False))
        with self.assertRaisesRegex(GgufError, "differs between layers"):
            derive_config(self.model(metadata(**{"qwen3.attention.head_count_kv": [2, 4]})))
        with self.assertRaisesRegex(GgufError, "only some layers have attn_q.bias"):
            derive_config(self.model(metadata(), TENSORS + [("blk.0.attn_q.bias", "F32", (32,))]))

    def test_other_architectures_get_generic_keys_or_are_refused(self):
        config = derive_config(self.model(metadata("granite")))
        self.assertNotIn("model_type", config)
        self.assertEqual((config["num_hidden_layers"], config["head_dim"]), (2, 8))
        with self.assertRaisesRegex(GgufError, "'gemma3' has no config rules.*sliding_window"):
            derive_config(self.model(metadata("gemma3", **{"gemma3.attention.sliding_window": 512})))

    def test_tokenizer_rules(self):
        tokenizer = derive_tokenizer(self.model(metadata()))
        self.assertEqual(tokenizer["model"]["vocab"], {t: i for i, t in enumerate(TOKENS[:6])})
        self.assertEqual([(t["content"], t["special"]) for t in tokenizer["added_tokens"]],
                         [("<|endoftext|>", True), ("<|im_end|>", True), ("<think>", False)])
        self.assertEqual(tokenizer["normalizer"], {"type": "NFC"})
        with self.assertRaisesRegex(GgufError, "'llama'; put the base model's tokenizer.json beside"):
            derive_tokenizer(self.model(metadata(**{"tokenizer.ggml.model": "llama"})))
        with self.assertRaisesRegex(GgufError, "tokenizer.ggml.pre 'tekken' has no pre-tokenizer rule"):
            derive_tokenizer(self.model(metadata(**{"tokenizer.ggml.pre": "tekken"})))

    def test_generation_defaults(self):
        meta = metadata(**{"general.sampling.temp": 0.6, "general.sampling.top_k": 20})   # temp stored as FLOAT32
        generation = derive_generation(self.model(meta))
        self.assertEqual({k: generation[k] for k in ("do_sample", "temperature", "top_k", "pad_token_id")},
                         {"do_sample": True, "temperature": 0.6, "top_k": 20, "pad_token_id": 0})


@unittest.skipUnless(LLAMA_CPP and HAS_TOKENIZERS, "needs CRIT_LLAMA_CPP (a converter checkout) and tokenizers")
class TokenizerConformanceTests(unittest.TestCase):
    """The checkout's models/ggml-vocab-<name>.gguf with .inp (inputs) and .out (Hugging Face token ids)."""

    def test_vocabulary_test_sets(self):
        from tokenizers import Tokenizer
        models = Path(LLAMA_CPP) / "models"
        for name in ("qwen2", "qwen35", "llama-bpe", "gpt-2"):
            with self.subTest(name):
                model = inspect_gguf_model(models / f"ggml-vocab-{name}.gguf")
                self.assertIn(model.metadata["tokenizer.ggml.pre"], PRE_TOKENIZERS)
                tokenizer = Tokenizer.from_str(json.dumps(derive_tokenizer(model)))
                inputs = (models / f"ggml-vocab-{name}.gguf.inp").read_text(encoding="utf-8")
                outputs = (models / f"ggml-vocab-{name}.gguf.out").read_text(encoding="utf-8").split("\n")
                cases = inputs.split("\n__ggml_vocab_test__\n")[:-1]
                self.assertGreater(len(cases), 40)
                for text, line in zip(cases, outputs):
                    self.assertEqual(tokenizer.encode(text, add_special_tokens=False).ids,
                                     [int(i) for i in line.split()], repr(text))


if __name__ == "__main__":
    unittest.main()
