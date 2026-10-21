"""Core's strict GGUF reader and the check of a GGUF against the base model's config.json and tokenizer beside it."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_core.model import (GgufError, check_gguf_folder, find_gguf, inspect_gguf, inspect_gguf_model,  # noqa: E402
                             load_model_config)
from gguf_writer import ARRAY, BOOL, FLOAT64, INT8, STRING, UINT8, gguf_bytes, tensor_bytes  # noqa: E402

TENSORS = [("token_embd.weight", "F16", (8, 32)), ("blk.0.attn_q.weight", "Q4_K", (4, 256)),
           ("blk.0.attn_norm.weight", "F32", (32,)), ("blk.0.ffn_up_exps.weight", "IQ3_S", (2, 3, 256))]
CONFIG = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 32, "intermediate_size": 64,
          "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
          "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "vocab_size": 8}
TOKENS = ["a", "b", "c", "ab", "<|end|>", "[PAD5]", "[PAD6]", "[PAD7]"]


def model_metadata(**overrides):
    values = {"general.architecture": "qwen3", "qwen3.block_count": 2, "qwen3.embedding_length": 32,
              "qwen3.feed_forward_length": 64, "qwen3.attention.head_count": 4, "qwen3.attention.head_count_kv": 2,
              "qwen3.attention.key_length": 8, "qwen3.context_length": 4096,
              "qwen3.attention.layer_norm_rms_epsilon": 1e-6, "qwen3.rope.freq_base": 1000000.0,
              "tokenizer.ggml.tokens": TOKENS, "tokenizer.ggml.merges": ["a b"]}
    values.update(overrides)
    return [(key, value) for key, value in values.items() if value is not None]


class GgufFiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, data, name="model.gguf"):
        path = self.root / name
        path.write_bytes(data)
        return path

    def rejects(self, data, message, name="model.gguf"):
        with self.assertRaisesRegex(GgufError, message):
            inspect_gguf(self.write(data, name))


class ReaderTests(GgufFiles):
    def test_metadata_tensors_and_offsets(self):
        data = bytes(range(256)) * 9
        metadata = [("general.architecture", "qwen3"), ("x.u8", 7, UINT8), ("x.i8", -3, INT8),
                    ("x.f64", 0.25, FLOAT64), ("x.flag", True, BOOL), ("x.text", "héllo"),
                    ("x.ints", [1, 2, 3]), ("x.words", ["a", "bc"]), ("x.nested", [[1, 2], [3]], ARRAY, ARRAY)]
        tensors = [("blk.0.attn_q.weight", "Q4_K", (4, 256), data[:tensor_bytes("Q4_K", (4, 256))]),
                   ("n", "F32", (7,))]
        path = self.write(gguf_bytes(metadata, tensors))
        file = inspect_gguf(path)
        self.assertEqual(file.version, 3)
        self.assertEqual(file.architecture, "qwen3")
        self.assertEqual(dict(file.metadata), {"general.architecture": "qwen3", "x.u8": 7, "x.i8": -3, "x.f64": 0.25,
                                               "x.flag": True, "x.text": "héllo", "x.ints": (1, 2, 3),
                                               "x.words": ("a", "bc"), "x.nested": ((1, 2), (3,))})
        self.assertEqual(file.value_types["x.nested"], "ARRAY[ARRAY[UINT32]]")
        self.assertEqual(file.value_types["x.u8"], "UINT8")
        q, n = file.tensors
        self.assertEqual((q.dtype, q.shape, q.ne, q.nbytes, q.elements), ("Q4_K", (4, 256), (256, 4), 576, 1024))
        self.assertEqual(path.read_bytes()[q.data_begin:q.data_end], data[:576])
        self.assertEqual((n.dtype, n.shape, n.nbytes, n.data_begin - q.data_end), ("F32", (7,), 28, 0))
        self.assertEqual(q.data_begin % 32, 0)
        self.assertEqual(file.header_sha256, __import__("hashlib").sha256(path.read_bytes()[:file.data_offset]).hexdigest())

    def test_custom_alignment_and_version_2(self):
        tensors = [("a", "F32", (3,)), ("b", "Q8_0", (1, 32))]
        file = inspect_gguf(self.write(gguf_bytes([("general.alignment", 64)], tensors, version=2, alignment=64)))
        self.assertEqual((file.version, file.alignment), (2, 64))
        self.assertEqual([t.data_begin % 64 for t in file.tensors], [0, 0])

    def test_rejections(self):
        good = [("a", "F32", (4,)), ("b", "F32", (4,))]
        self.rejects(gguf_bytes(magic=b"GGML"), "not a GGUF file")
        self.rejects(gguf_bytes(version=1), "version 1 is not supported")
        self.rejects(b"GGUF" + struct.pack(">IQQ", 3, 0, 0), "big-endian GGUF is not supported")
        self.rejects(gguf_bytes([("k", 1), ("k", 2)]), "'k' appears twice")
        self.rejects(gguf_bytes([("k", 1, 13)]), "unknown value type 13")
        self.rejects(gguf_bytes([("k", b"\xff\xfe", STRING)]), "not valid UTF-8")
        self.rejects(gguf_bytes([("k", [[[1]]], ARRAY, ARRAY)]), "nests arrays deeper than 2")
        self.rejects(gguf_bytes([("general.alignment", 48)]), "power of two")
        self.rejects(gguf_bytes([("general.alignment", -32, 5)]), "UINT32 power of two")
        self.rejects(gguf_bytes((), [("a", "F32", (4,)), ("a", "F32", (4,))]), "'a' appears twice")
        self.rejects(gguf_bytes((), [("a", "Q4_K", (2, 128))]), "not a multiple of Q4_K's 256-element block")
        self.rejects(gguf_bytes((), good, offsets=[0, 16]), "not a multiple of the 32-byte alignment")
        self.rejects(gguf_bytes((), good, offsets=[0, 64]), "hole at data offset 32")
        self.rejects(gguf_bytes((), [("a", "F32", (16,)), ("b", "F32", (4,))], offsets=[0, 32]), "overlap")
        self.rejects(gguf_bytes((), good, trailing=bytes(32)), "32 trailing bytes")
        self.rejects(gguf_bytes((), good)[:-40], "past the")
        self.rejects(gguf_bytes([("k", "x" * 64)])[:40], "past the end of the file")
        self.rejects(gguf_bytes(), "expected a .gguf file", name="model.bin")
        data = bytearray(gguf_bytes((), [("a", "F32", (4,))]))
        data[data.index(b"a") + 1:data.index(b"a") + 5] = struct.pack("<I", 5)
        self.rejects(bytes(data), "rank 5")
        data = bytearray(gguf_bytes((), [("a", "F32", (4,))]))
        data[data.index(b"a") + 13:data.index(b"a") + 17] = struct.pack("<I", 31)
        self.rejects(bytes(data), "unknown ggml type 31")

    def test_split_sets_join_in_order(self):
        for no, tensors in enumerate([[("a", "F32", (4,))], [("b", "F32", (8,))], [("c", "Q8_0", (2, 32))]]):
            metadata = [("split.no", no, 2), ("split.count", 3, 2), ("split.tensors.count", 3, 4)]
            if no == 0:
                metadata.insert(0, ("general.architecture", "llama"))
            self.write(gguf_bytes(metadata, tensors), f"m-{no + 1:05d}-of-00003.gguf")
        self.write(gguf_bytes([("general.architecture", "clip")]), "mmproj-m-f16.gguf")
        self.assertEqual(find_gguf(self.root).name, "m-00001-of-00003.gguf")
        model = inspect_gguf_model(self.root)
        self.assertEqual(list(model.tensors), ["a", "b", "c"])
        self.assertEqual([file.path.name for file in model.files],
                         ["m-00001-of-00003.gguf", "m-00002-of-00003.gguf", "m-00003-of-00003.gguf"])
        self.assertEqual((model.architecture, model.tensor_bytes), ("llama", 16 + 32 + 68))
        (self.root / "m-00002-of-00003.gguf").unlink()
        with self.assertRaisesRegex(GgufError, "split part m-00002-of-00003.gguf is missing"):
            inspect_gguf_model(self.root)

    def test_split_parts_must_say_their_place(self):
        self.write(gguf_bytes([("split.no", 0, 2), ("split.count", 2, 2)]), "m-00001-of-00002.gguf")
        self.write(gguf_bytes([("split.no", 0, 2), ("split.count", 2, 2)]), "m-00002-of-00002.gguf")
        with self.assertRaisesRegex(GgufError, "do not say part 2 of 2"):
            inspect_gguf_model(self.root / "m-00001-of-00002.gguf")
        with self.assertRaisesRegex(GgufError, "not the first part"):
            inspect_gguf_model(self.root / "m-00002-of-00002.gguf")

    def test_one_model_per_folder(self):
        self.assertIsNone(find_gguf(self.root))
        self.write(gguf_bytes(), "a.gguf")
        self.write(gguf_bytes(), "b.gguf")
        with self.assertRaisesRegex(GgufError, "2 .gguf files of 2 models"):
            find_gguf(self.root)


class FolderCheckTests(GgufFiles):
    def folder(self, metadata=None, config=None, vocab=None, merges=("a b",), template=None):
        self.write(gguf_bytes(metadata or model_metadata(), TENSORS))
        (self.root / "config.json").write_text(json.dumps(config or CONFIG))
        vocab = vocab or {token: i for i, token in enumerate(TOKENS[:4])}
        tokenizer = {"model": {"type": "BPE", "vocab": vocab, "merges": list(merges)},
                     "added_tokens": [{"id": 4, "content": "<|end|>", "special": True}]}
        (self.root / "tokenizer.json").write_text(json.dumps(tokenizer))
        if template is not None:
            (self.root / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
        return check_gguf_folder(inspect_gguf_model(self.root), load_model_config(self.root), self.root)

    def test_matching_folder(self):
        check = self.folder()
        self.assertEqual((check.errors, check.notes), ((), ()))

    def test_shape_and_arithmetic_disagreements_are_errors(self):
        check = self.folder(model_metadata(**{"qwen3.block_count": 3, "qwen3.attention.head_count_kv": 4,
                                              "qwen3.rope.freq_base": 10000.0}))
        self.assertEqual(check.errors, ("layers: GGUF 3, config.json 2", "key/value heads: GGUF 4, config.json 2",
                                        "RoPE base: GGUF 10000.0, config.json 1000000.0"))
        with self.assertRaisesRegex(GgufError, "the GGUF and the folder's files disagree: layers"):
            check.raise_errors(self.root)

    def test_draft_blocks_and_per_layer_arrays(self):
        # block_count counts the MTP blocks; zero KV heads mark recurrent layers.
        check = self.folder(model_metadata(**{"qwen3.block_count": 3, "qwen3.nextn_predict_layers": 1,
                                              "qwen3.attention.head_count_kv": [0, 2]}))
        self.assertEqual(check.errors, ())

    def test_tokenizer_disagreements(self):
        self.assertEqual(self.folder(vocab={"a": 0, "b": 1, "x": 2, "ab": 3}).errors,
                         ("token 2: GGUF 'c', tokenizer.json 'x'",))
        self.assertEqual(self.folder(merges=("a c",)).errors,
                         ("BPE merges differ from merge 0 (GGUF 1, tokenizer.json 1)",))
        self.assertEqual(self.folder(vocab={"a": 0, "b": 1, "c": 2, "ab": 3, "z": 9}).errors,
                         ("tokenizer.json has 1 tokens past the GGUF's 8, from id 9",))

    def test_notes(self):
        check = self.folder(model_metadata(**{"qwen3.context_length": 8192, "tokenizer.chat_template": "{{ x }}"}),
                            template="{{ y }}")
        self.assertEqual(check.errors, ())
        self.assertEqual(check.notes, ("the GGUF declares 8192 positions, config.json 4096; Eval uses config.json's",
                                       "the folder's chat template differs from the GGUF's tokenizer.chat_template; "
                                       "the folder's is used"))


if __name__ == "__main__":
    unittest.main()
