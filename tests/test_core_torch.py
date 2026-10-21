"""Core's reader for PyTorch state dicts (crit_core.model.torch_pickle) and the "torch" model source.

The files are written by tests/torch_writer.py, so these tests need no torch; with torch installed, files that
torch.save itself writes are read the same way."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_core.model import TorchFileError, inspect_torch, inspect_torch_model, open_model  # noqa: E402
from torch_writer import Call, Global, pickle_bytes, tensor_call, torch_bytes  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None


class TorchReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, tensors=(), **options):
        path = self.root / name
        path.write_bytes(torch_bytes(list(tensors), **options))
        return path

    def test_tensors_point_at_their_bytes(self):
        embed = bytes(range(16))
        path = self.write("pytorch_model.bin", [
            ("model.embed_tokens.weight", "BF16", (2, 4), embed),
            ("lm_head.weight", "BF16", (2, 4), None, "0", 0),     # tied: the embedding's storage
            ("half", "BF16", (1, 4), None, "0", 4),                # a row of it: an offset into the storage
            ("scale", "F8_E4M3", (3,), b"xyz"),
            ("step", "I64", (), struct.pack("<q", 7))])
        data = path.read_bytes()
        found = {t.name: (t.dtype, t.shape, data[t.data_begin:t.data_end]) for t in inspect_torch(path).tensors}
        self.assertEqual(found, {"model.embed_tokens.weight": ("BF16", (2, 4), embed),
                                 "lm_head.weight": ("BF16", (2, 4), embed), "half": ("BF16", (1, 4), embed[8:]),
                                 "scale": ("F8_E4M3", (3,), b"xyz"), "step": ("I64", (), struct.pack("<q", 7))})

    def test_pickles_that_are_not_state_dicts_are_refused(self):
        tensor = tensor_call("F32", (2,), "0")
        cases = {
            "runs code": (Call(Global("os", "system"), "echo crit"), "references os.system.*does not run it"),
            "holds a number": ({"w": tensor, "n": 3}, "'n' is not a tensor"),
            "is a list": (Call(Global("builtins", "list")), "references builtins.list"),
            "views a transpose": ({"w": tensor_call("F32", (2, 3), "0", stride=(1, 2))}, "non-contiguous view"),
            "overruns its storage": ({"w": tensor_call("F32", (2,), "0", offset=1)}, "ends past its storage"),
        }
        for label, (state, error) in cases.items():
            with self.subTest(label):
                path = self.write("bad.bin", [("w", "F32", (2,), None)], state=state)
                with self.assertRaisesRegex(TorchFileError, error):
                    inspect_torch(path)

    def test_files_torch_save_did_not_write(self):
        legacy = self.root / "legacy.bin"
        legacy.write_bytes(b"\x80\x02\x8a\x0a\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19" + pickle_bytes(1))
        with self.assertRaisesRegex(TorchFileError, "legacy format"):
            inspect_torch(legacy)
        engine = self.root / "model.bin"   # another program's weights (CTranslate2 names its file model.bin)
        engine.write_bytes(b"\x02\x00\x00\x00" + bytes(60))
        with self.assertRaisesRegex(TorchFileError, "not a file torch.save wrote"):
            inspect_torch(engine)
        self.assertIsNone(open_model(engine).format)
        self.assertEqual(open_model(legacy).format, "torch")   # its own, so the refusal names the legacy format

    def test_a_sharded_checkpoint_follows_its_index(self):
        self.write("pytorch_model-00001-of-00002.bin", [("a", "F32", (2,), None)])
        self.write("pytorch_model-00002-of-00002.bin", [("b", "F16", (4,), None)])
        weight_map = {"a": "pytorch_model-00001-of-00002.bin", "b": "pytorch_model-00002-of-00002.bin"}
        (self.root / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": weight_map}))
        self.write("training_args.bin", [("x", "F32", (1,), None)])   # a trainer's state, not the model
        checkpoint = inspect_torch_model(self.root)
        self.assertEqual({name: t.shard.name for name, t in checkpoint.tensors.items()}, weight_map)
        self.assertEqual(checkpoint.tensor_bytes, 16)
        source = open_model(self.root)
        self.assertEqual(source.format, "torch")
        self.assertEqual([f.path.name for f in source.weight_files()], sorted(weight_map.values()))
        for wrong, error in (({"a": weight_map["b"]}, "1 weight_map tensors are in no file, first 'a'"),
                             ({"b": weight_map["a"], "c": weight_map["b"]}, "does not put 'b' in")):
            (self.root / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": {**weight_map, **wrong}}))
            with self.assertRaisesRegex(TorchFileError, error):
                inspect_torch_model(self.root)

    def test_safetensors_beside_a_state_dict_are_the_weights(self):
        # Hugging Face repositories often ship both; transformers loads the safetensors, and so does crit.
        self.write("pytorch_model.bin", [("w", "F32", (2,), None)])
        header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
        (self.root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))
        self.assertEqual(open_model(self.root).format, "safetensors")
        self.assertEqual(open_model(self.root / "pytorch_model.bin").format, "torch")   # unless named

    @unittest.skipUnless(HAS_TORCH, "needs torch")
    def test_torch_save_files_read_alike(self):
        import torch
        torch.manual_seed(0)
        embed = torch.randn(6, 4, dtype=torch.bfloat16)
        state = {"embed.weight": embed, "lm_head.weight": embed,
                 "fp8.weight": torch.randn(3, 4).to(torch.float8_e4m3fn), "rows": torch.randn(10, 4)[2:5], "param": torch.nn.Parameter(torch.randn(2, 2)),
                 "count": torch.tensor(3)}
        path = self.root / "model.pt"
        torch.save(state, path)
        data = path.read_bytes()
        for tensor in inspect_torch(path).tensors:
            expected = state[tensor.name].detach().contiguous().reshape(-1).view(torch.uint8)
            self.assertEqual(data[tensor.data_begin:tensor.data_end], expected.numpy().tobytes(), tensor.name)
        torch.save({"t": torch.randn(3, 4).t()}, path)
        with self.assertRaisesRegex(TorchFileError, "non-contiguous"):
            inspect_torch(path)
        torch.save({"x": torch.randn(2)}, path, _use_new_zipfile_serialization=False)
        with self.assertRaisesRegex(TorchFileError, "legacy format"):
            inspect_torch(path)


if __name__ == "__main__":
    unittest.main()
