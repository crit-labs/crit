"""Core's ONNX reader (crit_core.model.onnx) and the "onnx" model source.

The files are written by tests/onnx_writer.py, so these tests need no onnx package; with it installed, a graph
built by onnx.helper reads the same way."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_core.model import OnnxError, inspect_onnx, open_model  # noqa: E402
from crit_core.model.onnx import description_folder  # noqa: E402
from onnx_writer import floats, onnx_bytes  # noqa: E402

FLOAT, UINT8 = 1, 2
HAS_ONNX = importlib.util.find_spec("onnx") is not None


class OnnxReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, initializers, nodes, name="model.onnx"):
        path = self.root / name
        path.write_bytes(onnx_bytes(initializers, nodes))
        return path

    def test_weights_take_their_modules_names(self):
        path = self.write(
            [("model.norm.weight", FLOAT, [2], floats(1, 2)),
             ("onnx::MatMul_7", FLOAT, [2, 3], floats(*range(6))),       # a Linear's weight, transposed
             ("onnx::Gemm_8", FLOAT, [3, 2], floats(*range(6))),         # Gemm with transB: stored [out, in]
             ("onnx::Expand_9", FLOAT, [1], floats(0)),                  # a shape constant, not a weight
             ("model.wte.weight", FLOAT, [2, 3], floats(*range(6)))],    # named, used as stored (Conv1D)
            [("MatMul", "/model/layers.0/mlp/up_proj/MatMul", ["x", "onnx::MatMul_7"]),
             ("Gemm", "/model/layers.0/mlp/down_proj/Gemm", ["y", "onnx::Gemm_8"], {"transB": 1}),
             ("MatMul", "/model/h.0/attn/c_attn/MatMul", ["z", "model.wte.weight"]),
             ("Expand", "/model/Expand", ["w", "onnx::Expand_9"])])
        model = inspect_onnx(path)
        data = path.read_bytes()
        found = {t.name: (t.source_name, t.shape, t.transposed, data[t.data_begin:t.data_end]) for t in model.tensors}
        self.assertEqual(found, {
            "model.norm.weight": ("model.norm.weight", (2,), False, floats(1, 2)),
            "model.layers.0.mlp.up_proj.weight": ("onnx::MatMul_7", (2, 3), True, floats(*range(6))),
            "model.layers.0.mlp.down_proj.weight": ("onnx::Gemm_8", (3, 2), False, floats(*range(6))),
            "model.wte.weight": ("model.wte.weight", (2, 3), False, floats(*range(6)))})

    def test_matmul_nbits_inputs_carry_the_node_attributes(self):
        attributes = {"K": 32, "N": 2, "bits": 4, "block_size": 32}
        path = self.write([("w_Q4", UINT8, [2, 1, 16], bytes(32)), ("w_scales", FLOAT, [2, 1], floats(1, 1)),
                           ("w_zero_points", UINT8, [2, 1], bytes(2))],
                          [("MatMulNBits", "/lm_head/MatMul_Q4", ["x", "w_Q4", "w_scales", "w_zero_points"],
                            attributes)])
        tensors = {t.name: t for t in inspect_onnx(path).tensors}
        self.assertEqual(sorted(tensors), ["lm_head.nbits_scales", "lm_head.nbits_weight", "lm_head.nbits_zero_points"])
        self.assertEqual(dict(tensors["lm_head.nbits_weight"].attributes), attributes)

    def test_external_data_stays_beside_the_model_and_in_bounds(self):
        (self.root / "model.onnx.data").write_bytes(bytes(64))
        good = {"location": "model.onnx.data", "offset": 32, "length": 24}
        path = self.write([("model.embed.weight", FLOAT, [2, 3], None, good)], [])
        tensor = inspect_onnx(path).tensors[0]
        self.assertEqual((tensor.shard.name, tensor.data_begin, tensor.data_end), ("model.onnx.data", 32, 56))
        self.assertEqual(inspect_onnx(path).data_files, ("model.onnx.data",))
        cases = {"past its end": ({**good, "offset": 48}, "needs bytes \\[48, 72\\)"),
                 "outside the folder": ({**good, "location": "../model.onnx.data"}, "not a file beside the model"),
                 "missing": ({**good, "location": "other.data"}, "other.data is missing"),
                 "the wrong length": ({**good, "length": 20}, "its shape needs 24")}
        for label, (external, error) in cases.items():
            with self.subTest(label):
                path = self.write([("model.embed.weight", FLOAT, [2, 3], None, external)], [])
                with self.assertRaisesRegex(OnnxError, error):
                    inspect_onnx(path)

    def test_graphs_crit_does_not_read_are_refused(self):
        path = self.write([("w", 3, [4], bytes(4))], [("MatMulInteger", "/a/MatMul", ["x", "w"])])
        with self.assertRaisesRegex(OnnxError, "MatMulInteger .* holds weights crit does not read"):
            inspect_onnx(path)
        path = self.write([("w", FLOAT, [2], floats(1))], [])
        with self.assertRaisesRegex(OnnxError, "holds 4 bytes; its shape needs 8"):
            inspect_onnx(path)
        (self.root / "model.onnx").write_bytes(b"\x0a\xff")
        with self.assertRaisesRegex(OnnxError, "truncated|runs past"):
            inspect_onnx(self.root / "model.onnx")

    def test_the_model_source(self):
        self.write([("model.norm.weight", FLOAT, [2], floats(1, 2))], [])
        (self.root / "config.json").write_text(json.dumps({"num_hidden_layers": 3}))
        source = open_model(self.root)
        self.assertEqual((source.format, source.folder, source.config().num_hidden_layers), ("onnx", self.root, 3))
        self.assertEqual([f.path.name for f in source.weight_files()], ["model.onnx"])
        # The layout of exported repositories: config.json at the root, the graphs in onnx/.
        (self.root / "onnx").mkdir()
        self.write([("model.norm.weight", FLOAT, [2], floats(1, 2))], [], "onnx/model_q4.onnx")
        self.assertEqual(open_model(self.root / "onnx/model_q4.onnx").folder, self.root)
        self.assertEqual(description_folder(self.root / "onnx"), self.root)
        self.write([("x", FLOAT, [1], floats(1))], [], "onnx/model.onnx")
        with self.assertRaisesRegex(OnnxError, "2 .onnx files; name the one"):
            open_model(self.root / "onnx")
        (self.root / "genai_config.json").write_text("{}")
        with self.assertRaisesRegex(OnnxError, "onnxruntime-genai's model builder"):
            open_model(self.root / "model.onnx")

    @unittest.skipUnless(HAS_ONNX, "needs onnx")
    def test_onnx_helper_graphs_read_alike(self):
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper, save_model
        weight = np.arange(6, dtype=np.float32).reshape(2, 3)
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "onnx::MatMul_3"], ["y"], name="/model/proj/MatMul")], "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
            [numpy_helper.from_array(weight, "onnx::MatMul_3"),
             numpy_helper.from_array(np.ones(4, dtype=np.float16), "model.norm.weight")])
        for external in (False, True):
            with self.subTest(external=external):
                path = self.root / f"helper-{external}.onnx"
                save_model(helper.make_model(graph), path, save_as_external_data=external,
                           size_threshold=0, location=f"helper-{external}.data")
                found = {t.name: t for t in inspect_onnx(path).tensors}
                tensor = found["model.proj.weight"]
                data = tensor.shard.read_bytes()[tensor.data_begin:tensor.data_end]
                self.assertEqual((data, tensor.transposed), (weight.tobytes(), True))
                self.assertEqual(found["model.norm.weight"].dtype, "F16")


if __name__ == "__main__":
    unittest.main()
