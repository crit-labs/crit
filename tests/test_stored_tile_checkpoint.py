"""stored-tile-gemm's checkpoint sources: NVFP4 (compressed-tensors, ModelOpt) and FP8 matrices read as stored, their
global scales as the kernels' divisors, checked against the host tile model."""
import importlib.util
import json
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core" / "python"))

from crit_core.model import SchemeError, open_tensors  # noqa: E402

CODE = ROOT / "inference" / "stored-tile-gemm" / "code"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"stored_tile_{name}", CODE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checkpoint = _load("checkpoint")
layout = _load("layout")


def write_safetensors(path, tensors):
    """tensors: {name: (dtype, shape, bytes)} -> a .safetensors file."""
    header, payload = {}, bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(payload))
    return path


def f32(*values):
    return struct.pack(f"<{len(values)}f", *values)


def bf16(*values):
    return b"".join(struct.pack("<f", v)[2:] for v in values)


class TileSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rng = random.Random(7)

    def reader(self, tensors, name="model.safetensors"):
        reader = open_tensors(write_safetensors(self.root / name, tensors))
        self.addCleanup(reader.close)
        return reader

    def nvfp4(self, n, k, experts=None):
        lead = () if experts is None else (experts,)
        count = (experts or 1) * n
        codes = bytes(self.rng.randrange(256) for _ in range(count * k // 2))
        scales = bytes(self.rng.choice((0x30, 0x38, 0x40, 0x44)) for _ in range(count * k // 16))   # E4M3 0.5..3
        return codes, scales, lead + (n, k // 2), lead + (n, k // 16)

    def dequantized(self, source, row, col):
        """weight[row][col] from the source the way the tiles' consumers compute it: code * scale / global."""
        byte = source.codes[row * source.k // 2 + col // 2]
        code = byte & 0xF if col % 2 == 0 else byte >> 4
        scale = layout.e4m3(source.scales[row * source.k // 16 + col // 16])
        return layout.e2m1(code) * scale / source.weight_global_scale

    def test_compressed_tensors_and_modelopt_nvfp4_agree(self):
        codes, scales, cshape, sshape = self.nvfp4(32, 128)
        ct = self.reader({"m.weight_packed": ("U8", cshape, codes), "m.weight_scale": ("F8_E4M3", sshape, scales),
                          "m.weight_global_scale": ("F32", (1,), f32(4.0)),
                          "m.input_global_scale": ("F32", (1,), f32(2.0))})
        a = checkpoint.tile_source(ct, "m", {"quant_method": "compressed-tensors"})
        self.assertEqual((a.format, a.n, a.k, a.weight_global_scale, a.input_global_scale), ("fp4", 32, 128, 4.0, 2.0))
        self.assertEqual((bytes(a.codes), bytes(a.scales)), (codes, scales))
        mo = self.reader({"m.weight": ("U8", cshape, codes), "m.weight_scale": ("F8_E4M3", sshape, scales),
                          "m.weight_scale_2": ("F32", (), f32(0.25)), "m.input_scale": ("F32", (), f32(0.5))},
                         "modelopt.safetensors")
        b = checkpoint.tile_source(mo, "m", {"quant_method": "modelopt"})
        self.assertEqual((b.weight_global_scale, b.input_global_scale, b.parts.layout), (4.0, 2.0, "modelopt"))
        for row, col in ((0, 0), (5, 17), (31, 127), (16, 64)):
            self.assertEqual(self.dequantized(a, row, col), self.dequantized(b, row, col))

    def test_nvfp4_codes_fill_the_host_tile_model(self):
        codes, scales, cshape, sshape = self.nvfp4(16, 64)
        source = checkpoint.tile_source(self.reader({
            "m.weight_packed": ("U8", cshape, codes), "m.weight_scale": ("F8_E4M3", sshape, scales),
            "m.weight_global_scale": ("F32", (1,), f32(1.0))}), "m", {"quant_method": "compressed-tensors"})
        rows = [[(source.codes[r * 32 + c // 2] >> (4 * (c % 2))) & 0xF for c in range(64)] for r in range(16)]
        tile = sum(layout.fp4_tile(rows), [])
        self.assertEqual(len(tile), 512)
        self.assertEqual(sorted(tile), sorted(source.codes[:512]))   # a permutation of the stored bytes

    def test_stacked_experts(self):
        codes, scales, cshape, sshape = self.nvfp4(16, 64, experts=3)
        reader = self.reader({"e.weight_packed": ("U8", cshape, codes), "e.weight_scale": ("F8_E4M3", sshape, scales),
                              "e.weight_global_scale": ("F32", (1,), f32(1.0))})
        one = checkpoint.tile_source(reader, "e", {"quant_method": "compressed-tensors"}, expert=2)
        self.assertEqual(bytes(one.codes), codes[2 * 16 * 32:])
        self.assertEqual(bytes(one.scales), scales[2 * 16 * 4:])
        for expert in (None, 3):
            with self.assertRaisesRegex(SchemeError, "needs an expert index"):
                checkpoint.tile_source(reader, "e", {"quant_method": "compressed-tensors"}, expert=expert)

    def test_fp8_tensor_and_row_scales(self):
        codes = bytes(self.rng.randrange(256) for _ in range(16 * 32))
        reader = self.reader({"t.weight": ("F8_E4M3", (16, 32), codes), "t.weight_scale": ("F32", (1,), f32(0.5)),
                              "r.weight": ("F8_E4M3", (16, 32), codes),
                              "r.weight_scale": ("BF16", (16, 1), bf16(*[0.25 * (i + 1) for i in range(16)]))})
        tensor = checkpoint.tile_source(reader, "t")
        self.assertEqual((tensor.format, list(tensor.row_scale), bytes(tensor.codes)), ("fp8", [0.5] * 16, codes))
        self.assertEqual(list(checkpoint.tile_source(reader, "r").row_scale), [0.25 * (i + 1) for i in range(16)])

    def test_refusals(self):
        reader = self.reader({
            "block.weight": ("F8_E4M3", (256, 256), bytes(256 * 256)),
            "block.weight_scale_inv": ("F32", (2, 2), f32(1, 1, 1, 1)),
            "e5.weight": ("F8_E5M2", (16, 32), bytes(512)), "e5.weight_scale": ("F32", (1,), f32(1.0)),
            "odd.weight": ("F8_E4M3", (16, 48), bytes(16 * 48)), "odd.weight_scale": ("F32", (1,), f32(1.0)),
            "plain.weight": ("BF16", (16, 32), bytes(1024)),
            "zero.weight_packed": ("U8", (16, 32), bytes(512)), "zero.weight_scale": ("F8_E4M3", (16, 4), bytes(64)),
            "zero.weight_global_scale": ("F32", (1,), f32(0.0))})
        for base, message in (("block", "block scales"), ("e5", "E4M3"), ("odd", "does not fill"),
                              ("plain", "stored as plain"), ("zero", "not positive")):
            with self.subTest(base=base), self.assertRaisesRegex(SchemeError, message):
                checkpoint.tile_source(reader, base, {"quant_method": "compressed-tensors"})
        with self.assertRaises(KeyError):
            checkpoint.tile_source(reader, "missing")


if __name__ == "__main__":
    unittest.main()
