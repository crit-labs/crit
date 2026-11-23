"""crit_plugins.encodings: a stored matrix's exact values (FP8 in blocks included) and its NVFP4 and FP8 re-encodings,
code for code what quantized-projections' reference codec (nvfp4.py) gives."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference" / "core" / "python"))

try:
    import torch
except ImportError:   # pragma: no cover
    torch = None


def _codec():
    spec = importlib.util.spec_from_file_location("qp_nvfp4", ROOT / "inference/techniques/projections/quantized-projections/code/nvfp4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "torch")
class EncodingTests(unittest.TestCase):
    def blocks(self):
        """An FP8 matrix of 6 x 5 blocks of 4 x 8 values (the last row and column block cut short), one BF16
        multiplier a block, as DeepSeek-style checkpoints store it (weight_scale_inv)."""
        g = torch.Generator().manual_seed(4)
        codes = torch.randint(0, 0x7F, (22, 37), dtype=torch.uint8, generator=g)
        codes[1::3] |= 0x80
        scale = (torch.rand(6, 5, generator=g) * 0.01 + 0.001).to(torch.bfloat16)
        return {"weight": codes.view(torch.float8_e4m3fn), "scale": scale.float(), "block": (4, 8)}

    def test_fp8_blocks_read_as_each_code_times_its_blocks_multiplier(self):
        from crit_plugins import encodings
        tensors = self.blocks()
        found = encodings.values(tensors, "fp8:block", "cpu", 3, 22)
        codes = tensors["weight"].to(torch.float64)
        for r, c in ((3, 0), (5, 8), (21, 36), (12, 17)):
            want = float(codes[r, c]) * float(tensors["scale"][r // 4, c // 8])
            self.assertEqual(float(found[r - 3, c]), want, (r, c))

    def test_nvfp4_codes_and_scales_are_the_reference_codecs(self):
        from crit_plugins import encodings
        codec = _codec()
        g = torch.Generator().manual_seed(5)
        weight = (torch.randn(6, 48, generator=g) * 0.3).to(torch.bfloat16)
        found, fmt = encodings.encode({"weight": weight}, "bf16", "nvfp4", "cpu")
        self.assertEqual(fmt, "nvfp4")
        values = weight.double()
        g32 = float(torch.tensor(codec.global_scale(values.reshape(-1).tolist()), dtype=torch.float32))
        self.assertEqual(float(found["global_scale"]), g32)
        for r in range(6):
            codes, scales = codec.quantize_row(values[r].tolist(), g32)
            packed = [codes[2 * k] | (codes[2 * k + 1] << 4) for k in range(24)]
            self.assertEqual(found["scale"][r].tolist(), list(scales), r)
            self.assertEqual(found["weight"][r].tolist(), packed, r)

    def test_fp8_blocks_re_encode_to_nvfp4_and_fp8_rows(self):
        from crit_plugins import encodings
        tensors = self.blocks()
        tensors["weight"] = tensors["weight"][:, :32]
        tensors["scale"] = tensors["scale"][:, :4]
        exact = encodings.values(tensors, "fp8:block", "cpu", 0, 22)
        rows, fmt = encodings.encode(tensors, "fp8:block", "fp8")
        self.assertEqual(fmt, "fp8:channel")
        back = rows["weight"].to(torch.float64) * rows["scale"].double()[:, None]
        self.assertLess(float((back - exact).abs().max() / exact.abs().max()), 1 / 8)
        packed, fmt = encodings.encode(tensors, "fp8:block", "nvfp4")
        self.assertEqual((fmt, tuple(packed["weight"].shape), tuple(packed["scale"].shape)), ("nvfp4", (22, 16), (22, 2)))


if __name__ == "__main__":
    unittest.main()
