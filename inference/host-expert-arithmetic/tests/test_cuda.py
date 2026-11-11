"""host-expert-arithmetic's CUDA form on int-block-experts: decode rows and declared prompt rows the CPU rows' bits in
W4A4 and W4A8 (NVFP4 and INT4 weights, one global scale or one a row), and with prompt_gemm native the W4A4 prompt rows
on gemm-native within FP32 accumulation of the FP64 product of the same quantized activations, the facts saying they
are not row invariant."""
import unittest

from crit_plugins.testing import PluginTestCase

try:
    import torch
except ImportError:
    torch = None


def stored(generator, rows, cols, per_row=True):
    g = torch.rand(rows, generator=generator) * 2 + 1
    return {"weight": torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator),
            "scale": torch.randint(0x30, 0x40, (rows, cols // 16), dtype=torch.uint8, generator=generator),
            "global_scale": g if per_row else g[:1], "input_scale": torch.tensor(30.0)}


def on(tensors, device):
    return {k: v.to(device) for k, v in tensors.items()}


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "the CUDA form runs on CUDA")
class CudaFormTests(PluginTestCase):
    def setUp(self):
        from crit_plugins.kernels import load_kernel
        from pathlib import Path
        self.ib = load_kernel("int-block-experts", folder=Path(__file__).resolve().parents[1])
        if self.ib is None:
            self.skipTest("int-block-experts is not here")

    def test_decode_and_declared_prompt_rows_are_the_cpu_rows_bits(self):
        generator = torch.Generator().manual_seed(7)
        for format, activations, per_row in (("nvfp4", "e2m1x2", True), ("nvfp4", "int8", False),
                                             ("int4", "int8", True), ("int4", "e2m1x2", False)):
            with self.subTest(format=format, activations=activations):
                options = {"activations": activations, "prompt_gemm": "declared"}
                p = self.assemble(options=options)["projections"]
                tensors = stored(generator, 160, 512, per_row)
                cpu, cuda = p.load(tensors, format), p.load(on(tensors, "cuda"), format)
                x = torch.randn(70, 512, generator=generator) * 3
                want = p.matmul(x, cpu, torch.float32)
                got = p.matmul(x.cuda(), cuda, torch.float32).cpu()
                self.assertTrue(torch.equal(got.view(torch.int32), want.view(torch.int32)))
                for rows in (1, 5, 16):
                    part = p.matmul(x[:rows].cuda(), cuda, torch.float32).cpu()
                    self.assertTrue(torch.equal(part.view(torch.int32), want[:rows].view(torch.int32)), rows)
                self.assertTrue(p.facts()["row_invariance"]["prompt_rows"])

    def test_native_w4a4_prompt_rows_are_within_fp32_accumulation_of_the_exact_product(self):
        generator = torch.Generator().manual_seed(9)
        for format in ("nvfp4", "int4"):
            with self.subTest(format=format):
                p = self.assemble(options={"activations": "e2m1x2"})["projections"]
                declared = self.assemble(options={"activations": "e2m1x2", "prompt_gemm": "declared"})["projections"]
                tensors = on(stored(generator, 192, 1024), "cuda")
                m, d = p.load(tensors, format), declared.load(tensors, format)
                x = (torch.randn(300, 1024, generator=generator) * 3).cuda()
                got = p.matmul(x, m, torch.float32).double()
                # the exact product of the quantized activations (quantize_nvfp4's codes and scales) and the weights
                codes, scales = self.ib.quantize_nvfp4(x.float(), 30.0)
                e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                                    dtype=torch.float64, device="cuda")
                def nibbles(c, table):
                    return torch.stack([table[(c & 15).long()], table[(c >> 4).long()]], -1).reshape(c.shape[0], -1)
                a = nibbles(codes, e2m1) * scales.view(torch.float8_e4m3fn).double().repeat_interleave(16, 1) / 30.0
                int4 = torch.tensor([float(c if c < 8 else c - 16) for c in range(16)], dtype=torch.float64,
                                    device="cuda")
                w = nibbles(tensors["weight"], int4 if format == "int4" else e2m1)
                w = w * tensors["scale"].view(torch.float8_e4m3fn).double().repeat_interleave(16, 1) \
                    / tensors["global_scale"].double()[:, None]
                exact, magnitude = a @ w.t(), a.abs() @ w.abs().t()
                self.assertLessEqual(float(((got - exact).abs() - magnitude * 2e-6).max()), 1e-30)
                want = declared.matmul(x, d, torch.float32).double()
                self.assertLess(float((got - want).abs().max() / want.abs().max()), 1e-5)
                self.assertFalse(p.facts()["row_invariance"]["prompt_rows"])
                # decode rows keep the declared order (the CPU rows' bits) under prompt_gemm native too
                self.assertTrue(torch.equal(p.matmul(x[:9], m, torch.float32), declared.matmul(x[:9], d, torch.float32)))


if __name__ == "__main__":
    unittest.main()
