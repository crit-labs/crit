"""host-expert-arithmetic's device form on the CPU: every row's bits those of the row alone (assertKeepsRows, prompt rows
too), the native rows and the torch path giving the same bits in every activation format, the weight-only rows the host
form's FP32 order on the same matrix, and each format within its quantization's error of the dense product."""
import math
import unittest

from crit_plugins.testing import PluginTestCase

try:
    import torch
except ImportError:
    torch = None


def stored(generator, rows, cols, int4=False):
    return {"weight": torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator),
            "scale": torch.randint(0x30, 0x40, (rows, cols // 16), dtype=torch.uint8, generator=generator),
            "global_scale": torch.rand(rows, generator=generator) * 2 + 1, "input_scale": torch.tensor(30.0)}


def dense(m, int4=False):
    codes = m["weight"]
    nib = torch.stack([(codes & 15).long(), (codes >> 4).long()], -1).reshape(codes.shape[0], -1)
    if int4:
        values = torch.where(nib < 8, nib, nib - 16).double()
    else:
        magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
        values = torch.where(nib < 8, magnitude[nib & 7], -magnitude[nib & 7])
    scales = m["scale"].view(torch.float8_e4m3fn).double().repeat_interleave(16, dim=1)
    return values * scales / m["global_scale"].double()[:, None]


@unittest.skipIf(torch is None, "the device form's tensors are torch's")
class HostArithmeticDeviceTests(PluginTestCase):
    def test_rows_keep_their_bits_alone_and_batched(self):
        for activations in ("fp32", "e2m1x2", "int8"):
            with self.subTest(activations=activations):
                p = self.assemble(options={"activations": activations})["projections"]
                self.assertKeepsRows(p)
                m = p.load(stored(torch.Generator().manual_seed(2), 96, 256), "nvfp4")
                x = torch.randn(40, 256, generator=torch.Generator().manual_seed(3)).bfloat16()
                whole = p.matmul(x, m, torch.float32)
                for first, count in ((0, 1), (5, 8), (9, 31)):
                    self.assertTrue(torch.equal(p.matmul(x[first:first + count].clone(), m, torch.float32),
                                                whole[first:first + count]))

    def test_the_native_rows_and_the_torch_path_give_the_same_bits(self):
        generator = torch.Generator().manual_seed(5)
        for format, activations in (("nvfp4", "fp32"), ("nvfp4", "e2m1x2"), ("nvfp4", "int8"), ("int4", "int8"),
                                    ("int4", "e2m1x2")):
            with self.subTest(format=format, activations=activations):
                native = self.assemble(options={"activations": activations})["projections"]
                if not native._rows():
                    self.skipTest("the native rows do not build here")
                fallback = self.assemble(options={"activations": activations, "native": "off"})["projections"]
                tensors = stored(generator, 80, 192, format == "int4")
                x = torch.randn(6, 192, generator=generator) * 3
                a = native.matmul(x, native.load(tensors, format), torch.float32)
                b = fallback.matmul(x, fallback.load(tensors, format), torch.float32)
                self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)))

    def test_weight_only_rows_are_the_host_forms_fp32_order(self):
        p = self.assemble()["projections"]
        w = [[float(v) for v in row] for row in (torch.rand(8, 64, generator=torch.Generator().manual_seed(7)) * 2 - 1)]
        x = [[float(v) for v in row] for row in torch.randn(3, 64, generator=torch.Generator().manual_seed(8))]
        host_matrix = p.prepare(w)
        host_matrix.g = 2.0 ** round(math.log2(host_matrix.g))   # a post both forms round alike (the codes stay)
        tensors = {"weight": torch.tensor(list(host_matrix.packed), dtype=torch.uint8).reshape(8, 32),
                   "scale": torch.tensor(list(host_matrix.scales), dtype=torch.uint8).reshape(8, 4),
                   "global_scale": torch.tensor(host_matrix.g, dtype=torch.float32)}
        device = p.matmul(torch.tensor(x), p.load(tensors, "nvfp4"), torch.float32)
        self.assertEqual(device.tolist(), p.project(host_matrix, x))

    def test_each_format_is_within_its_quantization_error(self):
        generator = torch.Generator().manual_seed(9)
        for format, activations, bound in (("nvfp4", "fp32", 1e-6), ("nvfp4", "int8", 0.02), ("nvfp4", "e2m1x2", 0.2),
                                           ("int4", "int8", 0.02)):
            with self.subTest(format=format, activations=activations):
                p = self.assemble(options={"activations": activations})["projections"]
                tensors = stored(generator, 64, 256, format == "int4")
                x = torch.randn(5, 256, generator=generator)
                out = p.matmul(x, p.load(tensors, format), torch.float32).double()
                want = x.double() @ dense(tensors, format == "int4").T
                self.assertLess(float((out - want).norm() / want.norm()), bound)

    def test_it_refuses_what_it_does_not_compute(self):
        p = self.assemble(options={"activations": "e2m1x2"})["projections"]
        tensors = stored(torch.Generator().manual_seed(1), 16, 64)
        del tensors["input_scale"]
        with self.assertRaisesRegex(ValueError, "input_scale"):
            p.load(tensors, "nvfp4")
        with self.assertRaisesRegex(ValueError, "loads bf16, nvfp4, int4"):
            p.load(tensors, "fp8:channel")
