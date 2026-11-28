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

    def test_it_states_the_level_its_loaded_rows_keep(self):
        p = self.assemble(options={"activations": "int8"})["projections"]
        self.assertEqual(p.exactness(), {})
        p.load(stored(torch.Generator().manual_seed(2), 96, 256), "nvfp4")
        self.assertEqual(p.exactness(), {"decode": "declared", "prompt": "declared"})   # the declared order, on the CPU
        p.load({"weight": torch.randn(4, 64)}, "bf16")   # a 16-bit matrix: torch, one row at a time, in both phases
        self.assertEqual(p.exactness(), {"decode": "row-exact", "prompt": "exact"})

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

    def test_the_quantizer_and_the_swiglu_keep_their_bits_on_every_isa(self):
        rows = self.assemble(options={"activations": "e2m1x2"})["projections"]._rows()
        if not rows or rows.int_isa() == "scalar":
            self.skipTest("no vector integer path here to compare with the scalar order")
        fast, inf, nan = rows.int_isa(), float("inf"), float("nan")
        g = torch.Generator().manual_seed(11)
        # A block whose scale is 1: its values are the quotients, the E2M1 midpoints among them (ties go to the even
        # code), and blocks of zeros, of infinities and NaNs, of values no scale reaches.
        ties = torch.tensor([6.0, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5.0, -5.0, 0.0])
        odd = torch.tensor([0.0, -0.0, 1e-30, -1e-30, inf, -inf, nan, 1e30, -1e30, 3.4e38, 448.0, -448.0, 7.0, -7.0,
                            6.0001, -5.9999])
        for spread in (1e-3, 1.0, 37.5, 1e4):
            x = torch.cat([ties, torch.zeros(16), odd, torch.full((16,), nan), torch.randn(60 * 16, generator=g) * spread])
            for global_scale in (1.0, 0.5, 30.0, 977.0):
                want = rows.quantize_e2m1x2_on("scalar", x, global_scale)
                got = rows.quantize_e2m1x2_on(fast, x, global_scale)
                self.assertTrue(torch.equal(got[0], want[0]), (spread, global_scale))
                self.assertTrue(torch.equal(got[1].view(torch.int32), want[1].view(torch.int32)), (spread, global_scale))
        self.assertEqual(rows.quantize_e2m1x2_on(fast, ties, 1.0)[0].tolist(),
                         [12, 0, 0, 2, -2, 2, -2, 4, -4, 4, -4, 8, -8, 8, -8, 0])   # twice E2M1: ties to the even code
        gate = torch.cat([torch.randn(4096, generator=g) * 8,
                          torch.tensor([0.0, -0.0, 88.0, 88.5, -87.0, -87.5, 200.0, -200.0, 1e-20, -1e-20, inf, -inf, 0.5])])
        up = torch.randn(gate.numel(), generator=g) * 3    # 4109 values: a tail shorter than a vector too
        self.assertTrue(torch.equal(rows.swiglu_on(fast, gate, up).view(torch.int32),
                                    rows.swiglu_on("scalar", gate, up).view(torch.int32)))

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


@unittest.skipIf(torch is None, "the device form's tensors are torch's")
class CudaRowsPhaseTests(PluginTestCase):
    """The CUDA form picks a call's kernel by its phase: with prompt_gemm native, W4A4 prompt rows take gemm_native
    however few they are (a short chunk the bits of the same rows in a longer one), decode calls the declared rows.
    On the CPU with a recording kernel module in place of int-block-experts."""

    def test_a_prompt_call_takes_the_prompt_gemm_at_every_count(self):
        from importlib import import_module
        plugin = self.assemble()["projections"]
        CudaRows = import_module(f"{plugin.code.__name__}.device_int").CudaRows
        ran = []

        class Kernels:
            def quantize_nvfp4(self, x, scale):
                return x, x

            quantize_e2m1x2 = quantize_int8 = quantize_nvfp4

            def gemm_native(self, *args, **kwargs):
                ran.append("native")

            def gemm_staged(self, *args, **kwargs):
                ran.append("staged")

            def rows(self, *args, **kwargs):
                ran.append("rows")

            def row_jobs(self, jobs, device):
                return jobs

        from types import SimpleNamespace
        rows = CudaRows(Kernels(), "native")
        matrix = SimpleNamespace(rows=4, activations="e2m1x2",
                                 tensors={"table": None, "jobs": torch.zeros(3, dtype=torch.int32), "input": 1.0,
                                          "int4": False, "inverse": torch.ones(4)})
        for count, prompt, want in ((1, True, "native"), (5, True, "native"), (40, True, "native"),
                                    (1, False, "rows"), (8, False, "rows")):
            ran.clear()
            rows.matmul(torch.zeros(count, 32), matrix, None, prompt=prompt)
            self.assertEqual(ran, [want], (count, prompt))
