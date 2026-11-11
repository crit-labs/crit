"""rounds:projections at real widths: every decode arithmetic captured in a CUDA graph and replayed on new inputs gives
the eager bits, and a row of a k-row call (k 1..8) is the row alone, on crit's kernels (built on this GPU) and on the
fallbacks alike."""
import unittest

from crit_plugins.testing import PluginTestCase

# A wide projection (a fused q/k/v or gate/up) and a down projection of a model of hidden size 5120.
SHAPES = ((10240, 5120), (5120, 17408))


class OnCuda:
    device = "cuda"


class RoundTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("captured rounds run on CUDA")
        import torch
        self.torch = torch
        self.g = torch.Generator().manual_seed(11)

    def stored(self, format, n, k):
        torch, g = self.torch, self.g
        if format == "bf16":
            return {"weight": (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)}
        if format == "fp8:channel":
            codes = torch.randint(0, 0x7E, (n, k), generator=g, dtype=torch.uint8)
            codes |= torch.randint(0, 2, (n, k), generator=g, dtype=torch.uint8) << 7
            return {"weight": codes.view(torch.float8_e4m3fn), "scale": torch.rand(n, generator=g) / 100 + 1e-3}
        if format == "mxfp8":
            codes = torch.randint(0, 0x7E, (n, k), generator=g, dtype=torch.uint8)
            return {"weight": codes.view(torch.float8_e4m3fn),
                    "scale": torch.randint(118, 128, (n, k // 32), generator=g, dtype=torch.uint8)}
        return {"weight": torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8),
                "scale": (torch.rand(n, k // 16, generator=g) * 2 + 0.01).to(torch.float8_e4m3fn),
                "global_scale": torch.tensor(500.0), "input_scale": torch.tensor(300.0)}

    def check(self, plugin, m, k):
        torch = self.torch
        x = torch.randn(8, k, generator=self.g).cuda().to(torch.bfloat16)
        alone = [plugin.matmul(x[r:r + 1].clone(), m, torch.float32)[0] for r in range(8)]
        for rows in range(1, 9):
            batch = plugin.matmul(x[:rows], m, torch.float32)
            for r in range(rows):
                self.assertTrue(torch.equal(batch[r], alone[r]), (m.format, m.decode, rows, r))
        for rows in (1, 3, 8):
            static = x[:rows].clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                plugin.matmul(static, m, torch.float32)   # the round's eager run before its capture
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = plugin.matmul(static, m, torch.float32)
            fresh = torch.randn(rows, k, generator=self.g).cuda().to(torch.bfloat16)
            static.copy_(fresh)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(captured, plugin.matmul(fresh, m, torch.float32)), (m.format, m.decode, rows))

    def test_every_decode_arithmetic_captures_and_keeps_rows_at_real_widths(self):
        cases = [("bf16", {}), ("fp8:channel", {}), ("fp8:channel", {"decode_activations": "bf16"}), ("mxfp8", {}),
                 ("nvfp4", {"activations": "bf16"}), ("nvfp4", {}), ("nvfp4", {"decode_activations": "bf16"})]
        for format, options in cases:
            plugin = self.assemble(options=options, context=OnCuda())["projections"]
            self.assertEqual(plugin.round_problems(), [])
            for n, k in SHAPES:
                with self.subTest(format=format, options=options, shape=(n, k)):
                    self.check(plugin, plugin.load(self.stored(format, n, k), format), k)
        self.assertIn("rounds:projections", self.registry()["quantized-projections"]["provides"])

    def test_triton_kernels_are_compiled_at_load_and_a_first_call_inside_a_capture_names_the_warm_up(self):
        torch = self.torch
        plugin = self.assemble(options={"activations": "bf16", "gemv": "triton"}, context=OnCuda())["projections"]
        m = plugin.load(self.stored("nvfp4", 96, 256), "nvfp4")
        if m.triton is None:
            self.skipTest("the Triton GEMV comes with decode-gemv, not here")
        static = torch.randn(5, 256, generator=self.g).cuda().to(torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):   # the load warmed every row count the GEMV takes
            plugin.matmul(static, m)
        import importlib
        from crit_plugins.assemble import load_module
        nvfp4 = importlib.import_module(load_module(self.registry()["quantized-projections"]).device_matrices.__package__
                                        + ".device_nvfp4")
        warm = set(nvfp4._WARM)
        nvfp4._WARM.clear()   # as if the load had not warmed it
        try:
            with self.assertRaisesRegex(RuntimeError, "run the step once eagerly before capturing it"):
                with torch.cuda.graph(torch.cuda.CUDAGraph()):
                    plugin.matmul(static, m)
        finally:
            nvfp4._WARM.update(warm)


if __name__ == "__main__":
    unittest.main()
