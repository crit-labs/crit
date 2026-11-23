"""quantized-projections' decode kernels per matrix: with gemv auto, FP8 matrices' 16-bit decode rows on
decode-gemv's dot GEMV (one kernel for every decode row count, a row's bits the same alone and in a call of
up to 16 rows at any offset, the stored weights' exact products in FP32 sums, compiled at load; without the kernel, the
tile); W4A4 decode rows on mma-rows' block-scaled FP4 MMA; decode_formats setting the decode activations per format;
the facts naming each role's kernel and activations."""
import unittest
from unittest.mock import patch

from crit_plugins.testing import PluginTestCase

SHAPES = ((16384, 5120), (5120, 6144), (2560, 640))   # a 27B's qkvz and out, a hybrid MoE's shared-expert down


class OnCuda:
    device = "cuda"


class DotTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("the dot GEMV runs on CUDA")
        import importlib
        import torch
        from crit_plugins.assemble import load_module
        self.torch = torch
        self.matrices = load_module(self.registry()["quantized-projections"]).device_matrices
        self.nvfp4 = importlib.import_module(self.matrices.__package__ + ".device_nvfp4")
        self.g = torch.Generator().manual_seed(29)
        if self.nvfp4.dot("cuda") is None:
            self.skipTest("the dot GEMV comes with decode-gemv, not here")

    def fp8(self, n, k):
        torch, g = self.torch, self.g
        codes = torch.randint(0, 0x7E, (n, k), generator=g, dtype=torch.uint8)
        codes |= torch.randint(0, 2, (n, k), generator=g, dtype=torch.uint8) << 7
        return {"weight": codes.view(torch.float8_e4m3fn), "scale": torch.rand(n, generator=g) / 100 + 1e-3}

    def plugin(self, **options):
        return self.assemble(options={"decode_activations": "bf16", "prompt_gemm": "pinned", **options},
                             context=OnCuda())["projections"]

    def test_fp8_decode_rows_keep_their_bits_at_every_offset_and_multiply_the_exact_weights(self):
        torch = self.torch
        plugin = self.plugin()
        for n, k in SHAPES:
            stored = self.fp8(n, k)
            m = plugin.load(stored, "fp8:channel", role=f"r{n}x{k}")
            self.assertEqual(m.path, "dot")
            x = torch.randn(24, k, generator=self.g)
            x[:, torch.randperm(k, generator=self.g)[:8]] *= 60.0
            x = x.cuda().to(torch.bfloat16)
            exact = x.double() @ (stored["weight"].double() * stored["scale"].double()[:, None]).cuda().t()
            for out in (torch.float32, None):
                alone = torch.cat([plugin.matmul(x[r:r + 1].clone(), m, out) for r in range(24)])
                if out == torch.float32:
                    self.assertLess(float((alone.double() - exact).abs().max() / exact.abs().max()), 2e-5)
                for rows in range(1, 17):
                    for first in (0, 3, 24 - rows):
                        self.assertTrue(torch.equal(plugin.matmul(x[first:first + rows].clone(), m, out),
                                                    alone[first:first + rows]), (n, k, rows, first, out))
        facts = plugin.facts()
        self.assertEqual(facts["decode_paths"]["fp8"], ["dot"])
        self.assertEqual(facts["decode_kernels"], {f"r{n}x{k}": {"kernels": ["dot"], "activations": ["bf16"]}
                                                   for n, k in SHAPES})
        self.assertEqual(facts["row_invariance"]["rows"], 16)

    def test_other_matrices_keep_the_tile_and_without_the_kernel_fp8_takes_it(self):
        torch = self.torch
        plugin = self.plugin()
        nvfp4 = {"weight": torch.randint(0, 256, (1024, 512), generator=self.g, dtype=torch.uint8),
                 "scale": (torch.rand(1024, 64, generator=self.g) * 2 + 0.01).to(torch.float8_e4m3fn),
                 "global_scale": torch.tensor(500.0), "input_scale": torch.tensor(300.0)}
        self.assertEqual(plugin.load(nvfp4, "nvfp4", role="gate_up").path, "mma")
        self.assertEqual(plugin.facts()["decode_kernels"], {"gate_up": {"kernels": ["mma"], "activations": ["bf16"]}})
        quantized = self.assemble(context=OnCuda())["projections"]   # W8A8 decode rows: the tile's quantizer
        self.assertEqual(quantized.load(self.fp8(1024, 1024), "fp8:channel").path, "mma")
        with patch.object(self.nvfp4, "dot", lambda device: None):
            self.assertEqual(self.plugin().load(self.fp8(1024, 1024), "fp8:channel").path, "mma")
        self.assertEqual(self.plugin(gemv="mma").load(self.fp8(1024, 1024), "fp8:channel").path, "mma")

    def test_no_request_compiles_the_dot_gemv_after_load(self):
        torch = self.torch
        import triton
        plugin = self.plugin()
        m = plugin.load(self.fp8(2048, 2048), "fp8:channel")
        compiled = []
        hook = triton.knobs.runtime.jit_cache_hook
        triton.knobs.runtime.jit_cache_hook = lambda *a, **k: compiled.append(k.get("fn")) or False
        try:
            buffer = torch.randn(40, 4096, generator=self.g).cuda().to(torch.bfloat16)
            for rows in range(1, 17):
                for out in (None, torch.float32):
                    plugin.matmul(buffer[:rows, :2048], m, out)   # a strided slice, as a model's buffers give
                    plugin.matmul(buffer[:rows, :2048].contiguous(), m, out)
            torch.cuda.synchronize()
        finally:
            triton.knobs.runtime.jit_cache_hook = hook
        self.assertEqual(compiled, [])


class Fp4DecodeTests(PluginTestCase):
    """W4A4 decode rows (decode_activations prompt) on mma-rows' block-scaled FP4 MMA: one plan for every decode row
    count, a row's bits the same alone and in a call of up to 16 rows at any offset, within FP32 sums of the BF16-MMA
    path on the same quantized activations, the facts naming the kernel and the activations."""

    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("the FP4 MMA runs on CUDA")
        import importlib
        import torch
        from crit_plugins.assemble import load_module
        self.torch = torch
        self.matrices = load_module(self.registry()["quantized-projections"]).device_matrices
        self.nvfp4 = importlib.import_module(self.matrices.__package__ + ".device_nvfp4")
        module = self.nvfp4.mma("cuda")
        if module is None or not callable(getattr(module, "fp4_mma", None)) or not module.fp4_mma("cuda"):
            self.skipTest("the FP4 MMA comes with mma-rows on the sm_120 family, not here")
        self.g = torch.Generator().manual_seed(31)

    def stored(self, n, k, runs=None):
        torch, g = self.torch, self.g
        found = {"weight": torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8),
                 "scale": (torch.rand(n, k // 16, generator=g) * 2 + 0.01).to(torch.float8_e4m3fn),
                 "global_scale": torch.tensor(500.0), "input_scale": torch.tensor(300.0)}
        if runs:
            found["global_scale"] = torch.cat([torch.full((r,), 400.0 + 50 * i) for i, r in enumerate(runs)])
        return found

    def test_w4a4_decode_rows_keep_their_bits_and_match_the_bf16_mma_path(self):
        torch = self.torch
        fp4 = self.assemble(options={"decode_activations": "prompt", "prompt_gemm": "pinned"},
                            context=OnCuda())["projections"]
        for n, k, runs in ((34816, 5120, (17408, 17408)), (5120, 17408, None), (9216, 7168, (8192, 512, 512))):
            stored = self.stored(n, k, runs)
            m = fp4.load(stored, "nvfp4", role=f"r{n}")
            self.assertEqual((m.path, m.mma_acts["decode"], m.tile_rows), ("mma_fp4", "nvfp4_mma", 0))
            x = torch.randn(24, k, generator=self.g)
            x[:, torch.randperm(k, generator=self.g)[:8]] *= 60.0
            x = x.cuda().to(torch.bfloat16)
            for out in (torch.float32, None):
                alone = torch.cat([fp4.matmul(x[r:r + 1].clone(), m, out) for r in range(24)])
                for rows in range(1, 17):
                    for first in (0, 5, 24 - rows):
                        self.assertTrue(torch.equal(fp4.matmul(x[first:first + rows].clone(), m, out),
                                                    alone[first:first + rows]), (n, rows, first, out))
            with patch.object(self.nvfp4.mma("cuda"), "fp4_mma", lambda device=None: False):
                bf16 = self.assemble(options={"decode_activations": "prompt"}, context=OnCuda())["projections"]
                b = bf16.load(stored, "nvfp4")
            self.assertEqual(b.mma_acts["decode"], "nvfp4")
            ref = torch.cat([bf16.matmul(x[r:r + 1].clone(), b, torch.float32) for r in range(16)])
            got = fp4.matmul(x[:16], m, torch.float32)
            self.assertLess(float((got - ref).abs().max() / ref.abs().max()), 1e-5)
            del m, b
        facts = fp4.facts()
        self.assertEqual(facts["decode_kernels"]["r5120"], {"kernels": ["mma_fp4"], "activations": ["nvfp4"]})
        self.assertIn("mma_fp4", facts["decode_paths"]["nvfp4"])
        self.assertEqual(facts["row_invariance"]["rows"], 16)
        w4a16 = self.assemble(options={"decode_activations": "bf16"}, context=OnCuda())["projections"]
        self.assertEqual(w4a16.load(self.stored(1024, 1024), "nvfp4", role="down").path, "mma")
        self.assertEqual(w4a16.facts()["decode_kernels"]["down"], {"kernels": ["mma"], "activations": ["bf16"]})
        tile = self.assemble(options={"decode_activations": "prompt", "gemv": "mma"}, context=OnCuda())["projections"]
        m = tile.load(self.stored(1024, 1024), "nvfp4", role="gate")
        self.assertEqual((m.path, m.mma_acts["decode"]), ("mma", "nvfp4"))   # gemv mma keeps the BF16 tile


class DecodeFormatsTests(PluginTestCase):
    """decode_formats: each matrix's decode activations from its format at load (a named format's, else
    decode_activations'), the same bits as the uniform option gives that format, the facts saying which; bad pairs
    refused."""

    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("FP8 and NVFP4 matrices multiply on CUDA")
        import torch
        self.torch = torch
        self.g = torch.Generator().manual_seed(37)

    def matrices(self):
        torch, g = self.torch, self.g
        codes = torch.randint(0, 0x7E, (2048, 1024), generator=g, dtype=torch.uint8)
        codes |= torch.randint(0, 2, (2048, 1024), generator=g, dtype=torch.uint8) << 7
        fp8 = {"weight": codes.view(torch.float8_e4m3fn), "scale": torch.rand(2048, generator=g) / 100 + 1e-3}
        nvfp4 = {"weight": torch.randint(0, 256, (1024, 512), generator=g, dtype=torch.uint8),
                 "scale": (torch.rand(1024, 64, generator=g) * 2 + 0.01).to(torch.float8_e4m3fn),
                 "global_scale": torch.tensor(500.0), "input_scale": torch.tensor(300.0)}
        return fp8, nvfp4

    def plugin(self, **options):
        return self.assemble(options={"prompt_gemm": "pinned", **options}, context=OnCuda())["projections"]

    def test_each_format_takes_its_decode_activations_and_the_uniform_options_bits(self):
        torch = self.torch
        fp8, nvfp4 = self.matrices()
        x = torch.randn(16, 1024, generator=self.g).cuda().to(torch.bfloat16)
        split = self.plugin(decode_formats="nvfp4=prompt, fp8=bf16")
        f, n = split.load(fp8, "fp8:channel", role="out"), split.load(nvfp4, "nvfp4", role="down")
        self.assertEqual((f.decode, n.decode), ("bf16", "prompt"))
        facts = split.facts()
        self.assertEqual(facts["decode_activations"], {"fp8": "bf16", "nvfp4": "nvfp4"})
        self.assertEqual(facts["decode_kernels"]["out"]["activations"], ["bf16"])
        self.assertEqual(facts["decode_kernels"]["down"]["activations"], ["nvfp4"])
        for decode, matrix, stored, format in (("bf16", f, fp8, "fp8:channel"), ("prompt", n, nvfp4, "nvfp4")):
            uniform = self.plugin(decode_activations=decode)
            u = uniform.load(stored, format)
            self.assertEqual(u.path, matrix.path)
            for rows in (1, 4, 9, 16):
                self.assertTrue(torch.equal(split.matmul(x[:rows], matrix, torch.float32),
                                            uniform.matmul(x[:rows], u, torch.float32)), (format, rows))
        both = self.plugin(decode_activations="bf16", decode_formats="fp8=prompt")
        both.load(fp8, "fp8:channel")
        both.load(nvfp4, "nvfp4")
        self.assertEqual(both.facts()["decode_activations"], {"fp8": "fp8", "nvfp4": "bf16"})

    def test_decode_formats_refuses_what_it_does_not_name(self):
        for text in ("mxfp8=prompt", "fp8=fp8", "nvfp4", "fp8=bf16,fp8=prompt", "int8=bf16"):
            with self.assertRaises(ValueError, msg=text):
                self.plugin(decode_formats=text)


class CrossoverTests(PluginTestCase):
    """gemv crossover: a weight-only NVFP4 matrix's decode calls of up to the crossover the two kernels' sweep
    reports measure (the GEMV's tuned time no more than the tile's at every timed count up to it) on decode-gemv's
    CUDA GEMV with its bits, longer ones on mma-rows' tile with the tile's bits; the facts' row_invariance rows the
    crossover; without the reports, or on another device's, every call on the tile."""

    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("the GEMV and the tile run on CUDA")
        import importlib
        import torch
        from crit_plugins.assemble import load_module
        self.torch = torch
        self.matrices = load_module(self.registry()["quantized-projections"]).device_matrices
        self.nvfp4 = importlib.import_module(self.matrices.__package__ + ".device_nvfp4")
        if self.nvfp4.mma("cuda") is None or self.nvfp4.native("cuda") is None:
            self.skipTest("mma-rows and decode-gemv are not both here")
        self.g = torch.Generator().manual_seed(41)

    def reports(self, n, k, gemv, tile, device=None):
        """Fake sweep reports at (n, k): {rows: us} for decode-gemv's gemv-cuda (NVFP4, weights 0) and mma-rows'
        decode-tile."""
        name = device or self.torch.cuda.get_device_name(0)
        found = {"decode-gemv": ({"device": name}, {"structures": {"gemv-cuda": [
                     {"shape": {"weights": 0, "N": n, "K": k}, "rows": r, "tuned_us": us} for r, us in gemv.items()]}}),
                 "mma-rows": ({"device": name}, {"structures": {"decode-tile": [
                     {"shape": {"weights": 0, "N": n, "K": k}, "rows": r, "tuned_us": us} for r, us in tile.items()]}})}
        return lambda kernel: found.get(kernel)

    def test_calls_up_to_the_measured_crossover_take_the_gemv_and_longer_ones_the_tile(self):
        torch = self.torch
        n, k = 2048, 1024
        stored = {"weight": torch.randint(0, 256, (n, k // 2), generator=self.g, dtype=torch.uint8),
                  "scale": (torch.rand(n, k // 16, generator=self.g) * 2 + 0.01).to(torch.float8_e4m3fn),
                  "global_scale": torch.tensor(500.0)}
        options = {"activations": "bf16", "decode_activations": "bf16"}
        x = torch.randn(16, k, generator=self.g).cuda().to(torch.bfloat16)
        gemv = self.assemble(options={**options, "gemv": "native"}, context=OnCuda())["projections"]
        tile = self.assemble(options={**options, "gemv": "mma"}, context=OnCuda())["projections"]
        g, t = gemv.load(stored, "nvfp4"), tile.load(stored, "nvfp4")
        timed = self.reports(n, k, {1: 10.0, 2: 11.0, 4: 15.0, 8: 20.0, 16: 30.0},
                             {1: 11.0, 2: 11.0, 4: 12.0, 8: 13.0, 16: 15.0})
        with patch.object(self.matrices, "_report", timed):
            cross = self.assemble(options={**options, "gemv": "crossover"}, context=OnCuda())["projections"]
            m = cross.load(stored, "nvfp4", role="down")
        self.assertEqual((m.gemv_rows, m.path, m.invariant_rows), (2, "gemv_mma", 2))
        for rows in range(1, 17):
            want = gemv.matmul(x[:rows], g, torch.float32) if rows <= 2 else tile.matmul(x[:rows], t, torch.float32)
            self.assertTrue(torch.equal(cross.matmul(x[:rows], m, torch.float32), want), rows)
        facts = cross.facts()
        self.assertEqual((facts["row_invariance"]["rows"], facts["decode_crossover"]), (2, {"down": 2}))
        self.assertEqual(facts["decode_kernels"]["down"]["kernels"], ["gemv_mma"])
        for found in (lambda kernel: None, self.reports(n, k, {1: 12.0}, {1: 11.0}),
                      self.reports(n, k, {1: 10.0}, {1: 11.0}, device="another GPU")):
            with patch.object(self.matrices, "_report", found):
                m = self.assemble(options={**options, "gemv": "crossover"}, context=OnCuda())["projections"].load(
                    stored, "nvfp4")
            self.assertEqual((m.gemv_rows, m.path), (0, "mma"))
        with self.assertRaises(ValueError):
            self.assemble(options={**options, "gemv": "crossover", "prompt_gemm": "tile"}, context=OnCuda())


if __name__ == "__main__":
    unittest.main()
