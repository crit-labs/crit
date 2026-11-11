"""quantized-projections with prompt_gemm tile: every format's prompt rows through mma-rows' prompt tile with
the decode rows' activations and plan, so a prompt row's output has the bits of the same row decoded alone (a scored
token's log-probability is the served one's: preserve mode's exact honesty), in any chunk, into every output type, at
real widths; the facts say prompt_exact; a selection that cannot keep it says so or is refused."""
import unittest
from unittest.mock import patch

from crit_plugins.testing import PluginTestCase

SHAPES = ((10240, 5120), (5120, 17408))   # a wide projection and a down projection of a model of hidden size 5120
CASES = [("bf16", {}), ("fp8:channel", {}), ("fp8:channel", {"decode_activations": "bf16"}),
         ("fp8:channel", {"fp8_activations": "bf16"}), ("mxfp8", {}), ("nvfp4", {"activations": "bf16"}),
         ("nvfp4", {}), ("nvfp4", {"decode_activations": "bf16"}), ("nvfp4", {"decode_formats": "nvfp4=bf16"}),
         ("fp8:channel", {"decode_formats": "fp8=bf16"})]
TILE = {"prompt_gemm": "tile", "gemv": "mma"}   # every decode row on the tile (gemv auto takes other kernels)


class OnCuda:
    device = "cuda"


class ExactPromptTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("the prompt tile runs on CUDA")
        import importlib
        import torch
        from crit_plugins.assemble import load_module
        self.torch = torch
        module = load_module(self.registry()["quantized-projections"])
        self.nvfp4 = importlib.import_module(module.device_matrices.__package__ + ".device_nvfp4")
        mma = self.nvfp4.mma("cuda")
        if mma is None or getattr(mma, "mm_prompt", None) is None:
            self.skipTest("the prompt tile comes with mma-rows' prompt tile, not here")
        self.g = torch.Generator().manual_seed(23)

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

    def test_every_format_s_prompt_rows_have_the_decode_rows_bits_at_real_widths(self):
        torch = self.torch
        for format, options in CASES:
            plugin = self.assemble(options={**options, **TILE}, context=OnCuda())["projections"]
            for n, k in SHAPES:
                with self.subTest(format=format, options=options, shape=(n, k)):
                    m = plugin.load(self.stored(format, n, k), format)
                    self.assertEqual((m.path, m.gemm), ("mma", "tile"))
                    x = torch.randn(300, k, generator=self.g).cuda().to(torch.bfloat16)
                    for out in (torch.float32, None):
                        alone = torch.cat([plugin.matmul(x[r:r + 1].clone(), m, out) for r in range(300)])
                        for rows in (17, 64, 65, 129, 300):
                            self.assertTrue(torch.equal(plugin.matmul(x[:rows], m, out), alone[:rows]),
                                            (format, options, (n, k), rows, out))
                        decode = torch.cat([plugin.matmul(x[r:r + 16], m, out) for r in range(0, 288, 16)])
                        self.assertTrue(torch.equal(decode, alone[:288]), "decode calls of 16 rows")
                    del m
            facts = plugin.facts()
            self.assertEqual(facts["row_invariance"]["prompt_exact"], True, (format, options))
            self.assertEqual(facts["row_invariance"]["prompt_rows"], True)
            arithmetic = facts["prompt_arithmetic"][format.split(":")[0]]
            self.assertEqual(arithmetic["activations"], facts["decode_activations"][format.split(":")[0]]
                             if format != "bf16" else "bf16", (format, options, facts))

    def test_a_long_prompt_chunk_keeps_every_row_s_decode_bits(self):
        torch = self.torch
        plugin = self.assemble(options={"activations": "bf16", **TILE}, context=OnCuda())["projections"]
        m = plugin.load(self.stored("nvfp4", 5120, 17408), "nvfp4")
        x = torch.randn(2048, 17408, generator=self.g).cuda().to(torch.bfloat16)
        found = plugin.matmul(x, m, torch.float32)
        alone = torch.cat([plugin.matmul(x[r:r + 1].clone(), m, torch.float32) for r in range(2048)])
        self.assertTrue(torch.equal(found, alone))

    def test_without_the_prompt_tile_the_facts_say_prompt_rows_are_not_exact(self):
        torch = self.torch

        class Without:   # the module of an earlier mma-rows: the decode tile, no prompt tile
            def __init__(self, module):
                self.module = module

            def __getattr__(self, name):
                if name == "mm_prompt":
                    raise AttributeError(name)
                return getattr(self.module, name)
        real = self.nvfp4.mma("cuda")
        with patch.object(self.nvfp4, "mma", lambda device: Without(real)):
            plugin = self.assemble(options=TILE, context=OnCuda())["projections"]
            m = plugin.load(self.stored("fp8:channel", 1024, 2048), "fp8:channel")
        self.assertEqual((m.path, m.gemm), ("mma", "provider"))
        self.assertEqual(plugin.facts()["row_invariance"]["prompt_exact"], False)
        x = torch.randn(40, 2048, generator=self.g).cuda().to(torch.bfloat16)
        self.assertEqual(tuple(plugin.matmul(x, m).shape), (40, 1024))

    def test_the_projections_contract_checks_the_claim(self):
        plugin = self.assemble(options=TILE, context=OnCuda())["projections"]
        self.assertKeepsRows(plugin)
        self.assertTrue(plugin.facts()["row_invariance"]["prompt_exact"])

    def test_a_selection_that_cannot_keep_the_decode_bits_is_refused(self):
        for gemv in ("auto", "native"):   # auto: W4A4 and FP8 W8A16 decode rows off the tile
            with self.assertRaisesRegex(ValueError, "gemv mma"):
                self.assemble(options={"prompt_gemm": "tile", "gemv": gemv}, context=OnCuda())
        with self.assertRaisesRegex(ValueError, "in bf16"):
            self.assemble(options={**TILE, "activations": "bf16", "prompt": "fp16_acc"},
                          context=OnCuda())


if __name__ == "__main__":
    unittest.main()
