"""quantized-projections with prompt_gemm pinned: every format's prompt rows through crit's GEMMs in one
configuration per matrix shape (fp8-block-rows' and int-block-experts' gemm-native launched as for PIN_ROWS tokens,
fixed-tile-gemm for 16-bit operands), so a prompt row's bits are the same at every prompt length, in any chunk and in
a padded bucket captured in a graph; the facts say so (row_invariance prompt_rows); the values agree with the
provider's path in the order of their sums only; without the kernels, one row a call keeps the bits. The same tests
run prompt_gemm pinned_cublaslt (cuBLASLt with one algorithm per GEMM shape and format, pinned-gemm: the yardstick)."""
import unittest
from unittest.mock import patch

from crit_plugins.testing import PluginTestCase

# (format, options, N, K, runs): runs, for NVFP4 W4A4, the rows of each global scale of a stacked matrix (q/k/v,
# gate/up); the widths of a model of hidden size 5120 cut to keep 4096 calls short.
CASES = [("nvfp4", {}, 2560, 1024, (1024, 1024, 512)), ("nvfp4", {}, 1536, 2048, None),
         ("fp8:channel", {}, 2048, 1024, None), ("fp8:channel", {"fp8_activations": "bf16"}, 2048, 1024, None),
         ("nvfp4", {"activations": "bf16"}, 2048, 1024, None), ("bf16", {}, 1024, 1024, None),
         ("mxfp8", {}, 1024, 1024, None)]


class OnCuda:
    device = "cuda"


class PinnedTests(PluginTestCase):
    GEMM = "pinned"

    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("pinned prompt GEMMs run on CUDA")
        import importlib
        import torch
        from crit_plugins.assemble import load_module
        self.torch = torch
        self.matrices = load_module(self.registry()["quantized-projections"]).device_matrices
        self.nvfp4 = importlib.import_module(self.matrices.__package__ + ".device_nvfp4")
        self.g = torch.Generator().manual_seed(23)

    def kernel_here(self):
        if self.GEMM == "pinned_cublaslt":
            if self.nvfp4.pinned("cuda") is None:
                self.skipTest("pinned_cublaslt prompt GEMMs come with pinned-gemm, not here")
            return
        if None in (self.nvfp4.fp8_block("cuda"), self.nvfp4.int_block("cuda"), self.nvfp4.tiles_kernel()):
            self.skipTest("pinned prompt GEMMs come with fp8-block-rows, int-block-experts and fixed-tile-gemm")

    def on_kernel(self, m):
        """The matrix's prompt rows go through the pinned GEMMs, not one row a call."""
        if self.GEMM == "pinned_cublaslt":
            return m.pinned is not None or m.format == "mxfp8" or m.activations == "bf16"
        return m.prompt_kernel.endswith(" pinned") or m.prompt_kernel == "fixed-tile-gemm"

    def stored(self, format, n, k, runs=None):
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
        found = {"weight": torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8),
                 "scale": (torch.rand(n, k // 16, generator=g) * 2 + 0.01).to(torch.float8_e4m3fn),
                 "global_scale": torch.tensor(500.0), "input_scale": torch.tensor(300.0)}
        if runs:
            found["global_scale"] = torch.cat([torch.full((r,), 400.0 + 50 * i) for i, r in enumerate(runs)])
        return found

    def loaded(self, format, options, n, k, runs=None):
        plugin = self.assemble(options={**options, "prompt_gemm": self.GEMM}, context=OnCuda())["projections"]
        return plugin, plugin.load(self.stored(format, n, k, runs), format)

    def test_a_prompt_row_s_bits_are_the_same_at_every_row_count_past_the_decode_rows(self):
        """Every row count from 17 to 4096 (calls of up to 16 rows are decode rows, each row's bits its own), into
        BF16 and FP32: the first rows of a call equal the first rows of the 4096-row call, and so do rows at other
        offsets (a chunk past the first, another sequence's rows in a shared pass)."""
        torch = self.torch
        self.kernel_here()
        for format, options, n, k, runs in CASES:
            plugin, m = self.loaded(format, options, n, k, runs)
            self.assertTrue(self.on_kernel(m), (format, options, m.prompt_kernel))
            if self.GEMM == "pinned" and m.native_gemm is not None:   # the default-knob build, in the facts' record
                from crit_plugins import kernels
                name = {"nvfp4": "int-block-experts"}.get(m.format, "fp8-block-rows") + kernels.UNTUNED
                self.assertEqual(kernels.loaded()[name]["source"], "built", name)
                self.assertIn("quantized-projections", kernels.loaded()[name]["plugins"])
            self.assertTrue(plugin.facts()["row_invariance"]["prompt_rows"], (format, options))
            x = torch.randn(4096, k, generator=self.g).cuda().to(torch.bfloat16)
            for out in (None, torch.float32):
                with self.subTest(format=format, options=options, out=out):
                    whole = plugin.matmul(x, m, out)
                    rows = range(17, 4097) if out is None else (17, 63, 64, 65, 127, 128, 129, 511, 513, 1025, 2049)
                    for count in rows:
                        if not torch.equal(plugin.matmul(x[:count], m, out), whole[:count]):
                            self.fail(f"{format} {options}: the first {count} rows differ from the 4096-row call's")
                    for first, count in ((17, 40), (1025, 1023), (2048, 2048), (3000, 583)):
                        self.assertTrue(torch.equal(plugin.matmul(x[first:first + count].clone(), m, out),
                                                    whole[first:first + count]), (first, count))

    def test_a_padded_bucket_replays_the_rows_of_a_longer_prompt(self):
        """prompts:projections through crit_plugins.prompt_graphs: a chunk in a captured bucket of 64, 128 or 512 rows
        (the rest padding) gives exactly the rows the same tokens get at the start of a 583-row prompt."""
        torch = self.torch
        self.kernel_here()
        from crit_plugins.prompt_graphs import PromptGraphs
        for format, options, n, k, runs in CASES[:3] + CASES[4:5]:
            plugin, m = self.loaded(format, options, n, k, runs)
            table = torch.randn(97, k, generator=self.g).cuda().to(torch.bfloat16)

            def forward(batch):
                return plugin.matmul(table.index_select(0, batch.tokens_t), m, torch.float32)
            prompts = PromptGraphs(forward, device="cuda", buckets=(64, 128, 512))
            self.assertEqual(prompts.prepare(), [64, 128, 512])
            tokens = torch.randint(0, 97, (583,), generator=self.g).tolist()
            whole = plugin.matmul(table[tokens], m, torch.float32)
            for count in (17, 23, 64, 65, 100, 128, 129, 300, 512):
                out = prompts.run(0, 0, tokens[:count])
                self.assertIsNotNone(out, count)
                self.assertTrue(torch.equal(out[:count], whole[:count]), (format, options, count))

    def test_pinned_rows_agree_with_the_provider_s_in_the_order_of_their_sums(self):
        torch = self.torch
        self.kernel_here()
        for format, options, n, k, runs in CASES:
            with self.subTest(format=format, options=options):
                stored = self.stored(format, n, k, runs)
                plugin = self.assemble(options={**options, "prompt_gemm": self.GEMM}, context=OnCuda())["projections"]
                provider = self.assemble(options={**options, "prompt_gemm": "provider"}, context=OnCuda())["projections"]
                m, p = plugin.load(stored, format), provider.load(stored, format)
                x = torch.randn(700, k, generator=self.g).cuda().to(torch.bfloat16)
                pinned, theirs = plugin.matmul(x, m, torch.float32), provider.matmul(x, p, torch.float32)
                self.assertLess(float((pinned - theirs).abs().max() / theirs.abs().max()), 2e-5)
                bf16 = plugin.matmul(x, m)
                self.assertEqual(bf16.dtype, torch.bfloat16)
                self.assertLess(float((bf16.float() - theirs).abs().max() / theirs.abs().max()), 1e-2)

    def test_other_models_shapes_keep_prompt_rows(self):
        """Shapes of a 27B hybrid (hidden 5120: W4A4 gate/up runs and down, FP8 attention), a 33B NVFP4 dense model
        (hidden 7168) and a hybrid MoE's dense projections (hidden 2560: FP8 down to 640 columns, BF16 mixers of 320
        rows), at the row counts a prompt meets at chunk and bucket edges."""
        torch = self.torch
        self.kernel_here()
        shapes = (("nvfp4", {}, 34816, 5120, (17408, 17408)), ("nvfp4", {}, 5120, 17408, None),
                  ("fp8:channel", {}, 16384, 5120, None), ("fp8:channel", {}, 5120, 6144, None),
                  ("nvfp4", {}, 44032, 7168, (22016, 22016)), ("nvfp4", {"activations": "bf16"}, 7168, 22016, None),
                  ("fp8:channel", {}, 2560, 640, None), ("fp8:channel", {}, 1280, 2560, None),
                  ("bf16", {}, 320, 10240, None), ("bf16", {}, 10240, 320, None))
        for format, options, n, k, runs in shapes:
            with self.subTest(format=format, options=options, shape=(n, k)):
                plugin, m = self.loaded(format, options, n, k, runs)
                x = torch.randn(4096, k, generator=self.g).cuda().to(torch.bfloat16)
                whole = plugin.matmul(x, m)
                for count in (17, 23, 64, 65, 129, 513, 1025, 2048, 2049):
                    self.assertTrue(torch.equal(plugin.matmul(x[:count], m), whole[:count]), count)
                self.assertTrue(torch.equal(plugin.matmul(x[1025:3073].clone(), m), whole[1025:3073]))
                del whole, x, m

    def test_without_the_kernel_one_row_a_call_keeps_the_bits(self):
        torch = self.torch
        import contextlib
        names = ("untuned", "tiles_kernel") if self.GEMM == "pinned" else ("pinned",)
        with contextlib.ExitStack() as stack:
            for name in names:
                stack.enter_context(patch.object(self.nvfp4, name, lambda *a: None))
            for format, options, n, k, runs in (CASES[0], CASES[2], CASES[4], CASES[5]):
                with self.subTest(format=format, options=options):
                    plugin, m = self.loaded(format, options, 256, k, runs and (128, 128))
                    self.assertIsNone(m.pinned)
                    self.assertIsNone(m.native_gemm)
                    self.assertTrue(plugin.facts()["row_invariance"]["prompt_rows"])
                    x = torch.randn(60, k, generator=self.g).cuda().to(torch.bfloat16)
                    whole = plugin.matmul(x, m)
                    for count in (17, 33, 59):
                        self.assertTrue(torch.equal(plugin.matmul(x[:count], m), whole[:count]), count)

    def test_the_contract_s_prompt_row_check_passes_and_other_prompts_are_refused(self):
        self.kernel_here()
        plugin = self.assemble(options={"prompt_gemm": self.GEMM}, context=OnCuda())["projections"]
        self.assertKeepsRows(plugin)
        self.assertTrue(plugin.facts()["row_invariance"]["prompt_rows"])
        for prompt in ("fp16_acc", "fp8"):
            with self.assertRaises(ValueError):
                self.assemble(options={"activations": "bf16", "prompt": prompt, "prompt_gemm": self.GEMM},
                              context=OnCuda())
        provider = self.assemble(options={"prompt_gemm": "provider"}, context=OnCuda())["projections"]
        provider.load(self.stored("nvfp4", 256, 256), "nvfp4")
        self.assertFalse(provider.facts()["row_invariance"]["prompt_rows"])

    def test_the_pinned_workspace_is_declared(self):
        plugin = self.registry()["quantized-projections"]
        from crit_plugins.assemble import load_module
        memory = load_module(plugin).memory(OnCuda(), {})
        self.assertEqual(memory, {"device": self.matrices.BUFFER_BYTES})
        self.assertGreaterEqual(self.matrices.BUFFER_BYTES, self.matrices.GEMM_BYTES + self.matrices.WORKSPACE_BYTES)


class CublasltPinnedTests(PinnedTests):
    GEMM = "pinned_cublaslt"

    def test_the_pinned_workspace_is_declared(self):
        self.skipTest("the same declaration: PinnedTests checks it")


if __name__ == "__main__":
    unittest.main()
