"""quantized-projections on crit-labs/mma-rows' tile (gemv mma): every format's decode rows and short prompt rows
through one fixed tile, a row's bits the same in any call of up to the facts' rows, padded prompt buckets captured in
a graph equal to the unpadded chunk (prompts:projections), the prompt rows' arithmetic agreeing with the prompt path's;
and the prompt GEMMs' decode workspace a fixed, declared size, a chunk of matrix rows at a time (the factory makes it,
memory() declares it)."""
import unittest
from unittest.mock import patch

from crit_plugins.testing import PluginTestCase

SHAPES = ((10240, 5120), (5120, 17408))   # a wide projection and a down projection of a model of hidden size 5120
CASES = [("bf16", {}), ("fp8:channel", {}), ("fp8:channel", {"decode_activations": "bf16"}), ("mxfp8", {}),
         ("nvfp4", {"activations": "bf16"}), ("nvfp4", {}), ("nvfp4", {"decode_activations": "bf16"})]


class OnCuda:
    device = "cuda"


class TileTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("the tile runs on CUDA")
        import torch
        from crit_plugins.assemble import load_module
        self.torch = torch
        self.matrices = load_module(self.registry()["quantized-projections"]).device_matrices
        import importlib
        self.nvfp4 = importlib.import_module(self.matrices.__package__ + ".device_nvfp4")
        self.g = torch.Generator().manual_seed(17)
        # These tests are the BF16-MMA tile's: W4A4 decode rows off the FP4 MMA (tests/test_decode_kernels.py has it),
        # and FP8 weight-only decode rows on the tile (gemv mma) rather than on the dot GEMV.
        module = self.nvfp4.mma("cuda")
        if module is not None and hasattr(module, "fp4_mma"):
            patcher = patch.object(module, "fp4_mma", lambda device=None: False)
            patcher.start()
            self.addCleanup(patcher.stop)

    def assemble(self, use=None, options=None, **kwargs):
        return super().assemble(use=use, options={"gemv": "mma", **(options or {})}, **kwargs)

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

    def tile_here(self):
        if self.nvfp4.mma("cuda") is None:
            self.skipTest("the tile comes with crit-labs/mma-rows, not here")

    def test_every_format_keeps_a_row_s_bits_in_any_call_up_to_the_facts_rows_at_real_widths(self):
        torch = self.torch
        self.tile_here()
        for format, options in CASES:
            plugin = self.assemble(options={**options, "tile_rows": "128"}, context=OnCuda())["projections"]
            for n, k in SHAPES:
                with self.subTest(format=format, options=options, shape=(n, k)):
                    m = plugin.load(self.stored(format, n, k), format)
                    self.assertEqual(m.path, "mma")
                    most = plugin.facts()["row_invariance"]["rows"]
                    self.assertEqual(most, 16 if options.get("decode_activations") == "bf16" else 128)
                    x = torch.randn(most, k, generator=self.g).cuda().to(torch.bfloat16)
                    alone = torch.cat([plugin.matmul(x[r:r + 1].clone(), m, torch.float32) for r in range(most)])
                    for rows in (2, 3, 8, 15, 16, 17, 31, 64, 65, 100, most):
                        if rows <= most:
                            self.assertTrue(torch.equal(plugin.matmul(x[:rows], m, torch.float32), alone[:rows]),
                                            (format, options, rows))
        self.assertIn("prompts:projections", self.registry()["quantized-projections"]["provides"])

    def test_a_speed_configuration_s_decode_rows_keep_their_bits_at_every_offset_with_outlier_activations(self):
        """W4A4 matrices decoding 16-bit rows (decode_activations bf16) with pinned prompt GEMMs: a 27B's FP8 head
        (248,320 x 5,120, W8A16), a stacked NVFP4 gate/up with two global scales on blocked scales (W4A16) and a BF16
        96-row matrix; activations with outlier channels and per-row scales, as hidden states have. Every row of a
        call of k = 1..16 rows at every offset of a 24-row window, into FP32 and BF16, equals the row alone."""
        torch = self.torch
        self.tile_here()
        plugin = self.assemble(options={"decode_activations": "bf16", "prompt_gemm": "pinned"},
                               context=OnCuda())["projections"]
        gate_up = self.stored("nvfp4", 34816, 5120)
        gate_up["global_scale"] = torch.cat([torch.full((17408,), 6400.0), torch.full((17408,), 5800.0)])
        head = self.stored("fp8:channel", 248320, 5120)
        for format, stored in (("fp8:channel", head), ("nvfp4", gate_up), ("bf16", self.stored("bf16", 96, 5120))):
            m = plugin.load(stored, format)
            self.assertEqual(m.path, "mma")
            x = torch.randn(24, 5120, generator=self.g)
            x[:, torch.randperm(5120, generator=self.g)[:8]] *= 60.0
            x = (x * (torch.rand(24, 1, generator=self.g) * 3 + 0.2)).cuda().to(torch.bfloat16)
            for out in (torch.float32, None):
                alone = torch.cat([plugin.matmul(x[r:r + 1].clone(), m, out) for r in range(24)])
                for k in range(1, 17):
                    for first in range(0, 24 - k + 1):
                        if not torch.equal(plugin.matmul(x[first:first + k].clone(), m, out), alone[first:first + k]):
                            self.fail(f"{format}: rows {first}..{first + k - 1} of a {k}-row call differ from the "
                                      f"rows alone ({out or 'default'} out)")
            del m
        self.assertEqual(plugin.facts()["row_invariance"], {"rows": 16, "prompt_rows": True, "prompt_exact": False})

    def test_auto_tile_rows_follow_each_matrix_s_prompt_arithmetic_and_the_facts_take_the_fewest(self):
        self.tile_here()
        auto = self.matrices.AUTO_TILES
        expected = {("bf16", ()): auto["bf16"], ("fp8:channel", ()): auto["w8a8"], ("mxfp8", ()): auto["weight_only"],
                    ("nvfp4", (("activations", "bf16"),)): auto["weight_only"], ("nvfp4", ()): auto["w4a4"]}
        for (format, options), tile in expected.items():
            plugin = self.assemble(options=dict(options), context=OnCuda())["projections"]
            m = plugin.load(self.stored(format, 1024, 1024), format)
            self.assertEqual((m.tile_rows, plugin.facts()["row_invariance"]["rows"]), (tile, max(16, tile)),
                             (format, options))
        mixed = self.assemble(options={"activations": "bf16"}, context=OnCuda())["projections"]
        mixed.load(self.stored("nvfp4", 1024, 1024), "nvfp4")
        mixed.load(self.stored("fp8:channel", 1024, 1024), "fp8:channel")
        self.assertEqual(mixed.facts()["row_invariance"]["rows"], 16)   # the fewest: the FP8 matrix's decode rows

    def test_other_models_shapes_keep_rows_and_agree_with_the_prompt_path(self):
        """Shapes of a 33B NVFP4 dense model (hidden 7168) and of a hybrid MoE's dense projections (hidden 2560: FP8
        attention and shared-expert matrices down to 640 columns, BF16 mixers of 320 rows or 320 columns)."""
        torch = self.torch
        self.tile_here()
        shapes = (("nvfp4", {"activations": "bf16"}, 9216, 7168), ("nvfp4", {}, 7168, 22016),
                  ("fp8:channel", {}, 16384, 2560), ("fp8:channel", {}, 2560, 640), ("fp8:channel", {}, 1280, 2560),
                  ("bf16", {}, 320, 10240), ("bf16", {}, 10240, 320))
        for format, options, n, k in shapes:
            with self.subTest(format=format, shape=(n, k)):
                plugin = self.assemble(options={**options, "tile_rows": "128"}, context=OnCuda())["projections"]
                m = plugin.load(self.stored(format, n, k), format)
                self.assertEqual(m.path, "mma")
                x = torch.randn(200, k, generator=self.g).cuda().to(torch.bfloat16)
                alone = torch.cat([plugin.matmul(x[r:r + 1].clone(), m, torch.float32) for r in range(128)])
                for rows in (4, 16, 23, 128):
                    self.assertTrue(torch.equal(plugin.matmul(x[:rows], m, torch.float32), alone[:rows]), rows)
                prompt = plugin.matmul(x, m, torch.float32)
                self.assertLess(float((prompt[:128] - alone).abs().max() / prompt.abs().max()), 2e-5)

    def test_tile_rows_agree_with_the_prompt_path_in_order_only_and_beyond_them_the_prompt_path_serves(self):
        torch = self.torch
        self.tile_here()
        for format, options in CASES:
            plugin = self.assemble(options={**options, "tile_rows": "128"}, context=OnCuda())["projections"]
            n, k = SHAPES[0]
            m = plugin.load(self.stored(format, n, k), format)
            x = torch.randn(200, k, generator=self.g).cuda().to(torch.bfloat16)
            prompt = plugin.matmul(x, m, torch.float32)          # 200 rows: past the tile, the prompt path
            tiles = torch.cat([plugin.matmul(x[r:r + 100], m, torch.float32) for r in (0, 100)])
            gap = float((prompt - tiles).abs().max() / prompt.abs().max())
            self.assertLess(gap, 2e-5, (format, options))

    def test_a_padded_bucket_replays_the_unpadded_chunk_s_bits(self):
        """prompts:projections through crit_plugins.prompt_graphs: a chunk in a captured bucket of 16, 64 or 128 rows
        (the rest padding) gives exactly the unpadded chunk's rows."""
        torch = self.torch
        self.tile_here()
        from crit_plugins.prompt_graphs import PromptGraphs
        for format, options in (("nvfp4", {"activations": "bf16"}), ("fp8:channel", {}), ("nvfp4", {})):
            plugin = self.assemble(options={**options, "tile_rows": "128"}, context=OnCuda())["projections"]
            n, k = SHAPES[1]
            m = plugin.load(self.stored(format, n, k), format)
            table = torch.randn(97, k, generator=self.g).cuda().to(torch.bfloat16)

            def forward(batch):
                return plugin.matmul(table.index_select(0, batch.tokens_t), m, torch.float32)
            prompts = PromptGraphs(forward, device="cuda", buckets=(16, 64, 128))
            self.assertEqual(prompts.prepare(), [16, 64, 128])
            for count in (1, 5, 16, 23, 64, 100, 128):
                tokens = torch.randint(0, 97, (count,), generator=self.g).tolist()
                out = prompts.run(0, 0, tokens)
                self.assertIsNotNone(out, count)
                eager = plugin.matmul(table[tokens], m, torch.float32)
                self.assertTrue(torch.equal(out[:count], eager), (format, options, count))

    def test_fp8_prompt_rows_with_16_bit_activations_multiply_the_exact_weights(self):
        """fp8_activations bf16 (W8A16): the E4M3 weights widened exactly to BF16, FP32 sums, the row scale after, on
        the tile and past it; decode rows follow; the facts give the arithmetic class."""
        torch = self.torch
        n, k = 2048, 1024
        stored = self.stored("fp8:channel", n, k)
        exact_w = stored["weight"].double() * stored["scale"].double()[:, None]
        x = torch.randn(300, k, generator=self.g).cuda().to(torch.bfloat16)
        exact = x.double() @ exact_w.cuda().t()
        for options in ({"fp8_activations": "bf16"}, {"fp8_activations": "bf16", "prompt_gemm": "fixed_tiles"}):
            plugin = self.assemble(options={**options, "tile_rows": "128"}, context=OnCuda())["projections"]
            m = plugin.load(stored, "fp8:channel")
            for rows in (1, 16, 100, 300):
                found = plugin.matmul(x[:rows], m, torch.float32).double()
                self.assertLess(float((found - exact[:rows]).abs().max() / exact[:rows].abs().max()), 1e-5, rows)
            facts = plugin.facts()
            self.assertEqual(facts["prompt_activations"], {"fp8": "bf16"})
            self.assertEqual(facts["decode_activations"], {"fp8": "bf16"})
            self.assertEqual(facts["prompt_arithmetic"], {"fp8": {"weights": "exact", "activations": "bf16",
                                                                   "sums": "fp32"}})
            if options.get("prompt_gemm") == "fixed_tiles" and self.nvfp4.tiles_kernel() is not None:
                whole = plugin.matmul(x, m)
                self.assertTrue(torch.equal(plugin.matmul(x[50:250].clone(), m), whole[50:250]))
        w8a8 = self.assemble(context=OnCuda())["projections"]
        w8a8.load(stored, "fp8:channel")
        self.assertEqual(w8a8.facts()["prompt_arithmetic"]["fp8"]["activations"], "fp8")

    def test_fused_prompt_quantizers_and_epilogue_scales_give_the_fp32_path_s_values(self):
        """Prompt rows past the tile: FP8 through cuBLASLt with the row and column scales in its epilogue, W4A4 one
        block-scaled GEMM a run of rows sharing a global scale (stacked matrices, runs on and off 128-row tiles),
        BF16 out; within BF16 rounding of the FP32 path (the scales applied after the sums)."""
        torch = self.torch
        plugin = self.assemble(context=OnCuda())["projections"]
        x = torch.randn(300, 1024, generator=self.g).cuda().to(torch.bfloat16)
        for split in (4096, 4112):
            stored = self.stored("nvfp4", 10240, 1024)
            stored["global_scale"] = torch.cat([torch.full((split,), 500.0), torch.full((10240 - split,), 300.0)])
            m = plugin.load(stored, "nvfp4")
            self.assertEqual([(a, b) for a, b, _ in m.runs], [(0, split), (split, 10240)])
            full, half = plugin.matmul(x, m, torch.float32), plugin.matmul(x, m)
            self.assertEqual(half.dtype, torch.bfloat16)
            self.assertTrue(bool(((half.float() - full).abs() <= full.abs() / 128 + 1e-3 * full.abs().max()).all()),
                            split)
        m = plugin.load(self.stored("fp8:channel", 2048, 1024), "fp8:channel")
        full, half = plugin.matmul(x, m, torch.float32), plugin.matmul(x, m)
        self.assertTrue(bool(((half.float() - full).abs() <= full.abs() / 128 + 1e-3 * full.abs().max()).all()))

    def test_a_weight_only_matrix_s_prompt_decoder_is_compiled_at_load(self):
        """The Triton decoder its prompt rows past the tile use runs once at load: a first prompt there pays no
        compile, and a captured prompt chunk can hold it."""
        torch = self.torch
        if self.nvfp4.rows_kernel() is None:
            self.skipTest("the Triton decoder comes with crit-labs/decode-gemv, not here")
        for prompt, dtype in (("bf16", torch.bfloat16), ("fp16_acc", torch.float16)):
            self.nvfp4._WARM.discard(("decode", dtype))
            plugin = self.assemble(options={"activations": "bf16", "prompt": prompt}, context=OnCuda())["projections"]
            plugin.load(self.stored("nvfp4", 512, 1024), "nvfp4")
            self.assertIn(("decode", dtype), self.nvfp4._WARM, prompt)

    def test_after_load_no_request_compiles_a_triton_kernel(self):
        """Every path's Triton kernels (prompt decoders, fixed-tile prompt GEMM, Triton GEMV) are compiled at load for
        the specializations requests meet: decode and prompt calls of many row counts, into BF16 and FP32, compile
        nothing (Triton's JIT cache hook, which runs on every compile)."""
        torch = self.torch
        if self.nvfp4.rows_kernel() is None:
            self.skipTest("the Triton kernels come with crit-labs/decode-gemv and fixed-tile-gemm, not here")
        import triton
        compiled = []

        def hook(*args, **kwargs):
            compiled.append(kwargs.get("fn") or args)
            return False
        cases = [("nvfp4", {"activations": "bf16"}), ("nvfp4", {"activations": "bf16", "prompt": "fp16_acc"}),
                 ("nvfp4", {"activations": "bf16", "prompt": "fp8"}), ("nvfp4", {"activations": "bf16", "gemv": "triton"}),
                 ("bf16", {"prompt_gemm": "fixed_tiles"}), ("mxfp8", {"prompt_gemm": "fixed_tiles"})]
        if self.nvfp4.tiles_kernel() is not None:
            cases.append(("nvfp4", {"activations": "bf16", "prompt_gemm": "fixed_tiles"}))
        before = triton.knobs.runtime.jit_cache_hook
        try:
            for format, options in cases:
                plugin = self.assemble(options=options, context=OnCuda())["projections"]
                m = plugin.load(self.stored(format, 2048, 1024), format)
                triton.knobs.runtime.jit_cache_hook = hook
                for rows in (1, 3, 8, 16, 23, 64, 100, 213, 300, 1500, 3000, 3001):
                    x = torch.randn(rows, 1024, generator=self.g).cuda().to(torch.bfloat16)
                    for out in (None, torch.float32):
                        plugin.matmul(x, m, out)
                torch.cuda.synchronize()
                triton.knobs.runtime.jit_cache_hook = before
                self.assertEqual(compiled, [], (format, options))
        finally:
            triton.knobs.runtime.jit_cache_hook = before

    def test_a_call_looks_up_no_registry_kernel(self):
        """The registry modules a matrix's calls use are held from load: a call asks crit_plugins.kernels for none."""
        torch = self.torch
        from unittest.mock import patch
        cases = [("nvfp4", {"activations": "bf16"}), ("nvfp4", {}), ("fp8:channel", {}), ("mxfp8", {}), ("bf16", {}),
                 ("nvfp4", {"activations": "bf16", "gemv": "native", "prompt_gemm": "fixed_tiles"})]
        for format, options in cases:
            plugin = self.assemble(options=options, context=OnCuda())["projections"]
            m = plugin.load(self.stored(format, 2048, 1024), format)
            asked = []
            original = self.nvfp4.kernel
            with patch.object(self.nvfp4, "kernel", lambda name: asked.append(name) or original(name)):
                for rows in (1, 7, 16, 40, 300):
                    plugin.matmul(torch.randn(rows, 1024, generator=self.g).cuda().to(torch.bfloat16), m)
            self.assertEqual(asked, [], (format, options))

    def test_the_decode_workspace_is_the_declared_buffers_and_prompt_gemms_decode_a_chunk_at_a_time(self):
        torch = self.torch
        from crit_plugins.assemble import load_module
        module = load_module(self.registry()["quantized-projections"])
        declared = module.memory(OnCuda(), {})
        self.assertEqual(declared, {"device": self.matrices.BUFFER_BYTES})
        self.assertEqual(module.memory(type("Host", (), {"device": "cpu"})(), {}), {})
        before = torch.cuda.memory_allocated()
        plugin = self.assemble(options={"activations": "bf16", "gemv": "native"}, context=OnCuda())["projections"]
        self.assertLessEqual(torch.cuda.memory_allocated() - before, declared["device"])
        self.assertIn(self.matrices.device_key("cuda"), self.matrices._BUFFERS)
        # A weight-only NVFP4 matrix of 160 MiB of BF16 values: its load takes its own tensors, no workspace its size.
        n, k = 16384, 5120
        stored = self.stored("nvfp4", n, k)
        before = torch.cuda.memory_allocated()
        m = plugin.load(stored, "nvfp4", encode=None)
        grown = torch.cuda.memory_allocated() - before
        own = sum(t.numel() * t.element_size() for t in m.tensors.values() if hasattr(t, "numel"))
        self.assertLess(grown, own + (8 << 20))
        self.assertEqual(len(self.matrices.row_chunks(m)), 3)   # chunks of 64 MiB of BF16 values (6528 rows)
        # Prompt rows past the tile decode a chunk of rows at a time: the stored values in FP32 sums.
        x = torch.randn(300, k, generator=self.g).cuda().to(torch.bfloat16)
        exact = x.double() @ self.matrices.dense(m).double().t()
        found = plugin.matmul(x, m, torch.float32).double()
        self.assertLess(float((found - exact).abs().max() / exact.abs().max()), 1e-5)
        # Fixed tiles: chunking the matrix's rows changes no bits (the fallback, a torch GEMM a row, picks its kernel by
        # the chunk's rows too).
        if self.nvfp4.tiles_kernel() is None:
            return
        fixed = self.assemble(options={"activations": "bf16", "prompt_gemm": "fixed_tiles"}, context=OnCuda())
        f = fixed["projections"].load(self.stored("nvfp4", 4096, 1024), "nvfp4")
        x = torch.randn(300, 1024, generator=self.g).cuda().to(torch.bfloat16)
        whole = fixed["projections"].matmul(x, f, torch.float32)
        with patch.object(self.matrices, "WORKSPACE_BYTES", 1 << 20):   # 512 rows a chunk
            self.assertEqual(len(self.matrices.row_chunks(f)), 8)
            self.assertTrue(torch.equal(fixed["projections"].matmul(x, f, torch.float32), whole))


if __name__ == "__main__":
    unittest.main()
