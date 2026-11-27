import random

from crit_plugins.testing import PluginTestCase


def matrix(rows, cols, seed=1):
    rng = random.Random(seed)
    return [[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]


class QuantizedProjectionsTests(PluginTestCase):
    # each format's options and how far its decoded weights may lie from the dense ones
    FORMATS = [({}, 0.1), ({"format": "int4"}, 0.1), ({"format": "int8", "group": 16}, 0.1),
               ({"format": "nvfp4"}, 0.2), ({"format": "nvfp4", "four_over_six": True}, 0.2)]

    def test_it_has_the_projections_interface(self):
        for options, _ in self.FORMATS:
            self.assertImplements(self.assemble(options=options))

    def test_it_projects_with_its_decoded_weights_as_the_dense_projection_does(self):
        dense = self.reference()["projections"]
        x = matrix(3, 64, seed=2)
        for options, _ in self.FORMATS:
            p = self.assemble(options=options)["projections"]
            w = p.prepare(matrix(5, 64))
            got, want = p.project(w, x), dense.project(dense.prepare(p.dense(w)), x)
            self.assertClose(got, want, atol=1e-9, rtol=1e-9)

    def test_its_decoded_weights_stay_near_the_dense_ones(self):
        for options, bound in self.FORMATS:
            p = self.assemble(options=options)["projections"]
            original = matrix(4, 64, seed=3)
            decoded = p.dense(p.prepare(original))
            error = max(abs(a - b) for ra, rb in zip(decoded, original) for a, b in zip(ra, rb))
            self.assertLess(error, bound, options)

    def test_prompt_gemm_auto_follows_the_level_the_engine_asks_of_its_prompt_rows(self):
        from crit_plugins import kernels
        self.addCleanup(kernels.use_exactness, None)
        for asked, gemm, gemv in ((None, "auto", "auto"), ("repeatable", "auto", "auto"),
                                  ("row-exact", "pinned", "auto"), ("exact", "tile", "fp4"),
                                  ("declared", "tile", "fp4")):
            kernels.use_exactness(asked and {"decode": "row-exact", "prompt": asked})
            p = self.assemble()["projections"]
            self.assertEqual((p.gemm, p.gemv), (gemm, gemv), asked)
            self.assertEqual(p.exactness(), {})   # no matrix loaded: it states no level yet
        kernels.use_exactness("exact")
        # An option set to one way is that way (the loader then compares what it keeps with what is asked); a decode
        # kernel named with it is kept where the prompt tile takes it and refused, naming the level, where it cannot.
        self.assertEqual(self.assemble(options={"prompt_gemm": "native"})["projections"].gemm, "native")
        p = self.assemble(options={"gemv": "mma"})["projections"]
        self.assertEqual((p.gemm, p.gemv), ("tile", "mma"))
        with self.assertRaisesRegex(ValueError, "prompt_gemm tile \\(auto, for the exact prompt rows the "
                                                "configuration's exactness asks\\) gives prompt rows the decode "
                                                "tile's bits: gemv mma or fp4, not triton"):
            self.assemble(options={"gemv": "triton"})
        kernels.use_exactness({"decode": "row-exact", "prompt": "row-exact"})
        with self.assertRaisesRegex(ValueError, "prompt_gemm pinned \\(auto, for the row-exact prompt rows the "
                                                "configuration's exactness asks\\) runs weight-only NVFP4 prompts "
                                                "in bf16, not fp8"):
            self.assemble(options={"activations": "bf16", "prompt": "fp8"})

    def test_head_auto_follows_the_quality_mode_the_engine_states(self):
        from crit_plugins import kernels
        self.addCleanup(kernels.use_quality_mode, None)
        for mode, option, head, by in ((None, "auto", "stored", None), ("preserve", "auto", "stored", None),
                                       ("speed", "auto", "fp8", "speed"), ("speed", "stored", "stored", None),
                                       ("preserve", "fp8", "fp8", None), (None, "fp8", "fp8", None)):
            kernels.use_quality_mode(mode)
            p = self.assemble(options={"head": option})["projections"]
            self.assertEqual((p.head, p.head_by), (head, by), (mode, option))
        kernels.use_quality_mode("speed")   # auto is the default
        self.assertEqual(self.assemble()["projections"].head, "fp8")
        with self.assertRaisesRegex(ValueError, "quality mode 'fast' is not one of preserve, speed"):
            kernels.use_quality_mode("fast")

    def test_resolve_prices_the_checkpoints_16_bit_matrices_as_load_holds_them(self):
        # crit_plugins.assemble.PRICES: the stored and held bytes of every 16-bit target matrix (not a routed expert,
        # an embedding table or an expert tower), the head by its role, read from the headers alone
        import sys
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from crit_plugins import kernels
        torch = __import__("torch")
        from safetensors.torch import save_file
        self.addCleanup(kernels.use_quality_mode, None)
        module = sys.modules[type(self.assemble()["projections"]).__module__]
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(folder, ignore_errors=True))
        bf16 = lambda r, c: torch.zeros(r, c, dtype=torch.bfloat16)   # noqa: E731
        save_file({"model.layers.0.self_attn.q_proj.weight": bf16(32, 96),
                   "model.layers.0.mlp.experts.0.down_proj.weight": bf16(32, 96),   # the experts slot's
                   "model.embed_tokens.weight": bf16(40, 96),                        # a lookup table
                   "model.norm.weight": torch.zeros(96, dtype=torch.bfloat16),       # not a matrix
                   "lm_head.weight": bf16(40, 96)}, str(folder / "model.safetensors"))
        (folder / "config.json").write_text('{"model_type": "llama", "hidden_size": 96}', encoding="utf-8")
        context = SimpleNamespace(model=folder)
        options = {"head": "auto", "trunk": "auto", "trunk_keep": "router"}
        kernels.use_quality_mode("speed")
        found = module.resolve(context, options, None)
        pricer = self.assemble()["projections"]
        self.assertEqual(found["stored_matrix_bytes"], 2 * (32 * 96 + 40 * 96))
        self.assertEqual(found["held_matrix_bytes"], pricer.held_bytes("bf16", 32, 96, 32 * 96 * 2)
                         + pricer.held_bytes("bf16", 40, 96, 40 * 96 * 2, "head"))
        self.assertLess(found["held_matrix_bytes"], found["stored_matrix_bytes"])   # FP8 in speed
        kernels.use_quality_mode("preserve")   # stored in preserve: held as stored
        found = module.resolve(context, options, None)
        self.assertEqual(found["held_matrix_bytes"], found["stored_matrix_bytes"])
        # No head tensor: the head is tied to the embeddings, which the model plugin loads through the slot as a
        # matrix beside the table: held as the head is, stored nowhere else (the table stays a lookup table)
        tied = folder / "tied"
        tied.mkdir()
        save_file({"model.layers.0.self_attn.q_proj.weight": bf16(32, 96),
                   "model.embed_tokens.weight": bf16(40, 96)}, str(tied / "model.safetensors"))
        (tied / "config.json").write_text('{"model_type": "llama", "hidden_size": 96}', encoding="utf-8")
        kernels.use_quality_mode("speed")
        found = module.resolve(SimpleNamespace(model=tied), options, None)
        self.assertEqual(found["stored_matrix_bytes"], 2 * 32 * 96)
        self.assertEqual(found["held_matrix_bytes"], pricer.held_bytes("bf16", 32, 96, 32 * 96 * 2)
                         + pricer.held_bytes("bf16", 40, 96, 40 * 96 * 2, "head"))

    def test_a_plan_prices_each_matrix_at_the_bytes_load_holds_it_in(self):
        # quality mode speed: trunk and head auto are fp8, trunk_keep's default leaves the router as stored
        from crit_plugins import kernels
        self.addCleanup(kernels.use_quality_mode, None)
        torch = __import__("torch")
        kernels.use_quality_mode("speed")
        p = self.assemble()["projections"]
        codec = p.held_bytes.__globals__["device_matrices"]
        w = torch.randn(40, 96, generator=torch.Generator().manual_seed(3), dtype=torch.float32).to(torch.bfloat16)

        def encoded(rows):   # the bytes of the codes and scales the encoder makes of `rows` zero-padded rows
            weight = torch.cat([w, w.new_zeros(rows - 40, 96)]) if rows > 40 else w[:rows]
            return codec.stored_bytes(*codec.reencode({"weight": weight}, "bf16", "fp8"))
        stored = lambda rows, columns=96: rows * columns * 2   # noqa: E731
        self.assertEqual(p.held_bytes("bf16", 32, 96, stored(32), "down"), encoded(32))
        self.assertEqual(p.held_bytes("bf16", 32, 96, stored(32)), encoded(32))   # a role the plan does not know
        self.assertEqual(p.held_bytes("bf16", 40, 96, stored(40), "o"), encoded(48))   # padded to 48 rows
        self.assertEqual(p.held_bytes("bf16", 32, 96, stored(32), "router"), stored(32))   # trunk_keep
        self.assertEqual(p.held_bytes("bf16", 32, 90, stored(32, 90), "gate"), stored(32, 90))   # 90 columns
        self.assertEqual(p.held_bytes("bf16", 32, 96, stored(32), "head"), encoded(32))
        self.assertEqual(p.held_bytes("bf16", 40, 96, stored(40), "head"), stored(40))   # the head is never padded
        self.assertEqual(p.held_bytes("nvfp4", 32, 96, 1234, "down"), 1234)   # stored quantized: as stored
        self.assertEqual(self.assemble(options={"trunk_keep": "down"})["projections"].held_bytes(
            "bf16", 32, 96, stored(32), "down"), stored(32))
        kernels.use_quality_mode("preserve")
        p = self.assemble()["projections"]
        self.assertEqual([p.held_bytes("bf16", 32, 96, stored(32), r) for r in ("down", "head")], [stored(32)] * 2)

    def test_the_formats_differ(self):
        w = matrix(2, 64, seed=4)
        decoded = {options.get("format", "int8"): self.assemble(options=options)["projections"]
                   for options, _ in self.FORMATS[:2] + self.FORMATS[3:4]}
        rows = {name: p.dense(p.prepare(w)) for name, p in decoded.items()}
        self.assertNotEqual(rows["int8"], rows["int4"])
        self.assertNotEqual(rows["int8"], rows["nvfp4"])
