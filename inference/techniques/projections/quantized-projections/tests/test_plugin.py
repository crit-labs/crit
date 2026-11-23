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

    def test_the_formats_differ(self):
        w = matrix(2, 64, seed=4)
        decoded = {options.get("format", "int8"): self.assemble(options=options)["projections"]
                   for options, _ in self.FORMATS[:2] + self.FORMATS[3:4]}
        rows = {name: p.dense(p.prepare(w)) for name, p in decoded.items()}
        self.assertNotEqual(rows["int8"], rows["int4"])
        self.assertNotEqual(rows["int8"], rows["nvfp4"])
